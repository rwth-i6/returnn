"""
Tiled Triton depthwise 1-D convolution for the torch backend (stride 1, no dilation).

torch routes a depthwise conv1d on CUDA to its native depthwise-2d kernels,
whose weight gradient parallelises over the (channel, tap) outputs
and reduces the whole batch x time extent inside each of them,
which for the Conformer conv block costs more than everything else in the layer.
These kernels tile (row, channel) with the channels innermost,
so the input is used in the (batch, time, channel) layout it already has
(no transpose to (batch, channel, time) and back).
The rows are the flattened (batch, time) pairs, so a tile spans batch entries,
with the taps masked at the entry boundaries.

    out[b,t,c] = bias[c] + sum_k w[c,k] * x[b,t+k-pad_l,c]
    dx[b,t,c]  = sum_k w[c,k] * dout[b,t-k+pad_l,c]
    dw[c,k]    = sum_{b,t} dout[b,t,c] * x[b,t+k-pad_l,c]
    db[c]      = sum_{b,t} dout[b,t,c]

dw and db reduce over all rows: each program walks a contiguous range of rows,
keeps f32 sums per (tap, row, channel) of its tile in registers and reduces over the rows once at the end,
into an f32 (row split, tap, channel) buffer of a fixed number of splits, summed afterwards,
so the result is deterministic and the scratch does not grow with the rows.
Reducing once per program instead of once per tile and tap is what makes this kernel cheap:
per tile, the reductions and the (row block, tap, channel) stores cost more than the products.

A window no longer than the filter (the chunked Conformer convolves 24 frames with 32 taps)
leaves most taps of every row outside it, so there the forward and the input gradient loop over
the rows of the window instead of the taps, with the filter read as (tap, channel) to keep the
loads contiguous. They add the terms in the order the tap loop does, so the result is the same bit for bit.
Guard the import at the caller (needs Triton; the jit decorators run at import time).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


_BLOCK_R, _BLOCK_C = 32, 128
# the row loops, measured on an H100 for 2000 windows of 24 frames, 1024 channels, 32 taps, forward to 9 or
# 24 rows: 0.13 and 0.30 ms against 0.23 and 0.56 of the tap loop, the input gradient 0.13 and 0.30 against 0.40 and 0.52
_BLOCK_R_ROWS, _BLOCK_C_ROWS = 16, 128
# measured on an H100 for (chunks, 24, 1024) with 32 taps: 0.43 ms against 1.19 ms of the per-tile reduction
_BLOCK_R_DW, _BLOCK_C_DW = 2, 32
_DW_SPLITS = 128
# f32 accumulator entries per thread the dw kernel is sized for, it picks its warps from it
_DW_ACC_PER_THREAD = 64


def is_available() -> bool:
    """:return: whether the kernel can run (needs a CUDA device)"""
    return torch.cuda.is_available()


# noinspection PyPep8Naming,PyUnresolvedReferences
@triton.jit
def _dw_fwd(
    X,
    W,
    Bias,
    Out,
    n_rows,
    n_time_in,
    n_time_out,
    n_chan,
    pad_l,
    HAS_BIAS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    KW: tl.constexpr,
):
    """out[b,t,c] = bias[c] + sum_k w[c,k] * x[b,t+k-pad_l,c], one program per (row block, channel block)"""
    pid_r, pid_c = tl.program_id(0), tl.program_id(1)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    r_mask = offs_r < n_rows
    c_mask = offs_c < n_chan
    b = offs_r // n_time_out
    t = offs_r % n_time_out
    row_in = (b * n_time_in + t - pad_l).to(tl.int64)
    acc = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)
    for k in tl.static_range(KW):
        t_in = t + k - pad_l
        m = (t_in >= 0)[:, None] & (t_in < n_time_in)[:, None] & r_mask[:, None] & c_mask[None, :]
        x = tl.load(X + (row_in + k)[:, None] * n_chan + offs_c[None, :], mask=m, other=0.0).to(tl.float32)
        wk = tl.load(W + offs_c * KW + k, mask=c_mask, other=0.0).to(tl.float32)
        acc += x * wk[None, :]
    if HAS_BIAS:
        acc += tl.load(Bias + offs_c, mask=c_mask, other=0.0).to(tl.float32)[None, :]
    o_mask = r_mask[:, None] & c_mask[None, :]
    tl.store(Out + offs_r.to(tl.int64)[:, None] * n_chan + offs_c[None, :], acc.to(Out.dtype.element_ty), mask=o_mask)


# noinspection PyPep8Naming,PyUnresolvedReferences
@triton.jit
def _dw_bwd_dx(
    DO,
    W,
    DX,
    n_rows,
    n_time_in,
    n_time_out,
    n_chan,
    pad_l,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    KW: tl.constexpr,
):
    """dx[b,t,c] = sum_k w[c,k] * dout[b,t-k+pad_l,c], the correlation with the flipped filter, rows over the input"""
    pid_r, pid_c = tl.program_id(0), tl.program_id(1)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    r_mask = offs_r < n_rows
    c_mask = offs_c < n_chan
    b = offs_r // n_time_in
    t = offs_r % n_time_in
    row_out = (b * n_time_out + t + pad_l).to(tl.int64)
    acc = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)
    for k in tl.static_range(KW):
        t_out = t - k + pad_l
        m = (t_out >= 0)[:, None] & (t_out < n_time_out)[:, None] & r_mask[:, None] & c_mask[None, :]
        g = tl.load(DO + (row_out - k)[:, None] * n_chan + offs_c[None, :], mask=m, other=0.0).to(tl.float32)
        wk = tl.load(W + offs_c * KW + k, mask=c_mask, other=0.0).to(tl.float32)
        acc += g * wk[None, :]
    o_mask = r_mask[:, None] & c_mask[None, :]
    tl.store(DX + offs_r.to(tl.int64)[:, None] * n_chan + offs_c[None, :], acc.to(DX.dtype.element_ty), mask=o_mask)


# noinspection PyPep8Naming,PyUnresolvedReferences
@triton.jit
def _dw_fwd_rows(
    X,
    WT,
    Bias,
    Out,
    n_rows,
    n_time_in,
    n_time_out,
    n_chan,
    pad_l,
    HAS_BIAS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    KW: tl.constexpr,
):
    """
    out[b,t,c] = bias[c] + sum_s w[c,s-t+pad_l] * x[b,s,c], the forward of :func:`_dw_fwd` for a window no longer
    than the filter: loops over the input rows s, which visits the taps in increasing order like the tap loop.
    WT is the filter as (tap, channel).
    """
    pid_r, pid_c = tl.program_id(0), tl.program_id(1)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    r_mask = offs_r < n_rows
    c_mask = offs_c < n_chan
    b = offs_r // n_time_out
    t = offs_r % n_time_out
    row_in0 = (b * n_time_in).to(tl.int64)
    acc = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)
    for s in range(0, n_time_in):
        k = s - t + pad_l
        m = (k >= 0)[:, None] & (k < KW)[:, None] & r_mask[:, None] & c_mask[None, :]
        x = tl.load(X + (row_in0 + s)[:, None] * n_chan + offs_c[None, :], mask=m, other=0.0).to(tl.float32)
        wk = tl.load(WT + k[:, None] * n_chan + offs_c[None, :], mask=m, other=0.0).to(tl.float32)
        acc += x * wk
    if HAS_BIAS:
        acc += tl.load(Bias + offs_c, mask=c_mask, other=0.0).to(tl.float32)[None, :]
    o_mask = r_mask[:, None] & c_mask[None, :]
    tl.store(Out + offs_r.to(tl.int64)[:, None] * n_chan + offs_c[None, :], acc.to(Out.dtype.element_ty), mask=o_mask)


# noinspection PyPep8Naming,PyUnresolvedReferences
@triton.jit
def _dw_bwd_dx_rows(
    DO,
    WT,
    DX,
    n_rows,
    n_time_in,
    n_time_out,
    n_chan,
    pad_l,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    KW: tl.constexpr,
):
    """
    dx[b,s,c] = sum_t w[c,s-t+pad_l] * dout[b,t,c], the input gradient of :func:`_dw_bwd_dx` for an output no longer
    than the filter: loops over the output rows t from the last one, which visits the taps in increasing order like
    the tap loop. WT is the filter as (tap, channel).
    """
    pid_r, pid_c = tl.program_id(0), tl.program_id(1)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    r_mask = offs_r < n_rows
    c_mask = offs_c < n_chan
    b = offs_r // n_time_in
    s = offs_r % n_time_in
    row_out0 = (b * n_time_out).to(tl.int64)
    acc = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)
    for i in range(0, n_time_out):
        t = n_time_out - 1 - i
        k = s - t + pad_l
        m = (k >= 0)[:, None] & (k < KW)[:, None] & r_mask[:, None] & c_mask[None, :]
        g = tl.load(DO + (row_out0 + t)[:, None] * n_chan + offs_c[None, :], mask=m, other=0.0).to(tl.float32)
        wk = tl.load(WT + k[:, None] * n_chan + offs_c[None, :], mask=m, other=0.0).to(tl.float32)
        acc += g * wk
    o_mask = r_mask[:, None] & c_mask[None, :]
    tl.store(DX + offs_r.to(tl.int64)[:, None] * n_chan + offs_c[None, :], acc.to(DX.dtype.element_ty), mask=o_mask)


# noinspection PyPep8Naming,PyUnresolvedReferences
@triton.jit
def _dw_bwd_dw(
    X,
    DO,
    P,
    n_rows,
    n_time_in,
    n_time_out,
    n_chan,
    pad_l,
    rows_per_prog,
    HAS_BIAS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    KW: tl.constexpr,
    KW_P2: tl.constexpr,
    KW_P: tl.constexpr,
):
    """
    P[s,k,c] = sum_{rows of split s} dout[b,t,c] * x[b,t+k-pad_l,c], plus the dout row sum as tap KW for the bias.
    One program per (row split, channel block), looping over the split in tiles of BLOCK_R rows.
    """
    pid_s, pid_c = tl.program_id(0), tl.program_id(1)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < n_chan
    taps = tl.arange(0, KW_P2)
    tap_mask = taps < KW
    acc = tl.zeros((KW_P2, BLOCK_R, BLOCK_C), dtype=tl.float32)
    g_acc = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)
    row_start = pid_s * rows_per_prog
    row_end = tl.minimum(row_start + rows_per_prog, n_rows)
    for r0 in range(row_start, row_end, BLOCK_R):
        offs_r = r0 + tl.arange(0, BLOCK_R)
        r_mask = offs_r < row_end
        b = offs_r // n_time_out
        t = offs_r % n_time_out
        g = tl.load(
            DO + offs_r.to(tl.int64)[:, None] * n_chan + offs_c[None, :],
            mask=r_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        # (tap, row): the input frame of every tap, masked at the entry boundaries
        t_in = t[None, :] + taps[:, None] - pad_l
        valid = (t_in >= 0) & (t_in < n_time_in) & tap_mask[:, None] & r_mask[None, :]
        row_in = (b * n_time_in)[None, :] + t_in
        x = tl.load(
            X + row_in.to(tl.int64)[:, :, None] * n_chan + offs_c[None, None, :],
            mask=valid[:, :, None] & c_mask[None, None, :],
            other=0.0,
        ).to(tl.float32)
        acc += g[None, :, :] * x
        if HAS_BIAS:
            g_acc += g
    p_base = P + pid_s.to(tl.int64) * KW_P * n_chan
    tl.store(
        p_base + taps[:, None] * n_chan + offs_c[None, :],
        tl.sum(acc, axis=1),
        mask=tap_mask[:, None] & c_mask[None, :],
    )
    if HAS_BIAS:
        tl.store(p_base + KW * n_chan + offs_c, tl.sum(g_acc, axis=0), mask=c_mask)


def _launch_fwd(x, w, bias, pad_l: int, n_time_out: int, blocks) -> torch.Tensor:
    """:return: the conv output (batch, time_out, channel), the arguments as in :class:`_DepthwiseConv1d`"""
    n_batch, n_time_in, n_chan = x.shape
    width = w.shape[1]
    out = x.new_empty((n_batch, n_time_out, n_chan))
    n_rows = n_batch * n_time_out
    if n_time_in <= width:
        w_t = w.t().contiguous()
        grid = (triton.cdiv(n_rows, _BLOCK_R_ROWS), triton.cdiv(n_chan, _BLOCK_C_ROWS))
        _dw_fwd_rows[grid](
            x,
            w_t,
            bias if bias is not None else w_t,
            out,
            n_rows,
            n_time_in,
            n_time_out,
            n_chan,
            pad_l,
            HAS_BIAS=bias is not None,
            BLOCK_R=_BLOCK_R_ROWS,
            BLOCK_C=_BLOCK_C_ROWS,
            KW=width,
            num_warps=4,
        )
        return out
    block_r, block_c = blocks[0], blocks[1]
    grid = (triton.cdiv(n_rows, block_r), triton.cdiv(n_chan, block_c))
    _dw_fwd[grid](
        x,
        w,
        bias if bias is not None else w,
        out,
        n_rows,
        n_time_in,
        n_time_out,
        n_chan,
        pad_l,
        HAS_BIAS=bias is not None,
        BLOCK_R=block_r,
        BLOCK_C=block_c,
        KW=width,
        num_warps=4,
    )
    return out


def _launch_bwd(x, w, d_out, *, has_bias: bool, pad_l: int, blocks, need_dx: bool, need_dw_db: bool):
    """
    :return: (dx or None, the f32 sums (width + has_bias, channel) the filter and bias gradients are read from or None)
    """
    n_batch, n_time_in, n_chan = x.shape
    n_time_out = d_out.shape[1]
    width = w.shape[1]
    block_r, block_c, block_r_dw, block_c_dw = blocks
    dx = summed = None
    if need_dx:
        dx = torch.empty_like(x)
        n_rows = n_batch * n_time_in
        if n_time_out <= width:
            grid = (triton.cdiv(n_rows, _BLOCK_R_ROWS), triton.cdiv(n_chan, _BLOCK_C_ROWS))
            _dw_bwd_dx_rows[grid](
                d_out,
                w.t().contiguous(),
                dx,
                n_rows,
                n_time_in,
                n_time_out,
                n_chan,
                pad_l,
                BLOCK_R=_BLOCK_R_ROWS,
                BLOCK_C=_BLOCK_C_ROWS,
                KW=width,
                num_warps=4,
            )
        else:
            grid = (triton.cdiv(n_rows, block_r), triton.cdiv(n_chan, block_c))
            _dw_bwd_dx[grid](
                d_out,
                w,
                dx,
                n_rows,
                n_time_in,
                n_time_out,
                n_chan,
                pad_l,
                BLOCK_R=block_r,
                BLOCK_C=block_c,
                KW=width,
                num_warps=4,
            )
    if need_dw_db:
        n_rows = n_batch * n_time_out
        # a fixed number of splits, each a multiple of the row tile, so the scratch is independent of the rows
        rows_per_prog = max(triton.cdiv(triton.cdiv(n_rows, _DW_SPLITS), block_r_dw), 1) * block_r_dw
        n_splits = max(triton.cdiv(n_rows, rows_per_prog), 1)
        width_p2 = triton.next_power_of_2(width)
        num_warps = max(1, min(8, width_p2 * block_r_dw * block_c_dw // (32 * _DW_ACC_PER_THREAD)))
        partial = torch.empty((n_splits, width + int(has_bias), n_chan), dtype=torch.float32, device=x.device)
        grid = (n_splits, triton.cdiv(n_chan, block_c_dw))
        _dw_bwd_dw[grid](
            x,
            d_out,
            partial,
            n_rows,
            n_time_in,
            n_time_out,
            n_chan,
            pad_l,
            rows_per_prog,
            HAS_BIAS=has_bias,
            BLOCK_R=block_r_dw,
            BLOCK_C=block_c_dw,
            KW=width,
            KW_P2=width_p2,
            KW_P=width + int(has_bias),
            num_warps=num_warps,
        )
        summed = partial.sum(dim=0)
    return dx, summed


class _DepthwiseConv1d(torch.autograd.Function):
    """The conv with its gradients through the three kernels."""

    @staticmethod
    def forward(ctx, x, w, bias, pad_l, n_time_out, blocks):
        """
        :param x: (batch, time_in, channel), contiguous
        :param w: (channel, width), contiguous
        :param bias: (channel,) or None
        :param pad_l: frames of zero padding before the first input frame
        :param n_time_out: output frames per batch entry
        :param blocks: (row block, channel block) for the conv kernels and (row block, channel block) for the dw kernel
        :return: (batch, time_out, channel)
        """
        out = _launch_fwd(x, w, bias, pad_l, n_time_out, blocks)
        ctx.save_for_backward(x, w)
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.pad_l = pad_l
        ctx.blocks = blocks
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, d_out):
        """
        :param d_out: (batch, time_out, channel)
        :return: the gradients for x, w and bias, None for the other arguments
        """
        x, w = ctx.saved_tensors
        width = w.shape[1]
        has_bias = ctx.bias_dtype is not None
        dx, summed = _launch_bwd(
            x,
            w,
            d_out.contiguous(),
            has_bias=has_bias,
            pad_l=ctx.pad_l,
            blocks=ctx.blocks,
            need_dx=ctx.needs_input_grad[0],
            need_dw_db=ctx.needs_input_grad[1] or ctx.needs_input_grad[2],
        )
        dw = db = None
        if summed is not None:
            if ctx.needs_input_grad[1]:
                dw = summed[:width].t().contiguous().to(w.dtype)
            if has_bias and ctx.needs_input_grad[2]:
                db = summed[width].to(ctx.bias_dtype)
        return dx, dw, db, None, None, None


_HAVE_LIB_OPS = False
if hasattr(torch.library, "custom_op"):  # torch >= 2.4
    # Opaque ops with fake implementations and a registered backward, like in rel_pos_att_triton:
    # AOT tracing (the compiled step of torch_cuda_graph, no Dynamo) runs on fake tensors,
    # which the Triton launch of the autograd.Function above cannot take.
    # Only a traced call goes through them, the eager path stays the autograd.Function.

    @torch.library.custom_op("returnn::depthwise_conv1d_fwd", mutates_args=())
    def _lib_fwd(
        x: torch.Tensor,
        w: torch.Tensor,
        bias: Optional[torch.Tensor],
        pad_l: int,
        n_time_out: int,
        block_r: int,
        block_c: int,
        block_r_dw: int,
        block_c_dw: int,
    ) -> torch.Tensor:
        # contiguous: Inductor feeds custom ops in whatever layout it likes, the fake promises a contiguous output
        bias = bias.contiguous() if bias is not None else None
        blocks = (block_r, block_c, block_r_dw, block_c_dw)
        return _launch_fwd(x.contiguous(), w.contiguous(), bias, pad_l, n_time_out, blocks)

    @_lib_fwd.register_fake
    def _lib_fwd_fake(x, w, bias, pad_l, n_time_out, block_r, block_c, block_r_dw, block_c_dw):
        del w, bias, pad_l, block_r, block_c, block_r_dw, block_c_dw
        return x.new_empty((x.shape[0], n_time_out, x.shape[2]))

    @torch.library.custom_op("returnn::depthwise_conv1d_bwd", mutates_args=())
    def _lib_bwd(
        x: torch.Tensor,
        w: torch.Tensor,
        d_out: torch.Tensor,
        has_bias: bool,
        pad_l: int,
        block_r: int,
        block_c: int,
        block_r_dw: int,
        block_c_dw: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dx, summed = _launch_bwd(
            x.contiguous(),
            w.contiguous(),
            d_out.contiguous(),
            has_bias=has_bias,
            pad_l=pad_l,
            blocks=(block_r, block_c, block_r_dw, block_c_dw),
            need_dx=True,
            need_dw_db=True,
        )
        return dx, summed

    @_lib_bwd.register_fake
    def _lib_bwd_fake(x, w, d_out, has_bias, pad_l, block_r, block_c, block_r_dw, block_c_dw):
        del d_out, pad_l, block_r, block_c, block_r_dw, block_c_dw
        return x.new_empty(tuple(x.shape)), x.new_empty((w.shape[1] + int(has_bias), w.shape[0]), dtype=torch.float32)

    def _lib_setup_context(ctx, inputs, output):
        x, w, bias, pad_l, _, block_r, block_c, block_r_dw, block_c_dw = inputs
        ctx.save_for_backward(x, w)
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.pad_l = pad_l
        ctx.blocks = (block_r, block_c, block_r_dw, block_c_dw)

    def _lib_backward(ctx, d_out):
        x, w = ctx.saved_tensors
        width = w.shape[1]
        has_bias = ctx.bias_dtype is not None
        dx, summed = torch.ops.returnn.depthwise_conv1d_bwd(x, w, d_out, has_bias, ctx.pad_l, *ctx.blocks)
        dw = summed[:width].t().contiguous().to(w.dtype)
        db = summed[width].to(ctx.bias_dtype) if has_bias else None
        return dx, dw, db, None, None, None, None, None, None

    torch.library.register_autograd("returnn::depthwise_conv1d_fwd", _lib_backward, setup_context=_lib_setup_context)

    _HAVE_LIB_OPS = True


def traceable() -> bool:
    """:return: whether a traced step (fake tensors) can take the conv, through the opaque ops"""
    return _HAVE_LIB_OPS


def depthwise_conv1d(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    pad_l: int,
    n_time_out: int,
    blocks: Tuple[int, int, int, int] = (_BLOCK_R, _BLOCK_C, _BLOCK_R_DW, _BLOCK_C_DW),
) -> torch.Tensor:
    """
    :param x: (batch, time_in, channel)
    :param w: (channel, width), one filter per channel
    :param bias: (channel,) or None
    :param pad_l: frames of zero padding before the first input frame, "same" uses (width - 1) // 2
    :param n_time_out: output frames per batch entry, time_in + pad_l + pad_r - width + 1
    :param blocks: row and channel block of the conv kernels, then of the dw kernel, powers of two.
        The row loops of a window no longer than the filter use their own blocks.
    :return: (batch, time_out, channel), in the dtype of x, accumulated in f32
    """
    assert x.ndim == 3 and w.ndim == 2 and x.shape[2] == w.shape[0]
    assert bias is None or bias.shape == (w.shape[0],)
    operands = (x, w) if bias is None else (x, w, bias)
    assert all(t.dtype in (torch.float16, torch.bfloat16, torch.float32) for t in operands), [t.dtype for t in operands]
    assert all(v > 0 and v & (v - 1) == 0 for v in blocks), blocks
    bias = bias.contiguous() if bias is not None else None
    if _HAVE_LIB_OPS and type(x) not in (torch.Tensor, torch.nn.Parameter):
        # a traced call (fake or functional tensors)
        return torch.ops.returnn.depthwise_conv1d_fwd(x.contiguous(), w.contiguous(), bias, pad_l, n_time_out, *blocks)
    return _DepthwiseConv1d.apply(x.contiguous(), w.contiguous(), bias, pad_l, n_time_out, tuple(blocks))
