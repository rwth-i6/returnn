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

dw and db reduce over all rows: each program stores its tile's partial sums
into an f32 (row block, tap, channel) buffer, summed afterwards, so the result is deterministic.
Guard the import at the caller (needs Triton; the jit decorators run at import time).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


_BLOCK_R, _BLOCK_C = 32, 128
_BLOCK_R_DW, _BLOCK_C_DW = 64, 64


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
def _dw_bwd_dw(
    X,
    DO,
    P,
    n_rows,
    n_time_in,
    n_time_out,
    n_chan,
    pad_l,
    HAS_BIAS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    KW: tl.constexpr,
    KW_P: tl.constexpr,
):
    """P[rb,k,c] = sum_{rows of block rb} dout[b,t,c] * x[b,t+k-pad_l,c], plus the dout row sum as tap KW for the bias"""
    pid_r, pid_c = tl.program_id(0), tl.program_id(1)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    r_mask = offs_r < n_rows
    c_mask = offs_c < n_chan
    b = offs_r // n_time_out
    t = offs_r % n_time_out
    row_in = (b * n_time_in + t - pad_l).to(tl.int64)
    g = tl.load(
        DO + offs_r.to(tl.int64)[:, None] * n_chan + offs_c[None, :], mask=r_mask[:, None] & c_mask[None, :], other=0.0
    ).to(tl.float32)
    p_base = P + pid_r.to(tl.int64) * KW_P * n_chan
    for k in tl.static_range(KW):
        t_in = t + k - pad_l
        m = (t_in >= 0)[:, None] & (t_in < n_time_in)[:, None] & r_mask[:, None] & c_mask[None, :]
        x = tl.load(X + (row_in + k)[:, None] * n_chan + offs_c[None, :], mask=m, other=0.0).to(tl.float32)
        tl.store(p_base + k * n_chan + offs_c, tl.sum(g * x, axis=0), mask=c_mask)
    if HAS_BIAS:
        tl.store(p_base + KW * n_chan + offs_c, tl.sum(g, axis=0), mask=c_mask)


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
        n_batch, n_time_in, n_chan = x.shape
        width = w.shape[1]
        out = x.new_empty((n_batch, n_time_out, n_chan))
        n_rows = n_batch * n_time_out
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
        d_out = d_out.contiguous()
        n_batch, n_time_in, n_chan = x.shape
        n_time_out = d_out.shape[1]
        width = w.shape[1]
        has_bias = ctx.bias_dtype is not None
        block_r, block_c, block_r_dw, block_c_dw = ctx.blocks
        dx = dw = db = None
        if ctx.needs_input_grad[0]:
            dx = torch.empty_like(x)
            n_rows = n_batch * n_time_in
            grid = (triton.cdiv(n_rows, block_r), triton.cdiv(n_chan, block_c))
            _dw_bwd_dx[grid](
                d_out,
                w,
                dx,
                n_rows,
                n_time_in,
                n_time_out,
                n_chan,
                ctx.pad_l,
                BLOCK_R=block_r,
                BLOCK_C=block_c,
                KW=width,
                num_warps=4,
            )
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            n_rows = n_batch * n_time_out
            n_row_blocks = triton.cdiv(n_rows, block_r_dw)
            partial = torch.empty((n_row_blocks, width + int(has_bias), n_chan), dtype=torch.float32, device=x.device)
            grid = (n_row_blocks, triton.cdiv(n_chan, block_c_dw))
            _dw_bwd_dw[grid](
                x,
                d_out,
                partial,
                n_rows,
                n_time_in,
                n_time_out,
                n_chan,
                ctx.pad_l,
                HAS_BIAS=has_bias,
                BLOCK_R=block_r_dw,
                BLOCK_C=block_c_dw,
                KW=width,
                KW_P=width + int(has_bias),
                num_warps=4,
            )
            summed = partial.sum(dim=0)
            if ctx.needs_input_grad[1]:
                dw = summed[:width].t().contiguous().to(w.dtype)
            if has_bias and ctx.needs_input_grad[2]:
                db = summed[width].to(ctx.bias_dtype)
        return dx, dw, db, None, None, None


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
    :param blocks: row and channel block of the conv kernels, then of the dw kernel, powers of two
    :return: (batch, time_out, channel), in the dtype of x, accumulated in f32
    """
    assert x.ndim == 3 and w.ndim == 2 and x.shape[2] == w.shape[0]
    assert bias is None or bias.shape == (w.shape[0],)
    operands = (x, w) if bias is None else (x, w, bias)
    assert all(t.dtype in (torch.float16, torch.bfloat16, torch.float32) for t in operands), [t.dtype for t in operands]
    assert all(v > 0 and v & (v - 1) == 0 for v in blocks), blocks
    bias = bias.contiguous() if bias is not None else None
    return _DepthwiseConv1d.apply(x.contiguous(), w.contiguous(), bias, pad_l, n_time_out, tuple(blocks))
