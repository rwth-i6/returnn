"""
Tiled Triton depthwise 1-D convolution (stride 1, no dilation), shared by the torch backend
(:mod:`returnn.torch.util.depthwise_conv_triton`) and the JAX backend (:mod:`returnn.jax.util.depthwise_conv_triton`).

The kernels tile (row, channel) with the channels innermost,
so the input is used in the (batch, time, channel) layout it already has.
The rows are the flattened (batch, time) pairs, so a tile spans batch entries,
with the taps masked at the entry boundaries. A single entry (batch 1) is the 2-D (time, channel) packed case.

    out[b,t,c] = bias[c] + sum_k w[c,k] * x[b,t+k-pad_l,c]
    dx[b,t,c]  = sum_k w[c,k] * dout[b,t-k+pad_l,c]
    dw[c,k]    = sum_{b,t} dout[b,t,c] * x[b,t+k-pad_l,c]
    db[c]      = sum_{b,t} dout[b,t,c]

dw and db reduce over all rows. Each program walks a contiguous range of rows,
keeps f32 sums per (tap, row, channel) of its tile in registers and reduces over the rows once at the end,
into an f32 (row split, tap, channel) buffer of a fixed number of splits, summed afterwards,
so the result is deterministic and the scratch does not grow with the rows.
Reducing once per program instead of once per tile and tap is what makes this kernel cheap,
since per tile the reductions and the (row block, tap, channel) stores cost more than the products.

A window no longer than the filter (the chunked Conformer convolves 24 frames with 32 taps)
leaves most taps of every row outside it, so there the forward and the input gradient loop over
the rows of the window instead of the taps. They add the terms in the order the tap loop does,
so the result is the same bit for bit.

The filter comes with its two strides, so each backend keeps its own layout,
(channel, width) in torch and (width, channel) in JAX.
Every kernel takes its output last, since ``jax_triton.triton_call`` binds the outputs after the inputs.
The Triton import is guarded, so this module imports without Triton and then defines no kernels.
"""

from __future__ import annotations

from typing import Tuple

try:
    import triton
    import triton.language as tl
except ImportError:  # optional dependency
    triton = tl = None


BLOCK_R, BLOCK_C = 32, 128
# the row loops, measured on an H100 for 2000 windows of 24 frames, 1024 channels, 32 taps,
# take 0.13 and 0.30 ms for a forward to 9 or 24 rows against 0.23 and 0.56 of the tap loop,
# and 0.13 and 0.30 against 0.40 and 0.52 for the input gradient
BLOCK_R_ROWS, BLOCK_C_ROWS = 16, 128
# measured on an H100 for (chunks, 24, 1024) with 32 taps, 0.43 ms against 1.19 ms of the per-tile reduction
BLOCK_R_DW, BLOCK_C_DW = 2, 32
DW_SPLITS = 128
# f32 accumulator entries per thread the dw kernel is sized for, it picks its warps from it
DW_ACC_PER_THREAD = 64


def cdiv(a: int, b: int) -> int:
    """:return: ceil(a / b)"""
    return -(-a // b)


def dw_launch(n_rows: int, width: int, *, has_bias: bool, block_r: int, block_c: int) -> Tuple[int, int, int, int, int]:
    """
    Launch geometry of :func:`dw_bwd_dw`: a fixed number of splits, each a multiple of the row tile,
    so the scratch does not grow with the rows.

    :param n_rows: output rows, batch times output frames
    :param width: filter taps
    :param has_bias: whether the bias gradient is summed as an extra tap
    :param block_r: row tile of the dw kernel
    :param block_c: channel tile of the dw kernel
    :return: (rows per program, splits, taps in the scratch, taps rounded up to a power of two, warps)
    """
    rows_per_prog = max(cdiv(cdiv(n_rows, DW_SPLITS), block_r), 1) * block_r
    n_splits = max(cdiv(n_rows, rows_per_prog), 1)
    width_p2 = 1 << (width - 1).bit_length()
    num_warps = max(1, min(8, width_p2 * block_r * block_c // (32 * DW_ACC_PER_THREAD)))
    return rows_per_prog, n_splits, width + int(has_bias), width_p2, num_warps


if triton is not None:
    # noinspection PyPep8Naming,PyUnresolvedReferences
    @triton.jit
    def dw_fwd(
        X,
        W,
        Bias,
        n_rows,
        n_time_in,
        n_time_out,
        n_chan,
        pad_l,
        stride_wc,
        stride_wk,
        Out,
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
            wk = tl.load(W + offs_c * stride_wc + k * stride_wk, mask=c_mask, other=0.0).to(tl.float32)
            acc += x * wk[None, :]
        if HAS_BIAS:
            acc += tl.load(Bias + offs_c, mask=c_mask, other=0.0).to(tl.float32)[None, :]
        o_mask = r_mask[:, None] & c_mask[None, :]
        o_offs = offs_r.to(tl.int64)[:, None] * n_chan + offs_c[None, :]
        tl.store(Out + o_offs, acc.to(Out.dtype.element_ty), mask=o_mask)

    # noinspection PyPep8Naming,PyUnresolvedReferences
    @triton.jit
    def dw_bwd_dx(
        DO,
        W,
        n_rows,
        n_time_in,
        n_time_out,
        n_chan,
        pad_l,
        stride_wc,
        stride_wk,
        DX,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
        KW: tl.constexpr,
    ):
        """
        dx[b,t,c] = sum_k w[c,k] * dout[b,t-k+pad_l,c], the correlation with the flipped filter, rows over the input
        """
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
            wk = tl.load(W + offs_c * stride_wc + k * stride_wk, mask=c_mask, other=0.0).to(tl.float32)
            acc += g * wk[None, :]
        o_mask = r_mask[:, None] & c_mask[None, :]
        o_offs = offs_r.to(tl.int64)[:, None] * n_chan + offs_c[None, :]
        tl.store(DX + o_offs, acc.to(DX.dtype.element_ty), mask=o_mask)

    # noinspection PyPep8Naming,PyUnresolvedReferences
    @triton.jit
    def dw_fwd_rows(
        X,
        W,
        Bias,
        n_rows,
        n_time_in,
        n_time_out,
        n_chan,
        pad_l,
        stride_wc,
        stride_wk,
        Out,
        HAS_BIAS: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
        KW: tl.constexpr,
    ):
        """
        out[b,t,c] = bias[c] + sum_s w[c,s-t+pad_l] * x[b,s,c], the forward of :func:`dw_fwd` for a window no longer
        than the filter. It loops over the input rows s, which visits the taps in increasing order like the tap loop.
        The filter loads are contiguous with a (tap, channel) layout.
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
            wk = tl.load(W + k[:, None] * stride_wk + offs_c[None, :] * stride_wc, mask=m, other=0.0).to(tl.float32)
            acc += x * wk
        if HAS_BIAS:
            acc += tl.load(Bias + offs_c, mask=c_mask, other=0.0).to(tl.float32)[None, :]
        o_mask = r_mask[:, None] & c_mask[None, :]
        o_offs = offs_r.to(tl.int64)[:, None] * n_chan + offs_c[None, :]
        tl.store(Out + o_offs, acc.to(Out.dtype.element_ty), mask=o_mask)

    # noinspection PyPep8Naming,PyUnresolvedReferences
    @triton.jit
    def dw_bwd_dx_rows(
        DO,
        W,
        n_rows,
        n_time_in,
        n_time_out,
        n_chan,
        pad_l,
        stride_wc,
        stride_wk,
        DX,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
        KW: tl.constexpr,
    ):
        """
        dx[b,s,c] = sum_t w[c,s-t+pad_l] * dout[b,t,c], the input gradient of :func:`dw_bwd_dx` for an output no longer
        than the filter. It loops over the output rows t from the last one, which visits the taps in increasing order
        like the tap loop. The filter loads are contiguous with a (tap, channel) layout.
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
            wk = tl.load(W + k[:, None] * stride_wk + offs_c[None, :] * stride_wc, mask=m, other=0.0).to(tl.float32)
            acc += g * wk
        o_mask = r_mask[:, None] & c_mask[None, :]
        o_offs = offs_r.to(tl.int64)[:, None] * n_chan + offs_c[None, :]
        tl.store(DX + o_offs, acc.to(DX.dtype.element_ty), mask=o_mask)

    # noinspection PyPep8Naming,PyUnresolvedReferences
    @triton.jit
    def dw_bwd_dw(
        X,
        DO,
        n_rows,
        n_time_in,
        n_time_out,
        n_chan,
        pad_l,
        rows_per_prog,
        P,
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
            # per (tap, row) the input frame of every tap, masked at the entry boundaries
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
