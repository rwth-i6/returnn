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

The tap loop loads and converts every input element once per tap, as the shifted tiles of the taps overlap.
The window kernels (``dw_fwd_window``, ``dw_bwd_dx_window``, ``dw_bwd_dw_window``) load every row of a program's
window once instead, as one (channel,) vector, and add it into the f32 accumulators of the rows it serves,
with the accumulators and the taps as tuples of row vectors, which Triton unrolls into registers.
A program owns a block of consecutive rows of one batch entry, so all masks and addresses are scalars per row.
They add the terms in the order the tap loops do, so the result is again the same bit for bit.

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


# swept on an H100 in bf16 with 1024 channels and 32 taps, the forward plus input gradient at (1, 20401) and
# (20, 1021) take 0.36 ms with 32x128 against 0.39 with 16x128 and 0.40 with 128x32
BLOCK_R_CONV, BLOCK_C_CONV = 32, 128
# the row loops, measured on an H100 for 2000 windows of 24 frames, 1024 channels, 32 taps,
# take 0.13 and 0.30 ms for a forward to 9 or 24 rows against 0.23 and 0.56 of the tap loop,
# and 0.13 and 0.30 against 0.40 and 0.52 for the input gradient
BLOCK_R_ROWS, BLOCK_C_ROWS = 16, 128
# swept on the same shapes plus (2000, 24) over 1 to 8 rows, 16 to 128 channels and 64 to 512 splits,
# 0.19 ms at the two long shapes and 0.39 at (2000, 24) against 0.24 and 0.47 with 2x32
BLOCK_R_DW, BLOCK_C_DW = 1, 128
DW_SPLITS = 128
# f32 accumulator entries per thread the dw kernel is sized for, it picks its warps from it
DW_ACC_PER_THREAD = 64
# window kernels, on an H100 at (1, 104000), 1024 channels, 32 taps: 0.92 ms against 0.95 with 16x256, 1.07 with 32x128
BLOCK_R_WINDOW, BLOCK_C_WINDOW = 32, 256


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

    # noinspection PyPep8Naming,PyUnresolvedReferences
    @triton.jit
    def _dw_window_add_row(acc, ws, x, D: tl.constexpr, BLOCK_R: tl.constexpr, KW: tl.constexpr):
        """:return: acc with the window row x at offset D added into the rows i it serves, with the tap D - i"""
        for i in tl.static_range(BLOCK_R):
            # the index arithmetic stays inline: assigned to a name, a static index becomes a tensor
            if D - i >= 0:
                if D - i < KW:
                    acc = acc[:i] + (acc[i] + x * ws[D - i],) + acc[i + 1 :]
        return acc

    # noinspection PyPep8Naming,PyUnresolvedReferences
    @triton.jit
    def _dw_window(
        Src,
        W,
        Bias,
        n_time_src,
        n_time_dst,
        n_chan,
        shift,
        stride_wc,
        stride_wk,
        Dst,
        HAS_BIAS: tl.constexpr,
        FLIP: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
        KW: tl.constexpr,
    ):
        """
        dst[b,t,c] = bias[c] + sum_d w'[c,d] * src[b,t+d+shift,c], with w' = w, or w with the taps reversed if FLIP.
        One program per (row block of one batch entry, channel block).
        The window rows are visited in increasing order, or in decreasing order if FLIP,
        which is the increasing order of the taps of w in both cases.
        """
        pid_r, pid_c = tl.program_id(0), tl.program_id(1)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = offs_c < n_chan
        n_blocks = (n_time_dst + BLOCK_R - 1) // BLOCK_R
        b = (pid_r // n_blocks).to(tl.int64)
        t0 = (pid_r % n_blocks) * BLOCK_R
        src_base = Src + b * n_time_src * n_chan + offs_c
        ws = ()
        for k in tl.static_range(KW):
            if FLIP:
                w_offs = offs_c * stride_wc + (KW - 1 - k) * stride_wk
            else:
                w_offs = offs_c * stride_wc + k * stride_wk
            ws = ws + (tl.load(W + w_offs, mask=c_mask, other=0.0).to(tl.float32),)
        acc = (tl.zeros((BLOCK_C,), dtype=tl.float32),) * BLOCK_R
        for j in tl.static_range(BLOCK_R + KW - 1):
            if FLIP:
                t_src = t0 + (BLOCK_R + KW - 2 - j) + shift
            else:
                t_src = t0 + j + shift
            m = c_mask & (t_src >= 0) & (t_src < n_time_src)
            x = tl.load(src_base + t_src.to(tl.int64) * n_chan, mask=m, other=0.0).to(tl.float32)
            if FLIP:
                acc = _dw_window_add_row(acc, ws, x, BLOCK_R + KW - 2 - j, BLOCK_R, KW)
            else:
                acc = _dw_window_add_row(acc, ws, x, j, BLOCK_R, KW)
        dst_base = Dst + (b * n_time_dst + t0) * n_chan + offs_c
        for i in tl.static_range(BLOCK_R):
            o = acc[i]
            if HAS_BIAS:
                o = o + tl.load(Bias + offs_c, mask=c_mask, other=0.0).to(tl.float32)
            tl.store(dst_base + i * n_chan, o.to(Dst.dtype.element_ty), mask=c_mask & (t0 + i < n_time_dst))

    # noinspection PyPep8Naming,PyUnresolvedReferences
    @triton.jit
    def dw_fwd_window(
        X,
        W,
        Bias,
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
        The forward of :func:`dw_fwd`, one program per (block of output rows of a batch entry, channel block),
        with the grid (batch * ceil(n_time_out / BLOCK_R), ceil(n_chan / BLOCK_C)).
        """
        _dw_window(
            X,
            W,
            Bias,
            n_time_in,
            n_time_out,
            n_chan,
            -pad_l,
            stride_wc,
            stride_wk,
            Out,
            HAS_BIAS,
            False,
            BLOCK_R,
            BLOCK_C,
            KW,
        )

    # noinspection PyPep8Naming,PyUnresolvedReferences
    @triton.jit
    def dw_bwd_dx_window(
        DO,
        W,
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
        The input gradient of :func:`dw_bwd_dx`, one program per (block of input rows of a batch entry, channel block),
        with the grid (batch * ceil(n_time_in / BLOCK_R), ceil(n_chan / BLOCK_C)).
        """
        _dw_window(
            DO,
            W,
            W,
            n_time_out,
            n_time_in,
            n_chan,
            pad_l - (KW - 1),
            stride_wc,
            stride_wk,
            DX,
            False,
            True,
            BLOCK_R,
            BLOCK_C,
            KW,
        )

    # noinspection PyPep8Naming,PyUnresolvedReferences
    @triton.jit
    def _dw_window_add_row_dw(acc, gs, x, J: tl.constexpr, BLOCK_R: tl.constexpr, KW: tl.constexpr):
        """:return: acc with the products of the window row x at offset J and the dout rows i added, at the tap J - i"""
        for i in tl.static_range(BLOCK_R):
            if J - i >= 0:
                if J - i < KW:
                    acc = acc[: J - i] + (acc[J - i] + gs[i] * x,) + acc[J - i + 1 :]
        return acc

    # noinspection PyPep8Naming,PyUnresolvedReferences
    @triton.jit
    def dw_bwd_dw_window(
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
        KW_P: tl.constexpr,
    ):
        """
        The partial sums P of :func:`dw_bwd_dw`, one program per (row split, channel block).
        It walks its split in blocks of up to BLOCK_R output rows, clipped at the entry boundaries,
        and adds the rows in increasing order, so P equals the one of :func:`dw_bwd_dw` with a row tile of 1.
        """
        pid_s, pid_c = tl.program_id(0), tl.program_id(1)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = offs_c < n_chan
        acc = (tl.zeros((BLOCK_C,), dtype=tl.float32),) * KW
        g_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
        r = pid_s * rows_per_prog
        row_end = tl.minimum(r + rows_per_prog, n_rows)
        while r < row_end:
            b = (r // n_time_out).to(tl.int64)
            t0 = r % n_time_out
            n_valid = tl.minimum(tl.minimum(n_time_out - t0, row_end - r), BLOCK_R)
            do_base = DO + r.to(tl.int64) * n_chan + offs_c
            gs = ()
            for i in tl.static_range(BLOCK_R):
                g = tl.load(do_base + i * n_chan, mask=c_mask & (i < n_valid), other=0.0).to(tl.float32)
                gs = gs + (g,)
                if HAS_BIAS:
                    g_acc += g
            x_base = X + (b * n_time_in + t0 - pad_l) * n_chan + offs_c
            for j in tl.static_range(BLOCK_R + KW - 1):
                t_in = t0 + j - pad_l
                m = c_mask & (t_in >= 0) & (t_in < n_time_in) & (j < n_valid + KW - 1)
                x = tl.load(x_base + j * n_chan, mask=m, other=0.0).to(tl.float32)
                acc = _dw_window_add_row_dw(acc, gs, x, j, BLOCK_R, KW)
            r += n_valid
        p_base = P + pid_s.to(tl.int64) * KW_P * n_chan + offs_c
        for k in tl.static_range(KW):
            tl.store(p_base + k * n_chan, acc[k], mask=c_mask)
        if HAS_BIAS:
            tl.store(p_base + KW * n_chan, g_acc, mask=c_mask)
