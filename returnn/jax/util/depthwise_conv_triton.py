"""
Tiled Triton depthwise 1-D convolution for the JAX backend.

A 1-D depthwise conv has no cuDNN kernel,
so XLA lowers it to a 2-D grouped implicit-GEMM plus layout transforms,
which costs far more than the arithmetic.
:func:`returnn.jax.frontend._backend._conv_depthwise_1d` avoids that
with a weighted sum of shifted copies, but that reads the input once per tap.
The kernels of :mod:`returnn.triton.depthwise_conv`, shared with the torch backend,
keep the taps of one row block in L1 instead.
The input is the contiguous 2-D (time, channel) packed layout, a single entry for the kernels,
and the filter has the (width, channel) layout.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from returnn.triton import depthwise_conv as kernels


def depthwise_conv1d_available(x, w) -> bool:
    """
    :param x: candidate input, must be the contiguous 2-D (time, channel) packed layout
    :param w: filter, (width, channel)
    :return: whether the Triton path applies, callers fall back to the shifted-sum otherwise
    """
    if kernels.triton is None:
        return False
    return x.ndim == 2 and w.ndim == 2 and x.shape[-1] == w.shape[-1]


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def depthwise_conv1d(
    x, w, pad_l: int, blocks=(kernels.BLOCK_R_CONV, kernels.BLOCK_C_CONV, kernels.BLOCK_R_DW, kernels.BLOCK_C_DW)
):
    """
    :param x: (time, channel), the contiguous packed layout
    :param w: (width, channel), one filter per channel
    :param pad_l: left padding, "same" uses (width - 1) // 2
    :param blocks: row and channel block of the conv kernels, then of the dw kernel, powers of two.
        The row loops of a window no longer than the filter use their own blocks.
    :return: (time, channel)
    """
    return _fwd(x, w, pad_l, blocks)[0]


def _fwd(x, w, pad_l, blocks):
    """
    :return: (out, residuals for the backward)
    """
    import jax_triton

    n_time, n_chan = x.shape
    width = w.shape[0]
    if n_time <= width:
        kernel, block_r, block_c = kernels.dw_fwd_rows, kernels.BLOCK_R_ROWS, kernels.BLOCK_C_ROWS
    else:
        kernel, block_r, block_c = kernels.dw_fwd, blocks[0], blocks[1]
    out = jax_triton.triton_call(
        x,
        w,
        w,
        n_time,
        n_time,
        n_time,
        n_chan,
        pad_l,
        1,
        n_chan,
        kernel=kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(kernels.cdiv(n_time, block_r), kernels.cdiv(n_chan, block_c)),
        HAS_BIAS=False,
        BLOCK_R=block_r,
        BLOCK_C=block_c,
        KW=width,
    )
    return out, (x, w)


def _bwd(pad_l, blocks, res, d_out):
    """
    :return: (dx, dw)
    """
    import jax_triton

    x, w = res
    n_time, n_chan = x.shape
    width = w.shape[0]
    if n_time <= width:
        kernel, block_r, block_c = kernels.dw_bwd_dx_rows, kernels.BLOCK_R_ROWS, kernels.BLOCK_C_ROWS
    else:
        kernel, block_r, block_c = kernels.dw_bwd_dx, blocks[0], blocks[1]
    dx = jax_triton.triton_call(
        d_out,
        w,
        n_time,
        n_time,
        n_time,
        n_chan,
        pad_l,
        1,
        n_chan,
        kernel=kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(kernels.cdiv(n_time, block_r), kernels.cdiv(n_chan, block_c)),
        BLOCK_R=block_r,
        BLOCK_C=block_c,
        KW=width,
    )
    rows_per_prog, n_splits, width_p, width_p2, num_warps = kernels.dw_launch(
        n_time, width, has_bias=False, block_r=blocks[2], block_c=blocks[3]
    )
    partial_sums = jax_triton.triton_call(
        x,
        d_out,
        n_time,
        n_time,
        n_time,
        n_chan,
        pad_l,
        rows_per_prog,
        kernel=kernels.dw_bwd_dw,
        out_shape=jax.ShapeDtypeStruct((n_splits, width_p, n_chan), jnp.float32),
        grid=(n_splits, kernels.cdiv(n_chan, blocks[3])),
        HAS_BIAS=False,
        BLOCK_R=blocks[2],
        BLOCK_C=blocks[3],
        KW=width,
        KW_P2=width_p2,
        KW_P=width_p,
        num_warps=num_warps,
    )
    return dx, partial_sums.sum(axis=0).astype(w.dtype)


depthwise_conv1d.defvjp(_fwd, _bwd)
