"""
Tiled Triton depthwise 1-D convolution for the JAX backend.

A 1-D depthwise conv has no cuDNN kernel,
so XLA lowers it to a 2-D grouped implicit-GEMM plus layout transforms,
which costs far more than the arithmetic.
:func:`returnn.jax.frontend._backend._conv_depthwise_1d` avoids that
with a weighted sum of shifted copies, but that reads the input once per tap.
This kernel keeps the taps of one time block in L1 instead.

    out[t,c] = sum_k w[k,c] * x[t+k-pad_l, c]
    dx[t,c]  = sum_k w[k,c] * dout[t-k+pad_l, c]
    dw[k,c]  = sum_t dout[t,c] * x[t+k-pad_l, c]

dw reduces over the whole time axis,
so each program accumulates its own time-block partial
into the small (width, C) output with atomics.
"""

from __future__ import annotations
from typing import Tuple

from functools import partial

import jax
import jax.numpy as jnp

try:
    import jax_triton
    import triton
    import triton.language as tl
except ImportError:  # optional dependency, same as the rel-pos attention kernels
    jax_triton = triton = None

# Small time blocks keep the halo in L1.
# Chosen by sweep at the Conformer shape.
_BLOCK_T, _BLOCK_C = 16, 128


def depthwise_conv1d_available(x, w, block_t: int = _BLOCK_T) -> bool:
    """
    :param x: candidate input, must be the contiguous 2-D (time, channel) packed layout
    :param w: filter, (width, channel)
    :param block_t: time block, only powers of two compile
    :return: whether the Triton path applies; callers fall back to the shifted-sum otherwise
    """
    if jax_triton is None or triton is None:
        return False
    if x.ndim != 2 or w.ndim != 2 or x.shape[-1] != w.shape[-1]:
        return False
    return block_t & (block_t - 1) == 0


if triton is not None:

    @triton.jit
    def _dw_fwd(X, W, n_time, n_chan, pad_l, Out, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr, KW: tl.constexpr):
        """out[t,c] = sum_k w[k,c] * x[t+k-pad_l, c]"""
        pid_t, pid_c = tl.program_id(0), tl.program_id(1)
        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = offs_c < n_chan
        acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
        for k in tl.static_range(KW):
            t_in = offs_t + k - pad_l
            m = (t_in >= 0)[:, None] & (t_in < n_time)[:, None] & c_mask[None, :]
            x = tl.load(X + t_in[:, None] * n_chan + offs_c[None, :], mask=m, other=0.0).to(tl.float32)
            wk = tl.load(W + k * n_chan + offs_c, mask=c_mask, other=0.0).to(tl.float32)
            acc += x * wk[None, :]
        o_mask = (offs_t[:, None] < n_time) & c_mask[None, :]
        tl.store(Out + offs_t[:, None] * n_chan + offs_c[None, :], acc.to(Out.dtype.element_ty), mask=o_mask)

    @triton.jit
    def _dw_bwd_dx(DO, W, n_time, n_chan, pad_l, DX, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr, KW: tl.constexpr):
        """dx[t,c] = sum_k w[k,c] * dout[t-k+pad_l, c], the correlation with the flipped filter"""
        pid_t, pid_c = tl.program_id(0), tl.program_id(1)
        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = offs_c < n_chan
        acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
        for k in tl.static_range(KW):
            t_in = offs_t - k + pad_l
            m = (t_in >= 0)[:, None] & (t_in < n_time)[:, None] & c_mask[None, :]
            g = tl.load(DO + t_in[:, None] * n_chan + offs_c[None, :], mask=m, other=0.0).to(tl.float32)
            wk = tl.load(W + k * n_chan + offs_c, mask=c_mask, other=0.0).to(tl.float32)
            acc += g * wk[None, :]
        o_mask = (offs_t[:, None] < n_time) & c_mask[None, :]
        tl.store(DX + offs_t[:, None] * n_chan + offs_c[None, :], acc.to(DX.dtype.element_ty), mask=o_mask)

    @triton.jit
    def _dw_bwd_dw(X, DO, DW, n_time, n_chan, pad_l, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr, KW: tl.constexpr):
        """dw[k,c] = sum_t dout[t,c] * x[t+k-pad_l, c], accumulated across time blocks"""
        pid_t, pid_c = tl.program_id(0), tl.program_id(1)
        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = offs_c < n_chan
        t_mask = offs_t < n_time
        g = tl.load(
            DO + offs_t[:, None] * n_chan + offs_c[None, :], mask=t_mask[:, None] & c_mask[None, :], other=0.0
        ).to(tl.float32)
        for k in tl.static_range(KW):
            t_in = offs_t + k - pad_l
            m = (t_in >= 0)[:, None] & (t_in < n_time)[:, None] & c_mask[None, :]
            x = tl.load(X + t_in[:, None] * n_chan + offs_c[None, :], mask=m, other=0.0).to(tl.float32)
            tl.atomic_add(DW + k * n_chan + offs_c, tl.sum(g * x, axis=0), mask=c_mask)


def _grid(n_time: int, n_chan: int, block_t: int, block_c: int) -> Tuple[int, int]:
    """
    :param n_time:
    :param n_chan:
    :param block_t:
    :param block_c:
    :return: launch grid over (time, channel) tiles
    """
    return triton.cdiv(n_time, block_t), triton.cdiv(n_chan, block_c)


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def depthwise_conv1d(x, w, pad_l: int, block_t: int = _BLOCK_T, block_c: int = _BLOCK_C):
    """
    :param x: (time, channel), the contiguous packed layout
    :param w: (width, channel), one filter per channel
    :param pad_l: left padding; "same" uses (width - 1) // 2
    :param block_t:
    :param block_c:
    :return: (time, channel)
    """
    return _fwd(x, w, pad_l, block_t, block_c)[0]


def _fwd(x, w, pad_l, block_t, block_c):
    """
    :return: (out, residuals for the backward)
    """
    n_time, n_chan = x.shape
    # the kernels taking out_shape list their output last:
    # jax_triton appends outputs after the inputs, so a mid-signature output binds to a scalar
    out = jax_triton.triton_call(
        x,
        w,
        n_time,
        n_chan,
        pad_l,
        kernel=_dw_fwd,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=_grid(n_time, n_chan, block_t, block_c),
        BLOCK_T=block_t,
        BLOCK_C=block_c,
        KW=w.shape[0],
        num_warps=4,
    )
    return out, (x, w)


def _bwd(pad_l, block_t, block_c, res, d_out):
    """
    :return: (dx, dw)
    """
    x, w = res
    n_time, n_chan = x.shape
    width = w.shape[0]
    dx = jax_triton.triton_call(
        d_out,
        w,
        n_time,
        n_chan,
        pad_l,
        kernel=_dw_bwd_dx,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=_grid(n_time, n_chan, block_t, block_c),
        BLOCK_T=block_t,
        BLOCK_C=block_c,
        KW=width,
        num_warps=4,
    )
    # atomics add into their target, so it must start zeroed and be f32
    dw_ref = jax.new_ref(jnp.zeros((width, n_chan), dtype=jnp.float32))
    jax_triton.triton_call(
        x,
        d_out,
        dw_ref,
        n_time,
        n_chan,
        pad_l,
        kernel=_dw_bwd_dw,
        out_type=(),
        grid=_grid(n_time, n_chan, block_t, block_c),
        BLOCK_T=block_t,
        BLOCK_C=block_c,
        KW=width,
        num_warps=4,
    )
    return dx, dw_ref[...].astype(w.dtype)


depthwise_conv1d.defvjp(_fwd, _bwd)
