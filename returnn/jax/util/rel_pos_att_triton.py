"""
Packed rel-pos self-attention for the JAX backend, on the Triton kernels the PyTorch backend uses.

Why this and not the traceable alternatives:
a per-seq scan needs a STATIC window per sequence,
so it costs ``batch_bound * T_cap^2`` -- exactly padded attention.
A dense block-diagonal mask costs ``(sum len)^2``, ~14x more than padded.
Only a varlen kernel computes the ``sum len^2`` that packing is worth
(~15x below padded at this batching).

The kernels in :mod:`returnn.torch.util.rel_pos_att_triton` are plain ``@triton.jit``
(pointers, strides, constexprs), so they are framework-neutral;
only their wrapper is torch (autograd.Function, torch tensors).
Here they get a JAX wrapper:
:func:`jax_triton.triton_call` for the launch,
:func:`jax.custom_vjp` for the gradient.

Two mechanical differences to the torch launcher:
- jax_triton appends outputs AFTER the inputs,
  while the kernels take ``Out``/``Lse`` in the middle,
  so thin ``@triton.jit`` wrappers below re-order them.
- strides are passed explicitly;
  JAX arrays here are contiguous, so they follow from the shape.
"""

from __future__ import annotations
from typing import Tuple
from functools import partial

import math

import jax
import jax.numpy as jnp

# The kernels, not the torch wrapper. That module imports torch at module level; torch is present in
# every env that has this backend, but the kernels themselves touch nothing torch-specific.
from returnn.torch.util.rel_pos_att_triton import (
    _rel_pos_fwd_kernel,
    _rel_pos_bwd_kernel_delta,
    _rel_pos_bwd_kernel_dkv,
    _rel_pos_bwd_kernel_dq,
)

import triton


def rel_pos_att_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    bd: jax.Array,
    seq_starts: jax.Array,
    seq_lens: jax.Array,
    max_len: int,
    *,
    dropout_p: float = 0.0,
    seed: int = 0,
    scale: float = None,
) -> Tuple[jax.Array, jax.Array]:
    """
    :param q: [total, heads, dim], packed
    :param k: [total, heads, dim], same packing as q
    :param v: [total, heads, dim], same packing as q
    :param bd: [total, heads, 2*max_len-1], the rel-pos bias per query row
    :param seq_starts: [batch], first row of each sequence in the packed buffer
    :param seq_lens: [batch]
    :param max_len: static bound on the per-seq length, so the grid and the bias width are static
    :param dropout_p: attention dropout
    :param seed: dropout seed
    :param scale: qk scale, 1/sqrt(dim) if not given
    :return: (out [total, heads, dim], lse [total, heads])
    """
    import jax_triton as jt

    total, n_heads, d = q.shape
    r = bd.shape[-1]
    assert r == 2 * max_len - 1, f"bias width {r} != 2*{max_len}-1"
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    # a kernel launch needs all operands on ONE device,
    # and starts/lens come from the dataset side, so they can still be on the host.
    # A CUDA-only process hides that; it fails once a CPU device exists.
    # Under tracing the compiler places everything, so there is nothing to move.
    if not isinstance(q, jax.core.Tracer):
        dev = next(iter(q.devices()))
        seq_starts, seq_lens = (
            a if isinstance(a, jax.core.Tracer) or next(iter(a.devices())) == dev else jax.device_put(a, dev)
            for a in (seq_starts, seq_lens)
        )
    n_batch = seq_starts.shape[0]
    block_m, block_n = 64, 64
    # Out and Lse are in-out Refs, in their own argument positions, so the shared kernel is called
    # unchanged. jax_triton's out_shape appends outputs AFTER the inputs, which this kernel's
    # signature does not match, and a wrapper kernel calling it does not compile.
    # zeros, not empty, for the same reason as the torch launcher: the kernel writes only valid
    # rows, so junk rows would carry garbage into the residual adds.
    out_ref = jax.new_ref(jnp.zeros_like(q))
    lse_ref = jax.new_ref(jnp.zeros((total, n_heads), dtype=jnp.float32))
    jt.triton_call(
        q,
        k,
        v,
        bd,
        out_ref,
        lse_ref,
        seq_starts,
        seq_lens,
        jnp.asarray(seed, dtype=jnp.int32),
        float(dropout_p),
        float(scale),
        n_heads * d,  # stride_qt
        d,  # stride_qh
        n_heads * r,  # stride_bt
        r,  # stride_bh
        n_heads * d,  # stride_ot
        d,  # stride_oh
        kernel=_rel_pos_fwd_kernel,
        out_type=(),
        grid=(triton.cdiv(max_len, block_m), n_batch * n_heads),
        H=n_heads,
        D=d,
        R=r,
        center=max_len - 1,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        ENABLE_DROPOUT=dropout_p > 0.0,
        IEEE=q.dtype == jnp.float32,
    )
    return out_ref[...], lse_ref[...]


def rel_pos_att_bwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    bd: jax.Array,
    seq_starts: jax.Array,
    seq_lens: jax.Array,
    max_len: int,
    lse: jax.Array,
    d_out: jax.Array,
    *,
    dropout_p: float = 0.0,
    seed: int = 0,
    scale: float = None,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    :param q: as in :func:`rel_pos_att_fwd`
    :param k:
    :param v:
    :param bd:
    :param seq_starts:
    :param seq_lens:
    :param max_len:
    :param lse: from the forward
    :param d_out: gradient w.r.t. the forward's output
    :param dropout_p:
    :param seed: the forward's seed, so the same dropout mask is regenerated
    :param scale:
    :return: (dq, dk, dv, dbd), all f32

    Three kernels, as in the torch launcher: delta (recomputed in-kernel, in f32 -- the bf16 delta
    biased the pos_bias grads), then dk/dv, then dq/dbd.
    """
    import jax_triton as jt

    # same co-location requirement as the forward (see rel_pos_att_fwd)
    if not isinstance(q, jax.core.Tracer):
        dev = next(iter(q.devices()))
        seq_starts, seq_lens = (
            a if isinstance(a, jax.core.Tracer) or next(iter(a.devices())) == dev else jax.device_put(a, dev)
            for a in (seq_starts, seq_lens)
        )
    total, n_heads, d = q.shape
    r = bd.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    n_batch = seq_starts.shape[0]
    block_m, block_n = (64, 64) if d <= 64 else (32, 32)
    seed_arr = jnp.asarray(seed, dtype=jnp.int32)
    strides = (n_heads * d, d, n_heads * r, r)
    consts = dict(
        H=n_heads,
        D=d,
        R=r,
        center=max_len - 1,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        ENABLE_DROPOUT=dropout_p > 0.0,
        IEEE=q.dtype == jnp.float32,
    )

    delta_ref = jax.new_ref(jnp.zeros((total, n_heads), dtype=jnp.float32))
    dq_ref = jax.new_ref(jnp.zeros(q.shape, dtype=jnp.float32))
    dk_ref = jax.new_ref(jnp.zeros(k.shape, dtype=jnp.float32))
    dv_ref = jax.new_ref(jnp.zeros(v.shape, dtype=jnp.float32))
    dbd_ref = jax.new_ref(jnp.zeros(bd.shape, dtype=jnp.float32))

    jt.triton_call(
        q,
        k,
        v,
        bd,
        d_out,
        lse,
        delta_ref,
        seq_starts,
        seq_lens,
        seed_arr,
        float(dropout_p),
        float(scale),
        *strides,
        kernel=_rel_pos_bwd_kernel_delta,
        out_type=(),
        grid=(triton.cdiv(max_len, block_m), n_batch * n_heads),
        **consts,
    )
    jt.triton_call(
        q,
        k,
        v,
        bd,
        d_out,
        lse,
        delta_ref,
        dk_ref,
        dv_ref,
        seq_starts,
        seq_lens,
        seed_arr,
        float(dropout_p),
        float(scale),
        *strides,
        kernel=_rel_pos_bwd_kernel_dkv,
        out_type=(),
        grid=(triton.cdiv(max_len, block_n), n_batch * n_heads),
        **consts,
    )
    jt.triton_call(
        q,
        k,
        v,
        bd,
        d_out,
        lse,
        delta_ref,
        dq_ref,
        dbd_ref,
        seq_starts,
        seq_lens,
        seed_arr,
        float(dropout_p),
        float(scale),
        *strides,
        kernel=_rel_pos_bwd_kernel_dq,
        out_type=(),
        grid=(triton.cdiv(max_len, block_m), n_batch * n_heads),
        **consts,
    )
    return dq_ref[...], dk_ref[...], dv_ref[...], dbd_ref[...]


@partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9))
def rel_pos_att_varlen(q, k, v, bd, seq_starts, seq_lens, max_len, dropout_p, seed, scale):
    """
    :param q: [total, heads, dim], packed
    :param k: [total, heads, dim]
    :param v: [total, heads, dim]
    :param bd: [total, heads, 2*max_len-1]
    :param seq_starts: [batch]
    :param seq_lens: [batch]
    :param max_len: static per-seq bound
    :param dropout_p:
    :param seed:
    :param scale:
    :return: out [total, heads, dim]

    Differentiable packed rel-pos self-attention. The counterpart of the torch backend's
    ``_RelPosAttVarlen`` autograd.Function, on the same kernels.
    """
    out, _ = rel_pos_att_fwd(q, k, v, bd, seq_starts, seq_lens, max_len, dropout_p=dropout_p, seed=seed, scale=scale)
    return out


def _rel_pos_att_varlen_fwd(q, k, v, bd, seq_starts, seq_lens, max_len, dropout_p, seed, scale):
    """
    :return: (out, residuals for the backward)
    """
    out, lse = rel_pos_att_fwd(q, k, v, bd, seq_starts, seq_lens, max_len, dropout_p=dropout_p, seed=seed, scale=scale)
    return out, (q, k, v, bd, seq_starts, seq_lens, lse)


def _rel_pos_att_varlen_bwd(max_len, dropout_p, seed, scale, res, d_out):
    """
    :return: gradients for the differentiable arguments
    """
    q, k, v, bd, seq_starts, seq_lens, lse = res
    dq, dk, dv, dbd = rel_pos_att_bwd(
        q, k, v, bd, seq_starts, seq_lens, max_len, lse, d_out, dropout_p=dropout_p, seed=seed, scale=scale
    )
    return (
        dq.astype(q.dtype),
        dk.astype(k.dtype),
        dv.astype(v.dtype),
        dbd.astype(bd.dtype),
        None,  # seq_starts
        None,  # seq_lens
    )


rel_pos_att_varlen.defvjp(_rel_pos_att_varlen_fwd, _rel_pos_att_varlen_bwd)
