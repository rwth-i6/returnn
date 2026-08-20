"""
Packed (varlen) scaled dot-product attention for the JAX backend, in Triton.

Covers what :mod:`returnn.jax.util.rel_pos_att_triton` does not:
causal self-attention,
and cross-attention, where queries and keys/values carry DIFFERENT packings
(decoder queries over a packed encoder).
That is the whole decoder side of an AED model.
Without it the packed backend has to unpack -> attend -> repack,
throwing away the packing it just built.

Why a kernel at all, when JAX ships attention:
``jax.nn.dot_product_attention`` takes seq lens only in the PADDED (B,T,N,H) layout,
and the Pallas GPU kernels are padded-layout too,
so either would first have to unpack.
A dense block-diagonal mask over the packed axis costs ``total_q * total_k``, far above padded.
Only a varlen kernel computes the ``sum(len_q*len_k)`` that packing is worth.

The torch backend gets this from the aten flash varlen kernels,
so there is no kernel to share (unlike the rel-pos ones).
These are modelled on it closely,
including the separate f32 delta pass -- see :func:`_sdpa_bwd_kernel_delta`.

Dropout uses Triton's philox,
so its MASKS do not match the aten flash kernels', only the distribution does.
Cross-framework checks therefore run at ``dropout_p = 0``.

ENV REQUIREMENT: the ptxas XLA finds first must be new enough for the PTX Triton emits
(3.7 emits 8.7, needing CUDA >= 12.8),
else a compiled step dies with a late, opaque ``ptxas ... Unsupported .version 8.7``.
An older cluster module easily shadows a newer toolkit;
the nvidia-cuda-nvcc wheel in a jax env ships a suitable one.
Env setup, deliberately not fixed from inside RETURNN:
a library rewriting PATH at import would silently override a deliberate toolchain choice.
"""

from __future__ import annotations
from typing import Optional, Tuple
from functools import partial

import math

import jax
import jax.numpy as jnp

import triton
import triton.language as tl


# noinspection PyPep8Naming,PyUnresolvedReferences
@triton.jit
def _sdpa_fwd_kernel(
    Q,
    K,
    V,
    Out,
    Lse,
    CuQ,
    LenQ,
    CuK,
    LenK,
    Seed,
    dropout_p,
    scale,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_ot,
    stride_oh,
    H: tl.constexpr,
    D: tl.constexpr,
    rand_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IEEE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    q_start = tl.load(CuQ + b)
    q_len = tl.load(LenQ + b)
    k_start = tl.load(CuK + b)
    k_len = tl.load(LenK + b)
    if pid_m * BLOCK_M >= q_len:
        return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # local q rows
    offs_d = tl.arange(0, D)
    q_rows = q_start + offs_m
    q_mask = offs_m < q_len
    q = tl.load(Q + q_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=q_mask[:, None], other=0.0)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    # causal: columns beyond this q block cannot contribute, so the loop stops early
    # (the wrapper requires equal q/k lengths there, so the diagonal is well defined)
    hi = tl.minimum(k_len, (pid_m + 1) * BLOCK_M) if IS_CAUSAL else k_len
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)  # local kv cols
        n_mask = offs_n < k_len
        k_rows = k_start + offs_n
        k = tl.load(K + k_rows[:, None] * stride_kt + h * stride_kh + offs_d[None, :], mask=n_mask[:, None], other=0.0)
        if IEEE:
            s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale  # (M, N)
        else:
            s = tl.dot(q, tl.trans(k)) * scale  # (M, N)
        valid = q_mask[:, None] & n_mask[None, :]
        if IS_CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        # a fully masked block leaves m_new = -inf; exp(-inf - -inf) is nan, so pin those rows
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - m_safe[:, None])
        alpha = tl.exp(tl.where(m_i == float("-inf"), float("-inf"), m_i) - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        if ENABLE_DROPOUT:
            # philox offsets in int32, unique per (global q row, local col), head in the seed
            # -- see the rel-pos kernels for why 64-bit offsets are avoided here
            offs = q_rows[:, None] * rand_stride + offs_n[None, :]
            keep = tl.rand(tl.load(Seed) + h * 1000003, offs) >= dropout_p
            p_use = tl.where(keep, p / (1.0 - dropout_p), 0.0)
        else:
            p_use = p
        v_blk = tl.load(
            V + k_rows[:, None] * stride_kt + h * stride_kh + offs_d[None, :], mask=n_mask[:, None], other=0.0
        )
        if IEEE:
            acc += tl.dot(p_use.to(v_blk.dtype), v_blk, input_precision="ieee")
        else:
            acc += tl.dot(p_use.to(v_blk.dtype), v_blk)
        m_i = m_new
    # a q row with no valid column at all (empty kv seq) has l_i = 0: keep its output at 0
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    tl.store(
        Out + q_rows[:, None] * stride_ot + h * stride_oh + offs_d[None, :],
        acc / l_safe[:, None],
        mask=q_mask[:, None],
    )
    tl.store(Lse + q_rows * H + h, tl.where(l_i == 0.0, float("-inf"), m_i + tl.log(l_safe)), mask=q_mask)


# noinspection PyPep8Naming,PyUnresolvedReferences
@triton.jit
def _sdpa_bwd_kernel_delta(
    Q,
    K,
    V,
    DO,
    Lse,
    Delta,
    CuQ,
    LenQ,
    CuK,
    LenK,
    Seed,
    dropout_p,
    scale,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    H: tl.constexpr,
    D: tl.constexpr,
    rand_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IEEE: tl.constexpr,
):
    # delta_i = sum_j p_ij * dp_ij,
    # recomputed in f32 from the same p/dp the other bwd kernels use.
    # NOT the flash shortcut delta = rowsum(out * do):
    # that identity holds only for an exact out = sum_j p_ij v_j,
    # while the stored out is bf16, accumulated from a bf16-rounded p.
    # ds = p * (dp - delta) sits on the cancellation dp ~= delta
    # at the dominant entries of sharp attention rows,
    # so the rounding mismatch takes over exactly there, as a per-row BIAS rather than noise.
    # That is what collapsed training around ep 7 through the rel-pos bias grads.
    # Cost: one extra attention-shaped pass, fwd and memory unchanged.
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    q_start = tl.load(CuQ + b)
    q_len = tl.load(LenQ + b)
    k_start = tl.load(CuK + b)
    k_len = tl.load(LenK + b)
    if pid_m * BLOCK_M >= q_len:
        return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    m_mask = offs_m < q_len
    q_rows = q_start + offs_m
    q = tl.load(Q + q_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=m_mask[:, None], other=0.0)
    do = tl.load(DO + q_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=m_mask[:, None], other=0.0)
    lse = tl.load(Lse + q_rows * H + h, mask=m_mask, other=0.0)
    delta = tl.zeros([BLOCK_M], dtype=tl.float32)
    hi = tl.minimum(k_len, (pid_m + 1) * BLOCK_M) if IS_CAUSAL else k_len
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < k_len
        k_rows = k_start + offs_n
        k = tl.load(K + k_rows[:, None] * stride_kt + h * stride_kh + offs_d[None, :], mask=n_mask[:, None], other=0.0)
        v = tl.load(V + k_rows[:, None] * stride_kt + h * stride_kh + offs_d[None, :], mask=n_mask[:, None], other=0.0)
        if IEEE:
            s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
            dp = tl.dot(do, tl.trans(v), input_precision="ieee")
        else:
            s = tl.dot(q, tl.trans(k)) * scale
            dp = tl.dot(do, tl.trans(v))
        valid = m_mask[:, None] & n_mask[None, :]
        if IS_CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        p = tl.exp(tl.where(valid, s, float("-inf")) - lse[:, None])
        if ENABLE_DROPOUT:
            offs = q_rows[:, None] * rand_stride + offs_n[None, :]
            keep = tl.rand(tl.load(Seed) + h * 1000003, offs) >= dropout_p
            dp = tl.where(keep, dp / (1.0 - dropout_p), 0.0)
        delta += tl.sum(tl.where(valid, p * dp, 0.0), 1)
    tl.store(Delta + q_rows * H + h, delta, mask=m_mask)


# noinspection PyPep8Naming,PyUnresolvedReferences
@triton.jit
def _sdpa_bwd_kernel_dkv(
    Q,
    K,
    V,
    DO,
    Lse,
    Delta,
    DK,
    DV,
    CuQ,
    LenQ,
    CuK,
    LenK,
    Seed,
    dropout_p,
    scale,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    H: tl.constexpr,
    D: tl.constexpr,
    rand_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IEEE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    q_start = tl.load(CuQ + b)
    q_len = tl.load(LenQ + b)
    k_start = tl.load(CuK + b)
    k_len = tl.load(LenK + b)
    if pid_n * BLOCK_N >= k_len:
        return
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    n_mask = offs_n < k_len
    k_rows = k_start + offs_n
    k = tl.load(K + k_rows[:, None] * stride_kt + h * stride_kh + offs_d[None, :], mask=n_mask[:, None], other=0.0)
    v = tl.load(V + k_rows[:, None] * stride_kt + h * stride_kh + offs_d[None, :], mask=n_mask[:, None], other=0.0)
    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    # causal: only q rows at or below this k block see these columns
    lo = pid_n * BLOCK_N if IS_CAUSAL else 0
    for start_m in range(lo, q_len, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        m_mask = offs_m < q_len
        q_rows = q_start + offs_m
        q = tl.load(Q + q_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=m_mask[:, None], other=0.0)
        do = tl.load(
            DO + q_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=m_mask[:, None], other=0.0
        )
        lse = tl.load(Lse + q_rows * H + h, mask=m_mask, other=0.0)
        delta = tl.load(Delta + q_rows * H + h, mask=m_mask, other=0.0)
        if IEEE:
            s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        else:
            s = tl.dot(q, tl.trans(k)) * scale
        valid = m_mask[:, None] & n_mask[None, :]
        if IS_CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        p = tl.exp(tl.where(valid, s, float("-inf")) - lse[:, None])
        if ENABLE_DROPOUT:
            offs = q_rows[:, None] * rand_stride + offs_n[None, :]
            keep = tl.rand(tl.load(Seed) + h * 1000003, offs) >= dropout_p
            p_use = tl.where(keep, p / (1.0 - dropout_p), 0.0)
        else:
            p_use = p
        if IEEE:
            dv += tl.dot(tl.trans(p_use.to(do.dtype)), do, input_precision="ieee")
            dp = tl.dot(do, tl.trans(v), input_precision="ieee")
        else:
            dv += tl.dot(tl.trans(p_use.to(do.dtype)), do)
            dp = tl.dot(do, tl.trans(v))
        if ENABLE_DROPOUT:
            # the same keep as above (both blocks share the ENABLE_DROPOUT constexpr)
            # noinspection PyUnboundLocalVariable
            dp = tl.where(keep, dp / (1.0 - dropout_p), 0.0)
        ds = tl.where(valid, p * (dp - delta[:, None]), 0.0)
        if IEEE:
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q, input_precision="ieee") * scale
        else:
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q) * scale
    tl.store(DK + k_rows[:, None] * stride_kt + h * stride_kh + offs_d[None, :], dk, mask=n_mask[:, None])
    tl.store(DV + k_rows[:, None] * stride_kt + h * stride_kh + offs_d[None, :], dv, mask=n_mask[:, None])


# noinspection PyPep8Naming,PyUnresolvedReferences
@triton.jit
def _sdpa_bwd_kernel_dq(
    Q,
    K,
    V,
    DO,
    Lse,
    Delta,
    DQ,
    CuQ,
    LenQ,
    CuK,
    LenK,
    Seed,
    dropout_p,
    scale,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    H: tl.constexpr,
    D: tl.constexpr,
    rand_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IEEE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    q_start = tl.load(CuQ + b)
    q_len = tl.load(LenQ + b)
    k_start = tl.load(CuK + b)
    k_len = tl.load(LenK + b)
    if pid_m * BLOCK_M >= q_len:
        return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    m_mask = offs_m < q_len
    q_rows = q_start + offs_m
    q = tl.load(Q + q_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=m_mask[:, None], other=0.0)
    do = tl.load(DO + q_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=m_mask[:, None], other=0.0)
    lse = tl.load(Lse + q_rows * H + h, mask=m_mask, other=0.0)
    delta = tl.load(Delta + q_rows * H + h, mask=m_mask, other=0.0)
    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    hi = tl.minimum(k_len, (pid_m + 1) * BLOCK_M) if IS_CAUSAL else k_len
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < k_len
        k_rows = k_start + offs_n
        k = tl.load(K + k_rows[:, None] * stride_kt + h * stride_kh + offs_d[None, :], mask=n_mask[:, None], other=0.0)
        v = tl.load(V + k_rows[:, None] * stride_kt + h * stride_kh + offs_d[None, :], mask=n_mask[:, None], other=0.0)
        if IEEE:
            s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
            dp = tl.dot(do, tl.trans(v), input_precision="ieee")
        else:
            s = tl.dot(q, tl.trans(k)) * scale
            dp = tl.dot(do, tl.trans(v))
        valid = m_mask[:, None] & n_mask[None, :]
        if IS_CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        p = tl.exp(tl.where(valid, s, float("-inf")) - lse[:, None])
        if ENABLE_DROPOUT:
            offs = q_rows[:, None] * rand_stride + offs_n[None, :]
            keep = tl.rand(tl.load(Seed) + h * 1000003, offs) >= dropout_p
            dp = tl.where(keep, dp / (1.0 - dropout_p), 0.0)
        ds = tl.where(valid, p * (dp - delta[:, None]), 0.0)
        if IEEE:
            dq += tl.dot(ds.to(k.dtype), k, input_precision="ieee") * scale
        else:
            dq += tl.dot(ds.to(k.dtype), k) * scale
    tl.store(DQ + q_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], dq, mask=m_mask[:, None])


def _on_device_of(ref, *arrays):
    """
    :param ref: array whose device the others must share
    :param arrays: index arrays (starts/lens), often built on the host
    :return: the arrays, moved to ref's device where needed

    A kernel launch needs all operands on ONE device.
    Starts/lens come from the dataset side and can stay on the host,
    which a CUDA-only process hides (nothing else can be on CPU there),
    but which fails as soon as a CPU device exists at all.
    Under tracing there is no device to query and the compiler places everything.
    """
    if isinstance(ref, jax.core.Tracer):
        return arrays
    dev = next(iter(ref.devices()))
    return tuple(
        a if isinstance(a, jax.core.Tracer) or next(iter(a.devices())) == dev else jax.device_put(a, dev)
        for a in arrays
    )


def _strides(q, k):
    """
    :param q: [total_q, heads, dim]
    :param k: [total_k, heads, dim]
    :return: (stride_qt, stride_qh, stride_kt, stride_kh); JAX arrays here are contiguous
    """
    _, n_heads, d = q.shape
    del k  # same heads/dim, so the same strides follow
    return n_heads * d, d, n_heads * d, d


def sdpa_varlen_fwd(
    q, k, v, cu_q, len_q, cu_k, len_k, max_q, max_k, *, is_causal=False, dropout_p=0.0, seed=0, scale=None
) -> Tuple[jax.Array, jax.Array]:
    """
    forward, see :func:`sdpa_varlen`. Returns (out, lse).
    """
    import jax_triton as jt

    total_q, n_heads, d = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    cu_q, len_q, cu_k, len_k = _on_device_of(q, cu_q, len_q, cu_k, len_k)
    n_batch = cu_q.shape[0]
    block_m, block_n = 64, 64
    ieee = q.dtype == jnp.float32
    # zeros, not empty: the kernel writes only valid rows (per-block early exit), so gap rows
    # would keep arbitrary garbage that residual adds and weight grads then spread
    out_ref = jax.new_ref(jnp.zeros_like(q))
    lse_ref = jax.new_ref(jnp.zeros((total_q, n_heads), dtype=jnp.float32))
    stride_qt, stride_qh, stride_kt, stride_kh = _strides(q, k)
    jt.triton_call(
        q,
        k,
        v,
        out_ref,
        lse_ref,
        cu_q,
        len_q,
        cu_k,
        len_k,
        jnp.asarray(seed, dtype=jnp.int32),
        float(dropout_p),
        float(scale),
        stride_qt,
        stride_qh,
        stride_kt,
        stride_kh,
        stride_qt,  # stride_ot
        stride_qh,  # stride_oh
        kernel=_sdpa_fwd_kernel,
        out_type=(),
        grid=(triton.cdiv(max_q, block_m), n_batch * n_heads),
        H=n_heads,
        D=d,
        rand_stride=max_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        IS_CAUSAL=is_causal,
        ENABLE_DROPOUT=dropout_p > 0.0,
        IEEE=ieee,
    )
    return out_ref[...], lse_ref[...]


def sdpa_varlen_bwd(
    do, q, k, v, out, lse, cu_q, len_q, cu_k, len_k, max_q, max_k, *, is_causal, dropout_p, seed, scale
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    backward, see :func:`sdpa_varlen`. Returns (dq, dk, dv).
    """
    import jax_triton as jt

    total_q, n_heads, d = q.shape
    del out  # delta is recomputed, see _sdpa_bwd_kernel_delta
    n_batch = cu_q.shape[0]
    block_m, block_n = 64, 64
    ieee = q.dtype == jnp.float32
    stride_qt, stride_qh, stride_kt, stride_kh = _strides(q, k)
    common = (cu_q, len_q, cu_k, len_k, jnp.asarray(seed, dtype=jnp.int32), float(dropout_p), float(scale))
    consts = dict(
        H=n_heads,
        D=d,
        rand_stride=max_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        IS_CAUSAL=is_causal,
        ENABLE_DROPOUT=dropout_p > 0.0,
        IEEE=ieee,
    )
    delta_ref = jax.new_ref(jnp.zeros((total_q, n_heads), dtype=jnp.float32))
    jt.triton_call(
        q,
        k,
        v,
        do,
        lse,
        delta_ref,
        *common,
        stride_qt,
        stride_qh,
        stride_kt,
        stride_kh,
        kernel=_sdpa_bwd_kernel_delta,
        out_type=(),
        grid=(triton.cdiv(max_q, block_m), n_batch * n_heads),
        **consts,
    )
    delta = delta_ref[...]
    dk_ref = jax.new_ref(jnp.zeros_like(k))
    dv_ref = jax.new_ref(jnp.zeros_like(v))
    jt.triton_call(
        q,
        k,
        v,
        do,
        lse,
        delta,
        dk_ref,
        dv_ref,
        *common,
        stride_qt,
        stride_qh,
        stride_kt,
        stride_kh,
        kernel=_sdpa_bwd_kernel_dkv,
        out_type=(),
        grid=(triton.cdiv(max_k, block_n), n_batch * n_heads),
        **consts,
    )
    dq_ref = jax.new_ref(jnp.zeros_like(q))
    jt.triton_call(
        q,
        k,
        v,
        do,
        lse,
        delta,
        dq_ref,
        *common,
        stride_qt,
        stride_qh,
        stride_kt,
        stride_kh,
        kernel=_sdpa_bwd_kernel_dq,
        out_type=(),
        grid=(triton.cdiv(max_q, block_m), n_batch * n_heads),
        **consts,
    )
    return dq_ref[...], dk_ref[...], dv_ref[...]


@partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11, 12))
def sdpa_varlen(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    cu_q: jax.Array,
    len_q: jax.Array,
    cu_k: jax.Array,
    len_k: jax.Array,
    max_q: int,
    max_k: int,
    is_causal: bool,
    dropout_p: float,
    seed: int,
    scale: Optional[float],
) -> jax.Array:
    """
    :param q: [total_q, heads, dim], packed
    :param k: [total_k, heads, dim], packed, possibly a DIFFERENT packing than q (cross-attention)
    :param v: [total_k, heads, dim], same packing as k
    :param cu_q: [batch], first row of each query seq
    :param len_q: [batch]
    :param cu_k: [batch], first row of each kv seq
    :param len_k: [batch]
    :param max_q: static bound on the per-seq query length (grid size)
    :param max_k: static bound on the per-seq kv length (grid size, dropout offset stride)
    :param is_causal: only valid with q and k packed identically (self-attention)
    :param dropout_p: attention dropout, philox (NOT the aten flash RNG, see the module docstring)
    :param seed: dropout seed
    :param scale: qk scale, 1/sqrt(dim) if not given
    :return: out [total_q, heads, dim]

    Differentiable packed varlen attention. The JAX counterpart of the torch backend's aten flash
    varlen path.
    """
    return sdpa_varlen_fwd(
        q,
        k,
        v,
        cu_q,
        len_q,
        cu_k,
        len_k,
        max_q,
        max_k,
        is_causal=is_causal,
        dropout_p=dropout_p,
        seed=seed,
        scale=scale,
    )[0]


def _sdpa_varlen_fwd_vjp(q, k, v, cu_q, len_q, cu_k, len_k, max_q, max_k, is_causal, dropout_p, seed, scale):
    """
    :return: (out, residuals)

    Argument order: the fwd rule takes the PRIMAL order (statics last),
    while the bwd rule below takes the nondiff args FIRST.
    That asymmetry is jax's, and getting it wrong binds q to cu_q.
    """
    out, lse = sdpa_varlen_fwd(
        q,
        k,
        v,
        cu_q,
        len_q,
        cu_k,
        len_k,
        max_q,
        max_k,
        is_causal=is_causal,
        dropout_p=dropout_p,
        seed=seed,
        scale=scale,
    )
    return out, (q, k, v, out, lse, cu_q, len_q, cu_k, len_k)


def _sdpa_varlen_bwd_vjp(max_q, max_k, is_causal, dropout_p, seed, scale, res, do):
    """:return: grads matching the differentiable args of :func:`sdpa_varlen`"""
    q, k, v, out, lse, cu_q, len_q, cu_k, len_k = res
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    dq, dk, dv = sdpa_varlen_bwd(
        do,
        q,
        k,
        v,
        out,
        lse,
        cu_q,
        len_q,
        cu_k,
        len_k,
        max_q,
        max_k,
        is_causal=is_causal,
        dropout_p=dropout_p,
        seed=seed,
        scale=scale,
    )
    return dq, dk, dv, None, None, None, None


sdpa_varlen.defvjp(_sdpa_varlen_fwd_vjp, _sdpa_varlen_bwd_vjp)
