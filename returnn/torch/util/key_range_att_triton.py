"""
Attention in which every query row attends one contiguous range of key rows,
with post-softmax weight dropout, as a Triton kernel over packed (varlen) buffers.
Used by :func:`returnn.frontend._packed_backend.PackedBackend.scaled_dot_product_attention_key_ranges`,
e.g. for a label which attends the encoder frames of its own chunk, or of all chunks up to its own.

Layout:
  q: (total_q, H, D), k: (total_k, H, D), v: (total_k, H, Dv), rows of any packed layout, gaps allowed
  lo, hi: (total_q,) int32, query row i attends the key rows [lo[i], hi[i]) of the k buffer.
      An empty range (lo >= hi) gives a zero output row and no gradient.
Dropout and seed work like in :mod:`rel_pos_att_triton`,
philox on (query row, key row) with int32 offsets plus a per-head-mixed seed,
the seed a 1-elem int32 device tensor, so CUDA-graph replays draw fresh masks.
The backward recomputes delta in f32 (see :mod:`rel_pos_att_triton` for why).
dk and dv are key-owned, each key block loops over the query blocks whose ranges reach it
(from the range span of every query block), so there are no atomics and the result is deterministic.
Guard the import at the caller (it needs Triton, the jit decorators run at import time).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
import math

import torch
import triton
import triton.language as tl

from .rel_pos_att_triton import _seed_tensor, dump_mask


def is_available() -> bool:
    """:return: whether the kernel can run (needs a CUDA device)"""
    return torch.cuda.is_available()


# noinspection PyPep8Naming,PyUnresolvedReferences
@triton.jit
def _key_range_fwd_kernel(
    Q,
    K,
    V,
    Out,
    Lse,
    Lo,
    Hi,
    Seed,
    dropout_p,
    scale,
    n_q,
    n_k,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_ot,
    stride_oh,
    H: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IEEE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < n_q
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    lo = tl.load(Lo + offs_m, mask=m_mask, other=0)
    hi = tl.load(Hi + offs_m, mask=m_mask, other=0)
    # one span covers the key rows any row of this block attends, empty rows excluded
    nonempty = m_mask & (lo < hi)
    span_lo = tl.min(tl.where(nonempty, lo, n_k), 0)
    span_hi = tl.max(tl.where(nonempty, hi, 0), 0)
    q = tl.load(
        Q + offs_m[:, None] * stride_qt + h * stride_qh + offs_d[None, :],
        mask=m_mask[:, None] & (offs_d[None, :] < D),
        other=0.0,
    )
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    for start_n in range((span_lo // BLOCK_N) * BLOCK_N, span_hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < n_k
        k = tl.load(
            K + offs_n[:, None] * stride_kt + h * stride_kh + offs_d[None, :],
            mask=n_mask[:, None] & (offs_d[None, :] < D),
            other=0.0,
        )
        if IEEE:
            s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        else:
            s = tl.dot(q, tl.trans(k)) * scale
        valid = (offs_n[None, :] >= lo[:, None]) & (offs_n[None, :] < hi[:, None]) & m_mask[:, None]
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        # a row without any valid key so far keeps m = -inf, exp against 0 keeps it at zero mass
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - m_safe[:, None])
        alpha = tl.exp(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        if ENABLE_DROPOUT:
            # int32 philox offsets, unique per (query row, key row), the head goes into the seed
            # (overflow wraps for total_q * total_k > 2^31, harmless as in rel_pos_att_triton)
            offs = offs_m[:, None] * n_k + offs_n[None, :]
            keep = tl.rand(tl.load(Seed) + h * 1000003, offs) >= dropout_p
            p_use = tl.where(keep, p / (1.0 - dropout_p), 0.0)
        else:
            p_use = p
        v = tl.load(
            V + offs_n[:, None] * stride_vt + h * stride_vh + offs_dv[None, :],
            mask=n_mask[:, None] & (offs_dv[None, :] < DV),
            other=0.0,
        )
        if IEEE:
            acc += tl.dot(p_use.to(v.dtype), v, input_precision="ieee")
        else:
            acc += tl.dot(p_use.to(v.dtype), v)
        m_i = m_new
    has_keys = l_i > 0.0
    l_safe = tl.where(has_keys, l_i, 1.0)
    out = tl.where(has_keys[:, None], acc / l_safe[:, None], 0.0)
    tl.store(
        Out + offs_m[:, None] * stride_ot + h * stride_oh + offs_dv[None, :],
        out,
        mask=m_mask[:, None] & (offs_dv[None, :] < DV),
    )
    tl.store(Lse + offs_m * H + h, tl.where(has_keys, m_i + tl.log(l_safe), 0.0), mask=m_mask)


# noinspection PyPep8Naming,PyUnresolvedReferences
@triton.jit
def _key_range_bwd_kernel_q(
    Q,
    K,
    V,
    DO,
    Lse,
    Delta,
    DQ,
    Lo,
    Hi,
    Seed,
    dropout_p,
    scale,
    n_q,
    n_k,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_dot,
    stride_doh,
    H: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IEEE: tl.constexpr,
    COMPUTE_DELTA: tl.constexpr,
):
    # query-owned pass, run twice, first with COMPUTE_DELTA (delta_i = sum_j p_ij dp_ij in f32,
    # from the same recomputed p and dp as the other passes, see rel_pos_att_triton), then for dq
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < n_q
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    lo = tl.load(Lo + offs_m, mask=m_mask, other=0)
    hi = tl.load(Hi + offs_m, mask=m_mask, other=0)
    nonempty = m_mask & (lo < hi)
    span_lo = tl.min(tl.where(nonempty, lo, n_k), 0)
    span_hi = tl.max(tl.where(nonempty, hi, 0), 0)
    q = tl.load(
        Q + offs_m[:, None] * stride_qt + h * stride_qh + offs_d[None, :],
        mask=m_mask[:, None] & (offs_d[None, :] < D),
        other=0.0,
    )
    do = tl.load(
        DO + offs_m[:, None] * stride_dot + h * stride_doh + offs_dv[None, :],
        mask=m_mask[:, None] & (offs_dv[None, :] < DV),
        other=0.0,
    )
    lse = tl.load(Lse + offs_m * H + h, mask=m_mask, other=0.0)
    if COMPUTE_DELTA:
        delta = tl.zeros([BLOCK_M], dtype=tl.float32)
    else:
        delta = tl.load(Delta + offs_m * H + h, mask=m_mask, other=0.0)
    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    for start_n in range((span_lo // BLOCK_N) * BLOCK_N, span_hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < n_k
        k = tl.load(
            K + offs_n[:, None] * stride_kt + h * stride_kh + offs_d[None, :],
            mask=n_mask[:, None] & (offs_d[None, :] < D),
            other=0.0,
        )
        v = tl.load(
            V + offs_n[:, None] * stride_vt + h * stride_vh + offs_dv[None, :],
            mask=n_mask[:, None] & (offs_dv[None, :] < DV),
            other=0.0,
        )
        if IEEE:
            s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
            dp = tl.dot(do, tl.trans(v), input_precision="ieee")
        else:
            s = tl.dot(q, tl.trans(k)) * scale
            dp = tl.dot(do, tl.trans(v))
        valid = (offs_n[None, :] >= lo[:, None]) & (offs_n[None, :] < hi[:, None]) & m_mask[:, None]
        p = tl.where(valid, tl.exp(s - lse[:, None]), 0.0)
        if ENABLE_DROPOUT:
            offs = offs_m[:, None] * n_k + offs_n[None, :]
            keep = tl.rand(tl.load(Seed) + h * 1000003, offs) >= dropout_p
            dp = tl.where(keep, dp / (1.0 - dropout_p), 0.0)
        if COMPUTE_DELTA:
            delta += tl.sum(p * dp, 1)
        else:
            ds = p * (dp - delta[:, None])
            if IEEE:
                dq += tl.dot(ds.to(k.dtype), k, input_precision="ieee") * scale
            else:
                dq += tl.dot(ds.to(k.dtype), k) * scale
    if COMPUTE_DELTA:
        tl.store(Delta + offs_m * H + h, delta, mask=m_mask)
    else:
        tl.store(
            DQ + offs_m[:, None] * stride_qt + h * stride_qh + offs_d[None, :],
            dq,
            mask=m_mask[:, None] & (offs_d[None, :] < D),
        )


# noinspection PyPep8Naming,PyUnresolvedReferences
@triton.jit
def _key_range_bwd_kernel_kv(
    Q,
    K,
    V,
    DO,
    Lse,
    Delta,
    DK,
    DV_,
    Lo,
    Hi,
    SpanLo,
    SpanHi,
    Seed,
    dropout_p,
    scale,
    n_q,
    n_k,
    n_q_blocks,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_dot,
    stride_doh,
    H: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IEEE: tl.constexpr,
):
    # key-owned, this block of key rows accumulates dk and dv over every query block whose span reaches it
    pid_n = tl.program_id(0)
    h = tl.program_id(1)
    n_start = pid_n * BLOCK_N
    offs_n = n_start + tl.arange(0, BLOCK_N)
    n_mask = offs_n < n_k
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    k = tl.load(
        K + offs_n[:, None] * stride_kt + h * stride_kh + offs_d[None, :],
        mask=n_mask[:, None] & (offs_d[None, :] < D),
        other=0.0,
    )
    v = tl.load(
        V + offs_n[:, None] * stride_vt + h * stride_vh + offs_dv[None, :],
        mask=n_mask[:, None] & (offs_dv[None, :] < DV),
        other=0.0,
    )
    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DV], dtype=tl.float32)
    for qb in range(0, n_q_blocks):
        block_lo = tl.load(SpanLo + qb)
        block_hi = tl.load(SpanHi + qb)
        if (block_lo < n_start + BLOCK_N) & (block_hi > n_start):
            offs_m = qb * BLOCK_M + tl.arange(0, BLOCK_M)
            m_mask = offs_m < n_q
            lo = tl.load(Lo + offs_m, mask=m_mask, other=0)
            hi = tl.load(Hi + offs_m, mask=m_mask, other=0)
            q = tl.load(
                Q + offs_m[:, None] * stride_qt + h * stride_qh + offs_d[None, :],
                mask=m_mask[:, None] & (offs_d[None, :] < D),
                other=0.0,
            )
            do = tl.load(
                DO + offs_m[:, None] * stride_dot + h * stride_doh + offs_dv[None, :],
                mask=m_mask[:, None] & (offs_dv[None, :] < DV),
                other=0.0,
            )
            lse = tl.load(Lse + offs_m * H + h, mask=m_mask, other=0.0)
            delta = tl.load(Delta + offs_m * H + h, mask=m_mask, other=0.0)
            if IEEE:
                s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
            else:
                s = tl.dot(q, tl.trans(k)) * scale
            valid = (offs_n[None, :] >= lo[:, None]) & (offs_n[None, :] < hi[:, None])
            valid = valid & m_mask[:, None] & n_mask[None, :]
            p = tl.where(valid, tl.exp(s - lse[:, None]), 0.0)
            if ENABLE_DROPOUT:
                offs = offs_m[:, None] * n_k + offs_n[None, :]
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
                # noinspection PyUnboundLocalVariable
                dp = tl.where(keep, dp / (1.0 - dropout_p), 0.0)
            ds = p * (dp - delta[:, None])
            if IEEE:
                dk += tl.dot(tl.trans(ds.to(q.dtype)), q, input_precision="ieee") * scale
            else:
                dk += tl.dot(tl.trans(ds.to(q.dtype)), q) * scale
    tl.store(
        DK + offs_n[:, None] * stride_kt + h * stride_kh + offs_d[None, :],
        dk,
        mask=n_mask[:, None] & (offs_d[None, :] < D),
    )
    tl.store(
        DV_ + offs_n[:, None] * stride_vt + h * stride_vh + offs_dv[None, :],
        dv,
        mask=n_mask[:, None] & (offs_dv[None, :] < DV),
    )


def _blocks(d: int, dv: int) -> Tuple[int, int]:
    """:return: (BLOCK_M, BLOCK_N), smaller tiles for large head dims (shared-memory limit)"""
    return (64, 64) if max(d, dv) <= 64 else (32, 32)


def _feat_block(d: int) -> int:
    """:return: the power-of-2 tile over a head dim, at least 16 (tl.dot)"""
    return max(16, triton.next_power_of_2(d))


def _block_spans(lo: torch.Tensor, hi: torch.Tensor, block_m: int, n_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :return: per block of block_m query rows the span of key rows its ranges cover,
        (span_lo, span_hi) (n_q_blocks,) int32, (n_k, 0) for a block without any key.
        Pure device ops (no host sync), so it records into a CUDA graph.
    """
    n_q = lo.shape[0]
    n_blocks = triton.cdiv(n_q, block_m)
    nonempty = lo < hi
    lo_m = torch.where(nonempty, lo, torch.full_like(lo, n_k))
    hi_m = torch.where(nonempty, hi, torch.zeros_like(hi))
    pad = n_blocks * block_m - n_q
    if pad:
        lo_m = torch.nn.functional.pad(lo_m, (0, pad), value=n_k)
        hi_m = torch.nn.functional.pad(hi_m, (0, pad), value=0)
    span_lo = lo_m.view(n_blocks, block_m).amin(1).to(torch.int32).contiguous()
    span_hi = hi_m.view(n_blocks, block_m).amax(1).to(torch.int32).contiguous()
    return span_lo, span_hi


def key_range_att_fwd(q, k, v, lo, hi, *, dropout_p=0.0, seed=0, scale=None):
    """forward, see :func:`key_range_att`. Returns (out, lse)."""
    n_q, n_heads, d = q.shape
    n_k, _, dv = v.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    seed = _seed_tensor(seed, q.device)
    # zeros, not empty, so a zero-row buffer that launches nothing still returns a defined output
    out = torch.zeros(n_q, n_heads, dv, device=q.device, dtype=v.dtype)
    lse = torch.zeros(n_q, n_heads, device=q.device, dtype=torch.float32)
    if n_q == 0 or n_k == 0:
        return out, lse
    block_m, block_n = _blocks(d, dv)
    _key_range_fwd_kernel[(triton.cdiv(n_q, block_m), n_heads)](
        q,
        k,
        v,
        out,
        lse,
        lo,
        hi,
        seed,
        dropout_p,
        scale,
        n_q,
        n_k,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        out.stride(0),
        out.stride(1),
        H=n_heads,
        D=d,
        DV=dv,
        BLOCK_D=_feat_block(d),
        BLOCK_DV=_feat_block(dv),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        ENABLE_DROPOUT=dropout_p > 0.0,
        IEEE=q.dtype == torch.float32,
    )
    return out, lse


def key_range_att_bwd(q, k, v, lo, hi, lse, d_out, *, dropout_p=0.0, seed=0, scale=None):
    """backward, see :func:`key_range_att`. Returns (dq, dk, dv) in f32."""
    n_q, n_heads, d = q.shape
    n_k, _, dv_dim = v.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    seed = _seed_tensor(seed, q.device)
    delta = torch.zeros(n_q, n_heads, device=q.device, dtype=torch.float32)
    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)
    if n_q == 0 or n_k == 0:
        return dq, dk, dv
    block_m, block_n = _blocks(d, dv_dim)
    common = dict(
        H=n_heads,
        D=d,
        DV=dv_dim,
        BLOCK_D=_feat_block(d),
        BLOCK_DV=_feat_block(dv_dim),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        ENABLE_DROPOUT=dropout_p > 0.0,
        IEEE=q.dtype == torch.float32,
    )
    strides = (
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        d_out.stride(0),
        d_out.stride(1),
    )
    q_grid = (triton.cdiv(n_q, block_m), n_heads)
    for compute_delta in (True, False):
        _key_range_bwd_kernel_q[q_grid](
            q,
            k,
            v,
            d_out,
            lse,
            delta,
            dq,
            lo,
            hi,
            seed,
            dropout_p,
            scale,
            n_q,
            n_k,
            *strides,
            COMPUTE_DELTA=compute_delta,
            **common,
        )
    span_lo, span_hi = _block_spans(lo, hi, block_m, n_k)
    _key_range_bwd_kernel_kv[(triton.cdiv(n_k, block_n), n_heads)](
        q,
        k,
        v,
        d_out,
        lse,
        delta,
        dk,
        dv,
        lo,
        hi,
        span_lo,
        span_hi,
        seed,
        dropout_p,
        scale,
        n_q,
        n_k,
        span_lo.shape[0],
        *strides,
        **common,
    )
    return dq, dk, dv


def dump_keep_mask(n_q: int, n_heads: int, n_k: int, *, dropout_p: float, seed, device) -> torch.Tensor:
    """
    :return: the kept-weight boolean mask (n_q, H, n_k) over (query row, head, key row) the kernels use
        for the given seed, for exact-parity tests against an eager reference
    """
    return dump_mask(n_q, n_heads, n_k, n_k, dropout_p=dropout_p, seed=seed, device=device)


# noinspection PyAbstractClass
class _KeyRangeAtt(torch.autograd.Function):
    """autograd wrapper, see :func:`key_range_att`"""

    # noinspection PyMethodOverriding
    @staticmethod
    def forward(q, k, v, lo, hi, dropout_p, seed, scale):
        """forward. lse is a formal (non-differentiable) output, needed by the backward."""
        return key_range_att_fwd(q, k, v, lo, hi, dropout_p=dropout_p, seed=seed, scale=scale)

    @staticmethod
    def setup_context(ctx, inputs, output):
        """setup ctx for backward"""
        q, k, v, lo, hi, dropout_p, seed, scale = inputs
        _, lse = output
        ctx.save_for_backward(q, k, v, lo, hi, lse)
        ctx.dropout_p, ctx.seed, ctx.scale = dropout_p, seed, scale
        ctx.mark_non_differentiable(lse)

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(ctx, d_out, d_lse):
        """backward. d_lse unused (lse non-differentiable)."""
        d_lse  # noqa  # unused
        q, k, v, lo, hi, lse = ctx.saved_tensors
        dq, dk, dv = key_range_att_bwd(
            q, k, v, lo, hi, lse, d_out.contiguous(), dropout_p=ctx.dropout_p, seed=ctx.seed, scale=ctx.scale
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None, None, None


def key_range_att(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    *,
    dropout_p: float = 0.0,
    seed: Optional[Union[int, torch.Tensor]] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    See the module docstring for the layout.

    :param q: (total_q, H, D)
    :param k: (total_k, H, D)
    :param v: (total_k, H, Dv)
    :param lo: (total_q,) int, first key row every query row attends
    :param hi: (total_q,) int, end (exclusive) of the key rows every query row attends
    :param dropout_p: post-softmax weight dropout probability
    :param seed: philox seed, int or 1-elem int tensor.
        By default drawn on the data's device when dropout is active
        (the CUDA philox generator is graph-managed,
        so a capture gets a fresh seed, thus fresh masks, on each replay)
    :param scale: applied to the q k^T term (default 1/sqrt(D))
    :return: attention output, (total_q, H, Dv), dtype of v, zero on rows with an empty range
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if seed is None:
        seed = torch.randint(0, 2**31 - 1, (1,), dtype=torch.int32, device=q.device) if dropout_p > 0 else 0
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    lo, hi = lo.to(torch.int32).contiguous(), hi.to(torch.int32).contiguous()
    out, _ = _KeyRangeAtt.apply(q, k, v, lo, hi, dropout_p, seed, scale)
    return out
