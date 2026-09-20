"""
Chunked relative-positional self-attention with post-softmax weight dropout as a Triton kernel,
the attention of the chunked Conformer (:class:`ChunkedRelPosSelfAttention` in i6_experiments):
every chunk of S rows attends over its own S rows and the first C rows of each of the MEM
previous chunks, zero keys and values before the first chunk (they keep their softmax mass, only
through the position term), so the keys and values are never replicated MEM + 1 times.

Layout (like :mod:`rel_pos_att_triton`):
  q, k, v: (total, H, D), rows = chunk * S + position, sequences concatenated along the first axis
  seq_starts, seq_lens: (B,) int32, in rows (a multiple of S each), any packed layout
  bd: (total, H, R), precomputed position term (matrix b+d), pre-scaled, R = MEM*C + 2*S - 1:
      score(q=(qc,qi), k=(kc,kp)) = scale*q k^T + bd[row, h, center + (kc-qc)*C + kp - qi],
      center = MEM*C + S - 1, which is the rel-shift of the concatenated history axis.
Dropout, seed and the row-owned bd gradient work like :mod:`rel_pos_att_triton`,
with the dropout stream keyed by (global row, position-term index).
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
def _chunked_fwd_kernel(
    Q,
    K,
    V,
    BD,
    Out,
    Lse,
    SeqStarts,
    SeqLens,
    Seed,
    dropout_p,
    scale,
    stride_qt,
    stride_qh,
    stride_bt,
    stride_bh,
    stride_ot,
    stride_oh,
    R,
    center,
    MEM,
    H: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IEEE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    seq_start = tl.load(SeqStarts + b)
    seq_len = tl.load(SeqLens + b)
    m0 = pid_m * BLOCK_M
    if m0 >= seq_len:
        return
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q_rows = seq_start + offs_m
    q_mask = offs_m < seq_len
    qc = offs_m // S
    qi = offs_m - qc * S
    q = tl.load(
        Q + q_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :],
        mask=q_mask[:, None],
        other=0.0,
    )
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    lo = (m0 // S - MEM) * S
    hi = tl.minimum(((m0 + BLOCK_M - 1) // S + 1) * S, seq_len)
    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kc = (offs_n + MEM * S) // S - MEM
        kp = offs_n - kc * S
        n_load = (offs_n >= 0) & (offs_n < seq_len)
        k_rows = seq_start + offs_n
        k = tl.load(
            K + k_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :],
            mask=n_load[:, None],
            other=0.0,
        )
        if IEEE:
            s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        else:
            s = tl.dot(q, tl.trans(k)) * scale
        dc = kc[None, :] - qc[:, None]
        in_band = (dc == 0) | ((dc < 0) & (dc >= -MEM) & (kp[None, :] < C))
        in_band = in_band & q_mask[:, None] & (offs_n[None, :] < seq_len)
        idx = center + dc * C + (kp[None, :] - qi[:, None])
        bd = tl.load(BD + q_rows[:, None] * stride_bt + h * stride_bh + idx, mask=in_band, other=0.0)
        s = tl.where(in_band, s + bd, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - m_safe[:, None])
        alpha = tl.exp(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        if ENABLE_DROPOUT:
            offs = q_rows[:, None] * R + idx
            rand = tl.rand(tl.load(Seed) + h * 1000003, offs)
            keep = rand >= dropout_p
            p_use = tl.where(keep, p / (1.0 - dropout_p), 0.0)
        else:
            p_use = p
        v_blk = tl.load(
            V + k_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :],
            mask=n_load[:, None],
            other=0.0,
        )
        if IEEE:
            acc += tl.dot(p_use.to(v_blk.dtype), v_blk, input_precision="ieee")
        else:
            acc += tl.dot(p_use.to(v_blk.dtype), v_blk)
        m_i = m_new
    out = acc / l_i[:, None]
    tl.store(
        Out + q_rows[:, None] * stride_ot + h * stride_oh + offs_d[None, :],
        out,
        mask=q_mask[:, None],
    )
    lse = m_i + tl.log(l_i)
    tl.store(Lse + q_rows * H + h, lse, mask=q_mask)


# noinspection PyPep8Naming
@triton.jit
def _chunked_bwd_kernel_delta(
    Q,
    K,
    V,
    BD,
    DO,
    Lse,
    Delta,
    SeqStarts,
    SeqLens,
    Seed,
    dropout_p,
    scale,
    stride_qt,
    stride_qh,
    stride_bt,
    stride_bh,
    R,
    center,
    MEM,
    H: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IEEE: tl.constexpr,
):
    # delta_i = sum_j p_ij dp_ij from the recomputed p and dp, see _rel_pos_bwd_kernel_delta for why
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    seq_start = tl.load(SeqStarts + b)
    seq_len = tl.load(SeqLens + b)
    m0 = pid_m * BLOCK_M
    if m0 >= seq_len:
        return
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    m_mask = offs_m < seq_len
    q_rows = seq_start + offs_m
    qc = offs_m // S
    qi = offs_m - qc * S
    q = tl.load(Q + q_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=m_mask[:, None], other=0.0)
    do = tl.load(DO + q_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=m_mask[:, None], other=0.0)
    lse = tl.load(Lse + q_rows * H + h, mask=m_mask, other=0.0)
    delta = tl.zeros([BLOCK_M], dtype=tl.float32)
    lo = (m0 // S - MEM) * S
    hi = tl.minimum(((m0 + BLOCK_M - 1) // S + 1) * S, seq_len)
    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kc = (offs_n + MEM * S) // S - MEM
        kp = offs_n - kc * S
        n_load = (offs_n >= 0) & (offs_n < seq_len)
        k_rows = seq_start + offs_n
        k = tl.load(K + k_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=n_load[:, None], other=0.0)
        v = tl.load(V + k_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=n_load[:, None], other=0.0)
        if IEEE:
            s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
            dp = tl.dot(do, tl.trans(v), input_precision="ieee")
        else:
            s = tl.dot(q, tl.trans(k)) * scale
            dp = tl.dot(do, tl.trans(v))
        dc = kc[None, :] - qc[:, None]
        valid = (dc == 0) | ((dc < 0) & (dc >= -MEM) & (kp[None, :] < C))
        valid = valid & m_mask[:, None] & (offs_n[None, :] < seq_len)
        idx = center + dc * C + (kp[None, :] - qi[:, None])
        bd = tl.load(BD + q_rows[:, None] * stride_bt + h * stride_bh + idx, mask=valid, other=0.0)
        s = tl.where(valid, s + bd, float("-inf"))
        p = tl.exp(s - lse[:, None])
        if ENABLE_DROPOUT:
            offs = q_rows[:, None] * R + idx
            keep = tl.rand(tl.load(Seed) + h * 1000003, offs) >= dropout_p
            dp = tl.where(keep, dp / (1.0 - dropout_p), 0.0)
        delta += tl.sum(tl.where(valid, p * dp, 0.0), 1)
    tl.store(Delta + q_rows * H + h, delta, mask=m_mask)


# noinspection PyPep8Naming
@triton.jit
def _chunked_bwd_kernel_dkv(
    Q,
    K,
    V,
    BD,
    DO,
    Lse,
    Delta,
    DK,
    DV,
    SeqStarts,
    SeqLens,
    Seed,
    dropout_p,
    scale,
    stride_qt,
    stride_qh,
    stride_bt,
    stride_bh,
    R,
    center,
    MEM,
    H: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IEEE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    seq_start = tl.load(SeqStarts + b)
    seq_len = tl.load(SeqLens + b)
    n0 = pid_n * BLOCK_N
    if n0 >= seq_len:
        return
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    n_mask = offs_n < seq_len
    k_rows = seq_start + offs_n
    kc = offs_n // S
    kp = offs_n - kc * S
    k = tl.load(K + k_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=n_mask[:, None], other=0.0)
    v = tl.load(V + k_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=n_mask[:, None], other=0.0)
    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    lo = (n0 // S) * S
    hi = tl.minimum(((n0 + BLOCK_N - 1) // S + MEM + 1) * S, seq_len)
    for start_m in range(lo, hi, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        m_mask = offs_m < seq_len
        q_rows = seq_start + offs_m
        qc = offs_m // S
        qi = offs_m - qc * S
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
        dc = kc[None, :] - qc[:, None]
        valid = (dc == 0) | ((dc < 0) & (dc >= -MEM) & (kp[None, :] < C))
        valid = valid & m_mask[:, None] & n_mask[None, :]
        idx = center + dc * C + (kp[None, :] - qi[:, None])
        bd = tl.load(BD + q_rows[:, None] * stride_bt + h * stride_bh + idx, mask=valid, other=0.0)
        s = tl.where(valid, s + bd, float("-inf"))
        p = tl.exp(s - lse[:, None])
        if ENABLE_DROPOUT:
            offs = q_rows[:, None] * R + idx
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
        ds = tl.where(valid, ds, 0.0)
        if IEEE:
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q, input_precision="ieee") * scale
        else:
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q) * scale
    tl.store(DK + k_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], dk, mask=n_mask[:, None])
    tl.store(DV + k_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], dv, mask=n_mask[:, None])


# noinspection PyPep8Naming
@triton.jit
def _chunked_bwd_kernel_dq(
    Q,
    K,
    V,
    BD,
    DO,
    Lse,
    Delta,
    DQ,
    DBD,
    SeqStarts,
    SeqLens,
    Seed,
    dropout_p,
    scale,
    stride_qt,
    stride_qh,
    stride_bt,
    stride_bh,
    R,
    center,
    MEM,
    H: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IEEE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    seq_start = tl.load(SeqStarts + b)
    seq_len = tl.load(SeqLens + b)
    m0 = pid_m * BLOCK_M
    if m0 >= seq_len:
        return
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    m_mask = offs_m < seq_len
    q_rows = seq_start + offs_m
    qc = offs_m // S
    qi = offs_m - qc * S
    q = tl.load(Q + q_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=m_mask[:, None], other=0.0)
    do = tl.load(DO + q_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=m_mask[:, None], other=0.0)
    lse = tl.load(Lse + q_rows * H + h, mask=m_mask, other=0.0)
    delta = tl.load(Delta + q_rows * H + h, mask=m_mask, other=0.0)
    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    lo = (m0 // S - MEM) * S
    hi = tl.minimum(((m0 + BLOCK_M - 1) // S + 1) * S, seq_len)
    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kc = (offs_n + MEM * S) // S - MEM
        kp = offs_n - kc * S
        n_load = (offs_n >= 0) & (offs_n < seq_len)
        k_rows = seq_start + offs_n
        k = tl.load(K + k_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=n_load[:, None], other=0.0)
        v = tl.load(V + k_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=n_load[:, None], other=0.0)
        if IEEE:
            s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        else:
            s = tl.dot(q, tl.trans(k)) * scale
        dc = kc[None, :] - qc[:, None]
        valid = (dc == 0) | ((dc < 0) & (dc >= -MEM) & (kp[None, :] < C))
        valid = valid & m_mask[:, None] & (offs_n[None, :] < seq_len)
        idx = center + dc * C + (kp[None, :] - qi[:, None])
        bd = tl.load(BD + q_rows[:, None] * stride_bt + h * stride_bh + idx, mask=valid, other=0.0)
        s = tl.where(valid, s + bd, float("-inf"))
        p = tl.exp(s - lse[:, None])
        if IEEE:
            dp = tl.dot(do, tl.trans(v), input_precision="ieee")
        else:
            dp = tl.dot(do, tl.trans(v))
        if ENABLE_DROPOUT:
            offs = q_rows[:, None] * R + idx
            keep = tl.rand(tl.load(Seed) + h * 1000003, offs) >= dropout_p
            dp = tl.where(keep, dp / (1.0 - dropout_p), 0.0)
        ds = p * (dp - delta[:, None])
        ds = tl.where(valid, ds, 0.0)
        if IEEE:
            dq += tl.dot(ds.to(k.dtype), k, input_precision="ieee") * scale
        else:
            dq += tl.dot(ds.to(k.dtype), k) * scale
        tl.store(DBD + q_rows[:, None] * stride_bt + h * stride_bh + idx, ds, mask=valid)
    tl.store(DQ + q_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], dq, mask=m_mask[:, None])


def _geometry(bd, chunk_size: int, kept_rows: int, history: int):
    """:return: (R, center) of the position term for the given chunk geometry, checked against bd"""
    r = bd.shape[-1]
    assert r == history * kept_rows + 2 * chunk_size - 1, (r, chunk_size, kept_rows, history)
    return r, history * kept_rows + chunk_size - 1


def chunked_rel_pos_att_fwd(
    q, k, v, bd, seq_starts, seq_lens, max_rows, *, chunk_size, kept_rows, history, dropout_p=0.0, seed=0, scale=None
):
    """forward, see :func:`chunked_rel_pos_att`. Returns (out, lse)."""
    total, n_heads, d = q.shape
    r, center = _geometry(bd, chunk_size, kept_rows, history)
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    seed = _seed_tensor(seed, q.device)
    out = torch.zeros_like(q)
    lse = torch.empty(total, n_heads, device=q.device, dtype=torch.float32)
    n_batch = seq_starts.numel()
    block_m, block_n = (64, 64) if d <= 64 else (32, 32)
    grid = (triton.cdiv(max_rows, block_m), n_batch * n_heads)
    _chunked_fwd_kernel[grid](
        q,
        k,
        v,
        bd,
        out,
        lse,
        seq_starts,
        seq_lens,
        seed,
        dropout_p,
        scale,
        q.stride(0),
        q.stride(1),
        bd.stride(0),
        bd.stride(1),
        out.stride(0),
        out.stride(1),
        r,
        center,
        history,
        H=n_heads,
        D=d,
        S=chunk_size,
        C=kept_rows,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        ENABLE_DROPOUT=dropout_p > 0.0,
        IEEE=q.dtype == torch.float32,
    )
    return out, lse


def chunked_rel_pos_att_bwd(
    q,
    k,
    v,
    bd,
    seq_starts,
    seq_lens,
    max_rows,
    lse,
    d_out,
    *,
    chunk_size,
    kept_rows,
    history,
    dropout_p=0.0,
    seed=0,
    scale=None,
):
    """backward, see :func:`chunked_rel_pos_att`. Returns (dq, dk, dv, dbd)."""
    _, n_heads, d = q.shape
    r, center = _geometry(bd, chunk_size, kept_rows, history)
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    seed = _seed_tensor(seed, q.device)
    delta = torch.zeros(q.shape[0], n_heads, device=q.device, dtype=torch.float32)
    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)
    dbd = torch.zeros_like(bd, dtype=torch.float32)
    n_batch = seq_starts.numel()
    block_m, block_n = (64, 64) if d <= 64 else (32, 32)
    common = (seq_starts, seq_lens, seed, dropout_p, scale, q.stride(0), q.stride(1), bd.stride(0), bd.stride(1))
    args = dict(
        R=r,
        center=center,
        MEM=history,
        H=n_heads,
        D=d,
        S=chunk_size,
        C=kept_rows,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        ENABLE_DROPOUT=dropout_p > 0.0,
        IEEE=q.dtype == torch.float32,
    )
    _chunked_bwd_kernel_delta[(triton.cdiv(max_rows, block_m), n_batch * n_heads)](
        q, k, v, bd, d_out, lse, delta, *common, **args
    )
    _chunked_bwd_kernel_dkv[(triton.cdiv(max_rows, block_n), n_batch * n_heads)](
        q, k, v, bd, d_out, lse, delta, dk, dv, *common, **args
    )
    _chunked_bwd_kernel_dq[(triton.cdiv(max_rows, block_m), n_batch * n_heads)](
        q, k, v, bd, d_out, lse, delta, dq, dbd, *common, **args
    )
    return dq, dk, dv, dbd


def dump_keep_mask(total, n_heads, r, *, dropout_p, seed, device):
    """
    :return: the kept-weight mask (total, H, R) over the position-term index the kernels key their
        dropout stream by, for exact-parity tests against an eager reference
    """
    return dump_mask(total, n_heads, r, r, dropout_p=dropout_p, seed=seed, device=device)


# noinspection PyAbstractClass
class _ChunkedRelPosAtt(torch.autograd.Function):
    """autograd wrapper, see :func:`chunked_rel_pos_att`"""

    # noinspection PyMethodOverriding
    @staticmethod
    def forward(q, k, v, bd, seq_starts, seq_lens, max_rows, chunk_size, kept_rows, history, dropout_p, seed, scale):
        """forward. lse is a formal (non-differentiable) output, needed by the backward."""
        return chunked_rel_pos_att_fwd(
            q,
            k,
            v,
            bd,
            seq_starts,
            seq_lens,
            max_rows,
            chunk_size=chunk_size,
            kept_rows=kept_rows,
            history=history,
            dropout_p=dropout_p,
            seed=seed,
            scale=scale,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        """setup ctx for backward"""
        q, k, v, bd, seq_starts, seq_lens, max_rows, chunk_size, kept_rows, history, dropout_p, seed, scale = inputs
        out, lse = output
        ctx.save_for_backward(q, k, v, bd, seq_starts, seq_lens, lse)
        ctx.geometry = (max_rows, chunk_size, kept_rows, history)
        ctx.dropout_p, ctx.seed, ctx.scale = dropout_p, seed, scale
        ctx.mark_non_differentiable(lse)

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(ctx, d_out, d_lse):
        """backward. d_lse unused (lse non-differentiable)."""
        d_lse  # noqa  # unused
        q, k, v, bd, seq_starts, seq_lens, lse = ctx.saved_tensors
        max_rows, chunk_size, kept_rows, history = ctx.geometry
        dq, dk, dv, dbd = chunked_rel_pos_att_bwd(
            q,
            k,
            v,
            bd,
            seq_starts,
            seq_lens,
            max_rows,
            lse,
            d_out.contiguous(),
            chunk_size=chunk_size,
            kept_rows=kept_rows,
            history=history,
            dropout_p=ctx.dropout_p,
            seed=ctx.seed,
            scale=ctx.scale,
        )
        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dbd.to(bd.dtype)) + (None,) * 9


_HAVE_LIB_OPS = False
if hasattr(torch.library, "custom_op"):  # torch >= 2.4
    # Opaque ops with fake implementations and a registered backward, like in rel_pos_att_triton:
    # AOT tracing (the compiled step of torch_cuda_graph, no Dynamo) runs on fake tensors,
    # which the Triton launch of the autograd.Function above cannot take.

    @torch.library.custom_op("returnn::chunked_rel_pos_att_fwd", mutates_args=())
    def _lib_fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bd: torch.Tensor,
        seq_starts: torch.Tensor,
        seq_lens: torch.Tensor,
        max_rows: int,
        chunk_size: int,
        kept_rows: int,
        history: int,
        dropout_p: float,
        seed: torch.Tensor,
        scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # contiguous: Inductor feeds custom ops in whatever layout it likes, the fake promises contiguous outputs
        out, lse = chunked_rel_pos_att_fwd(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            bd.contiguous(),
            seq_starts.contiguous(),
            seq_lens.contiguous(),
            max_rows,
            chunk_size=chunk_size,
            kept_rows=kept_rows,
            history=history,
            dropout_p=dropout_p,
            seed=seed,
            scale=scale,
        )
        return out, lse

    @_lib_fwd.register_fake
    def _lib_fwd_fake(
        q, k, v, bd, seq_starts, seq_lens, max_rows, chunk_size, kept_rows, history, dropout_p, seed, scale
    ):
        del k, v, bd, seq_starts, seq_lens, max_rows, chunk_size, kept_rows, history, dropout_p, seed, scale
        total, n_heads, _ = q.shape
        return q.new_empty(tuple(q.shape)), q.new_empty((total, n_heads), dtype=torch.float32)

    @torch.library.custom_op("returnn::chunked_rel_pos_att_bwd", mutates_args=())
    def _lib_bwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bd: torch.Tensor,
        seq_starts: torch.Tensor,
        seq_lens: torch.Tensor,
        max_rows: int,
        lse: torch.Tensor,
        d_out: torch.Tensor,
        chunk_size: int,
        kept_rows: int,
        history: int,
        dropout_p: float,
        seed: torch.Tensor,
        scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dq, dk, dv, dbd = chunked_rel_pos_att_bwd(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            bd.contiguous(),
            seq_starts.contiguous(),
            seq_lens.contiguous(),
            max_rows,
            lse.contiguous(),
            d_out.contiguous(),
            chunk_size=chunk_size,
            kept_rows=kept_rows,
            history=history,
            dropout_p=dropout_p,
            seed=seed,
            scale=scale,
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dbd.to(bd.dtype)

    @_lib_bwd.register_fake
    def _lib_bwd_fake(
        q, k, v, bd, seq_starts, seq_lens, max_rows, lse, d_out, chunk_size, kept_rows, history, dropout_p, seed, scale
    ):
        del seq_starts, seq_lens, max_rows, lse, d_out, chunk_size, kept_rows, history, dropout_p, seed, scale
        return (
            q.new_empty(tuple(q.shape)),
            k.new_empty(tuple(k.shape)),
            v.new_empty(tuple(v.shape)),
            bd.new_empty(tuple(bd.shape)),
        )

    def _lib_setup_context(ctx, inputs, output):
        q, k, v, bd, seq_starts, seq_lens, max_rows, chunk_size, kept_rows, history, dropout_p, seed, scale = inputs
        _, lse = output
        ctx.save_for_backward(q, k, v, bd, seq_starts, seq_lens, lse, seed)
        ctx.geometry = (max_rows, chunk_size, kept_rows, history)
        ctx.dropout_p, ctx.scale = dropout_p, scale

    def _lib_backward(ctx, d_out, d_lse):
        d_lse  # noqa  # unused (lse non-differentiable)
        q, k, v, bd, seq_starts, seq_lens, lse, seed = ctx.saved_tensors
        max_rows, chunk_size, kept_rows, history = ctx.geometry
        dq, dk, dv, dbd = torch.ops.returnn.chunked_rel_pos_att_bwd(
            q,
            k,
            v,
            bd,
            seq_starts,
            seq_lens,
            max_rows,
            lse,
            d_out,
            chunk_size,
            kept_rows,
            history,
            ctx.dropout_p,
            seed,
            ctx.scale,
        )
        return (dq, dk, dv, dbd) + (None,) * 9

    torch.library.register_autograd("returnn::chunked_rel_pos_att_fwd", _lib_backward, setup_context=_lib_setup_context)

    _HAVE_LIB_OPS = True


def chunked_rel_pos_att(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bd: torch.Tensor,
    seq_starts: torch.Tensor,
    seq_lens: torch.Tensor,
    max_rows: int,
    *,
    chunk_size: int,
    kept_rows: int,
    history: int,
    dropout_p: float = 0.0,
    seed: Optional[Union[int, torch.Tensor]] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    See the module docstring for the layout.

    :param q: (total, H, D), rows = chunk * chunk_size + position
    :param k: (total, H, D)
    :param v: (total, H, D)
    :param bd: (total, H, R), pre-scaled position term, R = history * kept_rows + 2 * chunk_size - 1
    :param seq_starts: (B,), int32, start row of each seq (a multiple of chunk_size)
    :param seq_lens: (B,), int32, rows of each seq (a multiple of chunk_size)
    :param max_rows: max rows of a seq
    :param chunk_size: rows per chunk (center plus right context)
    :param kept_rows: rows of a previous chunk that stay in the history (the center)
    :param history: number of previous chunks in the history
    :param dropout_p: post-softmax weight dropout probability
    :param seed: philox seed, int or 1-elem int tensor, drawn on the device when dropout is active
    :param scale: applied to the q k^T term (default 1/sqrt(D))
    :return: attention output, (total, H, D), dtype of q
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if seed is None:
        seed = torch.randint(0, 2**31 - 1, (1,), dtype=torch.int32, device=q.device) if dropout_p > 0 else 0
    q, k, v, bd = q.contiguous(), k.contiguous(), v.contiguous(), bd.contiguous()
    if _HAVE_LIB_OPS:
        out, _ = torch.ops.returnn.chunked_rel_pos_att_fwd(
            q,
            k,
            v,
            bd,
            seq_starts,
            seq_lens,
            int(max_rows),
            chunk_size,
            kept_rows,
            history,
            dropout_p,
            _seed_tensor(seed, q.device),
            scale,
        )
        return out
    out, _ = _ChunkedRelPosAtt.apply(
        q, k, v, bd, seq_starts, seq_lens, max_rows, chunk_size, kept_rows, history, dropout_p, seed, scale
    )
    return out
