"""
Varlen (packed) relative-positional self-attention with post-softmax weight dropout
as a Triton kernel -- the one variant no stock kernel covers
(flash: no bias; FlexAttention: no dropout).
Used by :func:`returnn.frontend._packed_backend.PackedBackend.rel_pos_self_attention`.

Layout (like flash_attn_varlen):
  q, k, v: (total, H, D) -- sequences concatenated along the first axis
  seq_starts, seq_lens: (B,) int32 -- any packed layout, gaps between the seqs allowed
  bd: (total, H, R) -- precomputed position term (matrix b+d), pre-scaled,
      R = 2*max_len-1, centered: score(i,j) = scale*q_i k_j^T + bd[i, h, center+j-i]
Dropout: philox hash on (global row, local col) + a per-head-mixed seed,
so the backward recomputes the identical mask, no rng state saved.
The seed is a 1-elem int32 device tensor read in-kernel (tl.load),
so a captured device draw gives fresh masks across CUDA-graph replays
(a host int would be baked into the captured launch).
The bd gradient is row-owned (each query row writes its own bd row), no atomics.
Guard the import at the caller (needs Triton; the jit decorators run at import time).
"""

from __future__ import annotations

from typing import Optional, Union, Tuple
import math

import torch
import triton
import triton.language as tl


def is_available() -> bool:
    """:return: whether the kernel can run (needs a CUDA device)"""
    return torch.cuda.is_available()


def _seed_tensor(seed: Union[int, torch.Tensor], device) -> torch.Tensor:
    """
    :param seed: int or 1-elem int tensor
    :param device:
    :return: 1-elem int32 device tensor, read in-kernel via tl.load
    """
    if isinstance(seed, torch.Tensor):
        assert seed.numel() == 1
        return seed.to(device=device, dtype=torch.int32)
    return torch.full((1,), int(seed), dtype=torch.int32, device=device)


# noinspection PyPep8Naming,PyUnresolvedReferences
@triton.jit
def _rel_pos_fwd_kernel(
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
    H: tl.constexpr,
    D: tl.constexpr,
    R,
    center,
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
    if pid_m * BLOCK_M >= seq_len:
        return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # local q rows
    offs_d = tl.arange(0, D)
    q_rows = seq_start + offs_m
    q_mask = offs_m < seq_len
    q = tl.load(
        Q + q_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :],
        mask=q_mask[:, None],
        other=0.0,
    )
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)  # local kv cols
        n_mask = offs_n < seq_len
        k_rows = seq_start + offs_n
        k = tl.load(
            K + k_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :],
            mask=n_mask[:, None],
            other=0.0,
        )
        if IEEE:
            s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale  # (M, N)
        else:
            s = tl.dot(q, tl.trans(k)) * scale  # (M, N)
        rel = center + offs_n[None, :] - offs_m[:, None]  # (M, N), in [0, R)
        rel_valid = q_mask[:, None] & n_mask[None, :]
        bd = tl.load(
            BD + q_rows[:, None] * stride_bt + h * stride_bh + rel,
            mask=rel_valid,
            other=0.0,
        )
        s = s + bd
        s = tl.where(rel_valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        if ENABLE_DROPOUT:
            # philox offsets in INT32 (64-bit per-element math here made dropout ~20x slower);
            # unique per (global q row, local col), the head goes into the seed.
            # (Overflow wraps for total * R > 2^31: distant rows may then share a stream -- harmless.)
            offs = q_rows[:, None] * R + offs_n[None, :]
            rand = tl.rand(tl.load(Seed) + h * 1000003, offs)
            keep = rand >= dropout_p
            p_use = tl.where(keep, p / (1.0 - dropout_p), 0.0)
        else:
            p_use = p
        v_blk = tl.load(
            V + k_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :],
            mask=n_mask[:, None],
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


def rel_pos_att_fwd(q, k, v, bd, seq_starts, seq_lens, max_len, *, dropout_p=0.0, seed=0, scale=None):
    """forward, see :func:`rel_pos_att_varlen`. Returns (out, lse)."""
    total, n_heads, d = q.shape
    r = bd.shape[-1]
    assert r == 2 * max_len - 1
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    seed = _seed_tensor(seed, q.device)
    # zeros, not empty: the kernel writes only the valid rows (per-block early-exit),
    # so gap/junk rows would keep arbitrary garbage (possibly inf/nan),
    # which downstream residual adds spread and linear-layer weight grads mix in (x^T dy -> nan).
    out = torch.zeros_like(q)  # input dtype; the accumulation is f32 internally
    lse = torch.empty(total, n_heads, device=q.device, dtype=torch.float32)
    n_batch = seq_starts.numel()
    block_m, block_n = 64, 64
    grid = (triton.cdiv(max_len, block_m), n_batch * n_heads)
    _rel_pos_fwd_kernel[grid](
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
        H=n_heads,
        D=d,
        R=r,
        center=max_len - 1,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        ENABLE_DROPOUT=dropout_p > 0.0,
        IEEE=q.dtype == torch.float32,
    )
    return out, lse


# noinspection PyPep8Naming,PyUnresolvedReferences
@triton.jit
def _dropout_mask_kernel(
    Mask,
    Seed,
    dropout_p,
    R,
    max_len,
    stride_mt,
    stride_mh,
    BLOCK_N: tl.constexpr,
):
    # dump keep-mask per (global q row, head, local col) into (total, H, max_len)
    i = tl.program_id(0)  # global row
    h = tl.program_id(1)
    offs_n = tl.arange(0, BLOCK_N)
    offs = i * R + offs_n
    rand = tl.rand(tl.load(Seed) + h * 1000003, offs)
    keep = rand >= dropout_p
    tl.store(Mask + i * stride_mt + h * stride_mh + offs_n, keep.to(tl.int8), mask=offs_n < max_len)


# noinspection PyPep8Naming
@triton.jit
def _rel_pos_bwd_kernel_delta(
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
    H: tl.constexpr,
    D: tl.constexpr,
    R,
    center,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IEEE: tl.constexpr,
):
    # delta_i = sum_j p_ij * dp_ij in f32, from the same recomputed p/dp the other bwd kernels use.
    #
    # Why a dedicated pass, instead of the flash-style shortcut delta = rowsum(out * d_out)?
    # The shortcut rests on the identity sum_j p_ij dp_ij = do_i . out_i,
    # which holds only for out = sum_j p_ij v_j EXACTLY.
    # The stored out is bf16 (and was computed with p rounded to bf16 for the tensor-core dot),
    # while dp = do v^T is f32-accumulated,
    # so shortcut-delta and dp disagree at bf16 rounding scale (~2^-8).
    # ds = p * (dp - delta) lives on the cancellation dp ~= delta
    # at the dominant entries of sharp (trained) attention rows:
    # the true difference shrinks with sharpness, the rounding mismatch does not,
    # so it takes over exactly there -- and not as noise but as a per-row BIAS
    # (one delta error hits the whole row coherently).
    # ds is also the bd grad verbatim (DBD below), reduced over all rows
    # into the few pos-bias params: the bias survives the reduction,
    # drifts the pos term, sharpens attention further, and compounds into
    # training collapse (observed at head dim 128 by ~ep 7, 64 later).
    # The padded path is immune: torch's softmax bwd computes the row sum
    # from the same stored p and dp it multiplies, exact by construction --
    # this pass restores that property without materializing p
    # (f32 out storage would only shrink the mismatch, not remove it:
    # the fwd p is still rounded for the p.v dot).
    # Cost: one extra attention-shaped pass; fwd and memory unchanged.
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    seq_start = tl.load(SeqStarts + b)
    seq_len = tl.load(SeqLens + b)
    if pid_m * BLOCK_M >= seq_len:
        return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    m_mask = offs_m < seq_len
    q_rows = seq_start + offs_m
    q = tl.load(Q + q_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=m_mask[:, None], other=0.0)
    do = tl.load(DO + q_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=m_mask[:, None], other=0.0)
    lse = tl.load(Lse + q_rows * H + h, mask=m_mask, other=0.0)
    delta = tl.zeros([BLOCK_M], dtype=tl.float32)
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_len
        k_rows = seq_start + offs_n
        k = tl.load(K + k_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=n_mask[:, None], other=0.0)
        v = tl.load(V + k_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=n_mask[:, None], other=0.0)
        if IEEE:
            s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
            dp = tl.dot(do, tl.trans(v), input_precision="ieee")
        else:
            s = tl.dot(q, tl.trans(k)) * scale
            dp = tl.dot(do, tl.trans(v))
        rel = center + offs_n[None, :] - offs_m[:, None]
        valid = m_mask[:, None] & n_mask[None, :]
        bd = tl.load(BD + q_rows[:, None] * stride_bt + h * stride_bh + rel, mask=valid, other=0.0)
        s = tl.where(valid, s + bd, float("-inf"))
        p = tl.exp(s - lse[:, None])
        if ENABLE_DROPOUT:
            # int32 offsets, head in the seed -- see the fwd kernel
            offs = q_rows[:, None] * R + offs_n[None, :]
            keep = tl.rand(tl.load(Seed) + h * 1000003, offs) >= dropout_p
            dp = tl.where(keep, dp / (1.0 - dropout_p), 0.0)
        delta += tl.sum(tl.where(valid, p * dp, 0.0), 1)
    tl.store(Delta + q_rows * H + h, delta, mask=m_mask)


# noinspection PyPep8Naming
@triton.jit
def _rel_pos_bwd_kernel_dkv(
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
    H: tl.constexpr,
    D: tl.constexpr,
    R,
    center,
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
    if pid_n * BLOCK_N >= seq_len:
        return
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    n_mask = offs_n < seq_len
    k_rows = seq_start + offs_n
    k = tl.load(K + k_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=n_mask[:, None], other=0.0)
    v = tl.load(V + k_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=n_mask[:, None], other=0.0)
    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    for start_m in range(0, seq_len, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        m_mask = offs_m < seq_len
        q_rows = seq_start + offs_m
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
        rel = center + offs_n[None, :] - offs_m[:, None]
        valid = m_mask[:, None] & n_mask[None, :]
        bd = tl.load(BD + q_rows[:, None] * stride_bt + h * stride_bh + rel, mask=valid, other=0.0)
        s = tl.where(valid, s + bd, float("-inf"))
        p = tl.exp(s - lse[:, None])  # (M, N), normalized probs
        if ENABLE_DROPOUT:
            # int32 offsets, head in the seed -- see the fwd kernel
            offs = q_rows[:, None] * R + offs_n[None, :]
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
            # same keep as above (computed once; both blocks share the ENABLE_DROPOUT constexpr)
            # noinspection PyUnboundLocalVariable
            dp = tl.where(keep, dp / (1.0 - dropout_p), 0.0)
        ds = p * (dp - delta[:, None])  # (M, N)
        ds = tl.where(valid, ds, 0.0)
        if IEEE:
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q, input_precision="ieee") * scale
        else:
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q) * scale
    tl.store(DK + k_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], dk, mask=n_mask[:, None])
    tl.store(DV + k_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], dv, mask=n_mask[:, None])


# noinspection PyPep8Naming
@triton.jit
def _rel_pos_bwd_kernel_dq(
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
    H: tl.constexpr,
    D: tl.constexpr,
    R,
    center,
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
    if pid_m * BLOCK_M >= seq_len:
        return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    m_mask = offs_m < seq_len
    q_rows = seq_start + offs_m
    q = tl.load(Q + q_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=m_mask[:, None], other=0.0)
    do = tl.load(DO + q_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=m_mask[:, None], other=0.0)
    lse = tl.load(Lse + q_rows * H + h, mask=m_mask, other=0.0)
    delta = tl.load(Delta + q_rows * H + h, mask=m_mask, other=0.0)
    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_len
        k_rows = seq_start + offs_n
        k = tl.load(K + k_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=n_mask[:, None], other=0.0)
        v = tl.load(V + k_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=n_mask[:, None], other=0.0)
        if IEEE:
            s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        else:
            s = tl.dot(q, tl.trans(k)) * scale
        rel = center + offs_n[None, :] - offs_m[:, None]
        valid = m_mask[:, None] & n_mask[None, :]
        bd = tl.load(BD + q_rows[:, None] * stride_bt + h * stride_bh + rel, mask=valid, other=0.0)
        s = tl.where(valid, s + bd, float("-inf"))
        p = tl.exp(s - lse[:, None])
        if IEEE:
            dp = tl.dot(do, tl.trans(v), input_precision="ieee")
        else:
            dp = tl.dot(do, tl.trans(v))
        if ENABLE_DROPOUT:
            # int32 offsets, head in the seed -- see the fwd kernel
            offs = q_rows[:, None] * R + offs_n[None, :]
            keep = tl.rand(tl.load(Seed) + h * 1000003, offs) >= dropout_p
            dp = tl.where(keep, dp / (1.0 - dropout_p), 0.0)
        ds = p * (dp - delta[:, None])
        ds = tl.where(valid, ds, 0.0)
        if IEEE:
            dq += tl.dot(ds.to(k.dtype), k, input_precision="ieee") * scale
        else:
            dq += tl.dot(ds.to(k.dtype), k) * scale
        # bias grad: row-owned, d_bd[i, h, rel] += ds[i, j]; rel bins unique per (i, j) within the row
        tl.store(DBD + q_rows[:, None] * stride_bt + h * stride_bh + rel, ds, mask=valid)
    tl.store(DQ + q_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], dq, mask=m_mask[:, None])


def rel_pos_att_bwd(q, k, v, bd, seq_starts, seq_lens, max_len, out, lse, d_out, *, dropout_p=0.0, seed=0, scale=None):
    """backward, see :func:`rel_pos_att_varlen`. Returns (dq, dk, dv, dbd)."""
    _, n_heads, d = q.shape
    r = bd.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    seed = _seed_tensor(seed, q.device)
    del out  # unused since delta is recomputed in-kernel (see _rel_pos_bwd_kernel_delta); kept in the API
    delta = torch.zeros(q.shape[0], n_heads, device=q.device, dtype=torch.float32)
    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)
    dbd = torch.zeros_like(bd, dtype=torch.float32)
    n_batch = seq_starts.numel()
    # smaller tiles for large head dims (shared-memory limit)
    block_m, block_n = (64, 64) if d <= 64 else (32, 32)
    args = dict(
        H=n_heads,
        D=d,
        R=r,
        center=max_len - 1,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        ENABLE_DROPOUT=dropout_p > 0.0,
        IEEE=q.dtype == torch.float32,
    )
    _rel_pos_bwd_kernel_delta[(triton.cdiv(max_len, block_m), n_batch * n_heads)](
        q,
        k,
        v,
        bd,
        d_out,
        lse,
        delta,
        seq_starts,
        seq_lens,
        seed,
        dropout_p,
        scale,
        q.stride(0),
        q.stride(1),
        bd.stride(0),
        bd.stride(1),
        **args,
    )
    _rel_pos_bwd_kernel_dkv[(triton.cdiv(max_len, block_n), n_batch * n_heads)](
        q,
        k,
        v,
        bd,
        d_out,
        lse,
        delta,
        dk,
        dv,
        seq_starts,
        seq_lens,
        seed,
        dropout_p,
        scale,
        q.stride(0),
        q.stride(1),
        bd.stride(0),
        bd.stride(1),
        **args,
    )
    _rel_pos_bwd_kernel_dq[(triton.cdiv(max_len, block_m), n_batch * n_heads)](
        q,
        k,
        v,
        bd,
        d_out,
        lse,
        delta,
        dq,
        dbd,
        seq_starts,
        seq_lens,
        seed,
        dropout_p,
        scale,
        q.stride(0),
        q.stride(1),
        bd.stride(0),
        bd.stride(1),
        **args,
    )
    return dq, dk, dv, dbd


def dump_mask(total, n_heads, max_len, r, *, dropout_p, seed, device):
    """
    :return: the kept-weight boolean mask (total, H, max_len) the kernels use for the given seed,
        for exact-parity tests against an eager reference.
    """
    seed = _seed_tensor(seed, device)
    mask = torch.zeros(total, n_heads, max_len, dtype=torch.int8, device=device)
    block_n = triton.next_power_of_2(max_len)
    _dropout_mask_kernel[(total, n_heads)](
        mask,
        seed,
        dropout_p,
        R=r,
        max_len=max_len,
        stride_mt=mask.stride(0),
        stride_mh=mask.stride(1),
        BLOCK_N=block_n,
    )
    return mask.bool()


# noinspection PyAbstractClass
class _RelPosAttVarlen(torch.autograd.Function):
    """
    autograd wrapper, see :func:`rel_pos_att_varlen`.
    Modern API (forward without ctx + setup_context):
    required for functorch transforms (e.g. functionalize -> the make_fx/Inductor path);
    the old combined-forward style makes those silently reject the Function.
    """

    # noinspection PyMethodOverriding
    @staticmethod
    def forward(q, k, v, bd, seq_starts, seq_lens, max_seq_len, dropout_p, seed, scale):
        """forward. lse is a formal (non-differentiable) output, needed by the backward."""
        out, lse = rel_pos_att_fwd(
            q, k, v, bd, seq_starts, seq_lens, max_seq_len, dropout_p=dropout_p, seed=seed, scale=scale
        )
        return out, lse

    @staticmethod
    def setup_context(ctx, inputs, output):
        """setup ctx for backward"""
        q, k, v, bd, seq_starts, seq_lens, max_seq_len, dropout_p, seed, scale = inputs
        out, lse = output
        ctx.save_for_backward(q, k, v, bd, seq_starts, seq_lens, out, lse)
        ctx.max_seq_len, ctx.dropout_p, ctx.seed, ctx.scale = max_seq_len, dropout_p, seed, scale
        ctx.mark_non_differentiable(lse)

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(ctx, d_out, d_lse):
        """backward. d_lse unused (lse non-differentiable)."""
        d_lse  # noqa  # unused
        q, k, v, bd, seq_starts, seq_lens, out, lse = ctx.saved_tensors
        dq, dk, dv, dbd = rel_pos_att_bwd(
            q,
            k,
            v,
            bd,
            seq_starts,
            seq_lens,
            ctx.max_seq_len,
            out,
            lse,
            d_out.contiguous(),
            dropout_p=ctx.dropout_p,
            seed=ctx.seed,
            scale=ctx.scale,
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dbd.to(bd.dtype), None, None, None, None, None, None


_HAVE_LIB_OPS = False
if hasattr(torch.library, "custom_op"):  # torch >= 2.4

    @torch.library.custom_op("returnn::rel_pos_att_fwd", mutates_args=())
    def _lib_fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bd: torch.Tensor,
        seq_starts: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        dropout_p: float,
        seed: torch.Tensor,
        scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return rel_pos_att_fwd(
            q, k, v, bd, seq_starts, seq_lens, max_seq_len, dropout_p=dropout_p, seed=seed, scale=scale
        )

    @_lib_fwd.register_fake
    def _lib_fwd_fake(q, k, v, bd, seq_starts, seq_lens, max_seq_len, dropout_p, seed, scale):
        del k, v, bd, seq_starts, seq_lens, max_seq_len, dropout_p, seed, scale
        total, n_heads, _ = q.shape
        return torch.empty_like(q), q.new_empty((total, n_heads), dtype=torch.float32)

    @torch.library.custom_op("returnn::rel_pos_att_bwd", mutates_args=())
    def _lib_bwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bd: torch.Tensor,
        seq_starts: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        out: torch.Tensor,
        lse: torch.Tensor,
        d_out: torch.Tensor,
        dropout_p: float,
        seed: torch.Tensor,
        scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dq, dk, dv, dbd = rel_pos_att_bwd(
            q,
            k,
            v,
            bd,
            seq_starts,
            seq_lens,
            max_seq_len,
            out,
            lse,
            d_out.contiguous(),
            dropout_p=dropout_p,
            seed=seed,
            scale=scale,
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dbd.to(bd.dtype)

    @_lib_bwd.register_fake
    def _lib_bwd_fake(q, k, v, bd, seq_starts, seq_lens, max_seq_len, out, lse, d_out, dropout_p, seed, scale):
        del seq_starts, seq_lens, max_seq_len, out, lse, d_out, dropout_p, seed, scale
        return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), torch.empty_like(bd)

    def _lib_setup_context(ctx, inputs, output):
        q, k, v, bd, seq_starts, seq_lens, max_seq_len, dropout_p, seed, scale = inputs
        out, lse = output
        ctx.save_for_backward(q, k, v, bd, seq_starts, seq_lens, out, lse, seed)
        ctx.max_seq_len, ctx.dropout_p, ctx.scale = max_seq_len, dropout_p, scale

    def _lib_backward(ctx, d_out, d_lse):
        d_lse  # noqa  # unused (lse non-differentiable)
        q, k, v, bd, seq_starts, seq_lens, out, lse, seed = ctx.saved_tensors
        dq, dk, dv, dbd = torch.ops.returnn.rel_pos_att_bwd(
            q, k, v, bd, seq_starts, seq_lens, ctx.max_seq_len, out, lse, d_out, ctx.dropout_p, seed, ctx.scale
        )
        return dq, dk, dv, dbd, None, None, None, None, None, None

    torch.library.register_autograd("returnn::rel_pos_att_fwd", _lib_backward, setup_context=_lib_setup_context)

    def _fused_bd(qv: torch.Tensor, pos_emb: torch.Tensor, bd_scale: float, dtype: torch.dtype) -> torch.Tensor:
        # position term (total,H,R) from qv (total,H,D) and pos_emb (R,H,D),
        # pre-scaled per the kernel contract (the kernel scales only q k^T)
        return (torch.einsum("thd,rhd->thr", qv, pos_emb) * bd_scale).to(dtype)

    @torch.library.custom_op("returnn::rel_pos_att_fused_bd_fwd", mutates_args=())
    def _lib_fused_fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qv: torch.Tensor,
        pos_emb: torch.Tensor,
        seq_starts: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        dropout_p: float,
        seed: torch.Tensor,
        bd_scale: float,
        scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # bd lives only inside this opaque op (computed here, RECOMPUTED in the bwd op):
        # it never becomes a graph intermediate,
        # so AOT/Inductor cannot retain one bound-sized (total,H,R) buffer per layer for the backward.
        # contiguous: Inductor feeds custom ops in whatever layout it likes
        # (no stride constraint registered for these op names),
        # and the fake promises a contiguous output -- normalize here, both ops
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        bd = _fused_bd(qv.contiguous(), pos_emb.contiguous(), bd_scale, q.dtype)
        out, lse = rel_pos_att_fwd(
            q, k, v, bd, seq_starts, seq_lens, max_seq_len, dropout_p=dropout_p, seed=seed, scale=scale
        )
        return out.contiguous(), lse.contiguous()

    @_lib_fused_fwd.register_fake
    def _lib_fused_fwd_fake(q, k, v, qv, pos_emb, seq_starts, seq_lens, max_seq_len, dropout_p, seed, bd_scale, scale):
        del k, v, qv, pos_emb, seq_starts, seq_lens, max_seq_len, dropout_p, seed, bd_scale, scale
        total, n_heads, d = q.shape
        # plain new_empty (NOT empty_like):
        # the fake must promise the real op's contiguous output layout,
        # regardless of the traced input layout
        return q.new_empty((total, n_heads, d)), q.new_empty((total, n_heads), dtype=torch.float32)

    @torch.library.custom_op("returnn::rel_pos_att_fused_bd_bwd", mutates_args=())
    def _lib_fused_bwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qv: torch.Tensor,
        pos_emb: torch.Tensor,
        seq_starts: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        out: torch.Tensor,
        lse: torch.Tensor,
        d_out: torch.Tensor,
        dropout_p: float,
        seed: torch.Tensor,
        bd_scale: float,
        scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        qv, pos_emb = qv.contiguous(), pos_emb.contiguous()
        bd = _fused_bd(qv, pos_emb, bd_scale, q.dtype)
        dq, dk, dv, dbd = rel_pos_att_bwd(
            q,
            k,
            v,
            bd,
            seq_starts,
            seq_lens,
            max_seq_len,
            out,
            lse,
            d_out.contiguous(),
            dropout_p=dropout_p,
            seed=seed,
            scale=scale,
        )
        # chain rule through the fused position term,
        # einsums in the io dtype (matches the out-of-op matmul autograd of the non-fused path)
        dbd = (dbd * bd_scale).to(qv.dtype)
        d_qv = torch.einsum("thr,rhd->thd", dbd, pos_emb)
        d_pos = torch.einsum("thr,thd->rhd", dbd, qv)
        return (
            dq.to(q.dtype).contiguous(),
            dk.to(k.dtype).contiguous(),
            dv.to(v.dtype).contiguous(),
            d_qv.contiguous(),
            d_pos.to(pos_emb.dtype).contiguous(),
        )

    @_lib_fused_bwd.register_fake
    def _lib_fused_bwd_fake(
        q, k, v, qv, pos_emb, seq_starts, seq_lens, max_seq_len, out, lse, d_out, dropout_p, seed, bd_scale, scale
    ):
        # plain new_empty (NOT empty_like), see _lib_fused_fwd_fake
        del seq_starts, seq_lens, max_seq_len, out, lse, d_out, dropout_p, seed, bd_scale, scale
        return (
            q.new_empty(tuple(q.shape)),
            k.new_empty(tuple(k.shape)),
            v.new_empty(tuple(v.shape)),
            qv.new_empty(tuple(qv.shape)),
            pos_emb.new_empty(tuple(pos_emb.shape)),
        )

    def _lib_fused_setup_context(ctx, inputs, output):
        q, k, v, qv, pos_emb, seq_starts, seq_lens, max_seq_len, dropout_p, seed, bd_scale, scale = inputs
        out, lse = output
        ctx.save_for_backward(q, k, v, qv, pos_emb, seq_starts, seq_lens, out, lse, seed)
        ctx.max_seq_len, ctx.dropout_p, ctx.bd_scale, ctx.scale = max_seq_len, dropout_p, bd_scale, scale

    def _lib_fused_backward(ctx, d_out, d_lse):
        d_lse  # noqa  # unused (lse non-differentiable)
        q, k, v, qv, pos_emb, seq_starts, seq_lens, out, lse, seed = ctx.saved_tensors
        dq, dk, dv, d_qv, d_pos = torch.ops.returnn.rel_pos_att_fused_bd_bwd(
            q,
            k,
            v,
            qv,
            pos_emb,
            seq_starts,
            seq_lens,
            ctx.max_seq_len,
            out,
            lse,
            d_out,
            ctx.dropout_p,
            seed,
            ctx.bd_scale,
            ctx.scale,
        )
        return dq, dk, dv, d_qv, d_pos, None, None, None, None, None, None, None

    torch.library.register_autograd(
        "returnn::rel_pos_att_fused_bd_fwd", _lib_fused_backward, setup_context=_lib_fused_setup_context
    )

    _HAVE_LIB_OPS = True


def rel_pos_att_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bd: torch.Tensor,
    seq_starts: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    *,
    dropout_p: float = 0.0,
    seed: Optional[Union[int, torch.Tensor]] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    See the module docstring for the layout.

    :param q: (total, H, D)
    :param k: (total, H, D)
    :param v: (total, H, D)
    :param bd: (total, H, R), pre-scaled position term, R = 2*max_seq_len-1
    :param seq_starts: (B,), int32. start of each seq in the total axis (any layout, gaps allowed)
    :param seq_lens: (B,), int32
    :param max_seq_len: max seq len (R = 2*max_seq_len-1)
    :param dropout_p: post-softmax weight dropout probability
    :param seed: philox seed, int or 1-elem int tensor.
        default: drawn on the data's device when dropout is active
        (the CUDA philox generator is graph-managed,
        so a capture gets a fresh seed, thus fresh masks, on each replay)
    :param scale: applied to the q k^T term (default 1/sqrt(D))
    :return: attention output, (total, H, D), dtype of q
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if seed is None:
        seed = torch.randint(0, 2**31 - 1, (1,), dtype=torch.int32, device=q.device) if dropout_p > 0 else 0
    q, k, v, bd = q.contiguous(), k.contiguous(), v.contiguous(), bd.contiguous()
    if _HAVE_LIB_OPS:
        seed_t = _seed_tensor(seed, q.device)
        out, _ = torch.ops.returnn.rel_pos_att_fwd(
            q, k, v, bd, seq_starts, seq_lens, max_seq_len, dropout_p, seed_t, scale
        )
        return out
    out, _ = _RelPosAttVarlen.apply(q, k, v, bd, seq_starts, seq_lens, max_seq_len, dropout_p, seed, scale)
    return out


def rel_pos_att_varlen_fused_bd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: torch.Tensor,
    pos_emb: torch.Tensor,
    seq_starts: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    *,
    dropout_p: float = 0.0,
    seed: Optional[Union[int, torch.Tensor]] = None,
    bd_scale: Optional[float] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Like :func:`rel_pos_att_varlen`,
    but takes the factors of the position term (``bd[t,h,r] = bd_scale * sum_d qv[t,h,d] pos_emb[r,h,d]``)
    instead of the materialized ``bd``,
    and computes ``bd`` inside the op boundary -- in the forward AND again in the backward.
    Memory: ``bd`` (total, H, R) is the largest per-layer attention activation;
    keeping it out of the autograd/AOT graph means nothing retains one per layer for the backward
    (recompute-from-qv, one extra einsum in the backward).
    Requires torch >= 2.4 (:func:`torch.library.custom_op`); callers check ``have_lib_ops``.

    :param q: (total, H, D), content query (query + pos_bias_u, projected)
    :param k: (total, H, D)
    :param v: (total, H, D)
    :param qv: (total, H, D), position query (query + pos_bias_v, projected)
    :param pos_emb: (R, H, D), R = 2*max_seq_len-1, centered layout
    :param seq_starts: (B,), int32, like :func:`rel_pos_att_varlen`
    :param seq_lens: (B,), int32
    :param max_seq_len: max seq len (R = 2*max_seq_len-1)
    :param dropout_p: post-softmax weight dropout probability
    :param seed: philox seed, like :func:`rel_pos_att_varlen`
    :param bd_scale: applied to the position term (default 1/sqrt(D))
    :param scale: applied to the q k^T term (default 1/sqrt(D))
    :return: attention output, (total, H, D), dtype of q
    """
    assert _HAVE_LIB_OPS
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if bd_scale is None:
        bd_scale = 1.0 / math.sqrt(q.shape[-1])
    if seed is None:
        seed = torch.randint(0, 2**31 - 1, (1,), dtype=torch.int32, device=q.device) if dropout_p > 0 else 0
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    qv, pos_emb = qv.contiguous(), pos_emb.contiguous()
    seed_t = _seed_tensor(seed, q.device)
    out, _ = torch.ops.returnn.rel_pos_att_fused_bd_fwd(
        q, k, v, qv, pos_emb, seq_starts, seq_lens, max_seq_len, dropout_p, seed_t, bd_scale, scale
    )
    return out


def have_lib_ops() -> bool:
    """
    :return: whether the torch.library custom ops are registered (torch >= 2.4),
        see :func:`rel_pos_att_varlen_fused_bd`
    """
    return _HAVE_LIB_OPS
