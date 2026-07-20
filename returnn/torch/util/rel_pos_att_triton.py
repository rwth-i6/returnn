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
Dropout: philox hash on (global row, head, local col) + seed,
so the backward recomputes the identical mask, no rng state saved.
The bd gradient is row-owned (each query row writes its own bd row), no atomics.
Guard the import at the caller (needs Triton; the jit decorators run at import time).
"""

from __future__ import annotations

from typing import Optional
import math

import torch
import triton
import triton.language as tl


def is_available() -> bool:
    """:return: whether the kernel can run (needs a CUDA device)"""
    return torch.cuda.is_available()


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
    seed,
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
    total,
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
            # philox offsets unique per (global q row, head, local col)
            offs = (q_rows[:, None].to(tl.int64) * H + h) * R + offs_n[None, :].to(tl.int64)
            rand = tl.rand(seed, offs)
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
    total, n_heads, d = q.shape
    r = bd.shape[-1]
    assert r == 2 * max_len - 1
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    out = torch.empty_like(q)  # input dtype; the accumulation is f32 internally
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
        total=total,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        ENABLE_DROPOUT=dropout_p > 0.0,
        IEEE=q.dtype == torch.float32,
    )
    return out, lse


def _reference(q, k, v, bd, cu_seqlens, max_len, *, scale):
    # eager per-seq reference, dropout 0
    outs = []
    center = max_len - 1
    for b in range(cu_seqlens.numel() - 1):
        s0, s1 = int(cu_seqlens[b]), int(cu_seqlens[b + 1])
        qb, kb, vb, bdb = q[s0:s1].float(), k[s0:s1].float(), v[s0:s1].float(), bd[s0:s1].float()
        ln = s1 - s0
        s = torch.einsum("ihd,jhd->hij", qb, kb) * scale
        idx = center + torch.arange(ln)[None, :] - torch.arange(ln)[:, None]  # (i, j)
        s = s + bdb.permute(1, 0, 2).gather(2, idx.unsqueeze(0).expand(s.shape[0], -1, -1).to(q.device))
        w = torch.softmax(s, dim=-1)
        outs.append(torch.einsum("hij,jhd->ihd", w, vb))
    return torch.cat(outs, 0)


@triton.jit
def _dropout_mask_kernel(
    Mask,
    cu_seqlens,
    seed,
    dropout_p,
    H: tl.constexpr,
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
    offs = (i.to(tl.int64) * H + h) * R + offs_n.to(tl.int64)
    rand = tl.rand(seed, offs)
    keep = rand >= dropout_p
    tl.store(Mask + i * stride_mt + h * stride_mh + offs_n, keep.to(tl.int8), mask=offs_n < max_len)


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
    seed,
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
            offs = (q_rows[:, None].to(tl.int64) * H + h) * R + offs_n[None, :].to(tl.int64)
            keep = tl.rand(seed, offs) >= dropout_p
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
            offs = (q_rows[:, None].to(tl.int64) * H + h) * R + offs_n[None, :].to(tl.int64)
            keep = tl.rand(seed, offs) >= dropout_p
            dp = tl.where(keep, dp / (1.0 - dropout_p), 0.0)
        ds = p * (dp - delta[:, None])  # (M, N)
        ds = tl.where(valid, ds, 0.0)
        if IEEE:
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q, input_precision="ieee") * scale
        else:
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q) * scale
    tl.store(DK + k_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], dk, mask=n_mask[:, None])
    tl.store(DV + k_rows[:, None] * stride_qt + h * stride_qh + offs_d[None, :], dv, mask=n_mask[:, None])


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
    seed,
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
            offs = (q_rows[:, None].to(tl.int64) * H + h) * R + offs_n[None, :].to(tl.int64)
            keep = tl.rand(seed, offs) >= dropout_p
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
    total, n_heads, d = q.shape
    r = bd.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    delta = (d_out.float() * out.float()).sum(-1)  # (total, H)
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


def dump_mask(cu_seqlens, total, n_heads, max_len, r, *, dropout_p, seed, device):
    mask = torch.zeros(total, n_heads, max_len, dtype=torch.int8, device=device)
    block_n = triton.next_power_of_2(max_len)
    _dropout_mask_kernel[(total, n_heads)](
        mask,
        cu_seqlens,
        seed,
        dropout_p,
        H=n_heads,
        R=r,
        max_len=max_len,
        stride_mt=mask.stride(0),
        stride_mh=mask.stride(1),
        BLOCK_N=block_n,
    )
    return mask.bool()


class _RelPosAttVarlen(torch.autograd.Function):
    """autograd wrapper, see :func:`rel_pos_att_varlen`"""

    # noinspection PyMethodOverriding
    @staticmethod
    def forward(ctx, q, k, v, bd, seq_starts, seq_lens, max_seq_len, dropout_p, seed, scale):
        """forward"""
        out, lse = rel_pos_att_fwd(
            q, k, v, bd, seq_starts, seq_lens, max_seq_len, dropout_p=dropout_p, seed=seed, scale=scale
        )
        ctx.save_for_backward(q, k, v, bd, seq_starts, seq_lens, out, lse)
        ctx.max_seq_len, ctx.dropout_p, ctx.seed, ctx.scale = max_seq_len, dropout_p, seed, scale
        return out

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(ctx, d_out):
        """backward"""
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
    seed: Optional[int] = None,
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
    :param seed: philox seed. default: drawn from the torch default generator
    :param scale: applied to the q k^T term (default 1/sqrt(D))
    :return: attention output, (total, H, D), dtype of q
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (), device="cpu"))
    q, k, v, bd = q.contiguous(), k.contiguous(), v.contiguous(), bd.contiguous()
    return _RelPosAttVarlen.apply(q, k, v, bd, seq_starts, seq_lens, max_seq_len, dropout_p, seed, scale)
