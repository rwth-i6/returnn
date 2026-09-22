"""
Triton kernels for the packed monotonic RNN-T loss, see :mod:`returnn.torch.util.monotonic_rnnt`.

The loss needs three numbers per lattice cell, the log normalizer, the blank log probability and the log
probability of the next reference label, and its gradient needs the softmax of the cell again. Those two
passes are bandwidth bound over ``[cells, vocab]``.

The forward-backward recursion is the opposite, tiny tensors and one step per frame, so as torch ops it is
pure launch overhead, about 10 kernels per frame and two sweeps. It runs here as one kernel per sweep with
one program per sequence instead, the frames looped inside the kernel and the prefixes held as a block.

Nothing in here materializes a normalized ``[cells, vocab]`` tensor, which is what makes the loss fit: at
the production budget that tensor alone is several GB.
"""

from __future__ import annotations

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:
    _NEG_INF = tl.constexpr(-3.4028234663852886e38)

    @triton.jit
    def _cell_stats_kernel(
        logits_ptr,
        label_ptr,
        lse_ptr,
        blank_lp_ptr,
        label_lp_ptr,
        vocab,
        blank,
        BLOCK: tl.constexpr,
    ):
        """one program per cell, reduces its row to the normalizer and the two log probabilities"""
        cell = tl.program_id(0)
        base = cell.to(tl.int64) * vocab
        # the running maximum starts at the sentinel, a block of minus infinity would give inf minus inf
        running_max = _NEG_INF
        running_sum = 0.0
        for start in range(0, vocab, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < vocab
            x = tl.load(logits_ptr + base + offs, mask=mask, other=_NEG_INF).to(tl.float32)
            block_max = tl.max(x, axis=0)
            new_max = tl.maximum(running_max, block_max)
            running_sum = running_sum * tl.exp(running_max - new_max) + tl.sum(
                tl.where(mask, tl.exp(x - new_max), 0.0), axis=0
            )
            running_max = new_max
        lse = running_max + tl.log(running_sum)
        label = tl.load(label_ptr + cell)
        blank_logit = tl.load(logits_ptr + base + blank)
        label_logit = tl.load(logits_ptr + base + label)
        tl.store(lse_ptr + cell, lse)
        tl.store(blank_lp_ptr + cell, blank_logit - lse)
        tl.store(label_lp_ptr + cell, label_logit - lse)

    @triton.jit
    def _cell_grad_kernel(
        logits_ptr,
        label_ptr,
        lse_ptr,
        blank_grad_ptr,
        label_grad_ptr,
        out_ptr,
        vocab,
        blank,
        BLOCK: tl.constexpr,
    ):
        """one program per cell, writes the gradient of its row from the two edge posteriors"""
        cell = tl.program_id(0)
        base = cell.to(tl.int64) * vocab
        lse = tl.load(lse_ptr + cell)
        blank_grad = tl.load(blank_grad_ptr + cell)
        label_grad = tl.load(label_grad_ptr + cell)
        label = tl.load(label_ptr + cell)
        total = blank_grad + label_grad
        for start in range(0, vocab, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < vocab
            x = tl.load(logits_ptr + base + offs, mask=mask, other=float("-inf")).to(tl.float32)
            # a cell carrying no posterior contributes nothing, and its row may not even be normalizable
            grad = tl.where(total == 0.0, 0.0, total * tl.exp(x - lse))
            grad = tl.where(offs == blank, grad - blank_grad, grad)
            grad = tl.where(offs == label, grad - label_grad, grad)
            tl.store(out_ptr + base + offs, grad, mask=mask)


if triton is not None:

    @triton.jit
    def _forward_scan_kernel(
        blank_lp_ptr,
        label_lp_ptr,
        offsets_ptr,
        frame_lens_ptr,
        label_lens_ptr,
        alpha_ptr,
        total_ptr,
        num_cells,
        frame_stride,
        max_frames,
        max_prefix,
        BLOCK: tl.constexpr,
    ):
        """one program per sequence, sweeps the frames forward and writes the score of every lattice position"""
        seq = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        in_block = offs < max_prefix
        label_len = tl.load(label_lens_ptr + seq).to(tl.int64)
        frame_len = tl.load(frame_lens_ptr + seq).to(tl.int64)
        base = tl.load(offsets_ptr + seq).to(tl.int64)
        stride = label_len + 1
        valid = in_block & (offs <= label_len)
        row = alpha_ptr + seq.to(tl.int64) * max_prefix + offs

        tl.store(row, tl.where(offs == 0, 0.0, _NEG_INF), mask=in_block)
        for frame in range(0, max_frames):
            tl.debug_barrier()
            here = row + frame * frame_stride
            alpha = tl.load(here, mask=in_block, other=_NEG_INF)
            alpha_prev = tl.load(here - 1, mask=in_block & (offs >= 1), other=_NEG_INF)
            cell = base + frame * stride + offs
            blank_lp = tl.load(blank_lp_ptr + tl.minimum(cell, num_cells - 1), mask=valid, other=_NEG_INF)
            label_prev = tl.load(
                label_lp_ptr + tl.maximum(tl.minimum(cell - 1, num_cells - 1), 0),
                mask=valid & (offs >= 1),
                other=_NEG_INF,
            )
            stay = alpha + blank_lp
            emit = alpha_prev + label_prev
            # the floor keeps two impossible edges at minus infinity instead of taking inf minus inf
            top = tl.maximum(tl.maximum(stay, emit), _NEG_INF)
            updated = tl.where(valid, top + tl.log(tl.exp(stay - top) + tl.exp(emit - top)), _NEG_INF)
            tl.store(here + frame_stride, tl.where(frame < frame_len, updated, alpha), mask=in_block)

        tl.debug_barrier()
        final = tl.load(row + max_frames * frame_stride, mask=in_block, other=0.0)
        tl.store(total_ptr + seq, tl.sum(tl.where(offs == label_len, final, 0.0), axis=0))

    @triton.jit
    def _backward_scan_kernel(
        blank_lp_ptr,
        label_lp_ptr,
        offsets_ptr,
        frame_lens_ptr,
        label_lens_ptr,
        alpha_ptr,
        total_ptr,
        weight_ptr,
        beta_ptr,
        blank_grad_ptr,
        label_grad_ptr,
        num_cells,
        frame_stride,
        max_frames,
        max_prefix,
        BLOCK: tl.constexpr,
    ):
        """one program per sequence, sweeps the frames backward and scatters both edge posteriors onto the cells"""
        seq = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        in_block = offs < max_prefix
        label_len = tl.load(label_lens_ptr + seq).to(tl.int64)
        frame_len = tl.load(frame_lens_ptr + seq).to(tl.int64)
        base = tl.load(offsets_ptr + seq).to(tl.int64)
        norm = tl.load(total_ptr + seq)
        weight = tl.load(weight_ptr + seq)
        stride = label_len + 1
        valid = in_block & (offs <= label_len)
        start = tl.where(offs == label_len, 0.0, _NEG_INF)
        seq_row = seq.to(tl.int64) * max_prefix + offs
        row = beta_ptr + seq_row

        tl.store(row, start, mask=in_block)
        for back in range(0, max_frames):
            frame = max_frames - 1 - back
            tl.debug_barrier()
            beta = tl.load(row, mask=in_block, other=_NEG_INF)
            ahead = tl.load(row + 1, mask=in_block & (offs + 1 < max_prefix), other=_NEG_INF)
            cell = tl.minimum(base + frame * stride + offs, num_cells - 1)
            blank_lp = tl.load(blank_lp_ptr + cell, mask=valid, other=_NEG_INF)
            label_lp = tl.load(label_lp_ptr + cell, mask=valid, other=_NEG_INF)
            alpha = tl.load(alpha_ptr + seq_row + frame * frame_stride, mask=in_block, other=_NEG_INF)
            # a sequence without any alignment has its normalizer at the sentinel, where the posterior of
            # every cell would come out as exp(0), so its cells get no gradient at all
            inside = valid & (frame < frame_len) & (norm > _NEG_INF)
            blank_post = tl.where(inside, tl.exp(alpha + blank_lp + beta - norm) * weight, 0.0)
            label_post = tl.where(inside & (offs < label_len), tl.exp(alpha + label_lp + ahead - norm) * weight, 0.0)
            tl.store(blank_grad_ptr + cell, blank_post, mask=inside)
            tl.store(label_grad_ptr + cell, label_post, mask=inside)
            stay = blank_lp + beta
            emit = label_lp + ahead
            top = tl.maximum(tl.maximum(stay, emit), _NEG_INF)
            updated = tl.where(valid, top + tl.log(tl.exp(stay - top) + tl.exp(emit - top)), _NEG_INF)
            tl.debug_barrier()
            tl.store(row, tl.where(frame >= frame_len, start, updated), mask=in_block)


def _block_for(max_prefix: int) -> int:
    """
    :param max_prefix: prefixes the kernel holds as one block
    :return: the next power of two, at least 16
    """
    block = 16
    while block < max_prefix:
        block *= 2
    return block


def forward_scan(
    blank_lp: torch.Tensor,
    label_lp: torch.Tensor,
    offsets: torch.Tensor,
    frame_lens: torch.Tensor,
    label_lens: torch.Tensor,
    max_frames: int,
    max_prefix: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param blank_lp: [cells] blank log probability
    :param label_lp: [cells] next label log probability
    :param offsets: [B] first cell of every sequence
    :param frame_lens: [B] frames per sequence
    :param label_lens: [B] labels per sequence
    :param max_frames: frames to sweep, a static bound
    :param max_prefix: prefixes to sweep, U_max + 1
    :return: (total [B] log likelihood, alpha [max_frames + 1, B, max_prefix])
    """
    assert blank_lp.is_cuda and triton is not None, "monotonic rnnt: the scan kernels need cuda and triton"
    batch_size = frame_lens.shape[0]
    alpha = torch.empty((max_frames + 1, batch_size, max_prefix), dtype=torch.float32, device=blank_lp.device)
    total = torch.empty((batch_size,), dtype=torch.float32, device=blank_lp.device)
    _forward_scan_kernel[(batch_size,)](
        blank_lp,
        label_lp,
        offsets,
        frame_lens,
        label_lens,
        alpha,
        total,
        blank_lp.shape[0],
        batch_size * max_prefix,
        max_frames,
        max_prefix,
        BLOCK=_block_for(max_prefix),
    )
    return total, alpha


def backward_scan(
    blank_lp: torch.Tensor,
    label_lp: torch.Tensor,
    offsets: torch.Tensor,
    frame_lens: torch.Tensor,
    label_lens: torch.Tensor,
    alpha: torch.Tensor,
    total: torch.Tensor,
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param blank_lp: [cells] blank log probability
    :param label_lp: [cells] next label log probability
    :param offsets: [B] first cell of every sequence
    :param frame_lens: [B] frames per sequence
    :param label_lens: [B] labels per sequence
    :param alpha: [max_frames + 1, B, max_prefix] from :func:`forward_scan`
    :param total: [B] log likelihood
    :param weight: [B] incoming gradient of the log likelihood
    :return: (blank gradient [cells], label gradient [cells]) wrt the two log probabilities
    """
    assert blank_lp.is_cuda and triton is not None, "monotonic rnnt: the scan kernels need cuda and triton"
    frames_plus_one, batch_size, max_prefix = alpha.shape
    beta = torch.empty((batch_size, max_prefix), dtype=torch.float32, device=blank_lp.device)
    blank_grad = torch.zeros_like(blank_lp)
    label_grad = torch.zeros_like(label_lp)
    _backward_scan_kernel[(batch_size,)](
        blank_lp,
        label_lp,
        offsets,
        frame_lens,
        label_lens,
        alpha,
        total,
        weight,
        beta,
        blank_grad,
        label_grad,
        blank_lp.shape[0],
        batch_size * max_prefix,
        frames_plus_one - 1,
        max_prefix,
        BLOCK=_block_for(max_prefix),
    )
    return blank_grad, label_grad


def cell_stats(logits: torch.Tensor, next_label: torch.Tensor, blank: int) -> Tuple[torch.Tensor, ...]:
    """
    :param logits: [cells, vocab] unnormalized
    :param next_label: [cells] the label the emitting edge of every cell carries
    :param blank: blank index
    :return: (log normalizer, blank log prob, label log prob), each [cells]
    """
    assert logits.is_cuda and triton is not None, "monotonic rnnt: the cell kernels need cuda and triton"
    assert logits.is_contiguous(), "monotonic rnnt: the cell kernels address the logits row by row"
    cells, vocab = logits.shape
    out = [torch.empty(cells, dtype=torch.float32, device=logits.device) for _ in range(3)]
    _cell_stats_kernel[(cells,)](logits, next_label, out[0], out[1], out[2], vocab, blank, BLOCK=1024, num_warps=8)
    return tuple(out)


def cell_grad(
    logits: torch.Tensor,
    next_label: torch.Tensor,
    lse: torch.Tensor,
    blank_grad: torch.Tensor,
    label_grad: torch.Tensor,
    blank: int,
) -> torch.Tensor:
    """
    :param logits: [cells, vocab] unnormalized
    :param next_label: [cells] the label the emitting edge of every cell carries
    :param lse: [cells] log normalizer from :func:`cell_stats`
    :param blank_grad: [cells] gradient of the loss wrt the blank log probability
    :param label_grad: [cells] gradient of the loss wrt the label log probability
    :param blank: blank index
    :return: [cells, vocab] gradient wrt the logits
    """
    assert logits.is_cuda and triton is not None, "monotonic rnnt: the cell kernels need cuda and triton"
    cells, vocab = logits.shape
    # the gradient goes back to logits, so it is written in their dtype and the store converts it
    out = torch.empty_like(logits)
    _cell_grad_kernel[(cells,)](
        logits, next_label, lse, blank_grad, label_grad, out, vocab, blank, BLOCK=1024, num_warps=8
    )
    return out
