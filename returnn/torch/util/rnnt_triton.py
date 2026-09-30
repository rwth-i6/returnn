"""
Triton kernels for the two sweeps of the packed RNN-T loss, see :mod:`returnn.torch.util.rnnt`.

The reductions over the vocabulary are those of the monotonic loss and are taken from there
(:func:`returnn.torch.util.monotonic_rnnt_triton.cell_stats` and ``cell_grad``).

The sweeps differ. A label stays in its frame, so a cell depends on its left neighbour in the same frame
and no sweep can take a frame at once. The cells of one anti-diagonal (frame plus prefix constant) only
depend on the anti-diagonal before, so each sweep runs as one kernel with one program per sequence,
the anti-diagonals looped inside the kernel and the prefixes held as a block.

The scores are kept per packed cell. A dense buffer over (anti-diagonals, sequences, prefixes) would,
at the bounds of a captured step, be several hundred MB for a lattice of some ten thousand cells.
"""

from __future__ import annotations

from typing import Tuple

import torch

from .monotonic_rnnt_triton import _block_for

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    _NEG_INF = tl.constexpr(-3.4028234663852886e38)

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
        max_diagonals,
        max_prefix,
        BLOCK: tl.constexpr,
    ):
        """one program per sequence, sweeps the anti-diagonals forward and writes the score of every cell"""
        seq = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        label_len = tl.load(label_lens_ptr + seq).to(tl.int64)
        frame_len = tl.load(frame_lens_ptr + seq).to(tl.int64)
        base = tl.load(offsets_ptr + seq).to(tl.int64)
        stride = label_len + 1
        in_lattice = (offs < max_prefix) & (offs <= label_len)

        for diagonal in range(0, max_diagonals):
            tl.debug_barrier()
            frame = diagonal - offs
            inside = in_lattice & (frame >= 0) & (frame < frame_len)
            cell = tl.maximum(tl.minimum(base + frame * stride + offs, num_cells - 1), 0)
            # the cell above, which a blank leaves to this one
            from_above = inside & (frame >= 1)
            above = tl.maximum(cell - stride, 0)
            stay = tl.load(alpha_ptr + above, mask=from_above, other=_NEG_INF) + tl.load(
                blank_lp_ptr + above, mask=from_above, other=_NEG_INF
            )
            # the left neighbour in the same frame, which a label leaves to this one
            from_left = inside & (offs >= 1)
            left = tl.maximum(cell - 1, 0)
            emit = tl.load(alpha_ptr + left, mask=from_left, other=_NEG_INF) + tl.load(
                label_lp_ptr + left, mask=from_left, other=_NEG_INF
            )
            # the floor keeps two impossible edges at minus infinity instead of taking inf minus inf
            top = tl.maximum(tl.maximum(stay, emit), _NEG_INF)
            reached = top + tl.log(tl.exp(stay - top) + tl.exp(emit - top))
            # the first cell has no edge into it, every path starts there
            reached = tl.where((diagonal == 0) & (offs == 0), 0.0, reached)
            tl.store(alpha_ptr + cell, reached, mask=inside)

        tl.debug_barrier()
        # every path ends by the blank of the last cell, a slot without frames has no cell at all
        last = tl.maximum(tl.minimum(base + (frame_len - 1) * stride + label_len, num_cells - 1), 0)
        ends = tl.load(alpha_ptr + last) + tl.load(blank_lp_ptr + last)
        tl.store(total_ptr + seq, tl.where(frame_len > 0, ends, _NEG_INF))

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
        max_diagonals,
        max_prefix,
        BLOCK: tl.constexpr,
    ):
        """one program per sequence, sweeps the anti-diagonals backward and writes both edge posteriors of the cells"""
        seq = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        label_len = tl.load(label_lens_ptr + seq).to(tl.int64)
        frame_len = tl.load(frame_lens_ptr + seq).to(tl.int64)
        base = tl.load(offsets_ptr + seq).to(tl.int64)
        norm = tl.load(total_ptr + seq)
        weight = tl.load(weight_ptr + seq)
        stride = label_len + 1
        in_lattice = (offs < max_prefix) & (offs <= label_len)

        for back in range(0, max_diagonals):
            diagonal = max_diagonals - 1 - back
            tl.debug_barrier()
            frame = diagonal - offs
            inside = in_lattice & (frame >= 0) & (frame < frame_len)
            cell = tl.maximum(tl.minimum(base + frame * stride + offs, num_cells - 1), 0)
            blank_lp = tl.load(blank_lp_ptr + cell, mask=inside, other=_NEG_INF)
            label_lp = tl.load(label_lp_ptr + cell, mask=inside, other=_NEG_INF)
            alpha = tl.load(alpha_ptr + cell, mask=inside, other=_NEG_INF)
            # a blank leaves to the cell below, and from the last cell of the sequence out of the lattice
            to_below = inside & (frame + 1 < frame_len)
            below = tl.minimum(cell + stride, num_cells - 1)
            ahead = tl.load(beta_ptr + below, mask=to_below, other=_NEG_INF)
            ends = inside & (frame + 1 == frame_len) & (offs == label_len)
            ahead = tl.where(ends, 0.0, ahead)
            # a label leaves to the right neighbour in the same frame
            to_right = inside & (offs < label_len)
            right = tl.minimum(cell + 1, num_cells - 1)
            beside = tl.load(beta_ptr + right, mask=to_right, other=_NEG_INF)
            stay = blank_lp + ahead
            emit = label_lp + beside
            top = tl.maximum(tl.maximum(stay, emit), _NEG_INF)
            remaining = top + tl.log(tl.exp(stay - top) + tl.exp(emit - top))
            tl.store(beta_ptr + cell, remaining, mask=inside)
            # a slot without frames has its normalizer at the sentinel and no cell, so nothing is written for it
            scored = inside & (norm > _NEG_INF)
            blank_post = tl.where(scored & (to_below | ends), tl.exp(alpha + stay - norm) * weight, 0.0)
            label_post = tl.where(scored & to_right, tl.exp(alpha + emit - norm) * weight, 0.0)
            tl.store(blank_grad_ptr + cell, blank_post, mask=inside)
            tl.store(label_grad_ptr + cell, label_post, mask=inside)


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
    :return: (total [B] log likelihood, alpha [cells] the score of every cell)
    """
    assert blank_lp.is_cuda and triton is not None, "rnnt: the scan kernels need cuda and triton"
    assert frame_lens.is_contiguous() and label_lens.is_contiguous(), "rnnt: the kernels index the lengths by sequence"
    batch_size = frame_lens.shape[0]
    alpha = torch.empty_like(blank_lp)
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
        max_frames + max_prefix - 1,
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
    max_frames: int,
    max_prefix: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param blank_lp: [cells] blank log probability
    :param label_lp: [cells] next label log probability
    :param offsets: [B] first cell of every sequence
    :param frame_lens: [B] frames per sequence
    :param label_lens: [B] labels per sequence
    :param alpha: [cells] from :func:`forward_scan`
    :param total: [B] log likelihood
    :param weight: [B] incoming gradient of the log likelihood
    :param max_frames: frames to sweep, the bound the forward sweep ran with
    :param max_prefix: prefixes to sweep, U_max + 1
    :return: (blank gradient [cells], label gradient [cells]) wrt the two log probabilities
    """
    assert blank_lp.is_cuda and triton is not None, "rnnt: the scan kernels need cuda and triton"
    assert frame_lens.is_contiguous() and label_lens.is_contiguous(), "rnnt: the kernels index the lengths by sequence"
    batch_size = frame_lens.shape[0]
    beta = torch.empty_like(blank_lp)
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
        max_frames + max_prefix - 1,
        max_prefix,
        BLOCK=_block_for(max_prefix),
    )
    return blank_grad, label_grad
