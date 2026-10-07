"""
RNN-T full-sum loss over a packed lattice.

A blank moves on to the next frame and a label stays in its frame, so a path through the (frames, prefixes)
lattice may emit several labels in one frame, and every sequence with at least one frame is alignable.
A frame is whatever the lattice has as its rows, the encoder frames or, for a joint that reads a whole
chunk of them at once, the chunks.

The lattice is packed exactly like the one of the monotonic loss (:mod:`returnn.torch.util.monotonic_rnnt`),
per sequence the frame index outer and the prefix index inner, so the layout helpers are shared.
Lengths stay on the device, nothing here reads them on the host, so the loss traces and captures.

A cell depends on the cell above it (blank) and on its left neighbour in the same frame (label).
The monotonic recursion can take a whole frame at once, since there a cell only depends on the frame before.
Here the cells that are independent of each other are those of one anti-diagonal (frame plus prefix constant),
so both sweeps run over the anti-diagonals.

On cuda everything runs as Triton kernels, the per-cell reductions over the vocabulary of the monotonic loss
(:mod:`returnn.torch.util.monotonic_rnnt_triton`) and the two sweeps (:mod:`returnn.triton.rnnt`, launched in
:mod:`returnn.torch.util.rnnt_triton`), behind one opaque custom op pair, so ``aot_function`` traces it and
nothing unrolls the loop over the anti-diagonals into the compiled graph.

Elsewhere the forward recursion runs as torch ops on a normalized tensor and autograd differentiates it.
That path is the reference the kernels are tested against, and since it reaches the gradient by a
different route it also checks the hand-written backward sweep.
"""

from __future__ import annotations

from typing import Tuple

import torch

from .custom_op import custom_op
from .monotonic_rnnt import cell_offsets, next_label_per_cell


def _diagonal_cells(
    offsets: torch.Tensor,
    frame_lens: torch.Tensor,
    label_lens: torch.Tensor,
    max_frames: int,
    max_prefix: int,
    num_cells: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param offsets: [B] first cell of every sequence
    :param frame_lens: [B] frames per sequence
    :param label_lens: [B] labels per sequence
    :param max_frames: frames to run over
    :param max_prefix: prefixes to run over, U_max + 1
    :param num_cells: rows of the packed buffer, positions outside a lattice clamp into it
    :return: (cell [D, B, P] the packed cell of every prefix on every anti-diagonal,
        inside [D, B, P] whether that position is in the lattice of its sequence),
        with D = max_frames + max_prefix - 1 anti-diagonals
    """
    device = offsets.device
    diagonal = torch.arange(max_frames + max_prefix - 1, device=device).view(-1, 1, 1)
    prefix = torch.arange(max_prefix, device=device).view(1, 1, -1)
    frame = diagonal - prefix
    lens = label_lens.long().view(1, -1, 1)
    inside = (frame >= 0) & (frame < frame_lens.long().view(1, -1, 1)) & (prefix <= lens)
    cell = offsets.view(1, -1, 1) + frame * (lens + 1) + prefix
    return torch.clamp(cell, min=0, max=num_cells - 1), inside


def _forward_scores(
    blank_lp: torch.Tensor,
    label_lp: torch.Tensor,
    frame_lens: torch.Tensor,
    label_lens: torch.Tensor,
    max_frames: int,
    max_prefix: int,
) -> torch.Tensor:
    """
    Runs the forward recursion over the anti-diagonals of the packed lattice.

    :param blank_lp: [cells] blank log probability
    :param label_lp: [cells] next label log probability
    :param frame_lens: [B] frames per sequence
    :param label_lens: [B] labels per sequence
    :param max_frames: frames to run over
    :param max_prefix: prefixes to run over, U_max + 1
    :return: total [B] log likelihood, the sentinel for a sequence without frames
    """
    batch_size = frame_lens.shape[0]
    device, dtype = blank_lp.device, blank_lp.dtype
    neg_inf = torch.finfo(dtype).min
    offsets, _cells = cell_offsets(frame_lens, label_lens)
    cell, inside = _diagonal_cells(offsets, frame_lens, label_lens, max_frames, max_prefix, blank_lp.shape[0])
    # outside the lattice of a sequence the index lands on cells that are not its own, so those are masked
    blank_rows = blank_lp[cell].masked_fill(~inside, neg_inf)
    label_rows = label_lp[cell].masked_fill(~inside, neg_inf)
    lens = label_lens.long().unsqueeze(1)
    # the last cell of a sequence, its last frame and its whole prefix, sits on this anti-diagonal
    last = frame_lens.long() - 1 + label_lens.long()
    has_frames = frame_lens > 0
    pad = torch.full((batch_size, 1), neg_inf, dtype=dtype, device=device)
    start = torch.full((batch_size, max_prefix), neg_inf, dtype=dtype, device=device)
    start[:, 0] = 0.0

    total = torch.full((batch_size,), neg_inf, dtype=dtype, device=device)
    alpha = start
    for diagonal in range(cell.shape[0]):
        if diagonal > 0:
            # both predecessors sit on the anti-diagonal before, the cell above at the same prefix, left through
            # its blank, and the left neighbour at the prefix before, left through its label,
            # and the floor keeps a dead edge finite, logaddexp of two minus infinities has a nan gradient
            above = (alpha + blank_rows[diagonal - 1]).clamp(min=neg_inf)
            left = (alpha + label_rows[diagonal - 1]).clamp(min=neg_inf)
            alpha = torch.logaddexp(above, torch.cat([pad, left[:, :-1]], dim=1))
        alpha = torch.where(inside[diagonal], alpha, torch.full_like(alpha, neg_inf))
        leaving = alpha + blank_rows[diagonal]
        total = torch.where(has_frames & (last == diagonal), torch.gather(leaving, 1, lens).squeeze(1), total)
    return total


_HAVE_LIB_OPS = False
if hasattr(torch.library, "custom_op"):  # torch >= 2.4

    @custom_op("returnn::rnnt_fwd", mutates_args=())
    def _lib_fwd(
        logits: torch.Tensor,
        next_label: torch.Tensor,
        frame_lens: torch.Tensor,
        label_lens: torch.Tensor,
        blank: int,
        max_frames: int,
        max_prefix: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from .monotonic_rnnt_triton import cell_stats
        from .rnnt_triton import forward_scan

        offsets, _cells = cell_offsets(frame_lens, label_lens)
        row_max, log_sum, blank_lp, label_lp = cell_stats(logits, next_label, blank)
        total, alpha = forward_scan(blank_lp, label_lp, offsets, frame_lens, label_lens, max_frames, max_prefix)
        return total, row_max, log_sum, blank_lp, label_lp, alpha

    @_lib_fwd.register_fake
    def _lib_fwd_fake(logits, next_label, frame_lens, label_lens, blank, max_frames, max_prefix):
        del next_label, label_lens, blank, max_frames, max_prefix
        batch_size, cells = frame_lens.shape[0], logits.shape[0]
        cell_vec = logits.new_empty((cells,), dtype=torch.float32)
        return (
            logits.new_empty((batch_size,), dtype=torch.float32),
            cell_vec,
            torch.empty_like(cell_vec),
            torch.empty_like(cell_vec),
            torch.empty_like(cell_vec),
            torch.empty_like(cell_vec),
        )

    @custom_op("returnn::rnnt_bwd", mutates_args=())
    def _lib_bwd(
        logits: torch.Tensor,
        next_label: torch.Tensor,
        row_max: torch.Tensor,
        log_sum: torch.Tensor,
        blank_lp: torch.Tensor,
        label_lp: torch.Tensor,
        alpha: torch.Tensor,
        frame_lens: torch.Tensor,
        label_lens: torch.Tensor,
        d_total: torch.Tensor,
        blank: int,
        max_frames: int,
        max_prefix: int,
    ) -> torch.Tensor:
        from .monotonic_rnnt_triton import cell_grad
        from .rnnt_triton import backward_scan

        offsets, _cells = cell_offsets(frame_lens, label_lens)
        # the sweep gives the posteriors of the log likelihood, d_total carries the sign of the loss
        blank_grad, label_grad = backward_scan(
            blank_lp, label_lp, offsets, frame_lens, label_lens, alpha, -d_total, max_frames, max_prefix
        )
        return cell_grad(logits, next_label, row_max, log_sum, blank_grad, label_grad, blank)

    @_lib_bwd.register_fake
    def _lib_bwd_fake(
        logits,
        next_label,
        row_max,
        log_sum,
        blank_lp,
        label_lp,
        alpha,
        frame_lens,
        label_lens,
        d_total,
        blank,
        max_frames,
        max_prefix,
    ):
        del next_label, row_max, log_sum, blank_lp, label_lp, alpha, frame_lens, label_lens, d_total, blank
        del max_frames, max_prefix
        return torch.empty_like(logits)

    def _lib_setup_context(ctx, inputs, output):
        logits, next_label, frame_lens, label_lens, blank, max_frames, max_prefix = inputs
        _total, row_max, log_sum, blank_lp, label_lp, alpha = output
        ctx.save_for_backward(logits, next_label, row_max, log_sum, blank_lp, label_lp, alpha, frame_lens, label_lens)
        ctx.blank = blank
        ctx.max_frames = max_frames
        ctx.max_prefix = max_prefix

    def _lib_backward(ctx, d_total, d_row_max, d_log_sum, d_blank_lp, d_label_lp, d_alpha):
        del d_row_max, d_log_sum, d_blank_lp, d_label_lp, d_alpha  # unused, only total feeds the loss
        logits, next_label, row_max, log_sum, blank_lp, label_lp, alpha, frame_lens, label_lens = ctx.saved_tensors
        grad_logits = torch.ops.returnn.rnnt_bwd(
            logits,
            next_label,
            row_max,
            log_sum,
            blank_lp,
            label_lp,
            alpha,
            frame_lens,
            label_lens,
            d_total,
            ctx.blank,
            ctx.max_frames,
            ctx.max_prefix,
        )
        return grad_logits, None, None, None, None, None, None

    torch.library.register_autograd("returnn::rnnt_fwd", _lib_backward, setup_context=_lib_setup_context)
    _HAVE_LIB_OPS = True


def rnnt_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    frame_lens: torch.Tensor,
    label_lens: torch.Tensor,
    *,
    blank: int,
    max_frames: int,
) -> torch.Tensor:
    """
    Full-sum negative log likelihood of the transducer over a packed lattice.

    :param logits: [cells, V] unnormalized, per sequence the frame index outer and the prefix inner
    :param labels: [B, U_max] the reference labels, padded
    :param frame_lens: [B] frames per sequence
    :param label_lens: [B] labels per sequence, any number of them for a sequence with a frame
    :param blank: blank index
    :param max_frames: frames the recursion runs over, at least the longest sequence of the batch.
        A static bound such as the declared capacity, since reading the batch's own maximum would be a host read.
    :return: [B] the negative log likelihood, zero where a sequence has no frame
    """
    if logits.dim() != 2:
        raise ValueError(f"rnnt: logits must be [cells, V], got shape {tuple(logits.shape)}")
    if not 0 <= blank < logits.shape[1]:
        raise ValueError(f"rnnt: blank {blank} outside the vocabulary of {logits.shape[1]}")
    if logits.shape[0] == 0:
        return logits.sum() * torch.zeros(frame_lens.shape[0], dtype=torch.float32, device=logits.device)
    logits = logits.contiguous()
    # the kernels index the lengths by sequence and ignore strides
    frame_lens, label_lens = frame_lens.contiguous(), label_lens.contiguous()
    max_prefix = int(labels.shape[1]) + 1
    next_label = next_label_per_cell(labels, frame_lens, label_lens, blank, logits.shape[0])
    if logits.is_cuda:
        assert _HAVE_LIB_OPS, "rnnt: the loss needs torch.library.custom_op, so torch >= 2.4"
        total = torch.ops.returnn.rnnt_fwd(logits, next_label, frame_lens, label_lens, blank, max_frames, max_prefix)[0]
    else:
        source = logits if logits.dtype in (torch.float32, torch.float64) else logits.float()
        # the rows a capacity leaves past the cells of the batch belong to no sequence, whatever they hold
        _offsets, cells = cell_offsets(frame_lens, label_lens)
        unused = (torch.arange(logits.shape[0], device=logits.device) >= cells.sum()).unsqueeze(1)
        # a row without any finite logit has no probability mass, its log probabilities are minus infinity, not nan
        dead = torch.isneginf(source).all(dim=-1, keepdim=True)
        log_probs = torch.log_softmax(source.masked_fill(dead | unused, 0.0), dim=-1).masked_fill(dead, float("-inf"))
        blank_lp = log_probs[:, blank]
        label_lp = torch.gather(log_probs, 1, next_label.unsqueeze(1)).squeeze(1)
        total = _forward_scores(blank_lp, label_lp, frame_lens, label_lens, max_frames, max_prefix)
    return torch.where(frame_lens > 0, -total, torch.zeros_like(total))
