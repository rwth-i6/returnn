"""
Monotonic RNN-T full-sum loss over a packed lattice.

Every alignment spends exactly one frame per lattice step, so a path through the (frames, prefixes)
lattice either stays on its prefix (blank) or advances it (the next reference label), and a sequence
is alignable only while its label count does not exceed its frame count.

The lattice is packed: the activations of all sequences sit in one axis, per sequence the frame index
outer and the prefix index inner, which is the layout ``i6_native_ops.monotonic_rnnt`` takes as well.
Lengths stay on the device, nothing here reads them on the host, so the loss traces and captures.

On cuda everything runs as Triton kernels (:mod:`returnn.triton.monotonic_rnnt`, launched in
:mod:`returnn.torch.util.monotonic_rnnt_triton`), the per-cell
reductions over the vocabulary and both sweeps of the forward-backward recursion, and no normalized
``[cells, vocab]`` tensor is ever materialized. The whole loss sits behind one opaque custom op pair, so
``aot_function`` traces it and nothing unrolls the frame loop into the compiled graph.

Elsewhere the forward recursion runs as torch ops on a normalized tensor and autograd differentiates it.
That path is the reference the kernels are tested against, and since it reaches the gradient by a
different route it also checks the hand-written backward sweep.
"""

from __future__ import annotations

from typing import Tuple

import torch

from .custom_op import custom_op


def _cell_offsets(frame_lens: torch.Tensor, label_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param frame_lens: [B] frames per sequence
    :param label_lens: [B] labels per sequence
    :return: (offsets [B] of the first cell of every sequence, cells [B] per sequence)
    """
    cells = frame_lens.long() * (label_lens.long() + 1)
    offsets = torch.cumsum(cells, dim=0) - cells
    return offsets, cells


def _lattice_index(
    frame_lens: torch.Tensor, label_lens: torch.Tensor, total: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The sequence, frame and prefix every packed lattice cell belongs to.

    Cells past the batch's own sum, which a static capacity leaves over, land on the last sequence with
    a frame index past its end. The loss masks them, a caller that indexes a padded buffer with them has
    to clamp the frame to that buffer instead.

    :param frame_lens: [B] frames per sequence
    :param label_lens: [B] labels per sequence
    :param total: cells to lay out, the sum over the batch or a static capacity above it
    :return: (sequence [total], frame [total], prefix [total]) int64
    """
    offsets, cells = _cell_offsets(frame_lens, label_lens)
    # the cells a capacity leaves over go to the last sequence, and repeat_interleave with a declared
    # output size stays static, unlike searchsorted, which Inductor only takes as an extern fallback
    spans = torch.cat([cells[:-1], (cells[-1] + total - cells.sum()).unsqueeze(0)])
    seq = torch.repeat_interleave(torch.arange(frame_lens.shape[0], device=frame_lens.device), spans, output_size=total)
    stride = (label_lens.long() + 1)[seq]
    within = torch.arange(total, device=frame_lens.device) - offsets[seq]
    return seq, torch.div(within, stride, rounding_mode="floor"), within % stride


def _next_label_per_cell(
    labels: torch.Tensor, frame_lens: torch.Tensor, label_lens: torch.Tensor, blank: int, total: int
) -> torch.Tensor:
    """
    The label every cell's emitting edge carries, blank where the prefix is already complete.

    :param labels: [B, U_max] the reference labels, padded
    :param frame_lens: [B] frames per sequence
    :param label_lens: [B] labels per sequence
    :param blank: blank index, used where a cell has no emitting edge
    :param total: cells to lay out, see :func:`_lattice_index`
    :return: [total] int64
    """
    if labels.shape[1] == 0:
        return torch.full((total,), blank, dtype=torch.int64, device=frame_lens.device)
    seq, _frame, prefix = _lattice_index(frame_lens, label_lens, total)
    lens = label_lens.long()[seq]
    index = torch.clamp(prefix, max=torch.clamp(lens - 1, min=0))
    label = labels.long()[seq, index]
    return torch.where(prefix < lens, label, torch.full_like(label, blank))


def _cell_index(
    offsets: torch.Tensor, label_lens: torch.Tensor, max_frames: int, max_prefix: int, num_cells: int
) -> torch.Tensor:
    """
    :param offsets: [B] first cell of every sequence
    :param label_lens: [B] labels per sequence
    :param max_frames: frames to run over
    :param max_prefix: prefixes to run over, U_max + 1
    :param num_cells: rows of the packed buffer, positions past a sequence clamp into it
    :return: [max_frames, B, max_prefix] the packed cell of every lattice position
    """
    device = offsets.device
    frame = torch.arange(max_frames, device=device).view(-1, 1, 1)
    prefix = torch.arange(max_prefix, device=device).view(1, 1, -1)
    stride = (label_lens.long() + 1).view(1, -1, 1)
    return torch.clamp(offsets.view(1, -1, 1) + frame * stride + prefix, max=num_cells - 1)


def _forward_scores(
    blank_rows: torch.Tensor,
    label_rows: torch.Tensor,
    frame_lens: torch.Tensor,
    label_lens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Runs the forward recursion over the packed lattice, laid out as one row per frame.

    :param blank_rows: [T, B, P] blank log probability
    :param label_rows: [T, B, P] next label log probability
    :param frame_lens: [B] frames per sequence
    :param label_lens: [B] labels per sequence
    :return: (total [B] log likelihood, alpha [T, B, P])
    """
    max_frames, batch_size, max_prefix = blank_rows.shape
    device, dtype = blank_rows.device, blank_rows.dtype
    neg_inf = torch.finfo(dtype).min
    prefix = torch.arange(max_prefix, device=device).unsqueeze(0)
    lens = label_lens.long().unsqueeze(1)
    valid = prefix <= lens
    pad = torch.full((batch_size, 1), neg_inf, dtype=dtype, device=device)
    # outside the lattice of a sequence the index lands on cells that are not its own, so those are masked
    frames = torch.arange(max_frames, device=device).view(-1, 1, 1)
    outside = ~(valid.unsqueeze(0) & (frames < frame_lens.view(1, -1, 1)))
    blank_rows = blank_rows.masked_fill(outside, neg_inf)
    label_rows = label_rows.masked_fill(outside, neg_inf)

    alpha_frames = []
    alpha = torch.full((batch_size, max_prefix), neg_inf, dtype=dtype, device=device)
    alpha[:, 0] = 0.0
    total = torch.full((batch_size,), neg_inf, dtype=dtype, device=device)
    for frame in range(max_frames):
        alpha_frames.append(alpha)
        emit = alpha + label_rows[frame]
        # the floor keeps a dead edge finite, logaddexp of two minus infinities has a nan gradient
        stay = (alpha + blank_rows[frame]).clamp(min=neg_inf)
        move = torch.cat([pad, emit[:, :-1]], dim=1).clamp(min=neg_inf)
        updated = torch.logaddexp(stay, move)
        updated = torch.where(valid, updated, torch.full_like(updated, neg_inf))
        alpha = torch.where((frame < frame_lens).unsqueeze(1), updated, alpha)
        total = torch.where((frame + 1) == frame_lens, torch.gather(alpha, 1, lens).squeeze(1), total)
    return total, torch.stack(alpha_frames)


_HAVE_LIB_OPS = False
if hasattr(torch.library, "custom_op"):  # torch >= 2.4

    @custom_op("returnn::monotonic_rnnt_fwd", mutates_args=())
    def _lib_fwd(
        logits: torch.Tensor,
        next_label: torch.Tensor,
        frame_lens: torch.Tensor,
        label_lens: torch.Tensor,
        blank: int,
        max_frames: int,
        max_prefix: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from .monotonic_rnnt_triton import cell_stats, forward_scan

        offsets, _cells = _cell_offsets(frame_lens, label_lens)
        row_max, log_sum, blank_lp, label_lp = cell_stats(logits, next_label, blank)
        total, alpha = forward_scan(blank_lp, label_lp, offsets, frame_lens, label_lens, max_frames, max_prefix)
        return total, row_max, log_sum, blank_lp, label_lp, alpha

    @_lib_fwd.register_fake
    def _lib_fwd_fake(logits, next_label, frame_lens, label_lens, blank, max_frames, max_prefix):
        del next_label, label_lens, blank
        batch_size, cells = frame_lens.shape[0], logits.shape[0]
        cell_vec = logits.new_empty((cells,), dtype=torch.float32)
        return (
            logits.new_empty((batch_size,), dtype=torch.float32),
            cell_vec,
            torch.empty_like(cell_vec),
            torch.empty_like(cell_vec),
            torch.empty_like(cell_vec),
            logits.new_empty((max_frames + 1, batch_size, max_prefix), dtype=torch.float32),
        )

    @custom_op("returnn::monotonic_rnnt_bwd", mutates_args=())
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
    ) -> torch.Tensor:
        from .monotonic_rnnt_triton import backward_scan, cell_grad

        offsets, _cells = _cell_offsets(frame_lens, label_lens)
        # the sweep gives the posteriors of the log likelihood, d_total carries the sign of the loss
        blank_grad, label_grad = backward_scan(blank_lp, label_lp, offsets, frame_lens, label_lens, alpha, -d_total)
        return cell_grad(logits, next_label, row_max, log_sum, blank_grad, label_grad, blank)

    @_lib_bwd.register_fake
    def _lib_bwd_fake(
        logits, next_label, row_max, log_sum, blank_lp, label_lp, alpha, frame_lens, label_lens, d_total, blank
    ):
        del next_label, row_max, log_sum, blank_lp, label_lp, alpha, frame_lens, label_lens, d_total, blank
        return torch.empty_like(logits)

    def _lib_setup_context(ctx, inputs, output):
        logits, next_label, frame_lens, label_lens, blank, _max_frames, _max_prefix = inputs
        _total, row_max, log_sum, blank_lp, label_lp, alpha = output
        ctx.save_for_backward(logits, next_label, row_max, log_sum, blank_lp, label_lp, alpha, frame_lens, label_lens)
        ctx.blank = blank

    def _lib_backward(ctx, d_total, d_row_max, d_log_sum, d_blank_lp, d_label_lp, d_alpha):
        del d_row_max, d_log_sum, d_blank_lp, d_label_lp, d_alpha  # unused, only total feeds the loss
        logits, next_label, row_max, log_sum, blank_lp, label_lp, alpha, frame_lens, label_lens = ctx.saved_tensors
        grad_logits = torch.ops.returnn.monotonic_rnnt_bwd(
            logits, next_label, row_max, log_sum, blank_lp, label_lp, alpha, frame_lens, label_lens, d_total, ctx.blank
        )
        return grad_logits, None, None, None, None, None, None

    torch.library.register_autograd("returnn::monotonic_rnnt_fwd", _lib_backward, setup_context=_lib_setup_context)
    _HAVE_LIB_OPS = True


def monotonic_rnnt_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    frame_lens: torch.Tensor,
    label_lens: torch.Tensor,
    *,
    blank: int,
    max_frames: int,
) -> torch.Tensor:
    """
    Full-sum negative log likelihood of the monotonic transducer over a packed lattice.

    :param logits: [cells, V] unnormalized, per sequence the frame index outer and the prefix inner
    :param labels: [B, U_max] the reference labels, padded
    :param frame_lens: [B] frames per sequence
    :param label_lens: [B] labels per sequence, at most the frame count
    :param blank: blank index
    :param max_frames: frames the recursion runs over, at least the longest sequence of the batch.
        A static bound such as the declared capacity, since reading the batch's own maximum would be a host read.
    :return: [B] the negative log likelihood, zero (also in the gradient) where a sequence has no alignment,
        as ``zero_infinity`` does in :func:`torch.nn.functional.ctc_loss`
    """
    if logits.dim() != 2:
        raise ValueError(f"monotonic rnnt: logits must be [cells, V], got shape {tuple(logits.shape)}")
    if logits.shape[0] == 0:
        return logits.sum() * torch.zeros(frame_lens.shape[0], dtype=torch.float32, device=logits.device)
    logits = logits.contiguous()
    # the kernels index the lengths by sequence and ignore strides
    frame_lens, label_lens = frame_lens.contiguous(), label_lens.contiguous()
    max_prefix = int(labels.shape[1]) + 1
    next_label = _next_label_per_cell(labels, frame_lens, label_lens, blank, logits.shape[0])
    if logits.is_cuda:
        assert _HAVE_LIB_OPS, "monotonic rnnt: the loss needs torch.library.custom_op, so torch >= 2.4"
        total = torch.ops.returnn.monotonic_rnnt_fwd(
            logits, next_label, frame_lens, label_lens, blank, max_frames, max_prefix
        )[0]
    else:
        offsets, cells = _cell_offsets(frame_lens, label_lens)
        source = logits if logits.dtype in (torch.float32, torch.float64) else logits.float()
        # the rows a capacity leaves past the cells of the batch belong to no sequence, whatever they hold
        unused = (torch.arange(logits.shape[0], device=logits.device) >= cells.sum()).unsqueeze(1)
        # a row without any finite logit has no probability mass, its log probabilities are minus infinity, not nan
        dead = torch.isneginf(source).all(dim=-1, keepdim=True)
        log_probs = torch.log_softmax(source.masked_fill(dead | unused, 0.0), dim=-1).masked_fill(dead, float("-inf"))
        blank_lp = log_probs[:, blank]
        label_lp = torch.gather(log_probs, 1, next_label.unsqueeze(1)).squeeze(1)
        index = _cell_index(offsets, label_lens, max_frames, max_prefix, logits.shape[0])
        total, _alpha = _forward_scores(blank_lp[index], label_lp[index], frame_lens, label_lens)
    alignable = (label_lens <= frame_lens) & (frame_lens > 0)
    return torch.where(alignable, -total, torch.zeros_like(total))
