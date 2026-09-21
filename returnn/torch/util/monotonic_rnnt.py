"""
Monotonic RNN-T full-sum loss over a packed lattice.

Every alignment spends exactly one frame per lattice step, so a path through the (frames, prefixes)
lattice either stays on its prefix (blank) or advances it (the next reference label), and a sequence
is alignable only while its label count does not exceed its frame count.

The lattice is packed: the activations of all sequences sit in one axis, per sequence the frame index
outer and the prefix index inner, which is the layout ``i6_native_ops.monotonic_rnnt`` takes as well.
Lengths stay on the device, nothing here reads them on the host, so the loss traces and captures.

On cuda everything runs as Triton kernels (:mod:`returnn.torch.util.monotonic_rnnt_triton`), the per-cell
reductions over the vocabulary and both sweeps of the forward-backward recursion, and no normalized
``[cells, vocab]`` tensor is ever materialized. The whole loss sits behind one opaque custom op pair, so
``aot_function`` traces it and nothing unrolls the frame loop into the compiled graph.

Elsewhere the forward recursion runs as torch ops on a normalized tensor and autograd differentiates it.
That path is the reference the kernels are tested against, and since it reaches the gradient by a
different route it also checks the hand-written backward sweep.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def cell_offsets(frame_lens: torch.Tensor, label_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param frame_lens: [B] frames per sequence
    :param label_lens: [B] labels per sequence
    :return: (offsets [B] of the first cell of every sequence, cells [B] per sequence)
    """
    cells = frame_lens.long() * (label_lens.long() + 1)
    offsets = torch.cumsum(cells, dim=0) - cells
    return offsets, cells


def reachable_cells(frame_lens: torch.Tensor, label_lens: torch.Tensor) -> torch.Tensor:
    """
    Cells that lie on at least one complete alignment, which is fewer than the full rectangle.

    A state (t, u) is reachable from the start only while u <= t and reaches the end only while the
    frames left cover the labels left, u >= U - T + t, so the lattice is a band and the cells outside
    it contribute nothing. Dropping them is exact, it prunes no alignment.

    :param frame_lens: [B] frames per sequence
    :param label_lens: [B] labels per sequence
    :return: [B] the number of cells inside the band
    """
    frames, labels = frame_lens.long(), label_lens.long()
    return torch.clamp(frames * (labels + 1) - labels * labels, min=0)


def lattice_index(
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
    offsets, cells = cell_offsets(frame_lens, label_lens)
    # the cells a capacity leaves over go to the last sequence, and repeat_interleave with a declared
    # output size stays static, unlike searchsorted, which Inductor only takes as an extern fallback
    spans = torch.cat([cells[:-1], (cells[-1] + total - cells.sum()).unsqueeze(0)])
    seq = torch.repeat_interleave(
        torch.arange(frame_lens.shape[0], device=frame_lens.device), spans, output_size=total
    )
    stride = (label_lens.long() + 1)[seq]
    within = torch.arange(total, device=frame_lens.device) - offsets[seq]
    return seq, torch.div(within, stride, rounding_mode="floor"), within % stride


def total_cells(frame_lens: torch.Tensor, label_lens: torch.Tensor) -> int:
    """
    :param frame_lens: [B] frames per sequence
    :param label_lens: [B] labels per sequence
    :return: the cells of the whole batch, a host read that a captured step replaces by its capacity
    """
    _offsets, cells = cell_offsets(frame_lens, label_lens)
    return int(cells.sum().item())


def lattice_operands(
    enc: torch.Tensor, pred: torch.Tensor, frame_lens: torch.Tensor, label_lens: torch.Tensor, total: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Picks the encoder frame and the predictor state that meet in every packed lattice cell.

    :param enc: [B, T, D_enc] padded encoder output
    :param pred: [B, U_max + 1, D_pred] padded predictor output
    :param frame_lens: [B] frames per sequence
    :param label_lens: [B] labels per sequence
    :param total: cells to lay out, see :func:`lattice_index`
    :return: ([total, D_enc], [total, D_pred])
    """
    seq, frame, prefix = lattice_index(frame_lens, label_lens, total)
    frame = torch.clamp(frame, max=enc.shape[1] - 1)
    prefix = torch.clamp(prefix, max=pred.shape[1] - 1)
    enc_cells = enc.reshape(-1, enc.shape[-1])[seq * enc.shape[1] + frame]
    pred_cells = pred.reshape(-1, pred.shape[-1])[seq * pred.shape[1] + prefix]
    return enc_cells, pred_cells


def next_label_per_cell(
    labels: torch.Tensor, frame_lens: torch.Tensor, label_lens: torch.Tensor, blank: int, total: int
) -> torch.Tensor:
    """
    The label every cell's emitting edge carries, blank where the prefix is already complete.

    :param labels: [B, U_max] the reference labels, padded
    :param frame_lens: [B] frames per sequence
    :param label_lens: [B] labels per sequence
    :param blank: blank index, used where a cell has no emitting edge
    :param total: cells to lay out, see :func:`lattice_index`
    :return: [total] int64
    """
    if labels.shape[1] == 0:
        return torch.full((total,), blank, dtype=torch.int64, device=frame_lens.device)
    seq, _frame, prefix = lattice_index(frame_lens, label_lens, total)
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

    alpha_frames = []
    alpha = torch.full((batch_size, max_prefix), neg_inf, dtype=dtype, device=device)
    alpha[:, 0] = 0.0
    total = torch.full((batch_size,), neg_inf, dtype=dtype, device=device)
    for frame in range(max_frames):
        alpha_frames.append(alpha)
        emit = alpha + label_rows[frame]
        updated = torch.logaddexp(alpha + blank_rows[frame], torch.cat([pad, emit[:, :-1]], dim=1))
        updated = torch.where(valid, updated, torch.full_like(updated, neg_inf))
        alpha = torch.where((frame < frame_lens).unsqueeze(1), updated, alpha)
        total = torch.where((frame + 1) == frame_lens, torch.gather(alpha, 1, lens).squeeze(1), total)
    return total, torch.stack(alpha_frames)


_HAVE_LIB_OPS = False
if hasattr(torch.library, "custom_op"):  # torch >= 2.4

    @torch.library.custom_op("returnn::monotonic_rnnt_fwd", mutates_args=())
    def _lib_fwd(
        logits: torch.Tensor,
        next_label: torch.Tensor,
        frame_lens: torch.Tensor,
        label_lens: torch.Tensor,
        blank: int,
        max_frames: int,
        max_prefix: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from .monotonic_rnnt_triton import cell_stats, forward_scan

        offsets, _cells = cell_offsets(frame_lens, label_lens)
        lse, blank_lp, label_lp = cell_stats(logits, next_label, blank)
        total, alpha = forward_scan(blank_lp, label_lp, offsets, frame_lens, label_lens, max_frames, max_prefix)
        return total, lse, blank_lp, label_lp, alpha

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
            logits.new_empty((max_frames + 1, batch_size, max_prefix), dtype=torch.float32),
        )

    @torch.library.custom_op("returnn::monotonic_rnnt_bwd", mutates_args=())
    def _lib_bwd(
        logits: torch.Tensor,
        next_label: torch.Tensor,
        lse: torch.Tensor,
        blank_lp: torch.Tensor,
        label_lp: torch.Tensor,
        alpha: torch.Tensor,
        total: torch.Tensor,
        frame_lens: torch.Tensor,
        label_lens: torch.Tensor,
        d_total: torch.Tensor,
        blank: int,
    ) -> torch.Tensor:
        from .monotonic_rnnt_triton import backward_scan, cell_grad

        offsets, _cells = cell_offsets(frame_lens, label_lens)
        # the sweep gives the posteriors of the log likelihood, d_total carries the sign of the loss
        blank_grad, label_grad = backward_scan(
            blank_lp, label_lp, offsets, frame_lens, label_lens, alpha, total, -d_total
        )
        return cell_grad(logits, next_label, lse, blank_grad, label_grad, blank).to(logits.dtype)

    @_lib_bwd.register_fake
    def _lib_bwd_fake(logits, next_label, lse, blank_lp, label_lp, alpha, total, frame_lens, label_lens, d_total, blank):
        del next_label, lse, blank_lp, label_lp, alpha, total, frame_lens, label_lens, d_total, blank
        return torch.empty_like(logits)

    def _lib_setup_context(ctx, inputs, output):
        logits, next_label, frame_lens, label_lens, blank, _max_frames, _max_prefix = inputs
        total, lse, blank_lp, label_lp, alpha = output
        ctx.save_for_backward(logits, next_label, lse, blank_lp, label_lp, alpha, total, frame_lens, label_lens)
        ctx.blank = blank

    def _lib_backward(ctx, d_total, d_lse, d_blank_lp, d_label_lp, d_alpha):
        d_lse, d_blank_lp, d_label_lp, d_alpha  # noqa  # unused, only total feeds the loss
        logits, next_label, lse, blank_lp, label_lp, alpha, total, frame_lens, label_lens = ctx.saved_tensors
        grad_logits = torch.ops.returnn.monotonic_rnnt_bwd(
            logits, next_label, lse, blank_lp, label_lp, alpha, total, frame_lens, label_lens, d_total, ctx.blank
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
    max_frames: Optional[int] = None,
) -> torch.Tensor:
    """
    Full-sum negative log likelihood of the monotonic transducer over a packed lattice.

    :param logits: [cells, V] unnormalized, per sequence the frame index outer and the prefix inner
    :param labels: [B, U_max] the reference labels, padded
    :param frame_lens: [B] frames per sequence
    :param label_lens: [B] labels per sequence, at most the frame count
    :param blank: blank index
    :param max_frames: frames the recursion runs over, the longest sequence of the batch by default.
        A traced or captured step passes the declared capacity instead, since reading the batch's own
        maximum is a host read.
    :return: [B] the negative log likelihood, zero where a sequence has no alignment
    """
    assert logits.dim() == 2, logits.shape
    if logits.shape[0] == 0:
        return logits.sum() * torch.zeros(frame_lens.shape[0], dtype=torch.float32, device=logits.device)
    logits = logits.contiguous()
    if max_frames is None:
        max_frames = int(frame_lens.max().item())
    max_prefix = int(labels.shape[1]) + 1
    next_label = next_label_per_cell(labels, frame_lens, label_lens, blank, logits.shape[0])
    if logits.is_cuda:
        assert _HAVE_LIB_OPS, "monotonic rnnt: the loss needs torch.library.custom_op, so torch >= 2.4"
        total = torch.ops.returnn.monotonic_rnnt_fwd(
            logits.float(), next_label, frame_lens, label_lens, blank, max_frames, max_prefix
        )[0]
    else:
        offsets, _cells = cell_offsets(frame_lens, label_lens)
        source = logits if logits.dtype in (torch.float32, torch.float64) else logits.float()
        log_probs = torch.log_softmax(source, dim=-1)
        blank_lp = log_probs[:, blank]
        label_lp = torch.gather(log_probs, 1, next_label.unsqueeze(1)).squeeze(1)
        index = _cell_index(offsets, label_lens, max_frames, max_prefix, logits.shape[0])
        total, _alpha = _forward_scores(blank_lp[index], label_lp[index], frame_lens, label_lens)
    alignable = (label_lens <= frame_lens) & (frame_lens > 0)
    return torch.where(alignable, -total, torch.zeros_like(total))
