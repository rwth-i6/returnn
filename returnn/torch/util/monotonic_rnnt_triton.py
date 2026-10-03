"""
Torch launchers of the Triton kernels of the packed monotonic RNN-T loss, see :mod:`returnn.torch.util.monotonic_rnnt`.
The kernels are in :mod:`returnn.triton.monotonic_rnnt`, which describes them.
"""

from __future__ import annotations

from typing import Tuple

import torch

from returnn.triton import monotonic_rnnt as kernels


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
    assert blank_lp.is_cuda and kernels.triton is not None, "monotonic rnnt: the scan kernels need cuda and triton"
    batch_size = frame_lens.shape[0]
    alpha = torch.empty((max_frames + 1, batch_size, max_prefix), dtype=torch.float32, device=blank_lp.device)
    total = torch.empty((batch_size,), dtype=torch.float32, device=blank_lp.device)
    kernels.forward_scan_kernel[(batch_size,)](
        blank_lp,
        label_lp,
        offsets,
        frame_lens,
        label_lens,
        blank_lp.shape[0],
        batch_size * max_prefix,
        max_frames,
        max_prefix,
        alpha,
        total,
        BLOCK=kernels.block_for(max_prefix),
    )
    return total, alpha


def backward_scan(
    blank_lp: torch.Tensor,
    label_lp: torch.Tensor,
    offsets: torch.Tensor,
    frame_lens: torch.Tensor,
    label_lens: torch.Tensor,
    alpha: torch.Tensor,
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param blank_lp: [cells] blank log probability
    :param label_lp: [cells] next label log probability
    :param offsets: [B] first cell of every sequence
    :param frame_lens: [B] frames per sequence
    :param label_lens: [B] labels per sequence
    :param alpha: [max_frames + 1, B, max_prefix] from :func:`forward_scan`
    :param weight: [B] incoming gradient of the log likelihood
    :return: (blank gradient [cells], label gradient [cells]) wrt the two log probabilities
    """
    assert blank_lp.is_cuda and kernels.triton is not None, "monotonic rnnt: the scan kernels need cuda and triton"
    frames_plus_one, batch_size, max_prefix = alpha.shape
    beta = torch.empty((batch_size, max_prefix), dtype=torch.float32, device=blank_lp.device)
    blank_grad = torch.zeros_like(blank_lp)
    label_grad = torch.zeros_like(label_lp)
    kernels.backward_scan_kernel[(batch_size,)](
        blank_lp,
        label_lp,
        offsets,
        frame_lens,
        label_lens,
        alpha,
        weight,
        blank_lp.shape[0],
        batch_size * max_prefix,
        frames_plus_one - 1,
        max_prefix,
        beta,
        blank_grad,
        label_grad,
        BLOCK=kernels.block_for(max_prefix),
    )
    return blank_grad, label_grad


def cell_stats(logits: torch.Tensor, next_label: torch.Tensor, blank: int) -> Tuple[torch.Tensor, ...]:
    """
    :param logits: [cells, vocab] unnormalized
    :param next_label: [cells] the label the emitting edge of every cell carries
    :param blank: blank index
    :return: (row maximum, log sum of the exponentials past it, blank log prob, label log prob), each [cells]
    """
    assert logits.is_cuda and kernels.triton is not None, "monotonic rnnt: the cell kernels need cuda and triton"
    assert logits.is_contiguous(), "monotonic rnnt: the cell kernels address the logits row by row"
    cells, vocab = logits.shape
    out = [torch.empty(cells, dtype=torch.float32, device=logits.device) for _ in range(4)]
    # noinspection PyArgumentList
    kernels.cell_stats_kernel[(cells,)](
        logits,
        next_label,
        vocab,
        blank,
        out[0],
        out[1],
        out[2],
        out[3],
        BLOCK=kernels.CELL_BLOCK,
        num_warps=kernels.CELL_NUM_WARPS,
    )
    return tuple(out)


def cell_grad(
    logits: torch.Tensor,
    next_label: torch.Tensor,
    row_max: torch.Tensor,
    log_sum: torch.Tensor,
    blank_grad: torch.Tensor,
    label_grad: torch.Tensor,
    blank: int,
) -> torch.Tensor:
    """
    :param logits: [cells, vocab] unnormalized
    :param next_label: [cells] the label the emitting edge of every cell carries
    :param row_max: [cells] row maximum from :func:`cell_stats`
    :param log_sum: [cells] log sum of the exponentials past the row maximum from :func:`cell_stats`
    :param blank_grad: [cells] gradient of the loss wrt the blank log probability
    :param label_grad: [cells] gradient of the loss wrt the label log probability
    :param blank: blank index
    :return: [cells, vocab] gradient wrt the logits
    """
    assert logits.is_cuda and kernels.triton is not None, "monotonic rnnt: the cell kernels need cuda and triton"
    cells, vocab = logits.shape
    out = torch.empty_like(logits)
    # noinspection PyArgumentList
    kernels.cell_grad_kernel[(cells,)](
        logits,
        next_label,
        row_max,
        log_sum,
        blank_grad,
        label_grad,
        vocab,
        blank,
        out,
        BLOCK=kernels.CELL_BLOCK,
        num_warps=kernels.CELL_NUM_WARPS,
    )
    return out
