"""
Torch launchers of the Triton kernels of the two sweeps of the packed RNN-T loss, see :mod:`returnn.torch.util.rnnt`.
The kernels are in :mod:`returnn.triton.rnnt`, which describes them.
"""

from __future__ import annotations

from typing import Tuple

import torch

from returnn.triton import monotonic_rnnt
from returnn.triton import rnnt as kernels


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
    assert blank_lp.is_cuda and kernels.triton is not None, "rnnt: the scan kernels need cuda and triton"
    batch_size = frame_lens.shape[0]
    alpha = torch.empty_like(blank_lp)
    total = torch.empty((batch_size,), dtype=torch.float32, device=blank_lp.device)
    kernels.forward_scan_kernel[(batch_size,)](
        blank_lp,
        label_lp,
        offsets,
        frame_lens,
        label_lens,
        blank_lp.shape[0],
        max_frames + max_prefix - 1,
        max_prefix,
        alpha,
        total,
        BLOCK=monotonic_rnnt.block_for(max_prefix),
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
    :param weight: [B] incoming gradient of the log likelihood
    :param max_frames: frames to sweep, the bound the forward sweep ran with
    :param max_prefix: prefixes to sweep, U_max + 1
    :return: (blank gradient [cells], label gradient [cells]) wrt the two log probabilities
    """
    assert blank_lp.is_cuda and kernels.triton is not None, "rnnt: the scan kernels need cuda and triton"
    batch_size = frame_lens.shape[0]
    beta = torch.empty_like(blank_lp)
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
        max_frames + max_prefix - 1,
        max_prefix,
        beta,
        blank_grad,
        label_grad,
        BLOCK=monotonic_rnnt.block_for(max_prefix),
    )
    return blank_grad, label_grad
