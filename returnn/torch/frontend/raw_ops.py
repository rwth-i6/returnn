"""
Combine raw ops which are not directly supported by PyTorch.
"""

from __future__ import annotations

import torch


def squared_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    :param a:
    :param b:
    :return: (a - b) ** 2
    """
    return torch.square(a - b)
