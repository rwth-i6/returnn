"""
Frontend for exposing PyTorch-specific functionality.
"""

from __future__ import annotations

import torch

from returnn.frontend_api import Frontend


class TorchFrontend(Frontend[torch.Tensor]):
    """
    PyTorch frontend
    """

    T = torch.Tensor
