"""
Frontend for exposing PyTorch-specific functionality.
"""

from __future__ import annotations

import torch

from returnn.frontend_api import Frontend


# Ignore this warning until we really expect that we implemented everything.
# noinspection PyAbstractClass
class TorchFrontend(Frontend[torch.Tensor]):
    """
    PyTorch frontend
    """

    RawTensorType = torch.Tensor
