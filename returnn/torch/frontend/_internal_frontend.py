"""
Internal frontend for PyTorch frontend.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch

from returnn._internal_frontend_api import InternalFrontend
from returnn.tensor import Tensor

_TT = Tensor[torch.Tensor]


# Ignore this warning until we implemented everything.
# noinspection PyAbstractClass
class TorchInternalFrontend(InternalFrontend[torch.Tensor]):
    """
    Internal frontend for PyTorch frontend.
    """

    @staticmethod
    def get_dtype_name_raw(raw_tensor: torch.Tensor) -> str:
        """
        :return: dtype of raw tensor, as string
        """
        return str(raw_tensor.dtype).replace("torch.", "")

    @staticmethod
    def get_ndim_raw(raw_tensor: torch.Tensor) -> int:
        """
        :return: ndim of raw tensor
        """
        return raw_tensor.dim()

    @staticmethod
    def get_known_shape_raw(raw_tensor: torch.Tensor) -> Tuple[Optional[int]]:
        """
        :return: shape of raw tensor; here for PyTorch the full shape is always known
        """
        return tuple(raw_tensor.size())
