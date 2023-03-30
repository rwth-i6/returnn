"""
Types
"""

from __future__ import annotations
from typing import Union, Type, TYPE_CHECKING
import numpy
from returnn.tensor import Tensor, Dim
from ._backend import global_backend


__all__ = ["RawTensorTypes", "Tensor", "Dim", "get_raw_tensor_type"]


RawTensorTypes = Union[int, float, complex, numpy.number, numpy.ndarray, bool, str]


def get_raw_tensor_type() -> Type:
    """
    :return: the raw tensor type of the current selected backend, e.g. ``torch.Tensor`` or ``tf.Tensor``
    """
    if TYPE_CHECKING:
        import torch

        return torch.Tensor  # just as an example
    return global_backend.RawTensorType
