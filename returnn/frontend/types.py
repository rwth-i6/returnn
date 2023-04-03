"""
Types
"""

from __future__ import annotations
from typing import Union, Type, TYPE_CHECKING
import numpy
from returnn.tensor import Tensor, Dim, TensorDict
from ._backend import global_backend
import returnn.frontend as rf


try:
    from typing import Protocol
except ImportError:
    try:
        from typing_extensions import Protocol
    except ImportError:
        Protocol = object


__all__ = ["RawTensorTypes", "Tensor", "Dim", "get_raw_tensor_type", "GetModelFunc", "StepFunc"]


RawTensorTypes = Union[int, float, complex, numpy.number, numpy.ndarray, bool, str]


class GetModelFunc(Protocol):
    """get model func"""

    def __call__(self, *, epoch: int, step: int) -> rf.Module:
        ...


class StepFunc(Protocol):
    """step func"""

    def __call__(self, *, model: rf.Module, extern_data: TensorDict) -> None:
        ...


def get_raw_tensor_type() -> Type:
    """
    :return: the raw tensor type of the current selected backend, e.g. ``torch.Tensor`` or ``tf.Tensor``
    """
    if TYPE_CHECKING:
        import torch

        return torch.Tensor  # just as an example
    return global_backend.RawTensorType
