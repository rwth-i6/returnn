"""
Types
"""

from __future__ import annotations
from typing import Union, Type, TYPE_CHECKING, Protocol, Sequence
import numpy
from returnn.tensor import Tensor, Dim, TensorDict
import returnn.frontend as rf


__all__ = ["RawTensorTypes", "ItemKeyType", "Tensor", "Dim", "get_raw_tensor_type", "GetModelFunc", "StepFunc"]


RawTensorTypes = Union[int, float, complex, numpy.number, numpy.ndarray, bool, str]
ItemKeyType = Union[RawTensorTypes, Tensor, slice, Sequence[Union[RawTensorTypes, Tensor, slice]]]


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
    from ._backend import global_backend

    return global_backend.RawTensorType
