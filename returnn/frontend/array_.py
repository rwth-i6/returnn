"""
Array (Tensor) functions
"""

from __future__ import annotations
from typing import Union, TypeVar
from returnn.tensor import Tensor
from ._backend import global_backend
from .types import RawTensorTypes

T = TypeVar("T")


def convert_to_tensor(value: Union[Tensor, T, RawTensorTypes]) -> Tensor[T]:
    """
    :param value: tensor, or scalar raw tensor or some other scalar value
    :return: tensor
    """
    if isinstance(value, Tensor):  # fast path
        return value
    return global_backend.convert_to_tensor(value=value)
