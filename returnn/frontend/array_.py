"""
Array (Tensor) functions
"""

from __future__ import annotations
from typing import Union, TypeVar
from returnn.tensor import Tensor
from ._api import RawTensorTypes, global_frontend

T = TypeVar("T")


def convert_to_tensor(value: Union[Tensor, T, RawTensorTypes]) -> Tensor[T]:
    """
    :param value: tensor, or scalar raw tensor or some other scalar value
    :return: tensor
    """
    if isinstance(value, Tensor):  # fast path
        return value
    return global_frontend.convert_to_tensor(value=value)
