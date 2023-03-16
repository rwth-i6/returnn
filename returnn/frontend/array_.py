"""
Array (Tensor) functions
"""

from __future__ import annotations
from typing import Union, TypeVar, Sequence
from returnn.util.basic import NotSpecified
from returnn.tensor import Tensor, Dim
from ._api import RawTensorTypes, get_frontend_by_tensor, global_frontend

T = TypeVar("T")


def convert_to_tensor(value: Union[Tensor, T, RawTensorTypes]) -> Tensor[T]:
    """
    :param value: tensor, or scalar raw tensor or some other scalar value
    :return: tensor
    """
    if isinstance(value, Tensor):  # fast path
        return value
    return global_frontend.convert_to_tensor(value=value)
