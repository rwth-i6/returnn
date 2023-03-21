"""
Array (Tensor) functions
"""

from __future__ import annotations
from typing import Union, TypeVar
from returnn.tensor import Tensor, Dim
from ._backend import global_backend
from .types import RawTensorTypes

T = TypeVar("T")

__all__ = ["convert_to_tensor", "gather"]


def convert_to_tensor(value: Union[Tensor, T, RawTensorTypes]) -> Tensor[T]:
    """
    :param value: tensor, or scalar raw tensor or some other scalar value
    :return: tensor
    """
    if isinstance(value, Tensor):  # fast path
        return value
    return global_backend.convert_to_tensor(value=value)


# noinspection PyUnusedLocal
def gather(
    source: Tensor,
    *,
    indices: Union[Tensor, int],
    axis: Dim,
    clip_to_valid: bool = False,
) -> Tensor:
    """
    Gathers slices on a specified axis from the source using indices.
    If the source is of the shape ``[B,D,F1]``, and indices of shape ``[B,F2]``,
    this will yield output of the shape ``[B,F2,F1]`` where

    ``output[b,f2,f1] = source[b,indices[b,f2],f1]``

    (if ``D`` is the axis to gather from).
    In general, all shared axes of the input and the positions will be considered as batch-axes.

    The ``indices`` argument can also be an ``int``.
    In this case, this simply gives ``source[indices]`` one the specified ``axis``.

    :param source:
    :param indices: indices used to select the slices of the source from.
        If another layer, must be of type ``int32`` or ``int64``.
        Can also specify a constant ``int``.
    :param axis: The axis into which we gather the indices into
    :param clip_to_valid: if True, the indices will be clipped to the valid range of the input
        Also taking seq lengths into account.
    :return: layer
    """
    raise NotImplementedError
