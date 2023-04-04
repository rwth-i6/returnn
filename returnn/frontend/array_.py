"""
Array (Tensor) functions
"""

from __future__ import annotations
from typing import Optional, Union, TypeVar, Sequence
import numpy
from returnn.tensor import Tensor, Dim
from ._backend import global_backend, get_backend_by_raw_tensor_type
from .types import RawTensorTypes

T = TypeVar("T")

__all__ = ["convert_to_tensor", "constant", "gather"]


def convert_to_tensor(
    value: Union[Tensor, T, RawTensorTypes],
    *,
    dims: Sequence[Dim] = None,
    dtype: Optional[str] = None,
    sparse_dim: Optional[Dim] = None,
    shape: Sequence[Dim] = None,
) -> Tensor[T]:
    """
    :param value: tensor, or scalar raw tensor or some other scalar value
    :param dims:
    :param dtype:
    :param sparse_dim:
    :param shape: alias for dims, for some older code
    :return: tensor
    """
    if isinstance(value, Tensor):  # fast path
        return value
    if isinstance(value, (int, float, complex, bool, str, numpy.number, numpy.ndarray)):
        backend = global_backend
    else:
        backend = get_backend_by_raw_tensor_type(type(value))
    if dims is None and shape is not None:
        dims = shape
    if dims is None:
        dims = ()
    return backend.convert_to_tensor(value=value, dims=dims, dtype=dtype, sparse_dim=sparse_dim)


constant = convert_to_tensor  # alias for some older code


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
    In this case, this simply gives ``source[indices]`` on the specified ``axis``.

    :param source:
    :param indices: indices used to select the slices of the source from.
        If another tensor, must be of type ``int32`` or ``int64``.
        Can also specify a constant ``int``.
    :param axis: The axis into which we gather the indices into
    :param clip_to_valid: if True, the indices will be clipped to the valid range of the input
        Also taking seq lengths into account.
    :return: gathered values
    """
    # noinspection PyProtectedMember
    return source._raw_backend.gather(source, indices=indices, axis=axis, clip_to_valid=clip_to_valid)
