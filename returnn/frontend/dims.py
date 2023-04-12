"""
Utilities for dimension tags, dimensions, axes.
"""


from __future__ import annotations
from typing import Union, TypeVar, Sequence
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from ._backend import get_backend_by_tensor, global_backend

T = TypeVar("T")

__all__ = ["range_over_dim", "dim_match_priority_when_needed", "num_elements_of_shape"]


def range_over_dim(dim: Dim) -> Tensor[T]:
    """
    :param dim:
    :return: tensor with shape [dim]
    """
    if dim.dyn_size_ext:
        backend = get_backend_by_tensor(dim.dyn_size_ext, fallback=global_backend)
    else:
        backend = global_backend
    return backend.range_over_dim(dim)


def dim_match_priority_when_needed(dim: Dim, *other_dims: Dim) -> Dim:
    """
    :return: maybe copy of dim with higher match_priority if needed to distinguish from other_dims
    """
    if dim in other_dims:
        return dim.copy(match_priority=1)
    return dim


def num_elements_of_shape(dims: Sequence[Dim]) -> Union[int, Tensor]:
    """
    :param dims:
    :return: num elements of a tensor of shape dims, properly considering masking
    """
    if all(dim.is_static() for dim in dims):
        n = 1
        for dim in dims:
            n *= dim.dimension
        return n

    n = 1
    dims = list(dims)
    dims.sort(key=lambda dim: -dim.dyn_size_ext.batch_ndim if dim.dyn_size_ext else 0)
    while dims:
        dim = dims.pop(0)
        if dim.is_static():
            n *= dim.dimension
            continue
        # E.g. dyn_size_ext is shape [B], and self has shape [B,T].
        # Due to the sorting of dims above, dims will be [T,B], and we will first process T.
        # We want to sum over dyn_size_ext, but then we need to remove the other dims it covers.
        for dim_ in dim.dyn_size_ext.dims:
            assert dim_ in dims  # num elements not really well-defined then
            assert not dim_.need_masking()  # not implemented
            dims.remove(dim_)
        n_ = rf.reduce_sum(dim.dyn_size_ext, axis=dim.dyn_size_ext.dims)
        n *= n_
    return n
