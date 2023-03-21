"""
Utilities for dimension tags, dimensions, axes.
"""


from __future__ import annotations
from typing import TypeVar
from returnn.tensor import Tensor, Dim
from ._backend import get_backend_by_tensor, global_backend

T = TypeVar("T")

__all__ = ["range_over_dim", "dim_match_priority_when_needed"]


def range_over_dim(dim: Dim) -> Tensor[T]:
    """
    :param dim:
    :return: tensor with shape [dim]
    """
    if dim.dyn_size_ext:
        rf = get_backend_by_tensor(dim.dyn_size_ext, fallback=global_backend)
    else:
        rf = global_backend
    return rf.range_over_dim(dim)


def dim_match_priority_when_needed(dim: Dim, *other_dims: Dim) -> Dim:
    """
    :return: maybe copy of dim with higher match_priority if needed to distinguish from other_dims
    """
    if dim in other_dims:
        return dim.copy(match_priority=1)
    return dim
