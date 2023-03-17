"""
Utilities for dimension tags, dimensions, axes.
"""


from __future__ import annotations
from typing import TypeVar
from returnn.tensor import Tensor, Dim
from ._backend import get_backend_by_tensor, global_backend

T = TypeVar("T")

__all__ = ["range_over_dim"]


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
