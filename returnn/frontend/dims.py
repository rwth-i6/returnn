"""
Utilities for dimension tags, dimensions, axes.
"""


from __future__ import annotations
from typing import TypeVar
from returnn.tensor import Tensor, Dim
from ._api import get_frontend_by_tensor, global_frontend

T = TypeVar("T")


def range_over_dim(dim: Dim) -> Tensor[T]:
    """
    :param dim:
    :return: tensor with shape [dim]
    """
    if dim.dyn_size_ext:
        rf = get_frontend_by_tensor(dim.dyn_size_ext, fallback=global_frontend)
    else:
        rf = global_frontend
    return rf.range_over_dim(dim)
