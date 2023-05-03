"""
Constant / full / fill / zeros / ones, etc
"""

from __future__ import annotations
from typing import Optional, Sequence
from returnn.tensor import Tensor, Dim
from .types import RawTensorTypes
from ._backend import global_backend
import returnn.frontend as rf


__all__ = ["full", "constant", "fill", "zeros", "ones", "zeros_like", "ones_like"]


def full(
    *, dims: Sequence[Dim], fill_value: RawTensorTypes, dtype: Optional[str] = None, sparse_dim: Optional[Dim] = None
) -> Tensor:
    """
    full

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.full.html

    :param dims: shape
    :param fill_value: scalar to fill the tensor
    :param dtype:
    :param sparse_dim:
    """
    if dtype is None:
        if isinstance(fill_value, int):
            dtype = rf.get_default_int_dtype()
        elif isinstance(fill_value, float):
            dtype = rf.get_default_float_dtype()
        elif isinstance(fill_value, bool):
            dtype = "bool"
        else:
            raise ValueError(f"cannot infer dtype from {fill_value!r} or type ({type(fill_value)})")
    return global_backend.full(dims, fill_value, dtype=dtype, sparse_dim=sparse_dim)


constant = full  # alias for some older code
fill = full  # alias for TF users


def zeros(dims: Sequence[Dim], *, dtype: Optional[str] = None, sparse_dim: Optional[Dim] = None) -> Tensor:
    """
    zeros. float by default.
    """
    return full(dims=dims, fill_value=0, dtype=dtype or rf.get_default_float_dtype(), sparse_dim=sparse_dim)


def ones(dims: Sequence[Dim], *, dtype: Optional[str] = None, sparse_dim: Optional[Dim] = None) -> Tensor:
    """
    ones. float by default.
    """
    return full(dims=dims, fill_value=1, dtype=dtype or rf.get_default_float_dtype(), sparse_dim=sparse_dim)


def zeros_like(other: Tensor) -> Tensor:
    """zeros like other"""
    return zeros(dims=other.dims, dtype=other.dtype, sparse_dim=other.sparse_dim)


def ones_like(other: Tensor) -> Tensor:
    """ones like other"""
    return ones(dims=other.dims, dtype=other.dtype, sparse_dim=other.sparse_dim)
