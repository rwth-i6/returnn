"""
Constant / full / fill / zeros / ones, etc
"""

from __future__ import annotations
from typing import Optional, Union, Sequence
import numpy
from returnn.tensor import Tensor, Dim
from .types import RawTensorTypes
from ._backend import global_backend
import returnn.frontend as rf


__all__ = ["full", "fill", "constant", "zeros", "ones", "zeros_like", "ones_like"]


def full(
    *,
    dims: Sequence[Dim],
    fill_value: Union[RawTensorTypes, Tensor],
    dtype: Optional[str] = None,
    device: Optional[str] = None,
    sparse_dim: Optional[Dim] = None,
    feature_dim: Optional[Dim] = None,
    like: Optional[Tensor] = None,
) -> Tensor:
    """
    full, fill, constant.

    https://data-apis.org/array-api/latest/API_specification/generated/array_api.full.html

    Also see :func:`convert_to_tensor`.

    :param dims: shape
    :param fill_value: scalar to fill the tensor
    :param dtype:
    :param device:
    :param sparse_dim:
    :param feature_dim:
    :param like: the result uses the backend and storage layout of this tensor, for the dims it shares with it.
        E.g. with a packed ``like`` (:func:`returnn.frontend.pack`) and ``dims`` covering its packed dims,
        the result is packed alike, without the padded intermediate.
        Use it when the result is combined with ``like`` anyway.
    """
    if dtype is None:
        if isinstance(fill_value, bool):
            dtype = "bool"
        elif isinstance(fill_value, int):
            dtype = rf.get_default_int_dtype()
        elif isinstance(fill_value, float):
            dtype = rf.get_default_float_dtype()
        elif isinstance(fill_value, Tensor):
            dtype = fill_value.dtype
        else:
            raise ValueError(f"cannot infer dtype from {fill_value!r} or type ({type(fill_value)})")
    if isinstance(fill_value, numpy.ndarray):
        assert fill_value.shape == (), (
            f"full/fill/constant: expect scalar fill_value, got array with shape {fill_value.shape}.\n"
            "Use rf.convert_to_tensor to convert an arbitrary array to a tensor."
        )
    if isinstance(fill_value, Tensor):
        assert fill_value.dims == (), (
            f"full/fill/constant: expect scalar fill_value, got tensor with shape {fill_value.dims}."
        )
    backend = global_backend
    if like is not None and like.raw_tensor is not None:
        backend = like._raw_backend
    return backend.full(
        dims, fill_value, dtype=dtype, device=device, sparse_dim=sparse_dim, feature_dim=feature_dim, like=like
    )


fill = full  # alias for TF users


def constant(
    fill_value: RawTensorTypes,
    *,
    dims: Sequence[Dim],
    dtype: Optional[str] = None,
    device: Optional[str] = None,
    sparse_dim: Optional[Dim] = None,
    feature_dim: Optional[Dim] = None,
    like: Optional[Tensor] = None,
) -> Tensor:
    """alias to :func:`full`, mapping `value` to `fill_value`. also see :func:`convert_to_tensor`"""
    return full(
        dims=dims,
        fill_value=fill_value,
        dtype=dtype,
        device=device,
        sparse_dim=sparse_dim,
        feature_dim=feature_dim,
        like=like,
    )


def zeros(
    dims: Sequence[Dim],
    *,
    dtype: Optional[str] = None,
    device: Optional[str] = None,
    sparse_dim: Optional[Dim] = None,
    feature_dim: Optional[Dim] = None,
    like: Optional[Tensor] = None,
) -> Tensor:
    """
    zeros. float by default. See :func:`full`.
    """
    return full(
        dims=dims,
        fill_value=0,
        dtype=dtype or rf.get_default_float_dtype(),
        device=device,
        sparse_dim=sparse_dim,
        feature_dim=feature_dim,
        like=like,
    )


def ones(
    dims: Sequence[Dim],
    *,
    dtype: Optional[str] = None,
    device: Optional[str] = None,
    sparse_dim: Optional[Dim] = None,
    feature_dim: Optional[Dim] = None,
    like: Optional[Tensor] = None,
) -> Tensor:
    """
    ones. float by default. See :func:`full`.
    """
    return full(
        dims=dims,
        fill_value=1,
        dtype=dtype or rf.get_default_float_dtype(),
        device=device,
        sparse_dim=sparse_dim,
        feature_dim=feature_dim,
        like=like,
    )


def zeros_like(other: Tensor) -> Tensor:
    """zeros like other"""
    return zeros(
        dims=other.dims,
        dtype=other.dtype,
        device=other.device,
        sparse_dim=other.sparse_dim,
        feature_dim=other.feature_dim,
        like=other,
    )


def ones_like(other: Tensor) -> Tensor:
    """ones like other"""
    return ones(
        dims=other.dims,
        dtype=other.dtype,
        device=other.device,
        sparse_dim=other.sparse_dim,
        feature_dim=other.feature_dim,
        like=other,
    )
