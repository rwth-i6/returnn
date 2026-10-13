"""
Base class for Tensor mixins. Just to define the attribs.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Generic, Optional, TypeVar, Union, Tuple

if TYPE_CHECKING:
    from ._tensor_extra import _TensorExtra
    from returnn.util.basic import NotSpecified
    from .dim import Dim

RawTensorType = TypeVar("RawTensorType")  # e.g. torch.Tensor, tf.Tensor, numpy.ndarray, ...


class _TensorMixinBase(Generic[RawTensorType]):
    __slots__ = ()

    name: str
    _dims: Tuple[Dim, ...]
    dtype: str
    sparse_dim: Optional[Dim]
    _feature_dim_axis: Optional[Union[int, NotSpecified]]
    _raw_tensor: Optional[RawTensorType]
    raw_tensor: Optional[RawTensorType]
    version: int
    _extra: Optional[_TensorExtra]
