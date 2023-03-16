"""
Base class for Tensor mixins. Just to define the attribs.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union, Tuple

if TYPE_CHECKING:
    from ._tensor_extra import _TensorExtra
    from returnn.util.basic import NotSpecified
    import returnn.tensor.tensor as _t
    from .dim import Dim


class _TensorMixinBase:
    name: str
    _dims: Tuple[Dim, ...]
    dtype: str
    sparse_dim: Optional[Dim]
    _feature_dim_axis: Optional[Union[int, NotSpecified]]
    _raw_tensor: Optional[_t.RawTensorType]
    raw_tensor: Optional[_t.RawTensorType]
    version: int
    _extra: Optional[_TensorExtra]
