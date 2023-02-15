"""
Represents a dimension of a tensor.
A dimension can come with further information such as individual sequence lengths.
"""


from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union

from ._dim_extra import _DimExtra, _DimMixin
from . import tensor


class Dim(_DimMixin):
    """
    Represents a dimension of a tensor.
    This potentially comes with further information such as individual sequence lengths.
    """

    __slots__ = ("name", "capacity", "size", "dyn_size_ext", "_extra")

    name: Optional[str]
    capacity: Optional[int]  # shape[axis] in the raw tensor (might need power-of-two or static shape), None if dynamic
    size: Optional[int]  # shape[axis] in the represented tensor if static, None if dynamic, then dyn_size_ext
    dyn_size_ext: Optional[tensor.Tensor]
    _extra: Optional[_DimExtra]

    def __init__(
        self,
        dimension: Optional[Union[int, tensor.Tensor]],
        *,
        name: Optional[str] = None,
        capacity: Optional[int] = None,
        **kwargs,
    ):
        self.name = name
        if dimension is None:
            pass
        elif isinstance(dimension, int):
            self.size = dimension
        elif isinstance(dimension, tensor.Tensor):
            pass
