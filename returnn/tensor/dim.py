"""
Represents a dimension of a tensor.
A dimension can come with further information such as individual sequence lengths.

This identifies one axis/dimension, like a time-dimension, etc.
This was called ``DimensionTag`` earlier, and referred to as dimension tag.

This is used by :class:`Tensor` (earlier ``Data``). See :func:`Tensor.dims`.
This would be passed as ``dims`` when creating a :class:`Tensor` instance.

It is not to specify the specific axis in a specific Tensor,
but to specify the content and dimension.
I.e. if we have the same Dim for two Data instances,
the dimensions should match. I.e.:

    data1.dims[i] == data2.dims[j]
      =>  data1.raw_tensor.shape[i] == data2.raw_tensor.shape[j]

This also includes further information such as sequence lengths
or a vocabulary.

Deprecated: We differentiate between the batch dim, spatial dim or feature dim,
although that is just flag and in many contexts there is no real difference
between a spatial dim and a feature dim (the batch dim is often handled differently).

"""

from __future__ import annotations
from typing import Optional, Union

from ._dim_extra import _DimExtra, _DimMixin, DimTypes
from . import tensor as _t


__all__ = ["Dim", "batch_dim", "single_step_dim", "VerifyOutShapeException"]


class Dim(_DimMixin):
    """
    Represents a dimension of a tensor.
    This potentially comes with further information such as individual sequence lengths.
    See the module docstring.
    """

    Types = DimTypes  # old alias

    __slots__ = ("name", "capacity", "size", "dyn_size_ext", "_dyn_size_max_value", "_extra")

    name: Optional[str]
    capacity: Optional[int]  # shape[axis] in the raw tensor (might need power-of-two or static shape), None if dynamic
    size: Optional[int]  # shape[axis] in the represented tensor if static, None if dynamic, then dyn_size_ext
    dyn_size_ext: Optional[_t.Tensor]
    _dyn_size_max_value: Optional[_t.Tensor]  # scalar
    _extra: Optional[_DimExtra]

    def __init__(
        self,
        dimension: Optional[Union[int, _t.Tensor]],
        *,
        name: Optional[str] = None,
        capacity: Optional[int] = None,
        dyn_size_ext: Optional[_t.Tensor] = None,
        description: Optional[str] = None,
        **kwargs,
    ):
        # dimension is the most common way to specify whether it is static or dynamic,
        # and if dynamic, we can directly pass the dynamic sizes.
        # It also infers reasonable defaults for capacity, if this is not set explicitly.
        # This logic here also covers the old __init__ option dyn_size_ext.
        if dimension is None:
            self.capacity = capacity
            self.size = None
            self.dyn_size_ext = dyn_size_ext.copy() if dyn_size_ext else None
        elif isinstance(dimension, int):
            self.capacity = capacity or dimension
            self.size = dimension
            self.dyn_size_ext = None
        elif isinstance(dimension, _t.Tensor):
            if not dimension.dtype.startswith("int") and not dimension.dtype.startswith("uint"):
                raise TypeError(f"unexpected dtype for dimension: {dimension.dtype}")
            self.capacity = capacity
            self.size = None
            self.dyn_size_ext = dimension.copy()
        else:
            raise TypeError(f"unexpected dimension type: {type(dimension)}")
        if not name and not description and self.dyn_size_ext:
            name = self.dyn_size_ext.name
        self.name = name or description
        self._dyn_size_max_value = None
        self._extra = None

        if kwargs:
            self._handle_extra_kwargs(**kwargs)

    def __repr__(self):
        return "Dim{%s}" % self.short_repr()


# Global batch dim, which would usually be used the dataloader.
batch_dim = Dim(kind=Dim.Types.Batch, description="global batch", dimension=None)

# This indicates to perform a single step execution of some layer which can potentially have recurrent state.
single_step_dim = Dim(description="single-step", kind=Dim.Types.Spatial, special=True, dimension=1)


class VerifyOutShapeException(Exception):
    """
    Exception via :func:`Tensor.verify_out_shape`.
    """
