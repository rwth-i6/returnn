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

    Public attributes:

    * ``name``: short name, mostly for debugging.
    * ``size``: static size (``shape[axis]`` of the represented tensor), or None if dynamic.
    * ``dyn_size_ext``: the sizes as a :class:`Tensor` (e.g. shape [Batch]) if dynamic, else None.
    * ``capacity``: static upper bound of the sizes = ``shape[axis]`` of the (padded) raw tensor
      where a static shape is required (e.g. CUDA-graph capture / static tracing);
      for static dims this defaults to ``size``.

    Further public properties (see :class:`_DimMixin`), most relevant:

    * ``dimension``: alias of ``size``.
    * ``description``: long description, unique-ish, for debugging and repr.
    * ``kind``: batch / spatial / feature (:class:`DimTypes`).
    * ``derived_from_tag``: some other dim this one was derived from
      (reduced, down/up sampled, padded, ...). Marks ONLY the dependency,
      no relation of the sizes is implied.
      See also the ``bounded_by`` ``__init__`` arg, which additionally bounds the sizes.
    * ``derived_from_op``: the exact dim-math operation (e.g. ``a + b``, ``a * b``)
      this dim was created from, if any. Implies exact size and capacity propagation.
    * ``vocab``, ``batch``, ``control_flow_ctx``: further optional metadata.
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
        bounded_by: Optional[Dim] = None,
        **kwargs,
    ):
        """
        :param dimension: static size (int), or the dynamic sizes as a :class:`Tensor`
            (e.g. shape [Batch], int dtype), or None (dynamic, sizes defined later).
            Also infers a reasonable default for ``capacity`` (for static dims).
        :param name: short name, mostly for debugging.
        :param capacity: static upper bound of the sizes, see the class docstring.
        :param dyn_size_ext: the dynamic sizes (older alternative to passing them via ``dimension``).
        :param description: long description, see the class docstring.
        :param bounded_by: declares that this dim's sizes never exceed that dim's
            (thus e.g. its capacity is a valid capacity here too, resolved lazily,
            also when declared later).
            This is a stronger statement than ``derived_from_tag``,
            which it also implies and sets.
        :param kwargs: further (rarer) options, see :class:`_DimExtra`:
            ``kind`` (see :class:`DimTypes`),
            ``derived_from_tag`` (dependency marker ONLY, no size relation implied),
            ``derived_from_op`` (exact dim-math relation),
            ``vocab``, ``batch``, ``match_priority``, ``auto_generated``, ...
        """
        # dimension is the most common way to specify whether it is static or dynamic,
        # and if dynamic, we can directly pass the dynamic sizes.
        # It also infers reasonable defaults for capacity, if this is not set explicitly.
        # This logic here also covers the old __init__ option dyn_size_ext.
        if dimension is None:
            self.capacity = capacity
            self.size = None
            self.dyn_size_ext = dyn_size_ext.copy() if dyn_size_ext is not None else None
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
        if not name and not description and self.dyn_size_ext is not None:
            name = self.dyn_size_ext.name
        self.name = name or description
        self._dyn_size_max_value = None
        self._extra = None

        if bounded_by is not None:
            # This dim's sizes never exceed ``bounded_by``'s extent
            # (a stronger statement than a plain derived_from_tag,
            # which marks an arbitrary dependency: down/up sampled, padded, ...),
            # thus it e.g. inherits its capacity, resolved lazily
            # (so a capacity declared on the source AFTER this dim was created still applies).
            # It also implies the dependency, thus derived_from_tag is set as well.
            # See :func:`_DimMixin._derived_capacity`.
            kwargs["derived_from_tag"] = bounded_by
        if kwargs:
            self._handle_extra_kwargs(**kwargs)
        if bounded_by is not None:
            self._make_extra().bounded_by = bounded_by

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
