"""
The :class:`Tensor` class represents a tensor.
This is framework-agnostic, i.e. is fine for TensorFlow, PyTorch, or any other framework.
(Earlier, this was the class ``Data`` in RETURNN, and ``Data`` is now an alias for this class.)

This class is to describe a tensor,
i.e. its shape and properties like
whether we should consider it sparse data (i.e. it represents indices).
Each dimension is described by :class:`Dim`.

This is used in :class:`TFNetwork` to describe the dataset external data (:class:`ExternData`)
as well as in every layer's output and in many other parts of the code.

See :ref:`data`.

This is designed in a way that is efficient for eager-mode execution such as PyTorch,
but at the same time compatible with older RETURNN code.

Discussion on the move from loosely TF-specific ``Data`` to framework-agnostic ``Tensor``:
https://github.com/rwth-i6/returnn/issues/1165
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union, Sequence, Tuple

if TYPE_CHECKING:
    import numpy
    import tensorflow as tf
    import torch

    RawTensorType = Union[numpy.ndarray, tf.Tensor, torch.Tensor]

from returnn.util.basic import NotSpecified
from .dim import Dim
from ._tensor_extra import _TensorExtra, _TensorMixin
from . import _tensor_extra


class Tensor(_TensorMixin):
    """
    Represents a tensor, in a frame-agnostic way. See the module docstring.
    """

    size_dtype = "int32"

    __slots__ = ("name", "_dims", "dtype", "sparse_dim", "_raw_tensor", "_extra")

    name: str
    _dims: Tuple[Dim]
    dtype: str
    sparse_dim: Optional[Dim]
    _raw_tensor: Optional[RawTensorType]
    _extra: Optional[_TensorExtra]

    def __init__(
        self,
        name: str,
        dims: Optional[Sequence[Dim]] = None,
        dtype: Optional[str] = None,
        *,
        sparse_dim: Optional[Dim] = None,
        raw_tensor: Optional[RawTensorType] = None,
        **kwargs,
    ):
        if dims is None:
            # old code
            dims, sparse_dim = _tensor_extra.infer_dim_tags(name=name, sparse_dim=sparse_dim, **kwargs)
        if dtype is None:
            # old defaults
            dtype = "int32" if sparse_dim else "float32"

        self.name = name
        self._dims = tuple(dims)
        self.dtype = dtype
        self.sparse_dim = sparse_dim
        self._raw_tensor = None  # assignment below
        self._extra = None  # type: Optional[_TensorExtra]

        if kwargs:
            self._extra = _TensorExtra(tensor=self, **kwargs)
        if raw_tensor is not None:
            self.raw_tensor = raw_tensor  # assignment via property, to have extra checks


# dispatch table for frameworks for sanity_check
