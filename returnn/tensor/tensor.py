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
from typing import Optional, Sequence, Tuple, Generic, TypeVar

from .dim import Dim
from ._tensor_extra import _TensorExtra, _TensorMixin
from . import _tensor_extra

RawTensorType = TypeVar("RawTensorType")  # e.g. torch.Tensor, tf.Tensor, numpy.ndarray, ...


class Tensor(_TensorMixin, Generic[RawTensorType]):
    """
    Represents a tensor, in a frame-agnostic way. See the module docstring.
    """

    size_dtype = "int32"

    __slots__ = ("name", "_dims", "dtype", "sparse_dim", "_raw_tensor", "version", "_extra")

    name: str
    _dims: Tuple[Dim, ...]
    dtype: str
    sparse_dim: Optional[Dim]
    _raw_tensor: Optional[RawTensorType]
    version: int
    _extra: Optional[_TensorExtra]

    def __init__(
        self,
        name: str,
        dims: Optional[Sequence[Dim]] = None,
        dtype: Optional[str] = None,
        *,
        sparse_dim: Optional[Dim] = None,
        raw_tensor: Optional[RawTensorType] = None,
        version: Optional[int] = None,
        **kwargs,
    ):
        """
        :param name:
        :param dims: the shape, where each dimension is described by a :class:`Dim`.
        :param dtype: e.g. "float32" or "int64"
        :param sparse_dim: when the values are indices into some dimension, this is the dimension.
            You can also interpret the whole tensor as a sparse representation of a dense one-hot tensor,
            where this sparse_dim becomes the additional dense dimension.
        :param raw_tensor: the raw tensor, e.g. numpy array, TF tensor, or PyTorch tensor
        :param version: behavior version just for Tensor. If not specified, and `dims` is None (old code),
            it uses version 1.
            - v1: the old behavior of Data. Specifically, time_dim_axis and feature_dim_axis are used
                and automatically inferred when not specified.
            - v2: time_dim_axis, feature_dim_axis are None by default.
        :param kwargs: see :func:`_handle_extra_kwargs`, :func:`infer_dim_tags`
        """
        if "sparse" in kwargs and sparse_dim is None:
            sparse_dim = _tensor_extra.infer_sparse_dim(name=name, sparse_dim=sparse_dim, **kwargs)
        if dims is not None:
            assert "shape" not in kwargs and "dim_tags" not in kwargs  # probably old code got this wrong
            if version is None:
                version = 2
        else:
            # old code
            dims = _tensor_extra.infer_dim_tags(name=name, sparse_dim=sparse_dim, **kwargs)
            if version is None:
                version = 1
        if dtype is None:
            # old defaults
            dtype = "int32" if sparse_dim else "float32"

        self.name = name
        self._dims = tuple(dims)
        self.dtype = dtype
        self.sparse_dim = sparse_dim
        self._raw_tensor = None  # assignment below
        self.version = version
        self._extra = None  # type: Optional[_TensorExtra]

        if kwargs:
            self._handle_extra_kwargs(**kwargs)
        if raw_tensor is not None:
            self.raw_tensor = raw_tensor  # assignment via property, to have extra checks

    @property
    def dims(self) -> Tuple[Dim, ...]:
        """
        :return: dim tags
        """
        return self._dims

    @property
    def raw_tensor(self) -> Optional[RawTensorType]:
        """
        :return: raw tensor
        """
        return self._raw_tensor

    @raw_tensor.setter
    def raw_tensor(self, value: Optional[RawTensorType]):
        """
        :param value:
        """
        self._raw_tensor = value
        self.sanity_check(assume_complete=False)


# dispatch table for frameworks for sanity_check
