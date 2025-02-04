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
from typing import Optional, Union, Sequence, Tuple, Generic, TypeVar, Set

from returnn.util.basic import NotSpecified
from .dim import Dim
from ._tensor_extra import _TensorExtra, _TensorMixin
from ._tensor_op_overloads import _TensorOpOverloadsMixin
import returnn.tensor._tensor_extra as _tensor_extra


__all__ = ["Tensor"]


RawTensorType = TypeVar("RawTensorType")  # e.g. torch.Tensor, tf.Tensor, numpy.ndarray, ...


class Tensor(_TensorMixin, _TensorOpOverloadsMixin, Generic[RawTensorType]):
    """
    Represents a tensor, in a frame-agnostic way. See the module docstring.
    """

    size_dtype = "int32"

    __slots__ = ("name", "_dims", "dtype", "sparse_dim", "_raw_tensor", "_feature_dim_axis", "version", "_extra")

    name: str
    _dims: Tuple[Dim, ...]
    dtype: str
    sparse_dim: Optional[Dim]
    _raw_tensor: Optional[RawTensorType]

    _feature_dim_axis: Optional[Union[int, NotSpecified]]  # https://github.com/rwth-i6/returnn/issues/1273
    version: int
    _extra: Optional[_TensorExtra]

    # This is potentially replaced by native implementation.
    def __init__(
        self,
        name: str,
        dims: Optional[Sequence[Dim]] = None,
        dtype: Optional[str] = None,
        *,
        sparse_dim: Optional[Dim] = None,
        feature_dim: Optional[Dim] = None,
        feature_dim_axis: Optional[Union[int, NotSpecified]] = NotSpecified,
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
        if ("sparse" in kwargs or "vocab" in kwargs) and sparse_dim is None:
            sparse_dim = _tensor_extra.infer_sparse_dim(name=name, **kwargs)
        if dims is not None:
            assert "shape" not in kwargs and "dim_tags" not in kwargs  # probably old code got this wrong
            if version is None:
                version = 2
        else:
            # old code
            dims = _tensor_extra.infer_dim_tags(
                name=name, sparse_dim=sparse_dim, feature_dim_axis=feature_dim_axis, **kwargs
            )
            if version is None:
                version = 1
        if dtype is None:
            # old defaults
            if version == 1:
                dtype = "int32" if sparse_dim else "float32"
            else:
                raise ValueError("Tensor dtype needs to be specified")

        self.name = name
        self._dims = tuple(dims)
        self.dtype = dtype
        self.sparse_dim = sparse_dim
        self._raw_tensor = None  # assignment below
        self.version = version
        self._extra = None  # type: Optional[_TensorExtra]

        if feature_dim_axis is NotSpecified:
            # version == 1 leaves it as NotSpecified,
            #   which enables some dynamic default behavior, like taking the last axis.
            # version >= 2 avoids any default behavior, and just sets it to None.
            if version >= 2:
                feature_dim_axis = None
        elif feature_dim_axis is None:
            pass
        elif isinstance(feature_dim_axis, int):
            assert not self.sparse_dim, "cannot have feature_dim_axis when sparse"
            if feature_dim_axis < 0:
                feature_dim_axis += self.batch_ndim
            assert 0 <= feature_dim_axis < self.batch_ndim
        else:
            raise TypeError(f"unexpected feature_dim_axis type {type(feature_dim_axis)}")
        self._feature_dim_axis = feature_dim_axis
        if feature_dim:
            self.feature_dim = feature_dim

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
    def dims_set(self) -> Set[Dim]:
        """
        :return: set of dim tags. in all high-level code, the order of dims is irrelevant.
          The order must not play a role
          (RETURNN principles: https://github.com/rwth-i6/returnn/wiki/RETURNN-principles).
          Note that we do not include any implicit dims here.
          Also see :func:`verify_out_shape` and https://github.com/rwth-i6/returnn/issues/1153.
        """
        return set(self._dims)

    # This is potentially replaced by native implementation (Tensor._raw_tensor.__get__)
    @property
    def raw_tensor(self) -> Optional[RawTensorType]:
        """
        :return: raw tensor
        """
        return self._raw_tensor

    # This is potentially replaced by native implementation (_native tensor_raw_tensor_setter).
    @raw_tensor.setter
    def raw_tensor(self, value: Optional[RawTensorType]):
        """
        :param value:
        """
        if value is not None:
            # Small part from sanity _heck.
            # noinspection PyProtectedMember,PyShadowingNames
            import returnn.frontend._backend as _backend_api

            backend = _backend_api.get_backend_by_raw_tensor_type(type(value))
            raw_shape = backend.get_known_shape_raw(value)
            assert len(raw_shape) == len(self._dims), f"Mismatching shape ndim: Raw tensor {raw_shape} vs Tensor {self}"
            for i, dim in enumerate(self._dims):
                if dim.dimension is None:
                    continue  # we allow anything in the raw_tensor dim
                if raw_shape[i] != dim.dimension:
                    raise Exception(
                        f"Mismatching shape: Raw tensor {raw_shape} vs Tensor {self};\n"
                        + backend.format_graph_output(value, max_depth=3)
                    )
            if not backend.executing_eagerly():
                backend.set_known_shape_raw(value, self.batch_shape)
            assert backend.get_dtype_name_raw(value) == self.dtype, (
                f"{self} dtype {self.dtype} does not match " f"raw tensor dtype {backend.get_dtype_name_raw(value)}"
            )
        self._raw_tensor = value

    @property
    def feature_dim(self) -> Optional[Dim]:
        """
        :return: self.dims[self.feature_dim_axis] or None.
            See https://github.com/rwth-i6/returnn/issues/1273 for some discussion.
        """
        # first fast paths
        if self._feature_dim_axis is None:
            return None
        if isinstance(self._feature_dim_axis, int):
            return self._dims[self._feature_dim_axis]
        if self.feature_dim_axis is None:
            return None
        return self._dims[self.feature_dim_axis]

    @feature_dim.setter
    def feature_dim(self, value: Optional[Dim]):
        """
        :param value:
        """
        if value is None:
            self._feature_dim_axis = None
            return
        assert not self.sparse_dim, "cannot have feature_dim_axis when sparse"
        self._feature_dim_axis = self.get_axis_from_description(value, allow_int=False)

    @property
    def device(self) -> Optional[str]:
        """
        :return: device
        """
        if self.raw_tensor is None:
            return None
        return self._raw_backend.get_device(self)
