"""
Backwards-compatible functions and attribs for the old ``Data`` class,
or just rarely used attribs, such that we can save memory for the common case.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union, Type, Sequence, Tuple, List, Dict, Set
import os

if TYPE_CHECKING:
    # Those are only used for TensorFlow, or they are deprecated.
    from returnn.tf.util.data import BatchInfo, SearchBeam
    from .control_flow_ctx import ControlFlowContext

    from .tensor import Tensor  # just for type hints; otherwise use _t.Tensor

    # noinspection PyProtectedMember
    from returnn.frontend._backend import Backend

from returnn.util.basic import NotSpecified
from .dim import Dim, batch_dim, VerifyOutShapeException
import returnn.tensor.tensor as _t
import returnn.tensor.marked_dim as _m

from ._tensor_mixin_base import _TensorMixinBase


class _TensorExtra:
    def __init__(
        self,
        *,
        tensor: Tensor,
        time_dim_axis=NotSpecified,
        available_for_inference=True,
        batch=None,
        beam=None,
        control_flow_ctx=None,
    ):
        """
        :param tensor:
        :param int|None|NotSpecified time_dim_axis: where we have the time dim axis, after we added the batch-dim.
            this is often 1. however, can be None if there is no time-dim.
        :param bool available_for_inference: e.g. the extern data "classes" is usually not available for inference
        :param BatchInfo|None batch:
        :param SearchBeam|None beam: the batch-dim could be extended by a beam-size,
            such that it represents the merged dims [batch, beam_size].
        :param ControlFlowContext|None control_flow_ctx:
        """
        self.tensor = tensor
        if beam and batch:
            assert batch.beam == beam
        self.batch = batch
        del batch
        self.beam = beam
        del beam
        self.control_flow_ctx = control_flow_ctx
        self.available_for_inference = available_for_inference

        if time_dim_axis is NotSpecified:
            # just like feature_dim_axis, time_dim_axis is just None with version>=2, without any default behavior.
            if self.tensor.version >= 2:
                time_dim_axis = None
        elif time_dim_axis is None:
            pass
        elif isinstance(time_dim_axis, int):
            assert self.tensor.version == 1
            assert 0 <= time_dim_axis < self.tensor.batch_ndim
        else:
            raise TypeError(f"unexpected time_dim_axis type {type(time_dim_axis)}")
        self.time_dim_axis = time_dim_axis

    def __getstate__(self):
        d = vars(self)
        d["batch"] = None  # BatchInfo pickling not supported
        return d


class _TensorMixin(_TensorMixinBase):
    @staticmethod
    def from_tensor(x) -> Tensor:
        """
        :param tf.Tensor x:
        """
        assert x.get_shape().is_fully_defined()
        x_shape = x.get_shape().as_list()
        return _t.Tensor(name=str(x.op.name), shape=x_shape, batch_dim_axis=None, dtype=x.dtype.name, placeholder=x)

    @staticmethod
    def template_from_constant(
        x, name, dtype=None, shape=None, with_batch_dim=False, sparse_dim=None, feature_dim=None
    ) -> Tensor:
        """
        :param int|float|bool|numpy.ndarray x: not actually assigned to the returned Data, just for the shape and dtype
        :param str name:
        :param str|None dtype:
        :param list[Dim|int]|tuple[Dim|int]|None shape: for verification, and defining dim tags.
          might also already include the batch-dim. (Then with_batch_dim is ignored.)
        :param bool with_batch_dim:
        :param Dim|None sparse_dim:
        :param Dim|None feature_dim:
        :return: data template
        """
        import numpy

        if dtype is None:
            if isinstance(x, bool):
                dtype = "bool"
            elif isinstance(x, int):
                dtype = "int32"
            elif isinstance(x, float):
                dtype = "float32"
            elif isinstance(x, numpy.ndarray):
                dtype = str(x.dtype)
            else:
                raise TypeError("%r: cannot handle value %r of type %r" % (name, x, type(x)))
        shape_ = x.shape if isinstance(x, numpy.ndarray) else ()
        if shape is not None:
            if len(shape) > len(shape_) == 0:
                pass  # Scalar given, complex shape wanted. Allow this.
            else:
                assert len(shape) == len(shape_), "%r: shape does not match in ndim, %r vs %r" % (name, shape, shape_)
        else:
            shape = shape_
        dim_tags = []
        for i, d in enumerate(shape):
            d_ = shape_[i] if len(shape_) > 0 else None
            if isinstance(d, Dim):
                if len(shape_) > 0:
                    assert d.dimension == d_
            elif isinstance(d, int):
                if len(shape_) > 0:
                    assert d == d_
                d = Dim(
                    kind=Dim.Types.Spatial if i < len(shape) - 1 else Dim.Types.Feature,
                    description="%s:static:%i" % (name, i),
                    auto_generated=True,
                    dimension=d,
                )
            else:
                raise TypeError("%r shape[%i] invalid type %r in shape %r" % (name, i, type(d), shape))
            dim_tags.append(d)
        if with_batch_dim and batch_dim not in dim_tags:
            dim_tags.insert(0, batch_dim)
        return _t.Tensor(name=name, dim_tags=dim_tags, dtype=dtype, sparse_dim=sparse_dim, feature_dim=feature_dim)

    def _handle_extra_kwargs(
        self,
        *,
        shape=None,
        sparse=None,
        dim=NotSpecified,
        batch_dim_axis=NotSpecified,
        dim_tags=None,
        placeholder=None,
        size_placeholder=None,
        auto_create_placeholders=False,
        vocab=None,
        same_dim_tags_as=None,
        **kwargs,
    ):
        """
        :param shape:
        :param sparse:
        :param dim:
        :param batch_dim_axis:
        :param Sequence[Dim]|None dim_tags:
            If tuple/list, this specifies the whole (batch) shape.
        :param tf.Tensor|None placeholder: with added batch-dim
        :param dict[int,tf.Tensor]|None size_placeholder: for every dynamic dim, this will describe the size.
            The keys are the axes but counted without batch-dim, and the value is the size.
            The size is always a tensor of shape (batch,), i.e. the size can be different for each sequence in a batch.
            This is the old deprecated way. Now this is all part of :class:`Dim`.
        :param bool auto_create_placeholders: This will create a tf.placeholder.
            This is deprecated. Rather, the placeholder should be created outside and passed in.
        :param str|dict[str]|returnn.datasets.util.vocabulary.Vocabulary|None vocab: vocab of the feature dim
            or sparse dim.
            This is deprecated. Rather, the vocab is part of the :class:`Dim`.
        :param dict[int|str,Dim]|None same_dim_tags_as: will mark our dimension tags to be the same
        """
        assert isinstance(self, _t.Tensor)
        shape, sparse, dim, batch_dim_axis, dim_tags  # noqa  # unused here, handled in infer_dim_tags

        if vocab is not None:
            from returnn.datasets.util.vocabulary import Vocabulary

            if isinstance(vocab, str):
                vocab = Vocabulary(vocab)
            elif isinstance(vocab, dict):
                vocab = Vocabulary.create_vocab(**vocab)
            assert isinstance(vocab, Vocabulary)
            assert self.sparse, "%s should represent indices of %s" % (self, vocab)
            assert self.dim == vocab.num_labels, "%s dims do not match with vocab %s" % (self, vocab)
            self.sparse_dim.vocab = vocab

        if kwargs:
            self._extra = _TensorExtra(tensor=self, **kwargs)

        # The size_placeholder is for each variable length dimension in shape, i.e. excluding the batch-dim.
        if size_placeholder:
            self.size_placeholder = size_placeholder

        if same_dim_tags_as:
            for _axis, _dim_tag in sorted(same_dim_tags_as.items()):
                _axis = self.get_axis_from_description(_axis)
                assert isinstance(_dim_tag, Dim)
                base_tag = self._dims[_axis]
                if base_tag != _dim_tag:
                    base_tag.declare_same_as(_dim_tag)
                self._dims = self._dims[:_axis] + (_dim_tag,) + self._dims[_axis + 1 :]

        if placeholder is not None:
            self.raw_tensor = placeholder
        elif auto_create_placeholders:
            # Need to import backend here directly, as we have not assigned the tensor yet,
            # and thus cannot infer it from its type.
            # This option auto_create_placeholders is anyway really only for TensorFlow
            # and should not be used in other backends,
            # even in graph-based backends.
            # Rather, the logic to create placeholders should be done elsewhere.
            # noinspection PyProtectedMember
            from returnn.tf.frontend_low_level._backend import TFBackend

            self.raw_tensor = TFBackend.create_placeholder_raw(self)
            # Do that after same_dim_tags_as handling.
            _auto_create_size_placeholders_on_dim_tags(name=self.name, dim_tags=self._dims)

        self._adapt_batch_consistent_dim_tags()  # TODO where to move this? not needed in general...
        self.sanity_check(assume_complete=False)  # TODO still needed?

    # potentially replaced by native code
    @property
    def _raw_backend(self) -> Optional[Type[Backend]]:
        """
        :return: the backend for the raw tensor
        """
        # noinspection PyProtectedMember,PyShadowingNames
        import returnn.frontend._backend as _backend_api

        if self._raw_tensor is None:
            return None
        return _backend_api.get_backend_by_raw_tensor_type(type(self._raw_tensor))

    @property
    def control_flow_ctx(self) -> Optional[ControlFlowContext]:
        """
        :return: control flow ctx (graph-based)
        """
        if not self._extra:
            return None
        return self._extra.control_flow_ctx

    @control_flow_ctx.setter
    def control_flow_ctx(self, value: Optional[ControlFlowContext]):
        if value == self.control_flow_ctx:
            return
        self._make_extra().control_flow_ctx = value

    @property
    def available_for_inference(self) -> bool:
        """
        :return: available for inference
        """
        if not self._extra:
            return True
        return self._extra.available_for_inference

    @available_for_inference.setter
    def available_for_inference(self, value: bool):
        if value == self.available_for_inference:
            return
        self._make_extra().available_for_inference = value

    def _make_extra(self: Tensor) -> _TensorExtra:
        if not self._extra:
            self._extra = _TensorExtra(tensor=self)
        return self._extra

    def sanity_check(self, ignore_placeholder=False, assume_complete=True):
        """
        Performs some sanity checks on self, and raises exceptions if something is not sane.

        :param bool ignore_placeholder:
        :param bool assume_complete:
        """
        special_axes_dict = {"time_dim_axis": self.time_dim_axis, "feature_dim_axis": self.feature_dim_axis}
        batch_dim_axis = self.batch_dim_axis
        batch_ndim = self.batch_ndim
        for axis_name, axis in special_axes_dict.items():
            assert axis is None or 0 <= axis < batch_ndim, "%s: axis %s (%i) invalid" % (self, axis_name, axis)
        if batch_dim_axis is not None:
            for axis_name, axis in special_axes_dict.items():
                assert axis != batch_dim_axis, "%s: axis %s (%i) must be different from batch_dim_axis (%i)" % (
                    self,
                    axis_name,
                    axis,
                    batch_dim_axis,
                )
        if self.sparse_dim is not None:
            assert special_axes_dict["feature_dim_axis"] is None, (
                "%s: If sparse, there cannot be a feature dim axis." % self
            )
        for axis, tag in enumerate(self._dims):
            if tag.is_batch_dim():
                assert axis == batch_dim_axis, "%s: invalid %s" % (self, tag)
                continue  # further checks will assume not batch
            # Note: tag.kind (feature or spatial) is independent from self.feature_dim_axis.
            if tag.batch and self.batch:
                assert tag.batch == self.batch or self.batch.is_broadcast()
            if tag.dyn_size_ext:
                assert tag.dyn_size_ext.dtype in {"int32", "int64"}
                if tag.dyn_size_ext.have_batch_axis():
                    assert tag.batch == tag.dyn_size_ext.batch
                # Sanity check on dyn_size_ext should already have been done earlier.
        if not ignore_placeholder and self._raw_tensor is not None:
            # Note: We could just call self.placeholder.set_shape.
            # However, we are more explicit.
            # We assume that the placeholder has already a known shape, and error otherwise.
            backend = self._raw_backend
            raw_shape = backend.get_known_shape_raw(self._raw_tensor)
            assert len(raw_shape) == batch_ndim, f"Mismatching shape ndim: Raw tensor {raw_shape} vs Tensor {self}"
            for i in range(batch_ndim):
                if self._dims[i].dimension is None:
                    continue  # we allow anything in the placeholder
                if raw_shape[i] != self._dims[i].dimension:
                    raise Exception(
                        f"Mismatching shape: Raw tensor {raw_shape} vs Tensor {self};\n"
                        + backend.format_graph_output(self._raw_tensor, max_depth=3)
                    )
            backend.set_known_shape_raw(self._raw_tensor, self.batch_shape)
            assert backend.get_dtype_name_raw(self._raw_tensor) == self.dtype, (
                f"{self} dtype {self.dtype} does not match "
                f"raw tensor dtype {backend.get_dtype_name_raw(self._raw_tensor)}"
            )
        if assume_complete:
            for tag in self._dims:
                if tag.is_batch_dim():
                    continue
                if tag.is_dynamic():
                    assert tag.dyn_size_ext, "%s sanity_check: dynamic dim %s undefined" % (self, tag)
                    if not ignore_placeholder:
                        if tag.dyn_size_ext.placeholder is None:
                            tag.complete_dyn_size()
                        if self.placeholder is not None:
                            assert (
                                tag.dyn_size_ext.placeholder is not None
                            ), "%s sanity_check: dynamic dim %s value unknown" % (self, tag)
                assert tag.is_dim_known()

    def get_runtime_sanity_check_op(self: Tensor):
        """
        :return: op which does a couple of runtime sanity checks on the placeholder
        :rtype: tensorflow.Operation|Any
        """
        assert self._raw_tensor is not None
        return self._raw_backend.runtime_sanity_checks(self)

    def verify_out_shape(self, out_shape, allow_missing_implicit_dims=False):
        """
        Verifies that ``out_shape`` matches our shape, i.e. specifically the dim tags.
          https://github.com/rwth-i6/returnn/issues/706
        Throws an exception if this is not the case.

        :param set[Dim|_MarkedDim]|tuple|list out_shape:
          It must be a set, with the only exception when it is empty (then it doesn't matter).
          See :func:`dim_tags_set`.
        :param bool allow_missing_implicit_dims:
        """
        actual_dims_str = "{%s}" % ", ".join(
            [str(d) for d in list(self.dim_tags) + sorted(self.dim_tags_set_implicit_only_wrapped)]
        )  # noqa
        expected_dims_str = "{%s}" % ", ".join([str(d) for d in sorted(out_shape)])
        self_dim_tags = self.dim_tags_set_implicit
        self_dim_tags_implicit_only = self.dim_tags_set_implicit_only_wrapped
        if not out_shape:  # allow also empty list or empty tuple
            if self_dim_tags:
                raise VerifyOutShapeException(
                    "%s verify_out_shape:\n" % self
                    + "Actual dims: %s\nExpected empty out_shape: %s" % (actual_dims_str, expected_dims_str)
                )
            return
        if not isinstance(out_shape, set):
            # out_shape is not empty (tested above), so must be a set
            raise TypeError("%s verify_out_shape: expects a set but got %s" % (self, type(out_shape)))
        remaining = set(self_dim_tags)
        for dim in out_shape:
            if isinstance(dim, Dim):
                dim_tag = dim
            elif isinstance(dim, _m.ImplicitDim):
                dim_tag = dim.tag
                if dim not in self_dim_tags_implicit_only:
                    raise VerifyOutShapeException(
                        (
                            "%s verify_out_shape:\n"
                            "Actual dims: %s\n"
                            "Expected out_shape: %s\n"
                            "%s is not an implicit dim in self"
                        )
                        % (self, actual_dims_str, expected_dims_str, dim)
                    )
            elif isinstance(dim, _m.OptionalDim):
                dim_tag = dim.tag
                if dim_tag not in remaining:
                    continue
            else:
                raise TypeError(
                    "%s verify_out_shape with out_shape %s: expect dim tags but got %s" % (self, out_shape, type(dim))
                )
            if dim_tag not in remaining:
                if (
                    dim_tag in self_dim_tags
                ):  # can happen e.g. if specified once as implicit dim and then also as explicit
                    raise VerifyOutShapeException(
                        "%s verify_out_shape does not match:\n" % self
                        + "Actual dims: %s\nExpected out_shape: %s\n" % (actual_dims_str, expected_dims_str)
                        + "Dim %s multiple times in out_shape" % dim
                    )
                raise VerifyOutShapeException(
                    "%s verify_out_shape:\n" % self
                    + "Actual dims: %s\nExpected out_shape: %s\n" % (actual_dims_str, expected_dims_str)
                    + "Dim %s not in self" % dim
                )
            remaining.discard(dim_tag)
        if remaining:
            if allow_missing_implicit_dims and remaining.issubset(self.dim_tags_set_implicit_only):
                pass  # ok
            else:
                raise VerifyOutShapeException(
                    "%s verify_out_shape missing dims:\n" % self
                    + "Actual dims: %s\nExpected out_shape: %s\n" % (actual_dims_str, expected_dims_str)
                    + "Missing dims: %s" % ", ".join(map(str, sorted(remaining)))
                )

    def get_placeholder_kwargs(self, with_batch=True):
        """
        :param bool with_batch:
        :return: kwargs for tf.compat.v1.placeholder
        :rtype: dict[str]
        """
        return dict(name=self.name, dtype=self.dtype, shape=self.batch_shape if with_batch else self.shape)

    def get_axes_with_size(self):
        """
        :return: list of axes which can vary in size for each entry of the batch-dim, e.g. the time-dim-axis.
          The axis index is counted without the batch-dim.
        :rtype: list[int]
        """
        return [i for (i, dim) in enumerate(self.shape) if dim is None]

    # Note that the logic here is replicated for a native copy() and copy_template(),
    # so keep any changes in sync.
    def get_kwargs(self, *, include_special_axes=True):
        """
        :param bool include_special_axes: whether to include time and feature special axis marker
        :return: relevant attrib items for copying
        :rtype: dict[str]
        """
        keys = ["name", "dims", "dtype"]
        if include_special_axes:
            if self.version <= 1 and self.time_dim_axis_or_unspecified is not NotSpecified:
                keys += ["time_dim_axis"]
            if self.feature_dim_axis_or_unspecified is not NotSpecified:
                keys += ["feature_dim_axis"]
        if self.sparse_dim:
            # Sparse is False by default.
            # And the dim is inferred from the feature dim, or otherwise does not make sense.
            keys += ["sparse_dim"]
        if self.version == 1:  # version 2 is default, and there is only v1 or v2
            keys += ["version"]
        if self._extra:
            if self.batch is not None:
                keys += ["batch"]
            if self.beam is not None:
                keys += ["beam"]
            if self.control_flow_ctx:
                keys += ["control_flow_ctx"]
            if not self.available_for_inference:
                keys += ["available_for_inference"]
        return {key: getattr(self, key) for key in keys}

    def get_description(self, with_name=True, with_placeholder=False, catch_exceptions=False):
        """
        :param bool with_name:
        :param bool with_placeholder:
        :param bool catch_exceptions:
        :return: description of self. also used for __repr__
        :rtype: str
        """
        # Avoid redundant information (most information is covered in batch_shape_meta).
        # Also try to avoid confusion (e.g. `shape` vs `batch_shape`).
        keys = []
        if self.sparse:
            keys.append("dtype")
            keys.append("sparse_dim")
        else:
            if self.dtype != "float32":
                keys.append("dtype")
        if with_placeholder:
            keys.append("placeholder")
        if not self.available_for_inference:
            keys.append("available_for_inference")
        if self.beam is not None:
            # With batch, it should be contained already in batch_shape_meta (if everything is correct).
            # We anyway add it, in case sth is incorrect or incomplete.
            if not self.batch or self.batch.beam != self.beam:
                keys.append("beam")
        args = []
        if with_name:
            name = getattr(self, "name", None)
            args += [repr(name) if name else "<undefined>"]
        try:
            batch_shape_meta = "[%s]" % ",".join(self.get_batch_axes_short_description())
        except Exception as exc:
            if catch_exceptions:
                batch_shape_meta = "<!%s: %s>" % (type(exc).__name__, exc)
            else:
                raise
        args += [batch_shape_meta]
        for key in keys:
            try:
                value_repr = repr(getattr(self, key))
            except Exception as exc:
                if catch_exceptions:
                    value_repr = "<!%s: %s>" % (type(exc).__name__, exc)
                else:
                    raise
            args += ["%s=%s" % (key, value_repr)]
        if self.control_flow_ctx:
            try:
                value_repr = self.control_flow_ctx.repr_inner()
            except Exception as exc:
                if catch_exceptions:
                    value_repr = "<!%s: %s>" % (type(exc).__name__, exc)
                else:
                    raise
            args += ["ctx=" + value_repr]
        return "Tensor{%s}" % ", ".join(args)

    def get_batch_axes_short_description(self, special_axes=True):
        """
        :param bool special_axes: special markers for old-style time_dim_axis and feature_dim_axis
        :rtype: list[str]
        """
        res = []
        for axis, dim_tag in enumerate(self.dim_tags):
            descriptions = []
            if axis == self.batch_dim_axis:
                if self.batch:
                    descriptions.append(self.batch.short_repr())
                else:
                    descriptions.append("B?")
            if special_axes:
                if axis == self.time_dim_axis:
                    descriptions.append("T")
                if axis == self.feature_dim_axis:
                    descriptions.append("F")
            if self.batch_shape[axis] is None:
                if axis == self.batch_dim_axis:
                    pass  # expected
                else:
                    descriptions.append(dim_tag.short_repr())
            elif axis != self.batch_dim_axis or not self.batch:
                descriptions.append(dim_tag.short_repr())
            res.append("|".join(descriptions) or "?")
        return res

    def get_compare_key(self):
        """
        :return: some key which can be used for compare functions, i.e. such that
          cmp(get_compare_key(self), get_compare_key(other)) == cmp(self, other),
          i.e. we define some order by that.
          Note that this order is not totally fixed, and might change.
        :rtype: object
        """
        return (
            self.dtype,
            self.shape,
            self.batch_dim_axis,
            self.feature_dim_axis,
            self.time_dim_axis,
            self.dim_tags,
            self.batch,
            self.beam,
        )

    def __repr__(self):
        return self.get_description(catch_exceptions=True)

    def __hash__(self):
        return id(self)

    def _sis_hash(self):
        if self.raw_tensor is not None and hasattr(self.raw_tensor, "_sis_hash"):
            # noinspection PyProtectedMember
            return self.raw_tensor._sis_hash()

        from sisyphus.hash import sis_hash_helper  # noqa

        return sis_hash_helper(self.get_kwargs())

    def __getstate__(self):
        d = {k: getattr(self, k) for k in self.__slots__}
        d["_raw_tensor"] = None  # do not store the TF tensors
        return d

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def reset(self: Tensor):
        """
        Reset raw_tensor and batch info.
        """
        self._raw_tensor = None
        self.batch = None

    def _adapt_batch_consistent_dim_tags(self):
        if not self._extra:
            return
        if not self.batch:  # uninitialized or not relevant for this framework
            return
        dims = tuple(tag.get_for_batch_ctx(batch=self.batch, ctx=self.control_flow_ctx) for tag in self._dims)
        assert all(dims)
        dims: Tuple[Dim, ...]
        self._dims = dims

    # Note that this has a native implementation (_native tensor_copy).
    def copy(self, name: Optional[str] = None) -> _t.Tensor:
        """
        :param name: if given, will overwrite this name
        :return: copy of myself, using self.get_kwargs(), and with placeholder and size_placeholder
        """
        data = _t.Tensor(**self.get_kwargs())
        data._raw_tensor = self._raw_tensor
        if name:
            data.name = name
        return data

    def copy_as_batch_major(self) -> _t.Tensor:
        """
        :return: copy of myself with batch_dim_axis == 0
        """
        return self.copy_with_batch_dim_axis(0)

    def copy_as_time_major(self) -> _t.Tensor:
        """
        :return: copy of myself with time_dim_axis == 0
        """
        assert self.time_dim_axis is not None
        return self.copy_with_time_dim_axis(0)

    def copy_with_batch_dim_axis(self, batch_dim_axis) -> _t.Tensor:
        """
        :param int batch_dim_axis:
        :return: copy of myself with specific batch_dim_axis
        """
        assert self.batch_dim_axis is not None
        return self.copy_move_axis(self.batch_dim_axis, batch_dim_axis)

    def copy_with_time_dim_axis(self, time_dim_axis) -> _t.Tensor:
        """
        :param int time_dim_axis:
        :return: copy of myself with specific time_dim_axis
        """
        assert self.time_dim_axis is not None
        return self.copy_move_axis(self.time_dim_axis, time_dim_axis)

    def copy_transpose(self: Tensor, perm: Sequence[Union[int, Dim]], *, allow_int: bool = True) -> _t.Tensor:
        """
        :param perm: permutation of the axes. Maps the new axes to the old axes
        :param allow_int: allow int as axis, otherwise only :class:`Dim`
        :return: copy of myself with permuted axes
        """
        assert len(perm) == len(self._dims), f"{self}: invalid perm {perm!r} length"
        if not perm:
            return self.copy()
        if allow_int and isinstance(perm[0], int):
            assert all(isinstance(a, int) for a in perm), f"{self}: invalid perm {perm!r} types"
            assert set(perm) == set(range(len(perm))), f"{self}: invalid perm {perm!r}"
            return self._copy_compatible_to_dims_with_perm([self._dims[i] for i in perm], perm)
        else:
            assert all(isinstance(a, Dim) for a in perm), f"{self}: invalid perm {perm!r} types"
            return self.copy_compatible_to_dims(perm)

    def copy_move_axis(self, old_axis, new_axis) -> _t.Tensor:
        """
        :param int old_axis: counted with batch-dim
        :param int new_axis: counted with batch-dim
        :return: copy of myself with moved axis (see :func:`move_axis`)
        """
        if old_axis < 0:
            old_axis += self.batch_ndim
            assert old_axis >= 0
        assert 0 <= old_axis < self.batch_ndim
        if new_axis < 0:
            new_axis += self.batch_ndim
            assert new_axis >= 0
        assert 0 <= new_axis < self.batch_ndim
        if old_axis == new_axis:
            return self.copy()

        perm = list(range(self.batch_ndim))
        old = perm.pop(old_axis)
        perm.insert(new_axis, old)
        return self.copy_transpose(perm)

    def copy_swap_axes(self, axis1, axis2) -> _t.Tensor:
        """
        Like :func:`Tensor.copy_move_axis`, but keeps all other axes unchanged.
        :param int axis1: counted with batch-dim
        :param int axis2: counted with batch-dim
        :return: copy of myself with moved axis (see :func:`swapaxes`)
        """
        if axis1 < 0:
            axis1 += self.batch_ndim
        assert 0 <= axis1 < self.batch_ndim
        if axis2 < 0:
            axis2 += self.batch_ndim
        assert 0 <= axis2 < self.batch_ndim
        if axis1 == axis2:
            return self.copy()

        perm = list(range(self.batch_ndim))
        perm[axis1], perm[axis2] = perm[axis2], perm[axis1]
        return self.copy_transpose(perm)

    def copy_as_bt_or_tb_major(self) -> _t.Tensor:
        """
        :return: copy of myself in batch-time-major or time-batch-major
        """
        assert self.have_batch_axis() and self.have_time_axis()
        if self.batch_dim_axis == 0:
            return self.copy_with_time_dim_axis(1)
        if self.time_dim_axis == 0:
            return self.copy_with_batch_dim_axis(1)
        if self.batch_dim_axis > self.time_dim_axis:
            return self.copy_as_time_major().copy_as_bt_or_tb_major()
        return self.copy_as_batch_major().copy_as_bt_or_tb_major()

    def copy_with_feature_dim_axis(self, feature_dim_axis) -> _t.Tensor:
        """
        :param int feature_dim_axis: can also be negative
        :return: copy of myself with specific feature dim axis
        """
        assert self.feature_dim_axis is not None
        return self.copy_move_axis(self.feature_dim_axis, feature_dim_axis)

    def copy_as_batch_feature_major(self) -> _t.Tensor:
        """
        :return: copy of self with batch_dim_axis == 0 and feature_dim_axis == 1
        """
        assert self.batch_dim_axis is not None
        assert self.feature_dim_axis is not None
        data = self.copy_as_batch_major()
        data = data.copy_with_feature_dim_axis(1)
        return data

    def copy_as_time_batch_major(self) -> _t.Tensor:
        """
        :return: copy of self with batch_dim_axis == 1 and time_dim_axis == 0
        """
        assert self.have_batch_axis() and self.have_time_axis()
        data = self.copy_as_bt_or_tb_major()
        if data.time_dim_axis == 1:
            data = data.copy_move_axis(0, 1)
        return data

    def copy_as_batch_spatial_major(self) -> _t.Tensor:
        """
        :return: copy with batch_dim_axis == 0, then all dynamic axes, then any other spatial axes, last feature axis
        """
        data = self.copy_as_batch_major()
        if data.feature_dim_axis is not None:
            data = data.copy_with_feature_last()
        if data.size_placeholder:
            for i, (j, size) in enumerate(sorted(data.size_placeholder.items())):
                data = data.copy_move_axis(data.get_batch_axis(j), i + 1)
        if data.feature_dim_axis is not None:
            assert data.feature_dim_axis == data.batch_ndim - 1
            # Maybe reset feature_dim_axis to unspecified.
            if data.feature_dim_axis_or_unspecified is not NotSpecified:
                if data._default_feature_dim_axis() == data.feature_dim_axis:
                    data.feature_dim_axis = NotSpecified
        return data

    def copy_with_feature_last(self) -> _t.Tensor:
        """
        :return: copy of self with feature_dim_axis being the very last axis
        """
        assert self.feature_dim_axis is not None
        return self.copy_with_feature_dim_axis(-1)

    def copy_add_batch_dim(self, batch_dim_axis, batch=None, dim_tag=None) -> _t.Tensor:
        """
        :param int batch_dim_axis:
        :param BatchInfo|None batch:
        :param Dim|None dim_tag:
        :return: copy of myself with added batch-dim
        """
        if self.have_batch_axis():
            raise Exception(
                f"{self} copy_add_batch_dim: already has batch-dim at axis {self.batch_dim_axis},"
                f" cannot add tag {dim_tag!r}"
            )
        assert self.batch_dim_axis is None
        if batch_dim_axis < 0:
            assert batch_dim_axis + self.batch_ndim + 1 >= 0
            batch_dim_axis += self.batch_ndim + 1
        assert 0 <= batch_dim_axis <= self.batch_ndim
        data_opts = self.get_kwargs(include_special_axes=False)
        placeholder = self.placeholder
        if placeholder is not None:
            backend = self._raw_backend
            placeholder = backend.expand_dims_raw(placeholder, batch_dim_axis)
            if batch:
                batch_dim_ = batch.dim
            elif dim_tag:
                if dim_tag.dyn_size_ext:
                    assert dim_tag.dyn_size_ext.dims == ()
                    assert dim_tag.dyn_size_ext.raw_tensor is not None
                    batch_dim_ = dim_tag.dyn_size_ext.raw_tensor
                elif dim_tag.dimension:
                    batch_dim_ = dim_tag.dimension
                else:
                    raise Exception(f"{self} copy_add_batch_dim: unknown batch dim for {dim_tag!r}")
            else:
                raise Exception(f"{self} copy_add_batch_dim: unknown batch dim ")
            if not isinstance(batch_dim_, int) or batch_dim_ != 1:
                placeholder = backend.expand_raw(placeholder, batch_dim_axis, batch_dim_)
        dim_tags = list(self.dim_tags)
        if dim_tag:
            assert dim_tag.is_batch_dim()
            assert dim_tag.batch == batch
            if batch:
                assert dim_tag.dimension == batch.static_dim or dim_tag.dimension is None
        elif batch:
            dim_tag = batch.batch_dim_tag
        else:
            dim_tag = Dim(
                kind=Dim.Types.Batch, description="batch", dimension=batch.static_dim if batch else None, batch=batch
            )
        dim_tags.insert(batch_dim_axis, dim_tag)
        data_opts["dims"] = dim_tags
        if batch:
            data_opts["batch"] = batch
            data_opts["beam"] = batch.beam
        other_special_axes = self.get_special_axes_dict(counted_with_batch_dim=True, only_available=True)
        for k, a in other_special_axes.items():
            data_opts[k] = a if (a < batch_dim_axis) else (a + 1)
        return _t.Tensor(placeholder=placeholder, **data_opts)

    def copy_add_spatial_dim(self, spatial_dim_axis=None, dim=1, auto_time_dim_axis=True) -> _t.Tensor:
        """
        :param int|None spatial_dim_axis: counted with batch-dim. if there is no time-dim, this will be it.
        :param int|None dim:
        :param bool auto_time_dim_axis:
        :return: copy of myself with added spatial-dim
        """
        if dim is None:
            assert not self.placeholder
        dim_tag = Dim(description="added_spatial", dimension=dim, kind=Dim.Types.Spatial)
        if spatial_dim_axis is None:
            spatial_dim_axis = self.get_default_new_axis_for_dim_tag(dim_tag)
        v = self.copy_add_dim_by_tag(dim_tag, unbroadcast=True, axis=spatial_dim_axis)
        if auto_time_dim_axis and self.time_dim_axis is None:
            v.time_dim_axis = spatial_dim_axis
        return v

    def copy_add_feature_dim(self, axis=None) -> _t.Tensor:
        """
        :param int|None axis:
        :return: self with a new feature dim axis with dim 1.
          If there is an existing feature dim, the new feature dim will be added right after.
          If we are sparse, we don't add a feature dim, but it becomes a spatial dim instead.
        """
        if self.sparse:
            # By definition, we don't have a feature dim. We allow this though. We just make it a spatial axis.
            return self.copy_add_spatial_dim(spatial_dim_axis=axis)
        dim_tag = Dim(description="feature1", dimension=1, kind=Dim.Types.Feature)
        if axis is None:
            axis = self.get_default_new_axis_for_dim_tag(dim_tag)
        v = self.copy_add_dim_by_tag(dim_tag, axis=axis)
        if v.feature_dim_axis_or_unspecified is not NotSpecified:
            v.feature_dim_axis = NotSpecified
        if axis < 0:
            axis += v.batch_ndim
            assert axis >= 0
        assert 0 <= axis < v.batch_ndim
        if v.feature_dim_axis != axis:
            v.feature_dim_axis = axis
        return v

    def get_default_new_axis_for_dim_tag(self, dim_tag: Dim) -> int:
        """
        :param dim_tag:
        """
        if dim_tag.is_batch_dim():
            return 0
        # Note: if dim_tag is feature, but we are sparse, we just treat is as spatial, handled below.
        if dim_tag.is_feature_dim() and not self.sparse:
            if self.feature_dim_axis is not None:
                return self.feature_dim_axis + 1  # after existing feature-dim
            else:
                return self.batch_ndim  # at the end
        if dim_tag.is_dynamic() and self.get_dynamic_axes():
            return self.get_dynamic_axes()[-1] + 1  # after existing dynamic axis
        if dim_tag.is_spatial_dim() and self.get_spatial_batch_axes():
            return self.get_spatial_batch_axes()[-1] + 1  # after the existing spatial dim
        elif dim_tag.is_spatial_dim() and self.feature_dim_axis is not None:
            return self.feature_dim_axis  # add it before the feature dim
        else:
            return self.batch_ndim  # add it at the end

    def copy_add_dim_by_tag(self, dim_tag, unbroadcast=False, axis=None) -> _t.Tensor:
        """
        :param Dim dim_tag:
        :param bool unbroadcast: If True unbroadcast the newly added axis.
          Will infer the unbroadcast shape via :func:`Dim.get_dim_value`
        :param int|None axis:
        """
        assert dim_tag.can_be_used_as_dim()

        if axis is None:
            axis = self.get_default_new_axis_for_dim_tag(dim_tag=dim_tag)
        if axis < 0:
            axis += self.batch_ndim + 1
        assert 0 <= axis <= self.batch_ndim

        if dim_tag.is_batch_dim():
            if unbroadcast:
                return self.copy_add_batch_dim(batch_dim_axis=axis, batch=dim_tag.batch, dim_tag=dim_tag)
            else:
                if dim_tag.batch or self.batch:
                    from returnn.tf.util.data import BatchInfo

                    batch_info = BatchInfo.make_global_broadcast_batch_info()
                else:
                    batch_info = None
                if dim_tag and dim_tag.dimension == 1 and dim_tag.batch == batch_info:
                    pass  # keep it
                else:
                    dim_tag = Dim(
                        kind=Dim.Types.Batch,
                        description="batch-broadcast",
                        dimension=1,
                        batch=batch_info,
                        auto_generated=True,
                    )
                return self.copy_add_batch_dim(batch_dim_axis=axis, batch=batch_info, dim_tag=dim_tag)

        data_opts = self.get_kwargs()
        # Note: if dim_tag is feature, but we are sparse, we just make it spatial
        if self.sparse and dim_tag.is_feature_dim():
            dim_tag = dim_tag.copy(same_as_self=True, kind=Dim.Types.Spatial)
        if not unbroadcast and dim_tag.dimension != 1:
            dim_tag = Dim(
                kind=dim_tag.kind,
                description="%s_dummy_dim1" % (dim_tag.description or "unnamed"),
                dimension=1,
                auto_generated=True,
            )
        data_opts["dims"] = self._dims[:axis] + (dim_tag,) + self._dims[axis:]
        other_special_axes = self.get_special_axes_dict(counted_with_batch_dim=True, only_available=True)
        for k, a in other_special_axes.items():
            data_opts[k] = a if (a < axis) else (a + 1)
        if dim_tag.is_feature_dim() and self.feature_dim_axis is None:
            data_opts.pop("feature_dim_axis", None)  # fall back to default
        if dim_tag.is_spatial_dim() and self.time_dim_axis is None:
            data_opts.pop("time_dim_axis", None)  # fall back to default
        if self.placeholder is not None:
            backend = self._raw_backend
            placeholder = backend.expand_dims_raw(self.placeholder, axis)
            if dim_tag.dimension is None or dim_tag.dimension > 1:
                placeholder = backend.expand_raw(placeholder, axis, dim_tag.get_dim_value())
            data_opts["placeholder"] = placeholder
        return _t.Tensor(**data_opts)

    def copy_split_feature_dim(self, new_feature_dim) -> _t.Tensor:
        """
        Split it into (new_spatial_dim, new_feat_dim), in that order.
        This will increase the feature_dim_axis by one.

        :param int new_feature_dim: will be the new dim
        """
        assert not self.sparse
        assert self.feature_dim_axis is not None
        assert self.dim is not None
        assert self.dim % new_feature_dim == 0, "must be a multiple of the input feature dim"
        feature_dim_rem = self.dim // new_feature_dim
        new_feature_dim_axis = self.feature_dim_axis + 1
        data_opts = self.get_kwargs(include_special_axes=False)
        dim_tag_split_rem = Dim(
            kind=Dim.Types.Spatial,
            description="feature_split_rem_%i" % feature_dim_rem,
            auto_generated=True,
            dimension=feature_dim_rem,
        )
        dim_tag_new = Dim(
            kind=self.dim_tags[self.feature_dim_axis].kind,
            description="feature_split_new_%i" % new_feature_dim,
            auto_generated=True,
            dimension=new_feature_dim,
        )
        dim_tags = (
            self.dim_tags[: self.feature_dim_axis]
            + (dim_tag_split_rem, dim_tag_new)
            + self.dim_tags[self.feature_dim_axis + 1 :]
        )
        data_opts["dims"] = dim_tags
        other_special_axes = self.get_special_axes_dict(counted_with_batch_dim=True, only_available=True)
        other_special_axes.pop("feature_dim_axis", None)
        for k, a in other_special_axes.items():
            data_opts[k] = a if (a < new_feature_dim_axis) else (a + 1)
        if self.placeholder is not None:
            backend = self._raw_backend
            backend.set_known_shape_raw(self.placeholder, self.batch_shape)
            old_shape = backend.get_shape_tuple_raw(self.placeholder)
            new_shape = (
                old_shape[: self.feature_dim_axis]
                + (feature_dim_rem, new_feature_dim)
                + old_shape[self.feature_dim_axis + 1 :]
            )
            data_opts["placeholder"] = backend.reshape_raw(self.placeholder, new_shape)
        return _t.Tensor(**data_opts)

    def copy_extend_batch(self, batch) -> _t.Tensor:
        """
        Similar as copy_compatible_to with unbroadcast=True,
        we would possibly extend/expand our batch dim.
        See :class:`BatchInfo`.
        This assumes that we already have a batch dim
        (otherwise see :func:`copy_add_batch_dim`).

        This excludes any beam expansion, which is handled explicitly elsewhere
        (e.g. see :func:`copy_extend_with_beam`).

        :param BatchInfo batch:
        """
        assert self.have_batch_axis()
        assert self.batch, "%s: batch unset" % self
        data = self.copy()
        batch = batch.copy_set_beam(data.beam)
        if data.batch.beam != data.beam:  # Check for some buggy code.
            data.batch = data.batch.copy_set_beam(data.beam)
        if data.batch == batch:
            return data
        data.batch = batch
        self._adapt_batch_consistent_dim_tags()
        if self.placeholder is not None:
            assert self._raw_backend.is_tensorflow
            # This can only work if the batch is expanded.
            assert set(self.batch.virtual_dims).issubset(batch.virtual_dims)
            import tensorflow as tf
            from returnn.tf.util.basic import get_shape
            from returnn.util.basic import ensure_list_of_type
            from returnn.tf.util.data import BatchInfo

            with tf.name_scope("copy_extend_batch"):
                axis = self.batch_dim_axis
                x = self.placeholder
                shape = get_shape(x)
                # Only fixed dims supported/implemented (no packed dims).
                old_dims = ensure_list_of_type(self.batch.virtual_dims, BatchInfo.FixedDim)
                new_dims = ensure_list_of_type(batch.virtual_dims, BatchInfo.FixedDim)
                batch_broadcast_shape = []  # type: List[Union[tf.Tensor,int]]  # fill below
                ndim_batch_split = self.batch_ndim - 1 + len(new_dims)
                tiles = [1] * ndim_batch_split  # type: List[Union[tf.Tensor,int]]  # overwrite below
                old_idx = 0
                for new_idx, new_dim in enumerate(new_dims):
                    old_dim = old_dims[old_idx] if old_idx < len(old_dims) else None
                    if old_dim == new_dim:
                        batch_broadcast_shape.append(old_dim.size)
                        old_idx += 1
                    else:
                        batch_broadcast_shape.append(1)
                        tiles[axis + new_idx] = new_dim.size
                assert old_idx == len(old_dims)
                shape_batch_split = shape[:axis] + batch_broadcast_shape + shape[axis + 1 :]
                x = tf.reshape(x, shape_batch_split)
                x = tf.tile(x, tiles)
                shape = shape[:axis] + [batch.dim] + shape[axis + 1 :]
                x = tf.reshape(x, shape)
                data.placeholder = x
        return data

    def copy_compatible_to(
        self: Tensor,
        data: Tensor,
        add_dims=True,
        unbroadcast=False,
        except_feature=False,
        except_axis=None,
        check_sparse=True,
        check_dtype=True,
    ) -> Tensor:
        """
        :param data: other data which the returned tensor should be compatible to
          It would add any missing axes with a dim 1 axis for automatic broadcasting (with add_dims=True).
          It currently does not check whether existing dims match.
        :param bool add_dims: whether to add (broadcast, or unbroadcasted) dims. throws error if missing dim
        :param bool unbroadcast: if True, all added broadcast axes (axes with dim 1) will be tiled such that they match
        :param bool except_feature: if unbroadcast, do not unbroadcast the feature dim
        :param Dim|int|None except_axis: if unbroadcast, do not unbroadcast this axis
        :param bool check_sparse:
        :param bool check_dtype:
        :returns: Tensor, might add broadcast dimensions
        """
        assert not check_sparse or self.sparse == data.sparse
        assert not check_dtype or self.dtype == data.dtype
        v = self.copy()
        if v.have_batch_axis() and data.have_batch_axis() and v.batch and data.batch and v.batch != data.batch:
            v = v.copy_extend_batch(data.batch)
        v.sparse_dim = data.sparse_dim  # we will later reset it. this is to better count the axes (feature and spatial)
        if v.batch_dim_axis is not None and data.batch_dim_axis is None:
            raise ValueError("copy_compatible_to: self %r has batch-dim, but target data %r has not" % (self, data))
        if data.batch_ndim < v.batch_ndim:
            raise ValueError("copy_compatible_to: self %r already has more dims than target data %r" % (self, data))

        is_equal_opts = dict(
            allow_same_feature_dim=True,
            allow_same_spatial_dim=True,
            treat_feature_as_spatial=True,
            ignore_feature_dim=True,
        )
        mapped_axes = data.find_matching_dim_map(v, list(range(v.batch_ndim)), is_equal_opts)  # maps v -> data
        assert len(mapped_axes) == v.batch_ndim

        except_axis_int = (
            data.get_axis_from_description(except_axis, allow_int=True) if except_axis is not None else None
        )

        for target_axis in range(data.batch_ndim):
            new_v_axis = min(target_axis, v.batch_ndim)
            if target_axis not in mapped_axes.values():
                if not add_dims:
                    raise ValueError(
                        "%s.copy_compatible_to(%s) not allowed, axis %i (%s) not in source"
                        % (self, data, target_axis, data.dim_tags[target_axis])
                    )
                # Dim in data, but not in v
                unbroadcast_axis = (
                    unbroadcast
                    and not (except_feature and data.feature_dim_axis == target_axis)
                    and not (except_axis_int is not None and except_axis_int == target_axis)
                )
                v = v.copy_add_dim_by_tag(data.get_dim_tag(target_axis), axis=new_v_axis, unbroadcast=unbroadcast_axis)
                # Keep mapped_axes consistent
                mapped_axes = {v_ax + (1 if v_ax >= new_v_axis else 0): trg_ax for v_ax, trg_ax in mapped_axes.items()}
                mapped_axes[new_v_axis] = target_axis
            else:
                # Dim exists in both data and in v. Maybe order is wrong.
                matching_v_axes = [v_ax for v_ax, trg_ax in mapped_axes.items() if trg_ax == target_axis]
                assert len(matching_v_axes) == 1
                matching_v_axis = matching_v_axes[0]
                if target_axis != matching_v_axis:
                    # Order was wrong
                    v = v.copy_swap_axes(matching_v_axis, new_v_axis)
                    # Keep mapped_axes consistent
                    mapped_axes[matching_v_axis], mapped_axes[new_v_axis] = (
                        mapped_axes[new_v_axis],
                        mapped_axes[matching_v_axis],
                    )

        assert v.batch_ndim == data.batch_ndim
        assert all(mapped_axes[ax] == ax for ax in range(v.batch_ndim))

        if self.version == 1:
            # Ensure time_dim_axis and feature_dim_axis is same as in data
            assert v.batch_dim_axis == data.batch_dim_axis  # there is only at most one batch_dim_axis
            if v.time_dim_axis != data.time_dim_axis:
                v.time_dim_axis = NotSpecified
                if v.time_dim_axis != data.time_dim_axis:
                    v.time_dim_axis = data.time_dim_axis
            if v.feature_dim_axis != data.feature_dim_axis:
                v.feature_dim_axis = NotSpecified
                if v.feature_dim_axis != data.feature_dim_axis:
                    v.feature_dim_axis = data.feature_dim_axis
        else:
            if v.feature_dim_axis != data.feature_dim_axis:
                v.feature_dim_axis = data.feature_dim_axis

        # Reset sparse
        if self.sparse:
            v.feature_dim_axis = NotSpecified
            v.sparse_dim = self.sparse_dim  # reset

        v.sanity_check()
        return v

    # This func matches _native _permuteAndExtend logic.
    # This func has a native implementation (_native tensor_get_out_permutation_to_dims).
    def get_out_permutation_to_dims(self, dims: Sequence[Dim]) -> List[int]:
        """
        :param dims: superset of our dims
        :return: out_permutation list of len dims, where -1 means no match, and otherwise the axis in self.
            Thus, sorted(p for p in out_permutation if p >= 0) == range(len(self._dims)).
        """
        out_permutation: List[int] = []
        count = 0
        taken = [False] * len(self._dims)
        for dim in dims:
            candidates: List[int] = []  # axis in self
            for j in range(len(self._dims)):
                if taken[j]:
                    continue
                if dim is self._dims[j]:
                    candidates = [j]  # prefer that one
                    break
                if dim == self._dims[j]:
                    candidates.append(j)

            if not candidates:
                out_permutation.append(-1)
            elif len(candidates) == 1:
                out_permutation.append(candidates[0])
                taken[candidates[0]] = True
                count += 1
            else:
                max_match_priority_idx = None
                max_match_priority = None
                count_same_match_priority = 0
                for j in range(len(candidates)):
                    dim = self._dims[candidates[j]]
                    match_priority = dim.match_priority
                    if j > 0 and match_priority == max_match_priority:
                        count_same_match_priority += 1
                    if j == 0 or match_priority > max_match_priority:
                        max_match_priority = match_priority
                        count_same_match_priority = 1
                        max_match_priority_idx = j
                assert count_same_match_priority >= 1
                if count_same_match_priority > 1:
                    raise ValueError(
                        f"{self}: dim {dim} is ambiguous, from tensor dims {self._dims} and raw_tensor shape {dims}"
                    )
                out_permutation.append(candidates[max_match_priority_idx])
                taken[candidates[max_match_priority_idx]] = True
                count += 1

        assert count == len(self._dims)
        assert len(out_permutation) == len(dims)
        return out_permutation

    # This function has a native implementation (_native tensor_copy_compatible_to_dims).
    def copy_compatible_to_dims(self: _t.Tensor, dims: Sequence[Dim]) -> _t.Tensor:
        """
        Simpler variant of :func:`copy_compatible_to` which just takes a list of dims,
        and uses simple Dim equality.

        :param dims:
        :return: self with dims permuted and broadcast dims added
        """
        out_permutation = self.get_out_permutation_to_dims(dims)
        if out_permutation == list(range(len(self._dims))):
            return self.copy()
        return self._copy_compatible_to_dims_with_perm(dims, out_permutation)

    def _copy_compatible_to_dims_with_perm(self, dims: Sequence[Dim], out_permutation: Sequence[int]):
        raw_tensor = self._raw_tensor
        if raw_tensor is not None:
            backend = self._raw_backend
            raw_shape = backend.get_shape_tuple_raw(raw_tensor)
            raw_tensor = backend.transpose_raw(raw_tensor, [p for p in out_permutation if p >= 0])
            raw_tensor = backend.reshape_raw(raw_tensor, [raw_shape[p] if p >= 0 else 1 for p in out_permutation])
        out_dims = [
            (
                dims[i]
                if p >= 0
                else Dim(
                    kind=dims[i].kind,
                    description="%s_bc_dim1" % (dims[i].description or "unnamed"),
                    dimension=1,
                    auto_generated=True,
                )
            )
            for i, p in enumerate(out_permutation)
        ]
        kwargs = self.get_kwargs(include_special_axes=False)
        kwargs["dims"] = out_dims
        kwargs["raw_tensor"] = raw_tensor
        res = _t.Tensor(**kwargs)
        if self.version <= 1:
            if self.time_dim_axis is None:
                if res.time_dim_axis is not None:
                    res.time_dim_axis = None
            else:
                axis = out_permutation.index(self.time_dim_axis)
                assert axis >= 0
                if res.time_dim_axis != axis:
                    res.time_dim_axis = axis
        if self.feature_dim_axis is None:
            if res.feature_dim_axis is not None:
                res.feature_dim_axis = None
        else:
            axis = out_permutation.index(self.feature_dim_axis)
            assert axis >= 0
            if res.feature_dim_axis != axis:
                res.feature_dim_axis = axis
        return res

    # This function has a native implementation (_native tensor_copy_compatible_to_dims_raw).
    def copy_compatible_to_dims_raw(self: _t.Tensor, dims: Sequence[Dim]) -> _t.RawTensorType:
        """
        Simpler variant of :func:`copy_compatible_to` which just takes a list of dims,
        and uses simple Dim equality.
        This adds broadcast dims for any missing dims.

        :param dims:
        :return: raw tensor from self with dims permuted and broadcast dims added
        """
        raw_tensor = self._raw_tensor
        assert raw_tensor is not None, f"{self} copy_compatible_to_dims_raw: no raw tensor"
        out_permutation = self.get_out_permutation_to_dims(dims)
        if out_permutation == list(range(len(self._dims))):
            return raw_tensor
        backend = self._raw_backend
        raw_shape = backend.get_shape_tuple_raw(raw_tensor)
        raw_tensor = backend.transpose_raw(raw_tensor, [p for p in out_permutation if p >= 0])
        raw_tensor = backend.reshape_raw(raw_tensor, [raw_shape[p] if p >= 0 else 1 for p in out_permutation])
        return raw_tensor

    def copy_time_flattened(self) -> _t.Tensor:
        """
        :return: copy of myself where the time-axis is flattened away into the batch-dim-axis.
          See :func:`get_placeholder_time_flattened` and :func:`flatten_with_seq_len_mask for more details.
        """
        assert self.batch_dim_axis is not None
        assert self.time_dim_axis is not None
        data_opts = self.get_kwargs(include_special_axes=False)
        if self.placeholder is not None:
            data_opts["placeholder"] = self.get_placeholder_time_flattened()
        dim_tag = self.dim_tags[self.time_dim_axis]
        dim_tag = Dim(
            kind=Dim.Types.Spatial,
            description="%s_flattened" % (dim_tag.description or "unnamed"),
            auto_generated=True,
            dimension=None,
        )
        data_opts["dims"] = (dim_tag,) + tuple(
            tag for (i, tag) in enumerate(self.dim_tags) if i not in (self.batch_dim_axis, self.time_dim_axis)
        )
        data_opts["time_dim_axis"] = None
        data_opts.pop("feature_dim_axis", None)
        return _t.Tensor(**data_opts)

    def copy_extend_with_beam(self, beam) -> Tensor:
        """
        :param SearchBeam|None beam:
        :return: copy of myself where the batch-dim is extended/multiplied by beam_size, using tile_transposed
        """
        data = self.copy()
        if data.beam and data.beam == beam:
            return data
        assert data.beam is None, "incompatible beam (%r vs %r)" % (data.beam, beam)
        if beam is None:
            return data
        data.beam = beam
        assert data.batch
        data.batch = data.batch.copy_set_beam(beam)
        if data.placeholder is not None:
            assert data._raw_backend.is_tensorflow
            import tensorflow as tf
            from returnn.tf.util.basic import get_valid_scope_name_from_str, same_control_flow_ctx, tile_transposed

            with tf.name_scope("%s_data_extend_with_beam" % get_valid_scope_name_from_str(self.name)):
                with same_control_flow_ctx(data.placeholder):
                    data.placeholder = tile_transposed(
                        data.placeholder, axis=data.batch_dim_axis, multiples=beam.beam_size
                    )
                    setattr(data.placeholder, "_RETURNN_beam_expanded_base_data", self)
        data._adapt_batch_consistent_dim_tags()
        return data

    def copy_merge_into_batch(self, axes) -> Tensor:
        """
        :param list[int] axes: All axes to be merged into the batch axis.
          Must include the batch_dim_axis. The order is kept.
        :return: copy of myself where the the given axes are merged into the batch dim
        """
        assert self.batch
        assert self.batch_dim_axis in axes
        assert sorted(set(axes)) == sorted(axes)
        min_axis = min(axes)
        axes = list(axes)
        data = self.copy()
        if axes != list(range(min_axis, min_axis + len(axes))):
            rem_axes_start = list(range(min_axis))
            rem_axes_end = [a for a in range(min_axis, self.batch_ndim) if a not in axes]
            data = data.copy_transpose(rem_axes_start + axes + rem_axes_end)
            axes = list(range(min_axis, min_axis + len(axes)))
            assert data.batch_dim_axis in axes
        tensor = data.placeholder
        batch = data.batch
        data = data.copy_template()
        batch_idx = 0
        for axis in axes:
            if axis == data.batch_dim_axis:
                batch_idx = len(batch.virtual_dims)  # add all remaining axes behind
                continue
            batch = batch.copy_extend_with_padded_or_fixed_dim_tag(dim_tag=data.dim_tags[axis], new_dim_idx=batch_idx)
            batch_idx += 1
        for axis in reversed(sorted(axes)):
            if axis != data.batch_dim_axis:
                data = data.copy_template_excluding_axis(axis)
        data.batch = batch
        if tensor is not None:
            assert self._raw_backend.is_tensorflow
            import tensorflow as tf
            from returnn.tf.util.basic import get_shape

            shape = get_shape(tensor)
            tensor = tf.reshape(tensor, shape[:min_axis] + [-1] + shape[min_axis + len(axes) :])
            data.placeholder = tensor
        return data

    def copy_squeeze_axes(self, axes) -> Tensor:
        """
        :param list[int] axes: counted with batch dim
        :return: copy of myself, with squeezed axes
        """
        assert isinstance(axes, (list, tuple))
        assert all(self.batch_shape[axis] == 1 for axis in axes)
        assert all(0 <= axis < self.batch_ndim for axis in axes)
        if not axes:
            return self.copy()
        data_opts = self.get_kwargs(include_special_axes=False)
        if self._raw_tensor is not None:
            backend = self._raw_backend
            data_opts["raw_tensor"] = backend.squeeze_raw(self._raw_tensor, axes)
        data_opts["dims"] = [tag for (i, tag) in enumerate(self._dims) if i not in axes]
        if self.time_dim_axis is not None and self.time_dim_axis_or_unspecified is not NotSpecified:
            if self.time_dim_axis not in axes:
                data_opts["time_dim_axis"] = self.time_dim_axis - len(
                    [axis for axis in axes if axis < self.time_dim_axis]
                )
        if self.feature_dim_axis is not None and self.feature_dim_axis_or_unspecified is not NotSpecified:
            if self.feature_dim_axis not in axes:
                data_opts["feature_dim_axis"] = self.feature_dim_axis - len(
                    [axis for axis in axes if axis < self.feature_dim_axis]
                )
        return _t.Tensor(**data_opts)

    # Note that this has a native implementation (_native tensor_copy_template).
    def copy_template(self, name=None, *, dtype=None) -> _t.Tensor:
        """
        :param str|None name:
        :param str|None dtype:
        :return: copy of myself, using self.get_kwargs(), without placeholder
        """
        kwargs = self.get_kwargs()
        if name:
            kwargs["name"] = name
        if dtype:
            kwargs["dtype"] = dtype
        return _t.Tensor(**kwargs)

    def copy_template_dense(self, name=None, dtype=None) -> Tensor:
        """
        :param str|None name:
        :param str|None dtype:
        :return: copy of myself, using self.get_kwargs(), without placeholder
        """
        out = self.copy_template(name=name)
        if out.sparse:
            feat_dim = out.sparse_dim
            out.sparse = False
            out.dtype = "float32"
            out = out.copy_add_dim_by_tag(dim_tag=feat_dim, unbroadcast=True, axis=-1)
            out.feature_dim_axis = NotSpecified
            assert out.feature_dim_axis == out.batch_ndim - 1
        if dtype:
            out.dtype = dtype
        return out

    def copy_template_excluding_axis(self, exclude_axis, name=None) -> _t.Tensor:
        """
        :param int exclude_axis: axis to be removed.
        :param str|None name: if set, this will be the new name.
        :return: copy of myself excluding exclude_axis axis, without placeholder.
        """
        kwargs = self.get_kwargs(include_special_axes=False)
        if exclude_axis < 0:
            exclude_axis += self.batch_ndim
            assert exclude_axis >= 0
        assert 0 <= exclude_axis < self.batch_ndim
        if exclude_axis == self.feature_dim_axis:
            kwargs.pop("dim", None)
        other_special_axes = self.get_special_axes_dict(counted_with_batch_dim=True, only_available=True)
        for axis_name, axis in other_special_axes.items():
            if axis == exclude_axis:
                continue
            kwargs[axis_name] = axis if (axis < exclude_axis) else (axis - 1)
        new_dim_tags = list(self.dim_tags)
        del new_dim_tags[exclude_axis]
        kwargs["dims"] = new_dim_tags
        if name:
            kwargs["name"] = name
        return _t.Tensor(**kwargs)

    def copy_template_excluding_spatial_dim(self, spatial_axis_num, name=None) -> Tensor:
        """
        :param int spatial_axis_num: index in self.get_spatial_batch_axes()
        :param str|None name: if set, this will be the new name
        :return: copy of myself excluding the time-dimension without placeholder
        """
        spatial_axes = self.get_spatial_batch_axes()
        if spatial_axis_num < 0:
            spatial_axis_num += len(spatial_axes)
            assert spatial_axis_num >= 0
        assert 0 <= spatial_axis_num < len(spatial_axes)
        axis_to_exclude = spatial_axes[spatial_axis_num]
        return self.copy_template_excluding_axis(exclude_axis=axis_to_exclude, name=name)

    def copy_template_excluding_time_dim(self, name=None) -> _t.Tensor:
        """
        :param str|None name: if set, this will be the new name
        :return: copy of myself excluding the time-dimension without placeholder
        """
        assert self.time_dim_axis is not None
        return self.copy_template_excluding_axis(exclude_axis=self.time_dim_axis, name=name)

    def copy_template_adding_time_dim(self, name=None, time_dim_axis=0) -> _t.Tensor:
        """
        Adds a time-dim-axis.
        If a time-dim-axis already exists, it will anyway create this new one.

        :param str|None name: if set, this will be the new name
        :param int time_dim_axis: the new time-dim-axis index
        :return: copy of myself adding the time-dimension without placeholder
        """
        if time_dim_axis < 0:
            time_dim_axis += self.batch_ndim + 1
            assert time_dim_axis >= 0
        assert 0 <= time_dim_axis <= self.batch_ndim
        kwargs = self.get_kwargs(include_special_axes=False)
        dim_tag = Dim(kind=Dim.Types.Time, description="unknown_time", dimension=None, auto_generated=True)
        dim_tags = self.dim_tags[:time_dim_axis] + (dim_tag,) + self.dim_tags[time_dim_axis:]
        kwargs["dims"] = dim_tags
        other_special_axes = self.get_special_axes_dict(counted_with_batch_dim=True, only_available=True)
        other_special_axes.pop("time_dim_axis", None)
        for axis_name, axis in other_special_axes.items():
            kwargs[axis_name] = axis if (axis < time_dim_axis) else (axis + 1)
        kwargs["time_dim_axis"] = time_dim_axis
        if name:
            kwargs["name"] = name
        return _t.Tensor(**kwargs)

    def copy_template_replace_dim_tag(self, axis, new_dim_tag, name=None) -> _t.Tensor:
        """
        :param int axis:
        :param Dim new_dim_tag:
        :param str|None name: new name
        """
        assert new_dim_tag.can_be_used_as_dim()
        if axis < 0:
            assert axis + self.batch_ndim >= 0
            axis += self.batch_ndim
        assert 0 <= axis < self.batch_ndim
        opts = self.get_kwargs()
        if self.dim_tags[axis].is_batch_dim():
            opts.pop("batch", None)
        if new_dim_tag.is_batch_dim():
            if self.time_dim_axis == axis:
                opts.pop("time_dim_axis", None)
            if self.feature_dim_axis == axis:
                opts.pop("feature_dim_axis", None)
        dim_tags = self.dim_tags[:axis] + (new_dim_tag,) + self.dim_tags[axis + 1 :]
        opts["dims"] = dim_tags
        if self.feature_dim_axis_or_unspecified is not NotSpecified:
            if (
                self.feature_dim_axis == axis
                and self.dim_tags[axis].is_feature_dim()
                and not new_dim_tag.is_feature_dim()
            ):
                opts["feature_dim_axis"] = None
        if name:
            opts["name"] = name
        return _t.Tensor(**opts)

    def copy_template_replace_dim(self, axis, new_dim, new_size=None) -> _t.Tensor:
        """
        :param int axis:
        :param int|None new_dim:
        :param tf.Tensor|None new_size:
        """
        dim_tag = self.dim_tags[axis]
        if dim_tag.is_batch_dim():
            assert new_dim is None
            return self.copy_template()  # nothing to do
        dim_tag = Dim(
            kind=dim_tag.kind,
            description="%s_replaced" % (dim_tag.description or "unnamed"),
            auto_generated=True,
            dimension=new_dim,
            dyn_size=new_size,
        )
        return self.copy_template_replace_dim_tag(axis=axis, new_dim_tag=dim_tag)

    def copy_template_new_dim_tags(self, new_dim_tags, name=None, keep_special_axes=False) -> _t.Tensor:
        """
        :param list[Dim]|tuple[Dim] new_dim_tags:
        :param str|None name:
        :param bool keep_special_axes:
        """
        if keep_special_axes:
            assert len(new_dim_tags) == self.batch_ndim
        opts = self.get_kwargs(include_special_axes=keep_special_axes)
        opts["dims"] = new_dim_tags
        if name:
            opts["name"] = name
        return _t.Tensor(**opts)

    def copy_template_set_ctx(self, ctx) -> Tensor:
        """
        :param ControlFlowContext ctx:
        :return: new Tensor instance
        """
        kwargs = self.get_kwargs()
        kwargs["control_flow_ctx"] = ctx
        return _t.Tensor(**kwargs)

    def copy_template_unpack_batch(self) -> Tensor:
        """
        If the batch dim contains a :class:`BatchInfo.PackedDim`,
        unpack it and restore the data from before the packing.
        """
        assert self.have_batch_axis()
        assert self.batch, "%s: batch unset" % self
        data = self.copy()
        kwargs = self.get_kwargs(include_special_axes=False)
        from returnn.tf.util.data import BatchInfo

        dim_tags = []
        for dim_tag in data.dim_tags:
            if dim_tag.is_batch_dim() and dim_tag.batch and len(dim_tag.batch.virtual_dims) > 0:
                batch = dim_tag.batch
                new_batch_dim_tag = None
                for virtual_dim in batch.virtual_dims:
                    if isinstance(virtual_dim, BatchInfo.PackedDim):
                        dim_tags.append(virtual_dim.dim_tag)
                        batch = batch.copy_remove_dim(virtual_dim)
                    elif isinstance(virtual_dim, BatchInfo.GlobalBatchDim):
                        assert not new_batch_dim_tag
                        if batch is None or batch.is_global_batch():
                            new_batch_dim_tag = batch_dim  # reuse global batch dim
                        else:
                            new_batch_dim_tag = Dim(
                                kind=Dim.Types.Batch, description=dim_tag.description, dimension=None
                            )
                        dim_tags.append(new_batch_dim_tag)
                assert new_batch_dim_tag, "%s: batch info %r invalid" % (self, batch)
                new_batch_dim_tag.batch = batch
                kwargs["batch"] = batch
            else:
                dim_tags.append(dim_tag)

        kwargs["dims"] = dim_tags
        return _t.Tensor(**kwargs)

    def _get_variable_dim_pattern(self):
        """
        :return: tuple with bools specifying which dims of the shape (excluding batch-dim) are of variable length.
         e.g. (time,feature), shape=(None,128), this returns (True, False)
        :rtype: tuple[bool]
        """
        return tuple([dim is None for dim in self.shape])

    def _get_var_len_axes(self):
        return [i for (i, d) in enumerate(self._get_variable_dim_pattern()) if d]

    def matches_var_dim_pattern(self, other: _t.Tensor) -> bool:
        """
        :param other:
        :return: whether the variable-dims pattern matches,
          i.e. same variable dims (get_variable_dim_pattern), same time dim, excluding batch-dim.
          i.e. the size_placeholder should be compatible.
          (deprecated)
        """
        if self.time_dim_axis_excluding_batch != other.time_dim_axis_excluding_batch:
            return False
        return self._get_var_len_axes() == other._get_var_len_axes()

    @property
    def dim_tags(self) -> Tuple[Dim, ...]:
        """
        :return: dim tags, i.e. the shape. (alias for :func:`dims`)
        """
        return self._dims

    @property
    def shape(self) -> Tuple[Optional[int], ...]:
        """
        :return: shape *without* batch-dim. e.g. (time,feat) = (None,128)
            see also :func:`batch_shape` or `dims`
        """
        return tuple(tag.dimension for tag in self._dims if not tag.is_batch_dim())

    @shape.setter
    def shape(self, shape):
        """
        :param tuple[int|None] shape:
        """
        if tuple(shape) == self.shape:
            return
        raise Exception("%s: setting the shape is not allowed (new shape %s)" % (self, shape))

    @property
    def batch_shape(self) -> Tuple[Optional[int], ...]:
        """
        :return: shape with added batch-dim. e.g. (batch,time,feat) = (None,None,128)
        """
        return tuple(tag.dimension for tag in self.dim_tags)

    # noinspection PyShadowingNames
    def get_batch_shape(self, batch_dim):
        """
        :param int|tf.Tensor|None batch_dim:
        :return: shape with added batch-dim. e.g. (batch,time,feat) = (None,None,128)
        :rtype: tuple[int|None]
        """
        if self.batch_dim_axis is not None:
            return self.shape[: self.batch_dim_axis] + (batch_dim,) + self.shape[self.batch_dim_axis :]
        return self.shape

    def get_dynamic_batch_shape(self):
        """
        :rtype: list[int|tf.Tensor]
        """
        return [self.get_dim(axis) for axis in range(self.batch_ndim)]

    def have_varying_shape_in_ctx(self):
        """
        :return: whether the (dynamic) shape can change in this control flow context.
          E.g. when self.control_flow_context is a loop, and we have one dynamic dim
          where dyn_size_ext has the same control_flow_context
          (such that dyn_size_ext has e.g. shape [B,T] outside the loop).
          This can be relevant for accumulating values of self.placeholder
          e.g. via tf.TensorArray.
        :rtype: bool
        """
        return any(tag.control_flow_ctx for tag in self.dim_tags)

    @property
    def size_placeholder(self: Tensor):
        """
        For compatibility, return a proxy object which behaves like the original dict.

        :rtype: _SizePlaceholderProxy
        """
        return _SizePlaceholderProxy(self)

    @size_placeholder.setter
    def size_placeholder(self, sizes):
        """
        :param dict[int,tf.Tensor]|None sizes:
        """
        if sizes is None:
            return
        for axis_wo_b, size in sizes.items():
            self.set_dynamic_size(axis=self.get_batch_axis(axis_wo_b), sizes=size)

    @property
    def shape_dense(self):
        """
        :return: shape with feature dim axis
        :rtype: tuple[int|None]
        """
        if self.sparse:
            return self.shape + (self.dim,)  # by default, assume at the end
        return self.shape

    @property
    def batch_shape_dense(self):
        """
        :rtype: tuple[int|None]
        """
        if self.sparse:
            return self.batch_shape + (self.dim,)
        return self.batch_shape

    @property
    def dim_tags_sparse(self):
        """
        :return: dim tags without feature dim axis
        :rtype: tuple[Dim]
        """
        if self.sparse or not self.have_feature_axis():
            return self.dim_tags
        return self.dim_tags[: self.feature_dim_axis] + self.dim_tags[self.feature_dim_axis + 1 :]

    @property
    def dim_tags_set_implicit_only_wrapped(self):
        """
        :return: Dim tags implicit by sparse dim, or dynamic sizes, and not present as explicit dims.
          Also see :func:`dim_tags_set`.
        :rtype: set[_ImplicitDim]
        """
        self_dim_tags = set(self.dim_tags)
        dims = set()
        if self.sparse_dim and self.sparse_dim not in self_dim_tags:
            dims.add(_m.ImplicitSparseDim(self.sparse_dim))
        for dim in self.dim_tags:
            if dim.dyn_size_ext:
                for dim_ in dim.dyn_size_ext.dim_tags:
                    if dim_ not in self_dim_tags:
                        dims.add(_m.ImplicitDynSizeDim(dim_))
        return dims

    @property
    def dim_tags_set_implicit_only(self):
        """
        :return: Dim tags implicit by sparse dim, or dynamic sizes, and not present as explicit dims.
          Also see :func:`dim_tags_set`.
        :rtype: set[Dim]
        """
        return set(dim.tag for dim in self.dim_tags_set_implicit_only_wrapped)

    @property
    def dim_tags_set_implicit(self):
        """
        This is mostly intended to be used for verification, such as ``out_shape`` in a layer.
          https://github.com/rwth-i6/returnn/issues/706

        We return a set because when dim tags (dimensions, and the shape) are checked,
        we never want that the order plays any role.
          https://github.com/rwth-i6/returnn/wiki/RETURNN-principles
        Further, dimension tags should ideally be unique.
          https://github.com/rwth-i6/returnn/issues/632
        (This is not enforced currently, but we should not treat this specially now.)

        :return: set of dim tags
        :rtype: set[Dim]
        """
        dims = set(self.dim_tags)
        dims.update(self.dim_tags_set_implicit_only)
        return dims

    def remaining_dims(self: _t.Tensor, remove: Optional[Union[Dim, Sequence[Dim]]] = None) -> List[Dim]:
        """
        :param remove: dims to remove from self.dims
        :return: ordered batch dims
        """
        batch_dims = list(self._dims)
        if not remove:
            pass
        elif isinstance(remove, Dim):
            batch_dims.remove(remove)
        else:
            for remove_ in remove:
                batch_dims.remove(remove_)
        return batch_dims

    @property
    def ndim(self):
        """
        :rtype: int
        :return: ndim counted without batch-dim
        """
        return len(self.shape)

    @property
    def ndim_dense(self):
        """
        :rtype: int
        :return: ndim counted without batch-dim, added by 1 if we are sparse
        """
        if self.sparse:
            return self.ndim + 1
        return self.ndim

    @property
    def batch_ndim(self):
        """
        :rtype: int
        :return: ndim counted with batch-dim
        """
        return len(self._dims)

    @property
    def batch_ndim_dense(self):
        """
        :rtype: int
        :return: ndim counted with batch-dim, added by 1 if we are sparse
        """
        if self.sparse:
            return self.batch_ndim + 1
        return self.batch_ndim

    @property
    def is_time_major(self):
        """
        :return: whether this is in time-major format, i.e. (time,batch,...)
        :rtype: bool
        """
        return self.time_dim_axis == 0

    @property
    def is_batch_major(self):
        """
        :return: whether this is in batch-major format, i.e. (batch,...)
        :rtype: bool
        """
        return self.batch_dim_axis == 0

    @property
    def is_batch_feature_major(self):
        """
        :return: whether this is in batch-feature-major format, i.e. (batch,feature,...) (NC...)
        :rtype: bool
        """
        return self.batch_dim_axis == 0 and self.feature_dim_axis == 1

    @property
    def batch_dim_axis(self):
        """
        :return: batch dim axis, counted with batch-dim
        :rtype: int|None
        """
        return _batch_dim_axis_from_dim_tags_tuple(self._dims)

    @batch_dim_axis.setter
    def batch_dim_axis(self, axis):
        """
        :param int|None axis:
        """
        if axis == self.batch_dim_axis:
            return
        raise Exception("%s: cannot set batch_dim_axis = %s" % (self, axis))

    def _default_feature_dim_axis(self):
        """
        :return: feature dim axis, counted with batch-dim
        :rtype: int|None
        """
        if self.version >= 2:
            return None
        return _default_feature_dim_axis(
            batch_dim_axis=self.batch_dim_axis,
            time_dim_axis=self.time_dim_axis,
            batch_shape=self.batch_shape,
            sparse=self.sparse,
        )

    @property
    def feature_dim_axis(self):
        """
        :return: feature dim axis, counted with batch-dim
        :rtype: int|None
        """
        if self._feature_dim_axis is not NotSpecified:
            return self._feature_dim_axis
        return self._default_feature_dim_axis()

    @feature_dim_axis.setter
    def feature_dim_axis(self: _t.Tensor, value):
        """
        :param int|None|NotSpecified value:
        """
        assert value is NotSpecified or value is None or isinstance(value, int)
        if self.feature_dim_axis_or_unspecified == value:
            return
        if self.version >= 2 and value is NotSpecified:
            value = None
        if isinstance(value, int):
            assert 0 <= value < self.batch_ndim
        self._feature_dim_axis = value

    @property
    def feature_dim_axis_or_unspecified(self):
        """
        :return: feature dim axis, counted with batch-dim. could also be unspecified
        :rtype: int|None|NotSpecified
        """
        return self._feature_dim_axis

    @property
    def time_dim_axis(self) -> Optional[int]:
        """
        :return: time dim axis (deprecated)
        """
        if self.version >= 2:
            return None
        if self.time_dim_axis_or_unspecified is not NotSpecified:
            return self.time_dim_axis_or_unspecified
        return _default_time_dim_axis_dim_tags(self.dim_tags)

    @time_dim_axis.setter
    def time_dim_axis(self: _t.Tensor, value):
        """
        :param int|None|NotSpecified value:
        """
        assert value is NotSpecified or value is None or isinstance(value, int)
        if self.time_dim_axis_or_unspecified == value:
            return
        if self.version >= 2 and value in (None, NotSpecified):
            return
        assert self.version == 1, "time_dim_axis is deprecated"
        if isinstance(value, int):
            assert 0 <= value < self.batch_ndim
        self._make_extra().time_dim_axis = value

    @property
    def time_dim_axis_or_unspecified(self):
        """
        :return: time dim axis, counted with batch-dim. could also be unspecified
        :rtype: int|None|NotSpecified
        """
        if self.version >= 2:
            return NotSpecified
        if not self._extra:
            return NotSpecified
        return self._extra.time_dim_axis

    @property
    def time_dim_axis_excluding_batch(self):
        """
        :rtype: int|None
        """
        if self.time_dim_axis is None:
            return None
        return self.get_batch_axis_excluding_batch(self.time_dim_axis)

    @property
    def placeholder(self):
        """
        (Old alias for raw_tensor.)

        :rtype: T
        """
        return self._raw_tensor

    @placeholder.setter
    def placeholder(self: _t.Tensor, value: Optional[_t.RawTensorType]):
        """
        (Old alias for raw_tensor.)

        :param value:
        """
        self.raw_tensor = value

    @property
    def batch(self):
        """
        :rtype: BatchInfo|None
        """
        if not self._extra:
            return None
        return self._extra.batch

    @batch.setter
    def batch(self, batch):
        """
        :param BatchInfo|None batch:
        """
        if batch:
            assert batch.beam == self.beam
        if self.batch == batch:  # fast path
            return
        self._make_extra().batch = batch
        self._adapt_batch_consistent_dim_tags()

    @property
    def beam(self):
        """
        :rtype: SearchBeam|None
        """
        if not self._extra:
            return None
        if self._extra.beam:
            return self._extra.beam
        if self._extra.batch:
            return self._extra.batch.beam
        return None

    @beam.setter
    def beam(self, beam):
        """
        :param SearchBeam|None beam:
        """
        if self.beam == beam:
            return
        # No check for batch.beam, as the batch is usually set only later.
        self._make_extra().beam = beam
        if self._extra.batch:
            self._extra.batch = self.batch.copy_set_beam(beam=beam)
            self._adapt_batch_consistent_dim_tags()

    @property
    def dim(self):
        """
        :rtype: int|None
        """
        tag = self.feature_dim_or_sparse_dim
        if tag:
            return tag.dimension
        return None

    @dim.setter
    def dim(self, dim):
        """
        It is deprecated to explicitly set this.
        We just have this here to support some legacy code.
        It does nothing but checks the validity.

        :param int|None dim:
        """
        assert dim == self.dim

    @property
    def feature_dim_or_sparse_dim(self: Tensor):
        """
        :return: if we have a feature dim, return its dim tag. if we are sparse, return the sparse_dim. otherwise None
        :rtype: Dim|None
        """
        if self.sparse_dim:
            return self.sparse_dim
        feature_dim_axis = self.feature_dim_axis
        if feature_dim_axis is not None:
            return self._dims[feature_dim_axis]
        return None

    @property
    def sparse(self):
        """
        :rtype: bool
        :return: whether the values represent class indices. see ``sparse_dim``
        """
        return self.sparse_dim is not None

    @sparse.setter
    def sparse(self, sparse):
        """
        It is deprecated to explicitly set this.
        We just have this here to support some legacy code.

        :param bool sparse:
        """
        if self.sparse == sparse:
            return
        if not sparse:
            self.sparse_dim = None
            return
        raise Exception("%s: setting sparse=True not supported anymore. set sparse_dim instead" % self)

    @property
    def vocab(self):
        """
        :rtype: returnn.datasets.util.vocabulary.Vocabulary|None
        """
        if self.sparse_dim:
            return self.sparse_dim.vocab
        if self.have_feature_axis():
            return self.dim_tags[self.feature_dim_axis].vocab
        return None

    @vocab.setter
    def vocab(self, vocab):
        """
        :param returnn.datasets.util.vocabulary.Vocabulary|None vocab:
        """
        raise Exception("%s: setting vocab not supported anymore. set sparse_dim instead" % self)

    def time_dimension(self) -> Union[int, _t.RawTensorType]:
        """
        :return: shape(placeholder)[time_dim_axis], int scalar
        :rtype: tf.Tensor
        """
        assert self.time_dim_axis is not None
        return self.get_dim(self.time_dim_axis)

    def get_dim(self, axis: int) -> Union[int, _t.RawTensorType]:
        """
        :param axis: counted with batch-dim
        :return: shape[axis]
        """
        if self.batch_shape[axis] is not None:
            return self.batch_shape[axis]
        assert self._raw_tensor is not None
        backend = self._raw_backend
        return backend.get_shape_tuple_raw(self._raw_tensor)[axis]

    def get_placeholder_as_time_major(self):
        """
        :rtype: tf.Tensor
        """
        assert self.placeholder is not None
        return self.copy_as_time_major().placeholder

    def get_placeholder_as_batch_major(self):
        """
        :rtype: tf.Tensor
        """
        assert self.placeholder is not None
        return self.copy_as_batch_major().placeholder

    def get_placeholder_with_runtime_sanity_checks(self):
        """
        :return: identity(self.placeholder) with added checks
        :rtype: tf.Tensor
        """
        assert self._raw_tensor is not None
        backend = self._raw_backend
        return backend.identity_with_control_dependencies_raw(self._raw_tensor, [self.get_runtime_sanity_check_op()])

    def get_placeholder_time_flattened(self):
        """
        :return: via :func:`flatten_with_seq_len_mask`
        :rtype: tensorflow.Tensor
        """
        assert self._raw_backend.is_tensorflow, "get_placeholder_time_flattened only implemented for TF yet"
        from returnn.tf.util.basic import flatten_with_seq_len_mask, get_shape
        import tensorflow as tf

        assert self.placeholder is not None
        assert self.have_time_axis()
        # flatten_with_seq_len_mask only works if either time_dim_axis or batch_dim_axis is 0:
        assert 0 in [self.time_dim_axis, self.batch_dim_axis]
        time_dim = self.get_time_dim_tag()
        if time_dim.need_masking():
            assert time_dim.dyn_size_ext.dims == (self.get_batch_dim_tag(),)  # not implemented otherwise
            return flatten_with_seq_len_mask(
                self.placeholder,
                time_dim.dyn_size,
                batch_dim_axis=self.batch_dim_axis,
                time_dim_axis=self.time_dim_axis,
            )
        else:  # static time
            x = tf.transpose(
                self.placeholder,
                [self.batch_dim_axis, self.time_dim_axis]
                + [i for i in range(self.batch_ndim) if i not in [self.batch_dim_axis, self.time_dim_axis]],
            )
            shape = get_shape(x)
            return tf.reshape(x, [shape[0] * shape[1]] + shape[2:])

    def get_placeholder_flattened(self, keepdims=False):
        """
        :param bool keepdims: if set, it will add broadcast dimensions after the flattening behind the first axis
        :rtype: tf.Tensor
        :return: placeholder where all dynamic axes are flattened into a single axis.
          e.g. for the usual case (batch, time, dim), it becomes (batch'|time', dim),
          or (batch, time, height, dim) will also become (batch'|time', dim).
          with keep_dims, (batch, time, height, dim) will become (batch'|time', 1, 1, dim).
        """
        assert self.placeholder is not None
        assert self._raw_backend.is_tensorflow, "get_placeholder_flattened only implemented for TF yet"
        import tensorflow as tf

        x = self.placeholder
        orig_dyn_axes = self.get_spatial_batch_axes() + [self.batch_dim_axis]
        dyn_axes = list(orig_dyn_axes)
        if dyn_axes == [self.batch_dim_axis]:
            return x
        assert 0 in dyn_axes, "would need some transpose, not supported at the moment"
        assert len(dyn_axes) > 1
        orig_num_dyn_axes = len(dyn_axes)
        ndim = len(self.batch_shape)
        if self.have_time_axis():
            x = self.get_placeholder_time_flattened()
            removed_axis = max(self.time_dim_axis, self.batch_dim_axis)
            dyn_axes.remove(removed_axis)
            dyn_axes = [(i if (i < removed_axis) else (i - 1)) for i in dyn_axes]
            ndim -= 1
        if len(dyn_axes) > 1:
            shape = tf.shape(x)
            x = tf.reshape(
                x, [tf.reduce_prod([shape[i] for i in dyn_axes])] + [shape[i] for i in range(ndim) if i not in dyn_axes]
            )
            dyn_axes = [0]
        assert dyn_axes == [0]
        if keepdims and orig_num_dyn_axes >= 2:
            for i in orig_dyn_axes:
                if i not in dyn_axes:
                    x = tf.expand_dims(x, axis=i)
            x.set_shape([None] * self.batch_ndim)
        return x

    def get_axes(self, exclude_time=False, exclude_batch=False, exclude_feature=False):
        """
        :param bool exclude_time: will filter out the time-axis
        :param bool exclude_batch: will filter out the batch-axis
        :param bool exclude_feature: will filter out the feature-axis
        :return: list of axes, like `range(len(self.shape))`, calculated with batch dim.
        :rtype: list[int]
        """
        axes = list(range(len(self.batch_shape)))
        if exclude_time and self.time_dim_axis is not None:
            axes.pop(axes.index(self.time_dim_axis))
        if exclude_batch and self.batch_dim_axis is not None:
            axes.pop(axes.index(self.batch_dim_axis))
        if exclude_feature and self.feature_dim_axis is not None:
            axes.pop(axes.index(self.feature_dim_axis))
        return axes

    @classmethod
    def _verify_axis_int_from_description(cls, allow_int=NotSpecified):
        """
        Call this when you have the case that ``axis`` or ``axes``
        in :func:`get_axes_from_description` or :func:`get_axis_from_description`
        was specified as int.

        :param bool|NotSpecified allow_int:
        """
        msg = "Do not specify axis as int but as str or Dim instead."
        if allow_int is NotSpecified:
            from returnn.util import BehaviorVersion

            BehaviorVersion.require(condition=False, message=msg, version=5)
        if allow_int:
            return
        raise Exception(msg)

    @classmethod
    def _verify_axis_order_dependent(cls):
        """
        Call this when you have the case that ``axis`` or ``axes``
        in :func:`get_axes_from_description` or :func:`get_axis_from_description`
        depends on the order of the axes.
        """
        from returnn.util import BehaviorVersion

        BehaviorVersion.require(
            condition=False,
            message="Do not specify axis or axes in a way that depends on the order of the axes.",
            version=7,
        )

    def _make_valid_int_axis(self, axis):
        """
        :param int axis: counted with batch. anything in [-ndim,ndim-1]
        :return: axis in [0,ndim-1]
        :rtype: int
        """
        if axis < 0:
            assert axis + self.batch_ndim >= 0
            axis += self.batch_ndim
        assert axis < self.batch_ndim
        return axis

    def get_axes_from_description(self, axes, allow_int=NotSpecified):
        """
        :param int|list[int]|str|typing.Sequence[str|Dim]|Dim|None axes: one axis or multiple axis, or none.
            This is counted with batch-dim, which by default is axis 0 (see enforce_batch_dim_axis).
            It also accepts the special tokens "B"|"batch", "spatial", "spatial_except_time", or "F"|"feature",
            and more (see the code).
        :param bool|NotSpecified allow_int: whether to allow an int directly.
            in almost all cases, it is better to use a symbolic name
            to specify an axis, as different layers could reorder them,
            and maybe also change their behavior in the future.
        :return: list of axes, counted with batch-dim
        :rtype: list[int]
        """
        if axes is None or (isinstance(axes, str) and axes == ""):
            return []
        if isinstance(axes, Dim):
            # Once we have not guaranteed unique dim tags, multiple axes could match.
            # https://github.com/rwth-i6/returnn/issues/632
            dims = [i for (i, tag) in enumerate(self.dim_tags) if tag == axes]
            if len(dims) > 1:
                max_match_priority = max(self.dim_tags[i].match_priority for i in dims)
                dims = [i for i in dims if self.dim_tags[i].match_priority == max_match_priority]
            assert len(dims) <= 1, (
                "%s: matching dim %s must be unique,"
                " use `match_priority` to resolve the matching order of ambiguous dimensions" % (self, axes)
            )
            return dims
        if isinstance(axes, int):
            self._verify_axis_int_from_description(allow_int=allow_int)
            return [self._make_valid_int_axis(axes)]
        assert isinstance(axes, (str, int, list, tuple, Sequence))
        if isinstance(axes, str):
            import re

            axes = axes.lower()
            if axes in ["b", "batch"]:
                assert self.batch_dim_axis is not None
                return [self.batch_dim_axis]
            elif axes == "spatial":
                return self.get_spatial_batch_axes()
            elif re.match("(s|spatial):-?\\d+$", axes):
                self._verify_axis_order_dependent()
                s = int(axes.split(":")[1])
                spatial_axes = self.get_spatial_batch_axes()
                if s < 0:
                    s += len(spatial_axes)
                assert s < len(spatial_axes), "%s get_axes_from_description: %r invalid" % (self, axes)
                return [spatial_axes[s]]
            elif axes in ["dyn", "dynamic"]:
                return self.get_dynamic_axes()
            elif re.match("(d|dyn|dynamic):-?\\d+$", axes):
                self._verify_axis_order_dependent()
                s = int(axes.split(":")[1])
                dyn_axes = self.get_dynamic_axes()
                if s < 0:
                    s += len(dyn_axes)
                assert 0 <= s < len(dyn_axes), "%s get_axes_from_description: %r invalid" % (self, axes)
                return [dyn_axes[s]]
            elif axes == "spatial_except_time":
                axes = self.get_spatial_batch_axes()
                assert self.time_dim_axis is not None
                axes.remove(self.time_dim_axis)
                return axes
            elif axes in ["t", "time"]:
                assert self.time_dim_axis is not None
                return [self.time_dim_axis]
            elif axes == "t?":
                return [self.time_dim_axis] if self.time_dim_axis is not None else []
            elif axes == "except_time":  # also except batch
                axes = list(range(self.batch_ndim))
                axes.remove(self.batch_dim_axis)
                if self.time_dim_axis is not None:
                    axes.remove(self.time_dim_axis)
                return axes
            elif axes == "except_batch":
                axes = list(range(self.batch_ndim))
                axes.remove(self.batch_dim_axis)
                return axes
            elif re.match("(except_batch):-?\\d+$", axes):
                self._verify_axis_order_dependent()
                s = int(axes.split(":")[1])
                non_batch_axes = list(range(self.batch_ndim))
                if self.batch_dim_axis is not None:
                    non_batch_axes.remove(self.batch_dim_axis)
                if s < 0:
                    s += len(non_batch_axes)
                assert 0 <= s < len(non_batch_axes), "%s get_axes_from_description: %r invalid" % (self, axes)
                return [non_batch_axes[s]]
            elif axes == "*":
                return list(range(self.batch_ndim))
            elif axes == "static":
                return self.get_static_axes()
            elif re.match("(static):-?\\d+$", axes):
                self._verify_axis_order_dependent()
                s = int(axes.split(":")[1])
                static_axes = self.get_static_axes()
                if s < 0:
                    s += len(static_axes)
                assert 0 <= s < len(static_axes), "%s get_axes_from_description: %r invalid" % (self, axes)
                return [static_axes[s]]
            elif re.match("(dim):\\d+$", axes):
                s = int(axes.split(":")[1])
                dims = [a for a in range(self.batch_ndim) if self.batch_shape[a] == s]
                assert dims, "%s get_axes_from_description: 'dim:%i' not found" % (self, s)
                assert len(dims) == 1, "%s get_axes_from_description: 'dim:%i' only allowed when unique" % (self, s)
                return dims
            elif axes in ["f", "feature", "non_spatial"]:
                return self.get_feature_batch_axes()
            elif all([a in "btf" for a in axes]):
                return self.get_axes_from_description(list(axes))
            elif axes.startswith("stag:"):  # spatial tag
                return [self.get_axis_by_tag_name(axes[len("stag:") :], spatial_only=True)]
            elif axes.startswith("stag-single:"):  # spatial tag which possibly matches multiple spatial axes
                # in this case, a name of form "stag-single:<idx>:<name> is expected.
                # idx is relative to the matching stags,
                # i.e., it is the index among the list of spatial dims matching the name
                # Note: no _verify_axis_order_dependent here because as long as we do not enforce unique dim tags
                # (https://github.com/rwth-i6/returnn/issues/632), we can have multiple axes with the same tag,
                # and then we need to be able to differentiate between them by order.
                _, idx_s, name = axes.split(":", 2)  # stag-single:<idx>:<name>
                idx = int(idx_s)
                return [self.get_axes_by_tag_name(name, spatial_only=True)[idx]]
            elif axes.startswith("tag:"):  # any tag
                return [self.get_axis_by_tag_name(axes[len("tag:") :])]
            raise Exception("invalid axis mode %r" % axes)
        assert isinstance(axes, (tuple, list, Sequence)), "invalid axes %r" % axes
        flat_axes = []
        for i in axes:
            if isinstance(i, int):
                self._verify_axis_int_from_description(allow_int=allow_int)
                flat_axes.append(self._make_valid_int_axis(i))
            else:
                assert isinstance(i, (str, tuple, list, Dim))
                flat_axes += self.get_axes_from_description(i, allow_int=allow_int)
        res = []
        for i in flat_axes:
            if i not in res:
                res.append(i)
        return res

    def get_dim_tag_from_description(self, axis):
        """
        :param str|Dim axis:
        :return: our matching dim tag. this assumes it exists.
        :rtype: Dim
        """
        axis_int = self.get_axis_from_description(axis, allow_int=False)
        return self.dim_tags[axis_int]

    def get_axis_from_description(self, axis, allow_int=NotSpecified):
        """
        :param int|str|Dim axis:
        :param bool|NotSpecified allow_int:
        :return: axis, counted with batch-dim
        :rtype: int
        """
        if isinstance(axis, Dim):  # fast path
            res_idx: Optional[int] = None
            res_tag: Optional[Dim] = None
            for i, tag in enumerate(self._dims):
                tag: Dim
                if tag is axis or tag == axis:
                    if res_tag is None or res_tag.match_priority < tag.match_priority:
                        res_idx = i
                        res_tag = tag
                        continue
                    if res_tag.match_priority > tag.match_priority:
                        continue
                    raise Exception(
                        f"{self}: get_axis_from_description({axis}) not unique."
                        f" use match_priority to resolve ambiguity"
                    )
            if res_idx is None:
                raise Exception(f"{self}: get_axis_from_description({axis}) not found")
            return res_idx
        axes = self.get_axes_from_description(axis, allow_int=allow_int)
        assert axes, "%s: %r axis not found" % (self, axis)
        assert len(axes) == 1, "%r: %r is not a unique axis but %r" % (self, axis, axes)
        return axes[0]

    def get_description_from_axis(self, axis):
        """
        :param int axis:
        :return: some canonical description, such that ``self.get_axis_from_description(res) == axis``.
          This is quite heuristically for now. We use both strings as also Dim when appropriate.
          The behavior could potentially change in the future, also the condition will always hold.
        :rtype: str|Dim
        """
        assert 0 <= axis < self.batch_ndim
        if axis == self.batch_dim_axis:
            return "B"
        dim_tag = self.dim_tags[axis]
        # It's possible that dim tags are not unique (https://github.com/rwth-i6/returnn/issues/632).
        matching_tags = [i for (i, tag) in enumerate(self.dim_tags) if tag == dim_tag]
        if dim_tag.dyn_size_ext and len(matching_tags) == 1:
            return dim_tag
        if axis == self.time_dim_axis:
            return "T"  # this might change
        if axis == self.feature_dim_axis:
            return "F"  # this might change
        if len(matching_tags) == 1:
            # Fallback with dim tag
            return dim_tag
        # Do not use indexed static or dynamic
        # because we want to avoid relying on the axis order as much as possible.
        # However, as we do not have unique dim tags in this case,
        # we have to rely at least on the order of this dim tag.
        # Use stag-single.
        name = dim_tag.description
        matching_axes = self.get_axes_by_tag_name(name, spatial_only=True)
        assert axis in matching_axes
        return "stag-single:%i:%s" % (
            matching_axes.index(axis) - len(matching_axes),
            name,
        )  # negative because this is likely more robust

    def has_axis(self, axis):
        """
        :param str|Dim axis:
        :return: whether the axis exists
        :rtype: bool
        """
        axes = self.get_axes_from_description(axis, allow_int=False)
        return len(axes) > 0

    def get_axes_by_tag_name(self, name, spatial_only=False):
        """
        :param str name: the tag name, or part of it (must be unique, and must exist)
        :param bool spatial_only:
        :rtype: list[int]
        """
        dim_tags = self.get_batch_shape_dim_tags()
        matching_dim_tags = [
            (axis, tag)
            for axis, tag in enumerate(dim_tags)
            if name.lower() in tag.description.lower() or name.lower() in tag.get_same_base().description.lower()
        ]
        if spatial_only:
            spatial_axes = self.get_spatial_batch_axes()
            matching_dim_tags = [
                (axis, tag) for axis, tag in matching_dim_tags if axis in spatial_axes or tag.is_spatial_dim()
            ]
        return [ax for ax, _ in matching_dim_tags]

    def get_axis_by_tag_name(self, name, spatial_only=False):
        """
        :param str name: the tag name, or part of it (must be unique, and must exist)
        :param bool spatial_only:
        :rtype: int
        """
        matching_dim_tags = self.get_axes_by_tag_name(name, spatial_only)
        assert len(matching_dim_tags) > 0, "%r: no %stag found with name %r" % (
            self,
            "spatial " if spatial_only else "",
            name,
        )
        assert len(matching_dim_tags) == 1, "%r: tag name %r is not unique in dim tags %r" % (
            self,
            name,
            self.get_batch_shape_dim_tags(),
        )
        return matching_dim_tags[0]

    def get_batch_axis_excluding_batch(self, axis):
        """
        :param int axis: counted with batch-dim
        :return: axis counted without batch-dim
        :rtype: int|None
        """
        return _get_axis_wo_b(axis, batch_dim_axis=self.batch_dim_axis, batch_ndim=self.batch_ndim)

    def have_dim_tag(self, tag, include_implicit=True, unique=False):
        """
        :param Dim tag:
        :param bool include_implicit:
        :param bool unique:
        :rtype: bool
        """
        dims = list(self.dim_tags)
        if include_implicit:
            dims.extend(self.dim_tags_set_implicit_only)
        matching_dims = [dim for dim in dims if dim == tag]
        return (len(matching_dims) == 1) if unique else (len(matching_dims) >= 1)

    def get_batch_axis(self, axis):
        """
        :param int axis: counted without batch-dim
        :return: axis counted with batch-dim
        :rtype: int
        """
        return _get_axis_wb(axis, batch_dim_axis=self.batch_dim_axis)

    def have_batch_axis(self):
        """
        :rtype: bool
        """
        return self.batch_dim_axis is not None

    def have_time_axis(self):
        """
        :rtype: bool
        """
        return self.time_dim_axis is not None

    def have_feature_axis(self):
        """
        :rtype: bool
        """
        return self.feature_dim_axis is not None

    def is_time_axis_dynamic(self):
        """
        :return: whether there are different seq-lens for the time, or all the same (static)
        :rtype: bool
        """
        assert self.time_dim_axis is not None
        if self.placeholder is None:
            # Run at template construction time.
            return self.batch_shape[self.time_dim_axis_excluding_batch] is None
        if self.time_dim_axis_excluding_batch in self.size_placeholder:
            return True
        assert isinstance(
            self.shape[self.time_dim_axis_excluding_batch], int
        ), "%s: dynamic time axis dim (None) (axis %i) but size_placeholder %r misses information" % (
            self,
            self.time_dim_axis,
            self.size_placeholder,
        )
        return False

    def is_axis_dynamic(self, axis):
        """
        :param int axis: counted with batch-dim axis
        :return: dynamic, i.e. we have it in size_placeholder.
            Note that this does not perfectly match with :func:`get_dynamic_axes`,
            but more with :func:`is_time_axis_dynamic`,
            although probably in most (all?) cases it should match.
            If True, you can get the size via :func:`get_dynamic_size`.
        :rtype: bool
        """
        if axis == self.batch_dim_axis:
            return False
        return self.batch_shape[axis] is None

    def has_dynamic_size(self, axis):
        """
        :param int axis: counted with batch-dim axis. implies that you can call :func:`get_dynamic_size`.
        :rtype: bool
        """
        return self.dim_tags[axis].dyn_size is not None

    def get_dynamic_size(self, axis):
        """
        :param int axis: counted with batch-dim axis. :func:`get_dynamic_size` should be True.
        :return: shape (B,)
        :rtype: tf.Tensor
        """
        tag = self.dim_tags[axis]
        assert tag.dyn_size is not None, "%s: axis %i has no dyn size" % (self, axis)
        return tag.dyn_size

    def set_dynamic_size(self, axis, sizes):
        """
        :param int axis: counted with batch-dim
        :param tf.Tensor sizes: shape [B]
        """
        # Note: The following code is somewhat ugly patchwork
        # to fix some other currently incomplete or buggy behavior of some layers
        # which introduce sizes without correctly setting the dim tag.
        # The beam information is also missing currently.
        # We make the ugly assumption that when it is unset,
        # the first usage should hopefully define the correct beam.
        if getattr(sizes, "_RETURNN_dyn_size_beam", NotSpecified) is NotSpecified:
            sizes._RETURNN_dyn_size_beam = self.beam
        if self.beam and getattr(sizes, "_RETURNN_dyn_size_beam", None) != self.beam:
            tag = Dim.get_tag_from_size_tensor(sizes)
            assert tag and self.batch
            tag = tag.get_for_batch_ctx(batch=self.batch, ctx=self.control_flow_ctx)
            assert tag.dyn_size is not None
            sizes = tag.dyn_size

        sizes_tag = Dim.get_tag_from_size_tensor(sizes)
        if sizes_tag:
            assert sizes_tag.is_same_size_tensor(sizes)
        tag = self._dims[axis]
        assert tag.is_dynamic()
        if tag.is_same_size_tensor(sizes):
            pass  # nothing to do
        elif tag.dyn_size is None:
            if sizes_tag:  # special rule for older code: overtake previous existing
                assert sizes_tag.is_same_size_tensor(sizes)
                tag = sizes_tag
            else:
                # Assign now. This should also set the dim tag on sizes.
                tag = tag.set_tag_on_size_tensor(sizes, batch=self.batch)
        else:
            # Reset to some new size.
            # Use new dim tag, or previous existing attached to size.
            assert sizes_tag, "%s: assign dyn sizes %s without defined dim tag" % (self, sizes)
            tag = sizes_tag
        if self.batch:
            tag = tag.get_for_batch_ctx(batch=self.batch, ctx=self.control_flow_ctx)
        if tag is not self._dims[axis]:
            self._dims = self._dims[:axis] + (tag,) + self._dims[axis + 1 :]
        if tag.dyn_size is None:
            tag.dyn_size = sizes

    def get_dynamic_axes(self):
        """
        :return: list of axes, counted with batch-dim axis (but we exclude the batch dim axis itself)
        :rtype: list[int]
        """
        return [axis for axis, dim in enumerate(self.batch_shape) if axis != self.batch_dim_axis and dim is None]

    def get_static_axes(self):
        """
        :return: list of axes, counted with batch-dim axis (but we exclude the batch dim axis itself)
        :rtype: list[int]
        """
        return [axis for axis, dim in enumerate(self.batch_shape) if axis != self.batch_dim_axis and dim is not None]

    def mark_same_time(self, tags, must_match=False):
        """
        If the given dimension tag matches any of our axes, we set our time axis to the selected one.

        :param set[Dim]|Dim tags:
        :param bool must_match: if True, throw an exception if not found
        :return: whether we have found the same
        :rtype: bool
        """
        if isinstance(tags, Dim):
            tags = {tags}
        assert all(isinstance(tag, Dim) for tag in tags)
        for axis, dim_tag in enumerate(self.dim_tags):
            if dim_tag in tags:
                self.time_dim_axis = axis
                return True
        if must_match:
            raise Exception("%s mark_same_time: %s not found" % (self, tags))
        return False

    def is_same_time_dim(self, other: Tensor) -> bool:
        """
        Checks whether we have a matching/compatible time dim.

        :param other:
        """
        assert self.have_time_axis()
        if not other.have_time_axis():
            return False
        tag_self = self.get_dim_tag(self.time_dim_axis)
        tag_other = other.get_dim_tag(other.time_dim_axis)
        return tag_self == tag_other

    def get_sequence_lengths(self) -> _t.RawTensorType:
        """
        Deprecated. Access the information directly from dim tags, whatever you need.

        Warning: This assumes TensorFlow in the fallback case.

        :return: seq lens tensor of shape [B] of dtype int32. also see :func:`get_dynamic_size`
        :rtype: tf.Tensor
        """
        assert self.time_dim_axis is not None
        dim = self._dims[self.time_dim_axis]
        assert isinstance(dim, Dim)
        if dim.dyn_size_ext:
            if dim.dyn_size_ext.raw_tensor is None:
                dim.complete_dyn_size()
            assert dim.dyn_size_ext.raw_tensor is not None
            return dim.dyn_size_ext.raw_tensor
        assert self.batch_shape[self.time_dim_axis] is not None
        assert self.batch_dim_axis is not None
        batch_dim_ = self._dims[self.batch_dim_axis]
        assert isinstance(batch_dim_, Dim)
        if batch_dim_.dyn_size_ext and batch_dim_.dyn_size_ext.raw_tensor is not None:
            backend = batch_dim_.dyn_size_ext._raw_backend
            return backend.fill_raw([batch_dim_.dyn_size_ext.raw_tensor], dim.size)
        import tensorflow as tf

        return tf.fill([self.get_batch_dim()], dim.size)

    def get_sequence_mask(self):
        """
        :return: seq mask of shape (batch,time) if we are batch-major, else (time,batch) if we are time-major
        :rtype: tf.Tensor
        """
        # noinspection PyProtectedMember
        from returnn.frontend._backend import get_backend_by_raw_tensor_type

        assert self.time_dim_axis is not None
        assert self.batch_dim_axis is not None
        dyn_seq_len = self.get_sequence_lengths()
        backend = get_backend_by_raw_tensor_type(type(dyn_seq_len))

        if self.is_time_major:
            assert self.batch_dim_axis == 1
            return backend.sequence_mask_raw(dyn_seq_len, batch_major=False)
        else:
            assert self.batch_dim_axis == 0
            assert self.time_dim_axis == 1
            return backend.sequence_mask_raw(dyn_seq_len, batch_major=True)

    def get_sequence_mask_broadcast(self: Tensor, axis=None) -> _t.RawTensorType:
        """
        :param Dim|int|None axis:
        :return: seq mask of shape ((batch,time) or (time,batch)) + (1,)s for remaining dims
          if BT or TB major, and axis is T or None.
          In general compatible to placeholder, i.e. same ndim, with broadcast dims.
          We assert here that the axis is dynamic (:func:`is_axis_dynamic`), i.e. we have the size.
        """
        if isinstance(axis, Dim):
            axis = self.get_axis_from_description(axis)
        if axis is None:
            assert self.time_dim_axis is not None
            axis = self.time_dim_axis
        if axis < 0:
            assert axis + self.batch_ndim > 0
            axis += self.batch_ndim
        assert 0 <= axis < self.batch_ndim
        assert axis != self.batch_dim_axis
        tag: Dim = self.dim_tags[axis]
        assert tag.dyn_size_ext and tag.dyn_size_ext.raw_tensor is not None
        backend = tag.dyn_size_ext._raw_backend
        assert set(tag.dyn_size_ext.dim_tags).issubset(self.dim_tags)  # https://github.com/rwth-i6/returnn/issues/721
        with backend.name_scope_raw("get_sequence_mask_broadcast"):
            if (
                backend.have_sequence_mask_raw()
                and tag.dyn_size_ext.have_batch_axis()
                and tag.dyn_size_ext.batch_ndim == 1
            ):  # just [B]
                # This is the common case where the size is of shape [B].
                # We make use of sequence_mask or sequence_mask_time_major in that case,
                # which is optimized by caching.
                size = tag.dyn_size
                seq_mask = backend.sequence_mask_raw(size, batch_major=axis >= self.batch_dim_axis)  # (B,T) or (T,B)
                shape = [1] * self.batch_ndim  # type: List[Union[int,_t.RawTensorType]]
                shape[self.batch_dim_axis] = self.get_batch_dim()
                shape[axis] = tag.get_dim_value()
                seq_mask = backend.reshape_raw(seq_mask, shape)
                assert seq_mask.get_shape().ndims == self.batch_ndim
            else:  # size is something unusual, not just [B], but e.g. [B,S] or so
                seq_mask = self.get_sequence_mask_tensor(axis).copy_compatible_to_dims_raw(self.dims)
        return seq_mask

    def get_sequence_mask_tensor(self: Tensor, axis: int) -> Tensor:
        """
        :param axis:
        :return: mask
        """
        if axis < 0:
            assert axis + self.batch_ndim > 0
            axis += self.batch_ndim
        assert 0 <= axis < self.batch_ndim
        assert axis != self.batch_dim_axis
        tag: Dim = self.dim_tags[axis]
        return tag.get_mask(dim_order=self.dims, device=self.device)

    def get_sequence_lengths_broadcast(self, axis=None):
        """
        :param int|None axis:
        :return: seq len of some shape which is broadcastable to self.placeholder.
          Note that this is not always possible, e.g. when the seq len has shape [B]
          but the tensor has just shape [T]. We currently throw an error then.
        :rtype: tf.Tensor
        """
        if axis is None:
            assert self.time_dim_axis is not None
            axis = self.time_dim_axis
        if axis < 0:
            assert axis + self.batch_ndim > 0
            axis += self.batch_ndim
        assert 0 <= axis < self.batch_ndim
        assert axis != self.batch_dim_axis
        tag = self.dim_tags[axis]
        assert tag.dyn_size_ext
        return tag.dyn_size_ext.copy_compatible_to(self, check_dtype=False, check_sparse=False).placeholder

    def num_elements(self: Tensor) -> Union[int, Tensor]:
        """
        :return: number of elements in this tensor, i.e. prod(self.shape)
        :rtype: tf.Tensor
        """
        import returnn.frontend as rf

        return rf.num_elements_of_shape(self.dims)

    def copy_masked(
        self: Tensor,
        mask_value: Union[Tensor, float, int, _t.RawTensorType],
        *,
        dims: Optional[Sequence[Union[Dim, int]]] = None,
        allow_int: bool = NotSpecified,
    ) -> Tensor:
        """
        :param mask_value:
        :param dims:
        :param allow_int: in dims
        """
        assert self.raw_tensor is not None
        if dims is None:
            axes = range(self.batch_ndim)
        else:
            axes = [self.get_axis_from_description(dim, allow_int=allow_int) for dim in dims]
            assert len(set(axes)) == len(dims), f"{self} copy_masked, dims {dims} not unique, axes {axes}"

        # Code was originally in TF util mask_dyn_seq_len_nd, here rewritten with RETURNN frontend (RF).

        # Filter out some axes which should not be used for masking.
        axes_ = []
        for axis in axes:
            tag: Dim = self.dims[axis]
            if not tag.need_masking():
                continue
            # It only makes sense to apply for this axis if the dyn size dims are all existing in x itself.
            # E.g. if the dyn_size_ext shape is [B] but the shape of x is just [T] (without B),
            # then we do not need masking.
            if set(tag.dyn_size_ext.dim_tags).issubset(self.dim_tags):
                axes_.append(axis)
        axes = axes_

        if not axes:
            return self.copy()

        use_padding_info = False
        tf_util = None
        if self._raw_backend.is_tensorflow:
            import returnn.tf.util.basic as tf_util

            use_padding_info = isinstance(mask_value, (int, float))
            if use_padding_info:
                d = tf_util.get_padding_info_dict_ref(self.raw_tensor)
                existing_pad_values = [d.get(self.dim_tags[axis]) for axis in axes]
                if set(existing_pad_values) == {mask_value}:
                    return self.copy()  # nothing to do

        import returnn.frontend as rf

        mask = None
        for axis in axes:
            mask_ = self._dims[axis].get_mask(dim_order=self.dims, device=self.device)
            mask = rf.logical_and(mask, mask_) if mask is not None else mask_
        assert isinstance(mask, _t.Tensor)
        res = rf.where(mask, self, mask_value)
        if use_padding_info:
            d = tf_util.get_padding_info_dict_ref(res.raw_tensor)
            d.clear()
            d.update({self.dim_tags[axis]: mask_value for axis in axes})
        return res

    def get_batch_dim(self) -> Union[_t.RawTensorType, int]:
        """
        Warning: This assumes TensorFlow and is also mostly TF specific.

        :return: batch dim
        """
        assert self.batch_dim_axis is not None
        if self.batch:
            if self.beam:
                assert self.batch.beam == self.beam
            dim = self.batch.dim
            if not isinstance(dim, int):
                batch_dim_ = self.dim_tags[self.batch_dim_axis]
                batch_dim_.set_tag_on_size_tensor(dim, batch=self.batch)
            return dim
        # Note: We need this fallback code for now
        # until we consistently have set self.batch correctly in all cases.
        from returnn.tf.layers.base import LayerBase

        batch = LayerBase.get_recent_layer().get_batch_info()
        batch = batch.copy_set_beam(self.beam)
        return batch.dim

    def get_batch_dim_tag(self):
        """
        :rtype: Dim
        """
        assert self.have_batch_axis()
        return self.dim_tags[self.batch_dim_axis]

    def get_static_batch_dim(self):
        """
        :rtype: int|None
        """
        # Do not fallback to get_batch_dim or get_recent_layer or so. This should be safe.
        if self.batch:
            return self.batch.static_dim
        if self.have_batch_axis():
            return self.get_batch_dim_tag().dimension
        return None

    def get_spatial_batch_axes(self):
        """
        :rtype: list[int]
        :return: list of axes which are not batch axes and not feature or which are time axis or dynamic.
          counted with batch-dim.
        """
        return [
            axis
            for axis in range(self.batch_ndim)
            if axis != self.batch_dim_axis
            and (axis != self.feature_dim_axis or axis == self.time_dim_axis or self.batch_shape[axis] is None)
        ]

    def get_spatial_axes(self):
        """
        :rtype: list[int]
        :return: list of axes which are not feature and batch axes, counted without batch-dim.
        """
        return [self.get_batch_axis_excluding_batch(axis) for axis in self.get_spatial_batch_axes()]

    def get_feature_batch_axes(self):
        """
        :rtype: list[int]
        :return: list of axes which are feature axes, counted with batch-dim.
            currently there is only one or zero such axis.
        """
        if self.feature_dim_axis is not None:
            return [self.feature_dim_axis]
        return []

    def get_feature_axes(self):
        """
        :rtype: list[int]
        :return: list of axes which are feature axes, counted without batch-dim.
        """
        return [self.get_batch_axis_excluding_batch(axis) for axis in self.get_feature_batch_axes()]

    # Exclude "batch_dim_axis" now because that is always inferred from dim tags.
    SpecialAxesNames = ("time_dim_axis", "feature_dim_axis")

    def get_special_axes_dict(self, counted_with_batch_dim=True, only_available=False):
        """
        :param bool counted_with_batch_dim:
        :param bool only_available:
        :return: dict axis-name -> axis
        :rtype: dict[str,int]
        """
        axes = list(self.SpecialAxesNames)
        d = {k: getattr(self, k) for k in axes}
        if not counted_with_batch_dim:
            d = {k: self.get_batch_axis_excluding_batch(v) if (v is not None) else None for (k, v) in d.items()}
        if only_available:
            d = {k: v for (k, v) in d.items() if v is not None}
            if self.feature_dim_axis_or_unspecified is NotSpecified:  # special rule
                d.pop("feature_dim_axis", None)
        return d

    def get_bc_spatial_batch_shape(self):
        """
        :return: shape which will broadcast along all spatial dimensions and time/batch dim
        :rtype: tuple[int|None]
        """
        dyn_axes = self.get_spatial_batch_axes()
        if self.batch_dim_axis is not None:
            dyn_axes += [self.batch_dim_axis]
        return tuple([1 if (axis in dyn_axes) else dim for axis, dim in enumerate(self.batch_shape)])

    def get_bc_shape(self, opts=None):
        """
        :param dict[Dim|str|list[Dim|str]|tuple[Dim|str],int|str|None]|None opts:
            ``key`` specifies the axes.
            ``value`` 1 ('x') is broadcasting, -1 (None) is not broadcasting
            Axes should not be defined multiple times.
            The default behavior if an axis is not specified is like :func:`get_bc_spatial_batch_shape`,
            i.e. it will broadcast in batch and spatial dims only.
            Or if "*" is in the dict, this overwrites the default behavior for all axes.
        :return: shape where 1 means broadcasting, None or >1 means not broadcasting.
            can be used for :func:`TFUtil.dropout`
        :rtype: tuple[int|None]
        """
        if opts is None:
            opts = {}
        default_axes_map = dict(enumerate(self.get_bc_spatial_batch_shape()))
        axes_map = {}  # int -> int|None
        for key, value in opts.items():
            assert value in (-1, 1, "x", None), "%r get_bc_shape: invalid value in opts %r" % (self, opts)
            if value == "x":
                value = 1
            if value == -1:
                value = None
            key_axes = self.get_axes_from_description(key)
            for key_axis in key_axes:
                assert key_axis not in axes_map, "%r get_bc_shape: axis %i is defined multiple times in opts %r" % (
                    self,
                    key_axis,
                    opts,
                )
                assert 0 <= key_axis < self.batch_ndim, "%r get_bc_shape: invalid axis %i in opts %r" % (
                    self,
                    key_axis,
                    opts,
                )
                (axes_map if key != "*" else default_axes_map)[key_axis] = (
                    self.batch_shape[key_axis] if value is None else value
                )
        # Fill in remaining axes by defaults, just as in get_bc_spatial_batch_shape.
        remaining_axes = sorted(set(range(self.batch_ndim)).difference(axes_map.keys()))
        for axis in remaining_axes:
            axes_map[axis] = default_axes_map[axis]
        assert sorted(axes_map.keys()) == list(range(self.batch_ndim))
        return tuple([axes_map[i] for i in range(self.batch_ndim)])

    def get_scope_name(self):
        """
        :return: via self.placeholder or any self.size_placeholder, or None
        :rtype: str|None
        """
        if self.placeholder is not None:
            return os.path.dirname(self.placeholder.name)
        if self.size_placeholder:
            for i, v in sorted(self.size_placeholder.items()):
                if v is not None:
                    return os.path.dirname(v.name)
        return None

    def get_full_name(self):
        """
        :return: if we have a defined scope (via :func:`self.get_scope_name`), then scope_name + "/" + self.name,
          otherwise just self.name
        :rtype: str
        """
        scope_name = self.get_scope_name()
        if scope_name:
            return "%s/%s" % (scope_name, self.name)
        return self.name

    def get_dim_tag(self, axis):
        """
        :param int axis: counted with batch-dim
        :rtype: Dim
        """
        return self._dims[axis]

    def get_time_dim_tag(self):
        """
        :rtype: Dim
        """
        assert self.time_dim_axis is not None
        return self.get_dim_tag(self.time_dim_axis)

    def get_dyn_size_tags(self):
        """
        :return: all dim tags with dynamic size
        :rtype: list[Dim]
        """
        return [dim_tag for dim_tag in self._dims if dim_tag.is_dynamic_seq_length()]

    def get_size_dim_tag(self, number):
        """
        :param int number: index in sorted(size_placeholder.keys())
        :rtype: Dim
        """
        dyn_size_tags = self.get_dyn_size_tags()
        return dyn_size_tags[number]

    def get_batch_shape_dim_tags(self):
        """
        :return: list of dimension tags, for each axis (counted with batch dim, i.e. len is batch_ndim)
        :rtype: tuple[Dim]
        """
        return self.dim_tags

    @classmethod
    def get_common_data(
        cls, sources: List[Tensor], ignore_feature_dim=False, allow_broadcast_all_sources=NotSpecified, name=None
    ) -> Optional[Tensor]:
        """
        :param sources:
        :param bool ignore_feature_dim: when set, the feature dim does not have to match in the sources
        :param bool|NotSpecified allow_broadcast_all_sources:
        :param str|None name:
        :return: some generic data where the sources should be compatible to (with copy_compatible_to),
          i.e. it contains the union of all axes from all sources (least common multiple).
          This is always a template, and a new copy.
        """
        from returnn.util import BehaviorVersion

        if not sources:
            return None
        assert sources
        if len(sources) == 1:
            return sources[0].copy_template()
        max_ndim = max([s.batch_ndim for s in sources])
        if any(src.batch for src in sources):
            from returnn.tf.util.data import BatchInfo

            common_batch = BatchInfo.get_common_batch_info([src.batch for src in sources if src.batch])
        else:
            common_batch = None
        # Try with the (first) largest.
        common = [s for s in sources if s.batch_ndim == max_ndim][0]
        common = common.copy_template(name=name)
        common.beam = None  # this will be reset
        if common_batch:
            common.batch = common_batch.copy_set_beam(None)  # the beam will be reset
        if any([s.beam for s in sources]):
            from returnn.tf.util.data import SearchBeam

            # Note: we don't use copy_extend_with_beam
            # because we don't want to create any ops in the TF graph at this point.
            common.beam = SearchBeam.get_combined_beam(*[s.beam for s in sources])
        is_equal_opts = dict(
            ignore_feature_dim=ignore_feature_dim,
            treat_feature_as_spatial=True,
            allow_same_spatial_dim=True,
            undefined_matches=True,
            derived_matches=True,
        )
        if BehaviorVersion.get() < 11:
            is_equal_opts["broadcast_matches"] = True
        all_dim_tags, tags_dict = Dim.get_all_dimension_tags(sources, is_equal_opts=is_equal_opts)
        # Check for missing tags, and add those.
        for dim_tag in all_dim_tags:
            common_tag = Dim.get_existing_tag_from_collection(dim_tag, common.dim_tags, is_equal_opts=is_equal_opts)
            if common_tag:
                # Already have this tag. However, maybe we have a better one.
                # Dim.get_all_dimension_tags() would have selected that.
                if dim_tag != common_tag:
                    axis = common.dim_tags.index(common_tag)
                    common = common.copy_template_replace_dim_tag(axis=axis, new_dim_tag=dim_tag)
            else:
                axis = common.get_default_new_axis_for_dim_tag(dim_tag)
                common = common.copy_add_dim_by_tag(dim_tag, unbroadcast=True, axis=axis)
        if all(s.batch_ndim < common.batch_ndim for s in sources):
            from returnn.util.basic import validate_broadcast_all_sources

            validate_broadcast_all_sources(
                allow_broadcast_all_sources=allow_broadcast_all_sources, inputs=sources, common=common
            )
        return common

    def find_matching_dims(self: Tensor, dim_tag: Dim, is_equal_opts) -> List[int]:
        """
        Finds the dimensions of this Tensor that match another Dim

        :param dim_tag:
        :param dict[str,bool]|None is_equal_opts: passed to Dim.is_equal
        :return: a list of matching axes, counted with batch dim. Sorted in ascending order
        """
        return [axis for axis in range(self.batch_ndim) if self.get_dim_tag(axis).is_equal(dim_tag, **is_equal_opts)]

    def find_matching_dim_map(self: Tensor, other: Tensor, other_axes, is_equal_opts=None) -> Dict[int, int]:
        """
        Looks up all other_axes of another Tensor in this Tensor. Does not allow duplicates.

        :param other:
        :param list[int] other_axes: list of axes of ``other``, counted with batch dim, to be mapped
        :param dict[str,bool]|None is_equal_opts: passed to Dim.is_equal
        :return: dict mapping other axes (from ``other_axes``) to own axes, all counted with batch dim
        """
        if is_equal_opts is None:
            is_equal_opts = dict(
                allow_same_feature_dim=True, allow_same_spatial_dim=True, treat_feature_as_spatial=True
            )

        def map_other_axis_to_self(other_axis: int, taken_self_axes: Set[int]) -> int:
            """
            :param other_axis: counted with batch dim
            :param taken_self_axes: axes that should not be used again
            :return: the axis of ``self`` that matches ``other_axis``, counted with batch dim
            """
            other_axis_dim_tag = other.dims[other_axis]
            is_equal_opts_ = None
            matching = None
            # First, try without any is_equal_opts. This is the most restrictive case.
            # Try with the given is_equal_opts.
            # Try harder by allowing broadcasting to match.
            # If still not, then also allow one single dyn_size to be unknown.
            for opt in [{}, is_equal_opts, "broadcast_matches", "unknown_spatial_matches"]:
                if isinstance(opt, dict):
                    is_equal_opts_ = opt.copy()
                elif isinstance(opt, str):
                    if opt in is_equal_opts_:
                        continue
                    is_equal_opts_[opt] = True
                matching = [
                    self_axis
                    for self_axis in self.find_matching_dims(other_axis_dim_tag, is_equal_opts_)
                    if self_axis not in taken_self_axes
                ]
                if opt == "unknown_spatial_matches":
                    assert (
                        len(matching) <= 1
                    ), "cannot match axes %s from %s to %s, failed at other %s, not unique after %s" % (
                        other_axes,
                        other,
                        self,
                        other_axis,
                        opt,
                    )
                if matching:
                    break
            assert matching, "cannot match the axes %s from %s to %s. Failing at axis %s, tag %s" % (
                other_axes,
                other,
                self,
                other_axis,
                other.dim_tags[other_axis],
            )
            if len(matching) == 1:
                return matching[0]
            # If there are multiple matches (e.g. because two axes have the same feature dim), leave their order intact.
            # We do this by always choosing the first unused match which is the smallest axes.
            # However, take match_priority into account, and prefer the highest match_priority.
            # And if there is a dim identity, prefer that even more.
            max_match_priority = max(dim.match_priority for dim in self.dims)
            return max(
                matching,
                key=lambda ax: (
                    (max_match_priority + 1) if (self.dims[ax] is other_axis_dim_tag) else self.dims[ax].match_priority
                ),
            )

        other_to_self_mapping = {}
        for axis in other_axes:
            other_to_self_mapping[axis] = map_other_axis_to_self(axis, set(other_to_self_mapping.values()))
        assert len(other_to_self_mapping) == len(other_axes), "other_axes may not contain duplicates"
        return other_to_self_mapping

    def is_valid_in_current_graph(self: _t.Tensor) -> bool:
        """
        :return: whether the raw tensor is valid in the current graph.
            In eager mode, this is always True.
        """
        if self._raw_tensor is None:
            return True
        return self._raw_backend.is_valid_in_current_graph(self)

    def mark_as_loss(
        self: Tensor,
        name: str,
        *,
        scale: Optional[float] = 1.0,
        as_error: bool = False,
        use_normalized_loss: bool = False,
        use_flatten_frames: bool = True,
        custom_inv_norm_factor: Optional[Tensor] = None,
    ) -> None:
        """
        Mark this as a loss.
        Please refer to :func:`RunCtx.mark_as_loss` for more details.

        :param name:
        :param scale:
        :param as_error:
        :param use_normalized_loss:
        :param use_flatten_frames:
        :param custom_inv_norm_factor:
        """
        import returnn.frontend as rf

        rf.get_run_ctx().mark_as_loss(
            loss=self,
            name=name,
            scale=scale,
            as_error=as_error,
            use_normalized_loss=use_normalized_loss,
            use_flatten_frames=use_flatten_frames,
            custom_inv_norm_factor=custom_inv_norm_factor,
        )

    def mark_as_output(self: Tensor, name: str, *, shape: Optional[Sequence[Dim]] = None) -> None:
        """
        Mark this as an output.
        See :func:`RunCtx.mark_as_output` for more details.

        :param name:
        :param shape:
        """
        import returnn.frontend as rf

        rf.get_run_ctx().mark_as_output(self, name=name, dims=shape)

    def mark_as_default_output(self: Tensor, *, shape: Optional[Sequence[Dim]] = None) -> None:
        """
        Mark this as the default output.
        See :func:`RunCtx.mark_as_default_output` for more details.

        :param shape:
        """
        import returnn.frontend as rf

        rf.get_run_ctx().mark_as_default_output(self, shape=shape)


def infer_sparse_dim(
    *,
    name: str,
    sparse: Optional[bool] = None,
    sparse_dim,
    dim=NotSpecified,
    **_other_kwargs,
) -> Optional[Dim]:
    """
    :param name:
    :param sparse:
    :param sparse_dim:
    :param dim:
    :return: sparse dim
    """
    if sparse is None:
        sparse = sparse_dim not in (None, NotSpecified)
    if sparse_dim in (None, NotSpecified):
        if sparse:
            assert dim is not NotSpecified, "need dim (num classes) if sparse"
            assert dim is None or isinstance(dim, int)
            sparse_dim = Dim(
                kind=Dim.Types.Feature,
                dimension=dim,
                description="%s:sparse-dim" % name,
                auto_generated=True,
            )
        else:
            sparse_dim = None
    if sparse_dim is not None:
        assert isinstance(sparse_dim, Dim)
        assert sparse_dim.can_be_used_as_dim()
        assert sparse
        if dim is not NotSpecified:
            assert sparse_dim.dimension == dim
    else:
        assert not sparse
    return sparse_dim


def infer_dim_tags(
    *,
    name,
    batch_dim_axis=NotSpecified,
    time_dim_axis=NotSpecified,
    feature_dim_axis=NotSpecified,
    dim_tags: Optional[Sequence[Dim]] = None,
    shape: Optional[Sequence[Optional[int]]] = None,
    sparse_dim: Optional[Dim] = None,
    dim=NotSpecified,
    size_placeholder=None,
    auto_create_placeholders=False,
    batch=None,
    **_other_kwargs,
) -> Tuple[Dim, ...]:
    """
    :param name:
    :param int|None|NotSpecified batch_dim_axis: where we add the batch-dim.
      e.g. shape=(time,...), 0 -> (batch,time,...), 1 -> (time,batch,...).
      Default is 0.
      This is normally always set, and a lot of code expects this. However, you can set it to None
      if this Tensor does not have a batch-dim.
    :param int|None|NotSpecified time_dim_axis: where we have the time dim axis, after we added the batch-dim.
      this is often 1. however, can be None if there is no time-dim.
    :param int|None|NotSpecified feature_dim_axis: feature dim axis. by default it's the last one
    :param dim_tags:
    :param shape: including time-dim (can be None). excluding batch-dim.
        e.g. (time,feat)=(None,128)
    :param sparse_dim:
    :param int|None|NotSpecified dim: feature dimension, shape[-1] if not sparse, otherwise like num_classes
    :param size_placeholder:
    :param auto_create_placeholders:
    :param batch:
    :return: dims
    """
    if dim_tags is not None:
        return tuple(dim_tags)
    if batch_dim_axis is NotSpecified:
        batch_dim_axis = 0
    if shape is None:
        if time_dim_axis is NotSpecified:
            time_dim_axis = _default_time_dim_axis_no_shape(
                batch_dim_axis=batch_dim_axis, feature_dim_axis=feature_dim_axis
            )
        shape, time_dim_axis = _infer_default_shape_and_time(
            batch_dim_axis=batch_dim_axis,
            feature_dim_axis=feature_dim_axis,
            time_dim_axis=time_dim_axis,
            sparse=bool(sparse_dim),
            dim=dim,
        )
    else:
        if time_dim_axis is NotSpecified:
            time_dim_axis = _default_time_dim_axis(batch_dim_axis=batch_dim_axis, shape=shape)
    dims = _infer_dim_tags_tuple_from_shape(
        shape,
        batch_dim_axis=batch_dim_axis,
        time_dim_axis=time_dim_axis,
        feature_dim_axis=feature_dim_axis,
        size_placeholder=size_placeholder,
        name=name,
        extern_data=auto_create_placeholders,
        sparse=bool(sparse_dim),
        batch=batch,
    )
    if dim is not NotSpecified:
        if sparse_dim:
            assert sparse_dim.dimension == dim
        else:
            if feature_dim_axis is None:
                assert dim is None
            elif feature_dim_axis is NotSpecified:
                pass
            else:
                assert dims[feature_dim_axis].dimension == dim
    return dims


class _SizePlaceholderProxy:
    """
    This is a proxy object to emulate the original Tensor.size_placeholder behavior,
    which was a dict[int,tf.Tensor], axis_wo_batch -> sizes.
    """

    def __init__(self, data: Tensor):
        """
        :param data:
        """
        self.data = data

    def _assert_sane_axis_wo_batch(self, idx):
        assert isinstance(idx, int) and 0 <= idx < self.data.ndim

    def __contains__(self, item):
        if not isinstance(item, int):
            return False
        if not 0 <= item < self.data.ndim:
            return False
        return self.data.has_dynamic_size(axis=self.data.get_batch_axis(item))

    def __getitem__(self, item):
        self._assert_sane_axis_wo_batch(item)
        return self.data.get_dynamic_size(axis=self.data.get_batch_axis(item))

    def __setitem__(self, key, value):
        self._assert_sane_axis_wo_batch(key)
        self.data.set_dynamic_size(axis=self.data.get_batch_axis(key), sizes=value)

    def __delitem__(self, key):
        self._assert_sane_axis_wo_batch(key)
        raise Exception("%s: cannot delete items from size_placeholder" % self.data)

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

    def __bool__(self):
        return bool(self.keys())

    __nonzero__ = __bool__  # Python 3 wants __bool__, Python 2.7 wants __nonzero__

    def __repr__(self):
        return repr(self.as_dict())

    def get(self, axis_wo_b, default=None):
        """
        :param int axis_wo_b:
        :param tf.Tensor|None default:
        :rtype: tf.Tensor|None
        """
        if axis_wo_b in self:
            return self[axis_wo_b]
        return default

    def pop(self, axis_wo_b, *default):
        """
        :param int axis_wo_b:
        """
        if default and axis_wo_b not in self:
            (default,) = default
            return default
        res = self[axis_wo_b]
        del self[axis_wo_b]
        return res

    def clear(self):
        """
        Remove all.
        """
        raise Exception("%s: cannot clear size_placeholder" % self.data)

    def keys(self):
        """
        :rtype: list[int]
        """
        return [i for i in range(self.data.ndim) if i in self]

    def values(self):
        """
        :rtype: list[tf.Tensor]
        """
        return [self[i] for i in self.keys()]

    def items(self):
        """
        :rtype: list[(int,tf.Tensor)]
        """
        return [(i, self[i]) for i in self.keys()]

    def copy(self):
        """
        :return: a copy-like object
        :rtype: dict[int,tf.Tensor]
        """
        return self.as_dict()

    def as_dict(self):
        """
        :rtype: dict[int,tf.Tensor]
        """
        return dict(self.items())


def _batch_dim_axis_from_dim_tags_tuple(dim_tags):
    """
    :param Sequence[Dim] dim_tags:
    :return: batch_dim_axis. int or None if not existing
    :rtype: int|None
    """
    for axis, dim_tag in enumerate(dim_tags):
        if dim_tag.is_batch_dim():
            return axis
    return None


def _batch_shape_from_shape(shape, batch_dim_axis):
    """
    :param Sequence[int|None] shape: without batch-dim
    :param int|None batch_dim_axis:
    :return: shape with batch dim if existing
    :rtype: tuple[int|None]
    """
    shape = tuple(shape)
    if batch_dim_axis is not None:
        assert 0 <= batch_dim_axis <= len(shape)
        return shape[:batch_dim_axis] + (None,) + shape[batch_dim_axis:]
    else:
        return shape


# noinspection PyShadowingNames
def _create_size_placeholder(name, axis_wo_b, tag, batch_dim):
    """
    :param str name:
    :param int axis_wo_b:
    :param Dim tag:
    :param Dim|None batch_dim:
    """
    # Note on batch info: Usually, this is called early when no global batch info is initialized yet.
    # Then it is later initialized via ExternData.init_batch_info.
    # Some other external code (e.g. returnn-common) might have set custom batch info
    # on some Tensor instance, and via Tensor._adapt_batch_consistent_dim_tags / Dim.get_for_batch_ctx,
    # that might have been set on uninitialized dim tags as well.
    # Now when we get such dim tags here, they would have some Dim.batch info set
    # but this does not correspond to the global batch info which we will get here.
    # Only trust batch_dim here, or if that batch info is unset, then leave it uninitialized,
    # or even explicitly set it to None, see below.
    from returnn.tf import compat as tf_compat
    from returnn.tf.util.basic import reuse_name_scope

    with reuse_name_scope("extern_data/placeholders/%s" % name, absolute=True):
        dyn_size_name = "%s_dim%i_size" % (name, axis_wo_b)
        if not tag.dyn_size_ext:
            dyn_size_ext = _t.Tensor(
                name=dyn_size_name,
                dtype=_t.Tensor.size_dtype,
                dim_tags=[batch_dim] if batch_dim else [],
                batch=None,  # expected in ExternData.init_batch_info
            )
        else:
            dyn_size_ext = tag.dyn_size_ext.copy_template()
            dyn_size_ext.batch = None  # expected in ExternData.init_batch_info
        dyn_size = tf_compat.v1.placeholder(
            name=dyn_size_name, dtype=dyn_size_ext.dtype, shape=dyn_size_ext.batch_shape
        )
        dyn_size_ext.placeholder = dyn_size
        if dyn_size_ext.batch:
            tag.set_dyn_size_ext_for_batch_ctx(
                batch=dyn_size_ext.batch, ctx=dyn_size_ext.control_flow_ctx, dyn_size_ext=dyn_size_ext
            )
            # Do not set tag.batch. set_dyn_size_ext_for_batch_ctx should cover this.
        else:
            tag.reset_batch_ctx()  # reset, it is anyway invalid, see above
            tag.dyn_size_ext = dyn_size_ext
        tag.set_tag_on_size_tensor(dyn_size)


def _infer_dim_tags_tuple_from_shape(
    shape, batch_dim_axis, time_dim_axis, feature_dim_axis, sparse, batch, size_placeholder, name, extern_data
):
    """
    :param tuple[int|None]|list[int|None] shape: this is without batch-dim-axis
    :param int|None batch_dim_axis:
    :param int|None time_dim_axis:
    :param int|None|NotSpecified feature_dim_axis:
    :param bool sparse:
    :param BatchInfo|None batch:
    :param dict[int,tf.Tensor]|None size_placeholder: key is axis without batch-dim
    :param bool extern_data:
    :param str name:
    :return: dim tags tuple
    :rtype: tuple[Dim]
    """
    assert isinstance(shape, (tuple, list))
    shape = tuple(shape)
    batch_shape = _batch_shape_from_shape(shape, batch_dim_axis=batch_dim_axis)
    if feature_dim_axis is NotSpecified:
        feature_dim_axis = _default_feature_dim_axis(
            batch_dim_axis=batch_dim_axis, time_dim_axis=time_dim_axis, batch_shape=batch_shape, sparse=sparse
        )
    elif feature_dim_axis is not None:
        if feature_dim_axis < 0:
            feature_dim_axis += len(batch_shape)
        assert 0 <= feature_dim_axis < len(batch_shape)
    dim_tags = {}
    if batch_dim_axis is not None and batch_dim_axis not in dim_tags:
        if batch is None or batch.is_global_batch():
            batch_dim_ = batch_dim  # global batch dim
        else:
            batch_dim_ = Dim(kind=Dim.Types.Batch, description="batch:%s" % name, batch=batch, dimension=None)
        dim_tags[batch_dim_axis] = batch_dim_
    # Note: Consistent to Tensor.get_dim_tag,
    # prefer interpretation as spatial axis if there is a dynamic size or this is marked as time axis.
    if size_placeholder:
        for axis_wo_b, size in size_placeholder.items():
            axis = _get_axis_wb(axis_wo_b, batch_dim_axis=batch_dim_axis)
            if axis in dim_tags:
                continue
            tag = Dim.get_tag_from_size_tensor(size)
            if tag:
                dim_tags[axis] = tag
    # See Tensor.get_spatial_batch_axes
    spatial_axes = [
        axis
        for axis in range(len(batch_shape))
        if axis != batch_dim_axis and (axis != feature_dim_axis or axis == time_dim_axis or batch_shape[axis] is None)
    ]
    for axis in range(len(batch_shape)):
        tag = dim_tags.get(axis)
        axis_wo_b = _get_axis_wo_b(axis, batch_dim_axis=batch_dim_axis)
        dyn_size = size_placeholder.get(axis_wo_b) if (size_placeholder and axis_wo_b is not None) else None
        dim = batch_shape[axis]
        if extern_data and dim is None and dyn_size is None and axis != batch_dim_axis:
            if not tag:
                if axis == time_dim_axis:
                    tag_name = "time"
                else:
                    tag_name = "spatial%i" % axis
                tag = Dim(
                    description="%s:var:extern_data:%s" % (tag_name, name),
                    # Spatial dim tag, even if axis == feature_dim_axis. This is to keep the old behavior.
                    # This is such that Dim.is_equal behaves as before, e.g. in Tensor.get_common_data.
                    kind=Dim.Types.Spatial,
                    batch=batch,
                    auto_generated=True,
                    dimension=None,
                )
                tag.dyn_size_ext = _t.Tensor(
                    name="%s_dim%i_size" % (name, axis_wo_b), dtype=_t.Tensor.size_dtype, shape=(), batch=batch
                )
                dim_tags[axis] = tag
            dyn_size = tag.dyn_size
        if tag:
            # Just some sanity checks.
            assert isinstance(tag, Dim)
            assert tag.dimension == dim
            if dyn_size is not None:
                assert tag.is_same_size_tensor(dyn_size)
            continue
        if axis == feature_dim_axis and dyn_size is None and axis != time_dim_axis:
            tag = Dim(
                kind=Dim.Types.Feature,
                dimension=dim,
                description="feature:%s" % name,
                batch=batch if dim is None else None,
                undefined=dim is None,
                auto_generated=True,
            )
        else:
            assert axis in spatial_axes
            description = "time" if axis == time_dim_axis else "spatial%i" % spatial_axes.index(axis)
            if dyn_size is not None:
                # Note: This case is uncommon/unexpected
                # (we should have a dim-tag on the dyn_size above),
                # so be verbose,
                # and fix such cases if possible
                # (i.e. for all newly created dynamic size tensors, set the dim-tag).
                description += ":var:%r" % dyn_size.name
            elif dim is None:
                description += ":var-unk"
            else:
                description += ":static%i" % dim
            description += ":%s" % name
            tag = Dim(
                kind=Dim.Types.Spatial,
                description=description,
                dimension=dim,
                dyn_size=dyn_size,
                batch=batch if dim is None else None,
                undefined=dim is None and dyn_size is None,
                auto_generated=True,
            )
        if dim is None and tag.dyn_size_ext is None:
            tag.dyn_size_ext = _t.Tensor(
                name="%s_dim%i_size" % (name, axis_wo_b), dtype=_t.Tensor.size_dtype, shape=(), batch=batch
            )
            if dyn_size is not None:
                tag.dyn_size_ext.placeholder = dyn_size
        dim_tags[axis] = tag
    assert sorted(dim_tags.keys()) == list(range(len(batch_shape)))
    return tuple(dim_tags[axis] for axis in range(len(batch_shape)))


def _auto_create_size_placeholders_on_dim_tags(name, dim_tags):
    """
    :param str name:
    :param tuple[Dim] dim_tags:
    """
    batch_dim_axis = _batch_dim_axis_from_dim_tags_tuple(dim_tags)
    batch_dim_ = dim_tags[batch_dim_axis] if batch_dim_axis is not None else None
    if batch_dim_:
        # Do this first, in case the batch dim is used elsewhere,
        # to avoid that we use some invalid batch info.
        # noinspection PyProtectedMember
        batch_dim_._validate_in_current_graph()
    for axis, tag in enumerate(dim_tags):
        # noinspection PyProtectedMember
        tag._validate_in_current_graph()
        if tag.is_batch_dim():
            continue
        if not tag.is_dynamic():
            continue
        if tag.dyn_size is not None:
            continue
        axis_wo_b = _get_axis_wo_b(axis, batch_dim_axis=batch_dim_axis)
        _create_size_placeholder(name=name, axis_wo_b=axis_wo_b, tag=tag, batch_dim=batch_dim_)


def _get_axis_wo_b(axis_wb, batch_dim_axis, batch_ndim=None):
    """
    :param int axis_wb: counted with batch-dim
    :param int|None batch_dim_axis:
    :param int|None batch_ndim: only used for axis_wb < 0. might be unknown (None)
    :return: axis counted without batch-dim
    :rtype: int|None
    """
    if axis_wb < 0:
        assert batch_ndim is not None
        assert axis_wb + batch_ndim >= 0
        axis_wb += batch_ndim
        # Do this check only in this case;
        # we call this function early in construction where batch_ndim might be invalid.
        assert 0 <= axis_wb < batch_ndim
    if batch_dim_axis is None:
        return axis_wb
    if axis_wb == batch_dim_axis:
        return None
    if axis_wb < batch_dim_axis:
        return axis_wb
    return axis_wb - 1


def _get_axis_wb(axis_wo_b, batch_dim_axis):
    """
    :param int axis_wo_b: counted without batch-dim
    :param int|None batch_dim_axis:
    :return: axis counted with batch-dim
    :rtype: int
    """
    if batch_dim_axis is None:
        return axis_wo_b
    if axis_wo_b >= batch_dim_axis:
        return axis_wo_b + 1
    return axis_wo_b


def _infer_default_shape_and_time(batch_dim_axis, time_dim_axis, feature_dim_axis, sparse, dim):
    """
    This is the logic to infer some sensible/default shape when it is not specified.
    As this is somewhat adhoc, this is not recommended to be used anymore.

    :param int|None batch_dim_axis:
    :param int|None time_dim_axis:
    :param int|None|NotSpecified feature_dim_axis:
    :param bool sparse:
    :param int|None dim:
    :return: shape (without batch dim), time_dim_axis
    :rtype: (tuple[int|None],int|None)
    """
    if time_dim_axis is not None:
        assert time_dim_axis != batch_dim_axis
        shape = (None,) * (_get_axis_wo_b(time_dim_axis, batch_dim_axis=batch_dim_axis) + 1)
    else:  # no time-dim-axis
        shape = ()
    if not sparse and feature_dim_axis is not None:
        assert dim is not NotSpecified, "no shape specified, not sparse, feature_dim_axis existing -> need dim"
        if feature_dim_axis is NotSpecified or feature_dim_axis == -1:
            shape = shape + (dim,)
        else:
            assert 0 <= feature_dim_axis != batch_dim_axis
            feature_dim_axis_wo_batch = _get_axis_wo_b(feature_dim_axis, batch_dim_axis=batch_dim_axis)
            if feature_dim_axis_wo_batch < len(shape):
                shape = shape[:-feature_dim_axis_wo_batch] + (dim,) + shape[feature_dim_axis_wo_batch + 1 :]
            else:
                shape = shape + (None,) * (feature_dim_axis_wo_batch - len(shape)) + (dim,)
                assert len(shape) == feature_dim_axis_wo_batch + 1
    return shape, time_dim_axis


def _default_time_dim_axis(batch_dim_axis, shape):
    """
    :param int|None batch_dim_axis:
    :param Sequence[int|None] shape: without batch-dim
    :return: time dim axis, counted with batch-dim
    :rtype: int|None
    """
    if batch_dim_axis is None:
        time_dim_axis = None
    else:
        # Do not select the batch dim axis, or any axis with None dim.
        # Note that we currently allow to select the same as the feature dim axis,
        # in case the feature dim is None.
        taken_axes = {batch_dim_axis}
        batch_shape = _batch_shape_from_shape(shape, batch_dim_axis=batch_dim_axis)
        for axis, _dim in enumerate(batch_shape):
            if _dim is not None:
                taken_axes.add(axis)
        available_axes = [i for i in range(len(batch_shape)) if i not in taken_axes]
        if available_axes:
            time_dim_axis = available_axes[0]
        else:
            time_dim_axis = None
    return time_dim_axis


def _default_time_dim_axis_no_shape(batch_dim_axis, feature_dim_axis):
    """
    :param int|None batch_dim_axis:
    :param int|None|NotSpecified feature_dim_axis:
    :return: time dim axis, counted with batch-dim
    :rtype: int|None
    """
    if batch_dim_axis is None:
        time_dim_axis = None
    else:
        # By default if not specified, we have a time dim.
        taken_axes = {batch_dim_axis}
        if isinstance(feature_dim_axis, int):
            taken_axes.add(feature_dim_axis)
        time_dim_axis = [i for i in range(max(taken_axes) + 2) if i not in taken_axes][0]
    return time_dim_axis


def _default_time_dim_axis_dim_tags(dim_tags):
    """
    :param list[Dim]|tuple[Dim] dim_tags:
    :return: time dim axis, counted with batch-dim
    :rtype: int|None
    """
    # Consistent to _default_time_dim_axis.
    # Any spatial dynamic axis.
    # Or otherwise any dynamic axis (including maybe feature).
    # Not using any static axes.
    dim_tags_dyn_spatial = [i for i, tag in enumerate(dim_tags) if tag.is_spatial_dim() and tag.dimension is None]
    if dim_tags_dyn_spatial:
        return dim_tags_dyn_spatial[0]
    dim_tags_dyn = [i for i, tag in enumerate(dim_tags) if not tag.is_batch_dim() and tag.dimension is None]
    if dim_tags_dyn:
        return dim_tags_dyn[0]
    return None


def _default_feature_dim_axis(batch_dim_axis, time_dim_axis, batch_shape, sparse):
    """
    :param int|None batch_dim_axis:
    :param int|None time_dim_axis:
    :param tuple[int|None] batch_shape:
    :param bool sparse:
    :return: feature dim axis, counted with batch-dim
    :rtype: int|None
    """
    if sparse:
        return None
    batch_ndim = len(batch_shape)
    ndim = batch_ndim if batch_dim_axis is None else (batch_ndim - 1)
    if ndim == 0:
        return None
    axes = [i for i in range(batch_ndim) if i not in [batch_dim_axis, time_dim_axis]]
    if not axes:
        return None
    static_axes = [i for i in axes if batch_shape[i] is not None]
    # Prefer last static, if available.
    if static_axes:
        return static_axes[-1]
    return axes[-1]
