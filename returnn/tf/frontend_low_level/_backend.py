"""
Backend for exposing TensorFlow-specific functionality.
"""

from __future__ import annotations
from typing import Optional, Any, Dict, List, Union, Sequence, Tuple
from dataclasses import dataclass
import contextlib
import string
import numpy
import tensorflow as tf

import returnn.tf.compat as tf_compat
from returnn.util.basic import NotSpecified, is_onnx_export_global, get_global_inf_value, RefIdEq
from returnn.tensor import Tensor, Dim
from returnn.tf.util import basic as tf_util

# noinspection PyProtectedMember
from returnn.frontend._backend import Backend, register_backend_by_tensor_type
from returnn.frontend import RawTensorTypes
import returnn.frontend as rf

# noinspection PyProtectedMember
from returnn.frontend import _random_journal

__all__ = ["TFBackend", "DeferredVariable"]

_TT = Tensor[tf.Tensor]


class DeferredVariable:
    """
    Stands in as the raw tensor of an rf.Parameter whose tf.Variable does not exist yet,
    see :func:`TFBackend.deferred_parameter_creation`.

    It carries only what parameter construction needs (dtype, shape);
    any attempt to use it in an op raises, at graph construction time.
    """

    def __init__(self, *, dtype: tf.DType, shape: Sequence[int]):
        self.dtype = dtype
        self.shape = tf.TensorShape(shape)

    def __repr__(self):
        return f"<DeferredVariable {self.dtype.name} {self.shape.as_list()}>"

    def set_shape(self, shape):
        """
        No-op. The shape is known and fixed; this exists because the Tensor.raw_tensor setter calls it.

        :param shape:
        """

    def __tf_tensor__(self, dtype=None, name=None):
        raise RuntimeError(
            f"{self}: the parameter variable was not created yet."
            f" TFBackend.create_parameters(model) must run before the model is used."
        )


@dataclass
class _DeferredParam:
    """What we know about a parameter before its tf.Variable exists."""

    initial: Optional[tf.Tensor] = None
    trainable: bool = True
    device: Optional[str] = None


# Ignore this warning until we really expect that we implemented everything.
# noinspection PyAbstractClass
class TFBackend(Backend[tf.Tensor]):
    """
    TensorFlow low-level backend, operating on tf.Tensor
    """

    name = "tf"
    RawTensorType = tf.Tensor
    is_tensorflow = True

    @staticmethod
    def executing_eagerly() -> bool:
        """
        :return: whether we are in eager execution mode
        """
        return tf.executing_eagerly()

    @staticmethod
    def should_pickle_tensor(raw_tensor: tf.Tensor) -> bool:
        """
        :return: whether the tensor should be included in a pickle or set to `None`.
        """

        from tensorflow.python.framework.ops import EagerTensor

        # Can not pickle symbolic TF tensors.
        #
        # See for discussion:
        #  - https://github.com/rwth-i6/returnn/issues/1541
        #  - https://github.com/rwth-i6/returnn/issues/1763
        return isinstance(raw_tensor, EagerTensor)

    @staticmethod
    def get_dtype_name_raw(raw_tensor: tf.Tensor) -> str:
        """
        :return: dtype of raw tensor, as string
        """
        return raw_tensor.dtype.base_dtype.name

    @staticmethod
    def as_dtype_raw(dtype_name: str) -> tf.DType:
        """
        :param dtype_name: e.g. "float32"
        :return: dtype object
        """
        dtype = getattr(tf, dtype_name)
        assert isinstance(dtype, tf.DType)
        return dtype

    @staticmethod
    def get_ndim_raw(raw_tensor: tf.Tensor) -> int:
        """
        :return: ndim of raw tensor. assumes it is known
        """
        assert raw_tensor.shape.ndims is not None
        return raw_tensor.shape.ndims

    @staticmethod
    def get_shape_raw(raw_tensor: tf.Tensor) -> tf.Tensor:
        """
        :return: shape of raw tensor
        """
        with tf_util.same_control_flow_ctx(raw_tensor):
            return tf.shape(raw_tensor)

    @staticmethod
    def get_shape_tuple_raw(raw_tensor: tf.Tensor) -> Tuple[Union[int, tf.Tensor], ...]:
        """
        :return: shape of raw tensor. assumes that ndim is known
        """
        shape = raw_tensor.shape.as_list()
        if all([dim is not None for dim in shape]):
            return tuple(shape)
        with tf_util.same_control_flow_ctx(raw_tensor):
            shape_dynamic = tf.shape(raw_tensor)
            for axis, dim in enumerate(shape):
                if dim is None:
                    shape[axis] = shape_dynamic[axis]
            return tuple(shape)

    @staticmethod
    def get_known_shape_raw(raw_tensor: tf.Tensor) -> Tuple[Optional[int], ...]:
        """
        :return: shape of raw tensor, int for static known, None otherwise. assumes that ndim is known.
        """
        return tuple(raw_tensor.shape.as_list())

    @staticmethod
    def set_known_shape_raw(raw_tensor: tf.Tensor, shape: Tuple[Optional[int], ...]) -> None:
        """
        wrap tf.Tensor.set_shape
        """
        raw_tensor.set_shape(shape)

    @staticmethod
    def fill_raw(shape: Union[Sequence[Union[int, tf.Tensor]], tf.Tensor], value: Union[Any, tf.Tensor]) -> tf.Tensor:
        """
        :param shape: shape
        :param value: value to fill
        :return: raw tensor filled with value everywhere
        """
        with tf_util.same_control_flow_ctx([shape, value]):
            return tf.fill(shape, value)

    @staticmethod
    def compare_raw(a: tf.Tensor, kind: str, b: tf.Tensor) -> tf.Tensor:
        """
        :param a:
        :param kind: "equal", "less", "less_equal", "greater", "greater_equal", "not_equal"
        :param b:
        :return: a `kind` b
        """
        assert a.shape.ndims == b.shape.ndims or a.shape.ndims == 0 or b.shape.ndims == 0
        op = getattr(tf, kind)  # e.g. tf.equal
        with tf_util.same_control_flow_ctx([a, b]):
            return op(a, b)

    @staticmethod
    def combine_raw(a: tf.Tensor, kind: str, b: tf.Tensor) -> tf.Tensor:
        """
        :param a:
        :param kind: "add", "sub", "mul", "truediv", "floordiv", "mod", "pow",
            "maximum", "minimum", "logical_and", "logical_or", "squared_difference"
        :param b:
        :return: a `kind` b
        """
        assert a.shape.ndims == b.shape.ndims or a.shape.ndims == 0 or b.shape.ndims == 0
        if kind == "floordiv" and is_onnx_export_global():
            op = tf_util.onnx_compat_floor_div
        elif kind == "logaddexp":
            # TF has no logaddexp op; this is the numerically stable form
            op = lambda a_, b_: tf.maximum(a_, b_) + tf.math.log1p(tf.exp(-tf.abs(a_ - b_)))  # noqa: E731
        else:
            kind = {
                "sub": "subtract",
                "mul": "multiply",
            }.get(kind, kind)
            op = getattr(tf, kind, None)  # e.g. tf.add
            # In tf v2, some ops like floordiv or mod exist in the tf.math namespace instead
            if op is None:
                op = getattr(tf.math, kind)
        with tf_util.same_control_flow_ctx([a, b]):
            return op(a, b)

    @staticmethod
    def where(
        cond: Tensor,
        true_: Union[Tensor, rf.RawTensorTypes],
        false_: Union[Tensor, rf.RawTensorTypes],
        *,
        allow_broadcast_all_sources: bool = False,
    ) -> Tensor:
        """where"""
        if isinstance(true_, Tensor):
            dtype = true_.dtype
        elif isinstance(false_, Tensor):
            dtype = false_.dtype
        else:
            dtype = None
        true_ = rf.convert_to_tensor(true_, _backend=TFBackend, device=cond.device, dtype=dtype)
        false_ = rf.convert_to_tensor(false_, _backend=TFBackend, device=cond.device, dtype=dtype)
        out = Tensor.get_common_data(
            [true_, false_, cond], allow_broadcast_all_sources=allow_broadcast_all_sources, name="where"
        )
        out.dtype = true_.dtype
        out.sparse_dim = true_.sparse_dim or false_.sparse_dim
        out.feature_dim = true_.feature_dim or false_.feature_dim
        cond_bc_raw = cond.copy_compatible_to_dims_raw(out.dims)
        true_bc_raw = true_.copy_compatible_to_dims_raw(out.dims)
        false_bc_raw = false_.copy_compatible_to_dims_raw(out.dims)
        out.raw_tensor = tf_util.where_bc(cond_bc_raw, true_bc_raw, false_bc_raw)
        return out

    @staticmethod
    def reshape_raw(raw_tensor: tf.Tensor, shape: Union[Sequence[Union[int, tf.Tensor]], tf.Tensor]) -> tf.Tensor:
        """
        :param raw_tensor: raw tensor
        :param shape: new shape
        :return: reshaped raw tensor
        """
        with tf_util.same_control_flow_ctx([raw_tensor, shape]):
            return tf.reshape(raw_tensor, shape)

    @classmethod
    def squeeze_raw(cls, raw_tensor: tf.Tensor, axes: Sequence[int]) -> tf.Tensor:
        """
        :param raw_tensor: raw tensor
        :param axes: axes to squeeze
        :return: squeezed raw tensor
        """
        known_shape = raw_tensor.shape.as_list()
        assert all([known_shape[axis] == 1 for axis in axes])
        with tf_util.same_control_flow_ctx(raw_tensor):
            return tf.squeeze(raw_tensor, axis=axes)

    @staticmethod
    def transpose_raw(raw_tensor: tf.Tensor, perm: Sequence[int]) -> tf.Tensor:
        """
        :param raw_tensor:
        :param perm: e.g. [0, 2, 1]
        :return: permuted (transposed) raw tensor; wraps tf.transpose
        """
        with tf_util.same_control_flow_ctx(raw_tensor):
            return tf.transpose(raw_tensor, perm)

    @staticmethod
    def expand_dims_raw(raw_tensor: tf.Tensor, axis: int) -> tf.Tensor:
        """
        :param raw_tensor:
        :param axis: e.g. 1
        :return: raw tensor with new axis; wraps tf.expand_dims
        """
        with tf_util.same_control_flow_ctx(raw_tensor):
            return tf.expand_dims(raw_tensor, axis=axis)

    @staticmethod
    def expand_raw(raw_tensor: tf.Tensor, axis: int, dim: Union[int, tf.Tensor]) -> tf.Tensor:
        """
        :param raw_tensor:
        :param axis: shape[axis] must be 1
        :param dim: the new dim for shape[axis]
        :return: shape[axis] expands to dim
        """
        assert raw_tensor.shape.as_list()[axis] == 1
        with tf_util.same_control_flow_ctx(raw_tensor):
            return tf.tile(raw_tensor, [1] * axis + [dim] + [1] * (raw_tensor.shape.ndims - axis - 1))

    @staticmethod
    def copy(tensor: Tensor) -> Tensor:
        """copy"""
        out = tensor.copy_template()
        with tf_util.same_control_flow_ctx(tensor):
            out.raw_tensor = tf.identity(tensor.raw_tensor)
        return out

    @staticmethod
    def cast_raw(raw_tensor: tf.Tensor, dtype: str) -> tf.Tensor:
        """cast"""
        with tf_util.same_control_flow_ctx(raw_tensor):
            return tf.cast(raw_tensor, dtype)

    @staticmethod
    def activation_raw(raw_tensor: tf.Tensor, func: str) -> tf.Tensor:
        """
        :param raw_tensor:
        :param func: e.g. "tanh"
        :return: raw tensor after activation
        """
        assert func in Backend._AllowedActivationFuncs
        func = _ACTIVATION_FUNC_NAME_MAP.get(func, func)
        if hasattr(tf.math, func):
            f = getattr(tf.math, func)
        elif hasattr(tf.nn, func):
            f = getattr(tf.nn, func)
        elif hasattr(tf, func):
            f = getattr(tf, func)
        else:
            raise ValueError(f"unknown activation function {func!r}")
        with tf_util.same_control_flow_ctx(raw_tensor):
            return f(raw_tensor)

    @staticmethod
    def stop_gradient(tensor: Tensor) -> Tensor:
        """
        :param tensor:
        :return: tensor with stopped gradient
        """
        out = tensor.copy_template("stop_gradient")
        with tf_util.same_control_flow_ctx(tensor):
            out.raw_tensor = tf.stop_gradient(tensor.raw_tensor)
        return out

    @staticmethod
    def softmax(tensor: Tensor, *, axis: Dim, use_mask: bool = True) -> Tensor:
        """
        :param tensor:
        :param axis:
        :param use_mask:
        :return: softmax over axis
        """
        out = tensor.copy_template("softmax")
        with tf_util.same_control_flow_ctx(tensor):
            x_raw = tensor.raw_tensor
            if use_mask and axis.need_masking():
                mask = _seq_mask_raw(tensor, tensor.dims.index(axis))
                x_raw = tf_util.where_bc(mask, x_raw, -get_global_inf_value())
            out.raw_tensor = tf.nn.softmax(x_raw, axis=tensor.dims.index(axis))
        return out

    @staticmethod
    def log_softmax(tensor: Tensor, *, axis: Dim, use_mask: bool = True) -> Tensor:
        """
        :param tensor:
        :param axis:
        :param use_mask:
        :return: log_softmax over axis
        """
        out = tensor.copy_template("log_softmax")
        with tf_util.same_control_flow_ctx(tensor):
            x_raw = tensor.raw_tensor
            if use_mask and axis.need_masking():
                mask = _seq_mask_raw(tensor, tensor.dims.index(axis))
                x_raw = tf_util.where_bc(mask, x_raw, -get_global_inf_value())
            out.raw_tensor = tf.nn.log_softmax(x_raw, axis=tensor.dims.index(axis))
        return out

    @staticmethod
    def have_sequence_mask_raw() -> bool:
        """
        :return: whether we have sequence_mask
        """
        return True

    @staticmethod
    def sequence_mask_raw(lengths: tf.Tensor, *, batch_major: bool = True) -> tf.Tensor:
        """
        Wraps around tf.sequence_mask().
        It will cache the value inside the passed object so that we don't recompute it multiple times.

        :param lengths: shape (batch,)
        :param batch_major:
        :return: tensor mask of shape (batch,maxlen) if batch_major else (maxlen,batch) of type bool
        """
        if batch_major:
            return tf_util.sequence_mask(lengths)
        else:
            return tf_util.sequence_mask_time_major(lengths)

    @staticmethod
    @contextlib.contextmanager
    def name_scope_raw(name: str) -> Any:
        """
        :param name: name of scope
        :return: context manager
        """
        with tf.name_scope(name):
            yield

    @staticmethod
    @contextlib.contextmanager
    def control_dependencies_raw(dependencies: Sequence[Union[tf.Tensor, tf.Operation]]) -> Any:
        """
        :param dependencies: list of tensors or operations
        :return: context manager
        """
        with tf.control_dependencies(dependencies):
            yield

    @staticmethod
    def identity_with_control_dependencies_raw(raw_tensor: tf.Tensor, dependencies: Sequence[Any]) -> tf.Tensor:
        """
        :param raw_tensor:
        :param dependencies: list of tensors or operations
        :return: identity of tensor with control dependencies
        """
        with tf.control_dependencies(dependencies), tf_util.same_control_flow_ctx(raw_tensor):
            return tf.identity(raw_tensor)

    @staticmethod
    def create_placeholder_raw(tensor: _TT) -> tf.Tensor:
        """
        :return: tf.placeholder in TF
        """
        with tf.name_scope("extern_data/placeholders/%s/" % tensor.name):
            return tf_compat.v1.placeholder(**tensor.get_placeholder_kwargs(with_batch=True))

    @staticmethod
    def runtime_sanity_checks(tensor: _TT) -> tf.Operation:
        """
        Runtime checks
        """
        checks = []
        with tf.name_scope("runtime_sanity_check"), tf_util.same_control_flow_ctx(tensor):
            shape = tf.shape(tensor.placeholder)
            # noinspection PyShadowingNames
            batch_dim = shape[tensor.batch_dim_axis] if tensor.have_batch_axis() else 1
            rank = tf.rank(tensor.placeholder)
            data = ["Data.get_runtime_sanity_check_op:", str(tensor), "shape", shape]
            for i, tag in enumerate(tensor.dim_tags):
                if tag.dyn_size is not None:
                    data += ["dyn_size[%i] (%s)" % (i, tag), tag.dyn_size, ".shape", tf.shape(tag.dyn_size)]
            checks += [tf.Assert(tf.equal(rank, tensor.batch_ndim), data + ["-> invalid rank"])]
            if tensor.have_batch_axis():
                batch_dim_via_info = tensor.get_batch_dim()
                checks += [
                    tf.Assert(
                        tf.equal(batch_dim, batch_dim_via_info),
                        data + ["-> invalid batch dim info", batch_dim_via_info],
                    )
                ]
            for i in range(tensor.batch_ndim):
                if tensor.batch_shape[i] is not None:
                    checks += [
                        tf.Assert(tf.equal(shape[i], tensor.batch_shape[i]), data + ["-> invalid shape[%i]" % i])
                    ]
                dyn_size_ext = tensor.dim_tags[i].dyn_size_ext
                if dyn_size_ext is not None and dyn_size_ext.placeholder is not None:
                    dyn_size = dyn_size_ext.placeholder
                    if dyn_size_ext.have_batch_axis() and tensor.have_batch_axis():
                        checks += [
                            tf.Assert(
                                tf.equal(tf.shape(dyn_size)[dyn_size_ext.batch_dim_axis], batch_dim),
                                data + ["-> invalid axis %i tag dyn size batch dim" % i],
                            )
                        ]
                    checks += [
                        tf.Assert(
                            # Note: in almost all cases, we have equality here.
                            # However, not strictly in all cases, e.g. DecideLayer, maybe some others...
                            # But that should not be more than 1 less.
                            tf.logical_or(
                                tf.logical_and(
                                    tf.less_equal(tf.reduce_max(dyn_size), shape[i]),
                                    tf.greater_equal(tf.reduce_max(dyn_size), shape[i] - 1),
                                ),
                                # In other rare cases, this might be a broadcast dim
                                # (e.g. as initial values of att weights for a rec loop).
                                tf.equal(1, shape[i]),
                            ),
                            data + ["-> invalid shape[%i] or max(dyn_size[%i])" % (i, i)],
                        )
                    ]
                    dyn_size_ext_sanity_checks_op = dyn_size_ext.get_runtime_sanity_check_op()
                    assert dyn_size_ext_sanity_checks_op is not None, f"{dyn_size_ext} {dyn_size_ext.raw_tensor}?"
                    checks += [dyn_size_ext_sanity_checks_op]
            return tf.group(*checks)

    @staticmethod
    def is_valid_in_current_graph(tensor: _TT) -> bool:
        """
        :return: whether the tensor is valid in the current graph
        """
        if tensor.raw_tensor is None:
            return True
        if tf_compat.executing_eagerly():
            return True  # always true in eager mode
        g = tf_util.get_root_graph()
        return tf_util.get_root_graph(tensor.raw_tensor.graph) is g

    @staticmethod
    def format_graph_output(raw_tensor: tf.Tensor, *, max_depth: Optional[int] = None) -> str:
        """
        :param raw_tensor:
        :param max_depth:
        """
        return tf_util.format_graph_output(raw_tensor, max_depth=max_depth)

    @staticmethod
    def convert_to_tensor(
        value: Union[_TT, tf.Tensor, RawTensorTypes],
        *,
        dims: Sequence[Dim],
        dtype: str,
        sparse_dim: Optional[Dim] = None,
        feature_dim: Optional[Dim] = None,
        device: Optional[str] = None,
        name: Optional[str] = None,
    ) -> _TT:
        """convert to tensor"""
        if isinstance(value, Tensor):
            return value
        with tf.control_dependencies(None):
            value = tf.convert_to_tensor(value, dtype=dtype)
        assert isinstance(value, tf.Tensor)
        return Tensor(
            name or "const", raw_tensor=value, dims=dims, dtype=dtype, sparse_dim=sparse_dim, feature_dim=feature_dim
        )

    @staticmethod
    def range_over_dim(dim: Dim, *, dtype: Optional[str] = None, device: Optional[str] = None) -> _TT:
        """
        :param dim:
        :param dtype:
        :param device:
        :return: range over dim
        """
        if not dtype and dim.dyn_size_ext is not None:
            dtype = dim.dyn_size_ext.dtype
        if not dtype:
            dtype = rf.get_default_array_index_dtype()
        out = Tensor(
            name=dim.description or "range_over_dim",
            dims=[dim],
            sparse_dim=dim if dtype.startswith("int") or dtype.startswith("uint") else None,
            dtype=dtype,
        )
        dim_value = dim.get_dim_value()
        with tf_util.same_control_flow_ctx(dim_value):
            out.raw_tensor = tf.range(0, dim_value, dtype=out.dtype)
        return out

    @staticmethod
    def reduce(source: _TT, *, mode: str, axis: Union[Dim, Sequence[Dim]], use_mask: bool = True) -> _TT:
        """Reduce"""
        assert mode in Backend._AllowedReduceModes
        with tf_util.same_control_flow_ctx(source):
            x = source
            axes = x.get_axes_from_description(axis)
            if use_mask in (None, NotSpecified):
                use_mask = any(x.has_dynamic_size(a) for a in axes)
            out_data = x.copy_template()
            dim_tags = [dim_tag for i, dim_tag in enumerate(x.dim_tags) if i not in axes]
            out_data = out_data.copy_template_new_dim_tags(dim_tags)
            sparse_out = mode.lower().startswith("arg")
            if sparse_out:
                assert len(axes) == 1
                out_data.sparse_dim = x.dim_tags[axes[0]]
                out_data.dtype = "int32"
            assert isinstance(use_mask, bool)
            mode = mode.lower()
            reduce_abs_funcs = {
                name: getattr(tf, "reduce_%s" % name) for name in ["max", "min", "sum", "logsumexp", "any", "all"]
            }
            reduce_rel_func = {"mean": tf.reduce_mean}
            arg_funcs = {name: getattr(tf, name) for name in ["argmax", "argmin"]}
            funcs = dict(list(reduce_abs_funcs.items()) + list(reduce_rel_func.items()) + list(arg_funcs.items()))
            assert mode in funcs, "invalid mode %r. choose from: %r" % (mode, funcs)
            f = funcs[mode]
            x_ = x.placeholder
            # Check if we should ignore some frames, e.g. via masking.
            correction_factor = None
            if use_mask and any(x.has_dynamic_size(a) for a in axes):
                if x.batch_dim_axis in axes and x.time_dim_axis in axes and len(axes) == 2:
                    assert mode not in arg_funcs, "unexpected arg reduce for multiple axes"
                    # Flattening.
                    axes = [a if (a < x.time_dim_axis) else (a - 1) for a in axes if a != x.time_dim_axis]
                    x = x.copy_time_flattened()
                    x_ = x.placeholder

                else:
                    # Fhe fastest and simplest way is masking.
                    for axis in axes:
                        if axis == x.batch_dim_axis:
                            continue
                        if not x.has_dynamic_size(axis):
                            continue
                        mask = _seq_mask_raw(x, axis)

                        zeros = tf.zeros((), dtype=x.placeholder.dtype)
                        # Cannot call x.placeholder.dtype.{min,max} in case input is e.g. a bool
                        if x.placeholder.dtype.is_floating or x.placeholder.dtype.is_integer:
                            if f in (tf.reduce_mean, tf.reduce_sum):
                                replacement_value = zeros
                            elif f in (tf.reduce_max, tf.reduce_logsumexp, tf.argmax):
                                replacement_value = zeros + x.placeholder.dtype.min
                            elif f in (tf.reduce_min, tf.argmin):
                                replacement_value = zeros + x.placeholder.dtype.max
                            else:
                                raise ValueError("unexpected reduce function %r" % f)
                        elif x.placeholder.dtype.is_bool:
                            if f in (tf.reduce_any,):
                                replacement_value = zeros
                            elif f in (tf.reduce_all,):
                                replacement_value = tf.ones((), dtype=x.placeholder.dtype)
                            else:
                                raise ValueError("unexpected reduce function %r" % f)
                        else:
                            raise TypeError("reduce: unexpected input type %r from input %s" % (x.placeholder.dtype, x))

                        x_ = tf_util.where_bc(mask, x_, replacement_value, name="x_masked_axis_%i" % axis)
                        if f == tf.reduce_mean:
                            tag = x.dim_tags[axis]
                            assert tag.dyn_size_ext is not None  # checked above
                            size_all = tf.shape(x.placeholder)[axis]
                            size_actual = tag.dyn_size_ext
                            while any(d not in out_data.dim_tags for d in size_actual.dim_tags):
                                # We have some axis (e.g. B) which is not in the output.
                                # We need to remove this.
                                # https://github.com/rwth-i6/returnn/issues/1242
                                i, d = [
                                    (i, d) for i, d in enumerate(size_actual.dim_tags) if d not in out_data.dim_tags
                                ][0]
                                assert not d.need_masking()  # not implemented
                                size_all *= d.get_dim_value()
                                s = tf.reduce_sum(size_actual.placeholder, axis=i)
                                size_actual = size_actual.copy_template_excluding_axis(i)
                                size_actual.placeholder = s
                            seq_len_bc = size_actual.copy_compatible_to(
                                out_data, check_sparse=False, check_dtype=False
                            ).placeholder
                            seq_len_bc = tf.maximum(seq_len_bc, 1)  # avoid nan
                            correction_factor_ = tf.cast(size_all, tf.float32) / tf.cast(seq_len_bc, tf.float32)
                            correction_factor = tf_util.optional_mul(correction_factor, correction_factor_)
            if mode in arg_funcs:
                assert len(axes) == 1, "For argmax/argmin, only one reduction axis is supported"
                y = f(x_, axis=axes[0], output_type=tf.int32)
            else:
                y = f(x_, axis=axes)
                y = tf_util.optional_mul(y, correction_factor)
            out_data.raw_tensor = y
            return out_data

    @staticmethod
    def is_finite(x: Tensor) -> Tensor:
        """is finite"""
        out = x.copy_template("is_finite", dtype="bool")
        with tf_util.same_control_flow_ctx(x):
            out.raw_tensor = tf.math.is_finite(x.raw_tensor)
        return out

    @staticmethod
    def clip_by_value(
        x: Tensor,
        clip_value_min: Union[Tensor, rf.RawTensorTypes],
        clip_value_max: Union[Tensor, rf.RawTensorTypes],
        *,
        allow_broadcast_all_sources: bool = False,
    ) -> Tensor:
        """clip by value"""
        clip_value_min = rf.convert_to_tensor(clip_value_min, _backend=TFBackend, device=x.device)
        clip_value_max = rf.convert_to_tensor(clip_value_max, _backend=TFBackend, device=x.device)
        out = Tensor.get_common_data(
            [x, clip_value_min, clip_value_max],
            allow_broadcast_all_sources=allow_broadcast_all_sources,
            name="clip_by_value",
        )
        out.dtype = x.dtype
        out.sparse_dim = x.sparse_dim
        out.feature_dim = x.feature_dim
        x_bc_raw = x.copy_compatible_to_dims_raw(out.dims)
        min_bc_raw = clip_value_min.copy_compatible_to_dims_raw(out.dims)
        max_bc_raw = clip_value_max.copy_compatible_to_dims_raw(out.dims)
        # tf.clip_by_value broadcasts the bounds against x, but not x itself,
        # so x must already have the full shape (e.g. a scalar clipped by per-seq bounds).
        x_bc_raw = tf.broadcast_to(x_bc_raw, _shape_raw(out.dims))
        out.raw_tensor = tf.clip_by_value(x_bc_raw, min_bc_raw, max_bc_raw)
        return out

    @staticmethod
    def matmul(a: _TT, b: _TT, *, reduce: Union[Dim, Sequence[Dim]], use_mask: bool = True) -> _TT:
        """
        batched matmul of a and b, see base class doc string

        Implemented via tf.einsum, which lowers to (batched) matmul for these patterns
        and, unlike an explicit transpose + reshape, does not need static shapes.
        """
        if isinstance(reduce, Dim):
            reduce = [reduce]
        if use_mask and any(dim.dyn_size_ext is not None for dim in reduce):
            raise NotImplementedError(f"matmul: masking over dynamic reduce dims {reduce} not implemented")
        a_dims, b_dims = a.dims, b.dims
        assert all(dim in a_dims or dim in b_dims for dim in reduce), (
            f"matmul: reduce dims {reduce} must occur in a {a} or b {b}"
        )
        # One letter per AXIS, not per dim:
        # the same dim can occur twice in one input (square matrices, told apart by match_priority).
        # Shared letters mark the reduce axes and the common (batch) axes,
        # every remaining axis gets its own letter.
        letters = iter(string.ascii_lowercase)
        a_letters: List[Optional[str]] = [None] * len(a_dims)
        b_letters: List[Optional[str]] = [None] * len(b_dims)
        a_reduce_axes, b_reduce_axes = [], []
        for dim in reduce:
            letter = next(letters)
            if dim in a_dims:
                a_reduce_axes.append(a.get_axis_from_description(dim))
                a_letters[a_reduce_axes[-1]] = letter
            if dim in b_dims:
                b_reduce_axes.append(b.get_axis_from_description(dim))
                b_letters[b_reduce_axes[-1]] = letter
        common_dims = [dim for i, dim in enumerate(a_dims) if dim in b_dims and i not in a_reduce_axes]
        a_common_axes = [a_dims.index(dim) for dim in common_dims]
        b_common_axes = [b_dims.index(dim) for dim in common_dims]
        assert not set(b_common_axes) & set(b_reduce_axes), (
            f"matmul: reduce dims {reduce} overlap with common dims in b {b}"
        )
        for a_axis, b_axis in zip(a_common_axes, b_common_axes):
            letter = next(letters)
            a_letters[a_axis] = letter
            b_letters[b_axis] = letter
        a_unique_axes = [i for i in range(len(a_dims)) if i not in a_reduce_axes and i not in a_common_axes]
        b_unique_axes = [i for i in range(len(b_dims)) if i not in b_reduce_axes and i not in b_common_axes]
        result_dims = common_dims + [a_dims[i] for i in a_unique_axes] + [b_dims[i] for i in b_unique_axes]
        reduce_size = b_dims[b_reduce_axes[0]].dimension if len(b_reduce_axes) == 1 else None
        if not common_dims and len(a_reduce_axes) == 1 and len(b_unique_axes) == 1 and reduce_size is not None:
            # The rf.Linear case: flatten all other axes of a and do ONE plain 2D matmul.
            # Measured on CPU with [64,200,512] x [512,512], per call:
            # 16.5 ms this way, 23.7 ms via einsum, 25.0 ms via a broadcasting BatchMatMul.
            with tf_util.same_control_flow_ctx([a, b]):
                a_raw = _transpose_raw(a.raw_tensor, a_unique_axes + a_reduce_axes)
                b_raw = _transpose_raw(b.raw_tensor, b_reduce_axes + b_unique_axes)
                raw_result = tf.matmul(tf.reshape(a_raw, [-1, reduce_size]), b_raw)
                out_shape = list(TFBackend.get_shape_tuple_raw(a_raw)[:-1]) + [b_dims[b_unique_axes[0]].get_dim_value()]
                if any(isinstance(d, tf.Tensor) for d in out_shape):
                    out_shape = tf.stack(out_shape)
                raw_result = tf.reshape(raw_result, out_shape)
            return Tensor(
                "dot", dims=result_dims, raw_tensor=raw_result, dtype=TFBackend.get_dtype_name_raw(raw_result)
            )
        for axis in a_unique_axes:
            a_letters[axis] = next(letters)
        for axis in b_unique_axes:
            b_letters[axis] = next(letters)
        out_letters = (
            [a_letters[i] for i in a_common_axes]
            + [a_letters[i] for i in a_unique_axes]
            + [b_letters[i] for i in b_unique_axes]
        )
        subscripts = "%s,%s->%s" % ("".join(a_letters), "".join(b_letters), "".join(out_letters))
        with tf_util.same_control_flow_ctx([a, b]):
            raw_result = tf.einsum(subscripts, a.raw_tensor, b.raw_tensor)
        return Tensor("dot", dims=result_dims, raw_tensor=raw_result, dtype=TFBackend.get_dtype_name_raw(raw_result))

    # noinspection PyShadowingBuiltins
    @staticmethod
    def conv(
        source: Tensor,
        *,
        in_dim: Dim,
        out_dim: Dim,
        in_spatial_dims: Sequence[Dim],
        out_spatial_dims: Optional[Sequence[Dim]] = None,
        filter: Tensor,
        filter_size: Sequence[Dim],
        padding: Union[str, int, Sequence[int]],
        strides: Optional[Union[int, Sequence[int]]] = None,
        dilation_rate: Optional[Union[int, Sequence[int]]] = None,
        groups: Optional[int] = None,
        bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Sequence[Dim]]:
        """
        :param source:
        :param in_dim:
        :param out_dim:
        :param in_spatial_dims:
        :param out_spatial_dims:
        :param filter:
        :param filter_size:
        :param padding: "same", "valid", or explicit
        :param strides:
        :param dilation_rate:
        :param groups:
        :param bias:
        :return: conv output, out spatial dims

        tf.nn.convolution takes "SAME" / "VALID" directly, also with striding,
        so unlike the torch backend there is no padding arithmetic here.
        """
        if not out_spatial_dims:
            out_spatial_dims = rf.make_conv_out_spatial_dims(
                in_spatial_dims=in_spatial_dims,
                filter_size=filter_size,
                strides=strides or 1,
                dilation_rate=dilation_rate or 1,
                padding=padding,
            )
        filter_in_dim = in_dim if not groups or groups == 1 else in_dim // groups
        batch_dims = [d for d in source.dims if d not in (in_dim,) + tuple(in_spatial_dims)]
        with tf_util.same_control_flow_ctx([source, filter]):
            # TF wants the data channels-last and the filter as [*filter_size, in_dim/groups, out_dim]
            filter_raw = filter.copy_compatible_to_dims_raw(tuple(filter_size) + (filter_in_dim, out_dim))
            source = source.copy_transpose(batch_dims + list(in_spatial_dims) + [in_dim])
            src_raw = source.raw_tensor
            src_shape = TFBackend.get_shape_tuple_raw(src_raw)
            if len(batch_dims) != 1:  # merge the batch dims into one
                src_raw = tf.reshape(src_raw, _shape_raw([-1] + list(src_shape[len(batch_dims) :])))
            if isinstance(padding, str):
                tf_padding = padding.upper()
            else:  # explicit padding: pad here, then convolve without padding
                pads = padding if isinstance(padding, (list, tuple)) else [padding] * len(filter_size)
                src_raw = tf.pad(src_raw, [[0, 0]] + [[p, p] for p in pads] + [[0, 0]])
                tf_padding = "VALID"
            out_raw = tf.nn.convolution(
                src_raw, filter_raw, strides=strides, padding=tf_padding, dilations=dilation_rate
            )
            if bias is not None:
                out_raw = out_raw + bias.copy_compatible_to_dims_raw([out_dim])
            if len(batch_dims) != 1:  # and split them again
                out_raw = tf.reshape(
                    out_raw, _shape_raw(list(batch_dims) + list(TFBackend.get_shape_tuple_raw(out_raw)[1:]))
                )
        out = Tensor(
            "conv",
            dims=batch_dims + list(out_spatial_dims) + [out_dim],
            feature_dim=out_dim,
            dtype=TFBackend.get_dtype_name_raw(out_raw),
            raw_tensor=out_raw,
        )
        return out, out_spatial_dims

    @staticmethod
    def pool(
        source: Tensor,
        *,
        mode: str,
        pool_size: Sequence[int],
        padding: Union[str, int, Sequence[int]] = "valid",
        dilation_rate: Union[Sequence[int], int] = 1,
        strides: Sequence[int],
        in_spatial_dims: Sequence[Dim],
        out_spatial_dims: Optional[Sequence[Dim]] = None,
    ) -> Tuple[Tensor, Sequence[Dim]]:
        """
        :param source:
        :param mode: "max" or "avg"
        :param pool_size:
        :param padding:
        :param dilation_rate:
        :param strides:
        :param in_spatial_dims:
        :param out_spatial_dims:
        :return: pooled output, out spatial dims
        """
        if not out_spatial_dims:
            out_spatial_dims = rf.make_conv_out_spatial_dims(
                in_spatial_dims=in_spatial_dims,
                filter_size=pool_size,
                strides=strides,
                dilation_rate=dilation_rate,
                padding=padding,
            )
        tf_mode = {"max": "MAX", "avg": "AVG", "mean": "AVG"}.get(mode.lower())
        if not tf_mode:
            raise NotImplementedError(f"pool: mode {mode!r} not implemented")
        # tf.nn.pool wants these per spatial dim, it does not broadcast a scalar
        nd = len(in_spatial_dims)
        pool_size = list(pool_size) if isinstance(pool_size, (list, tuple)) else [pool_size] * nd
        strides = list(strides) if isinstance(strides, (list, tuple)) else [strides] * nd
        dilation_rate = list(dilation_rate) if isinstance(dilation_rate, (list, tuple)) else [dilation_rate] * nd
        rest_dims = [d for d in source.dims if d not in in_spatial_dims]
        with tf_util.same_control_flow_ctx(source):
            source = source.copy_transpose(rest_dims + list(in_spatial_dims))
            src_shape = TFBackend.get_shape_tuple_raw(source.raw_tensor)
            # tf.nn.pool wants [batch, *spatial, channels]:
            # all non-spatial dims become the batch, and a dummy channel axis is appended
            src_raw = tf.reshape(source.raw_tensor, _shape_raw([-1] + list(src_shape[len(rest_dims) :]) + [1]))
            if isinstance(padding, str):
                tf_padding = padding.upper()
            else:
                pads = padding if isinstance(padding, (list, tuple)) else [padding] * len(pool_size)
                src_raw = tf.pad(src_raw, [[0, 0]] + [[p, p] for p in pads] + [[0, 0]])
                tf_padding = "VALID"
            out_raw = tf.nn.pool(
                src_raw,
                window_shape=pool_size,
                pooling_type=tf_mode,
                strides=strides,
                padding=tf_padding,
                dilations=dilation_rate,
            )
            out_shape = list(rest_dims) + list(TFBackend.get_shape_tuple_raw(out_raw)[1:-1])
            out_raw = tf.reshape(out_raw, _shape_raw(out_shape))
        out = Tensor(
            "pool",
            dims=rest_dims + list(out_spatial_dims),
            dtype=TFBackend.get_dtype_name_raw(out_raw),
            sparse_dim=source.sparse_dim,
            raw_tensor=out_raw,
        )
        if source.feature_dim and source.feature_dim in out.dims:
            out.feature_dim = source.feature_dim
        return out, out_spatial_dims

    @staticmethod
    def stft(
        x: Tensor,
        *,
        in_spatial_dim: Dim,
        frame_step: int,
        frame_length: int,
        fft_length: int,
        window_use_frame_length: bool = True,
        align_window_left: bool = True,
        window_enforce_even: bool = True,
        out_spatial_dim: Dim,
        out_dim: Dim,
    ) -> Tensor:
        """
        :param x:
        :param in_spatial_dim:
        :param frame_step:
        :param frame_length:
        :param fft_length:
        :param window_use_frame_length: window of frame_length (not fft_length), what tf.signal.stft does
        :param align_window_left: zero-pad the window on the right, what tf.signal.stft does
        :param window_enforce_even: round the window length down to even
        :param out_spatial_dim:
        :param out_dim:
        :return: stft of x, complex

        tf.signal.stft is the reference behavior which the other backends emulate,
        so this is just a direct call, with the window built to match the flags.
        """
        assert window_use_frame_length and align_window_left, (
            f"stft: tf.signal.stft windows by frame_length, left-aligned;"
            f" got window_use_frame_length={window_use_frame_length} align_window_left={align_window_left}"
        )
        win_len = frame_length - (frame_length % 2) if window_enforce_even else frame_length

        def _window_fn(length, dtype):
            # length is frame_length, but symbolic, so compare the python ints instead
            del length
            window = tf.signal.hann_window(win_len, dtype=dtype)
            if win_len < frame_length:
                window = tf.pad(window, [[0, frame_length - win_len]])
            return window

        batch_dims = [d for d in x.dims if d != in_spatial_dim]
        x = x.copy_transpose(batch_dims + [in_spatial_dim])
        with tf_util.same_control_flow_ctx(x):
            # keeps the leading dims and appends [frames, fft_unique_bins]
            y_raw = tf.signal.stft(
                x.raw_tensor,
                frame_length=frame_length,
                frame_step=frame_step,
                fft_length=fft_length,
                window_fn=_window_fn,
            )
        return Tensor(
            "stft",
            dims=batch_dims + [out_spatial_dim, out_dim],
            feature_dim=out_dim,
            dtype=TFBackend.get_dtype_name_raw(y_raw),
            raw_tensor=y_raw,
        )

    @staticmethod
    def get_random_state() -> Dict[str, bytes]:
        """
        :return: random state
        """
        # Like the TF-layers backend: the stateful TF random ops keep their state in the session,
        # and there is no TF1 API to read it out. Reproducibility comes from set_random_seed.
        return {}

    @staticmethod
    def set_random_state(state: Dict[str, bytes]) -> None:
        """
        :param state: as returned by :func:`get_random_state`
        """
        assert not state, f"TF: cannot restore a random state, got {state}"

    @staticmethod
    def merge_dims(source: Tensor, *, dims: Sequence[Dim], out_dim: Dim) -> Tensor:
        """
        :param source:
        :param dims: the dims to merge, at least two
        :param out_dim: the merged dim
        :return: source with dims merged into out_dim
        """
        assert len(dims) >= 2
        first_axis = min([source.dims.index(d) for d in dims])
        pre_dims = source.dims[:first_axis]
        post_dims = [d for d in source.dims if d not in dims and d not in pre_dims]
        source = source.copy_transpose(tuple(pre_dims) + tuple(dims) + tuple(post_dims), allow_int=False)
        out = Tensor(
            "merge_dims",
            dims=pre_dims + (out_dim,) + tuple(post_dims),
            dtype=source.dtype,
            sparse_dim=source.sparse_dim,
        )
        if source.feature_dim is not None:
            out.feature_dim = out_dim if source.feature_dim in dims else source.feature_dim
        with tf_util.same_control_flow_ctx(source):
            # the merged block is the single -1, the other axes are taken from the raw shape,
            # so a dynamic dim needs no get_dim_value()
            src_shape = TFBackend.get_shape_tuple_raw(source.raw_tensor)
            out_shape = list(src_shape[: len(pre_dims)]) + [-1] + list(src_shape[len(pre_dims) + len(dims) :])
            out.raw_tensor = tf.reshape(source.raw_tensor, _shape_raw(out_shape))
        return out

    @staticmethod
    def split_dims(
        source: Tensor,
        *,
        axis: Dim,
        dims: Sequence[Dim],
        pad_to_multiples: Optional[bool] = None,
        pad_value: Union[None, int, float] = None,
    ) -> Tensor:
        """
        :param source:
        :param axis: the dim to split
        :param dims: what to split it into
        :param pad_to_multiples: not implemented
        :param pad_value: not implemented
        :return: source with axis replaced by dims
        """
        assert pad_to_multiples in (None, False), "split_dims: pad_to_multiples not implemented"
        axis_ = source.get_axis_from_description(axis)
        out_dims = source.dims[:axis_] + tuple(dims) + source.dims[axis_ + 1 :]
        split_sizes = [d.dimension if d.dimension is not None else -1 for d in dims]
        if split_sizes.count(-1) > 1:  # one -1 at most, so fall back to the explicit sizes
            split_sizes = [d.get_dim_value() for d in dims]
        with tf_util.same_control_flow_ctx(source):
            src_shape = TFBackend.get_shape_tuple_raw(source.raw_tensor)
            out_shape = list(src_shape[:axis_]) + split_sizes + list(src_shape[axis_ + 1 :])
            out_raw = tf.reshape(source.raw_tensor, _shape_raw(out_shape))
        out = Tensor("split_dims", dims=out_dims, dtype=source.dtype, sparse_dim=source.sparse_dim, raw_tensor=out_raw)
        if source.feature_dim and source.feature_dim != axis:
            out.feature_dim = source.feature_dim
        return out

    @staticmethod
    def reshape(source: Tensor, in_dims: Sequence[Dim], out_dims: Sequence[Dim]) -> Tensor:
        """
        :param source: e.g. (..., in_dims, ...)
        :param in_dims: the dims to reshape, not necessarily all dims of the source
        :param out_dims: what to reshape them into
        :return: e.g. (..., out_dims, ...)
        """
        in_dims_axes = [source.get_axis_from_description(d, allow_int=False) for d in in_dims]
        assert sorted(set(in_dims_axes)) == sorted(in_dims_axes), f"reshape {source}: invalid in_dims {in_dims}"
        insert_axis = min(in_dims_axes)
        dims = list(source.dims)
        permute = list(range(len(source.dims)))
        for axis in sorted(set(in_dims_axes), reverse=True):
            dims.pop(axis)
            permute.pop(axis)
        permute = permute[:insert_axis] + in_dims_axes + permute[insert_axis:]
        source = source.copy_transpose(permute)
        dims = dims[:insert_axis] + list(out_dims) + dims[insert_axis:]
        out = Tensor("reshape", dims=dims, dtype=source.dtype, sparse_dim=source.sparse_dim)
        if source.feature_dim and source.feature_dim not in in_dims:
            out.feature_dim = source.feature_dim
        with tf_util.same_control_flow_ctx(source):
            out.raw_tensor = tf.reshape(source.raw_tensor, _shape_raw(dims))
        return out

    @staticmethod
    def split(source: Tensor, *, axis: Dim, out_dims: Sequence[Dim]) -> Tuple[Tensor, ...]:
        """
        :param source:
        :param axis: some static axis
        :param out_dims: sum(out_dims) == axis
        :return: one tensor per out_dim, with axis replaced by it
        """
        axis_int = source.get_axis_from_description(axis)
        with tf_util.same_control_flow_ctx(source):
            out_raw_list = tf.split(source.raw_tensor, [d.get_dim_value() for d in out_dims], axis=axis_int)
        out_tuple = tuple(
            source.copy_template_replace_dim_tag(axis=axis_int, new_dim_tag=dim, name=f"split{i}")
            for i, dim in enumerate(out_dims)
        )
        for out, out_raw in zip(out_tuple, out_raw_list):
            out.raw_tensor = out_raw
        return out_tuple

    @staticmethod
    def expand_dim(source: Tensor, dim: Dim) -> Tensor:
        """
        :param source:
        :param dim:
        :return: source with dim added
        """
        assert dim not in source.dims
        # Some heuristic where to put the new dim (same as the other backends).
        axis = len(source.dims)  # default: at the end
        if dim.is_static() and source.have_feature_axis():
            axis = source.feature_dim_axis
        if dim.is_dynamic():
            for i, d in reversed(list(enumerate(source.dims))):
                if d.is_dynamic():
                    axis = i + 1
                    break
        new_dims = list(source.dims)
        new_dims.insert(axis, dim)
        out = source.copy_template_new_dim_tags(new_dims)
        if source.feature_dim:
            out.feature_dim = source.feature_dim
        with tf_util.same_control_flow_ctx(source):
            out_raw = tf.expand_dims(source.raw_tensor, axis=axis)
            if dim.is_dynamic() or dim.dimension != 1:
                # TF has no stride-0 view like torch expand, so this materializes
                multiples = [1] * len(new_dims)
                multiples[axis] = dim.get_dim_value()
                out_raw = tf.tile(out_raw, _shape_raw(multiples))
        out.raw_tensor = out_raw
        return out

    @staticmethod
    def concat(*sources: Tuple[Tensor, Dim], allow_broadcast: bool = False, out_dim: Dim) -> Tensor:
        """
        :param sources: tensors with the dim to concat over
        :param allow_broadcast: whether the sources may have differing other dims
        :param out_dim:
        :return: concatenated tensor
        """
        axis = sources[0][0].get_axis_from_description(sources[0][1])
        other_dims = list(sources[0][0].dims)
        other_dims.remove(sources[0][1])
        need_broadcast = False
        if allow_broadcast:
            for source, dim in sources[1:]:
                assert dim in source.dims
                if set(source.dims) - {dim} != set(other_dims):
                    need_broadcast = True
                for dim_ in source.dims:
                    if dim_ != dim and dim_ not in other_dims:
                        other_dims.append(dim_)
        sources_raw = []
        for source, dim in sources:
            templ_dims = other_dims[:axis] + [dim] + other_dims[axis:]
            if allow_broadcast and need_broadcast:
                templ = Tensor(source.name, dims=templ_dims, dtype=source.dtype, sparse_dim=source.sparse_dim)
                sources_raw.append(source.copy_compatible_to(templ, unbroadcast=True).raw_tensor)
            else:
                assert set(templ_dims) == set(source.dims), (
                    f"concat {source} {dim} not allowed with allow_broadcast=False"
                )
                sources_raw.append(source.copy_transpose(templ_dims).raw_tensor)
        with tf_util.same_control_flow_ctx(sources_raw):
            out_raw = tf.concat(sources_raw, axis=axis)
        out = Tensor(
            "concat",
            dims=other_dims[:axis] + [out_dim] + other_dims[axis:],
            dtype=TFBackend.get_dtype_name_raw(out_raw),
            sparse_dim=sources[0][0].sparse_dim,
            raw_tensor=out_raw,
        )
        if sources[0][0].feature_dim and sources[0][0].feature_dim != sources[0][1]:
            out.feature_dim = sources[0][0].feature_dim
        return out

    @staticmethod
    def full(
        dims: Sequence[Dim],
        fill_value: Union[RawTensorTypes, Tensor],
        *,
        dtype: str,
        device: Optional[str] = None,
        sparse_dim: Optional[Dim] = None,
        feature_dim: Optional[Dim] = None,
    ) -> Tensor:
        """
        :param dims:
        :param fill_value:
        :param dtype:
        :param device:
        :param sparse_dim:
        :param feature_dim:
        :return: tensor filled with fill_value
        """
        if isinstance(fill_value, Tensor):
            fill_value = fill_value.raw_tensor
        with tf_util.same_control_flow_ctx([d.get_dim_value() for d in dims]):
            raw_tensor = tf.fill(_shape_raw(dims), tf.cast(fill_value, TFBackend.as_dtype_raw(dtype)))
        return Tensor(
            "full", dims=dims, sparse_dim=sparse_dim, feature_dim=feature_dim, dtype=dtype, raw_tensor=raw_tensor
        )

    @staticmethod
    def cumsum(source: Tensor, *, spatial_dim: Dim) -> Tensor:
        """
        :param source:
        :param spatial_dim:
        :return: cumsum over spatial dim
        """
        axis = source.get_axis_from_description(spatial_dim)
        out = source.copy_template("cumsum")
        with tf_util.same_control_flow_ctx(source):
            out.raw_tensor = tf.cumsum(source.raw_tensor, axis=axis)
        return out

    # noinspection PyShadowingBuiltins
    @staticmethod
    def top_k(
        source: Tensor,
        *,
        axis: Union[Dim, Sequence[Dim]],
        k: Union[int, Tensor],
        k_dim: Optional[Dim] = None,
        sorted: bool = True,
    ) -> Tuple[Tensor, Union[Tensor, Sequence[Tensor]], Dim]:
        """
        :param source:
        :param axis: one dim, or multiple dims which are then searched jointly
        :param k:
        :param k_dim:
        :param sorted:
        :return: values, indices (one tensor per axis), k_dim
        """
        if not k_dim:
            k_dim = Dim(k, name="top-k-dim")
        axes = [axis] if isinstance(axis, Dim) else list(axis)
        if any(a.need_masking() for a in axes):
            # tf.math.top_k has no masking, so push the masked entries to the bottom
            mask_value = source.raw_tensor.dtype.min
            for a in axes:
                if a.need_masking():
                    source = rf.where(a.get_mask(dim_order=source.dims), source, mask_value)
        with tf_util.same_control_flow_ctx(source):
            if isinstance(axis, (list, tuple)):
                # tf.math.top_k works on the last axis, so move the searched dims there and flatten them
                source = source.copy_transpose([d for d in source.dims if d not in axis] + list(axis))
                flat_shape = list(TFBackend.get_shape_tuple_raw(source.raw_tensor)[: -len(axis)]) + [-1]
                source_raw_flat = tf.reshape(source.raw_tensor, _shape_raw(flat_shape))
                values_raw, indices_raw = tf.math.top_k(source_raw_flat, k=k_dim.get_dim_value(), sorted=sorted)
                values = source.copy_template_new_dim_tags(
                    new_dim_tags=source.dims[: -len(axis)] + (k_dim,), name="top_k_values"
                )
                if source.feature_dim and source.feature_dim in values.dims:
                    values.feature_dim = source.feature_dim
                values.raw_tensor = values_raw
                indices_out = []
                for i, a in reversed(list(enumerate(axis))):
                    # the flat index decomposes into one index per searched dim
                    indices_out_raw = indices_raw % a.dimension
                    indices_raw = indices_raw // a.dimension
                    indices = values.copy_template(name=f"top_k_indices_{a.name or i}")
                    indices.feature_dim = None
                    indices.dtype = TFBackend.get_dtype_name_raw(indices_out_raw)
                    indices.sparse_dim = a
                    indices.raw_tensor = indices_out_raw
                    indices_out.insert(0, indices)
                return values, indices_out, k_dim
            assert isinstance(axis, Dim)
            source = source.copy_move_axis(source.get_axis_from_description(axis, allow_int=False), -1)
            axis_int = len(source.dims) - 1
            values_raw, indices_raw = tf.math.top_k(source.raw_tensor, k=k_dim.get_dim_value(), sorted=sorted)
        values = source.copy_template_replace_dim_tag(axis=axis_int, new_dim_tag=k_dim, name="top_k_values")
        values.raw_tensor = values_raw
        indices = source.copy_template_replace_dim_tag(axis=axis_int, new_dim_tag=k_dim, name="top_k_indices")
        indices.feature_dim = None
        indices.dtype = TFBackend.get_dtype_name_raw(indices_raw)
        indices.sparse_dim = axis
        indices.raw_tensor = indices_raw
        return values, indices, k_dim

    @staticmethod
    def pad(
        source: Tensor,
        *,
        axes: Sequence[Dim],
        padding: Sequence[Tuple[Union[Dim, int, Tensor], Union[Dim, int, Tensor]]],
        out_dims: Sequence[Dim],
        handle_dynamic_dims: bool,
        mode: str = "constant",
        value: Optional[Union[rf.RawTensorTypes, Tensor]] = None,
    ) -> Tensor:
        """
        :param source:
        :param axes:
        :param padding: per axis, (left, right)
        :param out_dims:
        :param handle_dynamic_dims: mask the padding of the sequences shorter than the padded dim
        :param mode: "constant" or "reflect"
        :param value: for mode "constant"
        :return: padded tensor
        """
        assert len(out_dims) == len(axes) == len(padding)
        raw_pad = []  # per source axis, (left, right), the tf.pad order
        for dim in source.dims:
            if dim not in axes:
                raw_pad.append((0, 0))
                continue
            left, right = padding[axes.index(dim)]
            if isinstance(left, Dim):
                if handle_dynamic_dims:
                    assert not left.need_masking(), f"pad: left {left} needs masking, not supported currently"
                left = left.get_dim_value()
            elif isinstance(left, Tensor):
                if handle_dynamic_dims:
                    assert not left.dims, f"pad: left {left} needs masking, not supported currently"
                left = tf.reduce_max(left.raw_tensor)
            if isinstance(right, Dim):
                right = right.get_dim_value()
            elif isinstance(right, Tensor):
                right = tf.reduce_max(right.raw_tensor)
            raw_pad.append((left, right))
        if (isinstance(value, Tensor) and value.dims == ()) or not isinstance(value, Tensor):
            if isinstance(value, Tensor):
                value_raw = value.raw_tensor
            else:
                value_raw = 0 if value is None else value
            tf_mode = {"constant": "CONSTANT", "reflect": "REFLECT"}.get(mode)
            if not tf_mode:
                raise NotImplementedError(f"pad: mode {mode!r} not implemented")
            out = source.copy_template_new_dim_tags(
                [out_dims[axes.index(dim)] if dim in axes else dim for dim in source.dims], keep_special_axes=True
            )
            with tf_util.same_control_flow_ctx(source):
                paddings = _shape_raw([_shape_raw(lr) for lr in raw_pad])
                if tf_mode == "CONSTANT":
                    out.raw_tensor = tf.pad(source.raw_tensor, paddings, mode=tf_mode, constant_values=value_raw)
                else:
                    out.raw_tensor = tf.pad(source.raw_tensor, paddings, mode=tf_mode)
        else:  # non-scalar value: build it by concat instead
            assert isinstance(value, Tensor) and value.dims
            assert all(dim in source.dims and dim not in axes for dim in value.dims)
            assert len(axes) == 1, "pad: non-scalar value only implemented for a single axis"
            pad_left, pad_right = padding[0]
            pad_left = pad_left if isinstance(pad_left, Dim) else Dim(pad_left, name="pad_left")
            pad_right = pad_right if isinstance(pad_right, Dim) else Dim(pad_right, name="pad_right")
            out = TFBackend.concat(
                *(
                    ([(rf.expand_dim(value, pad_left), pad_left)] if pad_left.dimension else [])
                    + [(source, axes[0])]
                    + ([(rf.expand_dim(value, pad_right), pad_right)] if pad_right.dimension else [])
                ),
                allow_broadcast=True,
                out_dim=out_dims[0],
            )
        if handle_dynamic_dims and any(dim.need_masking() for dim in out_dims):
            if all(right == 0 for _, right in raw_pad) and mode != "circular":
                return out  # nothing was padded on the right, so nothing to mask
            if mode != "constant":
                raise NotImplementedError(f"pad: mode {mode} not implemented with dynamic dims")
            for out_dim, middle, (left, right) in zip(out_dims, axes, padding):
                if not (
                    middle.need_masking()
                    or (isinstance(left, Dim) and left.need_masking())
                    or (isinstance(left, Tensor) and left.dims)
                ):
                    continue
                if not (isinstance(right, (Dim, Tensor)) or (isinstance(right, int) and right > 0)):
                    continue
                if isinstance(left, Dim):
                    left = left.get_size_tensor()
                # the padded frames beyond (left + real len) are junk from the padding of the shorter seqs
                mask = rf.compare_bc(rf.range_over_dim(out_dim), "<", left + middle.get_size_tensor())
                if isinstance(value, Tensor):
                    other = value.copy_compatible_to_dims_raw(out.dims)
                else:
                    # the fill value must have the tensor dtype: tf.where does not promote
                    # (e.g. a float tensor with the int scalar 0)
                    other = tf.cast(0 if value is None else value, out.raw_tensor.dtype)
                out.raw_tensor = tf_util.where_bc(mask.copy_compatible_to_dims_raw(out.dims), out.raw_tensor, other)
        return out

    @staticmethod
    def gather(source: Tensor, *, indices: Union[Tensor, int], axis: Dim, clip_to_valid: bool = False) -> Tensor:
        """
        :param source:
        :param indices: all dims shared with the source (except axis) are batch dims
        :param axis: the axis to gather from
        :param clip_to_valid: clip the indices to the valid range, taking seq lens into account
        :return: gathered values
        """
        axis_int = source.get_axis_from_description(axis, allow_int=False)
        if isinstance(indices, int):
            if not clip_to_valid:  # fast path: a plain slice
                out = Tensor(
                    "gather",
                    dims=list(source.dims[:axis_int]) + list(source.dims[axis_int + 1 :]),
                    dtype=source.dtype,
                    sparse_dim=source.sparse_dim,
                )
                if source.feature_dim and source.feature_dim in out.dims:
                    out.feature_dim = source.feature_dim
                with tf_util.same_control_flow_ctx(source):
                    out.raw_tensor = source.raw_tensor[(slice(None),) * axis_int + (indices,)]
                return out
            indices = Tensor(
                "indices_int",
                dims=(),
                dtype=rf.get_default_array_index_dtype(),
                raw_tensor=tf.constant(indices, dtype=TFBackend.as_dtype_raw(rf.get_default_array_index_dtype())),
            )
        assert isinstance(indices, Tensor), f"gather: unsupported indices {indices!r}"
        if clip_to_valid:
            if axis.dyn_size_ext is not None:
                indices = rf.clip_by_value(
                    indices,
                    0,
                    rf.relu(rf.cast(axis.get_dyn_size_ext_for_device(indices.device), indices.dtype) - 1),
                    allow_broadcast_all_sources=True,
                )
            else:
                indices = indices.copy()
                indices.raw_tensor = tf.clip_by_value(indices.raw_tensor, 0, axis.get_dim_value() - 1)
        # tf.gather handles the batch dims itself, as long as they lead in both source and indices
        index_own_dims = [d for d in indices.dims if d not in source.dims or d == axis]
        common_dims = [d for d in indices.dims if d not in index_own_dims]
        rest_dims = [d for i, d in enumerate(source.dims) if i != axis_int and d not in common_dims]
        out = Tensor(
            "gather",
            dims=common_dims + index_own_dims + rest_dims,
            dtype=source.dtype,
            sparse_dim=source.sparse_dim,
        )
        if source.feature_dim and source.feature_dim in out.dims:
            out.feature_dim = source.feature_dim
        with tf_util.same_control_flow_ctx([source, indices]):
            source_raw = source.copy_compatible_to_dims_raw(common_dims + [axis] + rest_dims)
            indices_raw = indices.copy_compatible_to_dims_raw(common_dims + index_own_dims)
            indices_raw = tf.cast(indices_raw, tf.int32)
            out.raw_tensor = tf.gather(source_raw, indices_raw, axis=len(common_dims), batch_dims=len(common_dims))
        return out

    @staticmethod
    def masked_select(
        tensor: Tensor, *, mask: Tensor, dims: Sequence[Dim], out_dim: Optional[Dim] = None
    ) -> Tuple[Tensor, Dim]:
        """
        :param tensor:
        :param mask:
        :param dims: the dims of the mask, their order defines the format
        :param out_dim:
        :return: tensor with the mask dims replaced by a single new dim, and that dim
        """
        assert mask.dtype == "bool"
        assert set(mask.dims) == set(dims)
        remaining_dims = [d for d in tensor.dims if d not in mask.dims]
        if not out_dim:
            out_dim = Dim(None, name="masked_select")
        with tf_util.same_control_flow_ctx([tensor, mask]):
            in_raw = tensor.copy_compatible_to_dims_raw(tuple(dims) + tuple(remaining_dims))
            mask_raw = mask.copy_compatible_to_dims_raw(tuple(dims))
            # unbroadcast: the input may be broadcast over some of the mask dims
            in_raw = tf.broadcast_to(
                in_raw, _shape_raw(list(dims) + [tf.shape(in_raw)[len(dims) + i] for i in range(len(remaining_dims))])
            )
            out_raw = tf.boolean_mask(in_raw, mask_raw)
            if out_dim.dyn_size_ext is None:
                out_dim.dyn_size_ext = Tensor("masked_select_size", dims=(), dtype="int32")
            if out_dim.dyn_size_ext.raw_tensor is None:
                out_dim.dyn_size_ext.raw_tensor = tf.shape(out_raw)[0]
        out = Tensor(
            "masked_select",
            dims=(out_dim,) + tuple(remaining_dims),
            dtype=tensor.dtype,
            sparse_dim=tensor.sparse_dim,
            feature_dim=tensor.feature_dim if tensor.feature_dim in remaining_dims else None,
            raw_tensor=out_raw,
        )
        return out, out_dim

    @staticmethod
    def slice(
        source: Tensor,
        *,
        axis: Dim,
        start: Optional[Union[int, Tensor]] = None,
        end: Optional[Union[int, Tensor]] = None,
        step: Optional[Union[int, Tensor]] = None,
        size: Optional[Union[int, Tensor, Dim]] = None,
        out_dim: Dim,
    ) -> Tensor:
        """
        :param source:
        :param axis:
        :param start:
        :param end:
        :param step:
        :param size: alternative to end
        :param out_dim:
        :return: source sliced on axis
        """
        assert step is None or (isinstance(step, int) and step == 1), "slice: step != 1 not yet implemented"
        axis_int = source.get_axis_from_description(axis, allow_int=False)
        out = source.copy_template_replace_dim_tag(axis=axis_int, new_dim_tag=out_dim)
        if isinstance(start, Tensor):
            assert start.dims == ()
            start = start.raw_tensor
        elif start is None:
            start = 0
        if isinstance(size, Dim):
            assert end is None
            size = size.get_dim_value()
        elif isinstance(size, Tensor):
            assert end is None
            size = size.raw_tensor if size.dims == () else tf.reduce_max(size.raw_tensor)
        elif isinstance(size, int):
            pass
        elif size is None:
            if isinstance(end, Tensor):
                end = end.raw_tensor if end.dims == () else tf.reduce_max(end.raw_tensor)
            elif isinstance(end, int):
                if end < 0:
                    end += axis.get_dim_value()
            elif end is None:
                end = axis.get_dim_value()
            else:
                raise TypeError(f"slice: unsupported type for end: {type(end)}")
            size = end - start
        else:
            raise TypeError(f"slice: unsupported type for size: {type(size)}")
        begin = [0] * len(source.dims)
        begin[axis_int] = start
        sizes = [-1] * len(source.dims)  # -1 = all remaining
        sizes[axis_int] = size
        with tf_util.same_control_flow_ctx(source):
            out.raw_tensor = tf.slice(source.raw_tensor, _shape_raw(begin), _shape_raw(sizes))
        return out

    @staticmethod
    def flip_no_mask(source: Tensor, *, axis: Dim) -> Tensor:
        """
        :param source:
        :param axis:
        :return: source reversed over axis, ignoring masking
        """
        axis_int = source.get_axis_from_description(axis)
        out = source.copy_template("flip")
        with tf_util.same_control_flow_ctx(source):
            out.raw_tensor = tf.reverse(source.raw_tensor, axis=[axis_int])
        return out

    _deferred_parameter_creation = False
    _deferred_params: Dict[RefIdEq, _DeferredParam] = {}

    @staticmethod
    @contextlib.contextmanager
    def deferred_parameter_creation():
        """
        Inside this context, parameters do not get their tf.Variable yet:
        the variable name should come from the module hierarchy,
        which only exists once the model is fully constructed.
        Wrap the model construction in this, then call :func:`create_parameters`::

            with TFBackend.deferred_parameter_creation():
                model = get_model(epoch=..., step=...)
            TFBackend.create_parameters(model)

        Until then the parameters hold a :class:`DeferredVariable`, which raises on any use.
        """
        prev = TFBackend._deferred_parameter_creation
        try:
            TFBackend._deferred_parameter_creation = True
            yield
        finally:
            TFBackend._deferred_parameter_creation = prev

    @staticmethod
    def create_parameter_raw(
        tensor: rf.Parameter, *, device: Optional[str] = None
    ) -> Union[tf.Variable, DeferredVariable]:
        """
        :return: parameter (by default trainable), or a DeferredVariable, see deferred_parameter_creation
        """
        shape = [d.get_dim_value() for d in tensor.dims]
        assert all(isinstance(d, int) for d in shape), f"parameter {tensor} needs static dims, got shape {shape}"
        dtype = TFBackend.as_dtype_raw(tensor.dtype)
        trainable = not (tensor.dtype.startswith("int") or tensor.dtype.startswith("uint") or tensor.dtype == "bool")
        if TFBackend._deferred_parameter_creation:
            TFBackend._deferred_params[RefIdEq(tensor)] = _DeferredParam(trainable=trainable, device=device)
            return DeferredVariable(dtype=dtype, shape=shape)
        # Eager creation (ad-hoc use outside a model): the name is whatever rf.Parameter was
        # constructed with, so the variables end up as "parameter", "parameter_1", ...
        # The variable gets a zero initializer here;
        # the real initial value arrives later via set_parameter_initial_value.
        device_ctx = tf.device(device) if device else contextlib.nullcontext()
        with tf.control_dependencies(None), device_ctx:
            return tf.Variable(tf.zeros(shape, dtype=dtype), trainable=trainable, dtype=dtype, name=tensor.name)

    @classmethod
    def create_parameters(cls, model: rf.Module) -> None:
        """
        Creates the tf.Variable of every parameter of the model,
        named after its position in the module hierarchy (`.` -> `/`).
        Call this right after the model was constructed
        inside :func:`deferred_parameter_creation`, before the model is used.

        :param model:
        """
        for name, param in model.named_parameters():
            raw = param.raw_tensor
            assert isinstance(raw, DeferredVariable), (
                f"parameter {name} has raw tensor {raw!r};"
                f" expected a DeferredVariable, i.e. the model built inside deferred_parameter_creation()"
            )
            state = cls._deferred_params.pop(RefIdEq(param), _DeferredParam())
            device_ctx = tf.device(state.device) if state.device else contextlib.nullcontext()
            with tf.control_dependencies(None), device_ctx:
                if state.initial is not None:
                    # broadcast: a scalar init (e.g. bias 0.0) would otherwise give a scalar variable
                    initial = tf.broadcast_to(state.initial, raw.shape.as_list())
                else:
                    initial = tf.zeros(raw.shape.as_list(), dtype=raw.dtype)
                var = tf.Variable(initial, trainable=state.trainable, dtype=raw.dtype, name=name.replace(".", "/"))
            param.raw_tensor = var

    @staticmethod
    def set_parameter_initial_value(param: rf.Parameter, value: Union[None, Tensor, rf.RawTensorTypes]) -> None:
        """
        :param param: parameter
        :param value: initial value
        """
        raw = param.raw_tensor
        if value is None:
            return  # keep the zero init
        with tf.control_dependencies(None):
            if isinstance(value, Tensor):
                value_raw = value.copy_compatible_to_dims_raw(param.dims)
            else:
                value_raw = tf.convert_to_tensor(value, dtype=raw.dtype.base_dtype)
            if isinstance(raw, DeferredVariable):
                # The value tensor is built HERE, not at variable creation,
                # so a random init keeps its place in the op order (and thus the RandomJournal order).
                TFBackend._deferred_params[RefIdEq(param)].initial = value_raw
                return
            var = TFBackend._get_param_var(param)
            value_raw = tf.broadcast_to(value_raw, var.shape.as_list())
            # Without deferral, rf.Parameter provides the initial value only after the variable exists,
            # so replace the variable's initializer (the zeros assign built at creation).
            # These two attributes are where TF stores it;
            # tf.compat.v1.global_variables_initializer() then runs OUR value.
            var._initializer_op = tf_compat.v1.assign(var, value_raw).op
            var._initial_value = value_raw

    @staticmethod
    def set_parameter_trainable(param: rf.Parameter, trainable: bool) -> None:
        """
        :param param: parameter
        :param trainable: whether the parameter should be trainable
        """
        if isinstance(param.raw_tensor, DeferredVariable):
            TFBackend._deferred_params[RefIdEq(param)].trainable = trainable
            return
        var = TFBackend._get_param_var(param)
        # TF fixes trainability at variable creation, via membership in the trainable-variables collection,
        # so update that collection instead.
        # Membership is tested by identity: `var == other` would build an elementwise-compare op.
        coll = tf_compat.v1.get_default_graph().get_collection_ref(tf_compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        contained = any(v is var for v in coll)
        if trainable and not contained:
            coll.append(var)
        elif not trainable and contained:
            coll[:] = [v for v in coll if v is not var]

    @staticmethod
    def set_random_seed(seed: int) -> None:
        """
        :param seed: sets the graph-level seed, which the stateful random ops derive their seeds from
        """
        tf_compat.v1.set_random_seed(seed)

    @staticmethod
    @contextlib.contextmanager
    def random_journal_replay(journal: _random_journal.RandomJournal):
        """
        Replay recorded random numbers instead of drawing new ones.
        Used to compare backends op by op, see tests/rf_utils.py.
        """
        prev_journal = TFBackend._random_journal
        try:
            TFBackend._random_journal = journal
            yield
        finally:
            TFBackend._random_journal = prev_journal

    _random_journal = None  # type: Optional[_random_journal.RandomJournal]

    @staticmethod
    def random(
        *,
        dims: Sequence[Dim],
        dtype: str,
        device: Optional[str] = None,
        sparse_dim: Optional[Dim] = None,
        feature_dim: Optional[Dim] = None,
        distribution: str,
        mean: Optional[Union[int, float, Tensor]] = None,
        stddev: Optional[Union[int, float, Tensor]] = None,
        bound: Optional[Union[int, float, Tensor]] = None,
        minval: Optional[Union[int, float, Tensor]] = None,
        maxval: Optional[Union[int, float, Tensor]] = None,
        seed: Optional[Union[int, Sequence[int], numpy.ndarray]] = None,
        algorithm: Optional[str] = None,
        explicit_state: Optional[Tensor] = None,
        auto_update_state: Optional[bool] = None,
        static: Optional[bool] = None,
        out: Optional[Tensor] = None,
    ) -> Tensor:
        """
        random. See `rf.random` for details.
        """
        assert out is None, "TF random: out not supported"
        assert explicit_state is None and auto_update_state is None, "TF random: explicit state not implemented"
        assert algorithm is None, f"TF random: algorithm {algorithm!r} not implemented"
        res = Tensor(
            name=f"random_{distribution}", dims=dims, dtype=dtype, sparse_dim=sparse_dim, feature_dim=feature_dim
        )
        if TFBackend._random_journal is not None:
            # The graph is built in the same order as the recorded run,
            # so reading the journal at graph construction time keeps the sequences aligned.
            entry = TFBackend._random_journal.get_next(new_out_template=res)
            assert isinstance(entry.out.raw_tensor, numpy.ndarray)
            res.raw_tensor = tf.constant(entry.out.raw_tensor, dtype=TFBackend.as_dtype_raw(dtype))
            return res
        dtype_raw = TFBackend.as_dtype_raw(dtype)
        shape = [d.get_dim_value() for d in dims]
        seed_raw = None
        if static:
            assert seed is not None, "TF random: static needs a seed"
            seed_flat = numpy.asarray(seed).flatten()
            # the stateless ops take a seed pair
            seed_raw = [int(seed_flat[0]), int(seed_flat[1]) if seed_flat.size > 1 else 0]
        else:
            assert seed is None, "TF random: seed only together with static"
        if distribution == "uniform":
            assert mean is None and stddev is None, "TF random uniform: mean/stddev not supported"
            if bound is not None:
                assert minval is None and maxval is None, "TF random uniform: bound together with minval/maxval"
                minval, maxval = -bound, bound
            minval = _random_scalar_arg(minval, 0)
            maxval = _random_scalar_arg(maxval, 1 if dtype_raw.is_floating else None)
            assert maxval is not None, "TF random uniform: maxval required for integer dtype"
            if seed_raw is not None:
                raw = tf.random.stateless_uniform(shape, seed_raw, minval=minval, maxval=maxval, dtype=dtype_raw)
            else:
                raw = tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=dtype_raw)
        elif distribution in ("normal", "truncated_normal"):
            assert minval is None and maxval is None and bound is None, f"TF random {distribution}: got minval/maxval"
            mean = _random_scalar_arg(mean, 0)
            stddev = _random_scalar_arg(stddev, 1)
            if seed_raw is not None:
                func = tf.random.stateless_normal if distribution == "normal" else tf.random.stateless_truncated_normal
                raw = func(shape, seed_raw, mean=mean, stddev=stddev, dtype=dtype_raw)
            else:
                func = tf.random.normal if distribution == "normal" else tf.random.truncated_normal
                raw = func(shape, mean=mean, stddev=stddev, dtype=dtype_raw)
        else:
            raise NotImplementedError(f"TF random: distribution {distribution!r} not implemented")
        res.raw_tensor = raw
        return res

    @staticmethod
    def _get_param_var(param: rf.Parameter) -> tf.Variable:
        var = param.raw_tensor
        assert isinstance(var, tf.Variable), f"parameter {param} has raw tensor {var!r}, expected tf.Variable"
        return var


# So that an rf.Parameter holding a DeferredVariable still dispatches to this backend
# (the native dispatch consults this same table).
register_backend_by_tensor_type(DeferredVariable, TFBackend)


# RF activation names which TF spells differently
_ACTIVATION_FUNC_NAME_MAP = {"neg": "negative"}


def _seq_mask_raw(tensor: Tensor, axis: int) -> tf.Tensor:
    """
    :param tensor:
    :param axis: an axis with a dynamic size
    :return: seq mask, broadcastable to the tensor dims

    Deliberately not Tensor.get_sequence_mask_broadcast:
    its fast path (taken because this backend has sequence_mask_raw) resolves the batch dim via
    Tensor.get_batch_dim -> LayerBase.get_recent_layer, i.e. the net-dict layer registry,
    which does not exist when the graph comes from RF code directly.
    """
    return tensor.get_sequence_mask_tensor(axis).copy_compatible_to_dims_raw(tensor.dims)


def _shape_raw(dims_or_sizes: Sequence[Union[Dim, int, tf.Tensor]]) -> Union[List[int], tf.Tensor]:
    """
    :param dims_or_sizes: dims, or their sizes directly
    :return: shape for tf.reshape / tf.fill / tf.tile:
        a plain list while all sizes are static, else a stacked tensor
    """
    sizes = [d.get_dim_value() if isinstance(d, Dim) else d for d in dims_or_sizes]
    if any(isinstance(s, tf.Tensor) for s in sizes):
        return tf.stack(sizes)
    return sizes


def _transpose_raw(raw_tensor: tf.Tensor, perm: Sequence[int]) -> tf.Tensor:
    """
    :param raw_tensor:
    :param perm:
    :return: transposed raw tensor, or the input itself if perm is the identity
    """
    if list(perm) == list(range(len(perm))):
        return raw_tensor
    return tf.transpose(raw_tensor, perm)


def _random_scalar_arg(value: Union[None, int, float, Tensor], default: Union[None, int, float]) -> Any:
    """
    :param value: distribution argument of :func:`TFBackend.random`
    :param default: used when value is None
    :return: raw scalar
    """
    if value is None:
        return default
    if isinstance(value, Tensor):
        assert value.dims == (), f"TF random: only scalar distribution args supported, got {value}"
        return value.raw_tensor
    return value
