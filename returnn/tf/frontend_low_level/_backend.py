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
                mask = tensor.get_sequence_mask_broadcast(axis=axis)
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
                mask = tensor.get_sequence_mask_broadcast(axis=axis)
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
                        mask = x.get_sequence_mask_broadcast(axis=axis)

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
        for axis in a_unique_axes:
            a_letters[axis] = next(letters)
        for axis in b_unique_axes:
            b_letters[axis] = next(letters)
        out_letters = (
            [a_letters[i] for i in a_common_axes]
            + [a_letters[i] for i in a_unique_axes]
            + [b_letters[i] for i in b_unique_axes]
        )
        result_dims = common_dims + [a_dims[i] for i in a_unique_axes] + [b_dims[i] for i in b_unique_axes]
        subscripts = "%s,%s->%s" % ("".join(a_letters), "".join(b_letters), "".join(out_letters))
        with tf_util.same_control_flow_ctx([a, b]):
            raw_result = tf.einsum(subscripts, a.raw_tensor, b.raw_tensor)
        return Tensor("dot", dims=result_dims, raw_tensor=raw_result, dtype=TFBackend.get_dtype_name_raw(raw_result))

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
