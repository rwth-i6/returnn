"""
Backend for exposing TensorFlow-specific functionality.
"""

from __future__ import annotations
from typing import Optional, Any, Union, Sequence, Tuple
import contextlib
import tensorflow as tf

import returnn.tf.compat as tf_compat
from returnn.util.basic import NotSpecified, is_onnx_export_global
from returnn.tensor import Tensor, Dim
from returnn.tf.util import basic as tf_util

# noinspection PyProtectedMember
from returnn.frontend._backend import Backend
from returnn.frontend import RawTensorTypes
import returnn.frontend as rf

_TT = Tensor[tf.Tensor]


# Ignore this warning until we really expect that we implemented everything.
# noinspection PyAbstractClass
class TFBackend(Backend[tf.Tensor]):
    """
    TensorFlow low-level backend, operating on tf.Tensor
    """

    RawTensorType = tf.Tensor
    is_tensorflow = True

    @staticmethod
    def executing_eagerly() -> bool:
        """
        :return: whether we are in eager execution mode
        """
        return tf.executing_eagerly()

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
    def get_shape_tuple_raw(raw_tensor: tf.Tensor) -> Tuple[Union[int, tf.Tensor]]:
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
    def get_known_shape_raw(raw_tensor: tf.Tensor) -> Tuple[Optional[int]]:
        """
        :return: shape of raw tensor, int for static known, None otherwise. assumes that ndim is known.
        """
        return tuple(raw_tensor.shape.as_list())

    @staticmethod
    def set_known_shape_raw(raw_tensor: tf.Tensor, shape: Tuple[Optional[int]]) -> None:
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
                if dyn_size_ext and dyn_size_ext.placeholder is not None:
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
        device: Optional[str] = None,
        name: Optional[str] = None,
    ) -> _TT:
        """
        :param value:
        :param dims:
        :param dtype:
        :param sparse_dim:
        :param device:
        :param name:
        :return: tensor
        """
        if isinstance(value, Tensor):
            return value
        with tf.control_dependencies(None):
            value = tf.convert_to_tensor(value, dtype=dtype)
        assert isinstance(value, tf.Tensor)
        return Tensor(name or "const", raw_tensor=value, dims=dims, dtype=dtype, sparse_dim=sparse_dim)

    @staticmethod
    def range_over_dim(dim: Dim, *, dtype: Optional[str] = None, device: Optional[str] = None) -> _TT:
        """
        :param dim:
        :param dtype:
        :param device:
        :return: range over dim
        """
        if not dtype and dim.dyn_size_ext:
            dtype = dim.dyn_size_ext.dtype
        if not dtype:
            dtype = rf.get_default_array_index_dtype()
        out = Tensor(
            name=dim.description or "range_over_dim",
            dims=[dim],
            sparse_dim=dim,
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
