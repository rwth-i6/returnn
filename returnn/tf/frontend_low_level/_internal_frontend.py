"""
Internal frontend for TF low-level frontend.
"""

from __future__ import annotations
from typing import Optional, Union, Any, Sequence, Tuple
import tensorflow as tf
import contextlib

from returnn._internal_frontend_api import InternalFrontend
from returnn.tensor import Tensor
from returnn.tf.util import basic as tf_util
from returnn.tf import compat as tf_compat

_TT = Tensor[tf.Tensor]


class TFInternalFrontend(InternalFrontend[tf.Tensor]):
    """
    Internal frontend for TF low-level frontend.
    """

    @staticmethod
    def get_dtype_name_raw(raw_tensor: tf.Tensor) -> str:
        """
        :return: dtype of raw tensor, as string
        """
        return raw_tensor.dtype.base_dtype.name

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
        :param kind: "equal"|"==", "less"|"<", "less_equal"|"<=", "greater"|">", "greater_equal"|">=", "not_equal"|"!="
        :param b:
        :return: a `kind` b
        """
        assert a.shape.ndims == b.shape.ndims
        kind = {
            "==": "equal",
            "<=": "less_equal",
            "<": "less",
            ">=": "greater_equal",
            ">": "greater",
            "!=": "not_equal",
            "<>": "not_equal",
        }.get(kind, kind)
        op = getattr(tf, kind)  # e.g. tf.equal
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
    def create_placeholder(tensor: _TT) -> tf.Tensor:
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
        with tf.name_scope("runtime_sanity_check"):
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
                    checks += [dyn_size_ext.get_runtime_sanity_check_op()]
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
