"""
Gradient checkpointing.
"""

from __future__ import annotations

from typing import List, Tuple
import typing
import contextlib
import weakref
import tensorflow as tf
from returnn.tf import compat as tf_compat
from returnn.tf.util import basic as tf_util


_grad_checkpoints = weakref.WeakKeyDictionary()  # type: weakref.WeakKeyDictionary[tf.Graph, List[Tuple[int, int]]]


@contextlib.contextmanager
def gradient_checkpoint_scope():
    """
    :return: context manager, where all tensors created inside the scope will be recomputed at backprop time,
        based on existing tensors which have been created earlier outside the scope.

    If prepare_gradient_checkpointing() is not called later, this does not have any effect.
    If no gradients are being calculated, this also does not have any effect.
    """
    graph = tf_compat.v1.get_default_graph()
    # noinspection PyProtectedMember
    first_op_id = graph._last_id + 1  # inclusive
    yield
    # noinspection PyProtectedMember
    last_op_id = graph._last_id + 1  # exclusive
    assert last_op_id > first_op_id
    for op_id in range(first_op_id, last_op_id):
        # noinspection PyProtectedMember
        op = graph._nodes_by_id[op_id]
        if getattr(op, "_RETURNN_gradient_checkpoint_exclude", False):
            continue
        op._RETURNN_gradient_checkpoint_first_op_id = first_op_id
    _grad_checkpoints.setdefault(graph, []).append((first_op_id, last_op_id))


@contextlib.contextmanager
def gradient_checkpoint_exclude_scope():
    """
    :return: context manager, where all tensors created inside the scope will be excluded
        for recomputation at backprop time.
    """
    graph = tf_compat.v1.get_default_graph()
    # noinspection PyProtectedMember
    first_op_id = graph._last_id + 1  # inclusive
    yield
    # noinspection PyProtectedMember
    last_op_id = graph._last_id + 1  # exclusive
    assert last_op_id >= first_op_id
    for op_id in range(first_op_id, last_op_id):
        # noinspection PyProtectedMember
        op = graph._nodes_by_id[op_id]
        if getattr(op, "_RETURNN_gradient_checkpoint_first_op_id", None) is not None:
            # There was some inner gradient_checkpoint_scope() again...
            continue
        op._RETURNN_gradient_checkpoint_exclude = True


def prepare_gradient_checkpointing():
    """
    Call this after the computation graph for calculating the model + loss has been created,
    before the gradients are calculated (before tf.gradients is called).

    This will create a copy of all the ops from within the gradient_checkpoint_scope() scope.

    This patches the op._gradient_function of all consuming ops
    to use the copied ops instead.
    So effectively, for backpropagation, it will recalculate all such tensors.
    """
    from tensorflow.python.framework import ops

    copied_ops = {}  # type: typing.Dict[tf.Operation, tf.Operation]  # old -> new

    # noinspection PyShadowingNames
    def _copy_op(op: tf.Operation) -> tf.Operation:
        if op in copied_ops:
            return copied_ops[op]

        new_inputs = []
        for x in op.inputs:
            x = _map_tensor(x)
            new_inputs.append(x)

        with tf_util.same_control_flow_ctx(op.outputs[0]), tf.name_scope(""):
            new_op = tf_util.copy_op(op, inputs=new_inputs, name=op.name)
        _set_wrapped_grad_func(new_op)
        copied_ops[op] = new_op
        return new_op

    def _map_tensor(x: tf.Tensor) -> tf.Tensor:
        assert isinstance(x, tf.Tensor)
        if getattr(x.op, "_RETURNN_gradient_checkpoint_first_op_id", None) is not None:
            x_op_copy = _copy_op(x.op)
            x = x_op_copy.outputs[x.value_index]
        return x

    # noinspection PyShadowingNames
    def _set_wrapped_grad_func(op: tf.Operation):
        # Make sure to not wrap it multiple times.
        if getattr(op, "_RETURNN_gradient_checkpoint_wrapped_grad_func", None) is not None:
            return

        try:
            orig_grad_func = ops.get_gradient_function(op)
        except LookupError:
            return  # no gradient function
        if orig_grad_func is None:
            return

        class _WrappedOp:
            def __init__(self):
                self.op = op
                self._inputs = tuple(_map_tensor(x) for x in op.inputs)
                self._outputs = tuple(_map_tensor(x) for x in op.outputs)

            @property
            def inputs(self) -> Tuple[tf.Tensor, ...]:
                """inputs"""
                return self._inputs

            @property
            def outputs(self) -> Tuple[tf.Tensor, ...]:
                """outputs"""
                return self._outputs

            def get_attr(self, name: str):
                """get_attr"""
                return self.op.get_attr(name)

        _wrapped_op = _WrappedOp()

        # noinspection PyShadowingNames
        def _wrapped_grad_func(op, *out_grads):
            assert op is _wrapped_op.op  # wrapped grad func not intended to be called for other ops
            return orig_grad_func(_wrapped_op, *out_grads)

        op._gradient_function = _wrapped_grad_func
        op._RETURNN_gradient_checkpoint_wrapped_grad_func = _wrapped_grad_func

    for graph, ls in _grad_checkpoints.items():
        for first_op_id, last_op_id in ls:
            for op_id in range(first_op_id, last_op_id):
                # noinspection PyProtectedMember
                op = graph._nodes_by_id[op_id]
                assert isinstance(op, tf.Operation)
                if getattr(op, "_RETURNN_gradient_checkpoint_exclude", False):
                    continue
                _copy_op(op)
                for op_in in op.inputs:
                    assert isinstance(op_in, tf.Tensor)
                    _set_wrapped_grad_func(op_in.op)
                for op_out in op.outputs:
                    assert isinstance(op_out, tf.Tensor)
                    for op_ in op_out.consumers():
                        assert isinstance(op_, tf.Operation)
                        _set_wrapped_grad_func(op_)
