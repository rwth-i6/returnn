"""
Gradient checkpointing.
"""

from __future__ import annotations

from typing import List, Tuple, Dict
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
    first_op_id = len(graph.get_operations())  # inclusive
    yield
    ops = graph.get_operations()
    last_op_id = len(ops)  # exclusive
    assert last_op_id > first_op_id
    for op_id in range(first_op_id, last_op_id):
        op = ops[op_id]
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
    first_op_id = len(graph.get_operations())  # inclusive
    yield
    ops = graph.get_operations()
    last_op_id = len(ops)  # exclusive
    assert last_op_id >= first_op_id
    for op_id in range(first_op_id, last_op_id):
        op = ops[op_id]
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

    copied_ops: Dict[Tuple[int, str], tf.Operation] = {}  # graph id, old op name -> new op
    copy_op_stack_depth = 0

    class _DeepCopyError(Exception):
        # noinspection PyShadowingNames
        def __init__(self, op: tf.Operation):
            super().__init__(f"deep copy err: {op}")
            self.op = op

    # noinspection PyShadowingNames
    def _copy_op(op: tf.Operation) -> tf.Operation:
        if (id(op.graph), op.name) in copied_ops:
            return copied_ops[(id(op.graph), op.name)]

        nonlocal copy_op_stack_depth
        if copy_op_stack_depth >= 1:
            # Avoid deep recursions here, as this can get very deep on big graphs.
            # So do a flat construction in the loop below.
            raise _DeepCopyError(op)

        try:
            copy_op_stack_depth += 1

            new_op = None
            copy_op_queue = [op]
            while copy_op_queue:
                op = copy_op_queue[-1]

                # Maybe got copied in the meantime.
                if (id(op.graph), op.name) in copied_ops:
                    copy_op_queue.pop(-1)
                    new_op = copied_ops[(id(op.graph), op.name)]
                    continue

                try:
                    new_inputs = []
                    for x in op.inputs:
                        x = _map_tensor(x)
                        new_inputs.append(x)

                except _DeepCopyError as exc:
                    copy_op_queue.append(exc.op)
                    continue

                with tf_util.same_control_flow_ctx(op.outputs[0]), tf.name_scope(""):
                    new_op = tf_util.copy_op(op, inputs=new_inputs, name=op.name)
                copied_ops[(id(op.graph), op.name)] = new_op
                _set_wrapped_grad_func(new_op)

                assert op is copy_op_queue[-1]
                copy_op_queue.pop(-1)

        finally:
            copy_op_stack_depth -= 1
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

            def _get_control_flow_context(self):
                """get control flow ctx"""
                # noinspection PyProtectedMember
                return self.op._get_control_flow_context()

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
