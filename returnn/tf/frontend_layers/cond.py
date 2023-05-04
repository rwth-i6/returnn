"""
Conditional logic

https://github.com/rwth-i6/returnn_common/issues/24
"""

from __future__ import annotations
from typing import Any, List, TypeVar, Generic, Callable
from tensorflow.python.util import nest
from returnn.tensor import Tensor, ControlFlowContext
import returnn.frontend as rf
import returnn.tf.frontend_layers as rfl
from . import _utils


T = TypeVar("T")


__all__ = ["Cond", "CondModule"]


class Cond(Generic[T]):
    """
    Conditional branching. Basically behaves like ``if ... else ...``.
    Only one branch will be executed, and the condition needs to be a bool scalar.
    This wraps to :class:`CondLayer` in RETURNN and to ``tf.cond`` in TensorFlow.

    Example::

        with Cond(cond) as cond_obj:
          cond_obj.true = mod_true_case(x)
          cond_obj.false = mod_false_case(x)
          y = cond_obj.result

    Corresponds to::

        if cond:
          y = mod_true_case(x)
        else:
          y = mod_false_case(x)

    The context scope has two states corresponding to the True and False computation branch.
    The initial state is the True branch.
    Assigning ``cond_obj.true`` has the side effect of switching the computation to the False branch.
    """

    def __init__(self, condition: Tensor, *, name: str = "cond"):
        self.condition = condition
        self._entered = False
        self._entered_state = True
        self._true_value = None
        self._true_value_set = False
        self._false_value = None
        self._false_value_set = False
        self._result_value = None
        self.layer_module = CondModule(cond=self)
        self.name_ctx = rfl.Layer(
            module=self.layer_module, suggested_name=name, parent=rfl.Layer.current_ctx(), can_access_children=False
        )
        self.name_ctx.custom_layer_name_scope = ""
        self.true_branch_control_flow_ctx = ControlFlowContext(
            kind=ControlFlowContext.Types.Cond, outer_ctx=self.name_ctx.control_flow_ctx()
        )
        self.true_branch_name_ctx = rfl.Layer(
            module=self.layer_module,
            suggested_name="true",
            parent=self.name_ctx,
            virtual=True,
            can_access_children=False,
            new_control_flow_ctx=self.true_branch_control_flow_ctx,
        )
        self.true_branch_name_ctx.is_subnet = True
        self.true_branch_name_ctx.extend_reserved_names({"output"})
        self.false_branch_control_flow_ctx = ControlFlowContext(
            kind=ControlFlowContext.Types.Cond, outer_ctx=self.name_ctx.control_flow_ctx()
        )
        self.false_branch_name_ctx = rfl.Layer(
            module=self.layer_module,
            suggested_name="false",
            parent=self.name_ctx,
            virtual=True,
            can_access_children=False,
            new_control_flow_ctx=self.false_branch_control_flow_ctx,
        )
        self.false_branch_name_ctx.is_subnet = True
        self.false_branch_name_ctx.extend_reserved_names({"output"})
        self._extra_ops_true_branch: List[Tensor] = []
        self._extra_ops_false_branch: List[Tensor] = []
        self._false_branch_prehooks: List[Callable[[], Any]] = []
        self._false_branch_posthooks: List[Callable[[], Any]] = []

    def __repr__(self):
        return f"Cond{self.name_ctx}"

    def __enter__(self):
        assert not self._entered, f"{self} cannot enter twice"
        self._entered = True
        self._entered_state = True
        self.true_branch_name_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # First exit any scopes and do cleanup without throwing any exceptions.
        if self._entered:
            if self._true_value is None:
                self.true_branch_name_ctx.__exit__(exc_type, exc_val, exc_tb)
            elif self._false_value is None:
                self.false_branch_name_ctx.__exit__(exc_type, exc_val, exc_tb)
        if not exc_type:  # only do error checking if there was no other exception
            assert self._entered
            assert self._true_value is not None, f"{self} you need to call else_()"
            assert self._false_value is not None, f"{self} you need to call end()"
        self._entered = False

    @property
    def true(self) -> T:
        """
        The getter usually would not be used.
        """
        return self._true_value

    @true.setter
    def true(self, true_value: T):
        """
        Defines the True branch value.
        Enter the False branch.
        Assign self.false afterwards.
        """
        assert self._entered, f"{self} you need to be in the context scope"
        assert self._entered_state is True, f"{self} you cannot enter the else branch twice"
        assert not self._true_value_set
        if isinstance(true_value, Tensor):
            true_value = _utils.copy(true_value, name=self.true_branch_name_ctx.get_child("output"))
        else:
            values_flat = nest.flatten(true_value)  # type: List[Tensor]
            assert values_flat
            for i, v in enumerate(values_flat):
                if v is None:
                    # Dummy value.
                    # Allow this in case the whole output is just None.
                    # Maybe there were other side effects.
                    v = rf.zeros((), dtype="int32")
                assert isinstance(v, Tensor), f"unexpected {true_value!r}, only expects tensors, got {type(v)}"
                if i == 0:
                    values_flat[i] = _utils.copy(v, name=self.true_branch_name_ctx.get_child("output"))
                else:
                    values_flat[i] = _utils.mark_as_output_in_scope(v, scope=self.true_branch_name_ctx)
            if len(values_flat) == 0:
                # maybe there are just ops with side effects, maybe also only in the other branch
                # dummy output
                _utils.copy(rf.zeros((), dtype="int32"), name=self.true_branch_name_ctx.get_child("output"))
            true_value = nest.pack_sequence_as(true_value, values_flat)
        for op in self._extra_ops_true_branch:
            _utils.mark_as_output_in_scope(op, scope=self.true_branch_name_ctx)
        self.true_branch_name_ctx.__exit__(None, None, None)
        self.false_branch_name_ctx.__enter__()
        self.false_branch_name_ctx.extend_reserved_names(
            {v.raw_tensor.name for v in self.true_branch_name_ctx.marked_outputs}
        )
        self._true_value = true_value
        self._true_value_set = True
        self._entered_state = False
        for hook in self._false_branch_prehooks:
            hook()

    @property
    def false(self) -> T:
        """
        The getter usually would not be used.
        """
        return self._false_value

    @false.setter
    def false(self, false_value: T):
        """
        Define the False branch value.
        After this, self.result is available.
        """
        assert self._entered, f"{self} you need to be in the context scope"
        assert (
            self._entered_state is False
        ), f"{self} you need to be in the False branch, have assigned :func:`true` before"
        assert not self._false_value_set
        nest.assert_same_structure(self._true_value, false_value)
        # This needs to match the true() setter logic.
        if isinstance(false_value, Tensor):
            false_value = _utils.copy(false_value, name=self.false_branch_name_ctx.get_child("output"))
        else:
            true_values_flat = nest.flatten(self._true_value)  # type: List[Tensor]
            false_values_flat = nest.flatten(false_value)  # type: List[Tensor]
            assert false_values_flat and len(false_values_flat) == len(true_values_flat)
            for i, (true_v, false_v) in enumerate(zip(true_values_flat, false_values_flat)):
                assert isinstance(true_v, Tensor)
                if false_v is None:  # see above
                    false_v = rf.zeros((), dtype="int32")  # dummy value
                else:
                    assert isinstance(
                        false_v, Tensor
                    ), f"unexpected {false_value!r}, only expects tensors, got {type(false_v)}"
                assert true_v.raw_tensor.parent is self.true_branch_name_ctx
                name = true_v.raw_tensor.name
                assert name not in self.false_branch_name_ctx.children
                false_values_flat[i] = _utils.copy(false_v, name=self.false_branch_name_ctx.get_child(name))
                if name != "output":
                    false_values_flat[i].raw_tensor.layer_dict["is_output_layer"] = True
            if len(false_values_flat) == 0:
                # Same as in true branch, see above.
                _utils.copy(rf.zeros((), dtype="int32"), name=self.true_branch_name_ctx.get_child("output"))
            false_value = nest.pack_sequence_as(false_value, false_values_flat)
        for true_out in self.true_branch_name_ctx.marked_outputs:
            name = true_out.raw_tensor.name
            if name in self.false_branch_name_ctx.children:
                continue
            _utils.zeros_like_as_output_in_scope(true_out, name=self.false_branch_name_ctx.get_child(name))
        for op in self._extra_ops_false_branch:
            op = _utils.mark_as_output_in_scope(op, scope=self.false_branch_name_ctx)
            name = op.raw_tensor.name
            assert name not in self.true_branch_name_ctx.children
            _utils.zeros_like_as_output_in_scope(op, name=self.true_branch_name_ctx.get_child(name))
        self.false_branch_name_ctx.__exit__(None, None, None)
        self._false_value = false_value
        self._false_value_set = True
        for cb in self._false_branch_posthooks:
            cb()
        self._result_value = self.layer_module()

    @property
    def result(self) -> T:
        """
        :return: the result, after you assigned :func:`true` and :func:`false`.
        """
        assert self._true_value_set, f"{self} you need to have defined the true value"
        assert self._false_value_set, f"{self} you need to have defined the false value"
        return self._result_value

    def add_op_to_current_branch(self, op: Tensor):
        """
        :param op: like an assign_op. the value of the tensor is irrelevant, the underlying op is relevant
        """
        assert self._entered, f"{self} you need to be in the context scope"
        (self._extra_ops_true_branch if self._entered_state else self._extra_ops_false_branch).append(op)

    def add_other_branch_prehook(self, callback: Callable[[], Any]):
        """add prehook to the other branch"""
        assert self._entered, f"{self} you need to be in the context scope"
        if not self._entered_state:
            return  # already in false branch, will not enter any other branch
        self._false_branch_prehooks.insert(0, callback)

    def add_other_branch_posthook(self, callback: Callable[[], Any]):
        """add posthook to the other branch"""
        assert self._entered, f"{self} you need to be in the context scope"
        if not self._entered_state:
            return  # already in false branch, will not enter any other branch
        self._false_branch_posthooks.append(callback)


class CondModule(rf.Module):
    """
    This module is used internally by :class:`Cond` to create the RETURNN :class:`CondLayer` for the conditional code.
    This module would not be directly used by the user.
    """

    def __init__(self, cond: Cond):
        super(CondModule, self).__init__()
        self.cond = cond

    def __call__(self):
        """
        Makes layer dict for this loop, i.e. a RecLayer.

        :return: structure like true_value/false_value
        """
        name_ctx = self.cond.name_ctx
        # noinspection PyProtectedMember
        true_value, false_value = self.cond._true_value, self.cond._false_value
        true_values_flat = nest.flatten(true_value)  # type: List[Tensor]
        false_values_flat = nest.flatten(false_value)  # type: List[Tensor]
        assert len(true_values_flat) == len(false_values_flat)
        res = rfl.make_layer(
            {
                "class": "cond",
                "from": [],
                "condition": self.cond.condition,
                "true_layer": {
                    "class": "subnetwork",
                    "from": [],
                    "subnetwork": self.cond.true_branch_name_ctx.make_net(),
                },
                "false_layer": {
                    "class": "subnetwork",
                    "from": [],
                    "subnetwork": self.cond.false_branch_name_ctx.make_net(),
                },
            },
            name=name_ctx,
            predefined_out_data=true_values_flat[0].copy_template()
            if true_values_flat
            else Tensor("dummy", (), "int32"),
        )

        if true_values_flat:
            results = []
            for i, (true_v, false_v) in enumerate(zip(true_values_flat, false_values_flat)):
                assert isinstance(true_v, Tensor) and isinstance(false_v, Tensor)
                assert true_v.raw_tensor.parent is self.cond.true_branch_name_ctx
                name = true_v.raw_tensor.name
                if i == 0:
                    results.append(res)
                else:
                    # noinspection PyProtectedMember
                    results.append(rfl._get_sub_layer(res, name, data=true_v.copy_template()))
                results[-1].raw_tensor.layer_extra_dependencies.extend(
                    (self.cond.condition.raw_tensor, true_v.raw_tensor, false_v.raw_tensor)
                )
            res = nest.pack_sequence_as(true_value, results)
            if not results:
                results = [res]
        else:
            results = [res]

        # noinspection PyProtectedMember
        if self.cond._extra_ops_true_branch or self.cond._extra_ops_false_branch:
            # Make sure this is registered as output layer.
            assert not name_ctx.inner_control_flow()  # not implemented
            out = results[0]
            out = _utils.copy(out, name=name_ctx.root.get_new_child(out.name))
            out.raw_tensor.layer_dict["is_output_layer"] = True
            name_ctx.root.marked_outputs.append(out)

        return res
