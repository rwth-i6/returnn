"""
Conditional logic

https://github.com/rwth-i6/returnn_common/issues/24
"""

from __future__ import annotations
from typing import List, TypeVar, Generic
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
        self._false_value = None
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
        assert true_value is not None
        assert self._true_value is None
        if isinstance(true_value, Tensor):
            true_value = _utils.copy(true_value, name=self.true_branch_name_ctx.get_child("output"))
        else:
            values_flat = nest.flatten(true_value)  # type: List[Tensor]
            assert values_flat
            for i, v in enumerate(values_flat):
                assert isinstance(v, Tensor), f"unexpected {true_value!r}, only expects tensors, got {type(v)}"
                if i == 0:
                    values_flat[i] = _utils.copy(v, name=self.true_branch_name_ctx.get_child("output"))
                else:
                    values_flat[i] = _utils.mark_as_output_in_scope(v, scope=self.true_branch_name_ctx)
            true_value = nest.pack_sequence_as(true_value, values_flat)
        self.true_branch_name_ctx.__exit__(None, None, None)
        self.false_branch_name_ctx.__enter__()
        self._true_value = true_value
        self._entered_state = False

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
        assert false_value is not None
        assert self._false_value is None
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
                assert isinstance(
                    false_v, Tensor
                ), f"unexpected {false_value!r}, only expects tensors, got {type(false_v)}"
                assert true_v.raw_tensor.parent is self.true_branch_name_ctx
                name = true_v.raw_tensor.name
                false_values_flat[i] = _utils.copy(false_v, name=self.false_branch_name_ctx.get_child(name))
                if name != "output":
                    false_values_flat[i].raw_tensor.layer_dict["is_output_layer"] = True
            false_value = nest.pack_sequence_as(false_value, false_values_flat)
        self.false_branch_name_ctx.__exit__(None, None, None)
        self._false_value = false_value
        self._result_value = self.layer_module()

    @property
    def result(self) -> T:
        """
        :return: the result, after you assigned :func:`true` and :func:`false`.
        """
        assert self._true_value is not None, f"{self} you need to have defined the true value"
        assert self._false_value is not None, f"{self} you need to have defined the false value"
        assert self._result_value is not None
        return self._result_value


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
            predefined_out_data=true_values_flat[0].copy_template(),
        )

        results = []
        for i, (true_v, false_v) in enumerate(zip(true_values_flat, false_values_flat)):
            assert isinstance(true_v, Tensor) and isinstance(false_v, Tensor)
            assert true_v.raw_tensor.parent is self.cond.true_branch_name_ctx
            name = true_v.raw_tensor.name
            if i == 0:
                results.append(res)
            else:
                # noinspection PyProtectedMember
                results.append(rfl._get_sub_layer(res, name, data=true_v.data.copy_template()))
            results[-1].raw_tensor.layer_extra_dependencies.extend(
                (self.cond.condition.raw_tensor, true_v.raw_tensor, false_v.raw_tensor)
            )
        return nest.pack_sequence_as(true_value, results)
