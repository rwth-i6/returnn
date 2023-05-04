"""
Parameter assign logic.

This code is in its own file because it is somewhat complex.
The complexity comes from making sure that the parameter assign logic is executed in the right order
for this graph-based backend.
"""

from __future__ import annotations

from tensorflow.python.util import nest
from returnn.tensor import Tensor
from ... import frontend as rf
from .layer import Layer
from .. import frontend_layers as rfl
from . import _utils


def parameter_assign(param: rf.Parameter[Layer], value: Tensor, op: str = "assign") -> None:
    """
    Parameter assign.

    :param param:
    :param value:
    :param op:
    :return:
    """
    # Find proper layer context to create the assign op.
    # This is about finding the proper control flow context.
    layer_ctx = rfl.Layer.top()
    root_ctx = layer_ctx.root
    while layer_ctx and not layer_ctx.new_control_flow_ctx:
        layer_ctx = layer_ctx.parent
    with layer_ctx or root_ctx:
        op_ = rfl.make_layer({"class": "variable_assign", "var": param, "value": value, "op": op}, name=f"param_{op}")

    if layer_ctx:
        assert isinstance(layer_ctx.module, rfl.CondModule)  # not implemented otherwise
        layer_ctx.module.cond.add_op_to_current_branch(op_)

        def _other_branch_prehook():
            # Reset the param to the original value. We cannot depend on the true-branch value.
            assign_helper.reset_to_old_param()

        def _other_branch_posthook():
            # We are resetting the param now to be a variable-read.
            # So, any usage of the old param value should point to our copy.
            assign_helper.map_param_usage_to_old_param_copy()

            # Now set it again to the true-branch value.
            if param.raw_tensor.layer_dict["class"] == "variable_read":
                # Can keep that, just add the control dep.
                param.raw_tensor.layer_dict["control_dependencies"].append(op_)
            else:
                # Set again.
                assign_helper.reassign_param_variable_read()

        layer_ctx.module.cond.add_other_branch_prehook(_other_branch_prehook)
        layer_ctx.module.cond.add_other_branch_posthook(_other_branch_posthook)

    # Make sure it is always executed.
    _utils.mark_as_output_in_scope(op_, scope=root_ctx)

    assign_helper = _AssignHelper(param=param, assign_op=op_, op_kind=op)
    assign_helper.reassign_param_variable_read()
    assign_helper.map_param_usage_to_old_param_copy()


class _AssignHelper:
    def __init__(self, *, param: rf.Parameter[Layer], assign_op: Tensor, op_kind: str):
        self.param = param
        self.assign_op = assign_op
        self.op_kind = op_kind

        # We need to make sure that any further access to the param will use the new value.
        # So replace the raw param with an identity layer with control deps.
        self.old_param_copy: Tensor[rfl.Layer] = param.copy()
        self.old_param_raw: rfl.Layer = param.raw_tensor

    def reassign_param_variable_read(self):
        """reassign"""
        self.old_param_raw.tensor = self.old_param_copy
        self.param.raw_tensor = None  # reassign
        root_ctx = rfl.Layer.top().root
        with root_ctx:
            # We expect that RETURNN handles this properly.
            rfl.make_layer(
                {"class": "variable_read", "var": self.old_param_copy, "control_dependencies": [self.assign_op]},
                name=f"{self.param.name}_{self.op_kind}",
                out=self.param,
            )

    def map_param_usage_to_old_param_copy(self):
        """map"""

        def _map_usage_value(v):
            if isinstance(v, Tensor) and v is self.param:
                return self.old_param_copy
            return v

        for usage in self.old_param_raw.usages:
            usage.layer_dict = nest.map_structure(_map_usage_value, usage.layer_dict)

    def reset_to_old_param(self):
        """reset"""
        self.param.raw_tensor = self.old_param_raw
        self.old_param_raw.tensor = self.param
