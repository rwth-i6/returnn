"""
Parameter assign logic.

This code is in its own file because it is somewhat complex.
The complexity comes from making sure that the parameter assign logic is executed in the right order
for this graph-based backend.
"""

from __future__ import annotations
from typing import Optional, Sequence
from tensorflow.python.util import nest
from returnn.tensor import Tensor
from ... import frontend as rf
from .layer import Layer
from .. import frontend_layers as rfl
from . import _utils


def parameter_assign(param: rf.Parameter, value: Tensor, *, op: str = "assign") -> None:
    """
    Parameter assign.

    :param param:
    :param value:
    :param op:
    :return:
    """
    if param.raw_tensor.layer_dict["class"] == "variable":
        # This is the first assignment on the variable.
        # There might be some previous reads.
        # To make sure that any previous reads are performed before the assign,
        # we need to replace any previous access by an explicit variable-read.
        assign_helper_initial_read = _AssignHelper(
            param=param, read_layer_name=f"{param.name}_initial_read", read_control_deps=None
        )
        assign_helper_initial_read.reassign_param_variable_read()
        # Copy over the usage refs, such that we properly replace the instance below.
        param.raw_tensor.usages.extend(assign_helper_initial_read.old_param_raw.usages)

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
            if param.raw_tensor is assign_helper.old_param_raw:
                # Set again.
                assign_helper.reassign_param_variable_read()
            else:
                # Some other code has overwritten it again in the meantime.
                # This was probably another parameter_assign.
                assert param.raw_tensor.layer_dict["class"] == "variable_read"
                # Can keep that, just add the control dep.
                param.raw_tensor.layer_dict["control_dependencies"].append(op_)

        layer_ctx.module.cond.add_other_branch_prehook(_other_branch_prehook)
        layer_ctx.module.cond.add_other_branch_posthook(_other_branch_posthook)

    # Make sure it is always executed.
    _utils.mark_as_output_in_scope(op_, scope=root_ctx)

    assign_helper = _AssignHelper(param=param, read_layer_name=f"{param.name}_after_{op}", read_control_deps=[op_])
    assign_helper.reassign_param_variable_read()
    assign_helper.map_param_usage_to_old_param_copy()


class _AssignHelper:
    def __init__(
        self, *, param: rf.Parameter[Layer], read_layer_name: str, read_control_deps: Optional[Sequence[Tensor]]
    ):
        self.param = param
        self.read_layer_name = read_layer_name
        self.read_control_deps = read_control_deps

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
                {"class": "variable_read", "var": self.old_param_copy, "control_dependencies": self.read_control_deps},
                name=self.read_layer_name,
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
