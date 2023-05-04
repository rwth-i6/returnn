"""
Functions to get the network dict based on the config
via the API get_model, train_step, forward_step.

https://github.com/rwth-i6/returnn/issues/1120
"""

from __future__ import annotations
from typing import Any, Dict, Tuple
from tensorflow.python.util import nest
from returnn.util.basic import BehaviorVersion
from returnn.tensor import TensorDict, Tensor, Dim
from returnn.config import get_global_config
import returnn.frontend as rf
from .. import frontend_layers as rfl
from . import _utils


__all__ = ["get_net_dict"]


def get_net_dict(
    *,
    epoch: int,
    step: int,
) -> Tuple[Dict[str, Any], rf.Module]:
    """called from the RETURNN config"""
    BehaviorVersion.set_min_behavior_version(rfl.min_returnn_behavior_version)
    rf.select_backend_returnn_layers_tf()
    rfl.Layer.reset_default_root()

    config = get_global_config()

    # See :mod:`rf.rand` docstring for an explanation of this logic.
    random_seed = config.int("random_seed", 42)
    random_seed = (epoch * 193939 + step * 19937 + random_seed * 27644437 + 479001599) % (2**31)
    rf.set_random_seed(random_seed)

    extern_data = TensorDict()
    extern_data_dict = config.typed_value("extern_data")
    extern_data.update(extern_data_dict, auto_convert=True)
    assert isinstance(extern_data, TensorDict)
    for data in extern_data.data.values():
        rfl.register_extern_data(data)

    get_model_func = config.typed_value("get_model")
    model = get_model_func(epoch=epoch, step=step)

    task = config.value("task", None)
    if task in {"train", "eval"}:
        rf.init_train_step_run_ctx(train_flag=rfl.make_layer({"class": "train_flag"}, name="train_flag"))
        train_step_func = get_global_config().typed_value("train_step")
        train_step_func(
            model=model,
            extern_data=extern_data,
        )
    elif task in {"forward", "search"}:
        rf.init_forward_step_run_ctx()
        forward_step_func = get_global_config().typed_value("forward_step")
        forward_step_func(
            model=model,
            extern_data=extern_data,
        )
    else:
        raise ValueError(f"invalid task {task!r}")

    root_scope = rfl.Layer.top().root

    for loss in rf.get_run_ctx().losses.values():
        loss_t = _utils.copy(loss.loss, name=root_scope.get_new_child(suggested_name=loss.name))
        loss_t.raw_tensor.layer_dict["loss"] = "as_is"
        loss_opts = {}
        if loss.scale != 1:
            assert "loss_scale" not in loss_t.raw_tensor.layer_dict
            loss_opts["scale"] = loss.scale
        if loss.as_error:
            loss_opts["as_error"] = True
        if loss.use_normalized_loss:
            loss_opts["use_normalized_loss"] = True
        if not loss.use_flatten_frames:
            loss_opts["use_flatten_frames"] = False
        if loss.custom_inv_norm_factor is not None:
            loss_opts["custom_inv_norm_factor"] = loss.custom_inv_norm_factor
        if loss_opts:
            loss_t.raw_tensor.layer_dict["loss_opts"] = loss_opts
        # Add it to the root name scope marked_losses list.
        # Note that this logic might change.
        root_scope.marked_losses.append(loss_t)

    for out in rf.get_run_ctx().outputs.data.values():
        if out.name == "output" and out.name not in root_scope.children:
            layer = root_scope.get_child(out.name)
        else:
            layer = root_scope.get_new_child(suggested_name=out.name)
        out_t = _utils.copy(out, name=layer)
        if layer.name != "output":
            out_t.raw_tensor.layer_dict["is_output_layer"] = True
        root_scope.marked_outputs.append(out_t)

    net_dict = root_scope.get_returnn_config(root_module=model).get_net_dict_raw_dict()

    def _cleanup_net_dict_value(elem):
        assert not isinstance(elem, Tensor), f"not expected to see Tensor {elem} in net dict"
        if isinstance(elem, Dim):
            # Dim dyn_size_ext might be Tensor[rfl.Layer],
            # but now the TF engine actually wants to have Tensor[tf.Tensor].
            # Reset it now. The TF engine should redefine it again.
            if elem.dyn_size_ext:
                elem.dyn_size_ext.raw_tensor = None
        return elem

    net_dict = nest.map_structure(_cleanup_net_dict_value, net_dict)
    return net_dict, model
