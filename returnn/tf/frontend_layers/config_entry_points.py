"""
Functions to get the network dict based on the config
via the API get_model, train_step, forward_step.

https://github.com/rwth-i6/returnn/issues/1120
"""

from __future__ import annotations
from typing import Any, Dict
from returnn.util.basic import BehaviorVersion
from returnn.tensor import TensorDict
from .. import frontend_layers as rfl


def get_net_dict(*, epoch: int, step: int, task: str) -> Dict[str, Any]:
    """called from the RETURNN config"""
    from returnn.config import get_global_config

    BehaviorVersion.set_min_behavior_version(rfl.min_returnn_behavior_version)

    config = get_global_config()
    rfl.Layer.reset_default_root()

    extern_data_dict = config.typed_value("extern_data")
    extern_data = TensorDict()
    extern_data.update(extern_data_dict, auto_convert=True)
    for data in extern_data.data.values():
        rfl.register_extern_data(data)

    get_model_func = config.typed_value("get_model")
    model = get_model_func(epoch=epoch, step=step)

    if task == "train":
        train_step_func = config.typed_value("train_step")
        train_step_func(
            model=model,
            extern_data=extern_data,
        )
    elif task == "forward":
        forward_step_func = config.typed_value("forward_step")
        forward_step_func(
            model=model,
            extern_data=extern_data,
        )
    else:
        raise ValueError(f"invalid task {task!r}")

    net_dict = rfl.Layer.top().root.get_returnn_config().get_net_dict_raw_dict(root_module=model)
    return net_dict
