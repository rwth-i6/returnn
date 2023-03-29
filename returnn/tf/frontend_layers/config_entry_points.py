"""
Functions to get the network dict based on the config
via the API get_model, train_step, forward_step.

https://github.com/rwth-i6/returnn/issues/1120
"""

from __future__ import annotations
from typing import Any, Optional, Union, Dict
from returnn.util.basic import BehaviorVersion
from returnn.tensor import TensorDict
from returnn.config import get_global_config
import returnn.frontend as rf
from .. import frontend_layers as rfl

try:
    from typing import Protocol
except ImportError:
    try:
        from typing_extensions import Protocol
    except ImportError:
        Protocol = object


class GetModelFunc(Protocol):
    """get model func"""

    def __call__(self, *, epoch: int, step: int) -> rf.Module:
        ...


class StepFunc(Protocol):
    """step func"""

    def __call__(self, *, model: rf.Module, extern_data: TensorDict) -> None:
        ...


def get_net_dict(
    *,
    epoch: int,
    step: int,
    get_model_func: Optional[GetModelFunc] = None,
    extern_data: Optional[TensorDict],
    step_func: Union[str, StepFunc],
) -> Dict[str, Any]:
    """called from the RETURNN config"""
    BehaviorVersion.set_min_behavior_version(rfl.min_returnn_behavior_version)
    rf.select_backend_returnn_layers_tf()
    rfl.Layer.reset_default_root()

    if extern_data is None:
        extern_data = TensorDict()
        extern_data_dict = get_global_config().typed_value("extern_data")
        extern_data.update(extern_data_dict, auto_convert=True)
    for data in extern_data.data.values():
        rfl.register_extern_data(data)

    if get_model_func is None:
        get_model_func = get_global_config().typed_value("get_model")
    model = get_model_func(epoch=epoch, step=step)

    if callable(step_func):
        step_func(model=model, extern_data=extern_data)
    elif step_func == "train":
        train_step_func = get_global_config().typed_value("train_step")
        train_step_func(
            model=model,
            extern_data=extern_data,
        )
    elif step_func == "forward":
        forward_step_func = get_global_config().typed_value("forward_step")
        forward_step_func(
            model=model,
            extern_data=extern_data,
        )
    else:
        raise ValueError(f"invalid step_func {step_func!r}")

    net_dict = rfl.Layer.top().root.get_returnn_config().get_net_dict_raw_dict(root_module=model)
    return net_dict
