"""
Utils for modules
"""

from __future__ import annotations
from typing import Collection
import torch


def convert_parameters_to_buffers(
    module: torch.nn.Module, parameter_names: Collection[str], *, deep: bool = True, persistent: bool
):
    """
    :param module:
    :param parameter_names:
    :param deep: parameter_name can contain '.' to access submodules
    :param persistent: whether the buffer is persistent. if True, the buffer will be saved to the state_dict.
        passed to module.register_buffer.
    """
    for parameter_name in parameter_names:
        convert_parameter_to_buffer(module, parameter_name, deep=deep, persistent=persistent)


def convert_parameter_to_buffer(module: torch.nn.Module, parameter_name: str, *, deep: bool = True, persistent: bool):
    """
    :param module:
    :param parameter_name:
    :param deep: parameter_name can contain '.' to access submodules
    :param persistent: whether the buffer is persistent. if True, the buffer will be saved to the state_dict.
        passed to module.register_buffer.
    """
    if "." in parameter_name:
        if not deep:
            raise ValueError("parameter_name can't contain '.' when deep is False")
        module_path, _, parameter_name = parameter_name.rpartition(".")
        module = module.get_submodule(module_path)

    parameter = getattr(module, parameter_name)
    if not isinstance(parameter, torch.nn.Parameter):
        raise ValueError(f"{parameter_name} is not a torch.nn.Parameter, got type {type(parameter).__name__}")
    delattr(module, parameter_name)
    parameter.requires_grad = False
    module.register_buffer(parameter_name, parameter, persistent=persistent)
