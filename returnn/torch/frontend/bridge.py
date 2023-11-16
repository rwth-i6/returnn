"""
Direct bridge between pure PyTorch code and modules
and RF code and modules.

https://github.com/rwth-i6/returnn/issues/1287
"""

from __future__ import annotations
from typing import Optional
import torch
import returnn.frontend as rf
from returnn.tensor import Dim


def pt_module_to_rf_module(pt_module: torch.nn.Module) -> rf.Module:
    """
    :param pt_module: torch module
    :return: RF module
    """
    assert isinstance(pt_module, torch.nn.Module)
    if isinstance(pt_module, _RFModuleAsPTModule):
        return pt_module.rf_module
    return _PTModuleAsRFModule(pt_module=pt_module)


def pt_module_to_wrapped_rf_module(pt_module: torch.nn.Module) -> Optional[rf.Module]:
    """
    :param pt_module: torch module
    :return: RF module if the torch module is a wrapped RF module, or None otherwise
    """
    assert isinstance(pt_module, torch.nn.Module)
    if isinstance(pt_module, _RFModuleAsPTModule):
        return pt_module.rf_module
    return None


def rf_module_to_pt_module(rf_module: rf.Module, *, aux_params_as_buffers: bool = True) -> torch.nn.Module:
    """
    :param rf_module: RF module
    :param aux_params_as_buffers: whether to map RF auxiliary parameters to PyTorch buffers,
        otherwise to normal parameters, i.e. they occur in model.named_parameters().
        Note that even when they are part of model.named_parameters(),
        aux params usually don't have a gradient, and then they are not updated by the optimizer.
        Historically, this was False.
        Now, this is True by default, as this is more reasonable.
        Note that the optimizer state dict will change if you change this,
        however, we will automatically convert such optimizer state dict.
    :return: torch module
    """
    assert isinstance(rf_module, rf.Module)
    if isinstance(rf_module, _PTModuleAsRFModule):
        return rf_module.pt_module
    return _RFModuleAsPTModule(rf_module=rf_module, aux_params_as_buffers=aux_params_as_buffers)


class _PTModuleAsRFModule(rf.Module):
    def __init__(self, pt_module: torch.nn.Module):
        super().__init__()
        self._pt_module = pt_module

        # recurse=False because param names cannot contain "."
        for name, pt_param in pt_module.named_parameters(recurse=False):
            rf_param = rf.Parameter(
                raw_tensor=pt_param,
                dims=[Dim(d) for d in pt_param.shape],
                dtype=str(pt_param.dtype).split(".")[-1],
            )
            setattr(self, name, rf_param)

        for name, pt_param in pt_module.named_buffers(recurse=False):
            rf_param = rf.Parameter(
                raw_tensor=pt_param,
                dims=[Dim(d) for d in pt_param.shape],
                dtype=str(pt_param.dtype).split(".")[-1],
                auxiliary=True,
            )
            setattr(self, name, rf_param)

        for name, rf_mod in pt_module.named_children():
            pt_mod = rf_module_to_pt_module(rf_mod)
            setattr(self, name, pt_mod)

    @property
    def pt_module(self) -> rf.Module:
        """RF module"""
        return self._pt_module

    def __call__(self, *args, **kwargs):
        """forward"""
        return self._pt_module(*args, **kwargs)


class _RFModuleAsPTModule(torch.nn.Module):
    def __init__(self, rf_module: rf.Module, *, aux_params_as_buffers: bool = True):
        super().__init__()
        self._rf_module = rf_module
        self._aux_params_as_buffers = aux_params_as_buffers

        # recurse=False because param names cannot contain "."
        for name, rf_param in rf_module.named_parameters(recurse=False):
            pt_param = rf_param.raw_tensor
            assert isinstance(pt_param, torch.nn.Parameter)
            if rf_param.auxiliary and aux_params_as_buffers:
                self.register_buffer(name, pt_param)
            else:
                self.register_parameter(name, pt_param)

        for name, rf_mod in rf_module.named_children():
            pt_mod = rf_module_to_pt_module(rf_mod, aux_params_as_buffers=aux_params_as_buffers)
            self.add_module(name, pt_mod)

    def _get_name(self):
        return self._rf_module.__class__.__name__ + "[RF→PT]"

    @property
    def rf_module(self) -> rf.Module:
        """RF module"""
        return self._rf_module

    def forward(self, *args, **kwargs):
        """forward"""
        return self._rf_module(*args, **kwargs)

    def _apply(self, fn):
        super()._apply(fn)

        # This could get called via `rf_module.to(device)`,
        # and there are cases where the Parameter.data was not updated inplace
        # but instead a new Parameter was created.
        # Update the corresponding RF Parameter.
        for name, rf_param in self._rf_module.named_parameters(recurse=False):
            pt_param = getattr(self, name)
            if rf_param.auxiliary and self._aux_params_as_buffers:
                assert isinstance(pt_param, torch.Tensor)  # but not torch.nn.Parameter
                # See similar logic in torch.nn.Module._apply.
                pt_param = torch.nn.Parameter(pt_param, pt_param.requires_grad)
            else:
                assert isinstance(pt_param, torch.nn.Parameter), (
                    f"{self}.{name} is not a Parameter" f" but {type(pt_param).__name__}"
                )
            rf_param.raw_tensor = pt_param
