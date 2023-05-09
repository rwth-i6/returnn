"""
Direct bridge between pure PyTorch code and modules
and RF code and modules.

https://github.com/rwth-i6/returnn/issues/1287
"""

from __future__ import annotations
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


def rf_module_to_pt_module(rf_module: rf.Module) -> torch.nn.Module:
    """
    :param rf_module: RF module
    :return: torch module
    """
    assert isinstance(rf_module, rf.Module)
    if isinstance(rf_module, _PTModuleAsRFModule):
        return rf_module.pt_module
    return _RFModuleAsPTModule(rf_module=rf_module)


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
    def __init__(self, rf_module: rf.Module):
        super().__init__()
        self._rf_module = rf_module

        # recurse=False because param names cannot contain "."
        for name, rf_param in rf_module.named_parameters(recurse=False):
            pt_param = rf_param.raw_tensor
            assert isinstance(pt_param, torch.nn.Parameter)
            self.register_parameter(name, pt_param)

        for name, rf_mod in rf_module.named_children():
            pt_mod = rf_module_to_pt_module(rf_mod)
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
