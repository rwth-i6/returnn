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
    if isinstance(pt_module, RFModuleAsPTModule):
        return pt_module.rf_module
    return PTModuleAsRFModule(pt_module=pt_module)


def wrapped_pt_module_to_rf_module(pt_module: torch.nn.Module) -> Optional[rf.Module]:
    """
    :param pt_module: torch module
    :return: RF module if the torch module is a wrapped RF module, or None otherwise
    """
    assert isinstance(pt_module, torch.nn.Module)
    if isinstance(pt_module, RFModuleAsPTModule):
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
    if isinstance(rf_module, PTModuleAsRFModule):
        return rf_module.pt_module
    return RFModuleAsPTModule(rf_module=rf_module, aux_params_as_buffers=aux_params_as_buffers)


class PTModuleAsRFModule(rf.Module):
    """
    Wrapped module.

    It is recommended to use :func:`pt_module_to_rf_module` instead of using this directly.
    """

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

        for name, pt_mod in pt_module.named_children():
            rf_mod = pt_module_to_rf_module(pt_mod)
            setattr(self, name, rf_mod)

    @property
    def pt_module(self) -> rf.Module:
        """RF module"""
        return self._pt_module

    def __call__(self, *args, **kwargs):
        """forward"""
        return self._pt_module(*args, **kwargs)


class RFModuleAsPTModule(torch.nn.Module):
    """
    Wrapped module.

    It is recommended to use :func:`rf_module_to_pt_module` instead of using this directly.
    """

    def __init__(self, rf_module: rf.Module, *, aux_params_as_buffers: bool = True):
        super().__init__()
        self._rf_module = rf_module
        self._aux_params_as_buffers = aux_params_as_buffers
        self._is_initializing = True

        # recurse=False because param names cannot contain ".", will add submodules below recursively.
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

        self._is_initializing = False

    def _get_name(self):
        return self._rf_module.__class__.__name__ + "[RF→PT]"

    @property
    def rf_module(self) -> rf.Module:
        """RF module"""
        return self._rf_module

    def forward(self, *args, **kwargs):
        """forward"""
        return self._rf_module(*args, **kwargs)

    def _apply(self, *args, **kwargs):
        # Note: Use generic *args, **kwargs, as the signature slightly changed,
        # `recurse` was added in PyTorch 2.0 or so.
        super()._apply(*args, **kwargs)

        # This could get called via `rf_module.to(device)`,
        # and there are cases where the Parameter.data was not updated inplace
        # but instead a new Parameter was created.
        # Update the corresponding RF Parameter.
        # recurse=False because super()._apply() already recursively calls _apply().
        for name, rf_param in self._rf_module.named_parameters(recurse=False):
            pt_param = getattr(self, name)
            if rf_param.auxiliary and self._aux_params_as_buffers:
                if not isinstance(pt_param, torch.nn.Parameter):
                    assert isinstance(pt_param, torch.Tensor)  # but not torch.nn.Parameter
                    # See similar logic in torch.nn.Module._apply.
                    pt_param = torch.nn.Parameter(pt_param, pt_param.requires_grad)
            # Otherwise, we do not care whether it is a torch.nn.Parameter or not.
            # Its type might have changed due to convert_parameters_to_buffers.
            # Just make sure it is a tensor.
            assert isinstance(pt_param, torch.Tensor)
            # noinspection PyProtectedMember
            rf_param.dtype = rf_param._raw_backend.get_dtype_name_raw(pt_param)  # dtype might have changed
            rf_param.raw_tensor = pt_param

    def register_parameter(self, name: str, param: Optional[torch.nn.Parameter]) -> None:
        """(re)register parameter"""
        super().register_parameter(name, param)
        if param is None or self._is_initializing:
            return  # just ignore
        rf_param = getattr(self._rf_module, name, None)
        if not isinstance(rf_param, rf.Parameter):
            return  # just ignore
        assert isinstance(
            param, torch.nn.Parameter
        ), f"{self} register_parameter {name}: did not get a Parameter but {type(param).__name__}"
        rf_param.raw_tensor = param

    def register_buffer(self, name: str, tensor: Optional[torch.Tensor], persistent: bool = True) -> None:
        """(re)register buffer"""
        super().register_buffer(name, tensor, persistent=persistent)
        if tensor is not None and persistent and self._aux_params_as_buffers and not self._is_initializing:
            rf_param = getattr(self._rf_module, name, None)
            if not isinstance(rf_param, rf.Parameter):
                return  # just ignore
            # See similar logic in torch.nn.Module._apply.
            pt_param = torch.nn.Parameter(tensor, tensor.requires_grad)
            rf_param.raw_tensor = pt_param
