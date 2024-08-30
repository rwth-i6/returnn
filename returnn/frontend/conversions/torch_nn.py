"""
Import some of the torch.nn modules.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import returnn.frontend as rf

if TYPE_CHECKING:
    import torch


def import_params_torch_linear_to_rf(model_pt: torch.nn.Linear, model_rf: rf.Linear):
    """
    import params from torch.nn.Linear to rf.Linear
    """
    import torch

    assert isinstance(model_pt, torch.nn.Linear)
    assert isinstance(model_rf, rf.Linear)
    assert model_rf.with_bias == (model_pt.bias is not None)

    with torch.no_grad():
        model_rf.weight.raw_tensor.copy_(model_pt.weight.T)  # (in,out)
        if model_rf.with_bias:
            model_rf.bias.raw_tensor.copy_(model_pt.bias)  # (out,)


def import_params_torch_conv1d_to_rf(model_pt: torch.nn.Conv1d, model_rf: rf.Conv1d):
    """
    import params from torch.nn.Conv1d to rf.Conv1d
    """
    import torch

    assert isinstance(model_pt, torch.nn.Conv1d)
    assert isinstance(model_rf, rf.Conv1d)
    assert model_rf.with_bias == (model_pt.bias is not None)

    with torch.no_grad():
        # Torch shape: out_channels, in_channels // groups, *kernel_size
        # RF shape: self.out_dim, self.filter_in_dim, *self.filter_size, i.e. should be same
        model_rf.filter.raw_tensor.copy_(model_pt.weight)
        if model_rf.with_bias:
            model_rf.bias.raw_tensor.copy_(model_pt.bias)


def import_params_torch_layer_norm_to_rf(model_pt: torch.nn.LayerNorm, model_rf: rf.LayerNorm):
    """
    Import the parameters from torch.nn.LayerNorm to rf.LayerNorm.
    """
    import torch

    assert isinstance(model_pt, torch.nn.LayerNorm)
    assert isinstance(model_rf, rf.LayerNorm)
    assert model_pt.weight.shape[0] == model_rf.in_dim.dimension

    with torch.no_grad():
        model_rf.scale.raw_tensor.copy_(model_pt.weight)  # (in,)
        model_rf.bias.raw_tensor.copy_(model_pt.bias)  # (in,)

    num_params_pt = 0
    for k, v in model_pt.named_parameters():
        num_params_pt += v.numel()
    num_params_rf = 0
    for k, v in model_rf.named_parameters():
        assert isinstance(v.raw_tensor, torch.nn.Parameter)
        num_params_rf += v.num_elements()
    assert num_params_rf == num_params_pt
