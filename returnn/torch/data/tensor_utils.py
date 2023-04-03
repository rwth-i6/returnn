"""
Tensor utils.
"""

from __future__ import annotations
import numpy
import torch
from returnn.tensor import Tensor


def tensor_numpy_to_torch_(x: Tensor[numpy.ndarray]):
    """
    torch.from_numpy() on Tensor, including dims
    """
    if x.raw_tensor is None or isinstance(x.raw_tensor, torch.Tensor):
        pass
    else:
        assert isinstance(x.raw_tensor, numpy.ndarray)
        x.raw_tensor = torch.from_numpy(x.raw_tensor)
    for dim in x.dims:
        if dim.dyn_size_ext:
            tensor_numpy_to_torch_(dim.dyn_size_ext)


def tensor_torch_to_numpy_(x: Tensor[torch.Tensor]):
    """
    .numpy() on Tensor, including dims
    """
    if x.raw_tensor is None or isinstance(x.raw_tensor, numpy.ndarray):
        pass
    else:
        assert isinstance(x.raw_tensor, torch.Tensor)
        x.raw_tensor = x.raw_tensor.detach().cpu().numpy()
    for dim in x.dims:
        if dim.dyn_size_ext:
            tensor_torch_to_numpy_(dim.dyn_size_ext)
