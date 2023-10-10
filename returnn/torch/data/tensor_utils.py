"""
Tensor utils.
"""

from __future__ import annotations
import numpy
import torch
from returnn.tensor import Tensor, TensorDict


def tensor_dict_numpy_to_torch_(x: TensorDict):
    """
    :func:`tensor_numpy_to_torch_` on all values
    """
    for v in x.data.values():
        tensor_numpy_to_torch_(v)


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
        dim.transform_tensors(tensor_numpy_to_torch_)


def tensor_dict_torch_to_numpy_(x: TensorDict):
    """
    :func:`tensor_torch_to_numpy_` on all values
    """
    for v in x.data.values():
        tensor_torch_to_numpy_(v)


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
        dim.transform_tensors(tensor_torch_to_numpy_)
