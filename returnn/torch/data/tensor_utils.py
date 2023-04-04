"""
Tensor utils.
"""

from __future__ import annotations
import numpy
import torch
from typing import Optional
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


def fill_tensor_from_raw_tensor(tensor: Tensor, raw_tensor: torch.Tensor, size: Optional[torch.Tensor]) -> Tensor:
    """
    Fill an unset Tensor template (e.g. from ExternData) with a PyTorch raw tensor

    :param tensor: tensor template
    :param raw_tensor: extern_data_raw entry
    :param size: <key>:seq_len entry if available
    """
    tensor.dtype = str(raw_tensor.dtype).split(".")[-1]  # just overwrite for now...
    tensor.raw_tensor = raw_tensor

    # set batch size
    batch_size = raw_tensor.size()[0]
    tensor.dims[0].size = batch_size

    if size is not None:
        # Sequence lengths have to be on CPU for the later call to rnn.pack_padded_sequence
        size = size.cpu()
        tensor.dims[1].dyn_size_ext.dtype = str(size.dtype).split(".")[-1]  # just overwrite for now...
        tensor.dims[1].dyn_size_ext.raw_tensor = size

    return tensor
