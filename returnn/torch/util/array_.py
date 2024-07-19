"""
Array (Tensor) functions
"""

from __future__ import annotations
from typing import Optional, Union
import torch


# noinspection PyShadowingBuiltins
def masked_select(input: torch.Tensor, mask: torch.Tensor, *, mask_len: Optional[Union[int, torch.Tensor]] = None):
    """
    Like :func:`torch.masked_select` but much more efficient,
    both in terms of memory and computation time,
    both on CPU and GPU.

    See here for the issues with :func:`torch.masked_select`:
    https://github.com/rwth-i6/returnn/issues/1584
    https://github.com/pytorch/pytorch/issues/30246
    https://github.com/pytorch/pytorch/issues/56896

    :param input: [mask_dims..., remaining_dims...]
    :param mask: [mask_dims...], binary mask to index with. if it has less dims than ``input``,
        the remaining dims are broadcasted.
    :param mask_len: if given, the length of the mask. this avoids a CUDA synchronization.
    :return: selected elements, shape [mask_len, remaining_dims...]
    """
    assert input.ndim >= mask.ndim
    assert all(input.shape[i] == mask.shape[i] for i in range(mask.ndim))
    mask_flat = mask.flatten()
    if mask_len is not None:
        indices = nonzero(mask_flat, out_len=mask_len)  # [out_len]
    else:
        indices = torch.nonzero(mask_flat).squeeze(1)  # [out_len]
    input_flat = input.flatten(end_dim=mask.ndim - 1)
    return input_flat[indices]


def nonzero(mask: torch.Tensor, *, out_len: Union[int, torch.Tensor]) -> torch.Tensor:
    """
    This has the advantage over :func:`torch.nonzero`
    that we do not need to perform a CUDA synchronization.
    We can avoid that when we know the output length in advance.

    :param mask: flattened (dim() == 1) mask, bool
    :param out_len:
    :return: indices of True elements, shape [out_len].
        like ``mask.nonzero().flatten()``
    """
    assert mask.dim() == 1 and mask.dtype == torch.bool
    # Sort currently does not support bool dtype on CUDA, thus cast to int.
    idx = torch.argsort(mask.to(torch.int8), stable=True, descending=True)  # [in_len]
    idx = idx[:out_len]  # [out_len]
    return idx
