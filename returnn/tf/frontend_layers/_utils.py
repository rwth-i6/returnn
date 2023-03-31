"""
Some utilities for internal use
"""

from __future__ import annotations
from typing import Union, Iterable, List
from returnn.tensor import Tensor
from returnn.util.basic import RefIdEq
from .. import frontend_layers as rfl


def unique_tensor_list(tensors: Iterable[Tensor]) -> List[Tensor]:
    """
    :param list[Tensor] tensors:
    :return: list with unique tensors
    :rtype: list[Tensor]
    """
    seen = set()  # over ref, not tensor (which is not hashable)
    out = []
    for tensor in tensors:
        if RefIdEq(tensor) not in seen:
            out.append(tensor)
            seen.add(RefIdEq(tensor))
    return out


def copy(tensor: Tensor[rfl.Layer], *, name: Union[rfl.Layer, str]) -> Tensor[rfl.Layer]:
    """copy"""
    return rfl.make_layer({"class": "copy", "from": tensor}, name=name)
