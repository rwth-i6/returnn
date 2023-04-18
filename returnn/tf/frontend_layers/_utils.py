"""
Some utilities for internal use
"""

from __future__ import annotations
from typing import Optional, Union, Iterable, List
from returnn.util.basic import NotSpecified
from returnn.tensor import Tensor, Dim
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


def get_last_hidden_state(
    source: Tensor,
    *,
    out_dim: Optional[Dim] = NotSpecified,
    combine: str = NotSpecified,
    key: Optional[Union[str, int]] = NotSpecified,
) -> Tensor:
    """
    Will combine (concat or add or so) all the last hidden states from all sources.

    :param nn.Tensor source:
    :param nn.Dim|None out_dim:
    :param str combine: "concat" or "add"
    :param str|int|None key: for the state, which could be a namedtuple. see :func:`RnnCellLayer.get_state_by_key`
    :return: layer
    """
    if not isinstance(source, Tensor):
        raise TypeError(f"_get_last_hidden_state: unexpected type for source {source!r}, need tensor")
    args = {
        "out_dim": out_dim,
        "combine": combine,
        "key": key,
    }
    args = {key: value for (key, value) in args.items() if value is not NotSpecified}
    return rfl.make_layer({"class": "get_last_hidden_state", "from": source, **args}, name="get_last_hidden_state")
