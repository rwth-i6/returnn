"""
Some utilities for internal use
"""

from __future__ import annotations
from typing import Optional, Union, Sequence, Iterable, List
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


def identity_with_control_deps(
    tensor: Tensor[rfl.Layer],
    control_deps: Sequence[Tensor[rfl.Layer]],
    *,
    name: Optional[Union[str, rfl.Layer]] = None,
) -> Tensor[rfl.Layer]:
    """
    :param tensor:
    :param control_deps:
    :param name:
    :return: tensor with control deps
    """
    return rfl.make_layer({"class": "identity", "from": tensor, "control_dependencies": control_deps}, name=name)


def zeros_like_as_output_in_scope(tensor: Tensor, *, name: rfl.Layer):
    """
    :param tensor:
    :param name:
    :return:
    """
    args = {}
    if tensor.sparse_dim:
        args["sparse_dim"] = tensor.sparse_dim
    shape_deps = rfl.get_dim_deps(tensor.dims)
    if shape_deps:
        args["shape_deps"] = shape_deps
    res = rfl.make_layer(
        {
            "class": "constant",
            "value": 0,
            "shape": tensor.dims,
            "dtype": tensor.dtype,
            **args,
            "is_output_layer": True,
        },
        name=name,
    )
    name.parent.marked_outputs.append(res)
    return res


def mark_as_output_in_scope(tensor: Tensor, scope: rfl.Layer) -> Tensor:
    """
    Mark this as an output.
    """
    assert tensor.raw_tensor.layer_dict, f"mark_as_output can only be called on a layer, not a layer-ref {tensor}."
    res = tensor
    if tensor.raw_tensor is scope.children.get("output"):
        pass  # not needed
    elif tensor.raw_tensor.parent is not scope:
        res = copy(tensor, name=scope.get_new_child(suggested_name=tensor.raw_tensor.get_abs_name(join_str="_")))
        res.raw_tensor.layer_dict["is_output_layer"] = True
    else:
        assert tensor.raw_tensor.parent is scope
        assert tensor.raw_tensor.layer_dict
        tensor.raw_tensor.layer_dict["is_output_layer"] = True
    scope.marked_outputs.append(res)
    return res


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
