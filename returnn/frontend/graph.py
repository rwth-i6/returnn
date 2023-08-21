"""
Graph utils.

They might be just available for graph-based frameworks,
but not necessarily.
"""

from __future__ import annotations
from typing import Union, Sequence, Callable, List
from returnn.tensor import Tensor


__all__ = [
    "get_tensor_dependencies",
    "get_tensor_consumers",
    "walk_tensor_consumers",
]


def get_tensor_dependencies(x: Tensor) -> Sequence[Tensor]:
    """
    :param x: tensor
    :return: list of tensors which are inputs to x
    """
    # noinspection PyProtectedMember
    return x._raw_backend.get_tensor_dependencies(x)


def get_tensor_consumers(x: Tensor) -> Sequence[Tensor]:
    """
    :param x: tensor
    :return: list of tensors which consume x
    """
    # noinspection PyProtectedMember
    return x._raw_backend.get_tensor_consumers(x)


def walk_tensor_consumers(
    seed: Union[Tensor, Sequence[Tensor]],
    *,
    filter_outputs: Callable[[Tensor], bool] = None,
    ending_condition: Callable[[Tensor], bool] = None,
) -> List[Tensor]:
    """
    :param seed: tensor
    :param filter_outputs: if given, this function will be called with each tensor,
        and if it returns False, the tensor will be skipped.
    :param ending_condition: if given, this function will be called with each tensor,
        and if it returns True, we return.
    :return: yields all tensors which consume seed, and their consumers, recursively
    """
    seen = set()
    queue = [seed] if isinstance(seed, Tensor) else list(seed)
    outputs = []
    while queue:
        x = queue.pop(0)
        if x in seen:
            continue
        seen.add(x)
        if not filter_outputs or filter_outputs(x):
            outputs.append(x)
        if ending_condition and ending_condition(x):
            break
        queue.extend(get_tensor_consumers(x))
    return outputs
