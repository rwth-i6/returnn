"""
Reduce
"""

from __future__ import annotations
from typing import Union, TypeVar, Sequence
from returnn.util.basic import NotSpecified
from returnn.tensor import Tensor, Dim
from ._backend import get_backend_by_tensor

T = TypeVar("T")


def reduce(
    source: Tensor[T],
    *,
    mode: str,
    axis: Union[Dim, Sequence[Dim]],
    use_time_mask: bool = NotSpecified,
) -> Tensor[T]:
    """
    Reduce the tensor along the given axis

    :param source:
    :param mode: "sum", "max", "min", "mean", "logsumexp", "any", "all", "argmin", "argmax"
    :param axis:
    :param use_time_mask: if True (default), use the time mask (part of dim tag) to ignore padding frames
    :return: tensor with axis removed
    """
    rf = get_backend_by_tensor(source)
    return rf.reduce(source=source, mode=mode, axis=axis, use_time_mask=use_time_mask)
