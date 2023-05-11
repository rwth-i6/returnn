"""
Reduce
"""

from __future__ import annotations
from typing import Optional, Union, TypeVar, Sequence
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf

T = TypeVar("T")

__all__ = [
    "reduce",
    "reduce_sum",
    "reduce_max",
    "reduce_min",
    "reduce_mean",
    "reduce_logsumexp",
    "reduce_any",
    "reduce_all",
    "reduce_argmin",
    "reduce_argmax",
    "reduce_out",
]


def reduce(
    source: Tensor[T],
    *,
    mode: str,
    axis: Union[Dim, Sequence[Dim]],
    use_mask: bool = True,
) -> Tensor[T]:
    """
    Reduce the tensor along the given axis

    :param source:
    :param mode: "sum", "max", "min", "mean", "logsumexp", "any", "all", "argmin", "argmax"
    :param axis:
    :param use_mask: if True (default), use the time mask (part of dim tag) to ignore padding frames
    :return: tensor with axis removed
    """
    # noinspection PyProtectedMember
    return source._raw_backend.reduce(source=source, mode=mode, axis=axis, use_mask=use_mask)


def reduce_sum(source: Tensor[T], *, axis: Union[Dim, Sequence[Dim]], use_mask: bool = True) -> Tensor[T]:
    """
    Reduce the tensor along the given axis

    :param source:
    :param axis:
    :param use_mask: if True (default), use the time mask (part of dim tag) to ignore padding frames
    :return: tensor with axis removed
    """
    return reduce(source=source, mode="sum", axis=axis, use_mask=use_mask)


def reduce_max(source: Tensor[T], *, axis: Union[Dim, Sequence[Dim]], use_mask: bool = True) -> Tensor[T]:
    """
    Reduce the tensor along the given axis

    :param source:
    :param axis:
    :param use_mask: if True (default), use the time mask (part of dim tag) to ignore padding frames
    :return: tensor with axis removed
    """
    return reduce(source=source, mode="max", axis=axis, use_mask=use_mask)


def reduce_min(source: Tensor[T], *, axis: Union[Dim, Sequence[Dim]], use_mask: bool = True) -> Tensor[T]:
    """
    Reduce the tensor along the given axis

    :param source:
    :param axis:
    :param use_mask: if True (default), use the time mask (part of dim tag) to ignore padding frames
    :return: tensor with axis removed
    """
    return reduce(source=source, mode="min", axis=axis, use_mask=use_mask)


def reduce_mean(source: Tensor[T], *, axis: Union[Dim, Sequence[Dim]], use_mask: bool = True) -> Tensor[T]:
    """
    Reduce the tensor along the given axis

    :param source:
    :param axis:
    :param use_mask: if True (default), use the time mask (part of dim tag) to ignore padding frames
    :return: tensor with axis removed
    """
    return reduce(source=source, mode="mean", axis=axis, use_mask=use_mask)


def reduce_logsumexp(source: Tensor[T], *, axis: Union[Dim, Sequence[Dim]], use_mask: bool = True) -> Tensor[T]:
    """
    Reduce the tensor along the given axis

    :param source:
    :param axis:
    :param use_mask: if True (default), use the time mask (part of dim tag) to ignore padding frames
    :return: tensor with axis removed
    """
    return reduce(source=source, mode="logsumexp", axis=axis, use_mask=use_mask)


def reduce_any(source: Tensor[T], *, axis: Union[Dim, Sequence[Dim]], use_mask: bool = True) -> Tensor[T]:
    """
    Reduce the tensor along the given axis

    :param source:
    :param axis:
    :param use_mask: if True (default), use the time mask (part of dim tag) to ignore padding frames
    :return: tensor with axis removed
    """
    return reduce(source=source, mode="any", axis=axis, use_mask=use_mask)


def reduce_all(source: Tensor[T], *, axis: Union[Dim, Sequence[Dim]], use_mask: bool = True) -> Tensor[T]:
    """
    Reduce the tensor along the given axis

    :param source:
    :param axis:
    :param use_mask: if True (default), use the time mask (part of dim tag) to ignore padding frames
    :return: tensor with axis removed
    """
    return reduce(source=source, mode="all", axis=axis, use_mask=use_mask)


def reduce_argmin(source: Tensor[T], *, axis: Union[Dim, Sequence[Dim]], use_mask: bool = True) -> Tensor[T]:
    """
    Reduce the tensor along the given axis

    :param source:
    :param axis:
    :param use_mask: if True (default), use the time mask (part of dim tag) to ignore padding frames
    :return: tensor with axis removed
    """
    return reduce(source=source, mode="argmin", axis=axis, use_mask=use_mask)


def reduce_argmax(source: Tensor[T], *, axis: Union[Dim, Sequence[Dim]], use_mask: bool = True) -> Tensor[T]:
    """
    Reduce the tensor along the given axis

    :param source:
    :param axis:
    :param use_mask: if True (default), use the time mask (part of dim tag) to ignore padding frames
    :return: tensor with axis removed
    """
    return reduce(source=source, mode="argmax", axis=axis, use_mask=use_mask)


def reduce_out(
    source: Tensor,
    *,
    mode: str,
    num_pieces: int,
    out_dim: Optional[Dim] = None,
) -> Tensor:
    """
    Combination of :class:`SplitDimsLayer` applied to the feature dim
    and :class:`ReduceLayer` applied to the resulting feature dim.
    This can e.g. be used to do maxout.

    :param source:
    :param mode: "sum" or "max" or "mean"
    :param num_pieces: how many elements to reduce. The output dimension will be input.dim // num_pieces.
    :param out_dim:
    :return: out, with feature_dim set to new dim
    """
    assert source.feature_dim
    parts_dim = Dim(num_pieces, name="readout-parts")
    if not out_dim:
        out_dim = source.feature_dim // parts_dim
    out = rf.split_dims(source, axis=source.feature_dim, dims=(out_dim, parts_dim))
    out = reduce(out, mode=mode, axis=parts_dim)
    out.feature_dim = out_dim
    return out
