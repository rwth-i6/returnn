"""
Reduce
"""

from __future__ import annotations
from typing import Optional, Union, TypeVar, Sequence, Tuple
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
    "reduce_logmeanexp",
    "reduce_any",
    "reduce_all",
    "reduce_argmin",
    "reduce_argmax",
    "reduce_out",
    "RunningMean",
    "top_k",
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


def reduce_logmeanexp(source: Tensor[T], *, axis: Union[Dim, Sequence[Dim]], use_mask: bool = True) -> Tensor[T]:
    """
    Reduce the tensor along the given axis

    :param source:
    :param axis:
    :param use_mask: if True (default), use the time mask (part of dim tag) to ignore padding frames
    :return: tensor with axis removed
    """
    s = reduce_logsumexp(source, axis=axis, use_mask=use_mask)
    return s - rf.safe_log(rf.cast(rf.num_elements_of_shape(axis, use_mask=use_mask, device=s.device), s.dtype))


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


class RunningMean(rf.Module):
    """
    Running mean, using exponential moving average, using the formula::

        # E.g. for some input [B,T,F], reduce to [F], when the mean vector is [F].
        new_value = reduce_mean(new_value, axis=[d for d in x.dims if d not in mean.dims])

        new_mean = alpha * new_value + (1 - alpha) * old_mean
                 = old_mean + alpha * (new_value - old_mean)  # more numerically stable

    (Like the TF :class:`AccumulateMeanLayer`.)
    (Similar is also the running mean in :class:`BatchNorm`.)
    """

    def __init__(
        self,
        in_dim: Union[Dim, Sequence[Dim]],
        *,
        alpha: float,
        dtype: Optional[str] = None,
        is_prob_distribution: Optional[bool] = None,
        update_only_in_train: bool = True,
    ):
        """
        :param in_dim: the dim of the mean vector, or the shape.
        :param alpha: factor for new_value. 0.0 means no update, 1.0 means always the new value.
            Also called momentum. E.g. 0.1 is a common value, or less, like 0.001.
        :param dtype: the dtype of the mean vector
        :param is_prob_distribution: if True, will initialize the mean vector with 1/in_dim.
        :param update_only_in_train: if True (default), will only update the mean vector in training mode.
            False means it will always update.
        """
        super().__init__()
        self.in_dim = in_dim
        self.shape = (in_dim,) if isinstance(in_dim, Dim) else in_dim
        assert all(isinstance(d, Dim) for d in self.shape)
        self.alpha = alpha
        self.is_prob_distribution = is_prob_distribution
        self.mean = rf.Parameter(self.shape, dtype=dtype, auxiliary=True)
        if is_prob_distribution:
            assert in_dim.dimension is not None
            self.mean.initial = 1.0 / in_dim.dimension
        self.update_only_in_train = update_only_in_train

    def __call__(self, x: Tensor) -> Tensor:
        """
        :param x: shape [..., F]
        :return: shape [F]
        """

        def _update_running_stats():
            assert all(d in self.shape for d in x.dims)
            x_ = rf.reduce_mean(x, axis=[d for d in x.dims if d not in self.shape])
            self.mean.assign_add(self.alpha * (x_ - self.mean))

        rf.cond((not self.update_only_in_train) or rf.get_run_ctx().train_flag, _update_running_stats, lambda: None)
        return self.mean


# noinspection PyShadowingBuiltins
def top_k(
    source: Tensor,
    *,
    axis: Union[Dim, Sequence[Dim]],
    k: Optional[Union[int, Tensor]] = None,
    k_dim: Optional[Dim] = None,
    sorted: bool = True,
) -> Tuple[Tensor, Union[Tensor, Sequence[Tensor]], Dim]:
    """
    Basically wraps tf.nn.top_k.
    Returns the top_k values and the indices.

    For an input [B,D] with axis=D, the output and indices values are shape [B,K].

    It's somewhat similar to :func:`reduce` with max and argmax.
    The axis dim is reduced and then a new dim for K is added.

    Axis can also cover multiple axes, such as [beam,classes].
    In that cases, there is not a single "indices" sub-layer,
    but sub-layers "indices0" .. "indices{N-1}"
    corresponding to each axis, in the same order.

    All other axes are treated as batch dims.

    :param source:
    :param axis: the axis to do the top_k on, which is reduced, or a sequence of axes
    :param k: the "K" in "TopK"
    :param k_dim: the new axis dim for K. if not provided, will be automatically created.
    :param sorted:
    :return: values, indices (sequence if axis is a sequence), k_dim
    """
    if k is None:
        assert k_dim, "top_k: either provide `k` or `k_dim`"
        k = k_dim.dimension or k_dim.dyn_size_ext
        assert k is not None, f"top_k: k_dim {k_dim} undefined and no k provided"
    # noinspection PyProtectedMember
    return source._raw_backend.top_k(source, axis=axis, k=k, k_dim=k_dim, sorted=sorted)
