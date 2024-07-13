"""
Label smoothing
"""

from __future__ import annotations
from typing import Optional, Union, Sequence
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


__all__ = ["label_smoothing", "smooth_one_hot", "label_smoothed_log_prob_gradient"]


def label_smoothing(prob: Tensor, smoothing: Union[Tensor, float], *, axis: Optional[Dim] = None) -> Tensor:
    """
    Label smoothing, often used for cross entropy.

    In case of sparse data, it will become dense (via :func:`smooth_one_hot`)
    and the target label will get probability (1 - smoothing).
    """
    if not axis:
        assert prob.feature_dim or prob.sparse_dim
        axis = prob.feature_dim or prob.sparse_dim
    if prob.sparse_dim:
        assert prob.sparse_dim == axis
        return rf.smooth_one_hot(prob, label_prob=1.0 - smoothing)
    else:
        assert axis in prob.dims
        # Make it consistent to the sparse case.
        # Value of 1.0 should result in (1 - smoothing).
        # Value of 0.0 should result in smoothing / (dim - 1).
        # Sum over all should still remain 1.0.
        dim = axis.dimension
        floor_prob = smoothing / (dim - 1)
        factor = 1.0 - dim * floor_prob
        # Case for prob[i] == 0 is clear.
        # Case for prob[i] == 1: 1 - dim * floor_prob + floor_prob = 1 + (1 - dim) * floor_prob = 1 - smoothing
        # Sum over all: 1 - dim * floor_prob + floor_prob * dim = 1
        return prob * factor + floor_prob


def smooth_one_hot(source: Tensor, *, label_prob: Union[Tensor, float]) -> Tensor:
    """
    Smooth variant of :func:`one_hot`.
    Uses ``label_prob`` for the labels and ``(1 - label_prob) / (dim - 1)`` for the remaining values.
    This is used for label smoothing.
    """
    assert source.sparse_dim
    if source.sparse_dim.dimension is None:
        raise NotImplementedError(f"smooth_one_hot({source}) not implemented for dynamic dims")
    return rf.sparse_to_dense(
        source, label_value=label_prob, other_value=(1.0 - label_prob) / (source.sparse_dim.dimension - 1)
    )


def label_smoothed_log_prob_gradient(
    log_prob: Tensor,
    smoothing: Union[Tensor, float],
    *,
    axis: Optional[Dim] = None,
    exclude_labels: Optional[Sequence[int]] = None,
) -> Tensor:
    """
    :param log_prob: shape [...,D] (not necessarily the same as loss)
    :param smoothing: smoothing factor, for :func:`label_smoothing`
    :param axis: label axis in ``log_prob`` (D). uses feature_dim by default
    :param exclude_labels: list of labels to exclude from smoothing (e.g. blank)
    :return: ``log_prob``, but the gradient is smoothed

    Assume some cross-entropy-like loss::

        loss = - sum_i target_prob[i] * log_prob[i] .

    The sum is over the label indices i (corresponding to the ``axis`` argument).
    Then the gradient of loss w.r.t. log_prob[i] is::

        grad_logprob[i] loss = -target_prob[i] .

    We assume that the negative gradient is a probability distribution
    (potentially scaled by some factor, e.g. when you scale the loss by some factor)
    and apply :func:`label_smoothing` on it.
    More specifically, we apply the same scale and shift as in the :func:`label_smoothing` function
    via :func:`scaled_gradient_ext`.

    Note, this is also the case for CTC or RNNT loss,
    that the negative gradient of the loss w.r.t. the log-probabilities
    is a probability distribution.

    Common usage example::

        # E.g. there was some log_softmax, or anything to get log probs.
        log_probs = model.get_log_probs(...)

        # Now apply label smoothing on the log prob gradients.
        log_probs = rf.label_smoothed_log_prob_gradient(log_probs, 0.1)

        # E.g. CE, CTC, or similar, any kind of NLL should work.
        loss = loss_func(log_probs, targets)
        loss.sum().backward()

    Just as a side remark: assume::

        log_prob = log_softmax(z) .

    The gradient of log_softmax is::

        grad_z[j] log_prob[i] = delta(i==j) - softmax(z)[j] .

    Then the gradient w.r.t. z[j] is::

        grad_z[j] loss = sum_i (grad_logprob[i] loss) (grad_z[j] logprob[i])
                       = sum_i -target_prob[i] delta(i==j) + target_prob[i] softmax(z)[j]
                       = -target_prob[j] + (sum_i target_prob[i]) softmax(z)[j]
                       = softmax(z)[j] - target_prob[j]    # assuming (sum_i target_prob[i]) == 1

    """
    if not isinstance(smoothing, Tensor) and smoothing == 0:
        return log_prob  # no-op
    if not axis:
        assert log_prob.feature_dim
        axis = log_prob.feature_dim
    # See formula above for label_smoothing.
    dim = axis.dimension
    if not exclude_labels:
        # See formula in code comments in label_smoothing above.
        floor_prob = smoothing / (dim - 1)
        factor = 1.0 - dim * floor_prob
    else:  # have exclude_labels
        # Sum of prob over included labels does not change with this factor.
        # For prob[i] == 1, we still have (1 - smoothing) for the smoothed prob.
        floor_prob = smoothing / (dim - len(exclude_labels) - 1)
        factor = 1.0 - (dim - len(exclude_labels)) * floor_prob
        indices = rf.range_over_dim(axis)
        mask = True
        for label in exclude_labels:
            mask = mask & (indices != label)
        factor = rf.where(mask, factor, 1.0)
        floor_prob = rf.where(mask, floor_prob, 0.0)
    # The gradient is expected to be the negative target prob, thus negative floor_prob.
    # The gradient is expected to be 0. for masked frames, thus the clipping logic.
    return rf.scaled_gradient_ext(log_prob, scale=factor, shift=-floor_prob, scale_shift_by_sum_over_axis=axis)
