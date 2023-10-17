"""
Loss functions
"""

from __future__ import annotations
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


__all__ = ["cross_entropy", "ctc_loss"]


def cross_entropy(
    *,
    estimated: Tensor,
    target: Tensor,
    axis: Dim,
    estimated_type: str,
) -> Tensor:
    """
    ``target`` is supposed to be in probability space (normalized). It can also be sparse, i.e. contain class indices.
    ``estimated`` can be probs, log-probs or logits, specified via ``estimated_type``.

    Assuming both are in probability space, the cross entropy is:

        H(target,estimated) = -reduce_sum(target * log(estimated), axis=axis)
                            = -matmul(target, log(estimated), reduce=axis)

    In case you want label smoothing, you can use e.g.::

        ce = nn.cross_entropy(
            target=nn.label_smoothing(target, 0.1),
            estimated=estimated)

    :param estimated: probs, log-probs or logits, specified via ``estimated_type``
    :param target: probs, normalized, can also be sparse
    :param axis: class labels dim over which softmax is computed
    :param estimated_type: "probs", "log-probs" or "logits"
    :return: cross entropy (same Dims as 'estimated' but without 'axis')
    """

    if estimated_type == "logits":
        # This is a common case and most backends provide optimized functions for it.
        # noinspection PyProtectedMember
        return estimated._raw_backend.softmax_cross_entropy_with_logits(logits=estimated, targets=target, axis=axis)
    if estimated_type == "probs":
        log_prob = rf.log(estimated)  # TODO: make numerically stable
    elif estimated_type == "log-probs":
        log_prob = estimated
    else:
        raise ValueError("estimated_type must be 'probs', 'log-probs' or 'logits'")
    if target.sparse_dim:
        return -rf.gather(log_prob, indices=target, axis=axis)
    return -rf.matmul(target, log_prob, reduce=axis)


def ctc_loss(
    *,
    logits: Tensor,
    targets: Tensor,
    input_spatial_dim: Dim,
    targets_spatial_dim: Dim,
    blank_index: int,
    max_approx: bool = False,
) -> Tensor:
    """
    Calculates the CTC loss.

    Internally, this uses :func:`returnn.tf.native_op.ctc_loss`
    which is equivalent to tf.nn.ctc_loss but more efficient.

    Output is of shape [B].

    :param logits: (before softmax). shape [B...,input_spatial,C]
    :param targets: sparse. shape [B...,targets_spatial] -> C
    :param input_spatial_dim: spatial dim of input logits
    :param targets_spatial_dim: spatial dim of targets
    :param blank_index: vocab index of the blank symbol
    :param max_approx: if True, use max instead of sum over alignments (max approx, Viterbi)
    :return: loss shape [B...]
    """
    # noinspection PyProtectedMember
    return logits._raw_backend.ctc_loss(
        logits=logits,
        targets=targets,
        input_spatial_dim=input_spatial_dim,
        targets_spatial_dim=targets_spatial_dim,
        blank_index=blank_index,
        max_approx=max_approx,
    )
