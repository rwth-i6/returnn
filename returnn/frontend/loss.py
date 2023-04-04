"""
Loss functions
"""

from __future__ import annotations
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


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
