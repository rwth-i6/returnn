"""
Loss functions
"""

from __future__ import annotations
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


__all__ = ["cross_entropy", "ctc_loss", "edit_distance"]


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
    logits_normalized: bool = False,
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
    :param logits_normalized: whether the logits are already normalized (e.g. via log-softmax)
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
        logits_normalized=logits_normalized,
        targets=targets,
        input_spatial_dim=input_spatial_dim,
        targets_spatial_dim=targets_spatial_dim,
        blank_index=blank_index,
        max_approx=max_approx,
    )


def edit_distance(a: Tensor, a_spatial_dim: Dim, b: Tensor, b_spatial_dim: Dim, *, dtype: str = "int32") -> Tensor:
    """
    :param a: [B,Ta]
    :param a_spatial_dim: Ta
    :param b: [B,Tb]
    :param b_spatial_dim: Tb
    :param dtype:
    :return: [B]
    """
    import numpy  # just for iinfo on dtype to get max value

    # The axis permutation is just an efficiency optimization.
    a = a.copy_transpose([a_spatial_dim] + a.remaining_dims(a_spatial_dim))
    b = b.copy_transpose([b_spatial_dim] + b.remaining_dims(b_spatial_dim))
    dev = a.device
    max_dist_err = numpy.iinfo(dtype).max
    n_a_max_len = a_spatial_dim.get_dim_value()
    n_b_max_len = b_spatial_dim.get_dim_value()
    if int(n_a_max_len) < int(n_b_max_len):
        a, b = b, a
        a_spatial_dim, b_spatial_dim = b_spatial_dim, a_spatial_dim
        n_a_max_len, n_b_max_len = n_b_max_len, n_a_max_len
    # Now n_a_max_len >= n_b_max_len.
    batch_dims = a.remaining_dims(a_spatial_dim)
    for dim in b.remaining_dims(b_spatial_dim):
        if dim not in batch_dims:
            batch_dims.append(dim)
    a_seq_len = a_spatial_dim.get_dyn_size_ext_for_device(dev)  # [B]
    b_seq_len = b_spatial_dim.get_dyn_size_ext_for_device(dev)  # [B]
    a_tensor_ext, (a_spatial_dim_ext,) = rf.pad(
        a, axes=[a_spatial_dim], padding=[(b_spatial_dim, b_spatial_dim)], handle_dynamic_dims=False
    )  # [Tb+Ta+Tb,B]
    a_spatial_dim_ext: Dim
    b_tensor_flipped = rf.reverse_sequence(b, axis=b_spatial_dim, handle_dynamic_dims=False)  # [Tb,B]
    entry_idx_ = rf.range_over_dim(b_spatial_dim, device=dev)  # [Tb]->Tb
    b_spatial_dim1 = b_spatial_dim + 1
    buffer_dim = Dim(3 * b_spatial_dim1.get_dim_value_tensor(), name="buffer")
    buffer = rf.Parameter([buffer_dim] + batch_dims, device=dev, dtype=dtype, auxiliary=True)  # [3*(Tb+1),B]
    buffer_offsets = [0, b_spatial_dim1.get_dim_value_tensor(), b_spatial_dim1.get_dim_value_tensor() * 2]
    result = rf.where((a_seq_len == 0) & (b_seq_len == 0), 0, max_dist_err)  # [B]  # noqa

    # We are going diagonal over (Ta+1) and (Tb+1). (Similar as RETURNN native EditDistanceOp.)
    # You need to draw the grid on paper to understand all the index math...
    for u in range(1, n_a_max_len + n_b_max_len + 1):
        prev2_dist, _ = rf.slice(
            buffer, axis=buffer_dim, start=buffer_offsets[u % 3], size=b_spatial_dim1, out_dim=b_spatial_dim1
        )  # [Tb+1,B]
        prev_dist, _ = rf.slice(
            buffer, axis=buffer_dim, start=buffer_offsets[(u + 1) % 3], size=b_spatial_dim1, out_dim=b_spatial_dim1
        )  # [Tb+1,B]
        cur_dist_start_offset = buffer_offsets[(u + 2) % 3]

        del_cost = (
            rf.slice(prev_dist, axis=b_spatial_dim1, end=b_spatial_dim.get_dim_value_tensor(), out_dim=b_spatial_dim)[0]
            + 1
        )  # [Tb,B]
        ins_cost = rf.slice(prev_dist, axis=b_spatial_dim1, start=1, out_dim=b_spatial_dim)[0] + 1  # [Tb,B]
        sub_cost = rf.slice(prev2_dist, axis=b_spatial_dim1, start=1, out_dim=b_spatial_dim)[0] + rf.cast(
            rf.slice(a_tensor_ext, axis=a_spatial_dim_ext, start=u - 1, size=b_spatial_dim, out_dim=b_spatial_dim)[0]
            != b_tensor_flipped,
            dtype=dtype,
        )
        min_cost = rf.minimum(del_cost, ins_cost, sub_cost)  # [Tb,B]
        t_a_gt_zero_mask = entry_idx_ > n_b_max_len - u  # [Tb]

        buffer.assign_key(
            axis=buffer_dim,
            key=slice(cur_dist_start_offset, cur_dist_start_offset + b_spatial_dim.get_dim_value_tensor()),
            key_dim=b_spatial_dim,
            value=rf.where(t_a_gt_zero_mask, min_cost, u),
        )
        # last entry in cur_dist, that is where t_b == 0
        buffer.assign_key(
            axis=buffer_dim, key=cur_dist_start_offset + b_spatial_dim.get_dim_value_tensor(), key_dim=None, value=u
        )

        end_offset_a = n_b_max_len + a_seq_len - u  # [B]
        end_offset_b = n_b_max_len - b_seq_len  # [B]
        result = rf.where(
            end_offset_a == end_offset_b,
            rf.gather(buffer, axis=buffer_dim, indices=cur_dist_start_offset + end_offset_a, clip_to_valid=True),
            result,
        )

    return result
