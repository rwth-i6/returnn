"""
Loss functions
"""

from __future__ import annotations
from typing import Optional, Tuple
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


__all__ = [
    "cross_entropy",
    "ctc_loss",
    "ctc_best_path",
    "ctc_greedy_decode",
    "ctc_durations_from_path",
    "ctc_no_label_loop_blank_durations_from_path",
    "edit_distance",
]


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
    use_native_op: Optional[bool] = None,
    label_loop: bool = True,
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
    :param use_native_op: whether to use our native op
    :param label_loop:
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
        use_native_op=use_native_op,
        label_loop=label_loop,
    )


def ctc_best_path(
    *,
    logits: Tensor,
    logits_normalized: bool = False,
    targets: Tensor,
    input_spatial_dim: Dim,
    targets_spatial_dim: Dim,
    blank_index: int,
    label_loop: bool = True,
) -> Tensor:
    """
    Calculates the CTC best path.

    :param logits: (before softmax). shape [B...,input_spatial,C]
    :param logits_normalized: whether the logits are already normalized (e.g. via log-softmax)
    :param targets: sparse. shape [B...,targets_spatial] -> C
    :param input_spatial_dim: spatial dim of input logits
    :param targets_spatial_dim: spatial dim of targets
    :param blank_index: vocab index of the blank symbol
    :param label_loop: whether label loops are allowed (standard for CTC). False is like RNA topology.
    :return: best path, shape [B...,targets_spatial] -> C
    """
    # noinspection PyProtectedMember
    return logits._raw_backend.ctc_best_path(
        logits=logits,
        logits_normalized=logits_normalized,
        targets=targets,
        input_spatial_dim=input_spatial_dim,
        targets_spatial_dim=targets_spatial_dim,
        blank_index=blank_index,
        label_loop=label_loop,
    )


def ctc_greedy_decode(
    logits: Tensor,
    *,
    in_spatial_dim: Dim,
    blank_index: int,
    out_spatial_dim: Optional[Dim] = None,
    target_dim: Optional[Dim] = None,
    wb_target_dim: Optional[Dim] = None,
) -> Tuple[Tensor, Dim]:
    """
    Greedy CTC decode.

    :return: (labels, out_spatial_dim)
    """
    if wb_target_dim is None:
        assert logits.feature_dim
        wb_target_dim = logits.feature_dim

    labels = rf.reduce_argmax(logits, axis=wb_target_dim)
    labels = rf.cast(labels, "int32")

    labels_shifted = rf.shift_right(labels, axis=in_spatial_dim, pad_value=blank_index)
    mask_repeat = labels != labels_shifted
    labels, out_spatial_dim = rf.masked_select(
        labels,
        mask=(labels != blank_index) & mask_repeat,
        dims=[in_spatial_dim],
        out_dim=out_spatial_dim,
    )

    if target_dim:
        # Set correct sparse_dim. Only currently implemented if blank comes after.
        assert target_dim.dimension == blank_index
        labels.sparse_dim = target_dim

    return labels, out_spatial_dim


def ctc_durations_from_path(
    *,
    path: Tensor,
    path_spatial_dim: Dim,
    blank_index: int,
    targets_spatial_dim: Optional[Dim] = None,
    out_spatial_dim: Optional[Dim] = None,
    check_dims: bool = True,
    stop_on_failed_check: bool = True,
) -> Tuple[Tensor, Dim]:
    """
    Given a CTC path (alignment), compute the durations of each label + blanks.
    Specifically, assuming that we have N labels in the target sequence,
    there are N labels and N+1 blank durations,
    (one before the first label, one after the last label, and one between each pair of labels),
    resulting in a total of 2N+1 durations.
    The returned durations tensor will have shape [B,...,T'] where T' = 2 * N + 1,
    corresponding to durations for state sequence [blank_0, label_1, blank_1, label_2, ..., label_N, blank_N].

    :param path: CTC path (alignment), shape [B...,path_spatial_dim] -> label indices (including blanks)
    :param path_spatial_dim: spatial dim of path
    :param blank_index: index of the blank label
    :param targets_spatial_dim: if given, asserts that the computed number of labels matches this size
    :param out_spatial_dim: if given, asserts that the output spatial dim size matches 2 * target_spatial_dim + 1
    :param check_dims: whether to check the dimensions sizes
    :param stop_on_failed_check: whether to raise an error on failed check
    :return: (durations, out_spatial_dim).
        durations shape [B...,out_spatial_dim] where out_spatial_dim = 2 * N + 1,
        where N is the number of labels in the target sequence.
    """
    # example path:   [_ _ a a b _ _ c c c _]
    path_shifted = rf.shift_right(path, axis=path_spatial_dim, pad_value=blank_index)
    # path_shifted:   [_ _ _ a a b _ _ c c c]
    new_label_mask = rf.logical_and(path != blank_index, path != path_shifted)
    new_label_mask = new_label_mask.copy_masked(False, dims=[path_spatial_dim])
    num_labels = rf.reduce_sum(rf.cast(new_label_mask, "int32"), axis=path_spatial_dim)
    if targets_spatial_dim is not None:
        if check_dims:
            rf.assert_(
                targets_spatial_dim.get_size_tensor(device=num_labels.device) == num_labels,
                "target_spatial_dim size does not match number of labels in path",
                stop=stop_on_failed_check,
            )
    else:
        targets_spatial_dim = Dim(
            rf.copy_to_device(num_labels, rf.get_default_dim_size_device()), name="target_spatial"
        )
    # new_label_mask: [0 0 1 0 1 0 0 1 0 0 0]
    blank_idx = rf.cumsum(rf.cast(new_label_mask, "int32"), spatial_dim=path_spatial_dim)
    # label_idx = blank_idx - 1
    # label_idx:    [-1 -1 0 0 1 1 1 2 2 2 2]
    # blank_idx:      [0 0 1 1 2 2 2 3 3 3 3]
    blank_idx_x2 = blank_idx * 2
    # blank_idx_x2:   [0 0 2 2 4 4 4 6 6 6 6]
    state_idx = blank_idx_x2 + rf.where(path == blank_index, 0, -1)
    # state_idx:      [0 0 1 1 3 4 4 5 5 5 6]
    if out_spatial_dim is not None:
        if check_dims:
            rf.assert_(
                out_spatial_dim.get_size_tensor(device=num_labels.device) == num_labels * 2 + 1,
                "out_spatial_dim size does not match 2 * target_spatial_dim + 1",
                stop=stop_on_failed_check,
            )
    else:
        out_spatial_dim = targets_spatial_dim * 2 + 1
    out = rf.scatter(rf.ones_like(state_idx), indices=state_idx, indices_dim=path_spatial_dim, out_dim=out_spatial_dim)
    # out state seq: [ _ a _ b _ c _ ]
    # out:           [ 2 2 0 1 2 3 1 ]
    return out, out_spatial_dim


def ctc_no_label_loop_blank_durations_from_path(
    *,
    path: Tensor,
    path_spatial_dim: Dim,
    blank_index: int,
    targets_spatial_dim: Optional[Dim] = None,
    out_spatial_dim: Optional[Dim] = None,
    check_dims: bool = True,
    stop_on_failed_check: bool = True,
) -> Tuple[Tensor, Dim]:
    """
    Given a CTC-without-label-loop (``label_loop=False`` in :func:`ctc_best_path`) (RNA) path (alignment),
    compute the durations of all the blanks.
    Specifically, assuming that we have N labels in the target sequence,
    there are N+1 blank durations
    (one before the first label, one after the last label, and one between each pair of labels).

    :param path: CTC path (alignment), shape [B...,path_spatial_dim] -> label indices (including blanks)
    :param path_spatial_dim: spatial dim of path
    :param blank_index: index of the blank label
    :param targets_spatial_dim: if given, asserts that the computed number of labels matches this size
    :param out_spatial_dim: if given, asserts that the output spatial dim size matches target_spatial_dim + 1
    :param check_dims: whether to check the dimensions sizes
    :param stop_on_failed_check: whether to raise an error on failed check
    :return: (durations, out_spatial_dim),
        durations is for the blank labels,
        durations shape [B...,out_spatial_dim] where out_spatial_dim = N + 1,
        where N is the number of labels in the target sequence.
    """
    # example path:   [_ _ _ a b _ _ c _]
    new_label_mask = path != blank_index
    new_label_mask = new_label_mask.copy_masked(False, dims=[path_spatial_dim])
    num_labels = rf.reduce_sum(rf.cast(new_label_mask, "int32"), axis=path_spatial_dim)
    if targets_spatial_dim is not None:
        if check_dims:
            rf.assert_(
                targets_spatial_dim.get_size_tensor(device=num_labels.device) == num_labels,
                "target_spatial_dim size does not match number of labels in path",
                stop=stop_on_failed_check,
            )
    else:
        targets_spatial_dim = Dim(
            rf.copy_to_device(num_labels, rf.get_default_dim_size_device()), name="target_spatial"
        )
    # new_label_mask: [0 0 0 1 1 0 0 1 0]
    blank_idx = rf.cumsum(rf.cast(new_label_mask, "int32"), spatial_dim=path_spatial_dim)
    # blank_idx:      [0 0 0 1 2 2 2 3 3]
    blank_idx = rf.where(
        (path == blank_index) & rf.sequence_mask(path_spatial_dim, device=path.device),
        blank_idx,
        rf.reduce_max(num_labels, axis=num_labels.dims) + 1,
    )
    # blank_idx:      [0 0 0 4 4 2 2 4 3]
    if out_spatial_dim is not None:
        if check_dims:
            rf.assert_(
                out_spatial_dim.get_size_tensor(device=num_labels.device) == num_labels + 1,
                "out_spatial_dim size does not match 2 * target_spatial_dim + 1",
                stop=stop_on_failed_check,
            )
    else:
        out_spatial_dim = targets_spatial_dim + 1
    out_spatial_dim_ext = out_spatial_dim + 1  # for the extra label index used above
    out = rf.scatter(
        rf.ones_like(blank_idx), indices=blank_idx, indices_dim=path_spatial_dim, out_dim=out_spatial_dim_ext
    )
    out, _ = rf.slice(out, axis=out_spatial_dim_ext, size=out_spatial_dim)
    # out state seq: [ _ a _ b _ c _ ]
    # out:           [ 3   0   2   1 ]
    return out, out_spatial_dim


def edit_distance(a: Tensor, a_spatial_dim: Dim, b: Tensor, b_spatial_dim: Dim, *, dtype: str = "int32") -> Tensor:
    """
    :param a: [B,Ta]
    :param a_spatial_dim: Ta
    :param b: [B,Tb]
    :param b_spatial_dim: Tb
    :param dtype:
    :return: [B]
    """
    # noinspection PyProtectedMember
    backend = a._raw_backend
    if backend.have_edit_distance():
        return backend.edit_distance(a, a_spatial_dim, b, b_spatial_dim)

    from numpy import iinfo

    # The axis permutation is just an efficiency optimization.
    a = a.copy_transpose([a_spatial_dim] + a.remaining_dims(a_spatial_dim))
    b = b.copy_transpose([b_spatial_dim] + b.remaining_dims(b_spatial_dim))
    dev = a.device
    max_dist_err = iinfo(dtype).max
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
