"""
SpecAugment, https://arxiv.org/abs/1904.08779
"""

from __future__ import annotations
from typing import Optional, Union, Collection, Tuple
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


__all__ = ["specaugment", "random_mask", "mask"]


def specaugment(
    x: Tensor,
    *,
    spatial_dim: Dim,
    feature_dim: Optional[Dim] = None,
    global_train_step_dependent: bool = True,
    only_on_train: bool = True,
    max_consecutive_spatial_dims: Union[int, Tensor] = 20,
    max_consecutive_feature_dims: Optional[int] = None,
    num_spatial_mask_factor: int = 100,
    num_masks_per_seq: Optional[bool] = None,
    steps: Tuple[int, int, int] = (0, 1000, 2000),
) -> Tensor:
    """
    SpecAugment, https://arxiv.org/abs/1904.08779

    :param x:
    :param spatial_dim:
    :param feature_dim:
    :param global_train_step_dependent:
    :param only_on_train:
    :param max_consecutive_spatial_dims:
    :param max_consecutive_feature_dims:
    :param num_spatial_mask_factor:
    :param num_masks_per_seq: the spatial num-masks range follows each seq's OWN length
        instead of the batch max, i.e. the augmentation of a seq no longer depends on
        what else is in the batch.
        Default (None): global config option ``rf_specaugment_num_masks_per_seq`` if set,
        else behavior version >= 28.
    :param steps:
    """
    if num_masks_per_seq is None:
        num_masks_per_seq = _should_use_num_masks_per_seq()
    if feature_dim is None:
        assert x.feature_dim
        feature_dim = x.feature_dim
    if max_consecutive_feature_dims is None:
        max_consecutive_feature_dims = feature_dim.dimension // 5
    if global_train_step_dependent:
        step = rf.get_run_ctx().step
        if isinstance(step, Tensor) and step.raw_tensor is not None and step.device not in (None, "cpu"):
            # device-resident step (e.g. under CUDA-graph capture, updated in place by the engine):
            # keep the schedule computation on its device, so it stays in-graph
            # and one captured graph is valid across the `steps` boundaries
            step_device = step.device
        else:
            step_device = "cpu"
        with rf.set_default_device_ctx(step_device):
            step0 = rf.where(step >= steps[0], 1, 0)
            step1 = rf.where(step >= steps[1], 1, 0)
            step2 = rf.where(step >= steps[2], 1, 0)
    else:
        step0 = step1 = step2 = 1

    def _mask_branch():
        x_masked = x
        spatial_num_bound = None
        feature_num_bound = None
        if rf.is_static_traceable():
            # static traceable (see :func:`rf.is_static_traceable`) needs a static num-masks bound,
            # and thus the declared capacity.
            # The bounds span the WHOLE step schedule
            # (max step factors: spatial 1+1+2*1 = 4, feature 2*1+1+2*1 = 5),
            # so one trace / captured graph covers all schedule phases.
            # static int if the dim has a declared capacity (then the num range is static too),
            # else the dynamic max (tensor)
            spatial_len_max = spatial_dim.get_dim_value_tensor()
            assert isinstance(spatial_len_max, int), (
                "specaugment: static traceable (rf.is_static_traceable) requires spatial_dim capacity"
            )
            spatial_num_bound = min(max(spatial_len_max // num_spatial_mask_factor, 2) * 4, spatial_len_max)
            feature_num_bound = 5
            # The capacity is ONLY the bound (the static trip count).
            # The num-masks RANGE must follow the true length:
            # with the capacity also as the range,
            # the number of masks scales with the declared capacity instead of the data
            # (all mask positions land inside the seq -- top_k excludes padded positions),
            # i.e. raising the capacity silently over-masks and degrades training.
            sizes = spatial_dim.get_dyn_size_ext_for_device(x.device)
            spatial_len = sizes if num_masks_per_seq else rf.reduce_max(sizes, axis=sizes.dims)
        else:  # not static tracing
            if num_masks_per_seq:
                # per-seq range in eager: the lens (on cpu, like the eager num draws in random_mask)
                spatial_len = spatial_dim.get_dyn_size_ext_for_device(None)
            else:
                spatial_len = spatial_dim.get_dim_value_tensor()
        # time mask
        if num_spatial_mask_factor > 0 and (
            isinstance(max_consecutive_spatial_dims, Tensor) or max_consecutive_spatial_dims > 0
        ):
            x_masked = random_mask(
                x_masked,
                mask_axis=spatial_dim,
                broadcast_axis=feature_dim,
                min_num=rf.minimum(step1 + step2, spatial_len),
                max_num=rf.minimum(
                    rf.maximum(spatial_len // num_spatial_mask_factor, 2) * (step0 + step1 + step2 * 2), spatial_len
                ),
                max_dims=max_consecutive_spatial_dims,
                max_num_bound=spatial_num_bound,
            )
        # feature mask
        if max_consecutive_feature_dims > 0:
            x_masked = random_mask(
                x_masked,
                mask_axis=feature_dim,
                broadcast_axis=spatial_dim,
                min_num=step1 + step2,
                max_num=step0 * 2 + step1 + step2 * 2,
                max_dims=max_consecutive_feature_dims,
                max_num_bound=feature_num_bound,
            )
        return x_masked

    return rf.cond(
        rf.get_run_ctx().is_train_flag_enabled(func=specaugment) | (not only_on_train), _mask_branch, lambda: x
    )


def random_mask(
    x: Tensor,
    *,
    mask_axis: Dim,
    broadcast_axis: Union[Dim, Collection[Dim]],
    min_num: Union[int, Tensor],
    max_num: Union[int, Tensor],
    max_dims: Union[int, Tensor],
    mask_value: Union[int, float, Tensor] = 0.0,
    max_num_bound: Optional[int] = None,
) -> Tensor:
    """
    :param x: (batch,time,feature)
    :param mask_axis: axis to mask
    :param broadcast_axis: one or multiple, which should be broadcasted over.
      The remaining axes not specified by mask_axis and broadcast_axis are not broadcasted over
      and treated as batch dims.
      E.g. in [B,T,D], with mask_axis=F, broadcast_axis=T, it creates masks [B,F].
    :param min_num:
    :param max_num: inclusive
    :param max_dims: inclusive
    :param mask_value:
    :param max_num_bound: static upper bound for max_num.
        Required for the static-traceable path (see :func:`rf.is_static_traceable`) when max_num is a Tensor
        (for int max_num, max_num itself is the bound).
        With static traceable, the mask loop runs a fixed number of iterations (the bound),
        gated per sequence (i < num), instead of a data-dependent while loop,
        and num is drawn on the data's device instead of cpu
        (same mask distribution, different RNG consumption).
    """
    batch_dims = list(x.dims)
    batch_dims.remove(mask_axis)
    if isinstance(broadcast_axis, Dim):
        batch_dims.remove(broadcast_axis)
    else:
        for a in broadcast_axis:
            batch_dims.remove(a)
    num_bound: Optional[int] = None
    if isinstance(min_num, int) and isinstance(max_num, int) and min_num == max_num:
        num = min_num
        max_num = num
    else:
        if rf.is_static_traceable():
            # Static traceable (see :func:`rf.is_static_traceable`):
            # num drawn on the data's device (tensor bounds are handled device-side by the backend),
            # and below a fixed trip count (loop to the bound, gate per seq)
            # instead of the data-dependent while loop.
            if isinstance(max_num, int):
                num_bound = max_num
            else:
                assert max_num_bound is not None, (
                    "random_mask: static traceable (rf.is_static_traceable) requires max_num_bound for tensor max_num"
                )
                num_bound = max_num_bound
            # k for top_k must not exceed the axis (get_dim_value: the capacity / static size, a host int)
            num_bound = min(num_bound, mask_axis.get_dim_value())
            num_device = x.device
        else:
            num_device = "cpu"
        num = rf.random_uniform(batch_dims, minval=min_num, maxval=max_num + 1, dtype="int32", device=num_device)
        if num_bound is None:
            max_num = rf.reduce_max(num, axis=num.dims)
    _, indices, k_dim = rf.top_k(
        rf.random_uniform(batch_dims + [mask_axis], minval=0.0, maxval=1.0, device=x.device),
        axis=mask_axis,
        k=num if isinstance(num, int) else (num_bound if num_bound is not None else max_num),
    )
    # indices should be sorted, and of shape (batch,num), entries (int32) in [0,dim).
    # Apply ALL masks in one fused pass
    # (one amount draw, one broadcast compare over (batch,k,dim), one any-reduce, one where)
    # instead of a per-mask loop -- k-proportional kernels fewer, same mask distribution.
    # With tensor num, mask slots beyond a seq's num are gated off.
    dim = mask_axis.get_size_tensor_or_int(device=indices.device)
    pos = rf.cast(indices, dtype=dim.dtype if isinstance(dim, Tensor) else rf.get_default_array_index_dtype())
    amount = rf.random_uniform(pos.dims, minval=1, maxval=max_dims + 1, dtype=pos.dtype, device=pos.device)
    pos2 = rf.minimum(pos + amount, dim)
    idxs = rf.range_over_dim(mask_axis, dtype=pos.dtype, device=pos.device)  # (dim,)
    cond = rf.compare_bc(idxs, ">=", pos) & rf.compare_bc(idxs, "<", pos2)  # (batch,k,dim)
    if isinstance(num, Tensor):
        num = rf.copy_to_device(num, x.device)
        cond = cond & rf.compare_bc(rf.range_over_dim(k_dim, device=num.device), "<", num)
    cond = rf.reduce_any(cond, axis=k_dim)
    x = rf.where(cond, mask_value, x)
    return x


def mask(
    x: Tensor,
    *,
    mask_axis: Dim,
    pos: Tensor,
    max_amount: Union[int, Tensor],
    mask_value: Union[int, float, Tensor] = 0.0,
) -> Tensor:
    """
    :param x: (batch,time,[feature]). any dim not mask_axis or in pos.shape will be broadcasted over
    :param mask_axis:
    :param pos: (batch,) (or multiple batch dims)
    :param max_amount: inclusive
    :param mask_value:
    """
    dim = mask_axis.get_size_tensor_or_int(device=pos.device)
    pos = rf.cast(pos, dtype=dim.dtype if isinstance(dim, Tensor) else rf.get_default_array_index_dtype())
    amount = rf.random_uniform(pos.dims, minval=1, maxval=max_amount + 1, dtype=pos.dtype, device=pos.device)
    pos2 = rf.minimum(pos + amount, dim)
    idxs = rf.range_over_dim(mask_axis, dtype=pos.dtype, device=pos.device)  # (dim,)
    cond = rf.compare_bc(idxs, ">=", pos) & rf.compare_bc(idxs, "<", pos2)  # (batch,dim)
    x = rf.where(cond, mask_value, x)
    return x


def _should_use_num_masks_per_seq() -> bool:
    """
    :return: default for the ``num_masks_per_seq`` option of :func:`specaugment`.

    Check the global RETURNN config for the ``rf_specaugment_num_masks_per_seq`` option.
    If that is not specified, with behavior version >= 28,
    the num-masks range follows each seq's own length,
    with behavior version <= 27 the batch max.
    """
    from returnn.config import get_global_config
    from returnn.util.basic import BehaviorVersion

    config = get_global_config(raise_exception=False)
    config_value = None
    if config:
        if "rf_specaugment_num_masks_per_seq" in config.typed_dict:
            config_value = config.typed_dict["rf_specaugment_num_masks_per_seq"]
            assert config_value is None or isinstance(config_value, bool)
        elif "rf_specaugment_num_masks_per_seq" in config.dict:
            config_value = config.bool("rf_specaugment_num_masks_per_seq", None)
    if config_value is not None:
        return config_value
    return BehaviorVersion.get() >= 28
