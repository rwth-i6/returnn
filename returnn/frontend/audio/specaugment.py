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
    steps: Tuple[int, int, int] = (0, 1000, 2000),
) -> Tensor:
    """
    SpecAugment, https://arxiv.org/abs/1904.08779
    """
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
        # static int if the dim has a declared capacity (then the num range is static too),
        # else the dynamic max (tensor)
        spatial_len = spatial_dim.get_dim_value_tensor()
        spatial_num_bound = None
        feature_num_bound = None
        if rf.is_static_traceable():
            # static traceable (see :func:`rf.is_static_traceable`) needs a static num-masks bound,
            # and thus the declared capacity.
            # The bounds span the WHOLE step schedule
            # (max step factors: spatial 1+1+2*1 = 4, feature 2*1+1+2*1 = 5),
            # so one trace / captured graph covers all schedule phases.
            assert isinstance(spatial_len, int), (
                "specaugment: static traceable (rf.is_static_traceable) requires spatial_dim capacity"
            )
            spatial_num_bound = min(max(spatial_len // num_spatial_mask_factor, 2) * 4, spatial_len)
            feature_num_bound = 5
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
    # indices should be sorted, and of shape (batch,num), entries (int32) in [0,dim)
    if isinstance(num, int) or num_bound is not None:
        # fixed trip count; with tensor num, iterations beyond a seq's num are gated off
        # (same mask distribution)
        for i in range(num if isinstance(num, int) else num_bound):
            y = mask(
                x,
                mask_axis=mask_axis,
                pos=rf.gather(indices, axis=k_dim, indices=i),
                max_amount=max_dims,
                mask_value=mask_value,
            )
            x = y if isinstance(num, int) else rf.where(rf.greater(num, i), y, x)
    else:

        def _body(s):
            i_, x_ = s
            y = mask(
                x_,
                mask_axis=mask_axis,
                pos=rf.gather(indices, axis=k_dim, indices=rf.copy_to_device(i_, indices.device)),
                max_amount=max_dims,
                mask_value=mask_value,
            )
            y = rf.where(rf.copy_to_device(rf.less(i_, num), y.device), y, x_)
            return i_ + 1, y

        _, x = rf.while_loop(
            cond=lambda s: s[0] < max_num,
            body=_body,
            initial=(rf.constant(0, dims=(), device="cpu"), x),
        )
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
