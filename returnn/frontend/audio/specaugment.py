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
    max_consecutive_spatial_dims: int = 20,
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
        with rf.set_default_device_ctx("cpu"):
            step = rf.get_run_ctx().step
            step0 = rf.where(step >= steps[0], 1, 0)
            step1 = rf.where(step >= steps[1], 1, 0)
            step2 = rf.where(step >= steps[2], 1, 0)
    else:
        step0 = step1 = step2 = 1

    def _mask_branch():
        x_masked = x
        spatial_len = spatial_dim.get_dim_value_tensor()
        # time mask
        if max_consecutive_spatial_dims > 0 and num_spatial_mask_factor > 0:
            x_masked = random_mask(
                x_masked,
                mask_axis=spatial_dim,
                broadcast_axis=feature_dim,
                min_num=rf.minimum(step1 + step2, spatial_len),
                max_num=rf.minimum(
                    rf.maximum(spatial_len // num_spatial_mask_factor, 2) * (step0 + step1 + step2 * 2), spatial_len
                ),
                max_dims=max_consecutive_spatial_dims,
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
    """
    batch_dims = list(x.dims)
    batch_dims.remove(mask_axis)
    if isinstance(broadcast_axis, Dim):
        batch_dims.remove(broadcast_axis)
    else:
        for a in broadcast_axis:
            batch_dims.remove(a)
    if isinstance(min_num, int) and isinstance(max_num, int) and min_num == max_num:
        num = min_num
        max_num = num
    else:
        num = rf.random_uniform(batch_dims, minval=min_num, maxval=max_num + 1, dtype="int32", device="cpu")
        max_num = rf.reduce_max(num, axis=num.dims)
    _, indices, k_dim = rf.top_k(
        rf.random_uniform(batch_dims + [mask_axis], minval=0.0, maxval=1.0, device=x.device),
        axis=mask_axis,
        k=num if isinstance(num, int) else rf.reduce_max(num, axis=num.dims),
    )
    # indices should be sorted, and of shape (batch,num), entries (int32) in [0,dim)
    if isinstance(num, int):
        for i in range(num):
            x = mask(
                x,
                mask_axis=mask_axis,
                pos=rf.gather(indices, axis=k_dim, indices=i),
                max_amount=max_dims,
                mask_value=mask_value,
            )
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
    dim = mask_axis.get_size_tensor()
    dim = rf.copy_to_device(dim, pos.device)
    pos = rf.cast(pos, dtype=dim.dtype)
    amount = rf.random_uniform(pos.dims, minval=1, maxval=max_amount + 1, dtype=pos.dtype, device=pos.device)
    pos2 = rf.minimum(pos + amount, dim)
    idxs = rf.range_over_dim(mask_axis, dtype=pos.dtype, device=pos.device)  # (dim,)
    cond = rf.compare_bc(idxs, ">=", pos) & rf.compare_bc(idxs, "<", pos2)  # (batch,dim)
    x = rf.where(cond, mask_value, x)
    return x
