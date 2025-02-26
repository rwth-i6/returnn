"""
Utilities for dimension tags, dimensions, axes.
"""

from __future__ import annotations
from typing import Optional, Union, TypeVar, Sequence, Tuple
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from ._backend import get_backend_by_tensor, global_backend

T = TypeVar("T")

__all__ = [
    "range_over_dim",
    "range_over_dim_strided",
    "range_over_merged_dims",
    "linspace_over_dim",
    "replace_dim",
    "replace_dim_v2",
    "set_sparse_dim",
    "dim_match_priority_when_needed",
    "num_elements_of_shape",
    "masked_fraction_of_shape",
    "last_frame_position_of_dim",
    "use_mask_default",
]


def range_over_dim(dim: Dim, *, dtype: Optional[str] = None, device: Optional[str] = None) -> Tensor[T]:
    """
    :param dim:
    :param dtype:
    :param device,
    :return: tensor with shape [dim]
    """
    if dim.dyn_size_ext is not None:
        backend = get_backend_by_tensor(dim.dyn_size_ext, fallback=global_backend)
    else:
        backend = global_backend
    return backend.range_over_dim(dim, dtype=dtype, device=device)


def range_over_dim_strided(
    dim: Dim,
    *,
    stride: Union[int, Tensor],
    out_dim: Optional[Dim] = None,
    dtype: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[Tensor[T], Dim]:
    """
    :param dim:
    :param stride:
    :param out_dim:
    :param dtype:
    :param device,
    :return: tensor with shape [dim], out_dim
    """
    if out_dim is None:
        out_dim = dim.ceildiv_right(stride)
    return rf.range_over_dim(out_dim, dtype=dtype, device=device) * stride, out_dim


def range_over_merged_dims(
    dims: Sequence[Dim], *, dtype: Optional[str] = None, device: Optional[str] = None
) -> Tensor[T]:
    """
    This is if you want to index into a merged dim.
    Related: :func:`rf.merge_dims`.

    :param dims:
    :param dtype:
    :param device:
    :return: tensor with shape [dim_0, ..., dim_n] -> sparse_dim = merged_dim, where merged_dim = dim_0 * ... * dim_n
    """
    assert len(dims) >= 1
    merged_dim = dims[0]
    for dim in dims[1:]:
        merged_dim *= dim
    indices = rf.range_over_dim(merged_dim, dtype=dtype, device=device)
    if len(dims) > 1:
        indices = rf.split_dims(indices, axis=merged_dim, dims=dims)
    return indices


def linspace_over_dim(
    dim: Dim,
    start: Union[float, Tensor] = 0.0,
    end: Union[float, Tensor] = 1.0,
    *,
    dtype: Optional[str] = None,
    device: Optional[str] = None,
) -> Tensor:
    """
    Linearly spaced values over a dim.

    :param dim: dim to range over
    :param start: start value
    :param end: end value
    :param dtype: dtype of the output tensor
    :param device: device of the output tensor
    :return: tensor with shape [dim] containing linearly spaced values between start and end
    """
    if dtype is None:
        dtype = rf.get_default_float_dtype()
    indices = rf.range_over_dim(dim, dtype=dtype, device=device)
    linspace = indices / rf.cast(rf.maximum(dim.get_size_tensor(device=indices.device), 1), dtype=indices.dtype)
    space_len = end - start
    if not isinstance(space_len, (int, float)) or space_len != 1:
        linspace *= space_len
    if not isinstance(start, (int, float)) or start != 0:
        linspace += start
    return linspace


def replace_dim(source: Tensor, *, in_dim: Dim, out_dim: Optional[Dim] = None) -> Tuple[Tensor, Dim]:
    """
    Also see: :func:`replace_dim_v2`, :func:`rf.merge_dims`, :func:`rf.split_dims`.

    :param source:
    :param in_dim: should be in ``source.dims``, to be replaced.
        If you want to replace the ``source.sparse_dim``, see :func:`set_sparse_dim`.
    :param out_dim: If not given, will create a new dim with the same size as ``in_dim``.
        Note: If the size of ``out_dim`` is different from ``in_dim``,
        currently the dim tag is replaced and there is no error -- this is not checked.
    :return: ``source`` with ``in_dim`` replaced by ``out_dim``, and ``out_dim``.
    """
    if not out_dim:
        out_dim = in_dim.copy(same_as_self=False, description="new-dim")
    # noinspection PyProtectedMember
    return source._raw_backend.replace_dim(source, in_dim=in_dim, out_dim=out_dim), out_dim


def replace_dim_v2(
    source: Tensor, *, in_dim: Dim, out_dim: Dim, allow_expand: bool = True, allow_shrink: bool = True
) -> Tensor:
    """
    Extends :func:`replace_dim` by also allowing to expand or shrink the dim
    (or rather, to not ignore this; when :func:`replace_dim` is used on a dim with different size,
     it will ignore this and anyway accept the new dim tag (currently)).

    :param source:
    :param in_dim: should be in ``source.dims``, to be replaced.
        If you want to replace the ``source.sparse_dim``, see :func:`set_sparse_dim`.
    :param out_dim: should not be in ``source.dims``, to be replaced.
        Note: In contrast to :func:`replace_dim`, you must provide this explicitly.
    :param allow_expand: if True, allow to expand the dim, i.e. if ``out_dim.size > in_dim.size``.
    :param allow_shrink: if True, allow to shrink the dim, i.e. if ``out_dim.size < in_dim.size``.
    :return: ``source`` with ``in_dim`` replaced by ``out_dim``.
    """
    if not rf.is_executing_eagerly():
        raise NotImplementedError  # just not implemented yet. we can do via :func:`cond`
    if in_dim not in source.dims:
        raise ValueError(f"replace_dim_v2: dim {in_dim} not in {source}")
    if out_dim in source.dims:
        raise ValueError(f"replace_dim_v2: dim {out_dim} already in {source}")
    old_size = in_dim.get_dim_value()
    new_size = out_dim.get_dim_value()
    if new_size == old_size:
        res, _ = rf.replace_dim(source, in_dim=in_dim, out_dim=out_dim)
    elif new_size > old_size:
        if not allow_expand:
            raise ValueError(
                f"replace_dim_v2: not allowed to expand: {old_size} -> {new_size},"
                f" for {in_dim=} {out_dim=}, in {source=}"
            )
        res, _ = rf.pad(
            source,
            axes=[in_dim],
            padding=[(0, out_dim.get_dim_value_tensor() - in_dim.get_dim_value_tensor())],
            out_dims=[out_dim],
            value=0,
        )
    else:
        if not allow_shrink:
            raise ValueError(
                f"replace_dim_v2: not allowed to shrink: {old_size} -> {new_size},"
                f" for {in_dim=} {out_dim=}, in {source=}"
            )
        res, _ = rf.slice(source, axis=in_dim, size=out_dim)
    return res


def set_sparse_dim(source: Tensor, sparse_dim: Dim) -> Tensor:
    """
    :param source:
    :param sparse_dim:
    :return: source with sparse_dim set
    """
    # noinspection PyProtectedMember
    return source._raw_backend.set_sparse_dim(source, sparse_dim)


def dim_match_priority_when_needed(dim: Dim, *other_dims: Dim) -> Dim:
    """
    :return: maybe copy of dim with higher match_priority if needed to distinguish from other_dims

    Why or when is this needed?

    For activation values, this should never be needed,
    and all dims should be unique.

    In case of self-attention, the standard way is to create a separate distinct dim
    to perform the attention reduction over.
    See :class:`SelfAttention`.

    However, in case of weight matrices, it is not unusual to have the same dim for both the input and output,
    so a square weight matrix.
    When reduction is performed in :func:`matmul`, we want to match the input feature dim
    to the dim in the weight matrix with higher match priority.

    So :func:`dim_match_priority_when_needed` would be applied on the input feature dim.

    https://github.com/rwth-i6/returnn/pull/871
    https://github.com/rwth-i6/returnn_common/issues/17#issuecomment-997463222
    """
    if dim in other_dims:
        return dim.copy(match_priority=1)
    return dim


def num_elements_of_shape(
    dims: Union[Dim, Sequence[Dim]], *, use_mask: bool = True, device: Optional[str] = None
) -> Union[int, Tensor]:
    """
    :param dims:
    :param use_mask:
    :param device: only for the case when we return a Tensor. by default, this is CPU (just as the size tensor).
    :return: num elements of a tensor of shape dims, properly considering masking
    """
    if isinstance(dims, Dim):
        dims = [dims]
    if not use_mask:
        n = 1
        for dim in dims:
            n *= dim.get_dim_value_tensor()
        return n
    if all(dim.is_static() for dim in dims):
        n = 1
        for dim in dims:
            n *= dim.size
        return n
    if len(dims) == 1:
        return dims[0].get_size_tensor(device=device)

    n: Union[int, Tensor] = 1
    postponed_dims = []
    for i, dim in enumerate(dims):
        # E.g. if dim==B, and some other dim dyn_size_ext has B, then we need to postpone this.
        related_dims = []
        for j, dim_ in enumerate(dims):
            if i == j:
                continue
            if dim_.dyn_size_ext is not None and dim in dim_.dyn_size_ext.dims:
                related_dims.append(dim_)
        if not related_dims:
            if dim.is_static():
                n *= dim.size
            else:
                n *= dim.get_size_tensor(device=device)
        else:
            postponed_dims.append(dim)
    if postponed_dims:
        n: Tensor
        n = rf.reduce_sum(n, axis=postponed_dims)
    return n


def masked_fraction_of_shape(dims: Union[Dim, Sequence[Dim]], *, inverse: bool = False) -> Union[int, float, Tensor]:
    """
    :param dims:
    :param inverse: if True, return the inverse of the fraction
    :return: :func:`num_elements_of_shape`(dims) / prod(dims) if not inverse else prod(dims) / num_elements
    """
    if isinstance(dims, Dim):
        dims = [dims]
    if not any(dim.need_masking() for dim in dims):
        return 1
    num_elems_masked = num_elements_of_shape(dims)
    num_elems_total = 1
    for dim in dims:
        num_elems_total *= dim.get_dim_value_tensor()
    return (num_elems_masked / num_elems_total) if not inverse else (num_elems_total / num_elems_masked)


def last_frame_position_of_dim(
    dim: Dim, *, device: Optional[str] = None, allow_scalar_on_cpu: bool = True
) -> Union[int, Tensor]:
    """
    :param dim:
    :param device:
    :param allow_scalar_on_cpu: if device is not given, we would use rf.get_default_device() for dynamic sizes.
        If this is True, we would allow to return a scalar dynamic size on CPU.
    :return: last frame position of dim
    """
    if dim.size is not None:
        if dim.size > 0:
            return dim.size - 1
        return 0
    if device is None:
        if allow_scalar_on_cpu and not dim.dyn_size_ext.dims:
            device = "cpu"
        else:
            device = rf.get_default_device()
    pos = dim.get_dyn_size_ext_for_device(device) - 1
    pos = rf.maximum(pos, 0)
    pos.sparse_dim = dim
    return pos


def use_mask_default(
    *, default: Optional[bool] = None, default_false_for_behavior_version_up_to: Optional[int] = None
) -> Optional[bool]:
    """
    Check the global RETURNN config for the ``rf_use_mask``
    on what default we should use for the ``use_mask`` argument in various functions
    (e.g. :func:`conv`, :func:`pool`, :func:`reduce`, :func:`matmul`, ...).

    See issue `#1691 <https://github.com/rwth-i6/returnn/issues/1691>`__.

    :param default: what to return if it is not defined in the config,
        and ``default_false_for_behavior_version_up_to`` does not apply.
    :param default_false_for_behavior_version_up_to: if it is not defined in the config,
        and if this is set, and the behavior version is less or equal,
        then return False by default, i.e. do not use the mask by default, if it is not defined in the config.
        This takes precedence over `default`.
    :return: what to use for the ``use_mask`` argument by default
    """
    from returnn.config import get_global_config

    config = get_global_config(raise_exception=False)
    config_value = None
    if config:
        if "rf_use_mask" in config.typed_dict:
            config_value = config.typed_dict["rf_use_mask"]
            assert config_value is None or isinstance(config_value, bool)
        elif "rf_use_mask" in config.dict:
            config_value = config.bool("rf_use_mask", None)
    if config_value is not None:
        return config_value

    if default_false_for_behavior_version_up_to is not None:
        from returnn.util.basic import BehaviorVersion

        if BehaviorVersion.get() <= default_false_for_behavior_version_up_to:
            return False
    return default
