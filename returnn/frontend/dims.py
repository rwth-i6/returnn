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
    "replace_dim",
    "dim_match_priority_when_needed",
    "num_elements_of_shape",
    "masked_fraction_of_shape",
]


def range_over_dim(dim: Dim, *, dtype: Optional[str] = None, device: Optional[str] = None) -> Tensor[T]:
    """
    :param dim:
    :param dtype:
    :param device,
    :return: tensor with shape [dim]
    """
    if dim.dyn_size_ext:
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


def replace_dim(source: Tensor, *, in_dim: Dim, out_dim: Optional[Dim] = None) -> Tuple[Tensor, Dim]:
    """
    Also see: :func:`rf.merge_dims`, :func:`rf.split_dims`.

    :param source:
    :param in_dim:
    :param out_dim:
    :return: source with in_dim replaced by out_dim, and new out_dim.
        this does not work for the sparse_dim. see :func:`set_sparse_dim` for that case.
    """
    if not out_dim:
        out_dim = in_dim.copy(same_as_self=False, description="new-dim")
    # noinspection PyProtectedMember
    return source._raw_backend.replace_dim(source, in_dim=in_dim, out_dim=out_dim), out_dim


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


def num_elements_of_shape(dims: Union[Dim, Sequence[Dim]], *, use_mask: bool = True) -> Union[int, Tensor]:
    """
    :param dims:
    :param use_mask:
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
            n *= dim.dimension
        return n

    n: Union[int, Tensor] = 1
    postponed_dims = []
    for i, dim in enumerate(dims):
        # E.g. if dim==B, and some other dim dyn_size_ext has B, then we need to postpone this.
        related_dims = []
        for j, dim_ in enumerate(dims):
            if i == j:
                continue
            if dim_.dyn_size_ext and dim in dim_.dyn_size_ext.dims:
                related_dims.append(dim_)
        if not related_dims:
            if dim.is_static():
                n *= dim.dimension
            else:
                n *= dim.dyn_size_ext
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
