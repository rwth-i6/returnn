"""
Utilities for dimension tags, dimensions, axes.
"""


from __future__ import annotations
from typing import Optional, Union, TypeVar, Sequence, Tuple
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from ._backend import get_backend_by_tensor, global_backend

T = TypeVar("T")

__all__ = ["range_over_dim", "replace_dim", "dim_match_priority_when_needed", "num_elements_of_shape"]


def range_over_dim(dim: Dim, *, dtype: Optional[str] = None) -> Tensor[T]:
    """
    :param dim:
    :param dtype:
    :return: tensor with shape [dim]
    """
    if dim.dyn_size_ext:
        backend = get_backend_by_tensor(dim.dyn_size_ext, fallback=global_backend)
    else:
        backend = global_backend
    return backend.range_over_dim(dim, dtype=dtype)


def replace_dim(source: Tensor, *, in_dim: Dim, out_dim: Optional[Dim] = None) -> Tuple[Tensor, Dim]:
    """
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


def num_elements_of_shape(dims: Sequence[Dim]) -> Union[int, Tensor]:
    """
    :param dims:
    :return: num elements of a tensor of shape dims, properly considering masking
    """
    if all(dim.is_static() for dim in dims):
        n = 1
        for dim in dims:
            n *= dim.dimension
        return n

    n = 1
    dims = list(dims)
    dims.sort(key=lambda dim: -dim.dyn_size_ext.batch_ndim if dim.dyn_size_ext else 0)
    while dims:
        dim = dims.pop(0)
        if dim.is_static():
            n *= dim.dimension
            continue
        # E.g. dyn_size_ext is shape [B], and self has shape [B,T].
        # Due to the sorting of dims above, dims will be [T,B], and we will first process T.
        # We want to sum over dyn_size_ext, but then we need to remove the other dims it covers.
        for dim_ in dim.dyn_size_ext.dims:
            assert dim_ in dims  # num elements not really well-defined then
            assert not dim_.need_masking()  # not implemented
            dims.remove(dim_)
        n_ = rf.reduce_sum(dim.dyn_size_ext, axis=dim.dyn_size_ext.dims)
        n *= n_
    return n
