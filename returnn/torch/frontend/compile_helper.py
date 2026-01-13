"""
Helpers to improve torch.compile on RF code.
"""

from __future__ import annotations
from typing import Any, Iterable, List, Tuple

from returnn.tensor import Tensor, Dim

# noinspection PyProtectedMember
from returnn.frontend import _native

_is_set_up = False


def setup():
    """
    Set up the torch.compile helpers for RF code, also including :class:`Tensor` and :class:`Dim`.
    """

    global _is_set_up
    if _is_set_up:
        return
    _is_set_up = True  # only try once

    assert not _native.is_set_up(), "Call this setup() as early as possible."
    _native.set_enabled(False)

    # noinspection PyProtectedMember
    from torch.utils._pytree import register_pytree_node

    register_pytree_node(Tensor, _tensor_flatten, _tensor_unflatten)
    register_pytree_node(Dim, _dim_flatten, _dim_unflatten)


def _tensor_flatten(t: Tensor) -> Tuple[List[Any], Any]:
    """
    Flatten the tensor for PyTree.
    """
    return [t.raw_tensor, t.dims, t.sparse_dim], [
        t.name,
        t.dtype,
        t.version,
        t.feature_dim_axis_or_unspecified,
        t.time_dim_axis_or_unspecified,
    ]


def _tensor_unflatten(values: Iterable[Any], metadata: Any) -> Tensor:
    """
    Unflatten the tensor from PyTree.
    """
    raw_tensor, dims, sparse_dim = values
    name, dtype, version, feature_dim_axis, time_dim_axis = metadata
    return Tensor(
        name=name,
        dims=dims,
        dtype=dtype,
        sparse_dim=sparse_dim,
        feature_dim_axis=feature_dim_axis,
        time_dim_axis=time_dim_axis,
        raw_tensor=raw_tensor,
        version=version,
    )


def _dim_flatten(d: Dim) -> Tuple[List[Any], Any]:
    """
    Flatten the dim for PyTree.
    """
    return [d.dyn_size_ext], [d.name, d.dimension, d.size]


def _dim_unflatten(values: Iterable[Any], metadata: Any) -> Dim:
    """
    Unflatten the dim from PyTree.
    """
    (dyn_size_ext,) = values
    name, dimension, size = metadata
    # TODO this creates a new instance... this is maybe wrong?
    return Dim(name=name, dimension=dimension, size=size, dyn_size_ext=dyn_size_ext)
