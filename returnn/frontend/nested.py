"""
Some utility functions on nested structures.
"""

from __future__ import annotations
from typing import TypeVar, Optional, Union, Sequence, Tuple, Dict
import functools
import re
import tree
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


__all__ = ["mask_nested", "gather_nested", "masked_select_nested", "masked_scatter_nested"]


T = TypeVar("T")


def mask_nested(
    s: T,
    *,
    mask: Tensor,
    mask_cpu: Optional[Tensor] = None,
    mask_value: Union[T, Tensor, float, None],
    dim_map: Optional[Dict[Dim, Dim]] = None,
    allow_dim_extension: bool = True,
) -> T:
    """
    Applies where(mask, s, mask_value) for nested structures.

    :param s:
    :param mask:
    :param mask_cpu: mask tensor for CPU. this is used e.g. for dyn dim sizes
    :param mask_value:
    :param dim_map:
    :param allow_dim_extension:
    :return: s with masked values
    """
    if dim_map is None:
        dim_map = {}
    partial_kwargs = dict(mask=mask, mask_cpu=mask_cpu, dim_map=dim_map, allow_dim_extension=allow_dim_extension)
    structures = [s]
    if type(s) is type(mask_value):  # mask_value also same nested structure?
        tree.assert_same_structure(s, mask_value)
        structures.append(mask_value)
    else:
        partial_kwargs["mask_value"] = mask_value
    tree.map_structure(functools.partial(_mask_prepare_dims, **partial_kwargs), *structures)
    return tree.map_structure(functools.partial(_mask, **partial_kwargs), *structures)


def _mask_prepare_dims(
    s: T,
    mask_value: Union[T, Tensor, float, None],
    *,
    mask: Tensor,
    mask_cpu: Optional[Tensor] = None,
    dim_map: Dict[Dim, Dim],
    allow_dim_extension: bool,
) -> T:
    if isinstance(s, Dim):
        if mask_value is None:
            return s  # not sure if always correct...
        assert isinstance(mask_value, Dim)
        if s == mask_value:
            return s
        if not allow_dim_extension:
            dim_size_dims = set()
            if s.dyn_size_ext is not None:
                dim_size_dims.update(s.dyn_size_ext.dims_set)
            if mask_value.dyn_size_ext is not None:
                dim_size_dims.update(mask_value.dyn_size_ext.dims_set)
            if not mask.dims_set.issubset(dim_size_dims):
                assert not mask.dims_set.intersection(dim_size_dims)  # not sure...
                return s
        new_dyn_size = _mask(
            s.get_size_tensor(),
            mask=mask,
            mask_cpu=mask_cpu,
            mask_value=mask_value.get_size_tensor(),
            dim_map=dim_map,
            allow_dim_extension=allow_dim_extension,
        )
        new_dim = Dim(new_dyn_size, name=_extend_dim_name(s.name))
        dim_map[s] = dim_map[mask_value] = new_dim
        return new_dim
    return s


def _mask(
    s: T,
    mask_value: Union[T, Tensor, float, None],
    *,
    mask: Tensor,
    mask_cpu: Optional[Tensor] = None,
    dim_map: Dict[Dim, Dim],
    allow_dim_extension: bool,
) -> T:
    if s is None:
        return s
    if isinstance(s, Tensor):
        if s.device == "cpu" and mask_cpu is not None:
            mask = mask_cpu
        if dim_map:
            for d in s.dims:
                if d in dim_map:
                    s = rf.replace_dim_v2(s, in_dim=d, out_dim=dim_map[d])
            if isinstance(mask_value, Tensor):
                for d in mask_value.dims:
                    if d in dim_map:
                        mask_value = rf.replace_dim_v2(mask_value, in_dim=d, out_dim=dim_map[d])
        if not allow_dim_extension and isinstance(mask_value, Tensor):
            if not s.dims_set.issuperset(mask_value.dims_set):
                return s
        if not allow_dim_extension or mask_value is None or (isinstance(mask_value, (int, float)) and mask_value == 0):
            if mask.dims_set.issubset(s.dims_set):
                return rf.where(mask, s, mask_value)
            assert not mask.dims_set.intersection(s.dims_set)  # not sure...
            return s
        assert isinstance(mask_value, (int, float, Tensor))
        return rf.where(mask, s, mask_value, allow_broadcast_all_sources=True)
    if isinstance(s, Dim):
        if mask_value is None:
            return s
        assert isinstance(mask_value, Dim)
        if s == mask_value:
            return s
        return dim_map.get(s, s)
    raise TypeError(f"_mask: unexpected {s!r} type {type(s).__name__}")


def gather_nested(s: T, *, indices: Tensor, dim_map: Optional[Dict[Dim, Dim]] = None) -> T:
    """
    This is like :func:`gather`, but for nested structures.

    :param s: nested structure
    :param indices: indices tensor. see :func:`gather`
    :param dim_map: if given, this will be updated with the new dim map
    :return: s with gathered tensors
    """
    assert indices.sparse_dim
    if dim_map is None:
        dim_map = {}
    tree.map_structure(functools.partial(_gather_prepare_dims, indices=indices, dim_map=dim_map), s)
    s = tree.map_structure(functools.partial(_gather, indices=indices, dim_map=dim_map), s)
    return s


def _gather_prepare_dims(s: T, *, indices: Tensor, dim_map: Dict[Dim, Dim]) -> T:
    if isinstance(s, Dim):
        if s.dimension is not None:  # static
            return s
        if s in dim_map:
            return dim_map[s]
        if indices.sparse_dim in s.dyn_size_ext.dims:
            new_dyn_size = _gather(s.dyn_size_ext, indices=indices, dim_map=dim_map)
            new_dim = Dim(new_dyn_size, name=_extend_dim_name(s.name))
            dim_map[s] = new_dim
            return new_dim
        return s
    # everything else ignored at this stage
    return s


def _gather(s: T, *, indices: Tensor, dim_map: Optional[Dict[Dim, Dim]] = None) -> T:
    if isinstance(s, Tensor):
        if dim_map and any(d in dim_map for d in s.dims):
            for d in s.dims:
                if d in dim_map:
                    s = rf.replace_dim_v2(s, in_dim=d, out_dim=dim_map[d])
        if indices.sparse_dim in s.dims:
            # really the default case, otherwise e.g. scalar or so, independent of beam
            s = rf.gather(s, indices=indices)
        return s
    if isinstance(s, Dim):
        if s.dimension is not None:  # static
            return s
        if dim_map and s in dim_map:
            return dim_map[s]
        assert indices.sparse_dim not in s.dyn_size_ext.dims  # not expected, should be in dim_map
        return s
    raise TypeError(f"_gather: unexpected type ({type(s)})")


def masked_select_nested(
    s: T,
    *,
    mask: Tensor,
    mask_cpu: Optional[Tensor] = None,
    dims: Sequence[Dim],
    out_dim: Optional[Dim] = None,
    dim_map: Optional[Dict[Dim, Dim]] = None,
) -> Tuple[T, Dim, Dict[Dim, Dim]]:
    """
    This is like :func:`masked_select`, but for nested structures.

    :param s: nested structure
    :param mask: mask tensor. see :func:`masked_select`
    :param mask_cpu: mask tensor for CPU. this is used e.g. for dyn dim sizes
    :param dims: dims to mask. see :func:`masked_select`
    :param out_dim: the packed out dim. see :func:`masked_select`. if not given, a new one will be created.
    :param dim_map: if given, this will be updated with the new dim map
    :return: s with masked dims, out_dim, and a newly created dim map
    """
    if out_dim is None:
        out_dim = Dim(None, name="packed_new_label")  # Flat_Batch_InBeam
    if dim_map is None:
        dim_map = {}
    tree.map_structure(
        functools.partial(
            _masked_select_prepare_dims,
            mask=mask_cpu if mask_cpu is not None else mask,
            dims=dims,
            out_dim=out_dim,
            dim_map=dim_map,
        ),
        s,
    )
    s = tree.map_structure(
        functools.partial(
            _masked_select,
            mask=mask,
            mask_cpu=mask_cpu,
            dims=dims,
            out_dim=out_dim,
            dim_map=dim_map,
        ),
        s,
    )
    return s, out_dim, dim_map


def _masked_select_prepare_dims(s, *, mask: Tensor, dims: Sequence[Dim], out_dim: Dim, dim_map: Dict[Dim, Dim]):
    if isinstance(s, Dim):
        if s.dimension is not None:  # static
            return s
        if not any(d in s.dyn_size_ext.dims for d in dims):
            return s
        if s in dim_map:
            return dim_map[s]
        new_dyn_size = _masked_select(s.dyn_size_ext, mask=mask, dims=dims, out_dim=out_dim, dim_map=dim_map)
        new_dim = Dim(new_dyn_size, name=_extend_dim_name(s.name))
        dim_map[s] = new_dim
        return new_dim
    # everything else ignored at this stage
    return s


def _masked_select(
    s: T, *, mask: Tensor, mask_cpu: Optional[Tensor] = None, dims: Sequence[Dim], out_dim: Dim, dim_map: Dict[Dim, Dim]
) -> T:
    if isinstance(s, Tensor):
        if not any(d in s.dims for d in dims):
            return s  # e.g. scalar or so, independent from dims
        if s.device == "cpu" and mask_cpu is not None:
            mask = mask_cpu
        # For the masked_select, we need that all masked dims are present, so add them if not.
        # (E.g., when we mask [batch,beam], but we only have [batch], we need to add the beam dim.)
        if any(d not in s.dims for d in dims):
            s = rf.expand_dims(s, dims=[d for d in dims if d not in s.dims])
        # The packing itself (masked_select).
        s, _ = rf.masked_select(s, mask=mask, dims=dims, out_dim=out_dim)
        # In the resulting tensor, potentially replace dims.
        # In addition to the dim replacement, we also might need to slice, as the size might be smaller.
        if any(d in dim_map for d in s.dims):
            for d in s.dims:
                if d in dim_map:
                    s, _ = rf.slice(s, axis=d, size=dim_map[d])
        return s
    if isinstance(s, Dim):
        if s.dimension is not None:  # static
            return s
        if not any(d in s.dyn_size_ext.dims for d in dims):
            return s
        assert s in dim_map
        return dim_map[s]
    raise TypeError(f"_masked_select: unexpected type ({type(s)})")


def masked_scatter_nested(
    s: T,
    backup: T,
    *,
    mask: Tensor,
    mask_cpu: Tensor,
    dims: Sequence[Dim],
    in_dim: Dim,
    masked_select_dim_map: Dict[Dim, Dim],
    masked_scatter_dim_map: Optional[Dict[Dim, Dim]] = None,
) -> T:
    """
    Reverse of :func:`masked_select_nested`.

    :param s: nested structure, where dims are packed, i.e. (in_dim,...)
    :param backup: nested structure, where we scatter into. tensors like (dims...,...)
    :param mask: mask tensor. see :func:`masked_scatter`/:func:`masked_select`
    :param mask_cpu: mask tensor for CPU. this is used e.g. for dyn dim sizes. see :func:`masked_scatter`
    :param dims: dims to mask. see :func:`masked_scatter`/:func:`masked_select`
    :param in_dim: the packed in dim. see :func:`masked_scatter`
    :param masked_select_dim_map: the dim map from :func:`masked_select_nested`.
        This describes how to map dims from s to backup.
    :param masked_scatter_dim_map: for any new dims created by this function, this will be updated
    :return: backup with s scattered in
    """
    reverse_dim_map = {v: k for k, v in masked_select_dim_map.items()}
    if masked_scatter_dim_map is None:
        masked_scatter_dim_map = {}

    tree.map_structure(
        functools.partial(
            _masked_scatter_merge_dims,
            mask=mask_cpu,
            dims=dims,
            in_dim=in_dim,
            reverse_dim_map=reverse_dim_map,
            merged_dim_map=masked_scatter_dim_map,
        ),
        s,
        backup,
    )
    s = tree.map_structure(
        functools.partial(
            _masked_scatter,
            mask=mask,
            mask_cpu=mask_cpu,
            dims=dims,
            in_dim=in_dim,
            reverse_dim_map=reverse_dim_map,
            merged_dim_map=masked_scatter_dim_map,
        ),
        s,
        backup,
    )
    return s


def _masked_scatter_merge_dims(
    s: T,
    backup: T,
    *,
    mask: Tensor,
    dims: Sequence[Dim],
    in_dim: Dim,
    reverse_dim_map: Dict[Dim, Dim],
    merged_dim_map: Dict[Dim, Dim],
) -> T:
    if isinstance(s, Dim):
        # This is slightly more complex than in the _masked_select case:
        # We need to merge the s and backup depending on the mask.
        if s in reverse_dim_map:
            s = reverse_dim_map[s]
        if s == backup:
            return s
        if s in merged_dim_map:
            return merged_dim_map[s]
        # Note: s/backup might even be static dims.
        new_size = _masked_scatter(
            s.get_size_tensor(),
            backup.get_size_tensor(),
            mask=mask,
            dims=dims,
            in_dim=in_dim,
            reverse_dim_map=reverse_dim_map,
            merged_dim_map=merged_dim_map,
        )
        assert new_size.dims_set == (
            (s.get_size_tensor().dims_set | backup.get_size_tensor().dims_set) - {in_dim}
        ) | set(dims)
        new_dim = Dim(new_size, name=backup.name)
        merged_dim_map[s] = new_dim
        merged_dim_map[backup] = new_dim
        return new_dim
    # everything else ignored at this stage
    return s


def _masked_scatter(
    s: T,
    backup: T,
    *,
    mask: Tensor,
    mask_cpu: Optional[Tensor] = None,
    dims: Sequence[Dim],
    in_dim: Dim,
    reverse_dim_map: Dict[Dim, Dim],
    merged_dim_map: Dict[Dim, Dim],
) -> T:
    if isinstance(s, Tensor):
        assert isinstance(backup, Tensor)
        if s.device == "cpu" and mask_cpu is not None:
            mask = mask_cpu
        if in_dim not in s.dims:
            s = rf.expand_dim(s, in_dim)
        # Do the reverse of _masked_select above.
        # First replace the dims back.
        if any(d in reverse_dim_map for d in s.dims):
            for d in s.dims:
                if d in reverse_dim_map:
                    s = rf.replace_dim_v2(s, in_dim=d, out_dim=reverse_dim_map[d], allow_shrink=False)
        # We also might need to replace newly merged dims, both in s and backup.
        for d in s.dims:
            if d in merged_dim_map:
                s = rf.replace_dim_v2(s, in_dim=d, out_dim=merged_dim_map[d])
        for d in backup.dims:
            if d in merged_dim_map:
                backup = rf.replace_dim_v2(backup, in_dim=d, out_dim=merged_dim_map[d])
        # The unpacking itself (reversing the masked_select, i.e. masked_scatter).
        s = rf.masked_scatter(s, backup, mask=mask, dims=dims, in_dim=in_dim)
        return s
    if isinstance(s, Dim):
        # This is slightly more complex than in the _masked_select case:
        # We need to merge the s and backup depending on the mask.
        if s in reverse_dim_map:
            s = reverse_dim_map[s]
        if s in merged_dim_map:
            return merged_dim_map[s]
        return s
    raise TypeError(f"_masked_scatter: unexpected type ({type(s)})")


def _extend_dim_name(name: str) -> str:
    # check ends with _<num>
    m = re.match(r"^(.*)_(\d+)$", name)
    if m:
        return f"{m.group(1)}_{int(m.group(2)) + 1}"
    return name + "_1"
