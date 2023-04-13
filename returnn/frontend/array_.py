"""
Array (Tensor) functions
"""

from __future__ import annotations
from typing import Optional, Union, Type, TypeVar, Sequence, Tuple
import numpy
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from ._backend import Backend, global_backend, get_backend_by_raw_tensor_type
from .types import RawTensorTypes

T = TypeVar("T")

__all__ = [
    "convert_to_tensor",
    "constant",
    "cast",
    "merge_dims",
    "split_dims",
    "split",
    "cum_concat_step",
    "masked_select",
    "pack",
    "gather",
]


def convert_to_tensor(
    value: Union[Tensor, T, RawTensorTypes],
    *,
    dims: Sequence[Dim] = None,
    dtype: Optional[str] = None,
    sparse_dim: Optional[Dim] = None,
    shape: Sequence[Dim] = None,
    name: Optional[str] = None,
    _backend: Optional[Type[Backend]] = None,
) -> Tensor[T]:
    """
    :param value: tensor, or scalar raw tensor or some other scalar value
    :param dims:
    :param dtype:
    :param sparse_dim:
    :param shape: alias for dims, for some older code
    :param name:
    :param _backend:
    :return: tensor
    """
    if isinstance(value, Tensor):  # fast path
        return value
    if dims is None and shape is not None:
        dims = shape  # old code
    if isinstance(value, (int, float, complex, bool, str, numpy.number)):
        if _backend is None:
            _backend = global_backend
        if dims is None:
            dims = ()
        if dtype is None:
            if isinstance(value, int):
                dtype = rf.get_default_int_dtype()
            elif isinstance(value, float):
                dtype = rf.get_default_float_dtype()
            elif isinstance(value, bool):
                dtype = "bool"
            elif isinstance(value, str):
                dtype = "string"
            elif isinstance(value, numpy.number):
                dtype = value.dtype.name
            else:
                raise ValueError(f"number {value} type {type(value)} needs explicit `dtype` specification")
    elif isinstance(value, numpy.ndarray):
        if _backend is None:
            # Small exception: Do not use the NumpyBackend but the global backend in this case.
            _backend = global_backend
        if dims is None:
            dims = [Dim(d) for d in value.shape]
        if dtype is None:
            dtype = value.dtype.name
    else:
        value_backend = get_backend_by_raw_tensor_type(type(value))
        if _backend is None:
            _backend = value_backend
        if dims is None:
            dims = [
                value_backend.get_new_dim_raw(value, d, name=(name or "const") + f"_dim{d}")
                for d in range(value_backend.get_ndim_raw(value))
            ]
        if dtype is None:
            dtype = value_backend.get_dtype_name_raw(value)
    return _backend.convert_to_tensor(value=value, dims=dims, dtype=dtype, sparse_dim=sparse_dim, name=name)


constant = convert_to_tensor  # alias for some older code


def cast(tensor: Tensor, dtype: str) -> Tensor:
    """
    :param tensor:
    :param dtype:
    :return: tensor with the same data, but with a different dtype
    """
    # noinspection PyProtectedMember
    return tensor._raw_backend.cast(tensor, dtype=dtype)


def merge_dims(
    source: Tensor,
    *,
    dims: Sequence[Dim],
    out_dim: Optional[Dim] = None,
) -> Tuple[Tensor, Dim]:
    """
    Merges a list of axes into a single one. (Flatten the dims.)
    E.g. input is (batch, width, height, dim) and dims=(width,height), then we get (batch, width*height, dim).
    Or input is (batch, time, height, dim) and axes=(height,dim), then we get (batch, time, height*dim).

    :param source:
    :param dims:
    :param out_dim:
    :return: tensor, out_dim
    """
    # noinspection PyProtectedMember
    return source._raw_backend.merge_dims(source, dims=dims, out_dim=out_dim)


def split_dims(
    source: Tensor,
    *,
    axis: Dim,
    dims: Sequence[Dim],
    pad_to_multiples: Optional[bool] = None,
    pad_value: Union[None, int, float] = None,
) -> Tensor:
    """
    Splits one axis into multiple axes.
    E.g. if you know that your feature-dim is composed by a window,
    i.e. the input is (batch, time, window * feature),
    you can set axis="F", dims=(window, -1),
    and you will get the output (batch, time, window, feature).

    If the split axis has a dynamic length,
    exactly one of the axes that we split into need to also have a dynamic length.
    You can e.g. use this to split the input dimension into smaller "chunks" of a fixed window size.
    E.g. you could have input (batch, time, feature) and set axis="T", dims=(-1, window),
    to get output (batch, split_time, window, feature).
    In this case, the exact sequence lengths are lost and everything is padded to multiples of the window size using
    the given padding value.
    Use :class:`ReinterpretDataLayer` to receive back the original sequence lengths after merging.

    Also see :class:`SplitBatchTimeLayer`.
    Also see :class:`MergeDimsLayer` which can undo this operation.

    :param source:
    :param axis: e.g. "F"
    :param dims: what the axis should be split into. e.g. (window, -1)
    :param pad_to_multiples: If true, input will be padded to the next multiple of the product of the
        static dims, such that splitting is actually possible.
        By default this is done iff the axis has a dynamic size
    :param pad_value: What pad value to use for pad_to_multiples
    :return: source with axis replaced by dims
    """
    # noinspection PyProtectedMember
    return source._raw_backend.split_dims(
        source, axis=axis, dims=dims, pad_to_multiples=pad_to_multiples, pad_value=pad_value
    )


def split(source: Tensor, *, axis: Dim, out_dims: Sequence[Dim]) -> Tuple[Tensor, ...]:
    """
    Split the input on the specified axis (by default feature).
    Basically a wrapper around tf.split.

    :param source: {..., axis}
    :param axis: some static axis
    :param out_dims: list of dims where sum(out_dims) == axis
    :return: tuple of tensors, same amount as out_dims,
        with the same shape as source, but with the specified axis replaced by the out_dims
    """
    # noinspection PyProtectedMember
    return source._raw_backend.split(source, axis=axis, out_dims=out_dims)


def cum_concat_step(
    source: Tensor, *, prev_accum: Tensor, axis: Dim, out_spatial_dim: Optional[Dim] = None
) -> Tuple[Tensor, Dim]:
    """
    Concatenates all previous frames over a time-axis.
    See RETURNN :class:`CumConcatLayer` for details.

    :param source: same dims as prev_accum except for the accum axis
    :param prev_accum: previous accumulated tensor, shape {..., axis}
    :param axis: the axis to accumulate over
    :param out_spatial_dim: if given, the spatial dim of the output will be this dim. axis+1.
    :return: (accumulated, out_spatial_dim). accumulated shape {..., out_spatial_dim},
        same shape as prev_accum with axis replaced by out_spatial_dim.
    """
    if not out_spatial_dim:
        out_spatial_dim = axis + 1
    # noinspection PyProtectedMember
    return (
        source._raw_backend.cum_concat_step(source, prev_accum=prev_accum, axis=axis, out_spatial_dim=out_spatial_dim),
        out_spatial_dim,
    )


def masked_select(
    tensor: Tensor, *, mask: Tensor, dims: Sequence[Dim], out_dim: Optional[Dim] = None
) -> Tuple[Tensor, Dim]:
    """
    :param tensor:
    :param mask:
    :param dims: the order of the dims defines the format. those dims should be exactly the dims of the mask.
    :param out_dim:
    :return: tensor where all dims in mask/dims are removed and replaced by a new dim.
        the new dim is also returned.
        if mask==True for all elements, the returned tensor would be simply the flattened input tensor.
    """
    # noinspection PyProtectedMember
    return tensor._raw_backend.masked_select(tensor, mask=mask, dims=dims, out_dim=out_dim)


def pack(
    source: Tensor, *, dims: Sequence[Dim], enforce_sorted: bool = True, out_dim: Optional[Dim] = None
) -> Tuple[Tensor, Dim]:
    """
    Like pack_padded_sequence. Usually the sequences are padded when they have different lengths.
    Packing means to only store the non-padded frames.
    This uses :func:`masked_select` internally based on the mask of non-masked frames.

    :param source:
    :param dims: dims in source to pack. the order defines the format. first dim is major, etc.
        if there are no padded frames, e.g. dims=[B,T] would just result in the [B*T,...] reshaped tensor.
    :param enforce_sorted: seqs in the dims are reordered (stable sort) such that longest seqs come first.
    :param out_dim:
    :return: packed tensor, new packed dim
    """
    assert not enforce_sorted  # not implemented yet...
    assert len(dims) > 0
    dyn_dims = [d for d in dims if d.is_dynamic()]
    assert len(dyn_dims) == 1  # not implemented otherwise yet...
    mask = source.get_sequence_mask_tensor(source.get_axis_from_description(dyn_dims[0]))
    return rf.masked_select(source, mask=mask, dims=dims, out_dim=out_dim)


# noinspection PyUnusedLocal
def gather(
    source: Tensor,
    *,
    indices: Union[Tensor, int],
    axis: Dim,
    clip_to_valid: bool = False,
) -> Tensor:
    """
    Gathers slices on a specified axis from the source using indices.
    If the source is of the shape ``[B,D,F1]``, and indices of shape ``[B,F2]``,
    this will yield output of the shape ``[B,F2,F1]`` where

    ``output[b,f2,f1] = source[b,indices[b,f2],f1]``

    (if ``D`` is the axis to gather from).
    In general, all shared axes of the input and the positions will be considered as batch-axes.

    The ``indices`` argument can also be an ``int``.
    In this case, this simply gives ``source[indices]`` on the specified ``axis``.

    :param source:
    :param indices: indices used to select the slices of the source from.
        If another tensor, must be of type ``int32`` or ``int64``.
        Can also specify a constant ``int``.
    :param axis: The axis into which we gather the indices into
    :param clip_to_valid: if True, the indices will be clipped to the valid range of the input
        Also taking seq lengths into account.
    :return: gathered values
    """
    # noinspection PyProtectedMember
    return source._raw_backend.gather(source, indices=indices, axis=axis, clip_to_valid=clip_to_valid)
