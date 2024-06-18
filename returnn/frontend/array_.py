"""
Array (Tensor) functions
"""

from __future__ import annotations
from typing import Optional, Union, Type, TypeVar, Sequence, Tuple
import logging
import numpy
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from ._backend import Backend, global_backend, get_backend_by_raw_tensor_type
from .types import RawTensorTypes

T = TypeVar("T")

__all__ = [
    "convert_to_tensor",
    "constant",
    "copy",
    "cast",
    "merge_dims",
    "split_dims",
    "reshape",
    "split",
    "expand_dim",
    "squeeze",
    "window",
    "concat",
    "concat_features",
    "pad",
    "cum_concat_step",
    "stack",
    "masked_select",
    "masked_scatter",
    "sequence_mask",
    "pack_padded",
    "gather",
    "scatter",
    "slice",
    "shift_right",
    "reverse_sequence",
    "where",
    "search_sorted",
    "sparse_to_dense",
    "one_hot",
]


def convert_to_tensor(
    value: Union[Tensor, T, RawTensorTypes],
    *,
    dims: Sequence[Dim] = None,
    dtype: Optional[str] = None,
    sparse_dim: Optional[Dim] = None,
    shape: Sequence[Dim] = None,
    device: Optional[str] = None,
    keep_scalar_on_cpu: bool = False,
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
    :param device:
    :param keep_scalar_on_cpu: if the value is already on the CPU, keep it there, even if `device` is sth else
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
            if isinstance(value, bool):
                dtype = "bool"
            elif isinstance(value, int):
                dtype = rf.get_default_int_dtype()
            elif isinstance(value, float):
                dtype = rf.get_default_float_dtype()
            elif isinstance(value, str):
                dtype = "string"
            elif isinstance(value, numpy.number):
                dtype = value.dtype.name
            else:
                raise ValueError(f"number {value} type {type(value)} needs explicit `dtype` specification")
        if keep_scalar_on_cpu:
            device = "cpu"
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
    return _backend.convert_to_tensor(
        value=value, dims=dims, dtype=dtype, sparse_dim=sparse_dim, device=device, name=name
    )


constant = convert_to_tensor  # alias for some older code


def copy(tensor: Tensor) -> Tensor:
    """
    :param tensor:
    :return: copy of tensor.
        In eager-based frameworks, it is really a copy.
        In graph-based frameworks, it might be just a copied reference if it would be immutable.
        This is really only relevant when operating on tensors which can conceptually be mutated,
        such as variables (:class:`Parameter`).
    """
    # noinspection PyProtectedMember
    return tensor._raw_backend.copy(tensor)


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

    :func:`rf.split_dims` is the reverse operation.

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

    Also see :func:`rf.merge_dims` which can undo this operation.

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


def reshape(source: Tensor, in_dims: Sequence[Dim], out_dims: Sequence[Dim]) -> Tensor:
    """
    Wraps tf.reshape.

    You should use :func:`split_dims` or :func:`merge_dims`
    when you want to split or merge dimensions.
    This here is for doing any other kind of reshape.
    This can be used for clever indexing, slicing, padding tricks.

    :param source: e.g. (..., old_dims, ...)
    :param in_dims: the old dims which should be reshaped into new_dims.
      This should only cover those dims which should be reshaped,
      not all the dims of the source.
    :param out_dims: the new dims which should be reshaped from old_dims.
      This is excluding any of the other dims in the source.
    :return: e.g. (..., new_dims, ...)
    """
    # noinspection PyProtectedMember
    return source._raw_backend.reshape(source, in_dims=in_dims, out_dims=out_dims)


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


def expand_dim(source: Tensor, dim: Dim) -> Tensor:
    """
    Expand the source by the given dimension.

    Note that this is *never* needed for broadcasting.
    All broadcasting should always happen automatically.

    This might be needed for convolution or concatenation.
    """
    # noinspection PyProtectedMember
    return source._raw_backend.expand_dim(source, dim=dim)


def squeeze(source: Tensor, axis: Dim) -> Tensor:
    """
    Removes the axis with dimension of extend 1 from the source.
    """
    assert axis.dimension == 1, f"squeeze {source}: axis {axis} is not of extend 1"
    # noinspection PyProtectedMember
    return source._raw_backend.squeeze(source, axis=axis)


def window(
    source: Tensor,
    *,
    spatial_dim: Dim,
    window_dim: Dim,
    window_right: Optional[Union[Dim, int]] = None,
    window_left: Optional[Union[Dim, int]] = None,
    padding: str = "same",
    pad_value: Optional[Union[int, float]] = None,
    stride: int = 1,
) -> Tuple[Tensor, Dim]:
    """
    Follows the same idea as RETURNN tf_util.windowed,
    using clever padding and reshaping.

    :param source:
    :param spatial_dim:
    :param window_dim:
    :param window_left:
    :param window_right:
    :param padding: "same" or "valid"
    :param pad_value:
    :param stride:
    :return: out, out_spatial_dim
    """
    assert window_dim.dimension is not None
    if padding == "same":
        out_spatial_dim = spatial_dim
        if window_right is not None:
            if isinstance(window_right, int):
                window_right = Dim(window_right, name="window_right")
            assert isinstance(window_right, Dim)
        if window_left is not None:
            if isinstance(window_left, int):
                window_left = Dim(window_left, name="window_left")
            assert isinstance(window_left, Dim)
        if window_right is None:
            if window_left is None:
                window_right = window_dim // 2
                window_left = window_dim.ceildiv_right(2) - 1
            else:
                window_right = window_dim - window_left - 1
        if window_left is None:
            window_left = window_dim - window_right - 1
        source, (in_spatial_dim,) = rf.pad(
            source,
            axes=[spatial_dim],
            padding=[(window_left, window_right)],
            value=pad_value,
        )
        # shape[0] == n_time + window - 1
    elif padding == "valid":
        in_spatial_dim = spatial_dim
        out_spatial_dim = spatial_dim - window_dim + 1
    else:
        raise ValueError(f"invalid padding {padding!r}")

    if stride > 1:
        start_times, out_spatial_dim = rf.range_over_dim_strided(out_spatial_dim, stride=stride)  # (n_out_time,)
        win_range = rf.range_over_dim(window_dim)  # (window,)
        indices = rf.combine_bc(start_times, "+", win_range)  # (n_out_time,window)
        final = rf.gather(source, indices=indices, axis=in_spatial_dim)  # (n_out_time,window,...)
        return final, out_spatial_dim

    tiled_dimshuffle = rf.expand_dim(source, dim=window_dim)  # (window,n_time+window-1,...)
    # We want to shift every dim*time block by one to the left.
    # To do this, we interpret that we have one more time frame (i.e. n_time+window).
    # We have to do some dimshuffling so that we get the right layout, then we can flatten,
    # add some padding, and then dimshuffle it back.
    # Then we can take out the first n_time frames.
    tiled_flat, flat_dim = rf.merge_dims(tiled_dimshuffle, dims=(window_dim, in_spatial_dim))
    rem = window_dim
    tiled_flat_pad_right, (flat_dim_ext,) = rf.pad(tiled_flat, axes=[flat_dim], padding=[(0, rem)], value=pad_value)
    # add time frame, (window,n_time+window,...)
    out_time_ext = out_spatial_dim + window_dim
    tiled_reshape_shift = rf.reshape(tiled_flat_pad_right, in_dims=[flat_dim_ext], out_dims=[window_dim, out_time_ext])
    final, _ = rf.slice(tiled_reshape_shift, axis=out_time_ext, size=out_spatial_dim)  # (window,n_time,...)
    if stride > 1:
        final, out_spatial_dim = rf.slice(final, axis=out_spatial_dim, step=stride)
    return final, out_spatial_dim


def concat(
    *sources: Tuple[Tensor, Dim],
    allow_broadcast: bool = False,
    out_dim: Optional[Dim] = None,
) -> Tuple[Tensor, Dim]:
    """
    Concatenates multiple sources in the specified dimension.

    Also see :func:`stack`.

    :param sources: list of (tensor, dim) pairs. dim is the axis to concatenate on.
    :param allow_broadcast: if True, the sources can have different dims, and the result will be broadcasted.
    :param out_dim: reuse existing dim for the resulting concatenated dim, if given
    :return: concatenated tensor, out_dim
    """
    assert sources
    if not allow_broadcast:
        dims = sources[0][0].dims_set - {sources[0][1]}
        for src, dim in sources:
            assert src.dims_set - {dim} == dims, f"concat {sources}, need allow_broadcast=True"
    if not out_dim:
        out_dim = sum(d for _, d in sources)
    # noinspection PyProtectedMember
    return sources[0][0]._raw_backend.concat(*sources, allow_broadcast=allow_broadcast, out_dim=out_dim), out_dim


def concat_features(*sources: Tensor, allow_broadcast=False) -> Tensor:
    """
    Concatenates multiple sources, using feature_dim of each source,
    so make sure that the feature_dim is correctly set.
    """
    src_pairs = []
    for src in sources:
        assert src.feature_dim is not None
        src_pairs.append((src, src.feature_dim))
    res, out_dim = concat(*src_pairs, allow_broadcast=allow_broadcast)
    res.feature_dim = out_dim
    return res


def pad(
    source: Tensor,
    *,
    axes: Sequence[Dim],
    padding: Sequence[Tuple[Union[Dim, int], Union[Dim, int]]],
    out_dims: Optional[Sequence[Dim]] = None,
    mode: str = "constant",
    value: Optional[Union[rf.RawTensorTypes, Tensor]] = None,
    handle_dynamic_dims: Optional[bool] = None,
) -> Tuple[Tensor, Sequence[Dim]]:
    """
    Pad values left/right in the specified axes.

    :param source:
    :param axes: which axes to add padding to
    :param padding: list of (left, right) padding for each axis
    :param out_dims: (optional) predefined out dims for each padded dim in axes. will automatically create if not given
    :param mode: 'constant', 'reflect', 'replicate' or 'circular'
    :param value: (optional) value to pad with in "constant" mode
    :param handle_dynamic_dims: True: when doing right padding on a dynamic dim, value will be added after the seq end,
        not at the end of the dimension. False: value will be added at the end of the dimension.
        By default, in behavior version >=21, this is True, in older versions, this is False.
    :return: padded tensor, out_dims. out dims are for each dim in axes
    """
    assert len(axes) == len(padding)
    if not out_dims:
        for left, right in padding:
            if isinstance(left, Dim):
                assert not left.need_masking(), f"padding {padding} does not support dynamic left padding"
            if isinstance(right, Dim):
                assert not right.need_masking(), f"padding {padding} does not support dynamic right padding"
            # Note that even dynamic middle dims is not exactly correct...
        out_dims = [left + middle + right for middle, (left, right) in zip(axes, padding)]
    if handle_dynamic_dims is None:
        handle_dynamic_dims = _pad_handle_dynamic_dims_default(axes, padding, mode=mode)
    # noinspection PyProtectedMember
    return (
        source._raw_backend.pad(
            source,
            axes=axes,
            padding=padding,
            out_dims=out_dims,
            handle_dynamic_dims=handle_dynamic_dims,
            mode=mode,
            value=value,
        ),
        out_dims,
    )


_pad_handle_dynamic_dims_shown_warning = False


def _pad_handle_dynamic_dims_default(
    pad_axes: Sequence[Dim], padding: Sequence[Tuple[Union[Dim, int], Union[Dim, int]]], *, mode: str
) -> bool:
    """
    :param pad_axes: list of axes to pad
    :param padding: list of (left, right) padding for each axis
    :param mode: 'constant', 'reflect', 'replicate' or 'circular'
    :return: True if dynamic dims should be handled as specified in the default behavior
    """
    from returnn.util.basic import BehaviorVersion

    if BehaviorVersion.get() >= 21:
        return True

    # Check whether not handling the dynamic dims is safe. Print a warning if not safe.
    global _pad_handle_dynamic_dims_shown_warning
    if not _pad_handle_dynamic_dims_shown_warning:
        for middle, (left, right) in zip(pad_axes, padding):
            middle: Dim
            if not middle.need_masking() and (isinstance(left, int) or not left.need_masking()):
                continue
            if mode != "circular" and isinstance(right, int) and right == 0:
                continue

            logging.getLogger("returnn.frontend").warning(
                f"rf.pad applied on dynamic dim {middle} but handle_dynamic_dims=False used by default"
                f" due to behavior version {BehaviorVersion.get()} < 21."
                " Set handle_dynamic_dims explicitly to avoid the warning,"
                " or switch to a new behavior version >= 21."
                " (This warning is only printed once.)"
            )
            _pad_handle_dynamic_dims_shown_warning = True
            break
    return False


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


def stack(sources: Sequence[Tensor], *, out_dim: Optional[Dim] = None) -> Tuple[Tensor, Dim]:
    """
    Stack the sources in a new dimension.
    All sources must have the same shape.

    :param sources:
    :param out_dim: if given, use this as the new dim
    :return: stacked tensor, out_dim
    """
    if not sources:
        raise ValueError("no sources to stack")
    if not out_dim:
        out_dim = Dim(len(sources), name="stack")
    # noinspection PyProtectedMember
    return sources[0]._raw_backend.stack(sources, out_dim=out_dim), out_dim


def masked_select(
    tensor: Tensor, *, mask: Tensor, dims: Sequence[Dim], out_dim: Optional[Dim] = None
) -> Tuple[Tensor, Dim]:
    """
    In TF, this is ``boolean_mask``.
    The inverse of this is :func:`masked_scatter`.

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


def masked_scatter(source: Tensor, *, mask: Tensor, dims: Sequence[Dim], in_dim: Dim) -> Tensor:
    """
    The inverse of :func:`masked_select`.

    :param source: [in_dim, F...]
    :param mask: [dims...] -> bool (e.g. [B,T])
    :param dims: the order of the dims defines the format. those dims should be exactly the dims of the mask.
    :param in_dim: the dim of the source which should be scattered into the mask.
    :return: [dims..., F...]
    """
    # noinspection PyProtectedMember
    return source._raw_backend.masked_scatter(source, mask=mask, dims=dims, in_dim=in_dim)


def sequence_mask(dims: Union[Dim, Sequence[Dim]], *, device: Optional[str] = None) -> Tensor:
    """
    :param dims:
    :param device:
    """
    if isinstance(dims, Dim):
        dims = [dims]
    assert len(dims) > 0
    dyn_dims = [d for d in dims if d.need_masking()]
    assert len(dyn_dims) == 1  # not implemented otherwise yet...
    return dyn_dims[0].get_mask(dim_order=dims, device=device)


def pack_padded(
    source: Tensor, *, dims: Sequence[Dim], enforce_sorted: bool = False, out_dim: Optional[Dim] = None
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
    mask = rf.sequence_mask(dims, device=source.device)
    assert mask.dims_set == set(dims)
    return rf.masked_select(source, mask=mask, dims=dims, out_dim=out_dim)


# noinspection PyUnusedLocal
def gather(
    source: Tensor,
    *,
    indices: Union[Tensor, int],
    axis: Optional[Dim] = None,
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

    :func:`scatter` is the inverse.

    :param source: [batch_dims..., axis, feature_dims...]
    :param indices: [batch_dims..., indices_dims...] indices used to select the slices of the source from.
        If another tensor, must be of type ``int32`` or ``int64``.
        Can also specify a constant ``int``.
        Batch dims are automatically determined as the common dims of source and indices.
    :param axis: The axis into which we gather the indices into.
        If not given, indices must be a tensor and the sparse_dim will be used.
    :param clip_to_valid: if True, the indices will be clipped to the valid range of the input
        Also taking seq lengths into account.
    :return: [batch_dims..., indices_dims..., feature_dims...] gathered values
    """
    if not axis:
        assert isinstance(indices, Tensor) and indices.sparse_dim
        axis = indices.sparse_dim
    # noinspection PyProtectedMember
    return source._raw_backend.gather(source, indices=indices, axis=axis, clip_to_valid=clip_to_valid)


def scatter(
    source: Tensor,
    *,
    indices: Tensor,
    indices_dim: Union[Dim, Sequence[Dim]],
    out_dim: Optional[Union[Dim, Sequence[Dim]]] = None,
) -> Tensor:
    """
    Scatters into new zero-tensor.
    If entries in indices are duplicated, the corresponding values in source will be added together
    (scatter_add in PyTorch).
    (TF segment_sum can be implemented via this.)

    :param source: [batch_dims..., indices_dim(s)..., feature_dims...]
    :param indices: [batch_dims..., indices_dim(s)...] -> out_dim
    :param indices_dim:
    :param out_dim: The indices target dim.
        If not given, will be automatically determined as the sparse_dim from indices.
        If multiple out dims, use indices into the merged out dims,
        and then we use :func:`rf.split_dims` afterwards.
    :return: [batch_dims..., out_dim(s)..., feature_dims...]
    """
    if not out_dim:
        assert isinstance(indices, Tensor) and indices.sparse_dim
        out_dim = indices.sparse_dim
    # noinspection PyProtectedMember
    return source._raw_backend.scatter(source, indices=indices, indices_dim=indices_dim, out_dim=out_dim)


# noinspection PyShadowingBuiltins
def slice(
    source: Tensor,
    *,
    axis: Dim,
    start: Optional[Union[int, Tensor]] = None,
    end: Optional[Union[int, Tensor]] = None,
    step: Optional[Union[int, Tensor]] = None,
    size: Optional[Union[int, Tensor, Dim]] = None,
    out_dim: Optional[Dim] = None,
) -> Tuple[Tensor, Dim]:
    """
    Slicing on the input, i.e. ``x[start:end:step]`` in some axis.

    If size is given, it takes out a slice-range like ``x[start:start + size]``.

    This function allows a non-scalar start points.

    :param source:
    :param axis:
    :param start:
    :param end:
    :param step:
    :param size:
    :param out_dim:
    :return: tensor, out_dim
    """
    if not out_dim:
        # Note: We cover not really all possible cases here but just those we really support.
        # (Actually, it might still be a superset of cases we really support,
        # but this might depend on the backend.)
        if size is not None:
            if isinstance(size, Dim):
                out_dim = size
            elif isinstance(size, (int, Tensor)):
                out_dim = Dim(size, name="slice")
            else:
                raise TypeError(f"invalid type {type(size)} for size {size}")
            assert step is None or (isinstance(step, int) and step == 1)

        else:  # size is None
            if start is None:
                start = 0
            if isinstance(start, int) and start >= 0:
                if end is None:
                    out_dim = axis.sub_left(start)
                elif isinstance(end, int):
                    if end < 0:
                        out_dim = axis.sub_left(start).sub_right(-end)
                    else:
                        out_dim = Dim(end - start, name="slice")
                elif isinstance(end, Tensor):
                    out_dim = Dim(end - start, name="slice")
                else:
                    raise TypeError(f"invalid type {type(end)} for end {end}")
            elif isinstance(start, int) and start < 0:
                if end is None:
                    out_dim = Dim(-start, name="slice")
                elif isinstance(end, int):
                    assert end < 0
                    out_dim = Dim(-start + end, name="slice")
                else:
                    raise TypeError(f"invalid type {type(end)} for end {end}")
            elif isinstance(start, Tensor):
                out_dim = Dim(
                    ((axis.get_size_tensor() if end is None else end) - start) % axis.get_size_tensor(), name="slice"
                )
            else:
                raise TypeError(f"invalid type {type(start)} for start {start}")

            if step is None or (isinstance(step, int) and step == 1):
                pass
            elif isinstance(step, int):
                out_dim = out_dim.ceildiv_right(step)
            elif isinstance(step, Tensor):
                step_dim = Dim(step, name="step")
                out_dim = out_dim.ceildiv_right(step_dim)
            else:
                raise TypeError(f"invalid type {type(step)} for step {step}")
    # noinspection PyProtectedMember
    return (
        source._raw_backend.slice(source, axis=axis, start=start, end=end, step=step, size=size, out_dim=out_dim),
        out_dim,
    )


def shift_right(source: Tensor, *, axis: Dim, pad_value: Union[rf.RawTensorTypes, Tensor], amount: int = 1) -> Tensor:
    """shift right by amount, pad left with left_pad"""
    padded, (padded_dim,) = rf.pad(source, axes=[axis], padding=[(amount, 0)], mode="constant", value=pad_value)
    padded_slice, _ = rf.slice(padded, axis=padded_dim, size=axis)
    return padded_slice


def reverse_sequence(tensor: Tensor, *, axis: Dim) -> Tensor:
    """
    Similar as tf.reverse_sequence, or Torch flip (but taking seq lengths into account).

    :param tensor:
    :param axis:
    :return: reversed tensor, same dims
    """
    indices = rf.combine_bc(axis.get_size_tensor(), "-", rf.range_over_dim(axis)) - 1
    return rf.gather(tensor, indices=indices, axis=axis, clip_to_valid=True)


def where(
    cond: Union[Tensor, rf.RawTensorTypes],
    true_: Union[Tensor, rf.RawTensorTypes],
    false_: Union[Tensor, rf.RawTensorTypes],
    *,
    allow_broadcast_all_sources: bool = False,
) -> Tensor:
    """
    Wraps tf.where, which is SwitchLayer in RETURNN.

    :return: true_ if cond else false_, elemwise.
    """
    cond = rf.convert_to_tensor(cond)
    # noinspection PyProtectedMember
    return cond._raw_backend.where(cond, true_, false_, allow_broadcast_all_sources=allow_broadcast_all_sources)


def search_sorted(
    sorted_seq: Tensor, values: Tensor, *, axis: Dim, side: str = "left", out_dtype: str = "int32"
) -> Tensor:
    """
    :param sorted_seq: [SharedDims...,axis], sequence of numbers, sorted low to high in the given axis.
    :param values: [SharedDims...,OtherDims...], sequence of numbers to search for in ``sorted_seq``.
    :param axis:
    :param side: "left" or "right"
    :param out_dtype:
    :return: [SharedDims...,OtherDims...] -> axis, indices in axis in ``sorted_seq`` such that
        sorted_seq[i-1] < value <= sorted_seq[i] if side=="left",
        sorted_seq[i-1] <= value < sorted_seq[i] if side=="right".
    """
    # noinspection PyProtectedMember
    return sorted_seq._raw_backend.search_sorted(sorted_seq, values, axis=axis, side=side, out_dtype=out_dtype)


def sparse_to_dense(
    labels: Union[Tensor, rf.RawTensorTypes],
    *,
    label_value: Union[Tensor, rf.RawTensorTypes],
    other_value: Union[Tensor, rf.RawTensorTypes],
    axis: Optional[Dim] = None,
) -> Tensor:
    """
    Converts a sparse tensor to a dense one.

    This is a more generic variant of "one_hot".

    Note that usually this is not needed as most other functions should handle sparse tensors just fine
    and much more efficiently than they would be with dense tensors.
    """
    labels = rf.convert_to_tensor(labels)
    if not axis:
        assert labels.sparse_dim, "sparse_to_dense: either provide `axis` or `labels` with sparse_dim"
        axis = labels.sparse_dim
    indices = rf.range_over_dim(axis)
    return where(rf.compare_bc(labels, "==", indices), label_value, other_value)


def one_hot(source: Tensor) -> Tensor:
    """
    one_hot. special case of :func:`sparse_to_dense`.

    Note that usually this is not needed as most other functions should handle sparse tensors just fine
    and much more efficiently than they would be with dense tensors.
    """
    return sparse_to_dense(source, label_value=1.0, other_value=0.0)
