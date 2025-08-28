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
from . import _utils

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
    "expand_dims",
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
    "pad_packed",
    "gather",
    "scatter",
    "scatter_mean",
    "scatter_argmax",
    "scatter_logsumexp",
    "scatter_logmeanexp",
    "slice",
    "shift_right",
    "shift_left",
    "reverse_sequence",
    "where",
    "sort",
    "search_sorted",
    "sparse_to_dense",
    "one_hot",
    "top_k_mask",
    "top_p_mask",
]


def convert_to_tensor(
    value: Union[Tensor, T, RawTensorTypes, list, tuple],
    *,
    dims: Sequence[Dim] = None,
    dtype: Optional[str] = None,
    sparse_dim: Optional[Dim] = None,
    feature_dim: Optional[Dim] = None,
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
    :param feature_dim:
    :param shape: alias for dims, for some older code
    :param name:
    :param device:
    :param keep_scalar_on_cpu: if the value is already on the CPU, keep it there, even if `device` is sth else
    :param _backend:
    :return: tensor
    """
    if isinstance(value, Tensor):  # fast path
        return value
    if isinstance(value, (tuple, list)):
        value = numpy.array(value, dtype=dtype)
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
        value=value, dims=dims, dtype=dtype, sparse_dim=sparse_dim, feature_dim=feature_dim, device=device, name=name
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
    if not dims:
        if out_dim:
            assert out_dim.dimension == 1
        else:
            out_dim = Dim(1, name="ext")
        return rf.expand_dim(source, out_dim), out_dim
    if len(dims) == 1:
        if out_dim is None or out_dim == dims[0]:
            return source, dims[0]
        return rf.replace_dim(source, in_dim=dims[0], out_dim=out_dim)
    if out_dim is None:
        from returnn.util.basic import prod

        if any(d.need_masking() for d in dims[1:]):
            # The dynamic sizes as calculated via dim math would not correctly describe how the tensor looks like.
            # This would then potentially discard some of the data in the tensor in subsequent operations,
            # when masking is applied.
            # Thus, discard the dynamic sizes, and just treat it as a flat dim with scalar dynamic size.
            # https://github.com/rwth-i6/returnn/issues/1694
            # See also similar logic in :func:`concat`.
            out_dim = Dim(prod(d.get_dim_value_tensor() for d in dims), name="merged")
        else:
            out_dim = prod(dims)
    # noinspection PyProtectedMember
    return source._raw_backend.merge_dims(source, dims=dims, out_dim=out_dim), out_dim


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


def expand_dims(source: Tensor, dims: Sequence[Dim]) -> Tensor:
    """
    Expand multiple dims, via :func:`expand_dim`.
    """
    for dim in dims:
        source = expand_dim(source, dim)
    return source


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
    use_mask: Optional[bool] = None,
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
    :param use_mask: whether we should mask to make sure the zero padding is correct
    :return: out, out_spatial_dim
    """
    if spatial_dim.need_masking():
        if use_mask is None:
            use_mask = rf.use_mask_default(default=True, default_false_for_behavior_version_up_to=22)
        if use_mask:
            source = source.copy_masked(0, dims=[spatial_dim])
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
    handle_dynamic_dims: Optional[bool] = None,
) -> Tuple[Tensor, Dim]:
    """
    Concatenates multiple sources in the specified dimension.

    Also see :func:`stack`.

    :param sources: list of (tensor, dim) pairs. dim is the axis to concatenate on.
    :param allow_broadcast: if True, the sources can have different dims, and the result will be broadcasted.
    :param out_dim: reuse existing dim for the resulting concatenated dim, if given
    :param handle_dynamic_dims:
    :return: concatenated tensor, out_dim
    """
    assert sources
    if not allow_broadcast:
        dims = sources[0][0].dims_set - {sources[0][1]}
        for src, dim in sources:
            assert src.dims_set - {dim} == dims, f"concat {sources}, need allow_broadcast=True"
    need_handle_dynamic_dims = False
    for src, dim in sources[:-1]:
        if dim.need_masking():
            need_handle_dynamic_dims = True
    if handle_dynamic_dims is None:
        handle_dynamic_dims = need_handle_dynamic_dims
    if not out_dim:
        if handle_dynamic_dims or not need_handle_dynamic_dims:
            out_dim = sum(d for _, d in sources)
        else:  # not handle_dynamic_dims but need_handle_dynamic_dims
            # There are dynamic dims, but we don't want to handle them.
            # So, summing the dims would be incorrect.
            # Just add the dim values.
            out_dim = Dim(sum(d.get_dim_value_tensor() for _, d in sources if d.dimension is not None), name="concat")
    if handle_dynamic_dims:
        out_non_masked_dim = Dim(sum(d.get_dim_value_tensor() for _, d in sources))
        # noinspection PyProtectedMember
        out = sources[0][0]._raw_backend.concat(*sources, allow_broadcast=allow_broadcast, out_dim=out_non_masked_dim)
        masks = []
        for _, dim in sources:
            masks.append(
                dim.get_mask(dim_order=(dim,) + dim.dyn_size_ext.dims, device=out.device)
                if dim.need_masking()
                else rf.constant(True, dims=[dim], device=out.device)
            )
        # noinspection PyProtectedMember
        mask_concat = sources[0][0]._raw_backend.concat(
            *[(mask, dim) for (_, dim), mask in zip(sources, masks)], allow_broadcast=True, out_dim=out_non_masked_dim
        )
        out, _ = rf.masked_select(out, mask=mask_concat, dims=[out_non_masked_dim], out_dim=out_dim)
    else:
        # noinspection PyProtectedMember
        out = sources[0][0]._raw_backend.concat(*sources, allow_broadcast=allow_broadcast, out_dim=out_dim)
    return out, out_dim


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
    padding: Sequence[Tuple[Union[Dim, int, Tensor], Union[Dim, int, Tensor]]],
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
    if handle_dynamic_dims is None:
        handle_dynamic_dims = _pad_handle_dynamic_dims_default(axes, padding, mode=mode)
    if not out_dims:
        out_dims = [
            (left + middle + right)
            if handle_dynamic_dims or not _pad_need_dyn_dim_handling(middle, left, right, mode=mode)
            else _pad_sum_dims_no_dyn_dim_handling(middle, left, right)
            for middle, (left, right) in zip(axes, padding)
        ]
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
            if not _pad_need_dyn_dim_handling(middle, left, right, mode=mode):
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


def _pad_need_dyn_dim_handling(
    middle: Dim, left: Union[Dim, int, Tensor], right: Union[Dim, int, Tensor], *, mode: str
) -> bool:
    if (
        not middle.need_masking()
        and (isinstance(left, int) or (isinstance(left, Dim) and not left.need_masking()))
        or (isinstance(left, Tensor) and not left.dims)
    ):
        return False
    if mode != "circular" and isinstance(right, int) and right == 0:
        return False
    return True


def _pad_sum_dims_no_dyn_dim_handling(
    middle: Dim, left: Union[Dim, int, Tensor], right: Union[Dim, int, Tensor]
) -> Dim:
    """
    This gets called when we need to handle dyn dims, but handle_dynamic_dims=False.
    See also the same logic in :func:`concat`.
    """
    if isinstance(left, Dim):
        left = left.get_dim_value_tensor()
    elif isinstance(left, int):
        pass
    elif isinstance(left, Tensor):
        assert left.dims == ()  # scalar
    else:
        raise TypeError(f"invalid left pad {left}")
    if isinstance(right, Dim):
        right = right.get_dim_value_tensor()
    elif isinstance(right, int):
        pass
    elif isinstance(right, Tensor):
        assert right.dims == ()  # scalar
    else:
        raise TypeError(f"invalid right pad {right}")
    return Dim(left + middle.get_dim_value_tensor() + right, name="pad")


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
    # Note: Before, we had a backend function just for this.
    # In case of TF-layers, this was using CumConcatLayer.
    # This would allow for automatic optimization when inside a RecLayer.
    # However, we don't really need this for eager frameworks,
    # and we want to simplify this for now,
    # using pure RF code.
    if not out_spatial_dim:
        out_spatial_dim = axis + 1
    out, (out_spatial_dim,) = rf.pad(
        prev_accum, axes=[axis], padding=[(0, 1)], out_dims=[out_spatial_dim], value=source, handle_dynamic_dims=True
    )
    return out, out_spatial_dim


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
    This will pack the tensor based on the mask.
    In TF, this is ``boolean_mask``.
    The inverse of this is :func:`masked_scatter`.

    Related: :func:`pack_padded`, which uses :func:`sequence_mask` as the mask.

    :param tensor:
    :param mask:
    :param dims: The same or a subset of the dims of ``mask``.
        The order of the dims defines the packed format (in ``out_dim``).
        If only a single dim is given, only this dim is packed, while all other dims are kept as is.
    :param out_dim: packed ``dims`` into this dim. Automatically created if not given, and then returned.
    :return: Tensor where all ``dims`` are removed and replaced by a new dim (``out_dim``).
        The new dim (``out_dim``) is also returned.
        If mask==True for all elements, the returned tensor would be simply the flattened input tensor.
    """
    mask_dims_set = set(mask.dims)
    dims_set = set(dims)
    if not dims_set.issubset(mask_dims_set):
        raise ValueError(f"masked_select: dims {dims} not subset of mask dims {mask.dims}")
    if not dims_set:
        raise ValueError(f"masked_select: dims {dims} empty")
    if dims_set == mask_dims_set:
        # noinspection PyProtectedMember
        return tensor._raw_backend.masked_select(tensor, mask=mask, dims=dims, out_dim=out_dim)
    # Separate implementation for the case where we have a subset of the mask dims, specifically one single dim.
    # See https://github.com/rwth-i6/returnn/issues/1605 for discussion.
    mask = mask.copy_masked(mask_value=False)
    if len(dims) > 1:
        # Flatten it, in the specified order.
        tensor, in_dim = rf.merge_dims(tensor, dims=dims)
        mask, _ = rf.merge_dims(mask, dims=dims, out_dim=in_dim)
    else:
        (in_dim,) = dims
    in_dim: Dim
    idxs = rf.cumsum(rf.cast(mask, "int32"), spatial_dim=in_dim)  # [T,B] -> idx in T' + 1
    new_size = rf.gather(idxs, indices=in_dim.get_dim_value_tensor() - 1, axis=in_dim)  # [B]
    if out_dim is None:
        out_dim = Dim(new_size, name="masked_select")
    elif out_dim.dyn_size_ext is None:
        out_dim.dyn_size_ext = new_size
    elif out_dim.dyn_size_ext is not None and out_dim.dyn_size_ext.raw_tensor is None:
        out_dim.dyn_size_ext.raw_tensor = new_size.raw_tensor
    new_time = rf.reduce_max(new_size, axis=new_size.dims)  # T'
    idxs = rf.where(mask, idxs - 1, new_time)  # new_time is the padding idx
    ext_out_dim = out_dim + 1  # one more for the padded data
    ext_res = rf.scatter(tensor, indices=idxs, indices_dim=in_dim, out_dim=ext_out_dim)
    res, _ = rf.slice(ext_res, axis=ext_out_dim, size=out_dim)
    return res, out_dim


def masked_scatter(
    source: Tensor, backup: Optional[Tensor] = None, *, mask: Tensor, dims: Sequence[Dim], in_dim: Dim
) -> Tensor:
    """
    The inverse of :func:`masked_select`.

    :param source: [in_dim, F...]
    :param backup: [dims..., F...] (or subset of those dims). zero if not given.
    :param mask: [dims...] -> bool (e.g. [B,T])
    :param dims: the order of the dims defines the format. those dims should be exactly the dims of the mask.
    :param in_dim: the dim of the source which should be scattered into the mask.
    :return: [dims..., F...]
    """
    mask_dims_set = set(mask.dims)
    dims_set = set(dims)
    if not dims_set.issubset(mask_dims_set):
        raise ValueError(f"masked_scatter: dims {dims} not subset of mask dims {mask.dims}")
    if not dims_set:
        raise ValueError(f"masked_scatter: dims {dims} empty")
    if dims_set == mask_dims_set:
        # noinspection PyProtectedMember
        return source._raw_backend.masked_scatter(source, backup=backup, mask=mask, dims=dims, in_dim=in_dim)
    # Separate implementation for the case where we have a subset of the mask dims.
    # Keep this consistent to masked_select above.
    mask = mask.copy_masked(mask_value=False)
    if len(dims) > 1:
        # Flatten it, in the specified order.
        mask_, dim_ = rf.merge_dims(mask, dims=dims)
    else:
        mask_ = mask
        (dim_,) = dims
    dim_: Dim
    idxs = rf.cumsum(rf.cast(mask_, "int32"), spatial_dim=dim_)  # [dim_] -> idx in dim_/in_dim + 1
    idxs = rf.split_dims(idxs, dims=dims, axis=dim_)  # [dims...]
    idxs = rf.where(mask, idxs - 1, 0)  # [dim_] -> idx in in_dim
    res = rf.gather(source, axis=in_dim, indices=idxs)
    if backup is None:
        backup = 0
    res = rf.where(mask, res, backup)
    return res


def sequence_mask(dims: Union[Dim, Sequence[Dim]], *, device: Optional[str] = None) -> Tensor:
    """
    :param dims:
    :param device:
    :return: mask based on the sequence lengths
    """
    if isinstance(dims, Dim):
        dims = [dims]
    assert len(dims) > 0
    dyn_dims = [d for d in dims if d.need_masking()]
    if not dyn_dims:
        return rf.constant(True, dims=())
    mask = True
    for dim in dyn_dims:
        mask = rf.opt_logical_and(mask, dim.get_mask(dim_order=dims, device=device))
    return mask


def pack_padded(
    source: Tensor, *, dims: Sequence[Dim], enforce_sorted: bool = False, out_dim: Optional[Dim] = None
) -> Tuple[Tensor, Dim]:
    """
    Like pack_padded_sequence. Usually the sequences are padded when they have different lengths.
    Packing means to only store the non-padded frames.
    This uses :func:`masked_select` internally based on the mask of non-masked frames.

    See :func:`pad_packed` for the inverse operation.

    :param source:
    :param dims: dims in source to pack. the order defines the format. first dim is major, etc.
        if there are no padded frames, e.g. dims=[B,T] would just result in the [B*T,...] reshaped tensor.
    :param enforce_sorted: seqs in the dims are reordered (stable sort) such that longest seqs come first.
    :param out_dim:
    :return: packed tensor, new packed dim
    """
    assert not enforce_sorted  # not implemented yet...
    mask = rf.sequence_mask(dims, device=source.device)
    # Note: We could already calculate out_dim here, as follows:
    #   out_dim = Dim(rf.num_elements_of_shape(dims), name="packed")
    # This could trigger a more efficient calculation path in masked_select,
    # where we can avoid a CUDA host-device synchronization, e.g. in the PyTorch backend.
    # However, in our benchmarks so far, it seems it's not helping so far,
    # so we don't do this, to also avoid the (minor) overhead of num_elements_of_shape here.
    # See: https://github.com/rwth-i6/returnn/pull/1593
    # This might change in the future when we have this:
    # https://github.com/pytorch/pytorch/issues/131256
    return rf.masked_select(source, mask=mask, dims=dims, out_dim=out_dim)


def pad_packed(source: Tensor, *, in_dim: Dim, dims: Sequence[Dim]) -> Tensor:
    """
    Inverse of :func:`pack_padded`, i.e. unpack the sequence, i.e. pad it back to the original length.
    """
    mask = rf.sequence_mask(dims, device=source.device)
    return rf.masked_scatter(source, mask=mask, in_dim=in_dim, dims=dims)


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
    mode: str = "sum",
    fill_value: Optional[Union[int, float]] = None,
    out_dim: Optional[Union[Dim, Sequence[Dim]]] = None,
    use_mask: Optional[bool] = None,
) -> Tensor:
    """
    Scatters into new zero-tensor.
    If entries in indices are duplicated, with ``mode=="sum"``,
    the corresponding values in source will be added together
    (``scatter_add`` in PyTorch),
    or otherwise it will take the respective reduction.

    ``scatter`` is the inverse of :func:`gather`.

    (segment_sum can be implemented via this.)

    :param source: [batch_dims..., indices_dim(s)..., feature_dims...]
    :param indices: [batch_dims..., indices_dim(s)...] -> out_dim
    :param indices_dim:
    :param mode: "sum", "max", "min", "mean", "logsumexp", "logmeanexp", "argmax".
        (Note: If you ever need another mode, please open an issue/PR.)
    :param fill_value:
    :param out_dim: The indices target dim.
        If not given, will be automatically determined as the sparse_dim from indices.
        If multiple out dims, use indices into the merged out dims,
        and then we use :func:`rf.split_dims` afterwards.
    :param use_mask:
    :return: [batch_dims..., out_dim(s)..., feature_dims...]
    """
    if mode == "mean":
        return scatter_mean(source, indices=indices, indices_dim=indices_dim, fill_value=fill_value, out_dim=out_dim)
    if mode == "logsumexp":
        return scatter_logsumexp(
            source, indices=indices, indices_dim=indices_dim, fill_value=fill_value, out_dim=out_dim
        )
    if mode == "logmeanexp":
        return scatter_logmeanexp(
            source, indices=indices, indices_dim=indices_dim, fill_value=fill_value, out_dim=out_dim
        )
    if mode == "argmax":
        return scatter_argmax(source, indices=indices, indices_dim=indices_dim, invalid_idx=fill_value, out_dim=out_dim)
    if not out_dim:
        assert isinstance(indices, Tensor) and indices.sparse_dim
        out_dim = indices.sparse_dim
    if fill_value is None:
        if mode == "sum":
            fill_value = 0
        elif mode == "max":
            if "int" in source.dtype:
                fill_value = numpy.iinfo(source.raw_tensor.dtype).min
            else:
                fill_value = float("-inf")
        elif mode == "min":
            if "int" in source.dtype:
                fill_value = numpy.iinfo(source.raw_tensor.dtype).max
            else:
                fill_value = float("inf")
        else:
            raise ValueError(f"scatter: invalid mode {mode!r}")
    indices_dim = indices_dim if isinstance(indices_dim, (list, tuple)) else [indices_dim]
    if any(dim.need_masking() for dim in indices_dim):
        if use_mask is None:
            use_mask = rf.use_mask_default(default=True, default_false_for_behavior_version_up_to=22)
        if use_mask:
            source = source.copy_masked(fill_value, dims=indices_dim)
    else:
        use_mask = False
    # noinspection PyProtectedMember
    out = source._raw_backend.scatter(
        source, indices=indices, indices_dim=indices_dim, mode=mode, fill_value=fill_value, out_dim=out_dim
    )
    if use_mask and mode != "sum":
        # Make sure we don't leave any infinities in the output.
        out = out.copy_masked(0, dims=[out_dim])
    return out


def scatter_mean(
    source: Tensor,
    *,
    indices: Tensor,
    indices_dim: Union[Dim, Sequence[Dim]],
    fill_value: Optional[Union[int, float]] = None,
    out_dim: Optional[Union[Dim, Sequence[Dim]]] = None,
) -> Tensor:
    """
    Scatters into new zero-tensor.
    If entries in indices are duplicated, the corresponding values in source will be mean'ed together.
    This is like :func:`scatter` with ``mode="mean"``.

    :param source: [batch_dims..., indices_dim(s)..., feature_dims...]
    :param indices: [batch_dims..., indices_dim(s)...] -> out_dim
    :param indices_dim:
    :param fill_value:
    :param out_dim: The indices target dim.
        If not given, will be automatically determined as the sparse_dim from indices.
        If multiple out dims, use indices into the merged out dims,
        and then we use :func:`rf.split_dims` afterwards.
    :return: [batch_dims..., out_dim(s)..., feature_dims...]
    """
    ones = rf.ones(dims=indices.dims, dtype=source.dtype, device=source.device)
    counts = rf.scatter(ones, indices=indices, indices_dim=indices_dim, fill_value=1, out_dim=out_dim)
    y = scatter(source, indices=indices, indices_dim=indices_dim, fill_value=fill_value, out_dim=out_dim)
    return y / counts


def scatter_argmax(
    source: Tensor,
    *,
    indices: Tensor,
    indices_dim: Union[Dim, Sequence[Dim]],
    invalid_idx: int = -1,
    out_dim: Optional[Union[Dim, Sequence[Dim]]] = None,
) -> Tensor:
    """
    Get the index in src which has the max value for each index in index.

    This is like :func:`scatter` with ``mode="argmax"``.

    :param source: [batch_dims..., indices_dim(s)..., feature_dims...]
    :param indices: [batch_dims..., indices_dim(s)...] -> out_dim
    :param indices_dim:
    :param invalid_idx: in case some of the output entries are never set (via ``indices``),
        this will be used as the value.
    :param out_dim: The indices target dim.
    :return: [batch_dims..., out_dim(s)..., feature_dims...]
    """
    import numpy

    if not out_dim:
        assert isinstance(indices, Tensor) and indices.sparse_dim
        out_dim = indices.sparse_dim

    # For the shape comments, use [B,I,F] for shorter source, [B,O,F] for shorter output.
    # use scatter to get the max value for each index
    out_max = rf.scatter(source, indices=indices, indices_dim=indices_dim, mode="max", out_dim=out_dim)  # [B,O,F]
    src_max = rf.gather(out_max, indices=indices, axis=out_dim)  # [B,I,F] -> max value or invalid_value

    max_invalid_idx = numpy.iinfo(indices.dtype).max

    # then use gather to get the max value back to src.
    # then mask the src with the max value.
    src_max_mask = src_max == source
    src_max_mask = src_max_mask.copy_masked(False)
    src_indices = rf.where(
        src_max_mask, rf.range_over_dim(indices_dim, dtype=indices.dtype, device=source.device), max_invalid_idx
    )  # [B,I,F] -> I

    # now scatter the min of src_indices into tensor
    out = rf.scatter(
        src_indices, indices=indices, indices_dim=indices_dim, mode="min", fill_value=invalid_idx, out_dim=out_dim
    )  # [B,O,F] -> I or invalid_idx or max_invalid_idx

    if max_invalid_idx != invalid_idx:
        out = rf.where(out != max_invalid_idx, out, invalid_idx)  # [B,O,F] -> I or invalid_idx
    return out


def scatter_logsumexp(
    source: Tensor,
    *,
    indices: Tensor,
    indices_dim: Union[Dim, Sequence[Dim]],
    fill_value: Optional[Union[int, float]] = None,
    out_dim: Optional[Union[Dim, Sequence[Dim]]] = None,
) -> Tensor:
    """
    Scatters into new zero-tensor.
    If entries in indices are duplicated, the corresponding values in source will be log-sum-exp'ed together.
    This is like :func:`scatter` with ``mode="logsumexp"``.

    :param source: [batch_dims..., indices_dim(s)..., feature_dims...]
    :param indices: [batch_dims..., indices_dim(s)...] -> out_dim
    :param indices_dim:
    :param fill_value:
    :param out_dim: The indices target dim.
        If not given, will be automatically determined as the sparse_dim from indices.
        If multiple out dims, use indices into the merged out dims,
        and then we use :func:`rf.split_dims` afterwards.
    :return: [batch_dims..., out_dim(s)..., feature_dims...]
    """
    if not out_dim:
        assert isinstance(indices, Tensor) and indices.sparse_dim
        out_dim = indices.sparse_dim
    with rf.stop_gradient_scope():
        max_x = rf.scatter(source, indices=indices, indices_dim=indices_dim, mode="max", out_dim=out_dim)  # [D_out,...]
        max_x_ = rf.gather(max_x, indices=indices, axis=out_dim)  # [D_src,...]
    src_ = rf.exp(source - max_x_)
    if fill_value is not None:
        fill_value = rf.exp(fill_value - max_x_)
    tensor = rf.scatter(
        src_, indices=indices, indices_dim=indices_dim, mode="sum", fill_value=fill_value, out_dim=out_dim
    )
    tensor = rf.log(tensor)
    tensor = rf.where(rf.is_neg_infinite(max_x), rf.zeros((), dtype=source.dtype, device=source.device), tensor)
    tensor += max_x
    return tensor


def scatter_logmeanexp(
    source: Tensor,
    *,
    indices: Tensor,
    indices_dim: Union[Dim, Sequence[Dim]],
    fill_value: Optional[Union[int, float]] = None,
    out_dim: Optional[Union[Dim, Sequence[Dim]]] = None,
) -> Tensor:
    """
    Scatters into new zero-tensor.
    If entries in indices are duplicated, the corresponding values in source will be log-mean-exp'ed together.
    This is like :func:`scatter` with ``mode="logmeanexp"``.

    :param source: [batch_dims..., indices_dim(s)..., feature_dims...]
    :param indices: [batch_dims..., indices_dim(s)...] -> out_dim
    :param indices_dim:
    :param fill_value:
    :param out_dim: The indices target dim.
        If not given, will be automatically determined as the sparse_dim from indices.
        If multiple out dims, use indices into the merged out dims,
        and then we use :func:`rf.split_dims` afterwards.
    :return: [batch_dims..., out_dim(s)..., feature_dims...]
    """
    ones = rf.ones(dims=indices.dims, dtype=source.dtype, device=source.device)
    counts = rf.scatter(ones, indices=indices, indices_dim=indices_dim, fill_value=1, out_dim=out_dim)
    y = scatter_logsumexp(source, indices=indices, indices_dim=indices_dim, fill_value=fill_value, out_dim=out_dim)
    return y - rf.log(counts)


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
    :param step: we only support a positive step here. If you want to flip the sequence,
        use :func:`reverse_sequence`, maybe with ``handle_dynamic_dims=False``.
    :param size:
    :param out_dim:
    :return: tensor, out_dim
    """
    if isinstance(step, int) and step <= 0:
        raise ValueError(f"rf.slice with non-positive step {step} not supported")
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


def shift_left(source: Tensor, *, axis: Dim, pad_value: Union[rf.RawTensorTypes, Tensor], amount: int = 1) -> Tensor:
    """shift left by amount, pad right with right_pad"""
    padded, (padded_dim,) = rf.pad(
        source, axes=[axis], padding=[(0, amount)], mode="constant", value=pad_value, handle_dynamic_dims=True
    )
    padded_slice, _ = rf.slice(padded, axis=padded_dim, start=amount, size=axis)
    return padded_slice


def reverse_sequence(tensor: Tensor, *, axis: Dim, handle_dynamic_dims: bool = True) -> Tensor:
    """
    Similar as tf.reverse_sequence, or Torch flip (but taking seq lengths into account).

    :param tensor:
    :param axis:
    :param handle_dynamic_dims: if True, the dynamic dims will be handled correctly,
        i.e. a padded seq "a b c <pad>" will be reversed to "c b a <pad>".
        If False, the dynamic dims will be handled as if they are static,
        i.e. a padded seq "a b c <pad>" will be reversed to "<pad> c b a".
    :return: reversed tensor, same dims
    """
    if not handle_dynamic_dims or not axis.need_masking():
        # noinspection PyProtectedMember
        return tensor._raw_backend.flip_no_mask(tensor, axis=axis)
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
    if not isinstance(cond, Tensor):
        backend = _utils.get_backend_from_tensors(true_, false_)
        cond = rf.convert_to_tensor(cond, _backend=backend)
    # noinspection PyProtectedMember
    return cond._raw_backend.where(cond, true_, false_, allow_broadcast_all_sources=allow_broadcast_all_sources)


def sort(source: Tensor, *, axis: Dim, descending: bool = False, stable: bool = True) -> Tuple[Tensor, Tensor, Dim]:
    """
    Sorts the source tensor along the given axis.

    See also :func:`top_k`.
    :func:`top_k` with ``k=axis.get_size_tensor()`` is equivalent to this function.

    :param source: {other_dims..., axis}
    :param axis: The axis to sort along.
    :param descending: If True, sort in descending order, otherwise in ascending order.
    :param stable: If True, use a stable sorting algorithm (not reordering equal elements).
        Note that many frameworks (Torch, TensorFlow) have ``stable=False`` by default.
        ``stable=False`` can be faster.
    :return: sorted tensor, indices tensor, out_dim. both tensors have the shape {other_dims..., out_dim},
        i.e. ``axis`` replaced by ``out_dim``.
        indices tensor has sparse_dim set to ``axis``.
    """
    # noinspection PyProtectedMember
    return source._raw_backend.sort(source, axis=axis, descending=descending, stable=stable)


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


def top_k_mask(values: Tensor, *, axis: Dim, k: Union[int, Tensor]) -> Tensor:
    """
    Top-k filtering.

    :param values: {other_dims..., axis}
    :param axis:
    :param k: the number of top values to keep
    :return: mask {other_dims..., axis} of the top-k values
    """
    _, indices, k_dim = rf.top_k(values, axis=axis, k=k)
    mask = rf.scatter(rf.full(dims=indices.dims, fill_value=True), indices=indices, indices_dim=k_dim, fill_value=False)
    return mask


def top_p_mask(
    probs: Tensor,
    *,
    axis: Dim,
    p: Union[float, Tensor],
    one_more: bool = True,
) -> Tensor:
    """
    Top-p filtering, e.g. as used in Nucleus sampling (https://arxiv.org/abs/1904.09751).

    :param probs: {probs_dims..., axis}
    :param axis:
    :param p: the probability mass to keep
    :param one_more: if True (default), keep also the first token above the threshold.
        (It's enabled by default to follow the behavior of the original implementation.)
    :return: mask {probs_dims..., axis} of the top-p tokens.
        ``sum(probs[mask]) <= p``, or slightly more if ``one_more`` is True.
    """
    assert 0.0 <= p <= 1.0
    if isinstance(p, Tensor):
        assert axis not in p.dims
    # https://github.com/ari-holtzman/degen/blob/master/gen.py
    sorted_probs, sorted_indices, sorted_dim = rf.sort(probs, axis=axis, descending=True)
    cum_probs = rf.cumsum(sorted_probs, spatial_dim=sorted_dim)
    mask = cum_probs <= p  # {probs_dims..., sorted_dim}
    if one_more:
        # keep also the first token above the threshold
        mask = rf.shift_right(mask, axis=sorted_dim, pad_value=True)
    mask = rf.scatter(mask, indices=sorted_indices, indices_dim=sorted_dim)
    return mask
