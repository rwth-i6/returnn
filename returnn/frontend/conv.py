"""
Convolution, transposed convolution, pooling
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple, Union
from returnn.util.basic import next_type_attrib_in_mro_chain
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


__all__ = [
    "conv",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "transposed_conv",
    "TransposedConv1d",
    "TransposedConv2d",
    "TransposedConv3d",
    "pool",
    "max_pool",
    "max_pool1d",
    "pool1d",
    "pool2d",
    "pool3d",
    "make_conv_out_spatial_dims",
]


# noinspection PyAbstractClass
class _ConvOrTransposedConv(rf.Module):
    """
    Base class for both convolution and transposed convolution.
    """

    nd: Optional[int] = None  # set in the subclasses, e.g. 1 for Conv1d, etc
    _transposed: bool  # set in the subclasses _Conv or _TransposedConv
    groups: Optional[int] = None

    def __init__(
        self,
        in_dim: Dim,
        out_dim: Dim,
        filter_size: Union[Sequence[Union[int, Dim]], int, Dim],
        *,
        padding: str,
        with_bias: bool,
    ):
        """
        :param Dim in_dim:
        :param Dim out_dim:
        :param filter_size: (width,), (height,width) or (depth,height,width) for 1D/2D/3D conv.
          the input data ndim must match, or you can add dimensions via input_expand_dims or input_add_feature_dim.
          it will automatically swap the batch-dim to the first axis of the input data.
        :param padding: "same" or "valid"
        :param with_bias:
        """
        super().__init__()
        assert isinstance(in_dim, Dim) and isinstance(out_dim, Dim)
        self.in_dim = rf.dim_match_priority_when_needed(in_dim, out_dim)
        self.out_dim = out_dim
        if isinstance(filter_size, (int, Dim)):
            if self.nd in (None, 1):
                filter_size = [filter_size]
            else:
                filter_size = [filter_size] * self.nd
        assert isinstance(filter_size, (tuple, list))
        if self.nd:
            assert self.nd == len(filter_size)
        else:
            self.nd = len(filter_size)
        self.filter_size = [
            s if isinstance(s, Dim) else Dim(s, name=f"filter-dim{i}") for i, s in enumerate(filter_size)
        ]
        self.padding = padding
        filter_in_dim = in_dim
        if self.groups is not None and self.groups > 1:
            assert not self._transposed  # not implemented
            filter_in_dim //= self.groups
        filter_in_dim = rf.dim_match_priority_when_needed(filter_in_dim, self.out_dim)
        self.filter_in_dim = filter_in_dim
        self.filter = rf.Parameter(
            ([self.out_dim, self.filter_in_dim] if not self._transposed else [self.in_dim, self.out_dim])
            + self.filter_size
        )
        self.filter.initial = rf.init.Glorot()
        self.with_bias = with_bias
        self.bias = None  # type: Optional[rf.Parameter]
        if self.with_bias:
            self.bias = rf.Parameter([self.out_dim])
            self.bias.initial = 0.0

    def _call_nd1(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        out_spatial_dim: Optional[Dim] = None,
    ) -> Tuple[Tensor, Dim]:
        assert self.nd == 1
        out, (out_spatial_dim,) = next_type_attrib_in_mro_chain(self.__class__, "__call__", self.__class__._call_nd1)(
            self,
            source,
            in_spatial_dims=[in_spatial_dim],
            out_spatial_dims=[out_spatial_dim] if out_spatial_dim else None,
        )
        return out, out_spatial_dim


class _Conv(_ConvOrTransposedConv):
    """
    A generic convolution layer which supports 1D, 2D and 3D convolution.
    Base class for :class:`Conv1d`, :class:`Conv2d`, :class:`Conv3d`.
    """

    _transposed = False

    # noinspection PyShadowingBuiltins,PyShadowingNames
    def __init__(
        self,
        in_dim: Dim,
        out_dim: Dim,
        filter_size: Union[Sequence[Union[int, Dim]], int, Dim],
        *,
        padding: str,
        strides: Optional[Union[int, Sequence[int]]] = None,
        dilation_rate: Optional[Union[int, Sequence[int]]] = None,
        groups: Optional[int] = None,
        with_bias: bool = True,
    ):
        """
        :param Dim in_dim:
        :param Dim out_dim:
        :param filter_size: (width,), (height,width) or (depth,height,width) for 1D/2D/3D conv.
          the input data ndim must match, or you can add dimensions via input_expand_dims or input_add_feature_dim.
          it will automatically swap the batch-dim to the first axis of the input data.
        :param str padding: "same" or "valid"
        :param int|Sequence[int] strides: strides for the spatial dims,
          i.e. length of this tuple should be the same as filter_size, or a single int.
        :param int|Sequence[int] dilation_rate: dilation for the spatial dims
        :param int groups: grouped convolution
        :param bool with_bias: if True, will add a bias to the output features
        """
        self.groups = groups
        super().__init__(in_dim=in_dim, out_dim=out_dim, filter_size=filter_size, padding=padding, with_bias=with_bias)
        if isinstance(strides, int):
            strides = [strides] * self.nd
        self.strides = strides
        self.dilation_rate = dilation_rate

    def __call__(
        self,
        source: Tensor,
        *,
        in_spatial_dims: Sequence[Dim],
        out_spatial_dims: Optional[Sequence[Dim]] = None,
    ) -> Tuple[Tensor, Sequence[Dim]]:
        return conv(
            source,
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            in_spatial_dims=in_spatial_dims,
            out_spatial_dims=out_spatial_dims,
            filter=self.filter,
            filter_size=self.filter_size,
            padding=self.padding,
            strides=self.strides,
            dilation_rate=self.dilation_rate,
            groups=self.groups,
            bias=self.bias if self.with_bias else None,
        )


# noinspection PyShadowingBuiltins
def conv(
    source: Tensor,
    *,
    in_dim: Dim,
    out_dim: Dim,
    in_spatial_dims: Sequence[Dim],
    out_spatial_dims: Optional[Sequence[Dim]] = None,
    filter: Tensor,
    filter_size: Sequence[Dim],
    padding: Union[str, int, Sequence[int]],
    strides: Optional[Union[int, Sequence[int]]] = None,
    dilation_rate: Optional[Union[int, Sequence[int]]] = None,
    groups: Optional[int] = None,
    bias: Optional[Tensor] = None,
    use_mask: Optional[bool] = None,
) -> Tuple[Tensor, Sequence[Dim]]:
    """
    Generic N-D convolution.

    :param source:
    :param in_dim: input channels
    :param out_dim: output channels
    :param in_spatial_dims: On what dimensions to operate on.
        The number of specified dims (1, 2 or 3) specifies whether this is 1D, 2D or 3D convolution.
        The order is consistent with the order of the ``filter_size``, ``strides``, etc.
    :param out_spatial_dims:
    :param filter:
    :param filter_size: defines the order of dims in ``filter``
        such that it matches the order of ``in_spatial_dims``.
    :param padding: "valid" or "same" or int. "valid" is like padding=0.
        padding="same" will pad such that the output has the same spatial dimensions as the input
        (in case of stride=1), or otherwise ceildiv(input, stride).
        The specific padding in padding="same" with stride>1 has changed with behavior version >=24
        (or global config option ``rf_use_consistent_same_padding``)
        and is now consistent independent of dimension size.
        See :func:`_consistent_same_padding` for more details.
    :param strides: the default (if it is None) is 1
    :param dilation_rate:
    :param groups:
    :param bias:
    :param use_mask: Whether to mask the input tensor based on seq lengths
        such that the padding in the padded tensor is ignored
        (it will mask with 0).
        With behavior version >=23, this is enabled by default,
        or configured with global config option ``rf_use_mask``.
        (Also see :func:`use_mask_default`).
    :return: out, out_spatial_dims
    """
    if any(in_spatial_dim.need_masking() for in_spatial_dim in in_spatial_dims):
        if use_mask is None:
            use_mask = rf.use_mask_default(default=True, default_false_for_behavior_version_up_to=22)
        if use_mask:
            source = source.copy_masked(0, dims=in_spatial_dims)
    for in_spatial_dim in in_spatial_dims:
        if in_spatial_dim not in source.dims:
            raise ValueError(f"conv: source {source} does not have spatial dim {in_spatial_dim}")
    if padding == "same" and _any_is_non_default(strides, default=1) and _should_use_consistent_same_padding():
        source, in_spatial_dims, padding = _consistent_same_padding(
            source, in_spatial_dims=in_spatial_dims, filter_size=filter_size, dilation_rate=dilation_rate, pad_value=0
        )
    # noinspection PyProtectedMember
    out, out_spatial_dims = source._raw_backend.conv(
        source,
        in_dim=in_dim,
        out_dim=out_dim,
        in_spatial_dims=in_spatial_dims,
        out_spatial_dims=out_spatial_dims,
        filter=filter,
        filter_size=filter_size,
        padding=padding,
        strides=strides,
        dilation_rate=dilation_rate,
        groups=groups,
        bias=bias,
    )
    return out, out_spatial_dims


class Conv1d(_Conv):
    """
    1D convolution
    """

    nd = 1

    def __init__(
        self,
        in_dim: Dim,
        out_dim: Dim,
        filter_size: Union[int, Dim],
        *,
        padding: str,
        strides: Optional[int] = None,
        dilation_rate: Optional[int] = None,
        groups: Optional[int] = None,
        with_bias: bool = True,
    ):
        """
        :param Dim in_dim:
        :param Dim out_dim:
        :param int|Dim filter_size:
        :param str padding: "same" or "valid"
        :param int|None strides: strides for the spatial dims,
          i.e. length of this tuple should be the same as filter_size, or a single int.
        :param int|None dilation_rate: dilation for the spatial dims
        :param int groups: grouped convolution
        :param bool with_bias: if True, will add a bias to the output features
        """
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            filter_size=[filter_size],
            padding=padding,
            strides=strides,
            dilation_rate=dilation_rate,
            groups=groups,
            with_bias=with_bias,
        )

    __call__ = _ConvOrTransposedConv._call_nd1


class Conv2d(_Conv):
    """
    2D convolution
    """

    nd = 2


class Conv3d(_Conv):
    """
    3D convolution
    """

    nd = 3


class _TransposedConv(_ConvOrTransposedConv):
    """
    Transposed convolution, sometimes also called deconvolution.
    See :func:`tf.nn.conv2d_transpose` (currently we support 1D/2D).
    """

    nd: Optional[int] = None
    _transposed = True

    # noinspection PyShadowingBuiltins,PyShadowingNames
    def __init__(
        self,
        in_dim: Dim,
        out_dim: Dim,
        filter_size: Sequence[Union[int, Dim]],
        *,
        padding: str,
        remove_padding: Union[Sequence[int], int] = 0,
        output_padding: Optional[Union[Sequence[Optional[int]], int]] = None,
        strides: Optional[Sequence[int]] = None,
        with_bias: bool = True,
    ):
        """
        :param Dim in_dim:
        :param Dim out_dim:
        :param list[int] filter_size:
        :param list[int]|None strides: specifies the upscaling. by default, same as filter_size
        :param str padding: "same" or "valid"
        :param list[int]|int remove_padding:
        :param list[int|None]|int|None output_padding:
        :param bool with_bias: whether to add a bias. enabled by default
        """
        super().__init__(in_dim=in_dim, out_dim=out_dim, filter_size=filter_size, padding=padding, with_bias=with_bias)
        self.strides = strides
        self.remove_padding = remove_padding
        self.output_padding = output_padding

    def __call__(
        self,
        source: Tensor,
        *,
        in_spatial_dims: Sequence[Dim],
        out_spatial_dims: Optional[Sequence[Dim]] = None,
    ) -> Tuple[Tensor, Sequence[Dim]]:
        return transposed_conv(
            source,
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            in_spatial_dims=in_spatial_dims,
            out_spatial_dims=out_spatial_dims,
            filter=self.filter,
            filter_size=self.filter_size,
            padding=self.padding,
            remove_padding=self.remove_padding,
            output_padding=self.output_padding,
            strides=self.strides,
            bias=self.bias if self.with_bias else None,
        )


# noinspection PyShadowingBuiltins
def transposed_conv(
    source: Tensor,
    *,
    in_dim: Dim,
    out_dim: Dim,
    in_spatial_dims: Sequence[Dim],
    out_spatial_dims: Optional[Sequence[Dim]] = None,
    filter: Tensor,
    filter_size: Sequence[Dim],
    padding: str,
    remove_padding: Union[Sequence[int], int] = 0,
    output_padding: Optional[Union[Sequence[Optional[int]], int]] = None,
    strides: Optional[Sequence[int]] = None,
    bias: Optional[Tensor] = None,
    use_mask: Optional[bool] = None,
) -> Tuple[Tensor, Sequence[Dim]]:
    """transposed conv"""
    if any(in_spatial_dim.need_masking() for in_spatial_dim in in_spatial_dims):
        if use_mask is None:
            use_mask = rf.use_mask_default(default=True, default_false_for_behavior_version_up_to=22)
        if use_mask:
            source = source.copy_masked(0, dims=in_spatial_dims)
    if padding == "same" and _any_is_non_default(strides, default=1) and _should_use_consistent_same_padding():
        # I don't really know what this should mean here... Investigate this further...
        raise NotImplementedError("consistent same padding not implemented for transposed conv")
    # noinspection PyProtectedMember
    out, out_spatial_dims = source._raw_backend.transposed_conv(
        source=source,
        in_dim=in_dim,
        out_dim=out_dim,
        in_spatial_dims=in_spatial_dims,
        out_spatial_dims=out_spatial_dims,
        filter=filter,
        filter_size=filter_size,
        padding=padding,
        remove_padding=remove_padding,
        output_padding=output_padding,
        strides=strides,
        bias=bias,
    )
    return out, out_spatial_dims


class TransposedConv1d(_TransposedConv):
    """
    1D transposed convolution
    """

    nd = 1

    __call__ = _ConvOrTransposedConv._call_nd1


class TransposedConv2d(_TransposedConv):
    """
    2D transposed convolution
    """

    nd = 2


class TransposedConv3d(_TransposedConv):
    """
    3D transposed convolution
    """

    nd = 3


def pool(
    source: Tensor,
    *,
    nd: Optional[int] = None,
    mode: str,
    pool_size: Union[Sequence[int], int],
    padding: Union[str, int, Sequence[int]] = "valid",
    dilation_rate: Union[Sequence[int], int] = 1,
    strides: Optional[Union[Sequence[int], int]] = None,
    in_spatial_dims: Union[Sequence[Dim], Dim],
    out_spatial_dims: Optional[Union[Sequence[Dim], Dim]] = None,
    use_mask: Optional[bool] = None,
) -> Tuple[Tensor, Sequence[Dim]]:
    """
    Generic N-D pooling.

    :param source:
    :param nd:
    :param mode: "max" or "avg"
    :param pool_size: shape of the window of each reduce
    :param padding: "valid" or "same" or int. "valid" is like padding=0.
        padding="same" will pad such that the output has the same spatial dimensions as the input
        (in case of stride=1), or otherwise ceildiv(input, stride).
        The specific padding in padding="same" with stride>1 has changed with behavior version >=24
        (or global config option ``rf_use_consistent_same_padding``)
        and is now consistent independent of dimension size.
        See :func:`_consistent_same_padding` for more details.
    :param dilation_rate:
    :param strides: the default (if it is None) will be set to pool_size (in contrast to :func:`conv`)
    :param in_spatial_dims:
    :param out_spatial_dims:
    :param use_mask: Whether to mask the input tensor based on seq lengths
        such that the padding in the padded tensor is ignored
        (for max-pooling, it will mask with -inf, for avg-pooling with 0).
        With behavior version >=23, this is enabled by default,
        or configured with global config option ``rf_use_mask``.
        (Also see :func:`use_mask_default`).
    :return: out, out_spatial_dims
    """
    if isinstance(in_spatial_dims, Dim):
        in_spatial_dims = [in_spatial_dims]
    assert isinstance(in_spatial_dims, (list, tuple))
    assert all(isinstance(d, Dim) for d in in_spatial_dims)
    if nd is None:
        nd = len(in_spatial_dims)
    else:
        assert nd == len(in_spatial_dims)
    if out_spatial_dims is not None:
        if isinstance(out_spatial_dims, Dim):
            out_spatial_dims = [out_spatial_dims]
    if isinstance(pool_size, int):
        pool_size = [pool_size] * nd
    assert isinstance(pool_size, (list, tuple))
    assert len(pool_size) == nd
    if not strides:
        strides = pool_size
    elif isinstance(strides, int):
        strides = [strides] * nd
    assert isinstance(strides, (list, tuple)) and len(strides) == nd and all(isinstance(s, int) for s in strides)

    if any(in_spatial_dim.need_masking() for in_spatial_dim in in_spatial_dims):
        if use_mask is None:
            use_mask = rf.use_mask_default(default=True, default_false_for_behavior_version_up_to=22)
        if use_mask:
            source = source.copy_masked({"max": float("-inf"), "avg": 0}[mode], dims=in_spatial_dims)
    else:
        use_mask = False

    if padding == "same" and _any_is_non_default(strides, default=1) and _should_use_consistent_same_padding():
        source, in_spatial_dims, padding = _consistent_same_padding(
            source,
            in_spatial_dims=in_spatial_dims,
            filter_size=pool_size,
            dilation_rate=dilation_rate,
            pad_value={"max": float("-inf"), "avg": 0}[mode],
        )

    # noinspection PyProtectedMember
    out, out_spatial_dims = source._raw_backend.pool(
        source=source,
        mode=mode,
        pool_size=pool_size,
        padding=padding,
        dilation_rate=dilation_rate,
        strides=strides,
        in_spatial_dims=in_spatial_dims,
        out_spatial_dims=out_spatial_dims,
    )
    if use_mask and mode == "max":
        # We masked with -inf for max-pooling to get correct pooling at the boundaries.
        # However, the resulting tensor might have -inf in it, and it is better to mask it out.
        out = out.copy_masked(0, dims=out_spatial_dims)
    return out, out_spatial_dims


def max_pool(
    source: Tensor,
    *,
    pool_size: Union[Sequence[int], int],
    padding: str = "valid",
    dilation_rate: Union[Sequence[int], int] = 1,
    strides: Optional[Union[Sequence[int], int]] = None,
    in_spatial_dims: Union[Sequence[Dim], Dim],
    out_spatial_dims: Optional[Union[Sequence[Dim], Dim]] = None,
) -> Tuple[Tensor, Sequence[Dim]]:
    """max-pool"""
    return pool(
        source=source,
        mode="max",
        pool_size=pool_size,
        padding=padding,
        dilation_rate=dilation_rate,
        strides=strides,
        in_spatial_dims=in_spatial_dims,
        out_spatial_dims=out_spatial_dims,
    )


def max_pool1d(
    source: Tensor,
    *,
    pool_size: int,
    padding: str = "valid",
    dilation_rate: int = 1,
    strides: Optional[int] = None,
    in_spatial_dim: Dim,
    out_spatial_dim: Optional[Dim] = None,
) -> Tuple[Tensor, Dim]:
    """max pool"""
    return pool1d(
        source=source,
        mode="max",
        pool_size=pool_size,
        padding=padding,
        dilation_rate=dilation_rate,
        strides=strides,
        in_spatial_dim=in_spatial_dim,
        out_spatial_dim=out_spatial_dim,
    )


def pool1d(
    source: Tensor,
    *,
    mode: str,
    pool_size: int,
    padding: str = "valid",
    dilation_rate: int = 1,
    strides: Optional[int] = None,
    in_spatial_dim: Dim,
    out_spatial_dim: Optional[Dim] = None,
) -> Tuple[Tensor, Dim]:
    """
    1D pooling.

    :param Tensor source:
    :param str mode: "max" or "avg"
    :param tuple[int] pool_size: shape of the window of each reduce
    :param str padding: "valid" or "same"
    :param tuple[int]|int dilation_rate:
    :param tuple[int]|int|None strides: in contrast to tf.nn.pool, the default (if it is None) will be set to pool_size
    :param Sequence[Dim] in_spatial_dim:
    :param Sequence[Dim]|None out_spatial_dim:
    :return: layer, out_spatial_dim
    """
    assert isinstance(in_spatial_dim, Dim)
    out, (out_spatial_dim,) = pool(
        source=source,
        mode=mode,
        pool_size=pool_size,
        padding=padding,
        dilation_rate=dilation_rate,
        strides=strides,
        in_spatial_dims=[in_spatial_dim],
        out_spatial_dims=[out_spatial_dim] if out_spatial_dim is not None else None,
    )
    return out, out_spatial_dim


def pool2d(
    source: Tensor,
    *,
    mode: str,
    pool_size: Union[Sequence[int], int],
    padding: str = "valid",
    dilation_rate: Union[Sequence[int], int] = 1,
    strides: Optional[Union[Sequence[int], int]] = None,
    in_spatial_dims: Sequence[Dim],
    out_spatial_dims: Optional[Sequence[Dim]] = None,
) -> Tuple[Tensor, Sequence[Dim]]:
    """
    2D pooling.

    :param Tensor source:
    :param str mode: "max" or "avg"
    :param tuple[int] pool_size: shape of the window of each reduce
    :param str padding: "valid" or "same"
    :param tuple[int]|int dilation_rate:
    :param tuple[int]|int|None strides: in contrast to tf.nn.pool, the default (if it is None) will be set to pool_size
    :param Sequence[Dim] in_spatial_dims:
    :param Sequence[Dim]|None out_spatial_dims:
    :return: layer, out_spatial_dims
    """
    assert len(in_spatial_dims) == 2
    return pool(
        source=source,
        mode=mode,
        pool_size=pool_size,
        padding=padding,
        dilation_rate=dilation_rate,
        strides=strides,
        in_spatial_dims=in_spatial_dims,
        out_spatial_dims=out_spatial_dims,
    )


def pool3d(
    source: Tensor,
    *,
    mode: str,
    pool_size: Union[Sequence[int], int],
    padding: str = "valid",
    dilation_rate: Union[Sequence[int], int] = 1,
    strides: Optional[Union[Sequence[int], int]] = None,
    in_spatial_dims: Sequence[Dim],
    out_spatial_dims: Optional[Sequence[Dim]] = None,
) -> Tuple[Tensor, Sequence[Dim]]:
    """
    3D pooling.

    :param Tensor source:
    :param str mode: "max" or "avg"
    :param tuple[int] pool_size: shape of the window of each reduce
    :param str padding: "valid" or "same"
    :param tuple[int]|int dilation_rate:
    :param tuple[int]|int|None strides: in contrast to tf.nn.pool, the default (if it is None) will be set to pool_size
    :param Sequence[Dim] in_spatial_dims:
    :param Sequence[Dim]|None out_spatial_dims:
    :return: layer, out_spatial_dims
    """
    assert len(in_spatial_dims) == 3
    return pool(
        source=source,
        mode=mode,
        pool_size=pool_size,
        padding=padding,
        dilation_rate=dilation_rate,
        strides=strides,
        in_spatial_dims=in_spatial_dims,
        out_spatial_dims=out_spatial_dims,
    )


def make_conv_out_spatial_dims(
    in_spatial_dims: Sequence[Dim],
    *,
    filter_size: Union[Sequence[Union[int, Dim]], int, Dim],
    padding: Union[str, int, Sequence[int]],
    strides: Union[Sequence[int], int] = 1,
    dilation_rate: Union[Sequence[int], int] = 1,
    description_prefix: Optional[str] = None,
) -> Sequence[Dim]:
    """create out spatial dims from in spatial dims"""
    nd = len(in_spatial_dims)
    if isinstance(filter_size, (int, Dim)):
        filter_size = [filter_size] * nd
    filter_size = [d.dimension if isinstance(d, Dim) else d for d in filter_size]
    assert all(isinstance(s, int) for s in filter_size)
    if isinstance(strides, int):
        strides = [strides] * nd
    if isinstance(dilation_rate, int):
        dilation_rate = [dilation_rate] * nd
    assert nd == len(in_spatial_dims) == len(filter_size) == len(strides) == len(dilation_rate)
    if isinstance(padding, (int, str)):
        padding = [padding] * nd
    padding = [p.lower() if isinstance(p, str) else p for p in padding]
    out_spatial_dims = []
    for i in range(nd):
        in_spatial_dim = in_spatial_dims[i]
        if (filter_size[i] == strides[i] == 1 and padding[i] in ("valid", "same", 0)) or (
            strides[i] == 1 and padding[i] == "same"
        ):
            out_spatial_dims.append(in_spatial_dim)
        else:
            out_spatial_dim = _calc_out_dim(
                in_dim=in_spatial_dim,
                filter_size=filter_size[i],
                stride=strides[i],
                dilation_rate=dilation_rate[i],
                padding=padding[i],
            )
            assert isinstance(out_spatial_dim, Dim)
            if description_prefix and out_spatial_dim != in_spatial_dim:
                out_spatial_dim.name = f"{description_prefix}:spatial{i}"
            if in_spatial_dim.dyn_size_ext is not None and out_spatial_dim.dyn_size_ext is None:
                out_spatial_dim.dyn_size_ext = _calc_out_dim(
                    in_dim=in_spatial_dim.dyn_size_ext,
                    filter_size=filter_size[i],
                    stride=strides[i],
                    dilation_rate=dilation_rate[i],
                    padding=padding[i],
                )
            out_spatial_dims.append(out_spatial_dim)
    return out_spatial_dims


def _calc_out_dim(in_dim, filter_size, stride, padding, dilation_rate=1):
    """
    Copied and adapted from TF ConvLayer.calc_out_dim.

    :param T|int|Tensor|torch.Tensor|tensorflow.Tensor|Dim in_dim: dimension in some axis
    :param int filter_size: e.g. 2, for the corresponding axis
    :param int stride: e.g. 1, for the corresponding axis
    :param int dilation_rate: e.g. 1
    :param str|int padding: "valid" or "same" or int
    :return: the output dimension
    :rtype: T
    """

    def ceildiv(a, b):
        """
        :param T|int|Tensor|torch.Tensor|tensorflow.Tensor a:
        :param T|int|Tensor|torch.Tensor|tensorflow.Tensor b:
        :rtype: T
        """
        if isinstance(b, int) and b == 1:
            return a
        if isinstance(a, Tensor):
            return rf.ceil_divide(a, b)
        return -(-a // b)

    padding = padding.lower() if isinstance(padding, str) else padding
    # See tf.compat.v1.nn.convolution() documentation for more.
    if padding == "same":
        if isinstance(in_dim, Dim):
            return in_dim.ceildiv_right(stride)
        return ceildiv(in_dim, stride)
    elif padding == "valid" or isinstance(padding, int):
        if isinstance(padding, int) and padding != 0:
            assert padding > 0
            in_dim = padding + in_dim + padding
        if isinstance(in_dim, Dim):
            filter_left_dilated = (filter_size - 1) * dilation_rate // 2
            filter_right_dilated = (filter_size - 1) * dilation_rate - filter_left_dilated
            valid_part = in_dim.sub_left(filter_left_dilated).sub_right(filter_right_dilated)
            return valid_part.ceildiv_right(stride)
        return ceildiv(in_dim - (filter_size - 1) * dilation_rate, stride)
    else:
        raise ValueError(f"invalid padding {padding!r} (type {type(padding).__name__})")


def _should_use_consistent_same_padding() -> bool:
    """
    :return: whether to use the new consistent same padding with :func:`_consistent_same_padding`.

    This is only needed for the case when we have striding and padding="same".
    See :func:`_consistent_same_padding` for more details.

    Check the global RETURNN config for the ``rf_use_consistent_same_padding``
    on how we should handle the ``padding="same"`` case for convolution/pooling when there is striding.
    If that is not specified, with behavior version >=24, we will use the new consistent same padding,
    with behavior version <=23, we will not use it.

    See issue `#1693 <https://github.com/rwth-i6/returnn/issues/1693>`__.
    """
    from returnn.config import get_global_config

    config = get_global_config(raise_exception=False)
    config_value = None
    if config:
        if "rf_use_consistent_same_padding" in config.typed_dict:
            config_value = config.typed_dict["rf_use_consistent_same_padding"]
            assert config_value is None or isinstance(config_value, bool)
        elif "rf_use_consistent_same_padding" in config.dict:
            config_value = config.bool("rf_use_consistent_same_padding", None)
    if config_value is not None:
        return config_value

    from returnn.util.basic import BehaviorVersion

    return BehaviorVersion.get() >= 24


def _consistent_same_padding(
    source: Tensor,
    *,
    in_spatial_dims: Sequence[Dim],
    filter_size: Optional[Union[int, Dim, Sequence[int], Sequence[Dim]]],
    dilation_rate: Optional[Union[int, Sequence[int]]] = None,
    pad_value: Union[int, float],
) -> Tuple[Tensor, Sequence[Dim], Union[int, Sequence[int]]]:
    """
    In case of striding and padding="same", the standard padding that we do (following TensorFlow)
    depends on the current dimension size.
    It adds padding left and right such that the first and last window
    will have the same amount of padding (+-1).
    With stride=1, this is the standard (filter_size-1)/2 left and right padding,
    but with stride>1, this is not the case anymore.
    (See also the explanation and calculation of padding in :func:`returnn.torch.frontend._backend.TorchBackend.conv`.)
    However, the problem with this behavior is with batching:
    The padding now depends on the longest sequence in the batch,
    and thus is arbitrary for any of the other sequences.

    The new consistent same padding adds padding independent of the current dimension size (largest seq in batch).
    We just do the same as with stride=1, i.e. (filter_size-1)/2 left and right padding.

    :return: source or padded source, in_spatial_dims or new in_spatial_dims, new padding on top of the output
    """
    filter_size = _make_sequence(filter_size or 1, nd=len(in_spatial_dims))
    dilation_rate = _make_sequence(dilation_rate or 1, nd=len(in_spatial_dims))
    filter_size_ints = [s.dimension if isinstance(s, Dim) else s for s in filter_size]
    if all(s % 2 == 1 for s in filter_size_ints):
        # In this case, we can pass padding as integer to the backend, so that it adds the same padding left/right.
        return source, in_spatial_dims, [(s // 2) * d for s, d in zip(filter_size_ints, dilation_rate)]
    # Need to use the custom padding here.
    paddings = []
    for s, d in zip(filter_size, dilation_rate):
        pad_left = (s - 1) * d // 2
        pad_right = (s - 1) * d - pad_left
        paddings.append((pad_left, pad_right))
    # We expect that masking was already done before (or we don't care about it), thus handle_dynamic_dims=False.
    out_dims = [(left + middle + right) for middle, (left, right) in zip(in_spatial_dims, paddings)]
    source, in_spatial_dims = rf.pad(
        source, axes=in_spatial_dims, padding=paddings, value=pad_value, handle_dynamic_dims=False, out_dims=out_dims
    )
    return source, in_spatial_dims, 0


def _make_sequence(value: Union[int, Sequence[int]], *, nd: int) -> Sequence[int]:
    if isinstance(value, int):
        return [value] * nd
    assert len(value) == nd
    return value


def _any_is_non_default(single_or_seq: Optional[Union[int, Sequence[int]]], *, default: int) -> bool:
    if single_or_seq is None:
        return False
    if isinstance(single_or_seq, int):
        return single_or_seq != default
    return any(i != default for i in single_or_seq)
