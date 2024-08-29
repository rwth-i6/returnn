"""
Conformer model, variant of Transformer with additional convolution, introduced for speech recognition.
Ref: https://arxiv.org/abs/2005.08100

About details of the specific implementation and other implementations, see:
https://github.com/rwth-i6/returnn_common/issues/233
"""

from __future__ import annotations
from typing import Optional, Union, Any, Tuple, List, Dict, Callable
import copy as _copy
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.util.basic import NotSpecified
from .base import ISeqDownsamplingEncoder
from ..decoder.transformer import FeedForward, make_norm


class ConformerPositionwiseFeedForward(FeedForward):
    """
    Conformer position-wise feedforward neural network layer
        FF -> Activation -> Dropout -> FF
    """

    def __init__(
        self,
        out_dim: Dim,
        *,
        ff_dim: Union[Dim, int] = NotSpecified,
        dropout: float = 0.1,
        activation: Union[Callable[[Tensor], Tensor], Dict[str, Any], rf.Module] = rf.swish,
        **kwargs,
    ):
        """
        :param out_dim: output feature dimension
        :param ff_dim: dimension of the feed-forward layers
        :param dropout: dropout value
        :param activation: activation function. swish by default, unlike the base :class:`FeedForward`
        """
        if activation is NotSpecified:
            activation = rf.swish
        super().__init__(out_dim=out_dim, ff_dim=ff_dim, dropout=dropout, activation=activation, **kwargs)


class ConformerConvBlock(rf.Module):
    """
    Conformer convolution block
        FF -> GLU -> depthwise conv -> BN -> Swish -> FF
    """

    def __init__(self, out_dim: Dim, *, kernel_size: int, norm: Union[rf.BatchNorm, Any]):
        """
        :param out_dim: output feature dimension
        :param kernel_size: kernel size of depthwise convolution
        :param norm: Batch norm originally
        """
        super().__init__()
        self.out_dim = out_dim

        self.positionwise_conv1 = rf.Linear(out_dim, 2 * out_dim)
        self.depthwise_conv = rf.Conv1d(
            out_dim, out_dim, filter_size=kernel_size, groups=out_dim.dimension, padding="same"
        )
        self.positionwise_conv2 = rf.Linear(out_dim, out_dim)
        if not callable(norm):
            raise TypeError(f"{self}: unexpected norm type {norm!r}")
        self.norm = norm

    def __call__(self, inp: Tensor, *, spatial_dim: Dim) -> Tensor:
        """forward"""
        x_conv1 = self.positionwise_conv1(inp)
        x_act, _ = rf.gating(x_conv1)
        x_depthwise_conv, _ = self.depthwise_conv(x_act, in_spatial_dim=spatial_dim)
        x_normed = self.norm(x_depthwise_conv)
        x_swish = rf.swish(x_normed)
        x_conv2 = self.positionwise_conv2(x_swish)
        return x_conv2


class ConformerConvSubsample(ISeqDownsamplingEncoder):
    """
    Conv 2D block with optional max-pooling or striding.

    References:

      https://github.com/espnet/espnet/blob/4138010fb66ad27a43e8bee48a4932829a0847ae/espnet/nets/pytorch_backend/transformer/subsampling.py#L162
      https://github.com/rwth-i6/returnn-experiments/blob/5852e21f44d5450909dee29d89020f1b8d36aa68/2022-swb-conformer-hybrid-sat/table_1_and_2/reduced_dim.config#L226
      (actually the latter is different...)

    To get the ESPnet case, for example Conv2dSubsampling6, use these options
    (out_dim is the model dim of the encoder)

      out_dims=[out_dim, out_dim],  # ESPnet standard, but this might be too large
      filter_sizes=[3, 5],
      strides=[2, 3],
      padding="valid",
    """

    def __init__(
        self,
        in_dim: Dim,
        *,
        out_dims: List[Dim],
        filter_sizes: List[Union[int, Tuple[int, int]]],
        strides: Optional[List[Union[int, Tuple[int, int]]]] = None,
        pool_sizes: Optional[List[Tuple[int, int]]] = None,
        activation: Callable[[Tensor], Tensor] = rf.relu,
        padding: str = "same",
    ):
        """
        :param out_dims: the number of output channels. last element is the output feature dimension
        :param filter_sizes: a list of filter sizes for the conv layer
        :param pool_sizes: a list of pooling factors applied after conv layer
        :param activation: the activation function
        :param padding: 'same' or 'valid'
        """
        super().__init__()

        self.pool_sizes = pool_sizes
        self.activation = activation

        self.conv_layers: rf.ModuleList[rf.Conv2d] = rf.ModuleList()
        if strides is None:
            strides = [1] * len(out_dims)
        assert len(out_dims) == len(filter_sizes) == len(strides) > 0
        self._dummy_in_dim = Dim(1, name="dummy-input-feature-dim")
        self.in_dim = in_dim
        prev_out_dim = self._dummy_in_dim
        second_spatial_dim = in_dim
        for i, (filter_size, stride, out_dim) in enumerate(zip(filter_sizes, strides, out_dims)):
            conv = rf.Conv2d(prev_out_dim, out_dim, filter_size=filter_size, strides=stride, padding=padding)
            self.conv_layers.append(conv)
            (second_spatial_dim,) = rf.make_conv_out_spatial_dims(
                [second_spatial_dim], filter_size=conv.filter_size[1], strides=conv.strides[1], padding=padding
            )
            if self.pool_sizes and i < len(self.pool_sizes):
                (second_spatial_dim,) = rf.make_conv_out_spatial_dims(
                    [second_spatial_dim],
                    filter_size=self.pool_sizes[i][1],
                    strides=self.pool_sizes[i][1],
                    padding="same",
                )
            prev_out_dim = out_dim
        self._final_second_spatial_dim = second_spatial_dim
        self.out_dim = second_spatial_dim * prev_out_dim

    def __call__(self, source: Tensor, *, in_spatial_dim: Dim) -> Tuple[Tensor, Dim]:
        """forward"""
        assert self.in_dim in source.dims
        in_spatial_dims = [in_spatial_dim, self.in_dim]
        in_dim = self._dummy_in_dim
        x = rf.expand_dim(source, dim=in_dim)
        for i, conv_layer in enumerate(self.conv_layers):
            x, in_spatial_dims = conv_layer(x, in_spatial_dims=in_spatial_dims)
            in_dim = conv_layer.out_dim
            x = self.activation(x)
            if self.pool_sizes and i < len(self.pool_sizes):
                x, in_spatial_dims = rf.pool2d(
                    x, in_spatial_dims=in_spatial_dims, pool_size=self.pool_sizes[i], padding="same", mode="max"
                )
        x, in_spatial_dims[-1] = rf.replace_dim(x, out_dim=self._final_second_spatial_dim, in_dim=in_spatial_dims[-1])
        out, _ = rf.merge_dims(x, dims=[self._final_second_spatial_dim, in_dim])
        return out, in_spatial_dims[0]


class ConformerEncoderLayer(rf.Module):
    """
    Represents a conformer block
    """

    def __init__(
        self,
        out_dim: Dim = Dim(512, name="conformer-enc-default-out-dim"),
        *,
        ff: Union[type, Dict[str, Any], rf.Module] = NotSpecified,
        ff_dim: Dim = NotSpecified,
        ff_activation: Union[Callable[[Tensor], Tensor], Dict[str, Any], rf.Module] = NotSpecified,
        dropout: float = 0.1,
        conv_kernel_size: int = 32,
        conv_norm: Union[rf.BatchNorm, type, Dict[str, Any], Any] = NotSpecified,
        conv_norm_opts: Optional[Dict[str, Any]] = None,
        num_heads: int = 4,
        self_att: Optional[Union[rf.RelPosSelfAttention, rf.Module, type, Dict[str, Any], Any]] = None,
        self_att_opts: Optional[Dict[str, Any]] = None,
        att_dropout: float = 0.1,
        norm: Union[type, Dict[str, Any], rf.Module, Callable] = rf.LayerNorm,
    ):
        """
        :param out_dim: the output feature dimension
        :param ff_dim: the dimension of feed-forward layers. 2048 originally, or 4 times out_dim
        :param ff_activation: activation function for feed-forward network
        :param dropout: the dropout value for the FF block
        :param conv_kernel_size: the kernel size of depthwise convolution in the conv block
        :param conv_norm: used for the conv block. Batch norm originally
        :param conv_norm_opts: for nn.BatchNorm or other conv_norm type.
          In case of nn.BatchNorm, uses use_mask=False by default.
            use_mask means whether to properly mask the spatial dim in batch norm.
            Most existing implementations don't do this. Except of RETURNN.
            It's faster when you don't do this.
        :param num_heads: the number of attention heads
        :param self_att: the self-attention layer. RelPosSelfAttention originally and default
        :param self_att_opts: options for the self-attention layer, for :class:`nn.RelPosSelfAttention`
        :param att_dropout: attention dropout value
        :param norm: pre-normalization for FF, conv and attention blocks
        """
        super().__init__()

        self.dropout = dropout
        self.dropout_broadcast = rf.dropout_broadcast_default()
        self.out_dim = out_dim

        self.ffn1 = make_ff(ff=ff, out_dim=out_dim, ff_dim=ff_dim, dropout=dropout, ff_activation=ff_activation)
        self.ffn1_layer_norm = make_norm(norm, out_dim)

        self.ffn2 = make_ff(ff=ff, out_dim=out_dim, ff_dim=ff_dim, dropout=dropout, ff_activation=ff_activation)
        self.ffn2_layer_norm = make_norm(norm, out_dim)

        if conv_norm is NotSpecified or conv_norm is rf.BatchNorm:
            conv_norm_opts = conv_norm_opts.copy() if conv_norm_opts else {}
            conv_norm_opts.setdefault("use_mask", False)
            conv_norm = rf.BatchNorm(out_dim, **conv_norm_opts)
        elif isinstance(conv_norm, type):
            conv_norm = conv_norm(out_dim, **(conv_norm_opts or {}))
        elif isinstance(conv_norm, dict):
            conv_norm = rf.build_from_dict(conv_norm, out_dim, **(conv_norm_opts or {}))
        if not callable(conv_norm):
            raise TypeError(f"{self}: unexpected conv_norm type {conv_norm!r}")
        self.conv_block = ConformerConvBlock(out_dim=out_dim, kernel_size=conv_kernel_size, norm=conv_norm)
        self.conv_layer_norm = make_norm(norm, out_dim)

        if self_att is None or isinstance(self_att, (dict, type)):
            self_att_opts_ = dict(
                in_dim=out_dim,
                proj_dim=out_dim,
                key_dim_total=out_dim,
                value_dim_total=out_dim,
                num_heads=num_heads,
                att_dropout=att_dropout,
            )
            if self_att_opts:
                self_att_opts_.update(self_att_opts)
            if self_att is None:
                self.self_att = rf.RelPosSelfAttention(**self_att_opts_)
            elif isinstance(self_att, type):
                self.self_att = self_att(**self_att_opts_)
            elif isinstance(self_att, dict):
                self_att_opts_ = {k: v for (k, v) in self_att_opts_.items() if k not in self_att}
                self.self_att = rf.build_from_dict(self_att, **self_att_opts_)
            else:
                raise TypeError(f"{self}: invalid type: self_att {self_att!r}")
        else:
            if not callable(self_att):
                raise TypeError(f"{self}: invalid non-callable: self_att {self_att!r}")
            self.self_att = self_att
        self.self_att_layer_norm = make_norm(norm, out_dim)

        self.final_layer_norm = make_norm(norm, out_dim)

    def __call__(self, inp: Tensor, *, spatial_dim: Dim) -> Tensor:
        """forward"""
        # FFN
        x_ffn1_ln = self.ffn1_layer_norm(inp)
        x_ffn1 = self.ffn1(x_ffn1_ln)
        x_ffn1_out = 0.5 * rf.dropout(x_ffn1, self.dropout, axis=self.dropout_broadcast and self.out_dim) + inp

        # MHSA
        x_mhsa_ln = self.self_att_layer_norm(x_ffn1_out)
        x_mhsa = self.self_att(x_mhsa_ln, axis=spatial_dim)
        x_mhsa = rf.dropout(x_mhsa, self.dropout, axis=self.dropout_broadcast and self.out_dim)
        x_mhsa_out = x_mhsa + x_ffn1_out

        # Conv
        x_conv_ln = self.conv_layer_norm(x_mhsa_out)
        x_conv = self.conv_block(x_conv_ln, spatial_dim=spatial_dim)
        x_conv_out = rf.dropout(x_conv, self.dropout, axis=self.dropout_broadcast and self.out_dim) + x_mhsa_out

        # FFN
        x_ffn2_ln = self.ffn2_layer_norm(x_conv_out)
        x_ffn2 = self.ffn2(x_ffn2_ln)
        x_ffn2_out = 0.5 * rf.dropout(x_ffn2, self.dropout, axis=self.dropout_broadcast and self.out_dim) + x_conv_out

        # last LN layer
        return self.final_layer_norm(x_ffn2_out)


class ConformerEncoder(ISeqDownsamplingEncoder):
    """
    Represents Conformer encoder architecture
    """

    def __init__(
        self,
        in_dim: Dim,
        out_dim: Dim = Dim(512, name="conformer-enc-default-out-dim"),
        *,
        num_layers: int,
        input_layer: Optional[Union[ConformerConvSubsample, ISeqDownsamplingEncoder, rf.Module, Any]],
        input_embedding_scale: float = 1.0,
        input_dropout: float = 0.1,
        ff_dim: Dim = NotSpecified,
        ff_activation: Union[Callable[[Tensor], Tensor], Dict[str, Any], rf.Module] = NotSpecified,
        dropout: float = 0.1,
        conv_kernel_size: int = NotSpecified,
        conv_norm: Union[rf.BatchNorm, type, Dict[str, Any], Any] = NotSpecified,
        num_heads: int = 4,
        att_dropout: float = 0.1,
        encoder_layer: Optional[Union[ConformerEncoderLayer, rf.Module, type, Dict[str, Any], Any]] = None,
        encoder_layer_opts: Optional[Dict[str, Any]] = None,
        sequential=rf.Sequential,
    ):
        """
        :param out_dim: the output feature dimension
        :param num_layers: the number of encoder layers
        :param input_layer: input/frontend/prenet with potential subsampling.
            (x, in_spatial_dim) -> (y, out_spatial_dim)
        :param input_embedding_scale: applied after input_layer. 1.0 by default for historic reasons.
            In std Transformer, also ESPnet E-Branchformer and Conformer, this is sqrt(out_dim).
        :param input_dropout: applied after input_projection(input_layer(x))
        :param ff_dim: the dimension of feed-forward layers. 2048 originally, or 4 times out_dim
        :param ff_activation: activation function for feed-forward network
        :param dropout: the dropout value for the FF block
        :param conv_kernel_size: the kernel size of depthwise convolution in the conv block
        :param conv_norm: used for the conv block. Batch norm originally
        :param num_heads: the number of attention heads
        :param att_dropout: attention dropout value
        :param encoder_layer: an instance of :class:`ConformerEncoderLayer` or similar
        :param encoder_layer_opts: options for the encoder layer
        :param sequential:
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.dropout_broadcast = rf.dropout_broadcast_default()

        # TODO once we figured out good defaults, we would create ConformerConvSubsample here when not given
        self.input_layer = input_layer
        self.input_projection = (
            rf.Linear(self.input_layer.out_dim if self.input_layer else self.in_dim, self.out_dim, with_bias=False)
            if input_layer
            else None
        )
        self.input_embedding_scale = input_embedding_scale
        self.input_dropout = input_dropout

        if not encoder_layer or isinstance(encoder_layer, (dict, type)):
            encoder_layer_opts_ = dict(
                out_dim=out_dim,
                ff_dim=ff_dim,
                ff_activation=ff_activation,
                dropout=dropout,
                conv_kernel_size=conv_kernel_size,
                conv_norm=conv_norm,
                num_heads=num_heads,
                att_dropout=att_dropout,
            )
            encoder_layer_opts_ = {k: v for (k, v) in encoder_layer_opts_.items() if v is not NotSpecified}
            if encoder_layer_opts:
                encoder_layer_opts_.update(encoder_layer_opts)
            if not encoder_layer:
                encoder_layer = ConformerEncoderLayer(**encoder_layer_opts_)
            elif isinstance(encoder_layer, type):
                encoder_layer = encoder_layer(**encoder_layer_opts_)
            elif isinstance(encoder_layer, dict):
                # Note: Reuse all the encoder_layer_opts_.
                # If this does not make sense for the specific encoder_layer class here,
                # we would suggest to use a different ConformerEncoder class.
                # (The alternative, to not reuse encoder_layer_opts_ here,
                #  would probably be more confusing, as those options are all ignored then.
                #  It's also not clear what args to pass then and what not.)
                # (Maybe we should do a ConformerEncoderV2 if this is confusing here...)
                encoder_layer_opts_ = {k: v for (k, v) in encoder_layer_opts_.items() if k not in encoder_layer}
                encoder_layer = rf.build_from_dict(encoder_layer, **encoder_layer_opts_)
            else:
                raise TypeError(f"unexpected encoder_layer {encoder_layer!r}")
        else:
            if not callable(encoder_layer):
                raise TypeError(f"{self}: invalid non-callable encoder_layer {encoder_layer!r}")

        self.layers = sequential(_copy.deepcopy(encoder_layer) for _ in range(num_layers))

    def __call__(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Dim]:
        """forward"""
        if self.input_layer:
            x_subsample, out_spatial_dim = self.input_layer(source, in_spatial_dim=in_spatial_dim)
        else:
            x_subsample, out_spatial_dim = source, in_spatial_dim
        x = self.input_projection(x_subsample) if self.input_projection else x_subsample
        if self.input_embedding_scale != 1.0:
            x = x * self.input_embedding_scale
        x = rf.dropout(x, self.input_dropout, axis=self.dropout_broadcast and self.out_dim)
        x = self.layers(x, spatial_dim=out_spatial_dim, collected_outputs=collected_outputs)
        return x, out_spatial_dim


def make_ff(
    *,
    out_dim: Dim,
    ff: Union[type, Dict[str, Any], rf.Module],
    ff_dim: Union[Dim, int],
    ff_activation: Union[Callable[[Tensor], Tensor], Dict[str, Any], rf.Module],
    dropout: float,
) -> Union[ConformerPositionwiseFeedForward, rf.Module]:
    """
    make the feed-forward part of the Conformer layer
    """
    if ff is NotSpecified:
        ff = ConformerPositionwiseFeedForward
    if isinstance(ff, rf.Module):
        ff = _copy.deepcopy(ff)
    else:
        ff_kwargs = dict(out_dim=out_dim, ff_dim=ff_dim, dropout=dropout, activation=ff_activation)
        ff_kwargs = {k: v for (k, v) in ff_kwargs.items() if v is not NotSpecified}
        if isinstance(ff, type):
            ff = ff(**ff_kwargs)
        elif isinstance(ff, dict):
            ff = rf.build_from_dict(ff, **ff_kwargs)
        else:
            raise TypeError(f"unexpected ff type {ff!r}")
    assert isinstance(ff, rf.Module)
    return ff
