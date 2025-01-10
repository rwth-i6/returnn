"""
Conformer model, variant of Transformer with additional convolution, introduced for speech recognition.
Ref: https://arxiv.org/abs/2005.08100

About details of the specific implementation and other implementations, see:
https://github.com/rwth-i6/returnn_common/issues/233

V2: Split frontend and main encoder.
"""

from __future__ import annotations
from typing import Optional, Union, Any, Tuple, Dict, Callable
import copy as _copy
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.util.basic import NotSpecified
from .base import ISeqDownsamplingEncoder, ISeqFramewiseEncoder
from .conformer import ConformerConvSubsample, ConformerEncoderLayer


class ConformerFrontend(ISeqDownsamplingEncoder):
    """
    This is just the combination of:
    - input_layer (ConformerConvSubsample)
    - input_projection (Linear without bias)
    - input_embedding_scale
    - input_dropout

    This is intended to be used together with :class:`ConformerEncoderV2`.
    """

    def __init__(
        self,
        in_dim: Dim,
        out_dim: Dim = Dim(512, name="conformer-enc-default-out-dim"),
        *,
        input_layer: Optional[Union[ConformerConvSubsample, ISeqDownsamplingEncoder, rf.Module, Any]],
        input_embedding_scale: float = 1.0,
        input_dropout: float = 0.1,
    ):
        """
        :param in_dim: input features (e.g. MFCC)
        :param out_dim: the output feature dimension
        :param input_layer: input/frontend/prenet with potential subsampling.
            (x, in_spatial_dim) -> (y, out_spatial_dim)
        :param input_embedding_scale: applied after input_layer. 1.0 by default for historic reasons.
            In std Transformer, also ESPnet E-Branchformer and Conformer, this is sqrt(out_dim).
        :param input_dropout: applied after input_projection(input_layer(x))
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout_broadcast = rf.dropout_broadcast_default()

        # TODO once we figured out good defaults, we would create ConformerConvSubsample here when not given
        if callable(input_layer) or input_layer is None:
            pass  # leave it as is
        elif isinstance(input_layer, dict):
            input_layer = rf.build_from_dict(input_layer, in_dim)
            input_layer: ConformerConvSubsample  # maybe not true, but assume for some attribs
        else:
            raise TypeError(f"unexpected input_layer {input_layer!r}")
        self.input_layer = input_layer
        self.input_projection = (
            rf.Linear(self.input_layer.out_dim if self.input_layer else self.in_dim, self.out_dim, with_bias=False)
            if input_layer
            else None
        )
        self.input_embedding_scale = input_embedding_scale
        self.input_dropout = input_dropout

    def __call__(self, source: Tensor, *, in_spatial_dim: Dim) -> Tuple[Tensor, Dim]:
        """forward"""
        if self.input_layer:
            x_subsample, out_spatial_dim = self.input_layer(source, in_spatial_dim=in_spatial_dim)
        else:
            x_subsample, out_spatial_dim = source, in_spatial_dim
        x = self.input_projection(x_subsample) if self.input_projection else x_subsample
        if self.input_embedding_scale != 1.0:
            x = x * self.input_embedding_scale
        x = rf.dropout(x, self.input_dropout, axis=self.dropout_broadcast and self.out_dim)
        return x, out_spatial_dim


class ConformerEncoderV2(ISeqFramewiseEncoder):
    """
    Conformer encoder, without frontend.

    V2: Without the input_layer / frontend module, i.e. just the conformer layers.
    Use ConformerFrontend for the frontend.
    To get the V1 case, add this in front:
    """

    def __init__(
        self,
        out_dim: Dim = Dim(512, name="conformer-enc-default-out-dim"),
        *,
        num_layers: int,
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

        self.out_dim = out_dim
        self.dropout = dropout
        self.dropout_broadcast = rf.dropout_broadcast_default()

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

    def __call__(self, x: Tensor, *, spatial_dim: Dim, collected_outputs: Optional[Dict[str, Tensor]] = None) -> Tensor:
        """forward"""
        return self.layers(x, spatial_dim=spatial_dim, collected_outputs=collected_outputs)
