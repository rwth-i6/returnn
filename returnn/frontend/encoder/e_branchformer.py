"""
E-Branchformer (https://arxiv.org/pdf/2210.00077)

Example usage:

.. code-block:: python

    import returnn.frontend as rf
    from returnn.frontend.encoder.conformer import ConformerEncoder
    from returnn.frontend.encoder.e_branchformer import EBranchformerLayer

    model = ConformerEncoder(
        out_dim=...,  # model dim, output_size in ESPnet
        num_layers=...,  # num_blocks in ESPnet
        encoder_layer=rf.build_dict(
            EBranchformerLayer,
            ff_dim=...,  # linear_units in ESPnet
            num_heads=...,  # attention_heads in ESPnet
            cgmlp_ff_dim=...,  # half of cgmlp_linear_units in ESPnet
            cgmlp_conv_kernel=...,  # cgmlp_conv_kernel in ESPnet
            merge_conv_kernel=...,  # merge_conv_kernel in ESPnet
        ),
    )

"""

from __future__ import annotations
from typing import Union, Any, Callable, Dict
import copy as _copy
from returnn.util.basic import NotSpecified
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from ..decoder.transformer import make_norm
from .conformer import make_ff


__all__ = ["EBranchformerLayer", "FeedForwardConvGated", "Merge"]


class EBranchformerLayer(rf.Module):
    """
    E-Branchformer layer, e.g. to be used in the :class:`returnn.frontend.encoder.conformer.ConformerEncoder`.

    See the module docstring :mod:`returnn.frontend.encoder.e_branchformer` for an example.
    """

    def __init__(
        self,
        out_dim: Dim = Dim(512, name="conformer-enc-default-out-dim"),
        *,
        ff: Union[type, Dict[str, Any], rf.Module] = NotSpecified,
        ff_dim: Union[Dim, int] = NotSpecified,
        ff_activation: Union[Callable[[Tensor], Tensor], Dict[str, Any], rf.Module] = NotSpecified,
        dropout: float = 0.1,
        num_heads: int = 4,
        self_att: Union[rf.RelPosSelfAttention, rf.Module, type, Dict[str, Any], Any] = NotSpecified,
        att_dropout: float = 0.1,
        cgmlp: Union[type, Dict[str, Any]] = NotSpecified,
        cgmlp_ff_dim: Union[Dim, int] = NotSpecified,
        cgmlp_conv_kernel: int = 31,
        merge_conv_kernel: int = 3,
        norm: Union[type, Dict[str, Any], rf.Module, Callable] = rf.LayerNorm,
    ):
        super().__init__()

        self.dropout = dropout
        self.dropout_broadcast = rf.dropout_broadcast_default()
        self.out_dim = out_dim

        self.ffn1 = make_ff(ff=ff, out_dim=out_dim, ff_dim=ff_dim, dropout=dropout, ff_activation=ff_activation)
        self.ffn1_layer_norm = make_norm(norm, out_dim)

        self.ffn2 = make_ff(ff=ff, out_dim=out_dim, ff_dim=ff_dim, dropout=dropout, ff_activation=ff_activation)
        self.ffn2_layer_norm = make_norm(norm, out_dim)

        if self_att is NotSpecified or isinstance(self_att, (dict, type)):
            self_att_opts_ = dict(
                in_dim=out_dim,
                proj_dim=out_dim,
                key_dim_total=out_dim,
                value_dim_total=out_dim,
                num_heads=num_heads,
                att_dropout=att_dropout,
            )
            self_att_opts_ = {k: v for (k, v) in self_att_opts_.items() if v is not NotSpecified}
            if self_att is NotSpecified:
                self.self_att = rf.RelPosSelfAttention(**self_att_opts_)
            elif isinstance(self_att, type):
                self.self_att = self_att(**self_att_opts_)
            elif isinstance(self_att, dict):
                self_att_opts_ = {k: v for (k, v) in self_att_opts_.items() if k not in self_att}
                self.self_att = rf.build_from_dict(self_att, **self_att_opts_)
            else:
                raise TypeError(f"{self}: invalid type: self_att {self_att!r}")
        elif isinstance(self_att, rf.Module):
            self.self_att = _copy.deepcopy(self_att)
        else:
            raise TypeError(f"{self}: invalid self_att {self_att!r} type {type(self_att).__name__}")
        self.self_att_layer_norm = make_norm(norm, out_dim)

        if cgmlp is NotSpecified:
            cgmlp = FeedForwardConvGated
        cgmlp_opts = {"ff_dim": cgmlp_ff_dim, "kernel_size": cgmlp_conv_kernel, "dropout": dropout}
        cgmlp_opts = {k: v for (k, v) in cgmlp_opts.items() if v is not NotSpecified}
        if isinstance(cgmlp, type):
            cgmlp = cgmlp(out_dim, **cgmlp_opts)
        elif isinstance(cgmlp, dict):
            cgmlp_opts = {k: v for (k, v) in cgmlp_opts.items() if k not in cgmlp}
            cgmlp = rf.build_from_dict(cgmlp, out_dim, **cgmlp_opts)
        elif isinstance(cgmlp, rf.Module):
            cgmlp = _copy.deepcopy(cgmlp)
        if not isinstance(cgmlp, rf.Module):
            raise TypeError(f"unexpected cgmlp {cgmlp!r} type {type(cgmlp).__name__}")
        self.cgmlp: Union[FeedForwardConvGated, rf.Module] = cgmlp
        self.cgmlp_layer_norm = make_norm(norm, out_dim)

        self.merge = Merge(in_dim1=out_dim, in_dim2=out_dim, out_dim=out_dim, merge_conv_kernel=merge_conv_kernel)

        self.final_layer_norm = make_norm(norm, out_dim)

    def __call__(self, inp: Tensor, *, spatial_dim: Dim) -> Tensor:
        """forward"""
        # FFN
        x_ffn1_ln = self.ffn1_layer_norm(inp)
        x_ffn1 = self.ffn1(x_ffn1_ln)
        x_ffn1_out = 0.5 * rf.dropout(x_ffn1, self.dropout, axis=self.dropout_broadcast and self.out_dim) + inp

        # Branch 1: MHSA
        x_mhsa_ln = self.self_att_layer_norm(x_ffn1_out)
        x_mhsa = self.self_att(x_mhsa_ln, axis=spatial_dim)
        x_mhsa = rf.dropout(x_mhsa, self.dropout, axis=self.dropout_broadcast and self.out_dim)

        # Branch 2: cgMLP
        x_cgmlp_ln = self.cgmlp_layer_norm(x_ffn1_out)
        x_cgmlp = self.cgmlp(x_cgmlp_ln, spatial_dim=spatial_dim)
        x_cgmlp = rf.dropout(x_cgmlp, self.dropout, axis=self.dropout_broadcast and self.out_dim)

        # Merge branches
        x_merge = self.merge(x_mhsa, x_cgmlp, spatial_dim=spatial_dim)
        x_merge = rf.dropout(x_merge, self.dropout, axis=self.dropout_broadcast and self.out_dim)
        x_merge_out = x_merge + x_ffn1_out

        # FFN
        x_ffn2_ln = self.ffn2_layer_norm(x_merge_out)
        x_ffn2 = self.ffn2(x_ffn2_ln)
        x_ffn2_out = 0.5 * rf.dropout(x_ffn2, self.dropout, axis=self.dropout_broadcast and self.out_dim) + x_merge_out

        # last LN layer
        return self.final_layer_norm(x_ffn2_out)


class FeedForwardConvGated(rf.Module):
    """
    Convolutional Gating MLP (cgMLP) as introduced in https://openreview.net/forum?id=RA-zVvZLYIy
    and then used by the E-Branchformer model (https://arxiv.org/pdf/2210.00077).
    It uses the Convolutional Spatial Gating Unit (CSGU).
    This is the local extractor branch in the E-Branchformer model.

    Related is the :class:`returnn.frontend.decoder.transformer.FeedForwardGated` module.
    """

    def __init__(
        self,
        out_dim: Dim,
        *,
        ff_dim: Union[Dim, int] = NotSpecified,
        kernel_size: int = 31,
        dropout: float = 0.1,
        activation: Union[Callable[[Tensor], Tensor], Dict[str, Any], rf.Module] = rf.gelu,
        gate_activation: Union[Callable[[Tensor], Tensor], Dict[str, Any], rf.Module] = rf.identity,
        with_bias: bool = True,
        norm: Union[type, Dict[str, Any], rf.Module, Callable] = rf.LayerNorm,
    ):
        """
        :param out_dim: the encoder (e.g. E-Branchformer) model dim. (usually 256 or 512)
        :param ff_dim: intermediate dimension.
            This is like cgmlp_linear_units/2 in ESPnet.
            Note the 1/2 factor, which is because in ESPnet, you specify the total dimension,
            before it is split for the gating,
            while here, you specify the dimension for the gating part.
            Common settings are 2048/2 or 3072/2.
            In the paper, they mention a factor of 3 of the model dimension (factor 6 for ESPnet setting).
        :param kernel_size: for the depthwise convolution (usually 31)
        :param dropout:
        :param activation: activation function after the first linear layer, for both parts.
            default as in the paper: gelu.
            Note, in :class:`returnn.frontend.decoder.transformer.FeedForwardGated`,
            the ``activation`` arg is like ``gate_activation`` here.
        :param gate_activation: activation function for the gate part, before the gating (mult) is applied.
            default as in the paper: identity.
            Note, in :class:`returnn.frontend.decoder.transformer.FeedForwardGated`,
            the ``activation`` arg is like ``gate_activation`` here.
        :param with_bias: whether to use bias in the linear layers and conv layer. default as in the paper: True.
        :param norm: normalization layer. default as in the paper: LayerNorm.
        """
        super().__init__()

        if ff_dim is NotSpecified:
            ff_dim = out_dim * 3  # somewhat arbitrary. with 512, this is 3072/2.
        if isinstance(ff_dim, int):
            ff_dim = Dim(ff_dim, name="e-branchformer-ff-dim")
        if not isinstance(ff_dim, Dim):
            raise TypeError(f"E-Branchformer FeedForwardConvGated: unexpected ff_dim {ff_dim!r} type {type(ff_dim)}")

        self.out_dim = out_dim
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.dropout_broadcast = rf.dropout_broadcast_default()
        if activation is NotSpecified:
            activation = rf.gelu
        self.activation = _make_activation(activation)
        if gate_activation is NotSpecified:
            gate_activation = rf.identity
        self.gate_activation = _make_activation(gate_activation)

        self.linear_ff = rf.Linear(out_dim, 2 * ff_dim, with_bias=with_bias)  # x2 to split for gating
        self.norm = make_norm(norm, ff_dim)
        self.conv = rf.Conv1d(  # depthwise convolution
            ff_dim, ff_dim, filter_size=kernel_size, groups=ff_dim.dimension, padding="same", with_bias=with_bias
        )
        self.linear_out = rf.Linear(ff_dim, out_dim, with_bias=with_bias)

    def __call__(self, x: Tensor, *, spatial_dim: Dim) -> Tensor:
        """forward"""
        x = self.linear_ff(x)
        x = self.activation(x)

        # Now the convolutional spatial gating part (CSGU).
        # Note: Different order than in ESPnet ConvolutionalSpatialGatingUnit,
        # but consistent to our own FeedForwardGated.
        x_g, x_r = rf.split(x, axis=self.linear_ff.out_dim, out_dims=[self.ff_dim] * 2)
        x_g = self.norm(x_g)
        x_g, _ = self.conv(x_g, in_spatial_dim=spatial_dim)
        x_g = self.gate_activation(x_g)
        x = x_g * x_r

        x = rf.dropout(x, self.dropout, axis=self.dropout_broadcast and self.linear_out.in_dim)
        x = self.linear_out(x)
        return x


class Merge(rf.Module):
    """
    The merge module from the E-Branchformer model.
    """

    def __init__(self, *, in_dim1: Dim, in_dim2: Dim, out_dim: Dim, merge_conv_kernel: int = 3):
        super().__init__()
        self.in_dim1 = in_dim1
        self.in_dim2 = in_dim2
        self.concat_in_dim = in_dim1 + in_dim2
        self.out_dim = out_dim
        self.depthwise_conv_fusion = rf.Conv1d(
            self.concat_in_dim,
            self.concat_in_dim,
            groups=self.concat_in_dim.dimension,
            filter_size=merge_conv_kernel,
            padding="same",
        )
        self.merge_proj = rf.Linear(self.concat_in_dim, out_dim)

    def __call__(self, x1: Tensor, x2: Tensor, *, spatial_dim: Dim) -> Tensor:
        """forward"""
        x_concat, _ = rf.concat((x1, self.in_dim1), (x2, self.in_dim2), out_dim=self.concat_in_dim)
        x_conv, _ = self.depthwise_conv_fusion(x_concat, in_spatial_dim=spatial_dim)
        y = self.merge_proj(x_concat + x_conv)
        return y


def _make_activation(
    activation: Union[Callable[[Tensor], Tensor], Dict[str, Any], rf.Module],
) -> Union[Callable[[Tensor], Tensor], rf.Module]:
    if isinstance(activation, dict):
        activation = rf.build_from_dict(activation)
    elif isinstance(activation, rf.Module):
        activation = _copy.deepcopy(activation)
    elif isinstance(activation, type):
        activation = activation()
    if not callable(activation):
        raise TypeError(f"unexpected activation type {activation!r}")
    return activation
