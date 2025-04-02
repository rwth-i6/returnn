"""
(Label-sync) Transformer decoder, optionally including cross attention to encoder

Also see :mod:`returnn.frontend.encoder.transformer`.

References:

    (Original paper of course)
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer
    https://github.com/pytorch-labs/gpt-fast
    https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    https://github.com/karpathy/nanoGPT/blob/master/model.py
    https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/transformer/transformer_decoder.py
"""

from __future__ import annotations
from typing import Optional, Any, Union, Tuple, Dict, Callable, Sequence
from types import FunctionType
import functools
import logging
import copy as _copy
from returnn.util.basic import NotSpecified, BehaviorVersion
from returnn.util.math import ceil_div
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, single_step_dim


class TransformerDecoder(rf.Module):
    """
    Represents the Transformer decoder architecture
    """

    def __init__(
        self,
        encoder_dim: Optional[Dim],
        vocab_dim: Dim,
        model_dim: Union[Dim, int] = Dim(512, name="transformer-dec-default-model-dim"),
        *,
        num_layers: int,
        ff: Union[type, Dict[str, Any], rf.Module] = NotSpecified,
        ff_dim: Union[Dim, int] = NotSpecified,
        ff_activation: Union[Callable[[Tensor], Tensor], Dict[str, Any], rf.Module] = NotSpecified,
        pos_enc: Union[None, Callable, Dict[str, Any], rf.Module] = rf.sinusoidal_positional_encoding,
        dropout: float = 0.1,
        num_heads: int = 8,
        att_dropout: float = 0.1,
        norm: Union[type, Dict[str, Any], rf.Module, Callable] = rf.LayerNorm,
        layer: Optional[Union[TransformerDecoderLayer, rf.Module, type, Dict[str, Any], Any]] = None,
        layer_opts: Optional[Dict[str, Any]] = None,
        embed_dim: Optional[Dim] = None,
        share_embedding: bool = None,
        input_embedding_scale: float = None,
        input_dropout: float = None,
        logits_with_bias: bool = False,
        sequential=rf.Sequential,
        **compat_kwargs,
    ):
        """
        :param encoder_dim: for cross-attention. None if no cross-attention.
        :param vocab_dim:
        :param model_dim: the output feature dimension
        :param num_layers: the number of encoder layers
        :param ff: feed-forward / MLP block. Default is :class:`FeedForward`
        :param ff_dim: the dimension of feed-forward layers. 2048 originally, or 4 times out_dim
        :param ff_activation: activation function for feed-forward network
        :param pos_enc: positional encoding. Default is sinusoidal positional encoding.
        :param dropout: the dropout value for the FF block
        :param num_heads: the number of attention heads
        :param att_dropout: attention dropout value
        :param norm: pre-normalization for FF and attention blocks
        :param layer: an instance of :class:`TransformerDecoderLayer` or similar
        :param layer_opts: options for the decoder layer
        :param embed_dim: if given, will first have an embedding [vocab,embed] and then a linear [embed,model].
        :param share_embedding:
        :param input_embedding_scale:
        :param input_dropout:
        :param logits_with_bias:
        :param sequential:
        """
        super().__init__()

        if compat_kwargs:
            if "decoder_layer" in compat_kwargs:  # compatibility, we used to have this before
                assert layer is None
                layer = compat_kwargs.pop("decoder_layer")
            if "decoder_layer_opts" in compat_kwargs:  # compatibility, we used to have this before
                assert layer_opts is None
                layer_opts = compat_kwargs.pop("decoder_layer_opts")
            if compat_kwargs:
                raise TypeError(f"unexpected kwargs {compat_kwargs!r}")

        if not isinstance(vocab_dim, Dim):
            raise TypeError(f"TransformerDecoder: unexpected vocab_dim {vocab_dim!r} type {type(vocab_dim)}")
        if isinstance(model_dim, int):
            model_dim = Dim(model_dim, name="transformer-dec-model-dim")
        if not isinstance(model_dim, Dim):
            raise TypeError(f"TransformerDecoder: unexpected model_dim {model_dim!r} type {type(model_dim)}")

        self.encoder_dim = encoder_dim
        self.vocab_dim = vocab_dim
        self.model_dim = model_dim
        self.embed_dim = embed_dim

        # We could make this optional or configurable if we ever need to.
        # Or maybe you would just have another separate implementation of this module then...
        self.input_embedding = rf.Embedding(vocab_dim, embed_dim or model_dim)

        self.input_embedding_proj = None
        if embed_dim:
            self.input_embedding_proj = rf.Linear(embed_dim, model_dim, with_bias=False)

        if pos_enc is None:
            pass
        elif isinstance(pos_enc, dict):
            pos_enc = rf.build_from_dict(pos_enc, feat_dim=embed_dim or model_dim)
        elif isinstance(pos_enc, rf.Module):
            pass
        elif isinstance(pos_enc, FunctionType):
            pos_enc = functools.partial(pos_enc, feat_dim=embed_dim or model_dim)
        else:
            raise TypeError(f"unexpected pos_enc type {pos_enc!r}")
        self.pos_enc = pos_enc
        if share_embedding is None:
            if BehaviorVersion.get() < 20:
                logging.getLogger("returnn.frontend").warning(
                    "TransformerDecoder share_embedding default is False"
                    f" with your behavior version {BehaviorVersion.get()}."
                    " Explicitly set share_embedding or switch to a new behavior version >= 20."
                )
            share_embedding = True if BehaviorVersion.get() >= 20 else False
        if input_embedding_scale is None:
            if BehaviorVersion.get() < 20:
                logging.getLogger("returnn.frontend").warning(
                    "TransformerDecoder input_embedding_scale default is suboptimal"
                    f" with your behavior version {BehaviorVersion.get()}."
                    " Explicitly set input_embedding_scale or switch to a new behavior version >= 20."
                )
            input_embedding_scale = model_dim.dimension**0.5 if BehaviorVersion.get() >= 20 else 1.0
        self.input_embedding_scale = input_embedding_scale
        if input_dropout is None:
            if dropout > 0 and BehaviorVersion.get() < 20:
                logging.getLogger("returnn.frontend").warning(
                    "TransformerDecoder input_dropout default is suboptimal"
                    f" with your behavior version {BehaviorVersion.get()}."
                    " Explicitly set input_dropout or switch to a new behavior version >= 20."
                )
            input_dropout = dropout if BehaviorVersion.get() >= 20 else 0.0
        self.input_dropout = input_dropout

        if not layer or isinstance(layer, (dict, type)):
            layer_opts_ = dict(
                encoder_dim=encoder_dim,
                out_dim=model_dim,
                ff=ff,
                ff_dim=ff_dim,
                ff_activation=ff_activation,
                dropout=dropout,
                num_heads=num_heads,
                att_dropout=att_dropout,
                norm=norm,
            )
            layer_opts_ = {k: v for (k, v) in layer_opts_.items() if v is not NotSpecified}
            if layer_opts:
                layer_opts_.update(layer_opts)
            if not layer:
                layer = TransformerDecoderLayer(**layer_opts_)
            elif isinstance(layer, type):
                layer = layer(**layer_opts_)
            elif isinstance(layer, dict):
                layer_opts_ = {k: v for (k, v) in layer_opts_.items() if k not in layer}
                layer = rf.build_from_dict(layer, **layer_opts_)
            else:
                raise TypeError(f"unexpected layer {layer!r}")

        self.layers = sequential(_copy.deepcopy(layer) for _ in range(num_layers))

        self.final_layer_norm = make_norm(norm, model_dim)

        self.logits = rf.Linear(model_dim, vocab_dim, with_bias=logits_with_bias)

        if share_embedding:
            assert not embed_dim and not logits_with_bias, "not supported together with share_embedding"
            self.logits.weight = self.input_embedding.weight

    def default_initial_state(self, *, batch_dims: Sequence[Dim]) -> rf.State:
        """default initial state"""
        state = rf.State({k: v.default_initial_state(batch_dims=batch_dims) for k, v in self.layers.items()})
        state.pos = rf.zeros((), dtype="int32", device="cpu")
        return state

    def transform_encoder(self, encoder: Tensor, *, axis: Dim) -> rf.State:
        """
        Transform encoder output.
        Note that the Transformer decoder usually expects that layer-norm was applied already on the encoder output.
        """
        return rf.State({k: v.transform_encoder(encoder, axis=axis) for k, v in self.layers.items()})

    def __call__(
        self,
        source: Tensor,
        *,
        spatial_dim: Dim,
        state: rf.State,
        encoder: Optional[rf.State] = None,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
        output_only_last_frame: bool = False,
    ) -> Tuple[Tensor, rf.State]:
        """
        forward, single step or whole sequence.

        :param source: labels
        :param spatial_dim: single_step_dim or spatial dim of source
        :param state: e.g. via :func:`default_initial_state`
        :param encoder: via :func:`transform_encoder`
        :param collected_outputs:
        :param output_only_last_frame: if True, and spatial_dim is not single_step_dim,
            the returned logits will only be for the last frame
        :return: logits, new state
        """
        new_state = rf.State()

        decoded = self.input_embedding(source) * self.input_embedding_scale
        if self.pos_enc is not None:
            decoded = decoded + self.pos_enc(spatial_dim=spatial_dim, offset=state.pos)
        decoded = rf.dropout(decoded, self.input_dropout)
        if self.input_embedding_proj is not None:
            decoded = self.input_embedding_proj(decoded)

        new_state.pos = state.pos + (1 if spatial_dim == single_step_dim else spatial_dim.get_size_tensor())

        for layer_name, layer in self.layers.items():
            layer: TransformerDecoderLayer  # or similar
            decoded, new_state[layer_name] = layer(
                decoded,
                spatial_dim=spatial_dim,
                state=state[layer_name],
                encoder=encoder[layer_name] if encoder else None,
            )
            if collected_outputs is not None:
                collected_outputs[layer_name] = decoded

        if output_only_last_frame and spatial_dim != single_step_dim:
            decoded = rf.gather(decoded, axis=spatial_dim, indices=rf.last_frame_position_of_dim(spatial_dim))

        decoded = self.final_layer_norm(decoded)
        logits = self.logits(decoded)

        return logits, new_state


class TransformerDecoderLayer(rf.Module):
    """
    Represents a conformer block
    """

    def __init__(
        self,
        encoder_dim: Optional[Dim],
        out_dim: Dim = Dim(512, name="transformer-dec-default-out-dim"),
        *,
        ff: Union[type, Dict[str, Any], rf.Module] = NotSpecified,
        ff_dim: Union[Dim, int] = NotSpecified,
        ff_activation: Union[Callable[[Tensor], Tensor], Dict[str, Any], rf.Module] = NotSpecified,
        dropout: float = 0.1,
        num_heads: int = 8,
        self_att: Optional[
            Union[rf.CausalSelfAttention, rf.RelPosCausalSelfAttention, rf.Module, type, Dict[str, Any]]
        ] = None,
        self_att_opts: Optional[Dict[str, Any]] = None,
        att_dropout: float = 0.1,
        norm: Union[type, Dict[str, Any], rf.Module, Callable] = rf.LayerNorm,
    ):
        """
        :param encoder_dim: for cross-attention. None if no cross-attention.
        :param out_dim: the output feature dimension
        :param ff: feed-forward / MLP block. Default is :class:`FeedForward`
        :param ff_dim: the dimension of feed-forward layers. 2048 originally, or 4 times out_dim
        :param ff_activation: activation function for feed-forward network
        :param dropout: the dropout value for the FF block
        :param num_heads: the number of attention heads
        :param self_att: the self-attention layer. CausalSelfAttention originally and default
        :param self_att_opts: options for the self-attention layer, for :class:`nn.RelPosSelfAttention`
        :param att_dropout: attention dropout value
        :param norm: pre-normalization for FF and attention blocks
        """
        super().__init__()

        self.encoder_dim = encoder_dim
        self.dropout = dropout
        self.dropout_broadcast = rf.dropout_broadcast_default()
        self.out_dim = out_dim

        if ff is NotSpecified:
            ff = FeedForward
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

        self.ff = ff
        self.ff_layer_norm = make_norm(norm, out_dim)

        if self_att is None or isinstance(self_att, type) or isinstance(self_att, dict):
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
                self.self_att = rf.CausalSelfAttention(**self_att_opts_)
            elif isinstance(self_att, type):
                self.self_att = self_att(**self_att_opts_)
            elif isinstance(self_att, dict):
                self.self_att = rf.build_from_dict(self_att, **self_att_opts_)
            else:
                raise TypeError(f"unexpected self_att type {self_att!r}")
        elif isinstance(self_att, rf.Module):
            self.self_att = _copy.deepcopy(self_att)
        else:
            raise TypeError(f"unexpected self_att type {self_att!r}")
        self.self_att_layer_norm = make_norm(norm, out_dim)

        self.cross_att = None
        self.cross_att_layer_norm = None
        if encoder_dim is not None:
            self.cross_att = rf.CrossAttention(
                encoder_dim=self.encoder_dim,
                query_in_dim=out_dim,
                proj_dim=out_dim,
                key_dim_total=out_dim,
                value_dim_total=out_dim,
                num_heads=num_heads,
                att_dropout=att_dropout,
            )
            self.cross_att_layer_norm = make_norm(norm, out_dim)

    def default_initial_state(self, *, batch_dims: Sequence[Dim]) -> rf.State:
        """default initial state"""
        return rf.State(self_att=self.self_att.default_initial_state(batch_dims=batch_dims))

    def transform_encoder(self, encoder: Tensor, *, axis: Dim) -> rf.State:
        """Transform the encoder output."""
        assert self.cross_att is not None
        return rf.State(cross_att=self.cross_att.transform_encoder(encoder, axis=axis))

    def __call__(
        self, x: Tensor, *, spatial_dim: Dim, state: rf.State, encoder: Optional[rf.State] = None
    ) -> Tuple[Tensor, rf.State]:
        """forward"""
        # (multi-head) self-attention (MHSA or simply SA)
        new_state = rf.State()
        x_sa_ln = self.self_att_layer_norm(x)
        x_sa, new_state.self_att = self.self_att(x_sa_ln, axis=spatial_dim, state=state.self_att)
        x_sa = rf.dropout(x_sa, self.dropout, axis=self.dropout_broadcast and self.out_dim)
        x = x_sa + x

        # (multi-head) cross-attention (CA)
        if self.cross_att is not None:
            x_ca_ln = self.cross_att_layer_norm(x)
            x_ca = self.cross_att(x_ca_ln, encoder.cross_att)
            x_ca = rf.dropout(x_ca, self.dropout, axis=self.dropout_broadcast and self.out_dim)
            x = x_ca + x

        # feed-forward (FF)
        x_ff_ln = self.ff_layer_norm(x)
        x_ff = self.ff(x_ff_ln)
        x_ff = rf.dropout(x_ff, self.dropout, axis=self.dropout_broadcast and self.out_dim)
        x = x_ff + x

        return x, new_state


class FeedForward(rf.Module):
    """
    Transformer position-wise feedforward neural network layer
        FF -> Activation -> Dropout -> FF
    """

    def __init__(
        self,
        out_dim: Dim,
        *,
        ff_dim: Optional[Union[Dim, int]] = NotSpecified,
        dropout: float = 0.1,
        activation: Union[Callable[[Tensor], Tensor], Dict[str, Any], rf.Module] = rf.relu,
        with_bias: bool = True,
    ):
        """
        :param out_dim: output feature dimension
        :param ff_dim: dimension of the feed-forward layers
        :param dropout: dropout value
        :param activation: activation function, relu by default
        :param with_bias: whether to use bias in the linear layers.
            True by default for compatibility, but nowadays it's common to use without bias.
        """
        super().__init__()

        if isinstance(ff_dim, int):
            ff_dim = Dim(ff_dim, name="transformer-ff-dim")
        if ff_dim is NotSpecified or ff_dim is None:
            ff_dim = out_dim * 4
        if not isinstance(ff_dim, Dim):
            raise TypeError(f"Transformer FeedForward: unexpected ff_dim {ff_dim!r} type {type(ff_dim)}")

        if activation is NotSpecified:
            activation = rf.relu
        elif isinstance(activation, dict):
            activation = rf.build_from_dict(activation)
        elif not callable(activation):
            raise TypeError(f"{self}: unexpected activation type {activation!r}")

        self.out_dim = out_dim
        self.dropout = dropout
        self.dropout_broadcast = rf.dropout_broadcast_default()
        self.activation = activation

        self.linear_ff = rf.Linear(out_dim, ff_dim, with_bias=with_bias)
        self.linear_out = rf.Linear(ff_dim, out_dim, with_bias=with_bias)

    def __call__(self, inp: Tensor) -> Tensor:
        """forward"""
        x_ff1 = self.linear_ff(inp)
        x_act = self.activation(x_ff1)
        x_drop = rf.dropout(x_act, self.dropout, axis=self.dropout_broadcast and self.linear_ff.out_dim)
        x_ff2 = self.linear_out(x_drop)
        return x_ff2


class FeedForwardGated(rf.Module):
    """
    E.g. with f=swish=silu:
    SwiGLU, from `GLU Variants Improve Transformer <https://arxiv.org/abs/2002.05202>`__::

        f(Linear(x)) * Linear(x)

    This is a feed-forward block based on SwiGLU, as defined in the paper.

    Alternative to :class:`FeedForward`.
    """

    def __init__(
        self,
        out_dim: Dim,
        *,
        ff_dim: Optional[Union[Dim, int]] = NotSpecified,
        dropout: float = 0.1,
        activation: Union[Callable[[Tensor], Tensor], Dict[str, Any], rf.Module] = rf.swish,
        gate_activation: Union[Callable[[Tensor], Tensor], Dict[str, Any], rf.Module] = rf.identity,
        with_bias: bool = False,
    ):
        """
        :param out_dim:
        :param ff_dim: intermediate dimension.
            Unlike :class:`FeedForward`:
            If not provided, factor 4*2/3 to keep same number of parameters as in the original :class:`FeedForward`,
            just as in the paper, and also making it a multiple of 256.
        :param dropout:
        :param activation: activation function for the gating. unlike :class:`FeedForward`, default is swish.
        :param with_bias: whether to use bias in the linear layers.
            unlike :class:`FeedForward`, default is False.
        """
        super().__init__()

        if isinstance(ff_dim, int):
            ff_dim = Dim(ff_dim, name="transformer-ff-dim")
        if ff_dim is NotSpecified or ff_dim is None:
            # Factor 4 as usual.
            # The additional factor 2/3 to keep same number of parameters as in the original FF block,
            # just as in the paper.
            ff_dim_ = out_dim.dimension * 4 * 2 // 3
            ff_dim_ = ceil_div(ff_dim_, 256) * 256  # make multiple of 256
            ff_dim = Dim(ff_dim_, name="transformer-ff-dim")
        if not isinstance(ff_dim, Dim):
            raise TypeError(f"Transformer FeedForward: unexpected ff_dim {ff_dim!r} type {type(ff_dim)}")

        if activation is NotSpecified:
            activation = rf.swish
        elif isinstance(activation, dict):
            activation = rf.build_from_dict(activation)
        elif not callable(activation):
            raise TypeError(f"{self}: unexpected activation type {activation!r}")
        if gate_activation is NotSpecified:
            gate_activation = rf.identity
        elif isinstance(gate_activation, dict):
            gate_activation = rf.build_from_dict(gate_activation)
        elif not callable(gate_activation):
            raise TypeError(f"{self}: unexpected gate_activation type {gate_activation!r}")

        self.out_dim = out_dim
        self.dropout = dropout
        self.dropout_broadcast = rf.dropout_broadcast_default()
        self.activation = activation
        self.gate_activation = gate_activation

        # Factor 2 because we concatenate the two paths.
        self.linear_ff = rf.Linear(out_dim, 2 * ff_dim, with_bias=with_bias)
        self.linear_out = rf.Linear(ff_dim, out_dim, with_bias=with_bias)

    def __call__(self, inp: Tensor) -> Tensor:
        """forward"""
        x_ff1 = self.linear_ff(inp)
        x_ff1a, x_ff1b = rf.split(x_ff1, axis=self.linear_ff.out_dim, out_dims=[self.linear_out.in_dim] * 2)
        x_act = self.activation(x_ff1a) * self.gate_activation(x_ff1b)
        x_drop = rf.dropout(x_act, self.dropout, axis=self.dropout_broadcast and self.linear_out.in_dim)
        x_ff2 = self.linear_out(x_drop)
        return x_ff2


def make_norm(norm: Union[type, Dict[str, Any], rf.Module, Callable], out_dim: Dim) -> Union[rf.Module, Callable]:
    """
    :param norm: norm type or dict or module or callable. e.g. ``rf.LayerNorm``
    :param out_dim: model/out dim
    :return: norm module or callable. e.g. ``rf.LayerNorm(out_dim)``
    """
    if isinstance(norm, type):
        norm = norm(out_dim)
    elif isinstance(norm, dict):
        norm = rf.build_from_dict(norm, out_dim)
    elif isinstance(norm, rf.Module):
        norm = _copy.deepcopy(norm)
    if not callable(norm):
        raise TypeError(f"unexpected norm type {norm!r}")
    return norm
