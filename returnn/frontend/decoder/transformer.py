"""
(Label-sync) Transformer decoder, including cross attention to encoder

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
import functools
import logging
import copy as _copy
from returnn.util.basic import NotSpecified, BehaviorVersion
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, single_step_dim


class TransformerDecoder(rf.Module):
    """
    Represents Transformer decoder architecture
    """

    def __init__(
        self,
        encoder_dim: Optional[Dim],
        vocab_dim: Dim,
        model_dim: Dim = Dim(512, name="transformer-dec-default-model-dim"),
        *,
        num_layers: int,
        ff_dim: Dim = NotSpecified,
        ff_activation: Callable[[Tensor], Tensor] = rf.relu,
        dropout: float = 0.1,
        num_heads: int = 8,
        att_dropout: float = 0.1,
        decoder_layer: Optional[Union[TransformerDecoderLayer, rf.Module, type, Any]] = None,
        decoder_layer_opts: Optional[Dict[str, Any]] = None,
        embed_dim: Optional[Dim] = None,
        share_embedding: bool = None,
        input_embedding_scale: float = None,
        input_dropout: float = None,
        logits_with_bias: bool = False,
        sequential=rf.Sequential,
    ):
        """
        :param encoder_dim: for cross-attention. None if no cross-attention.
        :param vocab_dim:
        :param model_dim: the output feature dimension
        :param num_layers: the number of encoder layers
        :param ff_dim: the dimension of feed-forward layers. 2048 originally, or 4 times out_dim
        :param ff_activation: activation function for feed-forward network
        :param dropout: the dropout value for the FF block
        :param num_heads: the number of attention heads
        :param att_dropout: attention dropout value
        :param decoder_layer: an instance of :class:`TransformerDecoderLayer` or similar
        :param decoder_layer_opts: options for the encoder layer
        :param embed_dim: if given, will first have an embedding [vocab,embed] and then a linear [embed,model].
        :param share_embedding:
        :param input_embedding_scale:
        :param input_dropout:
        :param logits_with_bias:
        :param sequential:
        """
        super().__init__()

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

        # This could also be configurable...
        self.pos_enc = functools.partial(
            rf.sinusoidal_positional_encoding, feat_dim=embed_dim or model_dim, dtype=self.input_embedding.weight.dtype
        )
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

        if not decoder_layer or isinstance(decoder_layer, type):
            decoder_layer_opts_ = dict(
                encoder_dim=encoder_dim,
                out_dim=model_dim,
                ff_dim=ff_dim,
                ff_activation=ff_activation,
                dropout=dropout,
                num_heads=num_heads,
                att_dropout=att_dropout,
            )
            if decoder_layer_opts:
                decoder_layer_opts_.update(decoder_layer_opts)
            if not decoder_layer:
                decoder_layer = TransformerDecoderLayer(**decoder_layer_opts_)
            elif isinstance(decoder_layer, type):
                decoder_layer = decoder_layer(**decoder_layer_opts_)
            else:
                raise TypeError(f"unexpected decoder_layer {decoder_layer!r}")

        self.layers = sequential(_copy.deepcopy(decoder_layer) for _ in range(num_layers))

        self.final_layer_norm = rf.LayerNorm(model_dim)

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
    ) -> Tuple[Tensor, rf.State]:
        """
        forward, single step or whole sequence.

        :param source: labels
        :param spatial_dim: single_step_dim or spatial dim of source
        :param state: e.g. via :func:`default_initial_state`
        :param encoder: via :func:`transform_encoder`
        :param collected_outputs:
        :return: logits, new state
        """
        new_state = rf.State()

        decoded = self.input_embedding(source) * self.input_embedding_scale
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
        ff_dim: Dim = NotSpecified,
        ff_activation: Callable[[Tensor], Tensor] = rf.relu,
        dropout: float = 0.1,
        num_heads: int = 8,
        self_att: Optional[Union[rf.CausalSelfAttention, rf.RelPosCausalSelfAttention, rf.Module, type, Any]] = None,
        self_att_opts: Optional[Dict[str, Any]] = None,
        att_dropout: float = 0.1,
    ):
        """
        :param encoder_dim: for cross-attention. None if no cross-attention.
        :param out_dim: the output feature dimension
        :param ff_dim: the dimension of feed-forward layers. 2048 originally, or 4 times out_dim
        :param ff_activation: activation function for feed-forward network
        :param dropout: the dropout value for the FF block
        :param num_heads: the number of attention heads
        :param self_att: the self-attention layer. RelPosSelfAttention originally and default
        :param self_att_opts: options for the self-attention layer, for :class:`nn.RelPosSelfAttention`
        :param att_dropout: attention dropout value
        """
        super().__init__()

        self.encoder_dim = encoder_dim
        self.dropout = dropout
        self.dropout_broadcast = rf.dropout_broadcast_default()
        self.out_dim = out_dim

        if ff_dim is None:
            ff_dim = 4 * out_dim
        self.ff = FeedForward(out_dim=out_dim, ff_dim=ff_dim, dropout=dropout, activation=ff_activation)
        self.ff_layer_norm = rf.LayerNorm(out_dim)

        if self_att is None or isinstance(self_att, type):
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
            else:
                self.self_att = self_att(**self_att_opts_)
        else:
            self.self_att = self_att
        self.self_att_layer_norm = rf.LayerNorm(out_dim)

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
            self.cross_att_layer_norm = rf.LayerNorm(out_dim)

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
        ff_dim: Optional[Dim] = NotSpecified,
        dropout: float,
        activation: Callable[[Tensor], Tensor],
    ):
        """
        :param out_dim: output feature dimension
        :param ff_dim: dimension of the feed-forward layers
        :param dropout: dropout value
        :param activation: activation function
        """
        super().__init__()

        if ff_dim is NotSpecified:
            ff_dim = out_dim * 4

        self.out_dim = out_dim
        self.dropout = dropout
        self.dropout_broadcast = rf.dropout_broadcast_default()
        self.activation = activation

        self.linear_ff = rf.Linear(out_dim, ff_dim)
        self.linear_out = rf.Linear(ff_dim, out_dim)

    def __call__(self, inp: Tensor) -> Tensor:
        """forward"""
        x_ff1 = self.linear_ff(inp)
        x_act = self.activation(x_ff1)
        x_drop = rf.dropout(x_act, self.dropout, axis=self.dropout_broadcast and self.linear_ff.out_dim)
        x_ff2 = self.linear_out(x_drop)
        return x_ff2
