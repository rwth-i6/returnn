"""
Transformer encoder

Also see :mod:`returnn.frontend.decoder.transformer`.
"""

from __future__ import annotations
from typing import Optional, Union, Any, Callable, Sequence, Dict
import functools
import copy as _copy
from types import FunctionType
from returnn.util.basic import NotSpecified
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.frontend.decoder.transformer import make_norm, FeedForward


class TransformerEncoder(rf.Module):
    """
    Represents the Transformer encoder architecture
    """

    def __init__(
        self,
        vocab_dim: Dim,
        model_dim: Union[Dim, int] = Dim(512, name="transformer-enc-default-model-dim"),
        *,
        num_layers: int,
        ff: Union[type, Dict[str, Any], rf.Module] = NotSpecified,
        pos_enc: Union[None, Callable, Dict[str, Any], rf.Module] = rf.sinusoidal_positional_encoding,
        dropout: float = 0.1,
        num_heads: int = 8,
        att_dropout: float = 0.1,
        norm: Union[type, Dict[str, Any], rf.Module, Callable] = rf.LayerNorm,
        layer: Optional[Union[TransformerEncoderLayer, rf.Module, type, Dict[str, Any], Any]] = None,
        layer_opts: Optional[Dict[str, Any]] = None,
        embed_dim: Optional[Dim] = None,
        input_embedding: Union[None, rf.Module, type, Dict[str, Any]] = rf.Embedding,
        input_embedding_scale: float = None,
        input_dropout: float = None,
        sequential=rf.Sequential,
        **compat_kwargs,
    ):
        """
        :param vocab_dim:
        :param model_dim: the output feature dimension
        :param num_layers: the number of encoder layers
        :param ff: feed-forward / MLP block. Default is :class:`FeedForward`
        :param pos_enc: positional encoding. Default is sinusoidal positional encoding.
        :param dropout: the dropout value for the FF block
        :param num_heads: the number of attention heads
        :param att_dropout: attention dropout value
        :param norm: pre-normalization for FF and attention blocks
        :param layer: an instance of :class:`TransformerEncoderLayer` or similar
        :param layer_opts: options for the encoder layer
        :param embed_dim: if given, will first have an embedding [vocab,embed] and then a linear [embed,model].
        :param input_embedding:
        :param input_embedding_scale:
        :param input_dropout:
        :param sequential:
        """
        super().__init__()

        if compat_kwargs:
            if "decoder_layer" in compat_kwargs:  # compatibility, we (weirdly) used to have this before
                assert layer is None
                layer = compat_kwargs.pop("decoder_layer")
            if compat_kwargs:
                raise TypeError(f"unexpected kwargs {compat_kwargs!r}")

        if not isinstance(vocab_dim, Dim):
            raise TypeError(f"TransformerDecoder: unexpected vocab_dim {vocab_dim!r} type {type(vocab_dim)}")
        if isinstance(model_dim, int):
            model_dim = Dim(model_dim, name="transformer-dec-model-dim")
        if not isinstance(model_dim, Dim):
            raise TypeError(f"TransformerDecoder: unexpected model_dim {model_dim!r} type {type(model_dim)}")

        self.vocab_dim = vocab_dim
        self.model_dim = model_dim
        self.embed_dim = embed_dim

        if input_embedding is None or isinstance(input_embedding, rf.Module):
            pass
        elif isinstance(input_embedding, type):
            input_embedding: rf.Embedding = input_embedding(vocab_dim, embed_dim or model_dim)
        elif isinstance(input_embedding, dict):
            input_embedding = rf.build_from_dict(input_embedding, vocab_dim, embed_dim or model_dim)
        else:
            raise TypeError(f"unexpected input_embedding {input_embedding!r} type {type(input_embedding)}")
        self.input_embedding = input_embedding

        self.input_embedding_proj = None
        if embed_dim:
            self.input_embedding_proj = rf.Linear(embed_dim, model_dim, with_bias=False)

        if pos_enc is None:
            pass
        elif isinstance(pos_enc, dict):
            pos_enc = rf.build_from_dict(pos_enc, feat_dim=embed_dim or model_dim, dtype=rf.get_default_float_dtype())
        elif isinstance(pos_enc, rf.Module):
            pass
        elif isinstance(pos_enc, FunctionType):
            pos_enc = functools.partial(pos_enc, feat_dim=embed_dim or model_dim, dtype=rf.get_default_float_dtype())
        else:
            raise TypeError(f"unexpected pos_enc {pos_enc!r} type {type(pos_enc)}")
        self.pos_enc = pos_enc
        if input_embedding_scale is None:
            input_embedding_scale = model_dim.dimension**0.5
        self.input_embedding_scale = input_embedding_scale
        if input_dropout is None:
            input_dropout = dropout
        self.input_dropout = input_dropout

        if not layer or isinstance(layer, (dict, type)):
            layer_opts_ = dict(
                out_dim=model_dim,
                ff=ff,
                dropout=dropout,
                num_heads=num_heads,
                att_dropout=att_dropout,
                norm=norm,
            )
            layer_opts_ = {k: v for (k, v) in layer_opts_.items() if v is not NotSpecified}
            if layer_opts:
                layer_opts_.update(layer_opts)
            if not layer:
                layer = TransformerEncoderLayer(**layer_opts_)
            elif isinstance(layer, type):
                layer = layer(**layer_opts_)
            elif isinstance(layer, dict):
                layer_opts_ = {k: v for (k, v) in layer_opts_.items() if k not in layer}
                layer = rf.build_from_dict(layer, **layer_opts_)
            else:
                raise TypeError(f"unexpected layer {layer!r}")

        self.layers = sequential(_copy.deepcopy(layer) for _ in range(num_layers))

        self.final_layer_norm = make_norm(norm, model_dim)

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
        self, source: Tensor, *, spatial_dim: Dim, collected_outputs: Optional[Dict[str, Tensor]] = None
    ) -> Tensor:
        """
        forward, single step or whole sequence.

        :param source: labels
        :param spatial_dim: single_step_dim or spatial dim of source
        :param collected_outputs:
        :return: final encoder output, after final layer norm
        """
        if self.input_embedding is not None:
            decoded = self.input_embedding(source) * self.input_embedding_scale
        else:
            assert self.model_dim in source.dims
            decoded = source
        if self.pos_enc is not None:
            decoded = decoded + self.pos_enc(spatial_dim=spatial_dim)
        decoded = rf.dropout(decoded, self.input_dropout)
        if self.input_embedding_proj is not None:
            decoded = self.input_embedding_proj(decoded)

        for layer_name, layer in self.layers.items():
            layer: TransformerEncoderLayer  # or similar
            decoded = layer(decoded, spatial_dim=spatial_dim)
            if collected_outputs is not None:
                collected_outputs[layer_name] = decoded

        decoded = self.final_layer_norm(decoded)
        return decoded


class TransformerEncoderLayer(rf.Module):
    """
    Represents a Transformer encoder block
    """

    def __init__(
        self,
        out_dim: Dim = Dim(512, name="transformer-enc-default-out-dim"),
        *,
        ff: Union[type, Dict[str, Any], rf.Module] = NotSpecified,
        dropout: float = 0.1,
        num_heads: int = 8,
        self_att: Optional[
            Union[rf.CausalSelfAttention, rf.RelPosCausalSelfAttention, rf.Module, type, Dict[str, Any]]
        ] = None,
        att_dropout: float = 0.1,
        norm: Union[type, Dict[str, Any], rf.Module, Callable] = rf.LayerNorm,
    ):
        """
        :param out_dim: the output feature dimension
        :param ff: feed-forward / MLP block. Default is :class:`FeedForward`
        :param dropout: the dropout value for the FF block
        :param num_heads: the number of attention heads
        :param self_att: the self-attention layer. SelfAttention originally and default
        :param att_dropout: attention dropout value
        :param norm: pre-normalization for FF and attention blocks
        """
        super().__init__()

        self.dropout = dropout
        self.dropout_broadcast = rf.dropout_broadcast_default()
        self.out_dim = out_dim

        if ff is NotSpecified:
            ff = FeedForward
        if isinstance(ff, rf.Module):
            ff = _copy.deepcopy(ff)
        else:
            ff_kwargs = dict(out_dim=out_dim, dropout=dropout)
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
            if self_att is None:
                self.self_att = rf.SelfAttention(**self_att_opts_)
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

    def default_initial_state(self, *, batch_dims: Sequence[Dim]) -> rf.State:
        """default initial state"""
        return rf.State(self_att=self.self_att.default_initial_state(batch_dims=batch_dims))

    def __call__(self, x: Tensor, *, spatial_dim: Dim) -> Tensor:
        """forward"""
        # (multi-head) self-attention (MHSA or simply SA)
        x_sa_ln = self.self_att_layer_norm(x)
        x_sa = self.self_att(x_sa_ln, axis=spatial_dim)
        x_sa = rf.dropout(x_sa, self.dropout, axis=self.dropout_broadcast and self.out_dim)
        x = x_sa + x

        # feed-forward (FF)
        x_ff_ln = self.ff_layer_norm(x)
        x_ff = self.ff(x_ff_ln)
        x_ff = rf.dropout(x_ff, self.dropout, axis=self.dropout_broadcast and self.out_dim)
        x = x_ff + x

        return x
