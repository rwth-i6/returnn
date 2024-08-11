"""
Attention
"""

from __future__ import annotations
from typing import Tuple, Union, Optional, Sequence
import weakref
import logging
from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf


__all__ = [
    "dot_attention",
    "SelfAttentionBase",
    "SelfAttention",
    "CausalSelfAttention",
    "CausalSelfAttentionState",
    "RotaryPosSelfAttention",
    "RotaryPosCausalSelfAttention",
    "RelPosSelfAttention",
    "RelPosCausalSelfAttention",
    "CrossAttention",
    "LearnedRelativePositionalEncoding",
    "relative_positional_encoding",
    "sinusoidal_positional_encoding",
]


def dot_attention(
    query: Tensor,
    keys: Tensor,
    values: Tensor,
    *,
    key_dim: Dim,
    axis: Dim,
    att_dropout: float = 0.0,
    att_dropout_broadcast: Optional[bool] = None,
) -> Tensor:
    """
    Calculates attention over the given axis, for given key dim.
    Any other unrelated axes do not matter here.
    This can be used for multi-head or single head.
    The query can have other dimensions or not.

    :param query: {..., key_dim}. For self-attention, do not use the `axis` as in `keys` and `values`,
        but rather replace it by another new dim via :func:`replace_dim`.
    :param keys: {..., axis, key_dim}
    :param values: {..., axis}
    :param key_dim: dim in keys and query, to be reduced to calculate the attention energies.
    :param axis: in keys and values, to apply attention on. softmax will be over this axis, and then it will be reduced
    :param att_dropout: dropout for attention weights
    :param att_dropout_broadcast: whether to broadcast over all but ``axis``.
        normally not wanted. disabled by default since behavior version 19.
    :return: like values but with axis removed, and maybe any additional axes from query
    """
    query *= key_dim.dimension**-0.5
    energy = rf.matmul(query, keys, reduce=key_dim)
    att_weights = rf.softmax(energy, axis=axis)
    if att_dropout_broadcast is None:
        att_dropout_broadcast = _att_dropout_broadcast_default()
    att_weights = rf.dropout(att_weights, att_dropout, axis=att_dropout_broadcast and axis)
    # Masking not needed because softmax should already have masked,
    # so we have 0.0 att weights for padded frames.
    att = rf.matmul(att_weights, values, reduce=axis, use_mask=False)
    if values.feature_dim in att.dims:
        att.feature_dim = values.feature_dim
    return att


# noinspection PyAbstractClass
class SelfAttentionBase(rf.Module):
    """
    Shared base class for (non-causal) self attention (:class:`SelfAttention`)
    and causal self attention (:class:`CausalSelfAttention`).

    It uses :func:`dot_attention` for multi-headed dot-attention.
    """

    def __init__(
        self,
        in_dim: Dim,
        proj_dim: Optional[Dim],
        *,
        key_dim_total: Dim,
        value_dim_total: Dim,
        num_heads: Union[int, Dim],
        with_bias: bool = True,
        att_dropout: float = 0.1,
        att_dropout_broadcast: Optional[bool] = None,
    ):
        """
        :param in_dim: input dim
        :param proj_dim: if given, will add a final linear projection to this dim.
            otherwise no projection after the attention
        :param key_dim_total: total key dim. should be a multiple of num_heads
        :param value_dim_total: total value dim. should be a multiple of num_heads
        :param num_heads: number of heads
        :param with_bias: whether to add bias to qkv and proj linear projections.
            Was False in original Transformer, but many recent implementations use True by default.
            Also see: https://github.com/rwth-i6/returnn_common/issues/234.
        :param att_dropout: dropout for attention weights
        :param att_dropout_broadcast: whether to broadcast over all but ``axis``.
            normally not wanted. disabled by default since behavior version 19.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = proj_dim if proj_dim else value_dim_total
        if isinstance(num_heads, int):
            num_heads = Dim(num_heads, name="num_heads")
        self.key_dim_total = key_dim_total
        self.key_dim_per_head = key_dim_total.div_left(num_heads)
        self.value_dim_total = value_dim_total
        self.value_dim_per_head = value_dim_total.div_left(num_heads)
        self.num_heads = num_heads
        self.qkv_dim_total = 2 * key_dim_total + value_dim_total
        self.qkv_dim_per_head = 2 * self.key_dim_per_head + self.value_dim_per_head
        self.qkv = rf.Linear(in_dim, self.qkv_dim_total, with_bias=with_bias)
        # In Fairseq MultiheadAttention, they use:
        #   nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))  (same for q_proj, v_proj),
        # where xavier_uniform_ means:
        #   std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        #   a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        #   _no_grad_uniform_(tensor, -a, a)
        # Out nn.init.VarianceScaling with mode="fan_avg", distribution="uniform":
        #   scale = scale * 2.0 / float(fan_in + fan_out)
        #   limit = math.sqrt(3.0 * scale)
        #   nn.random(distribution="uniform", minval=-limit, maxval=limit, ...)
        # Our fan_out is 3 times larger than in Fairseq, because we concatenate q,k,v.
        # Assuming fan_in = fan_out, it means a factor 2 in the denominator.
        # So our default (Glorot, which is VarianceScaling with mode="fan_avg", distribution="uniform", scale=1.0)
        # is already the same as Fairseq.
        # The bias init is different, but not sure how important this is.
        if proj_dim:
            self.proj = rf.Linear(value_dim_total, proj_dim, with_bias=with_bias)
        else:
            self.proj = None
        self.att_dropout = att_dropout
        if att_dropout_broadcast is None:
            att_dropout_broadcast = _att_dropout_broadcast_default()
        self.att_dropout_broadcast = att_dropout_broadcast

    def forward_qkv(self, source: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        :return: q,k,v
        """
        qkv = self.qkv(source)
        qkv = rf.split_dims(qkv, axis=self.qkv_dim_total, dims=(self.num_heads, self.qkv_dim_per_head))
        q, k, v = rf.split(
            qkv,
            axis=self.qkv_dim_per_head,
            out_dims=(self.key_dim_per_head, self.key_dim_per_head, self.value_dim_per_head),
        )
        return q, k, v

    def attention(self, q: Tensor, k: Tensor, v: Tensor, *, kv_axis: Dim) -> Tensor:
        """apply attention"""
        att = dot_attention(
            q,
            k,
            v,
            key_dim=self.key_dim_per_head,
            axis=kv_axis,
            att_dropout=self.att_dropout,
            att_dropout_broadcast=self.att_dropout_broadcast,
        )
        output, _ = rf.merge_dims(att, dims=(self.num_heads, self.value_dim_per_head), out_dim=self.value_dim_total)
        if self.proj:
            output = self.proj(output)
        return output


class SelfAttention(SelfAttentionBase):
    """
    Classic self attention on sequence level
    """

    def __call__(self, source: Tensor, *, axis: Dim) -> Tensor:
        """forward"""
        q, k, v = self.forward_qkv(source)
        kv_axis = Dim(None, name=f"{axis.name}-kv")
        k, _ = rf.replace_dim(k, in_dim=axis, out_dim=kv_axis)
        v, _ = rf.replace_dim(v, in_dim=axis, out_dim=kv_axis)
        return self.attention(q, k, v, kv_axis=kv_axis)


class CausalSelfAttention(SelfAttentionBase):
    """
    Classic causal self attention
    """

    def __call__(
        self,
        source: Tensor,
        axis: Dim,
        *,
        state: Optional[CausalSelfAttentionState] = None,
    ) -> Tuple[Tensor, CausalSelfAttentionState]:
        """forward"""
        q, k, v = self.forward_qkv(source)
        k, v, hist_dim, new_state = _causal_self_att_step(k, v, axis=axis, state=state, self=self)
        output = self.attention(q, k, v, kv_axis=hist_dim)
        return output, new_state

    def default_initial_state(self, *, batch_dims: Sequence[Dim]) -> CausalSelfAttentionState:
        """
        For causal attention.
        """
        # Note: This dim tag is wrong. It should match to the expand_dim inside __call__.
        # So the dim tag itself should be part of the layer state, and we need to define the initial value of it here.
        # This is not really supported, in various ways, also including RETURNN.
        # We just keep this code in place to be prepared for that.
        # The reason it works right now is that we do an optimization where we replace zero init state by 0.
        expand_dim = Dim(0, name="self_att_expand_dim_init")
        return CausalSelfAttentionState(
            k_accum=rf.zeros(list(batch_dims) + [expand_dim, self.num_heads, self.key_dim_per_head]),
            v_accum=rf.zeros(list(batch_dims) + [expand_dim, self.num_heads, self.value_dim_per_head]),
            accum_axis=expand_dim,
        )


def _causal_self_att_step(
    k: Tensor,
    v: Tensor,
    *,
    axis: Dim,
    state: Optional[CausalSelfAttentionState],
    self: rf.Module,
) -> Tuple[Tensor, Tensor, Dim, CausalSelfAttentionState]:
    if axis == single_step_dim:
        assert state, f"{self}: need state for single step"
        k, hist_dim = rf.cum_concat_step(k, prev_accum=state.k_accum, axis=state.accum_axis)
        v, _ = rf.cum_concat_step(v, prev_accum=state.v_accum, out_spatial_dim=hist_dim, axis=state.accum_axis)
    else:
        if state and state.accum_axis.dimension != 0:
            raise NotImplementedError(  # need to concat ...
                f"{self}: on sequence over {axis} with initial state {state} not implemented yet"
            )
        # See CumConcatLayer and https://github.com/rwth-i6/returnn/issues/391 for the idea.
        hist_dim = Dim(rf.range_over_dim(axis, device="cpu") + 1, name=f"{axis.description}:kv")
        k, _ = rf.replace_dim(k, in_dim=axis, out_dim=hist_dim)
        v, _ = rf.replace_dim(v, in_dim=axis, out_dim=hist_dim)
    new_state = CausalSelfAttentionState()
    new_state.k_accum = k
    new_state.v_accum = v
    new_state.accum_axis = hist_dim
    return k, v, hist_dim, new_state


class CausalSelfAttentionState(rf.State):
    """
    State for :class:`StepwiseCausalSelfAttention`.
    """

    def __init__(self, *_args, k_accum: Tensor = None, v_accum: Tensor = None, accum_axis: Dim = None):
        """
        :param k_accum: accumulated keys
        :param v_accum: accumulated values
        :param accum_axis:
        """
        super().__init__(*_args)
        if not _args:
            self.k_accum = k_accum
            self.v_accum = v_accum
            self.accum_axis = accum_axis


class RotaryPosSelfAttention(SelfAttention):
    """
    Rotary positional encoding (RoPE)-based self attention
    """

    def __call__(self, source: Tensor, *, axis: Dim) -> Tensor:
        """forward"""
        q, k, v = self.forward_qkv(source)

        # Apply RoPE using sinusoidal positional encoding.
        # Note: base is a bit different in rf.sinusoidal_positional_encoding (like the original)
        # vs how it's commonly used for RoPE.
        # log(base) / (dim / 2 - 1) = log(10_000) * 2 / dim
        # <=> log(base) = log(10_000) * (dim / 2 - 1) * 2 / dim = log(10_000) * (1 - 2 / dim)
        # <=> base = 10_000 ** (1 - 2 / dim)
        pos_enc = rf.sinusoidal_positional_encoding(
            spatial_dim=axis,
            feat_dim=self.key_dim_per_head,
            base=10_000 ** (1 - 2 / self.key_dim_per_head.dimension),
        )  # [T,D]
        q = _apply_rope(q, pos_enc, self.key_dim_per_head)
        k = _apply_rope(k, pos_enc, self.key_dim_per_head)

        kv_axis = Dim(None, name=f"{axis.name}-kv")
        k, _ = rf.replace_dim(k, in_dim=axis, out_dim=kv_axis)
        v, _ = rf.replace_dim(v, in_dim=axis, out_dim=kv_axis)
        output = self.attention(q, k, v, kv_axis=kv_axis)
        return output


class RotaryPosCausalSelfAttention(CausalSelfAttention):
    """
    Rotary positional encoding (RoPE)-based causal self attention
    """

    def __call__(
        self,
        source: Tensor,
        axis: Dim,
        *,
        state: Optional[CausalSelfAttentionState] = None,
    ) -> Tuple[Tensor, CausalSelfAttentionState]:
        """forward"""
        q, k, v = self.forward_qkv(source)
        k, v, hist_dim, new_state = _causal_self_att_step(k, v, axis=axis, state=state, self=self)

        # Apply RoPE using sinusoidal positional encoding.
        # Note: base is a bit different in rf.sinusoidal_positional_encoding (like the original)
        # vs how it's commonly used for RoPE.
        # log(base) / (dim / 2 - 1) = log(10_000) * 2 / dim
        # <=> log(base) = log(10_000) * (dim / 2 - 1) * 2 / dim = log(10_000) * (1 - 2 / dim)
        # <=> base = 10_000 ** (1 - 2 / dim)
        pos_enc = rf.sinusoidal_positional_encoding(
            spatial_dim=hist_dim,
            feat_dim=self.key_dim_per_head,
            base=10_000 ** (1 - 2 / self.key_dim_per_head.dimension),
        )  # [T,D]
        q = _apply_rope(
            q,
            (
                rf.gather(pos_enc, axis=hist_dim, indices=hist_dim.dyn_size_ext - 1)
                if axis == single_step_dim
                else rf.replace_dim(pos_enc, in_dim=hist_dim, out_dim=axis)[0]
            ),
            self.key_dim_per_head,
        )
        k = _apply_rope(k, pos_enc, self.key_dim_per_head)

        output = self.attention(q, k, v, kv_axis=hist_dim)
        return output, new_state


def _apply_rope(x: Tensor, pos_enc: Tensor, feat_dim: Dim) -> Tensor:
    """
    :param x: [...,T,D] or [...,D]
    :param pos_enc: [T,D] or [D]
    :param feat_dim: D
    :return: [...,T,D] or [...,D]
    """
    feat_half_dim = feat_dim.div_left(2)
    pe_imag, pe_real = rf.split(pos_enc, axis=feat_dim, out_dims=[feat_half_dim] * 2)  # [T,D/2]
    # pe_imag = sin, pe_real = cos
    d2 = Dim(2, name="complex")
    x = rf.split_dims(x, axis=feat_dim, dims=(feat_half_dim, d2))  # [...,T,D/2,2]
    x_real = rf.gather(x, indices=0, axis=d2)
    x_imag = rf.gather(x, indices=1, axis=d2)
    x_real_ = x_real * pe_real - x_imag * pe_imag
    x_imag_ = x_real * pe_imag + x_imag * pe_real
    x_, _ = rf.stack((x_real_, x_imag_), out_dim=d2)  # [...,T,D/2,2]
    x_, _ = rf.merge_dims(x_, dims=(feat_half_dim, d2), out_dim=feat_dim)  # [...,T,D]
    return x_


class RelPosSelfAttention(SelfAttentionBase):
    """
    Self-attention with relative positional encoding.
    This covers both Shawn et al. self-att rel pos 2018 (https://arxiv.org/abs/1803.02155),
    and Dai et al. Transformer-XL style 2019 (https://arxiv.org/abs/1901.02860).

    It uses :func:`relative_positional_encoding` or :class:`LearnedRelativePositionalEncoding`.

    To get Shawn et al. self-att rel pos 2018 / RETURNN SelfAttentionLayer + RelativePositionalEncodingLayer:
    - with_bias = False (at least that was the RETURNN behavior)
    - with_linear_pos = False
    - with_pos_bias = False
    - learnable_pos_emb = True
    - separate_pos_emb_per_head = False (at least that was the RETURNN default)

    To get Dai et al. Transformer-XL style 2019:
    - with_bias = False would be like the paper, however, in most implementations it is True (default)
    - with_linear_pos = True (default)
    - with_pos_bias = True (default)
    - learnable_pos_emb = True (default)
    - separate_pos_emb_per_head = True (default)

    Further details:
    https://github.com/rwth-i6/returnn_common/wiki/Relative-positional-encoding

    Code references, partly adapted from there:
    https://github.com/espnet/espnet/blob/4138010fb66ad27a43e8bee48a4932829a0847ae/espnet/nets/pytorch_backend/transformer/embedding.py#L260
    https://github.com/kimiyoung/transformer-xl/blob/44781ed21dbaec88b280f74d9ae2877f52b492a5/tf/model.py#L4
    """

    def __init__(
        self,
        in_dim: Dim,
        proj_dim: Optional[Dim],
        *,
        key_dim_total: Dim,
        value_dim_total: Dim,
        num_heads: Union[int, Dim],
        with_bias: bool = True,
        with_linear_pos: bool = True,
        with_pos_bias: bool = True,
        learnable_pos_emb: bool = False,
        learnable_pos_emb_clipping: int = 16,
        separate_pos_emb_per_head: bool = True,
        pos_emb_dropout: float = 0.0,
        att_dropout: float = 0.1,
    ):
        super().__init__(
            in_dim=in_dim,
            proj_dim=proj_dim,
            key_dim_total=key_dim_total,
            value_dim_total=value_dim_total,
            num_heads=num_heads,
            with_bias=with_bias,
            att_dropout=att_dropout,
        )
        self.separate_pos_emb_per_head = separate_pos_emb_per_head
        if with_linear_pos:
            self.pos_emb_feat_dim = self.in_dim
        elif separate_pos_emb_per_head:
            self.pos_emb_feat_dim = self.key_dim_total
        else:
            self.pos_emb_feat_dim = self.key_dim_per_head
        # linear transformation for positional encoding
        self.linear_pos = None
        if with_linear_pos:
            self.linear_pos = rf.Linear(
                self.in_dim, self.key_dim_total if separate_pos_emb_per_head else self.key_dim_per_head, with_bias=False
            )
        self.learned_pos_emb = None
        if learnable_pos_emb:
            self.learned_pos_emb = LearnedRelativePositionalEncoding(
                self.pos_emb_feat_dim, clipping=learnable_pos_emb_clipping
            )
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = None
        self.pos_bias_v = None
        if with_pos_bias:
            self.pos_bias_u = rf.Parameter((self.num_heads, self.key_dim_per_head))
            self.pos_bias_v = rf.Parameter((self.num_heads, self.key_dim_per_head))
            self.pos_bias_u.initial = rf.init.Glorot()
            self.pos_bias_v.initial = rf.init.Glorot()
        self.pos_emb_dropout = pos_emb_dropout

    def __call__(self, source: Tensor, *, axis: Dim, **_kwargs) -> Tensor:
        """forward"""
        if self.learned_pos_emb is not None:
            pos_emb, pos_emb_spatial_dim = self.learned_pos_emb(query_spatial_dim=axis, key_value_spatial_dim=axis)
        else:
            pos_emb, pos_emb_spatial_dim = relative_positional_encoding(
                query_spatial_dim=axis, key_value_spatial_dim=axis, feat_dim=self.pos_emb_feat_dim
            )
        if self.pos_emb_dropout:
            pos_emb = rf.dropout(pos_emb, self.pos_emb_dropout)
        if self.linear_pos is not None:
            pos_emb = self.linear_pos(pos_emb)
        if self.separate_pos_emb_per_head:
            pos_emb = rf.split_dims(pos_emb, axis=self.key_dim_total, dims=(self.num_heads, self.key_dim_per_head))
        # pos_emb: (head, 2*time1-1, d_k)

        q, k, v = self.forward_qkv(source)
        hist_dim = Dim(None, name=f"{axis.description}:kv")
        k, _ = rf.replace_dim(k, in_dim=axis, out_dim=hist_dim)
        v, _ = rf.replace_dim(v, in_dim=axis, out_dim=hist_dim)
        q_with_bias_u = (q + self.pos_bias_u) if self.pos_bias_u is not None else q  # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v) if self.pos_bias_v is not None else q  # (batch, head, time1, d_k)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = rf.matmul(q_with_bias_u, k, reduce=self.key_dim_per_head)

        # compute matrix b and matrix d
        # (batch, head, time1, 2*time1-1)
        matrix_bd = rf.matmul(q_with_bias_v, pos_emb, reduce=self.key_dim_per_head)
        matrix_bd = _rel_pos_enc_shift(matrix_bd, axis, pos_emb_spatial_dim, hist_dim)

        scores = matrix_ac + matrix_bd  # (batch, head, time1, time2)
        scores *= self.key_dim_per_head.dimension**-0.5
        att_weights = rf.softmax(scores, axis=hist_dim)
        att_weights = rf.dropout(att_weights, self.att_dropout, axis=self.att_dropout_broadcast and hist_dim)
        # Masking not needed because softmax should already have masked,
        # so we have 0.0 att weights for padded frames.
        att = rf.matmul(att_weights, v, reduce=hist_dim, use_mask=False)
        output, _ = rf.merge_dims(att, dims=(self.num_heads, self.value_dim_per_head), out_dim=self.value_dim_total)
        if self.proj:
            output = self.proj(output)
        return output

    # provide this func for compat with some existing code
    @staticmethod
    def _rel_shift(x: Tensor, axis: Dim, pos_emb_spatial_dim: Dim, hist_dim: Dim) -> Tensor:
        return _rel_pos_enc_shift(x, axis, pos_emb_spatial_dim, hist_dim)


def _rel_pos_enc_shift(x: Tensor, axis: Dim, pos_emb_spatial_dim: Dim, hist_dim: Dim) -> Tensor:
    """
    :param x: [B,H,T,T*2-1]
    :param axis: T
    :param pos_emb_spatial_dim: T*2-1
    :param hist_dim: T' (equal to T but separate dim)
    :return: [B,H,T,T']
    """
    batch_dims = x.remaining_dims((axis, pos_emb_spatial_dim))
    x_padded, (pos_emb_spatial_dim_,) = rf.pad(
        x, axes=[pos_emb_spatial_dim], padding=[(1, 0)], value=0.0
    )  # [B,H,T,T*2]

    # Reshape + slice trickery. You need to draw the 2D arrays on paper to understand this.
    # Also see similar trickery in :func:`window`.
    x_padded = rf.reshape(x_padded, (axis, pos_emb_spatial_dim_), (pos_emb_spatial_dim_, axis))  # [B,H,T*2,T]
    x_padded, pos_emb_spatial_dim_ = rf.slice(x_padded, axis=pos_emb_spatial_dim_, start=1)  # [B,H,T*2-1,T]
    x_padded = rf.reshape(x_padded, (pos_emb_spatial_dim_, axis), (axis, pos_emb_spatial_dim_))  # [B,H,T,T*2-1]
    x_padded, _ = rf.slice(x_padded, axis=pos_emb_spatial_dim_, size=hist_dim)  # [B,H,T,T']
    x_padded.verify_out_shape(set(batch_dims) | {axis, hist_dim})
    return x_padded


class RelPosCausalSelfAttention(CausalSelfAttention):
    """
    Self-attention with relative positional encoding.
    This covers both Shawn et al. self-att rel pos 2018 (https://arxiv.org/abs/1803.02155),
    and Dai et al. Transformer-XL style 2019 (https://arxiv.org/abs/1901.02860).

    It uses :func:`relative_positional_encoding` or :class:`LearnedRelativePositionalEncoding`.

    Same defaults as :class:`RelPosSelfAttention`, which is mostly Transformer-XL style.

    Further details:
    https://github.com/rwth-i6/returnn_common/wiki/Relative-positional-encoding

    Code references, partly adapted from there:
    https://github.com/espnet/espnet/blob/4138010fb66ad27a43e8bee48a4932829a0847ae/espnet/nets/pytorch_backend/transformer/embedding.py#L260
    https://github.com/kimiyoung/transformer-xl/blob/44781ed21dbaec88b280f74d9ae2877f52b492a5/tf/model.py#L4
    """

    def __init__(
        self,
        in_dim: Dim,
        proj_dim: Optional[Dim],
        *,
        key_dim_total: Dim,
        value_dim_total: Dim,
        num_heads: Union[int, Dim],
        with_bias: bool = True,
        with_linear_pos: bool = True,
        with_pos_bias: bool = True,
        learnable_pos_emb: bool = False,
        learnable_pos_emb_clipping: int = 16,
        separate_pos_emb_per_head: bool = True,
        pos_emb_dropout: float = 0.0,
        att_dropout: float = 0.1,
    ):
        super().__init__(
            in_dim=in_dim,
            proj_dim=proj_dim,
            key_dim_total=key_dim_total,
            value_dim_total=value_dim_total,
            num_heads=num_heads,
            with_bias=with_bias,
            att_dropout=att_dropout,
        )
        self.separate_pos_emb_per_head = separate_pos_emb_per_head
        if with_linear_pos:
            self.pos_emb_feat_dim = self.in_dim
        elif separate_pos_emb_per_head:
            self.pos_emb_feat_dim = self.key_dim_total
        else:
            self.pos_emb_feat_dim = self.key_dim_per_head
        # linear transformation for positional encoding
        self.linear_pos = None
        if with_linear_pos:
            self.linear_pos = rf.Linear(
                self.in_dim, self.key_dim_total if separate_pos_emb_per_head else self.key_dim_per_head, with_bias=False
            )
        self.learned_pos_emb = None
        if learnable_pos_emb:
            self.learned_pos_emb = LearnedRelativePositionalEncoding(
                self.pos_emb_feat_dim, clipping=learnable_pos_emb_clipping
            )
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = None
        self.pos_bias_v = None
        if with_pos_bias:
            self.pos_bias_u = rf.Parameter((self.num_heads, self.key_dim_per_head))
            self.pos_bias_v = rf.Parameter((self.num_heads, self.key_dim_per_head))
            self.pos_bias_u.initial = rf.init.Glorot()
            self.pos_bias_v.initial = rf.init.Glorot()
        self.pos_emb_dropout = pos_emb_dropout

    def __call__(
        self, source: Tensor, *, axis: Dim, state: Optional[CausalSelfAttentionState] = None
    ) -> Tuple[Tensor, CausalSelfAttentionState]:
        """forward"""
        q, k, v = self.forward_qkv(source)
        k, v, hist_dim, new_state = _causal_self_att_step(k, v, axis=axis, state=state, self=self)

        if self.learned_pos_emb is not None:
            pos_emb, pos_emb_spatial_dim = self.learned_pos_emb(query_spatial_dim=axis, key_value_spatial_dim=hist_dim)
        else:
            pos_emb, pos_emb_spatial_dim = relative_positional_encoding(
                query_spatial_dim=axis, key_value_spatial_dim=hist_dim, feat_dim=self.pos_emb_feat_dim
            )
        if self.pos_emb_dropout:
            pos_emb = rf.dropout(pos_emb, self.pos_emb_dropout)
        if self.linear_pos is not None:
            pos_emb = self.linear_pos(pos_emb)
        if self.separate_pos_emb_per_head:
            pos_emb = rf.split_dims(pos_emb, axis=self.key_dim_total, dims=(self.num_heads, self.key_dim_per_head))
        # pos_emb: (head, 2*time1-1, d_k)

        q_with_bias_u = (q + self.pos_bias_u) if self.pos_bias_u is not None else q  # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v) if self.pos_bias_v is not None else q  # (batch, head, time1, d_k)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = rf.matmul(q_with_bias_u, k, reduce=self.key_dim_per_head)

        # compute matrix b and matrix d
        # (batch, head, time1, 2*time1-1)
        matrix_bd = rf.matmul(q_with_bias_v, pos_emb, reduce=self.key_dim_per_head)
        matrix_bd = _rel_pos_enc_shift(matrix_bd, axis, pos_emb_spatial_dim, hist_dim)

        scores = matrix_ac + matrix_bd  # (batch, head, time1, time2)
        scores *= self.key_dim_per_head.dimension**-0.5
        att_weights = rf.softmax(scores, axis=hist_dim)
        att_weights = rf.dropout(att_weights, self.att_dropout, axis=self.att_dropout_broadcast and hist_dim)
        # Masking not needed because softmax should already have masked,
        # so we have 0.0 att weights for padded frames.
        att = rf.matmul(att_weights, v, reduce=hist_dim, use_mask=False)
        output, _ = rf.merge_dims(att, dims=(self.num_heads, self.value_dim_per_head), out_dim=self.value_dim_total)
        if self.proj:
            output = self.proj(output)
        return output, new_state


class CrossAttention(rf.Module):
    """
    Cross attention

    It uses :func:`dot_attention` for multi-headed dot-attention.
    """

    def __init__(
        self,
        encoder_dim: Dim,
        query_in_dim: Dim,
        proj_dim: Optional[Dim],
        *,
        key_dim_total: Dim,
        value_dim_total: Dim,
        num_heads: Union[int, Dim],
        with_bias: bool = True,
        att_dropout: float = 0.1,
        att_dropout_broadcast: Optional[bool] = None,
    ):
        """
        :param encoder_dim: encoder output dim = input dim for key-value
        :param query_in_dim: input dim for query
        :param proj_dim: if given, will add a final linear projection to this dim.
            otherwise no projection after the attention
        :param key_dim_total: total key dim. should be a multiple of num_heads
        :param value_dim_total: total value dim. should be a multiple of num_heads
        :param num_heads: number of heads
        :param with_bias: whether to add bias to qkv and proj linear projections.
            Was False in original Transformer, but many recent implementations use True by default.
            Also see: https://github.com/rwth-i6/returnn_common/issues/234.
        :param att_dropout: dropout for attention weights
        :param att_dropout_broadcast: whether to broadcast over all but ``axis``.
            normally not wanted. disabled by default since behavior version 19.
        """
        super().__init__()
        self.encoder_dim = encoder_dim
        self.query_in_dim = query_in_dim
        self.out_dim = proj_dim if proj_dim else value_dim_total
        if isinstance(num_heads, int):
            num_heads = Dim(num_heads, name="num_heads")
        self.key_dim_total = key_dim_total
        self.key_dim_per_head = key_dim_total.div_left(num_heads)
        self.value_dim_total = value_dim_total
        self.value_dim_per_head = value_dim_total.div_left(num_heads)
        self.num_heads = num_heads
        self.kv_dim_total = key_dim_total + value_dim_total
        self.kv_dim_per_head = self.key_dim_per_head + self.value_dim_per_head
        self.kv = rf.Linear(encoder_dim, self.kv_dim_total, with_bias=with_bias)
        self.q = rf.Linear(query_in_dim, self.key_dim_total, with_bias=with_bias)
        # In Fairseq MultiheadAttention, they use:
        #   nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))  (same for q_proj, v_proj),
        # where (Xavier Glorot) xavier_uniform_ means:
        #   std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        #   a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        #   _no_grad_uniform_(tensor, -a, a)
        # Out nn.init.VarianceScaling with mode="fan_avg", distribution="uniform":
        #   scale = scale * 2.0 / float(fan_in + fan_out)
        #   limit = math.sqrt(3.0 * scale)
        #   nn.random(distribution="uniform", minval=-limit, maxval=limit, ...)
        # Xavier Glorot: VarianceScaling with mode="fan_avg", distribution="uniform", scale=1.0.
        self.kv.weight.initial = rf.init.Glorot(scale=3 / 4)
        self.q.weight.initial = rf.init.Glorot(scale=1 / 2)
        # The bias init is different, but not sure how important this is.
        if proj_dim:
            self.proj = rf.Linear(value_dim_total, proj_dim, with_bias=with_bias)
        else:
            self.proj = None
        self.att_dropout = att_dropout
        if att_dropout_broadcast is None:
            att_dropout_broadcast = _att_dropout_broadcast_default()
        self.att_dropout_broadcast = att_dropout_broadcast

    def transform_encoder(self, encoder: Tensor, *, axis: Dim) -> rf.State:
        """
        Transformer encoder output. This is intended as an initial API suggestion.
        """
        k, v = self.forward_kv(encoder)
        return rf.State(k=k, v=v, kv_axis=axis)

    def forward_kv(self, source: Tensor) -> Tuple[Tensor, Tensor]:
        """
        This would be calculated once for the whole sequence (batch)
        and then always reused for :func:`attention`.

        :return: k,v
        """
        qkv = self.kv(source)
        qkv = rf.split_dims(qkv, axis=self.kv_dim_total, dims=(self.num_heads, self.kv_dim_per_head))
        k, v = rf.split(
            qkv,
            axis=self.kv_dim_per_head,
            out_dims=(self.key_dim_per_head, self.value_dim_per_head),
        )
        return k, v

    def forward_query(self, source: Tensor) -> Tensor:
        """
        This is calculated for every different query.

        :return: q
        """
        q = self.q(source)
        q = rf.split_dims(q, axis=self.key_dim_total, dims=(self.num_heads, self.key_dim_per_head))
        return q

    def __call__(self, q: Tensor, encoder: rf.State) -> Tensor:
        q = self.forward_query(q)
        return self.attention(q=q, **encoder)

    def attention(self, q: Tensor, k: Tensor, v: Tensor, *, kv_axis: Dim) -> Tensor:
        """apply attention"""
        att = dot_attention(
            q,
            k,
            v,
            key_dim=self.key_dim_per_head,
            axis=kv_axis,
            att_dropout=self.att_dropout,
            att_dropout_broadcast=self.att_dropout_broadcast,
        )
        output, _ = rf.merge_dims(att, dims=(self.num_heads, self.value_dim_per_head), out_dim=self.value_dim_total)
        if self.proj:
            output = self.proj(output)
        return output


class LearnedRelativePositionalEncoding(rf.Module):
    """
    Learnable relative positional encoding.

    E.g. as used in Shawn et al, 2018 (https://arxiv.org/abs/1803.02155).

    https://github.com/rwth-i6/returnn_common/wiki/Relative-positional-encoding
    """

    def __init__(self, feat_dim: Dim, *, clipping: int = 16, dtype: Optional[str] = None, causal: bool = False):
        """
        :param feat_dim: feature dim, for the emb matrix and output
        :param clipping: max distance to consider. emb matrix shape is [2 * clipping + 1, feat_dim] if not causal,
            else [clipping + 1, feat].
            The first and last frame will be the clipping frames.
        :param dtype: for the emb matrix and output
        """
        super(LearnedRelativePositionalEncoding, self).__init__()
        self.feat_dim = feat_dim
        self.clipping = clipping
        self.clipped_spatial_dim = Dim((1 if causal else 2) * clipping + 1, name="learned-rel-pos")
        self.causal = causal
        self.pos_emb = rf.Parameter((self.clipped_spatial_dim, self.feat_dim), dtype=dtype)

    def __call__(
        self,
        *,
        query_spatial_dim: Dim,
        key_value_spatial_dim: Dim,
        query_offset: Optional[Union[int, Tensor]] = None,
    ) -> Tuple[Tensor, Dim]:
        """
        same interface as :func:`relative_positional_encoding`

        :return: tensor of shape [spatial_dim * 2 - 1, feat_dim], and the out spatial dim (spatial_dim * 2 - 1).
            In the center is the rel pos i-j=0. All to the right are for i-j>0, all to the left for i-j<0.
        """
        indices, out_spatial_dim = _make_indices(query_spatial_dim, key_value_spatial_dim, query_offset)
        indices = rf.clip_by_value(indices, -self.clipping, 0 if self.causal else self.clipping)
        # Shift values to be >= 0. Each integer still uniquely identifies a relative position difference.
        indices = indices + self.clipping

        encoding = rf.gather(self.pos_emb, indices=indices, axis=self.clipped_spatial_dim)  # [out_spatial_dim,n_out]
        return encoding, out_spatial_dim

    def full_matrix(
        self,
        *,
        query_spatial_dim: Dim,
        key_value_spatial_dim: Dim,
        query_offset: Optional[Union[int, Tensor]] = None,
    ) -> Tensor:
        """
        :return: as full matrix [query_spatial_dim,key_value_spatial_dim,feat_dim].
            however, note that __call__ is usually to be preferred, as this gives a more efficient format.
        """
        # Very similar logic as in __call__, _make_indices.
        kv_pos_vec = rf.range_over_dim(key_value_spatial_dim)  # [kv_len]
        if query_spatial_dim == single_step_dim:
            assert query_offset is None  # not sure if any custom query offset makes sense?
            # Assume the kv are the accumulated history, and query is cur frame of it,
            # corresponding to the last frame of the kv.
            query_offset = key_value_spatial_dim.get_size_tensor() - 1
            indices = kv_pos_vec - query_offset  # [q_len,kv_len]
        else:
            q_pos_vec = rf.range_over_dim(query_spatial_dim)  # [q_len]
            indices = rf.combine_bc(kv_pos_vec, "-", q_pos_vec + query_offset)  # [q_len,kv_len]
        indices = rf.clip_by_value(indices, -self.clipping, 0 if self.causal else self.clipping)
        indices = indices + self.clipping
        encoding = rf.gather(self.pos_emb, indices=indices, axis=self.clipped_spatial_dim)  # [q_len,kv_len,n_out]
        return encoding


def _make_indices(
    query_spatial_dim: Dim,
    key_value_spatial_dim: Dim,
    query_offset: Optional[Union[int, Tensor]] = None,
) -> Tuple[Tensor, Dim]:
    kv_pos_vec = rf.range_over_dim(key_value_spatial_dim)  # [kv_len]

    # See also RelativePositionalEncodingLayer
    if query_spatial_dim == single_step_dim:
        indices = kv_pos_vec
        out_spatial_dim = key_value_spatial_dim
        assert query_offset is None  # not sure if any custom query offset makes sense?
        # Assume the kv are the accumulated history, and query is cur frame of it,
        # corresponding to the last frame of the kv.
        query_offset = key_value_spatial_dim.get_size_tensor() - 1
    else:
        query_spatial_dim_m1 = query_spatial_dim - 1
        q_pos_vec = rf.range_over_dim(query_spatial_dim_m1)  # [q_len-1]

        # We want to have all distances as in rf.combine_bc(kv_pos_vec, "-", q_pos_vec) with shape [q_len,kv_len].
        # We want to store only non-duplicates.
        # The min value is with kv_pos=0, q_pos=q_len-1: -(q_len-1)
        # The max value is with kv_pos=kv_len-1, q_pos=0: k_len-1
        indices, out_spatial_dim = rf.concat(
            (q_pos_vec - query_spatial_dim_m1.get_dim_value_tensor(), query_spatial_dim_m1),
            (kv_pos_vec, key_value_spatial_dim),
        )
    if query_offset is not None:
        indices = indices - query_offset

    return indices, out_spatial_dim


_relative_positional_encoding_cache = weakref.WeakKeyDictionary()  # run ctx -> (spatial_dim, feat_dim) -> enc


def relative_positional_encoding(
    *,
    query_spatial_dim: Dim,
    key_value_spatial_dim: Dim,
    feat_dim: Dim,
    query_offset: int = 0,
    dtype: Optional[str] = None,
) -> Tuple[Tensor, Dim]:
    """
    Implements relative positional encoding, Transformer-XL style (https://arxiv.org/abs/1901.02860),
    as used for example by :class:`RelPosSelfAttention`.

    Code references, partly adapted from there:
    https://github.com/espnet/espnet/blob/4138010fb66ad27a43e8bee48a4932829a0847ae/espnet/nets/pytorch_backend/transformer/embedding.py#L260
    https://github.com/kimiyoung/transformer-xl/blob/44781ed21dbaec88b280f74d9ae2877f52b492a5/tf/model.py#L4

    Note that this encoding is stored in a cache so that it is only calculated once.
    and then reused.

    Note that we could extend the implementation later to also buffer it
    even across mini-batches, like the ESPnet implementation does,
    e.g. by storing it in an auxiliary variable and increasing its size when needed.
    But this is not done yet, to keep the code simple.

    :return: tensor of shape [spatial_dim * 2 - 1, feat_dim], and the out spatial dim (spatial_dim * 2 - 1).
      In the center is the rel pos i-j=0. All to the right are for i-j>0, all to the left for i-j<0.
    """
    if not dtype:
        dtype = rf.get_default_float_dtype()
    cache = _relative_positional_encoding_cache.setdefault(rf.get_run_ctx(), {})
    cache_key = (query_spatial_dim, key_value_spatial_dim, feat_dim, query_offset, dtype)
    if cache_key in cache:
        return cache[cache_key]
    import math

    with rf.control_flow_ctx(None):
        # See also RelativePositionalEncodingLayer, LearnedRelativePositionalEncoding
        indices, out_spatial_dim = _make_indices(query_spatial_dim, key_value_spatial_dim, query_offset)

        feat2_dim = feat_dim.div_left(2)
        div_term = rf.exp(rf.range_over_dim(feat2_dim, dtype=dtype) * -(2.0 * math.log(1e4) / feat_dim.dimension))
        arg_sin = rf.combine_bc(rf.cast(indices, dtype), "*", div_term)
        arg_cos = arg_sin + math.pi / 2.0
        arg, feat_dim_ = rf.concat((arg_sin, feat2_dim), (arg_cos, feat2_dim))
        arg, feat_dim_ = rf.replace_dim(arg, in_dim=feat_dim_, out_dim=feat_dim)
        emb = rf.sin(arg)
        emb.verify_out_shape(
            {out_spatial_dim, feat_dim} if out_spatial_dim != single_step_dim else {feat_dim},
            allow_missing_implicit_dims=True,
        )
        emb.feature_dim = feat_dim
        cache[cache_key] = emb, out_spatial_dim
        return emb, out_spatial_dim


_sinusoidal_positional_encoding_cache = weakref.WeakKeyDictionary()  # run ctx -> (spatial_dim, feat_dim) -> enc


def sinusoidal_positional_encoding(
    *,
    spatial_dim: Dim,
    feat_dim: Dim,
    offset: Optional[Union[int, Tensor]] = None,
    base: Union[int, float] = 1e4,
    dtype: Optional[str] = None,
    device: Optional[str] = None,
) -> Tensor:
    """
    Implements absolute sinusoidal positional encoding.

    Code adopted from :func:`relative_positional_encoding`
    and our TF util :func:`get_positional_encoding`.

    Note that this encoding is stored in a cache so that it is only calculated once.
    and then reused.

    Note that we could extend the implementation later to also buffer it
    even across mini-batches, like the ESPnet implementation does,
    e.g. by storing it in an auxiliary variable and increasing its size when needed.
    But this is not done yet, to keep the code simple.

    :return: tensor of shape [spatial_dim, feat_dim] if spatial_dim != single_step_dim else [feat_dim]
    """
    if not dtype:
        dtype = rf.get_default_float_dtype()
    if not device:
        device = rf.get_default_device()
    cache = _sinusoidal_positional_encoding_cache.setdefault(rf.get_run_ctx(), {})
    cache_key = (spatial_dim, feat_dim, offset, base, dtype, device)
    if cache_key in cache:
        return cache[cache_key]
    import math

    with rf.control_flow_ctx(None):
        # See also RelativePositionalEncodingLayer, LearnedRelativePositionalEncoding
        if spatial_dim == single_step_dim:
            assert offset is not None
            indices = rf.convert_to_tensor(offset, device=device)  # scalar
        else:
            indices = rf.range_over_dim(spatial_dim, device=device)  # [len]
            if offset is not None:
                indices = indices + offset
        indices = rf.copy_to_device(indices, device)

        feat2_dim = feat_dim.div_left(2)
        div_term = rf.exp(
            rf.range_over_dim(feat2_dim, dtype=dtype, device=device) * -(math.log(base) / (feat2_dim.dimension - 1))
        )
        arg_sin = rf.combine_bc(rf.cast(indices, dtype), "*", div_term)
        arg_cos = arg_sin + math.pi / 2.0
        arg, feat_dim_ = rf.concat((arg_sin, feat2_dim), (arg_cos, feat2_dim))
        arg, feat_dim_ = rf.replace_dim(arg, in_dim=feat_dim_, out_dim=feat_dim)
        emb = rf.sin(arg)
        emb.verify_out_shape(
            {spatial_dim, feat_dim} if spatial_dim != single_step_dim else {feat_dim}, allow_missing_implicit_dims=True
        )
        emb.feature_dim = feat_dim
        cache[cache_key] = emb
        return emb


_att_dropout_broadcast_shown_warning = False


def _att_dropout_broadcast_default() -> bool:
    from returnn.config import get_global_config
    from returnn.util.basic import BehaviorVersion

    config = get_global_config(raise_exception=False)
    if config:
        opt = config.bool("rf_att_dropout_broadcast", None)
        if opt is not None:
            return opt
        opts = config.bool("rf_dropout_broadcast", None)  # also see :func:`dropout_broadcast_default`
        if opts is not None:
            return opts

    if BehaviorVersion.get() <= 18:
        global _att_dropout_broadcast_shown_warning
        if not _att_dropout_broadcast_shown_warning:
            _att_dropout_broadcast_shown_warning = True
            logging.getLogger("returnn.frontend").warning(
                "Attention dropout uses broadcasting. This is old behavior and likely not what you want."
                " Set config option 'rf_att_dropout_broadcast' to False to disable this,"
                " or switch to a new behavior version >= 19."
                " (This warning is only printed once.)"
            )
        return True  # old default
    return False
