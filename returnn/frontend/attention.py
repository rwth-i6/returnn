"""
Attention
"""


from __future__ import annotations
from typing import Tuple, Union, Optional, Sequence
import weakref
from returnn.util.py_compat import Protocol
from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf


__all__ = [
    "AttentionFunc",
    "dot_attention",
    "SelfAttentionBase",
    "SelfAttention",
    "CausalSelfAttention",
    "CausalSelfAttentionState",
    "RelPosSelfAttention",
    "LearnedRelativePositionalEncoding",
    "relative_positional_encoding",
]


class AttentionFunc(Protocol):
    """Protocol defining a generic attention function"""

    def __call__(
        self,
        query: Tensor,
        keys: Tensor,
        values: Tensor,
        *,
        key_dim: Dim,
        axis: Dim,
        att_dropout: float = 0.1,
    ):
        ...


def dot_attention(
    query: Tensor, keys: Tensor, values: Tensor, *, key_dim: Dim, axis: Dim, att_dropout: float = 0.0
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
    :return: like values but with axis removed, and maybe any additional axes from query
    """
    query *= key_dim.dimension**-0.5
    energy = rf.matmul(query, keys, reduce=key_dim)
    att_weights = rf.softmax(energy, axis=axis)
    att_weights = rf.dropout(att_weights, att_dropout, axis=axis)
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
        if proj_dim:
            self.proj = rf.Linear(value_dim_total, proj_dim, with_bias=with_bias)
        else:
            self.proj = None
        self.att_dropout = att_dropout

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
        att = dot_attention(q, k, v, key_dim=self.key_dim_per_head, axis=kv_axis, att_dropout=self.att_dropout)
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
        state: CausalSelfAttentionState,
    ) -> Tuple[Tensor, CausalSelfAttentionState]:
        """forward"""
        assert axis == single_step_dim  # not implemented otherwise currently...
        q, k, v = self.forward_qkv(source)
        assert state
        hist_dim = Dim(None, name="kv-history")
        new_state = CausalSelfAttentionState()
        k, _ = rf.cum_concat_step(k, prev_accum=state.k_accum, out_spatial_dim=hist_dim, axis=state.accum_axis)
        v, _ = rf.cum_concat_step(v, prev_accum=state.v_accum, out_spatial_dim=hist_dim, axis=state.accum_axis)
        new_state.k_accum = k
        new_state.v_accum = v
        new_state.accum_axis = hist_dim
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


class CausalSelfAttentionState(rf.State):
    """
    State for :class:`StepwiseCausalSelfAttention`.
    """

    def __init__(self, *, k_accum: Tensor = None, v_accum: Tensor = None, accum_axis: Dim = None):
        """
        :param k_accum: accumulated keys
        :param v_accum: accumulated values
        :param accum_axis:
        """
        super().__init__()
        self.k_accum = k_accum
        self.v_accum = v_accum
        self.accum_axis = accum_axis


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
        super(RelPosSelfAttention, self).__init__(
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
            pos_emb, pos_emb_spatial_dim = self.learned_pos_emb(axis)
        else:
            pos_emb, pos_emb_spatial_dim = relative_positional_encoding(axis, self.pos_emb_feat_dim)
        if self.pos_emb_dropout:
            pos_emb = rf.dropout(pos_emb, self.pos_emb_dropout, axis=pos_emb.dims)
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
        matrix_bd = self._rel_shift(matrix_bd, axis, pos_emb_spatial_dim, hist_dim)

        scores = matrix_ac + matrix_bd  # (batch, head, time1, time2)
        scores *= self.key_dim_per_head.dimension**-0.5
        att_weights = rf.softmax(scores, axis=hist_dim)
        att_weights = rf.dropout(att_weights, self.att_dropout, axis=hist_dim)
        # Masking not needed because softmax should already have masked,
        # so we have 0.0 att weights for padded frames.
        att = rf.matmul(att_weights, v, reduce=hist_dim, use_mask=False)
        output, _ = rf.merge_dims(att, dims=(self.num_heads, self.value_dim_per_head), out_dim=self.value_dim_total)
        if self.proj:
            output = self.proj(output)
        return output

    @classmethod
    def _rel_shift(cls, x: Tensor, axis: Dim, pos_emb_spatial_dim: Dim, hist_dim: Dim) -> Tensor:
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


class LearnedRelativePositionalEncoding(rf.Module):
    """
    Learnable relative positional encoding.

    E.g. as used in Shawn et al, 2018 (https://arxiv.org/abs/1803.02155).

    https://github.com/rwth-i6/returnn_common/wiki/Relative-positional-encoding
    """

    def __init__(self, feat_dim: Dim, *, clipping: int = 16, dtype: str = "float32"):
        """
        :param feat_dim: feature dim, for the emb matrix and output
        :param clipping: max distance to consider. emb matrix shape is [2 * clipping + 1, feat_dim].
          The first and last frame will be the clipping frames.
        :param dtype: for the emb matrix and output
        """
        super(LearnedRelativePositionalEncoding, self).__init__()
        self.feat_dim = feat_dim
        self.clipping = clipping
        self.clipped_spatial_dim = Dim(2 * clipping + 1, name="learned-rel-pos")
        self.pos_emb = rf.Parameter((self.clipped_spatial_dim, self.feat_dim), dtype=dtype)

    def __call__(self, spatial_dim: Dim) -> Tuple[Tensor, Dim]:
        """
        same interface as :func:`relative_positional_encoding`

        :return: tensor of shape [spatial_dim * 2 - 1, feat_dim], and the out spatial dim (spatial_dim * 2 - 1).
          In the center is the rel pos i-j=0. All to the right are for i-j>0, all to the left for i-j<0.
        """
        out_spatial_dim = spatial_dim - 1 + spatial_dim
        mat_spatial_size = self.clipping + 1

        def _expand_emb():
            # spatial_dim > self.clipping, need to expand
            left = rf.gather(self.pos_emb, axis=self.clipped_spatial_dim, indices=0)
            right = rf.gather(
                self.pos_emb, axis=self.clipped_spatial_dim, indices=self.clipped_spatial_dim.dimension - 1
            )
            remaining_dim = spatial_dim - mat_spatial_size
            left = rf.expand_dim(left, dim=remaining_dim)
            right = rf.expand_dim(right, dim=remaining_dim)
            concat, out_spatial_dim_ = rf.concat(
                (left, remaining_dim), (self.pos_emb, self.clipped_spatial_dim), (right, remaining_dim)
            )
            concat, out_spatial_dim_ = rf.replace_dim(concat, in_dim=out_spatial_dim_, out_dim=out_spatial_dim)
            return concat

        def _cut_emb():
            # spatial_dim <= self.clipping, can cut out
            cut, _ = rf.slice(
                self.pos_emb,
                axis=self.clipped_spatial_dim,
                start=mat_spatial_size - spatial_dim.get_dim_value_tensor(),
                size=out_spatial_dim,
            )
            return cut

        emb = rf.cond(
            pred=spatial_dim.get_dim_value_tensor() > mat_spatial_size,
            true_fn=_expand_emb,
            false_fn=_cut_emb,
        )
        return emb, out_spatial_dim


_relative_positional_encoding_cache = weakref.WeakKeyDictionary()  # run ctx -> (spatial_dim, feat_dim) -> enc


def relative_positional_encoding(spatial_dim: Dim, feat_dim: Dim, *, dtype: str = None) -> Tuple[Tensor, Dim]:
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
    if (spatial_dim, feat_dim) in cache:
        return cache[(spatial_dim, feat_dim)]
    import math

    with rf.control_flow_ctx(None):
        position_pos = rf.range_over_dim(spatial_dim, dtype=dtype)
        position_neg = -spatial_dim.get_dim_value_tensor() + rf.range_over_dim(spatial_dim - 1) + 1
        position_neg = rf.cast(position_neg, dtype=dtype)
        position, out_spatial_dim = rf.concat((position_neg, spatial_dim - 1), (position_pos, spatial_dim))
        feat2_dim = feat_dim.div_left(2)
        div_term = rf.exp(rf.range_over_dim(feat2_dim, dtype=dtype) * -(2.0 * math.log(10000.0) / feat_dim.dimension))
        arg_sin = rf.combine_bc(position, "*", div_term)
        arg_cos = arg_sin + math.pi / 2.0
        arg, feat_dim_ = rf.concat((arg_sin, feat2_dim), (arg_cos, feat2_dim))
        arg, feat_dim_ = rf.replace_dim(arg, in_dim=feat_dim_, out_dim=feat_dim)
        emb = rf.sin(arg)
        emb.verify_out_shape({out_spatial_dim, feat_dim}, allow_missing_implicit_dims=True)
        emb.feature_dim = feat_dim
        cache[(spatial_dim, feat_dim)] = emb, out_spatial_dim
        return emb, out_spatial_dim
