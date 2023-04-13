"""
Attention
"""


from __future__ import annotations
from typing import Tuple, Union, Optional, Sequence
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
    att = rf.matmul(att_weights, values, reduce=axis, disable_masking=True)
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
