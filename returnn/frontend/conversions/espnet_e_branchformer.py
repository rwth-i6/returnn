"""
Import ESPnet E-Branchformer model parameters
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Union
import returnn.frontend as rf
from returnn.frontend.encoder.e_branchformer import EBranchformerLayer, FeedForwardConvGated
from returnn.frontend.decoder.transformer import FeedForward

if TYPE_CHECKING:
    import torch
    from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoderLayer, ConvolutionalGatingMLP
    from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
        PositionwiseFeedForward,
    )
    from espnet.nets.pytorch_backend.transformer.attention import (
        MultiHeadedAttention,
        RelPositionMultiHeadedAttention,
    )


def import_params_espnet_e_branchformer_layer_to_rf(
    model_espnet: EBranchformerEncoderLayer, model_rf: EBranchformerLayer
):
    """
    Import params from ESPnet E-Branchformer layer to
    RF :class:`returnn.frontend.encoder.e_branchformer.EBranchformerLayer`.
    """
    from .torch_nn import (
        import_params_torch_conv1d_to_rf,
        import_params_torch_layer_norm_to_rf,
        import_params_torch_linear_to_rf,
    )
    from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoderLayer
    from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
        PositionwiseFeedForward,
    )

    assert isinstance(model_espnet, EBranchformerEncoderLayer)
    assert isinstance(model_rf, EBranchformerLayer)

    assert isinstance(model_espnet.feed_forward, PositionwiseFeedForward)
    assert isinstance(model_espnet.feed_forward_macaron, PositionwiseFeedForward)

    import_params_espnet_positionwise_feed_forward_to_rf(model_espnet.feed_forward_macaron, model_rf.ffn1)
    import_params_espnet_positionwise_feed_forward_to_rf(model_espnet.feed_forward, model_rf.ffn2)

    import_params_torch_layer_norm_to_rf(model_espnet.norm_ff_macaron, model_rf.ffn1_layer_norm)
    import_params_torch_layer_norm_to_rf(model_espnet.norm_ff, model_rf.ffn2_layer_norm)
    import_params_torch_layer_norm_to_rf(model_espnet.norm_mha, model_rf.self_att_layer_norm)
    import_params_torch_layer_norm_to_rf(model_espnet.norm_mlp, model_rf.cgmlp_layer_norm)
    import_params_torch_layer_norm_to_rf(model_espnet.norm_final, model_rf.final_layer_norm)

    # noinspection PyTypeChecker
    import_params_espnet_multi_headed_attention_to_rf(model_espnet.attn, model_rf.self_att)

    # noinspection PyTypeChecker
    import_params_espnet_convolutional_gating_mlp_to_rf(model_espnet.cgmlp, model_rf.cgmlp)

    import_params_torch_conv1d_to_rf(model_espnet.depthwise_conv_fusion, model_rf.merge.depthwise_conv_fusion)
    import_params_torch_linear_to_rf(model_espnet.merge_proj, model_rf.merge.merge_proj)

    num_params_espnet = 0
    for k, v in model_espnet.named_parameters():
        num_params_espnet += v.numel()
    num_params_rf = 0
    for k, v in model_rf.named_parameters():
        num_params_rf += v.num_elements()
    assert num_params_rf == num_params_espnet, f"num params RF {num_params_rf} != params ESPnet {num_params_espnet}"


def import_params_espnet_positionwise_feed_forward_to_rf(model_espnet: PositionwiseFeedForward, model_rf: FeedForward):
    """import"""
    from .torch_nn import import_params_torch_linear_to_rf

    assert model_rf.linear_ff.with_bias and model_rf.linear_out.with_bias
    import_params_torch_linear_to_rf(model_espnet.w_1, model_rf.linear_ff)
    import_params_torch_linear_to_rf(model_espnet.w_2, model_rf.linear_out)


def import_params_espnet_multi_headed_attention_to_rf(
    model_espnet: Union[MultiHeadedAttention, RelPositionMultiHeadedAttention],
    model_rf: Union[rf.SelfAttention, rf.RelPosSelfAttention],
):
    """import"""
    import torch
    from .torch_nn import import_params_torch_linear_to_rf
    from espnet.nets.pytorch_backend.transformer.attention import (
        MultiHeadedAttention,
        RelPositionMultiHeadedAttention,
    )

    assert isinstance(model_espnet, (MultiHeadedAttention, RelPositionMultiHeadedAttention))
    assert isinstance(model_rf, (rf.SelfAttention, rf.RelPosSelfAttention))
    assert model_espnet.h == model_rf.num_heads.dimension
    assert model_espnet.d_k == model_rf.key_dim_per_head.dimension
    dim = model_espnet.d_k * model_espnet.h
    nh = model_espnet.h
    hdim = dim // nh

    with torch.no_grad():
        # Torch Linear: (out,in), but RF has (in,out).
        q = model_espnet.linear_q.weight.T.reshape(dim, nh, hdim)  # (in,h,out/h)
        k = model_espnet.linear_k.weight.T.reshape(dim, nh, hdim)  # (in,h,out/h)
        v = model_espnet.linear_v.weight.T.reshape(dim, nh, hdim)  # (in,h,out/h)
        q_bias = model_espnet.linear_q.bias.reshape(nh, hdim)  # (h,out/h)
        k_bias = model_espnet.linear_k.bias.reshape(nh, hdim)  # (h,out/h)
        v_bias = model_espnet.linear_v.bias.reshape(nh, hdim)  # (h,out/h)
        qkv = torch.cat([q, k, v], dim=2)  # (in,h,out/h*3)
        qkv = qkv.reshape(dim, 3 * dim)  # (in,out*3)
        qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=1).reshape(dim * 3)  # (out*3,)
        model_rf.qkv.weight.raw_tensor.copy_(qkv)
        model_rf.qkv.bias.raw_tensor.copy_(qkv_bias)

        import_params_torch_linear_to_rf(model_espnet.linear_out, model_rf.proj)

        if isinstance(model_espnet, RelPositionMultiHeadedAttention):
            assert isinstance(model_rf, rf.RelPosSelfAttention)
            assert model_rf.linear_pos is not None
            assert model_rf.pos_bias_u is not None and model_rf.pos_bias_v is not None

            import_params_torch_linear_to_rf(model_espnet.linear_pos, model_rf.linear_pos)
            _reorder_rel_pos_emb_espnet_to_rf_(model_rf.linear_pos.weight.raw_tensor, dim=0)
            model_rf.pos_bias_u.raw_tensor.copy_(model_espnet.pos_bias_u)
            model_rf.pos_bias_v.raw_tensor.copy_(model_espnet.pos_bias_v)
        else:
            assert not isinstance(model_rf, rf.RelPosSelfAttention)

    num_params_espnet = 0
    for k, v in model_espnet.named_parameters():
        num_params_espnet += v.numel()
    num_params_rf = 0
    for k, v in model_rf.named_parameters():
        num_params_rf += v.num_elements()
    assert num_params_rf == num_params_espnet, f"num params RF {num_params_rf} != params ESPnet {num_params_espnet}"


def _reorder_rel_pos_emb_espnet_to_rf(x: torch.Tensor, *, dim=-1) -> torch.Tensor:
    if dim < 0:
        dim += x.ndim
    assert 0 <= dim < x.ndim
    if dim != x.ndim - 1:
        x = x.transpose(dim, -1)
    # x: [..., D]
    # x feat dims is sin/cos repeated after each other
    *o, d = x.shape
    x = x.reshape(*o, d // 2, 2)  # [..., D/2, 2]
    # PT goes over indices T-1,T-2,...,0,1,2,...,T-1.
    # RF goes the other way around.
    # We don't flip here, to show that a linear transformation of the features is also fine.
    # Flipping cos has no effect.
    # Flipping sin would be equivalent to negating the positional encoding.
    x[..., 0] = -x[..., 0]
    # RF has first the sin, then the cos.
    x = x.transpose(-1, -2).reshape(*o, d)  # [..., D]
    if dim != x.ndim - 1:  # transpose back
        x = x.transpose(dim, -1)
    return x


def _reorder_rel_pos_emb_espnet_to_rf_(x: torch.Tensor, *, dim=-1):
    import torch

    with torch.no_grad():
        x.copy_(_reorder_rel_pos_emb_espnet_to_rf(x, dim=dim))


def import_params_espnet_convolutional_gating_mlp_to_rf(
    model_espnet: ConvolutionalGatingMLP, model_rf: FeedForwardConvGated
):
    """import"""
    from .torch_nn import (
        import_params_torch_linear_to_rf,
        import_params_torch_layer_norm_to_rf,
        import_params_torch_conv1d_to_rf,
    )
    from espnet2.asr.encoder.e_branchformer_encoder import ConvolutionalGatingMLP

    assert isinstance(model_espnet, ConvolutionalGatingMLP)
    assert isinstance(model_rf, FeedForwardConvGated)

    import_params_torch_linear_to_rf(model_espnet.channel_proj1[0], model_rf.linear_ff)
    _reorder_espnet_cgmlp_linear_ff_to_rf_(model_rf.linear_ff.weight.raw_tensor)
    if model_rf.linear_ff.with_bias:
        _reorder_espnet_cgmlp_linear_ff_to_rf_(model_rf.linear_ff.bias.raw_tensor)
    import_params_torch_linear_to_rf(model_espnet.channel_proj2, model_rf.linear_out)
    import_params_torch_layer_norm_to_rf(model_espnet.csgu.norm, model_rf.norm)
    import_params_torch_conv1d_to_rf(model_espnet.csgu.conv, model_rf.conv)
    assert model_espnet.csgu.linear is None

    num_params_espnet = 0
    for k, v in model_espnet.named_parameters():
        num_params_espnet += v.numel()
    num_params_rf = 0
    for k, v in model_rf.named_parameters():
        num_params_rf += v.num_elements()
    assert num_params_rf == num_params_espnet, f"num params RF {num_params_rf} != params ESPnet {num_params_espnet}"


def _reorder_espnet_cgmlp_linear_ff_to_rf_(w: torch.Tensor):
    import torch

    dims = list(w.shape)
    with torch.no_grad():
        w.copy_(w.reshape(*dims[:-1], 2, dims[-1] // 2).flip(-2).reshape(*dims))
