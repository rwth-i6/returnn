"""
Import the parameters from the HuggingFace Llama model (PyTorch).
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Union
import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder, TransformerDecoderLayer, FeedForwardGated

if TYPE_CHECKING:
    from transformers.models.llama.modeling_llama import (
        LlamaModel,
        LlamaForCausalLM,
        LlamaDecoderLayer,
        LlamaMLP,
        LlamaRMSNorm,
        LlamaAttention,
    )


def import_params_hf_llama_to_rf_transformer_decoder(
    model_hf: Union[LlamaModel, LlamaForCausalLM], model_rf: TransformerDecoder
):
    """
    Import params from HF Llama model to RF :class:`TransformerDecoder`.
    """
    import torch
    from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM, LlamaDecoderLayer

    print("HF Model:")
    print(model_hf)
    print("Parameters:")
    num_params_hf = 0
    for k, v in model_hf.named_parameters():
        print(f"{k}: {list(v.shape)} {v.dtype}")
        num_params_hf += v.numel()
    print("Total number of parameters:", num_params_hf)

    print("RF Model:")
    print(model_rf)
    print("Parameters:")
    num_params_rf = 0
    for k, v in model_rf.named_parameters():
        print(f"{k}: {list(v.dims)} {v.dtype}")
        assert isinstance(v.raw_tensor, torch.nn.Parameter)
        num_params_rf += v.num_elements()
    print("Total number of parameters:", num_params_rf)
    # Check if the number of parameters is the same below.
    # First import individual sub modules.
    # We might detect any mismatches there, and this will easy the debugging.

    lm_head = None
    if isinstance(model_hf, LlamaForCausalLM):
        lm_head = model_hf.lm_head
        model_hf = model_hf.model
    else:
        # Exclude logits.
        num_params_rf -= model_rf.logits.weight.num_elements()
    assert isinstance(model_hf, LlamaModel)
    assert model_hf.norm.weight.shape[0] == model_rf.model_dim.dimension

    assert len(model_hf.layers) == len(model_rf.layers)
    for i, (layer_hf, layer_rf) in enumerate(zip(model_hf.layers, model_rf.layers)):
        assert isinstance(layer_hf, LlamaDecoderLayer)
        assert isinstance(layer_rf, TransformerDecoderLayer)
        import_params_hf_llama_decoder_layer_to_rf(layer_hf, layer_rf)

    assert model_hf.embed_tokens.weight.shape == model_rf.input_embedding.weight.raw_tensor.shape
    with torch.no_grad():
        model_rf.input_embedding.weight.raw_tensor.copy_(model_hf.embed_tokens.weight)  # (vocab,hidden)

    assert isinstance(model_rf.final_layer_norm, rf.RMSNorm)
    import_params_hf_llama_rms_norm_to_rf(model_hf.norm, model_rf.final_layer_norm)

    if lm_head is not None:
        assert lm_head.bias is None and model_rf.logits.bias is None  # not implemented
        # Torch Linear: (out,in), but RF has (in,out).
        with torch.no_grad():
            model_rf.logits.weight.raw_tensor.copy_(lm_head.weight.T)  # (hidden,vocab)

    assert num_params_rf == num_params_hf, f"missmatch num params: RF {num_params_rf} != HF {num_params_hf}"


def import_params_hf_llama_decoder_layer_to_rf(model_hf: LlamaDecoderLayer, model_rf: TransformerDecoderLayer):
    """
    Import the parameters from the HF Llama decoder layer.
    """
    import torch

    assert model_hf.hidden_size == model_rf.out_dim.dimension

    print("HF Model:")
    print(model_hf)
    print("Parameters:")
    num_params_hf = 0
    for k, v in model_hf.named_parameters():
        print(f"{k}: {list(v.shape)} {v.dtype}")
        num_params_hf += v.numel()
    print("Total number of parameters:", num_params_hf)

    print("RF Model:")
    print(model_rf)
    print("Parameters:")
    num_params_rf = 0
    for k, v in model_rf.named_parameters():
        print(f"{k}: {list(v.dims)} {v.dtype}")
        assert isinstance(v.raw_tensor, torch.nn.Parameter)
        num_params_rf += v.num_elements()
    print("Total number of parameters:", num_params_rf)
    # Check if the number of parameters is the same below.
    # First import individual sub modules.
    # We might detect any mismatches there, and this will easy the debugging.

    assert isinstance(model_rf.ff, FeedForwardGated), f"unexpected: {model_rf.ff}"
    import_params_hf_llama_mlp_to_rf_feed_forward_gated(model_hf.mlp, model_rf.ff)

    assert isinstance(model_rf.self_att, rf.RotaryPosCausalSelfAttention), f"unexpected: {model_rf.self_att}"
    import_params_hf_llama_att_to_rf_rotary_att(model_hf.self_attn, model_rf.self_att)

    assert isinstance(model_rf.self_att_layer_norm, rf.RMSNorm), f"unexpected: {model_rf.self_att_layer_norm}"
    import_params_hf_llama_rms_norm_to_rf(model_hf.input_layernorm, model_rf.self_att_layer_norm)

    assert isinstance(model_rf.ff_layer_norm, rf.RMSNorm), f"unexpected: {model_rf.ff_layer_norm}"
    import_params_hf_llama_rms_norm_to_rf(model_hf.post_attention_layernorm, model_rf.ff_layer_norm)

    assert num_params_rf == num_params_hf


def import_params_hf_llama_mlp_to_rf_feed_forward_gated(model_hf: LlamaMLP, model_rf: FeedForwardGated):
    """
    Import the parameters from the HF Llama MLP module.
    """
    import torch

    assert model_hf.hidden_size == model_rf.out_dim.dimension == model_rf.linear_ff.in_dim.dimension

    print("HF Model:")
    print(model_hf)
    print("Parameters:")
    num_params_hf = 0
    for k, v in model_hf.named_parameters():
        print(f"{k}: {list(v.shape)} {v.dtype}")
        num_params_hf += v.numel()
    print("Total number of parameters:", num_params_hf)

    print("RF Model:")
    print(model_rf)
    print("Parameters:")
    num_params_rf = 0
    for k, v in model_rf.named_parameters():
        print(f"{k}: {list(v.dims)} {v.dtype}")
        assert isinstance(v.raw_tensor, torch.nn.Parameter)
        num_params_rf += v.num_elements()
    print("Total number of parameters:", num_params_rf)
    assert num_params_rf == num_params_hf

    # Torch Linear: (out,in), but RF has (in,out).
    w1 = model_hf.gate_proj.weight.T  # (in,out)
    w2 = model_hf.up_proj.weight.T  # (in,out)
    w3 = model_hf.down_proj.weight.T  # (out,in)
    assert model_hf.gate_proj.bias is None  # not implemented
    assert model_hf.up_proj.bias is None  # not implemented
    assert model_hf.down_proj.bias is None  # not implemented
    with torch.no_grad():
        w = torch.cat((w1, w2), dim=1)  # (in,out*2)
        model_rf.linear_ff.weight.raw_tensor.copy_(w)
        model_rf.linear_out.weight.raw_tensor.copy_(w3)


def import_params_hf_llama_rms_norm_to_rf(model_hf: LlamaRMSNorm, model_rf: rf.RMSNorm):
    """
    Import the parameters from the HF Llama RMSNorm module.
    """
    import torch

    assert model_hf.weight.shape[0] == model_rf.in_dim.dimension

    print("HF Model:")
    print(model_hf)
    print("Parameters:")
    num_params_hf = 0
    for k, v in model_hf.named_parameters():
        print(f"{k}: {list(v.shape)} {v.dtype}")
        num_params_hf += v.numel()
    print("Total number of parameters:", num_params_hf)

    print("RF Model:")
    print(model_rf)
    print("Parameters:")
    num_params_rf = 0
    for k, v in model_rf.named_parameters():
        print(f"{k}: {list(v.dims)} {v.dtype}")
        assert isinstance(v.raw_tensor, torch.nn.Parameter)
        num_params_rf += v.num_elements()
    print("Total number of parameters:", num_params_rf)
    assert num_params_rf == num_params_hf

    w = model_hf.weight  # (in,)
    with torch.no_grad():
        model_rf.scale.raw_tensor.copy_(w)


def import_params_hf_llama_att_to_rf_rotary_att(model_hf: LlamaAttention, model_rf: rf.RotaryPosCausalSelfAttention):
    """
    Import the parameters from the HF Llama attention module.
    """
    import torch

    assert model_hf.num_heads == model_rf.num_heads.dimension
    assert model_hf.hidden_size == model_rf.in_dim.dimension
    dim = model_hf.hidden_size
    nh = model_hf.num_heads
    hdim = dim // nh

    print("HF Model:")
    print(model_hf)
    print("Parameters:")
    num_params_hf = 0
    for k, v in model_hf.named_parameters():
        print(f"{k}: {list(v.shape)} {v.dtype}")
        num_params_hf += v.numel()
    print("Total number of parameters:", num_params_hf)

    print("RF Model:")
    print(model_rf)
    print("Parameters:")
    num_params_rf = 0
    for k, v in model_rf.named_parameters():
        print(f"{k}: {list(v.dims)} {v.dtype}")
        assert isinstance(v.raw_tensor, torch.nn.Parameter)
        num_params_rf += v.num_elements()
    print("Total number of parameters:", num_params_rf)
    assert num_params_rf == num_params_hf, f"num params RF {num_params_rf} != params HF {num_params_hf}"

    # Torch Linear: (out,in), but RF has (in,out).
    q = model_hf.q_proj.weight.T.reshape(dim, nh, hdim)  # (in,h,out/h)
    k = model_hf.k_proj.weight.T.reshape(dim, nh, hdim)  # (in,h,out/h)
    v = model_hf.v_proj.weight.T.reshape(dim, nh, hdim)  # (in,h,out/h)
    q = q.reshape(dim, nh, 2, hdim // 2).transpose(-1, -2).flatten(-2)  # reorder complex numbers
    k = k.reshape(dim, nh, 2, hdim // 2).transpose(-1, -2).flatten(-2)  # reorder complex numbers
    qkv = torch.cat([q, k, v], dim=2)  # (in,h,out/h*3)
    qkv = qkv.reshape(dim, 3 * dim)
    assert model_hf.q_proj.bias is None  # not implemented
    with torch.no_grad():
        model_rf.qkv.weight.raw_tensor.copy_(qkv)
        model_rf.proj.weight.raw_tensor.copy_(model_hf.o_proj.weight.T)
