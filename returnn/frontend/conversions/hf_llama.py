"""
Import the parameters from the HuggingFace Llama model.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import returnn.frontend as rf

if TYPE_CHECKING:
    from transformers.models.llama.modeling_llama import LlamaAttention


def import_params_hf_llama_att_to_rf_rotary_att(model_hf: LlamaAttention, model_rf: rf.RotaryPosCausalSelfAttention):
    """
    Import the parameters from the HF attention module.
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
    assert num_params_rf == num_params_hf

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
