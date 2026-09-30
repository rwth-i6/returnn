"""
Import the parameters from HuggingFace Qwen2 / Qwen3 models (PyTorch)
(also Llama-like models with GQA and qkv bias)
into a RF :class:`TransformerDecoder` with :class:`rf.RotaryPosGroupedQueryCausalSelfAttention`.

The per-param mapping (:func:`get_hf_qwen_param_for_rf`) works directly on a HF state dict,
so it can also be used when loading a checkpoint (e.g. RETURNN ``preload_from_files`` with a custom load func).
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Mapping, Set
import re
from returnn.frontend.decoder.transformer import TransformerDecoder

if TYPE_CHECKING:
    import torch


__all__ = ["import_params_hf_qwen_to_rf_transformer_decoder", "get_hf_qwen_param_for_rf", "hf_rope_reorder"]


def import_params_hf_qwen_to_rf_transformer_decoder(model_hf: torch.nn.Module, model_rf: TransformerDecoder):
    """
    Import params from HF Qwen2/Qwen3 (``...Model`` or ``...ForCausalLM``) to RF :class:`TransformerDecoder`.
    RF params without a HF counterpart (e.g. LoRA) are left as they are.
    """
    import torch

    config = model_hf.config
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    hf_prefix = "model." if hasattr(model_hf, "lm_head") else ""
    state_dict = model_hf.state_dict()
    used_hf: Set[str] = set()

    for name, param in model_rf.named_parameters():
        value = get_hf_qwen_param_for_rf(
            name,
            state_dict,
            head_dim=head_dim,
            tie_word_embeddings=config.tie_word_embeddings,
            hf_prefix=hf_prefix,
            used_hf_names=used_hf,
        )
        if value is None:
            continue
        assert tuple(value.shape) == tuple(param.raw_tensor.shape), (
            f"{name}: HF shape {tuple(value.shape)} != RF shape {tuple(param.raw_tensor.shape)}"
        )
        with torch.no_grad():
            param.raw_tensor.copy_(value)

    unused = set(state_dict.keys()) - used_hf
    if config.tie_word_embeddings:
        unused.discard("lm_head.weight")
    unused = {k for k in unused if not k.endswith("rotary_emb.inv_freq")}
    assert not unused, f"HF params not imported: {sorted(unused)}"


def get_hf_qwen_param_for_rf(
    rf_name: str,
    hf_state_dict: Mapping[str, torch.Tensor],
    *,
    head_dim: int,
    tie_word_embeddings: bool,
    hf_prefix: str = "model.",
    used_hf_names: Optional[Set[str]] = None,
) -> Optional[torch.Tensor]:
    """
    :param rf_name: RF param name, relative to the :class:`TransformerDecoder`, e.g. ``layers.0.self_att.q.weight``
    :param hf_state_dict: HF state dict. may be lazy (only ``__getitem__`` and ``__contains__`` are used)
    :param head_dim: attention head dim
    :param tie_word_embeddings: whether the RF (and HF) logits weight is the input embedding weight.
        RF then shares the (vocab,hidden) embedding param, so ``logits.weight`` has that layout.
    :param hf_prefix: prefix of the base model params in the HF state dict,
        e.g. ``"model."`` for ``Qwen2ForCausalLM``, ``""`` for ``Qwen2Model``
    :param used_hf_names: if given, the used HF names are added to it
    :return: the value in RF layout, or None if there is no HF counterpart (e.g. LoRA params)
    """
    import torch

    def _get(hf_name: str) -> torch.Tensor:
        if used_hf_names is not None:
            used_hf_names.add(hf_name)
        return hf_state_dict[hf_name]

    p = hf_prefix
    if rf_name == "input_embedding.weight":
        return _get(f"{p}embed_tokens.weight")  # (vocab,hidden)
    if rf_name == "logits.weight":
        if tie_word_embeddings:
            return _get(f"{p}embed_tokens.weight")  # (vocab,hidden)
        return _get("lm_head.weight").T  # (hidden,vocab)
    if rf_name == "final_layer_norm.scale":
        return _get(f"{p}norm.weight")

    m = re.fullmatch(r"layers\.(\d+)\.(.+)", rf_name)
    if not m:
        return None
    lp = f"{p}layers.{m.group(1)}."
    sub = m.group(2)
    # Torch Linear: (out,in), but RF has (in,out).
    if sub == "self_att_layer_norm.scale":
        return _get(f"{lp}input_layernorm.weight")
    if sub == "ff_layer_norm.scale":
        return _get(f"{lp}post_attention_layernorm.weight")
    if sub == "ff.linear_ff.weight":
        return torch.cat((_get(f"{lp}mlp.gate_proj.weight").T, _get(f"{lp}mlp.up_proj.weight").T), dim=1)
    if sub == "ff.linear_out.weight":
        return _get(f"{lp}mlp.down_proj.weight").T
    m = re.fullmatch(r"self_att\.(q|k|v|proj)\.(weight|bias)", sub)
    if m:
        hf_proj = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "proj": "o_proj"}[m.group(1)]
        value = _get(f"{lp}self_attn.{hf_proj}.{m.group(2)}")
        if m.group(2) == "weight":
            value = value.T
        if m.group(1) in ("q", "k"):
            value = hf_rope_reorder(value, head_dim=head_dim)
        return value
    m = re.fullmatch(r"self_att\.(q|k)_norm\.scale", sub)
    if m:
        return hf_rope_reorder(_get(f"{lp}self_attn.{m.group(1)}_norm.weight"), head_dim=head_dim)
    return None


def hf_rope_reorder(x: torch.Tensor, *, head_dim: int) -> torch.Tensor:
    """
    HF applies RoPE on the two halves of each head (``rotate_half``),
    RF on interleaved pairs (see :func:`rf.attention._apply_rope`).
    Reorder the last axis (``num_heads * head_dim``, or ``head_dim``) accordingly.
    The attention energies stay the same, as q and k get the same reordering.
    """
    return x.unflatten(-1, (-1, 2, head_dim // 2)).transpose(-1, -2).flatten(-3)
