"""
PyTorch utility for Rotary Position Embedding (RoPE).

Provides a ``torch.compile``-compiled core kernel for applying RoPE.
The kernel is numerically identical to the plain RF reference implementation
:func:`returnn.frontend.attention._apply_rope_real`
but fuses the four element-wise operations into a single kernel launch, avoiding intermediate allocations.
It also supports all PyTorch dtypes natively (including bfloat16 and float16) without any upcasting.
"""

from __future__ import annotations
import torch


def apply_rope(x: torch.Tensor, pos_enc: torch.Tensor) -> torch.Tensor:
    """
    RoPE kernel operating on raw PyTorch tensors.

    This function is supposed to be compiled.

    :param x: input tensor ``[..., D]`` with feat dim last; any float dtype
    :param pos_enc: positional encoding ``[..., D]`` broadcast-compatible with *x*;
        first ``D/2`` entries along last axis are sin, second ``D/2`` are cos
    :return: rotated tensor with the same shape and dtype as *x*
    """
    feat_dim = x.shape[-1]
    half = feat_dim // 2
    pe_sin, pe_cos = torch.split(pos_enc, half, dim=-1)
    x_pairs = x.reshape(*x.shape[:-1], half, 2)
    x_r, x_i = torch.unbind(x_pairs, dim=-1)
    out_r = x_r * pe_cos - x_i * pe_sin
    out_i = x_r * pe_sin + x_i * pe_cos
    return torch.stack([out_r, out_i], dim=-1).reshape(x.shape)


# Wrap with torch.compile to fuse all ops into a single kernel launch, eliminating intermediate allocations.
# dynamic=True avoids recompilation for each new (batch_size, seq_len, …) combination;
# different numbers of input dimensions still produce separate traces, but that is O(ndim) compilations at most.
try:
    apply_rope = torch.compile(apply_rope, dynamic=True)
except Exception as e:
    import warnings

    warnings.warn(f"RETURNN: torch.compile for apply_rope failed: {e}", stacklevel=1)
