"""
Low-rank adaptation (LoRA), `LoRA: Low-Rank Adaptation of Large Language Models <https://arxiv.org/abs/2106.09685>`__.
"""

from __future__ import annotations
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim
from . import _utils


__all__ = ["LoRALinear"]


class LoRALinear(rf.Linear):
    """
    Linear transformation with a low-rank update::

        y = x W + b + scale * dropout(x) A B

    with ``A: [in,rank]``, ``B: [rank,out]``, ``B`` zero-initialized, so it starts as the plain linear.
    Initialization and scaling follow PEFT (``scale = alpha / rank``, or ``alpha / sqrt(rank)`` with rsLoRA).
    """

    def __init__(
        self,
        in_dim: Dim,
        out_dim: Dim,
        *,
        with_bias: bool = True,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        use_rslora: bool = False,
        freeze_base: bool = True,
    ):
        """
        :param in_dim:
        :param out_dim:
        :param with_bias: bias of the base linear
        :param rank: LoRA rank r
        :param alpha: LoRA alpha
        :param dropout: dropout on the input of the low-rank path
        :param use_rslora: rank-stabilized LoRA (https://arxiv.org/abs/2312.03732): scale ``alpha / sqrt(rank)``
        :param freeze_base: make the base weight and bias non-trainable
        """
        super().__init__(in_dim, out_dim, with_bias=with_bias)
        self.rank_dim = Dim(rank, name="lora-rank")
        self.lora_a = rf.Parameter((in_dim, self.rank_dim))
        # PEFT: kaiming_uniform_(a=sqrt(5)), i.e. U(-1/sqrt(fan_in), 1/sqrt(fan_in))
        self.lora_a.initial = rf.init.VarianceScaling(scale=1.0 / 3.0, mode="fan_in", distribution="uniform")
        self.lora_b = rf.Parameter((self.rank_dim, out_dim))
        self.lora_b.initial = 0.0
        self.lora_dropout = dropout
        self.lora_scale = alpha / (rank**0.5 if use_rslora else rank)
        if freeze_base:
            self.weight.trainable = False
            if self.bias is not None:
                self.bias.trainable = False

    def __call__(self, source: Tensor) -> Tensor:
        out = super().__call__(source)
        x = rf.dropout(source, self.lora_dropout)
        x = rf.matmul(x, self.lora_a, reduce=self.in_dim)
        x = rf.matmul(x, self.lora_b, reduce=self.rank_dim)
        out = _utils.keep_dtype(out + x * self.lora_scale, out.dtype)
        out.feature_dim = self.out_dim
        return out
