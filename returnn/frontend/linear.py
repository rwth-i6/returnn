"""
Provides the :class:`Linear` module.
"""

from __future__ import annotations
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim


__all__ = ["Linear", "Embedding"]


class Linear(rf.Module):
    """
    Linear transformation.
    """

    def __init__(self, in_dim: Dim, out_dim: Dim, *, with_bias=True):
        super().__init__()
        assert isinstance(in_dim, Dim) and isinstance(out_dim, Dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = rf.Parameter((rf.dim_match_priority_when_needed(self.in_dim, self.out_dim), self.out_dim))
        self.weight.initial = rf.init.Glorot()
        self.with_bias = with_bias
        self.bias = None
        if with_bias:
            self.bias = rf.Parameter((self.out_dim,))
            self.bias.initial = 0.0

    def __call__(self, source: Tensor) -> Tensor:
        if not isinstance(source, Tensor):
            raise TypeError(f"{self}: source must be a Tensor but got {type(source)}")
        if self.in_dim not in source.dims:
            raise ValueError(f"{self}: input {source} does not have in_dim {self.in_dim}")
        out = rf.matmul(source, self.weight, reduce=self.in_dim)
        out.feature_dim = self.out_dim
        if self.with_bias:
            out += self.bias
        return out


class Embedding(rf.Module):
    """
    Embedding.
    """

    def __init__(self, in_dim: Dim, out_dim: Dim):
        super().__init__()
        assert isinstance(in_dim, Dim) and isinstance(out_dim, Dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = rf.Parameter((rf.dim_match_priority_when_needed(self.in_dim, self.out_dim), self.out_dim))
        self.weight.initial = rf.init.Glorot()  # TODO this is non-standard, maybe suboptimal, check this...

    def __call__(self, source: Tensor) -> Tensor:
        if not isinstance(source, Tensor):
            raise TypeError(f"{self}: source must be a Tensor but got {type(source)}")
        if self.in_dim != source.sparse_dim:
            raise ValueError(f"{self}: input {source} does not have in_dim {self.in_dim}")
        out = rf.gather(self.weight, indices=source, axis=self.in_dim)
        out.feature_dim = self.out_dim
        return out
