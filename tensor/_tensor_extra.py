"""
Backwards-compatible functions for the old ``Data`` class.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from .dim import Dim
    from .tensor import Tensor


class _TensorExtra:
    def __init__(self, *, tensor: Tensor):
        self.tensor = tensor


class _TensorMixin:
    name: str
    _dims: Tuple[Dim]
    dtype: str
    sparse_dim: Optional[Dim]
    _raw_tensor: object
    _extra: Optional[_TensorExtra]
