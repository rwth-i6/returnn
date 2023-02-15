"""
Backwards-compatible functions and attribs for the old ``Dim`` class,
or just rarely used attribs, such that we can save memory for the common case.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple

from .dim import Dim
from .tensor import Tensor


class _DimExtra:
    def __init__(self, *, dim: Dim):
        self.dim = dim


class _DimMixin:
    pass
