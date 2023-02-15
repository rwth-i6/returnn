"""
Backwards-compatible functions and attribs for the old ``Dim`` class,
or just rarely used attribs, such that we can save memory for the common case.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple

from . import dim as _d
from . import tensor as _t


class _DimExtra:
    def __init__(self, *, dim: _d.Dim):
        self.dim = dim


class _DimMixin:
    pass
