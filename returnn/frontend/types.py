"""
Types
"""

from __future__ import annotations
from typing import Union
import numpy

__all__ = ["RawTensorTypes"]

RawTensorTypes = Union[int, float, complex, numpy.number, numpy.ndarray, bool, str]
