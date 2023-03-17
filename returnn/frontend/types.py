"""
Types
"""

from __future__ import annotations
from typing import Union
import numpy
from returnn.tensor import Tensor, Dim

__all__ = ["RawTensorTypes", "Tensor", "Dim"]

RawTensorTypes = Union[int, float, complex, numpy.number, numpy.ndarray, bool, str]
