"""
Some mathematical functions, in pure NumPy.
"""

from __future__ import annotations
from typing import Union, Dict
import numpy


def ceil_div(a: int, b: int) -> int:
    """ceil(a / b)"""
    return -(-a // b)


def next_power_of_two(n: int) -> int:
    """next power of two, >= n"""
    return 2 ** (int(n - 1).bit_length())


class PiecewiseLinear:
    """
    Piecewise linear function.
    """

    def __init__(self, values: Dict[Union[int, float], Union[int, float]]):
        self._sorted_items = sorted(values.items())
        self._sorted_keys = numpy.array([x for x, _ in self._sorted_items])
        self._sorted_values = numpy.array([y for _, y in self._sorted_items])

    def __call__(self, x: Union[int, float]) -> Union[int, float]:
        assert x is not None
        steps = self._sorted_keys
        values = self._sorted_values
        return numpy.interp(x, steps, values)
