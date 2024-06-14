"""
Some mathematical functions, in pure NumPy.
"""

from __future__ import annotations
from typing import Union, Dict
from bisect import bisect_right


def next_power_of_two(n: int) -> int:
    """next power of two, >= n"""
    return 2 ** (int(n - 1).bit_length())


class PiecewiseLinear:
    """
    Piecewise linear function.
    """

    def __init__(self, values: Dict[Union[int, float], Union[int, float]]):
        self._sorted_items = sorted(values.items())
        self._sorted_keys = [x for x, _ in self._sorted_items]
        self._sorted_values = [y for _, y in self._sorted_items]

    def __call__(self, x: Union[int, float]) -> Union[int, float]:
        steps = self._sorted_keys
        values = self._sorted_values

        i = bisect_right(steps, x)
        if i == 0:
            return values[0]
        if i < len(steps):
            step = steps[i]
            last_step = steps[i - 1]
            assert step > x >= last_step
            if last_step == x:
                return values[i - 1]
            factor = (x - last_step) / (step - last_step)
            return values[i] * factor + values[i - 1] * (1 - factor)
        return values[-1]
