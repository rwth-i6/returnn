"""
Some mathematical functions, in pure NumPy.
"""


from __future__ import annotations


def next_power_of_two(n: int) -> int:
    """next power of two, >= n"""
    return 2 ** (int(n - 1).bit_length())
