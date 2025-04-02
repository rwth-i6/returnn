"""
Some mathematical functions, in pure NumPy.
"""

from __future__ import annotations
from typing import Union, Optional, Any, Sequence, Dict
import numpy
import hashlib
import bisect


def ceil_div(a: int, b: int) -> int:
    """ceil(a / b)"""
    return -(-a // b)


def next_power_of_two(n: int) -> int:
    """next power of two, >= n"""
    return 2 ** (int(n - 1).bit_length())


class PiecewiseLinear:
    """
    Piecewise linear function.
    (Basically wraps ``numpy.interp``.)
    """

    def __init__(
        self,
        values: Dict[Union[int, float], Union[int, float]],
        *,
        kw_name: Optional[str] = None,
        ignore_other_kwargs: bool = False,
    ):
        """
        :param values: dict x -> y. Everything between the x values is linearly interpolated.
            Everything outside is repeated from the nearest x value.
        :param kw_name: keyword argument name to use in the __call__. Other keyword arguments are ignored.
        :param ignore_other_kwargs: if True, ignore other keyword arguments in the __call__.
        """
        self._sorted_items = sorted(values.items())
        self._sorted_keys = numpy.array([x for x, _ in self._sorted_items])
        self._sorted_values = numpy.array([y for _, y in self._sorted_items])
        self._kw_name = kw_name
        self._ignore_other_kwargs = ignore_other_kwargs

    def __getstate__(self):
        # Note: I was implementing __getnewargs_ex__, but we cannot use this because of this Sisyphus bug:
        # https://github.com/rwth-i6/sisyphus/issues/215
        kwargs = {"values": dict(self._sorted_items)}
        if self._kw_name is not None:
            kwargs["kw_name"] = self._kw_name
        if self._ignore_other_kwargs:
            kwargs["ignore_other_kwargs"] = True
        return kwargs

    def __setstate__(self, state):
        self.__init__(**state)

    def __repr__(self) -> str:
        kwargs = self.__getstate__()
        values = kwargs.pop("values")
        all_args_str = ", ".join([repr(values)] + [f"{k}={v!r}" for k, v in kwargs.items()])
        return f"{self.__class__.__name__}({all_args_str})"

    def __call__(self, *args: Union[int, float], **kwargs) -> Union[int, float]:
        if self._kw_name:
            if args:
                raise TypeError(f"{self}: Expected zero positional arguments, got {args!r}")
            x = kwargs.pop(self._kw_name, None)
        else:
            if len(args) != 1:
                raise TypeError(f"{self}: Expected one positional argument, got {args!r}")
            x = args[0]
        if not self._ignore_other_kwargs:
            if kwargs:
                raise TypeError(f"{self}: Unexpected keyword arguments: {kwargs!r}")
        assert x is not None
        steps = self._sorted_keys
        values = self._sorted_values
        return numpy.interp(x, steps, values)


class StepFunction:
    """
    Step function. (Piecewise constant function.)

    It will find the nearest x value and return the corresponding y value.
    The corresponding y value can be anything here, not just numbers.
    """

    def __init__(
        self,
        values: Dict[Union[int, float], Any],
        *,
        kw_name: Optional[str] = None,
        ignore_other_kwargs: bool = False,
    ):
        """
        :param values: dict x -> y. Everything between the x values is the value of the nearest x.
        :param kw_name: keyword argument name to use in the __call__. Other keyword arguments are ignored.
        :param ignore_other_kwargs: if True, ignore other keyword arguments in the __call__.
        """
        assert values, f"{self}: No values provided"
        self._values = values
        self._sorted_keys = sorted(values.keys())
        self._kw_name = kw_name
        self._ignore_other_kwargs = ignore_other_kwargs

    def __getstate__(self):
        # Note: I was implementing __getnewargs_ex__, but we cannot use this because of this Sisyphus bug:
        # https://github.com/rwth-i6/sisyphus/issues/215
        kwargs = {"values": self._values}
        if self._kw_name is not None:
            kwargs["kw_name"] = self._kw_name
        if self._ignore_other_kwargs:
            kwargs["ignore_other_kwargs"] = True
        return kwargs

    def __setstate__(self, state):
        self.__init__(**state)

    def __repr__(self) -> str:
        kwargs = self.__getstate__()
        values = kwargs.pop("values")
        all_args_str = ", ".join([repr(values)] + [f"{k}={v!r}" for k, v in kwargs.items()])
        return f"{self.__class__.__name__}({all_args_str})"

    def __call__(self, *args: Union[int, float], **kwargs) -> Union[int, float]:
        if self._kw_name:
            if args:
                raise TypeError(f"{self}: Expected zero positional arguments, got {args!r}")
            x = kwargs.pop(self._kw_name, None)
        else:
            if len(args) != 1:
                raise TypeError(f"{self}: Expected one positional argument, got {args!r}")
            x = args[0]
        if not self._ignore_other_kwargs:
            if kwargs:
                raise TypeError(f"{self}: Unexpected keyword arguments: {kwargs!r}")
        assert x is not None
        keys = self._sorted_keys
        assert keys, f"{self}: No keys provided"
        i = bisect.bisect_left(keys, x)
        # keys[:i] have e < x, and all e in keys[i:] have e >= x
        # keys[i] is the smallest e >= x
        if i >= len(keys) or (i > 0 and (x - keys[i - 1]) <= (keys[i] - x)):
            i -= 1
        return self._values[keys[i]]


def simplify_and_format_number(n: Union[int, float]) -> str:
    """Format a number, removing trailing zeros and the decimal point if it is an integer"""
    if isinstance(n, (int, float)):
        return str(n).rstrip("0").rstrip(".")
    else:
        raise TypeError(f"Expected int or float, got {n!r} type {type(n)}")


def merge_random_seeds(data_sources: Sequence[int], *, num_bytes: int = 4, signed: bool = False) -> int:
    """
    :param data_sources: A list of integers. We expect that they are all representable as 64-bit signed integers.
    :param num_bytes: for the output seed.
    :param signed: whether the output seed should be signed.
    :return: A num_bytes*8-bit integer seed, deterministically derived from the input data.
    """
    # Convert each integer to bytes and concatenate them
    combined = b"".join(int(source).to_bytes(8, "big", signed=True) for source in data_sources)
    # Use SHA-256 to hash the combined bytes
    hash_digest = hashlib.sha256(combined).digest()
    # Convert the hash digest to an integer seed
    seed = int.from_bytes(hash_digest[:num_bytes], "big", signed=signed)
    return seed
