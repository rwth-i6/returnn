"""
DType helpers
"""

from __future__ import annotations
from contextlib import contextmanager


__all__ = [
    "get_default_float_dtype",
    "set_default_float_dtype",
    "set_default_float_dtype_ctx",
    "get_default_int_dtype",
    "get_default_array_index_dtype",
    "is_float_dtype",
]


_default_float_dtype: str = "float32"
_default_int_dtype: str = "int32"


def get_default_float_dtype() -> str:
    """
    https://data-apis.org/array-api/latest/API_specification/data_types.html#default-data-types

    :return: default dtype for float
    """
    return _default_float_dtype


def set_default_float_dtype(dtype: str):
    """
    Set the default float dtype

    :param dtype: the new default float dtype
    """
    global _default_float_dtype
    assert isinstance(dtype, str)
    _default_float_dtype = dtype


@contextmanager
def set_default_float_dtype_ctx(dtype: str):
    """
    :param dtype: see :func:`get_default_float_dtype`
    """
    global _default_float_dtype
    assert isinstance(dtype, str)
    old_default_float_dtype = _default_float_dtype
    try:
        _default_float_dtype = dtype
        yield
    finally:
        _default_float_dtype = old_default_float_dtype


def get_default_int_dtype() -> str:
    """
    https://data-apis.org/array-api/latest/API_specification/data_types.html#default-data-types

    :return: default dtype for int
    """
    return _default_int_dtype


def get_default_array_index_dtype() -> str:
    """
    https://data-apis.org/array-api/latest/API_specification/data_types.html#default-data-types

    :return: default dtype for array index - currently just the same as :func:`get_default_int_dtype`
    """
    return get_default_int_dtype()


def is_float_dtype(dtype: str) -> bool:
    """
    :return: whether the dtype is float, e.g. it supports backprop etc
    """
    return dtype.startswith("float")
