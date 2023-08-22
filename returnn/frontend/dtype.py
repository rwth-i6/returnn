"""
DType helpers
"""

from __future__ import annotations


__all__ = ["get_default_float_dtype", "get_default_int_dtype", "get_default_array_index_dtype", "is_float_dtype"]


_default_float_dtype: str = "float32"
_default_int_dtype: str = "int32"


def get_default_float_dtype() -> str:
    """
    https://data-apis.org/array-api/latest/API_specification/data_types.html#default-data-types

    :return: default dtype for float
    """
    return _default_float_dtype


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
