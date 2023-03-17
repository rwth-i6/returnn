"""
DType helpers
"""

from __future__ import annotations


__all__ = ["get_default_int_dtype", "get_default_float_dtype"]


_default_int_dtype: str = "int32"
_default_float_dtype: str = "float32"


def get_default_int_dtype() -> str:
    """
    :return: default dtype for int
    """
    return _default_int_dtype


def get_default_float_dtype() -> str:
    """
    :return: default dtype for float
    """
    return _default_float_dtype
