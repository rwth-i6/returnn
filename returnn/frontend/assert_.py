"""
Assertion utility functions for validating conditions in Python code.
"""

from __future__ import annotations
from typing import Union
import returnn.frontend as rf
from returnn.tensor import Tensor


__all__ = ["assert_"]


def assert_(condition: Union[Tensor, bool], message: str, *, stop: bool = True) -> None:
    """
    Asserts that a given condition is True.
    If the condition is False, raises an AssertionError with the provided message.
    This runs async on GPU.

    :param condition:
    :param message:
    :param stop: if True, raises an AssertionError on failure; if False, only logs a warning
    :return: nothing
    """
    if isinstance(condition, bool):
        if not condition:
            if stop:
                raise AssertionError(message)
            else:
                print(f"[ASSERT FAILED WARNING]: {message}")

    elif isinstance(condition, Tensor):
        if condition.dims:
            condition = rf.reduce_all(condition, axis=condition.dims)  # reduce to scalar
        # noinspection PyProtectedMember
        condition._raw_backend.assert_(condition, message=message, stop=stop)

    else:
        raise TypeError(f"Condition must be a boolean or a Tensor, got {type(condition)}")
