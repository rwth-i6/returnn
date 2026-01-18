"""
Assertion utility functions for validating conditions in Python code.
"""

from typing import Union
from returnn.tensor import Tensor


__all__ = ["assert_"]


def assert_(condition: Union[Tensor, bool], message: str):
    """
    Asserts that a given condition is True.
    If the condition is False, raises an AssertionError with the provided message.
    This runs async on GPU.

    :param condition:
    :param message:
    :return: nothing
    """
    if isinstance(condition, bool):
        if not condition:
            raise AssertionError(message)

    elif isinstance(condition, Tensor):
        # noinspection PyProtectedMember
        condition._raw_backend.assert_(condition, message=message)

    else:
        raise TypeError(f"Condition must be a boolean or a Tensor, got {type(condition)}")
