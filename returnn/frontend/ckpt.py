"""
Gradient checkpointing
"""

from __future__ import annotations
from typing import Callable, TypeVar

__all__ = ["gradient_checkpoint"]


T = TypeVar("T")


def gradient_checkpoint(fn: Callable[..., T], *args: ...) -> T:
    """
    Computes the callable `fn` under a gradient checkpoint, i.e. does not save
    intermediate tensors and instead recomputes intermediate values during backprop
    to trade memory against compute.

    Any random state is saved and restored during invocation of `fn`.

    :param fn: the callable for computing the value at hand
    :param args: arguments for `fn`
    :return: the tensor computed by `fn`
    """

    if not args:
        raise ValueError(f"cannot gradient checkpoint without any input args")
    return args[0]._raw_backend.gradient_checkpoint(fn, args)
