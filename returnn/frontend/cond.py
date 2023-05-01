"""
Conditional execution.

Why did we choose this API?
To allow for both eager-based and graph-based frameworks,
and also to avoid any magic happening here.
https://github.com/rwth-i6/returnn/issues/1282
"""

from __future__ import annotations
from typing import Union, TypeVar, Callable
from returnn.tensor import Tensor


T = TypeVar("T")


def cond(pred: Union[bool, Tensor], true_fn: Callable[[], T], false_fn: Callable[[], T]) -> T:
    """
    :param pred:
    :param true_fn:
    :param false_fn:
    :return: true_fn() if pred else false_fn()
    """
    if isinstance(pred, bool):
        if pred:
            return true_fn()
        else:
            return false_fn()
    assert isinstance(pred, Tensor) and pred.dims == () and pred.dtype == "bool"
    # noinspection PyProtectedMember
    backend = pred._raw_backend
    if backend.executing_eagerly():
        if bool(pred.raw_tensor):
            return true_fn()
        else:
            return false_fn()
    return backend.cond(pred, true_fn, false_fn)
