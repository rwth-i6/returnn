"""
:func:`torch.library.custom_op` for modules with ``from __future__ import annotations``.
"""

from __future__ import annotations

from typing import Callable, Iterable
import typing

import torch


def custom_op(name: str, *, mutates_args: Iterable[str]) -> Callable[[Callable], Callable]:
    """
    Like :func:`torch.library.custom_op` as a decorator,
    but resolves the string annotations of ``from __future__ import annotations`` first,
    in the globals of the decorated function.
    torch < 2.7 cannot infer the op schema from string annotations
    (torch 2.4 accepts none, torch 2.5 and 2.6 resolve them in their own module only).

    :param name: op name, "namespace::name"
    :param mutates_args: names of the arguments the op mutates
    :return: decorator, which returns the custom op
    """

    def _decorator(fn: Callable) -> Callable:
        fn.__annotations__ = typing.get_type_hints(fn)
        return torch.library.custom_op(name, fn, mutates_args=mutates_args)

    return _decorator
