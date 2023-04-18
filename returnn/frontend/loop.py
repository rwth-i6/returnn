"""
While loop.

Why did we choose this API?
To allow for both eager-based and graph-based frameworks,
and also to avoid any magic happening here.
https://github.com/rwth-i6/returnn/issues/1282
"""

from __future__ import annotations


__all__ = ["while_loop"]


def while_loop(cond, body, loop_vars):
    """
    It executes::

        while cond(loop_vars):
            loop_vars = body(loop_vars)

    :param cond:
    :param body:
    :param loop_vars: initial loop vars
    :return: final loop vars
    """
    # noinspection PyProtectedMember
    backend = loop_vars[0]._raw_backend
    if backend.executing_eagerly():
        while cond(loop_vars):
            loop_vars = body(loop_vars)
        return loop_vars
    raise NotImplementedError("while_loop() not implemented for backend %r" % backend)
