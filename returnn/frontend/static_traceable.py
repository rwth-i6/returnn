"""
Global flag requesting static traceable code.

When enabled, RF code paths should prefer a fixed (bounded) computation structure:
static shapes (upper-bound buffers, see e.g. :class:`Dim` capacity),
fixed trip counts (loop to a declared bound, gate per element)
instead of data-dependent control flow,
device-side randomness,
and no host reads of device values.
Such code can be captured into a CUDA graph and traced by torch.compile.

Note that enabling this can change random number consumption
(e.g. bounded loops draw for the gated-off iterations too),
so results differ from the default mode for a fixed seed
(the distribution is unchanged).
"""

from __future__ import annotations
import contextlib

__all__ = [
    "is_static_traceable",
    "set_static_traceable",
    "set_static_traceable_ctx",
    "static_traceable_generation",
]


_static_traceable = False
_static_traceable_generation = 0


def is_static_traceable() -> bool:
    """
    :return: whether static traceable code is requested, see the module docstring
    """
    return _static_traceable


def set_static_traceable(flag: bool):
    """
    :param flag: see the module docstring
    """
    global _static_traceable, _static_traceable_generation
    if flag and not _static_traceable:
        _static_traceable_generation += 1
    _static_traceable = flag


def static_traceable_generation() -> int:
    """
    :return: counts the static-traceable activations (each False -> True transition).
        E.g. the CUDA-graph train step (:mod:`returnn.torch.util.graph_capture`)
        enables the ctx once per train step, so within that regime this identifies the step.
        Usable to scope caches to one step (see :class:`returnn.frontend._cache.Cache`):
        results computed under the ctx are only valid within that step
        (under capture they live in the captured graph region).
    """
    return _static_traceable_generation


@contextlib.contextmanager
def set_static_traceable_ctx(flag: bool = True):
    """
    :param flag: see the module docstring
    """
    old = _static_traceable
    try:
        set_static_traceable(flag)
        yield
    finally:
        set_static_traceable(old)
