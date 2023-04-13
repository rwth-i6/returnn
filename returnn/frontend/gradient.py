"""
Utilities which affect the gradient
"""

from __future__ import annotations
from returnn.tensor import Tensor


__all__ = ["stop_gradient"]


def stop_gradient(source: Tensor) -> Tensor:
    """wraps tf.stop_gradient or torch detach"""
    # noinspection PyProtectedMember
    return source._raw_backend.stop_gradient(source)
