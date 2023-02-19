"""
API for tensor backends such as PyTorch, TensorFlow or also higher-level like RETURNN.

These are used whenever some operation on :class:`Tensor` is performed,
and it has a raw_tensor set.

It uses a dispatch table of the type of raw_tensor to the backend API.

The user can also register new backends for other types of raw tensors,
thus this is public.
"""

from __future__ import annotations


class TensorBackend:
    """
    Tensor backend API
    """
