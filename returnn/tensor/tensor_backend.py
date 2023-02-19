"""
API for tensor backends such as PyTorch, TensorFlow or also higher-level like RETURNN.

These are used whenever some operation on :class:`Tensor` is performed,
and it has a raw_tensor set.

It uses a dispatch table of the type of raw_tensor to the backend API.

The user can also register new backends for other types of raw tensors,
thus this is public.
"""

from __future__ import annotations
from typing import TypeVar, Generic, Type, Dict
from . import tensor as _t

T = TypeVar("T")


_dispatch_table = {}  # type: Dict[Type, TensorBackend]


def get_backend(tensor_type: Type[T]) -> TensorBackend[T]:
    """
    :param tensor_type:
    """
    return _dispatch_table[tensor_type]


class TensorBackend(Generic[T]):
    """
    Tensor backend API
    """

    def create_placeholder(self, tensor: _t.Tensor) -> T:
        """
        :return: tf.placeholder in TF

        This is really only for TensorFlow for the deprecated option auto_create_placeholders
        and should not be used in other backends,
        even in graph-based backends.
        Rather, the logic to create placeholders should be done elsewhere.
        """
        raise Exception(f"{self}.create_placeholder not supported")
