"""
Tensor layout: the backend, the device and the storage layout of a tensor,
which new tensors can follow (``like`` arg of e.g. :func:`returnn.frontend.full`).
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Any, Type, Tuple
from returnn.tensor import Tensor

if TYPE_CHECKING:
    from ._backend import Backend

__all__ = ["TensorLayout", "get_tensor_layout"]


class TensorLayout:
    """
    Where and how a tensor is stored, as far as a new tensor can follow it
    (``like`` arg of :func:`returnn.frontend.full`, :func:`returnn.frontend.random`, ...):
    the backend, the device, and a backend-specific storage layout
    (e.g. the packing of a packed tensor, see :func:`returnn.frontend.pack`).
    Not the dtype and not the dims.

    Get it via :func:`get_tensor_layout`.
    Equal layouts mean that new tensors are created the same way,
    so it can be part of a cache key.
    """

    def __init__(self, *, backend: Type[Backend], device: Optional[str], storage: Optional[Any] = None):
        """
        :param backend: the backend of the tensor, which creates new tensors like it
        :param device:
        :param storage: backend-specific, None for the plain storage of the backend.
            Hashable, equal for the same storage layout.
        """
        self.backend = backend
        self.device = device
        self.storage = storage

    def __repr__(self):
        return f"TensorLayout(backend={self.backend.__name__}, device={self.device!r}, storage={self.storage!r})"

    def _key(self) -> Tuple[Any, ...]:
        return self.backend, self.device, self.storage

    def __eq__(self, other):
        if isinstance(other, TensorLayout):
            return self._key() == other._key()
        return False

    def __hash__(self):
        return hash(self._key())


def get_tensor_layout(x: Tensor) -> TensorLayout:
    """
    :param x:
    :return: the layout of ``x``, see :class:`TensorLayout`.
        Without a raw tensor: the global backend, no device, plain storage.
    """
    if x.raw_tensor is None:
        from ._backend import global_backend

        return TensorLayout(backend=global_backend.__class__, device=None)
    # noinspection PyProtectedMember
    return x._raw_backend.get_tensor_layout(x)
