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
    Also not the axis order of the raw tensor: a new tensor gets its dims in the given order.
    But the storage layout includes everything that places the elements in the buffer,
    e.g. for a packed tensor its packing: the order of the packed dims (the first one major),
    the gap frames, the alignment, the layout lens and the buffer size (the packed dim).

    Get it via :func:`get_tensor_layout`.
    Equal layouts mean that new tensors are created the same way,
    so it can be part of a cache key.
    """

    def __init__(self, *, backend: Type[Backend], device: Optional[str], storage: Optional[Any] = None):
        """
        :param backend: the backend of the tensor, which creates new tensors like it
        :param device:
        :param storage: backend-specific, None for the plain storage of the backend.
            Hashable, equal for the same storage layout,
            and with ``cache_key_parts()`` as :func:`TensorLayout.cache_key_parts`.
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

    def cache_key_parts(self) -> Tuple[Any, ...]:
        """
        :return: the layout as key parts for :class:`returnn.frontend._cache.Cache`:
            tensors (by identity of the raw tensor), dims, nested layouts, backend classes and plain values
        """
        return self.backend, self.device, self.storage.cache_key_parts() if self.storage is not None else None


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
