"""
TensorArray, :class:`TensorArray`.
This is mostly intended for loops, see for example :func:`scan`.
"""

from __future__ import annotations
from typing import Optional, Any, Type
from returnn.tensor import Tensor, Dim
from ._backend import global_backend, Backend


__all__ = ["TensorArray"]


class TensorArray:
    """
    TensorArray.
    Think of this like a list of tensors.
    E.g. if each tensor has shape (B, D), and we have N tensors,
    stacking them together would give us a tensor of shape (N, B, D).
    Reversely, unstacking a tensor of shape (N, B, D) on the N axis
    would give us a list of N tensors of shape (B, D).

    We use a functional API,
    and each modifying operation (push_back) returns a new TensorArray object.
    This is to make sure it works well together with both eager-based and graph-based frameworks.

    Internally, the backend functions give us some opaque tensor array object
    (e.g. TF TensorArray, or maybe just a pure Python list of tensors in case of eager-based frameworks).
    """

    def __init__(
        self,
        tensor_template: Tensor,
        *,
        _backend_tensor_array: Optional[Any] = None,
        _backend: Optional[Type[Backend]] = None,
    ):
        self.tensor_template = tensor_template
        if _backend is None:
            _backend = global_backend
        if _backend_tensor_array is None:
            _backend_tensor_array = _backend.tensor_array_create()
        self._backend = _backend
        self._backend_tensor_array = _backend_tensor_array  # type is specific by the backend

    @classmethod
    def unstack(cls, tensor: Tensor, *, axis: Dim) -> TensorArray:
        """unstack"""
        # noinspection PyProtectedMember
        backend = tensor._raw_backend
        axis_int = tensor.get_axis_from_description(axis)
        tensor_template = tensor.copy_template().copy_template_excluding_axis(axis_int)
        return TensorArray(
            tensor_template=tensor_template,
            _backend_tensor_array=backend.tensor_array_unstack(tensor, axis=axis),
            _backend=backend,
        )

    def __getitem__(self, index: Tensor) -> Tensor:
        return self._backend.tensor_array_get_item(self._backend_tensor_array, index)

    def push_back(self, tensor: Tensor) -> TensorArray:
        """push_back"""
        assert tensor.dims == self.tensor_template.dims
        assert tensor.dtype == self.tensor_template.dtype
        assert tensor.sparse_dim == self.tensor_template.sparse_dim
        backend_tensor_array = self._backend.tensor_array_push_back(self._backend_tensor_array, tensor)
        return TensorArray(
            tensor_template=self.tensor_template, _backend_tensor_array=backend_tensor_array, _backend=self._backend
        )

    def stack(self, *, axis: Dim) -> Tensor:
        """stack"""
        return self._backend.tensor_array_stack(
            self._backend_tensor_array, axis=axis, tensor_template=self.tensor_template
        )
