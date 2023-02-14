"""
The :class:`Tensor` class represents a tensor.
This is framework-agnostic, i.e. is fine for TensorFlow, PyTorch, or any other framework.
(Earlier, this was the class ``Data`` in RETURNN, and ``Data`` is now an alias for this class.)

This is designed in a way that is efficient for eager-mode execution such as PyTorch,
but at the same time compatible with older RETURNN code.

Discussion on the move from loosely TF-specific ``Data`` to framework-agnostic ``Tensor``:
https://github.com/rwth-i6/returnn/issues/1165
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from .dim import Dim

from ._tensor_extra import _TensorExtra, _TensorMixin


class Tensor(_TensorMixin):
    """
    Represents a tensor, in a frame-agnostic way. See the module docstring.
    """

    __slots__ = ("name", "_dims", "dtype", "sparse_dim", "_raw_tensor", "_extra")

    name: str
    _dims: Tuple[Dim]
    dtype: str
    sparse_dim: Optional[Dim]
    _raw_tensor: object
    _extra: Optional[_TensorExtra]

    def __init__(
        self,
        name: str,
        dims: Sequence[Dim],
        dtype: str,
        *,
        sparse_dim: Dim = None,
        raw_tensor: Optional[object] = None,
        **kwargs,
    ):
        self.name = name
        self._dims = tuple(dims)
        self.dtype = dtype
        self.sparse_dim = sparse_dim
        self._raw_tensor = None  # type: Optional[object]  # assignment below
        self._extra = None  # type: Optional[_TensorExtra]

        if kwargs:
            self._extra = _TensorExtra(tensor=self, **kwargs)
        if raw_tensor is not None:
            self.raw_tensor = raw_tensor


# dispatch table for frameworks for sanity_check
