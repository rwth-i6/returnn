"""
Public backend-related functions.
"""

from __future__ import annotations

from . import _backend

# And some functions from the internal backend API.
from ._backend import select_backend_torch, select_backend_returnn_layers_tf

__all__ = ["select_backend_torch", "select_backend_returnn_layers_tf", "is_backend_raw_tensor_dim_tag_independent"]


def is_backend_raw_tensor_dim_tag_independent() -> bool:
    """
    :return: whether raw tensors of the backend are independent of :class:`Dim`
        (Usually yes, e.g. :class:`tf.Tensor` or :class:`torch.Tensor`,
        but the TF-layers backend is an exception.)
    """
    return _backend.global_backend.is_backend_raw_tensor_dim_tag_independent
