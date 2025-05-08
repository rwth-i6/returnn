"""
Device handling.
"""

from __future__ import annotations
from typing import Optional
from contextlib import contextmanager
from returnn.tensor import Tensor


__all__ = ["copy_to_device", "get_default_device", "set_default_device", "set_default_device_ctx"]


_default_device: Optional[str] = None


def copy_to_device(x: Tensor, device: Optional[str] = None) -> Tensor:
    """
    Copy tensor to device.

    :param x: tensor
    :param device:
    :return: tensor on device
    """
    if not device:
        device = get_default_device()
    if not device:
        return x
    if x.raw_tensor is None:
        return x
    if x.device == device:
        return x
    # noinspection PyProtectedMember
    return x._raw_backend.copy_to_device(x, device)


def get_default_device() -> Optional[str]:
    """
    :return: default device, where to put new tensors (via random number generators, constant, range_over_dim, etc)
    """
    return _default_device


def set_default_device(device: Optional[str]):
    """
    :param device: see :func:`get_default_device`
    """
    global _default_device
    _default_device = device


@contextmanager
def set_default_device_ctx(device: Optional[str]):
    """
    :param device: see :func:`get_default_device`
    """
    global _default_device
    old_device = _default_device
    try:
        _default_device = device
        yield
    finally:
        _default_device = old_device
