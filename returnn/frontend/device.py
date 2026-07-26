"""
Device handling.
"""

from __future__ import annotations
from typing import Optional
from contextlib import contextmanager
from returnn.tensor import Tensor


__all__ = [
    "copy_to_device",
    "get_default_device",
    "set_default_device",
    "set_default_device_ctx",
    "get_default_dim_size_device",
]


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
    if device == "cpu" and x.device is not None:
        import returnn.frontend as rf

        if rf.is_static_traceable():
            raise Exception(
                f"copy_to_device: device-to-host copy of {x} under static traceable"
                f" (a sync, illegal under CUDA-graph capture)."
                f" Compute on the data's device instead (pass the proper device at the call site)."
            )
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


def get_default_dim_size_device() -> Optional[str]:
    """
    :return: default device, where to put new tensors for dim sizes (Dim.dyn_size_ext)
    """
    return "cpu"
