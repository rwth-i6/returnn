"""
Device handling.
"""


from __future__ import annotations
from typing import Optional
from returnn.tensor import Tensor


def copy_to_device(x: Tensor, device: Optional[str]) -> Tensor:
    """
    Copy tensor to device.

    :param x: tensor
    :param device:
    :return: tensor on device
    """
    if not device:
        return x
    if x.raw_tensor is None:
        return x
    if x.device == device:
        return x
    # noinspection PyProtectedMember
    return x._raw_backend.copy_to_device(x, device)
