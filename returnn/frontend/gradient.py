"""
Utilities which affect the gradient
"""

from __future__ import annotations
from typing import Optional, Union
from returnn.tensor import Tensor, Dim


__all__ = ["set_requires_gradient", "gradient", "stop_gradient", "scaled_gradient", "scaled_gradient_ext"]


def set_requires_gradient(source: Tensor):
    """
    :param source:
    :return: nothing, modifies source in-place
    """
    # noinspection PyProtectedMember
    return source._raw_backend.set_requires_gradient(source)


def gradient(y: Tensor, x: Tensor) -> Tensor:
    """
    :param y: some scalar
    :param x: some tensor
    :return: gradient of y w.r.t. x
    """
    # noinspection PyProtectedMember
    return y._raw_backend.gradient(y, x)


def stop_gradient(source: Tensor) -> Tensor:
    """wraps tf.stop_gradient or torch detach"""
    # noinspection PyProtectedMember
    return source._raw_backend.stop_gradient(source)


def scaled_gradient(source: Tensor, scale: Union[float, Tensor]) -> Tensor:
    """
    :param source:
    :param scale: if constant 0., will use :func:`stop_gradient`.
        Can be used as gradient reversal layer (with negative factor).
    :return: source with scaled gradient
    """
    if not isinstance(scale, Tensor) and scale == 0.0:
        return stop_gradient(source)
    # noinspection PyProtectedMember
    return source._raw_backend.scaled_gradient(source, scale)


def scaled_gradient_ext(
    source: Tensor,
    *,
    scale: Union[float, Tensor],
    shift: Optional[Union[float, Tensor]] = None,
    scale_shift_by_sum_over_axis: Optional[Dim] = None,
) -> Tensor:
    """
    Just `identity` in the forward pass.
    Scales the gradient by some factor in backprop.
    Can be used as gradient reversal layer (with negative factor).
    For TF, uses :func:`returnn.tf.util.basic.scaled_gradient`, or :func:`tf.stop_gradient`

    :param source:
    :param scale: if constant 0. and no shift, will use :func:`stop_gradient`
    :param shift:
    :param scale_shift_by_sum_over_axis: if given, calculates the sum over this axis (absolute values)
        and multiplies the shift value by this sum.
    :return: source with transformed gradient
    """
    # noinspection PyProtectedMember
    return source._raw_backend.scaled_gradient_ext(
        source, scale=scale, shift=shift, scale_shift_by_sum_over_axis=scale_shift_by_sum_over_axis
    )
