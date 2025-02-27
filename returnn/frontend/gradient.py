"""
Utilities which affect the gradient
"""

from __future__ import annotations
from typing import Optional, Union
from returnn.tensor import Tensor, Dim
from ._backend import global_backend

__all__ = [
    "set_requires_gradient",
    "gradient",
    "stop_gradient",
    "stop_gradient_scope",
    "scaled_gradient",
    "scaled_gradient_ext",
    "gradient_checkpoint_scope",
]


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


def stop_gradient_scope():
    """
    Create a stop gradient scope.
    All tensors created within this scope will have their gradient stopped.

    Example::

        a = ...
        b = ...
        with stop_gradient_scope():
            x = a + b
        y = x * c

    In this example, the tensor ``x`` will have its gradient stopped,
    i.e. the gradient of ``x`` w.r.t. ``a`` and ``b`` will be zero.

    :return: context manager which enables stopping the gradient. It supports __enter__ and __exit__,
        and the intended usage is with the `with` statement.
    """
    return global_backend.stop_gradient_scope()


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


def gradient_checkpoint_scope():
    """
    Create a gradient checkpoint scope.
    All tensors created within this scope will not be stored for backpropagation,
    but will be recomputed on the fly during backpropagation.

    Example::

        a = ...
        b = ...
        c = ...
        with gradient_checkpoint_scope():
            x = a + b
        y = x * c

    In this example, the tensor ``x`` will not be stored for backpropagation,
    i.e. the computation ``x = a + b`` will be recomputed during backpropagation.

    See :class:`returnn.torch.util.gradient_checkpoint.gradient_checkpoint_scope` for more documentation
    for the PyTorch specific implementation.

    :return: context manager which enables gradient checkpointing. It supports __enter__ and __exit__,
        and the intended usage is with the `with` statement.
    """
    return global_backend.gradient_checkpoint_scope()
