"""
Scaled gradients for backward pass.
This also covers gradient reversal, which is simply the case with scale=-1.
We actually extend the simple scaling by some further optional transformations like shifting.

The code is adapted from our TF implementation, see :func:`returnn.tf.util.basic.scaled_gradient`.

For some discussion on the specific implementation, see:
https://discuss.pytorch.org/t/gradient-scaling-reversal/186392

Also see other reference implementations:
https://github.com/facebookresearch/fairseq/blob/100cd91db19bb/fairseq/modules/grad_multiply.py
https://github.com/janfreyberg/pytorch-revgrad/blob/449fa763a76d/src/pytorch_revgrad/functional.py
https://github.com/tadeephuy/GradientReversal/blob/5d9857d63/gradient_reversal/functional.py
"""


from __future__ import annotations
from typing import Optional, Union
import torch


# noinspection PyMethodOverriding,PyAbstractClass,PyMissingOrEmptyDocstring
class _ScaledGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def scaled_gradient(x: torch.Tensor, scale: float) -> torch.Tensor:
    """
    :param x:
    :param scale:
    :return: just x, however, in backward pass, the gradient is scaled by the given factor
    """
    return _ScaledGradient.apply(x, scale)


# noinspection PyMethodOverriding,PyAbstractClass,PyMissingOrEmptyDocstring
class _ScaledGradientExt(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        scale: Union[float, torch.Tensor] = 1.0,
        shift: Optional[Union[float, torch.Tensor]] = None,
        scale_shift_by_sum_over_axis: Optional[int] = None,
    ):
        ctx.scale = scale
        ctx.shift = shift
        ctx.scale_shift_by_sum_over_axis = scale_shift_by_sum_over_axis
        return x

    @staticmethod
    def backward(ctx, grad):
        grad_out = grad
        if isinstance(ctx.scale, torch.Tensor) or ctx.scale != 1:
            grad_out = grad_out * ctx.scale
        if ctx.shift is not None and (isinstance(ctx.shift, torch.Tensor) or ctx.shift != 0):
            if ctx.scale_shift_by_sum_over_axis is not None:
                m = torch.sum(torch.abs(grad), dim=ctx.scale_shift_by_sum_over_axis, keepdim=True)
                grad_out = grad_out + ctx.shift * m
            else:
                grad_out = grad_out + ctx.shift
        return grad_out, None, None, None


def scaled_gradient_ext(
    x: torch.Tensor,
    *,
    scale: Union[float, torch.Tensor] = 1.0,
    shift: Optional[Union[float, torch.Tensor]] = None,
    scale_shift_by_sum_over_axis: Optional[int] = None,
):
    """
    :param x:
    :param scale: will scale gradient by this value
    :param shift: will shift gradient by this value
    :param scale_shift_by_sum_over_axis: if given, will scale and shift by the sum over the given axis
    :return: just x, but gradient in backward pass will be transformed accordingly
    """
    return _ScaledGradientExt.apply(x, scale, shift, scale_shift_by_sum_over_axis)
