"""
Normalization functions such as batch norm
"""


from __future__ import annotations
from typing import Sequence, Union, Tuple
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


__all__ = ["moments", "LayerNorm"]


def moments(x: Tensor, axis: Union[Dim, Sequence[Dim]]) -> Tuple[Tensor, Tensor]:
    """
    :param x: input
    :param axis: the axis to be reduced, to calculate statistics over
    :return: mean, variance. it has the same shape as the input with the axis removed
    """
    mean = rf.reduce_mean(x, axis=axis)
    # stop_gradient does not change the gradient here
    variance = rf.reduce_mean(rf.squared_difference(x, rf.stop_gradient(mean)), axis=axis)
    return mean, variance


class LayerNorm(rf.Module):
    """
    `Layer normalization <https://arxiv.org/abs/1607.06450>`__.

    Note that we *just* normalize over the feature-dim axis here.
    This is consistent to the default behavior of :class:`tf.keras.layers.LayerNormalization`
    and also how it is commonly used in many models, including Transformer.

    However, there are cases where it would be common to normalize over all axes except batch-dim,
    or all axes except batch and time.
    For a more generic variant, see :func:`norm`.
    """

    def __init__(self, in_dim: Union[rf.Dim, Sequence[rf.Dim]], *, eps: float = 1e-6):
        super().__init__()
        self.in_dim = in_dim
        self.eps = eps
        self.scale = rf.Parameter([self.in_dim] if isinstance(self.in_dim, rf.Dim) else self.in_dim)
        self.scale.initial = 1.0
        self.bias = rf.Parameter(self.scale.dims)
        self.bias.initial = 0.0

    def __call__(self, x: Tensor) -> Tensor:
        mean = rf.reduce_mean(x, axis=self.in_dim)
        variance = rf.reduce_mean(rf.squared_difference(x, mean), axis=self.in_dim)
        norm_x = (x - mean) * rf.rsqrt(variance + self.eps)
        return norm_x * self.scale + self.bias
