"""
Common parameter initialization functions.

https://github.com/rwth-i6/returnn/wiki/Parameter-initialization
"""

from __future__ import annotations
from typing import Union, Sequence, Optional
from returnn.tensor import Tensor, Dim
import math
from .. import frontend as rf


__all__ = ["ParamInit", "ParamInitType", "Normal", "VarianceScaling", "Glorot", "He", "HeNormal", "HeUniform"]


class ParamInit:
    """API for param init"""

    def __call__(
        self,
        dims: Sequence[Dim],
        dtype: str,
        *,
        sparse_dim: Optional[Dim] = None,
        device: Optional[str] = None,
        out: Optional[Tensor] = None,
    ) -> Union[Tensor, rf.RawTensorTypes]:
        raise NotImplementedError


ParamInitType = Union[Tensor, rf.RawTensorTypes, ParamInit]


class Normal(ParamInit):
    """
    Initialization by normal distribution (truncated by default),
    independent of the dimensions (fan in/out).

    See :class:`VarianceScaling` and derivatives for variants which depend on fan in/out.
    """

    def __init__(self, stddev: float, *, truncated: bool = True, dtype: str = None):
        self.stddev = stddev
        self.truncated = truncated
        if dtype is None:
            dtype = rf.get_default_float_dtype()
        self.dtype = dtype
        if self.stddev <= 0.0:
            raise ValueError(f"Argument `stddev` must be a positive float. Received: {self.stddev}")

    def __call__(
        self,
        dims: Sequence[Dim],
        dtype: str,
        *,
        sparse_dim: Optional[Dim] = None,
        device: Optional[str] = None,
        out: Optional[Tensor] = None,
    ) -> Tensor:
        if dtype is None:
            dtype = self.dtype
        return rf.random(
            distribution="truncated_normal" if self.truncated else "normal",
            static=True,
            dims=dims,
            mean=0.0,
            stddev=self.stddev,
            dtype=dtype,
            sparse_dim=sparse_dim,
            device=device,
            out=out,
        )


class VarianceScaling(ParamInit):
    """
    Provides a generalized way for initializing weights.
    All the common initialization methods are special cases
    such as Xavier Glorot and Kaiming He.

    Code adopted from TensorFlow VarianceScaling.
    """

    scale = 1.0
    mode = "fan_in"  # fan_in, fan_out, fan_avg
    distribution = "truncated_normal"  # normal (=truncated_normal), untruncated_normal, truncated_normal, uniform
    dtype: str  # rf.get_default_float_dtype() by default

    def __init__(self, scale: float = None, mode: str = None, distribution: str = None, dtype: str = None):
        if scale is not None:
            self.scale = scale
        if mode is not None:
            self.mode = mode
        if distribution is not None:
            self.distribution = distribution
        if dtype is None:
            dtype = rf.get_default_float_dtype()
        self.dtype = dtype

        if self.scale <= 0.0:
            raise ValueError(f"Argument `scale` must be a positive float. Received: {self.scale}")
        if self.mode not in {"fan_in", "fan_out", "fan_avg"}:
            raise ValueError(
                f"Argument `mode` should be one of ('fan_in', 'fan_out', 'fan_avg'). Received: {self.mode}"
            )
        if self.distribution not in {"normal", "uniform", "truncated_normal", "untruncated_normal"}:
            raise ValueError(
                "Argument `distribution` should be one of "
                "('normal', 'uniform', 'truncated_normal', 'untruncated_normal'). "
                f"Received: {self.distribution}"
            )

    def __call__(
        self,
        dims: Sequence[Dim],
        dtype: str,
        *,
        sparse_dim: Optional[Dim] = None,
        device: Optional[str] = None,
        out: Optional[Tensor] = None,
    ) -> Tensor:
        if dtype is None:
            dtype = self.dtype
        scale = self.scale
        fan_in, fan_out = _compute_fans(dims)
        if self.mode == "fan_in":
            scale /= max(1.0, fan_in)
        elif self.mode == "fan_out":
            scale /= max(1.0, fan_out)
        else:
            scale /= max(1.0, (fan_in + fan_out) / 2.0)
        return self._random(dims=dims, dtype=dtype, scale=scale, sparse_dim=sparse_dim, device=device, out=out)

    def _random(
        self,
        dims: Sequence[Dim],
        dtype: Optional[str] = None,
        *,
        scale: float,
        sparse_dim: Optional[Dim] = None,
        device: Optional[str] = None,
        out: Optional[Tensor] = None,
    ) -> Tensor:
        if self.distribution in {"truncated_normal", "normal"}:
            # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = math.sqrt(scale) / 0.87962566103423978
            return rf.random(
                distribution="truncated_normal",
                static=True,
                dims=dims,
                mean=0.0,
                stddev=stddev,
                dtype=dtype,
                sparse_dim=sparse_dim,
                device=device,
                out=out,
            )
        elif self.distribution == "untruncated_normal":
            stddev = math.sqrt(scale)
            return rf.random(
                distribution="normal",
                static=True,
                dims=dims,
                mean=0.0,
                stddev=stddev,
                dtype=dtype,
                sparse_dim=sparse_dim,
                device=device,
                out=out,
            )
        elif self.distribution == "uniform":
            limit = math.sqrt(3.0 * scale)
            return rf.random(
                distribution="uniform",
                static=True,
                dims=dims,
                minval=-limit,
                maxval=limit,
                dtype=dtype,
                sparse_dim=sparse_dim,
                device=device,
                out=out,
            )
        else:
            raise ValueError(f"invalid distribution {self.distribution!r}")


class Glorot(VarianceScaling):
    """
    Xavier Glorot (http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).
    scale 1, fan_avg, uniform
    """

    scale = 1.0
    mode = "fan_avg"
    distribution = "uniform"


class He(VarianceScaling):
    """
    Kaiming He (https://arxiv.org/pdf/1502.01852.pdf).
    scale 2, fan_in, normal
    """

    scale = 2.0
    mode = "fan_in"
    distribution = "normal"


HeNormal = He


class HeUniform(He):
    """
    He-init (:class:`He`) but using a uniform distribution.
    scale 2, fan_in, uniform
    """

    distribution = "uniform"


def _compute_fans(dims: Sequence[Dim]):
    """Computes the number of input and output units for a weight shape.

    Args:
      dims: Integer shape tuple or TF tensor shape.

    Returns:
      A tuple of integer scalars (fan_in, fan_out).
    """
    dims = [dim.dimension for dim in dims]
    if len(dims) < 1:  # Just to avoid errors for constants.
        fan_in = fan_out = 1
    elif len(dims) == 1:
        fan_in = fan_out = dims[0]
    elif len(dims) == 2:
        fan_in = dims[0]
        fan_out = dims[1]
    else:
        # Assuming convolution kernels (1D, 2D, 3D, or more).
        # kernel shape: (out_dim, in_dim, ...)
        receptive_field_size = 1
        for dim in dims[2:]:
            receptive_field_size *= dim
        fan_in = dims[1] * receptive_field_size
        fan_out = dims[0] * receptive_field_size
    return fan_in, fan_out
