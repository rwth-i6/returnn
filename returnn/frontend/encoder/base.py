"""
Base interface for any kind of encoder.

This is basically any generic function x -> y.

Note that in practice, when designing some model,
this interface is even not needed,
because you only care about the final encoded vectors,
and not how you got there.
Automatic differentiation will automatically
also train the encoder.
So, for most purpose, e.g. for a decoder (see :mod:`..decoder.base`),
you only care about some encoded vector of type :class:`Tensor`.
"""

from __future__ import annotations
from typing import Tuple, Union
from abc import ABC
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


class IEncoder(rf.Module, ABC):
    """
    Generic encoder interface

    The encoder is a function x -> y.
    The input can potentially be sparse or dense.
    The output is dense with feature dim `out_dim`.
    """

    out_dim: Dim

    def __call__(self, source: Tensor) -> Tensor:
        """
        Encode the input
        """
        raise NotImplementedError


class ISeqFramewiseEncoder(rf.Module, ABC):
    """
    This specializes IEncoder that it operates on a sequence.
    The output sequence length here is the same as the input.
    """

    out_dim: Dim

    def __call__(self, source: Tensor, *, spatial_dim: Dim) -> Tensor:
        raise NotImplementedError


class ISeqDownsamplingEncoder(rf.Module, ABC):
    """
    This is more specific than IEncoder in that it operates on a sequence.
    The output sequence length here is shorter than the input.

    This is a common scenario for speech recognition
    where the input might be on 10ms/frame
    and the output might cover 30ms/frame or 60ms/frame or so.
    """

    out_dim: Dim
    # In most cases (pooling, conv), the output sequence length will bei ceildiv(input_seq_len, factor)
    # and factor is an integer.
    # However, this is not a hard condition.
    # The downsampling factor only describes the linear factor in the limit.
    downsample_factor: Union[int, float]

    def __call__(self, source: Tensor, *, in_spatial_dim: Dim) -> Tuple[Tensor, Dim]:
        raise NotImplementedError
