"""
Frontend for exposing PyTorch-specific functionality.
"""

from __future__ import annotations
from typing import Union, Sequence

import torch

from returnn.frontend_api import Frontend, RawTensorTypes
from ._internal_frontend import TorchInternalFrontend
from returnn.tensor import Tensor, Dim

from returnn.util.basic import prod

_TT = Tensor[torch.Tensor]


# Ignore this warning until we really expect that we implemented everything.
# noinspection PyAbstractClass
class TorchFrontend(Frontend[torch.Tensor]):
    """
    PyTorch frontend
    """

    RawTensorType = torch.Tensor
    _internal_frontend = TorchInternalFrontend

    @staticmethod
    def convert_to_tensor(value: Union[Tensor, torch.Tensor, RawTensorTypes]) -> Tensor[torch.Tensor]:
        """
        :param value:
        :return: tensor
        """
        if isinstance(value, Tensor):
            return value
        value = torch.tensor(value)
        assert isinstance(value, torch.Tensor)
        assert value.shape.as_list() == [], f"scalar expected, got {value}"
        return Tensor("const", raw_tensor=value, dims=[], dtype=value.dtype.base_dtype.name)

    @staticmethod
    def dot(a: _TT, b: _TT, *, reduce: Union[Dim, Sequence[Dim]]) -> _TT:
        """
        dot-product of a and b, see base class doc string
        """
        if isinstance(reduce, Dim):
            reduce = [reduce]

        a_dims = a.dims
        b_dims = b.dims

        assert all(
            dim in a_dims for dim in reduce
        ), f"'a' does not have the specified reduce dim(s) {reduce} (a dims: {a_dims})"
        assert all(
            dim in b_dims for dim in reduce
        ), f"'b' does not have the specified reduce dim(s) {reduce} (b dims: {b_dims})"

        reduce_dims_sorted_by_a = [dim for dim in a_dims if dim in reduce]

        a_reduce_axes = [a_dims.index(reduce_dim) for reduce_dim in reduce_dims_sorted_by_a]
        b_reduce_axes = [b_dims.index(reduce_dim) for reduce_dim in reduce_dims_sorted_by_a]

        a_unique_axes = [i for (i, dim) in enumerate(a_dims) if dim not in b_dims]
        b_unique_axes = [i for (i, dim) in enumerate(b_dims) if dim not in a_dims]

        common_dims = [dim for dim in a_dims if dim in b_dims and dim not in reduce]

        a_common_axes = [a_dims.index(common_dim) for common_dim in common_dims]
        b_common_axes = [b_dims.index(common_dim) for common_dim in common_dims]

        a_raw = a.raw_tensor
        b_raw = b.raw_tensor

        a_shape = a_raw.shape
        b_shape = b_raw.shape

        common_axes_shape = tuple(a_shape[i] for i in a_common_axes)
        b_common_axes_shape = tuple(b_shape[i] for i in b_common_axes)
        assert common_axes_shape == b_common_axes_shape, "Tensor shape for common Dims of a and b does not match."

        common_axes_total_dimension = prod(common_axes_shape)

        a_unique_axes_shape = tuple(a_shape[i] for i in a_unique_axes)
        b_unique_axes_shape = tuple(b_shape[i] for i in b_unique_axes)

        a_unique_axes_total_dimension = prod(a_unique_axes_shape)
        b_unique_axes_total_dimension = prod(b_unique_axes_shape)

        reduce_axes_shape = tuple(a_shape[i] for i in a_reduce_axes)
        b_reduce_axes_shape = tuple(b_shape[i] for i in b_reduce_axes)
        assert reduce_axes_shape == b_reduce_axes_shape, "Tensor shape for reduce Dims does not match between a and b."

        reduce_axes_total_dimension = prod(reduce_axes_shape)

        a_raw = torch.permute(a_raw, a_common_axes + a_unique_axes + a_reduce_axes)
        b_raw = torch.permute(b_raw, b_common_axes + b_reduce_axes + b_unique_axes)

        if common_axes_total_dimension == 1:  # standard matrix multiplication
            a_raw = torch.reshape(a_raw, (a_unique_axes_total_dimension, reduce_axes_total_dimension))
            b_raw = torch.reshape(b_raw, (reduce_axes_total_dimension, b_unique_axes_total_dimension))

            raw_result = torch.mm(a_raw, b_raw)

        else:  # batched matrix multiplication
            a_raw = torch.reshape(
                a_raw, (common_axes_total_dimension, a_unique_axes_total_dimension, reduce_axes_total_dimension)
            )
            b_raw = torch.reshape(
                b_raw, (common_axes_total_dimension, reduce_axes_total_dimension, b_unique_axes_total_dimension)
            )

            raw_result = torch.bmm(a_raw, b_raw)

        raw_result = torch.reshape(raw_result, common_axes_shape + a_unique_axes_shape + b_unique_axes_shape)

        a_unique_dims = [a_dims[i] for i in a_unique_axes]
        b_unique_dims = [b_dims[i] for i in b_unique_axes]
        result_dims = common_dims + a_unique_dims + b_unique_dims

        result_tensor = Tensor(name="dot", dims=result_dims, raw_tensor=raw_result)

        return result_tensor
