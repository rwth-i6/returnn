"""
Allow to use Numpy arrays.
This backend will probably never be feature-complete.
It just has the bare minimum such that the user can assign Numpy arrays to Tensor.raw_tensor.
"""

from __future__ import annotations
from typing import Union, Sequence, Tuple
import numpy
from returnn.util.basic import NotSpecified
from returnn.tensor import Tensor, Dim
from ._backend import Backend


# We do not expect that we will ever implement all the methods of the Backend interface.
# noinspection PyAbstractClass
class NumpyBackend(Backend[numpy.ndarray]):
    """Numpy backend"""

    RawTensorType = numpy.ndarray

    @staticmethod
    def executing_eagerly() -> bool:
        """executing eagerly"""
        return True

    @staticmethod
    def get_dtype_name_raw(raw_tensor: numpy.ndarray) -> str:
        """
        :return: dtype of raw tensor, as string
        """
        return raw_tensor.dtype.name

    @staticmethod
    def as_dtype_raw(dtype_name: str) -> numpy.dtype:
        """
        :param dtype_name: e.g. "float32"
        :return: dtype object
        """
        return numpy.dtype(dtype_name)

    @staticmethod
    def get_ndim_raw(raw_tensor: numpy.ndarray) -> int:
        """
        :return: ndim of raw tensor. assumes it is known
        """
        return raw_tensor.ndim

    @staticmethod
    def get_shape_raw(raw_tensor: numpy.ndarray) -> Tuple[int]:
        """
        :return: shape of raw tensor
        """
        return raw_tensor.shape

    @staticmethod
    def get_shape_tuple_raw(raw_tensor: numpy.ndarray) -> Tuple[int]:
        """
        :return: shape of raw tensor. assumes that ndim is known.
            In eager frameworks, all dims are int.
        """
        return raw_tensor.shape

    @staticmethod
    def get_known_shape_raw(raw_tensor: numpy.ndarray) -> Tuple[int]:
        """
        :return: shape of raw tensor, int for static known, None otherwise. assumes that ndim is known.
            This will not create any ops.
            In eager frameworks, all dims are known.
        """
        return raw_tensor.shape

    @staticmethod
    def reduce(
        source: Tensor[numpy.ndarray],
        *,
        mode: str,
        axis: Union[Dim, Sequence[Dim]],
        use_time_mask: bool = NotSpecified,
    ) -> Tensor[numpy.ndarray]:
        """reduce"""
        assert mode in Backend._AllowedReduceModes
        if isinstance(axis, Dim):
            assert not axis.need_masking()  # not implemented
        else:
            assert all(not dim.need_masking() for dim in axis)  # not implemented
        func = getattr(numpy, mode)
        raw_dims = (
            [source.get_axis_from_description(axis)]
            if isinstance(axis, Dim)
            else [source.get_axis_from_description(dim) for dim in axis]
        )
        res_dims = [dim for i, dim in enumerate(source.dims) if i not in raw_dims]
        if not res_dims:
            # All are reduced. Need numpy.array() to get a tensor again.
            raw_result = numpy.array(func(source.raw_tensor))
        else:
            raw_result = func(source.raw_tensor, axis=raw_dims)
        res = Tensor(
            name=f"reduce_{mode}",
            raw_tensor=raw_result,
            dims=res_dims,
            dtype=source.dtype,
            sparse_dim=source.sparse_dim,
        )
        return res
