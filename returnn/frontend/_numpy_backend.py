"""
Allow to use Numpy arrays.
This backend will probably never be feature-complete.
It just has the bare minimum such that the user can assign Numpy arrays to Tensor.raw_tensor.
"""

from __future__ import annotations
from typing import Tuple
import numpy
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
