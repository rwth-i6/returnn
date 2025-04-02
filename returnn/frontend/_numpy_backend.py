"""
Allow to use Numpy arrays.
This backend will probably never be feature-complete.
It just has the bare minimum such that the user can assign Numpy arrays to Tensor.raw_tensor.
"""

from __future__ import annotations
from typing import Optional, Union, Sequence, Tuple
import numpy
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from ._backend import Backend
from returnn.frontend import RawTensorTypes


# We do not expect that we will ever implement all the methods of the Backend interface.
# noinspection PyAbstractClass
class NumpyBackend(Backend[numpy.ndarray]):
    """Numpy backend"""

    name = "numpy"
    RawTensorType = numpy.ndarray

    @staticmethod
    def executing_eagerly() -> bool:
        """executing eagerly"""
        return True

    @staticmethod
    def get_dtype_name_raw(raw_tensor: numpy.ndarray) -> str:
        """
        :return: dtype of raw tensor, as string. e.g. "int64" etc.
        """
        dtype_name = raw_tensor.dtype.name
        # See returnn.datasets.util.strings.str_to_numpy_array.
        if dtype_name.startswith("str"):
            return "string"  # ignore the bit-length, it supports variable length
        return dtype_name

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
    def get_shape_raw(raw_tensor: numpy.ndarray) -> Tuple[int, ...]:
        """
        :return: shape of raw tensor
        """
        return raw_tensor.shape

    @staticmethod
    def get_shape_tuple_raw(raw_tensor: numpy.ndarray) -> Tuple[int, ...]:
        """
        :return: shape of raw tensor. assumes that ndim is known.
            In eager frameworks, all dims are int.
        """
        return raw_tensor.shape

    @staticmethod
    def get_known_shape_raw(raw_tensor: numpy.ndarray) -> Tuple[int, ...]:
        """
        :return: shape of raw tensor, int for static known, None otherwise. assumes that ndim is known.
            This will not create any ops.
            In eager frameworks, all dims are known.
        """
        return raw_tensor.shape

    @staticmethod
    def convert_to_tensor(
        value: Union[Tensor, numpy.ndarray, RawTensorTypes],
        *,
        dims: Sequence[Dim],
        dtype: str,
        sparse_dim: Optional[Dim] = None,
        feature_dim: Optional[Dim] = None,
        device: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Tensor[numpy.ndarray]:
        """convert to tensor"""
        if isinstance(value, Tensor):
            return value
        if isinstance(value, numpy.ndarray):
            name = name or "raw_tensor"
        else:
            name = name or "const"
            value = numpy.array(value, dtype=NumpyBackend.as_dtype_raw(dtype))
        assert isinstance(value, numpy.ndarray)
        return Tensor(name, dims=dims, dtype=dtype, sparse_dim=sparse_dim, feature_dim=feature_dim, raw_tensor=value)

    @staticmethod
    def expand_dims_raw(raw_tensor: numpy.ndarray, axis: int) -> numpy.ndarray:
        """
        :param raw_tensor:
        :param axis: e.g. 1
        :return: raw tensor with new axis
        """
        return numpy.expand_dims(raw_tensor, axis)

    @staticmethod
    def transpose_raw(raw_tensor: numpy.ndarray, perm: Sequence[int]) -> numpy.ndarray:
        """
        :param raw_tensor:
        :param perm: e.g. [0, 2, 1]
        :return: permuted (transposed) raw tensor
        """
        if all(p == i for i, p in enumerate(perm)):
            return raw_tensor
        return raw_tensor.transpose(tuple(perm))

    @staticmethod
    def reshape_raw(
        raw_tensor: numpy.ndarray, shape: Union[Sequence[Union[int, numpy.ndarray]], numpy.ndarray]
    ) -> numpy.ndarray:
        """reshape raw"""
        return numpy.reshape(raw_tensor, shape)

    @staticmethod
    def compare_raw(a: numpy.ndarray, kind: str, b: numpy.ndarray) -> numpy.ndarray:
        """
        :param a:
        :param kind: "equal", "less", "less_equal", "greater", "greater_equal", "not_equal"
        :param b:
        :return: a `kind` b
        """
        assert a.ndim == b.ndim or a.ndim == 0 or b.ndim == 0
        op = getattr(numpy, kind)  # e.g. numpy.equal
        return op(a, b)

    _CombineKindMap = {"mul": numpy.multiply}

    @staticmethod
    def combine_raw(a: numpy.ndarray, kind: str, b: numpy.ndarray) -> numpy.ndarray:
        """
        :param a:
        :param kind: "add", "sub", "mul", "truediv", "floordiv", "mod", "pow",
            "maximum", "minimum", "logical_and", "logical_or", "squared_difference"
        :param b:
        :return: a `kind` b
        """
        assert a.ndim == b.ndim or a.ndim == 0 or b.ndim == 0
        op = getattr(numpy, kind, None)  # e.g. numpy.add
        if not op:
            op = NumpyBackend._CombineKindMap.get(kind)
            if not op:
                raise ValueError(f"RF NumpyBackend: combine kind {kind!r} not supported")
        res = op(a, b)
        if not isinstance(res, numpy.ndarray):
            res = numpy.array(res)
        return res

    @staticmethod
    def where(
        cond: Tensor,
        true_: Union[Tensor, rf.RawTensorTypes],
        false_: Union[Tensor, rf.RawTensorTypes],
        *,
        allow_broadcast_all_sources: bool = False,
    ) -> Tensor:
        """where"""
        if isinstance(true_, Tensor):
            dtype = true_.dtype
        elif isinstance(false_, Tensor):
            dtype = false_.dtype
        else:
            dtype = None
        true_ = rf.convert_to_tensor(true_, _backend=NumpyBackend, dtype=dtype)
        false_ = rf.convert_to_tensor(false_, _backend=NumpyBackend, dtype=dtype)
        out = Tensor.get_common_data(
            [true_, false_, cond], allow_broadcast_all_sources=allow_broadcast_all_sources, name="where"
        )
        out.dtype = true_.dtype
        out.sparse_dim = true_.sparse_dim or false_.sparse_dim
        out.feature_dim = true_.feature_dim or false_.feature_dim
        cond_bc_raw = cond.copy_compatible_to_dims_raw(out.dims)
        true_bc_raw = true_.copy_compatible_to_dims_raw(out.dims)
        false_bc_raw = false_.copy_compatible_to_dims_raw(out.dims)
        out.raw_tensor = numpy.where(cond_bc_raw, true_bc_raw, false_bc_raw)
        return out

    @staticmethod
    def range_over_dim(dim: Dim, *, dtype: Optional[str] = None, device: Optional[str] = None) -> Tensor[numpy.ndarray]:
        """
        :param dim:
        :param dtype:
        :param device:
        :return: tensor with shape [dim]
        """
        if not dtype and dim.dyn_size_ext is not None:
            dtype = dim.dyn_size_ext.dtype
        if not dtype:
            dtype = rf.get_default_array_index_dtype()
        out = Tensor(
            "range",
            dims=[dim],
            sparse_dim=dim if dtype.startswith("int") or dtype.startswith("uint") else None,
            dtype=dtype,
        )
        out.raw_tensor = numpy.arange(dim.get_dim_value(), dtype=NumpyBackend.as_dtype_raw(out.dtype))
        return out

    @staticmethod
    def reduce(
        source: Tensor[numpy.ndarray],
        *,
        mode: str,
        axis: Union[Dim, Sequence[Dim]],
        use_mask: bool = True,
    ) -> Tensor[numpy.ndarray]:
        """reduce"""
        assert mode in Backend._AllowedReduceModes
        if use_mask:
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

    @staticmethod
    def activation_raw(raw_tensor: numpy.ndarray, func: str) -> numpy.ndarray:
        """
        :param raw_tensor:
        :param func: "tanh", "sigmoid", "relu", ...
        :return: raw tensor with elementwise activation applied
        """
        if func == "relu":
            return numpy.array(numpy.maximum(raw_tensor, 0))
        raise NotImplementedError("NumpyBackend: activation %r not implemented" % func)
