"""
High-level backend for RETURNN layers
"""

from __future__ import annotations
from typing import Union, Sequence, Optional, Any, Tuple

from returnn.util.basic import NotSpecified
from returnn.tensor import Tensor, Dim

# noinspection PyProtectedMember
from returnn.frontend._backend import Backend

from .layer import Layer
from .. import frontend_layers as rfl
from ... import frontend as rf
from . import dims as _dims
from returnn.frontend import RawTensorTypes


# Ignore this warning until we really expect that we implemented everything.
# noinspection PyAbstractClass
class ReturnnLayersBackend(Backend[Layer]):
    """
    RETURNN layers backend (using TF), where raw_tensor represents a RETURNN layer
    """

    RawTensorType = Layer
    is_tensorflow = True

    @staticmethod
    def executing_eagerly() -> bool:
        """executing eagerly"""
        return False

    @staticmethod
    def get_dtype_name_raw(raw_tensor: Layer) -> str:
        """dtype"""
        return raw_tensor.tensor.dtype

    @staticmethod
    def as_dtype_raw(dtype_name: str) -> str:
        """dtype"""
        return dtype_name

    @staticmethod
    def get_ndim_raw(raw_tensor: Layer) -> int:
        """ndim"""
        return raw_tensor.tensor.batch_ndim

    @staticmethod
    def get_shape_raw(raw_tensor: Layer) -> Layer:
        """shape"""
        raise NotImplementedError

    @staticmethod
    def get_shape_tuple_raw(raw_tensor: Layer) -> Tuple[Union[int, Layer]]:
        """shape"""
        raise NotImplementedError

    @staticmethod
    def get_known_shape_raw(raw_tensor: Layer) -> Tuple[Optional[int]]:
        """known shape"""
        return raw_tensor.tensor.batch_shape

    @staticmethod
    def fill_raw(shape: Union[Sequence[Union[int, Layer]], Layer], value: Union[Any, Layer]) -> Layer:
        """fill raw"""
        raise Exception("fill_raw not supported in layers backend because dim tags would be unknown")

    @staticmethod
    def compare_raw(a: Layer, kind: str, b: Layer) -> Layer:
        """compare"""
        raise NotImplementedError  # TODO

    @staticmethod
    def combine_raw(a: Layer, kind: str, b: Layer) -> Layer:
        """combine"""
        raise NotImplementedError  # TODO

    @staticmethod
    def reshape_raw(raw_tensor: Layer, shape: Union[Sequence[Union[int, Layer]], Layer]) -> Layer:
        """reshape_raw"""
        raise Exception("reshape_raw not supported in layers backend because dim tags would be unknown")

    @staticmethod
    def transpose_raw(raw_tensor: Layer, perm: Sequence[int]) -> Layer:
        """transpose_raw is a no-op in this backend"""
        return raw_tensor

    @staticmethod
    def activation_raw(raw_tensor: Layer, func: str) -> Layer:
        """activation"""
        return rfl.make_layer(
            {"class": "activation", "activation": func, "from": raw_tensor.tensor}, name=func
        ).raw_tensor

    @staticmethod
    def softmax(tensor: Tensor, *, axis: Dim) -> Tensor:
        """softmax"""
        return rfl.make_layer({"class": "softmax_over_spatial", "axis": axis, "from": tensor}, name="softmax")

    @staticmethod
    def log_softmax(tensor: Tensor, *, axis: Dim) -> Tensor:
        """log softmax"""
        return rfl.make_layer(
            {"class": "softmax_over_spatial", "axis": axis, "from": tensor, "log_space": True}, name="log_softmax"
        )

    @staticmethod
    def sequence_mask_raw(lengths: Layer, *, batch_major: bool = True) -> Layer:
        """sequence mask"""
        raise NotImplementedError  # TODO

    @staticmethod
    def create_parameter_raw(tensor: rf.Parameter) -> Layer:
        """create parameter"""
        return rfl.make_layer(
            {"class": "variable", "shape": tensor.dims, "dtype": tensor.dtype}, name=tensor.name, existing_tensor=tensor
        ).raw_tensor

    @staticmethod
    def set_parameter_initial_value(param: rf.Parameter, value: Union[None, Tensor, rf.RawTensorTypes]) -> None:
        """set parameter initial value"""
        raise NotImplementedError  # TODO

    @staticmethod
    def convert_to_tensor(
        value: Union[Tensor, Layer, RawTensorTypes],
        *,
        dims: Sequence[Dim] = (),
        dtype: Optional[str] = None,
        sparse_dim: Optional[Dim] = None,
    ) -> Tensor[Layer]:
        """convert to tensor"""
        if isinstance(value, Tensor):
            assert value.dims == tuple(dims)
            if dtype:
                assert value.dtype == dtype
            return value
        kwargs = {}
        dim_deps = _dims.get_dim_deps(dims)
        if dim_deps:
            kwargs["shape_deps"] = dim_deps
        return rfl.make_layer(
            {"class": "constant", "value": value, "shape": dims, "dtype": dtype, **kwargs}, name="constant"
        )

    @staticmethod
    def matmul(a: Tensor, b: Tensor, *, reduce: Union[Dim, Sequence[Dim]]) -> Tensor:
        """matmul"""
        return rfl.make_layer({"class": "dot", "from": [a, b], "reduce": reduce}, name="matmul")

    @staticmethod
    def range_over_dim(dim: Dim) -> Tensor:
        """range over dim"""
        return rfl.make_layer(
            {"class": "range_in_axis", "from": _dims.get_dim_deps(dim), "axis": dim}, name="range_over_dim"
        )

    @staticmethod
    def reduce(
        source: Tensor, *, mode: str, axis: Union[Dim, Sequence[Dim]], use_time_mask: bool = NotSpecified
    ) -> Tensor:
        """Reduce"""
        assert mode in Backend._AllowedReduceModes
        kwargs = {}
        if use_time_mask is not NotSpecified:
            kwargs["use_time_mask"] = use_time_mask
        return rfl.make_layer(
            {"class": "reduce", "from": source, "mode": mode, "axis": axis, **kwargs}, name=f"reduce_{mode}"
        )
