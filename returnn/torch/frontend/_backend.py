"""
Backend for exposing PyTorch-specific functionality.
"""

from __future__ import annotations
from typing import Optional, Union, Sequence, Tuple
import torch
import numpy

from returnn.tensor import Tensor, Dim
from returnn.util.basic import prod, NotSpecified

# noinspection PyProtectedMember
from returnn.frontend._backend import Backend
from returnn.frontend import RawTensorTypes
import returnn.frontend as rf


_TT = Tensor[torch.Tensor]


# Ignore this warning until we really expect that we implemented everything.
# noinspection PyAbstractClass
class TorchBackend(Backend[torch.Tensor]):
    """
    PyTorch backend
    """

    RawTensorType = torch.Tensor

    @staticmethod
    def executing_eagerly() -> bool:
        """
        :return: whether we are executing eagerly
        """
        return True

    @staticmethod
    def get_dtype_name_raw(raw_tensor: torch.Tensor) -> str:
        """
        :return: dtype of raw tensor, as string
        """
        return str(raw_tensor.dtype).replace("torch.", "")

    @staticmethod
    def as_dtype_raw(dtype_name: str) -> torch.dtype:
        """
        :param dtype_name: e.g. "float32"
        :return: dtype object
        """
        dtype = getattr(torch, dtype_name)
        assert isinstance(dtype, torch.dtype)
        return dtype

    @staticmethod
    def get_ndim_raw(raw_tensor: torch.Tensor) -> int:
        """
        :return: ndim of raw tensor
        """
        return raw_tensor.dim()

    @staticmethod
    def get_known_shape_raw(raw_tensor: torch.Tensor) -> Tuple[Optional[int]]:
        """
        :return: shape of raw tensor; here for PyTorch the full shape is always known
        """
        return tuple(raw_tensor.size())

    @staticmethod
    def expand_dims_raw(raw_tensor: torch.Tensor, axis: int) -> torch.Tensor:
        """
        :param raw_tensor:
        :param axis: e.g. 1
        :return: raw tensor with new axis
        """
        return raw_tensor.unsqueeze(axis)

    @staticmethod
    def activation_raw(raw_tensor: torch.Tensor, func: str) -> torch.Tensor:
        """
        :param raw_tensor:
        :param func: e.g. "tanh"
        :return: raw tensor after activation
        """
        assert func in Backend._AllowedActivationFuncs
        if hasattr(torch, func):
            f = getattr(torch, func)
        elif hasattr(torch.nn.functional, func):
            f = getattr(torch.nn.functional, func)
        else:
            raise ValueError(f"unknown activation function {func!r}")
        return f(raw_tensor)

    @staticmethod
    def softmax(tensor: Tensor, *, axis: Dim) -> Tensor:
        """
        :param tensor:
        :param axis:
        :return: softmax over axis
        """
        out = tensor.copy_template("softmax")
        out.raw_tensor = torch.softmax(tensor.raw_tensor, dim=tensor.dims.index(axis))
        return out

    @staticmethod
    def log_softmax(tensor: Tensor, *, axis: Dim) -> Tensor:
        """
        :param tensor:
        :param axis:
        :return: log_softmax over axis
        """
        out = tensor.copy_template("log_softmax")
        out.raw_tensor = torch.log_softmax(tensor.raw_tensor, dim=tensor.dims.index(axis))
        return out

    @staticmethod
    def create_parameter_raw(tensor: Tensor) -> torch.nn.Parameter:
        """
        :return: parameter
        """
        assert all(d.is_static() for d in tensor.dims)
        data = torch.zeros(*(d.dimension for d in tensor.dims), dtype=TorchBackend.as_dtype_raw(tensor.dtype))
        return torch.nn.Parameter(data)

    @staticmethod
    def set_parameter_initial_value(param: rf.Parameter, value: Union[None, Tensor, rf.RawTensorTypes]) -> None:
        """
        :param param: parameter
        :param value: initial value
        """
        if value is None:
            value = 0
        raw_param = param.raw_tensor
        assert isinstance(raw_param, torch.nn.Parameter)
        if isinstance(value, Tensor):
            with torch.no_grad():
                raw_param[:] = value.raw_tensor
        else:
            with torch.no_grad():
                raw_param[:] = value

    @staticmethod
    def compare_raw(a: torch.Tensor, kind: str, b: torch.Tensor) -> torch.Tensor:
        """
        :param a:
        :param kind: "equal"|"==", "less"|"<", "less_equal"|"<=", "greater"|">", "greater_equal"|">=",
            "not_equal"|"!="|"<>"
        :param b:
        :return: a `kind` b
        """
        assert a.dim() == b.dim()
        kind = {
            "==": "eq",  # eq is different to equal; eq returns a torch Tensor
            "<=": "less_equal",
            "<": "less",
            ">=": "greater_equal",
            ">": "greater",
            "!=": "not_equal",
            "<>": "not_equal",
        }.get(kind, kind)
        op = getattr(torch, kind)  # e.g. torch.equal
        return op(a, b)

    @staticmethod
    def combine_raw(a: torch.Tensor, kind: str, b: torch.Tensor) -> torch.Tensor:
        """
        :param a:
        :param kind: "add"|"+", "sub"|"-", "mul"|"*", "truediv"|"/", "floordiv"|"//", "mod"|"%", "pow"|"**",
            "max"|"maximum", "min"|"minimum", "logical_and", "logical_or", "squared_difference"
        :param b:
        :return: a `kind` b
        """
        assert a.dim() == b.dim()
        kind = {
            "+": "add",
            "-": "sub",
            "*": "mul",
            "/": "true_divide",
            "truediv": "true_divide",
            "//": "floor_divide",
            "floordiv": "floor_divide",
            "%": "remainder",  # Python-like modulo, not C-like (torch.fmod)
            "mod": "remainder",
            "**": "pow",
            "max": "maximum",
            "min": "minimum",
        }.get(kind, kind)
        op = getattr(torch, kind)  # e.g. torch.add
        return op(a, b)

    @staticmethod
    def convert_to_tensor(
        value: Union[Tensor, torch.Tensor, RawTensorTypes],
        *,
        dims: Sequence[Dim] = (),
        dtype: Optional[str] = None,
        sparse_dim: Optional[Dim] = None,
    ) -> Tensor[torch.Tensor]:
        """
        :param value:
        :param dims:
        :param dtype:
        :param sparse_dim:
        :return: tensor
        """
        if isinstance(value, Tensor):
            return value
        value = torch.tensor(value, dtype=TorchBackend.as_dtype_raw(dtype) if dtype else None)
        assert isinstance(value, torch.Tensor)
        dtype = dtype or TorchBackend.get_dtype_name_raw(value)
        return Tensor("const", raw_tensor=value, dims=dims, dtype=dtype, sparse_dim=sparse_dim)

    @staticmethod
    def full(
        dims: Sequence[Dim], fill_value: RawTensorTypes, *, dtype: str, sparse_dim: Optional[Dim] = None
    ) -> Tensor:
        """
        :param dims:
        :param fill_value:
        :param dtype:
        :param sparse_dim:
        :return: tensor
        """
        shape = [dim.get_dim_value() for dim in dims]
        raw_tensor = torch.full(shape, fill_value, dtype=TorchBackend.as_dtype_raw(dtype))
        return Tensor("full", dims=dims, sparse_dim=sparse_dim, dtype=dtype, raw_tensor=raw_tensor)

    @staticmethod
    def matmul(a: _TT, b: _TT, *, reduce: Union[Dim, Sequence[Dim]]) -> _TT:
        """
        batched matmul of a and b, see base class doc string
        """
        if isinstance(reduce, Dim):
            reduce = [reduce]

        if any(dim.dyn_size_ext for dim in reduce):
            raise NotImplementedError("masking in matmul reduce not yet implemented")
        assert a.dtype == b.dtype, f"matmul: dtypes do not match: {a} vs {b}"

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

        result_tensor = Tensor(name="dot", dims=result_dims, raw_tensor=raw_result, dtype=a.dtype)

        return result_tensor

    @staticmethod
    def reduce(
        source: Tensor[torch.Tensor],
        *,
        mode: str,
        axis: Union[Dim, Sequence[Dim]],
        use_time_mask: bool = NotSpecified,
    ) -> Tensor[torch.Tensor]:
        """reduce"""
        assert mode in Backend._AllowedReduceModes
        if isinstance(axis, Dim):
            assert not axis.need_masking()  # not implemented
        else:
            assert all(not dim.need_masking() for dim in axis)  # not implemented
        func = getattr(torch, mode)
        raw_dims = [source.dims.index(axis)] if isinstance(axis, Dim) else [source.dims.index(dim) for dim in axis]
        res_dims = [dim for i, dim in enumerate(source.dims) if i not in raw_dims]
        if not res_dims:
            raw_result = func(source.raw_tensor)
        elif len(raw_dims) == 1:
            raw_result = func(source.raw_tensor, dim=raw_dims[0])
            if mode in ["max", "min"]:
                # result is a tuple (values, indices). https://pytorch.org/docs/stable/generated/torch.max.html
                raw_result, _ = raw_result
        else:
            assert mode == "sum"  # not implemented otherwise for multiple axes
            raw_result = func(source.raw_tensor, dim=raw_dims)
        res = Tensor(
            name=f"reduce_{mode}",
            raw_tensor=raw_result,
            dims=res_dims,
            dtype=source.dtype,
            sparse_dim=source.sparse_dim,
        )
        return res

    @staticmethod
    def random(
        *,
        dims: Sequence[Dim],
        dtype: str,
        sparse_dim: Optional[Dim] = None,
        distribution: str,
        mean: Optional[Union[int, float, Tensor]] = None,
        stddev: Optional[Union[int, float, Tensor]] = None,
        bound: Optional[Union[int, float, Tensor]] = None,
        minval: Optional[Union[int, float, Tensor]] = None,
        maxval: Optional[Union[int, float, Tensor]] = None,
        seed: Optional[Union[int, Sequence[int], numpy.ndarray]] = None,
        algorithm: Optional[str] = None,
        explicit_state: Optional[Tensor] = None,
        auto_update_state: Optional[bool] = None,
        static: Optional[bool] = None,
    ) -> Tensor:
        """
        random. See `rf.random` for details.
        """
        shape = [d.get_dim_value() for d in dims]
        dtype_ = TorchBackend.as_dtype_raw(dtype)
        out = Tensor(name=f"random_{distribution}", dims=dims, dtype=dtype, sparse_dim=sparse_dim)
        assert explicit_state is None  # not implemented otherwise
        assert static and seed is None  # not implemented otherwise
        assert auto_update_state is None  # not implemented otherwise
        if distribution == "uniform":
            assert mean is None and stddev is None  # not implemented otherwise
            if dtype_.is_floating_point:
                out.raw_tensor = torch.rand(*shape, dtype=dtype_)
                if minval is None:
                    minval = 0
                if maxval is None:
                    maxval = 1
                if isinstance(minval, Tensor) or isinstance(maxval, Tensor) or minval != 0 or maxval != 1:
                    out = out * (maxval - minval) + minval
            else:
                if minval is None:
                    minval = 0
                assert maxval is not None, "maxval must be specified for integer random uniform"
                out.raw_tensor = torch.randint(minval, maxval, shape, dtype=dtype_)
        elif distribution == "normal":
            assert minval is None and maxval is None
            out.raw_tensor = torch.randn(*shape, dtype=dtype_)
            if mean is None:
                mean = 0
            if stddev is None:
                stddev = 1
            if isinstance(stddev, Tensor) or stddev != 1:
                out = out * stddev
            if isinstance(mean, Tensor) or mean != 0:
                out = out + mean
        else:
            raise NotImplementedError(f"random distribution {distribution} not implemented")
        return out
