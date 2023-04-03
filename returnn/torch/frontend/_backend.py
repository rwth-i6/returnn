"""
Backend for exposing PyTorch-specific functionality.
"""

from __future__ import annotations
from typing import Optional, Union, Sequence, Tuple, Dict
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
    def set_random_seed(seed: int):
        """
        :param seed:
        """
        torch.random.manual_seed(seed)

    @staticmethod
    def get_random_state() -> Dict[str, bytes]:
        """
        :return: random state
        """
        res = {
            "cpu": torch.random.get_rng_state().detach().cpu().numpy().tobytes(),
        }
        cuda_states = [state.detach().cpu().numpy().tobytes() for state in torch.cuda.get_rng_state_all()]
        if len(cuda_states) == 1:
            res["cuda"] = cuda_states[0]
        elif len(cuda_states) > 1:
            for i, state in enumerate(cuda_states):
                res[f"cuda{i}"] = state
        return res

    @staticmethod
    def set_random_state(state: Dict[str, bytes]):
        """
        :param state: as returned by :func:`get_random_state`.
            This might not always be successful (e.g. different hardware, different backend version),
            so the calling code should always have called set_random_seed before to have the random generators
            in a reasonable fallback state.
        """
        if "cpu" in state:
            torch.random.set_rng_state(torch.from_numpy(numpy.frombuffer(state["cpu"], dtype="uint8")))
        if "cuda" in state:
            torch.cuda.set_rng_state_all(torch.from_numpy(numpy.frombuffer(state["cuda"], dtype="uint8")))
        for k, v in state.items():
            if k.startswith("cuda"):
                i = int(k[4:])
                torch.cuda.set_rng_state(torch.from_numpy(numpy.frombuffer(v, dtype="uint8")), i)

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
    def create_parameter_raw(tensor: rf.Parameter) -> torch.nn.Parameter:
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
        if isinstance(value, torch.Tensor):
            name = "raw_tensor"
        else:
            name = "const"
            value = torch.tensor(value, dtype=TorchBackend.as_dtype_raw(dtype) if dtype else None)
        assert isinstance(value, torch.Tensor)
        dtype = dtype or TorchBackend.get_dtype_name_raw(value)
        return Tensor(name, dims=dims, dtype=dtype, sparse_dim=sparse_dim, raw_tensor=value)

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

        if len(reduce) > 1:
            reduce = list(reduce)
            reduce.sort(key=lambda dim: a_dims.index(dim))

        # matmul might get square matrices as arguments, where a dim could occur multiple times.
        # This complicates the logic here, and we properly have to handle match_priority.
        a_reduce_axes = [a.get_axis_from_description(reduce_dim) for reduce_dim in reduce]
        b_reduce_axes = [b.get_axis_from_description(reduce_dim) for reduce_dim in reduce]

        # We assume that dim tags in remaining dims are unique.
        common_dims = [dim for i, dim in enumerate(a_dims) if dim in b_dims and i not in a_reduce_axes]
        a_common_axes = [a_dims.index(common_dim) for common_dim in common_dims]
        b_common_axes = [b_dims.index(common_dim) for common_dim in common_dims]

        a_unique_axes = [i for i in range(len(a_dims)) if i not in a_reduce_axes and i not in a_common_axes]
        b_unique_axes = [i for i in range(len(b_dims)) if i not in b_reduce_axes and i not in b_common_axes]

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
        out: Optional[Tensor[torch.Tensor]] = None,
    ) -> Tensor:
        """
        random. See `rf.random` for details.
        """
        shape = [d.get_dim_value() for d in dims]
        dtype_ = TorchBackend.as_dtype_raw(dtype)
        if out is None:
            out = Tensor(name=f"random_{distribution}", dims=dims, dtype=dtype, sparse_dim=sparse_dim)
            out.raw_tensor = torch.empty(shape, dtype=dtype_)
        assert explicit_state is None  # not implemented otherwise
        generator = None  # using the global default from PT
        assert isinstance(static, bool)
        if static:
            assert seed is not None
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            assert seed is None
        assert auto_update_state is None  # not implemented otherwise
        if distribution == "uniform":
            assert mean is None and stddev is None  # not implemented otherwise
            if dtype_.is_floating_point:
                if minval is None:
                    minval = 0
                if maxval is None:
                    maxval = 1
                if isinstance(minval, Tensor):
                    assert minval.dims == (), f"only scalar minval supported, got {minval}"
                    minval = minval.raw_tensor
                if isinstance(maxval, Tensor):
                    assert maxval.dims == (), f"only scalar maxval supported, got {maxval}"
                    maxval = maxval.raw_tensor
                with torch.no_grad():
                    out.raw_tensor.uniform_(minval, maxval, generator=generator)  # noqa
            else:
                if minval is None:
                    minval = 0
                assert maxval is not None, "maxval must be specified for integer random uniform"
                if isinstance(minval, Tensor):
                    assert minval.dims == (), f"only scalar minval supported, got {minval}"
                    minval = minval.raw_tensor
                if isinstance(maxval, Tensor):
                    assert maxval.dims == (), f"only scalar maxval supported, got {maxval}"
                    maxval = maxval.raw_tensor
                with torch.no_grad():
                    out.raw_tensor.random_(minval, maxval, generator=generator)
        elif distribution == "normal":
            assert minval is None and maxval is None
            if mean is None:
                mean = 0
            if stddev is None:
                stddev = 1
            if isinstance(mean, Tensor):
                assert mean.dims == (), f"only scalar mean supported, got {mean}"
                mean = mean.raw_tensor
            if isinstance(stddev, Tensor):
                assert stddev.dims == (), f"only scalar stddev supported, got {stddev}"
                stddev = stddev.raw_tensor
            with torch.no_grad():
                out.raw_tensor.normal_(mean, stddev, generator=generator)
        elif distribution == "truncated_normal":
            if mean is None:
                mean = 0
            if stddev is None:
                stddev = 1
            if minval is None:
                minval = mean - 2 * stddev
            if maxval is None:
                maxval = mean + 2 * stddev

            from . import _rand

            _rand.no_grad_trunc_normal_(out.raw_tensor, mean=mean, std=stddev, a=minval, b=maxval, generator=generator)
        else:
            raise NotImplementedError(f"random distribution {distribution} not implemented")
        return out
