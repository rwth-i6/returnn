"""
Backend for exposing JAX-specific functionality.

JAX dispatches op-by-op like PyTorch, so this backend is eager,
and the same code runs unchanged inside ``jax.jit``
(the raw tensors are tracers there, see :func:`returnn.frontend._backend.get_backend_by_raw_tensor_type`).
Everything that must NOT read a value on the host (shapes, seq lens)
therefore follows the same rules as the static-traceable regime of the PyTorch backend.
"""

from __future__ import annotations
from typing import Optional, Union, Sequence, Tuple
import numpy
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as _logsumexp

from returnn.tensor import Tensor, Dim

# noinspection PyProtectedMember
from returnn.frontend._backend import Backend
from returnn.frontend import RawTensorTypes
import returnn.frontend as rf

_TT = Tensor[jax.Array]


# Ignore this warning until we really expect that we implemented everything.
# noinspection PyAbstractClass
class JaxBackend(Backend[jax.Array]):
    """
    JAX backend
    """

    name = "jax"
    RawTensorType = jax.Array

    @staticmethod
    def executing_eagerly() -> bool:
        """
        :return: whether we are executing eagerly.
            True also under jit: the RF code path is the same, only the raw tensors are tracers.
        """
        return True

    @staticmethod
    def assert_(condition: Tensor, message: str, *, stop: bool = True):
        """assert"""
        assert condition.dims == (), "condition for assert must be a scalar"
        if bool(condition.raw_tensor):
            return
        if stop:
            raise AssertionError(message)
        print(f"[ASSERT FAILED WARNING]: {message}")

    @staticmethod
    def raw_to_numpy(raw_tensor: jax.Array) -> numpy.ndarray:
        """
        :param raw_tensor:
        :return: numpy array with the same content
        """
        return numpy.asarray(jax.device_get(raw_tensor))

    @staticmethod
    def get_dtype_name_raw(raw_tensor: jax.Array) -> str:
        """
        :return: dtype of raw tensor, as string
        """
        return raw_tensor.dtype.name

    @staticmethod
    def as_dtype_raw(dtype_name: str) -> numpy.dtype:
        """
        :param dtype_name: e.g. "float32"
        :return: dtype object.
            JAX uses NumPy dtypes (bfloat16 and friends come from ml_dtypes, registered by jax).
        """
        return jnp.dtype(dtype_name)

    @staticmethod
    def get_ndim_raw(raw_tensor: jax.Array) -> int:
        """
        :return: ndim of raw tensor
        """
        return raw_tensor.ndim

    @staticmethod
    def get_shape_raw(raw_tensor: jax.Array) -> Tuple[int, ...]:
        """
        :return: shape of raw tensor.
            Always static: JAX has no dynamic shapes, that is the point of the bound-shape regime.
        """
        return tuple(raw_tensor.shape)

    @staticmethod
    def get_shape_tuple_raw(raw_tensor: jax.Array) -> Tuple[int, ...]:
        """
        :return: shape of raw tensor
        """
        return tuple(raw_tensor.shape)

    @staticmethod
    def get_known_shape_raw(raw_tensor: jax.Array) -> Tuple[Optional[int], ...]:
        """
        :return: shape of raw tensor; here for JAX the full shape is always known
        """
        return tuple(raw_tensor.shape)

    @staticmethod
    def get_new_dim_raw(raw_tensor: jax.Array, axis: int, *, name: str) -> Dim:
        """
        :param raw_tensor:
        :param axis:
        :param name:
        :return: new Dim object
        """
        return Dim(int(raw_tensor.shape[axis]), name=name)

    @staticmethod
    def get_device(x: Tensor[jax.Array]) -> Optional[str]:
        """
        :return: device of the tensor, in the RF naming ("cpu", "cuda:0", ...)
        """
        raw_tensor: jax.Array = x.raw_tensor
        if raw_tensor is None:
            return None
        return _device_to_str(raw_tensor.device)

    @staticmethod
    def copy_to_device(x: Tensor, device: Optional[str]) -> Tensor:
        """
        :param x:
        :param device:
        """
        if not device:
            return x
        x = x.copy()
        x.raw_tensor = jax.device_put(x.raw_tensor, _device_from_str(device))
        return x

    @staticmethod
    def convert_to_tensor(
        value: Union[Tensor, jax.Array, RawTensorTypes],
        *,
        dims: Sequence[Dim],
        dtype: str,
        sparse_dim: Optional[Dim] = None,
        feature_dim: Optional[Dim] = None,
        device: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Tensor[jax.Array]:
        """convert to tensor"""
        if isinstance(value, Tensor):
            return value
        if isinstance(value, jax.Array):
            name = name or "raw_tensor"
        else:
            name = name or "const"
            value = jnp.asarray(value, dtype=JaxBackend.as_dtype_raw(dtype))
        if device:
            value = jax.device_put(value, _device_from_str(device))
        return Tensor(name, dims=dims, dtype=dtype, sparse_dim=sparse_dim, feature_dim=feature_dim, raw_tensor=value)

    @staticmethod
    def copy(tensor: Tensor[jax.Array]) -> Tensor[jax.Array]:
        """
        copy.
        JAX arrays are immutable, so the raw tensor can be shared;
        only the Tensor wrapper needs to be fresh.
        """
        out = tensor.copy_template()
        out.raw_tensor = tensor.raw_tensor
        return out

    @staticmethod
    def cast_raw(raw_tensor: jax.Array, dtype: str) -> jax.Array:
        """cast"""
        return raw_tensor.astype(JaxBackend.as_dtype_raw(dtype))

    @staticmethod
    def expand_dims_raw(raw_tensor: jax.Array, axis: int) -> jax.Array:
        """
        :param raw_tensor:
        :param axis: e.g. 1
        :return: raw tensor with new axis
        """
        return jnp.expand_dims(raw_tensor, axis)

    @staticmethod
    def expand_raw(raw_tensor: jax.Array, axis: int, dim: Union[int, jax.Array]) -> jax.Array:
        """
        :param raw_tensor:
        :param axis: shape[axis] must be 1
        :param dim: the new dim for shape[axis]
        :return: shape[axis] expanded to dim
        """
        return jnp.broadcast_to(raw_tensor, raw_tensor.shape[:axis] + (dim,) + raw_tensor.shape[axis + 1 :])

    @staticmethod
    def transpose_raw(raw_tensor: jax.Array, perm: Sequence[int]) -> jax.Array:
        """
        :param raw_tensor:
        :param perm: e.g. [0, 2, 1]
        :return: permuted (transposed) raw tensor
        """
        if all(p == i for i, p in enumerate(perm)):
            return raw_tensor
        return jnp.transpose(raw_tensor, tuple(perm))

    @staticmethod
    def reshape_raw(raw_tensor: jax.Array, shape: Union[Sequence[Union[int, jax.Array]], jax.Array]) -> jax.Array:
        """reshape raw"""
        return jnp.reshape(raw_tensor, tuple(shape))

    @staticmethod
    def compare_raw(a: jax.Array, kind: str, b: jax.Array) -> jax.Array:
        """
        :param a:
        :param kind: "equal", "less", "less_equal", "greater", "greater_equal", "not_equal"
        :param b:
        :return: a `kind` b
        """
        assert a.ndim == b.ndim or a.ndim == 0 or b.ndim == 0
        op = getattr(jnp, kind)  # e.g. jnp.equal
        return op(a, b)

    @staticmethod
    def combine_raw(a: jax.Array, kind: str, b: jax.Array) -> jax.Array:
        """
        :param a:
        :param kind: "add", "sub", "mul", "truediv", "floordiv", "mod", "pow",
            "maximum", "minimum", "logical_and", "logical_or", "squared_difference"
        :param b:
        :return: a `kind` b
        """
        assert a.ndim == b.ndim or a.ndim == 0 or b.ndim == 0
        if kind == "squared_difference":
            return jnp.square(jnp.subtract(a, b))
        op = getattr(jnp, _CombineKindMap.get(kind, kind), None)
        if not op:
            raise ValueError(f"RF JaxBackend: combine kind {kind!r} not supported")
        return op(a, b)

    @staticmethod
    def activation_raw(raw_tensor: jax.Array, func: str) -> jax.Array:
        """
        :param raw_tensor:
        :param func: e.g. "tanh"
        :return: raw tensor after activation
        """
        assert func in Backend._AllowedActivationFuncs
        f = _ActivationFuncMap.get(func)
        if not f:
            # jnp has the numpy-ish ones, jax.nn the neural ones, jax.lax the rest (rsqrt).
            for mod in (jnp, jax.nn, jax.lax):
                f = getattr(mod, func, None)
                if f:
                    break
            if not f:
                raise ValueError(f"unknown activation function {func!r}")
        return f(raw_tensor)

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
        true_ = rf.convert_to_tensor(true_, _backend=JaxBackend, dtype=dtype, device=cond.device)
        false_ = rf.convert_to_tensor(false_, _backend=JaxBackend, dtype=dtype, device=cond.device)
        out = Tensor.get_common_data(
            [true_, false_, cond], allow_broadcast_all_sources=allow_broadcast_all_sources, name="where"
        )
        out.dtype = true_.dtype
        out.sparse_dim = true_.sparse_dim or false_.sparse_dim
        out.feature_dim = true_.feature_dim or false_.feature_dim
        out.raw_tensor = jnp.where(
            cond.copy_compatible_to_dims_raw(out.dims),
            true_.copy_compatible_to_dims_raw(out.dims),
            false_.copy_compatible_to_dims_raw(out.dims),
        )
        return out

    @staticmethod
    def range_over_dim(dim: Dim, *, dtype: Optional[str] = None, device: Optional[str] = None) -> Tensor[jax.Array]:
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
        raw = jnp.arange(dim.get_dim_value(), dtype=JaxBackend.as_dtype_raw(dtype))
        out.raw_tensor = jax.device_put(raw, _device_from_str(device)) if device else raw
        return out

    @staticmethod
    def reduce(
        source: Tensor[jax.Array],
        *,
        mode: str,
        axis: Union[Dim, Sequence[Dim]],
        use_mask: bool = True,
    ) -> Tensor[jax.Array]:
        """reduce"""
        assert mode in Backend._AllowedReduceModes
        if use_mask:
            # Masked reduce needs the seq mask, which comes with the array ops.
            if isinstance(axis, Dim):
                assert not axis.need_masking(), "RF JaxBackend: masked reduce not implemented yet"
            else:
                assert all(not dim.need_masking() for dim in axis), "RF JaxBackend: masked reduce not implemented yet"
        axes = [axis] if isinstance(axis, Dim) else list(axis)
        raw_axes = [source.get_axis_from_description(dim) for dim in axes]
        res_dims = [dim for i, dim in enumerate(source.dims) if i not in raw_axes]
        if mode in ("argmin", "argmax"):
            assert len(raw_axes) == 1, f"RF JaxBackend: {mode} needs exactly one axis, got {axes}"
            raw_result = getattr(jnp, mode)(source.raw_tensor, axis=raw_axes[0])
            out_dtype = rf.get_default_array_index_dtype()
            raw_result = raw_result.astype(JaxBackend.as_dtype_raw(out_dtype))
            sparse_dim = axes[0]
        else:
            # logsumexp is not in jnp, it lives in jax.scipy.special
            assert mode in _ReduceModeMap or mode == "logsumexp", f"RF JaxBackend: reduce mode {mode!r}"
            func = _ReduceModeMap.get(mode, _logsumexp)
            raw_result = func(source.raw_tensor, axis=tuple(raw_axes))
            out_dtype = "bool" if mode in ("any", "all") else source.dtype
            sparse_dim = source.sparse_dim
        return Tensor(
            name=f"reduce_{mode}",
            raw_tensor=raw_result,
            dims=res_dims,
            dtype=out_dtype,
            sparse_dim=sparse_dim,
        )


_CombineKindMap = {
    "sub": "subtract",
    "mul": "multiply",
    "truediv": "true_divide",
    "floordiv": "floor_divide",
    "pow": "power",
    "mod": "remainder",
}

_ActivationFuncMap = {
    "neg": jnp.negative,
    # jax.nn.gelu defaults to the tanh approximation, torch.nn.functional.gelu does not.
    # RF semantics follow the exact (erf) form.
    "gelu": lambda x: jax.nn.gelu(x, approximate=False),
}

_ReduceModeMap = {"sum": jnp.sum, "max": jnp.max, "min": jnp.min, "mean": jnp.mean, "any": jnp.any, "all": jnp.all}


def _device_to_str(device: jax.Device) -> str:
    """
    :param device: JAX device
    :return: RF device string, i.e. the PyTorch naming ("cpu", "cuda:0", ...)
    """
    if device.platform == "cpu":
        return "cpu"
    if device.platform == "gpu":
        return f"cuda:{device.id}"
    return f"{device.platform}:{device.id}"


def _device_from_str(device: str) -> jax.Device:
    """
    :param device: RF device string ("cpu", "cuda", "cuda:1", ...)
    :return: JAX device
    """
    kind, _, idx = device.partition(":")
    if kind == "cuda":
        kind = "gpu"
    devices = jax.devices(kind)
    return devices[int(idx)] if idx else devices[0]
