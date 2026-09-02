"""
Backend for exposing JAX-specific functionality.

JAX dispatches op-by-op like PyTorch, so this backend is eager,
and the same code runs unchanged inside ``jax.jit``
(the raw tensors are tracers there, see :func:`returnn.frontend._backend.get_backend_by_raw_tensor_type`).
Everything that must not read a value on the host (shapes, seq lens)
therefore follows the same rules as the static-traceable regime of the PyTorch backend.
"""

from __future__ import annotations
from typing import Any, Callable, Optional, Union, Sequence, Tuple, List, Dict
import contextlib
from functools import partial
import itertools
import numpy
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as _logsumexp
import optax

from returnn.tensor import Tensor, Dim
from returnn.util.basic import get_global_inf_value, prod

# noinspection PyProtectedMember
from returnn.frontend._backend import Backend
from returnn.frontend import RawTensorTypes
import returnn.frontend as rf

_TT = Tensor[jax.Array]


class _BufferTensorArray:
    """
    A TensorArray as a preallocated buffer plus a write index.

    The eager path keeps a Python list, which cannot be a ``lax.while_loop`` carry:
    the carry must have one fixed shape across iterations.
    So a TensorArray that travels through the graph loop takes this form,
    which needs the entry count up front (``TensorArray(capacity=...)``).
    """

    def __init__(
        self,
        *,
        capacity: int,
        buffer: Optional[jax.Array] = None,
        index: Optional[jax.Array] = None,
        template: Optional[Tensor] = None,
    ):
        """
        :param capacity: max number of entries
        :param buffer: [capacity, ...], allocated on the first push
        :param index: next write position
        :param template: dims and dtype of one entry, taken from the first push
        """
        self.capacity = capacity
        self.buffer = buffer
        self.index = jnp.asarray(0, dtype=jnp.int32) if index is None else index
        self.template = template


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
        """
        assert, on a traced condition as well

        Under tracing the check becomes part of the program:
        a ``lax.cond`` whose taken branch calls back to the host,
        as a plain Python ``assert`` would test a tracer and always pass.
        """
        assert condition.dims == (), "condition for assert must be a scalar"
        raw = condition.raw_tensor

        def _failed():
            if stop:
                raise AssertionError(message)
            print(f"[ASSERT FAILED WARNING]: {message}")

        if isinstance(raw, jax.core.Tracer):
            jax.lax.cond(jnp.logical_not(raw), lambda _: jax.debug.callback(_failed) or 0, lambda _: 0, 0)
            return
        if bool(raw):
            return
        _failed()

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
        raw_tensor: Optional[jax.Array] = x.raw_tensor
        if raw_tensor is None:
            return None
        # Under a JAX transform (jit, grad, vmap) the raw tensor is a tracer, which has no device:
        # the value does not exist yet, so its placement is not a question that can be answered here.
        # RF treats None as unknown and then does not force any placement, which is what we want.
        device = getattr(raw_tensor, "device", None)
        if device is None:
            return None
        return _device_to_str(device)

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
        # Scalars are deliberately not placed on a device.
        # RF asks for scalars on the CPU (keep_scalar_on_cpu), which is a PyTorch optimization:
        # there a CPU scalar can meet a CUDA tensor.
        # In JAX a device_put commits the array,
        # and an op mixing arrays committed to different devices is an error,
        # so committing the scalar would break every "tensor + 2.0" on the GPU.
        # Left uncommitted, JAX co-locates it with the other operand.
        if device and value.ndim > 0:
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
    def should_pickle_tensor(raw_tensor: jax.Array) -> bool:
        """
        Never: under tracing a raw is a tracer, which cannot be pickled at all.
        DistributeFilesDataset spawns workers mid-epoch, and spawning pickles the global config,
        which holds the extern_data dims the engine filled in -- that killed a run.
        """
        return False

    @staticmethod
    def compare_raw(a: jax.Array, kind: str, b: jax.Array) -> jax.Array:
        """
        :param a:
        :param kind: "equal", "less", "less_equal", "greater", "greater_equal", "not_equal"
        :param b:
        :return: a `kind` b
        """
        assert a.ndim == b.ndim or a.ndim == 0 or b.ndim == 0
        a, b = _match_device(a, b)
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
        a, b = _match_device(a, b)
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
        if mode in ("sum", "mean", "logsumexp"):
            # mixed precision: accumulating in the reduced dtype is where it hurts most,
            # and the norms/statistics of the model are built on these
            source = rf.amp_cast_float32(source)
        axes = [axis] if isinstance(axis, Dim) else list(axis)
        raw_axes = [source.get_axis_from_description(dim) for dim in axes]
        res_dims = [dim for i, dim in enumerate(source.dims) if i not in raw_axes]
        correction_factor = None
        if use_mask and any(dim.need_masking() for dim in axes):
            source = source.copy()
            dtype = source.raw_tensor.dtype
            # replace the padded frames by the neutral element of the reduction
            if mode in ("max", "logsumexp", "argmax"):
                mask_value = _dtype_min(dtype)
            elif mode in ("min", "argmin"):
                mask_value = _dtype_max(dtype)
            elif mode == "sum":
                mask_value = 0
            elif mode == "mean":
                # summing over the padded frames as 0 and dividing by the full width is off by exactly this factor
                mask_value = 0
                # source.device is None while tracing (jax.value_and_grad traces even in eager mode),
                # and masked_fraction_of_shape then builds the factor on the cpu by default,
                # where it cannot meet the traced result. The default device is where the step runs.
                correction_factor = rf.masked_fraction_of_shape(
                    axes, inverse=True, device=source.device or rf.get_default_device()
                )
            elif mode == "all":
                mask_value = True
            elif mode == "any":
                mask_value = False
            else:
                raise NotImplementedError(f"RF JaxBackend: reduce_{mode} with masking on {source!r}")
            for dim in axes:
                if dim.need_masking():
                    mask = source.get_sequence_mask_broadcast(dim)
                    source.raw_tensor = jnp.where(mask, source.raw_tensor, jnp.asarray(mask_value, dtype=dtype))
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
            # read the dtype off the result, do not assume the source's:
            # summing int32 gives int64 (numpy promotion, visible because x64 is on), same as torch does
            out_dtype = JaxBackend.get_dtype_name_raw(raw_result)
            sparse_dim = source.sparse_dim
        if correction_factor is not None:
            # cast: the factor is computed from the seq lens and can come out as float64 under x64,
            # which would silently widen the result away from its declared dtype
            factor = correction_factor.copy_compatible_to_dims_raw(res_dims)
            factor, raw_result = _match_device(factor.astype(raw_result.dtype), raw_result)
            raw_result = raw_result * factor
        return Tensor(
            name=f"reduce_{mode}",
            raw_tensor=raw_result,
            dims=res_dims,
            dtype=out_dtype,
            sparse_dim=sparse_dim,
        )

    @staticmethod
    def gather(source: Tensor, *, indices: Union[Tensor, int], axis: Dim, clip_to_valid: bool = False) -> Tensor:
        """gather"""
        axis_int = source.get_axis_from_description(axis, allow_int=False)
        if isinstance(indices, int):
            out = Tensor(
                "gather",
                dims=source.dims[:axis_int] + source.dims[axis_int + 1 :],
                dtype=source.dtype,
                sparse_dim=source.sparse_dim,
            )
            if source.feature_dim and source.feature_dim in out.dims:
                out.feature_dim = source.feature_dim
            index = indices
            if clip_to_valid and index != 0:
                index = min(max(index, 0), source.raw_tensor.shape[axis_int] - 1)
            out.raw_tensor = jax.lax.index_in_dim(source.raw_tensor, index, axis=axis_int, keepdims=False)
            return out
        assert isinstance(indices, Tensor), f"gather: unsupported type for indices: {type(indices)}"
        if clip_to_valid:
            if axis.dyn_size_ext is not None:
                indices = rf.clip_by_value(
                    indices,
                    0,
                    rf.relu(rf.cast(axis.get_dyn_size_ext_for_device(indices.device), indices.dtype) - 1),
                    allow_broadcast_all_sources=True,
                )
            else:
                indices = indices.copy()
                indices.raw_tensor = jnp.clip(indices.raw_tensor, 0, source.raw_tensor.shape[axis_int] - 1)
        index_own_dims = [dim for dim in indices.dims if dim not in source.dims or dim == axis]
        out = Tensor(
            "gather",
            dims=list(source.dims[:axis_int]) + index_own_dims + list(source.dims[axis_int + 1 :]),
            dtype=source.dtype,
            sparse_dim=source.sparse_dim,
        )
        if source.feature_dim and source.feature_dim in out.dims:
            out.feature_dim = source.feature_dim
        if indices.dims_set.intersection(source.dims_set - {axis}):
            # indices vary along dims of the source: one index per (batch, ...) position, i.e. take_along_axis.
            # Bring indices into exactly the source's layout, with axis replaced by the index's own dims.
            indices = indices.copy_compatible_to(out, check_dtype=False, check_sparse=False, unbroadcast=True)
            if len(index_own_dims) == 1:
                index_own_dims_flat = index_own_dims[0]
            elif len(index_own_dims) == 0:
                index_own_dims_flat = Dim(1, name="dummy")
                indices = indices.copy_add_dim_by_tag(index_own_dims_flat, unbroadcast=True, axis=axis_int)
            else:
                indices, index_own_dims_flat = rf.merge_dims(indices, dims=index_own_dims)
            index_ext_dims = list(source.dims)
            index_ext_dims[axis_int] = index_own_dims_flat
            assert indices.dims == tuple(index_ext_dims)
            # the indices should be the same device as the source
            idx_raw = _to_device_of(indices.raw_tensor.astype(jnp.int32), source.raw_tensor)
            out_raw = jnp.take_along_axis(source.raw_tensor, idx_raw, axis=axis_int, mode="clip")
            if len(index_own_dims) == 0:
                out_raw = jnp.squeeze(out_raw, axis=axis_int)
            elif len(index_own_dims) > 1:
                out_raw = jnp.reshape(out_raw, [d.get_dim_value() for d in out.dims])
            out.raw_tensor = out_raw
        else:
            # indices are independent of the source's other dims: a plain take along axis
            idx_raw = _to_device_of(indices.raw_tensor.astype(jnp.int32).reshape(-1), source.raw_tensor)
            out_raw = jnp.take(source.raw_tensor, idx_raw, axis=axis_int, mode="clip")
            out_shape = (
                source.raw_tensor.shape[:axis_int]
                + tuple(indices.raw_tensor.shape)
                + source.raw_tensor.shape[axis_int + 1 :]
            )
            out.raw_tensor = jnp.reshape(out_raw, out_shape)
        return out

    @staticmethod
    def scatter(
        source: Tensor,
        *,
        indices: Tensor,
        indices_dim: Union[Dim, Sequence[Dim]],
        mode: str,
        fill_value: Union[int, float],
        out_dim: Union[Dim, Sequence[Dim]],
    ) -> Tensor:
        """
        Scatters into a new tensor filled with fill_value.
        Duplicated indices are combined by mode.

        :param source: [batch_dims..., indices_dim(s)..., feature_dims...]
        :param indices: [batch_dims..., indices_dim(s)...] -> out_dim
        :param indices_dim:
        :param mode: "sum", "max", "min"
        :param fill_value: must be the neutral element of mode
            (JAX's scatter always combines with what is already there, so a non-neutral fill would leak in)
        :param out_dim:
        :return: [batch_dims..., out_dim, feature_dims...]
        """
        indices_dim = [indices_dim] if isinstance(indices_dim, Dim) else list(indices_dim)
        assert indices.dtype.startswith("int")
        if isinstance(out_dim, Dim):
            out_dim = [out_dim]
        else:
            out_dim = list(out_dim)
        out_flat_dim = out_dim[0]
        for dim in out_dim[1:]:
            out_flat_dim = out_flat_dim * dim
        batch_dims = indices.remaining_dims(indices_dim)
        feature_dims = source.remaining_dims(batch_dims + indices_dim)
        if len(indices_dim) > 1:
            indices, indices_flat_dim = rf.merge_dims(indices, dims=indices_dim)
            source, _ = rf.merge_dims(source, dims=indices_dim, out_dim=indices_flat_dim)
        else:
            indices_flat_dim = indices_dim[0]
        source = source.copy_transpose(batch_dims + [indices_flat_dim] + feature_dims)
        indices = indices.copy_compatible_to(
            source, unbroadcast=True, add_dims=True, check_sparse=False, check_dtype=False
        )
        out_dims = batch_dims + [out_flat_dim] + feature_dims
        out_shape = [d.get_dim_value() for d in out_dims]
        # flatten to [batch, indices, feature] so one advanced-index scatter covers every case
        batch_size = int(numpy.prod([d.get_dim_value() for d in batch_dims], dtype="int64")) if batch_dims else 1
        feature_size = int(numpy.prod([d.get_dim_value() for d in feature_dims], dtype="int64")) if feature_dims else 1
        src_raw = jnp.reshape(source.raw_tensor, (batch_size, -1, feature_size))
        idx_raw = jnp.reshape(indices.raw_tensor.astype(jnp.int32), (batch_size, -1, feature_size))
        out_raw = jnp.full((batch_size, out_flat_dim.get_dim_value(), feature_size), fill_value, dtype=src_raw.dtype)
        b_idx = jnp.arange(batch_size)[:, None, None]
        f_idx = jnp.arange(feature_size)[None, None, :]
        if mode == "sum":
            assert fill_value == 0, f"RF JaxBackend: scatter sum needs fill_value 0, got {fill_value}"
            out_raw = out_raw.at[b_idx, idx_raw, f_idx].add(src_raw)
        elif mode == "max":
            out_raw = out_raw.at[b_idx, idx_raw, f_idx].max(src_raw)
        elif mode == "min":
            out_raw = out_raw.at[b_idx, idx_raw, f_idx].min(src_raw)
        else:
            raise ValueError(f"RF JaxBackend: scatter mode {mode!r} not supported")
        res = Tensor(
            "scatter",
            dims=out_dims,
            dtype=source.dtype,
            sparse_dim=source.sparse_dim,
            raw_tensor=jnp.reshape(out_raw, out_shape),
        )
        if len(out_dim) > 1:
            res = rf.split_dims(res, axis=out_flat_dim, dims=out_dim)
        return res

    @staticmethod
    def pad(
        source: Tensor,
        *,
        axes: Sequence[Dim],
        padding: Sequence[Tuple[Union[Dim, int, Tensor], Union[Dim, int, Tensor]]],
        out_dims: Sequence[Dim],
        handle_dynamic_dims: bool,
        mode: str = "constant",
        value: Optional[Union[rf.RawTensorTypes, Tensor]] = None,
    ) -> Tensor:
        """pad"""
        assert len(out_dims) == len(axes) == len(padding)
        raw_pad = []
        for dim in source.dims:
            if dim not in axes:
                raw_pad.append((0, 0))
                continue
            left, right = padding[axes.index(dim)]
            raw_pad.append((_pad_amount(left, handle_dynamic_dims), _pad_amount(right, handle_dynamic_dims)))
        jnp_mode = _PadModeMap.get(mode, mode)
        value_ = (value.raw_tensor if not value.dims else None) if isinstance(value, Tensor) else value
        if isinstance(value, Tensor) and value.dims:
            # jnp.pad fills with a scalar only, so build the padding from the value itself,
            # as the torch backend does
            assert all(dim in source.dims and dim not in axes for dim in value.dims)
            assert len(axes) == 1, "RF JaxBackend: pad with a non-scalar value, only a single axis is supported"
            assert jnp_mode == "constant", f"RF JaxBackend: pad mode {mode} with a non-scalar value"
            pad_left, pad_right = padding[0]
            pad_left = pad_left if isinstance(pad_left, Dim) else Dim(pad_left, name="pad_left")
            pad_right = pad_right if isinstance(pad_right, Dim) else Dim(pad_right, name="pad_right")
            out = JaxBackend.concat(
                *(
                    ([(rf.expand_dim(value, pad_left), pad_left)] if pad_left.dimension else [])
                    + [(source, axes[0])]
                    + ([(rf.expand_dim(value, pad_right), pad_right)] if pad_right.dimension else [])
                ),
                allow_broadcast=True,
                out_dim=out_dims[0],
            )
        else:
            out = source.copy_template_new_dim_tags(
                [out_dims[axes.index(dim)] if dim in axes else dim for dim in source.dim_tags], keep_special_axes=True
            )
            if jnp_mode == "constant":
                out.raw_tensor = jnp.pad(
                    source.raw_tensor, raw_pad, mode="constant", constant_values=value_ if value_ is not None else 0
                )
            else:
                out.raw_tensor = jnp.pad(source.raw_tensor, raw_pad, mode=jnp_mode)
        if handle_dynamic_dims and any(dim.need_masking() for dim in out_dims):
            if all(right == 0 for _, right in raw_pad) and mode != "circular":
                return out  # nothing padded on the right, so no valid frame moved
            assert mode == "constant", (
                f"RF JaxBackend: pad mode {mode} not implemented with dynamic dims and handle_dynamic_dims=True"
            )
            for out_dim, middle, (left, right) in zip(out_dims, axes, padding):
                if not (middle.need_masking() or (isinstance(left, Dim) and left.need_masking())):
                    continue
                if isinstance(left, Dim):
                    left = left.get_size_tensor(device=out.device)
                mask = rf.compare_bc(
                    rf.range_over_dim(out_dim, device=out.device),
                    "<",
                    left + middle.get_size_tensor(device=out.device),
                )
                out.raw_tensor = jnp.where(
                    mask.copy_compatible_to_dims_raw(out.dims),
                    out.raw_tensor,
                    jnp.asarray(value_ or 0, dtype=out.raw_tensor.dtype),
                )
        return out

    # --- shapes and indexing

    @classmethod
    def squeeze_raw(cls, raw_tensor: jax.Array, axes: Sequence[int]) -> jax.Array:
        """squeeze"""
        return jnp.squeeze(raw_tensor, axis=tuple(axes))

    @staticmethod
    def merge_dims(source: Tensor, *, dims: Sequence[Dim], out_dim: Dim) -> Tensor:
        """
        :param source:
        :param dims: dims to merge, len >= 2
        :param out_dim: the merged dim
        :return: source with dims merged into out_dim
        """
        assert len(dims) >= 2
        first_axis = min(source.dims.index(d) for d in dims)
        pre_dims = source.dims[:first_axis]
        post_dims = [d for d in source.dims if d not in dims and d not in pre_dims]
        source = source.copy_transpose(tuple(pre_dims) + tuple(dims) + tuple(post_dims), allow_int=False)
        out = Tensor(
            "merge_dims",
            dims=pre_dims + (out_dim,) + tuple(post_dims),
            dtype=source.dtype,
            sparse_dim=source.sparse_dim,
        )
        # -1 for the merged block, so no dim value has to be read on the host
        src_shape = source.raw_tensor.shape
        out_shape = list(src_shape[: len(pre_dims)]) + [-1] + list(src_shape[len(pre_dims) + len(dims) :])
        out.raw_tensor = jnp.reshape(source.raw_tensor, out_shape)
        if source.feature_dim is not None:
            out.feature_dim = out_dim if source.feature_dim in dims else source.feature_dim
        return out

    @staticmethod
    def split_dims(
        source: Tensor,
        *,
        axis: Dim,
        dims: Sequence[Dim],
        pad_to_multiples: Optional[bool] = None,
        pad_value: Union[None, int, float] = None,
    ) -> Tensor:
        """split dims"""
        assert pad_to_multiples in (None, False), "RF JaxBackend: split_dims pad_to_multiples not implemented"
        axis_int = source.get_axis_from_description(axis)
        out_dims = source.dims[:axis_int] + tuple(dims) + source.dims[axis_int + 1 :]
        src_shape = source.raw_tensor.shape
        split_sizes = [d.dimension if d.dimension is not None else -1 for d in dims]
        if split_sizes.count(-1) > 1:
            split_sizes = [d.get_dim_value() for d in dims]
        out_shape = list(src_shape[:axis_int]) + split_sizes + list(src_shape[axis_int + 1 :])
        out = Tensor(
            "split_dims",
            dims=out_dims,
            dtype=source.dtype,
            sparse_dim=source.sparse_dim,
            raw_tensor=jnp.reshape(source.raw_tensor, out_shape),
        )
        if source.feature_dim and source.feature_dim != axis:
            out.feature_dim = source.feature_dim
        return out

    @staticmethod
    def reshape(source: Tensor, in_dims: Sequence[Dim], out_dims: Sequence[Dim]) -> Tensor:
        """reshape"""
        in_dims_axes = [source.get_axis_from_description(d, allow_int=False) for d in in_dims]
        assert sorted(set(in_dims_axes)) == sorted(in_dims_axes), f"reshape {source}: invalid in_dims {in_dims}"
        insert_axis = min(in_dims_axes)
        dims = list(source.dim_tags)
        permute = list(range(source.batch_ndim))
        for axis in sorted(set(in_dims_axes), reverse=True):
            dims.pop(axis)
            permute.pop(axis)
        # bring the in_dims next to each other, in the given order, at the position of the first of them
        permute = permute[:insert_axis] + in_dims_axes + permute[insert_axis:]
        source = source.copy_transpose(permute)
        dims = dims[:insert_axis] + list(out_dims) + dims[insert_axis:]
        out = Tensor("reshape", dims=dims, dtype=source.dtype, sparse_dim=source.sparse_dim)
        if source.feature_dim and source.feature_dim not in in_dims:
            out.feature_dim = source.feature_dim
        out.raw_tensor = jnp.reshape(source.raw_tensor, [d.get_dim_value() for d in dims])
        return out

    @staticmethod
    def split(source: Tensor, *, axis: Dim, out_dims: Sequence[Dim]) -> Tuple[Tensor, ...]:
        """split"""
        axis_int = source.get_axis_from_description(axis)
        sizes = [d.get_dim_value() for d in out_dims]
        # jnp.split takes split points, not sizes
        bounds = list(itertools.accumulate(sizes))[:-1]
        raws = jnp.split(source.raw_tensor, bounds, axis=axis_int)
        out_tuple = tuple(
            source.copy_template_replace_dim_tag(axis=axis_int, new_dim_tag=dim, name=f"split{i}")
            for i, dim in enumerate(out_dims)
        )
        for out, raw in zip(out_tuple, raws):
            out.raw_tensor = raw
        return out_tuple

    @staticmethod
    def expand_dim(source: Tensor, dim: Dim) -> Tensor:
        """expand dim"""
        assert dim not in source.dims
        # same placement heuristic as the other backends
        axis = len(source.dims)
        if dim.is_static() and source.have_feature_axis():
            axis = source.feature_dim_axis
        if dim.is_dynamic():
            for i, d in reversed(list(enumerate(source.dims))):
                if d.is_dynamic():
                    axis = i + 1
                    break
        new_dims = list(source.dims)
        new_dims.insert(axis, dim)
        out = source.copy_template_new_dim_tags(new_dims)
        if source.feature_dim:
            out.feature_dim = source.feature_dim
        out_raw = jnp.expand_dims(source.raw_tensor, axis)
        if dim.is_dynamic() or dim.dimension != 1:
            out_raw = jnp.broadcast_to(
                out_raw, tuple(dim.get_dim_value() if d == dim else s for d, s in zip(out.dims, out_raw.shape))
            )
        out.raw_tensor = out_raw
        return out

    @staticmethod
    def squeeze(source: Tensor, axis: Dim) -> Tensor:
        """squeeze"""
        axis_int = source.get_axis_from_description(axis)
        out = source.copy_template_excluding_axis(axis_int)
        out.raw_tensor = jnp.squeeze(source.raw_tensor, axis=axis_int)
        return out

    @staticmethod
    def concat(*sources: Tuple[Tensor, Dim], allow_broadcast: bool = False, out_dim: Dim) -> Tensor:
        """concat"""
        axis = sources[0][0].get_axis_from_description(sources[0][1])
        other_dims = list(sources[0][0].dims)
        other_dims.remove(sources[0][1])
        need_broadcast = False
        if allow_broadcast:
            for source, dim in sources[1:]:
                assert dim in source.dims
                if set(source.dims) - {dim} != set(other_dims):
                    need_broadcast = True
                for dim_ in source.dims:
                    if dim_ != dim and dim_ not in other_dims:
                        other_dims.append(dim_)
        sources_raw = []
        for source, dim in sources:
            templ_dims = other_dims[:axis] + [dim] + other_dims[axis:]
            if allow_broadcast and need_broadcast:
                templ = Tensor(source.name, dims=templ_dims, dtype=source.dtype, sparse_dim=source.sparse_dim)
                sources_raw.append(source.copy_compatible_to(templ, unbroadcast=True).raw_tensor)
            else:
                assert set(templ_dims) == set(source.dims), (
                    f"concat {source} {dim} not allowed with allow_broadcast=False"
                )
                sources_raw.append(source.copy_transpose(templ_dims).raw_tensor)
        out_raw = jnp.concatenate(sources_raw, axis=axis)
        out = Tensor(
            "concat",
            dims=other_dims[:axis] + [out_dim] + other_dims[axis:],
            dtype=JaxBackend.get_dtype_name_raw(out_raw),
            sparse_dim=sources[0][0].sparse_dim,
            raw_tensor=out_raw,
        )
        if sources[0][0].feature_dim and sources[0][0].feature_dim != sources[0][1]:
            out.feature_dim = sources[0][0].feature_dim
        return out

    @staticmethod
    def stack(sources: Sequence[Tensor], *, out_dim: Dim) -> Tensor:
        """stack"""
        out_dims = (out_dim,) + sources[0].dims
        out = Tensor("stack", dims=out_dims, dtype=sources[0].dtype, sparse_dim=sources[0].sparse_dim)
        out.raw_tensor = jnp.stack([s.copy_compatible_to_dims_raw(out_dims[1:]) for s in sources], axis=0)
        return out

    @staticmethod
    def unstack(source: Tensor, *, axis: Dim) -> Tuple[Tensor, ...]:
        """unstack via torch.unbind"""
        axis_int = source.dims.index(axis)
        template = source.copy_template_excluding_axis(axis_int)
        result = []
        for i in range(source.raw_tensor.shape[axis_int]):
            out = template.copy_template()
            out.raw_tensor = jax.lax.index_in_dim(source.raw_tensor, i, axis=axis_int, keepdims=False)
            result.append(out)
        return tuple(result)

    # a Python list while eager; a _BufferTensorArray when it must cross the graph loop
    TensorArrayType = Union[List[Tensor], _BufferTensorArray]

    @staticmethod
    def while_loop(
        cond: Callable[[Any], Union[bool, Tensor]],
        body: Callable[[Any], Any],
        initial: Any,
    ) -> Any:
        """
        ``jax.lax.while_loop`` over the RF loop-var structure: the loop stays in the graph
        instead of being unrolled, so the body is traced once whatever the trip count.

        Loop vars are the :class:`Tensor` and :class:`TensorArray` entries of the structure;
        everything else is loop-invariant and must come back unchanged from ``body``.
        The carry is fixed-shape, as ``lax`` requires:
        a Tensor may not change its dims per iteration,
        and a TensorArray needs ``TensorArray(capacity=...)`` so it is a buffer, not a list.

        :param cond: gets the loop vars, returns a scalar bool Tensor
        :param body: gets the loop vars, returns the next loop vars, same structure
        :param initial: initial loop vars
        :return: final loop vars
        """
        import tree
        from returnn.frontend.tensor_array import TensorArray

        flat_initial = list(tree.flatten(initial))
        var_idxs = [i for i, v in enumerate(flat_initial) if isinstance(v, (Tensor, TensorArray))]
        assert var_idxs, f"while_loop: no Tensor/TensorArray among the loop vars {initial}"

        # noinspection shadowing-names
        def _init_carry(v: Union[Tensor, TensorArray]) -> Any:
            """:param v: one loop var :return: its raw carry"""
            if isinstance(v, Tensor):
                return v.raw_tensor
            # noinspection PyProtectedMember
            raw = v._backend_tensor_array
            assert isinstance(raw, _BufferTensorArray), (
                f"while_loop: TensorArray {v.tensor_template} must be created with a capacity"
                f" to cross the graph loop, as lax.while_loop cannot grow the carry"
            )
            buffer = raw.buffer
            if buffer is None:  # nothing pushed yet, so allocate from the template
                tmpl = v.tensor_template
                shape = tuple(d.get_dim_value() for d in tmpl.dims)
                buffer = jnp.zeros((raw.capacity,) + shape, dtype=JaxBackend.as_dtype_raw(tmpl.dtype))
            return buffer, raw.index

        def _carry_like(v: Union[Tensor, TensorArray], ref: Union[Tensor, TensorArray]) -> Any:
            """
            :param v: the loop var after one body call
            :param ref: the same slot before it
            :return: its raw carry, in ``ref``'s dim order

            An RF Tensor is dim-order agnostic, but the lax carry is not:
            the body may return the same tensor transposed, which lax rejects as a type change.
            """
            if isinstance(v, Tensor):
                return v.copy_compatible_to_dims_raw(ref.dims)
            return _init_carry(v)

        # noinspection shadowing-names
        def _rebuild(carry: Sequence[Any]) -> Any:
            """:param carry: raw values :return: the loop vars as body/cond expect them"""
            out = list(flat_initial)
            for i, raw in zip(var_idxs, carry):
                value = flat_initial[i]
                if isinstance(value, Tensor):
                    tensor = value.copy_template()
                    tensor.raw_tensor = raw
                    out[i] = tensor
                else:
                    buffer, index = raw
                    # noinspection PyProtectedMember
                    prev = value._backend_tensor_array
                    out[i] = TensorArray(
                        tensor_template=value.tensor_template,
                        capacity=prev.capacity,
                        _backend_tensor_array=_BufferTensorArray(
                            capacity=prev.capacity,
                            buffer=buffer,
                            index=index,
                            template=prev.template if prev.template is not None else value.tensor_template,
                        ),
                        _backend=JaxBackend,
                    )
            return tree.unflatten_as(initial, out)

        def _cond(carry: Sequence[Any]):
            res = cond(_rebuild(carry))
            assert isinstance(res, Tensor) and res.dims == () and res.dtype == "bool", (
                f"while_loop: cond must return a scalar bool Tensor, got {res}"
            )
            return res.raw_tensor

        def _body(carry: Sequence[Any]):
            new = body(_rebuild(carry))
            tree.assert_same_structure(initial, new)
            flat_new = list(tree.flatten(new))
            for i in var_idxs:
                before, after = flat_initial[i], flat_new[i]
                if isinstance(before, Tensor):
                    assert isinstance(after, Tensor) and before.dims_set == after.dims_set, (
                        f"while_loop: loop var {i} changed its dims, {before} -> {after}."
                        f" The graph loop needs one fixed shape across iterations;"
                        f" give the dim a capacity instead of growing it."
                    )
            return tuple(_carry_like(flat_new[i], flat_initial[i]) for i in var_idxs)

        init = tuple(_init_carry(flat_initial[i]) for i in var_idxs)
        # One device for the whole carry, as lax requires.
        # The loop counter is on CPU by design, for the host-driven eager loop;
        # in the graph loop the control flow lives on the device with everything else.
        dev = None
        for raw in tree.flatten(init):
            if isinstance(raw, jax.core.Tracer):
                dev = None
                break
            d = next(iter(raw.devices()), None)
            if d is not None and d.platform != "cpu":
                dev = d
                break
        if dev is not None:
            init = tree.map_structure(partial(_to_device, dev=dev), init)
        # Inside lax.while_loop every shape is static and every value is a tracer,
        # so we must take a dim's capacity, not its dyn size.
        with rf.set_static_traceable_ctx(True):
            final = jax.lax.while_loop(_cond, _body, init)
        return _rebuild(final)

    @classmethod
    def tensor_array_create(cls, *, capacity: Optional[int] = None) -> TensorArrayType:
        """
        :param capacity: if given, the array can travel through the graph loop
        :return: empty TensorArray
        """
        if capacity is None:
            return []  # eager: a Python list, grows on demand
        return _BufferTensorArray(capacity=capacity)

    @staticmethod
    def tensor_array_push_back(tensor_array: TensorArrayType, tensor: Tensor) -> TensorArrayType:
        """push_back"""
        if isinstance(tensor_array, list):
            return tensor_array + [tensor]
        # the buffer has the template's dim order, the pushed tensor need not
        template = tensor_array.template if tensor_array.template is not None else tensor.copy_template()
        raw = tensor.copy_transpose(template.dims).raw_tensor
        buffer = tensor_array.buffer
        if buffer is None:
            # the entry shape is only known at the first write
            buffer = jnp.zeros((tensor_array.capacity,) + raw.shape, dtype=raw.dtype)
        buffer = jax.lax.dynamic_update_index_in_dim(buffer, raw, tensor_array.index, axis=0)
        return _BufferTensorArray(
            capacity=tensor_array.capacity,
            buffer=buffer,
            index=tensor_array.index + 1,
            template=template,
        )

    @staticmethod
    def tensor_array_get_item(tensor_array: TensorArrayType, index: Union[int, Tensor]) -> Tensor:
        """get_item"""
        if isinstance(tensor_array, list):
            if isinstance(index, int):
                return tensor_array[index]
            if not isinstance(index.raw_tensor, jax.core.Tracer):
                return tensor_array[int(index.raw_tensor)]
            # Traced index, e.g. the loop counter of the graph loop reading an unstacked array.
            # The list has to become one array to be indexed in the graph;
            # its entries are loop invariant, so XLA hoists the stack out of the loop.
            assert tensor_array, "TensorArray: get_item on an empty array"
            template = tensor_array[0].copy_template()
            stacked = jnp.stack([t.copy_transpose(template.dims).raw_tensor for t in tensor_array], axis=0)
            out = template.copy_template()
            out.raw_tensor = jax.lax.dynamic_index_in_dim(stacked, index.raw_tensor, axis=0, keepdims=False)
            return out
        assert tensor_array.buffer is not None, "TensorArray: get_item before any push_back"
        idx = index if isinstance(index, int) else index.raw_tensor
        raw = jax.lax.dynamic_index_in_dim(tensor_array.buffer, idx, axis=0, keepdims=False)
        out = tensor_array.template.copy_template()
        out.raw_tensor = raw
        return out

    @staticmethod
    def tensor_array_unstack(tensor: Tensor, *, axis: Dim) -> TensorArrayType:
        """unstack"""
        axis_int = tensor.get_axis_from_description(axis)
        template = tensor.copy_template().copy_template_excluding_axis(axis_int)
        out = []
        for i in range(tensor.raw_tensor.shape[axis_int]):
            entry = template.copy_template()
            entry.raw_tensor = jax.lax.index_in_dim(tensor.raw_tensor, i, axis=axis_int, keepdims=False)
            out.append(entry)
        return out

    @staticmethod
    def tensor_array_stack(tensor_array: TensorArrayType, *, axis: Dim, tensor_template: Tensor) -> Tensor:
        """stack"""
        if isinstance(tensor_array, _BufferTensorArray):
            # already contiguous along axis 0, which is what the buffer was allocated for
            assert tensor_array.buffer is not None, "TensorArray: stack before any push_back"
            template = tensor_array.template if tensor_array.template is not None else tensor_template
            out = template.copy_template().copy_add_dim_by_tag(axis, unbroadcast=True, axis=0)
            out.raw_tensor = tensor_array.buffer
            return out
        if tensor_array:
            # the stored tensors carry the better template (dim order),
            # and TensorArray already checked that they are compatible
            tensor_template = tensor_array[0].copy_template()
        out = tensor_template.copy_add_dim_by_tag(axis, unbroadcast=True, axis=0)
        if not tensor_array:
            return rf.zeros_like(out)
        out.raw_tensor = jnp.stack(
            [tensor.copy_transpose(tensor_template.dims).raw_tensor for tensor in tensor_array], axis=0
        )
        return out

    @staticmethod
    def full(
        dims: Sequence[Dim],
        fill_value: Union[RawTensorTypes, Tensor],
        *,
        dtype: str,
        device: Optional[str] = None,
        sparse_dim: Optional[Dim] = None,
        feature_dim: Optional[Dim] = None,
    ) -> Tensor:
        """full"""
        shape = tuple(dim.get_dim_value() for dim in dims)
        if isinstance(fill_value, Tensor):
            fill_value = fill_value.raw_tensor
        raw = jnp.full(shape, fill_value, dtype=JaxBackend.as_dtype_raw(dtype))
        device = device or rf.get_default_device()
        if device:
            raw = jax.device_put(raw, _device_from_str(device))
        return Tensor("full", dims=dims, sparse_dim=sparse_dim, feature_dim=feature_dim, dtype=dtype, raw_tensor=raw)

    @staticmethod
    def slice(
        source: Tensor,
        *,
        axis: Dim,
        start: Optional[Union[int, Tensor]] = None,
        end: Optional[Union[int, Tensor]] = None,
        step: Optional[Union[int, Tensor]] = None,
        size: Optional[Union[int, Tensor, Dim]] = None,
        out_dim: Dim,
    ) -> Tensor:
        """slice"""
        assert step is None or (isinstance(step, int) and step == 1), "RF JaxBackend: slice step != 1 not implemented"
        axis_int = source.get_axis_from_description(axis, allow_int=False)
        out = source.copy_template_replace_dim_tag(axis=axis_int, new_dim_tag=out_dim)
        if isinstance(start, Tensor):
            assert start.dims == ()
            start = start.raw_tensor
        elif start is None:
            start = 0
        # The size must be static: JAX has no data-dependent shapes.
        # The raw shape always is static, so prefer it over Dim.get_dim_value(),
        # which is a traced value for a dynamic dim (its max over the seq lens).
        axis_len = source.raw_tensor.shape[axis_int]
        if isinstance(size, Dim):
            assert end is None
            size = _static_size(size)
        elif isinstance(size, Tensor):
            assert end is None and size.dims == (), f"RF JaxBackend: slice size {size} must be scalar and static"
            size = int(size.raw_tensor)
        elif size is None:
            if isinstance(end, Tensor):
                assert end.dims == (), f"RF JaxBackend: slice end {end} must be scalar and static"
                end = int(end.raw_tensor)
            elif isinstance(end, int):
                if end < 0:
                    end += axis_len
            elif end is None:
                end = axis_len
            else:
                raise TypeError(f"slice: unsupported type for end: {type(end)}")
            assert not isinstance(start, jax.Array), (
                f"RF JaxBackend: slice with a device-side start needs an explicit static size, got {start}"
            )
            size = end - start
        elif not isinstance(size, int):
            raise TypeError(f"slice: unsupported type for size: {type(size)}")
        out.raw_tensor = jax.lax.dynamic_slice_in_dim(source.raw_tensor, start, int(size), axis=axis_int)
        return out

    @staticmethod
    def flip_no_mask(source: Tensor, *, axis: Dim) -> Tensor:
        """flip, ignoring masking"""
        axis_int = source.get_axis_from_description(axis, allow_int=False)
        out = source.copy_template("flip")
        out.raw_tensor = jnp.flip(source.raw_tensor, axis=axis_int)
        return out

    @staticmethod
    def cumsum(source: Tensor, *, spatial_dim: Dim) -> Tensor:
        """cumsum"""
        axis = source.get_axis_from_description(spatial_dim)
        out = source.copy_template("cumsum")
        out.raw_tensor = jnp.cumsum(source.raw_tensor, axis=axis, dtype=source.raw_tensor.dtype)
        return out

    @staticmethod
    def search_sorted(
        sorted_seq: Tensor, values: Tensor, *, axis: Dim, side: str = "left", out_dtype: str = "int32"
    ) -> Tensor:
        """search sorted"""
        if out_dtype not in ("int32", "int64"):
            raise NotImplementedError(f"search_sorted: out_dtype {out_dtype} not supported")
        if axis not in sorted_seq.dims:
            raise ValueError(f"search_sorted: axis {axis} not in sorted_seq {sorted_seq}")
        if axis.need_masking() and axis.dyn_size_ext is not None and axis.dyn_size_ext.batch_ndim > 0:
            # A per-batch-entry length would mean a different sorted sequence per entry,
            # which this does not handle.
            # A scalar length is fine: the tail entries are junk,
            # and every caller here treats the result there as meaningless (see _dev_seq_local).
            # need_masking() alone cannot decide this:
            # in the bound regime it is True for any dynamic dim.
            raise NotImplementedError(f"search_sorted: per-entry dynamic axis {axis} not supported")
        sorted_seq_dims = [dim for dim in sorted_seq.dims if dim != axis] + [axis]
        for dim in sorted_seq_dims[:-1]:
            if dim not in values.dims:
                raise ValueError(f"search_sorted: dim {dim} in sorted_seq {sorted_seq} but not in values {values}")
        values_rem_dims = [dim for dim in values.dims if dim not in sorted_seq_dims[:-1]]
        values_dims = sorted_seq_dims[:-1] + values_rem_dims
        sorted_seq_raw = sorted_seq.copy_compatible_to_dims_raw(sorted_seq_dims)
        values_raw = values.copy_compatible_to_dims_raw(values_dims)
        if len(values_rem_dims) != 1:
            values_raw = values_raw.reshape(values_raw.shape[: len(sorted_seq_dims[:-1])] + (-1,))
        out = Tensor("search_sorted", dims=sorted_seq_dims[:-1] + values_rem_dims, dtype=out_dtype, sparse_dim=axis)
        # jnp.searchsorted takes a 1-D sequence, so the shared dims are vmapped over,
        # where torch.searchsorted batches them itself
        search = partial(jnp.searchsorted, side=side)
        for _ in sorted_seq_dims[:-1]:
            search = jax.vmap(search, in_axes=(0, 0))
        out_raw = search(sorted_seq_raw, values_raw)
        out_raw = out_raw.astype(jnp.int32 if out_dtype == "int32" else jnp.int64)
        if len(values_rem_dims) != 1:
            out_raw = out_raw.reshape([dim.get_dim_value() for dim in out.dims])
        out.raw_tensor = out_raw
        return out

    @staticmethod
    def is_finite(x: Tensor) -> Tensor:
        """is finite"""
        out = x.copy_template("is_finite", dtype="bool")
        out.raw_tensor = jnp.isfinite(x.raw_tensor)
        return out

    @staticmethod
    def is_infinite(x: Tensor) -> Tensor:
        """is positive or negative infinite"""
        out = x.copy_template("is_infinite", dtype="bool")
        out.raw_tensor = jnp.isinf(x.raw_tensor)
        return out

    @staticmethod
    def is_neg_infinite(x: Tensor) -> Tensor:
        """is negative infinite"""
        out = x.copy_template("is_neg_infinite", dtype="bool")
        out.raw_tensor = jnp.isneginf(x.raw_tensor)
        return out

    @staticmethod
    def clip_by_value(
        x: Tensor,
        clip_value_min: Union[Tensor, rf.RawTensorTypes],
        clip_value_max: Union[Tensor, rf.RawTensorTypes],
        *,
        allow_broadcast_all_sources: bool = False,
    ) -> Tensor:
        """clip by value"""
        clip_value_min = rf.convert_to_tensor(clip_value_min, _backend=JaxBackend, device=x.device)
        clip_value_max = rf.convert_to_tensor(clip_value_max, _backend=JaxBackend, device=x.device)
        out = Tensor.get_common_data(
            [x, clip_value_min, clip_value_max],
            allow_broadcast_all_sources=allow_broadcast_all_sources,
            name="clip_by_value",
        )
        out.dtype = x.dtype
        out.sparse_dim = x.sparse_dim
        out.feature_dim = x.feature_dim
        out.raw_tensor = jnp.clip(
            x.copy_compatible_to_dims_raw(out.dims),
            clip_value_min.copy_compatible_to_dims_raw(out.dims),
            clip_value_max.copy_compatible_to_dims_raw(out.dims),
        )
        return out

    @staticmethod
    def lerp(
        start: Tensor, end: Tensor, weight: Union[float, Tensor], *, allow_broadcast_all_sources: bool = False
    ) -> Tensor:
        """lerp"""
        weight = rf.convert_to_tensor(weight, _backend=JaxBackend, device=start.device)
        out = Tensor.get_common_data(
            [start, end, weight], allow_broadcast_all_sources=allow_broadcast_all_sources, name="lerp"
        )
        start_raw = start.copy_compatible_to_dims_raw(out.dims)
        end_raw = end.copy_compatible_to_dims_raw(out.dims)
        weight_raw = weight.copy_compatible_to_dims_raw(out.dims)
        out.raw_tensor = start_raw + weight_raw * (end_raw - start_raw)
        return out

    @staticmethod
    def have_edit_distance() -> bool:
        """whether edit distance is available"""
        return True

    @staticmethod
    def edit_distance(a: Tensor, a_spatial_dim: Dim, b: Tensor, b_spatial_dim: Dim) -> Tensor:
        """edit distance"""
        a_batch_dims = a.remaining_dims(a_spatial_dim)
        b_batch_dims = b.remaining_dims(b_spatial_dim)
        batch_dims = a_batch_dims + [d for d in b_batch_dims if d not in a_batch_dims]
        a_raw = a.copy_compatible_to_dims_raw(batch_dims + [a_spatial_dim], unbroadcast=True)
        b_raw = b.copy_compatible_to_dims_raw(batch_dims + [b_spatial_dim], unbroadcast=True)
        a_seq_len = a_spatial_dim.dyn_size_ext.copy_compatible_to_dims_raw(batch_dims, unbroadcast=True)
        b_seq_len = b_spatial_dim.dyn_size_ext.copy_compatible_to_dims_raw(batch_dims, unbroadcast=True)
        batch_shape = [_static_size(d) for d in batch_dims]
        batch_n_elems = prod(batch_shape)
        a_raw = jnp.reshape(a_raw, (batch_n_elems, a_raw.shape[-1]))
        b_raw = jnp.reshape(b_raw, (batch_n_elems, b_raw.shape[-1]))
        dist_raw = _levenshtein(
            a_raw, b_raw, jnp.reshape(a_seq_len, (batch_n_elems,)), jnp.reshape(b_seq_len, (batch_n_elems,))
        )
        return rf.convert_to_tensor(jnp.reshape(dist_raw, batch_shape), name="edit_distance", dims=batch_dims)

    @staticmethod
    def masked_scatter(
        source: Tensor, backup: Optional[Tensor] = None, *, mask: Tensor, dims: Sequence[Dim], in_dim: Dim
    ) -> Tensor:
        """
        :param source: the values to place, over ``in_dim`` (+ remaining dims)
        :param backup: what the unselected positions get; zeros when not given
        :param mask: over ``dims``
        :param dims: the target layout
        :param in_dim: the packed axis of ``source``
        :return: ``dims`` (+ remaining dims), source scattered where the mask holds

        The inverse of :func:`masked_select`, and written the same way:
        the mask's prefix sum gives each target position its slot in ``source``.
        A gather, not a scatter, so it stays traceable and needs no data-dependent shape.
        """
        assert mask.dtype == "bool"
        assert set(mask.dims) == set(dims)
        assert in_dim in source.dims
        remaining_dims = [d for d in source.dims if d not in mask.dims and d != in_dim]
        source_raw = source.copy_compatible_to_dims_raw((in_dim,) + tuple(remaining_dims))
        rest_shape = source_raw.shape[1:]

        out_dims = tuple(dims) + tuple(remaining_dims)
        mask_raw = jnp.broadcast_to(
            mask.copy_compatible_to_dims_raw(tuple(dims)), tuple(d.get_dim_value() for d in dims)
        )
        mask_flat = jnp.reshape(mask_raw, (-1,))
        n_slots = source_raw.shape[0]
        # target position -> its slot in source; unselected positions are clamped and then discarded
        pos = jnp.cumsum(mask_flat.astype(jnp.int32)) - 1
        pos = jnp.clip(pos, 0, max(n_slots - 1, 0))
        gathered = jnp.take(source_raw, pos, axis=0)  # [prod(dims), ...rest]

        if backup is None:
            base = jnp.zeros(gathered.shape, dtype=gathered.dtype)
        else:
            b = backup
            for d in out_dims:
                if d not in b.dims:
                    b = rf.expand_dim(b, dim=d)
            base = jnp.reshape(b.copy_compatible_to_dims_raw(out_dims), (-1,) + tuple(rest_shape))
        sel = jnp.reshape(mask_flat, (-1,) + (1,) * len(rest_shape))
        out_raw = jnp.where(sel, gathered, base)
        out_raw = jnp.reshape(out_raw, tuple(d.get_dim_value() for d in dims) + tuple(rest_shape))
        return Tensor(
            "masked_scatter",
            dims=out_dims,
            dtype=source.dtype,
            sparse_dim=source.sparse_dim,
            feature_dim=source.feature_dim,
            raw_tensor=out_raw,
        )

    @staticmethod
    def masked_select(
        tensor: Tensor, *, mask: Tensor, dims: Sequence[Dim], out_dim: Optional[Dim] = None
    ) -> Tuple[Tensor, Dim]:
        """
        :param tensor:
        :param mask: over ``dims``
        :param dims: the order of these defines the packing order
        :param out_dim:
        :return: the selected elements, with ``dims`` replaced by one new dim, and that dim

        JAX shapes cannot depend on values, so under tracing the output gets a static size:
        the capacity of ``out_dim`` if it declares one, else the full mask size.
        Selected elements are packed at the front, zeros after,
        the same contract as ``masked_select_bound`` of the PyTorch graph-capture path.
        """
        assert mask.dtype == "bool"
        assert set(mask.dims) == set(dims)
        remaining_dims = [d for d in tensor.dims if d not in mask.dims]
        templ_dims = tuple(dims) + tuple(remaining_dims)
        full_shape = tuple(d.get_dim_value() for d in templ_dims)
        in_raw = jnp.broadcast_to(tensor.copy_compatible_to_dims_raw(templ_dims), full_shape)
        mask_raw = jnp.broadcast_to(
            mask.copy_compatible_to_dims_raw(tuple(dims)), tuple(d.get_dim_value() for d in dims)
        )
        rest_shape = full_shape[len(dims) :]
        in_flat = jnp.reshape(in_raw, (-1,) + rest_shape)
        mask_flat = jnp.reshape(mask_raw, (-1,))
        if not out_dim:
            out_dim = Dim(None, name="masked_select")
        if isinstance(mask_raw, jax.core.Tracer):
            bound = out_dim.capacity if out_dim.capacity is not None else mask_flat.shape[0]
            pos = jnp.cumsum(mask_flat.astype(jnp.int32)) - 1  # element -> its slot
            # masked-out elements, and selected ones beyond the bound, go to a dump slot, dropped below
            pos = jnp.where(mask_flat, jnp.minimum(pos, bound), bound)
            # gather-based select (out[slot] = in_flat[inv[slot]]) through the inverse permutation:
            # every index stays 1-D, and the gradient is the scatter-add of one gather
            inv = jnp.zeros((bound + 1,), dtype=jnp.int32).at[pos].set(jnp.arange(mask_flat.shape[0], dtype=jnp.int32))
            out_len = jnp.sum(mask_flat.astype(jnp.int32))
            out_raw = jnp.take(in_flat, inv[:bound], axis=0)
            # slots past the selected count point at stale inv entries: zero them, also so that
            # no gradient reaches the elements they happen to point at
            slot_valid = jnp.reshape(jnp.arange(bound) < out_len, (-1,) + (1,) * len(rest_shape))
            out_raw = jnp.where(slot_valid, out_raw, jnp.zeros((), dtype=out_raw.dtype))
            size_raw = out_len.astype(jnp.int64)
        else:
            (indices,) = jnp.nonzero(mask_flat)
            out_raw = jnp.take(in_flat, indices, axis=0)
            bound = int(out_raw.shape[0])
            size_raw = jnp.asarray(bound, dtype=jnp.int64)
        if out_dim.dyn_size_ext is None:
            out_dim.dyn_size_ext = Tensor("masked_select_size", dims=(), dtype="int64")
        if out_dim.dyn_size_ext.raw_tensor is None:
            out_dim.dyn_size_ext.raw_tensor = size_raw
        out_dim.capacity = bound
        return (
            Tensor(
                "masked_select",
                dims=(out_dim,) + tuple(remaining_dims),
                dtype=tensor.dtype,
                sparse_dim=tensor.sparse_dim,
                feature_dim=tensor.feature_dim,
                raw_tensor=out_raw,
            ),
            out_dim,
        )

    # --- signal / search

    # noinspection PyShadowingBuiltins
    @staticmethod
    def top_k(
        source: Tensor,
        *,
        axis: Union[Dim, Sequence[Dim]],
        k: Union[int, Tensor],
        k_dim: Optional[Dim] = None,
        sorted: bool = True,
    ) -> Tuple[Tensor, Union[Tensor, Sequence[Tensor]], Dim]:
        """top_k"""
        if not k_dim:
            k_dim = Dim(k, name="top-k-dim")
        axes = [axis] if isinstance(axis, Dim) else list(axis)
        if any(a.need_masking() for a in axes):
            # masked-out positions must never win, so push them to the smallest representable value
            mask_value = _dtype_min(source.raw_tensor.dtype)
            source = source.copy()
            for a in axes:
                if a.need_masking():
                    source = rf.where(a.get_mask(dim_order=source.dims, device=source.device), source, mask_value)
        k_value = _static_size(k_dim)

        if isinstance(axis, (list, tuple)):
            # flatten the axes into one, take top k there, then unravel the flat index per axis
            source = source.copy_transpose([d for d in source.dims if d not in axis] + list(axis))
            flat_shape = source.raw_tensor.shape[: source.batch_ndim - len(axis)] + (-1,)
            values_raw, indices_raw = jax.lax.top_k(jnp.reshape(source.raw_tensor, flat_shape), k_value)
            values = source.copy_template_new_dim_tags(
                new_dim_tags=source.dims[: -len(axis)] + (k_dim,), name="top_k_values"
            )
            if source.feature_dim and source.feature_dim in values.dims:
                values.feature_dim = source.feature_dim
            values.raw_tensor = values_raw
            indices_out = []
            for i, a in reversed(list(enumerate(axis))):
                indices_out_raw = indices_raw % a.dimension
                indices_raw = indices_raw // a.dimension
                indices = values.copy_template(name=f"top_k_indices_{a.name or i}")
                indices.feature_dim = None
                indices.dtype = JaxBackend.get_dtype_name_raw(indices_out_raw)
                indices.sparse_dim = a
                indices.raw_tensor = indices_out_raw
                indices_out.insert(0, indices)
            return values, indices_out, k_dim

        assert isinstance(axis, Dim)
        axis_int = source.get_axis_from_description(axis, allow_int=False)
        # jax.lax.top_k works on the last axis
        source = source.copy_move_axis(axis_int, -1)
        axis_int = source.batch_ndim - 1
        values_raw, indices_raw = jax.lax.top_k(source.raw_tensor, k_value)
        values = source.copy_template_replace_dim_tag(axis=axis_int, new_dim_tag=k_dim, name="top_k_values")
        values.raw_tensor = values_raw
        indices = source.copy_template_replace_dim_tag(axis=axis_int, new_dim_tag=k_dim, name="top_k_indices")
        indices.feature_dim = None
        indices.dtype = JaxBackend.get_dtype_name_raw(indices_raw)
        indices.sparse_dim = axis
        indices.raw_tensor = indices_raw
        return values, indices, k_dim

    @staticmethod
    def stft(
        x: Tensor,
        *,
        in_spatial_dim: Dim,
        frame_step: int,
        frame_length: int,
        fft_length: int,
        window_use_frame_length: bool = True,
        align_window_left: bool = True,
        window_enforce_even: bool = True,
        out_spatial_dim: Dim,
        out_dim: Dim,
    ) -> Tensor:
        """
        Short-time Fourier transform.

        Written out (frame -> window -> rfft) rather than calling a library STFT,
        whose conventions differ (window vs FFT length, periodic vs symmetric Hann).
        RF's semantics are the TF/SciPy ones, which the PyTorch backend also emulates.

        :param x:
        :param in_spatial_dim:
        :param frame_step:
        :param frame_length:
        :param fft_length:
        :param window_use_frame_length: window covers frame_length, not fft_length (the TF/SciPy convention)
        :param align_window_left: a shorter window sits at the left of the frame (TF/SciPy), not centered (librosa)
        :param window_enforce_even:
        :param out_spatial_dim:
        :param out_dim:
        :return: [batch_dims..., out_dim, out_spatial_dim], complex
        """
        batch_dims = [d for d in x.dims if d != in_spatial_dim]
        x = x.copy_transpose(batch_dims + [in_spatial_dim])
        x_raw = jnp.reshape(x.raw_tensor, (-1, x.raw_tensor.shape[-1]))

        if frame_length < fft_length and window_use_frame_length:
            # TF/SciPy window the frame_length, PyTorch/librosa the fft_length.
            # Padding the difference to the right makes the frame count match the TF convention.
            x_raw = jnp.pad(x_raw, ((0, 0), (0, fft_length - frame_length)))
        if frame_length > x_raw.shape[1]:
            # no full frame fits
            y = Tensor("stft", dims=batch_dims + [out_dim, out_spatial_dim], feature_dim=out_dim, dtype="complex64")
            y.raw_tensor = jnp.zeros([_static_size(d) for d in y.dims], dtype=jnp.complex64)
            return y
        if window_enforce_even:
            frame_length -= frame_length % 2

        # torch.hann_window / tf are periodic by default; jnp.hanning is symmetric, hence the +1 and drop
        window = jnp.hanning(frame_length + 1)[:-1].astype(x_raw.dtype)
        if frame_length < fft_length:
            if align_window_left:
                window = jnp.pad(window, (0, fft_length - frame_length))
            else:
                pad_left = (fft_length - frame_length) // 2
                window = jnp.pad(window, (pad_left, fft_length - frame_length - pad_left))

        num_frames = 1 + (x_raw.shape[1] - fft_length) // frame_step
        frame_idx = jnp.arange(num_frames)[:, None] * frame_step + jnp.arange(fft_length)[None, :]
        frames = x_raw[:, frame_idx] * window  # [B', frames, fft_length]
        y_raw = jnp.fft.rfft(frames, n=fft_length, axis=-1)  # [B', frames, freq]
        y_raw = jnp.swapaxes(y_raw, -1, -2)  # [B', freq, frames]
        y = Tensor("stft", dims=batch_dims + [out_dim, out_spatial_dim], dtype=JaxBackend.get_dtype_name_raw(y_raw))
        y.feature_dim = out_dim
        y.raw_tensor = jnp.reshape(y_raw, [_static_size(d) for d in y.dims])
        return y

    # --- losses

    @staticmethod
    def softmax_cross_entropy_with_logits(*, logits: Tensor, targets: Tensor, axis: Dim):
        """
        Efficient cross entropy.

        Written in RF ops rather than a fused kernel:
        XLA fuses the log_softmax with the gather / weighted sum anyway,
        so there is nothing to win from a special case here.

        :param logits: unnormalized scores over axis
        :param targets: probabilities over axis, or sparse indices into axis
        :param targets: class labels dim over which softmax is computed
        :param axis:
        :return: cross entropy, same dims as logits but without axis
        """
        assert axis in logits.dims, "Specified axis not present in logits."
        log_probs = rf.log_softmax(logits, axis=axis)
        if targets.sparse_dim:
            assert targets.sparse_dim == axis, (
                f"softmax_cross_entropy_with_logits: targets sparse dim {targets.sparse_dim} != axis {axis}"
            )
            return -rf.gather(log_probs, indices=targets, axis=axis)
        assert axis in targets.dims, "Specified axis not present in targets."
        return -rf.reduce_sum(targets * log_probs, axis=axis)

    @staticmethod
    def sdpa_varlen_raw(
        *,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        seq_starts_q: jax.Array,
        seq_lens_q: jax.Array,
        seq_starts_kv: jax.Array,
        seq_lens_kv: jax.Array,
        max_len_q: int,
        max_len_kv: int,
        is_causal: bool,
        dropout_p: float,
        scale: float,
    ) -> Optional[jax.Array]:
        """Packed varlen attention on the Triton kernels, see :func:`Backend.sdpa_varlen_raw`"""
        import jax

        if jax.devices()[0].platform != "gpu":
            return None  # the kernels are CUDA-only
        try:
            from returnn.jax.util.sdpa_varlen_triton import sdpa_varlen
        except ImportError:
            return None  # no jax-triton in this env
        return sdpa_varlen(
            query,
            key,
            value,
            seq_starts_q,
            seq_lens_q,
            seq_starts_kv,
            seq_lens_kv,
            max_len_q,
            max_len_kv,
            is_causal,
            dropout_p,
            0,
            scale,
        )

    @staticmethod
    def ctc_loss_packed_raw(**kwargs):
        """CTC loss on a packed logits buffer, see :func:`Backend.ctc_loss_packed_raw`"""
        from returnn.jax.util import native_op

        return native_op.ctc_loss_packed(**kwargs)

    @staticmethod
    def ctc_loss(
        *,
        logits: Tensor,
        logits_normalized: bool = False,
        targets: Tensor,
        input_spatial_dim: Dim,
        targets_spatial_dim: Dim,
        blank_index: int,
        max_approx: bool = False,
        use_native_op: Optional[bool] = None,
        label_loop: bool = True,
    ) -> Tensor:
        """
        CTC loss, via our fast-Baum-Welch native op or via optax; see ``use_native_op``.

        :param logits: [batch_dims..., input_spatial_dim, vocab]
        :param logits_normalized: whether logits are already log probs
            (both paths log_softmax internally, which is a no-op on normalized input)
        :param targets: [batch_dims..., targets_spatial_dim] -> vocab
        :param input_spatial_dim:
        :param targets_spatial_dim:
        :param blank_index:
        :param max_approx: not implemented
        :param use_native_op: our fast-Baum-Welch native op instead of optax's DP.
            Default (None) is the native op: faster (measured 2.1x on fwd+bwd),
            the only path that can do max_approx / label_loop=False,
            and it avoids the f64 promotion optax's jax.nn.one_hot causes under x64.
        :param label_loop: only the standard label loop
        :return: loss [batch_dims...], summed over time, not normalized
        """
        if use_native_op is None:
            use_native_op = True
        if max_approx:
            raise NotImplementedError("RF JaxBackend: ctc_loss max_approx not implemented")
        if not label_loop:
            raise NotImplementedError("RF JaxBackend: ctc_loss label_loop=False not implemented")
        assert targets.sparse_dim and targets.sparse_dim.dimension <= logits.feature_dim.dimension
        logits = rf.amp_cast_float32(logits)  # mixed precision: losses are computed in float32
        batch_dims = logits.remaining_dims((input_spatial_dim, logits.feature_dim))
        batch_dims_targets = targets.remaining_dims(targets_spatial_dim)
        if set(batch_dims) != set(batch_dims_targets):
            logits = rf.expand_dims(logits, [d for d in batch_dims_targets if d not in batch_dims])
            targets = rf.expand_dims(targets, [d for d in batch_dims if d not in batch_dims_targets])
            batch_dims = logits.remaining_dims((input_spatial_dim, logits.feature_dim))
        batch_shape = [d.get_dim_value() for d in batch_dims]
        batch_n_elems = prod(batch_shape)

        # optax wants [B, T, C] / [B, S], with 1.0 marking padded frames
        logits_raw = logits.copy_compatible_to_dims_raw(batch_dims + [input_spatial_dim, logits.feature_dim])
        logits_raw = jnp.reshape(logits_raw, (batch_n_elems,) + logits_raw.shape[-2:])
        targets_raw = targets.copy_compatible_to_dims_raw(batch_dims + [targets_spatial_dim])
        targets_raw = jnp.reshape(
            jnp.broadcast_to(targets_raw, tuple(batch_shape) + (targets_spatial_dim.get_dim_value(),)),
            (batch_n_elems, targets_spatial_dim.get_dim_value()),
        )
        input_lengths = jnp.reshape(
            jnp.broadcast_to(input_spatial_dim.dyn_size_ext.copy_compatible_to_dims_raw(batch_dims), batch_shape),
            (batch_n_elems,),
        )
        target_lengths = jnp.reshape(
            jnp.broadcast_to(targets_spatial_dim.dyn_size_ext.copy_compatible_to_dims_raw(batch_dims), batch_shape),
            (batch_n_elems,),
        )
        logit_paddings = (jnp.arange(logits_raw.shape[1])[None, :] >= input_lengths[:, None]).astype(logits_raw.dtype)
        label_paddings = (jnp.arange(targets_raw.shape[1])[None, :] >= target_lengths[:, None]).astype(logits_raw.dtype)
        if use_native_op:
            # The fast-Baum-Welch native op, the same kernels the TF and PyTorch backends use.
            # It does not go through jax.nn.one_hot, so the f64 promotion described below
            # cannot arise either.
            from returnn.jax.util.native_op import ctc_loss as _native_ctc_loss

            loss_raw = _native_ctc_loss(
                logits=logits_raw,
                logits_seq_lens=input_lengths,
                targets=targets_raw.astype(jnp.int32),
                targets_seq_lens=target_lengths,
                blank_index=blank_index,
            )
        else:
            # x64 is on for this backend (int64 seq lens need it),
            # and optax's ctc_loss builds its label one-hot with jax.nn.one_hot,
            # whose default dtype is jnp.float_ = float64 under x64.
            # The einsum against the [B,T,vocab] log-probs then runs as an f64 GEMM:
            # measured as the largest single op of the step
            # (~20% of device time, plus ~7% of f32->f64 converts).
            # Tracing with x64 off keeps jnp.float_ at f32;
            # the inputs are already f32/int32, so nothing else changes.
            loss_raw = None
            with _x64_disabled():
                loss_raw = optax.ctc_loss(
                    logits=logits_raw,
                    logit_paddings=logit_paddings,
                    labels=targets_raw.astype(jnp.int32),
                    label_paddings=label_paddings,
                    blank_id=blank_index,
                )
        return Tensor(
            name="ctc_loss",
            dims=batch_dims,
            raw_tensor=jnp.reshape(loss_raw, batch_shape),
            dtype=JaxBackend.get_dtype_name_raw(loss_raw),
        )

    # --- normalization

    @staticmethod
    def batch_norm(
        source: _TT,
        *,
        in_dim: Union[Dim, Sequence[Dim]],
        running_mean: Optional[Tensor],
        running_variance: Optional[Tensor],
        gamma: Optional[Tensor],
        beta: Optional[Tensor],
        epsilon: float,
        momentum: float,
        affine: bool,
        use_mask: bool,
    ) -> _TT:
        """batch norm

        The unmasked path; :class:`rf.BatchNorm` handles masking itself.
        Written out so that the detail deciding parity with PyTorch is visible:
        the biased variance normalizes, the unbiased one goes into the running estimate.
        """
        if use_mask:
            raise NotImplementedError("batch_norm with masking not implemented")
        if (running_mean is None) != (running_variance is None):
            raise ValueError("running_mean and running_variance must be both None or both not None")
        assert isinstance(in_dim, Dim)  # multiple dims not supported yet
        if affine:
            if gamma is None or beta is None:
                raise ValueError("gamma and beta must be given if affine=True")
            if not gamma.dims == beta.dims == (in_dim,):
                raise ValueError(f"gamma and beta must have shape [{in_dim}], got gamma {gamma} and beta {beta}")
        if running_mean is not None and not running_mean.dims == running_variance.dims == (in_dim,):
            raise ValueError(
                f"running_mean and running_variance must have shape [{in_dim}],"
                f" got running_mean {running_mean} and running_variance {running_variance}"
            )
        feat_axis = source.get_axis_from_description(in_dim)
        x = source.raw_tensor
        reduce_axes = tuple(i for i in range(x.ndim) if i != feat_axis)
        # broadcast shape of the per-feature statistics, to apply them to x
        stats_shape = tuple(x.shape[i] if i == feat_axis else 1 for i in range(x.ndim))
        train_flag = rf.get_run_ctx().is_train_flag_enabled(func=rf.BatchNorm.__call__)
        use_current_batch_stats = train_flag or running_mean is None
        if use_current_batch_stats:
            mean = jnp.mean(x, axis=reduce_axes)
            variance = jnp.mean(jnp.square(x - jnp.reshape(mean, stats_shape)), axis=reduce_axes)
            if running_mean is not None:
                count = x.size // x.shape[feat_axis]
                # as torch: the running variance tracks the unbiased estimate
                unbiased_variance = variance * (count / (count - 1)) if count > 1 else variance
                for param, value in ((running_mean, mean), (running_variance, unbiased_variance)):
                    new = param.raw_tensor * (1.0 - momentum) + value.astype(param.raw_tensor.dtype) * momentum
                    param.assign(Tensor(param.name, dims=param.dims, dtype=param.dtype, raw_tensor=new))
        else:
            mean, variance = running_mean.raw_tensor, running_variance.raw_tensor
        out_raw = (x - jnp.reshape(mean, stats_shape)) * jax.lax.rsqrt(jnp.reshape(variance, stats_shape) + epsilon)
        if affine:
            out_raw = out_raw * jnp.reshape(gamma.raw_tensor, stats_shape) + jnp.reshape(beta.raw_tensor, stats_shape)
        out = source.copy_template()
        out.raw_tensor = out_raw.astype(x.dtype)
        out.feature_dim = in_dim
        return out

    # --- convolution

    # noinspection PyShadowingBuiltins
    @staticmethod
    def conv(
        source: Tensor,
        *,
        in_dim: Dim,
        out_dim: Dim,
        in_spatial_dims: Sequence[Dim],
        out_spatial_dims: Optional[Sequence[Dim]] = None,
        filter: Tensor,
        filter_size: Sequence[Dim],
        padding: Union[str, int, Sequence[int]],
        strides: Optional[Union[int, Sequence[int]]] = None,
        dilation_rate: Optional[Union[int, Sequence[int]]] = None,
        groups: Optional[int] = None,
        bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Sequence[Dim]]:
        """conv"""
        if not out_spatial_dims:
            out_spatial_dims = rf.make_conv_out_spatial_dims(
                in_spatial_dims=in_spatial_dims,
                filter_size=filter_size,
                strides=strides or 1,
                dilation_rate=dilation_rate or 1,
                padding=padding,
            )
        n_spatial = len(filter_size)
        # mixed precision, as for matmul: the conv itself runs in the reduced dtype, the bias follows
        source, filter, bias = rf.amp_cast_compute(source, filter, bias)
        filter_in_dim = in_dim if not groups or groups == 1 else in_dim // groups
        filter = filter.copy_transpose((out_dim, filter_in_dim) + tuple(filter_size))
        batch_dims = [d for d in source.dims if d not in (in_dim,) + tuple(in_spatial_dims)]
        # conv_general_dilated takes the layout as dimension_numbers,
        # so with a single batch dim the input can be fed where it lies,
        # instead of being transposed into (N, C, *spatial) first
        # -- that transpose was a full materialised copy of the input on every call.
        # The output is still produced as (N, C, *spatial),
        # which is the order this returns anyway.
        # Several batch dims still need the merge, hence the transpose, so that path is unchanged.
        _src_dims = list(source.dims)
        _strides_seq = _to_seq(strides or 1, n_spatial)
        _dilation_seq = _to_seq(dilation_rate or 1, n_spatial)
        # A 1-D depthwise conv has no cuDNN kernel,
        # so XLA routes it through the 2-D implicit-GEMM path,
        # whose layout transforms cost more than the arithmetic.
        # See _conv_depthwise_1d.
        if (
            n_spatial == 1
            and groups
            and groups == in_dim.dimension == out_dim.dimension
            and _strides_seq[0] == 1
            and _dilation_seq[0] == 1
            # any number of batch dims, including none: the packed layout is just (time, feature).
            # Distinctness is what matters, so the axis lookups below are unambiguous.
            and len(set(_src_dims)) == len(_src_dims)
        ):
            return _conv_depthwise_1d(
                source,
                in_dim=in_dim,
                out_dim=out_dim,
                in_spatial_dim=in_spatial_dims[0],
                out_spatial_dim=out_spatial_dims[0],
                filter=filter,
                filter_size=filter_size[0],
                padding=padding,
                bias=bias,
            )
        if len(batch_dims) == 1 and len(set(_src_dims)) == len(_src_dims):
            dim_numbers = jax.lax.ConvDimensionNumbers(
                lhs_spec=(_src_dims.index(batch_dims[0]), _src_dims.index(in_dim))
                + tuple(_src_dims.index(d) for d in in_spatial_dims),
                rhs_spec=tuple(range(n_spatial + 2)),  # filter is (out, in, *spatial) from above
                out_spec=tuple(range(n_spatial + 2)),  # (N, C, *spatial)
            )
            src_raw = source.raw_tensor
        else:
            dim_numbers = _conv_dim_numbers(n_spatial)
            source = source.copy_transpose(batch_dims + [in_dim] + list(in_spatial_dims))
            src_raw = jnp.reshape(
                source.raw_tensor,
                [-1, in_dim.get_dim_value()] + [d.get_dim_value() for d in in_spatial_dims],
            )
        # JAX's "SAME" is the TF convention (out = ceil(in / stride), extra padding on the right),
        # which is what rf.make_conv_out_spatial_dims computes,
        # so unlike the PyTorch backend there is no need to emulate strided "same" padding by hand.
        out_raw = jax.lax.conv_general_dilated(
            src_raw,
            filter.raw_tensor,
            window_strides=_strides_seq,
            padding=_conv_padding(padding, n_spatial),
            rhs_dilation=_dilation_seq,
            dimension_numbers=dim_numbers,
            feature_group_count=groups or 1,
        )
        if bias is not None:
            out_raw = out_raw + jnp.reshape(bias.raw_tensor, (1, -1) + (1,) * n_spatial)
        out = Tensor(
            "conv",
            dims=batch_dims + [out_dim] + list(out_spatial_dims),
            dtype=JaxBackend.get_dtype_name_raw(out_raw),
        )
        out.raw_tensor = (
            out_raw if len(batch_dims) == 1 else jnp.reshape(out_raw, [d.get_dim_value() for d in out.dims])
        )
        out.feature_dim = out_dim
        return out, out_spatial_dims

    @staticmethod
    def pool(
        source: Tensor,
        *,
        mode: str,
        pool_size: Sequence[int],
        padding: Union[str, int, Sequence[int]] = "valid",
        dilation_rate: Union[Sequence[int], int] = 1,
        strides: Sequence[int],
        in_spatial_dims: Sequence[Dim],
        out_spatial_dims: Optional[Sequence[Dim]] = None,
    ) -> Tuple[Tensor, Sequence[Dim]]:
        """pool"""
        if out_spatial_dims is None:
            out_spatial_dims = rf.make_conv_out_spatial_dims(
                in_spatial_dims=in_spatial_dims,
                filter_size=pool_size,
                strides=strides,
                dilation_rate=dilation_rate,
                padding=padding,
            )
        n_spatial = len(in_spatial_dims)
        assert len(strides) == n_spatial == len(pool_size)
        in_spatial_dims = list(in_spatial_dims)
        dims = list(source.dims)
        assert len(set(in_spatial_dims)) == n_spatial and all(d in dims for d in in_spatial_dims), (
            f"RF JaxBackend pool: {in_spatial_dims} must be distinct dims of {source}"
        )
        # The window is built in place -- 1 on every non-spatial axis --
        # instead of transposing the spatial dims to the back first.
        # reduce_window does not care where they sit,
        # and the transpose was a full materialised copy of the input on every call:
        # measured at 1.59x on the whole ConformerConvSubsample, with bit-identical output.
        axes = [dims.index(d) for d in in_spatial_dims]
        rank = len(dims)
        window = [1] * rank
        window_strides = [1] * rank
        window_dilation = [1] * rank
        for _axis, _size, _stride, _dil in zip(axes, pool_size, strides, _to_seq(dilation_rate or 1, n_spatial)):
            window[_axis], window_strides[_axis], window_dilation[_axis] = _size, _stride, _dil
        window, window_strides, window_dilation = tuple(window), tuple(window_strides), tuple(window_dilation)
        pad = _conv_padding(padding, n_spatial)
        if not isinstance(pad, str):
            _full_pad = [(0, 0)] * rank
            for _axis, _pair in zip(axes, pad):
                _full_pad[_axis] = _pair
            pad = _full_pad
        src_raw = source.raw_tensor
        dtype = src_raw.dtype
        # Non-overlapping max pooling as reshape + max over a new axis.
        # reduce_window's gradient is select_and_scatter,
        # which rescans every window instead of using saved argmax indices.
        if (
            mode == "max"
            and all(d == 1 for d in window_dilation)
            and all(_s == _w for _s, _w in zip(strides, pool_size))
            # any, not all: the Conformer pools (1, 2), i.e. window 1 on time and 2 on mel
            and any(_w > 1 for _w in pool_size)
            and _pool_no_padding(pad, axes)
            and all(src_raw.shape[_a] is not None for _a in axes)
        ):
            out_raw = _pool_max_reshape(src_raw, axes, pool_size)
        elif mode == "max":
            # The init value must be the monoid identity (-inf), not just a very small number:
            # only then does JAX lower this to reduce_window_max, which has a transpose rule.
            # With finfo.min it builds a generic reduce_window, and the backward pass then fails with
            # "Linearization failed to produce known values for all output primals".
            init = -numpy.inf if jnp.issubdtype(dtype, jnp.floating) else _dtype_min(dtype)
            out_raw = jax.lax.reduce_window(
                src_raw, init, jax.lax.max, window, window_strides, pad, window_dilation=window_dilation
            )
        elif mode == "avg":
            assert all(d == 1 for d in window_dilation), "RF JaxBackend: dilation_rate only supported for max_pool"
            sums = jax.lax.reduce_window(src_raw, jnp.zeros((), dtype), jax.lax.add, window, window_strides, pad)
            # divide by the number of real frames per window, i.e. torch's count_include_pad=False
            counts = jax.lax.reduce_window(
                jnp.ones_like(src_raw), jnp.zeros((), dtype), jax.lax.add, window, window_strides, pad
            )
            out_raw = sums / counts
        else:
            raise NotImplementedError(f"RF JaxBackend: pool mode {mode!r} not implemented")
        # reduce_window kept the input's dim order;
        # the RF contract (and the cross-backend parity test) is batch dims first, spatial last,
        # so reorder here rather than on the input --
        # the pooled output is smaller than what it was pooled from, so this copies less.
        out_dims = [out_spatial_dims[in_spatial_dims.index(d)] if d in in_spatial_dims else d for d in dims]
        out = Tensor("pool", dims=out_dims, dtype=source.dtype)
        out.raw_tensor = out_raw
        if source.feature_dim and source.feature_dim in out.dims:
            out.feature_dim = source.feature_dim
        batch_dims = [d for d in out_dims if d not in out_spatial_dims]
        if out_dims != batch_dims + list(out_spatial_dims):
            out = out.copy_transpose(batch_dims + list(out_spatial_dims))
        return out, out_spatial_dims

    # --- random

    # JAX has no implicit global RNG: keys are values, and every draw must consume a fresh one.
    # So the global stream lives here, and each draw splits it.
    _rng_key: Optional[jax.Array] = None

    @staticmethod
    def set_random_seed(seed: int):
        """
        :param seed:
        """
        JaxBackend._rng_key = jax.random.key(seed)

    @staticmethod
    def get_random_state() -> Dict[str, bytes]:
        """
        :return: random state
        """
        return {"jax": numpy.asarray(jax.random.key_data(JaxBackend._get_rng_key_())).tobytes()}

    @staticmethod
    def set_random_state(state: Dict[str, bytes]):
        """
        :param state: as returned by :func:`get_random_state`
        """
        if "jax" not in state:
            return
        key_data = numpy.frombuffer(state["jax"], dtype=numpy.uint32)
        JaxBackend._rng_key = jax.random.wrap_key_data(jnp.asarray(key_data))

    @staticmethod
    def _get_rng_key_() -> jax.Array:
        """
        :return: the global key, seeded from entropy on first use (like the implicit global RNGs elsewhere)
        """
        if JaxBackend._rng_key is None:
            JaxBackend.set_random_seed(int(numpy.random.SeedSequence().generate_state(1)[0]))
        return JaxBackend._rng_key

    @staticmethod
    def _next_rng_key_() -> jax.Array:
        """
        :return: a fresh key, advancing the global stream
        """
        key, sub = jax.random.split(JaxBackend._get_rng_key_())
        JaxBackend._rng_key = key
        return sub

    @staticmethod
    def random(
        *,
        dims: Sequence[Dim],
        dtype: str,
        device: Optional[str] = None,
        sparse_dim: Optional[Dim] = None,
        feature_dim: Optional[Dim] = None,
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
        out: Optional[Tensor[jax.Array]] = None,
    ) -> Tensor:
        """
        random. See :func:`rf.random` for details.

        The drawn values do not match any other backend's for the same seed (different PRNG).
        Cross-backend comparisons therefore copy parameters over, they do not replay RNG streams.
        """
        assert explicit_state is None and auto_update_state is None, "RF JaxBackend: random state args not implemented"
        assert algorithm is None, f"RF JaxBackend: random algorithm {algorithm!r} not implemented"
        if out is None:
            out = Tensor(
                name=f"random_{distribution}", dims=dims, dtype=dtype, sparse_dim=sparse_dim, feature_dim=feature_dim
            )
        shape = tuple(d.get_dim_value() for d in dims)
        dtype_ = JaxBackend.as_dtype_raw(dtype)
        key = jax.random.key(seed) if static else JaxBackend._next_rng_key_()
        if static:
            assert seed is not None
        else:
            assert seed is None
        # The key decides where the values are drawn,
        # and JAX refuses a computation whose arguments live on different devices.
        # So the key and the bounds go to the requested device.
        # SpecAugment is the case that needs this:
        # it asks for the number of masks on the cpu
        # (to keep the data-dependent loop off the accelerator) while the data is on the gpu.
        target_device = _device_from_str(device) if device else None
        if target_device is not None:
            key = jax.device_put(key, target_device)

        def _arg(v, default):
            """
            :param v: a number, or a Tensor which may be per-element (SpecAugment draws per-seq bounds)
            :param default: when v is None
            :return: a number or a raw array broadcastable to the output shape,
                which is what jax.random accepts for these arguments
            """
            if v is None:
                return default
            if isinstance(v, Tensor):
                v = v.raw_tensor if not v.dims else v.copy_compatible_to_dims_raw(out.dims)
            if target_device is not None and isinstance(v, jax.Array):
                v = jax.device_put(v, target_device)
            return v

        if distribution == "uniform":
            assert mean is None and stddev is None
            minval_ = _arg(minval, 0)
            if jnp.issubdtype(dtype_, jnp.floating):
                raw = jax.random.uniform(key, shape, dtype=dtype_, minval=minval_, maxval=_arg(maxval, 1))
            else:
                assert maxval is not None, "maxval must be specified for integer random uniform"
                raw = jax.random.randint(key, shape, minval_, _arg(maxval, None), dtype=dtype_)
        elif distribution == "normal":
            assert minval is None and maxval is None
            mean_, stddev_ = _arg(mean, 0), _arg(stddev, 1)
            raw = mean_ + stddev_ * jax.random.normal(key, shape, dtype=dtype_)
        elif distribution == "truncated_normal":
            mean_, stddev_ = _arg(mean, 0), _arg(stddev, 1)
            minval_ = _arg(minval, mean_ - 2 * stddev_)
            maxval_ = _arg(maxval, mean_ + 2 * stddev_)
            # jax.random.truncated_normal bounds are in standard-normal units
            raw = mean_ + stddev_ * jax.random.truncated_normal(
                key, (minval_ - mean_) / stddev_, (maxval_ - mean_) / stddev_, shape, dtype=dtype_
            )
        else:
            raise NotImplementedError(f"RF JaxBackend: random distribution {distribution!r} not implemented")
        out.raw_tensor = jax.device_put(raw, _device_from_str(device)) if device else raw
        return out

    # --- parameters

    @staticmethod
    def create_parameter_raw(tensor: rf.Parameter, *, device: Optional[str] = None) -> jax.Array:
        """
        :return: parameter, zero-initialized (the initial value is set separately)
        """
        raw = jnp.zeros(tuple(d.get_dim_value() for d in tensor.dims), dtype=JaxBackend.as_dtype_raw(tensor.dtype))
        device = device or rf.get_default_device()
        return jax.device_put(raw, _device_from_str(device)) if device else raw

    @staticmethod
    def set_parameter_initial_value(param: rf.Parameter, value: Union[None, Tensor, rf.RawTensorTypes]) -> None:
        """
        :param param: parameter
        :param value: initial value
        """
        if value is None:
            value = 0
        if isinstance(value, Tensor):
            value_raw = value.copy_compatible_to_dims_raw(param.dims)
        else:
            value_raw = jnp.asarray(value)
        dtype = JaxBackend.as_dtype_raw(param.dtype)
        param.raw_tensor = jnp.broadcast_to(value_raw.astype(dtype), param.raw_tensor.shape)

    @staticmethod
    def set_parameter_trainable(param: rf.Parameter, trainable: bool) -> None:
        """
        set trainable.

        Records the resolved flag, available only here:
        ``rf.Parameter.trainable`` returns the value as given, so None wherever unspecified.
        Reading the property instead put rf.BatchNorm's running stats through the optimizer.
        """
        # noinspection PyUnresolvedReferences
        param.jax_trainable = trainable

    @staticmethod
    def parameter_assign(param: rf.Parameter, value: Tensor, *, op: str = "assign") -> None:
        """param assign"""
        value_raw = value.copy_compatible_to_dims_raw(param.dims)
        value_raw = value_raw.astype(JaxBackend.as_dtype_raw(param.dtype))
        if op == "assign":
            param.raw_tensor = jnp.broadcast_to(value_raw, param.raw_tensor.shape)
        elif op == "add":
            param.raw_tensor = param.raw_tensor + value_raw
        else:
            raise ValueError(f"Parameter {param} assign: Unsupported op: {op}")

    @staticmethod
    def parameter_move_to(param: rf.Parameter, *, device: Optional[str] = None, dtype: Optional[str] = None):
        """to"""
        raw = param.raw_tensor
        if dtype:
            raw = raw.astype(JaxBackend.as_dtype_raw(dtype))
        if device:
            raw = jax.device_put(raw, _device_from_str(device))
        param.raw_tensor = raw

    # --- gradients

    @staticmethod
    def stop_gradient(tensor: Tensor) -> Tensor:
        """stop grad"""
        out = tensor.copy()
        out.raw_tensor = jax.lax.stop_gradient(out.raw_tensor)
        return out

    @staticmethod
    def gradient(y: Tensor, x: Tensor) -> Tensor:
        """gradient"""
        raise NotImplementedError(
            "RF JaxBackend: rf.gradient(y, x) has no JAX equivalent."
            " JAX has no tape, so a gradient exists only for a FUNCTION,"
            " and by the time you hold y, the computation that produced it is gone."
            " Differentiate the step function instead (jax.grad / jax.value_and_grad),"
            " which is what the engine does."
        )

    @staticmethod
    def scaled_gradient(tensor: Tensor, scale: Union[float, Tensor]) -> Tensor:
        """scaled gradient"""
        out = tensor.copy()
        out.raw_tensor = _scale_grad(out.raw_tensor, _raw(scale, out.raw_tensor.dtype))
        return out

    @staticmethod
    def scaled_gradient_ext(
        x: Tensor,
        *,
        scale: Union[float, Tensor] = 1.0,
        shift: Optional[Union[float, Tensor]] = None,
        scale_shift_by_sum_over_axis: Optional[Dim] = None,
    ):
        """scaled gradient ext"""
        out = x.copy()
        dtype = out.raw_tensor.dtype
        if shift is None:
            out.raw_tensor = _scale_grad(out.raw_tensor, _raw(scale, dtype))
            return out
        axis = (
            x.get_axis_from_description(scale_shift_by_sum_over_axis, allow_int=False)
            if scale_shift_by_sum_over_axis is not None
            else None
        )
        out.raw_tensor = _scale_shift_grad(out.raw_tensor, _raw(scale, dtype), _raw(shift, dtype), axis)
        return out

    # --- math

    @staticmethod
    def matmul(a: _TT, b: _TT, *, reduce: Union[Dim, Sequence[Dim]], use_mask: bool = True) -> _TT:
        """
        batched matmul of a and b, see base class doc string.

        Works on axis indices, not on dim identity, because a dim can occur twice in one tensor
        (rf.Linear from a dim to itself). `get_axis_from_description` resolves which occurrence
        is the reduce axis; an einsum over dim-keyed letters would silently take a diagonal.
        """
        if isinstance(reduce, Dim):
            reduce = [reduce]
        if use_mask and any(dim.dyn_size_ext is not None for dim in reduce):
            raise NotImplementedError("RF JaxBackend: masking in matmul reduce not yet implemented")
        # mixed precision: this is the op AMP exists for. It also covers the parameters,
        # which stay float32 and are cast here, where they are used.
        a, b = rf.amp_cast_compute(a, b)
        a_dims, b_dims = a.dims, b.dims
        assert all(dim in a_dims + b_dims for dim in reduce), "Some reduce Dims not in a or b."

        if not all(dim in a_dims for dim in reduce) or not all(dim in b_dims for dim in reduce):
            # reduce dims not on both sides: a generic einsum handles it (no repeated dims in this case)
            result_dims = [dim for dim in a_dims if dim not in reduce]
            result_dims += [dim for dim in b_dims if dim not in reduce and dim not in a_dims]
            letters = {}
            for dim in a_dims + b_dims:
                if dim not in letters:
                    letters[dim] = chr(97 + len(letters))
            subscripts = "%s,%s->%s" % tuple(
                "".join(letters[dim] for dim in dims_) for dims_ in (a_dims, b_dims, result_dims)
            )
            raw_result = jnp.einsum(subscripts, a.raw_tensor, b.raw_tensor)
            return Tensor(
                "dot", dims=result_dims, raw_tensor=raw_result, dtype=JaxBackend.get_dtype_name_raw(raw_result)
            )

        if len(reduce) > 1:
            reduce = sorted(reduce, key=lambda dim: a_dims.index(dim))
        a_reduce_axes = [a.get_axis_from_description(dim) for dim in reduce]
        b_reduce_axes = [b.get_axis_from_description(dim) for dim in reduce]
        common_dims = [dim for i, dim in enumerate(a_dims) if dim in b_dims and i not in a_reduce_axes]
        a_common_axes = [a_dims.index(dim) for dim in common_dims]
        b_common_axes = [b_dims.index(dim) for dim in common_dims]
        a_unique_axes = [i for i in range(len(a_dims)) if i not in a_reduce_axes and i not in a_common_axes]
        b_unique_axes = [i for i in range(len(b_dims)) if i not in b_reduce_axes and i not in b_common_axes]

        a_shape, b_shape = a.raw_tensor.shape, b.raw_tensor.shape
        common_shape = tuple(a_shape[i] for i in a_common_axes)
        a_unique_shape = tuple(a_shape[i] for i in a_unique_axes)
        b_unique_shape = tuple(b_shape[i] for i in b_unique_axes)
        assert common_shape == tuple(b_shape[i] for i in b_common_axes), "common dims of a and b do not match"
        assert tuple(a_shape[i] for i in a_reduce_axes) == tuple(b_shape[i] for i in b_reduce_axes), (
            "reduce dims of a and b do not match"
        )

        a_raw = jnp.transpose(a.raw_tensor, a_common_axes + a_unique_axes + a_reduce_axes)
        b_raw = jnp.transpose(b.raw_tensor, b_common_axes + b_reduce_axes + b_unique_axes)
        reduce_total = prod(tuple(a_shape[i] for i in a_reduce_axes))
        raw_result = jnp.matmul(
            jnp.reshape(a_raw, (prod(common_shape), prod(a_unique_shape), reduce_total)),
            jnp.reshape(b_raw, (prod(common_shape), reduce_total, prod(b_unique_shape))),
        )
        raw_result = jnp.reshape(raw_result, common_shape + a_unique_shape + b_unique_shape)
        result_dims = common_dims + [a_dims[i] for i in a_unique_axes] + [b_dims[i] for i in b_unique_axes]
        return Tensor("dot", dims=result_dims, raw_tensor=raw_result, dtype=JaxBackend.get_dtype_name_raw(raw_result))

    @staticmethod
    def softmax(tensor: Tensor, *, axis: Dim, use_mask: bool = True) -> Tensor:
        """
        :param tensor:
        :param axis:
        :param use_mask:
        :return: softmax over axis
        """
        return _softmax(tensor, axis=axis, use_mask=use_mask, log=False)

    @staticmethod
    def log_softmax(tensor: Tensor, *, axis: Dim, use_mask: bool = True) -> Tensor:
        """
        :param tensor:
        :param axis:
        :param use_mask:
        :return: log_softmax over axis
        """
        return _softmax(tensor, axis=axis, use_mask=use_mask, log=True)


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


_PadModeMap = {"replicate": "edge", "circular": "wrap"}


def _to_seq(value: Union[int, Sequence[int]], n: int) -> Tuple[int, ...]:
    """
    :param value: a scalar to broadcast, or a sequence
    :param n: number of spatial dims
    :return: value as a tuple of length n
    """
    if isinstance(value, int):
        return (value,) * n
    value = tuple(value)
    assert len(value) == n, f"expected {n} values, got {value}"
    return value


def _conv_padding(padding: Union[str, int, Sequence[int]], n: int) -> Union[str, Sequence[Tuple[int, int]]]:
    """
    :param padding: "same" / "valid", one amount for all spatial dims, or one per spatial dim
    :param n: number of spatial dims
    :return: what jax.lax expects: "SAME" / "VALID", or explicit (low, high) per spatial dim
    """
    if isinstance(padding, str):
        assert padding.lower() in ("same", "valid"), f"invalid padding {padding!r}"
        return padding.upper()
    if isinstance(padding, int):
        return [(padding, padding)] * n
    out = [(p, p) if isinstance(p, int) else tuple(p) for p in padding]
    assert len(out) == n, f"expected {n} paddings, got {padding}"
    return out


def _conv_dim_numbers(n_spatial: int) -> Tuple[str, str, str]:
    """
    :param n_spatial: number of spatial dims
    :return: (lhs, rhs, out) layout labels for jax.lax.conv_general_dilated,
        matching the (N, C, *spatial) / (O, I, *filter) layout the RF conv builds
    """
    spatial = "HWDEFG"[:n_spatial]
    assert len(spatial) == n_spatial, f"conv with {n_spatial} spatial dims not supported"
    return "NC" + spatial, "OI" + spatial, "NC" + spatial


def _pool_no_padding(pad, axes: Sequence[int]) -> bool:
    """
    :param pad: what _conv_padding returned: "SAME" / "VALID", or explicit (low, high) per axis
    :param axes: the spatial axes
    :return: whether pooling adds no padding, so the windows tile the axis exactly
    """
    if isinstance(pad, str):
        return pad == "VALID"
    return all(pad[_a] == (0, 0) for _a in axes)


def _pool_max_reshape(src_raw, axes: Sequence[int], pool_size: Sequence[int]):
    """
    Non-overlapping max pool, as a reshape and a max over the split axis.

    Equivalent to reduce_window with window == strides and no padding,
    which drops any remainder; the slice below does the same.
    The point is the gradient:
    reducing over a plain axis lowers to a comparison-select, not to select_and_scatter.

    :param src_raw: the input
    :param axes: spatial axes to pool
    :param pool_size: window per spatial axis
    :return: the pooled output, same rank as the input
    """
    out = src_raw
    for _axis, _size in zip(axes, pool_size):
        n_out = out.shape[_axis] // _size
        if n_out * _size != out.shape[_axis]:
            out = jax.lax.slice_in_dim(out, 0, n_out * _size, axis=_axis)
        shape = list(out.shape)
        out = jnp.reshape(out, shape[:_axis] + [n_out, _size] + shape[_axis + 1 :])
        out = jnp.max(out, axis=_axis + 1)
    return out


def _conv_depthwise_1d(
    source: Tensor,
    *,
    in_dim: Dim,
    out_dim: Dim,
    in_spatial_dim: Dim,
    out_spatial_dim: Dim,
    filter: Tensor,
    filter_size: Dim,
    padding: Union[str, int, Sequence[int]],
    bias: Optional[Tensor],
) -> Tuple[Tensor, Sequence[Dim]]:
    """
    Depthwise 1-D conv, one filter per channel, stride 1 and no dilation,
    written as a weighted sum of shifted copies of the input.
    This keeps it inside XLA's own fusion,
    instead of cuDNN's 2-D grouped path with its layout transforms.

    :param source: with in_dim and in_spatial_dim
    :param in_dim: channel dim, equal to out_dim and to the group count
    :param out_dim:
    :param in_spatial_dim:
    :param out_spatial_dim:
    :param filter: already transposed to (out_dim, 1, filter_size)
    :param filter_size: kernel width
    :param padding:
    :param bias: over out_dim, or None
    :return: output in the source's own axis order, and (out_spatial_dim,)
    """
    src_raw = source.raw_tensor
    src_dims = list(source.dims)
    rank = len(src_dims)
    feat_ax, time_ax = src_dims.index(in_dim), src_dims.index(in_spatial_dim)
    width = filter_size.dimension
    pad = _conv_padding(padding, 1)
    if pad == "SAME":
        # the TF convention that rf.make_conv_out_spatial_dims assumes: the odd frame goes right
        pad = [((width - 1) // 2, width // 2)]
    elif pad == "VALID":
        pad = [(0, 0)]
    # Tiled Triton kernel where it applies:
    # the contiguous (time, channel) packed layout with "same" padding.
    # Every other layout falls through to the shifted-sum, which is general.
    if rank == 2 and time_ax == 0 and feat_ax == 1 and pad[0][0] + pad[0][1] == width - 1:
        from returnn.jax.util import depthwise_conv_triton

        filter_2d = jnp.transpose(filter.raw_tensor[:, 0, :])
        if depthwise_conv_triton.depthwise_conv1d_available(src_raw, filter_2d):
            out_raw = depthwise_conv_triton.depthwise_conv1d(src_raw, filter_2d, pad[0][0])
            if bias is not None:
                out_raw = out_raw + bias.raw_tensor
            out = Tensor(
                "conv",
                dims=[out_dim if d == in_dim else out_spatial_dim if d == in_spatial_dim else d for d in src_dims],
                dtype=JaxBackend.get_dtype_name_raw(out_raw),
            )
            out.raw_tensor = out_raw
            out.feature_dim = out_dim
            return out, (out_spatial_dim,)
    pad_width = [(0, 0)] * rank
    pad_width[time_ax] = pad[0]
    padded = jnp.pad(src_raw, pad_width)
    # from the padded shape rather than out_spatial_dim, which need not be static here
    out_shape = list(padded.shape)
    out_shape[time_ax] = padded.shape[time_ax] - width + 1
    weight_shape = [1] * rank
    weight_shape[feat_ax] = filter.raw_tensor.shape[0]
    # f32 accumulation, as cuDNN does: summing this many taps in bf16 would lose the low bits
    acc = jnp.zeros(out_shape, dtype=jnp.float32)
    for i in range(width):
        start = [0] * rank
        start[time_ax] = i
        shifted = jax.lax.dynamic_slice(padded, start, out_shape)
        weight = jnp.reshape(filter.raw_tensor[:, 0, i], weight_shape)
        acc += shifted.astype(jnp.float32) * weight.astype(jnp.float32)
    out_raw = acc.astype(src_raw.dtype)
    if bias is not None:
        out_raw = out_raw + jnp.reshape(bias.raw_tensor, weight_shape)
    out = Tensor(
        "conv",
        dims=[out_dim if d == in_dim else out_spatial_dim if d == in_spatial_dim else d for d in src_dims],
        dtype=JaxBackend.get_dtype_name_raw(out_raw),
    )
    out.raw_tensor = out_raw
    out.feature_dim = out_dim
    return out, (out_spatial_dim,)


def _dtype_min(dtype) -> Union[int, float]:
    """
    :param dtype:
    :return: smallest representable value, i.e. the neutral element of max
    """
    return jnp.finfo(dtype).min if jnp.issubdtype(dtype, jnp.floating) else jnp.iinfo(dtype).min


def _dtype_max(dtype) -> Union[int, float]:
    """
    :param dtype:
    :return: largest representable value, i.e. the neutral element of min
    """
    return jnp.finfo(dtype).max if jnp.issubdtype(dtype, jnp.floating) else jnp.iinfo(dtype).max


def _levenshtein(a: jax.Array, b: jax.Array, a_len: jax.Array, b_len: jax.Array) -> jax.Array:
    """
    Batched Levenshtein distance.

    A scan over the rows with an inner scan over the columns, as the recurrence is sequential
    in both directions: O(Ta * Tb) scan steps, fine for labels, not for raw audio lengths.
    (The PyTorch backend calls RETURNN's native op here instead.)

    :param a: [B, Ta] labels
    :param b: [B, Tb] labels
    :param a_len: [B]
    :param b_len: [B]
    :return: [B] distance, taken at each sequence's own lengths
    """
    n_batch, len_b = b.shape
    len_a = a.shape[1]
    dtype = jnp.int32
    a, b = a.astype(jnp.int32), b.astype(jnp.int32)
    a_len, b_len = a_len.astype(jnp.int32), b_len.astype(jnp.int32)

    # row i holds the distances of a[:i] to every prefix of b, so row 0 is just the number of insertions
    row0 = jnp.broadcast_to(jnp.arange(len_b + 1, dtype=dtype), (n_batch, len_b + 1))
    # for an empty a, the distance is the length of b
    res0 = jnp.where(a_len == 0, b_len, 0).astype(dtype)

    # noinspection PyShadowingNames
    def _row(carry, i):
        prev, res = carry
        a_i = a[:, i]

        def _cell(left, j):
            # left = the new row's cell j-1; prev holds the previous row
            cost = (a_i != b[:, j]).astype(dtype)
            val = jnp.minimum(jnp.minimum(prev[:, j + 1] + 1, left + 1), prev[:, j] + cost)
            return val, val

        first = jnp.full((n_batch,), i + 1, dtype=dtype)  # deleting the first i+1 symbols of a
        _, cells = jax.lax.scan(_cell, first, jnp.arange(len_b))
        new_row = jnp.concatenate([first[:, None], jnp.swapaxes(cells, 0, 1)], axis=1)
        # each sequence's answer is this row read at its own b_len, taken when the row reaches its own a_len
        at_b_len = jnp.take_along_axis(new_row, b_len[:, None], axis=1)[:, 0]
        res = jnp.where(i + 1 == a_len, at_b_len, res)
        return (new_row, res), None

    (_, res), _ = jax.lax.scan(_row, (row0, res0), jnp.arange(len_a))
    return res


def _static_size(dim: Dim) -> int:
    """
    :param dim:
    :return: the extent of the dim as a Python int, for use as a shape.

    JAX shapes can never depend on traced values.
    A static dim gives its size directly, a dynamic one its capacity if it declares one,
    and outside of tracing its concrete max is fine as well.
    """
    if dim.dimension is not None:
        return dim.dimension
    if dim.capacity is not None:
        return dim.capacity
    try:
        return int(dim.get_dim_value())
    except jax.errors.ConcretizationTypeError:
        raise AssertionError(
            f"RF JaxBackend: dynamic dim {dim} is used as a shape under tracing,"
            " so it needs a static capacity (Dim(..., capacity=...))"
        ) from None


def _pad_amount(amount: Union[Dim, int, Tensor], handle_dynamic_dims: bool) -> int:
    """
    :param amount: one side of one axis' padding
    :param handle_dynamic_dims:
    :return: the amount as a static int (JAX shapes cannot depend on data)
    """
    if isinstance(amount, Dim):
        if handle_dynamic_dims:
            assert not amount.need_masking(), f"pad: {amount} needs masking, not supported currently"
        return amount.get_dim_value()
    if isinstance(amount, Tensor):
        assert amount.dims == (), f"RF JaxBackend: pad amount {amount} must be scalar and static"
        return int(amount.raw_tensor)
    if isinstance(amount, int):
        return amount
    raise TypeError(f"pad: invalid amount {amount!r}")


def _raw(value: Union[float, Tensor], dtype) -> jax.Array:
    """
    :param value: scalar, or a Tensor holding one
    :param dtype: dtype to convert a plain scalar to
    :return: the value as a JAX array, so that a custom_vjp gets a well-defined aval for it
    """
    if isinstance(value, Tensor):
        return value.raw_tensor
    return jnp.asarray(value, dtype=dtype)


@jax.custom_vjp
def _scale_grad(x: jax.Array, scale: jax.Array) -> jax.Array:
    """
    :param x:
    :param scale:
    :return: x unchanged; the backward pass scales the gradient by scale (scale=-1 is gradient reversal)
    """
    del scale
    return x


def _scale_grad_fwd(x: jax.Array, scale: jax.Array):
    return x, scale


def _scale_grad_bwd(scale: jax.Array, grad: jax.Array):
    # the cotangent for scale itself is zero: it is a knob on the gradient, not an input of the value
    return grad * scale, jnp.zeros_like(scale)


_scale_grad.defvjp(_scale_grad_fwd, _scale_grad_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def _scale_shift_grad(x: jax.Array, scale: jax.Array, shift: jax.Array, axis: Optional[int]) -> jax.Array:
    """
    :param x:
    :param scale:
    :param shift:
    :param axis: if given, the shift is weighted by the summed absolute gradient over this axis
    :return: x unchanged; the backward pass scales and shifts the gradient
    """
    del scale, shift, axis
    return x


def _scale_shift_grad_fwd(x: jax.Array, scale: jax.Array, shift: jax.Array, axis: Optional[int]):
    del axis
    return x, (scale, shift)


def _scale_shift_grad_bwd(axis: Optional[int], res, grad: jax.Array):
    scale, shift = res
    grad_out = grad * scale
    if axis is not None:
        grad_out = grad_out + shift * jnp.sum(jnp.abs(grad), axis=axis, keepdims=True)
    else:
        grad_out = grad_out + shift
    return grad_out, jnp.zeros_like(scale), jnp.zeros_like(shift)


_scale_shift_grad.defvjp(_scale_shift_grad_fwd, _scale_shift_grad_bwd)


def _softmax(tensor: Tensor, *, axis: Dim, use_mask: bool, log: bool) -> Tensor:
    """
    :param tensor:
    :param axis:
    :param use_mask: mask out the padded frames of a dynamic axis before normalizing
    :param log: log_softmax instead of softmax
    :return: (log_)softmax over axis
    """
    tensor = rf.amp_cast_float32(tensor)  # mixed precision: normalization is done in float32
    out = tensor.copy_template("log_softmax" if log else "softmax")
    axis_int = tensor.dims.index(axis)
    raw = tensor.raw_tensor
    any_valid = None
    if use_mask and axis.need_masking():
        mask = tensor.get_sequence_mask_broadcast(axis=axis)
        inf_value = get_global_inf_value()
        raw = jnp.where(mask, raw, -inf_value)
        # A fully masked row (a zero-length filler seq of the bound-shape regime)
        # would give NaN from (-inf) - (-inf), poisoning everything downstream.
        # Substitute a finite uniform row before the softmax, so not even its backward sees NaN,
        # and define the result of those rows explicitly afterwards.
        any_valid = jnp.any(mask, axis=axis_int, keepdims=True)
        raw = jnp.where(any_valid, raw, jnp.zeros_like(raw))
    out_raw = (jax.nn.log_softmax if log else jax.nn.softmax)(raw, axis=axis_int)
    if any_valid is not None:
        fill = jnp.full_like(out_raw, -get_global_inf_value()) if log else jnp.zeros_like(out_raw)
        out_raw = jnp.where(any_valid, out_raw, fill)
    out.dtype = JaxBackend.get_dtype_name_raw(out_raw)
    out.raw_tensor = out_raw
    return out


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


def _match_device(a, b):
    """
    :param a:
    :param b:
    :return: the two, on one device: a scalar operand is moved to where the other one lives.

    RETURNN keeps the dynamic sizes of dims on the CPU by design,
    so scalars derived from them meet device tensors constantly (a masked reduce_mean does).
    PyTorch promotes a 0-dim operand to the other one's device, JAX rejects the computation,
    so the transfer happens here, at the one place binary ops go through.
    """
    if not isinstance(a, jax.Array) or not isinstance(b, jax.Array):
        return a, b
    if a.ndim and b.ndim:  # same rule as torch: only scalars are auto-moved to meet the other operand
        return a, b
    if isinstance(a, jax.core.Tracer) or isinstance(b, jax.core.Tracer):
        return a, b  # inside a trace there is nothing to compare, and no transfer to insert
    if a.device == b.device:
        return a, b
    if a.ndim == 0:
        return jax.device_put(a, b.device), b
    return a, jax.device_put(b, a.device)


@contextlib.contextmanager
def _x64_disabled():
    """
    Trace the enclosed code with x64 off, so ``jnp.float_`` is float32.

    jax 0.11 dropped ``jax.experimental.disable_x64``; the config flag is the remaining lever.
    It only affects dtype canonicalization at trace time, which is what the callers here need.
    """
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _device_from_str(device: str) -> Optional[jax.Device]:
    """
    :param device: RF device string ("cpu", "cuda", "cuda:1", ...)
    :return: the JAX device, or None if that platform is not available in this process
        (e.g. JAX_PLATFORMS=cuda restricts it to the GPU).
        None means "wherever JAX puts it", which device_put accepts,
        and is the right answer for a placement RF asks for as an optimization.
    """
    kind, _, idx = device.partition(":")
    if kind == "cuda":
        kind = "gpu"
    try:
        devices = jax.devices(kind)
    except RuntimeError:
        return None
    return devices[int(idx)] if idx else devices[0]


def _to_device_of(x: jax.Array, ref: jax.Array) -> jax.Array:
    """
    :param x: array to place
    :param ref: array whose device decides the placement
    :return: ``x`` on ``ref``'s device
    """
    # Tracers are on one device by construction, and devices() on one forces concretization.
    # rf.is_static_traceable() alone is not enough: it is set around the jitted train step,
    # but not inside a lax.while_loop trace, which also passes tracers here.
    if rf.is_static_traceable() or isinstance(x, jax.core.Tracer) or isinstance(ref, jax.core.Tracer):
        return x
    dx = next(iter(x.devices()), None)
    dr = next(iter(ref.devices()), None)
    if dx is None or dr is None or dx == dr:
        return x
    return jax.device_put(x, dr)


def _to_device(x: jax.Array, dev: jax.Device) -> jax.Array:
    """
    :param x: array to place
    :param dev:
    """
    if isinstance(x, jax.core.Tracer):
        return x
    return jax.device_put(x, dev)
