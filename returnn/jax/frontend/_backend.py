"""
Backend for exposing JAX-specific functionality.

JAX dispatches op-by-op like PyTorch, so this backend is eager,
and the same code runs unchanged inside ``jax.jit``
(the raw tensors are tracers there, see :func:`returnn.frontend._backend.get_backend_by_raw_tensor_type`).
Everything that must NOT read a value on the host (shapes, seq lens)
therefore follows the same rules as the static-traceable regime of the PyTorch backend.
"""

from __future__ import annotations
from typing import Optional, Union, Sequence, Tuple, Dict
from functools import partial
import itertools
import numpy
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as _logsumexp

from returnn.tensor import Tensor, Dim
from returnn.util.basic import get_global_inf_value, prod

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
        """
        :param source:
        :param axis: dims to reduce over
        :param mode: "sum", "max", "min", "mean", "logsumexp", "any", "all", "argmin", "argmax"
        :param use_mask: mask out the padded frames of a dynamic axis first
        :return: reduced tensor
        """
        assert mode in Backend._AllowedReduceModes
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
                correction_factor = rf.masked_fraction_of_shape(axes, inverse=True, device=source.device)
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
            out_dtype = "bool" if mode in ("any", "all") else source.dtype
            sparse_dim = source.sparse_dim
        if correction_factor is not None:
            raw_result = raw_result * correction_factor.copy_compatible_to_dims_raw(res_dims)
        return Tensor(
            name=f"reduce_{mode}",
            raw_tensor=raw_result,
            dims=res_dims,
            dtype=out_dtype,
            sparse_dim=sparse_dim,
        )

    @staticmethod
    def gather(source: Tensor, *, indices: Union[Tensor, int], axis: Dim, clip_to_valid: bool = False) -> Tensor:
        """
        :param source:
        :param indices: index tensor, or a single index
        :param axis: the dim of source to index into
        :param clip_to_valid: clip the indices into the valid range of axis first
        :return: source with axis replaced by the dims of indices
        """
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
            out_raw = jnp.take_along_axis(
                source.raw_tensor, indices.raw_tensor.astype(jnp.int32), axis=axis_int, mode="clip"
            )
            if len(index_own_dims) == 0:
                out_raw = jnp.squeeze(out_raw, axis=axis_int)
            elif len(index_own_dims) > 1:
                out_raw = jnp.reshape(out_raw, [d.get_dim_value() for d in out.dims])
            out.raw_tensor = out_raw
        else:
            # indices are independent of the source's other dims: a plain take along axis
            out_raw = jnp.take(
                source.raw_tensor, indices.raw_tensor.astype(jnp.int32).reshape(-1), axis=axis_int, mode="clip"
            )
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
        """
        :param source:
        :param axes: dims to pad
        :param padding: (left, right) per axis
        :param out_dims: resulting dims, per axis
        :param handle_dynamic_dims: re-mask the padded-in frames of a dynamic axis afterwards
        :param mode: "constant", "reflect", "replicate" or "circular"
        :param value: for mode "constant"
        :return: padded tensor
        """
        assert len(out_dims) == len(axes) == len(padding)
        assert not isinstance(value, Tensor) or value.dims == (), (
            "RF JaxBackend: pad with a non-scalar value not implemented"
        )
        raw_pad = []
        for dim in source.dims:
            if dim not in axes:
                raw_pad.append((0, 0))
                continue
            left, right = padding[axes.index(dim)]
            raw_pad.append((_pad_amount(left, handle_dynamic_dims), _pad_amount(right, handle_dynamic_dims)))
        out = source.copy_template_new_dim_tags(
            [out_dims[axes.index(dim)] if dim in axes else dim for dim in source.dim_tags], keep_special_axes=True
        )
        value_ = value.raw_tensor if isinstance(value, Tensor) else value
        jnp_mode = _PadModeMap.get(mode, mode)
        if jnp_mode == "constant":
            out.raw_tensor = jnp.pad(source.raw_tensor, raw_pad, mode="constant", constant_values=value_ or 0)
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
        """
        :param raw_tensor:
        :param axes: axes to squeeze
        :return: squeezed raw tensor
        """
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
        """
        :param source:
        :param axis: the dim to split
        :param dims: what to split it into
        :param pad_to_multiples: not implemented
        :param pad_value: not implemented
        :return: source with axis replaced by dims
        """
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
        """
        :param source:
        :param in_dims: dims of source to reshape, need not be adjacent
        :param out_dims: what to reshape them into, same total size
        :return: source with in_dims replaced by out_dims
        """
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
        """
        :param source:
        :param axis: static axis to split
        :param out_dims: parts, summing to axis
        :return: one tensor per out_dim
        """
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
        """
        :param source:
        :param dim: the new dim
        :return: source with dim added (broadcast, not copied, where possible)
        """
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
        """
        :param source:
        :param axis: dim of size 1
        :return: source without axis
        """
        axis_int = source.get_axis_from_description(axis)
        out = source.copy_template_excluding_axis(axis_int)
        out.raw_tensor = jnp.squeeze(source.raw_tensor, axis=axis_int)
        return out

    @staticmethod
    def concat(*sources: Tuple[Tensor, Dim], allow_broadcast: bool = False, out_dim: Dim) -> Tensor:
        """
        :param sources: (tensor, its dim to concat over) pairs
        :param allow_broadcast:
        :param out_dim:
        :return: concatenated tensor
        """
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
        """
        :param sources:
        :param out_dim: the new leading dim
        :return: stacked tensor
        """
        out_dims = (out_dim,) + sources[0].dims
        out = Tensor("stack", dims=out_dims, dtype=sources[0].dtype, sparse_dim=sources[0].sparse_dim)
        out.raw_tensor = jnp.stack([s.copy_compatible_to_dims_raw(out_dims[1:]) for s in sources], axis=0)
        return out

    @staticmethod
    def unstack(source: Tensor, *, axis: Dim) -> Tuple[Tensor, ...]:
        """
        :param source:
        :param axis: static axis to unstack
        :return: one tensor per index of axis, each without axis
        """
        axis_int = source.dims.index(axis)
        template = source.copy_template_excluding_axis(axis_int)
        result = []
        for i in range(source.raw_tensor.shape[axis_int]):
            out = template.copy_template()
            out.raw_tensor = jax.lax.index_in_dim(source.raw_tensor, i, axis=axis_int, keepdims=False)
            result.append(out)
        return tuple(result)

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
        """
        :param dims:
        :param fill_value:
        :param dtype:
        :param device:
        :param sparse_dim:
        :param feature_dim:
        :return: tensor of the given shape, filled with fill_value
        """
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
        """
        :param source:
        :param axis:
        :param start: may be a device scalar (dynamic_slice clamps it, like torch.narrow)
        :param end:
        :param step: only 1
        :param size:
        :param out_dim:
        :return: sliced tensor
        """
        assert step is None or (isinstance(step, int) and step == 1), "RF JaxBackend: slice step != 1 not implemented"
        axis_int = source.get_axis_from_description(axis, allow_int=False)
        out = source.copy_template_replace_dim_tag(axis=axis_int, new_dim_tag=out_dim)
        if isinstance(start, Tensor):
            assert start.dims == ()
            start = start.raw_tensor
        elif start is None:
            start = 0
        # the size must be static: JAX has no data-dependent shapes
        if isinstance(size, Dim):
            assert end is None
            size = size.get_dim_value()
        elif isinstance(size, Tensor):
            assert end is None and size.dims == (), f"RF JaxBackend: slice size {size} must be scalar and static"
            size = int(size.raw_tensor)
        elif size is None:
            if isinstance(end, Tensor):
                assert end.dims == (), f"RF JaxBackend: slice end {end} must be scalar and static"
                end = int(end.raw_tensor)
            elif isinstance(end, int):
                if end < 0:
                    end += axis.get_dim_value()
            elif end is None:
                end = axis.get_dim_value()
            else:
                raise TypeError(f"slice: unsupported type for end: {type(end)}")
            size = end - int(start) if not isinstance(start, jax.Array) else end - start
        elif not isinstance(size, int):
            raise TypeError(f"slice: unsupported type for size: {type(size)}")
        out.raw_tensor = jax.lax.dynamic_slice_in_dim(source.raw_tensor, start, int(size), axis=axis_int)
        return out

    @staticmethod
    def flip_no_mask(source: Tensor, *, axis: Dim) -> Tensor:
        """
        :param source:
        :param axis:
        :return: source reversed along axis, ignoring masking
        """
        axis_int = source.get_axis_from_description(axis, allow_int=False)
        out = source.copy_template("flip")
        out.raw_tensor = jnp.flip(source.raw_tensor, axis=axis_int)
        return out

    @staticmethod
    def cumsum(source: Tensor, *, spatial_dim: Dim) -> Tensor:
        """
        :param source:
        :param spatial_dim:
        :return: cumsum over spatial_dim
        """
        axis = source.get_axis_from_description(spatial_dim)
        out = source.copy_template("cumsum")
        out.raw_tensor = jnp.cumsum(source.raw_tensor, axis=axis, dtype=source.raw_tensor.dtype)
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
        """
        :param start:
        :param end:
        :param weight:
        :param allow_broadcast_all_sources:
        :return: start + weight * (end - start)
        """
        weight = rf.convert_to_tensor(weight, _backend=JaxBackend, device=start.device)
        out = Tensor.get_common_data(
            [start, end, weight], allow_broadcast_all_sources=allow_broadcast_all_sources, name="lerp"
        )
        start_raw = start.copy_compatible_to_dims_raw(out.dims)
        end_raw = end.copy_compatible_to_dims_raw(out.dims)
        weight_raw = weight.copy_compatible_to_dims_raw(out.dims)
        out.raw_tensor = start_raw + weight_raw * (end_raw - start_raw)
        return out

    # --- convolution

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
        """
        :param source:
        :param in_dim: input feature dim
        :param out_dim: output feature dim
        :param in_spatial_dims:
        :param out_spatial_dims:
        :param filter: [out_dim, in_dim // groups, *filter_size]
        :param filter_size:
        :param padding: "same", "valid", or explicit amounts
        :param strides:
        :param dilation_rate:
        :param groups: depthwise conv = groups == in_dim
        :param bias:
        :return: (output, out_spatial_dims)
        """
        if not out_spatial_dims:
            out_spatial_dims = rf.make_conv_out_spatial_dims(
                in_spatial_dims=in_spatial_dims,
                filter_size=filter_size,
                strides=strides or 1,
                dilation_rate=dilation_rate or 1,
                padding=padding,
            )
        n_spatial = len(filter_size)
        filter_in_dim = in_dim if not groups or groups == 1 else in_dim // groups
        filter = filter.copy_transpose((out_dim, filter_in_dim) + tuple(filter_size))
        batch_dims = [d for d in source.dims if d not in (in_dim,) + tuple(in_spatial_dims)]
        # conv wants (N, C, *spatial)
        source = source.copy_transpose(batch_dims + [in_dim] + list(in_spatial_dims))
        if len(batch_dims) == 1:
            src_raw = source.raw_tensor
        else:
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
            window_strides=_to_seq(strides or 1, n_spatial),
            padding=_conv_padding(padding, n_spatial),
            rhs_dilation=_to_seq(dilation_rate or 1, n_spatial),
            dimension_numbers=_conv_dim_numbers(n_spatial),
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
        """
        :param source:
        :param mode: "max" or "avg"
        :param pool_size:
        :param padding:
        :param dilation_rate:
        :param strides:
        :param in_spatial_dims:
        :param out_spatial_dims:
        :return: (output, out_spatial_dims)
        """
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
        batch_dims = [d for d in source.dims if d not in tuple(in_spatial_dims)]
        source = source.copy_transpose(batch_dims + list(in_spatial_dims))
        # all batch-like dims merged into one leading axis; the window is 1 there
        src_raw = jnp.reshape(source.raw_tensor, [-1] + [d.get_dim_value() for d in in_spatial_dims])
        window = (1,) + tuple(pool_size)
        window_strides = (1,) + tuple(strides)
        window_dilation = (1,) + _to_seq(dilation_rate or 1, n_spatial)
        pad = _conv_padding(padding, n_spatial)
        if not isinstance(pad, str):
            pad = [(0, 0)] + list(pad)
        dtype = src_raw.dtype
        if mode == "max":
            out_raw = jax.lax.reduce_window(
                src_raw, _dtype_min(dtype), jax.lax.max, window, window_strides, pad, window_dilation=window_dilation
            )
        elif mode == "avg":
            assert all(d == 1 for d in window_dilation), "RF JaxBackend: dilation_rate only supported for max_pool"
            sums = jax.lax.reduce_window(src_raw, jnp.zeros((), dtype), jax.lax.add, window, window_strides, pad)
            # divide by the number of REAL frames per window, i.e. torch's count_include_pad=False
            counts = jax.lax.reduce_window(
                jnp.ones_like(src_raw), jnp.zeros((), dtype), jax.lax.add, window, window_strides, pad
            )
            out_raw = sums / counts
        else:
            raise NotImplementedError(f"RF JaxBackend: pool mode {mode!r} not implemented")
        out = Tensor("pool", dims=batch_dims + list(out_spatial_dims), dtype=source.dtype)
        out.raw_tensor = jnp.reshape(out_raw, [d.get_dim_value() for d in out.dims])
        if source.feature_dim and source.feature_dim in out.dims:
            out.feature_dim = source.feature_dim
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

        The drawn values do NOT match any other backend's for the same seed (different PRNG).
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

        def _scalar(v, default):
            if v is None:
                return default
            if isinstance(v, Tensor):
                assert v.dims == (), f"RF JaxBackend: only scalar supported here, got {v}"
                return v.raw_tensor
            return v

        if distribution == "uniform":
            assert mean is None and stddev is None
            minval_ = _scalar(minval, 0)
            if jnp.issubdtype(dtype_, jnp.floating):
                maxval_ = _scalar(maxval, 1)
                if isinstance(maxval, Tensor) and maxval.dims:
                    # per-element upper bound: affine transform of U[0,1)
                    maxval_ = maxval.copy_compatible_to_dims_raw(out.dims)
                    raw = minval_ + jax.random.uniform(key, shape, dtype=dtype_) * (maxval_ - minval_)
                else:
                    raw = jax.random.uniform(key, shape, dtype=dtype_, minval=minval_, maxval=maxval_)
            else:
                assert maxval is not None, "maxval must be specified for integer random uniform"
                maxval_ = _scalar(maxval, None)
                raw = jax.random.randint(key, shape, minval_, maxval_, dtype=dtype_)
        elif distribution == "normal":
            assert minval is None and maxval is None
            mean_, stddev_ = _scalar(mean, 0), _scalar(stddev, 1)
            raw = mean_ + stddev_ * jax.random.normal(key, shape, dtype=dtype_)
        elif distribution == "truncated_normal":
            mean_, stddev_ = _scalar(mean, 0), _scalar(stddev, 1)
            minval_ = _scalar(minval, mean_ - 2 * stddev_)
            maxval_ = _scalar(maxval, mean_ + 2 * stddev_)
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
        :param tensor:
        :param device:
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

        Nothing to do on the raw side: JAX arrays carry no requires_grad,
        and which parameters get differentiated is decided by the engine
        when it builds the pytree it passes to jax.grad, from rf.Parameter.trainable.
        """

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
        """
        :param param:
        :param device:
        :param dtype:
        """
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
        """
        :param y:
        :param x:
        :return: nothing -- this cannot work on JAX
        """
        raise NotImplementedError(
            "RF JaxBackend: rf.gradient(y, x) has no JAX equivalent."
            " JAX has no tape, so a gradient exists only for a FUNCTION,"
            " and by the time you hold y, the computation that produced it is gone."
            " Differentiate the step function instead (jax.grad / jax.value_and_grad),"
            " which is what the engine does."
        )

    @staticmethod
    def scaled_gradient(tensor: Tensor, scale: Union[float, Tensor]) -> Tensor:
        """
        :param tensor:
        :param scale:
        :return: just the tensor, but its gradient is scaled by the given factor
        """
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
        """
        :param x:
        :param scale: will scale gradient by this value
        :param shift: will shift gradient by this value
        :param scale_shift_by_sum_over_axis: if given, will scale and shift by the sum over the given axis
        :return: just x, but gradient in backward pass will be transformed accordingly
        """
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

        Works on axis INDICES, not on dim identity, because a dim can occur twice in one tensor
        (rf.Linear from a dim to itself, e.g. the attention output projection).
        `get_axis_from_description` resolves which occurrence is the reduce axis (via match_priority);
        an einsum over dim-keyed letters could not express that and would silently take a diagonal.
        """
        if isinstance(reduce, Dim):
            reduce = [reduce]
        if use_mask and any(dim.dyn_size_ext is not None for dim in reduce):
            raise NotImplementedError("RF JaxBackend: masking in matmul reduce not yet implemented")
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
    return x


def _scale_shift_grad_fwd(x: jax.Array, scale: jax.Array, shift: jax.Array, axis: Optional[int]):
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
        # Substitute a finite uniform row BEFORE the softmax, so not even its backward sees NaN,
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
