"""
Internal utils
"""

from __future__ import annotations
from typing import Union, Optional, Any, Type, TypeVar, Sequence, Tuple, List
import numpy

from returnn import frontend as _global_rf
from returnn.frontend._backend import Backend, get_backend_by_raw_tensor_type
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf

T = TypeVar("T")


def get_backend_from_tensors(*args):
    """
    :param args:
    :return: frontend, fallback to global frontend
    """
    for x in args:
        if isinstance(x, Tensor):
            # noinspection PyProtectedMember
            return x._raw_backend
    return _global_rf


def get_dtype_name(x: Union[T, Tensor[T], int, float]) -> str:
    """
    :param x: tensor
    :return: dtype of tensor, as string
    """
    if isinstance(x, Tensor):
        return x.dtype
    elif isinstance(x, int):
        return rf.get_default_int_dtype()
    elif isinstance(x, float):
        return rf.get_default_float_dtype()
    else:
        backend = get_backend_by_raw_tensor_type(type(x))
        return backend.get_dtype_name_raw(x)


def is_int(x: Union[T, Tensor[T], int, float]) -> bool:
    """
    :param x:
    :return: whether the dtype is int
    """
    dtype = get_dtype_name(x)
    return dtype.startswith("int") or dtype.startswith("uint")


# There is a native implementation of this (_native tensorCopyTemplateSimple, compareAndCombine).
def bin_op_out_template(
    backend: Type[Backend],
    a: Union[Tensor[T], int, float, numpy.number],
    b: Union[Tensor[T], int, float, numpy.number],
    *,
    name: str,
    copy_sparse_dim: bool = True,
    allow_broadcast_all_sources: Optional[bool] = None,
    dim_order: Optional[Sequence[Dim]] = None,
    allow_scalar: bool = True,
) -> Tuple[Tensor[T], T, T]:
    """
    make template for output tensor of binary op

    :param backend:
    :param a:
    :param b:
    :param name: for returned Tensor. no other functionality
    :param copy_sparse_dim:
    :param allow_broadcast_all_sources: if True, it is allowed that neither a nor b has all dims of the result.
        Not needed when out_dims is specified explicitly.
    :param dim_order: defines the order of the resulting dims. if None, it is automatically inferred from a and b.
        Not all the dims of a and b need to be specified here, and there could also be other dims in the dim_order.
    :param allow_scalar: if True, it is allowed that a or b is a scalar, and then no broadcast dims are added.
        This can be relevant to allow things like x * 2, where x in on GPU, and then PyTorch allows 2 to stay on CPU.
    :return: out, a_raw, b_raw
    """
    src_dtype = None
    src_device = None
    if isinstance(a, Tensor):
        src_dtype = a.dtype
        src_device = a.device
    elif isinstance(b, Tensor):
        src_dtype = b.dtype
        src_device = b.device
    a = rf.convert_to_tensor(a, dtype=src_dtype, device=src_device, keep_scalar_on_cpu=allow_scalar, _backend=backend)
    src_dtype = src_dtype or a.dtype
    b = rf.convert_to_tensor(b, dtype=src_dtype, device=src_device, keep_scalar_on_cpu=allow_scalar, _backend=backend)
    # sanity checks
    # noinspection PyProtectedMember
    assert a._raw_backend == b._raw_backend, "Cannot combine tensors from two different frontends, e.g. TF and PT"
    all_dims = []
    for dim in a.dims + b.dims:
        if dim in all_dims:
            continue
        # Not simply `all_dims.append(dim)`,
        # because a dim might occur multiple times in a.dims or b.dims
        # (with different match_priority),
        # e.g. in the case of square matrices.
        # Still it is the common case that they are unique,
        # and this allows for a faster path.
        if a.dims.count(dim) <= 1 and b.dims.count(dim) <= 1:
            all_dims.append(dim)
            continue
        if a.dims.count(dim) >= b.dims.count(dim):
            all_dims.extend([dim_ for dim_ in a.dims if dim_ == dim])
        else:
            all_dims.extend([dim_ for dim_ in b.dims if dim_ == dim])
    if all(set(x.dims) != set(all_dims) for x in (a, b)):
        if allow_broadcast_all_sources is False:
            raise ValueError(f"compare: sources {a!r} {b!r} not allowed with allow_broadcast_all_sources=False")
        elif allow_broadcast_all_sources is None:
            raise ValueError(f"compare: sources {a!r} {b!r} require explicit allow_broadcast_all_sources=True")
        elif allow_broadcast_all_sources is True:
            pass
        else:
            raise TypeError(f"invalid type for allow_broadcast_all_sources: {type(allow_broadcast_all_sources)}")
    if dim_order:
        all_dims.sort(key=lambda d: dim_order.index(d) if d in dim_order else len(dim_order))
    out = Tensor(name, dims=all_dims, dtype=src_dtype)
    out.feature_dim = res_feature_dim(a, b)
    if copy_sparse_dim:
        out.sparse_dim = res_sparse_dim(a, b)
    if not allow_scalar or a.dims:
        a_raw = a.copy_compatible_to_dims_raw(all_dims)
    else:
        a_raw = a.raw_tensor
    if not allow_scalar or b.dims:
        b_raw = b.copy_compatible_to_dims_raw(all_dims)
    else:
        b_raw = b.raw_tensor
    return out, a_raw, b_raw


def res_feature_dim(a: Tensor, b: Tensor) -> Optional[Dim]:
    """
    :param a:
    :param b:
    :return: feature dim if consistent or None
    """
    if a.feature_dim and not b.feature_dim:
        return a.feature_dim
    if b.feature_dim and not a.feature_dim:
        return b.feature_dim
    if a.feature_dim and b.feature_dim and a.feature_dim == b.feature_dim:
        return a.feature_dim
    return None


def res_sparse_dim(a: Tensor, b: Tensor) -> Optional[Dim]:
    """
    :param a:
    :param b:
    :return: sparse dim if consistent or None
    """
    if a.sparse_dim and not b.sparse_dim:
        return a.sparse_dim
    if b.sparse_dim and not a.sparse_dim:
        return b.sparse_dim
    if a.sparse_dim and b.sparse_dim and a.sparse_dim == b.sparse_dim:
        return a.sparse_dim
    return None


def strided_slice_raw_key(
    tensor: Tensor,
    axis: Optional[Union[Dim, Sequence[Dim]]] = None,
    key: Optional[rf.ItemKeyType] = None,
    key_dim: Optional[Union[Dim, Sequence[Dim]]] = None,
) -> Tuple[Union[slice, int, T, Sequence[Union[None, slice, int, T]]], Tuple[Dim, ...]]:
    """
    Given an axis and a key, return a raw key that can be used to index into a raw tensor,
    as in `raw_tensor[key]`.
    The tensor is needed to infer the raw axis index.

    :return: raw item key, resulting dims of reduced tensor
    """
    if key is None or (isinstance(key, (list, tuple)) and len(key) == 0):
        return slice(None, None), tensor.dims
    if isinstance(key, Tensor) and key.dtype == "bool":
        raise NotImplementedError("strided_slice: boolean mask")
    if isinstance(key, (slice, int, numpy.number, numpy.ndarray, Tensor)):
        if axis is None:
            axis = _slice_find_sparse_dim(key)
        if not isinstance(axis, Dim):
            raise TypeError(f"strided_slice: must specify axis for key, got {type(axis).__name__}")
        axis_int = tensor.get_axis_from_description(axis)
        key_raw = _map_slice_value_raw(key)
        if _slice_value_is_reduce(key):
            res_dims = tuple(d for (i, d) in enumerate(tensor.dims) if i != axis_int)
        else:
            if not isinstance(key_dim, Dim):
                raise TypeError(f"strided_slice: expected key_dim of type Dim, got {key_dim}")
            res_dims = tuple(d if i != axis_int else key_dim for (i, d) in enumerate(tensor.dims))
        if axis_int == 0:
            return key_raw, res_dims
        return (None,) * axis_int + (key_raw,), res_dims
    if not isinstance(key, (list, tuple)):
        raise TypeError(f"strided_slice: unexpected key type: {type(key).__name__}")
    if axis is not None:
        if not isinstance(axis, (list, tuple)):
            raise TypeError(
                f"strided_slice: key is sequence, thus expect axis to be sequence as well, got {type(axis).__name__}"
            )
        if len(axis) != len(key):
            raise ValueError(f"strided_slice: mismatching axis seq len {len(axis)} and key seq length {len(key)}")
    if key_dim is not None:
        if not isinstance(key_dim, (list, tuple)):
            raise TypeError(
                f"strided_slice: key is sequence, thus expect key_dim to be sequence as well,"
                f" got {type(key_dim).__name__}"
            )
        if len(key_dim) != len(key):
            raise ValueError(f"strided_slice: mismatching key_dim seq len {len(key_dim)} and key seq length {len(key)}")
    raw_out_keys = {}  # per raw axis
    raw_out_dims: List[Optional[Dim]] = list(tensor.dims)
    for i, key_ in enumerate(key):
        axis_ = None
        if axis is not None:
            axis_ = axis[i]
        if axis_ is None:
            axis_ = _slice_find_sparse_dim(key_)
        if not isinstance(axis_, Dim):
            raise TypeError(f"strided_slice: must specify axis for key sequence, got {type(axis_).__name__}")
        axis_int = tensor.get_axis_from_description(axis_)
        key_raw = _map_slice_value_raw(key_)
        if axis_int in raw_out_keys:
            raise ValueError(f"strided_slice: duplicate axis {axis_} in sequence")
        raw_out_keys[axis_int] = key_raw
        if _slice_value_is_reduce(key_):
            raw_out_dims[axis_int] = None
        else:
            key_dim_ = None
            if key_dim is not None:
                key_dim_ = key_dim[i]
            if not isinstance(key_dim_, Dim):
                raise TypeError(f"strided_slice: expected key_dim for key sequence, got {type(key_dim_).__name__}")
            raw_out_dims[axis_int] = key_dim_
    out = []
    for i in range(0, max(raw_out_keys) + 1):
        out.append(raw_out_keys.get(i))
    return tuple(out), tuple(d for d in raw_out_dims if d is not None)


def _slice_find_sparse_dim(v: Union[Tensor, slice, Any]) -> Optional[Dim]:
    if isinstance(v, Tensor):
        return v.sparse_dim
    if isinstance(v, slice):
        attribs = {k: getattr(v, k) for k in ("start", "stop", "step")}
        tensors = {k: v for (k, v) in attribs.items() if isinstance(v, Tensor)}
        sparse_dims = {k: v.sparse_dim for (k, v) in tensors.items() if v.sparse_dim}
        sparse_dims_ = list(set(sparse_dims.values()))
        if len(sparse_dims_) == 0:
            return None
        if len(sparse_dims_) == 1:
            return sparse_dims_[0]
        raise ValueError(f"strided_slice: multiple different sparse dims in slice {v}: {sparse_dims}")
    return None


def _map_slice_value_raw(
    v: Union[None, slice, int, numpy.number, numpy.ndarray, Tensor[T]]
) -> Union[None, slice, int, numpy.number, T]:
    if v is None:
        return None
    if isinstance(v, slice):
        return slice(_map_slice_value_raw(v.start), _map_slice_value_raw(v.stop), _map_slice_value_raw(v.step))
    if isinstance(v, (int, numpy.number)):
        return v
    if isinstance(v, numpy.ndarray):
        assert v.ndim <= 1, f"strided_slice: expect scalar or vector, got array with shape {v.shape}"
        return v
    if isinstance(v, Tensor):
        assert len(v.dims) <= 1, f"strided_slice: expect scalar or vector, got Tensor with dims {v.dims}"
        return v.raw_tensor
    raise TypeError(f"strided_slice: got unexpected value of type {type(v).__name__}")


def _slice_value_is_reduce(v: Union[None, slice, int, numpy.number, numpy.ndarray, Tensor[T]]) -> bool:
    if v is None:
        return False
    if isinstance(v, slice):
        return False
    if isinstance(v, (int, numpy.number)):
        return True
    if isinstance(v, numpy.ndarray):
        assert v.ndim <= 1, f"strided_slice: expect scalar or vector, got array with shape {v.shape}"
        return v.ndim == 0
    if isinstance(v, Tensor):
        assert len(v.dims) <= 1, f"strided_slice: expect scalar or vector, got Tensor with dims {v.dims}"
        return v.dims == 0
    raise TypeError(f"strided_slice: got unexpected value of type {type(v).__name__}")
