"""
Internal utils
"""

from __future__ import annotations
from typing import Union, Optional, Type, TypeVar, Sequence, Tuple
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


def bin_op_out_template(
    backend: Type[Backend],
    a: Union[Tensor[T], int, float, numpy.number],
    b: Union[Tensor[T], int, float, numpy.number],
    *,
    name: str,
    res_dtype: Optional[str],
    allow_broadcast_all_sources: Optional[bool] = None,
    dim_order: Optional[Sequence[Dim]] = None,
    allow_scalar: bool = True,
) -> Tuple[Tensor[T], Tensor[T], Tensor[T]]:
    """
    make template for output tensor of binary op

    :param backend:
    :param a:
    :param b:
    :param name: for returned Tensor. no other functionality
    :param res_dtype: if not given, infer from a and b
    :param allow_broadcast_all_sources: if True, it is allowed that neither a nor b has all dims of the result.
        Not needed when out_dims is specified explicitly.
    :param dim_order: defines the order of the resulting dims. if None, it is automatically inferred from a and b.
        Not all the dims of a and b need to be specified here, and there could also be other dims in the dim_order.
    :param allow_scalar: if True, it is allowed that a or b is a scalar, and then no broadcast dims are added.
        This can be relevant to allow things like x * 2, where x in on GPU, and then PyTorch allows 2 to stay on CPU.
    :return: out, a, b
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
    if res_dtype is None:
        assert (
            a.dtype == b.dtype
        ), f"For now only operations with Tensors of the same dtypes are supported, got {a} and {b}"
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
    if res_dtype is None:
        res_dtype = src_dtype
    out = Tensor(name, dims=all_dims, dtype=res_dtype)
    out.feature_dim = res_feature_dim(a, b)
    if not allow_scalar or a.dims:
        a = a.copy_compatible_to(out, check_dtype=False, check_sparse=False)
    if not allow_scalar or b.dims:
        b = b.copy_compatible_to(out, check_dtype=False, check_sparse=False)
    return out, a, b


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
