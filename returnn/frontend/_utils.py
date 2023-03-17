"""
Internal utils
"""

from __future__ import annotations
from typing import Union, Optional, Type, TypeVar, Sequence, Tuple
import numpy

from returnn import frontend as _global_rf
from returnn.frontend._backend import Backend
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


def get_dtype_name(backend: Type[Backend], x: Union[T, Tensor[T], int, float]) -> str:
    """
    :param backend:
    :param x: tensor
    :return: dtype of tensor, as string
    """
    if isinstance(x, backend.RawTensorType):
        return backend.get_dtype_name_raw(x)
    elif isinstance(x, Tensor):
        return x.dtype
    elif isinstance(x, int):
        return rf.get_default_int_dtype()
    elif isinstance(x, float):
        return rf.get_default_float_dtype()
    else:
        raise TypeError(f"unexpected type {type(x)}")


def is_int(backend: Type[Backend], x: Union[T, Tensor[T], int, float]) -> bool:
    """
    :param backend:
    :param x:
    :return: whether the dtype is int
    """
    dtype = get_dtype_name(backend, x)
    return dtype.startswith("int") or dtype.startswith("uint")


def bin_op_out_template(
    backend: Type[Backend],
    a: Union[Tensor[T], int, float, numpy.number],
    b: Union[Tensor[T], int, float, numpy.number],
    *,
    name: str,
    dtype: str,
    allow_broadcast_all_sources: Optional[bool] = None,
    dim_order: Optional[Sequence[Dim]] = None,
) -> Tuple[Tensor[T], Tensor[T], Tensor[T]]:
    """
    make template for output tensor of binary op

    :param backend:
    :param a:
    :param b:
    :param name: str
    :param dtype:
    :param allow_broadcast_all_sources: if True, it is allowed that neither a nor b has all dims of the result.
        Not needed when out_dims is specified explicitly.
    :param dim_order: defines the order of the resulting dims. if None, it is automatically inferred from a and b.
        Not all the dims of a and b need to be specified here, and there could also be other dims in the dim_order.
    :return: out, a, b
    """
    a = backend.convert_to_tensor(a)
    b = backend.convert_to_tensor(b)
    # sanity checks
    # noinspection PyProtectedMember
    assert a._raw_backend == b._raw_backend, "Cannot combine tensors from two different frontends, e.g. TF and PT"
    assert a.dtype == a.dtype, "For now only operations with Tensors of the same dtypes are supported."
    all_dims = []
    for dim in a.dims + b.dims:
        if dim not in all_dims:
            all_dims.append(dim)
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
    out = Tensor(name, dims=all_dims, dtype=dtype)
    a = a.copy_compatible_to(out, check_dtype=False, check_sparse=False)
    b = b.copy_compatible_to(out, check_dtype=False, check_sparse=False)
    return out, a, b
