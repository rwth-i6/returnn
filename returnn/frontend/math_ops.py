"""
Math ops
"""

from __future__ import annotations
import typing
from typing import Optional, Sequence, Union
import numpy
from returnn.tensor import Tensor, Dim
from .types import RawTensorTypes as _RawTensorTypes

__all__ = ["compare", "combine"]


@typing.overload
def compare(
    a: Tensor,
    kind: str,
    b: Tensor,
    *,
    allow_broadcast_all_sources: Optional[bool] = None,
    dim_order: Optional[Sequence[Dim]] = None,
) -> Tensor:
    """compare with two tensors"""


def compare(
    a: Union[Tensor, _RawTensorTypes],
    kind: str,
    b: Union[Tensor, _RawTensorTypes],
    *,
    allow_broadcast_all_sources: Optional[bool] = None,
    dim_order: Optional[Sequence[Dim]] = None,
) -> Tensor:
    """
    :param a:
    :param kind: "equal"|"==", "less"|"<", "less_equal"|"<=", "greater"|">", "greater_equal"|">=", "not_equal"|"!="
    :param b:
    :param allow_broadcast_all_sources: if True, it is allowed that neither a nor b has all dims of the result.
        Not needed when out_dims is specified explicitly.
    :param dim_order: defines the order of the resulting dims. if None, it is automatically inferred from a and b.
        Not all the dims of a and b need to be specified here, and there could also be other dims in the dim_order.
    :return: element-wise comparison of a and b
    """
    from . import _utils as utils

    backend = utils.get_backend_from_tensors(a, b)
    out, a, b = utils.bin_op_out_template(
        backend,
        a,
        b,
        name="compare",
        dtype="bool",
        allow_broadcast_all_sources=allow_broadcast_all_sources,
        dim_order=dim_order,
    )
    out.raw_tensor = backend.compare_raw(a.raw_tensor, kind, b.raw_tensor)
    return out


@typing.overload
def combine(
    a: Tensor,
    kind: str,
    b: Tensor,
    *,
    allow_broadcast_all_sources: Optional[bool] = None,
    dim_order: Optional[Sequence[Dim]] = None,
) -> Tensor:
    """combine with two tensors"""


def combine(
    a: Union[Tensor, _RawTensorTypes],
    kind: str,
    b: Union[Tensor, _RawTensorTypes],
    *,
    allow_broadcast_all_sources: Optional[bool] = None,
    dim_order: Optional[Sequence[Dim]] = None,
) -> Union[Tensor, _RawTensorTypes]:
    """
    :param a:
    :param kind: "add"|"+", "sub"|"-", "mul"|"*", "truediv"|"/", "floordiv"|"//", "mod"|"%", "pow"|"**",
        "max"|"maximum", "min"|"minimum", "logical_and", "logical_or", "squared_difference"
    :param b:
    :param allow_broadcast_all_sources: if True, it is allowed that neither a nor b has all dims of the result.
        Not needed when out_dims is specified explicitly.
    :param dim_order: defines the order of the resulting dims. if None, it is automatically inferred from a and b.
        Not all the dims of a and b need to be specified here, and there could also be other dims in the dim_order.
    :return: element-wise combination of a and b
    """
    from . import _utils as utils

    backend = utils.get_backend_from_tensors(a, b)
    if isinstance(b, (int, float, bool, numpy.number)):
        if b == 1 and kind in {"truediv", "/", "floordiv", "//", "pow", "**", "mul", "*"}:
            return a
        if b == -1 and kind in {"truediv", "/", "floordiv", "//", "pow", "**", "mul", "*"}:
            return -a
        if b == 0 and kind in {"add", "+", "sub", "-"}:
            return a
        if b and kind in {"logical_and", "logical_or"}:
            return a
    if isinstance(a, (int, float, bool, numpy.number)):
        if a == 1 and kind in {"pow", "**"}:
            return a
        if a == 1 and kind in {"mul", "*"}:
            return b
        if a == 0 and kind in {"add", "+", "sub", "-"}:
            return b
        if a and kind in {"logical_and", "logical_or"}:
            return b
    # Truediv checks for int/int division
    if kind in {"truediv", "/"}:
        if utils.is_int(backend, a) and utils.is_int(backend, b):
            raise ValueError(
                "Dividing a Tensor of type int by an integer is disallowed. Please convert the Tensor to float."
            )
    out, a, b = utils.bin_op_out_template(
        backend,
        a,
        b,
        name="combine",
        dtype=a.dtype,
        allow_broadcast_all_sources=allow_broadcast_all_sources,
        dim_order=dim_order,
    )
    out.raw_tensor = backend.combine_raw(a.raw_tensor, kind, b.raw_tensor)
    return out
