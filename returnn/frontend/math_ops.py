"""
Math ops
"""

from __future__ import annotations
import typing
from typing import Optional, Sequence, Union
import numpy
from returnn.tensor import Tensor, Dim
from .types import RawTensorTypes as _RawTensorTypes

__all__ = [
    "compare",
    "combine",
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "add",
    "sub",
    "mul",
    "true_divide",
    "floor_divide",
    "neg",
    "mod",
    "pow",
    "logical_and",
    "logical_or",
    "logical_not",
    "exp",
    "log",
    "log1p",
    "sqrt",
    "square",
    "abs",
    "tanh",
    "sigmoid",
    "sin",
    "cos",
    "ceil",
    "floor",
    "round",
    "relu",
    "elu",
    "selu",
]


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


def equal(a: Tensor, b: Tensor) -> Tensor:
    """equal"""
    return compare(a, "equal", b)


def less(a: Tensor, b: Tensor) -> Tensor:
    """less"""
    return compare(a, "less", b)


def less_equal(a: Tensor, b: Tensor) -> Tensor:
    """less_equal"""
    return compare(a, "less_equal", b)


def greater(a: Tensor, b: Tensor) -> Tensor:
    """greater"""
    return compare(a, "greater", b)


def greater_equal(a: Tensor, b: Tensor) -> Tensor:
    """greater_equal"""
    return compare(a, "greater_equal", b)


def not_equal(a: Tensor, b: Tensor) -> Tensor:
    """not_equal"""
    return compare(a, "not_equal", b)


def add(a: Tensor, b: Tensor) -> Tensor:
    """add"""
    return combine(a, "add", b)


def sub(a: Tensor, b: Tensor) -> Tensor:
    """sub"""
    return combine(a, "sub", b)


def mul(a: Tensor, b: Tensor) -> Tensor:
    """mul"""
    return combine(a, "mul", b)


def true_divide(a: Tensor, b: Tensor) -> Tensor:
    """truediv"""
    return combine(a, "truediv", b)


def floor_divide(a: Tensor, b: Tensor) -> Tensor:
    """floordiv"""
    return combine(a, "floordiv", b)


def neg(a: Tensor) -> Tensor:
    """neg"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation_raw(a.raw_tensor, "neg")


def mod(a: Tensor, b: Tensor) -> Tensor:
    """mod"""
    return combine(a, "mod", b)


# noinspection PyShadowingBuiltins
def pow(a: Tensor, b: Tensor) -> Tensor:
    """pow"""
    return combine(a, "pow", b)


def logical_and(a: Tensor, b: Tensor) -> Tensor:
    """logical_and"""
    return combine(a, "logical_and", b)


def logical_or(a: Tensor, b: Tensor) -> Tensor:
    """logical_or"""
    return combine(a, "logical_or", b)


def logical_not(a: Tensor) -> Tensor:
    """logical_not"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation_raw(a.raw_tensor, "logical_not")


def exp(a: Tensor) -> Tensor:
    """exp"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation_raw(a.raw_tensor, "exp")


def log(a: Tensor) -> Tensor:
    """log"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation_raw(a.raw_tensor, "log")


def log1p(a: Tensor) -> Tensor:
    """log1p"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation_raw(a.raw_tensor, "log1p")


def sqrt(a: Tensor) -> Tensor:
    """sqrt"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation_raw(a.raw_tensor, "sqrt")


def square(a: Tensor) -> Tensor:
    """square"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation_raw(a.raw_tensor, "square")


# noinspection PyShadowingBuiltins
def abs(a: Tensor) -> Tensor:
    """abs"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation_raw(a.raw_tensor, "abs")


def tanh(a: Tensor) -> Tensor:
    """tanh"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation_raw(a.raw_tensor, "tanh")


def sigmoid(a: Tensor) -> Tensor:
    """sigmoid"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation_raw(a.raw_tensor, "sigmoid")


def sin(a: Tensor) -> Tensor:
    """sin"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation_raw(a.raw_tensor, "sin")


def cos(a: Tensor) -> Tensor:
    """cos"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation_raw(a.raw_tensor, "cos")


def ceil(a: Tensor) -> Tensor:
    """ceil"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation_raw(a.raw_tensor, "ceil")


def floor(a: Tensor) -> Tensor:
    """floor"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation_raw(a.raw_tensor, "floor")


# noinspection PyShadowingBuiltins
def round(a: Tensor) -> Tensor:
    """round"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation_raw(a.raw_tensor, "round")


def relu(a: Tensor) -> Tensor:
    """relu"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation_raw(a.raw_tensor, "relu")


def elu(a: Tensor) -> Tensor:
    """elu"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation_raw(a.raw_tensor, "elu")


def selu(a: Tensor) -> Tensor:
    """selu"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation_raw(a.raw_tensor, "selu")
