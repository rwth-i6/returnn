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
    "squared_difference",
    "logical_and",
    "logical_or",
    "logical_not",
    "identity",
    "exp",
    "expm1",
    "log",
    "log1p",
    "sqrt",
    "rsqrt",
    "square",
    "abs",
    "tanh",
    "sigmoid",
    "log_sigmoid",
    "sin",
    "cos",
    "ceil",
    "floor",
    "round",
    "relu",
    "elu",
    "selu",
    "silu",
    "softmax",
    "log_softmax",
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


_CompareMap = {
    "==": "equal",
    "eq": "equal",
    "<=": "less_equal",
    "<": "less",
    ">=": "greater_equal",
    ">": "greater",
    "!=": "not_equal",
    "<>": "not_equal",
}


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

    kind = _CompareMap.get(kind, kind)

    backend = utils.get_backend_from_tensors(a, b)
    return backend.compare(a, kind, b, allow_broadcast_all_sources=allow_broadcast_all_sources, dim_order=dim_order)


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


_CombineMap = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "truediv",
    "//": "floordiv",
    "%": "mod",
    "**": "pow",
    "max": "maximum",
    "min": "minimum",
}


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
    from . import _utils

    kind = _CombineMap.get(kind, kind)

    if isinstance(b, (int, float, bool, numpy.number)):
        if b == 1 and kind in {"truediv", "floordiv", "pow", "mul"}:
            return a
        if b == -1 and kind in {"truediv", "floordiv", "pow", "mul"}:
            return -a
        if b == 0 and kind in {"add", "sub"}:
            return a
        if b and kind in {"logical_and", "logical_or"}:
            return a
    if isinstance(a, (int, float, bool, numpy.number)):
        if a == 1 and kind == "pow":
            return a
        if a == 1 and kind == "mul":
            return b
        if a == 0 and kind in {"add", "sub"}:
            return b
        if a and kind in {"logical_and", "logical_or"}:
            return b
    # Truediv checks for int/int division
    if kind in {"truediv", "/"}:
        if _utils.is_int(a) and _utils.is_int(b):
            raise ValueError(
                "Dividing a Tensor of type int by an integer is disallowed. Please convert the Tensor to float."
            )
    backend = _utils.get_backend_from_tensors(a, b)
    return backend.combine(a, kind, b, allow_broadcast_all_sources=allow_broadcast_all_sources, dim_order=dim_order)


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
    return a._raw_backend.activation(a, "neg")


def mod(a: Tensor, b: Tensor) -> Tensor:
    """mod"""
    return combine(a, "mod", b)


# noinspection PyShadowingBuiltins
def pow(a: Tensor, b: Tensor) -> Tensor:
    """pow"""
    return combine(a, "pow", b)


def squared_difference(a: Tensor, b: Tensor) -> Tensor:
    """squared_difference"""
    return combine(a, "squared_difference", b)


def logical_and(a: Tensor, b: Tensor) -> Tensor:
    """logical_and"""
    return combine(a, "logical_and", b)


def logical_or(a: Tensor, b: Tensor) -> Tensor:
    """logical_or"""
    return combine(a, "logical_or", b)


def logical_not(a: Tensor) -> Tensor:
    """logical_not"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation(a, "logical_not")


def identity(x: Tensor) -> Tensor:
    """
    Identity function. Just to have one canonical. Does nothing, returns the input.
    """
    return x


def exp(a: Tensor) -> Tensor:
    """exp"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation(a, "exp")


def expm1(a: Tensor) -> Tensor:
    """expm1"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation(a, "expm1")


def log(a: Tensor) -> Tensor:
    """log"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation(a, "log")


def log1p(a: Tensor) -> Tensor:
    """log1p"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation(a, "log1p")


def sqrt(a: Tensor) -> Tensor:
    """sqrt"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation(a, "sqrt")


def rsqrt(a: Tensor) -> Tensor:
    """rsqrt"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation(a, "rsqrt")


def square(a: Tensor) -> Tensor:
    """square"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation(a, "square")


# noinspection PyShadowingBuiltins
def abs(a: Tensor) -> Tensor:
    """abs"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation(a, "abs")


def tanh(a: Tensor) -> Tensor:
    """tanh"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation(a, "tanh")


def sigmoid(a: Tensor) -> Tensor:
    """sigmoid"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation(a, "sigmoid")


def log_sigmoid(a: Tensor) -> Tensor:
    """log_sigmoid"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation(a, "log_sigmoid")


def sin(a: Tensor) -> Tensor:
    """sin"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation(a, "sin")


def cos(a: Tensor) -> Tensor:
    """cos"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation(a, "cos")


def ceil(a: Tensor) -> Tensor:
    """ceil"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation(a, "ceil")


def floor(a: Tensor) -> Tensor:
    """floor"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation(a, "floor")


# noinspection PyShadowingBuiltins
def round(a: Tensor) -> Tensor:
    """round"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation(a, "round")


def relu(a: Tensor) -> Tensor:
    """relu"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation(a, "relu")


def elu(a: Tensor) -> Tensor:
    """elu"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation(a, "elu")


def selu(a: Tensor) -> Tensor:
    """selu"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation(a, "selu")


def silu(a: Tensor) -> Tensor:
    """silu / swish.

    The SiLU activation function was introduced in "Gaussian Error Linear Units
    (GELUs)" [Hendrycks et al. 2016](https://arxiv.org/abs/1606.08415) and
    "Sigmoid-Weighted Linear Units for Neural Network Function Approximation in
    Reinforcement Learning"
    [Elfwing et al. 2017](https://arxiv.org/abs/1702.03118) and was independently
    discovered (and called swish) in "Searching for Activation Functions"
    [Ramachandran et al. 2017](https://arxiv.org/abs/1710.05941)
    """
    # noinspection PyProtectedMember
    return a._raw_backend.activation(a, "silu")


swish = silu  # alias


def softmax(a: Tensor, *, axis: Dim) -> Tensor:
    """softmax"""
    # noinspection PyProtectedMember
    return a._raw_backend.softmax(a, axis=axis)


def log_softmax(a: Tensor, *, axis: Dim) -> Tensor:
    """log_softmax"""
    # noinspection PyProtectedMember
    return a._raw_backend.log_softmax(a, axis=axis)
