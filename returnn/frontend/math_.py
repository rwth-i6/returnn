"""
Math ops
"""

from __future__ import annotations
import typing
from typing import Optional, Sequence, Union, Tuple, overload
import numpy
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from .types import RawTensorTypes as _RawTensorTypes

__all__ = [
    "compare",
    "compare_bc",
    "combine",
    "combine_bc",
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
    "ceil_divide",
    "neg",
    "reciprocal",
    "mod",
    "pow",
    "squared_difference",
    "logical_and",
    "logical_or",
    "logical_not",
    "opt_logical_or",
    "opt_logical_and",
    "log_add_exp",
    "is_finite",
    "is_infinite",
    "is_neg_infinite",
    "maximum",
    "minimum",
    "clip_by_value",
    "identity",
    "exp",
    "expm1",
    "log",
    "safe_log",
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
    "relu_square",
    "elu",
    "selu",
    "silu",
    "swish",
    "gelu",
    "softmax",
    "log_softmax",
    "gating",
    "lerp",
    "cumsum",
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


def compare_bc(
    a: Tensor,
    kind: str,
    b: Tensor,
    *,
    dim_order: Optional[Sequence[Dim]] = None,
) -> Tensor:
    """:func:`compare` with allow_broadcast_all_sources=True"""
    return compare(a, kind, b, allow_broadcast_all_sources=True, dim_order=dim_order)


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
        "max"|"maximum", "min"|"minimum", "logical_and", "logical_or", "squared_difference",
        "logaddexp"
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


def combine_bc(
    a: Tensor,
    kind: str,
    b: Tensor,
    *,
    dim_order: Optional[Sequence[Dim]] = None,
) -> Tensor:
    """:func:`combine` with allow_broadcast_all_sources=True"""
    return combine(a, kind, b, allow_broadcast_all_sources=True, dim_order=dim_order)


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


def ceil_divide(a: Tensor, b: Tensor) -> Tensor:
    """ceildiv"""
    return -(-a // b)


def neg(a: Tensor) -> Tensor:
    """neg"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation(a, "neg")


def reciprocal(a: Tensor) -> Tensor:
    """reciprocal / inverse, i.e. 1/a"""
    # noinspection PyProtectedMember
    return a._raw_backend.activation(a, "reciprocal")


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


@overload
def opt_logical_or(a: bool, b: bool) -> bool:
    """logical or"""


def opt_logical_or(a: Union[Tensor, bool], b: Union[Tensor, bool]) -> Union[Tensor, bool]:
    """logical or"""
    if isinstance(a, bool):
        if a:
            return True
        return b
    if isinstance(b, bool):
        if b:
            return True
        return a
    return combine(a, "logical_or", b)


@overload
def opt_logical_and(a: bool, b: bool) -> bool:
    """logical and"""


def opt_logical_and(a: Union[Tensor, bool], b: Union[Tensor, bool]) -> Union[Tensor, bool]:
    """logical and"""
    if isinstance(a, bool):
        if not a:
            return False
        return b
    if isinstance(b, bool):
        if not b:
            return False
        return a
    return combine(a, "logical_and", b)


def log_add_exp(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes log(exp(a) + exp(b)) in a numerically stable way.
    This is useful for log probabilities, e.g. in beam search.

    See also: func:`reduce_logsumexp`.
    """
    return combine(a, "logaddexp", b)


def is_finite(a: Tensor) -> Tensor:
    """is finite"""
    # noinspection PyProtectedMember
    return a._raw_backend.is_finite(a)


def is_infinite(a: Tensor) -> Tensor:
    """is positive or negative infinite"""
    # noinspection PyProtectedMember
    return a._raw_backend.is_infinite(a)


def is_neg_infinite(a: Tensor) -> Tensor:
    """is negative infinite"""
    # noinspection PyProtectedMember
    return a._raw_backend.is_neg_infinite(a)


def maximum(a: Tensor, b: Union[Tensor, _RawTensorTypes], *other_tensors) -> Tensor:
    """maximum"""
    if not other_tensors:
        return combine(a, "maximum", b)
    res = combine(a, "maximum", b)
    for t in other_tensors:
        res = combine(res, "maximum", t)
    return res


def minimum(a: Tensor, b: Union[Tensor, _RawTensorTypes], *other_tensors) -> Tensor:
    """minimum"""
    if not other_tensors:
        return combine(a, "minimum", b)
    res = combine(a, "minimum", b)
    for t in other_tensors:
        res = combine(res, "minimum", t)
    return res


def clip_by_value(
    x: Tensor,
    clip_value_min: Union[Tensor, _RawTensorTypes],
    clip_value_max: Union[Tensor, _RawTensorTypes],
    *,
    allow_broadcast_all_sources: bool = False,
) -> Tensor:
    """clip by value"""
    # noinspection PyProtectedMember
    return x._raw_backend.clip_by_value(
        x, clip_value_min, clip_value_max, allow_broadcast_all_sources=allow_broadcast_all_sources
    )


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


_SafeLogEps = {"float16": 6e-08, "bfloat16": 9.1835e-41, "float32": 1.4013e-45, "float64": 4.9407e-324}


def safe_log(a: Tensor, *, eps: Optional[float] = None) -> Tensor:
    """safe_log"""
    if eps is None:
        if a.dtype not in _SafeLogEps:
            raise NotImplementedError(f"safe_log not implemented for dtype {a.dtype}")
        eps = _SafeLogEps[a.dtype]
    # noinspection PyProtectedMember
    return a._raw_backend.safe_log(a, eps=eps)


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


def relu_square(x: Tensor) -> Tensor:
    """
    :return: ``relu(x) ** 2``
    """
    return square(relu(x))


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


def gelu(a: Tensor) -> Tensor:
    """
    Compute the Gaussian Error Linear Unit (GELU) activation function.
    Gaussian error linear unit (GELU) computes `x * P(X <= x)`, where `P(X) ~ N(0, 1)`.
    Ref: [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415).
    This here is without the tanh-approximation.

    :param a:
    """
    # noinspection PyProtectedMember
    return a._raw_backend.activation(a, "gelu")


def softmax(a: Tensor, *, axis: Dim, use_mask: bool = True) -> Tensor:
    """softmax"""
    # noinspection PyProtectedMember
    return a._raw_backend.softmax(a, axis=axis, use_mask=use_mask)


def log_softmax(a: Tensor, *, axis: Dim, use_mask: bool = True) -> Tensor:
    """log_softmax"""
    # noinspection PyProtectedMember
    return a._raw_backend.log_softmax(a, axis=axis, use_mask=use_mask)


def gating(
    x: Tensor, *, axis: Optional[Dim] = None, gate_func=sigmoid, act_func=identity, out_dim: Optional[Dim] = None
) -> Tuple[Tensor, Dim]:
    """
    Like in gated linear unit (GLU): https://arxiv.org/abs/1612.08083
    GLU refers also to the linear transformation before the gating -- this is why this function is not called GLU.
    GLU uses gate_func=sigmoid and act_func=identity (the defaults here).

    There are other potential gating variants you might be interested at.
    See for example: https://arxiv.org/abs/2002.05202, e.g. gate_func=gelu.
    """
    if axis is None:
        assert x.feature_dim is not None, f"gating {x}: need tensor with feature dim set, or explicit `axis`"
        axis = x.feature_dim
    assert axis.is_static() and axis.dimension % 2 == 0, f"gating {x}: need static dim, and even, got {axis}"
    if not out_dim:
        out_dim = axis.div_left(2)

    a, b = rf.split(x, axis=axis, out_dims=[out_dim, out_dim])
    return act_func(a) * gate_func(b), out_dim


def lerp(
    start: Tensor, end: Tensor, weight: Union[float, Tensor], *, allow_broadcast_all_sources: bool = False
) -> Tensor:
    """
    Linear interpolation between start and end.
    (Some backends might provide an optimized version of this.)

    :param start:
    :param end:
    :param weight: scalar or tensor
    :param allow_broadcast_all_sources:
    :return: start + weight * (end - start)
    """
    # noinspection PyProtectedMember
    return start._raw_backend.lerp(start, end, weight, allow_broadcast_all_sources=allow_broadcast_all_sources)


def cumsum(source: Tensor, *, spatial_dim: Dim) -> Tensor:
    """
    Applies cumsum.
    """
    # noinspection PyProtectedMember
    return source._raw_backend.cumsum(source, spatial_dim=spatial_dim)
