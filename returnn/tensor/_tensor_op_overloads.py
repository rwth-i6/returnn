"""
Tensor binary operations mixin.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .tensor import Tensor  # just for type hints; otherwise use _t.Tensor
    import returnn.frontend.types as _rf_types

from ._tensor_mixin_base import _TensorMixinBase


class _TensorOpOverloadsMixin(_TensorMixinBase):

    # --- comparisons

    def __eq__(self: Tensor, other: Union[_rf_types.RawTensorTypes, Tensor]) -> Tensor:
        return _rf().compare(self, "==", other)

    def __ne__(self: Tensor, other: Union[_rf_types.RawTensorTypes, Tensor]) -> Tensor:
        return _rf().compare(self, "!=", other)

    def __lt__(self: Tensor, other: Union[_rf_types.RawTensorTypes, Tensor]) -> Tensor:
        return _rf().compare(self, "<", other)

    def __le__(self: Tensor, other: Union[_rf_types.RawTensorTypes, Tensor]) -> Tensor:
        return _rf().compare(self, "<=", other)

    def __gt__(self: Tensor, other: Union[_rf_types.RawTensorTypes, Tensor]) -> Tensor:
        return _rf().compare(self, ">", other)

    def __ge__(self: Tensor, other: Union[_rf_types.RawTensorTypes, Tensor]) -> Tensor:
        return _rf().compare(self, ">=", other)

    # --- math binary and unary ops

    def __add__(self: Tensor, other: Union[_rf_types.RawTensorTypes, Tensor]) -> Tensor:
        return _rf().combine(self, "+", other)

    def __radd__(self: Tensor, other: Union[_rf_types.RawTensorTypes, Tensor]) -> Tensor:
        return _rf().combine(other, "+", self)

    def __sub__(self: Tensor, other: Union[_rf_types.RawTensorTypes, Tensor]) -> Tensor:
        return _rf().combine(self, "-", other)

    def __rsub__(self: Tensor, other: Union[_rf_types.RawTensorTypes, Tensor]) -> Tensor:
        return _rf().combine(other, "-", self)

    def __mul__(self: Tensor, other: Union[_rf_types.RawTensorTypes, Tensor]) -> Tensor:
        return _rf().combine(self, "*", other)

    def __rmul__(self: Tensor, other: Union[_rf_types.RawTensorTypes, Tensor]) -> Tensor:
        return _rf().combine(other, "*", self)

    def __truediv__(self: Tensor, other: Union[_rf_types.RawTensorTypes, Tensor]) -> Tensor:
        return _rf().combine(self, "/", other)

    def __rtruediv__(self: Tensor, other: Union[_rf_types.RawTensorTypes, Tensor]) -> Tensor:
        return _rf().combine(other, "/", self)

    def __floordiv__(self: Tensor, other: Union[_rf_types.RawTensorTypes, Tensor]) -> Tensor:
        return _rf().combine(self, "//", other)

    def __rfloordiv__(self: Tensor, other: Union[_rf_types.RawTensorTypes, Tensor]) -> Tensor:
        return _rf().combine(other, "//", self)

    def __mod__(self: Tensor, other: Union[_rf_types.RawTensorTypes, Tensor]) -> Tensor:
        return _rf().combine(self, "%", other)

    def __rmod__(self: Tensor, other: Union[_rf_types.RawTensorTypes, Tensor]) -> Tensor:
        return _rf().combine(other, "%", self)

    def __pow__(self: Tensor, other: Union[_rf_types.RawTensorTypes, Tensor]) -> Tensor:
        return _rf().combine(self, "**", other)

    def __rpow__(self: Tensor, other: Union[_rf_types.RawTensorTypes, Tensor]) -> Tensor:
        return _rf().combine(other, "**", self)

    def __neg__(self):  # -x
        if True:  # avoid warning: abstract base class...
            raise NotImplementedError  # TODO

    def __invert__(self):  # ~x
        if True:
            raise NotImplementedError  # TODO

    def __abs__(self):
        if True:
            raise NotImplementedError  # TODO

    def __ceil__(self):
        if True:
            raise NotImplementedError  # TODO

    def __floor__(self):
        if True:
            raise NotImplementedError  # TODO

    def __and__(self: Tensor, other: Union[_rf_types.RawTensorTypes, Tensor]) -> Tensor:
        return _rf().combine(self, "logical_and", other)

    def __rand__(self: Tensor, other: Union[_rf_types.RawTensorTypes, Tensor]) -> Tensor:
        return _rf().combine(other, "logical_and", self)

    def __or__(self: Tensor, other: Union[_rf_types.RawTensorTypes, Tensor]) -> Tensor:
        return _rf().combine(self, "logical_or", other)

    def __ror__(self: Tensor, other: Union[_rf_types.RawTensorTypes, Tensor]) -> Tensor:
        return _rf().combine(other, "logical_or", self)


def _rf():
    import returnn.frontend as rf

    return rf
