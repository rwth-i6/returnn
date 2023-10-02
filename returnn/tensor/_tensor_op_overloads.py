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

    # Note that all those ops have native implementations as well,
    # so keep the logic in sync.

    # --- comparisons

    def __eq__(self: Tensor, other: Union[_rf_types.RawTensorTypes, Tensor]) -> Union[Tensor, bool]:
        # When comparing to some other invalid type, return False, not a Tensor.
        # This is to allow easy equality checks with other random objects.
        # See for example here: https://github.com/rwth-i6/returnn/pull/1284
        if self.raw_tensor is None:
            # The other op overloads would actually raise some exception in this case.
            # However, here just return False.
            return False

        import returnn.frontend as rf

        valid_types = (rf.Tensor, self._raw_backend.RawTensorType) + tuple(rf.RawTensorTypes.__args__)
        if isinstance(other, valid_types):
            return _rf().compare(self, "==", other)
        return False

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

    # --- math binary ops

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

    def __and__(self: Tensor, other: Union[_rf_types.RawTensorTypes, Tensor]) -> Tensor:
        return _rf().combine(self, "logical_and", other)

    def __rand__(self: Tensor, other: Union[_rf_types.RawTensorTypes, Tensor]) -> Tensor:
        return _rf().combine(other, "logical_and", self)

    def __or__(self: Tensor, other: Union[_rf_types.RawTensorTypes, Tensor]) -> Tensor:
        return _rf().combine(self, "logical_or", other)

    def __ror__(self: Tensor, other: Union[_rf_types.RawTensorTypes, Tensor]) -> Tensor:
        return _rf().combine(other, "logical_or", self)

    # --- math unary ops

    def __neg__(self: Tensor):  # -x
        return _rf().neg(self)

    def __invert__(self: Tensor):  # ~x, but for bool, treat is as logical_not, and otherwise not supported
        return _rf().logical_not(self)

    def __abs__(self: Tensor):
        return _rf().abs(self)

    def __ceil__(self: Tensor):
        return _rf().ceil(self)

    def __floor__(self: Tensor):
        return _rf().floor(self)


def _rf():
    import returnn.frontend as rf

    return rf
