"""
Frontend API

The convention for the user is to do::

    from returnn import frontend as rf
"""

from __future__ import annotations
import typing
from typing import TYPE_CHECKING, Optional, TypeVar, Generic, Dict, Type, Union, Sequence
import numpy

from returnn.util.basic import NotSpecified

if TYPE_CHECKING:
    from returnn.tensor import Tensor, Dim
    from returnn._internal_frontend_api import InternalFrontend

RawTensorTypes = Union[int, float, complex, numpy.number, numpy.ndarray, bool, str]

T = TypeVar("T")  # tf.Tensor, torch.Tensor or so


class Frontend(Generic[T]):
    """
    Abstract base class for the frontend, operating on tensor type T, i.e. :class:`Tensor[T]`.

    This class and instances do not have any state,
    and all functions are staticmethod (or classmethod).
    """

    # class attribs set by derived classes
    RawTensorType: Type[T]
    is_tensorflow: bool = False  # whether this framework uses TensorFlow
    _internal_frontend: Type[InternalFrontend[T]]

    # class private attribs
    _default_int_dtype: str = "int32"
    _default_float_dtype: str = "float32"

    def __init__(self):
        raise Exception("do not instantiate this class")

    @classmethod
    def get_default_int_dtype(cls) -> str:
        """
        :return: default dtype for int
        """
        return cls._default_int_dtype

    @classmethod
    def get_default_float_dtype(cls) -> str:
        """
        :return: default dtype for float
        """
        return cls._default_float_dtype

    @staticmethod
    def convert_to_tensor(value: Union[Tensor, T, RawTensorTypes]) -> Tensor[T]:
        """
        :param value: tensor, or scalar raw tensor or some other scalar value
        :return: tensor
        """
        raise NotImplementedError

    # noinspection PyNestedDecorators
    @typing.overload
    @classmethod
    def compare(
        cls,
        a: Tensor,
        kind: str,
        b: Tensor,
        *,
        allow_broadcast_all_sources: Optional[bool] = None,
        dim_order: Optional[Sequence[Dim]] = None,
    ) -> Tensor:
        """compare with two tensors"""

    @classmethod
    def compare(
        cls,
        a: Union[Tensor, RawTensorTypes],
        kind: str,
        b: Union[Tensor, RawTensorTypes],
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
        from ._frontend import utils

        out, a, b = utils.bin_op_out_template(
            cls,
            a,
            b,
            name="compare",
            dtype="bool",
            allow_broadcast_all_sources=allow_broadcast_all_sources,
            dim_order=dim_order,
        )
        out.raw_tensor = cls._internal_frontend.compare_raw(a.raw_tensor, kind, b.raw_tensor)
        return out

    # noinspection PyNestedDecorators
    @typing.overload
    @classmethod
    def combine(
        cls,
        a: Tensor,
        kind: str,
        b: Tensor,
        *,
        allow_broadcast_all_sources: Optional[bool] = None,
        dim_order: Optional[Sequence[Dim]] = None,
    ) -> Tensor:
        """combine with two tensors"""

    @classmethod
    def combine(
        cls,
        a: Union[Tensor, RawTensorTypes],
        kind: str,
        b: Union[Tensor, RawTensorTypes],
        *,
        allow_broadcast_all_sources: Optional[bool] = None,
        dim_order: Optional[Sequence[Dim]] = None,
    ) -> Union[Tensor, RawTensorTypes]:
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
        from ._frontend import utils

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
            if utils.is_int(cls, a) and utils.is_int(cls, b):
                raise ValueError(
                    "Dividing a Tensor of type int by an integer is disallowed. Please convert the Tensor to float."
                )
        out, a, b = utils.bin_op_out_template(
            cls,
            a,
            b,
            name="combine",
            dtype=a.dtype,
            allow_broadcast_all_sources=allow_broadcast_all_sources,
            dim_order=dim_order,
        )
        out.raw_tensor = cls._internal_frontend.combine_raw(a.raw_tensor, kind, b.raw_tensor)
        return out

    @staticmethod
    def dot(a: Tensor[T], b: Tensor[T], *, reduce: Union[Dim, Sequence[Dim]]) -> Tensor[T]:
        """
        This performs a dot-product of two sources a and b.
        The underlying operation is a batched matmul (shared..., I, J) * (shared..., J, K) -> (shared..., I, K).
        The inputs a and b are transformed internally into the required shapes in the following way:
        The axis J is specified via the Dim given as 'reduce'. If multiple reduce Dims are given the corresponding axes
        are merged into one before the matmul via a reshape. All other matching Dims in a and b will be treated as
        batch dimensions ('shared...'). Dims unique to a and b define the axes I and K, respectively. (Multiple or no
        unique axes in a and b are supported too.)

        Depending on which Dims exist in a, b and reduce this dot operation can be used to compute scaling, scalar
        product, outer product, matrix-vector multiplication, matrix-matrix multiplication etc. (all possibly batched).

        :param a:
        :param b:
        :param reduce: Dims over which to perform the product, have to be present in both a and b
        :return: result of dot product, Dim order: common axes as sorted in a, unique axes of a (in order),
            unique axes of b (in order)
        """
        raise NotImplementedError

    @staticmethod
    def range_over_dim(dim: Dim) -> Tensor[T]:
        """
        :param dim:
        :return: tensor with shape [dim]
        """
        raise NotImplementedError

    @staticmethod
    def reduce(
        source: Tensor[T],
        *,
        mode: str,
        axis: Union[Dim, Sequence[Dim]],
        use_time_mask: bool = NotSpecified,
    ) -> Tensor[T]:
        """
        Reduce the tensor along the given axis

        :param source:
        :param mode: "sum", "max", "min", "mean", "logsumexp", "any", "all", "argmin", "argmax"
        :param axis:
        :param use_time_mask: if True (default), use the time mask (part of dim tag) to ignore padding frames
        :return: tensor with axis removed
        """
        raise NotImplementedError


# We use a global instance, and we modify __class__ inplace,
# such that any reference to this can be updated.
# This is exposed to the user as `returnn.frontend`.
# The __class__ assignment is done in `select_engine`.
# Use object.__new__ because we disallow creating instances of Frontend.
global_frontend = object.__new__(Frontend)

_dispatch_table = {}  # type: Dict[Type, Type[Frontend]]


def select_frontend_returnn_layers_tf():
    """
    Selects the RETURNN layers frontend (based on TF).
    """
    import tensorflow as tf

    frontend = get_frontend_by_tensor_type(tf.Tensor)  # side-effect: register it
    global_frontend.__class__ = frontend

    # TODO returnn layer type, register_frontend_by_tensor_type for that
    #   global_frontend.__class__ = ReturnnLayersFrontend


def select_frontend_torch():
    """
    Selects the PyTorch (low-level) frontend.
    """
    import torch

    frontend = get_frontend_by_tensor_type(torch.Tensor)  # side-effect: register it
    global_frontend.__class__ = frontend


def get_frontend_by_tensor_type(tensor_type: Type[T]) -> Type[Frontend[T]]:
    """
    :param tensor_type:
    """
    if tensor_type not in _dispatch_table:
        # We don't register all possible subclasses in the dispatch table.
        # Check through the MRO.
        for type_ in tensor_type.__mro__:
            if type_ in _dispatch_table:
                # Also register it for faster future lookups.
                _dispatch_table[tensor_type] = _dispatch_table[type_]
                return _dispatch_table[type_]
    if tensor_type not in _dispatch_table:
        # It would be registered if there was any select_engine or select_frontend_* call.
        # However, some code might not have done that, so for the common cases,
        # we do it here.
        if tensor_type.__module__.split(".")[0] == "tensorflow":
            from returnn.tf.frontend_low_level import TFFrontend

            frontend_type = TFFrontend
            tensor_types = _get_tensor_types_tf()
        elif tensor_type.__module__.split(".")[0] == "torch":
            from returnn.torch.frontend import TorchFrontend

            frontend_type = TorchFrontend
            tensor_types = _get_tensor_types_torch()
        else:
            raise Exception(f"unknown tensor type {tensor_type}")
        assert any(issubclass(tensor_type, type_) for type_ in tensor_types)
        for type_ in tensor_types:
            register_frontend_by_tensor_type(type_, frontend_type)
        return frontend_type
    return _dispatch_table[tensor_type]


def register_frontend_by_tensor_type(tensor_type: Type[T], frontend: Type[Frontend[T]]):
    """
    :param tensor_type:
    :param frontend:
    """
    _dispatch_table[tensor_type] = frontend


def _get_tensor_types_tf():
    """
    :return: tuple of relevant tensor types in TF.
        Note that it is not so important to cover all, as we also check issubclass as a fallback.
    """
    import tensorflow as tf

    ls = [tf.Tensor, tf.Variable]
    return tuple(ls)


def _get_tensor_types_torch():
    """
    :return: tuple of relevant tensor types in PyTorch.
        Note that it is not so important to cover all, as we also check issubclass as a fallback.
    """
    import torch

    ls = [torch.Tensor, torch.nn.Parameter]
    return tuple(ls)
