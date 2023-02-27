"""
Frontend API

The convention for the user is to do::

    from returnn import frontend as rf
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, TypeVar, Generic, Dict, Type, Union, Sequence
import numpy

from returnn.util.basic import NotSpecified
from returnn import tensor as _t

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

    RawTensorType: Type[T]
    is_tensorflow: bool = False  # whether this framework uses TensorFlow
    _internal_frontend: Type[InternalFrontend[T]]

    @staticmethod
    def convert_to_tensor(value: Union[Tensor, T, RawTensorTypes]) -> Tensor[T]:
        """
        :param value: tensor, or scalar raw tensor or some other scalar value
        :return: tensor
        """
        raise NotImplementedError

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
        a = cls.convert_to_tensor(a)
        b = cls.convert_to_tensor(b)
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
        out = _t.Tensor("compare", dims=all_dims, dtype="bool")
        a = a.copy_compatible_to(out, check_sparse=False, check_dtype=False)
        b = b.copy_compatible_to(out, check_sparse=False, check_dtype=False)
        out.raw_tensor = cls._internal_frontend.compare_raw(a.raw_tensor, kind, b.raw_tensor)
        return out

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
global_frontend = Frontend()

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
