"""
Frontend API

The convention for the user is to do::

    from returnn import frontend as rf
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, TypeVar, Generic, Any, Dict, Type, Union, Sequence, Tuple
import contextlib
import numpy

from returnn.util.basic import NotSpecified
from returnn import tensor as _t

if TYPE_CHECKING:
    from returnn.tensor import Tensor, Dim

T = TypeVar("T")  # tf.Tensor, torch.Tensor or so
RawTensorTypes = Union[int, float, complex, numpy.number, numpy.ndarray, bool, str]


class Frontend(Generic[T]):
    """
    Abstract base class for the frontend, operating on tensor type T, i.e. :class:`Tensor[T]`.

    This class and instances do not have any state,
    and all functions are staticmethod (or classmethod).
    """

    RawTensorType: Type[T]
    is_tensorflow: bool = False  # whether this framework uses TensorFlow

    @staticmethod
    def get_dtype_name_raw(raw_tensor: T) -> str:
        """
        :return: dtype of raw tensor, as string
        """
        raise NotImplementedError

    @staticmethod
    def get_ndim_raw(raw_tensor: T) -> int:
        """
        :return: ndim of raw tensor. assumes it is known
        """
        raise NotImplementedError

    @staticmethod
    def get_shape_raw(raw_tensor: T) -> T:
        """
        :return: shape of raw tensor
        """
        raise NotImplementedError

    @staticmethod
    def get_shape_tuple_raw(raw_tensor: T) -> Tuple[Union[int, T]]:
        """
        :return: shape of raw tensor. assumes that ndim is known.
            In eager frameworks, all dims are int.
        """
        raise NotImplementedError

    @staticmethod
    def get_known_shape_raw(raw_tensor: T) -> Tuple[Optional[int]]:
        """
        :return: shape of raw tensor, int for static known, None otherwise. assumes that ndim is known.
            This will not create any ops.
            In eager frameworks, all dims are known.
        """
        raise NotImplementedError

    @staticmethod
    def set_known_shape_raw(raw_tensor: T, shape: Tuple[Optional[int]]) -> None:
        """
        Sets the known shape of the raw tensor.
        This is only supported in graph-based frameworks,
        and just performs a check in eager frameworks.
        """
        # Default implementation for eager-based frameworks.
        assert all(dim is not None for dim in shape)
        rf = get_frontend_by_tensor_type(type(raw_tensor))
        existing_shape = rf.get_known_shape_raw(raw_tensor)
        assert all(dim is not None for dim in existing_shape)
        assert shape == existing_shape

    @staticmethod
    def fill_raw(shape: Union[Sequence[Union[int, T]], T], value: Union[Any, T]) -> T:
        """
        :param shape: shape
        :param value: scalar value to fill
        :return: raw tensor filled with value everywhere
        """
        raise NotImplementedError

    @staticmethod
    def compare_raw(a: T, kind: str, b: T) -> T:
        """
        :param a:
        :param kind: "equal"|"==", "less"|"<", "less_equal"|"<=", "greater"|">", "greater_equal"|">=", "not_equal"|"!="
        :param b:
        :return: a `kind` b
        """
        raise NotImplementedError

    @staticmethod
    def reshape_raw(raw_tensor: T, shape: Union[Sequence[Union[int, T]], T]) -> T:
        """
        :param raw_tensor: raw tensor
        :param shape: new shape
        :return: reshaped raw tensor
        """
        raise NotImplementedError

    @classmethod
    def squeeze_raw(cls, raw_tensor: T, axes: Sequence[int]) -> T:
        """
        :param raw_tensor: raw tensor
        :param axes: axes to squeeze
        :return: squeezed raw tensor
        """
        # Default implementation using reshape_raw.
        known_shape = cls.get_known_shape_raw(raw_tensor)
        assert all([known_shape[axis] == 1 for axis in axes])
        new_shape = [dim for a, dim in enumerate(cls.get_shape_tuple_raw(raw_tensor)) if a not in axes]
        return cls.reshape_raw(raw_tensor, new_shape)

    @staticmethod
    def transpose_raw(raw_tensor: T, perm: Sequence[int]) -> T:
        """
        :param raw_tensor: raw tensor
        :param perm: permutation
        :return: transposed raw tensor
        """
        raise NotImplementedError

    @staticmethod
    def expand_dims_raw(raw_tensor: T, axis: int) -> T:
        """
        :param raw_tensor:
        :param axis:
        :return: raw tensor with new axis
        """
        raise NotImplementedError

    @staticmethod
    def expand_raw(raw_tensor: T, axis: int, dim: Union[int, T]) -> T:
        """
        :param raw_tensor:
        :param axis: shape[axis] must be 1
        :param dim: the new dim for shape[axis]
        :return: shape[axis] expands to dim.
            in PyTorch or other frameworks which support custom strides,
            this is an efficient view and not a copy.
        """
        raise NotImplementedError

    @staticmethod
    def sequence_mask_raw(lengths: T, *, batch_major: bool = True) -> T:
        """
        Like tf.sequence_mask().

        :param lengths: shape (batch,)
        :param batch_major:
        :return: tensor mask of shape (batch,maxlen) if batch_major else (maxlen,batch) of type bool
        """
        raise NotImplementedError

    @staticmethod
    @contextlib.contextmanager
    def name_scope_raw(name: str) -> Any:
        """
        Default implementation for eager-based frameworks:
        Do nothing, tensors do not have a name.

        :param name:
        :return: context manager
        """
        # Default implementation for eager-based frameworks
        pass  # nothing to do

    @staticmethod
    @contextlib.contextmanager
    def control_dependencies_raw(dependencies: Sequence[Any]) -> Any:
        """
        Default implementation for eager-based frameworks:
        Do nothing, we expect that the dependencies are already executed.

        :param dependencies: raw tensors or ops
        :return: context manager
        """
        # Default implementation for eager-based frameworks
        yield

    @staticmethod
    def identity_with_control_dependencies_raw(raw_tensor: T, dependencies: Sequence[Any]) -> T:
        """
        Default implementation for eager-based frameworks:
        Do nothing, we expect that the dependencies are already executed.

        :param raw_tensor: raw tensor
        :param dependencies: raw tensors or ops
        :return: raw tensor
        """
        # Default implementation for eager-based frameworks
        return raw_tensor

    @staticmethod
    def create_placeholder(tensor: Tensor) -> T:
        """
        :return: tf.placeholder in TF

        This is really only for TensorFlow for the deprecated option auto_create_placeholders
        and should not be used in other backends,
        even in graph-based backends.
        Rather, the logic to create placeholders should be done elsewhere.
        """
        raise Exception("create_placeholder not supported by backend")

    @staticmethod
    def runtime_sanity_checks(tensor: Tensor) -> Any:
        """
        Checks whether the tensor.raw_tensor is consistent with the tensor metadata.

        In graph-based frameworks (TF graph), we return some operation here.
        In eager frameworks, we would not return anything but instead directly perform the checks.
        """
        # By default, we do not do any checks. This is optional for the backend.
        pass

    @staticmethod
    def is_valid_in_current_graph(tensor: Tensor) -> bool:
        """
        :return: whether the raw tensor is valid in the current graph.
            In eager-mode frameworks, this is always true -- there is no graph.
        """
        return True

    @staticmethod
    def format_graph_output(raw_tensor: T, *, max_depth: Optional[int] = None) -> str:
        """
        :return: the computation graph leading to this tensor formatted.
            In eager-mode frameworks, this is not supported and returns None.
        """
        return "<no-graph>"

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
    ) -> Tensor:
        """
        :param a:
        :param kind: "equal"|"==", "less"|"<", "less_equal"|"<=", "greater"|">", "greater_equal"|">=", "not_equal"|"!="
        :param b:
        :param allow_broadcast_all_sources: if True, it is allowed that neither a nor b has all dims of the result.
            Not needed when out_dims is specified explicitly.
        :param out_dims: shape of the result. if None, it is automatically inferred from a and b
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
            if allow_broadcast_all_sources is None:
                raise ValueError(f"compare: sources {a!r} {b!r} require explicit allow_broadcast_all_sources=True")
        out = _t.Tensor("compare", dims=all_dims, dtype="bool")
        a = a.copy_compatible_to(out, check_sparse=False, check_dtype=False)
        b = b.copy_compatible_to(out, check_sparse=False, check_dtype=False)
        out.raw_tensor = cls.compare_raw(a.raw_tensor, kind, b.raw_tensor)
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
