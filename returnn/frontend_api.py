"""
Frontend API
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, TypeVar, Generic, Any, Dict, Type, Union, Sequence, Tuple

from returnn.util.basic import NotSpecified

if TYPE_CHECKING:
    from returnn.tensor import Tensor, Dim

T = TypeVar("T")  # tf.Tensor, torch.Tensor or so


class Frontend(Generic[T]):
    """
    Abstract base class for the frontend, operating on tensor type T, i.e. :class:`Tensor[T]`.

    This class and instances do not have any state,
    and all functions are staticmethod (or classmethod).
    """

    T: Type[T]
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
    def get_shape_raw(raw_tensor: T) -> Tuple[Union[int, T]]:
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
        existing_shape = get_frontend_by_tensor_type(raw_tensor).get_known_shape_raw(raw_tensor)
        assert all(dim is not None for dim in existing_shape)
        assert shape == existing_shape

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
    from returnn.tf.frontend_low_level import TFFrontend
    from returnn.tf.frontend_layers import ReturnnLayersFrontend

    register_frontend_by_tensor_type(tf.Tensor, TFFrontend)
    # TODO returnn layer type, register_frontend_by_tensor_type for that
    global_frontend.__class__ = ReturnnLayersFrontend


def select_frontend_torch():
    """
    Selects the PyTorch (low-level) frontend.
    """
    import torch
    from returnn.torch.frontend import TorchFrontend

    register_frontend_by_tensor_type(torch.Tensor, TorchFrontend)
    global_frontend.__class__ = TorchFrontend


def get_frontend_by_tensor_type(tensor_type: Type[T]) -> Type[Frontend[T]]:
    """
    :param tensor_type:
    """
    if tensor_type not in _dispatch_table:
        # It would be registered if there was any select_engine or select_frontend_* call.
        # However, some code might not have done that, so for the common cases,
        # we do it here.
        if tensor_type.__module__.split(".")[0] == "tensorflow":
            import tensorflow as tf
            from returnn.tf.frontend_low_level import TFFrontend

            assert tensor_type is tf.Tensor
            register_frontend_by_tensor_type(tf.Tensor, TFFrontend)
        elif tensor_type.__module__.split(".")[0] == "torch":
            import torch
            from returnn.torch.frontend import TorchFrontend

            assert tensor_type is torch.Tensor
            register_frontend_by_tensor_type(torch.Tensor, TorchFrontend)
        else:
            raise Exception(f"unknown tensor type {tensor_type}")
    return _dispatch_table[tensor_type]


def register_frontend_by_tensor_type(tensor_type: Type[T], frontend: Type[Frontend[T]]):
    """
    :param tensor_type:
    :param frontend:
    """
    _dispatch_table[tensor_type] = frontend
