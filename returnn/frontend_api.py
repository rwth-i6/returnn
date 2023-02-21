"""
Frontend API
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, TypeVar, Generic, Any, Dict, Type, Union, Sequence

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

    @classmethod
    def create_placeholder(cls, tensor: Tensor) -> T:
        """
        :return: tf.placeholder in TF

        This is really only for TensorFlow for the deprecated option auto_create_placeholders
        and should not be used in other backends,
        even in graph-based backends.
        Rather, the logic to create placeholders should be done elsewhere.
        """
        raise Exception(f"{cls}.create_placeholder not supported")

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
    def format_graph_output(tensor: T, *, max_depth: Optional[int] = None) -> Optional[str]:
        """
        :return: the computation graph leading to this tensor formatted.
            In eager-mode frameworks, this is not supported and returns None.
        """
        return None

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
    return _dispatch_table[tensor_type]


def register_frontend_by_tensor_type(tensor_type: Type[T], frontend: Type[Frontend[T]]):
    """
    :param tensor_type:
    :param frontend:
    """
    _dispatch_table[tensor_type] = frontend
