"""
Internal frontend for PyTorch frontend.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch

from returnn._internal_frontend_api import InternalFrontend
from returnn.tensor import Tensor

_TT = Tensor[torch.Tensor]


# Ignore this warning until we implemented everything.
# noinspection PyAbstractClass
class TorchInternalFrontend(InternalFrontend[torch.Tensor]):
    """
    Internal frontend for PyTorch frontend.
    """

    RawTensorType = torch.Tensor

    @staticmethod
    def get_dtype_name_raw(raw_tensor: torch.Tensor) -> str:
        """
        :return: dtype of raw tensor, as string
        """
        return str(raw_tensor.dtype).replace("torch.", "")

    @staticmethod
    def get_ndim_raw(raw_tensor: torch.Tensor) -> int:
        """
        :return: ndim of raw tensor
        """
        return raw_tensor.dim()

    @staticmethod
    def get_known_shape_raw(raw_tensor: torch.Tensor) -> Tuple[Optional[int]]:
        """
        :return: shape of raw tensor; here for PyTorch the full shape is always known
        """
        return tuple(raw_tensor.size())

    @staticmethod
    def expand_dims_raw(raw_tensor: torch.Tensor, axis: int) -> torch.Tensor:
        """
        :param raw_tensor:
        :param axis: e.g. 1
        :return: raw tensor with new axis
        """
        return raw_tensor.unsqueeze(axis)

    @staticmethod
    def compare_raw(a: torch.Tensor, kind: str, b: torch.Tensor) -> torch.Tensor:
        """
        :param a:
        :param kind: "equal"|"==", "less"|"<", "less_equal"|"<=", "greater"|">", "greater_equal"|">=",
            "not_equal"|"!="|"<>"
        :param b:
        :return: a `kind` b
        """
        assert a.dim() == b.dim()
        kind = {
            "==": "eq",
            "<=": "less_equal",
            "<": "less",
            ">=": "greater_equal",
            ">": "greater",
            "!=": "not_equal",
            "<>": "not_equal",
        }.get(kind, kind)
        op = getattr(torch, kind)  # e.g. torch.equal
        return op(a, b)

    @staticmethod
    def combine_raw(a: torch.Tensor, kind: str, b: torch.Tensor) -> torch.Tensor:
        """
        :param a:
        :param kind: "add"|"+", "sub"|"-", "mul"|"*", "truediv"|"/", "floordiv"|"//", "mod"|"%", "pow"|"**",
            "max"|"maximum", "min"|"minimum", "logical_and", "logical_or", "squared_difference"
        :param b:
        :return: a `kind` b
        """
        assert a.dim() == b.dim()
        kind = {
            "+": "add",
            "-": "sub",
            "*": "mul",
            "/": "true_divide",
            "truediv": "true_divide",
            "//": "floor_divide",
            "floordiv": "floor_divide",
            "%": "remainder",  # Python-like modulo, not C-like (torch.fmod)
            "mod": "remainder",
            "**": "pow",
            "max": "maximum",
            "min": "minimum",
        }.get(kind, kind)
        op = getattr(torch, kind)  # e.g. torch.add
        return op(a, b)
