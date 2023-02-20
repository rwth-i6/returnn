"""
Frontend API
"""

from __future__ import annotations
from typing import Union, Sequence

from returnn.tensor import Tensor, Dim
from returnn.util.basic import NotSpecified


class Frontend:
    """
    Abstract base class for the frontend
    """

    @staticmethod
    def reduce(
        source: Tensor,
        *,
        mode: str,
        axis: Union[Dim, Sequence[Dim]],
        use_time_mask: bool = NotSpecified,
    ) -> Tensor:
        """
        Reduce the tensor along the given axis

        :param source:
        :param mode: "max", "min", "mean", "sum", "prod", "any", "all"
        :param axis:
        :param use_time_mask: if True, use the time mask (part of dim tag) to ignore padding frames
        :return: tensor with axis removed
        """
        raise NotImplementedError
