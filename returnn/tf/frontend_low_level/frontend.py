"""
Frontend for exposing TensorFlow-specific functionality.
"""

from __future__ import annotations
from typing import Union, Sequence

from returnn.util.basic import NotSpecified
from returnn.frontend_api import Frontend
from returnn.tensor import Tensor, Dim


class TFFrontend(Frontend):
    """
    TensorFlow low-level frontend, operating on tf.Tensor
    """

    @staticmethod
    def reduce(
        source: Tensor, *, mode: str, axis: Union[Dim, Sequence[Dim]], use_time_mask: bool = NotSpecified
    ) -> Tensor:
        """Reduce"""
        raise NotImplementedError  # TODO
