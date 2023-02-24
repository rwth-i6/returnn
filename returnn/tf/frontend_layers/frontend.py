"""
High-level frontend for RETURNN layers
"""

from __future__ import annotations
from typing import Union, Sequence

from returnn.util.basic import NotSpecified
from returnn.frontend_api import Frontend
from returnn.tensor import Tensor, Dim


# Ignore this warning until we really expect that we implemented everything.
# noinspection PyAbstractClass
class ReturnnLayersFrontend(Frontend):
    """
    RETURNN layers frontend (using TF), where raw_tensor represents a RETURNN layer
    """

    is_tensorflow = True

    @staticmethod
    def reduce(
        source: Tensor, *, mode: str, axis: Union[Dim, Sequence[Dim]], use_time_mask: bool = NotSpecified
    ) -> Tensor:
        """Reduce"""
        raise NotImplementedError  # TODO
