"""
High-level backend for RETURNN layers
"""

from __future__ import annotations
from typing import Union, Sequence

from returnn.util.basic import NotSpecified
from returnn.tensor import Tensor, Dim

# noinspection PyProtectedMember
from returnn.frontend._backend import Backend


# Ignore this warning until we really expect that we implemented everything.
# noinspection PyAbstractClass
class ReturnnLayersBackend(Backend):
    """
    RETURNN layers backend (using TF), where raw_tensor represents a RETURNN layer
    """

    is_tensorflow = True

    @staticmethod
    def reduce(
        source: Tensor, *, mode: str, axis: Union[Dim, Sequence[Dim]], use_time_mask: bool = NotSpecified
    ) -> Tensor:
        """Reduce"""
        assert mode in Backend._AllowedReduceModes
        return source  # TODO
