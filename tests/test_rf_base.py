"""
RETURNN frontend (returnn.frontend) tests
"""

from __future__ import annotations
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim
from rf_utils import run_model


def test_simple_net_linear():
    in_dim, out_dim = Dim(7, name="in"), Dim(13, name="out")

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.linear = rf.Linear(in_dim, out_dim)

        def __call__(self, x: Tensor) -> Tensor:
            """
            Forward
            """
            return self.linear(x)

    run_model(_Net)
