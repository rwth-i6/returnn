"""
RETURNN frontend (returnn.frontend) tests
"""

from __future__ import annotations
import _setup_test_env  # noqa
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim
from rf_utils import run_model


def test_simple_net_linear():
    batch_dim = Dim(Tensor("batch", [], dtype="int32"))
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim, out_dim = Dim(7, name="in"), Dim(13, name="out")
    extern_data = Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32")

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = rf.Linear(in_dim, out_dim)
            self.layer2 = rf.Linear(out_dim, out_dim)

        def __call__(self, x: Tensor) -> Tensor:
            """
            Forward
            """
            x = rf.relu(self.layer1(x))
            x = self.layer2(x)
            return x

    run_model(_Net, extern_data)
