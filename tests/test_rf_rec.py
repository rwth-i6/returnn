"""
RETURNN frontend (returnn.frontend) tests
"""

from __future__ import annotations
from typing import Tuple
import _setup_test_env  # noqa
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, TensorDict, batch_dim
from rf_utils import run_model


def test_lstm():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim, out_dim = Dim(7, name="in"), Dim(13, name="out")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
            "classes": Tensor("classes", [batch_dim, time_dim], dtype="int32", sparse_dim=out_dim),
        }
    )

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.lstm = rf.LSTM(in_dim, out_dim)
            # better for the test to also have some random values in the bias, not zeros
            self.lstm.bias.initial = rf.init.Glorot()

        def __call__(self, x: Tensor, *, spatial_dim: Dim, state: rf.LstmState) -> Tuple[Tensor, rf.LstmState]:
            return self.lstm(x, state=state, spatial_dim=spatial_dim)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        state = rf.LstmState(
            h=rf.random_normal(dims=[batch_dim, out_dim], dtype="float32"),
            c=rf.random_normal(dims=[batch_dim, out_dim], dtype="float32"),
        )
        out, new_state = model(extern_data["data"], state=state, spatial_dim=time_dim)
        out.mark_as_output("out", shape=(batch_dim, time_dim, out_dim))
        new_state.h.mark_as_output("h", shape=(batch_dim, out_dim))
        new_state.c.mark_as_output("c", shape=(batch_dim, out_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)
