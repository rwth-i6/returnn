"""
RETURNN frontend (returnn.frontend) tests
"""

from __future__ import annotations
from typing import Tuple
import _setup_test_env  # noqa
import returnn.frontend as rf
from returnn.frontend import State
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

        def __call__(self, x: Tensor, s: State) -> Tuple[Tensor, State]:
            return self.lstm(x, s)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        first_dim_state = Dim(1, name="blstm_times_nlayers")
        state = State()
        state.h = rf.random(distribution="normal", dims=[first_dim_state, batch_dim, out_dim], dtype="float32")
        state.c = rf.random(distribution="normal", dims=[first_dim_state, batch_dim, out_dim], dtype="float32")
        out, new_state = model(extern_data["data"], state)
        out.mark_as_default_output()

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)
