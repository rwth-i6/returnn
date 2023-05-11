"""
RETURNN frontend (returnn.frontend) tests
"""

from __future__ import annotations
from typing import Tuple
import _setup_test_env  # noqa
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, TensorDict, batch_dim
from rf_utils import run_model


def test_while_loop():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor) -> Tensor:
            def _cond(s: Tuple[Tensor, Tensor]):
                t, s_ = s
                if t.raw_tensor.__class__.__module__.startswith("torch"):
                    print("**", t.raw_tensor, rf.reduce_sum(s_, axis=s_.dims).raw_tensor)
                return rf.logical_and(rf.reduce_sum(s_, axis=s_.dims) < 50, t < time_dim.get_dim_value_tensor())

            def _body(s):
                t, s_ = s
                return t + 1, s_ + rf.abs(rf.gather(x, indices=t, axis=time_dim))

            _, final_s = rf.while_loop(
                _cond,
                _body,
                initial=(rf.zeros((), dtype=rf.get_default_array_index_dtype()), rf.zeros((batch_dim, in_dim))),
            )
            return final_s

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, test_tensorflow=False)
