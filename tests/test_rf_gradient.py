"""
RETURNN frontend (returnn.frontend) tests
"""

from __future__ import annotations
import _setup_test_env  # noqa
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, TensorDict, batch_dim
from rf_utils import run_model


def test_scaled_gradient():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, model: rf.Module, extern_data: TensorDict):
        model  # noqa  # unused
        data = extern_data["data"]
        rf.set_requires_gradient(data)

        out = rf.scaled_gradient(data, scale=-0.5)
        out.mark_as_default_output(shape=(batch_dim, time_dim, in_dim))

        grad = rf.gradient(rf.reduce_sum(out, axis=out.dims, use_mask=False), data)
        grad.mark_as_output("grad")

    run_model(extern_data, lambda *, epoch, step: rf.Module(), _forward_step)
