"""
RETURNN frontend (returnn.frontend) tests
"""

from __future__ import annotations
import _setup_test_env  # noqa
import numpy
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


def test_scaled_gradient_tensor_scale():
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

        # a scalar, and one over a subset of the dims, broadcast like an elementwise op
        scale_scalar = rf.convert_to_tensor(-0.5)
        scale_per_feat = rf.cast(rf.range_over_dim(in_dim), "float32") - 3.0
        out = rf.scaled_gradient(rf.scaled_gradient(data, scale=scale_scalar), scale=scale_per_feat)
        out.mark_as_default_output(shape=(batch_dim, time_dim, in_dim))

        grad = rf.gradient(rf.reduce_sum(out, axis=out.dims, use_mask=False), data)
        grad.mark_as_output("grad")

    out = run_model(extern_data, lambda *, epoch, step: rf.Module(), _forward_step)
    rows = out["grad"].copy_transpose([batch_dim, time_dim, in_dim]).raw_tensor.reshape(-1, 7)
    rows = rows[rows.any(axis=1)]  # the padded frames are zeroed
    assert len(rows) and (rows == (numpy.arange(7) - 3.0) * -0.5).all(), rows
