"""
RETURNN frontend (returnn.frontend) tests
"""

from __future__ import annotations
from typing import Tuple
import _setup_test_env  # noqa
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, TensorDict, batch_dim
from rf_utils import run_model


def test_conv1d():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    out_dim = Dim(13, name="out")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            # Use some downsampling + valid padding to test dim tag math.
            self.conv = rf.Conv1d(in_dim, out_dim, 4, strides=3, padding="valid")

        def __call__(self, x: rf.Tensor) -> Tuple[Tensor, Dim]:
            return self.conv(x, in_spatial_dim=time_dim)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out, dim = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, dim, out_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_conv1d_depthwise():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    out_dim = Dim(7 * 3, name="out")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            # Use some downsampling + valid padding to test dim tag math.
            self.conv = rf.Conv1d(
                in_dim,
                out_dim,
                4,
                groups=in_dim.dimension,
                padding="valid",
            )

        def __call__(self, x: rf.Tensor) -> Tuple[Tensor, Dim]:
            x, dim = self.conv(x, in_spatial_dim=time_dim)
            return x, dim

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out, spatial_dim = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, spatial_dim, out_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)
