"""
RETURNN frontend (returnn.frontend) tests
"""

from __future__ import annotations
from typing import Tuple
import _setup_test_env  # noqa
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, TensorDict, batch_dim
from rf_utils import run_model


def test_stft():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: rf.Tensor, *, in_spatial_dim: Dim) -> Tuple[Tensor, Dim, Dim]:
            return rf.stft(x, in_spatial_dim=in_spatial_dim, frame_step=3, frame_length=5, fft_length=6)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out, out_spatial_dim, out_dim = model(extern_data["data"], in_spatial_dim=time_dim)
        out.mark_as_default_output(shape=(batch_dim, out_spatial_dim, out_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)
