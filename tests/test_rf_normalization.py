"""
RETURNN frontend (returnn.frontend) tests
"""

from __future__ import annotations
import _setup_test_env  # noqa
from collections import OrderedDict
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, TensorDict, batch_dim
from rf_utils import run_model


def test_batch_norm():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.bn = rf.BatchNorm(in_dim, use_mask=False)

        def __call__(self, out: Tensor) -> Tensor:
            """
            Forward
            """
            out = self.bn(out)
            return out

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, time_dim, in_dim))

    # Note: no test_single_batch_entry=False needed here because we currently don't check the running stats,
    # and the output currently uses the initial running stats, i.e. should be the same for all batches.
    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_batch_norm_masking():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.bn = rf.BatchNorm(in_dim, use_mask=True, track_running_stats=False)

        def __call__(self, out: Tensor) -> Tensor:
            out = self.bn(out)
            return out

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, time_dim, in_dim))

    run_model(
        extern_data,
        lambda *, epoch, step: _Net(),
        _forward_step,
        # BatchNorm by definition uses the batch dim.
        # Needed here because track_running_stats=False and thus use_current_batch_stats=True.
        test_single_batch_entry=False,
    )
