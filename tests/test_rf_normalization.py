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


def _moments_test_tensors():
    """
    :return: (batch dim, feature dim, dynamic time dim, static time dim, raw values with a mean far above the stddev)
    """
    import torch

    rf.select_backend_torch()
    batch = Dim(2, name="batch")
    feat = Dim(3, name="feat")
    time_sizes = Tensor("time_size", dims=[batch], dtype="int32", raw_tensor=torch.tensor([5, 3], dtype=torch.int32))
    dyn_time = Dim(time_sizes, name="time")
    static_time = Dim(5, name="static_time")
    torch.manual_seed(42)
    # mean 1e3 with stddev 0.1: E[x^2] - E[x]^2 cancels to noise in float32, the two-pass form does not
    raw = 1.0e3 + torch.randn(2, 5, 3) * 0.1
    return batch, feat, dyn_time, static_time, raw


def test_moments_distributed_matches_local():
    """Without a process group the distributed moments must match the local ones, also when the mean dominates."""
    import torch

    batch, feat, dyn_time, static_time, raw = _moments_test_tensors()
    for time_dim, use_mask in ((dyn_time, True), (dyn_time, False), (static_time, True), (static_time, False)):
        x = Tensor("x", dims=[batch, time_dim, feat], dtype="float32", raw_tensor=raw)
        mean, variance = rf.moments(x, axis=[batch, time_dim], use_mask=use_mask, distributed=True)
        ref_mean, ref_variance = rf.moments(x, axis=[batch, time_dim], use_mask=use_mask)
        torch.testing.assert_close(mean.raw_tensor, ref_mean.raw_tensor, rtol=1e-6, atol=1e-3)
        torch.testing.assert_close(variance.raw_tensor, ref_variance.raw_tensor, rtol=1e-2, atol=1e-4)
        assert float(variance.raw_tensor.min()) > 0.0, (time_dim, use_mask, variance.raw_tensor)


def test_batch_norm_distributed_keeps_use_mask():
    """Distributed BatchNorm must normalize over the same frames as the local one, masked or not."""
    import torch

    batch, feat, dyn_time, _static_time, _raw = _moments_test_tensors()
    torch.manual_seed(3)
    raw = torch.randn(2, 5, 3)
    # padding of the shorter sequence, far off so that masked and unmasked statistics clearly differ
    raw[1, 3:] = 50.0
    for use_mask in (True, False):
        rf.init_train_step_run_ctx(train_flag=True, step=0, epoch=1)
        x = Tensor("x", dims=[batch, dyn_time, feat], dtype="float32", raw_tensor=raw)
        local = rf.BatchNorm(feat, use_mask=use_mask)
        distributed = rf.BatchNorm(feat, use_mask=use_mask, distributed=True)
        out_local = local(x)
        out_distributed = distributed(x)
        torch.testing.assert_close(
            out_distributed.copy_compatible_to_dims_raw([batch, dyn_time, feat]),
            out_local.copy_compatible_to_dims_raw([batch, dyn_time, feat]),
            rtol=1e-5,
            atol=1e-5,
        )
