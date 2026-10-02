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


def test_moments_float32_float16_variance_overflow():
    """
    With ``rf_moments_float32``, the statistics of float16 input stay float32,
    as e.g. a variance of 90000 does not fit into float16.
    """
    import torch
    from returnn.config import Config, global_config_ctx

    rf.select_backend_torch()
    dim = Dim(2, name="dim")
    x = Tensor("x", dims=[dim], dtype="float16", raw_tensor=torch.tensor([-300.0, 300.0], dtype=torch.float16))
    with global_config_ctx(Config({"rf_moments_float32": True})):
        mean, variance = rf.moments(x, axis=dim)
    assert (mean.dtype, variance.dtype) == ("float32", "float32")
    assert (mean.raw_tensor.item(), variance.raw_tensor.item()) == (0.0, 90000.0)


def test_moments_compute_dtype_overrides_config():
    """
    An explicit ``compute_dtype`` wins over ``rf_moments_float32`` in both directions.
    """
    import torch
    from returnn.config import Config, global_config_ctx

    rf.select_backend_torch()
    dim = Dim(3, name="dim")
    x = Tensor("x", dims=[dim], dtype="bfloat16", raw_tensor=torch.tensor([1.0, 2.0, 4.0], dtype=torch.bfloat16))
    for flag, compute_dtype, want in [
        (True, None, "float32"),
        (False, None, "bfloat16"),
        (True, "bfloat16", "bfloat16"),
        (False, "float32", "float32"),
    ]:
        with global_config_ctx(Config({"rf_moments_float32": flag})):
            mean, variance = rf.moments(x, axis=dim, compute_dtype=compute_dtype)
        assert (mean.dtype, variance.dtype) == (want, want), (flag, compute_dtype, mean.dtype, variance.dtype)


def test_moments_distributed_matches_local():
    """
    Without a process group the distributed moments must match the local ones, also when the mean dominates.
    At mean 1e3 and stddev 0.1, E[x^2] - E[x]^2 cancels to noise in float32, the two-pass form does not.
    """
    import torch

    rf.select_backend_torch()
    batch = Dim(2, name="batch")
    feat = Dim(3, name="feat")
    time_sizes = Tensor("time_size", dims=[batch], dtype="int32", raw_tensor=torch.tensor([5, 3], dtype=torch.int32))
    dyn_time = Dim(time_sizes, name="time")
    static_time = Dim(5, name="static_time")
    torch.manual_seed(42)
    raw = 1.0e3 + torch.randn(2, 5, 3) * 0.1
    for time_dim, use_mask in ((dyn_time, True), (static_time, True), (static_time, False)):
        x = Tensor("x", dims=[batch, time_dim, feat], dtype="float32", raw_tensor=raw)
        mean, variance = rf.moments(x, axis=[batch, time_dim], use_mask=use_mask, distributed=True)
        ref_mean, ref_variance = rf.moments(x, axis=[batch, time_dim], use_mask=use_mask)
        torch.testing.assert_close(mean.raw_tensor, ref_mean.raw_tensor, rtol=1e-6, atol=1e-3)
        torch.testing.assert_close(variance.raw_tensor, ref_variance.raw_tensor, rtol=1e-2, atol=1e-4)
        assert float(variance.raw_tensor.min()) > 0.0, (time_dim, use_mask, variance.raw_tensor)
