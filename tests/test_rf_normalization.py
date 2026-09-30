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


def test_moments_and_batch_norm_keep_use_mask():
    """
    With ``rf_moments_use_fixed_masking``, the local mean takes use_mask like the variance,
    and distributed BatchNorm normalizes over the same frames as the local one, masked or not.
    """
    import torch
    from returnn.config import Config, global_config_ctx

    rf.select_backend_torch()
    batch = Dim(2, name="batch")
    feat = Dim(3, name="feat")
    time_sizes = Tensor("time_size", dims=[batch], dtype="int32", raw_tensor=torch.tensor([5, 3], dtype=torch.int32))
    time_dim = Dim(time_sizes, name="time")
    torch.manual_seed(3)
    raw = torch.randn(2, 5, 3)
    # padding of the shorter sequence, far off so that masked and unmasked statistics clearly differ
    raw[1, 3:] = 50.0
    x = Tensor("x", dims=[batch, time_dim, feat], dtype="float32", raw_tensor=raw)
    with global_config_ctx(Config({"rf_moments_use_fixed_masking": True})):
        for use_mask, rows in ((True, torch.cat([raw[0], raw[1, :3]])), (False, raw.reshape(-1, 3))):
            mean, variance = rf.moments(x, axis=[batch, time_dim], use_mask=use_mask)
            torch.testing.assert_close(mean.raw_tensor, rows.mean(dim=0))
            torch.testing.assert_close(variance.raw_tensor, (rows - rows.mean(dim=0)).square().mean(dim=0))

            rf.init_train_step_run_ctx(train_flag=True, step=0, epoch=1)
            local = rf.BatchNorm(feat, use_mask=use_mask)
            distributed = rf.BatchNorm(feat, use_mask=use_mask, distributed=True)
            torch.testing.assert_close(
                distributed(x).copy_compatible_to_dims_raw([batch, time_dim, feat]),
                local(x).copy_compatible_to_dims_raw([batch, time_dim, feat]),
                rtol=1e-5,
                atol=1e-5,
            )
