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


def test_mel_filterbank():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    feat_dim = Dim(10, name="mel")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, source: rf.Tensor, *, in_spatial_dim: Dim) -> Tuple[Tensor, Dim]:
            # log mel filterbank features
            source, in_spatial_dim, in_dim_ = rf.stft(
                source, in_spatial_dim=in_spatial_dim, frame_step=3, frame_length=5, fft_length=6
            )
            source = rf.abs(source) ** 2.0
            source = rf.audio.mel_filterbank(source, in_dim=in_dim_, out_dim=feat_dim, sampling_rate=16)
            return source, in_spatial_dim

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out, out_spatial_dim = model(extern_data["data"], in_spatial_dim=time_dim)
        out.mark_as_default_output(shape=(batch_dim, out_spatial_dim, feat_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)
    # Run again to specifically test the caching logic.
    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_audio_specaugment():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32", feature_dim=in_dim),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, model: rf.Module, extern_data: TensorDict):
        model  # noqa  # unused
        data = extern_data["data"]
        out = rf.audio.specaugment(data, spatial_dim=time_dim, feature_dim=in_dim, only_on_train=False)
        out.mark_as_default_output(shape=(batch_dim, time_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: rf.Module(), _forward_step)
