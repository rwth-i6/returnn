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


def test_functional_conv1d_same_padding():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    out_dim = Dim(13, name="out")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: rf.Tensor) -> Tuple[Tensor, Dim]:
            filter_size = Dim(4, name="filter_size")
            filters = rf.ones((filter_size, in_dim, out_dim), dtype=x.dtype)
            y, (out_spatial_dim,) = rf.conv(
                x,
                filter=filters,
                in_dim=in_dim,
                out_dim=out_dim,
                in_spatial_dims=[time_dim],
                filter_size=[filter_size],
                strides=1,
                padding="same",
            )
            return y, out_spatial_dim

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out, dim = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, dim, out_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_conv1d_same_padding():
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
            self.conv = rf.Conv1d(in_dim, out_dim, 4, padding="same")

        def __call__(self, x: rf.Tensor) -> Tuple[Tensor, Dim]:
            return self.conv(x, in_spatial_dim=time_dim)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out, dim = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, dim, out_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_functional_conv1d_stride_same_padding():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(1, name="in")
    out_dim = Dim(1, name="out")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: rf.Tensor) -> Tuple[Tensor, Dim]:
            x = rf.ones(x.dims, dtype=x.dtype)
            filter_size = Dim(4, name="filter_size")
            filters = rf.ones((filter_size, in_dim, out_dim), dtype=x.dtype)
            y, (out_spatial_dim,) = rf.conv(
                x,
                filter=filters,
                in_dim=in_dim,
                out_dim=out_dim,
                in_spatial_dims=[time_dim],
                filter_size=[filter_size],
                strides=3,
                padding="same",
            )
            return y, out_spatial_dim

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out, dim = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, dim, out_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, dyn_dim_max_sizes={time_dim: 7})
    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, dyn_dim_max_sizes={time_dim: 9})


def test_conv1d_stride_same_padding():
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
            self.conv = rf.Conv1d(in_dim, out_dim, 4, strides=3, padding="same")

        def __call__(self, x: rf.Tensor) -> Tuple[Tensor, Dim]:
            return self.conv(x, in_spatial_dim=time_dim)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out, dim = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, dim, out_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_conv1d_same_out():
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
            self.conv = rf.Conv1d(in_dim, in_dim, 4, padding="same")

        def __call__(self, x: rf.Tensor) -> Tensor:
            x, _ = self.conv(x, in_spatial_dim=time_dim)
            return x

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, time_dim, in_dim))

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


def test_maxpool1d_padding_valid():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: rf.Tensor, *, in_spatial_dim: Dim) -> Tuple[Tensor, Dim]:
            return rf.max_pool1d(x, pool_size=3, padding="valid", in_spatial_dim=in_spatial_dim)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out, out_spatial_dim = model(extern_data["data"], in_spatial_dim=time_dim)
        out.mark_as_default_output(shape=(batch_dim, out_spatial_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_maxpool1d_padding_same():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: rf.Tensor, *, in_spatial_dim: Dim) -> Tuple[Tensor, Dim]:
            return rf.max_pool1d(x, pool_size=3, padding="same", in_spatial_dim=in_spatial_dim)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out, out_spatial_dim = model(extern_data["data"], in_spatial_dim=time_dim)
        out.mark_as_default_output(shape=(batch_dim, out_spatial_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, dyn_dim_max_sizes={time_dim: 7})
    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, dyn_dim_max_sizes={time_dim: 9})


def test_maxpool1d_stride_padding_same():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: rf.Tensor, *, in_spatial_dim: Dim) -> Tuple[Tensor, Dim]:
            return rf.max_pool1d(x, pool_size=4, strides=3, padding="same", in_spatial_dim=in_spatial_dim)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out, out_spatial_dim = model(extern_data["data"], in_spatial_dim=time_dim)
        out.mark_as_default_output(shape=(batch_dim, out_spatial_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, dyn_dim_max_sizes={time_dim: 7})
    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, dyn_dim_max_sizes={time_dim: 9})
