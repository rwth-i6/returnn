"""
RETURNN frontend (returnn.frontend) tests
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy
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

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, test_single_batch_entry=True)


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

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, test_single_batch_entry=True)


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

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, test_single_batch_entry=True)


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

    for t in [7, 8, 9]:
        run_model(
            extern_data,
            lambda *, epoch, step: _Net(),
            _forward_step,
            dyn_dim_max_sizes={time_dim: t},
            test_single_batch_entry=True,
        )


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

    for t in [7, 9]:
        run_model(
            extern_data,
            lambda *, epoch, step: _Net(),
            _forward_step,
            dyn_dim_max_sizes={time_dim: t},
            test_single_batch_entry=True,
        )


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

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, test_single_batch_entry=True)


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

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, test_single_batch_entry=True)


def test_conv1d_depthwise_cuda_triton_path():
    """on CUDA the depthwise conv runs through the Triton kernel in the (batch, time, feat) layout, values as torch"""
    import unittest
    from unittest import mock
    import torch

    if not torch.cuda.is_available():
        raise unittest.SkipTest("needs CUDA")
    try:
        import triton  # noqa
    except ImportError as exc:
        raise unittest.SkipTest(f"triton not available ({exc})")
    rf.select_backend_torch()
    batch, time, feat = Dim(2, name="batch"), Dim(23, name="time"), Dim(6, name="feat")
    gen = torch.Generator().manual_seed(5)
    for filter_size, padding in [(4, "same"), (5, "same"), (5, "valid")]:
        with rf.set_default_device_ctx("cuda"):
            rf.set_random_seed(3)
            conv = rf.Conv1d(feat, feat, filter_size=filter_size, groups=feat.dimension, padding=padding)
        x = Tensor("x", dims=[batch, time, feat], dtype="float32")
        x.raw_tensor = torch.randn(2, 23, 6, generator=gen).cuda()
        ref = torch.nn.functional.conv1d(
            x.raw_tensor.transpose(1, 2), conv.filter.raw_tensor, conv.bias.raw_tensor, padding=padding, groups=6
        ).transpose(1, 2)
        with mock.patch.object(torch.nn.functional, "conv1d", side_effect=AssertionError("torch conv1d fallback")):
            out, out_time = conv(x, in_spatial_dim=time)
        assert out.dims == (batch, out_time, feat), (filter_size, padding, out.dims)
        torch.testing.assert_close(out.raw_tensor, ref, rtol=1e-5, atol=1e-5)
    with rf.set_default_device_ctx("cuda"):
        conv = rf.Conv1d(feat, feat, filter_size=2, groups=feat.dimension, padding="same")
    with torch.no_grad():
        conv.filter.raw_tensor.copy_(torch.tensor([1.003, -1.0]).expand(6, 1, 2))
        conv.bias.raw_tensor.zero_()
    for dtype in ("float32", "bfloat16"):
        x_amp = Tensor("x", dims=[batch, time, feat], dtype=dtype)
        x_amp.raw_tensor = torch.ones(2, 23, 6).to("cuda", getattr(torch, dtype))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref = torch.nn.functional.conv1d(
                x_amp.raw_tensor.transpose(1, 2), conv.filter.raw_tensor, conv.bias.raw_tensor, padding="same", groups=6
            ).transpose(1, 2)
            with mock.patch.object(torch.nn.functional, "conv1d", side_effect=AssertionError("torch conv1d fallback")):
                out, _ = conv(x_amp, in_spatial_dim=time)
        assert out.raw_tensor.dtype == ref.dtype == torch.bfloat16, (dtype, out.raw_tensor.dtype)
        assert float(out.raw_tensor[0, 0, 0]) == 0.0, "the autocast path rounds the filter to bfloat16"
        torch.testing.assert_close(out.raw_tensor, ref, rtol=0, atol=0)
    x_wide = Tensor("x", dims=[batch, time, feat], dtype="float64")
    x_wide.raw_tensor = torch.full((2, 23, 6), 1e50, dtype=torch.float64, device="cuda")
    for param in (conv.filter, conv.bias):
        param.raw_tensor.data = param.raw_tensor.data.double()
        param.dtype = "float64"
    out, out_time = conv(x_wide, in_spatial_dim=time)
    ref = torch.nn.functional.conv1d(
        x_wide.raw_tensor.transpose(1, 2), conv.filter.raw_tensor, conv.bias.raw_tensor, padding="same", groups=6
    ).transpose(1, 2)
    out_raw = out.copy_transpose([batch, out_time, feat]).raw_tensor
    assert out_raw.dtype == torch.float64
    torch.testing.assert_close(out_raw, ref)


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

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, test_single_batch_entry=True)


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

    for t in [7, 8, 9]:
        run_model(
            extern_data,
            lambda *, epoch, step: _Net(),
            _forward_step,
            dyn_dim_max_sizes={time_dim: t},
            test_single_batch_entry=True,
        )


def test_maxpool1d_stride_padding_same():
    from returnn.util.math import ceil_div

    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    pool_size = 4
    strides = 3
    extern_data = TensorDict({"data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32")})

    # noinspection PyShadowingNames
    def _forward_step(*, extern_data: TensorDict, **_kwargs):
        x = extern_data["data"]
        out, out_spatial_dim = rf.max_pool1d(
            x, pool_size=pool_size, strides=strides, padding="same", in_spatial_dim=time_dim
        )
        out.mark_as_default_output(shape=(batch_dim, out_spatial_dim, in_dim))

    for t in [7, 8, 9]:
        out_d = run_model(
            extern_data,
            lambda **_kwargs: rf.Module(),
            _forward_step,
            dyn_dim_max_sizes={time_dim: t},
            test_single_batch_entry=True,
        )
        out = out_d["output"]
        out_spatial_dim = out.dims[1]
        assert out_spatial_dim.get_dim_value() == ceil_div(t, strides)


def test_maxpool1d_stride_border_cond():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: rf.Tensor, *, in_spatial_dim: Dim) -> Tuple[Tensor, Dim]:
            return rf.max_pool1d(x, pool_size=6, strides=3, padding="valid", in_spatial_dim=in_spatial_dim)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out, out_spatial_dim = model(extern_data["data"], in_spatial_dim=time_dim)
        out.mark_as_default_output(shape=(batch_dim, out_spatial_dim, in_dim))

    out = run_model(
        extern_data,
        lambda *, epoch, step: _Net(),
        _forward_step,
        dyn_dim_max_sizes={time_dim: 9},
        # 2 means we get: ceildiv(2 - 6 + 1, 3) == -1.
        # So this checks that there is correctly a relu on the size.
        dyn_dim_min_sizes={time_dim: 2},
        # Note: Currently not the single batch test because there is another problem with RF PT pool,
        # which does not correctly handle this case. We get:
        #   RuntimeError: max_pool1d() Invalid computed output size: -1
        test_single_batch_entry=False,
    )
    out = out["output"]
    (out_spatial_dim,) = out.get_dyn_size_tags()
    assert isinstance(out_spatial_dim, Dim)
    out_sizes = out_spatial_dim.dyn_size_ext.raw_tensor
    print("out sizes:", out_sizes)
    assert isinstance(out_sizes, numpy.ndarray)
    assert min(out_sizes) != max(out_sizes)  # not all the same
    assert min(out_sizes) == 0


def test_maxpool1d_stride1_padding_same():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: rf.Tensor, *, in_spatial_dim: Dim) -> Tuple[Tensor, Dim]:
            return rf.max_pool1d(x, pool_size=3, strides=1, padding="same", in_spatial_dim=in_spatial_dim)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out, out_spatial_dim = model(extern_data["data"], in_spatial_dim=time_dim)
        out.mark_as_default_output(shape=(batch_dim, out_spatial_dim, in_dim))

    for t in [7, 8, 9]:
        run_model(
            extern_data,
            lambda *, epoch, step: _Net(),
            _forward_step,
            dyn_dim_max_sizes={time_dim: t},
            test_single_batch_entry=True,
        )


def test_avgpool1d_stride1_padding_same():
    time_dim = Dim(10, name="time")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: rf.Tensor, *, in_spatial_dim: Dim) -> Tuple[Tensor, Dim]:
            return rf.pool1d(x, mode="avg", pool_size=3, strides=1, padding="same", in_spatial_dim=in_spatial_dim)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out, _ = model(extern_data["data"], in_spatial_dim=time_dim)
        out.mark_as_default_output(shape=[batch_dim, time_dim])

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, test_single_batch_entry=True)


def test_avgpool1d_stride_padding_same():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: rf.Tensor, *, in_spatial_dim: Dim) -> Tuple[Tensor, Dim]:
            return rf.pool1d(x, mode="avg", pool_size=4, strides=3, padding="same", in_spatial_dim=in_spatial_dim)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out, out_spatial_dim = model(extern_data["data"], in_spatial_dim=time_dim)
        out.mark_as_default_output(shape=[batch_dim, out_spatial_dim])

    for t in [7, 8, 9]:
        run_model(
            extern_data,
            lambda *, epoch, step: _Net(),
            _forward_step,
            dyn_dim_max_sizes={time_dim: t},
            test_single_batch_entry=True,
        )


def test_make_conv_out_spatial_dims_after_pad():
    rf.select_backend_torch()
    batch_dim = Dim(3, name="batch")
    time1_dim = Dim(rf.convert_to_tensor([5, 6, 7], dims=[batch_dim], name="time1"))
    time2_dim = Dim(rf.convert_to_tensor([20, 21, 22], dims=[batch_dim], name="time2"))
    time3_dim = Dim(rf.convert_to_tensor([10, 11, 12], dims=[batch_dim], name="time3"))
    in_spatial_dim = time1_dim + time2_dim + time3_dim
    filter_size = 17
    (out_spatial_dim,) = rf.make_conv_out_spatial_dims(
        [in_spatial_dim], filter_size=filter_size, strides=1, padding="valid"
    )
    print("out_spatial_dim:", out_spatial_dim)
    sizes = out_spatial_dim.dyn_size.numpy().tolist()
    print("sizes:", sizes)
    expected_sizes = (time1_dim.dyn_size + time2_dim.dyn_size + time3_dim.dyn_size - (filter_size - 1)).numpy().tolist()
    print("expected_sizes:", expected_sizes)
    assert sizes == expected_sizes


def test_transposed_conv1d():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    out_dim = Dim(13, name="out")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __init__(self, filter_size: int, strides: Optional[int], padding: str):
            super().__init__()
            self.conv = rf.TransposedConv1d(in_dim, out_dim, filter_size, strides=strides, padding=padding)

        def __call__(self, x: rf.Tensor) -> Tuple[Tensor, Dim]:
            return self.conv(x, in_spatial_dim=time_dim)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out, dim = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, dim, out_dim))

    for fs, s, p, ts in ((4, 3, "valid", [7, 8, 9]), (3, 3, "valid", [7, 8, 9]), (2, None, "valid", [7])):
        for t in ts:
            run_model(
                extern_data, lambda *, epoch, step: _Net(fs, s, p), _forward_step, dyn_dim_max_sizes={time_dim: t}
            )
