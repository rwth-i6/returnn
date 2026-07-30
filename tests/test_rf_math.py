"""
RETURNN frontend (returnn.frontend) tests
"""

from __future__ import annotations

import _setup_test_env  # noqa

import sys
import unittest

from returnn.util import better_exchook
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, TensorDict, batch_dim
from rf_utils import run_model


def test_neg():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor) -> Tensor:
            return -x

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, time_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_compare_bc():
    beam_dim = Dim(3, name="beam")
    in_dim = Dim(7, name="in")
    extern_data = TensorDict({"idx": Tensor("idx", [batch_dim, beam_dim], dtype="int32", sparse_dim=in_dim)})

    # noinspection PyShadowingNames,PyUnusedLocal
    def _forward_step(*, model: rf.Module, extern_data: TensorDict):
        idx = extern_data["idx"]
        cond = rf.compare_bc(idx, "!=", rf.range_over_dim(in_dim))
        cond.mark_as_default_output(shape=(batch_dim, beam_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: rf.Module(), _forward_step)


def test_logical_or():
    beam_dim = Dim(3, name="beam")
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "a": Tensor("a", [batch_dim, beam_dim], dtype="bool"),
            "b": Tensor("b", [batch_dim, beam_dim, in_dim], dtype="bool"),
        }
    )

    # noinspection PyShadowingNames,PyUnusedLocal
    def _forward_step(*, model: rf.Module, extern_data: TensorDict):
        a, b = extern_data["a"], extern_data["b"]
        cond = a | b
        cond.mark_as_default_output(shape=(batch_dim, beam_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: rf.Module(), _forward_step)


def test_squared_difference():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "a": Tensor("a", [batch_dim, time_dim, in_dim], dtype="float32"),
            "b": Tensor("b", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, a: Tensor, b: Tensor) -> Tensor:
            return rf.squared_difference(a, b)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["a"], extern_data["b"])
        out.mark_as_default_output(shape=(batch_dim, time_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_log_add_exp():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "a": Tensor("a", [batch_dim, time_dim, in_dim], dtype="float32"),
            "b": Tensor("b", [in_dim], dtype="float32"),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, extern_data: TensorDict, **_):
        out = rf.log_add_exp(extern_data["a"], extern_data["b"])
        out.mark_as_default_output(shape=(batch_dim, time_dim, in_dim))

    run_model(extern_data, lambda **_: rf.Module(), _forward_step)


def test_abs_complex():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="complex64"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor) -> Tensor:
            return rf.abs(x)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, time_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_relu():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor) -> Tensor:
            return rf.relu(x)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, time_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_log_sigmoid():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, model: rf.Module, extern_data: TensorDict):
        out = rf.log_sigmoid(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, time_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: rf.Module(), _forward_step)


def test_cumsum():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim], dtype="int32"),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, model: rf.Module, extern_data: TensorDict):
        out = rf.cumsum(extern_data["data"], spatial_dim=time_dim)
        out.mark_as_default_output(shape=(batch_dim, time_dim))

    run_model(extern_data, lambda *, epoch, step: rf.Module(), _forward_step)


def test_softmax_fully_masked_zero_len_seq():
    """
    Bound-shape regime (:func:`rf.is_static_traceable`, e.g. ``torch_cuda_graph``):
    batches are padded with zero-length filler seqs, whose softmax rows are fully masked.
    The stable softmax gives NaN there ((-inf) - (-inf)) -- the backend defines those rows
    as 0 (log_softmax: -inf) instead. Without that, downstream masked ops absorb the NaNs
    via mask-multiply (NaN * 0 = NaN) and poison the whole batch (e.g. batch norm stats).
    """

    import torch
    from returnn.tensor import Tensor, Dim

    batch_dim = Dim(2, name="batch")
    time_dim = Dim(
        Tensor("time", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([5, 0], dtype=torch.int32)),
        # Static tracing requires a declared capacity (get_dim_value = the capacity there).
        capacity=5,
    )
    feat_dim = Dim(4, name="feat")
    x = Tensor("x", dims=[batch_dim, time_dim, feat_dim], dtype="float32")
    x.raw_tensor = torch.randn(2, 5, 4, generator=torch.Generator().manual_seed(42))
    with rf.set_static_traceable_ctx():
        out = rf.softmax(x, axis=time_dim)
        out_log = rf.log_softmax(x, axis=time_dim)
    raw = out.copy_transpose((batch_dim, time_dim, feat_dim)).raw_tensor
    assert torch.isfinite(raw).all()
    assert (raw[1] == 0).all()  # the zero-length seq: fully-masked rows -> all 0
    sums = raw[0].sum(dim=0)  # valid seq: proper distribution over the (masked) time axis
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    raw_log = out_log.copy_transpose((batch_dim, time_dim, feat_dim)).raw_tensor
    assert not torch.isnan(raw_log).any()
    assert (raw_log[1] == -torch.inf).all()  # log space: fully-masked rows -> -inf (log 0)


if __name__ == "__main__":
    better_exchook.install()
    if len(sys.argv) <= 1:
        for k, v in sorted(globals().items()):
            if k.startswith("test_"):
                print("-" * 40)
                print("Executing: %s" % k)
                try:
                    v()
                except unittest.SkipTest as exc:
                    print("SkipTest:", exc)
                print("-" * 40)
        print("Finished all tests.")
    else:
        assert len(sys.argv) >= 2
        for arg in sys.argv[1:]:
            print("Executing: %s" % arg)
            if arg in globals():
                globals()[arg]()  # assume function and execute
            else:
                eval(arg)  # assume Python code and execute
