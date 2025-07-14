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
