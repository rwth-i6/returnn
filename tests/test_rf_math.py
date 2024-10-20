"""
RETURNN frontend (returnn.frontend) tests
"""

from __future__ import annotations
import _setup_test_env  # noqa
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


def test_eq():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "a": Tensor("a", [batch_dim, time_dim, in_dim], dtype="int32"),
            "b": Tensor("b", [batch_dim, in_dim], dtype="int32"),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, extern_data: TensorDict, **_kwargs):
        out = extern_data["a"] == extern_data["b"]
        out.mark_as_default_output(shape=(batch_dim, time_dim, in_dim))

    run_model(extern_data, lambda **_kwargs: rf.Module(), _forward_step)


def test_neq():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "a": Tensor("a", [batch_dim, time_dim, in_dim], dtype="int32"),
            "b": Tensor("b", [batch_dim, in_dim], dtype="int32"),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, extern_data: TensorDict, **_kwargs):
        out = extern_data["a"] != extern_data["b"]
        out.mark_as_default_output(shape=(batch_dim, time_dim, in_dim))

    run_model(extern_data, lambda **_kwargs: rf.Module(), _forward_step)


def test_neq_broadcast_exception():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    other_time_dim = Dim(Tensor("other_time", [batch_dim], dtype="int32"))
    extern_data = TensorDict(
        {
            "a": Tensor("a", [batch_dim, time_dim], dtype="int32"),
            "b": Tensor("b", [batch_dim, other_time_dim], dtype="int32"),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, extern_data: TensorDict, **_kwargs):
        a, b = extern_data["a"], extern_data["b"]
        try:
            _ = a != b
        except ValueError as e:
            print("Got exception:", e)
            assert "require explicit allow_broadcast_all_sources=True" in str(e) or "require broadcasting to" in str(
                e
            ), f"exception unexpected: {e}"
        else:
            raise Exception("Expected ValueError")
        a.mark_as_default_output(shape=(batch_dim, time_dim))

    run_model(extern_data, lambda **_kwargs: rf.Module(), _forward_step)


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
