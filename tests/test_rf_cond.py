"""
RETURNN frontend (returnn.frontend) tests
"""

from __future__ import annotations
import _setup_test_env  # noqa
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, TensorDict, batch_dim
from rf_utils import run_model


def test_cond():
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
            self.linear_true = rf.Linear(in_dim, out_dim)
            self.linear_false = rf.Linear(in_dim, out_dim)

        def __call__(self, x: Tensor) -> Tensor:
            return rf.cond(
                pred=batch_dim.get_dim_value_tensor() % 2 == 0,
                true_fn=lambda: self.linear_true(x),
                false_fn=lambda: self.linear_false(x),
            )

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, time_dim, out_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_cond_via_time_even():
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
            self.linear_true = rf.Linear(in_dim, out_dim)
            self.linear_false = rf.Linear(in_dim, out_dim)

        def __call__(self, x: Tensor) -> Tensor:
            return rf.cond(
                pred=time_dim.get_dim_value_tensor() % 2 == 0,
                true_fn=lambda: self.linear_true(x),
                false_fn=lambda: self.linear_false(x),
            )

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, time_dim, out_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, dyn_dim_max_sizes={time_dim: 5})
    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, dyn_dim_max_sizes={time_dim: 6})


def test_cond_shared_params():
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
            self.linear = rf.Linear(in_dim, out_dim)

        def __call__(self, x: Tensor) -> Tensor:
            return rf.cond(
                pred=time_dim.get_dim_value_tensor() % 2 == 0,
                true_fn=lambda: self.linear(x),
                false_fn=lambda: self.linear(x * 2.0),
            )

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, time_dim, out_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, dyn_dim_max_sizes={time_dim: 5})
    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, dyn_dim_max_sizes={time_dim: 6})


def test_cond_twice_shared_params():
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
            self.pre_linear = rf.Linear(in_dim, out_dim)
            self.linear_true = rf.Linear(out_dim, out_dim)
            self.linear_false = rf.Linear(out_dim, out_dim)

        def __call__(self, x: Tensor) -> Tensor:
            x = self.pre_linear(x)
            x = rf.cond(
                pred=time_dim.get_dim_value_tensor() % 2 == 0,
                true_fn=lambda: self.linear_true(x),
                false_fn=lambda: self.linear_false(x),
            )
            x = rf.cond(
                pred=time_dim.get_dim_value_tensor() % 2 == 1,
                true_fn=lambda: self.linear_true(x),
                false_fn=lambda: self.linear_false(x),
            )
            return x

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, time_dim, out_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, dyn_dim_max_sizes={time_dim: 5})
    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, dyn_dim_max_sizes={time_dim: 6})


# TODO some more from RETURNN-common, like:
#  - test_cond_random
#  - test_cond_new_axis
#  - test_cond_dim (https://github.com/rwth-i6/returnn/pull/1262)
#  - test_cond_multiple_outputs
