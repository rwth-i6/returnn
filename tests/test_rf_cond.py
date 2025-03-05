"""
RETURNN frontend (returnn.frontend) tests
"""

from __future__ import annotations
from typing import Tuple
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

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, test_single_batch_entry=False)


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

    run_model(
        extern_data,
        lambda *, epoch, step: _Net(),
        _forward_step,
        dyn_dim_max_sizes={time_dim: 5},
        test_single_batch_entry=False,
    )
    run_model(
        extern_data,
        lambda *, epoch, step: _Net(),
        _forward_step,
        dyn_dim_max_sizes={time_dim: 6},
        test_single_batch_entry=False,
    )


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

    run_model(
        extern_data,
        lambda *, epoch, step: _Net(),
        _forward_step,
        dyn_dim_max_sizes={time_dim: 5},
        test_single_batch_entry=False,
    )
    run_model(
        extern_data,
        lambda *, epoch, step: _Net(),
        _forward_step,
        dyn_dim_max_sizes={time_dim: 6},
        test_single_batch_entry=False,
    )


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

    run_model(
        extern_data,
        lambda *, epoch, step: _Net(),
        _forward_step,
        dyn_dim_max_sizes={time_dim: 5},
        test_single_batch_entry=False,
    )
    run_model(
        extern_data,
        lambda *, epoch, step: _Net(),
        _forward_step,
        dyn_dim_max_sizes={time_dim: 6},
        test_single_batch_entry=False,
    )


def test_cond_param_assign():
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
            self.param = rf.Parameter(dims=(), dtype="int32")
            self.param.initial = 2

        def __call__(self, x: Tensor) -> Tensor:
            # No extra care should be needed for graph-based backends.
            rf.cond(
                pred=time_dim.get_dim_value_tensor() % 2 == 0,
                true_fn=lambda: self.param.assign_add(3),
                false_fn=lambda: None,
            )
            return self.param

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=())

    out1 = run_model(
        extern_data,
        lambda *, epoch, step: _Net(),
        _forward_step,
        dyn_dim_max_sizes={time_dim: 5},
        test_single_batch_entry=False,
    )
    out2 = run_model(
        extern_data,
        lambda *, epoch, step: _Net(),
        _forward_step,
        dyn_dim_max_sizes={time_dim: 6},
        test_single_batch_entry=False,
    )
    assert out1["output"].raw_tensor == 2
    assert out2["output"].raw_tensor == 5


def test_cond_param_assign2():
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
            self.param = rf.Parameter(dims=(), dtype="int32")
            self.param.initial = 2

        def __call__(self, x: Tensor) -> Tensor:
            # No extra care should be needed for graph-based backends.
            rf.cond(
                pred=time_dim.get_dim_value_tensor() % 2 == 0,
                true_fn=lambda: self.param.assign_add(3),
                false_fn=lambda: self.param.assign_add(7),
            )
            return self.param

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=())

    out1 = run_model(
        extern_data,
        lambda *, epoch, step: _Net(),
        _forward_step,
        dyn_dim_max_sizes={time_dim: 5},
        test_single_batch_entry=False,
    )
    out2 = run_model(
        extern_data,
        lambda *, epoch, step: _Net(),
        _forward_step,
        dyn_dim_max_sizes={time_dim: 6},
        test_single_batch_entry=False,
    )
    assert out1["output"].raw_tensor == 9
    assert out2["output"].raw_tensor == 5


def test_cond_param_assign3():
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
            self.param = rf.Parameter(dims=(), dtype="int32")
            self.param.initial = 2

        def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
            # No extra care should be needed for graph-based backends.
            return (
                rf.cond(
                    pred=time_dim.get_dim_value_tensor() % 2 == 0,
                    true_fn=lambda: (self.param.assign_add(3), rf.convert_to_tensor(42))[-1],
                    false_fn=lambda: self.param * 3,
                ),
                self.param,
            )

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out, param = model(extern_data["data"])
        out.mark_as_default_output(shape=())
        param.mark_as_output(shape=(), name="param")

    out1 = run_model(
        extern_data,
        lambda *, epoch, step: _Net(),
        _forward_step,
        dyn_dim_max_sizes={time_dim: 5},
        test_single_batch_entry=False,
    )
    out2 = run_model(
        extern_data,
        lambda *, epoch, step: _Net(),
        _forward_step,
        dyn_dim_max_sizes={time_dim: 6},
        test_single_batch_entry=False,
    )
    assert out1["output"].raw_tensor == 6 and out1["param"].raw_tensor == 2
    assert out2["output"].raw_tensor == 42 and out2["param"].raw_tensor == 5


# TODO some more from RETURNN-common, like:
#  - test_cond_random
#  - test_cond_new_axis
#  - test_cond_dim (https://github.com/rwth-i6/returnn/pull/1262)
#  - test_cond_multiple_outputs
