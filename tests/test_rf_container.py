"""
RETURNN frontend (returnn.frontend) tests
"""

from __future__ import annotations
import _setup_test_env  # noqa
from collections import OrderedDict
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, TensorDict, batch_dim
from rf_utils import run_model


def test_module_list():
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
            self.base_dim = Dim(3, name="linear-out")
            dims = [self.base_dim + i for i in range(4)]
            in_dims = [in_dim] + dims[:-1]
            self.out_dim = dims[-1]
            self.ls = rf.ModuleList([rf.Linear(in_dim_, out_dim_) for in_dim_, out_dim_ in zip(in_dims, dims)])

        def __call__(self, out: Tensor) -> Tensor:
            """
            Forward
            """
            for layer in self.ls:
                out = layer(out)
            return out

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, time_dim, model.out_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_module_slice_set_del():
    rf.select_backend_torch()
    base_dim = Dim(3, name="linear-out")
    dims = [base_dim + i for i in range(4)]
    in_dim = Dim(7, name="in")
    in_dims = [in_dim] + dims[:-1]
    layers = rf.ModuleList([rf.Linear(in_dim_, out_dim_) for in_dim_, out_dim_ in zip(in_dims, dims)])
    assert len(layers) == 4 and [k for k, v in layers.items()] == ["0", "1", "2", "3"]
    orig_layers = layers[:]
    assert isinstance(orig_layers, rf.ModuleList)
    assert len(orig_layers) == 4 and [k for k, v in orig_layers.items()] == ["0", "1", "2", "3"]
    del layers[2:]
    assert len(layers) == 2 and [k for k, v in layers.items()] == ["0", "1"]
    layers[:] = orig_layers
    assert len(layers) == 4 and [k for k, v in layers.items()] == ["0", "1", "2", "3"]


def test_sequential_base_case():
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
            dims = [Dim(1, name="feat1"), Dim(2, name="feat2"), Dim(3, name="feat3")]
            in_dims = [in_dim] + dims[:-1]
            self.out_dim = dims[-1]
            self.seq = rf.Sequential(rf.Linear(in_dim_, out_dim_) for in_dim_, out_dim_ in zip(in_dims, dims))

        def __call__(self, data: Tensor) -> Tensor:
            """
            Forward
            """
            seq = self.seq(data)
            return seq

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, time_dim, model.out_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_sequential_named_case():
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
            dims = [Dim(1, name="feat1"), Dim(2, name="feat2"), Dim(3, name="feat3")]
            self.out_dim = dims[-1]
            x = OrderedDict()
            x["one"] = rf.Linear(in_dim, dims[0])
            x["two"] = rf.Linear(dims[0], dims[1])
            x["three"] = rf.Linear(dims[1], dims[2])
            self.seq = rf.Sequential(x)

        def __call__(self, data: Tensor) -> Tensor:
            """
            Forward
            """
            seq = self.seq(data)
            return seq

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, time_dim, model.out_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_parameter_list():
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
            self.param_list = rf.ParameterList([rf.Parameter([in_dim]) for _ in range(3)])

        def __call__(self, data: Tensor) -> Tensor:
            """
            Forward
            """
            for param in self.param_list:
                data += param
            return data

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, time_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)
