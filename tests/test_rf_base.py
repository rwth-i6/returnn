"""
RETURNN frontend (returnn.frontend) tests
"""

from __future__ import annotations
from typing import Tuple
from unittest import SkipTest
import _setup_test_env  # noqa
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, TensorDict, batch_dim
from rf_utils import run_model, run_model_torch_train


# Keep test_linear_direct and test_linear first here to have some very canonical examples.


def _setup():
    rf.select_backend_torch()  # enables some of the native optimizations


_setup()


def test_linear_direct():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim, out_dim = Dim(7, name="in"), Dim(13, name="out")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
            "classes": Tensor("classes", [batch_dim, time_dim], dtype="int32", sparse_dim=out_dim),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, model: rf.Linear, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output()

    run_model(extern_data, lambda *, epoch, step: rf.Linear(in_dim, out_dim), _forward_step)


def test_linear():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim, out_dim = Dim(7, name="in"), Dim(13, name="out")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
            "classes": Tensor("classes", [batch_dim, time_dim], dtype="int32", sparse_dim=out_dim),
        }
    )

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.linear = rf.Linear(in_dim, out_dim)

        def __call__(self, x: Tensor) -> Tensor:
            return self.linear(x)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output()

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


# Now come some tests for some base functionality.


def test_state():
    # https://github.com/rwth-i6/returnn/issues/1329
    import tree

    s = rf.LstmState(h=Tensor("h", (), "float32"), c=Tensor("c", (), "float32"))
    res = tree.map_structure(lambda x: x, s)
    assert isinstance(res, rf.LstmState)
    assert res is not s
    assert res.h is s.h and res.c is s.c


# And now more module tests.


def test_2layers():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim, hidden_dim, out_dim = Dim(7, name="in"), Dim(11, name="hidden"), Dim(13, name="out")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
            "classes": Tensor("classes", [batch_dim, time_dim], dtype="int32", sparse_dim=out_dim),
        }
    )

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = rf.Linear(in_dim, hidden_dim)
            self.layer2 = rf.Linear(hidden_dim, out_dim)

        def __call__(self, x: Tensor) -> Tensor:
            x = rf.relu(self.layer1(x))
            x = self.layer2(x)
            return x

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output()

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_linear_same_dim():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim, out_dim = Dim(7, name="in"), Dim(13, name="out")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
            "classes": Tensor("classes", [batch_dim, time_dim], dtype="int32", sparse_dim=out_dim),
        }
    )

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = rf.Linear(in_dim, out_dim)
            self.layer2 = rf.Linear(out_dim, out_dim)

        def __call__(self, x: Tensor) -> Tensor:
            x = rf.relu(self.layer1(x))
            x = self.layer2(x)
            return x

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output()

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_linear_cross_entropy():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim, out_dim = Dim(7, name="in"), Dim(13, name="out")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
            "classes": Tensor("classes", [batch_dim, time_dim], dtype="int32", sparse_dim=out_dim),
        }
    )

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = rf.Linear(in_dim, out_dim)
            self.layer2 = rf.Linear(out_dim, out_dim)

        def __call__(self, x: Tensor) -> Tensor:
            x = rf.relu(self.layer1(x))
            x = self.layer2(x)
            return x

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        logits = model(extern_data["data"])
        targets = extern_data["classes"]
        loss = rf.cross_entropy(estimated=logits, estimated_type="logits", target=targets, axis=out_dim)
        loss.mark_as_default_output()

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_linear_ctc():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    target_time_dim = Dim(Tensor("target_time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    hidden_dim = Dim(13, name="hidden")
    out_dim = Dim(11, name="classes")
    out_wb_dim = out_dim + 1
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
            "classes": Tensor("classes", [batch_dim, target_time_dim], dtype="int32", sparse_dim=out_dim),
        }
    )

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = rf.Linear(in_dim, hidden_dim)
            self.layer2 = rf.Linear(hidden_dim, out_wb_dim)

        def __call__(self, x: Tensor) -> Tensor:
            x = rf.relu(self.layer1(x))
            x = self.layer2(x)
            return x

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        logits = model(extern_data["data"])
        targets = extern_data["classes"]
        loss = rf.ctc_loss(
            logits=logits,
            targets=targets,
            input_spatial_dim=time_dim,
            targets_spatial_dim=target_time_dim,
            blank_index=out_wb_dim.dimension - 1,
        )
        loss.mark_as_default_output()

    run_model(
        extern_data,
        lambda *, epoch, step: _Net(),
        _forward_step,
        dyn_dim_min_sizes={time_dim: 4, target_time_dim: 2},
        dyn_dim_max_sizes={time_dim: 11, target_time_dim: 5},
    )


def test_dropout():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor) -> Tensor:
            return rf.dropout(x, 0.5, axis=rf.dropout_broadcast_default() and in_dim, on_forward=True)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, time_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_dim_value():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor) -> Tensor:
            res = rf.ones((), dtype="int64")
            for d in x.dims:
                res *= rf.cast(rf.convert_to_tensor(d.get_dim_value_tensor()), "int64")
            return res

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=())

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_dim_mask():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor) -> Tensor:
            mask1 = time_dim.get_mask()
            mask1.verify_out_shape({batch_dim, time_dim})
            mask2 = time_dim.get_mask()
            assert mask1 is mask2  # cache
            time_dim_copy = Dim(None, name="time_copy")
            time_dim_copy.copy_from(time_dim)
            assert time_dim_copy != time_dim
            mask3 = time_dim_copy.get_mask()
            mask3.verify_out_shape({batch_dim, time_dim_copy})
            assert mask1 is not mask3  # other dims
            if rf.is_backend_raw_tensor_dim_tag_independent():
                assert mask1.raw_tensor is mask3.raw_tensor  # cache even across copy
            return mask1

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, time_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_num_elements_of_shape():
    import numpy as np

    batch_dim_ = Dim(5, name="batch")
    enc_dim = Dim(Tensor("enc", dims=[batch_dim_], dtype="int64"))
    dec_dim = Dim(Tensor("dec", dims=[batch_dim_], dtype="int64"))
    enc_dim.dyn_size_ext.raw_tensor = np.array([17, 16, 15, 13, 12])
    dec_dim.dyn_size_ext.raw_tensor = np.array([11, 10, 8, 7, 5])
    calc_n_enc = sum(enc_dim.dyn_size_ext.raw_tensor)
    calc_n_dec = sum(dec_dim.dyn_size_ext.raw_tensor)
    calc_n_prod = sum(enc_dim.dyn_size_ext.raw_tensor * dec_dim.dyn_size_ext.raw_tensor)
    assert rf.num_elements_of_shape([batch_dim_]) == batch_dim_.dimension
    n_b_enc = rf.num_elements_of_shape([batch_dim_, enc_dim])
    assert calc_n_enc == n_b_enc.raw_tensor.item()
    n_b_dec = rf.num_elements_of_shape([dec_dim, batch_dim_])
    assert calc_n_dec == n_b_dec.raw_tensor.item()
    n_prod = rf.num_elements_of_shape([batch_dim_, enc_dim, dec_dim])
    assert calc_n_prod == n_prod.raw_tensor.item()


def test_convert_to_tensor_numpy_backend():
    import numpy as np
    from returnn.frontend._numpy_backend import NumpyBackend

    x = rf.convert_to_tensor(1, dims=(), dtype="int32", _backend=NumpyBackend)
    assert isinstance(x.raw_tensor, np.ndarray)
    assert x.raw_tensor.dtype == np.int32
    assert x.raw_tensor.item() == 1


def test_param_assign():
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

        def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
            # No extra care should be needed for graph-based backends.
            a = rf.copy(self.param)
            self.param.assign(5)
            b = rf.copy(self.param)
            self.param.assign(7)
            return a, b, self.param

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        a, b, c = model(extern_data["data"])
        a.mark_as_output("a", shape=())
        b.mark_as_output("b", shape=())
        c.mark_as_output("c", shape=())

    out = run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)
    assert out["a"].raw_tensor == 2 and out["b"].raw_tensor == 5 and out["c"].raw_tensor == 7


def test_loss_normalized():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    # noinspection PyShadowingNames
    def _train_step(*, model: rf.Module, extern_data: TensorDict):
        model  # unused  # noqa
        x = extern_data["data"]

        loss = rf.reduce_sum(x, axis=in_dim)  # [B,T]
        loss.mark_as_loss("loss", use_normalized_loss=True)

        loss_custom_norm = rf.reduce_sum(loss, axis=time_dim)  # [B]
        loss_custom_norm.mark_as_loss(
            "loss_custom_norm", custom_inv_norm_factor=time_dim.get_size_tensor(), use_normalized_loss=True
        )

    run_model_torch_train(extern_data, lambda *, epoch, step: rf.Module(), _train_step)


def test_loss_normalization():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    use_normalized = False
    use_custom_inv_norm_factor = False

    # noinspection PyShadowingNames
    def _train_step(*, model: rf.Module, extern_data: TensorDict):
        model  # unused  # noqa
        x = extern_data["data"]

        loss = rf.reduce_sum(x, axis=in_dim)  # [B,T]
        loss.mark_as_loss(
            "loss",
            custom_inv_norm_factor=time_dim.get_size_tensor() if use_custom_inv_norm_factor else None,
            use_normalized_loss=use_normalized,
        )

    res1 = run_model_torch_train(extern_data, lambda *, epoch, step: rf.Module(), _train_step)
    res2 = run_model_torch_train(extern_data, lambda *, epoch, step: rf.Module(), _train_step)
    assert res1 == res2  # check deterministic
    use_normalized = True
    res3 = run_model_torch_train(extern_data, lambda *, epoch, step: rf.Module(), _train_step)
    assert res3["loss:summed"] == res2["loss:summed"] and res3["loss:inv_norm_factor"] == res2["loss:inv_norm_factor"]
    use_custom_inv_norm_factor = True
    res4 = run_model_torch_train(extern_data, lambda *, epoch, step: rf.Module(), _train_step)
    assert res4["loss:summed"] == res2["loss:summed"] and res4["loss:inv_norm_factor"] == res2["loss:inv_norm_factor"]


def test_rf_range_over_dim():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, model: rf.Module, extern_data: TensorDict):
        rf.range_over_dim(time_dim).mark_as_output("range", shape=[time_dim])
        rf.range_over_dim(time_dim, dtype="float32").mark_as_output("range_float", shape=[time_dim])

    run_model(extern_data, lambda *, epoch, step: rf.Module(), _forward_step)


def test_build_dict():
    d = rf.build_dict(rf.BatchNorm, use_mask=True)
    assert d == {"class": "rf.BatchNorm", "use_mask": True}


def test_build_dict_func():
    d = rf.build_dict(rf.relu)
    assert d == {"class": "rf.relu"}


def test_build_from_dict():
    rf.select_backend_torch()
    in_dim = Dim(7, name="in")
    mod = rf.build_from_dict({"class": "rf.BatchNorm", "use_mask": True}, in_dim, eps=1e-4)
    assert isinstance(mod, rf.BatchNorm)
    assert mod.in_dim == in_dim
    assert mod.use_mask
    assert mod.eps == 1e-4


def test_build_from_dict_func():
    import functools

    func = rf.build_from_dict({"class": "rf.relu"})
    assert func is rf.relu
    func = rf.build_from_dict({"class": "rf.combine", "kind": "+", "b": 1})
    assert isinstance(func, functools.partial)
    assert func.func is rf.combine
    assert func.keywords == {"kind": "+", "b": 1}


def test_build_from_dict_func_native():
    from types import BuiltinFunctionType

    rf.select_backend_torch()  # enables some of the native optimizations
    assert isinstance(rf.combine, BuiltinFunctionType)  # due to native optimizations
    func = rf.build_from_dict({"class": "rf.combine"})
    assert func is rf.combine


def test_parametrization():
    from functools import partial

    rf.select_backend_torch()  # any, doesn't really matter for the test
    rf.init_train_step_run_ctx(train_flag=True)  # such that dropout is used below

    in_dim = Dim(7, name="in")
    out_dim = Dim(13, name="out")
    mod = rf.Linear(in_dim, out_dim)
    orig_weight = mod.weight
    assert isinstance(orig_weight, rf.Parameter)
    orig_bias = mod.bias

    # Test parametrization.
    rf.register_parametrization(mod, "weight", partial(rf.dropout, drop_prob=0.5))
    assert rf.is_parametrized(mod)
    assert rf.is_parametrized(mod, "weight")
    weight = mod.weight
    assert weight is not orig_weight and not isinstance(weight, rf.Parameter)
    params = dict(mod.named_parameters())
    assert set(params.keys()) == {"weight", "bias"}
    assert params["weight"] is orig_weight
    assert params["bias"] is orig_bias

    rf.init_train_step_run_ctx(train_flag=False)
    weight = mod.weight
    assert weight is orig_weight  # no dropout in eval mode

    rf.init_train_step_run_ctx(train_flag=True)  # such that dropout would be used again
    rf.remove_parametrization(mod, "weight")
    weight = mod.weight
    assert weight is orig_weight
    assert not rf.is_parametrized(mod, "weight")
    assert not rf.is_parametrized(mod)
    params = dict(mod.named_parameters())
    assert set(params.keys()) == {"weight", "bias"}
    assert params["weight"] is orig_weight
    assert params["bias"] is orig_bias


def test_weight_noise():
    import torch

    if torch.__version__ < (2, 0):
        raise SkipTest("Torch version too old for this test (gradient_checkpoint_scope needs Torch >= 2.0)")
    rf.select_backend_torch()  # any, doesn't really matter for the test
    rf.init_train_step_run_ctx(train_flag=True)  # such that weight noise is used below

    in_dim = Dim(7, name="in")
    out_dim = Dim(13, name="out")
    mod = rf.Linear(in_dim, out_dim)
    orig_weight = mod.weight
    assert isinstance(orig_weight, rf.Parameter)
    orig_bias = mod.bias

    # Test parametrization.
    rf.weight_noise(mod, "weight", std=0.1)
    assert rf.is_parametrized(mod, "weight")
    weight = mod.weight
    assert weight is not orig_weight and not isinstance(weight, rf.Parameter)
    params = dict(mod.named_parameters())
    assert set(params.keys()) == {"weight", "bias"}
    assert params["weight"] is orig_weight
    assert params["bias"] is orig_bias

    class _Model(rf.Module):
        def __init__(self):
            super().__init__()
            self.linear = rf.Linear(in_dim, out_dim)

    model = _Model()
    for mod in model.modules():
        for param_name, param in mod.named_parameters(recurse=False):
            if param_name.endswith("bias"):  # no bias
                continue
            print("***", mod, param_name, param)
            rf.weight_noise(mod, param_name, std=0.0025)

    # https://github.com/rwth-i6/returnn/issues/1580
    conv = rf.Conv1d(in_dim, out_dim, 3, padding="same")
    rf.weight_noise(conv, "filter", std=0.1)
    time_dim = Dim(11, name="time")
    conv(rf.random_normal([time_dim, in_dim]), in_spatial_dim=time_dim)
