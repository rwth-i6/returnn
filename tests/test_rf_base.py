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


def test_dim_size_after_declare_same_as():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim], dtype="float32"),
        }
    )
    out_spatial_dim = Dim(None, name="out_spatial")

    # noinspection PyShadowingNames
    def _forward_step(**_kwargs):
        assert time_dim.dyn_size_ext.raw_tensor is not None
        enc_spatial_dim = time_dim * 2
        assert enc_spatial_dim.dyn_size_ext.raw_tensor is not None
        out_spatial_dim.declare_same_as(enc_spatial_dim)
        assert enc_spatial_dim.dyn_size_ext.raw_tensor is not None
        assert out_spatial_dim.dyn_size_ext.raw_tensor is not None
        out_spatial_dim.dyn_size_ext.mark_as_default_output(shape=[batch_dim])

    # Running it twice can trigger some weird behavior in Dim.declare_same_as.
    run_model(extern_data, lambda **_kwargs: rf.Module(), _forward_step, test_tensorflow=False)
    run_model(extern_data, lambda **_kwargs: rf.Module(), _forward_step, test_tensorflow=False)


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


def test_ctc_loss():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    target_time_dim = Dim(Tensor("target_time", [batch_dim], dtype="int32"))
    out_dim = Dim(11, name="classes")
    out_wb_dim = out_dim + 1
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, out_wb_dim], dtype="float32", feature_dim=out_wb_dim),
            "classes": Tensor("classes", [batch_dim, target_time_dim], dtype="int32", sparse_dim=out_dim),
        }
    )

    def _forward_step(*, extern_data: TensorDict, **_kwargs):
        logits = extern_data["data"]
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
        lambda **_kwargs: rf.Module(),
        _forward_step,
        dyn_dim_min_sizes={time_dim: 4, target_time_dim: 2},
        dyn_dim_max_sizes={time_dim: 11, target_time_dim: 5},
    )


def test_ctc_loss_broadcast():
    branch_dim = Dim(3, name="branch")
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    target_time_dim = Dim(Tensor("target_time", [batch_dim], dtype="int32"))
    out_dim = Dim(11, name="classes")
    out_wb_dim = out_dim + 1
    extern_data = TensorDict(
        {
            "data": Tensor(
                "data", [batch_dim, branch_dim, time_dim, out_wb_dim], dtype="float32", feature_dim=out_wb_dim
            ),
            "classes": Tensor("classes", [batch_dim, target_time_dim], dtype="int32", sparse_dim=out_dim),
        }
    )

    def _forward_step(*, extern_data: TensorDict, **_kwargs):
        logits = extern_data["data"]
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
        lambda **_kwargs: rf.Module(),
        _forward_step,
        test_tensorflow=False,
        dyn_dim_min_sizes={time_dim: 4, target_time_dim: 2},
        dyn_dim_max_sizes={time_dim: 11, target_time_dim: 5},
    )


def test_edit_distance():
    import numpy
    import torch
    from typing import Sequence
    from collections import namedtuple
    import itertools

    def _edit_distance_ref_b1(a: Sequence[int], b: Sequence[int]) -> int:
        """
        Reference implementation for edit distance.
        """
        n = len(a) + 1
        m = len(b) + 1
        d = torch.zeros((n, m), dtype=torch.int32)
        for i in range(n):
            d[i, 0] = i
        for j in range(m):
            d[0, j] = j
        for j in range(1, m):
            for i in range(1, n):
                if a[i - 1] == b[j - 1]:
                    d[i, j] = d[i - 1, j - 1]
                else:
                    d[i, j] = min(
                        d[i - 1, j] + 1,  # deletion
                        d[i, j - 1] + 1,  # insertion
                        d[i - 1, j - 1] + 1,  # substitution
                    )
        return int(d[n - 1, m - 1])

    # noinspection PyShadowingNames
    def _edit_distance_ref(a: Tensor, a_spatial_dim: Dim, b: Tensor, b_spatial_dim: Dim) -> torch.Tensor:
        """
        Reference implementation for edit distance.
        """
        batch_dim = a.dims[0]
        assert a.dims == (batch_dim, a_spatial_dim) and b.dims == (batch_dim, b_spatial_dim)
        res = []
        for i in range(batch_dim.dimension):
            assert a_spatial_dim.dyn_size[i] <= a.raw_tensor.size(1)
            assert b_spatial_dim.dyn_size[i] <= b.raw_tensor.size(1)
            res.append(
                _edit_distance_ref_b1(
                    a.raw_tensor[i, : a_spatial_dim.dyn_size[i]], b.raw_tensor[i, : b_spatial_dim.dyn_size[i]]
                )
            )
        return torch.tensor(res, dtype=torch.int32)

    # noinspection PyShadowingNames
    def _check_edit_distance(a: Tensor, a_spatial_dim: Dim, b: Tensor, b_spatial_dim: Dim):
        ref = _edit_distance_ref(a, a_spatial_dim, b, b_spatial_dim)
        res = rf.edit_distance(a, a_spatial_dim, b, b_spatial_dim)
        assert res.raw_tensor.shape == ref.shape == a_spatial_dim.dyn_size.shape == b_spatial_dim.dyn_size.shape
        assert len(ref.shape) == 1
        print("ref:", ref, "res:", res.raw_tensor)
        batch_size = ref.shape[0]
        for i in range(batch_size):
            assert res.raw_tensor[i] == ref[i], (
                f"batch idx i={i}, a[i]={a.raw_tensor[i]} len {a_spatial_dim.dyn_size[i]},"
                f" b[i]={b.raw_tensor[i]} len {b_spatial_dim.dyn_size[i]},"
                f" ref[i]={ref[i]}, res[i]={res.raw_tensor[i]};\n"
                f" a={a.raw_tensor} lens {a_spatial_dim.dyn_size},"
                f" b={b.raw_tensor} lens {b_spatial_dim.dyn_size}"
            )
        assert (res.raw_tensor == ref).all()

    SizedTensor = namedtuple("SizedTensor", ["tensor", "seq_lens"])

    _SeqsB1 = [
        SizedTensor(torch.tensor([[1, 2, 3, 4]]), torch.tensor([4])),
        SizedTensor(torch.tensor([[1, 2, 3]]), torch.tensor([3])),
        SizedTensor(torch.tensor([[1, 2, 4]]), torch.tensor([3])),
        SizedTensor(torch.tensor([[1, 4]]), torch.tensor([2])),
        SizedTensor(torch.tensor([[5, 2, 4]]), torch.tensor([3])),
        SizedTensor(torch.tensor([[]], dtype=torch.int64), torch.tensor([0])),
    ]

    for a, b in itertools.product(_SeqsB1, _SeqsB1):
        a: SizedTensor
        b: SizedTensor
        # noinspection PyShadowingNames
        batch_dim = Dim(1, name="batch")
        a_spatial_dim = Dim(Tensor("a_sizes", [batch_dim], dtype="int64", raw_tensor=a.seq_lens))
        b_spatial_dim = Dim(Tensor("b_sizes", [batch_dim], dtype="int64", raw_tensor=b.seq_lens))
        a_ = Tensor("a", [batch_dim, a_spatial_dim], dtype="int64", raw_tensor=a.tensor)
        b_ = Tensor("b", [batch_dim, b_spatial_dim], dtype="int64", raw_tensor=b.tensor)
        _check_edit_distance(a_, a_spatial_dim, b_, b_spatial_dim)

    rnd = numpy.random.RandomState(42)
    for a, b in itertools.product(_SeqsB1, _SeqsB1):
        batch_size = rnd.randint(2, 11)
        a_max_len = rnd.randint(a.seq_lens[0], a.seq_lens[0] + 5)
        b_max_len = rnd.randint(b.seq_lens[0], b.seq_lens[0] + 5)
        a_sizes = rnd.randint(0, a_max_len + 1, size=(batch_size,))
        b_sizes = rnd.randint(0, b_max_len + 1, size=(batch_size,))
        a_sizes[0] = a.seq_lens[0]
        b_sizes[0] = b.seq_lens[0]
        a_max_len = max(a_sizes)
        b_max_len = max(b_sizes)
        a_values = rnd.randint(0, 10, (batch_size, a_max_len))
        b_values = rnd.randint(0, 10, (batch_size, b_max_len))
        a_values[0, : a.seq_lens[0]] = a.tensor[0, : a.seq_lens[0]]
        b_values[0, : b.seq_lens[0]] = b.tensor[0, : b.seq_lens[0]]
        a_sizes = torch.tensor(a_sizes, dtype=torch.int32)
        b_sizes = torch.tensor(b_sizes, dtype=torch.int32)

        # noinspection PyShadowingNames
        batch_dim = Dim(batch_size, name="batch")
        a_spatial_dim = Dim(Tensor("a_sizes", [batch_dim], dtype="int32", raw_tensor=a_sizes))
        b_spatial_dim = Dim(Tensor("b_sizes", [batch_dim], dtype="int32", raw_tensor=b_sizes))
        a_ = Tensor("a", [batch_dim, a_spatial_dim], dtype="int64", raw_tensor=torch.tensor(a_values))
        b_ = Tensor("b", [batch_dim, b_spatial_dim], dtype="int64", raw_tensor=torch.tensor(b_values))
        _check_edit_distance(a_, a_spatial_dim, b_, b_spatial_dim)


def test_audio_log_mel_filterbank_from_raw_bfloat16():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    out_dim = Dim(80, name="freq")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim], dtype="float32"),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, extern_data: TensorDict, **_kwargs):
        audio = extern_data["data"]
        audio = rf.cast(audio, "bfloat16")
        out, out_spatial_dim = rf.audio.log_mel_filterbank_from_raw(audio, in_spatial_dim=time_dim, out_dim=out_dim)
        assert out.dtype == "bfloat16"
        out = rf.cast(out, "float32")  # the test framework doesn't support bfloat16 currently due to Numpy...
        out.mark_as_default_output(shape=(batch_dim, out_spatial_dim, out_dim))

    run_model(
        extern_data,
        lambda **_kwargs: rf.Module(),
        _forward_step,
        dyn_dim_min_sizes={time_dim: 2000},
        dyn_dim_max_sizes={time_dim: 3000},
        test_tensorflow=False,
    )


def test_copy_compatible_to_empty():
    import torch

    beam_dim = Dim(3, name="beam")
    hist_dim = Dim(0, name="hist")
    hist_bc_dim = Dim(1, name="hist_bc")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, beam_dim], dtype="int32"),
        }
    )
    out_template = Tensor("out_template", [batch_dim, beam_dim, hist_dim], dtype="int32")

    # noinspection PyShadowingNames
    def _forward_step(*, extern_data: TensorDict, **_kwargs):
        data = extern_data["data"]

        out = data.copy_compatible_to(out_template)
        print("out:", out)
        assert (
            len(out.dims) == 3
            and out.dims[:2] == (batch_dim, beam_dim)
            and out.dims[2] != hist_dim
            and out.dims[2].dimension == 1
        )
        out, _ = rf.replace_dim(out, in_dim=out.dims[2], out_dim=hist_bc_dim)  # just that we have consistent output
        out.mark_as_output("out", shape=out.dims)

        out_unbroadcast = data.copy_compatible_to(out_template, unbroadcast=True)
        print("out_unbroadcast:", out_unbroadcast)
        assert out_unbroadcast.dims == out_template.dims
        if isinstance(out_unbroadcast.raw_tensor, torch.Tensor):
            print("out_unbroadcast.raw_tensor:", out_unbroadcast.raw_tensor)
            assert out_unbroadcast.raw_tensor.shape == (
                batch_dim.get_dim_value(),
                beam_dim.dimension,
                hist_dim.dimension,
            )
        out_unbroadcast.mark_as_output("out_unbroadcast", shape=out_unbroadcast.dims)

        out_raw = data.copy_compatible_to_dims_raw(out_template.dims)
        if isinstance(out_raw, torch.Tensor):
            print("out raw:", out_raw)
            assert out_raw.shape == out.raw_tensor.shape == (batch_dim.get_dim_value(), beam_dim.dimension, 1)
            torch.testing.assert_close(out_raw, out.raw_tensor)

    run_model(extern_data, lambda **_kwargs: rf.Module(), _forward_step, test_tensorflow=False)
