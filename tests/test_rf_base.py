"""
RETURNN frontend (returnn.frontend) tests
"""

from __future__ import annotations
from typing import Tuple
import _setup_test_env  # noqa
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, TensorDict, batch_dim
from rf_utils import run_model, run_model_torch_train


# Keep test_linear_direct and test_linear first here to have some very canonical examples.


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
