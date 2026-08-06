"""
Tests for the pure (low-level) TF backend of the RETURNN frontend
(:mod:`returnn.tf.frontend_low_level`, ``backend = "tensorflow"``),
as opposed to the TF-layers backend (``backend = "tensorflow-net-dict"``).

Each test runs the same RF model code on PyTorch and on the TF backend and compares the outputs,
via :func:`rf_utils.run_model` with ``tf_low_level=True``.
The same comparison runs for all the other ``test_rf_*`` tests with ``RETURNN_TEST_RF_TF_LOW_LEVEL=1``.
"""

from __future__ import annotations
from typing import Tuple
import _setup_test_env  # noqa
import numpy
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, TensorDict, batch_dim
from returnn.tf.frontend_low_level import DeferredVariable
from rf_utils import run_model


def test_linear():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict({"data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32")})
    out_dim = Dim(5, name="out")

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.linear = rf.Linear(in_dim, out_dim)

        def __call__(self, x: Tensor) -> Tensor:
            return self.linear(x)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, time_dim, out_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, tf_low_level=True)


def test_layer_norm():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict({"data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32")})

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.norm = rf.LayerNorm(in_dim)

        def __call__(self, x: Tensor) -> Tensor:
            return self.norm(x)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, time_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, tf_low_level=True)


def test_linear_softmax():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict({"data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32")})
    classes_dim = Dim(5, name="classes")

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.linear = rf.Linear(in_dim, classes_dim)

        def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
            logits = self.linear(x)
            return rf.softmax(logits, axis=classes_dim), rf.log_softmax(logits, axis=classes_dim)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        probs, log_probs = model(extern_data["data"])
        probs.mark_as_default_output(shape=(batch_dim, time_dim, classes_dim))
        log_probs.mark_as_output("log_probs", shape=(batch_dim, time_dim, classes_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, tf_low_level=True)


def test_parameter_names_and_init():
    # The variables are created after the model (deferred), so they can be named
    # after their position in the module hierarchy, and the initial values must survive that.
    import tensorflow as tf
    import returnn.tf.compat as tf_compat
    from returnn.tf.frontend_low_level import TFBackend

    # noinspection PyProtectedMember
    from returnn.frontend import _backend

    in_dim = Dim(7, name="in")
    out_dim = Dim(5, name="out")

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.linear = rf.Linear(in_dim, out_dim)
            self.norm = rf.LayerNorm(out_dim)

    _backend.select_backend_tf()
    with tf_compat.v1.Graph().as_default(), tf_compat.v1.Session().as_default() as session:
        with TFBackend.deferred_parameter_creation():
            net = _Net()
        assert all(isinstance(p.raw_tensor, DeferredVariable) for _, p in net.named_parameters())
        TFBackend.create_parameters(net)

        names = {name: TFBackend.get_parameter_variable(p) for name, p in net.named_parameters()}
        assert set(names) == {"linear.weight", "linear.bias", "norm.scale", "norm.bias"}
        for name, var in names.items():
            assert isinstance(var, tf.Variable)
            assert var.op.name == name.replace(".", "/"), f"{name}: unexpected variable name {var.op.name}"

        # the standard init path must produce the values rf.Parameter asked for,
        # incl. a scalar init broadcast to the full param shape
        session.run(tf_compat.v1.global_variables_initializer())
        numpy.testing.assert_almost_equal(session.run(names["norm.scale"]), numpy.ones([5]))
        numpy.testing.assert_almost_equal(session.run(names["norm.bias"]), numpy.zeros([5]))
        assert session.run(names["linear.weight"]).std() > 0  # Glorot init, not zeros


def test_matmul_common_and_unique_dims():
    # matmul with a reduce dim, a common (batch-like) dim and unique dims on both sides
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict({"data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32")})
    heads_dim = Dim(3, name="heads")
    out_dim = Dim(5, name="out")

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.weight = rf.Parameter([in_dim, heads_dim, out_dim])
            self.weight.initial = rf.init.Glorot()

        def __call__(self, x: Tensor) -> Tensor:
            return rf.matmul(x, self.weight, reduce=in_dim)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, time_dim, heads_dim, out_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, tf_low_level=True)
