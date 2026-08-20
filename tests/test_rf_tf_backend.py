"""
Tests for the pure (low-level) TF backend of the RETURNN frontend
(:mod:`returnn.tf.frontend_low_level`, ``backend = "tensorflow"``),
as opposed to the TF-layers backend (``backend = "tensorflow-net-dict"``).

Each test runs the same RF model code on PyTorch and on the TF backend and compares the outputs,
via :func:`rf_utils.run_model` with ``tf_low_level=True``.
The same comparison runs for all the other ``test_rf_*`` tests with ``RETURNN_TEST_RF_TF_LOW_LEVEL=1``.
"""

from __future__ import annotations
from typing import Dict, Tuple
import os

# These tests compare the TF backend against PyTorch for exactness.
# TF enables TF32 by default since Ampere, torch does not,
# so matmuls would run at different precision on the two sides
# (test_linear, test_matmul*, test_full_model*, test_depthwise_conv_grad_runs).
# Set before TF is imported below, and via the CUDA-level switch rather than
# tf.config.experimental.enable_tensor_float_32_execution:
# that call initializes the TF context at import time,
# which makes test_engine_dynamic_learning_rate and test_engine_weight_decay_modules_blacklist
# fail in a full-file run while still passing alone.
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
# Same story on CPU: test_full_model_torch_checkpoint_parity drifts to enc diff/scale ~1e-1
# exactly on the AMX-capable GitHub runners (hardware is a per-job lottery) and passes at
# ~2e-6 on every AVX2-only runner: 6 green AVX2 jobs vs the 1 red AMX job on 2026-08-10,
# including a same-commit rerun pair whose result flipped with the hardware
# (cpu_feature_guard: "AVX2 FMA" vs "... AVX512_BF16 ... AMX_BF16").
# ~1e-1 is bf16-class for this depth (the amp test needs atol 0.2*scale for real bf16),
# far beyond f32 rounding, so the suspect is TF 2.10's oneDNN using AMX/bf16 kernels on
# f32 data -- the correlation is hardware-controlled, the oneDNN sub-mechanism is not pinned.
# Disable oneDNN so the CPU path is plain f32 Eigen on every runner.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import _setup_test_env  # noqa
import shutil
import tempfile
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


def test_full_model():
    # The model of tests/test_rf_packed.py::test_full_model_packed_traced_program_replay, padded:
    # Conformer encoder (conv subsample, rel-pos self-att, depthwise conv) + Transformer decoder
    # + aux CTC head, with both losses. Marking the total as "loss" also compares the input grads.
    extern_data, get_model, forward_step, dims = _full_model_setup()
    run_model(
        extern_data,
        get_model,
        forward_step,
        # the CTC input (after the 6x subsampling) must stay longer than the targets
        dyn_dim_max_sizes={dims["time_dim"]: 32, dims["target_time_dim"]: 4},
        dyn_dim_min_sizes={dims["time_dim"]: 24, dims["target_time_dim"]: 2},
        tf_low_level=True,
    )


def test_depthwise_conv_grad_runs():
    # A grouped conv whose gradient is BUILT but not RUNNABLE is the CI failure mode
    # ("Gradients for grouped convolutions are not supported on CPU", TF 2.10):
    # tf.gradients returns a tensor and only session.run fails, so the fallback probe
    # in TFBackend must have executed the gradient to answer correctly.
    # Marking the sum as "loss" makes the harness run the input gradient too,
    # which is what fails when the probe got it wrong.
    # noinspection PyProtectedMember
    from returnn.tf.frontend_low_level._backend import _grouped_conv_grad_supported

    print("grouped conv gradient supported here:", _grouped_conv_grad_supported())

    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(4, name="in")
    extern_data = TensorDict({"data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32")})

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            # groups == in_dim: the depthwise case the Conformer conv block uses
            self.conv = rf.Conv1d(in_dim, in_dim, filter_size=3, groups=in_dim.dimension, padding="same")

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out, _ = model.conv(extern_data["data"], in_spatial_dim=time_dim)
        out.mark_as_default_output(shape=(batch_dim, time_dim, in_dim))
        rf.reduce_sum(out, axis=out.dims).mark_as_output("loss", shape=())

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, tf_low_level=True)


def test_scatter_logsumexp_stop_gradient_scope():
    # rf.scatter_logsumexp is the RF-internal user of rf.stop_gradient_scope,
    # which this backend cannot implement faithfully (see TFBackend.stop_gradient_scope).
    # Marking the total as "loss" makes the harness compare the input gradients too,
    # so this checks that the no-op scope really leaves values AND gradients correct.
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(5, name="in")
    out_dim = Dim(3, name="out")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
            "indices": Tensor("indices", [batch_dim, time_dim], dtype="int32", sparse_dim=out_dim),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, model: rf.Module, extern_data: TensorDict):
        out = rf.scatter_logsumexp(
            extern_data["data"], indices=extern_data["indices"], indices_dim=time_dim, out_dim=out_dim
        )
        out.mark_as_default_output(shape=(batch_dim, out_dim, in_dim))
        rf.reduce_sum(out, axis=(out_dim, in_dim)).mark_as_output("loss", shape=(batch_dim,))

    run_model(extern_data, lambda *, epoch, step: rf.Module(), _forward_step, tf_low_level=True)


def test_train_step_with_updater():
    # The core of a TF engine step for a pure-RF model:
    # extern data as placeholders, get_model + train_step under the TF backend,
    # the run-ctx losses handed to RETURNN's existing Updater, then a few session.run steps.
    # This is what the BackendEngine.TensorFlow path has to wire up (see the project plan, item 8).
    from returnn.config import Config

    # noinspection PyProtectedMember
    from returnn.frontend import _backend

    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(9, name="in")
    out_dim = Dim(5, name="out")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
            "classes": Tensor("classes", [batch_dim, time_dim], dtype="int32", sparse_dim=out_dim),
        }
    )

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.linear = rf.Linear(in_dim, Dim(16, name="hidden"))
            self.out = rf.Linear(self.linear.out_dim, out_dim)

        def __call__(self, x: Tensor) -> Tensor:
            return self.out(rf.relu(self.linear(x)))

    def _train_step(*, model: _Net, extern_data: TensorDict) -> Tensor:
        logits = model(extern_data["data"])
        loss = rf.cross_entropy(estimated=logits, target=extern_data["classes"], axis=out_dim, estimated_type="logits")
        return rf.reduce_mean(loss, axis=loss.dims)

    n_batch, n_time = 3, 7
    rnd = numpy.random.RandomState(42)
    feed_data = rnd.randn(n_batch, n_time, in_dim.dimension).astype("float32")
    feed_classes = rnd.randint(0, out_dim.dimension, size=(n_batch, n_time)).astype("int32")
    feed_sizes = numpy.array([7, 5, 6], dtype="int32")

    config = Config({"optimizer": {"class": "adam"}, "learning_rate": 0.1})
    _backend.select_backend_tf()
    losses = []
    # the batch dim gets a TF tensor below, so restore the global state even when this fails
    prev_batch_dyn_size_ext = batch_dim.dyn_size_ext
    try:
        _train_loop(config, extern_data, _Net, _train_step, losses, feed_data, feed_classes, feed_sizes, time_dim)
    finally:
        batch_dim.dyn_size_ext = prev_batch_dyn_size_ext
        rf.select_backend_torch()

    print("losses:", ["%.4f" % v for v in losses])
    assert losses[-1] < losses[0] * 0.9, f"loss did not decrease: {losses}"


# noinspection PyShadowingNames
def _train_loop(config, extern_data, net_cls, train_step, losses, feed_data, feed_classes, feed_sizes, time_dim):
    """the graph + session part of :func:`test_train_step_with_updater`"""
    import tensorflow as tf
    import returnn.tf.compat as tf_compat
    from returnn.config import global_config_ctx
    from returnn.tf.updater import Updater
    from returnn.tf.frontend_low_level import TFBackend

    with tf_compat.v1.Graph().as_default(), tf_compat.v1.Session().as_default() as session, global_config_ctx(config):
        rf.set_random_seed(42)
        # extern data as placeholders, the way the engine feeds a dataset.
        # No TFNetwork / ExternData: this path builds the graph from RF code, it has no layers.
        for value in extern_data.data.values():
            value.raw_tensor = TFBackend.create_placeholder_raw(value)
            for dim in value.dims:
                if dim.is_dynamic() and dim.dyn_size_ext is not None and dim.dyn_size_ext.raw_tensor is None:
                    dim.dyn_size_ext.raw_tensor = TFBackend.create_placeholder_raw(dim.dyn_size_ext)

        # the batch dim gets its size from the data placeholder
        # (the net-dict path does this via BatchInfo; the RF path needs it for masked reduces)
        batch_dim.dyn_size_ext = Tensor("batch", dims=(), dtype="int32")
        batch_dim.dyn_size_ext.raw_tensor = tf.shape(extern_data.data["data"].raw_tensor)[0]

        with TFBackend.deferred_parameter_creation():
            model = net_cls()
        TFBackend.create_parameters(model)
        loss = train_step(model=model, extern_data=extern_data)

        global_train_step_var = tf.Variable(0, dtype="int64", trainable=False, name="global_step")
        updater = Updater(
            config=config,
            initial_learning_rate=0.1,
            objective=loss.raw_tensor,
            global_train_step_var=global_train_step_var,
        )
        updater.set_trainable_vars([TFBackend.get_parameter_variable(p) for _, p in model.named_parameters()])
        optim_op = updater.get_optim_op()  # creates the optimizer slots, so before the init
        session.run(tf_compat.v1.global_variables_initializer())
        updater.init_optimizer_vars(session)
        updater.set_learning_rate(0.1, session=session)

        feed_dict = {
            extern_data.data["data"].raw_tensor: feed_data,
            extern_data.data["classes"].raw_tensor: feed_classes,
            time_dim.dyn_size_ext.raw_tensor: feed_sizes,
        }
        for _ in range(10):
            loss_v, _ = session.run([loss.raw_tensor, optim_op], feed_dict=feed_dict)
            losses.append(float(loss_v))


def test_train_from_dataset():
    # The data half of a TF engine step: batches from a real RETURNN Dataset,
    # assembled by the backend-independent batch_to_raw_dict, fed into the placeholders.
    # Together with test_train_step_with_updater this is everything the engine loop does per step.
    import tensorflow as tf
    import returnn.tf.compat as tf_compat
    from returnn.config import Config, global_config_ctx
    from returnn.datasets.generating import DummyDataset
    from returnn.engine.batch import batch_to_raw_dict
    from returnn.tf.updater import Updater
    from returnn.tf.frontend_low_level import TFBackend

    # noinspection PyProtectedMember
    from returnn.frontend import _backend

    n_data_dim, n_classes_dim, seq_len = 2, 3, 5
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(n_data_dim, name="in")
    out_dim = Dim(n_classes_dim, name="out")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
            "classes": Tensor("classes", [batch_dim, time_dim], dtype="int32", sparse_dim=out_dim),
        }
    )
    dataset = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=8, seq_len=seq_len)
    dataset.init_seq_order(epoch=1)

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.out = rf.Linear(in_dim, out_dim)

        def __call__(self, x: Tensor) -> Tensor:
            return self.out(x)

    def _train_step(*, model: _Net, extern_data: TensorDict) -> Tensor:
        logits = model(extern_data["data"])
        loss = rf.cross_entropy(estimated=logits, target=extern_data["classes"], axis=out_dim, estimated_type="logits")
        return rf.reduce_mean(loss, axis=loss.dims)

    config = Config({"optimizer": {"class": "adam"}, "learning_rate": 0.05})
    _backend.select_backend_tf()
    losses = []
    prev_batch_dyn_size_ext = batch_dim.dyn_size_ext
    try:
        with (
            tf_compat.v1.Graph().as_default(),
            tf_compat.v1.Session().as_default() as session,
            global_config_ctx(config),
        ):
            rf.set_random_seed(42)
            for value in extern_data.data.values():
                value.raw_tensor = TFBackend.create_placeholder_raw(value)
                for dim in value.dims:
                    if dim.is_dynamic() and dim.dyn_size_ext is not None and dim.dyn_size_ext.raw_tensor is None:
                        dim.dyn_size_ext.raw_tensor = TFBackend.create_placeholder_raw(dim.dyn_size_ext)
            batch_dim.dyn_size_ext = Tensor("batch", dims=(), dtype="int32")
            batch_dim.dyn_size_ext.raw_tensor = tf.shape(extern_data.data["data"].raw_tensor)[0]

            with TFBackend.deferred_parameter_creation():
                model = _Net()
            TFBackend.create_parameters(model)
            loss = _train_step(model=model, extern_data=extern_data)

            global_train_step_var = tf.Variable(0, dtype="int64", trainable=False, name="global_step")
            updater = Updater(
                config=config,
                initial_learning_rate=0.05,
                objective=loss.raw_tensor,
                global_train_step_var=global_train_step_var,
            )
            updater.set_trainable_vars([TFBackend.get_parameter_variable(p) for _, p in model.named_parameters()])
            optim_op = updater.get_optim_op()
            session.run(tf_compat.v1.global_variables_initializer())
            updater.init_optimizer_vars(session)
            updater.set_learning_rate(0.05, session=session)

            n_steps = 0
            for epoch in range(1, 6):
                dataset.init_seq_order(epoch=epoch)
                batches = dataset.generate_batches(recurrent_net=False, batch_size=20, max_seqs=4)
                epoch_losses = []
                while batches.has_more():
                    (batch,) = batches.peek_next_n(1)
                    raw = batch_to_raw_dict(
                        batch, dataset=dataset, extern_data=extern_data, data_keys=["data", "classes"]
                    )
                    feed_dict = {
                        extern_data.data["data"].raw_tensor: raw["data"],
                        extern_data.data["classes"].raw_tensor: raw["classes"],
                        time_dim.dyn_size_ext.raw_tensor: raw["data_seq_lens"],
                    }
                    loss_v, _ = session.run([loss.raw_tensor, optim_op], feed_dict=feed_dict)
                    epoch_losses.append(float(loss_v))
                    n_steps += 1
                    batches.advance(1)
                losses.append(sum(epoch_losses) / len(epoch_losses))
                print("epoch %i: %d steps, mean loss %.4f" % (epoch, len(epoch_losses), losses[-1]))
            # the Updater increments the step counter as part of the optim op
            assert int(session.run(global_train_step_var)) == n_steps
    finally:
        batch_dim.dyn_size_ext = prev_batch_dyn_size_ext
        rf.select_backend_torch()

    assert losses[-1] < losses[0], f"loss did not decrease over epochs: {losses}"


def test_checkpoint_save_load():
    # Checkpoints for the RF path: the TF Saver takes an explicit {name: var} mapping,
    # so the checkpoint keys are the RF parameter names (module hierarchy), independent of
    # what the variables happen to be called in the graph. That is what makes a checkpoint
    # written here loadable by name from another backend (and what the engine needs).
    import tensorflow as tf
    import returnn.tf.compat as tf_compat
    from returnn.tf.frontend_low_level import TFBackend

    # noinspection PyProtectedMember
    from returnn.frontend import _backend

    in_dim = Dim(4, name="in")
    out_dim = Dim(3, name="out")

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.linear = rf.Linear(in_dim, out_dim)
            self.norm = rf.LayerNorm(out_dim)

    ckpt_dir = tempfile.mkdtemp(prefix="returnn-test-rf-tf-ckpt-")
    ckpt_path = ckpt_dir + "/model.001"
    _backend.select_backend_tf()
    try:
        # write
        with tf_compat.v1.Graph().as_default(), tf_compat.v1.Session().as_default() as session:
            rf.set_random_seed(42)
            with TFBackend.deferred_parameter_creation():
                model = _Net()
            TFBackend.create_parameters(model)
            var_by_name = {name: TFBackend.get_parameter_variable(p) for name, p in model.named_parameters()}
            saver = tf_compat.v1.train.Saver(var_by_name)
            session.run(tf_compat.v1.global_variables_initializer())
            written = {name: session.run(var) for name, var in var_by_name.items()}
            saver.save(session, ckpt_path)

        assert set(name for name, _ in tf.train.list_variables(ckpt_path)) == set(var_by_name)

        # read back into a fresh graph and model
        with tf_compat.v1.Graph().as_default(), tf_compat.v1.Session().as_default() as session:
            rf.set_random_seed(1234)  # different init, so a failed restore would show
            with TFBackend.deferred_parameter_creation():
                model2 = _Net()
            TFBackend.create_parameters(model2)
            var_by_name2 = {name: TFBackend.get_parameter_variable(p) for name, p in model2.named_parameters()}
            session.run(tf_compat.v1.global_variables_initializer())
            tf_compat.v1.train.Saver(var_by_name2).restore(session, ckpt_path)
            restored = {name: session.run(var) for name, var in var_by_name2.items()}
    finally:
        rf.select_backend_torch()
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    assert set(written) == set(restored) == {"linear.weight", "linear.bias", "norm.scale", "norm.bias"}
    for name in written:
        numpy.testing.assert_array_equal(restored[name], written[name], err_msg=f"param {name} differs")


def test_engine_train():
    # The engine (returnn/tf/engine_rf.py) driving a config end to end:
    # epoch loop, learning-rate control, dev evaluation, checkpoint per epoch.
    from returnn.config import Config, global_config_ctx
    from returnn.datasets.generating import DummyDataset
    from returnn.tf.engine_rf import Engine

    # noinspection PyProtectedMember
    from returnn.frontend import _backend

    n_data_dim, n_classes_dim, seq_len = 2, 3, 5
    train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=8, seq_len=seq_len)
    train_data.init_seq_order(epoch=1)
    dev_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=4, seq_len=seq_len)
    dev_data.init_seq_order(epoch=1)

    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(n_data_dim, name="in")
    out_dim = Dim(n_classes_dim, name="out")

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.out = rf.Linear(in_dim, out_dim)

    # noinspection PyShadowingNames
    def _get_model(*, epoch: int, step: int, **_kwargs) -> rf.Module:
        return _Net()

    # noinspection PyShadowingNames
    def _train_step(*, model: _Net, extern_data: TensorDict, **_kwargs):
        logits = model.out(extern_data["data"])
        loss = rf.cross_entropy(estimated=logits, target=extern_data["classes"], axis=out_dim, estimated_type="logits")
        loss.mark_as_loss("ce")

    model_dir = tempfile.mkdtemp(prefix="returnn-test-rf-tf-engine-")
    config = Config(
        {
            "backend": "tensorflow",
            "model": model_dir + "/model",
            "extern_data": {
                "data": {"dims": [batch_dim, time_dim, in_dim], "dtype": "float32"},
                "classes": {"dims": [batch_dim, time_dim], "sparse_dim": out_dim, "dtype": "int32"},
            },
            "get_model": _get_model,
            "train_step": _train_step,
            "optimizer": {"class": "adam"},
            "learning_rate": 0.05,
            "batch_size": 20,
            "max_seqs": 4,
            "num_epochs": 3,
        }
    )

    _backend.select_backend_tf()
    prev_batch_dyn_size_ext = batch_dim.dyn_size_ext
    try:
        # the engine resolves the backend and the existing model files via the GLOBAL config
        with global_config_ctx(config):
            engine = Engine(config=config)
            engine.init_train_from_config(config=config, train_data=train_data, dev_data=dev_data)
            engine.train()
            assert engine.epoch == 4  # trained epochs 1..3
            scores = {
                key: engine.learning_rate_control.get_epoch_error_dict(epoch) for epoch, key in [(1, "ep1"), (3, "ep3")]
            }
            print("scores:", scores)
            assert scores["ep3"]["train_loss"] < scores["ep1"]["train_loss"], scores
            assert "dev_loss" in scores["ep3"], scores
            # the checkpoint of the last epoch is there, and holds the RF parameter names
            import tensorflow as tf

            ckpt = engine.get_epoch_model_filename(epoch=3)
            assert set(name for name, _ in tf.train.list_variables(ckpt)) == {
                "out.weight",
                "out.bias",
                "global_step",
                "epoch",
            }

            # a fresh engine continues from that checkpoint instead of starting over,
            # including the step counter (else every step-based schedule would restart)
            engine2 = Engine(config=config)
            engine2.init_train_from_config(config=config, train_data=train_data, dev_data=dev_data)
            assert engine2.epoch == 4, "did not continue from the existing checkpoint"
            assert engine2.global_train_step == engine.global_train_step > 0, (
                engine2.global_train_step,
                engine.global_train_step,
            )
    finally:
        batch_dim.dyn_size_ext = prev_batch_dyn_size_ext
        rf.select_backend_torch()
        shutil.rmtree(model_dir, ignore_errors=True)


def test_engine_train_tf_jit():
    # tf_jit: XLA auto-clustering on the session (global_jit_level), see _make_tf_session_config.
    # The engine seeds the model init from (epoch, step, random_seed), so two runs of the same
    # config are init-identical and the jit run must reproduce the non-jit losses up to
    # XLA-vs-Eigen float reassociation.
    from returnn.config import Config, global_config_ctx
    from returnn.datasets.generating import DummyDataset
    from returnn.tf.engine_rf import Engine

    # noinspection PyProtectedMember
    from returnn.frontend import _backend

    n_data_dim, n_classes_dim, seq_len = 2, 3, 5

    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(n_data_dim, name="in")
    out_dim = Dim(n_classes_dim, name="out")

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.out = rf.Linear(in_dim, out_dim)

    # noinspection PyShadowingNames
    def _get_model(*, epoch: int, step: int, **_kwargs) -> rf.Module:
        return _Net()

    # noinspection PyShadowingNames
    def _train_step(*, model: _Net, extern_data: TensorDict, **_kwargs):
        logits = model.out(extern_data["data"])
        loss = rf.cross_entropy(estimated=logits, target=extern_data["classes"], axis=out_dim, estimated_type="logits")
        loss.mark_as_loss("ce")

    scores = {}
    _backend.select_backend_tf()
    prev_batch_dyn_size_ext = batch_dim.dyn_size_ext
    try:
        for tf_jit in (False, True):
            train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=8, seq_len=seq_len)
            train_data.init_seq_order(epoch=1)
            model_dir = tempfile.mkdtemp(prefix="returnn-test-rf-tf-engine-jit-")
            config = Config(
                {
                    "backend": "tensorflow",
                    "model": model_dir + "/model",
                    "tf_jit": tf_jit,
                    "extern_data": {
                        "data": {"dims": [batch_dim, time_dim, in_dim], "dtype": "float32"},
                        "classes": {"dims": [batch_dim, time_dim], "sparse_dim": out_dim, "dtype": "int32"},
                    },
                    "get_model": _get_model,
                    "train_step": _train_step,
                    "optimizer": {"class": "adam"},
                    "learning_rate": 0.05,
                    "batch_size": 20,
                    "max_seqs": 4,
                    "num_epochs": 2,
                }
            )
            try:
                with global_config_ctx(config):
                    engine = Engine(config=config)
                    engine.init_train_from_config(config=config, train_data=train_data)
                    engine.train()
                    scores[tf_jit] = engine.learning_rate_control.get_epoch_error_dict(2)["train_loss:ce"]
            finally:
                shutil.rmtree(model_dir, ignore_errors=True)
        print("train_loss:ce ep 2, tf_jit False vs True:", scores)
        assert scores[True] < 2.0, scores  # sane (ce starts around ln(3)=1.1, must not blow up)
        numpy.testing.assert_allclose(scores[False], scores[True], rtol=1e-4)
    finally:
        batch_dim.dyn_size_ext = prev_batch_dyn_size_ext
        rf.select_backend_torch()


def test_generate_batches_packed_batch_size():
    # The sum-of-lens batch budget (packed_batch_size) in the shared batcher
    # (Dataset._generate_batches), which the TF RF engine passes through.
    # Fixed seq len 5: budget 12 holds 2 seqs (sum 10), a third (15) exceeds it;
    # batch_size None is allowed when the packed budget is given.
    from returnn.datasets.generating import DummyDataset

    dataset = DummyDataset(input_dim=2, output_dim=3, num_seqs=7, seq_len=5)
    dataset.init_seq_order(epoch=1)
    batches = dataset.generate_batches(recurrent_net=True, batch_size=None, max_seqs=-1, packed_batch_size=12)
    seqs_per_batch = []
    while batches.has_more():
        (batch,) = batches.peek_next_n(1)
        seqs_per_batch.append(batch.num_slices)
        batches.advance(1)
    assert seqs_per_batch == [2, 2, 2, 1], seqs_per_batch


def test_engine_train_packed():
    # packed_tensors on the TF engine: the extern data is repacked IN-GRAPH into packed storage
    # (engine _pack_extern_data), so the model computes on sum(lens) frames instead of
    # n_seqs * max_len; the feed stays the padded placeholders. Varying seq lens (Task12AX,
    # 10..100) make the packing non-trivial. Same batching in the packed and padded run,
    # so the losses must agree up to reduce order. A third run additionally uses the
    # sum-of-lens batch budget (packed_batch_size, batch_size None) and just has to train sanely
    # (different batch composition, so no loss comparison).
    from returnn.config import Config, global_config_ctx
    from returnn.datasets.generating import Task12AXDataset
    from returnn.tf.engine_rf import Engine

    # noinspection PyProtectedMember
    from returnn.frontend import _backend

    n_data_dim, n_classes_dim = 9, 2

    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(n_data_dim, name="in")
    out_dim = Dim(n_classes_dim, name="out")

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.out = rf.Linear(in_dim, out_dim)

    # noinspection PyShadowingNames
    def _get_model(*, epoch: int, step: int, **_kwargs) -> rf.Module:
        return _Net()

    # noinspection PyShadowingNames
    def _train_step(*, model: _Net, extern_data: TensorDict, **_kwargs):
        logits = model.out(extern_data["data"])
        loss = rf.cross_entropy(estimated=logits, target=extern_data["classes"], axis=out_dim, estimated_type="logits")
        loss.mark_as_loss("ce")

    scores = {}
    _backend.select_backend_tf()
    prev_batch_dyn_size_ext = batch_dim.dyn_size_ext
    try:
        for name, extra_opts in {
            "padded": {},
            "packed": {"packed_tensors": True},
            # budget 120 with lens 10..100 really bites (with 200, max_seqs 3 dominated and the
            # batch composition -- and thus the score -- was identical to the other variants)
            "packed_budget": {"packed_tensors": True, "batch_size": None, "packed_batch_size": 120},
            # tf_static_shapes: every step feeds the same shapes (see _staticize_feed);
            # LAST in the dict: they set capacities on the shared dims, harmless without the
            # static flag, but keep the plain variants unpolluted anyway
            "static_padded": {
                "tf_static_shapes": {"batch_size_bound": 3, "dim_capacity": {"data": 100, "classes": 100}},
            },
            # packed + static: the feed itself is packed (flat total-bound buffer, _make_packed_feed);
            # max_seqs 3 seqs of len <= 100 make 300 a valid total bound
            "packed_static": {
                "packed_tensors": True,
                "tf_static_shapes": {
                    "batch_size_bound": 3,
                    "dim_capacity": {"data": 100, "classes": 100},
                    "packed_total_bound": {"data": 300, "classes": 300},
                },
            },
        }.items():
            train_data = Task12AXDataset(num_seqs=8)
            train_data.init_seq_order(epoch=1)
            model_dir = tempfile.mkdtemp(prefix="returnn-test-rf-tf-engine-packed-")
            config = Config(
                {
                    "backend": "tensorflow",
                    "model": model_dir + "/model",
                    "extern_data": {
                        "data": {"dims": [batch_dim, time_dim, in_dim], "dtype": "float32"},
                        "classes": {"dims": [batch_dim, time_dim], "sparse_dim": out_dim, "dtype": "int32"},
                    },
                    "get_model": _get_model,
                    "train_step": _train_step,
                    "optimizer": {"class": "adam"},
                    "learning_rate": 0.05,
                    "batch_size": 300,
                    "max_seqs": 3,
                    "num_epochs": 2,
                    **extra_opts,
                }
            )
            try:
                with global_config_ctx(config):
                    engine = Engine(config=config)
                    engine.init_train_from_config(config=config, train_data=train_data)
                    engine.train()
                    scores[name] = engine.learning_rate_control.get_epoch_error_dict(2)["train_loss:ce"]
            finally:
                shutil.rmtree(model_dir, ignore_errors=True)
        print("train_loss:ce ep 2 per variant:", scores)
        # ce starts around ln(2)=0.69 and the task is learnable, so anything near/above 1 is broken
        assert all(s < 1.0 for s in scores.values()), scores
        # identical batching in these variants, so identical losses up to reduce order
        # (packed_budget differs by construction: other batch composition)
        for name in ("packed", "static_padded", "packed_static"):
            numpy.testing.assert_allclose(scores["padded"], scores[name], rtol=1e-4, err_msg=name)
    finally:
        batch_dim.dyn_size_ext = prev_batch_dyn_size_ext
        batch_dim.size = None  # the static variants staticize the GLOBAL batch dim
        batch_dim.capacity = None
        rf.select_backend_torch()


def test_engine_train_run_ctx_epoch():
    # rf.get_run_ctx().epoch inside the step must advance with the epochs.
    # The graph is built ONCE for all epochs, so a Python int would bake the build-time epoch in
    # (and every epoch-scheduled dropout/schedule would silently freeze) -- it is an in-graph
    # variable, assigned per epoch. The step here scales the loss by the epoch, so a frozen
    # epoch shows up directly in the reported scores.
    from returnn.config import Config, global_config_ctx
    from returnn.datasets.generating import DummyDataset
    from returnn.tf.engine_rf import Engine

    # noinspection PyProtectedMember
    from returnn.frontend import _backend

    n_data_dim, n_classes_dim, seq_len = 2, 3, 5

    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(n_data_dim, name="in")
    out_dim = Dim(n_classes_dim, name="out")

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.out = rf.Linear(in_dim, out_dim)

    # noinspection PyShadowingNames
    def _train_step(*, model: _Net, extern_data: TensorDict, **_kwargs):
        logits = model.out(extern_data["data"])
        loss = rf.cross_entropy(estimated=logits, target=extern_data["classes"], axis=out_dim, estimated_type="logits")
        loss *= rf.cast(rf.get_run_ctx().get_epoch_tensor(), loss.dtype)
        loss.mark_as_loss("ce_x_epoch")

    _backend.select_backend_tf()
    prev_batch_dyn_size_ext = batch_dim.dyn_size_ext
    model_dir = tempfile.mkdtemp(prefix="returnn-test-rf-tf-engine-epoch-")
    try:
        train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=4, seq_len=seq_len)
        train_data.init_seq_order(epoch=1)
        config = Config(
            {
                "backend": "tensorflow",
                "model": model_dir + "/model",
                "extern_data": {
                    "data": {"dims": [batch_dim, time_dim, in_dim], "dtype": "float32"},
                    "classes": {"dims": [batch_dim, time_dim], "sparse_dim": out_dim, "dtype": "int32"},
                },
                "get_model": lambda **_kwargs: _Net(),
                "train_step": _train_step,
                "optimizer": {"class": "adam"},
                "learning_rate": 0.0,  # params fixed, so only the epoch factor moves the loss
                "batch_size": 20,
                "num_epochs": 3,
            }
        )
        with global_config_ctx(config):
            engine = Engine(config=config)
            engine.init_train_from_config(config=config, train_data=train_data)
            engine.train()
            scores = {
                ep: engine.learning_rate_control.get_epoch_error_dict(ep)["train_loss:ce_x_epoch"] for ep in (1, 2, 3)
            }
        print("train_loss:ce_x_epoch per epoch:", scores)
        # with a frozen epoch all three would be equal
        numpy.testing.assert_allclose(scores[2], 2 * scores[1], rtol=1e-5)
        numpy.testing.assert_allclose(scores[3], 3 * scores[1], rtol=1e-5)
    finally:
        batch_dim.dyn_size_ext = prev_batch_dyn_size_ext
        rf.select_backend_torch()
        shutil.rmtree(model_dir, ignore_errors=True)


def test_engine_forward_preload_from_files():
    # preload_from_files on the TF engine (combined recog models preload e.g. an LM):
    # after init, the prefixed params must equal the PRELOAD source checkpoint
    # and the remaining params the MAIN checkpoint -- verified directly against both files.
    import tensorflow as tf
    from returnn.config import Config, global_config_ctx
    from returnn.datasets.generating import DummyDataset
    from returnn.tf.engine_rf import Engine
    from returnn.tf.frontend_low_level import TFBackend

    # noinspection PyProtectedMember
    from returnn.frontend import _backend

    n_data_dim, n_classes_dim, seq_len = 2, 3, 5

    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(n_data_dim, name="in")
    out_dim = Dim(n_classes_dim, name="out")

    class _SubNet(rf.Module):  # what gets preloaded, e.g. the LM
        def __init__(self):
            super().__init__()
            self.out = rf.Linear(in_dim, out_dim)

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.out = rf.Linear(in_dim, out_dim)
            self.lm = _SubNet()

    # noinspection PyShadowingNames
    def _train_step(*, model, extern_data: TensorDict, **_kwargs):
        logits = model.out(extern_data["data"]) if not isinstance(model, _Net) else model.lm.out(extern_data["data"])
        loss = rf.cross_entropy(estimated=logits, target=extern_data["classes"], axis=out_dim, estimated_type="logits")
        loss.mark_as_loss("ce")

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict, **_kwargs):
        out = model.out(extern_data["data"])
        out.mark_as_output("out", shape=(batch_dim, time_dim, out_dim))

    extern_data_opts = {
        "data": {"dims": [batch_dim, time_dim, in_dim], "dtype": "float32"},
        "classes": {"dims": [batch_dim, time_dim], "sparse_dim": out_dim, "dtype": "int32"},
    }
    _backend.select_backend_tf()
    prev_batch_dyn_size_ext = batch_dim.dyn_size_ext
    dirs = {}
    try:
        # two 1-epoch trainings with different seeds: the main model (with .lm inside) and,
        # separately, the bare sub model whose checkpoint gets preloaded under the "lm." prefix
        for name, net_cls, seed in (("main", _Net, 1), ("sub", _SubNet, 2)):
            train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=4, seq_len=seq_len)
            train_data.init_seq_order(epoch=1)
            model_dir = tempfile.mkdtemp(prefix=f"returnn-test-rf-tf-engine-preload-{name}-")
            dirs[name] = model_dir
            config = Config(
                {
                    "backend": "tensorflow",
                    "model": model_dir + "/model",
                    "random_seed": seed,
                    "extern_data": extern_data_opts,
                    "get_model": lambda *, net_cls=net_cls, **_kwargs: net_cls(),
                    "train_step": _train_step,
                    "optimizer": {"class": "adam"},
                    "learning_rate": 0.05,
                    "batch_size": 20,
                    "num_epochs": 1,
                }
            )
            with global_config_ctx(config):
                engine = Engine(config=config)
                engine.init_train_from_config(config=config, train_data=train_data)
                engine.train()

        fwd_config = Config(
            {
                "backend": "tensorflow",
                "load": dirs["main"] + "/model.001",
                "preload_from_files": {"lm": {"prefix": "lm.", "filename": dirs["sub"] + "/model.001"}},
                "extern_data": extern_data_opts,
                "get_model": lambda **_kwargs: _Net(),
                "forward_step": _forward_step,
            }
        )
        with global_config_ctx(fwd_config):
            engine = Engine(config=fwd_config)
            engine.init_network_from_config(fwd_config)
            variables = {name: TFBackend.get_parameter_variable(p) for name, p in engine.get_model().named_parameters()}
            loaded = engine.session.run(variables)
        main_reader = tf.train.load_checkpoint(dirs["main"] + "/model.001")
        sub_reader = tf.train.load_checkpoint(dirs["sub"] + "/model.001")
        # different seeds: the preload must actually CHANGE the lm params vs the main checkpoint
        assert not numpy.allclose(main_reader.get_tensor("lm.out.weight"), sub_reader.get_tensor("out.weight"))
        for name, value in loaded.items():
            if name.startswith("lm."):
                expected = sub_reader.get_tensor(name[len("lm.") :])
            else:
                expected = main_reader.get_tensor(name)
            assert numpy.array_equal(value, expected), f"param {name} not from the expected checkpoint"
    finally:
        batch_dim.dyn_size_ext = prev_batch_dyn_size_ext
        rf.select_backend_torch()
        for model_dir in dirs.values():
            shutil.rmtree(model_dir, ignore_errors=True)


def test_full_model_torch_checkpoint_parity():
    # Item 9: take the target model's parameters from a real PyTorch checkpoint file
    # (as the PT engine writes it), load them into the TF model by RF parameter name,
    # and check that the outputs agree. This is the decisive correctness test:
    # unlike the other comparisons it does not rely on both sides drawing the same random init.
    import torch
    import returnn.tf.compat as tf_compat
    from returnn.tensor.utils import tensor_dict_fill_random_numpy_
    from returnn.torch.data.tensor_utils import tensor_dict_numpy_to_torch_
    from returnn.tf.frontend_low_level import TFBackend
    from returnn.tf.checkpoint_rf import load_torch_checkpoint_into_model, get_model_params

    # noinspection PyProtectedMember
    from returnn.frontend import _backend

    extern_data, get_model, forward_step, dims = _full_model_setup()
    extern_data.reset_content()
    tensor_dict_fill_random_numpy_(
        extern_data,
        dyn_dim_max_sizes={dims["time_dim"]: 32, dims["target_time_dim"]: 4},
        dyn_dim_min_sizes={dims["time_dim"]: 24, dims["target_time_dim"]: 2},
    )
    extern_data_raw = extern_data.as_raw_tensor_dict(expected_value_type=numpy.ndarray)
    ckpt_dir = tempfile.mkdtemp(prefix="returnn-test-rf-tf-parity-")
    ckpt_file = ckpt_dir + "/model.pt"

    def _outputs() -> Dict[str, numpy.ndarray]:
        rf.init_forward_step_run_ctx(epoch=1, step=0)
        forward_step(model=model, extern_data=extern_data)
        return rf.get_run_ctx().outputs

    try:
        # PT: build the model, write a checkpoint the way the PT engine does, and run it
        rf.select_backend_torch()
        rf.set_random_seed(42)
        extern_data.assign_from_raw_tensor_dict_(extern_data_raw)
        tensor_dict_numpy_to_torch_(extern_data)
        model = get_model(epoch=1, step=0)
        torch.save(
            {"model": {name: p.raw_tensor for name, p in model.named_parameters()}, "epoch": 1, "step": 0}, ckpt_file
        )
        outputs_pt = _outputs()
        # read the sizes now: the same Dim objects get re-bound to TF tensors below
        dyn_sizes = _dyn_sizes_numpy(outputs_pt)
        templates = {key: value.copy_template() for key, value in outputs_pt.data.items()}
        out_pt = {key: value.raw_tensor.detach().numpy() for key, value in outputs_pt.data.items()}

        # TF: fresh model with a DIFFERENT init, then the PT parameters loaded into it by name
        extern_data.reset_content()
        _backend.select_backend_tf()
        with tf_compat.v1.Graph().as_default(), tf_compat.v1.Session().as_default() as session:
            rf.set_random_seed(1234)
            extern_data.assign_from_raw_tensor_dict_(extern_data_raw)
            _tensor_dict_numpy_to_tf(extern_data)
            with TFBackend.deferred_parameter_creation():
                model = get_model(epoch=1, step=0)
            TFBackend.create_parameters(model)
            outputs_tf = _outputs()
            session.run(tf_compat.v1.global_variables_initializer())
            before = get_model_params(model, session)
            load_torch_checkpoint_into_model(model, session, ckpt_file)
            after = get_model_params(model, session)
            out_tf = session.run({key: value.raw_tensor for key, value in outputs_tf.data.items()})
    finally:
        extern_data.reset_content()
        extern_data.assign_from_raw_tensor_dict_(extern_data_raw)
        rf.select_backend_torch()
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    # the load really changed the parameters (otherwise the comparison below would prove nothing)
    assert any(not numpy.allclose(before[name], after[name]) for name in before), "checkpoint load was a no-op"
    # "ctc" is the one output whose value depends on the TF version: it is the result of
    # tf.nn.ctc_loss, and TF 2.10 (what CI pins) computes it about 5e-3 relative away from what
    # TF 2.18/2.20 compute on identical inputs. That is the framework's op, not this backend:
    # the same PyTorch reference comes out identical on both platforms, our own float32 CTC
    # agrees with a float64 reference to 5e-8, and the loss is well conditioned here
    # (perturbing its logits by rel 1e-6 moves it by 4e-7).
    # Loosening it costs no coverage, because "logits" guards the same encoder far more sharply:
    # measured, it responds to an encoder perturbation ~750x more strongly than "ctc" does,
    # so an encoder that had really drifted would fail there first.
    out_tf = {key: _mask_padded(templates[key], value, dyn_sizes) for key, value in out_tf.items()}
    out_pt = {key: _mask_padded(templates[key], value, dyn_sizes) for key, value in out_pt.items()}
    _assert_all_close("checkpoint parity", out_tf, out_pt, rtol={"ctc": 2e-2})


def test_native_ctc_two_instances_joint_fetch():
    # Two native CTC instances fetched in ONE session.run used to corrupt each other
    # (both sometimes returning the no-path sentinel 0 for valid sequences):
    # the FastBaumWelch scratch state_buffer was an op INPUT the kernel wrote into,
    # and CSE merged the identical zeros subgraphs, so concurrent instances shared one buffer.
    # The op now takes only the buffer SHAPE and allocates kernel-owned scratch (is_buffer).
    import tensorflow as tf
    import returnn.tf.compat as tf_compat
    from returnn.tf import native_op as tf_native_op

    rng = numpy.random.default_rng(3)
    n_time, n_batch, dim, n_targets = 6, 5, 12, 3
    input_lens = numpy.array([6, 5, 6, 4, 5], "int32")
    target_lens = numpy.array([3, 2, 3, 3, 3], "int32")
    # no adjacent repeats -> every seq has a valid alignment
    targets_np = numpy.stack([rng.permutation(dim - 1)[:n_targets] for _ in range(n_batch)]).astype("int32")

    def _make(logits_np):
        return tf_native_op.ctc_loss(
            logits=tf.constant(logits_np),
            logits_seq_lens=tf.constant(input_lens),
            logits_time_major=False,
            targets=tf.constant(targets_np),
            targets_seq_lens=tf.constant(target_lens),
            blank_index=dim - 1,
            zero_infinity=False,
        )

    logits_a = rng.standard_normal((n_batch, n_time, dim)).astype("float32")
    logits_b = rng.standard_normal((n_batch, n_time, dim)).astype("float32")
    with tf_compat.v1.Graph().as_default(), tf_compat.v1.Session() as session:
        loss_a, loss_b = _make(logits_a), _make(logits_b)
        sep_a = session.run(loss_a)
        sep_b = session.run(loss_b)
        for _ in range(3):  # the corruption was schedule-dependent, repeat a few times
            joint_a, joint_b = session.run((loss_a, loss_b))
            numpy.testing.assert_allclose(sep_a, joint_a, rtol=0, atol=0, err_msg="a: joint fetch differs")
            numpy.testing.assert_allclose(sep_b, joint_b, rtol=0, atol=0, err_msg="b: joint fetch differs")
    assert numpy.all(numpy.isfinite(sep_a)) and numpy.all(sep_a > 0)
    assert numpy.all(numpy.isfinite(sep_b)) and numpy.all(sep_b > 0)


def test_engine_eval_no_dropout():
    # The engine builds ONE graph, so the train flag has to be dynamic: eval must run without
    # dropout. With dropout p=0.9 a train-graph eval would be far off, and eval is deterministic
    # only if the flag really reaches rf.dropout, so both properties are checked here.
    from returnn.config import Config, global_config_ctx
    from returnn.datasets.generating import DummyDataset
    from returnn.tf.engine_rf import Engine

    # noinspection PyProtectedMember
    from returnn.frontend import _backend

    n_data_dim, n_classes_dim, seq_len = 2, 3, 5
    dev_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=4, seq_len=seq_len)
    dev_data.init_seq_order(epoch=1)
    train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=4, seq_len=seq_len)
    train_data.init_seq_order(epoch=1)

    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(n_data_dim, name="in")
    out_dim = Dim(n_classes_dim, name="out")

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.out = rf.Linear(in_dim, out_dim)

    # noinspection PyShadowingNames
    def _get_model(*, epoch: int, step: int, **_kwargs) -> rf.Module:
        return _Net()

    # noinspection PyShadowingNames
    def _train_step(*, model: _Net, extern_data: TensorDict, **_kwargs):
        x = rf.dropout(extern_data["data"], 0.9, axis=in_dim)
        loss = rf.cross_entropy(
            estimated=model.out(x), target=extern_data["classes"], axis=out_dim, estimated_type="logits"
        )
        loss.mark_as_loss("ce")

    config = Config(
        {
            "backend": "tensorflow",
            "extern_data": {
                "data": {"dims": [batch_dim, time_dim, in_dim], "dtype": "float32"},
                "classes": {"dims": [batch_dim, time_dim], "sparse_dim": out_dim, "dtype": "int32"},
            },
            "get_model": _get_model,
            "train_step": _train_step,
            "optimizer": {"class": "adam"},
            "learning_rate": 0.0,  # no updates, so repeated evals must give the same number
            "batch_size": 20,
            "max_seqs": 4,
            "num_epochs": 2,
        }
    )

    _backend.select_backend_tf()
    prev_batch_dyn_size_ext = batch_dim.dyn_size_ext
    try:
        with global_config_ctx(config):
            engine = Engine(config=config)
            engine.init_train_from_config(config=config, train_data=train_data, dev_data=dev_data)
            engine.train()
            errors = [engine.learning_rate_control.get_epoch_error_dict(ep) for ep in (1, 2)]
            dev_ce = [e["dev_loss:ce"] for e in errors]
            train_ce = [e["train_loss:ce"] for e in errors]
    finally:
        batch_dim.dyn_size_ext = prev_batch_dyn_size_ext
        rf.select_backend_torch()

    print("dev ce:", dev_ce, "train ce:", train_ce)
    # Two independent checks that the flag really reaches rf.dropout:
    # 1. eval is deterministic -- with dropout on, each run would draw a fresh mask.
    assert dev_ce[0] == dev_ce[1], f"eval is not deterministic, dropout still active? {dev_ce}"
    # 2. the value differs from the dropped-out one. With p=0.9 the train inputs are nearly all
    #    zero, so the logits are ~bias=0 and the train CE sits at ln(num_classes); the eval CE
    #    sees the real inputs and must differ from that.
    assert abs(train_ce[0] - numpy.log(n_classes_dim)) < 1e-4, f"expected the train CE at ln(3), got {train_ce}"
    assert abs(dev_ce[0] - numpy.log(n_classes_dim)) > 1e-3, f"eval CE looks dropped-out: {dev_ce}"


def test_engine_dynamic_learning_rate():
    # The whole LR schedule of a real setup can live in dynamic_learning_rate, so the engine must
    # apply it per step. Without it the run would silently train at a constant LR.
    from returnn.config import Config, global_config_ctx
    from returnn.datasets.generating import DummyDataset
    from returnn.tf.engine_rf import Engine

    # noinspection PyProtectedMember
    from returnn.frontend import _backend

    n_data_dim, n_classes_dim, seq_len = 2, 3, 5
    train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=8, seq_len=seq_len)
    train_data.init_seq_order(epoch=1)

    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(n_data_dim, name="in")
    out_dim = Dim(n_classes_dim, name="out")
    seen = []

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.out = rf.Linear(in_dim, out_dim)

    # noinspection PyShadowingNames
    def _get_model(*, epoch: int, step: int, **_kwargs) -> rf.Module:
        return _Net()

    # noinspection PyShadowingNames
    def _train_step(*, model: _Net, extern_data: TensorDict, **_kwargs):
        loss = rf.cross_entropy(
            estimated=model.out(extern_data["data"]),
            target=extern_data["classes"],
            axis=out_dim,
            estimated_type="logits",
        )
        loss.mark_as_loss("ce")

    def _dyn_lr(
        *, global_train_step: int, epoch: int, epoch_continuous: float, learning_rate: float, **_kwargs
    ) -> float:
        seen.append((global_train_step, epoch, epoch_continuous, learning_rate))
        return 0.001 * (global_train_step + 1)  # a schedule the engine cannot guess

    config = Config(
        {
            "backend": "tensorflow",
            "extern_data": {
                "data": {"dims": [batch_dim, time_dim, in_dim], "dtype": "float32"},
                "classes": {"dims": [batch_dim, time_dim], "sparse_dim": out_dim, "dtype": "int32"},
            },
            "get_model": _get_model,
            "train_step": _train_step,
            "optimizer": {"class": "adam"},
            "learning_rate": 0.5,
            "dynamic_learning_rate": _dyn_lr,
            "batch_size": 20,
            "max_seqs": 4,
            "num_epochs": 2,
        }
    )

    _backend.select_backend_tf()
    prev_batch_dyn_size_ext = batch_dim.dyn_size_ext
    try:
        with global_config_ctx(config):
            engine = Engine(config=config)
            engine.init_train_from_config(config=config, train_data=train_data)
            engine.train()
            # the LR the optimizer ended up with must be the one the schedule asked for
            final_lr = float(engine.session.run(engine._updater.learning_rate_var))
    finally:
        batch_dim.dyn_size_ext = prev_batch_dyn_size_ext
        rf.select_backend_torch()

    print("dyn_lr called with (step, epoch, epoch_continuous, lr):", seen)
    assert seen, "dynamic_learning_rate was never called -- the schedule would be ignored"
    steps = [s for s, _, _, _ in seen]
    assert steps == sorted(steps) and steps[0] == 0, f"unexpected step sequence: {steps}"
    assert len(steps) == len(set(steps)), f"a step was repeated: {steps}"
    assert {e for _, e, _, _ in seen} == {1, 2}, f"epochs not passed through: {seen}"
    assert all(lr == 0.5 for _, _, _, lr in seen), f"base learning_rate not passed through: {seen}"
    # epoch_continuous comes from the dataset's complete_frac, so it advances within [epoch-1, epoch)
    # from the very first epoch, and ends at the epoch boundary
    for ep in (1, 2):
        got = [ec for _, e, ec, _ in seen if e == ep]
        assert got == sorted(got), f"epoch {ep} epoch_continuous not monotonic: {got}"
        assert ep - 1 < got[0] <= got[-1] == float(ep), f"epoch {ep} epoch_continuous: {got}"
    numpy.testing.assert_allclose(final_lr, 0.001 * (steps[-1] + 1), rtol=1e-6)


def test_engine_eval_datasets_save_interval_cleanup():
    # eval_datasets / save_interval / cleanup_old_models decide what a long run reports and keeps
    # on disk, so all three must actually take effect rather than be ignored.
    import glob
    import tempfile
    from returnn.config import Config, global_config_ctx
    from returnn.datasets.generating import DummyDataset
    from returnn.tf.engine_rf import Engine

    # noinspection PyProtectedMember
    from returnn.frontend import _backend

    n_data_dim, n_classes_dim, seq_len = 2, 3, 5
    train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=8, seq_len=seq_len)
    train_data.init_seq_order(epoch=1)

    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(n_data_dim, name="in")
    out_dim = Dim(n_classes_dim, name="out")

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.out = rf.Linear(in_dim, out_dim)

    # noinspection PyShadowingNames
    def _get_model(**_kwargs) -> rf.Module:
        return _Net()

    # noinspection PyShadowingNames
    def _train_step(*, model: _Net, extern_data: TensorDict, **_kwargs):
        rf.cross_entropy(
            estimated=model.out(extern_data["data"]),
            target=extern_data["classes"],
            axis=out_dim,
            estimated_type="logits",
        ).mark_as_loss("ce")

    num_epochs = 6
    keep_last_n = 2
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = Config(
            {
                "backend": "tensorflow",
                "extern_data": {
                    "data": {"dims": [batch_dim, time_dim, in_dim], "dtype": "float32"},
                    "classes": {"dims": [batch_dim, time_dim], "sparse_dim": out_dim, "dtype": "int32"},
                },
                "get_model": _get_model,
                "train_step": _train_step,
                "optimizer": {"class": "adam"},
                "learning_rate": 0.01,
                "batch_size": 20,
                "max_seqs": 4,
                "num_epochs": num_epochs,
                "model": tmp_dir + "/model/epoch",
                "learning_rate_file": tmp_dir + "/learning_rates",
                # the extra eval dataset a real setup uses for its devtrain scores
                "eval_datasets": {
                    "devtrain": {
                        "class": "DummyDataset",
                        "input_dim": n_data_dim,
                        "output_dim": n_classes_dim,
                        "num_seqs": 4,
                        "seq_len": seq_len,
                    }
                },
                "save_interval": 2,
                "cleanup_old_models": {"keep_last_n": keep_last_n, "keep_best_n": 1, "keep": []},
            }
        )

        _backend.select_backend_tf()
        prev_batch_dyn_size_ext = batch_dim.dyn_size_ext
        try:
            with global_config_ctx(config):
                engine = Engine(config=config)
                engine.init_train_from_config(config=config, train_data=train_data)
                assert set(engine.eval_datasets.keys()) == {"devtrain"}, engine.eval_datasets
                engine.train()
                lr_control = engine.learning_rate_control
        finally:
            batch_dim.dyn_size_ext = prev_batch_dyn_size_ext
            rf.select_backend_torch()

        saved = sorted(int(fn.split(".")[-2]) for fn in glob.glob(tmp_dir + "/model/epoch.*.index"))
        print("saved epochs on disk:", saved)
        # save_interval=2 plus the final epoch; cleanup then keeps only the last n (plus the best)
        assert saved, "nothing was saved"
        assert all(ep % 2 == 0 for ep in saved), f"save_interval ignored: {saved}"
        assert saved[-1] == num_epochs, f"final epoch not saved: {saved}"
        assert len(saved) <= keep_last_n + 1, f"cleanup_old_models ignored: {saved}"
        assert set(range(num_epochs - 2 * keep_last_n + 2, num_epochs + 1, 2)).issubset(saved), saved

    # the extra eval dataset must have produced scores, else it silently did nothing
    for epoch in range(1, num_epochs + 1):
        errors = lr_control.get_epoch_error_dict(epoch)
        assert any(key.startswith("devtrain_") for key in errors), f"epoch {epoch}: {errors}"


def test_engine_tf_amp_bfloat16():
    # tf_amp must actually reach the graph: the matmuls in bfloat16, the parameters and the
    # optimizer still float32. Silently running fp32 would look fine but be a different setup.
    from returnn.config import Config, global_config_ctx
    from returnn.datasets.generating import DummyDataset
    from returnn.tf.engine_rf import Engine
    from returnn.tf.frontend_low_level import TFBackend

    # noinspection PyProtectedMember
    from returnn.frontend import _backend

    n_data_dim, n_classes_dim, seq_len = 2, 3, 5
    train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=8, seq_len=seq_len)
    train_data.init_seq_order(epoch=1)

    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(n_data_dim, name="in")
    out_dim = Dim(n_classes_dim, name="out")

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.hidden = rf.Linear(in_dim, Dim(4, name="hidden"))
            self.out = rf.Linear(self.hidden.out_dim, out_dim)

    # noinspection PyShadowingNames
    def _get_model(**_kwargs) -> rf.Module:
        return _Net()

    # noinspection PyShadowingNames
    def _train_step(*, model: _Net, extern_data: TensorDict, **_kwargs):
        rf.cross_entropy(
            estimated=model.out(rf.relu(model.hidden(extern_data["data"]))),
            target=extern_data["classes"],
            axis=out_dim,
            estimated_type="logits",
        ).mark_as_loss("ce")

    config = Config(
        {
            "backend": "tensorflow",
            "extern_data": {
                "data": {"dims": [batch_dim, time_dim, in_dim], "dtype": "float32"},
                "classes": {"dims": [batch_dim, time_dim], "sparse_dim": out_dim, "dtype": "int32"},
            },
            "get_model": _get_model,
            "train_step": _train_step,
            "optimizer": {"class": "adam"},
            "learning_rate": 0.01,
            "batch_size": 20,
            "max_seqs": 4,
            "num_epochs": 1,
            "tf_amp": "bfloat16",
        }
    )

    _backend.select_backend_tf()
    prev_batch_dyn_size_ext = batch_dim.dyn_size_ext
    try:
        with global_config_ctx(config):
            engine = Engine(config=config)
            engine.init_train_from_config(config=config, train_data=train_data)
            assert rf.get_amp_policy() is None, "the policy must not leak out of the graph build"
            graph = engine.session.graph
            matmuls = [op for op in graph.get_operations() if op.type in ("MatMul", "BatchMatMulV2", "Einsum")]
            assert matmuls, "no matmul in the graph?"
            dtypes = {out.dtype.name for op in matmuls for out in op.outputs}
            assert "bfloat16" in dtypes, f"tf_amp did not reach the matmuls, dtypes {dtypes}"
            params = dict(engine.get_model().named_parameters())
            assert params and all(p.dtype == "float32" for p in params.values()), (
                f"parameters must stay float32: {[(n, p.dtype) for n, p in params.items()]}"
            )
            # the loss is float32 (losses are computed there), and training still runs
            assert engine._loss.dtype.name == "float32", engine._loss.dtype
            engine.train()
            after = engine.session.run({name: TFBackend.get_parameter_variable(p) for name, p in params.items()})
    finally:
        batch_dim.dyn_size_ext = prev_batch_dyn_size_ext
        rf.select_backend_torch()

    for name, value in after.items():
        assert value.dtype == numpy.float32, f"{name}: {value.dtype}"
        assert numpy.all(numpy.isfinite(value)), f"{name} is not finite after training in bf16"


def test_engine_log_grad_norm_and_batch_size():
    # log_grad_norm and log_batch_size are diagnostics of a real run, and both are per step,
    # so they must be in the graph / in the step log and must not land in the epoch scores.
    import io
    from returnn.config import Config, global_config_ctx
    from returnn.datasets.generating import DummyDataset
    from returnn.log import log
    from returnn.tf.engine_rf import Engine

    # noinspection PyProtectedMember
    from returnn.frontend import _backend

    n_data_dim, n_classes_dim, seq_len = 2, 3, 5
    train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=8, seq_len=seq_len)
    train_data.init_seq_order(epoch=1)

    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(n_data_dim, name="in")
    out_dim = Dim(n_classes_dim, name="out")

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.out = rf.Linear(in_dim, out_dim)

    # noinspection PyShadowingNames
    def _get_model(**_kwargs) -> rf.Module:
        return _Net()

    # noinspection PyShadowingNames
    def _train_step(*, model: _Net, extern_data: TensorDict, **_kwargs):
        rf.cross_entropy(
            estimated=model.out(extern_data["data"]),
            target=extern_data["classes"],
            axis=out_dim,
            estimated_type="logits",
        ).mark_as_loss("ce")

    config = Config(
        {
            "backend": "tensorflow",
            "extern_data": {
                "data": {"dims": [batch_dim, time_dim, in_dim], "dtype": "float32"},
                "classes": {"dims": [batch_dim, time_dim], "sparse_dim": out_dim, "dtype": "int32"},
            },
            "get_model": _get_model,
            "train_step": _train_step,
            "optimizer": {"class": "adam"},
            "learning_rate": 0.01,
            "batch_size": 20,
            "max_seqs": 4,
            "num_epochs": 1,
            "log_grad_norm": True,
            "log_batch_size": True,
            "gradient_clip_global_norm": 5.0,
        }
    )

    _backend.select_backend_tf()
    prev_batch_dyn_size_ext = batch_dim.dyn_size_ext
    prev_v5 = log.v5
    captured = io.StringIO()
    try:
        with global_config_ctx(config):
            engine = Engine(config=config)
            engine.init_train_from_config(config=config, train_data=train_data)
            assert "grad_norm:p2" in engine._extra_fetches, engine._extra_fetches
            log.v5 = captured
            engine.train()
            lr_control = engine.learning_rate_control
    finally:
        log.v5 = prev_v5
        batch_dim.dyn_size_ext = prev_batch_dyn_size_ext
        rf.select_backend_torch()

    out = captured.getvalue()
    print(out)
    step_lines = [line for line in out.splitlines() if line.startswith("ep 1 train, step ")]
    assert step_lines, f"no per-step log line in:\n{out}"
    for line in step_lines:
        assert "grad_norm:p2 " in line, line
        assert "batch_size:" in line, line
        assert "loss " in line, line
    # the diagnostics are not scores of the epoch
    errors = lr_control.get_epoch_error_dict(1)
    assert not any("grad_norm" in key or "batch_size" in key for key in errors), errors


def test_engine_forward_with_callback():
    # The forward task end to end: train a model, save it, then run the "forward" task on a fresh
    # engine and check the callback gets one entry per sequence, with the batch dim gone and the
    # padding cut off. That per-seq shaping is where a forward implementation usually goes wrong.
    import tempfile
    from returnn.config import Config, global_config_ctx
    from returnn.datasets.generating import DummyDataset
    from returnn.forward_iface import ForwardCallbackIface
    from returnn.tf.engine_rf import Engine

    # noinspection PyProtectedMember
    from returnn.frontend import _backend

    n_data_dim, n_classes_dim, seq_len = 2, 3, 5
    data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=7, seq_len=seq_len)
    data.init_seq_order(epoch=1)

    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(n_data_dim, name="in")
    out_dim = Dim(n_classes_dim, name="out")

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.out = rf.Linear(in_dim, out_dim)

    # noinspection PyShadowingNames
    def _get_model(**_kwargs) -> rf.Module:
        return _Net()

    # noinspection PyShadowingNames
    def _train_step(*, model: _Net, extern_data: TensorDict, **_kwargs):
        rf.cross_entropy(
            estimated=model.out(extern_data["data"]),
            target=extern_data["classes"],
            axis=out_dim,
            estimated_type="logits",
        ).mark_as_loss("ce")

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict, **_kwargs):
        logits = model.out(extern_data["data"])
        logits.mark_as_output("logits", shape=(batch_dim, time_dim, out_dim))
        rf.reduce_max(logits, axis=out_dim).mark_as_output("best", shape=(batch_dim, time_dim))

    seen = []

    def _dyn_size(dim: Dim):
        """:return: the dim's size for this sequence, None if it is static"""
        return None if dim.dyn_size_ext is None else int(dim.dyn_size_ext.raw_tensor)

    class _Callback(ForwardCallbackIface):
        def __init__(self):
            self.model = None

        def init(self, *, model):
            self.model = model

        def process_seq(self, *, seq_tag: str, outputs: TensorDict):
            # dims[0] is the spatial dim of both outputs, the batch dim being gone here
            seen.append(
                (
                    seq_tag,
                    {
                        k: (v.raw_tensor.shape, v.dims[0].dimension, _dyn_size(v.dims[0]))
                        for k, v in outputs.data.items()
                    },
                )
            )

        def finish(self):
            seen.append(("<finish>", {}))

    with tempfile.TemporaryDirectory() as tmp_dir:
        common = {
            "backend": "tensorflow",
            "extern_data": {
                "data": {"dims": [batch_dim, time_dim, in_dim], "dtype": "float32"},
                "classes": {"dims": [batch_dim, time_dim], "sparse_dim": out_dim, "dtype": "int32"},
            },
            "get_model": _get_model,
            "batch_size": 20,
            "max_seqs": 3,
            "model": tmp_dir + "/model/epoch",
            "learning_rate_file": tmp_dir + "/learning_rates",
        }

        _backend.select_backend_tf()
        prev_batch_dyn_size_ext = batch_dim.dyn_size_ext
        try:
            train_config = Config(
                {**common, "train_step": _train_step, "optimizer": {"class": "adam"}, "num_epochs": 1}
            )
            with global_config_ctx(train_config):
                engine = Engine(config=train_config)
                engine.init_train_from_config(config=train_config, train_data=data)
                engine.train()

            # a FRESH engine, forward task only, loading that checkpoint
            fwd_config = Config({**common, "forward_step": _forward_step, "task": "forward"})
            with global_config_ctx(fwd_config):
                engine = Engine(config=fwd_config)
                engine.init_network_from_config(config=fwd_config)
                callback = _Callback()
                engine.forward_with_callback(dataset=data, callback=callback)
                assert callback.model is not None, "callback.init(model=...) was not called"
        finally:
            batch_dim.dyn_size_ext = prev_batch_dyn_size_ext
            rf.select_backend_torch()

    assert seen and seen[-1][0] == "<finish>", f"finish() not called last: {[t for t, _ in seen]}"
    entries = seen[:-1]
    assert len(entries) == data.num_seqs, f"expected {data.num_seqs} seqs, got {len(entries)}"
    tags = [t for t, _ in entries]
    assert len(set(tags)) == len(tags), f"duplicate seq tags: {tags}"
    # not data.get_tag(i): the dataset only keeps the currently loaded seqs, and iteration is over
    assert tags == ["seq-%i" % i for i in range(data.num_seqs)], tags
    for tag, shapes in entries:
        # batch dim gone and the padding cut off,
        # but the spatial dim stays dynamic, as in the PyTorch engine:
        # the callback reads the seq length off dyn_size_ext,
        # so a static dim would leave the output without any length.
        assert shapes["logits"] == ((seq_len, n_classes_dim), None, seq_len), (tag, shapes)
        assert shapes["best"] == ((seq_len,), None, seq_len), (tag, shapes)


def test_engine_train_extern_data_dim_tags_without_size_template():
    # A dim declared as `Dim(None, name=...)` in extern_data (what `dim_tags` in a real config
    # gives) carries NO dyn_size_ext template -- unlike Dim(Tensor(...)), which every other test
    # here uses. The engine must create the template itself, otherwise the dim stays size-less
    # and anything that masks (rf.dropout's seq mask, copy_masked, masked reduces) hits None.
    # This is what the first production launch died on.
    from returnn.config import Config, global_config_ctx
    from returnn.datasets.generating import DummyDataset
    from returnn.tf.engine_rf import Engine

    # noinspection PyProtectedMember
    from returnn.frontend import _backend

    n_data_dim, n_classes_dim, seq_len = 2, 3, 5
    train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=8, seq_len=seq_len)
    train_data.init_seq_order(epoch=1)

    time_dim = Dim(None, name="time")  # no size template, as a config's dim_tags gives it
    in_dim = Dim(n_data_dim, name="in")
    out_dim = Dim(n_classes_dim, name="out")
    assert time_dim.dyn_size_ext is None, "the point of this test"

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.out = rf.Linear(in_dim, out_dim)

    # noinspection PyShadowingNames
    def _get_model(**_kwargs) -> rf.Module:
        return _Net()

    # noinspection PyShadowingNames
    def _train_step(*, model: _Net, extern_data: TensorDict, **_kwargs):
        x = extern_data["data"]
        # masking needs the dim's size: this is what failed in production
        x = x.copy_masked(0.0, dims=[time_dim])
        loss = rf.cross_entropy(
            estimated=model.out(x), target=extern_data["classes"], axis=out_dim, estimated_type="logits"
        )
        loss.mark_as_loss("ce")
        # a masked reduce over the spatial dim needs it too
        rf.reduce_mean(loss, axis=loss.dims).mark_as_loss("ce_mean", as_error=True)

    config = Config(
        {
            "backend": "tensorflow",
            "extern_data": {
                "data": {"dims": [batch_dim, time_dim, in_dim], "dtype": "float32"},
                "classes": {"dims": [batch_dim, time_dim], "sparse_dim": out_dim, "dtype": "int32"},
            },
            "get_model": _get_model,
            "train_step": _train_step,
            "optimizer": {"class": "adam"},
            "learning_rate": 0.01,
            "batch_size": 20,
            "max_seqs": 3,
            "num_epochs": 1,
        }
    )

    _backend.select_backend_tf()
    prev_batch_dyn_size_ext = batch_dim.dyn_size_ext
    try:
        with global_config_ctx(config):
            engine = Engine(config=config)
            engine.init_train_from_config(config=config, train_data=train_data)
            # the engine created the missing template and wired a placeholder to it
            assert time_dim.dyn_size_ext is not None, "engine did not create the size template"
            assert time_dim.dyn_size_ext.raw_tensor is not None, "size template got no placeholder"
            engine.train()
    finally:
        batch_dim.dyn_size_ext = prev_batch_dyn_size_ext
        time_dim.reset_raw()
        rf.select_backend_torch()


def test_reshape_ops_keep_static_shape_with_placeholders():
    # tf.reshape with a stacked (tensor) target shape loses the STATIC shape,
    # and RF then rejects the raw tensor
    # ("Mismatching shape: Raw tensor (None, None, None) vs Tensor ...").
    # It only shows up when the input shapes are dynamic, i.e. fed through PLACEHOLDERS:
    # run_model feeds constants, which is why the full-model tests never caught it,
    # while the production run died on it inside ConformerConvSubsample.
    import returnn.tf.compat as tf_compat
    from returnn.tf.frontend_low_level import TFBackend

    # noinspection PyProtectedMember
    from returnn.frontend import _backend

    _backend.select_backend_tf()
    prev_batch_dyn_size_ext = batch_dim.dyn_size_ext
    try:
        with tf_compat.v1.Graph().as_default(), tf_compat.v1.Session().as_default() as session:
            import tensorflow as tf

            time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
            f1, f2 = Dim(3, name="f1"), Dim(4, name="f2")
            x = Tensor("x", dims=[batch_dim, time_dim, f1, f2], dtype="float32")
            x.raw_tensor = TFBackend.create_placeholder_raw(x)
            time_dim.dyn_size_ext.raw_tensor = TFBackend.create_placeholder_raw(time_dim.dyn_size_ext)
            batch_dim.dyn_size_ext = Tensor("batch", dims=(), dtype="int32")
            batch_dim.dyn_size_ext.raw_tensor = tf.shape(x.raw_tensor)[0]

            merged, merged_dim = rf.merge_dims(x, dims=[f1, f2])
            # the static part of the shape must survive, else RF would already have rejected it
            assert merged.raw_tensor.shape.as_list() == [None, None, 12], merged.raw_tensor.shape
            split = rf.split_dims(merged, axis=merged_dim, dims=[f1, f2])
            assert split.raw_tensor.shape.as_list() == [None, None, 3, 4], split.raw_tensor.shape

            values = numpy.random.RandomState(42).normal(size=(2, 5, 3, 4)).astype("float32")
            out_merged, out_split = session.run(
                [merged.raw_tensor, split.raw_tensor],
                feed_dict={x.raw_tensor: values, time_dim.dyn_size_ext.raw_tensor: numpy.array([5, 4], "int32")},
            )
    finally:
        batch_dim.dyn_size_ext = prev_batch_dyn_size_ext
        time_dim.reset_raw()
        rf.select_backend_torch()

    assert out_merged.shape == (2, 5, 12) and out_split.shape == (2, 5, 3, 4)
    numpy.testing.assert_allclose(out_merged, values.reshape(2, 5, 12), rtol=1e-6)
    numpy.testing.assert_allclose(out_split, values, rtol=1e-6)


def test_engine_weight_decay_modules_blacklist():
    # The production config passes optimizer.weight_decay_modules_blacklist.
    # The optimizer class does not know that option ("Argument(s) not recognized"),
    # and weight decay must be off for exactly those parameters,
    # so the engine tags their variables with per-variable optimizer opts instead.
    # Same rule as the PyTorch updater: biases + the blacklisted module types.
    from returnn.config import Config, global_config_ctx
    from returnn.datasets.generating import DummyDataset
    from returnn.tf.engine_rf import Engine
    from returnn.tf.frontend_low_level import TFBackend

    # noinspection PyProtectedMember
    from returnn.frontend import _backend

    n_data_dim, n_classes_dim, seq_len = 2, 3, 5
    train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=4, seq_len=seq_len)
    train_data.init_seq_order(epoch=1)

    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(n_data_dim, name="in")
    out_dim = Dim(n_classes_dim, name="out")

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.emb = rf.Embedding(out_dim, in_dim)  # blacklisted type
            self.lin = rf.Linear(in_dim, out_dim)  # weight -> decay, bias -> no decay

    # noinspection PyShadowingNames
    def _get_model(**_kwargs) -> rf.Module:
        return _Net()

    # noinspection PyShadowingNames
    def _train_step(*, model: _Net, extern_data: TensorDict, **_kwargs):
        x = extern_data["data"] + rf.reduce_mean(model.emb(extern_data["classes"]), axis=time_dim)
        rf.cross_entropy(
            estimated=model.lin(x), target=extern_data["classes"], axis=out_dim, estimated_type="logits"
        ).mark_as_loss("ce")

    config = Config(
        {
            "backend": "tensorflow",
            "extern_data": {
                "data": {"dims": [batch_dim, time_dim, in_dim], "dtype": "float32"},
                "classes": {"dims": [batch_dim, time_dim], "sparse_dim": out_dim, "dtype": "int32"},
            },
            "get_model": _get_model,
            "train_step": _train_step,
            "optimizer": {
                "class": "adamw",
                "weight_decay": 0.01,
                "weight_decay_modules_blacklist": ["rf.Embedding"],
            },
            "learning_rate": 0.01,
            "batch_size": 20,
            "max_seqs": 2,
            "num_epochs": 1,
        }
    )

    # noinspection PyProtectedMember
    from returnn.tf.engine_rf import _no_weight_decay_params

    _backend.select_backend_tf()
    prev_batch_dyn_size_ext = batch_dim.dyn_size_ext
    try:
        with global_config_ctx(config):
            engine = Engine(config=config)
            engine.init_train_from_config(config=config, train_data=train_data)
            model = engine.get_model()
            all_names = {name for name, _ in model.named_parameters()}
            no_wd = set(_no_weight_decay_params(model, config.typed_value("optimizer")))
            decayed = {name for name, p in model.named_parameters() if any(p is q for q in engine._weight_decay_params)}
            # "adamw" must NOT reach the updater: it maps to a Keras optimizer whose wrapper is
            # dead on Keras 3. The engine runs TF1-native Adam and decays itself.
            assert engine._weight_decay == 0.01, engine._weight_decay
            before = engine.session.run(TFBackend.get_parameter_variable(dict(model.named_parameters())["lin.weight"]))
            engine.train()
            after = engine.session.run(TFBackend.get_parameter_variable(dict(model.named_parameters())["lin.weight"]))
    finally:
        batch_dim.dyn_size_ext = prev_batch_dyn_size_ext
        rf.select_backend_torch()

    # the embedding (blacklisted type) and every bias are excluded; a plain weight is not
    assert "emb.weight" in no_wd, no_wd
    assert "lin.bias" in no_wd, no_wd
    assert "lin.weight" in all_names and "lin.weight" not in no_wd, no_wd
    # and that is exactly the set the engine decays
    assert decayed == all_names - no_wd, (decayed, all_names - no_wd)
    assert not numpy.allclose(before, after), "the decayed parameter did not move at all"


def test_ctc_loss_target_longer_than_input():
    # Real data contains sequences whose targets do not fit the (subsampled) input.
    # PyTorch's ctc_loss is called with zero_infinity=True and contributes 0 for them;
    # TF's classic kernel raises "Not enough time for target transition sequence"
    # unless told to ignore them.
    # This is what killed the production run after 595 steps,
    # so pin the behaviour on both sides.
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    targets_dim = Dim(Tensor("targets", [batch_dim], dtype="int32"))
    wb_dim = Dim(6, name="wb")  # blank is the LAST class -> the classic TF kernel
    vocab_dim = Dim(5, name="vocab")
    extern_data = TensorDict(
        {
            "logits": Tensor("logits", [batch_dim, time_dim, wb_dim], dtype="float32", feature_dim=wb_dim),
            "targets": Tensor("targets", [batch_dim, targets_dim], dtype="int32", sparse_dim=vocab_dim),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, model: rf.Module, extern_data: TensorDict):
        loss = rf.ctc_loss(
            logits=extern_data["logits"],
            targets=extern_data["targets"],
            input_spatial_dim=time_dim,
            targets_spatial_dim=targets_dim,
            blank_index=wb_dim.dimension - 1,
        )
        loss.mark_as_default_output(shape=(batch_dim,))
        # marking the total as "loss" makes the harness compare the INPUT GRADIENTS too, which is
        # where zeroing an inf could still go wrong (a 0 * inf in the backward would give nan)
        rf.reduce_sum(loss, axis=loss.dims).mark_as_output("loss", shape=())

    # targets (up to 6) can exceed the input length (down to 2), i.e. the degenerate case
    run_model(
        extern_data,
        lambda *, epoch, step: rf.Module(),
        _forward_step,
        dyn_dim_max_sizes={time_dim: 8, targets_dim: 6},
        dyn_dim_min_sizes={time_dim: 2, targets_dim: 4},
        tf_low_level=True,
    )


def _full_model_setup():
    """
    :return: extern_data, get_model, forward_step, dims.
        The model of tests/test_rf_packed.py::test_full_model_packed_traced_program_replay, padded.
        forward_step marks the encoder output "enc", "logits", "ctc" and the total per-seq "loss".
    """
    from returnn.frontend.encoder.conformer import (
        ConformerEncoder,
        ConformerEncoderLayer,
        ConformerConvSubsample,
        ConformerPositionwiseFeedForward,
    )
    from returnn.frontend.decoder.transformer import TransformerDecoder, FeedForwardGated

    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    target_time_dim = Dim(Tensor("target_time", [batch_dim], dtype="int32"))
    in_dim = Dim(8, name="feat")
    vocab_dim = Dim(11, name="vocab")
    wb_vocab_dim = Dim(12, name="vocab_wb")
    enc_dim = Dim(32, name="enc")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
            "classes": Tensor("classes", [batch_dim, target_time_dim], dtype="int32", sparse_dim=vocab_dim),
        }
    )

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.encoder = ConformerEncoder(
                in_dim,
                enc_dim,
                ff_dim=Dim(24, name="enc-ff"),
                input_layer=ConformerConvSubsample(
                    in_dim,
                    out_dims=[Dim(4, name="conv1"), Dim(4, name="conv2"), Dim(4, name="conv3")],
                    filter_sizes=[(3, 3), (3, 3), (3, 3)],
                    pool_sizes=[(1, 2)],
                    strides=[(1, 1), (3, 1), (2, 1)],
                ),
                encoder_layer=rf.build_dict(
                    ConformerEncoderLayer,
                    ff=rf.build_dict(
                        ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                    ),
                    num_heads=2,
                    conv_norm_opts={"use_mask": True},
                ),
                num_layers=2,
                dropout=0.0,
                att_dropout=0.0,
            )
            self.decoder = TransformerDecoder(
                enc_dim,
                vocab_dim,
                Dim(32, name="dec"),
                num_layers=2,
                num_heads=2,
                norm=rf.build_dict(rf.RMSNorm),
                ff=rf.build_dict(FeedForwardGated),
                layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                dropout=0.0,
                att_dropout=0.0,
            )
            self.aux_logits = rf.Linear(enc_dim, wb_vocab_dim)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        targets = extern_data["classes"]
        enc_out, enc_spatial_dim = model.encoder(extern_data["data"], in_spatial_dim=time_dim)
        log_probs = rf.log_softmax(model.aux_logits(enc_out), axis=wb_vocab_dim)
        ctc = rf.ctc_loss(
            logits=log_probs,
            logits_normalized=True,
            targets=targets,
            input_spatial_dim=enc_spatial_dim,
            targets_spatial_dim=target_time_dim,
            blank_index=wb_vocab_dim.dimension - 1,
        )
        enc_state = model.decoder.transform_encoder(enc_out, axis=enc_spatial_dim)
        logits, _ = model.decoder(
            targets,
            spatial_dim=target_time_dim,
            state=model.decoder.default_initial_state(batch_dims=[batch_dim]),
            encoder=enc_state,
        )
        ce = rf.cross_entropy(estimated=logits, target=targets, axis=vocab_dim, estimated_type="logits")
        # marked so a parity failure localizes: if "enc" matches and "logits" does not,
        # the divergence is in the decoder, not the encoder
        enc_out.mark_as_output("enc", shape=(batch_dim, enc_spatial_dim, enc_dim))
        logits.mark_as_output("logits", shape=(batch_dim, target_time_dim, vocab_dim))
        ctc.mark_as_output("ctc", shape=(batch_dim,))
        (ctc + rf.reduce_sum(ce, axis=target_time_dim)).mark_as_output("loss", shape=(batch_dim,))

    dims = {"time_dim": time_dim, "target_time_dim": target_time_dim, "vocab_dim": vocab_dim}
    return extern_data, (lambda *, epoch, step: _Net()), _forward_step, dims


def test_full_model_tf_amp_bfloat16():
    # The full model under mixed precision: LayerNorm / RMSNorm, rel-pos self-att, the depthwise
    # conv and the CTC loss all go through the amp cast sites, so a wrong cast shows up here
    # as a dtype error or as a result that is nowhere near the float32 one.
    # Both graphs use the SAME parameters, so the casts are the only difference between them.
    import returnn.tf.compat as tf_compat
    from returnn.tf.frontend_low_level import TFBackend
    from returnn.tensor.utils import tensor_dict_fill_random_numpy_

    # noinspection PyProtectedMember
    from returnn.frontend import _backend

    extern_data, get_model, forward_step, dims = _full_model_setup()
    extern_data.reset_content()
    tensor_dict_fill_random_numpy_(
        extern_data,
        dyn_dim_max_sizes={dims["time_dim"]: 32, dims["target_time_dim"]: 4},
        dyn_dim_min_sizes={dims["time_dim"]: 24, dims["target_time_dim"]: 2},
    )
    extern_data_raw = extern_data.as_raw_tensor_dict(expected_value_type=numpy.ndarray)

    def _outputs(model) -> TensorDict:
        rf.init_forward_step_run_ctx(epoch=1, step=0)
        forward_step(model=model, extern_data=extern_data)
        return rf.get_run_ctx().outputs

    try:
        _backend.select_backend_tf()
        with tf_compat.v1.Graph().as_default(), tf_compat.v1.Session().as_default() as session:
            rf.set_random_seed(42)
            extern_data.assign_from_raw_tensor_dict_(extern_data_raw)
            _tensor_dict_numpy_to_tf(extern_data)
            with TFBackend.deferred_parameter_creation():
                model = get_model(epoch=1, step=0)
            TFBackend.create_parameters(model)

            fetches_f32 = {key: value.raw_tensor for key, value in _outputs(model).data.items()}
            with rf.set_amp_policy_ctx("bfloat16"):
                out_amp = _outputs(model)
            fetches_amp = {key: value.raw_tensor for key, value in out_amp.data.items()}
            assert set(fetches_f32) == set(fetches_amp) and fetches_f32
            assert rf.get_amp_policy() is None, "the policy must not leak out of the scope"

            session.run(tf_compat.v1.global_variables_initializer())
            res_f32, res_amp = session.run((fetches_f32, fetches_amp))
    finally:
        extern_data.reset_content()
        extern_data.assign_from_raw_tensor_dict_(extern_data_raw)
        rf.select_backend_torch()

    # The losses stay float32. The logits do not, and must not: they are the output of a matmul,
    # which is exactly what runs in the reduced dtype (PyTorch autocast leaves them there too).
    for key in ("loss", "ctc"):
        assert out_amp.data[key].dtype == "float32", f"{key} dtype {out_amp.data[key].dtype} under amp"
    assert out_amp.data["logits"].dtype == "bfloat16", out_amp.data["logits"].dtype
    for key in sorted(res_f32):
        # bfloat16 is an ml_dtypes extension dtype; older numpy refuses to compare it against
        # float32 at all ("do not have a common DType"), so widen before comparing
        a = numpy.asarray(res_f32[key], dtype=numpy.float32)
        b = numpy.asarray(res_amp[key], dtype=numpy.float32)
        scale = float(numpy.max(numpy.abs(a)))
        print(f"{key}: float32 vs bf16 max abs diff {numpy.max(numpy.abs(a - b)):.4e}, scale {scale:.4e}")
        assert numpy.all(numpy.isfinite(b)), f"{key} not finite under amp"
        # bf16 keeps ~3 significant digits and the error accumulates over the layers,
        # so this asserts that it is the same computation, not the same number
        numpy.testing.assert_allclose(a, b, rtol=0.2, atol=0.2 * scale + 1e-3, err_msg=key)


def test_full_model_param_grads():
    # Every PARAMETER gradient of the full model, TF vs PT.
    # (test_full_model already covers the outputs and the input gradients.)
    # The param values must be identical on both sides, so the PT run records its random draws
    # and the TF run replays them, as rf_utils does for the output comparison.
    import tensorflow as tf
    import returnn.tf.compat as tf_compat
    import returnn.torch.frontend as rft
    from returnn.tf.frontend_low_level import TFBackend
    from returnn.tensor.utils import tensor_dict_fill_random_numpy_
    from returnn.torch.data.tensor_utils import tensor_dict_numpy_to_torch_

    # noinspection PyProtectedMember
    from returnn.frontend import _backend

    extern_data, get_model, forward_step, dims = _full_model_setup()
    extern_data.reset_content()
    tensor_dict_fill_random_numpy_(
        extern_data,
        dyn_dim_max_sizes={dims["time_dim"]: 32, dims["target_time_dim"]: 4},
        dyn_dim_min_sizes={dims["time_dim"]: 24, dims["target_time_dim"]: 2},
    )
    extern_data_raw = extern_data.as_raw_tensor_dict(expected_value_type=numpy.ndarray)

    def _total_loss(model) -> Tensor:
        rf.init_forward_step_run_ctx(epoch=1, step=0)
        forward_step(model=model, extern_data=extern_data)
        loss = rf.get_run_ctx().outputs["loss"]
        return rf.reduce_sum(loss, axis=loss.dims)

    try:
        with rft.TorchBackend.random_journal_record() as journal:
            rf.select_backend_torch()
            rf.set_random_seed(42)
            extern_data.assign_from_raw_tensor_dict_(extern_data_raw)
            tensor_dict_numpy_to_torch_(extern_data)
            model = get_model(epoch=1, step=0)
            # the batch-norm running stats are not trainable and have no gradient
            params_pt = {name: p for name, p in model.named_parameters() if p.raw_tensor.requires_grad}
            _total_loss(model).raw_tensor.backward()
            missing_pt = [name for name, p in params_pt.items() if p.raw_tensor.grad is None]
            assert not missing_pt, "no PT grad for %s" % missing_pt
            grads_pt = {name: p.raw_tensor.grad.detach().numpy() for name, p in params_pt.items()}

        extern_data.reset_content()
        _backend.select_backend_tf()
        with TFBackend.random_journal_replay(journal):
            with tf_compat.v1.Graph().as_default(), tf_compat.v1.Session().as_default() as session:
                rf.set_random_seed(42)
                extern_data.assign_from_raw_tensor_dict_(extern_data_raw)
                _tensor_dict_numpy_to_tf(extern_data)
                with TFBackend.deferred_parameter_creation():
                    model = get_model(epoch=1, step=0)
                TFBackend.create_parameters(model)
                params_tf = dict(model.named_parameters())
                names = sorted(grads_pt)
                variables = [TFBackend.get_parameter_variable(params_tf[name]) for name in names]
                grads_raw = tf.gradients(_total_loss(model).raw_tensor, variables)
                missing = [n for n, g in zip(names, grads_raw) if g is None]
                assert not missing, f"no TF grad for {missing}"
                session.run(tf_compat.v1.global_variables_initializer())
                grads_tf = dict(zip(names, session.run(grads_raw)))
        assert journal.reached_end()
    finally:
        extern_data.reset_content()
        extern_data.assign_from_raw_tensor_dict_(extern_data_raw)
        rf.select_backend_torch()

    _assert_all_close("param grads", grads_tf, grads_pt, rtol={})


def _dyn_sizes_numpy(outputs: TensorDict) -> Dict[Dim, numpy.ndarray]:
    """
    :param outputs: whose dims are currently backed by raw tensors
    :return: per dynamic dim its per-sequence sizes, as numpy

    Taken while the PyTorch side is still live: the same Dim objects are re-bound to TF tensors
    afterwards, so the sizes have to be read out before that.
    """
    sizes = {}
    for value in outputs.data.values():
        for dim in value.dims:
            size = dim.dyn_size_ext
            if size is None or not size.dims or size.raw_tensor is None:
                continue  # static, or a scalar size (the batch dim)
            raw = size.raw_tensor
            sizes[dim] = numpy.asarray(raw.detach().cpu().numpy() if hasattr(raw, "detach") else raw)
    return sizes


def _mask_padded(template: Tensor, raw: numpy.ndarray, sizes: Dict[Dim, numpy.ndarray]) -> numpy.ndarray:
    """
    :param template: gives the dims of each axis
    :param raw:
    :param sizes: as :func:`_dyn_sizes_numpy` returns
    :return: raw with the padded positions zeroed

    Padded positions hold whatever the ops left there. Nothing constrains them to agree between
    two backends, and they are not part of the model's output, so comparing them compares noise.
    Here that is not a detail: with target lengths 2..4 in a [B, 4, V] logits tensor,
    up to half the entries are padding.
    """
    out = numpy.asarray(raw)
    if batch_dim not in template.dims:
        return out
    batch_axis = template.dims.index(batch_dim)
    for axis, dim in enumerate(template.dims):
        lens = sizes.get(dim)
        if lens is None:
            continue
        idx_shape = [1] * out.ndim
        idx_shape[axis] = -1
        len_shape = [1] * out.ndim
        len_shape[batch_axis] = -1
        mask = numpy.arange(out.shape[axis]).reshape(idx_shape) < lens.reshape(len_shape)
        out = numpy.where(mask, out, numpy.zeros((), dtype=out.dtype))
    return out


def _assert_all_close(
    name: str, values_tf: Dict[str, numpy.ndarray], values_pt: Dict[str, numpy.ndarray], *, rtol: Dict[str, float]
):
    """
    :param name: what is compared, for the message
    :param values_tf:
    :param values_pt: the reference
    :param rtol: per key; a key not listed uses _DEFAULT_RTOL

    Compares EVERY key and reports all of them before failing.
    Comparing only up to the first mismatch (what a plain loop of ``assert_allclose`` does)
    hides which quantities are actually off -- and since the keys are compared in sorted order,
    which one is reported is alphabetical rather than informative.

    The tolerance is relative to each tensor's own magnitude rather than per element.
    These are sums accumulated over a whole model, so the meaningful scale is the tensor,
    not the individual entry: with a flat ``atol`` a small entry inside a tensor whose
    entries reach 7 is held to a precision float32 cannot deliver.
    """
    assert set(values_tf) == set(values_pt), f"{name}: keys {sorted(values_tf)} vs {sorted(values_pt)}"
    lines, failed = [], []
    for key in sorted(values_pt):
        a, b = values_tf[key], values_pt[key]
        diff = float(numpy.max(numpy.abs(a - b)))
        scale = float(numpy.max(numpy.abs(b)))
        key_rtol = rtol.get(key, _DEFAULT_RTOL)
        tol = key_rtol * max(scale, 1.0)
        ok = diff <= tol
        lines.append(
            f"  {'ok ' if ok else 'BAD'} {key:55s} max|diff| {diff:.3e}  scale {scale:.3e}"
            f"  diff/scale {diff / max(scale, 1e-30):.3e}  (rtol {key_rtol:.0e})"
        )
        if not ok:
            failed.append(key)
    print(f"{name}: TF vs PT")
    print("\n".join(lines))
    # the full table also goes into the assert message: the pytest summary shows only that,
    # and a failure report without the magnitudes is useless
    assert not failed, f"{name}: {failed} differ beyond tolerance\n" + "\n".join(lines)


# float32 through a model this deep, compared across two frameworks and two BLAS/kernel stacks.
# Measured headroom: the worst diff/scale is ~2.5e-6 locally and ~2.2e-6 on CI (TF 2.10 / torch 2.0),
# so 1e-5 keeps roughly 4x margin without being able to absorb a real defect.
_DEFAULT_RTOL = 1e-5


def _tensor_dict_numpy_to_tf(x: TensorDict):
    """tf.constant() on all values, including their dims"""
    import tensorflow as tf

    def _convert(v: Tensor):
        if isinstance(v.raw_tensor, numpy.ndarray):
            v.raw_tensor = tf.constant(v.raw_tensor)
        for dim in v.dims:
            dim.transform_tensors(_convert)

    for v_ in x.data.values():
        _convert(v_)


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
