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
            assert set(name for name, _ in tf.train.list_variables(ckpt)) == {"out.weight", "out.bias"}

            # a fresh engine continues from that checkpoint instead of starting over
            engine2 = Engine(config=config)
            engine2.init_train_from_config(config=config, train_data=train_data, dev_data=dev_data)
            assert engine2.epoch == 4, "did not continue from the existing checkpoint"
    finally:
        batch_dim.dyn_size_ext = prev_batch_dyn_size_ext
        rf.select_backend_torch()
        shutil.rmtree(model_dir, ignore_errors=True)


def _full_model_setup():
    """
    :return: extern_data, get_model, forward_step, dims.
        The model of tests/test_rf_packed.py::test_full_model_packed_traced_program_replay, padded.
        forward_step marks "logits", "ctc" and the total per-seq "loss".
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
        logits.mark_as_output("logits", shape=(batch_dim, target_time_dim, vocab_dim))
        ctc.mark_as_output("ctc", shape=(batch_dim,))
        (ctc + rf.reduce_sum(ce, axis=target_time_dim)).mark_as_output("loss", shape=(batch_dim,))

    dims = {"time_dim": time_dim, "target_time_dim": target_time_dim, "vocab_dim": vocab_dim}
    return extern_data, (lambda *, epoch, step: _Net()), _forward_step, dims


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

    assert set(grads_pt) == set(grads_tf)
    for name in sorted(grads_pt):
        print("comparing grad %r %s" % (name, grads_pt[name].shape))
        numpy.testing.assert_allclose(
            grads_tf[name], grads_pt[name], rtol=1e-4, atol=1e-5, err_msg="grad %s differs" % name
        )


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
