# start: nosetests $this_file --nologcapture

from __future__ import annotations

import _setup_test_env  # noqa
import tensorflow as tf
import sys
import os
from nose.tools import assert_equal, assert_not_equal, assert_is_instance
from numpy.testing.utils import assert_almost_equal, assert_allclose
import unittest
import numpy.testing
from pprint import pprint
from returnn.util import better_exchook

from returnn.config import Config
from returnn.tf.network import *
from returnn.tf.layers.rec import *
from returnn.tf.util.basic import is_gpu_available
from returnn.tf.util.data import Data, Dim, SpatialDim, FeatureDim


@unittest.skip("for testing only...")
def test_a_crash_seg_fault():
    """
    Just testing our signal handlers...
    """
    import ctypes

    invalid_ptr = ctypes.cast(1, ctypes.POINTER(ctypes.c_int))
    # Access it. This will crash!
    print(invalid_ptr.contents)


@unittest.skip("for testing only...")
def test_a_crash_abort():
    """
    Just testing our signal handlers...
    """
    import ctypes

    # Warning! Will crash!
    ctypes.pythonapi.abort()


@contextlib.contextmanager
def make_scope():
    """
    :rtype: tf.compat.v1.Session
    """
    with tf.Graph().as_default() as graph:
        with tf_compat.v1.Session(graph=graph) as session:
            yield session


class SimpleCumSumCell(BaseRNNCell):
    """
    Implements cumsum.
    """

    def __init__(self, num_units):
        """
        :param int num_units:
        """
        super(SimpleCumSumCell, self).__init__()
        self._num_units = num_units

    @property
    def output_size(self):
        """
        :rtype: int
        """
        return self._num_units

    @property
    def state_size(self):
        """
        :rtype: int
        """
        return self._num_units

    # noinspection PyMethodOverriding
    def call(self, inputs, state):
        """
        :param tf.Tensor inputs:
        :param tf.Tensor state:
        :return: (output, state)
        :rtype: (tf.Tensor, tf.Tensor)
        """
        inputs.set_shape((None, self._num_units))
        state.set_shape((None, self._num_units))
        current_state = inputs + state
        return current_state, current_state


# Currently there is no good API to register an external rec cell class...
RecLayer._create_rnn_cells_dict()
RecLayer._rnn_cells_dict["cumsum"] = SimpleCumSumCell


def _check_train_simple_network(network, num_steps=10):
    num_inputs = 4
    num_outputs = 3

    config = Config()
    config.update(
        {
            "num_inputs": num_inputs,
            "num_outputs": {"data": [num_inputs, 2], "classes": [num_outputs, 2]},  # dense output
            "network": network,
            "optimizer": {"class": "adam"},
            "learning_rate": 0.1,
            "debug_add_check_numerics_ops": True,
        }
    )

    random = numpy.random.RandomState(seed=1)
    limit = 1.0

    def make_feed_dict(step, seq_len=10):
        d = {
            network.extern_data.get_batch_info().dim: 1,
            network.extern_data.data["data"].placeholder: random.uniform(-limit, limit, (1, seq_len, num_inputs)),
            network.extern_data.data["data"].size_placeholder[0]: numpy.array([seq_len]),
            network.extern_data.data["classes"].placeholder: random.uniform(-limit, limit, (1, seq_len, num_outputs)),
            network.extern_data.data["classes"].size_placeholder[0]: numpy.array([seq_len]),
        }
        if isinstance(network.epoch_step, tf.Tensor):
            d[network.epoch_step] = step
        return d

    with make_scope() as session:
        print("Create network...")
        tf_compat.v1.set_random_seed(42)
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_dict["network"])
        network.print_network_info()
        network.initialize_params(session=session)

        from returnn.tf.updater import Updater

        updater = Updater(config=config, network=network)
        updater.set_learning_rate(config.float("learning_rate", 1.0), session=session)
        updater.set_trainable_vars(network.get_trainable_params())
        updater.init_optimizer_vars(session=session)

        loss = None
        for step in range(num_steps):
            loss, _, _ = session.run(
                [network.get_total_loss(), updater.get_optim_op(), network.get_post_control_dependencies()],
                feed_dict=make_feed_dict(step=step),
            )
            print("step %i, loss: %f" % (step, loss))
    return loss


def test_rec_nativelstm():
    _check_train_simple_network({"output": {"class": "rec", "unit": "nativelstm", "loss": "mse", "from": "data:data"}})


def test_rec_nativelstm2():
    _check_train_simple_network({"output": {"class": "rec", "unit": "nativelstm2", "loss": "mse", "from": "data:data"}})


def test_rec_zoneout_lstm():
    from returnn.tf.util.data import batch_dim, FeatureDim

    out_dim = FeatureDim("lstm-out", 7)
    _check_train_simple_network(
        {
            "rec": {
                "class": "rec",
                "from": "data",
                "optimize_move_layers_out": False,
                "unit": {
                    "lstm": {
                        "class": "rec",
                        "unit": "ZoneoutLSTM",
                        "unit_opts": {"zoneout_factor_cell": 0.1, "zoneout_factor_output": 0.1},
                        "from": "data:source",
                        "out_dim": out_dim,
                        "out_shape": {batch_dim, out_dim},
                    },
                    "output": {"class": "copy", "from": "lstm"},
                },
            },
            "output": {"class": "linear", "from": "rec", "loss": "mse"},
        }
    )


def test_rec_rhn():
    _check_train_simple_network(
        {"output": {"class": "rec", "unit": "rhn", "unit_opts": {"dropout": 0.1}, "loss": "mse", "from": "data:data"}}
    )


def test_rec_rhn_nan():
    _check_train_simple_network(
        {
            "output": {
                "class": "rec",
                "unit": "rhn",
                "unit_opts": {"dropout": 0.9, "dropout_seed": 1},
                "loss": "mse",
                "from": "data:data",
            }
        }
    )


def test_rhn_nan():
    """
    Behaves just like :func:`test_rec_rhn_nan`.
    """
    random = numpy.random.RandomState(seed=1)
    num_inputs = 4
    num_outputs = 3
    seq_len = 10
    limit = 1.0
    loop_variants = ["RecLayer", "dynamic_rnn", "while_loop", "unroll", "unroll_simple"]
    loop_variant = "unroll_simple"

    with make_scope() as session:
        print("create graph")
        tf_compat.v1.set_random_seed(42)
        src_placeholder = tf_compat.v1.placeholder(tf.float32, (None, seq_len, num_inputs), name="src_placeholder")
        tgt_placeholder = tf_compat.v1.placeholder(tf.float32, (None, seq_len, num_outputs), name="tgt_placeholder")
        batch_size = tf.shape(src_placeholder)[0]

        def make_feed_dict():
            return {
                src_placeholder: random.uniform(-limit, limit, (1, seq_len, num_inputs)),
                tgt_placeholder: random.uniform(-limit, limit, (1, seq_len, num_outputs)),
            }

        from returnn.tf.util.basic import xavier_initializer

        default_var_initializer = xavier_initializer(seed=13)
        with tf_compat.v1.variable_scope(
            tf_compat.v1.get_variable_scope(), initializer=default_var_initializer
        ) as scope:
            assert loop_variant in loop_variants
            if loop_variant in ["RecLayer", "dynamic_rnn"]:
                if loop_variant == "RecLayer":
                    # Here I get nan.
                    net = TFNetwork(config=Config(), extern_data=ExternData(), train_flag=True)
                    with net.register_network_scope():
                        from returnn.tf.layers.base import InternalLayer

                        src_layer = InternalLayer(
                            name="src",
                            network=net,
                            output=Data(
                                "src",
                                shape=(None, num_inputs),
                                placeholder=src_placeholder,
                                size_placeholder={0: tf.convert_to_tensor([seq_len])},
                            ),
                        )
                        with tf.name_scope("output"):
                            rec_layer = RecLayer(
                                name="output",
                                network=net,
                                output=Data("out", shape=(None, num_outputs)),
                                sources=[src_layer],
                                unit="rhn",
                                unit_opts={"dropout": 0.9, "dropout_seed": 1, "batch_size": batch_size},
                            )
                    y = rec_layer.output.placeholder
                    y = tf.transpose(y, [1, 0, 2])
                elif loop_variant == "dynamic_rnn":
                    rhn = RHNCell(
                        num_units=num_outputs, is_training=True, dropout=0.9, dropout_seed=1, batch_size=batch_size
                    )
                    # Will get y in (time,batch,ydim).
                    from tensorflow.python.ops import rnn

                    x = tf.transpose(src_placeholder, [1, 0, 2])
                    x = rhn.get_input_transformed(x)
                    y, final_state = rnn.dynamic_rnn(
                        cell=rhn, inputs=x, time_major=True, sequence_length=[seq_len], dtype=tf.float32
                    )
                    y = tf.transpose(y, [1, 0, 2])
                loss = tf.reduce_sum(tf.reduce_mean(tf_compat.v1.squared_difference(tgt_placeholder, y), axis=-1))
            else:
                rhn = RHNCell(
                    num_units=num_outputs, is_training=True, dropout=0.9, dropout_seed=1, batch_size=batch_size
                )
                state = rhn.zero_state(batch_size, tf.float32)
                x = tf.transpose(src_placeholder, [1, 0, 2])
                x = rhn.get_input_transformed(x)
                input_ta = tf.TensorArray(tf.float32, size=seq_len, element_shape=(None, num_outputs * 2))
                input_ta = input_ta.unstack(x)
                target_ta = tf.TensorArray(tf.float32, size=seq_len, element_shape=(None, num_outputs))
                target_ta = target_ta.unstack(tf.transpose(tgt_placeholder, [1, 0, 2]))
                loss_ta = tf.TensorArray(tf.float32, size=seq_len, element_shape=(None,))

                def loop_iter(i, state, loss_ta):
                    output, state = rhn(inputs=input_ta.read(i), state=state)
                    frame_loss = tf.reduce_mean(tf_compat.v1.squared_difference(target_ta.read(i), output), axis=1)
                    assert frame_loss.get_shape().ndims == 1  # (batch,)
                    # frame_loss = tf.Print(frame_loss, ["frame", i, "loss", tf.reduce_sum(frame_loss)])
                    loss_ta = loss_ta.write(i, frame_loss)
                    return i + 1, state, loss_ta

                if loop_variant == "while_loop":
                    i, state, loss_ta = tf.while_loop(
                        lambda i, *args: tf.less(i, seq_len), loop_iter, (0, state, loss_ta)
                    )
                    loss = tf.reduce_sum(loss_ta.stack())
                elif loop_variant == "unroll":
                    # Unroll the loop here.
                    i = 0
                    while i < seq_len:
                        i, state, loss_ta = loop_iter(i, state, loss_ta)
                    loss = tf.reduce_sum(loss_ta.stack())
                elif loop_variant == "unroll_simple":
                    loss = 0.0
                    for i in range(seq_len):
                        output, state = rhn(inputs=x[i], state=state)
                        frame_loss = tf.reduce_mean(
                            tf_compat.v1.squared_difference(tgt_placeholder[:, i], output), axis=1
                        )
                        # frame_loss = tf.Print(frame_loss, ['frame', i, 'loss', frame_loss, 'SE of', tgt_placeholder[:, i], output])
                        assert frame_loss.get_shape().ndims == 1  # (batch,)
                        loss += tf.reduce_sum(frame_loss)
                else:
                    assert False, "unexpected loop variant %r" % loop_variant
        optimizer = tf_compat.v1.train.AdamOptimizer(learning_rate=0.1, epsilon=1e-16, use_locking=False)
        minimize_op = optimizer.minimize(loss)
        from returnn.tf.util.basic import add_check_numerics_ops

        # check_op = add_check_numerics_ops()
        check_op = tf.no_op()

        print("variables:")
        train_vars = tf_compat.v1.trainable_variables() + tf_compat.v1.get_collection(
            tf_compat.v1.GraphKeys.TRAINABLE_RESOURCE_VARIABLES
        )
        print(train_vars)
        var_norms = [tf.nn.l2_loss(v) for v in train_vars]
        print("init vars")
        session.run(tf_compat.v1.global_variables_initializer())
        print("graph size:", session.graph_def.ByteSize())
        print("train")
        for s in range(10):
            loss_val, _, _ = session.run([loss, minimize_op, check_op], feed_dict=make_feed_dict())
            print("step %i, loss: %f" % (s, loss_val))
            # var_norm_vals = session.run(var_norms)
            # print('var norms:')
            # for (v, x) in zip(train_vars, var_norm_vals):
            #  print(' ', v, ':', x)


def test_state_keep_over_epoch():
    random = numpy.random.RandomState(seed=1)
    num_inputs = 4
    num_outputs = 3
    batch_size = 5
    seq_len = 10
    limit = 1.0
    src_seq = random.uniform(-limit, limit, (batch_size, seq_len, num_inputs))
    net_dict = {
        "output": {
            "class": "rec",
            "unit": "rhn",
            "initial_state": "keep_over_epoch",
            "n_out": num_outputs,
            "from": "data:data",
        }
    }

    with make_scope() as session:
        print("create graph")
        tf_compat.v1.set_random_seed(42)
        net = TFNetwork(extern_data=ExternData({"data": {"shape": (None, num_inputs)}}))
        net.construct_from_dict(net_dict)
        net.initialize_params(session)
        print("vars:")
        print(tf_compat.v1.global_variables())
        src = net.extern_data.data["data"].placeholder
        src_seq_len = net.extern_data.data["data"].size_placeholder[0]
        out = net.get_default_output_layer().output.get_placeholder_as_batch_major()
        out_val = numpy.zeros((batch_size, 0, num_outputs))
        print("run on parts")
        part_seq_len = 2
        for step, t in enumerate(range(0, seq_len, part_seq_len)):
            out_val_part, _ = session.run(
                [out, net.get_post_control_dependencies()],
                feed_dict={
                    net.epoch_step: step,
                    net.extern_data.get_batch_info().dim: batch_size,
                    src_seq_len: [part_seq_len] * batch_size,
                    src: src_seq[:, t : t + part_seq_len],
                },
            )
            assert out_val_part.shape == (batch_size, part_seq_len, num_outputs)
            out_val = numpy.concatenate([out_val, out_val_part], axis=1)
        assert out_val.shape == (batch_size, seq_len, num_outputs)
        print("run full")
        out_val_full, _ = session.run(
            [out, net.get_post_control_dependencies()],
            feed_dict={
                net.epoch_step: 0,
                net.extern_data.get_batch_info().dim: batch_size,
                src_seq_len: [seq_len] * batch_size,
                src: src_seq,
            },
        )
        assert out_val_full.shape == out_val.shape
        assert_almost_equal(out_val, out_val_full)
    print("ok!")


def test_lstm_initial_state_zero():
    _check_train_simple_network(
        {"output": {"class": "rec", "unit": "lstm", "loss": "mse", "initial_state": "zeros", "from": "data:data"}}
    )


def test_lstm_initial_state_var():
    _check_train_simple_network(
        {"output": {"class": "rec", "unit": "lstm", "loss": "mse", "initial_state": "var", "from": "data:data"}}
    )


def test_nativelstm2_initial_state_var():
    _check_train_simple_network(
        {"output": {"class": "rec", "unit": "nativelstm2", "loss": "mse", "initial_state": "var", "from": "data:data"}}
    )


def test_nativelstm2_initial_state_keep_epoch():
    _check_train_simple_network(
        {
            "output": {
                "class": "rec",
                "unit": "nativelstm2",
                "loss": "mse",
                "initial_state": "keep_over_epoch",
                "from": "data:data",
            }
        }
    )


def test_slow_TensorArray():
    """
    Seems to be some strange hang, probably related to tf.TensorArray.
    https://github.com/tensorflow/tensorflow/issues/18117

    My output with TF 1.5.0:
      ...
      create graph
      variables:
      ...
      init vars
      graph size: 222234
      train
      step 0, loss: 5.506713, time: 10.675434
      step 1, loss: 7.865020, time: 0.003913
      step 2, loss: 5.450877, time: 0.003354
      step 3, loss: 3.361173, time: 0.003227
      step 4, loss: 4.493120, time: 0.003563
      step 5, loss: 5.137649, time: 0.003203
      step 6, loss: 3.610677, time: 0.003376
      step 7, loss: 3.657249, time: 0.003544
      step 8, loss: 4.405594, time: 0.003454
      step 9, loss: 4.380188, time: 0.003491

    My output with TF 1.7.0:
      ...
      init vars
      graph size: 225096
      train
      step 0, loss: 4.614282, time: 0.329974
      step 1, loss: 7.103771, time: 0.003420
      step 2, loss: 4.263576, time: 0.003305
      step 3, loss: 2.140168, time: 0.003355
      step 4, loss: 3.948706, time: 0.003271
      step 5, loss: 3.063313, time: 0.003162
      step 6, loss: 4.229179, time: 0.003354
      step 7, loss: 4.908344, time: 0.003289
      step 8, loss: 4.345730, time: 0.003188
    """
    import time

    random = numpy.random.RandomState(seed=1)
    num_inputs = 4
    num_outputs = 3
    seq_len = 10
    limit = 1.0

    def linear(x, output_dim):
        input_dim = x.get_shape().dims[-1].value
        assert input_dim is not None
        with tf_compat.v1.variable_scope("linear", reuse=tf_compat.v1.AUTO_REUSE):
            weights = tf_compat.v1.get_variable("W", shape=(input_dim, output_dim))
            bias = tf_compat.v1.get_variable("b", shape=(output_dim,))
        assert x.get_shape().ndims == 2  # (batch,input_dim)
        return tf.matmul(x, weights) + bias

    with make_scope() as session:
        print("create graph")
        src_placeholder = tf_compat.v1.placeholder(tf.float32, (None, seq_len, num_inputs), name="src_placeholder")
        tgt_placeholder = tf_compat.v1.placeholder(tf.float32, (None, seq_len, num_outputs), name="tgt_placeholder")
        batch_size = tf.shape(src_placeholder)[0]

        def make_feed_dict():
            return {
                src_placeholder: random.uniform(-limit, limit, (1, seq_len, num_inputs)),
                tgt_placeholder: random.uniform(-limit, limit, (1, seq_len, num_outputs)),
            }

        state = tf.zeros((batch_size, num_outputs))
        loss_ta = tf.TensorArray(tf.float32, size=seq_len, element_shape=(None,))
        # Unroll the loop here.
        for f in range(seq_len):
            inputs = src_placeholder[:, f]
            x = tf.concat([inputs, state], axis=-1)
            with tf_compat.v1.variable_scope("h"):
                h = tf.tanh(linear(x, num_outputs))
            with tf_compat.v1.variable_scope("t"):
                t = tf.sigmoid(linear(x, num_outputs))
            state += t * (h - state)
            frame_loss = tf.reduce_mean(tf_compat.v1.squared_difference(tgt_placeholder[:, f], state), axis=1)
            assert frame_loss.get_shape().ndims == 1  # (batch,)
            loss_ta = loss_ta.write(f, frame_loss)
        loss = tf.reduce_sum(loss_ta.stack())
        optimizer = tf_compat.v1.train.AdamOptimizer(learning_rate=0.1, epsilon=1e-16, use_locking=False)
        minimize_op = optimizer.minimize(loss)

        print("variables:")
        train_vars = tf_compat.v1.trainable_variables() + tf_compat.v1.get_collection(
            tf_compat.v1.GraphKeys.TRAINABLE_RESOURCE_VARIABLES
        )
        print(train_vars)
        print("init vars")
        session.run(tf_compat.v1.global_variables_initializer())
        print("graph size:", session.graph_def.ByteSize())
        print("train")
        for s in range(10):
            start_time = time.time()
            loss_val, _ = session.run([loss, minimize_op], feed_dict=make_feed_dict())
            print("step %i, loss: %f, time: %f" % (s, loss_val, time.time() - start_time))


def test_deterministic_TensorArray():
    num_inputs = 4
    num_outputs = 3
    seq_len = 10
    limit = 1.0

    first_run_loss = None
    for r in range(3):
        print(">>> run %i" % r)
        random = numpy.random.RandomState(seed=1)

        with make_scope() as session:
            tf_compat.v1.set_random_seed(42)
            print("create graph")
            src_placeholder = tf_compat.v1.placeholder(tf.float32, (None, seq_len, num_inputs), name="src_placeholder")
            tgt_placeholder = tf_compat.v1.placeholder(tf.float32, (None, seq_len, num_outputs), name="tgt_placeholder")
            batch_size = tf.shape(src_placeholder)[0]

            def make_feed_dict():
                return {
                    src_placeholder: random.uniform(-limit, limit, (1, seq_len, num_inputs)),
                    tgt_placeholder: random.uniform(-limit, limit, (1, seq_len, num_outputs)),
                }

            cell = rnn_cell.BasicRNNCell(num_units=num_outputs)
            state = cell.zero_state(batch_size, tf.float32)
            loss_ta = tf.TensorArray(tf.float32, size=seq_len, element_shape=(None,))
            # Unroll the loop here.
            for i in range(seq_len):
                keep_prob = 0.9
                # uniform [keep_prob, 1.0 + keep_prob)
                random_tensor = keep_prob
                random_tensor += tf_compat.v1.random_uniform((batch_size, cell.state_size), seed=1, dtype=state.dtype)
                # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
                binary_tensor = tf.floor(random_tensor)
                noise_h = binary_tensor / keep_prob
                state *= noise_h

                output, state = cell(inputs=src_placeholder[:, i], state=state)
                frame_loss = tf.reduce_mean(tf_compat.v1.squared_difference(tgt_placeholder[:, i], output), axis=1)
                assert frame_loss.get_shape().ndims == 1  # (batch,)
                loss_ta = loss_ta.write(i, frame_loss)
            loss = tf.reduce_sum(loss_ta.stack())
            optimizer = tf_compat.v1.train.AdamOptimizer(learning_rate=0.1, epsilon=1e-16, use_locking=False)
            minimize_op = optimizer.minimize(loss)

            print("variables:")
            train_vars = tf_compat.v1.trainable_variables() + tf_compat.v1.get_collection(
                tf_compat.v1.GraphKeys.TRAINABLE_RESOURCE_VARIABLES
            )
            print(train_vars)
            print("init vars")
            session.run(tf_compat.v1.global_variables_initializer())
            print("graph size:", session.graph_def.ByteSize())
            print("train")
            loss_val = None
            for s in range(10):
                loss_val, _ = session.run([loss, minimize_op], feed_dict=make_feed_dict())
                print("step %i, loss: %f" % (s, loss_val))
            assert loss_val is not None
            if r == 0:
                first_run_loss = loss_val
            else:
                assert numpy.isclose(first_run_loss, loss_val)


def test_rec_subnet_template_exception_handling_reraise():
    # https://github.com/rwth-i6/returnn/issues/995
    class _EncoderException(Exception):
        pass

    def _check(num_exceptions):
        class _Counter:
            c = 0

        with make_scope() as session:

            def _enc_func(source, **_):
                _Counter.c += 1
                if _Counter.c <= num_exceptions:
                    print("raise exception %i" % _Counter.c)
                    raise _EncoderException("exception %i" % _Counter.c)
                return source(0)

            config = Config({"extern_data": {"data": {"dim": 7}, "classes": {"dim": 3, "sparse": True}}})
            net_dict = {
                "encoder": {"class": "eval", "from": "data", "eval": _enc_func},
                "enc_mean": {"class": "reduce", "mode": "mean", "axis": "T", "from": "encoder"},
                "output": {
                    "class": "rec",
                    "from": [],
                    "target": "classes",
                    "unit": {
                        "prev_embed": {"class": "linear", "from": "prev:output", "n_out": 7},
                        "combine": {
                            "class": "linear",
                            "from": ["prev_embed", "base:enc_mean"],
                            "activation": "relu",
                            "n_out": 7,
                        },
                        "prob": {"class": "softmax", "from": "combine", "loss": "ce", "target": "classes"},
                        "output": {
                            "class": "choice",
                            "beam_size": 4,
                            "from": "prob",
                            "target": "classes",
                            "initial_output": 0,
                        },
                        "end": {"class": "compare", "from": "output", "value": 0},
                    },
                },
            }
            network = TFNetwork(config=config, train_flag=True)
            network.construct_from_dict(net_dict)

    # No exception at all. Should just be fine. Just a sanity check.
    _check(0)

    # Raising an exception inside the base network.
    # This will happen during the RecLayer template construction,
    # and due to the template construction logic which tries to recover,
    # it would still have worked in earlier versions.
    # However, we don't want that.
    # We want that any exceptions from base layers are directly re-raised.
    try:
        _check(1)
    except _EncoderException as exc:
        print("expected exception:", exc)
        pass  # expected
    else:
        assert False, "expected exception"


def test_rec_subnet_with_choice():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    in_dim = FeatureDim("feat", 3)
    out_dim = FeatureDim("classes", 4)
    time_dim = SpatialDim("time")
    with tf_compat.v1.Session():
        config = Config()
        config.update(
            {
                "extern_data": {
                    "data": {"dim_tags": [batch_dim, time_dim, in_dim]},
                    "classes": {"dim_tags": [batch_dim, time_dim], "sparse_dim": out_dim},
                },
                "network": {
                    "output": {
                        "class": "rec",
                        "from": "data:data",
                        "target": "classes",
                        "unit": {
                            "prob": {"class": "softmax", "from": ["prev:output"], "loss": "ce", "target": "classes"},
                            "output": {
                                "class": "choice",
                                "beam_size": 4,
                                "from": ["prob"],
                                "target": "classes",
                                "initial_output": 0,
                            },
                        },
                    },
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_dict["network"])


@unittest.skipIf(not is_gpu_available(), "no gpu on this system")
def test_RecLayer_get_cudnn_params_size():
    try:
        from tensorflow.contrib.cudnn_rnn.ops.gen_cudnn_rnn_ops import cudnn_rnn_params_size
    except ImportError:  # TF 2
        from tensorflow.python.ops.gen_cudnn_rnn_ops import cudnn_rnn_params_size

    def check(
        num_units,
        input_size,
        rnn_mode="lstm",
        num_layers=1,
        direction="unidirectional",
        input_mode="linear_input",
        T=tf.float32,
        S=tf.int32,
    ):
        common_kwargs = dict(
            rnn_mode=rnn_mode,
            num_units=num_units,
            input_size=input_size,
            num_layers=num_layers,
            direction=direction,
            input_mode=input_mode,
        )
        cu_size = cudnn_rnn_params_size(T=T, S=S, **common_kwargs)[0]
        my_size = RecLayer._get_cudnn_param_size(**common_kwargs)
        assert_equal(cu_size.eval(), my_size)

    with tf_compat.v1.Session():
        check(rnn_mode="lstm", num_units=5, input_size=3)
        check(rnn_mode="lstm", num_units=5, input_size=5)
        check(rnn_mode="gru", num_units=7, input_size=5)
        check(rnn_mode="gru", num_units=7, input_size=7)
        check(rnn_mode="rnn_tanh", num_units=7, input_size=7)
        check(rnn_mode="lstm", num_units=5, input_size=3, direction="bidirectional")
        check(rnn_mode="lstm", num_units=5, input_size=3, direction="bidirectional", num_layers=2)
        check(rnn_mode="lstm", num_units=5, input_size=3, direction="bidirectional", num_layers=7)
        check(rnn_mode="lstm", num_units=5, input_size=3, direction="bidirectional", input_mode="auto_select")
        check(
            rnn_mode="lstm",
            num_units=5,
            input_size=3,
            direction="bidirectional",
            num_layers=7,
            input_mode="auto_select",
        )
        check(
            rnn_mode="lstm",
            num_units=5,
            input_size=3,
            direction="unidirectional",
            num_layers=7,
            input_mode="auto_select",
        )
        check(rnn_mode="lstm", num_units=5, input_size=3, direction="bidirectional", input_mode="skip_input")
        check(
            rnn_mode="lstm", num_units=5, input_size=3, direction="bidirectional", num_layers=7, input_mode="skip_input"
        )


@unittest.skipIf(not is_gpu_available(), "no gpu on this system")
def test_cudnn_save_restore():
    from pprint import pprint
    import tempfile, shutil, os
    from tensorflow.python.training.saver import BaseSaverBuilder

    model_tmp_dir = tempfile.mkdtemp("tmp-checkpoint")
    model_filename = model_tmp_dir + "/model"
    try:
        num_inputs = 4
        input_data = numpy.array(
            [[[1, -0.2, 0.3, -4], [2, -0.6, 0.7, -1.8], [1, 0.3, -0.1, -0.8], [0.1, -0.2, 0.2, 0.8]]], dtype="float32"
        )
        seq_lens = numpy.array([4], dtype="int32")
        assert input_data.shape == (1, seq_lens[0], num_inputs)
        num_outputs = 3

        print("Storing network with cuDNN.")
        tf_compat.v1.reset_default_graph()
        with tf_compat.v1.Session() as session:
            config1 = Config()
            config1.update(
                {
                    "num_outputs": num_outputs,
                    "num_inputs": num_inputs,
                    "network": {
                        "layer1": {"class": "rec", "n_out": 6, "unit": "CudnnLSTM", "from": "data:data"},
                        "layer2": {"class": "rec", "n_out": 6, "unit": "CudnnLSTM", "from": ["layer1"]},
                        "output": {"class": "linear", "activation": None, "n_out": num_outputs, "from": ["layer2"]},
                    },
                }
            )
            network1 = TFNetwork(config=config1, train_flag=True)
            network1.construct_from_dict(config1.typed_dict["network"])
            network1.initialize_params(session=session)
            params = {}  # type: dict[str,dict[str,numpy.ndarray]]  # layer -> param -> numpy.ndarray
            for layer_name, layer1 in sorted(network1.layers.items()):
                print("layer: %r" % layer_name)
                assert isinstance(layer1, LayerBase)
                params[layer_name] = {}
                for param_name, param1 in sorted(layer1.params.items()):
                    print("  param %r: %r" % (param_name, param1))
                    params[layer_name][param_name] = param1.eval(session)
                    if param1 in layer1.saveable_param_replace:
                        saveable_object = layer1.saveable_param_replace[param1]
                        print("    saveable object: %r" % saveable_object)
                        assert isinstance(saveable_object, BaseSaverBuilder.SaveableObject)
                        print("      op: %r" % saveable_object.op)
                        print("      name: %r" % saveable_object.name)
                        for spec in saveable_object.specs:
                            print("      spec: %r" % spec)
                            assert isinstance(spec, BaseSaverBuilder.SaveSpec)
                            print("        name: %r" % spec.name)
                            print("        tensor: %r" % spec.tensor)
                            print("        tensor shape: %r" % (session.run(spec.tensor).shape,))
            output_data1 = session.run(
                network1.get_default_output_layer().output.placeholder,
                feed_dict={
                    network1.extern_data.get_batch_info().dim: len(seq_lens),
                    network1.extern_data.data["data"].placeholder: input_data,
                    network1.extern_data.data["data"].size_placeholder[0]: seq_lens,
                },
            )
            assert_equal(output_data1.shape, (seq_lens[0], 1, num_outputs))  # (time, batch, dim)
            print("Saveable params:")
            pprint(network1.get_saveable_params_list())
            network1.save_params_to_file(filename=model_filename, session=session)
        print()

        # First test if we can load the same network as-is. This will involve the RNNParamsSaveable.
        print("Testing restore of same network with cuDNN.")
        tf_compat.v1.reset_default_graph()
        with tf_compat.v1.Session() as session:
            network1a = TFNetwork(config=config1, train_flag=True)
            network1a.construct_from_dict(config1.typed_dict["network"])
            print("Saveable params:")
            pprint(network1a.get_saveable_params_list())
            network1a.load_params_from_file(filename=model_filename, session=session)
            for layer_name, layer1 in sorted(network1a.layers.items()):
                print("layer: %r" % layer_name)
                for param_name, param1 in sorted(layer1.params.items()):
                    print("  param %r: %r" % (param_name, param1))
                    param1old = params[layer_name][param_name]
                    param1new = param1.eval(session)
                    assert_equal(param1old.shape, param1new.shape)
                    # Unfortunately, this doesn't seem to be the case.
                    # Also, doesn't need to be, because they have two biases, so it's not unique.
                    # assert param1old.ndim == 1
                    # for i in range(param1old.shape[0]):
                    #  assert_almost_equal(param1old[i], param1new[i])
                    # numpy.testing.assert_almost_equal(param1old, param1new)
            output_data1a = session.run(
                network1a.get_default_output_layer().output.placeholder,
                feed_dict={
                    network1a.extern_data.get_batch_info().dim: len(seq_lens),
                    network1a.extern_data.data["data"].placeholder: input_data,
                    network1a.extern_data.data["data"].size_placeholder[0]: seq_lens,
                },
            )
            numpy.testing.assert_almost_equal(output_data1, output_data1a)
        print()

        print("Testing restore of network with LSTMBlockCell.")
        tf_compat.v1.reset_default_graph()
        with tf_compat.v1.Session() as session:
            # Now, in CPU, we would automatically use LSTMBlockCell instead.
            # Check if the import of the model works correctly in load_params_from_file().
            config2 = Config()
            config2.update(
                {
                    "num_outputs": num_outputs,
                    "num_inputs": num_inputs,
                    "network": {
                        "layer1": {"class": "rec", "n_out": 6, "unit": "LSTMBlockFused", "from": "data:data"},
                        "layer2": {"class": "rec", "n_out": 6, "unit": "LSTMBlockFused", "from": ["layer1"]},
                        "output": {"class": "linear", "activation": None, "n_out": num_outputs, "from": ["layer2"]},
                    },
                }
            )
            network2 = TFNetwork(config=config2, train_flag=True)
            network2.construct_from_dict(config2.typed_dict["network"])
            print("Saveable params:")
            pprint(network2.get_saveable_params_list())
            network2.load_params_from_file(filename=model_filename, session=session)
            output_data2 = session.run(
                network2.get_default_output_layer().output.placeholder,
                feed_dict={
                    network2.extern_data.get_batch_info().dim: len(seq_lens),
                    network2.extern_data.data["data"].placeholder: input_data,
                    network2.extern_data.data["data"].size_placeholder[0]: seq_lens,
                },
            )
            # Not sure if sth is incorrect... Only decimal=2 works.
            numpy.testing.assert_almost_equal(output_data1, output_data2, decimal=2)

    except Exception:
        print("test_cudnn_save_restore failed")
        sys.excepthook(*sys.exc_info())
        raise unittest.SkipTest("cuDNN RNN broken, but not so important now...")

    finally:
        shutil.rmtree(model_tmp_dir)


@unittest.skip("broken in TF. waiting to be fixed. https://github.com/tensorflow/tensorflow/issues/9370")
@unittest.skipIf(not is_gpu_available(), "no gpu on this system")
def test_cudnn_rnn_params_to_canonical():
    # https://github.com/tensorflow/tensorflow/issues/9370
    from tensorflow.contrib.cudnn_rnn import CudnnLSTM  # noqa

    with tf_compat.v1.Session() as session:

        def check(**kwargs):
            print("kwargs:", kwargs)
            model = CudnnLSTM(**kwargs)
            params = tf.Variable(tf_compat.v1.random_uniform([model.params_size()], seed=1), validate_shape=False)
            session.run(params.initializer)
            s1 = model.params_size().eval()
            print("param size:", s1)
            # s2 = sum([wts.eval().shape[0] for wtss in model.params_to_canonical(params) for wts in wtss])
            weights, biases = model.params_to_canonical(params)
            for p in weights:
                print("weight:", p, "shape:", tf.shape(p).eval())
            for p in biases:
                print("bias:", p, "shape:", tf.shape(p).eval())
            s2 = sum([tf.reduce_prod(tf.shape(p)).eval() for p in weights + biases])
            print("summed up size:", s2)
            assert_equal(s1, s2)

        check(num_layers=1, num_units=5, input_size=3, direction="unidirectional")
        check(num_layers=1, num_units=5, input_size=3, direction="bidirectional")  # fails in TF 1.2.0
        check(num_layers=2, num_units=5, input_size=3, direction="bidirectional")


def test_RecLayer_NativeLstm_Nan():
    print("test_RecLayer_NativeLstm_Nan()")
    print("GPU available:", is_gpu_available())
    numpy.set_printoptions(precision=15)
    num_inputs = 4
    num_outputs = 3

    config = Config()
    config.update(
        {
            "num_inputs": num_inputs,
            "num_outputs": {"data": [num_inputs, 2], "classes": [num_outputs, 2]},  # dense output
            "network": {"output": {"class": "rec", "unit": "NativeLSTM", "loss": "mse", "from": "data:data"}},
            "optimizer": {"class": "adam"},
            "debug_grad_summaries": True,
            "debug_save_updater_vars": True,
            "debug_add_check_numerics_ops": True,
        }
    )

    print("Reset default graph...")
    tf_compat.v1.reset_default_graph()
    print("Create network...")
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])

    # Depending on the seed, I get nan earlier, later, or not at all.
    # limit=5.0: seed=3 -> nan in step 4094. seed=1 -> nan in step 2463.
    random = numpy.random.RandomState(seed=1)
    limit = 10.0  # The higher, the more likely you get nan.

    def make_feed_dict(seq_len=10):
        return {
            network.extern_data.get_batch_info().dim: 1,
            network.extern_data.data["data"].placeholder: random.uniform(-limit, limit, (1, seq_len, num_inputs)),
            network.extern_data.data["data"].size_placeholder[0]: numpy.array([seq_len]),
            network.extern_data.data["classes"].placeholder: random.uniform(-limit, limit, (1, seq_len, num_outputs)),
            network.extern_data.data["classes"].size_placeholder[0]: numpy.array([seq_len]),
        }

    print("Creating session...")
    with tf_compat.v1.Session() as session:
        print("Init params...")
        network.initialize_params(session=session)
        print("Test run...")
        output_data1 = session.run(network.get_default_output_layer().output.placeholder, feed_dict=make_feed_dict(5))
        assert_equal(output_data1.shape, (5, 1, num_outputs))  # (time, batch, dim)

        layer = network.layers["output"]
        loss_t = network.get_total_loss() * layer.loss.get_normalization_factor()
        weights_t = layer.params["W"]
        (weights_grad_t,) = tf.gradients(network.get_objective(), weights_t)

        def find_op_by_type(type_name):
            for op in session.graph.get_operations():
                assert isinstance(op, tf.Operation)
                if op.type == type_name:
                    return op

        lstm_grad_op = find_op_by_type("GradOfLstmGenericBase")
        assert lstm_grad_op is not None
        lstm_grad_ins_t = list(lstm_grad_op.inputs)
        lstm_grad_outs_t = list(lstm_grad_op.outputs)
        lstm_grad_func = _lstm_grad_op(session=session)
        demo_grad_t = lstm_grad_func(*_demo_lstm_grad_args())
        demo_grad2_input_placeholders = [tf_compat.v1.placeholder(v.dtype) for v in lstm_grad_ins_t]
        demo_grad2_t = lstm_grad_func(*demo_grad2_input_placeholders)[1]

        print("Create updater...")
        from returnn.tf.updater import Updater

        updater = Updater(config=config, network=network)
        updater.set_trainable_vars(network.get_trainable_params())
        updater.set_learning_rate(0.1, session=session)
        updater.init_optimizer_vars(session=session)
        optim_op = updater.get_optim_op()
        assert isinstance(updater.get_default_optimizer(), tf_compat.v1.train.AdamOptimizer)
        adam_weights_m_t = updater.get_slot(var=weights_t, name="m")
        adam_weights_v_t = updater.get_slot(var=weights_t, name="v")
        assert isinstance(adam_weights_m_t, tf.Variable)
        assert isinstance(adam_weights_v_t, tf.Variable)
        summaries_t = tf_compat.v1.summary.merge_all()

        # https://github.com/tensorflow/tensorflow/blob/03beb65cecbc1e49ea477bca7f54543134b31d53/tensorflow/core/kernels/training_ops_gpu.cu.cc
        adam_update_t = adam_weights_m_t / (tf.sqrt(adam_weights_v_t) + 1e-8)

        import tempfile

        tmp_tf_logdir = tempfile.mkdtemp("tmp-tf-log")
        print("Write TF logs to:", tmp_tf_logdir)
        writer = tf_compat.v1.summary.FileWriter(tmp_tf_logdir)
        writer.add_graph(session.graph)

        print("Training...")
        recent_info = []  # type: typing.List[typing.Dict[str]]
        for i in range(1000):  # increase this to 10k or so for further testing
            feed_dict = make_feed_dict(5)
            weights_grad, lstm_grad_ins, lstm_grad_outs = session.run(
                [weights_grad_t, lstm_grad_ins_t, lstm_grad_outs_t], feed_dict=feed_dict
            )
            try:
                if not numpy.all(numpy.isfinite(weights_grad)):
                    raise Exception("weights_grad has inf or nan.")
                loss, _opt, summaries, weights, adam_update = session.run(
                    [loss_t, optim_op, summaries_t, weights_t, adam_update_t], feed_dict=feed_dict
                )
            except Exception as exc:
                print("Exception in step %i." % i)
                print(exc)
                print("Most recent summaries:")
                summary_proto = tf_compat.v1.Summary()
                summary_proto.ParseFromString(recent_info[-1]["summaries"])
                for val in summary_proto.value:
                    # Assuming all summaries are scalars.
                    print("  %s: %r" % (val.tag, val.simple_value))
                print("Most recent weights:")
                print(recent_info[-1]["weights"])
                print("Current weights:")
                print(session.run(weights_t))
                print("Most recent Adam update:")
                print(recent_info[-1]["adam_update"])
                print("Current Adam update:")
                print(session.run(adam_update_t))
                print("Used weights grad:")
                print(weights_grad)
                print("GradOfLstmGenericBase inputs:")
                for t, v in zip(lstm_grad_ins_t, lstm_grad_ins):
                    print("%r:" % t)
                    print(repr(v))
                print("GradOfLstmGenericBase outputs:")
                for t, v in zip(lstm_grad_outs_t, lstm_grad_outs):
                    print("%r:" % t)
                    print(repr(v))
                print("Demo grad:")
                print(session.run(demo_grad_t))
                print("Demo grad2:")
                print(
                    session.run(
                        demo_grad2_t, feed_dict={k: v for (k, v) in zip(demo_grad2_input_placeholders, lstm_grad_ins)}
                    )
                )
                print("Demo grad2 via eval:")
                print(
                    session.run(
                        demo_grad2_t,
                        feed_dict={
                            k: eval(repr(v), vars(numpy))
                            for (k, v) in zip(demo_grad2_input_placeholders, lstm_grad_ins)
                        },
                    )
                )
                print("Demo grad2 via args:")
                print(
                    session.run(
                        demo_grad2_t,
                        feed_dict={k: v for (k, v) in zip(demo_grad2_input_placeholders, _demo_lstm_grad_args())},
                    )
                )
                raise Exception("Exception in step %i." % i)
            writer.add_summary(summaries, global_step=i)
            if len(recent_info) > 1000:
                recent_info.pop(0)
            recent_info.append(
                {"step": i, "loss": loss, "summaries": summaries, "weights": weights, "adam_update": adam_update}
            )
            if not numpy.isfinite(loss) or i % 100 == 0:
                print("step %i, loss: %r" % (i, loss))
            assert numpy.isfinite(loss)

    print("Done.")
    import shutil

    shutil.rmtree(tmp_tf_logdir)

    from returnn.tf.util.basic import stop_event_writer_thread

    stop_event_writer_thread(writer)


def find_op_by_type(session, type_name):
    """
    :param tf.compat.v1.Session session:
    :param str type_name:
    :rtype: tf.Operation|None
    """
    for op in session.graph.get_operations():
        assert isinstance(op, tf.Operation)
        if op.type == type_name:
            return op


def _lstm_grad_op(session, verbose=True):
    """
    :param tf.compat.v1.Session session:
    :return: grad function
    """
    lstm_grad_op = find_op_by_type(session=session, type_name="LstmGenericBase")
    assert lstm_grad_op is not None
    if verbose:
        print("op:", lstm_grad_op)

    from tensorflow.python.framework import ops

    grad_func = ops.get_gradient_function(lstm_grad_op)
    if verbose:
        print("grad_func:", grad_func)
    grad_op = grad_func.grad_op
    if verbose:
        print("grad_op:", grad_op, grad_op.__doc__)
    return grad_op


def _demo_lstm_grad_args(factor=1.0, ones_like=False):
    from numpy import array, float32

    n_time = 5
    n_batch = 1
    n_out = 3
    # <tf.Tensor 'output/rec/W_re/read:0' shape=(3, 12) dtype=float32>:
    W_re = array(
        [
            [
                -2.193344831466675,
                1.360482335090637,
                0.294201552867889,
                1.242056131362915,
                -0.18156972527504,
                -0.50642192363739,
                1.264044165611267,
                0.108740165829659,
                1.768813014030457,
                -3.442604303359985,
                -0.812745451927185,
                -0.213397994637489,
            ],
            [
                5.140193462371826,
                -2.941965818405151,
                -0.055521309375763,
                1.96869695186615,
                -1.29790472984314,
                0.034493416547775,
                -1.094233393669128,
                -0.767234861850739,
                -1.832728981971741,
                2.556174278259277,
                1.285072922706604,
                2.783454418182373,
            ],
            [
                -3.460673093795776,
                0.700069725513458,
                -1.184944987297058,
                -3.619244337081909,
                3.242199659347534,
                -0.404601752758026,
                -2.755020618438721,
                -0.827874422073364,
                1.487833738327026,
                -1.766772627830505,
                -0.019650995731354,
                -1.590330123901367,
            ],
        ],
        dtype=float32,
    )
    if ones_like:
        W_re = numpy.ones_like(W_re)
    if factor != 1:
        W_re *= factor
    assert W_re.shape == (n_out, n_out * 4)
    # <tf.Tensor 'output/rec/zeros:0' shape=(?, ?) dtype=float32>:
    cell_state = array([[0.0, 0.0, 0.0]], dtype=numpy.float32)
    assert cell_state.shape == (n_batch, n_out)
    # <tf.Tensor 'extern_data/placeholders/data/sequence_mask_time_major/index_cast_float32:0' shape=(?, ?) dtype=float32>:
    index_float = array([[1.0], [1.0], [1.0], [1.0], [1.0]], dtype=numpy.float32)
    # <tf.Tensor 'output/rec/LstmGenericBase:0' shape=(?, ?, 3) dtype=float32>:
    assert index_float.shape == (n_time, n_batch)
    in0 = array(
        [
            [[-9.368341172266703e-12, -1.167426881865996e-18, 6.303897243924439e-04]],
            [[1.045539761435066e-07, -7.615810632705688e-01, 2.735287125688046e-06]],
            [[7.604487538337708e-01, -8.968127929165348e-08, 7.615941762924194e-01]],
            [[5.488518013407884e-07, -8.968121534280726e-08, 7.616176009178162e-01]],
            [[3.996720618921200e-19, -9.847509092886231e-12, 9.616374969482422e-01]],
        ],
        dtype=float32,
    )
    if ones_like:
        in0 = numpy.ones_like(in0)
    if factor != 1:
        in0 *= factor
    assert in0.shape == (n_time, n_batch, n_out)
    # <tf.Tensor 'output/rec/LstmGenericBase:1' shape=(?, ?, 12) dtype=float32>:
    in1 = array(
        [
            [
                [
                    -9.481454683879509e-12,
                    -9.999690055847168e-01,
                    9.999994039535522e-01,
                    9.481454683879509e-12,
                    9.999690055847168e-01,
                    1.000000000000000e00,
                    7.535594544194613e-12,
                    1.300011009361175e-19,
                    1.000000000000000e00,
                    9.880700707435608e-01,
                    1.532898954536958e-18,
                    8.277241722680628e-04,
                ]
            ],
            [
                [
                    1.000000000000000e00,
                    -9.999688863754272e-01,
                    2.735287125688046e-06,
                    1.000000000000000e00,
                    7.444035166059848e-09,
                    2.734021336436854e-06,
                    1.052110642194748e-01,
                    9.999998807907104e-01,
                    1.265849758347315e-09,
                    1.372830524815072e-07,
                    1.000000000000000e00,
                    1.000000000000000e00,
                ]
            ],
            [
                [
                    9.972782731056213e-01,
                    -8.968127929165348e-08,
                    1.000000000000000e00,
                    9.972844123840332e-01,
                    2.056131756665299e-35,
                    1.000000000000000e00,
                    1.915361472288072e-22,
                    8.968407172460502e-08,
                    8.604143175716672e-09,
                    1.000000000000000e00,
                    1.000000000000000e00,
                    1.000000000000000e00,
                ]
            ],
            [
                [
                    5.488518013407884e-07,
                    -8.968121534280726e-08,
                    1.000055909156799e00,
                    5.488518013407884e-07,
                    6.375615251193742e-25,
                    1.000000000000000e00,
                    1.951400955893235e-17,
                    9.999992847442627e-01,
                    5.593653258983977e-05,
                    1.000000000000000e00,
                    1.000000000000000e00,
                    1.000000000000000e00,
                ]
            ],
            [
                [
                    9.999997615814209e-01,
                    -3.767583223179827e-07,
                    2.000055789947510e00,
                    9.999997615814209e-01,
                    2.870771140806028e-07,
                    1.000000000000000e00,
                    8.848448883325144e-12,
                    1.000000000000000e00,
                    1.000000000000000e00,
                    5.247835948298600e-19,
                    2.613746983115561e-05,
                    9.975166320800781e-01,
                ]
            ],
        ],
        dtype=float32,
    )
    if ones_like:
        in1 = numpy.ones_like(in1)
    if factor != 1:
        in1 *= factor
    assert in1.shape == (n_time, n_batch, n_out * 4)
    # <tf.Tensor 'gradients/objective/loss/output/loss_init/flatten_with_seq_len_mask/swapaxes/transpose_grad/transpose:0' shape=(?, ?, 3) dtype=float32>:
    grad_in = array(
        [
            [[0.576846659183502, -0.19706067442894, -0.684425234794617]],
            [[1.117202281951904, 0.946405112743378, -0.533451914787292]],
            [[0.822037994861603, 1.044727325439453, -1.008405923843384]],
            [[-0.755452394485474, -0.606451511383057, 0.335312634706497]],
            [[0.122484095394611, 1.015499114990234, 0.080147251486778]],
        ],
        dtype=float32,
    )
    if ones_like:
        grad_in = numpy.ones_like(grad_in)
    if factor != 1:
        grad_in *= factor
    assert grad_in.shape == (n_time, n_batch, n_out)
    zeros2 = array([[0.0, 0.0, 0.0]], dtype=numpy.float32)
    assert zeros2.shape == (n_batch, n_out)
    # Args:
    #  v_h: A `Tensor` of type `float32`.
    #  c: A `Tensor` of type `float32`.
    #  i: A `Tensor` of type `float32`.
    #  y: A `Tensor` of type `float32`.
    #  h: A `Tensor` of type `float32`.
    #  d_y: A `Tensor` of type `float32`.
    #  d_d: A `Tensor` of type `float32`.
    #  name: A name for the operation (optional).
    # Returns:
    #  A tuple of `Tensor` objects (z, out_v_h, out_c, dummy_out_1).
    #  z: A `Tensor` of type `float32`.
    #  out_v_h: A `Tensor` of type `float32`.
    #  out_c: A `Tensor` of type `float32`.
    #  dummy_out_1: A `Tensor` of type `float32`.
    return W_re, cell_state, index_float, in0, in1, grad_in, zeros2


def test_GradOfLstmGenericBase_simple_nan():
    print("test_GradOfLstmGenericBase_simple_nan()")
    print("GPU available:", is_gpu_available())
    print("Create LSTM op...")
    from returnn.tf.native_op import make_lstm_op

    op_func = make_lstm_op(compiler_opts=dict(verbose=True))
    print("op_func:", op_func)

    def dummy_call():
        n_time = 1
        n_batch = 1
        n_out = 1
        Z = tf.zeros((n_time, n_batch, n_out * 4))
        V_h = tf.zeros((n_out, n_out * 4))
        c = tf.zeros((n_batch, n_out))
        i = tf.ones((n_time, n_batch))
        return op_func(Z, V_h, c, i)

    dummy = dummy_call()
    with tf_compat.v1.Session() as session:
        print("dummy out:", session.run(list(dummy)))
        grad_op = _lstm_grad_op(session)
        args = _demo_lstm_grad_args()
        placeholders = [tf_compat.v1.placeholder(v.dtype) for v in args]
        lstm_grad_t = list(grad_op(*placeholders))
        for kwargs in [{}]:  # [{"factor": 0}, {"ones_like": True}, {"ones_like": True, "factor": -1}, {}]:
            print("Testing lstm grad args %r." % kwargs)
            args = _demo_lstm_grad_args(**kwargs)
            outs = session.run(lstm_grad_t, feed_dict=dict(zip(placeholders, args)))
            for out, descr, i in zip(outs, ["z", "out_v_h", "out_c", "dummy_out"], range(4)):
                assert isinstance(out, numpy.ndarray)
                print("(%i) %s:" % (i, descr))
                print(out)
            for out in outs:
                assert numpy.all(numpy.isfinite(out))
            print("Seems ok.")
        print("All ok!")


def test_rec_RecStepInfoLayer():
    n_batch = 1
    n_time = 3
    net_dict = {
        "output": {
            "class": "rec",
            "from": "data",
            "unit": {
                "output": {"class": "copy", "from": ":i"},
            },
        }
    }
    config = Config(
        {
            "debug_print_layer_output_template": True,
            "extern_data": {
                "data": {"sparse": True, "dim": 3},
            },
        }
    )
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)
        inp = net.extern_data.data["data"]
        out = net.get_default_output_layer().output
        assert out.time_dim_axis == 0 and out.batch_dim_axis is None and out.shape == (None,) and out.dtype == "int32"
        out_v = session.run(
            out.placeholder,
            feed_dict={
                net.extern_data.get_batch_info().dim: 1,
                inp.placeholder: [[0] * n_time],
                inp.size_placeholder[0]: [n_time],
            },
        )
        assert isinstance(out_v, numpy.ndarray)
        assert out_v.shape == (n_time,)
        assert_equal(out_v.tolist(), [0, 1, 2])


def test_rec_RecStepInfoLayer_broadcast_moved_out():
    # https://github.com/rwth-i6/returnn/issues/637
    net_dict = {
        "output": {
            "class": "rec",
            "from": "data",
            "unit": {
                "segment_starts": {
                    "class": "switch",
                    "condition": "prev:output_is_not_blank",
                    "true_from": ":i",
                    "false_from": "prev:segment_starts",
                    "initial_output": 0,
                },
                "segment_lens0": {
                    "class": "combine",
                    "kind": "sub",
                    "from": [":i", "segment_starts"],
                    "is_output_layer": True,
                },
                "output_prob": {"class": "softmax", "from": "data:source", "target": "classes", "loss": "ce"},
                "output": {
                    "class": "choice",
                    "target": "classes",
                    "beam_size": 3,
                    "from": "output_prob",
                    "input_type": "prob",
                    "initial_output": 0,
                },
                "output_is_not_blank": {
                    "class": "compare",
                    "from": "output",
                    "value": 0,
                    "kind": "not_equal",
                    "initial_output": True,
                },
            },
        }
    }
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    in_dim = FeatureDim("feat", 3)
    out_dim = FeatureDim("classes", 5)
    time_dim = SpatialDim("time")
    config = Config(
        {
            "debug_print_layer_output_template": True,
            "extern_data": {
                "data": {"dim_tags": [batch_dim, time_dim, in_dim]},
                "classes": {"dim_tags": [batch_dim, time_dim], "sparse_dim": out_dim},
            },
        }
    )
    from test_TFNetworkLayer import make_feed_dict

    with make_scope() as session:
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(net_dict)
        out = net.get_default_output_layer().output
        out_v = session.run(out.placeholder, feed_dict=make_feed_dict(net.extern_data, same_time=True))
        assert isinstance(out_v, numpy.ndarray)


def test_rec_RecLastOutputLayer():
    from returnn.tf.util.data import (
        Dim,
        batch_dim,
        single_step_dim,
        SpatialDim,
        FeatureDim,
        ImplicitDynSizeDim,
        ImplicitSparseDim,
    )

    time_dim = SpatialDim("time")
    input_dim = FeatureDim("input", 3)

    config = Config(
        dict(
            extern_data={
                "data": {
                    "dim_tags": (batch_dim, time_dim, input_dim),
                    "dtype": "float32",
                    "available_for_inference": True,
                }
            }
        )
    )

    net_dict = {
        "output": {"class": "copy", "from": "add", "out_shape": {batch_dim, input_dim}},
        "loop": {
            "class": "rec",
            "from": [],
            "unit": {
                "rec_unstack": {
                    "class": "rec_unstack",
                    "from": "base:range_in_axis",
                    "axis": time_dim,
                    "out_shape": {},
                },
                "add": {
                    "class": "combine",
                    "from": ["prev:add", "rec_unstack"],
                    "kind": "add",
                    "initial_output": "base:zeros",
                    "need_last": True,
                    "out_shape": {batch_dim, input_dim},
                },
                "output": {"class": "copy", "from": "rec_unstack", "out_shape": {}},
            },
            "axis": time_dim,
            "out_shape": {ImplicitDynSizeDim(batch_dim), time_dim},
            "name_scope": "",
        },
        "range_in_axis": {
            "class": "range_in_axis",
            "from": "data:data",
            "axis": time_dim,
            "out_shape": {ImplicitDynSizeDim(batch_dim), time_dim},
        },
        "zeros": {"class": "constant", "value": 0, "shape": [batch_dim, input_dim], "dtype": "int32"},
        "add": {
            "class": "rec_last_output",
            "rec_layer": "loop",
            "sub_layer_name": "add",
            "out_shape": {batch_dim, input_dim},
        },
    }

    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)
        in_ = net.extern_data.get_default_input_data()
        out = net.get_default_output_layer().output.copy_as_batch_major()
        from test_TFNetworkLayer import make_feed_dict

        out_v, seq_lens = session.run(
            (out.placeholder, in_.get_sequence_lengths()), feed_dict=make_feed_dict(net.extern_data)
        )
        print(out_v, seq_lens)
        sum_over_i = seq_lens * (seq_lens - 1) // 2
        print(sum_over_i)
        assert (sum_over_i[:, None] == out_v).all()


def test_rec_explicit_lstm():
    net_dict = {
        "lstm": {
            "class": "rec",
            "from": "data",
            "unit": {
                "input": {"class": "copy", "from": ["prev:output", "data:source"]},
                "input_gate": {"class": "linear", "from": "input", "activation": "sigmoid", "n_out": 10},
                "forget_gate": {"class": "linear", "from": "input", "activation": "sigmoid", "n_out": 10},
                "output_gate": {"class": "linear", "from": "input", "activation": "sigmoid", "n_out": 10},
                "cell_in": {"class": "linear", "from": "input", "activation": "tanh", "n_out": 10},
                "c": {
                    "class": "eval",
                    "from": ["input_gate", "cell_in", "forget_gate", "prev:c"],
                    "eval": "source(0) * source(1) + source(2) * source(3)",
                },
                "output": {"class": "eval", "from": ["output_gate", "c"], "eval": "source(0) * source(1)"},
            },
        },
        "output": {"class": "softmax", "loss": "ce", "from": "lstm"},
    }
    config = Config(
        {
            "num_inputs": 9,
            "num_outputs": 2,
            "debug_print_layer_output_template": True,
        }
    )
    with make_scope() as session:
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(net_dict)
        loss = net.get_total_loss()
        from test_TFNetworkLayer import make_feed_dict

        feed_dict = make_feed_dict(net.extern_data, same_time=True)
        fetches = net.get_fetches_dict()
        optimizer = tf_compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
        fetches["optim_op"] = optimizer.minimize(loss=loss)
        session.run(tf_compat.v1.global_variables_initializer())
        res = session.run(fetches, feed_dict=feed_dict)
        pprint(res)


def test_RecUnstackLayer_rec_no_input_explicit_axis():
    time_dim = SpatialDim("time")
    net_dict = {
        "output": {
            "class": "rec",
            "from": [],
            "axis": time_dim,
            "unit": {
                "input": {"class": "rec_unstack", "from": "base:data:data"},
                "output": {"class": "combine", "kind": "add", "from": ["prev:output", "input"]},
            },
        },
    }
    config = Config(
        {
            "extern_data": {"data": {"dim": 5, "same_dim_tags_as": {"T": time_dim}}},
        }
    )
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)
        rec_layer = net.get_layer("output")
        assert isinstance(rec_layer, RecLayer)
        cell = rec_layer.cell
        from returnn.tf.layers.rec import _SubnetworkRecCell

        assert isinstance(cell, _SubnetworkRecCell)
        assert_equal(cell.input_layers_moved_out, ["input"])
        assert_equal(cell.layers_in_loop, ["output"])
        in_data = net.extern_data.get_default_input_data()
        out_data = rec_layer.output
        assert_equal(in_data.get_time_dim_tag(), out_data.get_time_dim_tag())
        from test_TFNetworkLayer import make_feed_dict

        session.run(out_data.placeholder, feed_dict=make_feed_dict([in_data]))


def test_RecUnstackLayer_rec_no_input_declare_rec_time():
    net_dict = {
        "output": {
            "class": "rec",
            "from": [],
            "unit": {
                "input": {"class": "rec_unstack", "from": "base:data:data", "axis": "T", "declare_rec_time": True},
                "output": {"class": "combine", "kind": "add", "from": ["prev:output", "input"]},
            },
        },
    }
    config = Config(
        {
            "extern_data": {"data": {"dim": 5}},
        }
    )
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)
        rec_layer = net.get_layer("output")
        assert isinstance(rec_layer, RecLayer)
        cell = rec_layer.cell
        from returnn.tf.layers.rec import _SubnetworkRecCell

        assert isinstance(cell, _SubnetworkRecCell)
        assert_equal(cell.input_layers_moved_out, ["input"])
        assert_equal(cell.layers_in_loop, ["output"])
        in_data = net.extern_data.get_default_input_data()
        out_data = rec_layer.output
        assert_equal(in_data.get_time_dim_tag(), out_data.get_time_dim_tag())
        from test_TFNetworkLayer import make_feed_dict

        session.run(out_data.placeholder, feed_dict=make_feed_dict([in_data]))


def test_search_no_rec_explicit():
    from returnn.tf.layers.rec import _SubnetworkRecCell

    beam_size = 3
    logits = numpy.array([[1.0, 2.0, 3.0, 0.0], [0.0, 4.5, 3.0, 6.0], [5.0, 8.0, 7.5, 0.0]], dtype="float32")
    # Labels/scores of each beam in search should be:
    # frame 0: labels [2, 1, 0], scores [3., 2., 1.], source beam idxs [0, 0, 0]
    # frame 1: labels [3, 3, 1], scores [9., 8., 7.5], source beam idxs [0, 1, 2]
    # frame 2: labels [1, 2, 1], scores [17., 16.5, 16.], source beam idxs [0, 0, 1]
    # Thus, the final three label seqs of the beam search should be:
    # - [2, 3, 1] with score 17.
    # - [2, 3, 2] with score 16.5
    # - [1, 3, 1] with score 16.
    expected_final_seqs = [[2, 3, 1], [2, 3, 2], [1, 3, 1]]
    expected_debug_out = [
        {"src_beam_idxs": [0, 0, 0], "scores": [3.0, 2.0, 1.0], "labels": [2, 1, 0], "step": 0},
        {"src_beam_idxs": [0, 1, 0], "scores": [9.0, 8.0, 7.5], "labels": [3, 3, 1], "step": 1},
        {"src_beam_idxs": [0, 0, 1], "scores": [17.0, 16.5, 16.0], "labels": [1, 2, 1], "step": 2},
    ]
    assert len(expected_final_seqs) == len(expected_debug_out) == beam_size
    n_time = 3
    n_classes = 4
    assert_equal(logits.shape, (n_time, n_classes))
    n_batch = 1
    logits = numpy.expand_dims(logits, axis=0)
    assert_equal(logits.shape, (n_batch, n_time, n_classes))
    print("logits:")
    print(logits)

    ChoiceLayer._debug_out = []

    net_dict = {
        "output": {
            "class": "rec",
            "from": ["data"],
            "unit": {
                "output": {
                    "class": "choice",
                    "from": ["data:source"],
                    "input_type": "log_prob",
                    "explicit_search_source": "prev:output",
                    "initial_output": 0,
                    "beam_size": beam_size,
                    "target": "classes",
                }
            },
        }
    }
    extern_data = ExternData(
        {"data": {"dim": n_classes}, "classes": {"dim": n_classes, "sparse": True, "available_for_inference": False}}
    )
    net = TFNetwork(extern_data=extern_data, search_flag=True, train_flag=False, eval_flag=False)
    net.construct_from_dict(net_dict)
    assert_equal(net.used_data_keys, {"data"})  # not classes
    rec_layer = net.layers["output"]
    assert isinstance(rec_layer, RecLayer)
    subnet = rec_layer.cell
    assert isinstance(subnet, _SubnetworkRecCell)
    assert_equal(subnet.layers_in_loop, ["output"])
    sub_layer = subnet.net.layers["output"]
    assert isinstance(sub_layer, ChoiceLayer)
    assert_equal(sub_layer.output.beam.beam_size, beam_size)
    assert_equal(rec_layer.output.beam.beam_size, beam_size)
    input_search_choices = net.get_search_choices(sources=rec_layer.sources)
    assert not input_search_choices
    assert rec_layer.output.is_time_major
    assert_equal(rec_layer.get_search_beam_size(), beam_size)
    feed_dict = {
        net.extern_data.get_batch_info().dim: 1,
        net.extern_data.data["data"].placeholder: logits,
        net.extern_data.data["data"].size_placeholder[0]: [n_time],
    }
    with tf_compat.v1.Session() as session:
        assert_equal(session.run(net.get_data_batch_dim(), feed_dict=feed_dict), n_batch)
        out, out_sizes = session.run(
            (rec_layer.output.placeholder, rec_layer.output.get_sequence_lengths()), feed_dict=feed_dict
        )
        print("output seq lens:", out_sizes)
        print("output:")
        print(out)
        assert isinstance(out_sizes, numpy.ndarray)
        assert isinstance(out, numpy.ndarray)
        assert_equal(out_sizes.shape, (n_batch * beam_size,))
        assert_equal(out.shape, (n_time, n_batch * beam_size))
        assert_equal(out_sizes.tolist(), [n_time] * beam_size)
        out = numpy.reshape(out, (n_time, n_batch, beam_size))

    print("Debug out:")
    debug_out = ChoiceLayer._debug_out
    ChoiceLayer._debug_out = []
    pprint(debug_out)

    # Assume that beams are sorted by score. See above.
    for beam in range(beam_size):
        out_seq = out[:, 0, beam].tolist()
        expected_seq = expected_final_seqs[beam]
        print("beam %i, out seq %r, expected seq %r" % (beam, out_seq, expected_seq))
        assert_equal(out_seq, expected_final_seqs[beam])

    assert len(debug_out) == n_time
    # Could be that it is not in order (because of parallel execution of the loop).
    debug_out = sorted(debug_out, key=lambda k: k["step"])
    for t in range(n_time):
        debug_t = debug_out[t]
        expected_debug_t = expected_debug_out[t]
        assert isinstance(debug_t, dict) and isinstance(expected_debug_t, dict)
        for k, v in sorted(expected_debug_t.items()):
            assert k in debug_t
            out_v = debug_t[k]
            if isinstance(v, int):
                assert_equal(v, out_v)
            else:
                assert isinstance(out_v, numpy.ndarray)
                assert out_v.shape[0] == n_batch, "t %i, k %r, v %r" % (t, k, v)
                out_v = out_v[0]
                assert_equal(v, out_v.tolist(), "t %i, k %r" % (t, k))
    print("Seems fine.")


def test_search_no_rec_explicit_dyn_len():
    from returnn.tf.layers.rec import _SubnetworkRecCell

    beam_size = 3
    logits = numpy.array(
        [[-1.0, -2.0, -3.0, -9.0], [-0.6, -6.0, -0.5, -2.0], [-0.4, -0.6, -0.7, -1.0]], dtype="float32"
    )
    # Let the 0 label be the EOS symbol. We use length normalization.
    # Labels/scores of each beam in search should be:
    # frame 0: labels [0, 1, 2], scores [-1., -2., -3.], source beam idxs [0, 0, 0]
    # frame 1: labels [0, 2, 0], scores [-2., -2.5, -2.6], source beam idxs [0, 1, 1]
    # frame 2: labels [0, 0, 1], scores [-2.9, -3., -3.1], source beam idxs [1, 0, 1]
    # Thus, the final three label seqs of the beam search should be:
    # - [1, 2, 0] with score -2.9
    # - [0, 0, 0] with score -3.
    # - [1, 2, 1] with score -3.1
    expected_final_seqs = [[1, 2, 0], [0, 0, 0], [1, 2, 1]]
    expected_final_seq_lens = [2, 0, 3]
    expected_debug_out = [
        {"src_beam_idxs": [0, 0, 0], "scores": [-1.0, -2.0, -3.0], "labels": [0, 1, 2], "step": 0},
        {"src_beam_idxs": [0, 1, 1], "scores": [-2.0, -2.5, -2.6], "labels": [0, 2, 0], "step": 1},
        {"src_beam_idxs": [1, 0, 1], "scores": [-2.9, -3.0, -3.1], "labels": [0, 0, 1], "step": 2},
    ]
    assert len(expected_final_seqs) == len(expected_debug_out) == beam_size
    n_time = 3
    n_classes = 4
    assert_equal(logits.shape, (n_time, n_classes))
    n_batch = 1
    logits = numpy.expand_dims(logits, axis=0)
    assert_equal(logits.shape, (n_batch, n_time, n_classes))
    print("logits:")
    print(logits)

    ChoiceLayer._debug_out = []

    net_dict = {
        "output": {
            "class": "rec",
            "from": ["data"],
            "max_seq_len": n_time,
            "unit": {
                "output": {
                    "class": "choice",
                    "from": ["data:source"],
                    "input_type": "log_prob",
                    "explicit_search_source": "prev:output",
                    "initial_output": 0,
                    "beam_size": beam_size,
                    "length_normalization": True,
                    "target": "classes",
                },
                "end": {"class": "compare", "from": ["output"], "value": 0},
            },
        }
    }
    extern_data = ExternData(
        {"data": {"dim": n_classes}, "classes": {"dim": n_classes, "sparse": True, "available_for_inference": False}}
    )
    net = TFNetwork(extern_data=extern_data, search_flag=True, train_flag=False, eval_flag=False)
    net.construct_from_dict(net_dict)
    assert_equal(net.used_data_keys, {"data"})  # not classes
    rec_layer = net.layers["output"]
    assert isinstance(rec_layer, RecLayer)
    subnet = rec_layer.cell
    assert isinstance(subnet, _SubnetworkRecCell)
    assert_equal(set(subnet.layers_in_loop), {"output", "end"})
    sub_layer = subnet.net.layers["output"]
    assert isinstance(sub_layer, ChoiceLayer)
    assert_equal(sub_layer.output.beam.beam_size, beam_size)
    assert_equal(rec_layer.output.beam.beam_size, beam_size)
    input_search_choices = net.get_search_choices(sources=rec_layer.sources)
    assert not input_search_choices
    assert rec_layer.output.is_time_major
    assert_equal(rec_layer.get_search_beam_size(), beam_size)
    feed_dict = {
        net.extern_data.get_batch_info().dim: 1,
        net.extern_data.data["data"].placeholder: logits,
        net.extern_data.data["data"].size_placeholder[0]: [n_time],
    }
    with tf_compat.v1.Session() as session:
        assert_equal(session.run(net.get_data_batch_dim(), feed_dict=feed_dict), n_batch)
        out, out_sizes = session.run(
            (rec_layer.output.placeholder, rec_layer.output.get_sequence_lengths()), feed_dict=feed_dict
        )
        print("output seq lens:", out_sizes)
        assert isinstance(out_sizes, numpy.ndarray)
        assert isinstance(out, numpy.ndarray)
        assert_equal(out_sizes.shape, (n_batch * beam_size,))
        assert_equal(out.shape, (n_time, n_batch * beam_size))
    out = numpy.reshape(out, (n_time, n_batch, beam_size))
    print("output:")
    print(out)

    print("Debug out:")
    debug_out = ChoiceLayer._debug_out
    ChoiceLayer._debug_out = []
    pprint(debug_out)

    assert_equal(out_sizes.tolist(), expected_final_seq_lens)

    # Assume that beams are sorted by score. See above.
    for beam in range(beam_size):
        out_seq = out[:, 0, beam].tolist()
        expected_seq = expected_final_seqs[beam]
        print("beam %i, out seq %r, expected seq %r" % (beam, out_seq, expected_seq))
        assert_equal(out_seq, expected_final_seqs[beam])

    assert len(debug_out) == n_time
    # Could be that it is not in order (because of parallel execution of the loop).
    debug_out = sorted(debug_out, key=lambda k: k["step"])
    for t in range(n_time):
        debug_t = debug_out[t]
        expected_debug_t = expected_debug_out[t]
        assert isinstance(debug_t, dict) and isinstance(expected_debug_t, dict)
        for k, v in sorted(expected_debug_t.items()):
            assert k in debug_t
            out_v = debug_t[k]
            if isinstance(v, int):
                assert_equal(v, out_v)
            else:
                assert isinstance(out_v, numpy.ndarray)
                assert out_v.shape[0] == n_batch, "t %i, k %r, v %r" % (t, k, v)
                out_v = out_v[0]
                assert_allclose(v, out_v.tolist(), err_msg="t %i, k %r" % (t, k))
    print("Seems fine.")


def test_search_multi_choice():
    """
    This is a complex test, which defines a rec layer with multiple search choices (:class:`ChoiceLayer`),
    which have different beam sizes.
    Then we perform search, and get all the results, including beam scores, beam source indices, etc.
    Then we go through the results, and reproduce every single search step.
    """
    from returnn.tf.network import help_on_tf_exception

    rnd = numpy.random.RandomState(42)
    num_choices = 2
    n_batch = 2
    n_time = 3
    n_hidden = 11
    beam_size1, beam_size2 = 3, 5  # use different beam sizes
    dim1, dim2 = 7, 7
    net_dict = {
        "encoder": {"class": "copy", "from": "data"},
        "output": {
            "class": "rec",
            "from": [],
            "target": ["target1", "target2"],
            "max_seq_len": n_time * 3,
            "include_eos": True,
            "unit": {
                "lin1": {"class": "linear", "from": "prev:choice2", "activation": "relu", "n_out": n_hidden},
                "lin1a": {"class": "eval", "from": "prev:lin2", "eval": "source(0) * -0.5"},
                "lin1b": {"class": "combine", "kind": "add", "from": ["lin1", "lin1a"]},
                "prob1": {"class": "linear", "activation": "log_softmax", "from": "lin1b", "n_out": dim1},
                "choice1": {
                    "class": "choice",
                    "from": "prob1",
                    "input_type": "log_prob",
                    "beam_size": beam_size1,
                    "target": "target1",
                    "initial_output": 0,
                },
                "lin2": {"class": "linear", "from": "choice1", "activation": "relu", "n_out": n_hidden},
                "lin2a": {"class": "eval", "from": "lin1", "eval": "source(0) * -0.5"},
                "lin2b": {"class": "combine", "kind": "add", "from": ["lin2", "lin2a"]},
                "prob2": {"class": "linear", "activation": "log_softmax", "from": "lin2b", "n_out": dim2},
                "choice2": {
                    "class": "choice",
                    "from": "prob2",
                    "input_type": "log_prob",
                    "beam_size": beam_size2,
                    "target": "target2",
                    "initial_output": "base:encoder",  # just to enforce that we have a difference in both beams
                },
                "output": {"class": "copy", "from": "choice2"},  # define some output
                "end": {
                    "class": "eval",
                    "from": [":i", "choice2"],
                    "out_type": {"dtype": "bool"},
                    "eval": lambda source, **kwargs: (
                        source(1),  # just mark as used, but do not actually use
                        # Note: We use include_eos, thus the ending frame will be included.
                        tf.greater_equal(source(0), n_time - 1),
                    )[-1],
                },
            },
        },
    }
    # Mark all as output layers, and also add raw copies (to get the original beams).
    subnet_dict = net_dict["output"]["unit"]
    assert isinstance(subnet_dict, dict)
    for name in ["choice1", "choice2"]:
        subnet_dict["_%s_src_beams" % name] = {"class": "choice_get_src_beams", "from": name}
        subnet_dict["_%s_beam_scores" % name] = {"class": "choice_get_beam_scores", "from": name}
    relevant_layer_names = []
    for name, layer_desc in list(subnet_dict.items()):
        if name in ["end", "output"]:
            continue
        assert isinstance(layer_desc, dict)
        relevant_layer_names.append(name)
        layer_desc["is_output_layer"] = True
        subnet_dict["%s_raw" % name] = {"class": "decide_keep_beam", "from": name, "is_output_layer": True}
    config = Config(
        {
            "debug_print_layer_output_template": True,
            "extern_data": {
                "data": {"shape": (), "sparse": True, "dim": dim2},
                "target1": {"sparse": True, "dim": dim1},
                "target2": {"sparse": True, "dim": dim2},
            },
        }
    )

    with make_scope() as session:
        net = TFNetwork(config=config, search_flag=True)
        net.construct_from_dict(net_dict)
        net.print_network_info()
        params = net.get_params_list()
        # Note: Make param initialization explicit here, and make sure this stays the same in the future.
        # The test will depend on exactly these params.
        for param in sorted(params, key=lambda p: p.name):
            param.load(rnd.normal(size=param.shape.as_list()).astype("float32"), session=session)
        input_data = net.extern_data.data["data"]
        input_values = [1, 2]
        assert len(input_values) == n_batch
        feed_dict = {
            net.extern_data.get_batch_info().dim: len(input_values),
            input_data.placeholder: numpy.array(input_values),
        }
        output_data = net.get_default_output_layer().output
        assert output_data.time_dim_axis == 0 and output_data.batch_dim_axis == 1 and output_data.shape == (None,)
        loop_vars = {
            name: (net.get_layer("output/%s" % name), net.get_layer("output/%s_raw" % name))
            for name in relevant_layer_names
        }  # type: typing.Dict[str,typing.Tuple[LayerBase,LayerBase]]
        fetches = (
            output_data.placeholder,
            {name: (l1.output.placeholder, l2.output.placeholder) for (name, (l1, l2)) in loop_vars.items()},
            {
                param.op.name[len("output/rec/") :]: param
                for param in net.get_params_list()
                if param.op.name.startswith("output/rec/lin")
            },
        )

        print("TF execution...")
        try:
            output_value, loop_var_values, lin_params = session.run(fetches, feed_dict=feed_dict)
        except Exception as exc:
            print(exc)
            help_on_tf_exception(session=session, exception=exc, fetches=fetches, feed_dict=feed_dict)
            raise
        # Now we don't need the TF session anymore. We fetched everything we need.

    assert output_value.shape == (n_time, n_batch * beam_size2)
    output_value = output_value.reshape(n_time, n_batch, beam_size2)
    print("output:")
    print(output_value)
    for name, (l1, l2) in loop_vars.items():
        l1_value, l2_value = loop_var_values[name]
        print(l1, "shape:", l1_value.shape)
        print(l2, "shape:", l2_value.shape)  # all the "raw" variants
        assert l1_value.shape[:2] == (n_time, n_batch * beam_size2)  # it's after the (final) second choice
        if name.startswith("_"):
            continue
        if (name.endswith("1") and name != "choice1") or name.endswith("1b") or name in ["choice2", "lin2a"]:
            # All these are after the second choice.
            assert l2_value.shape[:2] == (n_time, n_batch * beam_size2)
        else:
            # After the first choice.
            assert l2_value.shape[:2] == (n_time, n_batch * beam_size1)

    # Now we go through each time step, and check whether all choices (and the whole search) were performed correctly.
    print("Check search...")
    # Needed vars for the beam search.
    cur_beam_size = 1
    scores_base = numpy.zeros((n_batch, 1))  # initial beam scores
    # Keep track of the existing values.
    prev_choice_val = numpy.array(input_values).reshape((n_batch, cur_beam_size))
    prev_lin_val = numpy.zeros((n_batch, cur_beam_size, n_hidden))
    lin_values = numpy.zeros((n_batch, cur_beam_size, num_choices, n_time, n_hidden))
    prob_values = [numpy.zeros((n_batch, cur_beam_size, n_time, dim)) for dim in [dim1, dim2]]
    choices_values = numpy.zeros((n_batch, cur_beam_size, num_choices, n_time), dtype="int32")
    # Loop through the choices.
    for t in range(n_time):
        print("t:", t)
        for choice_idx in range(num_choices):
            dim = [dim1, dim2][choice_idx]
            prev_dim = [dim2, dim1][choice_idx]
            beam_size = [beam_size1, beam_size2][choice_idx]
            prev_beam_size = [beam_size2, beam_size1][
                choice_idx
            ]  # always like cur_beam_size, except of in the beginning
            choice_name = "choice%i" % (choice_idx + 1)
            print("choice:", choice_name, "dim:", dim, "beam size:", beam_size, "incoming beam size:", cur_beam_size)

            assert prev_choice_val.shape == (n_batch, cur_beam_size)
            weights = lin_params["lin%i/W" % (choice_idx + 1)]
            assert weights.shape == (prev_dim, n_hidden)
            assert numpy.all(0 <= prev_choice_val) and numpy.all(prev_choice_val < prev_dim)
            x = weights[prev_choice_val].copy()
            assert x.shape == (n_batch, cur_beam_size, n_hidden)
            x = x + lin_params["lin%i/b" % (choice_idx + 1)]
            x = numpy.maximum(0.0, x)  # relu

            if t < n_time:
                raw_lin_values = loop_var_values["lin%i" % (choice_idx + 1)][1][t]
                assert raw_lin_values.shape == (n_batch * prev_beam_size, n_hidden)
                raw_lin_values = raw_lin_values.reshape((n_batch, prev_beam_size, n_hidden))
                if t == 0 and choice_idx == 0 and cur_beam_size == 1 and prev_beam_size > cur_beam_size:
                    # In the very beginning, the beam size can be larger than what we really have (size 1).
                    # See comments in ChoiceLayer.
                    # We silently ignore and allow this. This is an implementation detail, which also might change later.
                    # Just take the first.
                    raw_lin_values = raw_lin_values[:, :1]
                assert raw_lin_values.shape == x.shape == (n_batch, cur_beam_size, n_hidden)
                numpy.testing.assert_allclose(raw_lin_values, x, rtol=1e-5)
                for b in range(n_batch):
                    for i in range(cur_beam_size):
                        # Note: This is not strictly guaranteed, but very likely. As we guarantee the same random seed,
                        # this will stay valid.
                        # Also, given this property, we can later do some more clever checks.
                        assert numpy.sum(raw_lin_values[b, i]) > 0.0
            lin_values[:, :, choice_idx, t] = x

            if t < n_time:
                raw_lin_b_values = loop_var_values["lin%ib" % (choice_idx + 1)][1][t]
                assert raw_lin_b_values.shape == (n_batch * prev_beam_size, n_hidden)
                raw_lin_b_values = raw_lin_b_values.reshape((n_batch, prev_beam_size, n_hidden))
                if t == 0 and choice_idx == 0 and cur_beam_size == 1 and prev_beam_size > cur_beam_size:
                    raw_lin_b_values = raw_lin_b_values[:, :1]
                assert raw_lin_b_values.shape == x.shape == prev_lin_val.shape == (n_batch, cur_beam_size, n_hidden)
                for b in range(n_batch):
                    for i in range(cur_beam_size):
                        if t > 0 or choice_idx > 0:
                            assert numpy.sum(prev_lin_val[b, i]) > 0.0  # see above
                        numpy.testing.assert_allclose(raw_lin_b_values[b, i], (x - prev_lin_val * 0.5)[b, i], rtol=1e-5)

            scores_in = loop_var_values["prob%i" % (choice_idx + 1)][1][t]
            assert isinstance(scores_in, numpy.ndarray)
            assert scores_in.shape == (n_batch * prev_beam_size, dim)
            scores_in = scores_in.reshape((n_batch, prev_beam_size, dim))
            if t == 0 and choice_idx == 0 and cur_beam_size == 1 and prev_beam_size > cur_beam_size:
                scores_in = scores_in[:, :1]
            assert scores_in.shape == (n_batch, cur_beam_size, dim)
            # In log space, and log prob distribution.
            assert numpy.all(scores_in < 0) and numpy.all(numpy.isclose(numpy.sum(numpy.exp(scores_in), axis=-1), 1.0))
            prob_values[choice_idx][:, :, t] = scores_in

            raw_choices = loop_var_values[choice_name][1][t]
            raw_src_beams = loop_var_values["_%s_src_beams" % choice_name][1][t]
            raw_beam_scores = loop_var_values["_%s_beam_scores" % choice_name][1][t]
            assert raw_choices.shape == raw_src_beams.shape == raw_beam_scores.shape == (n_batch * beam_size,)
            raw_choices = raw_choices.reshape((n_batch, beam_size))
            raw_src_beams = raw_src_beams.reshape((n_batch, beam_size))
            raw_beam_scores = raw_beam_scores.reshape((n_batch, beam_size))

            scores_combined = scores_base[:, :, None] + scores_in
            assert scores_combined.shape == (n_batch, cur_beam_size, dim)

            # Now doing top-k, very explicitly.
            for b in range(n_batch):
                s = [(-scores_combined[b, i, j], (i, j)) for i in range(cur_beam_size) for j in range(dim)]
                s.sort()
                for i in range(beam_size):
                    print(
                        " batch %i, new beam %i: src beam %i, label %i, score %f"
                        % (b, i, s[i][1][0], s[i][1][1], -s[i][0])
                    )
                    numpy.testing.assert_allclose(-s[i][0], raw_beam_scores[b, i], rtol=1e-5)
                    assert_equal(s[i][1][0], raw_src_beams[b, i])
                    assert_equal(s[i][1][1], raw_choices[b, i])

            # Select src beams.
            assert lin_values.shape == (n_batch, cur_beam_size, num_choices, n_time, n_hidden)
            lin_values = numpy.array([lin_values[b, raw_src_beams[b]] for b in range(n_batch)])
            assert lin_values.shape == (n_batch, beam_size, num_choices, n_time, n_hidden)
            assert choices_values.shape == (n_batch, cur_beam_size, num_choices, n_time)
            choices_values = numpy.array([choices_values[b, raw_src_beams[b]] for b in range(n_batch)])
            assert choices_values.shape == (n_batch, beam_size, num_choices, n_time)
            for c in range(num_choices):
                assert_equal(prob_values[c].shape, (n_batch, cur_beam_size, n_time, [dim1, dim2][c]))
            prob_values = [
                numpy.array([prob_values[c][b, raw_src_beams[b]] for b in range(n_batch)]) for c in range(num_choices)
            ]
            for c in range(num_choices):
                assert_equal(prob_values[c].shape, (n_batch, beam_size, n_time, [dim1, dim2][c]))

            # Ok. Update the beam.
            scores_base = raw_beam_scores
            cur_beam_size = beam_size
            choices_values[:, :, choice_idx, t] = raw_choices.copy()
            prev_choice_val = raw_choices.copy()
            prev_lin_val = lin_values[:, :, choice_idx, t].copy()

    print("Our output:")
    print(choices_values[:, :, 1].transpose([2, 0, 1]))

    # Now check if the final selected output and choices are correct.
    assert cur_beam_size == beam_size2
    assert output_value.shape == (n_time, n_batch, cur_beam_size)
    for c in range(num_choices):
        for t in range(n_time):
            selected_choices = loop_var_values["choice%i" % (c + 1)][0][t]
            assert selected_choices.shape == (n_batch * cur_beam_size,)
            selected_choices = selected_choices.reshape(n_batch, cur_beam_size)
            numpy.testing.assert_equal(selected_choices, choices_values[:, :, c, t])
            if c == 1:
                numpy.testing.assert_equal(output_value[t], choices_values[:, :, c, t])
            selected_prob = loop_var_values["prob%i" % (c + 1)][0][t]
            assert selected_prob.shape == (n_batch * cur_beam_size, [dim1, dim2][c])
            selected_prob = selected_prob.reshape(n_batch, cur_beam_size, [dim1, dim2][c])
            numpy.testing.assert_allclose(selected_prob, prob_values[c][:, :, t], rtol=1e-5)
            selected_lin = loop_var_values["lin%i" % (c + 1)][0][t]
            assert selected_lin.shape == (n_batch * cur_beam_size, n_hidden)
            selected_lin = selected_lin.reshape(n_batch, cur_beam_size, n_hidden)
            numpy.testing.assert_allclose(selected_lin, lin_values[:, :, c, t], rtol=1e-5)

    # We also don't strictly enforce this, but this should be likely, i.e. that batch0 is different from batch1.
    assert numpy.any(output_value[:, 0] != output_value[:, 1])


def test_search_multi_choice_simple_keep_beams():
    """
    Mostly like :func:`test_search_multi_choice`, but a bit simplified / stripped down,
    so it is a good base to copy and extend for other cases.
    Also, we extend here by keep_beams.
    """
    from returnn.tf.network import help_on_tf_exception

    rnd = numpy.random.RandomState(42)
    num_choices = 2
    n_batch = 2
    n_time = 3
    n_hidden = 11
    beam_size1, beam_size2 = 3 * 5, 5  # use different beam sizes
    dim1, dim2 = 17, 7
    net_dict = {
        "encoder": {"class": "copy", "from": "data"},
        "output": {
            "class": "rec",
            "from": [],
            "target": ["target1", "target2"],
            "max_seq_len": n_time * 3,
            "include_eos": True,
            "unit": {
                "lin1": {"class": "linear", "from": "prev:choice2", "activation": "relu", "n_out": n_hidden},
                "lin1a": {"class": "eval", "from": "prev:lin2", "eval": "source(0) * -0.5"},
                "lin1b": {"class": "combine", "kind": "add", "from": ["lin1a", "lin1"]},  # order is diff, by intention
                "prob1": {"class": "linear", "activation": "log_softmax", "from": "lin1b", "n_out": dim1},
                "choice1": {
                    "class": "choice",
                    "from": "prob1",
                    "input_type": "log_prob",
                    "beam_size": beam_size1,
                    "keep_beams": True,
                    "target": "target1",
                    "initial_output": 0,
                },
                "lin2": {"class": "linear", "from": "choice1", "activation": "relu", "n_out": n_hidden},
                "lin2a": {"class": "eval", "from": "lin1", "eval": "source(0) * -0.5"},
                "lin2b": {"class": "combine", "kind": "add", "from": ["lin2a", "lin2"]},  # order is diff, by intention
                "prob2": {"class": "linear", "activation": "log_softmax", "from": "lin2b", "n_out": dim2},
                "choice2": {
                    "class": "choice",
                    "from": "prob2",
                    "input_type": "log_prob",
                    "beam_size": beam_size2,
                    "target": "target2",
                    "initial_output": "base:encoder",  # just to enforce that we have a difference in both beams
                },
                "output": {"class": "copy", "from": "choice2"},  # define some output
                "end": {
                    "class": "eval",
                    "from": [":i", "choice2"],
                    "out_type": {"dtype": "bool"},
                    "eval": lambda source, **kwargs: (
                        source(1),  # just mark as used, but do not actually use
                        # Note: We use include_eos, thus the ending frame will be included.
                        tf.greater_equal(source(0), n_time - 1),
                    )[-1],
                },
            },
        },
    }
    # Mark all as output layers, and also add raw copies (to get the original beams).
    subnet_dict = net_dict["output"]["unit"]
    assert isinstance(subnet_dict, dict)
    for name in ["choice1", "choice2"]:
        subnet_dict["_%s_src_beams" % name] = {"class": "choice_get_src_beams", "from": name}
        subnet_dict["_%s_beam_scores" % name] = {"class": "choice_get_beam_scores", "from": name}
    relevant_layer_names = []
    for name, layer_desc in list(subnet_dict.items()):
        if name in ["end", "output"]:
            continue
        assert isinstance(layer_desc, dict)
        relevant_layer_names.append(name)
        layer_desc["is_output_layer"] = True
        subnet_dict["%s_raw" % name] = {"class": "decide_keep_beam", "from": name, "is_output_layer": True}
    config = Config(
        {
            "debug_print_layer_output_template": True,
            "extern_data": {
                "data": {"shape": (), "sparse": True, "dim": dim2},
                "target1": {"sparse": True, "dim": dim1},
                "target2": {"sparse": True, "dim": dim2},
            },
        }
    )

    with make_scope() as session:
        net = TFNetwork(config=config, search_flag=True)
        net.construct_from_dict(net_dict)
        net.print_network_info()
        params = net.get_params_list()
        # Note: Make param initialization explicit here, and make sure this stays the same in the future.
        # The test will depend on exactly these params.
        for param in sorted(params, key=lambda p: p.name):
            param.load(rnd.normal(size=param.shape.as_list()).astype("float32"), session=session)
        input_data = net.extern_data.data["data"]
        input_values = [1, 2]
        assert len(input_values) == n_batch
        feed_dict = {
            net.extern_data.get_batch_info().dim: len(input_values),
            input_data.placeholder: numpy.array(input_values),
        }
        output_data = net.get_default_output_layer().output
        assert output_data.time_dim_axis == 0 and output_data.batch_dim_axis == 1 and output_data.shape == (None,)
        loop_vars = {
            name: (net.get_layer("output/%s" % name), net.get_layer("output/%s_raw" % name))
            for name in relevant_layer_names
        }  # type: typing.Dict[str,typing.Tuple[LayerBase,LayerBase]]
        fetches = (
            output_data.placeholder,
            {name: (l1.output.placeholder, l2.output.placeholder) for (name, (l1, l2)) in loop_vars.items()},
            {
                param.op.name[len("output/rec/") :]: param
                for param in net.get_params_list()
                if param.op.name.startswith("output/rec/lin")
            },
        )

        print("TF execution...")
        try:
            output_value, loop_var_values, lin_params = session.run(fetches, feed_dict=feed_dict)
        except Exception as exc:
            print(exc)
            help_on_tf_exception(session=session, exception=exc, fetches=fetches, feed_dict=feed_dict)
            raise
        # Now we don't need the TF session anymore. We fetched everything we need.

    assert output_value.shape == (n_time, n_batch * beam_size2)
    output_value = output_value.reshape(n_time, n_batch, beam_size2)
    print("output:")
    print(output_value)

    # Now we go through each time step, and check whether all choices (and the whole search) were performed correctly.
    print("Check search...")
    # Needed vars for the beam search.
    cur_beam_size = 1
    scores_base = numpy.zeros((n_batch, 1))  # initial beam scores
    # Keep track of the existing values.
    prev_choice_val = numpy.array(input_values).reshape((n_batch, cur_beam_size))
    prev_lin_val = numpy.zeros((n_batch, cur_beam_size, n_hidden))
    lin_values = numpy.zeros((n_batch, cur_beam_size, num_choices, n_time, n_hidden))
    choices_values = numpy.zeros((n_batch, cur_beam_size, num_choices, n_time), dtype="int32")
    # Loop through the choices.
    for t in range(n_time):
        print("t:", t)
        for choice_idx in range(num_choices):
            dim = [dim1, dim2][choice_idx]
            prev_dim = [dim2, dim1][choice_idx]
            beam_size = [beam_size1, beam_size2][choice_idx]
            prev_beam_size = [beam_size2, beam_size1][
                choice_idx
            ]  # always like cur_beam_size, except of in the beginning
            choice_name = "choice%i" % (choice_idx + 1)
            print("choice:", choice_name, "dim:", dim, "beam size:", beam_size, "incoming beam size:", cur_beam_size)

            assert prev_choice_val.shape == (n_batch, cur_beam_size)
            weights = lin_params["lin%i/W" % (choice_idx + 1)]
            assert weights.shape == (prev_dim, n_hidden)
            assert numpy.all(0 <= prev_choice_val) and numpy.all(prev_choice_val < prev_dim)
            x = weights[prev_choice_val].copy()
            assert x.shape == (n_batch, cur_beam_size, n_hidden)
            x = x + lin_params["lin%i/b" % (choice_idx + 1)]
            x = numpy.maximum(0.0, x)  # relu

            raw_lin_values = loop_var_values["lin%i" % (choice_idx + 1)][1][t]
            assert raw_lin_values.shape == (n_batch * prev_beam_size, n_hidden)
            raw_lin_values = raw_lin_values.reshape((n_batch, prev_beam_size, n_hidden))
            if t == 0 and choice_idx == 0 and cur_beam_size == 1 and prev_beam_size > cur_beam_size:
                # In the very beginning, the beam size can be larger than what we really have (size 1).
                # See comments in ChoiceLayer.
                # We silently ignore and allow this. This is an implementation detail, which also might change later.
                # Just take the first.
                raw_lin_values = raw_lin_values[:, :1]
            assert raw_lin_values.shape == x.shape == (n_batch, cur_beam_size, n_hidden)
            numpy.testing.assert_allclose(raw_lin_values, x, rtol=1e-5)
            for b in range(n_batch):
                for i in range(cur_beam_size):
                    # Note: This is not strictly guaranteed, but very likely. As we guarantee the same random seed,
                    # this will stay valid.
                    # Also, given this property, we can later do some more clever checks.
                    assert numpy.sum(raw_lin_values[b, i]) > 0.0
            lin_values[:, :, choice_idx, t] = x

            raw_lin_b_values = loop_var_values["lin%ib" % (choice_idx + 1)][1][t]
            assert raw_lin_b_values.shape == (n_batch * prev_beam_size, n_hidden)
            raw_lin_b_values = raw_lin_b_values.reshape((n_batch, prev_beam_size, n_hidden))
            if t == 0 and choice_idx == 0 and cur_beam_size == 1 and prev_beam_size > cur_beam_size:
                raw_lin_b_values = raw_lin_b_values[:, :1]
            assert raw_lin_b_values.shape == x.shape == prev_lin_val.shape == (n_batch, cur_beam_size, n_hidden)
            for b in range(n_batch):
                for i in range(cur_beam_size):
                    if t > 0 or choice_idx > 0:
                        assert numpy.sum(prev_lin_val[b, i]) > 0.0  # see above
                    numpy.testing.assert_allclose(raw_lin_b_values[b, i], (x - prev_lin_val * 0.5)[b, i], rtol=1e-5)

            scores_in = loop_var_values["prob%i" % (choice_idx + 1)][1][t]
            assert isinstance(scores_in, numpy.ndarray)
            assert scores_in.shape == (n_batch * prev_beam_size, dim)
            scores_in = scores_in.reshape((n_batch, prev_beam_size, dim))
            if t == 0 and choice_idx == 0 and cur_beam_size == 1 and prev_beam_size > cur_beam_size:
                scores_in = scores_in[:, :1]
            assert scores_in.shape == (n_batch, cur_beam_size, dim)
            # In log space, and log prob distribution.
            assert numpy.all(scores_in < 0) and numpy.all(numpy.isclose(numpy.sum(numpy.exp(scores_in), axis=-1), 1.0))

            raw_choices = loop_var_values[choice_name][1][t]
            raw_src_beams = loop_var_values["_%s_src_beams" % choice_name][1][t]
            raw_beam_scores = loop_var_values["_%s_beam_scores" % choice_name][1][t]
            assert raw_choices.shape == raw_src_beams.shape == raw_beam_scores.shape == (n_batch * beam_size,)
            raw_choices = raw_choices.reshape((n_batch, beam_size))
            raw_src_beams = raw_src_beams.reshape((n_batch, beam_size))
            raw_beam_scores = raw_beam_scores.reshape((n_batch, beam_size))

            scores_combined = scores_base[:, :, None] + scores_in
            assert scores_combined.shape == (n_batch, cur_beam_size, dim)

            # Now doing top-k, very explicitly.
            for b in range(n_batch):
                if choice_idx > 0:
                    s = [(-scores_combined[b, i, j], (i, j)) for i in range(cur_beam_size) for j in range(dim)]
                    s.sort()
                else:  # for choice_idx == 0, we used keep_beams=True
                    assert beam_size % cur_beam_size == 0
                    s = []
                    for i in range(cur_beam_size):
                        s_ = [(-scores_combined[b, i, j], (i, j)) for j in range(dim)]
                        s_.sort()
                        s.extend(s_[: beam_size // cur_beam_size])
                    assert len(s) == beam_size
                for i in range(beam_size):
                    print(
                        " batch %i, new beam %i: src beam %i, label %i, score %f"
                        % (b, i, s[i][1][0], s[i][1][1], -s[i][0])
                    )
                    numpy.testing.assert_allclose(-s[i][0], raw_beam_scores[b, i], rtol=1e-5)
                    assert_equal(s[i][1][0], raw_src_beams[b, i])
                    assert_equal(s[i][1][1], raw_choices[b, i])

            # Select src beams.
            assert lin_values.shape == (n_batch, cur_beam_size, num_choices, n_time, n_hidden)
            lin_values = numpy.array([lin_values[b, raw_src_beams[b]] for b in range(n_batch)])
            assert lin_values.shape == (n_batch, beam_size, num_choices, n_time, n_hidden)
            assert choices_values.shape == (n_batch, cur_beam_size, num_choices, n_time)
            choices_values = numpy.array([choices_values[b, raw_src_beams[b]] for b in range(n_batch)])
            assert choices_values.shape == (n_batch, beam_size, num_choices, n_time)

            # Ok. Update the beam.
            scores_base = raw_beam_scores
            cur_beam_size = beam_size
            choices_values[:, :, choice_idx, t] = raw_choices.copy()
            prev_choice_val = raw_choices.copy()
            prev_lin_val = lin_values[:, :, choice_idx, t].copy()

    print("Our output:")
    print(choices_values[:, :, 1].transpose([2, 0, 1]))

    # Now check if the final selected output and choices are correct.
    assert cur_beam_size == beam_size2
    assert output_value.shape == (n_time, n_batch, cur_beam_size)
    for t in range(n_time):
        numpy.testing.assert_equal(output_value[t], choices_values[:, :, -1, t])

    # We also don't strictly enforce this, but this should be likely, i.e. that batch0 is different from batch1.
    assert numpy.any(output_value[:, 0] != output_value[:, 1])


def test_rec_layer_multi_choice_search_resolve():
    AttNumHeads = 1
    EncKeyTotalDim = 10
    beam_size = 3
    target = "classes"

    net_dict = {
        "lstm0_pool": {"class": "pool", "mode": "max", "padding": "same", "pool_size": (3,), "from": "data"},
        "encoder0": {"class": "copy", "from": "lstm0_pool"},
        "encoder": {"class": "copy", "from": "encoder0"},
        "enc_ctx": {
            "class": "linear",
            "activation": None,
            "with_bias": True,
            "from": ["encoder"],
            "n_out": EncKeyTotalDim,
        },
        "enc_value": {"class": "copy", "from": "encoder"},  # (B, enc-T, D)
        "inv_fertility": {
            "class": "linear",
            "activation": "sigmoid",
            "with_bias": False,
            "from": ["encoder"],
            "n_out": AttNumHeads,
        },
        "enc_seq_len": {"class": "length", "from": "encoder", "sparse": True},
        "output": {
            "class": "rec",
            "from": [],
            "back_prop": False,
            "unit": {
                "weight_feedback": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["prev:accum_att_weights"],
                    "n_out": EncKeyTotalDim,
                },
                "s_transformed": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["s"],
                    "n_out": EncKeyTotalDim,
                },
                "energy_in": {
                    "class": "combine",
                    "kind": "add",
                    "from": ["base:enc_ctx", "s_transformed", "weight_feedback"],
                    "n_out": EncKeyTotalDim,
                },
                "energy_tanh": {"class": "activation", "activation": "tanh", "from": "energy_in"},
                "energy": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["energy_tanh"],
                    "n_out": AttNumHeads,
                },  # (B, enc-T, H)
                "energy1": {"class": "squeeze", "axis": "f", "from": "energy"},  # (B, enc-T)
                "energy2": {"class": "reinterpret_data", "from": "energy1", "set_axes": {"t": "stag:lstm"}},
                # Segment boundaries:
                # - t0/t1/t is the right side (inclusive)
                # - prev:t is the left side (exclusive)
                # - t_start/prev_t_plus1 is the left side (inclusive)
                "prev_t_plus1": {"class": "eval", "from": "prev:t", "eval": "source(0) + 1"},
                "t_start": {
                    "class": "eval",
                    "from": ["prev_t_plus1", "base:enc_seq_len"],
                    "eval": "tf.minimum(source(0), source(1) - 1)",
                },  # to avoid nans
                "t_weights": {
                    "class": "softmax_over_spatial",
                    "from": "energy2",
                    "axis": "stag:lstm",
                    "start": "t_start",
                },
                "t_weights1": {
                    # ChoiceLayer works on the feature axis.
                    "class": "reinterpret_data",
                    "from": "t_weights",
                    "set_axes": {"f": "stag:lstm"},
                },
                "t0": {
                    "class": "choice",
                    "from": "t_weights1",
                    "target": None,
                    "beam_size": beam_size * 4,
                    "keep_beams": True,
                    "length_normalization": False,
                    "initial_output": -1,
                },  # (B,)
                # Note: If beam-size > enc_seq_len, we end up with invalid t in the beam. Fix that.
                "t1": {
                    "class": "eval",
                    "from": ["t0", "t_start", "base:enc_seq_len"],
                    "eval": "tf.clip_by_value(source(0), source(1), source(2) - 1)",
                },
                "t": {"class": "copy", "from": "t1", "initial_output": -1, "is_output_layer": True},
                "window_start": {"class": "eval", "from": "t", "eval": "source(0) - 5"},
                "att_weights": {
                    "class": "softmax_over_spatial",
                    "from": "energy2",
                    "axis": "stag:lstm",
                    "window_start": "window_start",
                    "window_size": 10,
                },  # (B, enc-T)
                "att_soft": {
                    "class": "generic_attention",
                    "weights": "att_weights",
                    "base": "base:enc_value",
                },  # (B, V)
                "att": {"class": "copy", "from": "att_soft"},
                "accum_att_weights": {
                    "class": "eval",
                    "from": ["prev:accum_att_weights", "att_weights", "base:inv_fertility"],
                    "eval": "source(0) + source(1) * source(2) * 0.5",
                    "out_type": {"dim": AttNumHeads, "shape": (None, AttNumHeads)},
                },
                "s": {
                    "class": "rec",
                    "unit": "nativelstm2",
                    "from": ["prev:target_embed", "prev:att"],
                    "n_out": 1000,
                    "dropout": 0.2,
                },
                "readout_in": {
                    "class": "linear",
                    "from": ["s", "prev:target_embed", "att"],
                    "activation": None,
                    "n_out": 1000,
                    "dropout": 0.3,
                },
                "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
                "output_prob": {
                    "class": "softmax",
                    "from": ["readout"],
                    "dropout": 0.3,
                    "target": target,
                    "loss": None,
                },
                "target_embed": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["output"],
                    "n_out": 621,
                    "initial_output": "var",
                },
                "output": {
                    "class": "choice",
                    "target": target,
                    "beam_size": beam_size,
                    "from": ["output_prob"],
                    "initial_output": 0,
                    "search": True,
                    "length_normalization": True,
                },
                # "end": {"class": "compare", "from": ["t", "base:enc_seq_len"], "kind": "greater"},
                "end": {"class": "compare", "from": "output", "value": 0},
            },
            "target": [target],
            "max_seq_len": "max_len_from('base:encoder0')",
        },
    }

    config = Config(
        {
            "debug_print_layer_output_template": True,
        }
    )

    print("Create network")
    n_src_dim = 7
    n_tgt_dim = 11

    from returnn.datasets.generating import StaticDataset
    from returnn.tf.data_pipeline import FeedDictDataProvider
    from returnn.engine.batch import Batch, BatchSetGenerator

    dataset = StaticDataset(
        data=[
            {"data": numpy.random.normal(size=(11, n_src_dim)).astype("float32"), target: numpy.array([3, 6, 0])},
            {
                "data": numpy.random.normal(size=(13, n_src_dim)).astype("float32"),
                target: numpy.array([3, 6, 2, 1, 4, 5, 0]),
            },
        ],
        output_dim={"data": [n_src_dim, 2], "classes": [n_tgt_dim, 1]},
    )
    dataset.init_seq_order(epoch=1)
    batch = Batch()
    batch.add_sequence_as_slice(seq_idx=0, seq_start_frame=0, length=dataset.get_seq_length(0))
    batch.add_sequence_as_slice(seq_idx=1, seq_start_frame=0, length=dataset.get_seq_length(1))
    print("batch:", batch, "num frames:", batch.get_total_num_frames())
    print("batch dims:", batch.max_num_frames_per_slice * batch.num_slices)
    batch_generator = iter([batch])
    batches = BatchSetGenerator(dataset, generator=batch_generator)

    with make_scope() as session:
        extern_data = ExternData(
            {"data": {"dim": n_src_dim}, target: {"dim": n_tgt_dim, "sparse": True, "available_for_inference": False}}
        )

        net = TFNetwork(extern_data=extern_data, search_flag=True, train_flag=False, config=config)
        net.construct_from_dict(net_dict)

        out_layer = net.layers["output"]
        assert isinstance(out_layer, RecLayer)
        net.initialize_params(session)
        data_provider = FeedDictDataProvider(
            tf_session=session, extern_data=extern_data, data_keys=["data"], dataset=dataset, batches=batches
        )
        feed_dict, meta_step_info = data_provider.get_feed_dict(single_threaded=True)
        try:
            print("Output:")
            out = session.run(out_layer.output.placeholder, feed_dict=feed_dict)
            pprint(out)
        except Exception as exc:
            print()
            print("EXCEPTION " + "-" * 60)
            print("Exception happened:", str(exc).splitlines()[0])
            print()
            out_layer.cell._handle_construct_exception()
            print()
            print("TF debug info:")
            help_on_tf_exception(
                session=session,
                exception=exc,
                fetches=out_layer.output.placeholder,
                feed_dict=feed_dict,
                meta_step_info=meta_step_info,
                extern_data=data_provider.extern_data,
            )
            raise


def test_search_SplitBatchBeamLayer():
    from returnn.tf.util.data import batch_dim

    time_dim = SpatialDim("time")
    beam_dim = SpatialDim("beam", 3)
    classes_dim = FeatureDim("classes", 13)

    def make_dummy_beam(src):
        return {
            "class": "rec",
            "from": src,
            "target": "classes",
            "unit": {
                "lin1": {"class": "linear", "from": "data:source", "out_dim": classes_dim},
                "lin2": {"class": "linear", "from": "prev:output", "out_dim": classes_dim},
                "combined": {"class": "combine", "kind": "add", "from": ["lin1", "lin2"]},
                "prob": {"class": "activation", "from": "combined", "activation": "softmax"},
                "output": {
                    "class": "choice",
                    "from": "prob",
                    "beam_size": beam_dim.dimension,
                    "target": "classes",
                    "initial_output": 0,
                },
            },
        }

    net_dict = {
        "nom": make_dummy_beam("data"),
        "den": make_dummy_beam("data"),
        # Get the best beam score of output
        "nom_beam_scores": {"class": "choice_get_beam_scores", "from": "nom"},
        "nom_beam_scores_best": {"class": "decide", "from": "nom_beam_scores"},
        # Get the sum of all beam scores of extra.search:output
        "den_beam_scores": {"class": "choice_get_beam_scores", "from": "den"},
        "den_beam_scores_split": {"class": "split_batch_beam", "from": "den_beam_scores", "beam_dim": beam_dim},
        "den_beam_scores_logsumexp": {
            "class": "reduce",
            "mode": "logsumexp",
            "from": "den_beam_scores_split",
            "axis": beam_dim,
        },
        "log_prob": {
            "class": "combine",
            "from": ["nom_beam_scores_best", "den_beam_scores_logsumexp"],
            "kind": "sub",
            "is_output_layer": True,
        },
    }

    config = Config(
        {
            "extern_data": {
                "data": {"dim_tags": [batch_dim, time_dim, FeatureDim("input-dim", 7)]},
                "classes": {"dim_tags": [batch_dim, time_dim], "sparse_dim": classes_dim},
            },
        }
    )

    with make_scope() as session:
        net = TFNetwork(search_flag=True, train_flag=False, config=config)
        net.construct_from_dict(net_dict)
        net.initialize_params(session)
        den_beam_scores = net.get_layer("den_beam_scores").output
        assert den_beam_scores.beam and den_beam_scores.beam.beam_size == beam_dim.dimension
        assert den_beam_scores.dim_tags == (batch_dim,)
        den_beam_scores_split = net.get_layer("den_beam_scores_split").output
        assert not den_beam_scores_split.beam
        assert den_beam_scores_split.dim_tags == (batch_dim, beam_dim)
        log_prob = net.get_layer("log_prob").output
        from test_TFNetworkLayer import make_feed_dict

        n_batch = 5
        den_beam_scores_split_np, log_prob_np = session.run(
            (den_beam_scores_split.placeholder, log_prob.placeholder),
            feed_dict=make_feed_dict(net.extern_data, n_batch=n_batch),
        )
        print("den_beam_scores_split_np:", den_beam_scores_split_np)
        print("log_prob_np:", log_prob_np)
        assert den_beam_scores_split_np.shape == (n_batch, beam_dim.dimension) and log_prob_np.shape == (n_batch,)


def test_ChoiceLayer_no_loop_dep():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim, ImplicitSparseDim

    time_dim = SpatialDim("time")
    feat_dim = FeatureDim("feat", 5)

    config = Config(dict(extern_data={"data": {"dim_tags": (batch_dim, time_dim, feat_dim)}}))

    enc_dim = FeatureDim("enc", 10)
    classes_dim = FeatureDim("classes", 3)

    net_dict = {
        "encoder": {"class": "linear", "from": "data", "out_dim": enc_dim},
        "loop": {
            "class": "rec",
            "from": [],
            "unit": {
                "rec_unstack": {
                    "class": "rec_unstack",
                    "from": "base:encoder",
                    "axis": time_dim,
                    "out_shape": {batch_dim, enc_dim},
                },
                "log_prob": {
                    "class": "linear",
                    "from": "rec_unstack",
                    "out_dim": classes_dim,
                    "activation": "log_softmax",
                },
                "choice": {
                    "class": "choice",
                    "from": "log_prob",
                    "target": None,
                    "beam_size": 3,
                    "search": True,
                    "input_type": "log_prob",
                    "length_normalization": False,
                    "out_shape": {batch_dim, ImplicitSparseDim(classes_dim)},
                },
                "output": {"class": "copy", "from": "choice", "out_shape": {batch_dim, ImplicitSparseDim(classes_dim)}},
            },
            "axis": time_dim,
            "out_shape": {batch_dim, time_dim, ImplicitSparseDim(classes_dim)},
            "name_scope": "",
        },
        "output": {
            "class": "copy",
            "from": "loop/output",
            "out_shape": {batch_dim, time_dim, ImplicitSparseDim(classes_dim)},
        },
    }

    with make_scope() as session:
        net = TFNetwork(config=config, search_flag=True)
        net.construct_from_dict(net_dict)
        net.initialize_params(session)
        output = net.get_default_output_layer().output
        from test_TFNetworkLayer import make_feed_dict

        session.run(output.placeholder, feed_dict=make_feed_dict(net.extern_data))


def test_target_with_beam():
    beam_size = 4
    teacher_target = "classes"
    student_target = "teacher_hypotheses_data"  # "teacher_hypotheses_data" is the teacher's beam data

    AttNumHeads = 1
    EncKeyTotalDim = 5
    EncValueTotalDim = 5
    EncValuePerHeadDim = EncValueTotalDim // AttNumHeads

    net_dict = {
        # Teacher seq2seq model
        "teacher_encoder": {
            "class": "linear",
            "activation": None,
            "from": "data:data",
            "n_out": EncValueTotalDim,
            "trainable": False,
        },  # dim: EncValueTotalDim
        "teacher_enc_ctx": {
            "class": "linear",
            "activation": None,
            "with_bias": True,
            "from": ["teacher_encoder"],
            "n_out": EncKeyTotalDim,
            "trainable": False,
        },  # preprocessed_attended in Blocks
        "teacher_inv_fertility": {
            "class": "linear",
            "activation": "sigmoid",
            "with_bias": False,
            "from": ["teacher_encoder"],
            "n_out": AttNumHeads,
            "trainable": False,
        },
        "teacher_enc_value": {
            "class": "split_dims",
            "axis": "F",
            "dims": (AttNumHeads, EncValuePerHeadDim),
            "from": ["teacher_encoder"],
            "trainable": False,
        },  # (B, enc-T, H, D'/H)
        "teacher_output": {
            "class": "rec",
            "from": [],
            "unit": {
                "output": {
                    "class": "choice",
                    "target": teacher_target,
                    "beam_size": beam_size,
                    "from": ["teacher_output_prob"],
                    "initial_output": 0,
                    "trainable": False,
                },
                "end": {"class": "compare", "from": ["output"], "value": 0, "trainable": False},
                "teacher_target_embed": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["output"],
                    "n_out": 5,
                    "initial_output": 0,
                    "trainable": False,
                },  # feedback_input
                "teacher_weight_feedback": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["prev:teacher_accum_att_weights"],
                    "n_out": EncKeyTotalDim,
                    "dropout": 0.3,
                    "trainable": False,
                },
                "teacher_s_transformed": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["teacher_s"],
                    "n_out": EncKeyTotalDim,
                    "dropout": 0.3,
                    "trainable": False,
                },
                "teacher_energy_in": {
                    "class": "combine",
                    "kind": "add",
                    "from": ["base:teacher_enc_ctx", "teacher_weight_feedback", "teacher_s_transformed"],
                    "n_out": EncKeyTotalDim,
                    "trainable": False,
                },
                "teacher_energy_tanh": {
                    "class": "activation",
                    "activation": "tanh",
                    "from": ["teacher_energy_in"],
                    "trainable": False,
                },
                "teacher_energy": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["teacher_energy_tanh"],
                    "n_out": AttNumHeads,
                    "trainable": False,
                },  # (B, enc-T, H)
                "teacher_att_weights": {
                    "class": "softmax_over_spatial",
                    "from": ["teacher_energy"],
                    "trainable": False,
                },
                # (B, enc-T, H)
                "teacher_accum_att_weights": {
                    "class": "eval",
                    "from": ["prev:teacher_accum_att_weights", "teacher_att_weights", "base:teacher_inv_fertility"],
                    "eval": "source(0) + source(1) * source(2) * 0.5",
                    "out_type": {"dim": AttNumHeads, "shape": (None, AttNumHeads)},
                    "trainable": False,
                },
                "teacher_att0": {
                    "class": "generic_attention",
                    "weights": "teacher_att_weights",
                    "base": "base:teacher_enc_value",
                    "trainable": False,
                },  # (B, H, V)
                "teacher_att": {
                    "class": "merge_dims",
                    "axes": ["dim:%i" % AttNumHeads, "dim:%i" % EncValuePerHeadDim],
                    "from": "teacher_att0",
                    "trainable": False,
                },
                # (B, H*V)
                "teacher_s": {
                    "class": "rnn_cell",
                    "unit": "LSTMBlock",
                    "from": ["prev:teacher_target_embed", "prev:teacher_att"],
                    "n_out": 5,
                    "dropout": 0.3,
                    "trainable": False,
                },  # transform
                "teacher_readout_in": {
                    "class": "linear",
                    "from": ["teacher_s", "prev:teacher_target_embed", "teacher_att"],
                    "activation": None,
                    "n_out": 10,
                    "dropout": 0.3,
                    "trainable": False,
                },
                # merge + post_merge bias
                "teacher_readout": {
                    "class": "reduce_out",
                    "mode": "max",
                    "num_pieces": 2,
                    "from": ["teacher_readout_in"],
                    "trainable": False,
                },
                "teacher_output_prob": {
                    "class": "softmax",
                    "from": ["teacher_readout"],
                    "dropout": 0.3,
                    "target": teacher_target,
                    "trainable": False,
                },
            },
            "target": teacher_target,
            "max_seq_len": "max_len_from('base:teacher_encoder') * 3",
            "trainable": False,
        },
        # the teacher's decision layer is actually not used,
        # since the hypotheses data is fetched from the teacher's choice layer (or teacher_output)...
        "teacher_decision": {
            "class": "decide",
            "from": ["teacher_output"],
            "loss": "edit_distance",
            "target": teacher_target,
            "loss_opts": {},
            "register_as_extern_data": "teacher_hypotheses",
            "trainable": False,
        },
        # Register Teacher's beam as external data
        "teacher_hypotheses": {
            "class": "copy",
            "from": ["extra.search:teacher_output"],
            "register_as_extern_data": student_target,
        },
        # Student seq2seq model
        "student_encoder": {"class": "linear", "activation": None, "from": "data:data", "n_out": EncValueTotalDim},
        # dim: EncValueTotalDim
        "student_enc_ctx": {
            "class": "linear",
            "activation": None,
            "with_bias": True,
            "from": ["student_encoder"],
            "n_out": EncKeyTotalDim,
        },  # preprocessed_attended in Blocks
        "student_inv_fertility": {
            "class": "linear",
            "activation": "sigmoid",
            "with_bias": False,
            "from": ["student_encoder"],
            "n_out": AttNumHeads,
        },
        "student_enc_value": {
            "class": "split_dims",
            "axis": "F",
            "dims": (AttNumHeads, EncValuePerHeadDim),
            "from": ["student_encoder"],
        },  # (B, enc-T, H, D'/H)
        "student_output": {
            "class": "rec",
            "from": [],
            "unit": {
                "output": {
                    "class": "choice",
                    "target": student_target,
                    "beam_size": beam_size,
                    "from": ["student_output_prob"],
                    "initial_output": 0,
                },
                "end": {"class": "compare", "from": ["output"], "value": 0},
                "student_target_embed": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["output"],
                    "n_out": 5,
                    "initial_output": 0,
                },  # feedback_input
                "student_weight_feedback": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["prev:student_accum_att_weights"],
                    "n_out": EncKeyTotalDim,
                    "dropout": 0.3,
                },
                "student_s_transformed": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["student_s"],
                    "n_out": EncKeyTotalDim,
                    "dropout": 0.3,
                },
                "student_energy_in": {
                    "class": "combine",
                    "kind": "add",
                    "from": ["base:student_enc_ctx", "student_weight_feedback", "student_s_transformed"],
                    "n_out": EncKeyTotalDim,
                },
                "student_energy_tanh": {"class": "activation", "activation": "tanh", "from": ["student_energy_in"]},
                "student_energy": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["student_energy_tanh"],
                    "n_out": AttNumHeads,
                },  # (B, enc-T, H)
                "student_att_weights": {"class": "softmax_over_spatial", "from": ["student_energy"]},  # (B, enc-T, H)
                "student_accum_att_weights": {
                    "class": "eval",
                    "from": ["prev:student_accum_att_weights", "student_att_weights", "base:student_inv_fertility"],
                    "eval": "source(0) + source(1) * source(2) * 0.5",
                    "out_type": {"dim": AttNumHeads, "shape": (None, AttNumHeads)},
                },
                "student_att0": {
                    "class": "generic_attention",
                    "weights": "student_att_weights",
                    "base": "base:student_enc_value",
                },  # (B, H, V)
                "student_att": {
                    "class": "merge_dims",
                    "axes": ["dim:%i" % AttNumHeads, "dim:%i" % EncValuePerHeadDim],
                    "from": "student_att0",
                },  # (B, H*V)
                "student_s": {
                    "class": "rnn_cell",
                    "unit": "LSTMBlock",
                    "from": ["prev:student_target_embed", "prev:student_att"],
                    "n_out": 5,
                    "dropout": 0.3,
                },  # transform
                "student_readout_in": {
                    "class": "linear",
                    "from": ["student_s", "prev:student_target_embed", "student_att"],
                    "activation": None,
                    "n_out": 6,
                    "dropout": 0.3,
                },  # merge + post_merge bias
                "student_readout": {
                    "class": "reduce_out",
                    "mode": "max",
                    "num_pieces": 2,
                    "from": ["student_readout_in"],
                },
                "student_output_prob": {
                    "class": "softmax",
                    "from": ["student_readout"],
                    "dropout": 0.3,
                    "target": student_target,
                    "loss": "ce",
                    "loss_opts": {"label_smoothing": 0.1},
                },
            },
            "target": student_target,
            "max_seq_len": "max_len_from('base:student_encoder') * 3",
        },
        # Declare Student's output as overall network's output
        "output": {"class": "copy", "from": ["student_output"]},
    }

    config = Config(
        {
            "debug_print_layer_output_template": True,
            "debug_print_layer_output_shape": True,
            "debug_runtime_sanity_checks": True,
        }
    )

    with make_scope() as session:
        extern_data = ExternData(
            {
                "data": {"dim": 3, "sparse": True},
                "classes": {"dim": 3, "sparse": True, "available_for_inference": False},
            }
        )
        network = TFNetwork(extern_data=extern_data, train_flag=True, config=config)
        network.construct_from_dict(net_dict)
        output_layer = network.get_default_output_layer()
        x_data = extern_data.get_data("data")
        y_data = output_layer.output

        session.run(tf_compat.v1.global_variables_initializer())

        from test_TFNetworkLayer import make_feed_dict

        feed_dict = make_feed_dict([network.extern_data.data[key] for key in ["data", "classes"]], n_batch=3)
        x, y, loss = session.run(
            (x_data.placeholder, y_data.placeholder, network.get_total_loss()), feed_dict=feed_dict
        )
        assert x.shape[x_data.batch_dim_axis] * beam_size == y.shape[y_data.batch_dim_axis]


def test_rec_layer_move_out_of_loop():
    from returnn.tf.layers.rec import _SubnetworkRecCell
    from returnn.tf.util.basic import get_global_train_flag_placeholder, stop_event_writer_thread

    n_src_dim = 5
    n_tgt_dim = 7
    beam_size = 12
    config = Config()
    config.update({"debug_print_layer_output_template": True})

    def get_net_dict():
        return {
            "source_embed": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "n_out": 6,
                "from": "data:data",
            },
            "lstm0_fw": {
                "class": "rec",
                "unit": "standardlstm",
                "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                "initial_state": "var",
                "n_out": 10,
                "direction": 1,
                "from": ["source_embed"],
            },
            "lstm0_bw": {
                "class": "rec",
                "unit": "standardlstm",
                "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                "initial_state": "var",
                "n_out": 10,
                "direction": -1,
                "from": ["source_embed"],
            },
            "lstm1_fw": {
                "class": "rec",
                "unit": "standardlstm",
                "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                "initial_state": "var",
                "n_out": 10,
                "direction": 1,
                "from": ["lstm0_fw", "lstm0_bw"],
            },
            "lstm1_bw": {
                "class": "rec",
                "unit": "standardlstm",
                "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                "initial_state": "var",
                "n_out": 10,
                "direction": -1,
                "from": ["lstm0_fw", "lstm0_bw"],
            },
            "encoder": {"class": "copy", "from": ["lstm1_fw", "lstm1_bw"]},
            "enc_ctx": {"class": "linear", "activation": None, "with_bias": True, "from": ["encoder"], "n_out": 10},
            "fertility": {
                "class": "linear",
                "activation": "sigmoid",
                "with_bias": False,
                "from": ["encoder"],
                "n_out": 1,
            },
            "output": {
                "class": "rec",
                "from": [],
                "unit": {
                    "output": {
                        "class": "choice",
                        "target": "classes",
                        "beam_size": beam_size,
                        "from": ["output_prob"],
                        "initial_output": 0,
                    },
                    "end": {"class": "compare", "from": ["output"], "value": 0},
                    "target_embed": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["output"],
                        "n_out": 6,
                        "initial_output": "apply(0)",
                    },
                    "weight_feedback": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["prev:accum_att_weights"],
                        "n_out": 10,
                    },
                    "prev_s_state": {"class": "get_last_hidden_state", "from": ["prev:s"], "n_out": 20},
                    "prev_s_transformed": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["prev_s_state"],
                        "n_out": 10,
                    },
                    "energy_in": {
                        "class": "combine",
                        "kind": "add",
                        "from": ["base:enc_ctx", "weight_feedback", "prev_s_transformed"],
                        "n_out": 10,
                    },
                    "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
                    "energy": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["energy_tanh"],
                        "n_out": 1,
                    },
                    "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, enc-T, 1)
                    "accum_att_weights": {
                        "class": "eval",
                        "from": ["prev:accum_att_weights", "att_weights", "base:fertility"],
                        "eval": "source(0) + source(1) / (2.0 * source(2))",
                        "out_type": {"dim": 1, "shape": (None, 1)},
                    },
                    "att": {
                        "class": "generic_attention",
                        "weights": "att_weights",
                        "base": "base:encoder",
                        "auto_squeeze": True,
                    },
                    "s": {
                        "class": "rnn_cell",
                        "unit": "standardlstm",
                        "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                        "initial_state": "var",
                        "from": ["target_embed", "att"],
                        "n_out": 10,
                    },
                    "readout_in": {
                        "class": "linear",
                        "from": ["prev:s", "prev:target_embed", "att"],
                        "activation": None,
                        "n_out": 10,
                    },
                    "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
                    "output_prob": {"class": "softmax", "from": ["readout"], "target": "classes", "loss": "ce"},
                },
                "target": "classes",
                "max_seq_len": 20,
            },
            "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance", "target": "classes"},
        }

    print("Constructing search network.")
    tf_compat.v1.reset_default_graph()
    extern_data = ExternData(
        {
            "data": {"dim": n_src_dim, "sparse": True},
            "classes": {"dim": n_tgt_dim, "sparse": True, "available_for_inference": False},
        }
    )
    search_net = TFNetwork(extern_data=extern_data, search_flag=True, train_flag=False, eval_flag=True, config=config)
    search_net.construct_from_dict(get_net_dict())
    search_out_layer = search_net.layers["output"]
    assert isinstance(search_out_layer, RecLayer)
    assert isinstance(search_out_layer.cell, _SubnetworkRecCell)
    assert not search_out_layer.cell.input_layers_moved_out
    assert not search_out_layer.cell.output_layers_moved_out
    print("=" * 40)

    def train(net):
        """
        :param TFNetwork net:
        """
        from returnn.datasets.generating import StaticDataset
        from returnn.tf.data_pipeline import FeedDictDataProvider
        from returnn.engine.batch import Batch, BatchSetGenerator
        from returnn.util.basic import dict_joined

        dataset = StaticDataset(
            data=[
                {"data": numpy.array([2, 4, 1, 0]), "classes": numpy.array([3, 6, 0])},
                {"data": numpy.array([2, 4, 1, 3, 0]), "classes": numpy.array([3, 6, 2, 1, 4, 5, 0])},
            ],
            output_dim={"data": [n_src_dim, 1], "classes": [n_tgt_dim, 1]},
        )
        dataset.init_seq_order(epoch=1)
        batch = Batch()
        batch.add_sequence_as_slice(seq_idx=0, seq_start_frame=0, length=dataset.get_seq_length(0))
        batch.add_sequence_as_slice(seq_idx=1, seq_start_frame=0, length=dataset.get_seq_length(1))
        print("batch:", batch, "num frames:", batch.get_total_num_frames())
        print("batch dims:", batch.max_num_frames_per_slice * batch.num_slices)
        batch_generator = iter([batch])
        batches = BatchSetGenerator(dataset, generator=batch_generator)
        out_layer = net.layers["output"]
        assert isinstance(out_layer, RecLayer)
        assert isinstance(out_layer.cell, _SubnetworkRecCell)

        with tf_compat.v1.Session() as session:
            net.initialize_params(session)
            data_provider = FeedDictDataProvider(
                tf_session=session,
                extern_data=extern_data,
                data_keys=["data", "classes"],
                dataset=dataset,
                batches=batches,
            )
            feed_dict, meta_step_info = data_provider.get_feed_dict(single_threaded=True)
            if isinstance(net.train_flag, tf.Tensor):
                feed_dict[net.train_flag] = True
            try:
                out = session.run(
                    dict_joined(
                        {"data:%s:seq_len" % k: v.get_sequence_lengths() for (k, v) in net.extern_data.data.items()},
                        {"layer:%s:out_seq_len" % k: l.output.get_sequence_lengths() for (k, l) in net.layers.items()},
                        {
                            "rec_layer_in:%s:out_seq_len" % k: l.output.get_sequence_lengths()
                            for (k, l) in out_layer.cell.input_layers_net.layers.items()
                        }
                        if out_layer.cell.input_layers_net
                        else {},
                        {
                            "rec_layer_out:%s:out_seq_len" % k: l.output.get_sequence_lengths()
                            for (k, l) in out_layer.cell.output_layers_net.layers.items()
                        }
                        if out_layer.cell.output_layers_net
                        else {},
                    ),
                    feed_dict=feed_dict,
                )
                pprint(out)
                out = session.run({"objective": net.get_objective()}, feed_dict=feed_dict)
                pprint(out)
            except Exception as exc:
                print("Exception happened:", str(exc).splitlines()[0])
                print("Writing TF log file.")
                writer = tf_compat.v1.summary.FileWriter(".", filename_suffix="test_rec_layer_move_out_of_loop")
                writer.add_graph(session.graph)
                writer.close()
                stop_event_writer_thread(writer)
                raise

    print("Constructing train network.")
    tf_compat.v1.reset_default_graph()
    extern_data = ExternData(
        {
            "data": {"dim": n_src_dim, "sparse": True},
            "classes": {"dim": n_tgt_dim, "sparse": True, "available_for_inference": False},
        }
    )
    train_net = TFNetwork(
        extern_data=extern_data, search_flag=False, train_flag=get_global_train_flag_placeholder(), config=config
    )
    assert train_net.eval_flag is True
    train_net.construct_from_dict(get_net_dict())
    train_out_layer = train_net.layers["output"]
    assert isinstance(train_out_layer, RecLayer)
    assert isinstance(train_out_layer.cell, _SubnetworkRecCell)
    assert_equal(set(train_out_layer.cell.input_layers_moved_out), {"output", "target_embed"})
    assert_equal(set(train_out_layer.cell.output_layers_moved_out), {"output_prob", "readout_in", "readout"})
    train(train_net)
    print("=" * 40)

    print("Constructing train network with optimize_move_layers_out=False.")
    config.set("optimize_move_layers_out", False)
    tf_compat.v1.reset_default_graph()
    extern_data = ExternData(
        {
            "data": {"dim": n_src_dim, "sparse": True},
            "classes": {"dim": n_tgt_dim, "sparse": True, "available_for_inference": False},
        }
    )
    train_not_optim_net = TFNetwork(
        extern_data=extern_data, search_flag=False, train_flag=get_global_train_flag_placeholder(), config=config
    )
    assert train_not_optim_net.eval_flag is True
    train_not_optim_net.construct_from_dict(get_net_dict())
    train_not_optim_out_layer = train_not_optim_net.layers["output"]
    assert isinstance(train_not_optim_out_layer, RecLayer)
    assert isinstance(train_not_optim_out_layer.cell, _SubnetworkRecCell)
    assert not train_not_optim_out_layer.cell.input_layers_moved_out
    assert not train_not_optim_out_layer.cell.output_layers_moved_out
    train(train_not_optim_net)


def test_rec_layer_move_out_of_loop_keep_constraints():
    from returnn.tf.layers.rec import _SubnetworkRecCell
    from returnn.tf.util.basic import get_global_train_flag_placeholder

    n_src_dim = 5
    n_tgt_dim = 7
    beam_size = 12
    config = Config()
    config.update({"debug_print_layer_output_template": True})

    def get_net_dict(l2_target_embed=0.0, l2_readout_in=0.0):
        return {
            "source_embed": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "n_out": 6,
                "from": "data:data",
            },
            "lstm0_fw": {
                "class": "rec",
                "unit": "standardlstm",
                "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                "initial_state": "var",
                "n_out": 10,
                "direction": 1,
                "from": ["source_embed"],
            },
            "lstm0_bw": {
                "class": "rec",
                "unit": "standardlstm",
                "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                "initial_state": "var",
                "n_out": 10,
                "direction": -1,
                "from": ["source_embed"],
            },
            "lstm1_fw": {
                "class": "rec",
                "unit": "standardlstm",
                "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                "initial_state": "var",
                "n_out": 10,
                "direction": 1,
                "from": ["lstm0_fw", "lstm0_bw"],
            },
            "lstm1_bw": {
                "class": "rec",
                "unit": "standardlstm",
                "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                "initial_state": "var",
                "n_out": 10,
                "direction": -1,
                "from": ["lstm0_fw", "lstm0_bw"],
            },
            "encoder": {"class": "copy", "from": ["lstm1_fw", "lstm1_bw"]},
            "enc_ctx": {"class": "linear", "activation": None, "with_bias": True, "from": ["encoder"], "n_out": 10},
            "fertility": {
                "class": "linear",
                "activation": "sigmoid",
                "with_bias": False,
                "from": ["encoder"],
                "n_out": 1,
            },
            "output": {
                "class": "rec",
                "from": [],
                "unit": {
                    "output": {
                        "class": "choice",
                        "target": "classes",
                        "beam_size": beam_size,
                        "from": ["output_prob"],
                        "initial_output": 0,
                    },
                    "end": {"class": "compare", "from": ["output"], "value": 0},
                    "target_embed": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["output"],
                        "n_out": 6,
                        "initial_output": "apply(0)",
                        "L2": l2_target_embed,
                    },
                    "weight_feedback": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["prev:accum_att_weights"],
                        "n_out": 10,
                    },
                    "prev_s_state": {"class": "get_last_hidden_state", "from": ["prev:s"], "n_out": 20},
                    "prev_s_transformed": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["prev_s_state"],
                        "n_out": 10,
                    },
                    "energy_in": {
                        "class": "combine",
                        "kind": "add",
                        "from": ["base:enc_ctx", "weight_feedback", "prev_s_transformed"],
                        "n_out": 10,
                    },
                    "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
                    "energy": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["energy_tanh"],
                        "n_out": 1,
                    },
                    "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, enc-T, 1)
                    "accum_att_weights": {
                        "class": "eval",
                        "from": ["prev:accum_att_weights", "att_weights", "base:fertility"],
                        "eval": "source(0) + source(1) / (2.0 * source(2))",
                        "out_type": {"dim": 1, "shape": (None, 1)},
                    },
                    "att": {
                        "class": "generic_attention",
                        "weights": "att_weights",
                        "base": "base:encoder",
                        "auto_squeeze": True,
                    },
                    "s": {
                        "class": "rnn_cell",
                        "unit": "standardlstm",
                        "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                        "initial_state": "var",
                        "from": ["target_embed", "att"],
                        "n_out": 10,
                    },
                    "readout_in": {
                        "class": "linear",
                        "from": ["prev:s", "prev:target_embed", "att"],
                        "activation": None,
                        "n_out": 10,
                        "L2": l2_readout_in,
                    },
                    "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
                    "output_prob": {"class": "softmax", "from": ["readout"], "target": "classes", "loss": "ce"},
                },
                "target": "classes",
                "max_seq_len": 20,
            },
            "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance", "target": "classes"},
        }

    print("Constructing train network without constraints")
    tf_compat.v1.reset_default_graph()
    extern_data = ExternData(
        {
            "data": {"dim": n_src_dim, "sparse": True},
            "classes": {"dim": n_tgt_dim, "sparse": True, "available_for_inference": False},
        }
    )
    train_net = TFNetwork(
        extern_data=extern_data, search_flag=False, train_flag=get_global_train_flag_placeholder(), config=config
    )
    assert train_net.eval_flag is True
    train_net.construct_from_dict(get_net_dict(l2_target_embed=0.0, l2_readout_in=0.0))
    train_out_layer = train_net.layers["output"]
    assert isinstance(train_out_layer, RecLayer)
    assert isinstance(train_out_layer.cell, _SubnetworkRecCell)
    assert_equal(set(train_out_layer.cell.input_layers_moved_out), {"output", "target_embed"})
    assert_equal(set(train_out_layer.cell.output_layers_moved_out), {"output_prob", "readout_in", "readout"})
    assert_equal(train_net.get_total_constraints(), 0)

    print("Constructing train network with L2 norm on moved out input layer")
    tf_compat.v1.reset_default_graph()
    extern_data = ExternData(
        {
            "data": {"dim": n_src_dim, "sparse": True},
            "classes": {"dim": n_tgt_dim, "sparse": True, "available_for_inference": False},
        }
    )
    train_net = TFNetwork(
        extern_data=extern_data, search_flag=False, train_flag=get_global_train_flag_placeholder(), config=config
    )
    assert train_net.eval_flag is True
    train_net.construct_from_dict(get_net_dict(l2_target_embed=0.01, l2_readout_in=0.0))
    train_out_layer = train_net.layers["output"]
    assert isinstance(train_out_layer, RecLayer)
    assert isinstance(train_out_layer.cell, _SubnetworkRecCell)
    assert_equal(set(train_out_layer.cell.input_layers_moved_out), {"output", "target_embed"})
    assert_equal(set(train_out_layer.cell.output_layers_moved_out), {"output_prob", "readout_in", "readout"})
    assert_not_equal(train_net.get_total_constraints(), 0)

    print("Constructing train network with L2 norm on moved out output layer")
    tf_compat.v1.reset_default_graph()
    extern_data = ExternData(
        {
            "data": {"dim": n_src_dim, "sparse": True},
            "classes": {"dim": n_tgt_dim, "sparse": True, "available_for_inference": False},
        }
    )
    train_net = TFNetwork(
        extern_data=extern_data, search_flag=False, train_flag=get_global_train_flag_placeholder(), config=config
    )
    assert train_net.eval_flag is True
    train_net.construct_from_dict(get_net_dict(l2_target_embed=0.0, l2_readout_in=0.01))
    train_out_layer = train_net.layers["output"]
    assert isinstance(train_out_layer, RecLayer)
    assert isinstance(train_out_layer.cell, _SubnetworkRecCell)
    assert_equal(set(train_out_layer.cell.input_layers_moved_out), {"output", "target_embed"})
    assert_equal(set(train_out_layer.cell.output_layers_moved_out), {"output_prob", "readout_in", "readout"})
    assert_not_equal(train_net.get_total_constraints(), 0)


def test_rec_layer_move_out_of_loop_ref_att_generic_att():
    """
    This will move out :class:`GenericAttentionLayer` (and basically everything)
    because we provide some reference att weights.
    """
    from returnn.tf.layers.rec import _SubnetworkRecCell
    from returnn.tf.util.basic import get_global_train_flag_placeholder, stop_event_writer_thread

    n_src_dim = 5
    n_tgt_dim = 7
    beam_size = 12
    EncKeyTotalDim = 8
    AttNumHeads = 1
    EncValueTotalDim = 8
    EncValuePerHeadDim = EncValueTotalDim // AttNumHeads
    config = Config()
    config.update({"debug_print_layer_output_template": True})
    net_dict = {
        "ref_att_weights": {
            "class": "unflatten_nd",
            "from": "data:att_weights",
            "sizes": "data:att_weights_sizes",
            "num_axes": 2,
            "declare_same_sizes_as": {0: "data:classes", 1: "data"},
        },
        "source_embed": {"class": "linear", "activation": None, "with_bias": False, "n_out": 6, "from": "data"},
        "encoder": {"class": "linear", "from": "source_embed", "activation": "tanh", "n_out": EncValueTotalDim},
        "enc_ctx": {
            "class": "linear",
            "activation": None,
            "with_bias": True,
            "from": "encoder",
            "n_out": EncKeyTotalDim,
        },
        "enc_value": {
            "class": "split_dims",
            "axis": "F",
            "dims": (AttNumHeads, EncValuePerHeadDim),
            "from": "encoder",
        },  # (B, enc-T, H, D'/H)
        "inv_fertility": {
            "class": "linear",
            "activation": "sigmoid",
            "with_bias": False,
            "from": "encoder",
            "n_out": AttNumHeads,
        },
        "output": {
            "class": "rec",
            "from": ["ref_att_weights"],
            "unit": {
                "output": {
                    "class": "choice",
                    "target": "classes",
                    "beam_size": beam_size,
                    "from": ["output_prob"],
                    "initial_output": 0,
                },
                "end": {"class": "compare", "from": ["output"], "value": 0},
                "target_embed": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["output"],
                    "n_out": 6,
                    "initial_output": 0,
                },
                "weight_feedback": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["prev:accum_att_weights"],
                    "n_out": EncKeyTotalDim,
                },
                "prev_s_transformed": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["prev:s"],
                    "n_out": EncKeyTotalDim,
                },
                "energy_in": {
                    "class": "combine",
                    "kind": "add",
                    "from": ["base:enc_ctx", "weight_feedback", "prev_s_transformed"],
                    "n_out": EncKeyTotalDim,
                },
                "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
                "energy": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["energy_tanh"],
                    "n_out": AttNumHeads,
                    "is_output_layer": True,
                },
                "att_weights": {"class": "copy", "from": "data:source"},
                # "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, enc-T, H)
                "att_weights_with_fertility": {
                    "class": "eval",
                    "from": ["att_weights", "base:inv_fertility"],
                    "eval": "source(0) * source(1) * 0.5",
                },
                "accum_att_weights": {"class": "cumsum", "from": "att_weights_with_fertility"},
                "att0": {"class": "generic_attention", "weights": "att_weights", "base": "base:enc_value"},  # (B, H, V)
                "att": {
                    "class": "merge_dims",
                    "axes": ["dim:%i" % AttNumHeads, "dim:%i" % EncValuePerHeadDim],
                    "from": "att0",
                },  # (B, H*V)
                "s": {"class": "rnn_cell", "unit": "standardlstm", "from": ["target_embed", "att"], "n_out": 10},
                "readout_in": {
                    "class": "linear",
                    "from": ["prev:s", "prev:target_embed", "att"],
                    "activation": None,
                    "n_out": 10,
                },
                "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
                "output_prob": {"class": "softmax", "from": ["readout"], "target": "classes", "loss": "ce"},
            },
            "target": "classes",
            "max_seq_len": 20,
        },
        "att_distill_loss": {
            "class": "eval",
            "from": ["output/energy", "ref_att_weights"],
            "out_type": (lambda sources, **kwargs: sources[0].output.copy_template_excluding_spatial_dim(-1)),
            "eval": "softmax_cross_entropy_over_size("
            + "logits=source(0, as_data=True, auto_convert=False),"
            + "labels=source(1, as_data=True, auto_convert=False))",
            "loss": "as_is",
        },
    }

    def train(net, session):
        """
        :param TFNetwork net:
        :param tf.compat.v1.Session session:
        """
        from returnn.datasets.generating import StaticDataset
        from returnn.tf.data_pipeline import FeedDictDataProvider
        from returnn.engine.batch import Batch, BatchSetGenerator
        from returnn.util.basic import dict_joined, softmax

        rnd = numpy.random.RandomState(42)

        def create_rnd_flat_att_weights(dec_t, enc_t):
            w = rnd.normal(size=(dec_t, enc_t, AttNumHeads)).astype("float32")
            w = softmax(w, axis=1)
            w = w.reshape((dec_t * enc_t, AttNumHeads))
            return w

        dataset = StaticDataset(
            data=[
                {
                    "data": numpy.array([2, 4, 1, 0]),
                    "classes": numpy.array([3, 6, 0]),
                    "att_weights": create_rnd_flat_att_weights(3, 4),
                    "att_weights_sizes": numpy.array([3, 4]),
                },
                {
                    "data": numpy.array([2, 4, 1, 3, 0]),
                    "classes": numpy.array([3, 6, 2, 1, 4, 5, 0]),
                    "att_weights": create_rnd_flat_att_weights(7, 5),
                    "att_weights_sizes": numpy.array([7, 5]),
                },
            ],
            output_dim={
                "data": [n_src_dim, 1],
                "classes": [n_tgt_dim, 1],
                "att_weights": [AttNumHeads, 2],
                "att_weights_sizes": [1, 1],
            },
        )
        dataset.init_seq_order(epoch=1)
        batch = Batch()
        batch.add_sequence_as_slice(seq_idx=0, seq_start_frame=0, length=dataset.get_seq_length(0))
        batch.add_sequence_as_slice(seq_idx=1, seq_start_frame=0, length=dataset.get_seq_length(1))
        print("batch:", batch, "num frames:", batch.get_total_num_frames())
        print("batch dims:", batch.max_num_frames_per_slice * batch.num_slices)
        batch_generator = iter([batch])
        batches = BatchSetGenerator(dataset, generator=batch_generator)
        out_layer = net.layers["output"]
        assert isinstance(out_layer, RecLayer)
        assert isinstance(out_layer.cell, _SubnetworkRecCell)

        net.initialize_params(session)
        data_provider = FeedDictDataProvider(
            tf_session=session,
            extern_data=extern_data,
            data_keys=["data", "classes", "att_weights", "att_weights_sizes"],
            dataset=dataset,
            batches=batches,
        )
        feed_dict, meta_step_info = data_provider.get_feed_dict(single_threaded=True)
        if isinstance(net.train_flag, tf.Tensor):
            feed_dict[net.train_flag] = True
        try:
            print("session run for seq lens output:")
            out = session.run(
                dict_joined(
                    {"data:%s:seq_len" % k: v.get_sequence_lengths() for (k, v) in net.extern_data.data.items()},
                    {"layer:%s:out_seq_len" % k: l.output.get_sequence_lengths() for (k, l) in net.layers.items()},
                    {
                        "rec_layer_in:%s:out_seq_len" % k: l.output.get_sequence_lengths()
                        for (k, l) in out_layer.cell.input_layers_net.layers.items()
                    }
                    if out_layer.cell.input_layers_net
                    else {},
                    {
                        "rec_layer_out:%s:out_seq_len" % k: l.output.get_sequence_lengths()
                        for (k, l) in out_layer.cell.output_layers_net.layers.items()
                    }
                    if out_layer.cell.output_layers_net
                    else {},
                ),
                feed_dict=feed_dict,
            )
            pprint(out)
            print("session run for objective output:")
            losses, total_loss, total_constraints = net.get_losses_initialized(with_total=True)
            # TODO: strange ref att weights?
            out = session.run(
                {
                    "total_loss": total_loss,
                    "total_constraints": tf.convert_to_tensor(total_constraints),
                    "losses": {name: loss.get_loss_value() for name, loss in losses.items()},
                    "att_distill_loss_in": [s.output.placeholder for s in net.layers["att_distill_loss"].sources],
                    "att_distill_loss_out": net.layers["att_distill_loss"].output.placeholder,
                },
                feed_dict=feed_dict,
            )
            pprint(out)
        except Exception as exc:
            print("Exception happened:", str(exc).splitlines()[0])
            print("Writing TF log file.")
            writer = tf_compat.v1.summary.FileWriter(".", filename_suffix="test_rec_layer_move_out_of_loop")
            writer.add_graph(session.graph)
            writer.close()
            stop_event_writer_thread(writer)
            raise

    print("Constructing train network.")
    with make_scope() as session:
        extern_data = ExternData(
            {
                "data": {"dim": n_src_dim, "sparse": True},
                "classes": {"dim": n_tgt_dim, "sparse": True, "available_for_inference": False},
                "att_weights": {"shape": (None, AttNumHeads), "available_for_inference": False},
                "att_weights_sizes": {"shape": (None,), "dtype": "int32", "available_for_inference": False},
            }
        )
        train_net = TFNetwork(
            extern_data=extern_data, search_flag=False, train_flag=get_global_train_flag_placeholder(), config=config
        )
        assert train_net.eval_flag is True
        train_net.construct_from_dict(net_dict)
        train_out_layer = train_net.layers["output"]
        assert isinstance(train_out_layer, RecLayer)
        assert isinstance(train_out_layer.cell, _SubnetworkRecCell)
        assert_equal(train_out_layer.cell.layers_in_loop, [])  # all moved out :)
        rec_subnet = train_out_layer.cell.output_layers_net
        assert isinstance(rec_subnet, TFNetwork)
        att_layer = rec_subnet.layers["att"]
        assert att_layer.output.shape == (None, EncValueTotalDim) and att_layer.output.time_dim_axis is not None
        energy_in_layer = rec_subnet.layers["energy_in"]
        assert energy_in_layer.output.shape == (None, None, EncKeyTotalDim)
        train(train_net, session)


def test_same_spatial_dim_after_rec_layers():
    with make_scope() as session:
        config = Config({"debug_print_layer_output_template": True})
        extern_data = ExternData(
            {
                "data": {"dim": 13, "sparse": True},
                "classes": {"dim": 17, "sparse": True, "available_for_inference": False},
            }
        )
        net = TFNetwork(extern_data=extern_data, train_flag=True, config=config)
        net.construct_from_dict(
            {
                "source_embed": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "n_out": 6,
                    "from": "data:data",
                },
                "lstm0_fw": {
                    "class": "rec",
                    "unit": "standardlstm",
                    "n_out": 10,
                    "direction": 1,
                    "from": ["source_embed"],
                },
                "lstm0_bw": {
                    "class": "rec",
                    "unit": "standardlstm",
                    "n_out": 10,
                    "direction": -1,
                    "from": ["source_embed"],
                },
                "lstm1_fw": {
                    "class": "rec",
                    "unit": "standardlstm",
                    "n_out": 10,
                    "direction": 1,
                    "from": ["lstm0_fw", "lstm0_bw"],
                },
                "lstm1_bw": {
                    "class": "rec",
                    "unit": "standardlstm",
                    "n_out": 10,
                    "direction": -1,
                    "from": ["lstm0_fw", "lstm0_bw"],
                },
                "encoder": {"class": "copy", "from": ["lstm1_fw", "lstm1_bw"]},
                "enc_value": {"class": "split_dims", "axis": "F", "dims": (4, 5), "from": ["encoder"]},
                "output": {"class": "copy", "from": ["enc_value"]},
            }
        )
        size = extern_data.data["data"].get_size_dim_tag(0)
        print("data size:", size)
        for name in ["source_embed", "lstm0_fw", "lstm1_fw", "encoder", "enc_value", "output"]:
            layer = net.layers[name]
            layer_size = layer.output.get_size_dim_tag(0)
            print("layer:", layer, "size:", layer_size)
            assert size == layer_size, "no match for layer %r" % layer
        print("All good.")


def test_rec_layer_rnn_train_and_search():
    from returnn.tf.layers.rec import _SubnetworkRecCell

    n_src_dim = 5
    n_tgt_dim = 7
    beam_size = 3
    config = Config()
    config.update({"debug_print_layer_output_template": True, "debug_print_layer_output_shape": True})
    EncKeyTotalDim = 14
    AttNumHeads = 1
    EncValueTotalDim = 14
    EncValuePerHeadDim = EncValueTotalDim // AttNumHeads
    LstmDim = EncValueTotalDim // 2
    target = "classes"

    net_dict = {
        "lstm0_fw": {"class": "rec", "unit": "nativelstm2", "n_out": LstmDim, "direction": 1, "from": ["data"]},
        "lstm0_bw": {"class": "rec", "unit": "nativelstm2", "n_out": LstmDim, "direction": -1, "from": ["data"]},
        "lstm0_pool": {
            "class": "pool",
            "mode": "max",
            "padding": "same",
            "pool_size": (3,),
            "from": ["lstm0_fw", "lstm0_bw"],
            "trainable": False,
        },
        "lstm1_fw": {"class": "rec", "unit": "nativelstm2", "n_out": LstmDim, "direction": 1, "from": ["lstm0_pool"]},
        "lstm1_bw": {"class": "rec", "unit": "nativelstm2", "n_out": LstmDim, "direction": -1, "from": ["lstm0_pool"]},
        "encoder": {"class": "copy", "from": ["lstm1_fw", "lstm1_bw"]},  # dim: EncValueTotalDim
        "enc_ctx": {
            "class": "linear",
            "activation": None,
            "with_bias": True,
            "from": ["encoder"],
            "n_out": EncKeyTotalDim,
        },
        "inv_fertility": {
            "class": "linear",
            "activation": "sigmoid",
            "with_bias": False,
            "from": ["encoder"],
            "n_out": AttNumHeads,
        },
        "enc_value": {
            "class": "split_dims",
            "axis": "F",
            "dims": (AttNumHeads, EncValuePerHeadDim),
            "from": ["encoder"],
        },  # (B, enc-T, H, D'/H)
        "output": {
            "class": "rec",
            "from": [],
            "cheating": config.bool("cheating", False),
            "unit": {
                "output": {
                    "class": "choice",
                    "target": target,
                    "beam_size": beam_size,
                    "cheating": config.bool("cheating", False),
                    "from": ["output_prob"],
                    "initial_output": 0,
                },
                "end": {"class": "compare", "from": ["output"], "value": 0},
                "target_embed": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["output"],
                    "n_out": 10,
                    "initial_output": 0,
                },  # feedback_input
                "weight_feedback": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["prev:accum_att_weights"],
                    "n_out": EncKeyTotalDim,
                },
                "s_transformed": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["s"],
                    "n_out": EncKeyTotalDim,
                },
                "energy_in": {
                    "class": "combine",
                    "kind": "add",
                    "from": ["base:enc_ctx", "weight_feedback", "s_transformed"],
                    "n_out": EncKeyTotalDim,
                },
                "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
                "energy": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["energy_tanh"],
                    "n_out": AttNumHeads,
                },  # (B, enc-T, H)
                "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, enc-T, H)
                "accum_att_weights": {
                    "class": "eval",
                    "from": ["prev:accum_att_weights", "att_weights", "base:inv_fertility"],
                    "eval": "source(0) + source(1) * source(2) * 0.5",
                    "out_type": {"dim": AttNumHeads, "shape": (None, AttNumHeads)},
                },
                "att0": {"class": "generic_attention", "weights": "att_weights", "base": "base:enc_value"},  # (B, H, V)
                "att": {
                    "class": "merge_dims",
                    "axes": ["dim:%i" % AttNumHeads, "dim:%i" % EncValuePerHeadDim],
                    "from": "att0",
                },  # (B, H*V)
                "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["prev:target_embed", "prev:att"], "n_out": 10},
                "readout_in": {
                    "class": "linear",
                    "from": ["s", "prev:target_embed", "att"],
                    "activation": None,
                    "n_out": 10,
                },
                "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
                "output_prob": {
                    "class": "softmax",
                    "from": ["readout"],
                    "dropout": 0.3,
                    "target": target,
                    "loss": "ce",
                    "loss_opts": {"label_smoothing": 0.1},
                },
            },
            "target": target,
            "max_seq_len": "max_len_from('base:encoder')",
        },
        "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance", "target": target},
    }

    def run(train_flag=False, search_flag=False):
        """
        :param bool train_flag:
        :param bool search_flag:
        """
        print("Create network with train_flag=%r, search_flag=%r." % (train_flag, search_flag))

        from returnn.datasets.generating import StaticDataset
        from returnn.tf.data_pipeline import FeedDictDataProvider
        from returnn.engine.batch import Batch, BatchSetGenerator
        from returnn.util.basic import dict_joined

        dataset = StaticDataset(
            data=[
                {
                    "data": numpy.random.normal(size=(11, n_src_dim)).astype("float32"),
                    "classes": numpy.array([3, 6, 0]),
                },
                {
                    "data": numpy.random.normal(size=(13, n_src_dim)).astype("float32"),
                    "classes": numpy.array([3, 6, 2, 1, 4, 5, 0]),
                },
            ],
            output_dim={"data": [n_src_dim, 2], "classes": [n_tgt_dim, 1]},
        )
        dataset.init_seq_order(epoch=1)
        batch = Batch()
        batch.add_sequence_as_slice(seq_idx=0, seq_start_frame=0, length=dataset.get_seq_length(0))
        batch.add_sequence_as_slice(seq_idx=1, seq_start_frame=0, length=dataset.get_seq_length(1))
        print("batch:", batch, "num frames:", batch.get_total_num_frames())
        print("batch dims:", batch.max_num_frames_per_slice * batch.num_slices)
        batch_generator = iter([batch])
        batches = BatchSetGenerator(dataset, generator=batch_generator)

        with make_scope() as session:
            extern_data = ExternData(
                {
                    "data": {"dim": n_src_dim},
                    "classes": {"dim": n_tgt_dim, "sparse": True, "available_for_inference": False},
                }
            )

            net = TFNetwork(extern_data=extern_data, search_flag=search_flag, train_flag=train_flag, config=config)
            net.construct_from_dict(net_dict)

            out_layer = net.layers["output"]
            assert isinstance(out_layer, RecLayer)
            assert isinstance(out_layer.cell, _SubnetworkRecCell)
            net.initialize_params(session)
            data_provider = FeedDictDataProvider(
                tf_session=session,
                extern_data=extern_data,
                data_keys=["data", "classes"] if train_flag else ["data"],
                dataset=dataset,
                batches=batches,
            )
            feed_dict, meta_step_info = data_provider.get_feed_dict(single_threaded=True)
            if isinstance(net.train_flag, tf.Tensor):
                feed_dict[net.train_flag] = train_flag
            try:
                print("Output:")
                out = session.run(
                    dict_joined(
                        {"data:%s:seq_len" % k: v.get_sequence_lengths() for (k, v) in net.extern_data.data.items()},
                        {"layer:%s:out_seq_len" % k: l.output.get_sequence_lengths() for (k, l) in net.layers.items()},
                        {
                            "rec_layer_in:%s:out_seq_len" % k: l.output.get_sequence_lengths()
                            for (k, l) in out_layer.cell.input_layers_net.layers.items()
                        }
                        if out_layer.cell.input_layers_net
                        else {},
                        {
                            "rec_layer_out:%s:out_seq_len" % k: l.output.get_sequence_lengths()
                            for (k, l) in out_layer.cell.output_layers_net.layers.items()
                        }
                        if out_layer.cell.output_layers_net
                        else {},
                        {"output": out_layer.output.placeholder},
                        {"objective": tf.convert_to_tensor(net.get_objective())} if train_flag else {},
                    )
                    if train_flag
                    else {"output": out_layer.output.placeholder},
                    feed_dict=feed_dict,
                )
                pprint(out)
            except Exception as exc:
                print()
                print("EXCEPTION " + "-" * 60)
                print("Exception happened:", str(exc).splitlines()[0])
                print()
                out_layer.cell._handle_construct_exception("exc", exc)
                print()
                print("TF debug info:")
                help_on_tf_exception(
                    session=session,
                    exception=exc,
                    fetches=out_layer.output.placeholder,
                    feed_dict=feed_dict,
                    meta_step_info=meta_step_info,
                    extern_data=data_provider.extern_data,
                )
                raise

    run(train_flag=True)
    run(search_flag=True)


def test_rec_layer_local_att_train_and_search():
    # https://github.com/rwth-i6/returnn-experiments/blob/master/2019-asr-local-attention/librispeech/local-heuristic.argmax.win05.exp3.ctc.config
    # Note the small fix in p_t_in.
    from returnn.tf.layers.rec import _SubnetworkRecCell

    n_src_dim = 5
    n_tgt_dim = 7
    beam_size = 3
    config = Config()
    config.update({"debug_print_layer_output_template": True, "debug_print_layer_output_shape": True})
    EncKeyTotalDim = 14
    AttWindowSize = 5
    AttNumHeads = 1
    EncValueTotalDim = 14
    EncValuePerHeadDim = EncValueTotalDim // AttNumHeads
    LstmDim = EncValueTotalDim // 2
    target = "classes"

    net_dict = {
        # "lstm0_fw": {"class": "rec", "unit": "nativelstm2", "n_out": LstmDim, "direction": 1, "from": ["data"]},
        # "lstm0_bw": {"class": "rec", "unit": "nativelstm2", "n_out": LstmDim, "direction": -1, "from": ["data"]},
        # "lstm0_pool": {"class": "pool", "mode": "max", "padding": "same", "pool_size": (3,),
        #               "from": ["lstm0_fw", "lstm0_bw"], "trainable": False},
        # "lstm1_fw": {"class": "rec", "unit": "nativelstm2", "n_out": LstmDim, "direction": 1, "from": ["lstm0_pool"]},
        # "lstm1_bw": {"class": "rec", "unit": "nativelstm2", "n_out": LstmDim, "direction": -1, "from": ["lstm0_pool"]},
        # "encoder": {"class": "copy", "from": ["lstm1_fw", "lstm1_bw"]},  # dim: EncValueTotalDim
        "lstm0_pool": {"class": "pool", "mode": "max", "padding": "same", "pool_size": (3,), "from": "data:data"},
        "encoder": {"class": "rec", "unit": "nativelstm2", "from": "lstm0_pool", "n_out": EncValueTotalDim},
        "enc_ctx": {
            "class": "linear",
            "activation": None,
            "with_bias": True,
            "from": ["encoder"],
            "n_out": EncKeyTotalDim,
        },
        "inv_fertility": {
            "class": "linear",
            "activation": "sigmoid",
            "with_bias": False,
            "from": ["encoder"],
            "n_out": AttNumHeads,
        },
        "enc_value": {
            "class": "split_dims",
            "axis": "F",
            "dims": (AttNumHeads, EncValuePerHeadDim),
            "from": ["encoder"],
        },  # (B, enc-T, H, D'/H)
        "output": {
            "class": "rec",
            "from": [],
            "cheating": config.bool("cheating", False),
            "unit": {
                "output": {
                    "class": "choice",
                    "target": target,
                    "beam_size": beam_size,
                    "cheating": config.bool("cheating", False),
                    "from": ["output_prob"],
                    "initial_output": 0,
                },
                "end": {"class": "compare", "from": ["output"], "value": 0},
                "target_embed": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["output"],
                    "n_out": 21,
                    "initial_output": 0,
                },  # feedback_input
                "weight_feedback": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["prev:accum_att_weights"],
                    "n_out": EncKeyTotalDim,
                },
                "s_transformed": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["s"],
                    "n_out": EncKeyTotalDim,
                },
                # (T, B)
                # "p_t": {"class": "eval", "from": "p_t_in", "eval": "tf.to_float(source(0))", "out_type": {"dtype": "float32"}},
                # "p_t_in": {"class": "eval", "from": "prev:att_weights",
                #           "eval": "tf.squeeze(tf.argmax(source(0, auto_convert=False), axis=1, output_type=tf.int32), axis=1)",
                #           "out_type": {"shape": (), "batch_dim_axis": 0, "dtype": "int32"}},
                "p_t_in": {"class": "reduce", "from": "prev:att_weights", "mode": "argmax", "axis": "t"},
                # "p_t_print": {"class": "eval", "from": "p_t_in", "eval": "tf.Print(source(0), [tf.shape(source(0)),source(0)], \"p_t_in\", summarize=200)"},
                # "p_t": {"class": "eval", "from": "p_t_in", "eval": "tf.maximum(0., source(0)))" % (AttWindowSize // 2),
                # "out_type": {"sparse": False, "shape": (), "dtype": "float32"}, "initial_output": 0},
                # "energy_in_enc_ctx": {"class": "slice_nd", "from": ["base:enc_ctx"], "start": "p_t", "size": AttWindowSize},  # (B, size, 1000)
                "energy_in": {
                    "class": "combine",
                    "kind": "add",
                    "from": ["base:enc_ctx", "weight_feedback", "s_transformed"],
                    "n_out": EncKeyTotalDim,
                },
                "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
                "energy": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["energy_tanh"],
                    "n_out": AttNumHeads,
                },  # (B, enc-T, H)
                "energy_reinterpreted": {
                    "class": "reinterpret_data",
                    "enforce_batch_major": True,
                    "from": "energy",
                    "trainable": False,
                },
                "att_weights": {
                    "class": "softmax_over_spatial",
                    "from": ["energy_reinterpreted"],
                    "window_start": "p_t_in",
                    "window_size": AttWindowSize,
                },  # (B, enc-T, H)
                # "att_weights_print": {"class": "eval", "from": "att_weights", "eval": "tf.Print(source(0), [tf.shape(source(0)), source(0)], summarize=99)"},
                # "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, enc-T, H)
                # (B, T, H) + (B, T, H)
                "accum_att_weights": {
                    "class": "eval",
                    "from": ["prev:accum_att_weights", "att_weights", "base:inv_fertility"],
                    "eval": "source(0) + source(1) * source(2) * 0.5",
                    "out_type": {"dim": AttNumHeads, "shape": (None, AttNumHeads)},
                },
                "att0": {"class": "generic_attention", "weights": "att_weights", "base": "base:enc_value"},  # (B, H, V)
                "att": {
                    "class": "merge_dims",
                    "axes": ["dim:%i" % AttNumHeads, "dim:%i" % EncValuePerHeadDim],
                    "from": "att0",
                },  # (B, H*V)
                "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["prev:target_embed", "prev:att"], "n_out": 10},
                # transform
                "readout_in": {
                    "class": "linear",
                    "from": ["s", "prev:target_embed", "att"],
                    "activation": None,
                    "n_out": 20,
                },
                # merge + post_merge bias
                "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
                "output_prob": {
                    "class": "softmax",
                    "from": ["readout"],
                    "dropout": 0.3,
                    "target": target,
                    "loss": "ce",
                    "loss_opts": {"label_smoothing": 0.1},
                    "loss_only_on_non_search": True,
                },
            },
            "target": target,
            "max_seq_len": "max_len_from('base:encoder')",
        },
        "decision": {
            "class": "decide",
            "from": ["output"],
            "loss": "edit_distance",
            "target": target,
            "loss_only_on_non_search": False,
        },
        "ctc": {
            "class": "softmax",
            "from": ["encoder"],
            "loss": "ctc",
            "target": target,
            "loss_opts": {"beam_width": 1, "ctc_opts": {"ignore_longer_outputs_than_inputs": True}},
        },
    }

    def run(train_flag=False, search_flag=False):
        """
        :param bool train_flag:
        :param bool search_flag:
        """
        print("Create network with train_flag=%r, search_flag=%r." % (train_flag, search_flag))

        from returnn.datasets.generating import StaticDataset
        from returnn.tf.data_pipeline import FeedDictDataProvider
        from returnn.engine.batch import Batch, BatchSetGenerator
        from returnn.util.basic import dict_joined

        dataset = StaticDataset(
            data=[
                {
                    "data": numpy.random.normal(size=(11, n_src_dim)).astype("float32"),
                    "classes": numpy.array([3, 6, 0]),
                },
                {
                    "data": numpy.random.normal(size=(13, n_src_dim)).astype("float32"),
                    "classes": numpy.array([3, 6, 2, 1, 4, 5, 0]),
                },
            ],
            output_dim={"data": [n_src_dim, 2], "classes": [n_tgt_dim, 1]},
        )
        dataset.init_seq_order(epoch=1)
        batch = Batch()
        batch.add_sequence_as_slice(seq_idx=0, seq_start_frame=0, length=dataset.get_seq_length(0))
        batch.add_sequence_as_slice(seq_idx=1, seq_start_frame=0, length=dataset.get_seq_length(1))
        print("batch:", batch, "num frames:", batch.get_total_num_frames())
        print("batch dims:", batch.max_num_frames_per_slice * batch.num_slices)
        batch_generator = iter([batch])
        batches = BatchSetGenerator(dataset, generator=batch_generator)

        with make_scope() as session:
            extern_data = ExternData(
                {
                    "data": {"dim": n_src_dim},
                    "classes": {"dim": n_tgt_dim, "sparse": True, "available_for_inference": False},
                }
            )

            net = TFNetwork(extern_data=extern_data, search_flag=search_flag, train_flag=train_flag, config=config)
            net.construct_from_dict(net_dict)

            out_layer = net.layers["output"]
            assert isinstance(out_layer, RecLayer)
            assert isinstance(out_layer.cell, _SubnetworkRecCell)
            net.initialize_params(session)
            data_provider = FeedDictDataProvider(
                tf_session=session,
                extern_data=extern_data,
                data_keys=["data", "classes"] if train_flag else ["data"],
                dataset=dataset,
                batches=batches,
            )
            feed_dict, meta_step_info = data_provider.get_feed_dict(single_threaded=True)
            if isinstance(net.train_flag, tf.Tensor):
                feed_dict[net.train_flag] = train_flag
            fetches = (
                dict_joined(
                    {"data:%s:seq_len" % k: v.get_sequence_lengths() for (k, v) in net.extern_data.data.items()},
                    {"layer:%s:out_seq_len" % k: l.output.get_sequence_lengths() for (k, l) in net.layers.items()},
                    {
                        "rec_layer_in:%s:out_seq_len" % k: l.output.get_sequence_lengths()
                        for (k, l) in out_layer.cell.input_layers_net.layers.items()
                    }
                    if out_layer.cell.input_layers_net
                    else {},
                    {
                        "rec_layer_out:%s:out_seq_len" % k: l.output.get_sequence_lengths()
                        for (k, l) in out_layer.cell.output_layers_net.layers.items()
                    }
                    if out_layer.cell.output_layers_net
                    else {},
                    {"output": out_layer.output.placeholder},
                    {"objective": tf.convert_to_tensor(net.get_objective())} if train_flag else {},
                )
                if train_flag
                else {"output": out_layer.output.placeholder}
            )
            try:
                print("Output:")
                out = session.run(fetches, feed_dict=feed_dict)
                pprint(out)
            except Exception as exc:
                print()
                print("EXCEPTION " + "-" * 60)
                print("Exception happened:", type(exc), str(exc).splitlines()[0])
                print()
                out_layer.cell._handle_construct_exception("exc", exc)
                print()
                print("TF debug info:")
                help_on_tf_exception(
                    session=session,
                    exception=exc,
                    fetches=fetches,
                    feed_dict=feed_dict,
                    meta_step_info=meta_step_info,
                    extern_data=data_provider.extern_data,
                )
                raise

    run(train_flag=True)
    run(search_flag=True)


def test_same_spatial_dim_after_rec_layers_with_pool():
    with make_scope() as session:
        config = Config({"debug_print_layer_output_template": True})
        extern_data = ExternData(
            {
                "data": {"dim": 13, "sparse": True},
                "classes": {"dim": 17, "sparse": True, "available_for_inference": False},
                "att_weights": {"shape": (None, 1), "available_for_inference": False},
                "att_weights_sizes": {"shape": (None,), "dtype": "int32", "available_for_inference": False},
            }
        )
        net = TFNetwork(extern_data=extern_data, train_flag=True, config=config)
        net.construct_from_dict(
            {
                "ref_att_weights": {
                    "class": "unflatten_nd",
                    "from": "data:att_weights",
                    "sizes": "data:att_weights_sizes",
                    "num_axes": 2,
                    "declare_same_sizes_as": {0: "data:classes", 1: "encoder"},
                    "is_output_layer": True,
                },
                "source_embed": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "n_out": 6,
                    "from": "data:data",
                },
                "lstm0_fw": {
                    "class": "rec",
                    "unit": "standardlstm",
                    "n_out": 10,
                    "direction": 1,
                    "from": ["source_embed"],
                },
                "lstm0_bw": {
                    "class": "rec",
                    "unit": "standardlstm",
                    "n_out": 10,
                    "direction": -1,
                    "from": ["source_embed"],
                },
                "lstm0_pool": {
                    "class": "pool",
                    "mode": "max",
                    "padding": "same",
                    "pool_size": (2,),
                    "from": ["lstm0_fw", "lstm0_bw"],
                },
                "lstm1_fw": {
                    "class": "rec",
                    "unit": "standardlstm",
                    "n_out": 10,
                    "direction": 1,
                    "from": ["lstm0_pool"],
                },
                "lstm1_bw": {
                    "class": "rec",
                    "unit": "standardlstm",
                    "n_out": 10,
                    "direction": -1,
                    "from": ["lstm0_pool"],
                },
                "encoder": {"class": "copy", "from": ["lstm1_fw", "lstm1_bw"]},
                "enc_value": {"class": "split_dims", "axis": "F", "dims": (4, 5), "from": ["encoder"]},
                "output": {"class": "copy", "from": ["enc_value"]},
            }
        )
        size_enc0 = extern_data.data["data"].get_size_dim_tag(0)
        print("data size:", size_enc0)
        size_enc1 = net.layers["encoder"].output.get_size_dim_tag(0)
        print("encoder size:", size_enc1)
        assert size_enc0 != size_enc1
        for name in ["source_embed", "lstm0_fw"]:
            layer = net.layers[name]
            layer_size = layer.output.get_size_dim_tag(0)
            print("layer:", layer, "size:", layer_size)
            assert size_enc0 == layer_size != size_enc1, "no match for layer %r" % layer
        for name in ["lstm0_pool", "lstm1_fw", "encoder", "enc_value", "output", "ref_att_weights"]:
            layer = net.layers[name]
            layer_size = layer.output.get_size_dim_tag(-1)
            print("layer:", layer, "size:", layer_size)
            assert size_enc0 != layer_size == size_enc1, "no match for layer %r" % layer
        print("All good.")


def test_rec_layer_search_select_src():
    from returnn.tf.layers.rec import _SubnetworkRecCell

    n_src_dim = 5
    n_tgt_dim = 7
    beam_size = 12
    config = Config()
    config.update({"debug_print_layer_output_template": True, "optimize_move_layers_out": False})

    def get_net_dict():
        return {
            "source_embed": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "n_out": 6,
                "from": "data:data",
            },
            "lstm0_fw": {
                "class": "rec",
                "unit": "standardlstm",
                "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                "initial_state": "var",
                "n_out": 10,
                "direction": 1,
                "from": ["source_embed"],
            },
            "lstm0_bw": {
                "class": "rec",
                "unit": "standardlstm",
                "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                "initial_state": "var",
                "n_out": 10,
                "direction": -1,
                "from": ["source_embed"],
            },
            "lstm1_fw": {
                "class": "rec",
                "unit": "standardlstm",
                "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                "initial_state": "var",
                "n_out": 10,
                "direction": 1,
                "from": ["lstm0_fw", "lstm0_bw"],
            },
            "lstm1_bw": {
                "class": "rec",
                "unit": "standardlstm",
                "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                "initial_state": "var",
                "n_out": 10,
                "direction": -1,
                "from": ["lstm0_fw", "lstm0_bw"],
            },
            "encoder": {"class": "copy", "from": ["lstm1_fw", "lstm1_bw"]},
            "enc_ctx": {"class": "linear", "activation": None, "with_bias": True, "from": ["encoder"], "n_out": 10},
            "fertility": {
                "class": "linear",
                "activation": "sigmoid",
                "with_bias": False,
                "from": ["encoder"],
                "n_out": 1,
            },
            "output": {
                "class": "rec",
                "from": [],
                "unit": {
                    "output": {
                        "class": "choice",
                        "target": "classes",
                        "beam_size": beam_size,
                        "from": ["output_prob"],
                        "initial_output": 0,
                    },
                    "end": {"class": "compare", "from": ["output"], "value": 0},
                    "target_embed": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["output"],
                        "n_out": 6,
                        "initial_output": "apply(0)",
                    },
                    "weight_feedback": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["prev:accum_att_weights"],
                        "n_out": 10,
                    },
                    "prev_s_state": {"class": "get_last_hidden_state", "from": ["prev:s"], "n_out": 20},
                    "prev_s_transformed": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["prev_s_state"],
                        "n_out": 10,
                    },
                    "energy_in": {
                        "class": "combine",
                        "kind": "add",
                        "from": ["base:enc_ctx", "weight_feedback", "prev_s_transformed"],
                        "n_out": 10,
                    },
                    "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
                    "energy": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["energy_tanh"],
                        "n_out": 1,
                    },
                    "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, enc-T, 1)
                    "accum_att_weights": {
                        "class": "eval",
                        "from": ["prev:accum_att_weights", "att_weights", "base:fertility"],
                        "eval": "source(0) + source(1) / (2.0 * source(2))",
                        "out_type": {"dim": 1, "shape": (None, 1)},
                    },
                    "att": {
                        "class": "generic_attention",
                        "weights": "att_weights",
                        "base": "base:encoder",
                        "auto_squeeze": True,
                    },
                    "s": {
                        "class": "rnn_cell",
                        "unit": "standardlstm",
                        "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                        "initial_state": "var",
                        "from": ["target_embed", "att"],
                        "n_out": 10,
                    },
                    "readout_in": {
                        "class": "linear",
                        "from": ["prev:s", "prev:target_embed", "att"],
                        "activation": None,
                        "n_out": 10,
                    },
                    "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
                    "output_prob": {"class": "softmax", "from": ["readout"], "target": "classes", "loss": "ce"},
                },
                "target": "classes",
                "max_seq_len": 20,
            },
            "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance", "target": "classes"},
        }

    print("Constructing search network.")
    tf_compat.v1.reset_default_graph()
    extern_data = ExternData(
        {
            "data": {"dim": n_src_dim, "sparse": True},
            "classes": {"dim": n_tgt_dim, "sparse": True, "available_for_inference": False},
        }
    )
    search_net = TFNetwork(extern_data=extern_data, search_flag=True, train_flag=False, eval_flag=True, config=config)
    search_net.construct_from_dict(get_net_dict())
    search_out_layer = search_net.layers["output"]
    assert isinstance(search_out_layer, RecLayer)
    assert isinstance(search_out_layer.cell, _SubnetworkRecCell)
    assert not search_out_layer.cell.input_layers_moved_out
    assert not search_out_layer.cell.output_layers_moved_out
    print("Layers in the loop:")
    loop_net = search_out_layer.cell.net
    for name, layer in sorted(loop_net.layers.items()):
        print("  %r: %s" % (name, layer))
        print("    search choices:", layer.get_search_choices())
        print("    sources:")
        for src in layer.sources:
            print("      %s" % src)
        print("    other deps:")
        for dep in layer.get_dep_layers():
            if dep in layer.sources:
                continue
            print("      %s" % dep)
    loop_out_layer = loop_net.layers["output"]
    assert isinstance(loop_out_layer, ChoiceLayer)
    assert isinstance(loop_out_layer.search_choices, SearchChoices)
    all_src_choices = loop_out_layer.search_choices.get_src_choices_seq()
    assert len(all_src_choices) == 2
    cur_out_choice, prev_out_choice = all_src_choices
    assert isinstance(cur_out_choice, SearchChoices)
    assert isinstance(prev_out_choice, SearchChoices)
    assert cur_out_choice == loop_out_layer.search_choices
    prev_loop_out_layer = loop_net.layers["prev:output"]
    assert prev_out_choice == prev_loop_out_layer.search_choices
    assert RecLayer.is_prev_step_layer(prev_out_choice.owner)
    assert_equal(loop_net.layers["end"].get_search_choices(), cur_out_choice)
    assert_equal(loop_net.layers["target_embed"].get_search_choices(), cur_out_choice)
    assert_equal(loop_net.layers["prev:target_embed"].get_search_choices(), prev_out_choice)
    assert_equal(loop_net.layers["accum_att_weights"].get_search_choices(), prev_out_choice)
    assert_equal(loop_net.layers["prev:accum_att_weights"].get_search_choices(), prev_out_choice)  # will be transformed
    assert_equal(loop_net.layers["weight_feedback"].get_search_choices(), prev_out_choice)
    loop_net.debug_search_choices(loop_net.layers["s"])
    assert_equal(loop_net.layers["s"].get_search_choices(), cur_out_choice)
    assert_equal(loop_net.layers["prev:s"].get_search_choices(), prev_out_choice)
    assert_equal(loop_net.layers["prev_s_state"].get_search_choices(), prev_out_choice)
    assert_equal(loop_net.layers["energy_in"].get_search_choices(), prev_out_choice)
    assert_equal(loop_net.layers["att_weights"].get_search_choices(), prev_out_choice)
    assert_equal(loop_net.layers["att"].get_search_choices(), prev_out_choice)
    assert_equal(loop_net.layers["output_prob"].get_search_choices(), prev_out_choice)


def test_RnnCellLayer_with_time():
    from returnn.datasets.generating import DummyDataset
    from returnn.tf.layers.basic import InternalLayer, SourceLayer, ReduceLayer

    train_data = DummyDataset(input_dim=2, output_dim=3, num_seqs=10, seq_len=5)
    with make_scope() as session:
        extern_data = ExternData()
        extern_data.init_from_dataset(train_data)
        net = TFNetwork(extern_data=extern_data)
        with tf_compat.v1.variable_scope("input_no_time_l"):
            input_no_time_l = InternalLayer(
                name="input_no_time_l",
                network=net,
                output=Data(name="input_no_time_l", dim=train_data.num_inputs, time_dim_axis=None),
            )
            print("Input layer (without time-dim):", input_no_time_l)
            assert input_no_time_l.output.shape == (train_data.num_inputs,)
            assert input_no_time_l.output.time_dim_axis is None
            assert not input_no_time_l.output.sparse
            assert input_no_time_l.output.dim == input_no_time_l.output.shape[-1]
            input_no_time_l.output.placeholder = LayerBase.get_rec_initial_output(
                batch_dim=1, name="input_no_time_l", n_out=10, output=input_no_time_l.output, rec_layer=None
            )  # dummy
        with tf_compat.v1.variable_scope("prev_l1"):
            prev_l = InternalLayer(name="prev:l1", network=net, output=Data(name="prev_l1", dim=10, time_dim_axis=None))
            prev_l.rec_vars_outputs["state"] = RnnCellLayer.get_rec_initial_state(
                batch_dim=1,
                name="prev_l",
                n_out=10,
                unit="LSTMBlock",
                sources=[input_no_time_l],
                network=net,
            )
            print("Previous time layer:", prev_l)
        with tf_compat.v1.variable_scope("l1"):
            opts = dict(
                name="l1", n_out=10, unit="LSTMBlock", network=net, rec_previous_layer=prev_l, sources=[input_no_time_l]
            )
            opts["output"] = RnnCellLayer.get_out_data_from_opts(**opts)
            l1 = RnnCellLayer(**opts)
            print("RnnCell layer (no time):", l1)
            print("RnnCell layer (no time) params:", l1.params)
            assert l1.output.time_dim_axis is None
            assert l1.output.batch_dim_axis == 0
            assert l1.output.dim == 10
            assert l1.output.shape == (10,)
        with tf_compat.v1.variable_scope("data"):
            opts = dict(network=net, name="data")
            opts["output"] = SourceLayer.get_out_data_from_opts(**opts)
            input_l = SourceLayer(**opts)
            print("Input layer (with time-dim):", input_l)
            assert input_l.output.dim == input_no_time_l.output.dim
            assert input_l.output.shape == (None, input_l.output.dim)
            assert input_l.output.time_dim_axis == 1
            assert not input_l.output.sparse
        with tf_compat.v1.variable_scope("l2"):
            opts = dict(name="l2", n_out=10, unit="LSTMBlock", network=net, sources=[input_l])
            opts["output"] = RnnCellLayer.get_out_data_from_opts(**opts)
            l2 = RnnCellLayer(**opts)
            print("RnnCell layer (with time):", l2)
            print("RnnCell layer (with time) params:", l2.params)
            assert l2.output.time_dim_axis == 0
            assert l2.output.batch_dim_axis == 1
            assert l2.output.dim == 10
            assert l2.output.shape == (None, 10)
            assert_equal(set(l1.params.keys()), set(l2.params.keys()))
            for key in l1.params.keys():
                assert l1.params[key].shape == l2.params[key].shape


def test_rec_subnet_simple_rnn():
    with make_scope() as session:
        n_in, n_out = 2, 3
        config = Config()
        config.update(
            {
                "num_outputs": n_out,
                "num_inputs": n_in,
                "network": {
                    "output": {
                        "class": "rec",
                        "from": "data:data",
                        "unit": {
                            # Recurrent subnet here, operate on a single time-step:
                            "output": {
                                "class": "linear",
                                "from": ["prev:output", "data:source"],
                                "activation": "relu",
                                "n_out": n_out,
                            },
                        },
                        "n_out": n_out,
                    },
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_dict["network"])
        output_layer = network.get_default_output_layer(must_exist=True)
        assert isinstance(output_layer, RecLayer)
        cell = output_layer.cell
        from returnn.tf.layers.rec import _SubnetworkRecCell

        assert isinstance(cell, _SubnetworkRecCell)
        cell_sub_layer_out = cell.layer_data_templates["output"].output
        assert isinstance(cell_sub_layer_out, Data)
        assert cell_sub_layer_out.time_dim_axis is None and cell_sub_layer_out.batch_dim_axis == 0
        assert cell_sub_layer_out.feature_dim_axis == 1 and cell_sub_layer_out.dim == n_out
        assert cell_sub_layer_out.batch_shape == (None, n_out)
        network.initialize_params(session)
        weights_var = network.layers["output"].params["output/W"]
        assert_equal(weights_var.get_shape().as_list(), [n_out + n_in, n_out])
        weights_np = (numpy.arange(0, (n_out + n_in) * n_out) - (n_out + n_in) * n_out * 0.5) * 0.1
        weights_np = weights_np.reshape((n_out + n_in, n_out))
        network.get_var_assigner(weights_var).assign(value=weights_np, session=session)
        input_np = [[[0.7, 0.1], [-0.3, -0.1], [0.2, -0.1]], [[1.0, -0.4], [-0.2, 0.3], [0.0, 0.0]]]
        input_np = numpy.array(input_np, dtype="float32")
        input_seq_lens = [3, 2]
        n_batch = len(input_seq_lens)
        assert_equal(input_np.shape, (n_batch, max(input_seq_lens), n_in))
        input_placeholder = network.extern_data.data["data"].placeholder
        input_seq_lens_placeholder = network.extern_data.data["data"].size_placeholder[0]
        output_np, output_seq_lens = session.run(
            (output_layer.output.get_placeholder_as_batch_major(), output_layer.output.get_sequence_lengths()),
            feed_dict={
                network.extern_data.get_batch_info().dim: len(input_seq_lens),
                input_placeholder: input_np,
                input_seq_lens_placeholder: input_seq_lens,
            },
        )
        assert_equal(list(output_seq_lens), input_seq_lens)
        assert_equal(output_np.shape, (n_batch, max(input_seq_lens), n_out))
        output_last_np = numpy.zeros((n_batch, n_out), dtype="float32")
        output_calc_np = numpy.zeros((n_batch, max(input_seq_lens), n_out), dtype="float32")
        for t in range(max(input_seq_lens)):
            _in = numpy.concatenate([output_last_np, input_np[:, t]], axis=1)
            assert_equal(_in.shape, (n_batch, n_out + n_in))
            _out = numpy.dot(_in, weights_np)
            assert_equal(_out.shape, (n_batch, n_out))
            _out = numpy.maximum(_out, 0.0)  # relu
            output_last_np = _out
            output_calc_np[:, t] = _out
        print("Manually calculated output:")
        print(output_calc_np)
        assert_almost_equal(output_np, output_calc_np)
        print("Simple RNN is fine!")

    # Now, kind of a separate test: rnn_cell in subnetwork.
    with make_scope() as session:
        print("Test rnn_cell in subnet.")
        config = Config()
        config.update(
            {
                "num_outputs": n_out,
                "num_inputs": n_in,
                "network": {
                    "output": {
                        "class": "rec",
                        "from": "data:data",
                        "optimize_move_layers_out": False,  # We esp. want to test it perform a single step, for debugging.
                        "unit": {
                            # Recurrent subnet here, operate on a single time-step:
                            "output": {
                                "class": "subnetwork",
                                "from": ["data:source"],
                                "subnetwork": {
                                    # RnnCellLayer inside subnetwork
                                    "output": {
                                        "class": "rnn_cell",
                                        "unit": "BasicRNN",
                                        "unit_opts": {"activation": tf.nn.relu},
                                        "from": ["data"],
                                        "n_out": n_out,
                                    },
                                },
                                "n_out": n_out,
                            }
                        },
                        "n_out": n_out,
                    },
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_dict["network"])
        network.initialize_params(session)
        output_layer = network.layers["output"]
        weights_var = output_layer.params["output/output/rec/basic_rnn_cell/kernel"]
        assert_equal(weights_var.get_shape().as_list(), [n_out + n_in, n_out])
        # BasicRNNCell expects it as [inputs, state], but we have it as [state, inputs].
        weights_conv_np = numpy.concatenate([weights_np[n_out:], weights_np[:n_out]])
        network.get_var_assigner(weights_var).assign(value=weights_conv_np, session=session)
        input_placeholder = network.extern_data.data["data"].placeholder
        input_seq_lens_placeholder = network.extern_data.data["data"].size_placeholder[0]
        output_np, output_seq_lens = session.run(
            (output_layer.output.get_placeholder_as_batch_major(), output_layer.output.get_sequence_lengths()),
            feed_dict={
                network.extern_data.get_batch_info().dim: len(input_seq_lens),
                input_placeholder: input_np,
                input_seq_lens_placeholder: input_seq_lens,
            },
        )
        assert_equal(list(output_seq_lens), input_seq_lens)
        assert_equal(output_np.shape, (n_batch, max(input_seq_lens), n_out))
        print("rnn_cell subnet output:")
        print(output_np)
        assert_almost_equal(output_np, output_calc_np)
        print("rnn_cell also fine.")


def check_reclayer_optimize_out(
    subnet_layer_dict,
    other_subnet_layers=None,
    shared_base_net=None,
    rtol=1e-4,
    feat_dim=None,
    time_dim=None,
    train_flag: bool = True,
):
    """
    :param dict[str] subnet_layer_dict: opts for the output layer inside the rec-layer subnet
    :param dict[str,dict[str]] other_subnet_layers: other layers for the rec-layer subnet
    :param dict[str,dict[str]] shared_base_net:
    :param float rtol: for the final comparison check
    :param Dim|None feat_dim:
    :param Dim|None time_dim:
    :param train_flag:
    """
    from returnn.tf.util.data import batch_dim

    subnet_layer_dict = subnet_layer_dict.copy()
    if feat_dim:
        n_in = feat_dim.dimension
    else:
        n_in = 13
        feat_dim = Dim(kind=Dim.Types.Feature, dimension=n_in, description="input-feature")
    if not time_dim:
        time_dim = SpatialDim("time")
    n_out = subnet_layer_dict.get("n_out", 17)
    if subnet_layer_dict.get("out_dim", None):
        n_out = subnet_layer_dict["out_dim"].dimension
    n_batch = 5
    n_time = 7
    subnet_layer_dict["n_out"] = n_out
    subnet_layer_dict.setdefault("from", ["data:source"])
    rec_layer_dict = {
        "class": "rec",
        "from": ["data"],
        "unit": {"output": subnet_layer_dict},
        "n_out": n_out,
        "is_output_layer": True,
    }
    if other_subnet_layers:
        assert "output" not in other_subnet_layers
        rec_layer_dict["unit"].update(other_subnet_layers)
    config = Config(
        {
            "debug_print_layer_output_template": True,
            "extern_data": {"data": {"dim_tags": [batch_dim, time_dim, feat_dim]}},
        }
    )
    from returnn.tf.layers.rec import _SubnetworkRecCell

    with make_scope() as session:
        print("Create non-optimized rec layer (with subnet layer moved out)")
        rec_layer_dict["optimize_move_layers_out"] = False
        net1 = TFNetwork(config=config, train_flag=train_flag, name="<root_not_opt>")
        if shared_base_net:
            net1.construct_from_dict(shared_base_net)
            for key in shared_base_net:
                assert key in net1.layers
        net1.construct_from_dict({"output_not_opt": rec_layer_dict})
        rec_layer_dict["optimize_move_layers_out"] = True
        print("Create optimized rec layer (with subnet layer inside loop)")
        net2 = TFNetwork(config=config, extern_data=net1.extern_data, train_flag=train_flag, name="<root_opt>")
        if shared_base_net:
            for key in shared_base_net:
                net2.layers[key] = net1.layers[key]
        net2.construct_from_dict({"output_opt": rec_layer_dict})
        net1_reclayer = net1.layers["output_not_opt"]
        assert isinstance(net1_reclayer, RecLayer)
        net1_subnet = net1_reclayer.cell
        assert isinstance(net1_subnet, _SubnetworkRecCell)
        net2_reclayer = net2.layers["output_opt"]
        assert isinstance(net2_reclayer, RecLayer)
        net2_subnet = net2_reclayer.cell
        assert isinstance(net2_subnet, _SubnetworkRecCell)
        assert_equal(set(net1_subnet.input_layers_moved_out), set())
        assert_equal(set(net2_subnet.input_layers_moved_out), set())
        assert_equal(set(net1_subnet.output_layers_moved_out), set())
        # output_layers_moved_out will contain sublayers if present
        output_root_layers_moved_out = [
            name for name in net2_subnet.output_layers_moved_out if "/" not in name and name != ":i"
        ]
        assert_equal(set(output_root_layers_moved_out), {"output"}.union(set(other_subnet_layers or [])))
        assert_equal(
            [v.name.split("/")[1:] for v in net1.get_params_list()],
            [v.name.split("/")[1:] for v in net2.get_params_list()],
        )
        net1.initialize_params(session=session)
        net1_params = net1.layers["output_not_opt"].get_param_values_dict(session=session)
        print("params:", list(net1_params.keys()))
        net2.layers["output_opt"].set_param_values_by_dict(values_dict=net1_params, session=session)
        x_np = net1.random.normal(size=(n_batch, n_time, n_in))
        net1_output = net1.layers["output_not_opt"].output.copy_masked(0.0).copy_as_batch_major()
        net2_output = net2.layers["output_opt"].output.copy_masked(0.0).copy_as_batch_major()
        print("output_not_opt:", net1_output)
        print("output_opt:", net2_output)
        assert net1_output.batch_shape == net2_output.batch_shape
        feed_dict = {
            net1.extern_data.get_batch_info().dim: n_batch,
            net1.extern_data.data["data"].placeholder: x_np,
            net1.extern_data.data["data"].size_placeholder[0]: [n_time] * n_batch,
        }
        y1_np = session.run(net1_output.placeholder, feed_dict=feed_dict)
        print("y: (shape %r)" % (y1_np.shape,))
        print(y1_np)
        y2_np = session.run(net2_output.placeholder, feed_dict=feed_dict)
        assert y1_np.shape == y2_np.shape
        assert y1_np.shape[:2] == y2_np.shape[:2] == (n_batch, n_time)
        assert y1_np.any() and y2_np.any()
        if not numpy.allclose(y1_np, y2_np, rtol=rtol):
            print("Not equal!")
            print("Iterating over shape [B,T,...] = %s" % (y1_np.shape,))
            for b in range(n_batch):
                for t in range(n_time):
                    print("check batch %i, time %i" % (b, t))
                    y1_np_, y2_np_ = y1_np[b, t], y2_np[b, t]
                    for idx in numpy.ndindex(*y1_np_.shape):
                        y1_np__, y2_np__ = y1_np_[idx], y2_np_[idx]
                        allclose = numpy.allclose(y1_np__, y2_np__, rtol=rtol)
                        if allclose:
                            details = ["all close"]
                        else:
                            details = ["not all close", "max1:", numpy.max(y1_np__), "max2:", numpy.max(y2_np__)]
                        print("  check", idx, *details)
            assert_allclose(y1_np, y2_np, rtol=rtol)  # fail now


def test_reclayer_optimize_out_linear():
    check_reclayer_optimize_out({"class": "linear", "activation": "relu"})


def test_reclayer_optimize_out_conv1d_no_dim_tags():
    # https://github.com/rwth-i6/returnn/issues/573
    # https://github.com/rwth-i6/returnn/pull/789
    input_feat_dim = FeatureDim("in-feature", dimension=15)
    check_reclayer_optimize_out(
        {"class": "conv", "from": "split", "filter_size": [3], "padding": "same"},
        {"split": {"class": "split_dims", "from": "data:source", "axis": "F", "dims": (5, 3)}},
        feat_dim=input_feat_dim,
    )


def test_reclayer_optimize_out_conv1d():
    # https://github.com/rwth-i6/returnn/issues/573
    # https://github.com/rwth-i6/returnn/pull/789
    input_feat_dim = FeatureDim("in-feature", dimension=15)
    new_feat_dim = FeatureDim("split-feature", dimension=3)
    spatial_dim = SpatialDim("split-spatial", dimension=5)
    check_reclayer_optimize_out(
        {"class": "conv", "from": "split", "in_spatial_dims": [spatial_dim], "filter_size": [3], "padding": "same"},
        {
            "split": {
                "class": "split_dims",
                "from": "data:source",
                "axis": input_feat_dim,
                "dims": (spatial_dim, new_feat_dim),
            }
        },
        feat_dim=input_feat_dim,
    )


def test_reclayer_optimize_out_pool1d():
    # https://github.com/rwth-i6/returnn/issues/573
    # https://github.com/rwth-i6/returnn/pull/789
    input_feat_dim = FeatureDim("in-feature", dimension=15)
    new_feat_dim = FeatureDim("split-feature", dimension=3)
    spatial_dim = SpatialDim("split-spatial", dimension=5)
    check_reclayer_optimize_out(
        {"class": "linear", "from": "pool"},
        {
            "split": {
                "class": "split_dims",
                "from": "data:source",
                "axis": input_feat_dim,
                "dims": (spatial_dim, new_feat_dim),
            },
            # This is the test.
            "pool": {
                "class": "pool",
                "from": "split",
                "in_spatial_dims": [spatial_dim],
                "pool_size": [3],
                "padding": "same",
                "mode": "max",
            },
        },
        feat_dim=input_feat_dim,
    )


def test_reclayer_optimize_out_transposed_conv1d_no_dim_tags():
    # https://github.com/rwth-i6/returnn/issues/573
    input_feat_dim = FeatureDim("in-feature", dimension=15)
    check_reclayer_optimize_out(
        {"class": "transposed_conv", "from": "split", "filter_size": [3], "padding": "same"},
        {"split": {"class": "split_dims", "from": "data:source", "axis": "F", "dims": (5, 3)}},
        feat_dim=input_feat_dim,
    )


def test_reclayer_optimize_out_rnncell():
    check_reclayer_optimize_out({"class": "rnn_cell", "unit": "BasicLSTM"})


def test_reclayer_optimize_out_rec_zoneout():
    check_reclayer_optimize_out({"class": "rec", "unit": "ZoneoutLSTM"})


def test_reclayer_optimize_out_rec_nativelstm2():
    check_reclayer_optimize_out({"class": "rec", "unit": "NativeLstm2"})


def test_test_reclayer_optimize_out_inner_rec_layer():
    lstm_window_dim = SpatialDim("lstm-window", dimension=5)
    check_reclayer_optimize_out(
        {"class": "rec", "unit": "nativelstm2", "from": "win", "axis": lstm_window_dim},
        {
            "win": {
                "class": "window",
                "window_dim": lstm_window_dim,
                "window_right": 0,
                "from": "data:source",
            },  # (B,W,D)
        },
    )


def test_test_reclayer_optimize_out_onlineblstm():
    network = {}
    lstm_dim = 13
    lstm_window = 5
    lstm_window_dim = SpatialDim("lstm-window", dimension=lstm_window)

    def add_lstm(i, direction, src):
        name = "lstm%i_%s" % (i, {1: "fw", -1: "bw"}[direction])
        if direction > 0:
            network[name] = {"class": "rec", "unit": "nativelstm2", "n_out": lstm_dim, "from": src}
            return name
        network["%s_win" % name] = {
            "class": "window",
            "window_dim": lstm_window_dim,
            "window_right": 0,
            "from": src,
        }  # (B,T,W,D)
        network["%s_rec" % name] = {
            "class": "rec",
            "unit": "nativelstm2",
            "axis": lstm_window_dim,
            "n_out": lstm_dim,
            "direction": -1,
            "from": "%s_win" % name,
        }  # (B,T,W,D')
        network["%s_cur" % name] = {
            "class": "slice",
            "axis": lstm_window_dim,
            "slice_end": 1,
            "from": "%s_rec" % name,
        }  # (B,T,1,D')
        network["%s_cursq" % name] = {"class": "squeeze", "axis": "dim:1", "from": "%s_cur" % name}  # (B,T,D')
        return "%s_cursq" % name

    num_layers = 3
    src = "data:source"
    for i in range(num_layers):
        fwd = add_lstm(i, 1, src)
        bwd = add_lstm(i, -1, src)
        src = [fwd, bwd]
    check_reclayer_optimize_out({"class": "linear", "from": src}, network)


def test_reclayer_optimize_out_selfatt_left():
    check_reclayer_optimize_out(
        {"class": "self_attention", "attention_left_only": True, "num_heads": 2, "total_key_dim": 6, "n_out": 18}
    )


def test_reclayer_optimize_out_cum_concat_gen_self_att():
    new_dim = SpatialDim("cum_concat_new_dim")
    key_dim = FeatureDim("key_dim", dimension=5)
    value_dim = FeatureDim("value_dim", dimension=7)
    n_key = 5
    n_value = 7
    check_reclayer_optimize_out(
        {"class": "linear", "from": "att", "activation": None},
        {
            # This is very much the vanilla self attention,
            # implemented via the new generic way.
            # See https://github.com/rwth-i6/returnn/issues/391 for a long discussion.
            # Commented shapes are always for the layers inside the loop (not optimized).
            "qkv": {
                "class": "linear",
                "from": "data:source",
                "activation": None,
                "n_out": n_key * 2 + n_value,
            },  # [B,2*K+V]
            "qkv_split": {"class": "split", "from": "qkv", "out_dims": [key_dim, key_dim, value_dim]},
            "q": {"class": "copy", "from": "qkv_split/0"},  # inside [B,K]. optimized out [T,B,K]
            "k": {"class": "copy", "from": "qkv_split/1"},  # inside [B,K]. optimized out [T,B,K]
            "v": {"class": "copy", "from": "qkv_split/2"},  # inside [B,V]. optimized out [T,B,V]
            # cum_concat here. Note that the optimized-out shape is not as you might expect [T,max(t),B,K],
            # but instead using the optimized format, with extended dyn size on the special dim tag,
            # i.e. [t*,B,K], representing [T,t*,B,K].
            "k_accum": {
                "class": "cum_concat",
                "out_spatial_dim": new_dim,
                "from": "k",
            },  # inside [t,B,K]. opt out [t*,B,K]
            "v_accum": {
                "class": "cum_concat",
                "out_spatial_dim": new_dim,
                "from": "v",
            },  # inside [t,B,V]. opt out [t*,B,K]
            "energy": {
                "class": "dot",
                "from": ["q", "k_accum"],
                "red1": key_dim,
                "red2": key_dim,
                "var1": None,
                "var2": new_dim,
            },  # inside [B,t]. optimized out [T,B,t*]
            "att_weights": {
                "class": "softmax_over_spatial",
                "from": "energy",
                "axis": new_dim,
            },  # inside [B,t]. opt out [T,B,t*]
            "att": {
                "class": "dot",
                "from": ["att_weights", "v_accum"],
                "red1": new_dim,
                "red2": new_dim,
                "var1": None,
                "var2": value_dim,
            },  # inside [B,V]. opt out [T,B,V]
        },
    )


def test_reclayer_optimize_out_accum_loop_dyn_size():
    # We want to test for the case where some layer inside the loop
    # generates some dyn size of shape [B] which is different in each loop frame.
    # So outside the loop, the accumulated dyn size should be of shape [T,B] or [B,T].
    # To test this, we first generate some random seq lens based on the input data (shape [B,T,D]).
    from returnn.tf.util.basic import py_print
    from returnn.tf.util.data import batch_dim, Dim

    def _eval_seq_lens(source, **_kwargs):
        # Get some random varying seq lens.
        res = tf.cast(4.0 * source(0) / source(1) + 0.3 * tf.cast(source(2), tf.float32), tf.int32) + 1
        res = py_print(res, ["seq lens", res, "step :i", source(2)])
        return res

    new_time_dim = SpatialDim("T_new")
    feat_dim = FeatureDim("F", dimension=13)
    check_reclayer_optimize_out(
        feat_dim=feat_dim,
        subnet_layer_dict={"class": "linear", "from": "combine", "activation": None, "n_out": 3},
        other_subnet_layers={
            "exp_data": {"class": "activation", "from": "data:source", "activation": "exp"},  # >0
            "sum_exp_data": {"class": "reduce", "mode": "sum", "from": "exp_data", "axis": "F"},  # [B]
            "seq_lens": {
                "class": "eval",
                "from": ["sum_exp_data", "base:max_sum_exp_data", ":i"],
                "out_type": {"dtype": "int32"},
                "out_shape": {batch_dim},
                "eval": _eval_seq_lens,
            },  # [B]
            "range": {"class": "range_from_length", "from": "seq_lens", "out_spatial_dim": new_time_dim},  # [T_new]
            "combine": {
                "class": "eval",
                "from": ["data:source", "range"],
                "eval": "source(0) + 0.1 * tf.cast(source(1), tf.float32)",
                "out_shape": {batch_dim, new_time_dim, feat_dim},
            },  # [B,T_new,D]
        },
        shared_base_net={
            "exp_data": {"class": "activation", "from": "data", "activation": "exp"},  # >0
            "sum_exp_data": {"class": "reduce", "mode": "sum", "from": "exp_data", "axis": "F"},  # [B,T]
            "max_sum_exp_data": {
                "class": "reduce",
                "mode": "max",
                "from": "sum_exp_data",
                "axis": "T",
                "is_output_layer": True,
            },  # [B]
        },
    )


def test_reclayer_optimize_out_dot():
    # Used for multi-head dot-attention.
    AttNumHeads = 4
    EncKeyPerHeadDim = 5
    EncValuePerHeadDim = 7
    EncKeyTotalDim = AttNumHeads * EncKeyPerHeadDim
    EncValueTotalDim = AttNumHeads * EncValuePerHeadDim
    check_reclayer_optimize_out(
        {"class": "linear", "activation": None, "from": ["att"]},
        other_subnet_layers={
            "s": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "from": ["data:source"],
                "n_out": EncKeyTotalDim,
            },  # (B, D)  -- Q (query). D should be same as enc_ctx
            "att_query": {
                "class": "split_dims",
                "axis": "F",
                "dims": (AttNumHeads, EncKeyPerHeadDim),
                "from": ["s"],
            },  # (B, H, D/H)
            # Here is the main test, the dot-layer:
            "energy": {
                "class": "dot",
                "red1": "dim:%i" % EncKeyPerHeadDim,
                "red2": "dim:%i" % EncKeyPerHeadDim,
                "var1": "T",
                "var2": "T?",  # Note the "T?".
                "from": ["base:enc_ctx", "att_query"],
            },
            # energy inside the loop will be (B, H, enc-T, 1).
            # energy outside the loop will be (B, H, enc-T, dec-T). I.e. enc-T is still the first time axis.
            "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, enc-T, H, 1)
            "att0": {"class": "generic_attention", "weights": "att_weights", "base": "base:enc_value"},  # (B, H, V)
            "att": {
                "class": "merge_dims",
                "axes": ["dim:%i" % AttNumHeads, "dim:%i" % EncValuePerHeadDim],
                "from": "att0",
            },  # (B, H*V); Use "static" here.
        },
        shared_base_net={
            "encoder": {"class": "copy", "from": ["data"]},
            "enc_ctx0": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "from": ["encoder"],
                "n_out": EncKeyTotalDim,
            },  # (B, enc-T, D)
            "enc_ctx": {
                "class": "split_dims",
                "axis": "F",
                "dims": (AttNumHeads, EncKeyPerHeadDim),
                "from": ["enc_ctx0"],
                "is_output_layer": True,
            },  # (B, enc-T, H, D/H)
            "enc_value0": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "from": ["encoder"],
                "n_out": EncValueTotalDim,
            },
            "enc_value": {
                "class": "split_dims",
                "axis": "F",
                "dims": (AttNumHeads, EncValuePerHeadDim),
                "from": ["enc_value0"],
                "is_output_layer": True,
            },  # (B, enc-T, H, D/H)
        },
        rtol=1e-3,
    )


def test_reclayer_optimize_out_dot_consistent_axes():
    # https://github.com/rwth-i6/returnn/issues/569
    # Used for multi-head dot-attention.
    n_heads = 4
    n_key = 5
    n_value = 7
    n_key_total = n_heads * n_key
    n_value_total = n_heads * n_value
    check_reclayer_optimize_out(
        {"class": "linear", "activation": None, "from": "att"},
        other_subnet_layers={
            "s": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "from": "data:source",
                "n_out": n_key_total,
            },  # (B, D)  -- Q (query). D should be same as enc_ctx
            "att_query": {"class": "split_dims", "axis": "F", "dims": (n_heads, n_key), "from": "s"},  # (B, H, D/H)
            # att_query is (T, B, H, D/H) outside the loop.
            # Here is the main test, the dot-layer:
            "energy": {
                "class": "dot",
                "red1": "dim:%i" % n_key,
                "red2": "dim:%i" % n_key,
                "var1": "T",
                "var2": None,
                "add_var2_if_empty": False,
                "from": ["base:enc_ctx", "att_query"],
            },
            # energy inside the loop will be (B, H, T).
            # energy outside the loop should be (B, H, T, T). I.e. T is still the first time axis.
            # The logic should be that the dot layer would add any extra axes (due to this optimization, moving layer out)
            # to either common shared axes (which is implicit) or if it is only added to one input,
            # then to the respective var axes.
            # Note that in this test, there is no different encoder or decoder time dim.
            # It still works due to time-dim-axis being set to the first source.
            "att_weights": {"class": "softmax_over_spatial", "from": "energy"},  # (B, T, H)
            "att0": {"class": "generic_attention", "weights": "att_weights", "base": "base:enc_value"},  # (B, H, V)
            "att": {
                "class": "merge_dims",
                "axes": ["dim:%i" % n_heads, "dim:%i" % n_value],
                "from": "att0",
            },  # (B, H*V); Use "static" here.
        },
        shared_base_net={
            "encoder": {"class": "copy", "from": "data"},
            "enc_ctx0": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "from": "encoder",
                "n_out": n_key_total,
            },  # (B, enc-T, D)
            "enc_ctx": {
                "class": "split_dims",
                "axis": "F",
                "dims": (n_heads, n_key),
                "from": "enc_ctx0",
                "is_output_layer": True,
            },  # (B, enc-T, H, D/H)
            "enc_value0": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "from": "encoder",
                "n_out": n_value_total,
            },
            "enc_value": {
                "class": "split_dims",
                "axis": "F",
                "dims": (n_heads, n_value),
                "from": "enc_value0",
                "is_output_layer": True,
            },  # (B, enc-T, H, D/H)
        },
        rtol=1e-3,
    )


def test_reclayer_optimize_out_dot_consistent_axes_enc_dec():
    # https://github.com/rwth-i6/returnn/issues/569
    # Used for multi-head dot-attention.
    n_heads = 4
    n_key = 5
    n_value = 7
    n_key_total = n_heads * n_key
    n_value_total = n_heads * n_value
    check_reclayer_optimize_out(
        {"class": "linear", "activation": None, "from": "att"},
        other_subnet_layers={
            "s": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "from": "data:source",
                "n_out": n_key_total,
            },  # (B, D)  -- Q (query). D should be same as enc_ctx
            "att_query": {"class": "split_dims", "axis": "F", "dims": (n_heads, n_key), "from": "s"},  # (B, H, D/H)
            # att_query is (dec-T, B, H, D/H) outside the loop.
            # Here is the main test, the dot-layer:
            "energy": {
                "class": "dot",
                "red1": "dim:%i" % n_key,
                "red2": "dim:%i" % n_key,
                "var1": "T",
                "var2": None,
                "add_var2_if_empty": False,
                "from": ["base:enc_ctx", "att_query"],
            },
            # energy inside the loop will be (B, H, enc-T).
            # energy outside the loop should be (B, H, enc-T, dec-T). I.e. enc-T is still the first time axis.
            # The logic should be that the dot layer would add any extra axes (due to this optimization, moving layer out)
            # to either common shared axes (which is implicit) or if it is only added to one input,
            # then to the respective var axes.
            "att_weights": {"class": "softmax_over_spatial", "from": "energy"},  # (B, enc-T, H)
            "att0": {"class": "generic_attention", "weights": "att_weights", "base": "base:enc_value"},  # (B, H, V)
            "att": {
                "class": "merge_dims",
                "axes": ["dim:%i" % n_heads, "dim:%i" % n_value],
                "from": "att0",
            },  # (B, H*V); Use "static" here.
        },
        shared_base_net={
            # Use conv with padding valid to make sure we get another time dim,
            # such that the rec part above will not confuse this time dim with the rec time dim.
            "encoder": {"class": "conv", "from": "data", "filter_size": [3], "padding": "valid", "n_out": 5},
            "enc_ctx0": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "from": "encoder",
                "n_out": n_key_total,
            },  # (B, enc-T, D)
            "enc_ctx": {
                "class": "split_dims",
                "axis": "F",
                "dims": (n_heads, n_key),
                "from": "enc_ctx0",
                "is_output_layer": True,
            },  # (B, enc-T, H, D/H)
            "enc_value0": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "from": "encoder",
                "n_out": n_value_total,
            },
            "enc_value": {
                "class": "split_dims",
                "axis": "F",
                "dims": (n_heads, n_value),
                "from": "enc_value0",
                "is_output_layer": True,
            },  # (B, enc-T, H, D/H)
        },
        rtol=1e-3,
    )


def test_reclayer_optimize_out_dot_kv_in_rec():
    # Same as test_reclayer_optimize_out_dot, but with the att key/value layers declared INSIDE the rec layer.
    AttNumHeads = 4
    EncKeyPerHeadDim = 5
    EncValuePerHeadDim = 7
    EncKeyTotalDim = AttNumHeads * EncKeyPerHeadDim
    EncValueTotalDim = AttNumHeads * EncValuePerHeadDim
    check_reclayer_optimize_out(
        {"class": "linear", "activation": None, "from": ["att"]},
        other_subnet_layers={
            "s": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "from": ["data:source"],
                "n_out": EncKeyTotalDim,
            },  # (B, D)  -- Q (query). D should be same as enc_ctx
            "att_query": {
                "class": "split_dims",
                "axis": "F",
                "dims": (AttNumHeads, EncKeyPerHeadDim),
                "from": ["s"],
            },  # (B, H, D/H)
            # this does not depend on the classes, but you should still be able to define it here.
            "enc_ctx0": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "from": ["base:encoder"],
                "n_out": EncKeyTotalDim,
            },  # (B, enc-T, D)
            "enc_ctx": {
                "class": "split_dims",
                "axis": "F",
                "dims": (AttNumHeads, EncKeyPerHeadDim),
                "from": ["enc_ctx0"],
                "is_output_layer": True,
            },  # (B, enc-T, H, D/H)
            "enc_value0": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "from": ["base:encoder"],
                "n_out": EncValueTotalDim,
            },  # (B, enc-T, D)
            "enc_value": {
                "class": "split_dims",
                "axis": "F",
                "dims": (AttNumHeads, EncValuePerHeadDim),
                "from": ["enc_value0"],
                "is_output_layer": True,
            },  # (B, enc-T, H, D/H)
            "energy": {
                "class": "dot",
                "red1": "dim:%i" % EncKeyPerHeadDim,
                "red2": "dim:%i" % EncKeyPerHeadDim,
                "var1": "T",
                "var2": "T?",  # Note the "T?".
                "from": ["enc_ctx", "att_query"],
            },
            "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, enc-T, H, 1)
            "att0": {"class": "generic_attention", "weights": "att_weights", "base": "enc_value"},  # (B, H, V)
            "att": {
                "class": "merge_dims",
                "axes": ["dim:%i" % AttNumHeads, "dim:%i" % EncValuePerHeadDim],
                "from": "att0",
            },  # (B, H*V); Use "static" here.
        },
        shared_base_net={
            # need to mark 'encoder' as output layer, otherwise it will not be constructed.
            "encoder": {"class": "copy", "from": ["data"], "is_output_layer": True}
        },
        rtol=1e-3,
    )


def test_reclayer_optimize_out_softmax_over_spatial():
    # Used for multi-head dot-attention.
    AttNumHeads = 4
    EncKeyPerHeadDim = 5
    EncValuePerHeadDim = 7
    EncKeyTotalDim = AttNumHeads * EncKeyPerHeadDim
    EncValueTotalDim = AttNumHeads * EncValuePerHeadDim
    check_reclayer_optimize_out(
        {"class": "linear", "activation": None, "from": ["squeeze"]},
        other_subnet_layers={
            "s": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "from": ["data:source"],
                "n_out": EncKeyTotalDim,
            },  # (B, D)  -- Q (query). D should be same as enc_ctx
            "att_query": {
                "class": "split_dims",
                "axis": "F",
                "dims": (AttNumHeads, EncKeyPerHeadDim),
                "from": ["s"],
            },  # (B, H, D/H)
            "energy": {
                "class": "dot",
                "red1": "dim:%i" % EncKeyPerHeadDim,
                "red2": "dim:%i" % EncKeyPerHeadDim,
                "var1": "T",
                "var2": "T?",  # Note the "T?".
                "from": ["base:enc_ctx", "att_query"],
            },
            # energy inside the loop will be (B, H, enc-T, 1).
            # energy outside the loop will be (B, H, enc-T, dec-T). I.e. enc-T is still the first time axis.
            "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, H, enc-T, 1)
            "slice": {"class": "slice", "from": "att_weights", "axis": "t", "slice_end": 1},  # (B, H, 1, 1)
            "squeeze0": {"class": "squeeze", "from": "slice", "axis": "t"},  # (B, H, 1)
            "squeeze": {"class": "squeeze", "from": "squeeze0", "axis": "auto", "allow_no_op": True},  # (B, H)
        },
        shared_base_net={
            "encoder": {"class": "copy", "from": ["data"]},
            "enc_ctx0": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "from": ["encoder"],
                "n_out": EncKeyTotalDim,
            },  # (B, enc-T, D)
            "enc_ctx": {
                "class": "split_dims",
                "axis": "F",
                "dims": (AttNumHeads, EncKeyPerHeadDim),
                "from": ["enc_ctx0"],
                "is_output_layer": True,
            },  # (B, enc-T, H, D/H)
            "enc_value0": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "from": ["encoder"],
                "n_out": EncValueTotalDim,
            },
            "enc_value": {
                "class": "split_dims",
                "axis": "F",
                "dims": (AttNumHeads, EncValuePerHeadDim),
                "from": ["enc_value0"],
                "is_output_layer": True,
            },  # (B, enc-T, H, D/H)
        },
        rtol=1e-3,
    )


def test_reclayer_optimize_out_softmax_over_spatial_rev_dot():
    # Used for multi-head dot-attention.
    AttNumHeads = 4
    EncKeyPerHeadDim = 5
    EncValuePerHeadDim = 7
    EncKeyTotalDim = AttNumHeads * EncKeyPerHeadDim
    EncValueTotalDim = AttNumHeads * EncValuePerHeadDim
    check_reclayer_optimize_out(
        {"class": "linear", "activation": None, "from": ["squeeze"]},
        other_subnet_layers={
            "s": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "from": ["data:source"],
                "n_out": EncKeyTotalDim,
            },  # (B, D)  -- Q (query). D should be same as enc_ctx
            "att_query": {
                "class": "split_dims",
                "axis": "F",
                "dims": (AttNumHeads, EncKeyPerHeadDim),
                "from": ["s"],
            },  # (B, H, D/H)
            "energy": {
                "class": "dot",
                "red1": "dim:%i" % EncKeyPerHeadDim,
                "red2": "dim:%i" % EncKeyPerHeadDim,
                "var1": "T?",
                "var2": "T",  # Note the "T?".
                "from": ["att_query", "base:enc_ctx"],
            },
            # energy inside the loop will be (B, H, 1, enc-T).
            # energy outside the loop will be (B, H, dec-T, enc-T). I.e. dec-T is the first time axis.
            "att_weights": {
                "class": "softmax_over_spatial",
                "axis": "stag-single:-1:time",
                "from": ["energy"],
            },  # (B, enc-T, H, 1)
            "slice": {
                "class": "slice",
                "from": "att_weights",
                "axis": "stag-single:-1:time",
                "slice_end": 1,
            },  # (B, 1, H, 1)
            "squeeze0": {"class": "squeeze", "from": "slice", "axis": "stag:slice"},  # (B, H, 1)
            "squeeze": {"class": "squeeze", "from": "squeeze0", "axis": "auto", "allow_no_op": True},  # (B, H)
        },
        shared_base_net={
            "encoder": {"class": "copy", "from": ["data"]},
            "enc_ctx0": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "from": ["encoder"],
                "n_out": EncKeyTotalDim,
            },  # (B, enc-T, D)
            "enc_ctx": {
                "class": "split_dims",
                "axis": "F",
                "dims": (AttNumHeads, EncKeyPerHeadDim),
                "from": ["enc_ctx0"],
                "is_output_layer": True,
            },  # (B, enc-T, H, D/H)
            "enc_value0": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "from": ["encoder"],
                "n_out": EncValueTotalDim,
            },
            "enc_value": {
                "class": "split_dims",
                "axis": "F",
                "dims": (AttNumHeads, EncValuePerHeadDim),
                "from": ["enc_value0"],
                "is_output_layer": True,
            },  # (B, enc-T, H, D/H)
        },
        rtol=1e-3,
    )


def test_reclayer_optimize_out_masked_computation_unmask():
    check_reclayer_optimize_out(
        {"class": "linear", "activation": None, "from": "unmask"},
        other_subnet_layers={
            "sum": {"class": "reduce", "mode": "sum", "from": "data:source", "axis": "f"},  # [B]
            "mask": {"class": "compare", "from": "sum", "value": 0.0, "kind": "greater"},  # [B]
            "masked": {
                "class": "masked_computation",
                "mask": "mask",
                "from": "data:source",
                "unit": {"class": "rec", "unit": "NativeLstm2", "n_out": 17, "from": "data"},
            },
            "unmask": {"class": "unmask", "from": "masked", "mask": "mask"},
        },
    )


def test_reclayer_optimize_out_masked_computation():
    check_reclayer_optimize_out(
        {"class": "linear", "activation": None, "from": "masked"},
        other_subnet_layers={
            "sum": {"class": "reduce", "mode": "sum", "from": "data:source", "axis": "f"},  # [B]
            "mask": {"class": "compare", "from": "sum", "value": 0.0, "kind": "greater"},  # [B]
            "masked": {
                "class": "masked_computation",
                "mask": "mask",
                "from": "data:source",
                "unit": {"class": "rec", "unit": "NativeLstm2", "n_out": 17, "from": "data"},
            },
        },
    )


def test_reclayer_optimize_out_masked_computation_out_shape():
    from returnn.tf.util.data import batch_dim

    lstm_out_dim = FeatureDim("lstm-out", 17)
    check_reclayer_optimize_out(
        {"class": "linear", "activation": None, "from": "masked"},
        other_subnet_layers={
            "sum": {"class": "reduce", "mode": "sum", "from": "data:source", "axis": "f"},  # [B]
            "mask": {"class": "compare", "from": "sum", "value": 0.0, "kind": "greater"},  # [B]
            "masked": {
                "class": "masked_computation",
                "mask": "mask",
                "from": "data:source",
                "unit": {
                    "class": "subnetwork",
                    "from": "data",
                    "subnetwork": {
                        "output": {
                            "class": "rec",
                            "unit": "NativeLstm2",
                            "out_dim": lstm_out_dim,
                            "from": "data",
                            "out_shape": {batch_dim, lstm_out_dim},
                        }
                    },
                },
            },
        },
    )


def test_reclayer_optimize_out_access_split():
    check_reclayer_optimize_out(
        subnet_layer_dict={"class": "copy", "from": "split/0", "n_out": 5},
        other_subnet_layers={"split": {"class": "split", "from": ["data:source"], "size_splits": [5, 8]}},
    )


def test_reclayer_optimize_out_lstm4d():
    from returnn.tf.util.data import single_step_dim

    feat_dim = FeatureDim("feat", 15)
    check_reclayer_optimize_out(
        feat_dim=feat_dim,
        subnet_layer_dict={"class": "rec", "unit": "nativelstm2", "from": "split", "axis": single_step_dim, "n_out": 7},
        other_subnet_layers={"split": {"class": "split_dims", "from": "data:source", "axis": "F", "dims": (5, 3)}},
    )


def test_reclayer_optimize_out_lstm4d_zoneout():
    from returnn.tf.util.data import single_step_dim

    feat_dim = FeatureDim("feat", 15)
    check_reclayer_optimize_out(
        feat_dim=feat_dim,
        subnet_layer_dict={
            "class": "rec",
            "unit": "zoneoutlstm",
            "unit_opts": {"zoneout_factor_cell": 0.15, "zoneout_factor_output": 0.05},
            "from": "split",
            "axis": single_step_dim,
            "n_out": 7,
        },
        other_subnet_layers={"split": {"class": "split_dims", "from": "data:source", "axis": "F", "dims": (5, 3)}},
        train_flag=False,
    )


def test_SplitLayer_move_out_as_output_layer():
    with make_scope() as session:
        config = Config()
        config.update(
            {
                "debug_print_layer_output_template": True,
                "extern_data": {
                    "data": {"dim": 11, "available_for_inference": True},
                    "classes": {"dim": 10, "available_for_inference": False},
                },
                "network": {
                    "output": {
                        "class": "rec",
                        "from": [],
                        "max_seq_len": 5,
                        "target": "classes",
                        "unit": {
                            "input": {
                                "class": "linear",
                                "n_out": 10,
                                "from": ["prev:input"],
                                "with_bias": True,
                            },  # not optimized out
                            "split": {
                                "class": "split",
                                "from": "input",
                                "size_splits": [1, 9],
                            },  # goal: optimize this out
                            "output": {
                                "class": "linear",
                                "from": ["split/0", "split/1"],
                                "target": "classes",
                            },  # optimized out
                            "end_compare": {"class": "compare", "kind": "greater", "from": ["split/0"], "value": 0},
                            "end": {"class": "squeeze", "from": ["end_compare"], "axis": "F"},
                        },
                    }
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_dict["network"])
        rec_layer = network.get_layer("output")
        assert isinstance(rec_layer, RecLayer)
        rec_cell = rec_layer.cell
        assert set(rec_cell.input_layers_moved_out) == set()
        assert set(rec_cell.output_layers_moved_out) == {"output", "split", "split/0", "split/1"}
        assert set(rec_cell.layers_in_loop) == {"input"}

        # run but don't care about the result
        session.run(tf_compat.v1.global_variables_initializer())
        output_layer = network.get_default_output_layer(must_exist=True)
        from test_TFNetworkLayer import make_feed_dict

        feed_dict = make_feed_dict(network.extern_data)
        session.run(output_layer.output.placeholder, feed_dict=feed_dict)


def test_reclayer_inner_nativelstm1():
    # https://github.com/rwth-i6/returnn/issues/813
    # NativeLstm1 is broken inside RecLayer.
    # This should cause an exception. We test this here.
    def _test_inner_rec_unit(unit):
        print("*** test rec unit", unit)
        net_dict = {
            "output": {
                "class": "rec",
                "from": "data",
                "optimize_move_layers_out": False,  # make sure it stays inside the loop, for the actual test
                "unit": {"output": {"class": "rec", "unit": unit, "from": "data:source", "n_out": 7}},
            }
        }
        with make_scope() as session:
            config = Config({"extern_data": {"data": {"dim": 3}}})
            net = TFNetwork(config=config)
            net.construct_from_dict(net_dict)
            net.initialize_params(session)
            out = net.get_default_output_layer()
            from test_TFNetworkLayer import make_feed_dict

            session.run(out.output.placeholder, feed_dict=make_feed_dict(net.extern_data))

    for unit in ["NativeLstm2", "lstm"]:
        _test_inner_rec_unit(unit)  # should be no error
    for unit in ["NativeLstm"]:
        try:
            _test_inner_rec_unit(unit)
        except Exception as exc:
            print("Expected exception:", exc)
            assert "https://github.com/rwth-i6/returnn/issues/813" in str(exc)
        else:
            raise Exception("Expect to get exception for unit %r. https://github.com/rwth-i6/returnn/issues/813" % unit)


def test_reclayer_single_step():
    # https://github.com/rwth-i6/returnn/issues/847
    # Specifically, outside a rec layer here.
    n_lstm = 5
    net_dict = {
        "linear": {"class": "linear", "from": "data", "n_out": n_lstm},
        "reduce": {"class": "reduce", "mode": "mean", "from": "linear", "axis": "T"},
        "output": {
            "class": "rec",
            "unit": "lstm",
            "n_out": n_lstm,
            # No need to explicitly specify axis=single_step_dim here,
            # it should be automatic from the input which does not have any spatial dim.
            "from": "reduce",
            "state": {"h": "reduce", "c": "reduce"},
        },
    }
    with make_scope() as session:
        config = Config({"extern_data": {"data": {"dim": 3}}})
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)
        net.initialize_params(session)
        out = net.get_default_output_layer()
        assert out.output.shape == (n_lstm,)
        from test_TFNetworkLayer import make_feed_dict

        session.run(out.output.placeholder, feed_dict=make_feed_dict(net.extern_data))


def test_reclayer_single_step_unrelated_time():
    # https://github.com/rwth-i6/returnn/issues/847
    # Specifically, outside a rec layer here.
    from returnn.tf.util.data import single_step_dim

    n_lstm = 5
    net_dict = {"output": {"class": "rec", "unit": "lstm", "from": "data", "n_out": n_lstm, "axis": single_step_dim}}
    with make_scope() as session:
        n_input = 3
        config = Config({"extern_data": {"data": {"dim": n_input}}})
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)
        net.initialize_params(session)
        in_ = net.extern_data.get_default_input_data()
        out = net.get_default_output_layer()
        assert out.output.shape == (None, n_lstm)
        n_batch = 2
        n_time = 7
        seq_lens = [7, 5]
        in_np = numpy.random.RandomState(42).uniform(-1.0, 1.0, size=(n_batch, n_time, n_input)).astype(numpy.float32)
        out_np = session.run(
            out.output.placeholder,
            feed_dict={
                net.extern_data.get_batch_info().dim: n_batch,
                in_.placeholder: in_np,
                in_.get_sequence_lengths(): seq_lens,
            },
        )
        assert isinstance(out_np, numpy.ndarray) and out_np.shape == (n_batch, n_time, n_lstm)
        in_flat_np = in_np.reshape((n_batch * n_time, 1, n_input))
        seq_lens_flat = [1] * n_batch * n_time
        out_flat_np = session.run(
            out.output.placeholder,
            feed_dict={
                net.extern_data.get_batch_info().dim: n_batch * n_time,
                in_.placeholder: in_flat_np,
                in_.get_sequence_lengths(): seq_lens_flat,
            },
        )
        assert isinstance(out_flat_np, numpy.ndarray) and out_flat_np.shape == (n_batch * n_time, 1, n_lstm)
        out_np_ = out_flat_np.reshape((n_batch, n_time, n_lstm))
        numpy.testing.assert_equal(out_np, out_np_)


def test_reclayer_att_with_kv_in_rec():
    net_dict = {
        "decision": {
            "class": "decide",
            "from": ["output"],
            "loss": "edit_distance",
            "loss_opts": {},
            "target": "classes",
        },
        "encoder": {"activation": None, "class": "linear", "is_output_layer": True, "n_out": 5, "from": "data:data"},
        "output": {
            "class": "rec",
            "max_seq_len": 'max_len_from("base:encoder") * 3',
            "target": "classes",
            "from": [],
            "unit": {
                "embed": {"activation": None, "class": "linear", "from": ["prev:output"], "n_out": 7},
                "att_query0": {
                    "activation": None,
                    "class": "linear",
                    "from": ["embed"],
                    "n_out": 6,
                    "with_bias": False,
                },
                "att_query": {"axis": "F", "class": "split_dims", "dims": (2, 3), "from": ["att_query0"]},
                # does not depend on rec-time, but still here declared within rec-layer:
                "att_key0": {
                    "activation": None,
                    "class": "linear",
                    "from": ["base:encoder"],
                    "n_out": 6,
                    "with_bias": False,
                },
                "att_key": {"axis": "F", "class": "split_dims", "dims": (2, 3), "from": ["att_key0"]},
                "att_value0": {
                    "activation": None,
                    "class": "linear",
                    "from": ["base:encoder"],
                    "n_out": 6,
                    "with_bias": False,
                },
                "att_value": {"axis": "F", "class": "split_dims", "dims": (2, 3), "from": ["att_value0"]},
                "att_energy": {
                    "class": "dot",
                    "from": ["att_query", "att_key"],
                    "red1": "dim:%i" % 3,
                    "red2": "dim:%i" % 3,
                    "var1": "T?",
                    "var2": "T",
                },
                "att_weights": {"axis": "T", "class": "softmax_over_spatial", "from": ["att_energy"]},
                "att_output": {"class": "generic_attention", "weights": "att_weights", "base": "att_value"},
                "att_att": {"axes": ["dim:2", "dim:3"], "class": "merge_dims", "from": ["att_output"]},
                "end": {"class": "compare", "from": ["output"], "value": 0},
                "output": {
                    "beam_size": 4,
                    "class": "choice",
                    "from": ["output_prob"],
                    "initial_output": "zeros",
                    "is_output_layer": True,
                    "loss": "ce",
                    "target": "classes",
                },
                "output_prob": {"class": "softmax", "from": "att_att", "target": "classes"},
            },
        },
    }

    with make_scope():
        config = Config(
            {
                "debug_print_layer_output_template": True,
                "extern_data": {
                    "data": {"dim": 7, "sparse": True},
                    "classes": {"dim": 6, "sparse": True, "available_for_inference": False},
                },
            }
        )
        net = TFNetwork(config=config, search_flag=True, train_flag=False, eval_flag=False)
        net.construct_from_dict(net_dict)


def test_reclayer_enc_time_dim_eval():
    """
    line: assert self.placeholder.shape[i].value == self.batch_shape[i]
    locals:
      self = <local> Data(name='accum_output', shape=(None, 1), batch_shape_meta=[B,T|?,F|1])
      self.placeholder = <local> <tf.Tensor 'output/rec/accum/add:0' shape=(?, ?, ?) dtype=float32>
      self.placeholder.shape = <local> TensorShape([Dimension(None), Dimension(None), Dimension(None)]), len = 3
      i = <local> 2
      value = <not found>
      self.batch_shape = <local> (None, None, 1)

    """
    with make_scope() as session:
        config = Config()
        config.update(
            {
                "debug_print_layer_output_template": True,
                "debug_print_layer_output_shape": True,
                "extern_data": {
                    "encoder": {"dim": 11, "available_for_inference": True},
                    "decoder": {"dim": 13, "available_for_inference": True},
                },
                "network": {
                    "encoder": {"class": "copy", "from": "data:encoder"},
                    "enc1": {"class": "linear", "from": "encoder", "activation": "relu", "n_out": 1},  # (B,enc-T,1)
                    "enc0": {"class": "squeeze", "axis": "f", "from": "enc1"},  # (B,enc-T)
                    "output": {
                        "class": "rec",
                        "from": "data:decoder",  # just to define a different time-dim
                        "unit": {
                            "accum": {
                                "class": "eval",
                                "from": ["prev:accum", "base:enc0", "base:enc1"],
                                "out_type": {"dim": 1, "shape": (None, 1)},
                                "eval": """(py_print(source(0), ["shape0", tf.shape(source(0))]) +
                          py_print(source(1), ["shape1", tf.shape(source(1))]) *
                          py_print(source(2), ["shape2", tf.shape(source(2))]))""",
                            },
                            "output": {"class": "reduce", "axis": "stag:encoder", "mode": "max", "from": "accum"},
                        },
                    },
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_dict["network"])
        session.run(tf_compat.v1.global_variables_initializer())
        output_layer = network.get_default_output_layer(must_exist=True)
        from test_TFNetworkLayer import make_feed_dict

        feed_dict = make_feed_dict(network.extern_data)
        session.run(output_layer.output.placeholder, feed_dict=feed_dict)


def test_reclayer_subnetwork_sublayer():
    import threading
    from returnn.tf.layers.rec import _SubnetworkRecCell

    class _Closure:
        lock = threading.Lock()
        counter = 0

    def _subnet_base_dummy_func(x):
        with _Closure.lock:
            print("** subnet_base_dummy_func", _Closure.counter)
            _Closure.counter += 1
        return x

    def _subnet_base_eval_func(source, **_kwargs):
        x = source(0)
        (y,) = tf_compat.v1.py_func(_subnet_base_dummy_func, [x], [tf.float32], name="subnet_base_dummy_func")
        y.set_shape(x.get_shape())
        return y

    with make_scope() as session:
        config = Config()
        config.update(
            {
                "debug_print_layer_output_template": True,
                "debug_print_layer_output_shape": True,
                "extern_data": {
                    "data": {"dim": 5},
                },
                "network": {
                    "encoder": {"class": "eval", "from": "data", "eval": "source(0) + 1"},
                    "output": {
                        "class": "rec",
                        "from": "data",
                        "unit": {
                            "outside": {"class": "reduce", "mode": "max", "axis": "t", "from": "base:encoder"},
                            "subnet": {
                                "class": "subnetwork",
                                "from": "data:source",
                                "subnetwork": {
                                    "a": {"class": "eval", "from": "base:outside", "eval": _subnet_base_eval_func},
                                    "b": {"class": "combine", "kind": "add", "from": ["data", "a", "base:prev:inside"]},
                                    "output": {"class": "copy", "from": "b"},
                                },
                            },
                            "subnet_a": {"class": "copy", "from": "subnet/a"},
                            "subnet_b": {"class": "copy", "from": "subnet/b"},
                            "inside": {"class": "combine", "kind": "add", "from": ["subnet_a", "subnet_b", "subnet"]},
                            "output": {"class": "copy", "from": "inside"},
                        },
                    },
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_dict["network"])

        # With the new subnetwork construction logic, individual sub layers can be moved out of the recurrent loop.
        rec_layer = network.get_layer("output")
        assert isinstance(rec_layer, RecLayer)
        rec_cell = rec_layer.cell
        assert isinstance(rec_cell, _SubnetworkRecCell)
        assert_equal(set(rec_cell.input_layers_moved_out), {"outside", "subnet/a", "subnet_a"})
        assert_equal(set(rec_cell.layers_in_loop), {"inside", "subnet", "subnet/b", "subnet/output", "subnet_b"})
        assert_equal(set(rec_cell.output_layers_moved_out), {"output"})

        session.run(tf_compat.v1.global_variables_initializer())
        from test_TFNetworkLayer import make_feed_dict

        feed_dict = make_feed_dict(network.extern_data)
        session.run(rec_layer.output.placeholder, feed_dict=feed_dict)
        # We expect that the _subnet_base_eval_func is exactly called once.
        assert _Closure.counter == 1


def test_reclayer_prev_in_subnet():
    # https://github.com/rwth-i6/returnn/issues/698
    with make_scope() as session:
        config = Config()
        config.update(
            {
                "extern_data": {"data": {"dim": 5}},
                "network": {
                    "output": {
                        "class": "rec",
                        "from": "data",
                        "unit": {
                            "accum_sub": {
                                "class": "subnetwork",
                                "from": [],
                                "subnetwork": {
                                    "accum": {
                                        "class": "combine",
                                        "kind": "add",
                                        "from": ["prev:accum", "base:data:source"],
                                    },
                                    "output": {"class": "copy", "from": "accum"},
                                },
                            },
                            "output": {"class": "copy", "from": "accum_sub"},
                        },
                    },
                },
            }
        )
        network = TFNetwork(config=config)
        network.construct_from_dict(config.typed_dict["network"])

        rec_layer = network.get_layer("output")
        assert isinstance(rec_layer, RecLayer)
        from test_TFNetworkLayer import make_feed_dict

        session.run(rec_layer.output.placeholder, feed_dict=make_feed_dict(network.extern_data))


def test_reclayer_batch_feature_input():
    """
    Test if the RecLayer is capable of handling [B,F,T] input
    """
    with make_scope() as session:
        config = Config()
        config.update(
            {
                "debug_print_layer_output_template": True,
                "debug_print_layer_output_shape": True,
                "extern_data": {
                    "data": {"dim": 5, "shape": (5, None), "time_dim_axis": 2, "feature_dim_axis": 1},
                },
                "network": {
                    "output": {"class": "rec", "direction": 1, "from": "data", "n_out": 2, "unit": "nativelstm2"},
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_dict["network"])
        session.run(tf_compat.v1.global_variables_initializer())
        output_layer = network.get_default_output_layer(must_exist=True)
        from test_TFNetworkLayer import make_feed_dict

        feed_dict = make_feed_dict(network.extern_data)
        session.run(output_layer.output.placeholder, feed_dict=feed_dict)


def test_reclayer_opt_output_consistent_format():
    from returnn.tf.util.data import batch_dim, Dim

    time_dim = SpatialDim("time")
    feat_dim = FeatureDim("input-feature", dimension=5)
    config = Config({"extern_data": {"data": {"dim_tags": [batch_dim, time_dim, feat_dim]}}})
    net_dict = {
        "loop": {
            "class": "rec",
            "from": [],
            "max_seq_len": 10,
            "unit": {
                "constant": {"class": "constant", "value": 1.0, "shape": [batch_dim]},  # scalar
                "add": {
                    "class": "combine",
                    "from": ["prev:i", "constant"],
                    "kind": "add",
                    "out_shape": {batch_dim},
                },  # [B] via 'i'. [T,B] outside
                "i": {"class": "copy", "from": "add"},  # [B] with default behavior currently
                "constant_0": {"class": "constant", "value": 4.9},
                "greater_equal": {"class": "compare", "from": ["add", "constant_0"], "kind": "greater_equal"},
                "end": {"class": "copy", "from": "greater_equal"},  # [B] via 'i'
                "reduce": {
                    "class": "reduce",
                    "axis": "T",
                    "from": "base:data:data",
                    "mode": "mean",
                },  # [B,F] both inside/out
                "mul": {
                    "class": "combine",
                    "from": ["add", "reduce"],
                    "kind": "mul",
                    "out_shape": {batch_dim, feat_dim},
                },  # [B,F] inside. [T,B,F] outside
                "output": {"class": "copy", "from": "mul"},
            },
        },
        "output": {"class": "copy", "from": "loop/output"},
    }
    with make_scope() as session:
        network = TFNetwork(config=config)
        network.construct_from_dict(net_dict)
        out = network.get_layer("output").output
        from test_TFNetworkLayer import make_feed_dict

        n_batch = 3
        feed_dict = make_feed_dict(network.extern_data, n_batch=n_batch)
        _, seq_len = session.run((out.placeholder, out.get_sequence_lengths()), feed_dict=feed_dict)
        assert list(seq_len) == [4] * n_batch


def test_reclayer_reuse_params_partly_moved_out():
    # https://github.com/rwth-i6/returnn/issues/555
    from test_TFNetworkLayer import make_feed_dict
    from returnn.tf.layers.basic import LinearLayer
    from returnn.tf.layers.rec import _SubnetworkRecCell

    with make_scope() as session:
        net_dict = {
            "output": {
                "class": "rec",
                "from": "data",
                "unit": {
                    "input": {"class": "copy", "from": ["prev:output", "data:source"]},
                    "FF_0": {"activation": "tanh", "class": "linear", "from": ["input"], "n_out": 5},
                    "FF_1": {
                        "activation": "tanh",
                        "class": "linear",
                        "from": ["input"],
                        "n_out": 5,  # moved out
                        "reuse_params": {"map": {"W": {"reuse_layer": "FF_0"}, "b": {"reuse_layer": "FF_0"}}},
                    },
                    "output": {"class": "softmax", "loss": "ce", "from": ["FF_0"]},
                    "output1": {"class": "softmax", "loss": "ce", "from": ["FF_1"]},  # moved out
                },
            },
        }
        config = Config({"extern_data": {"data": {"dim": 3}, "classes": {"dim": 7}}})
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(net_dict)
        rec_layer = net.get_default_output_layer(must_exist=True)
        assert isinstance(rec_layer, RecLayer)
        cell = rec_layer.cell
        assert isinstance(cell, _SubnetworkRecCell)
        ff0_inside = cell.net.layers["FF_0"]
        ff0_outside = cell.output_layers_net.layers["FF_0"]
        assert isinstance(ff0_inside, LinearLayer)
        assert isinstance(ff0_outside, LayerBase)  # does not really matter which class
        assert ff0_inside.params == ff0_outside.params
        session.run(tf_compat.v1.global_variables_initializer())
        session.run(rec_layer.output.placeholder, feed_dict=make_feed_dict(net.extern_data))


class TransformerNetwork:
    def __init__(self):
        self.encN = 3
        self.decN = 3
        self.FFDim = 13
        self.EncKeyTotalDim = 7 * 4
        self.AttNumHeads = 4
        self.EncKeyPerHeadDim = self.EncKeyTotalDim // self.AttNumHeads
        self.EncValueTotalDim = self.EncKeyTotalDim
        self.EncValuePerHeadDim = self.EncValueTotalDim // self.AttNumHeads
        self.embed_weight = self.EncValueTotalDim**0.5

        self.embed_dropout = 0.0
        self.postprocess_dropout = 0.0  # 0.1
        self.act_dropout = 0.0  # 0.1
        self.attention_dropout = 0.0  # 0.1
        self.label_smoothing = 0.0  # 0.1

        self.ff_init = "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)"

    def add_trafo_enc_layer(self, d, inp, output):
        """
        :param dict[str,dict[str]] d:
        :param str inp:
        :param str output:
        """
        d[output + "_self_att_laynorm"] = {"class": "layer_norm", "from": [inp]}
        d[output + "_self_att_att"] = {
            "class": "self_attention",
            "num_heads": self.AttNumHeads,
            "total_key_dim": self.EncKeyTotalDim,
            "n_out": self.EncValueTotalDim,
            "from": [output + "_self_att_laynorm"],
            "attention_left_only": False,
            "attention_dropout": self.attention_dropout,
            "forward_weights_init": self.ff_init,
        }
        d[output + "_self_att_lin"] = {
            "class": "linear",
            "activation": None,
            "with_bias": False,
            "from": [output + "_self_att_att"],
            "n_out": self.EncValueTotalDim,
            "forward_weights_init": self.ff_init,
        }
        d[output + "_self_att_drop"] = {
            "class": "dropout",
            "from": [output + "_self_att_lin"],
            "dropout": self.postprocess_dropout,
        }
        d[output + "_self_att_out"] = {
            "class": "combine",
            "kind": "add",
            "from": [inp, output + "_self_att_drop"],
            "n_out": self.EncValueTotalDim,
        }
        #####
        d[output + "_ff_laynorm"] = {"class": "layer_norm", "from": [output + "_self_att_out"]}
        d[output + "_ff_conv1"] = {
            "class": "linear",
            "activation": "relu",
            "with_bias": True,
            "from": [output + "_ff_laynorm"],
            "n_out": self.FFDim,
            "forward_weights_init": self.ff_init,
        }
        d[output + "_ff_conv2"] = {
            "class": "linear",
            "activation": None,
            "with_bias": True,
            "from": [output + "_ff_conv1"],
            "dropout": self.act_dropout,
            "n_out": self.EncValueTotalDim,
            "forward_weights_init": self.ff_init,
        }
        d[output + "_ff_drop"] = {
            "class": "dropout",
            "from": [output + "_ff_conv2"],
            "dropout": self.postprocess_dropout,
        }
        d[output + "_ff_out"] = {
            "class": "combine",
            "kind": "add",
            "from": [output + "_self_att_out", output + "_ff_drop"],
            "n_out": self.EncValueTotalDim,
        }
        d[output] = {"class": "copy", "from": [output + "_ff_out"]}

    def add_trafo_dec_layer(self, db, d, inp, output):
        """
        :param dict[str,dict[str]] db:
        :param dict[str,dict[str]] d:
        :param str inp:
        :param str output:
        """
        pre_inp = [inp]
        d[output + "_self_att_laynorm"] = {"class": "layer_norm", "from": pre_inp}
        d[output + "_self_att_att"] = {
            "class": "self_attention",
            "num_heads": self.AttNumHeads,
            "total_key_dim": self.EncKeyTotalDim,
            "n_out": self.EncValueTotalDim,
            "from": [output + "_self_att_laynorm"],
            "attention_left_only": True,
            "attention_dropout": self.attention_dropout,
            "forward_weights_init": self.ff_init,
        }
        d[output + "_self_att_lin"] = {
            "class": "linear",
            "activation": None,
            "with_bias": False,
            "from": [output + "_self_att_att"],
            "n_out": self.EncValueTotalDim,
            "forward_weights_init": self.ff_init,
        }
        d[output + "_self_att_drop"] = {
            "class": "dropout",
            "from": [output + "_self_att_lin"],
            "dropout": self.postprocess_dropout,
        }
        d[output + "_self_att_out"] = {
            "class": "combine",
            "kind": "add",
            "from": [inp, output + "_self_att_drop"],
            "n_out": self.EncValueTotalDim,
        }
        #####
        d[output + "_att_laynorm"] = {"class": "layer_norm", "from": [output + "_self_att_out"]}
        d[output + "_att_query0"] = {
            "class": "linear",
            "activation": None,
            "with_bias": False,
            "from": [output + "_att_laynorm"],
            "n_out": self.EncValueTotalDim,
            "forward_weights_init": self.ff_init,
        }
        d[output + "_att_query"] = {
            "class": "split_dims",
            "axis": "F",
            "dims": (self.AttNumHeads, self.EncKeyPerHeadDim),
            "from": [output + "_att_query0"],
        }  # (B, H, D/H)
        db[output + "_att_key0"] = {
            "class": "linear",
            "activation": None,
            "with_bias": False,
            "from": ["encoder"],
            "n_out": self.EncKeyTotalDim,
            "forward_weights_init": self.ff_init,
        }  # (B, enc-T, D)
        db[output + "_att_value0"] = {
            "class": "linear",
            "activation": None,
            "with_bias": False,
            "from": ["encoder"],
            "n_out": self.EncValueTotalDim,
            "forward_weights_init": self.ff_init,
        }
        db[output + "_att_key"] = {
            "class": "split_dims",
            "axis": "F",
            "dims": (self.AttNumHeads, self.EncKeyPerHeadDim),
            "from": [output + "_att_key0"],
        }  # (B, enc-T, H, D/H)
        db[output + "_att_value"] = {
            "class": "split_dims",
            "axis": "F",
            "dims": (self.AttNumHeads, self.EncValuePerHeadDim),
            "from": [output + "_att_value0"],
        }  # (B, enc-T, H, D'/H)
        d[output + "_att_energy"] = {
            "class": "dot",
            "red1": "dim:%i" % self.EncKeyPerHeadDim,
            "red2": "dim:%i" % self.EncKeyPerHeadDim,
            "var1": "T",
            "var2": "T?",
            "from": ["base:" + output + "_att_key", output + "_att_query"],
        }  # (B, H, enc-T, 1)
        d[output + "_att_weights"] = {
            "class": "softmax_over_spatial",
            "from": [output + "_att_energy"],
            "energy_factor": self.EncKeyPerHeadDim**-0.5,
        }  # (B, enc-T, H, 1)

        d[output + "_att_weights_drop"] = {
            "class": "dropout",
            "dropout_noise_shape": {"*": None},
            "from": [output + "_att_weights"],
            "dropout": self.attention_dropout,
        }

        d[output + "_att0"] = {
            "class": "generic_attention",
            "weights": output + "_att_weights_drop",
            "base": "base:" + output + "_att_value",
        }  # (B, H, V)
        d[output + "_att_att"] = {
            "class": "merge_dims",
            "axes": ["dim:%i" % self.AttNumHeads, "dim:%i" % self.EncValuePerHeadDim],
            "from": [output + "_att0"],
        }  # (B, H*V) except_batch
        d[output + "_att_lin"] = {
            "class": "linear",
            "activation": None,
            "with_bias": False,
            "from": [output + "_att_att"],
            "n_out": self.EncValueTotalDim,
            "forward_weights_init": self.ff_init,
        }
        d[output + "_att_drop"] = {
            "class": "dropout",
            "from": [output + "_att_lin"],
            "dropout": self.postprocess_dropout,
        }
        d[output + "_att_out"] = {
            "class": "combine",
            "kind": "add",
            "from": [output + "_self_att_out", output + "_att_drop"],
            "n_out": self.EncValueTotalDim,
        }
        #####
        d[output + "_ff_laynorm"] = {"class": "layer_norm", "from": [output + "_att_out"]}
        d[output + "_ff_conv1"] = {
            "class": "linear",
            "activation": "relu",
            "with_bias": True,
            "from": [output + "_ff_laynorm"],
            "n_out": self.FFDim,
            "forward_weights_init": self.ff_init,
        }
        d[output + "_ff_conv2"] = {
            "class": "linear",
            "activation": None,
            "with_bias": True,
            "from": [output + "_ff_conv1"],
            "dropout": self.act_dropout,
            "n_out": self.EncValueTotalDim,
            "forward_weights_init": self.ff_init,
        }
        d[output + "_ff_drop"] = {
            "class": "dropout",
            "from": [output + "_ff_conv2"],
            "dropout": self.postprocess_dropout,
        }
        d[output + "_ff_out"] = {
            "class": "combine",
            "kind": "add",
            "from": [output + "_att_out", output + "_ff_drop"],
            "n_out": self.EncValueTotalDim,
        }
        d[output] = {"class": "copy", "from": [output + "_ff_out"]}

    def build(self):
        network = {
            "source_embed_raw": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "n_out": self.EncValueTotalDim,
                "forward_weights_init": self.ff_init,
                "from": "data:data",
            },
            "source_embed_weighted": {
                "class": "eval",
                "from": ["source_embed_raw"],
                "eval": "source(0) * %f" % self.embed_weight,
            },
            "source_embed_with_pos": {
                "class": "positional_encoding",
                "add_to_input": True,
                "from": ["source_embed_weighted"],
            },
            "source_embed": {"class": "dropout", "from": ["source_embed_with_pos"], "dropout": self.embed_dropout},
            # encoder stack is added by separate function
            "encoder": {"class": "layer_norm", "from": ["enc_%02d" % self.encN]},
            "output": {
                "class": "rec",
                "from": [],
                "unit": {
                    "output": {
                        "class": "choice",
                        "target": "classes",
                        "beam_size": 12,
                        "from": ["output_prob"],
                        "initial_output": 0,
                    },  # this is a vocab_id, make this flexible
                    "end": {"class": "compare", "from": ["output"], "value": 0},
                    "target_embed_raw": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["prev:output"],
                        "n_out": self.EncValueTotalDim,
                        "forward_weights_init": self.ff_init,
                    },
                    # there seems to be no <s> in t2t, they seem to use just the zero vector
                    "target_embed_weighted": {
                        "class": "eval",
                        "from": ["target_embed_raw"],
                        "eval": "source(0) * %f" % self.embed_weight,
                    },
                    "target_embed_with_pos": {
                        "class": "positional_encoding",
                        "add_to_input": True,
                        "from": ["target_embed_weighted"],
                    },
                    "target_embed": {
                        "class": "dropout",
                        "from": ["target_embed_with_pos"],
                        "dropout": self.embed_dropout,
                    },
                    # decoder stack is added by separate function
                    "decoder": {"class": "layer_norm", "from": ["dec_%02d" % self.decN]},
                    "output_prob": {
                        "class": "softmax",
                        "from": ["decoder"],
                        "dropout": 0.0,
                        "target": "classes",
                        "loss": "ce",
                        "loss_opts": {"label_smoothing": self.label_smoothing},
                        "with_bias": False,
                        "forward_weights_init": self.ff_init,
                        "is_output_layer": True,
                    },
                },
                "target": "classes",
                "max_seq_len": "max_len_from('base:encoder') * 3",
            },
            "decision": {
                "class": "decide",
                "from": ["output"],
                "loss": "edit_distance",
                "target": "classes",
                "loss_opts": {
                    # "debug_print": True
                },
            },
        }

        self.add_trafo_enc_layer(network, "source_embed", "enc_01")
        for n in range(1, self.encN):
            self.add_trafo_enc_layer(network, "enc_%02d" % n, "enc_%02d" % (n + 1))

        self.add_trafo_dec_layer(network, network["output"]["unit"], "target_embed", "dec_01")
        for n in range(1, self.decN):
            self.add_trafo_dec_layer(network, network["output"]["unit"], "dec_%02d" % n, "dec_%02d" % (n + 1))

        return network


def test_reclayer_optimize_out_transformer():
    from returnn.tf.layers.rec import _SubnetworkRecCell

    n_src_dim = 5
    n_tgt_dim = 7

    def get_config(optimize_out):
        """
        :param bool optimize_out:
        :rtype: Config
        """
        return Config(
            {
                "debug_print_layer_output_template": True,
                "debug_print_layer_output_shape": True,  # only for debugging
                "extern_data": {
                    "data": {"dim": n_src_dim, "sparse": True},
                    "classes": {"dim": n_tgt_dim, "sparse": True, "available_for_inference": False},
                },
                "network": TransformerNetwork().build(),
                "optimize_move_layers_out": optimize_out,
            }
        )

    def get_feed_dict(extern_data):
        """
        :param ExternData extern_data:
        :rtype: dict[tf.Tensor,numpy.ndarray]
        """
        rnd = numpy.random.RandomState(42)
        n_batch = 3
        n_dec_times = numpy.array([11, 8, 9], dtype=Data.size_dtype)
        n_dec_time = max(n_dec_times)
        n_enc_times = numpy.array([7, 13, 5], dtype=Data.size_dtype)
        n_enc_time = max(n_enc_times)
        data_np = rnd.randint(0, n_src_dim, size=(n_batch, n_enc_time), dtype=extern_data.data["data"].dtype)
        classes_np = rnd.randint(0, n_tgt_dim, size=(n_batch, n_dec_time), dtype=extern_data.data["classes"].dtype)
        return {
            extern_data.get_batch_info().dim: n_batch,
            extern_data.data["data"].placeholder: data_np,
            extern_data.data["data"].size_placeholder[0]: n_enc_times,
            extern_data.data["classes"].placeholder: classes_np,
            extern_data.data["classes"].size_placeholder[0]: n_dec_times,
        }

    def get_params():
        print("create initial net, get params...")
        config = get_config(optimize_out=True)
        with make_scope() as session:
            net = TFNetwork(train_flag=True, config=config)
            net.construct_from_dict(config.typed_value("network"))
            net.initialize_params(session=session)
            params = net.get_params_serialized(session=session)
            return params

    net_params = get_params()
    print()

    def get_out(optimize_out):
        """
        :param bool optimize_out:
        :rtype: numpy.ndarray
        """
        print("optimize out:", optimize_out)
        config = get_config(optimize_out=optimize_out)

        with make_scope() as session:
            net = TFNetwork(train_flag=True, config=config)
            net.construct_from_dict(config.typed_value("network"))
            net.initialize_params(session=session)
            net.set_params_by_serialized(net_params, session=session)
            rec_layer = net.get_layer("output")
            assert isinstance(rec_layer, RecLayer)
            cell = rec_layer.cell
            assert isinstance(cell, _SubnetworkRecCell)
            assert_equal(cell.input_layers_moved_out, [])
            if optimize_out:
                assert_equal(cell.layers_in_loop, [])  # all moved out
            out = net.get_layer("output/output_prob").output.copy_as_batch_major()
            assert out.batch_ndim == 3 and out.shape == (None, n_tgt_dim)
            feed_dict = get_feed_dict(extern_data=net.extern_data)
            try:
                out_np = session.run(out.placeholder, feed_dict=feed_dict)
            except tf.errors.OpError as exc:
                help_on_tf_exception(
                    session, exc, fetches=out.placeholder, feed_dict=feed_dict, extern_data=net.extern_data
                )
                raise
            return out_np

    out_opt_np = get_out(optimize_out=True)
    print()
    out_nopt_np = get_out(optimize_out=False)
    print()
    print("output:")
    print(out_opt_np)
    numpy.testing.assert_almost_equal(out_opt_np, out_nopt_np, decimal=5)
    print("Both are equal!")


def test_reclayer_move_out_input_train_and_search():
    from returnn.tf.layers.rec import _SubnetworkRecCell

    n_src_dim = 5
    n_tgt_dim = 7
    beam_size = 12

    def make_extern_data():
        return ExternData(
            {
                "data": {"dim": n_src_dim, "sparse": True},
                "classes": {"dim": n_tgt_dim, "sparse": True, "available_for_inference": False},
            }
        )

    config = Config()
    config.update(
        {
            "debug_print_layer_output_template": True,
            "network": {
                "encoder": {"class": "linear", "activation": "tanh", "n_out": 5, "from": "data:data"},
                "output": {
                    "class": "rec",
                    "from": [],
                    "unit": {
                        "target_embed_raw": {
                            "activation": None,
                            "class": "linear",
                            "from": ["prev:output"],
                            "n_out": 13,
                            "with_bias": False,
                        },
                        # In train, this is in output_layers_moved_out (like all layers).
                        # In search, this is in input_layers_moved_out.
                        "encoder_int": {
                            "activation": None,
                            "class": "linear",
                            "from": ["base:encoder"],
                            "n_out": 11,
                            "with_bias": False,
                        },
                        "encoder_reduced": {"class": "reduce", "mode": "sum", "axis": "T", "from": ["encoder_int"]},
                        "output_prob": {
                            "class": "softmax",
                            "from": ["target_embed_raw", "encoder_reduced"],
                            "target": "classes",
                            "loss": "ce",
                        },
                        "output": {
                            "class": "choice",
                            "target": "classes",
                            "beam_size": beam_size,
                            "from": ["output_prob"],
                            "initial_output": 0,
                        },
                        "end": {"class": "compare", "from": ["output"], "value": 0},
                    },
                    "target": "classes",
                    "max_seq_len": 20,
                },
                "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance", "target": "classes"},
            },
        }
    )

    print("Constructing train network.")
    with make_scope():
        extern_data = make_extern_data()
        net = TFNetwork(extern_data=extern_data, train_flag=True, config=config)
        net.construct_from_dict(config.typed_value("network"))
        rec_layer = net.get_layer("output")
        assert isinstance(rec_layer, RecLayer)
        cell = rec_layer.cell
        assert isinstance(cell, _SubnetworkRecCell)
        assert_equal(cell.input_layers_moved_out, [])
        assert_equal(
            cell.output_layers_moved_out,
            ["output_prob", "target_embed_raw", "output", "encoder_reduced", "encoder_int"],
        )

    print("Constructing search network.")
    with make_scope():
        extern_data = make_extern_data()
        net = TFNetwork(extern_data=extern_data, search_flag=True, train_flag=False, eval_flag=True, config=config)
        net.construct_from_dict(config.typed_value("network"))
        rec_layer = net.get_layer("output")
        assert isinstance(rec_layer, RecLayer)
        cell = rec_layer.cell
        assert isinstance(cell, _SubnetworkRecCell)
        assert "encoder_int" in cell.input_layers_moved_out


def test_reclayer_optimize_out_cumsum_step_by_step():
    from returnn.tf.util.data import batch_dim, Dim

    time_dim = SpatialDim("time")
    feat_dim = FeatureDim("feat", dimension=11)
    check_reclayer_optimize_out(
        subnet_layer_dict={
            "class": "cumsum",
            "axis": time_dim,
            "out_shape": {batch_dim, feat_dim},
            "out_dim": feat_dim,
        },
        feat_dim=feat_dim,
        time_dim=time_dim,
    )


def test_reclayer_optimize_out_cumsum_step_by_step_initial():
    from returnn.tf.util.data import batch_dim, Dim

    time_dim = SpatialDim("time")
    feat_dim = FeatureDim("feat", dimension=11)
    check_reclayer_optimize_out(
        subnet_layer_dict={
            "class": "cumsum",
            "initial_output": 1,
            "axis": time_dim,
            "out_shape": {batch_dim, feat_dim},
            "out_dim": feat_dim,
        },
        feat_dim=feat_dim,
        time_dim=time_dim,
    )


def test_reclayer_optimize_out_cumsum_unrelated_axis():
    from returnn.tf.util.data import batch_dim, Dim

    time_dim = SpatialDim("time")
    feat_dim = FeatureDim("feat", dimension=11)
    check_reclayer_optimize_out(
        subnet_layer_dict={
            "class": "cumsum",
            "axis": feat_dim,
            "out_shape": {batch_dim, feat_dim},
            "out_dim": feat_dim,
        },
        feat_dim=feat_dim,
        time_dim=time_dim,
    )


def test_reclayer_optimize_out_rel_pos_enc_layer():
    # https://github.com/rwth-i6/returnn/issues/1253
    time_dim = SpatialDim("time")
    feat_dim = FeatureDim("feat", dimension=11)
    check_reclayer_optimize_out(
        feat_dim=feat_dim,
        time_dim=time_dim,
        subnet_layer_dict={"class": "copy", "from": "self_att", "out_dim": feat_dim},
        other_subnet_layers={
            "rel_pos": {
                "class": "relative_positional_encoding",
                "out_dim": feat_dim,
                "from": "data:source",
            },  # [T_new, F]
            "self_att": {
                "class": "self_attention",
                "from": "data:source",
                "out_dim": feat_dim,
                "num_heads": 1,
                "total_key_dim": feat_dim.dimension,
                "key_shift": "rel_pos",
                "attention_left_only": True,
            },
        },
    )


def test_subnet_load_on_init_rec():
    import tempfile

    model_tmp_dir = tempfile.mkdtemp("tmp-checkpoint")
    model_filename = model_tmp_dir + "/model"
    with make_scope() as session:
        config = Config()
        n_in, n_hidden, n_out = 2, 5, 3
        config.update(
            {
                "num_outputs": n_out,
                "num_inputs": n_in,
                "network": {
                    "input": {
                        "class": "linear",
                        "n_out": n_hidden,
                        "activation": "identity",
                        "forward_weights_init": "random_normal_initializer(mean=0.0, stddev=1.0)",
                        "from": "data:data",
                    },
                    "lstm0": {
                        "class": "rec",
                        "unit": "lstm",
                        "forward_weights_init": "random_normal_initializer(mean=0.0, stddev=1.0)",
                        "recurrent_weights_init": "random_normal_initializer(mean=0.0, stddev=1.0)",
                        "bias_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
                        "n_out": n_hidden,
                        "direction": 1,
                        "from": ["input"],
                    },
                    "lstm1": {
                        "class": "rec",
                        "unit": "lstm",
                        "forward_weights_init": "random_normal_initializer(mean=0.0, stddev=1.0)",
                        "recurrent_weights_init": "random_normal_initializer(mean=0.0, stddev=1.0)",
                        "bias_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
                        "n_out": n_hidden,
                        "direction": 1,
                        "from": ["lstm0"],
                    },
                    "output": {
                        "class": "linear",
                        "activation": "identity",
                        "forward_weights_init": "random_normal_initializer(mean=0.0, stddev=1.0)",
                        "bias_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
                        "n_out": n_out,
                        "from": ["lstm1"],
                    },
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_dict["network"])
        network.initialize_params(session)
        params_orig_dump = network.get_params_serialized(session)
        print("lstm0:")
        print(params_orig_dump.values_dict["lstm0"]["W"])
        assert params_orig_dump.values_dict["lstm0"]["W"].any()
        network.save_params_to_file(filename=model_filename, session=session)

        # Simple forward.
        input_np = [[[0.7, 0.1], [-0.3, -0.1], [0.2, -0.1]], [[1.0, -0.4], [-0.2, 0.3], [0.0, 0.0]]]
        input_np = numpy.array(input_np, dtype="float32")
        input_seq_lens = [3, 2]
        n_batch = len(input_seq_lens)
        assert_equal(input_np.shape, (n_batch, max(input_seq_lens), n_in))
        input_placeholder = network.extern_data.data["data"].placeholder
        input_seq_lens_placeholder = network.extern_data.data["data"].size_placeholder[0]
        output_layer = network.get_default_output_layer(must_exist=True)
        output_orig_np, output_seq_lens = session.run(
            (output_layer.output.get_placeholder_as_batch_major(), output_layer.output.get_sequence_lengths()),
            feed_dict={
                network.extern_data.get_batch_info().dim: len(input_seq_lens),
                input_placeholder: input_np,
                input_seq_lens_placeholder: input_seq_lens,
            },
        )
        assert_equal(list(output_seq_lens), input_seq_lens)
        assert_equal(output_orig_np.shape, (n_batch, max(input_seq_lens), n_out))
        for t in range(max(output_seq_lens)):
            for b in range(n_batch):
                if t >= output_seq_lens[b]:
                    output_orig_np[b, t] = 0.0
        print("LSTM direct, output:")
        print(output_orig_np)

    with make_scope() as session:
        config = Config()
        config.update(
            {
                "num_outputs": n_out,
                "num_inputs": n_in,
                "network": {
                    "output": {
                        "class": "rec",
                        "from": "data:data",
                        "optimize_move_layers_out": False,  # We esp. want to test it perform a single step, for debugging.
                        "unit": {
                            # Recurrent subnet here, operate on a single time-step:
                            "output": {
                                "class": "subnetwork",
                                "from": ["data:source"],
                                # Note: This has to convert the params into the right format.
                                "load_on_init": model_filename,
                                "subnetwork": {
                                    "input": {
                                        "class": "linear",
                                        "n_out": n_hidden,
                                        "activation": "identity",
                                        "from": "data",
                                    },
                                    "lstm0": {
                                        "class": "rnn_cell",
                                        "unit": "LSTMBlock",
                                        "unit_opts": {"forget_bias": 0.0},
                                        "n_out": n_hidden,
                                        "from": ["input"],
                                    },
                                    "lstm1": {
                                        "class": "rnn_cell",
                                        "unit": "LSTMBlock",
                                        "unit_opts": {"forget_bias": 0.0},
                                        "n_out": n_hidden,
                                        "from": ["lstm0"],
                                    },
                                    "output": {
                                        "class": "linear",
                                        "activation": "identity",
                                        "n_out": n_out,
                                        "from": ["lstm1"],
                                    },
                                },
                                "n_out": n_out,
                            },
                        },
                        "n_out": n_out,
                    },
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_dict["network"])
        network.initialize_params(session)

        # First just check whether the params are the same.
        params_dump = network.get_params_serialized(session)
        params_dump = params_dump.values_dict["output"]
        for layer_name in ["input", "output"]:  # not lstms, their layout differs
            layer_orig = params_orig_dump.values_dict[layer_name]
            for param_name in ["W", "b"]:
                param_orig = layer_orig[param_name]
                param_subnet = params_dump["output/%s/%s" % (layer_name, param_name)]
                numpy.testing.assert_array_equal(param_orig, param_subnet)

        # Now also forward, and compare with previous.
        input_placeholder = network.extern_data.data["data"].placeholder
        input_seq_lens_placeholder = network.extern_data.data["data"].size_placeholder[0]
        output_layer = network.get_default_output_layer(must_exist=True)
        output_np, output_seq_lens = session.run(
            (output_layer.output.get_placeholder_as_batch_major(), output_layer.output.get_sequence_lengths()),
            feed_dict={
                network.extern_data.get_batch_info().dim: len(input_seq_lens),
                input_placeholder: input_np,
                input_seq_lens_placeholder: input_seq_lens,
            },
        )
        assert_equal(list(output_seq_lens), input_seq_lens)
        assert_equal(output_np.shape, (n_batch, max(input_seq_lens), n_out))
        for t in range(max(output_seq_lens)):
            for b in range(n_batch):
                if t >= output_seq_lens[b]:
                    output_np[b, t] = 0.0
        print("LSTM rec subnet, output:")
        print(output_np)
        assert_almost_equal(output_orig_np, output_np)
        print("They are equal!")


def test_reclayer_prev_template_with_out_shape():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    dim_tags = {
        "extern_data.data.dim_tags.1.time": SpatialDim("time"),
        "extern_data.data.dim_tags.2.input": FeatureDim("input", 13),
    }

    extern_data = {
        "data": {
            "dim_tags": (
                batch_dim,
                dim_tags["extern_data.data.dim_tags.1.time"],
                dim_tags["extern_data.data.dim_tags.2.input"],
            ),
            "dtype": "float32",
            "available_for_inference": True,
        }
    }

    net_dict = {
        "reduce": {
            "class": "reduce",
            "from": "data:data",
            "mode": "mean",
            "axis": dim_tags["extern_data.data.dim_tags.1.time"],
            "out_shape": {batch_dim, dim_tags["extern_data.data.dim_tags.2.input"]},
        },
        "output": {
            "class": "rec",
            "from": [],
            "unit": {
                "constant_1": {"class": "constant", "value": 1.0},
                "state.i": {"class": "combine", "from": ["prev:state.i", "constant_1"], "kind": "add", "out_shape": {}},
                "constant_2_B": {"class": "constant", "value": 5.0, "shape": [batch_dim]},
                "greater_equal": {
                    "class": "compare",
                    "from": ["state.i", "constant_2_B"],
                    "kind": "greater_equal",
                    "out_shape": {batch_dim},
                },
                "end": {"class": "copy", "from": "greater_equal", "out_shape": {batch_dim}},
                "mul": {
                    "class": "combine",
                    "from": ["state.i", "base:reduce"],
                    "kind": "mul",
                    "out_shape": {batch_dim, dim_tags["extern_data.data.dim_tags.2.input"]},
                },
                "output": {
                    "class": "copy",
                    "from": "mul",
                    "out_shape": {batch_dim, dim_tags["extern_data.data.dim_tags.2.input"]},
                },
            },
            "max_seq_len": 10,
            "include_eos": True,
        },
    }

    with make_scope() as session:
        config = Config({"extern_data": extern_data})
        network = TFNetwork(config=config)
        network.construct_from_dict(net_dict)
        from test_TFNetworkLayer import make_feed_dict

        session.run(
            network.get_default_output_layer().output.placeholder, feed_dict=make_feed_dict(network.extern_data)
        )


def test_reclayer_loop_independent_out_shape():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    dim_tags = {
        "extern_data.data.dim_tags.1.time": SpatialDim("time"),
        "extern_data.data.dim_tags.2.input": FeatureDim("input", 13),
    }

    extern_data = {
        "data": {
            "dim_tags": (
                batch_dim,
                dim_tags["extern_data.data.dim_tags.1.time"],
                dim_tags["extern_data.data.dim_tags.2.input"],
            ),
            "dtype": "float32",
            "available_for_inference": True,
        }
    }

    net_dict = {
        "output": {
            "class": "rec",
            "from": [],
            "unit": {
                "constant_1": {"class": "constant", "value": 1.0},
                "state.i": {"class": "combine", "from": ["prev:state.i", "constant_1"], "kind": "add", "out_shape": {}},
                "constant_2_B": {"class": "constant", "value": 5.0, "shape": [batch_dim]},
                "greater_equal": {
                    "class": "compare",
                    "from": ["state.i", "constant_2_B"],
                    "kind": "greater_equal",
                    "out_shape": {batch_dim},
                },
                "end": {"class": "copy", "from": "greater_equal", "out_shape": {batch_dim}},
                "reduce": {
                    "class": "reduce",
                    "from": "base:data:data",
                    "mode": "mean",
                    "axis": dim_tags["extern_data.data.dim_tags.1.time"],
                    "out_shape": {batch_dim, dim_tags["extern_data.data.dim_tags.2.input"]},
                },
                "mul": {
                    "class": "combine",
                    "from": ["state.i", "reduce"],
                    "kind": "mul",
                    "out_shape": {batch_dim, dim_tags["extern_data.data.dim_tags.2.input"]},
                },
                "output": {
                    "class": "copy",
                    "from": "mul",
                    "out_shape": {batch_dim, dim_tags["extern_data.data.dim_tags.2.input"]},
                },
            },
            "max_seq_len": 10,
            "include_eos": True,
        },
    }

    with make_scope() as session:
        config = Config({"extern_data": extern_data})
        network = TFNetwork(config=config)
        network.construct_from_dict(net_dict)
        from test_TFNetworkLayer import make_feed_dict

        session.run(
            network.get_default_output_layer().output.placeholder, feed_dict=make_feed_dict(network.extern_data)
        )


def test_reclayer_shape_via_initial_output():
    # corresponds to test_rec_simple_iter in returnn_common
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    extern_data_data_dim_tags_1_time_dim = SpatialDim("time")
    extern_data_data_dim_tags_2_input_dim = FeatureDim("input", 13)
    network_loop_axis_loop_dim_dim = SpatialDim("loop-dim")

    config = Config(
        {
            "extern_data": {
                "data": {
                    "dim_tags": (
                        batch_dim,
                        extern_data_data_dim_tags_1_time_dim,
                        extern_data_data_dim_tags_2_input_dim,
                    ),
                    "dtype": "float32",
                    "available_for_inference": True,
                }
            }
        }
    )

    net_dict = {
        "loop": {
            "class": "rec",
            "from": [],
            "unit": {
                "state.i": {
                    "class": "combine",
                    "from": ["prev:state.i", "constant"],
                    "kind": "add",
                    "out_shape": {batch_dim},
                    "initial_output": "base:zeros",
                },
                "constant": {"class": "constant", "value": 1.0},
                "constant_0": {"class": "constant", "value": 5.0},
                "greater_equal": {
                    "class": "compare",
                    "from": ["state.i", "constant_0"],
                    "kind": "greater_equal",
                    "out_shape": {batch_dim},
                },
                "end": {"class": "copy", "from": "greater_equal", "out_shape": {batch_dim}},
                "reduce": {
                    "class": "reduce",
                    "from": "base:data:data",
                    "mode": "mean",
                    "axis": extern_data_data_dim_tags_1_time_dim,
                    "out_shape": {batch_dim, extern_data_data_dim_tags_2_input_dim},
                },
                "mul": {
                    "class": "combine",
                    "from": ["state.i", "reduce"],
                    "kind": "mul",
                    "out_shape": {batch_dim, extern_data_data_dim_tags_2_input_dim},
                },
                "output": {
                    "class": "copy",
                    "from": "mul",
                    "out_shape": {batch_dim, extern_data_data_dim_tags_2_input_dim},
                },
            },
            "max_seq_len": 10,
            "axis": network_loop_axis_loop_dim_dim,
            "include_eos": True,
            "out_shape": {batch_dim, extern_data_data_dim_tags_2_input_dim, network_loop_axis_loop_dim_dim},
        },
        "zeros": {"class": "constant", "value": 0.0, "shape": [batch_dim]},
        "output": {
            "class": "copy",
            "from": "loop/output",
            "out_shape": {batch_dim, extern_data_data_dim_tags_2_input_dim, network_loop_axis_loop_dim_dim},
        },
    }

    with make_scope() as session:
        network = TFNetwork(config=config)
        network.construct_from_dict(net_dict)
        from test_TFNetworkLayer import make_feed_dict

        session.run(
            network.get_default_output_layer().output.placeholder, feed_dict=make_feed_dict(network.extern_data)
        )


def test_reclayer_explicit_rec_ff():
    # corresponds to test_rec_ff in returnn_common
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    extern_data_data_dim_tags_1_time_dim = SpatialDim("time")
    extern_data_data_dim_tags_2_input_dim = FeatureDim("input", 13)
    network_loop_unit_state_h_subnetwork_dot_reduce_add_1_linear_out_dim = FeatureDim("linear-out", 13)

    config = Config(
        {
            "extern_data": {
                "data": {
                    "dim_tags": (
                        batch_dim,
                        extern_data_data_dim_tags_1_time_dim,
                        extern_data_data_dim_tags_2_input_dim,
                    ),
                    "dtype": "float32",
                    "available_for_inference": True,
                }
            }
        }
    )

    net_dict = {
        "loop": {
            "class": "rec",
            "from": [],
            "unit": {
                "state.h": {
                    "class": "subnetwork",
                    "from": [],
                    "subnetwork": {
                        "dot": {
                            "class": "dot",
                            "from": ["base:concat", "weight"],
                            "reduce": extern_data_data_dim_tags_2_input_dim
                            + network_loop_unit_state_h_subnetwork_dot_reduce_add_1_linear_out_dim,
                            "out_shape": {
                                batch_dim,
                                network_loop_unit_state_h_subnetwork_dot_reduce_add_1_linear_out_dim,
                            },
                        },
                        "add": {
                            "class": "combine",
                            "from": ["dot", "bias"],
                            "kind": "add",
                            "out_shape": {
                                batch_dim,
                                network_loop_unit_state_h_subnetwork_dot_reduce_add_1_linear_out_dim,
                            },
                        },
                        "output": {
                            "class": "copy",
                            "from": "add",
                            "out_shape": {
                                batch_dim,
                                network_loop_unit_state_h_subnetwork_dot_reduce_add_1_linear_out_dim,
                            },
                        },
                        "weight": {
                            "class": "variable",
                            "shape": [
                                extern_data_data_dim_tags_2_input_dim
                                + network_loop_unit_state_h_subnetwork_dot_reduce_add_1_linear_out_dim,
                                network_loop_unit_state_h_subnetwork_dot_reduce_add_1_linear_out_dim,
                            ],
                            "dtype": "float32",
                        },
                        "bias": {
                            "class": "variable",
                            "shape": [network_loop_unit_state_h_subnetwork_dot_reduce_add_1_linear_out_dim],
                            "dtype": "float32",
                        },
                    },
                    "out_shape": {batch_dim, network_loop_unit_state_h_subnetwork_dot_reduce_add_1_linear_out_dim},
                    "initial_output": "base:zeros",
                },
                "rec_unstack": {
                    "class": "rec_unstack",
                    "from": "base:data:data",
                    "axis": extern_data_data_dim_tags_1_time_dim,
                    "out_shape": {batch_dim, extern_data_data_dim_tags_2_input_dim},
                },
                "concat": {
                    "class": "concat",
                    "from": (
                        ("rec_unstack", extern_data_data_dim_tags_2_input_dim),
                        ("prev:state.h", network_loop_unit_state_h_subnetwork_dot_reduce_add_1_linear_out_dim),
                    ),
                    "out_shape": {
                        batch_dim,
                        extern_data_data_dim_tags_2_input_dim
                        + network_loop_unit_state_h_subnetwork_dot_reduce_add_1_linear_out_dim,
                    },
                },
                "output": {
                    "class": "copy",
                    "from": "state.h",
                    "out_shape": {batch_dim, network_loop_unit_state_h_subnetwork_dot_reduce_add_1_linear_out_dim},
                },
            },
            "axis": extern_data_data_dim_tags_1_time_dim,
            "out_shape": {
                batch_dim,
                extern_data_data_dim_tags_1_time_dim,
                network_loop_unit_state_h_subnetwork_dot_reduce_add_1_linear_out_dim,
            },
        },
        "zeros": {
            "class": "constant",
            "value": 0.0,
            "shape": [batch_dim, network_loop_unit_state_h_subnetwork_dot_reduce_add_1_linear_out_dim],
        },
        "output": {
            "class": "copy",
            "from": "loop/output",
            "out_shape": {
                batch_dim,
                extern_data_data_dim_tags_1_time_dim,
                network_loop_unit_state_h_subnetwork_dot_reduce_add_1_linear_out_dim,
            },
        },
    }

    with make_scope() as session:
        network = TFNetwork(config=config)
        network.construct_from_dict(net_dict)
        network.initialize_params(session)
        from test_TFNetworkLayer import make_feed_dict

        session.run(
            network.get_default_output_layer().output.placeholder, feed_dict=make_feed_dict(network.extern_data)
        )


def test_reclayer_att_weights_output_layer():
    # https://github.com/rwth-i6/returnn/issues/1027
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time")
    input_dim = FeatureDim("input", 13)
    align_classes_dim = FeatureDim("align", 5)

    config = Config(
        {
            "extern_data": {
                "data": {
                    "dim_tags": (batch_dim, time_dim, input_dim),
                    "dtype": "float32",
                    "available_for_inference": True,
                },
                "alignment": {
                    "dim_tags": (
                        batch_dim,
                        time_dim,
                    ),
                    "sparse_dim": align_classes_dim,
                    "dtype": "int32",
                    "available_for_inference": True,
                },
            }
        }
    )

    att_heads = Dim(kind=Dim.Types.Spatial, description="att_heads", dimension=1)
    att_t = Dim(kind=Dim.Types.Spatial, description="att_t", dimension=None)
    label_axis = Dim(kind=Dim.Types.Spatial, description="label-axis", dimension=None)

    net_dict = {
        "encoder": {"class": "copy", "from": "data"},
        "existing_alignment": {"class": "copy", "from": "data:alignment"},
        "is_label": {
            "class": "compare",
            "from": "existing_alignment",
            "kind": "not_equal",
            "value": align_classes_dim.dimension - 1,
        },
        "label_ground_truth_masked": {
            "class": "reinterpret_data",
            "enforce_batch_major": True,
            "from": "label_ground_truth_masked0",
            "register_as_extern_data": "label_ground_truth",
            "set_sparse_dim": align_classes_dim - 1,
        },
        "label_ground_truth_masked0": {
            "class": "masked_computation",
            "from": "existing_alignment",
            "mask": "is_label",
            "unit": {"class": "copy", "from": "data"},
        },
        "label_model": {
            "back_prop": True,
            "class": "rec",
            "from": "data:label_ground_truth",
            "include_eos": True,
            "is_output_layer": True,
            "name_scope": "output/rec",
            "unit": {
                "att": {"axes": ["stag:heads", input_dim], "class": "merge_dims", "from": "att0"},
                "att0": {
                    "add_var2_if_empty": False,
                    "class": "dot",
                    "from": ["att_val_split", "att_weights"],
                    "reduce": "stag:att_t",
                    "var1": "f",
                    "var2": None,
                },
                "att_ctx": {
                    "L2": None,
                    "activation": None,
                    "class": "linear",
                    "dropout": 0.0,
                    "from": "segments",
                    "n_out": 10,
                    "name_scope": "/enc_ctx",
                    "with_bias": True,
                },
                "att_energy": {
                    "class": "reinterpret_data",
                    "from": "att_energy0",
                    "is_output_layer": False,
                    "set_dim_tags": {"f": att_heads},
                },
                "att_energy0": {
                    "activation": None,
                    "class": "linear",
                    "from": ["energy_tanh"],
                    "n_out": 1,
                    "name_scope": "energy",
                    "with_bias": False,
                },
                "att_energy_in": {
                    "class": "combine",
                    "from": ["att_ctx", "att_query"],
                    "kind": "add",
                    "n_out": 10,
                },
                "att_query": {
                    "activation": None,
                    "class": "linear",
                    "from": "lm",
                    "is_output_layer": False,
                    "n_out": 10,
                    "with_bias": False,
                },
                "att_val": {"class": "copy", "from": "segments"},
                "att_val_split": {
                    "class": "reinterpret_data",
                    "from": "att_val_split0",
                    "set_dim_tags": {"dim:1": att_heads},
                },
                "att_val_split0": {
                    "axis": "f",
                    "class": "split_dims",
                    "dims": (1, -1),
                    "from": "att_val",
                },
                "att_weights": {
                    "class": "dropout",
                    "dropout": 0.0,
                    "dropout_noise_shape": {"*": None},
                    "from": "att_weights0",
                    "is_output_layer": True,
                },
                "att_weights0": {
                    "axis": "stag:att_t",
                    "class": "softmax_over_spatial",
                    "energy_factor": 0.03125,
                    "from": "att_energy",
                },
                "energy_tanh": {
                    "activation": "tanh",
                    "class": "activation",
                    "from": ["att_energy_in"],
                },
                "input_embed": {
                    "class": "copy",
                    "from": "prev:target_embed",
                },
                "lm": {
                    "class": "rec",
                    "from": ["input_embed", "prev:att"],
                    "n_out": 10,
                    "name_scope": "lm/rec",
                    "unit": "nativelstm2",
                },
                "output": {
                    "beam_size": 4,
                    "cheating": "exclusive",
                    "class": "choice",
                    "from": "data",
                    "initial_output": 0,
                    "target": "label_ground_truth",
                },
                "segment_lens": {
                    "axis": "t",
                    "class": "gather",
                    "from": "base:data:segment_lens_masked",
                    "position": ":i",
                },
                "segment_starts": {
                    "axis": "t",
                    "class": "gather",
                    "from": "base:data:segment_starts_masked",
                    "position": ":i",
                },
                "segments": {
                    "class": "reinterpret_data",
                    "from": "segments0",
                    "set_dim_tags": {"stag:sliced-time:segments": att_t},
                },
                "segments0": {
                    "class": "slice_nd",
                    "from": "base:encoder",
                    "size": "segment_lens",
                    "start": "segment_starts",
                },
                "target_embed": {
                    "activation": None,
                    "class": "linear",
                    "from": "output",
                    "initial_output": 0,
                    "n_out": 6,
                    "with_bias": False,
                },
            },
        },
        "output": {
            "back_prop": False,
            "class": "rec",
            "from": "encoder",
            "include_eos": True,
            "initial_output": 0,
            "size_target": None,
            "target": "alignment",
            "unit": {
                "const1": {"class": "constant", "value": 1},
                "output": {
                    "beam_size": 4,
                    "cheating": None,
                    "class": "choice",
                    "from": "data",
                    "initial_output": 0,
                    "input_type": "log_prob",
                    "length_normalization": False,
                    "target": "alignment",
                },
                "output_emit": {
                    "class": "compare",
                    "from": "output",
                    "initial_output": True,
                    "kind": "not_equal",
                    "value": align_classes_dim.dimension - 1,
                },
                "segment_lens": {
                    "class": "combine",
                    "from": ["segment_lens0", "const1"],
                    "is_output_layer": True,
                    "kind": "add",
                },
                "segment_lens0": {
                    "class": "combine",
                    "from": [":i", "segment_starts"],
                    "kind": "sub",
                },
                "segment_starts": {
                    "class": "switch",
                    "condition": "prev:output_emit",
                    "false_from": "prev:segment_starts",
                    "initial_output": 0,
                    "is_output_layer": True,
                    "true_from": ":i",
                },
            },
        },
        "segment_lens_masked": {
            "class": "masked_computation",
            "from": "output/segment_lens",
            "mask": "is_label",
            "out_spatial_dim": label_axis,
            "register_as_extern_data": "segment_lens_masked",
            "unit": {"class": "copy", "from": "data"},
        },
        "segment_starts_masked": {
            "class": "masked_computation",
            "from": "output/segment_starts",
            "mask": "is_label",
            "out_spatial_dim": label_axis,
            "register_as_extern_data": "segment_starts_masked",
            "unit": {"class": "copy", "from": "data"},
        },
    }

    with make_scope() as session:
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(net_dict)
        network.initialize_params(session)
        from test_TFNetworkLayer import make_feed_dict

        fetches = network.get_fetches_dict()
        fetches["out"] = network.get_layer("label_model/att_weights").output.placeholder
        session.run(fetches, feed_dict=make_feed_dict(network.extern_data))


def test_reclayer_subnetwork_base_subnet():
    with make_scope() as session:
        net_dict = {
            "sub": {
                "class": "subnetwork",
                "from": [],
                "subnetwork": {
                    "linear": {"class": "eval", "from": "base:data:data", "eval": "source(0) * 0.9"},
                    "reduce": {"class": "reduce", "mode": "mean", "axis": "T", "from": "linear"},
                    "output": {"class": "copy", "from": "linear"},
                },
            },
            "sub2": {
                "class": "subnetwork",
                "from": [],
                "subnetwork": {
                    "rec": {
                        "class": "rec",
                        "from": "base:data:data",
                        "unit": {
                            "add": {
                                "class": "combine",
                                "kind": "add",
                                "from": ["data:source", "prev:add", "base:base:sub/reduce"],
                            },
                            "output": {"class": "copy", "from": "add"},
                        },
                    },
                    "output": {"class": "copy", "from": "rec"},
                },
            },
            "output": {"class": "copy", "from": "sub2"},
        }
        config = Config(dict(num_inputs=1, num_outputs=1))
        network = TFNetwork(config=config)
        network.construct_from_dict(net_dict)
        from test_TFNetworkLayer import make_feed_dict

        session.run(
            network.get_default_output_layer().output.placeholder, feed_dict=make_feed_dict(network.extern_data)
        )


def test_reclayer_scalar_size():
    with make_scope() as session:
        net_dict = {
            "len": {"class": "length", "from": "data"},
            "max_len": {"class": "reduce", "from": "len", "mode": "max", "axis": "B", "out_shape": ()},
            "range": {"class": "range_from_length", "from": "max_len"},
            "rec": {
                "class": "rec",
                "from": "range",
                "unit": {
                    "add": {
                        "class": "combine",
                        "kind": "add",
                        "from": ["data:source", "prev:add"],
                    },
                    "output": {"class": "copy", "from": "add"},
                },
            },
            "output": {"class": "copy", "from": "rec"},
        }
        config = Config({"extern_data": {"data": {"shape": (None, 3)}}})
        network = TFNetwork(config=config)
        network.construct_from_dict(net_dict)
        from test_TFNetworkLayer import make_feed_dict

        session.run(
            network.get_default_output_layer().output.placeholder, feed_dict=make_feed_dict(network.extern_data)
        )


def test_reclayer_scalar_size_last():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time")
    feat_dim = FeatureDim("feat", 5)
    config = Config(
        dict(
            extern_data={"data": {"dim_tags": (batch_dim, time_dim, feat_dim)}},
            debug_runtime_sanity_checks=True,
        )
    )

    top_k_dim = SpatialDim("top-k-dim")

    net_dict = {
        "test_specaugment_v2_name_scope_simplify": {
            "class": "subnetwork",
            "from": [],
            "subnetwork": {
                "random": {
                    "class": "subnetwork",
                    "from": [],
                    "subnetwork": {
                        "random": {
                            "class": "random",
                            "shape": [batch_dim],
                            "distribution": "uniform",
                            "minval": 1,
                            "maxval": 3,
                            "dtype": "int32",
                        },
                        "output": {"class": "copy", "from": "random", "out_shape": {batch_dim}},
                    },
                    "out_shape": {batch_dim},
                },
                "random_0": {
                    "class": "subnetwork",
                    "from": [],
                    "subnetwork": {
                        "random": {
                            "class": "random",
                            "shape": [batch_dim, feat_dim],
                            "distribution": "uniform",
                            "minval": 0.0,
                            "maxval": 1.0,
                        },
                        "output": {"class": "copy", "from": "random", "out_shape": {batch_dim, feat_dim}},
                    },
                    "out_shape": {batch_dim, feat_dim},
                },
                "reduce": {"class": "reduce", "from": "random", "mode": "max", "axis": (batch_dim,), "out_shape": {}},
                "top_k": {
                    "class": "top_k",
                    "from": "random_0",
                    "axis": feat_dim,
                    "k": "reduce",
                    "k_dim": top_k_dim,
                    "sorted": True,
                    "out_shape": {batch_dim, top_k_dim},
                },
                "loop": {
                    "class": "rec",
                    "from": [],
                    "unit": {
                        "state.x": {
                            "class": "copy",
                            "from": "test_specaugment_v2_name_scope_simplify._relu_0",
                            "initial_output": "base:base:data:data",
                            "out_shape": {batch_dim, time_dim, feat_dim},
                        },
                        "Loop.unstack": {
                            "class": "rec_unstack",
                            "from": "base:range_in_axis",
                            "axis": top_k_dim,
                            "out_shape": {},
                        },
                        "output": {"class": "copy", "from": "Loop.unstack", "out_shape": {}},
                        "test_specaugment_v2_name_scope_simplify._relu_0": {
                            "class": "activation",
                            "from": "prev:state.x",
                            "activation": "relu",
                            "need_last": True,
                            "out_shape": {batch_dim, time_dim, feat_dim},
                        },
                    },
                    "axis": top_k_dim,
                    "out_shape": {top_k_dim},
                    "name_scope": "",
                },
                "range_in_axis": {
                    "class": "range_in_axis",
                    "from": "top_k/indices",
                    "axis": top_k_dim,
                    "out_shape": {top_k_dim},
                },
                "test_specaugment_v2_name_scope_simplify._relu_0": {
                    "class": "rec_last_output",
                    "rec_layer": "loop",
                    "sub_layer_name": "test_specaugment_v2_name_scope_simplify._relu_0",
                    "out_shape": {batch_dim, time_dim, feat_dim},
                },
                "output": {
                    "class": "copy",
                    "from": "test_specaugment_v2_name_scope_simplify._relu_0",
                    "out_shape": {batch_dim, time_dim, feat_dim},
                },
            },
            "out_shape": {batch_dim, time_dim, feat_dim},
        },
        "output": {
            "class": "copy",
            "from": "test_specaugment_v2_name_scope_simplify",
            "out_shape": {batch_dim, time_dim, feat_dim},
        },
    }

    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)
        out = net.get_default_output_layer().output
        net.initialize_params(session)
        from test_TFNetworkLayer import make_feed_dict

        session.run((out.placeholder, out.get_sequence_lengths()), feed_dict=make_feed_dict(net.extern_data))


def test_reclayer_shape_from_initial():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time")
    input_dim = FeatureDim("input", 13)
    loop_dim = SpatialDim("loop-dim")

    config = Config({"extern_data": {"data": {"dim_tags": (batch_dim, time_dim, input_dim)}}})

    net_dict = {
        "output": {"class": "copy", "from": "loop/output", "out_shape": {batch_dim, input_dim, loop_dim}},
        "constant": {"class": "constant", "value": 10},
        "loop": {
            "class": "rec",
            "from": [],
            "unit": {
                "add": {
                    "class": "eval",
                    "from": "prev:add",
                    "eval": "source(0) + 1.0",
                    "initial_output": "base:zeros",
                    "out_shape": {batch_dim},
                },
                "greater_equal": {
                    "class": "compare",
                    "from": "add",
                    "kind": "greater_equal",
                    "value": 5.0,
                    "out_shape": {batch_dim},
                },
                "end": {"class": "copy", "from": "greater_equal", "out_shape": {batch_dim}},
                "reduce": {
                    "class": "reduce",
                    "from": "base:data:data",
                    "mode": "mean",
                    "axis": time_dim,
                    "out_shape": {batch_dim, input_dim},
                },
                "mul": {
                    "class": "combine",
                    "from": ["add", "reduce"],
                    "kind": "mul",
                    "out_shape": {batch_dim, input_dim},
                },
                "output": {"class": "copy", "from": "mul", "out_shape": {batch_dim, input_dim}},
            },
            "max_seq_len_via": "constant",
            "axis": loop_dim,
            "include_eos": True,
            "out_shape": {batch_dim, input_dim, loop_dim},
            "name_scope": "",
        },
        "zeros": {"class": "constant", "value": 0, "shape": [batch_dim], "dtype": "float32"},
    }

    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)
        out = net.get_default_output_layer().output
        from test_TFNetworkLayer import make_feed_dict

        session.run(out.placeholder, feed_dict=make_feed_dict(net.extern_data))


def test_reclayer_time_sync_target_diff():
    # https://github.com/rwth-i6/returnn/issues/1140
    from returnn.util.basic import BehaviorVersion
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim
    from returnn.tf.layers.rec import _SubnetworkRecCell

    src_dim = FeatureDim("src-feat", 5)
    tgt_dim = FeatureDim("tgt-classes", 7)
    tgt_with_blank_dim = tgt_dim + 1
    src_time_dim = SpatialDim("src-time")
    tgt_time_dim = SpatialDim("out-spatial")

    config = Config(
        {
            "extern_data": {
                "data": {"dim_tags": [batch_dim, src_time_dim, src_dim]},
                "classes": {
                    "dim_tags": [batch_dim, tgt_time_dim],
                    "sparse_dim": tgt_dim,
                    "available_for_inference": False,
                },
                "align_classes": {
                    "dim_tags": [batch_dim, src_time_dim],
                    "sparse_dim": tgt_with_blank_dim,
                    "available_for_inference": False,
                },
            },
            "network": {
                "encoder": {"class": "linear", "activation": "tanh", "n_out": 5, "from": "data:data"},
                "output": {
                    "class": "rec",
                    "from": "encoder",
                    "unit": {
                        "output_prob": {"class": "softmax", "from": "data:source", "out_dim": tgt_with_blank_dim},
                        # Note: This is actually not correct to have 'classes' here.
                        # In practice, in search, it would use output_prob and then have actually one more label.
                        # classes also has the wrong spatial dim, which actually causes the error.
                        # However, then this output is actually never used.
                        # We had such training configs for transducer, and we want to make sure that they still work.
                        # In that case, in search, the config switched to a different target, so that is why it worked.
                        "output": {
                            "class": "choice",
                            "target": "classes",
                            "beam_size": 12,
                            "from": "output_prob",
                            "initial_output": 0,
                        },
                        # Would also look different for recognition.
                        "classes_embed": {
                            "class": "linear",
                            "activation": "tanh",
                            "n_out": 5,
                            "from": "base:data:classes",
                        },
                        "joint": {
                            "class": "combine",
                            "from": ["output_prob", "classes_embed"],
                            "kind": "mul",
                            "allow_broadcast_all_sources": True,
                        },
                        # Dummy loss. In transducer, this would be the full-sum after joint network.
                        # Here we just need sth to trigger the dependencies.
                        "loss": {
                            "class": "eval",
                            "from": "joint",
                            "eval": "tf.reduce_mean(source(0,auto_convert=False))",
                            "out_type": {
                                "shape": (),
                                "dtype": "float32",
                                "batch_dim_axis": None,
                                "time_dim_axis": None,
                            },
                            "loss": "as_is",
                        },
                    },
                    "target": "classes",
                },
            },
        }
    )

    print("Constructing train network (old behavior).")
    with make_scope() as session:
        net = TFNetwork(train_flag=True, config=config)
        orig_behavior_version = BehaviorVersion._behavior_version
        try:
            BehaviorVersion._behavior_version = 0
            # The net dict requires an older behavior version. This is important for the test.
            # We want to make sure such old config still works.
            net.construct_from_dict(config.typed_value("network"))
        finally:
            BehaviorVersion._behavior_version = orig_behavior_version
        # Check whether we triggered the dim tag bug.
        assert src_time_dim != tgt_time_dim
        net.initialize_params(session)
        rec_layer = net.get_layer("output")
        assert isinstance(rec_layer, RecLayer)
        cell = rec_layer.cell
        assert isinstance(cell, _SubnetworkRecCell)
        assert_equal(cell.layers_in_loop, [])
        loss = net.get_total_loss()
        from test_TFNetworkLayer import make_feed_dict

        loss_v = session.run(loss, feed_dict=make_feed_dict(net.extern_data))
        print("Loss:", loss_v)

    print("Constructing train network (new behavior).")
    with make_scope():
        net = TFNetwork(train_flag=True, config=config)
        try:
            net.construct_from_dict(config.typed_value("network"))
        except BehaviorVersion.RequirementNotSatisfied as exc:
            assert "time-dim-tag mismatch" in str(exc)
            print("Got expected exception:", exc)
        else:
            raise Exception("did not get expected exception")


def test_convert_lstm_params_save_load():
    """
    Test conversions from different units to different units.
    """
    n_in, n_hidden0, n_hidden1, n_hidden2, n_out = 2, 5, 7, 11, 3

    def make_config(lstm_unit):
        """
        :param str lstm_unit:
        :rtype: Config
        """
        unit_opts = {}
        if lstm_unit.lower() in {"standardlstm", "basiclstm", "lstmblock", "lstmblockfused"}:
            # Other units would have forget bias 0.0 by default, or add it to the param at param initialization.
            # Thus explicitly use this for the calculation.
            unit_opts["forget_bias"] = 0.0
        net_dict = {
            "input": {
                "class": "linear",
                "n_out": n_hidden0,
                "activation": None,
                "with_bias": False,
                "from": "data:data",
            },
            "lstm1": {"class": "rec", "unit": lstm_unit, "unit_opts": unit_opts, "n_out": n_hidden1, "from": "input"},
            "lstm2": {"class": "rec", "unit": lstm_unit, "unit_opts": unit_opts, "n_out": n_hidden2, "from": "lstm1"},
            "output": {"class": "softmax", "n_out": n_out, "from": "lstm2"},
        }
        return Config({"num_outputs": n_out, "num_inputs": n_in, "network": net_dict})

    import tempfile

    model_tmp_dir = tempfile.mkdtemp("tmp-checkpoint")
    input_np = [[[0.7, 0.1], [-0.3, -0.1], [0.2, -0.1]], [[1.0, -0.4], [-0.2, 0.3], [0.0, 0.0]]]
    input_np = numpy.array(input_np, dtype="float32")
    input_seq_lens = [3, 2]
    n_batch = len(input_seq_lens)
    assert_equal(input_np.shape, (n_batch, max(input_seq_lens), n_in))

    def construct_load_save_forward(lstm_unit, prev_lstm_unit=None, output_ref=None):
        """
        :param str lstm_unit:
        :param str|None prev_lstm_unit:
        :param numpy.ndarray|None output_ref:
        """
        with make_scope() as session:
            config = make_config(lstm_unit)
            network = TFNetwork(config=config, train_flag=True)
            network.construct_from_dict(config.typed_dict["network"])
            if prev_lstm_unit:
                print("*** Load params from prev NN with LSTM unit %r -> %r." % (prev_lstm_unit, lstm_unit))
                network.load_params_from_file(filename="%s/model.%s" % (model_tmp_dir, prev_lstm_unit), session=session)
            else:
                print("*** Randomly initialize params.")
                network.initialize_params(session)
            network.save_params_to_file(filename="%s/model.%s" % (model_tmp_dir, lstm_unit), session=session)

            input_placeholder = network.extern_data.data["data"].placeholder
            input_seq_lens_placeholder = network.extern_data.data["data"].size_placeholder[0]
            output_layer = network.get_default_output_layer(must_exist=True)
            output_np, output_seq_lens = session.run(
                (output_layer.output.get_placeholder_as_batch_major(), output_layer.output.get_sequence_lengths()),
                feed_dict={
                    network.extern_data.get_batch_info().dim: len(input_seq_lens),
                    input_placeholder: input_np,
                    input_seq_lens_placeholder: input_seq_lens,
                },
            )
            assert_equal(list(output_seq_lens), input_seq_lens)
            assert_equal(output_np.shape, (n_batch, max(input_seq_lens), n_out))
            for t in range(max(output_seq_lens)):
                for b in range(n_batch):
                    if t >= output_seq_lens[b]:
                        output_np[b, t] = 0.0
            if output_ref is not None:
                assert_allclose(
                    output_ref,
                    output_np,
                    rtol=1e-5,
                    err_msg="Output not same after converting %s -> %s" % (prev_lstm_unit, lstm_unit),
                )
                print("*** All outputs match when converting %r -> %r." % (prev_lstm_unit, lstm_unit))
            return output_np

    output_ref = construct_load_save_forward(lstm_unit="lstm")  # using default LSTM

    for src_lstm_unit, tgt_lstm_unit in [
        ("lstm", "nativelstm"),
        ("nativelstm", "nativelstm2"),
        ("nativelstm2", "standardlstm"),
        ("nativelstm2", "basiclstm"),
        ("standardlstm", "lstmblock"),
        ("lstmblock", "standardlstm"),
        ("standardlstm", "nativelstm2"),
    ]:
        construct_load_save_forward(lstm_unit=tgt_lstm_unit, prev_lstm_unit=src_lstm_unit, output_ref=output_ref)


def test_KenLmStateLayer():
    import returnn.tf.util.ken_lm as tf_ken_lm

    if not tf_ken_lm.kenlm_checked_out():
        raise unittest.SkipTest("KenLM not checked out")
    tf_ken_lm.get_tf_mod(verbose=True)
    test_lm_file = tf_ken_lm.kenlm_dir + "/lm/test.arpa"
    assert os.path.exists(test_lm_file)
    from returnn.datasets.util.vocabulary import Vocabulary
    from returnn.tf.layers.base import InternalLayer
    import tempfile

    with make_scope() as session:
        with tempfile.NamedTemporaryFile(mode="w", prefix="vocab") as tmp_bpe_vocab_file:
            labels = "</s> <unk> be@@ yond imm@@ edi@@ ate conc@@ erns".split()
            bpe_vocab_dict = Vocabulary.create_vocab_dict_from_labels(labels)
            print("BPE vocab dict:", bpe_vocab_dict)
            tmp_bpe_vocab_file.write(repr(bpe_vocab_dict))
            tmp_bpe_vocab_file.flush()
            assert os.path.exists(tmp_bpe_vocab_file.name)

            net = TFNetwork(extern_data=ExternData())
            net.extern_data.register_data(
                Data(
                    name="data",
                    shape=(),
                    time_dim_axis=None,
                    dim=len(labels),
                    sparse=True,
                    auto_create_placeholders=True,
                )
            )
            data_layer = net.construct_layer(name="data", net_dict={})
            layer_base_opts = dict(name="output", network=net, sources=[data_layer])
            layer_out = KenLmStateLayer.get_out_data_from_opts(**layer_base_opts)
            rec_state = session.run(
                KenLmStateLayer.get_rec_initial_extra_outputs(batch_dim=1, rec_layer=None, **layer_base_opts)
            )
            print("initial recurrent state:", rec_state)
            assert isinstance(rec_state, dict)
            prev_layer = InternalLayer(name="prev:%s" % layer_base_opts["name"], network=net, output=layer_out.copy())
            prev_layer.rec_vars_outputs = {
                k: tf_compat.v1.placeholder(name="prev_layer_%s" % k, shape=v.shape, dtype=v.dtype)
                for (k, v) in rec_state.items()
            }
            with reuse_name_scope(KenLmStateLayer.cls_get_tf_scope_name(layer_base_opts["name"])):
                layer = KenLmStateLayer(
                    lm_file=test_lm_file,
                    vocab_file=tmp_bpe_vocab_file.name,
                    vocab_unknown_label="<unk>",
                    bpe_merge_symbol="@@",
                    output=layer_out,
                    rec_previous_layer=prev_layer,
                    **layer_base_opts,
                )
                net.layers[layer.name] = layer

            print("Init.")
            net.initialize_params(session=session)

            print("Ref score.")
            input_word_ids = [labels.index(w) for w in "be@@ yond imm@@ edi@@ ate conc@@ erns </s>".split()]
            ref_score_str_placeholder = tf_compat.v1.placeholder(tf.string, shape=(), name="ref_score_str_placeholder")
            tf_ref_score = tf_ken_lm.ken_lm_abs_score_strings(handle=layer.lm_handle, strings=ref_score_str_placeholder)
            ref_score = session.run(
                tf_ref_score, feed_dict={ref_score_str_placeholder: "beyond immediate concerns </s>"}
            )
            print("ref score:", ref_score)
            assert_almost_equal(ref_score, -9.251298)  # example from :func:`test_kenlm`

            print("Loop over %r." % ([labels[i] for i in input_word_ids],))
            abs_score = 0.0
            for i, word_id in enumerate(input_word_ids):
                print("input %i, word-idx %i, word %r" % (i, word_id, labels[word_id]))
                feed_dict = {net.extern_data.data["data"].placeholder: [word_id]}
                feed_dict.update({prev_layer.rec_vars_outputs[p]: v for (p, v) in rec_state.items()})
                rel_score_res, rec_state = session.run(
                    (layer.output.placeholder, layer.rec_vars_outputs), feed_dict=feed_dict
                )
                print("  score rel res:", rel_score_res, "state:", rec_state)
                abs_score += rel_score_res[0]
                print("  abs score:", abs_score)
                word_seq_so_far = rec_state["state"][0].decode("utf8").replace("@@ ", "").strip().split(" ")
                word_seq_so_far = ["<unk>" if "@@" in w else w for w in word_seq_so_far]
                res2 = session.run(tf_ref_score, feed_dict={ref_score_str_placeholder: " ".join(word_seq_so_far)})
                print("  word seq so far: %r" % (word_seq_so_far,), "score:", res2)
                assert_equal(res2, abs_score)

            assert_almost_equal(abs_score, ref_score)
            print("Scores are as expected.")


def test_KenLmStateLayer_dense():
    import returnn.tf.util.ken_lm as tf_ken_lm

    if not tf_ken_lm.kenlm_checked_out():
        raise unittest.SkipTest("KenLM not checked out")
    tf_ken_lm.get_tf_mod(verbose=True)
    test_lm_file = tf_ken_lm.kenlm_dir + "/lm/test.arpa"
    assert os.path.exists(test_lm_file)
    from returnn.datasets.util.vocabulary import Vocabulary
    from returnn.tf.layers.base import InternalLayer
    import tempfile

    with make_scope() as session:
        with tempfile.NamedTemporaryFile(mode="w", prefix="vocab") as tmp_bpe_vocab_file:
            labels = "</s> <unk> be@@ yond imm@@ edi@@ ate conc@@ erns".split()
            bpe_vocab_dict = Vocabulary.create_vocab_dict_from_labels(labels)
            print("BPE vocab dict:", bpe_vocab_dict)
            tmp_bpe_vocab_file.write(repr(bpe_vocab_dict))
            tmp_bpe_vocab_file.flush()
            assert os.path.exists(tmp_bpe_vocab_file.name)

            net = TFNetwork(extern_data=ExternData())
            net.extern_data.register_data(
                Data(
                    name="data",
                    shape=(),
                    time_dim_axis=None,
                    dim=len(labels),
                    sparse=True,
                    auto_create_placeholders=True,
                )
            )
            data_layer = net.construct_layer(name="data", net_dict={})
            layer_base_opts = dict(
                name="output",
                network=net,
                sources=[data_layer],
                lm_file=test_lm_file,
                vocab_file=tmp_bpe_vocab_file.name,
                vocab_unknown_label="<unk>",
                bpe_merge_symbol="@@",
                input_step_offset=1,
                dense_output=True,
            )
            layer_out = KenLmStateLayer.get_out_data_from_opts(**layer_base_opts)
            batch_dim = 1
            rec_state = session.run(
                KenLmStateLayer.get_rec_initial_extra_outputs(batch_dim=batch_dim, rec_layer=None, **layer_base_opts)
            )
            print("initial recurrent state:", rec_state)
            assert isinstance(rec_state, dict)
            prev_layer = InternalLayer(name="prev:%s" % layer_base_opts["name"], network=net, output=layer_out.copy())
            prev_layer.rec_vars_outputs = {
                k: tf_compat.v1.placeholder(name="prev_layer_%s" % k, shape=v.shape, dtype=v.dtype)
                for (k, v) in rec_state.items()
            }
            with reuse_name_scope(KenLmStateLayer.cls_get_tf_scope_name(layer_base_opts["name"])):
                layer = KenLmStateLayer(output=layer_out, rec_previous_layer=prev_layer, **layer_base_opts)
                net.layers[layer.name] = layer

            print("Init.")
            net.initialize_params(session=session)

            print("Ref score.")
            input_word_ids = [labels.index(w) for w in "be@@ yond imm@@ edi@@ ate conc@@ erns </s>".split()]
            ref_score_str_placeholder = tf_compat.v1.placeholder(tf.string, shape=(), name="ref_score_str_placeholder")
            tf_ref_score = tf_ken_lm.ken_lm_abs_score_strings(handle=layer.lm_handle, strings=ref_score_str_placeholder)
            ref_score = session.run(
                tf_ref_score, feed_dict={ref_score_str_placeholder: "beyond immediate concerns </s>"}
            )
            print("ref score:", ref_score)
            assert_almost_equal(ref_score, -9.251298)  # example from :func:`test_kenlm`

            print("Loop over %r." % ([labels[i] for i in input_word_ids],))
            abs_score = 0.0
            for i in range(len(input_word_ids)):
                if i == 0:
                    word_id = 0
                    word = ""
                else:
                    word_id = input_word_ids[i - 1]
                    word = labels[word_id]
                next_word_id = input_word_ids[i]
                next_word = labels[next_word_id]
                print(
                    "input %i, word-idx %i, word %r, next-word-idx %i, next-word %r"
                    % (i, word_id, word, next_word_id, next_word)
                )
                feed_dict = {
                    net.extern_data.get_batch_info().dim: 1,
                    net.extern_data.data["data"].placeholder: [word_id],
                }
                feed_dict.update({prev_layer.rec_vars_outputs[p]: v for (p, v) in rec_state.items()})
                rel_score_res, rec_state = session.run(
                    (layer.output.placeholder, layer.rec_vars_outputs), feed_dict=feed_dict
                )
                print("  score rel res:", rel_score_res)
                print("  state:", rec_state)
                assert rel_score_res.shape == (batch_dim, layer.vocab.num_labels)
                abs_score += rel_score_res[0][next_word_id]
                print("  abs score:", abs_score)
                word_seq_so_far = (
                    (rec_state["state"][0].decode("utf8") + next_word).replace("@@ ", "").strip().split(" ")
                )
                word_seq_so_far = ["<unk>" if "@@" in w else w for w in word_seq_so_far]
                res2 = session.run(tf_ref_score, feed_dict={ref_score_str_placeholder: " ".join(word_seq_so_far)})
                print("  word seq so far: %r" % (word_seq_so_far,), "score:", res2)
                assert_equal(res2, abs_score)

            assert_almost_equal(abs_score, ref_score)
            print("Scores are as expected.")


@unittest.skipIf(not is_gpu_available(), "no gpu on this system")
def test_BlocksparseLSTM_load_params_from_native_lstm():
    from returnn.tf.native_op import have_blocksparse_requirements, init_blocksparse

    if not have_blocksparse_requirements():
        raise unittest.SkipTest("no blocksparse requirements")
    init_blocksparse()

    random = numpy.random.RandomState(seed=1)
    num_inputs = 32
    num_outputs = 63
    num_outputs_sparse = 256
    batch_dim = 8
    seq_len = 5

    with make_scope() as session:
        print("create graph")
        tf_compat.v1.set_random_seed(42)
        src_placeholder = tf_compat.v1.placeholder(tf.float32, (batch_dim, seq_len, num_inputs), name="src_placeholder")
        seq_len_placeholder = tf_compat.v1.placeholder(tf.int32, (batch_dim,), name="seq_len_placeholder")
        feed_dict = {
            src_placeholder: random.uniform(-1.0, 1.0, (batch_dim, seq_len, num_inputs)),
            seq_len_placeholder: [seq_len] * batch_dim,
        }

        from returnn.tf.util.basic import xavier_initializer

        default_var_initializer = xavier_initializer(seed=13)
        with tf_compat.v1.variable_scope(
            tf_compat.v1.get_variable_scope(), initializer=default_var_initializer
        ) as scope:
            net = TFNetwork(config=Config(), extern_data=ExternData(), train_flag=False)
            with net.register_network_scope():
                from returnn.tf.layers.base import InternalLayer

                src_layer = InternalLayer(
                    name="src",
                    network=net,
                    output=Data(
                        "src",
                        shape=(None, num_inputs),
                        placeholder=src_placeholder,
                        size_placeholder={0: seq_len_placeholder},
                    ),
                )
                print("source layer:", src_layer)
                with tf.name_scope("nativelstm"):
                    layer1 = RecLayer(
                        name="nativelstm",
                        network=net,
                        output=Data("out", shape=(None, num_outputs), time_dim_axis=0, batch_dim_axis=1),
                        sources=[src_layer],
                        unit="NativeLSTM2",
                    )
                with tf.name_scope("blocksparselstm"):
                    layer2 = RecLayer(
                        name="blocksparselstm",
                        network=net,
                        output=Data("out", shape=(None, num_outputs_sparse), time_dim_axis=0, batch_dim_axis=1),
                        sources=[src_layer],
                        unit="BlocksparseLSTM",
                        unit_opts={"seed": 5, "connectivity": 1, "connectivity_dense": 2, "layer_norm": False},
                    )
                y1 = layer1.output.get_placeholder_as_batch_major()
                y2 = layer2.output.get_placeholder_as_batch_major()

        print("run")
        session.run(tf_compat.v1.global_variables_initializer())
        native_lstm_params = layer1.get_param_values_dict(session=session)
        np_y1 = session.run(y1, feed_dict=feed_dict)
        assert np_y1.shape == (batch_dim, seq_len, num_outputs)
        print("native output:")
        print(np_y1)
        bsmm_cell = layer2.cell
        assert isinstance(bsmm_cell, BlocksparseLSTMCell)
        for param in layer2.params.values():
            print("blocksparse LSTM param:", param)
            assert isinstance(param, tf.Variable)
            param.load(numpy.zeros(param.get_shape().as_list(), dtype="float32"), session=session)
        bsmm_cell.load_params_from_native_lstm(native_lstm_params, session=session)
        np_y2 = session.run(y2, feed_dict=feed_dict)
        assert np_y2.shape == (batch_dim, seq_len, num_outputs_sparse)
        np_y2 = np_y2[:, :, :num_outputs]
        assert_almost_equal(np_y1, np_y2)


def test_rec_layer_search_select_src_reuse_layer():
    from returnn.tf.layers.rec import _SubnetworkRecCell

    n_src_dim = 7
    n_tgt_dim = 7
    beam_size = 12
    config = Config()
    config.update({"debug_print_layer_output_template": True, "optimize_move_layers_out": False})

    def get_net_dict():
        return {
            "source_embed": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "n_out": 6,
                "from": "data:data",
            },
            "lstm0_fw": {
                "class": "rec",
                "unit": "standardlstm",
                "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                "initial_state": "var",
                "n_out": 10,
                "direction": 1,
                "from": ["source_embed"],
            },
            "lstm0_bw": {
                "class": "rec",
                "unit": "standardlstm",
                "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                "initial_state": "var",
                "n_out": 10,
                "direction": -1,
                "from": ["source_embed"],
            },
            "lstm1_fw": {
                "class": "rec",
                "unit": "standardlstm",
                "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                "initial_state": "var",
                "n_out": 10,
                "direction": 1,
                "from": ["lstm0_fw", "lstm0_bw"],
            },
            "lstm1_bw": {
                "class": "rec",
                "unit": "standardlstm",
                "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                "initial_state": "var",
                "n_out": 10,
                "direction": -1,
                "from": ["lstm0_fw", "lstm0_bw"],
            },
            "encoder": {"class": "copy", "from": ["lstm1_fw", "lstm1_bw"]},
            "enc_ctx": {"class": "linear", "activation": None, "with_bias": True, "from": ["encoder"], "n_out": 10},
            "fertility": {
                "class": "linear",
                "activation": "sigmoid",
                "with_bias": False,
                "from": ["encoder"],
                "n_out": 1,
            },
            "output": {
                "class": "rec",
                "from": [],
                "unit": {
                    "output": {
                        "class": "choice",
                        "target": "classes",
                        "beam_size": beam_size,
                        "from": ["output_prob"],
                        "initial_output": 0,
                    },
                    "end": {"class": "compare", "from": ["output"], "value": 0},
                    "target_embed": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["output"],
                        "n_out": 6,
                        "initial_output": "apply(0)",
                        "reuse_params": {"map": {"W": {"reuse_layer": "base:source_embed"}, "b": None}},
                    },
                    "weight_feedback": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["prev:accum_att_weights"],
                        "n_out": 10,
                    },
                    "prev_s_state": {"class": "get_last_hidden_state", "from": ["prev:s"], "n_out": 20},
                    "prev_s_transformed": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["prev_s_state"],
                        "n_out": 10,
                    },
                    "energy_in": {
                        "class": "combine",
                        "kind": "add",
                        "from": ["base:enc_ctx", "weight_feedback", "prev_s_transformed"],
                        "n_out": 10,
                    },
                    "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
                    "energy": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["energy_tanh"],
                        "n_out": 1,
                    },
                    "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, enc-T, 1)
                    "accum_att_weights": {
                        "class": "eval",
                        "from": ["prev:accum_att_weights", "att_weights", "base:fertility"],
                        "eval": "source(0) + source(1) / (2.0 * source(2))",
                        "out_type": {"dim": 1, "shape": (None, 1)},
                    },
                    "att": {
                        "class": "generic_attention",
                        "weights": "att_weights",
                        "base": "base:encoder",
                        "auto_squeeze": True,
                    },
                    "s": {
                        "class": "rnn_cell",
                        "unit": "standardlstm",
                        "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                        "initial_state": "var",
                        "from": ["target_embed", "att"],
                        "n_out": 10,
                    },
                    "readout_in": {
                        "class": "linear",
                        "from": ["prev:s", "prev:target_embed", "att"],
                        "activation": None,
                        "n_out": 10,
                    },
                    "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
                    "output_prob": {"class": "softmax", "from": ["readout"], "target": "classes", "loss": "ce"},
                },
                "target": "classes",
                "max_seq_len": 20,
            },
            "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance", "target": "classes"},
        }

    print("Constructing search network.")
    with make_scope() as session:
        extern_data = ExternData(
            {
                "data": {"dim": n_src_dim, "sparse": True},
                "classes": {"dim": n_tgt_dim, "sparse": True, "available_for_inference": False},
            }
        )
        search_net = TFNetwork(
            extern_data=extern_data, search_flag=True, train_flag=False, eval_flag=True, config=config
        )
        search_net.construct_from_dict(get_net_dict())
        search_out_layer = search_net.layers["output"]
        assert isinstance(search_out_layer, RecLayer)
        assert isinstance(search_out_layer.cell, _SubnetworkRecCell)
        assert not search_out_layer.cell.input_layers_moved_out
        assert not search_out_layer.cell.output_layers_moved_out
        print("Layers in the loop:")
        loop_net = search_out_layer.cell.net
        for name, layer in sorted(loop_net.layers.items()):
            print("  %r: %s" % (name, layer))
            print("    search choices:", layer.get_search_choices())
            print("    sources:")
            for src in layer.sources:
                print("      %s" % src)
            print("    other deps:")
            for dep in layer.get_dep_layers():
                if dep in layer.sources:
                    continue
                print("      %s" % dep)
        loop_out_layer = loop_net.layers["output"]
        assert isinstance(loop_out_layer, ChoiceLayer)
        assert isinstance(loop_out_layer.search_choices, SearchChoices)
        all_src_choices = loop_out_layer.search_choices.get_src_choices_seq()
        assert len(all_src_choices) == 2
        cur_out_choice, prev_out_choice = all_src_choices
        assert isinstance(cur_out_choice, SearchChoices)
        assert isinstance(prev_out_choice, SearchChoices)
        assert cur_out_choice == loop_out_layer.search_choices
        prev_loop_out_layer = loop_net.layers["prev:output"]
        assert prev_out_choice == prev_loop_out_layer.search_choices
        assert RecLayer.is_prev_step_layer(prev_out_choice.owner)
        assert_equal(loop_net.layers["end"].get_search_choices(), cur_out_choice)
        assert_equal(loop_net.layers["target_embed"].get_search_choices(), cur_out_choice)
        assert_equal(loop_net.layers["prev:target_embed"].get_search_choices(), prev_out_choice)
        assert_equal(loop_net.layers["accum_att_weights"].get_search_choices(), prev_out_choice)
        assert_equal(
            loop_net.layers["prev:accum_att_weights"].get_search_choices(), prev_out_choice
        )  # will be transformed
        assert_equal(loop_net.layers["weight_feedback"].get_search_choices(), prev_out_choice)
        assert_equal(loop_net.layers["s"].get_search_choices(), cur_out_choice)
        assert_equal(loop_net.layers["prev:s"].get_search_choices(), prev_out_choice)
        assert_equal(loop_net.layers["prev_s_state"].get_search_choices(), prev_out_choice)
        assert_equal(loop_net.layers["energy_in"].get_search_choices(), prev_out_choice)
        assert_equal(loop_net.layers["att_weights"].get_search_choices(), prev_out_choice)
        assert_equal(loop_net.layers["att"].get_search_choices(), prev_out_choice)
        assert_equal(loop_net.layers["output_prob"].get_search_choices(), prev_out_choice)


def test_onlineblstm():
    network = {}
    lstm_dim = 13
    lstm_window = 5

    def add_lstm(i, direction, src):
        name = "lstm%i_%s" % (i, {1: "fw", -1: "bw"}[direction])
        if direction > 0:
            network[name] = {
                "class": "rec",
                "unit": "lstmp",
                "n_out": lstm_dim,
                "dropout": 0.1,
                "L2": 0.01,
                "direction": 1,
                "from": src,
            }
            return name
        network["%s_win" % name] = {
            "class": "window",
            "window_size": lstm_window,
            "window_right": lstm_window - 1,
            "from": src,
        }  # (B,T,W,D)
        network["%s_mdims" % name] = {"class": "merge_dims", "axes": ["B", "T"], "from": ["%s_win" % name]}  # (B*T,W,D)
        network["%s_rdims" % name] = {
            "class": "reinterpret_data",
            "enforce_batch_major": True,
            "set_axes": {"T": "spatial"},
            "from": ["%s_mdims" % name],
        }  # (B*T,W,D)
        network["%s_rec" % name] = {
            "class": "rec",
            "unit": "lstmp",
            "n_out": lstm_dim,
            "dropout": 0.1,
            "L2": 0.01,
            "direction": -1,
            "from": ["%s_rdims" % name],
        }  # (B*T,W,D')
        network["%s_cur" % name] = {
            "class": "slice",
            "axis": "T",
            "slice_end": 1,
            "from": ["%s_rec" % name],
        }  # (B*T,1,D')
        network["%s_cursq" % name] = {"class": "squeeze", "axis": "T", "from": ["%s_cur" % name]}  # (B*T,D')
        network["%s_res" % name] = {
            "class": "split_batch_time",
            "base": src[0],
            "from": ["%s_cursq" % name],
        }  # (B,T,D')
        return "%s_res" % name

    num_layers = 6
    src = ["data"]
    for i in range(num_layers):
        fwd = add_lstm(i, 1, src)
        bwd = add_lstm(i, -1, src)
        src = [fwd, bwd]
    # Focal Loss, https://arxiv.org/abs/1708.02002
    network["output"] = {"class": "softmax", "loss": "ce", "loss_opts": {"focal_loss_factor": 2.0}, "from": src}
    config = Config({"num_inputs": 3, "num_outputs": 7})
    with make_scope() as session:
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(network)


def test_GenericAttentionLayer_basic0():
    from returnn.tf.layers.base import InternalLayer

    net = TFNetwork(extern_data=ExternData({"data": {"shape": (None, 5)}}), config=Config())
    time = SpatialDim("time")
    kwargs = dict(
        name="att",
        network=net,
        auto_squeeze=True,
        weights=InternalLayer(
            name="att_weights",
            network=net,
            output=Data(
                name="att_weights_output", shape=(None, 1), auto_create_placeholders=True, same_dim_tags_as={"T": time}
            ),
        ),
        base=InternalLayer(
            name="enc_value",
            network=net,
            output=Data(
                name="enc_value_output", shape=(None, 20), auto_create_placeholders=True, same_dim_tags_as={"T": time}
            ),
        ),
    )
    print("GenericAttentionLayer kwargs:")
    pprint(kwargs)
    kwargs["output"] = GenericAttentionLayer.get_out_data_from_opts(**kwargs)
    layer = GenericAttentionLayer(**kwargs)
    layer.output.sanity_check()
    assert layer.output.shape == (20,) and not layer.output.have_time_axis()


def test_GenericAttentionLayer_basic():
    from returnn.tf.layers.base import InternalLayer

    net = TFNetwork(extern_data=ExternData({"data": {"shape": (None, 5)}}), config=Config())
    # This is a common situation when the GenericAttentionLayer is inside a recurrent loop,
    # and it gets the encoder values from outside ("base:enc_value" or so),
    # and the attention weights from inside the loop, and they have the same time dim axis as the encoder values.
    time = SpatialDim("time")
    kwargs = dict(
        name="att",
        network=net,
        auto_squeeze=True,
        weights=InternalLayer(
            name="att_weights",
            network=net,
            output=Data(
                name="att_weights_output",
                shape=(None, 1),
                batch_dim_axis=1,
                auto_create_placeholders=True,
                same_dim_tags_as={"T": time},
            ),
        ),
        base=InternalLayer(
            name="enc_value",
            network=net,
            output=Data(
                name="enc_value_output",
                shape=(None, 1, 2048),
                batch_dim_axis=1,
                auto_create_placeholders=True,
                same_dim_tags_as={"T": time},
            ),
        ),
    )
    kwargs["output"] = GenericAttentionLayer.get_out_data_from_opts(**kwargs)
    layer = GenericAttentionLayer(**kwargs)
    layer.output.sanity_check()
    assert layer.output.shape == (1, 2048) and not layer.output.have_time_axis()


def test_GenericAttentionLayer_basic_multi_head():
    from returnn.tf.layers.base import InternalLayer

    net = TFNetwork(extern_data=ExternData({"data": {"shape": (None, 5)}}), config=Config())
    time = SpatialDim("time")
    num_heads = 8
    kwargs = dict(
        name="att",
        network=net,
        weights=InternalLayer(
            name="att_weights",
            network=net,
            output=Data(
                name="att_weights_output",
                shape=(None, num_heads),
                batch_dim_axis=1,
                auto_create_placeholders=True,
                same_dim_tags_as={"T": time},
            ),
        ),
        base=InternalLayer(
            name="enc_value",
            network=net,
            output=Data(
                name="enc_value_output",
                shape=(None, num_heads, 2048),
                batch_dim_axis=1,
                auto_create_placeholders=True,
                same_dim_tags_as={"T": time},
            ),
        ),
    )
    kwargs["output"] = GenericAttentionLayer.get_out_data_from_opts(**kwargs)
    layer = GenericAttentionLayer(**kwargs)
    layer.output.sanity_check()
    assert layer.output.shape == (num_heads, 2048) and not layer.output.have_time_axis()


def test_GenericAttentionLayer_weights_auto_squeeze_time_end():
    # Example: weights (B,1,T), base (B,T,V)
    from returnn.tf.layers.base import InternalLayer

    net = TFNetwork(extern_data=ExternData({"data": {"shape": (None, 5)}}), config=Config())
    time = SpatialDim("time")
    kwargs = dict(
        name="att",
        network=net,
        auto_squeeze=True,
        weights=InternalLayer(
            name="att_weights",
            network=net,
            output=Data(
                name="att_weights_output",
                shape=(1, None),
                time_dim_axis=2,
                auto_create_placeholders=True,
                same_dim_tags_as={"T": time},
            ),
        ),
        base=InternalLayer(
            name="enc_value",
            network=net,
            output=Data(
                name="enc_value_output", shape=(None, 2048), auto_create_placeholders=True, same_dim_tags_as={"T": time}
            ),
        ),
    )
    print("GenericAttentionLayer kwargs:")
    pprint(kwargs)
    kwargs["output"] = GenericAttentionLayer.get_out_data_from_opts(**kwargs)
    layer = GenericAttentionLayer(**kwargs)
    layer.output.sanity_check()
    assert layer.output.shape == (2048,) and not layer.output.have_time_axis()


def test_GenericAttentionLayer_weights_static_time_axis():
    # Example: weights (B,1,W), base (B,W,V), where W: window_size (static)
    window_size = 10
    from returnn.tf.layers.base import InternalLayer

    net = TFNetwork(extern_data=ExternData({"data": {"shape": (None, 5)}}), config=Config())
    time = SpatialDim("time")
    kwargs = dict(
        name="att",
        network=net,
        auto_squeeze=True,
        weights=InternalLayer(
            name="att_weights",
            network=net,
            output=Data(
                name="att_weights_output",
                shape=(1, 10),
                time_dim_axis=2,
                auto_create_placeholders=True,
                same_dim_tags_as={"T": time},
            ),
        ),
        base=InternalLayer(
            name="enc_value",
            network=net,
            output=Data(
                name="enc_value_output",
                shape=(10, 2048),
                time_dim_axis=1,
                auto_create_placeholders=True,
                same_dim_tags_as={"T": time},
            ),
        ),
    )
    print("GenericAttentionLayer kwargs:")
    pprint(kwargs)
    kwargs["output"] = GenericAttentionLayer.get_out_data_from_opts(**kwargs)
    layer = GenericAttentionLayer(**kwargs)
    layer.output.sanity_check()
    assert layer.output.shape == (2048,) and not layer.output.have_time_axis()


def test_GenericAttentionLayer_weights_heads_time_end():
    # Example: weights (B,H,T), base (B,T,H,V)
    from returnn.tf.layers.base import InternalLayer

    net = TFNetwork(extern_data=ExternData({"data": {"shape": (None, 5)}}), config=Config())
    time = SpatialDim("time")
    num_heads = 8
    kwargs = dict(
        name="att",
        network=net,
        weights=InternalLayer(
            name="att_weights",
            network=net,
            output=Data(
                name="att_weights_output",
                shape=(num_heads, None),
                time_dim_axis=2,
                auto_create_placeholders=True,
                same_dim_tags_as={"T": time},
            ),
        ),
        base=InternalLayer(
            name="enc_value",
            network=net,
            output=Data(
                name="enc_value_output",
                shape=(None, num_heads, 2048),
                auto_create_placeholders=True,
                same_dim_tags_as={"T": time},
            ),
        ),
    )
    print("GenericAttentionLayer kwargs:")
    pprint(kwargs)
    kwargs["output"] = GenericAttentionLayer.get_out_data_from_opts(**kwargs)
    layer = GenericAttentionLayer(**kwargs)
    layer.output.sanity_check()
    assert layer.output.shape == (num_heads, 2048) and not layer.output.have_time_axis()


def test_GenericAttentionLayer_weights_heads_auto_squeeze_time_end():
    # Example: weights (B,H,1,T), base (B,T,H,V)
    from returnn.tf.layers.base import InternalLayer

    net = TFNetwork(extern_data=ExternData({"data": {"shape": (None, 5)}}), config=Config())
    time = SpatialDim("time")
    num_heads = 8
    kwargs = dict(
        name="att",
        network=net,
        auto_squeeze=True,
        weights=InternalLayer(
            name="att_weights",
            network=net,
            output=Data(
                name="att_weights_output",
                shape=(num_heads, 1, None),
                time_dim_axis=3,
                auto_create_placeholders=True,
                same_dim_tags_as={"T": time},
            ),
        ),
        base=InternalLayer(
            name="enc_value",
            network=net,
            output=Data(
                name="enc_value_output",
                shape=(None, num_heads, 2048),
                auto_create_placeholders=True,
                same_dim_tags_as={"T": time},
            ),
        ),
    )
    print("GenericAttentionLayer kwargs:")
    pprint(kwargs)
    kwargs["output"] = GenericAttentionLayer.get_out_data_from_opts(**kwargs)
    layer = GenericAttentionLayer(**kwargs)
    layer.output.sanity_check()
    assert layer.output.shape == (num_heads, 2048) and not layer.output.have_time_axis()


def test_GenericAttentionLayer_extra_spatial():
    from returnn.tf.util.data import batch_dim
    from returnn.tf.layers.base import InternalLayer

    net = TFNetwork(extern_data=ExternData({"data": {"shape": (None, 5)}}), config=Config())
    # This is the situation when the GenericAttentionLayer is outside the recurrent loop,
    # and it gets some encoder values (with different time axis),
    # and the attention weights, which has two spatial axis, one of the decoder, and one of the encoder.
    dec_time = SpatialDim("dec time")
    enc_time = SpatialDim("enc time")
    feat1_dim = FeatureDim("feature1", dimension=1)
    kwargs = dict(
        name="att",
        network=net,
        weights=InternalLayer(
            name="att_weights",
            network=net,
            output=Data(
                name="att_weights_output",
                dim_tags=[batch_dim, dec_time, enc_time, feat1_dim],
                auto_create_placeholders=True,
            ),
        ),
        base=InternalLayer(
            name="enc_value",
            network=net,
            output=Data(
                name="enc_value_output",
                shape=(None, 1, 2048),
                batch_dim_axis=1,
                auto_create_placeholders=True,
                same_dim_tags_as={"t": enc_time},
            ),
        ),
    )
    print("GenericAttentionLayer kwargs:")
    pprint(kwargs)
    kwargs["output"] = GenericAttentionLayer.get_out_data_from_opts(**kwargs)
    layer = GenericAttentionLayer(**kwargs)
    layer.output.sanity_check()
    assert layer.output.shape == (1, None, 2048) and layer.output.have_time_axis()
    assert len(layer.output.size_placeholder) == 1
    assert list(layer.output.size_placeholder.values())[0] is layer.weights.output.size_placeholder[0]


def test_GenericAttentionLayer_extra_spatial_multi_head():
    from returnn.tf.util.data import batch_dim
    from returnn.tf.layers.base import InternalLayer

    net = TFNetwork(extern_data=ExternData({"data": {"shape": (None, 5)}}), config=Config())
    dec_time = SpatialDim("dec time")
    enc_time = SpatialDim("enc time")
    heads_dim = FeatureDim("heads", dimension=8)
    feat_dim = FeatureDim("feat", dimension=2048)
    kwargs = dict(
        name="att",
        network=net,
        weights=InternalLayer(
            name="att_weights",
            network=net,
            output=Data(
                name="att_weights_output",
                dim_tags=[batch_dim, dec_time, enc_time, heads_dim],
                auto_create_placeholders=True,
            ),
        ),
        base=InternalLayer(
            name="enc_value",
            network=net,
            output=Data(
                name="enc_value_output",
                dim_tags=[enc_time, batch_dim, heads_dim, feat_dim],
                auto_create_placeholders=True,
            ),
        ),
    )
    print("GenericAttentionLayer kwargs:")
    pprint(kwargs)
    kwargs["output"] = GenericAttentionLayer.get_out_data_from_opts(**kwargs)
    layer = GenericAttentionLayer(**kwargs)
    layer.output.sanity_check()
    assert layer.output.shape == (heads_dim.dimension, None, feat_dim.dimension) and layer.output.have_time_axis()
    assert len(layer.output.size_placeholder) == 1
    assert list(layer.output.size_placeholder.values())[0] is layer.weights.output.size_placeholder[0]


def test_MaskedComputationLayer_UnmaskLayer_in_loop():
    from test_TFNetworkLayer import make_feed_dict
    from returnn.tf.layers.rec import _SubnetworkRecCell

    with make_scope() as session:
        config = Config({"debug_print_layer_output_template": True})
        net = TFNetwork(extern_data=ExternData({"data": {"dim": 20, "sparse": True}}), config=config)
        net_dict = {
            "output": {
                "class": "rec",
                "from": "data",
                "optimize_move_layers_out": False,  # for this test, keep them all in
                "unit": {
                    "const1": {"class": "constant", "value": 1, "with_batch_dim": True},  # just to broadcast mask
                    "mask": {
                        "class": "eval",
                        "from": [":i", "const1"],
                        "out_type": {"dtype": "bool"},
                        "eval": "tf.equal(source(0) % 2, source(1))",
                    },
                    "masked": {
                        "class": "masked_computation",
                        "from": "data:source",
                        "mask": "mask",
                        "unit": {
                            "class": "subnetwork",
                            "from": "data",
                            "subnetwork": {
                                "input0": {"class": "cast", "from": "data", "dtype": "float32"},
                                "input1": {"class": "expand_dims", "axis": "f", "from": "input0"},
                                "output": {"class": "rec", "unit": "cumsum", "n_out": 1, "from": "input1"},
                            },
                        },
                    },
                    "unmask": {"class": "unmask", "from": "masked", "mask": "mask"},
                    "output": {"class": "squeeze", "from": "unmask", "axis": "f"},
                },
            }
        }
        net.construct_from_dict(net_dict)
        rec_layer = net.get_layer("output")
        assert isinstance(rec_layer, RecLayer)
        rec_cell = rec_layer.cell
        assert isinstance(rec_cell, _SubnetworkRecCell)
        assert "masked" in rec_cell.layers_in_loop
        assert "unmask" in rec_cell.layers_in_loop
        in_data = net.get_layer("data").output
        out_data = net.get_layer("output").output.copy_as_batch_major()
        feed_dict = make_feed_dict(net.extern_data)
        in_v, out_v = session.run((in_data.placeholder, out_data.placeholder), feed_dict=feed_dict)
        print(in_v)
        print(out_v)
        assert_equal(in_v.shape, out_v.shape)
        for b in range(in_v.shape[0]):
            x = 0.0
            for t in range(in_v.shape[1]):
                if t % 2 == 1:
                    y = x + in_v[b, t]
                else:
                    y = x
                numpy.testing.assert_almost_equal(y, out_v[b, t])
                x = y


def test_MaskedComputationLayer_UnmaskLayer_in_loop_opt():
    from test_TFNetworkLayer import make_feed_dict
    from returnn.tf.layers.basic import SubnetworkLayer
    from returnn.tf.layers.rec import _SubnetworkRecCell

    with make_scope() as session:
        config = Config({"debug_print_layer_output_template": True})
        net = TFNetwork(extern_data=ExternData({"data": {"dim": 20, "sparse": True}}), config=config)
        net_dict = {
            "output": {
                "class": "rec",
                "from": "data",
                "unit": {
                    "const1": {"class": "constant", "value": 1, "with_batch_dim": True},  # just to broadcast mask
                    "mask": {
                        "class": "eval",
                        "from": [":i", "const1"],
                        "out_type": {"dtype": "bool"},
                        "eval": "tf.equal(source(0) % 2, source(1))",
                    },
                    "masked": {
                        "class": "masked_computation",
                        "from": "data:source",
                        "mask": "mask",
                        "unit": {
                            "class": "subnetwork",
                            "from": "data",
                            "subnetwork": {
                                "input0": {"class": "cast", "from": "data", "dtype": "float32"},
                                "input1": {"class": "expand_dims", "axis": "f", "from": "input0"},
                                "output": {"class": "rec", "unit": "cumsum", "n_out": 1, "from": "input1"},
                            },
                        },
                    },
                    "unmask": {"class": "unmask", "from": "masked", "mask": "mask"},
                    "output": {"class": "squeeze", "from": "unmask", "axis": "f"},
                },
            }
        }
        net.construct_from_dict(net_dict)
        rec_layer = net.get_layer("output")
        assert isinstance(rec_layer, RecLayer)
        rec_cell = rec_layer.cell
        assert isinstance(rec_cell, _SubnetworkRecCell)
        assert not rec_cell.layers_in_loop  # all moved out
        in_data = net.get_layer("data").output
        out_data = net.get_layer("output").output.copy_as_batch_major()
        assert in_data.get_time_dim_tag() == out_data.get_time_dim_tag()
        masked_comp_layer = rec_cell.get_layer_from_outside("masked")
        assert isinstance(masked_comp_layer, MaskedComputationLayer)
        masked_comp_sub_layer = masked_comp_layer.sub_layer
        assert isinstance(masked_comp_sub_layer, SubnetworkLayer)
        masked_in0_layer = masked_comp_sub_layer.get_sub_layer("input0")
        masked_in_layer = masked_in0_layer.sources[0]
        extra = {
            "mask": rec_cell.get_layer_from_outside("mask").output.placeholder,
            "masked": masked_comp_layer.output.placeholder,
            "masked/input0": masked_in0_layer.output.placeholder,
            "masked/data": masked_in_layer.output.placeholder,
        }
        in_v, out_v, out_seq_lens_v, extra_v = session.run(
            (in_data.placeholder, out_data.placeholder, out_data.get_sequence_lengths(), extra),
            feed_dict=make_feed_dict(net.extern_data),
        )
        print(in_v)
        print(out_v)
        print("seq lens:", out_seq_lens_v)
        pprint(extra_v)
        assert_equal(in_v.shape, out_v.shape)
        for b in range(in_v.shape[0]):
            x = 0.0
            for t in range(in_v.shape[1]):
                if t >= out_seq_lens_v[b]:
                    continue
                if t % 2 == 1:
                    y = x + in_v[b, t]
                else:
                    y = x
                numpy.testing.assert_almost_equal(y, out_v[b, t])
                x = y


def test_MaskedComputationLayer_in_loop_auto_unmask():
    # https://github.com/rwth-i6/returnn/issues/769
    from test_TFNetworkLayer import make_feed_dict
    from returnn.tf.layers.rec import _SubnetworkRecCell

    for opt in [False, True]:
        print("*** using rec optimization:", opt)
        with make_scope() as session:
            config = Config({"debug_print_layer_output_template": True})
            net = TFNetwork(extern_data=ExternData({"data": {"dim": 20, "sparse": True}}), config=config)
            net_dict = {
                "output": {
                    "class": "rec",
                    "from": "data",
                    "optimize_move_layers_out": opt,  # test both variants
                    "unit": {
                        "const1": {"class": "constant", "value": 1, "with_batch_dim": True},  # just to broadcast mask
                        "mask": {
                            "class": "eval",
                            "from": [":i", "const1"],
                            "out_type": {"dtype": "bool"},
                            "eval": "tf.equal(source(0) % 2, source(1))",
                        },
                        "in": {"class": "reinterpret_data", "from": "data:source", "set_sparse": False},
                        "masked": {
                            "class": "masked_computation",
                            "from": "in",
                            "mask": "mask",
                            "unit": {"class": "cumsum", "from": "data", "initial_output": 1},
                        },
                        "masked_out": {"class": "copy", "from": "masked"},
                        "output": {"class": "eval", "from": ["masked_out", "in"], "eval": "source(0) + source(1) ** 2"},
                    },
                }
            }
            net.construct_from_dict(net_dict)
            rec_layer = net.get_layer("output")
            assert isinstance(rec_layer, RecLayer)
            rec_cell = rec_layer.cell
            assert isinstance(rec_cell, _SubnetworkRecCell)
            if opt:
                assert not rec_cell.layers_in_loop  # all moved out
            else:
                assert not rec_cell.input_layers_moved_out and not rec_cell.output_layers_moved_out  # none moved out
            in_data = net.get_layer("data").output
            out_data = net.get_layer("output").output.copy_as_batch_major()
            print("out:", out_data)
            assert in_data.get_time_dim_tag() == out_data.get_time_dim_tag()
            in_v, out_v, out_seq_lens_v = session.run(
                (in_data.placeholder, out_data.placeholder, out_data.get_sequence_lengths()),
                feed_dict=make_feed_dict(net.extern_data),
            )
            print(in_v)
            print(out_v)
            print("seq lens:", out_seq_lens_v)
            assert_equal(in_v.shape, out_v.shape)
            for b in range(in_v.shape[0]):
                x = 1
                for t in range(in_v.shape[1]):
                    if t >= out_seq_lens_v[b]:
                        continue
                    if t % 2 == 1:
                        x = x + in_v[b, t]
                    y = x + in_v[b, t] ** 2
                    numpy.testing.assert_almost_equal(y, out_v[b, t])


def test_MaskedComputationLayer_sub_layers():
    # https://github.com/rwth-i6/returnn/pull/984
    from test_TFNetworkLayer import make_feed_dict
    from returnn.tf.layers.rec import _SubnetworkRecCell

    for opt in [False, True]:
        print("*** using rec optimization:", opt)
        with make_scope() as session:
            config = Config({"debug_print_layer_output_template": True})
            net = TFNetwork(extern_data=ExternData({"data": {"dim": 20, "sparse": True}}), config=config)
            net_dict = {
                "output": {
                    "class": "rec",
                    "from": "data",
                    "optimize_move_layers_out": opt,  # test both variants
                    "unit": {
                        "const1": {"class": "constant", "value": 1, "with_batch_dim": True},  # just to broadcast mask
                        "mask": {
                            "class": "eval",
                            "from": [":i", "const1"],
                            "out_type": {"dtype": "bool"},
                            "eval": "tf.equal(source(0) % 2, source(1))",
                        },
                        "in": {"class": "reinterpret_data", "from": "data:source", "set_sparse": False},
                        "masked": {
                            "class": "masked_computation",
                            "mask": "mask",
                            "unit": {
                                "class": "subnetwork",
                                "initial_output": 0,
                                "subnetwork": {
                                    "sub1": {
                                        "class": "cumsum",
                                        "from": "base:in",
                                        "initial_output": 1,
                                        "is_output_layer": True,
                                    },
                                    "sub2": {
                                        "class": "cumsum",
                                        "from": "base:in",
                                        "initial_output": 2,
                                        "is_output_layer": True,
                                    },
                                    "output": {"class": "copy", "from": "base:in"},  # unused
                                },
                            },
                        },
                        "output": {
                            "class": "eval",
                            "from": ["masked/sub1", "masked/sub2"],
                            "eval": "source(0) * 2 + source(1) * 3",
                        },
                    },
                }
            }
            net.construct_from_dict(net_dict)
            rec_layer = net.get_layer("output")
            assert isinstance(rec_layer, RecLayer)
            rec_cell = rec_layer.cell
            assert isinstance(rec_cell, _SubnetworkRecCell)
            if opt:
                assert not rec_cell.layers_in_loop  # all moved out
            else:
                assert not rec_cell.input_layers_moved_out and not rec_cell.output_layers_moved_out  # none moved out
            in_data = net.get_layer("data").output
            out_data = net.get_layer("output").output.copy_as_batch_major()
            print("out:", out_data)
            assert in_data.get_time_dim_tag() == out_data.get_time_dim_tag()
            in_v, out_v, out_seq_lens_v = session.run(
                (in_data.placeholder, out_data.placeholder, out_data.get_sequence_lengths()),
                feed_dict=make_feed_dict(net.extern_data),
            )
            print(in_v)
            print(out_v)
            print("seq lens:", out_seq_lens_v)
            assert_equal(in_v.shape, out_v.shape)
            for b in range(in_v.shape[0]):
                x = 0
                for t in range(in_v.shape[1]):
                    if t >= out_seq_lens_v[b]:
                        continue
                    if t % 2 == 1:
                        x = x + in_v[b, t]
                    y = (1 + x) * 2 + (2 + x) * 3
                    numpy.testing.assert_almost_equal(out_v[b, t], y)


def test_MaskedComputationLayer_sub_layers_RecLayer_construct():
    from test_TFNetworkLayer import make_feed_dict
    from returnn.tf.layers.rec import _SubnetworkRecCell

    with make_scope() as session:
        config = Config({"debug_print_layer_output_template": True})
        net = TFNetwork(extern_data=ExternData({"data": {"dim": 20, "sparse": True}}), config=config)
        net_dict = {
            "output": {
                "class": "rec",
                "from": "data",
                "unit": {
                    "const1": {"class": "constant", "value": 1, "with_batch_dim": True},  # just to broadcast mask
                    "mask": {
                        "class": "eval",
                        "from": [":i", "const1"],
                        "out_type": {"dtype": "bool"},
                        "eval": "tf.equal(source(0) % 2, source(1))",
                    },
                    "in": {"class": "copy", "from": "data:source"},
                    "masked": {
                        "class": "masked_computation",
                        "mask": "mask",
                        "unit": {
                            "class": "subnetwork",
                            "subnetwork": {
                                "sub1": {"class": "linear", "from": "base:prev:output", "n_out": 3},
                                "sub2": {"class": "linear", "from": "base:in", "n_out": 3},
                                "output": {"class": "copy", "from": ["sub1", "sub2"]},  # unused
                            },
                        },
                    },
                    "output": {
                        "class": "eval",
                        "from": ["masked/sub1", "masked/sub2"],
                        "eval": "source(0) * 2 + source(1) * 3",
                    },
                },
            }
        }
        net.construct_from_dict(net_dict)
        net.initialize_params(session)
        rec_layer = net.get_layer("output")
        assert isinstance(rec_layer, RecLayer)
        rec_cell = rec_layer.cell
        assert isinstance(rec_cell, _SubnetworkRecCell)
        assert set(rec_cell.layers_in_loop).issuperset({"masked", "masked/sub1", "masked/sub2", "output"})
        in_data = net.get_layer("data").output
        out_data = net.get_layer("output").output.copy_as_batch_major()
        print("out:", out_data)
        assert in_data.get_time_dim_tag() == out_data.get_time_dim_tag()
        in_v, out_v, out_seq_lens_v = session.run(
            (in_data.placeholder, out_data.placeholder, out_data.get_sequence_lengths()),
            feed_dict=make_feed_dict(net.extern_data),
        )
        print(in_v)
        print(out_v)
        print("seq lens:", out_seq_lens_v)


def test_att_train_search_loss_prev_beam():
    beam_size = 1
    num_ner_labels = 13
    net_dict = {
        "output": {
            "class": "rec",
            "from": "data",
            "target": "classes",
            "unit": {
                "crf_rec_in": {"class": "linear", "from": "prev:classes_copy", "activation": None, "n_out": 10},
                "crf_rec": {
                    "class": "linear",
                    "from": ["raw", "crf_rec_in"],
                    "activation": "relu",
                    "target": "classes",
                },
                "enc_ctx_slice": {"class": "copy", "from": "data:source"},
                "source_embed_slice": {"class": "copy", "from": "data:source"},
                "output": {
                    "beam_size": beam_size,
                    "class": "choice",
                    "from": ["crf_rec"],
                    "initial_output": 0,
                    "target": "classes",
                },
                "raw": {
                    "class": "linear",
                    "from": ["enc_ctx_slice", "lstm_tgt"],
                    "n_out": num_ner_labels,
                    "activation": None,
                    "is_output_layer": True,
                },
                "classes_copy": {"class": "copy", "from": "output", "initial_output": 0},
                "is_O": {"class": "compare", "kind": "equal", "from": ["prev:classes_copy"], "value": 0},
                "tgt_lstm_input": {
                    "class": "switch",
                    "condition": "is_O",
                    "true_from": "source_embed_slice",
                    "false_from": "prev:target_embed",
                },
                "lstm_tgt": {"class": "rec", "from": ["tgt_lstm_input"], "n_out": 12, "unit": "nativelstm2"},
                "target_embed": {
                    "class": "linear",
                    "activation": None,
                    "from": "output",
                    "initial_output": 0,
                    "n_out": 10,
                },
            },
        },
        "loss_layer": {
            "class": "linear",
            "activation": "softmax",
            "target": "classes",
            "from": ["output/raw"],
            "loss": "ce",
        },
    }
    with make_scope() as session:
        config = Config({"debug_print_layer_output_template": True})
        net = TFNetwork(
            extern_data=ExternData({"data": {"dim": 10}, "classes": {"dim": 20, "sparse": True}}),
            config=config,
            train_flag=True,
            search_flag=True,
        )
        net.construct_from_dict(net_dict)
        net.maybe_construct_objective()
        print(net.losses_dict)

        net.initialize_params(session)

        from test_TFNetworkLayer import make_feed_dict

        feed_dict = make_feed_dict(net.extern_data, same_time=True)
        loss = session.run(net.total_loss, feed_dict=feed_dict)
        print("loss:", loss)


def test_MaskedComputationLayer_search_choices_resolution():
    beam_size = 3
    EncKeyTotalDim = 10
    AttNumHeads = 2
    target = "classes"
    num_classes = 13
    blank_idx = num_classes - 2
    from test_TFNetworkLayer import make_feed_dict

    net_dict = {
        "encoder": {"class": "linear", "from": "data", "activation": "relu", "n_out": EncKeyTotalDim},
        "enc_ctx": {"class": "copy", "from": "encoder"},
        "enc_value": {"class": "copy", "from": "encoder"},
        "inv_fertility": {"class": "linear", "activation": "sigmoid", "from": "encoder", "n_out": AttNumHeads},
        "output": {
            "class": "rec",
            "from": [],
            "unit": {
                "output": {
                    "class": "choice",
                    "target": target,
                    "beam_size": beam_size,
                    "from": ["output_prob"],
                    "initial_output": 0,
                },
                "end": {"class": "compare", "from": ["output"], "value": 0},
                "target_embed": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["output"],
                    "n_out": 12,
                    "initial_output": 0,
                },
                "weight_feedback": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["prev:accum_att_weights"],
                    "n_out": EncKeyTotalDim,
                },
                "s_transformed": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["masked_s"],
                    "n_out": EncKeyTotalDim,
                },
                "energy_in": {
                    "class": "combine",
                    "kind": "add",
                    "from": ["base:enc_ctx", "weight_feedback", "s_transformed"],
                    "n_out": EncKeyTotalDim,
                },
                "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
                "energy": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["energy_tanh"],
                    "n_out": AttNumHeads,
                },  # (B, enc-T, H)
                "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, enc-T, H)
                "accum_att_weights": {
                    "class": "eval",
                    "from": ["prev:accum_att_weights", "att_weights", "base:inv_fertility"],
                    "eval": "source(0) + source(1) * source(2) * 0.5",
                    "out_type": {"dim": AttNumHeads, "shape": (None, AttNumHeads)},
                },
                "att0": {
                    "class": "generic_attention",
                    "weights": "att_weights",
                    "base": "base:enc_value",
                    "auto_squeeze": False,
                },  # (B, H, V)
                "att": {
                    "class": "merge_dims",
                    "axes": ["dim:%i" % AttNumHeads, "dim:%i" % EncKeyTotalDim],
                    "from": "att0",
                },  # (B, H*V)
                "not_blank_mask": {
                    "class": "compare",
                    "from": ["output"],
                    "value": blank_idx,
                    "kind": "not_equal",
                    "initial_output": True,
                },
                "masked_s": {
                    "class": "masked_computation",
                    "mask": "prev:not_blank_mask",
                    "unit": {
                        "class": "rec",
                        "unit": "NativeLSTM2",
                        "from": ["prev:target_embed", "prev:att"],
                        "n_out": 10,
                    },
                    "from": "prev:output",
                },
                "readout_in": {
                    "class": "linear",
                    "from": ["masked_s", "prev:target_embed", "att"],
                    "activation": None,
                    "n_out": 10,
                },
                "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
                "output_prob": {"class": "softmax", "from": ["readout"], "target": target, "loss": "ce"},
            },
            "target": target,
            "max_seq_len": "max_len_from('base:encoder')",
        },
    }
    with make_scope() as session:
        config = Config({"debug_print_layer_output_template": True})
        extern_data = ExternData({"data": {"dim": 20, "sparse": True}, target: {"dim": num_classes, "sparse": True}})
        feed_dict = make_feed_dict(extern_data)

        print("***** Construct train net.")
        train_net = TFNetwork(extern_data=extern_data, config=config, train_flag=True)
        train_net.construct_from_dict(net_dict)
        loss = train_net.get_total_loss()
        optimizer = tf_compat.v1.train.AdamOptimizer(learning_rate=0.1)
        with tf.control_dependencies([optimizer.minimize(loss)]):
            loss = tf.identity(loss)
        session.run(tf_compat.v1.global_variables_initializer())
        loss_values = []
        for step in range(10):
            loss_values.append(session.run(loss, feed_dict=feed_dict))
            print("loss:", loss_values[-1])
        assert all([loss_values[i + 1] < loss_values[i] for i in range(len(loss_values) - 1)])
        print()

        print("***** Construct search net.")
        search_net = TFNetwork(extern_data=extern_data, config=config, search_flag=True)
        search_net.construct_from_dict(net_dict)
        out = search_net.get_default_output_layer().output.placeholder
        print("out:", session.run(out, feed_dict=feed_dict))


def test_MaskedComputationLayer_subnet_search_choices_resolution():
    beam_size = 3
    EncKeyTotalDim = 10
    AttNumHeads = 2
    target = "classes"
    num_classes = 13
    blank_idx = num_classes - 2
    from test_TFNetworkLayer import make_feed_dict

    net_dict = {
        "encoder": {"class": "linear", "from": "data", "activation": "relu", "n_out": EncKeyTotalDim},
        "enc_ctx": {"class": "copy", "from": "encoder"},
        "enc_value": {"class": "copy", "from": "encoder"},
        "inv_fertility": {"class": "linear", "activation": "sigmoid", "from": "encoder", "n_out": AttNumHeads},
        "output": {
            "class": "rec",
            "from": [],
            "unit": {
                "output": {
                    "class": "choice",
                    "target": target,
                    "beam_size": beam_size,
                    "from": ["output_prob"],
                    "initial_output": 0,
                },
                "end": {"class": "compare", "from": ["output"], "value": 0},
                "target_embed": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["output"],
                    "n_out": 12,
                    "initial_output": 0,
                },
                "weight_feedback": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["prev:accum_att_weights"],
                    "n_out": EncKeyTotalDim,
                },
                "s_transformed": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["masked_s"],
                    "n_out": EncKeyTotalDim,
                },
                "energy_in": {
                    "class": "combine",
                    "kind": "add",
                    "from": ["base:enc_ctx", "weight_feedback", "s_transformed"],
                    "n_out": EncKeyTotalDim,
                },
                "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
                "energy": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["energy_tanh"],
                    "n_out": AttNumHeads,
                },  # (B, enc-T, H)
                "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, enc-T, H)
                "accum_att_weights": {
                    "class": "eval",
                    "from": ["prev:accum_att_weights", "att_weights", "base:inv_fertility"],
                    "eval": "source(0) + source(1) * source(2) * 0.5",
                    "out_type": {"dim": AttNumHeads, "shape": (None, AttNumHeads)},
                },
                "att0": {
                    "class": "generic_attention",
                    "weights": "att_weights",
                    "base": "base:enc_value",
                    "auto_squeeze": False,
                },  # (B, H, V)
                "att": {
                    "class": "merge_dims",
                    "axes": ["dim:%i" % AttNumHeads, "dim:%i" % EncKeyTotalDim],
                    "from": "att0",
                },  # (B, H*V)
                "not_blank_mask": {
                    "class": "compare",
                    "from": ["output"],
                    "value": blank_idx,
                    "kind": "not_equal",
                    "initial_output": True,
                },
                "masked_s": {
                    "class": "masked_computation",
                    "mask": "prev:not_blank_mask",
                    "unit": {
                        "class": "subnetwork",
                        "from": "data",
                        "subnetwork": {
                            "lstm0": {
                                "class": "rnn_cell",
                                "unit": "basiclstm",
                                # Note: We just ignore the input data but access other layers from parent.
                                "from": ["base:prev:target_embed", "base:prev:att"],
                                "n_out": 11,
                            },
                            "output": {"class": "copy", "from": "lstm0"},
                        },
                    },
                    "from": "prev:output",
                },
                "readout_in": {
                    "class": "linear",
                    "from": ["masked_s", "prev:target_embed", "att"],
                    "activation": None,
                    "n_out": 10,
                },
                "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
                "output_prob": {"class": "softmax", "from": ["readout"], "target": target, "loss": "ce"},
            },
            "target": target,
            "max_seq_len": "max_len_from('base:encoder')",
        },
    }
    with make_scope() as session:
        config = Config({"debug_print_layer_output_template": True})
        extern_data = ExternData({"data": {"dim": 20, "sparse": True}, target: {"dim": num_classes, "sparse": True}})
        feed_dict = make_feed_dict(extern_data)

        print("***** Construct train net.")
        train_net = TFNetwork(extern_data=extern_data, config=config, train_flag=True)
        train_net.construct_from_dict(net_dict)
        loss = train_net.get_total_loss()
        optimizer = tf_compat.v1.train.AdamOptimizer(learning_rate=0.1)
        with tf.control_dependencies([optimizer.minimize(loss)]):
            loss = tf.identity(loss)
        session.run(tf_compat.v1.global_variables_initializer())
        loss_values = []
        for step in range(10):
            loss_values.append(session.run(loss, feed_dict=feed_dict))
            print("loss:", loss_values[-1])
        assert all([loss_values[i + 1] < loss_values[i] for i in range(len(loss_values) - 1)])
        print()

        print("***** Construct search net.")
        search_net = TFNetwork(extern_data=extern_data, config=config, search_flag=True)
        search_net.construct_from_dict(net_dict)
        out = search_net.get_default_output_layer().output.placeholder
        print("out:", session.run(out, feed_dict=feed_dict))


def test_MaskedComputationLayer_subnet_rec_search():
    beam_size = 3
    EncKeyTotalDim = 10
    AttNumHeads = 2
    target = "classes"
    num_classes = 13
    blank_idx = num_classes - 2
    from test_TFNetworkLayer import make_feed_dict

    net_dict = {
        "encoder": {"class": "linear", "from": "data", "activation": "relu", "n_out": EncKeyTotalDim},
        "enc_ctx": {"class": "copy", "from": "encoder"},
        "enc_value": {"class": "copy", "from": "encoder"},
        "inv_fertility": {"class": "linear", "activation": "sigmoid", "from": "encoder", "n_out": AttNumHeads},
        "output": {
            "class": "rec",
            "from": [],
            "unit": {
                "output": {
                    "class": "choice",
                    "target": target,
                    "beam_size": beam_size,
                    "from": ["output_prob"],
                    "initial_output": 0,
                },
                "end": {"class": "compare", "from": ["output"], "value": 0},
                "target_embed": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["output"],
                    "n_out": 12,
                    "initial_output": 0,
                },
                "weight_feedback": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["prev:accum_att_weights"],
                    "n_out": EncKeyTotalDim,
                },
                "s_transformed": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["masked_s_unmask"],
                    "n_out": EncKeyTotalDim,
                },
                "energy_in": {
                    "class": "combine",
                    "kind": "add",
                    "from": ["base:enc_ctx", "weight_feedback", "s_transformed"],
                    "n_out": EncKeyTotalDim,
                },
                "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
                "energy": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["energy_tanh"],
                    "n_out": AttNumHeads,
                },  # (B, enc-T, H)
                "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, enc-T, H)
                "accum_att_weights": {
                    "class": "eval",
                    "from": ["prev:accum_att_weights", "att_weights", "base:inv_fertility"],
                    "eval": "source(0) + source(1) * source(2) * 0.5",
                    "out_type": {"dim": AttNumHeads, "shape": (None, AttNumHeads)},
                },
                "att0": {
                    "class": "generic_attention",
                    "weights": "att_weights",
                    "base": "base:enc_value",
                    "auto_squeeze": False,
                },  # (B, H, V)
                "att": {
                    "class": "merge_dims",
                    "axes": ["dim:%i" % AttNumHeads, "dim:%i" % EncKeyTotalDim],
                    "from": "att0",
                },  # (B, H*V)
                "not_blank_mask": {
                    "class": "compare",
                    "from": ["output"],
                    "value": blank_idx,
                    "kind": "not_equal",
                    "initial_output": True,
                },
                "masked_s": {
                    "class": "masked_computation",
                    "mask": "prev:not_blank_mask",
                    "from": "prev:output",
                    "unit": {
                        "class": "subnetwork",
                        "from": "data",
                        "n_out": 11,
                        "subnetwork": {
                            "lstm0": {"class": "rec", "unit": "nativelstm2", "from": "data", "n_out": 11},
                            "output": {"class": "copy", "from": "lstm0"},
                        },
                    },
                },
                "masked_s_unmask": {"class": "unmask", "from": "masked_s", "mask": "prev:not_blank_mask"},
                "readout_in": {
                    "class": "linear",
                    "from": ["masked_s_unmask", "prev:target_embed", "att"],
                    "activation": None,
                    "n_out": 10,
                },
                "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
                "output_prob": {"class": "softmax", "from": ["readout"], "target": target, "loss": "ce"},
            },
            "target": target,
            "max_seq_len": "max_len_from('base:encoder')",
        },
    }
    with make_scope() as session:
        config = Config({"debug_print_layer_output_template": True})
        extern_data = ExternData({"data": {"dim": 20, "sparse": True}, target: {"dim": num_classes, "sparse": True}})
        feed_dict = make_feed_dict(extern_data)

        print("***** Construct search net.")
        search_net = TFNetwork(extern_data=extern_data, config=config, search_flag=True)
        search_net.construct_from_dict(net_dict)
        session.run(tf_compat.v1.global_variables_initializer())
        out = search_net.get_default_output_layer().output.placeholder
        print("out:", session.run(out, feed_dict=feed_dict))


def test_MaskedComputationLayer_subnet_trafo_search():
    beam_size = 3
    target = "classes"
    num_classes = 13
    blank_idx = num_classes - 2
    from test_TFNetworkLayer import make_feed_dict

    net_dict = {
        "encoder": {"class": "linear", "from": "data", "activation": "relu", "n_out": 10},
        "output": {
            "class": "rec",
            "from": "encoder",
            "unit": {
                "output": {
                    "class": "choice",
                    "target": target,
                    "beam_size": beam_size,
                    "from": "output_prob",
                    "initial_output": 0,
                },
                "target_embed": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": "output",
                    "n_out": 12,
                    "initial_output": 0,
                },
                "not_blank_mask": {
                    "class": "compare",
                    "from": "output",
                    "value": blank_idx,
                    "kind": "not_equal",
                    "initial_output": True,
                },
                "masked_s": {
                    "class": "masked_computation",
                    "mask": "prev:not_blank_mask",
                    "from": "prev:output",
                    "unit": {
                        "class": "subnetwork",
                        "from": "data",
                        "subnetwork": {
                            "lstm": {"class": "rec", "unit": "nativelstm2", "from": "data", "n_out": 11},
                            "selfatt": {
                                "class": "self_attention",
                                "num_heads": 2,
                                "total_key_dim": 6,
                                "from": "lstm",
                                "attention_left_only": True,
                                "n_out": 14,
                            },
                            "output": {"class": "copy", "from": "selfatt"},
                        },
                    },
                },
                "masked_s_unmask": {"class": "unmask", "from": "masked_s", "mask": "prev:not_blank_mask"},
                "readout_in": {
                    "class": "linear",
                    "from": ["masked_s_unmask", "prev:target_embed", "data:source"],
                    "activation": None,
                    "n_out": 10,
                },
                "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": "readout_in"},
                "output_prob": {"class": "softmax", "from": "readout", "target": target, "loss": "ce"},
            },
            "target": target,
        },
    }
    with make_scope() as session:
        config = Config({"debug_print_layer_output_template": True})
        extern_data = ExternData({"data": {"dim": 20, "sparse": True}, target: {"dim": num_classes, "sparse": True}})
        feed_dict = make_feed_dict(extern_data)

        print("***** Construct search net.")
        search_net = TFNetwork(extern_data=extern_data, config=config, search_flag=True)
        search_net.construct_from_dict(net_dict)
        session.run(tf_compat.v1.global_variables_initializer())
        out = search_net.get_default_output_layer().output.placeholder
        print("out:", session.run(out, feed_dict=feed_dict))


def test_MaskedComputationLayer_UnmaskLayer_masked_outside():
    from returnn.tf.layers.rec import _SubnetworkRecCell

    with make_scope() as session:
        config = Config({"debug_print_layer_output_template": True})
        net = TFNetwork(
            extern_data=ExternData({"data": {"dim": 20, "sparse": True}, "data_masked": {"dim": 20, "sparse": True}}),
            config=config,
        )
        net_dict = {
            "output": {
                "class": "rec",
                "from": "data",
                "unit": {
                    "const1": {"class": "constant", "value": 1, "with_batch_dim": True},  # just to broadcast mask
                    "in_loop_dummy": {"class": "combine", "kind": "add", "from": ["const1", "prev:in_loop_dummy"]},
                    "mask": {
                        "class": "eval",
                        "from": [":i", "const1", "in_loop_dummy"],
                        "out_type": {"dtype": "bool"},
                        "eval": "(tf.equal(source(0) % 2, source(1)), source(2))[0]",
                        "collocate_with": "in_loop_dummy",
                    },
                    "masked": {
                        "class": "masked_computation",
                        "from": "data:source",
                        "mask": "mask",
                        "masked_from": "base:data:data_masked",
                        "unit": {
                            "class": "subnetwork",
                            "from": "data",
                            "subnetwork": {
                                "input0": {"class": "cast", "from": "data", "dtype": "float32"},
                                "input1": {"class": "expand_dims", "axis": "f", "from": "input0"},
                                "output": {"class": "rec", "unit": "cumsum", "n_out": 1, "from": "input1"},
                            },
                        },
                    },
                    "unmask": {"class": "unmask", "from": "masked", "mask": "mask", "collocate_with": "in_loop_dummy"},
                    "output": {"class": "squeeze", "from": "unmask", "axis": "f"},
                },
            }
        }
        net.construct_from_dict(net_dict)
        rec_layer = net.get_layer("output")
        assert isinstance(rec_layer, RecLayer)
        rec_cell = rec_layer.cell
        assert isinstance(rec_cell, _SubnetworkRecCell)
        assert "masked" in rec_cell.input_layers_moved_out
        assert "unmask" in rec_cell.layers_in_loop
        in_data = net.get_layer("data").output
        out_data = net.get_layer("output").output.copy_as_batch_major()
        feed_dict = {
            net.extern_data.get_batch_info().dim: 2,
            net.extern_data.data["data"].placeholder: [[3, 4, 5, 6], [5, 4, -3, -4]],
            net.extern_data.data["data"].size_placeholder[0]: [4, 2],
            net.extern_data.data["data_masked"].placeholder: [[4, 6], [4, -4]],
            net.extern_data.data["data_masked"].size_placeholder[0]: [2, 1],
        }
        in_v, out_v = session.run((in_data.placeholder, out_data.placeholder), feed_dict=feed_dict)
        print(in_v)
        print(out_v)
        assert_equal(in_v.shape, out_v.shape)
        for b in range(in_v.shape[0]):
            x = 0.0
            for t in range(in_v.shape[1]):
                if t % 2 == 1:
                    y = x + in_v[b, t]
                else:
                    y = x
                numpy.testing.assert_almost_equal(y, out_v[b, t])
                x = y


def test_MaskedComputationLayer_outside():
    with make_scope() as session:
        config = Config({"debug_print_layer_output_template": True})
        tag = SpatialDim("time")
        net = TFNetwork(
            extern_data=ExternData(
                {
                    "data": {"dim": 20, "sparse": True, "same_dim_tags_as": {"t": tag}},
                    "mask": {"dim": 2, "dtype": "bool", "sparse": True, "same_dim_tags_as": {"t": tag}},
                }
            ),
            config=config,
        )
        net_dict = {
            "output": {
                "class": "masked_computation",
                "from": "data",
                "mask": "data:mask",
                "unit": {"class": "copy", "from": "data"},
            },
        }
        net.construct_from_dict(net_dict)
        in_data = net.get_layer("data").output
        in_mask_data = net.get_layer("data:mask").output
        out_data = net.get_layer("output").output.copy_as_batch_major()
        feed_dict = {
            net.extern_data.get_batch_info().dim: 2,
            net.extern_data.data["data"].placeholder: [[3, 4, 5, 6], [5, 4, -3, -4]],
            net.extern_data.data["data"].size_placeholder[0]: [4, 2],
            net.extern_data.data["mask"].placeholder: [[0, 1, 1, 0], [1, 0, 1, 0]],
            net.extern_data.data["mask"].size_placeholder[0]: [4, 2],
        }
        in_v, in_mask_v, in_lens, out_v, out_lens = session.run(
            (
                in_data.placeholder,
                in_mask_data.placeholder,
                in_data.get_sequence_lengths(),
                out_data.placeholder,
                out_data.get_sequence_lengths(),
            ),
            feed_dict=feed_dict,
        )
        assert isinstance(in_v, numpy.ndarray) and isinstance(in_mask_v, numpy.ndarray)
        assert isinstance(in_lens, numpy.ndarray)
        assert isinstance(out_v, numpy.ndarray) and isinstance(out_lens, numpy.ndarray)
        print(in_v)
        print(in_mask_v)
        print(in_lens)
        print(out_v)
        print(out_lens)
        assert in_v.shape == in_mask_v.shape and in_v.ndim == 2
        num_batch = in_v.shape[0]
        assert in_lens.shape == (num_batch,)
        assert out_v.ndim == 2 and out_v.shape[0] == num_batch and out_lens.shape == (num_batch,)
        for b in range(num_batch):
            t_ = 0
            for t in range(in_v.shape[1]):
                if t >= in_lens[b]:
                    break
                if not in_mask_v[b, t]:
                    continue
                assert_equal(in_v[b, t], out_v[b, t_])
                t_ += 1
            assert_equal(t_, out_lens[b])
        assert out_v.shape == (num_batch, max(out_lens))


def test_MaskedComputationLayer_name_scope():
    with make_scope() as session:
        from returnn.tf.util.data import batch_dim

        time_dim = SpatialDim("time")
        config = Config(
            {
                "extern_data": {
                    "data": {"dim": 20, "sparse": True, "dim_tags": (batch_dim, time_dim)},
                    "mask": {"dim": 2, "dtype": "bool", "sparse": True, "dim_tags": (batch_dim, time_dim)},
                }
            }
        )
        net = TFNetwork(config=config)
        net_dict = {
            "output": {
                "class": "masked_computation",
                "from": "data",
                "mask": "data:mask",
                "unit": {"class": "linear", "from": "data", "n_out": 7},
            },
        }
        net.construct_from_dict(net_dict)
        net.initialize_params(session)
        out_data = net.get_layer("output").output.copy_as_batch_major()
        from test_TFNetworkLayer import make_feed_dict

        feed_dict = make_feed_dict(net.extern_data)
        session.run(out_data.placeholder, feed_dict=feed_dict)
        params = net.get_params_list()
        print(params)
        assert len(params) == 2
        assert_equal(set(p.name for p in params), {"output/W:0", "output/b:0"})


def test_MaskedComputationLayer_rec_name_scope():
    with make_scope() as session:
        from returnn.tf.util.data import batch_dim

        time_dim = SpatialDim("time")
        config = Config(
            {
                "extern_data": {
                    "data": {"dim": 20, "sparse": True, "dim_tags": (batch_dim, time_dim)},
                    "mask": {"dim": 2, "dtype": "bool", "sparse": True, "dim_tags": (batch_dim, time_dim)},
                }
            }
        )
        net = TFNetwork(config=config)
        net_dict = {
            "output": {
                "class": "masked_computation",
                "from": "data",
                "mask": "data:mask",
                "unit": {"class": "rec", "unit": "lstm", "from": "data", "n_out": 7},
            },
        }
        net.construct_from_dict(net_dict)
        net.initialize_params(session)
        out_data = net.get_layer("output").output.copy_as_batch_major()
        from test_TFNetworkLayer import make_feed_dict

        feed_dict = make_feed_dict(net.extern_data)
        session.run(out_data.placeholder, feed_dict=feed_dict)
        params = net.get_params_list()
        print(params)
        assert len(params) == 3
        assert_equal(set(p.name for p in params), {"output/rec/W:0", "output/rec/W_re:0", "output/rec/b:0"})


def test_MaskedComputationLayer_subnet_name_scope():
    with make_scope() as session:
        from returnn.tf.util.data import batch_dim

        time_dim = SpatialDim("time")
        config = Config(
            {
                "extern_data": {
                    "data": {"dim": 20, "sparse": True, "dim_tags": (batch_dim, time_dim)},
                    "mask": {"dim": 2, "dtype": "bool", "sparse": True, "dim_tags": (batch_dim, time_dim)},
                }
            }
        )
        net = TFNetwork(config=config)
        net_dict = {
            "output": {
                "class": "masked_computation",
                "from": "data",
                "mask": "data:mask",
                "unit": {
                    "class": "subnetwork",
                    "from": "data",
                    "subnetwork": {
                        "linear": {"class": "linear", "from": "data", "n_out": 5},
                        "output": {"class": "copy", "from": "linear"},
                    },
                },
            },
        }
        net.construct_from_dict(net_dict)
        net.initialize_params(session)
        out_data = net.get_layer("output").output.copy_as_batch_major()
        from test_TFNetworkLayer import make_feed_dict

        feed_dict = make_feed_dict(net.extern_data)
        session.run(out_data.placeholder, feed_dict=feed_dict)
        params = net.get_params_list()
        print(params)
        assert len(params) == 2
        assert_equal(set(p.name for p in params), {"output/linear/W:0", "output/linear/b:0"})


def test_MaskedComputationLayer_rec_subnet_name_scope():
    with make_scope() as session:
        from returnn.tf.util.data import batch_dim

        time_dim = SpatialDim("time")
        feat_dim = FeatureDim("input", 5)
        config = Config(
            {
                "extern_data": {
                    "data": {"dim_tags": (batch_dim, time_dim, feat_dim)},
                    "mask": {
                        "dim": 2,
                        "dtype": "bool",
                        "sparse": True,
                        "dim_tags": (batch_dim, time_dim),
                        "available_for_inference": True,
                    },
                }
            }
        )
        net = TFNetwork(config=config)
        net_dict = {
            "output": {
                "class": "rec",
                "from": "data",
                "name_scope": "",
                "unit": {
                    "mask": {"class": "rec_unstack", "from": "base:data:mask"},
                    "masked_comp": {
                        "class": "masked_computation",
                        "from": "prev:recurrent",
                        "mask": "mask",
                        "name_scope": "",
                        "unit": {
                            "class": "subnetwork",
                            "from": "data",
                            "subnetwork": {
                                "linear": {"class": "linear", "from": "data", "out_dim": feat_dim},
                                "output": {"class": "copy", "from": "linear"},
                            },
                        },
                    },
                    "recurrent": {"class": "combine", "kind": "add", "from": ["masked_comp", "data:source"]},
                    "output": {"class": "copy", "from": "masked_comp"},
                },
            },
        }
        net.construct_from_dict(net_dict)
        net.initialize_params(session)
        out_data = net.get_layer("output").output.copy_as_batch_major()
        from test_TFNetworkLayer import make_feed_dict

        feed_dict = make_feed_dict(net.extern_data)
        session.run(out_data.placeholder, feed_dict=feed_dict)
        params = net.get_params_list()
        print(params)
        assert len(params) == 2
        assert_equal(set(p.name for p in params), {"linear/W:0", "linear/b:0"})


def test_MaskedComputationLayer_dyn_size_none():
    # https://github.com/rwth-i6/returnn/issues/1008
    with make_scope() as session:
        config = Config()
        net = TFNetwork(
            extern_data=ExternData({"data": {"dim": 20, "sparse": False}, "classes": {"dim": 20, "sparse": True}}),
            config=config,
            train_flag=False,
            search_flag=True,
        )
        net_dict = {
            "rec_loop": {
                "class": "rec",
                "from": "data",
                "unit": {
                    "lin": {"class": "linear", "activation": "softmax", "from": "prev:output", "n_out": 20},
                    "output": {
                        "class": "choice",
                        "from": "lin",
                        "beam_size": 4,
                        "target": "classes",
                        "initial_output": 0,
                    },
                },
            },
            "data_red": {"class": "reduce", "mode": "mean", "from": "data", "axis": "f"},
            "mask": {"class": "compare", "from": "rec_loop", "value": 10, "kind": "greater"},
            "masked_data": {
                "class": "masked_computation",
                "from": "rec_loop",
                "mask": "mask",
                "unit": {"class": "copy", "from": "data"},
            },
            "output": {"class": "decide", "from": "masked_data"},
        }
        net.construct_from_dict(net_dict)


def test_subnet_deps_search():
    beam_size = 3
    EncKeyTotalDim = 10
    AttNumHeads = 2
    target = "classes"
    num_classes = 13
    from test_TFNetworkLayer import make_feed_dict

    net_dict = {
        "encoder": {"class": "linear", "from": "data", "activation": "relu", "n_out": EncKeyTotalDim},
        "enc_ctx": {"class": "copy", "from": "encoder"},
        "enc_value": {"class": "copy", "from": "encoder"},
        "inv_fertility": {"class": "linear", "activation": "sigmoid", "from": "encoder", "n_out": AttNumHeads},
        "output": {
            "class": "rec",
            "from": [],
            "max_seq_len": "max_len_from('base:encoder')",
            "target": target,
            "unit": {
                "FF_0": {
                    "L2": 0.0005,
                    "activation": "tanh",
                    "class": "linear",
                    "from": ["prev:target_embed", "prev:prev_1_target_embed", "prev:prev_2_target_embed", "prev:att"],
                    "n_out": 10,
                    "with_bias": True,
                },
                "accum_att_weights": {
                    "class": "eval",
                    "eval": "source(0) + source(1) * source(2) * 0.5",
                    "from": ["prev:accum_att_weights", "att_weights", "base:inv_fertility"],
                    "out_type": {"dim": AttNumHeads, "shape": (None, AttNumHeads)},
                },
                "att": {
                    "axes": ["dim:%i" % AttNumHeads, "dim:%i" % EncKeyTotalDim],
                    "class": "merge_dims",
                    "from": "att0",
                },
                "att0": {
                    "base": "base:enc_value",
                    "class": "generic_attention",
                    "weights": "att_weights",
                    "auto_squeeze": False,
                },
                "att_weights": {"class": "softmax_over_spatial", "from": "energy"},
                "combo_output_prob": {
                    "class": "eval",
                    "eval": "safe_log(source(0)) - 0.22 * safe_log(source(1))",
                    "from": ["output_prob", "prior_output_prob"],
                },
                "end": {"class": "compare", "from": "output", "kind": "equal", "value": 0},
                "energy": {
                    "activation": None,
                    "class": "linear",
                    "from": "energy_tanh",
                    "n_out": AttNumHeads,
                    "with_bias": False,
                },
                "energy_in": {
                    "class": "combine",
                    "from": ["base:enc_ctx", "weight_feedback", "FF_0"],
                    "kind": "add",
                    "n_out": EncKeyTotalDim,
                },
                "energy_tanh": {"activation": "tanh", "class": "activation", "from": "energy_in"},
                "output": {
                    "beam_size": beam_size,
                    "class": "choice",
                    "from": "combo_output_prob",
                    "initial_output": 0,
                    "input_type": "log_prob",
                    "target": target,
                },
                "output_prob": {
                    "L2": 0.0005,
                    "class": "softmax",
                    "dropout": 0.3,
                    "from": "readout",
                    "loss": "ce",
                    "loss_opts": {"label_smoothing": 0.1},
                    "target": target,
                },
                "prev_1_target_embed": {"class": "copy", "from": "prev:target_embed"},
                "prev_2_target_embed": {"class": "copy", "from": "prev:prev_1_target_embed"},
                "prior_lm_output": {
                    "class": "subnetwork",
                    "from": "prev:output",
                    "subnetwork": {
                        "input": {"activation": "identity", "class": "linear", "n_out": 13, "from": "data"},
                        "lstm0": {
                            "L2": 0.0,
                            "class": "rec",
                            "direction": 1,
                            "dropout": 0.2,
                            "from": ["input"],
                            "n_out": 10,
                            "unit": "lstm",
                        },
                        "output": {
                            "activation": "identity",
                            "class": "linear",
                            "dropout": 0.2,
                            "from": ["lstm0"],
                            "n_out": num_classes,
                        },
                    },
                },
                "prior_output_prob": {
                    "activation": "softmax",
                    "class": "activation",
                    "from": "prior_lm_output",
                    "target": target,
                },
                "readout": {"class": "reduce_out", "from": "readout_in", "mode": "max", "num_pieces": 2},
                "readout_in": {
                    "activation": None,
                    "class": "linear",
                    "from": ["FF_0", "prev:target_embed", "att"],
                    "n_out": 10,
                    "with_bias": True,
                },
                "target_embed": {
                    "activation": None,
                    "class": "linear",
                    "from": "output",
                    "initial_output": 0,
                    "n_out": 13,
                    "with_bias": False,
                },
                "weight_feedback": {
                    "activation": None,
                    "class": "linear",
                    "from": "prev:accum_att_weights",
                    "n_out": 10,
                    "with_bias": False,
                },
            },
        },
    }
    with make_scope() as session:
        config = Config({"debug_print_layer_output_template": True})
        extern_data = ExternData({"data": {"dim": 20, "sparse": True}, target: {"dim": num_classes, "sparse": True}})
        feed_dict = make_feed_dict(extern_data)

        print("***** Construct search net.")
        search_net = TFNetwork(extern_data=extern_data, config=config, search_flag=True)
        search_net.construct_from_dict(net_dict)
        session.run(tf_compat.v1.global_variables_initializer())
        out = search_net.get_default_output_layer().output.placeholder
        print("out:", session.run(out, feed_dict=feed_dict))


def test_untrainable_sublayers():
    with make_scope() as session:
        config = Config()
        n_in, n_out = 2, 3
        net_dict = {
            "source_embed": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "n_out": 6,
                "from": "data:data",
            },
            "lstm0_fw": {
                "class": "rec",
                "unit": "standardlstm",
                "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                "initial_state": "var",
                "n_out": 10,
                "direction": 1,
                "from": ["source_embed"],
            },
            "lstm0_bw": {
                "class": "rec",
                "unit": "standardlstm",
                "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                "initial_state": "var",
                "n_out": 10,
                "direction": -1,
                "from": ["source_embed"],
            },
            "lstm1_fw": {
                "class": "rec",
                "unit": "standardlstm",
                "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                "initial_state": "var",
                "n_out": 10,
                "direction": 1,
                "from": ["lstm0_fw", "lstm0_bw"],
            },
            "lstm1_bw": {
                "class": "rec",
                "unit": "standardlstm",
                "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                "initial_state": "var",
                "n_out": 10,
                "direction": -1,
                "from": ["lstm0_fw", "lstm0_bw"],
            },
            "encoder": {"class": "copy", "from": ["lstm1_fw", "lstm1_bw"]},
            "enc_ctx": {"class": "linear", "activation": None, "with_bias": True, "from": ["encoder"], "n_out": 10},
            "fertility": {
                "class": "linear",
                "activation": "sigmoid",
                "with_bias": False,
                "from": ["encoder"],
                "n_out": 1,
            },
            "output": {
                "class": "rec",
                "from": [],
                "unit": {
                    "output": {
                        "class": "choice",
                        "target": "classes",
                        "beam_size": 12,
                        "from": ["output_prob"],
                        "initial_output": 0,
                    },
                    "end": {"class": "compare", "from": ["output"], "value": 0},
                    "target_embed": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["output"],
                        "n_out": 6,
                        "initial_output": "apply(0)",
                        "trainable": False,
                    },
                    "weight_feedback": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["prev:accum_att_weights"],
                        "n_out": 10,
                    },
                    "prev_s_state": {"class": "get_last_hidden_state", "from": ["prev:s"], "n_out": 20},
                    "prev_s_transformed": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["prev_s_state"],
                        "n_out": 10,
                        "trainable": False,
                    },
                    "energy_in": {
                        "class": "combine",
                        "kind": "add",
                        "from": ["base:enc_ctx", "weight_feedback", "prev_s_transformed"],
                        "n_out": 10,
                    },
                    "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
                    "energy": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["energy_tanh"],
                        "n_out": 1,
                    },
                    "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, enc-T, 1)
                    "accum_att_weights": {
                        "class": "eval",
                        "from": ["prev:accum_att_weights", "att_weights", "base:fertility"],
                        "eval": "source(0) + source(1) / (2.0 * source(2))",
                        "out_type": {"dim": 1, "shape": (None, 1)},
                    },
                    "att": {
                        "class": "generic_attention",
                        "weights": "att_weights",
                        "base": "base:encoder",
                        "auto_squeeze": True,
                    },
                    "s": {
                        "class": "rnn_cell",
                        "unit": "standardlstm",
                        "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                        "initial_state": "var",
                        "from": ["target_embed", "att"],
                        "n_out": 10,
                    },
                    "readout_in": {
                        "class": "linear",
                        "from": ["prev:s", "prev:target_embed", "att"],
                        "activation": None,
                        "n_out": 10,
                        "trainable": False,
                    },
                    "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
                    "output_prob": {"class": "softmax", "from": ["readout"], "target": "classes", "loss": "ce"},
                },
                "target": "classes",
                "max_seq_len": 20,
            },
            "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance", "target": "classes"},
        }
        config.update({"num_outputs": n_out, "num_inputs": n_in, "network": net_dict})
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_dict["network"])
        weight_input_layer_moved_out = network.layers["output"].params["target_embed/W"]
        assert weight_input_layer_moved_out not in set(network.get_trainable_params())

        weight_output_layer_moved_out = network.layers["output"].params["readout_in/W"]
        assert weight_output_layer_moved_out not in set(network.get_trainable_params())

        weight_internal = network.layers["output"].params["prev_s_transformed/W"]
        assert weight_internal not in set(network.get_trainable_params())


def test_untrainable_reclayer():
    with make_scope() as session:
        config = Config()
        n_in, n_out = 2, 3
        net_dict = {
            "source_embed": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "n_out": 6,
                "from": "data:data",
            },
            "lstm0_fw": {
                "class": "rec",
                "unit": "standardlstm",
                "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                "initial_state": "var",
                "n_out": 10,
                "direction": 1,
                "from": ["source_embed"],
            },
            "lstm0_bw": {
                "class": "rec",
                "unit": "standardlstm",
                "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                "initial_state": "var",
                "n_out": 10,
                "direction": -1,
                "from": ["source_embed"],
            },
            "lstm1_fw": {
                "class": "rec",
                "unit": "standardlstm",
                "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                "initial_state": "var",
                "n_out": 10,
                "direction": 1,
                "from": ["lstm0_fw", "lstm0_bw"],
            },
            "lstm1_bw": {
                "class": "rec",
                "unit": "standardlstm",
                "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                "initial_state": "var",
                "n_out": 10,
                "direction": -1,
                "from": ["lstm0_fw", "lstm0_bw"],
            },
            "encoder": {"class": "copy", "from": ["lstm1_fw", "lstm1_bw"]},
            "enc_ctx": {"class": "linear", "activation": None, "with_bias": True, "from": ["encoder"], "n_out": 10},
            "fertility": {
                "class": "linear",
                "activation": "sigmoid",
                "with_bias": False,
                "from": ["encoder"],
                "n_out": 1,
            },
            "output": {
                "class": "rec",
                "from": [],
                "trainable": False,
                "unit": {
                    "output": {
                        "class": "choice",
                        "target": "classes",
                        "beam_size": 12,
                        "from": ["output_prob"],
                        "initial_output": 0,
                    },
                    "end": {"class": "compare", "from": ["output"], "value": 0},
                    "target_embed": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["output"],
                        "n_out": 6,
                        "initial_output": "apply(0)",
                        "trainable": True,
                    },
                    "weight_feedback": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["prev:accum_att_weights"],
                        "n_out": 10,
                    },
                    "prev_s_state": {"class": "get_last_hidden_state", "from": ["prev:s"], "n_out": 20},
                    "prev_s_transformed": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["prev_s_state"],
                        "n_out": 10,
                        "trainable": True,
                    },
                    "energy_in": {
                        "class": "combine",
                        "kind": "add",
                        "from": ["base:enc_ctx", "weight_feedback", "prev_s_transformed"],
                        "n_out": 10,
                    },
                    "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
                    "energy": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["energy_tanh"],
                        "n_out": 1,
                    },
                    "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, enc-T, 1)
                    "accum_att_weights": {
                        "class": "eval",
                        "from": ["prev:accum_att_weights", "att_weights", "base:fertility"],
                        "eval": "source(0) + source(1) / (2.0 * source(2))",
                        "out_type": {"dim": 1, "shape": (None, 1)},
                    },
                    "att": {
                        "class": "generic_attention",
                        "weights": "att_weights",
                        "base": "base:encoder",
                        "auto_squeeze": True,
                    },
                    "s": {
                        "class": "rnn_cell",
                        "unit": "standardlstm",
                        "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                        "initial_state": "var",
                        "from": ["target_embed", "att"],
                        "n_out": 10,
                    },
                    "readout_in": {
                        "class": "linear",
                        "from": ["prev:s", "prev:target_embed", "att"],
                        "activation": None,
                        "n_out": 10,
                        "trainable": True,
                    },
                    "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
                    "output_prob": {"class": "softmax", "from": ["readout"], "target": "classes", "loss": "ce"},
                },
                "target": "classes",
                "max_seq_len": 20,
            },
            "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance", "target": "classes"},
        }
        config.update({"num_outputs": n_out, "num_inputs": n_in, "network": net_dict})
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_dict["network"])
        weight_input_layer_moved_out = network.layers["output"].params["target_embed/W"]
        assert weight_input_layer_moved_out not in set(network.get_trainable_params())

        weight_output_layer_moved_out = network.layers["output"].params["readout_in/W"]
        assert weight_output_layer_moved_out not in set(network.get_trainable_params())

        weight_internal = network.layers["output"].params["prev_s_transformed/W"]
        assert weight_internal not in set(network.get_trainable_params())


def test_trainable_sublayers():
    with make_scope() as session:
        config = Config()
        n_in, n_out = 2, 3
        net_dict = {
            "source_embed": {
                "class": "linear",
                "activation": None,
                "with_bias": False,
                "n_out": 6,
                "from": "data:data",
            },
            "lstm0_fw": {
                "class": "rec",
                "unit": "standardlstm",
                "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                "initial_state": "var",
                "n_out": 10,
                "direction": 1,
                "from": ["source_embed"],
            },
            "lstm0_bw": {
                "class": "rec",
                "unit": "standardlstm",
                "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                "initial_state": "var",
                "n_out": 10,
                "direction": -1,
                "from": ["source_embed"],
            },
            "lstm1_fw": {
                "class": "rec",
                "unit": "standardlstm",
                "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                "initial_state": "var",
                "n_out": 10,
                "direction": 1,
                "from": ["lstm0_fw", "lstm0_bw"],
            },
            "lstm1_bw": {
                "class": "rec",
                "unit": "standardlstm",
                "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                "initial_state": "var",
                "n_out": 10,
                "direction": -1,
                "from": ["lstm0_fw", "lstm0_bw"],
            },
            "encoder": {"class": "copy", "from": ["lstm1_fw", "lstm1_bw"]},
            "enc_ctx": {"class": "linear", "activation": None, "with_bias": True, "from": ["encoder"], "n_out": 10},
            "fertility": {
                "class": "linear",
                "activation": "sigmoid",
                "with_bias": False,
                "from": ["encoder"],
                "n_out": 1,
            },
            "output": {
                "class": "rec",
                "from": [],
                "unit": {
                    "output": {
                        "class": "choice",
                        "target": "classes",
                        "beam_size": 12,
                        "from": ["output_prob"],
                        "initial_output": 0,
                    },
                    "end": {"class": "compare", "from": ["output"], "value": 0},
                    "target_embed": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["output"],
                        "n_out": 6,
                        "initial_output": "apply(0)",
                    },
                    "weight_feedback": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["prev:accum_att_weights"],
                        "n_out": 10,
                    },
                    "prev_s_state": {"class": "get_last_hidden_state", "from": ["prev:s"], "n_out": 20},
                    "prev_s_transformed": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["prev_s_state"],
                        "n_out": 10,
                    },
                    "energy_in": {
                        "class": "combine",
                        "kind": "add",
                        "from": ["base:enc_ctx", "weight_feedback", "prev_s_transformed"],
                        "n_out": 10,
                    },
                    "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
                    "energy": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["energy_tanh"],
                        "n_out": 1,
                    },
                    "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, enc-T, 1)
                    "accum_att_weights": {
                        "class": "eval",
                        "from": ["prev:accum_att_weights", "att_weights", "base:fertility"],
                        "eval": "source(0) + source(1) / (2.0 * source(2))",
                        "out_type": {"dim": 1, "shape": (None, 1)},
                    },
                    "att": {
                        "class": "generic_attention",
                        "weights": "att_weights",
                        "base": "base:encoder",
                        "auto_squeeze": True,
                    },
                    "s": {
                        "class": "rnn_cell",
                        "unit": "standardlstm",
                        "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                        "initial_state": "var",
                        "from": ["target_embed", "att"],
                        "n_out": 10,
                    },
                    "readout_in": {
                        "class": "linear",
                        "from": ["prev:s", "prev:target_embed", "att"],
                        "activation": None,
                        "n_out": 10,
                    },
                    "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
                    "output_prob": {"class": "softmax", "from": ["readout"], "target": "classes", "loss": "ce"},
                },
                "target": "classes",
                "max_seq_len": 20,
            },
            "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance", "target": "classes"},
        }
        config.update({"num_outputs": n_out, "num_inputs": n_in, "network": net_dict})
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_dict["network"])
        weight_input_layer_moved_out = network.layers["output"].params["target_embed/W"]
        assert weight_input_layer_moved_out in set(network.get_trainable_params())

        weight_output_layer_moved_out = network.layers["output"].params["readout_in/W"]
        assert weight_output_layer_moved_out in set(network.get_trainable_params())

        weight_internal = network.layers["output"].params["prev_s_transformed/W"]
        assert weight_internal in set(network.get_trainable_params())


def test_subnet_keep_over_epoch_state_vars_saveable_params():
    with make_scope() as session:
        n_in, n_out = 2, 3
        config = Config(
            {
                "extern_data": {"data": {"dim": n_in}, "classes": {"dim": n_out, "sparse": True}},
            }
        )
        net_dict = {
            "base_blstm": {
                "class": "subnetwork",
                "from": "data",
                "subnetwork": {
                    "lstm0_fw": {
                        "class": "rec",
                        "unit": "lstm",
                        "initial_state": "keep_over_epoch",
                        "n_out": 10,
                        "from": "data",
                    },
                    "output": {"class": "copy", "from": "lstm0_fw"},
                },
            },
            "output": {"class": "softmax", "loss": "ce", "target": "classes", "from": "base_blstm", "n_out": n_out},
        }
        network = TFNetwork(config=config)
        network.construct_from_dict(net_dict)

        print("All global variables:")
        params = tf_compat.v1.global_variables()
        pprint(params)
        assert any("base_blstm/lstm0_fw/rec/W_re:0" == param.name for param in params)
        assert any("keep_state" in param.name for param in params)

        print("Network saveable params:")
        params = network.get_saveable_params_list()
        pprint(params)
        assert any("base_blstm/lstm0_fw/rec/W_re:0" == param.name for param in params)
        assert not any("keep_state" in param.name for param in params)


def test_OptimalCompletionsLayer():
    with make_scope() as session:
        from returnn.tf.layers.base import InternalLayer
        from returnn.tf.util.basic import expand_dims_unbroadcast

        net = TFNetwork(
            extern_data=ExternData({"target": {"dim": 20, "sparse": True}}),
            config=Config({"debug_print_layer_output_template": True}),
        )
        target = net.extern_data.data["target"]
        target_shape = tf.shape(target.placeholder)
        n_batch = target_shape[0]
        n_max_seq_len = target_shape[1]
        # Fake that we are inside a rec layer.
        net.set_rec_step_info(
            i=tf.convert_to_tensor(0, name="i"),
            prev_end_flag=expand_dims_unbroadcast(tf.convert_to_tensor(False), 0, n_batch),
            seq_lens=target.get_sequence_lengths(),
        )
        kwargs = dict(
            name="opt_completions",
            network=net,
            debug=True,
            target="target",
            sources=[
                InternalLayer(
                    name="last_row",
                    network=net,
                    output=Data(
                        name="last_row",
                        shape=(None,),
                        dtype="int32",
                        placeholder=expand_dims_unbroadcast(tf.range(n_max_seq_len + 1), 0, n_batch),
                        size_placeholder={0: target.get_sequence_lengths() + 1},
                    ),
                )
            ],
        )
        print("OptimalCompletionsLayer kwargs:")
        pprint(kwargs)
        kwargs["output"] = OptimalCompletionsLayer.get_out_data_from_opts(**kwargs)
        layer = OptimalCompletionsLayer(**kwargs)
        layer.output.sanity_check()
        out = session.run(
            layer.output.placeholder,
            feed_dict={
                net.extern_data.get_batch_info().dim: 1,
                target.placeholder: numpy.array([[3, 7, 8, 9, 13, 13, 0]]),
                target.size_placeholder[0]: numpy.array([7]),
            },
        )
        print(out)
        assert isinstance(out, numpy.ndarray)
        assert out.shape == (1, 20)
        assert out[0, 3] == 0 and all(out[0, :3] == 1) and all(out[0, 4:] == 1)


def test_extra_scatter_nd_search_train():
    from returnn.tf.util.data import batch_dim
    from returnn.tf.layers.rec import _SubnetworkRecCell

    rnd = numpy.random.RandomState(42)
    n_batch, n_enc_time, n_in, n_dec_time, n_out = 2, 11, 5, 7, 6
    target = "classes"
    LstmDim = 13
    EncValueTotalDim = LstmDim
    EncKeyTotalDim = LstmDim
    AttNumHeads = 1
    beam_size = 3

    def t_linear(source, **kwargs):
        import tensorflow as tf
        from returnn.tf.util.basic import where_bc

        enc = source(1, as_data=True, auto_convert=False)
        dec = source(0, as_data=True, auto_convert=False)
        enc_lens = enc.get_sequence_lengths()
        dec_lens = dec.get_sequence_lengths()
        dec_shape = tf.shape(dec.placeholder)
        dec_time_dim = dec_shape[dec.time_dim_axis]
        dec_times = tf.expand_dims(tf.range(dec_time_dim), axis=0)  # (1,dec-T)
        x = tf.cast(dec_times + 1, tf.float32)  # (1,dec-T)
        # We want: x[dec_len - 1] == enc_time - 1.
        factors = tf.maximum(tf.cast(enc_lens - 1, tf.float32), 0.0) / tf.maximum(
            tf.cast(dec_lens, tf.float32), 1.0
        )  # (B,)
        factors = tf.expand_dims(factors, axis=1)  # (B,1)
        x = x * factors  # (B,dec-T)
        x = tf.cast(tf.round(x), tf.int32)
        x = tf.minimum(x, tf.expand_dims(enc_lens - 1, axis=1))
        # fix cheating gold targets with end flag filter. must be 0
        x = where_bc(tf.less(dec_times, tf.expand_dims(dec_lens, axis=1)), x, 0)
        return x

    net_dict = {
        "lstm0_fw": {"class": "rec", "unit": "nativelstm2", "n_out": LstmDim, "direction": 1, "from": "data"},
        "lstm0_bw": {"class": "rec", "unit": "nativelstm2", "n_out": LstmDim, "direction": -1, "from": "data"},
        "lstm0_pool": {
            "class": "pool",
            "mode": "max",
            "padding": "same",
            "pool_size": (3,),
            "from": ["lstm0_fw", "lstm0_bw"],
        },
        "encoder0": {"class": "linear", "from": "data", "activation": "relu", "n_out": EncValueTotalDim},
        "encoder": {"class": "postfix_in_time", "postfix": 0.0, "from": "encoder0"},
        "enc_ctx": {
            "class": "linear",
            "activation": None,
            "with_bias": True,
            "from": "encoder",
            "n_out": EncKeyTotalDim,
        },
        "enc_value": {"class": "copy", "from": "encoder"},  # (B, enc-T, D)
        "enc0_seq_len": {"class": "length", "from": "encoder0", "sparse": True},
        "decision": {"class": "decide", "from": "extra.search_label:output"},  # for search task
        "extra.search1:t_search": {"class": "decide", "from": "extra.search1:output/t"},
        "extra.search2:t_search": {"class": "decide", "from": "extra.search2:output/t"},
        "0_data_target0": {
            "class": "postfix_in_time",
            "postfix": 0,
            "from": "data:%s" % target,
            "register_as_extern_data": "target0",
        },
        "1_data_t_linear": {
            "class": "eval",
            "from": ["data:target0", "encoder"],
            "eval": t_linear,
            "out_type": {
                "batch_dim_axis": 0,
                "time_dim_axis": 1,
                "shape": (None,),
                "sparse": True,
                "dtype": "int32",
                "dim": None,
            },
            "size_target": "target0",
            "register_as_extern_data": "t_linear",  # if task == "train" else None
        },
        "2_data_t_search_target1": {
            "class": "copy",
            "from": "extra.search1:t_search",
            "register_as_extern_data": "t_search_target1",  # if task == "train" else None
        },
        "2_data_t_search_target2": {
            "class": "copy",
            "from": "extra.search2:t_search",
            "register_as_extern_data": "t_search_target2",  # if task == "train" else None
        },
    }

    def get_output_dict(train, t_search, label_search, backprop, t_target, use_soft_att):
        """
        :param bool train:
        :param bool t_search:
        :param bool label_search:
        :param bool backprop:
        :param str|None t_target:
        :param bool use_soft_att:
        :rtype: dict[str]
        """
        if label_search:
            assert not t_target
        if t_target:
            assert not label_search

        def combine_soft_hard_att(self, source, **kwargs):
            # source(0) is hard att, source(1) is soft att
            print("combine_soft_hard_att, use soft att: %r" % use_soft_att)
            if use_soft_att:
                frac = 0.5
                return source(0) * frac + source(1) * (1.0 - frac)
            else:
                source(1)  # call, but ignore
                return source(0)  # only hard att

        t_rel_idxs_dim = Dim(kind=Dim.Types.Spatial, dimension=6, description="t_rel_idxs_dim")
        return {
            "class": "rec",
            "from": [],
            "back_prop": backprop,
            "unit": {
                "s_transformed": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["s"],
                    "n_out": EncKeyTotalDim,
                },
                "t_rel_var": {"class": "variable", "shape": (t_rel_idxs_dim, EncKeyTotalDim)},
                "t_rel_idxs_": {"class": "range", "limit": 6, "out_spatial_dim": t_rel_idxs_dim},
                "prev_t_": {"class": "reinterpret_data", "set_sparse": False, "from": "prev:t"},
                "t_rel_idxs": {
                    "class": "combine",
                    "kind": "add",
                    "from": ["prev_t_", "t_rel_idxs_"],
                    "out_shape": {batch_dim, t_rel_idxs_dim},
                },
                "energy_in": {
                    "class": "combine",
                    "kind": "add",
                    "from": ["base:enc_ctx", "s_transformed", "energy_in_t_rel_var"],
                    "n_out": EncKeyTotalDim,
                },
                "energy_in_t_rel_var": {
                    "class": "scatter_nd",
                    "from": "t_rel_var",
                    "position": "t_rel_idxs",
                    "position_axis": "dim:6",
                    "output_dim_via_time_from": "base:enc_ctx",
                    "filter_invalid_indices": True,
                },
                "energy_tanh": {"class": "activation", "activation": "tanh", "from": "energy_in"},
                "energy": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["energy_tanh"],
                    "n_out": AttNumHeads,
                },  # (B, enc-T, H)
                "energy1": {"class": "squeeze", "axis": "f", "from": "energy"},  # (B, enc-T)
                "energy2": {"class": "reinterpret_data", "from": "energy1", "set_axes": {"t": "stag:extern_data:data"}},
                "att_weights": {"class": "softmax_over_spatial", "from": "energy2", "start": "t_start"},  # (B, enc-T)
                # ChoiceLayer works on the feature axis.
                "att_weights1": {
                    "class": "reinterpret_data",
                    "from": "att_weights",
                    "set_axes": {"f": "stag:extern_data:data"},
                    "target": t_target if train else None,
                    "loss": "ce" if (train and t_target) else None,
                },
                "t0": {
                    "class": "choice",
                    "from": "att_weights1",
                    # "target": None,
                    "target": t_target,
                    "cheating": bool(t_target),  # add this in training
                    "beam_size": beam_size,
                    "length_normalization": False,
                    "initial_output": -1,
                },  # (B,)
                # Note: If beam-size > enc_seq_len, we end up with invalid t in the beam. Fix that.
                "t1": {
                    "class": "eval",
                    "from": ["t0", "base:enc0_seq_len"],
                    "eval": "tf.minimum(source(0), source(1))",
                },
                "t": {
                    # "class": "print",
                    "class": "copy",
                    "from": "t1",
                    "initial_output": -1,
                    "is_output_layer": bool(t_search),
                },
                # Only for debugging.
                "t_err": {
                    "class": "eval",
                    "from": ["t", "data:%s" % t_target],
                    "collocate_with": "t",
                    "eval": "tf.cast(tf.abs(source(0) - source(1)), tf.float32)",
                    "loss": "as_is" if (t_target and t_search) else None,
                    "out_type": {"dtype": "float32"},
                    "only_on_search": True,
                },
                "t_start": {
                    # Need right start for masking to avoid infs.
                    "class": "eval",
                    "from": ["prev:t", "data:%s" % t_target],
                    "eval": "tf.minimum(source(0), source(1))",
                }
                if t_target
                else {"class": "copy", "from": "prev:t"},
                "att0": {"class": "gather_nd", "position": "t", "from": "base:enc_value"},  # (B, V)
                "att1": {"class": "generic_attention", "weights": "att_weights", "base": "base:enc_value"},  # (B, V)
                "att1_": {
                    "class": "switch",
                    "condition": lambda **kw: use_soft_att,
                    "true_from": "att1",
                    "false_from": "att0",
                },
                "att": {"class": "eval", "from": ["att0", "att1_"], "eval": combine_soft_hard_att},
                "s": {"class": "rec", "unit": "nativelstm2", "from": ["prev:target_embed", "prev:att"], "n_out": 8},
                "readout_in": {
                    "class": "linear",
                    "from": ["s", "prev:target_embed", "att"],
                    "activation": None,
                    "n_out": 10,
                },
                "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
                "output_prob": {
                    "class": "softmax",
                    "from": ["readout"],
                    "target": "target0",
                    "loss": "ce" if train else None,
                },
                "target_embed": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["output"],
                    "n_out": 6,
                    "initial_output": "var",
                },
                "output": {
                    "class": "choice",
                    "target": "target0",
                    "beam_size": beam_size,
                    "from": ["output_prob"],
                    "initial_output": 0,
                    "search": label_search,
                    "length_normalization": label_search,
                },
                "end": {"class": "compare", "from": "output", "value": 0},
            },
            "target": ["target0", t_target] if t_target else ["target0"],
            "size_target": t_target,
            "max_seq_len": "max_len_from('base:encoder0')",
        }

    # Train task:
    net_dict["extra.search1:output"] = get_output_dict(
        train=False, t_search=True, label_search=False, backprop=False, t_target="t_linear", use_soft_att=True
    )
    net_dict["extra.search2:output"] = get_output_dict(
        train=False, t_search=True, label_search=False, backprop=False, t_target="t_linear", use_soft_att=False
    )
    net_dict["extra.1:output"] = get_output_dict(
        train=True, t_search=False, label_search=False, backprop=True, t_target="t_linear", use_soft_att=True
    )
    # extra.2 is basically like extra.1, only different t_target, and that should not make any difference for the
    # construction. But anyway, put it in as another variation.
    net_dict["extra.2:output"] = get_output_dict(
        train=True, t_search=False, label_search=False, backprop=True, t_target="t_search_target1", use_soft_att=True
    )
    # extra.3 does not use soft-attention anymore. That enables a couple of new optimizations in the rec loop,
    # esp now we should be able to move *everything* out.
    net_dict["extra.3:output"] = get_output_dict(
        train=True, t_search=False, label_search=False, backprop=True, t_target="t_search_target2", use_soft_att=False
    )
    # Search task:
    # net_dict["extra.search_label:output"] = get_output_dict(
    #   train=True, t_search=True, label_search=True, backprop=False, t_target=None, use_soft_att=False)

    config = Config()
    config.update(
        {
            "extern_data": {"data": {"dim": n_in}, target: {"dim": n_out, "sparse": True}},
            "debug_print_layer_output_template": True,
        }
    )

    with make_scope() as session:
        network = TFNetwork(config=config, train_flag=True)
        pprint(network.extern_data.data)
        network.construct_from_dict(net_dict)

        fetches = network.get_fetches_dict()
        data_input = network.extern_data.data["data"]
        data_target = network.extern_data.data[target]
        assert data_input.batch_shape == (None, None, n_in) and data_target.batch_shape == (None, None)

        train1_search_out = network.get_layer("extra.search1:output").output
        train1_out = network.get_layer("extra.1:output").output
        train2_search_out = network.get_layer("extra.search2:output").output
        train2_out = network.get_layer("extra.2:output").output
        train3_out_layer = network.get_layer("extra.3:output")
        train3_out = train3_out_layer.output
        # search_out = network.get_layer("extra.search_label:output").output

        assert isinstance(train3_out_layer, RecLayer)
        train3_out_layer_cell = train3_out_layer.cell
        assert isinstance(train3_out_layer_cell, _SubnetworkRecCell)
        assert not train3_out_layer_cell.layers_in_loop, "all should be moved out"

        session.run(
            tf_compat.v1.variables_initializer(tf_compat.v1.global_variables() + [network.global_train_step_var])
        )
        outputs = [
            train1_search_out.placeholder,
            train1_out.placeholder,
            train2_search_out.placeholder,
            train2_out.placeholder,
            train3_out.placeholder,
        ]
        info, out = session.run(
            (fetches, outputs),
            feed_dict={
                network.extern_data.get_batch_info().dim: n_batch,
                data_input.placeholder: rnd.normal(size=(n_batch, n_enc_time, n_in)).astype("float32"),
                data_input.size_placeholder[0]: numpy.array([n_enc_time] * n_batch, dtype="int32"),
                data_target.placeholder: rnd.randint(
                    0,
                    n_out,
                    size=(
                        n_batch,
                        n_dec_time,
                    ),
                    dtype="int32",
                ),
                data_target.size_placeholder[0]: numpy.array([n_dec_time] * n_batch, dtype="int32"),
            },
        )
        print(info)
        print(out)  # random...


def test_trafo_search_lm():
    rnd = numpy.random.RandomState(42)
    beam_size = 5
    ff_dim = 7
    num_heads = 2
    emb_dim = 5
    qk_dim = 6
    v_dim = qk_dim
    trans_out_dim = qk_dim

    net_dict = {
        "input": {"class": "slice", "axis": "T", "slice_end": -1, "from": "data"},
        "input_with_new_len": {"class": "reinterpret_data", "from": "input", "size_base": "data"},
        "target": {"class": "slice", "axis": "T", "slice_start": 1, "from": "data"},
        "target_with_new_len": {
            "class": "reinterpret_data",
            "from": "target",
            "size_base": "data",
            "register_as_extern_data": "targets",
        },
        "decision": {
            "class": "decide",
            "from": ["output"],
            "loss": "edit_distance",
            "target": "targets",
            "is_output_layer": True,
        },
        "output": {
            "class": "rec",
            "from": [],
            "target": "data",
            "max_seq_len": "max_len_from('data') * 3",
            "unit": {
                "prefix": {
                    "class": "eval",
                    "out_type": {"time_dim_axis": None, "shape": ()},
                    "collocate_with": "output_choice",
                    "from": ["base:data"],
                    "eval": "source(0, auto_convert=False)[:, tf.minimum(self.network.get_rec_step_index(), source(0, auto_convert=False, as_data=True).time_dimension() - 1)]",
                },  # Shape (None,).
                "in_prefix": {
                    "class": "eval",
                    "from": "base:data",
                    "collocate_with": "output_choice",
                    "out_type": {"time_dim_axis": None, "shape": (), "dtype": "bool", "dim": 2},
                    # True if still in SRC.
                    "eval": "tf.less(self.network.get_rec_step_index(), source(0, as_data=True, auto_convert=False).get_sequence_lengths())",
                },  # Shape (None,).
                "output": {
                    "class": "switch",
                    "condition": "in_prefix",
                    "true_from": "prefix",
                    "false_from": "output_choice",
                    "initial_output": 0,
                },
                "end": {
                    "class": "eval",
                    "from": ["base:data", "output"],
                    "collocate_with": "output_choice",
                    "out_type": {"time_dim_axis": None, "shape": (), "dtype": "bool", "dim": 2},
                    # Arbitrary. We can check that, though.
                    "eval": "(source(1, auto_convert=False),"  # just mark as used
                    " tf.greater_equal("
                    "   self.network.get_rec_step_index(),"
                    "   source(0, as_data=True, auto_convert=False).get_sequence_lengths() * 3 // 2))[-1]",
                },
                "output_choice": {
                    "class": "choice",
                    "target": "targets",
                    "beam_size": beam_size,
                    "from": ["prob_output"],
                    "initial_output": 0,
                },
                "prob_output": {
                    "class": "softmax",
                    "from": ["decoder"],
                    "loss": "ce",
                    "target": "targets",
                    "with_bias": True,
                },
                "decoder": {"class": "layer_norm", "from": ["dec_0"]},
                "dec_0": {"class": "copy", "from": ["dec_0_ff_out"]},
                "target_embed_raw": {
                    "activation": None,
                    "class": "linear",
                    "from": ["prev:output"],  # Note: Here was the bug.
                    "n_out": emb_dim,
                    "with_bias": False,
                },
                "target_embed": {"class": "dropout", "dropout": 0, "from": ["target_embed_raw"]},
                "target_embed_lin": {
                    "activation": None,
                    "class": "linear",
                    "from": ["target_embed"],
                    "n_out": trans_out_dim,
                    "with_bias": False,
                },
                "dec_0_self_att_laynorm": {"class": "layer_norm", "from": ["target_embed_lin"]},
                "dec_0_self_att_att": {
                    "attention_left_only": True,
                    "class": "self_attention",
                    "from": ["dec_0_self_att_laynorm"],
                    "n_out": v_dim,
                    "num_heads": num_heads,
                    "total_key_dim": qk_dim,
                },
                "dec_0_self_att_lin": {
                    "activation": None,
                    "class": "linear",
                    "from": ["dec_0_self_att_att"],
                    "n_out": trans_out_dim,
                    "with_bias": False,
                },
                "dec_0_self_att_drop": {"class": "dropout", "dropout": 0, "from": ["dec_0_self_att_lin"]},
                "dec_0_att_out": {
                    "class": "combine",
                    "from": ["target_embed_lin", "dec_0_self_att_drop"],
                    "kind": "add",
                    "n_out": trans_out_dim,
                    "trainable": True,
                },
                "dec_0_ff_laynorm": {"class": "layer_norm", "from": ["dec_0_att_out"]},
                "dec_0_ff_conv1": {
                    "activation": "relu",
                    "class": "linear",
                    "from": ["dec_0_ff_laynorm"],
                    "n_out": ff_dim,
                    "with_bias": True,
                },
                "dec_0_ff_conv2": {
                    "activation": None,
                    "class": "linear",
                    "from": ["dec_0_ff_conv1"],
                    "n_out": trans_out_dim,
                    "with_bias": True,
                },
                "dec_0_ff_drop": {"class": "dropout", "dropout": 0, "from": ["dec_0_ff_conv2"]},
                "dec_0_ff_out": {
                    "class": "combine",
                    "from": ["dec_0_att_out", "dec_0_ff_drop"],
                    "kind": "add",
                    "n_out": trans_out_dim,
                },
            },
        },
    }

    n_batch, n_in, n_time = 3, 19, 9
    n_out = n_in

    config = Config()
    config.update(
        {
            "extern_data": {"data": {"dim": n_out, "sparse": True}},
            "search_output_layer": "decision",
            "debug_print_layer_output_shape": True,
            "debug_print_layer_output_template": True,
        }
    )

    with make_scope() as session:
        network = TFNetwork(config=config, train_flag=False, search_flag=True)
        pprint(network.extern_data.data)
        network.construct_from_dict(net_dict)

        fetches = network.get_fetches_dict()
        data_input = network.extern_data.data["data"]
        assert data_input.batch_shape == (None, None)
        output_out = network.get_layer("decision").output
        assert (
            output_out.is_batch_major and output_out.sparse and output_out.dim == n_out and output_out.shape == (None,)
        )

        input_seq_lens = numpy.array([n_time, n_time - 5, n_time - 4], dtype="int32")
        assert input_seq_lens.shape == (n_batch,) and all(input_seq_lens > 0)
        input_seqs = rnd.randint(
            1,
            n_out,
            size=(
                n_batch,
                n_time,
            ),
            dtype="int32",
        )
        print("input:")
        print(input_seqs)
        print("lens:", input_seq_lens)

        session.run(
            tf_compat.v1.variables_initializer(tf_compat.v1.global_variables() + [network.global_train_step_var])
        )
        fetches = (fetches, output_out.placeholder, output_out.get_sequence_lengths())
        feed_dict = {
            network.extern_data.get_batch_info().dim: len(input_seq_lens),
            data_input.placeholder: input_seqs,
            data_input.size_placeholder[0]: input_seq_lens,
        }
        try:
            info, out_seqs, out_seq_lens = session.run(fetches, feed_dict=feed_dict)
        except Exception as exc:
            print("EXCEPTION:", type(exc), exc)
            help_on_tf_exception(session=session, exception=exc, fetches=fetches, feed_dict=feed_dict)
            raise
        print(info)
        print("output:")
        print(out_seqs)  # random...
        print("lens:", out_seq_lens)
        assert isinstance(out_seqs, numpy.ndarray) and isinstance(out_seq_lens, numpy.ndarray)
        assert len(out_seqs.shape) == 2 and out_seqs.shape[0] == n_batch
        assert out_seq_lens.shape == (n_batch,)

        for i in range(n_batch):
            assert out_seq_lens[i] == input_seq_lens[i] * 3 // 2  # we constructed the 'end' layer that way
            assert all(out_seqs[i, : input_seq_lens[i]] == input_seqs[i, : input_seq_lens[i]])


def test_SelfAttentionLayer_static_time():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim
    from test_TFNetworkLayer import make_feed_dict

    time_dim = SpatialDim("time", 13)
    feat_dim = FeatureDim("feat", 5)
    config = Config(
        {
            "extern_data": {"data": {"dim_tags": [batch_dim, time_dim, feat_dim], "time_dim_axis": 1}},
        }
    )

    net_dict = {
        "output": {
            "class": "self_attention",
            "from": "data",
            "n_out": 8,
            "num_heads": 2,
            "total_key_dim": 8,
        },
    }

    with make_scope() as session:
        network = TFNetwork(config=config)
        in_ = network.extern_data.get_default_input_data()
        print("in:", in_)
        network.construct_from_dict(net_dict)
        out = network.get_default_output_layer().output
        print("out:", out)
        assert out.dim_tags[:2] == in_.dim_tags[:2]
        assert out.batch_shape == (batch_dim.dimension, time_dim.dimension, 8)
        assert out.feature_dim_axis == in_.feature_dim_axis
        assert out.time_dim_axis == in_.time_dim_axis
        network.initialize_params(session)
        session.run(out.placeholder, feed_dict=make_feed_dict(network.extern_data))


def test_self_att_rec_state():
    rnd = numpy.random.RandomState(42)
    beam_size = 5
    n_batch, n_in, n_time = 3, 19, 9
    n_out = n_in

    config = Config()
    config.update(
        {
            "extern_data": {"data": {"dim": n_in, "sparse": True}, "classes": {"dim": n_out, "sparse": True}},
            "debug_print_layer_output_template": True,
        }
    )

    net_dict = {
        "encoder": {
            "class": "linear",
            "from": "data",
            "activation": None,
            "with_bias": False,
            "n_out": 7,
            "L2": 0.0001,
        },
        "encoder_red": {"class": "reduce", "axis": "t", "from": "encoder", "mode": "max"},
        "output": {
            "class": "rec",
            "from": [],
            "target": "classes",
            "max_seq_len": 13,
            "unit": {
                "end": {"class": "compare", "from": "output", "value": 0},
                "output": {
                    "class": "choice",
                    "target": "classes",
                    "beam_size": beam_size,
                    "from": "output_prob",
                    "initial_output": 0,
                },
                "output_prob": {
                    "class": "softmax",
                    "from": ["self_att", "base:encoder_red"],
                    "loss": "ce",
                    "target": "classes",
                },
                "prev_embed": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": "prev:output",
                    "n_out": 7,
                },
                "self_att": {
                    "class": "self_attention",
                    "from": "prev_embed",
                    "attention_left_only": True,
                    "n_out": 12,
                    "num_heads": 2,
                    "total_key_dim": 10,
                },
            },
        },
    }

    with make_scope() as session:
        network = TFNetwork(config=config, train_flag=False, search_flag=True)
        pprint(network.extern_data.data)
        network.construct_from_dict(net_dict)

        rec_layer = network.get_layer("output")
        assert isinstance(rec_layer, RecLayer)
        cell = rec_layer.cell
        from returnn.tf.layers.rec import _SubnetworkRecCell

        assert isinstance(cell, _SubnetworkRecCell)
        self_att_layer = cell.net.layers["self_att"]
        assert isinstance(self_att_layer, SelfAttentionLayer)
        print("Self attention layer hidden state:")
        print(self_att_layer.rec_vars_outputs)
        assert set(self_att_layer.rec_vars_outputs.keys()) == {"k_left", "v_left"}

        data_input = network.extern_data.data["data"]
        assert data_input.batch_shape == (None, None)
        output_out = rec_layer.output.copy_as_batch_major()

        input_seq_lens = numpy.array([n_time, n_time - 5, n_time - 4], dtype="int32")
        assert input_seq_lens.shape == (n_batch,) and all(input_seq_lens > 0)
        input_seqs = rnd.randint(
            1,
            n_out,
            size=(
                n_batch,
                n_time,
            ),
            dtype="int32",
        )
        print("input:")
        print(input_seqs)
        print("lens:", input_seq_lens)

        session.run(
            tf_compat.v1.variables_initializer(tf_compat.v1.global_variables() + [network.global_train_step_var])
        )
        fetches = (output_out.placeholder, output_out.get_sequence_lengths())
        feed_dict = {
            network.extern_data.get_batch_info().dim: len(input_seq_lens),
            data_input.placeholder: input_seqs,
            data_input.size_placeholder[0]: input_seq_lens,
        }
        try:
            out_seqs, out_seq_lens = session.run(fetches, feed_dict=feed_dict)
        except Exception as exc:
            print("EXCEPTION:", type(exc), exc)
            help_on_tf_exception(session=session, exception=exc, fetches=fetches, feed_dict=feed_dict)
            raise
        print("output:")
        print(out_seqs)  # random...
        print("lens:", out_seq_lens)
        assert isinstance(out_seqs, numpy.ndarray) and isinstance(out_seq_lens, numpy.ndarray)
        assert len(out_seqs.shape) == 2 and out_seqs.shape[0] == n_batch * beam_size
        assert out_seq_lens.shape == (n_batch * beam_size,)


def test_generalized_non_rec_self_attention():
    # https://github.com/rwth-i6/returnn/issues/391
    n_in = 11
    n_heads = 3
    n_key_dim_per_head = 5
    n_value_dim_per_head = 7
    n_key_dim_total = n_heads * n_key_dim_per_head
    n_value_dim_total = n_heads * n_value_dim_per_head
    time_dim = SpatialDim("time_dim")
    config = Config({"extern_data": {"data": {"dim": n_in, "same_dim_tags_as": {"T": time_dim}}}})
    net_dict_old = {
        "att_old": {
            "class": "self_attention",
            "from": "data",
            "n_out": n_value_dim_total,
            "num_heads": n_heads,
            "total_key_dim": n_key_dim_total,
            "is_output_layer": True,
        },  # [B,T,V']
    }
    new_dim = SpatialDim("new_self_att_dim")
    net_dict_new = {
        "qkv": {
            "class": "linear",
            "from": "data",
            "with_bias": False,
            "n_out": n_key_dim_total * 2 + n_value_dim_total,
        },  # [B,T,2*K'+V']
        "qkv_": {
            "class": "split_dims",
            "from": "qkv",
            "axis": "F",
            "dims": (n_heads, n_key_dim_per_head * 2 + n_value_dim_per_head),
        },
        "qkv_split": {
            "class": "split",
            "from": "qkv_",
            "size_splits": [n_key_dim_per_head, n_key_dim_per_head, n_value_dim_per_head],
        },
        "q": {"class": "copy", "from": "qkv_split/0"},  # [B,T,H,K]
        "k": {"class": "copy", "from": "qkv_split/1"},  # [B,T,H,K]
        "v": {"class": "copy", "from": "qkv_split/2"},  # [B,T,H,V]
        "q_": {"class": "eval", "from": "q", "eval": "source(0) * %f" % ((n_key_dim_total // n_heads) ** -0.5)},
        "k_": {"class": "reinterpret_data", "from": "k", "set_dim_tags": {"T": new_dim}},  # [B,T_new,H,K]
        "v_": {"class": "reinterpret_data", "from": "v", "set_dim_tags": {"T": new_dim}},  # [B,T_new,H,V]
        "energy": {
            "class": "dot",
            "from": ["q_", "k_"],
            "red1": "dim:%i" % n_key_dim_per_head,
            "red2": "dim:%i" % n_key_dim_per_head,
            "var1": time_dim,
            "var2": new_dim,
        },  # [B,H,T_new,T]
        "att_weights": {"class": "softmax_over_spatial", "from": "energy", "axis": new_dim},  # [B,H,T,T_new]
        "att": {
            "class": "dot",
            "from": ["att_weights", "v_"],
            "red1": new_dim,
            "red2": new_dim,
            "var1": time_dim,
            "var2": "dim:%i" % n_value_dim_per_head,
        },  # [B,H,T,V]
        "att_new": {
            "class": "merge_dims",
            "from": "att",
            "axes": ["dim:%i" % n_heads, "dim:%i" % n_value_dim_per_head],
            "is_output_layer": True,
        },  # [B,T,V']
    }
    with make_scope() as session:
        net = TFNetwork(config=config)
        in_data = net.extern_data.get_default_input_data()
        net.construct_from_dict(net_dict_old)
        net.construct_from_dict(net_dict_new)
        assert time_dim != new_dim
        assert time_dim == in_data.dim_tags[1]
        assert new_dim in net.get_layer("k_").output.dim_tags
        assert new_dim in net.get_layer("v_").output.dim_tags
        assert set(net.get_layer("energy").output.dim_tags).issuperset({new_dim, time_dim})
        assert time_dim in net.get_layer("att").output.dim_tags
        session.run(tf_compat.v1.variables_initializer(tf_compat.v1.global_variables() + [net.global_train_step_var]))
        from test_TFNetworkLayer import make_feed_dict

        feed_dict = make_feed_dict(net.extern_data)
        out_old_data = net.get_layer("att_old").output
        out_new_data = net.get_layer("att_new").output
        assert out_old_data.dim_tags[:2] == out_new_data.dim_tags[:2] == in_data.dim_tags[:2]
        assert out_old_data.batch_ndim == out_new_data.batch_ndim == 3
        assert out_old_data.batch_shape[-1] == out_new_data.batch_shape[-1] == n_value_dim_total
        out_old = session.run(out_old_data.placeholder, feed_dict=feed_dict)
        params_old = net.get_layer("att_old").params  # QKV
        params_new = net.get_layer("qkv").params  # W
        assert params_old and params_new and len(params_old) == len(params_new) == 1
        session.run(params_new["W"].assign(params_old["QKV"]))
        out_new = session.run(out_new_data.placeholder, feed_dict=feed_dict)
        assert isinstance(out_old, numpy.ndarray) and isinstance(out_new, numpy.ndarray)
        assert out_new.shape == out_old.shape
        n_batch, n_time, n_dim = out_new.shape
        seq_lens = session.run(in_data.get_sequence_lengths(), feed_dict=feed_dict)
        assert isinstance(seq_lens, numpy.ndarray)
        assert (n_batch,) == seq_lens.shape
        assert max(seq_lens) == n_time
        for b in range(n_batch):
            for t in range(seq_lens[b]):
                v1, v2 = out_old[b, t], out_new[b, t]
                numpy.testing.assert_allclose(v1, v2, rtol=1e-4)


def test_cumulated_attention_weights_search():
    rnd = numpy.random.RandomState(42)
    beam_size = 5
    dim = 7

    # Config works during training, but building graph raises exception during search:
    # Trying to reshape input tensor with n values into tensor with n * beam_size values
    net_dict = {
        "source_embed": {"class": "linear", "activation": None, "n_out": dim, "from": "data:data"},
        "output": {
            "class": "rec",
            "from": [],
            "max_seq_len": "max_len_from('base:source_embed') * 3",
            "target": "classes",
            "unit": {
                "target_embed": {"class": "linear", "activation": None, "from": ["prev:output"], "n_out": dim},
                "att_energy": {
                    "class": "dot",
                    "from": ["base:source_embed", "target_embed"],
                    "red1": "F",
                    "red2": "F",
                    "var1": "T",
                    "var2": "T?",
                    "add_var2_if_empty": False,
                },
                "cum_att_energy": {"class": "combine", "kind": "add", "from": ["prev:att_energy", "att_energy"]},
                "att_weights": {
                    "class": "softmax_over_spatial",
                    "from": ["cum_att_energy"],
                    "axis": "stag:extern_data:data",
                },
                "att": {"class": "generic_attention", "base": "base:source_embed", "weights": "att_weights"},
                "output_prob": {"class": "softmax", "from": ["att"], "loss": "ce", "target": "classes"},
                "output": {
                    "beam_size": beam_size,
                    "class": "choice",
                    "from": ["output_prob"],
                    "initial_output": 0,
                    "target": "classes",
                },
                "end": {"class": "compare", "from": ["output"], "value": 0},
            },
        },
        "decision": {
            "class": "decide",
            "from": ["output"],
            "loss": "edit_distance",
            "loss_opts": {},
            "target": "classes",
        },
    }

    n_batch, n_in, n_time = 3, 19, 9
    n_out = n_in

    config = Config()
    config.update(
        {
            "extern_data": {"data": {"dim": n_out, "sparse": True}, "classes": {"dim": n_out, "sparse": True}},
            "search_output_layer": "decision",
            "debug_print_layer_output_shape": True,
            "debug_runtime_sanity_checks": True,
            "debug_print_layer_output_template": True,
        }
    )

    # Try different permutations of the cum_att_energy inputs, previously these behaved differently
    for source_layers in [["prev:att_energy", "att_energy"], ["att_energy", "prev:att_energy"]]:
        with make_scope() as session:
            network = TFNetwork(config=config, train_flag=False, search_flag=True)
            pprint(network.extern_data.data)
            net_dict["output"]["unit"]["cum_att_energy"]["from"] = source_layers
            network.construct_from_dict(net_dict)

            fetches = network.get_fetches_dict()
            data_input = network.extern_data.data["data"]
            assert data_input.batch_shape == (None, None)
            output_out = network.get_layer("decision").output
            assert (
                output_out.is_batch_major
                and output_out.sparse
                and output_out.dim == n_out
                and output_out.shape == (None,)
            )

            input_seq_lens = numpy.array([n_time, n_time - 5, n_time - 4], dtype="int32")
            assert input_seq_lens.shape == (n_batch,) and all(input_seq_lens > 0)
            input_seqs = rnd.randint(
                1,
                n_out,
                size=(
                    n_batch,
                    n_time,
                ),
                dtype="int32",
            )
            print("input:")
            print(input_seqs)
            print("lens:", input_seq_lens)

            session.run(
                tf_compat.v1.variables_initializer(tf_compat.v1.global_variables() + [network.global_train_step_var])
            )
            fetches = (fetches, output_out.placeholder, output_out.get_sequence_lengths())
            feed_dict = {
                network.extern_data.get_batch_info().dim: len(input_seq_lens),
                data_input.placeholder: input_seqs,
                data_input.size_placeholder[0]: input_seq_lens,
            }
            try:
                info, out_seqs, out_seq_lens = session.run(fetches, feed_dict=feed_dict)
            except Exception as exc:
                print("EXCEPTION:", type(exc), exc)
                help_on_tf_exception(session=session, exception=exc, fetches=fetches, feed_dict=feed_dict)
                raise
            print(info)
            print("output:")
            print(out_seqs)  # random...
            print("lens:", out_seq_lens)
            assert isinstance(out_seqs, numpy.ndarray) and isinstance(out_seq_lens, numpy.ndarray)
            assert len(out_seqs.shape) == 2 and out_seqs.shape[0] == n_batch
            assert out_seq_lens.shape == (n_batch,)


def test_PositionalEncodingLayer_offset_no_rec():
    # Test `offset` option when `PositionalEncodingLayer` is out of loop.
    rnd = numpy.random.RandomState(42)
    n_batch, n_in, n_time, size = 3, 5, 7, 8
    n_out = n_in

    net_dict = {
        "input": {"class": "linear", "activation": None, "n_out": size, "from": "data:data"},
        "offset": {"class": "constant", "from": [], "value": 1},
        "raw_pos_enc": {"class": "positional_encoding", "add_to_input": False, "from": ["input"], "n_out": size},
        "const_pos_enc": {
            "class": "positional_encoding",
            "add_to_input": False,
            "constant": 42,
            "from": ["input"],
            "n_out": size,
        },
        "pos_enc": {"class": "positional_encoding", "add_to_input": True, "from": ["input"], "offset": "offset"},
        "output": {
            "class": "softmax",
            "target": "data",
            "from": ["pos_enc", "raw_pos_enc", "const_pos_enc"],
            "loss": "ce",
        },
    }

    config = Config()
    config.update({"extern_data": {"data": {"dim": n_out, "sparse": True}}, "debug_print_layer_output_template": True})

    with make_scope() as session:
        network = TFNetwork(config=config, train_flag=True)
        pprint(network.extern_data.data)
        network.construct_from_dict(net_dict)

        fetches = network.get_fetches_dict()
        data_input = network.extern_data.data["data"]
        assert data_input.batch_shape == (None, None)

        train_out = network.get_layer("output").output
        session.run(
            tf_compat.v1.variables_initializer(tf_compat.v1.global_variables() + [network.global_train_step_var])
        )
        rand_data = rnd.randint(
            0,
            n_out,
            size=(
                n_batch,
                n_time,
            ),
            dtype="int32",
        )
        outputs = [train_out.placeholder]
        info, out = session.run(
            (fetches, outputs),
            feed_dict={
                network.extern_data.get_batch_info().dim: n_batch,
                data_input.placeholder: rand_data,
                data_input.size_placeholder[0]: numpy.array([n_time] * n_batch, dtype="int32"),
            },
        )
        print(info)
        print(out)  # random...


def test_PositionalEncodingLayer_offset_in_rec():
    # Test `offset` option when `PositionalEncodingLayer` is in loop.
    rnd = numpy.random.RandomState(42)
    n_batch, n_in, n_time, size = 3, 5, 7, 8
    n_out = n_in

    net_dict = {
        "output": {
            "class": "rec",
            "from": "data",
            "target": "data",
            "unit": {
                "input": {"class": "linear", "activation": None, "from": "data:source", "n_out": size},
                "in_prefix": {  # Just a dummy switch which turns off once it sees an input with the ID 0.
                    "class": "eval",
                    "from": ["data:source", "prev:in_prefix"],
                    "initial_output": True,
                    "out_type": {"time_dim_axis": None, "dtype": "bool", "dim": 2},
                    "eval": "tf.logical_and(tf.not_equal(source(0, "
                    "auto_convert=False, as_data=True).copy_as_batch_major().placeholder, 0), source(1))",
                },
                # Increment by one as long as `in_prefix` is on.
                "counter": {
                    "class": "eval",
                    "from": ["in_prefix", "prev:counter"],
                    "initial_output": 0,
                    "out_type": {"time_dim_axis": None, "shape": (), "dtype": "int32", "dim": None, "sparse": False},
                    "eval": "where_bc(source(0, auto_convert=False, as_data=True).copy_as_batch_major().placeholder,"
                    "source(1, auto_convert=False, as_data=True).copy_as_batch_major().placeholder + "
                    "tf.ones(tf.shape(source(0, auto_convert=False, as_data=True).copy_as_batch_major().placeholder),"
                    "dtype=tf.int32),"
                    "source(1, auto_convert=False, as_data=True).copy_as_batch_major().placeholder)",
                },
                # 0 when `in_prefix` is on. Minus `counter` otherwise.
                "offset": {
                    "class": "eval",
                    "from": ["in_prefix", "counter"],
                    "out_type": {"time_dim_axis": None, "shape": (), "dtype": "int32", "sparse": False, "dim": None},
                    "eval": "where_bc(source(0, auto_convert=False, as_data=True).copy_as_batch_major().placeholder,"
                    "tf.zeros(tf.shape(source(1, auto_convert=False, as_data=True).copy_as_batch_major().placeholder),"
                    "dtype=tf.int32),"
                    "-source(1, auto_convert=False, as_data=True).copy_as_batch_major().placeholder)",
                },
                "pos_enc": {
                    "class": "positional_encoding",
                    "add_to_input": True,
                    "from": ["input"],
                    "offset": "offset",
                },
                "rnn_h": {
                    "class": "linear",
                    "activation": "tanh",
                    "from": ["prev:rnn_h", "pos_enc"],
                    "n_out": size,
                    "initial_output": 0,
                },
                "output": {"class": "softmax", "from": ["rnn_h"], "loss": "ce", "target": "data"},
            },
        }
    }

    config = Config()
    config.update({"extern_data": {"data": {"dim": n_out, "sparse": True}}, "debug_print_layer_output_template": True})

    with make_scope() as session:
        network = TFNetwork(config=config, train_flag=True)
        pprint(network.extern_data.data)
        network.construct_from_dict(net_dict)

        fetches = network.get_fetches_dict()
        data_input = network.extern_data.data["data"]
        assert data_input.batch_shape == (None, None)

        train_out = network.get_layer("output").output
        session.run(
            tf_compat.v1.variables_initializer(tf_compat.v1.global_variables() + [network.global_train_step_var])
        )
        rand_data = rnd.randint(
            0,
            n_out,
            size=(
                n_batch,
                n_time,
            ),
            dtype="int32",
        )
        outputs = [train_out.placeholder]
        info, out = session.run(
            (fetches, outputs),
            feed_dict={
                network.extern_data.get_batch_info().dim: n_batch,
                data_input.placeholder: rand_data,
                data_input.size_placeholder[0]: numpy.array([n_time] * n_batch, dtype="int32"),
            },
        )
        print(info)
        print(out)  # random...


def test_RelativePositionalEncodingLayer():
    # Test `RelativePositionalEncodingLayer` with an example similar to its example usage.
    rnd = numpy.random.RandomState(42)
    n_batch, n_out, n_time = 3, 5, 7
    EncKeyTotalDim = 9
    EncValueTotalDim = 18
    AttNumHeads = 3
    net_dict = {
        "rel_pos": {"class": "relative_positional_encoding", "from": "data", "n_out": EncKeyTotalDim // AttNumHeads},
        "output": {
            "class": "self_attention",
            "num_heads": AttNumHeads,
            "total_key_dim": EncKeyTotalDim,
            "n_out": EncValueTotalDim,
            "from": "data",
            "attention_left_only": False,
            "key_shift": "rel_pos",
        },
    }
    config = Config()
    config.update({"extern_data": {"data": {"dim": n_out, "sparse": False}}, "debug_print_layer_output_template": True})
    with make_scope() as session:
        network = TFNetwork(config=config, train_flag=True)
        pprint(network.extern_data.data)
        network.construct_from_dict(net_dict)
        fetches = network.get_fetches_dict()
        data_input = network.extern_data.data["data"]
        assert data_input.batch_shape == (None, None, n_out)
        train_out = network.get_layer("output").output
        session.run(
            tf_compat.v1.variables_initializer(tf_compat.v1.global_variables() + [network.global_train_step_var])
        )
        rand_data = rnd.rand(n_batch, n_time, n_out)
        outputs = [train_out.placeholder]
        info, out = session.run(
            (fetches, outputs),
            feed_dict={
                network.extern_data.get_batch_info().dim: n_batch,
                data_input.placeholder: rand_data,
                data_input.size_placeholder[0]: numpy.array([n_time] * n_batch, dtype="float32"),
            },
        )
        print(info)
        print(out)  # random...


def _build_self_attention_layer(
    d, input, output, inside_rec_layer, query_axis=None, num_heads=3, key_dim=7, value_dim=11, dropout=0.0
):
    """
    Essentially this does
      d[output + '_att'] = {"class": "self_attention", "num_heads": num_heads,
        "total_key_dim": num_heads * key_dim,
        "n_out": num_heads * value_dim, "from": [input],
        "attention_left_only": inside_rec_layer,
        "attention_dropout": dropout, "forward_weights_init": self.ff_init}
    But using multiple layers.
    """
    # Create (non-accumulated) query, key and value
    d[output + "_qkv0"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [input],
        "n_out": num_heads * (2 * key_dim + value_dim),
    }  # [B,T?,F|n*(2d_k+d_v)]
    d[output + "_qkv"] = {
        "class": "split_dims",
        "axis": "F",
        "dims": (num_heads, 2 * key_dim + value_dim),
        "from": [output + "_qkv0"],
    }  # [B,T?,n,F|2d_k+d_v]
    d[output + "_qkv_split"] = {
        "class": "split",
        "axis": "F",
        "size_splits": (key_dim, key_dim, value_dim),
        "from": [output + "_qkv"],
    }
    d[output + "_query"] = {"class": "copy", "from": [output + "_qkv_split/0"]}  # [B,T?,n,F|d_k]
    d[output + "_key"] = {"class": "copy", "from": [output + "_qkv_split/1"]}  # [B,T?,n,F|d_k]
    d[output + "_value"] = {"class": "copy", "from": [output + "_qkv_split/2"]}  # [B,T?,n,F|d_v]

    # Accumulate keys/values or rename the axis
    key_dim_tag = Dim(kind=Dim.Types.Time, description="self-att-keys", dimension=None)
    if inside_rec_layer:
        d[output + "_key_accum"] = {
            "class": "cum_concat",
            "from": [output + "_key"],
            "out_spatial_dim": key_dim_tag,
        }  # [B,T|rec-history,n,F|d_k]
        d[output + "_value_accum"] = {
            "class": "cum_concat",
            "from": [output + "_value"],
            "out_spatial_dim": key_dim_tag,
        }  # [B,T|rec-history,n,F|d_v]
    else:
        assert query_axis
        d[output + "_key_accum"] = {
            "class": "reinterpret_data",
            "set_dim_tags": {query_axis: key_dim_tag},
            "from": [output + "_key"],
        }  # [B,T|keys,n,F|d_k]
        d[output + "_value_accum"] = {
            "class": "reinterpret_data",
            "set_dim_tags": {query_axis: key_dim_tag},
            "from": [output + "_value"],
        }  # [B,T|keys,n,F|d_v]

    # Calculate the energies
    d[output + "_energy"] = {
        "class": "dot",
        "from": [output + "_query", output + "_key_accum"],
        "reduce": "dim:%i" % key_dim,
    }  # [B,n,T?,T|rec-history]

    d[output + "_weights"] = {
        "class": "softmax_over_spatial",
        "from": [output + "_energy"],
        "axis": key_dim_tag,
        "energy_factor": key_dim**-0.5,
    }  # [B,n,T?,T|rec-history]
    d[output + "_weights_drop"] = {
        "class": "dropout",
        "dropout_noise_shape": {"*": None},
        "from": [output + "_weights"],
        "dropout": dropout,
    }  # [B,n,T?,T|rec-history]

    d[output + "_output"] = {
        "class": "dot",
        "from": [output + "_weights_drop", output + "_value_accum"],
        "reduce": key_dim_tag,
    }  # [B,n,T?,F|d_v]
    d[output + "_att"] = {
        "class": "merge_dims",
        "axes": ["dim:%i" % num_heads, "dim:%i" % value_dim],
        "from": output + "_output",
    }  # [B,T?,F|n*d_v]


def test_CumConcatLayer_self_attention_equal_to_SelfAttentionLayer():
    from returnn.tf.util.data import batch_dim

    n_time = 13
    num_heads, key_dim, value_dim = 2, 3, 3
    for inside_rec_layer in [False, True]:
        with make_scope() as session:
            print("Testing inside_rec_layer=%s" % inside_rec_layer)

            # build net dict
            if inside_rec_layer:
                net_dict = {
                    "output": {
                        "class": "rec",
                        "target": "classes",
                        "from": [],
                        "unit": {
                            "const0": {"class": "constant", "value": 0.0, "shape": [batch_dim, 1]},
                            "lin": {
                                "class": "linear",
                                "from": ["prev:lin", "const0"],
                                "bias_init": "glorot_normal",
                                "n_out": 5,
                            },
                            "single_layer_att": {
                                "class": "self_attention",
                                "from": "lin",
                                "num_heads": num_heads,
                                "total_key_dim": num_heads * key_dim,
                                "n_out": num_heads * value_dim,
                                "attention_left_only": inside_rec_layer,
                                "is_output_layer": True,
                            },  # [B,T,F]
                            "multi_layer_att": None,  # [B,T,F], added below.
                            "output": {"class": "compare", "from": ["single_layer_att", "multi_layer_att"]},
                        },
                    }
                }
                _build_self_attention_layer(
                    net_dict["output"]["unit"],
                    "lin",
                    "multi_layer",
                    inside_rec_layer=True,
                    query_axis="stag:extern_data:classes",
                    num_heads=num_heads,
                    key_dim=key_dim,
                    value_dim=value_dim,
                )
                net_dict["output"]["unit"]["multi_layer_att"]["is_output_layer"] = True
                net_dict["output"]["unit"]["multi_layer_qkv0"][
                    "is_output_layer"
                ] = True  # we need to set the matrix here
            else:
                net_dict = {
                    "single_layer_att": {
                        "class": "self_attention",
                        "from": "data",
                        "num_heads": num_heads,
                        "total_key_dim": num_heads * key_dim,
                        "n_out": num_heads * value_dim,
                        "attention_left_only": inside_rec_layer,
                        "is_output_layer": True,
                    },  # [B,T,F]
                    "multi_layer_att": None,  # [B,T,F], added below.
                    "output": {"class": "compare", "from": ["single_layer_att", "multi_layer_att"]},
                }
                _build_self_attention_layer(
                    net_dict,
                    "data",
                    "multi_layer",
                    inside_rec_layer=False,
                    query_axis="stag:extern_data:data",
                    num_heads=num_heads,
                    key_dim=key_dim,
                    value_dim=value_dim,
                )
                net_dict["multi_layer_att"]["is_output_layer"] = True

            config = Config({"debug_print_layer_output_template": True, "optimize_move_layers_out": True})
            config.update(dict(num_inputs=num_heads * key_dim, num_outputs=num_heads * value_dim))
            network = TFNetwork(config=config, train_flag=True)
            from pprint import pprint

            pprint(net_dict)
            network.construct_from_dict(net_dict)
            session.run(tf.compat.v1.global_variables_initializer())

            if inside_rec_layer:
                single_layer = network.get_layer("output/single_layer_att")
                multi_layer = network.get_layer("output/multi_layer_att")
                single_weights = single_layer.params["QKV"]
                multi_weights = network.get_layer("output/multi_layer_qkv0").params["W"]
            else:
                single_layer = network.get_layer("single_layer_att")
                multi_layer = network.get_layer("multi_layer_att")
                single_weights = single_layer.params["QKV"]
                multi_weights = network.get_layer("multi_layer_qkv0").params["W"]

            assert_equal(single_layer.output.batch_shape, (None, None, num_heads * value_dim))
            assert_equal(multi_layer.output.batch_shape, (None, None, num_heads * value_dim))

            # set weights equal.
            assert_equal(single_weights.shape, multi_weights.shape)
            weights = numpy.random.rand(*single_weights.shape)
            session.run(tf.compat.v1.assign(single_weights, weights))
            session.run(tf.compat.v1.assign(multi_weights, weights))

            # fetch/compare outputs
            from test_TFNetworkLayer import make_feed_dict

            feed_dict = make_feed_dict(network.extern_data, same_time=True, n_time=n_time)
            single, multi = session.run(
                [single_layer.output.placeholder, multi_layer.output.placeholder], feed_dict=feed_dict
            )
            print("single layer output:")
            pprint(single)
            print("multi layer output:")
            pprint(multi)
            numpy.testing.assert_almost_equal(single, multi, decimal=5)
            print("They are equal!")


def test_CumConcatLayer_search():
    rnd = numpy.random.RandomState(42)
    beam_size = 5
    dim = 7

    net_dict = {
        "source_embed": {"class": "linear", "activation": None, "n_out": dim, "from": "data:data"},
        "source_pool": {"class": "reduce", "mode": "mean", "axis": "T", "from": "source_embed"},
        "output": {
            "class": "rec",
            "from": [],
            "max_seq_len": "max_len_from('base:source_embed') * 3",
            "target": "classes",
            "unit": {
                "target_embed": {"class": "linear", "activation": None, "from": "prev:output", "n_out": dim},
                "output_prob": {
                    "class": "softmax",
                    "from": ["self_att_att", "base:source_pool"],
                    "loss": "ce",
                    "target": "classes",
                },
                "output": {
                    "beam_size": beam_size,
                    "class": "choice",
                    "from": "output_prob",
                    "initial_output": 0,
                    "target": "classes",
                },
                "end": {"class": "compare", "from": "output", "value": 0},
            },
        },
        "decision": {"class": "decide", "from": "output", "loss": "edit_distance", "target": "classes"},
    }
    _build_self_attention_layer(net_dict["output"]["unit"], "target_embed", "self_att", inside_rec_layer=True)

    n_batch, n_in, n_time = 3, 19, 9
    n_out = n_in

    config = Config()
    config.update(
        {
            "extern_data": {"data": {"dim": n_out, "sparse": True}, "classes": {"dim": n_out, "sparse": True}},
            "search_output_layer": "decision",
            "debug_print_layer_output_shape": True,
            "debug_runtime_sanity_checks": True,
            "debug_print_layer_output_template": True,
        }
    )

    with make_scope() as session:
        network = TFNetwork(config=config, train_flag=False, search_flag=True)
        pprint(network.extern_data.data)
        network.construct_from_dict(net_dict)

        fetches = network.get_fetches_dict()
        data_input = network.extern_data.data["data"]
        assert data_input.batch_shape == (None, None)
        output_out = network.get_layer("decision").output
        assert (
            output_out.is_batch_major and output_out.sparse and output_out.dim == n_out and output_out.shape == (None,)
        )

        input_seq_lens = numpy.array([n_time, n_time - 5, n_time - 4], dtype="int32")
        assert input_seq_lens.shape == (n_batch,) and all(input_seq_lens > 0)
        input_seqs = rnd.randint(
            1,
            n_out,
            size=(
                n_batch,
                n_time,
            ),
            dtype="int32",
        )
        print("input:")
        print(input_seqs)
        print("lens:", input_seq_lens)

        session.run(
            tf_compat.v1.variables_initializer(tf_compat.v1.global_variables() + [network.global_train_step_var])
        )
        fetches = (fetches, output_out.placeholder, output_out.get_sequence_lengths())
        feed_dict = {
            network.extern_data.get_batch_info().dim: len(input_seq_lens),
            data_input.placeholder: input_seqs,
            data_input.size_placeholder[0]: input_seq_lens,
        }
        try:
            info, out_seqs, out_seq_lens = session.run(fetches, feed_dict=feed_dict)
        except Exception as exc:
            print("EXCEPTION:", type(exc), exc)
            help_on_tf_exception(session=session, exception=exc, fetches=fetches, feed_dict=feed_dict)
            raise
        print(info)
        print("output:")
        print(out_seqs)  # random...
        print("lens:", out_seq_lens)
        assert isinstance(out_seqs, numpy.ndarray) and isinstance(out_seq_lens, numpy.ndarray)
        assert len(out_seqs.shape) == 2 and out_seqs.shape[0] == n_batch
        assert out_seq_lens.shape == (n_batch,)


if __name__ == "__main__":
    try:
        better_exchook.install()
        if len(sys.argv) <= 1:
            for k, v in sorted(globals().items()):
                if k.startswith("test_"):
                    print("-" * 40)
                    print("Executing: %s" % k)
                    try:
                        v()
                    except unittest.SkipTest as exc:
                        print("SkipTest:", exc)
                    print("-" * 40)
            print("Finished all tests.")
        else:
            assert len(sys.argv) >= 2
            for arg in sys.argv[1:]:
                print("Executing: %s" % arg)
                if arg in globals():
                    globals()[arg]()  # assume function and execute
                else:
                    eval(arg)  # assume Python code and execute
    finally:
        import threading

        # if len(list(threading.enumerate())) > 1:
        #  print("Warning, more than one thread at exit:")
        #  better_exchook.dump_all_thread_tracebacks()
