
# start: nosetests $this_file --nologcapture

from __future__ import print_function

import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from nose.tools import assert_equal, assert_not_equal, assert_is_instance
from numpy.testing.utils import assert_almost_equal, assert_allclose
import unittest
import numpy.testing
from pprint import pprint
import contextlib
import better_exchook
better_exchook.replace_traceback_format_tb()

from Log import log
from Config import Config
from TFNetwork import *
from TFNetworkRecLayer import *
from TFUtil import is_gpu_available
import TFUtil
TFUtil.debug_register_better_repr()

import Debug
Debug.install_lib_sig_segfault()

try:
  import faulthandler
  # Enable after libSigSegfault, so that we have both,
  # because faulthandler will also call the original sig handler.
  faulthandler.enable()
except ImportError:
  print("no faulthandler")

log.initialize(verbosity=[5])

print("TensorFlow:", tf.__version__)


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
  :rtype: tf.Session
  """
  with tf.Graph().as_default() as graph:
    with tf.Session(graph=graph) as session:
      yield session


def _check_train_simple_network(network, num_steps=10):
  num_inputs = 4
  num_outputs = 3

  config = Config()
  config.update({
    "num_inputs": num_inputs,
    "num_outputs": {"data": [num_inputs, 2], "classes": [num_outputs, 2]},  # dense output
    "network": network,
    "adam": True,
    "learning_rate": 0.1,
    "debug_add_check_numerics_ops": True,
  })

  random = numpy.random.RandomState(seed=1)
  limit = 1.0

  def make_feed_dict(step, seq_len=10):
    d = {
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
    tf.set_random_seed(42)
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])
    network.print_network_info()
    network.initialize_params(session=session)

    from TFUpdater import Updater
    updater = Updater(config=config, network=network)
    updater.set_learning_rate(config.float("learning_rate", 1.0), session=session)
    updater.set_trainable_vars(network.get_trainable_params())
    updater.init_optimizer_vars(session=session)

    loss = None
    for step in range(num_steps):
      loss, _, _ = session.run(
        [network.get_total_loss(), updater.get_optim_op(), network.get_post_control_dependencies()],
        feed_dict=make_feed_dict(step=step))
      print("step %i, loss: %f" % (step, loss))
  return loss


def test_rec_nativelstm():
  _check_train_simple_network({"output": {"class": "rec", "unit": "nativelstm", "loss": "mse"}})


def test_rec_nativelstm2():
  _check_train_simple_network({"output": {"class": "rec", "unit": "nativelstm2", "loss": "mse"}})


def test_rec_rhn():
  _check_train_simple_network({
    "output": {
      "class": "rec", "unit": "rhn", "unit_opts": {"dropout": 0.1},
      "loss": "mse"}})


def test_rec_rhn_nan():
  _check_train_simple_network({
    "output": {
      "class": "rec", "unit": "rhn", "unit_opts": {"dropout": 0.9, "dropout_seed": 1},
      "loss": "mse"}})


def test_rhn_nan():
  """
  Behaves just like :func:`test_rec_rhn_nan`.
  """
  random = numpy.random.RandomState(seed=1)
  num_inputs = 4
  num_outputs = 3
  seq_len = 10
  limit = 1.0
  loop_variants = ["RecLayer", 'dynamic_rnn', 'while_loop', 'unroll', "unroll_simple"]
  loop_variant = "unroll_simple"

  with make_scope() as session:
    print("create graph")
    tf.set_random_seed(42)
    src_placeholder = tf.placeholder(tf.float32, (None, seq_len, num_inputs), name="src_placeholder")
    tgt_placeholder = tf.placeholder(tf.float32, (None, seq_len, num_outputs), name="tgt_placeholder")
    batch_size = tf.shape(src_placeholder)[0]

    def make_feed_dict():
      return {
        src_placeholder: random.uniform(-limit, limit, (1, seq_len, num_inputs)),
        tgt_placeholder: random.uniform(-limit, limit, (1, seq_len, num_outputs)),
      }

    from TFUtil import xavier_initializer
    default_var_initializer = xavier_initializer(seed=13)
    with tf.variable_scope(tf.get_variable_scope(), initializer=default_var_initializer) as scope:
      assert loop_variant in loop_variants
      if loop_variant in ["RecLayer", "dynamic_rnn"]:
        if loop_variant == "RecLayer":
          # Here I get nan.
          net = TFNetwork(config=Config(), extern_data=ExternData(), train_flag=True)
          with net.register_network_scope():
            from TFNetworkLayer import InternalLayer
            src_layer = InternalLayer(name='src', network=net, output=Data(
              'src', shape=(None, num_inputs), placeholder=src_placeholder, size_placeholder={0: [seq_len]}))
            with tf.name_scope("output"):
              rec_layer = RecLayer(
                name='output', network=net, output=Data("out", shape=(None, num_outputs)), sources=[src_layer],
                unit='rhn', unit_opts={"dropout": 0.9, "dropout_seed": 1, "batch_size": batch_size})
          y = rec_layer.output.placeholder
          y = tf.transpose(y, [1, 0, 2])
        elif loop_variant == "dynamic_rnn":
          rhn = RHNCell(num_units=num_outputs, is_training=True, dropout=0.9, dropout_seed=1, batch_size=batch_size)
          # Will get y in (time,batch,ydim).
          from tensorflow.python.ops import rnn
          x = tf.transpose(src_placeholder, [1, 0, 2])
          x = rhn.get_input_transformed(x)
          y, final_state = rnn.dynamic_rnn(
            cell=rhn, inputs=x, time_major=True,
            sequence_length=[seq_len], dtype=tf.float32)
          y = tf.transpose(y, [1, 0, 2])
        loss = tf.reduce_sum(tf.reduce_mean(tf.squared_difference(tgt_placeholder, y), axis=-1))
      else:
        rhn = RHNCell(num_units=num_outputs, is_training=True, dropout=0.9, dropout_seed=1, batch_size=batch_size)
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
          frame_loss = tf.reduce_mean(tf.squared_difference(target_ta.read(i), output), axis=1)
          assert frame_loss.get_shape().ndims == 1  # (batch,)
          # frame_loss = tf.Print(frame_loss, ["frame", i, "loss", tf.reduce_sum(frame_loss)])
          loss_ta = loss_ta.write(i, frame_loss)
          return i + 1, state, loss_ta

        if loop_variant == "while_loop":
          i, state, loss_ta = tf.while_loop(
            lambda i, *args: tf.less(i, seq_len),
            loop_iter,
            (0, state, loss_ta))
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
            frame_loss = tf.reduce_mean(tf.squared_difference(tgt_placeholder[:, i], output), axis=1)
            #frame_loss = tf.Print(frame_loss, ['frame', i, 'loss', frame_loss, 'SE of', tgt_placeholder[:, i], output])
            assert frame_loss.get_shape().ndims == 1  # (batch,)
            loss += tf.reduce_sum(frame_loss)
        else:
          assert False, "unexpected loop variant %r" % loop_variant
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1, epsilon=1e-16, use_locking=False)
    minimize_op = optimizer.minimize(loss)
    from TFUtil import add_check_numerics_ops
    #check_op = add_check_numerics_ops()
    check_op = tf.no_op()

    print('variables:')
    train_vars = (
      tf.trainable_variables() +
      tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
    print(train_vars)
    var_norms = [tf.nn.l2_loss(v) for v in train_vars]
    print('init vars')
    session.run(tf.global_variables_initializer())
    print('graph size:', session.graph_def.ByteSize())
    print('train')
    for s in range(10):
      loss_val, _, _ = session.run([loss, minimize_op, check_op], feed_dict=make_feed_dict())
      print("step %i, loss: %f" % (s, loss_val))
      #var_norm_vals = session.run(var_norms)
      #print('var norms:')
      #for (v, x) in zip(train_vars, var_norm_vals):
      #  print(' ', v, ':', x)


def test_state_keep_over_epoch():
  random = numpy.random.RandomState(seed=1)
  num_inputs = 4
  num_outputs = 3
  batch_size = 5
  seq_len = 10
  limit = 1.0
  src_seq = random.uniform(-limit, limit, (batch_size, seq_len, num_inputs))
  net_dict = {"output": {
    "class": "rec", "unit": "rhn", "initial_state": 'keep_over_epoch', 'n_out': num_outputs}}

  with make_scope() as session:
    print("create graph")
    tf.set_random_seed(42)
    net = TFNetwork(extern_data=ExternData({'data': {"shape": (None, num_inputs)}}))
    net.construct_from_dict(net_dict)
    net.initialize_params(session)
    print("vars:")
    print(tf.global_variables())
    src = net.extern_data.data["data"].placeholder
    src_seq_len = net.extern_data.data["data"].size_placeholder[0]
    out = net.get_default_output_layer().output.get_placeholder_as_batch_major()
    out_val = numpy.zeros((batch_size, 0, num_outputs))
    print('run on parts')
    part_seq_len = 2
    for step, t in enumerate(range(0, seq_len, part_seq_len)):
      out_val_part, _ = session.run([out, net.get_post_control_dependencies()], feed_dict={
        net.epoch_step: step, src_seq_len: [part_seq_len] * batch_size, src: src_seq[:, t:t + part_seq_len]})
      assert out_val_part.shape == (batch_size, part_seq_len, num_outputs)
      out_val = numpy.concatenate([out_val, out_val_part], axis=1)
    assert out_val.shape == (batch_size, seq_len, num_outputs)
    print('run full')
    out_val_full, _ = session.run(
      [out, net.get_post_control_dependencies()],
      feed_dict={net.epoch_step: 0, src_seq_len: [seq_len] * batch_size, src: src_seq})
    assert out_val_full.shape == out_val.shape
    assert_almost_equal(out_val, out_val_full)
  print('ok!')


def test_lstm_initial_state_zero():
  _check_train_simple_network({
    "output": {"class": "rec", "unit": "lstm", "loss": "mse", "initial_state": "zeros"}})


def test_lstm_initial_state_var():
  _check_train_simple_network({
    "output": {"class": "rec", "unit": "lstm", "loss": "mse", "initial_state": "var"}})


def test_nativelstm2_initial_state_var():
  _check_train_simple_network({
    "output": {"class": "rec", "unit": "nativelstm2", "loss": "mse", "initial_state": "var"}})


def test_nativelstm2_initial_state_keep_epoch():
  _check_train_simple_network({
    "output": {"class": "rec", "unit": "nativelstm2", "loss": "mse", "initial_state": "keep_over_epoch"}})


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
    with tf.variable_scope("linear", reuse=tf.AUTO_REUSE):
      weights = tf.get_variable("W", shape=(input_dim, output_dim))
      bias = tf.get_variable("b", shape=(output_dim,))
    assert x.get_shape().ndims == 2  # (batch,input_dim)
    return tf.matmul(x, weights) + bias

  with make_scope() as session:
    print("create graph")
    src_placeholder = tf.placeholder(tf.float32, (None, seq_len, num_inputs), name="src_placeholder")
    tgt_placeholder = tf.placeholder(tf.float32, (None, seq_len, num_outputs), name="tgt_placeholder")
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
      with tf.variable_scope('h'):
        h = tf.tanh(linear(x, num_outputs))
      with tf.variable_scope('t'):
        t = tf.sigmoid(linear(x, num_outputs))
      state += t * (h - state)
      frame_loss = tf.reduce_mean(tf.squared_difference(tgt_placeholder[:, f], state), axis=1)
      assert frame_loss.get_shape().ndims == 1  # (batch,)
      loss_ta = loss_ta.write(f, frame_loss)
    loss = tf.reduce_sum(loss_ta.stack())
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1, epsilon=1e-16, use_locking=False)
    minimize_op = optimizer.minimize(loss)

    print('variables:')
    train_vars = (
      tf.trainable_variables() +
      tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
    print(train_vars)
    print('init vars')
    session.run(tf.global_variables_initializer())
    print('graph size:', session.graph_def.ByteSize())
    print('train')
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
    print('>>> run %i' % r)
    random = numpy.random.RandomState(seed=1)

    with make_scope() as session:
      tf.set_random_seed(42)
      print("create graph")
      src_placeholder = tf.placeholder(tf.float32, (None, seq_len, num_inputs), name="src_placeholder")
      tgt_placeholder = tf.placeholder(tf.float32, (None, seq_len, num_outputs), name="tgt_placeholder")
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
        random_tensor += tf.random_uniform((batch_size, cell.state_size), seed=1, dtype=state.dtype)
        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        binary_tensor = tf.floor(random_tensor)
        noise_h = binary_tensor / keep_prob
        state *= noise_h

        output, state = cell(inputs=src_placeholder[:, i], state=state)
        frame_loss = tf.reduce_mean(tf.squared_difference(tgt_placeholder[:, i], output), axis=1)
        assert frame_loss.get_shape().ndims == 1  # (batch,)
        loss_ta = loss_ta.write(i, frame_loss)
      loss = tf.reduce_sum(loss_ta.stack())
      optimizer = tf.train.AdamOptimizer(learning_rate=0.1, epsilon=1e-16, use_locking=False)
      minimize_op = optimizer.minimize(loss)

      print('variables:')
      train_vars = (
        tf.trainable_variables() +
        tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
      print(train_vars)
      print('init vars')
      session.run(tf.global_variables_initializer())
      print('graph size:', session.graph_def.ByteSize())
      print('train')
      loss_val = None
      for s in range(10):
        loss_val, _ = session.run([loss, minimize_op], feed_dict=make_feed_dict())
        print("step %i, loss: %f" % (s, loss_val))
      assert loss_val is not None
      if r == 0:
        first_run_loss = loss_val
      else:
        assert numpy.isclose(first_run_loss, loss_val)


def test_rec_subnet_with_choice():
  with tf.Session():
    config = Config()
    config.update({
      "num_outputs": 3,
      "num_inputs": 4,
      "network": {
        "output": {"class": "rec", "target": "classes", "unit": {
          "prob": {"class": "softmax", "from": ["prev:output"], "loss": "ce", "target": "classes"},
          "output": {"class": "choice", "beam_size": 4, "from": ["prob"], "target": "classes", "initial_output": 0}
        }},
      }
    })
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])


@unittest.skipIf(not is_gpu_available(), "no gpu on this system")
def test_RecLayer_get_cudnn_params_size():
  from tensorflow.contrib.cudnn_rnn.ops.gen_cudnn_rnn_ops import cudnn_rnn_params_size

  def check(num_units, input_size,
            rnn_mode="lstm", num_layers=1, direction="unidirectional", input_mode="linear_input",
            T=tf.float32, S=tf.int32):
    common_kwargs = dict(
      rnn_mode=rnn_mode, num_units=num_units, input_size=input_size,
      num_layers=num_layers, direction=direction, input_mode=input_mode)
    cu_size = cudnn_rnn_params_size(T=T, S=S, **common_kwargs)[0]
    my_size = RecLayer._get_cudnn_param_size(**common_kwargs)
    assert_equal(cu_size.eval(), my_size)

  with tf.Session():
    check(rnn_mode="lstm", num_units=5, input_size=3)
    check(rnn_mode="lstm", num_units=5, input_size=5)
    check(rnn_mode="gru", num_units=7, input_size=5)
    check(rnn_mode="gru", num_units=7, input_size=7)
    check(rnn_mode="rnn_tanh", num_units=7, input_size=7)
    check(rnn_mode="lstm", num_units=5, input_size=3, direction="bidirectional")
    check(rnn_mode="lstm", num_units=5, input_size=3, direction="bidirectional", num_layers=2)
    check(rnn_mode="lstm", num_units=5, input_size=3, direction="bidirectional", num_layers=7)
    check(rnn_mode="lstm", num_units=5, input_size=3, direction="bidirectional", input_mode="auto_select")
    check(rnn_mode="lstm", num_units=5, input_size=3, direction="bidirectional", num_layers=7, input_mode="auto_select")
    check(rnn_mode="lstm", num_units=5, input_size=3, direction="unidirectional", num_layers=7, input_mode="auto_select")
    check(rnn_mode="lstm", num_units=5, input_size=3, direction="bidirectional", input_mode="skip_input")
    check(rnn_mode="lstm", num_units=5, input_size=3, direction="bidirectional", num_layers=7, input_mode="skip_input")


@unittest.skipIf(not is_gpu_available(), "no gpu on this system")
def test_cudnn_save_restore():
  from pprint import pprint
  import tempfile, shutil, os
  from tensorflow.python.training.saver import BaseSaverBuilder
  model_tmp_dir = tempfile.mkdtemp("tmp-checkpoint")
  model_filename = model_tmp_dir + "/model"
  try:
    num_inputs = 4
    input_data = numpy.array([[
      [1, -0.2, 0.3, -4],
      [2, -0.6, 0.7, -1.8],
      [1, 0.3, -0.1, -0.8],
      [0.1, -0.2, 0.2, .8]]],
      dtype="float32")
    seq_lens = numpy.array([4], dtype="int32")
    assert input_data.shape == (1, seq_lens[0], num_inputs)
    num_outputs = 3

    print("Storing network with cuDNN.")
    tf.reset_default_graph()
    with tf.Session() as session:
      config1 = Config()
      config1.update({
        "num_outputs": num_outputs,
        "num_inputs": num_inputs,
        "network": {
          "layer1": {"class": "rec", "n_out": 6, "unit": "CudnnLSTM"},
          "layer2": {"class": "rec", "n_out": 6, "unit": "CudnnLSTM", "from": ["layer1"]},
          "output": {"class": "linear", "activation": None, "n_out": num_outputs, "from": ["layer2"]}
        }
      })
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
          network1.extern_data.data["data"].placeholder: input_data,
          network1.extern_data.data["data"].size_placeholder[0]: seq_lens})
      assert_equal(output_data1.shape, (seq_lens[0], 1, num_outputs))  # (time, batch, dim)
      print("Saveable params:")
      pprint(network1.get_saveable_params_list())
      network1.save_params_to_file(filename=model_filename, session=session)
    print()

    # First test if we can load the same network as-is. This will involve the RNNParamsSaveable.
    print("Testing restore of same network with cuDNN.")
    tf.reset_default_graph()
    with tf.Session() as session:
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
          #assert param1old.ndim == 1
          #for i in range(param1old.shape[0]):
          #  assert_almost_equal(param1old[i], param1new[i])
          #numpy.testing.assert_almost_equal(param1old, param1new)
      output_data1a = session.run(
        network1a.get_default_output_layer().output.placeholder,
        feed_dict={
          network1a.extern_data.data["data"].placeholder: input_data,
          network1a.extern_data.data["data"].size_placeholder[0]: seq_lens})
      numpy.testing.assert_almost_equal(output_data1, output_data1a)
    print()

    print("Testing restore of network with LSTMBlockCell.")
    tf.reset_default_graph()
    with tf.Session() as session:
      # Now, in CPU, we would automatically use LSTMBlockCell instead.
      # Check if the import of the model works correctly in load_params_from_file().
      config2 = Config()
      config2.update({
        "num_outputs": num_outputs,
        "num_inputs": num_inputs,
        "network": {
          "layer1": {"class": "rec", "n_out": 6, "unit": "LSTMBlockFused"},
          "layer2": {"class": "rec", "n_out": 6, "unit": "LSTMBlockFused", "from": ["layer1"]},
          "output": {"class": "linear", "activation": None, "n_out": num_outputs, "from": ["layer2"]}
        }
      })
      network2 = TFNetwork(config=config2, train_flag=True)
      network2.construct_from_dict(config2.typed_dict["network"])
      print("Saveable params:")
      pprint(network2.get_saveable_params_list())
      network2.load_params_from_file(filename=model_filename, session=session)
      output_data2 = session.run(
        network2.get_default_output_layer().output.placeholder,
        feed_dict={
          network2.extern_data.data["data"].placeholder: input_data,
          network2.extern_data.data["data"].size_placeholder[0]: seq_lens})
      # Not sure if sth is incorrect... Only decimal=2 works.
      numpy.testing.assert_almost_equal(output_data1, output_data2, decimal=2)

  finally:
    shutil.rmtree(model_tmp_dir)


@unittest.skip("broken in TF. waiting to be fixed. https://github.com/tensorflow/tensorflow/issues/9370")
@unittest.skipIf(not is_gpu_available(), "no gpu on this system")
def test_cudnn_rnn_params_to_canonical():
  # https://github.com/tensorflow/tensorflow/issues/9370
  from tensorflow.contrib.cudnn_rnn import CudnnLSTM
  with tf.Session() as session:
    def check(**kwargs):
      print("kwargs:", kwargs)
      model = CudnnLSTM(**kwargs)
      params = tf.Variable(tf.random_uniform([model.params_size()], seed=1), validate_shape=False)
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

    check(num_layers=1, num_units=5, input_size=3, direction='unidirectional')
    check(num_layers=1, num_units=5, input_size=3, direction='bidirectional')  # fails in TF 1.2.0
    check(num_layers=2, num_units=5, input_size=3, direction='bidirectional')


def test_RecLayer_NativeLstm_Nan():
  print("test_RecLayer_NativeLstm_Nan()")
  print("GPU available:", is_gpu_available())
  numpy.set_printoptions(precision=15)
  num_inputs = 4
  num_outputs = 3

  config = Config()
  config.update({
    "num_inputs": num_inputs,
    "num_outputs": {"data": [num_inputs, 2], "classes": [num_outputs, 2]},  # dense output
    "network": {
      "output": {"class": "rec", "unit": "NativeLSTM", "loss": "mse"}
    },
    "adam": True,
    "debug_grad_summaries": True,
    "debug_save_updater_vars": True,
    "debug_add_check_numerics_ops": True,
  })

  print("Reset default graph...")
  tf.reset_default_graph()
  print("Create network...")
  network = TFNetwork(config=config, train_flag=True)
  network.construct_from_dict(config.typed_dict["network"])

  # Depending on the seed, I get nan earlier, later, or not at all.
  # limit=5.0: seed=3 -> nan in step 4094. seed=1 -> nan in step 2463.
  random = numpy.random.RandomState(seed=1)
  limit = 10.0  # The higher, the more likely you get nan.

  def make_feed_dict(seq_len=10):
    return {
      network.extern_data.data["data"].placeholder: random.uniform(-limit, limit, (1, seq_len, num_inputs)),
      network.extern_data.data["data"].size_placeholder[0]: numpy.array([seq_len]),
      network.extern_data.data["classes"].placeholder: random.uniform(-limit, limit, (1, seq_len, num_outputs)),
      network.extern_data.data["classes"].size_placeholder[0]: numpy.array([seq_len]),
    }

  print("Creating session...")
  with tf.Session() as session:
    print("Init params...")
    network.initialize_params(session=session)
    print("Test run...")
    output_data1 = session.run(network.get_default_output_layer().output.placeholder, feed_dict=make_feed_dict(5))
    assert_equal(output_data1.shape, (5, 1, num_outputs))  # (time, batch, dim)

    layer = network.layers["output"]
    loss_t = network.get_total_loss() * layer.loss.get_normalization_factor()
    weights_t = layer.params["W"]
    weights_grad_t, = tf.gradients(network.get_objective(), weights_t)

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
    demo_grad2_input_placeholders = [tf.placeholder(v.dtype) for v in lstm_grad_ins_t]
    demo_grad2_t = lstm_grad_func(*demo_grad2_input_placeholders)[1]

    print("Create updater...")
    from TFUpdater import Updater
    updater = Updater(config=config, network=network)
    updater.set_trainable_vars(network.get_trainable_params())
    updater.set_learning_rate(0.1, session=session)
    updater.init_optimizer_vars(session=session)
    optim_op = updater.get_optim_op()
    assert isinstance(updater.optimizer.get_default_optimizer(), tf.train.AdamOptimizer)
    adam_weights_m_t = updater.optimizer.get_slot(var=weights_t, name="m")
    adam_weights_v_t = updater.optimizer.get_slot(var=weights_t, name="v")
    assert isinstance(adam_weights_m_t, tf.Variable)
    assert isinstance(adam_weights_v_t, tf.Variable)
    summaries_t = tf.summary.merge_all()

    # https://github.com/tensorflow/tensorflow/blob/03beb65cecbc1e49ea477bca7f54543134b31d53/tensorflow/core/kernels/training_ops_gpu.cu.cc
    adam_update_t = adam_weights_m_t / (tf.sqrt(adam_weights_v_t) + 1e-8)

    import tempfile
    tmp_tf_logdir = tempfile.mkdtemp("tmp-tf-log")
    print("Write TF logs to:", tmp_tf_logdir)
    writer = tf.summary.FileWriter(tmp_tf_logdir)
    writer.add_graph(session.graph)

    print("Training...")
    recent_info = []  # type: list[dict[str]]
    for i in range(10000):
      feed_dict = make_feed_dict(5)
      weights_grad, lstm_grad_ins, lstm_grad_outs = session.run(
        [weights_grad_t, lstm_grad_ins_t, lstm_grad_outs_t], feed_dict=feed_dict)
      try:
        if not numpy.all(numpy.isfinite(weights_grad)):
          raise Exception("weights_grad has inf or nan.")
        loss, _opt, summaries, weights, adam_update = session.run(
          [loss_t, optim_op, summaries_t, weights_t, adam_update_t], feed_dict=feed_dict)
      except Exception as exc:
        print("Exception in step %i." % i)
        print(exc)
        print("Most recent summaries:")
        summary_proto = tf.Summary()
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
        print(session.run(
          demo_grad2_t, feed_dict={k: v for (k, v) in zip(demo_grad2_input_placeholders, lstm_grad_ins)}))
        print("Demo grad2 via eval:")
        print(session.run(
          demo_grad2_t, feed_dict={
            k: eval(repr(v), vars(numpy)) for (k, v) in zip(demo_grad2_input_placeholders, lstm_grad_ins)}))
        print("Demo grad2 via args:")
        print(session.run(
          demo_grad2_t, feed_dict={
            k: v for (k, v) in zip(demo_grad2_input_placeholders, _demo_lstm_grad_args())}))
        raise Exception("Exception in step %i." % i)
      writer.add_summary(summaries, global_step=i)
      if len(recent_info) > 1000:
        recent_info.pop(0)
      recent_info.append({
        "step": i, "loss": loss, "summaries": summaries, "weights": weights, "adam_update": adam_update})
      if not numpy.isfinite(loss) or i % 100 == 0:
        print("step %i, loss: %r" % (i, loss))
      assert numpy.isfinite(loss)

  print("Done.")
  import shutil
  shutil.rmtree(tmp_tf_logdir)


def find_op_by_type(session, type_name):
  """
  :param tf.Session session:
  :param str type_name:
  :rtype: tf.Operation|None
  """
  for op in session.graph.get_operations():
    assert isinstance(op, tf.Operation)
    if op.type == type_name:
      return op


def _lstm_grad_op(session, verbose=True):
  """
  :param tf.Session session:
  :return: grad function
  """
  lstm_grad_op = find_op_by_type(session=session, type_name="LstmGenericBase")
  assert lstm_grad_op is not None
  if verbose: print("op:", lstm_grad_op)

  from tensorflow.python.framework import ops
  grad_func = ops.get_gradient_function(lstm_grad_op)
  if verbose: print("grad_func:", grad_func)
  grad_op = grad_func.grad_op
  if verbose: print("grad_op:", grad_op, grad_op.__doc__)
  return grad_op


def _demo_lstm_grad_args(factor=1.0, ones_like=False):
  from numpy import array, float32
  n_time = 5
  n_batch = 1
  n_out = 3
  # <tf.Tensor 'output/rec/W_re/read:0' shape=(3, 12) dtype=float32>:
  W_re = \
    array([[-2.193344831466675, 1.360482335090637, 0.294201552867889,
            1.242056131362915, -0.18156972527504, -0.50642192363739,
            1.264044165611267, 0.108740165829659, 1.768813014030457,
            -3.442604303359985, -0.812745451927185, -0.213397994637489],
           [5.140193462371826, -2.941965818405151, -0.055521309375763,
            1.96869695186615, -1.29790472984314, 0.034493416547775,
            -1.094233393669128, -0.767234861850739, -1.832728981971741,
            2.556174278259277, 1.285072922706604, 2.783454418182373],
           [-3.460673093795776, 0.700069725513458, -1.184944987297058,
            -3.619244337081909, 3.242199659347534, -0.404601752758026,
            -2.755020618438721, -0.827874422073364, 1.487833738327026,
            -1.766772627830505, -0.019650995731354, -1.590330123901367]], dtype=float32)
  if ones_like:
    W_re = numpy.ones_like(W_re)
  if factor != 1:
    W_re *= factor
  assert W_re.shape == (n_out, n_out * 4)
  # <tf.Tensor 'output/rec/zeros:0' shape=(?, ?) dtype=float32>:
  cell_state = array([[ 0.,  0.,  0.]], dtype=numpy.float32)
  assert cell_state.shape == (n_batch, n_out)
  # <tf.Tensor 'extern_data/placeholders/data/sequence_mask_time_major/index_cast_float32:0' shape=(?, ?) dtype=float32>:
  index_float = \
    array([[ 1.],
           [ 1.],
           [ 1.],
           [ 1.],
           [ 1.]], dtype=numpy.float32)
  # <tf.Tensor 'output/rec/LstmGenericBase:0' shape=(?, ?, 3) dtype=float32>:
  assert index_float.shape == (n_time, n_batch)
  in0 = \
    array([[[-9.368341172266703e-12, -1.167426881865996e-18,
             6.303897243924439e-04]],
           [[1.045539761435066e-07, -7.615810632705688e-01,
             2.735287125688046e-06]],
           [[7.604487538337708e-01, -8.968127929165348e-08,
             7.615941762924194e-01]],
           [[5.488518013407884e-07, -8.968121534280726e-08,
             7.616176009178162e-01]],
           [[3.996720618921200e-19, -9.847509092886231e-12,
             9.616374969482422e-01]]], dtype=float32)
  if ones_like:
    in0 = numpy.ones_like(in0)
  if factor != 1:
    in0 *= factor
  assert in0.shape == (n_time, n_batch, n_out)
  # <tf.Tensor 'output/rec/LstmGenericBase:1' shape=(?, ?, 12) dtype=float32>:
  in1 = \
    array([[[-9.481454683879509e-12, -9.999690055847168e-01,
             9.999994039535522e-01, 9.481454683879509e-12,
             9.999690055847168e-01, 1.000000000000000e+00,
             7.535594544194613e-12, 1.300011009361175e-19,
             1.000000000000000e+00, 9.880700707435608e-01,
             1.532898954536958e-18, 8.277241722680628e-04]],
           [[1.000000000000000e+00, -9.999688863754272e-01,
             2.735287125688046e-06, 1.000000000000000e+00,
             7.444035166059848e-09, 2.734021336436854e-06,
             1.052110642194748e-01, 9.999998807907104e-01,
             1.265849758347315e-09, 1.372830524815072e-07,
             1.000000000000000e+00, 1.000000000000000e+00]],
           [[9.972782731056213e-01, -8.968127929165348e-08,
             1.000000000000000e+00, 9.972844123840332e-01,
             2.056131756665299e-35, 1.000000000000000e+00,
             1.915361472288072e-22, 8.968407172460502e-08,
             8.604143175716672e-09, 1.000000000000000e+00,
             1.000000000000000e+00, 1.000000000000000e+00]],
           [[5.488518013407884e-07, -8.968121534280726e-08,
             1.000055909156799e+00, 5.488518013407884e-07,
             6.375615251193742e-25, 1.000000000000000e+00,
             1.951400955893235e-17, 9.999992847442627e-01,
             5.593653258983977e-05, 1.000000000000000e+00,
             1.000000000000000e+00, 1.000000000000000e+00]],
           [[9.999997615814209e-01, -3.767583223179827e-07,
             2.000055789947510e+00, 9.999997615814209e-01,
             2.870771140806028e-07, 1.000000000000000e+00,
             8.848448883325144e-12, 1.000000000000000e+00,
             1.000000000000000e+00, 5.247835948298600e-19,
             2.613746983115561e-05, 9.975166320800781e-01]]], dtype=float32)
  if ones_like:
    in1 = numpy.ones_like(in1)
  if factor != 1:
    in1 *= factor
  assert in1.shape == (n_time, n_batch, n_out * 4)
  # <tf.Tensor 'gradients/objective/loss/output/loss_init/flatten_with_seq_len_mask/swapaxes/transpose_grad/transpose:0' shape=(?, ?, 3) dtype=float32>:
  grad_in = \
    array([[[0.576846659183502, -0.19706067442894, -0.684425234794617]],
           [[1.117202281951904, 0.946405112743378, -0.533451914787292]],
           [[0.822037994861603, 1.044727325439453, -1.008405923843384]],
           [[-0.755452394485474, -0.606451511383057, 0.335312634706497]],
           [[0.122484095394611, 1.015499114990234, 0.080147251486778]]], dtype=float32)
  if ones_like:
    grad_in = numpy.ones_like(grad_in)
  if factor != 1:
    grad_in *= factor
  assert grad_in.shape == (n_time, n_batch, n_out)
  zeros2 = array([[ 0.,  0.,  0.]], dtype=numpy.float32)
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
  from TFNativeOp import make_lstm_op
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
  with tf.Session() as session:
    print("dummy out:", session.run(list(dummy)))
    grad_op = _lstm_grad_op(session)
    args = _demo_lstm_grad_args()
    placeholders = [tf.placeholder(v.dtype) for v in args]
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
      }
    }
  }
  config = Config({
    "debug_print_layer_output_template": True,
    "extern_data": {
      "data": {"sparse": True, "dim": 3},
    }
  })
  with make_scope() as session:
    net = TFNetwork(config=config)
    net.construct_from_dict(net_dict)
    inp = net.extern_data.data["data"]
    out = net.get_default_output_layer().output
    assert out.time_dim_axis == 0 and out.batch_dim_axis == 1 and out.shape == (None,) and out.dtype == "int32"
    out_v = session.run(
      out.placeholder,
      feed_dict={
        inp.placeholder: [[0] * n_time],
        inp.size_placeholder[0]: [n_time]
      })
    assert isinstance(out_v, numpy.ndarray)
    assert out_v.shape == (n_time, n_batch)
    assert_equal(out_v[:, 0].tolist(), [0, 1, 2])


def test_search_no_rec_explicit():
  from TFNetworkRecLayer import _SubnetworkRecCell
  beam_size = 3
  logits = numpy.array([
    [1., 2., 3., 0.],
    [0., 4.5, 3., 6.],
    [5., 8., 7.5, 0.]], dtype="float32")
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
    {"src_beam_idxs": [0, 0, 0], "scores": [3., 2., 1.], "labels": [2, 1, 0], "step": 0},
    {"src_beam_idxs": [0, 1, 0], "scores": [9., 8., 7.5], "labels": [3, 3, 1], "step": 1},
    {"src_beam_idxs": [0, 0, 1], "scores": [17., 16.5, 16.], "labels": [1, 2, 1], "step": 2},
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
    "output": {"class": "rec", "from": ["data"], "unit": {
      "output": {
        "class": "choice", "from": ["data:source"], "input_type": "log_prob",
        "explicit_search_source": "prev:output", 'initial_output': 0,
        "beam_size": beam_size, "target": "classes"}
    }}
  }
  extern_data = ExternData({
    "data": {"dim": n_classes},
    "classes": {"dim": n_classes, "sparse": True, "available_for_inference": False}})
  net = TFNetwork(
    extern_data=extern_data, search_flag=True, train_flag=False, eval_flag=False)
  net.construct_from_dict(net_dict)
  assert_equal(net.used_data_keys, {"data"})  # not classes
  rec_layer = net.layers["output"]
  assert isinstance(rec_layer, RecLayer)
  subnet = rec_layer.cell
  assert isinstance(subnet, _SubnetworkRecCell)
  assert_equal(subnet.layers_in_loop, ["output"])
  sub_layer = subnet.net.layers["output"]
  assert isinstance(sub_layer, ChoiceLayer)
  assert_equal(sub_layer.output.beam_size, beam_size)
  assert_equal(rec_layer.output.beam_size, beam_size)
  input_search_choices = net.get_search_choices(sources=rec_layer.sources)
  assert not input_search_choices
  assert rec_layer.output.is_time_major
  assert_equal(rec_layer.get_search_beam_size(), beam_size)
  feed_dict = {
    net.extern_data.data["data"].placeholder: logits,
    net.extern_data.data["data"].size_placeholder[0]: [n_time]}
  with tf.Session() as session:
    assert_equal(session.run(net.get_data_batch_dim(), feed_dict=feed_dict), n_batch)
    out, out_sizes = session.run(
      (rec_layer.output.placeholder, rec_layer.output.get_sequence_lengths()),
      feed_dict=feed_dict)
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
  from TFNetworkRecLayer import _SubnetworkRecCell
  beam_size = 3
  logits = numpy.array([
    [-1., -2., -3., -9.],
    [-0.6, -6., -0.5, -2.],
    [-0.4, -0.6, -0.7, -1.]], dtype="float32")
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
    {"src_beam_idxs": [0, 0, 0], "scores": [-1., -2., -3.], "labels": [0, 1, 2], "step": 0},
    {"src_beam_idxs": [0, 1, 1], "scores": [-2., -2.5, -2.6], "labels": [0, 2, 0], "step": 1},
    {"src_beam_idxs": [1, 0, 1], "scores": [-2.9, -3., -3.1], "labels": [0, 0, 1], "step": 2},
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
    "output": {"class": "rec", "from": ["data"], "max_seq_len": n_time, "unit": {
      "output": {
        "class": "choice", "from": ["data:source"], "input_type": "log_prob",
        "explicit_search_source": "prev:output", 'initial_output': 0,
        "beam_size": beam_size, "length_normalization": True,
        "target": "classes"},
      "end": {"class": "compare", "from": ["output"], "value": 0}
    }}
  }
  extern_data = ExternData({
    "data": {"dim": n_classes},
    "classes": {"dim": n_classes, "sparse": True, "available_for_inference": False}})
  net = TFNetwork(
    extern_data=extern_data, search_flag=True, train_flag=False, eval_flag=False)
  net.construct_from_dict(net_dict)
  assert_equal(net.used_data_keys, {"data"})  # not classes
  rec_layer = net.layers["output"]
  assert isinstance(rec_layer, RecLayer)
  subnet = rec_layer.cell
  assert isinstance(subnet, _SubnetworkRecCell)
  assert_equal(set(subnet.layers_in_loop), {"output", "end"})
  sub_layer = subnet.net.layers["output"]
  assert isinstance(sub_layer, ChoiceLayer)
  assert_equal(sub_layer.output.beam_size, beam_size)
  assert_equal(rec_layer.output.beam_size, beam_size)
  input_search_choices = net.get_search_choices(sources=rec_layer.sources)
  assert not input_search_choices
  assert rec_layer.output.is_time_major
  assert_equal(rec_layer.get_search_beam_size(), beam_size)
  feed_dict = {
    net.extern_data.data["data"].placeholder: logits,
    net.extern_data.data["data"].size_placeholder[0]: [n_time]}
  with tf.Session() as session:
    assert_equal(session.run(net.get_data_batch_dim(), feed_dict=feed_dict), n_batch)
    out, out_sizes = session.run(
      (rec_layer.output.placeholder, rec_layer.output.get_sequence_lengths()),
      feed_dict=feed_dict)
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


def test_rec_layer_move_out_of_loop():
  from TFNetworkRecLayer import _SubnetworkRecCell
  from TFUtil import get_global_train_flag_placeholder
  n_src_dim = 5
  n_tgt_dim = 7
  beam_size = 12
  config = Config()
  config.update({"debug_print_layer_output_template": True})

  def get_net_dict():
    return {
      "source_embed": {"class": "linear", "activation": None, "with_bias": False, "n_out": 6},

      "lstm0_fw": {"class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                   "initial_state": "var", "n_out": 10, "direction": 1, "from": ["source_embed"]},
      "lstm0_bw": {"class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                   "initial_state": "var", "n_out": 10, "direction": -1, "from": ["source_embed"]},

      "lstm1_fw": {"class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                   "initial_state": "var", "n_out": 10, "direction": 1, "from": ["lstm0_fw", "lstm0_bw"]},
      "lstm1_bw": {"class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                   "initial_state": "var", "n_out": 10, "direction": -1, "from": ["lstm0_fw", "lstm0_bw"]},

      "encoder": {"class": "copy", "from": ["lstm1_fw", "lstm1_bw"]},
      "enc_ctx": {"class": "linear", "activation": None, "with_bias": True, "from": ["encoder"], "n_out": 10},
      "fertility": {"class": "linear", "activation": "sigmoid", "with_bias": False, "from": ["encoder"], "n_out": 1},

      "output": {"class": "rec", "from": [], "unit": {
        'output': {'class': 'choice', 'target': 'classes', 'beam_size': beam_size, 'from': ["output_prob"],
                   "initial_output": 0},
        "end": {"class": "compare", "from": ["output"], "value": 0},
        'target_embed': {'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'], "n_out": 6,
                         "initial_output": "apply(0)"},
        "weight_feedback": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev:accum_att_weights"],
                            "n_out": 10},
        "prev_s_state": {"class": "get_last_hidden_state", "from": ["prev:s"], "n_out": 20},
        "prev_s_transformed": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev_s_state"],
                               "n_out": 10},
        "energy_in": {"class": "combine", "kind": "add",
                      "from": ["base:enc_ctx", "weight_feedback", "prev_s_transformed"], "n_out": 10},
        "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
        "energy": {"class": "linear", "activation": None, "with_bias": False, "from": ["energy_tanh"], "n_out": 1},
        "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, enc-T, 1)
        "accum_att_weights": {"class": "eval", "from": ["prev:accum_att_weights", "att_weights", "base:fertility"],
                              "eval": "source(0) + source(1) / (2.0 * source(2))",
                              "out_type": {"dim": 1, "shape": (None, 1)}},
        "att": {"class": "generic_attention", "weights": "att_weights", "base": "base:encoder"},
        "s": {"class": "rnn_cell", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
              "initial_state": "var", "from": ["target_embed", "att"], "n_out": 10},
        "readout_in": {"class": "linear", "from": ["prev:s", "prev:target_embed", "att"], "activation": None,
                       "n_out": 10},
        "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
        "output_prob": {"class": "softmax", "from": ["readout"], "target": "classes", "loss": "ce"}
      }, "target": "classes", "max_seq_len": 20},

      "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance", "target": "classes"}
    }

  print("Constructing search network.")
  tf.reset_default_graph()
  extern_data = ExternData({
    "data": {"dim": n_src_dim, "sparse": True},
    "classes": {"dim": n_tgt_dim, "sparse": True, "available_for_inference": False}})
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
    from GeneratingDataset import StaticDataset
    from TFDataPipeline import FeedDictDataProvider
    from EngineBatch import Batch, BatchSetGenerator
    from Util import dict_joined
    dataset = StaticDataset(
      data=[
        {"data": numpy.array([2, 4, 1, 0]), "classes": numpy.array([3, 6, 0])},
        {"data": numpy.array([2, 4, 1, 3, 0]), "classes": numpy.array([3, 6, 2, 1, 4, 5, 0])}],
      output_dim={"data": [n_src_dim, 1], "classes": [n_tgt_dim, 1]})
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

    with tf.Session() as session:
      net.initialize_params(session)
      data_provider = FeedDictDataProvider(
        tf_session=session, extern_data=extern_data,
        data_keys=["data", "classes"],
        dataset=dataset, batches=batches)
      feed_dict, meta_step_info = data_provider.get_feed_dict(single_threaded=True)
      if isinstance(net.train_flag, tf.Tensor):
        feed_dict[net.train_flag] = True
      try:
        out = session.run(
          dict_joined(
            {"data:%s:seq_len" % k: v.get_sequence_lengths() for (k, v) in net.extern_data.data.items()},
            {"layer:%s:out_seq_len" % k: l.output.get_sequence_lengths() for (k, l) in net.layers.items()},
            {"rec_layer_in:%s:out_seq_len" % k: l.output.get_sequence_lengths()
             for (k, l) in out_layer.cell.input_layers_net.layers.items()} if out_layer.cell.input_layers_net else {},
            {"rec_layer_out:%s:out_seq_len" % k: l.output.get_sequence_lengths()
             for (k, l) in out_layer.cell.output_layers_net.layers.items()} if out_layer.cell.output_layers_net else {},
          ),
          feed_dict=feed_dict)
        pprint(out)
        out = session.run(
          {"objective": net.get_objective()},
          feed_dict=feed_dict)
        pprint(out)
      except Exception as exc:
        print("Exception happened:", str(exc).splitlines()[0])
        print("Writing TF log file.")
        writer = tf.summary.FileWriter(".", filename_suffix="test_rec_layer_move_out_of_loop")
        writer.add_graph(session.graph)
        writer.close()
        raise

  print("Constructing train network.")
  tf.reset_default_graph()
  extern_data = ExternData({
    "data": {"dim": n_src_dim, "sparse": True},
    "classes": {"dim": n_tgt_dim, "sparse": True, "available_for_inference": False}})
  train_net = TFNetwork(
    extern_data=extern_data, search_flag=False, train_flag=get_global_train_flag_placeholder(), config=config)
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
  tf.reset_default_graph()
  extern_data = ExternData({
    "data": {"dim": n_src_dim, "sparse": True},
    "classes": {"dim": n_tgt_dim, "sparse": True, "available_for_inference": False}})
  train_not_optim_net = TFNetwork(
    extern_data=extern_data, search_flag=False, train_flag=get_global_train_flag_placeholder(), config=config)
  assert train_not_optim_net.eval_flag is True
  train_not_optim_net.construct_from_dict(get_net_dict())
  train_not_optim_out_layer = train_not_optim_net.layers["output"]
  assert isinstance(train_not_optim_out_layer, RecLayer)
  assert isinstance(train_not_optim_out_layer.cell, _SubnetworkRecCell)
  assert not train_not_optim_out_layer.cell.input_layers_moved_out
  assert not train_not_optim_out_layer.cell.output_layers_moved_out
  train(train_not_optim_net)


def test_rec_layer_move_out_of_loop_keep_constraints():
  from TFNetworkRecLayer import _SubnetworkRecCell
  from TFUtil import get_global_train_flag_placeholder
  n_src_dim = 5
  n_tgt_dim = 7
  beam_size = 12
  config = Config()
  config.update({"debug_print_layer_output_template": True})

  def get_net_dict(l2_target_embed=0.0, l2_readout_in=0.0):
    return {
      "source_embed": {"class": "linear", "activation": None, "with_bias": False, "n_out": 6},

      "lstm0_fw": {"class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                   "initial_state": "var", "n_out": 10, "direction": 1, "from": ["source_embed"]},
      "lstm0_bw": {"class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                   "initial_state": "var", "n_out": 10, "direction": -1, "from": ["source_embed"]},

      "lstm1_fw": {"class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                   "initial_state": "var", "n_out": 10, "direction": 1, "from": ["lstm0_fw", "lstm0_bw"]},
      "lstm1_bw": {"class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                   "initial_state": "var", "n_out": 10, "direction": -1, "from": ["lstm0_fw", "lstm0_bw"]},

      "encoder": {"class": "copy", "from": ["lstm1_fw", "lstm1_bw"]},
      "enc_ctx": {"class": "linear", "activation": None, "with_bias": True, "from": ["encoder"], "n_out": 10},
      "fertility": {"class": "linear", "activation": "sigmoid", "with_bias": False, "from": ["encoder"], "n_out": 1},

      "output": {"class": "rec", "from": [], "unit": {
        'output': {'class': 'choice', 'target': 'classes', 'beam_size': beam_size, 'from': ["output_prob"],
                   "initial_output": 0},
        "end": {"class": "compare", "from": ["output"], "value": 0},
        'target_embed': {'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'], "n_out": 6,
                         "initial_output": "apply(0)", "L2": l2_target_embed},
        "weight_feedback": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev:accum_att_weights"],
                            "n_out": 10},
        "prev_s_state": {"class": "get_last_hidden_state", "from": ["prev:s"], "n_out": 20},
        "prev_s_transformed": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev_s_state"],
                               "n_out": 10},
        "energy_in": {"class": "combine", "kind": "add",
                      "from": ["base:enc_ctx", "weight_feedback", "prev_s_transformed"], "n_out": 10},
        "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
        "energy": {"class": "linear", "activation": None, "with_bias": False, "from": ["energy_tanh"], "n_out": 1},
        "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, enc-T, 1)
        "accum_att_weights": {"class": "eval", "from": ["prev:accum_att_weights", "att_weights", "base:fertility"],
                              "eval": "source(0) + source(1) / (2.0 * source(2))",
                              "out_type": {"dim": 1, "shape": (None, 1)}},
        "att": {"class": "generic_attention", "weights": "att_weights", "base": "base:encoder"},
        "s": {"class": "rnn_cell", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
              "initial_state": "var", "from": ["target_embed", "att"], "n_out": 10},
        "readout_in": {"class": "linear", "from": ["prev:s", "prev:target_embed", "att"], "activation": None,
                       "n_out": 10, "L2": l2_readout_in},
        "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
        "output_prob": {"class": "softmax", "from": ["readout"], "target": "classes", "loss": "ce"}
      }, "target": "classes", "max_seq_len": 20},

      "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance", "target": "classes"}
    }

  print("Constructing train network without constraints")
  tf.reset_default_graph()
  extern_data = ExternData({
    "data": {"dim": n_src_dim, "sparse": True},
    "classes": {"dim": n_tgt_dim, "sparse": True, "available_for_inference": False}})
  train_net = TFNetwork(
    extern_data=extern_data, search_flag=False, train_flag=get_global_train_flag_placeholder(), config=config)
  assert train_net.eval_flag is True
  train_net.construct_from_dict(get_net_dict(l2_target_embed=0.0, l2_readout_in=0.0))
  train_out_layer = train_net.layers["output"]
  assert isinstance(train_out_layer, RecLayer)
  assert isinstance(train_out_layer.cell, _SubnetworkRecCell)
  assert_equal(set(train_out_layer.cell.input_layers_moved_out), {"output", "target_embed"})
  assert_equal(set(train_out_layer.cell.output_layers_moved_out), {"output_prob", "readout_in", "readout"})
  assert_equal(train_net.get_total_constraints(), 0)

  print("Constructing train network with L2 norm on moved out input layer")
  tf.reset_default_graph()
  extern_data = ExternData({
    "data": {"dim": n_src_dim, "sparse": True},
    "classes": {"dim": n_tgt_dim, "sparse": True, "available_for_inference": False}})
  train_net = TFNetwork(
    extern_data=extern_data, search_flag=False, train_flag=get_global_train_flag_placeholder(), config=config)
  assert train_net.eval_flag is True
  train_net.construct_from_dict(get_net_dict(l2_target_embed=0.01, l2_readout_in=0.0))
  train_out_layer = train_net.layers["output"]
  assert isinstance(train_out_layer, RecLayer)
  assert isinstance(train_out_layer.cell, _SubnetworkRecCell)
  assert_equal(set(train_out_layer.cell.input_layers_moved_out), {"output", "target_embed"})
  assert_equal(set(train_out_layer.cell.output_layers_moved_out), {"output_prob", "readout_in", "readout"})
  assert_not_equal(train_net.get_total_constraints(), 0)

  print("Constructing train network with L2 norm on moved out output layer")
  tf.reset_default_graph()
  extern_data = ExternData({
    "data": {"dim": n_src_dim, "sparse": True},
    "classes": {"dim": n_tgt_dim, "sparse": True, "available_for_inference": False}})
  train_net = TFNetwork(
    extern_data=extern_data, search_flag=False, train_flag=get_global_train_flag_placeholder(), config=config)
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
  from TFNetworkRecLayer import _SubnetworkRecCell
  from TFUtil import get_global_train_flag_placeholder
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
      "from": "data:att_weights", "sizes": "data:att_weights_sizes", "num_axes": 2,
      "declare_same_sizes_as": {0: "data:classes", 1: "data"}},
    "source_embed": {"class": "linear", "activation": None, "with_bias": False, "n_out": 6, "from": "data"},
    "encoder": {"class": "linear", "from": "source_embed", "activation": "tanh", "n_out": EncValueTotalDim},
    "enc_ctx": {"class": "linear", "activation": None, "with_bias": True, "from": "encoder", "n_out": EncKeyTotalDim},
    "enc_value": {"class": "split_dims", "axis": "F", "dims": (AttNumHeads, EncValuePerHeadDim),
                  "from": "encoder"},  # (B, enc-T, H, D'/H)
    "inv_fertility": {"class": "linear", "activation": "sigmoid", "with_bias": False, "from": "encoder", "n_out": AttNumHeads},

    "output": {"class": "rec", "from": ["ref_att_weights"], "unit": {
      'output': {'class': 'choice', 'target': 'classes', 'beam_size': beam_size, 'from': ["output_prob"],
                 "initial_output": 0},
      "end": {"class": "compare", "from": ["output"], "value": 0},
      'target_embed': {'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'], "n_out": 6,
                       "initial_output": 0},
      "weight_feedback": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev:accum_att_weights"],
                          "n_out": EncKeyTotalDim},
      "prev_s_transformed": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev:s"],
                             "n_out": EncKeyTotalDim},
      "energy_in": {"class": "combine", "kind": "add",
                    "from": ["base:enc_ctx", "weight_feedback", "prev_s_transformed"], "n_out": EncKeyTotalDim},
      "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
      "energy": {"class": "linear", "activation": None, "with_bias": False, "from": ["energy_tanh"], "n_out": AttNumHeads, "is_output_layer": True},
      "att_weights": {"class": "copy", "from": "data:source"},
      # "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, enc-T, H)
      "att_weights_with_fertility": {"class": "eval", "from": ["att_weights", "base:inv_fertility"],
                                     "eval": "source(0) * source(1) * 0.5"},
      "accum_att_weights": {"class": "cumsum", "from": "att_weights_with_fertility"},
      "att0": {"class": "generic_attention", "weights": "att_weights", "base": "base:enc_value"},  # (B, H, V)
      "att": {"class": "merge_dims", "axes": "except_time", "from": ["att0"]},  # (B, H*V)
      "s": {"class": "rnn_cell", "unit": "standardlstm",
            "from": ["target_embed", "att"], "n_out": 10},
      "readout_in": {"class": "linear", "from": ["prev:s", "prev:target_embed", "att"], "activation": None,
                     "n_out": 10},
      "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
      "output_prob": {"class": "softmax", "from": ["readout"], "target": "classes", "loss": "ce"}
    }, "target": "classes", "max_seq_len": 20},

    "att_distill_loss": {
      "class": "eval", "from": ["output/energy", "ref_att_weights"],
      "out_type": (lambda sources, **kwargs: sources[0].output.copy_template_excluding_spatial_dim(-1)),
      "eval": "softmax_cross_entropy_over_size(" +
              "logits=source(0, as_data=True, auto_convert=False)," +
              "labels=source(1, as_data=True, auto_convert=False))",
      "loss": "as_is"},
  }

  def train(net, session):
    """
    :param TFNetwork net:
    :param tf.Session session:
    """
    from GeneratingDataset import StaticDataset
    from TFDataPipeline import FeedDictDataProvider
    from EngineBatch import Batch, BatchSetGenerator
    from Util import dict_joined, softmax
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
          "att_weights_sizes": numpy.array([3, 4])
        },
        {
          "data": numpy.array([2, 4, 1, 3, 0]),
          "classes": numpy.array([3, 6, 2, 1, 4, 5, 0]),
          "att_weights": create_rnd_flat_att_weights(7, 5),
          "att_weights_sizes": numpy.array([7, 5])
        }],
      output_dim={
        "data": [n_src_dim, 1],
        "classes": [n_tgt_dim, 1],
        "att_weights": [AttNumHeads, 2],
        "att_weights_sizes": [1, 1]})
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
      tf_session=session, extern_data=extern_data,
      data_keys=["data", "classes", "att_weights", "att_weights_sizes"],
      dataset=dataset, batches=batches)
    feed_dict, meta_step_info = data_provider.get_feed_dict(single_threaded=True)
    if isinstance(net.train_flag, tf.Tensor):
      feed_dict[net.train_flag] = True
    try:
      print("session run for seq lens output:")
      out = session.run(
        dict_joined(
          {"data:%s:seq_len" % k: v.get_sequence_lengths() for (k, v) in net.extern_data.data.items()},
          {"layer:%s:out_seq_len" % k: l.output.get_sequence_lengths() for (k, l) in net.layers.items()},
          {"rec_layer_in:%s:out_seq_len" % k: l.output.get_sequence_lengths()
           for (k, l) in out_layer.cell.input_layers_net.layers.items()} if out_layer.cell.input_layers_net else {},
          {"rec_layer_out:%s:out_seq_len" % k: l.output.get_sequence_lengths()
           for (k, l) in out_layer.cell.output_layers_net.layers.items()} if out_layer.cell.output_layers_net else {},
        ),
        feed_dict=feed_dict)
      pprint(out)
      print("session run for objective output:")
      losses, total_loss, total_constraints = net.get_losses_initialized(with_total=True)
      # TODO: strange ref att weights?
      out = session.run(
        {"total_loss": total_loss, "total_constraints": tf.convert_to_tensor(total_constraints),
         "losses": {name: loss.get_loss_value() for name, loss in losses.items()},
         "att_distill_loss_in": [s.output.placeholder for s in net.layers["att_distill_loss"].sources],
         "att_distill_loss_out": net.layers["att_distill_loss"].output.placeholder},
        feed_dict=feed_dict)
      pprint(out)
    except Exception as exc:
      print("Exception happened:", str(exc).splitlines()[0])
      print("Writing TF log file.")
      writer = tf.summary.FileWriter(".", filename_suffix="test_rec_layer_move_out_of_loop")
      writer.add_graph(session.graph)
      writer.close()
      raise

  print("Constructing train network.")
  with make_scope() as session:
    extern_data = ExternData({
      "data": {"dim": n_src_dim, "sparse": True},
      "classes": {"dim": n_tgt_dim, "sparse": True, "available_for_inference": False},
      "att_weights": {"shape": (None, AttNumHeads), "available_for_inference": False},
      "att_weights_sizes": {"shape": (None,), "dtype": "int32", "available_for_inference": False}})
    train_net = TFNetwork(
      extern_data=extern_data, search_flag=False, train_flag=get_global_train_flag_placeholder(), config=config)
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
    extern_data = ExternData({
      "data": {"dim": 13, "sparse": True},
      "classes": {"dim": 17, "sparse": True, "available_for_inference": False}})
    net = TFNetwork(extern_data=extern_data, train_flag=True, config=config)
    net.construct_from_dict({
      "source_embed": {"class": "linear", "activation": None, "with_bias": False, "n_out": 6},
      "lstm0_fw": {"class": "rec", "unit": "standardlstm", "n_out": 10, "direction": 1, "from": ["source_embed"]},
      "lstm0_bw": {"class": "rec", "unit": "standardlstm", "n_out": 10, "direction": -1, "from": ["source_embed"]},
      "lstm1_fw": {"class": "rec", "unit": "standardlstm", "n_out": 10, "direction": 1, "from": ["lstm0_fw", "lstm0_bw"]},
      "lstm1_bw": {"class": "rec", "unit": "standardlstm", "n_out": 10, "direction": -1, "from": ["lstm0_fw", "lstm0_bw"]},
      "encoder": {"class": "copy", "from": ["lstm1_fw", "lstm1_bw"]},
      "enc_value": {"class": "split_dims", "axis": "F", "dims": (4, 5), "from": ["encoder"]},
      "output": {"class": "copy", "from": ["enc_value"]},
    })
    size = extern_data.data["data"].get_size_dim_tag(0)
    print("data size:", size)
    for name in ["source_embed", "lstm0_fw", "lstm1_fw", "encoder", "enc_value", "output"]:
      layer = net.layers[name]
      layer_size = layer.output.get_size_dim_tag(0)
      print("layer:", layer, "size:", layer_size)
      assert size == layer_size, "no match for layer %r" % layer
    print("All good.")


def test_rec_layer_rnn_train_and_search():
  from TFNetworkRecLayer import _SubnetworkRecCell
  n_src_dim = 5
  n_tgt_dim = 7
  beam_size = 3
  config = Config()
  config.update({
    "debug_print_layer_output_template": True,
    "debug_print_layer_output_shape": True})
  EncKeyTotalDim = 14
  AttNumHeads = 1
  EncValueTotalDim = 14
  EncValuePerHeadDim = EncValueTotalDim // AttNumHeads
  LstmDim = EncValueTotalDim // 2
  target = "classes"

  net_dict = {
    "lstm0_fw": {"class": "rec", "unit": "nativelstm2", "n_out": LstmDim, "direction": 1, "from": ["data"]},
    "lstm0_bw": {"class": "rec", "unit": "nativelstm2", "n_out": LstmDim, "direction": -1, "from": ["data"]},
    "lstm0_pool": {"class": "pool", "mode": "max", "padding": "same", "pool_size": (3,),
                   "from": ["lstm0_fw", "lstm0_bw"], "trainable": False},
    "lstm1_fw": {"class": "rec", "unit": "nativelstm2", "n_out": LstmDim, "direction": 1, "from": ["lstm0_pool"]},
    "lstm1_bw": {"class": "rec", "unit": "nativelstm2", "n_out": LstmDim, "direction": -1, "from": ["lstm0_pool"]},
    "encoder": {"class": "copy", "from": ["lstm1_fw", "lstm1_bw"]},  # dim: EncValueTotalDim
    "enc_ctx": {"class": "linear", "activation": None, "with_bias": True, "from": ["encoder"],
                "n_out": EncKeyTotalDim},
    "inv_fertility": {"class": "linear", "activation": "sigmoid", "with_bias": False, "from": ["encoder"],
                      "n_out": AttNumHeads},
    "enc_value": {"class": "split_dims", "axis": "F", "dims": (AttNumHeads, EncValuePerHeadDim),
                  "from": ["encoder"]},  # (B, enc-T, H, D'/H)
    "output": {"class": "rec", "from": [], 'cheating': config.bool("cheating", False), "unit": {
      'output': {'class': 'choice', 'target': target, 'beam_size': beam_size,
                 'cheating': config.bool("cheating", False), 'from': ["output_prob"], "initial_output": 0},
      "end": {"class": "compare", "from": ["output"], "value": 0},
      'target_embed': {'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'], "n_out": 10,
                       "initial_output": 0},  # feedback_input
      "weight_feedback": {"class": "linear", "activation": None, "with_bias": False,
                          "from": ["prev:accum_att_weights"], "n_out": EncKeyTotalDim},
      "s_transformed": {"class": "linear", "activation": None, "with_bias": False, "from": ["s"],
                        "n_out": EncKeyTotalDim},
      "energy_in": {"class": "combine", "kind": "add", "from": ["base:enc_ctx", "weight_feedback", "s_transformed"],
                    "n_out": EncKeyTotalDim},
      "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
      "energy": {"class": "linear", "activation": None, "with_bias": False, "from": ["energy_tanh"],
                 "n_out": AttNumHeads},  # (B, enc-T, H)
      "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, enc-T, H)
      "accum_att_weights": {"class": "eval", "from": ["prev:accum_att_weights", "att_weights", "base:inv_fertility"],
                            "eval": "source(0) + source(1) * source(2) * 0.5",
                            "out_type": {"dim": AttNumHeads, "shape": (None, AttNumHeads)}},
      "att0": {"class": "generic_attention", "weights": "att_weights", "base": "base:enc_value"},  # (B, H, V)
      "att": {"class": "merge_dims", "axes": "except_batch", "from": ["att0"]},  # (B, H*V)
      "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["prev:target_embed", "prev:att"], "n_out": 10},
      "readout_in": {"class": "linear", "from": ["s", "prev:target_embed", "att"], "activation": None, "n_out": 10},
      "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
      "output_prob": {
        "class": "softmax", "from": ["readout"], "dropout": 0.3,
        "target": target, "loss": "ce", "loss_opts": {"label_smoothing": 0.1}}
    }, "target": target, "max_seq_len": "max_len_from('base:encoder')"},
    "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance", "target": target}
  }

  def run(train_flag=False, search_flag=False):
    """
    :param bool train_flag:
    :param bool search_flag:
    """
    print("Create network with train_flag=%r, search_flag=%r." % (train_flag, search_flag))

    from GeneratingDataset import StaticDataset
    from TFDataPipeline import FeedDictDataProvider
    from EngineBatch import Batch, BatchSetGenerator
    from Util import dict_joined
    dataset = StaticDataset(
      data=[
        {"data": numpy.random.normal(size=(11, n_src_dim)).astype("float32"),
         "classes": numpy.array([3, 6, 0])},
        {"data": numpy.random.normal(size=(13, n_src_dim)).astype("float32"),
         "classes": numpy.array([3, 6, 2, 1, 4, 5, 0])}],
      output_dim={"data": [n_src_dim, 2], "classes": [n_tgt_dim, 1]})
    dataset.init_seq_order(epoch=1)
    batch = Batch()
    batch.add_sequence_as_slice(seq_idx=0, seq_start_frame=0, length=dataset.get_seq_length(0))
    batch.add_sequence_as_slice(seq_idx=1, seq_start_frame=0, length=dataset.get_seq_length(1))
    print("batch:", batch, "num frames:", batch.get_total_num_frames())
    print("batch dims:", batch.max_num_frames_per_slice * batch.num_slices)
    batch_generator = iter([batch])
    batches = BatchSetGenerator(dataset, generator=batch_generator)

    with make_scope() as session:
      extern_data = ExternData({
        "data": {"dim": n_src_dim},
        "classes": {"dim": n_tgt_dim, "sparse": True, "available_for_inference": False}})

      net = TFNetwork(
        extern_data=extern_data, search_flag=search_flag, train_flag=train_flag, config=config)
      net.construct_from_dict(net_dict)

      out_layer = net.layers["output"]
      assert isinstance(out_layer, RecLayer)
      assert isinstance(out_layer.cell, _SubnetworkRecCell)
      net.initialize_params(session)
      data_provider = FeedDictDataProvider(
        tf_session=session, extern_data=extern_data,
        data_keys=["data", "classes"] if train_flag else ["data"],
        dataset=dataset, batches=batches)
      feed_dict, meta_step_info = data_provider.get_feed_dict(single_threaded=True)
      if isinstance(net.train_flag, tf.Tensor):
        feed_dict[net.train_flag] = train_flag
      try:
        print("Output:")
        out = session.run(
          dict_joined(
            {"data:%s:seq_len" % k: v.get_sequence_lengths() for (k, v) in net.extern_data.data.items()},
            {"layer:%s:out_seq_len" % k: l.output.get_sequence_lengths() for (k, l) in net.layers.items()},
            {"rec_layer_in:%s:out_seq_len" % k: l.output.get_sequence_lengths()
             for (k, l) in out_layer.cell.input_layers_net.layers.items()} if out_layer.cell.input_layers_net else {},
            {"rec_layer_out:%s:out_seq_len" % k: l.output.get_sequence_lengths()
             for (k, l) in out_layer.cell.output_layers_net.layers.items()} if out_layer.cell.output_layers_net else {},
            {"output": out_layer.output.placeholder},
            {"objective": tf.convert_to_tensor(net.get_objective())} if train_flag else {}
          ) if train_flag else {"output": out_layer.output.placeholder},
          feed_dict=feed_dict)
        pprint(out)
      except Exception as exc:
        print("Exception happened:", str(exc).splitlines()[0])
        help_on_tf_exception(
          session=session,
          exception=exc, fetches=out_layer.output.placeholder, feed_dict=feed_dict, meta_step_info=meta_step_info,
          extern_data=data_provider.extern_data)
        raise

  run(train_flag=True)
  run(search_flag=True)


def test_same_spatial_dim_after_rec_layers_with_pool():
  with make_scope() as session:
    config = Config({"debug_print_layer_output_template": True})
    extern_data = ExternData({
      "data": {"dim": 13, "sparse": True},
      "classes": {"dim": 17, "sparse": True, "available_for_inference": False},
      "att_weights": {"shape": (None, 1), "available_for_inference": False},
      "att_weights_sizes": {"shape": (None,), "dtype": "int32", "available_for_inference": False}})
    net = TFNetwork(extern_data=extern_data, train_flag=True, config=config)
    net.construct_from_dict({
      "ref_att_weights": {
        "class": "unflatten_nd",
        "from": "data:att_weights", "sizes": "data:att_weights_sizes", "num_axes": 2,
        "declare_same_sizes_as": {0: "data:classes", 1: "encoder"},
        "is_output_layer": True},
      "source_embed": {"class": "linear", "activation": None, "with_bias": False, "n_out": 6},
      "lstm0_fw": {"class": "rec", "unit": "standardlstm", "n_out": 10, "direction": 1, "from": ["source_embed"]},
      "lstm0_bw": {"class": "rec", "unit": "standardlstm", "n_out": 10, "direction": -1, "from": ["source_embed"]},
      "lstm0_pool": {"class": "pool", "mode": "max", "padding": "same", "pool_size": (2,), "from": ["lstm0_fw", "lstm0_bw"]},
      "lstm1_fw": {"class": "rec", "unit": "standardlstm", "n_out": 10, "direction": 1, "from": ["lstm0_pool"]},
      "lstm1_bw": {"class": "rec", "unit": "standardlstm", "n_out": 10, "direction": -1, "from": ["lstm0_pool"]},
      "encoder": {"class": "copy", "from": ["lstm1_fw", "lstm1_bw"]},
      "enc_value": {"class": "split_dims", "axis": "F", "dims": (4, 5), "from": ["encoder"]},
      "output": {"class": "copy", "from": ["enc_value"]},
    })
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
  from TFNetworkRecLayer import _SubnetworkRecCell
  n_src_dim = 5
  n_tgt_dim = 7
  beam_size = 12
  config = Config()
  config.update({"debug_print_layer_output_template": True, "optimize_move_layers_out": False})

  def get_net_dict():
    return {
      "source_embed": {"class": "linear", "activation": None, "with_bias": False, "n_out": 6},

      "lstm0_fw": {"class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                   "initial_state": "var", "n_out": 10, "direction": 1, "from": ["source_embed"]},
      "lstm0_bw": {"class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                   "initial_state": "var", "n_out": 10, "direction": -1, "from": ["source_embed"]},

      "lstm1_fw": {"class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                   "initial_state": "var", "n_out": 10, "direction": 1, "from": ["lstm0_fw", "lstm0_bw"]},
      "lstm1_bw": {"class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                   "initial_state": "var", "n_out": 10, "direction": -1, "from": ["lstm0_fw", "lstm0_bw"]},

      "encoder": {"class": "copy", "from": ["lstm1_fw", "lstm1_bw"]},
      "enc_ctx": {"class": "linear", "activation": None, "with_bias": True, "from": ["encoder"], "n_out": 10},
      "fertility": {"class": "linear", "activation": "sigmoid", "with_bias": False, "from": ["encoder"], "n_out": 1},

      "output": {"class": "rec", "from": [], "unit": {
        'output': {'class': 'choice', 'target': 'classes', 'beam_size': beam_size, 'from': ["output_prob"],
                   "initial_output": 0},
        "end": {"class": "compare", "from": ["output"], "value": 0},
        'target_embed': {'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'], "n_out": 6,
                         "initial_output": "apply(0)"},
        "weight_feedback": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev:accum_att_weights"],
                            "n_out": 10},
        "prev_s_state": {"class": "get_last_hidden_state", "from": ["prev:s"], "n_out": 20},
        "prev_s_transformed": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev_s_state"],
                               "n_out": 10},
        "energy_in": {"class": "combine", "kind": "add",
                      "from": ["base:enc_ctx", "weight_feedback", "prev_s_transformed"], "n_out": 10},
        "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
        "energy": {"class": "linear", "activation": None, "with_bias": False, "from": ["energy_tanh"], "n_out": 1},
        "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, enc-T, 1)
        "accum_att_weights": {"class": "eval", "from": ["prev:accum_att_weights", "att_weights", "base:fertility"],
                              "eval": "source(0) + source(1) / (2.0 * source(2))",
                              "out_type": {"dim": 1, "shape": (None, 1)}},
        "att": {"class": "generic_attention", "weights": "att_weights", "base": "base:encoder"},
        "s": {"class": "rnn_cell", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
              "initial_state": "var", "from": ["target_embed", "att"], "n_out": 10},
        "readout_in": {"class": "linear", "from": ["prev:s", "prev:target_embed", "att"], "activation": None,
                       "n_out": 10},
        "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
        "output_prob": {"class": "softmax", "from": ["readout"], "target": "classes", "loss": "ce"}
      }, "target": "classes", "max_seq_len": 20},

      "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance", "target": "classes"}
    }

  print("Constructing search network.")
  tf.reset_default_graph()
  extern_data = ExternData({
    "data": {"dim": n_src_dim, "sparse": True},
    "classes": {"dim": n_tgt_dim, "sparse": True, "available_for_inference": False}})
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
  from GeneratingDataset import DummyDataset
  from TFNetworkLayer import InternalLayer, SourceLayer, ReduceLayer
  train_data = DummyDataset(input_dim=2, output_dim=3, num_seqs=10, seq_len=5)
  with make_scope() as session:
    extern_data = ExternData()
    extern_data.init_from_dataset(train_data)
    net = TFNetwork(extern_data=extern_data)
    with tf.variable_scope("input_no_time_l"):
      input_no_time_l = InternalLayer(
        name="input_no_time_l", network=net, out_type={"dim": train_data.num_inputs, "time_dim_axis": None})
      print("Input layer (without time-dim):", input_no_time_l)
      assert input_no_time_l.output.shape == (train_data.num_inputs,)
      assert input_no_time_l.output.time_dim_axis is None
      assert not input_no_time_l.output.sparse
      assert input_no_time_l.output.dim == input_no_time_l.output.shape[-1]
      input_no_time_l.output.placeholder = LayerBase.get_rec_initial_output(
        batch_dim=1, name="input_no_time_l", n_out=10, output=input_no_time_l.output, rec_layer=None)  # dummy
    with tf.variable_scope("prev_l1"):
      prev_l = InternalLayer(name="prev:l1", network=net, out_type={"dim": 10, "time_dim_axis": None})
      prev_l.rec_vars_outputs["state"] = RnnCellLayer.get_rec_initial_state(
        batch_dim=1, name="prev_l", n_out=10, unit="LSTMBlock")
      print("Previous time layer:", prev_l)
    with tf.variable_scope("l1"):
      l1 = RnnCellLayer(
        n_out=10, unit="LSTMBlock", network=net, name="l1", rec_previous_layer=prev_l, sources=[input_no_time_l])
      print("RnnCell layer (no time):", l1)
      print("RnnCell layer (no time) params:", l1.params)
      assert l1.output.time_dim_axis is None
      assert l1.output.batch_dim_axis == 0
      assert l1.output.dim == 10
      assert l1.output.shape == (10,)
    with tf.variable_scope("data"):
      input_l = SourceLayer(network=net, name="data")
      print("Input layer (with time-dim):", input_l)
      assert input_l.output.dim == input_no_time_l.output.dim
      assert input_l.output.shape == (None, input_l.output.dim)
      assert input_l.output.time_dim_axis == 1
      assert not input_l.output.sparse
    with tf.variable_scope("l2"):
      l2 = RnnCellLayer(
        n_out=10, unit="LSTMBlock", network=net, name="l2", sources=[input_l])
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
    config.update({
      "num_outputs": n_out,
      "num_inputs": n_in,
      "network": {
        "output": {
          "class": "rec",
          "unit": {
            # Recurrent subnet here, operate on a single time-step:
            "output": {
              "class": "linear",
              "from": ["prev:output", "data:source"],
              "activation": "relu",
              "n_out": n_out},
          },
          "n_out": n_out},
      }
    })
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])
    output_layer = network.get_default_output_layer(must_exist=True)
    assert isinstance(output_layer, RecLayer)
    cell = output_layer.cell
    from TFNetworkRecLayer import _SubnetworkRecCell
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
    input_np = [
      [[0.7, 0.1], [-0.3, -0.1], [0.2, -0.1]],
      [[1.0, -0.4], [-0.2, 0.3], [0.0, 0.0]]]
    input_np = numpy.array(input_np, dtype="float32")
    input_seq_lens = [3, 2]
    n_batch = len(input_seq_lens)
    assert_equal(input_np.shape, (n_batch, max(input_seq_lens), n_in))
    input_placeholder = network.extern_data.data["data"].placeholder
    input_seq_lens_placeholder = network.extern_data.data["data"].size_placeholder[0]
    output_np, output_seq_lens = session.run(
      (output_layer.output.get_placeholder_as_batch_major(), output_layer.output.get_sequence_lengths()),
      feed_dict={input_placeholder: input_np, input_seq_lens_placeholder: input_seq_lens})
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
    config.update({
      "num_outputs": n_out,
      "num_inputs": n_in,
      "network": {
        "output": {
          "class": "rec",
          "optimize_move_layers_out": False,  # We esp. want to test it perform a single step, for debugging.
          "unit": {
            # Recurrent subnet here, operate on a single time-step:
            "output": {
              "class": "subnetwork", "from": ["data:source"], "subnetwork": {
                # RnnCellLayer inside subnetwork
                "output": {
                  "class": "rnn_cell",
                  "unit": "BasicRNN",
                  "unit_opts": {"activation": tf.nn.relu},
                  "from": ["data"],
                  "n_out": n_out},
                },
              "n_out": n_out}
          },
          "n_out": n_out},
      }
    })
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
      feed_dict={input_placeholder: input_np, input_seq_lens_placeholder: input_seq_lens})
    assert_equal(list(output_seq_lens), input_seq_lens)
    assert_equal(output_np.shape, (n_batch, max(input_seq_lens), n_out))
    print("rnn_cell subnet output:")
    print(output_np)
    assert_almost_equal(output_np, output_calc_np)
    print("rnn_cell also fine.")


def check_reclayer_optimize_out(subnet_layer_dict, other_subnet_layers=None, shared_base_net=None, rtol=1e-5):
  """
  :param dict[str] subnet_layer_dict: opts for the output layer inside the rec-layer subnet
  :param dict[str,dict[str]] other_subnet_layers: other layers for the rec-layer subnet
  :param dict[str,dict[str]] shared_base_net:
  :param float rtol: for the final comparison check
  """
  subnet_layer_dict = subnet_layer_dict.copy()
  n_in = 13
  n_out = subnet_layer_dict.get("n_out", 17)
  n_batch = 5
  n_time = 7
  subnet_layer_dict["n_out"] = n_out
  subnet_layer_dict.setdefault("from", ["data:source"])
  rec_layer_dict = {
    "class": "rec",
    "from": ["data"],
    "unit": {"output": subnet_layer_dict},
    "n_out": n_out,
    "is_output_layer": True
  }
  if other_subnet_layers:
    assert "output" not in other_subnet_layers
    rec_layer_dict["unit"].update(other_subnet_layers)
  config = Config({
    "debug_print_layer_output_template": True,
    "num_inputs": n_in,
    "num_outputs": n_out
  })
  from TFNetworkRecLayer import _SubnetworkRecCell
  with make_scope() as session:
    print("Create non-optimized rec layer (with subnet layer moved out)")
    rec_layer_dict["optimize_move_layers_out"] = False
    net1 = TFNetwork(config=config, train_flag=True, name="<root_not_opt>")
    if shared_base_net:
      net1.construct_from_dict(shared_base_net)
      for key in shared_base_net:
        assert key in net1.layers
    net1.construct_from_dict({"output_not_opt": rec_layer_dict})
    rec_layer_dict["optimize_move_layers_out"] = True
    print("Create optimized rec layer (with subnet layer inside loop)")
    net2 = TFNetwork(config=config, extern_data=net1.extern_data, train_flag=True, name="<root_opt>")
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
    assert_equal(set(net2_subnet.output_layers_moved_out), {"output"}.union(set(other_subnet_layers or [])))
    assert_equal([
      v.name.split("/")[1:] for v in net1.get_params_list()], [v.name.split("/")[1:] for v in net2.get_params_list()])
    net1.initialize_params(session=session)
    net1_params = net1.layers["output_not_opt"].get_param_values_dict(session=session)
    net2.layers["output_opt"].set_param_values_by_dict(values_dict=net1_params, session=session)
    x_np = net1.random.normal(size=(n_batch, n_time, n_in))
    net1_output = net1.layers["output_not_opt"].output.get_placeholder_as_batch_major()
    net2_output = net2.layers["output_opt"].output.get_placeholder_as_batch_major()
    feed_dict = {
      net1.extern_data.data["data"].placeholder: x_np,
      net1.extern_data.data["data"].size_placeholder[0]: [n_time] * n_batch}
    y1_np = session.run(net1_output, feed_dict=feed_dict)
    print("y: (shape %r)" % (y1_np.shape,))
    print(y1_np)
    y2_np = session.run(net2_output, feed_dict=feed_dict)
    assert_equal(y1_np.shape, (n_batch, n_time, n_out))
    assert_equal(y2_np.shape, (n_batch, n_time, n_out))
    assert y1_np.any() and y2_np.any()
    if not numpy.allclose(y1_np, y2_np, rtol=rtol):
      print("Not equal!")
      for b in range(n_batch):
        for t in range(n_time):
          for d in range(n_out):
            assert_allclose(y1_np[b, t, d], y2_np[b, t, d], rtol=rtol)
      assert_allclose(y1_np, y2_np, rtol=rtol)


def test_reclayer_optimize_out_linear():
  check_reclayer_optimize_out({"class": "linear", "activation": "relu"})


def test_reclayer_optimize_out_rnncell():
  check_reclayer_optimize_out({"class": "rnn_cell", "unit": "BasicLSTM"})


def test_reclayer_optimize_out_rec_nativelstm2():
  check_reclayer_optimize_out({"class": "rec", "unit": "NativeLstm2"})


def test_reclayer_optimize_out_selfatt_left():
  check_reclayer_optimize_out({
    "class": "self_attention", "attention_left_only": True, "num_heads": 2, "total_key_dim": 6, "n_out": 18})


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
      "s": {"class": "linear", "activation": None, "with_bias": False, "from": ["data:source"],
            "n_out": EncKeyTotalDim},  # (B, D)  -- Q (query). D should be same as enc_ctx
      "att_query": {"class": "split_dims", "axis": "F", "dims": (AttNumHeads, EncKeyPerHeadDim),
                    "from": ["s"]},  # (B, H, D/H)
      # Here is the main test, the dot-layer:
      "energy": {"class": "dot", "red1": -1, "red2": -1, "var1": "T", "var2": "T?",  # Note the "T?".
                 "from": ["base:enc_ctx", "att_query"]},
      # energy inside the loop will be (B, H, enc-T, 1).
      # energy outside the loop will be (B, H, enc-T, dec-T). I.e. enc-T is still the first time axis.
      "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, enc-T, H, 1)
      "att0": {"class": "generic_attention", "weights": "att_weights", "base": "base:enc_value"},  # (B, H, V)
      "att": {"class": "merge_dims", "axes": "static", "from": ["att0"]},  # (B, H*V); Use "static" here.
      },
    shared_base_net={
      "encoder": {"class": "copy", "from": ["data"]},
      "enc_ctx0": {"class": "linear", "activation": None, "with_bias": False, "from": ["encoder"],
                   "n_out": EncKeyTotalDim},  # (B, enc-T, D)
      "enc_ctx": {"class": "split_dims", "axis": "F", "dims": (AttNumHeads, EncKeyPerHeadDim),
                  "from": ["enc_ctx0"], "is_output_layer": True},  # (B, enc-T, H, D/H)
      "enc_value0": {"class": "linear", "activation": None, "with_bias": False, "from": ["encoder"],
                     "n_out": EncValueTotalDim},
      "enc_value": {"class": "split_dims", "axis": "F", "dims": (AttNumHeads, EncValuePerHeadDim),
                    "from": ["enc_value0"], "is_output_layer": True},  # (B, enc-T, H, D/H)
    },
    rtol=1e-3)


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
      "s": {"class": "linear", "activation": None, "with_bias": False, "from": ["data:source"],
            "n_out": EncKeyTotalDim},  # (B, D)  -- Q (query). D should be same as enc_ctx
      "att_query": {"class": "split_dims", "axis": "F", "dims": (AttNumHeads, EncKeyPerHeadDim),
                    "from": ["s"]},  # (B, H, D/H)
      "energy": {"class": "dot", "red1": -1, "red2": -1, "var1": "T", "var2": "T?",  # Note the "T?".
                 "from": ["base:enc_ctx", "att_query"]},
      # energy inside the loop will be (B, H, enc-T, 1).
      # energy outside the loop will be (B, H, enc-T, dec-T). I.e. enc-T is still the first time axis.
      "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, H, enc-T, 1)
      "slice": {"class": "slice", "from": "att_weights", "axis": "t", "slice_end": 1},  # (B, H, 1, 1)
      "squeeze0": {"class": "squeeze", "from": "slice", "axis": "t"},  # (B, H, 1)
      "squeeze": {"class": "squeeze", "from": "squeeze0", "axis": "auto", "allow_no_op": True},  # (B, H)
      },
    shared_base_net={
      "encoder": {"class": "copy", "from": ["data"]},
      "enc_ctx0": {"class": "linear", "activation": None, "with_bias": False, "from": ["encoder"],
                   "n_out": EncKeyTotalDim},  # (B, enc-T, D)
      "enc_ctx": {"class": "split_dims", "axis": "F", "dims": (AttNumHeads, EncKeyPerHeadDim),
                  "from": ["enc_ctx0"], "is_output_layer": True},  # (B, enc-T, H, D/H)
      "enc_value0": {"class": "linear", "activation": None, "with_bias": False, "from": ["encoder"],
                     "n_out": EncValueTotalDim},
      "enc_value": {"class": "split_dims", "axis": "F", "dims": (AttNumHeads, EncValuePerHeadDim),
                    "from": ["enc_value0"], "is_output_layer": True},  # (B, enc-T, H, D/H)
    },
    rtol=1e-3)


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
      "s": {"class": "linear", "activation": None, "with_bias": False, "from": ["data:source"],
            "n_out": EncKeyTotalDim},  # (B, D)  -- Q (query). D should be same as enc_ctx
      "att_query": {"class": "split_dims", "axis": "F", "dims": (AttNumHeads, EncKeyPerHeadDim),
                    "from": ["s"]},  # (B, H, D/H)
      "energy": {"class": "dot", "red1": -1, "red2": -1, "var1": "T?", "var2": "T",  # Note the "T?".
                 "from": ["att_query", "base:enc_ctx"]},
      # energy inside the loop will be (B, H, 1, enc-T).
      # energy outside the loop will be (B, H, dec-T, enc-T). I.e. dec-T is the first time axis.
      "att_weights": {"class": "softmax_over_spatial", "axis": "d:-1", "from": ["energy"]},  # (B, enc-T, H, 1)
      "slice": {"class": "slice", "from": "att_weights", "axis": "d:-1", "slice_end": 1},  # (B, 1, H, 1)
      "squeeze0": {"class": "squeeze", "from": "slice", "axis": "d:-1"},  # (B, H, 1)
      "squeeze": {"class": "squeeze", "from": "squeeze0", "axis": "auto", "allow_no_op": True},  # (B, H)
      },
    shared_base_net={
      "encoder": {"class": "copy", "from": ["data"]},
      "enc_ctx0": {"class": "linear", "activation": None, "with_bias": False, "from": ["encoder"],
                   "n_out": EncKeyTotalDim},  # (B, enc-T, D)
      "enc_ctx": {"class": "split_dims", "axis": "F", "dims": (AttNumHeads, EncKeyPerHeadDim),
                  "from": ["enc_ctx0"], "is_output_layer": True},  # (B, enc-T, H, D/H)
      "enc_value0": {"class": "linear", "activation": None, "with_bias": False, "from": ["encoder"],
                     "n_out": EncValueTotalDim},
      "enc_value": {"class": "split_dims", "axis": "F", "dims": (AttNumHeads, EncValuePerHeadDim),
                    "from": ["enc_value0"], "is_output_layer": True},  # (B, enc-T, H, D/H)
    },
    rtol=1e-3)


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
    config.update({
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
              "class": "eval", "from": ["prev:accum", "base:enc0", "base:enc1"],
              "out_type": {"dim": 1, "shape": (None, 1)},
              "eval": """(tf.Print(source(0), ["shape0", tf.shape(source(0))]) +
                          tf.Print(source(1), ["shape1", tf.shape(source(1))]) *
                          tf.Print(source(2), ["shape2", tf.shape(source(2))]))"""
            },
            "output": {
              "class": "reduce", "axis": "stag:encoder", "mode": "max", "from": "accum"},
          }},
      }
    })
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])
    session.run(tf.global_variables_initializer())
    output_layer = network.get_default_output_layer(must_exist=True)
    from test_TFNetworkLayer import make_feed_dict
    feed_dict = make_feed_dict(list(network.extern_data.data.values()))
    session.run(output_layer.output.placeholder, feed_dict=feed_dict)


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
    self.embed_weight = self.EncValueTotalDim ** 0.5

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
    d[output + '_self_att_laynorm'] = {"class": "layer_norm", "from": [inp]}
    d[output + '_self_att_att'] = {"class": "self_attention", "num_heads": self.AttNumHeads,
                                   "total_key_dim": self.EncKeyTotalDim,
                                   "n_out": self.EncValueTotalDim, "from": [output + '_self_att_laynorm'],
                                   "attention_left_only": False, "attention_dropout": self.attention_dropout,
                                   "forward_weights_init": self.ff_init}
    d[output + '_self_att_lin'] = {"class": "linear", "activation": None, "with_bias": False,
                                   "from": [output + '_self_att_att'], "n_out": self.EncValueTotalDim,
                                   "forward_weights_init": self.ff_init}
    d[output + '_self_att_drop'] = {"class": "dropout", "from": [output + '_self_att_lin'],
                                    "dropout": self.postprocess_dropout}
    d[output + '_self_att_out'] = {"class": "combine", "kind": "add", "from": [inp, output + '_self_att_drop'],
                                   "n_out": self.EncValueTotalDim}
    #####
    d[output + '_ff_laynorm'] = {"class": "layer_norm", "from": [output + '_self_att_out']}
    d[output + '_ff_conv1'] = {"class": "linear", "activation": "relu", "with_bias": True,
                               "from": [output + '_ff_laynorm'],
                               "n_out": self.FFDim, "forward_weights_init": self.ff_init}
    d[output + '_ff_conv2'] = {"class": "linear", "activation": None, "with_bias": True,
                               "from": [output + '_ff_conv1'], "dropout": self.act_dropout,
                               "n_out": self.EncValueTotalDim, "forward_weights_init": self.ff_init}
    d[output + '_ff_drop'] = {"class": "dropout", "from": [output + '_ff_conv2'], "dropout": self.postprocess_dropout}
    d[output + '_ff_out'] = {"class": "combine", "kind": "add",
                             "from": [output + '_self_att_out', output + '_ff_drop'],
                             "n_out": self.EncValueTotalDim}
    d[output] = {"class": "copy", "from": [output + '_ff_out']}

  def add_trafo_dec_layer(self, db, d, inp, output):
    """
    :param dict[str,dict[str]] db:
    :param dict[str,dict[str]] d:
    :param str inp:
    :param str output:
    """
    pre_inp = [inp]
    d[output + '_self_att_laynorm'] = {"class": "layer_norm", "from": pre_inp}
    d[output + '_self_att_att'] = {
      "class": "self_attention",
      "num_heads": self.AttNumHeads,
      "total_key_dim": self.EncKeyTotalDim,
      "n_out": self.EncValueTotalDim,
      "from": [output + '_self_att_laynorm'],
      "attention_left_only": True,
      "attention_dropout": self.attention_dropout,
      "forward_weights_init": self.ff_init}
    d[output + '_self_att_lin'] = {"class": "linear", "activation": None, "with_bias": False,
                                   "from": [output + '_self_att_att'], "n_out": self.EncValueTotalDim,
                                   "forward_weights_init": self.ff_init}
    d[output + '_self_att_drop'] = {"class": "dropout", "from": [output + '_self_att_lin'],
                                    "dropout": self.postprocess_dropout}
    d[output + '_self_att_out'] = {"class": "combine", "kind": "add", "from": [inp, output + '_self_att_drop'],
                                   "n_out": self.EncValueTotalDim}
    #####
    d[output + '_att_laynorm'] = {"class": "layer_norm", "from": [output + '_self_att_out']}
    d[output + '_att_query0'] = {"class": "linear", "activation": None, "with_bias": False,
                                 "from": [output + '_att_laynorm'],
                                 "n_out": self.EncValueTotalDim, "forward_weights_init": self.ff_init}
    d[output + '_att_query'] = {"class": "split_dims", "axis": "F", "dims": (self.AttNumHeads, self.EncKeyPerHeadDim),
                                "from": [output + '_att_query0']}  # (B, H, D/H)
    db[output + '_att_key0'] = {"class": "linear", "activation": None, "with_bias": False, "from": ["encoder"],
                                "n_out": self.EncKeyTotalDim, "forward_weights_init": self.ff_init}  # (B, enc-T, D)
    db[output + '_att_value0'] = {"class": "linear", "activation": None, "with_bias": False, "from": ["encoder"],
                                  "n_out": self.EncValueTotalDim, "forward_weights_init": self.ff_init}
    db[output + '_att_key'] = {"class": "split_dims", "axis": "F", "dims": (self.AttNumHeads, self.EncKeyPerHeadDim),
                               "from": [output + '_att_key0']}  # (B, enc-T, H, D/H)
    db[output + '_att_value'] = {"class": "split_dims", "axis": "F",
                                 "dims": (self.AttNumHeads, self.EncValuePerHeadDim),
                                 "from": [output + '_att_value0']}  # (B, enc-T, H, D'/H)
    d[output + '_att_energy'] = {"class": "dot", "red1": -1, "red2": -1, "var1": "T", "var2": "T?",
                                 "from": ['base:' + output + '_att_key', output + '_att_query']}  # (B, H, enc-T, 1)
    d[output + '_att_weights'] = {"class": "softmax_over_spatial", "from": [output + '_att_energy'],
                                  "energy_factor": self.EncKeyPerHeadDim ** -0.5}  # (B, enc-T, H, 1)

    d[output + '_att_weights_drop'] = {"class": "dropout", "dropout_noise_shape": {"*": None},
                                       "from": [output + '_att_weights'], "dropout": self.attention_dropout}

    d[output + '_att0'] = {"class": "generic_attention", "weights": output + '_att_weights_drop',
                           "base": 'base:' + output + '_att_value'}  # (B, H, V)
    d[output + '_att_att'] = {"class": "merge_dims", "axes": "static",
                              "from": [output + '_att0']}  # (B, H*V) except_batch
    d[output + '_att_lin'] = {"class": "linear", "activation": None, "with_bias": False,
                              "from": [output + '_att_att'],
                              "n_out": self.EncValueTotalDim, "forward_weights_init": self.ff_init}
    d[output + '_att_drop'] = {"class": "dropout", "from": [output + '_att_lin'], "dropout": self.postprocess_dropout}
    d[output + '_att_out'] = {"class": "combine", "kind": "add",
                              "from": [output + '_self_att_out', output + '_att_drop'],
                              "n_out": self.EncValueTotalDim}
    #####
    d[output + '_ff_laynorm'] = {"class": "layer_norm", "from": [output + '_att_out']}
    d[output + '_ff_conv1'] = {"class": "linear", "activation": "relu", "with_bias": True,
                               "from": [output + '_ff_laynorm'],
                               "n_out": self.FFDim, "forward_weights_init": self.ff_init}
    d[output + '_ff_conv2'] = {"class": "linear", "activation": None, "with_bias": True,
                               "from": [output + '_ff_conv1'], "dropout": self.act_dropout,
                               "n_out": self.EncValueTotalDim, "forward_weights_init": self.ff_init}
    d[output + '_ff_drop'] = {"class": "dropout", "from": [output + '_ff_conv2'], "dropout": self.postprocess_dropout}
    d[output + '_ff_out'] = {"class": "combine", "kind": "add", "from": [output + '_att_out', output + '_ff_drop'],
                             "n_out": self.EncValueTotalDim}
    d[output] = {"class": "copy", "from": [output + '_ff_out']}

  def build(self):
    network = {
      "source_embed_raw": {"class": "linear", "activation": None, "with_bias": False, "n_out": self.EncValueTotalDim,
                           "forward_weights_init": self.ff_init},
      "source_embed_weighted": {"class": "eval", "from": ["source_embed_raw"],
                                "eval": "source(0) * %f" % self.embed_weight},
      "source_embed_with_pos": {"class": "positional_encoding", "add_to_input": True,
                                "from": ["source_embed_weighted"]},
      "source_embed": {"class": "dropout", "from": ["source_embed_with_pos"], "dropout": self.embed_dropout},

      # encoder stack is added by separate function
      "encoder": {"class": "layer_norm", "from": ["enc_%02d" % self.encN]},

      "output": {"class": "rec", "from": [], "unit": {
        'output': {'class': 'choice', 'target': 'classes', 'beam_size': 12, 'from': ["output_prob"],
                   "initial_output": 0},  # this is a vocab_id, make this flexible
        "end": {"class": "compare", "from": ["output"], "value": 0},
        'target_embed_raw': {'class': 'linear', 'activation': None, "with_bias": False, 'from': ['prev:output'],
                             "n_out": self.EncValueTotalDim, "forward_weights_init": self.ff_init},
        # there seems to be no <s> in t2t, they seem to use just the zero vector
        "target_embed_weighted": {"class": "eval", "from": ["target_embed_raw"],
                                  "eval": "source(0) * %f" % self.embed_weight},
        "target_embed_with_pos": {"class": "positional_encoding", "add_to_input": True,
                                  "from": ["target_embed_weighted"]},
        "target_embed": {"class": "dropout", "from": ["target_embed_with_pos"], "dropout": self.embed_dropout},

        # decoder stack is added by separate function
        "decoder": {"class": "layer_norm", "from": ["dec_%02d" % self.decN]},

        "output_prob": {
          "class": "softmax", "from": ["decoder"], "dropout": 0.0,
          "target": "classes", "loss": "ce", "loss_opts": {"label_smoothing": self.label_smoothing},
          "with_bias": False, "forward_weights_init": self.ff_init,
          "is_output_layer": True
        }

      }, "target": "classes", "max_seq_len": "max_len_from('base:encoder') * 3"},

      "decision": {
        "class": "decide", "from": ["output"], "loss": "edit_distance", "target": "classes",
        "loss_opts": {
          # "debug_print": True
        }
      }

    }

    self.add_trafo_enc_layer(network, "source_embed", "enc_01")
    for n in range(1, self.encN):
      self.add_trafo_enc_layer(network, "enc_%02d" % n, "enc_%02d" % (n + 1))

    self.add_trafo_dec_layer(network, network["output"]["unit"], "target_embed", "dec_01")
    for n in range(1, self.decN):
      self.add_trafo_dec_layer(network, network["output"]["unit"], "dec_%02d" % n, "dec_%02d" % (n + 1))

    return network


def test_reclayer_optimize_out_transformer():
  from TFNetworkRecLayer import _SubnetworkRecCell
  n_src_dim = 5
  n_tgt_dim = 7

  def get_config(optimize_out):
    """
    :param bool optimize_out:
    :rtype: Config
    """
    return Config({
      "debug_print_layer_output_template": True,
      "debug_print_layer_output_shape": True,  # only for debugging
      "extern_data": {
        "data": {"dim": n_src_dim, "sparse": True},
        "classes": {"dim": n_tgt_dim, "sparse": True, "available_for_inference": False}},
      "network": TransformerNetwork().build(),
      "optimize_move_layers_out": optimize_out
    })

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
      extern_data.data["data"].placeholder: data_np,
      extern_data.data["data"].size_placeholder[0]: n_enc_times,
      extern_data.data["classes"].placeholder: classes_np,
      extern_data.data["classes"].size_placeholder[0]: n_dec_times}

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
      out_np = session.run(out.placeholder, feed_dict=get_feed_dict(extern_data=net.extern_data))
      return out_np

  out_opt_np = get_out(optimize_out=True)
  out_nopt_np = get_out(optimize_out=False)
  print("output:")
  print(out_opt_np)
  numpy.testing.assert_almost_equal(out_opt_np, out_nopt_np, decimal=5)
  print("Both are equal!")


def test_reclayer_move_out_input_train_and_search():
  from TFNetworkRecLayer import _SubnetworkRecCell
  n_src_dim = 5
  n_tgt_dim = 7
  beam_size = 12

  def make_extern_data():
    return ExternData({
      "data": {"dim": n_src_dim, "sparse": True},
      "classes": {"dim": n_tgt_dim, "sparse": True, "available_for_inference": False}})

  config = Config()
  config.update({
    "debug_print_layer_output_template": True,
    "network": {
      "encoder": {"class": "linear", "activation": "tanh", "n_out": 5},

      "output": {"class": "rec", "from": [], "unit": {

        'target_embed_raw': {'activation': None,
                             'class': 'linear',
                             'from': ['prev:output'],
                             'n_out': 13,
                             'with_bias': False},
        # In train, this is in output_layers_moved_out (like all layers).
        # In search, this is in input_layers_moved_out.
        'encoder_int': {'activation': None,
                        'class': 'linear',
                        'from': ['base:encoder'],
                        'n_out': 11,
                        'with_bias': False},
        "encoder_reduced": {"class": "reduce", "mode": "sum", "axis": "T", "from": ["encoder_int"]},

        "output_prob": {"class": "softmax", "from": ["target_embed_raw", "encoder_reduced"],
                        "target": "classes", "loss": "ce"},
        'output': {'class': 'choice', 'target': 'classes', 'beam_size': beam_size, 'from': ["output_prob"],
                   "initial_output": 0},
        "end": {"class": "compare", "from": ["output"], "value": 0},

      }, "target": "classes", "max_seq_len": 20},

      "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance", "target": "classes"}
    }})

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
      cell.output_layers_moved_out, ['output_prob', 'target_embed_raw', 'output', 'encoder_reduced', 'encoder_int'])

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


def test_subnet_load_on_init_rec():
  import tempfile
  model_tmp_dir = tempfile.mkdtemp("tmp-checkpoint")
  model_filename = model_tmp_dir + "/model"
  with make_scope() as session:
    config = Config()
    n_in, n_hidden, n_out = 2, 5, 3
    config.update({
      "num_outputs": n_out,
      "num_inputs": n_in,
      "network": {
        "input": {"class": "linear", "n_out": n_hidden, "activation": "identity",
                  "forward_weights_init": "random_normal_initializer(mean=0.0, stddev=1.0)"},
        "lstm0": {"class": "rec", "unit": "lstm",
                  "forward_weights_init": "random_normal_initializer(mean=0.0, stddev=1.0)",
                  "recurrent_weights_init": "random_normal_initializer(mean=0.0, stddev=1.0)",
                  "bias_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
                  "n_out": n_hidden, "direction": 1, "from": ["input"]},
        "lstm1": {"class": "rec", "unit": "lstm",
                  "forward_weights_init": "random_normal_initializer(mean=0.0, stddev=1.0)",
                  "recurrent_weights_init": "random_normal_initializer(mean=0.0, stddev=1.0)",
                  "bias_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
                  "n_out": n_hidden, "direction": 1, "from": ["lstm0"]},
        "output": {"class": "linear", "activation": "identity",
                   "forward_weights_init": "random_normal_initializer(mean=0.0, stddev=1.0)",
                   "bias_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
                   "n_out": n_out, "from": ["lstm1"]}
      }
    })
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])
    network.initialize_params(session)
    params_orig_dump = network.get_params_serialized(session)
    print("lstm0:")
    print(params_orig_dump.values_dict["lstm0"]["W"])
    assert(params_orig_dump.values_dict["lstm0"]["W"].any())
    network.save_params_to_file(filename=model_filename, session=session)

    # Simple forward.
    input_np = [
      [[0.7, 0.1], [-0.3, -0.1], [0.2, -0.1]],
      [[1.0, -0.4], [-0.2, 0.3], [0.0, 0.0]]]
    input_np = numpy.array(input_np, dtype="float32")
    input_seq_lens = [3, 2]
    n_batch = len(input_seq_lens)
    assert_equal(input_np.shape, (n_batch, max(input_seq_lens), n_in))
    input_placeholder = network.extern_data.data["data"].placeholder
    input_seq_lens_placeholder = network.extern_data.data["data"].size_placeholder[0]
    output_layer = network.get_default_output_layer(must_exist=True)
    output_orig_np, output_seq_lens = session.run(
      (output_layer.output.get_placeholder_as_batch_major(), output_layer.output.get_sequence_lengths()),
      feed_dict={input_placeholder: input_np, input_seq_lens_placeholder: input_seq_lens})
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
    config.update({
      "num_outputs": n_out,
      "num_inputs": n_in,
      "network": {
        "output": {
          "class": "rec",
          "optimize_move_layers_out": False,  # We esp. want to test it perform a single step, for debugging.
          "unit": {
            # Recurrent subnet here, operate on a single time-step:
            "output": {
              "class": "subnetwork",
              "from": ["data:source"],
              # Note: This has to convert the params into the right format.
              "load_on_init": model_filename,
              "subnetwork": {
                "input": {"class": "linear", "n_out": n_hidden, "activation": "identity"},
                "lstm0": {"class": "rnn_cell", "unit": "LSTMBlock", "unit_opts": {"forget_bias": 0.0},
                          "n_out": n_hidden, "from": ["input"]},
                "lstm1": {"class": "rnn_cell", "unit": "LSTMBlock", "unit_opts": {"forget_bias": 0.0},
                          "n_out": n_hidden, "from": ["lstm0"]},
                "output": {"class": "linear", "activation": "identity", "n_out": n_out, "from": ["lstm1"]}
              },
              "n_out": n_out},
          },
          "n_out": n_out},
      }
    })
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
      feed_dict={input_placeholder: input_np, input_seq_lens_placeholder: input_seq_lens})
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


def test_KenLmStateLayer():
  import TFKenLM
  TFKenLM.get_tf_mod(verbose=True)
  test_lm_file = TFKenLM.kenlm_dir + "/lm/test.arpa"
  assert os.path.exists(test_lm_file)
  from GeneratingDataset import Vocabulary
  from TFNetworkLayer import InternalLayer
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
      net.extern_data.register_data(Data(
        name="data", shape=(), time_dim_axis=None, dim=len(labels), sparse=True,
        auto_create_placeholders=True))
      data_layer = net.construct_layer(name="data", net_dict={})
      layer_base_opts = dict(name="output", network=net, sources=[data_layer])
      layer_out = KenLmStateLayer.get_out_data_from_opts(**layer_base_opts)
      rec_state = session.run(
        KenLmStateLayer.get_rec_initial_extra_outputs(batch_dim=1, rec_layer=None, **layer_base_opts))
      print("initial recurrent state:", rec_state)
      assert isinstance(rec_state, dict)
      prev_layer = InternalLayer(name="prev:%s" % layer_base_opts["name"], network=net, output=layer_out.copy())
      prev_layer.rec_vars_outputs = {
        k: tf.placeholder(name="prev_layer_%s" % k, shape=v.shape, dtype=v.dtype) for (k, v) in rec_state.items()}
      with reuse_name_scope(KenLmStateLayer.cls_get_tf_scope_name(layer_base_opts["name"])):
        layer = KenLmStateLayer(
          lm_file=test_lm_file,
          vocab_file=tmp_bpe_vocab_file.name, vocab_unknown_label="<unk>",
          bpe_merge_symbol="@@",
          output=layer_out, rec_previous_layer=prev_layer, **layer_base_opts)
        net.layers[layer.name] = layer

      print("Init.")
      net.initialize_params(session=session)

      print("Ref score.")
      input_word_ids = [labels.index(w) for w in "be@@ yond imm@@ edi@@ ate conc@@ erns </s>".split()]
      ref_score_str_placeholder = tf.placeholder(tf.string, shape=(), name="ref_score_str_placeholder")
      tf_ref_score = TFKenLM.ken_lm_abs_score_strings(handle=layer.lm_handle, strings=ref_score_str_placeholder)
      ref_score = session.run(tf_ref_score, feed_dict={ref_score_str_placeholder: "beyond immediate concerns </s>"})
      print("ref score:", ref_score)
      assert_almost_equal(ref_score, -9.251298)  # example from :func:`test_kenlm`

      print("Loop over %r." % ([labels[i] for i in input_word_ids],))
      abs_score = 0.0
      for i, word_id in enumerate(input_word_ids):
        print("input %i, word-idx %i, word %r" % (i, word_id, labels[word_id]))
        feed_dict = {net.extern_data.data["data"].placeholder: [word_id]}
        feed_dict.update({prev_layer.rec_vars_outputs[p]: v for (p, v) in rec_state.items()})
        rel_score_res, rec_state = session.run(
          (layer.output.placeholder, layer.rec_vars_outputs), feed_dict=feed_dict)
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
  import TFKenLM
  TFKenLM.get_tf_mod(verbose=True)
  test_lm_file = TFKenLM.kenlm_dir + "/lm/test.arpa"
  assert os.path.exists(test_lm_file)
  from GeneratingDataset import Vocabulary
  from TFNetworkLayer import InternalLayer
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
      net.extern_data.register_data(Data(
        name="data", shape=(), time_dim_axis=None, dim=len(labels), sparse=True,
        auto_create_placeholders=True))
      data_layer = net.construct_layer(name="data", net_dict={})
      layer_base_opts = dict(
        name="output", network=net, sources=[data_layer],
        lm_file=test_lm_file,
        vocab_file=tmp_bpe_vocab_file.name, vocab_unknown_label="<unk>",
        bpe_merge_symbol="@@",
        input_step_offset=1,
        dense_output=True)
      layer_out = KenLmStateLayer.get_out_data_from_opts(**layer_base_opts)
      batch_dim = 1
      rec_state = session.run(
        KenLmStateLayer.get_rec_initial_extra_outputs(batch_dim=batch_dim, rec_layer=None, **layer_base_opts))
      print("initial recurrent state:", rec_state)
      assert isinstance(rec_state, dict)
      prev_layer = InternalLayer(name="prev:%s" % layer_base_opts["name"], network=net, output=layer_out.copy())
      prev_layer.rec_vars_outputs = {
        k: tf.placeholder(name="prev_layer_%s" % k, shape=v.shape, dtype=v.dtype) for (k, v) in rec_state.items()}
      with reuse_name_scope(KenLmStateLayer.cls_get_tf_scope_name(layer_base_opts["name"])):
        layer = KenLmStateLayer(
          output=layer_out, rec_previous_layer=prev_layer, **layer_base_opts)
        net.layers[layer.name] = layer

      print("Init.")
      net.initialize_params(session=session)

      print("Ref score.")
      input_word_ids = [labels.index(w) for w in "be@@ yond imm@@ edi@@ ate conc@@ erns </s>".split()]
      ref_score_str_placeholder = tf.placeholder(tf.string, shape=(), name="ref_score_str_placeholder")
      tf_ref_score = TFKenLM.ken_lm_abs_score_strings(handle=layer.lm_handle, strings=ref_score_str_placeholder)
      ref_score = session.run(tf_ref_score, feed_dict={ref_score_str_placeholder: "beyond immediate concerns </s>"})
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
        print("input %i, word-idx %i, word %r, next-word-idx %i, next-word %r" % (
          i, word_id, word, next_word_id, next_word))
        feed_dict = {net.extern_data.data["data"].placeholder: [word_id]}
        feed_dict.update({prev_layer.rec_vars_outputs[p]: v for (p, v) in rec_state.items()})
        rel_score_res, rec_state = session.run(
          (layer.output.placeholder, layer.rec_vars_outputs), feed_dict=feed_dict)
        print("  score rel res:", rel_score_res)
        print("  state:", rec_state)
        assert rel_score_res.shape == (batch_dim, layer.vocab.num_labels)
        abs_score += rel_score_res[0][next_word_id]
        print("  abs score:", abs_score)
        word_seq_so_far = (rec_state["state"][0].decode("utf8") + next_word).replace("@@ ", "").strip().split(" ")
        word_seq_so_far = ["<unk>" if "@@" in w else w for w in word_seq_so_far]
        res2 = session.run(tf_ref_score, feed_dict={ref_score_str_placeholder: " ".join(word_seq_so_far)})
        print("  word seq so far: %r" % (word_seq_so_far,), "score:", res2)
        assert_equal(res2, abs_score)

      assert_almost_equal(abs_score, ref_score)
      print("Scores are as expected.")


@unittest.skipIf(not is_gpu_available(), "no gpu on this system")
def test_BlocksparseLSTM_load_params_from_native_lstm():
  from TFNativeOp import have_blocksparse_requirements, init_blocksparse
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
    tf.set_random_seed(42)
    src_placeholder = tf.placeholder(tf.float32, (batch_dim, seq_len, num_inputs), name="src_placeholder")
    seq_len_placeholder = tf.placeholder(tf.int32, (batch_dim,), name="seq_len_placeholder")
    feed_dict = {
      src_placeholder: random.uniform(-1.0, 1.0, (batch_dim, seq_len, num_inputs)),
      seq_len_placeholder: [seq_len] * batch_dim
    }

    from TFUtil import xavier_initializer
    default_var_initializer = xavier_initializer(seed=13)
    with tf.variable_scope(tf.get_variable_scope(), initializer=default_var_initializer) as scope:
      net = TFNetwork(config=Config(), extern_data=ExternData(), train_flag=False)
      with net.register_network_scope():
        from TFNetworkLayer import InternalLayer
        src_layer = InternalLayer(name='src', network=net, output=Data(
          'src', shape=(None, num_inputs), placeholder=src_placeholder, size_placeholder={0: seq_len_placeholder}))
        print("source layer:", src_layer)
        with tf.name_scope("nativelstm"):
          layer1 = RecLayer(
            name='nativelstm', network=net,
            output=Data("out", shape=(None, num_outputs), time_dim_axis=0, batch_dim_axis=1), sources=[src_layer],
            unit='NativeLSTM2')
        with tf.name_scope("blocksparselstm"):
          layer2 = RecLayer(
            name='blocksparselstm', network=net,
            output=Data("out", shape=(None, num_outputs_sparse), time_dim_axis=0, batch_dim_axis=1),
            sources=[src_layer],
            unit='BlocksparseLSTM',
            unit_opts={'seed': 5, 'connectivity': 1, 'connectivity_dense': 2, 'layer_norm': False})
        y1 = layer1.output.get_placeholder_as_batch_major()
        y2 = layer2.output.get_placeholder_as_batch_major()

    print("run")
    session.run(tf.global_variables_initializer())
    native_lstm_params = layer1.get_param_values_dict(session=session)
    np_y1 = session.run(y1, feed_dict=feed_dict)
    assert np_y1.shape == (batch_dim, seq_len, num_outputs)
    print('native output:')
    print(np_y1)
    bsmm_cell = layer2.cell
    assert isinstance(bsmm_cell, BlocksparseLSTMCell)
    for param in layer2.params.values():
      print('blocksparse LSTM param:', param)
      assert isinstance(param, tf.Variable)
      param.load(numpy.zeros(param.get_shape().as_list(), dtype='float32'), session=session)
    bsmm_cell.load_params_from_native_lstm(native_lstm_params, session=session)
    np_y2 = session.run(y2, feed_dict=feed_dict)
    assert np_y2.shape == (batch_dim, seq_len, num_outputs_sparse)
    np_y2 = np_y2[:, :, :num_outputs]
    assert_almost_equal(np_y1, np_y2)


def test_rec_layer_search_select_src_reuse_layer():
  from TFNetworkRecLayer import _SubnetworkRecCell
  n_src_dim = 5
  n_tgt_dim = 7
  beam_size = 12
  config = Config()
  config.update({"debug_print_layer_output_template": True, "optimize_move_layers_out": False})

  def get_net_dict():
    return {
      "source_embed": {"class": "linear", "activation": None, "with_bias": False, "n_out": 6},

      "lstm0_fw": {"class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                   "initial_state": "var", "n_out": 10, "direction": 1, "from": ["source_embed"]},
      "lstm0_bw": {"class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                   "initial_state": "var", "n_out": 10, "direction": -1, "from": ["source_embed"]},

      "lstm1_fw": {"class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                   "initial_state": "var", "n_out": 10, "direction": 1, "from": ["lstm0_fw", "lstm0_bw"]},
      "lstm1_bw": {"class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                   "initial_state": "var", "n_out": 10, "direction": -1, "from": ["lstm0_fw", "lstm0_bw"]},

      "encoder": {"class": "copy", "from": ["lstm1_fw", "lstm1_bw"]},
      "enc_ctx": {"class": "linear", "activation": None, "with_bias": True, "from": ["encoder"], "n_out": 10},
      "fertility": {"class": "linear", "activation": "sigmoid", "with_bias": False, "from": ["encoder"], "n_out": 1},

      "output": {"class": "rec", "from": [], "unit": {
        'output': {'class': 'choice', 'target': 'classes', 'beam_size': beam_size, 'from': ["output_prob"],
                   "initial_output": 0},
        "end": {"class": "compare", "from": ["output"], "value": 0},
        'target_embed': {'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'], "n_out": 6,
                         "initial_output": "apply(0)", 'reuse_params': {'map' : {'W' :{'reuse_layer': 'base:source_embed'}, 'b':None}}},
        "weight_feedback": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev:accum_att_weights"],
                            "n_out": 10},
        "prev_s_state": {"class": "get_last_hidden_state", "from": ["prev:s"], "n_out": 20},
        "prev_s_transformed": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev_s_state"],
                               "n_out": 10},
        "energy_in": {"class": "combine", "kind": "add",
                      "from": ["base:enc_ctx", "weight_feedback", "prev_s_transformed"], "n_out": 10},
        "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
        "energy": {"class": "linear", "activation": None, "with_bias": False, "from": ["energy_tanh"], "n_out": 1},
        "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, enc-T, 1)
        "accum_att_weights": {"class": "eval", "from": ["prev:accum_att_weights", "att_weights", "base:fertility"],
                              "eval": "source(0) + source(1) / (2.0 * source(2))",
                              "out_type": {"dim": 1, "shape": (None, 1)}},
        "att": {"class": "generic_attention", "weights": "att_weights", "base": "base:encoder"},
        "s": {"class": "rnn_cell", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
              "initial_state": "var", "from": ["target_embed", "att"], "n_out": 10},
        "readout_in": {"class": "linear", "from": ["prev:s", "prev:target_embed", "att"], "activation": None,
                       "n_out": 10},
        "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
        "output_prob": {"class": "softmax", "from": ["readout"], "target": "classes", "loss": "ce"}
      }, "target": "classes", "max_seq_len": 20},

      "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance", "target": "classes"}
    }

  print("Constructing search network.")
  with make_scope() as session:
    extern_data = ExternData({
      "data": {"dim": n_src_dim, "sparse": True},
      "classes": {"dim": n_tgt_dim, "sparse": True, "available_for_inference": False}})
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
      network[name] = {"class": "rec", "unit": "lstmp", "n_out": lstm_dim, "dropout": 0.1, "L2": 0.01, "direction": 1,
                       "from": src}
      return name
    network["%s_win" % name] = {"class": "window", "window_size": lstm_window, "window_right": lstm_window - 1,
                                "from": src}  # (B,T,W,D)
    network["%s_mdims" % name] = {"class": "merge_dims", "axes": "BT", "from": ["%s_win" % name]}  # (B*T,W,D)
    network["%s_rdims" % name] = {"class": "reinterpret_data", "enforce_batch_major": True, "set_axes": {"T": 1},
                                  "from": ["%s_mdims" % name]}  # (B*T,W,D)
    network["%s_rec" % name] = {
      "class": "rec", "unit": "lstmp", "n_out": lstm_dim, "dropout": 0.1, "L2": 0.01, "direction": -1,
      "from": ["%s_rdims" % name]}  # (B*T,W,D')
    network["%s_cur" % name] = {"class": "slice", "axis": "T", "slice_end": 1, "from": ["%s_rec" % name]}  # (B*T,1,D')
    network["%s_cursq" % name] = {"class": "squeeze", "axis": "T", "from": ["%s_cur" % name]}  # (B*T,D')
    network["%s_res" % name] = {"class": "split_batch_time", "base": src[0], "from": ["%s_cursq" % name]}  # (B,T,D')
    return "%s_res" % name

  num_layers = 6
  src = ["data"]
  for i in range(num_layers):
    fwd = add_lstm(i, 1, src)
    bwd = add_lstm(i, -1, src)
    src = [fwd, bwd]
  # Focal Loss, https://arxiv.org/abs/1708.02002
  network["output"] = {"class": "softmax", "loss": "ce", "loss_opts": {"focal_loss_factor": 2.0}, "from": src}
  config = Config({
    "num_inputs": 3,
    "num_outputs": 7
  })
  with make_scope() as session:
    net = TFNetwork(config=config, train_flag=True)
    net.construct_from_dict(network)


def test_GenericAttentionLayer_basic0():
  from TFNetworkLayer import InternalLayer
  net = TFNetwork(extern_data=ExternData(), config=Config({"debug_print_layer_output_template": True}))
  kwargs = dict(
    name="att", network=net,
    weights=InternalLayer(
      name="att_weights", network=net,
      output=Data(
        name='att_weights_output', shape=(None, 1), auto_create_placeholders=True)),
    base=InternalLayer(
      name="enc_value", network=net,
      output=Data(name='enc_value_output', shape=(None, 20), auto_create_placeholders=True)))
  print("GenericAttentionLayer kwargs:")
  pprint(kwargs)
  kwargs["output"] = GenericAttentionLayer.get_out_data_from_opts(**kwargs)
  layer = GenericAttentionLayer(**kwargs)
  layer.output.sanity_check()
  assert layer.output.shape == (20,) and not layer.output.have_time_axis()


def test_GenericAttentionLayer_basic():
  from TFNetworkLayer import InternalLayer
  net = TFNetwork(extern_data=ExternData(), config=Config({"debug_print_layer_output_template": True}))
  # This is a common situation when the GenericAttentionLayer is inside a recurrent loop,
  # and it gets the encoder values from outside ("base:enc_value" or so),
  # and the attention weights from inside the loop, and they have the same time dim axis as the encoder values.
  kwargs = dict(
    name="att", network=net,
    weights=InternalLayer(
      name="att_weights", network=net,
      output=Data(name='att_weights_output', shape=(None, 1), batch_dim_axis=1, auto_create_placeholders=True)),
    base=InternalLayer(
      name="enc_value", network=net,
      output=Data(name='enc_value_output', shape=(None, 1, 2048), batch_dim_axis=1, auto_create_placeholders=True)))
  kwargs["output"] = GenericAttentionLayer.get_out_data_from_opts(**kwargs)
  layer = GenericAttentionLayer(**kwargs)
  layer.output.sanity_check()
  assert layer.output.shape == (1, 2048) and not layer.output.have_time_axis()


def test_GenericAttentionLayer_basic_multi_head():
  from TFNetworkLayer import InternalLayer
  net = TFNetwork(extern_data=ExternData(), config=Config({"debug_print_layer_output_template": True}))
  num_heads = 8
  kwargs = dict(
    name="att", network=net,
    weights=InternalLayer(
      name="att_weights", network=net,
      output=Data(
        name='att_weights_output', shape=(None, num_heads), batch_dim_axis=1, auto_create_placeholders=True)),
    base=InternalLayer(
      name="enc_value", network=net,
      output=Data(
        name='enc_value_output', shape=(None, num_heads, 2048), batch_dim_axis=1, auto_create_placeholders=True)))
  kwargs["output"] = GenericAttentionLayer.get_out_data_from_opts(**kwargs)
  layer = GenericAttentionLayer(**kwargs)
  layer.output.sanity_check()
  assert layer.output.shape == (num_heads, 2048) and not layer.output.have_time_axis()


def test_GenericAttentionLayer_weights_auto_squeeze_time_end():
  # Example: weights (B,1,T), base (B,T,V)
  from TFNetworkLayer import InternalLayer
  net = TFNetwork(extern_data=ExternData(), config=Config({"debug_print_layer_output_template": True}))
  kwargs = dict(
    name="att", network=net,
    weights=InternalLayer(
      name="att_weights", network=net,
      output=Data(
        name='att_weights_output', shape=(1, None), time_dim_axis=2, auto_create_placeholders=True)),
    base=InternalLayer(
      name="enc_value", network=net,
      output=Data(name='enc_value_output', shape=(None, 2048), auto_create_placeholders=True)))
  print("GenericAttentionLayer kwargs:")
  pprint(kwargs)
  kwargs["output"] = GenericAttentionLayer.get_out_data_from_opts(**kwargs)
  layer = GenericAttentionLayer(**kwargs)
  layer.output.sanity_check()
  assert layer.output.shape == (2048,) and not layer.output.have_time_axis()


def test_GenericAttentionLayer_weights_static_time_axis():
  # Example: weights (B,1,W), base (B,W,V), where W: window_size (static)
  window_size = 10
  from TFNetworkLayer import InternalLayer
  net = TFNetwork(extern_data=ExternData(), config=Config({"debug_print_layer_output_template": True}))
  kwargs = dict(
    name="att", network=net,
    weights=InternalLayer(
      name="att_weights", network=net,
      output=Data(
        name='att_weights_output', shape=(1, 10), time_dim_axis=2, auto_create_placeholders=True)),
    base=InternalLayer(
      name="enc_value", network=net,
      output=Data(name='enc_value_output', shape=(10, 2048), time_dim_axis=1, auto_create_placeholders=True)))
  print("GenericAttentionLayer kwargs:")
  pprint(kwargs)
  kwargs["output"] = GenericAttentionLayer.get_out_data_from_opts(**kwargs)
  layer = GenericAttentionLayer(**kwargs)
  layer.output.sanity_check()
  assert layer.output.shape == (2048,) and not layer.output.have_time_axis()


def test_GenericAttentionLayer_weights_heads_time_end():
  # Example: weights (B,H,T), base (B,T,H,V)
  from TFNetworkLayer import InternalLayer
  net = TFNetwork(extern_data=ExternData(), config=Config({"debug_print_layer_output_template": True}))
  num_heads = 8
  kwargs = dict(
    name="att", network=net,
    weights=InternalLayer(
      name="att_weights", network=net,
      output=Data(
        name='att_weights_output', shape=(num_heads, None), time_dim_axis=2, auto_create_placeholders=True)),
    base=InternalLayer(
      name="enc_value", network=net,
      output=Data(name='enc_value_output', shape=(None, num_heads, 2048), auto_create_placeholders=True)))
  print("GenericAttentionLayer kwargs:")
  pprint(kwargs)
  kwargs["output"] = GenericAttentionLayer.get_out_data_from_opts(**kwargs)
  layer = GenericAttentionLayer(**kwargs)
  layer.output.sanity_check()
  assert layer.output.shape == (num_heads, 2048) and not layer.output.have_time_axis()


def test_GenericAttentionLayer_weights_heads_auto_squeeze_time_end():
  # Example: weights (B,H,1,T), base (B,T,H,V)
  from TFNetworkLayer import InternalLayer
  net = TFNetwork(extern_data=ExternData(), config=Config({"debug_print_layer_output_template": True}))
  num_heads = 8
  kwargs = dict(
    name="att", network=net,
    weights=InternalLayer(
      name="att_weights", network=net,
      output=Data(
        name='att_weights_output', shape=(num_heads, 1, None), time_dim_axis=3,
        auto_create_placeholders=True)),
    base=InternalLayer(
      name="enc_value", network=net,
      output=Data(name='enc_value_output', shape=(None, num_heads, 2048), auto_create_placeholders=True)))
  print("GenericAttentionLayer kwargs:")
  pprint(kwargs)
  kwargs["output"] = GenericAttentionLayer.get_out_data_from_opts(**kwargs)
  layer = GenericAttentionLayer(**kwargs)
  layer.output.sanity_check()
  assert layer.output.shape == (num_heads, 2048) and not layer.output.have_time_axis()


def test_GenericAttentionLayer_extra_spatial():
  from TFNetworkLayer import InternalLayer
  net = TFNetwork(extern_data=ExternData(), config=Config({"debug_print_layer_output_template": True}))
  # This is the situation when the GenericAttentionLayer is outside the recurrent loop,
  # and it gets some encoder values (with different time axis),
  # and the attention weights, which has two spatial axis, one of the decoder, and one of the encoder.
  dec_time = DimensionTag(kind=DimensionTag.Types.Spatial, description="dec time")
  enc_time = DimensionTag(kind=DimensionTag.Types.Spatial, description="enc time")
  kwargs = dict(
    name="att", network=net,
    weights=InternalLayer(
      name="att_weights", network=net,
      output=Data(
        name='att_weights_output', shape=(None, None, 1), auto_create_placeholders=True,
        same_dim_tags_as={"dyn:0": dec_time, "dyn:1": enc_time})),
    base=InternalLayer(
      name="enc_value", network=net,
      output=Data(
        name='enc_value_output', shape=(None, 1, 2048), batch_dim_axis=1, auto_create_placeholders=True,
        same_dim_tags_as={"t": enc_time})))
  print("GenericAttentionLayer kwargs:")
  pprint(kwargs)
  kwargs["output"] = GenericAttentionLayer.get_out_data_from_opts(**kwargs)
  layer = GenericAttentionLayer(**kwargs)
  layer.output.sanity_check()
  assert layer.output.shape == (1, None, 2048) and layer.output.have_time_axis()
  assert len(layer.output.size_placeholder) == 1
  assert list(layer.output.size_placeholder.values())[0] is layer.weights.output.size_placeholder[0]


def test_GenericAttentionLayer_extra_spatial_multi_head():
  from TFNetworkLayer import InternalLayer
  net = TFNetwork(extern_data=ExternData(), config=Config({"debug_print_layer_output_template": True}))
  dec_time = DimensionTag(kind=DimensionTag.Types.Spatial, description="dec time")
  enc_time = DimensionTag(kind=DimensionTag.Types.Spatial, description="enc time")
  num_heads = 8
  kwargs = dict(
    name="att", network=net,
    weights=InternalLayer(
      name="att_weights", network=net,
      output=Data(
        name='att_weights_output', shape=(None, None, num_heads), auto_create_placeholders=True,
        same_dim_tags_as={"dyn:0": dec_time, "dyn:1": enc_time})),
    base=InternalLayer(
      name="enc_value", network=net,
      output=Data(
        name='enc_value_output', shape=(None, num_heads, 2048), batch_dim_axis=1, auto_create_placeholders=True,
        same_dim_tags_as={"t": enc_time})))
  print("GenericAttentionLayer kwargs:")
  pprint(kwargs)
  kwargs["output"] = GenericAttentionLayer.get_out_data_from_opts(**kwargs)
  layer = GenericAttentionLayer(**kwargs)
  layer.output.sanity_check()
  assert layer.output.shape == (num_heads, None, 2048) and layer.output.have_time_axis()
  assert len(layer.output.size_placeholder) == 1
  assert list(layer.output.size_placeholder.values())[0] is layer.weights.output.size_placeholder[0]


def test_untrainable_sublayers():
  with make_scope() as session:
    config = Config()
    n_in, n_out = 2, 3
    net_dict = {
      "source_embed": {"class": "linear", "activation": None, "with_bias": False, "n_out": 6},

      "lstm0_fw": {"class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                   "initial_state": "var", "n_out": 10, "direction": 1, "from": ["source_embed"]},
      "lstm0_bw": {"class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                   "initial_state": "var", "n_out": 10, "direction": -1, "from": ["source_embed"]},

      "lstm1_fw": {"class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                   "initial_state": "var", "n_out": 10, "direction": 1, "from": ["lstm0_fw", "lstm0_bw"]},
      "lstm1_bw": {"class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                   "initial_state": "var", "n_out": 10, "direction": -1, "from": ["lstm0_fw", "lstm0_bw"]},

      "encoder": {"class": "copy", "from": ["lstm1_fw", "lstm1_bw"]},
      "enc_ctx": {"class": "linear", "activation": None, "with_bias": True, "from": ["encoder"], "n_out": 10},
      "fertility": {"class": "linear", "activation": "sigmoid", "with_bias": False, "from": ["encoder"], "n_out": 1},
      "output": {"class": "rec", "from": [], "unit": {
        'output': {'class': 'choice', 'target': 'classes', 'beam_size': 12, 'from': ["output_prob"],
                   "initial_output": 0},
        "end": {"class": "compare", "from": ["output"], "value": 0},
        'target_embed': {'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'], "n_out": 6,
                         "initial_output": "apply(0)", "trainable": False},
        "weight_feedback": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev:accum_att_weights"],
                            "n_out": 10},
        "prev_s_state": {"class": "get_last_hidden_state", "from": ["prev:s"], "n_out": 20},
        "prev_s_transformed": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev_s_state"],
                               "n_out": 10, "trainable": False},
        "energy_in": {"class": "combine", "kind": "add",
                      "from": ["base:enc_ctx", "weight_feedback", "prev_s_transformed"], "n_out": 10},
        "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
        "energy": {"class": "linear", "activation": None, "with_bias": False, "from": ["energy_tanh"], "n_out": 1},
        "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, enc-T, 1)
        "accum_att_weights": {"class": "eval", "from": ["prev:accum_att_weights", "att_weights", "base:fertility"],
                              "eval": "source(0) + source(1) / (2.0 * source(2))",
                              "out_type": {"dim": 1, "shape": (None, 1)}},
        "att": {"class": "generic_attention", "weights": "att_weights", "base": "base:encoder"},
        "s": {"class": "rnn_cell", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
              "initial_state": "var", "from": ["target_embed", "att"], "n_out": 10},
        "readout_in": {"class": "linear", "from": ["prev:s", "prev:target_embed", "att"], "activation": None,
                       "n_out": 10, "trainable": False},
        "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
        "output_prob": {"class": "softmax", "from": ["readout"], "target": "classes", "loss": "ce"}
      }, "target": "classes", "max_seq_len": 20},

      "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance", "target": "classes"}
    }
    config.update({"num_outputs": n_out,
      "num_inputs": n_in,
      "network": net_dict})
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])
    weight_input_layer_moved_out = network.layers["output"].params["target_embed/W"]
    assert(weight_input_layer_moved_out not in set(network.get_trainable_params()))

    weight_output_layer_moved_out = network.layers["output"].params["readout_in/W"]
    assert(weight_output_layer_moved_out not in set(network.get_trainable_params()))

    weight_internal = network.layers["output"].params["prev_s_transformed/W"]
    assert(weight_internal not in set(network.get_trainable_params()))


def test_untrainable_reclayer():
  with make_scope() as session:
    config = Config()
    n_in, n_out = 2, 3
    net_dict = {
      "source_embed": {"class": "linear", "activation": None, "with_bias": False, "n_out": 6},

      "lstm0_fw": {"class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                   "initial_state": "var", "n_out": 10, "direction": 1, "from": ["source_embed"]},
      "lstm0_bw": {"class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                   "initial_state": "var", "n_out": 10, "direction": -1, "from": ["source_embed"]},

      "lstm1_fw": {"class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                   "initial_state": "var", "n_out": 10, "direction": 1, "from": ["lstm0_fw", "lstm0_bw"]},
      "lstm1_bw": {"class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                   "initial_state": "var", "n_out": 10, "direction": -1, "from": ["lstm0_fw", "lstm0_bw"]},

      "encoder": {"class": "copy", "from": ["lstm1_fw", "lstm1_bw"]},
      "enc_ctx": {"class": "linear", "activation": None, "with_bias": True, "from": ["encoder"], "n_out": 10},
      "fertility": {"class": "linear", "activation": "sigmoid", "with_bias": False, "from": ["encoder"], "n_out": 1},
      "output": {"class": "rec", "from": [], "trainable": False, "unit": {
        'output': {'class': 'choice', 'target': 'classes', 'beam_size': 12, 'from': ["output_prob"],
                   "initial_output": 0},
        "end": {"class": "compare", "from": ["output"], "value": 0},
        'target_embed': {'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'], "n_out": 6,
                         "initial_output": "apply(0)", "trainable": True},
        "weight_feedback": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev:accum_att_weights"],
                            "n_out": 10},
        "prev_s_state": {"class": "get_last_hidden_state", "from": ["prev:s"], "n_out": 20},
        "prev_s_transformed": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev_s_state"],
                               "n_out": 10, "trainable": True},
        "energy_in": {"class": "combine", "kind": "add",
                      "from": ["base:enc_ctx", "weight_feedback", "prev_s_transformed"], "n_out": 10},
        "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
        "energy": {"class": "linear", "activation": None, "with_bias": False, "from": ["energy_tanh"], "n_out": 1},
        "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, enc-T, 1)
        "accum_att_weights": {"class": "eval", "from": ["prev:accum_att_weights", "att_weights", "base:fertility"],
                              "eval": "source(0) + source(1) / (2.0 * source(2))",
                              "out_type": {"dim": 1, "shape": (None, 1)}},
        "att": {"class": "generic_attention", "weights": "att_weights", "base": "base:encoder"},
        "s": {"class": "rnn_cell", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
              "initial_state": "var", "from": ["target_embed", "att"], "n_out": 10},
        "readout_in": {"class": "linear", "from": ["prev:s", "prev:target_embed", "att"], "activation": None,
                       "n_out": 10, "trainable": True},
        "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
        "output_prob": {"class": "softmax", "from": ["readout"], "target": "classes", "loss": "ce"}
      }, "target": "classes", "max_seq_len": 20},

      "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance", "target": "classes"}
    }
    config.update({"num_outputs": n_out,
      "num_inputs": n_in,
      "network": net_dict})
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])
    weight_input_layer_moved_out = network.layers["output"].params["target_embed/W"]
    assert(weight_input_layer_moved_out not in set(network.get_trainable_params()))

    weight_output_layer_moved_out = network.layers["output"].params["readout_in/W"]
    assert(weight_output_layer_moved_out not in set(network.get_trainable_params()))

    weight_internal = network.layers["output"].params["prev_s_transformed/W"]
    assert(weight_internal not in set(network.get_trainable_params()))


def test_trainable_sublayers():
  with make_scope() as session:
    config = Config()
    n_in, n_out = 2, 3
    net_dict = {
      "source_embed": {"class": "linear", "activation": None, "with_bias": False, "n_out": 6},

      "lstm0_fw": {"class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                   "initial_state": "var", "n_out": 10, "direction": 1, "from": ["source_embed"]},
      "lstm0_bw": {"class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                   "initial_state": "var", "n_out": 10, "direction": -1, "from": ["source_embed"]},

      "lstm1_fw": {"class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                   "initial_state": "var", "n_out": 10, "direction": 1, "from": ["lstm0_fw", "lstm0_bw"]},
      "lstm1_bw": {"class": "rec", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
                   "initial_state": "var", "n_out": 10, "direction": -1, "from": ["lstm0_fw", "lstm0_bw"]},

      "encoder": {"class": "copy", "from": ["lstm1_fw", "lstm1_bw"]},
      "enc_ctx": {"class": "linear", "activation": None, "with_bias": True, "from": ["encoder"], "n_out": 10},
      "fertility": {"class": "linear", "activation": "sigmoid", "with_bias": False, "from": ["encoder"], "n_out": 1},
      "output": {"class": "rec", "from": [], "unit": {
        'output': {'class': 'choice', 'target': 'classes', 'beam_size': 12, 'from': ["output_prob"],
                   "initial_output": 0},
        "end": {"class": "compare", "from": ["output"], "value": 0},
        'target_embed': {'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'], "n_out": 6,
                         "initial_output": "apply(0)"},
        "weight_feedback": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev:accum_att_weights"],
                            "n_out": 10},
        "prev_s_state": {"class": "get_last_hidden_state", "from": ["prev:s"], "n_out": 20},
        "prev_s_transformed": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev_s_state"],
                               "n_out": 10},
        "energy_in": {"class": "combine", "kind": "add",
                      "from": ["base:enc_ctx", "weight_feedback", "prev_s_transformed"], "n_out": 10},
        "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
        "energy": {"class": "linear", "activation": None, "with_bias": False, "from": ["energy_tanh"], "n_out": 1},
        "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, enc-T, 1)
        "accum_att_weights": {"class": "eval", "from": ["prev:accum_att_weights", "att_weights", "base:fertility"],
                              "eval": "source(0) + source(1) / (2.0 * source(2))",
                              "out_type": {"dim": 1, "shape": (None, 1)}},
        "att": {"class": "generic_attention", "weights": "att_weights", "base": "base:encoder"},
        "s": {"class": "rnn_cell", "unit": "standardlstm", "unit_opts": {"use_peepholes": True, "forget_bias": 0.0},
              "initial_state": "var", "from": ["target_embed", "att"], "n_out": 10},
        "readout_in": {"class": "linear", "from": ["prev:s", "prev:target_embed", "att"], "activation": None,
                       "n_out": 10},
        "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
        "output_prob": {"class": "softmax", "from": ["readout"], "target": "classes", "loss": "ce"}
      }, "target": "classes", "max_seq_len": 20},

      "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance", "target": "classes"}
    }
    config.update({"num_outputs": n_out,
      "num_inputs": n_in,
      "network": net_dict})
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])
    weight_input_layer_moved_out = network.layers["output"].params["target_embed/W"]
    assert(weight_input_layer_moved_out in set(network.get_trainable_params()))

    weight_output_layer_moved_out = network.layers["output"].params["readout_in/W"]
    assert(weight_output_layer_moved_out in set(network.get_trainable_params()))

    weight_internal = network.layers["output"].params["prev_s_transformed/W"]
    assert(weight_internal in set(network.get_trainable_params()))


def test_OptimalCompletionsLayer():
  with make_scope() as session:
    from TFNetworkLayer import InternalLayer
    from TFUtil import expand_dims_unbroadcast
    net = TFNetwork(
      extern_data=ExternData({"target": {"dim": 20, "sparse": True}}),
      config=Config({"debug_print_layer_output_template": True}))
    target = net.extern_data.data["target"]
    target_shape = tf.shape(target.placeholder)
    n_batch = target_shape[0]
    n_max_seq_len = target_shape[1]
    # Fake that we are inside a rec layer.
    net.set_rec_step_info(
      i=tf.convert_to_tensor(0, name="i"),
      end_flag=expand_dims_unbroadcast(tf.convert_to_tensor(False), 0, n_batch),
      seq_lens=target.get_sequence_lengths())
    kwargs = dict(
      name="opt_completions", network=net, debug=True, target="target",
      sources=[
        InternalLayer(
          name="last_row", network=net,
          output=Data(
            name="last_row", shape=(None,), dtype="int32",
            placeholder=expand_dims_unbroadcast(tf.range(n_max_seq_len + 1), 0, n_batch)))]
      )
    print("OptimalCompletionsLayer kwargs:")
    pprint(kwargs)
    kwargs["output"] = OptimalCompletionsLayer.get_out_data_from_opts(**kwargs)
    layer = OptimalCompletionsLayer(**kwargs)
    layer.output.sanity_check()
    out = session.run(
      layer.output.placeholder, feed_dict={
        target.placeholder: numpy.array([[3, 7, 8, 9, 13, 13, 0]]),
        target.size_placeholder[0]: numpy.array([7])})
    print(out)
    assert isinstance(out, numpy.ndarray)
    assert out.shape == (1, 20)
    assert out[0, 3] == 0 and all(out[0, :3] == 1) and all(out[0, 4:] == 1)


def test_extra_scatter_nd_search_train():
  from TFNetworkRecLayer import _SubnetworkRecCell
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
    from TFUtil import where_bc
    enc = source(1, as_data=True, auto_convert=False)
    dec = source(0, as_data=True, auto_convert=False)
    enc_lens = enc.get_sequence_lengths()
    dec_lens = dec.get_sequence_lengths()
    dec_shape = tf.shape(dec.placeholder)
    dec_time_dim = dec_shape[dec.time_dim_axis]
    dec_times = tf.expand_dims(tf.range(dec_time_dim), axis=0)  # (1,dec-T)
    x = tf.cast(dec_times + 1, tf.float32)  # (1,dec-T)
    # We want: x[dec_len - 1] == enc_time - 1.
    factors = (
      tf.maximum(tf.cast(enc_lens - 1, tf.float32), 0.0) /
      tf.maximum(tf.cast(dec_lens, tf.float32), 1.0))  # (B,)
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
    "lstm0_pool": {"class": "pool", "mode": "max", "padding": "same", "pool_size": (3,),
                   "from": ["lstm0_fw", "lstm0_bw"]},

    "encoder0": {"class": "linear", "from": "data", "activation": "relu", "n_out": EncValueTotalDim},
    "encoder": {"class": "postfix_in_time", "postfix": 0.0, "from": "encoder0"},

    "enc_ctx": {"class": "linear", "activation": None, "with_bias": True, "from": "encoder", "n_out": EncKeyTotalDim},
    "enc_value": {"class": "copy", "from": "encoder"},  # (B, enc-T, D)
    "enc0_seq_len": {"class": "length", "from": "encoder0", "sparse": True},

    "decision": {"class": "decide", "from": "extra.search_label:output"},  # for search task

    "extra.search1:t_search": {"class": "decide", "from": "extra.search1:output/t"},
    "extra.search2:t_search": {"class": "decide", "from": "extra.search2:output/t"},

    "0_data_target0": {
      "class": "postfix_in_time", "postfix": 0, "from": "data:%s" % target, "register_as_extern_data": "target0"},

    "1_data_t_linear": {
      "class": "eval", "from": ["data:target0", "encoder"], "eval": t_linear,
      "out_type": {"batch_dim_axis": 0, "time_dim_axis": 1, "shape": (None,), "sparse": True, "dtype": "int32",
                   "dim": None},
      "size_target": "target0",
      "register_as_extern_data": "t_linear"  # if task == "train" else None
    },

    "2_data_t_search_target1": {
      "class": "copy", "from": "extra.search1:t_search",
      "register_as_extern_data": "t_search_target1"  # if task == "train" else None
    },
    "2_data_t_search_target2": {
      "class": "copy", "from": "extra.search2:t_search",
      "register_as_extern_data": "t_search_target2"  # if task == "train" else None
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
        return source(0) * frac + source(1) * (1. - frac)
      else:
        source(1)  # call, but ignore
        return source(0)  # only hard att

    return {
      "class": "rec", "from": [], "back_prop": backprop,
      "unit": {
        "s_transformed": {"class": "linear", "activation": None, "with_bias": False, "from": ["s"],
                          "n_out": EncKeyTotalDim},
        "t_rel_var": {"class": "variable", "shape": (6, EncKeyTotalDim)},
        "t_rel_idxs_": {"class": "range", "limit": 6, "sparse": True},
        "t_rel_idxs": {"class": "combine", "kind": "add", "from": ["prev:t", "t_rel_idxs_"]},
        "energy_in": {"class": "combine", "kind": "add",
                      "from": ["base:enc_ctx", "s_transformed", "energy_in_t_rel_var"], "n_out": EncKeyTotalDim},
        "energy_in_t_rel_var": {
          "class": "scatter_nd", "from": "t_rel_var", "position": "t_rel_idxs", "position_axis": "except_batch:-1",
          "output_dim_via_time_from": "base:enc_ctx", "filter_invalid_indices": True},
        "energy_tanh": {"class": "activation", "activation": "tanh", "from": "energy_in"},

        "energy": {"class": "linear", "activation": None, "with_bias": False, "from": ["energy_tanh"],
                   "n_out": AttNumHeads},  # (B, enc-T, H)
        "energy1": {"class": "squeeze", "axis": "f", "from": "energy"},  # (B, enc-T)
        "energy2": {"class": "reinterpret_data", "from": "energy1", "set_axes": {"t": "stag:enc"}},
        "att_weights": {"class": "softmax_over_spatial", "from": "energy2", "start": "t_start"},  # (B, enc-T)
        # ChoiceLayer works on the feature axis.
        "att_weights1": {
          "class": "reinterpret_data", "from": "att_weights", "set_axes": {"f": "stag:enc"},
          "target": t_target if train else None, "loss": "ce" if (train and t_target) else None},

        "t0": {
          "class": "choice", "from": "att_weights1",
          # "target": None,
          "target": t_target, "cheating": bool(t_target),  # add this in training
          "beam_size": beam_size,
          "length_normalization": False, "initial_output": -1},  # (B,)
        # Note: If beam-size > enc_seq_len, we end up with invalid t in the beam. Fix that.
        "t1": {"class": "eval", "from": ["t0", "base:enc0_seq_len"], "eval": "tf.minimum(source(0), source(1))"},
        "t": {
          # "class": "print",
          "class": "copy",
          "from": "t1", "initial_output": -1, "is_output_layer": bool(t_search)},
        # Only for debugging.
        "t_err": {"class": "eval", "from": ["t", "data:%s" % t_target], "collocate_with": "t",
                  "eval": "tf.cast(tf.abs(source(0) - source(1)), tf.float32)",
                  "loss": "as_is" if (t_target and t_search) else None, "out_type": {"dtype": "float32"},
                  "only_on_search": True},

        "t_start": {
          # Need right start for masking to avoid infs.
          "class": "eval", "from": ["prev:t", "data:%s" % t_target],
          "eval": "tf.minimum(source(0), source(1))"}
        if t_target else
        {"class": "copy", "from": "prev:t"},

        "att0": {"class": "gather_nd", "position": "t", "from": "base:enc_value"},  # (B, V)
        "att1": {"class": "generic_attention", "weights": "att_weights", "base": "base:enc_value"},  # (B, V)
        "att1_": {"class": "switch", "condition": lambda **kw: use_soft_att, "true_from": "att1", "false_from": "att0"},
        "att": {"class": "eval", "from": ["att0", "att1_"], "eval": combine_soft_hard_att},

        "s": {"class": "rec", "unit": "nativelstm2", "from": ["prev:target_embed", "prev:att"], "n_out": 8},
        "readout_in": {"class": "linear", "from": ["s", "prev:target_embed", "att"], "activation": None,
                       "n_out": 10},
        "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
        "output_prob": {"class": "softmax", "from": ["readout"], "target": "target0", "loss": "ce" if train else None},

        'target_embed': {'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'],
                         "n_out": 6, "initial_output": "var"},
        'output': {
          'class': 'choice', 'target': "target0", 'beam_size': beam_size, 'from': ["output_prob"],
          "initial_output": 0,
          'search': label_search, "length_normalization": label_search},
        "end": {"class": "compare", "from": "output", "value": 0},
      },
      "target": ["target0", t_target] if t_target else ["target0"],
      "size_target": t_target,
      "max_seq_len": "max_len_from('base:encoder0')"}

  # Train task:
  net_dict["extra.search1:output"] = get_output_dict(
    train=False, t_search=True, label_search=False, backprop=False, t_target="t_linear", use_soft_att=True)
  net_dict["extra.search2:output"] = get_output_dict(
    train=False, t_search=True, label_search=False, backprop=False, t_target="t_linear", use_soft_att=False)
  net_dict["extra.1:output"] = get_output_dict(
    train=True, t_search=False, label_search=False, backprop=True, t_target="t_linear", use_soft_att=True)
  # extra.2 is basically like extra.1, only different t_target, and that should not make any difference for the
  # construction. But anyway, put it in as another variation.
  net_dict["extra.2:output"] = get_output_dict(
    train=True, t_search=False, label_search=False, backprop=True, t_target="t_search_target1", use_soft_att=True)
  # extra.3 does not use soft-attention anymore. That enables a couple of new optimizations in the rec loop,
  # esp now we should be able to move *everything* out.
  net_dict["extra.3:output"] = get_output_dict(
    train=True, t_search=False, label_search=False, backprop=True, t_target="t_search_target2", use_soft_att=False)
  # Search task:
  # net_dict["extra.search_label:output"] = get_output_dict(
  #   train=True, t_search=True, label_search=True, backprop=False, t_target=None, use_soft_att=False)

  config = Config()
  config.update({
    "extern_data": {"data": {"dim": n_in}, target: {"dim": n_out, "sparse": True}},
    "debug_print_layer_output_template": True})

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

    session.run(tf.variables_initializer(tf.global_variables() + [network.global_train_step]))
    outputs = [train1_search_out.placeholder, train1_out.placeholder,
               train2_search_out.placeholder, train2_out.placeholder, train3_out.placeholder]
    info, out = session.run(
      (fetches, outputs),
      feed_dict={
        data_input.placeholder: rnd.normal(size=(n_batch, n_enc_time, n_in)).astype("float32"),
        data_input.size_placeholder[0]: numpy.array([n_enc_time] * n_batch, dtype="int32"),
        data_target.placeholder: rnd.randint(0, n_out, size=(n_batch, n_dec_time,), dtype="int32"),
        data_target.size_placeholder[0]: numpy.array([n_dec_time] * n_batch, dtype="int32"),
      })
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
    "target_with_new_len": {"class": "reinterpret_data", "from": "target", "size_base": "data",
                            "register_as_extern_data": "targets"},
    "decision": {
      "class": "decide", "from": ["output"], "loss": "edit_distance", "target": "targets",
      "is_output_layer": True},

    'output': {
      'class': 'rec',
      "from": [],
      'target': 'data',
      "max_seq_len": "max_len_from('data') * 3",
      'unit': {
        'prefix': {
          'class': 'eval', 'out_type': {'time_dim_axis': None, 'shape': ()},
          "collocate_with": "output_choice",
          'from': ['base:data'],
          'eval': 'source(0, auto_convert=False)[:, tf.minimum(self.network.get_rec_step_index(), source(0, auto_convert=False, as_data=True).time_dimension() - 1)]'
        },  # Shape (None,).
        'in_prefix': {
          'class': 'eval', 'from': 'base:data',
          "collocate_with": "output_choice",
          'out_type': {'time_dim_axis': None, 'shape': (), "dtype": "bool", "dim": 2},
          # True if still in SRC.
          'eval': 'tf.less(self.network.get_rec_step_index(), source(0, as_data=True, auto_convert=False).get_sequence_lengths())'
        },  # Shape (None,).
        'output': {
          'class': 'switch', "condition": "in_prefix", 'true_from': 'prefix', "false_from": 'output_choice',
          "initial_output": 0},
        "end": {
          "class": "eval", "from": 'base:data', "collocate_with": "output_choice",
          "out_type": {'time_dim_axis': None, 'shape': (), "dtype": "bool", "dim": 2},
          # Arbitrary. We can check that, though.
          'eval': 'tf.greater_equal(self.network.get_rec_step_index(), source(0, as_data=True, auto_convert=False).get_sequence_lengths() * 3 // 2)'
          },

        'output_choice': {
          'class': 'choice', 'target': 'targets', 'beam_size': beam_size,
          'from': ["prob_output"], "initial_output": 0},

        "prob_output": {
          'class': 'softmax',
          'from': ['decoder'],
          'loss': 'ce',
          'target': 'targets',
          'with_bias': True},
        "decoder": {'class': 'layer_norm', 'from': ['dec_0']},
        'dec_0': {'class': 'copy', 'from': ['dec_0_ff_out']},

        'target_embed_raw': {'activation': None,
                             'class': 'linear',
                             'from': ['prev:output'],  # Note: Here was the bug.
                             'n_out': emb_dim,
                             'with_bias': False},
        'target_embed': {'class': 'dropout', 'dropout': 0, 'from': ['target_embed_raw']},
        'target_embed_lin': {'activation': None,
                             'class': 'linear',
                             'from': ['target_embed'],
                             'n_out': trans_out_dim,
                             'with_bias': False},
        'dec_0_self_att_laynorm': {'class': 'layer_norm', 'from': ['target_embed_lin']},
        'dec_0_self_att_att': {'attention_left_only': True,
                               'class': 'self_attention',
                               'from': ['dec_0_self_att_laynorm'],
                               'n_out': v_dim,
                               'num_heads': num_heads,
                               'total_key_dim': qk_dim},
        'dec_0_self_att_lin': {'activation': None,
                               'class': 'linear',
                               'from': ['dec_0_self_att_att'],
                               'n_out': trans_out_dim,
                               'with_bias': False},
        'dec_0_self_att_drop': {'class': 'dropout', 'dropout': 0, 'from': ['dec_0_self_att_lin']},
        'dec_0_att_out': {'class': 'combine',
                          'from': ['target_embed_lin', 'dec_0_self_att_drop'],
                          'kind': 'add',
                          'n_out': trans_out_dim,
                          'trainable': True},
        'dec_0_ff_laynorm': {'class': 'layer_norm', 'from': ['dec_0_att_out']},
        'dec_0_ff_conv1': {'activation': "relu",
                           'class': 'linear',
                           'from': ['dec_0_ff_laynorm'],
                           'n_out': ff_dim,
                           'with_bias': True},
        'dec_0_ff_conv2': {'activation': None,
                           'class': 'linear',
                           'from': ['dec_0_ff_conv1'],
                           'n_out': trans_out_dim,
                           'with_bias': True},
        'dec_0_ff_drop': {'class': 'dropout', 'dropout': 0, 'from': ['dec_0_ff_conv2']},
        'dec_0_ff_out': {'class': 'combine', 'from': ['dec_0_att_out', 'dec_0_ff_drop'], 'kind': 'add', 'n_out': trans_out_dim},
      }}
  }

  n_batch, n_in, n_time = 3, 19, 9
  n_out = n_in

  config = Config()
  config.update({
    "extern_data": {"data": {"dim": n_out, "sparse": True}},
    "search_output_layer": "decision",
    "debug_print_layer_output_template": True})

  with make_scope() as session:
    network = TFNetwork(config=config, train_flag=False, search_flag=True)
    pprint(network.extern_data.data)
    network.construct_from_dict(net_dict)

    fetches = network.get_fetches_dict()
    data_input = network.extern_data.data["data"]
    assert data_input.batch_shape == (None, None)
    output_out = network.get_layer("decision").output
    assert output_out.is_batch_major and output_out.sparse and output_out.dim == n_out and output_out.shape == (None,)

    input_seq_lens = numpy.array([n_time, n_time - 5, n_time - 4], dtype="int32")
    assert input_seq_lens.shape == (n_batch,) and all(input_seq_lens > 0)
    input_seqs = rnd.randint(1, n_out, size=(n_batch, n_time,), dtype="int32")
    print("input:")
    print(input_seqs)
    print("lens:", input_seq_lens)

    session.run(tf.variables_initializer(tf.global_variables() + [network.global_train_step]))
    info, out_seqs, out_seq_lens = session.run(
      (fetches, output_out.placeholder, output_out.get_sequence_lengths()),
      feed_dict={
        data_input.placeholder: input_seqs,
        data_input.size_placeholder[0]: input_seq_lens})
    print(info)
    print("output:")
    print(out_seqs)  # random...
    print("lens:", out_seq_lens)
    assert isinstance(out_seqs, numpy.ndarray) and isinstance(out_seq_lens, numpy.ndarray)
    assert len(out_seqs.shape) == 2 and out_seqs.shape[0] == n_batch
    assert out_seq_lens.shape == (n_batch,)

    for i in range(n_batch):
      assert out_seq_lens[i] == input_seq_lens[i] * 3 // 2  # we constructed the 'end' layer that way
      assert all(out_seqs[i, :input_seq_lens[i]] == input_seqs[i, :input_seq_lens[i]])


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
    #if len(list(threading.enumerate())) > 1:
    #  print("Warning, more than one thread at exit:")
    #  better_exchook.dump_all_thread_tracebacks()
