
# start: nosetests $this_file --nologcapture

from __future__ import print_function

import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import sys
sys.path += ["."]  # Python 3 hack
from nose.tools import assert_equal, assert_is_instance
import unittest
import numpy.testing
import better_exchook
better_exchook.replace_traceback_format_tb()

from Log import log
from Config import Config
from TFNetwork import *
from TFNetworkRecLayer import *
from TFUtil import is_gpu_available

log.initialize(verbosity=[5])


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
      output_data1 = session.run(
        network1.get_default_output_layer().output.placeholder,
        feed_dict={
          network1.extern_data.data["data"].placeholder: input_data,
          network1.extern_data.data["data"].size_placeholder[0]: seq_lens})
      assert_equal(output_data1.shape, (seq_lens[0], 1, num_outputs))  # (time, batch, dim)
      print("Saveable params:", network1.get_saveable_params_list())
      network1.save_params_to_file(filename=model_filename, session=session)
    print()

    # First test if we can load the same network as-is. This will involve the RNNParamsSaveable.
    print("Testing restore of same network with cuDNN.")
    tf.reset_default_graph()
    with tf.Session() as session:
      network1a = TFNetwork(config=config1, train_flag=True)
      network1a.construct_from_dict(config1.typed_dict["network"])
      network1a.load_params_from_file(filename=model_filename, session=session)
      for layer_name, layer1 in sorted(network1a.layers.items()):
        print("layer: %r" % layer_name)
        for param_name, param1 in sorted(layer1.params.items()):
          print("  param %r: %r" % (param_name, param1))
          param1old = params[layer_name][param_name]
          param1new = param1.eval(session)
          numpy.testing.assert_almost_equal(param1old, param1new)
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
      params = tf.Variable(tf.random_uniform([model.params_size()]), validate_shape=False)
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

    print("Create updater...")
    from TFUpdater import Updater
    updater = Updater(config=config, network=network, tf_session=session)
    updater.set_trainable_vars(network.get_trainable_params())
    updater.set_learning_rate(0.1)
    optim_op = updater.get_optim_op()
    layer = network.layers["output"]
    loss_t = network.get_total_loss() * layer.get_loss_normalization_factor()
    weights_t = layer.params["W"]
    summaries_t = tf.summary.merge_all()

    print("Training...")
    recent_info = []  # type: list[dict[str]]
    for i in range(10000):
      try:
        loss, _, summaries, weights = session.run(
          [loss_t, optim_op, summaries_t, weights_t], feed_dict=make_feed_dict(5))
      except tf.errors.InvalidArgumentError as exc:
        print("TensorFlow exception in step %i." % i)
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
        raise Exception("TF exception in step %i." % i)
      if len(recent_info) > 1000:
        recent_info.pop(0)
      recent_info.append({"step": i, "loss": loss, "summaries": summaries, "weights": weights})
      if not numpy.isfinite(loss) or i % 100 == 0:
        print("step %i, loss: %r" % (i, loss))
      assert numpy.isfinite(loss)


if __name__ == "__main__":
  try:
    better_exchook.install()
    if len(sys.argv) <= 1:
      for k, v in sorted(globals().items()):
        if k.startswith("test_"):
          print("-" * 40)
          print("Executing: %s" % k)
          v()
          print("-" * 40)
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
    if len(list(threading.enumerate())) > 1:
      print("Warning, more than one thread at exit:")
      better_exchook.dump_all_thread_tracebacks()
