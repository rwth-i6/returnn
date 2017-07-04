
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
