
import os
import os.path
import tensorflow as tf
import numpy as np
from TFNetwork import *
from TFNetworkSigProcLayer import *
from Config import Config
import contextlib

@contextlib.contextmanager
def make_scope():
  with tf.Graph().as_default() as graph:
    with tf.Session(graph=graph) as session:
      yield session

 
def test_melFilterbankLayer():
  with make_scope() as session:
    n_in, n_out = 257, 3
    layer_name = "mel_filterbank_layer"
    config = Config()
    config.update({
      "num_outputs": n_out,
      "num_inputs": n_in,
      "network": {
        layer_name: {
          "class": "mel_filterbank", "fft_size": 512, "nr_of_filters": n_out, "n_out": n_out, "is_output_layer": True}
      }})
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_value("network"))
    layer = network.layers[layer_name]
    test_out = session.run(layer.output.placeholder, {network.get_extern_data('data').placeholder: np.ones((1, 1, 257))})
    assert np.sum(test_out - np.asarray([28.27923584, 53.10634232, 99.71585846], dtype=np.float32)) < 1e-5

def test_complexLinearProjectionLayer():
  with make_scope() as session:
    n_in, n_out = 514, 128 
    layer_name = "clp_layer"
    config = Config()
    config.update({
      "num_outputs": n_out,
      "num_inputs": n_in,
      "network": {
        layer_name: {
          "class": "complex_linear_projection", "nr_of_filters": n_out, "n_out": n_out, "is_output_layer": True}
      }})
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_value("network"))
    layer = network.layers[layer_name]
    i_r = np.ones((1, n_in // 2))
    i_i = np.ones((1, n_in // 2)) * 0.5
    test_input = np.expand_dims(np.reshape(np.transpose(np.reshape(np.concatenate([i_r, i_i], axis=1), (1, 2, 257)), [0, 2, 1]), (1, 514)), 0)
    test_clp_kernel = np.ones((2, n_in // 2, 128))
    test_clp_output = session.run(layer.output.placeholder, {network.get_extern_data('data').placeholder: test_input, layer._clp_kernel: test_clp_kernel})
    assert test_clp_output[0, 0, 0] - 6.00722122 < 1e-5

if __name__ == "__main__":
    test_melFilterbankLayer()
    test_complexLinearProjectionLayer()
