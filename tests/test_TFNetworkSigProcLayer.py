
from __future__ import print_function

# Disable extensive TF debug verbosity. Must come before the first TF import.
import logging
logging.getLogger('tensorflow').disabled = True

import sys
import os
import os.path
import tensorflow as tf
import numpy as np
import unittest
import contextlib

# Allow Returnn imports.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from TFNetwork import *
from TFNetworkSigProcLayer import *
from Config import Config

import better_exchook
better_exchook.replace_traceback_format_tb()


@contextlib.contextmanager
def make_scope():
  """
  :rtype: tf.Session
  """
  with tf.Graph().as_default() as graph:
    with tf.Session(graph=graph) as session:
      yield session


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
    assert isinstance(layer, ComplexLinearProjectionLayer)
    i_r = np.ones((1, n_in // 2))
    i_i = np.ones((1, n_in // 2)) * 0.5
    test_input = np.expand_dims(np.reshape(np.transpose(
      np.reshape(np.concatenate([i_r, i_i], axis=1), (1, 2, 257)), [0, 2, 1]), (1, 514)), 0)
    test_clp_kernel = np.ones((2, n_in // 2, 128))
    test_clp_output = session.run(
      layer.output.placeholder,
      feed_dict={network.get_extern_data('data').placeholder: test_input, layer._clp_kernel: test_clp_kernel})
    assert test_clp_output[0, 0, 0] - 6.00722122 < 1e-5


if __name__ == "__main__":
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
