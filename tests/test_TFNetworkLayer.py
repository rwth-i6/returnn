
# start: nosetests $this_file --nologcapture

import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import sys
sys.path += ["."]  # Python 3 hack
from nose.tools import assert_equal, assert_is_instance
import contextlib
import numpy.testing
import better_exchook
better_exchook.replace_traceback_format_tb()

from Config import Config
from TFNetwork import *
from TFNetworkLayer import *


@contextlib.contextmanager
def make_scope():
  with tf.Graph().as_default() as graph:
    with tf.Session(graph=graph) as session:
      yield session


def test_activation_layer_net_construct():
  with make_scope() as session:
    num_inputs = 2
    config = Config()
    config.update({
      "num_outputs": 3,
      "num_inputs": num_inputs,
      "network": {
        "output": {"class": "activation", "activation": "relu", "from": ["data"]}
      }})
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_value("network"))
    out = network.get_default_output_layer().output.placeholder
    n_batch = 1
    seq_len = 3
    feed = {network.extern_data.get_default_input_data().placeholder:
            numpy.array([[[0, 0], [-1, -1], [2, 2]]], dtype="float32")}
    assert_equal(feed[network.extern_data.get_default_input_data().placeholder].shape, (n_batch, seq_len, num_inputs))
    v = session.run(out, feed_dict=feed)
    assert_equal(v.shape, (n_batch, seq_len, num_inputs))
    assert_equal(v.tolist(), [[[0, 0], [0, 0], [2, 2]]])


def test_activation_layer_net_construct_two_out():
  with make_scope() as session:
    num_inputs = 2
    config = Config()
    config.update({
      "num_outputs": 3,
      "num_inputs": num_inputs,
      "network": {
        "0out": {"class": "linear", "n_out": 1, "activation": "relu", "from": ["data"], "is_output_layer": True},
        "output": {"class": "activation", "activation": "relu", "from": ["data"]}
      }})
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_value("network"))
    session.run(tf.global_variables_initializer())
    out = network.layers["output"].output.placeholder
    out2 = network.layers["0out"].output.placeholder
    n_batch = 1
    seq_len = 3
    feed = {network.extern_data.get_default_input_data().placeholder:
            numpy.array([[[0, 0], [-1, -1], [2, 2]]], dtype="float32")}
    assert_equal(feed[network.extern_data.get_default_input_data().placeholder].shape, (n_batch, seq_len, num_inputs))
    v, v2 = session.run([out, out2], feed_dict=feed)
    assert_equal(v.shape, (n_batch, seq_len, num_inputs))
    assert_equal(v.tolist(), [[[0, 0], [0, 0], [2, 2]]])


def test_combine_layer_net_construct():
  with make_scope() as session:
    net_dict = {
      "lstm0_fw": {"class": "rec", "unit": "lstmp", "n_out": 5, "dropout": 0.0, "L2": 0.01, "direction": 1},
      "lstm0_bw": {"class": "rec", "unit": "lstmp", "n_out": 5, "dropout": 0.0, "L2": 0.01, "direction": -1},
      "lstm0_avg": {"class": "combine", "kind": "average", "from": ["lstm0_fw", "lstm0_bw"], "trainable": False},
      "output": {"class": "softmax", "loss": "ce", "from": ["lstm0_avg"]}
    }
    config = Config()
    config.update(dict(num_inputs=4, num_outputs=9))
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(net_dict)


def test_subnetwork_layer_net_construct():
  with make_scope() as session:
    net_dict = {
      "ff0": {"class": "forward", "activation": "tanh", "n_out": 3},
      "sub": {"class": "subnetwork", "from": ["ff0"], "subnetwork": {
        "ff1": {"class": "forward", "activation": "relu", "n_out": 2},
        "output": {"class": "forward", "activation": "relu", "n_out": 2}
      }},
      "output": {"class": "softmax", "loss": "ce", "from": ["sub"]}
    }
    config = Config()
    config.update(dict(num_inputs=4, num_outputs=3))
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(net_dict)
    assert_equal(network.layers["sub"].output.dim, 2)


def test_constant_layer():
  with make_scope() as session:
    config = Config()
    config.update({
      "num_outputs": 3,
      "num_inputs": 2,
      "network": {
        "output": {"class": "constant", "value": 42, "from": []}
      }
    })
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])
    out = network.get_default_output_layer(must_exist=True)
    v = session.run(out.output.placeholder)
    assert_equal(v.shape, (1,))  # (batch,), where batch==1 for broadcasting
    assert_equal(v[0], 42)


def test_compare_layer():
  with make_scope() as session:
    config = Config()
    config.update({
      "model": "/tmp/model",
      "num_outputs": 3,
      "num_inputs": 2,
      "network": {
        "const": {"class": "constant", "value": 3, "from": []},
        "output": {"class": "compare", "from": ["const"], "value": 3}
      }
    })
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])
    out = network.get_default_output_layer(must_exist=True)
    v = session.run(out.output.placeholder)
    assert_equal(v.shape, (1,))  # (batch,), where batch==1 for broadcasting
    assert_equal(v.dtype, numpy.dtype("bool"))
    assert_equal(v[0], True)


def test_layer_base_get_out_data_from_opts():
  with make_scope() as session:
    config = Config()
    config.update({
      "num_inputs": 4,
      "num_outputs": 3
    })
    network = TFNetwork(config=config)
    input_data = network.extern_data.data["data"]
    target_data = network.extern_data.data["classes"]
    assert input_data.dim == 4
    assert input_data.shape == (None, 4)
    assert not input_data.sparse
    assert input_data.dtype == "float32"
    assert target_data.dim == 3
    assert target_data.shape == (None,)
    assert target_data.sparse
    assert target_data.dtype == "int32"
    out = LayerBase._base_get_out_data_from_opts(network=network, name="output", target="classes")
    # Output data type is a non-sparse version of the targets by default.
    assert out.dim == target_data.dim
    assert out.shape == target_data.shape_dense
    assert not out.sparse
    assert out.dtype == "float32"


def test_SplitDimsLayer_resolve_dims():
  assert_equal(SplitDimsLayer._resolve_dims(old_dim=3 * 5, new_dims=(3, -1)), (3, 5))
  assert_equal(SplitDimsLayer._resolve_dims(old_dim=3 * 5, new_dims=(3, 5)), (3, 5))
  assert_equal(SplitDimsLayer._resolve_dims(old_dim=3 * 5, new_dims=(5, -1)), (5, 3))
  assert_equal(SplitDimsLayer._resolve_dims(old_dim=2 * 3 * 5, new_dims=(-1, 3, 5)), (2, 3, 5))
  assert_equal(SplitDimsLayer._resolve_dims(old_dim=2 * 3 * 5, new_dims=(2, -1, 5)), (2, 3, 5))
  assert_equal(SplitDimsLayer._resolve_dims(old_dim=2 * 3 * 5, new_dims=(2, 3, -1)), (2, 3, 5))
  assert_equal(SplitDimsLayer._resolve_dims(old_dim=2 * 3 * 5, new_dims=(2, 3, -1, 1)), (2, 3, 5, 1))


def test_ConvLayer_get_valid_out_dim():
  assert_equal(ConvLayer.calc_out_dim(in_dim=10, stride=1, filter_size=2, padding="same"), 10)
  assert_equal(ConvLayer.calc_out_dim(in_dim=10, stride=1, filter_size=3, padding="same"), 10)
  assert_equal(ConvLayer.calc_out_dim(in_dim=10, stride=1, filter_size=2, padding="valid"), 9)
  assert_equal(ConvLayer.calc_out_dim(in_dim=10, stride=1, filter_size=3, padding="valid"), 8)
  assert_equal(ConvLayer.calc_out_dim(in_dim=10, stride=2, filter_size=2, padding="valid"), 5)
  assert_equal(ConvLayer.calc_out_dim(in_dim=10, stride=3, filter_size=2, padding="valid"), 3)
  assert_equal(ConvLayer.calc_out_dim(in_dim=10, stride=3, filter_size=1, padding="valid"), 4)
  assert_equal(ConvLayer.calc_out_dim(in_dim=10, stride=3, filter_size=2, padding="same"), 4)
  assert_equal(ConvLayer.calc_out_dim(in_dim=41, stride=1, filter_size=2, padding="valid"), 40)
  assert_equal(ConvLayer.calc_out_dim(in_dim=40, stride=2, filter_size=2, padding="valid"), 20)


def test_reuse_params():
  with make_scope() as session:
    config = Config()
    n_in, n_out = 2, 3
    config.update({
      "num_outputs": n_out,
      "num_inputs": n_in,
      "network": {
        "l1": {"class": "linear", "activation": None, "n_out": n_out},
        "output": {"class": "linear", "activation": None, "n_out": n_out, "reuse_params": "l1"}
      }
    })
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])
    l1 = network.layers["l1"]
    l2 = network.layers["output"]
    assert_equal(set(l1.params.keys()), {"W", "b"})
    assert_equal(set(l2.params.keys()), {"W", "b"})
    assert l1.params["W"] is l2.params["W"]
    assert l1.params["b"] is l2.params["b"]
    assert_equal(set(network.get_trainable_params()), {l1.params["W"], l1.params["b"]})
