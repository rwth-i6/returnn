
# start: nosetests $this_file --nologcapture

import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import sys
sys.path += ["."]  # Python 3 hack
from nose.tools import assert_equal, assert_is_instance
import contextlib
import unittest
import numpy.testing
from pprint import pprint
import better_exchook
better_exchook.replace_traceback_format_tb()

from Config import Config
from TFNetwork import *
from TFNetworkLayer import *
from Log import log
import TFUtil
TFUtil.debugRegisterBetterRepr()

log.initialize(verbosity=[5])


@contextlib.contextmanager
def make_scope():
  with tf.Graph().as_default() as graph:
    with tf.Session(graph=graph) as session:
      yield session


def test_concat_sources():
  with make_scope() as session:
    network = TFNetwork(train_flag=True, extern_data=ExternData())
    n_batch = 5
    n_time = 3
    size_placeholder = {0: tf.constant(n_time, dtype=tf.int32, shape=(n_batch,))}
    src1 = InternalLayer(
      name="src1", network=network,
      output=Data(
        name="src1_output", shape=(None, 11), placeholder=tf.zeros((n_batch, n_time, 11)),
        size_placeholder=size_placeholder))
    print("src1 output:", src1.output)
    src2 = InternalLayer(
      name="src2", network=network,
      output=Data(
        name="src2_output", shape=(None, 13), placeholder=tf.zeros((n_batch, n_time, 13)),
        size_placeholder=size_placeholder))
    print("src2 output:", src2.output)
    out_kwargs = dict(name="out", sources=[src1, src2], network=network)
    out_output = CopyLayer.get_out_data_from_opts(**out_kwargs)
    print("out output:", out_output)
    assert out_output.dim == 11 + 13
    out = CopyLayer(output=out_output, **out_kwargs)
    session.run(out.output.placeholder)


def test_concat_sources_batch_dim():
  with make_scope() as session:
    network = TFNetwork(train_flag=True, extern_data=ExternData())
    n_batch = 5
    n_time = 3
    size_placeholder = {0: tf.constant(n_time, dtype=tf.int32, shape=(n_batch,))}
    src1 = InternalLayer(
      name="src1", network=network,
      output=Data(
        name="src1_output", shape=(None, 11), placeholder=tf.zeros((n_batch, n_time, 11)),
        size_placeholder=size_placeholder))
    print("src1 output:", src1.output)
    src2 = InternalLayer(
      name="src2", network=network,
      output=Data(
        name="src2_output", shape=(None, 13), time_dim_axis=0, batch_dim_axis=1,
        placeholder=tf.zeros((n_time, n_batch, 13)),
        size_placeholder=size_placeholder))
    print("src2 output:", src2.output)
    out_kwargs = dict(name="out", sources=[src1, src2], network=network)
    out_output = CopyLayer.get_out_data_from_opts(**out_kwargs)
    print("out output:", out_output)
    assert out_output.dim == 11 + 13
    assert out_output.batch_dim_axis == 0 and out_output.time_dim_axis == 1
    out = CopyLayer(output=out_output, **out_kwargs)
    session.run(out.output.placeholder)


def test_concat_sources_missing_dim():
  with make_scope() as session:
    network = TFNetwork(train_flag=True, extern_data=ExternData())
    n_batch = 5
    n_time = 3
    size_placeholder = {0: tf.constant(n_time, dtype=tf.int32, shape=(n_batch,))}
    src1 = InternalLayer(
      name="src1", network=network,
      output=Data(
        name="src1_output", shape=(None, 11), placeholder=tf.zeros((n_batch, n_time, 11)),
        size_placeholder=size_placeholder))
    print("src1 output:", src1.output)
    src2 = InternalLayer(
      name="src2", network=network,
      output=Data(
        name="src2_output", shape=(13,), time_dim_axis=None, batch_dim_axis=0,
        placeholder=tf.zeros((n_batch, 13)),
        size_placeholder={}))
    print("src2 output:", src2.output)
    out_kwargs = dict(name="out", sources=[src1, src2], network=network)
    out_output = CopyLayer.get_out_data_from_opts(**out_kwargs)
    print("out output:", out_output)
    assert out_output.dim == 11 + 13
    assert out_output.batch_dim_axis == 0 and out_output.time_dim_axis == 1
    out = CopyLayer(output=out_output, **out_kwargs)
    session.run(out.output.placeholder)


def test_batch_norm_vars():
  with make_scope() as session:
    n_in, n_out = 2, 3
    config = Config()
    layer_name = "layer1"
    config.update({
      "num_outputs": n_out,
      "num_inputs": n_in,
      "network": {
        layer_name: {
          "class": "linear", "activation": "relu", "batch_norm": True, "n_out": n_out, "is_output_layer": True}
      }})
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_value("network"))
    layer = network.layers[layer_name]
    print("layer:", layer)
    print("layer vars:")
    pprint(layer.params)
    assert layer.use_batch_norm
    bn_prefix = "batch_norm/%s_%s_output_" % (layer_name, layer_name)
    assert_equal(set(layer.params.keys()), {
      "W", "b", bn_prefix + "beta", bn_prefix + "mean", bn_prefix + "gamma", bn_prefix + "variance"})
    assert_equal(layer.params["W"].get_shape().as_list(), [n_in, n_out])
    assert_equal(layer.params["b"].get_shape().as_list(), [n_out])
    assert_equal(layer.params[bn_prefix + "beta"].get_shape().as_list(), [1, 1, n_out])
    assert_equal(layer.params[bn_prefix + "gamma"].get_shape().as_list(), [1, 1, n_out])
    assert_equal(layer.params[bn_prefix + "mean"].get_shape().as_list(), [1, 1, n_out])
    assert_equal(layer.params[bn_prefix + "variance"].get_shape().as_list(), [1, 1, n_out])


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


def test_dropout_layer_net_construct():
  with make_scope() as session:
    net_dict = {
      "drop": {"class": "dropout", "dropout": 0.3, "dropout_noise_shape": {"*": None}},
      "output": {"class": "softmax", "loss": "ce", "from": ["drop"]}
    }
    config = Config({"num_inputs": 4, "num_outputs": 9, "debug_print_layer_output_template": True})
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


def test_shift_layer():
  with make_scope() as session:
    import numpy as np
    batch_size = 8
    time_size = 20
    feat_size = 10
    shift_amount = 5  # right-shift of 5 elements
    config = Config()
    config.update({
      "num_outputs": feat_size,
      "num_inputs": feat_size,
      "network": {
        "output": {"class": "shift_axis", "from": ["data"], "amount": shift_amount, "pad": True, "axis": "T"}
      }
    })
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])
    out = network.get_default_output_layer(must_exist=True)
    input_data = np.ones(shape=(batch_size, time_size, feat_size))
    input_data[0, :, 0] = np.arange(time_size) # 0..time_size in time-axis
    feed_dict = {network.layers['data'].output.placeholder: input_data}
    v = session.run(out.output.placeholder, feed_dict)

    assert_equal(v.shape, (batch_size, time_size, feat_size))
    assert_equal(np.equal(v[0, shift_amount:, 0], np.arange(time_size-shift_amount)).all(), True)
    assert_equal((v[:,:shift_amount,:] == 0).all(), True)  # padding
    assert_equal((v[1:,shift_amount:,:] == 1).all(), True)


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


def test_ReduceLayer_reduce4d():
  config = Config()
  config.update({
    "num_inputs": 4,
    "num_outputs": 3,
    "debug_print_layer_output_template": True
  })
  network = TFNetwork(config=config)
  src_layer = InternalLayer(
    name="src", network=network, output=Data(name="src", shape=(None, 4, 512), auto_create_placeholders=True))
  print("src:", src_layer)
  opts = {
    'axes': "s:1",
    'keep_dims': True,
    'mode': 'mean',
    'name': 'c_out_reduce',
    'network': network,
    'sources': [src_layer]}
  out = ReduceLayer.get_out_data_from_opts(**opts)
  layer = ReduceLayer(output=out, **opts)
  print("layer:", layer)


def test_SplitDimsLayer_resolve_dims():
  assert_equal(SplitDimsLayer._resolve_dims(old_dim=3 * 5, new_dims=(3, -1)), (3, 5))
  assert_equal(SplitDimsLayer._resolve_dims(old_dim=3 * 5, new_dims=(3, 5)), (3, 5))
  assert_equal(SplitDimsLayer._resolve_dims(old_dim=3 * 5, new_dims=(5, -1)), (5, 3))
  assert_equal(SplitDimsLayer._resolve_dims(old_dim=2 * 3 * 5, new_dims=(-1, 3, 5)), (2, 3, 5))
  assert_equal(SplitDimsLayer._resolve_dims(old_dim=2 * 3 * 5, new_dims=(2, -1, 5)), (2, 3, 5))
  assert_equal(SplitDimsLayer._resolve_dims(old_dim=2 * 3 * 5, new_dims=(2, 3, -1)), (2, 3, 5))
  assert_equal(SplitDimsLayer._resolve_dims(old_dim=2 * 3 * 5, new_dims=(2, 3, -1, 1)), (2, 3, 5, 1))


def test_MergeDimsLayer():
  with make_scope() as session:
    net = TFNetwork(extern_data=ExternData())
    rnd = numpy.random.RandomState(42)

    def check(in_data_opts, in_static_shape, opts, out_data_shape, out_static_shape):
      """
      :param dict[str] in_data_opts:
      :param tuple[int] in_static_shape:
      :param dict[str] opts:
      :param tuple[int|None] out_data_shape:
      :param tuple[int] out_static_shape:
      """
      src = InternalLayer(name="src", network=net, out_type=in_data_opts)
      print("input:", src.output)
      src.output.placeholder = tf.constant(rnd.normal(size=in_static_shape).astype("float32"), dtype=tf.float32)
      src.output.size_placeholder = {}  # not sure if enough...
      opts = opts.copy()
      opts.update({"network": net, "name": "merge_dims_test", "sources": [src]})
      out_data = MergeDimsLayer.get_out_data_from_opts(**opts)
      print("output:", out_data)
      assert_equal(out_data.shape, out_data_shape)
      layer = MergeDimsLayer(output=out_data, **opts)
      assert_equal(layer.output.shape, out_data_shape)
      out_np = session.run(layer.output.placeholder)
      assert_equal(out_np.shape, out_static_shape)

    check({"shape": (4, 7), "time_dim_axis": None}, (2, 4, 7), {"axes": "except_batch"}, (4 * 7,), (2, 4 * 7))
    check({"shape": (4, None, 7), "time_dim_axis": None}, (2, 4, 3, 7), {"axes": "static"}, (None, 4 * 7), (2, 3, 4 * 7))
    check({"shape": (4, None, 7), "time_dim_axis": 2}, (2, 4, 3, 7), {"axes": "static"}, (None, 4 * 7), (2, 3, 4 * 7))


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
  assert_equal(ConvLayer.calc_out_dim(in_dim=2, stride=1, filter_size=3, padding="valid"), 0)


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


def test_reuse_params_map_custom():
  with make_scope() as session:
    config = Config()
    n_in, n_out = 2, 3
    config.update({
      "num_outputs": n_out,
      "num_inputs": n_in,
      "network": {
        "l1": {"class": "linear", "activation": "tanh", "with_bias": False, "n_out": 5},
        "output": {
          "class": "linear", "activation": None, "n_out": n_in, "from": ["l1"], "target": "data",
          "reuse_params": {
            "map": {
              "W": {
                "reuse_layer": "l1",
                "custom": (lambda reuse_layer, **kwargs: tf.transpose(reuse_layer.params["W"]))},
              "b": None}
          },
        }
      }
    })
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])
    l1 = network.layers["l1"]
    l2 = network.layers["output"]
    assert_equal(set(l1.params.keys()), {"W"})
    assert_equal(set(l2.params.keys()), {"b"})
    assert_equal(set(network.get_trainable_params()), {l1.params["W"], l2.params["b"]})


def test_reuse_params_map_custom_rev():
  with make_scope() as session:
    config = Config()
    n_in, n_out = 2, 3
    config.update({
      "num_outputs": n_out,
      "num_inputs": n_in,
      "network": {
        "output": {"class": "linear", "activation": "tanh", "with_bias": False, "from": ["l1"], "n_out": n_in},
        "l1": {
          "class": "linear", "activation": None, "n_out": 5, "from": ["data"], "target": "data",
          "reuse_params": {
            "map": {
              "W": {
                "reuse_layer": "output",
                "custom": (lambda reuse_layer, **kwargs: tf.transpose(reuse_layer.params["W"]))},
              "b": None}
          },
        }
      }
    })
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])
    l1 = network.layers["l1"]
    l2 = network.layers["output"]
    assert_equal(set(l1.params.keys()), {"b"})
    assert_equal(set(l2.params.keys()), {"W"})
    assert_equal(set(network.get_trainable_params()), {l2.params["W"], l1.params["b"]})


def test_reuse_params_map_custom_dep_loop():
  config = Config()
  n_in, n_out = 2, 3
  config.update({
    "num_outputs": n_out,
    "num_inputs": n_in,
    "network": {
      "encoder": {"class": "copy", "from": ["data"]},
      "enc_ctx": {"class": "linear", "activation": None, "with_bias": True, "from": ["encoder"], "n_out": 10},
      "inv_fertility": {"class": "linear", "activation": "sigmoid", "with_bias": False, "from": ["encoder"],
                        "n_out": 1},
      "output": {"class": "rec", "from": [], "unit": {
        'output': {'class': 'choice', 'target': 'classes', 'beam_size': 1, 'from': ["output_prob"],
                   "initial_output": 0},
        "end": {"class": "compare", "from": ["output"], "value": 0},
        'target_embed': {'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'], "n_out": 6,
                         "initial_output": 0},
        "weight_feedback": {"class": "linear", "activation": None, "with_bias": False,
                            "from": ["prev:accum_att_weights"], "n_out": 10},
        "prev_s_state": {"class": "get_last_hidden_state", "from": ["prev:s"], "n_out": 20},
        "prev_s_transformed": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev_s_state"],
                               "n_out": 10},
        "energy_in": {"class": "combine", "kind": "add",
                      "from": ["base:enc_ctx", "weight_feedback", "prev_s_transformed"], "n_out": 10},
        "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
        "energy": {"class": "linear", "activation": None, "with_bias": False, "from": ["energy_tanh"], "n_out": 1},
        "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},
        "accum_att_weights": {"class": "eval",
                              "from": ["prev:accum_att_weights", "att_weights", "base:inv_fertility"],
                              "eval": "source(0) + source(1) * source(2) * 0.5",
                              "out_type": {"dim": 1, "shape": (None, 1)}},
        "att": {"class": "generic_attention", "weights": "att_weights", "base": "base:encoder"},
        "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["target_embed", "att"], "n_out": 10},
        "readout_in": {"class": "linear", "from": ["prev:s", "prev:target_embed", "att"], "activation": None,
                       "n_out": 2 * 6},
        "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
        "output_prob": {
          "class": "softmax", "from": ["readout"], "dropout": 0.3,
          "reuse_params": {
            "map": {
              "W": {
                "reuse_layer": "target_embed",
                "custom": (lambda reuse_layer, **kwargs: tf.transpose(reuse_layer.params["W"]))},
              "b": None}},
          "target": "classes", "loss": "ce", "loss_opts": {"label_smoothing": 0.1}}
      }, "target": "classes", "max_seq_len": "max_len_from('base:encoder')"},
    }
  })
  with make_scope() as session:
    from TFNetworkRecLayer import RecLayer, _SubnetworkRecCell
    train_net = TFNetwork(config=config, train_flag=True)
    train_net.construct_from_dict(config.typed_dict["network"])
    train_rec_layer = train_net.layers["output"]
    assert isinstance(train_rec_layer, RecLayer)
    assert isinstance(train_rec_layer.cell, _SubnetworkRecCell)
    assert_equal(set(train_rec_layer.cell.input_layers_moved_out), {"output", "target_embed"})
    assert_equal(set(train_rec_layer.cell.output_layers_moved_out), {"output_prob", "readout", "readout_in"})
    assert isinstance(train_rec_layer.cell.output_layers_net, TFNetwork)
    assert_equal(set(train_rec_layer.cell.output_layers_net.layers["output_prob"].params.keys()), {"b"})
  with make_scope() as session:
    search_net = TFNetwork(config=config, train_flag=False, eval_flag=True, search_flag=True)
    search_net.construct_from_dict(config.typed_dict["network"])


def test_SliceLayer_output_placeholder():
  with make_scope() as session:
    net = TFNetwork(extern_data=ExternData())
    src = InternalLayer(name="src", network=net, out_type={"dim": 20, "sparse": True})
    src.output.placeholder = tf.constant([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]], dtype=tf.int32)
    src.output.size_placeholder = {0: tf.constant([5, 3, 2], dtype=tf.int32)}
    layer = SliceLayer(
      name="slice", network=net, axis="T", slice_step=2, slice_start=1, sources=[src],
      output=SliceLayer.get_out_data_from_opts(
        name="slice", network=net, axis="T", slice_step=2, slice_start=1, sources=[src]))
    out, seq_lens = session.run([layer.output.placeholder, layer.output.size_placeholder[0]])
    print(out)
    print(seq_lens)
    assert isinstance(out, numpy.ndarray)
    assert isinstance(seq_lens, numpy.ndarray)
    assert_equal(
      out.tolist(),
      [[2, 4],
       [7, 9],
       [12, 14]])
    assert_equal(seq_lens.tolist(), [2, 1, 1])


def test_WindowLayer_output_placeholder():
  with make_scope() as session:
    net = TFNetwork(extern_data=ExternData())
    src = InternalLayer(name="src", network=net, out_type={"dim": 20, "sparse": True})
    src.output.placeholder = tf.constant([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]], dtype=tf.int32)
    src.output.size_placeholder = {0: tf.constant([5, 3, 1], dtype=tf.int32)}
    layer = WindowLayer(
      name="window", network=net, axis="T", window_size=3, padding='valid', sources=[src],
      output=WindowLayer.get_out_data_from_opts(
        name="window", network=net, axis="T", window_size=3, padding='valid', sources=[src]))
    out, seq_lens = session.run([layer.output.placeholder, layer.output.size_placeholder[0]])
    print(out)
    print(seq_lens)
    assert isinstance(out, numpy.ndarray)
    assert isinstance(seq_lens, numpy.ndarray)
    assert_equal(
      out.tolist(),
      [[[1, 2, 3], [2, 3, 4], [3, 4, 5]],
       [[6, 7, 8], [7, 8, 9], [8, 9, 10]],
       [[11, 12, 13], [12, 13, 14], [13, 14, 15]]])
    assert_equal(seq_lens.tolist(), [3, 1, 0])


def test_conv_window_merge_dims():
  n_in = 1
  n_out = 13
  net_dict = {
    'conv_1': {'activation': 'abs',
               'class': 'conv',
               'filter_size': (4,),
               'n_out': 64,
               'padding': 'valid',
               'strides': 10},
    'pad_conv_1_time_dim': {'axes': 'time',
                            'class': 'pad',
                            'from': ['conv_1'],
                            'padding': 20},
    'conv_2': {'activation': 'abs',
               'class': 'conv',
               'filter_size': (2, 6),
               'from': ['pad_conv_1_time_dim'],
               'input_add_feature_dim': True,
               'n_out': 12,
               'padding': 'valid',
               'strides': 16},
    'flatten_conv': {'axes': 'except_time',
                     'class': 'merge_dims',
                     'from': ['conv_2'],
                     'n_out': 12},
    'window_1': {'class': 'window',
                 'from': ['flatten_conv'],
                 'window_size': 17},
    'flatten_window': {'axes': 'except_time',
                       'class': 'merge_dims',
                       'from': ['window_1']},
    'output': {'activation': None,
               'class': 'linear',
               'from': ['flatten_window'],
               'n_out': n_out},
  }
  config = Config({
    "num_outputs": n_out,
    "num_inputs": n_in,
    "debug_print_layer_output_template": True
  })
  with make_scope() as session:
    net = TFNetwork(config=config)
    print("extern data:")
    print(net.extern_data)
    # The construction itself is also the test.
    net.construct_from_dict(net_dict)
    out = net.get_default_output_layer()
    # Maybe this will not be the case in the future anymore;
    # however, if this test runs on CPU, currently the feature_dim_axis should always stay the default.
    # See also test_ConvLayer_feature_dim_unspecified.
    assert out.output.feature_dim_axis_or_unspecified is NotSpecified


def test_ConvLayer_feature_dim_unspecified():
  n_in = 1
  n_out = 13
  net_dict = {
    'output': {'activation': 'abs',
               'class': 'conv',
               'filter_size': (4,),
               'n_out': 64,
               'padding': 'valid',
               'strides': 10}}
  config = Config({
    "num_outputs": n_out,
    "num_inputs": n_in,
    "debug_print_layer_output_template": True
  })
  with make_scope() as session:
    net = TFNetwork(config=config)
    print("extern data:")
    print(net.extern_data)
    net.construct_from_dict(net_dict)
    out = net.get_default_output_layer()
    # Maybe this will not be the case in the future anymore;
    # however, if this test runs on CPU, currently the feature_dim_axis should always stay the default.
    assert out.output.feature_dim_axis_or_unspecified is NotSpecified


def test_conv_layer_NCHW():
  with make_scope() as session:
    import numpy as np
    net = TFNetwork(extern_data=ExternData())
    with tf.variable_scope("src_nhwc"):
      src_nhwc = InternalLayer(name="src_nhwc", network=net, out_type={"dim": 16,
                                                                       "shape": (None, 16, 16),
                                                                       "batch_dim_axis": 0,
                                                                       "time_dim_axis": 1,
                                                                       "feature_dim_axis": 3,
                                                                       "sparse": False
                                                                       })
      src_nhwc.output.placeholder = tf.placeholder(shape=(None, None, 16, 16), dtype=tf.float32)
      src_nhwc.output.size_placeholder = {0: tf.placeholder(shape=(None,), dtype=tf.int32)}
    with tf.variable_scope("src_nchw"):
      src_nchw = InternalLayer(name="src_nchw", network=net, out_type={"dim": 16,
                                                                       "shape": (16, None, 16),
                                                                       "batch_dim_axis": 0,
                                                                       "time_dim_axis": 2,
                                                                       "feature_dim_axis": 1,
                                                                       "sparse": False
                                                                       })
      src_nchw.output.placeholder = tf.placeholder(shape=(None, 16, None, 16), dtype=tf.float32)
      src_nchw.output.size_placeholder = {1: tf.placeholder(shape=(None,), dtype=tf.int32)}

    filters = 64
    filter_size = (5, 5)
    strides = (1, 2)
    padding = "VALID"

    with tf.variable_scope("conv_nhwc_from_nhwc"):
      conv_nhwc_from_nhwc = ConvLayer(
        name="conv_nhwc_from_nhwc", network=net, n_out=filters, filter_size=filter_size,
        padding=padding, strides=strides, auto_use_channel_first=False, sources=[src_nhwc],
        output=ConvLayer.get_out_data_from_opts(name="conv_nhwc_from_nhwc", n_out=filters,
                                                filter_size=filter_size, padding=padding,
                                                auto_use_channel_first=False,
                                                network=net, sources=[src_nhwc]))
    with tf.variable_scope("conv_nchw_from_nhwc"):
      conv_nchw_from_nhwc = ConvLayer(
        name="conv_nchw_from_nhwc", network=net, n_out=filters, filter_size=filter_size,
        padding=padding, strides=strides, auto_use_channel_first=True, sources=[src_nhwc],
        output=ConvLayer.get_out_data_from_opts(name="conv_nchw_from_nhwc", n_out=filters,
                                                filter_size=filter_size, padding=padding,
                                                auto_use_channel_first=True,
                                                network=net, sources=[src_nhwc]))
    with tf.variable_scope("conv_nchw_from_nchw"):
      conv_nchw_from_nchw = ConvLayer(
        name="conv_nchw_from_nchw", network=net, n_out=filters, filter_size=filter_size,
        padding=padding, strides=strides, auto_use_channel_first=True, sources=[src_nchw],
        output=ConvLayer.get_out_data_from_opts(name="conv_nchw_from_nchw", n_out=filters,
                                                filter_size=filter_size, padding=padding,
                                                auto_use_channel_first=True,
                                                network=net, sources=[src_nchw]))
    tf.global_variables_initializer().run()
    out, seq_lens = session.run([conv_nhwc_from_nhwc.output.placeholder,
                                 conv_nhwc_from_nhwc.output.size_placeholder[0]],
                                feed_dict={src_nhwc.output.placeholder: np.random.rand(10, 10, 16, 16),
                                           src_nhwc.output.size_placeholder[0]: np.full(shape=(10,), fill_value=10)}
                                )
    print(out.shape)
    assert_equal(out.shape, (10, 6, 6, 64))
    print(seq_lens)
    time_dim_axis = 1 if TFUtil.is_gpu_available() else 0
    out, seq_lens = session.run([conv_nchw_from_nhwc.output.placeholder,
                                 conv_nchw_from_nhwc.output.size_placeholder[time_dim_axis]],
                                feed_dict={src_nhwc.output.placeholder: np.random.rand(10, 10, 16, 16),
                                           src_nhwc.output.size_placeholder[0]: np.full(shape=(10,), fill_value=10)
                                })
    print(out.shape)
    if time_dim_axis == 1:
      assert_equal(out.shape, (10, 64, 6, 6))
    else:
      assert_equal(out.shape, (10, 6, 6, 64))
    print(seq_lens)
    if TFUtil.is_gpu_available():
      out, seq_lens = session.run([conv_nchw_from_nchw.output.placeholder,
                                   conv_nchw_from_nchw.output.size_placeholder[1]],
                                  feed_dict={src_nchw.output.placeholder: np.random.rand(10, 16, 10, 16),
                                             src_nchw.output.size_placeholder[1]: np.full(shape=(10,), fill_value=10)
                                  })
      print(out.shape)
      assert_equal(out.shape, (10, 64, 6, 6))
      print(seq_lens)


def test_pool_layer_NCHW():
  with make_scope() as session:
    import numpy as np
    net = TFNetwork(extern_data=ExternData())
    with tf.variable_scope("src_nhwc"):
      src_nhwc = InternalLayer(name="src_nhwc", network=net, out_type={"dim": 16,
                                                                       "shape": (None, 16, 16),
                                                                       "batch_dim_axis": 0,
                                                                       "time_dim_axis": 1,
                                                                       "feature_dim_axis": 3,
                                                                       "sparse": False
                                                                       })
      src_nhwc.output.placeholder = tf.placeholder(shape=(None, None, 16, 16), dtype=tf.float32)
      src_nhwc.output.size_placeholder = {0: tf.placeholder(shape=(None,), dtype=tf.int32)}
    with tf.variable_scope("src_nchw"):
      src_nchw = InternalLayer(name="src_nchw", network=net, out_type={"dim": 16,
                                                                       "shape": (16, None, 16),
                                                                       "batch_dim_axis": 0,
                                                                       "time_dim_axis": 2,
                                                                       "feature_dim_axis": 1,
                                                                       "sparse": False
                                                                       })
      src_nchw.output.placeholder = tf.placeholder(shape=(None, 16, None, 16), dtype=tf.float32)
      src_nchw.output.size_placeholder = {1: tf.placeholder(shape=(None,), dtype=tf.int32)}

    pool_size = (5, 5)
    strides = (1, 2)
    padding = "VALID"

    with tf.variable_scope("pool_nhwc_from_nhwc"):
      pool_nhwc_from_nhwc = PoolLayer(
        name="pool_nhwc_from_nhwc", network=net, mode="max", pool_size=pool_size,
        padding=padding, strides=strides, use_channel_first=False, sources=[src_nhwc],
        output=PoolLayer.get_out_data_from_opts(name="pool_nhwc_from_nhwc",
                                                pool_size=pool_size, padding=padding,
                                                use_channel_first=False,
                                                network=net, sources=[src_nhwc]))
    with tf.variable_scope("pool_nchw_from_nhwc"):
      pool_nchw_from_nhwc = PoolLayer(
        name="pool_nchw_from_nhwc", network=net, mode="max", pool_size=pool_size,
        padding=padding, strides=strides, use_channel_first=True, sources=[src_nhwc],
        output=PoolLayer.get_out_data_from_opts(name="pool_nchw_from_nhwc",
                                                pool_size=pool_size, padding=padding,
                                                use_channel_first=True,
                                                network=net, sources=[src_nhwc]))
    with tf.variable_scope("pool_nchw_from_nchw"):
      pool_nchw_from_nchw = PoolLayer(
        name="pool_nchw_from_nchw", network=net, mode="max", pool_size=pool_size,
        padding=padding, strides=strides, use_channel_first=True, sources=[src_nchw],
        output=PoolLayer.get_out_data_from_opts(name="pool_nchw_from_nchw",
                                                pool_size=pool_size, padding=padding,
                                                use_channel_first=True,
                                                network=net, sources=[src_nchw]))
    with tf.variable_scope("pool_nhwc_from_nchw"):
      pool_nhwc_from_nchw = PoolLayer(
        name="pool_nhwc_from_nchw", network=net, mode="max", pool_size=pool_size,
        padding=padding, strides=strides, use_channel_first=False, sources=[src_nchw],
        output=PoolLayer.get_out_data_from_opts(name="pool_nhwc_from_nchw",
                                                pool_size=pool_size, padding=padding,
                                                use_channel_first=False,
                                                network=net, sources=[src_nchw]))
    tf.global_variables_initializer().run()
    out, seq_lens = session.run([pool_nhwc_from_nhwc.output.placeholder,
                                 pool_nhwc_from_nhwc.output.size_placeholder[0]],
                                feed_dict={src_nhwc.output.placeholder: np.random.rand(10, 11, 16, 16),
                                           src_nhwc.output.size_placeholder[0]: np.full(shape=(10,), fill_value=11)}
                                )
    print(out.shape)
    assert_equal(out.shape, (10, 7, 6, 16))
    print(seq_lens)
    time_dim_axis = 1 if TFUtil.is_gpu_available() else 0
    out, seq_lens = session.run([pool_nchw_from_nhwc.output.placeholder,
                                 pool_nchw_from_nhwc.output.size_placeholder[time_dim_axis]],
                                feed_dict={src_nhwc.output.placeholder: np.random.rand(10, 11, 16, 16),
                                           src_nhwc.output.size_placeholder[0]: np.full(shape=(10,), fill_value=11)
                                })
    print(out.shape)
    if time_dim_axis == 1:
      assert_equal(out.shape, (10, 16, 7, 6))
    else:
      assert_equal(out.shape, (10, 7, 6, 16))
    print(seq_lens)
    if TFUtil.is_gpu_available():
      out, seq_lens = session.run([pool_nchw_from_nchw.output.placeholder,
                                   pool_nchw_from_nchw.output.size_placeholder[1]],
                                  feed_dict={src_nchw.output.placeholder: np.random.rand(10, 16, 11, 16),
                                             src_nchw.output.size_placeholder[1]: np.full(shape=(10,), fill_value=11)
                                  })
      print(out.shape)
      assert_equal(out.shape, (10, 16, 7, 6))
      print(seq_lens)
    out, seq_lens = session.run([pool_nhwc_from_nchw.output.placeholder,
                                 pool_nhwc_from_nchw.output.size_placeholder[0]],
                                feed_dict={src_nchw.output.placeholder: np.random.rand(10, 16, 11, 16),
                                           src_nchw.output.size_placeholder[1]: np.full(shape=(10,), fill_value=11)}
                                )
    print(out.shape)
    assert_equal(out.shape, (10, 7, 6, 16))
    print(seq_lens)


def test_ResizeLayer_fill_value():
  with make_scope() as session:
    net = TFNetwork(extern_data=ExternData())
    src = InternalLayer(name="src", network=net, out_type={"dim": 20, "sparse": True})
    src.output.placeholder = tf.constant([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=tf.int32)
    src.output.size_placeholder = {0: tf.constant([5, 3], dtype=tf.int32)}
    layer = ResizeLayer(
      name="resize", network=net, factor=3, axis="T", kind="fill", fill_value=19, sources=[src],
      output=ResizeLayer.get_out_data_from_opts(
        name="resize", network=net, factor=3, axis="T", kind="fill", sources=[src]))
    out, seq_lens = session.run([layer.output.placeholder, layer.output.size_placeholder[0]])
    print(out)
    print(seq_lens)
    assert isinstance(out, numpy.ndarray)
    assert isinstance(seq_lens, numpy.ndarray)
    assert_equal(
      out.tolist(),
      [[1, 19, 19,  2, 19, 19,  3, 19, 19,  4, 19, 19,  5, 19, 19,],
       [6, 19, 19,  7, 19, 19,  8, 19, 19,  9, 19, 19, 10, 19, 19]])
    assert_equal(seq_lens.tolist(), [15, 9])


def test_ResizeLayer_fill_dropout():
  with make_scope() as session:
    net = TFNetwork(extern_data=ExternData())
    src = InternalLayer(name="src", network=net, out_type={"dim": 20, "sparse": True})
    src_seqs = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    src_seq_lens = [5, 3]
    factor = 3
    fill_value = 19
    src.output.placeholder = tf.constant(src_seqs, dtype=tf.int32)
    src.output.size_placeholder = {0: tf.constant(src_seq_lens, dtype=tf.int32)}
    layer = ResizeLayer(
      name="resize", network=net,
      factor=factor, axis="T", kind="fill", fill_value=fill_value, fill_dropout=0.5, sources=[src],
      output=ResizeLayer.get_out_data_from_opts(
        name="resize", network=net, factor=factor, axis="T", kind="fill", sources=[src]))
    out, seq_lens = session.run([layer.output.placeholder, layer.output.size_placeholder[0]])
    print(out)
    print(seq_lens)
    assert isinstance(out, numpy.ndarray)
    assert isinstance(seq_lens, numpy.ndarray)
    # Non-deterministic output. But we can check some constraints.
    for i in range(len(src_seq_lens)):
      assert src_seq_lens[i] <= seq_lens[i] <= src_seq_lens[i] * factor
      assert_equal([s for s in out[i] if s != fill_value], src_seqs[i])


def test_DotLayer():
  with make_scope() as session:
    B = 2
    H = 3
    D = H * 5
    net = TFNetwork(extern_data=ExternData())
    a = InternalLayer(name="A", network=net, out_type={"shape": (None, H, D // H)})
    assert a.output.batch_dim_axis == 0
    assert a.output.time_dim_axis == 1
    assert a.output.shape == (None, H, D // H)
    assert a.output.dim == D // H
    a_seq_lens = [7, 3]
    assert len(a_seq_lens) == B
    a.output.placeholder = tf.reshape(
      tf.range(B * max(a_seq_lens) * D, dtype=tf.float32), (B, max(a_seq_lens), H, D // H))
    a.output.size_placeholder = {0: tf.constant(a_seq_lens, dtype=tf.int32)}
    b = InternalLayer(name="B", network=net, out_type={"shape": (H, D // H)})
    assert b.output.batch_dim_axis == 0
    assert b.output.shape == (H, D // H)
    assert b.output.dim == D // H
    b.output.placeholder = tf.reshape(tf.add(tf.range(B * D, dtype=tf.float32), 0.5), (B, H, D // H))
    kwargs = dict(
      name="dot", network=net, sources=[a, b], debug=True,
      red1=-1, red2=-1, var1="T", var2=None)
    layer = DotLayer(output=DotLayer.get_out_data_from_opts(**kwargs), **kwargs)
    print(layer, layer.output)
    assert layer.output.batch_dim_axis == 0
    assert layer.output.time_dim_axis == 2
    assert layer.output.shape == (H, None, 1)
    assert layer.output.dim == 1
    out, seq_lens = session.run([layer.output.placeholder, layer.output.size_placeholder[1]])
    print(out)
    print(seq_lens)
    assert isinstance(out, numpy.ndarray)
    assert isinstance(seq_lens, numpy.ndarray)
    assert_equal(seq_lens.tolist(), a_seq_lens)
    assert_equal(out.shape, (B, H, max(a_seq_lens), 1))


def test_subnet_load_on_init():
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
        "l1": {"class": "linear", "activation": None, "n_out": n_hidden},
        "output": {"class": "linear", "activation": None, "n_out": n_out, "from": ["l1"]}
      }
    })
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])
    network.initialize_params(session)
    params_orig_dump = network.get_params_serialized(session)
    print("l1:")
    print(params_orig_dump.values_dict["l1"]["W"])
    print("output:")
    print(params_orig_dump.values_dict["output"]["W"])
    assert(params_orig_dump.values_dict["l1"]["W"].any())
    assert(params_orig_dump.values_dict["output"]["W"].any())
    network.save_params_to_file(filename=model_filename, session=session)

  with make_scope() as session:
    config = Config()
    config.update({
      "num_outputs": n_out,
      "num_inputs": n_in,
      "network": {
        "l0": {"class": "linear", "activation": None, "n_out": n_in},
        "subnet": {"class": "subnetwork", "from": ["l0"], "load_on_init": model_filename, "subnetwork": {
          "l1": {"class": "linear", "activation": None, "n_out": n_hidden},
          "output": {"class": "linear", "activation": None, "n_out": n_out, "from": ["l1"]}
        }},
        "output": {"class": "linear", "activation": None, "n_out": n_out, "from": ["subnet"]}
      }
    })
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])
    network.initialize_params(session)
    params_dump = network.get_params_serialized(session)
    params_dump_subnet = params_dump.values_dict["subnet"]
    for layer_name in ["l1", "output"]:
      layer_orig = params_orig_dump.values_dict[layer_name]
      for param_name in ["W", "b"]:
        param_orig = layer_orig[param_name]
        param_subnet = params_dump_subnet[layer_name + "/" + param_name]
        numpy.testing.assert_array_equal(param_orig, param_subnet)


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
