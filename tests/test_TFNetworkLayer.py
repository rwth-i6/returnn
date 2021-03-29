
# start: nosetests $this_file --nologcapture
from __future__ import division

import _setup_test_env  # noqa
import logging
import os
import tensorflow as tf
import sys
from nose.tools import assert_equal, assert_is_instance
import contextlib
import unittest
import numpy.testing
from pprint import pprint
from returnn.util import better_exchook
from returnn.config import Config
from returnn.tf.network import *
from returnn.tf.layers.basic import *
from returnn.log import log
import returnn.tf.compat as tf_compat
import returnn.tf.util.basic as tf_util

print("TF version:", tf.__version__)
print("Numpy version:", numpy.__version__)


@contextlib.contextmanager
def make_scope():
  """
  :rtype: tf.compat.v1.Session
  """
  with tf.Graph().as_default() as graph:
    with tf_compat.v1.Session(graph=graph) as session:
      yield session


def make_feed_dict(data_list, same_time=False, n_batch=3, n_time=7):
  """
  :param list[returnn.tf.util.data.Data]|ExternData data_list:
  :param bool same_time:
  :param int n_batch:
  :param int n_time:
  :rtype: dict[tf.Tensor,numpy.ndarray]
  """
  if isinstance(data_list, ExternData):
    data_list = [value for (key, value) in sorted(data_list.data.items())]
  assert n_time > 0 and n_batch > 0
  rnd = numpy.random.RandomState(42)
  existing_sizes = {}  # type: typing.Dict[tf.Tensor,int]
  d = {}
  for data in data_list:
    shape = list(data.batch_shape)
    if data.batch_dim_axis is not None:
      shape[data.batch_dim_axis] = n_batch
    for axis, dim in enumerate(shape):
      if dim is None:
        axis_wo_b = data.get_batch_axis_excluding_batch(axis)
        assert axis_wo_b in data.size_placeholder
        dyn_size = data.size_placeholder[axis_wo_b]
        if dyn_size in existing_sizes:
          shape[axis] = existing_sizes[dyn_size]
          continue
        existing_sizes[dyn_size] = n_time
        shape[axis] = n_time
        dyn_size_v = numpy.array([n_time, max(n_time - 2, 1), max(n_time - 3, 1)])
        if dyn_size_v.shape[0] > n_batch:
          dyn_size_v = dyn_size_v[:n_batch]
        elif dyn_size_v.shape[0] < n_batch:
          dyn_size_v = numpy.concatenate(
            [dyn_size_v, rnd.randint(1, n_time + 1, size=(n_batch - dyn_size_v.shape[0],))], axis=0)
        d[dyn_size] = dyn_size_v
        if not same_time:
          n_time += 1
    print("%r %r: shape %r" % (data, data.placeholder, shape))
    if data.sparse:
      d[data.placeholder] = rnd.randint(0, data.dim or 13, size=shape, dtype=data.dtype)
    else:
      d[data.placeholder] = rnd.normal(size=shape).astype(data.dtype)
  return d


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


def test_concat_sources_dim1():
  with make_scope() as session:
    net_dict = {
      "lin1": {"class": "linear", "activation": "sigmoid", "n_out": 5},
      "lin2": {"class": "linear", "activation": "sigmoid", "n_out": 1},
      "concat": {"class": "copy", "from": ["lin1", "lin2"]},
      "output": {"class": "softmax", "loss": "ce", "from": "concat"}
    }
    config = Config({"debug_print_layer_output_template": True})
    config.update(dict(num_inputs=4, num_outputs=9))
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(net_dict)
    assert_equal(network.get_layer("concat").output.shape, (None, 6))
    out = network.get_default_output_layer()
    assert out.output.shape == (None, 9)
    feed_dict = make_feed_dict(network.extern_data.data.values(), same_time=True)
    session.run(tf_compat.v1.global_variables_initializer())
    session.run(out.output.placeholder, feed_dict=feed_dict)


def test_LinearLayer_batch_feature_major():
  with make_scope() as session:
    network = TFNetwork(config=Config(), extern_data=ExternData(), train_flag=True)
    n_in = 3
    n_out = 7
    source = InternalLayer(
      name="source", network=network, output=Data(
        name="source", shape=(n_in, None), time_dim_axis=2, auto_create_placeholders=True))
    assert source.output.feature_dim_axis == 1
    assert source.output.is_batch_feature_major
    out_template = LinearLayer.get_out_data_from_opts(
      name="lin", network=network, n_out=n_out, activation=None, sources=[source])
    out_template.sanity_check()
    assert out_template.shape == (n_out, None) and (out_template.feature_dim_axis, out_template.time_dim_axis) == (1, 2)
    assert out_template.is_batch_feature_major
    with tf_compat.v1.variable_scope("lin"):
      layer = LinearLayer(
        name="lin", network=network, n_out=n_out, activation=None, sources=[source], output=out_template)
    layer.output.sanity_check()
    n_batch = 5
    n_times = [13, 13, 11, 7, 5]
    assert len(n_times) == n_batch
    n_time = max(n_times)
    feed_dict = {
      source.output.placeholder: numpy.random.normal(size=(n_batch, n_in, n_time)).astype("float32"),
      source.output.size_placeholder[1]: numpy.array(n_times, dtype="int32")}
    session.run(tf_compat.v1.global_variables_initializer())
    session.run(layer.output.placeholder, feed_dict=feed_dict)


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


def test_batch_norm():
  with make_scope() as session:
    import numpy as np
    net = TFNetwork(extern_data=ExternData())
    net.train_flag = True
    with tf_compat.v1.variable_scope("src_nchw"):
      src_nhwc = InternalLayer(name="src_nchw", network=net, out_type={"dim": 16,
                                                                       "shape": (None, 16, 16),
                                                                       "batch_dim_axis": 0,
                                                                       "time_dim_axis": 1,
                                                                       "feature_dim_axis": 3,
                                                                       "sparse": False
                                                                       })
      src_nhwc.output.placeholder = tf_compat.v1.placeholder(shape=(None, None, 16, 16), dtype=tf.float32)
      src_nhwc.output.size_placeholder = {0: tf_compat.v1.placeholder(shape=(None,), dtype=tf.int32)}

    rnd = np.random.RandomState(42)
    mean =  tf.constant(rnd.rand(1, 1, 1, 16), name="rand_mean", dtype=tf.float32)
    variance = tf.constant(rnd.rand(1, 1, 1, 16), name="rand_var", dtype=tf.float32)
    input_data = rnd.rand(10, 11, 16, 16)
    seq_lens = np.array([11, 11, 11, 11, 11, 11, 11, 11, 11, 11])

    with tf_compat.v1.variable_scope("batch_norm_masked_nchw"):
      batch_norm_1 = BatchNormLayer(name="batch_norm_masked_nchw", network=net, masked_time=True,
                                    sample_mean=mean, sample_variance=variance,
                                    sources=[src_nhwc],
                                    output=BatchNormLayer.get_out_data_from_opts(name="batch_norm_masked_nchw",
                                                                                 sources=[src_nhwc],
                                                                                 network=net))
      batch_norm_1.post_init(layer_desc=None)
    with tf_compat.v1.variable_scope("batch_norm_nonmasked_nchw"):
      batch_norm_2 = BatchNormLayer(name="batch_norm_nonmasked_nchw", network=net, masked_time=False,
                                    sample_mean=mean, sample_variance=variance,
                                    sources=[src_nhwc],
                                    output=BatchNormLayer.get_out_data_from_opts(name="batch_norm_nonmasked_nchw",
                                                                                 sources=[src_nhwc],
                                                                                 network=net))
      batch_norm_2.post_init(layer_desc=None)
    tf_compat.v1.global_variables_initializer().run()
    out_1, seq_lens_1 = session.run([batch_norm_1.output.placeholder,
                                 batch_norm_1.output.size_placeholder[0]],
                                feed_dict={src_nhwc.output.placeholder: input_data,
                                           src_nhwc.output.size_placeholder[0]: seq_lens}
                                )
    out_2, seq_lens_2 = session.run([batch_norm_2.output.placeholder,
                                 batch_norm_2.output.size_placeholder[0]],
                                feed_dict={src_nhwc.output.placeholder: input_data,
                                           src_nhwc.output.size_placeholder[0]: seq_lens}
                                )
    assert np.array_equal(out_1, out_2)
    print(np.sum(out_1 - out_2))


def test_batch_norm_unequal_seq_len():
  with make_scope() as session:
    import numpy as np
    import numpy.testing as npt
    net = TFNetwork(extern_data=ExternData())
    net.train_flag = True
    with tf_compat.v1.variable_scope("src_nhwc"):
      src_nhwc = InternalLayer(name="src_nhwc", network=net, out_type={"dim": 16,
                                                                       "shape": (None, 16, 16),
                                                                       "batch_dim_axis": 0,
                                                                       "time_dim_axis": 1,
                                                                       "feature_dim_axis": 3,
                                                                       "sparse": False
                                                                       })
      src_nhwc.output.placeholder = tf_compat.v1.placeholder(shape=(None, None, 16, 16), dtype=tf.float32)
      src_nhwc.output.size_placeholder = {0: tf_compat.v1.placeholder(shape=(None,), dtype=tf.int32)}

    rnd = np.random.RandomState(42)
    mean = tf.constant(rnd.rand(1, 1, 1, 16), name="rand_mean", dtype=tf.float32)
    variance = tf.constant(rnd.rand(1, 1, 1, 16), name="rand_var", dtype=tf.float32)
    input_data = rnd.rand(10, 11, 16, 16).astype('f')
    input_data[2, 5:, :, :] = 0
    data_mean = np.mean(input_data, axis=(0, 1, 2), keepdims=True, dtype=np.float32)
    data_var = np.var(input_data, axis=(0, 1, 2), keepdims=True, dtype=np.float32)
    input_data_masked = np.copy(input_data)
    seq_lens = np.array([11, 11, 5, 11, 11, 11, 11, 11, 11, 11], dtype=np.float32)
    n1 = 9 * 11 * 16 + 5 * 16
    n2 = 10 * 11 * 16

    with tf_compat.v1.variable_scope("batch_norm_masked_nchw"):
      batch_norm_1 = BatchNormLayer(name="batch_norm_masked_nchw", network=net, masked_time=True,
                                    sample_mean=mean, sample_variance=variance,
                                    use_shift=False, use_std=False, epsilon=0.0,
                                    sources=[src_nhwc],
                                    output=BatchNormLayer.get_out_data_from_opts(name="batch_norm_masked_nchw",
                                                                                 sources=[src_nhwc],
                                                                                 network=net))
      batch_norm_1.post_init(layer_desc=None)
    with tf_compat.v1.variable_scope("batch_norm_nonmasked_nchw"):
      batch_norm_2 = BatchNormLayer(name="batch_norm_nonmasked_nchw", network=net, masked_time=False,
                                    sample_mean=mean, sample_variance=variance,
                                    use_shift=False, use_std=False, epsilon=0,
                                    sources=[src_nhwc],
                                    output=BatchNormLayer.get_out_data_from_opts(name="batch_norm_nonmasked_nchw",
                                                                                 sources=[src_nhwc],
                                                                                 network=net))
      batch_norm_2.post_init(layer_desc=None)
    tf_compat.v1.global_variables_initializer().run()
    out_1, seq_lens_1 = session.run([batch_norm_1.output.placeholder,
                                     batch_norm_1.output.size_placeholder[0]],
                                    feed_dict={src_nhwc.output.placeholder: input_data,
                                               src_nhwc.output.size_placeholder[0]: seq_lens}
                                    )
    out_2, seq_lens_2 = session.run([batch_norm_2.output.placeholder,
                                 batch_norm_2.output.size_placeholder[0]],
                                feed_dict={src_nhwc.output.placeholder: input_data_masked,
                                           src_nhwc.output.size_placeholder[0]: seq_lens}
                                )
    # Manually calculating batch_norm and compare to the tf output
    np_bn2 = (input_data - data_mean) * (1.0 / np.sqrt(data_var))
    npt.assert_array_almost_equal(np_bn2, out_2, decimal=5)
    # Manually calculating batch_norm with different seq_lens, having:
    # Mean_1 = n2 / n1 * Mean_2
    # Var_1 = n2 / n1 * (Var_2 + Mean_2 ^ 2 (1 - n2 / n1))
    # bn_1 = (x - Mean_1) * 1 / sqrt(Var_1)
    # Substituting Mean_1 and Var_1:
    np_bn1 = (input_data - n2 / n1 * data_mean) * \
             (1.0 / np.sqrt(n2 / n1 * (data_var + data_mean ** 2 * (1 - n2 / n1))))
    # Check with tf output.
    npt.assert_array_almost_equal(np_bn1, out_1, decimal=5)


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
    session.run(tf_compat.v1.global_variables_initializer())
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


def _test_simple_eval_func(s):
  with make_scope() as session:
    num_inputs = 2
    config = Config()
    config.update({
      "extern_data": {"data": {"dim": num_inputs}},
      "network": {
        "output": {"class": "eval", "eval": "%s(source(0))" % s, "from": "data"}
      }})
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_value("network"))
    feed = make_feed_dict(network.extern_data)
    out = network.get_default_output_layer().output.placeholder
    session.run(out, feed_dict=feed)


def test_simple_eval_tanh():
  _test_simple_eval_func("tf.tanh")


def test_simple_eval_sigmoid():
  _test_simple_eval_func("tf.sigmoid")


def _test_simple_activation(s):
  with make_scope() as session:
    num_inputs = 2
    config = Config()
    config.update({
      "extern_data": {"data": {"dim": num_inputs}},
      "network": {
        "output": {"class": "activation", "activation": s, "from": "data"}
      }})
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_value("network"))
    feed = make_feed_dict(network.extern_data)
    out = network.get_default_output_layer().output.placeholder
    session.run(out, feed_dict=feed)


def test_simple_activation_tanh():
  _test_simple_activation("tanh")


def test_simple_activation_log_sigmoid():
  _test_simple_activation("log_sigmoid")


def test_cnn_building_block():
  with make_scope() as session:
    num_inputs = 192
    channel_num = 32
    feature_dim = 6
    filters = 32
    filter_size = (3, 3)
    config = Config()
    config.update({
      "num_inputs": num_inputs,
      "num_outputs": filters,
      "network": {
        "split": {"class": "split_dims", "axis": "f", "dims": (channel_num, feature_dim), "from": ["data"]},
        "swap_axes": {"class": "swap_axes", "axis1": "s:1", "axis2": "f", "from": ["split"]},
        "c1": {"class": "conv", "n_out": filters, "filter_size": filter_size, "auto_use_channel_first": False,
               "strides": (1, 1), "dilation_rate": (1, 1), "padding": "SAME", "activation": None, "with_bias": False,
               "from": "swap_axes"},
        "bn1": {"class": "batch_norm", "from": "c1"},
        "y1": {"class": "activation", "activation": "relu", "batch_norm": False, "from": "bn1"},
        "c2": {"class": "conv", "n_out": filters, "filter_size": filter_size, "auto_use_channel_first": False,
               "strides": (1, 1), "dilation_rate": (1, 1), "padding": "SAME", "activation": None, "with_bias": False,
               "from": "y1"},
        "p": {"class": "combine", "kind": "add", "from": ["c2", "swap_axes"]},
        "bn2": {"class": "batch_norm", "from": "p"},
        "y2": {"class": "activation", "activation": "relu", "batch_norm": False, "from": "bn2"},

        "out_pool": {"class": "reduce", "mode": "avg", "axes": "s:1", "keep_dims": False, "from": "y2"},
        "output": {"class": "copy", "from": ["out_pool"], "is_output_layer": True}
      }})
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_value("network"))
    session.run(tf_compat.v1.global_variables_initializer())
    out = network.layers["output"].output.placeholder
    n_batch = 5
    seq_len = 10
    seq_lens = numpy.array([10, 10, 10, 10, 10], dtype=numpy.int32)
    feed = {network.extern_data.get_default_input_data().placeholder:
            numpy.random.rand(n_batch, seq_len, num_inputs).astype('f'),
            network.extern_data.get_default_input_data().size_placeholder[0]: seq_lens}
    v = session.run(out, feed_dict=feed)


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


def test_CombineLayer_simple_add():
  with make_scope() as session:
    net_dict = {
      "lin1": {"class": "linear", "activation": "sigmoid", "n_out": 5},
      "lin2": {"class": "linear", "activation": "sigmoid", "n_out": 5},
      "combine": {"class": "combine", "kind": "add", "from": ["lin1", "lin2"]},
      "output": {"class": "softmax", "loss": "ce", "from": "combine"}
    }
    config = Config({"debug_print_layer_output_template": True})
    config.update(dict(num_inputs=4, num_outputs=9))
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(net_dict)
    out = network.get_default_output_layer()
    feed_dict = make_feed_dict(network.extern_data.data.values(), same_time=True)
    session.run(tf_compat.v1.global_variables_initializer())
    session.run(out.output.placeholder, feed_dict=feed_dict)


def test_CombineLayer_broadcast():
  with make_scope() as session:
    net_dict = {
      "lin1": {"class": "linear", "activation": "sigmoid", "n_out": 5},
      "lin2": {"class": "linear", "activation": "sigmoid", "n_out": 1},
      "combine": {"class": "combine", "kind": "add", "from": ["lin1", "lin2"]},
      "output": {"class": "softmax", "loss": "ce", "from": "combine"}
    }
    config = Config({"debug_print_layer_output_template": True})
    config.update(dict(num_inputs=4, num_outputs=9))
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(net_dict)
    assert_equal(network.get_layer("combine").output.shape, (None, 5))
    out = network.get_default_output_layer()
    assert out.output.shape == (None, 9)
    feed_dict = make_feed_dict(network.extern_data.data.values(), same_time=True)
    session.run(tf_compat.v1.global_variables_initializer())
    session.run(out.output.placeholder, feed_dict=feed_dict)


def test_CombineLayer_broadcast_multiple():
  with make_scope() as session:
    net_dict = {
      "p1": {"class": "variable", "shape": (5, 5, 3), "add_batch_axis": False},
      "p2": {"class": "variable", "shape": (5, 1, 1), "add_batch_axis": False},
      "combine": {"class": "combine", "kind": "add", "from": ["p1", "p2"]},
      "output": {"class": "softmax", "loss": "ce", "from": "combine"}
    }
    config = Config({"debug_print_layer_output_template": True})
    config.update(dict(num_inputs=4, num_outputs=9))
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(net_dict)
    assert_equal(network.get_layer("combine").output.batch_shape, (5, 5, 3))
    out = network.get_default_output_layer()
    assert out.output.batch_shape == (5, 5, 9) and not out.output.have_batch_axis()
    feed_dict = make_feed_dict(network.extern_data.data.values(), same_time=True)
    session.run(tf_compat.v1.global_variables_initializer())
    out_v = session.run(out.output.placeholder, feed_dict=feed_dict)
    assert out_v.shape == out.output.batch_shape


def test_CombineLayer_different_batch_axis():
  # ["base:enc_ctx", "weight_feedback", "s_transformed"]
  # base:enc_ctx: Data(name='enc_ctx_output', shape=(None, 14), batch_dim_axis=1)
  # weight_feedback: Data(name='weight_feedback_output', shape=(None, 14), batch_dim_axis=1)
  # s_transformed: Data(name='s_transformed_output', shape=(14,), time_dim_axis=None)
  # out: Data(name='energy_in_output', shape=(None, 14), beam_size=3)
  with make_scope() as session:
    config = Config({"debug_print_layer_output_template": True})
    net = TFNetwork(config=config, extern_data=ExternData(), train_flag=True)
    n_dim = 7
    l1 = net.add_layer(
      name="enc_ctx", layer_class=InternalLayer,
      output=Data(name='enc_ctx_output', shape=(None, n_dim), auto_create_placeholders=True))
    l2 = net.add_layer(
      name="weight_feedback", layer_class=InternalLayer,
      output=Data(name='weight_feedback_output', shape=(None, n_dim), batch_dim_axis=1, auto_create_placeholders=True))
    l3 = net.add_layer(
      name="s_transformed", layer_class=InternalLayer,
      output=Data(name='s_transformed_output', shape=(n_dim,), time_dim_axis=None, auto_create_placeholders=True))
    out = net.add_layer(name="energy_in", layer_class=CombineLayer, kind="add", sources=[l1, l2, l3])
    print("out:", out)
    n_batch = 3
    n_time = 5
    session.run(out.output.placeholder, {
      l1.output.placeholder: numpy.random.normal(size=(n_batch, n_time, n_dim)).astype("float32"),
      l1.output.size_placeholder[0]: numpy.array([n_time] * 3),
      l2.output.placeholder: numpy.random.normal(size=(n_time, n_batch, n_dim)).astype("float32"),
      l2.output.size_placeholder[0]: numpy.array([n_time] * 3),
      l3.output.placeholder: numpy.random.normal(size=(n_batch, n_dim))})


def test_CombineLayer_two_time_dims():
  with make_scope() as session:
    n_dim = 5
    n_batch = 3
    n_time1 = 7
    n_time2 = 11
    rnd = numpy.random.RandomState(42)
    net_dict = {
      "output": {
        "class": "combine", "kind": "add",
        "from": ["data:in0", "data:in1", "data:in2"]}
    }
    config = Config({"debug_print_layer_output_template": True})
    extern_data = ExternData()
    in0 = Data(
      name="in0", shape=(None, None, n_dim), batch_dim_axis=1, auto_create_placeholders=True)
    in1 = Data(
      # same time as first in in0
      name="in1", shape=(None, n_dim), auto_create_placeholders=True)
    in2 = Data(
      # same time as in second in in0
      name="in2", shape=(None, n_dim), batch_dim_axis=1, auto_create_placeholders=True)
    extern_data.register_data(in0)
    extern_data.register_data(in1)
    extern_data.register_data(in2)
    in1.get_size_dim_tag(0).declare_same_as(in0.get_size_dim_tag(0))
    in2.get_size_dim_tag(0).declare_same_as(in0.get_size_dim_tag(1))
    print("ExternData all dimension tags (allow_same_feature_dim=True):")
    pprint(extern_data.get_all_dimension_tags(allow_same_feature_dim=True))
    network = TFNetwork(config=config, extern_data=extern_data, train_flag=True)
    network.construct_from_dict(net_dict)
    output = network.get_default_output_layer().output
    assert output.shape == (None, None, n_dim) and set(output.size_placeholder.keys()) == {0, 1}
    assert output.batch_dim_axis == 1 and output.time_dim_axis == 0
    time1_np = numpy.array([n_time1, n_time1 - 3, n_time1 - 2])
    assert min(time1_np) > 0 and max(time1_np) == n_time1 and len(time1_np) == n_batch
    time2_np = numpy.array([n_time2, n_time2 - 2, n_time2 - 5])
    assert min(time2_np) > 0 and max(time2_np) == n_time2 and len(time2_np) == n_batch
    in0_np = rnd.normal(size=(n_time1, n_batch, n_time2, n_dim)).astype("float32")
    in1_np = rnd.normal(size=(n_batch, n_time1, n_dim)).astype("float32")
    in2_np = rnd.normal(size=(n_time2, n_batch, n_dim)).astype("float32")
    out_np, out_sizes_np = session.run(
      fetches=(output.placeholder, output.size_placeholder),
      feed_dict={
        in0.placeholder: in0_np, in0.size_placeholder[0]: time1_np, in0.size_placeholder[1]: time2_np,
        in1.placeholder: in1_np, in1.size_placeholder[0]: time1_np,
        in2.placeholder: in2_np, in2.size_placeholder[0]: time2_np})
    assert isinstance(out_np, numpy.ndarray)
    assert isinstance(out_sizes_np, dict) and set(out_sizes_np.keys()) == {0, 1}
    out_time0_np, out_time1_np = out_sizes_np[0], out_sizes_np[1]
    assert isinstance(out_time0_np, numpy.ndarray) and isinstance(out_time1_np, numpy.ndarray)
    assert out_np.shape == (n_time1, n_batch, n_time2, n_dim)


def test_CombineLayer_two_time_dims_first_not_most_generic():
  with make_scope() as session:
    n_dim = 5
    n_batch = 3
    n_time1 = 7
    n_time2 = 11
    rnd = numpy.random.RandomState(42)
    net_dict = {
      "output": {
        "class": "combine", "kind": "add",
        "from": ["data:in1", "data:in0", "data:in2"]}
    }
    config = Config({"debug_print_layer_output_template": True})
    extern_data = ExternData()
    in0 = Data(
      name="in0", shape=(None, None, n_dim), batch_dim_axis=1, auto_create_placeholders=True)
    in1 = Data(
      # same time as first in in0
      name="in1", shape=(None, n_dim), auto_create_placeholders=True)
    in2 = Data(
      # same time as in second in in0
      name="in2", shape=(None, n_dim), batch_dim_axis=1, auto_create_placeholders=True)
    extern_data.register_data(in0)
    extern_data.register_data(in1)
    extern_data.register_data(in2)
    in1.get_size_dim_tag(0).declare_same_as(in0.get_size_dim_tag(0))
    in2.get_size_dim_tag(0).declare_same_as(in0.get_size_dim_tag(1))
    print("ExternData all dimension tags (allow_same_feature_dim=True):")
    pprint(extern_data.get_all_dimension_tags(allow_same_feature_dim=True))
    network = TFNetwork(config=config, extern_data=extern_data, train_flag=True)
    network.construct_from_dict(net_dict)
    output = network.get_default_output_layer().output
    assert output.shape == (None, None, n_dim) and set(output.size_placeholder.keys()) == {0, 1}
    assert output.batch_dim_axis == 1 and output.time_dim_axis == 0
    time1_np = numpy.array([n_time1, n_time1 - 3, n_time1 - 2])
    assert min(time1_np) > 0 and max(time1_np) == n_time1 and len(time1_np) == n_batch
    time2_np = numpy.array([n_time2, n_time2 - 2, n_time2 - 5])
    assert min(time2_np) > 0 and max(time2_np) == n_time2 and len(time2_np) == n_batch
    in0_np = rnd.normal(size=(n_time1, n_batch, n_time2, n_dim)).astype("float32")
    in1_np = rnd.normal(size=(n_batch, n_time1, n_dim)).astype("float32")
    in2_np = rnd.normal(size=(n_time2, n_batch, n_dim)).astype("float32")
    out_np, out_sizes_np = session.run(
      fetches=(output.placeholder, output.size_placeholder),
      feed_dict={
        in0.placeholder: in0_np, in0.size_placeholder[0]: time1_np, in0.size_placeholder[1]: time2_np,
        in1.placeholder: in1_np, in1.size_placeholder[0]: time1_np,
        in2.placeholder: in2_np, in2.size_placeholder[0]: time2_np})
    assert isinstance(out_np, numpy.ndarray)
    assert isinstance(out_sizes_np, dict) and set(out_sizes_np.keys()) == {0, 1}
    out_time0_np, out_time1_np = out_sizes_np[0], out_sizes_np[1]
    assert isinstance(out_time0_np, numpy.ndarray) and isinstance(out_time1_np, numpy.ndarray)
    assert out_np.shape == (n_time1, n_batch, n_time2, n_dim)


def test_CombineLayer_two_time_dims_first_not_most_generic_with_n_out():
  with make_scope() as session:
    n_dim = 5
    n_batch = 3
    n_time1 = 7
    n_time2 = 11
    rnd = numpy.random.RandomState(42)
    net_dict = {
      "output": {
        "class": "combine", "kind": "add", "n_out": n_dim,
        "from": ["data:in1", "data:in0", "data:in2"]}
    }
    config = Config({"debug_print_layer_output_template": True})
    extern_data = ExternData()
    in0 = Data(
      name="in0", shape=(None, None, n_dim), batch_dim_axis=1, auto_create_placeholders=True)
    in1 = Data(
      # same time as first in in0
      name="in1", shape=(None, n_dim), auto_create_placeholders=True)
    in2 = Data(
      # same time as in second in in0
      name="in2", shape=(None, n_dim), batch_dim_axis=1, auto_create_placeholders=True)
    extern_data.register_data(in0)
    extern_data.register_data(in1)
    extern_data.register_data(in2)
    in1.get_size_dim_tag(0).declare_same_as(in0.get_size_dim_tag(0))
    in2.get_size_dim_tag(0).declare_same_as(in0.get_size_dim_tag(1))
    print("ExternData all dimension tags (allow_same_feature_dim=True):")
    pprint(extern_data.get_all_dimension_tags(allow_same_feature_dim=True))
    network = TFNetwork(config=config, extern_data=extern_data, train_flag=True)
    network.construct_from_dict(net_dict)
    output = network.get_default_output_layer().output
    assert output.shape == (None, None, n_dim) and set(output.size_placeholder.keys()) == {0, 1}
    assert output.batch_dim_axis == 1 and output.time_dim_axis == 0
    time1_np = numpy.array([n_time1, n_time1 - 3, n_time1 - 2])
    assert min(time1_np) > 0 and max(time1_np) == n_time1 and len(time1_np) == n_batch
    time2_np = numpy.array([n_time2, n_time2 - 2, n_time2 - 5])
    assert min(time2_np) > 0 and max(time2_np) == n_time2 and len(time2_np) == n_batch
    in0_np = rnd.normal(size=(n_time1, n_batch, n_time2, n_dim)).astype("float32")
    in1_np = rnd.normal(size=(n_batch, n_time1, n_dim)).astype("float32")
    in2_np = rnd.normal(size=(n_time2, n_batch, n_dim)).astype("float32")
    out_np, out_sizes_np = session.run(
      fetches=(output.placeholder, output.size_placeholder),
      feed_dict={
        in0.placeholder: in0_np, in0.size_placeholder[0]: time1_np, in0.size_placeholder[1]: time2_np,
        in1.placeholder: in1_np, in1.size_placeholder[0]: time1_np,
        in2.placeholder: in2_np, in2.size_placeholder[0]: time2_np})
    assert isinstance(out_np, numpy.ndarray)
    assert isinstance(out_sizes_np, dict) and set(out_sizes_np.keys()) == {0, 1}
    out_time0_np, out_time1_np = out_sizes_np[0], out_sizes_np[1]
    assert isinstance(out_time0_np, numpy.ndarray) and isinstance(out_time1_np, numpy.ndarray)
    assert out_np.shape == (n_time1, n_batch, n_time2, n_dim)


def test_CombineLayer_time_broadcast():
  with make_scope() as session:
    n_batch, n_time, n_features = 3, 7, 5
    net_dict = {
      "output": {"class": "combine", "kind": "add", "from": ["data:in1", "data:in2"]},
    }
    config = Config({
      "debug_print_layer_output_template": True,
      "extern_data": {
        "in1": {"shape": (n_features, 1), "batch_dim_axis": None, "time_dim_axis": None, "feature_dim_axis": 0},
        "in2": {"shape": (n_features, None), "batch_dim_axis": 0, "time_dim_axis": 2}
      }
    })
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(net_dict)
    out = network.get_default_output_layer()
    assert_equal(out.output.batch_shape, (None, n_features, None))
    feed_dict = make_feed_dict(network.extern_data, n_batch=n_batch, n_time=n_time)
    session.run(tf_compat.v1.global_variables_initializer())
    out_v = session.run(out.output.placeholder, feed_dict=feed_dict)
    assert out_v.shape == (n_batch, n_features, n_time)


def test_CombineLayer_time_broadcast_swapped():
  with make_scope() as session:
    n_batch, n_time, n_features = 3, 7, 5
    net_dict = {
      "output": {"class": "combine", "kind": "add", "from": ["data:in1", "data:in2"]},
    }
    config = Config({
      "debug_print_layer_output_template": True,
      "extern_data": {
        "in1": {"shape": (n_features, None), "batch_dim_axis": 0, "time_dim_axis": 2},
        "in2": {"shape": (n_features, 1), "batch_dim_axis": None, "time_dim_axis": None, "feature_dim_axis": 0},
      }
    })
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(net_dict)
    out = network.get_default_output_layer()
    assert_equal(out.output.batch_shape, (None, n_features, None))
    feed_dict = make_feed_dict(network.extern_data, n_batch=n_batch, n_time=n_time)
    session.run(tf_compat.v1.global_variables_initializer())
    out_v = session.run(out.output.placeholder, feed_dict=feed_dict)
    assert out_v.shape == (n_batch, n_features, n_time)


def test_dot_layer_shuffled_remaining_dims_static():
  with make_scope() as session:
    import numpy as np
    net_dict = {
      "a": {"class": "split_dims", "axis": "static:0", "dims": (2, 3, 5)},
      "b": {"class": "transpose", "from": ["a"], "perm": {"static:0": "static:1", "static:1": "static:0"}},
      "dot": {
        "class": "dot", "from": ["a", "b"],
        "red1": "static:-1", "red2": "static:-1", "var1": None, "var2": None,
        "debug": True},
      "output": {"class": "merge_dims", "axes": "static", "from": ["dot"]}
    }
    config = Config()
    config.update({
      "extern_data": {"data": {"shape": (30,)}},
      "network": net_dict
    })
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])
    out = network.get_default_output_layer(must_exist=True)
    input_data = np.ones(shape=(17, 30))
    feed_dict = {network.layers['data'].output.placeholder: input_data}

    # just check that it runs
    session.run(out.output.placeholder, feed_dict)


def test_dot_layer_shuffled_remaining_dims_dynamic():
  with make_scope() as session:
    import numpy as np
    batch_size = 8
    time_size = 20
    feat_size = 10
    net_dict = {
      "a": {"class": "copy", "from": "data"},
      "b": {"class": "transpose", "from": ["a"], "perm": {"B": "T", "T": "B"}},
      "dot": {
        "class": "dot", "from": ["a", "b"],
        "red1": "F", "red2": "F", "var1": None, "var2": None,
        "debug": True},
      "output": {"class": "copy", "from": ["dot"]}
    }
    config = Config()
    config.update({
      "num_outputs": 1,
      "num_inputs": feat_size,
      "network": net_dict
    })
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])
    out = network.get_default_output_layer(must_exist=True)
    input_data = np.ones(shape=(batch_size, time_size, feat_size))
    feed_dict = {network.layers['data'].output.placeholder: input_data}

    # just check that it runs
    session.run(out.output.placeholder, feed_dict)


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
      "sub": {"class": "subnetwork", "from": "ff0", "subnetwork": {
        "ff1": {"class": "forward", "activation": "relu", "n_out": 2},
        "output": {"class": "forward", "activation": "relu", "n_out": 2}
      }},
      "output": {"class": "softmax", "loss": "ce", "from": "sub"}
    }
    config = Config()
    config.update(dict(num_inputs=4, num_outputs=3))
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(net_dict)
    assert_equal(network.layers["sub"].output.dim, 2)
    sub_layer = network.layers["sub"]
    assert isinstance(sub_layer, SubnetworkLayer)
    sub_layer_deps = sub_layer.get_dep_layers()
    assert sub_layer_deps, "%r no deps" % sub_layer
    all_deps = set()
    queue = [sub_layer]
    while queue:
      layer = queue.pop(0)
      if layer in all_deps:
        continue
      all_deps.add(layer)
      for dep in layer.get_dep_layers():
        if dep not in all_deps:
          queue.append(dep)
    assert network.layers["ff0"] in all_deps


def test_subnet_loss():
  with make_scope() as session:
    config = Config({
      "extern_data": {"data": {"dim": 1}},
      "debug_print_layer_output_template": True
    })
    net_dict = {
      "sub": {"class": "subnetwork", "from": [], "subnetwork": {
        "var": {"class": "variable", "shape": [1]},
        "loss": {"class": "copy", "from": "var", "loss": "as_is"},
        "output": {"class": "copy", "from": "var"}  # no dep on "loss"
      }},
      # Output dep on "sub" to trigger subnet creation.
      # In theory, it would be nice if the loss is also constructed without that,
      # but this doesn't work currently as "sub" is never constructed by the current heuristics.
      "output": {"class": "copy", "from": "sub"}
    }
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(net_dict)
    losses_dict, total_loss, total_constraints = network.get_losses_initialized(with_total=True)
    print("losses:")
    pprint(losses_dict)
    assert len(losses_dict) == 1 and set(losses_dict.keys()) == {"sub/loss"}


def test_subnet2_loss():
  with make_scope() as session:
    config = Config({
      "extern_data": {"data": {"dim": 1}},
      "debug_print_layer_output_template": True
    })
    net_dict = {
      "sub": {"class": "subnetwork", "from": [], "subnetwork": {
        "var": {"class": "variable", "shape": [1]},
        "loss": {"class": "copy", "from": "var", "loss": "as_is"},
        "output": {"class": "copy", "from": "var"}  # no dep on "loss"
      }},
      # Output dep on "sub" to trigger subnet creation.
      # In theory, it would be nice if the loss is also constructed without that,
      # but this doesn't work currently as "sub" is never constructed by the current heuristics.
      # Specifically depend on "sub/output", because in that case,
      # the SubnetworkLayer itself might not be created with the new subnet logic,
      # which is sth we want to test.
      "output": {"class": "copy", "from": "sub/output"}
    }
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(net_dict)
    losses_dict, total_loss, total_constraints = network.get_losses_initialized(with_total=True)
    print("losses:")
    pprint(losses_dict)
    assert len(losses_dict) == 1 and set(losses_dict.keys()) == {"sub/loss"}


def test_constant_layer():
  with make_scope() as session:
    config = Config()
    config.update({
      "num_outputs": 3,
      "num_inputs": 2,
      "network": {
        "output": {"class": "constant", "value": 42}
      }
    })
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])
    out = network.get_default_output_layer(must_exist=True)
    v = session.run(out.output.placeholder)
    assert_equal(v.shape, ())  # (batch,), where batch==1 for broadcasting
    assert_equal(v, 42)


def test_compare_layer():
  with make_scope() as session:
    config = Config()
    config.update({
      "model": "/tmp/test-compare-layer-model",
      "num_outputs": 3,
      "num_inputs": 2,
      "network": {
        "const": {"class": "constant", "value": 3},
        "output": {"class": "compare", "from": "const", "value": 3}
      }
    })
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])
    out = network.get_default_output_layer(must_exist=True)
    v = session.run(out.output.placeholder)
    assert_equal(v.shape, ())  # (batch,), where batch==1 for broadcasting
    assert_equal(v.dtype, numpy.dtype("bool"))
    assert_equal(v, True)


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


def test_SoftmaxOverSpatialLayer_start():
  with make_scope() as session:
    net = TFNetwork(extern_data=ExternData())
    rnd = numpy.random.RandomState(42)
    n_batch = 3
    n_time = 4
    n_dim = 7
    start_idxs = numpy.array([[3], [0], [1]]).astype("int32")  # (B, 1)
    input_np = rnd.normal(size=(n_batch, n_time, n_dim)).astype("float32")  # (B, T, D)
    src = InternalLayer(name="src", network=net, out_type={"shape": (n_time, n_dim), "time_dim_axis": 1})
    start = InternalLayer(name="start", network=net, out_type={"shape": (1,), "dtype": "int32"})
    start.output.placeholder = tf.constant(start_idxs)
    start.output.size_placeholder = {}
    print("input:", src.output)
    src.output.placeholder = tf.constant(input_np, dtype=tf.float32)
    src.output.size_placeholder = {0: tf.constant([n_time] * n_batch)}  # not sure if enough...
    opts = {"network": net, "name": "softmax_over_spatial_test", "sources": [src],
            "start": start, "use_time_mask": True}
    out_data = SoftmaxOverSpatialLayer.get_out_data_from_opts(**opts)
    print("output:", out_data)
    out_data.sanity_check(ignore_placeholder=True)  # placeholder might be overwritten later
    assert_equal(out_data.shape, (n_dim, n_time))  # layer moves time-dim to back
    layer = SoftmaxOverSpatialLayer(output=out_data, **opts)
    assert_equal(layer.output.shape, (n_dim, n_time))
    out_np = session.run(layer.output.placeholder)
    assert_equal(out_np.shape, (n_batch, n_dim, n_time))
    # check if masking worked
    range_idxs = numpy.ones_like(start_idxs) * numpy.expand_dims(numpy.arange(n_time), axis=0)
    cond = range_idxs < numpy.broadcast_to(start_idxs, [n_batch, n_time])  # (B, T)
    cond = numpy.expand_dims(cond, axis=1)
    cond = numpy.broadcast_to(cond, [n_batch, n_dim, n_time])  # (B, D, T)
    assert_equal(cond.sum(), n_dim*start_idxs.sum())  # check num of conds
    numpy.testing.assert_array_equal(out_np[cond], 0)


def test_SoftmaxOverSpatialLayer_window():
  with make_scope() as session:
    net = TFNetwork(extern_data=ExternData())
    rnd = numpy.random.RandomState(42)
    n_batch = 4
    n_time = 9
    n_dim = 1
    window_size = 5
    window_start_idxs = numpy.array([3, 0, 1, 7]).astype("int32")  # (B,)
    seqlens = numpy.array([5, 7, 3, 9])
    input_np = rnd.normal(size=(n_batch, n_time, n_dim)).astype("float32")  # (B, T, D)
    src = InternalLayer(name="src", network=net, out_type={"shape": (n_time, n_dim), "time_dim_axis": 1})
    window_start = InternalLayer(name="window_start", network=net, out_type={"shape": (), "dtype": "int32"})
    window_start.output.placeholder = tf.constant(window_start_idxs)  # (B,)
    window_start.output.size_placeholder = {}
    print("input:", src.output)
    src.output.placeholder = tf.constant(input_np, dtype=tf.float32)
    src.output.size_placeholder = {0: tf.constant(seqlens)}
    opts = {"network": net, "name": "softmax_over_spatial_test", "sources": [src],
            "window_start": window_start, "window_size": window_size}
    out_data = SoftmaxOverSpatialLayer.get_out_data_from_opts(**opts)
    print("output:", out_data)
    out_data.sanity_check(ignore_placeholder=True)  # placeholder might be overwritten later
    assert_equal(out_data.shape, (n_dim, n_time))  # layer moves time-dim to back
    layer = SoftmaxOverSpatialLayer(output=out_data, **opts)
    layer.output.sanity_check()
    assert_equal(layer.output.shape, (n_dim, n_time))
    out_np = session.run(layer.output.placeholder)
    assert_equal(out_np.shape, (n_batch, n_dim, n_time))
    # check if window masking worked:
    # handle edge cases correctly: (start is 0-based)
    # 1. if the energy time-dim is less than `window_size`, we adjust the window size.
    # 2. for each seq, we adjust the window so that no elements after the seq-len are indexed.
    # seq[0]: start=3, seqlen=5 -> [1, 1, 1, 1, 1, 0, 0, 0, 0]
    # seq[1]: start=0, seqlen=7 -> [1, 1, 1, 1, 1, 0, 0, 0, 0]
    # seq[2]: start=1, seqlen=3 -> [1, 1, 1, 0, 0, 0, 0, 0, 0]
    # seq[3]: start=7, seqlen=9 -> [0, 0, 0, 0, 1, 1, 1, 1, 1]
    mask = numpy.array([
      [0, 0, 0, 1, 1, 0, 0, 0, 0],
      [1, 1, 1, 1, 1, 0, 0, 0, 0],
      [0, 1, 1, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 1, 1, 1, 1]
    ], dtype=numpy.bool)  # (B, T)
    print("mask", mask)
    mask = numpy.expand_dims(mask, axis=1)
    mask = numpy.broadcast_to(mask, [n_batch, n_dim, n_time])  # (B, D, T)
    # check if layer output sums to one for each seq:
    out_sum = numpy.sum(out_np, axis=(1, 2))
    numpy.testing.assert_allclose(out_sum, [1]*n_batch, rtol=1e-5)
    numpy.testing.assert_allclose(out_np[~mask], 0, rtol=1e-5)  # check if masking worked


def test_SplitLayer_after_SplitDimsLayer():
  n_batch, n_time, n_in = 7, 3, 40
  config = Config({
    "extern_data": {"data": {"dim": n_in}},
    "debug_print_layer_output_template": True,
  })
  with make_scope():
    net = TFNetwork(config=config)
    net.construct_from_dict({
      "split_heads": {"class": "split_dims", "dims": (2, -1), "axis": "F"},  # [B,T,2,F|20]
      "split_qkv": {"class": "split", "size_splits": (5, 5, 10), "axis": "F", "from": "split_heads"},
      "output": {"class": "copy", "from": "split_qkv/0"}})  # [B,T,2,F|5]
    out_t = net.get_default_output_layer().output.placeholder
    assert out_t.shape.as_list() == [None, None, 2, 5]


def test_SplitLayer_search():
  n_batch, n_time, n_in, n_out = 7, 3, 10, 10
  beam_size = 4
  config = Config({
    "extern_data": {
      "data": {"dim": n_in},
      "classes": {"dim": n_out, "sparse": True, "available_for_inference": False}},
    "debug_print_layer_output_template": True
  })
  with make_scope():
    net = TFNetwork(config=config, search_flag=True, train_flag=False, eval_flag=True)
    net.construct_from_dict({
      "encoder_seq": {"class": "linear", "activation": "tanh", "n_out": 5},
      "encoder": {"class": "reduce", "mode": "sum", "from": ["encoder_seq"], "axis": "T"},
      "output": {"class": "rec", "from": [], "target": "classes", "max_seq_len": 20, "unit": {
        "embed": {"class": "linear", "from": ["prev:output"], "activation": None, "n_out": 10},
        "split": {"class": "split", "size_splits": (5, 5), "axis": "F", "from": ["embed"]},
        "output_prob": {"class": "softmax", "from": ["split/0", "base:encoder"], "target": "classes", "loss": "ce"},
        "output": {
          "class": "choice", "target": "classes", "beam_size": beam_size, "from": ["output_prob"], "initial_output": 0},
        "end": {"class": "compare", "from": ["output"], "value": 0}
      }},
      "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance", "target": "classes"}})


def test_SplitDimsLayer_simple_feat():
  n_batch, n_time, n_in = 7, 3, 20
  config = Config({
    "extern_data": {"data": {"dim": n_in}},
    "debug_print_layer_output_template": True,
  })
  with make_scope() as session:
    net = TFNetwork(config=config)
    net.construct_from_dict({
      "output": {"class": "split_dims", "axis": "f", "dims": (-1, 5)}})
    out_t = net.get_default_output_layer().output.placeholder
    assert out_t.shape.as_list() == [None, None, 4, 5]
    in_v = numpy.arange(0, n_batch * n_time * n_in).astype("float32").reshape((n_batch, n_time, n_in))
    out_v = session.run(out_t, feed_dict={net.extern_data.data["data"].placeholder: in_v})
    assert isinstance(out_v, numpy.ndarray)
    assert out_v.shape == (n_batch, n_time, 4, 5)
    numpy.testing.assert_almost_equal(out_v, in_v.reshape(out_v.shape))


def test_SplitDimsLayer_simple_time():
  n_batch, n_time, n_in = 7, 3, 20
  config = Config({
    "extern_data": {"data": {"dim": n_in}},
    "debug_print_layer_output_template": True,
  })
  with make_scope() as session:
    net = TFNetwork(config=config)
    net.construct_from_dict({
      "output": {"class": "split_dims", "axis": "t", "dims": (-1, 1)}})
    assert_equal(
      net.get_default_output_layer().output.get_dim_tag(1),
      net.extern_data.get_default_input_data().get_dim_tag(1))
    out_t = net.get_default_output_layer().output.placeholder
    assert out_t.shape.as_list() == [None, None, 1, 20]
    in_v = numpy.arange(0, n_batch * n_time * n_in).astype("float32").reshape((n_batch, n_time, n_in))
    out_v = session.run(out_t, feed_dict={net.extern_data.data["data"].placeholder: in_v})
    assert isinstance(out_v, numpy.ndarray)
    assert out_v.shape == (n_batch, n_time, 1, n_in)
    numpy.testing.assert_almost_equal(out_v, in_v.reshape(out_v.shape))


def test_SplitDimsLayer_simple_time2():
  n_batch, n_time, n_in = 7, 3, 20
  config = Config({
    "extern_data": {"data": {"dim": n_in}},
    "debug_print_layer_output_template": True,
  })
  with make_scope() as session:
    net = TFNetwork(config=config)
    net.construct_from_dict({
      "output": {"class": "split_dims", "axis": "t", "dims": (1, -1)}})
    assert_equal(
      net.get_default_output_layer().output.get_dim_tag(2),
      net.extern_data.get_default_input_data().get_dim_tag(1))
    out_t = net.get_default_output_layer().output.placeholder
    assert out_t.shape.as_list() == [None, 1, None, 20]
    in_v = numpy.arange(0, n_batch * n_time * n_in).astype("float32").reshape((n_batch, n_time, n_in))
    out_v = session.run(out_t, feed_dict={net.extern_data.data["data"].placeholder: in_v})
    assert isinstance(out_v, numpy.ndarray)
    assert out_v.shape == (n_batch, 1, n_time, n_in)
    numpy.testing.assert_almost_equal(out_v, in_v.reshape(out_v.shape))


def test_SplitDimsLayer_resolve_dims():
  assert_equal(SplitDimsLayer._resolve_dims(old_dim=3 * 5, new_dims=(3, -1)), (3, 5))
  assert_equal(SplitDimsLayer._resolve_dims(old_dim=3 * 5, new_dims=(3, 5)), (3, 5))
  assert_equal(SplitDimsLayer._resolve_dims(old_dim=3 * 5, new_dims=(5, -1)), (5, 3))
  assert_equal(SplitDimsLayer._resolve_dims(old_dim=2 * 3 * 5, new_dims=(-1, 3, 5)), (2, 3, 5))
  assert_equal(SplitDimsLayer._resolve_dims(old_dim=2 * 3 * 5, new_dims=(2, -1, 5)), (2, 3, 5))
  assert_equal(SplitDimsLayer._resolve_dims(old_dim=2 * 3 * 5, new_dims=(2, 3, -1)), (2, 3, 5))
  assert_equal(SplitDimsLayer._resolve_dims(old_dim=2 * 3 * 5, new_dims=(2, 3, -1, 1)), (2, 3, 5, 1))

  assert_equal(SplitDimsLayer._resolve_dims(old_dim=3 * 5, new_dims=(3, -1), pad_to_multiples=True), (3, 5))
  assert_equal(SplitDimsLayer._resolve_dims(old_dim=3 * 5 + 1, new_dims=(3, -1), pad_to_multiples=True), (3, 6))
  assert_equal(SplitDimsLayer._resolve_dims(old_dim=2 * 3 * 5, new_dims=(2, 3, -1), pad_to_multiples=True), (2, 3, 5))
  assert_equal(SplitDimsLayer._resolve_dims(old_dim=2 * 3 * 5, new_dims=(2, 3, -1, 1), pad_to_multiples=True), (2, 3, 5, 1))
  assert_equal(SplitDimsLayer._resolve_dims(old_dim=2 * 3 * 5 + 2, new_dims=(2, 3, -1), pad_to_multiples=True), (2, 3, 6))
  assert_equal(SplitDimsLayer._resolve_dims(old_dim=2 * 3 * 5 + 2, new_dims=(2, 3, -1, 1), pad_to_multiples=True), (2, 3, 6, 1))


def _check_MergeDimsLayer(session, in_data_opts, in_static_shape, opts, out_data_shape, out_static_shape,
                          in_sizes=None, out_sizes=None):
  """
  :param tf.compat.v1.Session session:
  :param dict[str] in_data_opts:
  :param tuple[int] in_static_shape:
  :param dict[str] opts: for MergeDimsLayer
  :param tuple[int|None] out_data_shape:
  :param tuple[int] out_static_shape:
  :param dict[int,tuple[int]]|None in_sizes:
  :param dict[int,tuple[int]]|None out_sizes:
  :rtype: MergeDimsLayer
  """
  net = TFNetwork(extern_data=ExternData())
  rnd = numpy.random.RandomState(42)
  src = InternalLayer(name="src", network=net, out_type=in_data_opts)
  print("input:", src.output)
  src.output.placeholder = tf.constant(rnd.normal(size=in_static_shape).astype("float32"), dtype=tf.float32)
  src.output.size_placeholder = {}
  if src.output.batch_dim_axis is not None:
    n_batch = in_static_shape[src.output.batch_dim_axis]
    for axis, dim in enumerate(src.output.batch_shape):
      axis_wo_b = src.output.get_batch_axis_excluding_batch(axis)
      if dim is None and axis_wo_b is not None:
        if in_sizes and axis_wo_b in in_sizes:
          src.output.size_placeholder[axis_wo_b] = tf.constant(in_sizes[axis_wo_b])
        else:
          src.output.size_placeholder[axis_wo_b] = tf.fill([n_batch], in_static_shape[axis])

  opts = opts.copy()
  print("opts:", opts)
  opts.update({"network": net, "name": "merge_dims_test", "sources": [src]})
  out_data = MergeDimsLayer.get_out_data_from_opts(**opts)
  out_data.sanity_check(ignore_placeholder=True)  # placeholder might be overwritten later
  assert_equal(out_data.shape, out_data_shape)
  layer = MergeDimsLayer(output=out_data, **opts)
  assert_equal(layer.output.shape, out_data_shape)
  out_np, size_placeholder = session.run([layer.output.placeholder, layer.output.size_placeholder])
  print("output:", out_data)
  assert_equal(out_np.shape, out_static_shape)

  if out_sizes:
    assert_equal(sorted(size_placeholder.keys()), sorted(out_sizes))
    for k in size_placeholder.keys():
      numpy.testing.assert_array_equal(size_placeholder[k], out_sizes[k])

  return layer


def test_MergeDimsLayer_basic():
  with make_scope() as session:
    _check_MergeDimsLayer(session, {"shape": (4, 7), "time_dim_axis": None}, (2, 4, 7), {"axes": "except_batch"}, (4 * 7,), (2, 4 * 7))
    _check_MergeDimsLayer(session, {"shape": (4, None, 7), "time_dim_axis": None}, (2, 4, 3, 7), {"axes": "static"}, (None, 4 * 7), (2, 3, 4 * 7))
    _check_MergeDimsLayer(session, {"shape": (4, None, 7), "time_dim_axis": 2}, (2, 4, 3, 7), {"axes": "static"}, (None, 4 * 7), (2, 3, 4 * 7))
    _check_MergeDimsLayer(session, {"shape": (1, None), "time_dim_axis": 2, "feature_dim_axis": 1}, (2, 1, 4), {"axes": "except_batch"}, (None,), (2, 4))


def test_MergeDimsLayer_size_placeholder():
  with make_scope() as session:
    _check_MergeDimsLayer(
      session,
      {"shape": (None, 2), "time_dim_axis": 1, "feature_dim_axis": 2}, (3, 4, 2), {"axes": "except_batch"}, (None,), (3, 8),
      in_sizes={0: (4, 2, 1)}, out_sizes={0: (8, 4, 2)})


def test_MergeDimsLayer_batch_time_ext():
  with make_scope() as session:
    n_batch = 11
    n_time = 13
    _check_MergeDimsLayer(
      session, {"shape": (None, 5, 3)}, (n_batch, n_time, 5, 3), {"axes": "BT"}, (5, 3), (n_batch * n_time, 5, 3))


def test_MergeDimsLayer_batch_time_time_major():
  with make_scope() as session:
    n_batch = 11
    n_time = 13
    layer = _check_MergeDimsLayer(
      session,
      {"shape": (None, 5), "time_dim_axis": 0, "batch_dim_axis": 1}, (n_time, n_batch, 5),
      {"axes": "BT"}, (5,), (n_time * n_batch, 5))
    assert layer.output.batch_dim_axis == 0
    assert layer.output.time_dim_axis is None


def test_MergeDimsLayer_batch_time_time_major_ext():
  with make_scope() as session:
    n_batch = 11
    n_time = 13
    layer = _check_MergeDimsLayer(
      session,
      {"shape": (None, 5, 3), "time_dim_axis": 0, "batch_dim_axis": 1}, (n_time, n_batch, 5, 3),
      {"axes": "BT"}, (5, 3), (n_time * n_batch, 5, 3))
    assert layer.output.batch_dim_axis == 0
    assert layer.output.time_dim_axis is None  # Note: This behavior was changed.


def test_MergeDimsLayer_except_time_ext():
  with make_scope() as session:
    n_batch = 11
    n_time = 13
    layer = _check_MergeDimsLayer(
      session,
      {"shape": (3, None, 5), "time_dim_axis": 2}, (n_batch, 3, n_time, 5),
      {"axes": "except_time"}, (None, 15), (n_batch, n_time, 15))
    assert layer.output.batch_dim_axis == 0 and layer.output.time_dim_axis == 1


def test_MergeDimsLayer_static_time():
  with make_scope() as session:
    n_batch = 11
    layer = _check_MergeDimsLayer(
      session,
      {"shape": (3, 5), "time_dim_axis": 1}, (n_batch, 3, 5),
      {"axes": "static"}, (15,), (n_batch, 15))
    assert layer.output.batch_dim_axis == 0 and layer.output.feature_dim_axis == 1
    assert layer.output.time_dim_axis is None


def test_MergeDimsLayer_dim_tags():
  n_batch = 3
  with make_scope() as session:
    net = TFNetwork(extern_data=ExternData())
    rnd = numpy.random.RandomState(42)

    src_data = Data("input", shape=(None, 1, 2, 1), feature_dim_axis=None)
    input_static_shape = (n_batch, 7, 1, 2, 1)
    src_data.placeholder = tf.constant(rnd.normal(size=input_static_shape).astype("float32"), dtype=tf.float32)
    src_data.size_placeholder = {}
    from returnn.tf.util.basic import DimensionTag
    # map axis_wo_batch -> (tag description, dyn_size)
    tag_names_with_dyn_size = {0: ("key-chunk", [4, 2, 3]), 1: ("key-window", [1, 1, 1]), 2: ("att-heads", [2, 2, 2])}
    for axis_wo_batch, (description, dyn_size) in tag_names_with_dyn_size.items():
      tag = DimensionTag(description=description, kind=DimensionTag.Types.Spatial)
      dyn_size = tf.constant(dyn_size)
      src_data.size_placeholder[axis_wo_batch] = dyn_size
      tag.set_tag_on_size_tensor(dyn_size)
    print('in data:', src_data)  # should be [B,T|'key-chunk',1|'key-window',2|'att-heads',1]
    assert (
      src_data.get_axis_by_tag_name('key-chunk') == 1 and src_data.get_axis_by_tag_name('key-window') == 2 and
      src_data.get_axis_by_tag_name('att-heads') == 3)

    merge_axes = ['stag:key-window', 'spatial:-1']
    print('merge axes:', merge_axes)

    src = InternalLayer(name="src", network=net, output=src_data)
    opts = {"network": net, "name": "merge_dims_test", "sources": [src], "axes": merge_axes}
    out_data = MergeDimsLayer.get_out_data_from_opts(**opts)
    out_data.sanity_check(ignore_placeholder=True)  # placeholder might be overwritten later
    print('template out data:', out_data)  # should be [B,T|'key-chunk',1|<anything>,2|'att-heads',1]
    assert out_data.shape == src_data.shape[:-1]
    assert out_data.get_axis_by_tag_name('key-chunk') == 1 and out_data.get_axis_by_tag_name('att-heads') == 3

    layer = MergeDimsLayer(output=out_data, **opts)
    layer.output.sanity_check()
    out_data = layer.output
    print('layer out data:', out_data)
    assert out_data.shape == src_data.shape[:-1]
    assert out_data.get_axis_by_tag_name('key-chunk') == 1 and out_data.get_axis_by_tag_name('att-heads') == 3


def test_MergeDimsLayer_SplitBatchTimeLayer_time_major():
  n_batch = 3
  n_time = 4
  n_input_dim = 5
  # Time major
  input_data = numpy.arange(n_time * n_batch * n_input_dim).reshape((n_time, n_batch, n_input_dim)).astype("float32")
  with make_scope() as session:
    seq_lens = [n_time, n_time - 1, n_time - 2]
    assert len(seq_lens) == n_batch and all([s > 0 for s in seq_lens])
    net = TFNetwork(extern_data=ExternData(), config=Config({"debug_print_layer_output_template": True}))
    input_layer = net.add_layer(
      "input", InternalLayer,
      output=Data(
        name="input", shape=(None, n_input_dim), time_dim_axis=0, batch_dim_axis=1,
        placeholder=tf.constant(input_data),
        size_placeholder={0: tf.constant(seq_lens)}))
    assert input_layer.output.is_time_major
    net.construct_from_dict({
      "merge_dims": {"class": "merge_dims", "from": "input", "axes": "BT"},
      "split_dims": {"class": "split_batch_time", "from": "merge_dims", "base": "input"},
      "output": {"class": "copy", "from": "split_dims"}
    })
    output = net.get_default_output_layer().output
    # Depending on implementation, output could be batch-major or time-major.
    output = output.copy_as_time_major()  # such that we can compare easily to input_data
    assert output.is_time_major and output.shape == (None, n_input_dim)
    output_data = session.run(output.placeholder)
    numpy.testing.assert_almost_equal(input_data, output_data)


def test_MergeDimsLayer_SplitBatchTimeLayer_two_time_axes():
  n_dim = 11
  with make_scope() as session:
    net = TFNetwork(
      config=Config({
        "extern_data": {"data": {"shape": (None, None, n_dim)}},
        "debug_print_layer_output_template": True}))
    feed_dict = make_feed_dict(net.extern_data)
    net.construct_from_dict({
      "merge_dims": {"class": "merge_dims", "from": "data", "axes": "BT"},
      "split_dims": {"class": "split_batch_time", "from": "merge_dims", "base": "data"},
      "output": {"class": "copy", "from": "split_dims"}
    })
    input_data = net.extern_data.get_default_input_data()
    assert set(input_data.size_placeholder.keys()) == {0, 1}
    assert input_data.size_placeholder[0].name != input_data.size_placeholder[1].name
    assert input_data.get_size_dim_tag(0) != input_data.get_size_dim_tag(1)
    merged_data = net.layers["merge_dims"].output
    assert set(merged_data.size_placeholder.keys()) == {0}
    assert merged_data.get_size_dim_tag(0) != input_data.get_size_dim_tag(0)
    assert merged_data.get_size_dim_tag(0) == input_data.get_size_dim_tag(1)  # like beam-search, still same dim-tag
    output_data = net.get_default_output_layer().output
    output_data = output_data.copy_as_batch_major()
    assert output_data.shape == (None, None, n_dim)
    assert output_data.get_size_dim_tag(0) == input_data.get_size_dim_tag(0)
    assert output_data.get_size_dim_tag(1) == input_data.get_size_dim_tag(1)
    input_value = session.run(input_data.placeholder, feed_dict=feed_dict)
    merged_value = session.run(merged_data.placeholder, feed_dict=feed_dict)
    output_value = session.run(output_data.placeholder, feed_dict=feed_dict)
    assert input_value.shape == output_value.shape
    assert input_value.shape[-1] == n_dim
    n_batch, n_time0, n_time1, _ = input_value.shape
    numpy.testing.assert_almost_equal(input_value, output_value)
    assert merged_value.shape == (n_batch * n_time0, n_time1, n_dim)
    numpy.testing.assert_almost_equal(input_value, merged_value.reshape(input_value.shape))
    merged_size = session.run(merged_data.size_placeholder[0], feed_dict=feed_dict)
    input_size0, input_size1 = session.run(
      (input_data.size_placeholder[0], input_data.size_placeholder[1]), feed_dict=feed_dict)
    assert input_size0.shape == input_size1.shape == (n_batch,)
    assert merged_size.shape == (n_batch * n_time0,)
    merged_size = merged_size.reshape(n_batch, n_time0)
    assert (merged_size == input_size1[:, None]).all()


def test_MergeDimsLayer_simple_feat():
  n_batch, n_time, n_in1, n_in2 = 7, 3, 10, 32
  config = Config({
    "extern_data": {"data": {"shape": (None, n_in1, n_in2)}},
    "debug_print_layer_output_template": True,
  })
  with make_scope() as session:
    net = TFNetwork(config=config)
    net.construct_from_dict({
      "output": {"class": "merge_dims", "axes": "static"}})
    out_t = net.get_default_output_layer().output.placeholder
    assert out_t.shape.as_list() == [None, None, n_in1 * n_in2]
    in_v = numpy.arange(0, n_batch * n_time * n_in1 * n_in2).astype("float32").reshape((n_batch, n_time, n_in1, n_in2))
    out_v = session.run(out_t, feed_dict={net.extern_data.data["data"].placeholder: in_v})
    assert isinstance(out_v, numpy.ndarray)
    assert out_v.shape == (n_batch, n_time, n_in1 * n_in2)
    numpy.testing.assert_almost_equal(out_v, in_v.reshape(out_v.shape))


def test_FlattenBatchLayer():
  from returnn.tf.util.data import BatchInfo
  n_batch, n_time, n_in = 3, 4, 2
  config = Config({
    "extern_data": {"data": {"dim": n_in, "dtype": "int32"}},
    "debug_print_layer_output_template": True,
  })
  with make_scope() as session:
    net = TFNetwork(config=config)
    net.construct_from_dict({
      "output": {"class": "flatten_batch", "batch_major": False}})
    in_data = net.extern_data.data["data"]
    out_data = net.get_default_output_layer().output
    assert out_data.batch_shape == (None, n_in) and not out_data.size_placeholder
    assert len(out_data.batch.virtual_dims) == 2
    batch_flat_dim0, batch_flat_dim1 = out_data.batch.virtual_dims
    assert isinstance(batch_flat_dim0, BatchInfo.PackedDim)
    assert isinstance(batch_flat_dim1, BatchInfo.GlobalBatchDim)
    assert batch_flat_dim0.sizes is in_data.size_placeholder[0]
    out_t = net.get_default_output_layer().output.placeholder
    assert out_t.shape.as_list() == [None, n_in]
    in_v = numpy.arange(0, n_batch * n_time * n_in).reshape((n_time, n_batch, n_in)).transpose(1, 0, 2)
    in_seq_lens = [4, 3, 2]
    out_v = session.run(out_t, feed_dict={
      in_data.placeholder: in_v,
      in_data.size_placeholder[0]: in_seq_lens})
    assert isinstance(out_v, numpy.ndarray)
    assert out_v.shape == (sum(in_seq_lens), n_in)
    assert_equal(out_v.tolist(), [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [18, 19]])


def test_SwitchLayer_const_no_time():
  config = Config({
    "extern_data": {
      "data": {"dim": 3, "sparse": True, "shape": ()},
    },
    "debug_print_layer_output_template": True,
  })
  with make_scope() as session:
    net = TFNetwork(config=config, train_flag=True)
    net.construct_from_dict({
      "input_eq_0": {"class": "compare", "from": "data", "value": 0},  # (B,)
      "const0": {"class": "constant", "value": 0},
      "const1": {"class": "constant", "value": 1},
      "switch": {"class": "switch", "condition": "input_eq_0", "true_from": "const1", "false_from": "const0"},
      "output": {"class": "copy", "from": "switch"}})
    net.print_network_info()
    feed_dict = make_feed_dict(net.extern_data.data.values())
    out = session.run(net.get_default_output_layer().output.placeholder, feed_dict=feed_dict)
    print(out)


def test_SwitchLayer_const():
  config = Config({
    "extern_data": {
      "data": {"dim": 3, "sparse": True},
    },
    "debug_print_layer_output_template": True,
  })
  with make_scope() as session:
    net = TFNetwork(config=config, train_flag=True)
    net.construct_from_dict({
      "input_eq_0": {"class": "compare", "from": "data", "value": 0},  # (B,T)
      "const0": {"class": "constant", "value": 0},
      "const1": {"class": "constant", "value": 1},
      "switch": {
        "class": "switch", "condition": "input_eq_0", "true_from": "const1", "false_from": "const0"
        },
      "output": {"class": "copy", "from": "switch"}})
    net.print_network_info()
    feed_dict = make_feed_dict(net.extern_data.data.values())
    out = session.run(net.get_default_output_layer().output.placeholder, feed_dict=feed_dict)
    print(out)


def test_SwitchLayer_masking():
  config = Config({
    "extern_data": {
      "data": {"dim": 3, "sparse": False},
    },
    "debug_print_layer_output_template": True,
  })
  with make_scope() as session:
    net = TFNetwork(config=config, train_flag=True)
    net.construct_from_dict({
      "projected": {"class": "gather", "from": "data", "axis": "F", "position": 0}, # (B,T)
      "mask": {"class": "compare", "from": "projected", "value": 0, "kind": "greater"},  # (B,T)
      "switch": {
        "class": "switch", "condition": "mask", "true_from": "data", "false_from": float("-inf")
        },
      "output": {"class": "copy", "from": "switch"}})
    net.print_network_info()
    feed_dict = make_feed_dict(net.extern_data.data.values())
    out = session.run(net.get_default_output_layer().output.placeholder, feed_dict=feed_dict)
    print(out)


def test_SwitchLayer_template_const_from():
  net = TFNetwork(extern_data=ExternData())
  # [T]
  condition = InternalLayer(network=net, name="condition", output=Data(
    "condition_output", batch_dim_axis=None, time_dim_axis=0, feature_dim_axis=None, shape=(None,)))
  true_from = 42
  # [B,F|2,T]
  false_from = InternalLayer(network=net, name="false_from", output=Data(
    "false_from_output", batch_dim_axis=0, time_dim_axis=2, feature_dim_axis=1, shape=(2, None), dim=2))

  # should be [B,F|2,T]
  switch = SwitchLayer.get_out_data_from_opts('switch', condition=condition, true_from=true_from,
    false_from=false_from)
  assert switch.batch_ndim == 3
  assert switch.batch_dim_axis == 0 and switch.time_dim_axis == 2 and switch.feature_dim_axis == 1
  assert switch.dim == 2


def test_CondLayer_subnetwork_train():
  n_batch, n_time, n_in, n_out = 3, 7, 11, 13
  config = Config({
    "extern_data": {
      "data": {"dim": n_in},
      "classes": {"dim": n_out, "sparse": True},
    },
    "debug_print_layer_output_template": True,
  })
  rnd = numpy.random.RandomState(42)
  with make_scope() as session:
    net = TFNetwork(config=config, train_flag=True)
    net.construct_from_dict({
      "src": {"class": "linear", "activation": "tanh", "n_out": 10, "from": "data"},
      "cond": {
        "class": "cond", "from": [],
        "condition": {
          "class": "eval", "from": [], "out_type": {"batch_dim_axis": None, "shape": (), "dtype": "bool"},
          "eval": "tf.equal(self.network.global_train_step % 2, 0)"
          },
        "true_layer": {
          "class": "subnetwork", "from": "src", "subnetwork": {
            "lin": {"class": "linear", "activation": "tanh", "n_out": 10},
            "res": {"class": "combine", "kind": "add", "from": ["data", "lin"]},
            "output": {"class": "print", "from": "res", "extra_print_args": ["true_layer"], "summarize": 1}
          }},
        "false_layer": {"class": "copy", "from": "src"}
        },
      "output": {"class": "softmax", "from": "cond", "loss": "ce", "target": "classes"}})
    net.print_network_info()
    cond_layer = net.get_layer("cond")
    assert isinstance(cond_layer, CondLayer)
    assert not tf_util.has_control_flow_context(cond_layer.output.placeholder)
    cond_true_layer = cond_layer.true_layer
    assert isinstance(cond_true_layer, SubnetworkLayer)
    # Check whether the execution of the true branch is actually conditionally.
    assert tf_util.has_control_flow_context(cond_true_layer.output.placeholder)
    trainable_vars = net.get_trainable_params()
    print("Trainable vars:")
    pprint(trainable_vars)
    cond_var = net.layers["cond"].params["lin/W"]
    assert cond_var in trainable_vars
    from returnn.tf.updater import Updater
    updater = Updater(config=config, network=net, initial_learning_rate=0.1)
    updater.set_trainable_vars(trainable_vars)
    updater.init_optimizer_vars(session)
    updater.set_learning_rate(value=updater.initial_learning_rate, session=session)
    net.initialize_params(session)
    in_v = rnd.normal(size=(n_batch, n_time, n_in)).astype("float32")
    targets_v = rnd.randint(0, n_out, size=(n_batch, n_time)).astype("int32")
    seq_lens_v = numpy.array([n_time, n_time - 1, n_time - 2])
    assert len(seq_lens_v) == n_batch
    feed_dict = {
      net.extern_data.data["data"].placeholder: in_v,
      net.extern_data.data["data"].size_placeholder[0]: seq_lens_v,
      net.extern_data.data["classes"].placeholder: targets_v,
      net.extern_data.data["classes"].size_placeholder[0]: seq_lens_v,
    }
    fetches = net.get_fetches_dict(with_summary=True, with_size=True)
    fetches["optim_op"] = updater.get_optim_op()
    try:
      loss = None
      initial_loss = float("inf")
      for i in range(10):
        step = session.run(net.global_train_step)
        print("step: %i" % step)
        assert i == step
        old_var_value = session.run(cond_var)
        result = session.run(feed_dict=feed_dict, fetches=fetches)
        loss = result["loss"]
        print("loss:", loss)
        if i == 0:
          initial_loss = loss
        new_var_value = session.run(cond_var)
        var_changed = (old_var_value != new_var_value).any()
        print("var changed:", var_changed)
        if i % 2 == 0:  # See cond layer, condition. Use true_layer every second iteration, starting with 0.
          # We used true_layer, thus the params should have been updated.
          assert var_changed
        else:
          # We did not use true_layer, thus the params should not have been updated.
          assert not var_changed
      assert loss is not None and loss < initial_loss and numpy.isfinite(initial_loss)
    except tf.errors.OpError as exc:
      print("TF exception:", type(exc).__name__, ":", exc)
      from returnn.tf.network import help_on_tf_exception
      help_on_tf_exception(session=session, exception=exc, fetches=fetches, feed_dict=feed_dict)
      raise


def test_ScatterNdLayer_RangeLayer():
  n_batch, n_time, n_ts, n_in, n_out = 2, 3, 6, 7, 11
  rnd = numpy.random.RandomState(42)
  config = Config({
    "debug_print_layer_output_template": True,
    "extern_data": {"data": {"dim": n_in}}
  })
  net_dict = {
    "t": {"class": "eval", "from": [], "eval": "tf.convert_to_tensor([1, 2])",
          "out_type": {"shape": (), "dtype": "int32", "sparse": True, "dim": None}},  # (B,)
    "range": {"class": "range", "limit": n_ts, "sparse": True},  # (Ts,)
    "add_t": {"class": "combine", "kind": "add", "from": ["t", "range"]},  # (T,Ts)
    "t_rel_var": {"class": "variable", "shape": (n_ts, n_out), "init": "glorot_uniform"},  # (B,Ts,D)
    "output": {"class": "scatter_nd", "from": "t_rel_var", "position": "add_t", "position_axis": -1,
               "output_dim_via_time_from": "data", "filter_invalid_indices": True}
  }
  with make_scope() as session:
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(net_dict)

    fetches = network.get_fetches_dict()
    data_input = network.extern_data.data["data"]
    out_layer = network.get_default_output_layer()
    assert isinstance(out_layer, ScatterNdLayer)
    assert out_layer.output.shape == (None, 11)
    assert out_layer.output.feature_dim_axis_or_unspecified is NotSpecified and out_layer.output.feature_dim_axis == 2
    assert out_layer.output.time_dim_axis == 1

    session.run(tf_compat.v1.variables_initializer(tf_compat.v1.global_variables() + [network.global_train_step]))
    info, out = session.run(
      (fetches, out_layer.output.placeholder),
      feed_dict={
        data_input.placeholder: rnd.normal(size=(n_batch, n_time, n_in)).astype("float32"),
        data_input.size_placeholder[0]: numpy.array([n_time] * n_batch, dtype="int32"),
      })
    print(info)
    print(out)  # random...


def _run_repeat_layer(session, net, input_data_layer):
  repetitions_data_layer = InternalLayer(
    name="repetitions", network=net,
    out_type={'shape': (None,), 'feature_dim_axis': None, 'dtype': 'int32'})
  repetitions_data_layer.output.placeholder = tf.constant(
    [[1, 3, 2, 1, 3, 4, 1, 1, 2, 1],
     [3, 2, 1, 3, 0, 1, 1, 0, 0, 0]], dtype="int32")  # [B, T] (sparse)
  repetitions_data_layer.output.size_placeholder = {0: tf.constant([10, 7], dtype="int32")}  # [B]

  opts = {
    'network': net, 'name': 'repeat_layer_test', 'sources': [input_data_layer],
    'repetitions': repetitions_data_layer, 'axis': 'T'}
  out_data = RepeatLayer.get_out_data_from_opts(**opts)
  out_data.sanity_check()
  print(out_data)
  repeat_layer = RepeatLayer(output=out_data, **opts)
  print(repeat_layer.output)

  output, size_placeholder = session.run([repeat_layer.output.placeholder, repeat_layer.output.size_placeholder])
  assert numpy.all(numpy.equal(size_placeholder[0], numpy.asarray([19, 11])))
  assert numpy.all(numpy.equal(output.shape, numpy.asarray([2, 19, 5])))
  # the 6 last positions of the second sequence need to be padded with zeros
  assert numpy.all(numpy.equal(output[1, 11:], 0))
  assert out_data.shape == (None, 5)
  assert out_data.batch_dim_axis == 0
  assert out_data.time_dim_axis == 1


def test_RepeatLayerBTF():
  with make_scope() as session:
    net = TFNetwork(extern_data=ExternData())
    input_data_layer = InternalLayer(name="src", network=net, out_type={'shape': (None, 5), 'dim': 5})
    input_data_layer.output.size_placeholder = {0: tf.constant([10, 7])}  # [B]
    input_data_layer.output.placeholder = tf_compat.v1.random_uniform((2, 10, 5))  # [B, T, F]

    _run_repeat_layer(session, net, input_data_layer)


def test_RepeatLayerTBF():
  with make_scope() as session:
    net = TFNetwork(extern_data=ExternData())
    input_data_layer = InternalLayer(
      name="src", network=net,
      out_type={'shape': (None, 5), 'dim': 5, 'batch_dim_axis': 1, 'time_dim_axis': 0})
    input_data_layer.output.size_placeholder = {0: tf.constant([10, 7])}  # [B]
    input_data_layer.output.placeholder = tf_compat.v1.random_uniform((10, 2, 5))  # [T, B, F]

    _run_repeat_layer(session, net, input_data_layer)


def test_RepeatLayerBFT():
  with make_scope() as session:
    net = TFNetwork(extern_data=ExternData())
    input_data_layer = InternalLayer(
      name="src", network=net,
      out_type={'shape': (5, None), 'dim': 5, 'time_dim_axis': 2, 'feature_dim_axis': 1})
    input_data_layer.output.size_placeholder = {1: tf.constant([10, 7])}  # [B]
    input_data_layer.output.placeholder = tf_compat.v1.random_uniform((2, 5, 10))  # [B, F, T]

    _run_repeat_layer(session, net, input_data_layer)


def test_TileLayer():
  with make_scope() as session:
    n_out = 5
    config = Config({
      "debug_print_layer_output_template": True,
      "extern_data": {
        "data": {"dim": n_out},
      }})
    net = TFNetwork(config=config, train_flag=True)
    net.construct_from_dict({
      "output": {"class": "tile", "multiples": {"F": 3}, "from": ["data"]}
    })
    session.run(tf_compat.v1.global_variables_initializer())
    out = net.layers["output"].output.placeholder
    n_batch = 3
    max_seq_len = 10
    feed = make_feed_dict(net.extern_data.data.values(), n_batch=n_batch, n_time=max_seq_len, same_time=True)
    v = session.run(out, feed_dict=feed)
    input_len = feed[net.extern_data.data["data"].size_placeholder[0]]
    input_data = feed[net.extern_data.data["data"].placeholder]

    ref = numpy.tile(input_data, [1, 1, 3])

    numpy.testing.assert_allclose(ref, v, rtol=1e-5)


def test_ScatterNdLayer_RangeLayer_RangeInAxisLayer():
  n_batch, n_time, n_ts, n_in, n_out = 2, 3, 6, 7, 11
  rnd = numpy.random.RandomState(42)
  config = Config({
    "debug_print_layer_output_template": True,
    "extern_data": {"data": {"dim": n_in}}
  })
  net_dict = {
    "t": {"class": "range_in_axis", "axis": "t", "keepdims": False, "from": "data", "sparse": True},  # (T,)
    "range": {"class": "range", "limit": n_ts, "sparse": True},  # (Ts,)
    "add_t": {"class": "combine", "kind": "add", "from": ["t", "range"]},  # (T,Ts)
    "t_rel_var": {"class": "variable", "shape": (n_ts, n_out), "init": "glorot_uniform"},  # (B,Ts,D)
    "output": {"class": "scatter_nd", "from": "t_rel_var", "position": "add_t", "position_axis": -1,
               "output_dim_via_time_from": "data", "filter_invalid_indices": True}
  }
  with make_scope() as session:
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(net_dict)

    fetches = network.get_fetches_dict()
    data_input = network.extern_data.data["data"]
    out_layer = network.get_default_output_layer()
    assert isinstance(out_layer, ScatterNdLayer)
    assert out_layer.output.shape == (None, None, 11)
    assert out_layer.output.feature_dim_axis_or_unspecified is NotSpecified and out_layer.output.feature_dim_axis == 3
    assert out_layer.output.time_dim_axis == 0

    session.run(tf_compat.v1.variables_initializer(tf_compat.v1.global_variables() + [network.global_train_step]))
    info, out = session.run(
      (fetches, out_layer.output.placeholder),
      feed_dict={
        data_input.placeholder: rnd.normal(size=(n_batch, n_time, n_in)).astype("float32"),
        data_input.size_placeholder[0]: numpy.array([n_time] * n_batch, dtype="int32"),
      })
    print(info)
    print(out)  # random...


def test_ScatterNdLayer_pos_batch_last_dim():
  config = Config({
    "debug_print_layer_output_template": True,
    "extern_data": {"data": {"dim": 13}}
  })
  with make_scope() as session:
    network = TFNetwork(config=config, train_flag=True)
    data = network.construct_layer({}, "data")
    pos = InternalLayer(
      name="pos", network=network,
      output=Data(
        name='pos', shape=(None, 6), dtype='int32', sparse=True, dim=None, batch_dim_axis=2,
        auto_create_placeholders=True))
    val = InternalLayer(
      name="val", network=network,
      output=Data(
        name='var', shape=(6, 11), time_dim_axis=None,
        auto_create_placeholders=True))
    scatter_opts = dict(
      name="scatter", network=network,
      sources=[val], position=pos, position_axis="except_batch:-1",
      output_dim_via_time_from=data, filter_invalid_indices=True)
    scatter_out_template = ScatterNdLayer.get_out_data_from_opts(**scatter_opts)
    print("scatter out:", scatter_out_template)
    assert scatter_out_template.shape == (None, None, 11) and scatter_out_template.batch_ndim == 4
    scatter = ScatterNdLayer(output=scatter_out_template, **scatter_opts)
    print("scatter out dim tags:")
    pprint(scatter.output.get_batch_shape_dim_tags())
    assert_equal(scatter.output.get_size_dim_tag(0), pos.output.get_time_dim_tag())
    assert_equal(scatter.output.get_size_dim_tag(1), data.output.get_time_dim_tag())
    session.run(scatter.output.placeholder, feed_dict=make_feed_dict([data.output, pos.output, val.output]))


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


def test_RandIntLayer():
  with make_scope() as session:
    from returnn.tf.util.data import DimensionTag
    n_out = 5
    config = Config({
      "debug_print_layer_output_template": True,
      "extern_data": {
        "data": {"dim": n_out}}
    })
    net = TFNetwork(config=config, train_flag=True)
    n_batch = 3
    max_seq_len = 10
    feed = make_feed_dict(net.extern_data.data.values(), n_batch=n_batch, n_time=max_seq_len, same_time=True)
    size_placeholder = net.extern_data.data["data"].size_placeholder[0]
    input_len = feed[size_placeholder]
    sz = (
      DimensionTag(description="feature", kind=DimensionTag.Types.Feature, dimension=5),
      DimensionTag(kind=DimensionTag.Types.Batch),
      net.extern_data.data["data"].get_size_dim_tag(0),
      3,
    )
    net.construct_from_dict({
      "output": {"class": "rand_int", "shape": sz, "minval": 3, "maxval": 10, "seed": 42}
    })
    session.run(tf_compat.v1.global_variables_initializer())
    out = net.layers["output"].output.placeholder
    v = session.run(out, feed_dict=feed)

    assert_equal(v.shape, (5, n_batch, max(input_len), 3))


def test_untrainable_params():
  with make_scope() as session:
    config = Config()
    n_in, n_out = 2, 3
    config.update({
      "num_outputs": n_out,
      "num_inputs": n_in,
      "network": {
        "l1": {"class": "linear", "activation": None, "n_out": n_out},
        "output": {"class": "linear", "activation": None, "from": ["l1"], "n_out": n_out, "trainable": False}
      }
    })
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])
    l1 = network.layers["l1"]
    l2 = network.layers["output"]
    assert_equal(set(network.get_trainable_params()), {l1.params["W"], l1.params["b"]})


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
        'output': {'class': 'choice', 'target': 'classes', 'beam_size': 5, 'from': ["output_prob"],
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
    print("Construct for training")
    from returnn.tf.layers.rec import RecLayer, _SubnetworkRecCell
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
    print("Construct for search")
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


def test_SliceLayer_NCHW():
  with make_scope() as session:
    import numpy as np
    net = TFNetwork(extern_data=ExternData())
    with tf_compat.v1.variable_scope("src_nchw"):
      src_nchw = InternalLayer(name="src_nchw", network=net, out_type={"dim": 16,
                                                                       "shape": (16, None, 16),
                                                                       "batch_dim_axis": 0,
                                                                       "time_dim_axis": 2,
                                                                       "feature_dim_axis": 1,
                                                                       "sparse": False
                                                                       })
      src_nchw.output.placeholder = tf_compat.v1.placeholder(shape=(None, 16, None, 16), dtype=tf.float32)
      src_nchw.output.size_placeholder = {1: tf_compat.v1.placeholder(shape=(None,), dtype=tf.int32)}
    with tf_compat.v1.variable_scope("src_nchw_feature_unspecified"):
      src_nchw_no_f = InternalLayer(name="src_nchw_feature_unspecified", network=net, out_type={"dim": 16,
                                                                                                "shape": (16, None, 16),
                                                                                                "batch_dim_axis": 0,
                                                                                                "time_dim_axis": 2,
                                                                                                "feature_dim_axis": NotSpecified,
                                                                                                "sparse": False
                                                                                                })
      src_nchw_no_f.output.placeholder = tf_compat.v1.placeholder(shape=(None, 16, None, 16), dtype=tf.float32)
      src_nchw_no_f.output.size_placeholder = {1: tf_compat.v1.placeholder(shape=(None,), dtype=tf.int32)}
    with tf_compat.v1.variable_scope("slice1"):
      slice1 = SliceLayer(
        name="slice1", network=net, axis="f", slice_step=2, sources=[src_nchw],
        output=SliceLayer.get_out_data_from_opts(name="slice1", axis="f", slice_step=2,
                                                 sources=[src_nchw]))
    with tf_compat.v1.variable_scope("slice2"):
      slice2 = SliceLayer(
        name="slice2", network=net, axis="f", slice_step=2, sources=[src_nchw_no_f],
        output=SliceLayer.get_out_data_from_opts(name="slice2", axis="f", slice_step=2,
                                                 sources=[src_nchw_no_f]))
    out1, out2 = session.run([slice1.output.placeholder, slice2.output.placeholder],
                             feed_dict={src_nchw.output.placeholder: np.random.rand(10, 16, 11, 16),
                                        src_nchw.output.size_placeholder[1]: np.full(shape=(10,), fill_value=11),
                                        src_nchw_no_f.output.placeholder: np.random.rand(10, 16, 11, 16),
                                        src_nchw_no_f.output.size_placeholder[1]: np.full(shape=(10,), fill_value=11)
                                        })
    assert out1.shape == (10, 8, 11, 16)
    assert slice1.output.dim == 8 and slice1.output.feature_dim_axis == 1
    assert out2.shape == (10, 16, 11, 8)
    assert slice2.output.dim == 8 and slice2.output.feature_dim_axis == 3


def test_GatherLayer():
  with make_scope() as session:
    import numpy as np
    net = TFNetwork(extern_data=ExternData())
    batch_dim, gather_dim, time_dim, feature_dim1, feature_dim2 = 3, 4, 2, 1, 2
    # [B, D, T, F1]
    values = InternalLayer(
      name="values", network=net,
      out_type={"batch_dim_axis": 0, "time_dim_axis": 2, "feature_dim_axis": 3,
        "shape": [gather_dim, None, feature_dim1], "sparse": False})
    # [B, T, F2]
    position = InternalLayer(
      name="position", network=net,
      out_type={"batch_dim_axis": 0, "time_dim_axis": 1, "shape": [None, feature_dim2],
        "sparse": True, "dim": gather_dim})

    random = np.random.RandomState(42)
    values_seqs = random.rand(batch_dim, gather_dim, time_dim, feature_dim1).astype('float32')
    seq_lens = random.randint(1, time_dim, size=[batch_dim])
    seq_lens_tensor = tf.constant(seq_lens, dtype=tf.int32)
    values.output.placeholder = tf.constant(values_seqs, dtype=tf.float32)
    values.output.size_placeholder = {1: seq_lens_tensor}
    position_seqs = random.randint(low=0, high=gather_dim, size=[batch_dim, time_dim, feature_dim2])
    position.output.placeholder = tf.constant(position_seqs, dtype=tf.int32)
    position.output.size_placeholder = {0: seq_lens_tensor}
    position.output.sanity_check()
    values.output.sanity_check()

    # should become [B, T, F2, F1]
    layer = GatherLayer(
      name="gather", network=net,
      sources=[values], position=position, axis="static:0",
      output=GatherLayer.get_out_data_from_opts(
        name="gather", sources=[values], position=position, axis="static:0"))
    layer.output.sanity_check()
    out_seqs, size = session.run([layer.output.placeholder, layer.output.size_placeholder])
    assert isinstance(out_seqs, numpy.ndarray)

    # test shapes
    print('shapes: values', values.output, 'position', position.output, 'output', layer.output)
    assert layer.output.batch_dim_axis == 0 and layer.output.time_dim_axis == 1
    assert layer.output.batch_shape == (None, None, feature_dim2, feature_dim1)
    assert np.shape(out_seqs) == (batch_dim, time_dim, feature_dim2, feature_dim1)
    assert layer.output.dtype == values.output.dtype
    assert np.array_equal(size[0], seq_lens)

    print('values [B, D, T, F1]:', values_seqs)
    print('position [B, T, F2]:', position_seqs)
    print('produced output [B, T, F2, F1]:', out_seqs)

    # test values
    for b in range(batch_dim):
      for t in range(seq_lens[b]):
        for f2 in range(feature_dim2):
          for f1 in range(feature_dim1):
            np.testing.assert_almost_equal(out_seqs[b, t, f2, f1], values_seqs[b, position_seqs[b, t, f2], t, f1])


def test_GatherLayer_constant_position():
  with make_scope() as session:
    import numpy as np
    net = TFNetwork(extern_data=ExternData())
    batch_dim, gather_dim, feature_dim1, feature_dim2 = 3, 4, 1, 2
    # [B, F1, D, F2]
    values = InternalLayer(
      name="values", network=net,
      out_type={"batch_dim_axis": 0, "feature_dim_axis": 3, "time_dim_axis": None,
        "shape": [feature_dim1, gather_dim, feature_dim2]})
    position = 3

    random = np.random.RandomState(42)
    values_seqs = random.rand(batch_dim, feature_dim1, gather_dim, feature_dim2).astype('float32')
    values.output.placeholder = tf.constant(values_seqs, dtype=tf.float32)
    values.output.sanity_check()

    # should become [B, F1, F2]
    layer = GatherLayer(
      name="gather", network=net,
      sources=[values], position=position, axis="static:-2",
      output=GatherLayer.get_out_data_from_opts(
        name="gather", sources=[values], position=position, axis="static:-2"))
    layer.output.sanity_check()
    out_seqs = session.run(layer.output.placeholder)
    assert isinstance(out_seqs, numpy.ndarray)

    # test shapes
    print('shapes: values', values.output, 'position', position, 'output', layer.output)
    assert layer.output.batch_dim_axis == 0 and layer.output.feature_dim_axis == 2
    assert layer.output.batch_shape == (None, feature_dim1, feature_dim2)
    assert np.shape(out_seqs) == (batch_dim, feature_dim1, feature_dim2)
    assert layer.output.dtype == values.output.dtype

    print('values [B, F1, D, F2]:', values_seqs)
    print('position:', position)
    print('produced output [B, F1, F2]:', out_seqs)

    # test values
    for b in range(batch_dim):
      for f1 in range(feature_dim1):
        for f2 in range(feature_dim2):
          np.testing.assert_almost_equal(out_seqs[b, f1, f2], values_seqs[b, f1, position, f2])


def test_SliceNdLayer():
  n_batch = 5
  n_time = 7
  n_dim = 11
  rnd = numpy.random.RandomState(42)
  seqs = rnd.randint(1, 100, (n_batch, n_time, n_dim)).astype("float32")  # all != 0
  seq_lens = numpy.array([n_time, n_time - 2, n_time - 3, n_time - 1, n_time - 4], dtype="int32")
  starts = numpy.array([2, 1, 3, n_time + 1, -1], dtype="int32")
  size = 5
  with make_scope() as session:
    net = TFNetwork(extern_data=ExternData())
    src = InternalLayer(name="src", network=net, out_type={"dim": n_dim})
    src.output.placeholder = tf.constant(seqs)
    src.output.size_placeholder = {0: tf.constant(seq_lens)}
    src.output.sanity_check()
    start = InternalLayer(name="start", network=net, out_type={"dim": None, "sparse": True, "time_dim_axis": None})
    start.output.placeholder = tf.constant(starts)
    start.output.sanity_check()
    kwargs = dict(name="slice", network=net, sources=[src], start=start, size=size)
    kwargs["output"] = SliceNdLayer.get_out_data_from_opts(**kwargs)
    layer = SliceNdLayer(**kwargs)
    print(layer)
    assert not layer.output.size_placeholder
    assert layer.output.batch_shape == (None, size, n_dim)
    out = session.run(layer.output.placeholder)
    print(out)
    assert isinstance(out, numpy.ndarray)
    assert out.shape == (n_batch, size, n_dim)
    for b in range(n_batch):
      s = starts[b]
      if s < 0:
        assert s + size > 0
        orig_seq = numpy.pad(seqs[b, :s + size], [(-s, 0), (0, 0)], "constant")
      else:
        orig_seq = seqs[b, s:s + size]
      if len(orig_seq) < size:
        orig_seq = numpy.pad(orig_seq, [(0, size - len(orig_seq)), (0, 0)], "constant")
      assert orig_seq.shape == (size, n_dim)
      orig_seq = numpy.where((numpy.arange(s, s + size) >= seq_lens[b])[:, None], 0.0, orig_seq)
      for t in range(size):
        numpy.testing.assert_equal(orig_seq[t], out[b, t])


def test_SliceNdLayer_dyn_size():
  n_batch = 4
  n_time = 7
  n_dim = 11
  rnd = numpy.random.RandomState(42)
  seqs = rnd.randint(1, 100, (n_batch, n_time, n_dim)).astype("float32")  # all != 0
  seq_lens = numpy.array([n_time, n_time - 2, n_time - 3, n_time - 1], dtype="int32")
  starts = numpy.array([2, 1, 3, n_time + 1], dtype="int32")
  size = None
  with make_scope() as session:
    net = TFNetwork(extern_data=ExternData())
    src = InternalLayer(name="src", network=net, out_type={"dim": n_dim})
    src.output.placeholder = tf.constant(seqs)
    src.output.size_placeholder = {0: tf.constant(seq_lens)}
    src.output.sanity_check()
    start = InternalLayer(name="start", network=net, out_type={"dim": None, "sparse": True, "time_dim_axis": None})
    start.output.placeholder = tf.constant(starts)
    start.output.sanity_check()
    kwargs = dict(name="slice", network=net, sources=[src], start=start, size=size)
    kwargs["output"] = SliceNdLayer.get_out_data_from_opts(**kwargs)
    layer = SliceNdLayer(**kwargs)
    print(layer)
    assert 0 in layer.output.size_placeholder
    assert layer.output.batch_shape == (None, size, n_dim)
    out = session.run(layer.output.placeholder)
    print(out)
    assert isinstance(out, numpy.ndarray)
    max_size = max(list(seq_lens - starts) + [0])
    assert out.shape == (n_batch, max_size, n_dim)
    for b in range(n_batch):
      s = starts[b]
      orig_seq = seqs[b, s:]
      if len(orig_seq) < max_size:
        orig_seq = numpy.pad(orig_seq, [(0, max_size - len(orig_seq)), (0, 0)], "constant")
      elif len(orig_seq) > max_size:
        orig_seq = orig_seq[:max_size]
      assert orig_seq.shape == (max_size, n_dim)
      orig_seq = numpy.where((numpy.arange(s, s + max_size) >= seq_lens[b])[:, None], 0.0, orig_seq)
      for t in range(max_size):
        numpy.testing.assert_equal(orig_seq[t], out[b, t])


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
                     'n_out': 48},
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
    with tf_compat.v1.variable_scope("src_nhwc"):
      src_nhwc = InternalLayer(name="src_nhwc", network=net, out_type={"dim": 16,
                                                                       "shape": (None, 16, 16),
                                                                       "batch_dim_axis": 0,
                                                                       "time_dim_axis": 1,
                                                                       "feature_dim_axis": 3,
                                                                       "sparse": False
                                                                       })
      src_nhwc.output.placeholder = tf_compat.v1.placeholder(shape=(None, None, 16, 16), dtype=tf.float32)
      src_nhwc.output.size_placeholder = {0: tf_compat.v1.placeholder(shape=(None,), dtype=tf.int32)}
    with tf_compat.v1.variable_scope("src_nchw"):
      src_nchw = InternalLayer(name="src_nchw", network=net, out_type={"dim": 16,
                                                                       "shape": (16, None, 16),
                                                                       "batch_dim_axis": 0,
                                                                       "time_dim_axis": 2,
                                                                       "feature_dim_axis": 1,
                                                                       "sparse": False
                                                                       })
      src_nchw.output.placeholder = tf_compat.v1.placeholder(shape=(None, 16, None, 16), dtype=tf.float32)
      src_nchw.output.size_placeholder = {1: tf_compat.v1.placeholder(shape=(None,), dtype=tf.int32)}

    filters = 64
    filter_size = (5, 5)
    strides = (1, 2)
    padding = "VALID"

    with tf_compat.v1.variable_scope("conv_nhwc_from_nhwc"):
      conv_nhwc_from_nhwc = ConvLayer(
        name="conv_nhwc_from_nhwc", network=net, n_out=filters, filter_size=filter_size,
        padding=padding, strides=strides, auto_use_channel_first=False, sources=[src_nhwc],
        output=ConvLayer.get_out_data_from_opts(name="conv_nhwc_from_nhwc", n_out=filters,
                                                filter_size=filter_size, padding=padding,
                                                auto_use_channel_first=False,
                                                network=net, sources=[src_nhwc]))
    with tf_compat.v1.variable_scope("conv_nchw_from_nhwc"):
      conv_nchw_from_nhwc = ConvLayer(
        name="conv_nchw_from_nhwc", network=net, n_out=filters, filter_size=filter_size,
        padding=padding, strides=strides, auto_use_channel_first=True, sources=[src_nhwc],
        output=ConvLayer.get_out_data_from_opts(name="conv_nchw_from_nhwc", n_out=filters,
                                                filter_size=filter_size, padding=padding,
                                                auto_use_channel_first=True,
                                                network=net, sources=[src_nhwc]))
    with tf_compat.v1.variable_scope("conv_nchw_from_nchw"):
      conv_nchw_from_nchw = ConvLayer(
        name="conv_nchw_from_nchw", network=net, n_out=filters, filter_size=filter_size,
        padding=padding, strides=strides, auto_use_channel_first=True, sources=[src_nchw],
        output=ConvLayer.get_out_data_from_opts(name="conv_nchw_from_nchw", n_out=filters,
                                                filter_size=filter_size, padding=padding,
                                                auto_use_channel_first=True,
                                                network=net, sources=[src_nchw]))
    tf_compat.v1.global_variables_initializer().run()
    out, seq_lens = session.run([conv_nhwc_from_nhwc.output.placeholder,
                                 conv_nhwc_from_nhwc.output.size_placeholder[0]],
                                feed_dict={src_nhwc.output.placeholder: np.random.rand(10, 10, 16, 16),
                                           src_nhwc.output.size_placeholder[0]: np.full(shape=(10,), fill_value=10)}
                                )
    print(out.shape)
    assert_equal(out.shape, (10, 6, 6, 64))
    print(seq_lens)
    time_dim_axis = 1 if tf_util.is_gpu_available() else 0
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
    if tf_util.is_gpu_available():
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
    with tf_compat.v1.variable_scope("src_nhwc"):
      src_nhwc = InternalLayer(name="src_nhwc", network=net, out_type={"dim": 16,
                                                                       "shape": (None, 16, 16),
                                                                       "batch_dim_axis": 0,
                                                                       "time_dim_axis": 1,
                                                                       "feature_dim_axis": 3,
                                                                       "sparse": False
                                                                       })
      src_nhwc.output.placeholder = tf_compat.v1.placeholder(shape=(None, None, 16, 16), dtype=tf.float32)
      src_nhwc.output.size_placeholder = {0: tf_compat.v1.placeholder(shape=(None,), dtype=tf.int32)}
    with tf_compat.v1.variable_scope("src_nchw"):
      src_nchw = InternalLayer(name="src_nchw", network=net, out_type={"dim": 16,
                                                                       "shape": (16, None, 16),
                                                                       "batch_dim_axis": 0,
                                                                       "time_dim_axis": 2,
                                                                       "feature_dim_axis": 1,
                                                                       "sparse": False
                                                                       })
      src_nchw.output.placeholder = tf_compat.v1.placeholder(shape=(None, 16, None, 16), dtype=tf.float32)
      src_nchw.output.size_placeholder = {1: tf_compat.v1.placeholder(shape=(None,), dtype=tf.int32)}

    pool_size = (5, 5)
    strides = (1, 2)
    padding = "VALID"

    with tf_compat.v1.variable_scope("pool_nhwc_from_nhwc"):
      pool_nhwc_from_nhwc = PoolLayer(
        name="pool_nhwc_from_nhwc", network=net, mode="max", pool_size=pool_size,
        padding=padding, strides=strides, use_channel_first=False, sources=[src_nhwc],
        output=PoolLayer.get_out_data_from_opts(name="pool_nhwc_from_nhwc",
                                                pool_size=pool_size, padding=padding,
                                                use_channel_first=False,
                                                network=net, sources=[src_nhwc]))
    with tf_compat.v1.variable_scope("pool_nchw_from_nhwc"):
      pool_nchw_from_nhwc = PoolLayer(
        name="pool_nchw_from_nhwc", network=net, mode="max", pool_size=pool_size,
        padding=padding, strides=strides, use_channel_first=True, sources=[src_nhwc],
        output=PoolLayer.get_out_data_from_opts(name="pool_nchw_from_nhwc",
                                                pool_size=pool_size, padding=padding,
                                                use_channel_first=True,
                                                network=net, sources=[src_nhwc]))
    with tf_compat.v1.variable_scope("pool_nchw_from_nchw"):
      pool_nchw_from_nchw = PoolLayer(
        name="pool_nchw_from_nchw", network=net, mode="max", pool_size=pool_size,
        padding=padding, strides=strides, use_channel_first=True, sources=[src_nchw],
        output=PoolLayer.get_out_data_from_opts(name="pool_nchw_from_nchw",
                                                pool_size=pool_size, padding=padding,
                                                use_channel_first=True,
                                                network=net, sources=[src_nchw]))
    with tf_compat.v1.variable_scope("pool_nhwc_from_nchw"):
      pool_nhwc_from_nchw = PoolLayer(
        name="pool_nhwc_from_nchw", network=net, mode="max", pool_size=pool_size,
        padding=padding, strides=strides, use_channel_first=False, sources=[src_nchw],
        output=PoolLayer.get_out_data_from_opts(name="pool_nhwc_from_nchw",
                                                pool_size=pool_size, padding=padding,
                                                use_channel_first=False,
                                                network=net, sources=[src_nchw]))
    tf_compat.v1.global_variables_initializer().run()
    out, seq_lens = session.run([pool_nhwc_from_nhwc.output.placeholder,
                                 pool_nhwc_from_nhwc.output.size_placeholder[0]],
                                feed_dict={src_nhwc.output.placeholder: np.random.rand(10, 11, 16, 16),
                                           src_nhwc.output.size_placeholder[0]: np.full(shape=(10,), fill_value=11)}
                                )
    print(out.shape)
    assert_equal(out.shape, (10, 7, 6, 16))
    print(seq_lens)
    time_dim_axis = 1 if tf_util.is_gpu_available() else 0
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
    if tf_util.is_gpu_available():
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


def test_ReduceLayer_NCHW():
  with make_scope() as session:
    import numpy as np
    net = TFNetwork(extern_data=ExternData())
    with tf_compat.v1.variable_scope("src_nchw"):
      src_nchw = InternalLayer(name="src_nchw", network=net, out_type={"dim": 16,
                                                                       "shape": (16, None, 16),
                                                                       "batch_dim_axis": 0,
                                                                       "time_dim_axis": 2,
                                                                       "feature_dim_axis": 1,
                                                                       "sparse": False
                                                                       })
      src_nchw.output.placeholder = tf_compat.v1.placeholder(shape=(None, 16, None, 16), dtype=tf.float32)
      src_nchw.output.size_placeholder = {1: tf_compat.v1.placeholder(shape=(None,), dtype=tf.int32)}
    with tf_compat.v1.variable_scope("reduce1"):
      reduce1 = ReduceLayer(
        name="reduce1", network=net, mode="max", axis="f", sources=[src_nchw],
        output=ReduceLayer.get_out_data_from_opts(name="reduce1", mode="max", axis="f",
                                                  sources=[src_nchw]))
    with tf_compat.v1.variable_scope("reduce2"):
      reduce2 = ReduceLayer(
        name="reduce2", network=net, mode="max", axis="b", sources=[src_nchw],
        output=ReduceLayer.get_out_data_from_opts(name="reduce2", mode="max", axis="b",
                                                  sources=[src_nchw]))
    out1, out2 = session.run([reduce1.output.placeholder, reduce2.output.placeholder],
                             feed_dict={src_nchw.output.placeholder: np.random.rand(10, 16, 11, 16),
                                        src_nchw.output.size_placeholder[1]: np.full(shape=(10,), fill_value=11)})
    assert_equal(out1.shape, (10, 11, 16))
    assert_equal(out2.shape, (16, 11, 16))
    assert reduce1.output.time_dim_axis == 1
    assert reduce2.output.feature_dim_axis == 0 and reduce2.output.dim == 16
    assert reduce2.output.batch_dim_axis is None


def test_Loss_NCHW():
  with make_scope() as session:
    import numpy as np
    net = TFNetwork(extern_data=ExternData())
    with tf_compat.v1.variable_scope("src_nchw"):
      src_nchw = InternalLayer(name="src_nchw", network=net, out_type={"dim": 16,
                                                                       "shape": (16, None),
                                                                       "batch_dim_axis": 0,
                                                                       "time_dim_axis": 2,
                                                                       "feature_dim_axis": 1,
                                                                       "sparse": False
                                                                       })
      src_nchw.output.placeholder = tf_compat.v1.placeholder(shape=(None, 16, None), dtype=tf.float32)
      src_nchw.output.size_placeholder = {1: tf_compat.v1.placeholder(shape=(None,), dtype=tf.int32)}

    with tf_compat.v1.variable_scope("activation"):
      activation = ActivationLayer(
        name="activation", activation="softmax", network=net, sources=[src_nchw],
        output=ActivationLayer.get_out_data_from_opts(name="activation", activation="softmax", network=net,
                                                      sources=[src_nchw]))

    target_placeholder = tf_compat.v1.placeholder(shape=(None, None, 16), dtype=tf.float32)
    target_size_placeholder = tf_compat.v1.placeholder(shape=(None,), dtype=tf.int32)
    target_data = Data(name="target", shape=(None, 16), placeholder=target_placeholder,
                       size_placeholder={0: target_size_placeholder},
                       time_dim_axis=1, feature_dim_axis=2)

    with tf_compat.v1.variable_scope("loss"):
      loss = CrossEntropyLoss(base_network=net)
      loss.init(output=activation.output, output_with_activation=activation.output_before_activation,
                target=target_data, layer=activation)

    random_input = np.random.rand(10, 16, 32)
    loss_out, out_flat = session.run([loss.get_value(), loss.output_flat],
                                     feed_dict={src_nchw.output.placeholder: random_input,
                                                src_nchw.output.size_placeholder[1]: np.full(shape=(10,), fill_value=32),
                                                target_placeholder: np.random.rand(10, 32, 16),
                                                target_size_placeholder: np.full(shape=(10,), fill_value=32)
                                                })
    print(loss_out)
    assert loss.output.feature_dim_axis == 2
    assert out_flat.shape == (320, 16)


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


def test_PostfixInTimeLayer():
  with make_scope() as session:
    import numpy as np
    net = TFNetwork(extern_data=ExternData())
    src = InternalLayer(name="src", network=net, out_type={"dim": 2, "dtype": "int32"})
    src_seqs = np.array([[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], [[6, 6], [7, 7], [8, 8], [0, 0], [0, 0]]])
    src_seq_lens = [5, 3]
    src.output.placeholder = tf.constant(src_seqs, dtype=tf.int32)
    src.output.size_placeholder = {0: tf.constant(src_seq_lens, dtype=tf.int32)}

    static_postfix = -7
    layer_postfix = InternalLayer(
      name="postfix", network=net, out_type={"dim": 2, "time_dim_axis": None, "dtype": "int32"})
    layer_postfix.output.placeholder = tf.constant([[-7, -8], [-9, -10]], dtype=tf.int32)

    for postfix in [static_postfix, layer_postfix]:
      for repeat in (1, 3):
        layer = PostfixInTimeLayer(
          name="postfix_in_time", network=net,
          sources=[src], postfix=postfix, repeat=repeat,
          output=PostfixInTimeLayer.get_out_data_from_opts(
            name="postfix_in_time", network=net, sources=[src], postfix=postfix, repeat=repeat))
        out, seq_lens = session.run([layer.output.placeholder, layer.output.size_placeholder[0]])
        print(out)
        print(seq_lens)
        assert isinstance(out, numpy.ndarray)
        assert isinstance(seq_lens, numpy.ndarray)
        assert out.shape == (2, 5 + repeat, 2)
        assert all(new_len == src_len + repeat for new_len, src_len in zip(seq_lens, src_seq_lens))
        assert out[0, src_seq_lens[0] - 1, 0] == src_seqs[0, src_seq_lens[0] - 1, 0]
        assert out[1, src_seq_lens[1] - 1, 0] == src_seqs[1, src_seq_lens[1] - 1, 0]
        assert out[0, src_seq_lens[0], 0] == -7
        assert out[0, src_seq_lens[0] + repeat - 1, 0] == -7


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

def test_DotLayer2():
  """ Test if DotLayer can handle inputs which dont have a batch-dim
  """
  with make_scope() as session:
    B = 3
    S1, S2, R, V = 2, 4, 8, 16
    net = TFNetwork(extern_data=ExternData())

    a = InternalLayer(name='A',
                      network=net,
                      out_type={'shape': (S1, S2, R),
                                'batch_dim_axis': 0,
                                'time_dim_axis': None})
    assert a.output.batch_dim_axis == 0
    assert a.output.time_dim_axis is None
    assert a.output.shape == (S1, S2, R)
    assert a.output.dim == R
    a.output.placeholder = tf.reshape(tf.range(B * S1 * S2 * R, dtype=tf.float32), (B, S1, S2, R))
    a.output.size_placeholder = {}

    b = InternalLayer(name='B',
                      network=net,
                      out_type={'shape': (S1, S2, R, V),
                                'batch_dim_axis': None,
                                'time_dim_axis': None})
    assert b.output.batch_dim_axis == None
    assert b.output.time_dim_axis == None
    assert b.output.shape == (S1, S2, R, V)
    assert b.output.dim == V
    b.output.placeholder = tf.reshape(tf.range(S1 * S2 * R * V, dtype=tf.float32), (S1, S2, R, V))
    b.output.size_placeholder = {}

    kwargs = dict(
      name="dot", network=net, sources=[a, b], debug=True,
      red1='F', red2='spatial:-1', var1='B', var2='F')
    layer = DotLayer(output=DotLayer.get_out_data_from_opts(**kwargs), **kwargs)
    print(layer, layer.output)
    assert layer.output.batch_dim_axis == 2
    assert layer.output.time_dim_axis is None
    assert layer.output.shape == (S1, S2, V)
    assert layer.output.batch_shape == (S1, S2, None, V)
    assert layer.output.dim == V
    out = session.run(layer.output.placeholder)
    assert isinstance(out, numpy.ndarray)
    assert_equal(out.shape, (S1, S2, B, V))


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


def test_ReuseParams_rec():
  print("test_ReuseParams_rec()")
  numpy.set_printoptions(precision=15)
  num_inputs = 100
  num_outputs = 15
  config = Config()
  config.update({
    "num_inputs": num_inputs,
    "num_outputs": {"data": [num_inputs, 2], "classes": [num_outputs, 1]},  # dense output
    "network": {
      "out1":         {"class": "softmax", "from": ["rec_fwd"], "loss": "ce", "n_out": num_outputs},
      "out2":         {"class": "softmax", "from": ["rec_fwd_copy"], "loss": "ce", "n_out": num_outputs},
      "rec_fwd":      {"class": "rec", "direction": 1, "from": ["data"], "n_out": 300, "unit": "lstmp"},
      "rec_fwd_copy": {"class": "rec", "direction": 1, "from": ["data"], "n_out": 300, "unit": "lstmp", "reuse_params": "rec_fwd"}
    },
    "adam": True,
    "target": "classes",
    "debug_grad_summaries": True,
    "debug_save_updater_vars": True,
    "debug_add_check_numerics_ops": True,
  })
  print("Creating network...")
  network = TFNetwork(config=config, train_flag=True)
  network.construct_from_dict(config.typed_dict["network"])
  random = numpy.random.RandomState(seed=1)
  def make_feed_dict(seq_len=10):
    return {
      network.extern_data.data["data"].placeholder: random.uniform(-1, 1, (1, seq_len, num_inputs)),
      network.extern_data.data["data"].size_placeholder[0]: numpy.array([seq_len]),
      network.extern_data.data["classes"].placeholder: random.randint(low=0, high=num_outputs, size=(1, seq_len)),
      network.extern_data.data["classes"].size_placeholder[0]: numpy.array([seq_len]),
    }
  print("Creating session...")
  with tf_compat.v1.Session() as session:
    print("Init params...")
    network.initialize_params(session=session)
    print("Testing reuse_params ...")
    feed = make_feed_dict(10)
    fwd_out, fwd_out_copy = session.run([network.layers["rec_fwd"].output.placeholder, network.layers["rec_fwd_copy"].output.placeholder], feed_dict=feed)
    numpy.testing.assert_array_equal(fwd_out, fwd_out_copy)


def test_ReuseParams_dep_loop():
  num_inputs = 10
  num_outputs = 15
  config = Config()
  config.update({
    "num_inputs": num_inputs,
    "num_outputs": {"data": [num_inputs, 2], "classes": [num_outputs, 1]},  # dense output
    "network": {
      "layer1": {"class": "rec", "from": "data", "n_out": 10, "unit": {
        "sub1": {
          "class": "linear", "from": ["data:source", "prev:output"], "activation": "relu", "n_out": 10
        },
        "sub2": {
          "class": "linear", "from": "sub1", "activation": "relu", "n_out": 10,
          "reuse_params": "base:layer2",  # circular dependency!
        },
        "output": {
          "class": "linear", "from": ["sub1", "sub2", "prev:output"], "activation": "relu", "n_out": 10
        }
      }},
      "layer2": {"class": "linear", "from": "layer1", "activation": "relu", "n_out": 10},
      "out": {"class": "softmax", "from": "layer2", "loss": "ce", "n_out": num_outputs},
    },
    "adam": True,
    "target": "classes",
    "debug_print_layer_output_template": True,
  })
  print("Creating network...")
  network = TFNetwork(config=config, train_flag=True)
  network.construct_from_dict(config.typed_dict["network"])

  params = network.get_params_list()
  pprint(params)
  l1 = network.get_layer("layer1")
  from returnn.tf.layers.rec import RecLayer, _SubnetworkRecCell
  assert isinstance(l1, RecLayer)
  cell = l1.cell
  assert isinstance(cell, _SubnetworkRecCell)
  l1 = cell.net.layers["sub2"]
  assert isinstance(l1, LinearLayer)
  assert tf_util.has_control_flow_context(l1.output.placeholder)  # this should be in the loop
  l2 = network.get_layer("layer2")
  assert not tf_util.has_control_flow_context(l2.output.placeholder)  # outside the loop
  assert isinstance(l2, LinearLayer)
  assert set(l1.params.keys()) == set(l2.params.keys()) == {"W", "b"}
  assert l1.params["W"] is l2.params["W"]
  assert l1.params["b"] is l2.params["b"]

  def make_feed_dict(seq_len=10):
    random = numpy.random.RandomState(seed=1)
    return {
      network.extern_data.data["data"].placeholder: random.uniform(-1, 1, (1, seq_len, num_inputs)),
      network.extern_data.data["data"].size_placeholder[0]: numpy.array([seq_len]),
      network.extern_data.data["classes"].placeholder: random.randint(low=0, high=num_outputs, size=(1, seq_len)),
      network.extern_data.data["classes"].size_placeholder[0]: numpy.array([seq_len]),
    }

  with tf_compat.v1.Session() as session:
    network.initialize_params(session=session)
    feed = make_feed_dict(10)
    # Not really needed (for testing reuse_params), but just test anyway.
    session.run(network.get_default_output_layer().output.placeholder, feed_dict=feed)


def test_ReuseParams_dep_loop_2():
  num_inputs = 10
  num_outputs = 15
  config = Config()
  config.update({
    "num_inputs": num_inputs,
    "num_outputs": {"data": [num_inputs, 2], "classes": [num_outputs, 1]},  # dense output
    "network": {
      "layer1": {"class": "rec", "from": "data", "n_out": 10, "unit": {
        "sub1": {
          "class": "linear", "from": ["data:source", "prev:output"], "activation": "relu", "n_out": 10
        },
        "sub2": {
          "class": "linear", "from": "sub1", "activation": "relu", "n_out": 10,
          "reuse_params": "base:layer2",  # circular dependency!
          "is_output_layer": True,
        },
        "output": {
          "class": "linear", "from": ["sub1", "sub2", "prev:output"], "activation": "relu", "n_out": 10
        }
      }},
      "layer2": {"class": "linear", "from": "layer1/sub2", "activation": "relu", "n_out": 10},
      "out": {"class": "softmax", "from": "layer2", "loss": "ce", "n_out": num_outputs},
    },
    "adam": True,
    "target": "classes",
    "debug_print_layer_output_template": True,
  })
  print("Creating network...")
  network = TFNetwork(config=config, train_flag=True)
  network.construct_from_dict(config.typed_dict["network"])

  params = network.get_params_list()
  pprint(params)
  l1 = network.get_layer("layer1")
  from returnn.tf.layers.rec import RecLayer, _SubnetworkRecCell
  assert isinstance(l1, RecLayer)
  cell = l1.cell
  assert isinstance(cell, _SubnetworkRecCell)
  l1 = cell.net.layers["sub2"]
  assert isinstance(l1, LinearLayer)
  assert tf_util.has_control_flow_context(l1.output.placeholder)  # this should be in the loop
  l2 = network.get_layer("layer2")
  assert not tf_util.has_control_flow_context(l2.output.placeholder)  # outside the loop
  assert isinstance(l2, LinearLayer)
  assert set(l1.params.keys()) == set(l2.params.keys()) == {"W", "b"}
  assert l1.params["W"] is l2.params["W"]
  assert l1.params["b"] is l2.params["b"]

  def make_feed_dict(seq_len=10):
    random = numpy.random.RandomState(seed=1)
    return {
      network.extern_data.data["data"].placeholder: random.uniform(-1, 1, (1, seq_len, num_inputs)),
      network.extern_data.data["data"].size_placeholder[0]: numpy.array([seq_len]),
      network.extern_data.data["classes"].placeholder: random.randint(low=0, high=num_outputs, size=(1, seq_len)),
      network.extern_data.data["classes"].size_placeholder[0]: numpy.array([seq_len]),
    }

  with tf_compat.v1.Session() as session:
    network.initialize_params(session=session)
    feed = make_feed_dict(10)
    # Not really needed (for testing reuse_params), but just test anyway.
    session.run(network.get_default_output_layer().output.placeholder, feed_dict=feed)


def test_ReuseParams_dep_loop_3():
  num_inputs = 10
  num_outputs = 15
  config = Config()
  config.update({
    "num_inputs": num_inputs,
    "num_outputs": {"data": [num_inputs, 2], "classes": [num_outputs, 1]},  # dense output
    "network": {
      "layer1": {"class": "rec", "from": "data", "n_out": 10, "unit": {
        "sub1": {
          "class": "linear", "from": ["data:source", "prev:output"], "activation": "relu", "n_out": 10,
          "is_output_layer": True,
        },
        "sub2": {
          "class": "linear", "from": "sub1", "activation": "relu", "n_out": 10,
          "reuse_params": "base:layer2",  # circular dependency!
        },
        "output": {
          "class": "linear", "from": ["sub1", "sub2", "prev:output"], "activation": "relu", "n_out": 10
        }
      }},
      "layer2": {"class": "linear", "from": "layer1/sub1", "activation": "relu", "n_out": 10},
      "out": {"class": "softmax", "from": "layer2", "loss": "ce", "n_out": num_outputs},
    },
    "adam": True,
    "target": "classes",
    "debug_print_layer_output_template": True,
  })
  print("Creating network...")
  network = TFNetwork(config=config, train_flag=True)
  network.construct_from_dict(config.typed_dict["network"])

  params = network.get_params_list()
  pprint(params)
  l1 = network.get_layer("layer1")
  from returnn.tf.layers.rec import RecLayer, _SubnetworkRecCell
  assert isinstance(l1, RecLayer)
  cell = l1.cell
  assert isinstance(cell, _SubnetworkRecCell)
  l1 = cell.net.layers["sub2"]
  assert isinstance(l1, LinearLayer)
  assert tf_util.has_control_flow_context(l1.output.placeholder)  # this should be in the loop
  l2 = network.get_layer("layer2")
  assert not tf_util.has_control_flow_context(l2.output.placeholder)  # outside the loop
  assert isinstance(l2, LinearLayer)
  assert set(l1.params.keys()) == set(l2.params.keys()) == {"W", "b"}
  assert l1.params["W"] is l2.params["W"]
  assert l1.params["b"] is l2.params["b"]

  def make_feed_dict(seq_len=10):
    random = numpy.random.RandomState(seed=1)
    return {
      network.extern_data.data["data"].placeholder: random.uniform(-1, 1, (1, seq_len, num_inputs)),
      network.extern_data.data["data"].size_placeholder[0]: numpy.array([seq_len]),
      network.extern_data.data["classes"].placeholder: random.randint(low=0, high=num_outputs, size=(1, seq_len)),
      network.extern_data.data["classes"].size_placeholder[0]: numpy.array([seq_len]),
    }

  with tf_compat.v1.Session() as session:
    network.initialize_params(session=session)
    feed = make_feed_dict(10)
    # Not really needed (for testing reuse_params), but just test anyway.
    session.run(network.get_default_output_layer().output.placeholder, feed_dict=feed)


def test_LossAsIs_custom_dim():
  config = Config()
  config.update({
    "extern_data": {
      "data": (40, 2),
      "classes": (10025, 1),
      "att_weights": {"shape": (None, None, 1)},
      "att_weights_sizes": {"shape": (None,), "dtype": "int32"}
    },
    "debug_print_layer_output_template": True,
  })
  print("Creating network...")
  network = TFNetwork(config=config, train_flag=True)
  net_dict = {
    "att_distill_loss": {
      "class": "eval", "from": ["energy", "att_weights"],
      "out_type": (lambda sources, **kwargs: sources[0].output.copy_template_excluding_spatial_dim(-1)),
      "eval": "softmax_cross_entropy_over_size(" +
              "logits=source(0, as_data=True, auto_convert=False)," +
              "labels=source(1, as_data=True, auto_convert=False))",
      "loss": "as_is"},
  }
  n_batch = 5
  n_enc_time = 11
  n_dec_time = 7
  with tf_compat.v1.Session() as session:
    enc_time = tf.constant([n_enc_time] * n_batch)
    dec_time = tf.constant([n_dec_time] * n_batch)
    network.add_layer(name="energy", layer_class=InternalLayer, output=Data(
      name="energy",
      shape=(None, None, 1), dim=1, batch_dim_axis=2,
      size_placeholder={0: dec_time, 1: enc_time},
      placeholder=tf.constant(numpy.random.normal(size=(n_dec_time, n_enc_time, n_batch, 1)).astype("float32"))))
    network.add_layer(name="att_weights", layer_class=InternalLayer, output=Data(
      name="att_weights",
      shape=(None, None, 1), dim=1, batch_dim_axis=0,
      size_placeholder={0: dec_time, 1: enc_time},
      placeholder=tf.expand_dims(
        tf.nn.softmax(
          tf.constant(numpy.random.normal(size=(n_batch, n_dec_time, n_enc_time)).astype("float32"))), -1)))
    network.construct_from_dict(net_dict)
    loss = session.run(network.get_total_loss())
    assert loss


def test_LossLayer_sublayers():
  from returnn.tf.util.basic import DimensionTag
  n_in, n_out = 7, 11
  time_tag = DimensionTag(kind=DimensionTag.Types.Spatial, description="time")

  config = Config({
    "extern_data": {
      'data': {"dim": n_in, "same_dim_tags_as": {"t": time_tag}},
      'classes': {'dim': n_out, 'dtype': 'int64', 'sparse': True, "same_dim_tags_as": {"t": time_tag}},
      'prev-classes': {'dim': n_out, 'dtype': 'int64', 'sparse': True, "same_dim_tags_as": {"t": time_tag}}},
    "debug_print_layer_output_template": True,
  })
  net_dict = {
    'encoder-output': {"class": "linear", "activation": "relu", "n_out": 10},

    'left-output': {'class': 'softmax', 'from': 'encoder-output', 'n_out': n_out},
    'left-output-ce': {'class': 'loss',
                       'from': 'left-output',
                       'loss': 'as_is',
                       'loss_': 'ce',
                       'loss_scale': 0,
                       'target_': 'prev-classes'},
    'left-err': {'class': 'copy', 'from': 'left-output-ce/error', 'loss': 'as_is', 'loss_scale': 0},
    'left-loss': {'class': 'copy', 'from': 'left-output-ce', 'loss': 'as_is'},

    'past-embed': {'activation': None, 'class': 'linear', 'from': ['data:prev-classes'], 'n_out': 10},
    'center-output': {'class': 'softmax', 'from': ['encoder-output', 'past-embed'], 'n_out': n_out},
    'center-output-ce': {'class': 'loss',
                         'from': 'center-output',
                         'loss': 'as_is',
                         'loss_': 'ce',
                         'loss_scale': 0,
                         'target_': 'classes'},
    'center-err': {'class': 'copy', 'from': 'center-output-ce/error', 'loss': 'as_is', 'loss_scale': 0},
    'center-loss': {'class': 'copy', 'from': 'center-output-ce', 'loss': 'as_is'},
  }
  print("Layers:", sorted(net_dict.keys()))

  print("Training")
  with make_scope() as session:
    network = TFNetwork(config=config, train_flag=True)
    # Check defaults available for inference.
    assert network.extern_data.data["data"].available_for_inference
    assert not network.extern_data.data["classes"].available_for_inference
    assert not network.extern_data.data["prev-classes"].available_for_inference
    network.construct_from_dict(net_dict)
    optimizer = tf_compat.v1.train.AdamOptimizer()
    fetches_dict = network.get_fetches_dict(should_train=True, should_eval=True, with_size=True)
    fetches_dict["optim_op"] = optimizer.minimize(network.get_objective())
    feed_dict = make_feed_dict(network.extern_data, same_time=True)
    session.run(tf_compat.v1.global_variables_initializer())
    for step in range(3):
      try:
        results = session.run(fetches_dict, feed_dict=feed_dict)
      except tf.errors.OpError as exc:
        help_on_tf_exception(
          session=session,
          exception=exc, fetches=fetches_dict, feed_dict=feed_dict,
          extern_data=network.extern_data)
        raise
      pprint(results)

  print("Forwarding")
  with make_scope() as session:
    network = TFNetwork(config=config, train_flag=False)
    network.construct_from_dict(net_dict)
    # Not sure if we would always expect that all layers with losses are constructed in this case,
    # but this is current behavior.
    # In any case, if we do that, they should infer the available_for_inference correctly.
    assert network.layers["encoder-output"].output.available_for_inference
    assert network.layers["left-output"].output.available_for_inference
    assert not network.layers["past-embed"].output.available_for_inference
    assert not network.layers["center-output"].output.available_for_inference
    assert not network.layers["center-output-ce"].output.available_for_inference
    assert not network.layers["center-loss"].output.available_for_inference
    assert not network.layers["center-err"].output.available_for_inference
    fetches_dict = network.get_fetches_dict(should_train=False, should_eval=False, with_size=True)
    feed_dict = make_feed_dict(network.extern_data, same_time=True)
    session.run(tf_compat.v1.global_variables_initializer())
    results = session.run(fetches_dict, feed_dict=feed_dict)
    pprint(results)

  print("Forwarding with fixed available-for-inference")
  with make_scope() as session:
    config.typed_dict["extern_data"]["prev-classes"]["available_for_inference"] = True
    network = TFNetwork(config=config, train_flag=False)
    network.construct_from_dict(net_dict)
    assert network.layers["encoder-output"].output.available_for_inference
    assert network.layers["left-output"].output.available_for_inference
    assert network.layers["past-embed"].output.available_for_inference
    assert network.layers["center-output"].output.available_for_inference
    assert not network.layers["center-output-ce"].output.available_for_inference
    assert not network.layers["center-loss"].output.available_for_inference
    assert not network.layers["center-err"].output.available_for_inference
    fetches_dict = network.get_fetches_dict(should_train=False, should_eval=False, with_size=True)
    feed_dict = make_feed_dict(network.extern_data, same_time=True)
    session.run(tf_compat.v1.global_variables_initializer())
    results = session.run(fetches_dict, feed_dict=feed_dict)
    pprint(results)


def test_param_variational_noise():
  from returnn.tf.util.basic import print_graph_output, find_ops_with_tensor_input
  config = Config({
    "debug_print_layer_output_template": True,
    "param_variational_noise": 0.075,
    "extern_data": {"data": {"dim": 7}}
  })
  with make_scope() as session:
    network = TFNetwork(config=config, train_flag=True)
    # Do subnetwork by intention, to test when we have multiple variable scopes.
    network.construct_from_dict({
      "output": {
        "class": "subnetwork",
        "subnetwork": {
          "output": {"class": "linear", "n_out": 13, "activation": "tanh"}
        }
      }
    })
    out = network.get_default_output_layer().output.placeholder
    print("output:")
    print_graph_output(out)
    params = network.get_params_list()
    print("params:", params)
    assert len(params) == 2  # weights and bias
    for param in params:
      print("param:", param)
      ops = find_ops_with_tensor_input(param, fetches=out)
      print("param graph:")
      print_graph_output(ops)
      assert len(ops) == 1 and "_variational_noise/" in ops[0].name


def test_LinearLayer_simple_train():
  config = Config()
  n_in, n_out = 7, 3
  config.update({
    "extern_data": {
      "data": (n_in, 2),
      "classes": (n_out, 1),
    },
    "debug_print_layer_output_template": True,
  })
  print("Creating network...")
  with tf.Graph().as_default():
    network = TFNetwork(config=config, train_flag=True)

    net_dict = {}
    layer_n_out = 10
    layer_common_args = {"class": "linear", "activation": "relu", "n_out": layer_n_out, "L2": 0.01}

    def layer(sources, **kwargs):
      args = kwargs.copy()
      for k, v in layer_common_args.items():
        args.setdefault(k, v)
      args.setdefault("from", sources)
      return args

    def make_network(num_layers):
      sources = ["data"]
      for i in range(num_layers):
        net_dict["layer%i" % i] = layer(sources=sources)
        sources = ["layer%i" % i]
      net_dict["output"] = {"class": "softmax", "loss": "ce", "from": sources}

    make_network(num_layers=3)
    network.construct_from_dict(net_dict)
    data_input = network.extern_data.get_default_input_data()
    data_target = network.extern_data.get_default_target_data()
    optimizer = tf_compat.v1.train.AdamOptimizer()
    network.maybe_construct_objective()
    update_op = optimizer.minimize(network.get_objective())
    n_batch = 5
    n_time = 11
    rnd = numpy.random.RandomState(42)
    with tf_compat.v1.Session() as session:
      session.run(tf_compat.v1.global_variables_initializer())
      for step in range(5):
        info, _ = session.run(
          (network.get_fetches_dict(), update_op),
          feed_dict={
            data_input.placeholder: rnd.normal(size=(n_batch, n_time, n_in)).astype("float32"),
            data_input.size_placeholder[0]: numpy.array([n_time] * n_batch, dtype="int32"),
            data_target.placeholder: rnd.randint(0, n_out, size=(n_batch, n_time), dtype="int32"),
            data_target.size_placeholder[0]: numpy.array([n_time] * n_batch, dtype="int32"),
          })
        print("step:", step, "info:", info)


def test_flat_net_construction():
  config = Config()
  n_in, n_out = 7, 3
  config.update({
    "extern_data": {
      "data": (n_in, 2),
      "classes": (n_out, 1),
    },
    "flat_net_construction": True,
    "debug_print_layer_output_template": True,
  })
  print("Creating network...")
  with tf.Graph().as_default():
    network = TFNetwork(config=config, train_flag=True)

    net_dict = {
      "pre0": {"class": "linear", "activation": "tanh", "from": "data", "n_out": 10},
      "pre1": {"class": "linear", "activation": "tanh", "from": "pre0", "n_out": 10},
      "pre2": {"class": "linear", "activation": "tanh", "from": "pre1", "n_out": 10}
    }
    layer_common_args = {"class": "copy"}

    def layer(sources, **kwargs):
      args = kwargs.copy()
      for k, v in layer_common_args.items():
        args.setdefault(k, v)
      args.setdefault("from", sources)
      return args

    def make_network(num_layers):
      sources = ["pre2"]
      for i in range(num_layers):
        net_dict["layer%i" % i] = layer(sources=sources)
        sources = ["layer%i" % i]
      net_dict["output"] = {"class": "softmax", "loss": "ce", "from": sources}

    make_network(num_layers=5000)
    network.construct_from_dict(net_dict)
    data_input = network.extern_data.get_default_input_data()
    data_target = network.extern_data.get_default_target_data()
    optimizer = tf_compat.v1.train.AdamOptimizer()
    network.maybe_construct_objective()
    update_op = optimizer.minimize(network.get_objective())
    n_batch = 5
    n_time = 11
    rnd = numpy.random.RandomState(42)
    with tf_compat.v1.Session() as session:
      session.run(tf_compat.v1.global_variables_initializer())
      for step in range(5):
        info, _ = session.run(
          (network.get_fetches_dict(), update_op),
          feed_dict={
            data_input.placeholder: rnd.normal(size=(n_batch, n_time, n_in)).astype("float32"),
            data_input.size_placeholder[0]: numpy.array([n_time] * n_batch, dtype="int32"),
            data_target.placeholder: rnd.randint(0, n_out, size=(n_batch, n_time), dtype="int32"),
            data_target.size_placeholder[0]: numpy.array([n_time] * n_batch, dtype="int32"),
          })
        print("step:", step, "info:", info)


def test_SyntheticGradientLayer():
  """
  Tests :class:`SyntheticGradientLayer`.
  """
  config = Config()
  n_in, n_out = 7, 3
  config.update({
    "extern_data": {
      "data": (n_in, 2),
      "classes": (n_out, 1),
    },
    "debug_print_layer_output_template": True,
  })
  print("Creating network...")
  with tf.Graph().as_default():
    network = TFNetwork(config=config, train_flag=True)

    net_dict = {}
    layer_n_out = 10
    layer_common_args = {"class": "linear", "activation": "relu", "n_out": layer_n_out, "L2": 0.01}

    def layer(sources, **kwargs):
      args = kwargs.copy()
      for k, v in layer_common_args.items():
        args.setdefault(k, v)
      args.setdefault("from", sources)
      return args

    def make_network(num_layers):
      sources = ["data"]
      for i in range(num_layers):
        net_dict["layer%i" % i] = layer(sources=sources)
        sources = ["layer%i" % i]
        net_dict["predict_grad%i" % i] = layer(sources=sources)
        net_dict["syn_grad%i" % i] = {"class": "synthetic_gradient", "gradient": "predict_grad%i" % i, "from": sources}
        sources = ["syn_grad%i" % i]
      net_dict["output"] = {"class": "softmax", "loss": "ce", "from": sources}

    make_network(num_layers=3)
    network.construct_from_dict(net_dict)
    data_input = network.extern_data.get_default_input_data()
    data_target = network.extern_data.get_default_target_data()
    from returnn.tf.updater import Updater
    updater = Updater(config=config, network=network, initial_learning_rate=0.001)
    updater.set_trainable_vars(tf_compat.v1.trainable_variables())
    update_op = updater.get_optim_op()
    assert updater.optim_meta_losses_dict
    fetches = network.get_fetches_dict()
    fetches.update(updater.optim_meta_losses_dict)

    n_batch = 5
    n_time = 11
    rnd = numpy.random.RandomState(42)
    with tf_compat.v1.Session() as session:
      session.run(tf_compat.v1.variables_initializer(tf_compat.v1.global_variables() + [network.global_train_step]))
      for step in range(5):
        info, _ = session.run(
          (fetches, update_op),
          feed_dict={
            data_input.placeholder: rnd.normal(size=(n_batch, n_time, n_in)).astype("float32"),
            data_input.size_placeholder[0]: numpy.array([n_time] * n_batch, dtype="int32"),
            data_target.placeholder: rnd.randint(0, n_out, size=(n_batch, n_time), dtype="int32"),
            data_target.size_placeholder[0]: numpy.array([n_time] * n_batch, dtype="int32"),
          })
        print("step:", step, "info:", info)


def test_TikhonovRegularizationLayer():
  """
  Tests :class:`TikhonovRegularizationLayer`.
  """
  config = Config()
  n_in, n_out = 7, 3
  config.update({
    "extern_data": {
      "data": (n_in, 2),
      "classes": (n_out, 1),
    },
    "debug_print_layer_output_template": True,
  })
  print("Creating network...")
  with tf.Graph().as_default():
    network = TFNetwork(config=config, train_flag=True)

    net_dict = {}
    layer_n_out = 10
    layer_common_args = {"class": "linear", "activation": "relu", "n_out": layer_n_out, "L2": 0.01}

    def layer(sources, **kwargs):
      args = kwargs.copy()
      for k, v in layer_common_args.items():
        args.setdefault(k, v)
      args.setdefault("from", sources)
      return args

    def make_network(num_layers):
      net_dict["input"] = {"class": "tikhonov_regularization", "meta_loss_scale": 0.1, "from": "data"}
      sources = ["input"]
      for i in range(num_layers):
        net_dict["layer%i" % i] = layer(sources=sources)
        sources = ["layer%i" % i]
      net_dict["output"] = {"class": "softmax", "loss": "ce", "loss_opts": {"use_fused": False}, "from": sources}

    make_network(num_layers=3)
    network.construct_from_dict(net_dict)
    data_input = network.extern_data.get_default_input_data()
    data_target = network.extern_data.get_default_target_data()
    from returnn.tf.updater import Updater
    updater = Updater(config=config, network=network, initial_learning_rate=0.001)
    updater.set_trainable_vars(tf_compat.v1.trainable_variables())
    update_op = updater.get_optim_op()
    assert updater.optim_meta_losses_dict
    fetches = network.get_fetches_dict()
    fetches.update(updater.optim_meta_losses_dict)

    n_batch = 5
    n_time = 11
    rnd = numpy.random.RandomState(42)
    with tf_compat.v1.Session() as session:
      session.run(tf_compat.v1.variables_initializer(tf_compat.v1.global_variables() + [network.global_train_step]))
      for step in range(5):
        info, _ = session.run(
          (fetches, update_op),
          feed_dict={
            data_input.placeholder: rnd.normal(size=(n_batch, n_time, n_in)).astype("float32"),
            data_input.size_placeholder[0]: numpy.array([n_time] * n_batch, dtype="int32"),
            data_target.placeholder: rnd.randint(0, n_out, size=(n_batch, n_time), dtype="int32"),
            data_target.size_placeholder[0]: numpy.array([n_time] * n_batch, dtype="int32"),
          })
        print("step:", step, "info:", info)


def test_split_info_input():
  from returnn.tf.util.basic import print_graph_output, find_ops_with_tensor_input
  config = Config({
    "debug_print_layer_output_template": True,
    "extern_data": {"data": {"dim": 7}}
  })
  net_dict = {
    "a": {"class": "linear", "activation": "tanh", "n_out": 11},
    "b": {"class": "linear", "activation": "tanh", "n_out": 13},
    "concat": {"class": "copy", "from": ["a", "b"]},
    "output": {"class": "linear", "activation": None, "with_bias": True, "from": ["concat"], "n_out": 17}
  }
  with make_scope() as session:
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(net_dict)
    out_weights = network.get_default_output_layer().params["W"]
    print("out_weights:", out_weights)
    assert isinstance(out_weights, tf.Variable)
    assert out_weights.get_shape().as_list() == [11 + 13, 17]
    # TODO: multiple checks:
    # the split info itself
    # the param init handling...
    # actually, for param init handling, input dim splits do not matter. they matter just for copying/growing-pretrain.
    # for param init handling, output dim split do matter.


def test_extra1():
  n_in, n_out = 2, 3
  config = Config({
    "extern_data": {"data": {"dim": n_in}},
    "debug_print_layer_output_template": True,
  })
  net_dict = {
    "input": {"class": "linear", "activation": "relu", "n_out": n_out, "from": "data"},
    "extra.2:input": {"class": "linear", "activation": None, "n_out": n_out, "from": "data"},
    # "extra.3:input automatically ...
    "output1": {"class": "copy", "from": "input", "is_output_layer": True},
    "output2": {"class": "activation", "from": "extra.2:input", "activation": "relu", "is_output_layer": True},
    "output3": {"class": "copy", "from": "extra.3:input", "is_output_layer": True},
  }
  with make_scope() as session:
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(net_dict)

    assert "extra.2" in network.extra_nets
    assert "extra.3" in network.extra_nets
    params = network.get_params_list()
    print("Params:", params)
    assert len(params) == 2  # W + b

    feed_dict = make_feed_dict(network.extern_data)
    session.run(tf_compat.v1.variables_initializer(tf_compat.v1.global_variables() + [network.global_train_step]))
    out1 = session.run(network.layers["output1"].output.placeholder, feed_dict=feed_dict)
    out2 = session.run(network.layers["output2"].output.placeholder, feed_dict=feed_dict)
    out3 = session.run(network.layers["output3"].output.placeholder, feed_dict=feed_dict)
    numpy.testing.assert_almost_equal(out1, out2)
    numpy.testing.assert_almost_equal(out2, out3)


def test_extra_subnet():
  n_in, n_out = 3, 3
  config = Config({
    "extern_data": {"data": {"dim": n_in}},
    "debug_print_layer_output_template": True,
  })
  net_dict = {
    "subnet": {
      "class": "subnetwork",
      "subnetwork": {
        "output": {"class": "linear", "activation": "relu", "n_out": n_out},
        "output2": {"class": "linear", "activation": "relu", "n_out": n_out, "is_output_layer": True},
      },
    },
    "extra.2:subnet": {
      "class": "subnetwork",
      "subnetwork": {
        "output": {"class": "copy"},
        "output2": {"class": "linear", "activation": None, "n_out": n_out, "is_output_layer": True},
      },
    },
    # extra.3:subnet automatically
    "sub1_output1": {"class": "copy", "from": "subnet/output", "is_output_layer": True},
    "sub1_output2": {"class": "copy", "from": "subnet/output2", "is_output_layer": True},
    "sub2_output1": {"class": "copy", "from": "extra.2:subnet/output", "is_output_layer": True},
    "sub2_output2": {
      "class": "activation", "activation": "relu", "from": "extra.2:subnet/output2", "is_output_layer": True},
    "sub3_output1": {"class": "copy", "from": "extra.3:subnet/output", "is_output_layer": True},
    "sub3_output2": {"class": "copy", "from": "extra.3:subnet/output2", "is_output_layer": True},
  }
  with make_scope() as session:
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(net_dict)
    assert "extra.2" in network.extra_nets
    assert "extra.3" in network.extra_nets
    params = network.get_params_list()
    print("Params:", params)
    assert len(params) == 4

    feed_dict = make_feed_dict(network.extern_data)
    session.run(tf_compat.v1.variables_initializer(tf_compat.v1.global_variables() + [network.global_train_step]))
    in_ = feed_dict[network.extern_data.data["data"].placeholder]
    sub1_out1 = session.run(network.layers["sub1_output1"].output.placeholder, feed_dict=feed_dict)
    sub1_out2 = session.run(network.layers["sub1_output2"].output.placeholder, feed_dict=feed_dict)
    sub2_out1 = session.run(network.layers["sub2_output1"].output.placeholder, feed_dict=feed_dict)
    sub2_out2 = session.run(network.layers["sub2_output2"].output.placeholder, feed_dict=feed_dict)
    sub3_out1 = session.run(network.layers["sub3_output1"].output.placeholder, feed_dict=feed_dict)
    sub3_out2 = session.run(network.layers["sub3_output2"].output.placeholder, feed_dict=feed_dict)
    numpy.testing.assert_almost_equal(sub1_out1, sub3_out1)
    numpy.testing.assert_almost_equal(sub1_out2, sub3_out2)
    numpy.testing.assert_almost_equal(sub1_out2, sub2_out2)
    numpy.testing.assert_almost_equal(in_, sub2_out1)


def test_extra_search():
  class Callbacks:
    history = []
    @classmethod
    def callback(cls, self, source, **kwargs):
      """
      :param LayerBase self:
      :param (int)->tf.Tensor source:
      :rtype: tf.Tensor
      """
      print("test_extra_search, callback: %r, %r; search flag %r" % (
        self.network.name, self, self.network.search_flag))
      cls.history.append(self)
      return source(0)

  n_batch, n_time, n_in, n_out = 2, 3, 7, 11
  rnd = numpy.random.RandomState(42)
  config = Config({
    "debug_print_layer_output_template": True,
    "extern_data": {"data": {"dim": n_in}}
  })
  net_dict = {
    "input": {"class": "eval", "eval": Callbacks.callback, "from": "search_post_output"},
    "extra.search:input": {"class": "eval", "eval": Callbacks.callback, "from": "data"},
    # Note: This 'output' layer is created twice: Once in main net, once in extra-net.
    "output": {"class": "subnetwork", "from": "input", "subnetwork": {
      "inner": {"class": "linear", "from": "data", "activation": "relu", "n_out": n_out},
      "output": {"class": "eval", "from": "inner", "eval": Callbacks.callback}
    }},
    "search_post_output": {"class": "linear", "from": "extra.search:output", "activation": "relu", "n_out": n_in}
  }
  with make_scope() as session:
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(net_dict)

    assert not network.search_flag
    assert "extra.search" in network.extra_nets
    extra_net = network.extra_nets["extra.search"]
    assert extra_net.search_flag
    assert "input" in network.layers
    assert "input" in extra_net.layers
    assert "output" in network.layers
    assert "output" in extra_net.layers
    layer_input = network.layers["input"]
    assert isinstance(layer_input, EvalLayer)
    assert layer_input in Callbacks.history
    assert layer_input.network is network
    layer_extra_input = extra_net.layers["input"]
    assert isinstance(layer_extra_input, EvalLayer)
    assert layer_extra_input in Callbacks.history
    assert layer_extra_input.network is extra_net
    layer_output = network.layers["output"]
    assert isinstance(layer_output, SubnetworkLayer)
    assert layer_output.network is network
    layer_output_output = layer_output.subnetwork.layers["output"]
    assert layer_output_output in Callbacks.history
    layer_extra_output = extra_net.layers["output"]
    assert isinstance(layer_extra_output, SubnetworkLayer)
    assert layer_extra_output.network is extra_net
    layer_extra_output_output = layer_extra_output.subnetwork.layers["output"]
    assert layer_extra_output_output in Callbacks.history

    fetches = network.get_fetches_dict()
    data_input = network.extern_data.data["data"]

    session.run(tf_compat.v1.variables_initializer(tf_compat.v1.global_variables() + [network.global_train_step]))
    info, out = session.run(
      (fetches, layer_output.output.placeholder),
      feed_dict={
        data_input.placeholder: rnd.normal(size=(n_batch, n_time, n_in)).astype("float32"),
        data_input.size_placeholder[0]: numpy.array([n_time] * n_batch, dtype="int32"),
      })
    print(info)
    print(out)  # random...


def test_HDFDumpLayer():
  import os
  from test_HDFDataset import get_test_tmp_file, DatasetTestReader, HDFDataset
  hdf_filename = get_test_tmp_file(".hdf")
  os.remove(hdf_filename)  # HDFDumpLayer expects that the file does not exist

  with make_scope() as session:
    n_in, n_out = 4, 3
    config = Config()
    config.update({
      "num_outputs": n_out,
      "num_inputs": n_in,
      "network": {
        "lstm": {"class": "rec", "unit": "LSTMBlock", "from": ["data"], "n_out": n_out},
        "dump": {"class": "hdf_dump", "filename": hdf_filename, "from": ["lstm"]},
        "output": {"class": "copy", "from": ["dump"]},
      }})
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_value("network"))

    session.run(tf_compat.v1.global_variables_initializer())
    out = network.layers["output"].output.placeholder
    n_batch = 1
    seq_len = 4
    input_data = numpy.array([[
      [1, -0.2, 0.3, -4],
      [2, -0.6, 0.7, -1.8],
      [1, 0.3, -0.1, -0.8],
      [0.1, -0.2, 0.2, .8]]],
      dtype="float32")
    input_tags = numpy.array([b"seq-0"], dtype="S5")
    seq_lens = numpy.array([seq_len], dtype="int32")
    assert input_data.shape == (n_batch, seq_lens[0], n_in)
    feed = {network.extern_data.data["data"].placeholder: input_data,
            network.extern_data.data["data"].size_placeholder[0]: seq_lens,
            network.extern_data.data["seq_tag"].placeholder: input_tags}
    assert_equal(feed[network.extern_data.get_default_input_data().placeholder].shape, (n_batch, seq_len, n_in))
    session.run([out, network.get_post_control_dependencies()], feed_dict=feed)

    network.call_graph_reset_callbacks()

  assert os.path.exists(hdf_filename)
  reader = DatasetTestReader(HDFDataset([hdf_filename]))
  reader.read_all()
  assert reader.num_seqs == 1
  assert reader.seq_tags == ["seq-0"]
  assert_equal(reader.seq_lens[0]["data"], seq_lens[0])
  assert_equal(reader.data["data"][0].shape, (seq_lens[0], n_out))


def test_HDFDumpLayer_sparse():
  import os
  from test_HDFDataset import get_test_tmp_file, DatasetTestReader, HDFDataset
  hdf_filename = get_test_tmp_file(".hdf")
  os.remove(hdf_filename)  # HDFDumpLayer expects that the file does not exist

  with make_scope() as session:
    n_in, n_out = 4, 5
    config = Config()
    config.update({
      "num_inputs": n_in,
      "num_outputs": n_out,
      "network": {
        "dump": {
          "class": "hdf_dump", "filename": hdf_filename, "from": "data:classes",
          "is_output_layer": True
        },
      }})
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_value("network"))

    session.run(tf_compat.v1.global_variables_initializer())
    n_batch = 1
    classes_data = numpy.array([[2, 5, 6]], dtype="int32")
    classes_seq_lens = [classes_data.shape[1]]
    assert classes_data.shape == (n_batch, classes_seq_lens[0])
    input_tags = numpy.array([b"seq-0"], dtype="S5")
    feed = {network.extern_data.data["classes"].placeholder: classes_data,
            network.extern_data.data["classes"].size_placeholder[0]: classes_seq_lens,
            network.extern_data.data["seq_tag"].placeholder: input_tags}
    session.run(network.get_fetches_dict(), feed_dict=feed)

    network.call_graph_reset_callbacks()

  assert os.path.exists(hdf_filename)
  reader = DatasetTestReader(HDFDataset([hdf_filename]))
  reader.read_all()
  assert reader.num_seqs == 1
  assert reader.seq_tags == ["seq-0"]
  assert_equal(reader.seq_lens[0]["data"], classes_seq_lens[0])
  assert_equal(reader.data["data"][0].shape, (classes_seq_lens[0],))
  assert_equal(reader.data_sparse["data"], True)
  assert_equal(reader.dataset.get_data_dim("data"), n_out)


def test_HDFDumpLayer_fixed_length():
  import os
  from test_HDFDataset import get_test_tmp_file, DatasetTestReader, HDFDataset
  hdf_filename = get_test_tmp_file(".hdf")
  os.remove(hdf_filename)  # HDFDumpLayer expects that the file does not exist

  with make_scope() as session:
    n_in, n_out = 4, 3
    config = Config()
    config.update({
      "num_outputs": n_out,
      "num_inputs": n_in,
      "network": {
        "lstm": {"class": "rec", "unit": "LSTMBlock", "from": ["data"], "n_out": n_out},
        "last_state": {"class": "get_last_hidden_state", "from": ["lstm"], "key": "h", "n_out": n_out},
        "last_state_expanded": {"class": "expand_dims", "from": ["last_state"], "axis": "T"},
        "dump": {"class": "hdf_dump", "filename": hdf_filename, "from": ["last_state_expanded"]},
        "output": {"class": "copy", "from": ["dump"]},
      }})
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_value("network"))

    session.run(tf_compat.v1.global_variables_initializer())
    out = network.layers["output"].output.placeholder
    n_batch = 1
    seq_len = 4
    input_data = numpy.array([[
      [1, -0.2, 0.3, -4],
      [2, -0.6, 0.7, -1.8],
      [1, 0.3, -0.1, -0.8],
      [0.1, -0.2, 0.2, .8]]],
      dtype="float32")
    input_tags = numpy.array([b"seq-0"], dtype="S5")
    seq_lens = numpy.array([seq_len], dtype="int32")
    assert input_data.shape == (n_batch, seq_lens[0], n_in)
    feed = {network.extern_data.data["data"].placeholder: input_data,
            network.extern_data.data["data"].size_placeholder[0]: seq_lens,
            network.extern_data.data["seq_tag"].placeholder: input_tags}
    session.run([out, network.get_post_control_dependencies()], feed_dict=feed)

    network.call_graph_reset_callbacks()

  assert os.path.exists(hdf_filename)
  reader = DatasetTestReader(HDFDataset([hdf_filename]))
  reader.read_all()
  assert reader.num_seqs == 1
  assert reader.seq_tags == ["seq-0"]
  assert_equal(reader.seq_lens[0]["data"], 1)
  assert_equal(reader.data["data"][0].shape, (1, n_out))


def test_HDFDumpLayer_extra():
  import os
  from test_HDFDataset import get_test_tmp_file, DatasetTestReader, HDFDataset
  hdf_filename = get_test_tmp_file(".hdf")
  os.remove(hdf_filename)  # HDFDumpLayer expects that the file does not exist

  with make_scope() as session:
    n_in = 5
    n_out1 = 7
    config = Config()
    config.update({
      "extern_data": {
        "data": {"dim": n_in},
        "classes1": {"dim": n_out1, "sparse": True},
        "classes2": {"dim": None, "dtype": "float32", "shape": ()},
      },
      "network": {
        "dump": {
          "class": "hdf_dump", "filename": hdf_filename,
          "from": "data",
          "extra": {"classes1": "data:classes1", "classes2": "data:classes2"},
          "is_output_layer": True
        },
      }})
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_value("network"))
    network.print_network_info()

    session.run(tf_compat.v1.global_variables_initializer())
    n_batch = 1
    input_data = numpy.array([[
      [1, -0.2, 0.3, -4, 5],
      [2, -0.6, 0.7, -1.8, 2.9],
      [1, 0.3, -0.1, -0.8, 0.5],
      [0.1, -0.2, 0.2, .8, -0.3]]],
      dtype="float32")
    input_seq_lens = [input_data.shape[1]]
    assert input_data.shape == (n_batch, input_seq_lens[0], n_in)
    classes1_data = numpy.array([[2, 5, 6]], dtype="int32")
    classes1_seq_lens = [classes1_data.shape[1]]
    assert classes1_data.shape == (n_batch, classes1_seq_lens[0])
    classes2_data = numpy.array([-7.89], dtype="float32")
    assert classes2_data.shape == (n_batch,)
    seq_tags = numpy.array([b"seq-0"], dtype="S5")
    feed = {
      network.extern_data.data["data"].placeholder: input_data,
      network.extern_data.data["data"].size_placeholder[0]: input_seq_lens,
      network.extern_data.data["classes1"].placeholder: classes1_data,
      network.extern_data.data["classes1"].size_placeholder[0]: classes1_seq_lens,
      network.extern_data.data["classes2"].placeholder: classes2_data,
      network.extern_data.data["seq_tag"].placeholder: seq_tags}
    fetches = network.get_fetches_dict()
    result = session.run(fetches, feed_dict=feed)
    pprint(result)

    network.call_graph_reset_callbacks()

  assert os.path.exists(hdf_filename)
  reader = DatasetTestReader(HDFDataset([hdf_filename]))
  reader.read_all()
  assert reader.num_seqs == 1
  assert reader.seq_tags == ["seq-0"]
  assert_equal(reader.seq_lens[0]["data"], input_seq_lens[0])
  assert_equal(reader.data["data"][0].shape, (input_seq_lens[0], n_in))
  assert_equal(reader.data["classes1"][0].shape, (classes1_seq_lens[0],))
  assert_equal(reader.data["classes2"][0].shape, (1,))
  numpy.testing.assert_almost_equal(reader.data["data"][0], input_data[0])
  numpy.testing.assert_equal(reader.data["classes1"][0], classes1_data[0])
  numpy.testing.assert_equal(reader.data["classes2"][0], [classes2_data[0]])


def test_HDFDumpLayer_dump_whole_batch_extra_sm():
  import os
  from test_HDFDataset import get_test_tmp_file, DatasetTestReader, HDFDataset
  hdf_filename = get_test_tmp_file(".hdf")
  os.remove(hdf_filename)  # HDFDumpLayer expects that the file does not exist
  rnd = numpy.random.RandomState(42)

  with make_scope() as session:
    n_in = 5
    config = Config()
    config.update({
      "extern_data": {
        "data": {"dim": n_in},
        "sm": dict(shape=(None, None)),
      },
      "network": {
        "dump": {
          "class": "hdf_dump", "filename": hdf_filename,
          "from": "data",
          "extra": {"sm": "data:sm"},
          "is_output_layer": True,
          "dump_whole_batches": True,
        },
      }})
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_value("network"))
    network.print_network_info()

    session.run(tf_compat.v1.global_variables_initializer())
    n_batch = 1
    input_data = numpy.array([[
      [1, -0.2, 0.3, -4, 5],
      [2, -0.6, 0.7, -1.8, 2.9],
      [1, 0.3, -0.1, -0.8, 0.5],
      [0.1, -0.2, 0.2, .8, -0.3]]],
      dtype="float32")
    input_seq_lens = [input_data.shape[1]]
    assert input_data.shape == (n_batch, input_seq_lens[0], n_in)
    sm_seq_lens1 = [13]
    sm_seq_lens2 = [17]
    sm_data = rnd.normal(size=(n_batch, sm_seq_lens1[0], sm_seq_lens2[0])).astype(dtype="float32")
    seq_tags = numpy.array([b"seq-0"], dtype="S5")
    feed = {
      network.extern_data.data["data"].placeholder: input_data,
      network.extern_data.data["data"].size_placeholder[0]: input_seq_lens,
      network.extern_data.data["sm"].placeholder: sm_data,
      network.extern_data.data["sm"].size_placeholder[0]: sm_seq_lens1,
      network.extern_data.data["sm"].size_placeholder[1]: sm_seq_lens2,
      network.extern_data.data["seq_tag"].placeholder: seq_tags}
    fetches = network.get_fetches_dict()
    result = session.run(fetches, feed_dict=feed)
    pprint(result)

    network.call_graph_reset_callbacks()

  assert os.path.exists(hdf_filename)
  reader = DatasetTestReader(HDFDataset([hdf_filename]))
  reader.read_all()
  assert reader.num_seqs == 1
  assert reader.seq_tags == ["seq-0"]
  assert_equal(reader.seq_lens[0]["data"], input_seq_lens[0])
  assert_equal(reader.data["data"][0].shape, (input_seq_lens[0], n_in))
  numpy.testing.assert_almost_equal(reader.data["data"][0], input_data[0])
  assert_equal(reader.data["sm"][0].shape, (sm_seq_lens1[0] * sm_seq_lens2[0],))
  numpy.testing.assert_equal(numpy.reshape(reader.data["sm"][0], sm_data[0].shape), sm_data[0])


def test_HDFDumpLayer_dump_whole_batch_extra_sm1():
  import os
  from test_HDFDataset import get_test_tmp_file, DatasetTestReader, HDFDataset
  hdf_filename = get_test_tmp_file(".hdf")
  os.remove(hdf_filename)  # HDFDumpLayer expects that the file does not exist
  rnd = numpy.random.RandomState(42)

  with make_scope() as session:
    n_in = 5
    config = Config()
    config.update({
      "extern_data": {
        "data": {"dim": n_in},
        "sm": dict(shape=(None, 1, None), batch_dim_axis=1, feature_dim_axis=2),
      },
      "network": {
        "dump": {
          "class": "hdf_dump", "filename": hdf_filename,
          "from": "data",
          "extra": {"sm": "data:sm"},
          "is_output_layer": True,
          "dump_whole_batches": True,
        },
      }})
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_value("network"))
    network.print_network_info()

    session.run(tf_compat.v1.global_variables_initializer())
    n_batch = 1
    input_data = numpy.array([[
      [1, -0.2, 0.3, -4, 5],
      [2, -0.6, 0.7, -1.8, 2.9],
      [1, 0.3, -0.1, -0.8, 0.5],
      [0.1, -0.2, 0.2, .8, -0.3]]],
      dtype="float32")
    input_seq_lens = [input_data.shape[1]]
    assert input_data.shape == (n_batch, input_seq_lens[0], n_in)
    sm_seq_lens1 = [13]
    sm_seq_lens2 = [17]
    sm_data = rnd.normal(size=(sm_seq_lens1[0], n_batch, 1, sm_seq_lens2[0])).astype(dtype="float32")
    seq_tags = numpy.array([b"seq-0"], dtype="S5")
    feed = {
      network.extern_data.data["data"].placeholder: input_data,
      network.extern_data.data["data"].size_placeholder[0]: input_seq_lens,
      network.extern_data.data["sm"].placeholder: sm_data,
      network.extern_data.data["sm"].size_placeholder[0]: sm_seq_lens1,
      network.extern_data.data["sm"].size_placeholder[2]: sm_seq_lens2,
      network.extern_data.data["seq_tag"].placeholder: seq_tags}
    fetches = network.get_fetches_dict()
    result = session.run(fetches, feed_dict=feed)
    pprint(result)

    network.call_graph_reset_callbacks()

  assert os.path.exists(hdf_filename)
  reader = DatasetTestReader(HDFDataset([hdf_filename]))
  reader.read_all()
  assert reader.num_seqs == 1
  assert reader.seq_tags == ["seq-0"]
  assert_equal(reader.data["data"][0].shape, (input_seq_lens[0], n_in))
  numpy.testing.assert_almost_equal(reader.data["data"][0], input_data[0])
  assert_equal(reader.data["sm"][0].shape, (sm_seq_lens1[0] * sm_seq_lens2[0],))
  sm_data_ = numpy.transpose(sm_data, (1, 0, 3, 2))
  numpy.testing.assert_equal(numpy.reshape(reader.data["sm"][0], sm_data_[0].shape), sm_data_[0])


def test_CrossEntropyLoss():
  with make_scope() as session:
    n_out = 13
    config = Config({
      "debug_print_layer_output_template": True,
      "extern_data": {
        "data": {"dim": n_out},
        "classes": {"dim": n_out, "sparse": True},
      }})
    net = TFNetwork(config=config, train_flag=True)
    net.construct_from_dict({
      "var": {"class": "variable", "shape": (n_out,)},
      "add": {"class": "combine", "kind": "add", "from": ["data", "var"]},
      "output": {
        "class": "activation", "from": "add", "activation": "softmax",
        "loss": "ce"},
    })
    losses_dict, total_loss, total_constraints = net.get_losses_initialized()
    print("Losses:")
    pprint(losses_dict)
    assert set(losses_dict.keys()) == {"output"}
    loss_holder = losses_dict["output"]
    assert isinstance(loss_holder, LossHolder)
    assert isinstance(loss_holder.loss, CrossEntropyLoss)
    session.run(tf_compat.v1.global_variables_initializer())
    print("Get loss:")
    feed_dict = make_feed_dict(net.extern_data.data.values(), same_time=True)
    print("random classes:", feed_dict[net.extern_data.data["classes"].placeholder])
    loss_t = loss_holder.get_loss_value()
    opt = tf_compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
    minimize_op = opt.minimize(loss_t)
    last_loss_v = float("inf")
    for step in range(3):
      loss_v, _ = session.run((loss_t, minimize_op), feed_dict=feed_dict)
      print("step %i, loss %f" % (step, loss_v))
      assert numpy.isfinite(loss_v) and numpy.isscalar(loss_v)
      assert loss_v < last_loss_v  # it's convex and we cannot overshoot
      last_loss_v = loss_v


def test_CrossEntropyLoss_masked_inf():
  with make_scope() as session:
    n_out = 13
    config = Config({
      "debug_print_layer_output_template": True,
      "extern_data": {
        "data": {"dim": n_out},
        "classes": {"dim": n_out, "sparse": True},
      }})
    mask_t = tf_compat.v1.placeholder(tf.bool, (n_out,), name="mask")

    def mask_func(source, **kwargs):
      x = source(0)
      assert x.shape.ndims == 3  # (B,T,n_out)
      from returnn.tf.util.basic import where_bc
      mask_bc = mask_t[None, None, :]  # (1,1,n_out)
      return where_bc(mask_bc, x, float("-inf"))

    net = TFNetwork(config=config, train_flag=True)
    net.construct_from_dict({
      "var": {"class": "variable", "shape": (n_out,)},  # such that we can check that there are no nan/inf grads
      "add": {"class": "combine", "kind": "add", "from": ["data", "var"]},
      "mask": {"class": "eval", "from": "add", "eval": mask_func},
      "output": {
        "class": "activation", "from": "mask", "activation": "softmax",
        "loss": "ce"},
    })
    losses_dict, total_loss, total_constraints = net.get_losses_initialized()
    print("Losses:")
    pprint(losses_dict)
    assert set(losses_dict.keys()) == {"output"}
    loss_holder = losses_dict["output"]
    assert isinstance(loss_holder, LossHolder)
    assert isinstance(loss_holder.loss, CrossEntropyLoss)
    session.run(tf_compat.v1.global_variables_initializer())
    print("Get loss:")
    feed_dict = make_feed_dict(net.extern_data.data.values(), same_time=True)
    mask_v = numpy.array([True] * n_out)
    feed_dict[mask_t] = mask_v
    loss_t = loss_holder.get_loss_value()
    opt = tf_compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
    minimize_op = opt.minimize(loss_t)
    last_loss_v = float("inf")
    for step in range(3):
      loss_v, _ = session.run((loss_t, minimize_op), feed_dict=feed_dict)
      print("step %i, loss %f" % (step, loss_v))
      assert numpy.isfinite(loss_v) and numpy.isscalar(loss_v)
      assert loss_v < last_loss_v  # it's convex and we cannot overshoot
      last_loss_v = loss_v
    print("Now mask.")
    feed_dict = make_feed_dict(net.extern_data.data.values(), same_time=True, n_batch=1, n_time=1)
    feed_dict[mask_t] = mask_v
    rnd_classes = feed_dict[net.extern_data.data["classes"].placeholder]
    print("random classes:", rnd_classes)
    mask_v[rnd_classes[0, 0]] = False
    var_t, = tf_compat.v1.trainable_variables()
    last_var_v = session.run(var_t)
    for step in range(3, 6):
      loss_v, _ = session.run((loss_t, minimize_op), feed_dict=feed_dict)
      print("step %i, loss %f" % (step, loss_v))
      assert numpy.isinf(loss_v) and numpy.isscalar(loss_v)
      var_v = session.run(var_t)
      assert numpy.isfinite(var_v).all()  # while the loss is inf, the gradients should be finite!
      assert not (var_v == last_var_v).all()  # and there also was some non-zero gradient!
      last_var_v = var_v


def test_CrossEntropyLoss_masked_inf_fake_upper_bound():
  # Almost the same as test_CrossEntropyLoss_masked_inf, but we use fake_upper_bound.
  with make_scope() as session:
    n_out = 13
    fake_upper_bound = 10.
    config = Config({
      "debug_print_layer_output_template": True,
      "extern_data": {
        "data": {"dim": n_out},
        "classes": {"dim": n_out, "sparse": True},
      }})
    mask_t = tf_compat.v1.placeholder(tf.bool, (n_out,), name="mask")

    def mask_func(source, **kwargs):
      x = source(0)
      assert x.shape.ndims == 3  # (B,T,n_out)
      from returnn.tf.util.basic import where_bc
      mask_bc = mask_t[None, None, :]  # (1,1,n_out)
      return where_bc(mask_bc, x, float("-inf"))

    net = TFNetwork(config=config, train_flag=True)
    net.construct_from_dict({
      "var": {"class": "variable", "shape": (n_out,)},  # such that we can check that there are no nan/inf grads
      "add": {"class": "combine", "kind": "add", "from": ["data", "var"]},
      "mask": {"class": "eval", "from": "add", "eval": mask_func},
      "output": {
        "class": "activation", "from": "mask", "activation": "softmax",
        "loss": "ce", "loss_opts": {"fake_upper_bound": fake_upper_bound}},
    })
    losses_dict, total_loss, total_constraints = net.get_losses_initialized()
    print("Losses:")
    pprint(losses_dict)
    assert set(losses_dict.keys()) == {"output"}
    loss_holder = losses_dict["output"]
    assert isinstance(loss_holder, LossHolder)
    assert isinstance(loss_holder.loss, CrossEntropyLoss)
    session.run(tf_compat.v1.global_variables_initializer())
    print("Get loss:")
    feed_dict = make_feed_dict(net.extern_data.data.values(), same_time=True)
    mask_v = numpy.array([True] * n_out)
    feed_dict[mask_t] = mask_v
    loss_t = loss_holder.get_loss_value()
    opt = tf_compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
    minimize_op = opt.minimize(loss_t)
    last_loss_v = float("inf")
    for step in range(3):
      loss_v, _ = session.run((loss_t, minimize_op), feed_dict=feed_dict)
      print("step %i, loss %f" % (step, loss_v))
      assert numpy.isfinite(loss_v) and numpy.isscalar(loss_v)
      assert loss_v < last_loss_v  # it's convex and we cannot overshoot
      last_loss_v = loss_v
    print("Now mask.")
    feed_dict = make_feed_dict(net.extern_data.data.values(), same_time=True, n_batch=1, n_time=1)
    feed_dict[mask_t] = mask_v
    rnd_classes = feed_dict[net.extern_data.data["classes"].placeholder]
    print("random classes:", rnd_classes)
    mask_v[rnd_classes[0, 0]] = False
    var_t, = tf_compat.v1.trainable_variables()
    last_var_v = session.run(var_t)
    for step in range(3, 6):
      loss_v, _ = session.run((loss_t, minimize_op), feed_dict=feed_dict)
      print("step %i, loss %f" % (step, loss_v))
      assert loss_v == fake_upper_bound and numpy.isscalar(loss_v)
      var_v = session.run(var_t)
      assert numpy.isfinite(var_v).all()  # while the loss is bounded, the gradients should be finite!
      assert not (var_v == last_var_v).all()  # and there also was some non-zero gradient!
      last_var_v = var_v


def test_reduce_mean_in_time():
  with make_scope() as session:
    n_out = 5
    config = Config({
      "debug_print_layer_output_template": True,
      "extern_data": {
        "data": {"dim": n_out},
      }})
    net = TFNetwork(config=config, train_flag=True)
    net.construct_from_dict({
      "output": {"class": "reduce", "mode": "mean", "axis": "T", "from": ["data"]}
    })
    session.run(tf_compat.v1.global_variables_initializer())
    out = net.layers["output"].output.placeholder
    n_batch = 3
    max_seq_len = 10
    feed = make_feed_dict(net.extern_data.data.values(), n_batch=n_batch, n_time=max_seq_len, same_time=True)
    v = session.run(out, feed_dict=feed)
    input_len = feed[net.extern_data.data["data"].size_placeholder[0]]
    input_data = feed[net.extern_data.data["data"].placeholder]

    ref = numpy.zeros([n_batch, n_out])
    for batch, seq_len in enumerate(input_len):
      ref[batch, :] = numpy.mean(input_data[batch, :seq_len, :], axis=0)

    numpy.testing.assert_allclose(ref, v, rtol=1e-5)


def test_reduce_mean_batch_time():
  with make_scope() as session:
    n_out = 5
    config = Config({
      "debug_print_layer_output_template": True,
      "extern_data": {
        "data": {"dim": n_out},
      }})
    net = TFNetwork(config=config, train_flag=True)
    net.construct_from_dict({
      "output": {"class": "reduce", "mode": "mean", "axis": ["B", "T"], "from": ["data"]}
    })
    session.run(tf_compat.v1.global_variables_initializer())
    out = net.layers["output"].output.placeholder
    n_batch = 3
    max_seq_len = 10
    feed = make_feed_dict(net.extern_data.data.values(), n_batch=n_batch, n_time=max_seq_len, same_time=True)
    v = session.run(out, feed_dict=feed)
    input_len = feed[net.extern_data.data["data"].size_placeholder[0]]
    input_data = feed[net.extern_data.data["data"].placeholder]

    input_data_masked = numpy.copy(input_data)
    for batch, seq_len in enumerate(input_len):
      input_data_masked[batch, seq_len:, :] = numpy.nan
    ref = numpy.nanmean(input_data_masked, axis=(0, 1))

    numpy.testing.assert_allclose(ref, v, rtol=1e-5)


def test_automatic_seq_lengths():
  with make_scope() as session:
    n_out = 5
    config = Config({
      "debug_print_layer_output_template": True,
      "extern_data": {
        "data": {"dim": n_out},
      }})
    net = TFNetwork(config=config, train_flag=True)
    net.construct_from_dict({
      "layer0": {
        'class': 'pad', 'mode': 'reflect', 'axes': 'spatial', 'padding': (3, 3), 'from': 'data'},  # len+6
      "layer1": {
        'class': 'conv', 'from': 'layer0', 'activation': None, 'with_bias': True, 'n_out': n_out,
        'filter_size': (7,), 'padding': 'valid', 'strides': (1,), 'dilation_rate': (1,)},  # max(len+6-6,0)
      "output": {"class": "copy", "from": "layer1"},
    })
    session.run(tf_compat.v1.global_variables_initializer())
    in_data = net.extern_data.get_default_input_data()
    out_data = net.layers["output"].output.copy_as_batch_spatial_major()
    assert_equal(out_data.shape, in_data.shape)
    n_batch = 3
    max_seq_len = 10
    feed = make_feed_dict([in_data], n_batch=n_batch, n_time=max_seq_len)
    out_lens = out_data.get_sequence_lengths()
    out_v, out_lens_v = session.run((out_data.placeholder, out_lens), feed_dict=feed)
    in_v = feed[in_data.placeholder]
    in_lens_v = feed[in_data.size_placeholder[0]]
    assert_equal(in_v.shape, out_v.shape)
    assert_equal(in_lens_v.tolist(), out_lens_v.tolist())
    # So far, everything should always be true, unless we have messed some op really up.
    # Now we want to do the main test, i.e. whether we get the same tensor.
    from returnn.tf.util.basic import print_graph_output
    print_graph_output(out_lens)
    assert out_lens is in_data.size_placeholder[0]


def test_automatic_seq_lengths2():
  with make_scope() as session:
    n_out = 5
    config = Config({
      "debug_print_layer_output_template": True,
      "extern_data": {
        "data": {"dim": n_out},
      }})
    net = TFNetwork(config=config, train_flag=True)
    net.construct_from_dict({
      "layer0": {
        'class': 'conv', 'from': 'data', 'activation': None, 'with_bias': True, 'n_out': n_out,
        'filter_size': (1,), 'padding': 'valid'},
      "output": {"class": "copy", "from": "layer0"},
    })
    session.run(tf_compat.v1.global_variables_initializer())
    in_data = net.extern_data.get_default_input_data()
    out_data = net.layers["output"].output.copy_as_batch_spatial_major()
    assert_equal(out_data.shape, in_data.shape)
    n_batch = 3
    max_seq_len = 10
    feed = make_feed_dict([in_data], n_batch=n_batch, n_time=max_seq_len)
    out_lens = out_data.get_sequence_lengths()
    out_v, out_lens_v = session.run((out_data.placeholder, out_lens), feed_dict=feed)
    in_v = feed[in_data.placeholder]
    in_lens_v = feed[in_data.size_placeholder[0]]
    assert_equal(in_v.shape, out_v.shape)
    assert_equal(in_lens_v.tolist(), out_lens_v.tolist())
    # So far, everything should always be true, unless we have messed some op really up.
    # Now we want to do the main test, i.e. whether we get the same tensor.
    from returnn.tf.util.basic import print_graph_output
    print_graph_output(out_lens)
    assert out_lens is in_data.size_placeholder[0]


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
