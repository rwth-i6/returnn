
# start: nosetests $this_file --nologcapture
from __future__ import division
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
  """
  :rtype: tf.Session
  """
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
    with tf.variable_scope("lin"):
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
    session.run(tf.global_variables_initializer())
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
    with tf.variable_scope("src_nchw"):
      src_nhwc = InternalLayer(name="src_nchw", network=net, out_type={"dim": 16,
                                                                       "shape": (None, 16, 16),
                                                                       "batch_dim_axis": 0,
                                                                       "time_dim_axis": 1,
                                                                       "feature_dim_axis": 3,
                                                                       "sparse": False
                                                                       })
      src_nhwc.output.placeholder = tf.placeholder(shape=(None, None, 16, 16), dtype=tf.float32)
      src_nhwc.output.size_placeholder = {0: tf.placeholder(shape=(None,), dtype=tf.int32)}

    rnd = np.random.RandomState(42)
    mean =  tf.constant(rnd.rand(1, 1, 1, 16), name="rand_mean", dtype=tf.float32)
    variance = tf.constant(rnd.rand(1, 1, 1, 16), name="rand_var", dtype=tf.float32)
    input_data = rnd.rand(10, 11, 16, 16)
    seq_lens = np.array([11, 11, 11, 11, 11, 11, 11, 11, 11, 11])

    with tf.variable_scope("batch_norm_masked_nchw"):
      batch_norm_1 = BatchNormLayer(name="batch_norm_masked_nchw", network=net, masked_time=True,
                                    sample_mean=mean, sample_variance=variance,
                                    sources=[src_nhwc],
                                    output=BatchNormLayer.get_out_data_from_opts(name="batch_norm_masked_nchw",
                                                                                 sources=[src_nhwc],
                                                                                 network=net))
      batch_norm_1.post_init(layer_desc=None)
    with tf.variable_scope("batch_norm_nonmasked_nchw"):
      batch_norm_2 = BatchNormLayer(name="batch_norm_nonmasked_nchw", network=net, masked_time=False,
                                    sample_mean=mean, sample_variance=variance,
                                    sources=[src_nhwc],
                                    output=BatchNormLayer.get_out_data_from_opts(name="batch_norm_nonmasked_nchw",
                                                                                 sources=[src_nhwc],
                                                                                 network=net))
      batch_norm_2.post_init(layer_desc=None)
    tf.global_variables_initializer().run()
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

    with tf.variable_scope("batch_norm_masked_nchw"):
      batch_norm_1 = BatchNormLayer(name="batch_norm_masked_nchw", network=net, masked_time=True,
                                    sample_mean=mean, sample_variance=variance,
                                    use_shift=False, use_std=False, epsilon=0.0,
                                    sources=[src_nhwc],
                                    output=BatchNormLayer.get_out_data_from_opts(name="batch_norm_masked_nchw",
                                                                                 sources=[src_nhwc],
                                                                                 network=net))
      batch_norm_1.post_init(layer_desc=None)
    with tf.variable_scope("batch_norm_nonmasked_nchw"):
      batch_norm_2 = BatchNormLayer(name="batch_norm_nonmasked_nchw", network=net, masked_time=False,
                                    sample_mean=mean, sample_variance=variance,
                                    use_shift=False, use_std=False, epsilon=0,
                                    sources=[src_nhwc],
                                    output=BatchNormLayer.get_out_data_from_opts(name="batch_norm_nonmasked_nchw",
                                                                                 sources=[src_nhwc],
                                                                                 network=net))
      batch_norm_2.post_init(layer_desc=None)
    tf.global_variables_initializer().run()
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


def _check_MergeDimsLayer(session, in_data_opts, in_static_shape, opts, out_data_shape, out_static_shape):
  """
  :param tf.Session session:
  :param dict[str] in_data_opts:
  :param tuple[int] in_static_shape:
  :param dict[str] opts: for MergeDimsLayer
  :param tuple[int|None] out_data_shape:
  :param tuple[int] out_static_shape:
  :rtype: MergeDimsLayer
  """
  net = TFNetwork(extern_data=ExternData())
  rnd = numpy.random.RandomState(42)
  src = InternalLayer(name="src", network=net, out_type=in_data_opts)
  print("input:", src.output)
  src.output.placeholder = tf.constant(rnd.normal(size=in_static_shape).astype("float32"), dtype=tf.float32)
  src.output.size_placeholder = {}  # not sure if enough...
  opts = opts.copy()
  print("opts:", opts)
  opts.update({"network": net, "name": "merge_dims_test", "sources": [src]})
  out_data = MergeDimsLayer.get_out_data_from_opts(**opts)
  print("output:", out_data)
  out_data.sanity_check(ignore_placeholder=True)  # placeholder might be overwritten later
  assert_equal(out_data.shape, out_data_shape)
  layer = MergeDimsLayer(output=out_data, **opts)
  assert_equal(layer.output.shape, out_data_shape)
  out_np = session.run(layer.output.placeholder)
  assert_equal(out_np.shape, out_static_shape)
  return layer


def test_MergeDimsLayer_basic():
  with make_scope() as session:
    _check_MergeDimsLayer(session, {"shape": (4, 7), "time_dim_axis": None}, (2, 4, 7), {"axes": "except_batch"}, (4 * 7,), (2, 4 * 7))
    _check_MergeDimsLayer(session, {"shape": (4, None, 7), "time_dim_axis": None}, (2, 4, 3, 7), {"axes": "static"}, (None, 4 * 7), (2, 3, 4 * 7))
    _check_MergeDimsLayer(session, {"shape": (4, None, 7), "time_dim_axis": 2}, (2, 4, 3, 7), {"axes": "static"}, (None, 4 * 7), (2, 3, 4 * 7))


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
    assert layer.output.time_dim_axis == 1  # Note: This is currently the behavior, but maybe we change that.


def test_MergeDimsLayer_except_time_ext():
  with make_scope() as session:
    n_batch = 11
    n_time = 13
    layer = _check_MergeDimsLayer(
      session,
      {"shape": (3, None, 5), "time_dim_axis": 2}, (n_batch, 3, n_time, 5),
      {"axes": "except_time"}, (None, 15), (n_batch, n_time, 15))
    assert layer.output.batch_dim_axis == 0 and layer.output.time_dim_axis == 1


def test_MergeDimsLayer_SplitBatchTimeLayer_time_major():
  n_batch = 3
  n_time = 4
  n_input_dim = 5
  # Time major
  input_data = numpy.arange(n_time * n_batch * n_input_dim).reshape((n_time, n_batch, n_input_dim)).astype("float32")
  with make_scope() as session:
    net = TFNetwork(extern_data=ExternData(), config=Config({"debug_print_layer_output_template": True}))
    input_layer = net.add_layer(
      "input", InternalLayer,
      output=Data(
        name="input", shape=(None, n_input_dim), time_dim_axis=0, batch_dim_axis=1,
        placeholder=tf.constant(input_data), size_placeholder={0: tf.constant([n_time] * n_batch)}))
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


def test_SliceLayer_NCHW():
  with make_scope() as session:
    import numpy as np
    net = TFNetwork(extern_data=ExternData())
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
    with tf.variable_scope("src_nchw_feature_unspecified"):
      src_nchw_no_f = InternalLayer(name="src_nchw_feature_unspecified", network=net, out_type={"dim": 16,
                                                                                                "shape": (16, None, 16),
                                                                                                "batch_dim_axis": 0,
                                                                                                "time_dim_axis": 2,
                                                                                                "feature_dim_axis": NotSpecified,
                                                                                                "sparse": False
                                                                                                })
      src_nchw_no_f.output.placeholder = tf.placeholder(shape=(None, 16, None, 16), dtype=tf.float32)
      src_nchw_no_f.output.size_placeholder = {1: tf.placeholder(shape=(None,), dtype=tf.int32)}
    with tf.variable_scope("slice1"):
      slice1 = SliceLayer(
        name="slice1", network=net, axis="f", slice_step=2, sources=[src_nchw],
        output=SliceLayer.get_out_data_from_opts(name="slice1", axis="f", slice_step=2,
                                                 sources=[src_nchw]))
    with tf.variable_scope("slice2"):
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


def test_ReduceLayer_NCHW():
  with make_scope() as session:
    import numpy as np
    net = TFNetwork(extern_data=ExternData())
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
    with tf.variable_scope("reduce1"):
      reduce1 = ReduceLayer(
        name="reduce1", network=net, mode="max", axis="f", sources=[src_nchw],
        output=ReduceLayer.get_out_data_from_opts(name="reduce1", mode="max", axis="f",
                                                  sources=[src_nchw]))
    with tf.variable_scope("reduce2"):
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
    with tf.variable_scope("src_nchw"):
      src_nchw = InternalLayer(name="src_nchw", network=net, out_type={"dim": 16,
                                                                       "shape": (16, None),
                                                                       "batch_dim_axis": 0,
                                                                       "time_dim_axis": 2,
                                                                       "feature_dim_axis": 1,
                                                                       "sparse": False
                                                                       })
      src_nchw.output.placeholder = tf.placeholder(shape=(None, 16, None), dtype=tf.float32)
      src_nchw.output.size_placeholder = {1: tf.placeholder(shape=(None,), dtype=tf.int32)}

    with tf.variable_scope("activation"):
      activation = ActivationLayer(name="activation", activation="softmax", network=net, sources=[src_nchw])

    target_placeholder = tf.placeholder(shape=(None, None, 16), dtype=tf.float32)
    target_size_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)
    target_data = Data(name="target", shape=(None, 16), placeholder=target_placeholder,
                       size_placeholder={0: target_size_placeholder},
                       time_dim_axis=1, feature_dim_axis=2)

    with tf.variable_scope("loss"):
      loss = CrossEntropyLoss(base_network=net)
      loss.init(output=activation.output, output_with_activation=activation.output_before_activation,
                target=target_data, layer=activation)

    random_input = np.random.rand(10, 16, 32)
    loss_out, out_flat = session.run([loss.get_value(), loss.output_before_softmax_flat],
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
  with tf.Session() as session:
    print("Init params...")
    network.initialize_params(session=session)
    print("Testing reuse_params ...")
    feed = make_feed_dict(10)
    fwd_out, fwd_out_copy = session.run([network.layers["rec_fwd"].output.placeholder, network.layers["rec_fwd_copy"].output.placeholder], feed_dict=feed)
    numpy.testing.assert_array_equal(fwd_out, fwd_out_copy)


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
  with tf.Session() as session:
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


def test_param_variational_noise():
  from TFUtil import print_graph_output, find_ops_with_tensor_input
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
      assert len(ops) == 1 and "param_variational_noise" in ops[0].name


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
