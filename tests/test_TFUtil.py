
# start test like this:  nosetests-2.7  tests/test_TFUtil.py

from __future__ import print_function


import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import sys
sys.path += ["."]  # Python 3 hack
from TFUtil import *
from nose.tools import assert_equal, assert_is_instance, assert_is, assert_in
from numpy.testing.utils import assert_almost_equal
import unittest
import numpy.testing
import better_exchook
better_exchook.replace_traceback_format_tb()


session = tf.InteractiveSession()


def test_tf_version_tuple():
  print("TF version:", tf.__version__)
  print("TF version tuple:", tf_version_tuple())


def test_Data():
  data = Data(name="my_data", shape=(None, 13))
  assert_equal(data.name, "my_data")
  assert_equal(data.dim, 13)
  assert_equal(data.batch_dim_axis, 0)
  assert_equal(data.time_dim_axis, 1)
  assert_equal(data.feature_dim_axis, 2)
  assert_equal(data.batch_ndim, 3)
  assert_equal(data.batch_shape, (None, None, 13))
  assert_equal(data.dtype, "float32")
  assert_equal(data.sparse, False)


def test_Data_dim():
  data = Data(name="my_data", dim=13)
  assert_equal(data.name, "my_data")
  assert_equal(data.dim, 13)
  assert_equal(data.batch_dim_axis, 0)
  assert_equal(data.time_dim_axis, 1)
  assert_equal(data.feature_dim_axis, 2)
  assert_equal(data.batch_ndim, 3)
  assert_equal(data.batch_shape, (None, None, 13))
  assert_equal(data.dtype, "float32")
  assert_equal(data.sparse, False)


def test_Data_copy_time_major():
  data = Data(name="my_data", dim=13)
  assert_equal(data.batch_dim_axis, 0)
  assert_equal(data.time_dim_axis, 1)
  assert_equal(data.feature_dim_axis, 2)
  assert_equal(data.batch_ndim, 3)
  data2 = data.copy_as_time_major()
  assert_equal(data2.time_dim_axis, 0)
  assert_equal(data2.batch_dim_axis, 1)
  assert_equal(data2.feature_dim_axis, 2)
  assert_equal(data2.batch_ndim, 3)


def test_Data_copy_batch_major():
  data = Data(name="my_data", dim=13, time_dim_axis=0, batch_dim_axis=1)
  assert_equal(data.time_dim_axis, 0)
  assert_equal(data.batch_dim_axis, 1)
  assert_equal(data.feature_dim_axis, 2)
  assert_equal(data.batch_ndim, 3)
  data2 = data.copy_as_batch_major()
  assert_equal(data2.batch_dim_axis, 0)
  assert_equal(data2.time_dim_axis, 1)
  assert_equal(data2.feature_dim_axis, 2)
  assert_equal(data2.batch_ndim, 3)


def test_Data_spatial_batch_axes():
  d1 = Data(name='ff_out_prior_output', shape=(1, 9001), dtype='float32', batch_dim_axis=None)
  d2 = Data(name='ff_out_output', shape=(None, 9001), dtype='float32')
  spatial_axes1 = d1.get_spatial_batch_axes()
  spatial_axes2 = d2.get_spatial_batch_axes()
  assert_equal(len(spatial_axes1), len(spatial_axes2))
  spatial_axes1 = d1.get_spatial_axes()
  spatial_axes2 = d2.get_spatial_axes()
  assert_equal(len(spatial_axes1), len(d1.get_spatial_batch_axes()))
  assert_equal(spatial_axes1, spatial_axes2)


def test_Data_copy_compatible_to_time_major():
  d1 = Data(name='ff_out_output', shape=(None, 9001), dtype='float32', batch_dim_axis=1)
  d2 = Data(name='ff_out_prior_output', shape=(9001,), dtype='float32', batch_dim_axis=None, time_dim_axis=None)
  d2a = d2.copy_compatible_to(d1)
  assert d2a.shape == (1, 9001)
  assert d2a.batch_dim_axis == d1.batch_dim_axis
  assert d2a.time_dim_axis == d1.time_dim_axis
  assert d2a.feature_dim_axis == d1.feature_dim_axis


def test_Data_copy_compatible_to_batch_major():
  d1 = Data(name='ff_out_output', shape=(None, 9001), dtype='float32')
  d2 = Data(name='ff_out_prior_output', shape=(9001,), dtype='float32', batch_dim_axis=None, time_dim_axis=None)
  d2a = d2.copy_compatible_to(d1)
  assert d2a.shape == (1, 9001)
  assert d2a.batch_dim_axis == d1.batch_dim_axis
  assert d2a.time_dim_axis == d1.time_dim_axis
  assert d2a.feature_dim_axis == d1.feature_dim_axis


def test_get_initializer_zero():
  shape = (2, 3)
  initializer = get_initializer(0.0)
  v = initializer(shape)
  assert_almost_equal(session.run(v), numpy.zeros(shape))


def test_get_initializer_const_formula():
  shape = (2, 3)
  initializer = get_initializer("log(1.0 / 4.0)")
  v = initializer(shape)
  assert_almost_equal(session.run(v), numpy.zeros(shape) + numpy.log(1.0 / 4.0))


def test_get_initializer_zeros():
  shape = (2, 3)
  initializer = get_initializer("zeros")
  v = initializer(shape)
  assert_almost_equal(session.run(v), numpy.zeros(shape))


def test_get_initializer_constant():
  shape = (2, 3)
  initializer = get_initializer("constant")
  v = initializer(shape)
  assert_almost_equal(session.run(v), numpy.zeros(shape))


def test_get_initializer_xavier():
  shape = (2, 3)
  initializer = get_initializer("xavier")
  v = initializer(shape)
  assert_equal(session.run(v).shape, shape)  # returns some random matrix


def test_get_initializer_glorot_uniform():
  shape = (2, 3)
  initializer = get_initializer("glorot_uniform")
  v = initializer(shape)
  assert_equal(session.run(v).shape, shape)  # returns some random matrix


def test_get_initializer_glorot_normal_with_scale():
  shape = (2, 3)
  initializer = get_initializer('VarianceScaling(scale=6.0, mode="fan_avg", distribution="normal")')
  v = initializer(shape)
  assert_equal(session.run(v).shape, shape)  # returns some random matrix


def test_get_initializer_uniform():
  shape = (2, 3)
  initializer = get_initializer("RandomUniform(-0.01, 0.01)")
  v = initializer(shape)
  assert_equal(session.run(v).shape, shape)  # returns some random matrix


def test_get_initializer_gauss():
  shape = (2, 3)
  initializer = get_initializer("RandomNormal(0.0, 0.01)")
  v = initializer(shape)
  assert_equal(session.run(v).shape, shape)  # returns some random matrix


def test_close_event_writer_thread():
  import threading
  import tempfile
  from tensorflow.python.summary.writer.event_file_writer import EventFileWriter, _EventLoggerThread

  def count_event_logger_threads():
    return len([t for t in threading.enumerate() if isinstance(t, _EventLoggerThread)])

  tmp_dir = tempfile.mkdtemp()
  writer = tf.summary.FileWriter(tmp_dir)
  assert_equal(count_event_logger_threads(), 1)
  assert isinstance(writer.event_writer, EventFileWriter)
  assert isinstance(writer.event_writer._worker, _EventLoggerThread)
  writer.close()

  # https://github.com/tensorflow/tensorflow/issues/4820
  # The _EventLoggerThread is still running (at least in TF 1.1.0).
  if writer and writer.event_writer and writer.event_writer._worker.is_alive():
    stop_event_writer_thread(writer.event_writer)
    assert_equal(count_event_logger_threads(), 0)


def test_single_strided_slice():
  x = tf.expand_dims(tf.range(10), axis=0)
  assert_equal(list(tf.shape(x).eval()), [1, 10])
  assert_equal(list(single_strided_slice(x, axis=1, begin=3, end=6, step=2)[0].eval()), [3, 5])
  assert_equal(list(single_strided_slice(x, axis=1, begin=4)[0].eval()), list(range(4, 10)))
  assert_equal(list(single_strided_slice(x, axis=1, end=3)[0].eval()), [0, 1, 2])
  assert_equal(list(single_strided_slice(x, axis=tf.constant(1), end=3)[0].eval()), [0, 1, 2])
  assert_equal(list(single_strided_slice(x, axis=tf.constant(-1), end=3)[0].eval()), [0, 1, 2])
  x2 = tf.reshape(tf.range(9), (3, 3))
  assert_equal(list(x2[0].eval()), [0, 1, 2])
  assert_equal(list(tf.squeeze(single_strided_slice(x2, axis=tf.constant(0), end=1), axis=0).eval()), [0, 1, 2])


def test_slice_pad_zeros():
  x = tf.constant([1, 2, 3, 4])
  assert_equal(list(slice_pad_zeros(x, begin=1, end=3).eval()), [2, 3])
  assert_equal(list(slice_pad_zeros(x, begin=-2, end=2).eval()), [0, 0, 1, 2])
  assert_equal(list(slice_pad_zeros(x, begin=-2, end=6).eval()), [0, 0, 1, 2, 3, 4, 0, 0])
  assert_equal(list(slice_pad_zeros(x, begin=2, end=6).eval()), [3, 4, 0, 0])


def test_circular_pad():
  x = tf.reshape(tf.range(9), (3, 3))
  assert_equal(list(x[0].eval()), [0, 1, 2])
  x_ref = numpy.array(
    [[0, 1, 2],
     [3, 4, 5],
     [6, 7, 8]])
  numpy.testing.assert_equal(x.eval(), x_ref)
  y = circular_pad(x, paddings=1)
  y_ref = numpy.array(
    [[8, 6, 7, 8, 6],
     [2, 0, 1, 2, 0],
     [5, 3, 4, 5, 3],
     [8, 6, 7, 8, 6],
     [2, 0, 1, 2, 0]])
  numpy.testing.assert_equal(y.eval(), y_ref)

  x = tf.expand_dims(tf.reshape(tf.range(9), (3, 3)), axis=2)
  assert_equal(list(x[0, :, 0].eval()), [0, 1, 2])
  x_ref = numpy.array(
    [[[0], [1], [2]],
     [[3], [4], [5]],
     [[6], [7], [8]]])
  numpy.testing.assert_equal(x.eval(), x_ref)
  y = circular_pad(x, paddings=1, axes=(0, 1))
  y_ref = numpy.array(
    [[[8], [6], [7], [8], [6]],
     [[2], [0], [1], [2], [0]],
     [[5], [3], [4], [5], [3]],
     [[8], [6], [7], [8], [6]],
     [[2], [0], [1], [2], [0]]])
  numpy.testing.assert_equal(y.eval(), y_ref)


def test_reuse_name_scope_double():
  with reuse_name_scope("double"):
    assert_equal(tf.get_default_graph()._name_stack, "double")
    with reuse_name_scope("sub"):
      assert_equal(tf.get_default_graph()._name_stack, "double/sub")
      assert_equal(get_current_name_scope(), "double/sub")


def test_reuse_name_scope_mix1():
  with reuse_name_scope("mix1"):
    assert_equal(tf.get_default_graph()._name_stack, "mix1")
    with tf.name_scope("sub"):
      assert_equal(tf.get_default_graph()._name_stack, "mix1/sub")
      # The following is not true because get_current_name_scope is only var-scope:
      # assert_equal(get_current_name_scope(), "mix1/sub")


def test_reuse_name_scope_mix2():
  with tf.name_scope("mix2"):
    with reuse_name_scope("sub"):
      assert_equal(tf.get_default_graph()._name_stack, "mix2/sub")
      # The following is not true because get_current_name_scope is only var-scope:
      # assert_equal(get_current_name_scope(), "mix2/sub")


def test_reuse_name_scope_mix3():
  with reuse_name_scope("mix3"):
    with tf.variable_scope("sub"):
      assert_equal(get_current_name_scope(), "mix3/sub")


def test_reuse_name_scope_mix4():
  with tf.variable_scope("mix4"):
    with reuse_name_scope("sub"):
      assert_equal(get_current_name_scope(), "mix4/sub")


def test_reuse_name_scope_2():
  with reuse_name_scope("lstm2"):
    with reuse_name_scope("rec") as scope:
      assert_is_instance(scope, tf.VariableScope)
      assert_equal(scope.name, "lstm2/rec")
      assert_equal(get_current_name_scope(), "lstm2/rec")
      with tf.name_scope("sub"):
        assert_equal(get_current_name_scope(), "lstm2/rec/sub")


def test_reuse_name_scope():
  with reuse_name_scope("lstm0"):
    with tf.variable_scope("rec"):
      a = tf.get_variable("a", shape=(3, 4))
      assert_is_instance(a, tf.Variable)
      assert_equal(a.name, "lstm0/rec/a:0")

      b = tf.Variable(name="b", initial_value=tf.zeros((2,)))
      assert_equal(b.name, "lstm0/rec/b:0")

  with reuse_name_scope("lstm0"):
    with reuse_name_scope("rec"):
      c = tf.Variable(name="c", initial_value=tf.zeros((2,)))
      assert_equal(c.name, "lstm0/rec/c:0")

      c2 = tf.Variable(name="c", initial_value=tf.zeros((2,)))
      assert_equal(c2.name, "lstm0/rec/c_1:0")


def test_reuse_name_scope_root():
  with reuse_name_scope("", absolute=True):
    pass


def test_reuse_var_scope():
  with tf.variable_scope("v1"):
    assert_equal(get_current_var_scope_name(), "v1")
    assert_equal(get_current_name_scope(), "v1")
    with tf.variable_scope("v2") as scope:
      assert_equal(get_current_var_scope_name(), "v1/v2")
      assert_equal(get_current_name_scope(), "v1/v2")
      with tf.name_scope("v3"):
        assert_equal(get_current_name_scope(), "v1/v2/v3")
        assert_equal(get_current_var_scope_name(), "v1/v2")
        assert_equal(scope.name, "v1/v2")
        # Note: tf.variable_scope(scope) is broken here.
        with reuse_name_scope(scope):
          assert_equal(get_current_var_scope_name(), "v1/v2")
          assert_equal(get_current_name_scope(), "v1/v2")


def test_name_var_scope_mixing():
  with tf.variable_scope("mv1"):
    assert_equal(get_current_var_scope_name(), "mv1")
    assert_equal(get_current_name_scope(), "mv1")
    with tf.variable_scope("v2") as scope:
      assert_equal(get_current_var_scope_name(), "mv1/v2")
      assert_equal(get_current_name_scope(), "mv1/v2")
      with tf.name_scope("v3"):
        assert_equal(get_current_name_scope(), "mv1/v2/v3")
        assert_equal(get_current_var_scope_name(), "mv1/v2")
        assert_equal(scope.name, "mv1/v2")
        # Note: tf.variable_scope("v4") is broken here.
        with reuse_name_scope("v4"):
          assert_equal(get_current_var_scope_name(), "mv1/v2/v3/v4")
          assert_equal(get_current_name_scope(), "mv1/v2/v3/v4")
          with reuse_name_scope(scope):
            assert_equal(get_current_var_scope_name(), "mv1/v2")
            assert_equal(get_current_name_scope(), "mv1/v2")


def test_reuse_name_scope_of_tensor():
  with tf.name_scope("scope1") as scope1:
    x = tf.constant(42)
  with tf.name_scope("scope2") as scope2:
    assert_equal(get_current_name_scope() + "/", scope2)
    with reuse_name_scope_of_tensor(x):
      assert_equal(get_current_name_scope() + "/", scope1)


def test_reuse_name_scope_of_tensor_root():
  x = tf.constant(42)
  with tf.name_scope("scope2") as scope2:
    assert_equal(get_current_name_scope() + "/", scope2)
    with reuse_name_scope_of_tensor(x):
      assert_equal(get_current_name_scope(), "")


def test_loop_var_creation():
  # Related TF bugs:
  # https://github.com/tensorflow/tensorflow/issues/3114
  # https://github.com/tensorflow/tensorflow/issues/4478
  # https://github.com/tensorflow/tensorflow/issues/8604

  # tf.reset_default_graph()  # Strange, this does not work.
  i = tf.constant(0)

  def body(i):
    # None of these works, with error:
    # InvalidArgumentError: The node 'while/w/Assign' has inputs from different frames.
    # The input 'while/j' is in frame 'while/while/'. The input 'while/w' is in frame ''.
    # w = tf.Variable(tf.constant(1))
    # w = tf.Variable(tf.constant_initializer(value=1, dtype=tf.int32)(shape=()))
    # However, resetting the control dependencies will also reset the frame.
    with var_creation_scope():
      w = tf.Variable(tf.constant(1))
    return [i + w]

  loop = tf.while_loop(lambda i: tf.less(i, 5), body, [i])
  session.run(tf.global_variables_initializer())


def test_gather_nd_grad():
  # https://github.com/tensorflow/tensorflow/issues/9406
  # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/gather_nd_op.cc
  # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/scatter_nd_op.cc
  # Fixed in TF 1.1.0.
  assert_min_tf_version((1, 1), "tf.gather_nd")
  n_base_time = 5
  n_in = 7
  n_beam = 3
  n_batch = 1
  base = tf.ones((n_base_time, n_batch, n_in))  # (base_time,batch,n_in)
  idxs_exp = tf.constant(0, shape=(n_beam, n_batch, 2), name="idxs_exp")  # (beam,batch,2), where the 2 stands for (base_time,batch)
  # Thus K == 2. gather_nd out will be idxs_exp.shape[:2] + params.shape[2:] = (beam,batch,n_in).
  gathered = tf.gather_nd(base, idxs_exp)  # (beam,batch,n_in)
  gathered_shape, _ = session.run([tf.shape(gathered), gathered])
  assert_equal(list(gathered_shape), [n_beam, n_batch, n_in])

  base_grad = tf.gradients(gathered, base)
  assert base_grad is not None
  session.run(base_grad)


def test_scatter_nd():
  # https://github.com/tensorflow/tensorflow/issues/9406
  # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/scatter_nd_op.cc
  # Fixed in TF 1.1.0.
  assert_min_tf_version((1, 1), "tf.scatter_nd")
  n_base_time = 5
  n_in = 7
  n_beam = 3
  n_batch = 1
  ref_grad = tf.scatter_nd(
    indices=tf.zeros((n_beam, n_batch, 2), dtype=tf.int32),
    updates=tf.ones((n_beam, n_batch, n_in)),
    shape=(n_base_time, n_batch, n_in))
  session.run(ref_grad)


def test_dimshuffle():
  x = tf.zeros((2, 3, 5))
  assert_equal(list(session.run(tf.shape(x))), [2, 3, 5])
  assert_equal(list(session.run(tf.shape(dimshuffle(x, (1, 2, 0))))), [3, 5, 2])
  assert_equal(list(session.run(tf.shape(dimshuffle(x, ('x', 1, 2, 0))))), [1, 3, 5, 2])
  assert_equal(list(session.run(tf.shape(dimshuffle(x, ('x', 1, 'x', 2, 'x', 0, 'x'))))), [1, 3, 1, 5, 1, 2, 1])
  x = tf.zeros((2, 1, 3))
  assert_equal(list(session.run(tf.shape(dimshuffle(x, (2, 0))))), [3, 2])
  assert_equal(list(session.run(tf.shape(dimshuffle(x, (2, 'x', 'x', 0))))), [3, 1, 1, 2])


def test_expand_multiple_dims():
  x = tf.zeros((2, 3, 5))
  assert_equal(list(session.run(tf.shape(x))), [2, 3, 5])
  assert_equal(list(session.run(tf.shape(expand_multiple_dims(x, (1, 2))))), [2, 1, 1, 3, 5])
  assert_equal(list(session.run(tf.shape(expand_multiple_dims(x, (1, 4))))), [2, 1, 3, 5, 1])
  assert_equal(list(session.run(tf.shape(expand_multiple_dims(x, (1, 3, 5))))), [2, 1, 3, 1, 5, 1])


def test_move_axis():
  x = tf.zeros((2, 3, 5))
  assert_equal(list(session.run(tf.shape(x))), [2, 3, 5])
  assert_equal(list(session.run(tf.shape(move_axis(x, old_axis=0, new_axis=1)))), [3, 2, 5])
  assert_equal(list(session.run(tf.shape(move_axis(x, old_axis=0, new_axis=2)))), [3, 5, 2])
  assert_equal(list(session.run(tf.shape(move_axis(x, old_axis=2, new_axis=0)))), [5, 2, 3])
  assert_equal(list(session.run(tf.shape(move_axis(x, old_axis=2, new_axis=1)))), [2, 5, 3])


def test_constant_with_shape():
  x = session.run(constant_with_shape(3, [2, 3]))
  assert_equal(x.shape, (2, 3))
  assert_equal(x.dtype, numpy.int32)
  assert_equal(x.flatten().tolist(), [3] * 2 * 3)

  x = session.run(constant_with_shape(7.0, [2, 3]))
  assert_equal(x.shape, (2, 3))
  assert_equal(x.dtype, numpy.float32)
  assert_equal(x.flatten().tolist(), [7.0] * 2 * 3)

  x = session.run(constant_with_shape(False, [2, 3]))
  assert_equal(x.shape, (2, 3))
  assert_equal(x.dtype, numpy.bool_)
  assert_equal(x.flatten().tolist(), [False] * 2 * 3)

  x = session.run(constant_with_shape(True, [2, 3]))
  assert_equal(x.shape, (2, 3))
  assert_equal(x.dtype, numpy.bool_)
  assert_equal(x.flatten().tolist(), [True] * 2 * 3)


def naive_windowed_batch(source, window):
  assert source.ndim == 3  # (time,batch,dim). not sure how to handle other cases
  n_time = source.shape[0]
  n_batch = source.shape[1]
  n_dim = source.shape[2]
  w_right = window // 2
  w_left = window - w_right - 1
  dtype = source.dtype
  pad_left = numpy.zeros((w_left, n_batch, n_dim), dtype=dtype)
  pad_right = numpy.zeros((w_right, n_batch, n_dim), dtype=dtype)
  padded = numpy.concatenate([pad_left, source, pad_right], axis=0)
  final = numpy.zeros((n_time, window, n_batch, n_dim), dtype=dtype)
  for t in range(n_time):
    for w in range(window):
      final[t, w] = padded[t + w]
  return final


def test_windowed_nd_small():
  n_time = 2
  n_batch = 2
  n_dim = 2
  window = 3
  source = numpy.arange(1, n_time*n_batch*n_dim + 1).reshape(n_time, n_batch, n_dim)
  print("source:")
  print(source)
  naive = naive_windowed_batch(source, window=window)
  real = windowed_nd(source, window_size=window, time_axis=0, new_window_axis=1).eval()
  print("naive:")
  print(naive)
  print("real:")
  print(real)
  numpy.testing.assert_almost_equal(naive, real)


def test_windowed_nd_big():
  n_time = 11
  n_batch = 5
  n_dim = 7
  window = 3
  numpy.random.seed(123)
  source = numpy.random.random((n_time, n_batch, n_dim)).astype("float32")
  naive = naive_windowed_batch(source, window=window)
  real = windowed_nd(source, window_size=window, time_axis=0, new_window_axis=1).eval()
  numpy.testing.assert_almost_equal(naive, real)


def naive_slice_nd(x, start, size):
  slices_shape = [x.shape[0], size] + list(x.shape)[2:]
  ys = numpy.zeros(shape=slices_shape)
  for i in range(len(start)):
    time_len = len(x[i])
    end = start[i] + size
    if time_len < end:
      end = time_len
    y = x[i][start[i]:end]

    # padding
    if time_len < start[i] + size:
       y = numpy.pad(y, [[0,start[i]+size-time_len], [0,0]], mode='constant')
    ys[i] = y
  return ys


def test_slice_nd_small():
  n_batch = 3
  n_time = 4
  n_dim = 2
  size = 2
  start = numpy.array([0,2,3]).astype("int32")
  source = numpy.arange(1, n_batch*n_time*n_dim + 1, dtype=numpy.float32).reshape(n_batch, n_time, n_dim).astype("float32")
  source_tf = tf.constant(source)
  naive = naive_slice_nd(source, start, size)
  real = slice_nd(source_tf, start=start, size=size).eval()
  print("source:")
  print(source)
  print("naive:")
  print(naive)
  print("real:")
  print(real)
  numpy.testing.assert_almost_equal(naive, real)


def test_slice_nd_big():
  n_batch = 8
  n_time = 12
  n_dim = 4
  size = 4
  numpy.random.seed(123)
  start = numpy.random.randint(low=0, high=12, size=(n_batch,), dtype="int32")
  source = numpy.random.random((n_batch, n_time, n_dim)).astype("float32")
  source_tf = tf.constant(source)
  naive = naive_slice_nd(source, start, size)
  real = slice_nd(source_tf, start=start, size=size).eval()
  print("source:")
  print(source)
  print("naive:")
  print(naive)
  print("real:")
  print(real)
  numpy.testing.assert_almost_equal(naive, real)


def test_CustomGradient_register_new_graph_generic_loss_and_error_signal():
  def check():
    with tf.Graph().as_default() as graph:
      with tf.Session(graph=graph) as session:
        custom_gradient.register_generic_loss_and_error_signal()
        x = tf.constant(2.)
        session.run(x)  # do some early call, before `generic_loss_and_error_signal` below
        y = custom_gradient.generic_loss_and_error_signal(loss=1., x=x, grad_x=3.)
        assert y.graph is graph
        grad_y, = tf.gradients(y, x)
        assert_equal(session.run([y, x, grad_y]), [1., 2., 3.])
  check()
  check()
  check()


def test_CustomGradient_generic_loss_and_error_signal_post_func():
  with tf.Graph().as_default() as graph:
    with tf.Session(graph=graph) as session:
      custom_gradient.register_generic_loss_and_error_signal()
      x = tf.constant(5.)
      y = custom_gradient.generic_loss_and_error_signal(loss=2., x=x, grad_x=3.)
      z = 2. * y
      assert y.graph is graph
      grad_z, = tf.gradients(z, x)
      assert_equal(session.run([z, x, grad_z]), [4., 5., 6.])


def test_global_tensor():
  class C:
    i = 0
  def f():
    C.i += 1
    return tf.constant(42, name="hello")
  x = global_tensor(f, name="hello")
  x2 = global_tensor(f, name="hello")
  x3 = global_tensor(f, name="hello")
  assert_equal(C.i, 1)
  assert_is(x, x2)
  assert_is(x, x3)
  assert_equal(x.eval(), 42)


def test_encode_raw_direct():
  raw = tf.decode_raw(tf.constant("ABC"), tf.uint8)
  assert_equal(list(raw.eval()), [65, 66, 67])


def test_encode_raw_simple():
  raw = tf.decode_raw(tf.constant("hello"), tf.uint8)
  back = encode_raw(raw)
  assert_equal(back.eval(), b"hello")


def test_encode_raw_seq_lens():
  strs = ["hello", "world", "a    "]  # all same lengths for tf.decode_raw
  strs_stripped = [s.strip() for s in strs]
  raw = tf.decode_raw(tf.constant(strs), tf.uint8)
  seq_lens = tf.constant([len(s) for s in strs_stripped])
  back = encode_raw(raw, seq_lens=seq_lens)
  assert_equal(list(back.eval()), [s.encode("utf8") for s in strs_stripped])


@unittest.skip("broken? https://github.com/tensorflow/tensorflow/issues/11240")
def test_sequential_control_dependencies():
  v = tf.Variable(initial_value=2, trainable=False, name="test_sequential_control_dependencies")
  with sequential_control_dependencies([
    lambda: v.initializer,
    lambda: tf.assign(v, 3),
    lambda: tf.assign(v, v.read_value() + 5)
  ]):
    x = v.read_value()
  assert_equal(x.eval(), 3 + 5)


@unittest.skip("broken? https://github.com/tensorflow/tensorflow/issues/11240")
def test_var_init():
  # upstream comment: use resource variables instead
  v = tf.Variable(initial_value=2, trainable=False, name="test_var_init")
  with tf.control_dependencies([v.initializer]):
    x = v.read_value()
  assert_equal(x.eval(), 2)


def test_resource_var_init():
  # https://github.com/tensorflow/tensorflow/issues/11240
  # Will use :class:`ResourceVariable`.
  v = tf.get_variable(
    initializer=tf.constant_initializer(2), shape=(),
    trainable=False, name="test_resource_var_init", use_resource=True)
  with tf.control_dependencies([v.initializer]):
    x = v.read_value()
  assert_equal(x.eval(), 2)


@unittest.skip("broken? see also test_var_init")  # TODO...
def test_true_once():
  x = true_once()
  assert_equal(x.eval(), True)
  assert_equal(x.eval(), False)
  assert_equal(x.eval(), False)
  assert_equal(x.eval(), False)


@unittest.skip("broken?")  # TODO...
def test_raise_OutOfRangeError():
  for j in range(2):
    x = raise_OutOfRangeError()
    for i in range(3):
      try:
        session.run(x)
        assert False, "should have raised OutOfRangeError"
      except tf.errors.OutOfRangeError:
        pass


def test_enforce_copy():
  v = tf.Variable(initial_value=2, trainable=False, name="test_copy")
  # with tf.control_dependencies([v.initializer]) does not work?
  session.run(v.initializer)
  a = tf.identity(v.read_value())
  b = enforce_copy(v.read_value())
  with tf.control_dependencies([a, b]):
    with tf.control_dependencies([tf.assign(v, 3)]):
      # `a` is a ref to v, thus also 3 now.
      # `b` is a copy, thus 2, as initially.
      x = tf.add(0, [a, b, v.read_value()])
  assert_equal(list(x.eval()), [3, 2, 3])


def test_Lock():
  lock = Lock()
  session.run(lock.init())
  v = tf.Variable(initial_value=0, trainable=False, name="test_Lock")
  session.run(v.initializer)
  with tf.control_dependencies([lock.lock()]):
    with tf.control_dependencies([v.assign_add(1)]):
      x = enforce_copy(v)
      with tf.control_dependencies([x, lock.unlock()]):
        x = tf.identity(x)
  # Just checking lock + unlock, not really the behavior.
  for i in range(5):
    assert_equal(x.eval(), i + 1)
    assert_equal(v.eval(), i + 1)


def test_Condition():
  cond = Condition()
  v = tf.Variable(initial_value=0, trainable=False, name="test_Condition")
  session.run([cond.init(), v.initializer])
  with sequential_control_dependencies([
    lambda: cond.lock.lock(),
    lambda: v.assign_add(2),
    lambda: cond.signal(),
    lambda: cond.lock.unlock()
  ]):
    s = tf.no_op()
  session.run(cond.lock.lock())
  from threading import Thread
  t = Thread(target=lambda: session.run(s))
  t.start()
  session.run(cond.wait())
  assert_equal(v.eval(), 2)
  t.join()
  session.run(cond.lock.unlock())


@unittest.skip("needs tensor_array.h, see https://github.com/tensorflow/tensorflow/issues/10527")
def test_GlobalTensorArray():
  GlobalTensorArrayOpMaker().get_op()


def test_TFArrayContainer():
  # Bug #10950 is fixed upstream, should be in TF 1.2.2.
  # https://stackoverflow.com/questions/44455722/create-my-own-resource-types-tf-resource
  # https://github.com/tensorflow/tensorflow/issues/1419
  ta = TFArrayContainer(dtype=tf.int32)
  print(ta._mod)
  print(ta._mod.array_container_create.__doc__)
  assert_equal(ta.get_size().eval(), 0)
  session.run(ta.set_size(3))
  assert_equal(ta.get_size().eval(), 3)
  session.run(ta.set(1, [1, 2, 3]))
  assert_equal(list(ta.get(1).eval()), [1, 2, 3])


@unittest.skip("does not work")
def test_TensorArray():
  # see https://stackoverflow.com/questions/44418036/
  # Reason is that the TensorArray uses a per-run ("per-step") resource manager,
  # thus it will not remember anything across session.run() calls.
  # This is by design.
  # Our :class:`GlobalTensorArrayOpMaker` could fix this.
  ta = tf.TensorArray(tf.int32, size=3)
  index = tf.placeholder(tf.int32)
  value = tf.placeholder(tf.int32)
  flow = tf.placeholder(tf.float32)
  ta_new = tf.TensorArray(dtype=ta.dtype, handle=ta.handle, flow=flow)
  write = ta_new.write(index, value).flow
  read = ta_new.read(index)
  f = 0
  f = session.run(write, feed_dict={index: 0, value: 1, flow: f})
  f = session.run(write, feed_dict={index: 1, value: 2, flow: f})
  assert_equal(session.run(read, feed_dict={index: 0, flow: f}), 1)
  assert_equal(session.run(read, feed_dict={index: 1, flow: f}), 2)


@unittest.skip("does not work")
def test_ExplicitRandomShuffleQueue():
  # see test_TensorArray, which is internally used by ExplicitRandomShuffleQueue
  queue = ExplicitRandomShuffleQueue(capacity=3, min_after_dequeue=2, dtypes=[tf.int32])
  placeholder = tf.placeholder(tf.int32, shape=())
  session.run(queue.init())
  enqueue = queue.enqueue(placeholder)
  dequeue = queue.dequeue()
  size = queue.size()
  session.run(enqueue, feed_dict={placeholder: 1})
  session.run(enqueue, feed_dict={placeholder: 2})
  session.run(enqueue, feed_dict={placeholder: 3})
  pool = {1, 2, 3}
  for i in range(3):
    d = session.run(dequeue)
    assert_in(d, pool)
    pool.remove(d)
    session.run(enqueue, feed_dict={placeholder: i + 4})
    pool.add(i + 4)
    assert_equal(session.run(size), len(pool))
  session.run(queue.min_after_dequeue_assign(0))
  while pool:
    d = session.run(dequeue)
    assert_in(d, pool)
    pool.remove(d)
  assert_equal(session.run(size), 0)
  session.run(enqueue, feed_dict={placeholder: 17})
  assert_equal(session.run(dequeue), 17)


def test_tfconv1d_evensize():
  filters = tf.constant([[[2.0]], [[3.0]]])  # [filter_width, in_channels, out_channels]
  assert isinstance(filters, tf.Tensor)
  assert_equal(filters.get_shape().as_list(), [2, 1, 1])
  value = tf.constant([[[5.0], [7.0]]])  # (batch, time, dim)
  assert isinstance(value, tf.Tensor)
  assert_equal(value.get_shape().as_list(), [1, 2, 1])
  res = tf.nn.conv1d(value, filters=filters, stride=1, padding="SAME", data_format="NHWC")
  resv = res.eval()
  assert isinstance(resv, numpy.ndarray)
  assert_equal(resv.shape, (1, 2, 1))  # (batch, time, dim)
  # Tests that the kernel-size of 2 is applied on current-frame + right-frame.
  # Note that in the Dataset with context_window = 2, it will do the corresponding thing,
  # i.e. adds one right-frame and no left-frame, such that if you use padding="VALID",
  # it will match the right frames.
  assert_almost_equal(resv, [[[2*5.0+3*7.0], [2*7.0]]])


def test_tf_tile():
  batch_size = 3
  beam_size = 5
  v = tf.constant([1, 2, 3])  # (batch,)
  v.set_shape((batch_size,))
  v2 = tf.tile(v, [beam_size])  # (beam*batch,)
  v2.set_shape((beam_size * batch_size,))
  print(v2.eval())
  assert_equal(list(v2.eval()), [1, 2, 3] * 5)
  v3 = tf.reshape(v2, [beam_size, batch_size])  # (beam,batch)
  r = v3.eval()
  print(r)
  assert isinstance(r, numpy.ndarray)
  for beam in range(beam_size):
    assert_equal(list(r[beam]), [1, 2, 3])


def test_tile_transposed():
  batch_size = 3
  beam_size = 5
  v = tf.constant([1, 2, 3])  # (batch,)
  v.set_shape((batch_size,))
  v2 = tile_transposed(v, axis=0, multiples=beam_size)  # (batch*beam,)
  v2.set_shape((batch_size * beam_size,))
  print(v2.eval())
  assert_equal(list(v2.eval()), [1] * 5 + [2] * 5 + [3] * 5)
  v3 = tf.reshape(v2, [batch_size, beam_size])  # (batch,beam)
  r = v3.eval()
  print(r)
  assert isinstance(r, numpy.ndarray)
  for beam in range(beam_size):
    assert_equal(list(r[:, beam]), [1, 2, 3])


def test_expand_dims_unbroadcast_instead_of_tf_tile():
  batch_size = 3
  beam_size = 5
  v = tf.constant([1, 2, 3])  # (batch,)
  v.set_shape((batch_size,))
  v2 = expand_dims_unbroadcast(v, axis=1, dim=beam_size)  # (batch,beam)
  v2.set_shape((batch_size, beam_size))
  r = v2.eval()
  print(r)
  assert isinstance(r, numpy.ndarray)
  for beam in range(beam_size):
    assert_equal(list(r[:, beam]), [1, 2, 3])


def test_where_nan():
  # via: https://stackoverflow.com/a/42497444/133374
  # @ops.RegisterGradient("Select")
  # def _SelectGrad(op, grad):
  #   c = op.inputs[0]
  #   x = op.inputs[1]
  #   zeros = array_ops.zeros_like(x)
  #   return (None, array_ops.where(c, grad, zeros),
  #           array_ops.where(c, zeros, grad))
  # SelectOp, https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/cwise_op_select.cc
  # We later check for nan. assert_equal does not work as-is because (nan == nan) is False.
  # Thus, we resort to this check:
  assert_equal(str(float("nan")), "nan")

  where_0_nan = tf.where(True, 0., float("nan"))
  print("where_0_nan:", where_0_nan.eval())
  assert_equal(where_0_nan.eval(), 0.0)

  x = tf.constant(0.)
  x_equal_0 = tf.equal(x, 0.)
  f = tf.where(x_equal_0, 0., 1. / x)
  grad_x = tf.gradients(f, x)[0]
  print("grad_x:", grad_x.eval())  # nan? or 0?
  # This is expected when you look at the resulting computation graph for the gradient.
  # You will have grad(1./x, x) * 0.0 in the graph in the back-propagation of the gradient, which is nan.
  assert_equal(str(grad_x.eval()), "nan")

  safe_x = tf.where(x_equal_0, 2., x)
  grad_safe_x = tf.where(x_equal_0, 0., 1. / safe_x)
  print("grad_safe_x:", grad_safe_x.eval())  # nan? ln(2)? 0?
  # This works, because at no time, there is nan in the back-propagation.
  assert_equal(grad_safe_x.eval(), 0.0)

  f = tf.cond(x_equal_0, lambda: 0., lambda: 1. / x)
  grad_cond_x = tf.gradients(f, x)[0]
  print("grad_cond_x:", grad_cond_x.eval())  # nan? or 0?
  # This is different than tf.where because really only one branch will go into the gradient.
  assert_equal(grad_cond_x.eval(), 0.0)


def test_variable_summaries():
  v = tf.Variable(initial_value=[[1.0, 2.0], [-4.0, -1.0]], name="test_variable_summaries")
  variable_summaries(v)
  variable_summaries(tf.square(v))
  session.run(v.initializer)
  session.run(tf.summary.merge_all())
  assert_almost_equal(session.run(variable_scalar_summaries_dict(v)["test_variable_summaries_mean"]), -0.5)


def test_VariableAssigner():
  v = tf.Variable(initial_value=1.)
  session.run(v.initializer)
  assert_equal(session.run(v), 1.)
  assigner = VariableAssigner(v)
  assigner.assign(value=2., session=session)
  assert_equal(session.run(v), 2.)


def test_VariableAssigner_ResourceVariable():
  v = tf.get_variable(
    initializer=tf.constant_initializer(1.), shape=(),
    name="test_VariableAssigner_ResourceVariable", use_resource=True)
  session.run(v.initializer)
  assert_equal(session.run(v), 1.)
  assigner = VariableAssigner(v)
  assigner.assign(value=2., session=session)
  assert_equal(session.run(v), 2.)


def test_map_labels():
  x = tf.constant([0, 1, 2, 3, 2, 1, 0])
  label_map = {0: 1, 1: 2, 2: 3, 3: 0}
  y = map_labels(x, label_map=label_map)
  assert_equal(session.run(y).tolist(), [1, 2, 3, 0, 3, 2, 1])


def test_map_labels_SparseTensor():
  x = tf.SparseTensor(
    indices=tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=tf.int64, name="x_indices"),
    values=tf.constant([0, 1, 2, 3], name="x_values"),
    dense_shape=tf.constant([3, 3], dtype=tf.int64, name="x_dense_shape"))
  label_map = {0: 1, 1: 2, 2: 3, 3: 0}
  y = map_labels(x, label_map=label_map)
  assert isinstance(y, tf.SparseTensor)
  y_eval = session.run(y)
  assert isinstance(y_eval, tf.SparseTensorValue)
  assert_equal(y_eval.values.tolist(), [1, 2, 3, 0])


def test_sparse_labels():
  x = tf.constant([[0, 1, 2, 3], [4, 5, 0, 0]], name="x")
  seq_lens = tf.constant([4, 2], name="seq_lens")
  y = sparse_labels(x, seq_lens=seq_lens)
  y_eval = session.run(y)
  assert isinstance(y_eval, tf.SparseTensorValue)
  assert isinstance(y_eval.indices, numpy.ndarray)
  assert isinstance(y_eval.values, numpy.ndarray)
  assert isinstance(y_eval.dense_shape, numpy.ndarray)
  assert_equal(y_eval.indices.tolist(), [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1]])
  assert_equal(y_eval.values.tolist(), [0, 1, 2, 3, 4, 5])
  assert_equal(y_eval.dense_shape.tolist(), [2, 4])


def test_remove_labels():
  x = tf.SparseTensor(
    indices=tf.constant([[0, 0], [0, 1], [0, 2], [1, 0]], dtype=tf.int64, name="x_indices"),
    values=tf.constant([0, 1, 2, 3], name="x_values"),
    dense_shape=tf.constant([3, 3], dtype=tf.int64, name="x_dense_shape"))
  labels = {1}
  y = remove_labels(x, labels=labels)
  assert isinstance(y, tf.SparseTensor)
  y_eval = session.run(y)
  assert isinstance(y_eval, tf.SparseTensorValue)
  assert isinstance(y_eval.indices, numpy.ndarray)
  assert isinstance(y_eval.values, numpy.ndarray)
  assert isinstance(y_eval.dense_shape, numpy.ndarray)
  assert_equal(y_eval.indices.tolist(), [[0, 0], [0, 1], [1, 0]])
  assert_equal(y_eval.values.tolist(), [0, 2, 3])
  assert_equal(y_eval.dense_shape.tolist(), [3, 2])


def test_supported_devices_for_op():
  op_name = "MatMul"
  devs = supported_devices_for_op(op_name)
  print("Supported devs for op %r: %r" % (op_name, devs))
  assert "CPU" in devs


def test_bleu_score():
  hyp = [1, 2, 3]
  truth = [2, 3]
  from Util import compute_bleu
  res = compute_bleu([truth], [hyp])
  print("res:", res)
  tf_res = session.run(bleu_score(
    hypothesis=[hyp], hyp_seq_lens=[len(hyp)],
    truth=[truth], truth_seq_lens=[len(truth)]
  ))
  print("TF res:", tf_res)
  assert isinstance(tf_res, numpy.ndarray)
  assert tf_res.shape == (1,)
  assert_almost_equal(tf_res, [res])
  assert_almost_equal(tf_res, [0.6389431])


def test_bleu_score_empty():
  hyp = []
  truth = [2, 3]
  from Util import compute_bleu
  res = compute_bleu([truth], [hyp])
  print("res:", res)
  tf_res = session.run(bleu_score(
    hypothesis=[hyp], hyp_seq_lens=[len(hyp)],
    truth=[truth], truth_seq_lens=[len(truth)]
  ))
  print("TF res:", tf_res)
  assert isinstance(tf_res, numpy.ndarray)
  assert tf_res.shape == (1,)
  assert_almost_equal(tf_res, [res])
  assert_almost_equal(tf_res, [0.0])


def test_clip_by_value_with_identity_grad():
  err_y = 42.0
  limit = 1.0
  limits = -limit, limit
  with tf.name_scope("test_safe_log_and_grad"):
    x_t = tf.placeholder(tf.float32, shape=(), name="x")
    y_t = clip_by_value_with_identity_grad(x_t, *limits)
    err_x_t, = tf.gradients(ys=y_t, xs=x_t, grad_ys=tf.constant(err_y))
    err2_x_t, = tf.gradients(ys=tf.clip_by_value(x_t, *limits), xs=x_t, grad_ys=tf.constant(err_y))

  for x in [0.0, -0.5, 0.5, -1.0, 1.0, -2.0, 2.0]:
    x = numpy.array(x, dtype="float32")
    y, err_x, err2_x = session.run([y_t, err_x_t, err2_x_t], feed_dict={x_t: x})
    print("x:", x, "y:", y, "err_x:", err_x, "err2_x:", err2_x)
    assert_equal(err_x, err_y)
    assert -limit <= y <= limit
    if abs(x) > limit:
      assert_equal(err2_x, 0.0)
    if abs(x) < limit:
      assert_equal(err2_x, err_y)


def test_safe_log_and_grad():
  with tf.name_scope("test_safe_log_and_grad"):
    x_t = tf.placeholder(tf.float32, shape=(), name="x")
    y_t = safe_log(x_t)
    err_x_t, = tf.gradients(ys=y_t, xs=x_t)
    check_numerics_op = add_check_numerics_ops([y_t, err_x_t])
    # For comparison:
    y2_t = tf.log(x_t)
    err2_x_t, = tf.gradients(ys=y2_t, xs=x_t)

  for x in [0.0, 100, 1e30, 1e-30]:
    x = numpy.array(x, dtype="float32")
    print("x:", x)
    assert numpy.isfinite(x).all()
    y, err_x = session.run([y_t, err_x_t], feed_dict={x_t: x})
    print("y:", y, "err_x:", err_x)
    y2, err2_x = session.run([y2_t, err2_x_t], feed_dict={x_t: x})
    print("y2:", y2, "err2_x:", err2_x)
    if not numpy.isfinite(y).all() or not numpy.isfinite(err_x).all():
      print("Warning, some nan or inf!")
      session.run(check_numerics_op, feed_dict={x_t: x})
    assert numpy.isfinite(y).all() and numpy.isfinite(err_x).all()
    assert err_x != 0.0  # there should be some gradient


def test_safe_exp_and_grad():
  with tf.name_scope("test_safe_log_and_grad"):
    x_t = tf.placeholder(tf.float32, shape=(), name="x")
    y_t = safe_exp(x_t)
    err_x_t, = tf.gradients(ys=y_t, xs=x_t)
    check_numerics_op = add_check_numerics_ops([y_t, err_x_t])
    # For comparison:
    y2_t = tf.exp(x_t)
    err2_x_t, = tf.gradients(ys=y2_t, xs=x_t)

  for x in [0.0, 100, 1e30, 1e-30, -1e30, -1e-30]:
    x = numpy.array(x, dtype="float32")
    print("x:", x)
    assert numpy.isfinite(x).all()
    y, err_x = session.run([y_t, err_x_t], feed_dict={x_t: x})
    print("y:", y, "err_x:", err_x)
    y2, err2_x = session.run([y2_t, err2_x_t], feed_dict={x_t: x})
    print("y2:", y2, "err2_x:", err2_x)
    if not numpy.isfinite(y).all() or not numpy.isfinite(err_x).all():
      print("Warning, some nan or inf!")
      session.run(check_numerics_op, feed_dict={x_t: x})
    assert numpy.isfinite(y).all() and numpy.isfinite(err_x).all()
    assert err_x != 0.0  # there should be some gradient


def test_lin_exp_normed_limits_not_nan():
  with tf.name_scope("test_lin_exp_normed_limits_not_nan"):
    x_t = tf.placeholder(tf.float32, shape=(None,), name="x")
    y_t = lin_exp_normed(x_t)
    # Also see :class:`CrossEntropyLoss`. here score instead of loss.
    score_t = safe_log(y_t[..., -1])
    err_x_t, = tf.gradients(ys=score_t, xs=x_t)
    check_numerics_op = add_check_numerics_ops([score_t, y_t, err_x_t])

  for x in [(0, 0), (1, -1), (100, 100), (100, -100), (1e20, 1e-20), (1e30, -1e30), (1e30, 1e30), (-1e30, -1e30)]:
    x = numpy.array(x, dtype="float32")
    print("x:", x)
    assert numpy.isfinite(x).all()
    score, y, err_x = session.run([score_t, y_t, err_x_t], feed_dict={x_t: x})
    print("score:", score, "y:", y, "err_x:", err_x)
    if not numpy.isfinite(y).all() or not numpy.isfinite(err_x).all():
      print("Warning, some nan or inf!")
      session.run(check_numerics_op, feed_dict={x_t: x})
    assert numpy.isfinite(y).all() and numpy.isfinite(err_x).all()
    # We constructed the examples in such a way that there should always be a gradient.
    assert any(err_x != 0.0)


def test_check_base_op_type_and_replace_softmax():
  with tf.name_scope("test_check_base_op_type_and_replace_softmax"):
    z = tf.constant([1.0, 2.0])
    x = tf.nn.softmax(z)
    y = tf.log(x)
    print("x:", x, list(x.op.inputs), "y:", y)
    y2 = check_base_op_type_and_replace(x, "Softmax", "LogSoftmax")
    print("y2:", y2)
    assert y2 is not None
    vy1, vy2 = session.run([y, y2])
    print("eval:", vy1, vy2)
    assert_almost_equal(vy1, vy2)


def test_check_base_op_type_and_replace_sigmoid():
  with tf.name_scope("test_check_base_op_type_and_replace_sigmoid"):
    z = tf.constant([1.0, 2.0])
    x = tf.sigmoid(z)
    y = tf.log(x)
    print("x:", x, list(x.op.inputs), "y:", y)
    y2 = check_base_op_type_and_replace(x, "Sigmoid", "LogSigmoid")
    print("y2:", y2)
    assert y2 is not None
    vy1, vy2 = session.run([y, y2])
    print("eval:", vy1, vy2)
    assert_almost_equal(vy1, vy2)


def test_string_merge():
  strings = [
    ["sub@@", "word", "test"],
    ["hel@@", "lo", "wo@@", "r@@", "ld"],
    ["foo"]]
  seq_lens = [len(seq) for seq in strings]
  max_len = max(seq_lens)
  strings = [seq + [""] * (max_len - len(seq)) for seq in strings]

  tf_strings = tf.placeholder(tf.string, [None, None])
  tf_seq_lens = tf.placeholder(tf.int32, [None])
  tf_res = string_merge(tf_strings, tf_seq_lens)
  res = session.run(tf_res, feed_dict={tf_strings: strings, tf_seq_lens: seq_lens})
  print(res)
  assert isinstance(res, numpy.ndarray)
  assert res.shape == (len(seq_lens),)
  res = res.tolist()
  print(res)
  res = [s.decode("utf8") for s in res]
  print(res)
  assert_equal(res, ["sub@@ word test", "hel@@ lo wo@@ r@@ ld", "foo"])


def test_string_replace():
  strings = ["sub@@ word test", "hel@@ lo wo@@ r@@ ld", "foo"]
  tf_strings = tf.placeholder(tf.string, [None])
  tf_res = string_replace(tf_strings, old="@@ ", new="")
  res = session.run(tf_res, feed_dict={tf_strings: strings})
  print(res)
  assert isinstance(res, numpy.ndarray)
  assert res.shape == (len(strings),)
  res = res.tolist()
  print(res)
  res = [s.decode("utf8") for s in res]
  print(res)
  assert_equal(res, ["subword test", "hello world", "foo"])


def test_words_split_get_sparse_tensor_length():
  strings = ["subword test", "a b c d", "hello world", "foo"]
  word_lens = [len(s.split(" ")) for s in strings]
  tf_strings = tf.placeholder(tf.string, [None])
  tf_words = words_split(tf_strings)
  tf_dense_words = tf.sparse_to_dense(
    tf_words.indices, tf_words.dense_shape, tf_words.values, default_value="")
  tf_num_words = get_sparse_tensor_length(tf_words)
  words, dense_words, num_words = session.run(
    [tf_words, tf_dense_words, tf_num_words], feed_dict={tf_strings: strings})
  print(words)
  print(dense_words)
  print(num_words)
  assert isinstance(words, tf.SparseTensorValue)
  assert isinstance(dense_words, numpy.ndarray)
  assert isinstance(num_words, numpy.ndarray)
  assert dense_words.shape == (len(word_lens), max(word_lens))
  assert num_words.shape == (len(strings),)
  dense_words = dense_words.tolist()
  print(dense_words)
  assert_equal(dense_words, [
    [b"subword", b"test", b"", b""], [b"a", b"b", b"c", b"d"],
    [b"hello", b"world", b"", b""], [b"foo", b"", b"", b""]])
  assert_equal(num_words.tolist(), word_lens)


def test_string_words_calc_wer():
  hyps = ["hello world", "a b c", "how are you", "good"]
  refs = ["hello nice world", "a x c d", "how are we", "good"]
  tf_hyps = tf.placeholder(tf.string, [None])
  tf_refs = tf.placeholder(tf.string, [None])
  tf_wer, tf_ref_num_words = string_words_calc_wer(hyps=tf_hyps, refs=tf_refs)
  wer, ref_num_words = session.run([tf_wer, tf_ref_num_words], {tf_hyps: hyps, tf_refs: refs})
  print(wer, ref_num_words)
  assert isinstance(wer, numpy.ndarray)
  assert isinstance(ref_num_words, numpy.ndarray)
  assert_equal(wer.tolist(), [1, 2, 1, 0])
  assert_equal(ref_num_words.tolist(), [3, 4, 3, 1])


def test_kenlm():
  import TFKenLM
  input_strings = ["beyond immediate concerns </s>"]
  test_lm_file = TFKenLM.kenlm_dir + "/lm/test.arpa"
  assert os.path.exists(test_lm_file)
  lm_tf = TFKenLM.ken_lm_load(filename=test_lm_file)
  input_strings_tf = tf.placeholder(tf.string, [None])
  output_scores_tf = TFKenLM.ken_lm_abs_score_strings(handle=lm_tf, strings=input_strings_tf)
  with tf.Session() as session:
    output_scores = session.run(output_scores_tf, feed_dict={input_strings_tf: input_strings})
  print("input strings:", input_strings)
  print("output scores:", output_scores)
  assert isinstance(output_scores, numpy.ndarray)
  assert_almost_equal(output_scores, [-9.251298])  # +log space, not +log10
  print("Score is as expected.")


def test_kenlm_bpe():
  import TFKenLM
  input_strings = [
    "beyond immediate concerns </s>",
    "be@@ yond imm@@ edi@@ ate conc@@ erns </s>",
    "be@@ yond imm@@",
    "be@@ yond <unk>"
    ]
  test_lm_file = TFKenLM.kenlm_dir + "/lm/test.arpa"
  assert os.path.exists(test_lm_file)
  lm_tf = TFKenLM.ken_lm_load(filename=test_lm_file)
  input_strings_tf = tf.placeholder(tf.string, [None])
  output_scores_tf = TFKenLM.ken_lm_abs_score_bpe_strings(handle=lm_tf, strings=input_strings_tf, bpe_merge_symbol="@@")
  with tf.Session() as session:
    output_scores = session.run(output_scores_tf, feed_dict={input_strings_tf: input_strings})
  print("input strings:", input_strings)
  print("output scores:", output_scores)
  assert isinstance(output_scores, numpy.ndarray)
  assert_equal(output_scores.shape, (len(input_strings),))
  assert_almost_equal(output_scores[0], -9.251298)  # example from above
  assert_equal(output_scores[0], output_scores[1])
  assert_equal(output_scores[2], output_scores[3])
  print("Scores are as expected.")


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
