
# start test like this:  nosetests-2.7  tests/test_TFUtil.py

from __future__ import print_function


import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import sys
sys.path += ["."]  # Python 3 hack
from TFUtil import *
from nose.tools import assert_equal, assert_is_instance, assert_is
import numpy.testing
import better_exchook
better_exchook.replace_traceback_format_tb()


session = tf.InteractiveSession()


def test_tf_version_tuple():
  print("TF version:", tf.__version__)
  print("TF version tuple:", tf_version_tuple())


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
  real = windowed_nd(source, window=window, time_axis=0, new_window_axis=1).eval()
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
  real = windowed_nd(source, window=window, time_axis=0, new_window_axis=1).eval()
  numpy.testing.assert_almost_equal(naive, real)


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
