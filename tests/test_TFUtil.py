
# start test like this:  nosetests-2.7  tests/test_TFUtil.py  --nologcapture

import tensorflow as tf
import sys
sys.path += ["."]  # Python 3 hack
from TFUtil import *
from nose.tools import assert_equal, assert_is_instance
import numpy.testing
import better_exchook
better_exchook.replace_traceback_format_tb()


session = tf.InteractiveSession()


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
  assert tf.__version__.split(".")[:2] >= [1, 1]
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
  assert tf.__version__.split(".")[:2] >= [1, 1]
  n_base_time = 5
  n_in = 7
  n_beam = 3
  n_batch = 1
  ref_grad = tf.scatter_nd(
    indices=tf.zeros((n_beam, n_batch, 2), dtype=tf.int32),
    updates=tf.ones((n_beam, n_batch, n_in)),
    shape=(n_base_time, n_batch, n_in))
  session.run(ref_grad)

