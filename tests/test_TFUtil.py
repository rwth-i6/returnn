
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
