
import sys
import os

import _setup_test_env  # noqa
import returnn.theano.native_op as theano_native_op
import numpy as np
from numpy.testing.utils import assert_almost_equal
import theano.tensor as T
import returnn.theano.util as theano_util
f32 = "float32"

from returnn.util import better_exchook
from returnn.log import log

better_exchook.replace_traceback_format_tb()
log.initialize()  # some code might need it
theano_util.monkey_patches()


def test_sparse_to_dense():
  n_time = 3
  n_batch = 2
  n_dim = 5
  s0 = np.array([[0,0], [0,1], [1,1], [1,2], [1,2], [2,2], [2,2]], dtype=f32)
  s1 = np.array([[1,2], [2,3], [1,1], [2,0], [4,1], [3,3], [4,4]], dtype=f32)
  w =  np.array([[1,2], [2,1], [1,2], [3,4], [5,6], [7,8], [9,9]], dtype=f32)
  m =  np.array([[1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [1,0]], dtype=f32)
  W = np.array(
    [[[0, 1, 2, 0, 0], [0, 0, 2, 0, 0]],
     [[0, 1, 3, 0, 5], [0, 2, 0, 1, 0]],
     [[0, 0, 0, 7, 9], [4, 6, 0, 8, 0]]], dtype=f32)
  assert W.shape == (n_time, n_batch, n_dim)
  W2 = theano_native_op.sparse_to_dense(s0, s1, w, m, n_time, n_dim).eval()
  assert_almost_equal(W, W2)


def test_sparse_to_dense_2():
  n_time = 3
  n_batch = 2
  n_dim = 5
  s0 = np.array([[0, 0], [0, 1],  [1, 1],  [1, 2],  [1, 2],  [2, 2],  [2, 2]], dtype=f32)
  s1 = np.array([[1, 2], [2, 3],  [1, 1],  [2, 0],  [4, 1],  [3, 3],  [4, 4]], dtype=f32)
  w =  np.array([[.3,1], [.7,.4], [.1,.6], [.3,.2], [.6,.3], [.4,.5], [.6,9]], dtype=f32)
  m =  np.array([[1, 1], [1, 1],  [1, 1],  [1, 1],  [1, 1],  [1, 1],  [1, 0]], dtype=f32)
  W = np.array(
    [[[0,.3,.7, 0, 0], [0, 0, 1, 0, 0]],
     [[0,.1,.3, 0,.6], [0,.6, 0,.4, 0]],
     [[0, 0, 0,.4,.6], [.2,.3,0,.5, 0]]], dtype=f32)
  assert W.shape == (n_time, n_batch, n_dim)
  W2 = theano_native_op.sparse_to_dense(s0, s1, w, m, n_time, n_dim).eval()
  assert_almost_equal(W, W2)
  assert_almost_equal(
    np.sum(W, axis=2),
    np.ones((n_time, n_batch), dtype=f32))


def test_crossentropy_softmax_and_gradient_z_sparse():
  n_time = 3
  n_batch = 2
  n_dim = 5
  s0 = np.array([[0, 0], [0, 1],  [1, 1],  [1, 2],  [1, 2],  [2, 2],  [2, 2]], dtype=f32)
  s1 = np.array([[1, 2], [2, 3],  [1, 1],  [2, 0],  [4, 1],  [3, 3],  [4, 4]], dtype=f32)
  w =  np.array([[.3,1], [.7,.4], [.1,.6], [.3,.2], [.6,.3], [.4,.5], [.6,9]], dtype=f32)
  m =  np.array([[1, 1], [1, 1],  [1, 1],  [1, 1],  [1, 1],  [1, 1],  [1, 0]], dtype=f32)
  print("y_target:\n%r" % theano_native_op.sparse_to_dense(s0, s1, w, m, n_time, n_dim).eval())
  np.random.seed(123)
  z = np.random.randn(n_time, n_batch, n_dim).astype(f32)
  print("z:\n%r" % z)
  print("y (softmax(z)):\n%r" % theano_util.softmax(z).eval())
  z_mask = np.array([[1,1], [1,1], [1,1]], dtype=f32)
  args = (z, z_mask, s0, s1, w, m)
  ce1, gradz1 = theano_native_op.crossentropy_softmax_and_gradient_z_sparse(*args)
  ce2, gradz2 = theano_native_op.crossentropy_softmax_and_gradient_z_sparse__slow(*args)
  ce1 = ce1.eval()
  ce2 = ce2.eval()
  gradz1 = gradz1.eval()
  gradz2 = gradz2.eval()
  print("ce1:\n%r" % ce1)
  print("ce2:\n%r" % ce2)
  print("gradz1:\n%r" % gradz1)
  print("gradz2:\n%r" % gradz2)
  assert_almost_equal(ce1, ce2, decimal=5)
  assert_almost_equal(gradz1, gradz2, decimal=5)


def test_crossentropy_softmax_and_gradient_z_sparse_viterbi():
  n_time = 3
  n_batch = 2
  n_dim = 5
  alignment = np.array([[0, 1], [1, 2], [2, 3]], dtype="int32")
  mask = np.array([[1, 1], [1, 1], [1, 1]], dtype=f32)
  y_t, y_i, y_w, y_mask = theano_native_op.onehot_to_sparse(alignment, mask)
  np.random.seed(123)
  z = np.random.randn(n_time, n_batch, n_dim).astype(f32)
  z_mask = np.array([[1,1], [1,1], [1,1]], dtype=f32)
  nll1, _pcx1 = T.nnet.crossentropy_softmax_1hot(x=T.as_tensor_variable(z).reshape((n_time * n_batch, n_dim)), y_idx=T.as_tensor_variable(alignment).reshape((n_time * n_batch,)))
  nll2, _gradz2 = theano_native_op.crossentropy_softmax_and_gradient_z_sparse(z, z_mask, y_t, y_i, y_w, y_mask)
  nll1 = nll1.eval()
  nll2 = nll2.eval()
  print("nll1:\n%r" % nll1)
  print("nll2:\n%r" % nll2)


def test_max_and_argmax_sparse():
  n_time = 3
  n_batch = 2
  n_dim = 5
  s0 = np.array([[0,0], [0,1], [1,1], [1,2], [1,2], [2,2], [2,2]], dtype=f32)
  s1 = np.array([[1,2], [2,3], [1,1], [2,0], [4,1], [3,3], [4,4]], dtype=f32)
  w =  np.array([[1,2], [2,1], [1,2], [3,4], [5,6], [7,8], [9,13]], dtype=f32)
  m =  np.array([[1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [1,0]], dtype=f32)
  print("W:\n%r" % theano_native_op.sparse_to_dense(s0, s1, w, m, n_time, n_dim).eval())
  init_out_max = T.zeros((n_time, n_batch), dtype=f32)
  init_out_arg = T.zeros((n_time, n_batch), dtype=f32)
  max1, arg1 = theano_native_op.max_and_argmax_sparse(s0, s1, w, m, init_out_max, init_out_arg)
  W = theano_native_op.sparse_to_dense(s0, s1, w, m, n_time, n_dim)
  assert W.ndim == 3
  max2, arg2 = T.max_and_argmax(W, axis=2)
  arg0 = np.array([[2, 2], [4, 1], [4, 3]])
  max0 = np.array([[2, 2], [5, 2], [9, 8]])
  arg1 = arg1.eval()
  arg2 = arg2.eval()
  max1 = max1.eval()
  max2 = max2.eval()
  print("arg0:\n%r" % arg0)
  print("arg1:\n%r" % arg1)
  print("arg2:\n%r" % arg2)
  print("max0:\n%r" % max0)
  print("max1:\n%r" % max1)
  print("max2:\n%r" % max2)
  assert_almost_equal(arg0, arg1)
  assert_almost_equal(arg0, arg2)
  assert_almost_equal(max0, max1)
  assert_almost_equal(max0, max2)
