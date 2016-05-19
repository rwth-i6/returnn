
import NativeOp
import numpy as np
from numpy.testing.utils import assert_almost_equal
import theano.tensor as T
import TheanoUtil
f32 = "float32"

import better_exchook
from Log import log

better_exchook.replace_traceback_format_tb()
log.initialize()  # some code might need it


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
  W2 = NativeOp.sparse_to_dense(s0, s1, w, m, n_time, n_dim).eval()
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
  W2 = NativeOp.sparse_to_dense(s0, s1, w, m, n_time, n_dim).eval()
  assert_almost_equal(W, W2)
  assert_almost_equal(
    np.sum(W, axis=2),
    np.ones((n_time, n_batch), dtype=f32)
  )


def test_crossentropy_softmax_and_gradient_z_sparse():
  n_time = 3
  n_batch = 2
  n_dim = 5
  s0 = np.array([[0, 0], [0, 1],  [1, 1],  [1, 2],  [1, 2],  [2, 2],  [2, 2]], dtype=f32)
  s1 = np.array([[1, 2], [2, 3],  [1, 1],  [2, 0],  [4, 1],  [3, 3],  [4, 4]], dtype=f32)
  w =  np.array([[.3,1], [.7,.4], [.1,.6], [.3,.2], [.6,.3], [.4,.5], [.6,9]], dtype=f32)
  m =  np.array([[1, 1], [1, 1],  [1, 1],  [1, 1],  [1, 1],  [1, 1],  [1, 0]], dtype=f32)
  np.random.seed(123)
  z = np.random.randn(n_time, n_batch, n_dim).astype(f32)
  print("z:\n%r" % z)
  print("softmax:\n%r" % TheanoUtil.softmax(z).eval())
  z_mask = np.array([[1,1], [1,1], [1,1]], dtype=f32)
  args = (z, z_mask, s0, s1, w, m)
  ce1, gradz1 = NativeOp.crossentropy_softmax_and_gradient_z_sparse(*args)
  ce2, gradz2 = NativeOp.crossentropy_softmax_and_gradient_z_sparse__slow(*args)
  ce1 = ce1.eval()
  ce2 = ce2.eval()
  gradz1 = gradz1.eval()
  gradz2 = gradz2.eval()
  print("ce1:\n%r" % ce1)
  print("ce2:\n%r" % ce2)
  print("gradz1:\n%r" % gradz1)
  print("gradz2:\n%r" % gradz2)
  assert_almost_equal(ce1, ce2)
  assert_almost_equal(gradz1, gradz2)
