
import NativeOp
import numpy as np
from numpy.testing.utils import assert_almost_equal
import theano.tensor as T
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
