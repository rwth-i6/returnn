
from __future__ import print_function

import os
import sys
import _setup_test_env  # noqa
from nose.tools import assert_equal, assert_is_instance, assert_in, assert_not_in, assert_true, assert_false, assert_greater, assert_almost_equal, assert_is
import numpy
import numpy.testing
from returnn.theano.util import *


monkey_patches()


def test_class_idx_seq_to_1_of_k():
  from theano.tensor.extra_ops import to_one_hot
  v = theano.tensor.as_tensor_variable(numpy.array([1, 2, 3, 5, 6]))
  out = to_one_hot(v, 10).eval()
  assert numpy.allclose(
      out,
      [[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]])
  out2 = class_idx_seq_to_1_of_k(v, 10).eval()
  assert numpy.allclose(out, out2)


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
  final = numpy.zeros((n_time, n_batch, n_dim * window), dtype=dtype)
  for t in range(n_time):
    for w in range(window):
      final[t, :, w * n_dim:(w + 1) * n_dim] = padded[t + w]
  return final


def test_windowed_batch_small():
  n_time = 2
  n_batch = 2
  n_dim = 2
  window = 3
  source = numpy.arange(1, n_time*n_batch*n_dim + 1).reshape(n_time, n_batch, n_dim)
  print("source:")
  print(source)
  naive = naive_windowed_batch(source, window=window)
  real = windowed_batch(source, window=window).eval()
  print("naive:")
  print(naive)
  print("real:")
  print(real)
  numpy.testing.assert_almost_equal(naive, real)


def test_windowed_batch_big():
  n_time = 11
  n_batch = 5
  n_dim = 7
  window = 3
  numpy.random.seed(123)
  source = numpy.random.random((n_time, n_batch, n_dim)).astype("float32")
  naive = naive_windowed_batch(source, window=window)
  real = windowed_batch(source, window=window).eval()
  numpy.testing.assert_almost_equal(naive, real)


def test_downsample_average():
  source = numpy.array([0.0, 1.0])
  d = downsample(T.as_tensor_variable(source), axis=0, factor=2, method="average").eval()
  numpy.testing.assert_allclose(d, numpy.array([0.5]))


def test_downsample_average_longer():
  source = numpy.array([0.0, 1.0, 2.0, 3.0, 4.0])
  d = downsample(T.as_tensor_variable(source), axis=0, factor=2, method="average").eval()
  numpy.testing.assert_allclose(d, numpy.array([0.5, 2.5]))


def test_downsample_max():
  source = numpy.array([0.0, 1.0])
  d = downsample(T.as_tensor_variable(source), axis=0, factor=2, method="max").eval()
  numpy.testing.assert_allclose(d, numpy.array([1.0]))


def test_downsample_min():
  source = numpy.array([0.0, 1.0])
  d = downsample(T.as_tensor_variable(source), axis=0, factor=2, method="min").eval()
  numpy.testing.assert_allclose(d, numpy.array([0.0]))


def test_downsample_max_ndim3():
  source = numpy.array([[[0.0, 1.0]]])
  d = downsample(T.as_tensor_variable(source), axis=2, factor=2, method="max").eval()
  numpy.testing.assert_allclose(d, numpy.array([[[1.0]]]))


def test_downsample_min_int():
  source = numpy.array([0, 1])
  d = downsample(T.as_tensor_variable(source), axis=0, factor=2, method="min").eval()
  numpy.testing.assert_allclose(d, numpy.array([0]))


def test_downsample_min_int_ndim2():
  source = numpy.array([[0, 3], [1, 4]])
  d = downsample(T.as_tensor_variable(source), axis=0, factor=2, method="min").eval()
  numpy.testing.assert_allclose(d, numpy.array([[0, 3]]))


def test_upsample():
  source = numpy.array([0.0, 1.0])
  u = upsample(T.as_tensor_variable(source), axis=0, factor=2).eval()
  numpy.testing.assert_allclose(u, numpy.array([0.0, 0.0, 1.0, 1.0]))


def test_upsample_target_len():
  source = numpy.array([0.0, 1.0])
  u = upsample(T.as_tensor_variable(source), axis=0, factor=2, target_axis_len=5).eval()
  numpy.testing.assert_allclose(u, numpy.array([0.0, 0.0, 1.0, 1.0, 1.0]))


def test_upsample_target_len_even():
  source = numpy.array([0.0, 1.0])
  u = upsample(T.as_tensor_variable(source), axis=0, factor=2, target_axis_len=4).eval()
  numpy.testing.assert_allclose(u, numpy.array([0.0, 0.0, 1.0, 1.0]))


def test_upsample_target_len_ndim3():
  source = numpy.array([[[0.0]], [[1.0]]])
  u = upsample(T.as_tensor_variable(source), axis=0, factor=2, target_axis_len=5).eval()
  numpy.testing.assert_allclose(u, numpy.array([[[0.0]], [[0.0]], [[1.0]], [[1.0]], [[1.0]]]))


def test_chunked_time_reverse():
  source = T.as_tensor_variable(numpy.array([0, 1, 2, 3, 4, 5, 6]))
  r = chunked_time_reverse(source, 3).eval()
  numpy.testing.assert_allclose(r, numpy.array([2, 1, 0, 5, 4, 3, 0]))


def test_indices_in_flatten_array():
  n_copies, n_cells = 5, 4
  n_complex_cells = n_cells // 2
  n_batch = 3
  static_rng = numpy.random.RandomState(1234)
  def make_permut():
    p = numpy.zeros((n_copies, n_cells), dtype="int32")
    for i in range(n_copies):
      p[i, :n_complex_cells] = static_rng.permutation(n_complex_cells)
      # Same permutation for imaginary part.
      p[i, n_complex_cells:] = p[i, :n_complex_cells] + n_complex_cells
    return T.constant(p)
  P = make_permut()  # (n_copies,n_cells) -> list of indices

  meminkey = T.as_tensor_variable(static_rng.rand(n_batch, n_cells).astype("float32"))
  i_t = T.ones((meminkey.shape[0],))  # (batch,)
  n_batch = i_t.shape[0]
  batches = T.arange(0, n_batch).dimshuffle(0, 'x', 'x')  # (batch,n_copies,n_cells)
  P_bc = P.dimshuffle('x', 0, 1)  # (batch,n_copies,n_cells)
  meminkeyP1 = meminkey[batches, P_bc]  # (batch,n_copies,n_cells)
  meminkeyP2 = meminkey.flatten()[indices_in_flatten_array(meminkey.ndim, meminkey.shape, batches, P_bc)]
  meminkeyP3 = meminkey[:, P]  # (batch,n_copies,n_cells)

  numpy.testing.assert_allclose(meminkeyP1.eval(), meminkeyP2.eval())
  numpy.testing.assert_allclose(meminkeyP1.eval(), meminkeyP3.eval())
