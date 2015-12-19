
from nose.tools import assert_equal, assert_is_instance, assert_in, assert_not_in, assert_true, assert_false, assert_greater, assert_almost_equal, assert_is
import numpy
import numpy.testing
from TheanoUtil import *


def naive_windowed_batch(source, window):
  assert source.ndim == 3  # (time,batch,dim). not sure how to handle other cases
  n_time = source.shape[0]
  n_batch = source.shape[1]
  n_dim = source.shape[2]
  w_right = window / 2
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
  print "source:"
  print source
  naive = naive_windowed_batch(source, window=window)
  real = windowed_batch(source, window=window).eval()
  print "naive:"
  print naive
  print "real:"
  print real
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
