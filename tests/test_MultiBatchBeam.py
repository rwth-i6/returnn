
import numpy
from nose.tools import assert_equal
from MultiBatchBeam import *
import better_exchook
better_exchook.replace_traceback_format_tb()


def naive_multi_batch_beam(array, start_idxs, batch_lens, beam_width, wrap_mode, idx_dim=0, batch_dim=1):
  assert array.ndim >= 2
  assert start_idxs.ndim == 1
  assert batch_lens.ndim == 1
  assert idx_dim < array.ndim
  assert batch_dim < array.ndim
  assert idx_dim != batch_dim
  n_batch = array.shape[batch_dim]
  assert start_idxs.shape == (n_batch, )
  assert batch_lens.shape == (n_batch, )

  assert idx_dim == 0, "not implemented otherwise"
  assert batch_dim == 1, "not implemented otherwise"

  beam = numpy.zeros((beam_width, n_batch) + array.shape[2:], dtype=array.dtype)
  for i0 in range(beam_width):
    for i1 in range(n_batch):
      idx = start_idxs[i1] + i0
      if wrap_mode == "wrap_around":
        idx = idx % batch_lens[i1]
      elif wrap_mode == "pad_zero":
        if idx < 0 or idx >= batch_lens[i1]:
          continue
      beam[i0, i1] = array[idx, i1]
  return beam


def numpy_multi_batch_beam(array, start_idxs, batch_lens, beam_width, wrap_mode, idx_dim=0, batch_dim=1):
  op = MultiBatchBeamOp(wrap_mode, idx_dim, batch_dim)
  beam_out = [None]
  op.perform(None, (array, start_idxs, batch_lens, beam_width), (beam_out,))
  return beam_out[0]


def compare_numpy_naive(*args):
  numpy_beam = numpy_multi_batch_beam(*args)
  naive_beam = naive_multi_batch_beam(*args)
  assert numpy_beam.shape == naive_beam.shape
  print "naive:"
  print naive_beam
  print "numpy:"
  print numpy_beam
  numpy.testing.assert_almost_equal(naive_beam, numpy_beam)
  return numpy_beam


def test_numpy_perform_simple():
  n_time = 5
  n_batch = 1
  n_dim = 1
  array = numpy.arange(n_time * n_batch * n_dim).reshape(n_time, n_batch, n_dim)
  start_idxs = numpy.array([0])
  batch_lens = numpy.array([n_time])
  beam_width = n_time
  beam = compare_numpy_naive(array, start_idxs, batch_lens, beam_width, "wrap_around")
  numpy.testing.assert_almost_equal(beam, array)


def test_numpy_perform_simple_1a():
  n_time = 5
  n_batch = 2
  array = numpy.arange(n_time * n_batch).reshape(n_time, n_batch)
  start_idxs = numpy.array([0, 0])
  batch_lens = numpy.array([n_time, n_time])
  beam_width = n_time
  beam = compare_numpy_naive(array, start_idxs, batch_lens, beam_width, "wrap_around")
  numpy.testing.assert_almost_equal(beam, array)


def test_numpy_perform_simple_1():
  n_time = 5
  n_batch = 2
  n_dim = 1
  array = numpy.arange(n_time * n_batch * n_dim).reshape(n_time, n_batch, n_dim)
  start_idxs = numpy.array([0, 0])
  batch_lens = numpy.array([n_time, n_time])
  beam_width = n_time
  beam = compare_numpy_naive(array, start_idxs, batch_lens, beam_width, "wrap_around")
  numpy.testing.assert_almost_equal(beam, array)


def test_numpy_perform_simple_2():
  n_time = 5
  n_batch = 2
  n_dim = 2
  array = numpy.arange(n_time * n_batch * n_dim).reshape(n_time, n_batch, n_dim)
  start_idxs = numpy.array([0, 0])
  batch_lens = numpy.array([n_time, n_time])
  beam_width = n_time
  beam = compare_numpy_naive(array, start_idxs, batch_lens, beam_width, "wrap_around")
  numpy.testing.assert_almost_equal(beam, array)


def test_numpy_perform_1():
  n_time = 11
  n_batch = 5
  n_dim = 3
  array = numpy.array([42,43,44] + range(n_time * n_batch * n_dim)[:-3]).reshape(n_time, n_batch, n_dim) + 0.1
  print "array shape:", array.shape
  start_idxs = numpy.array([1, -2, 10, 0, 1])
  batch_lens = numpy.array([11, 2, 11, 2, 11])
  beam_width = 5
  compare_numpy_naive(array, start_idxs, batch_lens, beam_width, "wrap_around")
