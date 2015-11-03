
import numpy
from nose.tools import assert_equal
import MultiBatchBeam
from MultiBatchBeam import *
import better_exchook
better_exchook.replace_traceback_format_tb()


naive_multi_batch_beam = MultiBatchBeam._naive_multi_batch_beam
simplified_numpy_multi_batch_beam = MultiBatchBeam._simplified_numpy_multi_batch_beam

def numpy_multi_batch_beam(array, start_idxs, batch_lens, beam_width, wrap_mode, idx_dim=0, batch_dim=1):
  op = MultiBatchBeamOp(wrap_mode, idx_dim, batch_dim)
  beam_out = [None]
  op.perform(None, (array, start_idxs, batch_lens, beam_width), (beam_out,))
  return beam_out[0]

def theano_multi_batch_beam(*args, **kwargs):
  res = MultiBatchBeam._theano_multi_batch_beam(*args, **kwargs)
  return res.eval()


def compare_implementations(*args, **kwargs):
  results = {}
  for method in ["numpy", "naive", "theano", "simplified_numpy"]:
    m = globals()["%s_multi_batch_beam" % method]
    try:
      res = m(*args, **kwargs)
    except NotImplementedError:
      pass
    else:
      results[method] = res
  assert len(results) > 1
  for k, v in sorted(results.items()):
    print "%s:" % k
    print v
  reference = sorted(results.keys())[0]
  for k in sorted(results.keys())[1:]:
    assert_equal(results[k].shape, results[reference].shape)
    numpy.testing.assert_almost_equal(results[k], results[reference])
  return results[reference]


def test_numpy_perform_simple():
  n_time = 5
  n_batch = 1
  n_dim = 1
  array = numpy.arange(n_time * n_batch * n_dim).reshape(n_time, n_batch, n_dim)
  start_idxs = numpy.array([0])
  batch_lens = numpy.array([n_time])
  beam_width = n_time
  beam = compare_implementations(array, start_idxs, batch_lens, beam_width, "wrap_around")
  numpy.testing.assert_almost_equal(beam, array)


def test_numpy_perform_simple_1a():
  n_time = 5
  n_batch = 2
  array = numpy.arange(n_time * n_batch).reshape(n_time, n_batch)
  start_idxs = numpy.array([0, 0])
  batch_lens = numpy.array([n_time, n_time])
  beam_width = n_time
  beam = compare_implementations(array, start_idxs, batch_lens, beam_width, "wrap_around")
  numpy.testing.assert_almost_equal(beam, array)


def test_numpy_perform_simple_1():
  n_time = 5
  n_batch = 2
  n_dim = 1
  array = numpy.arange(n_time * n_batch * n_dim).reshape(n_time, n_batch, n_dim)
  start_idxs = numpy.array([0, 0])
  batch_lens = numpy.array([n_time, n_time])
  beam_width = n_time
  beam = compare_implementations(array, start_idxs, batch_lens, beam_width, "wrap_around")
  numpy.testing.assert_almost_equal(beam, array)


def test_numpy_perform_simple_2():
  n_time = 5
  n_batch = 2
  n_dim = 2
  array = numpy.arange(n_time * n_batch * n_dim).reshape(n_time, n_batch, n_dim)
  start_idxs = numpy.array([0, 0])
  batch_lens = numpy.array([n_time, n_time])
  beam_width = n_time
  beam = compare_implementations(array, start_idxs, batch_lens, beam_width, "wrap_around")
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
  compare_implementations(array, start_idxs, batch_lens, beam_width, "wrap_around")


def test_numpy_perform_2_wrap():
  array = numpy.array([range(10)]).T
  start_idxs = numpy.array([-2])
  batch_lens = numpy.array([array.shape[0]])
  beam_width = 4
  beam = compare_implementations(array, start_idxs, batch_lens, beam_width, "wrap_around")
  assert beam.shape == (4, 1)
  assert_equal(list(beam[:, 0]), [8, 9, 0, 1])
