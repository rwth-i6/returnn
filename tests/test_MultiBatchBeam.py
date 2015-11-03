
import numpy
from nose.tools import assert_equal
import MultiBatchBeam
from MultiBatchBeam import *
import better_exchook
better_exchook.replace_traceback_format_tb()


naive_multi_batch_beam = MultiBatchBeam._naive_multi_batch_beam
simplified_numpy_multi_batch_beam = MultiBatchBeam._simplified_numpy_multi_batch_beam

def numpy_multi_batch_beam(array, start_idxs, batch_lens, beam_width, wrap_mode, idx_dim=0, batch_dim=1):
  array = T.as_tensor(array)
  start_idxs = T.as_tensor(start_idxs)
  batch_lens = T.as_tensor(batch_lens)
  beam_width = T.as_tensor(beam_width)
  op = MultiBatchBeamOp(wrap_mode, idx_dim, batch_dim)
  beam = op(array, start_idxs, batch_lens, beam_width)
  return beam.eval()

def theano_cpu_multi_batch_beam(*args, **kwargs):
  res = MultiBatchBeam._theano_cpu_multi_batch_beam(*args, **kwargs)
  return res.eval()

naive_multi_batch_beam_grad = MultiBatchBeam._naive_multi_batch_beam_grad

def numpy_multi_batch_beam_grad(array, start_idxs, batch_lens, beam_width, wrap_mode, idx_dim=0, batch_dim=1, output_grad=None):
  array = T.as_tensor(array)
  start_idxs = T.as_tensor(start_idxs)
  batch_lens = T.as_tensor(batch_lens)
  beam_width = T.as_tensor(beam_width)
  output_grad = T.as_tensor(output_grad)
  op = MultiBatchBeamOp(wrap_mode, idx_dim, batch_dim)
  D_array, D_start_idxs, D_batch_lens, D_beam_width = op.grad((array, start_idxs, batch_lens, beam_width), (output_grad, ))
  return D_array.eval()

def theano_cpu_multi_batch_beam_grad(array, start_idxs, batch_lens, beam_width, wrap_mode, idx_dim=0, batch_dim=1, output_grad=None):
  array = T.as_tensor(array)
  start_idxs = T.as_tensor(start_idxs)
  batch_lens = T.as_tensor(batch_lens)
  beam_width = T.as_tensor(beam_width)
  output_grad = T.as_tensor(output_grad)
  res = MultiBatchBeam._theano_cpu_multi_batch_beam(array, start_idxs, batch_lens, beam_width, wrap_mode, idx_dim, batch_dim)
  D_array = T.grad(None, wrt=array, known_grads={res: output_grad})
  return D_array.eval()


def compare_implementations(*args, **kwargs):
  results = {}
  for method in ["numpy", "naive", "theano_cpu", "simplified_numpy"]:
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


def compare_grad_implementations(*args, **kwargs):
  results = {}
  for method in ["numpy", "naive", "theano_cpu"]:
    m = globals()["%s_multi_batch_beam_grad" % method]
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


def test_grad_simple():
  array = numpy.array([range(10)], dtype="float32").T
  n_batch = array.shape[1]
  assert n_batch == 1
  start_idxs = numpy.array([-2])
  batch_lens = numpy.array([array.shape[0]])
  beam_width = 4
  D_beam = numpy.arange(4, dtype="float32").reshape(beam_width, n_batch)
  D_array = compare_grad_implementations(array, start_idxs, batch_lens, beam_width, "wrap_around", output_grad=D_beam)
  assert D_array.shape == array.shape
  assert_equal(list(D_array[:, 0]), [2, 3] + [0] * 6 + [0, 1])


def test_random_wrap():
  n_time = 100
  n_batch = 10
  n_dim = 5
  beam_width = 20
  numpy.random.seed(123)
  array = numpy.random.random(n_time * n_batch * n_dim).reshape(n_time, n_batch, n_dim)
  batch_lens = numpy.array([numpy.random.randint(n_time / 5, n_time) for i in range(n_batch)])
  start_idxs = numpy.array([numpy.random.randint(-n_time, n_time) for i in range(n_batch)])
  beam = compare_implementations(array, start_idxs, batch_lens, beam_width, "wrap_around")
  D_beam = numpy.random.random(beam.shape)
  D_array = compare_grad_implementations(array, start_idxs, batch_lens, beam_width, "wrap_around", output_grad=D_beam)
