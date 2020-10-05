
from __future__ import print_function

import sys
import os

import _setup_test_env  # noqa
import sys
import numpy
import theano
import theano.scan_module.scan_op
from nose.tools import assert_equal, assert_is, assert_is_instance
import returnn.theano.ops.multi_batch_beam as multi_batch_beam
from returnn.theano.ops.multi_batch_beam import *
import returnn.theano.ops.multi_batch_beam as multi_batch_beam_mod
import returnn.theano.util as theano_util
import theano.printing
from pprint import pprint
from returnn.util import better_exchook


theano_util.monkey_patches()


naive_multi_batch_beam = multi_batch_beam_mod._naive_multi_batch_beam

def numpy_multi_batch_beam(array, start_idxs, batch_lens, beam_width, wrap_mode, pad_left=0, pad_right=0, idx_dim=0, batch_dim=1):
  array = T.as_tensor(array)
  start_idxs = T.as_tensor(start_idxs)
  batch_lens = T.as_tensor(batch_lens)
  beam_width = T.as_tensor(beam_width)
  op = MultiBatchBeamOp(wrap_mode, idx_dim, batch_dim)
  beam = op(array, start_idxs, batch_lens, beam_width, pad_left, pad_right)
  return beam.eval()

def theano_cpu_multi_batch_beam(*args, **kwargs):
  res = multi_batch_beam_mod._theano_cpu_multi_batch_beam(*args, **kwargs)
  return res.eval()

naive_multi_batch_beam_grad = multi_batch_beam_mod._naive_multi_batch_beam_grad

def _eval_or_None(x):
  if isinstance(x.type, T.DisconnectedType):
    return None
  return x.eval()

def theano_op_multi_batch_beam_grad(array, start_idxs, batch_lens, beam_width, wrap_mode, pad_left=0, pad_right=0, idx_dim=0, batch_dim=1, output_grad=None):
  array = T.as_tensor(array)
  start_idxs = T.as_tensor(start_idxs)
  batch_lens = T.as_tensor(batch_lens)
  beam_width = T.as_tensor(beam_width)
  pad_left = T.as_tensor(pad_left)
  pad_right = T.as_tensor(pad_right)
  output_grad = T.as_tensor(output_grad)
  op = MultiBatchBeamOp(wrap_mode, idx_dim, batch_dim)
  D_array, D_start_idxs, D_batch_lens, D_beam_width, D_pad_left, D_pad_right = op.grad((array, start_idxs, batch_lens, beam_width, pad_left, pad_right), (output_grad, ))
  return list(map(_eval_or_None, [D_array, D_pad_left, D_pad_right]))

def theano_cpu_multi_batch_beam_grad(array, start_idxs, batch_lens, beam_width, wrap_mode, pad_left=0, pad_right=0, idx_dim=0, batch_dim=1, output_grad=None):
  array = T.as_tensor(array)
  start_idxs = T.as_tensor(start_idxs)
  batch_lens = T.as_tensor(batch_lens)
  beam_width = T.as_tensor(beam_width)
  pad_left = T.as_tensor(pad_left)
  pad_right = T.as_tensor(pad_right)
  output_grad = T.as_tensor(output_grad)
  res = multi_batch_beam_mod._theano_cpu_multi_batch_beam(array, start_idxs, batch_lens, beam_width, wrap_mode, pad_left, pad_right, idx_dim, batch_dim)
  D_array, D_pad_left, D_pad_right = T.grad(None, wrt=[array, pad_left, pad_right], known_grads={res: output_grad}, disconnected_inputs="ignore", return_disconnected="Disconnected")
  return list(map(_eval_or_None, [D_array, D_pad_left, D_pad_right]))


def compare_implementations(*args, **kwargs):
  results = {}
  for method in ["numpy", "naive", "theano_cpu"]:
    m = globals()["%s_multi_batch_beam" % method]
    try:
      res = m(*args, **kwargs)
    except NotImplementedError:
      pass
    else:
      results[method] = res
  assert len(results) > 1
  for k, v in sorted(results.items()):
    print("fwd %s:" % k)
    print(v)
  reference = sorted(results.keys())[0]
  for k in sorted(results.keys())[1:]:
    assert_equal(results[k].shape, results[reference].shape)
    numpy.testing.assert_almost_equal(results[k], results[reference])
  return results[reference]


def compare_grad_implementations(*args, **kwargs):
  results = {}
  for method in ["theano_op", "naive", "theano_cpu"]:
    m = globals()["%s_multi_batch_beam_grad" % method]
    try:
      res = m(*args, **kwargs)
    except NotImplementedError:
      pass
    else:
      results[method] = res
  assert len(results) > 1
  for k, v in sorted(results.items()):
    print("bwd %s:" % k)
    print(v)
  reference = sorted(results.keys())[0]
  for k in sorted(results.keys())[1:]:
    assert_equal(len(results[k]), len(results[reference]))
    for i in range(len(results[k])):
      if i > 0: break  # XXX: This is for D_pad_left / D_pad_right...
      if results[k][i] is None or results[reference][i] is None:
        assert_is(results[k][i], results[reference][i])
        continue
      assert_equal(results[k][i].shape, results[reference][i].shape)
      # The summation of the grad can be quite numerically unstable, thus the low decimal.
      numpy.testing.assert_almost_equal(results[k][i], results[reference][i], decimal=4)
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
  array = numpy.array([42,43,44] + list(range(n_time * n_batch * n_dim))[:-3]).reshape(n_time, n_batch, n_dim) + 0.1
  print("array shape:", array.shape)
  start_idxs = numpy.array([1, -2, 10, 0, 1])
  batch_lens = numpy.array([11, 2, 11, 2, 11])
  beam_width = 5
  compare_implementations(array, start_idxs, batch_lens, beam_width, "wrap_around")


def test_numpy_perform_2_wrap():
  array = numpy.array([list(range(10))]).T
  start_idxs = numpy.array([-2])
  batch_lens = numpy.array([array.shape[0]])
  beam_width = 4
  beam = compare_implementations(array, start_idxs, batch_lens, beam_width, "wrap_around")
  assert beam.shape == (4, 1)
  assert_equal(list(beam[:, 0]), [8, 9, 0, 1])


def test_grad_simple():
  array = numpy.array([list(range(10))], dtype="float32").T
  print("array shape:", array.shape)
  n_batch = array.shape[1]
  assert n_batch == 1
  start_idxs = numpy.array([-2])
  batch_lens = numpy.array([array.shape[0]])
  beam_width = 4
  D_beam = numpy.arange(4, dtype="float32").reshape(beam_width, n_batch)
  D_array, _, _ = compare_grad_implementations(array, start_idxs, batch_lens, beam_width, "wrap_around", output_grad=D_beam)
  assert D_array.shape == array.shape
  assert_equal(list(D_array[:, 0]), [2, 3] + [0] * 6 + [0, 1])


def test_random_wrap():
  n_time = 100
  n_batch = 10
  n_dim = 5
  beam_width = 20
  numpy.random.seed(123)
  array = numpy.random.random(n_time * n_batch * n_dim).reshape(n_time, n_batch, n_dim)
  batch_lens = numpy.array([numpy.random.randint(n_time // 5, n_time) for i in range(n_batch)])
  start_idxs = numpy.array([numpy.random.randint(-n_time, n_time) for i in range(n_batch)])
  beam = compare_implementations(array, start_idxs, batch_lens, beam_width, "wrap_around")
  D_beam = numpy.random.random(beam.shape)
  compare_grad_implementations(array, start_idxs, batch_lens, beam_width, "wrap_around", output_grad=D_beam)

def test_random_pad():
  n_time = 100
  n_batch = 10
  n_dim = 5
  beam_width = 20
  wrap_mode = "pad"
  pad_left = 42
  pad_right = -17
  numpy.random.seed(123)
  array = numpy.random.random(n_time * n_batch * n_dim).reshape(n_time, n_batch, n_dim)
  batch_lens = numpy.array([numpy.random.randint(n_time // 5, n_time) for i in range(n_batch)])
  start_idxs = numpy.array([numpy.random.randint(-n_time, n_time) for i in range(n_batch)])
  beam = compare_implementations(array, start_idxs, batch_lens, beam_width, wrap_mode, pad_left, pad_right)
  D_beam = numpy.random.random(beam.shape)
  compare_grad_implementations(array, start_idxs, batch_lens, beam_width, wrap_mode, pad_left, pad_right, output_grad=D_beam)

def test_inc_subtensor():
  # If there are some indexes multiple times in the subtensor,
  # we expect for inc_subtensor that they are all accumulated.
  a = T.inc_subtensor(T.arange(10)[[0, 3, 5, 0]], numpy.array([-1,-2,-3,-4])).eval()
  assert_equal(list(a), [-5,  1,  2,  1,  4,  2,  6,  7,  8,  9])

def test_inplace_grad_add_simple_on_zero():
  n_time = 10
  n_batch = 5
  n_dim = 2
  beam_width = 3
  wrap_mode = "pad"
  numpy.random.seed(123)
  batch_lens = numpy.array([numpy.random.randint(n_time // 5, n_time) for i in range(n_batch)])
  start_idxs = numpy.array([numpy.random.randint(-n_time, n_time) for i in range(n_batch)])
  D_beam = numpy.random.random((beam_width, n_batch, n_dim)).astype("float32")
  batch_lens = T.as_tensor_variable(batch_lens, name="batch_lens")
  start_idxs = T.as_tensor_variable(start_idxs, name="start_idxs")
  D_beam = T.as_tensor_variable(D_beam, name="D_beam")
  beam_width = theano.shared(beam_width, name="beam_width")

  D_array_and_pad_0 = T.zeros((n_time + 2, n_batch), dtype="float32")
  D_array_and_pad_1 = MultiBatchBeamGradAddOp(wrap_mode)(D_array_and_pad_0, start_idxs, batch_lens, beam_width, D_beam)
  D_array_and_pad_2 = MultiBatchBeamGradAddOp(wrap_mode)(D_array_and_pad_1, start_idxs, batch_lens, beam_width, D_beam)

  print("\ngraph for D_array_and_pad (unoptimized):")
  theano.printing.debugprint(D_array_and_pad_2)

  f = theano.function(inputs=[], outputs=[D_array_and_pad_2], mode="FAST_RUN")

  print("\ngraph for function (optimized):")
  theano.printing.debugprint(f.maker.fgraph)

  print("\ngraph toposort:")
  pprint(f.maker.fgraph.toposort())

  assert_is_instance(f.maker.fgraph, theano.gof.fg.FunctionGraph)
  out0 = f.maker.fgraph.outputs[0]
  print("\nout0:", out0, type(out0))
  assert_is_instance(out0, T.TensorVariable)
  assert_is_instance(out0.owner, theano.Apply)
  assert_is_instance(out0.owner.op, MultiBatchBeamGradAddOp)
  print("\nfinal op in fgraph:", out0.owner.op)
  assert_is(out0.owner.op.inplace, True)

def test_simple_inplace_scan():
  #theano.config.scan.allow_output_prealloc = False
  def remove_opt(name):
    d = theano.compile.optdb.__db__[name]
    assert len(d) == 1
    obj = list(d)[0]
    for k, v in theano.compile.optdb.__db__.items():
      if obj in v:
        v.remove(obj)
    del theano.compile.optdb.__db__[name]
    theano.compile.optdb._names.remove(name)
    del theano.compile.optdb.__position__[name]
  #remove_opt("scan_eqopt1")
  #remove_opt("scan_eqopt2")
  #remove_opt("scanOp_make_inplace")

  # inplace_elemwise_optimizer = T.opt.inplace_elemwise_optimizer_op(T.Elemwise)
  # theano.compile.optdb.register('inplace_elemwise_opt_2', inplace_elemwise_optimizer, 80,
  #                               'fast_run', 'inplace')

  def step(t, last):
    return [last + 1.0]
  # For the demo to work inplace, it must be a vector, not a scalar (Theano internals).
  o, updates = theano.scan(step, sequences=[T.arange(10)], outputs_info=[numpy.array([0.0])])
  assert len(updates) == 0

  o_last = T.tensor_copy(o[-1])
  f = theano.function(inputs=[], outputs=[o_last], mode="FAST_RUN")

  print("result:", f())
  print("f:")
  theano.printing.debugprint(f)

  o_last_opt = f.maker.fgraph.outputs[0]
  print("o_last_opt:", o_last_opt.owner)
  scan_v = o_last_opt.owner.inputs[0]
  assert isinstance(scan_v.owner.op, theano.scan_module.scan_op.Scan)
  print("scan:", scan_v)
  print("scan destroy_map:", scan_v.owner.op.destroy_map)
  print("scan inner func:", scan_v.owner.op.fn)

  node = scan_v.owner.op.fn.maker

  print("inner fgraph inputs:", node.fgraph.inputs)
  protected_inputs = [
      f.protected for f in node.fgraph._features if
      isinstance(f, theano.compile.function_module.Supervisor)]
  protected_inputs = sum(protected_inputs, [])  # flatten the list
  print("inner protected inputs:", protected_inputs, node.fgraph.outputs)

  #raise Exception("stop")

  theano.printing.debugprint(o_last_opt.owner)
  #assert isinstance(o_last_opt.owner.op, T.Subtensor)
  #assert isinstance(o_last_opt.owner.op, theano.scan_module.scan_op.Scan)

  #raise Exception("stop")


def test_inplace_grad_add():
  theano.config.scan.prefer_inplace = True
  theano.config.scan.greedy_non_seqs = True

  n_time = 10
  n_batch = 5
  n_dim = 2
  beam_width = 3
  wrap_mode = "pad"
  numpy.random.seed(123)
  array = numpy.random.random(n_time * n_batch * n_dim).astype("float32").reshape(n_time, n_batch, n_dim)
  batch_lens = numpy.array([numpy.random.randint(n_time // 5, n_time) for i in range(n_batch)])
  start_idxss = numpy.array([numpy.random.randint(-n_time, n_time) for i in range(n_batch * n_time)]).reshape(n_time, n_batch)
  D_beams = numpy.random.random((n_time, beam_width, n_batch, n_dim)).astype("float32")
  array = theano.shared(array, name="array")
  batch_lens = T.as_tensor_variable(batch_lens, name="batch_lens")
  start_idxss = T.as_tensor_variable(start_idxss, name="start_idxss")
  D_beams = T.as_tensor_variable(D_beams, name="D_beams")
  pad_left = T.as_tensor_variable(4, name="pad_left")  # XXX: Constant for now...
  pad_right = T.as_tensor_variable(-1, name="pad_right")  # XXX: Constant for now...
  beam_width = theano.shared(beam_width, name="beam_width")

  w = theano.shared(numpy.identity(n_dim, dtype="float32"))
  base = T.concatenate([T.nnet.sigmoid(T.dot(array, w))], axis=0)
  base.name = "base"

  def step(start_idxs, last_beam):
    beam = MultiBatchBeamOp(wrap_mode)(base, start_idxs, batch_lens, beam_width, pad_left, pad_right)
    return [beam]

  beam_zero = T.zeros((beam_width, n_batch, n_dim))
  beams, _ = theano.scan(step, sequences=[start_idxss], outputs_info=[beam_zero])

  #print "graph for beams:"
  #theano.printing.debugprint(beams)

  D_w, = T.grad(None, wrt=[w], known_grads={beams: D_beams})
  f = theano.function(inputs=[], outputs=[D_w], mode="FAST_RUN")

  print("\ngraph for first output (unoptimized):")
  theano.printing.debugprint(f.outputs[0])

  print("\ngraph for function (optimized):")
  theano.printing.debugprint(f.maker.fgraph)

  loop_apply = f.maker.fgraph.outputs[0].owner.inputs[0].owner
  print("loop_apply", loop_apply)

  #assert_is_instance(loop_apply.op, theano.scan_module.scan_op.Scan)

  #if not any([x.op.__class__.__name__ in ['GpuGemm', 'GpuGemv', 'GpuDot22', 'GpuElemwise']
  #            for x in f.maker.fgraph.toposort()]):
  #  print "It seems as if we don't use the GPU although we requested it."

  # TODO...
  #raise Exception("stop")
