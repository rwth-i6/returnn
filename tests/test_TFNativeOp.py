
# start test like this:  nosetests-2.7  tests/test_TFNativeOp.py  --nologcapture

from __future__ import print_function

import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import sys
sys.path += ["."]  # Python 3 hack
from TFNativeOp import *
from TFUtil import is_gpu_available, CudaEnv
import Util
import unittest
from nose.tools import assert_equal, assert_is_instance
import numpy
import numpy.testing
from numpy.testing.utils import assert_almost_equal, assert_allclose
import os
import better_exchook
better_exchook.replace_traceback_format_tb()


CudaEnv.verbose_find_cuda = True
session = tf.InteractiveSession()


def dump_info():
  numpy_path = os.path.dirname(numpy.__file__)
  print("Numpy path: %r" % numpy_path)
  so_files = Util.sysexecOut("find %s | grep \"\.so\"" % numpy_path, shell=True)
  print("Numpy so files:\n---\n%s\n---\n" % so_files)
  so_files = [f for f in so_files.splitlines() if f]
  ldd = "ldd"
  if sys.platform == "darwin":
    ldd = "otool -L"
  objdump = "objdump -T"
  if sys.platform == "darwin":
    objdump = "otool -IHGv"
  for f in so_files:
    cmd = "%s %s" % (ldd, f)
    print("$ %s" % cmd)
    out = Util.sysexecOut(cmd, shell=True)
    print(out)
    cmd = "%s %s | { grep sgemm || true; }" % (objdump, f)
    print("$ %s" % cmd)
    out = Util.sysexecOut(cmd, shell=True)
    print(out)


def test_dummy():
  dump_info()
  #assert False


def test_make_lstm_op_auto_cuda():
  try:
    make_lstm_op()
  except tf.errors.NotFoundError:
    dump_info()
    raise


def test_make_lstm_op_no_cuda():
  try:
    OpMaker.with_cuda = False
    make_lstm_op()
  except tf.errors.NotFoundError:
    dump_info()
    raise
  finally:
    OpMaker.with_cuda = None


def test_NativeLstmCell():
  n_time = 2
  n_batch = 1
  n_hidden = 3
  cell = NativeLstmCell(n_hidden=n_hidden)
  inputs = tf.zeros([n_time, n_batch, n_hidden * 4])
  index = tf.ones([n_time, n_batch])
  outputs, final_state = cell(inputs, index)


def test_NativeLstmLowMemCell():
  n_time = 2
  n_batch = 1
  n_in = 5
  n_hidden = 3
  cell = NativeLstmLowMemCell(n_hidden=n_hidden, n_input_dim=n_in)
  inputs = tf.zeros([n_time, n_batch, n_in])
  index = tf.ones([n_time, n_batch])
  outputs, final_state = cell(inputs, index)


def test_LstmLowMem_fwd_simple_1():
  lstm = make_op(NativeOp.LstmLowMem)
  n_time = 1
  n_batch = 1
  n_in = 1
  n_cells = 1
  vX = 0.2
  vb = 0.3
  vy0 = 0.1
  vc0 = 0.1
  X = tf.ones((n_time, n_batch, n_in)) * vX
  W = tf.ones((n_in + n_cells, n_cells * 4))
  b = tf.ones((n_cells * 4,)) * vb
  y0 = tf.ones((n_batch, n_cells)) * vy0
  c0 = tf.ones((n_batch, n_cells)) * vc0
  i = tf.ones((n_time, n_batch))
  start = tf.constant(0)
  step = tf.constant(1)
  Y, C, d = lstm(X, W, b, y0, c0, i, start, step)
  vY, vC, vd = session.run((Y, C, d))
  print("vY:", vY)
  print("vC:", vC)
  print("vd:", vd)
  assert_equal(vY.shape, (n_time, n_batch, n_cells))
  assert_equal(vC.shape, (n_time, n_batch, n_cells))
  assert_equal(d.shape, (n_batch, n_cells))
  vintern = vX + vy0 + vb
  vcellIn = numpy.tanh(vintern)
  vgates = 1. / (1. + numpy.exp(-vintern))  # sigmoid
  vc = vgates * vc0 + vgates * vcellIn
  vh = vgates * numpy.tanh(vc)
  assert_allclose(vY[0, 0, 0], vh)
  assert_allclose(vC[0, 0, 0], vc)
  assert_allclose(vd[0, 0], vc)


def test_LstmLowMem_bwd_simple_1():
  lstm = make_op(NativeOp.LstmLowMem)
  lstm_grad = lstm.grad_op
  n_time = 1
  n_batch = 1
  n_in = 1
  n_cells = 1
  vX = 0.2
  vb = 0.3
  vy0 = 0.1
  vc0 = 0.1
  vx_h = numpy.array([vX, vy0])
  print("vx_h:", vx_h)
  vintern = numpy.sum(vx_h) + vb
  print("vintern:", vintern)
  vcellIn = numpy.tanh(vintern)
  vgates = 1. / (1. + numpy.exp(-vintern))  # sigmoid
  print("vgates:", vgates)
  vc = vgates * vc0 + vgates * vcellIn
  vh = vgates * numpy.tanh(vc)
  X = tf.ones((n_time, n_batch, n_in)) * vX
  W = tf.ones((n_in + n_cells, n_cells * 4))
  b = tf.ones((n_cells * 4,)) * vb
  y0 = tf.ones((n_batch, n_cells)) * vy0
  c0 = tf.ones((n_batch, n_cells)) * vc0
  i = tf.ones((n_time, n_batch))
  start = tf.constant(0)
  step = tf.constant(1)
  Y = tf.ones((n_time, n_batch, n_cells)) * vh
  C = tf.ones((n_time, n_batch, n_cells)) * vc
  vDY = 1.5
  vDd = 1.2
  DY = tf.ones((n_time, n_batch, n_cells)) * vDY
  Dd = tf.ones((n_batch, n_cells)) * vDd
  DX, DW, Db, Dh, Dc = lstm_grad(X, W, b, y0, c0, i, start, step,   Y, C,   DY, Dd)
  vDX, vDW, vDb, vDh, vDc = session.run((DX, DW, Db, Dh, Dc))
  print("op vDX:", vDX)
  print("op vDW:", vDW)
  print("op vDb:", vDb)
  print("op vDh:", vDh)
  print("op vDc:", vDc)
  assert_equal(vDX.shape, (n_time, n_batch, n_in))
  assert_equal(vDW.shape, (n_in + n_cells, n_cells * 4))
  assert_equal(vDb.shape, (n_cells * 4,))
  assert_equal(vDh.shape, (n_batch, n_cells))
  assert_equal(vDc.shape, (n_batch, n_cells))
  vDh1 = vDY
  vgc = numpy.tanh(vc)
  vDoutGate_in = (1. - vgates) * vgates * vgc * vDh1
  vDc2 = (1. - vgc * vgc) * vgates * vDh1 + vDd
  vDcellIn_in = (1. - vcellIn * vcellIn) * vgates * vDc2
  vDinpGate_in = (1. - vgates) * vgates * vcellIn * vDc2
  vDfgtGate_in = (1. - vgates) * vgates * vc0 * vDc2
  vDintern = numpy.array([vDcellIn_in, vDinpGate_in, vDfgtGate_in, vDoutGate_in])
  vDb_ = vDintern
  assert_equal(vDb_.shape, vDb.shape)
  assert_allclose(vDb, vDb_)
  vDW_ = numpy.array([vX * vDintern, vy0 * vDintern])
  assert_equal(vDW_.shape, vDW.shape)
  assert_allclose(vDW, vDW_)
  vDx1 = numpy.sum(vDintern)
  assert_allclose(vDX, vDx1)
  vDh0 = numpy.sum(vDintern)
  assert_allclose(vDh, vDh0)
  vDc0 = vgates * vDc2
  assert_allclose(vDc, vDc0)


@unittest.skipIf(not is_gpu_available(), "no gpu on this system")
def test_FastBaumWelch():
  print("Make op...")
  op = make_fast_baum_welch_op(compiler_opts=dict(verbose=True))  # will be cached, used inside :func:`fast_baum_welch`
  print("Op:", op)
  n_batch = 3
  seq_len = 5
  n_classes = 10
  from Fsa import FastBwFsaShared
  fsa = FastBwFsaShared()
  fsa.add_inf_loop(state_idx=0, num_emission_labels=n_classes)
  fast_bw_fsa = fsa.get_fast_bw_fsa(n_batch=n_batch)
  edges = tf.constant(fast_bw_fsa.edges, dtype=tf.int32)
  weights = tf.constant(fast_bw_fsa.weights, dtype=tf.float32)
  start_end_states = tf.constant(fast_bw_fsa.start_end_states, dtype=tf.int32)
  am_scores = tf.constant(numpy.random.normal(size=(seq_len, n_batch, n_classes)), dtype=tf.float32)  # in -log space
  float_idx = tf.ones((seq_len, n_batch), dtype=tf.float32)
  print("Construct call...")
  fwdbwd, obs_scores = fast_baum_welch(
    am_scores=am_scores, float_idx=float_idx,
    edges=edges, weights=weights, start_end_states=start_end_states)
  print("Done.")
  print("Eval:")
  _, score = session.run([fwdbwd, obs_scores])
  print("score:", score)


@unittest.skipIf(not is_gpu_available(), "no gpu on this system")
def test_fast_bw_uniform():
  print("Make op...")
  op = make_fast_baum_welch_op(compiler_opts=dict(verbose=True))  # will be cached, used inside :func:`fast_baum_welch`
  # args: (am_scores, edges, weights, start_end_states, float_idx, state_buffer)
  print("Op:", op)
  n_batch = 3
  seq_len = 7
  n_classes = 5
  from Fsa import FastBwFsaShared
  fsa = FastBwFsaShared()
  for i in range(n_classes):
    fsa.add_edge(i, i + 1, emission_idx=i)  # fwd
    fsa.add_edge(i + 1, i + 1, emission_idx=i)  # loop
  assert n_classes <= seq_len
  fast_bw_fsa = fsa.get_fast_bw_fsa(n_batch=n_batch)
  edges = tf.constant(fast_bw_fsa.edges, dtype=tf.int32)
  weights = tf.constant(fast_bw_fsa.weights, dtype=tf.float32)
  start_end_states = tf.constant(fast_bw_fsa.start_end_states, dtype=tf.int32)
  am_scores = numpy.ones((seq_len, n_batch, n_classes), dtype="float32") * numpy.float32(1.0 / n_classes)
  am_scores = -numpy.log(am_scores)  # in -log space
  am_scores = tf.constant(am_scores, dtype=tf.float32)
  float_idx = tf.ones((seq_len, n_batch), dtype=tf.float32)
  # from TFUtil import sequence_mask_time_major
  # float_idx = tf.cast(sequence_mask_time_major(tf.convert_to_tensor(list(range(seq_len - n_batch + 1, seq_len + 1)))), dtype=tf.float32)
  print("Construct call...")
  fwdbwd, obs_scores = fast_baum_welch(
    am_scores=am_scores, float_idx=float_idx,
    edges=edges, weights=weights, start_end_states=start_end_states)
  print("Done.")
  print("Eval:")
  fwdbwd, score = session.run([fwdbwd, obs_scores])
  print("score:")
  print(repr(score))
  assert_equal(score.shape, (seq_len, n_batch))
  bw = numpy.exp(-fwdbwd)
  print("Baum-Welch soft alignment:")
  print(repr(bw))
  assert_equal(bw.shape, (seq_len, n_batch, n_classes))
  from numpy import array, float32
  if seq_len == n_classes:
    print("Extra check identity...")
    for i in range(n_batch):
      assert_almost_equal(numpy.identity(n_classes), bw[:, i])
  if seq_len == 7 and n_classes == 5:
    print("Extra check ref_align (7,5)...")
    assert_allclose(score, 8.55801582, rtol=1e-5)  # should be the same everywhere
    ref_align = \
      array([[[1., 0., 0., 0., 0.]],
             [[0.33333316, 0.66666663, 0., 0., 0.]],
             [[0.06666669, 0.53333354, 0.40000018, 0., 0.]],
             [[0., 0.20000014, 0.60000014, 0.19999999, 0.]],
             [[0., 0., 0.39999962, 0.53333312, 0.06666663]],
             [[0., 0., 0., 0.66666633, 0.33333316]],
             [[0., 0., 0., 0., 0.99999982]]], dtype=float32)
    assert_equal(ref_align.shape, (seq_len, 1, n_classes))
    ref_align = numpy.tile(ref_align, (1, n_batch, 1))
    assert_equal(ref_align.shape, bw.shape)
    # print("Reference alignment:")
    # print(repr(ref_align))
    print("mean square diff:", numpy.mean(numpy.square(ref_align - bw)))
    print("max square diff:", numpy.max(numpy.square(ref_align - bw)))
    assert_allclose(ref_align, bw, rtol=1e-5)
  print("Done.")


if __name__ == "__main__":
  try:
    better_exchook.install()
    if len(sys.argv) <= 1:
      for k, v in sorted(globals().items()):
        if k.startswith("test_"):
          print("-" * 40)
          print("Executing: %s" % k)
          v()
          print("-" * 40)
    else:
      assert len(sys.argv) >= 2
      for arg in sys.argv[1:]:
        print("Executing: %s" % arg)
        if arg in globals():
          globals()[arg]()  # assume function and execute
        else:
          eval(arg)  # assume Python code and execute
  finally:
    session.close()
    del session
    tf.reset_default_graph()
    import threading
    if len(list(threading.enumerate())) > 1:
      print("Warning, more than one thread at exit:")
      better_exchook.dump_all_thread_tracebacks()
