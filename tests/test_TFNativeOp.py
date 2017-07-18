
# start test like this:  nosetests-2.7  tests/test_TFNativeOp.py  --nologcapture


import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import sys
sys.path += ["."]  # Python 3 hack
from TFNativeOp import *
from TFUtil import is_gpu_available
import Util
import unittest
from nose.tools import assert_equal, assert_is_instance
import numpy
import numpy.testing
import os
import better_exchook
better_exchook.replace_traceback_format_tb()


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
  cell = NativeLstmCell(n_hidden)
  inputs = tf.zeros([n_time, n_batch, n_hidden * 4])
  index = tf.ones([n_time, n_batch])
  outputs, final_state = cell(inputs, index)


@unittest.skipIf(not is_gpu_available(), "no gpu on this system")
def test_FastBaumWelch():
  n_batch = 3
  seq_len = 5
  num_emission_labels = 10
  from Fsa import FastBwFsaShared
  fsa = FastBwFsaShared()
  fsa.add_inf_loop(state_idx=0, num_emission_labels=num_emission_labels)
  fast_bw_fsa = fsa.get_fast_bw_fsa(n_batch=n_batch)
  edges = tf.constant(fast_bw_fsa.edges, dtype=tf.int32)
  weights = tf.constant(fast_bw_fsa.weights, dtype=tf.float32)
  start_end_states = tf.constant(fast_bw_fsa.start_end_states, dtype=tf.int32)
  am_scores = tf.constant(numpy.random.normal(size=(seq_len, n_batch)), dtype=tf.float32)  # in -log space
  float_idx = tf.ones((seq_len, n_batch), dtype=tf.float32)
  fast_baum_welch(
    am_scores=am_scores, float_idx=float_idx,
    edges=edges, weights=weights, start_end_states=start_end_states)
