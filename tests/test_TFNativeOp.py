
# start test like this:  nosetests-2.7  tests/test_TFNativeOp.py  --nologcapture


import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import sys
sys.path += ["."]  # Python 3 hack
from TFNativeOp import *
import Util
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
  assert False


def test_make_lstm_op():
  try:
    make_lstm_op()
  except tf.errors.NotFoundError:
    dump_info()
    raise


def test_NativeLstmCell():
  n_time = 2
  n_batch = 1
  n_hidden = 3
  cell = NativeLstmCell(n_hidden)
  inputs = tf.zeros([n_time, n_batch, n_hidden * 4])
  index = tf.ones([n_time, n_batch])
  outputs, final_state = cell(inputs, index)
