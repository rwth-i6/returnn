
from __future__ import print_function

import _setup_test_env  # noqa
from subprocess import Popen, PIPE, STDOUT, CalledProcessError
import re
import os
import sys
from glob import glob
from pprint import pprint
import unittest
from nose.tools import assert_less, assert_in, assert_equal
from returnn.util import better_exchook


py = sys.executable
PY3 = sys.version_info[0] >= 3


def build_env():
  env_update = os.environ.copy()
  return env_update


def run(args, input=None):
  args = list(args)
  print("run:", args)
  # RETURNN by default outputs on stderr, so just merge both together
  p = Popen(args, stdout=PIPE, stderr=STDOUT, stdin=PIPE, env=build_env())
  out, _ = p.communicate(input=input)
  print("Return code is %i" % p.returncode)
  print("std out/err:\n---\n%s\n---\n" % out.decode("utf8"))
  if p.returncode != 0:
    raise CalledProcessError(cmd=args, returncode=p.returncode, output=out)
  return out.decode("utf8")


def filter_out(ls):
  """
  :param list[str] ls:
  :rtype: list[str]
  """
  if not isinstance(ls, list):
    ls = list(ls)
  res = []
  i = 0
  while i < len(ls):
    s = ls[i]
    if "tensorflow/core/" in s:  # some TF warnings
      i += 1
      continue
    # RuntimeWarning|FutureWarning are warnings and they include the code-line in the next output line
    if i + 1 < len(ls) and ls[i + 1].startswith("  "):
      if re.match(".*:\\d+: RuntimeWarning: numpy.*", s) or re.match(".*:\\d+: FutureWarning: .*", s):
        i += 2
        continue
    res.append(ls[i])
    i += 1
  return res


def count_start_with(ls, s):
  """
  :param list[str] ls:
  :param str s:
  :rtype: int
  """
  c = 0
  for l in ls:
    if l.startswith(s):
      c += 1
  return c


def test_filter_out():
  s = """
/home/travis/virtualenv/python2.7.14/lib/python2.7/site-packages/scipy/special/__init__.py:640: RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
  from ._ufuncs import *
/home/travis/virtualenv/python2.7.14/lib/python2.7/site-packages/h5py/_hl/group.py:22: RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
  from .. import h5g, h5i, h5o, h5r, h5t, h5l, h5p
RETURNN starting up, version 20180724.141845--git-7865d01, date/time 2018-07-24-13-11-47 (UTC+0000), pid 2196, cwd /home/travis/build/rwth-i6/returnn, Python /home/travis/virtualenv/python2.7.14/bin/python
faulthandler import error. No module named faulthandler
Theano: 0.9.0 (<site-package> in /home/travis/virtualenv/python2.7.14/lib/python2.7/site-packages/theano)
Task: No-operation
elapsed: 0:00:00.0001
"""  # noqa
  ls = filter(None, s.splitlines())
  ls = filter_out(ls)
  pprint(ls)
  assert_equal(len(ls), 5)


def test_returnn_startup():
  out = run([py, "rnn.py", "-x", "nop"])
  ls = out.splitlines()
  ls = filter_out(ls)
  assert 3 <= len(ls) <= 10  # not fixed because might change
  assert_equal(count_start_with(ls, "RETURNN starting up, version "), 1)
  assert_equal(count_start_with(ls, "Theano: "), 1)
  assert_in("Task: No-operation", ls)


def test_returnn_startup_verbose():
  out = run([py, "rnn.py", "-x", "nop", "++log_verbosity", "5"])
  ls = out.splitlines()
  ls = filter_out(ls)
  assert 3 <= len(ls) <= 10  # not fixed because might change
  assert_equal(count_start_with(ls, "RETURNN starting up, version "), 1)
  assert_equal(count_start_with(ls, "RETURNN command line options: "), 1)
  assert_equal(count_start_with(ls, "Theano: "), 1)
  assert_in("Task: No-operation", ls)
  assert_in("Quitting", ls)


def test_returnn_tf_startup():
  out = run([py, "rnn.py", "-x", "nop", "++use_tensorflow", "1", "++log_verbosity", "5"])
  ls = out.splitlines()
  ls = filter_out(ls)
  assert 3 <= len(ls) <= 100, "output:\n%s\n\nNum lines: %i" % ("\n".join(ls), len(ls))  # might change
  assert_equal(count_start_with(ls, "RETURNN starting up, version "), 1)
  assert_equal(count_start_with(ls, "TensorFlow: "), 1)
  assert_in("Task: No-operation", ls)
  assert_in("Quitting", ls)


def test_simple_log():
  code = """
from __future__ import print_function
print("hello stdout 1")
from returnn.log import log
log.initialize(verbosity=[], logs=[], formatter=[])
print("hello stdout 2")
print("hello log 1", file=log.v3)
print("hello log 2", file=log.v3)
  """
  out = run([py], input=code.encode("utf8"))
  assert_equal(out.splitlines(), [
    "hello stdout 1",
    "hello stdout 2",
    "hello log 1",
    "hello log 2"])


def test_StreamIO():
  try:
    import StringIO
  except ImportError:
    import io as StringIO
  buf = StringIO.StringIO()
  assert_equal(buf.getvalue(), "")
  print("buf: %r" % buf.getvalue())

  buf.write("hello")
  print("buf: %r" % buf.getvalue())
  assert_equal(buf.getvalue(), "hello")
  buf.truncate(0)  # should not change the position, thus the buffer is empty but position is len("hello")
  print("buf: %r" % buf.getvalue())
  assert_equal(buf.getvalue(), "")

  buf.write("hello")
  print("buf: %r" % buf.getvalue())
  if PY3:
    # This behavior is not correct in Python 2.7. https://bugs.python.org/issue30250
    assert_equal(buf.getvalue(), "\x00\x00\x00\x00\x00hello")  # zero-filled
  buf.truncate(0)
  buf.seek(0)
  print("buf: %r" % buf.getvalue())
  assert_equal(buf.getvalue(), "")

  buf.write("hello")
  print("buf: %r" % buf.getvalue())
  assert_equal(buf.getvalue(), "hello")
  buf.truncate(0)
  buf.seek(0)
  print("buf: %r" % buf.getvalue())
  assert_equal(buf.getvalue(), "")


if __name__ == "__main__":
  better_exchook.install()
  if len(sys.argv) <= 1:
    for k, v in sorted(globals().items()):
      if k.startswith("test_"):
        print("-" * 40)
        print("Executing: %s" % k)
        try:
          v()
        except unittest.SkipTest as exc:
          print("SkipTest:", exc)
        print("-" * 40)
    print("Finished all tests.")
  else:
    assert len(sys.argv) >= 2
    for arg in sys.argv[1:]:
      print("Executing: %s" % arg)
      if arg in globals():
        globals()[arg]()  # assume function and execute
      else:
        eval(arg)  # assume Python code and execute
  #  better_exchook.dump_all_thread_tracebacks()
