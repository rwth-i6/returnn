
from __future__ import print_function

import sys
sys.path += ["."]  # Python 3 hack

from subprocess import Popen, PIPE, STDOUT, CalledProcessError
import re
import os
import sys
from glob import glob
from nose.tools import assert_less, assert_in, assert_equal
import better_exchook
better_exchook.replace_traceback_format_tb()


py = sys.executable
PY3 = sys.version_info[0] >= 3


def build_env():
  env_update = os.environ.copy()
  return env_update


def run(args, input=None):
  args = list(args)
  print("run:", args)
  # crnn by default outputs on stderr, so just merge both together
  p = Popen(args, stdout=PIPE, stderr=STDOUT, stdin=PIPE, env=build_env())
  out, _ = p.communicate(input=input)
  print("Return code is %i" % p.returncode)
  print("std out/err:\n---\n%s\n---\n" % out.decode("utf8"))
  if p.returncode != 0:
    raise CalledProcessError(cmd=args, returncode=p.returncode, output=out)
  return out.decode("utf8")


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


def test_returnn_startup():
  out = run([py, "rnn.py", "-x", "nop"])
  ls = out.splitlines()
  assert 3 <= len(ls) <= 10  # not fixed because might change
  assert_equal(count_start_with(ls, "RETURNN starting up, version "), 1)
  assert_equal(count_start_with(ls, "Theano: "), 1)
  assert_in("Task: No-operation", ls)


def test_returnn_startup_verbose():
  out = run([py, "rnn.py", "-x", "nop", "++log_verbosity", "5"])
  ls = out.splitlines()
  assert 3 <= len(ls) <= 10  # not fixed because might change
  assert_equal(count_start_with(ls, "RETURNN starting up, version "), 1)
  assert_equal(count_start_with(ls, "RETURNN command line options: "), 1)
  assert_equal(count_start_with(ls, "Theano: "), 1)
  assert_in("Task: No-operation", ls)
  assert_in("Quitting", ls)


def test_returnn_tf_startup():
  out = run([py, "rnn.py", "-x", "nop", "++use_tensorflow", "1", "++log_verbosity", "5"])
  ls = out.splitlines()
  ls = [l for l in ls if "tensorflow/core/" not in l]  # filter out TF warnings
  assert 3 <= len(ls) <= 40, "\n".join(ls)  # not fixed because might change
  assert_equal(count_start_with(ls, "RETURNN starting up, version "), 1)
  assert_equal(count_start_with(ls, "TensorFlow: "), 1)
  assert_in("Task: No-operation", ls)
  assert_in("Quitting", ls)


def test_simple_log():
  code = """
from __future__ import print_function
print("hello stdout 1")
from Log import log
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
    # This behavior is not correct in Python 2.7. http://bugs.python.org/issue30250
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
  test_StreamIO()
