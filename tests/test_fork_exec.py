
"""
What we want: subprocess.Popen to always work.
Problem: It uses fork+exec internally in subprocess_fork_exec, via _posixsubprocess.fork_exec.
That is a problem because fork can trigger any atfork handlers registered via pthread_atfork,
and those can crash/deadlock in some cases.

https://github.com/tensorflow/tensorflow/issues/13802
https://github.com/xianyi/OpenBLAS/issues/240
https://trac.sagemath.org/ticket/22021
https://bugs.python.org/msg304587

"""

from __future__ import print_function

import os
import sys
import better_exchook
from nose.tools import assert_equal, assert_is_instance
from pprint import pprint

my_dir = os.path.dirname(os.path.abspath(__file__))


class CLib:
  c_file = "%s/demo_fork_exec.c" % my_dir
  so_file = "%s/demo_fork_exec.so" % my_dir

  def __init__(self):
    if not os.path.exists(self.so_file) or os.stat(self.c_file).st_mtime >= os.stat(self.so_file).st_mtime:
      self._compile()
    self._load_lib()

  def _compile(self):
    # see OpCodeCompiler for some reference
    common_opts = ["-shared", "-O0"]
    if sys.platform == "darwin":
      common_opts += ["-undefined", "dynamic_lookup"]
    common_opts += ["-fPIC"]
    common_opts += ["-g"]
    cmd_args = ["gcc"] + common_opts + [self.c_file, "-o", self.so_file]
    from subprocess import Popen, PIPE, STDOUT, CalledProcessError
    print("Compiler call: %s" % " ".join(cmd_args))
    proc = Popen(cmd_args, stdout=PIPE, stderr=STDOUT)
    stdout, stderr = proc.communicate()
    assert stderr is None  # should only have stdout
    if proc.returncode != 0:
      print("Compiling failed.")
      print("Original stdout/stderr:")
      print(stdout.decode("utf8"))
      raise CalledProcessError(returncode=proc.returncode, cmd=cmd_args)
    assert os.path.exists(self.so_file)

  def _load_lib(self):
    import ctypes
    self._lib = ctypes.cdll.LoadLibrary(self.so_file)
    self._lib.register_hello_from_child.restype = None  # void
    self._lib.register_hello_from_child.argtypes = ()
    self._lib.register_hello_from_fork_prepare.restype = None  # void
    self._lib.register_hello_from_fork_prepare.argtypes = ()
    self._lib.set_magic_number.restype = None  # void
    self._lib.set_magic_number.argtypes = (ctypes.c_long,)

  def set_magic_number(self, i):
    self._lib.set_magic_number(i)

  def register_hello_from_child(self):
    self._lib.register_hello_from_child()

  def register_hello_from_fork_prepare(self):
    self._lib.register_hello_from_fork_prepare()


clib = CLib()


def demo_hello_from_fork():
  print("Hello.")
  clib.set_magic_number(3)
  clib.register_hello_from_child()
  clib.register_hello_from_fork_prepare()
  sys.stdout.flush()
  pid = os.fork()
  if pid == 0:
    print("Hello from child after fork.")
    sys.exit()
  print("Hello from parent after fork.")
  os.waitpid(pid, 0)
  print("Bye.")


def demo_start_subprocess():
  print("Hello.")
  clib.set_magic_number(5)
  clib.register_hello_from_child()
  clib.register_hello_from_fork_prepare()
  sys.stdout.flush()
  from subprocess import check_call
  # Right now (2017-10-19), CPython will use subprocess_fork_exec which
  # uses fork+exec, so this will call the atfork handler.
  check_call("echo Hello from subprocess.", shell=True)
  print("Bye.")


def run_demo_check_output(name):
  """
  :param str name: e.g. "demo_hello_from_fork"
  :return: lines of stdout of the demo
  :rtype: list[str]
  """
  from subprocess import check_output
  output = check_output([sys.executable, __file__, name])
  return output.decode("utf8").splitlines()


def filter_demo_output(ls):
  """
  :param list[str] ls:
  :rtype: list[str]
  """
  ls = [l for l in ls if not l.startswith("Executing: ")]
  ls = [l for l in ls if not l.startswith("Compiler call: ")]
  assert ls[0] == "Hello."
  ls = ls[1:]
  assert ls[-1] == "Bye."
  ls = ls[:-1]
  return ls


def test_demo_hello_from_fork():
  ls = run_demo_check_output("demo_hello_from_fork")
  pprint(ls)
  ls = filter_demo_output(ls)
  pprint(ls)
  assert_equal(set(ls), {
    'Hello from child after fork.',
    'Hello from child atfork, magic number 3.',
    'Hello from atfork prepare, magic number 3.',
    'Hello from parent after fork.'})


def test_demo_start_subprocess():
  ls = run_demo_check_output("demo_start_subprocess")
  pprint(ls)
  ls = filter_demo_output(ls)
  pprint(ls)
  assert 'Hello from subprocess.' in ls
  # Maybe this is fixed/changed in some later CPython version.
  import platform
  print("Python impl:", platform.python_implementation())
  print("Python version:", sys.version_info[:3])
  if platform.python_implementation() == "CPython":
    if sys.version_info[0] == 2:
      old = sys.version_info[:3] <= (2, 7, 12)
    else:
      old = sys.version_info[:3] <= (3, 6, 1)
  else:
    old = False
  if old:
    print("atfork handler should have been called.")
    assert 'Hello from child atfork, magic number 5.' in ls
    assert 'Hello from atfork prepare, magic number 5.' in ls
  else:
    print("Not checking for atfork handler output.")


if __name__ == "__main__":
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
