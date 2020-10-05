
import os
import sys
import _setup_test_env  # noqa
try:
  from StringIO import StringIO
except ImportError:  # Python 3
  from io import BytesIO as StringIO
from returnn.util.task_system import *
import inspect
from nose.tools import assert_equal, assert_is_instance
from returnn.util import better_exchook
better_exchook.replace_traceback_format_tb()


def pickle_dumps(obj):
  sio = StringIO()
  p = Pickler(sio)
  p.dump(obj)
  return sio.getvalue()

def pickle_loads(s):
  p = Unpickler(StringIO(s))
  return p.load()


def test_pickle_anon_new_class():
  # New style class, defined here in this scope, so that we cannot find it in any module.
  class Foo(object):
    a = "class"
    b = "foo"
    def __init__(self):
      self.a = "hello"
    def f(self, a):
      return a

  s = pickle_dumps(Foo)
  Foo2 = pickle_loads(s)
  assert inspect.isclass(Foo2)
  assert Foo is not Foo2  # We get a new class.
  assert Foo2.a == "class"
  assert Foo2.b == "foo"

  inst = Foo2()
  assert inst.a == "hello"
  assert inst.b == "foo"
  assert inst.f(42) == 42


def test_pickle_anon_old_class():
  # Old style class, defined here in this scope, so that we cannot find it in any module.
  class Foo:
    a = "class"
    b = "foo"
    def __init__(self):
      self.a = "hello"
    def f(self, a):
      return a

  s = pickle_dumps(Foo)
  Foo2 = pickle_loads(s)
  assert inspect.isclass(Foo2)
  assert Foo is not Foo2  # We get a new class.
  assert Foo2.a == "class"
  assert Foo2.b == "foo"

  inst = Foo2()
  assert inst.a == "hello"
  assert inst.b == "foo"
  assert inst.f(42) == 42


def test_pickle_inst_anon_class():
  class Foo(object):
    a = "class"
    b = "foo"
    def __init__(self):
      self.a = "hello"
    def f(self, a):
      return a
  s = pickle_dumps(Foo())
  inst = pickle_loads(s)
  assert inst.a == "hello"
  assert inst.b == "foo"
  assert inst.f(42) == 42


class DemoClass:
  def method(self):
    return 42


def test_pickle():
  obj = DemoClass()
  s = pickle_dumps(obj.method)
  inst = pickle_loads(s)
  assert_equal(inst(), 42)


def test_AsyncTask():
  def func(asyncTask):
    """
    :type asyncTask: AsyncTask
    """
    print("Hello Async")
    asyncTask.conn.send("hello c2p")
    assert_equal(asyncTask.conn.recv(), "hello p2c")
  proc = AsyncTask(
    func=func,
    name="AsyncTask proc",
    mustExec=True,
    env_update={})
  assert_equal(proc.conn.recv(), "hello c2p")
  proc.conn.send("hello p2c")
  proc.join()


def test_AsyncTask_chdir():
  os.chdir("/")
  def func(asyncTask):
    """
    :type asyncTask: AsyncTask
    """
    print("Hello Async")
    asyncTask.conn.send("hello c2p")
    assert_equal(asyncTask.conn.recv(), "hello p2c")
  proc = AsyncTask(
    func=func,
    name="AsyncTask proc",
    mustExec=True,
    env_update={})
  assert_equal(proc.conn.recv(), "hello c2p")
  proc.conn.send("hello p2c")
  proc.join()
