
from StringIO import StringIO
from TaskSystem import *
import inspect

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
