import os
import sys
import _setup_test_env  # noqa

from io import BytesIO
from returnn.util.task_system import *
import inspect
import unittest
from nose.tools import assert_equal, assert_is_instance
from returnn.util import better_exchook

better_exchook.replace_traceback_format_tb()


def pickle_dumps(obj):
    sio = BytesIO()
    p = Pickler(sio)
    p.dump(obj)
    return sio.getvalue()


def pickle_loads(s):
    p = Unpickler(BytesIO(s))
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


def test_pickle_unicode_str():
    assert_equal(pickle_loads(pickle_dumps("â")), "â")


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
