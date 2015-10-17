
from nose.tools import assert_equal, assert_raises, assert_true, assert_is
from Util import *
import numpy as np


def test_cmd_true():
  r = cmd("true")
  assert_equal(r, [])


def test_cmd_false():
  assert_raises(CalledProcessError, lambda: cmd("false"))


def test_cmd_stdout():
  r = cmd("echo 1; echo 2;")
  assert_equal(r, ["1", "2"])


def test_cmd_stderr():
  r = cmd("echo x >/dev/stderr")
  assert_equal(r, [], "cmd() output should only cover stdout")


def test_uniq():
  assert (uniq(np.array([0, 1, 1, 1, 2, 2])) == np.array([0, 1, 2])).all()


def test_parse_orthography_into_symbols():
  assert_equal(list("hi"), parse_orthography_into_symbols("hi"))
  assert_equal(list(" hello "), parse_orthography_into_symbols(" hello "))
  assert_equal(list("  "), parse_orthography_into_symbols("  "))
  assert_equal(list("hello ") + ["[FOO]"] + list(" bar "), parse_orthography_into_symbols("hello [FOO] bar "))


def test_parse_orthography():
  assert_equal(list("hi ") + ["[HES]"] + list(" there") + ["[END]"], parse_orthography("hi [HES] there "))


def test_NumbersDict_minus_1():
  a = NumbersDict({'classes': 11, 'data': 11})
  b = NumbersDict(10)
  r = a - b
  print(a, b, r)
  assert_equal(r, NumbersDict(numbers_dict={'classes': 1, 'data': 1}, broadcast_value=-10))

def test_NumbersDict_eq_1():
  a = NumbersDict({'classes': 11, 'data': 11})
  b = NumbersDict(11)
  r1 = a.elem_eq(b)
  r2 = a == b
  print(a, b, r1, r2)
  assert_is(r1.value, None)
  assert_equal(r1.dict, {'classes': True, 'data': True})
  assert_equal(r1, NumbersDict({'classes': True, 'data': True}))
  assert_true(r2)


def test_collect_class_init_kwargs():
  class A(object):
    def __init__(self, a):
      pass
  class B(A):
    def __init__(self, b, **kwargs):
      super(B, self).__init__(**kwargs)
      pass
  class C(B):
    def __init__(self, b, c, **kwargs):
      super(C, self).__init__(**kwargs)
      pass

  kwargs = collect_class_init_kwargs(C)
  print kwargs
  assert_equal(sorted(kwargs), ["a", "b", "c"])
