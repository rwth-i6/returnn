# -*- coding: utf8 -*-

import sys
import os

my_dir = os.path.dirname(os.path.realpath(__file__))
import _setup_test_env  # noqa
from nose.tools import assert_equal, assert_not_equal, assert_raises, assert_true, assert_is
from numpy.testing.utils import assert_almost_equal
from returnn.util.basic import *
import numpy as np
import numpy
import unittest

from returnn.util import better_exchook
better_exchook.replace_traceback_format_tb()


def test_cmd_true():
  r = sys_cmd_out_lines("true")
  assert_equal(r, [])


def test_cmd_false():
  assert_raises(CalledProcessError, lambda: sys_cmd_out_lines("false"))


def test_cmd_stdout():
  r = sys_cmd_out_lines("echo 1; echo 2;")
  assert_equal(r, ["1", "2"])


def test_cmd_stderr():
  r = sys_cmd_out_lines("echo x >&2")
  assert_equal(r, [], "cmd() output should only cover stdout")


def test_hms():
  assert_equal(hms(5), "0:00:05")
  assert_equal(hms(65), "0:01:05")
  assert_equal(hms(65 + 60 * 60), "1:01:05")


def test_hms_fraction():
  assert_equal(hms_fraction(0, decimals=3), "0:00:00.000")
  assert_equal(hms_fraction(5, decimals=3), "0:00:05.000")
  assert_equal(hms_fraction(5.345, decimals=3), "0:00:05.345")
  assert_equal(hms_fraction(65.345, decimals=3), "0:01:05.345")


def test_uniq():
  assert (uniq(np.array([0, 1, 1, 1, 2, 2])) == np.array([0, 1, 2])).all()


def test_slice_pad_zeros():
  assert_equal(list(slice_pad_zeros(np.array([1, 2, 3, 4]), begin=1, end=3)), [2, 3])
  assert_equal(list(slice_pad_zeros(np.array([1, 2, 3, 4]), begin=-2, end=2)), [0, 0, 1, 2])
  assert_equal(list(slice_pad_zeros(np.array([1, 2, 3, 4]), begin=-2, end=6)), [0, 0, 1, 2, 3, 4, 0, 0])
  assert_equal(list(slice_pad_zeros(np.array([1, 2, 3, 4]), begin=2, end=6)), [3, 4, 0, 0])


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
  r1 = a.elem_eq(b, result_with_default=False)
  r2 = a.elem_eq(b, result_with_default=True)
  r2a = a == b
  print(a, b, r1, r2, r2a)
  assert_is(all(r2.values()), r2a)
  assert_is(r1.value, None)
  assert_equal(r1.dict, {'classes': True, 'data': True})
  assert_equal(r1, NumbersDict({'classes': True, 'data': True}))
  assert_is(r2.value, None)
  assert_equal(r2.dict, {"classes": True, "data": True})
  assert_true(r2a)


def test_NumbersDict_eq_2():
  a = NumbersDict(10)
  assert_equal(a, 10)
  assert_not_equal(a, 5)


def test_NumbersDict_mul():
  a = NumbersDict(numbers_dict={"data": 3, "classes": 2}, broadcast_value=1)
  b = a * 2
  assert isinstance(b, NumbersDict)
  assert b.value == 2
  assert_equal(b.dict, {"data": 6, "classes": 4})


def test_NumbersDict_float_div():
  a = NumbersDict(numbers_dict={"data": 3.0, "classes": 2.0}, broadcast_value=1.0)
  b = a / 2.0
  assert isinstance(b, NumbersDict)
  assert_almost_equal(b.value, 0.5)
  assert_equal(list(sorted(b.dict.keys())), ["classes", "data"])
  assert_almost_equal(b.dict["data"], 1.5)
  assert_almost_equal(b.dict["classes"], 1.0)


def test_NumbersDict_int_floordiv():
  a = NumbersDict(numbers_dict={"data": 3, "classes": 2}, broadcast_value=1)
  b = a // 2
  assert isinstance(b, NumbersDict)
  assert_equal(b.value, 0)
  assert_equal(list(sorted(b.dict.keys())), ["classes", "data"])
  assert_equal(b.dict["data"], 1)
  assert_equal(b.dict["classes"], 1)


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
  print(kwargs)
  assert_equal(sorted(kwargs), ["a", "b", "c"])


def test_terminal_size():
  terminal_size()


def test_try_get_caller_name():
  def sub():
    return try_get_caller_name()
  assert_equal(sub(), "test_try_get_caller_name")


def test_camel_case_to_snake_case():
  assert_equal(camel_case_to_snake_case("CamelCaseOp"), "camel_case_op")


def test_NativeCodeCompiler():
  native = NativeCodeCompiler(
    base_name="test_NativeCodeCompiler", code_version=1, code="""
    static int magic = 13;

    extern "C" void set_magic(int i) { magic = i; }
    extern "C" int get_magic() { return magic; }
    """)
  import ctypes
  lib = native.load_lib_ctypes()
  lib.set_magic.restype = None  # void
  lib.set_magic.argtypes = (ctypes.c_int,)
  lib.get_magic.restype = ctypes.c_int
  lib.get_magic.argtypes = ()

  assert_equal(lib.get_magic(), 13)
  lib.set_magic(42)
  assert_equal(lib.get_magic(), 42)


def test_Stats():
  rnd = numpy.random.RandomState(42)
  m = rnd.uniform(-2., 10., (1000, 3))
  mean_ref = numpy.mean(m, axis=0)
  var_ref = numpy.var(m, axis=0)
  std_dev_ref = numpy.std(m, axis=0)
  print("ref mean/var/stddev:", mean_ref, var_ref, std_dev_ref)
  assert_almost_equal(numpy.sqrt(var_ref), std_dev_ref)
  stats = Stats()
  t = 0
  while t < len(m):
    s = int(rnd.uniform(10, 100))
    m_sub = m[t:t + s]
    print("sub seq from t=%i, len=%i" % (t, len(m_sub)))
    stats.collect(m_sub)
    t += s
  mean = stats.get_mean()
  std_dev = stats.get_std_dev()
  print("mean/stddev:", mean, std_dev)
  assert_almost_equal(mean, mean_ref)
  assert_almost_equal(std_dev, std_dev_ref)
  m -= mean[None, :]
  m /= std_dev[None, :]
  stats2 = Stats()
  t = 0
  while t < len(m):
    s = int(rnd.uniform(10, 100))
    m_sub = m[t:t + s]
    stats2.collect(m_sub)
    t += s
  mean0 = stats2.get_mean()
  stddev1 = stats2.get_std_dev()
  print("normalized mean/stddev:", mean0, stddev1)
  assert_almost_equal(mean0, 0.)
  assert_almost_equal(stddev1, 1.)


def test_deepcopy():
  deepcopy({"a": 1, "b": 2, "c": [3, {}, (), [42, True]]})


def test_deepcopy_mod():
  o = deepcopy({"a": 1, "b": 2, "c": sys})
  assert isinstance(o, dict)
  assert o["c"] is sys


def test_deepcopy_config():
  from returnn.config import Config
  config = Config()
  deepcopy(config)


def test_deepcopy_builtins():
  user_ns = {}
  custom_exec("", "<source.py>", user_ns, user_ns)
  print(user_ns)
  assert "__builtins__" in user_ns
  assert isinstance(user_ns["__builtins__"], dict)
  o = deepcopy(user_ns)
  assert o["__builtins__"] is user_ns["__builtins__"]  # no copy, directly reference this module dict


def test_get_func_kwargs():
  def dummy_func(net, var, update_ops):
    pass

  assert_equal(list(getargspec(dummy_func).args), ["net", "var", "update_ops"])


def test_simple_obj_repr():
  class X:
    def __init__(self, a, b=13):
      self.a = a
      self.b = b

    __repr__ = simple_obj_repr

  x = X(a=42)
  x_repr = repr(x)

  assert_equal(x_repr, "X(a=42, b=13)")


def test_dict_diff_str():
  assert_equal(dict_diff_str({"a": 1, "b": 2}, {"a": 1, "b": 3}), "item 'b' differ. self: 2, other: 3")


def test_dict_diff_str_non_str_key():
  assert_equal(dict_diff_str({1: 1, 2: 2}, {1: 1, 2: 3}), "item 2 differ. self: 2, other: 3")


@unittest.skipIf(PY3, "only for Python 2")
def test_py2_utf8_str_to_unicode():
  assert_equal(py2_utf8_str_to_unicode("a"), "a")
  assert_is(type(py2_utf8_str_to_unicode("a")), str)
  assert_equal(py2_utf8_str_to_unicode("äöü"), u"äöü")
  assert_is(type(py2_utf8_str_to_unicode("äöü")), unicode)


def test_CollectionReadCheckCovered():
  x = CollectionReadCheckCovered.from_bool_or_dict(True)
  assert x and x.truth_value
  x = CollectionReadCheckCovered.from_bool_or_dict(False)
  assert not x or not x.truth_value
  x = CollectionReadCheckCovered.from_bool_or_dict({})
  assert not x or not x.truth_value
  x = CollectionReadCheckCovered.from_bool_or_dict({"a": "b"})
  assert x and x.truth_value


def test_version():
  import returnn
  print("RETURNN version:", returnn.__version__, returnn.__long_version__)
  assert "1.0.0" not in returnn.__version__ and "unknown" not in returnn.__long_version__


def test_logging():
  # There is no real test. But you can interactively test this.
  import logging
  logging.getLogger("returnn").debug("Hello from returnn logger.")
  logging.getLogger("returnn.import_").debug("Hello from returnn.import_ logger.")


def test_import_():
  from returnn.import_ import import_
  mod = import_("github.com/rwth-i6/returnn-experiments", "common/test.py", "20210302-01094bef2761")
  print("Loaded mod %s, name %s, file %s" % (mod, mod.__name__, mod.__file__))
  assert_equal(mod.hello(), "hello world")


def test_import_root_repo_mod():
  from returnn.import_ import import_
  mod = import_("github.com/rwth-i6/returnn_common", "test.py", "20210602-1bc6822")
  print("Loaded mod %s, name %s, file %s" % (mod, mod.__name__, mod.__file__))
  assert_equal(mod.hello(), "hello world")


def test_import_root_repo_pkg():
  from returnn.import_ import import_
  mod = import_("github.com/rwth-i6/returnn_common", ".", "20210602-1bc6822")
  print("Loaded mod %s, name %s, file %s" % (mod, mod.__name__, mod.__file__))
  from returnn_import.github_com.rwth_i6.returnn_common.v20210602162042_1bc6822b2fd1 import test
  assert_equal(test.hello(), "hello world")


def test_import_root_repo_sub_mod():
  from returnn.import_ import import_
  mod = import_("github.com/rwth-i6/returnn_common", "test/hello.py", "20210603-3752d77")
  print("Loaded mod %s, name %s, file %s" % (mod, mod.__name__, mod.__file__))
  assert_equal(mod.hello(), "hello world")


def test_import_pkg_py_import():
  from returnn.import_ import import_
  mod = import_("github.com/rwth-i6/returnn-experiments", "common", "20210302-01094bef2761")
  print("Loaded mod %s, name %s, file %s" % (mod, mod.__name__, mod.__file__))
  # noinspection PyUnresolvedReferences
  from returnn_import.github_com.rwth_i6.returnn_experiments.v20210302133012_01094bef2761 import common
  assert common is mod


def test_import_wrong_date():
  from returnn.import_ import import_
  from returnn.import_.common import InvalidVersion
  try:
    import_("github.com/rwth-i6/returnn-experiments", "common/test.py", "20210301-01094bef2761")
  except InvalidVersion as exc:
    print("got expected exception:", exc)
  else:
    raise Exception("We expected an invalid version exception but got nothing.")


def test_import_wrong_pkg_py_import():
  from pprint import pprint
  from returnn.import_ import import_
  from returnn.import_.common import _registered_modules, MissingExplicitImport
  # Use some other commit here which is not used by the other tests, to not mess up.
  mod = import_("github.com/rwth-i6/returnn-experiments", "common", "20210302-10beb9d1c57e")
  print("Loaded mod %s, name %s, file %s" % (mod, mod.__name__, mod.__file__))
  print("Registered modules:")
  pprint(_registered_modules)

  # Mod name, without "common".
  mod_name = "returnn_import.github_com.rwth_i6.returnn_experiments.v20210302153450_10beb9d1c57e"
  assert mod_name in _registered_modules
  assert mod_name in sys.modules
  # Remove from both to force a reload.
  del _registered_modules[mod_name]
  del sys.modules[mod_name]

  try:
    import returnn_import.github_com.rwth_i6.returnn_experiments.v20210302153450_10beb9d1c57e
  except MissingExplicitImport as exc:  # but *not* a normal ImportError
    print("Got expected exception:", exc)
  else:
    raise Exception("We expected an import error exception but got nothing.")


def test_literal_py_to_pickle():
  import ast
  from returnn.util import literal_py_to_pickle

  def check(s):
    print("check:", s)
    a = ast.literal_eval(s)
    b = literal_py_to_pickle.literal_eval(s)
    assert_equal(a, b)

  checks = [
    "0", "1",
    "0.0", "1.23",
    '""', '"abc"', "''", "'abc'",
    '"abc\\n\\x00\\x01\\"\'abc"',
    "'hällö'",
    "'k\\u200eý'",
    "[]", "[1]", "[1,2,3]",
    "{}", "{'a': 'b', 1: 2}",
    "{1}", "{1,2,3}",
    "None", "{'a': None, 'b':1, 'c':None, 'd':'d'}",
  ]
  for s in checks:
    check(s)


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
