
from nose.tools import assert_equal, assert_raises
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
  assert_equal(list("hello ") + ["FOO"] + list(" bar "), parse_orthography_into_symbols("hello [FOO] bar "))
