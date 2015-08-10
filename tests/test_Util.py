
from nose.tools import assert_equal, assert_raises
from Util import *
import numpy as np


def test_cmd_true():
  r = cmd("true")
  assert_equal(r, [])


def test_cmd_false():
  assert_raises(CalledProcessError, lambda: cmd("false"))


def test_uniq():
  assert (uniq(np.array([0, 1, 1, 1, 2, 2])) == np.array([0, 1, 2])).all()
