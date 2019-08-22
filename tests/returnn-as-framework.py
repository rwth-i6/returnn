#!/usr/bin/env python3

"""
This is being used by some tests.
"""

from __future__ import print_function
import sys
import os
import tempfile


_my_dir = os.path.dirname(os.path.abspath(__file__))
_base_dir = os.path.dirname(_my_dir)
assert os.path.exists("%s/rnn.py" % _base_dir)


def setup():
  """
  Setup env/sys.path such that `import returnn` works.
  """
  print("Setup for importing RETURNN as framework/package.")
  tmp_dir = tempfile.mkdtemp()
  print("Temp dir:", tmp_dir)
  os.symlink(_base_dir, "%s/returnn" % tmp_dir)
  sys.path.insert(0, tmp_dir)

  print("Import returnn module/package.")
  # noinspection PyUnresolvedReferences
  import returnn

  print("Setup better_exchook.")
  import returnn.better_exchook
  returnn.better_exchook.install()


def test_TaskSystem_Pickler():
  from returnn.TaskSystem import Pickler
  from returnn.Util import BytesIO
  stream = BytesIO()
  pickler = Pickler(stream)
  obj = {"foo": "bar"}  # some dummy dict
  pickler.dump(obj)


if __name__ == "__main__":
  print("RETURNN as framework/package.")
  setup()
  for arg in sys.argv[1:]:
    print("-" * 20)
    print("Run:", arg)
    print("-" * 20)
    eval(arg)
    print("-" * 20)
    print("Ok.")
    print("-" * 20)
