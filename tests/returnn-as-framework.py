#!/usr/bin/env python3

"""
This is being used by some tests.
"""

from __future__ import print_function
import sys
import os
import tempfile
import argparse


_my_dir = os.path.dirname(os.path.abspath(__file__))
_base_dir = os.path.dirname(_my_dir)
assert os.path.exists("%s/rnn.py" % _base_dir)


def setup(old_style=False, target_package_name="returnn"):
  """
  Setup env/sys.path such that `import returnn` works.

  :param bool old_style: make it an old-style setup
  :param str target_package_name:
  """
  print("Setup for importing RETURNN as framework/package.")
  tmp_env_path_dir = tempfile.mkdtemp()
  print("Temp dir:", tmp_env_path_dir)
  if old_style:
    print("Old-style setup!")
    src_dir = _base_dir
  else:
    src_dir = "%s/returnn" % _base_dir
  os.symlink(src_dir, "%s/%s" % (tmp_env_path_dir, target_package_name))
  sys.path.insert(0, tmp_env_path_dir)

  print("Import %s module/package." % target_package_name)
  if target_package_name == "returnn":
    # noinspection PyUnresolvedReferences
    import returnn
  else:
    __import__(target_package_name)

  print("Setup better_exchook.")
  if target_package_name == "returnn":
    if old_style:
      # noinspection PyUnresolvedReferences
      from returnn import better_exchook
      better_exchook.install()
    else:
      from returnn.util import better_exchook
      better_exchook.install()
  else:
    # Should always work. But only use for the fallback.
    __import__(target_package_name).better_exchook.install()


def test_TaskSystem_Pickler():
  from returnn.util.task_system import Pickler
  from returnn.util.task_system import BytesIO
  stream = BytesIO()
  pickler = Pickler(stream)
  obj = {"foo": "bar"}  # some dummy dict
  pickler.dump(obj)


def test_old_style_import_crnn_TFUtil():
  """
  This assumes that you use ``--returnn-package-name "crnn"`.
  """
  # noinspection PyUnresolvedReferences
  import crnn.TFUtil
  # noinspection PyUnresolvedReferences
  print("TF:", crnn.TFUtil.tf_version_tuple())


def test_old_style_import_TFUtil():
  # noinspection PyUnresolvedReferences
  import returnn.TFUtil
  # noinspection PyUnresolvedReferences
  print("TF:", returnn.TFUtil.tf_version_tuple())


if __name__ == "__main__":
  print("RETURNN as framework/package.")
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("py_eval_str", nargs="*")
  arg_parser.add_argument("--old-style", action="store_true")
  arg_parser.add_argument("--returnn-package-name", default="returnn")
  args = arg_parser.parse_args()
  setup(old_style=args.old_style, target_package_name=args.returnn_package_name)
  for py_eval_str in args.py_eval_str:
    print("-" * 20)
    print("Run:", py_eval_str)
    print("-" * 20)
    eval(py_eval_str)
    print("-" * 20)
    print("Ok.")
    print("-" * 20)
