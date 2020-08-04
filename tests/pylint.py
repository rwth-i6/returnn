#!/usr/bin/env python3

"""
Runs pylint on the RETURNN source code.
"""

import os
import sys
import subprocess
from glob import glob

my_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(my_dir)
sys.path.insert(0, base_dir)
os.chdir(base_dir)

from returnn.util import better_exchook  # noqa
better_exchook.install()


def setup():
  """
  Some generic setup.
  """
  # travis_fold: https://github.com/travis-ci/travis-ci/issues/1065
  print("travis_fold:start:script.install")
  subprocess.check_call(["pip", "install", "pylint", "better_exchook"])
  print("travis_fold:end:script.install")


def main():
  """
  Main entry point.
  """
  setup()
  from lint_common import ignore_count_for_files
  color = better_exchook.Color()
  num_relevant_files_with_errors = 0
  for filename in sorted(glob(base_dir + "/*.py")):
    base_filename = os.path.basename(filename)
    print("travis_fold:start:pylint.%s" % base_filename)
    proc = subprocess.Popen(["pylint", filename], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, _ = proc.communicate()
    stdout = stdout.decode("utf8")
    file_has_errors = proc.returncode != 0
    print(color.color(
      "File: %s" % base_filename,
      color="black" if (base_filename in ignore_count_for_files or not file_has_errors) else "red"))
    if "EXCEPTION" in stdout and "RecursionError" in stdout:
      # https://github.com/PyCQA/pylint/issues/1452
      # https://github.com/PyCQA/astroid/issues/437
      # Don't print full stdout. It will spam too much.
      print("PyLint issue #1452 triggered. https://github.com/PyCQA/pylint/issues/1452")
    elif base_filename in ignore_count_for_files:
      print(stdout[:1000])
      print("... (ignored further output; file is not relevant)")
    else:
      print(stdout)
    print("Return code:", proc.returncode)
    if file_has_errors:
      if base_filename in ignore_count_for_files:
        print("The inspection reports for this file are currently ignored.")
      else:
        print(color.color("The inspection reports for this file are fatal!", color="red"))
        num_relevant_files_with_errors += 1
    print("travis_fold:end:pylint.%s" % base_filename)
  if num_relevant_files_with_errors:
    print("Num relevant files with errors:", num_relevant_files_with_errors)
    sys.exit(1)


if __name__ == "__main__":
  main()
