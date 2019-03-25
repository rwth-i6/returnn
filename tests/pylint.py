#!/usr/bin/env python3

import os
import sys
import subprocess
from glob import glob

my_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(my_dir)
sys.path.insert(0, base_dir)
os.chdir(base_dir)

import better_exchook
better_exchook.install()

# travis_fold: https://github.com/travis-ci/travis-ci/issues/1065
print("travis_fold:start:script.install")
subprocess.check_call(["pip", "install", "pylint", "better_exchook"])
print("travis_fold:end:script.install")

for fn in sorted(glob(base_dir + "/*.py")):
  print("travis_fold:start:pylint.%s" % os.path.basename(fn))
  proc = subprocess.Popen(["pylint", fn], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  stdout, _ = proc.communicate()
  stdout = stdout.decode("utf8")
  if "EXCEPTION" in stdout and "RecursionError" in stdout:
    # https://github.com/PyCQA/pylint/issues/1452
    # https://github.com/PyCQA/astroid/issues/437
    # Don't print full stdout. It will spam too much.
    print("PyLint issue #1452 triggered. https://github.com/PyCQA/pylint/issues/1452")
  else:
    print(stdout)
  print("Return code:", proc.returncode)
  print("travis_fold:end:pylint.%s" % os.path.basename(fn))
