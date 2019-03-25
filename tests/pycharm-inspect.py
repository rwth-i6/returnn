#!/usr/bin/env python3

import os
import sys
import subprocess
import tempfile
from glob import glob

my_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(my_dir)
pycharm_dir = base_dir + "/extern_private/pycharm"
sys.path.insert(0, base_dir)
os.chdir(base_dir)

import better_exchook
better_exchook.install()

# travis_fold: https://github.com/travis-ci/travis-ci/issues/1065
print("travis_fold:start:script.install")
subprocess.check_call([my_dir + "/install_pycharm.sh"], cwd=os.path.dirname(pycharm_dir))
assert os.path.isdir(pycharm_dir)
tmp_dir = tempfile.mkdtemp()
print("travis_fold:end:script.install")

print("travis_fold:start:script.inspect")
subprocess.check_call([
  "%s/bin/inspect.sh" % pycharm_dir,
  base_dir,
  "%s/PyCharm-inspection-profile.xml" % my_dir,
  tmp_dir,
  "-v2"])
print("travis_fold:end:script.inspect")

fs = list(glob(tmp_dir + "/*.xml"))
assert fs

for fn in fs:
  print("travis_fold:start:inspect.%s" % os.path.basename(fn))
  print(open(fn).read())
  print("travis_fold:end:inspect.%s" % os.path.basename(fn))
