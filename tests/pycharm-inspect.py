#!/usr/bin/env python3

import os
import sys
import subprocess
import tempfile
from glob import glob

my_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(my_dir)
pycharm_dir = "%s/pycharm" % tempfile.mkdtemp()
sys.path.insert(0, base_dir)
os.chdir(base_dir)

import better_exchook
better_exchook.install()

# travis_fold: https://github.com/travis-ci/travis-ci/issues/1065
print("travis_fold:start:script.install")
subprocess.check_call([my_dir + "/install_pycharm.sh"], cwd=os.path.dirname(pycharm_dir))
assert os.path.isdir(pycharm_dir)
out_tmp_dir = tempfile.mkdtemp()
print("travis_fold:end:script.install")

# New clean source dir, where we symlink only the relevant src files.
print("travis_fold:start:script.prepare")
src_tmp_dir = tempfile.mkdtemp()
for fn in sorted(glob(base_dir + "/*.py")):
  os.symlink(fn, "%s/%s" % (src_tmp_dir, os.path.basename(fn)))
subprocess.check_call(["ls", src_tmp_dir], cwd=os.path.dirname(pycharm_dir))
print("travis_fold:end:script.prepare")

print("travis_fold:start:script.inspect")
subprocess.check_call([
  "%s/bin/inspect.sh" % pycharm_dir,
  src_tmp_dir,
  "%s/PyCharm-inspection-profile.xml" % my_dir,
  out_tmp_dir,
  "-v2"])
print("travis_fold:end:script.inspect")

fs = list(glob(out_tmp_dir + "/*.xml"))
assert fs

for fn in fs:
  print("travis_fold:start:inspect.%s" % os.path.basename(fn))
  print("File %s:" % os.path.basename(fn))
  print(open(fn).read())
  print("travis_fold:end:inspect.%s" % os.path.basename(fn))
