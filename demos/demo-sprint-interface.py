#!/usr/bin/env python3

"""
This is mostly intended to be run as a test.
This also demonstrates how the SprintInterface is being used by Sprint (RASR).
"""

import os
import tempfile
import sys


_my_dir = os.path.dirname(os.path.abspath(__file__))
_base_dir = os.path.dirname(_my_dir)
assert os.path.exists("%s/rnn.py" % _base_dir)


def main():
  tmp_dir = tempfile.mkdtemp()
  os.symlink(_base_dir, "%s/returnn" % tmp_dir)
  config_fn = "%s/returnn.config" % tmp_dir
  with open(config_fn, "w") as f:
    f.write("\n")
  sys.path.insert(0, tmp_dir)
  import returnn.SprintInterface as SprintInterface
  SprintInterface.init(
    inputDim=50, outputDim=4501, cudaEnabled=0, targetMode='forward-only',
    config='epoch:15,action:forward,configfile:%s' % config_fn)


if __name__ == '__main__':
  main()
