#!/usr/bin/env python

# This script will emulate a Sprint executable, so that we can use it for SprintDataset.
# This is useful for tests.
# To generate data, we can use the GeneratingDataset code.

import sys
import os

# Add parent dir to Python path so that we can use GeneratingDataset and other CRNN code.
my_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.normpath(my_dir + "/..")
if parent_dir not in sys.path:
  sys.path += [parent_dir]

def main(argv):
  print "DummySprintExec", argv
  sys.exit(1)

if __name__ == "__main__":
  main(sys.argv)
