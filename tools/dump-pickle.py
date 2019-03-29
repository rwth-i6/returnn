#!/usr/bin/env python3

from __future__ import print_function

import os
import sys

my_dir = os.path.dirname(os.path.abspath(__file__))
returnn_dir = os.path.dirname(my_dir)
sys.path.append(returnn_dir)

from argparse import ArgumentParser
import pickle
from Util import better_repr


def main():
 argparser = ArgumentParser()
 argparser.add_argument("file")
 args = argparser.parse_args()
 try:
   o = pickle.load(open(args.file, "rb"))
   print(better_repr(o))
 except BrokenPipeError:
   print("BrokenPipeError", file=sys.stderr)
   sys.exit(1)


if __name__ == "__main__":
  import better_exchook
  better_exchook.install()
  main()

