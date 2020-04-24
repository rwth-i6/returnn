#!/usr/bin/env python3

from __future__ import print_function

import os
import sys
from argparse import ArgumentParser

my_dir = os.path.dirname(os.path.abspath(__file__))
returnn_dir = os.path.dirname(my_dir)
sys.path.insert(0, returnn_dir)

# noinspection PyProtectedMember
from LmDataset import _iter_bliss


def main():
  parser = ArgumentParser(description="dump orth from Bliss XML file as-is")
  parser.add_argument("xml")
  args = parser.parse_args()
  corpus_filename = args.xml

  def callback(orth):
    """
    :param str orth:
    """
    print(orth)

  _iter_bliss(filename=corpus_filename, callback=callback)


if __name__ == '__main__':
  import better_exchook
  better_exchook.install()
  main()
