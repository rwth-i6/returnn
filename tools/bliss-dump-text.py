#!/usr/bin/env python3

"""
Goes through a Sprint Bliss XML, and dumps all the orthography on stdout.
"""

from __future__ import print_function

from argparse import ArgumentParser

import _setup_returnn_env  # noqa

# noinspection PyProtectedMember
from returnn.datasets.lm import _iter_bliss


def main():
  """
  Main entry.
  """
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
  from returnn.util import better_exchook
  better_exchook.install()
  main()
