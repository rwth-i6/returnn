#!/usr/bin/env python3

"""
Loads some pickle file and dumps the Python object on stdout.
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
import pickle

import _setup_returnn_env  # noqa
from returnn.util.basic import better_repr


def main():
    """
    Main entry.
    """
    arg_parser = ArgumentParser()
    arg_parser.add_argument("file")
    args = arg_parser.parse_args()
    try:
        o = pickle.load(open(args.file, "rb"))
        print(better_repr(o))
    except BrokenPipeError:
        print("BrokenPipeError", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    from returnn.util import better_exchook

    better_exchook.install()
    main()
