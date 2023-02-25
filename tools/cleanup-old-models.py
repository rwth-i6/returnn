#!/usr/bin/env python3

"""
This is a wrapper around :func:`returnn.tf.engine.Engine.cleanup_old_models`.
"""

from __future__ import annotations
import sys
import os
import argparse

import _setup_returnn_env  # noqa
from returnn.util import better_exchook
from returnn.log import log
from returnn.__main__ import init, finalize


def main():
    """
    Main entry.
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config")
    arg_parser.add_argument("--cwd", help="will change to this dir")
    arg_parser.add_argument("--model", help="model filenames")
    arg_parser.add_argument("--scores", help="learning_rate_control file, e.g. newbob.data")
    arg_parser.add_argument("--dry_run", action="store_true")
    args = arg_parser.parse_args()
    return_code = 0
    try:
        if args.cwd:
            os.chdir(args.cwd)
        init(
            extra_greeting="Delete old models.",
            config_filename=args.config or None,
            config_updates={"use_tensorflow": True, "need_data": False, "device": "cpu"},
        )
        from returnn.__main__ import engine, config

        if args.model:
            config.set("model", args.model)
        if args.scores:
            config.set("learning_rate_file", args.scores)
        if args.dry_run:
            config.set("dry_run", True)
        engine.cleanup_old_models(ask_for_confirmation=True)

    except KeyboardInterrupt:
        return_code = 1
        print("KeyboardInterrupt", file=getattr(log, "v3", sys.stderr))
        if getattr(log, "verbose", [False] * 6)[5]:
            sys.excepthook(*sys.exc_info())
    finalize()
    if return_code:
        sys.exit(return_code)


if __name__ == "__main__":
    better_exchook.install()
    main()
