#!/usr/bin/env python3

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import better_exchook
better_exchook.install()

from Log import log
from rnn import init, finalize

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--config")
arg_parser.add_argument("--cwd", help="will change to this dir")
arg_parser.add_argument("--model", help="model filenames")
arg_parser.add_argument("--scores", help="learning_rate_control file, e.g. newbob.data")
arg_parser.add_argument("--dry_run", action="store_true")


def main():
  args = arg_parser.parse_args()
  return_code = 0
  try:
    if args.cwd:
      os.chdir(args.cwd)
    init(
      extra_greeting="Delete old models.",
      config_filename=args.config or None,
      config_updates={
        "use_tensorflow": True,
        "need_data": False,
        "device": "cpu"})
    from rnn import engine, config
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
  main()
