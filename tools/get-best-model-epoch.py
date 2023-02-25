#!/usr/bin/env python3

"""
Based on the scores / errors in a learning rate file ("newbob" file),
calculates the best epochs.
"""

from __future__ import annotations

import sys
import argparse

# Returnn imports
import _setup_returnn_env  # noqa

from returnn.util import better_exchook
from returnn.log import log
from returnn.config import Config
from returnn.learning_rate_control import LearningRateControl


def main():
    """
    Main entry point.
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", help="RETURNN config")
    arg_parser.add_argument("--learning-rate-file", help="The learning rate file contains scores / errors per epoch.")
    arg_parser.add_argument("--key", help="key to use, e.g. 'dev_error'")
    arg_parser.add_argument("--n", type=int, default=5, help="print best N epochs")
    args = arg_parser.parse_args()

    if bool(args.config) == bool(args.learning_rate_file):
        print("Error: provide either --config or --learning-rate-file")
        arg_parser.print_help()
        sys.exit(1)

    if args.config:
        config = Config()
        config.load_file(args.config)
        lr = LearningRateControl.load_initial_from_config(config)
    elif args.learning_rate_file:
        lr = LearningRateControl(default_learning_rate=1, filename=args.learning_rate_file)  # default lr not relevant
    else:
        assert False, "should not get here with %r" % args

    epochs = sorted(lr.epoch_data.keys())
    if not epochs:
        print("Error: no epochs found")
        sys.exit(1)
    print("Loaded epochs", epochs[0], "..", epochs[-1])

    if args.key:
        key = args.key
        print("Using key %s" % key)
    else:
        last_epoch_with_error_info = None
        for ep in reversed(epochs):
            if lr.epoch_data[ep].error:
                last_epoch_with_error_info = ep
                break
        if last_epoch_with_error_info is None:
            print("Error: no scores/errors found")
            sys.exit(1)
        key = lr.get_error_key(last_epoch_with_error_info)
        print("Using key %s (auto via epoch %i)" % (key, last_epoch_with_error_info))

    epochs_ = []
    missing_epochs = []
    for ep in epochs:
        errors = lr.epoch_data[ep].error
        if key in errors:
            epochs_.append((errors[key], ep))
        else:
            missing_epochs.append(ep)
    if len(epochs_) == len(epochs):
        print("All epochs have the key.")
    else:
        print("Epochs missing the key:", missing_epochs)
    assert epochs_
    epochs_.sort()

    for value, ep in epochs_[: args.n]:
        errors = lr.epoch_data[ep].error
        print(
            ", ".join(
                ["Epoch %i" % ep, "%r %r" % (key, value)] + ["%r %r" % (k, v) for k, v in errors.items() if k != key]
            )
        )


if __name__ == "__main__":
    better_exchook.install()
    log.initialize(verbosity=[5])
    main()
