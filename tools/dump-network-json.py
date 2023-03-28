#!/usr/bin/env python

"""
Dumps the network topology as JSON on stdout.
"""

from __future__ import annotations

import sys
import argparse
import json
import typing

import _setup_returnn_env  # noqa
import returnn.__main__ as rnn
from returnn.log import log
from returnn.pretrain import pretrain_from_config
from returnn.config import network_json_from_config


config = None  # type: typing.Optional["returnn.config.Config"]


def init(config_filename, command_line_options):
    """
    :param str config_filename:
    :param list[str] command_line_options:
    """
    rnn.init_better_exchook()
    rnn.init_config(config_filename, command_line_options)
    global config
    config = rnn.config
    config.set("log", [])
    rnn.init_log()
    print("RETURNN dump-dataset starting up.", file=log.v3)


def main(argv):
    """
    Main entry.
    """
    arg_parser = argparse.ArgumentParser(description="Dump network as JSON.")
    arg_parser.add_argument("returnn_config_file")
    arg_parser.add_argument("--epoch", default=1, type=int)
    arg_parser.add_argument("--out", default="/dev/stdout")
    args = arg_parser.parse_args(argv[1:])
    init(config_filename=args.returnn_config_file, command_line_options=[])

    pretrain = pretrain_from_config(config)
    if pretrain:
        json_data = pretrain.get_network_json_for_epoch(args.epoch)
    else:
        json_data = network_json_from_config(config)

    f = open(args.out, "w")
    print(json.dumps(json_data, indent=2, sort_keys=True), file=f)
    f.close()

    rnn.finalize()


if __name__ == "__main__":
    main(sys.argv)
