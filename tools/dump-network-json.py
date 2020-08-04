#!/usr/bin/env python

from __future__ import print_function

import os
import sys

my_dir = os.path.dirname(os.path.abspath(__file__))
returnn_dir = os.path.dirname(my_dir)
sys.path.insert(0, returnn_dir)

import argparse
import returnn.__main__ as rnn
import json
from returnn.log import log
from returnn.pretrain import pretrain_from_config
from returnn.config import network_json_from_config


def init(config_filename, command_line_options):
  rnn.init_better_exchook()
  rnn.init_config(config_filename, command_line_options)
  global config
  config = rnn.config
  config.set("log", [])
  rnn.init_log()
  print("RETURNN dump-dataset starting up.", file=log.v3)
  rnn.init_config_json_network()


def main(argv):
  argparser = argparse.ArgumentParser(description='Dump network as JSON.')
  argparser.add_argument('returnn_config_file')
  argparser.add_argument('--epoch', default=1, type=int)
  argparser.add_argument('--out', default="/dev/stdout")
  args = argparser.parse_args(argv[1:])
  init(config_filename=args.returnn_config_file, command_line_options=[])

  pretrain = pretrain_from_config(config)
  if pretrain:
    network = pretrain.get_network_for_epoch(args.epoch)
  else:
    network = network_json_from_config(config)

  json_data = network.to_json_content()
  f = open(args.out, 'w')
  print(json.dumps(json_data, indent=2, sort_keys=True), file=f)
  f.close()

  rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
