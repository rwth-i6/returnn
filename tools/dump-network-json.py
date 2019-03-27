#!/usr/bin/env python

from __future__ import print_function

import os
import sys

my_dir = os.path.dirname(os.path.abspath(__file__))
returnn_dir = os.path.dirname(my_dir)
sys.path.append(returnn_dir)

import argparse
import rnn
import json
from Log import log
from Pretrain import pretrain_from_config
from Network import LayerNetwork


def init(configFilename, commandLineOptions):
  rnn.init_better_exchook()
  rnn.init_config(configFilename, commandLineOptions)
  global config
  config = rnn.config
  config.set("log", [])
  rnn.init_log()
  print("CRNN dump-dataset starting up.", file=log.v3)
  rnn.init_config_json_network()


def main(argv):
  argparser = argparse.ArgumentParser(description='Dump network as JSON.')
  argparser.add_argument('crnn_config_file')
  argparser.add_argument('--epoch', default=1, type=int)
  argparser.add_argument('--out', default="/dev/stdout")
  args = argparser.parse_args(argv[1:])
  init(configFilename=args.crnn_config_file, commandLineOptions=[])

  pretrain = pretrain_from_config(config)
  if pretrain:
    network = pretrain.get_network_for_epoch(args.epoch)
  else:
    network = LayerNetwork.from_config_topology(config)

  json_data = network.to_json_content()
  f = open(args.out, 'w')
  print(json.dumps(json_data, indent=2, sort_keys=True), file=f)
  f.close()

  rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
