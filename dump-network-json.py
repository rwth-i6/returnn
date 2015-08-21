#!/usr/bin/env python

import sys
import argparse
import rnn
import json
from Log import log
from Pretrain import pretrainFromConfig
from Network import LayerNetwork


def init(configFilename, commandLineOptions):
  rnn.initBetterExchook()
  rnn.initConfig(configFilename, commandLineOptions)
  global config
  config = rnn.config
  config.set("log", [])
  rnn.initLog()
  print >> log.v3, "CRNN dump-dataset starting up."
  rnn.initConfigJsonNetwork()


def main(argv):
  argparser = argparse.ArgumentParser(description='Dump network as JSON.')
  argparser.add_argument('crnn_config_file')
  argparser.add_argument('--epoch', default=1, type=int)
  argparser.add_argument('--out', default="/dev/stdout")
  args = argparser.parse_args(argv[1:])
  init(configFilename=args.crnn_config_file, commandLineOptions=[])

  pretrain = pretrainFromConfig(config)
  if pretrain:
    network = pretrain.get_network_for_epoch(args.epoch)
  else:
    network = LayerNetwork.from_config_topology(config)

  json_data = network.to_json_content()
  f = open(args.out, 'w')
  print >> f, json.dumps(json_data, indent=2, sort_keys=True)
  f.close()

  rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
