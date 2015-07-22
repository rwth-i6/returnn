#!/usr/bin/env python

import sys
import argparse
import rnn
from Log import log


def init(configFilename, commandLineOptions):
  rnn.initBetterExchook()
  rnn.initConfig(configFilename, commandLineOptions)
  global config
  config = rnn.config
  rnn.initLog()
  print >> log.v3, "CRNN dump-dataset starting up."
  rnn.initConfigJson()
  # Load network just like in std engine init to get the exact epoch-specific network.
  rnn.initEngine(devices=[])
  # This will init the network.
  rnn.engine.init_train_from_config(config, train_data=None)


def main(argv):
  argparser = argparse.ArgumentParser(description='Dump network as JSON.')
  argparser.add_argument('crnn_config_file')
  argparser.add_argument('--out', default="/dev/stdout")
  args = argparser.parse_args(argv[1:])
  init(configFilename=args.crnn_config_file, commandLineOptions=[])
  rnn.engine.network_dump_json(args.out)
  rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
