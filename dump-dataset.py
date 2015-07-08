#!/usr/bin/env python

import sys
import rnn
from Log import log
import argparse
import numpy


def dump_dataset(dataset, options):
  """
  :type dataset: Dataset.Dataset
  :param options: argparse.Namespace
  """
  print >> log.v3, "Epoch: %i" % options.epoch
  rnn.train_data.init_seq_order(options.epoch)

  print >> log.v3, "Dump prefix: %r" % options.dump_prefix

  seq_idx = options.startseq
  while dataset.is_less_than_num_seqs(seq_idx) and seq_idx <= options.endseq:
    dataset.load_seqs(seq_idx, seq_idx)
    data = dataset.get_data(seq_idx)
    targets = dataset.get_targets(seq_idx)
    numpy.savetxt(options.dump_prefix + "%i.data.txt" % seq_idx, data)
    numpy.savetxt(options.dump_prefix + "%i.targets.txt" % seq_idx, targets)
    seq_idx += 1

  print >> log.v3, "Done. More seqs which we did not dumped: %s" % dataset.is_less_than_num_seqs(seq_idx)


def init(configFilename, commandLineOptions):
  rnn.initBetterExchook()
  rnn.initThreadJoinHack()
  rnn.initConfig(configFilename, commandLineOptions)
  global config
  config = rnn.config
  rnn.initLog()
  print >> log.v3, "CRNN dump-dataset starting up."
  rnn.initFaulthandler()
  rnn.initConfigJson()
  rnn.initData()
  rnn.printTaskProperties()


def main(argv):
  argparser = argparse.ArgumentParser(description='Dump something from dataset.')
  argparser.add_argument('crnn_config_file')
  argparser.add_argument('--epoch', type=int, default=1)
  argparser.add_argument('--startseq', type=int, default=0, help='start seq idx (inclusive) (default: 0)')
  argparser.add_argument('--endseq', type=int, default=float('inf'), help='end seq idx (inclusive) (default: inf)')
  argparser.add_argument('--dump_prefix', default='/tmp/crnn.dump-dataset.')
  args = argparser.parse_args(argv[1:])
  init(configFilename=args.crnn_config_file, commandLineOptions=[])
  dump_dataset(rnn.train_data, args)
  rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
