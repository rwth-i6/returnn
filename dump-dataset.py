#!/usr/bin/env python

from __future__ import print_function

import sys
import rnn
from Log import log
import argparse
import numpy
from better_exchook import pretty_print


def dump_dataset(dataset, options):
  """
  :type dataset: Dataset.Dataset
  :param options: argparse.Namespace
  """
  print("Epoch: %i" % options.epoch, file=log.v3)
  rnn.train_data.init_seq_order(options.epoch)

  if options.type == "numpy":
    print("Dump files: %r*%r" % (options.dump_prefix, options.dump_postfix), file=log.v3)
  elif options.type == "stdout":
    print("Dump to stdout", file=log.v3)
  else:
    raise Exception("unknown dump option type %r" % options.type)

  seq_idx = options.startseq
  if options.endseq < 0:
    options.endseq = float("inf")
  while dataset.is_less_than_num_seqs(seq_idx) and seq_idx <= options.endseq:
    dataset.load_seqs(seq_idx, seq_idx + 1)
    data = dataset.get_data(seq_idx, "data")
    if options.type == "numpy":
      numpy.savetxt("%s%i.data%s" % (options.dump_prefix, seq_idx, options.dump_postfix), data)
    elif options.type == "stdout":
      print("seq %i data:" % seq_idx, pretty_print(data))
    for target in dataset.get_target_list():
      targets = dataset.get_targets(target, seq_idx)
      if options.type == "numpy":
        numpy.savetxt("%s%i.targets.%s%s" % (options.dump_prefix, seq_idx, target, options.dump_postfix), targets, fmt='%i')
      elif options.type == "stdout":
        print("seq %i target %r:" % (seq_idx, target), pretty_print(targets))

    seq_idx += 1

  print("Done. More seqs which we did not dumped: %s" % dataset.is_less_than_num_seqs(seq_idx), file=log.v1)


def init(configFilename, commandLineOptions):
  rnn.initBetterExchook()
  rnn.initThreadJoinHack()
  rnn.initConfig(configFilename, commandLineOptions)
  global config
  config = rnn.config
  rnn.initLog()
  print("CRNN dump-dataset starting up.", file=log.v1)
  rnn.initFaulthandler()
  rnn.initConfigJsonNetwork()
  rnn.initData()
  rnn.printTaskProperties()


def main(argv):
  argparser = argparse.ArgumentParser(description='Dump something from dataset.')
  argparser.add_argument('crnn_config_file')
  argparser.add_argument('--epoch', type=int, default=1)
  argparser.add_argument('--startseq', type=int, default=0, help='start seq idx (inclusive) (default: 0)')
  argparser.add_argument('--endseq', type=int, default=10, help='end seq idx (inclusive) or -1 (default: 10)')
  argparser.add_argument('--type', default='stdout', help="'numpy' or 'stdout'")
  argparser.add_argument('--dump_prefix', default='/tmp/crnn.dump-dataset.')
  argparser.add_argument('--dump_postfix', default='.txt.gz')
  args = argparser.parse_args(argv[1:])
  init(configFilename=args.crnn_config_file, commandLineOptions=[])
  dump_dataset(rnn.train_data, args)
  rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
