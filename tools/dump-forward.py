#!/usr/bin/env python

from __future__ import print_function

import os
import sys

my_dir = os.path.dirname(os.path.abspath(__file__))
returnn_dir = os.path.dirname(my_dir)
sys.path.append(returnn_dir)

import rnn
from Log import log
import argparse
import numpy
from better_exchook import pretty_print


def dump(dataset, options):
  """
  :type dataset: Dataset.Dataset
  :param options: argparse.Namespace
  """
  print("Epoch: %i" % options.epoch, file=log.v3)
  dataset.init_seq_order(options.epoch)

  output_dict = {}
  for name, layer in rnn.engine.network.layers.items():
    output_dict["%s:out" % name] = layer.output.placeholder
    for i, v in layer.output.size_placeholder.items():
      output_dict["%s:shape(%i)" % (name, layer.output.get_batch_axis(i))] = v

  seq_idx = options.startseq
  if options.endseq < 0:
    options.endseq = float("inf")
  while dataset.is_less_than_num_seqs(seq_idx) and seq_idx <= options.endseq:
    print("Seq idx: %i" % (seq_idx,), file=log.v3)
    out = rnn.engine.run_single(dataset=dataset, seq_idx=seq_idx, output_dict=output_dict)
    for name, v in sorted(out.items()):
      print("  %s: %s" % (name, pretty_print(v)))
    seq_idx += 1

  print("Done. More seqs which we did not dumped: %s" % dataset.is_less_than_num_seqs(seq_idx), file=log.v1)


def init(configFilename, commandLineOptions):
  rnn.init(
    config_filename=configFilename, command_line_options=commandLineOptions,
    config_updates={"log": None},
    extra_greeting="CRNN dump-forward starting up.")
  rnn.engine.init_train_from_config(config=rnn.config, train_data=rnn.train_data)
  # rnn.engine.init_network_from_config(rnn.config)


def main(argv):
  argparser = argparse.ArgumentParser(description='Forward something and dump it.')
  argparser.add_argument('crnn_config_file')
  argparser.add_argument('--epoch', type=int, default=1)
  argparser.add_argument('--startseq', type=int, default=0, help='start seq idx (inclusive) (default: 0)')
  argparser.add_argument('--endseq', type=int, default=10, help='end seq idx (inclusive) or -1 (default: 10)')
  args = argparser.parse_args(argv[1:])
  init(configFilename=args.crnn_config_file, commandLineOptions=[])
  dump(rnn.train_data, args)
  rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
