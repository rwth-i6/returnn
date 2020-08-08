#!/usr/bin/env python

"""
For debugging, go through some dataset, forward it through the net, and output the layer activations on stdout.
"""

from __future__ import print_function

import sys

import _setup_returnn_env  # noqa
import returnn.__main__ as rnn
from returnn.log import log
import argparse
from returnn.util.basic import pretty_print


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


def init(config_filename, command_line_options):
  """
  :param str config_filename:
  :param list[str] command_line_options:
  """
  rnn.init(
    config_filename=config_filename, command_line_options=command_line_options,
    config_updates={"log": None},
    extra_greeting="RETURNN dump-forward starting up.")
  rnn.engine.init_train_from_config(config=rnn.config, train_data=rnn.train_data)
  # rnn.engine.init_network_from_config(rnn.config)


def main(argv):
  """
  Main entry.
  """
  arg_parser = argparse.ArgumentParser(description='Forward something and dump it.')
  arg_parser.add_argument('returnn_config')
  arg_parser.add_argument('--epoch', type=int, default=1)
  arg_parser.add_argument('--startseq', type=int, default=0, help='start seq idx (inclusive) (default: 0)')
  arg_parser.add_argument('--endseq', type=int, default=10, help='end seq idx (inclusive) or -1 (default: 10)')
  args = arg_parser.parse_args(argv[1:])
  init(config_filename=args.returnn_config, command_line_options=[])
  dump(rnn.train_data, args)
  rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
