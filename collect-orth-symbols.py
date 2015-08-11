#!/usr/bin/env python

import os
import sys
import rnn
from Log import log
import argparse
from Util import hms, parse_orthography


def iter_dataset(dataset, options):
  """
  :type dataset: Dataset.Dataset
  :param options: argparse.Namespace
  """
  rnn.train_data.init_seq_order(epoch=1)

  assert "orth" in dataset.get_target_list()

  orth_symbols_filename = options.output
  assert not os.path.exists(orth_symbols_filename)
  orth_syms_file = open(orth_symbols_filename, "w")

  total_frame_len = 0
  total_orth_len = 0
  orth_syms_set = set()

  seq_idx = 0
  while dataset.is_less_than_num_seqs(seq_idx):
    dataset.load_seqs(seq_idx, seq_idx)
    total_frame_len += dataset.get_seq_length(seq_idx)[0]

    orth = dataset.get_targets("orth", seq_idx)
    orth_syms = parse_orthography(orth)
    orth_syms_set.update(orth_syms)
    total_orth_len += len(orth_syms)

    seq_idx += 1

  print >> log.v3, "Total frame len:", total_frame_len, "time:", hms(total_frame_len * (options.frame_time / 1000.0))
  print >> log.v3, "Total orth len:", total_orth_len, "fraction:", float(total_orth_len) / total_frame_len
  print >> log.v3, "Num symbols:", len(orth_syms_set)

  sys.stdout.flush()
  for orth_sym in sorted(orth_syms_set):
    sys.stdout.write(orth_sym + "\n")
    orth_syms_file.write(orth_sym + "\n")
  sys.stdout.flush()
  orth_syms_file.close()
  print >> log.v3, "Wrote orthography symbols to", orth_symbols_filename


def init(configFilename, commandLineOptions):
  rnn.initBetterExchook()
  rnn.initThreadJoinHack()
  rnn.initConfig(configFilename, commandLineOptions)
  global config
  config = rnn.config
  rnn.initLog()
  print >> log.v3, "CRNN collect-orth-symbols starting up."
  rnn.initFaulthandler()
  rnn.initConfigJson()
  rnn.initData()
  rnn.printTaskProperties()


def main(argv):
  argparser = argparse.ArgumentParser(description='Collect orth symbols.')
  argparser.add_argument('crnn_config_file')
  argparser.add_argument('--frame_time', type=int, default=10, help='time (in ms) per frame')
  argparser.add_argument('--output', default='orth_symbols')
  args = argparser.parse_args(argv[1:])
  init(configFilename=args.crnn_config_file, commandLineOptions=[])
  iter_dataset(rnn.train_data, args)
  rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
