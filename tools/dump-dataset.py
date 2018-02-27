#!/usr/bin/env python3

from __future__ import print_function

import os
import sys
import time

my_dir = os.path.dirname(os.path.abspath(__file__))
returnn_dir = os.path.dirname(my_dir)
sys.path.append(returnn_dir)

import rnn
from Log import log
import argparse
import numpy
from better_exchook import pretty_print
from Util import Stats, hms
import Util


def plot(m):
  """
  :param numpy.ndarray m:
  """
  print("Plotting matrix of shape %s." % (m.shape,))
  from matplotlib.pyplot import matshow, show
  matshow(m.transpose())
  show()


def dump_dataset(dataset, options):
  """
  :type dataset: Dataset.Dataset
  :param options: argparse.Namespace
  """
  print("Epoch: %i" % options.epoch, file=log.v3)
  dataset.init_seq_order(epoch=options.epoch)

  if options.get_num_seqs:
    print("Get num seqs.")
    print("estimated_num_seqs: %r" % dataset.estimated_num_seqs)
    try:
      print("num_seqs: %r" % dataset.num_seqs)
    except Exception as exc:
      print("num_seqs exception %r, which is valid, so we count." % exc)
      seq_idx = 0
      if dataset.get_target_list():
        default_target = dataset.get_target_list()[0]
      else:
        default_target = None
      while dataset.is_less_than_num_seqs(seq_idx):
        dataset.load_seqs(seq_idx, seq_idx + 1)
        if seq_idx % 10000 == 0:
          if default_target:
            targets = dataset.get_targets(default_target, seq_idx)
            postfix = " (targets = %r...)" % (targets[:10],)
          else:
            postfix = ""
          print("%i ...%s" % (seq_idx, postfix))
        seq_idx += 1
      print("accumulated num seqs: %i" % seq_idx)
    print("Done.")
    return

  if options.type == "numpy":
    print("Dump files: %r*%r" % (options.dump_prefix, options.dump_postfix), file=log.v3)
  elif options.type == "stdout":
    print("Dump to stdout", file=log.v3)
  elif options.type == "print_shape":
    print("Dump shape to stdout", file=log.v3)
  elif options.type == "plot":
    print("Plot.", file=log.v3)
  elif options.type == "null":
    print("No dump.")
  else:
    raise Exception("unknown dump option type %r" % options.type)

  start_time = time.time()
  stats = Stats() if (options.stats or options.dump_stats) else None
  seq_len_stats = {key: Stats() for key in dataset.get_data_keys()}
  seq_idx = options.startseq
  if options.endseq < 0:
    options.endseq = float("inf")
  while dataset.is_less_than_num_seqs(seq_idx) and seq_idx <= options.endseq:
    dataset.load_seqs(seq_idx, seq_idx + 1)
    complete_frac = dataset.get_complete_frac(seq_idx)
    start_elapsed = time.time() - start_time
    try:
      num_seqs_s = str(dataset.num_seqs)
    except NotImplementedError:
      try:
        num_seqs_s = "~%i" % dataset.estimated_num_seqs
      except TypeError:  # a number is required, not NoneType
        num_seqs_s = "?"
    progress_prefix = "%i/%s" % (seq_idx, num_seqs_s)
    progress = "%s (%.02f%%)" % (progress_prefix, complete_frac * 100)
    if complete_frac > 0:
      total_time_estimated = start_elapsed / complete_frac
      remaining_estimated = total_time_estimated - start_elapsed
      progress += " (%s)" % hms(remaining_estimated)
    data = dataset.get_data(seq_idx, options.key)
    if options.type == "numpy":
      numpy.savetxt("%s%i.data%s" % (options.dump_prefix, seq_idx, options.dump_postfix), data)
    elif options.type == "stdout":
      print("seq %s data:" % progress, pretty_print(data))
    elif options.type == "print_shape":
      print("seq %s data shape:" % progress, data.shape)
    elif options.type == "plot":
      plot(data)
    for target in dataset.get_target_list():
      targets = dataset.get_targets(target, seq_idx)
      if options.type == "numpy":
        numpy.savetxt("%s%i.targets.%s%s" % (options.dump_prefix, seq_idx, target, options.dump_postfix), targets, fmt='%i')
      elif options.type == "stdout":
        print("seq %i target %r:" % (seq_idx, target), pretty_print(targets))
      elif options.type == "print_shape":
        print("seq %i target %r shape:" % (seq_idx, target), targets.shape)
    seq_len = dataset.get_seq_length(seq_idx)
    for key in dataset.get_data_keys():
      seq_len_stats[key].collect([seq_len[key]])
    if stats:
      stats.collect(data)
    if options.type == "null":
      Util.progress_bar_with_time(complete_frac, prefix=progress_prefix)

    seq_idx += 1

  print("Done. Total time %s. More seqs which we did not dumped: %s" % (
    hms(time.time() - start_time), dataset.is_less_than_num_seqs(seq_idx)), file=log.v1)
  for key in dataset.get_data_keys():
    seq_len_stats[key].dump(stream_prefix="Seq-length %r " % key, stream=log.v2)
  if stats:
    stats.dump(output_file_prefix=options.dump_stats, stream_prefix="Data %r " % options.key, stream=log.v2)


def init(config_str):
  """
  :param str config_str: either filename to config-file, or dict for dataset
  """
  rnn.initBetterExchook()
  rnn.initThreadJoinHack()
  if config_str.strip().startswith("{"):
    print("Using dataset %s." % config_str)
    datasetDict = eval(config_str.strip())
    configFilename = None
  else:
    datasetDict = None
    configFilename = config_str
    print("Using config file %r." % configFilename)
    assert os.path.exists(configFilename)
  rnn.initConfig(configFilename=configFilename, commandLineOptions=[])
  global config
  config = rnn.config
  config.set("log", None)
  config.set("log_verbosity", 4)
  if datasetDict:
    config.set("train", datasetDict)
  rnn.initLog()
  print("Returnn dump-dataset starting up.", file=log.v1)
  rnn.returnnGreeting()
  rnn.initFaulthandler()
  rnn.initConfigJsonNetwork()
  rnn.initData()
  rnn.printTaskProperties()


def main(argv):
  argparser = argparse.ArgumentParser(description='Dump something from dataset.')
  argparser.add_argument('crnn_config', help="either filename to config-file, or dict for dataset")
  argparser.add_argument('--epoch', type=int, default=1)
  argparser.add_argument('--startseq', type=int, default=0, help='start seq idx (inclusive) (default: 0)')
  argparser.add_argument('--endseq', type=int, default=10, help='end seq idx (inclusive) or -1 (default: 10)')
  argparser.add_argument('--get_num_seqs', action="store_true")
  argparser.add_argument('--type', default='stdout', help="'numpy', 'stdout', 'plot', 'null'")
  argparser.add_argument('--dump_prefix', default='/tmp/crnn.dump-dataset.')
  argparser.add_argument('--dump_postfix', default='.txt.gz')
  argparser.add_argument("--key", default="data", help="data-key, e.g. 'data' or 'classes'. (default: 'data')")
  argparser.add_argument('--stats', action="store_true", help="calculate mean/stddev stats")
  argparser.add_argument('--dump_stats', help="file-prefix to dump stats to")
  args = argparser.parse_args(argv[1:])
  init(config_str=args.crnn_config)
  try:
    dump_dataset(rnn.train_data, args)
  except KeyboardInterrupt:
    print("KeyboardInterrupt")
    sys.exit(1)
  finally:
    rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
