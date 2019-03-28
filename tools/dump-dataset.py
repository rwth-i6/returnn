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
from Util import Stats, hms, pretty_print
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
  print("Dataset keys:", dataset.get_data_keys(), file=log.v3)
  print("Dataset target keys:", dataset.get_target_list(), file=log.v3)
  assert options.key in dataset.get_data_keys()

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
    if options.stdout_limit is not None:
      Util.set_pretty_print_default_limit(options.stdout_limit)
      numpy.set_printoptions(
        threshold=sys.maxsize if options.stdout_limit == float("inf") else int(options.stdout_limit))
    if options.stdout_as_bytes:
      Util.set_pretty_print_as_bytes(options.stdout_as_bytes)
  elif options.type == "print_tag":
    print("Dump seq tag to stdout", file=log.v3)
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
    if options.type == "print_tag":
      print("seq %s tag:" % (progress if log.verbose[2] else progress_prefix), dataset.get_tag(seq_idx))
    else:
      data = dataset.get_data(seq_idx, options.key)
      if options.type == "numpy":
        numpy.savetxt("%s%i.data%s" % (options.dump_prefix, seq_idx, options.dump_postfix), data)
      elif options.type == "stdout":
        print("seq %s tag:" % progress, dataset.get_tag(seq_idx))
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
          extra = ""
          if target in dataset.labels and len(dataset.labels[target]) > 1:
            labels = dataset.labels[target]
            if len(labels) < 1000 and all([len(l) == 1 for l in labels]):
              join_str = ""
            else:
              join_str = " "
            extra += " (%r)" % join_str.join(map(dataset.labels[target].__getitem__, targets))
          print("seq %i target %r: %s%s" % (seq_idx, target, pretty_print(targets), extra))
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
    hms(time.time() - start_time), dataset.is_less_than_num_seqs(seq_idx)), file=log.v2)
  for key in dataset.get_data_keys():
    seq_len_stats[key].dump(stream_prefix="Seq-length %r " % key, stream=log.v2)
  if stats:
    stats.dump(output_file_prefix=options.dump_stats, stream_prefix="Data %r " % options.key, stream=log.v1)


def init(config_str, verbosity):
  """
  :param str config_str: either filename to config-file, or dict for dataset
  :param int verbosity:
  """
  rnn.init_better_exchook()
  rnn.init_thread_join_hack()
  datasetDict = None
  configFilename = None
  if config_str.strip().startswith("{"):
    print("Using dataset %s." % config_str)
    datasetDict = eval(config_str.strip())
  elif config_str.endswith(".hdf"):
    datasetDict = {"class": "HDFDataset", "files": [config_str]}
    print("Using dataset %r." % datasetDict)
    assert os.path.exists(config_str)
  else:
    configFilename = config_str
    print("Using config file %r." % configFilename)
    assert os.path.exists(configFilename)
  rnn.init_config(config_filename=configFilename, default_config={"cache_size": "0"})
  global config
  config = rnn.config
  config.set("log", None)
  config.set("log_verbosity", verbosity)
  if datasetDict:
    config.set("train", datasetDict)
  rnn.init_log()
  print("Returnn dump-dataset starting up.", file=log.v2)
  rnn.returnn_greeting()
  rnn.init_faulthandler()
  rnn.init_config_json_network()
  rnn.init_data()
  rnn.print_task_properties()


def main():
  argparser = argparse.ArgumentParser(description='Dump something from dataset.')
  argparser.add_argument('crnn_config', help="either filename to config-file, or dict for dataset")
  argparser.add_argument('--epoch', type=int, default=1)
  argparser.add_argument('--startseq', type=int, default=0, help='start seq idx (inclusive) (default: 0)')
  argparser.add_argument('--endseq', type=int, default=10, help='end seq idx (inclusive) or -1 (default: 10)')
  argparser.add_argument('--get_num_seqs', action="store_true")
  argparser.add_argument('--type', default='stdout', help="'numpy', 'stdout', 'plot', 'null' (default 'stdout')")
  argparser.add_argument("--stdout_limit", type=float, default=None, help="e.g. inf to disable")
  argparser.add_argument("--stdout_as_bytes", action="store_true")
  argparser.add_argument("--verbosity", type=int, default=4, help="overwrites log_verbosity (default: 4)")
  argparser.add_argument('--dump_prefix', default='/tmp/crnn.dump-dataset.')
  argparser.add_argument('--dump_postfix', default='.txt.gz')
  argparser.add_argument("--key", default="data", help="data-key, e.g. 'data' or 'classes'. (default: 'data')")
  argparser.add_argument('--stats', action="store_true", help="calculate mean/stddev stats")
  argparser.add_argument('--dump_stats', help="file-prefix to dump stats to")
  args = argparser.parse_args()
  init(config_str=args.crnn_config, verbosity=args.verbosity)
  try:
    dump_dataset(rnn.train_data, args)
  except KeyboardInterrupt:
    print("KeyboardInterrupt")
    sys.exit(1)
  finally:
    rnn.finalize()


if __name__ == '__main__':
  main()
