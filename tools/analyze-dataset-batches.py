#!/usr/bin/env python3

"""
Goes through a dataset (similar as dump-dataset.py), but does the training batch creation,
and calculate statistics about how much seqs we have per batch, how much zero padding there is, etc.
"""

from __future__ import print_function, division

import os
import sys
import time
import typing
import argparse
import numpy
from pprint import pformat

import _setup_returnn_env  # noqa
import returnn.__main__ as rnn
from returnn.log import log
import returnn.util.basic as util
from returnn.util.basic import Stats, hms, NumbersDict
from returnn.datasets.basic import Batch, Dataset, init_dataset
from returnn.config import Config


config = None  # type: typing.Optional[Config]
dataset = None  # type: typing.Optional[Dataset]


def analyze_dataset(options):
  """
  :param options: argparse.Namespace
  """
  print("Epoch: %i" % options.epoch, file=log.v3)
  print("Dataset keys:", dataset.get_data_keys(), file=log.v3)
  print("Dataset target keys:", dataset.get_target_list(), file=log.v3)
  assert options.key in dataset.get_data_keys()

  terminal_width, _ = util.terminal_size()
  show_interactive_process_bar = (log.verbose[3] and (not log.verbose[5]) and terminal_width > 0)

  start_time = time.time()
  num_seqs_stats = Stats()
  if options.endseq < 0:
    options.endseq = float("inf")

  recurrent = True
  used_data_keys = dataset.get_data_keys()
  batch_size = config.typed_value('batch_size', 1)
  max_seqs = config.int('max_seqs', -1)
  seq_drop = config.float('seq_drop', 0.0)
  max_seq_length = config.typed_value('max_seq_length', None) or config.float('max_seq_length', 0)
  max_pad_size = config.typed_value("max_pad_size", None)

  batches = dataset.generate_batches(
    recurrent_net=recurrent,
    batch_size=batch_size,
    max_seqs=max_seqs,
    max_seq_length=max_seq_length,
    max_pad_size=max_pad_size,
    seq_drop=seq_drop,
    used_data_keys=used_data_keys)

  step = 0
  total_num_seqs = 0
  total_num_frames = NumbersDict()
  total_num_used_frames = NumbersDict()

  try:
    while batches.has_more():
      # See FeedDictDataProvider.
      batch, = batches.peek_next_n(1)
      assert isinstance(batch, Batch)
      if batch.start_seq > options.endseq:
        break
      dataset.load_seqs(batch.start_seq, batch.end_seq)
      complete_frac = batches.completed_frac()
      start_elapsed = time.time() - start_time
      try:
        num_seqs_s = str(dataset.num_seqs)
      except NotImplementedError:
        try:
          num_seqs_s = "~%i" % dataset.estimated_num_seqs
        except TypeError:  # a number is required, not NoneType
          num_seqs_s = "?"
      progress_prefix = "%i/%s" % (batch.start_seq, num_seqs_s)
      progress = "%s (%.02f%%)" % (progress_prefix, complete_frac * 100)
      if complete_frac > 0:
        total_time_estimated = start_elapsed / complete_frac
        remaining_estimated = total_time_estimated - start_elapsed
        progress += " (%s)" % hms(remaining_estimated)

      batch_max_time = NumbersDict.max([seq.frame_length for seq in batch.seqs]) * len(batch.seqs)
      batch_num_used_frames = sum([seq.frame_length for seq in batch.seqs], NumbersDict())
      total_num_seqs += len(batch.seqs)
      num_seqs_stats.collect(numpy.array([len(batch.seqs)]))
      total_num_frames += batch_max_time
      total_num_used_frames += batch_num_used_frames

      print(
        "%s, batch %i, num seqs %i, frames %s, used %s (%s)" % (
          progress, step, len(batch.seqs),
          batch_max_time, batch_num_used_frames, batch_num_used_frames / batch_max_time),
        file=log.v5)
      if show_interactive_process_bar:
        util.progress_bar_with_time(complete_frac, prefix=progress_prefix)

      step += 1
      batches.advance(1)

  finally:
    print("Done. Total time %s. More seqs which we did not dumped: %s" % (
      hms(time.time() - start_time), batches.has_more()), file=log.v2)
    print("Dataset epoch %i, order %r." % (dataset.epoch, dataset.seq_ordering))
    print("Num batches (steps): %i" % step, file=log.v1)
    print("Num seqs: %i" % total_num_seqs, file=log.v1)
    num_seqs_stats.dump(stream=log.v1, stream_prefix="Batch num seqs ")
    for key in used_data_keys:
      print("Data key %r:" % key, file=log.v1)
      print("  Num frames: %s" % total_num_frames[key], file=log.v1)
      print("  Num used frames: %s" % total_num_used_frames[key], file=log.v1)
      print("  Fraction used frames: %s" % (total_num_used_frames / total_num_frames)[key], file=log.v1)
    dataset.finish_epoch()


def init(config_str, config_dataset, use_pretrain, epoch, verbosity):
  """
  :param str config_str: either filename to config-file, or dict for dataset
  :param str|None config_dataset:
  :param bool use_pretrain: might overwrite config options, or even the dataset
  :param int epoch:
  :param int verbosity:
  """
  rnn.init_better_exchook()
  rnn.init_thread_join_hack()
  dataset_opts = None
  config_filename = None
  if config_str.strip().startswith("{"):
    print("Using dataset %s." % config_str)
    dataset_opts = eval(config_str.strip())
  elif config_str.endswith(".hdf"):
    dataset_opts = {"class": "HDFDataset", "files": [config_str]}
    print("Using dataset %r." % dataset_opts)
    assert os.path.exists(config_str)
  else:
    config_filename = config_str
    print("Using config file %r." % config_filename)
    assert os.path.exists(config_filename)
  rnn.init_config(config_filename=config_filename, default_config={"cache_size": "0"})
  global config
  config = rnn.config
  config.set("log", None)
  config.set("log_verbosity", verbosity)
  rnn.init_log()
  print("Returnn %s starting up." % __file__, file=log.v2)
  rnn.returnn_greeting()
  rnn.init_faulthandler()
  rnn.init_config_json_network()
  util.BackendEngine.select_engine(config=config)
  if not dataset_opts:
    if config_dataset:
      dataset_opts = "config:%s" % config_dataset
    else:
      dataset_opts = "config:train"
  if use_pretrain:
    from returnn.pretrain import pretrain_from_config
    pretrain = pretrain_from_config(config)
    if pretrain:
      print("Using pretrain %s, epoch %i" % (pretrain, epoch), file=log.v2)
      net_dict = pretrain.get_network_json_for_epoch(epoch=epoch)
      if "#config" in net_dict:
        config_overwrites = net_dict["#config"]
        print("Pretrain overwrites these config options:", file=log.v2)
        assert isinstance(config_overwrites, dict)
        for key, value in sorted(config_overwrites.items()):
          assert isinstance(key, str)
          orig_value = config.typed_dict.get(key, None)
          if isinstance(orig_value, dict) and isinstance(value, dict):
            diff_str = "\n" + util.obj_diff_str(orig_value, value)
          elif isinstance(value, dict):
            diff_str = "\n%r ->\n%s" % (orig_value, pformat(value))
          else:
            diff_str = " %r -> %r" % (orig_value, value)
          print("Config key %r for epoch %i:%s" % (key, epoch, diff_str), file=log.v2)
          config.set(key, value)
      else:
        print("No config overwrites for this epoch.", file=log.v2)
    else:
      print("No pretraining used.", file=log.v2)
  elif config.typed_dict.get("pretrain", None):
    print("Not using pretrain.", file=log.v2)
  dataset_default_opts = {}
  Dataset.kwargs_update_from_config(config, dataset_default_opts)
  print("Using dataset:", dataset_opts, file=log.v2)
  global dataset
  dataset = init_dataset(dataset_opts, default_kwargs=dataset_default_opts)
  assert isinstance(dataset, Dataset)
  dataset.init_seq_order(epoch=epoch)


def main():
  """
  Main entry.
  """
  arg_parser = argparse.ArgumentParser(description='Anaylize dataset batches.')
  arg_parser.add_argument('returnn_config', help="either filename to config-file, or dict for dataset")
  arg_parser.add_argument("--dataset", help="if given the config, specifies the dataset. e.g. 'dev'")
  arg_parser.add_argument('--epoch', type=int, default=1)
  arg_parser.add_argument('--endseq', type=int, default=-1, help='end seq idx (inclusive) or -1 (default: 10)')
  arg_parser.add_argument("--verbosity", type=int, default=5, help="overwrites log_verbosity (default: 4)")
  arg_parser.add_argument("--key", default="data", help="data-key, e.g. 'data' or 'classes'. (default: 'data')")
  arg_parser.add_argument("--use_pretrain", action="store_true")
  args = arg_parser.parse_args()
  init(
    config_str=args.returnn_config, config_dataset=args.dataset, epoch=args.epoch, use_pretrain=args.use_pretrain,
    verbosity=args.verbosity)
  try:
    analyze_dataset(args)
  except KeyboardInterrupt:
    print("KeyboardInterrupt")
    sys.exit(1)
  finally:
    rnn.finalize()


if __name__ == '__main__':
  main()
