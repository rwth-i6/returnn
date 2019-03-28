#!/usr/bin/env python3

from __future__ import print_function

import os
import sys
import time
import numpy

my_dir = os.path.dirname(os.path.abspath(__file__))
returnn_dir = os.path.dirname(my_dir)
sys.path.insert(0, returnn_dir)

import rnn
from Log import log
import argparse
from Util import Stats, hms
from Dataset import Dataset, init_dataset
import Util


def get_raw_strings(dataset, options):
  """
  :param Dataset dataset:
  :param options: argparse.Namespace
  :return: list of (seq tag, string)
  :rtype: list[(str,str)]
  """
  refs = []
  start_time = time.time()
  seq_len_stats = Stats()
  seq_idx = options.startseq
  if options.endseq < 0:
    options.endseq = float("inf")
  interactive = Util.is_tty() and not log.verbose[5]
  print("Iterating over %r." % dataset, file=log.v2)
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
    progress_prefix = "%i/%s" % (seq_idx, num_seqs_s,)
    progress = "%s (%.02f%%)" % (progress_prefix, complete_frac * 100)
    if complete_frac > 0:
      total_time_estimated = start_elapsed / complete_frac
      remaining_estimated = total_time_estimated - start_elapsed
      progress += " (%s)" % hms(remaining_estimated)
    seq_tag = dataset.get_tag(seq_idx)
    assert isinstance(seq_tag, str)
    ref = dataset.get_data(seq_idx, options.key)
    if isinstance(ref, numpy.ndarray):
      assert ref.shape == () or (ref.ndim == 1 and ref.dtype == numpy.uint8)
      if ref.shape == ():
        ref = ref.flatten()[0]  # get the entry itself (str or bytes)
      else:
        ref = ref.tobytes()
    if isinstance(ref, bytes):
      ref = ref.decode("utf8")
    assert isinstance(ref, str)
    seq_len_stats.collect([len(ref)])
    refs.append((seq_tag, ref))
    if interactive:
      Util.progress_bar_with_time(complete_frac, prefix=progress_prefix)
    elif log.verbose[5]:
      print(progress_prefix, "seq tag %r, ref len %i chars" % (seq_tag, len(ref)))
    seq_idx += 1
  print("Done. Num seqs %i. Total time %s." % (
    seq_idx, hms(time.time() - start_time)), file=log.v1)
  print("More seqs which we did not dumped: %s." % (
    dataset.is_less_than_num_seqs(seq_idx),), file=log.v1)
  seq_len_stats.dump(stream_prefix="Seq-length %r " % (options.key,), stream=log.v2)
  return refs


def init(config_filename, log_verbosity):
  """
  :param str config_filename: filename to config-file
  :param int log_verbosity:
  """
  rnn.init_better_exchook()
  rnn.init_thread_join_hack()
  if config_filename:
    print("Using config file %r." % config_filename)
    assert os.path.exists(config_filename)
  rnn.init_config(config_filename=config_filename, command_line_options=[])
  global config
  config = rnn.config
  config.set("task", "dump")
  config.set("log", None)
  config.set("log_verbosity", log_verbosity)
  rnn.init_log()
  print("Returnn dump-dataset-raw-strings starting up.", file=log.v1)
  rnn.returnn_greeting()
  rnn.init_faulthandler()


def main(argv):
  argparser = argparse.ArgumentParser(description='Dump raw strings from dataset. Same format as in search.')
  argparser.add_argument('--config', help="filename to config-file. will use dataset 'eval' from it")
  argparser.add_argument("--dataset", help="dataset, overwriting config")
  argparser.add_argument('--startseq', type=int, default=0, help='start seq idx (inclusive) (default: 0)')
  argparser.add_argument('--endseq', type=int, default=-1, help='end seq idx (inclusive) or -1 (default: -1)')
  argparser.add_argument("--key", default="raw", help="data-key, e.g. 'data' or 'classes'. (default: 'raw')")
  argparser.add_argument("--verbosity", default=4, type=int, help="5 for all seqs (default: 4)")
  argparser.add_argument("--out", required=True, help="out-file. py-format as in task=search")
  args = argparser.parse_args(argv[1:])
  assert args.config or args.dataset

  init(config_filename=args.config, log_verbosity=args.verbosity)
  if args.dataset:
    dataset = init_dataset(args.dataset)
  elif config.value("dump_data", "eval") in ["train", "dev", "eval"]:
    dataset = init_dataset(config.opt_typed_value(config.value("search_data", "eval")))
  else:
    dataset = init_dataset(config.opt_typed_value("wer_data"))
  dataset.init_seq_order(epoch=1)

  try:
    with open(args.out, "w") as output_file:
      refs = get_raw_strings(dataset=dataset, options=args)
      output_file.write("{\n")
      for seq_tag, ref in refs:
        output_file.write("%r: %r,\n" % (seq_tag, ref))
      output_file.write("}\n")
    print("Done. Wrote to %r." % args.out)
  except KeyboardInterrupt:
    print("KeyboardInterrupt")
    sys.exit(1)
  finally:
    rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
