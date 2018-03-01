#!/usr/bin/env python3

from __future__ import print_function

import os
import sys
import time
import tensorflow as tf
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
import TFUtil


class WerComputeGraph:
  def __init__(self):
    self.hyps = tf.placeholder(tf.string, [None])
    self.refs = tf.placeholder(tf.string, [None])
    self.wer, self.ref_num_words = TFUtil.string_words_calc_wer(hyps=self.hyps, refs=self.refs)
    self.total_wer_var = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
    self.total_ref_num_words_var = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
    self.update_total_wer = self.total_wer_var.assign_add(tf.reduce_sum(self.wer))
    self.update_ref_num_words = self.total_ref_num_words_var.assign_add(tf.reduce_sum(self.ref_num_words))
    self.updated_normalized_wer = \
      tf.cast(self.update_total_wer, tf.float32) / tf.cast(self.update_ref_num_words, tf.float32)

  def step(self, session, hyps, refs):
    """
    :param tf.Session session:
    :param list[str] hyps:
    :param list[str] refs:
    :return: updated normalized WER
    :rtype: float
    """
    return session.run(self.updated_normalized_wer, feed_dict={self.hyps: hyps, self.refs: refs})


def calc_wer_on_dataset(dataset, options, hyps):
  """
  :param Dataset dataset:
  :param options: argparse.Namespace
  :param dict[str,str] hyps: seq tag -> hyp string (words delimited by space)
  """
  start_time = time.time()
  seq_len_stats = {"refs": Stats(), "hyps": Stats()}
  seq_idx = options.startseq
  if options.endseq < 0:
    options.endseq = float("inf")
  wer = 1.0
  remaining_hyp_seq_tags = set(hyps.keys())
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
    progress_prefix = "%i/%s (WER %.02f%%)" % (seq_idx, num_seqs_s, wer * 100)
    progress = "%s (%.02f%%)" % (progress_prefix, complete_frac * 100)
    if complete_frac > 0:
      total_time_estimated = start_elapsed / complete_frac
      remaining_estimated = total_time_estimated - start_elapsed
      progress += " (%s)" % hms(remaining_estimated)
    seq_tag = dataset.get_tag(seq_idx)
    hyp = hyps[seq_tag]
    ref = dataset.get_data(seq_idx, options.key)
    if isinstance(ref, numpy.ndarray):
      assert ref.shape == ()
      ref = ref.flatten()[0]  # get the entry itself (str or bytes)
    if isinstance(ref, bytes):
      ref = ref.decode("utf8")
    assert isinstance(ref, str)
    wer = wer_compute.step(session, hyps=[hyp], refs=[ref])
    seq_len_stats["hyps"].collect([len(hyp)])
    seq_len_stats["refs"].collect([len(ref)])
    remaining_hyp_seq_tags.remove(seq_tag)
    Util.progress_bar_with_time(complete_frac, prefix=progress_prefix)
    seq_idx += 1

  print("Done. Num seqs %i. Total time %s." % (
    seq_idx, hms(time.time() - start_time)), file=log.v1)
  print("Remaining num hyp seqs %i. More seqs which we did not dumped: %s." % (
    len(remaining_hyp_seq_tags), dataset.is_less_than_num_seqs(seq_idx),), file=log.v1)
  print("Final WER: %.02f%%" % (wer * 100), file=log.v1)

  for key in ["hyps", "refs"]:
    seq_len_stats[key].dump(stream_prefix="Seq-length %r %r" % (key, options.key), stream=log.v2)


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
  config.set("task", "calculate_wer")
  config.set("log", None)
  config.set("log_verbosity", 4)
  if datasetDict:
    config.set("eval", datasetDict)
  rnn.initLog()
  print("Returnn calculate-word-error-rate starting up.", file=log.v1)
  rnn.returnnGreeting()
  rnn.initBackendEngine()
  assert Util.BackendEngine.is_tensorflow_selected(), "this is only for TensorFlow"
  rnn.initFaulthandler()
  rnn.initConfigJsonNetwork()
  rnn.printTaskProperties()


def main(argv):
  argparser = argparse.ArgumentParser(description='Dump something from dataset.')
  argparser.add_argument('crnn_config', help="either filename to config-file, or dict for dataset")
  argparser.add_argument("--hyps", help="hypotheses, dumped via search in py format")
  argparser.add_argument('--startseq', type=int, default=0, help='start seq idx (inclusive) (default: 0)')
  argparser.add_argument('--endseq', type=int, default=-1, help='end seq idx (inclusive) or -1 (default: -1)')
  argparser.add_argument("--key", default="raw", help="data-key, e.g. 'data' or 'classes'. (default: 'raw')")
  argparser.add_argument("--data")
  args = argparser.parse_args(argv[1:])

  init(config_str=args.crnn_config)
  if args.data:
    dataset = init_dataset(args.data)
  elif config.value("wer_data", "eval") in ["train", "dev", "eval"]:
    dataset = init_dataset(config.opt_typed_value(config.value("search_data", "eval")))
  else:
    dataset = init_dataset(config.opt_typed_value("wer_data"))
  dataset.init_seq_order(epoch=1)

  hyps = eval(open(args.hyps).read())
  assert isinstance(hyps, dict)
  assert len(hyps) > 0
  example_hyp = next(iter(hyps.items()))
  assert isinstance(example_hyp[0], str)  # seq tag
  assert isinstance(example_hyp[1], str)  # hyp

  global wer_compute
  wer_compute = WerComputeGraph()
  with tf.Session(config=tf.ConfigProto(device_count={"GPU": 0})) as _session:
    global session
    session = _session
    session.run(tf.global_variables_initializer())
    try:
      calc_wer_on_dataset(dataset=dataset, options=args, hyps=hyps)
    except KeyboardInterrupt:
      print("KeyboardInterrupt")
      sys.exit(1)
    finally:
      rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
