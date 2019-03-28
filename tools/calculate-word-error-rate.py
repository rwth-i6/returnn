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


def calc_wer_on_dataset(dataset, refs, options, hyps):
  """
  :param Dataset|None dataset:
  :param dict[str,str]|None refs: seq tag -> ref string (words delimited by space)
  :param options: argparse.Namespace
  :param dict[str,str] hyps: seq tag -> hyp string (words delimited by space)
  :return: WER
  :rtype: float
  """
  assert dataset or refs
  start_time = time.time()
  seq_len_stats = {"refs": Stats(), "hyps": Stats()}
  seq_idx = options.startseq
  if options.endseq < 0:
    options.endseq = float("inf")
  wer = 1.0
  remaining_hyp_seq_tags = set(hyps.keys())
  interactive = Util.is_tty() and not log.verbose[5]
  collected = {"hyps": [], "refs": []}
  max_num_collected = 1
  if dataset:
    dataset.init_seq_order(epoch=1)
  else:
    refs = sorted(refs.items(), key=lambda item: len(item[1]))
  while True:
    if seq_idx > options.endseq:
      break
    if dataset:
      if not dataset.is_less_than_num_seqs(seq_idx):
        break
      dataset.load_seqs(seq_idx, seq_idx + 1)
      complete_frac = dataset.get_complete_frac(seq_idx)
      seq_tag = dataset.get_tag(seq_idx)
      assert isinstance(seq_tag, str)
      ref = dataset.get_data(seq_idx, options.key)
      if isinstance(ref, numpy.ndarray):
        assert ref.shape == ()
        ref = ref.flatten()[0]  # get the entry itself (str or bytes)
      if isinstance(ref, bytes):
        ref = ref.decode("utf8")
      assert isinstance(ref, str)
      try:
        num_seqs_s = str(dataset.num_seqs)
      except NotImplementedError:
        try:
          num_seqs_s = "~%i" % dataset.estimated_num_seqs
        except TypeError:  # a number is required, not NoneType
          num_seqs_s = "?"
    else:
      if seq_idx >= len(refs):
        break
      complete_frac = (seq_idx + 1) / float(len(refs))
      seq_tag, ref = refs[seq_idx]
      assert isinstance(seq_tag, str)
      assert isinstance(ref, str)
      num_seqs_s = str(len(refs))

    start_elapsed = time.time() - start_time
    progress_prefix = "%i/%s (WER %.02f%%)" % (seq_idx, num_seqs_s, wer * 100)
    progress = "%s (%.02f%%)" % (progress_prefix, complete_frac * 100)
    if complete_frac > 0:
      total_time_estimated = start_elapsed / complete_frac
      remaining_estimated = total_time_estimated - start_elapsed
      progress += " (%s)" % hms(remaining_estimated)

    remaining_hyp_seq_tags.remove(seq_tag)
    hyp = hyps[seq_tag]
    seq_len_stats["hyps"].collect([len(hyp)])
    seq_len_stats["refs"].collect([len(ref)])
    collected["hyps"].append(hyp)
    collected["refs"].append(ref)

    if len(collected["hyps"]) >= max_num_collected:
      wer = wer_compute.step(session, **collected)
      del collected["hyps"][:]
      del collected["refs"][:]

    if interactive:
      Util.progress_bar_with_time(complete_frac, prefix=progress_prefix)
    elif log.verbose[5]:
      print(progress_prefix, "seq tag %r, ref/hyp len %i/%i chars" % (seq_tag, len(ref), len(hyp)))
    seq_idx += 1
  if len(collected["hyps"]) > 0:
    wer = wer_compute.step(session, **collected)
  print("Done. Num seqs %i. Total time %s." % (
    seq_idx, hms(time.time() - start_time)), file=log.v1)
  print("Remaining num hyp seqs %i." % (len(remaining_hyp_seq_tags),), file=log.v1)
  if dataset:
    print("More seqs which we did not dumped: %s." % dataset.is_less_than_num_seqs(seq_idx), file=log.v1)
  for key in ["hyps", "refs"]:
    seq_len_stats[key].dump(stream_prefix="Seq-length %r %r " % (key, options.key), stream=log.v2)
  if options.expect_full:
    assert not remaining_hyp_seq_tags, "There are still remaining hypotheses."
  return wer


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
  config.set("task", "calculate_wer")
  config.set("log", None)
  config.set("log_verbosity", log_verbosity)
  config.set("use_tensorflow", True)
  rnn.init_log()
  print("Returnn calculate-word-error-rate starting up.", file=log.v1)
  rnn.returnn_greeting()
  rnn.init_backend_engine()
  assert Util.BackendEngine.is_tensorflow_selected(), "this is only for TensorFlow"
  rnn.init_faulthandler()
  rnn.init_config_json_network()
  rnn.print_task_properties()


def load_hyps_refs(filename):
  """
  :param str filename:
  :return: dict of seq_tag -> ref
  :rtype: dict[str,str]
  """
  content = eval(open(filename).read())
  # See dump-dataset-raw-strings.py.
  # We expect that it is stored as a dict.
  assert isinstance(content, dict)
  assert len(content) > 0
  example_hyp = next(iter(content.items()))
  assert isinstance(example_hyp[0], str)  # seq tag
  assert isinstance(example_hyp[1], str)  # hyp
  return content


def main(argv):
  argparser = argparse.ArgumentParser(description='Dump something from dataset.')
  argparser.add_argument('--config', help="filename to config-file. will use dataset 'eval' from it")
  argparser.add_argument("--dataset", help="dataset, overwriting config")
  argparser.add_argument("--refs", help="same format as hyps. alternative to providing dataset/config")
  argparser.add_argument("--hyps", help="hypotheses, dumped via search in py format")
  argparser.add_argument('--startseq', type=int, default=0, help='start seq idx (inclusive) (default: 0)')
  argparser.add_argument('--endseq', type=int, default=-1, help='end seq idx (inclusive) or -1 (default: -1)')
  argparser.add_argument("--key", default="raw", help="data-key, e.g. 'data' or 'classes'. (default: 'raw')")
  argparser.add_argument("--verbosity", default=4, type=int, help="5 for all seqs (default: 4)")
  argparser.add_argument("--out", help="if provided, will write WER% (as string) to this file")
  argparser.add_argument("--expect_full", action="store_true", help="full dataset should be scored")
  args = argparser.parse_args(argv[1:])
  assert args.config or args.dataset or args.refs

  init(config_filename=args.config, log_verbosity=args.verbosity)
  dataset = None
  refs = None
  if args.refs:
    refs = load_hyps_refs(args.refs)
  elif args.dataset:
    dataset = init_dataset(args.dataset)
  elif config.value("wer_data", "eval") in ["train", "dev", "eval"]:
    dataset = init_dataset(config.opt_typed_value(config.value("search_data", "eval")))
  else:
    dataset = init_dataset(config.opt_typed_value("wer_data"))
  hyps = load_hyps_refs(args.hyps)

  global wer_compute
  wer_compute = WerComputeGraph()
  with tf.Session(config=tf.ConfigProto(device_count={"GPU": 0})) as _session:
    global session
    session = _session
    session.run(tf.global_variables_initializer())
    try:
      wer = calc_wer_on_dataset(dataset=dataset, refs=refs, options=args, hyps=hyps)
      print("Final WER: %.02f%%" % (wer * 100), file=log.v1)
      if args.out:
        with open(args.out, "w") as output_file:
          output_file.write("%.02f\n" % (wer * 100))
        print("Wrote WER%% to %r." % args.out)
    except KeyboardInterrupt:
      print("KeyboardInterrupt")
      sys.exit(1)
    finally:
      rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
