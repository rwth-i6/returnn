#!/usr/bin/env python3

"""
Calculates word error rate (WER).
"""

from __future__ import print_function

import os
import sys
import time
import tensorflow as tf
import numpy
import typing

import _setup_returnn_env  # noqa
import returnn.__main__ as rnn
from returnn.log import log
import argparse
from returnn.util.basic import Stats, hms
from returnn.datasets.basic import Dataset, init_dataset
import returnn.util.basic as util
import returnn.tf.compat as tf_compat
import returnn.tf.util.basic as tf_util


class WerComputeGraph:
  """
  Creates TF computation graph to calculate the WER.
  We accumulate the absolute number of edits and normalize by the accumulated seq lens.
  """

  def __init__(self):
    self.hyps = tf_compat.v1.placeholder(tf.string, [None])
    self.refs = tf_compat.v1.placeholder(tf.string, [None])
    self.wer, self.ref_num_words = tf_util.string_words_calc_wer(hyps=self.hyps, refs=self.refs)
    self.total_wer_var = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
    self.total_ref_num_words_var = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
    self.update_total_wer = self.total_wer_var.assign_add(tf.reduce_sum(self.wer))
    self.update_ref_num_words = self.total_ref_num_words_var.assign_add(tf.reduce_sum(self.ref_num_words))
    self.updated_normalized_wer = (
      tf.cast(self.update_total_wer, tf.float32) / tf.cast(self.update_ref_num_words, tf.float32))

  # noinspection PyShadowingNames
  def step(self, session, hyps, refs):
    """
    :param tf.compat.v1.Session session:
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
  interactive = util.is_tty() and not log.verbose[5]
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
      util.progress_bar_with_time(complete_frac, prefix=progress_prefix)
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


config = None  # type: typing.Optional["returnn.config.Config"]


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
  assert util.BackendEngine.is_tensorflow_selected(), "this is only for TensorFlow"
  rnn.init_faulthandler()
  rnn.init_config_json_network()
  rnn.print_task_properties()


def load_hyps_refs(filename):
  """
  :param str filename:
  :return: dict of seq_tag -> ref
  :rtype: dict[str,str]
  """
  if filename.endswith(".gz"):
    import gzip
    content_str = gzip.open(filename, "rt").read()
  else:
    content_str = open(filename).read()
  content = eval(content_str)
  # See dump-dataset-raw-strings.py.
  # We expect that it is stored as a dict.
  assert isinstance(content, dict)
  assert len(content) > 0
  example_hyp = next(iter(content.items()))
  assert isinstance(example_hyp[0], str)  # seq tag
  if isinstance(example_hyp[1], list):
    assert isinstance(example_hyp[1][0][1], str)
    # n-best list output format needs to be converted first
    # always pick the best (first) entry from the list (which is [(score, text), ...]
    content = {seq_tag: nbest_list[0][1] for seq_tag, nbest_list in content.items()}
  else:
    assert isinstance(example_hyp[1], str)
  return content


wer_compute = None  # type: typing.Optional[WerComputeGraph]
session = None  # type: typing.Optional[tf.compat.v1.Session]


def main(argv):
  """
  Main entry.
  """
  arg_parser = argparse.ArgumentParser(description='Dump something from dataset.')
  arg_parser.add_argument('--config', help="filename to config-file. will use dataset 'eval' from it")
  arg_parser.add_argument("--dataset", help="dataset, overwriting config")
  arg_parser.add_argument("--refs", help="same format as hyps. alternative to providing dataset/config")
  arg_parser.add_argument("--hyps", help="hypotheses, dumped via search in py format")
  arg_parser.add_argument('--startseq', type=int, default=0, help='start seq idx (inclusive) (default: 0)')
  arg_parser.add_argument('--endseq', type=int, default=-1, help='end seq idx (inclusive) or -1 (default: -1)')
  arg_parser.add_argument("--key", default="raw", help="data-key, e.g. 'data' or 'classes'. (default: 'raw')")
  arg_parser.add_argument("--verbosity", default=4, type=int, help="5 for all seqs (default: 4)")
  arg_parser.add_argument("--out", help="if provided, will write WER% (as string) to this file")
  arg_parser.add_argument("--expect_full", action="store_true", help="full dataset should be scored")
  args = arg_parser.parse_args(argv[1:])
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
  with tf_compat.v1.Session(config=tf_compat.v1.ConfigProto(device_count={"GPU": 0})) as _session:
    global session
    session = _session
    session.run(tf_compat.v1.global_variables_initializer())
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
