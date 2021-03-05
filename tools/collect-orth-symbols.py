#!/usr/bin/env python

"""
Collect orth symbols.
"""

from __future__ import print_function

import os
import sys

import _setup_returnn_env  # noqa
import returnn.__main__ as rnn
from returnn.log import log
from returnn.config import Config
import argparse
from returnn.util.basic import hms, human_size, parse_orthography, parse_orthography_into_symbols, unicode
import gzip
from xml.etree import ElementTree
import wave
import time


def found_sub_seq(sub_seq, seq):
  """
  :param list[str] sub_seq:
  :param list[str] seq:
  :rtype: bool
  """
  # Very inefficient naive implementation:
  for i in range(len(seq)):
    if seq[i:i+len(sub_seq)] == sub_seq:
      return True
  return False


def iter_dataset(dataset, options, callback):
  """
  :type dataset: Dataset.Dataset
  """
  dataset.init_seq_order(epoch=1)
  assert "orth" in dataset.get_target_list()

  seq_idx = 0
  while dataset.is_less_than_num_seqs(seq_idx):
    dataset.load_seqs(seq_idx, seq_idx)

    frame_len = dataset.get_seq_length(seq_idx)["data"]
    orth = dataset.get_targets("orth", seq_idx)
    callback(frame_len=frame_len, orth=orth)

    seq_idx += 1


def get_wav_time_len(filename):
  """
  :param str filename:
  :rtype: float
  """
  f = wave.open(filename)
  num_frames = f.getnframes()
  frame_rate = f.getframerate()
  f.close()
  return num_frames / float(frame_rate)


def iter_bliss(filename, options, callback):
  corpus_file = open(filename, 'rb')
  if filename.endswith(".gz"):
    corpus_file = gzip.GzipFile(fileobj=corpus_file)

  def getelements(tag):
    """Yield *tag* elements from *filename_or_file* xml incrementaly."""
    context = iter(ElementTree.iterparse(corpus_file, events=('start', 'end')))
    _, root = next(context) # get root element
    tree = [root]
    for event, elem in context:
      if event == "start":
        tree += [elem]
      elif event == "end":
        assert tree[-1] is elem
        tree = tree[:-1]
      if event == 'end' and elem.tag == tag:
        yield tree, elem
        root.clear() # free memory

  time_via_wav = False

  for tree, elem in getelements("segment"):
    if options.collect_time:
      start = float(elem.attrib.get('start', 0))
      if "end" in elem.attrib:
        end = float(elem.attrib['end'])
      else:
        if not time_via_wav:
          time_via_wav = True
          print("Time will be read from WAV recordings. Can be slow. Maybe use `--collect_time False`.", file=log.v3)
        rec_elem = tree[-1]
        assert rec_elem.tag == "recording"
        wav_filename = rec_elem.attrib["audio"]
        end = get_wav_time_len(wav_filename)
      assert end > start
      frame_len = (end - start) * (1000.0 / options.frame_time)
    else:
      frame_len = 0
    elem_orth = elem.find("orth")
    orth_raw = elem_orth.text or ""  # should be unicode
    orth_split = orth_raw.split()
    orth = " ".join(orth_split)

    callback(frame_len=frame_len, orth=orth)


def iter_txt(filename, options, callback):
  f = open(filename, 'rb')
  if filename.endswith(".gz"):
    f = gzip.GzipFile(fileobj=f)

  if options.collect_time:
    print("No time-info in txt.", file=log.v3)
    options.collect_time = False

  for line in f:
    line = line.strip()
    if not line:
      continue

    callback(frame_len=0, orth=line)


def collect_stats(options, iter_corpus):
  """
  :param options: argparse.Namespace
  """
  orth_symbols_filename = options.output
  if orth_symbols_filename:
    assert not os.path.exists(orth_symbols_filename)

  class Stats:
    count = 0
    process_last_time = time.time()
    total_frame_len = 0
    total_orth_len = 0
    orth_syms_set = set()

  if options.add_numbers:
    Stats.orth_syms_set.update(map(chr, list(range(ord("0"), ord("9") + 1))))
  if options.add_lower_alphabet:
    Stats.orth_syms_set.update(map(chr, list(range(ord("a"), ord("z") + 1))))
  if options.add_upper_alphabet:
    Stats.orth_syms_set.update(map(chr, list(range(ord("A"), ord("Z") + 1))))

  def cb(frame_len, orth):
    if frame_len >= options.max_seq_frame_len:
      return
    orth_syms = parse_orthography(orth)
    if len(orth_syms) >= options.max_seq_orth_len:
      return

    Stats.count += 1
    Stats.total_frame_len += frame_len

    if options.dump_orth_syms:
      print("Orth:", "".join(orth_syms), file=log.v3)
    if options.filter_orth_sym:
      if options.filter_orth_sym in orth_syms:
        print("Found orth:", "".join(orth_syms), file=log.v3)
    if options.filter_orth_syms_seq:
      filter_seq = parse_orthography_into_symbols(options.filter_orth_syms_seq)
      if found_sub_seq(filter_seq, orth_syms):
        print("Found orth:", "".join(orth_syms), file=log.v3)
    Stats.orth_syms_set.update(orth_syms)
    Stats.total_orth_len += len(orth_syms)

    # Show some progress if it takes long.
    if time.time() - Stats.process_last_time > 2:
      Stats.process_last_time = time.time()
      if options.collect_time:
        print(
          "Collect process, total frame len so far:", hms(Stats.total_frame_len * (options.frame_time / 1000.0)),
          file=log.v3)
      else:
        print("Collect process, total orth len so far:", human_size(Stats.total_orth_len), file=log.v3)

  iter_corpus(cb)

  if options.remove_symbols:
    filter_syms = parse_orthography_into_symbols(options.remove_symbols)
    Stats.orth_syms_set -= set(filter_syms)

  if options.collect_time:
    print(
      "Total frame len:", Stats.total_frame_len, "time:", hms(Stats.total_frame_len * (options.frame_time / 1000.0)),
      file=log.v3)
  else:
    print("No time stats (--collect_time False).", file=log.v3)
  print("Total orth len:", Stats.total_orth_len, "(%s)" % human_size(Stats.total_orth_len), end=' ', file=log.v3)
  if options.collect_time:
    print("fraction:", float(Stats.total_orth_len) / Stats.total_frame_len, file=log.v3)
  else:
    print("", file=log.v3)
  print("Average orth len:", float(Stats.total_orth_len) / Stats.count, file=log.v3)
  print("Num symbols:", len(Stats.orth_syms_set), file=log.v3)

  if orth_symbols_filename:
    orth_syms_file = open(orth_symbols_filename, "wb")
    for orth_sym in sorted(Stats.orth_syms_set):
      orth_syms_file.write(b"%s\n" % unicode(orth_sym).encode("utf8"))
    orth_syms_file.close()
    print("Wrote orthography symbols to", orth_symbols_filename, file=log.v3)
  else:
    print("Provide --output to save the symbols.", file=log.v3)


def init(config_filename=None):
  rnn.init_better_exchook()
  rnn.init_thread_join_hack()
  if config_filename:
    rnn.init_config(config_filename, command_line_options=[])
    rnn.init_log()
  else:
    log.initialize()
  print("RETURNN collect-orth-symbols starting up.", file=log.v3)
  rnn.init_faulthandler()
  if config_filename:
    rnn.init_config_json_network()
    rnn.init_data()
    rnn.print_task_properties()


def is_bliss(filename):
  try:
    corpus_file = open(filename, 'rb')
    if filename.endswith(".gz"):
      corpus_file = gzip.GzipFile(fileobj=corpus_file)
    context = iter(ElementTree.iterparse(corpus_file, events=('start', 'end')))
    _, root = next(context)  # get root element
    return True
  except IOError:  # 'Not a gzipped file' or so
    pass
  except ElementTree.ParseError:  # 'syntax error' or so
    pass
  return False


def is_crnn_config(filename):
  if filename.endswith(".gz"):
    return False
  try:
    config = Config()
    config.load_file(filename)
    return True
  except Exception:
    pass
  return False


def main(argv):
  argparser = argparse.ArgumentParser(description='Collect orth symbols.')
  argparser.add_argument('input', help="RETURNN config, Corpus Bliss XML or just txt-data")
  argparser.add_argument(
    '--frame_time', type=int, default=10, help='time (in ms) per frame. not needed for Corpus Bliss XML')
  argparser.add_argument('--collect_time', type=int, default=True, help="collect time info. can be slow in some cases")
  argparser.add_argument('--dump_orth_syms', action='store_true', help="dump all orthographies")
  argparser.add_argument('--filter_orth_sym', help="dump orthographies which match this filter")
  argparser.add_argument('--filter_orth_syms_seq', help="dump orthographies which match this filter")
  argparser.add_argument(
    '--max_seq_frame_len', type=int, default=float('inf'), help="collect only orthographies <= this max frame len")
  argparser.add_argument(
    '--max_seq_orth_len', type=int, default=float('inf'), help="collect only orthographies <= this max orth len")
  argparser.add_argument('--add_numbers', type=int, default=True, help="add chars 0-9 to orth symbols")
  argparser.add_argument('--add_lower_alphabet', type=int, default=True, help="add chars a-z to orth symbols")
  argparser.add_argument('--add_upper_alphabet', type=int, default=True, help="add chars A-Z to orth symbols")
  argparser.add_argument('--remove_symbols', default="(){}$", help="remove these chars from orth symbols")
  argparser.add_argument('--output', help='where to store the symbols (default: dont store)')
  args = argparser.parse_args(argv[1:])

  bliss_filename = None
  crnn_config_filename = None
  txt_filename = None
  if is_bliss(args.input):
    bliss_filename = args.input
  elif is_crnn_config(args.input):
    crnn_config_filename = args.input
  else:  # treat just as txt
    txt_filename = args.input
  init(config_filename=crnn_config_filename)

  if bliss_filename:
    iter_corpus = lambda cb: iter_bliss(bliss_filename, options=args, callback=cb)
  elif txt_filename:
    iter_corpus = lambda cb: iter_txt(txt_filename, options=args, callback=cb)
  else:
    iter_corpus = lambda cb: iter_dataset(rnn.train_data, options=args, callback=cb)
  collect_stats(args, iter_corpus)

  if crnn_config_filename:
    rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
