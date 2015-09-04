#!/usr/bin/env python

import os
import sys
import rnn
from Log import log
import argparse
from Util import hms, parse_orthography, parse_orthography_into_symbols
import gzip
import xml.etree.ElementTree as etree
from pprint import pprint
import wave
import time


def found_sub_seq(sub_seq, seq):
  # Very inefficient naive implementation:
  for i in range(len(seq)):
    if seq[i:i+len(sub_seq)] == sub_seq:
      return True
  return False


def iter_dataset(dataset, callback):
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
  f = wave.open(filename)
  num_frames = f.getnframes()
  framerate = f.getframerate()
  f.close()
  return num_frames / float(framerate)


def iter_bliss(filename, frame_time, callback):
  corpus_file = open(filename, 'rb')
  if filename.endswith(".gz"):
    corpus_file = gzip.GzipFile(fileobj=corpus_file)

  def getelements(tag):
    """Yield *tag* elements from *filename_or_file* xml incrementaly."""
    context = iter(etree.iterparse(corpus_file, events=('start', 'end')))
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

  for tree, elem in getelements("segment"):
    start = float(elem.attrib.get('start', 0))
    if "end" in elem.attrib:
      end = float(elem.attrib['end'])
    else:
      rec_elem = tree[-1]
      assert rec_elem.tag == "recording"
      wav_filename = rec_elem.attrib["audio"]
      end = get_wav_time_len(wav_filename)
    assert end > start
    frame_len = (end - start) * (1000.0 / frame_time)
    elem_orth = elem.find("orth")
    orth_raw = elem_orth.text  # should be unicode
    orth_split = orth_raw.split()
    orth = " ".join(orth_split)

    callback(frame_len=frame_len, orth=orth)


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
    Stats.orth_syms_set.update(map(chr, range(ord("0"), ord("9") + 1)))
  if options.add_lower_alphabet:
    Stats.orth_syms_set.update(map(chr, range(ord("a"), ord("z") + 1)))
  if options.add_upper_alphabet:
    Stats.orth_syms_set.update(map(chr, range(ord("A"), ord("Z") + 1)))

  def cb(frame_len, orth):
    if frame_len >= options.max_seq_frame_len:
      return

    Stats.count += 1
    Stats.total_frame_len += frame_len

    orth_syms = parse_orthography(orth)
    if options.dump_orth_syms:
      print >> log.v3, "Orth:", "".join(orth_syms)
    if options.filter_orth_sym:
      if options.filter_orth_sym in orth_syms:
        print >> log.v3, "Found orth:", "".join(orth_syms)
    if options.filter_orth_syms_seq:
      filter_seq = parse_orthography_into_symbols(options.filter_orth_syms_seq)
      if found_sub_seq(filter_seq, orth_syms):
        print >> log.v3, "Found orth:", "".join(orth_syms)
    Stats.orth_syms_set.update(orth_syms)
    Stats.total_orth_len += len(orth_syms)

    if time.time() - Stats.process_last_time > 2:
      Stats.process_last_time = time.time()
      print >> log.v3, "Collect process, total frame len so far:", hms(Stats.total_frame_len * (options.frame_time / 1000.0))

  iter_corpus(cb)

  if options.remove_symbols:
    filter_syms = parse_orthography_into_symbols(options.remove_symbols)
    Stats.orth_syms_set -= set(filter_syms)

  print >> log.v3, "Total frame len:", Stats.total_frame_len, "time:", hms(Stats.total_frame_len * (options.frame_time / 1000.0))
  print >> log.v3, "Total orth len:", Stats.total_orth_len, "fraction:", float(Stats.total_orth_len) / Stats.total_frame_len
  print >> log.v3, "Num symbols:", len(Stats.orth_syms_set)

  if orth_symbols_filename:
    orth_syms_file = open(orth_symbols_filename, "wb")
    for orth_sym in sorted(Stats.orth_syms_set):
      orth_syms_file.write("%s\n" % unicode(orth_sym).encode("utf8"))
    orth_syms_file.close()
    print >> log.v3, "Wrote orthography symbols to", orth_symbols_filename
  else:
    print >> log.v3, "Provide --output to save the symbols."


def init(configFilename=None):
  rnn.initBetterExchook()
  rnn.initThreadJoinHack()
  if configFilename:
    rnn.initConfig(configFilename, commandLineOptions=[])
    rnn.initLog()
  else:
    log.initialize()
  print >> log.v3, "CRNN collect-orth-symbols starting up."
  rnn.initFaulthandler()
  if configFilename:
    rnn.initConfigJsonNetwork()
    rnn.initData()
    rnn.printTaskProperties()


def is_bliss(filename):
  try:
    corpus_file = open(filename, 'rb')
    if filename.endswith(".gz"):
      corpus_file = gzip.GzipFile(fileobj=corpus_file)
    context = iter(etree.iterparse(corpus_file, events=('start', 'end')))
    _, root = next(context)  # get root element
    return True
  except IOError:  # 'Not a gzipped file' or so
    pass
  except etree.ParseError:  # 'syntax error' or so
    pass
  return False


def main(argv):
  argparser = argparse.ArgumentParser(description='Collect orth symbols.')
  argparser.add_argument('input', help="either crnn config or Bliss xml")
  argparser.add_argument('--frame_time', type=int, default=10, help='time (in ms) per frame')
  argparser.add_argument('--dump_orth_syms', action='store_true')
  argparser.add_argument('--filter_orth_sym')
  argparser.add_argument('--filter_orth_syms_seq')
  argparser.add_argument('--max_seq_frame_len', type=int, default=float('inf'))
  argparser.add_argument('--add_numbers', type=bool, default=True)
  argparser.add_argument('--add_lower_alphabet', type=bool, default=True)
  argparser.add_argument('--add_upper_alphabet', type=bool, default=True)
  argparser.add_argument('--remove_symbols', default="(){}$")
  argparser.add_argument('--output', help='where to store the symbols (default: dont store)')
  args = argparser.parse_args(argv[1:])

  if is_bliss(args.input):
    bliss_filename = args.input
    crnn_config_filename = None
  else:
    bliss_filename = None
    crnn_config_filename = args.input
  init(configFilename=crnn_config_filename)

  if bliss_filename:
    iter_corpus = lambda cb: iter_bliss(bliss_filename, frame_time=args.frame_time, callback=cb)
  else:
    iter_corpus = lambda cb: iter_dataset(rnn.train_data, callback=cb)
  collect_stats(args, iter_corpus)

  if crnn_config_filename:
    rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
