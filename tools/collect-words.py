#!/usr/bin/env python

from __future__ import print_function

import os
import sys

my_dir = os.path.dirname(os.path.abspath(__file__))
returnn_dir = os.path.dirname(my_dir)
sys.path.append(returnn_dir)

import rnn
from Log import log
from Config import Config
import argparse
from Util import hms, human_size, parse_orthography, parse_orthography_into_symbols, unicode
from LmDataset import Lexicon
import gzip
import xml.etree.ElementTree as etree
from pprint import pprint
import wave
import time


def iter_dataset(dataset, options, callback):
  """
  :type dataset: Dataset.Dataset
  """
  dataset.init_seq_order(epoch=1)
  assert "orth" in dataset.get_target_list()

  seq_idx = 0
  while dataset.is_less_than_num_seqs(seq_idx):
    dataset.load_seqs(seq_idx, seq_idx)

    orth = dataset.get_targets("orth", seq_idx)
    callback(orth=orth)

    seq_idx += 1


def iter_bliss(filename, options, callback):
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
    elem_orth = elem.find("orth")
    orth_raw = elem_orth.text  # should be unicode
    orth_split = orth_raw.split()
    orth = " ".join(orth_split)

    callback(orth=orth)


def iter_txt(filename, options, callback):
  f = open(filename, 'rb')
  if filename.endswith(".gz"):
    f = gzip.GzipFile(fileobj=f)

  for l in f:
    l = l.strip()
    if not l:
      continue

    callback(orth=l)


class CollectCorpusStats:
  def __init__(self, options, iter_corpus):
    """
    :param options: argparse.Namespace
    """

    self.options = options
    self.seq_count = 0
    self.words = set()
    self.total_word_len = 0
    self.process_last_time = time.time()

    iter_corpus(self._callback)

    print("Total word len:", self.total_word_len, "(%s)" % human_size(self.total_word_len), file=log.v3)
    print("Average orth len:", float(self.total_word_len) / self.seq_count, file=log.v3)
    print("Num word symbols:", len(self.words), file=log.v3)

  def _callback(self, orth):
    orth_words = parse_orthography(orth, prefix=[], postfix=[], word_based=True)

    self.seq_count += 1

    if self.options.dump_orth:
      print("Orth:", orth_words, file=log.v3)
    self.words.update(orth_words)
    self.total_word_len += len(orth_words)

    # Show some progress if it takes long.
    if time.time() - self.process_last_time > 2:
      self.process_last_time = time.time()
      print("Collect process, total word len so far:", human_size(self.total_word_len), file=log.v3)


def init(configFilename=None):
  rnn.init_better_exchook()
  rnn.init_thread_join_hack()
  if configFilename:
    rnn.init_config(configFilename, command_line_options=[])
    rnn.init_log()
  else:
    log.initialize()
  print("Returnn collect-words starting up.", file=log.v3)
  rnn.init_faulthandler()
  if configFilename:
    rnn.init_config_json_network()
    rnn.init_data()
    rnn.print_task_properties()


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
  argparser.add_argument('input', help="CRNN config, Corpus Bliss XML or just txt-data")
  argparser.add_argument("--dump_orth", action="store_true")
  argparser.add_argument("--lexicon")
  args = argparser.parse_args(argv[1:])

  bliss_filename = None
  crnn_config_filename = None
  txt_filename = None
  if is_bliss(args.input):
    bliss_filename = args.input
    print("Read Bliss corpus:", bliss_filename)
  elif is_crnn_config(args.input):
    crnn_config_filename = args.input
    print("Read corpus from Returnn config:", crnn_config_filename)
  else:  # treat just as txt
    txt_filename = args.input
    print("Read corpus from txt-file:", txt_filename)
  init(configFilename=crnn_config_filename)

  if bliss_filename:
    iter_corpus = lambda cb: iter_bliss(bliss_filename, options=args, callback=cb)
  elif txt_filename:
    iter_corpus = lambda cb: iter_txt(txt_filename, options=args, callback=cb)
  else:
    iter_corpus = lambda cb: iter_dataset(rnn.train_data, options=args, callback=cb)
  corpus_stats = CollectCorpusStats(args, iter_corpus)

  if args.lexicon:
    print("Lexicon:", args.lexicon)
    lexicon = Lexicon(args.lexicon)
    print("Words not in lexicon:")
    c = 0
    for w in sorted(corpus_stats.words):
      if w not in lexicon.lemmas:
        print(w)
        c += 1
    print("Count: %i (%f%%)" % (c, 100. * float(c) / len(corpus_stats.words)))
  else:
    print("No lexicon provided (--lexicon).")

  if crnn_config_filename:
    rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
