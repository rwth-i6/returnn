#!/usr/bin/env python

"""
Collect statistics about words in a corpus.
"""

from __future__ import annotations

import sys

import _setup_returnn_env  # noqa
import returnn.__main__ as rnn
from returnn.log import log
from returnn.config import Config
import argparse
from returnn.util.basic import human_size, parse_orthography
from returnn.datasets import Dataset
from returnn.datasets.lm import Lexicon
import gzip
from xml.etree import ElementTree
import time


def iter_dataset(dataset: Dataset, callback):
    """
    :param dataset:
    :param (*)->None callback:
    """
    dataset.init_seq_order(epoch=1)
    assert "orth" in dataset.get_target_list()

    seq_idx = 0
    while dataset.is_less_than_num_seqs(seq_idx):
        dataset.load_seqs(seq_idx, seq_idx)

        orth = dataset.get_data(seq_idx, "orth")
        callback(orth=orth)

        seq_idx += 1


def iter_bliss(filename, callback):
    """
    Iterate through a Sprint Bliss XML file.

    :param str filename:
    :param callback:
    """
    corpus_file = open(filename, "rb")
    if filename.endswith(".gz"):
        corpus_file = gzip.GzipFile(fileobj=corpus_file)

    # noinspection PyShadowingNames
    def get_elements(tag):
        """Yield *tag* elements from *filename_or_file* xml incrementally."""
        context = iter(ElementTree.iterparse(corpus_file, events=("start", "end")))
        _, root = next(context)  # get root element
        tree = [root]
        for event, elem in context:
            if event == "start":
                tree += [elem]
            elif event == "end":
                assert tree[-1] is elem
                tree = tree[:-1]
            if event == "end" and elem.tag == tag:
                yield tree, elem
                root.clear()  # free memory

    for tree, elem in get_elements("segment"):
        elem_orth = elem.find("orth")
        orth_raw = elem_orth.text or ""  # should be unicode
        orth_split = orth_raw.split()
        orth = " ".join(orth_split)

        callback(orth=orth)


def iter_txt(filename, callback):
    """
    Iterate through pure text file.

    :param str filename:
    :param callback:
    """
    f = open(filename, "rb")
    if filename.endswith(".gz"):
        f = gzip.GzipFile(fileobj=f)

    for line in f:
        line = line.strip()
        if not line:
            continue

        callback(orth=line)


class CollectCorpusStats:
    """
    Collect stats.
    """

    def __init__(self, options, iter_corpus):
        """
        :param options: argparse.Namespace
        :param iter_corpus:
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
        """
        :param str orth:
        """
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


def init(config_filename=None):
    """
    :param str config_filename:
    """
    rnn.init_better_exchook()
    rnn.init_thread_join_hack()
    if config_filename:
        rnn.init_config(config_filename, command_line_options=[])
        rnn.init_log()
    else:
        log.initialize()
    print("Returnn collect-words starting up.", file=log.v3)
    rnn.init_faulthandler()
    if config_filename:
        rnn.init_data()
        rnn.print_task_properties()


def is_bliss(filename):
    """
    :param str filename:
    :rtype: bool
    """
    try:
        corpus_file = open(filename, "rb")
        if filename.endswith(".gz"):
            corpus_file = gzip.GzipFile(fileobj=corpus_file)
        context = iter(ElementTree.iterparse(corpus_file, events=("start", "end")))
        _, root = next(context)  # get root element
        return True
    except IOError:  # 'Not a gzipped file' or so
        pass
    except ElementTree.ParseError:  # 'syntax error' or so
        pass
    return False


def is_returnn_config(filename):
    """
    :param str filename:
    :rtype: bool
    """
    if filename.endswith(".gz"):
        return False
    # noinspection PyBroadException
    try:
        config = Config()
        config.load_file(filename)
        return True
    except Exception:
        pass
    return False


def main(argv):
    """
    Main entry.
    """
    arg_parser = argparse.ArgumentParser(description="Collect orth symbols.")
    arg_parser.add_argument("input", help="RETURNN config, Corpus Bliss XML or just txt-data")
    arg_parser.add_argument("--dump_orth", action="store_true")
    arg_parser.add_argument("--lexicon")
    args = arg_parser.parse_args(argv[1:])

    bliss_filename = None
    crnn_config_filename = None
    txt_filename = None
    if is_bliss(args.input):
        bliss_filename = args.input
        print("Read Bliss corpus:", bliss_filename)
    elif is_returnn_config(args.input):
        crnn_config_filename = args.input
        print("Read corpus from RETURNN config:", crnn_config_filename)
    else:  # treat just as txt
        txt_filename = args.input
        print("Read corpus from txt-file:", txt_filename)
    init(config_filename=crnn_config_filename)

    if bliss_filename:

        def _iter_corpus(cb):
            return iter_bliss(bliss_filename, callback=cb)

    elif txt_filename:

        def _iter_corpus(cb):
            return iter_txt(txt_filename, callback=cb)

    else:

        def _iter_corpus(cb):
            return iter_dataset(rnn.train_data, callback=cb)

    corpus_stats = CollectCorpusStats(args, _iter_corpus)

    if args.lexicon:
        print("Lexicon:", args.lexicon)
        lexicon = Lexicon(args.lexicon)
        print("Words not in lexicon:")
        c = 0
        for w in sorted(corpus_stats.words):
            if w not in lexicon.lemmas:
                print(w)
                c += 1
        print("Count: %i (%f%%)" % (c, 100.0 * float(c) / len(corpus_stats.words)))
    else:
        print("No lexicon provided (--lexicon).")

    if crnn_config_filename:
        rnn.finalize()


if __name__ == "__main__":
    main(sys.argv)
