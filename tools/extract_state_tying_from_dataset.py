#!/usr/bin/env python3

"""
Extract state tying from dataset.
"""

from __future__ import print_function

import os
import gzip
from argparse import ArgumentParser
from pprint import pprint
from xml.etree import ElementTree
import collections
from collections import defaultdict
import typing

import _setup_returnn_env  # noqa
from returnn.datasets import init_dataset
from returnn.datasets.lm import Lexicon, AllophoneState
from returnn.log import log
from returnn.util.basic import uniq


def get_segment_name(tree):
  """
  :param tree:
  :return:
  """
  def _m(x):
    if "name" in x.attrib:
      return x.attrib["name"]
    if x.tag == "segment":
      return "1"
    assert False, "unknown name: %r, %r" % (x, vars(x))
  return "/".join(map(_m, tree))


def iter_bliss_orth(filename):
  """
  :param str filename:
  :return:
  """
  corpus_file = open(filename, 'rb')
  if filename.endswith(".gz"):
    corpus_file = gzip.GzipFile(fileobj=corpus_file)

  # noinspection PyShadowingNames
  def getelements(tag):
    """Yield *tag* elements from *filename_or_file* xml incrementally."""
    context = iter(ElementTree.iterparse(corpus_file, events=('start', 'end')))
    _, root = next(context)  # get root element
    tree = [root]
    for event, elem in context:
      if event == "start":
        tree += [elem]
      elif event == "end":
        assert tree[-1] is elem
        tree = tree[:-1]
      if event == 'end' and elem.tag == tag:
        yield tree, elem

  for tree, elem in getelements("segment"):
    elem_orth = elem.find("orth")
    orth_raw = elem_orth.text or ""  # should be unicode
    orth_split = orth_raw.split()
    orth = " ".join(orth_split)

    yield get_segment_name(tree + [elem]), orth


def iter_dataset_targets(dataset):
  """
  :type dataset: Dataset.Dataset
  """
  dataset.init_seq_order(epoch=1)
  seq_idx = 0
  while dataset.is_less_than_num_seqs(seq_idx):
    dataset.load_seqs(seq_idx, seq_idx + 1)
    segment_name = dataset.get_tag(seq_idx)
    targets = dataset.get_targets("classes", seq_idx)
    assert targets.ndim == 1  # sparse
    targets = targets.astype("int32")
    yield segment_name, targets
    seq_idx += 1


class OrthHandler:
  """
  Orthography handler.
  """

  allo_add_all = False  # only via lexicon

  def __init__(self, lexicon, si_label=None, allo_num_states=3, allo_context_len=1, allow_ci_in_words=True):
    """
    :param Lexicon lexicon:
    :param int si_label:
    :param int allo_num_states:
    :param int allo_context_len:
    :param bool allow_ci_in_words:
    """
    self.lexicon = lexicon
    self.phonemes = sorted(self.lexicon.phonemes.keys(), key=lambda s: self.lexicon.phonemes[s]["index"])
    self.word_boundary_phones = {-1: set(), 1: set()}
    self.phon_to_possible_ctx_via_lex = {-1: {}, 1: {}}
    for lemma in self.lexicon.lemmas.values():
      for pron in lemma["phons"]:
        phons = pron["phon"].split()
        assert phons
        self.word_boundary_phones[-1].add(phons[0])
        self.word_boundary_phones[1].add(phons[-1])
        for i in range(len(phons)):
          ps = [phons[i + j] if (0 <= (i + j) < len(phons)) else ""
                for j in [-1, 0, 1]]
          self.phon_to_possible_ctx_via_lex[1].setdefault(ps[1], set()).add(ps[2])
          self.phon_to_possible_ctx_via_lex[-1].setdefault(ps[1], set()).add(ps[0])
    for phone in self.lexicon.phoneme_list:
      if "" in self.phon_to_possible_ctx_via_lex[-1][phone]:
        self.phon_to_possible_ctx_via_lex[-1][phone].update(self.word_boundary_phones[1])
      if "" in self.phon_to_possible_ctx_via_lex[1][phone]:
        self.phon_to_possible_ctx_via_lex[1][phone].update(self.word_boundary_phones[-1])
    if allow_ci_in_words:
      for phone in self.lexicon.phoneme_list:
        self.phon_to_possible_ctx_via_lex[-1][phone].add("")
        self.phon_to_possible_ctx_via_lex[1][phone].add("")
    self.si_lemma = self.lexicon.lemmas["[SILENCE]"]
    self.si_phone = self.si_lemma["phons"][0]["phon"]  # type: str
    self.si_label = si_label
    self.allo_num_states = allo_num_states  # e.g. 3 -> 3-state HMM
    self.allo_context_len = allo_context_len  # e.g. 1 -> one left&right, i.e. triphone

  def expected_num_labels_for_monophone_state_tying(self):
    """
    Silence has 1 state, all others have allo_num_states.

    :rtype: int
    """
    num_phones = len(self.lexicon.phonemes)
    return (num_phones - 1) * self.allo_num_states + 1

  def iter_orth(self, orth):
    """
    :param str orth:
    :return: yields lemmas
    """
    symbols = list(orth.split())
    i = 0
    while i < len(symbols):
      symbol = symbols[i]
      try:
        lemma = self.lexicon.lemmas[symbol]
      except KeyError:
        if "/" in symbol:
          symbols[i:i + 1] = symbol.split("/")
          continue
        if "-" in symbol:
          symbols[i:i + 1] = symbol.split("-")
          continue
        raise
      i += 1
      yield lemma

  def _iter_possible_ctx(self, phon_id, direction):
    """
    :param str phon_id: e.g. "aa", "aw", "uh", "z", etc.
    :param int direction: 1 or -1
    :rtype: list[tuple[str]]
    """
    if self.lexicon.phonemes[phon_id]["variation"] == "none":
      return [()]
    if self.allo_add_all:
      res = [()]  # type: typing.List[typing.Tuple[str, ...]]
      res += [
        (p,)
        for p in sorted(self.lexicon.phonemes.keys())
        if self.lexicon.phonemes[p]["variation"] == "context"]
      return res
    return [
      ((p,) if p else ())
      for p in sorted(self.phon_to_possible_ctx_via_lex[direction][phon_id])]

  def num_states_for_phone(self, phon_id):
    """
    :param str phon_id:
    :return: number of allophone states for this phone
    :rtype: int
    """
    if phon_id == self.si_phone:
      return 1
    return self.allo_num_states

  def all_allophone_variations(self, phon, states=None, all_boundary_variations=False):
    """
    :param str phon:
    :param None|list[int] states: which states to yield for this phone
    :param bool all_boundary_variations:
    :return: yields AllophoneState's
    :rtype: list[AllophoneState]
    """
    if states is None:
      states = range(self.num_states_for_phone(phon))
    if all_boundary_variations:
      boundary_variations = [0, 1, 2, 3]
    else:
      boundary_variations = [0]
    for left_ctx in self._iter_possible_ctx(phon, -1):
      for right_ctx in self._iter_possible_ctx(phon, 1):
        for state in states:
          for boundary in boundary_variations:
            a = AllophoneState()
            a.id = phon
            a.context_history = left_ctx
            a.context_future = right_ctx
            a.state = state
            a.boundary = boundary
            if not all_boundary_variations:
              if not left_ctx:
                a.mark_initial()
              if not right_ctx:
                a.mark_final()
            yield a

  # noinspection PyMethodMayBeStatic
  def _phones_to_allos(self, phones):
    for p in phones:
      a = AllophoneState()
      a.id = p
      yield a

  def _allos_set_context(self, allos):
    if self.allo_context_len == 0:
      return
    ctx = []
    for a in allos:
      if self.lexicon.phonemes[a.id]["variation"] == "context":
        a.context_history = tuple(ctx)
        ctx += [a.id]
        ctx = ctx[-self.allo_context_len:]
      else:
        ctx = []
    ctx = []
    for a in reversed(allos):
      if self.lexicon.phonemes[a.id]["variation"] == "context":
        a.context_future = tuple(reversed(ctx))
        ctx += [a.id]
        ctx = ctx[-self.allo_context_len:]
      else:
        ctx = []

  def _allos_add_states(self, allos):
    for _a in allos:
      if _a.id == self.si_phone:
        yield _a
      else:  # non-silence
        for state in range(self.allo_num_states):
          a = AllophoneState()
          a.id = _a.id
          a.context_history = _a.context_history
          a.context_future = _a.context_future
          a.boundary = _a.boundary
          a.state = state
          yield a

  def orth_to_allophone_states(self, orth):
    """
    :param str orth: orthography as a str. orth.split() should give words in the lexicon
    :rtype: list[AllophoneState]
    :returns allophone state list. those will have repetitions etc
    """
    allos = []
    for lemma in self.iter_orth(orth):
      assert len(lemma["phons"]) == 1, "TODO..."
      phon = lemma["phons"][0]
      l_allos = list(self._phones_to_allos(phon["phon"].split()))
      l_allos[0].mark_initial()
      l_allos[-1].mark_final()
      allos += l_allos
    self._allos_set_context(allos)
    allos = list(self._allos_add_states(allos))
    return allos


def main():
  """
  Main entry.
  """
  arg_parser = ArgumentParser()
  arg_parser.add_argument("--action")
  arg_parser.add_argument("--print_seq", action='store_true')
  arg_parser.add_argument("--print_allos", action='store_true')
  arg_parser.add_argument("--print_targets", action='store_true')
  arg_parser.add_argument("--dataset")
  arg_parser.add_argument("--corpus")
  arg_parser.add_argument("--lexicon", help="filename")
  arg_parser.add_argument("--silence", type=int, help="index")
  arg_parser.add_argument("--context", default=1, type=int)
  arg_parser.add_argument("--hmm_states", default=3, type=int)
  arg_parser.add_argument("--state_tying_type", help="'monophone' or 'full'")
  arg_parser.add_argument("--state_tying_output", help="filename")
  arg_parser.add_argument("--allo_add_all", action="store_true")
  args = arg_parser.parse_args()

  dataset = init_dataset(args.dataset) if args.dataset else None
  corpus = dict(iter_bliss_orth(filename=args.corpus)) if args.corpus else None
  lexicon = Lexicon(filename=args.lexicon) if args.lexicon else None
  silence_label = args.silence

  if args.action == "show_corpus":
    pprint(corpus)
    return

  print("Num phones: %i" % len(lexicon.phonemes), file=log.v1)
  print("Phones: %r" % sorted(lexicon.phonemes.keys()), file=log.v1)

  orth_handler = OrthHandler(
    lexicon=lexicon,
    allo_context_len=args.context,
    allo_num_states=args.hmm_states)
  map_idx_to_allo = defaultdict(set)  # type: typing.Dict[int, typing.Set[AllophoneState]]
  map_allo_to_idx = {}  # type: typing.Dict[AllophoneState, int]
  if args.allo_add_all:
    orth_handler.allo_add_all = True

  print("Num HMM states: %i" % orth_handler.allo_num_states, file=log.v1)
  if args.state_tying_type == "monophone":
    print("Monophone state tying.", file=log.v1)
    num_labels = orth_handler.expected_num_labels_for_monophone_state_tying()
    all_label_idx_are_used = True
  elif args.state_tying_type == "full":
    print("Full state tying.", file=log.v1)
    phone_idxs = {k: i + 1 for (i, k) in enumerate(lexicon.phoneme_list)}  # +1 to keep 0 reserved as the term-symbol
    for phon in lexicon.phoneme_list:
      for allo in orth_handler.all_allophone_variations(phon, all_boundary_variations=True):
        allo_idx = allo.index(
          phone_idxs=phone_idxs,
          num_states=orth_handler.allo_num_states,
          context_length=orth_handler.allo_context_len)
        map_idx_to_allo[allo_idx].add(allo)
    num_labels = max(map_idx_to_allo.keys()) + 1
    all_label_idx_are_used = False
  else:
    raise Exception("invalid state tying type %r" % args.state_tying_type)
  print("Num labels: %i" % num_labels, file=log.v1)

  if dataset:
    count = 0
    for segment_name, targets in iter_dataset_targets(dataset):
      count += 1
      if silence_label is None or count == 1:
        likely_silence_label = collections.Counter(targets).most_common(1)[0][0]
        if silence_label is None:
          silence_label = likely_silence_label
        if silence_label != likely_silence_label:
          print("warning: silence %i but likely %i" % (silence_label, likely_silence_label), file=log.v2)
        print("Silence label: %i" % silence_label, file=log.v1)
        orth_handler.si_label = silence_label
        # Monophone state tying:
        for allo in orth_handler.all_allophone_variations(orth_handler.si_phone):
          map_idx_to_allo[silence_label].add(allo)
          map_allo_to_idx[allo] = silence_label
      assert segment_name in corpus
      orth = corpus[segment_name]
      allo_states = orth_handler.orth_to_allophone_states(orth=orth)
      if args.print_seq:
        print("%r %r" % (segment_name, orth))
      if args.print_allos:
        print("  allophone state seq: %r" % allo_states)
      tgt_seq = [t for t in uniq(targets) if t != silence_label]
      if args.print_targets:
        print("  target seq: %r" % (tgt_seq,))
      assert len(allo_states) == len(tgt_seq), "check --hmm_states or so"
      for allo, t in zip(allo_states, tgt_seq):
        allo.boundary = 0  # do not differ between boundaries
        allos = map_idx_to_allo[t]
        if allo in map_allo_to_idx:
          assert allo in allos, "bad mapping"
        else:
          assert allo not in allos
          allos.add(allo)
          map_allo_to_idx[allo] = t
      if len(map_idx_to_allo) >= num_labels:
        assert len(map_idx_to_allo) == num_labels
        assert 0 in map_idx_to_allo
        assert num_labels - 1 in map_idx_to_allo
        print("Finished with uniq mapping after %i sequences." % count, file=log.v1)
        break
      if count % 100 == 0:
        print("Have indices: %i (num labels: %i)" % (len(map_idx_to_allo), num_labels), file=log.v1)

    print("Finished. Have indices: %i (num labels: %i)" % (len(map_idx_to_allo), num_labels), file=log.v1)
    if len(map_idx_to_allo) < num_labels:
      found = []
      not_found = []
      for p in sorted(lexicon.phonemes.keys()):
        allo = AllophoneState(p, state=0)
        if allo in map_allo_to_idx:
          found.append(p)
        else:
          not_found.append(p)
      print("Phonemes found: %r" % found)
      print("Phonemes not found: %r" % not_found)

  if args.state_tying_output:
    assert not os.path.exists(args.state_tying_output)
    if all_label_idx_are_used:
      assert len(map_idx_to_allo) == num_labels
      assert 0 in map_idx_to_allo
      assert num_labels - 1 in map_idx_to_allo
    f = open(args.state_tying_output, "w")
    for i, allos in sorted(map_idx_to_allo.items()):
      for allo in allos:
        f.write("%s %i\n" % (allo.format(), i))
    f.close()
    print("Wrote state tying to %r." % args.state_tying_output, file=log.v1)

  print("The end.")


if __name__ == "__main__":
  from returnn.util import better_exchook
  better_exchook.install()
  log.initialize(verbosity=[2])
  main()
