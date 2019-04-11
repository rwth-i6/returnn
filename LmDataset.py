# -*- coding: utf8 -*-

"""
Provides :class:`LmDataset`, :class:`TranslationDataset`,
and some related helpers.
"""

from __future__ import print_function

import os
import sys
from Dataset import DatasetSeq
from CachedDataset2 import CachedDataset2
import gzip
import xml.etree.ElementTree as ElementTree
from Util import parse_orthography, parse_orthography_into_symbols, load_json, BackendEngine, unicode
from Log import log
import numpy
import time
import re
import typing
from random import Random


class LmDataset(CachedDataset2):
  """
  Dataset useful for language modeling.
  Reads simple txt files.
  """

  def __init__(self,
               corpus_file,
               orth_symbols_file=None,
               orth_symbols_map_file=None,
               orth_replace_map_file=None,
               word_based=False,
               seq_end_symbol="[END]",
               unknown_symbol="[UNKNOWN]",
               parse_orth_opts=None,
               phone_info=None,
               add_random_phone_seqs=0,
               auto_replace_unknown_symbol=False,
               log_auto_replace_unknown_symbols=10,
               log_skipped_seqs=10,
               error_on_invalid_seq=True,
               add_delayed_seq_data=False,
               delayed_seq_data_start_symbol="[START]",
               **kwargs):
    """
    After initialization, the corpus is represented by self.orths (as a list of sequences).
    The vocabulary is given by self.orth_symbols and self.orth_symbols_map gives the corresponding
    mapping from symbol to integer index.

    :param str|()->str|list[str]|()->list[str] corpus_file: Bliss XML or line-based txt. optionally can be gzip.
    :param dict|None phone_info: if you want to get phone seqs, dict with lexicon_file etc. see PhoneSeqGenerator
    :param str|()->str|None orth_symbols_file: list of orthography symbols, if you want to get orth symbol seqs
    :param str|()->str|None orth_symbols_map_file: list of orth symbols, each line: "symbol index"
    :param str|()->str|None orth_replace_map_file: JSON file with replacement dict for orth symbols
    :param bool word_based: whether to parse single words, or otherwise will be char-based
    :param str|None seq_end_symbol: what to add at the end, if given.
      will be set as postfix=[seq_end_symbol] or postfix=[] for parse_orth_opts.
    :param dict[str]|None parse_orth_opts: kwargs for parse_orthography()
    :param int add_random_phone_seqs: will add random seqs with the same len as the real seq as additional data
    :param bool|int log_auto_replace_unknown_symbols: write about auto-replacements with unknown symbol.
      if this is an int, it will only log the first N replacements, and then keep quiet.
    :param bool|int log_skipped_seqs: write about skipped seqs to logging, due to missing lexicon entry or so.
      if this is an int, it will only log the first N entries, and then keep quiet.
    :param bool error_on_invalid_seq: if there is a seq we would have to skip, error
    :param bool add_delayed_seq_data: will add another data-key "delayed" which will have the sequence
      delayed_seq_data_start_symbol + original_sequence[:-1]
    :param str delayed_seq_data_start_symbol: used for add_delayed_seq_data
    """
    super(LmDataset, self).__init__(**kwargs)

    if callable(corpus_file):
      corpus_file = corpus_file()
    if callable(orth_symbols_file):
      orth_symbols_file = orth_symbols_file()
    if callable(orth_symbols_map_file):
      orth_symbols_map_file = orth_symbols_map_file()
    if callable(orth_replace_map_file):
      orth_replace_map_file = orth_replace_map_file()

    print("LmDataset, loading file", corpus_file, file=log.v4)

    self.word_based = word_based
    self.seq_end_symbol = seq_end_symbol
    self.unknown_symbol = unknown_symbol
    self.parse_orth_opts = parse_orth_opts or {}
    self.parse_orth_opts.setdefault("word_based", self.word_based)
    self.parse_orth_opts.setdefault("postfix", [self.seq_end_symbol] if self.seq_end_symbol is not None else [])

    if orth_symbols_file:
      assert not phone_info
      assert not orth_symbols_map_file
      orth_symbols = open(orth_symbols_file).read().splitlines()
      self.orth_symbols_map = {sym: i for (i, sym) in enumerate(orth_symbols)}
      self.orth_symbols = orth_symbols
      self.labels["data"] = orth_symbols
      self.seq_gen = None
    if orth_symbols_map_file and orth_symbols_map_file.endswith('.pkl'):
      import pickle
      with open(orth_symbols_map_file, 'rb') as f:
        self.orth_symbols_map = pickle.load(f)
      self.orth_symbols = self.orth_symbols_map.keys()
      self.labels["data"] = list(self.orth_symbols)
      self.seq_gen = None
    elif orth_symbols_map_file:
      assert not phone_info
      orth_symbols_imap_list = [(int(b), a)
                                for (a, b) in [l.split(None, 1)
                                               for l in open(orth_symbols_map_file).read().splitlines()]]
      orth_symbols_imap_list.sort()
      assert orth_symbols_imap_list[0][0] == 0
      assert orth_symbols_imap_list[-1][0] == len(orth_symbols_imap_list) - 1
      self.orth_symbols_map = {sym: i for (i, sym) in orth_symbols_imap_list}
      self.orth_symbols = [sym for (i, sym) in orth_symbols_imap_list]
      self.labels["data"] = self.orth_symbols
      self.seq_gen = None
    else:
      assert not orth_symbols_file
      assert isinstance(phone_info, dict)
      self.seq_gen = PhoneSeqGenerator(**phone_info)
      self.orth_symbols = None
      self.labels["data"] = self.seq_gen.get_class_labels()
    if orth_replace_map_file:
      orth_replace_map = load_json(filename=orth_replace_map_file)
      assert isinstance(orth_replace_map, dict)
      self.orth_replace_map = {key: parse_orthography_into_symbols(v, word_based=self.word_based)
                               for (key, v) in orth_replace_map.items()}
      if self.orth_replace_map:
        if len(self.orth_replace_map) <= 5:
          print("  orth_replace_map: %r" % self.orth_replace_map, file=log.v5)
        else:
          print("  orth_replace_map: %i entries" % len(self.orth_replace_map), file=log.v5)
    else:
      self.orth_replace_map = {}

    num_labels = len(self.labels["data"])
    use_uint_types = False
    if BackendEngine.is_tensorflow_selected():
      use_uint_types = True
    if num_labels <= 2 ** 7:
      self.dtype = "int8"
    elif num_labels <= 2 ** 8 and use_uint_types:
      self.dtype = "uint8"
    elif num_labels <= 2 ** 31:
      self.dtype = "int32"
    elif num_labels <= 2 ** 32 and use_uint_types:
      self.dtype = "uint32"
    elif num_labels <= 2 ** 61:
      self.dtype = "int64"
    elif num_labels <= 2 ** 62 and use_uint_types:
      self.dtype = "uint64"
    else:
      raise Exception("cannot handle so much labels: %i" % num_labels)
    self.num_outputs = {"data": [len(self.labels["data"]), 1]}
    self.num_inputs = self.num_outputs["data"][0]
    self.seq_order = None
    self._tag_prefix = "line-"  # sequence tag is "line-n", where n is the line number (to be compatible with translation)  # nopep8
    self.auto_replace_unknown_symbol = auto_replace_unknown_symbol
    self.log_auto_replace_unknown_symbols = log_auto_replace_unknown_symbols
    self.log_skipped_seqs = log_skipped_seqs
    self.error_on_invalid_seq = error_on_invalid_seq
    self.add_random_phone_seqs = add_random_phone_seqs
    for i in range(add_random_phone_seqs):
      self.num_outputs["random%i" % i] = self.num_outputs["data"]
    self.add_delayed_seq_data = add_delayed_seq_data
    self.delayed_seq_data_start_symbol = delayed_seq_data_start_symbol
    if add_delayed_seq_data:
      self.num_outputs["delayed"] = self.num_outputs["data"]
      self.labels["delayed"] = self.labels["data"]

    if isinstance(corpus_file, list):  # If a list of files is provided, concatenate all.
      self.orths = []
      for file_name in corpus_file:
        self.orths += read_corpus(file_name)
    else:
      self.orths = read_corpus(corpus_file)
    # It's only estimated because we might filter some out or so.
    self._estimated_num_seqs = len(self.orths) // self.partition_epoch
    print("  done, loaded %i sequences" % len(self.orths), file=log.v4)

    self.next_orth_idx = 0
    self.next_seq_idx = 0
    self.num_skipped = 0
    self.num_unknown = 0

  def get_data_keys(self):
    """
    :rtype: list[str]
    """
    return sorted(self.num_outputs.keys())

  def get_target_list(self):
    """
    Unfortunately, the logic is swapped around for this dataset.
    "data" is the original data, which is usually the target,
    and you would use "delayed" as inputs.

    :rtype: list[str]
    """
    return ["data"]

  def get_data_dtype(self, key):
    """
    :param str key:
    :rtype: str
    """
    return self.dtype

  def init_seq_order(self, epoch=None, seq_list=None):
    """
    If random_shuffle_epoch1, for epoch 1 with "random" ordering, we leave the given order as is.
    Otherwise, this is mostly the default behavior.

    :param int|None epoch:
    :param list[str] | None seq_list: In case we want to set a predefined order.
    :rtype: bool
    :returns whether the order changed (True is always safe to return)
    """
    if seq_list and not self.error_on_invalid_seq:
      print("Setting error_on_invalid_seq to True since a seq_list is given. "
            "Please activate auto_replace_unknown_symbol if you want to prevent invalid sequences!",
            file=log.v4)
      self.error_on_invalid_seq = True
    super(LmDataset, self).init_seq_order(epoch=epoch, seq_list=seq_list)
    if not epoch:
      epoch = 1

    if seq_list is not None:
      self.seq_order = [int(s[len(self._tag_prefix):]) for s in seq_list]
    else:
      self.seq_order = self.get_seq_order_for_epoch(
        epoch=epoch, num_seqs=len(self.orths), get_seq_len=lambda i: len(self.orths[i]))
    self.next_orth_idx = 0
    self.next_seq_idx = 0
    self.num_skipped = 0
    self.num_unknown = 0
    if self.seq_gen:
      self.seq_gen.random_seed(epoch)
    return True

  def _reduce_log_skipped_seqs(self):
    if isinstance(self.log_skipped_seqs, bool):
      return
    assert isinstance(self.log_skipped_seqs, int)
    assert self.log_skipped_seqs >= 1
    self.log_skipped_seqs -= 1
    if not self.log_skipped_seqs:
      print("LmDataset: will stop logging about skipped sequences now", file=log.v4)

  def _reduce_log_auto_replace_unknown_symbols(self):
    if isinstance(self.log_auto_replace_unknown_symbols, bool):
      return
    assert isinstance(self.log_auto_replace_unknown_symbols, int)
    assert self.log_auto_replace_unknown_symbols >= 1
    self.log_auto_replace_unknown_symbols -= 1
    if not self.log_auto_replace_unknown_symbols:
      print("LmDataset: will stop logging about auto-replace with unknown symbol now", file=log.v4)

  def _collect_single_seq(self, seq_idx):
    """
    :type seq_idx: int
    :rtype: DatasetSeq | None
    :returns DatasetSeq or None if seq_idx >= num_seqs.
    """
    while True:
      if self.next_orth_idx >= len(self.seq_order):
        assert self.next_seq_idx <= seq_idx, "We expect that we iterate through all seqs."
        if self.num_skipped > 0:
          print("LmDataset: reached end, skipped %i sequences" % self.num_skipped)
        return None
      assert self.next_seq_idx == seq_idx, "We expect that we iterate through all seqs."
      true_idx = self.seq_order[self.next_orth_idx]
      orth = self.orths[true_idx]  # get sequence for the next index given by seq_order
      seq_tag = (self._tag_prefix + str(true_idx))
      self.next_orth_idx += 1
      if orth == "</s>":
        continue  # special sentence end symbol. empty seq, ignore.

      if self.seq_gen:
        try:
          phones = self.seq_gen.generate_seq(orth)
        except KeyError as e:
          if self.log_skipped_seqs:
            print("LmDataset: skipping sequence %r because of missing lexicon entry: %s" % (orth, e), file=log.v4)
            self._reduce_log_skipped_seqs()
          if self.error_on_invalid_seq:
            raise Exception("LmDataset: invalid seq %r, missing lexicon entry %r" % (orth, e))
          self.num_skipped += 1
          continue  # try another seq
        data = self.seq_gen.seq_to_class_idxs(phones, dtype=self.dtype)

      elif self.orth_symbols:
        orth_syms = parse_orthography(orth, **self.parse_orth_opts)
        while True:
          orth_syms = sum([self.orth_replace_map.get(s, [s]) for s in orth_syms], [])
          i = 0
          while i < len(orth_syms) - 1:
            if orth_syms[i:i+2] == [" ", " "]:
              orth_syms[i:i+2] = [" "]  # collapse two spaces
            else:
              i += 1
          if self.auto_replace_unknown_symbol:
            try:
              list(map(self.orth_symbols_map.__getitem__, orth_syms))  # convert to list to trigger map (it's lazy)
            except KeyError as e:
              if sys.version_info >= (3, 0):
                orth_sym = e.args[0]
              else:
                # noinspection PyUnresolvedReferences
                orth_sym = e.message
              if self.log_auto_replace_unknown_symbols:
                print("LmDataset: unknown orth symbol %r, adding to orth_replace_map as %r" % (
                  orth_sym, self.unknown_symbol), file=log.v3)
                self._reduce_log_auto_replace_unknown_symbols()
              self.orth_replace_map[orth_sym] = [self.unknown_symbol] if self.unknown_symbol is not None else []
              continue  # try this seq again with updated orth_replace_map
          break
        self.num_unknown += orth_syms.count(self.unknown_symbol)
        if self.word_based:
          orth_debug_str = repr(orth_syms)
        else:
          orth_debug_str = repr("".join(orth_syms))
        try:
          data = numpy.array(list(map(self.orth_symbols_map.__getitem__, orth_syms)), dtype=self.dtype)
        except KeyError as e:
          if self.log_skipped_seqs:
            print("LmDataset: skipping sequence %s because of missing orth symbol: %s" % (orth_debug_str, e),
                  file=log.v4)
            self._reduce_log_skipped_seqs()
          if self.error_on_invalid_seq:
            raise Exception("LmDataset: invalid seq %s, missing orth symbol %s" % (orth_debug_str, e))
          self.num_skipped += 1
          continue  # try another seq

      else:
        assert False

      targets = {}
      for i in range(self.add_random_phone_seqs):
        assert self.seq_gen  # not implemented atm for orths
        phones = self.seq_gen.generate_garbage_seq(target_len=data.shape[0])
        targets["random%i" % i] = self.seq_gen.seq_to_class_idxs(phones, dtype=self.dtype)
      if self.add_delayed_seq_data:
        targets["delayed"] = numpy.concatenate(
          ([self.orth_symbols_map[self.delayed_seq_data_start_symbol]], data[:-1])).astype(self.dtype)
        assert targets["delayed"].shape == data.shape
      self.next_seq_idx = seq_idx + 1
      return DatasetSeq(seq_idx=seq_idx, features=data, targets=targets, seq_tag=seq_tag)


def _is_bliss(filename):
  """
  :param str filename:
  :rtype: bool
  """
  try:
    corpus_file = open(filename, 'rb')
    if filename.endswith(".gz"):
      corpus_file = gzip.GzipFile(fileobj=corpus_file)
    context = iter(ElementTree.iterparse(corpus_file, events=('start', 'end')))
    _, root = next(context)  # get root element
    assert isinstance(root, ElementTree.Element)
    return root.tag == "corpus"
  except IOError:  # 'Not a gzipped file' or so
    pass
  except ElementTree.ParseError:  # 'syntax error' or so
    pass
  return False


def _iter_bliss(filename, callback):
  """
  :param str filename:
  :param (str)->None callback:
  """
  corpus_file = open(filename, 'rb')
  if filename.endswith(".gz"):
    corpus_file = gzip.GzipFile(fileobj=corpus_file)

  def getelements(tag):
    """
    Yield *tag* elements from *filename_or_file* xml incrementally.

    :param str tag:
    """
    context = iter(ElementTree.iterparse(corpus_file, events=('start', 'end')))
    _, root = next(context)  # get root element
    tree_ = [root]
    for event, elem_ in context:
      if event == "start":
        tree_ += [elem_]
      elif event == "end":
        assert tree_[-1] is elem_
        tree_ = tree_[:-1]
      if event == 'end' and elem_.tag == tag:
        yield tree_, elem_
        root.clear()  # free memory

  for tree, elem in getelements("segment"):
    elem_orth = elem.find("orth")
    orth_raw = elem_orth.text  # should be unicode
    orth_split = orth_raw.split()
    orth = " ".join(orth_split)

    callback(orth)


def _iter_txt(filename, callback):
  """
  :param str filename:
  :param (str)->None callback:
  """
  f = open(filename, 'rb')
  if filename.endswith(".gz"):
    f = gzip.GzipFile(fileobj=f)

  for line in f:
    try:
      line = line.decode("utf8")
    except UnicodeDecodeError:
      line = line.decode("latin_1")  # or iso8859_15?
    line = line.strip()
    if not line:
      continue
    callback(line)


def iter_corpus(filename, callback):
  """
  :param str filename:
  :param ((str)->None) callback:
  """
  if _is_bliss(filename):
    iter_f = _iter_bliss
  else:
    iter_f = _iter_txt
  iter_f(filename, callback)


def read_corpus(filename):
  """
  :param str filename:
  :return: list of orthographies
  :rtype: list[str]
  """
  out_list = []
  iter_corpus(filename, out_list.append)
  return out_list


class AllophoneState:
  """
  Represents one allophone (phone with context) state (number, boundary).
  In Sprint, see AllophoneStateAlphabet::index().
  """

  id = None  # u16 in Sprint. here just str
  context_history = ()  # list[u16] of phone id. here just list[str]
  context_future = ()  # list[u16] of phone id. here just list[str]
  boundary = 0  # s16. flags. 1 -> initial (@i), 2 -> final (@f)
  state = None  # s16, e.g. 0,1,2
  _attrs = ["id", "context_history", "context_future", "boundary", "state"]

  # noinspection PyShadowingBuiltins
  def __init__(self, id=None, state=None):
    """
    :param str id: phone
    :param int|None state:
    """
    self.id = id
    self.state = state

  def format(self):
    """
    :rtype: str
    """
    s = "%s{%s+%s}" % (
      self.id,
      "-".join(self.context_history) or "#",
      "-".join(self.context_future) or "#")
    if self.boundary & 1:
      s += "@i"
    if self.boundary & 2:
      s += "@f"
    if self.state is not None:
      s += ".%i" % self.state
    return s

  def __repr__(self):
    return self.format()

  def copy(self):
    """
    :rtype: AllophoneState
    """
    a = AllophoneState(id=self.id, state=self.state)
    for attr in self._attrs:
      if getattr(self, attr):
        setattr(a, attr, getattr(self, attr))
    return a

  def mark_initial(self):
    """
    Add flag to self.boundary.
    """
    self.boundary = self.boundary | 1

  def mark_final(self):
    """
    Add flag to self.boundary.
    """
    self.boundary = self.boundary | 2

  def phoneme(self, ctx_offset, out_of_context_id=None):
    """

    Phoneme::Id ContextPhonology::PhonemeInContext::phoneme(s16 pos) const {
      if (pos == 0)
        return phoneme_;
      else if (pos > 0) {
        if (u16(pos - 1) < context_.future.length())
          return context_.future[pos - 1];
        else
          return Phoneme::term;
      } else { verify(pos < 0);
        if (u16(-1 - pos) < context_.history.length())
          return context_.history[-1 - pos];
        else
          return Phoneme::term;
      }
    }

    :param int ctx_offset: 0 for center, >0 for future, <0 for history
    :param str|None out_of_context_id: what to return out of our context
    :return: phone-id from the offset
    :rtype: str
    """
    if ctx_offset == 0:
      return self.id
    if ctx_offset > 0:
      idx = ctx_offset - 1
      if idx >= len(self.context_future):
        return out_of_context_id
      return self.context_future[idx]
    if ctx_offset < 0:
      idx = -ctx_offset - 1
      if idx >= len(self.context_history):
        return out_of_context_id
      return self.context_history[idx]
    assert False

  def set_phoneme(self, ctx_offset, phone_id):
    """
    :param int ctx_offset: 0 for center, >0 for future, <0 for history
    :param str phone_id:
    """
    if ctx_offset == 0:
      self.id = phone_id
    elif ctx_offset > 0:
      idx = ctx_offset - 1
      assert idx == len(self.context_future)
      self.context_future = self.context_future + (phone_id,)
    elif ctx_offset < 0:
      idx = -ctx_offset - 1
      assert idx == len(self.context_history)
      self.context_history = self.context_history + (phone_id,)

  def phone_idx(self, ctx_offset, phone_idxs):
    """
    :param int ctx_offset: see self.phoneme()
    :param dict[str,int] phone_idxs:
    :rtype: int
    """
    phone = self.phoneme(ctx_offset=ctx_offset)
    if phone is None:
      return 0  # by definition in the Sprint C++ code: static const Id term = 0;
    else:
      return phone_idxs[phone] + 1

  def index(self, phone_idxs, num_states=3, context_length=1):
    """
    See self.from_index() for the inverse function.
    And see Sprint NoStateTyingDense::classify().

    :param dict[str,int] phone_idxs:
    :param int num_states: how much state per allophone
    :param int context_length: how much left/right context
    :rtype: int
    """
    assert max(len(self.context_history), len(self.context_future)) <= context_length
    assert 0 <= self.boundary < 4
    assert 0 <= self.state < num_states
    num_phones = max(phone_idxs.values()) + 1
    num_phone_classes = num_phones + 1  # 0 is the special no-context symbol
    result = 0
    for i in range(2 * context_length + 1):
      pos = i // 2
      if i % 2 == 1:
        pos = -pos - 1
      result *= num_phone_classes
      result += self.phone_idx(ctx_offset=pos, phone_idxs=phone_idxs)
    result *= num_states
    result += self.state
    result *= 4
    result += self.boundary
    return result

  @classmethod
  def from_index(cls, index, phone_ids, num_states=3, context_length=1):
    """
    Original Sprint C++ code:

        Mm::MixtureIndex NoStateTyingDense::classify(const AllophoneState& a) const {
            require_lt(a.allophone()->boundary, numBoundaryClasses_);
            require_le(0, a.state());
            require_lt(u32(a.state()), numStates_);
            u32 result = 0;
            for(u32 i = 0; i < 2 * contextLength_ + 1; ++i) {  // context len is usually 1
                // pos sequence: 0, -1, 1, [-2, 2, ...]
                s16 pos = i / 2;
                if(i % 2 == 1)
                    pos = -pos - 1;
                result *= numPhoneClasses_;
                u32 phoneIdx = a.allophone()->phoneme(pos);
                require_lt(phoneIdx, numPhoneClasses_);
                result += phoneIdx;
            }
            result *= numStates_;
            result += u32(a.state());
            result *= numBoundaryClasses_;
            result += a.allophone()->boundary;
            require_lt(result, nClasses_);
            return result;
        }

    Note that there is also AllophoneStateAlphabet::allophoneState, via Am/ClassicStateModel.cc,
    which unfortunately uses a different encoding.
    See :func:`from_classic_index`.

    :param int index:
    :param dict[int,str] phone_ids: reverse-map from self.index(). idx -> id
    :param int num_states: how much state per allophone
    :param int context_length: how much left/right context
    :rtype: int
    :rtype: AllophoneState
    """
    num_phones = max(phone_ids.keys()) + 1
    num_phone_classes = num_phones + 1  # 0 is the special no-context symbol
    code = index
    result = AllophoneState()
    result.boundary = code % 4
    code //= 4
    result.state = code % num_states
    code //= num_states
    for i in range(2 * context_length + 1):
      pos = i // 2
      if i % 2 == 1:
        pos = -pos - 1
      phone_idx = code % num_phone_classes
      code //= num_phone_classes
      result.set_phoneme(ctx_offset=pos, phone_id=phone_ids[phone_idx - 1] if phone_idx else "")
    return result

  @classmethod
  def from_classic_index(cls, index, allophones, max_states=6):
    """
    Via Sprint C++ Archiver.cc:getStateInfo():

        const u32 max_states = 6; // TODO: should be increased for non-speech
        for (state = 0; state < max_states; ++state) {
            if (emission >= allophones_.size())
            emission -= (1<<26);
            else break;
        }

    :param int index:
    :param int max_states:
    :param dict[int,AllophoneState] allophones:
    :rtype: AllophoneState
    """
    emission = index
    state = 0
    while state < max_states:
      if emission >= (1 << 26):
        emission -= (1 << 26)
        state += 1
      else:
        break
    a = allophones[emission].copy()
    a.state = state
    return a

  def __hash__(self):
    return hash(tuple([getattr(self, a) for a in self._attrs]))

  def __eq__(self, other):
    for a in self._attrs:
      if getattr(self, a) != getattr(other, a):
        return False
    return True

  def __ne__(self, other):
    return not self == other


class Lexicon:
  """
  Lexicon. Map of words to phoneme sequences (can have multiple pronunciations).
  """

  def __init__(self, filename):
    """
    :param str filename:
    """
    print("Loading lexicon", filename, file=log.v4)
    lex_file = open(filename, 'rb')
    if filename.endswith(".gz"):
      lex_file = gzip.GzipFile(fileobj=lex_file)
    self.phoneme_list = []  # type: typing.List[str]
    self.phonemes = {}  # type: typing.Dict[str,typing.Dict[str]]  # phone -> {index, symbol, variation}
    self.lemmas = {}  # type: typing.Dict[str,typing.Dict[str]]  # orth -> {orth, phons}

    context = iter(ElementTree.iterparse(lex_file, events=('start', 'end')))
    _, root = next(context)  # get root element
    tree = [root]
    for event, elem in context:
      if event == "start":
        tree += [elem]
      elif event == "end":
        assert tree[-1] is elem
        tree = tree[:-1]
        if elem.tag == "phoneme":
          symbol = elem.find("symbol").text.strip()  # should be unicode
          assert isinstance(symbol, (str, unicode))
          if elem.find("variation") is not None:
            variation = elem.find("variation").text.strip()
          else:
            variation = "context"  # default
          assert symbol not in self.phonemes
          assert variation in ["context", "none"]
          self.phoneme_list.append(symbol)
          self.phonemes[symbol] = {"index": len(self.phonemes), "symbol": symbol, "variation": variation}
          root.clear()  # free memory
        elif elem.tag == "phoneme-inventory":
          print("Finished phoneme inventory, %i phonemes" % len(self.phonemes), file=log.v4)
          root.clear()  # free memory
        elif elem.tag == "lemma":
          for orth_elem in elem.findall("orth"):
            orth = (orth_elem.text or "").strip()
            phons = [{"phon": e.text.strip(), "score": float(e.attrib.get("score", 0))} for e in elem.findall("phon")]
            assert orth not in self.lemmas
            self.lemmas[orth] = {"orth": orth, "phons": phons}
          root.clear()  # free memory
    print("Finished whole lexicon, %i lemmas" % len(self.lemmas), file=log.v4)


class StateTying:
  """
  Clustering of (allophone) states into classes.
  """

  def __init__(self, state_tying_file):
    """
    :param str state_tying_file:
    """
    self.allo_map = {}  # allophone-state-str -> class-idx
    self.class_map = {}  # class-idx -> set(allophone-state-str)
    ls = open(state_tying_file).read().splitlines()
    for l in ls:
      allo_str, class_idx_str = l.split()
      class_idx = int(class_idx_str)
      assert allo_str not in self.allo_map
      self.allo_map[allo_str] = class_idx
      self.class_map.setdefault(class_idx, set()).add(allo_str)
    min_class_idx = min(self.class_map.keys())
    max_class_idx = max(self.class_map.keys())
    assert min_class_idx == 0
    assert max_class_idx == len(self.class_map) - 1, "some classes are not represented"
    self.num_classes = len(self.class_map)


class PhoneSeqGenerator:
  """
  Generates phone sequences.
  """

  def __init__(self, lexicon_file,
               allo_num_states=3, allo_context_len=1,
               state_tying_file=None,
               add_silence_beginning=0.1, add_silence_between_words=0.1, add_silence_end=0.1,
               repetition=0.9, silence_repetition=0.95):
    """
    :param str lexicon_file: lexicon XML file
    :param int allo_num_states: how much HMM states per allophone (all but silence)
    :param int allo_context_len: how much context to store left and right. 1 -> triphone
    :param str | None state_tying_file: for state-tying, if you want that
    :param float add_silence_beginning: prob of adding silence at beginning
    :param float add_silence_between_words: prob of adding silence between words
    :param float add_silence_end: prob of adding silence at end
    :param float repetition: prob of repeating an allophone
    :param float silence_repetition: prob of repeating the silence allophone
    """
    self.lexicon = Lexicon(lexicon_file)
    self.phonemes = sorted(self.lexicon.phonemes.keys(), key=lambda s: self.lexicon.phonemes[s]["index"])
    self.rnd = Random(0)
    self.allo_num_states = allo_num_states
    self.allo_context_len = allo_context_len
    self.add_silence_beginning = add_silence_beginning
    self.add_silence_between_words = add_silence_between_words
    self.add_silence_end = add_silence_end
    self.repetition = repetition
    self.silence_repetition = silence_repetition
    self.si_lemma = self.lexicon.lemmas["[SILENCE]"]
    self.si_phone = self.si_lemma["phons"][0]["phon"]
    if state_tying_file:
      self.state_tying = StateTying(state_tying_file)
    else:
      self.state_tying = None

  def random_seed(self, seed):
    """
    :param int seed:
    """
    self.rnd.seed(seed)

  def get_class_labels(self):
    """
    :rtype: list[str]
    """
    if self.state_tying:
      # State tying labels. Represented by some allophone state str.
      return ["|".join(sorted(self.state_tying.class_map[i])) for i in range(self.state_tying.num_classes)]
    else:
      # The phonemes are the labels.
      return self.phonemes

  def seq_to_class_idxs(self, phones, dtype=None):
    """
    :param list[AllophoneState] phones: list of allophone states
    :param str dtype: eg "int32"
    :rtype: numpy.ndarray
    :returns 1D numpy array with the indices
    """
    if dtype is None:
      dtype = "int32"
    if self.state_tying:
      # State tying indices.
      return numpy.array([self.state_tying.allo_map[a.format()] for a in phones], dtype=dtype)
    else:
      # Phoneme indices. This must be consistent with get_class_labels.
      # It should not happen that we don't have some phoneme. The lexicon should not be inconsistent.
      return numpy.array([self.lexicon.phonemes[p.id]["index"] for p in phones], dtype=dtype)

  def _iter_orth(self, orth):
    """
    :param str orth:
    :rtype: typing.Iterator[typing.Dict[str]]
    """
    if self.rnd.random() < self.add_silence_beginning:
      yield self.si_lemma
    symbols = list(orth.split())
    i = 0
    while i < len(symbols):
      symbol = symbols[i]
      try:
        lemma = self.lexicon.lemmas[symbol]
      except KeyError:
        if "/" in symbol:
          symbols[i:i+1] = symbol.split("/")
          continue
        if "-" in symbol:
          symbols[i:i+1] = symbol.split("-")
          continue
        raise
      i += 1
      yield lemma
      if i < len(symbols):
        if self.rnd.random() < self.add_silence_between_words:
          yield self.si_lemma
    if self.rnd.random() < self.add_silence_end:
      yield self.si_lemma

  def orth_to_phones(self, orth):
    """
    :param str orth:
    :rtype: str
    """
    phones = []
    for lemma in self._iter_orth(orth):
      phon = self.rnd.choice(lemma["phons"])
      phones += [phon["phon"]]
    return " ".join(phones)

  # noinspection PyMethodMayBeStatic
  def _phones_to_allos(self, phones):
    for p in phones:
      a = AllophoneState()
      a.id = p
      yield a

  def _random_allo_silence(self, phone=None):
    if phone is None:
      phone = self.si_phone
    while True:
      a = AllophoneState()
      a.id = phone
      a.mark_initial()
      a.mark_final()
      a.state = 0  # silence only has one state
      yield a
      if self.rnd.random() >= self.silence_repetition:
        break

  def _allos_add_states(self, allos):
    for _a in allos:
      if _a.id == self.si_phone:
        for a in self._random_allo_silence(_a.id):
          yield a
      else:  # non-silence
        for state in range(self.allo_num_states):
          while True:
            a = AllophoneState()
            a.id = _a.id
            a.context_history = _a.context_history
            a.context_future = _a.context_future
            a.boundary = _a.boundary
            a.state = state
            yield a
            if self.rnd.random() >= self.repetition:
              break

  def _allos_set_context(self, allos):
    """
    :param list[AllophoneState] allos:
    """
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

  def generate_seq(self, orth):
    """
    :param str orth: orthography as a str. orth.split() should give words in the lexicon
    :rtype: list[AllophoneState]
    :returns allophone state list. those will have repetitions etc
    """
    allos = []  # type: typing.List[AllophoneState]
    for lemma in self._iter_orth(orth):
      phon = self.rnd.choice(lemma["phons"])
      l_allos = list(self._phones_to_allos(phon["phon"].split()))
      l_allos[0].mark_initial()
      l_allos[-1].mark_final()
      allos += l_allos
    self._allos_set_context(allos)
    allos = list(self._allos_add_states(allos))
    return allos

  def _random_phone_seq(self, prob_add=0.8):
    while True:
      yield self.rnd.choice(self.phonemes)
      if self.rnd.random() >= prob_add:
        break

  def _random_allo_seq(self, prob_word_add=0.8):
    allos = []
    while True:
      phones = self._random_phone_seq()
      w_allos = list(self._phones_to_allos(phones))
      w_allos[0].mark_initial()
      w_allos[-1].mark_final()
      allos += w_allos
      if self.rnd.random() >= prob_word_add:
        break
    self._allos_set_context(allos)
    return list(self._allos_add_states(allos))

  def generate_garbage_seq(self, target_len):
    """
    :param int target_len: len of the returned seq
    :rtype: list[AllophoneState]
    :returns allophone state list. those will have repetitions etc.
    It will randomly generate a sequence of phonemes and transform that
    into a list of allophones in a similar way than generate_seq().
    """
    allos = []
    while True:
      allos += self._random_allo_seq()
      # Add some silence so that left/right context is correct for further allophones.
      allos += list(self._random_allo_silence())
      if len(allos) >= target_len:
        allos = allos[:target_len]
        break
    return allos


class TranslationDataset(CachedDataset2):
  """
  Based on the conventions by our team for translation datasets.
  It gets a directory and expects these files:

      source.dev(.gz)?
      source.train(.gz)?
      source.vocab.pkl
      target.dev(.gz)?
      target.train(.gz)?
      target.vocab.pkl
  """

  MapToDataKeys = {"source": "data", "target": "classes"}  # just by our convention
  _main_data_key = None
  _main_classes_key = None

  def __init__(self, path, file_postfix, source_postfix="", target_postfix="",
               source_only=False,
               unknown_label=None,
               seq_list_file=None,
               use_cache_manager=False,
               **kwargs):
    """
    :param str path: the directory containing the files
    :param str file_postfix: e.g. "train" or "dev". it will then search for "source." + postfix and "target." + postfix.
    :param bool random_shuffle_epoch1: if True, will also randomly shuffle epoch 1. see self.init_seq_order().
    :param None|str source_postfix: will concat this at the end of the source. e.g.
    :param None|str target_postfix: will concat this at the end of the target.
      You might want to add some sentence-end symbol.
    :param bool source_only: if targets are not available
    :param str|None unknown_label: "UNK" or so. if not given, then will not replace unknowns but throw an error
    :param str seq_list_file: filename. line-separated list of line numbers defining fixed sequence order.
      multiple occurrences supported, thus allows for repeating examples while loading only once.
    :param bool use_cache_manager: uses :func:`Util.cf` for files
    """

    super(TranslationDataset, self).__init__(**kwargs)
    self.path = path
    self.file_postfix = file_postfix
    self.seq_list = [int(n) for n in open(seq_list_file).read().splitlines()] if seq_list_file else None
    if self._main_data_key is None:
      self._main_data_key = "data"
    if self._main_classes_key is None:
      self._main_classes_key = "classes"
    self._add_postfix = {self._main_data_key: source_postfix, self._main_classes_key: target_postfix}
    self._keys_to_read = [self._main_data_key, self._main_classes_key]
    self._use_cache_manager = use_cache_manager
    from threading import Lock, Thread
    self._lock = Lock()
    import os
    assert os.path.isdir(path)
    if source_only:
      self.MapToDataKeys = self.__class__.MapToDataKeys.copy()
      del self.MapToDataKeys["target"]
    self._data_files = {data_key: self._get_data_file(prefix) for (prefix, data_key) in self.MapToDataKeys.items()}
    self._data = {
      data_key: [] for data_key in self._data_files.keys()}  # type: typing.Dict[str,typing.List[numpy.ndarray]]
    self._data_len = None  # type: typing.Optional[int]
    self._vocabs = {data_key: self._get_vocab(prefix) for (prefix, data_key) in self.MapToDataKeys.items()}
    self.num_outputs = {k: [max(self._vocabs[k].values()) + 1, 1] for k in self._vocabs.keys()}  # all sparse
    assert all([v1 <= 2 ** 31 for (k, (v1, v2)) in self.num_outputs.items()])  # we use int32
    self.num_inputs = self.num_outputs[self._main_data_key][0]
    self._reversed_vocabs = {k: self._reverse_vocab(k) for k in self._vocabs.keys()}
    self.labels = {k: self._get_label_list(k) for k in self._vocabs.keys()}
    self._unknown_label = unknown_label
    self._seq_order = None  # type: typing.Optional[typing.List[int]]  # seq_idx -> line_nr
    self._tag_prefix = "line-"  # sequence tag is "line-n", where n is the line number
    self._thread = Thread(name="%r reader" % self, target=self._thread_main)
    self._thread.daemon = True
    self._thread.start()

  def _extend_data(self, k, data_strs):
    vocab = self._vocabs[k]
    data = [
      self._data_str_to_numpy(vocab, s.decode("utf8").strip() + self._add_postfix[k])
      for s in data_strs]
    with self._lock:
      self._data[k].extend(data)

  def _thread_main(self):
    from Util import interrupt_main
    # noinspection PyBroadException
    try:
      import better_exchook
      better_exchook.install()
      from Util import AsyncThreadRun

      # First iterate once over the data to get the data len as fast as possible.
      data_len = 0
      while True:
        ls = self._data_files[self._main_data_key].readlines(10 ** 4)
        data_len += len(ls)
        if not ls:
          break
      with self._lock:
        self._data_len = data_len
      self._data_files[self._main_data_key].seek(0, os.SEEK_SET)  # we will read it again below

      # Now, read and use the vocab for a compact representation in memory.
      keys_to_read = list(self._keys_to_read)
      while True:
        for k in keys_to_read:
          data_strs = self._data_files[k].readlines(10 ** 6)
          if not data_strs:
            assert len(self._data[k]) == self._data_len
            keys_to_read.remove(k)
            continue
          assert len(self._data[k]) + len(data_strs) <= self._data_len
          self._extend_data(k, data_strs)
        if not keys_to_read:
          break
      for k, f in list(self._data_files.items()):
        f.close()
        self._data_files[k] = None

    except Exception:
      sys.excepthook(*sys.exc_info())
      interrupt_main()

  def _transform_filename(self, filename):
    """
    :param str filename:
    :return: maybe transformed filename, e.g. via cache manager
    :rtype: str
    """
    if self._use_cache_manager:
      import Util
      filename = Util.cf(filename)
    return filename

  def _get_data_file(self, prefix):
    """
    :param str prefix: e.g. "source" or "target"
    :return: full filename
    :rtype: io.FileIO
    """
    import os
    filename = "%s/%s.%s" % (self.path, prefix, self.file_postfix)
    if os.path.exists(filename):
      return open(self._transform_filename(filename), "rb")
    if os.path.exists(filename + ".gz"):
      import gzip
      return gzip.GzipFile(self._transform_filename(filename + ".gz"), "rb")
    raise Exception("Data file not found: %r (.gz)?" % filename)

  def _get_vocab(self, prefix):
    """
    :param str prefix: e.g. "source" or "target"
    :rtype: dict[str,int]
    """
    import os
    filename = "%s/%s.vocab.pkl" % (self.path, prefix)
    if not os.path.exists(filename):
      raise Exception("Vocab file not found: %r" % filename)
    import pickle
    vocab = pickle.load(open(self._transform_filename(filename), "rb"))
    assert isinstance(vocab, dict)
    return vocab

  def _reverse_vocab(self, data_key):
    """
    Note that there might be multiple items in the vocabulary (e.g. "<S>" and "</S>")
    which map to the same label index.
    We sort the list by lexical order and the last entry for a particular label index is used ("<S>" in that example).

    :param str data_key: e.g. "data" or "classes"
    :rtype: dict[int,str]
    """
    return {v: k for (k, v) in sorted(self._vocabs[data_key].items())}

  def _get_label_list(self, data_key):
    """
    :param str data_key: e.g. "data" or "classes"
    :return: list of len num labels
    :rtype: list[str]
    """
    reversed_vocab = self._reversed_vocabs[data_key]
    assert isinstance(reversed_vocab, dict)
    num_labels = self.num_outputs[data_key][0]
    return list(map(reversed_vocab.__getitem__, range(num_labels)))

  def _data_str_to_numpy(self, vocab, s):
    """
    :param dict[str,int] vocab:
    :param str s:
    :rtype: numpy.ndarray
    """
    words = s.split()
    if self._unknown_label is None:
      try:
        words_idxs = list(map(vocab.__getitem__, words))
      except KeyError as e:
        raise Exception(
          "Can not handle unknown token without unknown_label: %s (%s)" % (str(e), bytes(str(e), 'utf-8')))
    else:
      unknown_label_id = vocab[self._unknown_label]
      words_idxs = [vocab.get(w, unknown_label_id) for w in words]
    return numpy.array(words_idxs, dtype=numpy.int32)

  def _get_data(self, key, line_nr):
    """
    :param str key: "data" or "classes"
    :param int line_nr:
    :return: 1D array
    :rtype: numpy.ndarray
    """
    import time
    last_print_time = 0
    last_print_len = None
    while True:
      with self._lock:
        if self._data_len is not None:
          assert line_nr <= self._data_len
        cur_len = len(self._data[key])
        if line_nr < cur_len:
          return self._data[key][line_nr]
      if cur_len != last_print_len and time.time() - last_print_time > 10:
        print("%r: waiting for %r, line %i (%i loaded so far)..." % (self, key, line_nr, cur_len), file=log.v3)
        last_print_len = cur_len
        last_print_time = time.time()
      time.sleep(1)

  def _get_data_len(self):
    """
    :return: num seqs of the whole underlying data
    :rtype: int
    """
    import time
    t = 0
    while True:
      with self._lock:
        if self._data_len is not None:
          return self._data_len
      if t == 0:
        print("%r: waiting for data length info..." % (self,), file=log.v3)
      time.sleep(1)
      t += 1

  def have_corpus_seq_idx(self):
    """
    :rtype: bool
    """
    return True

  def get_corpus_seq_idx(self, seq_idx):
    """
    :param int seq_idx:
    :rtype: int
    """
    if self._seq_order is None:
      return None
    return self._seq_order[seq_idx]

  def is_data_sparse(self, key):
    """
    :param str key:
    :rtype: bool
    """
    return True  # all is sparse

  def get_data_dtype(self, key):
    """
    :param str key:
    :rtype: str
    """
    return "int32"  # sparse -> label idx

  def init_seq_order(self, epoch=None, seq_list=None):
    """
    If random_shuffle_epoch1, for epoch 1 with "random" ordering, we leave the given order as is.
    Otherwise, this is mostly the default behavior.

    :param int|None epoch:
    :param list[str] | None seq_list: In case we want to set a predefined order.
    :rtype: bool
    :returns whether the order changed (True is always safe to return)
    """
    super(TranslationDataset, self).init_seq_order(epoch=epoch, seq_list=seq_list)
    if not epoch:
      epoch = 1

    if seq_list is None and self.seq_list:
      seq_list = self.seq_list
    if seq_list is not None:
      self._seq_order = [int(s[len(self._tag_prefix):]) for s in seq_list]
    else:
      num_seqs = self._get_data_len()
      self._seq_order = self.get_seq_order_for_epoch(
        epoch=epoch, num_seqs=num_seqs, get_seq_len=lambda i: len(self._get_data(key=self._main_data_key, line_nr=i)))
    self._num_seqs = len(self._seq_order)
    return True

  def _collect_single_seq(self, seq_idx):
    if seq_idx >= self._num_seqs:
      return None
    line_nr = self._seq_order[seq_idx]
    features = self._get_data(key=self._main_data_key, line_nr=line_nr)
    targets = self._get_data(key=self._main_classes_key, line_nr=line_nr)

    assert features is not None and targets is not None
    return DatasetSeq(
      seq_idx=seq_idx,
      seq_tag=self._tag_prefix + str(line_nr),
      features=features,
      targets=targets)


class ConfusionNetworkDataset(TranslationDataset):
  """
  This dataset allows for multiple (weighted) options for each word in the source sequence. In particular, it can be
  used to represent confusion networks. Two matrices (of dimension source length x max_density) will be provided as
  input to the network, one containing the word ids ("sparse_inputs") and one containing the weights ("sparse_weights").
  The matrices are read from the following input format (example):

  "__ALT__ we're|0.999659__were|0.000341148 a|0.977656__EPS|0.0223441 social|1.0 species|1.0"

  Input positions are separated by a space, different word options at one positions are separated by two underscores.
  Each word option has a weight appended to it, separated by "|". If "__ALT__" is missing, the line is interpreted
  as a regular plain text sentence. For this, all weights are set to 1.0 and only one word option is used at each
  position. Epsilon arcs of confusion networks can be represented by a special token (e.g. "EPS"), which has to be
  added to the source vocabulary.

  Via "seq_list_file" (see TranslationDataset) it is possible to give an explicit order of training examples. This
  can e.g. be used to repeat the confusion net part of the training data without loading it several times.
  """

  MapToDataKeys = {"source": "sparse_inputs", "target": "classes"}

  def __init__(self, max_density=20, **kwargs):
    """
    :param str path: the directory containing the files
    :param str file_postfix: e.g. "train" or "dev". it will then search for "source." + postfix and "target." + postfix.
    :param bool random_shuffle_epoch1: if True, will also randomly shuffle epoch 1. see self.init_seq_order().
    :param None|str source_postfix: will concat this at the end of the source. e.g.
    :param None|str target_postfix: will concat this at the end of the target.
      You might want to add some sentence-end symbol.
    :param bool source_only: if targets are not available
    :param str|None unknown_label: "UNK" or so. if not given, then will not replace unknowns but throw an error
    :param int max_density: the density of the confusion network: max number of arcs per slot
    """
    self._main_data_key = "sparse_inputs"
    self._keys_to_read = ["sparse_inputs", "classes"]
    self.density = max_density
    super(ConfusionNetworkDataset, self).__init__(**kwargs)
    if "sparse_weights" not in self._data.keys():
      self._data["sparse_weights"] = []

  def get_data_keys(self):
    """
    :rtype: list[str]
    """
    return ["sparse_inputs", "sparse_weights", "classes"]

  def is_data_sparse(self, key):
    """
    :param str key:
    :rtype: bool
    """
    if key == "sparse_weights":
      return False
    return True  # everything else is sparse

  def get_data_dtype(self, key):
    """
    :param str key:
    :rtype: str
    """
    if key == "sparse_weights":
      return "float32"
    return "int32"  # sparse -> label idx

  def get_data_shape(self, key):
    """
    :param str key:
    :rtype: list[int]
    """
    if key in ["sparse_inputs", "sparse_weights"]:
      return [self.density]
    return []

  def _load_single_confusion_net(self, words, vocab, postfix):
    """
    :param list[str] words:
    :param dict[str,int] vocab:
    :param str postfix:
    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    unknown_label_id = vocab[self._unknown_label]
    offset = 0
    postfix_index = None
    if postfix is not None:
      postfix_index = vocab.get(postfix, unknown_label_id)
      if postfix_index != unknown_label_id:
        offset = 1
    words_idxs = numpy.zeros(shape=(len(words) + offset, self.density), dtype=numpy.int32)
    words_confs = numpy.zeros(shape=(len(words) + offset, self.density), dtype=numpy.float32)
    for n in range(len(words)):
      arcs = words[n].split("__")
      for k in range(min(self.density, len(arcs))):
        (arc, conf) = arcs[k].split("|")
        words_idxs[n][k] = vocab.get(arc, unknown_label_id)
        words_confs[n][k] = float(conf)
    if offset != 0:
      words_idxs[len(words)][0] = postfix_index
      words_confs[len(words)][0] = 1
    return words_idxs, words_confs

  def _data_str_to_sparse_inputs(self, vocab, s, postfix=None):
    """
    :param dict[str,int] vocab:
    :param str s:
    :param str postfix:
    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    words = s.split()
    if words and words[0] == "__ALT__":
        words.pop(0)
        return self._load_single_confusion_net(words, vocab, postfix)

    if postfix is not None:
      words.append(postfix)
    unknown_label_id = vocab[self._unknown_label]
    words_idxs = numpy.array([vocab.get(w, unknown_label_id) for w in words], dtype=numpy.int32)
    words_confs = None  # creating matrices for plain text input is delayed to _collect_single_seq to save memory
    return words_idxs, words_confs

  def _extend_data(self, key, data_strs):
    """
    :param str key: the key ("sparse_inputs", or "classes")
    :param list[str|bytes] data_strs: array of input for the key
    """
    vocab = self._vocabs[key]
    if key == self._main_data_key:  # the sparse inputs and weights
      idx_data = []
      conf_data = []
      for s in data_strs:
        (words_idxs, words_confs) = self._data_str_to_sparse_inputs(
          vocab, s.decode("utf8").strip(), self._add_postfix[key])
        idx_data.append(words_idxs)
        conf_data.append(words_confs)
      with self._lock:
        self._data[key].extend(idx_data)
        self._data["sparse_weights"].extend(conf_data)
    else:  # the classes
      data = [
        self._data_str_to_numpy(vocab, s.decode("utf8").strip() + self._add_postfix[key])
        for s in data_strs]
      with self._lock:
        self._data[key].extend(data)

  def _collect_single_seq(self, seq_idx):
    if seq_idx >= self._num_seqs:
      return None
    line_nr = self._seq_order[seq_idx]
    features = {key: self._get_data(key=key, line_nr=line_nr) for key in self.get_data_keys()}
    if features['sparse_weights'] is None:
      seq = features[self._main_data_key]
      features[self._main_data_key] = numpy.zeros(shape=(len(seq), self.density), dtype=numpy.int32)
      features['sparse_weights'] = numpy.zeros(shape=(len(seq), self.density), dtype=numpy.float32)
      for n in range(len(seq)):
        features[self._main_data_key][n][0] = seq[n]
        features['sparse_weights'][n][0] = 1
    return DatasetSeq(
      seq_idx=seq_idx,
      seq_tag=self._tag_prefix + str(line_nr),
      features=features, targets=None)


'''
Cleaners are transformations that run over the input text at both training and eval time.
Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).

Code from here:
https://github.com/keithito/tacotron/blob/master/text/cleaners.py
https://github.com/keithito/tacotron/blob/master/text/numbers.py
'''


# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
# WARNING: Every change here means an incompatible change,
# so better leave it always as it is!
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misses'),
  ('ms', 'miss'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]


def expand_abbreviations(text):
  """
  :param str text:
  :rtype: str
  """
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text


def lowercase(text):
  """
  :param str text:
  :rtype: str
  """
  return text.lower()


def collapse_whitespace(text):
  """
  :param str text:
  :rtype: str
  """
  return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
  """
  :param str text:
  :rtype: str
  """
  # noinspection PyUnresolvedReferences,PyPackageRequirements
  from unidecode import unidecode
  return unidecode(text)


def basic_cleaners(text):
  """
  Basic pipeline that lowercases and collapses whitespace without transliteration.

  :param str text:
  :rtype: str
  """
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def transliteration_cleaners(text):
  """
  Pipeline for non-English text that transliterates to ASCII.

  :param str text:
  :rtype: str
  """
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def english_cleaners(text):
  """
  Pipeline for English text, including number and abbreviation expansion.
  :param str text:
  :rtype: str
  """
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = normalize_numbers(text)
  text = expand_abbreviations(text)
  text = collapse_whitespace(text)
  return text


def get_remove_chars(chars):
  """
  :param str|list[str] chars:
  :rtype: (str)->str
  """
  def remove_chars(text):
    """
    :param str text:
    :rtype: str
    """
    for c in chars:
      text = text.replace(c, " ")
    text = collapse_whitespace(text)
    return text
  return remove_chars


_inflect = None


def _get_inflect():
  global _inflect
  if _inflect:
    return _inflect
  # noinspection PyUnresolvedReferences,PyPackageRequirements
  import inflect
  _inflect = inflect.engine()
  return _inflect


_comma_number_re = re.compile(r'([0-9][0-9,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re = re.compile(r'£([0-9,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9.,]*[0-9]+)')
_ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
_number_re = re.compile(r'[0-9]+')


def _remove_commas(m):
  """
  :param typing.Match m:
  :rtype: str
  """
  return m.group(1).replace(',', '')


def _expand_decimal_point(m):
  """
  :param typing.Match m:
  :rtype: str
  """
  return m.group(1).replace('.', ' point ')


def _expand_dollars(m):
  """
  :param typing.Match m:
  :rtype: str
  """
  match = m.group(1)
  parts = match.split('.')
  if len(parts) > 2:
    return match + ' dollars'  # Unexpected format
  dollars = int(parts[0]) if parts[0] else 0
  cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
  if dollars and cents:
    dollar_unit = 'dollar' if dollars == 1 else 'dollars'
    cent_unit = 'cent' if cents == 1 else 'cents'
    return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
  elif dollars:
    dollar_unit = 'dollar' if dollars == 1 else 'dollars'
    return '%s %s' % (dollars, dollar_unit)
  elif cents:
    cent_unit = 'cent' if cents == 1 else 'cents'
    return '%s %s' % (cents, cent_unit)
  else:
    return 'zero dollars'


def _expand_ordinal(m):
  """
  :param typing.Match m:
  :rtype: str
  """
  return _get_inflect().number_to_words(m.group(0))


def _expand_number(m):
  """
  :param typing.Match m:
  :rtype: str
  """
  num = int(m.group(0))
  if 1000 < num < 3000:
    if num == 2000:
      return 'two thousand'
    elif 2000 < num < 2010:
      return 'two thousand ' + _get_inflect().number_to_words(num % 100)
    elif num % 100 == 0:
      return _get_inflect().number_to_words(num // 100) + ' hundred'
    else:
      return _get_inflect().number_to_words(num, andword='', zero='oh', group=2).replace(', ', ' ')
  else:
    return _get_inflect().number_to_words(num, andword='')


def normalize_numbers(text):
  """
  :param str text:
  :rtype: str
  """
  text = re.sub(_comma_number_re, _remove_commas, text)
  text = re.sub(_pounds_re, r'\1 pounds', text)
  text = re.sub(_dollars_re, _expand_dollars, text)
  text = re.sub(_decimal_number_re, _expand_decimal_point, text)
  text = re.sub(_ordinal_re, _expand_ordinal, text)
  text = re.sub(_number_re, _expand_number, text)
  return text


def _dummy_identity_pp(text):
  """
  :param str text:
  :rtype: str
  """
  return text


def get_post_processor_function(opts):
  """
  You might want to use :mod:`inflect` or :mod:`unidecode`
  for some normalization / cleanup.
  This function can be used to get such functions.

  :param str|list[str] opts: e.g. "english_cleaners", or "get_remove_chars(',/')"
  :return: function
  :rtype: (str)->str
  """
  if not opts:
    return _dummy_identity_pp
  if isinstance(opts, str):
    if "(" in opts:
      f = eval(opts)
    else:
      f = globals()[opts]
    assert callable(f)
    res_test = f("test")
    assert isinstance(res_test, str), "%r does not seem as a valid function str->str" % (opts,)
    return f
  assert isinstance(opts, list)
  if len(opts) == 1:
    return get_post_processor_function(opts[0])
  pps = [get_post_processor_function(pp) for pp in opts]

  def chained_post_processors(text):
    """
    :param str text:
    :rtype: str
    """
    for pp in pps:
      text = pp(text)
    return text

  return chained_post_processors


def _main():
  import better_exchook
  better_exchook.install()
  from argparse import ArgumentParser
  arg_parser = ArgumentParser()
  arg_parser.add_argument(
    "lm_dataset", help="Python eval string, should eval to dict" +
                       ", or otherwise filename, and will just dump")
  arg_parser.add_argument("--post_processor", nargs="*")
  args = arg_parser.parse_args()
  if not args.lm_dataset.startswith("{") and os.path.isfile(args.lm_dataset):
    callback = print
    if args.post_processor:
      pp = get_post_processor_function(args.post_processor)

      def callback(text):
        """
        :param str text:
        """
        print(pp(text))

    iter_corpus(args.lm_dataset, callback)
    sys.exit(0)

  log.initialize(verbosity=[5])
  print("LmDataset demo startup")
  kwargs = eval(args.lm_dataset)
  assert isinstance(kwargs, dict), "arg should be str of dict: %s" % args.lm_dataset
  print("Creating LmDataset with kwargs=%r ..." % kwargs)
  dataset = LmDataset(**kwargs)
  print("init_seq_order ...")
  dataset.init_seq_order(epoch=1)

  seq_idx = 0
  last_log_time = time.time()
  print("start iterating through seqs ...")
  while dataset.is_less_than_num_seqs(seq_idx):
    if seq_idx == 0:
      print("load_seqs with seq_idx=%i ...." % seq_idx)
    dataset.load_seqs(seq_idx, seq_idx + 1)

    if time.time() - last_log_time > 2.0:
      last_log_time = time.time()
      # noinspection PyProtectedMember
      print("Loading %s progress, %i/%i (%.0f%%) seqs loaded (%.0f%% skipped), (%.0f%% unknown) total syms %i ..." % (
            dataset.__class__.__name__, dataset.next_orth_idx, dataset.estimated_num_seqs,
            100.0 * dataset.next_orth_idx / dataset.estimated_num_seqs,
            100.0 * dataset.num_skipped / (dataset.next_orth_idx or 1),
            100.0 * dataset.num_unknown / dataset._num_timesteps_accumulated["data"],
            dataset._num_timesteps_accumulated["data"]))

    seq_idx += 1

  print("finished iterating, num seqs: %i" % seq_idx)
  print("dataset len:", dataset.len_info())


if __name__ == "__main__":
  _main()
