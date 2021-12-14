"""
Vocabulary related classes for targets such as BPE, SentencePieces etc...
"""

from __future__ import print_function

__all__ = [
  "Vocabulary",
  "BytePairEncoding",
  "SamplingBytePairEncoding",
  "SentencePieces",
  "CharacterTargets",
  "Utf8ByteTargets"
]

import sys
import typing

import numpy

from returnn.log import log
from returnn.util.basic import PY3


class Vocabulary(object):
  """
  Represents a vocabulary (set of words, and their ids).
  Used by :class:`BytePairEncoding`.
  """

  _cache = {}  # filename -> vocab dict, labels dict (see _parse_vocab)

  @classmethod
  def create_vocab(cls, **opts):
    """
    :param opts: kwargs for class
    :rtype: Vocabulary|BytePairEncoding|CharacterTargets
    """
    opts = opts.copy()
    clz = cls
    if "class" in opts:
      class_name = opts.pop("class")
      clz = globals()[class_name]
      assert issubclass(clz, Vocabulary), "class %r %r is not a subclass of %r" % (class_name, clz, cls)
    elif "bpe_file" in opts:
      clz = BytePairEncoding
    return clz(**opts)

  def __init__(self, vocab_file, seq_postfix=None, unknown_label="UNK", num_labels=None, labels=None):
    """
    :param str|None vocab_file:
    :param str|None unknown_label:
    :param int num_labels: just for verification
    :param list[int]|None seq_postfix: labels will be added to the seq in self.get_seq
    :param list[str]|None labels:
    """
    self.vocab_file = vocab_file
    self.unknown_label = unknown_label
    self.num_labels = None  # type: typing.Optional[int]  # will be set by _parse_vocab
    self.vocab = None  # type: typing.Optional[typing.Dict[str,int]]  # label->idx
    self.labels = labels

    self._parse_vocab()
    if num_labels is not None:
      assert self.num_labels == num_labels
    self.unknown_label_id = self.vocab[self.unknown_label] if self.unknown_label is not None else None
    self.seq_postfix = seq_postfix or []

  def __repr__(self):
    return "Vocabulary(%r, num_labels=%s, unknown_label=%r)" % (self.vocab_file, self.num_labels, self.unknown_label)

  def set_random_seed(self, seed):
    """
    This can be called for a new epoch or so.
    Usually it has no effect, as there is no randomness.
    However, some vocab class could introduce some sampling process.

    :param int seed:
    """
    pass  # usually there is no randomness, so ignore

  def _parse_vocab(self):
    """
    Sets self.vocab, self.labels, self.num_labels.
    """
    filename = self.vocab_file
    import pickle

    if self.labels is not None:
      self.vocab = {label: i for i, label in enumerate(self.labels)}
      assert self.unknown_label is None or self.unknown_label in self.vocab
      self.num_labels = len(self.labels)
    elif filename in self._cache:
      self.vocab, self.labels = self._cache[filename]
      assert self.unknown_label is None or self.unknown_label in self.vocab
      self.num_labels = len(self.labels)
    else:
      if filename[-4:] == ".pkl":
        d = pickle.load(open(filename, "rb"))
      else:
        d = eval(open(filename, "r").read())
        if not PY3:
          # Any utf8 string will not be a unicode string automatically, so enforce this.
          assert isinstance(d, dict)
          from returnn.util.basic import py2_utf8_str_to_unicode
          d = {py2_utf8_str_to_unicode(s): i for (s, i) in d.items()}
      assert isinstance(d, dict)
      assert self.unknown_label is None or self.unknown_label in d
      labels = {idx: label for (label, idx) in sorted(d.items())}
      min_label, max_label, num_labels = min(labels), max(labels), len(labels)
      assert 0 == min_label
      if num_labels - 1 < max_label:
        print("Vocab error: not all indices used? max label: %i" % max_label, file=log.v1)
        print("unused labels: %r" % ([i for i in range(max_label + 1) if i not in labels],), file=log.v2)
      assert num_labels - 1 == max_label
      self.num_labels = len(labels)
      self.vocab = d
      self.labels = [label for (idx, label) in sorted(labels.items())]
      self._cache[filename] = (self.vocab, self.labels)

  @classmethod
  def create_vocab_dict_from_labels(cls, labels):
    """
    This is exactly the format which we expect when we read it in self._parse_vocab.

    :param list[str] labels:
    :rtype: dict[str,int]
    """
    d = {label: idx for (idx, label) in enumerate(labels)}
    assert len(d) == len(labels), "some labels are provided multiple times"
    return d

  @classmethod
  def create_vocab_from_labels(cls, labels):
    """
    Creates a `Vocabulary` from the given labels. Depending on whether the labels are identified as
    bytes, characters or words a `Utf8ByteTargets`, `CharacterTargets` or `Vocabulary` vocab is created.

    :param list[str] labels:
    :rtype: Vocabulary
    """
    if len(labels) < 1000 and all([len(label) == 1 for label in labels]):
      # are these actually ordered raw bytes? -> assume utf8
      if all([ord(label) <= 255 and ord(label) == idx for idx, label in enumerate(labels)]):
        return Utf8ByteTargets()
      return CharacterTargets(vocab_file=None, labels=labels, unknown_label=None)
    return Vocabulary(vocab_file=None, labels=labels, unknown_label=None)

  def tf_get_init_variable_func(self, var):
    """
    :param tensorflow.Variable var:
    :rtype: (tensorflow.Session)->None
    """
    import tensorflow as tf
    from returnn.tf.util.basic import VariableAssigner
    assert isinstance(var, tf.Variable)
    assert var.dtype.base_dtype == tf.string
    assert var.shape.as_list() == [self.num_labels]
    assert len(self.labels) == self.num_labels

    def init_vocab_var(session):
      """
      :param tensorflow.Session session:
      """
      VariableAssigner(var).assign(session=session, value=self.labels)

    return init_vocab_var

  def get_seq(self, sentence):
    """
    :param str sentence: assumed to be seq of vocab entries separated by whitespace
    :rtype: list[int]
    """
    segments = sentence.split()
    return self.get_seq_indices(segments) + self.seq_postfix

  def get_seq_indices(self, seq):
    """
    :param list[str] seq:
    :rtype: list[int]
    """
    if self.unknown_label is not None:
      return [self.vocab.get(k, self.unknown_label_id) for k in seq]
    return [self.vocab[k] for k in seq]

  def get_seq_labels(self, seq):
    """
    :param list[int]|numpy.ndarray seq: 1D sequence
    :rtype: str
    """
    return " ".join(map(self.labels.__getitem__, seq))


class BytePairEncoding(Vocabulary):
  """
  Vocab based on Byte-Pair-Encoding (BPE).
  This will encode the text on-the-fly with BPE.

  Reference:
  Rico Sennrich, Barry Haddow and Alexandra Birch (2016). Neural Machine Translation of Rare Words with Subword Units.
  Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
  """

  def __init__(self, vocab_file, bpe_file, seq_postfix=None, unknown_label="UNK"):
    """
    :param str vocab_file:
    :param str bpe_file:
    :param list[int]|None seq_postfix: labels will be added to the seq in self.get_seq
    :param str|None unknown_label:
    """
    super(BytePairEncoding, self).__init__(vocab_file=vocab_file, seq_postfix=seq_postfix, unknown_label=unknown_label)
    from returnn.util.bpe import StandardBytePairEncoder
    self.bpe = StandardBytePairEncoder(bpe_codes_file=bpe_file, labels=self.labels)

  def get_seq(self, sentence):
    """
    :param str sentence:
    :rtype: list[int]
    """
    segments = self.bpe.segment_sentence(sentence)
    seq = self.get_seq_indices(segments)
    return seq + self.seq_postfix


class SamplingBytePairEncoding(Vocabulary):
  """
  Vocab based on Byte-Pair-Encoding (BPE).
  Like :class:`BytePairEncoding`, but here we randomly sample from different possible BPE splits.
  This will encode the text on-the-fly with BPE.
  """

  def __init__(self, vocab_file, breadth_prob, seq_postfix=None, unknown_label="UNK"):
    """
    :param str vocab_file:
    :param float breadth_prob:
    :param list[int]|None seq_postfix: labels will be added to the seq in self.get_seq
    :param str|None unknown_label:
    """
    super(SamplingBytePairEncoding, self).__init__(
      vocab_file=vocab_file, seq_postfix=seq_postfix, unknown_label=unknown_label)
    from returnn.util.bpe import SamplingBytePairEncoder
    self.rnd = numpy.random.RandomState(0)
    self.bpe = SamplingBytePairEncoder(
      labels=self.labels, breadth_prob=breadth_prob, rnd=self.rnd, unknown_label=unknown_label)

  def set_random_seed(self, seed):
    """
    :param int seed:
    """
    self.rnd.seed(seed)

  def get_seq(self, sentence):
    """
    :param str sentence:
    :rtype: list[int]
    """
    segments = self.bpe.segment_sentence(sentence)
    seq = self.get_seq_indices(segments)
    return seq + self.seq_postfix


class SentencePieces(Vocabulary):
  """
  Uses the SentencePiece software,
  which supports different kind of subword units (including BPE, unigram, ...).

  https://github.com/google/sentencepiece/
  https://github.com/google/sentencepiece/tree/master/python

  Dependency::

    pip3 install --user sentencepiece

  """

  def __init__(self, **opts):
    """
    :param str model_file: The sentencepiece model file path.
    :param str model_proto: The sentencepiece model serialized proto.
    :param type out_type: output type. int or str. (Default = int)
    :param bool add_bos: Add <s> to the result (Default = false)
    :param bool add_eos: Add </s> to the result (Default = false)
      <s>/</s> is added after reversing (if enabled).
    :param bool reverse: Reverses the tokenized sequence (Default = false)
    :param bool enable_sampling: (Default = false)
    :param int nbest_size: sampling parameters for unigram. Invalid for BPE-Dropout.
      nbest_size = {0,1}: No sampling is performed.
      nbest_size > 1: samples from the nbest_size results.
      nbest_size < 0: (Default). assuming that nbest_size is infinite and samples
        from the all hypothesis (lattice) using
        forward-filtering-and-backward-sampling algorithm.
    :param float alpha: Soothing parameter for unigram sampling, and dropout probability of
      merge operations for BPE-dropout. (Default = 0.1)
    """
    import sentencepiece as spm  # noqa
    self._opts = opts
    self._cache_key = opts.get("model_file", None)
    self.sp = spm.SentencePieceProcessor(**opts)  # noqa
    super(SentencePieces, self).__init__(
      vocab_file=None, seq_postfix=None, unknown_label=self.sp.IdToPiece(self.sp.unk_id()))

  def __repr__(self):
    return "SentencePieces(%r)" % (self._opts,)

  def _parse_vocab(self):
    self.num_labels = self.sp.vocab_size()
    if self._cache_key and self._cache_key in self._cache:
      self.vocab, self.labels = self._cache[self._cache_key]
      assert self.unknown_label in self.vocab and self.num_labels == len(self.vocab) == len(self.labels)
      return
    self.labels = [self.sp.id_to_piece(i) for i in range(self.num_labels)]  # noqa
    self.vocab = {label: i for (i, label) in enumerate(self.labels)}
    if self._cache_key:
      self._cache[self._cache_key] = (self.vocab, self.labels)

  def set_random_seed(self, seed):
    """
    :param int seed:
    """
    # Unfortunately, there is only a global seed,
    # and also, it will only be used for new threads
    # where the random generator was not used yet...
    # https://github.com/google/sentencepiece/issues/635
    import sentencepiece as spm  # noqa
    spm.set_random_generator_seed(seed)

  def get_seq(self, sentence):
    """
    :param str sentence: assumed to be seq of vocab entries separated by whitespace
    :rtype: list[int]
    """
    return self.sp.encode(sentence, out_type=int)  # noqa


class CharacterTargets(Vocabulary):
  """
  Uses characters as target labels.
  Also see :class:`Utf8ByteTargets`.
  """

  def __init__(self, vocab_file, seq_postfix=None, unknown_label="@", labels=None):
    """
    :param str|None vocab_file:
    :param list[int]|None seq_postfix: labels will be added to the seq in self.get_seq
    :param str|None unknown_label:
    :param list[str]|None labels:
    """
    super(CharacterTargets, self).__init__(
      vocab_file=vocab_file, seq_postfix=seq_postfix, unknown_label=unknown_label, labels=labels)

  def get_seq(self, sentence):
    """
    :param str sentence:
    :rtype: list[int]
    """
    if self.unknown_label is not None:
      seq = [self.vocab.get(k, self.unknown_label_id) for k in sentence]
    else:
      seq = [self.vocab[k] for k in sentence]
    return seq + self.seq_postfix

  def get_seq_labels(self, seq):
    """
    :param list[int]|numpy.ndarray seq: 1D sequence
    :rtype: str
    """
    return "".join(map(self.labels.__getitem__, seq))


class Utf8ByteTargets(Vocabulary):
  """
  Uses bytes as target labels from UTF8 encoded text. All bytes (0-255) are allowed.
  Also see :class:`CharacterTargets`.
  """

  def __init__(self, seq_postfix=None):
    """
    :param list[int]|None seq_postfix: labels will be added to the seq in self.get_seq
    """
    super(Utf8ByteTargets, self).__init__(vocab_file=None, seq_postfix=seq_postfix, unknown_label=None)

  def _parse_vocab(self):
    """
    Sets self.vocab, self.labels, self.num_labels.
    """
    self.vocab = {chr(i): i for i in range(256)}
    self.labels = [chr(i) for i in range(256)]
    self.num_labels = 256

  def get_seq(self, sentence):
    """
    :param str sentence:
    :rtype: list[int]
    """
    if sys.version_info[0] >= 3:
      seq = list(sentence.encode("utf8"))
    else:
      seq = list(bytearray(sentence.encode("utf8")))
    return seq + self.seq_postfix

  def get_seq_labels(self, seq):
    """
    :param list[int]|numpy.ndarray seq: 1D sequence
    :rtype: str
    """
    return bytearray(seq).decode(encoding="utf8")
