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

  def __init__(self, vocab_file, seq_postfix=None,
               unknown_label="UNK", bos_label=None, eos_label=None, pad_label=None,
               control_symbols=None, user_defined_symbols=None,
               num_labels=None, labels=None):
    """
    :param str|None vocab_file:
    :param str|int|None unknown_label: e.g. "UNK" or "<unk>"
    :param str|int|None bos_label: e.g. "<s>"
    :param str|int|None eos_label: e.g. "</s>"
    :param str|int|None pad_label: e.g. "<pad>"
    :param dict[str,str|int]|None control_symbols:
      https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md
    :param dict[str,str|int]|None user_defined_symbols:
      https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md
    :param int num_labels: just for verification
    :param list[int]|None seq_postfix: labels will be added to the seq in self.get_seq
    :param list[str]|None labels:
    """
    self.vocab_file = vocab_file
    self.unknown_label = unknown_label
    self.num_labels = None  # type: typing.Optional[int]  # will be set by _parse_vocab
    self._vocab = None  # type: typing.Optional[typing.Dict[str,int]]  # label->idx
    self._labels = labels

    self._parse_vocab()
    if num_labels is not None:
      assert self.num_labels == num_labels
    self.unknown_label_id = self.to_id(self.unknown_label, allow_none=True)
    if self.unknown_label_id is not None:
      self.unknown_label = self.id_to_label(self.unknown_label_id)
    self.bos_label_id = self.to_id(bos_label, allow_none=True)
    self.eos_label_id = self.to_id(eos_label, allow_none=True)
    self.pad_label_id = self.to_id(pad_label, allow_none=True)
    self.control_symbol_ids = {name: self.to_id(label) for name, label in (control_symbols or {}).items()}
    self.user_defined_symbol_ids = {name: self.to_id(label) for name, label in (user_defined_symbols or {}).items()}
    self.seq_postfix = seq_postfix or []

  def __repr__(self):
    parts = [repr(self.vocab_file), "num_labels=%s" % self.num_labels]
    if self.unknown_label_id is not None:
      parts.append("unknown_label=%r" % self.unknown_label)
    if self.bos_label_id is not None:
      parts.append("bos_label=%r" % self.id_to_label(self.bos_label_id))
    if self.eos_label_id is not None:
      parts.append("eos_label=%r" % self.id_to_label(self.eos_label_id))
    if self.pad_label_id is not None:
      parts.append("pad_label=%r" % self.id_to_label(self.pad_label_id))
    return "%s(%s)" % (self.__class__.__name__, ", ".join(parts))

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

    if self._labels is not None:
      self._vocab = {label: i for i, label in enumerate(self._labels)}
      self.num_labels = len(self._labels)
    elif filename in self._cache:
      self._vocab, self._labels = self._cache[filename]
      self.num_labels = len(self._labels)
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
      labels = {idx: label for (label, idx) in sorted(d.items())}
      min_label, max_label, num_labels = min(labels), max(labels), len(labels)
      assert 0 == min_label
      if num_labels - 1 < max_label:
        print("Vocab error: not all indices used? max label: %i" % max_label, file=log.v1)
        print("unused labels: %r" % ([i for i in range(max_label + 1) if i not in labels],), file=log.v2)
      assert num_labels - 1 == max_label
      self.num_labels = len(labels)
      self._vocab = d
      self._labels = [label for (idx, label) in sorted(labels.items())]
      self._cache[filename] = (self._vocab, self._labels)

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
  def create_vocab_from_labels(cls, labels, **kwargs):
    """
    Creates a `Vocabulary` from the given labels. Depending on whether the labels are identified as
    bytes, characters or words a `Utf8ByteTargets`, `CharacterTargets` or `Vocabulary` vocab is created.

    :param list[str] labels:
    :rtype: Vocabulary
    """
    kwargs = kwargs.copy()
    kwargs.setdefault("unknown_label", None)
    if len(labels) < 1000 and all([len(label) == 1 for label in labels]):
      # are these actually ordered raw bytes? -> assume utf8
      if all([ord(label) <= 255 and ord(label) == idx for idx, label in enumerate(labels)]):
        return Utf8ByteTargets()
      return CharacterTargets(vocab_file=None, labels=labels, **kwargs)
    return Vocabulary(vocab_file=None, labels=labels, **kwargs)

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
    assert len(self._labels) == self.num_labels

    def init_vocab_var(session):
      """
      :param tensorflow.Session session:
      """
      VariableAssigner(var).assign(session=session, value=self._labels)

    return init_vocab_var

  def to_id(self, label, default=KeyError, allow_none=False):
    """
    :param str|int|None label:
    :param str|type[KeyError]|None default:
    :param bool allow_none: whether label can be None. in this case, None is returned
    :rtype: int|None
    """
    if isinstance(label, str):
      return self.label_to_id(label, default=default)
    if isinstance(label, int):
      if self.is_id_valid(label):
        return label
      if default is KeyError:
        raise KeyError("invalid label id %i" % label)
      return None
    if label is None and allow_none:
      return None
    raise TypeError("invalid label type %r" % type(label))

  def label_to_id(self, label, default=KeyError):
    """
    :param str label:
    :param int|type[KeyError]|None default:
    :rtype: int|None
    """
    if default is KeyError:
      return self._vocab[label]
    return self._vocab.get(label, default)

  def id_to_label(self, idx, default=KeyError):
    """
    :param int idx:
    :param str|KeyError|None default:
    :rtype: str|None
    """
    if self.is_id_valid(idx):
      return self._labels[idx]
    if default is KeyError:
      raise KeyError("idx %i out of range" % idx)
    return default

  def is_id_valid(self, idx):
    """
    :param int idx:
    :rtype: bool
    """
    return 0 <= idx < len(self._labels)

  @property
  def labels(self):
    """
    :rtype: list[str]
    """
    return self._labels

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
      return [self._vocab.get(k, self.unknown_label_id) for k in seq]
    return [self._vocab[k] for k in seq]

  def get_seq_labels(self, seq):
    """
    :param list[int]|numpy.ndarray seq: 1D sequence
    :rtype: str
    """
    return " ".join(map(self._labels.__getitem__, seq))


class BytePairEncoding(Vocabulary):
  """
  Vocab based on Byte-Pair-Encoding (BPE).
  This will encode the text on-the-fly with BPE.

  Reference:
  Rico Sennrich, Barry Haddow and Alexandra Birch (2016). Neural Machine Translation of Rare Words with Subword Units.
  Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
  """

  def __init__(self, vocab_file, bpe_file, seq_postfix=None, **kwargs):
    """
    :param str vocab_file:
    :param str bpe_file:
    :param list[int]|None seq_postfix: labels will be added to the seq in self.get_seq
    """
    super(BytePairEncoding, self).__init__(vocab_file=vocab_file, seq_postfix=seq_postfix, **kwargs)
    from returnn.util.bpe import StandardBytePairEncoder
    self.bpe = StandardBytePairEncoder(bpe_codes_file=bpe_file, labels=self._labels)

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

  def __init__(self, vocab_file, breadth_prob, seq_postfix=None, **kwargs):
    """
    :param str vocab_file:
    :param float breadth_prob:
    :param list[int]|None seq_postfix: labels will be added to the seq in self.get_seq
    """
    super(SamplingBytePairEncoding, self).__init__(vocab_file=vocab_file, seq_postfix=seq_postfix, **kwargs)
    from returnn.util.bpe import SamplingBytePairEncoder
    self.rnd = numpy.random.RandomState(0)
    self.bpe = SamplingBytePairEncoder(
      labels=self._labels, breadth_prob=breadth_prob, rnd=self.rnd,
      unknown_label=self.id_to_label(self.unknown_label_id) if self.unknown_label_id is not None else None)

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
    :param dict[str,str|int]|None control_symbols:
      https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md
    :param dict[str,str|int]|None user_defined_symbols:
      https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md
    """
    import sentencepiece as spm  # noqa
    self._opts = opts
    self._cache_key = opts.get("model_file", None)
    control_symbols = opts.pop("control_symbols", None)
    user_defined_symbols = opts.pop("user_defined_symbols", None)
    self.sp = spm.SentencePieceProcessor(**opts)  # noqa
    super(SentencePieces, self).__init__(
      vocab_file=None, seq_postfix=None,
      unknown_label=self.sp.unk_id(), eos_label=self.sp.eos_id(), bos_label=self.sp.bos_id(),
      pad_label=self.sp.pad_id(), control_symbols=control_symbols, user_defined_symbols=user_defined_symbols)

  def __repr__(self):
    return "%s(%r)" % (self.__class__.__name__, self._opts)

  def _parse_vocab(self):
    self.num_labels = self.sp.vocab_size()
    # Do not load labels/vocab here. This is not really needed.

  @property
  def labels(self):
    """
    :rtype: list[str]
    """
    if self._cache_key and self._cache_key in self._cache:
      self._vocab, self._labels = self._cache[self._cache_key]
      assert self.num_labels == len(self._vocab) == len(self._labels)
    else:
      self._labels = [self.sp.id_to_piece(i) for i in range(self.num_labels)]  # noqa
      self._vocab = {label: i for (i, label) in enumerate(self._labels)}
      if self._cache_key:
        self._cache[self._cache_key] = (self._vocab, self._labels)
    return self._labels

  def is_id_valid(self, idx):
    """
    :param int idx:
    :rtype: bool
    """
    return not self.sp.IsUnused(idx)

  def id_to_label(self, idx, default=KeyError):
    """
    :param int idx:
    :param str|KeyError|None default:
    :rtype: str|None
    """
    if default is not KeyError and not self.is_id_valid(idx):
      return default
    return self.sp.IdToPiece(idx)

  def label_to_id(self, label, default=KeyError):
    """
    :param str label:
    :param int|type[KeyError]|None default:
    :rtype: int|None
    """
    res = self.sp.PieceToId(label)
    if res == self.unknown_label_id or res < 0 or res is None:
      # It could be that the label really is the unknown-label, or it could be that the label is unknown.
      if label == self.id_to_label(self.unknown_label_id):
        return self.unknown_label_id
      if default is KeyError:
        raise KeyError("label %r not found" % label)
      return default
    return res

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

  def __init__(self, vocab_file, seq_postfix=None, unknown_label="@", labels=None, **kwargs):
    """
    :param str|None vocab_file:
    :param list[int]|None seq_postfix: labels will be added to the seq in self.get_seq
    :param str|None unknown_label:
    :param list[str]|None labels:
    """
    super(CharacterTargets, self).__init__(
      vocab_file=vocab_file, seq_postfix=seq_postfix, unknown_label=unknown_label, labels=labels, **kwargs)

  def get_seq(self, sentence):
    """
    :param str sentence:
    :rtype: list[int]
    """
    if self.unknown_label is not None:
      seq = [self._vocab.get(k, self.unknown_label_id) for k in sentence]
    else:
      seq = [self._vocab[k] for k in sentence]
    return seq + self.seq_postfix

  def get_seq_labels(self, seq):
    """
    :param list[int]|numpy.ndarray seq: 1D sequence
    :rtype: str
    """
    return "".join(map(self._labels.__getitem__, seq))


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
    self._vocab = {chr(i): i for i in range(256)}
    self._labels = [chr(i) for i in range(256)]
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
