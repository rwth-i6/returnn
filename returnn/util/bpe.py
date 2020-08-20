
"""
Provide basic Byte-Pair-Encoding (BPE) utilities.
"""

import re


class StandardBytePairEncoder:
  """
  Code is partly taken from subword-nmt/apply_bpe.py.
  Author: Rico Sennrich, code under MIT license.

  Use operations learned with learn_bpe.py to encode a new text.
  The text will not be smaller, but use only a fixed vocabulary, with rare words
  encoded as variable-length sequences of subword units.

  Reference:
  Rico Sennrich, Barry Haddow and Alexandra Birch (2016). Neural Machine Translation of Rare Words with Subword Units.
  Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.

  """

  def __init__(self, bpe_codes_file, labels=None):
    """
    :param str bpe_codes_file: codes file
    :param list[str]|None labels: vocab
    """
    self.labels = labels
    # check version information
    bpe_file_first_line = open(bpe_codes_file, "r").readline()
    if bpe_file_first_line.startswith('#version:'):
      self._bpe_file_version = tuple(
        [int(x) for x in re.sub(r'(\.0+)*$', '', bpe_file_first_line.split()[-1]).split(".")])
    else:
      self._bpe_file_version = (0, 1)
    self._bpe_codes = [tuple(item.split()) for item in open(bpe_codes_file, "rb").read().decode("utf8").splitlines()]
    # some hacking to deal with duplicates (only consider first instance)
    self._bpe_codes = dict([(code, i) for (i, code) in reversed(list(enumerate(self._bpe_codes)))])
    self._bpe_codes_reverse = dict([(pair[0] + pair[1], pair) for pair, i in self._bpe_codes.items()])
    self._bpe_encode_cache = {}
    self._bpe_separator = '@@'

  @staticmethod
  def _get_pairs(word):
    """
    :param tuple[str] word: represented as tuple of symbols (symbols being variable-length strings)
    :return: set of symbol pairs in a word
    :rtype: set[(str,str)]
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
      pairs.add((prev_char, char))
      prev_char = char
    return pairs

  def _encode_word(self, orig):
    """
    Encode word based on list of BPE merge operations, which are applied consecutively.
    :param str orig:
    :rtype: tuple[str]
    """

    if orig in self._bpe_encode_cache:
      return self._bpe_encode_cache[orig]

    if self._bpe_file_version == (0, 1):
      word = tuple(orig) + ('</w>',)
    elif self._bpe_file_version == (0, 2):  # more consistent handling of word-final segments
      word = tuple(orig[:-1]) + (orig[-1] + '</w>',)
    else:
      raise NotImplementedError

    pairs = self._get_pairs(word)
    if not pairs:
      return orig

    while True:
      bigram = min(pairs, key=lambda pair: self._bpe_codes.get(pair, float('inf')))
      if bigram not in self._bpe_codes:
        break
      first, second = bigram
      new_word = []
      i = 0
      while i < len(word):
        try:
          j = word.index(first, i)
          new_word.extend(word[i:j])
          i = j
        except ValueError:
          new_word.extend(word[i:])
          break

        if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
          new_word.append(first + second)
          i += 2
        else:
          new_word.append(word[i])
          i += 1
      new_word = tuple(new_word)
      word = new_word
      if len(word) == 1:
        break
      else:
        pairs = self._get_pairs(word)

    # don't print end-of-word symbols
    if word[-1] == '</w>':
      word = word[:-1]
    elif word[-1].endswith('</w>'):
      word = word[:-1] + (word[-1].replace('</w>', ''),)

    if self.labels:
      word = self._check_vocab_and_split(word, self._bpe_codes_reverse, self.labels, self._bpe_separator)

    self._bpe_encode_cache[orig] = word
    return word

  def _check_vocab_and_split(self, orig, bpe_codes, vocab, separator):
    """
    Check for each segment in word if it is in-vocabulary,
    and segment OOV segments into smaller units by reversing the BPE merge operations
    """

    out = []

    for segment in orig[:-1]:
      if segment + separator in vocab:
        out.append(segment)
      else:
        # sys.stderr.write('OOV: {0}\n'.format(segment))
        for item in self._recursive_split(segment, bpe_codes, vocab, separator, False):
          out.append(item)

    segment = orig[-1]
    if segment in vocab:
      out.append(segment)
    else:
      # sys.stderr.write('OOV: {0}\n'.format(segment))
      for item in self._recursive_split(segment, bpe_codes, vocab, separator, True):
        out.append(item)

    return out

  def _recursive_split(self, segment, bpe_codes, vocab, separator, final=False):
    """Recursively split segment into smaller units (by reversing BPE merges)
    until all units are either in-vocabulary, or cannot be split further."""

    # noinspection PyBroadException
    try:
      if final:
        left, right = bpe_codes[segment + '</w>']
        right = right[:-4]
      else:
        left, right = bpe_codes[segment]
    except Exception:  # TODO fix
      # sys.stderr.write('cannot split {0} further.\n'.format(segment))
      yield segment
      return

    if left + separator in vocab:
      yield left
    else:
      for item in self._recursive_split(left, bpe_codes, vocab, separator, False):
        yield item

    if (final and right in vocab) or (not final and right + separator in vocab):
      yield right
    else:
      for item in self._recursive_split(right, bpe_codes, vocab, separator, final):
        yield item

  def segment_sentence(self, sentence):
    """
    Segment single sentence (whitespace-tokenized string) with BPE encoding.

    :param str sentence:
    :rtype: list[str]
    """

    output = []

    found_category = False
    skip_category = False

    for word in sentence.split():
      if word[0] == '$' and len(word) > 1:
        found_category = True
        output.append(word)
      elif found_category is True and word[0] == '{':
        skip_category = True
        output.append(word)
      elif skip_category is True and word[0] != '}':
        output.append(word)
      else:
        found_category = False
        skip_category = False
        new_word = self._encode_word(word)

        for item in new_word[:-1]:
          output.append(item + self._bpe_separator)
        output.append(new_word[-1])

    return output
