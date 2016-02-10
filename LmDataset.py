
import sys
from Dataset import DatasetSeq
from CachedDataset2 import CachedDataset2
import gzip
import xml.etree.ElementTree as etree
from Util import parse_orthography, parse_orthography_into_symbols, load_json, NumbersDict
from Log import log
import numpy
import time
from random import Random


class LmDataset(CachedDataset2):

  def __init__(self,
               corpus_file, phone_info=None, orth_symbols_file=None, orth_replace_map_file=None,
               log_skipped_seqs=False, **kwargs):
    """
    :param str corpus_file: Bliss XML or line-based txt. optionally can be gzip.
    :param dict | None phone_info: if you want to get phone seqs, dict with lexicon_file etc. see _PhoneSeqGenerator
    :param str | None orth_symbols_file: list of orthography symbols, if you want to get orth symbol seqs
    :param str | None orth_replace_map_file: JSON file with replacement dict for orth symbols
    :param bool log_skipped_seqs: log skipped seqs
    """
    super(LmDataset, self).__init__(**kwargs)

    if orth_symbols_file:
      assert not phone_info
      orth_symbols = open(orth_symbols_file).read().splitlines()
      self.orth_symbols_map = {sym: i for (i, sym) in enumerate(orth_symbols)}
      self.orth_symbols = orth_symbols
      self.labels["data"] = orth_symbols
      self.seq_gen = None
    else:
      assert not orth_symbols_file
      assert isinstance(phone_info, dict)
      self.seq_gen = _PhoneSeqGenerator(**phone_info)
      self.orth_symbols = None
      self.labels["data"] = self.seq_gen.get_class_labels()
    if orth_replace_map_file:
      orth_replace_map = load_json(filename=orth_replace_map_file)
      assert isinstance(orth_replace_map, dict)
      self.orth_replace_map = {key: parse_orthography_into_symbols(v)
                               for (key, v) in orth_replace_map.items()}
    else:
      self.orth_replace_map = {}

    if len(self.labels["data"]) <= 256:
      self.dtype = "int8"
    else:
      self.dtype = "int32"
    self.num_outputs = {"data": [len(self.labels["data"]), 1]}
    self.num_inputs = self.num_outputs["data"][0]
    self.seq_order = None
    self.log_skipped_seqs = log_skipped_seqs

    if _is_bliss(corpus_file):
      iter_f = _iter_bliss
    else:
      iter_f = _iter_txt
    self.orths = []
    print >> log.v4, "LmDataset, loading file", corpus_file
    iter_f(corpus_file, self.orths.append)
    # It's only estimated because we might filter some out or so.
    self._estimated_num_seqs = len(self.orths)

  def get_data_dtype(self, key):
    return self.dtype

  def init_seq_order(self, epoch=None, seq_list=None):
    assert seq_list is None
    super(LmDataset, self).init_seq_order(epoch=epoch)
    self.seq_order = self.get_seq_order_for_epoch(
      epoch=epoch, num_seqs=len(self.orths), get_seq_len=lambda i: len(self.orths[i]))
    self.next_orth_idx = 0
    self.num_skipped = 0
    self._num_timesteps_accumulated = NumbersDict(0)
    if self.seq_gen:
      self.seq_gen.random_seed(epoch or 1)

  def _collect_single_seq(self, seq_idx):
    """
    :type seq_idx: int
    :rtype: DatasetSeq | None
    :returns DatasetSeq or None if seq_idx >= num_seqs.
    """
    while True:
      if self.next_orth_idx >= len(self.orths):
        return None
      orth = self.orths[self.seq_order[self.next_orth_idx]]
      self.next_orth_idx += 1
      if orth == "</s>": continue  # special sentence end symbol. empty seq, ignore.

      if self.seq_gen:
        try:
          phones = self.seq_gen.generate_seq(orth)
          print phones
        except KeyError as e:
          if self.log_skipped_seqs:
            print >> log.v4, "LmDataset: skipping sequence %r because of missing lexicon entry: %s" % (
                             orth, e)
          self.num_skipped += 1
          continue
        data = self.seq_gen.seq_to_class_idxs(phones, dtype=self.dtype)

      elif self.orth_symbols:
        orth_syms = parse_orthography(orth)
        orth_syms = sum([self.orth_replace_map.get(s, [s]) for s in orth_syms], [])
        i = 0
        while i < len(orth_syms) - 1:
          if orth_syms[i:i+2] == [" ", " "]:
            orth_syms[i:i+2] = [" "]  # collapse two spaces
          else:
            i += 1
        try:
          data = numpy.array(map(self.orth_symbols_map.__getitem__, orth_syms), dtype=self.dtype)
        except KeyError as e:
          if self.log_skipped_seqs:
            print >> log.v4, "LmDataset: skipping sequence %r because of missing orth symbol: %s" % (
                             "".join(orth_syms), e)
          self.num_skipped += 1
          continue

      else:
        assert False

      self._num_timesteps_accumulated += len(data)
      return DatasetSeq(seq_idx=seq_idx, features=data, targets=None)


def _is_bliss(filename):
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


def _iter_bliss(filename, callback):
  corpus_file = open(filename, 'rb')
  if filename.endswith(".gz"):
    corpus_file = gzip.GzipFile(fileobj=corpus_file)

  def getelements(tag):
    """Yield *tag* elements from *filename_or_file* xml incrementally."""
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
        root.clear()  # free memory

  for tree, elem in getelements("segment"):
    elem_orth = elem.find("orth")
    orth_raw = elem_orth.text  # should be unicode
    orth_split = orth_raw.split()
    orth = " ".join(orth_split)

    callback(orth)


def _iter_txt(filename, callback):
  f = open(filename, 'rb')
  if filename.endswith(".gz"):
    f = gzip.GzipFile(fileobj=f)

  for l in f:
    try:
      l = l.decode("utf8")
    except UnicodeDecodeError:
      l = l.decode("latin_1")  # or iso8859_15?
    l = l.strip()
    if not l: continue
    callback(l)


class _AllophoneState:
  # In Sprint, see AllophoneStateAlphabet::index().
  id = None  # u16 in Sprint. here just str
  context_history = ()  # list[u16] of phone id. here just list[str]
  context_future = ()  # list[u16] of phone id. here just list[str]
  boundary = 0  # s16. flags. 1 -> initial (@i), 2 -> final (@f)
  state = None  # s16, e.g. 0,1,2

  def format(self):
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

  def mark_initial(self):
    self.boundary = self.boundary | 1

  def mark_final(self):
    self.boundary = self.boundary | 2


class _Lexicon:

  def __init__(self, filename):
    print >> log.v4, "Loading lexicon", filename
    lex_file = open(filename, 'rb')
    if filename.endswith(".gz"):
      lex_file = gzip.GzipFile(fileobj=lex_file)
    self.phonemes = {}
    self.lemmas = {}

    context = iter(etree.iterparse(lex_file, events=('start', 'end')))
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
          variation = elem.find("variation").text.strip()
          assert symbol not in self.phonemes
          assert variation in ["context", "none"]
          self.phonemes[symbol] = {"index": len(self.phonemes), "symbol": symbol, "variation": variation}
          root.clear()  # free memory
        elif elem.tag == "phoneme-inventory":
          print >> log.v4, "Finished phoneme inventory, %i phonemes" % len(self.phonemes)
          root.clear()  # free memory
        elif elem.tag == "lemma":
          orth = elem.find("orth").text.strip()
          phons = [{"phon": e.text.strip(), "score": float(e.attrib.get("score", 0))} for e in elem.findall("phon")]
          assert orth not in self.lemmas
          self.lemmas[orth] = {"orth": orth, "phons": phons}
          root.clear()  # free memory
    print >> log.v4, "Finished whole lexicon, %i lemmas" % len(self.lemmas)


class _PhoneSeqGenerator:
  def __init__(self, lexicon_file, allo_num_states=3, allo_context_len=1,
               add_silence_beginning=0.1, add_silence_between_words=0.1, add_silence_end=0.1,
               repetition=0.5, silence_repetition=0.7):
    """
    :param str lexicon_file: lexicon XML file
    :param int allo_num_states: how much HMM states per allophone (all but silence)
    :param int allo_context_len: how much context to store left and right. 1 -> triphone
    :param float add_silence_beginning: prob of adding silence at beginning
    :param float add_silence_between_words: prob of adding silence between words
    :param float add_silence_end: prob of adding silence at end
    :param float repetition: prob of repeating an allophone
    :param float silence_repetition: prob of repeating the silence allophone
    """
    self.lexicon = _Lexicon(lexicon_file)
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

  def random_seed(self, seed):
    self.rnd.seed(seed)

  def get_class_labels(self):
    # TODO context, etc (allophones)
    return sorted(self.lexicon.phonemes.keys(), key=lambda s: self.lexicon.phonemes[s]["index"])

  def seq_to_class_idxs(self, phones, dtype=None):
    if dtype is None: dtype = "int32"
    # It should not happen that we don't have some phoneme. The lexicon should not be inconsistent.
    return numpy.array([self.lexicon.phonemes[p.id]["index"] for p in phones], dtype=dtype)

  def _iter_orth(self, orth):
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
    phones = []
    for lemma in self._iter_orth(orth):
      phon = self.rnd.choice(lemma["phons"])
      phones += [phon["phon"]]
    return " ".join(phones)

  def _phon_to_allos(self, phon):
    for p in phon.split():
      a = _AllophoneState()
      a.id = p
      yield a

  def _allos_add_states(self, allos):
    for _a in allos:
      if _a.id == self.si_phone:
        while True:
          a = _AllophoneState()
          a.id = _a.id
          a.context_history = _a.context_history
          a.context_future = _a.context_future
          a.boundary = _a.boundary
          a.state = 0  # silence only has one state
          yield a
          if self.rnd.random() >= self.silence_repetition:
            break
      else:  # non-silence
        for state in range(self.allo_num_states):
          while True:
            a = _AllophoneState()
            a.id = _a.id
            a.context_history = _a.context_history
            a.context_future = _a.context_future
            a.boundary = _a.boundary
            a.state = state
            yield a
            if self.rnd.random() >= self.repetition:
              break

  def _allos_set_context(self, allos):
    ctx = []
    for a in allos:
      if self.lexicon.phonemes[a.id]["variation"] == "context":
        a.context_history = tuple(ctx)
        ctx += [a.id]
        ctx = ctx[-self.allo_context_len:]
      else:
        ctx = []
    for a in reversed(allos):
      if self.lexicon.phonemes[a.id]["variation"] == "context":
        a.context_future = tuple(reversed(ctx))
        ctx += [a.id]
        ctx = ctx[-self.allo_context_len:]
      else:
        ctx = []

  def generate_seq(self, orth):
    allos = []
    for lemma in self._iter_orth(orth):
      phon = self.rnd.choice(lemma["phons"])
      l_allos = list(self._phon_to_allos(phon["phon"]))
      l_allos[0].mark_initial()
      l_allos[-1].mark_final()
      allos += l_allos
    self._allos_set_context(allos)
    allos = list(self._allos_add_states(allos))
    return allos


def _main(argv):
  import better_exchook
  better_exchook.install()
  log.initialize(verbosity=[5])
  dataset = LmDataset(**eval(argv[0]))
  dataset.init_seq_order(epoch=1)

  seq_idx = 0
  last_log_time = time.time()
  while dataset.is_less_than_num_seqs(seq_idx):
    dataset.load_seqs(seq_idx, seq_idx + 1)

    if time.time() - last_log_time > 2.0:
      last_log_time = time.time()
      print >> log.v5, "Loading %s progress, %i/%i (%.0f%%) seqs loaded (%.0f%% skipped), total syms %i ..." % (
                       dataset.__class__.__name__, dataset.next_orth_idx, len(dataset.orths),
                       100.0 * dataset.next_orth_idx / len(dataset.orths),
                       100.0 * dataset.num_skipped / (dataset.next_orth_idx or 1),
                       dataset._num_timesteps_accumulated["data"])

    seq_idx += 1

  print >>log.v3, "dataset len:", dataset.len_info()


if __name__ == "__main__":
  _main(sys.argv[1:])
