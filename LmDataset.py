
import sys
from Dataset import DatasetSeq
from CachedDataset2 import CachedDataset2
import gzip
import xml.etree.ElementTree as etree
from Util import parse_orthography, parse_orthography_into_symbols, load_json, NumbersDict
from Log import log
import numpy
import time


class LmDataset(CachedDataset2):

  def __init__(self, corpus_file, orth_symbols_file, orth_replace_map_file=None, log_skipped_seqs=False, **kwargs):
    super(LmDataset, self).__init__(**kwargs)

    orth_symbols = open(orth_symbols_file).read().splitlines()
    if orth_replace_map_file:
      orth_replace_map = load_json(filename=orth_replace_map_file)
      assert isinstance(orth_replace_map, dict)
      self.orth_replace_map = {key: parse_orthography_into_symbols(v)
                               for (key, v) in orth_replace_map.items()}
    else:
      self.orth_replace_map = {}
    self.orth_symbols_map = {sym: i for (i, sym) in enumerate(orth_symbols)}

    if len(orth_symbols) <= 256:
      self.dtype = "int8"
    else:
      self.dtype = "int32"
    self.num_outputs = {"data": [len(orth_symbols), 1]}
    self.num_inputs = self.num_outputs["data"][0]
    self.orth_symbols = orth_symbols
    self.labels["data"] = orth_symbols
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


def _main(argv):
  import better_exchook
  better_exchook.install()
  log.initialize(verbosity=[5])
  dataset = LmDataset(*argv)
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
