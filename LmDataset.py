
import sys
from Dataset import Dataset
from GeneratingDataset import GeneratingDataset, StaticDataset
import gzip
import xml.etree.ElementTree as etree
from Util import parse_orthography, parse_orthography_into_symbols, load_json
from Log import log
import numpy
import time


class LmDataset(StaticDataset):

  def __init__(self, corpus_file, orth_symbols_file, replace_map_file=None, **kwargs):
    orth_symbols = open(orth_symbols_file).read().splitlines()
    if replace_map_file:
      replace_map = load_json(filename=replace_map_file)
      assert isinstance(replace_map, dict)
      replace_map = {key: parse_orthography_into_symbols(v)
                     for (key, v) in replace_map.items()}
    else:
      replace_map = {}
    orth_symbols_map = {sym: i for (i, sym) in enumerate(orth_symbols)}

    if len(orth_symbols) <= 256:
      dtype = "int8"
    else:
      dtype = "int32"

    if _is_bliss(corpus_file):
      iter_f = _iter_bliss
    else:
      iter_f = _iter_txt
    orths = []
    iter_f(corpus_file, orths.append)

    total_len = 0
    data = []
    last_log_time = time.time()
    for i, orth in enumerate(orths):
      orth_syms = parse_orthography(orth)
      orth_syms = sum([replace_map.get(s, [s]) for s in orth_syms], [])
      i = 0
      while i < len(orth_syms) - 1:
        if orth_syms[i:i+2] == [" ", " "]:
          orth_syms[i:i+2] = [" "]  # collapse two spaces
        else:
          i += 1
      targets = numpy.array(map(orth_symbols_map.__getitem__, orth_syms), dtype=dtype)
      total_len += len(targets)
      data.append({"data": targets})
      if time.time() - last_log_time > 2.0:
        last_log_time = time.time()
        print >> log.v5, "Loading %s progress, %i/%i (%.0f%%) seqs loaded, total syms %i ..." % (
                         self.__class__.__name__, len(data), len(orths), 100.0 * len(data) / len(orths), total_len)

    super(LmDataset, self).__init__(data=data, output_dim={"data": len(orth_symbols)}, **kwargs)
    self.dtype = dtype
    self.total_len = total_len
    self.replace_map = replace_map
    self.orth_symbols = orth_symbols
    self.labels["data"] = orth_symbols

  def get_data_dtype(self, key):
    return self.dtype

  def get_data_dim(self, key):
    return 1

  def get_num_timesteps(self):
    return self.total_len


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
    l = l.strip()
    if not l: continue
    callback(l)


def _main(argv):
  import better_exchook
  better_exchook.install()
  log.initialize(verbosity=[5])
  dataset = LmDataset(*argv)
  print >>log.v3, "dataset len:", dataset.len_info()


if __name__ == "__main__":
  _main(sys.argv[1:])
