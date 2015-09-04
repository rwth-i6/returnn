
import sys
from Dataset import Dataset
from GeneratingDataset import GeneratingDataset, StaticDataset
import gzip
import xml.etree.ElementTree as etree
from Util import parse_orthography
from Log import log
import numpy


class LmDataset(StaticDataset):

  def __init__(self, corpus_file, orth_symbols_file, **kwargs):
    self.orth_symbols = open(orth_symbols_file).read().splitlines()

    if len(self.orth_symbols) <= 256:
      self.dtype = "int8"
    else:
      self.dtype = "int32"

    data = []
    if _is_bliss(corpus_file):
      iter_f = _iter_bliss
    else:
      iter_f = _iter_txt
    def callback(orth):
      orth_syms = parse_orthography(orth)
      targets = numpy.array([self.orth_symbols.index(s) for s in orth_syms], dtype=self.dtype)
      data.append({"data": targets})
    iter_f(corpus_file, callback)

    super(LmDataset, self).__init__(data=data, output_dim={"data": len(self.orth_symbols)}, **kwargs)

  def get_data_dtype(self, key):
    return self.dtype

  def get_data_dim(self, key):
    return 1


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
        root.clear() # free memory

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
  log.initialize()
  dataset = LmDataset(*argv)
  print >>log.v3, "dataset len:", dataset.len_info()


if __name__ == "__main__":
  _main(sys.argv[1:])
