#!/usr/bin/env python3

"""
Goes through a Sprint Bliss XML file,
and writes out sequence lengths based on the start/end time,
into a Python-formatted file.
"""

from __future__ import print_function

import sys
from argparse import ArgumentParser
import gzip
from xml.etree import ElementTree
import itertools
import _setup_returnn_env  # noqa


class BlissItem:
  """
  Represents one entry in the Bliss XML.
  """
  def __init__(self, segment_name, recording_filename, start_time, end_time, orth):
    """
    :param str segment_name:
    :param str recording_filename:
    :param float start_time:
    :param float end_time:
    :param str orth:
    """
    self.segment_name = segment_name
    self.recording_filename = recording_filename
    self.start_time = start_time
    self.end_time = end_time
    self.orth = orth

  def __repr__(self):
    keys = ["segment_name", "recording_filename", "start_time", "end_time", "orth"]
    return "BlissItem(%s)" % ", ".join(["%s=%r" % (key, getattr(self, key)) for key in keys])

  @property
  def delta_time(self):
    """
    :rtype: float
    """
    return self.end_time - self.start_time


def iter_bliss(filename):
  """
  :param str filename:
  :return: yields BlissItem
  :rtype: list[BlissItem]
  """
  corpus_file = open(filename, 'rb')
  if filename.endswith(".gz"):
    corpus_file = gzip.GzipFile(fileobj=corpus_file)

  context = iter(ElementTree.iterparse(corpus_file, events=('start', 'end')))
  _, root = next(context)  # get root element
  name_tree = [root.attrib["name"]]
  elem_tree = [root]
  count_tree = [0]
  recording_filename = None
  for event, elem in context:
    if elem.tag == "recording":
      recording_filename = elem.attrib["audio"] if event == "start" else None
    if event == 'end' and elem.tag == "segment":
      elem_orth = elem.find("orth")
      orth_raw = elem_orth.text or ""  # should be unicode
      orth_split = orth_raw.split()
      orth = " ".join(orth_split)
      segment_name = "/".join(name_tree)
      yield BlissItem(
        segment_name=segment_name, recording_filename=recording_filename,
        start_time=float(elem.attrib["start"]), end_time=float(elem.attrib["end"]),
        orth=orth)
      root.clear()  # free memory
    if event == "start":
      count_tree[-1] += 1
      count_tree.append(0)
      elem_tree += [elem]
      elem_name = elem.attrib.get("name", None)
      if elem_name is None:
        elem_name = str(count_tree[-2])
      assert isinstance(elem_name, str)
      name_tree += [elem_name]
    elif event == "end":
      assert elem_tree[-1] is elem
      elem_tree = elem_tree[:-1]
      name_tree = name_tree[:-1]
      count_tree = count_tree[:-1]


def main():
  """
  Main entry.
  """
  arg_parser = ArgumentParser()
  arg_parser.add_argument("bliss_filename", nargs="+")
  arg_parser.add_argument("--output", default="/dev/stdout")
  args = arg_parser.parse_args()
  if args.output.endswith(".gz"):
    out = gzip.GzipFile(args.output, mode="wb")
  else:
    out = open(args.output, "wb")
  out.write(b"{\n")
  for bliss_item in itertools.chain(*[iter_bliss(fn) for fn in args.bliss_filename]):
    assert isinstance(bliss_item, BlissItem)
    seq_len = round(bliss_item.delta_time * 100.)  # assume 10ms frames, round
    out.write(b"%r: %i,\n" % (bliss_item.segment_name, seq_len))
  out.write(b"}\n")
  out.close()


if __name__ == "__main__":
  from returnn.util import better_exchook
  better_exchook.install()
  try:
    main()
  except BrokenPipeError:
    print("BrokenPipeError")
    sys.exit(1)
