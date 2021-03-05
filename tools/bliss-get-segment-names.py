#!/usr/bin/env python3

"""
Goes through a Sprint Bliss XML file,
and prints out segment names on stdout.

Additionally, it provides functionality to merge consecutive sequences together,
such that you increase the average sequence length.
This can be used for :class:`ConcatSeqsDataset`.
"""

from __future__ import print_function

import os
import sys
from argparse import ArgumentParser
import gzip
from xml.etree import ElementTree
import _setup_returnn_env  # noqa


class BlissItem:
  """
  Bliss item.
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
  arg_parser.add_argument("bliss_filename")
  arg_parser.add_argument("--subset_segment_file")
  arg_parser.add_argument("--output_type", default="", help="e.g. segment_name")
  arg_parser.add_argument("--merge_swb_ab", action="store_true")
  arg_parser.add_argument("--sort_by_time", action="store_true")
  arg_parser.add_argument("--merge_segs_up_to_time", type=float)
  args = arg_parser.parse_args()
  subset_segment_list = None
  if args.subset_segment_file:
    subset_segment_list = set(open(args.subset_segment_file).read().splitlines())
  rec_filenames = set()
  items_by_rec = {}
  for bliss_item in iter_bliss(args.bliss_filename):
    if subset_segment_list and bliss_item.segment_name not in subset_segment_list:
      continue
    rec_name = bliss_item.recording_filename
    assert rec_name, "invalid item %r" % bliss_item
    if args.merge_swb_ab:
      rec_name = os.path.basename(rec_name)
      rec_name, _ = os.path.splitext(rec_name)
      rec_filenames.add(rec_name)
      assert rec_name[-1] in "AB"
      rec_name = rec_name[:-1]
    else:
      rec_filenames.add(rec_name)
    items_by_rec.setdefault(rec_name, []).append(bliss_item)
  assert items_by_rec
  if args.merge_swb_ab:
    if subset_segment_list:
      for key in list(items_by_rec.keys()):
        if key + "A" not in rec_filenames or key + "B" not in rec_filenames:
          del items_by_rec[key]
      assert items_by_rec, "rec_filenames %r" % (rec_filenames,)
    else:
      for key in items_by_rec.keys():
        assert key + "A" in rec_filenames
        assert key + "B" in rec_filenames
  for key, ls in items_by_rec.items():
    assert isinstance(ls, list)
    if args.sort_by_time:
      ls.sort(key=lambda item: item.start_time)
  if args.merge_segs_up_to_time:
    for key, ls in items_by_rec.items():
      i = 0
      while i < len(ls):
        j = i + 1
        dt = ls[i].delta_time
        while j < len(ls):
          if dt + ls[j].delta_time > args.merge_segs_up_to_time:
            break
          dt += ls[j].delta_time
          j += 1
        if j > i + 1:
          ls[i:j] = [BlissItem(
            segment_name=";".join([item.segment_name for item in ls[i:j]]),
            recording_filename=ls[i].recording_filename,  # might be wrong if merge_swb_ab...
            start_time=0.0, end_time=dt,  # dummy times
            orth=" ".join([item.orth for item in ls[i:j]]))]
        i += 1
  output_types = args.output_type.split(",")
  for key, ls in items_by_rec.items():
    assert isinstance(ls, list)
    for item in ls:
      assert isinstance(item, BlissItem)
      if not output_types:
        print(item)
      else:
        print(" ".join([str(getattr(item, key)) for key in output_types]))


if __name__ == "__main__":
  from returnn.util import better_exchook
  better_exchook.install()
  try:
    main()
  except BrokenPipeError:
    print("BrokenPipeError")
    sys.exit(1)
