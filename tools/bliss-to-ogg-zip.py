#!/usr/bin/env python3

from __future__ import print_function

import typing
import os
import sys
from argparse import ArgumentParser
from decimal import Decimal
import tempfile
import gzip
import xml.etree.ElementTree as ElementTree
import zipfile
import shutil
from subprocess import check_call


my_dir = os.path.dirname(os.path.abspath(__file__))
returnn_dir = os.path.dirname(my_dir)
sys.path.insert(0, returnn_dir)


class BlissItem:
  def __init__(self, segment_name, recording_filename, start_time, end_time, orth, speaker_name=None):
    """
    :param str segment_name:
    :param str recording_filename:
    :param Decimal start_time:
    :param Decimal end_time:
    :param str orth:
    :param str|None speaker_name:
    """
    self.segment_name = segment_name
    self.recording_filename = recording_filename
    self.start_time = start_time
    self.end_time = end_time
    self.orth = orth
    self.speaker_name = speaker_name

  def __repr__(self):
    keys = ["segment_name", "recording_filename", "start_time", "end_time", "orth", "speaker_name"]
    return "BlissItem(%s)" % ", ".join(["%s=%r" % (key, getattr(self, key)) for key in keys])

  @property
  def delta_time(self):
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
      orth_raw = elem_orth.text  # should be unicode
      orth_split = orth_raw.split()
      orth = " ".join(orth_split)
      elem_speaker = elem.find("speaker")
      if elem_speaker is not None:
        speaker_name = elem_speaker.attrib["name"]
      else:
        speaker_name = None
      segment_name = "/".join(name_tree)
      yield BlissItem(
        segment_name=segment_name, recording_filename=recording_filename,
        start_time=Decimal(elem.attrib["start"]), end_time=Decimal(elem.attrib["end"]),
        orth=orth, speaker_name=speaker_name)
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


def longest_common_prefix(strings):
  """
  :param list[str]|set[str] strings:
  :rtype: str
  """
  if not strings:
      return ""
  min_s = min(strings)
  max_s = max(strings)
  if not min_s:
    return ""
  for i in range(len(min_s)):
    if max_s[i] != min_s[i]:
      return max_s[:i]
  return min_s[:]


def longest_common_postfix(strings):
  """
  :param list[str]|set[str] strings:
  :rtype: str
  """
  strings = ["".join(reversed(s)) for s in strings]
  res = longest_common_prefix(strings)
  return "".join(reversed(res))


def hms(s):
  """
  :param float|int s: seconds
  :return: e.g. "1:23:45" (hs:ms:secs). see hms_fraction if you want to get fractional seconds
  :rtype: str
  """
  m, s = divmod(s, 60)
  h, m = divmod(m, 60)
  return "%d:%02d:%02d" % (h, m, s)


def main():
  arg_parser = ArgumentParser()
  arg_parser.add_argument("bliss_filename")
  arg_parser.add_argument("--subset_segment_file")
  arg_parser.add_argument("--no_ogg", help="skip generating ogg files", action="store_true")
  arg_parser.add_argument("--no_cleanup", help="don't delete our temp files", action="store_true")
  arg_parser.add_argument("--output", help="output zip filename (if empty, dummy run)", required=True)
  args = arg_parser.parse_args()
  subset_segment_list = None
  if args.subset_segment_file:
    subset_segment_list = set(open(args.subset_segment_file).read().splitlines())
    assert subset_segment_list

  rec_filenames = set()
  seqs = []  # type: typing.List[BlissItem]
  for bliss_item in iter_bliss(args.bliss_filename):
    if subset_segment_list and bliss_item.segment_name not in subset_segment_list:
      continue
    seqs.append(bliss_item)
    rec_filenames.add(bliss_item.recording_filename)
  assert seqs
  if subset_segment_list:
    seq_names = set([seq.segment_name for seq in seqs])
    for seq_name in subset_segment_list:
      assert seq_name in seq_names
  print("Num seqs:", len(seqs))
  print("Num recordings:", len(rec_filenames))
  rec_filename_common_prefix = longest_common_prefix(rec_filenames)
  if not rec_filename_common_prefix.endswith("/"):
    if "/" in rec_filename_common_prefix:
      rec_filename_common_prefix = rec_filename_common_prefix[:rec_filename_common_prefix.rfind("/") + 1]
    else:
      rec_filename_common_prefix = ""
  print("Recordings common dir prefix:", rec_filename_common_prefix)
  rec_filename_common_postfix = longest_common_postfix(rec_filenames)
  if not rec_filename_common_postfix.startswith("."):
    if "." in rec_filename_common_postfix:
      rec_filename_common_postfix = rec_filename_common_postfix[rec_filename_common_postfix.find("."):]
    else:
      rec_filename_common_postfix = ""
  print("Recordings common postfix:", rec_filename_common_postfix)

  if args.output:
    zip_filename = args.output
    name, ext = os.path.splitext(os.path.basename(zip_filename))
    assert ext == ".zip"
  else:
    name = "dummy"
    zip_filename = None
  print("Dataset name:", name)

  total_duration = Decimal(0)
  total_num_chars = 0
  temp_dir = tempfile.mkdtemp()

  print("Temp dir for data:", temp_dir)
  dest_dirname = "%s/%s" % (temp_dir, name)
  dest_meta_filename = "%s/%s.txt" % (temp_dir, name)
  dest_meta_file = open(dest_meta_filename, "w")
  dest_meta_file.write("[\n")
  os.makedirs(dest_dirname, exist_ok=True)

  for seq in seqs:
    rec_filename = seq.recording_filename
    assert os.path.isfile(rec_filename)
    assert seq.start_time < seq.end_time and seq.delta_time > 0
    duration = seq.delta_time
    assert duration > 0
    total_duration += duration
    assert rec_filename.startswith(rec_filename_common_prefix) and rec_filename.endswith(rec_filename_common_postfix)
    rec_name = rec_filename[len(rec_filename_common_prefix):-len(rec_filename_common_postfix)]
    dest_filename = "%s/%s/%s_%s.ogg" % (dest_dirname, rec_name, seq.start_time, seq.end_time)
    os.makedirs(os.path.dirname(dest_filename), exist_ok=True)
    if args.no_ogg:
      print("no Ogg (%s -> %s)" % (os.path.basename(rec_filename), dest_filename[len(dest_dirname) + 1:]))
    else:
      if os.path.exists(dest_filename):
        print("already exists, delete: %s" % os.path.basename(dest_filename))
        os.remove(dest_filename)
      cmd = ["ffmpeg", "-i", rec_filename, "-ss", str(seq.start_time), "-t", str(duration), dest_filename]
      print("$ %s" % " ".join(cmd))
      check_call(cmd)
    dest_meta_file.write("{'text': %r, 'speaker_name': %r, 'file': %r, 'duration': %s},\n" % (
      seq.orth, seq.speaker_name, dest_filename[len(dest_dirname) + 1:], duration))
    total_num_chars += len(seq.orth)
  dest_meta_file.write("]\n")
  dest_meta_file.close()
  print("Total duration:", total_duration, "secs", "(%s)" % hms(total_duration))
  print("Total num chars:", total_num_chars)

  print("Dataset zip filename:", zip_filename if zip_filename else "(dummy run, no zip file)")
  if zip_filename:
    print("Zipping...")
    zip_file = zipfile.ZipFile(zip_filename, mode="a", compression=zipfile.ZIP_DEFLATED)
    for dirpath, dirnames, filenames in os.walk(temp_dir):
      for name in sorted(dirnames + filenames):
        path = "%s/%s" % (dirpath, name)
        assert path.startswith(temp_dir + "/")
        zip_path = path[len(temp_dir) + 1:]
        print(" Adding:", zip_path)
        zip_file.write(path, zip_path)

  if not args.no_cleanup:
    print("Cleaning up...")
    shutil.rmtree(temp_dir)
  else:
    print("Keeping temp dir:", temp_dir)

  print("Finished.")


if __name__ == "__main__":
  import better_exchook
  better_exchook.install()
  try:
    main()
  except BrokenPipeError:
    print("BrokenPipeError")
    sys.exit(1)
