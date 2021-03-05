#!/usr/bin/env python3

"""
Goes through a Sprint Bliss XML file, converts the audio to ogg (via ffmpeg),
and creates a corresponding ZIP files.
The created files are compatible with the :class:`OggZipDataset`.
"""

from __future__ import print_function

import typing
import os
import sys
from argparse import ArgumentParser
from decimal import Decimal
import tempfile
import gzip
import numpy
import xml.etree.ElementTree as ElementTree
import zipfile
import shutil
from subprocess import check_call
from glob import glob

import _setup_returnn_env  # noqa
import returnn.sprint.cache


class BlissItem:
  """
  Bliss item.
  """
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

  parser = ElementTree.XMLParser(target=ElementTree.TreeBuilder(), encoding="utf-8")
  context = iter(ElementTree.iterparse(corpus_file, parser=parser, events=('start', 'end')))
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


class SprintCacheHandler:
  """
  This is just to apply the same silence trimming on the raw audio samples
  which was applied on the features in the Sprint cache.
  We can reconstruct this information because the Sprint cache also has the exact timing information.
  """

  def __init__(self, opt, bliss_opt, raw_sample_rate, feat_sample_rate):
    """
    :param str opt: either filename or filename pattern
    :param str bliss_opt: either filename or filename pattern
    :param int raw_sample_rate:
    :param int feat_sample_rate:
    """
    self.sprint_cache = self._load_sprint_cache(opt)
    self.seg_times = self._collect_seg_times_from_bliss(bliss_opt)
    self.raw_sample_rate = raw_sample_rate
    self.feat_sample_rate = feat_sample_rate
    self.pp_counter = 0

  @staticmethod
  def _load_sprint_cache(opt):
    """
    :param str opt: either filename or filename pattern
    :rtype: SprintCache.FileArchiveBundle|SprintCache.FileArchive
    """
    if "*" in opt:
      sprint_cache_fns = glob(opt)
      assert sprint_cache_fns, "nothing found under sprint cache pattern %r" % (opt,)
      sprint_cache = returnn.sprint.cache.FileArchiveBundle()
      for fn in sprint_cache_fns:
        print("Load Sprint cache:", fn)
        sprint_cache.add_bundle_or_archive(fn)
    else:
      print("Load Sprint cache:", opt)
      sprint_cache = returnn.sprint.cache.open_file_archive(opt, must_exists=True)
    return sprint_cache

  @staticmethod
  def _collect_seg_times_from_bliss(opt):
    """
    :param str opt: either filename or filename pattern
    :rtype: dict[str,(Decimal,Decimal)]
    """
    if "*" in opt:
      items = []
      fns = glob(opt)
      assert fns, "nothing found under Bliss XML cache pattern %r" % (opt,)
      for fn in fns:
        print("Load Bliss XML:", fn)
        items.extend(iter_bliss(fn))
    else:
      print("Load Bliss XML:", opt)
      items = list(iter_bliss(opt))
    return {seq.segment_name: (seq.start_time, seq.end_time) for seq in items}

  # noinspection PyUnusedLocal
  def feature_post_process(self, feature_data, seq_name, **kwargs):
    """
    :param numpy.ndarray feature_data:
    :param str seq_name:
    :return: features
    :rtype: numpy.ndarray
    """
    assert feature_data.shape[1] == 1  # raw audio
    self.pp_counter += 1
    assert self.raw_sample_rate % self.feat_sample_rate == 0
    num_frames_per_feat = self.raw_sample_rate // self.feat_sample_rate
    assert num_frames_per_feat % 2 == 0
    allowed_variance_num_frames = num_frames_per_feat // 2  # allow some variance
    times, data = self.sprint_cache.read(seq_name, "feat")
    assert len(times) == len(data)
    prev_end_frame = None
    res_feature_data = []
    seq_time_offset = float(self.seg_times[seq_name][0])
    for (start_time, end_time), feat in zip(times, data):
      start_time -= seq_time_offset
      end_time -= seq_time_offset
      center_time = (start_time + end_time) / 2.
      start_frame = int(center_time * self.raw_sample_rate) - num_frames_per_feat // 2
      assert 0 <= start_frame < feature_data.shape[0]
      if prev_end_frame is not None:
        if prev_end_frame - allowed_variance_num_frames <= start_frame <= prev_end_frame + allowed_variance_num_frames:
          start_frame = prev_end_frame
        assert start_frame >= prev_end_frame
      end_frame = start_frame + num_frames_per_feat
      if feature_data.shape[0] < end_frame <= feature_data.shape[0] + allowed_variance_num_frames:
        res_feature_data.append(feature_data[start_frame:])
        res_feature_data.append(numpy.zeros((end_frame - feature_data.shape[0], 1), dtype=feature_data.dtype))
      else:
        assert end_frame <= feature_data.shape[0]
        res_feature_data.append(feature_data[start_frame:end_frame])
      prev_end_frame = end_frame
    res_feature_data = numpy.concatenate(res_feature_data, axis=0)
    assert res_feature_data.shape[0] % num_frames_per_feat == 0
    assert res_feature_data.shape[0] // num_frames_per_feat == len(data)
    return res_feature_data


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
  """
  Main entry.
  """
  arg_parser = ArgumentParser()
  arg_parser.add_argument("bliss_filename")
  arg_parser.add_argument("--subset_segment_file")
  arg_parser.add_argument("--no_ogg", help="skip generating ogg files", action="store_true")
  arg_parser.add_argument(
    "--no_conversion", help="skip ffmpeg call, assume audio is correct already", action="store_true")
  arg_parser.add_argument("--no_cleanup", help="don't delete our temp files", action="store_true")
  arg_parser.add_argument("--sprint_cache", help="filename of feature cache for synchronization")
  arg_parser.add_argument("--raw_sample_rate", help="sample rate of audio input", type=int, default=8000)
  arg_parser.add_argument("--feat_sample_rate", help="sample rate of features for sync", type=int, default=100)
  arg_parser.add_argument("--ffmpeg_loglevel", help="loglevel for ffmpeg calls", type=str, default="info")
  arg_parser.add_argument("--ffmpeg_acodec", help="force audio codec for ffmpeg calls", type=str)
  arg_parser.add_argument("--number_of_channels", help="force number of channels for output audio", type=int, default=0)
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

  sprint_cache_handler = None
  if args.sprint_cache:
    sprint_cache_handler = SprintCacheHandler(
      opt=args.sprint_cache, bliss_opt=args.bliss_filename,
      raw_sample_rate=args.raw_sample_rate, feat_sample_rate=args.feat_sample_rate)

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
    if args.sprint_cache:
      wav_tmp_filename = "%s/%s/%s_%s.wav" % (dest_dirname, rec_name, seq.start_time, seq.end_time)
      os.makedirs(os.path.dirname(wav_tmp_filename), exist_ok=True)
      cmd = ["ffmpeg"]
      if args.ffmpeg_acodec:
        cmd += ["-acodec", args.ffmpeg_acodec]  # https://trac.ffmpeg.org/ticket/2810
      cmd += ["-i", rec_filename, "-ss", str(seq.start_time), "-t", str(duration)]
      if args.number_of_channels > 0:
        cmd += ["-ac", str(args.number_of_channels)]
      cmd += [wav_tmp_filename, "-loglevel", args.ffmpeg_loglevel]
      print("$ %s" % " ".join(cmd))
      check_call(cmd)
      import soundfile  # pip install pysoundfile
      audio, sample_rate = soundfile.read(wav_tmp_filename)
      assert sample_rate == args.raw_sample_rate
      audio_synced = sprint_cache_handler.feature_post_process(numpy.expand_dims(audio, axis=1), seq.segment_name)
      soundfile.write(wav_tmp_filename, audio_synced, args.raw_sample_rate)
      source_filename = wav_tmp_filename
      start_time = 0
      limit_duration = False
    else:
      soundfile = audio_synced = sample_rate = wav_tmp_filename = None
      source_filename = rec_filename
      start_time = seq.start_time
      limit_duration = True
    dest_filename = "%s/%s/%s_%s.ogg" % (dest_dirname, rec_name, seq.start_time, seq.end_time)
    os.makedirs(os.path.dirname(dest_filename), exist_ok=True)
    if args.no_ogg:
      print("no Ogg (%s -> %s)" % (os.path.basename(rec_filename), dest_filename[len(dest_dirname) + 1:]))
    else:
      if os.path.exists(dest_filename):
        print("already exists, delete: %s" % os.path.basename(dest_filename))
        os.remove(dest_filename)
      if args.no_conversion:
        assert source_filename.endswith(".ogg")
        print("skip ffmpeg, copy instead (%s -> %s)" % (
          os.path.basename(source_filename), dest_filename[len(dest_dirname) + 1:]))
        shutil.copy(src=source_filename, dst=dest_filename)
      else:
        cmd = ["ffmpeg"]
        if args.ffmpeg_acodec:
          cmd += ["-acodec", args.ffmpeg_acodec]  # https://trac.ffmpeg.org/ticket/2810
        cmd += ["-i", source_filename]
        if args.number_of_channels > 0:
          cmd += ["-ac", str(args.number_of_channels)]
        if start_time:
          cmd += ["-ss", str(start_time)]
        if limit_duration:
          cmd += ["-t", str(duration)]
        cmd += [dest_filename, "-loglevel", args.ffmpeg_loglevel]
        print("$ %s" % " ".join(cmd))
        check_call(cmd)
    if args.sprint_cache:
      audio_ogg, sample_rate_ogg = soundfile.read(dest_filename)
      assert len(audio_synced) == len(audio_ogg), "Number of frames in synced wav and converted ogg do not match"
      assert sample_rate == sample_rate_ogg, "Sample rates in synced wav and converted ogg do not match"
      os.remove(wav_tmp_filename)
    dest_meta_file.write("{'text': %r, 'speaker_name': %r, 'file': %r, 'seq_name': %r, 'duration': %s},\n" % (
      seq.orth, seq.speaker_name, dest_filename[len(dest_dirname) + 1:], seq.segment_name, duration))
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
  from returnn.util import better_exchook
  better_exchook.install()
  try:
    main()
  except BrokenPipeError:
    print("BrokenPipeError")
    sys.exit(1)
