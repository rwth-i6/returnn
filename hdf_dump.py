#!/usr/bin/env python

import h5py as h5
import numpy
from Log import log
import rnn
import argparse
import sys
import HDFDataset
from Dataset import Dataset, init_dataset_via_str
from Config import Config
from Util import NumbersDict, human_size, progress_bar_with_time, try_run


def hdf_dataset_init(file_name):
  """
  :param str file_name: filename of hdf dataset file in the filesystem
  :rtype: h5py._hl.files.File
  """
  print >> log.v3, "Creating HDF dataset file %s" % file_name
  return h5.File(file_name, "w")


def hdf_dump_from_dataset(dataset, hdf_dataset, parser_args):
  """
  :param Dataset dataset: could be any dataset implemented as child of Dataset
  :type hdf_dataset: h5py._hl.files.File
  :param parser_args: argparse object from main()
  :return:
  """
  print >> log.v3, "Work on epoch: %i" % parser_args.epoch
  dataset.init_seq_order(parser_args.epoch)

  data_keys = sorted(dataset.get_data_keys())
  print >> log.v3, "Data keys:", data_keys
  if "orth" in data_keys:
    data_keys.remove("orth")

  # We need to do one run through the dataset to collect some stats like total len.
  print >> log.v3, "Collect stats, iterate through all data..."
  seq_idx = parser_args.start_seq
  seq_idxs = []
  seq_tags = []
  seq_lens = []
  total_seq_len = NumbersDict(0)
  max_tag_len = 0
  dataset_num_seqs = try_run(lambda: dataset.num_seqs, default=None)  # can be unknown
  if parser_args.end_seq != float("inf"):
    if dataset_num_seqs is not None:
      dataset_num_seqs = min(dataset_num_seqs, parser_args.end_seq)
    else:
      dataset_num_seqs = parser_args.end_seq
  if dataset_num_seqs is not None:
    dataset_num_seqs -= parser_args.start_seq
    assert dataset_num_seqs > 0
  while dataset.is_less_than_num_seqs(seq_idx) and seq_idx <= parser_args.end_seq:
    seq_idxs += [seq_idx]
    dataset.load_seqs(seq_idx, seq_idx + 1)
    seq_len = dataset.get_seq_length(seq_idx)
    seq_lens += [seq_len]
    tag = dataset.get_tag(seq_idx)
    seq_tags += [tag]
    max_tag_len = max(len(tag), max_tag_len)
    total_seq_len += seq_len
    if dataset_num_seqs is not None:
      progress_bar_with_time(float(seq_idx - parser_args.start_seq) / dataset_num_seqs)
    seq_idx += 1
  num_seqs = len(seq_idxs)

  assert num_seqs > 0
  shapes = {}
  for data_key in data_keys:
    assert data_key in total_seq_len.dict
    shape = [total_seq_len[data_key]]
    shape += dataset.get_data_shape(data_key)
    print >> log.v3, "Total len of %r is %s, shape %r, dtype %s" % (
                     data_key, human_size(shape[0]), shape, dataset.get_data_dtype(data_key))
    shapes[data_key] = shape

  print >> log.v3, "Set seq tags..."
  hdf_dataset.create_dataset('seqTags', shape=(num_seqs,), dtype="S%i" % (max_tag_len + 1))
  for i, tag in enumerate(seq_tags):
    hdf_dataset['seqTags'][i] = tag
    progress_bar_with_time(float(i) / num_seqs)

  print >> log.v3, "Set seq len info..."
  hdf_dataset.create_dataset(HDFDataset.attr_seqLengths, shape=(num_seqs, 2), dtype="int32")
  for i, seq_len in enumerate(seq_lens):
    data_len = seq_len["data"]
    targets_len = seq_len["classes"]
    for data_key in dataset.get_target_list():
      if data_key == "orth":
        continue
      assert seq_len[data_key] == targets_len, "different lengths in multi-target not supported"
    if targets_len is None:
      targets_len = data_len
    hdf_dataset[HDFDataset.attr_seqLengths][i] = [data_len, targets_len]
    progress_bar_with_time(float(i) / num_seqs)

  print >> log.v3, "Create arrays in HDF..."
  hdf_dataset.create_group('targets/data')
  hdf_dataset.create_group('targets/size')
  hdf_dataset.create_group('targets/labels')
  for data_key in data_keys:
    if data_key == "data":
      hdf_dataset.create_dataset(
        'inputs', shape=shapes[data_key], dtype=dataset.get_data_dtype(data_key))
    else:
      hdf_dataset['targets/data'].create_dataset(
        data_key, shape=shapes[data_key], dtype=dataset.get_data_dtype(data_key))
      hdf_dataset['targets/size'].attrs[data_key] = dataset.num_outputs[data_key]

    if data_key in dataset.labels:
      labels = dataset.labels[data_key]
      assert len(labels) == dataset.num_outputs[data_key][0]
    else:
      labels = ["%s-class-%i" % (data_key, i) for i in range(dataset.get_data_dim(data_key))]
    print >> log.v5, "Labels for %s:" % data_key, labels[:3], "..."
    max_label_len = max(map(len, labels))
    hdf_dataset['targets/labels'].create_dataset(data_key, (len(labels),), dtype="S%i" % (max_label_len + 1))
    for i, label in enumerate(labels):
      hdf_dataset['targets/labels'][data_key][i] = label

  # Again iterate through dataset, and set the data
  print >> log.v3, "Write data..."
  dataset.init_seq_order(parser_args.epoch)
  offsets = NumbersDict(0)
  for seq_idx, tag in zip(seq_idxs, seq_tags):
    dataset.load_seqs(seq_idx, seq_idx + 1)
    tag_ = dataset.get_tag(seq_idx)
    assert tag == tag_  # Just a check for sanity. We expect the same order.
    seq_len = dataset.get_seq_length(seq_idx)
    for data_key in data_keys:
      if data_key == "data":
        hdf_data = hdf_dataset['inputs']
      else:
        hdf_data = hdf_dataset['targets/data'][data_key]
      data = dataset.get_data(seq_idx, data_key)
      hdf_data[offsets[data_key]:offsets[data_key] + seq_len[data_key]] = data

    progress_bar_with_time(float(offsets["data"]) / total_seq_len["data"])

    offsets += seq_len

  assert offsets == total_seq_len  # Sanity check.

  # Set some old-format attribs. Not needed for newer CRNN versions.
  hdf_dataset.attrs[HDFDataset.attr_inputPattSize] = dataset.num_inputs
  hdf_dataset.attrs[HDFDataset.attr_numLabels] = dataset.num_outputs.get("classes", (0, 0))[0]

  print >> log.v3, "All done."


def hdf_close(hdf_dataset):
  """
  :param h5py._hl.files.File hdf_dataset: to close
  """
  hdf_dataset.close()


def init(config_filename, cmd_line_opts, dataset_config_str):
  """
  :param str config_filename: global config for CRNN
  :param list[str] cmd_line_opts: options for initConfig method
  :param str dataset_config_str: dataset via init_dataset_via_str()
  """
  rnn.initBetterExchook()
  rnn.initThreadJoinHack()
  if config_filename:
    rnn.initConfig(config_filename, cmd_line_opts)
    rnn.initLog()
  else:
    log.initialize(verbosity=[5])
  print >> log.v3, "CRNN dump-dataset starting up."
  rnn.initFaulthandler()
  rnn.initConfigJsonNetwork()
  if config_filename:
    rnn.initData()
    rnn.printTaskProperties()
    assert isinstance(rnn.train_data, Dataset)
    return rnn.train_data
  else:
    assert dataset_config_str
    dataset = init_dataset_via_str(dataset_config_str)
    print >> log.v3, "Source dataset:", dataset.len_info()
    return dataset


def _is_crnn_config(filename):
  if filename.endswith(".gz"):
    return False
  try:
    config = Config()
    config.load_file(filename)
    return True
  except Exception:
    pass
  return False


def main(argv):
  parser = argparse.ArgumentParser(description="Dump dataset or subset of dataset in external HDF dataset")
  parser.add_argument('config_file_or_dataset', type=str,
                      help="Config file for CRNN, or directly the dataset init string")
  parser.add_argument('hdf_filename', type=str, help="File name of the HDF dataset, which will be created")
  parser.add_argument('--start_seq', type=int, default=0, help="Start sequence index of the dataset to dump")
  parser.add_argument('--end_seq', type=int, default=float("inf"), help="End sequence index of the dataset to dump")
  parser.add_argument('--epoch', type=int, default=1, help="Optional start epoch for initialization")

  args = parser.parse_args(argv[1:])
  crnn_config = None
  dataset_config_str = None
  if _is_crnn_config(args.config_file_or_dataset):
    crnn_config = args.config_file_or_dataset
  else:
    dataset_config_str = args.config_file_or_dataset
  dataset = init(config_filename=crnn_config, cmd_line_opts=[], dataset_config_str=dataset_config_str)
  hdf_dataset = hdf_dataset_init(args.hdf_filename)
  hdf_dump_from_dataset(dataset, hdf_dataset, args)
  hdf_close(hdf_dataset)

  rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)


