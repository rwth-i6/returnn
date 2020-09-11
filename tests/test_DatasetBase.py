"""
Test cases for the new dataset API
"""
import better_exchook
import numpy
import os
import sys
import unittest

from nose.tools import assert_equal, assert_is_instance, assert_in, assert_not_in, assert_true, assert_false

from returnn.datasets.map import MapDatasetBase
from returnn.datasets.iterator import IteratorDatasetBase
from returnn.datasets.basic import DatasetSeq
from returnn.config import Config
from returnn.tf.engine import Engine


def _get_tmp_dir():
  """
  :return: dirname
  :rtype: str
  """
  import tempfile
  import shutil
  import atexit
  name = tempfile.mkdtemp()
  assert name and os.path.isdir(name) and not os.listdir(name)
  atexit.register(lambda: shutil.rmtree(name))
  return name


def _cleanup_old_models(config):
  """
  :param Config config:
  """
  model_prefix = config.value("model", None)
  from glob import glob
  files = glob("%s.*" % model_prefix)
  if files:
    print("Delete old models:", files)
    for fn in files:
      os.remove(fn)


class CustomMapDataset(MapDatasetBase):
  """
  A custom dataset providing only __len__ and a single fixed data stream
  """

  def __init__(self, **kwargs):
    super(CustomMapDataset, self).__init__(**kwargs)
    self.data = []
    self.data.append(numpy.asarray([[1, 1], ]))
    self.data.append(numpy.asarray([[3, 1], [3, 2], [3, 3]]))
    self.data.append(numpy.asarray([[2, 2], [2, 2]]))

  def __len__(self):
    return 3

  def __getitem__(self, seq_idx):
    return {'data': self.data[seq_idx]}


class CustomMapDatasetSeqLen(CustomMapDataset):
  """
  Extends the CustomMapDataset with a sequence length
  """

  def get_seq_len(self, seq_idx):
    """

    :param seq_idx:
    :return:
    """
    return len(self.data[seq_idx])


class CustomMapDatasetDualSeqLen(CustomMapDatasetSeqLen):
  """
  Extens the CustomMapDatasetSeqLen with a second data stream
  """

  def __init__(self, **kwargs):
    super(CustomMapDataset, self).__init__(**kwargs)
    self.data = []
    self.data.append(numpy.asarray([[1, 1, 1], ]))
    self.data.append(numpy.asarray([[1, 3, 1], [1, 3, 2], [1, 3, 3]]))
    self.data.append(numpy.asarray([[1, 2, 2], [1, 2, 2]]))

    self.data2 = []
    self.data2.append(numpy.asarray([[2, 1, 1], ]))
    self.data2.append(numpy.asarray([[2, 3, 1], [2, 3, 2], [2, 3, 3]]))
    self.data2.append(numpy.asarray([[2, 2, 2], [2, 2, 2]]))

    self.num_outputs = {'track1': {'shape': (None, 3), 'dim': 3},
                        'track2': {'shape': (None, 3), 'dim': 3}}

  def __getitem__(self, seq_idx):
    return {'track1': self.data[seq_idx],
            'track2': self.data2[seq_idx]}


class CustomIteratorDataset(IteratorDatasetBase):
  """
  Implements an iterative dataset with a character dictionary
  """

  def __init__(self, text_file, **kwargs):
    super(CustomIteratorDataset, self).__init__(**kwargs)
    self.text_file = open(text_file, "rt")

    self.dict = {'a': 1,
                 'b': 2,
                 'c': 3,
                 'd': 4}

    self.num_outputs = {'data': {'shape': (None,), 'sparse': True, 'dim': 5}}
    self.current_seq_idx = -1

  def __next__(self):
    line = next(self.text_file)  # will automatically raise StopIteration
    indices = numpy.asarray([self.dict[c] for c in line.strip()] + [0], dtype="int32")

    self.current_seq_idx += 1
    return DatasetSeq(self.current_seq_idx, indices, seq_tag="seq_%i" % self.current_seq_idx)

  def seek_epoch(self, epoch=None):
    """

    :param epoch:
    :return:
    """
    self.text_file.seek(0)
    self.current_seq_idx = -1


def test_map_dataset_init():
  """
  Test initializing a map-style dataset
  """
  data = CustomMapDataset()
  data.init_seq_order(epoch=1)
  data.load_seqs(0, 1)
  raw_ = data.get_data(seq_idx=0, key="data")
  assert_equal(raw_[0, 0], 1)


def test_map_dataset_sorting():
  """
  Test the sequence sorting of a map-style dataset
  """
  data = CustomMapDatasetSeqLen(seq_ordering="sorted")
  data.init_seq_order(epoch=1)
  data.load_seqs(0, 3)
  raw_ = data.get_data(seq_idx=0, key="data")
  assert_equal(raw_[0, 0], 1)
  raw_ = data.get_data(seq_idx=1, key="data")
  assert_equal(raw_[0, 0], 2)
  raw_ = data.get_data(seq_idx=2, key="data")
  assert_equal(raw_[0, 0], 3)


def test_map_dataset_forward():
  """
  Test running a forward task with a map-style dataset
  """
  import tempfile
  output_file = tempfile.mktemp(suffix=".hdf", prefix="nose-tf-forward")
  dataset = CustomMapDatasetDualSeqLen()
  dataset.init_seq_order(epoch=1)

  from returnn.tf.util.basic import DimensionTag
  enc_time = DimensionTag(kind=DimensionTag.Types.Spatial, description="track-time")

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "extern_data": {'track1': {'shape': (None, 3), 'same_dim_tags_as': {'T': enc_time}},
                    'track2': {'shape': (None, 3), 'same_dim_tags_as': {'T': enc_time}}},
    "network": {"output": {"class": "combine", "kind": "add", "from": ["data:track1", "data:track2"]}},
    "output_file": output_file,
  })
  _cleanup_old_models(config)

  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=dataset, dev_data=None, eval_data=None, )

  engine.forward_to_hdf(data=dataset, output_file=output_file, batch_size=2)

  engine.finalize()

  assert os.path.exists(output_file)
  import h5py
  with h5py.File(output_file, 'r') as f:
    assert f['inputs'].shape == (1 + 2 + 3, 3)
    assert f['seqLengths'].shape == (3, 2)
    assert_true(numpy.array_equal(f['inputs'][:f['seqLengths'][0][0]], [[3, 2, 2]]))
    assert_true(numpy.array_equal(f['inputs'][:f['seqLengths'][0][0]], [[3, 2, 2]]))
    assert f['seqTags'].shape == (3,)

  os.remove(output_file)


def make_tmp_textfile():
  """
  Create a fake text file
  """
  import tempfile
  text_file_path = tempfile.mktemp(suffix=".txt", prefix="nose-text")
  text_file = open(text_file_path, "wt")
  text_file.write("aabcd\nbabcd\ncabcd")
  text_file.close()
  return text_file_path


def test_iterator_dataset_init():
  """
  Test initializing an iterator-style dataset
  """
  text_file_path = make_tmp_textfile()
  data = CustomIteratorDataset(text_file=text_file_path)
  data.init_seq_order(epoch=1)
  data.load_seqs(0, 1)
  raw_ = data.get_data(seq_idx=0, key="data")
  assert_true(numpy.array_equal(raw_, [1, 1, 2, 3, 4, 0]))


def test_iterator_dataset_forward():
  """
  Test running a forward task with an iterator-style dataset
  """
  import tempfile
  text_file_path = make_tmp_textfile()
  dataset = CustomIteratorDataset(text_file=text_file_path)
  dataset.get_current_seq_order()
  dataset.init_seq_order(epoch=1)

  output_file = tempfile.mktemp(suffix=".hdf", prefix="nose-tf-forward")

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "network": {"output": {"class": "copy", "from": ["data:data"]}},
    "output_file": output_file,
    "batch_size": 100,
  })
  _cleanup_old_models(config)

  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=dataset, dev_data=None, eval_data=None, )

  engine.forward_to_hdf(data=dataset, output_file=output_file, batch_size=1)

  engine.finalize()

  import h5py
  with h5py.File(output_file, 'r') as f:
    assert f['inputs'].shape == (18,)
    assert_equal(f['seqLengths'].shape, (3, 2))
    assert_true(numpy.array_equal(f['inputs'][:f['seqLengths'][0][0]], [1, 1, 2, 3, 4, 0]))
    assert f['seqTags'].shape == (3,)


if __name__ == "__main__":
  better_exchook.install()
  if len(sys.argv) <= 1:
    for k, v in sorted(globals().items()):
      if k.startswith("test_"):
        print("-" * 40)
        print("Executing: %s" % k)
        try:
          v()
        except unittest.SkipTest as exc:
          print("SkipTest:", exc)
        print("-" * 40)
    print("Finished all tests.")
  else:
    assert len(sys.argv) >= 2
    for arg in sys.argv[1:]:
      print("Executing: %s" % arg)
      if arg in globals():
        globals()[arg]()  # assume function and execute
      else:
        eval(arg)  # assume Python code and execute
