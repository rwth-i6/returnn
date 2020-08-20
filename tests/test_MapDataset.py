
import better_exchook
import numpy
import os
import sys
import unittest

from nose.tools import assert_equal, assert_is_instance, assert_in, assert_not_in, assert_true, assert_false

from returnn.datasets.map import MapDataset
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


class CustomDataset(MapDataset):

  def __init__(self, **kwargs):
    super(CustomDataset, self).__init__(**kwargs)
    self.data = []
    self.data.append(numpy.asarray([[1, 1], ]))
    self.data.append(numpy.asarray([[3, 1], [3, 2], [3, 3]]))
    self.data.append(numpy.asarray([[2, 2], [2, 2]]))

  def __len__(self):
    return 3

  def __getitem__(self, seq_idx):
    return {'data': self.data[seq_idx]}


class CustomDatasetSeqLen(CustomDataset):

  def get_seq_len(self, seq_idx):
    return len(self.data[seq_idx])


class CustomDatasetDualSeqLen(CustomDataset):

  def __init__(self, **kwargs):
    super(CustomDataset, self).__init__(**kwargs)
    self.data = []
    self.data.append(numpy.asarray([[1, 1, 1], ]))
    self.data.append(numpy.asarray([[1, 3, 1], [1, 3, 2], [1, 3, 3]]))
    self.data.append(numpy.asarray([[1, 2, 2], [1, 2, 2]]))

    self.data2 = []
    self.data2.append(numpy.asarray([[2, 1, 1], ]))
    self.data2.append(numpy.asarray([[2, 3, 1], [2, 3, 2], [2, 3, 3]]))
    self.data2.append(numpy.asarray([[2, 2, 2], [2, 2, 2]]))

    #self.num_outputs = {'data': {'shape': (None, 3), 'dim': 3},
    #                    'data2': {'shape': (None, 3), 'dim': 3}}

  def get_seq_len(self, seq_idx):
    return len(self.data[seq_idx])

  def __getitem__(self, seq_idx):
    print("oida: %i" % seq_idx)
    return {'data': self.data[seq_idx],
            'data2': self.data2[seq_idx]}


def test_init():

  data = CustomDataset()
  data.init_seq_order(epoch=1)
  data.load_seqs(0, 1)
  raw_ = data.get_data(seq_idx=0, key="data")
  assert_equal(raw_[0, 0], 1)


def test_sorting():
  data = CustomDatasetSeqLen(seq_ordering="sorted")
  data.init_seq_order(epoch=1)
  data.load_seqs(0, 3)
  raw_ = data.get_data(seq_idx=2, key="data")
  print(raw_)
  assert_equal(raw_[0, 0], 3)


def test_forward():
  import tempfile
  output_file = tempfile.mktemp(suffix=".hdf", prefix="nose-tf-forward")
  dataset = CustomDatasetDualSeqLen()
  dataset.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "network": {"output": {"class": "combine", "kind": "add", "from": ["data:data", "data:data2"]}},
    "output_file": output_file,
  })
  _cleanup_old_models(config)

  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=dataset, dev_data=None, eval_data=None, )

  engine.forward_to_hdf(data=dataset, output_file=output_file, batch_size=2)

  engine.finalize()

  assert os.path.exists(output_file)
  #import h5py
  #with h5py.File(output_file, 'r') as f:
  #  assert f['inputs'].shape == (seq_len * num_seqs, n_classes_dim)
  #  assert f['seqLengths'].shape == (num_seqs, 2)
  #  assert f['seqTags'].shape == (num_seqs,)
  #  assert f.attrs['inputPattSize'] == n_classes_dim
  #  assert f.attrs['numSeqs'] == num_seqs
  #  assert f.attrs['numTimesteps'] == seq_len * num_seqs

  #from returnn.datasets.hdf import HDFDataset
  #ds = HDFDataset()
  #ds.add_file(output_file)

  #assert_equal(ds.num_inputs, n_classes_dim)  # forwarded input is network output
  #assert_equal(ds.get_num_timesteps(), seq_len * num_seqs)
  #assert_equal(ds.num_seqs, num_seqs)

  os.remove(output_file)


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
