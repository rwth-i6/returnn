
import os
import sys
my_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "%s/.." % my_dir)
sys.path.insert(0, "%s/../tools" % my_dir)  # for hdf_dump

from Dataset import Dataset
from HDFDataset import *
from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_raises
from nose.tools import raises
import Util
import h5py
import numpy as np
import os
import unittest
import better_exchook
better_exchook.install()
better_exchook.replace_traceback_format_tb()
Util.initThreadJoinHack()

from Log import log
log.initialize(verbosity=[5])


class TestHDFDataset(object):
  @classmethod
  def setup_class(cls):
    """
    :return:
     This method is run once before starting testing
    """

  @classmethod
  def teardown_class(cls):
    """
    This method is run once after completing all tests
    :return:
    """

  def setup(self):
    """
    This method is run before each test is going to be started
    :return:
    """

  def teardown(self):
    """
    This method is run after finishing of each test
    :return:
    """

  def test_init(self):
    """
    This method tests initialization of the HDFDataset class
    """
    toy_dataset = HDFDataset()
    assert_equal(toy_dataset.file_start, [0], "self.file_start init problem, should be [0]")
    assert_equal(toy_dataset.files, [], "self.files init problem, should be []")
    assert_equal(toy_dataset.file_seq_start, [], "self.file_seq_start init problem, should be []")
    return toy_dataset

  def test_addfile(self):
    """
    This method tests self.addfile function
    """
    toy_dataset = self.test_init()
    # TODO: auto-generate file, then use here
    #toy_dataset.add_file("/u/kulikov/develop/crnn/tests/toy_set.hdf")


def generate_dummy_hdf(num_datasets=1):
  for idx in range(1, num_datasets + 1):
    dataset = h5py.File('./dummy.%i.hdf5' % idx, 'w')
    dataset.create_group('streams')

    dataset['streams'].create_group('features')
    dataset['streams']['features'].attrs['parser'] = "feature_sequence"
    dataset['streams']['features'].create_group('data')

    dataset['streams'].create_group('classes')
    dataset['streams']['classes'].attrs['parser'] = "sparse"
    dataset['streams']['classes'].create_group('data')

    import string
    random = np.random.RandomState()
    feature_size = 13
    seq_names = ['dataset_%d_sequence_%d' % (idx, i) for i in range(100)]
    class_names = list(string.ascii_lowercase)
    num_classes = len(class_names)
    print(class_names, len(class_names))
    for idx in range(100):
      class_id = random.randint(low=0, high=num_classes)
      seq_len  = random.randint(low=1, high=20)
      features = random.rand(seq_len, feature_size)
      dataset['streams']['features']['data'].create_dataset(name=seq_names[idx], data=features, dtype='float32')
      dataset['streams']['classes']['data'].create_dataset(name=seq_names[idx], shape=(1,), data=np.int32(class_id), dtype='int32')

    dt = h5py.special_dtype(vlen=str)
    feature_names = dataset['streams']['classes'].create_dataset("feature_names", shape=(len(class_names),), dtype=dt)
    for id_x, orth in enumerate(class_names):
      feature_names[id_x] = orth

    dt = h5py.special_dtype(vlen=str)
    sequence_names_data = dataset.create_dataset("seq_names", shape=(len(seq_names),), dtype=dt)
    for ind, val in enumerate(seq_names):
      sequence_names_data[ind] = val

    dataset.close()
  return ['./dummy.%i.hdf5' % idx for idx in range(1, num_datasets + 1)]


def _get_tmp_file(suffix):
  """
  :param str suffix:
  :return: filename
  :rtype: str
  """
  import tempfile
  f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
  f.close()
  fn = f.name
  import atexit
  atexit.register(lambda: os.remove(fn))
  return fn


_hdf_cache = {}  # opts -> hdf fn


def generate_hdf_from_other(opts):
  """
  :param dict[str] opts:
  :return: hdf filename
  :rtype: str
  """
  # See test_hdf_dump.py and tools/hdf_dump.py.
  from Util import make_hashable
  cache_key = make_hashable(opts)
  if cache_key in _hdf_cache:
    return _hdf_cache[cache_key]
  fn = _get_tmp_file(suffix=".hdf")
  from Dataset import init_dataset
  dataset = init_dataset(opts)
  hdf_dataset = HDFDatasetWriter(fn)
  hdf_dataset.dump_from_dataset(dataset)
  hdf_dataset.close()
  _hdf_cache[cache_key] = fn
  return fn


def generate_hdf_from_dummy():
  """
  :return: hdf filename
  :rtype: str
  """
  return generate_hdf_from_other(
    {"class": "DummyDataset", "input_dim": 13, "output_dim": 7, "num_seqs": 23, "seq_len": 17})


def test_hdf_dump():
  generate_hdf_from_dummy()


class _DatasetCollectData:
  def __init__(self, dataset):
    """
    :param Dataset dataset:
    """
    self.dataset = dataset
    self.data_keys = None
    self.data_shape = {}  # key -> shape
    self.data_sparse = {}  # key -> bool
    self.data_dtype = {}  # key -> str
    self.data = {}  # key -> list
    self.seq_lens = []  # type: list[Util.NumbersDict]
    self.seq_tags = []
    self.num_seqs = None

  def read_all(self):
    dataset = self.dataset
    dataset.init_seq_order(epoch=1)
    data_keys = dataset.get_data_keys()
    self.data_keys = data_keys
    self.data_shape = {key: dataset.get_data_shape(key) for key in data_keys}
    self.data_sparse = {key: dataset.is_data_sparse(key) for key in data_keys}
    self.data_dtype = {key: dataset.get_data_dtype(key) for key in data_keys}
    seq_idx = 0
    while dataset.is_less_than_num_seqs(seq_idx):
      dataset.load_seqs(seq_idx, seq_idx + 1)
      for key in data_keys:
        data = dataset.get_data(seq_idx=seq_idx, key=key)
        self.data.setdefault(key, []).append(data)
      seq_len = dataset.get_seq_length(seq_idx=seq_idx)
      self.seq_lens.append(seq_len)
      seq_tag = dataset.get_tag(seq_idx)
      self.seq_tags.append(seq_tag)
      seq_idx += 1
    print("Iterated through %r, num seqs %i" % (dataset, seq_idx))
    self.num_seqs = seq_idx


def test_SimpleHDFWriter():
  fn = _get_tmp_file(suffix=".hdf")
  n_dim = 13
  writer = SimpleHDFWriter(filename=fn, dim=n_dim, labels=None)
  seq_lens1 = [11, 7, 5]
  writer.insert_batch(
    inputs=numpy.random.normal(size=(len(seq_lens1), max(seq_lens1), n_dim)).astype("float32"),
    seq_len=seq_lens1,
    seq_tag=["seq-%i" % i for i in range(len(seq_lens1))])
  seq_lens2 = [10, 13, 3, 2]
  writer.insert_batch(
    inputs=numpy.random.normal(size=(len(seq_lens2), max(seq_lens2), n_dim)).astype("float32"),
    seq_len=seq_lens2,
    seq_tag=["seq-%i" % (i + len(seq_lens1)) for i in range(len(seq_lens2))])
  writer.close()
  seq_lens = seq_lens1 + seq_lens2

  dataset = HDFDataset(files=[fn])
  reader = _DatasetCollectData(dataset=dataset)
  reader.read_all()
  assert "data" in reader.data_keys  # "classes" might be in there as well, although not really correct/existing
  assert reader.data_sparse["data"] is False
  assert list(reader.data_shape["data"]) == [n_dim]
  assert reader.data_dtype["data"] == "float32"
  assert len(seq_lens) == reader.num_seqs
  for i, seq_len in enumerate(seq_lens):
    assert reader.seq_lens[i]["data"] == seq_len


def dummy_iter_dataset(dataset):
  """
  :param Dataset dataset:
  """
  dataset.init_seq_order(epoch=1)
  data_keys = dataset.get_data_keys()
  seq_idx = 0
  while dataset.is_less_than_num_seqs(seq_idx):
    dataset.load_seqs(seq_idx, seq_idx + 1)
    for key in data_keys:
      dataset.get_data(seq_idx=seq_idx, key=key)
    seq_idx += 1
  print("Iterated through %r, num seqs %i" % (dataset, seq_idx))


def test_hdf_simple_iter():
  hdf_fn = generate_hdf_from_dummy()
  dataset = HDFDataset(files=[hdf_fn])
  dataset.initialize()
  dummy_iter_dataset(dataset)


def test_rnn_getCacheByteSizes_zero():
  from Config import Config
  config = Config({"cache_size": "0"})
  import rnn
  rnn.config = config
  sizes = rnn.getCacheByteSizes()
  assert len(sizes) == 3
  assert all([s == 0 for s in sizes])


def test_rnn_initData():
  hdf_fn = generate_hdf_from_dummy()
  from Config import Config
  config = Config({"cache_size": "0", "train": hdf_fn, "dev": hdf_fn})
  import rnn
  rnn.config = config
  rnn.initData()
  train, dev = rnn.train_data, rnn.dev_data
  assert train and dev
  assert isinstance(train, HDFDataset)
  assert isinstance(dev, HDFDataset)
  assert train.cache_byte_size_total_limit == dev.cache_byte_size_total_limit == 0
  assert train.cache_byte_size_limit_at_start == dev.cache_byte_size_limit_at_start == 0


def test_hdf_no_cache_iter():
  hdf_fn = generate_hdf_from_dummy()
  dataset = HDFDataset(files=[hdf_fn])
  dataset.initialize()
  assert dataset.cache_byte_size_limit_at_start == 0
  assert dataset.cache_byte_size_total_limit == 0

  class DummyCallback:
    def __init__(self):
      self.was_called = False

    def __call__(self, *args, **kwargs):
      print("DummyCallback called!")
      self.was_called = True

  assert hasattr(dataset, "_preload_seqs")
  dataset._preload_seqs = DummyCallback()

  dummy_iter_dataset(dataset)
  import time
  time.sleep(1)  # maybe threads are running in background...
  assert not dataset._preload_seqs.was_called


def test_siamese_triplet_sampling():
  datasets_path = generate_dummy_hdf(3)
  dataset = SiameseHDFDataset(input_stream_name="features", seq_label_stream="classes", files=datasets_path)

  dataset.initialize()
  for iter in range(1, 31):
    print("Initializing triplets... iteration %d" % iter)
    dataset.init_seq_order(epoch=iter)

    triplets = dataset.curr_epoch_triplets
    anchor_seq_names = [dataset.all_seq_names[id[0]] for id in triplets]
    same_class_seq_names = [dataset.all_seq_names[id[1]] for id in triplets]
    diff_class_seq_names = [dataset.all_seq_names[id[2]] for id in triplets]

    anchor_class = [dataset.seq_to_target[seq_id] for seq_id in anchor_seq_names]
    same_class = [dataset.seq_to_target[seq_id] for seq_id in same_class_seq_names]
    diff_class = [dataset.seq_to_target[seq_id] for seq_id in diff_class_seq_names]

    print("Testing pair sequences to belong to the same class...")
    assert (all(ac == same_class[id] for id, ac in enumerate(anchor_class)))
    print("Testing third element in a triplet to belong to a different class...")
    assert (all(ac != diff_class[id] for id, ac in enumerate(anchor_class)))
    print("------------------------------------------------------")

  print("Deleting temporary files...")
  for path in datasets_path:
    os.remove(path)
  print("Done.")


def test_siamese_collect_single_seq():
  datasets_path = generate_dummy_hdf(3)
  dataset = SiameseHDFDataset(input_stream_name="features", seq_label_stream="classes", files=datasets_path)

  dataset.initialize()
  dataset.init_seq_order(epoch=1)

  random = np.random.RandomState()
  seq_idx = random.randint(low=0, high=len(dataset.seq_name_to_idx))
  dataset_seq = dataset._collect_single_seq(seq_idx)
  print("Verify that single sequence consists of a triplet...")
  print(dataset_seq.features.keys())

  print("Deleting temporary files...")
  for path in datasets_path:
    os.remove(path)
  print("Done.")


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
