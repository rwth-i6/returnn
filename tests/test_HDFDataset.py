
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
Util.init_thread_join_hack()

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


class _DatasetReader:
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
  reader = _DatasetReader(dataset=dataset)
  reader.read_all()
  assert "data" in reader.data_keys  # "classes" might be in there as well, although not really correct/existing
  assert reader.data_sparse["data"] is False
  assert list(reader.data_shape["data"]) == [n_dim]
  assert reader.data_dtype["data"] == "float32"
  assert len(seq_lens) == reader.num_seqs
  for i, seq_len in enumerate(seq_lens):
    assert reader.seq_lens[i]["data"] == seq_len


def test_SimpleHDFWriter_small():
  fn = _get_tmp_file(suffix=".hdf")
  n_dim = 3
  writer = SimpleHDFWriter(filename=fn, dim=n_dim, labels=None)
  seq_lens = [2, 3]
  writer.insert_batch(
    inputs=numpy.random.normal(size=(len(seq_lens), max(seq_lens), n_dim)).astype("float32"),
    seq_len=seq_lens,
    seq_tag=["seq-%i" % i for i in range(len(seq_lens))])
  writer.close()

  dataset = HDFDataset(files=[fn])
  reader = _DatasetReader(dataset=dataset)
  reader.read_all()
  assert "data" in reader.data_keys  # "classes" might be in there as well, although not really correct/existing
  assert reader.data_sparse["data"] is False
  assert list(reader.data_shape["data"]) == [n_dim]
  assert reader.data_dtype["data"] == "float32"
  assert len(seq_lens) == reader.num_seqs
  for i, seq_len in enumerate(seq_lens):
    assert reader.seq_lens[i]["data"] == seq_len

  if sys.version_info[0] >= 3:  # gzip.compress is >=PY3
    print("raw content (gzipped):")
    import gzip
    print(repr(gzip.compress(open(fn, "rb").read())))


def test_read_simple_hdf():
  if sys.version_info[0] <= 2:  # gzip.decompress is >=PY3
    raise unittest.SkipTest
  # n_dim, seq_lens, raw_gzipped is via test_SimpleHDFWriter_small
  n_dim = 3
  seq_lens = [2, 3]
  raw_gzipped = (
    b'\x1f\x8b\x08\x00\x80\xc8f\\\x02\xff\xed\x9a=l\xd3@\x14\xc7\xefl\'XQ\x8a\xd2vhTT\xf0\x84X\x90\xc2V1\xb8TJ\xa0C'
    b'\x81\x8a\x04\xa9\x03BM\x85i"5Q\x82]$`h%\x18@\xea\xc6\x02\x1b\x03\x03\x1f\x0b\x0b\x1bm\x10S\xa7NLL\xb01\x96\x1d'
    b'\xa9\xd8\xbewI}\x8d\x9d\xb1i\xf9\xff\x86\\\xee\xf2\x9e\xef\xc3\xff\xbb{'
    b'\xf6\xe5\xc5\\\xf1\xeaHf2\xc3\x02L\x93\x19,'
    b'\xc7\x0e\xb2O|\xb7\xa3y\xf9\xfb\x12\xa5\x9c\xd2\xe7\x94\xbe\xd3d\xb9\x19\xfe\x96\xa7\xf2\x1c]\xdf\xd2E\xbeF\x8e'
    b'\x95[\xa5R`\xbd\xaf \xeby\x95\x12\xe9\x05\x06\xfeG\xe6J\xb3\x0bA\xbaH\xf9\x02\xa5;Z\xd4n\xb5\xba\xec\xac\xba'
    b'\x8c\xb9N{\xdei\xaex5W\x94\xd7\x9b\xad5O\x94W\xaa+nW\xaf\x83\xf44JzUu\x9de\xd3\xfe\\\t\x14{'
    b'\xda\xffn\x8a\xeb/T=\xaf\\\x7f\xec\x04:7\xfd\xe9\x14Z^\x89\xcc\x0f\x92\xbd\xefS '
    b'\x7f3\xf4o\xae5\x8a\xf5\x86\x1b\xeb\xc7\x99ZoF\xfa\xcdS\x97\xc5\xfc\x1aX\xaf\xf4\x1f\x91\xfe\x95z\xc3q=\xa7\xe5'
    b'&\xf9\xa7\xe2\xdb]v\xda\xf1\xed\xee\xdd\x9e|\xe28s\x96\x16>\\\xe6\x85=\xe7\xbc\xaf\xbdN\xeb\xca8\x17mKQ^\xd3'
    b'\xb4\xd0\xc1$\x7f\x9d\xab+\x89`\x8cZ\x1b\x18o\xec\xdc\xbf\xd3\xbb\xc3\xc3A\xf9\xc6\xcd"\xf7G\xda\x92:\x1dK\xb67'
    b'\xe5\xfak$\xdb\xc9\xd5\xfdg:\xd9N\xce\x8b\xd6\xd4\xf1^7\x0e\xebJ\xf4\x8c\x0b\x99t\xf5\xa9)\xfb\x9d\xd6\xd5Y.t\r'
    b'\xf4-\x86\xd6\xa2\xf9@z\xd3YTo\x9a\xbew\x8a|\rqY\xa3\xbf\xdeZG2\x1e\xc1>\xcb\xfb\xed\xb3V\xb2\xdf\xe6Y\xa5\xc0'
    b'\x88\x8e\x9b\x81-\n\x00\x00\x00\x00\xe0D1('
    b'\x8eN)\xcf\x99j|\xa9\xfb\xf1q`9j\x9d\xeb\xc6\xd1\x13&\x9bX\xef>_\xc6\xc6\xd3\xd3\xe3\xbdPS\x8f\x8f\xa7kG\x10G'
    b'\xeb\x87\xfa\x99\x1f\xe0\xb7iG\x9f\x86\xb5\x18\xbb\xb7\x8a]\\|\xfd\xc5\x8e\xe6\xd3\xca}@\\\x0e\x00\x00\x00'
    b'\x000lqt\xf4\x9cC}\x1f}\xf0\x9c#\xcd\x92\xce96\xe8\rm6R\xdf\xb0\x9fs\xb8N\xfbb!\xfc\xbc\x14\xf6Y\x06\xf9:\xa4'
    b'\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1cK>\xde>\xdf\xb9\xf6\xc3\xec\xfc'
    b'}\xf0\xc7\xbe7\xf5m\xe6\xf2\xe7\xa7\xf6^\xee\xc9\xd6\xefG\xb3\x9d\x87\xcf\x9c\xed_\x1f&\xbf\xbe\xd4\xcet\xae'
    b'\xaf\xbf\x99\xc9~\xda\xdd\xaa/\xbf\xb7\xef.\xeen\xbf^2"\xff\x82\xfd\x07\xf7a\x8c\xa2\xd4>\x00\x00')
  import gzip
  fn = _get_tmp_file(suffix=".hdf")
  with open(fn, "wb") as f:
    f.write(gzip.decompress(raw_gzipped))

  dataset = HDFDataset(files=[fn])
  reader = _DatasetReader(dataset=dataset)
  reader.read_all()
  assert "data" in reader.data_keys  # "classes" might be in there as well, although not really correct/existing
  assert reader.data_sparse["data"] is False
  assert list(reader.data_shape["data"]) == [n_dim]
  assert reader.data_dtype["data"] == "float32"
  assert len(seq_lens) == reader.num_seqs
  for i, seq_len in enumerate(seq_lens):
    assert reader.seq_lens[i]["data"] == seq_len


def test_SimpleHDFWriter_ndim1_var_len():
  # E.g. attention weights, shape (dec-time,enc-time) per seq.
  fn = _get_tmp_file(suffix=".hdf")
  writer = SimpleHDFWriter(filename=fn, dim=None, ndim=2, labels=None)
  dec_seq_lens1 = [11, 7, 5]
  enc_seq_lens1 = [13, 6, 8]
  batch1_data = numpy.random.normal(size=(len(dec_seq_lens1), max(dec_seq_lens1), max(enc_seq_lens1))).astype("float32")
  writer.insert_batch(
    inputs=batch1_data,
    seq_len={0: dec_seq_lens1, 1: enc_seq_lens1},
    seq_tag=["seq-%i" % i for i in range(len(dec_seq_lens1))])
  dec_seq_lens2 = [10, 13, 3, 2]
  enc_seq_lens2 = [11, 13, 5, 4]
  batch2_data = numpy.random.normal(size=(len(dec_seq_lens2), max(dec_seq_lens2), max(enc_seq_lens2))).astype("float32")
  writer.insert_batch(
    inputs=batch2_data,
    seq_len={0: dec_seq_lens2, 1: enc_seq_lens2},
    seq_tag=["seq-%i" % (i + len(dec_seq_lens1)) for i in range(len(dec_seq_lens2))])
  writer.close()
  dec_seq_lens = dec_seq_lens1 + dec_seq_lens2
  enc_seq_lens = enc_seq_lens1 + enc_seq_lens2

  dataset = HDFDataset(files=[fn])
  reader = _DatasetReader(dataset=dataset)
  reader.read_all()
  assert "data" in reader.data_keys  # "classes" might be in there as well, although not really correct/existing
  assert reader.data_sparse["data"] is False
  assert list(reader.data_shape["data"]) == []
  assert reader.data_dtype["data"] == "float32"
  assert "sizes" in reader.data_keys
  # sizes sparsity does not really matter...
  assert reader.data_dtype["sizes"] == "int32"
  assert list(reader.data_shape["sizes"]) == []
  assert len(dec_seq_lens) == len(enc_seq_lens) == reader.num_seqs
  for i, (dec_seq_len, enc_seq_len) in enumerate(zip(dec_seq_lens, enc_seq_lens)):
    assert reader.seq_lens[i]["data"] == dec_seq_len * enc_seq_len
    assert reader.seq_lens[i]["sizes"] == 2
    assert reader.data["sizes"][i].tolist() == [dec_seq_len, enc_seq_len], "got %r" % (
      reader.data["sizes"][i],)


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
      dataset.get_tag(seq_idx)
    seq_idx += 1
  print("Iterated through %r, num seqs %i" % (dataset, seq_idx))


def test_hdf_simple_iter():
  hdf_fn = generate_hdf_from_dummy()
  dataset = HDFDataset(files=[hdf_fn])
  dataset.initialize()
  dummy_iter_dataset(dataset)


def test_hdf_simple_iter_cached():
  hdf_fn = generate_hdf_from_dummy()
  dataset = HDFDataset(files=[hdf_fn], cache_byte_size=100)
  dataset.initialize()
  dummy_iter_dataset(dataset)


def test_rnn_getCacheByteSizes_zero():
  from Config import Config
  config = Config({"cache_size": "0"})
  import rnn
  rnn.config = config
  sizes = rnn.get_cache_byte_sizes()
  assert len(sizes) == 3
  assert all([s == 0 for s in sizes])


def test_rnn_initData():
  hdf_fn = generate_hdf_from_dummy()
  from Config import Config
  config = Config({"cache_size": "0", "train": hdf_fn, "dev": hdf_fn})
  import rnn
  rnn.config = config
  rnn.init_data()
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


def test_hdf_data_short_int_dtype():
  from GeneratingDataset import StaticDataset
  dataset = StaticDataset([
    {"data": numpy.array([1, 2, 3], dtype="uint8"), "classes": numpy.array([-1, 5], dtype="int16")}],
    output_dim={"data": (255, 1), "classes": (10, 1)})
  orig_data_dtype = dataset.get_data_dtype("data")
  orig_classes_dtype = dataset.get_data_dtype("classes")
  assert orig_data_dtype == "uint8" and orig_classes_dtype == "int16"

  hdf_fn = _get_tmp_file(suffix=".hdf")
  hdf_writer = HDFDatasetWriter(filename=hdf_fn)
  hdf_writer.dump_from_dataset(dataset, use_progress_bar=False)
  hdf_writer.close()

  hdf_dataset = HDFDataset(files=[hdf_fn])
  hdf_dataset.initialize()
  hdf_dataset.init_seq_order(epoch=1)
  hdf_data_dtype = hdf_dataset.get_data_dtype("data")
  hdf_classes_dtype = hdf_dataset.get_data_dtype("classes")
  assert hdf_data_dtype == orig_data_dtype and hdf_classes_dtype == orig_classes_dtype
  hdf_data_dim = hdf_dataset.get_data_dim("data")
  hdf_classes_dim = hdf_dataset.get_data_dim("classes")
  assert hdf_data_dim == 255 and hdf_classes_dim == 10
  hdf_data_shape = hdf_dataset.get_data_shape("data")
  hdf_classes_shape = hdf_dataset.get_data_shape("classes")
  assert hdf_data_shape == [] and hdf_classes_shape == []
  hdf_dataset.load_seqs(0, 1)
  hdf_data_data = hdf_dataset.get_data(0, "data")
  hdf_data_classes = hdf_dataset.get_data(0, "classes")
  assert hdf_data_data.dtype == orig_data_dtype and hdf_data_classes.dtype == orig_classes_dtype


def test_hdf_data_target_int32():
  from GeneratingDataset import StaticDataset
  dataset = StaticDataset([
    {"data": numpy.array([1, 2, 3], dtype="uint8"),
     "classes": numpy.array([2147483647, 2147483646, 2147483645], dtype="int32")}],
    output_dim={"data": (255, 1), "classes": (10, 1)})
  dataset.initialize()
  dataset.init_seq_order(epoch=0)
  dataset.load_seqs(0, 1)
  orig_classes_dtype = dataset.get_data_dtype("classes")
  orig_classes_seq = dataset.get_data(0, "classes")
  assert orig_classes_seq.shape == (3,) and orig_classes_seq[0] == 2147483647
  assert orig_classes_seq.dtype == orig_classes_dtype == "int32"

  hdf_fn = _get_tmp_file(suffix=".hdf")
  hdf_writer = HDFDatasetWriter(filename=hdf_fn)
  hdf_writer.dump_from_dataset(dataset, use_progress_bar=False)
  hdf_writer.close()

  hdf_dataset = HDFDataset(files=[hdf_fn])
  hdf_dataset.initialize()
  hdf_dataset.init_seq_order(epoch=1)
  hdf_classes_dtype = hdf_dataset.get_data_dtype("classes")
  assert hdf_classes_dtype == orig_classes_dtype
  hdf_classes_shape = hdf_dataset.get_data_shape("classes")
  assert hdf_classes_shape == []
  hdf_dataset.load_seqs(0, 1)
  hdf_data_classes = hdf_dataset.get_data(0, "classes")
  assert hdf_data_classes.dtype == orig_classes_dtype
  assert all(hdf_data_classes == orig_classes_seq)


def test_hdf_target_float_dtype():
  from GeneratingDataset import StaticDataset
  dataset = StaticDataset([
    {"data": numpy.array([1, 2, 3], dtype="float32"), "classes": numpy.array([-1, 5], dtype="float32")}],
    output_dim={"data": (1, 1), "classes": (1, 1)})
  orig_data_dtype = dataset.get_data_dtype("data")
  orig_classes_dtype = dataset.get_data_dtype("classes")
  assert orig_data_dtype == "float32" and orig_classes_dtype == "float32"

  hdf_fn = _get_tmp_file(suffix=".hdf")
  hdf_writer = HDFDatasetWriter(filename=hdf_fn)
  hdf_writer.dump_from_dataset(dataset, use_progress_bar=False)
  hdf_writer.close()

  hdf_dataset = HDFDataset(files=[hdf_fn])
  hdf_dataset.initialize()
  hdf_dataset.init_seq_order(epoch=1)
  hdf_data_dtype = hdf_dataset.get_data_dtype("data")
  hdf_classes_dtype = hdf_dataset.get_data_dtype("classes")
  assert hdf_data_dtype == orig_data_dtype and hdf_classes_dtype == orig_classes_dtype
  hdf_data_dim = hdf_dataset.get_data_dim("data")
  hdf_classes_dim = hdf_dataset.get_data_dim("classes")
  assert hdf_data_dim == 1 and hdf_classes_dim == 1
  hdf_data_shape = hdf_dataset.get_data_shape("data")
  hdf_classes_shape = hdf_dataset.get_data_shape("classes")
  assert hdf_data_shape == [] and hdf_classes_shape == []
  hdf_dataset.load_seqs(0, 1)
  hdf_data_data = hdf_dataset.get_data(0, "data")
  hdf_data_classes = hdf_dataset.get_data(0, "classes")
  assert hdf_data_data.dtype == orig_data_dtype and hdf_data_classes.dtype == orig_classes_dtype


def test_hdf_target_float_dense():
  from GeneratingDataset import StaticDataset
  dataset = StaticDataset([
    {"data": numpy.array([[1, 2, 3], [2, 3, 4]], dtype="float32"),
     "classes": numpy.array([[-1, 5], [-2, 4], [-3, 2]], dtype="float32")}])
  orig_data_dtype = dataset.get_data_dtype("data")
  orig_classes_dtype = dataset.get_data_dtype("classes")
  assert orig_data_dtype == "float32" and orig_classes_dtype == "float32"
  orig_data_shape = dataset.get_data_shape("data")
  orig_classes_shape = dataset.get_data_shape("classes")
  assert orig_data_shape == [3] and orig_classes_shape == [2]

  hdf_fn = _get_tmp_file(suffix=".hdf")
  hdf_writer = HDFDatasetWriter(filename=hdf_fn)
  hdf_writer.dump_from_dataset(dataset, use_progress_bar=False)
  hdf_writer.close()

  hdf_dataset = HDFDataset(files=[hdf_fn])
  hdf_dataset.initialize()
  hdf_dataset.init_seq_order(epoch=1)
  hdf_data_dtype = hdf_dataset.get_data_dtype("data")
  hdf_classes_dtype = hdf_dataset.get_data_dtype("classes")
  assert hdf_data_dtype == orig_data_dtype and hdf_classes_dtype == orig_classes_dtype
  hdf_data_dim = hdf_dataset.get_data_dim("data")
  hdf_classes_dim = hdf_dataset.get_data_dim("classes")
  assert hdf_data_dim == orig_data_shape[-1] and hdf_classes_dim == orig_classes_shape[-1]
  hdf_data_shape = hdf_dataset.get_data_shape("data")
  hdf_classes_shape = hdf_dataset.get_data_shape("classes")
  assert hdf_data_shape == orig_data_shape and hdf_classes_shape == orig_classes_shape
  hdf_dataset.load_seqs(0, 1)
  hdf_data_data = hdf_dataset.get_data(0, "data")
  hdf_data_classes = hdf_dataset.get_data(0, "classes")
  assert hdf_data_data.dtype == orig_data_dtype and hdf_data_classes.dtype == orig_classes_dtype


def test_HDFDataset_no_cache_efficiency():
  hdf_fn = generate_hdf_from_other({"class": "Task12AXDataset", "num_seqs": 23})
  hdf_dataset = HDFDataset(files=[hdf_fn], cache_byte_size=0)
  hdf_dataset.initialize()
  hdf_dataset.init_seq_order(epoch=1)
  hdf_dataset.load_seqs(0, 1)
  hdf_dataset.load_seqs(0, 2)
  hdf_dataset.load_seqs(0, 5)
  hdf_dataset.load_seqs(1, 2)
  hdf_dataset.load_seqs(1, 3)
  hdf_dataset.load_seqs(1, 7)
  hdf_dataset.load_seqs(3, 7)
  hdf_dataset.load_seqs(4, 10)
  # TODO... check alloc intervals etc


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
