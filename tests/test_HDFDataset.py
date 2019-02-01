
import sys
sys.path += ["."]  # Python 3 hack

from HDFDataset import HDFDataset, SiameseHDFDataset
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
