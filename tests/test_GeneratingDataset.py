
from nose.tools import assert_equal, assert_is_instance, assert_in, assert_not_in, assert_true, assert_false
from GeneratingDataset import GeneratingDataset
from Dataset import DatasetSeq
import numpy as np


class DummyDataset(GeneratingDataset):

  def __init__(self, input_dim, output_dim, num_seqs, seq_len=2):
    assert input_dim > 0
    assert output_dim > 0
    assert num_seqs > 0
    super(DummyDataset, self).__init__(input_dim=input_dim, output_dim=output_dim, num_seqs=num_seqs)
    self.seq_len = seq_len

  def generate_seq(self, seq_idx):
    seq_len = self.seq_len
    i1 = seq_idx
    i2 = i1 + seq_len * self.num_inputs
    features = np.array(range(i1, i2)).reshape((seq_len, self.num_inputs))
    i1, i2 = i2, i2 + seq_len
    targets = np.array(range(i1, i2))
    return DatasetSeq(seq_idx=seq_idx, features=features, targets=targets)


def test_init():
  dataset = DummyDataset(input_dim=2, output_dim=3, num_seqs=4)
  assert_equal(dataset.num_inputs, 2)
  assert_equal(dataset.num_outputs, {"classes": 3})
  assert_equal(dataset.num_seqs, 4)


def test_load_seqs():
  dataset = DummyDataset(input_dim=2, output_dim=3, num_seqs=4)
  dataset.init_seq_order(epoch=1)
  dataset.load_seqs(0, 1)
  dataset.load_seqs(1, 3)

