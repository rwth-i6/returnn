
from nose.tools import assert_equal, assert_is_instance, assert_in, assert_not_in, assert_true, assert_false
from GeneratingDataset import GeneratingDataset, DummyDataset
from Dataset import DatasetSeq
import numpy as np



def test_init():
  dataset = DummyDataset(input_dim=2, output_dim=3, num_seqs=4)
  assert_equal(dataset.num_inputs, 2)
  assert_equal(dataset.num_outputs, {"classes": [3, 1], "data": [2, 2]})
  assert_equal(dataset.num_seqs, 4)


def test_load_seqs():
  dataset = DummyDataset(input_dim=2, output_dim=3, num_seqs=4)
  dataset.init_seq_order(epoch=1)
  dataset.load_seqs(0, 1)
  dataset.load_seqs(1, 3)


