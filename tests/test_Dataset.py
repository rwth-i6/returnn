
from nose.tools import assert_equal, assert_is_instance, assert_in, assert_not_in, assert_true, assert_false
from GeneratingDataset import GeneratingDataset, DummyDataset
from Dataset import DatasetSeq
import numpy as np


def test_generate_batches():
  dataset = DummyDataset(input_dim=2, output_dim=3, num_seqs=20)
  dataset.init_seq_order(1)
  batch_gen = dataset.generate_batches(recurrent_net=False, max_seqs=2, batch_size=5)
  while batch_gen.has_more():
    batch_gen.peek_next_n(1)
    batch_gen.advance(1)

def test_generate_batches_recurrent():
  dataset = DummyDataset(input_dim=2, output_dim=3, num_seqs=20)
  dataset.init_seq_order(1)
  batch_gen = dataset.generate_batches(recurrent_net=True, max_seqs=2, batch_size=5)
  while batch_gen.has_more():
    batch_gen.peek_next_n(1)
    batch_gen.advance(1)
