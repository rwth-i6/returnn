
import sys
sys.path += ["."]  # Python 3 hack

from nose.tools import assert_equal, assert_is_instance, assert_in, assert_not_in, assert_true, assert_false
from GeneratingDataset import *
from Dataset import DatasetSeq
import numpy as np
import os
import unittest

import better_exchook
better_exchook.replace_traceback_format_tb()
from Log import log
log.initialize(verbosity=[5])



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


@unittest.skipIf(not os.path.exists("/tmp/enwik8.zip"), "we will not trigger the download")
def test_Enwik8Corpus_batch_num_seqs():
  dataset = Enwik8Corpus(path="/tmp", subset="validation", seq_len=13)
  dataset.init_seq_order(epoch=17)
  data = b""
  n = 0
  while dataset.is_less_than_num_seqs(n) and n < 100:
    dataset.load_seqs(n, n + 1)
    data += bytes(dataset.get_data(n, "data"))
    n += 1

  batch_size = 23
  batch_data = [b"" for i in range(batch_size)]
  dataset = Enwik8Corpus(path="/tmp", subset="validation", seq_len=9, batch_num_seqs=batch_size)
  dataset.init_seq_order(epoch=31)
  n = 0
  while dataset.is_less_than_num_seqs(n) and n < 100:
    dataset.load_seqs(n, n + 1)
    new_data = bytes(dataset.get_data(n, "data"))
    batch_data[n % batch_size] += new_data
    n += 1
  assert data.startswith(batch_data[0])


def test_StaticDataset_custom_keys():
  dataset = StaticDataset([{"source": numpy.array([1, 2, 3]), "target": numpy.array([3, 4, 5, 6, 7])}])
  dataset.init_seq_order(epoch=1)
  assert dataset.num_seqs == 1
  assert_equal(dataset.get_data_keys(), ["source", "target"])
  assert_equal(dataset.num_outputs["source"][1], 1)
  assert_equal(dataset.num_outputs["target"][1], 1)
  dataset.load_seqs(0, 1)
  assert_equal(list(dataset.get_data(0, "source")), [1, 2, 3])
  assert_equal(list(dataset.get_data(0, "target")), [3, 4, 5, 6, 7])


def test_StaticDataset_custom_keys_with_dims():
  dataset = StaticDataset(
    data=[{"source": numpy.array([1, 2, 3]), "target": numpy.array([3, 4, 5, 6, 7])}],
    output_dim={"source": [5, 1], "target": [10, 1]})
  dataset.init_seq_order(epoch=1)
  assert dataset.num_seqs == 1
  assert_equal(dataset.get_data_keys(), ["source", "target"])
  assert_equal(dataset.num_outputs["source"][1], 1)
  assert_equal(dataset.num_outputs["target"][1], 1)
  dataset.load_seqs(0, 1)
  assert_equal(list(dataset.get_data(0, "source")), [1, 2, 3])
  assert_equal(list(dataset.get_data(0, "target")), [3, 4, 5, 6, 7])
