
# -*- coding: utf8 -*-

import sys
import os

import _setup_test_env  # noqa
sys.path += ["tools"]

from hdf_dump import *
import os
from returnn.log import log
import tempfile
from returnn.datasets.generating import DummyDataset
from returnn.datasets.hdf import HDFDataset
from returnn.util.basic import DictAsObj
import unittest
from returnn.util import better_exchook
better_exchook.replace_traceback_format_tb()


log.initialize()

options = {
  "epoch": 1,
  "start_seq": 0,
  "end_seq": float("inf")
}


def test_hdf_dataset_init():
  hdf_filename = tempfile.mktemp(suffix=".hdf", prefix="nose-dataset-init")
  hdf_dataset_init(hdf_filename)
  assert os.path.exists(hdf_filename)
  os.remove(hdf_filename)


def test_hdf_create():
  hdf_filename = tempfile.mktemp(suffix=".hdf", prefix="nose-dataset-create")
  hdf_dataset = hdf_dataset_init(hdf_filename)
  assert os.path.exists(hdf_filename)

  dataset = DummyDataset(input_dim=2, output_dim=3, num_seqs=4)
  dataset.init_seq_order(epoch=1)

  hdf_dump_from_dataset(dataset, hdf_dataset, DictAsObj(options))
  hdf_close(hdf_dataset)

  os.remove(hdf_filename)


def test_hdf_create_and_load():
  hdf_filename = tempfile.mktemp(suffix=".hdf", prefix="nose-dataset-load")
  hdf_dataset = hdf_dataset_init(hdf_filename)
  assert os.path.exists(hdf_filename)

  dataset = DummyDataset(input_dim=2, output_dim=3, num_seqs=4)
  dataset.init_seq_order(epoch=1)

  hdf_dump_from_dataset(dataset, hdf_dataset, DictAsObj(options))
  hdf_close(hdf_dataset)

  loaded_dataset = HDFDataset()
  loaded_dataset.add_file(hdf_filename)

  os.remove(hdf_filename)


def test_hdf_create_unicode_labels():
  hdf_filename = tempfile.mktemp(suffix=".hdf", prefix="nose-dataset-create")
  hdf_dataset = hdf_dataset_init(hdf_filename)
  assert os.path.exists(hdf_filename)

  dataset = DummyDataset(input_dim=2, output_dim=3, num_seqs=4)
  assert "classes" in dataset.get_target_list()
  dataset.labels["classes"] = ['’', 'ä', 'x']  # have some Unicode chars here
  dataset.init_seq_order(epoch=1)

  hdf_dump_from_dataset(dataset, hdf_dataset, DictAsObj(options))
  hdf_close(hdf_dataset)

  os.remove(hdf_filename)


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
