
from hdf_dump import *
import os
from Log import log
import tempfile
from GeneratingDataset import DummyDataset
from HDFDataset import HDFDataset
from Util import DictAsObj
import better_exchook
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
