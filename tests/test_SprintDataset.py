
import sys
sys.path += ["."]  # Python 3 hack

from nose.tools import assert_equal, assert_is_instance, assert_in, assert_not_in, assert_true, assert_false
from Device import Device
from EngineUtil import assign_dev_data, assign_dev_data_single_seq
from EngineBatch import Batch
from Log import log
from Config import Config
from GeneratingDataset import GeneratingDataset
from Dataset import DatasetSeq
from SprintDataset import ExternSprintDataset
import numpy as np
import os
import sys
import better_exchook
better_exchook.install()
better_exchook.replace_traceback_format_tb()


dummyconfig_dict = {
  "num_inputs": 2,
  "num_outputs": 3,
  "hidden_size": (1,),
  "hidden_type": "forward",
  "activation": "relu",
  "bidirectional": False,
}

log.initialize(verbosity=[5])

os.chdir((os.path.dirname(__file__) or ".") + "/..")
assert os.path.exists("rnn.py")
sprintExecPath = "tests/DummySprintExec.py"


class DummyDevice(Device):

  def __init__(self, config=None, blocking=True):
    if not config:
      config = Config()
      config.update(dummyconfig_dict)
    super(DummyDevice, self).__init__(device="cpu", config=config, blocking=blocking)


def generate_batch(seq_idx, dataset):
  batch = Batch()
  batch.add_frames(seq_idx=seq_idx, seq_start_frame=0, length=dataset.get_seq_length(seq_idx))
  return batch


def test_assign_dev_data():
  config = Config()
  config.update(dummyconfig_dict)
  print("Create ExternSprintDataset")
  dataset = ExternSprintDataset(
    [sys.executable, sprintExecPath],
    "--*.feature-dimension=2 --*.trainer-output-dimension=3 --*.crnn-dataset=DummyDataset(2,3,4)")
  dataset.init_seq_order(epoch=1)
  assert_true(dataset.is_less_than_num_seqs(0))
  recurrent = False
  batch_generator = dataset.generate_batches(recurrent_net=recurrent, batch_size=512)
  batches = batch_generator.peek_next_n(2)
  assert_equal(len(batches), 2)
  print("Create Device")
  device = DummyDevice(config=config)
  success, num_batches = assign_dev_data(device, dataset, batches)
  assert_true(success)
  assert_equal(num_batches, len(batches))


def test_window():
  input_dim = 2
  output_dim = 3
  num_seqs = 2
  seq_len = 5
  window = 3
  dataset_kwargs = dict(
    sprintTrainerExecPath=[sys.executable, sprintExecPath],
    sprintConfigStr=" ".join([
      "--*.feature-dimension=%i" % input_dim,
      "--*.trainer-output-dimension=%i" % output_dim,
      "--*.crnn-dataset=DummyDataset(input_dim=%i,output_dim=%i,num_seqs=%i,seq_len=%i)" % (
        input_dim, output_dim, num_seqs, seq_len)]))
  dataset1 = ExternSprintDataset(**dataset_kwargs)
  dataset2 = ExternSprintDataset(window=window, **dataset_kwargs)
  try:
    dataset1.init_seq_order(epoch=1)
    dataset2.init_seq_order(epoch=1)
    dataset1.load_seqs(0, 1)
    dataset2.load_seqs(0, 1)
    assert_equal(dataset1.get_data_dim("data"), input_dim)
    assert_equal(dataset2.get_data_dim("data"), input_dim * window)
    data1 = dataset1.get_data(0, "data")
    data2 = dataset2.get_data(0, "data")
    assert_equal(data1.shape, (seq_len, input_dim))
    assert_equal(data2.shape, (seq_len, window * input_dim))
    data2a = data2.reshape(seq_len, window, input_dim)
    print("data1:")
    print(data1)
    print("data2:")
    print(data2)
    print("data1[0]:")
    print(data1[0])
    print("data2[0]:")
    print(data2[0])
    print("data2a[0,0]:")
    print(data2a[0, 0])
    assert_equal(list(data2a[0, 0]), [0] * input_dim)  # zero-padded left
    assert_equal(list(data2a[0, 1]), list(data1[0]))
    assert_equal(list(data2a[0, 2]), list(data1[1]))
    assert_equal(list(data2a[1, 0]), list(data1[0]))
    assert_equal(list(data2a[1, 1]), list(data1[1]))
    assert_equal(list(data2a[1, 2]), list(data1[2]))
    assert_equal(list(data2a[-1, 2]), [0] * input_dim)  # zero-padded right
  finally:
    dataset1.exit_handler()
    dataset2.exit_handler()


if __name__ == "__main__":
  test_assign_dev_data()
