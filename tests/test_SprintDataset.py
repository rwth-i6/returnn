
from nose.tools import assert_equal, assert_is_instance, assert_in, assert_not_in, assert_true, assert_false
from Device import Device
from EngineUtil import assign_dev_data, assign_dev_data_single_seq
from EngineBatch import Batch
from Log import log
from Config import Config
from GeneratingDataset import GeneratingDataset
from Dataset import DatasetSeq
from ExternSprintDataset import ExternSprintDataset
import numpy as np
import os
import sys
import better_exchook
better_exchook.replace_traceback_format_tb()


dummyconfig_dict = {
  "num_inputs": 2,
  "num_outputs": 3,
  "hidden_size": (1,),
  "hidden_type": "forward",
  "activation": "relu",
  "bidirectional": False,
}

log.initialize()

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
  device = DummyDevice(config=config)
  dataset = ExternSprintDataset(sprintExecPath,
                                "--*.feature-dimension=2 --*.trainer-output-dimension=3 "
                                "--*.crnn-dataset=DummyDataset(2,3,4)")
  dataset.init_seq_order(epoch=1)
  assert_true(dataset.is_less_than_num_seqs(0))
  recurrent = False
  batch_generator = dataset.generate_batches(recurrent_net=recurrent, batch_size=512)
  batches = batch_generator.peek_next_n(2)
  assert_equal(len(batches), 2)
  success, num_batches = assign_dev_data(device, dataset, batches)
  assert_true(success)
  assert_equal(num_batches, len(batches))
