
from nose.tools import assert_equal, assert_is_instance, assert_in, assert_not_in, assert_true, assert_false
from Device import Device
from EngineUtil import assign_dev_data, assign_dev_data_single_seq
from EngineBatch import Batch
from Log import log
from Config import Config
from GeneratingDataset import GeneratingDataset
from Dataset import DatasetSeq
import numpy as np
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


class DummyDevice(Device):

  def __init__(self, config=None, blocking=True):
    if not config:
      config = Config()
      config.update(dummyconfig_dict)
    super(DummyDevice, self).__init__(device="cpu", config=config, blocking=blocking)

  #def initialize(self, config, update_specs=None, json_content=None, train_param_args=None):
  #  pass


def test_DummyDevice():
  DummyDevice()


class DummyDataset(GeneratingDataset):

  def __init__(self, input_dim, output_dim, num_seqs, seq_len=2):
    assert input_dim > 0
    assert output_dim > 0
    assert num_seqs > 0
    super(DummyDataset, self).__init__(input_dim=input_dim, output_dim=output_dim, num_seqs=num_seqs)
    self.seq_len = seq_len
    self.init_seq_order(epoch=1)

  def generate_seq(self, seq_idx):
    seq_len = self.seq_len
    i1 = seq_idx
    i2 = i1 + seq_len * self.num_inputs
    features = np.array(range(i1, i2)).reshape((seq_len, self.num_inputs))
    i1, i2 = i2, i2 + seq_len
    targets = np.array(range(i1, i2))
    return DatasetSeq(seq_idx=seq_idx, features=features, targets=targets)


def test_DummyDataset():
  data = DummyDataset(input_dim=2, output_dim=3, num_seqs=10)
  assert_equal(data.num_seqs, 10)


def generate_batch(seq_idx, dataset):
  batch = Batch()
  batch.add_frames(seq_idx=seq_idx, seq_start_frame=0, length=dataset.get_seq_length(seq_idx))
  return batch


def test_assign_dev_data():
  config = Config()
  config.update(dummyconfig_dict)
  device = DummyDevice(config=config)
  dataset = DummyDataset(input_dim=config.int("num_inputs", 0),
                         output_dim=config.int("num_outputs", 0),
                         num_seqs=10)
  batches = [generate_batch(0, dataset), generate_batch(1, dataset)]
  success, num_batches = assign_dev_data(device, dataset, batches)
  assert_true(success)
  assert_equal(num_batches, len(batches))
