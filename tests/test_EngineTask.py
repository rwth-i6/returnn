
from __future__ import print_function

import _setup_test_env  # noqa
from nose.tools import assert_equal, assert_is_instance, assert_in, assert_greater
import time
import sys
from pprint import pprint

from returnn.theano.engine_task import TaskThread, TrainTaskThread, EvalTaskThread
from returnn.theano.device import Device
from returnn.config import Config
from returnn.log import log
from returnn.util.basic import hms, NumbersDict


config = Config()
config.update({
  "multiprocessing": False,
  "blocking": True,
  "device": "cpu",
  "num_epochs": 1,
  "num_inputs": 3,
  "num_outputs": 2,
})
config.network_topology_json = """
{
"output": {"class": "softmax", "loss": "ce"}
}
"""

class DummyBatches:
  def __init__(self):
    pass

  def completed_frac(self):
    return 0.5

  def get_current_batch_idx(self):
    return 0


class DummyNetwork:
  recurrent = True
  ctc_priors = None
  loss = "ce"


class DummyUpdater:
  isInitialized = True


def getDeviceBatchRunParent(dev, task):

  if task == "train":
    parent_clazz_base = TrainTaskThread
  elif task == "eval":
    parent_clazz_base = EvalTaskThread
  else:
    assert "invalid: %r" % task
    return

  class DummyDeviceBatchRunParent(parent_clazz_base):

    def __init__(self):
      init_kwargs = {}
      if parent_clazz_base is TrainTaskThread:
        init_kwargs.update(dict(
          learning_rate=1.0,
          updater=DummyUpdater()
        ))
      super(DummyDeviceBatchRunParent, self).__init__(
        network=DummyNetwork(),
        report_prefix="test %s run" % task,
        batches=DummyBatches(),
        devices=[dev],
        data=None,
        **init_kwargs
      )

      self.interactive = False
      self.start_time = time.time()

    def start(self):
      # Do nothing. Don't start the thread.
      print("DummyDeviceBatchRunParent: dummy start thread (ignored)")

  return DummyDeviceBatchRunParent()


class DummyDevice(Device):
  def __init__(self):
    super(DummyDevice, self).__init__("cpu", config=config, blocking=True)

  def run(self, task):
    print("DummyDevice run %r (ignored)" % task)
    self.run_called_count += 1

  def set_net_params(self, network):
    print("DummyDevice set_net_params (ignored)")


class DummyDeviceBatchRun(TaskThread.DeviceBatchRun):

  def __init__(self, task="train"):
    dev = DummyDevice()
    parent = getDeviceBatchRunParent(dev=dev, task=task)
    super(DummyDeviceBatchRun, self).__init__(parent=parent, devices=[dev])

  def start(self):
    # Do nothing. Don't start the thread.
    print("DummyDeviceBatchRun: dummy start thread (ignored)")

  def allocate(self):
    self.devices_batches_idx = self.parent.batches.get_current_batch_idx()
    assert len(self.alloc_devices) == 1
    self.devices_batches = [None] * len(self.alloc_devices)
    self.num_frames = NumbersDict(13)
    batch_dim = 1
    self.alloc_devices[0].alloc_data(shapes={
      "data": (self.num_frames["data"], batch_dim, config.typed_value("num_inputs")),
      "classes": (self.num_frames["classes"], batch_dim)})
    self.parent.num_frames += self.num_frames
    self.allocated = True

  def set_dummy_dev_output(self, output, outputs_format):
    assert len(self.alloc_devices) == 1
    dev = self.alloc_devices[0]
    assert isinstance(dev, Device)
    assert dev.blocking
    dev.output = output
    dev.outputs_format = outputs_format


def test_DeviceBatchRun_outputs_format():
  # TODO: This is broken...
  return

  dev_run = DummyDeviceBatchRun(task="train")
  assert len(dev_run.alloc_devices) == 1

  # Simulate epoch start.
  trainer = dev_run.parent
  dev_run.alloc_devices[0].start_epoch_stats()
  trainer.initialize()

  # Simulate one batch.
  dev_run.allocate()
  dev_run.device_run()
  dev_run.set_dummy_dev_output(outputs_format=["cost:foo"], output=[1.42])
  dev_run.finish()

  assert_is_instance(dev_run.result, dict)
  assert_in("results", dev_run.result)
  res_outputss = dev_run.result["results"]
  assert_is_instance(res_outputss, list)
  assert_equal(len(res_outputss), len(dev_run.alloc_devices))
  res_outputs = res_outputss[0]
  assert_is_instance(res_outputs, list)
  res_outputs_format = dev_run.result["result_format"]
  assert_is_instance(res_outputs_format, list)
  res = Device.make_result_dict(res_outputs, res_outputs_format)
  assert_is_instance(res, dict)
  pprint(res)

  # Simulate epoch end.
  print("train epoch score:", trainer.score, "elapsed:", hms(trainer.elapsed))
  trainer.finalize()
  dev_run.alloc_devices[0].finish_epoch_stats()

  # Now simulate the eval.
  dev_run = DummyDeviceBatchRun(task="eval")
  assert len(dev_run.alloc_devices) == 1

  # Simulate epoch start.
  tester = dev_run.parent
  dev_run.alloc_devices[0].start_epoch_stats()
  tester.initialize()

  # Simulate one batch.
  dev_run.allocate()
  dev_run.device_run()
  dev_run.set_dummy_dev_output(outputs_format=["cost:foo", "error:foo"], output=[1.42, 2.34])
  dev_run.finish()

  # Simulate epoch end.
  print("eval epoch elapsed:", hms(tester.elapsed))
  tester.finalize()
  dev_run.alloc_devices[0].finish_epoch_stats()

  print("eval results:", tester.score, tester.error)

  assert_is_instance(dev_run.result, dict)
  assert_in("results", dev_run.result)
  res_outputss = dev_run.result["results"]
  assert_is_instance(res_outputss, list)
  assert_equal(len(res_outputss), len(dev_run.alloc_devices))
  res_outputs = res_outputss[0]
  assert_is_instance(res_outputs, list)
  res_outputs_format = dev_run.result["result_format"]
  assert_is_instance(res_outputs_format, list)
  res = Device.make_result_dict(res_outputs, res_outputs_format)
  assert_is_instance(res, dict)
  pprint(res)

  assert_greater(tester.score, 0)
  assert_greater(tester.error, 0)
