

from Config import Config
from Engine import Engine
from Device import Device
from Log import log

log.initialize()


def test_Device_blocking_init():
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

  Device("cpu", config=config, blocking=True)

