
import sys
sys.path += ["."]  # Python 3 hack


from returnn.config import Config
from Engine import Engine
from Device import Device
from returnn.log import log
import TheanoUtil

log.initialize()
TheanoUtil.monkey_patches()


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

