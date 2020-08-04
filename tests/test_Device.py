
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from returnn.config import Config
from returnn.theano.engine import Engine
from returnn.theano.device import Device
from returnn.log import log
import returnn.theano.util as theano_util

log.initialize()
theano_util.monkey_patches()


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

