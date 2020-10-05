

import _setup_test_env  # noqa

from returnn.config import Config
from returnn.theano.engine import Engine
from returnn.theano.device import Device
from returnn.log import log
import returnn.theano.util as theano_util

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
