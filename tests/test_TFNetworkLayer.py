
# start: nosetests $this_file --nologcapture

import tensorflow as tf
import sys
sys.path += ["."]  # Python 3 hack
from nose.tools import assert_equal, assert_is_instance
import numpy.testing
import better_exchook
better_exchook.replace_traceback_format_tb()

from Config import Config
from TFNetwork import *
from TFNetworkLayer import *


def test_combine_layer_net_construct():
  with tf.Session():
    net_dict = {
      "lstm0_fw": {"class": "rec", "unit": "lstmp", "n_out": 5, "dropout": 0.0, "L2": 0.01, "direction": 1},
      "lstm0_bw": {"class": "rec", "unit": "lstmp", "n_out": 5, "dropout": 0.0, "L2": 0.01, "direction": -1},
      "lstm0_avg": {"class": "combine", "kind": "average", "from": ["lstm0_fw", "lstm0_bw"], "trainable": False},
      "output": {"class": "softmax", "loss": "ce", "from": ["lstm0_avg"]}
    }
    config = Config()
    config.update(dict(num_inputs=4, num_outputs=9))
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(net_dict)
