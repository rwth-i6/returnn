
from nose.tools import assert_equal, assert_is_instance, assert_in, assert_true, assert_false
from Pretrain import Pretrain, pretrainFromConfig
from Config import Config


config1_dict = {
  "num_inputs": 5,
  "num_outputs": 10,
  "hidden_size": (7, 8,),
  "hidden_type": "forward",
  "activation": "relu",
  "bidirectional": False,
}


def test_init_config1():
  config = Config()
  config.update(config1_dict)
  pretrainFromConfig(config)
