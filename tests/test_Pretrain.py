
from __future__ import print_function
import _setup_test_env  # noqa
from nose.tools import assert_equal, assert_in, assert_not_in, assert_true
from returnn.pretrain import pretrain_from_config
from returnn.config import Config


config1_dict = {
  "pretrain": "default",
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
  pretrain = pretrain_from_config(config)
  assert_true(pretrain)


def test_config1():
  config = Config()
  config.update(config1_dict)
  pretrain = pretrain_from_config(config)
  assert_equal(pretrain.get_train_num_epochs(), 2)
  net1_json = pretrain.get_network_json_for_epoch(1)
  net2_json = pretrain.get_network_json_for_epoch(2)
  net3_json = pretrain.get_network_json_for_epoch(3)
  assert_in("hidden_0", net1_json)
  assert_not_in("hidden_1", net1_json)
  assert_in("hidden_0", net2_json)
  assert_in("hidden_1", net2_json)
  assert_equal(net2_json, net3_json)
