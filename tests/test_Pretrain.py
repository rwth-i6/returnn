
from nose.tools import assert_equal, assert_is_instance, assert_in, assert_not_in, assert_true, assert_false
from Pretrain import Pretrain, pretrainFromConfig
from Config import Config


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
  pretrain = pretrainFromConfig(config)
  assert_true(pretrain)


def test_config1():
  config = Config()
  config.update(config1_dict)
  pretrain = pretrainFromConfig(config)
  assert_equal(pretrain.get_train_num_epochs(), 2)
  net1_json = pretrain._get_network_json_for_epoch(1)
  net2_json = pretrain._get_network_json_for_epoch(2)
  net3_json = pretrain._get_network_json_for_epoch(3)
  assert_in("hidden_0", net1_json)
  assert_not_in("hidden_1", net1_json)
  assert_in("hidden_0", net2_json)
  assert_in("hidden_1", net2_json)
  assert_equal(net2_json, net3_json)
