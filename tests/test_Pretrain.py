
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

config2_dict = {
  "pretrain": "default",
  "num_inputs": 40,
  "num_outputs": 4498,
  "bidirectional": True,
  "hidden_size": (500,500,500),
  "hidden_type": "lstm_opt",
  "activation": "sigmoid",
  "dropout": 0.1,
}

config3_dict = {
  "pretrain": "default",
  "num_inputs": 40,
  "num_outputs": 4498,
}

config3_json = """
{
"lstm0_fw" : { "class" : "lstm_opt", "n_out" : 500, "dropout": 0.1, "sampling" : 1, "reverse" : false },
"lstm0_bw" : { "class" : "lstm_opt", "n_out" : 500, "dropout": 0.1, "sampling" : 1, "reverse" : true },

"lstm1_fw" : { "class" : "lstm_opt", "n_out" : 500, "dropout": 0.1, "sampling" : 1, "reverse" : false, "from" : ["lstm0_fw", "lstm0_bw"] },
"lstm1_bw" : { "class" : "lstm_opt", "n_out" : 500, "dropout": 0.1, "sampling" : 1, "reverse" : true, "from" : ["lstm0_fw", "lstm0_bw"] },

"lstm2_fw" : { "class" : "lstm_opt", "n_out" : 500, "dropout": 0.1, "sampling" : 1, "reverse" : false, "from" : ["lstm1_fw", "lstm1_bw"] },
"lstm2_bw" : { "class" : "lstm_opt", "n_out" : 500, "dropout": 0.1, "sampling" : 1, "reverse" : true, "from" : ["lstm1_fw", "lstm1_bw"] },

"output" :   { "class" : "softmax", "loss" : "ce", "from" : ["lstm2_fw", "lstm2_bw"] }
}
"""


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


def test_config2():
  config = Config()
  config.update(config2_dict)
  pretrain = pretrainFromConfig(config)
  assert_equal(pretrain.get_train_num_epochs(), 3)


def test_config3():
  config = Config()
  config.update(config3_dict)
  config.network_topology_json = config3_json
  pretrain = pretrainFromConfig(config)
  assert_equal(pretrain.get_train_num_epochs(), 3)
