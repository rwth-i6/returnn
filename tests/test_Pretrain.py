from __future__ import annotations
import _setup_test_env  # noqa
from returnn.pretrain import pretrain_from_config
from returnn.config import Config


config_dict = {
    "pretrain": "default",
    "num_inputs": 40,
    "num_outputs": 4498,
}


net_dict = {
    "hidden_0": {"class": "linear", "n_out": 7, "dropout": 0.1, "activation": "relu"},
    "hidden_1": {"class": "linear", "n_out": 8, "dropout": 0.1, "activation": "relu", "from": ["hidden_0"]},
    "output": {"class": "softmax", "loss": "ce", "from": ["hidden_1"]},
}


net_dict2 = {
    "lstm0_fw": {"class": "lstm_opt", "n_out": 500, "dropout": 0.1, "sampling": 1, "reverse": False},
    "lstm0_bw": {"class": "lstm_opt", "n_out": 500, "dropout": 0.1, "sampling": 1, "reverse": True},
    "lstm1_fw": {
        "class": "lstm_opt",
        "n_out": 500,
        "dropout": 0.1,
        "sampling": 1,
        "reverse": False,
        "from": ["lstm0_fw", "lstm0_bw"],
    },
    "lstm1_bw": {
        "class": "lstm_opt",
        "n_out": 500,
        "dropout": 0.1,
        "sampling": 1,
        "reverse": True,
        "from": ["lstm0_fw", "lstm0_bw"],
    },
    "lstm2_fw": {
        "class": "lstm_opt",
        "n_out": 500,
        "dropout": 0.1,
        "sampling": 1,
        "reverse": False,
        "from": ["lstm1_fw", "lstm1_bw"],
    },
    "lstm2_bw": {
        "class": "lstm_opt",
        "n_out": 500,
        "dropout": 0.1,
        "sampling": 1,
        "reverse": True,
        "from": ["lstm1_fw", "lstm1_bw"],
    },
    "output": {"class": "softmax", "loss": "ce", "from": ["lstm2_fw", "lstm2_bw"]},
}


def test_config_net_dict1():
    config = Config()
    config.update(config_dict)
    config.typed_dict["network"] = net_dict
    pretrain = pretrain_from_config(config)
    assert pretrain.get_train_num_epochs() == 2
    net1_json = pretrain.get_network_json_for_epoch(1)
    net2_json = pretrain.get_network_json_for_epoch(2)
    net3_json = pretrain.get_network_json_for_epoch(3)
    assert "hidden_0" in net1_json
    assert "hidden_1" not in net1_json
    assert "hidden_0" in net2_json
    assert "hidden_1" in net2_json
    assert net2_json == net3_json


def test_config_net_dict2():
    config = Config()
    config.update(config_dict)
    config.typed_dict["network"] = net_dict2
    pretrain = pretrain_from_config(config)
    assert pretrain.get_train_num_epochs() == 3
