
import sys

import _setup_test_env  # noqa
import unittest
from nose.tools import assert_equal, assert_is_instance, assert_in, assert_true, assert_false
from returnn.network_description import LayerNetworkDescription
from returnn.config import Config
from returnn.util.basic import dict_diff_str
from pprint import pprint
from returnn.util import better_exchook
from returnn.util.basic import BackendEngine

try:
  # noinspection PyPackageRequirements
  import theano
  BackendEngine.select_engine(engine=BackendEngine.Theano)
except ImportError:
  theano = None


if theano:
  import returnn.theano.util

  returnn.theano.util.monkey_patches()

  from returnn.theano.network import LayerNetwork
  from returnn.theano.layers.hidden import ForwardLayer

else:
  LayerNetwork = None
  ForwardLayer = None


def test_init():
  n_in = 5
  n_out = {"classes": (10, 1)}
  desc = LayerNetworkDescription(
    num_inputs=n_in, num_outputs=n_out,
    hidden_info=[],
    output_info={},
    default_layer_info={})

  assert_equal(desc.num_inputs, n_in)
  assert_equal(desc.num_outputs, n_out)


def test_num_inputs_outputs_old():
  n_in = 5
  n_out = 10
  config = Config()
  config.update({"num_inputs": n_in, "num_outputs": n_out})
  num_inputs, num_outputs = LayerNetworkDescription.num_inputs_outputs_from_config(config)
  assert_equal(num_inputs, n_in)
  assert_is_instance(num_outputs, dict)
  assert_equal(len(num_outputs), 1)
  assert_in("classes", num_outputs)
  assert_equal(num_outputs["classes"], (n_out, 1))


def test_num_inputs_outputs_special_dataset():
  config = Config()
  config.update({
    "train": {"class": "CopyTaskDataset", "num_seqs": 1000, "nsymbols": 80, "minlen": 100, "maxlen": 100},
    "num_outputs": {"data": [80, 1], "classes": [80, 1]}})
  num_inputs, num_outputs = LayerNetworkDescription.num_inputs_outputs_from_config(config)
  assert_equal(num_inputs, 80)
  assert_in("data", num_outputs)
  assert_in("classes", num_outputs)
  assert_equal(num_outputs["classes"], (80, 1))
  assert_equal(num_outputs["data"], (80, 1))


config1_dict = {
  "num_inputs": 5,
  "num_outputs": 10,
  "hidden_size": (7, 8,),
  "hidden_type": "forward",
  "activation": "relu",
  "bidirectional": False,
}

config2_dict = {
  "use_theano": True,
  "pretrain": "default",
  "num_inputs": 40,
  "num_outputs": 4498,
  "bidirectional": True,
  "hidden_size": (500, 500, 500),
  "hidden_type": "lstm_opt",
  "activation": "sigmoid",
  "dropout": 0.1,
}



def test_config1_basic():
  config = Config()
  config.update(config1_dict)
  desc = LayerNetworkDescription.from_config(config)
  assert_is_instance(desc.hidden_info, list)
  assert_equal(len(desc.hidden_info), len(config1_dict["hidden_size"]))
  assert_equal(desc.num_inputs, config1_dict["num_inputs"])


@unittest.skipIf(not theano, "not theano")
def test_network_config1_init():
  config = Config()
  config.update(config1_dict)
  network = LayerNetwork.from_config_topology(config)
  assert_in("hidden_0", network.hidden)
  assert_in("hidden_1", network.hidden)
  assert_equal(len(network.hidden), 2)
  assert_is_instance(network.hidden["hidden_0"], ForwardLayer)
  assert_equal(network.hidden["hidden_0"].layer_class, "hidden")
  assert_false(network.recurrent)

  json_content = network.to_json_content()
  pprint(json_content)
  assert_in("hidden_0", json_content)
  assert_equal(json_content["hidden_0"]["class"], "hidden")
  assert_in("hidden_1", json_content)
  assert_in("output", json_content)


@unittest.skipIf(not theano, "not theano")
def test_NetworkDescription_to_json_config1():
  config = Config()
  config.update(config1_dict)
  desc = LayerNetworkDescription.from_config(config)
  desc_json_content = desc.to_json_content()
  pprint(desc_json_content)
  assert_in("hidden_0", desc_json_content)
  assert_equal(desc_json_content["hidden_0"]["class"], "forward")
  assert_in("hidden_1", desc_json_content)
  assert_in("output", desc_json_content)
  orig_network = LayerNetwork.from_description(desc)
  assert_in("hidden_0", orig_network.hidden)
  assert_in("hidden_1", orig_network.hidden)
  assert_equal(len(orig_network.hidden), 2)
  assert_is_instance(orig_network.hidden["hidden_0"], ForwardLayer)
  assert_equal(orig_network.hidden["hidden_0"].layer_class, "hidden")
  orig_json_content = orig_network.to_json_content()
  pprint(orig_json_content)
  assert_in("hidden_0", orig_json_content)
  assert_equal(orig_json_content["hidden_0"]["class"], "hidden")
  assert_in("hidden_1", orig_json_content)
  assert_in("output", orig_json_content)
  new_network = LayerNetwork.from_json(
    desc_json_content,
    config1_dict["num_inputs"],
    {"classes": (config1_dict["num_outputs"], 1)})
  new_json_content = new_network.to_json_content()
  if orig_json_content != new_json_content:
    print(dict_diff_str(orig_json_content, new_json_content))
    assert_equal(orig_json_content, new_network.to_json_content())


@unittest.skipIf(not theano, "not theano")
def test_config1_to_json_network_copy():
  config = Config()
  config.update(config1_dict)
  orig_network = LayerNetwork.from_config_topology(config)
  orig_json_content = orig_network.to_json_content()
  pprint(orig_json_content)
  new_network = LayerNetwork.from_json(orig_json_content, orig_network.n_in, orig_network.n_out)
  assert_equal(orig_network.n_in, new_network.n_in)
  assert_equal(orig_network.n_out, new_network.n_out)
  new_json_content = new_network.to_json_content()
  if orig_json_content != new_json_content:
    print(dict_diff_str(orig_json_content, new_json_content))
    assert_equal(orig_json_content, new_network.to_json_content())


@unittest.skipIf(not theano, "not theano")
def test_config2_bidirect_lstm():
  config = Config()
  config.update(config2_dict)
  desc = LayerNetworkDescription.from_config(config)
  assert_true(desc.bidirectional)
  network = LayerNetwork.from_config_topology(config)
  net_json = network.to_json_content()
  pprint(net_json)
  assert_in("output", net_json)
  assert_in("hidden_0_fw", net_json)
  assert_in("hidden_0_bw", net_json)
  assert_in("hidden_1_fw", net_json)
  assert_in("hidden_1_bw", net_json)
  assert_in("hidden_2_fw", net_json)
  assert_in("hidden_2_bw", net_json)
  assert_equal(net_json["output"]["from"], ["hidden_2_fw", "hidden_2_bw"])
  assert_equal(len(net_json), 7)


if __name__ == "__main__":
  better_exchook.install()
  if len(sys.argv) <= 1:
    for k, v in sorted(globals().items()):
      if k.startswith("test_"):
        print("-" * 40)
        print("Executing: %s" % k)
        try:
          v()
        except unittest.SkipTest as exc:
          print("SkipTest:", exc)
        print("-" * 40)
    print("Finished all tests.")
  else:
    assert len(sys.argv) >= 2
    for arg in sys.argv[1:]:
      print("Executing: %s" % arg)
      if arg in globals():
        globals()[arg]()  # assume function and execute
      else:
        eval(arg)  # assume Python code and execute
