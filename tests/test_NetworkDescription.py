

from nose.tools import assert_equal, assert_is_instance, assert_in
from NetworkDescription import LayerNetworkDescription
from Config import Config


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
  assert_equal(num_outputs["classes"], [n_out, 1])


config1_dict = {
  "num_inputs": 5,
  "num_outputs": 10,
  "hidden_size": (7, 8,),
  "hidden_type": "forward",
  "activation": "relu",
}


def test_config1():
  config = Config()
  config.update(config1_dict)
  desc = LayerNetworkDescription.from_config(config)
  assert_is_instance(desc.hidden_info, list)
  assert_equal(len(desc.hidden_info), len(config1_dict["hidden_size"]))
  assert_equal(desc.num_inputs, config1_dict["num_inputs"])
