

from nose.tools import assert_equal, assert_is_instance, assert_in, assert_true, assert_false
from Config import Config
from StringIO import StringIO
from Network import LayerNetwork
import h5py
import tempfile
import os


config_enc_dec1_json = """
{
"num_inputs": 5,
"num_outputs": 3,
"network": {
"proto_fw_1" : { "class" : "rec", "unit" : "lstmp", "n_out" : 7, "dropout" : 0.3, "direction" : 1 },
"proto_bw_1" : { "class" : "rec", "unit" : "lstmp", "n_out" : 7, "dropout" : 0.3, "direction" : -1 },

"proto_fw_2" : { "class" : "rec", "unit" : "lstmp", "n_out" : 7, "dropout" : 0.3, "direction" : 1, "from" : [ "proto_fw_1", "proto_bw_1" ] },
"proto_bw_2" : { "class" : "rec", "unit" : "lstmp", "n_out" : 7, "dropout" : 0.3, "direction" : -1, "from" : [ "proto_fw_1", "proto_bw_1" ] },

"proto_fw_3" : { "class" : "rec", "unit" : "lstmp", "n_out" : 7, "dropout" : 0.3, "direction" : 1, "from" : [ "proto_fw_2", "proto_bw_2" ] },
"proto_bw_3" : { "class" : "rec", "unit" : "lstmp", "n_out" : 7, "dropout" : 0.3, "direction" : -1, "from" : [ "proto_fw_2", "proto_bw_2" ] },

"encoder_fw" : { "class" : "rec", "unit" : "lstmp", "n_out" : 7, "dropout" : 0.3, "direction" : 1, "from" : [ "proto_fw_3", "proto_bw_3" ]  },
"encoder_bw" : { "class" : "rec", "unit" : "lstmp", "n_out" : 7, "dropout" : 0.3, "direction" : -1, "from" : [ "proto_fw_3", "proto_bw_3" ]  },
"decoder_fw" : { "class" : "rec", "unit" : "lstme", "n_out" : 7, "dropconnect" : 0.0, "direction" : 1, "attention" : "default", "attention_beam" : 0, "lm" : false, "encoder" : [ "encoder_bw" ], "from" : ["null"] },
"decoder_bw" : { "class" : "rec", "unit" : "lstme", "n_out" : 7, "dropconnect" : 0.0, "direction" : -1, "attention" : "default", "attention_beam" : 0, "lm" : false, "encoder" : [ "encoder_fw" ], "from" : ["null"] },

"output" : { "class" : "softmax", "from" : ["decoder_fw", "decoder_bw"] }
}
}
"""

def test_enc_dec1_init():
  config = Config()
  config.load_file(StringIO(config_enc_dec1_json))

  network_json = LayerNetwork.json_from_config(config)
  assert_true(network_json)
  network = LayerNetwork.from_json_and_config(network_json, config)
  assert_true(network)


def test_enc_dec1_hdf():
  filename = tempfile.mktemp(prefix="crnn-model-test")
  model = h5py.File(filename, "w")

  config = Config()
  config.load_file(StringIO(config_enc_dec1_json))
  network_json = LayerNetwork.json_from_config(config)
  assert_true(network_json)
  network = LayerNetwork.from_json_and_config(network_json, config)
  assert_true(network)

  network.save_hdf(model, epoch=42)
  model.close()

  loaded_model = h5py.File(filename, "r")
  loaded_net = LayerNetwork.from_hdf_model_topology(loaded_model)
  assert_true(loaded_net)
  assert_equal(sorted(network.hidden.keys()), sorted(loaded_net.hidden.keys()))
  assert_equal(sorted(network.y.keys()), sorted(loaded_net.y.keys()))
  assert_equal(sorted(network.j.keys()), sorted(loaded_net.j.keys()))

  os.remove(filename)
