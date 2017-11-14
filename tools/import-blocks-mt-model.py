#!/usr/bin/env python3

from __future__ import print_function

import os
import sys
import numpy
import re
from pprint import pprint
from nose.tools import assert_equal, assert_is_instance

my_dir = os.path.dirname(os.path.abspath(__file__))
returnn_dir = os.path.dirname(my_dir)
sys.path.append(returnn_dir)

import better_exchook
import rnn
import Util
from TFNetwork import TFNetwork


def main():
  rnn.init(
    commandLineOptions=sys.argv[1:],
    config_updates={
      "task": "nop", "log": None, "device": "cpu",
      "allow_random_model_init": True,
      "debug_add_check_numerics_on_output": False},
    extra_greeting="Import Blocks MT model.")
  assert Util.BackendEngine.is_tensorflow_selected()
  config = rnn.config

  # Load Blocks MT model params.
  if not config.has("blocks_mt_model"):
    print("Please provide the option blocks_mt_model.")
    sys.exit(1)
  blocks_mt_model_fn = config.value("blocks_mt_model", "")
  assert blocks_mt_model_fn
  assert os.path.exists(blocks_mt_model_fn)
  if os.path.isdir(blocks_mt_model_fn):
    blocks_mt_model_fn += "/params.npz"
    assert os.path.exists(blocks_mt_model_fn)
  blocks_mt_model = numpy.load(blocks_mt_model_fn)
  assert isinstance(blocks_mt_model, numpy.lib.npyio.NpzFile), "did not expect type %r in file %r" % (
    type(blocks_mt_model), blocks_mt_model_fn)
  print("Params found in Blocks model:")
  blocks_params = {}  # type: dict[str,numpy.ndarray]
  blocks_params_hierarchy = {}  # type: dict[str,dict[str]]
  blocks_total_num_params = 0
  for key in sorted(blocks_mt_model.keys()):
    value = blocks_mt_model[key]
    key = key.replace("-", "/")
    assert key[0] == "/"
    key = key[1:]
    blocks_params[key] = value
    print("  %s: %s, %s" % (key, value.shape, value.dtype))
    blocks_total_num_params += numpy.prod(value.shape)
    d = blocks_params_hierarchy
    for part in key.split("/"):
      d = d.setdefault(part, {})
  print("Blocks total num params: %i" % blocks_total_num_params)

  # Init our network structure.
  rnn.engine.use_dynamic_train_flag = True  # construct the net as in training
  rnn.engine.init_network_from_config()
  print("Our network model params:")
  our_params = {}  # type: dict[str,tf.Variable]
  our_total_num_params = 0
  for v in rnn.engine.network.get_params_list():
    key = v.name[:-2]
    our_params[key] = v
    print("  %s: %s, %s" % (key, v.shape, v.dtype.base_dtype.name))
    our_total_num_params += numpy.prod(v.shape.as_list())
  print("Our total num params: %i" % our_total_num_params)

  # Now matching...
  # bidirectionalencoder/EncoderLookUp0.W -> input embedding
  enc_name = "bidirectionalencoder"
  assert enc_name in blocks_params_hierarchy
  assert "EncoderLookUp0.W" in blocks_params_hierarchy[enc_name]
  num_encoder_layers = max([
    int(re.match(".*([0-9]+)", s).group(1))
    for s in blocks_params_hierarchy[enc_name]
    if s.startswith("EncoderBidirectionalLSTM")])
  print("Blocks num encoder layers: %i" % num_encoder_layers)
  expected_enc_entries = (
    ["EncoderLookUp0.W"] +
    ["EncoderBidirectionalLSTM%i" % i for i in range(1, num_encoder_layers + 1)])
  assert_equal(set(expected_enc_entries), set(blocks_params_hierarchy[enc_name].keys()))

  print("Finished importing.")


if __name__ == "__main__":
  better_exchook.install()
  main()
