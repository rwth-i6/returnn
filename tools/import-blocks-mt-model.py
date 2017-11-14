#!/usr/bin/env python3

from __future__ import print_function

import os
import sys
import numpy
import re
from pprint import pprint
from nose.tools import assert_equal, assert_is_instance
import tensorflow as tf

my_dir = os.path.dirname(os.path.abspath(__file__))
returnn_dir = os.path.dirname(my_dir)
sys.path.append(returnn_dir)

import better_exchook
import rnn
import Util
from TFNetwork import TFNetwork
from TFNetworkLayer import SourceLayer, LayerBase, LinearLayer


def get_network():
  """
  :rtype: TFNetwork
  """
  return rnn.engine.network


def get_input_layers():
  """
  :rtype: list[LayerBase]
  """
  ls = []
  for layer in get_network().layers.values():
    if len(layer.sources) != 1:
      continue
    if isinstance(layer.sources[0], SourceLayer):
      ls.append(layer)
  return ls


def find_our_input_embed_layer():
  """
  :rtype: LinearLayer
  """
  input_layers = get_input_layers()
  assert len(input_layers) == 1
  layer = input_layers[0]
  assert isinstance(layer, LinearLayer)
  return layer


def get_in_hierarchy(name, hierarchy):
  """
  :param str name: e.g. "decoder/sequencegenerator"
  :param dict[str,dict[str]] hierarchy: nested hierarchy
  :rtype: dict[str,dict[str]]
  """
  if "/" in name:
    name, rest = name.split("/", 2)
  else:
    rest = None
  if rest is None:
    return hierarchy[name]
  else:
    return get_in_hierarchy(rest, hierarchy[name])


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
  blocks_used_params = set()  # type: set[str]
  our_loaded_params = set()  # type: set[str]

  def load(our_var, blocks_param_name):
    assert isinstance(our_var, tf.Variable)
    assert isinstance(blocks_param_name, str)
    assert blocks_param_name in blocks_params
    blocks_used_params.add(blocks_param_name)
    our_loaded_params.add(our_var.name[:-2])
    our_var.load(blocks_params[blocks_param_name], session=rnn.engine.tf_session)

  enc_name = "bidirectionalencoder"
  enc_embed_name = "EncoderLookUp0.W"
  assert enc_name in blocks_params_hierarchy
  assert enc_embed_name in blocks_params_hierarchy[enc_name]  # input embedding
  num_encoder_layers = max([
    int(re.match(".*([0-9]+)", s).group(1))
    for s in blocks_params_hierarchy[enc_name]
    if s.startswith("EncoderBidirectionalLSTM")])
  blocks_input_dim, blocks_input_embed_dim = blocks_params["%s/%s" % (enc_name, enc_embed_name)].shape
  print("Blocks input dim: %i, embed dim: %i" % (blocks_input_dim, blocks_input_embed_dim))
  print("Blocks num encoder layers: %i" % num_encoder_layers)
  expected_enc_entries = (
    ["EncoderLookUp0.W"] +
    ["EncoderBidirectionalLSTM%i" % i for i in range(1, num_encoder_layers + 1)])
  assert_equal(set(expected_enc_entries), set(blocks_params_hierarchy[enc_name].keys()))

  our_input_layer = find_our_input_embed_layer()
  assert our_input_layer.input_data.dim == blocks_input_dim
  assert our_input_layer.output.dim == blocks_input_embed_dim
  assert not our_input_layer.with_bias
  load(our_input_layer.params["W"], "%s/%s" % (enc_name, enc_embed_name))

  dec_name = "decoder/sequencegenerator"
  dec_hierarchy_base = get_in_hierarchy(dec_name, blocks_params_hierarchy)
  assert_equal(set(dec_hierarchy_base.keys()), {"att_trans", "readout"})

  # TODO enc lstm layers... mostly straight forward

  print("Finished importing.")


if __name__ == "__main__":
  better_exchook.install()
  main()
