#!/usr/bin/env python3

"""
This script imports a t2t model into Returnn and compares the activations.
It currently assumes a specific Returnn network topology with specific layer names.
Example Returnn network topology:

.. code-block:: python

    network = {

    }

"""

from __future__ import print_function

import os
import sys

import tensorflow as tf

my_dir = os.path.dirname(os.path.abspath(__file__))
returnn_dir = os.path.dirname(my_dir)
sys.path.insert(0, returnn_dir)

import Util

import better_exchook
import rnn
from TFNetwork import TFNetwork
from TFNetworkLayer import SourceLayer, LayerBase, LinearLayer




from tensor2tensor.bin import t2t_trainer
#from tensor2tensor.data_generators import problem  # pylint: disable=unused-import
from tensor2tensor.data_generators import text_encoder
#from tensor2tensor.utils import decoding
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir




'''
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
    name, rest = name.split("/", 1)
  else:
    rest = None
  if rest is None:
    return hierarchy[name]
  else:
    return get_in_hierarchy(rest, hierarchy[name])


'''


FLAGS_problem = "translate_tmp" # a self defined problem to handle own data
FLAGS_data_dir = "/work/smt3/schamper/sandbox/t2t-multi30k/datadir" # the data created by the self defined problem
FLAGS_model = "transformer"
FLAGS_hparams_set = "transformer_base_single_gpu"
FLAGS_hparams = "" # default seems to be empty
FLAGS_output_dir = "/work/smt3/schamper/sandbox/t2t-multi30k/out_model" # the trained t2tmodel
FLAGS_score_file = "/work/smt3/schamper/sandbox/t2t-multi30k/to_score" # The data we want to test on


def create_hparams():
  return trainer_lib.create_hparams(
      FLAGS_hparams_set,
      FLAGS_hparams,
      data_dir=os.path.expanduser(FLAGS_data_dir),
      problem_name=FLAGS_problem)

def score_file(filename):
  """Score each line in a file and return the scores."""
  # Prepare model.
  hparams = create_hparams()
  encoders = registry.problem(FLAGS_problem).feature_encoders(FLAGS_data_dir)
  has_inputs = "inputs" in encoders

  # Prepare features for feeding into the model.
  if has_inputs:
    inputs_ph = tf.placeholder(dtype=tf.int32)  # Just length dimension.
    batch_inputs = tf.reshape(inputs_ph, [1, -1, 1, 1])  # Make it 4D.
  targets_ph = tf.placeholder(dtype=tf.int32)  # Just length dimension.
  batch_targets = tf.reshape(targets_ph, [1, -1, 1, 1])  # Make it 4D.
  features = {
      "inputs": batch_inputs,
      "targets": batch_targets,
  } if has_inputs else {"targets": batch_targets}

  # Prepare the model and the graph when model runs on features.
  model = registry.model(FLAGS_model)(hparams, tf.estimator.ModeKeys.EVAL)
  _, losses = model(features)
  saver = tf.train.Saver()

  with tf.Session() as sess:
    # Load weights from checkpoint.
    ckpts = tf.train.get_checkpoint_state(FLAGS_output_dir)
    ckpt = ckpts.model_checkpoint_path
    saver.restore(sess, ckpt)
    # Run on each line.
    results = []
    for line in open(filename):
      tab_split = line.split("\t")
      if len(tab_split) > 2:
        raise ValueError("Each line must have at most one tab separator.")
      if len(tab_split) == 1:
        targets = tab_split[0].strip()
      else:
        targets = tab_split[1].strip()
        inputs = tab_split[0].strip()
      # Run encoders and append EOS symbol.
      targets_numpy = encoders["targets"].encode(
          targets) + [text_encoder.EOS_ID]
      if has_inputs:
        inputs_numpy = encoders["inputs"].encode(inputs) + [text_encoder.EOS_ID]
      # Prepare the feed.
      feed = {
          inputs_ph: inputs_numpy,
          targets_ph: targets_numpy
      } if has_inputs else {targets_ph: targets_numpy}
      # Get the score.
      np_loss = sess.run(losses["training"], feed)
      results.append(np_loss)
  return results




def main():

  print("#####################################################")
  print("Loading returnn config")

  rnn.init(
    commandLineOptions=['/work/schamper/sandbox/returnn-transformer/2-layer-trafo_posemb_const/config.py'], #sys.argv[1:],
    config_updates={
      "task": "nop", "log": None, "device": "cpu",
      "allow_random_model_init": True,
      "debug_add_check_numerics_on_output": False},
    extra_greeting="Import t2t model.")
  assert Util.BackendEngine.is_tensorflow_selected()
  config = rnn.config

  '''
  tf.logging.set_verbosity(20)
  trainer_lib.set_random_seed(None)
  usr_dir.import_usr_dir(None)
  '''


  print("#####################################################")
  print("Loading t2t model + scoring")
  results = score_file(FLAGS_score_file)
  print(results)


if __name__ == "__main__":
  better_exchook.install()
  main()

