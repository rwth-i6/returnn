#!/usr/bin/env python3

"""
This script imports a t2t model into Returnn and compares the activations.
It currently assumes a specific Returnn network topology with specific layer names.
"""

from __future__ import print_function

import os
import sys
from pprint import pprint
import tensorflow as tf

import numpy

my_dir = os.path.dirname(os.path.abspath(__file__))
returnn_dir = os.path.dirname(my_dir)
sys.path.insert(0, returnn_dir)

import Util

import better_exchook
import rnn
from TFNetwork import TFNetwork
from TFNetworkLayer import SourceLayer, LayerBase, LinearLayer



import tensor2tensor
import tensor2tensor.models
from tensor2tensor.bin import t2t_trainer
#from tensor2tensor.data_generators import problem  # pylint: disable=unused-import
from tensor2tensor.data_generators import text_encoder
#from tensor2tensor.utils import decoding
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir


import ipdb


T2T_MODEL_DIR = "/work/smt3/schamper/sandbox/t2t-multi30k_2layers"
FLAGS_data_dir = T2T_MODEL_DIR + "/datadir" # the data created by the self defined problem
FLAGS_output_dir = T2T_MODEL_DIR + "/out_model" # the trained t2tmodel
FLAGS_score_file = T2T_MODEL_DIR + "/to_score" # The data we want to test on

FLAGS_problem = "translate_tmp" # a self defined problem to handle own data
FLAGS_model = "transformer"
FLAGS_hparams_set = "transformer_base_single_gpu"
FLAGS_hparams = "num_hidden_layers=2" # default is empty



def create_hparams():
  return trainer_lib.create_hparams(
      FLAGS_hparams_set,
      FLAGS_hparams,
      data_dir=os.path.expanduser(FLAGS_data_dir),
      problem_name=FLAGS_problem)


def score_file(filename):
  """
  Score each line in a file and return the scores.

  :param str filename: T2T checkpoint
  """
  # Prepare model.
  hparams = create_hparams()
  encoders = registry.problem(FLAGS_problem).feature_encoders(FLAGS_data_dir)

  # Prepare features for feeding into the model.
  inputs_ph = tf.placeholder(dtype=tf.int32, shape=(None, None))  # Just length dimension.
  targets_ph = tf.placeholder(dtype=tf.int32, shape=(None, None))  # Just length dimension.

  features = {
      "inputs": inputs_ph,
      "targets": targets_ph,
  }

  # Prepare the model and the graph when model runs on features.
  model = registry.model(FLAGS_model)(hparams, tf.estimator.ModeKeys.EVAL)
  assert isinstance(model, tensor2tensor.models.transformer.Transformer)
  #       final_output: tensor of logits with shape [batch_size, O, P, body_output_size.
  #       losses: either single loss as a scalar, a list, a tensor (to be averaged)
  #               or a dictionary of losses.
  final_output, losses = model(features)
  assert isinstance(losses, dict)
  saver = tf.train.Saver()

  with tf.Session() as sess:
    # Load weights from checkpoint.
    ckpts = tf.train.get_checkpoint_state(FLAGS_output_dir)
    ckpt = ckpts.model_checkpoint_path
    saver.restore(sess, ckpt)

    print(tf.trainable_variables())

    # Run on each line.
    results = []
    for line in open(filename):
      tab_split = line.split("\t")
      if len(tab_split) > 2:
        raise ValueError("Each line must have at most one tab separator.")
      assert len(tab_split) == 2
      targets = tab_split[1].strip()
      inputs = tab_split[0].strip()
      # Run encoders and append EOS symbol.
      targets_numpy = encoders["targets"].encode(targets) + [text_encoder.EOS_ID]
      inputs_numpy = encoders["inputs"].encode(inputs) + [text_encoder.EOS_ID]
      print(inputs_numpy)
      # Prepare the feed.
      feed = {
          inputs_ph: [inputs_numpy],
          targets_ph: [targets_numpy]
      }

      np_res = sess.run({"losses": losses, "final_output": final_output}, feed_dict=feed)
      pprint(np_res)

      tvars = tf.trainable_variables()

      ipdb.set_trace()





def main():
  print("#####################################################")
  print("Loading t2t model + scoring")
  #score_file(FLAGS_score_file)

  print("#####################################################")
  print("Loading returnn config")

  rnn.init(
    commandLineOptions=['/work/schamper/sandbox/returnn-transformer/2-layer-trafo_posemb/config.py'], #sys.argv[1:],
    config_updates={
      "task": "nop", "log": None, "device": "cpu",
      #"allow_random_model_init": True,
      "debug_add_check_numerics_on_output": False},
    extra_greeting="Import t2t model.")
  assert Util.BackendEngine.is_tensorflow_selected()
  config = rnn.config

  rnn.engine.init_train_from_config(config=config)
  network = rnn.engine.network
  assert isinstance(network, TFNetwork)
  print("Our network model params:")
  our_params = {}  # type: dict[str,tf.Variable]
  our_total_num_params = 0
  for v in network.get_params_list():
    key = v.name[:-2]
    our_params[key] = v
    print("  %s: %s, %s" % (key, v.shape, v.dtype.base_dtype.name))
    our_total_num_params += numpy.prod(v.shape.as_list())
  print("Our total num params: %i" % our_total_num_params)

  ipdb.set_trace()




if __name__ == "__main__":
  better_exchook.install()
  main()






def im():
  from tensorflow.python import pywrap_tensorflow
  file_name = "/work/smt3/bahar/debug/returnn/t2t-test/t2t-multi30k/out_model/model.ckpt-7506"
  reader = pywrap_tensorflow.NewCheckpointReader(file_name)
  var_to_shape_map = reader.get_variable_to_shape_map()
  # print (var_to_shape_map) ## a dictionary with tensor names as key and tensor shape as values

  print(var_to_shape_map.values())
  for key in sorted(var_to_shape_map):
    print("tensor_name: ", key)  ## Tensor name
  print("tensor_shape: ", var_to_shape_map[key])  ## Tensor shape


def im2():
  from tensorflow.python import pywrap_tensorflow
  file_name = "/work/smt3/bahar/debug/returnn/t2t-test/t2t-multi30k/out_model/model.ckpt-7506"
  reader = pywrap_tensorflow.NewCheckpointReader(file_name)
  var_to_shape_map = reader.get_variable_to_shape_map()
  # print (var_to_shape_map) ## a dictionary with tensor names as key and tensor shape as values

  # print (var_to_shape_map.values())
  tot_params = 0
  for key in sorted(var_to_shape_map):
    if not key.startswith('training'):
      val = var_to_shape_map[key]
      print("tensor_name:{: <120}{}".format(key, str(val)))
      tot_params = tot_params + reader.get_tensor(key).size
  # Print total number of params
  print("Total number of parameters: {}".format(tot_params))


def sssd():
  tf.get_default_graph().get_tensor_by_name(
    "transformer/body/decoder/layer_prepostprocess/layer_norm/layer_norm_scale:0")

