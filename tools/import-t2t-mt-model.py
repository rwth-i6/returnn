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


#T2T_MODEL_DIR = "/work/smt3/schamper/sandbox/t2t-multi30k_2layers"
T2T_MODEL_DIR = "/work/smt3/schamper/sandbox/out_model_2layer_512"
FLAGS_data_dir = T2T_MODEL_DIR + "/datadir" # the data created by the self defined problem
FLAGS_output_dir = T2T_MODEL_DIR + "/out_model" # the trained t2tmodel
FLAGS_score_file = T2T_MODEL_DIR + "/to_score" # The data we want to test on

FLAGS_problem = "translate_tmp" # a self defined problem to handle own data
FLAGS_model = "transformer"
FLAGS_hparams_set = "transformer_base_single_gpu"
#FLAGS_hparams = "num_hidden_layers=2" # default is empty
FLAGS_hparams = "num_hidden_layers=2,hidden_size=256,shared_embedding_and_softmax_weights=False,symbol_modality_num_shards=1,conv_first_kernel=0,use_target_space_embedding=False,filter_size=512"



def create_t2t_hparams():
  return trainer_lib.create_hparams(
      FLAGS_hparams_set,
      FLAGS_hparams,
      data_dir=os.path.expanduser(FLAGS_data_dir),
      problem_name=FLAGS_problem)


def t2t_score_file(filename):
  """
  Score each line in a file and return the scores.

  :param str filename: T2T checkpoint
  """
  # Prepare model.
  hparams = create_t2t_hparams()
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

  sess = tf.Session()
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
    return sess, tvars



FFDim = 512
EncKeyTotalDim = 256
AttNumHeads = 8
EncKeyPerHeadDim = EncKeyTotalDim // AttNumHeads
EncValueTotalDim = 256
EncValuePerHeadDim = EncValueTotalDim // AttNumHeads


# consider: /work/smt3/bahar/debug/returnn/t2t-test/02.07-2018-test/returnn/config-new.py

def add_trafo_enc_layer(d, inp, output):
  d[output + '_self_att_laynorm'] = {"class": "layer_norm", "from": [inp]}
  d[output + '_self_att_'] = {"class": "self_attention", "num_heads": AttNumHeads, "total_key_dim": EncKeyTotalDim,
                             "n_out": EncValueTotalDim, "from": [output + '_self_att_laynorm'], "attention_left_only": False}
  d[output + '_self_att_lin'] = {"class": "linear", "activation": None, "with_bias": False,
                                   "from": [output + '_self_att_'], "n_out": EncValueTotalDim}
  d[output + '_self_att_drop'] = {"class": "dropout", "from": [output + '_self_att_lin'], "dropout": 0.1}
  d[output + '_self_att_out'] = {"class": "combine", "kind": "add", "from": [inp, output + '_self_att_drop'],
                                 "n_out": EncValueTotalDim}
  #####
  d[output + '_ff_laynorm'] = {"class": "layer_norm", "from": [output + '_self_att_out']}
  d[output + '_ff_conv1'] = {"class": "linear", "activation": "relu", "with_bias": True, "from": [output + '_ff_laynorm'],
                            "n_out": FFDim}
  d[output + '_ff_conv2'] = {"class": "linear", "activation": None, "with_bias": True, "from": [output + '_ff_conv1'],
                            "n_out": EncValueTotalDim}
  d[output + '_ff_drop'] = {"class": "dropout", "from": [output + '_ff_conv2'], "dropout": 0.1}
  d[output + '_ff_out'] = {"class": "combine", "kind": "add", "from": [output + '_self_att_out', output + '_ff_drop'],
                           "n_out": EncValueTotalDim}
  d[output] = {"class": "copy", "from": [output + '_ff_out']}


def add_trafo_dec_layer(db, d, inp, output):
  d[output + '_self_att_laynorm'] = {"class": "layer_norm", "from": [inp]}
  d[output + '_self_att_'] = {"class": "self_attention", "num_heads": AttNumHeads, "total_key_dim": EncKeyTotalDim,
                               "n_out": EncValueTotalDim, "from": [output + '_self_att_laynorm'], "attention_left_only": True}
  d[output + '_self_att_lin'] = {"class": "linear", "activation": None, "with_bias": False,
                                 "from": [output + '_self_att_'], "n_out": EncValueTotalDim}
  d[output + '_self_att_drop'] = {"class": "dropout", "from": [output + '_self_att_lin'], "dropout": 0.1}
  d[output + '_self_att_out'] = {"class": "combine", "kind": "add", "from": [inp, output + '_self_att_drop'],
                                 "n_out": EncValueTotalDim}
  #####
  d[output + '_att_laynorm'] = {"class": "layer_norm", "from": [output + '_self_att_out']}
  d[output + '_att_query0'] = {"class": "linear", "activation": None, "with_bias": False, "from": [output + '_att_laynorm'],
                               "n_out": EncValueTotalDim}
  d[output + '_att_query'] = {"class": "split_dims", "axis": "F", "dims": (AttNumHeads, EncKeyPerHeadDim),
                              "from": [output + '_att_query0']}  # (B, H, D/H)
  db[output + '_att_key0'] = {"class": "linear", "activation": None, "with_bias": False, "from": ["encoder"],
                              "n_out": EncKeyTotalDim}  # (B, enc-T, D)
  db[output + '_att_value0'] = {"class": "linear", "activation": None, "with_bias": False, "from": ["encoder"],
                                "n_out": EncValueTotalDim}
  db[output + '_att_key'] = {"class": "split_dims", "axis": "F", "dims": (AttNumHeads, EncKeyPerHeadDim),
                             "from": [output + '_att_key0']}  # (B, enc-T, H, D/H)
  db[output + '_att_value'] = {"class": "split_dims", "axis": "F", "dims": (AttNumHeads, EncValuePerHeadDim),
                               "from": [output + '_att_value0']}  # (B, enc-T, H, D'/H)
  d[output + '_att_energy'] = {"class": "dot", "red1": -1, "red2": -1, "var1": "T", "var2": "T?",
                           "from": ['base:' + output + '_att_key', output + '_att_query']}  # (B, H, enc-T, 1)
  d[output + '_att_weights'] = {"class": "softmax_over_spatial", "from": [output + '_att_energy'],
                                "energy_factor": EncKeyPerHeadDim ** -0.5}  # (B, enc-T, H, 1)
  d[output + '_att0'] = {"class": "generic_attention", "weights": output + '_att_weights',
                         "base": 'base:' + output + '_att_value'}  # (B, H, V)
  d[output + '_att_'] = {"class": "merge_dims", "axes": "static", "from": [output + '_att0']}  # (B, H*V) except_batch
  d[output + '_att_lin'] = {"class": "linear", "activation": None, "with_bias": False, "from": [output + '_att_'],
                          "n_out": EncValueTotalDim}
  d[output + '_att_drop'] = {"class": "dropout", "from": [output + '_att_lin'], "dropout": 0.1}
  d[output + '_att_out'] = {"class": "combine", "kind": "add", "from": [output + '_self_att_out', output + '_att_drop'],
                         "n_out": EncValueTotalDim}
  #####
  d[output + '_ff_laynorm'] = {"class": "layer_norm", "from": [output + '_att_out']}
  d[output + '_ff_conv1'] = {"class": "linear", "activation": "relu", "with_bias": True, "from": [output + '_ff_laynorm'],
                            "n_out": FFDim}
  d[output + '_ff_conv2'] = {"class": "linear", "activation": None, "with_bias": True, "from": [output + '_ff_conv1'],
                            "n_out": EncValueTotalDim}
  d[output + '_ff_drop'] = {"class": "dropout", "from": [output + '_ff_conv2'], "dropout": 0.1}
  d[output + '_ff_out'] = {"class": "combine", "kind": "add", "from": [output + '_att_out', output + '_ff_drop'],
                           "n_out": EncValueTotalDim}
  d[output] = {"class": "copy", "from": [output + '_ff_out']}



# network
# (also defined by num_inputs & num_outputs)
returnn_network = {
  "source_embed_raw": {"class": "linear", "activation": None, "with_bias": False, "n_out": EncValueTotalDim},
  "source_embed_with_pos": {"class": "positional_encoding", "add_to_input": True, "from": ["source_embed_raw"],  "dropout": 0.1},
  "source_embed": {"class": "dropout", "from": ["source_embed_with_pos"], "dropout": 0.1},

  ## trafo layer added later

  "encoder": {"class": "layer_norm", "from": ["enc_N"]},

  "output": {"class": "rec", "from": [], "unit": {
    'output': {'class': 'choice', 'target': 'classes', 'beam_size': 12, 'from': ["output_prob"],
               "initial_output": 0},
    "end": {"class": "compare", "from": ["output"], "value": 0},
    'target_embed_raw': {'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'],
                     "n_out": EncValueTotalDim, "initial_output": 0},  # feedback_input
    "target_embed_with_pos": {"class": "positional_encoding", "add_to_input": True, "from": ["target_embed_raw"]},
    "target_embed": {"class": "dropout", "from": ["target_embed_with_pos"], "dropout": 0.1},

    ## trafo layer added later

    # ToDo: Add last linear layer???

    "decoder": {"class": "layer_norm", "from": ["dec_N"]},

    "output_prob": {
      "class": "softmax", "from": ["decoder"], "dropout": 0.0,
      "target": "classes", "loss": "ce", "loss_opts": {"label_smoothing": 0.1},
      "with_bias": False
    }

  }, "target": "classes", "max_seq_len": "max_len_from('base:encoder') * 3"},

}

add_trafo_enc_layer(returnn_network, "source_embed", "enc_1")
add_trafo_enc_layer(returnn_network, "enc_1", "enc_N")
add_trafo_dec_layer(returnn_network, returnn_network["output"]["unit"], "prev:target_embed", "dec_1")
add_trafo_dec_layer(returnn_network, returnn_network["output"]["unit"], "dec_1", "dec_N")


num_outputs = {'classes': [6115, 1], 'data': [6115, 1]}
num_inputs = num_outputs["data"][0]




def main():
  print("#####################################################")
  print("Loading t2t model + scoring")
  t2t_sess, t2t_tvars = t2t_score_file(FLAGS_score_file)


  print("#####################################################")
  print("Loading returnn config")

  rnn.init(
    config_updates={
      "use_tensorflow": True,
      "num_outputs": num_outputs,
      "num_inputs": num_inputs,
      "task": "nop", "log": None, "device": "cpu",
      "network": returnn_network,
      "debug_print_layer_output_template": True,
      "debug_add_check_numerics_on_output": False},
    extra_greeting="Import t2t model.")
  assert Util.BackendEngine.is_tensorflow_selected()
  config = rnn.config

  rnn.engine.init_train_from_config(config=config)
  network = rnn.engine.network
  assert isinstance(network, TFNetwork)


  print("t2t network model params:")
  t2t_params = {} # type: dict[str,tf.Variable]
  t2t_total_num_params = 0
  for v in t2t_tvars:
    key = v.name[:-2]
    t2t_params[key] = v
    print("  %s: %s, %s" % (key, v.shape, v.dtype.base_dtype.name))
    t2t_total_num_params += numpy.prod(v.shape.as_list())
  print("t2t total num params: %i" % t2t_total_num_params)




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

  print("Loading t2t params into our network:")
  for ret_var in our_params:
    if ret_var.name in ret_to_t2t:
      t2t_var = t2t_params[ret_to_t2t[ret_var.name]]
      ret_var.load(t2t_var.eval(t2t_sess), rnn.engine.tf_session)
    else:
      print("skpipped over %s" % ret_var.name)

  ipdb.set_trace()



# maps names of trainable para
ret_to_t2t = {
  'output/rec/target_embed_raw/W' : 'target_emb/weights_0',
  'source_embed_raw/W' : 'input_emb/weights_0',
  'encoder/scale' : 'encoder/layer_prepostprocess/layer_norm/layer_norm_scale',
  'encoder/bias' : 'encoder/layer_prepostprocess/layer_norm/layer_norm_bias',
  'output/rec/decoder/scale' : 'decoder/layer_prepostprocess/layer_norm/layer_norm_scale',
  'output/rec/decoder/bias' : 'decoder/layer_prepostprocess/layer_norm/layer_norm_bias',
  'output/rec/output_prob/W' : 'softmax/weights_0',


  'enc_1_self_att_laynorm/scale' : 'encoder/layer_0/self_attention/layer_prepostprocess/layer_norm/layer_norm_scale',
  'enc_1_self_att_laynorm/bias' : 'encoder/layer_0/self_attention/layer_prepostprocess/layer_norm/layer_norm_bias',
  'enc_1_self_att_/QKV' : ('encoder/layer_0/self_attention/multihead_attention/q/kernel',
                           'encoder/layer_0/self_attention/multihead_attention/k/kernel',
                           'encoder/layer_0/self_attention/multihead_attention/v/kernel'),
  'enc_1_self_att_lin/W' : 'encoder/layer_0/self_attention/multihead_attention/output_transform/kernel',
  'enc_1_ff_laynorm/scale' : 'encoder/layer_0/ffn/layer_prepostprocess/layer_norm/layer_norm_scale',
  'enc_1_ff_laynorm/bias' : 'encoder/layer_0/ffn/layer_prepostprocess/layer_norm/layer_norm_bias',
  'enc_1_ff_conv1/W' : 'encoder/layer_0/ffn/conv1/kernel',
  'enc_1_ff_conv1/b' : 'encoder/layer_0/ffn/conv1/bias',
  'enc_1_ff_conv2/W' : 'encoder/layer_0/ffn/conv2/kernel',
  'enc_1_ff_conv2/b' : 'encoder/layer_0/ffn/conv2/bias',

  'enc_N_self_att_laynorm/scale' : 'encoder/layer_1/self_attention/layer_prepostprocess/layer_norm/layer_norm_scale',
  'enc_N_self_att_laynorm/bias' : 'encoder/layer_1/self_attention/layer_prepostprocess/layer_norm/layer_norm_bias',
  'enc_N_self_att_/QKV' : ('encoder/layer_1/self_attention/multihead_attention/q/kernel',
                           'encoder/layer_1/self_attention/multihead_attention/k/kernel',
                           'encoder/layer_1/self_attention/multihead_attention/v/kernel'),
  'enc_N_self_att_lin/W' : 'encoder/layer_1/self_attention/multihead_attention/output_transform/kernel',
  'enc_N_ff_laynorm/scale' : 'encoder/layer_1/ffn/layer_prepostprocess/layer_norm/layer_norm_scale',
  'enc_N_ff_laynorm/bias' : 'encoder/layer_1/ffn/layer_prepostprocess/layer_norm/layer_norm_bias',
  'enc_N_ff_conv1/W' : 'encoder/layer_1/ffn/conv1/kernel',
  'enc_N_ff_conv1/b' : 'encoder/layer_1/ffn/conv1/bias',
  'enc_N_ff_conv2/W' : 'encoder/layer_1/ffn/conv2/kernel',
  'enc_N_ff_conv2/b' : 'encoder/layer_1/ffn/conv2/bias',



  'output/rec/dec_1_self_att_laynorm/scale' : 'decoder/layer_0/self_attention/layer_prepostprocess/layer_norm/layer_norm_scale',
  'output/rec/dec_1_self_att_laynorm/bias' : 'decoder/layer_0/self_attention/layer_prepostprocess/layer_norm/layer_norm_bias:',
  'output/rec/dec_1_self_att_/QKV' : '(decoder/layer_0/self_attention/multihead_attention/q/kernel,'
       'decoder/layer_0/self_attention/multihead_attention/k/kernel,'
       'decoder/layer_0/self_attention/multihead_attention/v/kernel)',
  'output/rec/dec_1_self_att_lin/W' : 'decoder/layer_0/self_attention/multihead_attention/output_transform/kernel',
  'output/rec/dec_1_att_laynorm/scale' : 'decoder/layer_0/encdec_attention/layer_prepostprocess/layer_norm/layer_norm_scale',
  'output/rec/dec_1_att_laynorm/bias' : 'decoder/layer_0/encdec_attention/layer_prepostprocess/layer_norm/layer_norm_bias',
  'output/rec/dec_1_att_query0/W' : 'decoder/layer_0/encdec_attention/multihead_attention/q/kernel',
  'dec_1_att_key0/W' : 'decoder/layer_0/encdec_attention/multihead_attention/k/kernel',
  'dec_1_att_value0/W' : 'decoder/layer_0/encdec_attention/multihead_attention/v/kernel',
  'output/rec/dec_1_att_lin/W' : 'decoder/layer_0/encdec_attention/multihead_attention/output_transform/kernel',
  'output/rec/dec_1_ff_laynorm/scale' : 'decoder/layer_0/ffn/layer_prepostprocess/layer_norm/layer_norm_scale',
  'output/rec/dec_1_ff_laynorm/bias' : 'decoder/layer_0/ffn/layer_prepostprocess/layer_norm/layer_norm_bias',
  'output/rec/dec_1_ff_conv1/W' : 'decoder/layer_0/ffn/conv1/kernel',
  'output/rec/dec_1_ff_conv1/b' : 'decoder/layer_0/ffn/conv1/bias',
  'output/rec/dec_1_ff_conv2/W' : 'decoder/layer_0/ffn/conv2/kernel',
  'output/rec/dec_1_ff_conv2/b' : 'decoder/layer_0/ffn/conv2/bias',

  'output/rec/dec_N_self_att_laynorm/scale' : 'decoder/layer_1/self_attention/layer_prepostprocess/layer_norm/layer_norm_scale',
  'output/rec/dec_N_self_att_laynorm/bias' : 'decoder/layer_1/self_attention/layer_prepostprocess/layer_norm/layer_norm_bias:',
  'output/rec/dec_N_self_att_/QKV' : '(decoder/layer_1/self_attention/multihead_attention/q/kernel,'
       'decoder/layer_1/self_attention/multihead_attention/k/kernel,'
       'decoder/layer_1/self_attention/multihead_attention/v/kernel)',
  'output/rec/dec_N_self_att_lin/W' : 'decoder/layer_1/self_attention/multihead_attention/output_transform/kernel',
  'output/rec/dec_N_att_laynorm/scale' : 'decoder/layer_1/encdec_attention/layer_prepostprocess/layer_norm/layer_norm_scale',
  'output/rec/dec_N_att_laynorm/bias' : 'decoder/layer_1/encdec_attention/layer_prepostprocess/layer_norm/layer_norm_bias',
  'output/rec/dec_N_att_query0/W' : 'decoder/layer_1/encdec_attention/multihead_attention/q/kernel',
  'dec_N_att_key0/W' : 'decoder/layer_1/encdec_attention/multihead_attention/k/kernel',
  'dec_N_att_value0/W' : 'decoder/layer_1/encdec_attention/multihead_attention/v/kernel',
  'output/rec/dec_N_att_lin/W' : 'decoder/layer_1/encdec_attention/multihead_attention/output_transform/kernel',
  'output/rec/dec_N_ff_laynorm/scale' : 'decoder/layer_1/ffn/layer_prepostprocess/layer_norm/layer_norm_scale',
  'output/rec/dec_N_ff_laynorm/bias' : 'decoder/layer_1/ffn/layer_prepostprocess/layer_norm/layer_norm_bias',
  'output/rec/dec_N_ff_conv1/W' : 'decoder/layer_1/ffn/conv1/kernel',
  'output/rec/dec_N_ff_conv1/b' : 'decoder/layer_1/ffn/conv1/bias',
  'output/rec/dec_N_ff_conv2/W' : 'decoder/layer_1/ffn/conv2/kernel',
  'output/rec/dec_N_ff_conv2/b' : 'decoder/layer_1/ffn/conv2/bias',
}












if __name__ == "__main__":
  better_exchook.install()
  main()
