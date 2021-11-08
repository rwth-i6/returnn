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
#from numpy.testing import assert_almost_equal

my_dir = os.path.dirname(os.path.abspath(__file__))
returnn_dir = os.path.dirname(my_dir)
sys.path.insert(0, returnn_dir)

import returnn.util.basic as util
from returnn.util import better_exchook
import returnn.__main__ as rnn
import returnn.tf.compat as tf_compat
from returnn.tf.network import TFNetwork
from returnn.tf.layers.basic import SourceLayer, LayerBase, LinearLayer



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

# model was trained with self defined problem 'translate_tmp' to handle own data paths
# but importing works also with compatible standard problem
FLAGS_problem = "translate_ende_wmt8k"
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
  inputs_ph = tf_compat.v1.placeholder(dtype=tf.int32, shape=(None, None))  # Just length dimension.
  targets_ph = tf_compat.v1.placeholder(dtype=tf.int32, shape=(None, None))  # Just length dimension.

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
  saver = tf_compat.v1.train.Saver()

  sess = tf_compat.v1.Session()
  # Load weights from checkpoint.
  ckpts = tf.train.get_checkpoint_state(FLAGS_output_dir)
  ckpt = ckpts.model_checkpoint_path
  saver.restore(sess, ckpt)

  # writer = tf.compat.v1.summary.FileWriter('logs', sess.graph)

  # writer.close()


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
    # Prepare the feed.
    feed = {
        inputs_ph: [inputs_numpy],
        targets_ph: [targets_numpy]
    }

    np_res = sess.run({"losses": losses, "final_output": final_output}, feed_dict=feed)
    pprint(np_res)

    tvars = tf_compat.v1.trainable_variables()

    print('t2t inputs_ph:', inputs_ph, inputs_numpy)
    print('t2t targets_ph:', targets_ph, targets_numpy)

    return sess, tvars, inputs_ph, targets_ph, losses



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
  d[output + '_att_energy'] = {"class": "dot", "red1": "static:-1", "red2": "static:-1", "var1": "T", "var2": "T?",
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
  "source_embed_weighted": {"class": "eval", "from": ["source_embed_raw"], "eval": "source(0) * (%i**0.5)" % EncValueTotalDim},
  "source_embed_with_pos": {"class": "positional_encoding", "add_to_input": True, "from": ["source_embed_weighted"],  "dropout": 0.1},
  "source_embed": {"class": "copy", "from": ["source_embed_with_pos"]},

  ## trafo layer added later

  "encoder": {"class": "layer_norm", "from": ["enc_N"]},

  "output": {"class": "rec", "from": [], "unit": {
    'output': {'class': 'choice', 'target': 'classes', 'beam_size': 12, 'from': ["output_prob"]},
    "end": {"class": "compare", "from": ["output"], "value": 0},
    'target_embed_raw': {'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'],
                     "n_out": EncValueTotalDim, "initial_output": 0},  # there seems to be no <s> in t2t, they seem to use just the zero vector
    "target_embed_weighted": {"class": "eval", "from": ["prev:target_embed_raw"], "eval": "source(0) * (%i**0.5)" % EncValueTotalDim},
    "target_embed_with_pos": {"class": "positional_encoding", "add_to_input": True, "from": ["target_embed_weighted"]},
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
add_trafo_dec_layer(returnn_network, returnn_network["output"]["unit"], "target_embed", "dec_1")
add_trafo_dec_layer(returnn_network, returnn_network["output"]["unit"], "dec_1", "dec_N")


num_outputs = {'classes': [6115, 1], 'data': [6115, 1]}
num_inputs = num_outputs["data"][0]




def main():
  print("#####################################################")
  print("Loading t2t model + scoring")
  t2t_sess, t2t_tvars, t2t_inputs_ph, t2t_targets_ph, t2t_losses = t2t_score_file(FLAGS_score_file)


  print("#####################################################")
  print("Loading returnn config")

  rnn.init(
    config_updates={
      "optimize_move_layers_out": True,
      "use_tensorflow": True,
      "num_outputs": num_outputs,
      "num_inputs": num_inputs,
      "task": "nop", "log": None, "device": "cpu",
      "network": returnn_network,
      "debug_print_layer_output_template": True,
      "debug_add_check_numerics_on_output": False},
    extra_greeting="Import t2t model.")
  assert util.BackendEngine.is_tensorflow_selected()
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



  print("Loading t2t params into our network:")
  for ret_var_name, ret_var in our_params.items():
    if ret_var_name in ret_to_t2t:
      t2t_var_names = ret_to_t2t[ret_var_name]
      # in return QKV params are concatenated into one tensor
      if isinstance(t2t_var_names, tuple):
        # params_np = numpy.concatenate([t2t_params[var_name].eval(t2t_sess) for var_name in t2t_var_names], axis=1) # Not enough...
        # More complex stacking necessary: head1 Q, head1 K, head1 V,   head2 Q, head2 K, head2 V, ...
        q_np = t2t_params[t2t_var_names[0]].eval(t2t_sess)
        k_np = t2t_params[t2t_var_names[1]].eval(t2t_sess)
        v_np = t2t_params[t2t_var_names[2]].eval(t2t_sess)
        qkv_dim_total = 2*EncKeyTotalDim+EncValueTotalDim
        params_np = numpy.empty((EncValueTotalDim, qkv_dim_total) ,dtype=q_np.dtype)
        qkv_dim_per_head = qkv_dim_total // AttNumHeads
        for i in range(EncKeyPerHeadDim):
          params_np[:, i::qkv_dim_per_head]                      = q_np[:, i::EncKeyPerHeadDim]
        for i in range(EncKeyPerHeadDim):
          params_np[:, i+EncKeyPerHeadDim::qkv_dim_per_head]     = k_np[:, i::EncKeyPerHeadDim]
        for i in range(EncValuePerHeadDim):
          params_np[:, i+2*EncKeyPerHeadDim::qkv_dim_per_head]   = v_np[:, i::EncValuePerHeadDim]
      else:
        t2t_var = t2t_params[t2t_var_names]
        params_np = t2t_var.eval(t2t_sess)
      if ret_var_name in ['output/rec/output_prob/W']: # ToDo: Something else to transpose?
        params_np = params_np.transpose()
      if ret_var_name in ["source_embed_raw/W", 'output/rec/target_embed_raw/W']:
        #params_np = params_np * (EncValueTotalDim**0.5) # ToDo: Only because of weight-tying?
        print("loading %s with * (EncValueTotalDim**0.5) or doing it in config" % ret_var.name)
      ret_var.load(params_np, rnn.engine.tf_session)
      print("loaded %s" % ret_var.name)
    else:
      print("skpipped over %s" % ret_var.name)


  ret_ph_train = rnn.engine.tf_session.graph.get_tensor_by_name("global_tensor_train_flag/train_flag:0")
  ret_ph_data = rnn.engine.tf_session.graph.get_tensor_by_name("extern_data/placeholders/data/data:0")
  ret_ph_data_dim =  rnn.engine.tf_session.graph.get_tensor_by_name("extern_data/placeholders/data/data_dim0_size:0")
  ret_ph_classes=  rnn.engine.tf_session.graph.get_tensor_by_name("extern_data/placeholders/classes/classes:0")
  ret_ph_classes_dim = rnn.engine.tf_session.graph.get_tensor_by_name("extern_data/placeholders/classes/classes_dim0_size:0")

  #ret_feed = {ret_ph_train: True, ret_ph_data: [[11, 78, 42, 670, 2415, 2, 134, 2, 61, 522, 2, 847, 2, 3353, 15, 33, 2534, 1], [3,6]], ret_ph_data_dim: [18, 2],
  #            ret_ph_classes: [[4, 60, 18, 46, 26, 2937, 520, 2, 1317, 2, 10, 642, 4, 639, 1], [2,5]], ret_ph_classes_dim:[14, 2]}

  #src = [[78, 1,0], [2, 134, 1]]; src_lens = [2,3]; trg = [[4, 60, 1], [639, 1, 0]]; trg_lens = [3,2]; ret_feed = {ret_ph_train: False, ret_ph_data: src, ret_ph_data_dim: src_lens, ret_ph_classes: trg, ret_ph_classes_dim: trg_lens}; t2t_feed = {t2t_inputs_ph: src, t2t_targets_ph: trg}

  src = [[2, 134, 1]]; src_lens = [3]; trg = [[4, 60, 1]]; trg_lens = [3]; ret_feed = {ret_ph_train: False, ret_ph_data: src, ret_ph_data_dim: src_lens, ret_ph_classes: trg, ret_ph_classes_dim: trg_lens}; t2t_feed = {t2t_inputs_ph: src, t2t_targets_ph: trg}


  compare_acts(network, t2t_sess, ret_feed, t2t_feed, act_ret_to_t2t)


  # filtered = [op for op in t2t_sess.graph.get_operations() if '/encoder/layer_0/self_attention' in op.name and op.type == 'MatMul']
  # filtered = [op for op in rnn.engine.tf_session.graph.get_operations() if 'enc_1_self_att_/' in op.name and op.type == 'MatMul']
  # for op in filtered: print(op.name)
#



  ipdb.set_trace()

def eval_ret_tensor(ret_lt_name, ret_feed):
  ret_act = rnn.engine.tf_session.graph.get_tensor_by_name(ret_lt_name)
  return rnn.engine.tf_session.run(ret_act, ret_feed)


def compare_acts(network, t2t_sess, ret_feed, t2t_feed, act_ret_to_t2t):
  for ret_lt_name, t2t_t_names in act_ret_to_t2t.items():
    ######################################################
    print(ret_lt_name, ':')
    transp = False
    if ':' in ret_lt_name: # activations are either extracted from graph (more fine-grained) or from a returnn layer
      ret_act = rnn.engine.tf_session.graph.get_tensor_by_name(ret_lt_name)
      transp = True
    else:
      ret_act = network.layers[ret_lt_name].output.get_placeholder_as_batch_major()
    ret_np = rnn.engine.tf_session.run(ret_act, ret_feed)
    print(ret_np.shape)
    if len(ret_np.shape) < 3:
      print(ret_np)
    else:
      if transp:
        ret_np = ret_np.transpose(1,0,2)
        print(ret_np.shape)
      print(ret_np[:,:,0:12])

    #####################################################
    print(t2t_t_names, ':')
    if isinstance(t2t_t_names, tuple):
      t2t_np = numpy.concatenate([t2t_sess.run(t2t_sess.graph.get_tensor_by_name(t2t_t_name), t2t_feed) for t2t_t_name in t2t_t_names], axis=1)
    else:
      t2t_np = t2t_sess.run(t2t_sess.graph.get_tensor_by_name(t2t_t_names), t2t_feed)
    print(t2t_np.shape)
    if len(ret_np.shape) < 3:
      print(t2t_np)
    else:
      print(t2t_np[:,:,0:12])

    print('allclose:', numpy.allclose(ret_np, t2t_np, rtol=1.e-3, atol=1.e-6,))

    if 'enc_1_self_att_/dot/MatMul:0' in ret_lt_name:
      print("Calculating attention for first head manually:")
      q = t2t_np[:, 0:32]
      q = q * ((256 / 8) ** (-0.5))
      k = t2t_np[:, 256 * 1 + 0:256 * 1 + 32]
      energy = numpy.dot(q, k.T)
      print(energy)
      print("after softmax")
      print(rnn.engine.tf_session.run(tf.nn.softmax(energy, axis=-1)))

    print("-----------------------------------------------------------------------------------------------------------")


act_ret_to_t2t = {
    'enc_1_self_att_laynorm' : 'transformer/parallel_0_4/transformer/transformer/body/encoder/layer_0/self_attention/layer_prepostprocess/layer_norm/add_1:0',
    'enc_1_self_att_out' : 'transformer/parallel_0_4/transformer/transformer/body/encoder/layer_0/self_attention/layer_postprocess/add:0',
    'enc_N_self_att_laynorm' : 'transformer/parallel_0_4/transformer/transformer/body/encoder/layer_1/self_attention/layer_prepostprocess/layer_norm/add_1:0',
    'enc_N_self_att_out' : 'transformer/parallel_0_4/transformer/transformer/body/encoder/layer_1/self_attention/layer_postprocess/add:0',
    'enc_1_self_att_/Softmax:0' : 'transformer/parallel_0_4/transformer/transformer/body/encoder/layer_0/self_attention/multihead_attention/dot_product_attention/Softmax:0',
    'encoder' : 'transformer/parallel_0_4/transformer/transformer/body/encoder/layer_prepostprocess/layer_norm/add_1:0',
    'output/rec/dec_1_self_att_laynorm/add:0' :  'transformer/parallel_0_4/transformer/transformer/body/decoder/layer_0/self_attention/layer_prepostprocess/layer_norm/add_1:0',
    'output/rec/dec_1_self_att_out/Add:0' : 'transformer/parallel_0_4/transformer/transformer/body/decoder/layer_0/self_attention/layer_postprocess/add:0',
    'output/rec/dec_N_self_att_out/Add:0' : 'transformer/parallel_0_4/transformer/transformer/body/decoder/layer_1/self_attention/layer_postprocess/add:0',
    'output/rec/output_prob/linear/dot/MatMul:0' : 'transformer/parallel_0_4/transformer/transformer/symbol_modality_6115_256_2/softmax/MatMul:0',
  }

#act_ret_to_t2t = { 'output/rec/dec_N_self_att_out/Add:0' : 'transformer/parallel_0_4/transformer/transformer/body/decoder/layer_1/self_attention/layer_postprocess/add:0', }
#act_ret_to_t2t = {'enc_1_self_att_/dot/MatMul:0' : tuple ('transformer/parallel_0_4/transformer/transformer/body/encoder/layer_0/self_attention/multihead_attention/%s/Tensordot/MatMul:0' % t for t in ['q', 'k', 'v']),}



# maps names of trainable params (returnn to t2t)
ret_to_t2t = {
  'output/rec/target_embed_raw/W' : 'transformer/symbol_modality_6115_256/target_emb/weights_0',
  'source_embed_raw/W' : 'transformer/symbol_modality_6115_256/input_emb/weights_0',
  'encoder/scale' : 'transformer/body/encoder/layer_prepostprocess/layer_norm/layer_norm_scale',
  'encoder/bias' : 'transformer/body/encoder/layer_prepostprocess/layer_norm/layer_norm_bias',
  'output/rec/decoder/scale' : 'transformer/body/decoder/layer_prepostprocess/layer_norm/layer_norm_scale',
  'output/rec/decoder/bias' : 'transformer/body/decoder/layer_prepostprocess/layer_norm/layer_norm_bias',
  'output/rec/output_prob/W' : 'transformer/symbol_modality_6115_256/softmax/weights_0', # has to be transposed


  'enc_1_self_att_laynorm/scale' : 'transformer/body/encoder/layer_0/self_attention/layer_prepostprocess/layer_norm/layer_norm_scale',
  'enc_1_self_att_laynorm/bias' : 'transformer/body/encoder/layer_0/self_attention/layer_prepostprocess/layer_norm/layer_norm_bias',
  'enc_1_self_att_/QKV' : ('transformer/body/encoder/layer_0/self_attention/multihead_attention/q/kernel',
                           'transformer/body/encoder/layer_0/self_attention/multihead_attention/k/kernel',
                           'transformer/body/encoder/layer_0/self_attention/multihead_attention/v/kernel'),
  'enc_1_self_att_lin/W' : 'transformer/body/encoder/layer_0/self_attention/multihead_attention/output_transform/kernel',
  'enc_1_ff_laynorm/scale' : 'transformer/body/encoder/layer_0/ffn/layer_prepostprocess/layer_norm/layer_norm_scale',
  'enc_1_ff_laynorm/bias' : 'transformer/body/encoder/layer_0/ffn/layer_prepostprocess/layer_norm/layer_norm_bias',
  'enc_1_ff_conv1/W' : 'transformer/body/encoder/layer_0/ffn/conv1/kernel',
  'enc_1_ff_conv1/b' : 'transformer/body/encoder/layer_0/ffn/conv1/bias',
  'enc_1_ff_conv2/W' : 'transformer/body/encoder/layer_0/ffn/conv2/kernel',
  'enc_1_ff_conv2/b' : 'transformer/body/encoder/layer_0/ffn/conv2/bias',

  'enc_N_self_att_laynorm/scale' : 'transformer/body/encoder/layer_1/self_attention/layer_prepostprocess/layer_norm/layer_norm_scale',
  'enc_N_self_att_laynorm/bias' : 'transformer/body/encoder/layer_1/self_attention/layer_prepostprocess/layer_norm/layer_norm_bias',
  'enc_N_self_att_/QKV' : ('transformer/body/encoder/layer_1/self_attention/multihead_attention/q/kernel',
                           'transformer/body/encoder/layer_1/self_attention/multihead_attention/k/kernel',
                           'transformer/body/encoder/layer_1/self_attention/multihead_attention/v/kernel'),
  'enc_N_self_att_lin/W' : 'transformer/body/encoder/layer_1/self_attention/multihead_attention/output_transform/kernel',
  'enc_N_ff_laynorm/scale' : 'transformer/body/encoder/layer_1/ffn/layer_prepostprocess/layer_norm/layer_norm_scale',
  'enc_N_ff_laynorm/bias' : 'transformer/body/encoder/layer_1/ffn/layer_prepostprocess/layer_norm/layer_norm_bias',
  'enc_N_ff_conv1/W' : 'transformer/body/encoder/layer_1/ffn/conv1/kernel',
  'enc_N_ff_conv1/b' : 'transformer/body/encoder/layer_1/ffn/conv1/bias',
  'enc_N_ff_conv2/W' : 'transformer/body/encoder/layer_1/ffn/conv2/kernel',
  'enc_N_ff_conv2/b' : 'transformer/body/encoder/layer_1/ffn/conv2/bias',



  'output/rec/dec_1_self_att_laynorm/scale' : 'transformer/body/decoder/layer_0/self_attention/layer_prepostprocess/layer_norm/layer_norm_scale',
  'output/rec/dec_1_self_att_laynorm/bias' : 'transformer/body/decoder/layer_0/self_attention/layer_prepostprocess/layer_norm/layer_norm_bias',
  'output/rec/dec_1_self_att_/QKV' : ('transformer/body/decoder/layer_0/self_attention/multihead_attention/q/kernel',
       'transformer/body/decoder/layer_0/self_attention/multihead_attention/k/kernel',
       'transformer/body/decoder/layer_0/self_attention/multihead_attention/v/kernel'),
  'output/rec/dec_1_self_att_lin/W' : 'transformer/body/decoder/layer_0/self_attention/multihead_attention/output_transform/kernel',
  'output/rec/dec_1_att_laynorm/scale' : 'transformer/body/decoder/layer_0/encdec_attention/layer_prepostprocess/layer_norm/layer_norm_scale',
  'output/rec/dec_1_att_laynorm/bias' : 'transformer/body/decoder/layer_0/encdec_attention/layer_prepostprocess/layer_norm/layer_norm_bias',
  'output/rec/dec_1_att_query0/W' : 'transformer/body/decoder/layer_0/encdec_attention/multihead_attention/q/kernel',
  'dec_1_att_key0/W' : 'transformer/body/decoder/layer_0/encdec_attention/multihead_attention/k/kernel',
  'dec_1_att_value0/W' : 'transformer/body/decoder/layer_0/encdec_attention/multihead_attention/v/kernel',
  'output/rec/dec_1_att_lin/W' : 'transformer/body/decoder/layer_0/encdec_attention/multihead_attention/output_transform/kernel',
  'output/rec/dec_1_ff_laynorm/scale' : 'transformer/body/decoder/layer_0/ffn/layer_prepostprocess/layer_norm/layer_norm_scale',
  'output/rec/dec_1_ff_laynorm/bias' : 'transformer/body/decoder/layer_0/ffn/layer_prepostprocess/layer_norm/layer_norm_bias',
  'output/rec/dec_1_ff_conv1/W' : 'transformer/body/decoder/layer_0/ffn/conv1/kernel',
  'output/rec/dec_1_ff_conv1/b' : 'transformer/body/decoder/layer_0/ffn/conv1/bias',
  'output/rec/dec_1_ff_conv2/W' : 'transformer/body/decoder/layer_0/ffn/conv2/kernel',
  'output/rec/dec_1_ff_conv2/b' : 'transformer/body/decoder/layer_0/ffn/conv2/bias',

  'output/rec/dec_N_self_att_laynorm/scale' : 'transformer/body/decoder/layer_1/self_attention/layer_prepostprocess/layer_norm/layer_norm_scale',
  'output/rec/dec_N_self_att_laynorm/bias' : 'transformer/body/decoder/layer_1/self_attention/layer_prepostprocess/layer_norm/layer_norm_bias',
  'output/rec/dec_N_self_att_/QKV' : ('transformer/body/decoder/layer_1/self_attention/multihead_attention/q/kernel',
       'transformer/body/decoder/layer_1/self_attention/multihead_attention/k/kernel',
       'transformer/body/decoder/layer_1/self_attention/multihead_attention/v/kernel'),
  'output/rec/dec_N_self_att_lin/W' : 'transformer/body/decoder/layer_1/self_attention/multihead_attention/output_transform/kernel',
  'output/rec/dec_N_att_laynorm/scale' : 'transformer/body/decoder/layer_1/encdec_attention/layer_prepostprocess/layer_norm/layer_norm_scale',
  'output/rec/dec_N_att_laynorm/bias' : 'transformer/body/decoder/layer_1/encdec_attention/layer_prepostprocess/layer_norm/layer_norm_bias',
  'output/rec/dec_N_att_query0/W' : 'transformer/body/decoder/layer_1/encdec_attention/multihead_attention/q/kernel',
  'dec_N_att_key0/W' : 'transformer/body/decoder/layer_1/encdec_attention/multihead_attention/k/kernel',
  'dec_N_att_value0/W' : 'transformer/body/decoder/layer_1/encdec_attention/multihead_attention/v/kernel',
  'output/rec/dec_N_att_lin/W' : 'transformer/body/decoder/layer_1/encdec_attention/multihead_attention/output_transform/kernel',
  'output/rec/dec_N_ff_laynorm/scale' : 'transformer/body/decoder/layer_1/ffn/layer_prepostprocess/layer_norm/layer_norm_scale',
  'output/rec/dec_N_ff_laynorm/bias' : 'transformer/body/decoder/layer_1/ffn/layer_prepostprocess/layer_norm/layer_norm_bias',
  'output/rec/dec_N_ff_conv1/W' : 'transformer/body/decoder/layer_1/ffn/conv1/kernel',
  'output/rec/dec_N_ff_conv1/b' : 'transformer/body/decoder/layer_1/ffn/conv1/bias',
  'output/rec/dec_N_ff_conv2/W' : 'transformer/body/decoder/layer_1/ffn/conv2/kernel',
  'output/rec/dec_N_ff_conv2/b' : 'transformer/body/decoder/layer_1/ffn/conv2/bias',
}












if __name__ == "__main__":
  better_exchook.install()
  main()
