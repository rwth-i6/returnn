#!/usr/bin/env python3


from __future__ import print_function

import os
import sys
import time
import numpy
import tensorflow as tf
from tensorflow.python.framework import graph_io

my_dir = os.path.dirname(os.path.abspath(__file__))
returnn_dir = os.path.dirname(my_dir)
sys.path.insert(0, returnn_dir)

import rnn
from Log import log
import argparse
import Util


def init(config_filename, log_verbosity):
  """
  :param str config_filename: filename to config-file
  :param int log_verbosity:
  """
  rnn.init_better_exchook()
  rnn.init_thread_join_hack()
  print("Using config file %r." % config_filename)
  assert os.path.exists(config_filename)
  rnn.init_config(config_filename=config_filename, command_line_options=[])
  global config
  config = rnn.config
  config.set("log", None)
  config.set("log_verbosity", log_verbosity)
  config.set("use_tensorflow", True)
  rnn.init_log()
  print("Returnn compile-native-op starting up.", file=log.v1)
  rnn.returnn_greeting()
  rnn.init_backend_engine()
  assert Util.BackendEngine.is_tensorflow_selected(), "this is only for TensorFlow"
  rnn.init_faulthandler()
  rnn.init_config_json_network()


def create_graph(train_flag, eval_flag, search_flag):
  """
  :param bool train_flag:
  :param bool eval_flag:
  :param bool search_flag:
  :return: adds to the current graph, and then returns the network
  :rtype: TFNetwork.TFNetwork
  """
  assert 'network' in config.typed_dict
  print("Loading network, train flag %s, eval flag %s, search flag %s" % (train_flag, eval_flag, search_flag))
  from TFEngine import Engine
  from TFNetwork import TFNetwork
  network, updater = Engine.create_network(
    config=config, rnd_seed=1,
    train_flag=train_flag, eval_flag=eval_flag, search_flag=search_flag,
    net_dict=config.typed_dict["network"])
  assert isinstance(network, TFNetwork)
  return network


def main(argv):
  argparser = argparse.ArgumentParser(description='Compile some op')
  argparser.add_argument('config', help="filename to config-file")
  argparser.add_argument('--train', type=int, default=0, help='0 disable (default), 1 enable, -1 dynamic')
  argparser.add_argument('--eval', type=int, default=0, help='calculate losses. 0 disable (default), 1 enable')
  argparser.add_argument('--search', type=int, default=0, help='beam search. 0 disable (default), 1 enable')
  argparser.add_argument("--verbosity", default=4, type=int, help="5 for all seqs (default: 4)")
  argparser.add_argument("--summaries_tensor_name")
  argparser.add_argument("--output_file", help='output pb, pbtxt or meta, metatxt file')
  argparser.add_argument("--output_file_model_params_list", help="line-based, names of model params")
  argparser.add_argument("--output_file_state_vars_list", help="line-based, name of state vars")
  args = argparser.parse_args(argv[1:])
  assert args.train in [0, 1, 2] and args.eval in [0, 1] and args.search in [0, 1]
  init(config_filename=args.config, log_verbosity=args.verbosity)
  with tf.Graph().as_default() as graph:
    assert isinstance(graph, tf.Graph)
    print("Create graph...")
    # See :func:`Engine._init_network`.
    tf.set_random_seed(42)
    if args.train < 0:
      from TFUtil import get_global_train_flag_placeholder
      train_flag = get_global_train_flag_placeholder()
    else:
      train_flag = bool(args.train)
    eval_flag = bool(args.eval)
    search_flag = bool(args.search)
    network = create_graph(train_flag=train_flag, eval_flag=eval_flag, search_flag=search_flag)

    from TFNetworkLayer import LayerBase
    for layer in network.layers.values():
      assert isinstance(layer, LayerBase)
      if layer.output.time_dim_axis is None:
        continue
      with layer.cls_layer_scope(layer.name):
        tf.identity(layer.output.get_placeholder_as_batch_major(), name="output_batch_major")

    tf.group(*network.get_post_control_dependencies(), name="post_control_dependencies")

    if args.summaries_tensor_name:
      summaries_tensor = tf.summary.merge_all()
      assert isinstance(summaries_tensor, tf.Tensor), "no summaries in the graph?"
      tf.identity(summaries_tensor, name=args.summaries_tensor_name)

    if args.output_file and os.path.splitext(args.output_file)[1] in [".meta", ".metatxt"]:
      # https://www.tensorflow.org/api_guides/python/meta_graph
      saver = tf.train.Saver(
        var_list=network.get_saveable_params_list(), max_to_keep=2 ** 31 - 1)
      graph_def = saver.export_meta_graph()
    else:
      graph_def = graph.as_graph_def(add_shapes=True)

    print("Graph collection keys:", graph.get_all_collection_keys())
    print("Graph num operations:", len(graph.get_operations()))
    print("Graph def size:", Util.human_bytes_size(graph_def.ByteSize()))

    if args.output_file:
      filename = args.output_file
      _, ext = os.path.splitext(filename)
      assert ext in [".pb", ".pbtxt", ".meta", ".metatxt"], 'filename %r extension invalid' % filename
      print("Write graph to file:", filename)
      graph_io.write_graph(
        graph_def,
        logdir=os.path.dirname(filename),
        name=os.path.basename(filename),
        as_text=ext.endswith("txt"))
    else:
      print("Use --output_file if you want to store the graph.")

    if args.output_file_model_params_list:
      print("Write model param list to:", args.output_file_model_params_list)
      with open(args.output_file_model_params_list, "w") as f:
        for param in network.get_params_list():
          assert param.name[-2:] == ":0"
          f.write("%s\n" % param.name[:-2])

    if args.output_file_state_vars_list:
      print("Write state var list to:", args.output_file_state_vars_list)
      from TFUtil import CollectionKeys
      with open(args.output_file_state_vars_list, "w") as f:
        for param in tf.get_collection(CollectionKeys.STATE_VARS):
          assert param.name[-2:] == ":0"
          f.write("%s\n" % param.name[:-2])


if __name__ == '__main__':
  main(sys.argv)
