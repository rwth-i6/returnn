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
  rnn.initBetterExchook()
  rnn.initThreadJoinHack()
  print("Using config file %r." % config_filename)
  assert os.path.exists(config_filename)
  rnn.initConfig(configFilename=config_filename, commandLineOptions=[])
  global config
  config = rnn.config
  config.set("log", None)
  config.set("log_verbosity", log_verbosity)
  config.set("use_tensorflow", True)
  rnn.initLog()
  print("Returnn compile-native-op starting up.", file=log.v1)
  rnn.returnnGreeting()
  rnn.initBackendEngine()
  assert Util.BackendEngine.is_tensorflow_selected(), "this is only for TensorFlow"
  rnn.initFaulthandler()
  rnn.initConfigJsonNetwork()


def create_graph(train_flag, eval_flag, search_flag):
  """
  :param bool train_flag:
  :param bool eval_flag:
  :param bool search_flag:
  :return: nothing, adds to the current graph
  """
  assert 'network' in config.typed_dict
  print("Loading network, train flag %s, eval flag %s, search flag %s" % (train_flag, eval_flag, search_flag))
  from TFEngine import Engine
  Engine.create_network(
    config=config, rnd_seed=1,
    train_flag=train_flag, eval_flag=eval_flag, search_flag=search_flag,
    net_dict=config.typed_dict["network"])


def main(argv):
  argparser = argparse.ArgumentParser(description='Compile some op')
  argparser.add_argument('config', help="filename to config-file")
  argparser.add_argument('--train', type=int, default=0, help='0 disable (default), 1 enable, -1 dynamic')
  argparser.add_argument('--eval', type=int, default=0, help='calculate losses. 0 disable (default), 1 enable')
  argparser.add_argument('--search', type=int, default=0, help='beam search. 0 disable (default), 1 enable')
  argparser.add_argument("--verbosity", default=4, type=int, help="5 for all seqs (default: 4)")
  argparser.add_argument("--output_file", help='output pb or pbtxt file')
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
    create_graph(train_flag=train_flag, eval_flag=eval_flag, search_flag=search_flag)

    print("Graph collection keys:", graph.get_all_collection_keys())
    print("Graph num operations:", len(graph.get_operations()))
    graph_def = graph.as_graph_def(add_shapes=True)
    print("Graph def size:", Util.human_bytes_size(graph_def.ByteSize()))

    if args.output_file:
      filename = args.output_file
      _, ext = os.path.splitext(filename)
      assert ext in [".pb", ".pbtxt"], 'filename %r extension should be pb or pbtxt' % filename
      print("Write graph to file:", filename)
      graph_io.write_graph(
        graph_def,
        logdir=os.path.dirname(filename),
        name=os.path.basename(filename),
        as_text=(ext == ".pbtxt"))
    else:
      print("Use --output_file if you want to store the graph.")


if __name__ == '__main__':
  main(sys.argv)
