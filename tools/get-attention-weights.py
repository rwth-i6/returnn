#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import numpy as np

my_dir = os.path.dirname(os.path.abspath(__file__))
returnn_dir = os.path.dirname(my_dir)
sys.path.append(returnn_dir)
sys.path.insert(0, "base/tools-multisetup")
import tools

default_python_bin = tools.Settings.default_python
returnn_dir_name = "base/%s" % tools.Settings.returnn_dir_name

import argparse
import rnn
from Log import log
from TFEngine import Runner


def inject_retrieval_code(args, layers):
  """
  Injects some retrieval code into the config

  :param layers:
  :type layers:
  :param args:
  :type args:
  :return:
  :rtype:
  """
  global config
  from TFNetworkRecLayer import RecLayer
  network = rnn.engine.network

  assert config is not None
  assert args.rec_layer in network.layers
  assert isinstance(network.layers[args.rec_layer], RecLayer)
  sub_cell = network.layers[args.rec_layer].cell
  subnet = sub_cell.net
  assert all([l in subnet.layers for l in layers]), \
    "layer to retrieve not in subnet"

  new_layers_descr = network.layers_desc.copy()
  for sub_layer in layers:
    rec_ret_layer = "rec_%s" % sub_layer
    if rec_ret_layer in network.layers:
      continue
    # (enc-D, B, enc-E, 1)
    descr = {rec_ret_layer:
               {"class": "get_rec_accumulated",
                "from": args.rec_layer,
                "sub_layer": sub_layer,
                "is_output_layer": True}
             }
    print("injecting", descr)
    new_layers_descr.update(descr)

    # assert that sub_layer inside subnet is a output-layer
    new_layers_descr[args.rec_layer]['unit'][sub_layer]["is_output_layer"] = True

  # reload config/network
  rnn.engine.maybe_init_new_network(new_layers_descr)


def init(configFilename, commandLineOptions, args):
  rnn.initBetterExchook()
  config_updates={
      "log": None,
      "task": "eval",
      "eval": "config:get_sprint_dataset(%r)" % args.data,
      "train": None,
      "dev": None,
      "need_data": True,
      }
  if args.epoch:
    config_updates["load_epoch"] = args.epoch
  if args.do_search:
    config_updates.update({
      "task": "search",
      "search_data": "config:get_sprint_dataset(%r)" % args.data,
      "search_do_eval": False,
      "beam_size": int(args.beam_size),
      "max_seq_length": 0,
      })

  rnn.init(
    configFilename=configFilename, commandLineOptions=commandLineOptions,
    config_updates=config_updates, extra_greeting="CRNN dump-forward starting up.")
  rnn.engine.init_train_from_config(config=rnn.config, train_data=None)

  if rnn.engine.pretrain:
    new_network_desc = rnn.engine.pretrain.get_network_json_for_epoch(rnn.engine.epoch)
    rnn.engine.maybe_init_new_network(new_network_desc)
  global config
  config = rnn.config
  config.set("log", [])
  rnn.initLog()
  print("CRNN get-attention-weights starting up.", file=log.v3)


def main(argv):
  argparser = argparse.ArgumentParser(description='Dump network as JSON.')
  argparser.add_argument("crnn_config_file", type=str)
  argparser.add_argument("--epoch", required=False, type=int)
  argparser.add_argument('--data', default="test")
  argparser.add_argument('--do_search', default=False, action='store_true')
  argparser.add_argument('--beam_size', default=12, type=int)
  argparser.add_argument('--dump_dir', required=True)
  argparser.add_argument("--device", default="gpu")
  argparser.add_argument("--layers", default=["att_weights"], action="append",
                         help="Layer of subnet to grab")
  argparser.add_argument("--rec_layer", default="output", help="Subnet layer to grab from")
  argparser.add_argument("--batch_size", type=int, default=5000)
  args = argparser.parse_args(argv[1:])

  if not os.path.exists(args.dump_dir):
    os.makedirs(args.dump_dir)

  model = ".".join(args.crnn_config_file.split("/")[-1].split(".")[:-1])

  init(configFilename=args.crnn_config_file, commandLineOptions=["--device", args.device], args=args)
  if isinstance(args.layers, str):
    layers = [args.layers]
  else:
    layers = args.layers
  inject_retrieval_code(args, layers)

  network = rnn.engine.network

  assert rnn.eval_data is not None, "provide evaluation data"
  extra_fetches = {}
  for rec_ret_layer in ["rec_%s" % l for l in layers]:
    extra_fetches[rec_ret_layer] = rnn.engine.network.layers[rec_ret_layer].output.placeholder
  extra_fetches.update({
    "output": network.get_default_output_layer().output.get_placeholder_as_batch_major(),
    "output_len": network.get_default_output_layer().output.get_sequence_lengths(), # decoder length
    "encoder_len": network.layers["encoder"].output.get_sequence_lengths(), # encoder length
    "seq_idx": network.get_extern_data("seq_idx", mark_data_key_as_used=True),
    "seq_tag": network.get_extern_data("seq_tag", mark_data_key_as_used=True),
    "target_data": network.get_extern_data("data", mark_data_key_as_used=True),
    "target_classes": network.get_extern_data("bpe", mark_data_key_as_used=True),
  })
  dataset_batch = rnn.eval_data.generate_batches(
    recurrent_net=network.recurrent,
    batch_size=args.batch_size,
    max_seqs=rnn.engine.max_seqs,
    max_seq_length=sys.maxsize,
    used_data_keys=network.used_data_keys)

  # (**dict[str,numpy.ndarray|str|list[numpy.ndarray|str])->None
  def fetch_callback(seq_idx, seq_tag, target_data, target_classes, output, output_len, encoder_len, **kwargs):
    data = {}
    for i in range(len(seq_idx)):
      data[i] = {
        'tag': seq_tag[i],
        'data': target_data[i],
        'classes': target_classes[i],
        'output': output[i],
        'output_len': output_len[i],
        'encoder_len': encoder_len[i],
      }
      for l in [("rec_%s" % l) for l in layers]:
        assert l in kwargs
        data[i][l] = kwargs[l]
      fname = os.path.join(args.dump_dir, '%s_ep%03d_data_%i_%i.npy' % (model, rnn.engine.epoch, seq_idx[0], seq_idx[-1]))
      np.save(fname, data)


  runner = Runner(engine=rnn.engine, dataset=rnn.eval_data,
                  batches=dataset_batch, train=False, extra_fetches=extra_fetches,
                  extra_fetches_callback=fetch_callback)
  runner.run(report_prefix="att-weights ")
  assert runner.finalized
  rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
