#!/usr/bin/env python3

"""
Dumps attention weights to Numpy npy files.

To load them::

    d = np.load("....npy").item()
    d = [v for (k, v) in d.items()]
    att_weights = d[-1]['rec_att_weights'].squeeze(axis=2)
    import matplotlib.pyplot as plt
    plt.matshow(att_weights)
    plt.show()

"""

from __future__ import print_function

import os
import sys
import numpy as np
import argparse

my_dir = os.path.dirname(os.path.abspath(__file__))
returnn_dir = os.path.dirname(my_dir)
sys.path.append(returnn_dir)

# Returnn imports
import rnn
from Log import log
from TFEngine import Runner
from Dataset import init_dataset
from Util import NumbersDict


def inject_retrieval_code(args, layers):
  """
  Injects some retrieval code into the config

  :param list[str] layers:
  :param args:
  """
  global config
  from TFNetworkRecLayer import RecLayer, _SubnetworkRecCell
  from TFNetwork import TFNetwork
  network = rnn.engine.network

  assert config is not None
  assert args.rec_layer in network.layers
  rec_layer = network.layers[args.rec_layer]
  assert isinstance(rec_layer, RecLayer)
  sub_cell = rec_layer.cell
  assert isinstance(sub_cell, _SubnetworkRecCell)
  subnet = sub_cell.net
  assert isinstance(subnet, TFNetwork)
  assert all([l in subnet.layers for l in layers]), "layer to retrieve not in subnet"

  new_layers_descr = network.layers_desc.copy()
  for sub_layer in layers:
    rec_ret_layer = "rec_%s" % sub_layer
    if rec_ret_layer in network.layers:
      continue
    # (enc-D, B, enc-E, 1)
    descr = {
      rec_ret_layer: {
        "class": "get_rec_accumulated",
        "from": args.rec_layer,
        "sub_layer": sub_layer,
        "is_output_layer": True
      }}
    print("injecting", descr)
    new_layers_descr.update(descr)

    # assert that sub_layer inside subnet is a output-layer
    new_layers_descr[args.rec_layer]['unit'][sub_layer]["is_output_layer"] = True

  # reload config/network
  rnn.engine.maybe_init_new_network(new_layers_descr)


def init_returnn(config_fn, cmd_line_opts, args):
  """
  :param str config_fn:
  :param list[str] cmd_line_opts:
  :param args: arg_parse object
  """
  rnn.initBetterExchook()
  config_updates = {
    "log": [],
    "task": "eval",
    "need_data": False}
  if args.epoch:
    config_updates["load_epoch"] = args.epoch
  if args.do_search:
    config_updates.update({
      "task": "search",
      "search_do_eval": False,
      "beam_size": args.beam_size,
      "max_seq_length": 0,
      })

  rnn.init(
    configFilename=config_fn, commandLineOptions=cmd_line_opts,
    config_updates=config_updates, extra_greeting="RETURNN get-attention-weights starting up.")
  global config
  config = rnn.config


def init_net():
  if config.has("load_epoch"):
    rnn.engine.init_network_from_config(config=rnn.config)
  else:
    # Will load the latest epoch.
    rnn.engine.init_train_from_config(config=rnn.config)

  if rnn.engine.pretrain:
    new_network_desc = rnn.engine.pretrain.get_network_json_for_epoch(rnn.engine.epoch)
    rnn.engine.maybe_init_new_network(new_network_desc)


def main(argv):
  argparser = argparse.ArgumentParser(description=__doc__)
  argparser.add_argument("config_file", type=str)
  argparser.add_argument("--epoch", required=False, type=int)
  argparser.add_argument('--data', default="train",
                         help="e.g. 'train', 'config:train', or sth like 'config:get_dataset('dev')'")
  argparser.add_argument('--do_search', default=False, action='store_true')
  argparser.add_argument('--beam_size', default=12, type=int)
  argparser.add_argument('--dump_dir', required=True)
  argparser.add_argument("--device", default="gpu")
  argparser.add_argument("--layers", default=["att_weights"], action="append",
                         help="Layer of subnet to grab")
  argparser.add_argument("--rec_layer", default="output", help="Subnet layer to grab from; decoder")
  argparser.add_argument("--enc_layer", default="encoder")
  argparser.add_argument("--batch_size", type=int, default=5000)
  argparser.add_argument("--seq_list", default=[], action="append", help="predefined list of seqs")
  argparser.add_argument("--min_seq_len", default="0", help="can also be dict")
  argparser.add_argument("--output_format", default="npy", help="npy or png")
  args = argparser.parse_args(argv[1:])

  model_name = ".".join(args.config_file.split("/")[-1].split(".")[:-1])

  init_returnn(config_fn=args.config_file, cmd_line_opts=["--device", args.device], args=args)

  if args.do_search:
    raise NotImplementedError
  min_seq_length = NumbersDict(eval(args.min_seq_len))

  if not os.path.exists(args.dump_dir):
    os.makedirs(args.dump_dir)
  assert args.output_format in ["npy", "png"]
  if args.output_format == "png":
    import matplotlib.pyplot as plt  # need to import early? https://stackoverflow.com/a/45582103/133374
  dataset_str = args.data
  if dataset_str in ["train", "dev", "eval"]:
    dataset_str = "config:%s" % dataset_str
  dataset = init_dataset(dataset_str)
  init_net()

  layers = args.layers
  assert isinstance(layers, list)
  inject_retrieval_code(args, layers)

  network = rnn.engine.network

  extra_fetches = {}
  for rec_ret_layer in ["rec_%s" % l for l in layers]:
    extra_fetches[rec_ret_layer] = rnn.engine.network.layers[rec_ret_layer].output.get_placeholder_as_batch_major()
  extra_fetches.update({
    "output": network.layers[args.rec_layer].output.get_placeholder_as_batch_major(),
    "output_len": network.layers[args.rec_layer].output.get_sequence_lengths(),  # decoder length
    "encoder_len": network.layers[args.enc_layer].output.get_sequence_lengths(),  # encoder length
    "seq_idx": network.get_extern_data("seq_idx"),
    "seq_tag": network.get_extern_data("seq_tag"),
    "target_data": network.get_extern_data(network.extern_data.default_input),
    "target_classes": network.get_extern_data(network.extern_data.default_target),
  })
  dataset.init_seq_order(epoch=rnn.engine.epoch, seq_list=args.seq_list or None)
  dataset_batch = dataset.generate_batches(
    recurrent_net=network.recurrent,
    batch_size=args.batch_size,
    max_seqs=rnn.engine.max_seqs,
    max_seq_length=sys.maxsize,
    min_seq_length=min_seq_length,
    used_data_keys=network.used_data_keys)

  # (**dict[str,numpy.ndarray|str|list[numpy.ndarray|str])->None
  def fetch_callback(seq_idx, seq_tag, target_data, target_classes, output, output_len, encoder_len, **kwargs):
    if args.output_format == "npy":
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
          out = kwargs[l][i]
          assert out.ndim >= 2
          assert out.shape[0] >= output_len[i] and out.shape[1] >= encoder_len[i]
          data[i][l] = out[:output_len[i], :encoder_len[i]]
        fname = args.dump_dir + '/%s_ep%03d_data_%i_%i.npy' % (model_name, rnn.engine.epoch, seq_idx[0], seq_idx[-1])
        np.save(fname, data)
    elif args.output_format == "png":
      for i in range(len(seq_idx)):
        for l in layers:
          fname = args.dump_dir + '/%s_ep%03d_plt_%s_%05i.png' % (model_name, rnn.engine.epoch, l, seq_idx[i])
          att_weights = kwargs["rec_%s" % l][i]
          att_weights = att_weights.squeeze(axis=2)
          print("Seq %i, %s: Dump att weights with shape %r to: %s" % (
            seq_idx[i], seq_tag[i], att_weights.shape, fname))
          plt.matshow(att_weights)
          plt.title(seq_tag[i])
          plt.savefig(fname)
          plt.close()
    else:
      raise NotImplementedError("output format %r" % args.output_format)

  runner = Runner(engine=rnn.engine, dataset=dataset,
                  batches=dataset_batch, train=False, extra_fetches=extra_fetches,
                  extra_fetches_callback=fetch_callback)
  runner.run(report_prefix="att-weights epoch %i" % rnn.engine.epoch)
  assert runner.finalized

  rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
