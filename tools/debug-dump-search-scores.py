#!/usr/bin/env python3

"""
Construct/compile the computation graph, and optionally save it to some file.
There are various options/variations for what task and what conditions you can create the graph,
e.g. for training, forwarding, search, or also step-by-step execution over a recurrent layer.

You can use ``debug-plot-search-scores.py`` to visualize some of this.
"""

from __future__ import print_function

import typing
import os
import sys
from pprint import pprint

import _setup_returnn_env  # noqa
import returnn.__main__ as rnn
from returnn.log import log
from returnn.config import Config
import argparse
import returnn.util.basic as util
from returnn.tf.engine import Engine
from returnn.datasets import init_dataset
from returnn.datasets.meta import MetaDataset
from returnn.util import better_exchook


config = None  # type: typing.Optional[Config]


def init(config_filename, log_verbosity, remaining_args=()):
  """
  :param str config_filename: filename to config-file
  :param int log_verbosity:
  :param list[str] remaining_args:
  """
  rnn.init_better_exchook()
  rnn.init_thread_join_hack()
  print("Using config file %r." % config_filename)
  assert os.path.exists(config_filename)
  rnn.init_config(
    config_filename=config_filename,
    command_line_options=remaining_args,
    extra_updates={
      "use_tensorflow": True,
      "log": None,
      "log_verbosity": log_verbosity,
      "task": "search",
    },
    default_config={
      "debug_print_layer_output_template": True,
    }
  )
  global config
  config = rnn.config
  rnn.init_log()
  print("Returnn %s starting up." % os.path.basename(__file__), file=log.v1)
  rnn.returnn_greeting()
  rnn.init_backend_engine()
  assert util.BackendEngine.is_tensorflow_selected(), "this is only for TensorFlow"
  rnn.init_faulthandler()
  better_exchook.replace_traceback_format_tb()  # makes some debugging easier
  rnn.init_config_json_network()


def prepare_compile(rec_layer_name, net_dict, cheating, dump_att_weights, hdf_filename, possible_labels):
  """
  :param str rec_layer_name:
  :param dict[str] net_dict: modify inplace
  :param bool cheating:
  :param bool dump_att_weights:
  :param str hdf_filename:
  :param dict[str,list[str]] possible_labels:
  """
  assert isinstance(net_dict, dict)
  assert rec_layer_name in net_dict
  rec_layer_dict = net_dict[rec_layer_name]
  assert rec_layer_dict["class"] == "rec"
  rec_layer_dict["include_eos"] = True
  rec_unit = rec_layer_dict["unit"]
  assert isinstance(rec_unit, dict)
  relevant_layer_names = []
  target = None
  for name, layer_desc in sorted(rec_unit.items()):
    assert isinstance(name, str)
    if name.startswith("#"):
      continue
    assert isinstance(layer_desc, dict)
    assert "class" in layer_desc
    class_name = layer_desc["class"]
    assert isinstance(class_name, str)
    if dump_att_weights and class_name == "softmax_over_spatial":
      print("Dump softmax_over_spatial layer %r." % name)
      rec_unit["_%s_spatial_sm_value" % name] = {"class": "copy", "from": name, "is_output_layer": True}
      relevant_layer_names.append("_%s_spatial_sm_value" % name)
      continue
    if class_name != "choice":  # only use choice layers for now
      continue
    if cheating and layer_desc["target"]:
      print("Enable cheating for layer %r with target %r." % (name, layer_desc["target"]))
      layer_desc["cheating"] = True
    if name == "output":
      target = layer_desc["target"]
    # Similar to test_search_multi_choice.
    rec_unit["_%s_value" % name] = {"class": "copy", "from": name}
    rec_unit["_%s_src_beams" % name] = {"class": "choice_get_src_beams", "from": name}
    rec_unit["_%s_beam_scores" % name] = {"class": "choice_get_beam_scores", "from": name}
    for name_ in ["_%s_value" % name, "_%s_src_beams" % name, "_%s_beam_scores" % name]:
      rec_unit[name_]["is_output_layer"] = True
      relevant_layer_names.append(name_)
      rec_unit["%s_raw" % name_] = {"class": "decide_keep_beam", "from": name_, "is_output_layer": True}
      relevant_layer_names.append("%s_raw" % name_)
  print("Collected layers:")
  pprint(relevant_layer_names)
  for i, name in enumerate(list(relevant_layer_names)):
    full_name = "%s/%s" % (rec_layer_name, name)
    if name.endswith("_raw"):
      relevant_layer_names[i] = full_name
    else:
      net_dict["%s_%s_final" % (rec_layer_name, name)] = {"class": "decide_keep_beam", "from": full_name}
      relevant_layer_names[i] = "%s_%s_final" % (rec_layer_name, name)
  net_dict["%s__final_beam_scores_" % rec_layer_name] = {"class": "choice_get_beam_scores", "from": rec_layer_name}
  net_dict["%s__final_beam_scores" % rec_layer_name] = {
    "class": "decide_keep_beam", "from": "%s__final_beam_scores_" % rec_layer_name}
  relevant_layer_names.append("%s__final_beam_scores" % rec_layer_name)
  net_dict["%s_final_decided_" % rec_layer_name] = {"class": "decide", "from": rec_layer_name}
  net_dict["%s_final_decided" % rec_layer_name] = {
    "class": "decide_keep_beam", "from": "%s_final_decided_" % rec_layer_name}
  if target and target in possible_labels:
    print("Using labels from target %r." % target)
  net_dict["debug_search_dump"] = {
    "class": "hdf_dump",
    "filename": hdf_filename,
    "from": "%s_final_decided" % rec_layer_name,
    "extra": {name.replace("/", "_"): name for name in relevant_layer_names},
    "labels": possible_labels.get(target, None),
    "is_output_layer": True,
    "dump_whole_batches": True,  # needed if there are different beam sizes...
  }


def main(argv):
  """
  Main entry.
  """
  arg_parser = argparse.ArgumentParser(description='Dump search scores and other info to HDF file.')
  arg_parser.add_argument('config', help="filename to config-file")
  arg_parser.add_argument("--dataset", default="config:train")
  arg_parser.add_argument("--epoch", type=int, default=-1, help="-1 for last epoch")
  arg_parser.add_argument("--output_file", help='hdf', required=True)
  arg_parser.add_argument("--rec_layer_name", default="output")
  arg_parser.add_argument("--cheating", action="store_true", help="add ground truth to the beam")
  arg_parser.add_argument("--att_weights", action="store_true", help="dump all softmax_over_spatial layers")
  arg_parser.add_argument("--verbosity", default=4, type=int, help="5 for all seqs (default: 4)")
  arg_parser.add_argument("--seq_list", nargs="+", help="use only these seqs")
  args, remaining_args = arg_parser.parse_known_args(argv[1:])
  init(config_filename=args.config, log_verbosity=args.verbosity, remaining_args=remaining_args)

  dataset = init_dataset(args.dataset)
  print("Dataset:")
  pprint(dataset)
  if args.seq_list:
    dataset.seq_tags_filter = set(args.seq_list)
    dataset.partition_epoch = 1  # reset
    if isinstance(dataset, MetaDataset):
      for sub_dataset in dataset.datasets.values():
        dataset.seq_tags_filter = set(args.seq_list)
        sub_dataset.partition_epoch = 1
    dataset.finish_epoch()  # enforce reset
  if dataset.seq_tags_filter is not None:
    print("Using sequences:")
    pprint(dataset.seq_tags_filter)
  if args.epoch >= 1:
    config.set("load_epoch", args.epoch)

  def net_dict_post_proc(net_dict):
    """
    :param dict[str] net_dict:
    :return: net_dict
    :rtype: dict[str]
    """
    prepare_compile(
      rec_layer_name=args.rec_layer_name, net_dict=net_dict,
      cheating=args.cheating, dump_att_weights=args.att_weights,
      hdf_filename=args.output_file, possible_labels=dataset.labels)
    return net_dict

  engine = Engine(config=config)
  engine.use_search_flag = True
  engine.init_network_from_config(config, net_dict_post_proc=net_dict_post_proc)
  engine.search(
    dataset,
    do_eval=config.bool("search_do_eval", True),
    output_layer_names=args.rec_layer_name)
  engine.finalize()
  print("Search finished.")
  assert os.path.exists(args.output_file), "hdf file not dumped?"


if __name__ == '__main__':
  main(sys.argv)
