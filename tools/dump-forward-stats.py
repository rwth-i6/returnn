#!/usr/bin/env python

"""
Forward some layer through the network over the dataset,
and collect statistics (mean,std_dev,min,max).
"""

from __future__ import annotations

import sys
import typing
import argparse

import _setup_returnn_env  # noqa
import returnn.__main__ as rnn
from returnn.log import log
from returnn.config import Config
from returnn.datasets import init_dataset, Dataset
from returnn.util.basic import Stats
from returnn.pretrain import pretrain_from_config
from returnn.tf.engine import Engine, Runner
from returnn.tf.network import TFNetwork


engine = None  # type: typing.Optional[Engine]
config = None  # type: typing.Optional[Config]
dataset = None  # type: typing.Optional[Dataset]


def dump(options):
    """
    :param options: argparse.Namespace
    """
    print("Epoch: %i" % options.epoch, file=log.v3)
    dataset.init_seq_order(options.epoch)

    stats = Stats()
    output = engine.network.get_layer(options.layer).output.copy_as_batch_major()

    def _extra_fetches_cb(inputs, **kwargs):
        n_batch = inputs.shape[0]
        # noinspection PyShadowingNames
        seq_len = {i: kwargs["seq_len_%i" % i] for i in output.size_placeholder.keys()}
        assert all([len(v) == n_batch for v in seq_len.values()])
        assert set(seq_len.keys()) == {0}  # not implemented otherwise
        for n in range(n_batch):
            stats.collect(inputs[n, : seq_len[0][n]])

    extra_fetches = {
        "inputs": output.placeholder,
    }
    for i, seq_len in output.size_placeholder.items():
        extra_fetches["seq_len_%i" % i] = seq_len
    batches = dataset.generate_batches(
        recurrent_net=True,  # Want seq lengths
        batch_size=config.typed_value("batch_size", 1),
        max_seqs=config.int("max_seqs", -1),
        used_data_keys=engine.network.get_used_data_keys(),
    )
    forwarder = Runner(
        engine=engine,
        dataset=dataset,
        batches=batches,
        train=False,
        eval=False,
        extra_fetches=extra_fetches,
        extra_fetches_callback=_extra_fetches_cb,
    )
    forwarder.run(report_prefix="forward")
    if not forwarder.finalized:
        print("Error happened. Exit now.")
        sys.exit(1)
    stats.dump(output_file_prefix=options.dump_stats, stream_prefix="Layer %r " % options.layer)


def init(config_filename, command_line_options, args):
    """
    :param str config_filename:
    :param list[str] command_line_options:
    :param args: argparse.Namespace
    """
    global config, engine, dataset
    rnn.init(
        config_filename=config_filename,
        command_line_options=command_line_options,
        config_updates={"log": None, "need_data": False},
        extra_greeting="RETURNN dump-forward starting up.",
    )
    config = rnn.config
    engine = rnn.engine

    dataset_str = args.dataset
    if dataset_str in {"train", "dev", "eval", "search_data"}:
        dataset_str = "config:%s" % dataset_str
    extra_dataset_kwargs = {}
    if args.reset_partition_epoch:
        print("NOTE: We are resetting partition epoch to %i." % (args.reset_partition_epoch,))
        extra_dataset_kwargs["partition_epoch"] = args.reset_partition_epoch
    if args.reset_seq_ordering:
        print("NOTE: We will use %r seq ordering." % (args.reset_seq_ordering,))
        extra_dataset_kwargs["seq_ordering"] = args.reset_seq_ordering
    if args.reset_epoch_wise_filter:
        extra_dataset_kwargs["epoch_wise_filter"] = eval(args.reset_epoch_wise_filter)
    dataset = init_dataset(dataset_str, extra_kwargs=extra_dataset_kwargs)
    if hasattr(dataset, "epoch_wise_filter") and args.reset_epoch_wise_filter is None:
        if dataset.epoch_wise_filter:
            print("NOTE: Resetting epoch_wise_filter to None.")
            dataset.epoch_wise_filter = None
    if args.reset_partition_epoch:
        assert dataset.partition_epoch == args.reset_partition_epoch
    if args.reset_seq_ordering:
        assert dataset.seq_ordering == args.reset_seq_ordering

    config.set("task", "eval")
    if args.load:
        config.set("load", args.load)

    epoch, model_epoch_filename = Engine.get_epoch_model(config)
    engine.pretrain = pretrain_from_config(config)
    engine.custom_get_net_dict = config.typed_value("get_network")
    net_dict = engine.get_net_dict_for_epoch(epoch)
    engine.make_tf_session()
    engine.network = TFNetwork(name="root")
    engine.network.construct_layer(net_dict, args.layer)
    print("Load model:", model_epoch_filename)
    engine.network.load_params_from_file(model_epoch_filename, session=engine.tf_session)


def main(argv):
    """
    Main entry.
    """
    arg_parser = argparse.ArgumentParser(description="Forward something and dump it.")
    arg_parser.add_argument("returnn_config")
    arg_parser.add_argument(
        "--dataset", help="if given the config, specifies the dataset. e.g. 'train'", default="train"
    )
    arg_parser.add_argument("--reset_partition_epoch", type=int, default=1)
    arg_parser.add_argument("--reset_seq_ordering", default="sorted_reverse")
    arg_parser.add_argument("--reset_epoch_wise_filter", default=None)
    arg_parser.add_argument("--layer", required=True)
    arg_parser.add_argument("--epoch", type=int, default=1, help="for the dataset")
    arg_parser.add_argument("--load", help="model to load")
    arg_parser.add_argument("--stats", action="store_true", help="calculate mean/stddev stats over stats_layer")
    arg_parser.add_argument("--dump_stats", help="file-prefix to dump stats to")
    args, remaining_args = arg_parser.parse_known_args(argv[1:])
    init(config_filename=args.returnn_config, command_line_options=remaining_args, args=args)
    dump(args)
    rnn.finalize()


if __name__ == "__main__":
    main(sys.argv)
