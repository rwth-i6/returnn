#!/usr/bin/env python3

"""
Dumps attention weights.

E.g. to Numpy npy files.
To load them::

    d = np.load("....npy").item()
    d = [v for (k, v) in d.items()]
    att_weights = d[-1]['rec_att_weights'].squeeze(axis=2)
    import matplotlib.pyplot as plt
    plt.matshow(att_weights)
    plt.show()

Or directly as png images.

Or into HDF. In this case, this tool is very similar to `rnn.py --task=forward`.

"""

from __future__ import annotations

import os
import sys
import numpy as np
import argparse
from glob import glob
import typing

# Returnn imports
import _setup_returnn_env  # noqa
import returnn.__main__ as rnn
from returnn.tf.engine import Runner
from returnn.datasets import init_dataset
from returnn.util.basic import NumbersDict, Stats, deep_update_dict_values


def inject_retrieval_code(net_dict, rec_layer_name, layers, dropout):
    """
    Injects some retrieval code into the config

    :param dict[str] net_dict:
    :param str rec_layer_name: name of rec layer
    :param list[str] layers: layers in rec layer to extract
    :param float|None dropout: to override, if given
    :return: net_dict
    :rtype: dict[str]
    """
    assert config is not None
    assert rec_layer_name in net_dict
    assert net_dict[rec_layer_name]["class"] == "rec"
    for layer in layers:
        assert layer in net_dict[rec_layer_name]["unit"], "layer %r not found" % layer

    new_layers_descr = net_dict.copy()  # actually better would be deepcopy...
    for sub_layer in layers:
        # assert that sub_layer inside subnet is a output-layer
        new_layers_descr[rec_layer_name]["unit"][sub_layer]["is_output_layer"] = True

    if dropout is not None:
        deep_update_dict_values(net_dict, "dropout", dropout)
        deep_update_dict_values(net_dict, "rec_weight_dropout", dropout)
    return new_layers_descr


config = None  # type: typing.Optional["returnn.config.Config"]


def init_returnn(config_fn, args):
    """
    :param str config_fn:
    :param args: arg_parse object
    """
    rnn.init_better_exchook()
    config_updates = {
        "log": [],
        "task": "eval",
        "need_data": False,  # we will load it explicitly
        "device": args.device if args.device else None,
    }
    if args.epoch:
        config_updates["load_epoch"] = args.epoch
    if args.do_search:
        config_updates.update(
            {
                "task": "search",
                "search_do_eval": False,
                "beam_size": args.beam_size,
                "max_seq_length": 0,
            }
        )

    rnn.init(
        config_filename=config_fn,
        config_updates=config_updates,
        extra_greeting="RETURNN get-attention-weights starting up.",
    )
    global config
    config = rnn.config


def init_net(args, layers):
    """
    :param args:
    :param list[str] layers:
    """

    def net_dict_post_proc(net_dict):
        """
        :param dict[str,dict[str]] net_dict:
        :rtype: dict[str,dict[str]]
        """
        return inject_retrieval_code(net_dict, rec_layer_name=args.rec_layer, layers=layers, dropout=args.dropout)

    rnn.engine.use_dynamic_train_flag = True  # will be set via Runner. maybe enabled if we want dropout
    rnn.engine.init_network_from_config(config=config, net_dict_post_proc=net_dict_post_proc)


def main(argv):
    """
    Main entry.
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("config_file", type=str, help="RETURNN config, or model-dir")
    argparser.add_argument("--epoch", type=int)
    argparser.add_argument(
        "--data", default="train", help="e.g. 'train', 'config:train', or sth like 'config:get_dataset('dev')'"
    )
    argparser.add_argument("--do_search", default=False, action="store_true")
    argparser.add_argument("--beam_size", default=12, type=int)
    argparser.add_argument("--dump_dir", help="for npy or png")
    argparser.add_argument("--output_file", help="hdf")
    argparser.add_argument("--device", help="gpu or cpu (default: automatic)")
    argparser.add_argument("--layers", default=["att_weights"], action="append", help="Layer of subnet to grab")
    argparser.add_argument("--rec_layer", default="output", help="Subnet layer to grab from; decoder")
    argparser.add_argument("--enc_layer", default="encoder")
    argparser.add_argument("--batch_size", type=int, default=5000)
    argparser.add_argument("--seq_list", default=[], action="append", help="predefined list of seqs")
    argparser.add_argument("--min_seq_len", default="0", help="can also be dict")
    argparser.add_argument("--num_seqs", default=-1, type=int, help="stop after this many seqs")
    argparser.add_argument("--output_format", default="npy", help="npy, png or hdf")
    argparser.add_argument("--dropout", default=None, type=float, help="if set, overwrites all dropout values")
    argparser.add_argument("--train_flag", action="store_true")
    argparser.add_argument("--reset_partition_epoch", type=int, default=1)
    argparser.add_argument("--reset_seq_ordering", default="sorted_reverse")
    argparser.add_argument("--reset_epoch_wise_filter", default=None)
    args = argparser.parse_args(argv[1:])

    layers = args.layers
    assert isinstance(layers, list)
    config_fn = args.config_file
    explicit_model_dir = None
    if os.path.isdir(config_fn):
        # Assume we gave a model dir.
        explicit_model_dir = config_fn
        train_log_dir_config_pattern = "%s/train-*/*.config" % config_fn
        train_log_dir_configs = sorted(glob(train_log_dir_config_pattern))
        assert train_log_dir_configs
        config_fn = train_log_dir_configs[-1]
        print("Using this config via model dir:", config_fn)
    else:
        assert os.path.isfile(config_fn)
    model_name = ".".join(config_fn.split("/")[-1].split(".")[:-1])

    init_returnn(config_fn=config_fn, args=args)
    if explicit_model_dir:
        config.set("model", "%s/%s" % (explicit_model_dir, os.path.basename(config.value("model", ""))))
    print("Model file prefix:", config.value("model", ""))

    if args.do_search:
        raise NotImplementedError
    min_seq_length = NumbersDict(eval(args.min_seq_len))

    assert args.output_format in ["npy", "png", "hdf"]
    if args.output_format in ["npy", "png"]:
        assert args.dump_dir
        if not os.path.exists(args.dump_dir):
            os.makedirs(args.dump_dir)
    plt = ticker = None
    if args.output_format == "png":
        import matplotlib.pyplot as plt  # need to import early? https://stackoverflow.com/a/45582103/133374
        import matplotlib.ticker as ticker

    dataset_str = args.data
    if dataset_str in ["train", "dev", "eval"]:
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

    init_net(args, layers)
    network = rnn.engine.network

    hdf_writer = None
    if args.output_format == "hdf":
        assert args.output_file
        assert len(layers) == 1
        sub_layer = network.get_layer("%s/%s" % (args.rec_layer, layers[0]))
        from returnn.datasets.hdf import SimpleHDFWriter

        hdf_writer = SimpleHDFWriter(filename=args.output_file, dim=sub_layer.output.dim, ndim=sub_layer.output.ndim)

    extra_fetches = {
        "output": network.layers[args.rec_layer].output.get_placeholder_as_batch_major(),
        "output_len": network.layers[args.rec_layer].output.get_sequence_lengths(),  # decoder length
        "encoder_len": network.layers[args.enc_layer].output.get_sequence_lengths(),  # encoder length
        "seq_idx": network.get_extern_data("seq_idx"),
        "seq_tag": network.get_extern_data("seq_tag"),
        "target_data": network.get_extern_data(network.extern_data.default_input),
        "target_classes": network.get_extern_data(network.extern_data.default_target),
    }
    for layer in layers:
        sub_layer = rnn.engine.network.get_layer("%s/%s" % (args.rec_layer, layer))
        extra_fetches["rec_%s" % layer] = sub_layer.output.get_placeholder_as_batch_major()
    dataset.init_seq_order(epoch=1, seq_list=args.seq_list or None)  # use always epoch 1, such that we have same seqs
    dataset_batch = dataset.generate_batches(
        recurrent_net=network.recurrent,
        batch_size=args.batch_size,
        max_seqs=rnn.engine.max_seqs,
        max_seq_length=sys.maxsize,
        min_seq_length=min_seq_length,
        max_total_num_seqs=args.num_seqs,
        used_data_keys=network.used_data_keys,
    )

    stats = {layer: Stats() for layer in layers}

    # (**dict[str,numpy.ndarray|str|list[numpy.ndarray|str])->None
    def fetch_callback(seq_idx, seq_tag, target_data, target_classes, output, output_len, encoder_len, **kwargs):
        """
        :param list[int] seq_idx: len is n_batch
        :param list[str] seq_tag: len is n_batch
        :param numpy.ndarray target_data: extern data default input (e.g. "data"), shape e.g. (B,enc-T,...)
        :param numpy.ndarray target_classes: extern data default target (e.g. "classes"), shape e.g. (B,dec-T,...)
        :param numpy.ndarray output: rec layer output, shape e.g. (B,dec-T,...)
        :param numpy.ndarray output_len: rec layer seq len, i.e. decoder length, shape (B,)
        :param numpy.ndarray encoder_len: encoder seq len, shape (B,)
        :param kwargs: contains "rec_%s" % l for l in layers, the sub layers (e.g att weights) we are interested in
        """
        n_batch = len(seq_idx)
        for i in range(n_batch):
            # noinspection PyShadowingNames
            for layer in layers:
                att_weights = kwargs["rec_%s" % layer][i]
                stats[layer].collect(att_weights.flatten())
        if args.output_format == "npy":
            data = {}
            for i in range(n_batch):
                data[i] = {
                    "tag": seq_tag[i],
                    "data": target_data[i],
                    "classes": target_classes[i],
                    "output": output[i],
                    "output_len": output_len[i],
                    "encoder_len": encoder_len[i],
                }
                # noinspection PyShadowingNames
                for layer in [("rec_%s" % layer) for layer in layers]:
                    assert layer in kwargs
                    out = kwargs[layer][i]
                    assert out.ndim >= 2
                    assert out.shape[0] >= output_len[i] and out.shape[1] >= encoder_len[i]
                    data[i][layer] = out[: output_len[i], : encoder_len[i]]
                fname = args.dump_dir + "/%s_ep%03d_data_%i_%i.npy" % (
                    model_name,
                    rnn.engine.epoch,
                    seq_idx[0],
                    seq_idx[-1],
                )
                np.save(fname, data)
        elif args.output_format == "png":
            for i in range(n_batch):
                # noinspection PyShadowingNames
                for layer in layers:
                    extra_postfix = ""
                    if args.dropout is not None:
                        extra_postfix += "_dropout%.2f" % args.dropout
                    elif args.train_flag:
                        extra_postfix += "_train"
                    fname = args.dump_dir + "/%s_ep%03d_plt_%05i_%s%s.png" % (
                        model_name,
                        rnn.engine.epoch,
                        seq_idx[i],
                        layer,
                        extra_postfix,
                    )
                    att_weights = kwargs["rec_%s" % layer][i]
                    att_weights = att_weights.squeeze(axis=2)  # (out,enc)
                    assert att_weights.shape[0] >= output_len[i] and att_weights.shape[1] >= encoder_len[i]
                    att_weights = att_weights[: output_len[i], : encoder_len[i]]
                    print(
                        "Seq %i, %s: Dump att weights with shape %r to: %s"
                        % (seq_idx[i], seq_tag[i], att_weights.shape, fname)
                    )
                    plt.matshow(att_weights)
                    title = seq_tag[i]
                    if dataset.can_serialize_data(network.extern_data.default_target):
                        title += "\n" + dataset.serialize_data(
                            network.extern_data.default_target, target_classes[i][: output_len[i]]
                        )
                        ax = plt.gca()
                        tick_labels = [
                            dataset.serialize_data(
                                network.extern_data.default_target, np.array([x], dtype=target_classes[i].dtype)
                            )
                            for x in target_classes[i][: output_len[i]]
                        ]
                        ax.set_yticklabels([""] + tick_labels, fontsize=8)
                        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
                    plt.title(title)
                    plt.savefig(fname)
                    plt.close()
        elif args.output_format == "hdf":
            assert len(layers) == 1
            att_weights = kwargs["rec_%s" % layers[0]]
            hdf_writer.insert_batch(inputs=att_weights, seq_len={0: output_len, 1: encoder_len}, seq_tag=seq_tag)
        else:
            raise Exception("output format %r" % args.output_format)

    runner = Runner(
        engine=rnn.engine,
        dataset=dataset,
        batches=dataset_batch,
        train=False,
        train_flag=bool(args.dropout) or args.train_flag,
        extra_fetches=extra_fetches,
        extra_fetches_callback=fetch_callback,
    )
    runner.run(report_prefix="att-weights epoch %i" % rnn.engine.epoch)
    for layer in layers:
        stats[layer].dump(stream_prefix="Layer %r " % layer)
    if not runner.finalized:
        print("Some error occured, not finalized.")
        sys.exit(1)

    if hdf_writer:
        hdf_writer.close()
    rnn.finalize()


if __name__ == "__main__":
    main(sys.argv)
