#!/usr/bin/env python3

"""
Iterates through any dataset, and prints/dump some information.
This can also be used to collect statistics over the data like mean/variance.
"""

from __future__ import annotations

import os
import sys
import time
import typing

import _setup_returnn_env  # noqa
from returnn import __main__ as rnn
from returnn.log import log
import argparse
import numpy
from returnn.datasets import init_dataset, Dataset
from returnn.util.basic import Stats, hms, hms_fraction, pretty_print, NumbersDict
from returnn.util import basic as util


dataset = None  # type: typing.Optional[Dataset]


def plot(m):
    """
    :param numpy.ndarray m:
    """
    print("Plotting matrix of shape %s." % (m.shape,))
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from matplotlib.pyplot import matshow, show

    matshow(m.transpose())
    show()


def dump_dataset(options):
    """
    :param options: argparse.Namespace
    """
    print("Epoch: %i" % options.epoch, file=log.v3)
    seq_list = None
    if options.seqtags:
        seq_list = options.seqtags.split(",")
    dataset.init_seq_order(epoch=options.epoch, seq_list=seq_list)
    print("Dataset keys:", dataset.get_data_keys(), file=log.v3)
    print("Dataset target keys:", dataset.get_target_list(), file=log.v3)
    print(
        "Dataset labels:",
        ", ".join(f"{k!r}: {v[:3]}... len {len(v)}" for k, v in dataset.labels.items()) or "None",
        file=log.v3,
    )
    assert (
        options.key in dataset.get_data_keys()
    ), f"key {options.key!r} not in {dataset.get_data_keys()} (targets {dataset.get_target_list()})"
    max_seq_length = NumbersDict(options.max_seq_length)
    min_seq_length = NumbersDict(options.min_seq_length)

    if options.get_num_seqs:
        print("Get num seqs.")
        if max_seq_length or min_seq_length:
            raise Exception("Cannot use --get_num_seqs together with --max_seq_length or --min_seq_length.")
        print("estimated_num_seqs: %r" % dataset.estimated_num_seqs)
        try:
            print("num_seqs: %r" % dataset.num_seqs)
        except Exception as exc:
            print("num_seqs exception %r, which is valid, so we count." % exc)
            seq_idx = 0
            if dataset.get_target_list():
                default_target = dataset.get_target_list()[0]
            else:
                default_target = None
            while dataset.is_less_than_num_seqs(seq_idx):
                dataset.load_seqs(seq_idx, seq_idx + 1)
                if seq_idx % 10000 == 0:
                    if default_target:
                        targets = dataset.get_data(seq_idx, default_target)
                        postfix = " (targets = %r...)" % (targets[:10],)
                    else:
                        postfix = ""
                    print("%i ...%s" % (seq_idx, postfix))
                seq_idx += 1
            print("accumulated num seqs: %i" % seq_idx)
        print("Done.")
        return

    dump_file = None
    if options.type == "numpy":
        print("Dump files: %r*%r" % (options.dump_prefix, options.dump_postfix), file=log.v3)
    elif options.type == "stdout":
        print("Dump to stdout", file=log.v3)
        if options.stdout_limit is not None:
            util.set_pretty_print_default_limit(options.stdout_limit)
            numpy.set_printoptions(
                threshold=sys.maxsize if options.stdout_limit == float("inf") else int(options.stdout_limit)
            )
        if options.stdout_as_bytes:
            util.set_pretty_print_as_bytes(options.stdout_as_bytes)
    elif options.type == "print_tag":
        print("Dump seq tag to stdout", file=log.v3)
    elif options.type == "dump_tag":
        dump_file = open("%sseq-tags.txt" % options.dump_prefix, "w")
        print("Dump seq tag to file: %s" % (dump_file.name,), file=log.v3)
    elif options.type == "dump_seq_len":
        dump_file = open("%sseq-lens.txt" % options.dump_prefix, "w")
        print("Dump seq lens to file: %s" % (dump_file.name,), file=log.v3)
        dump_file.write("{\n")
    elif options.type == "print_shape":
        print("Dump shape to stdout", file=log.v3)
    elif options.type == "plot":
        print("Plot.", file=log.v3)
    elif options.type == "interactive":
        print("Interactive debug shell.", file=log.v3)
    elif options.type == "null":
        if options.dump_stats:
            print("No dump (except stats).")
        else:
            print("No dump.")
    else:
        raise Exception("unknown dump option type %r" % options.type)

    start_time = time.time()
    stats = Stats() if (options.stats or options.dump_stats) else None
    seq_len_stats = {key: Stats() for key in dataset.get_data_keys()}
    seq_len_stats_filtered = {key: Stats() for key in dataset.get_data_keys()}
    seq_idx = options.startseq
    if options.endseq < 0:
        options.endseq = float("inf")
    while dataset.is_less_than_num_seqs(seq_idx) and seq_idx <= options.endseq:
        dataset.load_seqs(seq_idx, seq_idx + 1)
        complete_frac = dataset.get_complete_frac(seq_idx)
        start_elapsed = time.time() - start_time
        try:
            num_seqs_s = str(dataset.num_seqs)
        except NotImplementedError:
            try:
                num_seqs_s = "~%i" % dataset.estimated_num_seqs
            except TypeError:  # a number is required, not NoneType
                num_seqs_s = "?"
        seq_len = dataset.get_seq_length(seq_idx)
        for key in dataset.get_data_keys():
            seq_len_stats[key].collect([seq_len[key]])
        filter_out_seq_reasons = []
        if max_seq_length or min_seq_length:
            for key in dataset.get_data_keys():
                if max_seq_length.has_value_for(key) and seq_len[key] > max_seq_length[key]:
                    filter_out_seq_reasons.append(f"len({key}) = {seq_len[key]} > {max_seq_length[key]}")
                if min_seq_length.has_value_for(key) and seq_len[key] < min_seq_length[key]:
                    filter_out_seq_reasons.append(f"len({key}) = {seq_len[key]} < {min_seq_length[key]}")
        if not filter_out_seq_reasons:
            for key in dataset.get_data_keys():
                seq_len_stats_filtered[key].collect([seq_len[key]])
        progress_prefix = "%i/%s" % (seq_idx, num_seqs_s)
        progress = "%s (%.02f%%)" % (progress_prefix, complete_frac * 100)
        data = None
        if complete_frac > 0:
            total_time_estimated = start_elapsed / complete_frac
            remaining_estimated = total_time_estimated - start_elapsed
            progress += " (%s)" % hms(remaining_estimated)
        if filter_out_seq_reasons:
            if options.type != "null":
                print(
                    "seq %s tag %r filtered out: %s"
                    % (
                        progress,
                        dataset.get_tag(seq_idx),
                        ", ".join(filter_out_seq_reasons),
                    ),
                    file=log.v2,
                )
        elif options.type == "print_tag":
            print("seq %s tag:" % (progress if log.verbose[2] else progress_prefix), dataset.get_tag(seq_idx))
        elif options.type == "dump_tag":
            print("seq %s tag:" % (progress if log.verbose[2] else progress_prefix), dataset.get_tag(seq_idx))
            dump_file.write("%s\n" % dataset.get_tag(seq_idx))
        elif options.type == "dump_seq_len":
            seq_len = dataset.get_seq_length(seq_idx)[options.key]
            print(
                "seq %s tag:" % (progress if log.verbose[2] else progress_prefix),
                dataset.get_tag(seq_idx),
                "%r len:" % options.key,
                seq_len,
            )
            dump_file.write("%r: %r,\n" % (dataset.get_tag(seq_idx), seq_len))
        else:
            data = dataset.get_data(seq_idx, options.key)
            if options.type == "numpy":
                numpy.savetxt("%s%i.data%s" % (options.dump_prefix, seq_idx, options.dump_postfix), data)
            elif options.type == "stdout":
                print("seq %s tag:" % progress, dataset.get_tag(seq_idx))
                extra = ""
                if "data" in dataset.labels and len(dataset.labels["data"]) > 1:
                    assert dataset.can_serialize_data("data")
                    extra += " (%r)" % dataset.serialize_data(key="data", data=data)
                print("seq %s data: %s%s" % (progress, pretty_print(data), extra))
            elif options.type == "print_shape":
                print("seq %s data shape:" % progress, data.shape)
            elif options.type == "plot":
                plot(data)
            for target in dataset.get_target_list():
                targets = dataset.get_data(seq_idx, target)
                if options.type == "numpy":
                    numpy.savetxt(
                        "%s%i.targets.%s%s" % (options.dump_prefix, seq_idx, target, options.dump_postfix),
                        targets,
                        fmt="%i",
                    )
                elif options.type == "stdout":
                    extra = ""
                    if target in dataset.labels and len(dataset.labels[target]) > 1:
                        assert dataset.can_serialize_data(target)
                        extra += " (%r)" % dataset.serialize_data(key=target, data=targets)
                    print("seq %i target %r: %s%s" % (seq_idx, target, pretty_print(targets), extra))
                elif options.type == "print_shape":
                    print("seq %i target %r shape:" % (seq_idx, target), targets.shape)
            if options.type == "interactive":
                from returnn.util.debug import debug_shell

                debug_shell(locals())
        if stats and not filter_out_seq_reasons:
            stats.collect(data)
        if options.type == "null":
            util.progress_bar_with_time(complete_frac, prefix=progress_prefix)

        seq_idx += 1

    print(
        "Done. Total time %s. More seqs which we did not dumped: %s"
        % (hms_fraction(time.time() - start_time), dataset.is_less_than_num_seqs(seq_idx)),
        file=log.v2,
    )
    for key in dataset.get_data_keys():
        seq_len_stats[key].dump(stream_prefix="Seq-length %r " % key, stream=log.v2)
        if max_seq_length or min_seq_length:
            seq_len_stats_filtered[key].dump(stream_prefix="Seq-length-filtered %r " % key, stream=log.v2)
    if max_seq_length or min_seq_length:
        print("Used max_seq_length %r and min_seq_length %r." % (max_seq_length, min_seq_length), file=log.v3)
    if stats:
        stats.dump(output_file_prefix=options.dump_stats, stream_prefix="Data %r " % options.key, stream=log.v1)
    if options.type == "dump_seq_len":
        dump_file.write("}\n")
    if dump_file:
        print("Dumped to file:", dump_file.name, file=log.v2)
        dump_file.close()


def init(options):
    """
    :param argparse.Namespace options:
    """
    global dataset
    rnn.init_better_exchook()
    dataset_dict = None
    config_filename = None
    config_str = options.returnn_config
    if config_str.strip().startswith("{"):
        print("Using dataset %s." % config_str)
        dataset_dict = eval(config_str.strip())
    elif config_str.endswith(".hdf"):
        dataset_dict = {"class": "HDFDataset", "files": [config_str]}
        print("Using dataset %r." % dataset_dict)
        assert os.path.exists(config_str)
    else:
        config_filename = config_str
        print("Using config file %r." % config_filename)
        assert os.path.exists(config_filename)
    rnn.init_config(config_filename=config_filename, default_config={"cache_size": "0"})
    config = rnn.config
    config.set("log", None)
    config.set("log_verbosity", options.verbosity)
    config.set("torch_distributed", None)
    config.set("use_horovod", None)
    config_dataset = options.dataset
    if dataset_dict:
        assert not config_dataset
        dataset = init_dataset(dataset_dict)
    elif config_dataset and config_dataset != "train":
        print("Use dataset %r from config." % config_dataset)
        dataset = init_dataset("config:%s" % config_dataset)
    else:
        print("Use train dataset from config.")
        assert config.value("train", None)
        dataset = init_dataset("config:train")
    rnn.init_log()
    print("Returnn dump-dataset starting up.", file=log.v2)
    rnn.returnn_greeting()
    rnn.init_faulthandler()
    print("Dataset:", file=log.v2)
    print("  input:", dataset.num_inputs, "x", dataset.window, file=log.v2)
    print("  output:", dataset.num_outputs, file=log.v2)
    print(" ", dataset.len_info(fast=True) or "no info", file=log.v2)
    if options.max_seq_length == "config":
        options.max_seq_length = config.typed_value("max_seq_length", sys.maxsize)
    elif options.max_seq_length:
        options.max_seq_length = eval(options.max_seq_length)  # noqa
    if options.min_seq_length == "config":
        options.min_seq_length = config.typed_value("min_seq_length", 0)
    elif options.min_seq_length:
        options.min_seq_length = eval(options.min_seq_length)  # noqa


def main():
    """
    Main entry.
    """
    argparser = argparse.ArgumentParser(description="Dump something from dataset.")
    argparser.add_argument("returnn_config", help="either filename to config-file, or dict for dataset")
    argparser.add_argument("--dataset", help="if given the config, specifies the dataset. e.g. 'dev'")
    argparser.add_argument("--epoch", type=int, default=1)
    argparser.add_argument("--startseq", type=int, default=0, help="start seq idx (inclusive) (default: 0)")
    argparser.add_argument("--endseq", type=int, default=None, help="end seq idx (inclusive) or -1 (default: 10)")
    argparser.add_argument("--seqtags", type=str, default=None, help="comma-separated list of seq-tags to dump")
    argparser.add_argument("--get_num_seqs", action="store_true")
    argparser.add_argument("--type", default="stdout", help="'numpy', 'stdout', 'plot', 'null' (default 'stdout')")
    argparser.add_argument("--stdout_limit", type=float, default=None, help="e.g. inf to disable")
    argparser.add_argument("--stdout_as_bytes", action="store_true")
    argparser.add_argument("--verbosity", type=int, default=4, help="overwrites log_verbosity (default: 4)")
    argparser.add_argument("--dump_prefix", default="/tmp/returnn.dump-dataset.")
    argparser.add_argument("--dump_postfix", default=".txt.gz")
    argparser.add_argument("--key", default="data", help="data-key, e.g. 'data' or 'classes'. (default: 'data')")
    argparser.add_argument("--stats", action="store_true", help="calculate mean/stddev stats")
    argparser.add_argument("--dump_stats", help="file-prefix to dump stats to")
    argparser.add_argument("--max_seq_length", help="'config' or dict or int")
    argparser.add_argument("--min_seq_length", help="'config' or dict or int")
    args = argparser.parse_args()
    if args.endseq is None:
        args.endseq = 10 if not args.seqtags else -1
    init(args)
    try:
        dump_dataset(args)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        sys.exit(1)
    finally:
        rnn.finalize()


if __name__ == "__main__":
    main()
