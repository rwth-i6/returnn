#!/usr/bin/env python3

"""
Creates a HDF file, which can be read by :class:`HDFDataset`.
The input is any other dataset (:class:`Dataset`).
"""

from __future__ import annotations

import sys
import argparse

import _setup_returnn_env  # noqa
from returnn.log import log
import returnn.__main__ as rnn
import returnn.datasets.hdf as hdf_dataset_mod
from returnn.datasets import Dataset, init_dataset
from returnn.config import Config


def hdf_dataset_init(file_name):
    """
    :param str file_name: filename of hdf dataset file in the filesystem
    :rtype: hdf_dataset_mod.HDFDatasetWriter
    """
    return hdf_dataset_mod.HDFDatasetWriter(filename=file_name)


def hdf_dump_from_dataset(dataset, hdf_dataset, parser_args):
    """
    :param Dataset dataset: could be any dataset implemented as child of Dataset
    :param hdf_dataset_mod.HDFDatasetWriter hdf_dataset:
    :param parser_args: argparse object from main()
    """
    hdf_dataset.dump_from_dataset(
        dataset=dataset,
        epoch=parser_args.epoch,
        start_seq=parser_args.start_seq,
        end_seq=parser_args.end_seq,
        use_progress_bar=True,
    )


def hdf_close(hdf_dataset):
    """
    :param HDFDataset.HDFDatasetWriter hdf_dataset: to close
    """
    hdf_dataset.close()


def init(config_filename, cmd_line_opts, dataset_config_str):
    """
    :param str config_filename: global config for CRNN
    :param list[str] cmd_line_opts: options for init_config method
    :param str dataset_config_str: dataset via init_dataset_via_str()
    """
    rnn.init_better_exchook()
    rnn.init_thread_join_hack()
    if config_filename:
        rnn.init_config(config_filename, cmd_line_opts)
        rnn.init_log()
    else:
        log.initialize(verbosity=[5])
    print("Returnn hdf_dump starting up.", file=log.v3)
    rnn.init_faulthandler()
    if config_filename:
        rnn.init_data()
        rnn.print_task_properties()
        assert isinstance(rnn.train_data, Dataset)
        dataset = rnn.train_data
    else:
        assert dataset_config_str
        dataset = init_dataset(dataset_config_str)
    print("Source dataset:", dataset.len_info(), file=log.v3)
    return dataset


def _is_crnn_config(filename):
    """
    :param str filename:
    :rtype: bool
    """
    if filename.endswith(".gz"):
        return False
    if filename.endswith(".config"):
        return True
    # noinspection PyBroadException
    try:
        config = Config()
        config.load_file(filename)
        return True
    except Exception:
        pass
    return False


def main(argv):
    """
    Main entry.
    """
    parser = argparse.ArgumentParser(description="Dump dataset or subset of dataset into external HDF dataset")
    parser.add_argument(
        "config_file_or_dataset", type=str, help="Config file for RETURNN, or directly the dataset init string"
    )
    parser.add_argument("hdf_filename", type=str, help="File name of the HDF dataset, which will be created")
    parser.add_argument("--start_seq", type=int, default=0, help="Start sequence index of the dataset to dump")
    parser.add_argument("--end_seq", type=int, default=float("inf"), help="End sequence index of the dataset to dump")
    parser.add_argument("--epoch", type=int, default=1, help="Optional start epoch for initialization")

    args = parser.parse_args(argv[1:])
    returnn_config = None
    dataset_config_str = None
    if _is_crnn_config(args.config_file_or_dataset):
        returnn_config = args.config_file_or_dataset
    else:
        dataset_config_str = args.config_file_or_dataset
    dataset = init(config_filename=returnn_config, cmd_line_opts=[], dataset_config_str=dataset_config_str)
    hdf_dataset = hdf_dataset_init(args.hdf_filename)
    hdf_dump_from_dataset(dataset, hdf_dataset, args)
    hdf_close(hdf_dataset)

    rnn.finalize()


if __name__ == "__main__":
    main(sys.argv)
