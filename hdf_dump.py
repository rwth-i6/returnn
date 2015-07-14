#!/usr/bin/env python

import h5py as h5
import numpy as np
from Log import log
import rnn
import argparse
import sys


def hdf_dataset_init(file_name):
    '''
    :param file_name: string filename of hdf dataset file in the filesystem
    :return: h5py._hl.files.File
    '''
    print log.v3, "Creating HDF dataset file " + file_name
    return h5.File(file_name, "w")


def hdf_dump_from_dataset(dataset, hdf_dataset, parser_args):
    '''
    :param dataset: Dataset type, could be any dataset implemented as child of Dataset
    :param parser_args: argparse object from main()
    :return:
    '''
    print >> log.v3, "Work on epoch: %i" % parser_args.epoch
    rnn.train_data.init_seq_order(parser_args.epoch)

    seq_idx = parser_args.start_seq
    print seq_idx
    print parser_args.end_seq
    num_seqs = 0
    data = np.array([])
    targets = np.array([])
    while dataset.is_less_than_num_seqs(seq_idx) and seq_idx <= parser_args.end_seq:
        dataset.load_seqs(seq_idx, seq_idx)
        data = np.concatenate((data, dataset.get_data(seq_idx)))
        targets = np.concatenate((targets, dataset.get_targets(seq_idx)))
        seq_idx += 1
        num_seqs += 1

    hdf_dataset.create_dataset("inputs", data=data)
    hdf_dataset.create_dataset("targets", data=targets)


def hdf_close(hdf_dataset):
    '''
    :param hdf_dataset: h5py._hl.files.File to close
    '''
    hdf_dataset.close()


def init(configFilename, commandLineOptions):
    '''

    :param configFilename: global config for CRNN
    :param commandLineOptions: options for initConfig method
    :return:
    '''
    rnn.initBetterExchook()
    rnn.initThreadJoinHack()
    rnn.initConfig(configFilename, commandLineOptions)
    global config
    config = rnn.config
    rnn.initLog()
    print >> log.v3, "CRNN dump-dataset starting up."
    rnn.initFaulthandler()
    rnn.initConfigJson()
    rnn.initData()
    rnn.printTaskProperties()


def main(argv):
    parser = argparse.ArgumentParser(description="Dump dataset or subset of dataset in external HDF dataset")
    parser.add_argument('hdf_filename', type=str, help="File name of the HDF dataset, which will be created")
    parser.add_argument('start_seq', type=int, help="Start sequence index of the dataset to dump")
    parser.add_argument('end_seq', type=int, help="End sequence index of the dataset to dump")
    parser.add_argument('epoch', type=int, help="Start epoch for initialization")
    parser.add_argument('crnn_config', type=str, help="Global config file for CRNN")

    args = parser.parse_args(argv[1:])
    init(configFilename=args.crnn_config, commandLineOptions=[])
    hdf_dataset = hdf_dataset_init(args.hdf_filename)
    hdf_dump_from_dataset(rnn.train_data, hdf_dataset, args)
    hdf_close(hdf_dataset)
    rnn.finalize()


if __name__ == '__main__':
    main(sys.argv)


