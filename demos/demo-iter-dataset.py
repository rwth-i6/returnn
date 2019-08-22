#!/usr/bin/env python

from __future__ import print_function

import sys
import os

# Add parent dir to Python path so that we can use GeneratingDataset and other CRNN code.
my_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.normpath(my_dir + "/..")
if parent_dir not in sys.path:
  sys.path += [parent_dir]

import rnn
from Engine import Engine
from Log import log

dev_num_batches = 1


def iterateDataset(dataset, recurrent_net, batch_size, max_seqs):
  """
  :type dataset: Dataset.Dataset
  :type recurrent_net: bool
  :type batch_size: int
  :type max_seqs: int
  """
  batch_gen = dataset.generate_batches(recurrent_net=recurrent_net, batch_size=batch_size, max_seqs=max_seqs)
  while batch_gen.has_more():
    batches = batch_gen.peek_next_n(dev_num_batches)
    for batch in batches:
      dataset.load_seqs(batch.start_seq, batch.end_seq)
    batch_gen.advance(len(batches))


def iterateEpochs():
  start_epoch, start_batch = Engine.get_train_start_epoch_batch(config)
  final_epoch = Engine.config_get_final_epoch(config)

  print("Starting with epoch %i, batch %i." % (start_epoch, start_batch), file=log.v3)
  print("Final epoch is: %i" % final_epoch, file=log.v3)

  recurrent_net = "lstm" in config.value("hidden_type", "")  # good enough...
  batch_size = config.int('batch_size', 1)
  max_seqs = config.int('max_seqs', -1)

  for epoch in range(start_epoch, final_epoch + 1):
    print("Epoch %i." % epoch, file=log.v3)
    rnn.train_data.init_seq_order(epoch)
    iterateDataset(rnn.train_data, recurrent_net=recurrent_net, batch_size=batch_size, max_seqs=max_seqs)

  print("Finished all epochs.", file=log.v3)


def init(configFilename, commandLineOptions):
  rnn.init_better_exchook()
  rnn.init_thread_join_hack()
  rnn.init_config(configFilename, commandLineOptions)
  global config
  config = rnn.config
  rnn.init_log()
  print("CRNN demo-dataset starting up", file=log.v3)
  rnn.init_faulthandler()
  rnn.init_config_json_network()
  rnn.init_data()
  rnn.print_task_properties()


def main(argv):
  assert len(argv) >= 2, "usage: %s <config>" % argv[0]
  init(configFilename=argv[1], commandLineOptions=argv[2:])
  iterateEpochs()
  rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
