#!/usr/bin/env python3

"""
Benchmarking LSTM training times for different LSTM implementations, both on GPU and CPU.

Some runtimes:

GeForce GTX 680:
  Settings:
  {'batch_size': 2000,
   'chunking': '50:25',
   'max_seqs': 40,
   'n_hidden': 500,
   'num_layers': 5,
   'num_seqs': 500}
  Final results:
    GPU:CudnnLSTM: 0:00:12.3727
    GPU:NativeLSTM: 0:00:19.0306
    GPU:LSTMBlockFused: 0:00:41.5129
    GPU:LSTMBlock: 0:00:52.6296
    GPU:StandardLSTM: 0:00:53.2629
    GPU:BasicLSTM: 0:00:53.6150
    CPU:NativeLSTM: 0:02:37.1753
    CPU:LSTMBlockFused: 0:03:17.5448
    CPU:BasicLSTM: 0:03:28.9882
    CPU:StandardLSTM: 0:03:29.3730
    CPU:LSTMBlock: 0:03:33.7039

  Settings:
  {'batch_size': 2000,
   'chunking': '0',
   'max_seqs': 50,
   'n_hidden': 500,
   'num_layers': 5,
   'num_seqs': 500}
  Final results:
    GPU:CudnnLSTM: 0:00:12.8325
    GPU:NativeLSTM: 0:00:15.8518
    GPU:LSTMBlockFused: 0:01:05.2157
    GPU:StandardLSTM: 0:01:20.6564
    GPU:BasicLSTM: 0:01:20.9896
    GPU:LSTMBlock: 0:01:21.4046
    CPU:NativeLSTM: 0:02:12.5029
    CPU:LSTMBlockFused: 0:03:39.0270
    CPU:LSTMBlock: 0:03:51.9667
    CPU:StandardLSTM: 0:03:56.6404
    CPU:BasicLSTM: 0:03:58.1545
"""

import sys
import os
import time
from argparse import ArgumentParser
from pprint import pprint

sys.path += [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]

import better_exchook
from Log import log
from Config import Config
from Util import hms_fraction, describe_returnn_version, describe_tensorflow_version
from TFEngine import Engine
from TFUtil import setup_tf_thread_pools, is_gpu_available, print_available_devices
from Dataset import init_dataset, Dataset


LstmCellTypes = [
  "BasicLSTM", "StandardLSTM",
  "LSTMBlock", "LSTMBlockFused",
  "NativeLSTM", "NativeLstm2", "NativeLstmLowMem",
  "CudnnLSTM"
]

GpuOnlyCellTypes = ["CudnnLSTM"]

# Fixed by dataset. See make_config_dict().
_input_dim = 9
_output_dim = 2

# You can play around with these. E.g. use "chunking=0", "max_seqs=20" as command-line args.
base_settings = {
  "num_layers": 5,  # number of LSTM layers
  "n_hidden": 500,  # per direction, per layer, the number of units
  "num_seqs": 500,  # for the dataset generation
  "batch_size": 2000,  # upper limit for n_seqs * max_seq_len in a batch
  "max_seqs": 40,  # upper limit for n_seqs in a batch
  "chunking": "50:25"  # set to "0" to disable
}


def make_config_dict(lstm_unit, use_gpu):
  """
  :param str lstm_unit: "NativeLSTM", "LSTMBlock", "LSTMBlockFused", "CudnnLSTM", etc, one of LstmCellTypes
  :param bool use_gpu:
  :return: config dict
  :rtype: dict[str]
  """
  num_layers = base_settings["num_layers"]
  network = {}
  for i in range(num_layers):
    for direction in [-1, 1]:
      dir_str = {-1: "bwd", 1: "fwd"}[direction]
      layer = {"class": "rec", "unit": lstm_unit, "n_out": base_settings["n_hidden"], "direction": direction}
      if i > 0:
        layer["from"] = ["lstm%i_fwd" % i, "lstm%i_bwd" % i]
      network["lstm%i_%s" % (i + 1, dir_str)] = layer
  network["output"] = {
    "class": "softmax", "loss": "ce", "target": "classes",
    "from": ["lstm%i_fwd" % num_layers, "lstm%i_bwd" % num_layers]}
  return {
    "device": "gpu" if use_gpu else "cpu",
    "train": {"class": "Task12AXDataset", "num_seqs": base_settings["num_seqs"]},
    "num_inputs": _input_dim,
    "num_outputs": _output_dim,
    "num_epochs": 1,
    "model": None,  # don't save
    "tf_log_dir": None,  # no TF logs
    "tf_log_memory_usage": True,
    "network": network,
    # batching
    "batch_size": base_settings["batch_size"],
    "max_seqs": base_settings["max_seqs"],
    "chunking": base_settings["chunking"],
    # optimization
    "adam": True,
    "learning_rate": 0.01}


def benchmark(lstm_unit, use_gpu):
  """
  :param str lstm_unit: e.g. "LSTMBlock", one of LstmCellTypes
  :param bool use_gpu:
  :return: runtime in seconds of the training itself, excluding initialization
  :rtype: float
  """
  device = {True: "GPU", False: "CPU"}[use_gpu]
  key = "%s:%s" % (device, lstm_unit)
  print(">>> Start benchmark for %s." % key)
  config = Config()
  config.update(make_config_dict(lstm_unit=lstm_unit, use_gpu=use_gpu))
  dataset_kwargs = config.typed_value("train")
  Dataset.kwargs_update_from_config(config, dataset_kwargs)
  dataset = init_dataset(dataset_kwargs)
  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=dataset)
  print(">>> Start training now for %s." % key)
  start_time = time.time()
  engine.train()
  runtime = time.time() - start_time
  print(">>> Runtime of %s: %s" % (key, hms_fraction(runtime)))
  engine.finalize()
  return runtime


def main():
  global LstmCellTypes
  print("Benchmarking LSTMs.")
  better_exchook.install()
  print("Args:", " ".join(sys.argv))
  arg_parser = ArgumentParser()
  arg_parser.add_argument("cfg", nargs="*", help="opt=value, opt in %r" % sorted(base_settings.keys()))
  arg_parser.add_argument("--no-cpu", action="store_true")
  arg_parser.add_argument("--no-gpu", action="store_true")
  arg_parser.add_argument("--selected", help="comma-separated list from %r" % LstmCellTypes)
  arg_parser.add_argument("--no-setup-tf-thread-pools", action="store_true")
  args = arg_parser.parse_args()
  for opt in args.cfg:
    key, value = opt.split("=", 1)
    assert key in base_settings
    value_type = type(base_settings[key])
    base_settings[key] = value_type(value)
  print("Settings:")
  pprint(base_settings)

  log.initialize(verbosity=[4])
  print("Returnn:", describe_returnn_version(), file=log.v3)
  print("TensorFlow:", describe_tensorflow_version(), file=log.v3)
  print("Python:", sys.version.replace("\n", ""), sys.platform)
  if not args.no_setup_tf_thread_pools:
    setup_tf_thread_pools(log_file=log.v2)
  else:
    print("Not setting up the TF thread pools. Will be done automatically by TF to number of CPU cores.")
  if args.no_gpu:
    print("GPU will not be used.")
  else:
    print("GPU available: %r" % is_gpu_available())
  print_available_devices()

  if args.selected:
    LstmCellTypes = args.selected.split(",")
  benchmarks = {}
  if not args.no_gpu and is_gpu_available():
    for lstm_unit in LstmCellTypes:
      benchmarks["GPU:" + lstm_unit] = benchmark(lstm_unit=lstm_unit, use_gpu=True)
  if not args.no_cpu:
    for lstm_unit in LstmCellTypes:
      if lstm_unit in GpuOnlyCellTypes:
        continue
      benchmarks["CPU:" + lstm_unit] = benchmark(lstm_unit=lstm_unit, use_gpu=False)

  print("-" * 20)
  print("Settings:")
  pprint(base_settings)
  print("Final results:")
  for t, lstm_unit in sorted([(t, lstm_unit) for (lstm_unit, t) in sorted(benchmarks.items())]):
    print("  %s: %s" % (lstm_unit, hms_fraction(t)))
  print("Done.")


if __name__ == "__main__":
  main()
