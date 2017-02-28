#!/usr/bin/env python2.7

"""
This is the main entry point. You can execute this file.
"""

__author__ = "Patrick Doetsch"
__copyright__ = "Copyright 2014"
__credits__ = ["Patrick Doetsch", "Paul Voigtlaender" ]
__license__ = "RWTHOCR"
__maintainer__ = "Patrick Doetsch"
__email__ = "doetsch@i6.informatik.rwth-aachen.de"


import re
import os
import sys
import time
import json
import numpy
from optparse import OptionParser
from Log import log
from Device import Device, get_num_devices, TheanoFlags, getDevicesInitArgs
from Config import Config
from Engine import Engine
from Dataset import Dataset, init_dataset, init_dataset_via_str, get_dataset_class
from HDFDataset import HDFDataset
from Debug import initIPythonKernel, initBetterExchook, initFaulthandler, initCudaNotInMainProcCheck
from Util import initThreadJoinHack, custom_exec, describe_crnn_version, describe_theano_version, \
  describe_tensorflow_version, BackendEngine


config = None; """ :type: Config """
engine = None; """ :type: Engine | TFEngine.Engine """
train_data = None; """ :type: Dataset """
dev_data = None; """ :type: Dataset """
eval_data = None; """ :type: Dataset """
quit = False


def initConfig(configFilename=None, commandLineOptions=()):
  """
  :type configFilename: str
  :type commandLineOptions: list[str]
  Initializes the global config.
  """
  global config
  config = Config()
  if configFilename:
    assert os.path.isfile(configFilename), "config file not found"
    config.load_file(configFilename)
  if commandLineOptions and commandLineOptions[0][:1] not in ["-", "+"]:
    # Assume that this is a config filename.
    config.load_file(commandLineOptions[0])
    commandLineOptions = commandLineOptions[1:]
  parser = OptionParser()
  parser.add_option("-a", "--activation", dest = "activation", help = "[STRING/LIST] Activation functions: logistic, tanh, softsign, relu, identity, zero, one, maxout.")
  parser.add_option("-b", "--batch_size", dest = "batch_size", help = "[INTEGER/TUPLE] Maximal number of frames per batch (optional: shift of batching window).")
  parser.add_option("-c", "--chunking", dest = "chunking", help = "[INTEGER/TUPLE] Maximal number of frames per sequence (optional: shift of chunking window).")
  parser.add_option("-d", "--description", dest = "description", help = "[STRING] Description of experiment.")
  parser.add_option("-e", "--epoch", dest = "epoch", help = "[INTEGER] Starting epoch.")
  parser.add_option("-E", "--eval", dest = "eval", help = "[STRING] eval file path")
  parser.add_option("-f", "--gate_factors", dest = "gate_factors", help = "[none/local/global] Enables pooled (local) or separate (global) coefficients on gates.")
  parser.add_option("-g", "--lreg", dest = "lreg", help = "[FLOAT] L1 or L2 regularization.")
  parser.add_option("-i", "--save_interval", dest = "save_interval", help = "[INTEGER] Number of epochs until a new model will be saved.")
  parser.add_option("-j", "--dropout", dest = "dropout", help = "[FLOAT] Dropout probability (0 to disable).")
  #parser.add_option("-k", "--multiprocessing", dest = "multiprocessing", help = "[BOOLEAN] Enable multi threaded processing (required when using multiple devices).")
  parser.add_option("-k", "--output_file", dest = "output_file", help = "[STRING] Path to target file for network output.")
  parser.add_option("-l", "--log", dest = "log", help = "[STRING] Log file path.")
  parser.add_option("-L", "--load", dest = "load", help = "[STRING] load model file path.")
  parser.add_option("-m", "--momentum", dest = "momentum", help = "[FLOAT] Momentum term in gradient descent optimization.")
  parser.add_option("-n", "--num_epochs", dest = "num_epochs", help = "[INTEGER] Number of epochs that should be trained.")
  parser.add_option("-o", "--order", dest = "order", help = "[default/sorted/random] Ordering of sequences.")
  parser.add_option("-p", "--loss", dest = "loss", help = "[loglik/sse/ctc] Objective function to be optimized.")
  parser.add_option("-q", "--cache", dest = "cache", help = "[INTEGER] Cache size in bytes (supports notation for kilo (K), mega (M) and gigabtye (G)).")
  parser.add_option("-r", "--learning_rate", dest = "learning_rate", help = "[FLOAT] Learning rate in gradient descent optimization.")
  parser.add_option("-s", "--hidden_sizes", dest = "hidden_sizes", help = "[INTEGER/LIST] Number of units in hidden layers.")
  parser.add_option("-t", "--truncate", dest = "truncate", help = "[INTEGER] Truncates sequence in BPTT routine after specified number of timesteps (-1 to disable).")
  parser.add_option("-u", "--device", dest = "device", help = "[STRING/LIST] CPU and GPU devices that should be used (example: gpu0,cpu[1-6] or gpu,cpu*).")
  parser.add_option("-v", "--verbose", dest = "log_verbosity", help = "[INTEGER] Verbosity level from 0 - 5.")
  parser.add_option("-w", "--window", dest = "window", help = "[INTEGER] Width of sliding window over sequence.")
  parser.add_option("-x", "--task", dest = "task", help = "[train/forward/analyze] Task of the current program call.")
  parser.add_option("-y", "--hidden_type", dest = "hidden_type", help = "[VALUE/LIST] Hidden layer types: forward, recurrent, lstm.")
  parser.add_option("-z", "--max_sequences", dest = "max_seqs", help = "[INTEGER] Maximal number of sequences per batch.")
  parser.add_option("--config", dest="load_config", help="[STRING] load config")
  (options, args) = parser.parse_args(commandLineOptions)
  options = vars(options)
  for opt in options.keys():
    if options[opt] is not None:
      if opt == "load_config":
        config.load_file(options[opt])
      else:
        config.add_line(opt, options[opt])
  assert len(args) % 2 == 0, "expect (++key, value) config tuples in remaining args: %r" % args
  for i in range(0, len(args), 2):
    key, value = args[i:i+2]
    assert key[0:2] == "++", "expect key prefixed with '++' in (%r, %r)" % (key, value)
    if value[:2] == "+-": value = value[1:]  # otherwise we never could specify things like "++threshold -0.1"
    config.add_line(key=key[2:], value=value)
  # I really don't know where to put this otherwise:
  if config.bool("EnableAutoNumpySharedMemPickling", False):
    import TaskSystem
    TaskSystem.SharedMemNumpyConfig["enabled"] = True


def initLog():
  logs = config.list('log', [])
  log_verbosity = config.int_list('log_verbosity', [])
  log_format = config.list('log_format', [])
  log.initialize(logs = logs, verbosity = log_verbosity, formatter = log_format)


def initConfigJsonNetwork():
  # initialize postprocess config file
  if config.has('initialize_from_json'):
    json_file = config.value('initialize_from_json', '')
    assert os.path.isfile(json_file), "json file not found: " + json_file
    print >> log.v5, "loading network topology from json:", json_file
    config.network_topology_json = open(json_file).read().encode('utf8')


def initDevices():
  """
  :rtype: list[Device]
  """
  oldDeviceConfig = ",".join(config.list('device', ['default']))
  if BackendEngine.is_tensorflow_selected():
    if os.environ.get("TF_DEVICE"):
      config.set("device", os.environ.get("TF_DEVICE"))
      print >> log.v4, "Devices: Use %s via TF_DEVICE instead of %s." % \
                       (os.environ.get("TF_DEVICE"), oldDeviceConfig)
  if not BackendEngine.is_theano_selected():
    return None
  if "device" in TheanoFlags:
    # This is important because Theano likely already has initialized that device.
    config.set("device", TheanoFlags["device"])
    print >> log.v4, "Devices: Use %s via THEANO_FLAGS instead of %s." % \
                     (TheanoFlags["device"], oldDeviceConfig)
  devArgs = getDevicesInitArgs(config)
  assert len(devArgs) > 0
  devices = [Device(**kwargs) for kwargs in devArgs]
  for device in devices:
    while not device.initialized:
      time.sleep(0.25)
  if devices[0].blocking:
    print >> log.v4, "Devices: Used in blocking / single proc mode."
  else:
    print >> log.v4, "Devices: Used in multiprocessing mode."
  return devices


def getCacheByteSizes():
  """
  :rtype: (int,int,int)
  :returns cache size in bytes for (train,dev,eval)
  """
  import Util
  cache_sizes_user = config.list('cache_size', ["%iG" % Util.defaultCacheSizeInGBytes()])
  num_datasets = 1 + config.has('dev') + config.has('eval')
  cache_factor = 1.0
  if len(cache_sizes_user) == 1:
    cache_sizes_user *= 3
    cache_factor /= float(num_datasets)
  elif len(cache_sizes_user) == 2:
    cache_sizes_user.append('0')
  assert len(cache_sizes_user) == 3, "invalid amount of cache sizes specified"
  cache_sizes = []
  for cache_size_user in cache_sizes_user:
    cache_size = cache_factor * float(cache_size_user.replace('G', '').replace('M', '').replace('K', ''))
    assert len(cache_size_user) - len(str(cache_size)) <= 1, "invalid cache size specified"
    if cache_size_user.find('G') > 0:
      cache_size *= 1024 * 1024 * 1024
    elif cache_size_user.find('M') > 0:
      cache_size *= 1024 * 1024
    elif cache_size_user.find('K') > 0:
      cache_size *= 1024
    cache_size = int(cache_size) + 1 if int(cache_size) > 0 else 0
    cache_sizes.append(cache_size)
  return cache_sizes


def load_data(config, cache_byte_size, files_config_key, **kwargs):
  """
  :param Config config:
  :param int cache_byte_size:
  :param str files_config_key: such as "train" or "dev"
  :param dict[str] kwargs: passed on to init_dataset() or init_dataset_via_str()
  :rtype: (Dataset,int)
  :returns the dataset, and the cache byte size left over if we cache the whole dataset.
  """
  if not config.has(files_config_key):
    return None, 0
  kwargs = kwargs.copy()
  kwargs.setdefault("name", files_config_key)
  if config.is_typed(files_config_key) and isinstance(config.typed_value(files_config_key), dict):
    config_opts = config.typed_value(files_config_key)
    assert isinstance(config_opts, dict)
    kwargs.update(config_opts)
    if 'cache_byte_size' not in config_opts:
      if kwargs.get('class', None) == 'HDFDataset':
        kwargs["cache_byte_size"] = cache_byte_size
    Dataset.kwargs_update_from_config(config, kwargs)
    data = init_dataset(kwargs)
  else:
    config_str = config.value(files_config_key, "")
    data = init_dataset_via_str(config_str, config=config, cache_byte_size=cache_byte_size, **kwargs)
  cache_leftover = 0
  if isinstance(data, HDFDataset):
    cache_leftover = data.definite_cache_leftover
  return data, cache_leftover


def initData():
  """
  Initializes the globals train,dev,eval of type Dataset.
  """
  cache_byte_sizes = getCacheByteSizes()
  chunking = "0"
  if config.value("on_size_limit", "ignore") == "chunk":
    chunking = config.value("batch_size", "0")
  elif config.value('chunking', "0") == "1": # MLP mode
    chunking = "1"
  global train_data, dev_data, eval_data
  dev_data, extra_cache_bytes_dev = load_data(config, cache_byte_sizes[1], 'dev', chunking=chunking,
                                         seq_ordering="sorted", shuffle_frames_of_nseqs=0)
  eval_data, extra_cache_bytes_eval = load_data(config, cache_byte_sizes[2], 'eval', chunking=chunking,
                                           seq_ordering="sorted", shuffle_frames_of_nseqs=0)
  train_cache_bytes = cache_byte_sizes[0]
  if train_cache_bytes >= 0:
    # Maybe we have left over cache from dev/eval if dev/eval have cached everything.
    train_cache_bytes += extra_cache_bytes_dev + extra_cache_bytes_eval
  train_data, extra_train = load_data(config, train_cache_bytes, 'train')


def printTaskProperties(devices=None):
  """
  :type devices: list[Device]
  """

  if train_data:
    print >> log.v2, "Train data:"
    print >> log.v2, "  input:", train_data.num_inputs, "x", train_data.window
    print >> log.v2, "  output:", train_data.num_outputs
    print >> log.v2, " ", train_data.len_info() or "no info"
  if dev_data:
    print >> log.v2, "Dev data:"
    print >> log.v2, " ", dev_data.len_info() or "no info"
  if eval_data:
    print >> log.v2, "Eval data:"
    print >> log.v2, " ", eval_data.len_info() or "no info"

  if devices:
    print >> log.v3, "Devices:"
    for device in devices:
      print >> log.v3, "  %s: %s" % (device.name, device.device_name),
      print >> log.v3, "(units:", device.get_device_shaders(), \
                       "clock: %.02fGhz" % (device.get_device_clock() / 1024.0), \
                       "memory: %.01f" % (device.get_device_memory() / float(1024 * 1024 * 1024)) + "GB)",
      print >> log.v3, "working on", device.num_batches, "batches" if device.num_batches > 1 else "batch",
      print >> log.v3, "(update on device)" if device.update_specs['update_rule'] != 'none' else "(update on host)"


def initEngine(devices):
  """
  :type devices: list[Device]
  Initializes global engine.
  """
  global engine
  if BackendEngine.is_theano_selected():
    engine = Engine(devices)
  elif BackendEngine.is_tensorflow_selected():
    import TFEngine
    engine = TFEngine.Engine(config=config)
  else:
    raise NotImplementedError


def crnnGreeting():
  print >> log.v3, "CRNN starting up, version %s, pid %i" % (describe_crnn_version(), os.getpid())


def initBackendEngine():
  BackendEngine.select_engine(config=config)
  if BackendEngine.is_theano_selected():
    print >> log.v3, "Theano:", describe_theano_version()
  elif BackendEngine.is_tensorflow_selected():
    print >> log.v3, "TensorFlow:", describe_tensorflow_version()
    from Util import to_bool
    from TFUtil import debugRegisterBetterRepr
    if os.environ.get("DEBUG_TF_BETTER_REPR") and to_bool(os.environ.get("DEBUG_TF_BETTER_REPR")):
      debugRegisterBetterRepr()
  else:
    raise NotImplementedError


def init(configFilename=None, commandLineOptions=()):
  initBetterExchook()
  initThreadJoinHack()
  initConfig(configFilename=configFilename, commandLineOptions=commandLineOptions)
  initLog()
  crnnGreeting()
  initBackendEngine()
  initFaulthandler()
  if BackendEngine.is_theano_selected():
    if config.value('task', 'train') == "theano_graph":
      config.set("multiprocessing", False)
    if config.bool('multiprocessing', True):
      initCudaNotInMainProcCheck()
  if config.bool('ipython', False):
    initIPythonKernel()
  initConfigJsonNetwork()
  devices = initDevices()
  if needData():
    initData()
  printTaskProperties(devices)
  initEngine(devices)


def finalize():
  global quit
  quit = True
  sys.exited = True
  if BackendEngine.is_theano_selected():
    if engine:
      for device in engine.devices:
        device.terminate()
  elif BackendEngine.is_tensorflow_selected():
    if engine:
      engine.finalize()

def needData():
  task = config.value('task', 'train')
  if task == 'theano_graph':
    return False
  return True


def executeMainTask():
  st = time.time()
  task = config.value('task', 'train')
  if task == 'train':
    assert train_data.have_seqs(), "no train files specified, check train option: %s" % config.value('train', None)
    engine.init_train_from_config(config, train_data, dev_data, eval_data)
    engine.train()
  elif task == "eval":
    engine.init_train_from_config(config, train_data, dev_data, eval_data)
    engine.epoch = config.int("epoch", None)
    assert engine.epoch
    print >> log.v4, "Evaluate epoch", engine.epoch
    engine.eval_model()
  elif task == 'forward':
    assert eval_data is not None, 'no eval data provided'
    assert config.has('output_file'), 'no output file provided'
    combine_labels = config.value('combine_labels', '')
    output_file = config.value('output_file', '')
    engine.init_network_from_config(config)
    engine.forward_to_hdf(
      data=eval_data, output_file=output_file, combine_labels=combine_labels,
      batch_size=config.int('forward_batch_size', 0))
  elif task == 'compute_priors':
    assert train_data is not None, 'train data for priors should be provided'
    engine.init_network_from_config(config)
    engine.compute_priors(dataset=train_data, config=config)
  elif task == 'theano_graph':
    import theano.printing
    import theano.compile.io
    import theano.compile.function_module
    engine.start_epoch = 1
    engine.init_network_from_config(config)
    for task in config.list('theano_graph.task', ['train']):
      func = engine.devices[-1].get_compute_func(task)
      prefix = config.value("theano_graph.prefix", "current") + ".task"
      print >>log.v1, "dumping to %s.* ..." % prefix
      theano.printing.debugprint(func, file=open("%s.optimized_func.txt" % prefix, "w"))
      assert isinstance(func.maker, theano.compile.function_module.FunctionMaker)
      for inp in func.maker.inputs:
        assert isinstance(inp, theano.compile.io.In)
        if inp.update:
          theano.printing.debugprint(inp.update, file=open("%s.unoptimized.var_%s_update.txt" % (prefix, inp.name), "w"))
      theano.printing.pydotprint(func, format='png', var_with_name_simple=True,
                                 outfile = "%s.png" % prefix)
  elif task == 'analyze':  # anything based on the network + Device
    statistics = config.list('statistics', None)
    engine.init_network_from_config(config)
    engine.analyze(data=eval_data or dev_data, statistics=statistics)
  elif task == "analyze_data":  # anything just based on the data
    analyze_data(config)
  elif task == "classify":
    assert eval_data is not None, 'no eval data provided'
    assert config.has('label_file'), 'no output file provided'
    label_file = config.value('label_file', '')
    engine.init_network_from_config(config)
    engine.classify(engine.devices[0], eval_data, label_file)
  elif task == "daemon":
    engine.init_network_from_config(config)
    engine.daemon()
  else:
    assert False, "unknown task: %s" % task

  print >> log.v3, ("elapsed: %f" % (time.time() - st))


def analyze_data(config):
  dss = config.value('analyze_dataset', 'train')
  ds = {"train": train_data, "dev": dev_data, "eval": eval_data}[dss]
  epoch = config.int('epoch', 1)
  print >> log.v1, "Analyze dataset", dss, "epoch", epoch
  ds.init_seq_order(epoch=epoch)
  stat_prefix = config.value('statistics_save_prefix', 'statistics')
  dtype = config.value('statistics_dtype', 'float64')
  target = config.value('target', 'classes')
  data_key = config.value('data_key', 'data')
  assert ds.is_data_sparse(target), "need for prior calculation"
  assert not ds.is_data_sparse(data_key), "needed for mean/var estimation"
  from Util import inplace_increment, progress_bar_with_time, NumbersDict

  priors = numpy.zeros((ds.get_data_dim(target),), dtype=dtype)
  mean = numpy.zeros((ds.get_data_dim(data_key),), dtype=dtype)
  mean_sq = numpy.zeros((ds.get_data_dim(data_key),), dtype=dtype)
  total_targets_len = 0
  total_data_len = 0

  seq_idx = 0
  while ds.is_less_than_num_seqs(seq_idx):
    progress_bar_with_time(ds.get_complete_frac(seq_idx))
    ds.load_seqs(seq_idx, seq_idx + 1)
    targets = ds.get_data(seq_idx, target)
    inplace_increment(priors, targets, 1)
    total_targets_len += targets.shape[0]
    data = ds.get_data(seq_idx, data_key)
    new_total_data_len = total_data_len + data.shape[0]
    f = float(total_data_len) / new_total_data_len
    mean = mean * f + numpy.sum(data, axis=0) * (1.0 - f)
    mean_sq = mean_sq * f + numpy.sum(data * data, axis=0) * (1.0 - f)
    total_data_len = new_total_data_len
    seq_idx += 1
  log_priors = numpy.log(priors)
  log_priors -= numpy.log(NumbersDict(ds.get_num_timesteps())[target])
  var = numpy.sqrt(mean_sq - mean * mean)
  print >> log.v1, "Finished. %i total target frames, %i total data frames" % (total_targets_len, total_data_len)
  priors_fn = stat_prefix + ".log_priors.txt"
  mean_fn = stat_prefix + ".mean.txt"
  var_fn = stat_prefix + ".var.txt"
  print >> log.v1, "Dump priors to", priors_fn
  numpy.savetxt(priors_fn, log_priors)
  print >> log.v1, "Dump mean to", mean_fn
  numpy.savetxt(mean_fn, mean)
  print >> log.v1, "Dump var to", var_fn
  numpy.savetxt(var_fn, var)
  print >> log.v1, "Done."


def main(argv):
  return_code = 0
  try:
    assert len(argv) >= 2, "usage: %s <config>" % argv[0]
    init(commandLineOptions=argv[1:])
    executeMainTask()
  except KeyboardInterrupt:
    return_code = 1
    print >> getattr(log, "v3", sys.stderr), "KeyboardInterrupt"
  finalize()
  if return_code:
    sys.exit(return_code)


if __name__ == '__main__':
  main(sys.argv)
