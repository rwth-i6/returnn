#!/usr/bin/env python3

"""
Main entry point
================

This is the main entry point. You can execute this file.
See :func:`rnn.initConfig` for some arguments, or just run ``./rnn.py --help``.
See :ref:`tech_overview` for a technical overview.
"""

from __future__ import print_function

__author__ = "Patrick Doetsch"
__copyright__ = "Copyright 2014"
__credits__ = ["Patrick Doetsch", "Paul Voigtlaender" ]
__license__ = "RWTHOCR"
__maintainer__ = "Patrick Doetsch"
__email__ = "doetsch@i6.informatik.rwth-aachen.de"


import os
import sys
import time
import numpy
from Log import log
from Device import Device, TheanoFlags, getDevicesInitArgs
from Config import Config
from Engine import Engine
from Dataset import Dataset, init_dataset, init_dataset_via_str
from HDFDataset import HDFDataset
from Debug import initIPythonKernel, initBetterExchook, initFaulthandler, initCudaNotInMainProcCheck
from Util import initThreadJoinHack, describe_crnn_version, describe_theano_version, \
  describe_tensorflow_version, BackendEngine, get_tensorflow_version_tuple


config = None; """ :type: Config """
engine = None; """ :type: TFEngine.Engine | Engine """
train_data = None; """ :type: Dataset """
dev_data = None; """ :type: Dataset """
eval_data = None; """ :type: Dataset """
quit = False
server = None; """:type: Server"""


def initConfig(configFilename=None, commandLineOptions=(), extra_updates=None):
  """
  :param str|None configFilename:
  :param list[str]|tuple[str] commandLineOptions: e.g. ``sys.argv[1:]``
  :param dict[str]|None extra_updates:

  Initializes the global config.
  There are multiple sources which are used to init the config:

    * ``configFilename``, and maybe first item of ``commandLineOptions`` interpret as config filename
    * other options via ``commandLineOptions``
    * ``extra_updates``

  Note about the order/priority of these:

    * ``extra_updates``
    * options from ``commandLineOptions``
    * ``configFilename``
    * config filename from ``commandLineOptions[0]``
    * ``extra_updates``
    * options from ``commandLineOptions``

  ``extra_updates`` and ``commandLineOptions`` are used twice so that they are available
  when the config is loaded, which thus has access to them, and can e.g. use them via Python code.
  However, the purpose is that they overwrite any option from the config;
  that is why we apply them again in the end.

  ``commandLineOptions`` is applied after ``extra_updates`` so that the user has still the possibility
  to overwrite anything set by ``extra_updates``.
  """
  global config
  config = Config()

  config_filenames_by_cmd_line = []
  if commandLineOptions:
    # Assume that the first argument prefixed with "+" or "-" and all following is not a config file.
    i = 0
    for arg in commandLineOptions:
      if arg[:1] in "-+":
        break
      config_filenames_by_cmd_line.append(arg)
      i += 1
    commandLineOptions = commandLineOptions[i:]

  if extra_updates:
    config.update(extra_updates)
  if commandLineOptions:
    config.parse_cmd_args(commandLineOptions)
  if configFilename:
    config.load_file(configFilename)
  for fn in config_filenames_by_cmd_line:
    config.load_file(fn)
  if extra_updates:
    config.update(extra_updates)
  if commandLineOptions:
    config.parse_cmd_args(commandLineOptions)

  # I really don't know where to put this otherwise:
  if config.bool("EnableAutoNumpySharedMemPickling", False):
    import TaskSystem
    TaskSystem.SharedMemNumpyConfig["enabled"] = True
  # Server default options
  if config.value('task', 'train') == 'server':
    config.set('num_inputs', 2)
    config.set('num_outputs', 1)
    #config.set('network', [{'out': {'loss': 'ce', 'class': 'softmax', 'target': 'classes'}}])


def initLog():
  logs = config.list('log', [])
  log_verbosity = config.int_list('log_verbosity', [])
  log_format = config.list('log_format', [])
  log.initialize(logs=logs, verbosity=log_verbosity, formatter=log_format)


def initConfigJsonNetwork():
  # initialize postprocess config file
  if config.has('initialize_from_json'):
    json_file = config.value('initialize_from_json', '')
    assert os.path.isfile(json_file), "json file not found: " + json_file
    print("loading network topology from json:", json_file, file=log.v5)
    config.network_topology_json = open(json_file).read()


def initDevices():
  """
  :rtype: list[Device]
  """
  oldDeviceConfig = ",".join(config.list('device', ['default']))
  if BackendEngine.is_tensorflow_selected():
    if os.environ.get("TF_DEVICE"):
      config.set("device", os.environ.get("TF_DEVICE"))
      print("Devices: Use %s via TF_DEVICE instead of %s." %
            (os.environ.get("TF_DEVICE"), oldDeviceConfig), file=log.v4)
  if not BackendEngine.is_theano_selected():
    return None
  if config.value("task", "train") == "nop":
    return []
  if "device" in TheanoFlags:
    # This is important because Theano likely already has initialized that device.
    config.set("device", TheanoFlags["device"])
    print("Devices: Use %s via THEANO_FLAGS instead of %s." % \
                     (TheanoFlags["device"], oldDeviceConfig), file=log.v4)
  devArgs = getDevicesInitArgs(config)
  assert len(devArgs) > 0
  devices = [Device(**kwargs) for kwargs in devArgs]
  for device in devices:
    while not device.initialized:
      time.sleep(0.25)
  if devices[0].blocking:
    print("Devices: Used in blocking / single proc mode.", file=log.v4)
  else:
    print("Devices: Used in multiprocessing mode.", file=log.v4)
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
  :param kwargs: passed on to init_dataset() or init_dataset_via_str()
  :rtype: (Dataset,int)
  :returns the dataset, and the cache byte size left over if we cache the whole dataset.
  """
  if not config.bool_or_other(files_config_key, None):
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
  dev_data, extra_cache_bytes_dev = load_data(
    config, cache_byte_sizes[1], 'dev', chunking=chunking, seq_ordering="sorted", shuffle_frames_of_nseqs=0)
  eval_data, extra_cache_bytes_eval = load_data(
    config, cache_byte_sizes[2], 'eval', chunking=chunking, seq_ordering="sorted", shuffle_frames_of_nseqs=0)
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
    print("Train data:", file=log.v2)
    print("  input:", train_data.num_inputs, "x", train_data.window, file=log.v2)
    print("  output:", train_data.num_outputs, file=log.v2)
    print(" ", train_data.len_info() or "no info", file=log.v2)
  if dev_data:
    print("Dev data:", file=log.v2)
    print(" ", dev_data.len_info() or "no info", file=log.v2)
  if eval_data:
    print("Eval data:", file=log.v2)
    print(" ", eval_data.len_info() or "no info", file=log.v2)

  if devices:
    print("Devices:", file=log.v3)
    for device in devices:
      print("  %s: %s" % (device.name, device.device_name), end=' ', file=log.v3)
      print("(units:", device.get_device_shaders(),
            "clock: %.02fGhz" % (device.get_device_clock() / 1024.0),
            "memory: %.01f" % (device.get_device_memory() / float(1024 * 1024 * 1024)) + "GB)", end=' ', file=log.v3)
      print("working on", device.num_batches, "batches" if device.num_batches > 1 else "batch", end=' ', file=log.v3)
      print("(update on device)" if device.update_specs['update_rule'] != 'none' else "(update on host)", file=log.v3)


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


def returnnGreeting(configFilename=None, commandLineOptions=None):
  print(
    "RETURNN starting up, version %s, date/time %s, pid %i, cwd %s, Python %s" % (
      describe_crnn_version(), time.strftime("%Y-%m-%d-%H-%M-%S (UTC%z)"), os.getpid(), os.getcwd(), sys.executable),
    file=log.v3)
  if configFilename:
    print("RETURNN config: %s" % configFilename, file=log.v4)
    if os.path.islink(configFilename):
      print("RETURNN config is symlink to: %s" % os.readlink(configFilename), file=log.v4)
  if commandLineOptions is not None:
    print("RETURNN command line options: %s" % (commandLineOptions,), file=log.v4)


def initBackendEngine():
  BackendEngine.select_engine(config=config)
  if BackendEngine.is_theano_selected():
    print("Theano:", describe_theano_version(), file=log.v3)
  elif BackendEngine.is_tensorflow_selected():
    print("TensorFlow:", describe_tensorflow_version(), file=log.v3)
    if get_tensorflow_version_tuple()[0] == 0:
      print("Warning: TF <1.0 is not supported and likely broken.", file=log.v2)
    from TFUtil import debugRegisterBetterRepr, setup_tf_thread_pools
    setup_tf_thread_pools(log_file=log.v2)
    debugRegisterBetterRepr()
  else:
    raise NotImplementedError


def init(configFilename=None, commandLineOptions=(), config_updates=None, extra_greeting=None):
  """
  :param str|None configFilename:
  :param tuple[str]|list[str]|None commandLineOptions: e.g. sys.argv[1:]
  :param dict[str]|None config_updates: see :func:`initConfig`
  :param str|None extra_greeting:
  """
  initBetterExchook()
  initThreadJoinHack()
  initConfig(configFilename=configFilename, commandLineOptions=commandLineOptions, extra_updates=config_updates)
  if config.bool("patch_atfork", False):
    from Util import maybe_restart_returnn_with_atfork_patch
    maybe_restart_returnn_with_atfork_patch()
  initLog()
  if extra_greeting:
    print(extra_greeting, file=log.v1)
  returnnGreeting(configFilename=configFilename, commandLineOptions=commandLineOptions)
  initFaulthandler()
  initBackendEngine()
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
  if config.value('task', 'train') == 'server':
    import Server
    global server
    server = Server.Server(config)
  else:
    initEngine(devices)


def finalize():
  print("Quitting", file=getattr(log, "v4", sys.stderr))
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
  if config.has("need_data") and not config.bool("need_data", True):
    return False
  task = config.value('task', 'train')
  if task in ['theano_graph', "nop", "cleanup_old_models"]:
    return False
  return True


def executeMainTask():
  from Util import hms_fraction
  start_time = time.time()
  task = config.value('task', 'train')
  if task == 'train':
    assert train_data.have_seqs(), "no train files specified, check train option: %s" % config.value('train', None)
    engine.init_train_from_config(config, train_data, dev_data, eval_data)
    engine.train()
  elif task == "eval":
    engine.init_train_from_config(config, train_data, dev_data, eval_data)
    engine.epoch = config.int("epoch", None)
    assert engine.epoch
    print("Evaluate epoch", engine.epoch, file=log.v4)
    engine.eval_model(config.value("eval_output_file", ""))
  elif task in ['forward','hpx']:
    assert eval_data is not None, 'no eval data provided'
    combine_labels = config.value('combine_labels', '')
    engine.use_search_flag = config.bool("forward_use_search", False)
    if config.has("epoch"):
      config.set('load_epoch', config.int('epoch', 0))
    engine.init_network_from_config(config)
    output_file = config.value('output_file', 'dump-fwd-epoch-%i.hdf' % engine.epoch)
    engine.forward_to_hdf(
      data=eval_data, output_file=output_file, combine_labels=combine_labels,
      batch_size=config.int('forward_batch_size', 0))
  elif task == "search":
    engine.use_search_flag = True
    engine.init_network_from_config(config)
    if config.value("search_data", "eval") in ["train", "dev", "eval"]:
      data = {"train": train_data, "dev": dev_data, "eval": eval_data}[config.value("search_data", "eval")]
      assert data, "set search_data"
    else:
      data = init_dataset(config.opt_typed_value("search_data"))
    engine.search(
      data,
      do_eval=config.bool("search_do_eval", True),
      output_layer_name=config.value("search_output_layer", "output"),
      output_file=config.value("search_output_file", ""),
      output_file_format=config.value("search_output_file_format", "txt"))
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
      print("dumping to %s.* ..." % prefix, file=log.v1)
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
  elif task == "hyper_param_tuning":
    import HyperParamTuning
    tuner = HyperParamTuning.Optimization(config=config, train_data=train_data)
    tuner.work()
  elif task == "cleanup_old_models":
    engine.cleanup_old_models(ask_for_confirmation=True)
  elif task == "daemon":
    engine.init_network_from_config(config)
    engine.daemon(config)
  elif task == "server":
    print("Server Initiating", file=log.v1)
    server.run()
  elif task == "search_server":
    engine.use_search_flag = True
    engine.init_network_from_config(config)
    engine.web_server(port=config.int("web_server_port", 12380))
  elif task.startswith("config:"):
    action = config.typed_dict[task[len("config:"):]]
    print("Task: %r" % action, file=log.v1)
    assert callable(action)
    action()
  elif task.startswith("optional-config:"):
    action = config.typed_dict.get(task[len("optional-config:"):], None)
    if action is None:
      print("No task found for %r, so just quitting." % task, file=log.v1)
    else:
      print("Task: %r" % action, file=log.v1)
      assert callable(action)
      action()
  elif task == "nop":
    print("Task: No-operation", file=log.v1)
  else:
    assert False, "unknown task: %s" % task

  print(("elapsed: %s" % hms_fraction(time.time() - start_time)), file=log.v3)


def analyze_data(config):
  """
  :param Config config:
  """
  dss = config.value('analyze_dataset', 'train')
  ds = {"train": train_data, "dev": dev_data, "eval": eval_data}[dss]
  epoch = config.int('epoch', 1)
  print("Analyze dataset", dss, "epoch", epoch, file=log.v1)
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

  # Note: This is not stable! See :class:`Util.Stats` for a better alternative.
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
  std_dev = numpy.sqrt(mean_sq - mean * mean)
  print("Finished. %i total target frames, %i total data frames" % (total_targets_len, total_data_len), file=log.v1)
  priors_fn = stat_prefix + ".log_priors.txt"
  mean_fn = stat_prefix + ".mean.txt"
  std_dev_fn = stat_prefix + ".std_dev.txt"
  print("Dump priors to", priors_fn, file=log.v1)
  numpy.savetxt(priors_fn, log_priors)
  print("Dump mean to", mean_fn, file=log.v1)
  numpy.savetxt(mean_fn, mean)
  print("Dump std dev to", std_dev_fn, file=log.v1)
  numpy.savetxt(std_dev_fn, std_dev)
  print("Done.", file=log.v1)


def main(argv):
  return_code = 0
  try:
    assert len(argv) >= 2, "usage: %s <config>" % argv[0]
    init(commandLineOptions=argv[1:])
    executeMainTask()
  except KeyboardInterrupt:
    return_code = 1
    print("KeyboardInterrupt", file=getattr(log, "v3", sys.stderr))
    if getattr(log, "verbose", [False] * 6)[5]:
      sys.excepthook(*sys.exc_info())
  finalize()
  if return_code:
    sys.exit(return_code)


if __name__ == '__main__':
  main(sys.argv)
