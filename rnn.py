#!/usr/bin/env python2.7

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
from optparse import OptionParser
from Log import log
from Device import Device, get_num_devices
from Config import Config
from Engine import Engine
from Dataset import Dataset, init_dataset, init_dataset_via_str, get_dataset_class
from HDFDataset import HDFDataset
from Debug import initIPythonKernel, initBetterExchook, initFaulthandler
from Util import initThreadJoinHack
from SprintCommunicator import SprintCommunicator


TheanoFlags = {key: value for (key, value) in [s.split("=", 1) for s in os.environ.get("THEANO_FLAGS", "").split(",") if s]}

config = None; """ :type: Config """
engine = None; """ :type: Engine """
train_data = None; """ :type: Dataset """
dev_data = None; """ :type: Dataset """
eval_data = None; """ :type: Dataset """
quit = False


def initConfig(configFilename, commandLineOptions):
  """
  :type configFilename: str
  :type commandLineOptions: list[str]
  Inits the global config.
  """
  assert os.path.isfile(configFilename), "config file not found"
  global config
  config = Config()
  config.load_file(configFilename)
  parser = OptionParser()
  parser.add_option("-a", "--activation", dest = "activation", help = "[STRING/LIST] Activation functions: logistic, tanh, softsign, relu, identity, zero, one, maxout.")
  parser.add_option("-b", "--batch_size", dest = "batch_size", help = "[INTEGER/TUPLE] Maximal number of frames per batch (optional: shift of batching window).")
  parser.add_option("-c", "--chunking", dest = "chunking", help = "[INTEGER/TUPLE] Maximal number of frames per sequence (optional: shift of chunking window).")
  parser.add_option("-d", "--description", dest = "description", help = "[STRING] Description of experiment.")
  parser.add_option("-e", "--epoch", dest = "epoch", help = "[INTEGER] Starting epoch.")
  parser.add_option("-f", "--gate_factors", dest = "gate_factors", help = "[none/local/global] Enables pooled (local) or separate (global) coefficients on gates.")
  parser.add_option("-g", "--lreg", dest = "lreg", help = "[FLOAT] L1 or L2 regularization.")
  parser.add_option("-i", "--save_interval", dest = "save_interval", help = "[INTEGER] Number of epochs until a new model will be saved.")
  parser.add_option("-j", "--dropout", dest = "dropout", help = "[FLOAT] Dropout probability (0 to disable).")
  #parser.add_option("-k", "--multiprocessing", dest = "multiprocessing", help = "[BOOLEAN] Enable multi threaded processing (required when using multiple devices).")
  parser.add_option("-k", "--output_file", dest = "output_file", help = "[STRING] Path to target file for network output.")
  parser.add_option("-l", "--log", dest = "log", help = "[STRING] Log file path.")
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
  (options, args) = parser.parse_args(commandLineOptions)
  options = vars(options)
  for opt in options.keys():
    if options[opt] != None:
      config.add_line(opt, options[opt])
  assert len(args) % 2 == 0, "expect (++key, value) config tuples in remaining args: %r" % args
  for i in range(0, len(args), 2):
    key, value = args[i:i+2]
    assert key[0:2] == "++", "expect key prefixed with '++' in (%r, %r)" % (key, value)
    config.add_line(key=key[2:], value=value)

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


def maybeInitSprintCommunicator(device_proc):
  # initialize SprintCommunicator (if required)
  multiproc = config.bool('multiprocessing', True)
  if config.has('sh_mem_key') and ((not device_proc and not multiproc) or (device_proc and multiproc)):
    SprintCommunicator.instance = SprintCommunicator(config.int('sh_mem_key',-1))


def maybeFinalizeSprintCommunicator(device_proc):
  multiproc = config.bool('multiprocessing', True)
  if SprintCommunicator.instance is not None and ((not device_proc and not multiproc) or (device_proc and multiproc)):
    SprintCommunicator.instance.finalize()


def getDevicesInitArgs(config):
  """
  :type config: Config
  :rtype: list[dict[str]]
  """
  multiproc = config.bool('multiprocessing', True)
  if config.value('task', 'train') == "theano_graph":
    multiproc = False
  device_info = config.list('device', ['cpu0'])
  if len(device_info) == 1 and device_info[0] == 'json':
    try:
      specs = json.loads(open(config.value('initialize_from_json', '')).read())['worker']
    except Exception:
      raise Exception('Unable to parse worker information from json content')
    devices = [ { 'device' : specs[key]['device'], 'config' : config, 'blocking' : False, 'num_batches' : specs[key].pop('num_batches', 1), "update_specs" : specs[key].pop('update_specs', {}) } for key in specs ]
  else:
    device_tags = {}
    ncpus, ngpus = get_num_devices()
    if "all" in device_info:
      device_tags = { tag: [1,True] for tag in [ "cpu" + str(i) for i in xrange(ncpus)] + [ "gpu" + str(i) for i in xrange(ngpus)] }
    else:
      for info in device_info:
        device_update = True
        num_batches = 1
        if info[0] == '_':
          device_update = False
          info = info[1:]
        if ':' in info:
          num_batches = int(info.split(':')[1])
          info = info.split(':')[0]
        if len(info) == 3: info += "X"
        assert len(info) > 3, "invalid device: " + str(info) #str(info[:-1])
        utype = info[0:3]
        uid = info[3:]
        if uid == '*': uid = "[0-9]*"
        if uid == 'X': device_tags[info] = [num_batches, True]
        else:
          if utype == 'cpu':
            np = ncpus
          elif utype == 'gpu':
            np = ngpus
          else:
            np = 0
          match = False
          for p in xrange(np):
            if re.match(uid, str(p)):
              device_tags[utype + str(p)] = [num_batches, device_update]
              match = True
          assert match, "invalid device specified: " + info
    tags = sorted(device_tags.keys())
    if multiproc:
      assert len(tags) > 0
      if len(tags) == 1 and tags[0][-1] == 'X':
        newtag = tags[0][:-1] + 'Z'
        device_tags[newtag] = device_tags[tags[0]]
        tags[0] = newtag
      devices = [ {"device": tag, "config": config, "num_batches": device_tags[tag][0], "update_specs" : {'update_rule' : 'global' if device_tags[tag][1] else 'none'}} for tag in tags ]
      import TaskSystem
      if TaskSystem.isMainProcess:  # On a child process, we can have the gpu device.
        assert not TheanoFlags.get("device", "").startswith("gpu"), \
            "The main proc is not supposed to use the GPU in multiprocessing mode. Do not set device=gpu in THEANO_FLAGS."
    else:
      devices = [ {"device": tags[0], "config": config, "blocking": True} ]
    #if config.value("on_size_limit", "ignore") == "cpu" and devices[-1]["device"] != "cpu127":
    #  devices.append({"device": "cpu127", "config": config})
  return devices


def isUpdateOnDevice(config):
  """
  :type config: Config
  :rtype: bool
  """
  devArgs = getDevicesInitArgs(config)
  if config.value("update_on_device", "auto") == "auto":
    return len(devArgs) == 1
  updateOnDevice = config.bool("update_on_device", None)
  assert isinstance(updateOnDevice, bool)
  if updateOnDevice:
    assert len(devArgs) == 1, "Devices: update_on_device works only with a single device."
  return updateOnDevice


def initDevices():
  """
  :rtype: list[Device]
  """
  if "device" in TheanoFlags:
    # This is important because Theano likely already has initialized that device.
    oldDeviceConfig = ",".join(config.list('device', ['default']))
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
  cache_sizes_user = config.list('cache_size', ["1024G"])
  num_datasets = 1 + config.has('dev') + config.has('eval')
  cache_factor = 1.0
  if len(cache_sizes_user) == 1:
    cache_sizes_user *= 3
    cache_factor /= float(num_datasets)
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
  :type config: Config
  :type cache_byte_size: int
  :type chunking: str
  :type seq_ordering: str
  :rtype: (Dataset,int)
  :returns the dataset, and the cache byte size left over if we cache the whole dataset.
  """
  if not config.has(files_config_key):
    return None, 0
  if config.is_typed(files_config_key) and isinstance(config.typed_value(files_config_key), dict):
    new_kwargs = config.typed_value(files_config_key)
    assert isinstance(new_kwargs, dict)
    kwargs.update(new_kwargs)
    if 'cache_byte_size' not in new_kwargs:
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
  engine = Engine(devices)


def init(configFilename, commandLineOptions):
  initBetterExchook()
  initThreadJoinHack()
  initConfig(configFilename, commandLineOptions)
  initLog()
  print >> log.v3, "CRNN starting up, pid %i" % os.getpid()
  initFaulthandler()
  if config.bool('ipython', False):
    initIPythonKernel()
  initConfigJsonNetwork()
  maybeInitSprintCommunicator(device_proc=False)
  devices = initDevices()
  if needData():
    initData()
  printTaskProperties(devices)
  initEngine(devices)


def finalize():
  global quit
  quit = True
  sys.exited = True
  if engine:
    for device in engine.devices:
      device.terminate()
  maybeFinalizeSprintCommunicator(device_proc=False)


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
  elif task == 'forward':
    assert eval_data is not None, 'no eval data provided'
    assert config.has('output_file'), 'no output file provided'
    combine_labels = config.value('combine_labels', '')
    output_file = config.value('output_file', '')
    engine.init_network_from_config(config)
    engine.forward_to_hdf(eval_data, output_file, combine_labels)
  elif task == 'theano_graph':
    import theano.printing
    engine.start_epoch = 1
    engine.init_network_from_config(config)
    for task in config.list('theano_graph.task', ['train']):
      theano.printing.debugprint(engine.devices[-1].get_compute_func(task))
      theano.printing.pydotprint(engine.devices[-1].get_compute_func(task), format = 'png', var_with_name_simple = True,
                                 outfile = config.value("theano_graph.prefix", "current") + "." + task + ".png")
  elif task == 'analyze':
    statistics = config.list('statistics', ['confusion_matrix'])
    engine.init_network_from_config(config)
    engine.analyze(engine.devices[0], eval_data, statistics)
  elif task == "classify":
    assert eval_data is not None, 'no eval data provided'
    assert config.has('label_file'), 'no output file provided'
    label_file = config.value('label_file', '')
    engine.init_network_from_config(config)
    engine.classify(engine.devices[0], eval_data, label_file)
  else:
    assert False, "unknown task: %s" % task

  print >> log.v3, ("elapsed: %f" % (time.time() - st))


def main(argv):
  return_code = 0
  try:
    assert len(argv) >= 2, "usage: %s <config>" % argv[0]
    init(configFilename=argv[1], commandLineOptions=argv[2:])
    executeMainTask()
  except KeyboardInterrupt:
    return_code = 1
    print >> getattr(log, "v3", sys.stderr), "KeyboardInterrupt"
  finalize()
  if return_code:
    sys.exit(return_code)


if __name__ == '__main__':
  main(sys.argv)
