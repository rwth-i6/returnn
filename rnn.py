#! /usr/bin/python2.7

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
from optparse import OptionParser
import h5py
import json
import numpy
from Log import log
from Device import Device, get_num_devices
from Config import Config
from Engine import Engine
from Dataset import Dataset
from SprintCommunicator import SprintCommunicator


TheanoFlags = {key: value for (key, value) in [s.split("=", 1) for s in os.environ.get("THEANO_FLAGS", "").split(",") if s]}

config = None; """ :type: Config """
engine = None; """ :type: Engine """
last_epoch = 0  # For training. Loaded from NN model. This is the last epoch we trained on.
train = None; """ :type: Dataset """
dev   = None; """ :type: Dataset """
eval  = None; """ :type: Dataset """


def subtract_priors(network, train, config):
  if config.bool('subtract_priors', False):
    prior_scale = config.float('prior_scale', 0.0)
    priors = train.calculate_priori()
    priors[priors == 0] = 1e-10 #avoid priors of zero which would yield a bias of inf
    l = [p for p in network.params if p.name == 'b_softmax']
    assert len(l) == 1, len(l)
    b_softmax = l[0]
    b_softmax.set_value(b_softmax.get_value() - prior_scale * numpy.log(priors))


def initConfig(configFilename, commandLineOptions):
  """
  :type str configFilename:
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
      config.set(opt, options[opt])

def initLog():
  logs = config.list('log', [])
  log_verbosity = config.int_list('log_verbosity', [])
  log_format = config.list('log_format', [])
  log.initialize(logs = logs, verbosity = log_verbosity, formatter = log_format)

def initConfigJson():
  # initialize postprocess config file
  if config.has('initialize_from_json'):
    json_file = config.value('initialize_from_json', '')
    assert os.path.isfile(json_file), "json file not found: " + json_file
    print >> log.v5, "loading network topology from json:", json_file
    config.json = open(json_file).read()

def maybeInitSprintCommunicator():
  # initialize SprintCommunicator (if required)
  if config.has('sh_mem_key'):
    SprintCommunicator.instance = SprintCommunicator(config.int('sh_mem_key',-1))

def initDevices():
  """
  :rtype: list[Device]
  """
  multiproc = config.bool('multiprocessing', True)
  device_info = config.list('device', ['cpu0'])
  if "device" in TheanoFlags:
    if multiproc:
      assert not TheanoFlags["device"].startswith("gpu"), "the main proc is not supposed to use the GPU"
    else:
      print >> log.v2, "devices: use %s via THEANO_FLAGS instead of %s" % (TheanoFlags["device"], ", ".join(device_info))
      device_info = [TheanoFlags["device"]]
  device_tags = {}
  ncpus, ngpus = get_num_devices()
  if "all" in device_info:
    device_tags = { tag: 1 for tag in [ "cpu" + str(i) for i in xrange(ncpus)] + [ "gpu" + str(i) for i in xrange(ngpus)] }
  else:
    for info in device_info:
      num_batches = 1
      if ':' in info:
        num_batches = int(info.split(':')[1])
        info = info.split(':')[0]
      if len(info) == 3: info += "X"
      assert len(info) > 3, "invalid device: " + str(info) #str(info[:-1])
      utype = info[0:3]
      uid = info[3:]
      if uid == '*': uid = "[0-9]*"
      if uid == 'X': device_tags[info] = num_batches
      else:
        if utype == 'cpu':
          np = ncpus
        elif utype == 'gpu':
          np = ngpus
        match = False
        for p in xrange(np):
          if re.match(uid, str(p)):
            device_tags[utype + str(p)] = num_batches
            match = True
        assert match, "invalid device specified: " + info
  assert config.has('train'), "no train files specified"
  tags = sorted(device_tags.keys())
  if multiproc:
    devices = [ Device(tag, config, num_batches = device_tags[tag]) for tag in tags ]
  else:
    devices = [ Device(tags[0], config, blocking = True) ]
  return devices

def getCacheSizes():
  """
  :rtype: (int,int,int)
  :returns cache size for (train,dev,eval)
  """
  cache_sizes_user = config.list('cache_size', ["0"])
  sets = 1 + config.has('dev') + config.has('eval')
  cache_factor = 1.0
  if len(cache_sizes_user) == 1:
    cache_sizes_user *= 3
    cache_factor /= float(sets)
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

def initData():
  """
  Inits the globals train,dev,eval of type Dataset.
  """
  cache_sizes = getCacheSizes()
  chunking = "0"
  if config.value("on_size_limit", "ignore") == "chunk":
    chunking = config.value("batch_size", "0")
  global train, dev, eval
  dev,extra_dev = Dataset.load_data(config, cache_sizes[1], 'dev', chunking = chunking, batching = "sorted")
  eval,extra_eval = Dataset.load_data(config, cache_sizes[2], 'eval', chunking = chunking, batching = "sorted")
  extra_cache = cache_sizes[0] + (extra_dev + extra_eval - 0) * (cache_sizes[0] > 0)
  train,extra_train = Dataset.load_data(config, cache_sizes[0] + extra_cache, 'train')

def initNeuralNetwork(init_from_last_epoch=None):
  """
  Load neural network.
  :type init_from_last_epoch: int | None
  :param init_from_last_epoch: if set, will load the specific epoch model
  :rtype: Network.LayerNetwork
  """
  global last_epoch
  from Network import LayerNetwork

  modelFileName = None
  if init_from_last_epoch is not None:
    modelFileName = config.value('model', '')
    assert modelFileName
    modelFileName += ".%03d" % init_from_last_epoch
    print >> log.v1, "loading weights from prev epoch model %s" % modelFileName
  elif config.has('load'):
    modelFileName = config.value('load', '')
    print >> log.v1, "loading weights from", modelFileName

  if modelFileName is not None:
    model = h5py.File(modelFileName, "r")
    if config.bool('initialize_from_model', False):
      print >> log.v5, "initializing network topology from model"
      network = LayerNetwork.from_model(model)
      subtract_priors(network, train, config)
    else:
      network = LayerNetwork.from_config(config)
    last_epoch = network.load(model)
    model.close()
  else:
    network = LayerNetwork.from_config(config)
    last_epoch = 0

  if init_from_last_epoch is not None:
    assert init_from_last_epoch == last_epoch

  if config.has('dump_json'):
    fout = open(config.value('dump_json', ''), 'w')
    try:
      json_content = network.to_json()
      print json_content
      print "---------------"
      json_data = json.loads(json_content)
      print json_data
      print "---------------"
      print json.dumps(json_data, indent = 2)
      print "---------------"
      print >> fout, json.dumps(json_data, indent = 2)
    except ValueError:
      print >> log.v5, network.to_json()
      assert False, "JSON parsing failed"
    fout.close()
  return network

def printTaskProperties(devices, network):
  """
  :type devices: list[Device]
  :type network: Network.LayerNetwork
  """
  print >> log.v2, "Network layer topology:"
  print >> log.v2, "  input #:", network.n_in
  for i in xrange(len(network.hidden_info)):
    print >> log.v2, "  " + network.hidden_info[i][0] + " #:", network.hidden_info[i][1]
  print >> log.v2, "  output #:", network.n_out
  print >> log.v2, "net weights #:", network.num_params()
  print >> log.v2, "net params:", network.gparams
  print >> log.v5, "net output:", network.output

  if train:
    print >> log.v2, "Train data:"
    print >> log.v2, "input:", train.num_inputs, "x", train.window
    print >> log.v2, "output:", train.num_outputs
    print >> log.v2, "sequences:", train.num_seqs
    print >> log.v2, "frames:", train.num_timesteps
  if dev:
    print >> log.v2, "Dev data:"
    print >> log.v2, "sequences:", dev.num_seqs
    print >> log.v2, "frames:", dev.num_timesteps
  if eval:
    print >> log.v2, "Eval data:"
    print >> log.v2, "sequences:", eval.num_seqs
    print >> log.v2, "frames:", eval.num_timesteps

  print >> log.v3, "Devices:"
  for device in devices:
    print >> log.v3, device.name + ":", device.device_name,
    print >> log.v3, "(units:", device.get_device_shaders(), "clock: %.02f" % (device.get_device_clock() / 1024.0) + "Ghz memory: %.01f" % (device.get_device_memory() / float(1024 * 1024 * 1024)) + "GB)",
    print >> log.v3, "working on", device.num_batches, "batches" if device.num_batches > 1 else "batch"

def initEngine(devices, network):
  """
  :type devices: list[Device]
  :type network: Network.LayerNetwork
  Inits global engine.
  """
  global engine
  engine = Engine(devices, network)

def init(configFilename, commandLineOptions):
  initConfig(configFilename, commandLineOptions)
  initLog()
  print >> log.v3, "crnn starting up"
  initConfigJson()
  maybeInitSprintCommunicator()
  devices = initDevices()
  initData()
  network = initNeuralNetwork()
  printTaskProperties(devices, network)
  initEngine(devices, network)

def finalize():
  for device in engine.devices:
    device.terminate()
  if SprintCommunicator.instance is not None:
    SprintCommunicator.instance.finalize()

def executeMainTask():
  st = time.time()
  task = config.value('task', 'train')
  if task == 'train':
    engine.train_config(config, train, dev, eval, start_epoch=last_epoch+1)
  elif task == 'forward':
    assert eval is not None, 'no eval data provided'
    assert config.has('output_file'), 'no output file provided'
    combine_labels = config.value('combine_labels', '')
    output_file = config.value('output_file', '')
    engine.forward(engine.devices[0], eval, output_file, combine_labels)
  elif task == 'theano_graph':
    import theano
    for task in config.list('theano_graph.task', ['train']):
      theano.printing.pydotprint(engine.devices[-1].compute(task), format = 'png', var_with_name_simple = True,
                                 outfile = config.value("theano_graph.prefix", "current") + "." + task + ".png")
  elif task == 'analyze':
    statistics = config.list('statistics', ['confusion_matrix'])
    engine.analyze(engine.devices[0], eval, statistics)
  elif task == "classify":
    assert eval is not None, 'no eval data provided'
    assert config.has('label_file'), 'no output file provided'
    label_file = config.value('label_file', '')
    engine.classify(engine.devices[0], eval, label_file)
  else:
    assert False, "unknown task: %s" % task

  print >> log.v3, ("elapsed: %f" % (time.time() - st))

def main(argv):
  assert len(argv) >= 2, "usage: %s <config>" % argv[0]
  init(configFilename=argv[1], commandLineOptions=argv[2:])
  executeMainTask()
  finalize()

if __name__ == '__main__':
  main(sys.argv)
