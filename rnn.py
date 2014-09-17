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

from Log import log
from Device import Device, get_num_devices
from Config import Config
from optparse import OptionParser

def load_data(config, cache_size, key, chunking = "chunking", batching = "batching"):
  window = config.int('window', 1)
  if chunking == "chunking": chunking = config.value("chunking", "0")
  if batching == "batching": batching = config.value("batching", 'default')
  if config.has(key):
    data = Dataset(window, cache_size, chunking, batching)
    for f in config.list(key):
      data.add_file(f)
    return data, data.initialize()
  return None, 0
  
if __name__ == '__main__':
  # initialize config file
  assert os.path.isfile(sys.argv[1]), "config file not found"  
  config = Config()
  config.load_file(sys.argv[1])
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
  parser.add_option("-k", "--multiprocessing", dest = "multiprocessing", help = "[BOOLEAN] Enable multi threaded processing (required when using multiple devices).")
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
  parser.add_option("-v", "--verbose", dest = "verbose", help = "[INTEGER] Verbosity level from 0 - 5.")
  parser.add_option("-w", "--window", dest = "window", help = "[INTEGER] Width of sliding window over sequence.")
  parser.add_option("-x", "--task", dest = "task", help = "[train/forward/analyze] Task of the current program call.")
  parser.add_option("-y", "--hidden_type", dest = "hidden_type", help = "[VALUE/LIST] Hidden layer types: forward, recurrent, lstm.")
  parser.add_option("-z", "--max_sequences", dest = "max_seqs", help = "[INTEGER] Maximal number of sequences per batch.")
  (options, args) = parser.parse_args()
  options = vars(options)
  for opt in options.keys():
    if options[opt] != None:
      config.set(opt, options[opt])

  # initialize log file
  logs = config.list('log', [])
  log_verbosity = config.int_list('log_verbosity', [])
  log_format = config.list('log_format', [])
  log.initialize(logs = logs, verbosity = log_verbosity, formatter = log_format)
  # initialize devices
  device_info = config.list('device', ['cpu0'])
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
      assert len(info) > 3, "invalid device: " + str(info[:-1])
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
  if config.bool('multiprocessing', True):
    devices = [Device(tag, config, num_batches = device_tags[tag]) for tag in tags]
  else:
    import theano.tensor as T
    devices = [ Device(tags[0], config, blocking = True) ]
  # load data
  import theano.tensor as T
  from Dataset import Dataset
  from Network import LayerNetwork
  from Engine import Engine
  cache_sizes_user = config.list('cache_size', ["0"])
  sets = 1 + config.has('dev') + config.has('eval')
  cache_factor = 1.0
  if len(cache_sizes_user) == 1: 
    cache_sizes_user = cache_sizes_user * 3
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
  dev,extra_dev = load_data(config, cache_sizes[1], 'dev', chunking = "0", batching = "sorted")
  eval,extra_eval = load_data(config, cache_sizes[2], 'eval', chunking = "0", batching = "sorted")
  extra_cache = cache_sizes[0] + (extra_dev + extra_eval - 0) * (cache_sizes[0] > 0)
  train,extra_train = load_data(config, cache_sizes[0] + extra_cache, 'train')
  # initialize network
  network = LayerNetwork.from_config(config)
  task = config.value('task', 'train')
  start_batch = config.int('start_batch', 0)
  if config.has('load'):
    weights = config.value('load', '')
    print >> log.v1, "loading weights from", weights
    start_epoch = network.load(weights)
    if start_batch > 0:
      print >> log.v3, "starting at batch", start_batch
  else: start_epoch = 0
  # print task properties
  print >> log.v2, "Network:"
  print >> log.v2, "input:", train.num_inputs, "x", train.window
  for i in xrange(len(network.hidden_info)):
    print >> log.v2, network.hidden_info[i][0] + ":", network.hidden_info[i][1]
  print >> log.v2, "output:", train.num_outputs
  print >> log.v2, "weights:", network.num_params()
  print >> log.v2, "train:"
  print >> log.v2, "sequences:", train.num_seqs
  print >> log.v2, "frames:", train.num_timesteps
  if dev:
    print >> log.v2, "dev:"
    print >> log.v2, "sequences:", dev.num_seqs
    print >> log.v2, "frames:", dev.num_timesteps
  if eval:
    print >> log.v2, "eval:"
    print >> log.v2, "sequences:", eval.num_seqs
    print >> log.v2, "frames:", eval.num_timesteps
  print >> log.v3, "Devices:"
  for device in devices:
    print >> log.v3, device.name + ":", device.device_name,
    print >> log.v3, "(units:", device.get_device_shaders(), "clock: %.02f" % (device.get_device_clock() / 1024.0) + "Ghz memory: %.01f" % (device.get_device_memory() / float(1024 * 1024 * 1024)) + "GB)",
    print >> log.v3, "working on", device.num_batches, "batches" if device.num_batches > 1 else "batch"
  engine = Engine(devices, network)
  st = time.time()
  if task == 'train':    
    engine.train_config(config, train, dev, eval, start_epoch)
  elif task == 'forward':
    assert eval != None, 'no eval data provided'
    assert config.has('cache_file'), 'no output file provided'
    combine_labels = config.value('combine_labels', '')
    cache_file = config.value('cache_file', '')
    engine.forward(devices[0], eval, cache_file, combine_labels)
  elif task == 'analyze':
    statistics = config.list('statistics', ['confusion_matrix'])
    engine.analyze(devices[0], eval, statistics)
  elif task == "classify":
    assert eval != None, 'no eval data provided'
    assert config.has('label_file'), 'no output file provided'
    label_file = config.value('label_file', '')
    engine.classify(devices[0], eval, label_file)
  elif task == "daemon":                                            
    engine.run_daemon(train, dev, eval)
  for device in devices:
    device.terminate()
  print >> log.v3, ("elapsed: %f" % (time.time() - st))
