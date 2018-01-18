"""
The Server module manages connections via HTTP. It is especially
tuned to work well with multiple engine instances at once. Initiated
via rnn.py with a Server config. Here's a basic config:

#!returnn/rnn.py
task = "server"
port = 10687
max_engines = 2
network = {"out" : { "class" : "softmax", "loss" : "ce", "target":"classes" }}
"""

from __future__ import print_function

from array import array
import concurrent.futures
import datetime
import hashlib
import json
import os
import re
import struct
import sys
import time
import urllib
import urllib.request

import numpy as np
import tornado.web
from tornado.ioloop import IOLoop
from tornado.concurrent import run_on_executor

from Log import log
from GeneratingDataset import StaticDataset
from Device import Device, get_num_devices, TheanoFlags, getDevicesInitArgs
from EngineTask import ClassificationTaskThread
from Dataset import Dataset, init_dataset, init_dataset_via_str
from HDFDataset import HDFDataset
import Engine
import Config

# TODO: Look into making these non-global.
_max_amount_engines = 4
_engines = {}
_engine_usage = {}
_devices = {}
_classify_cache = {}
_configs = {}


class Server:
  def __init__(self, global_config):
    """
    Initializes the server with an empty config.
    :param global_config: Basic config of server. Requires a network paramater.
    """

    # Create temporary directories.
    self.base_dir   = os.path.abspath(Config.get_global_config().value('__file__', ''))
    self.config_dir = os.path.join(self.base_dir, "configs")
    self.data_dir   = os.path.join(self.base_dir, "data")

    for d in [self.config_dir, self.data_dir]:
      try:
        os.makedirs(d)
      except FileExistsError as e:
        pass

    self.application = tornado.web.Application([
      (r"/classify", ClassifyHandler),
      (r"/loadconfig", ConfigHandler, {'config_dir': self.config_dir}),
      (r"/train", TrainingHandler)
    ], debug=True)
    
    self.port = int(global_config.value('port', '3033'))
    global _max_amount_engines
    _max_amount_engines = int(global_config.value('max_engines', '5'))

  def run(self):
    print("Starting server on port: %d" % self.port, file=log.v3)
    self.application.listen(self.port)
    IOLoop.instance().start()
  

class ClassifyHandler(tornado.web.RequestHandler):
  max_workers = 4
  executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

  @run_on_executor
  def _classification_task(self, network, devices, data, batches):
    """
    Runs a classification task in an executor (async).
    :param network: Engine network.
    :param devices: Devices assigned to this engine.
    :param data: The data on which to classify.
    :param batches: Generated batches.
    :return:
    """
    # This will be executed in `executor` pool
    td = ClassificationTaskThread(network, devices, data, batches)
    td.join()
    return td
  
  @tornado.web.asynchronous
  @tornado.gen.coroutine
  def post(self, *args, **kwargs):
    # TODO: Write formal documentation
    """
    Method for handling classification via HTTP Post request. The following must
    be defined in the URL paramaters: engine_hash (engine hash which points to which
    engine to use), and the data itself in the body. If using binary data, the following
    URL paramaters must also be supplied: data_format='binary', data_shape=(<dim1,dim2>).
    If using a specific data type, you can supply it as the url parameter data_type.
    :param args:
    :param kwargs:
    :return: Either JSON with error or JSON list of generated outputs.
    """
    url_params = self.request.arguments
    output_dim = {}
    ret = {}
    data = {}
    data_format = 'json'
    data_type = 'float32'
    engine_hash = ''
    data_shape = ''
    # First get meta data from URL parameters
    if 'engine_hash' in url_params:
      engine_hash = url_params['engine_hash'][0].decode('utf8')
    if 'data_format' in url_params:
      data_format = url_params['data_format'][0].decode('utf8')
    if 'data_type' in url_params:
      # Possible options: https://docs.scipy.org/doc/numpy-1.10.1/user/basics.types.html
      data_type = url_params['data_type'][0].decode('utf8')
    if 'data_shape' in url_params:
      data_shape = url_params['data_shape'][0].decode('utf8')  # either '' or 'dim1,dim2'
    
    print('Received request, engine hash: %s, data format: %s, data type %s data shape: %s' %
          (engine_hash, data_format, data_type, data_shape), file=log.v5)
    # Load in engine and hash
    engine = _engines[engine_hash]
    network = engine.network
    devices = _devices[engine_hash]
    
    # Pre-process the data
    if data_format == 'json':
      data = json.loads(self.request.body)
      for k in data:
        try:
          data[k] = np.asarray(data[k], dtype=data_type)
          if k != 'data':
            output_dim[k] = network.n_out[k]  # = [network.n_in,2] if k == 'data' else network.n_out[k]
        except Exception:
          if k != 'data' and k not in network.n_out:
            ret['error'] = 'unknown target: %s' % k
          else:
            ret['error'] = 'unable to convert %s to an array from value %s' % (k, str(data[k]))
          break
    elif data_format == 'binary':
      float_array = array(self._get_type_code(data_type))
      try:
        float_array.frombytes(self.request.body)
      except Exception as e:
        print('Binary data error: %s' % str(e.message), file=log.v4)
        ret['error'] = 'Error during binary data conversion: ' + e.message
      shape = tuple(map(int, data_shape.split(',')))
      data['data'] = np.asarray(float_array.tolist(), dtype=data_type).reshape(shape)
    
    # Do dataset creation and classification.
    if 'error' not in ret:
      data = StaticDataset(data=[data], output_dim=output_dim)
      data.init_seq_order()
      batches = data.generate_batches(recurrent_net=network.recurrent,
                                      batch_size=sys.maxsize, max_seqs=1)
      ct = yield self._classification_task(network=network,
                                           devices=devices,
                                           data=data, batches=batches)
      assert len(ct.result.keys()) == 1  # we only added one sequence
      result_array = ct.result[next(iter(ct.result.keys()))]
      assert result_array.shape[0] == 1  # there is only one DeviceRun
      result_array = np.squeeze(np.reshape(result_array, result_array.shape[1:]))
      if data_format == 'json':
        ret = {'result': result_array.tolist()}
      elif data_format == 'binary':
        size_info = struct.pack('<LL', *result_array.shape)
        self.write(size_info)
        ret = result_array.tobytes()
    
    # Update engine usage for performance optimization
    _engine_usage[engine_hash] = datetime.datetime.now()
    self.write(ret)
    self.finish()
  
  def _get_type_code(self, format_to_convert):
    """
    Converts from Numpy format to python internal array format.
    :param format_to_convert: String of format used, e.g float32.
    :return: Returns a string of the type used for python arrays.
    """
    return {
      'float32': 'f',
      'float64': 'd',
      'int8': 'h',
      'int16': 'i',
      'int32': 'l',
    }[format_to_convert]


class ConfigHandler(tornado.web.RequestHandler):
  def initialize(self, config_dir, **kwargs):
    self.config_dir = config_dir

  def post(self, *args, **kwargs):
    """
    Handles the creation of a new engine based on a slightly modified config, supplied
    via HTTP Post. The request requires 1 URL parameter: new_config_url, which points
    to a URL (can be local) from where to download the config. This call is blocking.
    
    WARNING All configs must have the following added:
    extract_output_layer_name = "<OUTPUT LAYER ID>"
    :param args:
    :param kwargs:
    :return: ASCII encoded hash of the new engine based on the provided config.
    """
    # Overview: Clean up, create engine, download full config, create hash, and add it to the main list
    # TODO: Recheck clean up function
    global _max_amount_engines
    if (len(_engines) + 1) > _max_amount_engines:
        self._remove_oldest_engine()
    data = json.loads(self.request.body)
    config_url = data["new_config_url"]
    print('Received new config for new engine', file=log.v3)

    hash_engine = hashlib.new('ripemd160')
    hash_engine.update(config_url.encode('utf8'))
    hash_temp = hash_engine.hexdigest()

    if hash_temp in _configs:
      self.write(hash_temp)
      return

    # Download new config file and save to temp folder.
    print('loading config %s' % config_url, file=log.v5)
    config_file = os.path.join(self.config_dir, hash_temp + ".config")
    try:
      urllib.request.urlretrieve(config_url, config_file)
    except (urllib.request.URLError, urllib.request.HTTPError) as e:
      self.write('Error: Loading in config file from URL %s' % e)
      return

    print('reading config %s' % config_file, file=log.v5)
    # Load and setup config
    try:
      config = Config.Config()
      config.load_file(config_file)
    except Exception:
      self.write('Error: Processing config file')
      return
    #if config.value('task', 'daemon') != 'train':
    #  config.set(key='task', value='daemon')  # Assume we're only using for classification or training
    try:
      _devices[hash_temp] = self._init_devices(config=config)
    except Exception:
      self.write('Error: Loading devices failed')
      return
    
    print('Starting engine %s' % hash_temp, file=log.v5)
    new_engine = Engine.Engine(_devices[hash_temp])
    try:
      new_engine.init_network_from_config(config=config)
    except Exception:
      self.write('Error: Loading engine failed')
      return
    _engines[hash_temp] = new_engine
    _engine_usage[hash_temp] = datetime.datetime.now()
    _configs[hash_temp] = config
    print('Loaded new config from %s with hash %s, %d engines are running' % (config_url, hash_temp, len(_engines)), file=log.v3)
    self.write(hash_temp)
  
  def _init_devices(self, config):
    """
    Initiates the required devices for a config. Same as the funtion initDevices in
    rnn.py.
    :param config:
    :return: A list with the devices used.
    """
    oldDeviceConfig = ",".join(config.list('device', ['default']))
    if "device" in TheanoFlags:
      # This is important because Theano likely already has initialized that device.
      config.set("device", TheanoFlags["device"])
      print("Devices: Use %s via THEANO_FLAGS instead of %s." % (TheanoFlags["device"], oldDeviceConfig), file=log.v4)
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
  
  def _remove_oldest_engine(self):
    """ Removes the oldest engine."""
    oldest_time = datetime.datetime.max
    oldest_engine = None
    for engine in _engine_usage:
      if _engine_usage[engine] < oldest_time:
        oldest_engine = engine
        oldest_time = _engine_usage[engine]
    
    print('Cleaning up last used engine %s' % oldest_engine, file=log.v5)
    for device in _devices[oldest_engine]:
      device.terminate()
    del _engines[oldest_engine]
    del _devices[oldest_engine]
    del _engine_usage[oldest_engine]


# TODO: finish implementation
class TrainingHandler(tornado.web.RequestHandler):
  _data = {}
  
  def get(self):
    pass
  
  def post(self, *args, **kwargs):
    """
    Handles training of an engine based on new supplied data via HTTP Post. WIP, lots of
    internal bugs right now. Requires that the config is already loaded in, and that the training and
    dev data are supplied in the h5 format.
    :param args:
    :param kwargs:
    :return: JSON of training results.
    """
    data = json.loads(self.request.body)
    hash_engine = hashlib.new('ripemd160')
    print("Training engine on new data: ", data["engine_hash"], file=log.v4)
    # download training and dev data
    if data['download_data'] != 'False':
      urlmanager = urllib.request.URLopener()
      datapath = "data/"
      train_file = datapath + re.sub(r'\W+', '', "train" + str(datetime.datetime.now()))  # currently assume h5 data
      train_file_abs, train_file_headers = urlmanager.retrieve(data["training_url"], train_file)
      dev_file = datapath + re.sub(r'\W+', '', "dev" + str(datetime.datetime.now()))  # currently assume h5 data
      dev_file_abs, dev_file_headers = urlmanager.retrieve(data["dev_url"], dev_file)
    else:
      dev_file_abs = data["dev_url"]
      train_file_abs = data["training_url"]
    
    # configure engine
    engine_hash = data["engine_hash"]
    # TODO: set new training epochs, so that it actually trains
    # Current issue: we're changing the config and the corresponding epoch model, but the internal cache
    # of the engine doesn't let us update the internal epoch model Engine:86 -> get_epoch_model.
    # if overriden, check line Engine:345
    # why is Engine._epoch_model a class variable?
    
    # remove weight loading, as we're expecting a new topology
    _configs[engine_hash].set(key='load', value='')
    # update config to use this new data and save the future model
    _configs[engine_hash].set(key='train', value=train_file_abs)
    _configs[engine_hash].set(key='dev', value=dev_file_abs)
    _configs[engine_hash].set(key='model', value="/" + engine_hash)
    
    print("downloaded data")
    self._init_data(config=_configs[engine_hash], engine_id=engine_hash)
    print("initialised data")
    _engines[engine_hash].init_train_from_config(_configs[engine_hash], self._data[engine_hash][0],
                                                 self._data[engine_hash][1], self._data[engine_hash][2])
    print("initialised engine")
    # error currently at Engine:362
    _engines[engine_hash].train()
    self.write({'success': 'true'})
  
  def _get_cache_byte_sizes(self, config):
    """
    Same as getCacheByteSize in rnn.py.
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
  
  def _load_data(self, configT, cache_byte_size, files_config_key, **kwargs):
    """
    Loads the data into Returnn. Same as load_data in rnn.py.
    :param Config config:
    :param int cache_byte_size:
    :param str files_config_key: such as "train" or "dev"
    :param kwargs: passed on to init_dataset() or init_dataset_via_str()
    :rtype: (Dataset,int)
    :returns the dataset, and the cache byte size left over if we cache the whole dataset.
    """
    if not configT.has(files_config_key):
      return None, 0
    kwargs = kwargs.copy()
    kwargs.setdefault("name", files_config_key)
    # error somewhere here
    if configT.is_typed(files_config_key) and isinstance(configT.typed_value(files_config_key), dict):
      config_opts = configT.typed_value(files_config_key)
      assert isinstance(config_opts, dict)
      kwargs.update(config_opts)
      if 'cache_byte_size' not in config_opts:
        if kwargs.get('class', None) == 'HDFDataset':
          kwargs["cache_byte_size"] = cache_byte_size
      Dataset.kwargs_update_from_config(configT, kwargs)
      data = init_dataset(kwargs)
    else:
      config_str = configT.value(files_config_key, "")
      data = init_dataset_via_str(config_str, config=configT, cache_byte_size=cache_byte_size, **kwargs)
    cache_leftover = 0
    if isinstance(data, HDFDataset):
      cache_leftover = data.definite_cache_leftover
    return data, cache_leftover
  
  def _init_data(self, config, engine_id):
    """
    Initializes the globals train,dev,eval of type Dataset. Same as
    initData in rnn.py.
    """
    cache_byte_sizes = self._get_cache_byte_sizes(config=config)
    chunking = "0"
    if config.value("on_size_limit", "ignore") == "chunk":
      chunking = config.value("batch_size", "0")
    elif config.value('chunking', "0") == "1":  # MLP mode
      chunking = "1"
    # global train_data, dev_data, eval_data
    dev_data, extra_cache_bytes_dev = self._load_data(
      config, cache_byte_sizes[1], 'dev', chunking=chunking, seq_ordering="sorted", shuffle_frames_of_nseqs=0)
    eval_data, extra_cache_bytes_eval = self._load_data(
      config, cache_byte_sizes[2], 'eval', chunking=chunking, seq_ordering="sorted", shuffle_frames_of_nseqs=0)
    train_cache_bytes = cache_byte_sizes[0]
    if train_cache_bytes >= 0:
      # Maybe we have left over cache from dev/eval if dev/eval have cached everything.
      train_cache_bytes += extra_cache_bytes_dev + extra_cache_bytes_eval
    train_data, extra_train = self._load_data(config, train_cache_bytes, 'train')
    self._data[engine_id] = (train_data, extra_train, eval_data)
