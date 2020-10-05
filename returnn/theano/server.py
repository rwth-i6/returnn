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
from tornado import locks
from tornado.concurrent import Future
from tornado.ioloop import IOLoop
from tornado.queues import Queue, QueueEmpty
import tornado.web

from returnn.log import log
from returnn.datasets.generating import StaticDataset
from returnn.theano.device import Device
from returnn.util.basic import TheanoFlags
from returnn.config import get_devices_init_args
from returnn.theano.engine_task import ForwardTaskThread
import returnn.theano.engine
import returnn.config

# TODO: Look into making these non-global.
_max_amount_engines = 4


class ClassificationRequest:
  def __init__(self, data):
    self.data = data
    self.future = Future()


class Model:
  def __init__(self, config_file):
    self.lock = locks.Lock()
    self.classification_queue = Queue()

    print('loading config %s' % config_file, file=log.v5)
    # Load and setup config
    try:
      self.config = returnn.config.Config()
      self.config.load_file(config_file)
      self.pause_after_first_seq = self.config.float('pause_after_first_seq', 0.2)
      self.batch_size = self.config.int('batch_size', 5000)
      self.max_seqs = self.config.int('max_seqs', -1)
    except Exception:
      print('Error: loading config %s failed' % config_file, file=log.v1)
      raise

    try:
      self.devices = self._init_devices()
    except Exception:
      print('Error: Loading devices for config %s failed' % config_file, file=log.v1)
      raise

    print('Starting engine for config %s' % config_file, file=log.v5)
    self.engine = returnn.theano.engine.Engine(self.devices)
    try:
      self.engine.init_network_from_config(config=self.config)
    except Exception:
      print('Error: Loading network for config %s failed' % config_file, file=log.v1)
      raise

    IOLoop.current().spawn_callback(self.classify_in_background)

    self.last_used = datetime.datetime.now()

  def _init_devices(self):
    """
    Initiates the required devices for a config. Same as the funtion initDevices in
    rnn.py.
    :param config:
    :return: A list with the devices used.
    """
    oldDeviceConfig = ",".join(self.config.list('device', ['default']))
    if "device" in TheanoFlags:
      # This is important because Theano likely already has initialized that device.
      config.set("device", TheanoFlags["device"])
      print("Devices: Use %s via THEANO_FLAGS instead of %s." % (TheanoFlags["device"], oldDeviceConfig), file=log.v4)
    devArgs = get_devices_init_args(self.config)
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

  @tornado.gen.coroutine
  def classify_in_background(self):
    while True:
      requests = []
      # fetch first request
      r = yield self.classification_queue.get()
      requests.append(r)
      # grab all other waiting requests
      try:
        while True:
          requests.append(self.classification_queue.get_nowait())
      except QueueEmpty:
        pass

      output_dim = {}
      # Do dataset creation and classification.
      dataset = StaticDataset(data=[r.data for r in requests], output_dim=output_dim)
      dataset.init_seq_order()
      batches = dataset.generate_batches(recurrent_net=self.engine.network.recurrent,
                                         batch_size=self.batch_size, max_seqs=self.max_seqs)

      with (yield self.lock.acquire()):
        ctt = ForwardTaskThread(self.engine.network, self.devices, dataset, batches)
        yield ctt.join()

      try:
        for i in range(dataset.num_seqs):
          requests[i].future.set_result(ctt.result[i])
          self.classification_queue.task_done()
      except Exception as e:
        print('exception', e)
        raise

  @tornado.gen.coroutine
  def classify(self, data):
    self.last_used = datetime.datetime.now()
    request = ClassificationRequest(data)

    yield self.classification_queue.put(request)
    yield request.future

    # return request.future.result()


class Server:
  def __init__(self, global_config):
    """
    Initializes the server with an empty config.
    :param global_config: Basic config of server. Requires a network paramater.
    """

    # Create temporary directories.
    self.base_dir   = os.path.abspath(returnn.config.get_global_config().value('__file__', ''))
    self.config_dir = os.path.join(self.base_dir, "configs")
    self.data_dir   = os.path.join(self.base_dir, "data")

    for d in [self.config_dir, self.data_dir]:
      try:
        os.makedirs(d)
      except FileExistsError as e:
        pass

    self.models = {}

    self.application = tornado.web.Application([
      (r"/classify", ClassifyHandler, {'models': self.models}),
      (r"/loadconfig", ConfigHandler, {'config_dir': self.config_dir, 'models': self.models})
    ], debug=True, autoreload=False)

    self.port = int(global_config.value('port', '3033'))
    global _max_amount_engines
    _max_amount_engines = int(global_config.value('max_engines', '5'))

  def run(self):
    print("Starting server on port: %d" % self.port, file=log.v3)
    self.application.listen(self.port)
    IOLoop.instance().start()


class ClassifyHandler(tornado.web.RequestHandler):
  def initialize(self, models, **kwargs):
    self.models = models

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
    try:
      model = self.models[engine_hash]
    except KeyError:
      print('Error: Received request with unknown engine_hash %s' % engine_hash, file=log.v1)
      self.set_status(499, 'Unknown engine_hash')
      return
    network = model.engine.network

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
            self.set_status(499, 'unknown target: %s' % k)
          else:
            self.set_status(499, 'unable to convert %s to an array from value %s' % (k, str(data[k])))
          return
    elif data_format == 'binary':
      float_array = array(self._get_type_code(data_type))
      try:
        float_array.frombytes(self.request.body)
      except Exception as e:
        print('Binary data error: %s' % str(e.message), file=log.v4)
        self.set_status(499, 'Error during binary data conversion: ' + e.message)
        return
      shape = tuple(map(int, data_shape.split(',')))
      data['data'] = np.asarray(float_array.tolist(), dtype=data_type).reshape(shape)

    result = yield model.classify(data)
    print('writing results for request with shape %s' % data_shape)
    if data_format == 'json':
      self.write({'result': result.tolist()})
    elif data_format == 'binary':
      size_info = struct.pack('<LL', *result.shape)
      self.write(size_info)
      self.write(result.tobytes())


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
  load_config_lock = locks.Lock()

  def initialize(self, config_dir, models, **kwargs):
    self.config_dir = config_dir
    self.models     = models

  @tornado.web.asynchronous
  @tornado.gen.coroutine
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
    data = json.loads(self.request.body)
    config_url = data["new_config_url"]
    print('Received request to load config %s' % config_url, file=log.v3)

    hash_engine = hashlib.new('ripemd160')
    hash_engine.update(config_url.encode('utf8'))
    hash_val = hash_engine.hexdigest()

    with (yield self.load_config_lock.acquire()):
      if hash_val in self.models:
        print('Using existing model with hash %s' % hash_val, file=log.v3)
        self.write(hash_val)
        return

      # Download new config file and save to temp folder.
      print('loading config %s' % config_url, file=log.v5)
      config_file = os.path.join(self.config_dir, hash_val + ".config")
      try:
        urllib.request.urlretrieve(config_url, config_file)
      except (urllib.request.URLError, urllib.request.HTTPError) as e:
        self.write('Error: Loading in config file from URL %s' % e)
        return

      try:
        self.models[hash_val] = Model(config_file)
      except Exception as e:
        print('Error: loading model failed %s' % str(e), file=log.v1)
        self.set_status(499, str(e))
      else:
        self.write(hash_val)


# There used to be a training handler, but it was not finished and the server needed refactoring, so it was deleted to save some time. If you wish to revive it look in the VCS history for its code
