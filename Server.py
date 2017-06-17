

from __future__ import print_function

import numpy
import sys
import os
import tornado.web
from tornado.ioloop import IOLoop
from tornado import gen
import Device
import Engine
import hashlib
import Config
import json
from Log import log
import urllib
import datetime
from GeneratingDataset import StaticDataset



_engines = {}
_devices = []


class Server:

  def __init__(self, devices):
    """
        :type devices: list[Device.Device]
    """

    application = tornado.web.Application([
      (r"/classify", ClassifyHandler),
      (r"/loadconfig", ConfigHandler)
    ], debug=True)

    self._devices = devices

    print("Starting server", file=log.v3)
    application.listen(3033)
    IOLoop.instance().start()


#TODO: implement classification handler
class ClassifyHandler(tornado.web.RequestHandler):

  #@gen.coroutine
  def post(self, *args, **kwargs):
    #TODO: Make this async, and batch over a specific time period
    #TODO: finish this

    params = json.loads(self.request.body)
    output_dim = {}

    #first get meta data


    #process the real data
    """
    for k in params:
      try:
        params[k] = numpy.asarray(params[k], dtype='float32')
        if k != 'data':
          output_dim[k] = network.n_out[k]  # = [network.n_in,2] if k == 'data' else network.n_out[k]
      except Exception:
        if k != 'data' and not k in network.n_out:
          ret['error'] = 'unknown target: %s' % k
        else:
          ret['error'] = 'unable to convert %s to an array from value %s' % (k, str(params[k]))
        break
    if not 'error' in ret:
      data = StaticDataset(data=[params], output_dim=output_dim)
      data.init_seq_order()
      try:
        data = StaticDataset(data=[params], output_dim=output_dim)
        data.init_seq_order()
      except Exception:
        pass
      else:
        batches = data.generate_batches(recurrent_net=network.recurrent,
                                        batch_size=sys.maxsize, max_seqs=1)
    """

    pass


#EXAMPLE: curl -H "Content-Type: application/json" -X POST -d '{"new_config_url":"file:///home/nikita/Desktop/returnn-experiments-master/mnist-beginners/config/ff_3l_sgd.config"}' http://localhost:3033/loadconfig
class ConfigHandler(tornado.web.RequestHandler):


  #not async and blocking, as it is a critical operation
  def post(self, *args, **kwargs):

    #TODO: Add cleanup of old unsed engines

    data = json.loads(self.request.body)

    #for d in data:
    #  print('Data: ', d, file=log.v3)

    print('Received new config for new engine', file=log.v3)

    #Overview: create engine, download full config, create hash, and add it to the main list
    new_engine = Engine.Engine(_devices)

    #download new config file
    urlmanager = urllib.URLopener()
    config_file = str(datetime.datetime.now()) + ".config"
    urlmanager.retrieve(data["new_config_url"], config_file)

    #load and setup config
    config = Config.Config()
    config.load_file(config_file)
    new_engine.init_network_from_config(config=config)

    #generate ID
    hash_engine = hashlib.new('ripemd160')
    hash_engine.update(json.dumps(args) + str(datetime.datetime.now()))
    hash_temp = hash_engine.hexdigest()

    _engines[hash_temp] = new_engine

    self.write(hash_temp)

#TODO: implement training handler








