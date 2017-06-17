

from __future__ import print_function

import numpy
import sys
import os
import tornado.web
from tornado.ioloop import IOLoop
from tornado import gen
import Engine
import hashlib
import Config
import json
from Log import log
import urllib
import datetime
from GeneratingDataset import StaticDataset
from Device import Device, get_num_devices, TheanoFlags, getDevicesInitArgs
import time


_engines = {}
_devices = {}


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
        ret = {}
        
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
            if not hash in workers:
              workers[hash] = ClassificationTaskThread(network, devices, data, batches)
              workers[hash].json_params = params
              print("worker started:", hash, file=log.v3)
          ret['id_of_worker'] = { 'hash' : hash }
        """
    
        pass


#EXAMPLE: curl -H "Content-Type: application/json" -X POST -d '{"new_config_url":"file:///home/nikita/Desktop/returnn-experiments-master/mnist-beginners/config/ff_3l_sgd.config"}' http://localhost:3033/loadconfig
class ConfigHandler(tornado.web.RequestHandler):

    def get(self):
        pass
    
    #not async and blocking, as it is a critical operation
    def post(self, *args, **kwargs):

        #TODO: Add cleanup of old unused engines
    
        data = json.loads(self.request.body)
    
        #for d in data:
        #  print('Data: ', d, file=log.v3)
    
        print('Received new config for new engine', file=log.v3)
    
        #Overview: create engine, download full config, create hash, and add it to the main list
        
        # generate ID
        hash_engine = hashlib.new('ripemd160')
        hash_engine.update(json.dumps(args) + str(datetime.datetime.now()))
        hash_temp = hash_engine.hexdigest()

        # download new config file
        urlmanager = urllib.URLopener()
        config_file = str(datetime.datetime.now()) + ".config"
        urlmanager.retrieve(data["new_config_url"], config_file)
        
        # load and setup config
        config = Config.Config()
        config.load_file(config_file)
        config.set(key='task', value='daemon') #assume we're currently using only for online classification
        
        #load devices
        _devices[hash_temp] = self.init_devices(config=config)
        
        #load engine
        new_engine = Engine.Engine(_devices[hash_temp])
        new_engine.init_network_from_config(config=config)
        _engines[hash_temp] = new_engine
        
        print("Finished loading in, server running. Currently number of active engines: ", len(_engines), file=log.v3)
        
        self.write(hash_temp)
    
    
    def init_devices(self, config):
        
        #very basic and hacky
        #TODO: make this closer to rnn.py version, as in make it a function in rnn.py which isn't bound to the local vars of rnn.py (?)
        
        oldDeviceConfig = ",".join(config.list('device', ['default']))
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

#TODO: implement training handler








