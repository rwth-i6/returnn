

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
from EngineTask import ClassificationTaskThread
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor
from Dataset import Dataset, init_dataset, init_dataset_via_str
from HDFDataset import HDFDataset


_engines = {}
_engine_usage = {}
_devices = {}
_classify_cache = {}
_max_amount_engines = 1
_configs = {}
_data = {}


class Server:

    def __init__(self, global_config):
        """
            :type devices: list[Device.Device]
        """
    
        application = tornado.web.Application([
          (r"/classify", ClassifyHandler),
          (r"/loadconfig", ConfigHandler),
          (r"/train", TrainingHandler)
        ], debug=True)
        
        application.listen(int(global_config.value('port', '3033')))

        #self._max_amount_engines = global_config.value('max_engines', '5')
        
        print("Starting server on port: " + global_config.value('port', '3033'), file=log.v3)
        
        IOLoop.instance().start()


class ClassifyHandler(tornado.web.RequestHandler):
    
    MAX_WORKERS = 4
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    
    @run_on_executor
    def classification_task(self, network, devices, data, batches):
        #This will be executed in `executor` pool
        td = ClassificationTaskThread(network, devices, data, batches)
        td.join()
        return td

    @tornado.web.asynchronous
    @tornado.gen.coroutine
    def post(self, *args, **kwargs):
        #TODO: Make this batch over a specific time period
    
        params = json.loads(self.request.body)
        output_dim = {}
        ret = {}
        
        #first get meta data
        engine_hash = params['engine_hash']
        
        print('Received engine hash: ', engine_hash, file=log.v4)
        
        #delete unneccessary stuff so that the rest works
        del params['engine_hash']
        
        #load in engine and hash
        engine = _engines[engine_hash]
        network = engine.network
        devices = _devices[engine_hash]
        
        hash_engine = hashlib.new('ripemd160')
        hash_engine.update(json.dumps(params) + engine_hash)
        hash_temp = hash_engine.hexdigest()
        
        #process the data
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
            try:
                data = StaticDataset(data=[params], output_dim=output_dim)
                data.init_seq_order()
            except Exception:
                ret['error'] = 'Dataset server error'
                self.write(ret)
                pass
            else:
                batches = data.generate_batches(recurrent_net=network.recurrent,
                                                batch_size=sys.maxsize, max_seqs=1)
                if not hash_temp in _classify_cache:
                    print('Starting classification', file=log.v3)
                    #if we haven't yet processed this exact request, and saved it in the cache
                    _classify_cache[hash_temp] = yield self.classification_task(network=network,
                                                                                devices=devices,
                                                                                data=data, batches=batches)

                ret = {'result':
                     {k: _classify_cache[hash_temp].result[k].tolist() for k in _classify_cache[hash_temp].result}}
        
        #update engine usage for performance optimization
        _engine_usage[engine_hash] = datetime.datetime.now()
        
        print("Finished processing classification with ID: ", hash_temp, file=log.v4)
        
        self.write(ret)
        

    
    

#EXAMPLE: curl -H "Content-Type: application/json" -X POST -d '{"new_config_url":"file:///home/nikita/Desktop/returnn-experiments-master/mnist-beginners/config/ff_3l_sgd.config"}' http://localhost:3033/loadconfig
class ConfigHandler(tornado.web.RequestHandler):

    def get(self):
        pass
    
    #not async and blocking, as it is a critical operation
    def post(self, *args, **kwargs):
      
        #cleanup of old unused engines
        if len(_engines)+1 > _max_amount_engines:
            self.remove_oldest_engine()
        
        data = json.loads(self.request.body)
    
        print('Received new config for new engine', file=log.v3)
    
        #Overview: create engine, download full config, create hash, and add it to the main list
        
        # generate ID
        hash_engine = hashlib.new('ripemd160')
        hash_engine.update(json.dumps(args) + str(datetime.datetime.now()))
        hash_temp = hash_engine.hexdigest()

        # TODO: better fallback
        # download new config file
        urlmanager = urllib.URLopener()
        config_file = str(datetime.datetime.now()) + ".config"
        urlmanager.retrieve(data["new_config_url"], config_file)
        
        # load and setup config
        config = Config.Config()
        config.load_file(config_file)
        if config.value('task', 'daemon') != 'training':
          config.set(key='task', value='daemon') #assume we're only using for classification or training
   
        #load devices
        _devices[hash_temp] = self.init_devices(config=config)
        
        #load engine
        new_engine = Engine.Engine(_devices[hash_temp])
        new_engine.init_network_from_config(config=config)
        
        _engines[hash_temp] = new_engine
        _engine_usage[hash_temp] = datetime.datetime.now()
        _configs[hash_temp] = config
        
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
    
    
    def remove_oldest_engine(self):
        
        oldest_time = datetime.datetime.max
        oldest_engine = None
        
        for engine in _engine_usage:
          if _engine_usage[engine] < oldest_time:
            oldest_engine = engine
            oldest_time = _engine_usage[engine]
            
        print("Cleaning up last used engine ", oldest_engine, file=log.v5)
        for device in _devices[oldest_engine]:
            device.terminate()
        del _engines[oldest_engine]
        del _devices[oldest_engine]
        del _engine_usage[oldest_engine]
        


#requires that the congig is loaded in
#requires training and dev data URL to h5 files

class TrainingHandler(tornado.web.RequestHandler):
  
      def get(self):
        pass

      
      def post(self, *args, **kwargs):
        
        data = json.loads(self.request.body)

        hash_engine = hashlib.new('ripemd160')

        print("Training engine on new data: ", data["engine_hash"], file=log.v4)
        
        #download training and dev data
        urlmanager = urllib.URLopener()
        train_file = "train_" + str(datetime.datetime.now()) + ".h5" #currently assume h5 data
        urlmanager.retrieve(data["training_url"], train_file)
        dev_file = "dev_" + str(datetime.datetime.now()) + ".h5"  # currently assume h5 data
        urlmanager.retrieve(data["dev_url"], dev_file)
        
        #configure engine
        engine_hash = data["engine_hash"]
        
        # update config to use this and save the future model
        _configs[engine_hash].set(key='train', value="/" + train_file)
        _configs[engine_hash].set(key='dev', value="/" + dev_file)
        _configs[engine_hash].set(key='model', value="/" + engine_hash)
        
        print("downloaded data")
        
        print("here")
        
        self.initData(config=_configs[engine_hash], engine_id=engine_hash)

        print("initialised data")
        
        _engines[engine_hash].init_train_from_config(_configs[engine_hash], _data[engine_hash][0],
                                                     _data[engine_hash][1], _data[engine_hash][2])
        print("initialised engine")
        
        _engines[engine_hash].train()
        
        self.write({'success':'true'})
      
      
      def getCacheByteSizes(self, config):
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

      def load_data(self, configT, cache_byte_size, files_config_key, **kwargs):
        """
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
        #error somewhere here
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

      def initData(self, config, engine_id):
        """
        Initializes the globals train,dev,eval of type Dataset.
        """
        cache_byte_sizes = self.getCacheByteSizes(config=config)
        chunking = "0"
        if config.value("on_size_limit", "ignore") == "chunk":
          chunking = config.value("batch_size", "0")
        elif config.value('chunking', "0") == "1":  # MLP mode
          chunking = "1"
        #global train_data, dev_data, eval_data
        dev_data, extra_cache_bytes_dev = self.load_data(
          config, cache_byte_sizes[1], 'dev', chunking=chunking, seq_ordering="sorted", shuffle_frames_of_nseqs=0)
        eval_data, extra_cache_bytes_eval = self.load_data(
          config, cache_byte_sizes[2], 'eval', chunking=chunking, seq_ordering="sorted", shuffle_frames_of_nseqs=0)
        train_cache_bytes = cache_byte_sizes[0]
        if train_cache_bytes >= 0:
          # Maybe we have left over cache from dev/eval if dev/eval have cached everything.
          train_cache_bytes += extra_cache_bytes_dev + extra_cache_bytes_eval
        train_data, extra_train = self.load_data(config, train_cache_bytes, 'train')
        _data[engine_id] = (train_data, extra_train, eval_data)

#TODO: implement training handler








