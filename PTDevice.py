from __future__ import print_function

from TaskSystem import AsyncTask, ProcConnectionDied
from PTUpdater import Updater
from Util import cmd, progress_bar, dict_diff_str, hms, start_daemon_thread, interrupt_main, CalledProcessError, NumbersDict, custom_exec, dict_joined, attr_chain
from Log import log
from PTNetwork import LayerNetwork
from collections import OrderedDict
import numpy
import sys
import os
import signal
import time
import pickle
try:
  from thread import start_new_thread
except ImportError:
  # noinspection PyUnresolvedReferences
  from _thread import start_new_thread
import Debug
import re

import os
import sys
import torch
from torch.autograd import grad

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

floatX = 'float32'

class DummyModel(nn.Module):
  def __init__(self):
    super(DummyModel, self).__init__()
    self.lstm = nn.LSTM(50, 32, 1)
    self.output = nn.Linear(32, 4501)

  def forward(self, x, i):
    #x = x.permute(1,0,2)
    self.lstm.flatten_parameters()
    x, _ = self.lstm(x)
    x = x.permute(1,0,2)
    self.index = i
    self.pcx = F.softmax(self.output(x), dim=2)
    return self.pcx

  def cost(self, y):
    loss_function = nn.NLLLoss()
    i = self.index.view(-1).nonzero().long().view(-1)
    c = torch.log(torch.index_select(
          self.pcx.view(-1,self.pcx.shape[2]), 0, i))
    y = torch.index_select(y.view(-1), 0, i)
    loss = loss_function(c, y)
    return loss * self.index.float().sum()

  def errors(self, y):
    i = self.index.view(-1).nonzero().long().view(-1)
    y = torch.index_select(y.view(-1), 0, i)
    c = torch.index_select(torch.argmax(self.pcx,dim=2).view(-1), 0, i)
    return (y != c).float().sum() # * self.index.float().sum() / (self.logpcx.size(0) * self.logpcx.size(1))

def _get_num_devices():
  if os.name == 'nt':
    return 1, 1  # TODO
  elif sys.platform == 'darwin':
    return (
      int(cmd("sysctl -a | grep machdep.cpu.core_count | awk '{print $2}'")[0]),
      len(cmd("system_profiler SPDisplaysDataType | grep 'Chipset Model: NVIDIA' | cat")))
  else:
    num_cpus = len(cmd('cat /proc/cpuinfo | grep processor')) or 1
    num_gpus = 0
    try:
      num_gpus = len(cmd('nvidia-smi -L'))
    except CalledProcessError:
      pass
    return num_cpus, num_gpus

def get_device_attributes():
  # (shaders / CUDA cores, clock in MHz, memory in bytes)
  # https://en.wikipedia.org/wiki/GeForce_10_series
  # https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units
  attributes = {
                 "default" : (1000, 1020, 2 * 1024 * 1024 * 1024),
                 "GeForce GTX 580" : (512, 1714, 2 * 1024 * 1024 * 1024),
                 "GeForce GT 630M" : (96, 672, 2 * 1024 * 1024 * 1024),
                 "GeForce GT 650M" : (384, 900, 2 * 1024 * 1024 * 1024),
                 "GeForce GT 750M" : (384, 967, 2 * 1024 * 1024 * 1024),
                 "GeForce GTX 680" : (1536, 1020, 2 * 1024 * 1024 * 1024),
                 "GeForce GTX 750 Ti" : (640, 1110, 2 * 1024 * 1024 * 1024),
                 "GeForce GTX 760" : (2304, 980, 3 * 1024 * 1024 * 1024),
                 "GeForce GTX 770" : (1536, 1150, 2 * 1024 * 1024 * 1024),
                 "GeForce GTX 780" : (2304, 980, 3 * 1024 * 1024 * 1024),
                 "GeForce GTX 790" : (2304, 980, 3 * 1024 * 1024 * 1024),
                 "GeForce GTX 970" : (1664, 1178, 4 * 1024 * 1024 * 1024),
                 "GeForce GTX 980" : (2048, 1126, 4 * 1024 * 1024 * 1024),
                 "GeForce GTX 980 Ti" : (2048, 1126, 4 * 1024 * 1024 * 1024),
                 "GeForce GTX 1080" : (3584, 1480, 11 * 1024 * 1024 * 1024),
                 "GeForce GTX 1080 Ti" : (3584, 1480, 11 * 1024 * 1024 * 1024),
                 "GeForce GTX TITAN" : (2688, 837, 6 * 1024 * 1024 * 1024),
                 "Geforce GTX TITAN X" : (3584, 1417, 12 * 1024 * 1024 * 1024),
                 "GeForce GT 540M" : (2688, 837, 2 * 1024 * 1024 * 1024),
                 "Tesla K20c" : (2496, 706, 5 * 1024 * 1024 * 1024),
                 }
  cpu = 0
  #for clock in cmd('cat /proc/cpuinfo | grep "model name" | cut -d \'@\' -f 2 | tr -d \' \' | sed -e s/GHz//'):
  # Why is memory in bytes hard coded to 2GB for all cpus?
  if os.name != 'nt':
    if sys.platform == 'darwin':
      mhz = int(float(cmd("system_profiler  SPHardwareDataType | "
                          "grep 'Processor Speed' | awk '{print $3}'")[0].replace(',','.')) * 1024)
      for i in range(get_num_devices()[0]):
        attributes["cpu" + str(cpu)] = (1, mhz, 2 * 1024 * 1024 * 1024)
        cpu += 1
    else:
      for clock in cmd('cat /proc/cpuinfo | grep "cpu MHz" | cut -d \':\' -f 2 | sed \'s/^\\ //\''):
        attributes["cpu" + str(cpu)] = (1, int(float(clock)), 2 * 1024 * 1024 * 1024)
        cpu += 1
    attributes["cpu127"] = (1, 1, 32 * 1024 * 1024 * 1024) # what does this line do? Why add a cpu with 32GB?
  if not cpu:
    attributes["cpu0"] = (1, 1000, 2 * 1024 * 1024 * 1024)
  return attributes

_num_devices = None


def get_num_devices():
  """
  :return: (cpu count, gpu count)
  :rtype: (int, int)
  """
  global _num_devices
  if _num_devices is not None:
    return _num_devices
  _num_devices = _get_num_devices()
  return _num_devices

def getDevicesInitArgs(config):
  """
  :type config: Config.Config
  :rtype: list[dict[str]]
  """
  multiproc = config.bool('multiprocessing', True)
  device_info = config.list('device', ['cpu0'])
  if len(device_info) == 1 and device_info[0] == 'json':
    try:
      import json
      specs = json.loads(open(config.value('initialize_from_json', '')).read().replace('(','\"').replace(')','\"'))['worker']
    except Exception:
      raise Exception('Unable to parse worker information from json content')
    devices = [ { 'device' : specs[key]['device'], 'config' : config, 'blocking' : False, 'num_batches' : specs[key].pop('num_batches', 1), "update_specs" : specs[key].pop('update_specs', {}) } for key in specs ]
  else:
    device_tags = {}
    ngpux = 0
    ncpus, ngpus = get_num_devices()
    if "all" in device_info:
      device_tags = { tag: [1,True] for tag in [ "cpu" + str(i) for i in range(ncpus)] + [ "gpu" + str(i) for i in range(ngpus)] }
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
        if uid == 'X':
          ngpux += 1
          device_tags[info] = [num_batches, True]
        else:
          if utype == 'cpu':
            np = ncpus
          elif utype == 'gpu':
            np = ngpus
          else:
            np = 0
          match = False
          for p in range(np):
            if re.match(uid, str(p)):
              device_tags[utype + str(p)] = [num_batches, device_update]
              match = True
          assert match, "invalid device specified: " + info
    tags = sorted(device_tags.keys())
    assert len(tags) > 0
    if len(tags) == 1 and tags[0][-1] == 'X':
      newtag = tags[0][:-1] + 'Z'
      device_tags[newtag] = device_tags[tags[0]]
      tags[0] = newtag
    devices = [ {"device": tag, "config": config, "num_batches": device_tags[tag][0], "update_specs" : {'update_rule' : 'global' if device_tags[tag][1] else 'none'}} for tag in tags ]
    if len(devices) == 1 and ngpux > 1:
      devices = devices * ngpux
  return devices

class Device(object):
  def __init__(self, device, config, blocking=False, num_batches=1, update_specs=None):
    """
    :param str device: name, "gpu*" or "cpu*"
    :param Config.Config config: config
    :param bool blocking: False -> multiprocessing, otherwise its blocking
    :param int num_batches: num batches to train on this device
    :param dict update_specs
    """
    self.num_batches = num_batches
    self.blocking = blocking
    self.config = config
    self.output = None; " :type: list[numpy.ndarray] "
    self.outputs_format = None; " :type: list[str] "  # via self.result()
    self.train_outputs_format = None; " :type: list[str] "  # set via self.initialize()
    self.run_called_count = 0
    self.result_called_count = 0
    self.wait_for_result_call = False
    self.compute_total_time = 0
    #self.num_frames = NumbersDict(0)
    self.num_updates = 0
    self.epoch = None
    if not update_specs: update_specs = {}
    update_specs.setdefault('update_rule', 'global')
    update_specs.setdefault('update_params', {})
    update_specs.setdefault('layers', [])
    update_specs.setdefault('block_size', 0)
    self.update_specs = update_specs
    self.main_pid = os.getpid()

    self.name = device
    self.initialized = False
    start_new_thread(self.startProc, (device,))

  def __str__(self):
    async_str = "async (pid %i, ppid %i)" % (os.getpid(), os.getppid())
    return "<Device %s %s>" % (self.name, async_str)

  def startProc(self, *args, **kwargs):
    import better_exchook
    better_exchook.install()
    try:
      self._startProc(*args, **kwargs)
    except BaseException:
      try:
        sys.excepthook(*sys.exc_info())
      finally:
        # Exceptions are fatal. Stop now.
        interrupt_main()

  def _startProc(self, device_tag):
    assert not self.blocking
    if self.name[0:3] == "cpu":
      dev = torch.device("cpu")
    else:
      dev = torch.device("cuda:%d" % int(device_tag[3:]))
    env_update = {} # TODO
    self.proc = AsyncTask(
      func=self.process,
      name="Device %s proc" % self.name,
      mustExec=True,
      env_update=env_update)
    # The connection (duplex pipe) is managed by AsyncTask.
    self.input_queue = self.output_queue = self.proc.conn

    try:
      self.id = self.output_queue.recv(); """ :type: int """
      self.device_name = self.output_queue.recv(); """ :type: str """
      self.num_train_params = self.output_queue.recv(); """ :type: int """  # = len(trainnet.gparams)
      self.sync_used_targets()
    except ProcConnectionDied as e:
      print("Device proc %s (%s) died: %r" % (self.name, device_tag, e), file=log.v3)
      interrupt_main()
    if self.device_name in get_device_attributes().keys():
      self.attributes = get_device_attributes()[self.device_name]
    else:
      self.attributes = get_device_attributes()['default']
    self.name = device_tag[0:3] + str(self.id)
    self.initialized = True

  def detect_nan(self, i, node, fn):
    for output in fn.outputs:
      if numpy.isnan(output[0]).any():
        print(('Inputs : %s' % [input[0] for input in fn.inputs]))
        print(('Outputs: %s' % [output[0] for output in fn.outputs]))
        assert False, '*** NaN detected ***'

  def initialize(self, config, update_specs=None, json_content=None, train_param_args=None):
    """
    :type config: Config.Config
    :type json_content: dict[str] | str | None
    :type train_param_args: dict | None
    """
    if not update_specs: update_specs = {}
    update_specs.setdefault('update_rule', 'global')
    update_specs.setdefault('update_params', {})
    update_specs.setdefault('block_size', 0)
    update_specs.setdefault('layers', [])
    self.update_specs = update_specs
    self.block_size = update_specs['block_size']
    self.total_cost = 0
    target = config.value('target', 'classes')
    assert os.getpid() != self.main_pid # this won't work on Windows
    import h5py
    self.network_task = config.value('task', 'train')
    eval_flag = self.network_task in ['eval', 'forward']
    if json_content is not None:
      self.network = LayerNetwork.from_json_and_config(json_content, config) # TODO
    elif config.bool('initialize_from_model', False) and config.has('load'):
      model = h5py.File(config.value('load', ''), "r")
      self.network = LayerNetwork.from_hdf_model_topology(model, **PTLayerNetwork.init_args_from_config(config))
      model.close()
    else:
      self.network = LayerNetwork.from_config_topology(config)
    self.network.to(self.device)
    if config.has('load'):
      model = h5py.File(config.value('load', ''), "r")
      if 'update_step'in model.attrs:
        self.network.update_step = model.attrs['update_step']
      model.close()
    # initialize batch
    self.used_data_keys = ['data', 'classes']  # TODO #self.trainnet.get_used_data_keys()
    self.data = { k : numpy.zeros((1,), dtype='int32' if k == 'classes' else 'float32') for k in self.used_data_keys }
    self.index = { k : numpy.zeros((1,), dtype='uint8') for k in self.used_data_keys }
    print("Device train-network: Used data keys:", self.used_data_keys, file=log.v4)
    assert "data" in self.used_data_keys
    # TODO
    #self.data = {k: theano.shared(numpy.zeros((1,) * self.trainnet.y[k].ndim, dtype=self.trainnet.y[k].dtype),
    #                           borrow=True, name='y_%s' % k)
    #          for k in self.used_data_keys}
    #self.j = {k: theano.shared(numpy.zeros((1, 1), dtype='int8'), borrow=True, name='j_%s' % k)
    #          for k in self.used_data_keys}
    if log.verbose[4]: progress_bar()

    # initialize functions
    self.updater = None # TODO
    self.updater = optim.Adam(self.network.parameters(), lr = 0.0005)
    self.update_specs = update_specs

    self.forwarder = None
    self.use_inputs = False
    if self.network_task  == 'train':
      #if self.update_specs['update_rule'] == 'global':
      #  self.updater = Updater.initFromConfig(self.config)
      #elif self.update_specs['update_rule'] != 'none':
      #  self.updater = Updater.initRule(self.update_specs['update_rule'], **self.update_specs['update_params'])

      outputs = []
      #self.train_outputs_format = ["cost:" + out for out in sorted(self.trainnet.costs.keys())]
      self.train_outputs_format = ["cost:classes"]
      #self.updater.initVars(self.trainnet, self.gradients)

      # TODO
      #self.trainer = self.trainnet.theano.function(inputs=[self.block_start, self.block_end],
      #                               outputs=outputs,
      #                               givens=train_givens,
      #                               updates=self.updater.getUpdateList(),
      #                               on_unused_input=config.value('theano_on_unused_input', 'ignore'),
      #                               no_default_updates=exclude,
      #                               name="train_and_updater")

      self.test_outputs_format = ["cost:classes","error:classes"]
      #test_outputs = [self.testnet.errors[out] for out in sorted(self.testnet.errors.keys())]
      # TODO
      #self.tester = theano.function(inputs=[self.block_start, self.block_end],
      #                              outputs=test_outputs,
      #                              givens=test_givens,
      #                              on_unused_input=config.value('theano_on_unused_input', 'ignore'),
      #                              no_default_updates=True,
      #                              name="tester")
    elif self.network_task == 'forward':
      output_layer_name = config.value("extract_output_layer_name", "output")
      extractions = config.list('extract', ['log-posteriors'])
      givens = self.make_input_givens(self.testnet)
      for extract in extractions:
        param = None
        if ':' in extract:
          param = extract.split(':')[1]
          extract = extract.split(':')[0]
        elif extract == "log-posteriors":
          pass #TODO
        else:
          assert False, "invalid extraction: " + extract

      # TODO
      # self.extractor = theano.function(inputs = [],
      #                                  outputs = source if len(source) == 1 else [T.concatenate(source, axis=-1)],
      #                                  givens = givens,
      #                                  name = "extractor")

  def compute_run(self, task):
    compute_start_time = time.time()
    batch_dim = self.data["data"].shape[1]
    block_size = self.block_size if self.block_size else batch_dim
    if self.config.bool("debug_shell_first_compute", False):
      print("debug_shell_first_compute", file=log.v1)
      Debug.debug_shell(user_ns=locals(), user_global_ns=globals())
    if task in [ "train", "eval" ]:
      if task == 'train':
        self.network.zero_grad()
        self.network.train()
        self.network.exec()
      elif task == 'eval':
        self.network.eval()
        with torch.no_grad():
          self.network.exec()
      else:
        raise NotImplementedError
      costs = self.network.cost()
      if task == 'train': # TODO
        for i in range(len(costs)):
          costs[i].backward()
        self.updater.step()
      for i in range(len(costs)):
        costs[i] *= self.network.output['output'].index_out.float().sum().cpu()
      output = costs + (self.network.errors() if task != 'train' else [])
      for i in range(len(output)):
        output[i] = output[i].detach().cpu()
    else:
      assert False, "invalid command: " + task
    compute_end_time = time.time()
    if self.config.bool("debug_batch_compute_time", False):
      print("batch compute time:", compute_end_time - compute_start_time, file=log.v1)
    self.compute_total_time += compute_end_time - compute_start_time
    # output is a list the outputs which we specified when creating the Theano function in self.initialize().
    assert len(output) > 0  # In all cases, we have some output.
    outputs_format = None
    if task.startswith("train"):
      outputs_format = self.train_outputs_format
    elif task == "eval":
      outputs_format = self.test_outputs_format

    if outputs_format:
      for fmt, out in zip(outputs_format, output):
        if fmt.startswith('cost:'):
          self.total_cost += out

    return output, outputs_format

  def get_compute_func(self, task):
    if task == "train":
      return self.trainer
    raise NotImplementedError("for task: %r" % task)

  def process(self, asyncTask):
    """
    :type asyncTask: AsyncTask
    """
    device = self.name
    config = self.config
    global asyncChildGlobalDevice, deviceInstance
    asyncChildGlobalDevice = self
    deviceInstance = self
    try:
      # We do some minimal initialization, modelled after rnn.init().
      # This is needed because we are a new independent process. See startProc().
      import rnn
      rnn.initBetterExchook()
      rnn.config = config
      rnn.initLog()
      print("Device %s proc starting up, pid %i" % (device, os.getpid()), file=log.v3)
      print("Device %s proc: THEANO_FLAGS = %r" % (device, os.environ.get("THEANO_FLAGS", None)), file=log.v4)
      rnn.initFaulthandler()
      rnn.initConfigJsonNetwork()
      self.process_inner(device, config, self.update_specs, asyncTask)
    except ProcConnectionDied as e:
      print("Device %s proc, pid %i: Parent seem to have died: %s" % (device, os.getpid(), e), file=log.v2)
      sys.exit(1)
    except KeyboardInterrupt:
      # Killed by parent.
      print("Device %s proc got KeyboardInterrupt" % device, file=log.v4)
      sys.exit(1)
    except Exception as e:
      print("Device %s proc exception: %s" % (device, e), file=log.v2)
      sys.excepthook(*sys.exc_info())
      sys.exit(1)

  def process_inner(self, device, config, update_specs, asyncTask):
    """
    :type device: str
    :type config: Config.Config
    :type asyncTask: AsyncTask
    """
    # The connection (duplex pipe) is managed by AsyncTask.
    output_queue = input_queue = asyncTask.conn
    device_id = device.split(':')[-1]
    device_name = "default"
    output_queue.send(device_id) # TODO
    output_queue.send(device_name) # TODO

    self.device = torch.device(device_id.replace('gpu','cuda:'))

    custom_dev_init_code = config.value('custom_dev_init_code', None, list_join_str="\n")
    if custom_dev_init_code:
      custom_exec(custom_dev_init_code, "<custom dev init code string>", {}, dict_joined(globals(), locals()))

    self.initialize(config, update_specs=update_specs)
    output_queue.send(len(list(self.network.named_parameters())))
    print("Device %s proc, pid %i is ready for commands." % (device, os.getpid()), file=log.v4)
    network_params = []
    while True:
      cmd = input_queue.recv()
      if cmd == "stop":  # via self.terminate()
        output_queue.send("done")
        break
      elif cmd == "generic-exec":
        args = input_queue.recv()
        res = self._generic_exec(*args)
        output_queue.send("generic-exec-result")
        output_queue.send(res)
      elif cmd == "reset":  # via self.reset()
        self.epoch = input_queue.recv()
        #self.epoch_var.set_value(self.epoch)
        #if self.updater: # TODO
          #self.updater.reset()
      elif cmd == "reinit":  # via self.reinit()
        json_content = input_queue.recv()
        train_param_args = input_queue.recv()
        if self.need_reinit(json_content=json_content, train_param_args=train_param_args):
          self.initialize(config, update_specs=update_specs,
                          json_content=json_content, train_param_args=train_param_args)
        output_queue.send("reinit-ready")
        output_queue.send(len(self.trainnet.train_params_vars))
      elif cmd == "update-data":  # via self.update_data()
        t = {}
        target_keys = input_queue.recv()
        for k in target_keys:
          t[k] = input_queue.recv()
        for k in target_keys:
          self.index[k] = input_queue.recv()
          #self.index[k][self.index[k]==-1] = 0
        self.tags = input_queue.recv()
        update_start_time = time.time()
        # self.x == self.data["data"], will be set also here.
        for k in target_keys:
          self.data[k] = t[k].astype(self.data[k].dtype)
        for k in target_keys:
          self.network.index[k] = torch.from_numpy(self.index[k]).to(self.device)
          self.network.data[k] = torch.from_numpy(self.data[k]).to(self.device)
        continue # TODO

        for k in target_keys:
          self.data[k].set_value(t[k].astype(self.data[k].dtype), borrow = True)
        #self.c.set_value(c.astype('int32'), borrow = True)
        for k in target_keys:
          self.j[k].set_value(self.output_index[k].astype('int8'), borrow = True)
        try:
          self.tags_var.set_value(numpy.array(self.tags).view(dtype='int8').reshape((len(self.tags), max(map(len, self.tags)))))
        except:
          tags = [s.encode('utf-8') for s in self.tags]
          self.tags_var.set_value(numpy.array(tags).view(dtype='int8').reshape((len(tags), max(map(len, tags)))))
        self.update_total_time += time.time() - update_start_time
      elif cmd == "set-learning-rate":  # via self.set_learning_rate()
        learning_rate = input_queue.recv()
        #if self.updater: # TODO
        #  self.updater.setLearningRate(learning_rate)
      elif cmd == "set-net-params":  # via self.set_net_params()
        self.total_cost = 0
        #our_params_trainnet = self.trainnet.get_all_params_vars()
        #our_params_testnet = self.testnet.get_all_params_vars()
        #assert isinstance(our_params_trainnet, list)
        params_len = input_queue.recv()
        #net_params = [ None for x in self.network.parameters()]
        self.network.set_parameters([
          torch.from_numpy(numpy.fromstring(input_queue.recv_bytes(), dtype='float32'))
            for i in range(params_len)])
        assert input_queue.recv() == "end-set-net-params"
        continue # TODO
        assert len(params) == len(our_params_trainnet)
        if self.testnet_share_params:
          assert len(our_params_testnet) == 0
        else:
          assert len(params) == len(our_params_testnet)
        for i in range(params_len):
          param_str = params[i]
          param = numpy.fromstring(param_str, dtype=floatX)
          our_p_train = our_params_trainnet[i]
          our_param_shape = our_p_train.get_value(borrow=True, return_internal_type=True).shape
          assert numpy.prod(our_param_shape) == numpy.prod(param.shape)
          #assert numpy.isfinite(param).all()
          converted = param.reshape(our_param_shape)
          our_p_train.set_value(converted)
          if not self.testnet_share_params:
            our_params_testnet[i].set_value(converted)
      elif cmd == 'get-num-updates':
        if self.updater and False: # TODO
          output_queue.send(int(self.updater.i.get_value()))
        else:
          output_queue.send(0)
      elif cmd == 'get-total-cost':
        output_queue.send(self.total_cost)
      elif cmd == "get-net-train-params":  # via self.get_net_train_params()
        output_queue.send("net-train-params")
        output_queue.send(len(network_params))
        for p in network_params:
          output_queue.send_bytes(p)
        output_queue.send("end-get-net-train-params")
      elif cmd == "sync-net-train-params":
        network_params = []
        for p in self.network.parameters():
          network_params.append(p.detach().cpu().numpy().tostring())
      elif cmd == "task":  # via self.run()
        task = input_queue.recv()
        try:
          output, outputs_format = self.compute_run(task)
        except RuntimeError:
          print("warning: Runtime error on device", device_name, file=log.v2)
          output_queue.send("error")
          sys.excepthook(*sys.exc_info())
          # If there are any other than the main thread.
          # Actually, that would be unexpected.
          Debug.dumpAllThreadTracebacks()
          return
        except MemoryError:
          output_queue.send("error")
          raise
        output_queue.send("task-result")
        # We can get cuda_ndarray or other references to internal device memory.
        # We explicitly want to copy them over to CPU memory.
        output_queue.send([numpy.asarray(v) for v in output])
        #output_queue.send(output)
        output_queue.send(outputs_format)
      else:
        raise Exception("cmd %s unknown" % cmd)

  def sync_net_train_params(self):
    if not self.blocking:
      self.input_queue.send("sync-net-train-params")

  def get_net_train_params(self, network):
    assert self.main_pid == os.getpid()
    self.input_queue.send("get-net-train-params")
    r = self.output_queue.recv()
    assert r == "net-train-params"
    param_count = self.output_queue.recv()
    assert param_count == len(network.parameters())
    raw = [self.output_queue.recv_bytes() for i in range(param_count)]
    assert self.output_queue.recv() == "end-get-net-train-params"
    vars = network.parameters()
    res = []
    assert len(vars) == len(raw)
    for p,q in zip(vars, raw):
      res.append(numpy.fromstring(q, dtype=floatX).reshape(p.detach().numpy().shape))
    return res

  def set_net_encoded_params(self, network_params):
    """
    :type network_params: list[numpy.ndarray]
    This updates *all* params, not just the train params.
    """
    self.input_queue.send("set-net-params")
    self.input_queue.send(len(network_params))
    for p in network_params:
      self.input_queue.send_bytes(p.astype(floatX).tostring())
    self.input_queue.send("end-set-net-params")

  def set_net_params(self, network):
    """
    :type network: Network.LayerNetwork
    This updates *all* params, not just the train params.
    """
    assert self.main_pid == os.getpid()
    self.set_net_encoded_params([ p.detach().cpu().numpy() for p in network.parameters() ])

  def is_device_proc(self):
    return self.main_pid != os.getpid()

  def get_task_network(self):
    """
    :rtype: LayerNetwork
    """
    if self.network_task == "train":
      return self.trainnet
    else:
      return self.testnet

  def _host__get_used_targets(self):
    assert self.is_device_proc()
    return self.used_data_keys

  def sync_used_targets(self):
    """
    Updates self.used_targets for the host.
    """
    if self.is_device_proc():
      return  # Nothing to do.
    self.used_data_keys = self._generic_exec_on_dev("_host__get_used_targets")  # type: list[str]

  def alloc_data(self, shapes, max_ctc_length=0):
    """
    :param dict[str,list[int]] shapes: by data-key. format usually (time,batch,features)
    :type max_ctc_length: int
    """
    assert self.main_pid == os.getpid()
    assert all([s > 0 for s in shapes["data"]])
    self.targets = {k: numpy.full(shapes[k], -1, dtype=floatX) for k in self.used_data_keys}
    self.output_index = {k: numpy.zeros(shapes[k][0:2], dtype='uint8') for k in self.used_data_keys}
    self.tags = [None] * shapes["data"][1]  # type: list[str]  # seq-name for each batch slice

  def is_device_proc(self):
    if self.main_pid == os.getpid():
      return False  # We are on the host.
    return True  # We are the child proc.

  def _generic_exec(self, func_name, args, kwargs):
    assert self.is_device_proc()
    func = attr_chain(self, func_name)
    ret = func(*args, **kwargs)
    return ret

  def _generic_exec_on_dev(self, func_name, *args, **kwargs):
    assert not self.wait_for_result_call
    if self.is_device_proc():
      return self._generic_exec(func_name, args, kwargs)
    self.input_queue.send("generic-exec")
    self.input_queue.send((func_name, args, kwargs))
    r = self.output_queue.recv()
    assert r == "generic-exec-result"
    r = self.output_queue.recv()
    return r

  def update_data(self):
    assert self.main_pid == os.getpid()
    self.input_queue.send("update-data")
    target_keys = list(sorted(self.used_data_keys))
    self.input_queue.send(target_keys)
    for target in target_keys:
      self.input_queue.send(self.targets[target])
    for k in target_keys:
      self.input_queue.send(self.output_index[k])
    self.input_queue.send(self.tags)

  def set_learning_rate(self, learning_rate):
    """
    :type learning_rate: float
    """
    assert self.main_pid == os.getpid()
    self.input_queue.send("set-learning-rate")
    self.input_queue.send(learning_rate)

  def get_num_updates(self):
    assert self.main_pid == os.getpid()
    self.input_queue.send("get-num-updates")
    return int(self.output_queue.recv())

  def get_total_cost(self):
    assert self.main_pid == os.getpid()
    self.input_queue.send("get-total-cost")
    return float(self.output_queue.recv())

  def start_epoch_stats(self):
    if not self.is_device_proc():
      return self._generic_exec_on_dev("start_epoch_stats")
    self.epoch_start_time = time.time()
    self.compute_total_time = 0
    self.update_total_time = 0

  def finish_epoch_stats(self):
    if not self.is_device_proc():
      return self._generic_exec_on_dev("finish_epoch_stats")
    cur_time = time.time()
    total_time = cur_time - self.epoch_start_time
    total_time = max(total_time, 0.001)
    compute_frac = self.compute_total_time / total_time
    update_frac = self.update_total_time / total_time
    print("Device %s proc epoch time stats: total %s, %.02f%% computing, %.02f%% updating data" % \
                     (self.name, hms(total_time), compute_frac * 100, update_frac * 100), file=log.v4)

  def prepare(self, network, updater=None, train_param_args=None, epoch=None):
    """
    Call this from the main proc before we do anything else.
    This is called before we start any training, e.g. at the begin of an epoch.
    :type network: LayerNetwork
    :type updater: Updater | None
    :type train_param_args: dict | None
    """
    assert self.main_pid == os.getpid(), "Call this from the main proc."
    # Reinit if needed.
    #self.reinit(json_content=network.to_json_content(), train_param_args=train_param_args)
    self.set_net_params(network)
    self.epoch = epoch
    self.input_queue.send('reset')
    self.input_queue.send(epoch)

  def run(self, task):
    """
    :type task: str
    """
    self.task = task
    self.run_called_count += 1
    self.update_data()
    assert not self.wait_for_result_call
    self.wait_for_result_call = True

    self.output = None
    self.outputs_format = None
    self.input_queue.send("task")
    self.input_queue.send(task)

  def clear_memory(self, network):
    self.update_data()

  @staticmethod
  def make_result_dict(output, outputs_format):
    """
    :type output: list[numpy.ndarray]
    :type outputs_format: list[str]
    """
    assert len(output) == len(outputs_format)
    return dict(zip(outputs_format, output))

  def result(self):
    """
    :rtype: (list[numpy.ndarray], list[str] | None)
    :returns the outputs and maybe a format describing the output list
    See self.make_result_dict() how to interpret this list.
    See self.initialize() where the list is defined.
    """
    assert self.wait_for_result_call
    self.result_called_count += 1
    assert self.main_pid == os.getpid()
    assert self.result_called_count <= self.run_called_count
    if not self.proc.is_alive():
      print("Dev %s proc not alive anymore" % self.name, file=log.v4)
      return None, None
    # 60 minutes execution timeout by default
    timeout = self.config.float("device_timeout", 60 * 60)
    while timeout > 0:
      try:
        if self.output_queue.poll(1):
          r = self.output_queue.recv()
          if r == "error":
            print("Dev %s proc reported error" % self.name, file=log.v5)
            self.wait_for_result_call = False
            return None, None
          assert r == "task-result"
          output = self.output_queue.recv()
          outputs_format = self.output_queue.recv()
          assert output is not None
          self.wait_for_result_call = False
          return output, outputs_format
      except ProcConnectionDied as e:
        # The process is dying or died.
        print("Dev %s proc died: %s" % (self.name, e), file=log.v4)
        self.wait_for_result_call = False
        return None, None
      timeout -= 1
    print("Timeout (device_timeout = %s) expired for device %s" % (self.config.float("device_timeout", 60 * 60), self.name), file=log.v3)
    try:
      os.kill(self.proc.proc.pid, signal.SIGUSR1)
    except Exception as e:
      print("os.kill SIGUSR1 exception: %s" % e, file=log.v3)
    return None, None

  def forward(self, use_trainnet=False):
    assert self.is_device_proc()
    network = self.trainnet if use_trainnet else self.testnet
    if not self.forwarder:
      print("Device: Create forwarder, use trainnet:", use_trainnet, ", testnet_share_params:", self.testnet_share_params, file=log.v3)
      # # TODO
      #self.forwarder = theano.function(
      #  inputs=[],
      #  outputs=[layer.output for name, layer in sorted(network.output.items())],
      #  givens=self.make_input_givens(network),
      #  on_unused_input='warn',
      #  name="forwarder")
    assert len(outputs) == len(network.output)
    return {name: outputs[i] for i, (name, layer) in enumerate(sorted(network.output.items()))}

  def terminate(self):
    if not self.proc:
      return
    if not self.proc.is_alive():
      return
    assert self.main_pid == os.getpid()
    try:
      self.input_queue.send('stop')
    except ProcConnectionDied:
      pass
    self.proc.join(timeout=10)
    self.proc.terminate()
    self.proc = None

  # device properties
  def get_device_shaders(self):
    return self.attributes[0]

  def get_device_clock(self):
    return self.attributes[1]

  def get_device_memory(self):
    return self.attributes[2]
