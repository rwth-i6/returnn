from multiprocessing import Process, Queue
from Queue import Empty
from Util import cmd, progress_bar
from Log import log
from Network import LayerNetwork, GateLstmLayer
import numpy
import pynvml

def get_num_devices():
  return len(cmd('cat /proc/cpuinfo | grep processor')), len(cmd('nvidia-smi -L'))

def get_gpu_names():
  return cmd('nvidia-smi -L | cut -d \'(\' -f 1 | cut -d \' \' -f 3- | sed -e \'s/\\ $//\'')

def get_device_attributes():
  attributes = { "GeForce GTX 780" : (2304, 980, 3 * 1024 * 1024 * 1024),
                 "GeForce GTX 680" : (1536, 1020, 2 * 1024 * 1024 * 1024),
                 "GeForce GTX 970" : (1664, 1178, 4 * 1024 * 1024 * 1024),
                 "GeForce GTX TITAN" : (2688, 837, 6 * 1024 * 1024 * 1024),
                 "GeForce GTX 580" : (512, 1714, 2 * 1024 * 1024 * 1024),
                 "Tesla K20c" : (2496, 706, 5 * 1024 * 1024 * 1024), 
                 "GeForce GT 630M" : (96, 672, 2 * 1024 * 1024 * 1024),
                 "GeForce GTX 750 Ti" : (640, 1110, 2 * 1024 * 1024 * 1024)}
  #return int(cmd("grep NVIDIA /var/log/Xorg.0.log | grep Memory | head -n "+str(device + 1)+" | tail -n 1 | cut -d ' ' -f 7")[0]) * 1024
  cpu = 0
  #for clock in cmd('cat /proc/cpuinfo | grep "model name" | cut -d \'@\' -f 2 | tr -d \' \' | sed -e s/GHz//'):
  for clock in cmd('cat /proc/cpuinfo | grep "cpu MHz" | cut -d \':\' -f 2 | sed \'s/^\\ //\''):
    attributes["cpu" + str(cpu)] = (1, int(float(clock)), 2 * 1024 * 1024 * 1024)
    cpu += 1
  attributes["cpu127"] = (1, 1, 32 * 1024 * 1024 * 1024)
  return attributes

class Device():
  def __init__(self, device, config, blocking = False, num_batches = 1):
    pynvml.nvmlInit()
    self.input_queue = Queue()
    self.output_queue = Queue()
    self.num_batches = num_batches
    self.blocking = blocking
    if blocking:
      self.initialize(config)
      self.nparams = len(self.trainnet.gparams)
      if device[0:3] == 'gpu':
        import cuda_ndarray.cuda_ndarray as cuda
        self.id = cuda.active_device_number()
        self.device_name = cuda.active_device_name()
      else:
        self.id = 0
        self.device_name = 'cpu' + str(self.id)
    else:
      self.proc = Process(target = self.process, args = (device, config, self.input_queue, self.output_queue))
      self.proc.daemon = True
      self.proc.start()
      self.id = self.output_queue.get()
      self.device_name = self.output_queue.get()
      self.nparams = self.output_queue.get()
    self.attributes = get_device_attributes()[self.device_name]
    self.name = device[0:3] + str(self.id)
    self.config = config
    
  def restart(self):
    self.proc.terminate()
    #os.kill(self.proc.pid, signal.SIGKILL)
    self.proc = Process(target = self.process, args = (self.name, self.config, self.input_queue, self.output_queue))
    #self.proc.daemon = True
    while not self.input_queue.empty(): self.input_queue.get()
    while not self.output_queue.empty(): self.output_queue.get()
    self.proc.start()

  def detect_nan(self, i, node, fn):
    import theano
    for output in fn.outputs:
      if numpy.isnan(output[0]).any():
        #theano.printing.debugprint(node)
        print 'Inputs : %s' % [input[0] for input in fn.inputs]
        print 'Outputs: %s' % [output[0] for output in fn.outputs]
        assert False, '*** NaN detected ***'
    
  def initialize(self, config):
    import theano
    import theano.tensor as T
    import h5py
    self.network_task = config.value('task', 'train')
    mask = "unity"
    if sum(config.float_list('dropout', [0])) > 0.0:
      mask = "dropout"
    if config.bool('initialize_from_model', False) and config.has('load'):
      model = h5py.File(config.value('load', ''), "r")
      self.trainnet = LayerNetwork.from_model(model, mask)
      self.testnet = LayerNetwork.from_model(model, "unity")
      model.close()
    else:
      self.trainnet = LayerNetwork.from_config(config, mask)
      self.testnet = LayerNetwork.from_config(config, "unity")
    # initialize batch
    self.x = theano.shared(numpy.zeros((1, 1, 1), dtype = theano.config.floatX), borrow=True)
    self.y = theano.shared(numpy.zeros((1,), dtype = 'int32'), borrow=True)
    self.i = theano.shared(numpy.zeros((1, 1), dtype = 'int8'), borrow=True)
    if self.trainnet.loss == 'ctc':
      self.cp = theano.shared(numpy.zeros((1, 1), dtype = theano.config.floatX), borrow=True)
      self.c = T.cast(self.cp, 'int32')
    gparams = []
    for pi, param in enumerate(self.trainnet.gparams):
      if log.verbose[4]: progress_bar(float(pi) / len(self.trainnet.gparams), "calculating gradients ...")
      gparam = T.grad(self.trainnet.objective, param, known_grads = self.trainnet.known_grads)
      if False and param.name == 'lambda':
        f = theano.function(inputs = [],
                            outputs = [gparam],
                            givens = self.make_givens(self.trainnet))
        print >> log.v3, theano.printing.pp(gparam)
        print >> log.v3, "-------------------------------------------"
        print >> log.v3, theano.printing.pp(f.maker.fgraph.outputs[0])
      gparams.append(theano.Out(gparam, borrow = True))
    if log.verbose[4]: progress_bar()
    # initialize functions
    if self.network_task == 'train' or self.network_task == 'theano_graph':
      if self.trainnet.loss == 'ctc':
        train_givens = self.make_ctc_givens(self.trainnet)
        test_givens = self.make_ctc_givens(self.testnet)
      elif self.trainnet.loss == 'sprint':
        train_givens = self.make_sprint_givens(self.trainnet)
        test_givens = self.make_givens(self.testnet)
      else:
        train_givens = self.make_givens(self.trainnet)
        test_givens = self.make_givens(self.testnet)

      self.trainer = theano.function(inputs = [],
                                     outputs = [self.trainnet.cost] + gparams,
                                     givens = train_givens, no_default_updates=True)#,
                                     #mode = theano.compile.MonitorMode(post_func=self.detect_nan))

      self.tester = theano.function(inputs = [],
                                    outputs = [self.testnet.cost, self.testnet.errors],
                                    givens = test_givens, no_default_updates=True)
    elif self.network_task == 'forward':
      extractions = config.list('extract', ['log-posteriors'])
      source = []
      givens = self.make_input_givens(self.testnet)
      for extract in extractions:
        if extract == "log-posteriors":
          source.append(T.log(self.testnet.output.p_y_given_x))
        elif extract == "ctc-sil":
          feat = self.testnet.output.p_y_given_x
          feat = feat[:,:-1] #remove blank
          feat = feat / feat.sum(axis=1)[:,numpy.newaxis] #renormalize
          feat = T.log(feat)
          source.append(feat)
      	elif extract == "ce-errsig":
      	  feat = T.grad(self.testnet.cost, self.testnet.output.z) #TODO
      	  source.append(feat)
      	  givens = self.make_givens(self.testnet)
        elif "log-norm-hidden_" in extract:
          idx = int(extract.split('_')[1])
          source.append(T.log(T.nnet.softmax(T.reshape(self.testnet.hidden[idx].output, (self.testnet.hidden[idx].output.shape[0] * self.testnet.hidden[idx].output.shape[1], self.testnet.hidden[idx].output.shape[2])))))
        elif "gates_" in extract:
          idx = int(extract.split('_')[1])
          if idx > 0:
            hidden = self.testnet.hidden[idx - 1]
          else:
            hidden = self.testnet.reverse_hidden[-idx - 1]
          source.append(T.reshape(hidden.input_gate, (hidden.input_gate.shape[0] * hidden.input_gate.shape[1], hidden.input_gate.shape[2])))
          source.append(T.reshape(hidden.forget_gate, (hidden.forget_gate.shape[0] * hidden.forget_gate.shape[1], hidden.forget_gate.shape[2])))
          source.append(T.reshape(hidden.output_gate, (hidden.output_gate.shape[0] * hidden.output_gate.shape[1], hidden.output_gate.shape[2])))
        elif "hidden_" in extract:
          idx = int(extract.split('_')[1])
          if idx > 0:
            hidden = self.testnet.hidden[idx - 1]
          else:
            hidden = self.testnet.reverse_hidden[-idx - 1]
          source.append(T.reshape(hidden.output, (hidden.output.shape[0] * hidden.output.shape[1], hidden.output.shape[2])))
        else: assert False, "invalid extraction: " + extract
      self.extractor = theano.function(inputs = [],
                                       outputs = source,
                                       givens = givens)
    elif self.network_task == 'classify':
      self.classifier = theano.function(inputs = [],
                                        outputs = [T.argmax(self.testnet.output.p_y_given_x, axis = 1)],
                                        givens = self.make_input_givens(self.testnet))
    elif self.network_task == 'analyze':
      self.analyzer = theano.function(inputs = [],
                                      outputs = [self.testnet.output.p_y_given_x],
                                              #+ [self.testnet.jacobian],
                                              #+ [hidden.output for hidden in self.network.hidden]
                                              #+ [hidden.output for hidden in self.network.reverse_hidden],
                                      givens = self.make_input_givens(self.testnet))
  def compute(self, cmd):
    if cmd == "train":
      proc = self.trainer
    elif cmd == "eval":
      proc = self.tester
    elif cmd == "extract":
      proc = self.extractor
    elif cmd == 'classify':
      proc = self.classifier
    elif cmd == "analyze":
      proc = self.analyzer
    else: assert False, "invalid command: " + cmd
    return proc
  
  def process(self, device, config, input_queue, output_queue):
    if device[0:3] == 'gpu':
      import theano.sandbox.cuda
      import cuda_ndarray.cuda_ndarray as cuda
      if device == 'gpuX': device = 'gpu'
      theano.sandbox.cuda.use(device, force = True)
      #theano.sandbox.cuda.use(device, force = True, default_to_move_computation_to_gpu=True, move_shared_float32_to_gpu=True, enable_cuda=True)
      device_id = cuda.active_device_number()
      device_name = cuda.active_device_name()
    else:
      try:
        device_id = int(device[3:])
      except ValueError:
        device_id = 0
      device_name = 'cpu' + str(device_id)
    output_queue.put(device_id)
    output_queue.put(device_name)
    self.initialize(config)
    output_queue.put(len(self.trainnet.gparams))
    while True:
      cmd = input_queue.get()
      if cmd == "stop":
        output_queue.put("done")
        break
      elif cmd == "update":
        x = input_queue.get()
        t = input_queue.get()
        i = input_queue.get()
        if self.trainnet.loss == 'ctc':
          c = input_queue.get()
          self.cp.set_value(c)
        self.x.set_value(x.astype('float32'), borrow = True)
        self.y.set_value(t.astype('int32'), borrow = True)
        self.i.set_value(i.astype('int8'), borrow = True)
      else:
        params = input_queue.get()
        try:
          if cmd == "train": self.trainnet.set_params(params)
          else: self.testnet.set_params(params)
          result = self.compute(cmd)()
        except RuntimeError:
          print >> log.v2, "warning: Runtime error on device", device_name
          output_queue.put("error")
          return
        except MemoryError:
          output_queue.put("error")
          raise
        for output in result:
          output_queue.put(output)

  def update_data(self):
    if self.blocking:
      self.x.set_value(self.data, borrow = True)
      #self.t.set_value(self.targets, borrow = True)
      self.t.set_value(self.targets.flatten(), borrow = True)
      self.i.set_value(self.index, borrow = True)
      if self.trainnet.loss == 'ctc':
        self.cp.set_value(self.ctc_targets)
    else:
      self.input_queue.put("update")
      self.input_queue.put(self.data)
      self.input_queue.put(self.targets.flatten())
      self.input_queue.put(self.index)
      if self.config.value('loss','') == 'ctc':
        self.input_queue.put(self.ctc_targets)
  
  def run(self, task, network):
    self.task = task
    self.update_data()
    if self.blocking:
      if task == "train": self.trainnet.set_params(network.get_params())
      else: self.testnet.set_params(network.get_params())
      self.output = self.compute(task)()
    else:
      self.input_queue.put(task)
      self.input_queue.put(network.get_params())
      
  def clear_memory(self, network):
    #self.data = numpy.zeros((1, 1, 1), dtype = theano.config.floatX)
    #self.targets = numpy.zeros((1, 1), dtype = theano.config.floatX)
    #self.index = numpy.zeros((1, 1), dtype = theano.config.floatX)
    self.update_data()

  def result(self):
    if not self.blocking:
      self.output = []
      timeout = 60 # 5 minutes execution timeout
      while timeout > 0:
        try:
          score = self.output_queue.get(timeout = 5)
          if score == "error": return None
          self.output.append(score)
          break
        except Empty:
          if not self.proc.is_alive():
            return None
        timeout -= 1
      if timeout == 0:
        print >> log.v3, "Timeout expired for device", self.name
        return None
      if self.task == 'train':
        self.output += [ self.output_queue.get() for p in xrange(self.nparams) ]
      elif self.task == "eval":
        self.output.append(self.output_queue.get())
    return self.output 
    
  def terminate(self):
    if not self.blocking and self.proc.is_alive():
      self.input_queue.put('stop')
      self.proc.join()
      self.proc.terminate()
  # device properties
  def get_device_shaders(self): return self.attributes[0]
  def get_device_clock(self): return self.attributes[1]
  def get_device_memory(self): return self.attributes[2]
  def update_memory(self):
    self.memory = self.attributes[2] - 512 * 1024 * 1024
    if self.name[0:3] != 'cpu':
      self.memory = int(cmd("nvidia-smi -i "+ str(self.id) + " -q | grep -A 3 \"Memory Usage\" | tail -n 1 | cut -d ':' -f 2 | cut -d ' ' -f 2")[0])
    return self.memory

  def get_memory_info(self):
    hmap = [2, 3, 1, 0]
    handle = pynvml.nvmlDeviceGetHandleByIndex(hmap[self.id])
    return pynvml.nvmlDeviceGetMemoryInfo(handle)

  def make_givens(self, network):
    return [(network.x, self.x), (network.y, self.y), (network.i, self.i)]
  def make_input_givens(self, network):
    if network.recurrent:
      return [(network.x, self.x), (network.i, self.i)]
    else:
      return [(network.x, self.x)]
  def make_sprint_givens(self, network):
    return [(network.x, self.x), (network.i, self.i)]
  def make_ctc_givens(self, network):
    return [(network.x, self.x), (network.c, self.c), (network.i, self.i)]
