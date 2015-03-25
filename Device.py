from multiprocessing import Process, Pipe
from Updater import Updater
from Util import cmd, progress_bar, obj_diff_str
from Log import log
from Network import LayerNetwork
import numpy
import sys
import os
import errno
import time

def get_num_devices():
  if os.name == 'nt':
    return 1, 1 #TODO
  else:
    return len(cmd('cat /proc/cpuinfo | grep processor')) or 1, len(cmd('nvidia-smi -L'))

def get_gpu_names():
  if os.name == 'nt':
    return "GeForce GTX 770" #TODO
  else:
    return cmd('nvidia-smi -L | cut -d \'(\' -f 1 | cut -d \' \' -f 3- | sed -e \'s/\\ $//\'')

def get_device_attributes():
  # (shaders / CUDA cores, clock in MHz, memory in bytes)
  attributes = { "GeForce GTX 770" : (1536, 1150, 2 * 1024 * 1024 * 1024),
                 "GeForce GTX 780" : (2304, 980, 3 * 1024 * 1024 * 1024),
                 "GeForce GTX 680" : (1536, 1020, 2 * 1024 * 1024 * 1024),
                 "GeForce GTX 970" : (1664, 1178, 4 * 1024 * 1024 * 1024),
                 "GeForce GTX 980" : (2048, 1126, 4 * 1024 * 1024 * 1024),
                 "GeForce GTX TITAN" : (2688, 837, 6 * 1024 * 1024 * 1024),
                 "GeForce GTX 580" : (512, 1714, 2 * 1024 * 1024 * 1024),
                 "Tesla K20c" : (2496, 706, 5 * 1024 * 1024 * 1024),
                 "GeForce GT 630M" : (96, 672, 2 * 1024 * 1024 * 1024),
                 "GeForce GTX 750 Ti" : (640, 1110, 2 * 1024 * 1024 * 1024)}
  #return int(cmd("grep NVIDIA /var/log/Xorg.0.log | grep Memory | head -n "+str(device + 1)+" | tail -n 1 | cut -d ' ' -f 7")[0]) * 1024
  cpu = 0
  #for clock in cmd('cat /proc/cpuinfo | grep "model name" | cut -d \'@\' -f 2 | tr -d \' \' | sed -e s/GHz//'):
  if os.name != 'nt':
    for clock in cmd('cat /proc/cpuinfo | grep "cpu MHz" | cut -d \':\' -f 2 | sed \'s/^\\ //\''):
      attributes["cpu" + str(cpu)] = (1, int(float(clock)), 2 * 1024 * 1024 * 1024)
      cpu += 1
    attributes["cpu127"] = (1, 1, 32 * 1024 * 1024 * 1024)
  if not cpu:
    attributes["cpu0"] = (1, 1000, 2 * 1024 * 1024 * 1024)
  return attributes

class Device():
  def __init__(self, device, config, blocking=False, num_batches=1):
    """
    :param str device: name, "gpu*" or "cpu*"
    :param Config.Config config: config
    :param bool blocking: False -> multiprocessing, otherwise its blocking
    :param int num_batches: num batches to train on this device
    """
    try:
      import pynvml
    except ImportError:
      print "pynvml not available, memory information missing"
    else:
      try:
        pynvml.nvmlInit()
      except Exception as exc:
        print >> log.v3, "nvmlInit failed: %s" % exc
    self.num_batches = num_batches
    self.blocking = blocking
    self.config = config
    self.output = None; " :type: list[numpy.ndarray] "
    self.run_called_count = 0
    self.result_called_count = 0
    self.main_pid = os.getpid()
    if blocking:
      self.initialize(config)
      self.num_train_params = len(self.trainnet.train_params)
      if device[0:3] == 'gpu':
        import theano.sandbox.cuda as theano_cuda
        assert theano_cuda.cuda_available, "Theano CUDA support not available. Check that nvcc is in $PATH."
        if not theano_cuda.cuda_enabled: # already enabled when $THEANO_FLAGS=device=gpu
          if device == 'gpuX': device = 'gpu'
          theano_cuda.use(device=device, force=True)
        try:
          import cuda_ndarray.cuda_ndarray as cuda
        except ImportError as exc:
          raise Exception("Theano CUDA support seems broken: %s" % exc)
        self.id = cuda.active_device_number(); """ :type: int """
        self.device_name = cuda.active_device_name(); """ :type: str """
      else:
        self.id = 0
        self.device_name = 'cpu' + str(self.id)
    else:
      self.name = device
      self.startProc()
    self.attributes = get_device_attributes()[self.device_name]
    self.name = device[0:3] + str(self.id)

  def startProc(self):
    assert not self.blocking
    self.output_queue, self.input_queue = Pipe(duplex=True)
    self.proc = Process(
      target=self.process,
      args=(self.name, self.config, self.input_queue, self.output_queue),
      name="Device %s proc" % self.name)
    self.proc.daemon = True
    self.proc.start()
    # We are the parent process. We send/recv over output_queue.
    # Close input_queue so that the childs get an EOF when it reads and we have died.
    self.input_queue.close()
    self.input_queue = self.output_queue
    self.id = self.output_queue.recv(); """ :type: int """
    self.device_name = self.output_queue.recv(); """ :type: str """
    self.num_train_params = self.output_queue.recv(); """ :type: int """  # = len(trainnet.gparams)

  def restart(self):
    self.proc.terminate()
    #os.kill(self.proc.pid, signal.SIGKILL)
    self.startProc()

  def detect_nan(self, i, node, fn):
    for output in fn.outputs:
      if numpy.isnan(output[0]).any():
        #theano.printing.debugprint(node)
        print 'Inputs : %s' % [input[0] for input in fn.inputs]
        print 'Outputs: %s' % [output[0] for output in fn.outputs]
        assert False, '*** NaN detected ***'

  def initialize(self, config, network_description=None, train_param_args=None):
    """
    :type config: Config.Config
    :type network_description: Network.LayerNetworkDescription | None
    :type train_param_args: dict | None
    """
    if self.blocking:
      assert os.getpid() == self.main_pid
    else:
      assert os.getpid() != self.main_pid
    import theano
    import theano.tensor as T
    import h5py
    self.network_task = config.value('task', 'train')
    mask = "unity"
    if sum(config.float_list('dropout', [0])) > 0.0:
      mask = "dropout"
    if network_description is not None:
      self.trainnet = LayerNetwork.from_description(network_description, mask)
      self.testnet = LayerNetwork.from_description(network_description, "unity")
    elif config.bool('initialize_from_model', False) and config.has('load'):
      model = h5py.File(config.value('load', ''), "r")
      self.trainnet = LayerNetwork.from_hdf_model_topology(model, mask)
      self.testnet = LayerNetwork.from_hdf_model_topology(model, "unity")
      model.close()
    else:
      self.trainnet = LayerNetwork.from_config_topology(config, mask)
      self.testnet = LayerNetwork.from_config_topology(config, "unity")
    if train_param_args is not None:
      self.trainnet.declare_train_params(**train_param_args)
    # initialize batch
    self.x = theano.shared(numpy.zeros((1, 1, 1), dtype = theano.config.floatX), borrow=True)
    self.y = theano.shared(numpy.zeros((1,), dtype = 'int32'), borrow=True)
    self.i = theano.shared(numpy.zeros((1, 1), dtype = 'int8'), borrow=True)
    if self.trainnet.loss in ('ctc','ce_ctc'):
      self.cp = theano.shared(numpy.zeros((1, 1), dtype = theano.config.floatX), borrow=True)
      self.c = T.cast(self.cp, 'int32')
    gparams = []
    self.gradients = {}
    for pi, param in enumerate(self.trainnet.train_params):
      if log.verbose[4]: progress_bar(float(pi) / len(self.trainnet.train_params), "calculating gradients ...")
      gparam = T.grad(self.trainnet.objective, param, known_grads = self.trainnet.known_grads)
      self.gradients[param] = gparam
      if False and param.name == 'lambda':
        f = theano.function(inputs = [],
                            outputs = [gparam],
                            givens = self.make_givens(self.trainnet),
                            name = "f via trainnet")
        print >> log.v3, theano.printing.pp(gparam)
        print >> log.v3, "-------------------------------------------"
        print >> log.v3, theano.printing.pp(f.maker.fgraph.outputs[0])
      gparams.append(theano.Out(gparam, borrow = True))
    if log.verbose[4]: progress_bar()
    # initialize functions
    self.updater = None
    if self.network_task == 'train' or self.network_task == 'theano_graph':
      if self.trainnet.loss == 'ctc':
        train_givens = self.make_ctc_givens(self.trainnet)
        test_givens = self.make_ctc_givens(self.testnet)
      elif self.trainnet.loss == 'ce_ctc':
        train_givens = self.make_givens(self.trainnet)
        test_givens = self.make_ce_ctc_givens(self.testnet)
      elif self.trainnet.loss == 'sprint':
        train_givens = self.make_sprint_givens(self.trainnet)
        test_givens = self.make_givens(self.testnet)
      else:
        train_givens = self.make_givens(self.trainnet)
        test_givens = self.make_givens(self.testnet)

      self.updater = Updater.initFromConfig(config)

      if self.updater.updateOnDevice:
        self.updater.initVars(self.trainnet, self.gradients)
        self.train_and_updater = theano.function(inputs=[],
                                                 outputs=self.trainnet.results,
                                                 givens=train_givens,
                                                 updates=self.updater.getUpdateList(),
                                                 no_default_updates=False,
                                                 name="train_and_updater")

      else:
        self.trainer = theano.function(inputs = [],
                                       outputs = [self.trainnet.cost] + gparams, #TODO handle ctc_priors
                                       givens = train_givens,
                                       no_default_updates=False,
                                       name = "trainer")#,
                                       #mode = theano.compile.MonitorMode(post_func=self.detect_nan))

      self.tester = theano.function(inputs = [],
                                    outputs = [self.testnet.cost, self.testnet.errors],
                                    givens = test_givens,
                                    no_default_updates = True,
                                    name = "tester")
    elif self.network_task == 'forward':
      extractions = config.list('extract', ['log-posteriors'])
      source = []
      givens = self.make_input_givens(self.testnet)
      for extract in extractions:
        if extract == "log-posteriors":
          source.append(T.log(self.testnet.output.p_y_given_x))
        elif extract == "posteriors":
          source.append(self.testnet.output.p_y_given_x)
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
        else:
          assert False, "invalid extraction: " + extract
      self.extractor = theano.function(inputs = [],
                                       outputs = source,
                                       givens = givens,
                                       name = "extractor")
    elif self.network_task == 'classify':
      self.classifier = theano.function(inputs = [],
                                        outputs = [T.argmax(self.testnet.output.p_y_given_x, axis = 1)],
                                        givens = self.make_input_givens(self.testnet),
                                        name = "classifier")
    elif self.network_task == 'analyze':
      self.analyzer = theano.function(inputs = [],
                                      outputs = [self.testnet.output.p_y_given_x],
                                              #+ [self.testnet.jacobian],
                                              #+ [hidden.output for hidden in self.network.hidden]
                                              #+ [hidden.output for hidden in self.network.reverse_hidden],
                                      givens = self.make_input_givens(self.testnet),
                                      name = "analyzer")
  def compute(self, task):
    if task == "train_distributed":
      proc = self.trainer
    elif task == "train_and_update":
      proc = self.train_and_updater
    elif task == "eval":
      proc = self.tester
    elif task == "extract":
      proc = self.extractor
    elif task == 'classify':
      proc = self.classifier
    elif task == "analyze":
      proc = self.analyzer
    else:
      assert False, "invalid command: " + task
    assert proc, "theano.function not initialized for task %s, check self.initialize()" % task
    return proc

  def _checkGpuFuncs(self, device, device_id):
    if device[0:3] != 'gpu': return
    # Check if we use the GPU.
    # http://deeplearning.net/software/theano/tutorial/modes.html
    if self.network_task == "train":
      if self.updater.updateOnDevice:
        theano_func = self.train_and_updater
      else:
        theano_func = self.trainer
    else:
      return  # Too annoying to cover all cases...
    if not any([x.op.__class__.__name__ in ['GpuGemm', 'GpuGemv', 'GpuDot22', 'GpuElemwise']
                for x in theano_func.maker.fgraph.toposort()]):
      print >> log.v1, device + ":", "It seems as if we don't use the GPU although we requested it."
      import theano.printing
      theano.printing.debugprint(theano_func.maker.fgraph.outputs[0])
    else:
      print >> log.v3, device + ":", "Our Theano trainer functions looks like it will run on the GPU."

    try:
      import theano.sandbox.cuda
      theano_cuda = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray
      devProps = theano_cuda.device_properties(device_id)
      print >> log.v3, device + ":", "CUDA version %i" % devProps["driverVersion"]
    except Exception as exc:
      print >> log.v3, device + ":", "Exception while getting CUDA information. %s" % exc

  def process(self, device, *args, **kwargs):
    """
    :type device: str
    """
    try:
      self.process_inner(device, *args, **kwargs)
    except KeyboardInterrupt:
      # Killed by parent.
      print >> log.v2, "Device proc %s got KeyboardInterrupt" % device
      sys.excepthook(*sys.exc_info())
    except Exception:
      print >> log.v2, "Device proc %s exception:" % device
      sys.excepthook(*sys.exc_info())
      sys.exit(1)

  def process_inner(self, device, config, input_queue, output_queue):
    """
    :type device: str
    :type config: Config.Config
    :type input_queue: _multiprocessing.Connection
    :type output_queue: _multiprocessing.Connection
    """
    # We are the child. The queues are a duplex pipe.
    # We send/recv over the input_queue.
    # We close the output_queue so that the parent gets an EOF when we die.
    output_queue.close()
    output_queue = input_queue
    if device[0:3] == 'gpu':
      import theano.sandbox.cuda
      import cuda_ndarray.cuda_ndarray as cuda
      if device == 'gpuX': device = 'gpu'
      print "Use CUDA in device proc %s" % device
      assert not theano.sandbox.cuda.cuda_enabled, "Must not yet be enabled. Otherwise sth is screwed."
      theano.sandbox.cuda.use(device, force = True)
      #theano.sandbox.cuda.use(device, force = True, default_to_move_computation_to_gpu=True, move_shared_float32_to_gpu=True, enable_cuda=True)
      device_id = cuda.active_device_number()
      device_name = cuda.active_device_name()
      device = "gpu%i" % device_id
    else:
      try:
        device_id = int(device[3:])
      except ValueError:
        device_id = 0
      device_name = 'cpu%i' % device_id
    output_queue.send(device_id)
    output_queue.send(device_name)
    self.initialize(config)
    self._checkGpuFuncs(device, device_id)
    output_queue.send(len(self.trainnet.train_params))
    print >> log.v2, "Device %s proc, pid %i is ready for commands." % (device, os.getpid())
    while True:
      cmd = input_queue.recv()
      if cmd == "stop":  # via self.terminate()
        output_queue.send("done")
        break
      elif cmd == "reinit":  # via self.reinit()
        network_description = input_queue.recv()
        train_param_args = input_queue.recv()
        if self.need_reinit(network_description, train_param_args):
          self.initialize(config, network_description, train_param_args)
        output_queue.send("reinit-ready")
        output_queue.send(len(self.trainnet.train_params))
      elif cmd == "update-data":  # via self.update_data()
        x = input_queue.recv()
        t = input_queue.recv()
        i = input_queue.recv()
        if self.trainnet.loss in ('ctc','ce_ctc'):
          c = input_queue.recv()
          self.cp.set_value(c)
        self.x.set_value(x.astype('float32'), borrow = True)
        self.y.set_value(t.astype('int32'), borrow = True)
        self.i.set_value(i.astype('int8'), borrow = True)
      elif cmd == "set-learning-rate":  # via self.set_learning_rate()
        learning_rate = input_queue.recv()
        assert self.updater, "Only set if in train mode. Task = %s" % self.network_task
        assert self.updater.updateOnDevice
        self.updater.setLearningRate(learning_rate)
      elif cmd == "set-net-params":  # via self.set_net_params()
        params = input_queue.recv()
        assert isinstance(params, list)
        our_params_trainnet = self.trainnet.get_all_params()
        our_params_testnet = self.testnet.get_all_params()
        assert len(params) == len(our_params_trainnet) == len(our_params_testnet)
        for i in range(len(params)):
          our_params_trainnet[i].set_value(params[i])
          our_params_testnet[i].set_value(params[i])
      elif cmd == "get-net-train-params":  # via self.get_net_train_params()
        output_queue.send("net-train-params")
        # We can get cuda_ndarray or other references to internal device memory.
        # We explicitly want to copy them over to CPU memory.
        output_queue.send([numpy.asarray(p.get_value()) for p in self.trainnet.train_params])
      elif cmd == "task":  # via self.run()
        task = input_queue.recv()
        try:
          result = self.compute(task)()
        except RuntimeError:
          print >> log.v2, "warning: Runtime error on device", device_name
          output_queue.send("error")
          return
        except MemoryError:
          output_queue.send("error")
          raise
        output_queue.send("task-result")
        # We can get cuda_ndarray or other references to internal device memory.
        # We explicitly want to copy them over to CPU memory.
        output_queue.send([numpy.asarray(output) for output in result])
      else:
        raise Exception("cmd %s unknown" % cmd)

  def get_task_network(self):
    """
    :rtype: LayerNetwork
    """
    if self.network_task == "train":
      return self.trainnet
    else:
      return self.testnet

  def alloc_data(self, shape, max_ctc_length=0, pad=False):
    """
    :param list[int] shape: format (time,batch,features)
    :type max_ctc_length: int
    """
    assert len(shape) == 3
    assert all([s > 0 for s in shape])
    import theano
    self.data = numpy.zeros(shape, dtype=theano.config.floatX)
    self.targets = numpy.zeros(shape[0:2], dtype=theano.config.floatX)  # is actually the int idx of the target class
    self.ctc_targets = numpy.zeros((shape[1], max_ctc_length), dtype=theano.config.floatX)
    if pad:
      self.index = numpy.ones(shape[0:2], dtype='int8')
    else:
      self.index = numpy.zeros(shape[0:2], dtype='int8')
    self.tags = [None] * shape[1]  # TODO

  def update_data(self):
    # self.data is set in Engine.allocate_devices()
    if self.blocking:
      self.x.set_value(self.data, borrow = True)
      #self.t.set_value(self.targets, borrow = True)
      self.y.set_value(self.targets.flatten().astype('int32'), borrow = True)
      self.i.set_value(self.index, borrow = True)
      if self.trainnet.loss in ('ctc','ce_ctc'):
        self.cp.set_value(self.ctc_targets)
    else:
      assert self.main_pid == os.getpid()
      self.input_queue.send("update-data")
      self.input_queue.send(self.data)
      self.input_queue.send(self.targets.flatten())
      self.input_queue.send(self.index)
      if self.config.value('loss','') == 'ctc':
        self.input_queue.send(self.ctc_targets)

  def set_learning_rate(self, learning_rate):
    """
    :type learning_rate: float
    """
    assert self.updater, "Only set if in train mode. Task = %s" % self.network_task
    assert self.updater.updateOnDevice
    if self.blocking:
      self.updater.setLearningRate(learning_rate)
    else:
      assert self.main_pid == os.getpid()
      self.input_queue.send("set-learning-rate")
      self.input_queue.send(learning_rate)

  def get_net_train_params(self):
    if self.blocking:
      return [v.get_value(borrow=True, return_internal_type=True) for v in self.trainnet.train_params]
    else:
      assert self.main_pid == os.getpid()
      self.input_queue.send("get-net-train-params")
      r = self.output_queue.recv()
      assert r == "net-train-params"
      r = self.output_queue.recv()
      return r

  def set_net_params(self, network):
    """
    :type network: Network.LayerNetwork
    This updates *all* params, not just the train params.
    """
    if self.blocking:
      self.trainnet.set_params_by_dict(network.get_params_dict())
      self.testnet.set_params_by_dict(network.get_params_dict())
    else:
      assert self.main_pid == os.getpid()
      self.input_queue.send("set-net-params")
      self.input_queue.send([numpy.asarray(p.get_value()) for p in network.get_all_params()])

  def maybe_update_network(self, network):
    """
    This is usually called before we start a new batch.
    :type network: LayerNetwork
    """
    if not self.updater or self.updater.updateOnDevice:
      # We keep the model on the device and update it online.
      # Thus, no need to update it externally.
      return
    self.set_net_params(network)

  def need_reinit(self, network_description=None, train_param_args=None):
    assert self.trainnet
    if self.trainnet.description != network_description:
      print >> log.v3, "Device: reinit because network description differs. Diff:", \
                       obj_diff_str(self.trainnet.description, network_description)
      return True
    if train_param_args is None:
      train_param_args = self.trainnet.get_train_param_args_default()
    if self.trainnet.train_param_args != train_param_args:
      print >> log.v3, "Device: reinit because network train params differ"
      return True
    return False

  def reinit(self, network_description=None, train_param_args=None):
    """
    :type network_description: Network.LayerNetworkDescription
    :type train_param_args: dict
    :returns len of train_params
    :rtype: int
    Reinits for a new network topology. This can take a while
    because the gradients have to be recomputed.
    """
    assert self.main_pid == os.getpid(), "Call this from the main proc."
    if self.blocking:
      if self.need_reinit(network_description, train_param_args):
        self.initialize(self.config, network_description, train_param_args)
      return len(self.trainnet.train_params)
    else:
      self.input_queue.send("reinit")
      self.input_queue.send(network_description)
      self.input_queue.send(train_param_args)
      r = self.output_queue.recv()
      assert r == "reinit-ready"
      r = self.output_queue.recv()
      return r

  def prepare(self, network, updater=None, train_param_args=None):
    """
    Call this from the main proc before we do anything else.
    This is called before we start any training, e.g. at the begin of an epoch.
    :type network: LayerNetwork
    :type updater: Updater | None
    :type train_param_args: dict | None
    """
    assert self.main_pid == os.getpid(), "Call this from the main proc."
    if not self.blocking:
      # In blocking, we would have initialized our own updater via self.initialize().
      self.updater = updater
    # Reinit if needed.
    self.reinit(network.description, train_param_args)
    if not self.updater or self.updater.updateOnDevice:
      # If there is no updater, or we do the updates online, we must copy the net params now.
      # Otherwise we will always update the model via self.maybe_update_network().
      self.set_net_params(network)

  def run(self, task):
    """
    :type task: str
    """
    self.task = task
    self.run_called_count += 1
    self.update_data()
    if self.blocking:
      self.output = self.compute(task)()
    else:
      assert self.main_pid == os.getpid()
      self.output = None
      self.input_queue.send("task")
      self.input_queue.send(task)

  def clear_memory(self, network):
    #self.data = numpy.zeros((1, 1, 1), dtype = theano.config.floatX)
    #self.targets = numpy.zeros((1, 1), dtype = theano.config.floatX)
    #self.index = numpy.zeros((1, 1), dtype = theano.config.floatX)
    self.update_data()

  def result(self):
    self.result_called_count += 1
    if self.blocking:
      assert self.result_called_count == self.run_called_count
      return self.output
    else:
      assert self.main_pid == os.getpid()
      assert self.result_called_count <= self.run_called_count
      if not self.proc.is_alive():
        return None
      timeout = 60 * 5  # 5 minutes execution timeout
      while timeout > 0:
        try:
          if self.output_queue.poll(1):
            r = self.output_queue.recv()
            if r == "error": return None
            assert r == "task-result"
            self.output = self.output_queue.recv()
            return self.output
        except EOFError:
          # The process is dying or died.
          return None
        except IOError, e:
          if e.errno == errno.EINTR:
            # http://stackoverflow.com/questions/14136195
            # We can just keep trying.
            print >> log.v3, "Device proc %s gave us an EINTR." % self.name
            time.sleep(1)
          else:
            # The process is dying or died.
            return None
        timeout -= 1
      print >> log.v3, "Timeout expired for device", self.name
      return None

  def terminate(self):
    if not self.blocking and self.proc.is_alive():
      assert self.main_pid == os.getpid()
      self.input_queue.send('stop')
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
    try:
      import pynvml
    except ImportError as exc:
      return None
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
  def make_ce_ctc_givens(self, network):
    return [(network.x, self.x), (network.y, self.y), (network.c, self.c), (network.i, self.i)]
