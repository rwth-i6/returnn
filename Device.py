from TaskSystem import AsyncTask
from Updater import Updater
from Util import cmd, progress_bar, obj_diff_str, hms
from Log import log
from Network import LayerNetwork
from SprintCommunicator import SprintCommunicator
import numpy
import sys
import os
import errno
import time
import pickle


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
                 "GeForce GTX 750 Ti" : (640, 1110, 2 * 1024 * 1024 * 1024),
                 "GeForce GT 750M" : (384, 967, 2 * 1024 * 1024 * 1024),
                 }
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
    self.outputs_format = None; " :type: list[str] "  # via self.result()
    self.train_outputs_format = None; " :type: list[str] "  # set via self.initialize()
    self.run_called_count = 0
    self.result_called_count = 0
    self.compute_total_time = 0
    self.update_total_time = 0
    self.data = None
    self.main_pid = os.getpid()

    if blocking:
      self.initialize(config)
      self.num_train_params = len(self.trainnet.train_params_vars)
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
    # Note that we want a really new separate process, i.e. fork+exec, not just a fork.
    # This is to avoid many potential bugs, e.g. in Numpy or Theano.
    # See also the comment in TaskSystem.ExecingProcess.
    self.proc = AsyncTask(
      func=self.process,
      name="Device %s proc" % self.name,
      mustExec=True)
    # The connection (duplex pipe) is managed by AsyncTask.
    self.input_queue = self.output_queue = self.proc.conn

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
    :type network_description: NetworkDescription.LayerNetworkDescription | None
    :type train_param_args: dict | None
    """
    target = config.value('target', 'classes')
    if self.blocking:
      assert os.getpid() == self.main_pid
    else:
      assert os.getpid() != self.main_pid # this won't work on Windows
    import theano
    import theano.tensor as T
    import h5py
    self.network_task = config.value('task', 'train')
    mask = "unity"
    if sum(config.float_list('dropout', [0])) > 0.0:
      mask = "dropout"
    if network_description is not None:
      self.trainnet = LayerNetwork.from_description(network_description, mask, True)
      self.testnet = LayerNetwork.from_description(network_description, "unity", False)
    elif config.bool('initialize_from_model', False) and config.has('load'):
      model = h5py.File(config.value('load', ''), "r")
      self.trainnet = LayerNetwork.from_hdf_model_topology(model, mask, config.bool("sparse_input", False), target, True)
      self.testnet = LayerNetwork.from_hdf_model_topology(model, "unity", config.bool("sparse_input", False), target, False)
      model.close()
    else:
      self.trainnet = LayerNetwork.from_config_topology(config, mask, True)
      self.testnet = LayerNetwork.from_config_topology(config, "unity", False)
    if train_param_args is not None:
      self.trainnet.declare_train_params(**train_param_args)
    # initialize batch
    self.x = theano.shared(numpy.zeros((1, 1, 1), dtype = theano.config.floatX), borrow=True)
    self.y = {}
    for k in self.trainnet.y:
      if self.trainnet.y[k].type == T.ivector().type:
        self.y[k] = theano.shared(numpy.zeros((1,), dtype = 'int32'), borrow=True)
      else:
        self.y[k] = theano.shared(numpy.zeros((1,1), dtype = 'int32'), borrow=True)
    self.i = theano.shared(numpy.zeros((1, 1), dtype = 'int8'), borrow=True)
    if self.trainnet.loss in ('ctc','ce_ctc'):
      self.cp = theano.shared(numpy.zeros((1, 1), dtype = theano.config.floatX), borrow=True)
      self.c = T.cast(self.cp, 'int32')
    gparams = []
    self.gradients = { k : {} for k in self.y }
    if config.bool('debug_gradient_norm', False):
      # The gradient norm is useful as a check whether we are going to destroy our model (if this is inf/nan).
      # See self.fast_check_model_is_broken_from_result().
      self.gradient_norm = 0
    else:
      self.gradient_norm = None
    for target in self.y:
      for pi, param in enumerate(self.trainnet.train_params_vars):
        if log.verbose[4]: progress_bar(float(pi) / len(self.trainnet.train_params_vars), "calculating gradients ...")
        try:
          gparam = T.grad(self.trainnet.objective[target], param, known_grads = self.trainnet.known_grads)
        except theano.gradient.DisconnectedInputError:
          gparams.append(T.constant(0))
          continue
        self.gradients[target][param] = gparam
        if False and param.name == 'lambda':
          f = theano.function(inputs = [],
                              outputs = [gparam],
                              givens = self.make_givens(self.trainnet),
                              name = "f via trainnet")
          print >> log.v3, theano.printing.pp(gparam)
          print >> log.v3, "-------------------------------------------"
          print >> log.v3, theano.printing.pp(f.maker.fgraph.outputs[0])
        gparams.append(theano.Out(gparam, borrow = True))
        if self.gradient_norm is not None:
          self.gradient_norm += T.sum(gparam ** 2)
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

      # The function output lists must be consistent with TrainTaskThread.evaluate().
      self.train_outputs_format = ["cost"]
      outputs = [self.trainnet.cost[config.value('target', 'classes')]]
      if self.trainnet.ctc_priors is not None:
        self.train_outputs_format += ["ctc_priors"]
        outputs += [self.trainnet.ctc_priors]
      if self.gradient_norm is not None:
        self.train_outputs_format += ["gradient_norm"]
        outputs += [self.gradient_norm]

      if self.updater.updateOnDevice:
        self.updater.initVars(self.trainnet, self.gradients)
        self.train_and_updater = theano.function(inputs=[],
                                                 outputs=outputs,
                                                 givens=train_givens,
                                                 updates=self.updater.getUpdateList(),
                                                 on_unused_input='warn',
                                                 no_default_updates=False,
                                                 name="train_and_updater")

      else:
        self.train_outputs_format += ["gparams..."]
        outputs += gparams
        self.trainer = theano.function(inputs=[],
                                       outputs=outputs,
                                       givens=train_givens,
                                       no_default_updates=False,
                                       on_unused_input='warn',
                                       name="train_distributed")#,
                                       #mode = theano.compile.MonitorMode(post_func=self.detect_nan))

      self.tester = theano.function(inputs=[],
                                    outputs=[self.testnet.cost[config.value('target', 'classes')], self.testnet.errors[config.value('target', 'classes')]],
                                    givens=test_givens,
                                    on_unused_input='warn',
                                    no_default_updates=True,
                                    name="tester")

    elif self.network_task == 'forward':
      extractions = config.list('extract', ['log-posteriors'])
      source = []
      givens = self.make_input_givens(self.testnet)
      for extract in extractions:
        if extract == "classification":
          source.append(T.argmax(self.testnet.output['output'].p_y_given_x, axis = -1, keepdims = True))
        elif extract == "log-posteriors":
          source.append(T.log(self.testnet.output['output'].p_y_given_x))
        elif extract == "posteriors":
          source.append(self.testnet.output['output'].p_y_given_x)
        elif extract == "ctc-sil":
          feat = self.testnet.output['output'].p_y_given_x
          feat = feat[:,:-1] #remove blank
          feat = feat / feat.sum(axis=1)[:,numpy.newaxis] #renormalize
          feat = T.log(feat)
          source.append(feat)
        elif extract == "ce-errsig":
          feat = T.grad(self.testnet.cost, self.testnet.output['output'].z) #TODO
          source.append(feat)
          givens = self.make_givens(self.testnet)
        elif "log-norm-hidden_" in extract:
          idx = int(extract.split('_')[1])
          source.append(T.log(T.nnet.softmax(T.reshape(self.testnet.hidden[idx].output[target], (self.testnet.hidden[idx].output[target].shape[0] * self.testnet.hidden[idx].output[target].shape[1], self.testnet.hidden[idx].output[target].shape[2])))))
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
          source.append(T.reshape(hidden.output[target], (hidden.output[target].shape[0] * hidden.output[target].shape[1], hidden.output[target].shape[2])))
        elif extract in self.testnet.hidden:
          hidden = self.testnet.hidden[extract]
          source.append(T.reshape(hidden.output, (hidden.output.shape[0] * hidden.output.shape[1], hidden.output.shape[2])))
        else:
          assert False, "invalid extraction: " + extract
      self.extractor = theano.function(inputs = [],
                                       outputs = [T.concatenate(source, axis=1)],
                                       givens = givens,
                                       name = "extractor")

    elif self.network_task == 'classify':
      self.classifier = theano.function(inputs = [],
                                        outputs = [T.argmax(self.testnet.output['output'].p_y_given_x, axis = 1)],
                                        givens = self.make_input_givens(self.testnet),
                                        name = "classifier")

    elif self.network_task == 'analyze':
      self.analyzer = theano.function(inputs = [],
                                      outputs = [self.testnet.output['output'].p_y_given_x],
                                              #+ [self.testnet.jacobian],
                                              #+ [hidden.output for hidden in self.network.hidden]
                                              #+ [hidden.output for hidden in self.network.reverse_hidden],
                                      givens = self.make_input_givens(self.testnet),
                                      name = "analyzer")

  def get_compute_func(self, task):
    if task == "train":
      if self.updater.updateOnDevice:
        task = "train_and_update"
      else:
        task = "train_distributed"

    if task == "train_distributed":
      return self.trainer
    elif task == "train_and_update":
      return self.train_and_updater
    elif task == "eval":
      return self.tester
    elif task == "extract" or task == "forward":
      return self.extractor
    elif task == 'classify':
      return self.classifier
    elif task == "analyze":
      return self.analyzer
    else:
      assert False, "invalid command: " + task

  def compute_run(self, task):
    compute_func = self.get_compute_func(task)
    compute_start_time = time.time()
    output = compute_func()
    compute_end_time = time.time()
    self.compute_total_time += compute_end_time - compute_start_time
    # output is a list the outputs which we specified when creating the Theano function in self.initialize().
    assert len(output) > 0  # In all cases, we have some output.
    outputs_format = None
    if task.startswith("train"):
      outputs_format = self.train_outputs_format

    # In train, first output is the score.
    # If this is inf/nan, our model is probably broken.
    model_broken_info = self.fast_check_model_is_broken_from_result(output, outputs_format)
    if model_broken_info:
      self.handle_model_broken(model_broken_info)
      # Pass on, let the Engine decide what to do (or also just fail).

    return output, outputs_format

  def fast_check_model_is_broken_from_result(self, output, outputs_format):
    if not outputs_format:  # In train, we should always have this.
      return
    output_dict = self.make_result_dict(output, outputs_format)
    # Check only params which are small, i.e. not the whole gparams.
    RelevantAttribs = ["cost", "gradient_norm"]
    values = {attrib: numpy.asarray(output_dict[attrib])
              for attrib in RelevantAttribs
              if attrib in output_dict}
    for attrib, value in values.items():
      if not numpy.isfinite(value).all():
        return ", ".join(["%s = %s" % (k, v) for (k, v) in values.items()])
    return

  def handle_model_broken(self, info):
    print >> log.v1, "Model broken: %s" % info
    try:
      dump_file_name = "model_broken_dump.pickle.log"
      if os.path.exists(dump_file_name):
        i = 1
        while os.path.exists("%s.%i" % (dump_file_name, i)):
          i += 1
        dump_file_name = "%s.%i" % (dump_file_name, i)
      f = open(dump_file_name, "w")
      print >> log.v1, "Dumping model broken info to file %r." % dump_file_name
    except Exception, e:
      print >> log.v3, "Exception while opening model broken dump file. %s" % e
      return
    collected_info = {"info_str": str(info)}
    try:
      collected_info["dev_data"] = numpy.asarray(self.x.get_value())
      collected_info["dev_targets"] = numpy.asarray(self.y.get_value())
      collected_info["dev_index"] = numpy.asarray(self.i.get_value())
    except Exception, e:
      print >> log.v3, "Exception when getting device data. %s" % e
    try:
      train_params = [numpy.asarray(v.get_value()) for v in self.trainnet.train_params_vars]
      collected_info["train_params"] = train_params
    except Exception, e:
      print >> log.v3, "Exception when getting train params. %s" % e
    try:
      pickle.dump(collected_info, f)
      f.close()
    except Exception, e:
      print >> log.v3, "Exception when writing model broken info dump. %s" % e

  def _checkGpuFuncs(self, device, device_id):
    if device[0:3] != 'gpu': return
    # Check if we use the GPU.
    # http://deeplearning.net/software/theano/tutorial/modes.html
    theano_func = self.get_compute_func(self.network_task)
    if not any([x.op.__class__.__name__ in ['GpuGemm', 'GpuGemv', 'GpuDot22', 'GpuElemwise']
                for x in theano_func.maker.fgraph.toposort()]):
      print >> log.v1, device + ":", "It seems as if we don't use the GPU although we requested it."
      import theano.printing
      theano.printing.debugprint(theano_func.maker.fgraph.outputs[0])
    else:
      print >> log.v5, device + ":", "Our Theano trainer functions looks like it will run on the GPU."

    try:
      import theano.sandbox.cuda
      theano_cuda = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray
      devProps = theano_cuda.device_properties(device_id)
      print >> log.v5, device + ":", "CUDA version %i" % devProps["driverVersion"]
    except Exception as exc:
      print >> log.v3, device + ":", "Exception while getting CUDA information. %s" % exc

  def process(self, asyncTask):
    """
    :type asyncTask: AsyncTask
    """
    device = self.name
    config = self.config
    try:
      # We do some minimal initialization, modelled after rnn.init().
      # This is needed because we are a new independent process. See startProc().
      import rnn
      rnn.initBetterExchook()
      rnn.config = config
      rnn.initLog()
      print >> log.v3, "Device %s proc starting up" % device
      rnn.initFaulthandler()
      rnn.initConfigJson()
      rnn.maybeInitSprintCommunicator(device_proc=True)
      self.process_inner(device, config, asyncTask)
      rnn.maybeFinalizeSprintCommunicator(device_proc=True)
    except KeyboardInterrupt:
      # Killed by parent.
      print >> log.v2, "Device %s proc got KeyboardInterrupt" % device
      sys.excepthook(*sys.exc_info())
    except Exception as e:
      print >> log.v2, "Device %s proc exception: %s" % (device, e)
      sys.excepthook(*sys.exc_info())
      sys.exit(1)

  def process_inner(self, device, config, asyncTask):
    """
    :type device: str
    :type config: Config.Config
    :type asyncTask: AsyncTask
    """
    # The connection (duplex pipe) is managed by AsyncTask.
    output_queue = input_queue = asyncTask.conn
    if device[0:3] == 'gpu':
      import theano.sandbox.cuda
      import cuda_ndarray.cuda_ndarray as cuda
      if device == 'gpuX': device = 'gpu'
      #print "Use CUDA in device proc %s" % device
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
    output_queue.send(len(self.trainnet.train_params_vars))
    print >> log.v4, "Device %s proc, pid %i is ready for commands." % (device, os.getpid())
    while True:
      try:
        cmd = input_queue.recv()
      except EOFError:
        print >> log.v2, "Device %s proc, pid %i: Parent seem to have died." % (device, os.getpid())
        break  # Just exit.
      if cmd == "stop":  # via self.terminate()
        output_queue.send("done")
        break
      elif cmd == "generic-exec":
        args = input_queue.recv()
        res = self._generic_exec(*args)
        output_queue.send("generic-exec-result")
        output_queue.send(res)
      elif cmd == "reinit":  # via self.reinit()
        network_description = input_queue.recv()
        train_param_args = input_queue.recv()
        if self.need_reinit(network_description, train_param_args):
          self.initialize(config, network_description, train_param_args)
        output_queue.send("reinit-ready")
        output_queue.send(len(self.trainnet.train_params_vars))
      elif cmd == "update-data":  # via self.update_data()
        x = input_queue.recv()
        t = {}
        for k in self.y:
          t[k] = input_queue.recv()
        i = input_queue.recv()
        self.tags = input_queue.recv()
        update_start_time = time.time()
        if self.trainnet.loss in ('ctc','ce_ctc'):
          c = input_queue.recv()
          self.cp.set_value(c)
        if SprintCommunicator.instance is not None:
          SprintCommunicator.instance.segments = self.tags
        self.x.set_value(x.astype('float32'), borrow = True)
        for k in self.y:
          self.y[k].set_value(t[k].astype('int32'), borrow = True)
        #self.c.set_value(c.astype('int32'), borrow = True)
        self.i.set_value(i.astype('int8'), borrow = True)
        self.update_total_time += time.time() - update_start_time
      elif cmd == "set-learning-rate":  # via self.set_learning_rate()
        learning_rate = input_queue.recv()
        assert self.updater, "Only set if in train mode. Task = %s" % self.network_task
        assert self.updater.updateOnDevice
        self.updater.setLearningRate(learning_rate)
      elif cmd == "set-net-params":  # via self.set_net_params()
        params = input_queue.recv()
        assert isinstance(params, list)
        our_params_trainnet = self.trainnet.get_all_params_vars()
        our_params_testnet = self.testnet.get_all_params_vars()
        assert len(params) == len(our_params_trainnet) == len(our_params_testnet)
        for param, our_p_train, our_p_test in zip(params, our_params_trainnet, our_params_testnet):
          our_param_shape = our_p_train.get_value(borrow=True, return_internal_type=True).shape
          assert our_param_shape == param.shape
          assert numpy.isfinite(param).all()
          our_p_train.set_value(param)
          our_p_test.set_value(param)
      elif cmd == "get-net-train-params":  # via self.get_net_train_params()
        output_queue.send("net-train-params")
        # We can get cuda_ndarray or other references to internal device memory.
        # We explicitly want to copy them over to CPU memory.
        output_queue.send([numpy.asarray(p.get_value()) for p in self.trainnet.train_params_vars])
      elif cmd == "task":  # via self.run()
        task = input_queue.recv()
        try:
          output, outputs_format = self.compute_run(task)
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
        output_queue.send([numpy.asarray(v) for v in output])
        output_queue.send(outputs_format)
      else:
        raise Exception("cmd %s unknown" % cmd)

  def is_device_proc(self):
    if self.blocking:
      return True
    if self.main_pid == os.getpid():
      return False  # We are on the host.
    return True  # We are the child proc.

  def _generic_exec(self, func_name, args, kwargs):
    assert self.is_device_proc()
    func = getattr(self, func_name)
    ret = func(*args, **kwargs)
    return ret

  def _generic_exec_on_dev(self, func_name, *args, **kwargs):
    if self.is_device_proc():
      return self._generic_exec(self, func_name, args, kwargs)
    self.input_queue.send("generic-exec")
    self.input_queue.send((func_name, args, kwargs))
    r = self.output_queue.recv()
    assert r == "generic-exec-result"
    r = self.output_queue.recv()
    return r

  def get_task_network(self):
    """
    :rtype: LayerNetwork
    """
    if self.network_task == "train":
      return self.trainnet
    else:
      return self.testnet

  def alloc_data(self, input_shape, output_shape, targets, max_ctc_length=0, pad=False):
    """
    :param list[int] shape: format (time,batch,features)
    :type max_ctc_length: int
    """
    assert len(input_shape) == 3
    assert all([s > 0 for s in input_shape])
    import theano
    self.data = numpy.zeros(input_shape, dtype=theano.config.floatX)
    self.targets = {k: numpy.zeros(output_shape[k], dtype=theano.config.floatX) for k in targets}
    self.ctc_targets = numpy.zeros((output_shape['classes'][1], max_ctc_length), dtype=theano.config.floatX)
    if pad:
      self.index = numpy.ones(input_shape[0:2], dtype='int8')
    else:
      self.index = numpy.zeros(input_shape[0:2], dtype='int8')
    self.tags = [None] * input_shape[1]  # TODO

  def update_data(self):
    # self.data is set in Engine.allocate_devices()
    if self.blocking:
      update_start_time = time.time()
      self.x.set_value(self.data, borrow = True)
      #self.t.set_value(self.targets, borrow = True)
      for target in self.y:
        self.y[target].set_value(self.targets[target].flatten().astype('int32'), borrow = True)
      self.i.set_value(self.index, borrow = True)
      if SprintCommunicator.instance is not None:
        SprintCommunicator.instance.segments = self.tags
      if self.trainnet.loss in ('ctc','ce_ctc'):
        self.cp.set_value(self.ctc_targets)
      self.update_total_time += time.time() - update_start_time
    else:
      assert self.main_pid == os.getpid()
      self.input_queue.send("update-data")
      self.input_queue.send(self.data)
      for target in self.targetkeys:
        if len(self.targets[target].shape) == 3:
          #numpy.swapaxes(self.targets[target], 1, 2).
          self.input_queue.send(self.targets[target].reshape(self.targets[target].shape[0] * self.targets[target].shape[1], self.targets[target].shape[2]))
        else:
          self.input_queue.send(self.targets[target].flatten())
      self.input_queue.send(self.index)
      self.input_queue.send(self.tags)
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
      return [v.get_value(borrow=True, return_internal_type=True) for v in self.trainnet.train_params_vars]
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
      p = [numpy.asarray(p.get_value()) for p in network.get_all_params_vars()]
      self.input_queue.send(p)

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
    print >> log.v4, "Device %s proc epoch time stats: total %s, %.02f%% computing, %.02f%% updating data" % \
                     (self.name, hms(total_time), compute_frac * 100, update_frac * 100)

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
    :type network_description: NetworkDescription.LayerNetworkDescription
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
      return len(self.trainnet.train_params_vars)
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
    self.targetkeys = network.cost.keys()

  def run(self, task):
    """
    :type task: str
    """
    self.task = task
    self.run_called_count += 1
    self.update_data()
    if self.blocking:
      self.output, self.outputs_format = self.compute_run(task)
    else:
      assert self.main_pid == os.getpid()
      self.output = None
      self.outputs_format = None
      self.input_queue.send("task")
      self.input_queue.send(task)

  def clear_memory(self, network):
    #self.data = numpy.zeros((1, 1, 1), dtype = theano.config.floatX)
    #self.targets = numpy.zeros((1, 1), dtype = theano.config.floatX)
    #self.index = numpy.zeros((1, 1), dtype = theano.config.floatX)
    self.update_data()

  @staticmethod
  def make_result_dict(output, outputs_format):
    """
    :type output: list[numpy.ndarray]
    :type outputs_format: list[str]
    """
    d = {}; " :type: dict[str] "
    for i, attrib in enumerate(outputs_format):
      if attrib.endswith("..."):
        attrib = attrib[:-3]
        assert i < len(output)
        assert i == len(outputs_format) - 1
        d[attrib] = output
        return d
      d[attrib] = output[0]
      output = output[1:]
    assert len(output) == 0
    return d

  def result(self):
    """
    :rtype: (list[numpy.ndarray], list[str] | None)
    :returns the outputs and maybe a format describing the output list
    See self.make_result_dict() how to interpret this list.
    See self.initialize() where the list is defined.
    """
    self.result_called_count += 1
    if self.blocking:
      assert self.result_called_count == self.run_called_count
      return self.output, self.outputs_format
    else:
      assert self.main_pid == os.getpid()
      assert self.result_called_count <= self.run_called_count
      if not self.proc.is_alive():
        return None, None
      timeout = 60 * 5  # 5 minutes execution timeout
      while timeout > 0:
        try:
          if self.output_queue.poll(1):
            r = self.output_queue.recv()
            if r == "error": return None
            assert r == "task-result"
            self.output = self.output_queue.recv()
            self.outputs_format = self.output_queue.recv()
            return self.output, self.outputs_format
        except EOFError:
          # The process is dying or died.
          return None, None
        except IOError, e:
          if e.errno == errno.EINTR:
            # http://stackoverflow.com/questions/14136195
            # We can just keep trying.
            print >> log.v3, "Device proc %s gave us an EINTR." % self.name
            time.sleep(1)
          else:
            # The process is dying or died.
            return None, None
        timeout -= 1
      print >> log.v3, "Timeout expired for device", self.name
      return None, None

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
    return [(network.x, self.x), (network.i, self.i)] + [ (network.y[k], self.y[k]) for k in self.y ]
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
