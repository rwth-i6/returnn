
"""
This provides the Theano Op `SprintErrorSigOp` to get a loss and error signal
which is calculated via Sprint.
And there are helper classes to communicate with the Sprint subprocess
to transfer the posteriors and get back the loss and error signal.
It uses the SprintControl Sprint interface for the communication.
"""

import theano
import theano.tensor as T
import numpy
import sys
import os
import atexit
import signal
from TaskSystem import Pickler, Unpickler
from Util import eval_shell_str, make_hashable
from Log import log


class SprintSubprocessInstance:
  """
  The Sprint instance which is used to calculate the error signal.
  Communication is over a pipe. We pass the fds via cmd-line to the child proc.
  Basic protocol with the subprocess (encoded via pickle):
    P2C: tuple (cmd, *cmd_args). cmd is any str.
    C2P: tuple (status, *res_args). status == "ok" if no error.
  Commands:
    "init", name, version -> "ok", child_name, version
    "exit" -> (exit)
    "get_loss_and_error_signal", seg_name, seg_len, posteriors -> "ok", loss, error_signal
      Numpy arrays encoded via TaskSystem.Pickler (which is optimized for Numpy).
  On the Sprint side, we handle this via the SprintControl Sprint interface.
  """

  Version = 1  # increase when some protocol changes

  def __init__(self, sprintExecPath, sprintConfigStr="", sprintControlConfig=None):
    """
    :param str sprintExecPath: this executable will be called for the sub proc.
    :param str sprintConfigStr: passed to Sprint as command line args.
      can have "config:" prefix - in that case, looked up in config.
      handled via eval_shell_str(), can thus have lazy content (if it is callable, will be called).
    :param dict[str]|None sprintControlConfig: passed to SprintControl.init().
    """
    assert os.path.exists(sprintExecPath)
    self.sprintExecPath = sprintExecPath
    if sprintConfigStr.startswith("config:"):
      from Config import get_global_config
      config = get_global_config()
      assert config
      sprintConfigStr = config.typed_dict[sprintConfigStr[len("config:"):]]
    self.sprintConfig = eval_shell_str(sprintConfigStr)
    self.sprintControlConfig = sprintControlConfig
    self.child_pid = None
    self.parent_pid = os.getpid()
    # There is no generic way to see whether Python is exiting.
    # This is our workaround. We check for it in self.run_inner().
    self.python_exit = False
    atexit.register(self.exit_handler)
    self._cur_seg_name = None
    self._cur_posteriors_shape = None
    self.is_calculating = False
    self.init()

  def _exit_child(self, should_interrupt=False):
    if self.child_pid:
      interrupt = False
      expected_exit_status = 0 if not self.python_exit else None
      if self._join_child(wait=False, expected_exit_status=expected_exit_status) is False:  # Not yet terminated.
        interrupt = should_interrupt
        if interrupt:
          print >> log.v5, "SprintSubprocessInstance: interrupt child proc %i" % self.child_pid
          os.kill(self.child_pid, signal.SIGKILL)
        else:
          try: self._send(("exit",))
          except Exception: pass
      else:
        self.child_pid = None
      try: self.pipe_p2c[1].close()
      except IOError: pass
      try: self.pipe_c2p[0].close()
      except IOError: pass
      if self.child_pid:
        self._join_child(wait=True, expected_exit_status=0 if not interrupt else None)
        self.child_pid = None

  def _env_update_child(self):
    theano_flags = {key: value for (key, value)
                    in [s.split("=", 1) for s in os.environ.get("THEANO_FLAGS", "").split(",") if s]}

    # First set some sane default for compile dir.
    theano_flags.setdefault("compiledir_format",
                            "compiledir_%(platform)s-%(processor)s-%(python_version)s-%(python_bitwidth)s")
    compiledir_format = theano_flags["compiledir_format"]
    p = compiledir_format.find("--dev-")  # Device.startProc might have added that.
    if p >= 0: compiledir_format = compiledir_format[:p]
    compiledir_format += "--sprint-sub"
    theano_flags["compiledir_format"] = compiledir_format
    theano_flags["device"] = "cpu"  # Force CPU.
    theano_flags["force_device"] = True
    os.environ["THEANO_FLAGS"] = ",".join(["%s=%s" % (key, value) for (key, value) in sorted(theano_flags.items())])

  def _start_child(self):
    assert self.child_pid is None
    self.pipe_c2p = self._pipe_open()
    self.pipe_p2c = self._pipe_open()
    args = self._build_sprint_args()
    print >>log.v5, "SprintSubprocessInstance: exec", args

    pid = os.fork()
    if pid == 0:  # child
      print >> log.v5, "SprintSubprocessInstance: starting, pid %i" % os.getpid()
      try:
        self._env_update_child()
        sys.stdin.close()  # Force no tty stdin.
        self.pipe_c2p[0].close()
        self.pipe_p2c[1].close()
        os.execv(args[0], args)  # Does not return if successful.
      except BaseException:
        print >> log.v1, "SprintSubprocessInstance: Error when starting Sprint %r." % args
        sys.excepthook(*sys.exc_info())
      finally:
        os._exit(1)
        return  # Not reached.

    # parent
    self.pipe_c2p[1].close()
    self.pipe_p2c[0].close()
    self.child_pid = pid

    try:
      self._send(("init", "SprintSubprocessInstance", self.Version))
      ret = self._read()
      assert ret[0] == "ok" and len(ret) >= 3 and ret[2] == self.Version
    except Exception:
      print >> log.v1, "SprintSubprocessInstance: Sprint child process (%r) caused an exception." % args
      sys.excepthook(*sys.exc_info())
      raise Exception("SprintSubprocessInstance Sprint init failed")

  def _pipe_open(self):
    readend, writeend = os.pipe()
    readend = os.fdopen(readend, "r", 0)
    writeend = os.fdopen(writeend, "w", 0)
    return readend, writeend

  @property
  def _my_python_mod_path(self):
    return os.path.dirname(os.path.abspath(__file__))

  def _build_sprint_args(self):
    config_str = "c2p_fd:%i,p2c_fd:%i" % (
        self.pipe_c2p[1].fileno(), self.pipe_p2c[0].fileno())
    if self.sprintControlConfig:
      config_str += "," + ",".join(["%s:%s" % (k, v) for (k, v) in sorted(self.sprintControlConfig.items())])
    my_mod_name = "SprintControl"
    args = [
      self.sprintExecPath,
      # Enable Sprint PythonControl
      "--*.python-control-enabled=true",
      # Sprint PythonControl or PythonTrainer
      "--*.pymod-path=%s" % self._my_python_mod_path,
      "--*.pymod-name=%s" % my_mod_name,
      "--*.pymod-config=%s" % config_str,
      # Sprint PythonSegmentOrder
      "--*.python-segment-order=true",
      "--*.python-segment-order-pymod-path=%s" % self._my_python_mod_path,
      "--*.python-segment-order-pymod-name=%s" % my_mod_name,
      "--*.python-segment-order-config=%s" % config_str,
      "--*.python-segment-order-allow-copy=false"
    ]
    args += self.sprintConfig
    return args

  def _send(self, v):
    assert os.getpid() == self.parent_pid
    p = self.pipe_p2c[1]  # see _start_child
    Pickler(p).dump(v)
    p.flush()

  def _read(self):
    assert os.getpid() == self.parent_pid
    p = self.pipe_c2p[0]  # see _start_child
    return Unpickler(p).load()

  def _poll(self):
    assert os.getpid() == self.parent_pid
    p = self.pipe_c2p[0]  # see _start_child
    from select import select
    ready, _, _ = select([p.fileno()], [], [], 0)
    return bool(ready)

  def _join_child(self, wait=True, expected_exit_status=None):
    assert self.child_pid
    options = 0 if wait else os.WNOHANG
    pid, exit_status = os.waitpid(self.child_pid, options)
    if not wait and pid == 0:
      return False
    assert pid == self.child_pid
    if expected_exit_status is not None:
      assert exit_status == expected_exit_status, "Sprint exit code is %i" % exit_status
    return True

  def get_loss_and_error_signal__send(self, seg_name, seg_len, posteriors):
    """
    :param str seg_name: the segment name (seq_tag)
    :param int seg_len: the segment length in frames
    :param numpy.ndarray posteriors: 2d (time,label) float array, log probs
    """
    assert not self.is_calculating
    assert seg_name
    self._cur_seg_name = seg_name
    assert seg_len == posteriors.shape[0]
    self._cur_posteriors_shape = posteriors.shape
    try:
      self._send(("get_loss_and_error_signal", seg_name, seg_len, posteriors.astype("float32", copy=False)))
    except (IOError, EOFError):
      raise
    else:
      self.is_calculating = True

  def get_loss_and_error_signal__have_data(self):
    assert self.is_calculating
    return self._poll()

  def get_loss_and_error_signal__read(self):
    """
    :rtype (str, float, numpy.ndarray)
    :returns (seg_name, loss, error_signal). error_signal has the same shape as posteriors.
    """
    assert self.is_calculating
    try:
      ret = self._read()
    except (IOError, EOFError):
      raise
    else:
      self.is_calculating = False
    assert ret[0] == "ok" and len(ret) == 3, "Got unexpected return: %r" % (ret,)
    loss = ret[1]
    error_signal = ret[2]
    assert error_signal.shape == self._cur_posteriors_shape
    return self._cur_seg_name, loss, error_signal

  def exit_handler(self):
    assert os.getpid() == self.parent_pid
    self.python_exit = True
    self._exit_child(should_interrupt=True)

  def init(self):
    self._exit_child()
    self._start_child()


class SprintInstancePool:
  global_instances = {}  # sprint_opts -> SprintInstancePool instance

  @classmethod
  def get_global_instance(cls, sprint_opts):
    sprint_opts = make_hashable(sprint_opts)
    if sprint_opts in cls.global_instances:
      return cls.global_instances[sprint_opts]
    instance = SprintInstancePool(sprint_opts=sprint_opts)
    cls.global_instances[sprint_opts] = instance
    return instance

  def __init__(self, sprint_opts):
    assert isinstance(sprint_opts, dict)
    sprint_opts = sprint_opts.copy()
    self.max_num_instances = int(sprint_opts.pop("numInstances", 1))
    self.sprint_opts = sprint_opts
    self.instances = []; ":type: list[SprintSubprocessInstance]"

  def _maybe_create_new_instance(self):
    if len(self.instances) < self.max_num_instances:
      self.instances.append(SprintSubprocessInstance(**self.sprint_opts))
      return self.instances[-1]
    return None

  def _get_instance(self, i):
    assert i < self.max_num_instances
    if i >= len(self.instances):
      assert i == len(self.instances)
      self._maybe_create_new_instance()
    return self.instances[i]

  def get_batch_loss_and_error_signal(self, target, log_posteriors, seq_lengths):
    """
    :param str target: e.g. "classes". not yet passed over to Sprint.
    :param numpy.ndarray log_posteriors: 3d (time,batch,label)
    :param numpy.ndarray seq_lengths: 1d (batch)
    :rtype (numpy.ndarray, numpy.ndarray)
    :returns (loss, error_signal). error_signal has the same shape as posteriors.
    loss is a 1d-array (batch).

    Note that this accesses some global references, like global current seg info.
    """
    assert seq_lengths.ndim == 1
    assert log_posteriors.ndim == 3
    n_batch = seq_lengths.shape[0]
    assert n_batch == log_posteriors.shape[1]

    import Device
    index = Device.get_current_seq_index(target)  # (time,batch)
    assert index.ndim == 2
    assert index.shape[1] == n_batch
    assert (numpy.sum(index, axis=0) == seq_lengths).all()
    tags = Device.get_current_seq_tags()
    assert len(tags) == n_batch

    batch_loss = numpy.zeros((n_batch,), dtype="float32")
    batch_error_signal = numpy.zeros_like(log_posteriors, dtype="float32")
    # Very simple parallelism. We must avoid any form of multi-threading
    # because this can be problematic with Theano.
    # See: https://groups.google.com/forum/#!msg/theano-users/Pu4YKlZKwm4/eNcAegzaNeYJ
    # We also try to keep it simple here.
    for bb in range(0, n_batch, self.max_num_instances):
      for i in range(self.max_num_instances):
        b = bb + i
        if b >= n_batch: break
        instance = self._get_instance(i)
        instance.get_loss_and_error_signal__send(
          seg_name=tags[b], seg_len=seq_lengths[b], posteriors=log_posteriors[:seq_lengths[b], b])
      for i in range(self.max_num_instances):
        b = bb + i
        if b >= n_batch: break
        instance = self._get_instance(i)
        seg_name, loss, error_signal = instance.get_loss_and_error_signal__read()
        assert seg_name == tags[b]
        batch_loss[b] = loss
        batch_error_signal[:seq_lengths[b], b] = error_signal
    return batch_loss, batch_error_signal

  def get_free_instance(self):
    for inst in self.instances:
      if not inst.is_calculating:
        return inst
    return self._maybe_create_new_instance()


class SprintErrorSigOp(theano.Op):
  """
  Op: log_posteriors, seq_lengths -> loss, error_signal (grad w.r.t. z, i.e. before softmax is applied)
  """

  __props__ = ("target", "sprint_opts")

  def __init__(self, target, sprint_opts):
    super(SprintErrorSigOp, self).__init__()
    self.target = target  # default is "classes"
    self.sprint_opts = make_hashable(sprint_opts)
    self.sprint_instance_pool = None

  def make_node(self, log_posteriors, seq_lengths):
    log_posteriors = theano.tensor.as_tensor_variable(log_posteriors)
    seq_lengths = theano.tensor.as_tensor_variable(seq_lengths)
    assert seq_lengths.ndim == 1  # vector of seqs lengths
    return theano.Apply(self, [log_posteriors, seq_lengths], [T.fvector(), log_posteriors.type()])

  def perform(self, node, inputs, output_storage):
    log_posteriors, seq_lengths = inputs

    if numpy.isnan(log_posteriors).any():
      print >> log.v1, 'SprintErrorSigOp: log_posteriors contain NaN!'
    if numpy.isinf(log_posteriors).any():
      print >> log.v1, 'SprintErrorSigOp: log_posteriors contain Inf!'
      numpy.set_printoptions(threshold=numpy.nan)
      print >> log.v1, 'SprintErrorSigOp: log_posteriors:', log_posteriors

    if self.sprint_instance_pool is None:
      print >> log.v3, "SprintErrorSigOp: Starting Sprint %r" % self.sprint_opts
      self.sprint_instance_pool = SprintInstancePool.get_global_instance(sprint_opts=self.sprint_opts)

    loss, errsig = self.sprint_instance_pool.get_batch_loss_and_error_signal(self.target, log_posteriors, seq_lengths)
    #print >> log.v4, 'loss:', loss, 'errsig:', errsig
    output_storage[0][0] = loss
    output_storage[1][0] = errsig

    print >> log.v5, 'SprintErrorSigOp: avg frame loss for segments:', loss.sum() / seq_lengths.sum()
