
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
      Numpy arrays encoded via Numpy fromstring/tostring.
  """

  Version = 1  # increase when some protocol changes

  def __init__(self, sprintExecPath, sprintConfigStr):
    """
    :type sprintExecPath: str
    :type sprintConfigStr: str
    """
    assert os.path.exists(sprintExecPath)
    self.sprintExecPath = sprintExecPath
    if sprintConfigStr.startswith("config:"):
      from Config import get_global_config
      config = get_global_config()
      assert config
      sprintConfigStr = config.typed_dict[sprintConfigStr[len("config:"):]]
    self.sprintConfig = eval_shell_str(sprintConfigStr)
    self.child_pid = None
    self.parent_pid = os.getpid()
    # There is no generic way to see whether Python is exiting.
    # This is our workaround. We check for it in self.run_inner().
    self.python_exit = False
    atexit.register(self.exit_handler)
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

  def _start_child(self):
    assert self.child_pid is None
    self.pipe_c2p = self._pipe_open()
    self.pipe_p2c = self._pipe_open()
    args = self._build_sprint_args()
    print >>log.v5, "SprintSubprocessInstance: exec", args

    pid = os.fork()
    if pid == 0:  # child
      try:
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
    my_mod_name = "SprintControl"
    args = [
      self.sprintExecPath,
      # Sprint PythonControl or PythonTrainer
      "--*.pymod-path=%s" % self._my_python_mod_path,
      "--*.pymod-name=%s" % my_mod_name,
      "--*.pymod-config=%s" % config_str,
      # Sprint PythonSegmentOrder
      "--*.python-segment-order=true",
      "--*.python-segment-order-pymod-path=%s" % self._my_python_mod_path,
      "--*.python-segment-order-pymod-name=%s" % my_mod_name,
      "--*.python-segment-order-config=%s" % config_str
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

  def get_loss_and_error_signal(self, seg_name, seg_len, posteriors):
    """
    :param str seg_name: the segment name (seq_tag)
    :param int seg_len: the segment length in frames
    :param numpy.ndarray posteriors: 2d (time,label) float array
    :rtype (float, numpy.ndarray)
    :returns (loss, error_signal). error_signal has the same shape as posteriors.
    """
    posteriors_str = posteriors.astype('float32').tostring()
    try:
      self._send(("get_loss_and_error_signal", seg_name, seg_len, posteriors_str))
      ret = self._read()
    except (IOError, EOFError):
      raise
    assert ret[0] == "ok" and len(ret) == 3, "Got unexpected return: %r" % ret
    loss = ret[1]
    error_signal_str = ret[2]
    error_signal = numpy.fromstring(error_signal_str, dtype="float32")
    assert error_signal.shape == posteriors.shape
    return loss, error_signal

  def exit_handler(self):
    assert os.getpid() == self.parent_pid
    self.python_exit = True
    self._exit_child(should_interrupt=True)

  def init(self):
    self._exit_child()
    self._start_child()

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
    for b in range(n_batch):
      loss, error_signal = self.get_loss_and_error_signal(
        seg_name=tags[b], seg_len=seq_lengths[b], posteriors=log_posteriors[:, b])
      batch_loss[b] = loss
      batch_error_signal[:, b] = error_signal
    return batch_loss, batch_error_signal


class SprintErrorSigOp(theano.Op):
  """
  Op: posteriors, seq_lengths -> loss, error signal (grad w.r.t. z, i.e. before softmax is applied)
  """

  __props__ = ("target", "sprint_opts")

  def __init__(self, target, sprint_opts):
    super(SprintErrorSigOp, self).__init__()
    self.target = target  # default is "classes"
    self.sprint_opts = make_hashable(sprint_opts)
    self.sprint_instance = None

  def make_node(self, posteriors, seq_lengths):
    log_posteriors = T.log(theano.tensor.as_tensor_variable(posteriors))
    seq_lengths = theano.tensor.as_tensor_variable(seq_lengths)
    assert seq_lengths.ndim == 1  # vector of seqs lengths
    return theano.Apply(self, [log_posteriors, seq_lengths], [T.fvector(), posteriors.type()])

  def perform(self, node, inputs, output_storage):
    log_posteriors, seq_lengths = inputs

    if numpy.isnan(log_posteriors).any():
      print >> log.v1, 'log_posteriors contain NaN!'
    if numpy.isinf(log_posteriors).any():
      print >> log.v1, 'log_posteriors contain Inf!'
      numpy.set_printoptions(threshold=numpy.nan)
      print >> log.v1, 'log_posteriors:', log_posteriors

    if self.sprint_instance is None:
      print >> log.v3, "SprintErrorSigOp: Starting Sprint %r" % self.sprint_opts
      self.sprint_instance = SprintSubprocessInstance(**self.sprint_opts)

    loss, errsig = self.sprint_instance.get_batch_loss_and_error_signal(self.target, log_posteriors, seq_lengths)
    #print >> log.v4, 'loss:', loss, 'errsig:', errsig
    output_storage[0][0] = loss
    output_storage[1][0] = errsig

    print >> log.v5, 'avg frame loss for segments:', loss.sum() / seq_lengths.sum()
