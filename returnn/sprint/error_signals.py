
"""
This provides the Theano Op `SprintErrorSigOp` to get a loss and error signal
which is calculated via Sprint.
And there are helper classes to communicate with the Sprint subprocess
to transfer the posteriors and get back the loss and error signal.
It uses the SprintControl Sprint interface for the communication.
"""

from __future__ import print_function

import numpy
import sys
import os
import atexit
import signal
import typing
from threading import RLock, Thread
import returnn.util.task_system as task_system
from returnn.util.task_system import Pickler, Unpickler, numpy_set_unused
from returnn.util.basic import eval_shell_str, make_hashable, close_all_fds_except
from returnn.log import log


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

  # Keep argument names as is, as these are coming directly from a user config file.
  # noinspection PyPep8Naming
  def __init__(self, sprintExecPath, minPythonControlVersion=2, sprintConfigStr="", sprintControlConfig=None,
               usePythonSegmentOrder=True):
    """
    :param str sprintExecPath: this executable will be called for the sub proc.
    :param int minPythonControlVersion: will be checked in the subprocess. via Sprint PythonControl
    :param str sprintConfigStr: passed to Sprint as command line args.
      can have "config:" prefix - in that case, looked up in config.
      handled via eval_shell_str(), can thus have lazy content (if it is callable, will be called).
    :param dict[str]|None sprintControlConfig: passed to SprintControl.init().
    """
    assert os.path.exists(sprintExecPath)
    self.sprintExecPath = sprintExecPath
    self.minPythonControlVersion = minPythonControlVersion
    if sprintConfigStr.startswith("config:"):
      from returnn.config import get_global_config
      config = get_global_config()
      assert config
      sprintConfigStr = config.typed_dict[sprintConfigStr[len("config:"):]]
    self.sprintConfig = eval_shell_str(sprintConfigStr)
    self.sprintControlConfig = sprintControlConfig
    self.usePythonSegmentOrder = usePythonSegmentOrder
    self.child_pid = None  # type: typing.Optional[int]
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
          print("SprintSubprocessInstance: interrupt child proc %i" % self.child_pid, file=log.v5)
          os.kill(self.child_pid, signal.SIGKILL)
        else:
          # noinspection PyBroadException
          try:
            self._send(("exit",))
          except Exception:
            pass
      else:
        self.child_pid = None
      try:
        self.pipe_p2c[1].close()
      except IOError:
        pass
      try:
        self.pipe_c2p[0].close()
      except IOError:
        pass
      if self.child_pid:
        self._join_child(wait=True, expected_exit_status=0 if not interrupt else None)
        self.child_pid = None

  # noinspection PyMethodMayBeStatic
  def _env_update_child(self):
    theano_flags = {key: value for (key, value)
                    in [s.split("=", 1) for s in os.environ.get("THEANO_FLAGS", "").split(",") if s]}

    # First set some sane default for compile dir.
    theano_flags.setdefault("compiledir_format",
                            "compiledir_%(platform)s-%(processor)s-%(python_version)s-%(python_bitwidth)s")
    compiledir_format = theano_flags["compiledir_format"]
    p = compiledir_format.find("--dev-")  # Device.startProc might have added that.
    if p >= 0:
      compiledir_format = compiledir_format[:p]
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
    print("SprintSubprocessInstance: exec", args, file=log.v5)

    pid = os.fork()
    if pid == 0:  # child
      print("SprintSubprocessInstance: starting, pid %i" % os.getpid(), file=log.v5)
      # noinspection PyBroadException
      try:
        self._env_update_child()
        sys.stdin.close()  # Force no tty stdin.
        self.pipe_c2p[0].close()
        self.pipe_p2c[1].close()
        close_all_fds_except([0, 1, 2, self.pipe_c2p[1].fileno(), self.pipe_p2c[0].fileno()])
        os.execv(args[0], args)  # Does not return if successful.
      except BaseException:
        print("SprintSubprocessInstance: Error when starting Sprint %r." % args, file=log.v1)
        sys.excepthook(*sys.exc_info())
      finally:
        # noinspection PyUnresolvedReferences,PyProtectedMember
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
      print("SprintSubprocessInstance: Sprint child process (%r) caused an exception." % args, file=log.v1)
      sys.excepthook(*sys.exc_info())
      raise Exception("SprintSubprocessInstance Sprint init failed")

  # noinspection PyMethodMayBeStatic
  def _pipe_open(self):
    readend, writeend = os.pipe()
    if hasattr(os, "set_inheritable"):
      # https://www.python.org/dev/peps/pep-0446/
      os.set_inheritable(readend, True)
      os.set_inheritable(writeend, True)
    readend = os.fdopen(readend, "rb", 0)
    writeend = os.fdopen(writeend, "wb", 0)
    return readend, writeend

  @property
  def _my_python_mod_path(self):
    from returnn import __root_dir__
    return __root_dir__

  def _build_sprint_args(self):
    config_str = "c2p_fd:%i,p2c_fd:%i" % (
        self.pipe_c2p[1].fileno(), self.pipe_p2c[0].fileno())
    config_str += ",minPythonControlVersion:%i" % self.minPythonControlVersion
    if task_system.SharedMemNumpyConfig["enabled"]:
      config_str += ",EnableAutoNumpySharedMemPickling:True"
    if self.sprintControlConfig:
      config_str += "," + ",".join(["%s:%s" % (k, v) for (k, v) in sorted(self.sprintControlConfig.items())])
    my_mod_name = "returnn.sprint.control"
    args = [
      self.sprintExecPath,
      # Enable Sprint PythonControl
      "--*.python-control-enabled=true",
      # Sprint PythonControl or PythonTrainer
      "--*.pymod-path=%s" % self._my_python_mod_path,
      "--*.pymod-name=%s" % my_mod_name,
      "--*.pymod-config=%s" % config_str
    ]
    if self.usePythonSegmentOrder:
      args += [
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

  def get_loss_and_error_signal__send(self, seg_name, seg_len, log_posteriors):
    """
    :param str seg_name: the segment name (seq_tag)
    :param int seg_len: the segment length in frames
    :param numpy.ndarray log_posteriors: 2d (time,label) float array, log probs
    """
    assert not self.is_calculating
    assert seg_name
    self._cur_seg_name = seg_name
    assert seg_len == log_posteriors.shape[0]
    self._cur_posteriors_shape = log_posteriors.shape
    try:
      self._send(("get_loss_and_error_signal", seg_name, seg_len, log_posteriors.astype("float32", copy=False)))
    except (IOError, EOFError):
      raise
    else:
      self.is_calculating = True

  def get_loss_and_error_signal__have_data(self):
    """
    :rtype: bool
    """
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
    """
    Called at exit. Exit child.
    """
    assert os.getpid() == self.parent_pid
    self.python_exit = True
    self._exit_child(should_interrupt=True)

  def init(self):
    """
    Init. (Re)Start child.
    """
    self._exit_child()
    self._start_child()


class ReaderThread(Thread):
  """
  Sprint reader thread.
  """
  def __init__(self, instance, instance_idx, batch_idxs, tags, seq_lengths, log_posteriors,
               batch_loss, batch_error_signal):
    """
    :param SprintSubprocessInstance instance:
    :param int instance_idx:
    :param list[int] batch_idxs:
    :param list[str] tags: seq names, length = batch
    :param numpy.ndarray seq_lengths: 1d (batch)
    :param numpy.ndarray log_posteriors: 3d (time,batch,label)
    :param numpy.ndarray batch_loss: 1d (batch). will write result into it.
    :param numpy.ndarray batch_error_signal: 3d (time,batch,label). will write results into it.
    """
    super(ReaderThread, self).__init__(
      name="SprintErrorSignals reader thread for Sprint instance %i" % instance_idx)
    self.daemon = True
    self.instance_idx = instance_idx
    self.instance = instance
    self.batch_idxs = batch_idxs
    self.tags = tags
    self.seq_lengths = seq_lengths
    self.log_posteriors = log_posteriors
    self.batch_loss = batch_loss
    self.batch_error_signal = batch_error_signal
    self.exception = None
    self.start()

  def run(self):
    """
    Main thread func.
    """
    try:
      for b in self.batch_idxs:
        self.instance.get_loss_and_error_signal__send(
          seg_name=self.tags[b], seg_len=self.seq_lengths[b],
          log_posteriors=self.log_posteriors[:self.seq_lengths[b], b])
        seg_name, loss, error_signal = self.instance.get_loss_and_error_signal__read()
        assert seg_name == self.tags[b]
        self.batch_loss[b] = loss
        self.batch_error_signal[:self.seq_lengths[b], b] = error_signal
        numpy_set_unused(error_signal)
    except Exception as exc:
      self.exception = exc


class SprintInstancePool:
  """
  This is a pool of Sprint instances.
  First, for each unique sprint_opts, there is a singleton
    which can be accessed via get_global_instance.
  Then, this can be used in multiple ways.
    (1) get_batch_loss_and_error_signal.
    (2) ...
  """

  class_lock = RLock()
  global_instances = {}  # sprint_opts -> SprintInstancePool instance

  @classmethod
  def get_global_instance(cls, sprint_opts):
    """
    :param dict[str] sprint_opts:
    :rtype: SprintInstancePool
    """
    sprint_opts = make_hashable(sprint_opts)
    with cls.class_lock:
      if sprint_opts in cls.global_instances:
        return cls.global_instances[sprint_opts]
      instance = SprintInstancePool(sprint_opts=sprint_opts)
      cls.global_instances[sprint_opts] = instance
      return instance

  def __init__(self, sprint_opts):
    """
    :param dict[str] sprint_opts:
    """
    # The lock will not be acquired automatically on the public functions here as there is the valid
    # usage that only one thread will access it anyway.
    # So, take care of acquiring this lock yourself whenever you call here potentially from multiple threads.
    # All the code is not thread-safe, so this is important!
    self.lock = RLock()
    assert isinstance(sprint_opts, dict)
    sprint_opts = sprint_opts.copy()
    self.max_num_instances = int(sprint_opts.pop("numInstances", 1))
    self.sprint_opts = sprint_opts
    self.instances = []  # type: typing.List[SprintSubprocessInstance]

  def _maybe_create_new_instance(self):
    """
    :rtype: SprintSubprocessInstance|None
    """
    if len(self.instances) < self.max_num_instances:
      self.instances.append(SprintSubprocessInstance(**self.sprint_opts))
      return self.instances[-1]
    return None

  def _get_instance(self, i):
    """
    :param int i:
    :rtype: SprintSubprocessInstance
    """
    assert i < self.max_num_instances
    if i >= len(self.instances):
      assert i == len(self.instances)
      self._maybe_create_new_instance()
    return self.instances[i]

  def get_batch_loss_and_error_signal(self, log_posteriors, seq_lengths, tags=None):
    """
    :param numpy.ndarray log_posteriors: 3d (time,batch,label)
    :param numpy.ndarray seq_lengths: 1d (batch)
    :param list[str] tags: seq names, length = batch
    :rtype (numpy.ndarray, numpy.ndarray)
    :returns (loss, error_signal). error_signal has the same shape as posteriors.
    loss is a 1d-array (batch).

    Note that this accesses some global references, like global current seg info,
    via the current Device instance.
    Thus this is expected to be run from the Device host proc,
      inside from SprintErrorSigOp.perform.
    This also expects that we don't have chunked seqs.
    """
    assert seq_lengths.ndim == 1
    assert log_posteriors.ndim == 3
    n_batch = seq_lengths.shape[0]
    assert n_batch == log_posteriors.shape[1]

    if tags is None:
      raise NotImplementedError("This feature was removed with dropped Theano support")
    assert len(tags) == n_batch

    batch_loss = numpy.zeros((n_batch,), dtype="float32")
    batch_error_signal = numpy.zeros_like(log_posteriors, dtype="float32")

    # greedy solution to the scheduling problem
    sorted_length = sorted(enumerate(seq_lengths), key=lambda x: x[1], reverse=True)
    jobs = [[] for _ in range(self.max_num_instances)]
    joblen = [0] * self.max_num_instances
    for i, l in sorted_length:
      j = min(enumerate(joblen), key=lambda x: x[1])[0]  # noqa
      jobs[j].append(i)
      joblen[j] += l

    if self.max_num_instances > 1:
      threads = [
        ReaderThread(
          self._get_instance(i), i, jobs[i], tags, seq_lengths, log_posteriors, batch_loss, batch_error_signal)
        for i in range(self.max_num_instances)]
      for i, thread in enumerate(threads):
        thread.join()
        if thread.exception:
          raise thread.exception
    else:
      # Very simple parallelism. We must avoid any form of multi-threading
      # because this can be problematic with Theano.
      # See: https://groups.google.com/forum/#!msg/theano-users/Pu4YKlZKwm4/eNcAegzaNeYJ
      # We also try to keep it simple here.
      for bb in range(0, n_batch, self.max_num_instances):
        for i in range(self.max_num_instances):
          b = bb + i
          if b >= n_batch:
            break
          instance = self._get_instance(i)
          instance.get_loss_and_error_signal__send(
            seg_name=tags[b], seg_len=seq_lengths[b], log_posteriors=log_posteriors[:seq_lengths[b], b])
        for i in range(self.max_num_instances):
          b = bb + i
          if b >= n_batch:
            break
          instance = self._get_instance(i)
          seg_name, loss, error_signal = instance.get_loss_and_error_signal__read()
          assert seg_name == tags[b]
          batch_loss[b] = loss
          batch_error_signal[:seq_lengths[b], b] = error_signal
          numpy_set_unused(error_signal)
    return batch_loss, batch_error_signal

  def get_automata_for_batch(self, tags):
    """
    :param list[str]|numpy.ndarray tags: sequence names, used for Sprint (ndarray of shape (batch, max_str_len))
    :return: (edges, weights, start_end_states). all together in one automaton.
      edges are of shape (4, num_edges), each (from, to, emission-idx, seq-idx), of dtype uint32.
      weights are of shape (num_edges,), of dtype float32.
      start_end_states are of shape (2, batch), each (start,stop) state idx, batch = len(tags), of dtype uint32.
    :rtype: (numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """
    all_num_states = [None] * len(tags)  # type: typing.List[typing.Optional[int]]
    all_num_edges = [None] * len(tags)  # type: typing.List[typing.Optional[int]]
    all_edges = [None] * len(tags)  # type: typing.List[typing.Optional[numpy.ndarray]]
    all_weights = [None] * len(tags)  # type: typing.List[typing.Optional[numpy.ndarray]]
    for bb in range(0, len(tags), self.max_num_instances):
      for i in range(self.max_num_instances):
        b = bb + i
        if b >= len(tags):
          break
        instance = self._get_instance(i)
        if isinstance(tags[0], str):
          segment_name = tags[b]
        elif isinstance(tags[0], bytes):
          segment_name = tags[b].decode()
        else:
          segment_name = tags[b].view('S%d' % tags.shape[1])[0]
        assert isinstance(segment_name, str)
        # noinspection PyProtectedMember
        instance._send(("export_allophone_state_fsa_by_segment_name", segment_name))
      for i in range(self.max_num_instances):
        b = bb + i
        if b >= len(tags):
          break
        instance = self._get_instance(i)
        # noinspection PyProtectedMember
        r = instance._read()
        if r[0] != 'ok':
          raise RuntimeError(r[1])
        num_states, num_edges, edges, weights = r[1:]
        all_num_states[b] = num_states
        all_num_edges[b] = num_edges
        all_edges[b] = edges.reshape((3, num_edges))  # (from, to, emission-idx) for each edge, uint32
        all_weights[b] = weights  # for each edge, float32
    state_offset = 0
    for idx in range(len(all_edges)):
      num_edges = all_num_edges[idx]
      all_edges[idx][0:2, :] += state_offset
      state_offset += all_num_states[idx]
      # add sequence_idx. becomes (from, to, emission-idx, seq-idx) for each edge
      all_edges[idx] = numpy.vstack((all_edges[idx], numpy.ones((1, num_edges), dtype='uint32') * idx))

    start_end_states = numpy.empty((2, len(all_num_states)), dtype='uint32')
    state_offset = 0
    for idx, num_states in enumerate(all_num_states):
      start_end_states[0, idx] = state_offset
      start_end_states[1, idx] = state_offset + num_states - 1
      state_offset += num_states

    return numpy.hstack(all_edges), numpy.hstack(all_weights), start_end_states

  def get_free_instance(self):
    """
    :rtype: SprintSubprocessInstance|None
    """
    for inst in self.instances:
      if not inst.is_calculating:
        return inst
    return self._maybe_create_new_instance()
