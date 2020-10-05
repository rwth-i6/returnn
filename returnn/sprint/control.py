
"""
This is the Sprint interface implementation,
concretely for Sprint PythonControl, Sprint PythonSegmentOrder and Sprint SprintNnPythonLayer.
For reference, in Sprint code, see:
 * src/Nn/PythonControl.cc
 * src/Tools/NnTrainer/python_control_demo.py
This interface will behave similar to SprintExternInterface.
See SprintErrorSignals for the other end.
It can also be used as a PythonSegmentOrdering interface.
It also supports SprintNnPythonLayer.
"""

from __future__ import print_function

import os
import numpy
import typing
from threading import Condition

import returnn.__main__ as rnn
import returnn.util.task_system as task_system
import returnn.util.debug as debug
from returnn.util.task_system import Pickler, Unpickler, numpy_set_unused
from returnn.util.basic import to_bool, long

InitTypes = set()
Verbose = False  # disables all per-segment log messages
Quiet = False  # disables all but error messages

_orig_print = print


# noinspection PyShadowingBuiltins
def print(*args, **kwargs):
  """
  ``print`` replacement.

  :param args:
  :param kwargs:
  """
  if not Quiet:
    _orig_print(*args, **kwargs)


print("RETURNN SprintControl[pid %i] Python module load" % os.getpid())

rnn.init_better_exchook()
debug.init_faulthandler(sigusr1_chain=True)  # Sprint also handles SIGUSR1.
rnn.init_thread_join_hack()


# Start Sprint PythonControl interface. {

def init(name, reference, config, sprint_unit=None, version_number=None, callback=None, **kwargs):
  """
  This will be called by Sprint PythonControl.
  But we also call it ourselves e.g. in getSegmentList() and SprintNnPythonLayer.
  In this specific module, we expect that there is "c2p_fd" and "p2c_fd" in the config string
  to communicate with the parent process, which is usually handled by SprintErrorSignals.

  :param str name: this specifies the caller. e.g. "Sprint.PythonControl"
  :param reference: this is any object to identify the specific instance of the caller, if there are multiple.
  :param str config: this will be passed over from Sprint. you can configure that via --*.pymod-config.
  :param str sprint_unit: if this is called by Sprint PythonControl, this will specify which specific part
    of Sprint is using this PythonControl, because there can be multiple parts.
    E.g. there is "FeedForwardTrainer", "SegmentwiseNnTrainer" and "NnTrainer.pythonControl".
  :param int|None version_number: if this is called by Sprint PythonControl, this will set the version number.
    only newer Sprint versions will set this.
  :param function|None callback: if this is called by Sprint PythonControl, this might provide a callback.
    Only newer Sprint versions will provide this to init().
    This callback can be used for many different actions.
    It's supposed to be called like callback(action, **other_args), where action is a string.
    See Sprint PythonControl code about the possible actions and arguments.
  :param kwargs: all remaining args are specific for each caller.
  """

  config = config.split(",")
  config = {key: value for (key, value) in [s.split(":", 1) for s in config if s]}

  global Quiet
  if to_bool(config.get("quiet", False)):
    Quiet = True

  print(("RETURNN SprintControl[pid %i] init: "
         "name=%r, sprint_unit=%r, version_number=%r, callback=%r, ref=%r, config=%r, kwargs=%r") % (
    os.getpid(), name, sprint_unit, version_number, callback, reference, config, kwargs))
  InitTypes.add(name)

  global Verbose
  if to_bool(config.get("verbose", False)):
    Verbose = True

  if to_bool(config.get("EnableAutoNumpySharedMemPickling", False)) and not task_system.SharedMemNumpyConfig["enabled"]:
    task_system.SharedMemNumpyConfig["enabled"] = True
    print("RETURNN SprintControl[pid %i] EnableAutoNumpySharedMemPickling = True" % (os.getpid(),))

  # Remaining Sprint interface is in this PythonControl instance.
  return PythonControl.create(c2p_fd=int(config["c2p_fd"]), p2c_fd=int(config["p2c_fd"]),
                              name=name, reference=reference, config=config,
                              sprint_unit=sprint_unit,
                              version_number=version_number,
                              min_version_number=int(config["minPythonControlVersion"]),
                              callback=callback,
                              **kwargs)

# End Sprint PythonControl interface. }


# Start Sprint PythonSegmentOrder interface. {

# Keep names for compatibility.
# noinspection PyPep8Naming,PyUnusedLocal
def getSegmentList(corpusName, segmentList, config, **kwargs):
  """
  Sprint will directly call this function.
  """
  print("RETURNN SprintControl[pid %i] getSegmentList: corpus=%r, config=%r" % (os.getpid(), corpusName, config))

  # If we were not initialized via PythonControl interface, this will initialize us
  # and setup the communication channel (PythonControl).
  init(name="RETURNN.PythonSegmentOrder", reference=corpusName, config=config)
  PythonControl.instance.check_control_loop_running()
  for segment_name in PythonControl.instance.segment_list_iterator():
    if isinstance(segment_name, bytes):
      yield segment_name.decode('utf-8')
    else:
      yield segment_name

# End Sprint PythonSegmentOrder interface. }


# Start SprintNnPythonLayer. {

class SprintNnPythonLayer:
  """
  Sprint will directly call this class, i.e. create an instance of it.
  It implements the Sprint NN PythonLayer interface.
  """

  def __init__(self, config, **kwargs):
    print("RETURNN SprintControl[pid %i] SprintNnPythonLayer.__init__: %r, %r" % (os.getpid(), config, kwargs))
    # If we were not initialized via PythonControl interface, this will initialize us
    # and setup the communication channel (PythonControl).
    init(name="RETURNN.SprintNnPythonLayer", reference=self, config=config)
    self.input_size = None
    self.output_size = None

  # noinspection PyMethodMayBeStatic
  def finalize(self):
    """
    Called by Sprint at exit.
    """
    print("RETURNN SprintControl[pid %i] SprintNnPythonLayer.finalize" % (os.getpid(),))

  # noinspection PyPep8Naming
  def setInputDimension(self, stream, size):
    """
    :param int stream:
    :param int size:
    """
    print("RETURNN SprintControl[pid %i] SprintNnPythonLayer.setInputDimension: stream=%r, size=%r" % (
      os.getpid(), stream, size))
    assert stream == 0, "we only support a single input stream (for now)"
    self.input_size = size

  # noinspection PyPep8Naming
  def setOutputDimension(self, size):
    """
    :param int size:
    """
    print("RETURNN SprintControl[pid %i] SprintNnPythonLayer.setOutputDimension: %r" % (os.getpid(), size))
    self.output_size = size

  # noinspection PyPep8Naming,PyMethodMayBeStatic
  def initializeNetworkParameters(self):
    """
    Called by Sprint for param init.
    """
    print("RETURNN SprintControl[pid %i] SprintNnPythonLayer.initializeNetworkParameters" % (os.getpid(),))
    # Just ignore.

  # noinspection PyPep8Naming,PyMethodMayBeStatic
  def loadNetworkParameters(self, filename):
    """
    :param str filename:
    """
    print("RETURNN SprintControl[pid %i] SprintNnPythonLayer.loadNetworkParameters: %r" % (os.getpid(), filename))
    # Just ignore.

  # noinspection PyPep8Naming,PyMethodMayBeStatic
  def saveNetworkParameters(self, filename):
    """
    :param str filename:
    """
    print("RETURNN SprintControl[pid %i] SprintNnPythonLayer.saveNetworkParameters: %r" % (os.getpid(), filename))
    # Just ignore.

  # noinspection PyPep8Naming,PyMethodMayBeStatic
  def isTrainable(self):
    """
    :rtype: bool
    """
    # Always trainable.
    return True

  # noinspection PyPep8Naming,PyMethodMayBeStatic
  def getNumberOfFreeParameters(self):
    """
    :rtype: int
    """
    # For now, just ignore. Not important.
    return 0

  # noinspection PyShadowingBuiltins
  def forward(self, input):
    """
    :param input: tuple of input matrices of format (input_size,time). we ignore them.
    :return: single output matrix of format (output_size,time)
    """
    if Verbose:
      print("RETURNN SprintControl[pid %i] SprintNnPythonLayer.forward: %s" % (
        os.getpid(), input[0].shape if input else repr(input)[:10]))
    assert len(input) == 1
    assert input[0].ndim == 2
    assert input[0].shape[0] == self.input_size
    seg_len = input[0].shape[1]
    posteriors = PythonControl.instance.get_current_seg_posteriors(seg_len=seg_len)  # (time,label)
    if PythonControl.instance.posteriors_in_log_space:
      assert PythonControl.instance.sprint_knows_about_log_space_probs
    assert posteriors.shape == (seg_len, self.output_size)
    return posteriors.T

  # noinspection PyPep8Naming
  def backpropagate(self, errorSignalIn):
    """
    :param numpy.ndarray errorSignalIn: matrix of format (output_size,time)
    :return: tuple of matrices of format (input_size,time)
    :rtype: numpy.ndarray
    """
    if Verbose:
      print("RETURNN SprintControl[pid %i] SprintNnPythonLayer.backpropagate: %r" % (os.getpid(), errorSignalIn.shape))
    assert errorSignalIn.ndim == 2
    assert errorSignalIn.shape[0] == self.output_size
    seg_len = errorSignalIn.shape[1]
    PythonControl.instance.set_current_seg_error_signal(seg_len=seg_len, error_signal=errorSignalIn.T)
    # must return a 1-tuple
    return numpy.zeros((self.input_size, seg_len), dtype="float32"),

# End SprintNnPythonLayer. }


class PythonControl:
  """
  This will send data to RETURNN over a pipe.
  We expect that we are child process and the parent process has spawned us,

  An instance of this class is also the interface for multiple Sprint interfaces, i.e.:
    * PythonControl (standalone via NnTrainer tool)
    * PythonControl (via SegmentwiseNnTrainer)
    * implicitly PythonSegmentOrder (see code above)
  """

  Version = 1  # increase when some protocol changes
  instance = None  # type: typing.Optional[PythonControl]

  @classmethod
  def create(cls, **kwargs):
    """
    :param kwargs: passed to :class:`PythonControl`
    :rtype: PythonControl
    """
    if cls.instance:
      cls.instance._additional_init(**kwargs)
      return cls.instance
    print("RETURNN SprintControl[pid %i] PythonControl create %r" % (os.getpid(), kwargs))
    return PythonControl(**kwargs)

  def __init__(self, c2p_fd, p2c_fd, **kwargs):
    """
    :param int c2p_fd: child-to-parent file descriptor
    :param int p2c_fd: parent-to-child file descriptor
    """
    print("RETURNN SprintControl[pid %i] PythonControl init %r" % (os.getpid(), kwargs))
    assert not self.__class__.instance, "only one instance expected"
    self.__class__.instance = self
    self.cond = Condition()
    self.pipe_c2p = os.fdopen(c2p_fd, "wb")
    self.pipe_p2c = os.fdopen(p2c_fd, "rb")
    self.sprint_callback = None  # via self._init
    self.sprint_version_number = None  # via self._init
    self.callback = None  # either via Sprint, or self.own_threaded_callback
    self.loss_and_error_signal_via_sprint_callback = False
    # So, we get posteriors here from SprintErrorSignals. This will give us always log-probs for now.
    self.posteriors_in_log_space = True  # right now, not configurable
    # We directly forward these posteriors like they are to Sprint. So we assume
    # that Sprint knows about this. For CTC, this is e.g. the option --*.input-in-log-space=true.
    # The Sequence training setup would have a special BiasLayer after the PythonLayer
    # and Sprint assumes log-probs in that case.
    # For other cases, Sprint might not know. We can easily handle that here.
    self.sprint_knows_about_log_space_probs = True  # right now, not configurable
    self.control_loop_started = False
    self.control_loop_exited = False
    self.control_thread__have_new_seg = False
    self.control_thread__have_new_error_signal = False
    self.seg_name = None
    self.seg_len = None
    self.posteriors = None
    self.asked_for_posteriors = False
    self.notified_for_segment = False
    self.error_signal = None
    self.loss = None
    self._init(**kwargs)

  def _additional_init(self, **kwargs):
    print("RETURNN SprintControl[pid %i] PythonControl additional_init %r" % (os.getpid(), kwargs))
    self._init(**kwargs)

  # noinspection PyUnusedLocal
  def _init(self, name, sprint_unit=None, callback=None, version_number=None, min_version_number=None, **kwargs):
    if name == "Sprint.PythonControl":
      print("RETURNN SprintControl[pid %i] init for Sprint.PythonControl %r" % (os.getpid(), kwargs))
      assert min_version_number
      assert (version_number or 0) >= min_version_number, "need new Sprint"
      self.sprint_version_number = version_number
      if callback:
        self.sprint_callback = callback

  # noinspection PyUnusedLocal
  def init_processing(self, input_dim=None, output_dim=None, **kwargs):
    """
    This is called via Sprint when we use PythonControl to iterate the corpus,
    i.e. we set --*.action=python-control in Sprint in the NN trainer tool.
    We expect that we use the Sprint callback to calculate loss and error signal.
    This is called on the first segment.
    input_dim/output_dim are set iff we extract features/alignments.

    :param int|None input_dim:
    :param int|None output_dim:
    """
    print("RETURNN SprintControl[pid %i] init_processing input_dim=%r, output_dim=%r" % (
      os.getpid(), input_dim, output_dim))
    print("RETURNN SprintControl[pid %i] loss_and_error_signal_via_sprint_callback enabled" % (os.getpid(),))
    self.loss_and_error_signal_via_sprint_callback = True
    assert self.sprint_callback

  # noinspection PyUnusedLocal
  def process_segment(self, name, orthography, features=None, alignment=None, soft_alignment=None, **kwargs):
    """
    This is called via Sprint when we use PythonControl to iterate the corpus.

    :param str name: segment name
    :param str orthography: segment orth
    :param numpy.ndarray|None features:
    :param numpy.ndarray|None alignment:
    :param numpy.ndarray|None soft_alignment:
    """
    if Verbose:
      print("RETURNN SprintControl[pid %i] process_segment name=%r orth=%r" % (
        os.getpid(), name, orthography[:10] + "..."))
    assert self.loss_and_error_signal_via_sprint_callback
    assert self.seg_name == name  # via self.handle_cmd_get_loss_and_error_signal()
    assert self.posteriors.ndim == 2  # (time,dim)
    assert features is None, "in Sprint, set --*.extract-features=false"
    assert alignment is None, "in Sprint, set --*.extract-alignment=false"
    assert soft_alignment is None, "in Sprint, set --*.extract-alignment=false"
    loss, error_signal = (
      self._get_loss_and_error_signal_via_sprint_callback(
        seg_name=name, orthography=orthography, posteriors=self.posteriors))
    assert loss is not None
    assert error_signal is not None
    with self.cond:
      self.loss = loss
      self.error_signal = error_signal
      self.cond.notifyAll()

  def _get_loss_and_error_signal_via_sprint_callback(self, seg_name, orthography, posteriors):
    """
    :param str seg_name:
    :param str orthography:
    :param numpy.ndarray posteriors:
    :return: (loss, error_signal)
    :rtype: (float, numpy.ndarray)
    """
    if self.posteriors_in_log_space:
      if self.sprint_knows_about_log_space_probs:  # --*.input-in-log-space=true for CTC in Sprint
        # We assume that Sprint will return gradient for activation before softmax, e.g. y - \hat{y} for CE/CTC.
        output_error_type = "error-signal"
      else:
        posteriors = numpy.exp(posteriors)
        output_error_type = "error-signal-before-softmax"
    else:
      output_error_type = "error-signal-before-softmax"
    loss, error_signal = self.sprint_callback(
      "calculate_criterion",
      posteriors=posteriors.T, orthography=orthography,  # Sprint wants shape (dim,time).
      output_error_type=output_error_type,
      segment_name=seg_name)
    if loss is None:
      return self._default_skipped_loss(), self._default_skipped_error_signal(posteriors)
    error_signal = error_signal.T  # Sprint returns (dim,time) shape.
    assert error_signal.shape == posteriors.shape
    return loss, error_signal

  def _send(self, data):
    Pickler(self.pipe_c2p).dump(data)
    self.pipe_c2p.flush()

  def _read(self):
    return Unpickler(self.pipe_p2c).load()

  def close(self):
    """
    Close pipe.
    """
    self.pipe_c2p.close()
    self.pipe_p2c.close()

  def _handle_cmd_exit(self):
    self.close()
    raise SystemExit

  # noinspection PyUnusedLocal
  def _handle_cmd_init(self, name, version):
    assert version == self.Version
    return "SprintControl", self.Version

  def _handle_cmd_get_loss_and_error_signal(self, seg_name, seg_len, posteriors):
    """
    :param str seg_name: seg name
    :param int seg_len: the segment length in frames
    :param numpy.ndarray posteriors: 2d (time,label) float array

    See SprintErrorSignals.SprintSubprocessInstance.get_loss_and_error_signal().
    """
    assert isinstance(seg_len, (int, long, numpy.int32))
    assert seg_len > 0
    assert posteriors.ndim == 2
    assert posteriors.shape[0] == seg_len
    if Verbose:
      print("RETURNN SprintControl[pid %i] PythonControl handle_cmd_get_loss_and_error_signal: name=%r, len=%r" % (
        os.getpid(), seg_name, seg_len))
    with self.cond:
      self.control_thread__have_new_seg = True
      self.control_thread__have_new_error_signal = False
      if isinstance(seg_name, bytes):
        self.seg_name = seg_name.decode('utf-8')
      else:
        self.seg_name = seg_name
      self.seg_len = seg_len
      self.posteriors = posteriors
      self.error_signal = None
      self.loss = None
      self.asked_for_posteriors = False
      self.notified_for_segment = False
      self.cond.notifyAll()
    loss, error_signal = self.callback("get_loss_and_error_signal", seg_name, seg_len, posteriors)
    assert error_signal.shape == posteriors.shape
    with self.cond:
      self.control_thread__have_new_error_signal = True
      self.posteriors = None
      self.cond.notifyAll()
    numpy_set_unused(posteriors)
    error_signal = error_signal.astype('float32', copy=False)
    return loss, error_signal

  def _handle_cmd_export_allophone_state_fsa_by_segment_name(self, segment_name):
    return self.callback("export_allophone_state_fsa_by_segment_name", segment_name)

  def _handle_cmd(self, cmd, *args):
    """
    :param str cmd:
    :param args:
    :return: some tuple, whatever the func returns
    :rtype: tuple
    """
    func = getattr(self, "_handle_cmd_%s" % cmd)
    return func(*args)

  def handle_next(self):
    """
    Called by self.run_control_loop.
    We catch some message from our parent process, handle it and send back the result.
    """
    import sys
    args = self._read()
    try:
      if not isinstance(args, tuple):
        raise TypeError("expected tuple but got %r" % args)
      if len(args) < 1:
        raise Exception("need multiple args (cmd, ...)")
      res = self._handle_cmd(*args)
    except Exception as e:
      print("RETURNN SprintControl[pid %i] PythonControl handle_next exception" % (os.getpid(),))
      sys.excepthook(*sys.exc_info())
      self._send(("exception", str(e)))
    else:
      assert isinstance(res, tuple)
      self._send(("ok",) + res)

  def run_control_loop(self, callback, **kwargs):
    """
    Called by Sprint when we are in PythonControl run_control_loop mode.
    Also called by us via self.run_threaded_control_loop().
    """
    print("RETURNN SprintControl[pid %i] PythonControl run_control_loop: %r, %r" % (os.getpid(), callback, kwargs))
    print(
      "RETURNN SprintControl[pid %i] PythonControl run_control_loop control: %r" % (os.getpid(), callback("version")))
    self.callback = callback
    with self.cond:
      assert not self.control_loop_started
      self.control_loop_started = True
      self.cond.notifyAll()
    try:
      while True:
        self.handle_next()
    finally:
      with self.cond:
        self.control_loop_exited = True
        self.cond.notifyAll()

  # noinspection PyMethodMayBeStatic
  def exit(self, **kwargs):
    """
    Called by Sprint.
    """
    print("RETURNN SprintControl[pid %i] PythonControl exit: %r" % (os.getpid(), kwargs))

  def check_control_loop_running(self):
    """
    Called by Sprint.
    """
    if self.control_loop_started:
      print("RETURNN SprintControl[pid %i] PythonControl check_control_loop_running: already running" % (os.getpid(),))
      return
    self.run_threaded_control_loop()

  def run_threaded_control_loop(self):
    """
    Called by Sprint.
    """
    print("RETURNN SprintControl[pid %i] PythonControl run_threaded_control_loop" % (os.getpid(),))
    from threading import Thread

    def control_loop():
      """
      Control loop.
      """
      rnn.init_better_exchook()
      self.run_control_loop(self.own_threaded_callback)

    t = Thread(target=control_loop, name="SprintControl.PythonControl.threaded_control_loop")
    t.daemon = True
    t.start()
    while True:
      with self.cond:
        if self.control_loop_started:
          return
        assert t.is_alive()
        self.cond.wait(timeout=1)

  def own_threaded_callback(self, cmd, *args):
    """
    This is used if we run our own control loop via run_threaded_control_loop.
    """
    func = getattr(self, "own_tcb_%s" % cmd)
    return func(*args)

  # noinspection PyMethodMayBeStatic
  def own_tcb_version(self):
    """
    :return: version string
    :rtype: str
    """
    return "<version>RETURNN.own_threaded_callback</version>"

  # noinspection PyUnusedLocal
  def own_tcb_get_loss_and_error_signal(self, seg_name, seg_len, posteriors):
    """
    :param seg_name:
    :param seg_len:
    :param posteriors:
    :return:
    """
    # Wait until we get the loss and error signal.
    while True:
      with self.cond:
        if self.loss is not None and self.error_signal is not None:
          return self.loss, self.error_signal
        self.cond.wait(timeout=1)

  def init_segment(self, segment_name):
    """
    Called by Sprint PythonControl in FeedForwardTrainer/SegmentwiseNnTrainer.
    """
    if Verbose:
      print("RETURNN SprintControl[pid %i] init_segment %s" % (os.getpid(), segment_name))
    with self.cond:
      assert self.seg_name == segment_name
      self.notified_for_segment = True
      self.cond.notifyAll()

  def notify_segment_loss(self, segment_name, loss):
    """
    Called by Sprint PythonControl in FeedForwardTrainer/SegmentwiseNnTrainer.
    """
    if Verbose:
      print("RETURNN SprintControl[pid %i] notify_segment_loss %s %s" % (os.getpid(), segment_name, loss))
    self.set_current_seg_loss(seg_name=segment_name, loss=loss)

  def get_current_seg_posteriors(self, seg_len):
    """
    :param int seg_len: just for double checking, the length of the current segment
    :return: matrix (time,label)
    """
    with self.cond:
      assert self.seg_len == seg_len
      assert self.posteriors.shape[0] == seg_len
      self.asked_for_posteriors = True
      self.cond.notifyAll()
      return self.posteriors

  def set_current_seg_error_signal(self, seg_len, error_signal):
    """
    :param int seg_len: just for double checking, the length of the current segment
    :param error_signal: matrix (time,label)
    """
    with self.cond:
      assert self.seg_len == seg_len
      assert error_signal.ndim == 2
      assert error_signal.shape[0] == seg_len
      self.error_signal = error_signal
      self.cond.notifyAll()

  def set_current_seg_loss(self, seg_name, loss):
    """
    :param str|None seg_name: just for double checking, the name of the current segment. might be None
    :param float loss: the loss of the current seg
    """
    with self.cond:
      if seg_name:
        assert self.seg_name == seg_name
      self.loss = loss
      self.cond.notifyAll()

  # noinspection PyMethodMayBeStatic
  def _default_skipped_loss(self):
    """
    :rtype: float
    """
    return float("inf")

  # noinspection PyMethodMayBeStatic
  def _default_skipped_error_signal(self, posteriors):
    """
    :param numpy.ndarray posteriors:
    :rtype: numpy.ndarray
    """
    return numpy.zeros_like(posteriors)

  def _skip_segment_loss_and_error(self):
    with self.cond:
      assert self.posteriors is not None
      if self.loss is None:
        self.loss = self._default_skipped_loss()
      if self.error_signal is None:
        self.error_signal = self._default_skipped_error_signal(self.posteriors)
      self.cond.notifyAll()

  def _wait_for_control_loop_error_signal(self):
    while True:
      with self.cond:
        if self.control_thread__have_new_error_signal or self.control_thread__have_new_seg:
          break
        if self.control_loop_exited:
          break
        if self.loss is None or self.error_signal is None:
          break
        if Verbose:
          print("RETURNN SprintControl[pid %i] getSegmentList: wait for control loop to handle error signal" % (
            os.getpid(),))
        self.cond.wait(timeout=1)

  def segment_list_iterator(self):
    """
    :return: yields segment names
    :rtype: typing.Iterator[str]
    """
    with self.cond:
      assert self.control_loop_started

    while True:  # outer loop
      # wait until we get new segment
      while True:
        with self.cond:
          if self.control_thread__have_new_seg:
            assert self.seg_name
            seg_name = self.seg_name
            self.control_thread__have_new_seg = False
            break
          if self.control_loop_exited:
            return  # no more segments
          self.cond.wait(timeout=1)

      # We got a new segment name from the parent RETURNN process (via self.handle_cmd_get_loss_and_error_signal()).
      # We wait in this segment because we wait to get the error signal from Sprint
      # (via SprintNnPythonLayer.backpropagate()).
      # Sprint waits currently for us to get the new segment (in the PythonSegmentOrder code).
      # Once it gets it, it will call SprintNnPythonLayer.forward(), then calculate the loss and error signal
      # and then call SprintNnPythonLayer.backpropagate().
      if Verbose:
        print("RETURNN SprintControl[pid %i] getSegmentList, yield %r" % (os.getpid(), seg_name))
      yield seg_name

      # We might need to wait for the control loop thread.
      self._wait_for_control_loop_error_signal()

      # When we are back here, Sprint asks for the next segment.
      # It means that is has finished any processing with this segment.
      with self.cond:
        # See self.handle_cmd_get_loss_and_error_signal().
        # We are still stuck in there in the other thread, if not self.have_new_error_signal.
        # Maybe the PythonLayer was not used?
        # Or Sprint could not calculate the criterion for this segment (bad lattice or so).
        if not (self.control_thread__have_new_error_signal or self.control_thread__have_new_seg):
          print("RETURNN SprintControl[pid %i] getSegmentList, no error signal, skip segment: %s" % (
            os.getpid(), seg_name))
          if Verbose:
            # Print Sprint stacktrace.
            import signal
            os.kill(os.getpid(), signal.SIGUSR1)
          if not self.notified_for_segment:
            print(("RETURNN SprintControl[pid %i] getSegmentList: "
                   "Do you use PythonControl in the Sprint trainer? Got no segment notification.") % (os.getpid(),))
          if not self.asked_for_posteriors:
            print(("RETURNN SprintControl[pid %i] getSegmentList: "
                   "Do you use PythonLayer in Sprint? Did not get asked for posteriors.") % (os.getpid(),))
          self._skip_segment_loss_and_error()
          self._wait_for_control_loop_error_signal()
