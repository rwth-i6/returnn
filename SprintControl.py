
# This is the Sprint PythonControl interface implementation.
# For reference, in Sprint code, see:
#  * src/Nn/PythonControl.cc
#  * src/Tools/NnTrainer/python_control_demo.py
# This interface will behave similar as SprintExternInterface.
# See SprintErrorSignals for the other end.
# It can also be used as a PythonSegmentOrdering interface.
# It also supports SprintNnPythonLayer.

print("CRNN SprintControl Python module load")

import rnn
import Debug
import os
import numpy
from TaskSystem import Pickler, Unpickler
from threading import RLock, Condition


isInitialized = False
control = None; ":type: ExternControl"

rnn.initBetterExchook()
Debug.initFaulthandler(sigusr1_chain=True)  # Sprint also handles SIGUSR1.
rnn.initThreadJoinHack()

# Start Sprint PythonControl interface. {

def init(name, reference, config, **kwargs):
  print("CRNN SprintControl init: %r, %r, %r, %r" % (name, reference, config, kwargs))

  global isInitialized, control
  if isInitialized: return  # could be called multiple times from multiple sources
  isInitialized = True

  config = config.split(",")
  config = {key: value for (key, value) in [s.split(":", 1) for s in config if s]}

  control = ExternControl(c2p_fd=int(config["c2p_fd"]), p2c_fd=int(config["p2c_fd"]))

def run_control_loop(callback, **kwargs):
  print("CRNN SprintControl run_control_loop: %r, %r" % (callback, kwargs))
  print(">> Sprint Version: %r" % callback("version"))
  control.run_control_loop(sprint_callback=callback)

def exit(name, reference, **kwargs):
  print("CRNN SprintControl exit: %r, %r, %r" % (name, reference, kwargs))
  control.close()

# End Sprint PythonControl interface. }

# Start Sprint PythonSegmentOrder interface. {

def getSegmentList(corpusName, segmentList, config, **kwargs):
  print("CRNN SprintControl getSegmentList: %r, %r" % (corpusName, config))

  # Wrap the other interface.
  init(name="CRNN.PythonSegmentOrder", reference=corpusName, config=config)

  #control.run_control_loop(None)

  #while True:


# End Sprint PythonSegmentOrder interface. }

# Start SprintNnPythonLayer. {

class SprintNnPythonLayer:
  def __init__(self, config, **kwargs):
    print("SprintNnPythonLayer.__init__: %r, %r" % (config, kwargs))
    self.input_size = None
    self.output_size = None

  def setInputDimension(self, stream, size):
    print("SprintNnPythonLayer.setInputDimension: %r, %r" % (stream, size))
    assert stream == 0, "we only support a single input stream (for now)"
    self.input_size = size

  def setOutputDimension(self, size):
    print("SprintNnPythonLayer.setOutputDimension: %r" % size)
    self.output_size = size

  def initializeNetworkParameters(self):
    print("SprintNnPythonLayer.initializeNetworkParameters")
    # Just ignore.

  def loadNetworkParameters(self, filename):
    print("SprintNnPythonLayer.loadNetworkParameters: %r" % filename)
    # Just ignore.

  def saveNetworkParameters(self, filename):
    print("SprintNnPythonLayer.saveNetworkParameters: %r" % filename)
    # Just ignore.

  def isTrainable(self):
    # Always trainable.
    return True

  def getNumberOfFreeParameters(self):
    # For now, just ignore. Not important.
    return 0

  def forward(self, input):
    assert len(input) == 1
    input = input[0]  # single input stream
    # TODO...
    pass

  def backpropagate(self, errorSignalIn):
    # TODO...
    # must return a 1-tuple
    pass

# End SprintNnPythonLayer. }


class ExternControl:

  """
  This will send data to CRNN over a pipe.
  We expect that we are child process and the parent process has spawned us,
  """

  Version = 1  # increase when some protocol changes

  def __init__(self, c2p_fd, p2c_fd):
    """
    :param int c2p_fd: child-to-parent file descriptor
    :param int p2c_fd: parent-to-child file descriptor
    """
    self.cond = Condition()
    self.pipe_c2p = os.fdopen(c2p_fd, "w")
    self.pipe_p2c = os.fdopen(p2c_fd, "r")
    self.sprint_callback = None
    self.have_new_seg = False
    self.seg_name = None
    self.seg_len = None
    self.posteriors = None

  def _send(self, dataType, args=None):
    Pickler(self.pipe_c2p).dump((dataType, args))
    self.pipe_c2p.flush()

  def _read(self):
    return Unpickler(self.pipe_p2c).load()

  def close(self):
    self.pipe_c2p.close()
    self.pipe_p2c.close()

  def handle_cmd_exit(self):
    self.close()
    raise SystemExit

  def handle_cmd_init(self, name, version):
    assert version == self.Version
    return "SprintControl", self.Version

  def handle_get_loss_and_error_signal(self, seg_name, seg_len, posteriors_str):
    with self.cond:
      self.seg_name = seg_name
      self.have_new_seg = True
      self.cond.notifyAll()
    self.seg_len = seg_len
    self.posteriors = numpy.fromstring(posteriors_str, dtype="float32")
    # TODO...
    raise NotImplementedError

  def handle_cmd(self, cmd, *args):
    func = getattr(self, "handle_cmd_%s" % cmd, None)
    return func(*args)

  def handle_next(self):
    args = self._read()
    try:
      if not isinstance(args, tuple): raise TypeError("expected tuple but got %r" % args)
      if len(args) < 1: raise Exception("need multiple args (cmd, ...)")
      res = self.handle_cmd(*args)
    except Exception as e:
      self._send(("exception", str(e)))
    else:
      assert isinstance(res, tuple)
      self._send(("ok",) + res)

  def run_control_loop(self, sprint_callback):
    self.sprint_callback = sprint_callback
    while True:
      self.handle_next()
