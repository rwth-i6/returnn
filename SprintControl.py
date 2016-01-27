
# This is the Sprint PythonControl interface implementation.
# For reference, in Sprint code, see:
#  * src/Nn/PythonControl.cc
#  * src/Tools/NnTrainer/python_control_demo.py
# This interface will behave similar as SprintExternInterface.
# See SprintErrorSignals for the other end.

print("CRNN SprintControl Python module load")

import rnn
import Debug
import os
from TaskSystem import Pickler, Unpickler


startTime = None
isInitialized = False
control = None; ":type: ExternControl"

rnn.initBetterExchook()
Debug.initFaulthandler(sigusr1_chain=True)  # Sprint also handles SIGUSR1.
rnn.initThreadJoinHack()

# Start Sprint PythonControl interface. {

def init(name, reference, config, **kwargs):
  print("CRNN SprintControl init: %r, %r, %r, %r" % (name, reference, config, kwargs))

  global isInitialized, control
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
    self.pipe_c2p = os.fdopen(c2p_fd, "w")
    self.pipe_p2c = os.fdopen(p2c_fd, "r")
    self.sprint_callback = None

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

  def handle_cmd(self, cmd, *args):
    func = getattr(self, "handle_cmd_%s" % cmd, None)
    if not func:
      return self.sprint_callback(cmd, *args)
    return func(*args)

  def run_control_loop(self, sprint_callback):
    self.sprint_callback = sprint_callback
    while True:
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
