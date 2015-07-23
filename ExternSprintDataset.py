
import sys
import os
import thread
import atexit
import signal
from threading import Thread
from SprintDataset import SprintDataset
from TaskSystem import Pickler, Unpickler
from Util import eval_shell_str, interrupt_main
from Log import log


class ExternSprintDataset(SprintDataset):

  """
  This because like SprintDataset, except that we will start an external Sprint instance ourselves
  which will forward the data to us over a pipe.
  See SprintExternInterface.
  """

  def __init__(self, sprintTrainerExecPath, sprintConfigStr, *args, **kwargs):
    """
    :type sprintTrainerExecPath: str
    :type sprintConfigStr: str
    """
    super(ExternSprintDataset, self).__init__(*args, **kwargs)
    self.add_data_thread_id = None
    self.sprintTrainerExecPath = sprintTrainerExecPath
    self.sprintConfig = eval_shell_str(sprintConfigStr)
    self._num_seqs = None
    self.child_pid = None
    self.parent_pid = os.getpid()
    self.useMultipleEpochs()
    # There is no generic way to see whether Python is exiting.
    # This is our workaround. We check for it in self.run_inner().
    self.python_exit = False
    #atexit.register(self.exit_handler)
    self.init_epoch()

  def _exit_child(self, wait_thread=True):
    if self.child_pid:
      interrupt = False
      if self._join_child(wait=False, expected_exit_status=0) is False:  # Not yet terminated.
        interrupt = not self.reached_final_seq
        if interrupt:
          print >> log.v5, "ExternSprintDataset: interrupt child proc %i" % self.child_pid
          os.kill(self.child_pid, signal.SIGKILL)
      else:
        self.child_pid = None
      if wait_thread:
        # Load all remaining data so that the reader thread is not waiting in self.addNewData().
        while self.is_less_than_num_seqs(self.expected_load_seq_start + 1):
          self.load_seqs(self.expected_load_seq_start + 1, self.expected_load_seq_start + 2)
        self.reader_thread.join()
      try: self.pipe_p2c[1].close()
      except IOError: pass
      try: self.pipe_c2p[0].close()
      except IOError: pass
      if self.child_pid:
        self._join_child(wait=True, expected_exit_status=0 if not interrupt else None)
        self.child_pid = None

  def _start_child(self, epoch):
    assert self.child_pid is None
    self.pipe_c2p = self._pipe_open()
    self.pipe_p2c = self._pipe_open()
    args = self._build_sprint_args()
    print >>log.v5, "ExternSprintDataset: epoch", epoch, "exec", args

    pid = os.fork()
    if pid == 0:  # child
      try:
        sys.stdin.close()  # Force no tty stdin.
        self.pipe_c2p[0].close()
        self.pipe_p2c[1].close()
        os.execv(args[0], args)  # Does not return if successful.
      except BaseException:
        print >> log.v1, "ExternSprintDataset: Error when starting Sprint %r." % args
        sys.excepthook(*sys.exc_info())
      finally:
        os._exit(1)
        return  # Not reached.

    # parent
    self.pipe_c2p[1].close()
    self.pipe_p2c[0].close()
    self.child_pid = pid

    try:
      initSignal, (inputDim, outputDim, num_segments) = self._read_next_raw()
      assert initSignal == "init"
      assert isinstance(inputDim, int) and isinstance(outputDim, int)
      # Ignore num_segments. It can be totally different than the real number of sequences.
      self.setDimensions(inputDim, outputDim)
    except Exception:
      print >> log.v1, "ExternSprintDataset: Sprint child process (%r) caused an exception." % args
      sys.excepthook(*sys.exc_info())
      raise Exception("ExternSprintDataset Sprint init failed")

    self.reader_thread = Thread(target=self.reader_thread_proc, args=(pid, epoch,),
                                name="ExternSprintDataset reader thread")
    self.reader_thread.daemon = True
    self.reader_thread.start()

  def _pipe_open(self):
    readend, writeend = os.pipe()
    readend = os.fdopen(readend, "r", 0)
    writeend = os.fdopen(writeend, "w", 0)
    return readend, writeend

  @property
  def _my_python_mod_path(self):
    return os.path.dirname(os.path.abspath(__file__))

  def _build_sprint_args(self):
    return [self.sprintTrainerExecPath] + [
      "--*.seed=%i" % (self.crnnEpoch or 1),
      "--*.python-segment-order=true",
      "--*.python-segment-order-pymod-path=%s" % self._my_python_mod_path,
      "--*.python-segment-order-pymod-name=SprintExternInterface",
      "--*.use-data-source=false",
      "--*.trainer=python-trainer",
      "--*.pymod-path=%s" % self._my_python_mod_path,
      "--*.pymod-name=SprintExternInterface",
      "--*.pymod-config=action:ExternSprintDataset,c2p_fd:%i,p2c_fd:%i" % (self.pipe_c2p[1].fileno(),
                                                                           self.pipe_p2c[0].fileno())] + \
      self.sprintConfig

  def _read_next_raw(self):
    dataType, args = Unpickler(self.pipe_c2p[0]).load()
    return dataType, args

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

  def reader_thread_proc(self, child_pid, epoch):
    try:
      self.add_data_thread_id = thread.get_ident()

      self.initSprintEpoch(epoch)
      haveSeenTheWhole = False

      while not self.python_exit:
        try:
          dataType, args = self._read_next_raw()
        except (IOError, EOFError):
          with self.lock:
            if epoch != self.crnnEpoch:
              # We have passed on to a new epoch. This is a valid reason that the child has been killed.
              break
            if self.python_exit:
              break
          raise

        with self.lock:
          if epoch != self.crnnEpoch:
            break
          if self.python_exit:
            break

          if dataType == "data":
            features, targets = args
            self.addNewData(features, targets)
          elif dataType == "exit":
            haveSeenTheWhole = True
            break
          else:
            assert False, "not handled: (%r, %r)" % (dataType, args)

      if not self.python_exit:
        with self.lock:
          self.finishSprintEpoch()
          if haveSeenTheWhole:
            self._num_seqs = self.next_seq_to_be_added
      print >> log.v5, "ExternSprintDataset finished reading epoch %i" % epoch

    except Exception:
      # Catch all standard exceptions.
      # Don't catch KeyboardInterrupt here because that will get send by the main thread
      # when it is exiting. It's never by the user because SIGINT will always
      # trigger KeyboardInterrupt in the main thread only.
      try:
        print >> log.v1, "ExternSprintDataset reader failed"
        sys.excepthook(*sys.exc_info())
        print ""
      finally:
        # Exceptions are fatal. If we can recover, we should handle it in run_inner().
        interrupt_main()

  def exit_handler(self):
    assert os.getpid() == self.parent_pid
    self.python_exit = True
    self._exit_child(wait_thread=False)

  def init_epoch(self, epoch=None):
    if epoch is None:
      epoch = 1
    with self.lock:
      if epoch == self.crnnEpoch:
        return
      super(ExternSprintDataset, self).init_seq_order(epoch=epoch)
    self._exit_child()
    self._start_child(epoch)

  def init_seq_order(self, epoch=None):
    self.init_epoch(epoch)

  @property
  def num_seqs(self):
    with self.lock:
      assert self._num_seqs is not None
      return self._num_seqs

  def get_complete_frac(self, seq_idx):
    """
    :return: Returns a fraction (float in [0,1], always > 0) of how far we have advanced
      for this seq in the dataset.
      This does not have to be exact. This is only for the user.
    """
    with self.lock:
      if self._num_seqs is not None:
        return float(seq_idx + 1) / self._num_seqs
      return super(ExternSprintDataset, self).get_complete_frac(seq_idx)

  def len_info(self):
    with self.lock:
      if self._num_seqs is not None:
        return "ExternSprintDataset, %i seqs" % self._num_seqs
      else:
        return "ExternSprintDataset, unknown number of seqs"
