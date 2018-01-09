
"""
Implements the SprintDatasetBase and ExternSprintDataset classes, some Dataset subtypes.
Note that from the main RETURNN process, you probably want ExternSprintDataset instead.
"""

from __future__ import print_function

import atexit
import os
import signal
import sys
try:
  import thread
except ImportError:
  import _thread as thread
from threading import Condition, currentThread, Thread
import math
import time
import numpy

import TaskSystem
from Dataset import Dataset, DatasetSeq
from CachedDataset2 import CachedDataset2
from Log import log
from TaskSystem import Unpickler, numpy_copy_and_set_unused
from Util import eval_shell_str, interrupt_main, unicode


class SprintDatasetBase(Dataset):
  """
  In Sprint, we use this object for multiple purposes:
  - Multiple epoch handling via SprintInterface.getSegmentList().
    For this, we get the segment list from Sprint and use the Dataset
    shuffling method.
  - Fill in data which we get via SprintInterface.feedInput*().
    Note that each such input doesn't necessarily correspond to a single
    segment. This depends which type of FeatureExtractor is used in Sprint.
    If we use the BufferedFeatureExtractor in utterance mode, we will get
    one call for every segment and we get also segmentName as parameter.
    Otherwise, we will get batches of fixed size - in that case,
    it doesn't correspond to the segments.
    In any case, we use this data as-is as a new seq.
    Because of that, we cannot really know the number of seqs in advance,
    nor the total number of time frames, etc.

  If you want to use this directly in RETURNN, see ExternSprintDataset.
  """

  SprintCachedSeqsMax = 200
  SprintCachedSeqsMin = 100

  def __init__(self, target_maps=None, str_add_final_zero=False, input_stddev=1., bpe=None, **kwargs):
    """
    :param dict[str,str|dict] target_maps: e.g. {"speaker": "speaker_map.txt"}
    :param bool str_add_final_zero: adds e.g. "orth0" with '\0'-ending
    :param float input_stddev: if != 1, will divide the input "data" by that
    :param None|dict[str] bpe: if given, will be opts for :class:`BytePairEncoding`
    """
    super(SprintDatasetBase, self).__init__(**kwargs)
    if target_maps:
      assert isinstance(target_maps, dict)
      target_maps = target_maps.copy()
      for key, tmap in list(target_maps.items()):
        if isinstance(tmap, (str, unicode)):
          tmap = {l: i for (i, l) in enumerate(open(tmap).read().splitlines())}
        assert isinstance(tmap, dict)  # dict[str,int]
        target_maps[key] = tmap
    self.target_maps = target_maps
    self.str_add_final_zero = str_add_final_zero
    self.input_stddev = input_stddev
    self.bpe = None
    if bpe:
      from GeneratingDataset import BytePairEncoding
      self.bpe = BytePairEncoding(**bpe)
    self.cond = Condition(lock=self.lock)
    self.add_data_thread_id = thread.get_ident()  # This will be created in the Sprint thread.
    self.ready_for_data = False
    self.reached_final_seq = False
    self.reached_final_seq_seen_all = False
    self.multiple_epochs = False
    self._complete_frac = None
    self.sprintEpoch = None  # in SprintInterface.getSegmentList()
    self.crnnEpoch = None  # in CRNN train thread, Engine.train(). set via init_seq_order
    self.predefined_seq_list_order = None  # via init_seq_order
    self.sprintFinalized = False
    self._target_black_list = []  # if we get non numpy arrays and cannot convert them
    self._resetCache()
    assert self.shuffle_frames_of_nseqs == 0  # Currently broken. But just use Sprint itself to do this.

  def useMultipleEpochs(self):
    """
    Called via SprintInterface.getSegmentList().
    """
    self.multiple_epochs = True

  def setDimensions(self, inputDim, outputDim):
    """
    :type inputDim: int
    :type outputDim: int
    Called via python_train.
    """
    assert inputDim > 0
    self.num_inputs = inputDim
    self.num_outputs = {"data": (inputDim * self.window, 2)}
    if outputDim > 0:
      self.num_outputs["classes"] = (outputDim, 1)
    if self.bpe:
      self.num_outputs["bpe"] = (self.bpe.num_labels, 1)
      self.labels["bpe"] = self.bpe.labels
    self._base_init()
    # At this point, we are ready for data. In case we don't use the Sprint PythonSegmentOrdering
    # (SprintInterface.getSegmentList()), we must call this at least once.
    if not self.multiple_epochs:
      self.initSprintEpoch(None)

  def _resetCache(self):
    self.expected_load_seq_start = 0
    self.requested_load_seq_end = 0
    self.next_seq_to_be_added = 0
    self.reached_final_seq = False
    self.reached_final_seq_seen_all = False
    self._num_timesteps = 0
    self.added_data = []; " :type: list[DatasetSeq] "
    self.ready_for_data = True

  def initSprintEpoch(self, epoch):
    """
    :type epoch: int | None
    Called by SprintInterface.getSegmentList() when we start a new epoch.
    We must not call this via self.init_seq_order() because we will already have filled the cache by Sprint
    before the CRNN train thread starts the epoch.
    """
    with self.lock:
      self.sprintEpoch = epoch
      self.sprintFinalized = False
      self._resetCache()
      self.cond.notify_all()

  def finalizeSprint(self):
    """
    Called when SprintInterface.getSegmentList() ends.
    """
    with self.lock:
      self.sprintFinalized = True
      self.cond.notify_all()

  def init_seq_order(self, epoch=None, seq_list=None):
    """
    Called by CRNN train thread when we enter a new epoch.
    """
    super(SprintDatasetBase, self).init_seq_order(epoch=epoch, seq_list=seq_list)
    with self.lock:
      self.crnnEpoch = epoch
      self.predefined_seq_list_order = seq_list
      self.cond.notify_all()
      # No need to wait/check for Sprint thread here.
      # SprintInterface.getSegmentList() will wait for us.
    return True

  def _cleanupOldSeqCache(self, seqEnd):
    i = 0
    while i < len(self.added_data):
      if self.added_data[i].seq_idx >= seqEnd:
        break
      i += 1
    del self.added_data[:i]

  def waitForCrnnEpoch(self, epoch):
    """
    Called by SprintInterface.
    """
    with self.lock:
      while epoch != self.crnnEpoch:
        assert epoch > self.crnnEpoch
        self.cond.wait()

  def _waitForSeq(self, seqStart, seqEnd=None):
    """
    Called by CRNN train thread.
    """
    if seqEnd is None:
      seqEnd = seqStart + 1
    if seqEnd > self.requested_load_seq_end:
      self.requested_load_seq_end = seqEnd
      self.cond.notify_all()
    def check():
      if self.reached_final_seq:
        return True
      if self._haveSeqsAdded(seqStart, seqEnd):
        return True
      return False
    if check():
      return
    # We need to wait.
    assert thread.get_ident() != self.add_data_thread_id
    print("SprintDataset wait for seqs (%i,%i) (last added: %s) (current time: %s)" % \
                     (seqStart, seqEnd, self._latestAddedSeq(), time.strftime("%H:%M:%S")), file=log.v5)
    while not check():
      self.cond.wait()

  def _latestAddedSeq(self):
    if not self.added_data:
      return None
    return self.added_data[-1].seq_idx

  def _haveSeqsAdded(self, start, end=None):
    if end is None:
      end = start + 1
    if start >= end:
      return True
    for data in self.added_data:
      assert start >= data.seq_idx, "We expect that we only ask about the cache of the upcoming seqs."
      if data.seq_idx == start:
        start += 1
      if start >= end:
        return True
    return False

  def _getSeq(self, seq_idx):
    for data in self.added_data:
      if data.seq_idx == seq_idx:
        return data
    return None

  def is_cached(self, start, end):
    # Always False, to force that we call self._load_seqs().
    # This is important for our buffer management.
    return False

  def load_seqs(self, start, end):
    # Called by CRNN train thread.
    print("SprintDataset load_seqs in %s:" % currentThread().name, start, end, end=' ', file=log.v5)
    if start == end: return
    with self.lock:
      super(SprintDatasetBase, self).load_seqs(start, end)
      print("first features shape:", self._getSeq(start).features.shape, file=log.v5)

  def _load_seqs(self, start, end):
    # Called by CRNN train thread.
    # We expect that start increase monotonic on each call
    # for not-yet-loaded data.
    # This will already be called with _load_seqs_superset indices.
    assert start >= self.expected_load_seq_start
    if start > self.expected_load_seq_start:
      # Cleanup old data.
      self._cleanupOldSeqCache(start)
      self.expected_load_seq_start = start
      self.cond.notify_all()
    self._waitForSeq(start, end)

  def addNewData(self, features, targets=None, segmentName=None):
    """
    Adds a new seq.
    This is called via the Sprint main thread.
    :param numpy.ndarray features: format (input-feature,time) (via Sprint)
    :param dict[str,numpy.ndarray|str] targets: format (time) (idx of output-feature)
    :returns the sorted seq index
    :rtype: int
    """

    # is in format (feature,time)
    assert self.num_inputs == features.shape[0]
    T = features.shape[1]
    # must be in format: (time,feature)
    features = features.transpose()
    assert features.shape == (T, self.num_inputs)
    if self.input_stddev != 1:
      features /= self.input_stddev
    if self.window > 1:
      features = self.sliding_window(features)
      assert features.shape == (T, self.num_inputs * self.window)

    if targets is None:
      targets = {}
    if not isinstance(targets, dict):
      targets = {"classes": targets}
    if "classes" in targets:
      # 'classes' is always the alignment
      assert targets["classes"].shape == (T,), (  # is in format (time,)
        "Number of targets %s does not equal to number of features %s" % (targets["classes"].shape, (T,)))
    if self.bpe:
      assert "orth" in targets
      orth = targets["orth"]
      assert isinstance(orth, (str, unicode))
      assert "bpe" not in targets
      targets["bpe"] = numpy.array(self.bpe.get_seq(orth.strip()), dtype="int32")

    # Maybe convert some targets.
    if self.target_maps:
      for key, tmap in self.target_maps.items():
        assert key in targets
        v = tmap[targets[key]]
        v = numpy.asarray(v)
        if v.ndim == 0:
          v = numpy.zeros((T,), dtype=v.dtype) + v  # add time dimension
        targets[key] = v

    # Maybe remove some targets.
    for key in self._target_black_list:
      if key in targets:
        del targets[key]

    # Check if all targets are valid.
    for key, v in sorted(list(targets.items())):
      if isinstance(v, numpy.ndarray):
        continue  # ok
      if isinstance(v, unicode):
        v = v.encode("utf8")
      if isinstance(v, (str, bytes)):
        v = list(map(ord, v))
        v = numpy.array(v, dtype="uint8")
        targets[key] = v
        if self.str_add_final_zero:
          v = numpy.append(v, numpy.array([0], dtype=v.dtype))
          assert key + "0" not in targets
          targets[key + "0"] = v
        continue
      print("SprintDataset, we will ignore the target %r because it is not a numpy array: %r" % (key, v), file=log.v3)
      self._target_black_list += [key]
      del targets[key]

    with self.lock:
      # This gets called in the Sprint main thread.
      # If this is used together with SprintInterface.getSegmentList(), we are always in a state where
      # we just yielded a segment name, thus we are always in a Sprint epoch and thus ready for data.
      assert self.ready_for_data
      assert not self.reached_final_seq
      assert not self.sprintFinalized

      seq_idx = self.next_seq_to_be_added

      if self.predefined_seq_list_order:
        # Note: Only in ExternSprintDataset, we can reliably set the seq order for now.
        assert self.predefined_seq_list_order[seq_idx] == segmentName, "seq-order not as expected"

      self.next_seq_to_be_added += 1
      self._num_timesteps += T
      self.cond.notify_all()

      if seq_idx > self.requested_load_seq_end - 1 + self.SprintCachedSeqsMax:
        print("SprintDataset addNewData: seq=%i, len=%i. Cache filled, waiting to get loaded..." % (seq_idx, T), file=log.v5)
        while seq_idx > self.requested_load_seq_end - 1 + self.SprintCachedSeqsMin:
          assert not self.reached_final_seq
          assert seq_idx + 1 == self.next_seq_to_be_added
          self.cond.wait()

      self.added_data += [DatasetSeq(seq_idx, features, targets, seq_tag=segmentName)]
      self.cond.notify_all()
      return seq_idx

  def finishSprintEpoch(self, seen_all=True):
    """
    Called by SprintInterface.getSegmentList().
    This is in a state where Sprint asks for the next segment after we just finished an epoch.
    Thus, any upcoming self.addNewData() call will contain data from a segment in the new epoch.
    Thus, we finish the current epoch in Sprint.
    """
    with self.lock:
      self.reached_final_seq = True
      self.reached_final_seq_seen_all = seen_all
      self.ready_for_data = False
      self.cond.notify_all()

  def _shuffle_frames_in_seqs(self, start, end):
    assert False, "Shuffling in SprintDataset only via Sprint at the moment."

  def get_num_timesteps(self):
    with self.lock:
      assert self.reached_final_seq
      return self._num_timesteps

  @property
  def num_seqs(self):
    with self.lock:
      if self.predefined_seq_list_order:
        return len(self.predefined_seq_list_order)
      assert self.reached_final_seq
      return self.next_seq_to_be_added

  def have_seqs(self):
    with self.lock:
      if self.next_seq_to_be_added > 0:
        return True
      self._waitForSeq(0)
      return self.next_seq_to_be_added > 0

  def is_less_than_num_seqs(self, n):
    with self.lock:
      self._waitForSeq(n)
      return n < self.next_seq_to_be_added

  def get_target_list(self):
    with self.lock:
      if not self.added_data:
        self._waitForSeq(0)
      assert self.added_data
      return self.added_data[0].targets.keys()

  def set_complete_frac(self, frac):
    self._complete_frac = frac

  def get_complete_frac(self, seq_idx):
    with self.lock:
      if self.predefined_seq_list_order:
        return float(seq_idx + 1) / len(self.predefined_seq_list_order)
      elif self._complete_frac is not None:
        if not self.next_seq_to_be_added:
          return self._complete_frac
        else:
          # We can do somewhat better. self._complete_frac is for self.next_seq_to_be_added.
          return self._complete_frac * float(seq_idx + 1) / self.next_seq_to_be_added
      else:
        return super(SprintDatasetBase, self).get_complete_frac(seq_idx)

  def get_seq_length(self, sorted_seq_idx):
    with self.lock:
      self._waitForSeq(sorted_seq_idx)
      return self._getSeq(sorted_seq_idx).num_frames

  def get_input_data(self, sorted_seq_idx):
    with self.lock:
      self._waitForSeq(sorted_seq_idx)
      return self._getSeq(sorted_seq_idx).features

  def get_targets(self, target, sorted_seq_idx):
    with self.lock:
      self._waitForSeq(sorted_seq_idx)
      return self._getSeq(sorted_seq_idx).targets.get(target, None)

  def get_ctc_targets(self, sorted_seq_idx):
    assert False, "No CTC targets."

  def get_tag(self, sorted_seq_idx):
    with self.lock:
      self._waitForSeq(sorted_seq_idx)
      return self._getSeq(sorted_seq_idx).seq_tag


class ExternSprintDataset(SprintDatasetBase):
  """
  This is a Dataset which you can use directly in RETURNN.
  You can use it to get any type of data from Sprint (RWTH ASR toolkit),
  e.g. you can use Sprint to do feature extraction and preprocessing.

  This class is like SprintDatasetBase, except that we will start an external Sprint instance ourselves
  which will forward the data to us over a pipe.
  The Sprint subprocess will use SprintExternInterface to communicate with us.
  """

  def __init__(self, sprintTrainerExecPath, sprintConfigStr, partitionEpoch=1, **kwargs):
    """
    :param str|list[str] sprintTrainerExecPath:
    :param str | list[str] | ()->str | list[()->str] | ()->list[str] | ()->list[()->str] sprintConfigStr: via eval_shell_str
    """
    super(ExternSprintDataset, self).__init__(**kwargs)
    self.add_data_thread_id = None
    self.sprintTrainerExecPath = sprintTrainerExecPath
    self.sprintConfig = sprintConfigStr
    self.partitionEpoch = partitionEpoch
    self._num_seqs = None
    self.child_pid = None  # type: int|None
    self.parent_pid = os.getpid()
    self.seq_list_file = None
    self.useMultipleEpochs()
    # There is no generic way to see whether Python is exiting.
    # This is our workaround. We check for it in self.run_inner().
    self.python_exit = False
    atexit.register(self.exit_handler)
    self.init_epoch()

  def _exit_child(self, wait_thread=True):
    if self.child_pid:
      interrupt = False
      expected_exit_status = 0 if not self.python_exit else None
      if self._join_child(wait=False, expected_exit_status=expected_exit_status) is False:  # Not yet terminated.
        interrupt = not self.reached_final_seq_seen_all
        if interrupt:
          print("ExternSprintDataset: interrupt child proc %i" % self.child_pid, file=log.v5)
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
    print("ExternSprintDataset: epoch", epoch, "exec", args, file=log.v5)

    pid = os.fork()
    if pid == 0:  # child
      # In case we are in some test environment or so, recover the original stdout/stderr.
      sys.stdin = sys.__stdin__
      sys.stdout = sys.__stdout__
      sys.stderr = sys.__stderr__
      import better_exchook
      better_exchook.install()
      try:
        sys.stdin.close()  # Force no tty stdin.
        self.pipe_c2p[0].close()
        self.pipe_p2c[1].close()
        os.execv(args[0], args)  # Does not return if successful.
        print("ExternSprintDataset child exec failed.")
      except BaseException:
        print("ExternSprintDataset child: Error when starting Sprint %r." % args)
        sys.excepthook(*sys.exc_info())
      finally:
        print("ExternSprintDataset child: exit")
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
      print("ExternSprintDataset: Sprint child process (%r) caused an exception." % args, file=log.v1)
      sys.excepthook(*sys.exc_info())
      self._join_child()
      self.child_pid = None
      raise Exception("ExternSprintDataset Sprint init failed")

    self.reader_thread = Thread(target=self.reader_thread_proc, args=(pid, epoch,),
                                name="ExternSprintDataset reader thread")
    self.reader_thread.daemon = True
    self.reader_thread.start()

  def _pipe_open(self):
    readend, writeend = os.pipe()
    if hasattr(os, "set_inheritable"):
      # Python 3 by default will close all fds in subprocesses. This will avoid that.
      os.set_inheritable(readend, True)
      os.set_inheritable(writeend, True)
    readend = os.fdopen(readend, "rb", 0)
    writeend = os.fdopen(writeend, "wb", 0)
    return readend, writeend

  @property
  def _my_python_mod_path(self):
    return os.path.dirname(os.path.abspath(__file__))

  def _build_sprint_args(self):
    config_str = "action:ExternSprintDataset,c2p_fd:%i,p2c_fd:%i" % (
      self.pipe_c2p[1].fileno(), self.pipe_p2c[0].fileno())
    if TaskSystem.SharedMemNumpyConfig["enabled"]:
      config_str += ",EnableAutoNumpySharedMemPickling:True"
    epoch = self.crnnEpoch or 1
    if isinstance(self.sprintTrainerExecPath, (list, tuple)):
      args = list(self.sprintTrainerExecPath)
    else:
      args = [self.sprintTrainerExecPath]
    args += [
      "--*.seed=%i" % (epoch // self.partitionEpoch)]
    if self.partitionEpoch > 1:
      args += [
        "--*.corpus.partition=%i" % self.partitionEpoch,
        "--*.corpus.select-partition=%i" % (epoch % self.partitionEpoch)]
    args += [
      "--*.python-segment-order=true",
      "--*.python-segment-order-pymod-path=%s" % self._my_python_mod_path,
      "--*.python-segment-order-pymod-name=SprintExternInterface",
      "--*.use-data-source=false",
      "--*.trainer=python-trainer",
      "--*.pymod-path=%s" % self._my_python_mod_path,
      "--*.pymod-name=SprintExternInterface",
      "--*.pymod-config=%s" % config_str]
    if self.predefined_seq_list_order:
      import tempfile
      self.seq_list_file = tempfile.mktemp(prefix="crnn-sprint-predefined-seq-list")
      with open(self.seq_list_file, "w") as f:
        for tag in self.predefined_seq_list_order:
          f.write(tag)
          f.write("\n")
        f.close()
      args += ["--*.corpus.segments.file=%s" % self.seq_list_file]
    args += eval_shell_str(self.sprintConfig)
    return args

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

      while not self.python_exit and self.child_pid:
        try:
          dataType, args = self._read_next_raw()
        except (IOError, EOFError):
          with self.lock:
            if epoch != self.crnnEpoch:
              # We have passed on to a new epoch. This is a valid reason that the child has been killed.
              break
            if self.python_exit or not self.child_pid:
              break
          raise

        with self.lock:
          if epoch != self.crnnEpoch:
            break
          if self.python_exit or not self.child_pid:
            break

          if dataType == "data":
            segmentName, features, targets = args
            self.addNewData(numpy_copy_and_set_unused(features), numpy_copy_and_set_unused(targets), segmentName=segmentName)
          elif dataType == "exit":
            haveSeenTheWhole = True
            break
          else:
            assert False, "not handled: (%r, %r)" % (dataType, args)

      if self.seq_list_file:
        try:
          os.remove(self.seq_list_file)
        except Exception as e:
          print("ExternSprintDataset: error when removing %r: %r" % (self.seq_list_file, e), file=log.v5)
        finally:
          self.seq_list_file = None

      if not self.python_exit and self.child_pid:
        with self.lock:
          self.finishSprintEpoch(seen_all=haveSeenTheWhole)
          if haveSeenTheWhole:
            self._num_seqs = self.next_seq_to_be_added
      print("ExternSprintDataset finished reading epoch %i, seen all %r" % (epoch, haveSeenTheWhole), file=log.v5)

    except Exception:
      if not self.python_exit:
        # Catch all standard exceptions.
        # Don't catch KeyboardInterrupt here because that will get send by the main thread
        # when it is exiting. It's never by the user because SIGINT will always
        # trigger KeyboardInterrupt in the main thread only.
        try:
          print("ExternSprintDataset reader failed", file=log.v1)
          sys.excepthook(*sys.exc_info())
          print("")
        finally:
          # Exceptions are fatal. If we can recover, we should handle it in run_inner().
          interrupt_main()

  def exit_handler(self):
    assert os.getpid() == self.parent_pid
    self.python_exit = True
    self._exit_child(wait_thread=False)

  def init_epoch(self, epoch=None, seq_list=None):
    if epoch is None:
      epoch = 1
    with self.lock:
      if epoch == self.crnnEpoch and self.expected_load_seq_start == 0:
        return
      if epoch != self.crnnEpoch:
        if self._num_seqs is not None:
          self._estimated_num_seqs = self._num_seqs  # last epoch num_seqs is a good estimate
          self._num_seqs = None  # but we are not certain whether we have the same num_seqs for this epoch
      super(ExternSprintDataset, self).init_seq_order(epoch=epoch, seq_list=seq_list)
    self._exit_child(wait_thread=True)
    self._start_child(epoch)

  def init_seq_order(self, epoch=None, seq_list=None):
    self.init_epoch(epoch=epoch, seq_list=seq_list)
    return True

  @property
  def num_seqs(self):
    with self.lock:
      assert self._num_seqs is not None
      return self._num_seqs


class SprintCacheDataset(CachedDataset2):
  """
  Can directly read Sprint cache files (and bundle files).
  Supports both cached features and cached alignments.
  For alignments, you need to provide all options for the AllophoneLabeling class, such as allophone file, etc.
  """

  class SprintCacheReader(object):
    def __init__(self, data_key, filename, type=None, allophone_labeling=None):
      """
      :param str data_key: e.g. "data" or "classes"
      :param str filename: to Sprint cache archive
      :param str|None type: "feat" or "align"
      :param dict[str] allophone_labeling: kwargs for :class:`AllophoneLabeling`
      """
      self.data_key = data_key
      from SprintCache import open_file_archive
      self.sprint_cache = open_file_archive(filename)
      if not type:
        if data_key == "data":
          type = "feat"
        elif data_key == "classes":
          type = "align"
        else:
          # Some sensible defaults.
          if allophone_labeling:
            type = "align"
          else:
            type = "feat"
      assert type in ["feat", "align"]
      self.type = type
      self.allophone_labeling = None
      if allophone_labeling:
        from SprintCache import AllophoneLabeling
        self.allophone_labeling = AllophoneLabeling(**allophone_labeling)
        self.sprint_cache.setAllophones(self.allophone_labeling.allophone_file)
      else:
        assert type != "align", "need allophone_labeling for 'align' type"
      self.content_keys = [fn for fn in self.sprint_cache.file_list() if not fn.endswith(".attribs")]
      if type == "align":
        self.num_labels = self.allophone_labeling.num_labels
        if self.num_labels < 2 ** 7:
          self.dtype = "int8"
        elif self.num_labels < 2 ** 15:
          self.dtype = "int16"
        else:
          assert self.num_labels < 2 ** 31
          self.dtype = "int32"
        self.num_dims = 1
        if self.allophone_labeling.state_tying_by_allo_state_idx:
          self.type = "align_raw"
      elif type == "feat":
        self.num_labels = self._get_feature_dim()
        self.num_dims = 2
        self.dtype = "float32"
      else:
        assert False

    def _get_feature_dim(self):
      assert self.type == "feat"
      assert self.content_keys
      times, feats = self.sprint_cache.read(self.content_keys[0], "feat")
      assert len(times) == len(feats) > 0
      feat = feats[0]
      assert isinstance(feat, numpy.ndarray)
      assert feat.ndim == 1
      return feat.shape[0]

    def read(self, name):
      """
      :param str name: content-filename for sprint cache
      :return: numpy array of shape (time, [num_labels])
      :rtype: numpy.ndarray
      """
      res = self.sprint_cache.read(name, typ=self.type)
      if self.type == "align":
        label_seq = numpy.array([self.allophone_labeling.get_label_idx(a, s) for (t, a, s) in res], dtype=self.dtype)
        assert label_seq.shape == (len(res),)
        return label_seq
      elif self.type == "align_raw":
        label_seq = numpy.array([self.allophone_labeling.state_tying_by_allo_state_idx[a] for (t, a, s) in res], dtype=self.dtype)
        assert label_seq.shape == (len(res),)
        return label_seq
      elif self.type == "feat":
        times, feats = res
        assert len(times) == len(feats) > 0
        feat_mat = numpy.array(feats, dtype=self.dtype)
        assert feat_mat.shape == (len(times), self.num_labels)
        return feat_mat
      else:
        assert False

  def __init__(self, data, **kwargs):
    """
    :param dict[str,dict[str]] data: data-key -> dict which keys such as filename, see SprintCacheReader constructor
    """
    super(SprintCacheDataset, self).__init__(**kwargs)
    self.data = {key: self.SprintCacheReader(data_key=key, **opts) for (key, opts) in data.items()}
    self.seq_list_original = self.data["data"].content_keys
    self.seq_list_ordered = self.seq_list_original
    self._num_seqs = len(self.seq_list_original)
    self._check_matching_content_list()
    self.num_outputs = {key: (d.num_labels, d.num_dims) for (key, d) in self.data.items()}
    self.num_inputs = self.num_outputs["data"][0]
    self._seq_lens = None

  def _check_matching_content_list(self):
    data0 = self.data["data"]
    assert isinstance(data0, self.SprintCacheReader)
    sorted_list0 = sorted(data0.content_keys)
    for key, data in self.data.items():
      if key == "data":
        continue
      assert isinstance(data, self.SprintCacheReader)
      assert len(data.content_keys) == len(data0.content_keys)
      sorted_list1 = sorted(data.content_keys)
      for i in range(len(data.content_keys)):
        k0 = sorted_list0[i]
        k1 = sorted_list1[i]
        assert k0 == k1

  def init_seq_order(self, epoch=None, seq_list=None):
    assert not seq_list
    need_reinit = self.epoch is None or self.epoch != epoch
    super(SprintCacheDataset, self).init_seq_order(epoch=epoch, seq_list=seq_list)
    if not need_reinit:
      return False
    self._num_seqs = len(self.seq_list_original)
    data0 = self.data["data"]
    assert isinstance(data0, self.SprintCacheReader)
    get_seq_size = lambda s: data0.sprint_cache.ft[self.seq_list_original[s]].size
    seq_index = self.get_seq_order_for_epoch(epoch, self.num_seqs, get_seq_len=get_seq_size)
    self.seq_list_ordered = [self.seq_list_original[s] for s in seq_index]
    return True

  def get_dataset_seq_for_name(self, name, seq_idx=-1):
    data = {key: d.read(name) for (key, d) in self.data.items()}  # type: dict[str,numpy.ndarray]
    return DatasetSeq(seq_idx=seq_idx, seq_tag=name, features=data["data"], targets=data)

  def _collect_single_seq(self, seq_idx):
    """
    :type seq_idx: int
    :rtype: DatasetSeq | None
    :returns DatasetSeq or None if seq_idx >= num_seqs.
    """
    if seq_idx >= self.num_seqs:
      return None
    seq_tag = self.get_tag(seq_idx)  # type: str
    return self.get_dataset_seq_for_name(seq_idx=seq_idx, name=seq_tag)

  def get_data_keys(self):
    """
    :rtype: list[str]
    """
    return self.data.keys()

  def get_target_list(self):
    """
    :rtype: list[str]
    """
    return [key for (key, d) in self.data if d.type == "align"]

  def get_tag(self, sorted_seq_idx):
    """
    :rtype: str
    """
    return self.seq_list_ordered[sorted_seq_idx]


def demo():
  print("SprintDataset demo.")
  from argparse import ArgumentParser
  from Util import hms, progress_bar_with_time
  from Log import log
  from Config import Config
  from Dataset import init_dataset
  arg_parser = ArgumentParser()
  arg_parser.add_argument("--config", help="config with ExternSprintDataset", required=True)
  arg_parser.add_argument("--sprint_cache_dataset", help="kwargs dict for SprintCacheDataset", required=True)
  arg_parser.add_argument("--max_num_seqs", default=sys.maxint, type=int)
  arg_parser.add_argument("--action", default="compare", help="compare or benchmark")
  args = arg_parser.parse_args()
  log.initialize(verbosity=[4])
  sprint_cache_dataset_kwargs = eval(args.sprint_cache_dataset)
  assert isinstance(sprint_cache_dataset_kwargs, dict)
  sprint_cache_dataset = SprintCacheDataset(**sprint_cache_dataset_kwargs)
  print("SprintCacheDataset: %r" % sprint_cache_dataset)
  config = Config()
  config.load_file(args.config)
  dataset = init_dataset(config.typed_value("train"))
  print("Dataset via config: %r" % dataset)
  assert sprint_cache_dataset.num_inputs == dataset.num_inputs
  assert tuple(sprint_cache_dataset.num_outputs["classes"]) == tuple(dataset.num_outputs["classes"])
  sprint_cache_dataset.init_seq_order(epoch=1)

  if args.action == "compare":
    print("Iterating through dataset...")
    seq_idx = 0
    dataset.init_seq_order(epoch=1)
    while seq_idx < args.max_num_seqs:
      if not dataset.is_less_than_num_seqs(seq_idx):
        break
      dataset.load_seqs(seq_idx, seq_idx + 1)
      tag = dataset.get_tag(seq_idx)
      assert not tag.startswith("seq-"), "dataset does not provide tag-names for seqs"
      dataset_seq = sprint_cache_dataset.get_dataset_seq_for_name(tag)
      data = dataset.get_data(seq_idx, "data")
      targets = dataset.get_data(seq_idx, "classes")
      assert data.shape == dataset_seq.features.shape
      assert targets.shape == dataset_seq.targets["classes"].shape
      assert numpy.allclose(data, dataset_seq.features)
      assert numpy.allclose(targets, dataset_seq.targets["classes"])
      seq_idx += 1
      progress_bar_with_time(dataset.get_complete_frac(seq_idx))

    print("Finished through dataset. Num seqs: %i" % seq_idx)
    print("SprintCacheDataset has num seqs: %i." % sprint_cache_dataset.num_seqs)

  elif args.action == "benchmark":
    print("Iterating through dataset...")
    start_time = time.time()
    seq_tags = []
    seq_idx = 0
    dataset.init_seq_order(epoch=1)
    while seq_idx < args.max_num_seqs:
      if not dataset.is_less_than_num_seqs(seq_idx):
        break
      dataset.load_seqs(seq_idx, seq_idx + 1)
      tag = dataset.get_tag(seq_idx)
      assert not tag.startswith("seq-"), "dataset does not provide tag-names for seqs"
      seq_tags.append(tag)
      dataset.get_data(seq_idx, "data")
      dataset.get_data(seq_idx, "classes")
      seq_idx += 1
      progress_bar_with_time(dataset.get_complete_frac(seq_idx))
    print("Finished through dataset. Num seqs: %i, time: %f" % (seq_idx, time.time() - start_time))
    print("SprintCacheDataset has num seqs: %i." % sprint_cache_dataset.num_seqs)
    if hasattr(dataset, "exit_handler"):
      dataset.exit_handler()
    else:
      print("No way to stop any background tasks.")
    del dataset

    start_time = time.time()
    print("Iterating through SprintCacheDataset...")
    for i, tag in enumerate(seq_tags):
      sprint_cache_dataset.get_dataset_seq_for_name(tag)
      progress_bar_with_time(float(i) / len(seq_tags))
    print("Finished through SprintCacheDataset. time: %f" % (time.time() - start_time,))

  else:
    raise Exception("invalid action: %r" % args.action)


if __name__ == "__main__":
  import better_exchook
  better_exchook.install()
  try:
    demo()
  except KeyboardInterrupt:
    print("KeyboardInterrupt.")
    sys.exit(1)
