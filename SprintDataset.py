
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
import time
import numpy
import typing

import TaskSystem
from Dataset import Dataset, DatasetSeq
from CachedDataset2 import CachedDataset2
from Log import log
from TaskSystem import Unpickler, numpy_copy_and_set_unused
from Util import eval_shell_str, interrupt_main, unicode, PY3, BytesIO


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

  def __init__(self, target_maps=None, str_add_final_zero=False, input_stddev=1.,
               orth_post_process=None, bpe=None, orth_vocab=None,
               suppress_load_seqs_print=False,
               **kwargs):
    """
    :param dict[str,str|dict] target_maps: e.g. {"speaker": "speaker_map.txt"}
    :param bool str_add_final_zero: adds e.g. "orth0" with '\0'-ending
    :param float input_stddev: if != 1, will divide the input "data" by that
    :param str|list[str]|None orth_post_process: :func:`get_post_processor_function`, applied on orth
    :param None|dict[str] bpe: if given, will be opts for :class:`BytePairEncoding`
    :param None|dict[str] orth_vocab: if given, orth_vocab is applied to orth and orth_classes is an available target`
    :param bool suppress_load_seqs_print: less verbose
    """
    super(SprintDatasetBase, self).__init__(**kwargs)
    self.suppress_load_seqs_print = suppress_load_seqs_print
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
    self.labels["orth"] = [chr(i) for i in range(255)]
    self.orth_post_process = None
    if orth_post_process:
      from LmDataset import get_post_processor_function
      self.orth_post_process = get_post_processor_function(orth_post_process)
    self.bpe = None
    if bpe:
      from GeneratingDataset import BytePairEncoding
      self.bpe = BytePairEncoding(**bpe)
      self.labels["bpe"] = self.bpe.labels
    self.orth_vocab = None
    if orth_vocab:
      assert not bpe, "bpe has its own vocab"
      from GeneratingDataset import Vocabulary
      self.orth_vocab = Vocabulary(**orth_vocab)
      self.labels["orth_classes"] = self.orth_vocab.labels
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
    self._reset_cache()
    assert self.shuffle_frames_of_nseqs == 0  # Currently broken. But just use Sprint itself to do this.

  def use_multiple_epochs(self):
    """
    Called via SprintInterface.getSegmentList().
    """
    self.multiple_epochs = True

  def set_dimensions(self, input_dim, output_dim):
    """
    :type input_dim: int
    :type output_dim: int

    Called via python_train.
    """
    assert input_dim > 0
    self.num_inputs = input_dim
    self.num_outputs = {"data": (input_dim * self.window, 2)}
    if output_dim > 0:
      self.num_outputs["classes"] = (output_dim, 1)
    if self.bpe:
      self.num_outputs["bpe"] = (self.bpe.num_labels, 1)
    if self.orth_vocab:
      self.num_outputs["orth_classes"] = (self.orth_vocab.num_labels, 1)
    self.num_outputs["orth"] = (256, 1)
    self._base_init()
    # At this point, we are ready for data. In case we don't use the Sprint PythonSegmentOrdering
    # (SprintInterface.getSegmentList()), we must call this at least once.
    if not self.multiple_epochs:
      self.init_sprint_epoch(None)

  def _reset_cache(self):
    self.expected_load_seq_start = 0
    self.requested_load_seq_end = 0
    self.next_seq_to_be_added = 0
    self.reached_final_seq = False
    self.reached_final_seq_seen_all = False
    self._num_timesteps = 0
    self.added_data = []  # type: typing.List[DatasetSeq]
    self.ready_for_data = True

  def init_sprint_epoch(self, epoch):
    """
    :type epoch: int | None
    Called by SprintInterface.getSegmentList() when we start a new epoch.
    We must not call this via self.init_seq_order() because we will already have filled the cache by Sprint
    before the CRNN train thread starts the epoch.
    """
    with self.lock:
      self.sprintEpoch = epoch
      self.sprintFinalized = False
      self._reset_cache()
      self.cond.notify_all()

  def finalize_sprint(self):
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

  def _cleanup_old_seq_cache(self, seq_end):
    i = 0
    while i < len(self.added_data):
      if self.added_data[i].seq_idx >= seq_end:
        break
      i += 1
    del self.added_data[:i]

  def wait_for_returnn_epoch(self, epoch):
    """
    Called by SprintInterface.
    """
    with self.lock:
      while epoch != self.crnnEpoch:
        assert epoch > self.crnnEpoch
        self.cond.wait()

  def _wait_for_seq_can_pass_check(self, seq_start, seq_end):
    """
    :param int seq_start:
    :param int seq_end:
    :return: True if _waitForSeq can pass/return. False means that we need to wait more (until next signal)
    :rtype: bool
    """
    if self.reached_final_seq:
      return True
    if self._have_seqs_added(seq_start, seq_end):
      return True
    return False

  def _wait_for_seq(self, seq_start, seq_end=None):
    """
    Called by RETURNN train thread.
    Wait until we have seqs [seqStart,..,seqEnd-1] loaded,
    or we now that they will not be loaded anymore because we reached the end.

    :param int seq_start:
    :param int|None seq_end:
    """
    if seq_end is None:
      seq_end = seq_start + 1
    if seq_end > self.requested_load_seq_end:
      self.requested_load_seq_end = seq_end
      self.cond.notify_all()
    if self._wait_for_seq_can_pass_check(seq_start=seq_start, seq_end=seq_end):
      return
    # We need to wait.
    assert thread.get_ident() != self.add_data_thread_id
    print("%s %s: wait for seqs (%i,%i) (last added: %s) (current time: %s)" % (
      self, currentThread().name, seq_start, seq_end, self._latest_added_seq(), time.strftime("%H:%M:%S")), file=log.v5)
    while not self._wait_for_seq_can_pass_check(seq_start=seq_start, seq_end=seq_end):
      self.cond.wait()

  def _latest_added_seq(self):
    if not self.added_data:
      return None
    return self.added_data[-1].seq_idx

  def _have_seqs_added(self, start, end=None):
    if end is None:
      end = start + 1
    if start >= end:
      return True
    for data in self.added_data:
      assert start >= data.seq_idx, "%s: We expect that we only ask about the cache of the upcoming seqs." % self
      if data.seq_idx == start:
        start += 1
      if start >= end:
        return True
    return False

  def _get_seq(self, seq_idx):
    """
    :param int seq_idx:
    :rtype: DatasetSeq
    """
    for data in self.added_data:
      if data.seq_idx == seq_idx:
        return data
    return None

  def is_cached(self, start, end):
    """
    :param int start:
    :param int end:
    :rtype: bool
    """
    # Always False, to force that we call self._load_seqs().
    # This is important for our buffer management.
    return False

  def load_seqs(self, start, end):
    """
    Called by RETURNN train thread.

    :param int start:
    :param int end:
    """
    if start == end:
      return
    if not self.suppress_load_seqs_print:
      print("%s load_seqs in %s:" % (self, currentThread().name), start, end, end=' ', file=log.v5)
    with self.lock:
      super(SprintDatasetBase, self).load_seqs(start, end)
      if not self.suppress_load_seqs_print:
        print("first features shape:", self._get_seq(start).features["data"].shape, file=log.v5)

  def _load_seqs(self, start, end):
    """
    Called by RETURNN train thread.
    We expect that start increase monotonic on each call
    for not-yet-loaded data.
    This will already be called with _load_seqs_superset indices.

    :param int start:
    :param int end:
    """
    assert start >= self.expected_load_seq_start
    if start > self.expected_load_seq_start:
      # Cleanup old data.
      self._cleanup_old_seq_cache(start)
      self.expected_load_seq_start = start
      self.cond.notify_all()
    self._wait_for_seq(start, end)

  def add_new_data(self, features, targets=None, segment_name=None):
    """
    Adds a new seq.
    This is called via the Sprint main thread.

    :param numpy.ndarray features: format (input-feature,time) (via Sprint)
    :param dict[str,numpy.ndarray|str] targets: format (time) (idx of output-feature)
    :param str|None segment_name:
    :returns the sorted seq index
    :rtype: int
    """

    # is in format (feature,time)
    assert self.num_inputs == features.shape[0]
    num_frames = features.shape[1]
    # must be in format: (time,feature)
    features = features.transpose()
    assert features.shape == (num_frames, self.num_inputs)
    if self.input_stddev != 1:
      features /= self.input_stddev
    if self.window > 1:
      features = self.sliding_window(features)
      assert features.shape == (num_frames, self.num_inputs * self.window)

    if targets is None:
      targets = {}
    if not isinstance(targets, dict):
      targets = {"classes": targets}
    if "classes" in targets:
      # 'classes' is always the alignment
      assert targets["classes"].shape == (num_frames,), (  # is in format (time,)
        "Number of targets %s does not equal to number of features %s" % (targets["classes"].shape, (num_frames,)))
    if "orth" in targets:
      targets["orth"] = targets["orth"].decode("utf8").strip()
    if "orth" in targets and self.orth_post_process:
      targets["orth"] = self.orth_post_process(targets["orth"])
    if self.bpe:
      assert "orth" in targets
      orth = targets["orth"]
      assert isinstance(orth, (str, unicode))
      assert "bpe" not in targets
      targets["bpe"] = numpy.array(self.bpe.get_seq(orth), dtype="int32")
    if self.orth_vocab:
      assert not self.orth_post_process
      assert "orth" in targets
      orth = targets["orth"]
      assert isinstance(orth, (str, unicode))
      assert "orth_classes" not in targets
      targets["orth_classes"] = numpy.array(self.orth_vocab.get_seq(orth), dtype="int32")

    # Maybe convert some targets.
    if self.target_maps:
      for key, target_map in self.target_maps.items():
        assert key in targets
        v = target_map[targets[key]]
        v = numpy.asarray(v)
        if v.ndim == 0:
          v = numpy.zeros((num_frames,), dtype=v.dtype) + v  # add time dimension
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
        if PY3:
          assert isinstance(v, bytes)
          v = list(v)
        else:
          v = list(map(ord, v))
        v = numpy.array(v, dtype="uint8")
        targets[key] = v
        if self.str_add_final_zero:
          v = numpy.append(v, numpy.array([0], dtype=v.dtype))
          assert key + "0" not in targets
          targets[key + "0"] = v
        continue
      print("%s, we will ignore the target %r because it is not a numpy array: %r" % (self, key, v), file=log.v3)
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
        assert seq_idx < len(self.predefined_seq_list_order), "seq_idx %i, expected predef num seqs %i" % (
          seq_idx, len(self.predefined_seq_list_order))
        expected_seq_name = self.predefined_seq_list_order[seq_idx]
        if expected_seq_name != segment_name:
          if segment_name in self.predefined_seq_list_order:
            raise Exception("seq_idx %i expected to be tag %r but got tag %r; tag %r is at idx %i" % (
              seq_idx, expected_seq_name, segment_name, segment_name,
              self.predefined_seq_list_order.index(segment_name)))
          raise Exception("seq_idx %i expected to be tag %r but got tag %r; tag %r not found" % (
            seq_idx, expected_seq_name, segment_name, segment_name))

      self.next_seq_to_be_added += 1
      self._num_timesteps += num_frames
      self.cond.notify_all()

      if seq_idx > self.requested_load_seq_end - 1 + self.SprintCachedSeqsMax:
        print("%s add_new_data: seq=%i, len=%i. Cache filled, waiting to get loaded..." % (
          self, seq_idx, num_frames), file=log.v5)
        while seq_idx > self.requested_load_seq_end - 1 + self.SprintCachedSeqsMin:
          assert not self.reached_final_seq
          assert seq_idx + 1 == self.next_seq_to_be_added
          self.cond.wait()

      self.added_data += [DatasetSeq(seq_idx, features, targets, seq_tag=segment_name)]
      self.cond.notify_all()
      return seq_idx

  def finish_sprint_epoch(self, seen_all=True):
    """
    Called by SprintInterface.getSegmentList().
    This is in a state where Sprint asks for the next segment after we just finished an epoch.
    Thus, any upcoming self.add_new_data() call will contain data from a segment in the new epoch.
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
    """
    :rtype: int
    """
    with self.lock:
      assert self.reached_final_seq
      return self._num_timesteps

  @property
  def num_seqs(self):
    """
    :rtype: int
    """
    with self.lock:
      if self.predefined_seq_list_order:
        return len(self.predefined_seq_list_order)
      if not self.reached_final_seq:
        raise NotImplementedError
      return self.next_seq_to_be_added

  def have_seqs(self):
    """
    :rtype: bool
    """
    with self.lock:
      if self.next_seq_to_be_added > 0:
        return True
      self._wait_for_seq(0)
      return self.next_seq_to_be_added > 0

  def is_less_than_num_seqs(self, n):
    """
    :param int n:
    :rtype: bool
    """
    with self.lock:
      self._wait_for_seq(n)
      return n < self.next_seq_to_be_added

  def get_data_keys(self):
    """
    :rtype: list[str]
    """
    with self.lock:
      if not self.added_data:
        self._wait_for_seq(0)
      assert self.added_data
      return sorted(self.added_data[0].features.keys())

  def get_target_list(self):
    """
    :rtype: list[str]
    """
    keys = list(self.get_data_keys())
    if "data" in keys:
      keys.remove("data")
    return keys

  def set_complete_frac(self, frac):
    """
    :param float frac:
    """
    self._complete_frac = frac

  def get_complete_frac(self, seq_idx):
    """
    :param int seq_idx:
    :rtype: float
    """
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
    """
    :param int sorted_seq_idx:
    :rtype: Util.NumbersDict
    """
    with self.lock:
      self._wait_for_seq(sorted_seq_idx)
      return self._get_seq(sorted_seq_idx).num_frames

  def get_data(self, seq_idx, key):
    """
    :param int seq_idx:
    :param str key:
    :rtype: numpy.ndarray
    """
    with self.lock:
      self._wait_for_seq(seq_idx)
      return self._get_seq(seq_idx).features[key]

  def get_input_data(self, sorted_seq_idx):
    """
    :param int sorted_seq_idx:
    :rtype: numpy.ndarray
    """
    with self.lock:
      self._wait_for_seq(sorted_seq_idx)
      return self._get_seq(sorted_seq_idx).features["data"]

  def get_targets(self, target, sorted_seq_idx):
    """
    :param str target:
    :param int sorted_seq_idx:
    :rtype: numpy.ndarray
    """
    with self.lock:
      self._wait_for_seq(sorted_seq_idx)
      return self._get_seq(sorted_seq_idx).features.get(target, None)

  def get_ctc_targets(self, sorted_seq_idx):
    """
    :param int sorted_seq_idx:
    """
    assert False, "No CTC targets."

  def get_tag(self, sorted_seq_idx):
    """
    :param int sorted_seq_idx:
    :rtype: str
    """
    with self.lock:
      self._wait_for_seq(sorted_seq_idx)
      return self._get_seq(sorted_seq_idx).seq_tag


class ExternSprintDataset(SprintDatasetBase):
  """
  This is a Dataset which you can use directly in RETURNN.
  You can use it to get any type of data from Sprint (RWTH ASR toolkit),
  e.g. you can use Sprint to do feature extraction and preprocessing.

  This class is like SprintDatasetBase, except that we will start an external Sprint instance ourselves
  which will forward the data to us over a pipe.
  The Sprint subprocess will use SprintExternInterface to communicate with us.
  """

  # Do not change the argument names here, to not break existing configs.
  # noinspection PyPep8Naming
  def __init__(self, sprintTrainerExecPath, sprintConfigStr, partitionEpoch=None, **kwargs):
    """
    :param str|list[str] sprintTrainerExecPath:
    :param str | list[str] | ()->str | list[()->str] | ()->list[str] | ()->list[()->str] sprintConfigStr:
      via eval_shell_str
    :param int|None partitionEpoch: deprecated. use partition_epoch instead
    """
    super(ExternSprintDataset, self).__init__(**kwargs)
    self.add_data_thread_id = None
    self.sprint_trainer_exec_path = sprintTrainerExecPath
    self.sprint_config = sprintConfigStr
    if partitionEpoch:
      assert self.partition_epoch == 1, "don't provide partitionEpoch and partition_epoch"
      self.partition_epoch = partitionEpoch
    self._num_seqs = None
    self.child_pid = None  # type: typing.Optional[int]
    self.parent_pid = os.getpid()
    self.reader_thread = None  # type: typing.Optional[Thread]
    self.seq_list_file = None
    self.use_multiple_epochs()
    # There is no generic way to see whether Python is exiting.
    # This is our workaround. We check for it in self.run_inner().
    self.python_exit = False
    atexit.register(self._exit_handler)
    self.init_seq_order()

  def _exit_child(self, wait_thread=True):
    """
    :param bool wait_thread:
    """
    if self.child_pid:
      expected_exit_status = 0 if not self.python_exit else None
      if self._join_child(wait=False, expected_exit_status=expected_exit_status) is False:  # Not yet terminated.
        interrupt = not self.reached_final_seq_seen_all
        if interrupt:
          print("%s: interrupt child proc %s" % (self, self.child_pid), file=log.v5)
          os.kill(self.child_pid, signal.SIGKILL)
          # Also join such that the process is cleaned up, and pipes get closed.
          self._join_child(wait=True, expected_exit_status=None)
          self.child_pid = None
      else:  # child process terminated
        self.child_pid = None
      if wait_thread:
        # Load all remaining data so that the reader thread is not waiting in self.add_new_data().
        while self.is_less_than_num_seqs(self.expected_load_seq_start + 1):
          if self.reached_final_seq:  # this is set by the reader thread
            break
          self.load_seqs(self.expected_load_seq_start + 1, self.expected_load_seq_start + 2)
        self.reader_thread.join()
        self.reader_thread = None
      try:
        self.pipe_p2c[1].close()
      except IOError:
        pass
      try:
        self.pipe_c2p[0].close()
      except IOError:
        pass
      if self.child_pid:
        self._join_child(wait=True, expected_exit_status=0)
        self.child_pid = None

  def _start_child(self, epoch):
    """
    :param epoch:
    :return:
    """
    assert self.child_pid is None
    assert self.reader_thread is None
    self.pipe_c2p = self._pipe_open()
    self.pipe_p2c = self._pipe_open()
    args = self._build_sprint_args()
    print("%s: epoch" % self, epoch, "exec", args, file=log.v5)

    pid = os.fork()
    if pid == 0:  # child
      # In case we are in some test environment or so, recover the original stdout/stderr.
      sys.stdin = sys.__stdin__
      sys.stdout = sys.__stdout__
      sys.stderr = sys.__stderr__
      import better_exchook
      better_exchook.install()
      # noinspection PyBroadException
      try:
        sys.stdin.close()  # Force no tty stdin.
        self.pipe_c2p[0].close()
        self.pipe_p2c[1].close()
        os.execv(args[0], args)  # Does not return if successful.
        print("%s child exec failed." % self)
      except BaseException:
        print("%s child: Error when starting Sprint %r." % (self, args))
        sys.excepthook(*sys.exc_info())
      finally:
        print("%s child: exit" % self)
        # noinspection PyProtectedMember
        os._exit(1)
        return  # Not reached.

    # parent
    self.pipe_c2p[1].close()
    self.pipe_p2c[0].close()
    self.child_pid = pid

    try:
      init_signal, (input_dim, output_dim, num_segments) = self._read_next_raw()
      assert init_signal == b"init"
      assert isinstance(input_dim, int) and isinstance(output_dim, int)
      # Ignore num_segments. It can be totally different than the real number of sequences.
      self.set_dimensions(input_dim, output_dim)
    except Exception:
      print("%s: Sprint child process (%r) caused an exception." % (self, args), file=log.v1)
      sys.excepthook(*sys.exc_info())
      self._exit_child(wait_thread=False)
      raise Exception("%s Sprint init failed" % self)

    self.reader_thread = Thread(target=self._reader_thread_proc, args=(pid, epoch,),
                                name="%s reader thread" % self)
    self.reader_thread.daemon = True
    self.reader_thread.start()

  # noinspection PyMethodMayBeStatic
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
    """
    :rtype: str
    """
    return os.path.dirname(os.path.abspath(__file__))

  def _build_sprint_args(self):
    """
    :rtype: list[str]
    """
    config_str = "action:ExternSprintDataset,c2p_fd:%i,p2c_fd:%i" % (
      self.pipe_c2p[1].fileno(), self.pipe_p2c[0].fileno())
    if TaskSystem.SharedMemNumpyConfig["enabled"]:
      config_str += ",EnableAutoNumpySharedMemPickling:True"
    epoch = self.crnnEpoch or 1
    assert epoch >= 1
    if isinstance(self.sprint_trainer_exec_path, (list, tuple)):
      args = list(self.sprint_trainer_exec_path)
    else:
      args = [self.sprint_trainer_exec_path]
    # First the user options. Usually also involves loading some config.
    args += eval_shell_str(self.sprint_config)
    # Now our options. They might overwrite some of the config settings. (That is why we do it after the user opts.)
    args += [
      "--*.seed=%i" % ((epoch - 1) // self.partition_epoch)]
    if self.partition_epoch > 1:
      args += [
        "--*.corpus.partition=%i" % self.partition_epoch,
        "--*.corpus.select-partition=%i" % ((epoch - 1) % self.partition_epoch)]
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
      args += [
        "--*.corpus.segment-order-shuffle=false",
        "--*.corpus.segments.file=%s" % self.seq_list_file,
        "--*.corpus.segment-order=%s" % self.seq_list_file]
    return args

  def _read_next_raw(self):
    """
    :return: (data_type, args)
    :rtype: (str, object)
    """
    import struct
    size_raw = self.pipe_c2p[0].read(4)
    if len(size_raw) < 4:
      raise EOFError
    size, = struct.unpack("<i", size_raw)
    assert size > 0, "%s: We expect to get some non-empty package. Invalid Python mod in Sprint?" % (self,)
    stream = BytesIO()
    read_size = 0
    while read_size < size:
      data_raw = self.pipe_c2p[0].read(size - read_size)
      if len(data_raw) == 0:
        raise EOFError("%s: expected to read %i bytes but got EOF after %i bytes" % (self, size, read_size))
      read_size += len(data_raw)
      stream.write(data_raw)
    stream.seek(0)
    try:
      if PY3:
        # encoding is for converting Python2 strings to Python3.
        # Cannot use utf8 because Numpy will also encode the data as strings and there we need it as bytes.
        data_type, args = Unpickler(stream, encoding="bytes").load()
      else:
        data_type, args = Unpickler(stream).load()
    except EOFError:
      raise Exception("%s: parse error of %i bytes (%r)" % (self, size, stream.getvalue()))
    return data_type, args

  def _join_child(self, wait=True, expected_exit_status=None):
    """
    :param bool wait:
    :param int|None expected_exit_status:
    :return: whether the child has exited now
    :rtype: bool
    """
    assert self.child_pid
    options = 0 if wait else os.WNOHANG
    pid, exit_status = os.waitpid(self.child_pid, options)
    if not wait and pid == 0:
      return False
    assert pid == self.child_pid
    if expected_exit_status is not None:
      assert exit_status == expected_exit_status, "%s: Sprint exit code is %i" % (self, exit_status)
    return True

  def _reader_thread_proc(self, child_pid, epoch):
    """
    :param int child_pid:
    :param int epoch:
    """
    try:
      self.add_data_thread_id = thread.get_ident()

      self.init_sprint_epoch(epoch)
      have_seen_the_whole = False

      seq_count = 0
      while not self.python_exit and self.child_pid:
        try:
          data_type, args = self._read_next_raw()
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

          if data_type == b"data":
            seq_count += 1
            segment_name, features, targets = args
            if segment_name is not None:
              segment_name = segment_name.decode("utf8")
            assert isinstance(features, numpy.ndarray)
            if isinstance(targets, dict):
              targets = {key.decode("utf8"): value for (key, value) in targets.items()}
            self.add_new_data(
              numpy_copy_and_set_unused(features),
              numpy_copy_and_set_unused(targets),
              segment_name=segment_name)
          elif data_type == b"exit":
            have_seen_the_whole = True
            break
          else:
            assert False, "not handled: (%r, %r)" % (data_type, args)

      if self.seq_list_file:
        try:
          os.remove(self.seq_list_file)
        except Exception as e:
          print("%s: error when removing %r: %r" % (self, self.seq_list_file, e), file=log.v5)
        finally:
          self.seq_list_file = None

      if not self.python_exit:
        with self.lock:
          self.finish_sprint_epoch(seen_all=have_seen_the_whole)
          if have_seen_the_whole:
            self._num_seqs = self.next_seq_to_be_added
      print("%s (proc %i) finished reading epoch %i, seen all %r (finished), num seqs %i" % (
        self, child_pid, epoch, have_seen_the_whole, seq_count), file=log.v5)

    except Exception as exc:
      if not self.python_exit:
        # Catch all standard exceptions.
        # Don't catch KeyboardInterrupt here because that will get send by the main thread
        # when it is exiting. It's never by the user because SIGINT will always
        # trigger KeyboardInterrupt in the main thread only.
        if epoch == self.crnnEpoch:
          with self.lock:
            self.finish_sprint_epoch(seen_all=False)
        try:
          print("%s reader failed (%s)" % (self, exc), file=log.v1)
          sys.excepthook(*sys.exc_info())
          print("")
        finally:
          # Exceptions are fatal. If we can recover, we should handle it in run_inner().
          interrupt_main()

  def _exit_handler(self):
    """
    Called at exit.
    """
    assert os.getpid() == self.parent_pid
    self.python_exit = True
    self._exit_child(wait_thread=False)

  def init_seq_order(self, epoch=None, seq_list=None):
    """
    :param int epoch:
    :param list[str]|None seq_list:
    :rtype: bool
    """
    if seq_list:
      assert self.partition_epoch == 1, "specifying partition_epoch and using seq_list not supported"
    if epoch is None:
      epoch = 1
    with self.lock:
      if epoch == self.crnnEpoch and self.expected_load_seq_start == 0 and seq_list == self.predefined_seq_list_order:
        return
      # Reset epoch such that exiting the child will go smoothly.
      super(ExternSprintDataset, self).init_seq_order(epoch=None, seq_list=None)
    # Exit child, before we overwrite anything, such as new epoch or seq_list.
    self._exit_child(wait_thread=True)
    with self.lock:  # Lock should not be needed now, but just to make it clean.
      if self._num_seqs:
        self._estimated_num_seqs = self._num_seqs  # last epoch num_seqs is a good estimate
      self._num_seqs = None  # we are not certain whether we have the same num_seqs for this epoch
      super(ExternSprintDataset, self).init_seq_order(epoch=epoch, seq_list=seq_list)
    self._start_child(epoch)
    return True


class SprintCacheDataset(CachedDataset2):
  """
  Can directly read Sprint cache files (and bundle files).
  Supports both cached features and cached alignments.
  For alignments, you need to provide all options for the AllophoneLabeling class, such as allophone file, etc.
  """

  class SprintCacheReader(object):
    """
    Helper class to read a Sprint cache directly.
    """
    def __init__(self, data_key, filename, data_type=None, allophone_labeling=None):
      """
      :param str data_key: e.g. "data" or "classes"
      :param str filename: to Sprint cache archive
      :param str|None data_type: "feat" or "align"
      :param dict[str] allophone_labeling: kwargs for :class:`AllophoneLabeling`
      """
      self.data_key = data_key
      from SprintCache import open_file_archive
      self.sprint_cache = open_file_archive(filename)
      if not data_type:
        if data_key == "data":
          data_type = "feat"
        elif data_key == "classes":
          data_type = "align"
        else:
          # Some sensible defaults.
          if allophone_labeling:
            data_type = "align"
          else:
            data_type = "feat"
      assert data_type in ["feat", "align"]
      self.type = data_type
      self.allophone_labeling = None
      if allophone_labeling:
        from SprintCache import AllophoneLabeling
        self.allophone_labeling = AllophoneLabeling(**allophone_labeling)
        self.sprint_cache.set_allophones(self.allophone_labeling.allophone_file)
      else:
        assert data_type != "align", "need allophone_labeling for 'align' type"
      self.content_keys = [fn for fn in self.sprint_cache.file_list() if not fn.endswith(".attribs")]
      if data_type == "align":
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
      elif data_type == "feat":
        self.num_labels = self._get_feature_dim()
        self.num_dims = 2
        self.dtype = "float32"
      else:
        assert False

    def _get_feature_dim(self):
      """
      :rtype: int
      """
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
        label_seq = numpy.array(
          [self.allophone_labeling.state_tying_by_allo_state_idx[a] for (t, a, s) in res], dtype=self.dtype)
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
    """
    :param int epoch:
    :param list[str]|None seq_list:
    :rtype: bool
    """
    assert not seq_list
    need_reinit = self.epoch is None or self.epoch != epoch
    super(SprintCacheDataset, self).init_seq_order(epoch=epoch, seq_list=seq_list)
    if not need_reinit:
      return False
    self._num_seqs = len(self.seq_list_original)
    data0 = self.data["data"]
    assert isinstance(data0, self.SprintCacheReader)

    def get_seq_size(s):
      """
      :param int s:
      :rtype: int
      """
      return data0.sprint_cache.ft[self.seq_list_original[s]].size
    seq_index = self.get_seq_order_for_epoch(epoch, self.num_seqs, get_seq_len=get_seq_size)
    self.seq_list_ordered = [self.seq_list_original[s] for s in seq_index]
    return True

  def get_dataset_seq_for_name(self, name, seq_idx=-1):
    """
    :param str name:
    :param int seq_idx:
    :rtype: DatasetSeq
    """
    data = {key: d.read(name) for (key, d) in self.data.items()}  # type: typing.Dict[str,numpy.ndarray]
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
  """
  Demo.
  """
  print("SprintDataset demo.")
  from argparse import ArgumentParser
  from Util import progress_bar_with_time
  from Log import log
  from Config import Config
  from Dataset import init_dataset
  arg_parser = ArgumentParser()
  arg_parser.add_argument("--config", help="config with ExternSprintDataset", required=True)
  arg_parser.add_argument("--sprint_cache_dataset", help="kwargs dict for SprintCacheDataset", required=True)
  arg_parser.add_argument("--max_num_seqs", default=sys.maxsize, type=int)
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
      assert data.shape == dataset_seq.features["data"].shape
      assert targets.shape == dataset_seq.features["classes"].shape
      assert numpy.allclose(data, dataset_seq.features["data"])
      assert numpy.allclose(targets, dataset_seq.features["classes"])
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
