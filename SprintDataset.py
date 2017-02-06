
"""
Implements the SprintDataset class.
Note that from the main RETURNN process, you probably want ExternSprintDataset instead.
"""

import thread
from threading import Condition, currentThread
import math
import time
import numpy

from Dataset import Dataset, DatasetSeq
from Log import log


class SprintDataset(Dataset):
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

  def __init__(self, window=1, target_maps=None, **kwargs):
    assert window == 1
    super(SprintDataset, self).__init__(**kwargs)
    if target_maps:
      assert isinstance(target_maps, dict)
      target_maps = target_maps.copy()
      for key, tmap in list(target_maps.items()):
        if isinstance(tmap, (str, unicode)):
          tmap = {l: i for (i, l) in enumerate(open(tmap).read().splitlines())}
        assert isinstance(tmap, dict)  # dict[str,int]
        target_maps[key] = tmap
    self.target_maps = target_maps
    self.cond = Condition(lock=self.lock)
    self.add_data_thread_id = thread.get_ident()  # This will be created in the Sprint thread.
    self.ready_for_data = False
    self.reached_final_seq = False
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
    self.num_outputs = {"data": [inputDim, 2]}
    if outputDim > 0:
      self.num_outputs["classes"] = [outputDim, 1]
    # At this point, we are ready for data. In case we don't use the Sprint PythonSegmentOrdering
    # (SprintInterface.getSegmentList()), we must call this at least once.
    if not self.multiple_epochs:
      self.initSprintEpoch(None)

  def _resetCache(self):
    self.expected_load_seq_start = 0
    self.requested_load_seq_end = 0
    self.next_seq_to_be_added = 0
    self.reached_final_seq = False
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
    super(SprintDataset, self).init_seq_order(epoch=epoch, seq_list=seq_list)
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
    print >> log.v5, "SprintDataset wait for seqs (%i,%i) (last added: %s) (current time: %s)" % \
                     (seqStart, seqEnd, self._latestAddedSeq(), time.strftime("%H:%M:%S"))
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
    print >> log.v5, "SprintDataset load_seqs in %s:" % currentThread().name, start, end,
    if start == end: return
    with self.lock:
      super(SprintDataset, self).load_seqs(start, end)
      print >> log.v5, "first features shape:", self._getSeq(start).features.shape

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

    if targets is None:
      targets = {}
    if not isinstance(targets, dict):
      targets = {"classes": targets}
    if "classes" in targets:
      # 'classes' is always the alignment
      assert targets["classes"].shape == (T,), "Number of targets %s does not equal to number of features %s" % (targets["classes"].shape, (T,))  # is in format (time,)

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
        v = map(ord, v)
        v = numpy.array(v, dtype="uint8")
        targets[key] = v
        continue
      print >> log.v3, "SprintDataset, we will ignore the target %r because it is not a numpy array: %r" % (key, v)
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
        print >> log.v5, "SprintDataset addNewData: seq=%i, len=%i. Cache filled, waiting to get loaded..." % (seq_idx, T)
        while seq_idx > self.requested_load_seq_end - 1 + self.SprintCachedSeqsMin:
          assert not self.reached_final_seq
          assert seq_idx + 1 == self.next_seq_to_be_added
          self.cond.wait()

      self.added_data += [DatasetSeq(seq_idx, features, targets, seq_tag=segmentName)]
      self.cond.notify_all()
      return seq_idx

  def finishSprintEpoch(self):
    """
    Called by SprintInterface.getSegmentList().
    This is in a state where Sprint asks for the next segment after we just finished an epoch.
    Thus, any upcoming self.addNewData() call will contain data from a segment in the new epoch.
    Thus, we finish the current epoch in Sprint.
    """
    with self.lock:
      self.reached_final_seq = True
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
        return super(SprintDataset, self).get_complete_frac(seq_idx)

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
