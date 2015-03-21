
import thread
from threading import RLock, Condition, currentThread
from Dataset import Dataset
from Log import log


class SprintDataset(Dataset):

  # To emulate, we must set:
  #   num_seqs (for Engine.set_batch_size())
  #   seq_index (see Dataset.set_batching())
  #   seq_start
  #   seq_lengths
  #   alloc_intervals

  SprintCachedSeqsMax = 10
  SprintCachedSeqsMin = 5

  def __init__(self, window=1, cache_size=0, chunking="0", batching='default'):
    assert window == 1
    super(SprintDataset, self).__init__(window, cache_size, chunking, batching)
    self.lock = RLock()
    self.cond = Condition(lock=self.lock)
    self.finalized = False
    self.main_thread_id = thread.get_ident()
    self.add_data_thread_id = -1
    self.epoch = 0

  def initFromSegmentOrder(self, segmentList, segmentsInfo):
    """
    :type segmentList: list[str]
    :type segmentsInfo: dict[str,dict[str,object]]
    Called via segment_order.
    """
    assert segmentList, "expect a non-empty segment list"
    self.num_seqs = len(segmentList)
    self.segmentList = list(segmentList)
    self.segmentsInfo = segmentsInfo
    self.seq_index = list(range(0, self.num_seqs))  # sorted idx -> real idx
    self.seq_index_reversed = list(self.seq_index)
    for i, segName in enumerate(segmentList):
      segmentsInfo[segName]["idx"] = i
    self.seq_lengths = \
      [segmentsInfo[segName]["nframes"] for segName in segmentList]
    self.num_timesteps = sum(self.seq_lengths)
    assert self.num_timesteps > 0, "expect some frames"

  def setDimensions(self, inputDim, outputDim):
    """
    :type inputDim: int
    :type outputDim: int
    Called via python_train.
    """
    assert inputDim > 0 and outputDim > 0
    self.num_inputs = inputDim
    self.num_outputs = outputDim

  def initialize(self):
    """
    Called via SprintInterface.init().
    """
    if self.num_seqs == 0:
      # We have not called initFromSegmentOrder() yet.
      # Not an error though, because this gets called early via self.load_data().
      # Just ignore.
      return
    assert self.segmentList, "call initFromSegmentOrder()"
    assert self.num_inputs > 0 and self.num_outputs > 0, "call setDimensions()"
    super(SprintDataset, self).initialize()

  def getSegmentList(self):
    """
    Returns segmentList in current seq_index order.
    Called via segment_order.
    """
    return [self.segmentList[i] for i in self.seq_index]

  def load_seqs(self, start, end, free=True, fill=True):
    print >> log.v5, "SprintDataset load_seqs in %s:" % currentThread().name, start, end, free, fill
    if start == end: return
    if thread.get_ident() == self.main_thread_id:
      print >> log.v5, "SprintDataset load_seqs: ignore from main thread"
      return
    with self.lock:
      assert self.alloc_intervals  # Must be initialized.
      # We expect that start/end increase monotonic on each call.
      assert start >= self.expected_load_seq_start
      if start > self.expected_load_seq_start:
        # Cleanup old data.
        self.remove_alloc_interval(0, start)
        self.expected_load_seq_start = start
      assert end >= self.requested_load_seq_end
      self.requested_load_seq_end = end
      self.cond.notify_all()
      if not self._haveSeqsAdded(start, end):
        print >> log.v5, "SprintDataset load_seqs: have not seqs. waiting for addNewData..."
        assert self.add_data_thread_id != thread.get_ident()
        while not self._haveSeqsAdded(start, end):
          assert not self.finalized
          assert not self.seq_added_excluded.intersection(range(start, end)), "Excluded seqs are in this seq range."
          assert end - 1 > self.seq_added_last
          self.cond.wait()

  def have_seqs(self, start, end):
    """
    :param int start: start sorted seq idx
    :param int end: end sorted seq idx
    :returns whether this dataset includes the seq range.
    A call to self.load_seqs() must succeed if we return True.
    """
    with self.lock:
      if not super(SprintDataset, self).have_seqs(start, end):
        return False
      # Otherwise we cannot tell. Sprint could skip segments for whatever reason,
      # e.g. the buffer in BufferFeatureExtractor is too small for a segment.
      assert end >= self.requested_load_seq_end
      self.requested_load_seq_end = end
      self.cond.notify_all()
      if self.seq_added_last < end - 1:
        print "SprintDataset have_seqs: wait for addNewData..."
        assert self.add_data_thread_id != thread.get_ident()
        while self.seq_added_last < end - 1:
          assert not self.finalized
          self.cond.wait()
      return self._haveSeqsAdded(start, end)

  def _haveSeqsAdded(self, start, end=None):
    if end is None: end = start + 1
    return self.is_cached(start, end)

  def finalize(self):
    # Called by segment_order.
    with self.lock:
      self.finalized = True
      self.cond.notify_all()

  def init_seq_order(self, epoch=None):
    print >> log.v5, "SprintDataset init_seq_order in %s:" % currentThread().name, epoch
    # Sprint can iterate over the segment list before it called the trainer init.
    # The trainer will also control the sequence order itself.
    # Avoid a reinit for the same epoch. This is why we override this method.
    with self.lock:
      if epoch is not None and epoch == self.epoch:
        return  # Ignore.
      if epoch is None:
        # Don't reorder. Just initialize the remaining stuff.
        self.init_seqs()
      else:
        self.epoch = epoch
        super(SprintDataset, self).init_seq_order(epoch=epoch)
        for i, j in enumerate(self.seq_index):
          self.seq_index_reversed[j] = i
        self.seq_added_excluded = set()  # Via self.addNewData().
        self.seq_added_last = -1

  def init_seqs(self):
    super(SprintDataset, self).init_seqs()
    self.expected_load_seq_start = 0
    self.requested_load_seq_end = 0

  def _realIdxForSegmentName(self, segmentName):
    return self.segmentsInfo[segmentName]["idx"]

  def addNewData(self, segmentName, features, targets=None):
    """
    :type segmentName: str
    :param numpy.ndarray features: format (input-feature,time) (via Sprint)
    :param targets: format (time) (idx of output-feature)
    :returns the sorted seq index
    :rtype: int
    """

    # is in format (feature,time)
    assert self.num_inputs == features.shape[0]
    T = features.shape[1]
    # must be in format: (time,feature)
    features = features.transpose()
    assert features.shape == (T, self.num_inputs)

    if targets is not None:
      assert targets.shape == (T,)  # is in format (time,)

    with self.lock:
      self.add_data_thread_id = thread.get_ident()
      assert not self.finalized
      idxReal = self._realIdxForSegmentName(segmentName)
      assert idxReal < self.num_seqs
      idxSorted = self.seq_index_reversed[idxReal]
      assert not self._haveSeqsAdded(idxSorted)
      seqStart = self.seq_start[idxSorted]
      seqLen = self.seq_lengths[idxReal]
      assert (seqLen, self.num_inputs) == features.shape
      assert seqLen == T
      assert self.alloc_intervals

      print >> log.v5, "SprintDataset addNewData: seq=%i, len=%i" % (idxSorted, seqLen)

      # We expect a monotonic increasing sorted seq order.
      assert self.seq_added_last < idxSorted, "Order messed up, last added idx: %i" % self.seq_added_last
      if self.seq_added_last < idxSorted - 1:
        seqLeftOut = range(self.seq_added_last + 1, idxSorted)
        print >> log.v5, "SprintDataset addNewData: left out seqs: %s" % seqLeftOut
        self.seq_added_excluded.update(seqLeftOut)
      self.seq_added_last = idxSorted
      self.cond.notify_all()

      if idxSorted > self.requested_load_seq_end - 1 + self.SprintCachedSeqsMax:
        print >> log.v5, "SprintDataset addNewData: Cache filled, waiting to get loaded..."
        while idxSorted > self.requested_load_seq_end - 1 + self.SprintCachedSeqsMin:
          self.cond.wait()

      self.insert_alloc_interval(idxSorted)
      self._set_alloc_intervals_data(idxSorted, data=features)

      if targets is not None:
        self.targets[seqStart : seqStart + seqLen] = targets

      self.cond.notify_all()

      return idxSorted
