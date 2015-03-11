
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
    Called via python_train.
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
    print >> log.v5, currentThread().name, ": load_seqs", start, end, free, fill
    if start == end: return
    if thread.get_ident() == self.main_thread_id:
      print >> log.v5, "ignore load_seqs from this thread"
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
      if not self._haveSeqs(start, end):
        print >> log.v5, "have not seqs. waiting..."
      while not self._haveSeqs(start, end):
        assert not self.finalized
        self.cond.wait()

  def _haveSeqs(self, start, end=None):
    if end is None: end = start + 1
    return self.is_cached(start, end)

  def finalize(self):
    # Called by segment_order.
    with self.lock:
      self.finalized = True
      self.cond.notify_all()

  def init_seq_order(self, epoch=None):
    print >> log.v5, currentThread().name, "init_seq_order", epoch
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

  def init_seqs(self):
    super(SprintDataset, self).init_seqs()
    self.expected_load_seq_start = 0
    self.requested_load_seq_end = 0

  def _realIdxForSegmentName(self, segmentName):
    return self.segmentsInfo[segmentName]["idx"]

  def addNewData(self, segmentName, features, targets):
    """
    :type segmentName: str
    :param numpy.ndarray features: format (time,input-feature)
    :param targets: format (time) (idx of output-feature)
    """
    with self.lock:
      assert not self.finalized
      idxReal = self._realIdxForSegmentName(segmentName)
      assert idxReal < self.num_seqs
      idxSorted = self.seq_index_reversed[idxReal]
      assert not self._haveSeqs(idxSorted)
      seqStart = self.seq_start[idxSorted]
      seqLen = self.seq_lengths[idxReal]
      assert (seqLen, self.num_inputs) == features.shape
      assert (seqLen,) == targets.shape
      assert self.alloc_intervals

      print >> log.v5, "addNewData: seq=%i, len=%i" % (idxSorted, seqLen)

      if idxSorted > self.requested_load_seq_end - 1 + self.SprintCachedSeqsMax:
        print >> log.v5, "Cache filled, waiting to get loaded..."
        while idxSorted > self.requested_load_seq_end - 1 + self.SprintCachedSeqsMin:
          self.cond.wait()

      self.insert_alloc_interval(idxSorted)
      self._set_alloc_intervals_data(idxSorted, data=features)

      self.targets[seqStart : seqStart + seqLen] = targets

      self.cond.notify_all()

