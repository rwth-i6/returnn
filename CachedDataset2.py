

from Dataset import Dataset, DatasetSeq
import math


class CachedDataset2(Dataset):
  """
  Somewhat like CachedDataset, but different.
  Simpler in some sense. And more generic. Caching might be worse.
  """

  def __init__(self, **kwargs):
    super(CachedDataset2, self).__init__(**kwargs)
    self._num_timesteps = None
    self.epoch = None

  def init_seq_order(self, epoch=None, seq_list=None):
    """
    :type epoch: int|None
    :param list[str] | None seq_list: In case we want to set a predefined order.

    This is called when we start a new epoch, or at initialization.
    Call this when you reset the seq list.
    """
    super(CachedDataset2, self).init_seq_order(epoch=epoch, seq_list=seq_list)
    if not epoch:
      epoch = 1
    self.expected_load_seq_start = 0
    self.reached_final_seq = False
    self.added_data = []; " :type: list[DatasetSeq] "
    self._num_timesteps_accumulated = 0
    self._num_seqs = None
    self.epoch = epoch
    return True

  def _cleanup_old_seqs(self, seq_idx_end):
    i = 0
    while i < len(self.added_data):
      if self.added_data[i].seq_idx >= seq_idx_end:
        break
      i += 1
    del self.added_data[:i]

  def _get_seq(self, seq_idx):
    for data in self.added_data:
      if data.seq_idx == seq_idx:
        return data
    return None

  def is_cached(self, start, end):
    # Always False, to force that we call self._load_seqs().
    # This is important for our buffer management.
    return False

  @property
  def num_seqs(self):
    if self._num_seqs is not None:
      return self._num_seqs
    raise NotImplementedError

  def get_complete_frac(self, seq_idx):
    """
    :return: Returns a fraction (float in [0,1], always > 0) of how far we have advanced
      for this seq in the dataset.
      This does not have to be exact. This is only for the user.
    """
    try:
      num_seqs = self.num_seqs
    except Exception:
      num_seqs = None
    if num_seqs is not None:
      return float(seq_idx + 1) / self._num_seqs
    # Fallback.
    # We don't know. So:
    # Some monotonic increasing function in [0,1] which never reaches 1.
    return max(1.e-20, 1.0 - math.exp(-seq_idx * 1000))

  def _find_num_seqs(self, start, end):
    """
    We might not know the num seqs in advance.
    When we hit behind the end, now find the real num seqs via self.is_less_than_num_seqs().
    """
    try:
      return self.num_seqs
    except Exception:  # might not be defined yet.
      pass
    assert self.is_less_than_num_seqs(start)
    assert not self.is_less_than_num_seqs(end)
    # We could do binary search, but this is just simpler.
    for n in range(start + 1, end + 1):
      if not self.is_less_than_num_seqs(n):
        return n
    assert False  # should not get here

  def _load_seqs(self, start, end):
    """
    :param int start: inclusive seq idx start
    :param int end: exclusive seq idx end
    """
    # We expect that start increase monotonic on each call
    # for not-yet-loaded data.
    # This will already be called with _load_seqs_superset indices.
    assert start >= self.expected_load_seq_start
    if start > self.expected_load_seq_start:
      # Cleanup old data.
      self._cleanup_old_seqs(start)
      self.expected_load_seq_start = start
    if self.added_data:
      start = max(self.added_data[-1].seq_idx + 1, start)
    # We might not know the num seqs in advance, thus this more generic check.
    if not self.is_less_than_num_seqs(end):
      self._num_seqs = self._find_num_seqs(start, end)
      self.reached_final_seq = True
      end = self.num_seqs
    seqs = [self._collect_single_seq(seq_idx=seq_idx) for seq_idx in range(start, end)]
    self._num_timesteps_accumulated += sum([seq.num_frames for seq in seqs])
    self.added_data += seqs

  def _collect_single_seq(self, seq_idx):
    """
    :type seq_idx: int
    :rtype: DatasetSeq
    """
    raise NotImplementedError

  def get_num_timesteps(self):
    if self._num_timesteps is not None:
      return self._num_timesteps
    else:
      assert self.reached_final_seq
      return self._num_timesteps_accumulated

  def _load_something(self):
    if self.added_data:
      return
    self.load_seqs(self.expected_load_seq_start, self.expected_load_seq_start + 1)

  def get_seq_length(self, sorted_seq_idx):
    """
    :type sorted_seq_idx: int
    :rtype: int
    """
    # get_seq_length() can be called before the seq is loaded via load_seqs().
    # Thus, we just call load_seqs() ourselves here.
    assert sorted_seq_idx >= self.expected_load_seq_start
    self.load_seqs(self.expected_load_seq_start, sorted_seq_idx + 1)
    return self._get_seq(sorted_seq_idx).num_frames

  def get_input_data(self, sorted_seq_idx):
    return self._get_seq(sorted_seq_idx).features

  def get_targets(self, target, sorted_seq_idx):
    return self._get_seq(sorted_seq_idx).targets[target]

  def get_ctc_targets(self, sorted_seq_idx):
    return self._get_seq(sorted_seq_idx).ctc_targets

  def get_tag(self, sorted_seq_idx):
    return self._get_seq(sorted_seq_idx).seq_tag

  def get_target_list(self):
    self._load_something()
    return self.added_data[0].targets.keys()

  def is_data_sparse(self, key):
    """
    :type key: str
    :rtype: bool
    """
    if key in self.num_outputs:
      return self.num_outputs[key][1] == 1
    self._load_something()
    return len(self.added_data[0].features.shape) == 1

  def get_data_dim(self, key):
    """
    :type key: str
    :rtype: int
    :return: number of classes, no matter if sparse or not
    """
    if key in self.num_outputs:
      d = self.num_outputs[key][0]
      if self.added_data and not self.is_data_sparse(key):
        assert self.added_data[0].features.shape[1] == d
      return d
    self._load_something()
    if len(self.added_data[0].features.shape) == 1:
      return super(CachedDataset2, self).get_data_dim(key)  # unknown
    assert len(self.added_data[0].features.shape) == 2
    return self.added_data[0].features.shape[1]

  def get_data_dtype(self, key):
    self._load_something()
    return self.added_data[0].get_data(key).dtype
