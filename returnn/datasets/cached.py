
from __future__ import print_function
import gc
import sys
import time
import numpy
import threading
import typing
from .basic import Dataset
from returnn.log import log
from returnn.util import NumbersDict


class CachedDataset(Dataset):

  def __init__(self, cache_byte_size=0, **kwargs):
    """
    :param int cache_byte_size:
    """
    super(CachedDataset, self).__init__(**kwargs)
    self.cache_byte_size_total_limit = cache_byte_size
    if cache_byte_size == -1:
      self.cache_byte_size_limit_at_start = 1024 ** 4
    elif cache_byte_size == 0:
      self.cache_byte_size_limit_at_start = 0
    else:
     self.cache_byte_size_limit_at_start = max(cache_byte_size * 2 // 3, 1)
     self.cache_byte_size_total_limit = max(cache_byte_size - self.cache_byte_size_limit_at_start, 1)
    self.num_seqs_cached_at_start = 0
    self.cached_bytes_at_start = 0
    self.nbytes = 0
    self.start_cache_initialized = False
    self.definite_cache_leftover = 0
    self.cache_num_frames_free = 0
    self.preload_set = set([])
    self.preload_end = 0
    self.max_ctc_length = 0
    self.ctc_targets = None
    self.alloc_intervals = None  # type: typing.Optional[list]
    self._seq_start = []  # [numpy.array([0,0])]  # uses sorted seq idx, see set_batching()
    self._seq_index = []  # type: typing.List[int]  # Via init_seq_order(). seq_index idx -> hdf seq idx
    self._seq_index_inv = {}  # type: typing.Dict[int,int]  # Via init_seq_order(). hdf seq idx -> seq_index idx
    self._index_map = range(len(self._seq_index))  # sorted seq idx -> seq_index idx
    self._tag_idx = {}  # type: typing.Dict[str,int]  # map of tag -> real-seq-idx. call _update_tag_idx
    self.targets = {}
    self.target_keys = []  # the keys for which we provide data; we may have labels for additional keys in self.labels
    self.timestamps = None

  def initialize(self):
    """
    Initialization.
    """
    super(CachedDataset, self).initialize()

    if self.cache_byte_size_limit_at_start > 0:
      self.nbytes = numpy.array([], dtype=numpy.float32).itemsize * (self.num_inputs * self.window + 1 + 1)

      # Calculate cache sizes.
      temp_cache_size_bytes = max(0, self.cache_byte_size_total_limit)
      self.definite_cache_leftover = temp_cache_size_bytes if self.num_seqs_cached_at_start == self.num_seqs else 0
      self.cache_num_frames_free = temp_cache_size_bytes // self.nbytes

      print("cached %i seqs" % self.num_seqs_cached_at_start,
            "%s GB" % (self.cached_bytes_at_start / float(1024 * 1024 * 1024)),
            ("(fully loaded, %s GB left over)" if self.definite_cache_leftover else "(%s GB free)") %
            max(temp_cache_size_bytes / float(1024 * 1024 * 1024), 0),
            file=log.v4)

  def init_seq_order(self, epoch=None, seq_list=None, seq_order=None):
    """
    :type epoch: int|None
    :param list[str]|None seq_list: List of sequence tags, to set a predefined order.
    :param list[int]|None seq_order: List of corpus sequence indices, to set a predefined order.
    Initialize lists:
      self.seq_index  # sorted seq idx
    """
    super(CachedDataset, self).init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)
    if seq_order is not None:
      seq_index = seq_order
    elif seq_list is not None:
      self._update_tag_idx()
      seq_index = [self._tag_idx[tag] for tag in seq_list]
    else:
      seq_index = self.get_seq_order_for_epoch(epoch, self._num_seqs, lambda s: self._get_seq_length_by_real_idx(s)[0])

    old_index_map = self._index_map[:]
    self._index_map = range(len(seq_index))  # sorted seq idx -> seq_index idx

    if (isinstance(seq_index, numpy.ndarray) and numpy.array_equal(self._seq_index, seq_index)
        or self._seq_index == seq_index) and self.start_cache_initialized:
      return False

    if epoch is not None:
      # Give some hint to the user in case he is wondering why the cache is reloading.
      print("Reinitialize dataset seq order for epoch %i." % epoch, file=log.v4)

    if (self.cache_byte_size_limit_at_start == 0
        or self.num_seqs_cached_at_start != len(seq_index)
        or not self.start_cache_initialized):
      self._seq_index = seq_index
      self._seq_index_inv = {}  # reset, create later if needed
      self._init_seq_starts()
      self._init_alloc_intervals()
      self._init_start_cache()
      self.start_cache_initialized = True
    else:
      if not self._seq_index_inv:
        self._seq_index_inv = dict(zip(self._seq_index, range(len(self._seq_index))))  # hdf seq idx -> seq_index idx
      self._index_map = [self._seq_index_inv[i] for i in seq_index]  # sorted seq idx -> seq_index idx
      if self._index_map == old_index_map:
        return False
    return True

  def get_current_seq_order(self):
    assert self.cache_byte_size_limit_at_start == 0  # not implemented otherwise, we ignore _index_map
    return self._seq_index

  def _get_tag_by_real_idx(self, real_idx):
    raise NotImplementedError

  def _update_tag_idx(self):
    if self._tag_idx:
      return
    for i in range(self._num_seqs):
      self._tag_idx[self._get_tag_by_real_idx(i)] = i

  def batch_set_generator_cache_whole_epoch(self):
    return True

  def _init_alloc_intervals(self):
    if self.cache_byte_size_limit_at_start == 0:
      return
    assert self.num_seqs > 0
    assert self.num_inputs > 0
    assert self.window > 0
    self.preload_set = set([])
    self.alloc_intervals = \
      [(0, 0, numpy.zeros([1] + self.get_data_shape("data"), dtype=self.get_data_dtype("data"))),
       (self.num_seqs, self.num_seqs, numpy.zeros([1] + self.get_data_shape("data"), dtype=self.get_data_dtype("data")))]
    # self.alloc_intervals[i] is (idx start, idx end, data), where
    # idx start/end is the sorted seq idx start/end, end exclusive,
    # and data is a numpy.array.

  def _init_seq_starts(self):
    if self.cache_byte_size_limit_at_start == 0:
      return
    self._seq_start = [self._seq_start[0] * 0]  # idx like in seq_index, *not* real idx
    for i in range(self.num_seqs):
      ids = self._seq_index[i]
      self._seq_start.append(self._seq_start[-1] + self._get_seq_length_by_real_idx(ids))

  def _init_start_cache(self):
    if self.cache_byte_size_limit_at_start == 0:
      return
    if not self.alloc_intervals:
      return
    if not self.nbytes:
      return

    num_cached = 0
    cached_bytes = 0
    for i in range(self.num_seqs):
      if i == num_cached:
        nbytes = self.get_seq_length_nd(i)[0] * self.nbytes
        if self.cache_byte_size_limit_at_start >= cached_bytes + nbytes:
          num_cached = i + 1
          cached_bytes += nbytes

    self.num_seqs_cached_at_start = num_cached
    self.cached_bytes_at_start = cached_bytes
    if num_cached > 0:
      self.preload_end = num_cached
      if sys.version_info >= (3, 0):
        threading.Thread(target=self._preload_seqs, args=(0, num_cached), daemon=True).start()
      else:
        threading.Thread(target=self._preload_seqs, args=(0, num_cached)).start()

  def load_seqs(self, start, end):
    """
    Load data sequences.
    As a side effect, will modify / fill-up:
      self.alloc_intervals
      self.targets
    This does some extra logic for the cache and calls self._load_seqs()
    for the real loading.

    :param int start: start sorted seq idx
    :param int end: end sorted seq idx
    """
    assert start >= 0
    assert start <= end

    if self.is_cached(start, end, blocking=True):
      return

    if self.cache_byte_size_limit_at_start > 0:  # If the cache is enabled.
      self._load_seqs_with_cache(start, end)
      return self.is_cached(start, end, blocking=True)

    super(CachedDataset, self).load_seqs(start, end)

  def _load_seqs(self, start, end):
    raise NotImplementedError

  def _load_seqs_with_cache(self, start, end, clear=True):
    if not clear:
      # only remove as many frames as required
      num_needed_cache_frames = self.get_seq_start(end)[0] - self.get_seq_start(start)[0]
      if self.cache_num_frames_free < num_needed_cache_frames:
        self.cache_num_frames_free += self.delete(num_needed_cache_frames - self.cache_num_frames_free)
        gc.collect()
      self.cache_num_frames_free -= num_needed_cache_frames
      threading.Thread(target=self._preload_seqs,args=(start,end)).start()
    else:
      # First, delete everything.
      self.cache_num_frames_free += self.delete(None)
      gc.collect()
      # Preload as much as we can so that we fill up the cache.
      while end < self.num_seqs:
        num_needed_cache_frames = self.get_seq_length_nd(end)[0]
        if self.cache_num_frames_free - num_needed_cache_frames < 0:
          break
        self.cache_num_frames_free -= num_needed_cache_frames
        end += 1
      self.preload_end = end
      threading.Thread(target=self._preload_seqs,args=(start,end)).start()

  def _preload_seqs(self,start,end):
    print("Preloading cache from", start, "to", end, file=log.v4)
    super(CachedDataset, self).load_seqs(start, end)
    self.preload_end = self.num_seqs_cached_at_start

  def _shuffle_frames_in_seqs(self, start, end):
    """
    :type start: int
    :type end: int
    """
    assert start < end
    assert self.is_cached(start, end)
    alloc_idx = self.alloc_interval_index(start)
    alloc_start, alloc_end, alloc_data = self.alloc_intervals[alloc_idx]
    assert start >= alloc_start
    assert end <= alloc_end
    rnd = numpy.random.RandomState(start)  # Some deterministic way to shuffle!
    num_frames = self._seq_start[end][0] - self._seq_start[start][0]
    assert num_frames > 0
    perm = rnd.permutation(num_frames)
    alloc_offset = self._seq_start[start][0] - self._seq_start[alloc_start][0]
    assert alloc_offset + num_frames <= alloc_data.shape[0]
    # Permute alloc_data.
    data = alloc_data[alloc_offset:alloc_offset + num_frames]
    alloc_data[alloc_offset:alloc_offset + num_frames] = data[perm]
    # Permute targets.
    for k in self.targets:
      idx = self.target_keys.index(k) + 1
      targets = self.targets[k][self._seq_start[idx]:self._seq_start[start][idx] + num_frames]
      self.targets[k][self._seq_start[start][idx]:self._seq_start[start][idx] + self._seq_start[end][idx] - self._seq_start[start][idx]] = targets[perm]

  def _set_alloc_intervals_data(self, idc, data):
    """
    :param int idc: index of sorted seq idx
    :param numpy.ndarray data: raw data
    """
    idi = self.alloc_interval_index(idc)
    assert idi >= 0
    o = self._seq_start[idc][0] - self._seq_start[self.alloc_intervals[idi][0]][0]
    l = data.shape[0]
    x = data
    if self.window > 1:
      x = self._sliding_window(x)
    self.alloc_intervals[idi][2][o:o + l] = x

  def alloc_interval_index(self, ids):
    """
    :param int ids: sorted seq idx
    :return index in self.alloc_intervals
    :rtype: int
    """
    s = 0
    e = len(self.alloc_intervals)
    # Binary search.
    while s < e:
      i = (s + e) // 2
      alloc_start, alloc_end, _ = self.alloc_intervals[i]
      if alloc_start <= ids < alloc_end:
        return i
      elif alloc_start <= ids and ids >= alloc_end:
        if s == i:
          return -1
        s = i
      elif alloc_start > ids:
        if e == i:
          return -1
        e = i
      else:
        assert False
    return -1

  def _insert_alloc_interval(self, pos, value, merge=False):
    """
    Insert np.zeros into self.alloc_intervals.
    :param int pos: idx in self.alloc_intervals
    :param (int,int) value: (start,end) like in load_seqs(), sorted seq idx
    :rtype: int
    """
    if value[0] == value[1]:
      return 0
    ci = self.alloc_intervals[pos][1]
    ni = self.alloc_intervals[pos + 1][0]
    xc = self.alloc_intervals[pos][2]
    xn = self.alloc_intervals[pos + 1][2]
    if value[0] == ci and value[1] == ni and merge:
      nj = self.alloc_intervals[pos][0]
      nk = self.alloc_intervals[pos + 1][1]
      del self.alloc_intervals[pos]
      del self.alloc_intervals[pos]
      self.alloc_intervals.insert(pos,
        (nj,nk,
         numpy.concatenate(
           [xc,
            numpy.zeros(
              [self._seq_start[ni][0]] + self.get_data_shape("data"),
              dtype=self.get_data_dtype("data")),
            xn])))
      return 0
    elif value[0] == ci and merge:
      nj = self.alloc_intervals[pos][0]
      del self.alloc_intervals[pos]
      self.alloc_intervals.insert(pos, (nj,value[1],
                                        numpy.concatenate([xc, numpy.zeros([self._seq_start[value[1]][0] - self._seq_start[ci][0]] + self.get_data_shape("data"), dtype=self.get_data_dtype("data"))])))
      return 0
    elif value[1] == ni and merge:
      nk = self.alloc_intervals[pos + 1][1]
      del self.alloc_intervals[pos + 1]
      self.alloc_intervals.insert(pos + 1, (value[0], nk,
                                            numpy.concatenate([numpy.zeros([self._seq_start[ni][0] - self._seq_start[value[0]][0]] + self.get_data_shape("data"), dtype=self.get_data_dtype("data")), xc])))
      return 0
    else:
      self.alloc_intervals.insert(pos + 1,
        value + (numpy.zeros(
            [self._seq_start[value[1]][0] - self._seq_start[value[0]][0]] + self.get_data_shape("data"),
            dtype=self.get_data_dtype("data")),))
      return 1

  def _remove_alloc_interval(self, pos, value):
    """
    Remove data from self.alloc_intervals.
    :param int pos: idx in self.alloc_intervals
    :param (int,int) value: (start,end) like in load_seqs(), sorted seq idx
    :rtype: int
    """
    ci, ni, xi = self.alloc_intervals[pos]
    if value[0] == ci and value[1] == ni:
      del self.alloc_intervals[pos]
      return -1
    elif value[0] == ci:
      self.alloc_intervals.insert(pos, (value[1], ni, xi[self._seq_start[value[1]][0] - self._seq_start[ci][0]:]))
      del self.alloc_intervals[pos + 1]
      return 0
    elif value[1] == ni:
      self.alloc_intervals.insert(pos, (ci, value[0], xi[:self._seq_start[value[0]][0] - self._seq_start[ci][0]]))
      del self.alloc_intervals[pos + 1]
      return 0
    else:
      self.alloc_intervals.insert(pos, (value[1], ni, xi[self._seq_start[value[1]][0] - self._seq_start[ci][0]:]))
      self.alloc_intervals.insert(pos, (ci, value[0], xi[:self._seq_start[value[0]][0] - self._seq_start[ci][0]]))
      del self.alloc_intervals[pos + 2]
      return 1

  def _modify_alloc_intervals(self, start, end, invert):
    """
    Inserts/removes sorted seq idx range (start,end).
    :param int start: like in load_seqs(), sorted seq idx
    :param int end: like in load_seqs(), sorted seq idx
    :param bool invert: True->insert, False->remove
    :rtype: list[int]
    :return selection list, modified sorted seq idx in self.alloc_intervals
    """
    if end is None:
      end = start + 1
    if start == end:
      return
    assert start < end
    i = 0
    selection = []  # type: typing.List[int]
    modify = self._insert_alloc_interval if invert else self._remove_alloc_interval
    while i < len(self.alloc_intervals) - invert:
      ni = self.alloc_intervals[i + invert][1 - invert]  # insert mode: start idx of next alloc
      ci = self.alloc_intervals[i][invert]               # insert mode: end idx of cur alloc
      assert ci <= ni
      flag = ((ci <= start < ni), (ci < end <= ni), (ci < start and ni <= start) or (ci >= end and ni > end))
      if not flag[0] and not flag[1]:
        if not flag[2]:
          selection.extend(range(ci, ni))
          i += modify(i, (ci, ni))
      elif flag[1]:
        v = (start if flag[0] else ci, end)
        selection.extend(range(v[0], v[1]))
        i += modify(i, v)
        break
      elif flag[0]:
        selection.extend(range(start, ni))
        i += modify(i, (start, ni))
      i += 1
    if self.alloc_intervals[0][0] != 0:
      self.alloc_intervals.insert(0, (0, 0, numpy.zeros([1] + self.get_data_shape("data"), dtype=self.get_data_dtype("data"))))
    if self.alloc_intervals[-1][1] != self.num_seqs:
      self.alloc_intervals.append((self.num_seqs, self.num_seqs, numpy.zeros([1] + self.get_data_shape("data"), dtype=self.get_data_dtype("data"))))
    return selection

  def insert_alloc_interval(self, start, end=None):
    return self._modify_alloc_intervals(start, end, True)

  def remove_alloc_interval(self, start, end=None):
    return self._modify_alloc_intervals(start, end, False)

  def delete(self, nframes):
    """
    :param int|None nframes: how much frames to delete max.
      Note that this limit is not strict. We can end up
      deleting more than nframes.
    :return: number of frames deleted
    :rtype: int
    """
    if nframes is not None:
      if nframes == 0:
        return 0
      assert nframes > 0
    deleted = 0
    i = 0
    while (not nframes or deleted < nframes) and i < len(self.alloc_intervals):
      ai = self.alloc_intervals[i]
      if ai[1] > self.num_seqs_cached_at_start and ai[0] < ai[1]:
        removed = self.remove_alloc_interval(max(ai[0],self.num_seqs_cached_at_start), ai[1])
        self.preload_set -= set(removed)
        deleted += sum([self._get_seq_length_by_real_idx(self._seq_index[i])[0] for i in removed])
      else:
        i += 1
    return deleted

  @property
  def num_seqs(self):
    if self._index_map:
      return len(self._index_map)
    return self._num_seqs

  def is_cached(self, start, end, blocking = False):
    """
    :param int start: like in load_seqs(), sorted seq idx
    :param int end: like in load_seqs(), sorted seq idx
    :rtype: bool
    :returns whether we have the full range (start,end) of sorted seq idx
      cached in self.alloc_intervals (end is exclusive).
    """
    if self.cache_byte_size_total_limit == 0:  # disabled cache
      return False
    if start == end:
      return True  # Empty.
    assert start < end
    if blocking and end <= self.preload_end:
      while not set(range(start,end)) <= self.preload_set:
        time.sleep(0.2)
      return True
    return set(range(start,end)) <= self.preload_set

  def _get_seq_length_by_real_idx(self, real_seq_idx):
    """
    :param int real_seq_idx:
    :returns length of the sequence with index 'real_seq_idx'
    :rtype: numpy.ndarray
    """
    raise NotImplementedError

  def get_seq_length_nd(self, sorted_seq_idx):
    """
    :type sorted_seq_idx: int
    :rtype: numpy.ndarray
    """
    real_seq_idx = self._seq_index[self._index_map[sorted_seq_idx]]
    return self._get_seq_length_by_real_idx(real_seq_idx)

  def get_seq_length(self, seq_idx):
    """
    :rtype: NumbersDict
    """
    lengths = self.get_seq_length_nd(seq_idx)
    d = {}
    first_target_idx = 0
    # We allow using only targets. In this case self.num_inputs == 0 and the "data" key is not used.
    if self.num_inputs > 0:
      d["data"] = lengths[0]
      first_target_idx = 1
    for k, l in zip(self.target_keys, lengths[first_target_idx:]):
      d[k] = l
    return NumbersDict(d)

  def get_seq_start(self, sorted_seq_idx):
    """
    :type sorted_seq_idx: int
    :rtype: (int,int)
    """
    return self._seq_start[sorted_seq_idx]

  def get_times(self, sorted_seq_idx):
    seq_start = self.get_seq_start(sorted_seq_idx)[0]
    seq_len = self.get_seq_length_nd(sorted_seq_idx)[0]
    return self.timestamps[seq_start:seq_start + seq_len]

  def get_input_data(self, sorted_seq_idx):
    seq_idx = self._index_map[sorted_seq_idx]
    idi = self.alloc_interval_index(seq_idx)
    assert idi >= 0, "failed to get data for seq %i" % sorted_seq_idx
    alloc_start_seq, alloc_end_seq, alloc_data = self.alloc_intervals[idi]
    o = self.get_seq_start(seq_idx)[0] - self.get_seq_start(alloc_start_seq)[0]
    assert o >= 0
    l = self.get_seq_length_nd(sorted_seq_idx)[0]
    assert alloc_data.shape[0] >= o + l
    return alloc_data[o:o + l]

  def get_data_dim(self, key):
    if key == "data" and self.num_inputs > 0:  # if num_inputs == 0, we allow "data" as a target key
      return self.num_inputs * self.window
    return self.num_outputs[key][0]

  def get_targets(self, target, sorted_seq_idx):
    seq_idx = self._index_map[sorted_seq_idx]
    idx = self.target_keys.index(target) + 1
    seq_start = self.get_seq_start(seq_idx)[idx]
    seq_len = self.get_seq_length_nd(sorted_seq_idx)[idx]
    return self.targets[target][seq_start:seq_start + seq_len]

  def get_target_list(self):
    return list(self.targets.keys())

  def get_ctc_targets(self, sorted_seq_idx):
    ids = self._seq_index[self._index_map[sorted_seq_idx]]
    return self.ctc_targets[ids]

  def has_ctc_targets(self):
    return self.ctc_targets is not None

  def get_tag(self, sorted_seq_idx):
    raise NotImplementedError

  def have_corpus_seq_idx(self):
    return True

  def get_corpus_seq_idx(self, seq_idx):
    """
    :param int seq_idx: sorted sequence index from the current epoch, depending on seq_ordering
    :return: the sequence index as-is in the original corpus. only defined if self.have_corpus_seq_idx()
    :rtype: int
    """
    return self._seq_index[self._index_map[seq_idx]]
