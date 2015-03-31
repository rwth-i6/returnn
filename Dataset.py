#! /usr/bin/python2.7

__author__ = "Patrick Doetsch"
__copyright__ = "Copyright 2015"
__credits__ = ["Patrick Doetsch", "Paul Voigtlaender"]
__license__ = "RWTHASR"
__version__ = "0.9"
__maintainer__ = "Patrick Doetsch"
__email__ = "doetsch@i6.informatik.rwth-aachen.de"

import numpy
import h5py
from Log import log
import theano
import theano.tensor as T
import gc
from threading import RLock
from random import Random
from EngineBatch import Batch, BatchSetGenerator


class Dataset(object):
  @classmethod
  def load_data(cls, config, cache_byte_size, files_config_key=None, chunking="chunking",
                batching="batching", shuffle_frames_of_nseqs=None):
    """
    :type config: Config.Config
    :type cache_byte_size: int
    :type files_config_key: str
    :type chunking: str
    :type batching: str
    :returns the cache byte size left over if we cache the whole dataset.
    """
    window = config.int('window', 1)
    if chunking == "chunking": chunking = config.value("chunking", "0")
    if batching == "batching": batching = config.value("batching", 'default')
    if shuffle_frames_of_nseqs is None:
      shuffle_frames_of_nseqs = config.int('shuffle_frames_of_nseqs', 0)
    data = cls(window, cache_byte_size, chunking, batching, shuffle_frames_of_nseqs)
    extra_bytes = 0
    if files_config_key:
      if config.has(files_config_key):
        for f in config.list(files_config_key):
          data.add_file(f)
        extra_bytes = data.initialize()
      else:
        return None, 0
    return data, extra_bytes

  def __init__(self, window=1, cache_byte_size=0, chunking="0", batching='default',
               shuffle_frames_of_nseqs=0):
    self.lock = RLock()  # Used when manipulating our data potentially from multiple threads.
    self.files = []; """ :type: list[str] """
    self.num_inputs = 0
    self.num_outputs = 0
    self.window = window
    self.batching = batching
    self.cache_byte_size_total_limit = cache_byte_size
    if cache_byte_size < 0:
      self.cache_byte_size_limit_at_start = 1
    else:
      temp_cache_amount = 0.5
      self.cache_byte_size_limit_at_start = int((1.0 - temp_cache_amount) * cache_byte_size)
    self.num_seqs_cached_at_start = 0
    self.cached_bytes_at_start = 0
    self.seq_start = [0]  # uses sorted seq idx, see set_batching()
    self.seq_shift = [0]
    self.rnd_seed = 0  # See init_seq_order().
    self.file_start = [0]
    self.file_seq_start = []; """ :type: list[list[int]] """
    self.timestamps = []
    self.file_index = []; """ :type: list[int] """
    self.seq_index = []; """ :type: list[int] """  # Via init_seq_order().
    self.seq_lengths = []; """ :type: list[int] """  # uses real seq idx
    self.labels = []; """ :type: list[str] """
    self.tags = []; """ :type: list[str] """
    self.num_seqs = 0
    self.num_timesteps = 0
    self.chunk_size = int(chunking.split(':')[0])
    if ':' in chunking:
      self.chunk_step = int(chunking.split(':')[1])
      assert self.chunk_step > 0, "chunking step must be positive"
    else:
      self.chunk_step = self.chunk_size
    assert self.chunk_size >= 0, "chunk size must not be negative"
    self.shuffle_frames_of_nseqs = shuffle_frames_of_nseqs
    self.num_running_chars = 0
    self.max_ctc_length = 0
    self.ctc_targets = None
    self.isInitialized = False

  def add_file(self, filename):
    """
    Setups data:
      self.seq_lengths
      self.file_index
      self.file_start
      self.file_seq_start
    Use load_seqs() to load the actual data.
    :type filename: str
    """
    fin = h5py.File(filename, "r")
    labels = [ item.split('\0')[0] for item in fin["labels"][...].tolist() ]; """ :type: list[str] """
    if not self.labels:
      self.labels = labels
    assert len(self.labels) == len(labels), "expected " + str(len(self.labels)) + " got " + str(len(labels))
    tags = [ item.split('\0')[0] for item in fin["seqTags"][...].tolist() ]; """ :type: list[str] """
    self.files.append(filename)
    seq_start = [0]
    if 'times' in fin:
      self.timestamps.extend(fin['times'][...].tolist())
    for l,t in zip(fin['seqLengths'][...].tolist(), tags):
      if self.chunk_size == 0:
        self.seq_lengths.append(l)
        seq_start.append(seq_start[-1] + l)
        self.tags.append(t)
      else:
        while l > 0:
          chunk = min(l, self.chunk_size)
          self.seq_lengths.append(chunk)
          shift = min(chunk, self.chunk_step)
          seq_start.append(seq_start[-1] + shift)
          self.tags.append(t + "_" + str(len(seq_start) - 1))
          l -= shift
    self.file_seq_start.append(seq_start)
    nseqs = len(seq_start) - 1
    self.num_seqs += nseqs
    self.file_index.extend([len(self.files) - 1] * nseqs)
    self.file_start.append(self.file_start[-1] + nseqs)
    self.num_timesteps = sum(self.seq_lengths)
    if 'maxCTCIndexTranscriptionLength' in fin.attrs:
      self.max_ctc_length = max(self.max_ctc_length, fin.attrs['maxCTCIndexTranscriptionLength'])
    if self.num_inputs == 0:
      self.num_inputs = fin.attrs['inputPattSize']
    assert self.num_inputs == fin.attrs['inputPattSize'], "wrong input dimension in file " + filename + " (expected " + str(self.num_inputs) + " got " + str(fin.attrs['inputPattSize']) + ")"
    if self.num_outputs == 0:
      self.num_outputs = fin.attrs['numLabels']
    assert self.num_outputs == fin.attrs['numLabels'], "wrong number of labels in file " + filename  + " (expected " + str(self.num_outputs) + " got " + str(fin.attrs['numLabels']) + ")"
    if 'ctcIndexTranscription' in fin:
      if self.ctc_targets is None:
        self.ctc_targets = fin['ctcIndexTranscription'][...]
      else:
        tmp = fin['ctcIndexTranscription'][...]
        pad_width = self.max_ctc_length - tmp.shape[1]
        tmp = numpy.pad(tmp, ((0,0),(0,pad_width)), 'constant', constant_values=-1)
        pad_width = self.max_ctc_length - self.ctc_targets.shape[1]
        self.ctc_targets = numpy.pad(self.ctc_targets, ((0,0),(0,pad_width)), 'constant', constant_values=-1)
        self.ctc_targets = numpy.concatenate((self.ctc_targets, tmp))
      self.num_running_chars = numpy.sum(self.ctc_targets != -1)
    fin.close()

  def sliding_window(self, xr):
    """
    :type xr: numpy.ndarray
    :rtype: numpy.ndarray
    """
    from numpy.lib.stride_tricks import as_strided as ast
    x = numpy.concatenate([self.zpad, xr, self.zpad])
    return ast(x,
               shape=(x.shape[0] - self.window + 1, 1, self.window, self.num_inputs),
               strides=(x.strides[0], x.strides[1] * self.num_inputs) + x.strides
               ).reshape((xr.shape[0], self.num_inputs * self.window))

  def preprocess(self, seq):
    """
    :type xr: numpy.ndarray
    :rtype: numpy.ndarray
    """
    return seq

  def _insert_alloc_interval(self, pos, value):
    """
    Insert np.zeros into self.alloc_intervals.
    :param int pos: idx in self.alloc_intervals
    :param (int,int) value: (start,end) like in load_seqs(), sorted seq idx
    :rtype: int
    """
    ci = self.alloc_intervals[pos][1]
    ni = self.alloc_intervals[pos + 1][0]
    xc = self.alloc_intervals[pos][2]
    xn = self.alloc_intervals[pos + 1][2]
    if value[0] == ci and value[1] == ni:
      self.alloc_intervals.insert(pos,
        (self.alloc_intervals[pos][0],
         self.alloc_intervals[pos + 1][1],
         numpy.concatenate(
           [xc,
            numpy.zeros(
              (self.seq_start[ni] - self.seq_start[ci], self.num_inputs * self.window),
              dtype=theano.config.floatX),
            xn])))
      del self.alloc_intervals[pos + 1]
      del self.alloc_intervals[pos + 1]
      return 0
    elif value[0] == ci:
      self.alloc_intervals.insert(pos, (self.alloc_intervals[pos][0],
                                        value[1],
                                        numpy.concatenate([xc, numpy.zeros((self.seq_start[value[1]] - self.seq_start[ci], self.num_inputs * self.window), dtype=theano.config.floatX)])))
      del self.alloc_intervals[pos + 1]
      return 0
    elif value[1] == ni:
      self.alloc_intervals.insert(pos + 1, (value[0],
                                            self.alloc_intervals[pos + 1][1],
                                            numpy.concatenate([numpy.zeros((self.seq_start[ni] - self.seq_start[value[0]], self.num_inputs * self.window), dtype=theano.config.floatX), xc])))
      del self.alloc_intervals[pos + 2]
      return 0
    else:
      self.alloc_intervals.insert(pos + 1,
        value + (numpy.zeros(
            (self.seq_start[value[1]] - self.seq_start[value[0]],
             self.num_inputs * self.window),
            dtype=theano.config.floatX),))
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
      self.alloc_intervals.insert(pos, (value[1], ni, xi[self.seq_start[value[1]] - self.seq_start[ci]:]))
      del self.alloc_intervals[pos + 1]
      return 0
    elif value[1] == ni:
      self.alloc_intervals.insert(pos, (ci, value[0], xi[:self.seq_start[value[0]] - self.seq_start[ci]]))
      del self.alloc_intervals[pos + 1]
      return 0
    else:
      self.alloc_intervals.insert(pos, (value[1], ni, xi[self.seq_start[value[1]] - self.seq_start[ci]:]))
      self.alloc_intervals.insert(pos, (ci, value[0], xi[:self.seq_start[value[0]] - self.seq_start[ci]]))
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
    if end is None: end = start + 1
    if start == end: return
    assert start < end
    i = 0
    selection = []; """ :type: list[int] """
    modify = self._insert_alloc_interval if invert else self._remove_alloc_interval
    while i < len(self.alloc_intervals) - invert:
      ni = self.alloc_intervals[i + invert][1 - invert]  # insert mode: start idx of next alloc
      ci = self.alloc_intervals[i][invert]               # insert mode: end idx of cur alloc
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
      self.alloc_intervals.insert(0, (0, 0, numpy.zeros((1, self.num_inputs * self.window), dtype=theano.config.floatX)))
    if self.alloc_intervals[-1][1] != self.num_seqs:
      self.alloc_intervals.append((self.num_seqs, self.num_seqs, numpy.zeros((1, self.num_inputs * self.window), dtype=theano.config.floatX)))
    return selection

  def insert_alloc_interval(self, start, end=None):
    return self._modify_alloc_intervals(start, end, True)
  def remove_alloc_interval(self, start, end=None):
    return self._modify_alloc_intervals(start, end, False)

  def is_cached(self, start, end):
    """
    :param int start: like in load_seqs(), sorted seq idx
    :param int end: like in load_seqs(), sorted seq idx
    :rtype: bool
    :returns whether we have the full range (start,end) of sorted seq idx
      cached in self.alloc_intervals (end is exclusive).
    """
    if start == end: return True  # Empty.
    assert start < end
    s = 0
    e = len(self.alloc_intervals)
    # Binary search.
    while s < e:
      i = (s + e) / 2
      alloc_start, alloc_end, _ = self.alloc_intervals[i]
      if alloc_start <= start < alloc_end:
        return alloc_start < end <= alloc_end
      elif alloc_start <= start and start >= alloc_end:
        if s == i: return False
        s = i
      elif alloc_start > start:
        if e == i: return False
        e = i
      else:
        assert False
    return False

  def alloc_interval_index(self, ids):
    """
    :param int ids: index of sorted seq idx
    :return index in self.alloc_intervals
    :rtype: int
    """
    s = 0
    e = len(self.alloc_intervals)
    # Binary search.
    while s < e:
      i = (s + e) / 2
      alloc_start, alloc_end, _ = self.alloc_intervals[i]
      if alloc_start <= ids < alloc_end:
        return i
      elif alloc_start <= ids and ids >= alloc_end:
        if s == i: return -1
        s = i
      elif alloc_start > ids:
        if e == i: return -1
        e = i
      else:
        assert False
    return -1

  def delete(self, nframes):
    """
    :param int nframes: how much frames to delete max.
      Note that this limit is not strict. We can end up
      deleting more than nframes.
    :return: number of frames deleted
    :rtype: int
    """
    assert nframes >= 0
    deleted = 0
    i = 0
    while deleted < nframes and i < len(self.alloc_intervals):
      ai = self.alloc_intervals[i]
      if ai[1] > self.num_seqs_cached_at_start and ai[0] < ai[1]:
        s = max(ai[0], self.num_seqs_cached_at_start)
        deleted += sum([self.seq_lengths[self.seq_index[i]]
                        for i in self.remove_alloc_interval(s, ai[1])])
      i += 1
    return deleted

  def get_seq_length(self, sorted_seq_idx):
    """
    :type sorted_seq_idx: int
    :rtype: int
    """
    real_seq_idx = self.seq_index[sorted_seq_idx]
    return self.seq_lengths[real_seq_idx]

  def have_seqs(self, start, end):
    """
    :param int start: start sorted seq idx
    :param int end: end sorted seq idx
    :returns whether this dataset includes the seq range
    """
    if start < 0: return False
    if end > self.num_seqs: return False
    return True

  def load_seqs(self, start, end, free=True, fill=True):
    """
    Load data sequences.
    As a side effect, will modify / fill-up:
      self.alloc_intervals
      self.targets
    This does some extra logic for the cache and calls self._load_seqs()
    for the real loading.

    :param int start: start sorted seq idx
    :param int end: end sorted seq idx
    :param bool free: check and maybe free cache
    :param bool fill: after freeing, fill up cache
    """
    assert start >= 0
    assert start <= end
    assert start < self.num_seqs
    assert end <= self.num_seqs
    if self.is_cached(start, end): return

    if self.cache_byte_size_total_limit > 0 and free:  # If the cache is enabled.
      num_needed_cache_frames = self.seq_start[end] - self.seq_start[start]
      if self.cache_num_frames_free < num_needed_cache_frames:
        self.cache_num_frames_free += self.delete(num_needed_cache_frames - self.cache_num_frames_free)
        gc.collect()

      self.cache_num_frames_free -= num_needed_cache_frames

      if fill:
        # First, delete everything.
        self.cache_num_frames_free += self.delete(self.num_timesteps)
        gc.collect()
        # Load as much as we can so that we fill up the cache.
        while end < self.num_seqs:
          ids = self.seq_index[end]
          num_needed_cache_frames = self.seq_lengths[ids]
          if self.cache_num_frames_free - num_needed_cache_frames < 0:
            break
          self.cache_num_frames_free -= num_needed_cache_frames
          end += 1
        self.load_seqs(start, end, free=False)
        if end == self.num_seqs:
          # Preload from the start for the next epoch.
          start = self.num_seqs_cached_at_start
          end = start
          while end < start:
            num_needed_cache_frames = self.seq_lengths[self.seq_index[end]]
            if self.cache_num_frames_free - num_needed_cache_frames < 0:
              break
            self.cache_num_frames_free -= num_needed_cache_frames
            end += 1
          if end != start:
            self.load_seqs(start, end, free=False)
        return

    if self.shuffle_frames_of_nseqs > 0:
      # We always load N seqs at once and shuffle all their frames.
      start, end = self._load_seqs_superset(start, end)
      self._load_seqs(start, end)
      while start < end:
        self.shuffle_seqs(start, min(self.num_seqs, start + self.shuffle_frames_of_nseqs))
        start += self.shuffle_frames_of_nseqs
    else:
      self._load_seqs(start, end)

  def _load_seqs_superset(self, start, end):
    """
    :type start: int
    :type end: int
    :returns the superset (start,end) of seqs to be loaded.
    For shuffle_frames_of_nseqs > 0, we always load N seqs at once
    and shuffle all their frames,
    thus start/end will be aligned to self.shuffle_frames_of_nseqs,
    except of the very end.
    """
    assert start <= end
    assert start < self.num_seqs
    if self.shuffle_frames_of_nseqs > 0:
      m = self.shuffle_frames_of_nseqs
      start -= start % m
      end = min(self.num_seqs, end + (m - (end % m)) % m)
    return start, end

  def shuffle_seqs(self, start, end):
    assert start < end
    assert self.is_cached(start, end)
    alloc_idx = self.alloc_interval_index(start)
    alloc_start, alloc_end, alloc_data = self.alloc_intervals[alloc_idx]
    assert start >= alloc_start
    assert end <= alloc_end
    rnd = numpy.random.RandomState(start)  # Some deterministic way to shuffle!
    num_frames = self.seq_start[end] - self.seq_start[start]
    assert num_frames > 0
    perm = rnd.permutation(num_frames)
    alloc_offset = self.seq_start[start] - self.seq_start[alloc_start]
    assert alloc_offset + num_frames <= alloc_data.shape[0]
    # Permute alloc_data.
    data = alloc_data[alloc_offset:alloc_offset + num_frames]
    alloc_data[alloc_offset:alloc_offset + num_frames] = data[perm]
    # Permute targets.
    targets = self.targets[self.seq_start[start]:self.seq_start[start] + num_frames]
    self.targets[self.seq_start[start]:self.seq_start[start] + num_frames] = targets[perm]

  def _load_seqs(self, start, end):
    """
    Load data sequences.
    As a side effect, will modify / fill-up:
      self.alloc_intervals
      self.targets

    :param int start: start sorted seq idx
    :param int end: end sorted seq idx
    """
    selection = self.insert_alloc_interval(start, end)
    assert len(selection) <= end - start, "DEBUG: more sequences requested (" + str(len(selection)) + ") as required (" + str(end-start) + ")"
    file_info = [ [] for l in xrange(len(self.files)) ]; """ :type: list[(int,int)] """
    # file_info[i] is (sorted seq idx from selection, real seq idx)
    for idc in selection:
      ids = self.seq_index[idc]
      file_info[self.file_index[ids]].append((idc,ids))
    for i in xrange(len(self.files)):
      if len(file_info[i]) == 0:
        continue
      print >> log.v4, "loading file", self.files[i]
      fin = h5py.File(self.files[i], 'r')
      inputs = fin['inputs'][...]; """ :type: numpy.ndarray """
      if 'targetClasses' in fin:
        targs = fin['targetClasses'][...]
      for idc, ids in file_info[i]:
        s = ids - self.file_start[i]
        p = self.file_seq_start[i][s]
        l = self.seq_lengths[ids]
        if 'targetClasses' in fin:
          y = targs[p : p + l]; """ :type: list[int] """
          self.targets[self.seq_start[idc] : self.seq_start[idc] + l] = y
        x = inputs[p : p + l]
        self._set_alloc_intervals_data(idc, data=x)
      fin.close()
    gc.collect()
    assert self.is_cached(start, end)

  def _set_alloc_intervals_data(self, idc, data):
    """
    :param int idc: index of sorted seq idx
    :param numpy.ndarray data: raw data
    """
    idi = self.alloc_interval_index(idc)
    assert idi >= 0
    o = self.seq_start[idc] - self.seq_start[self.alloc_intervals[idi][0]]
    l = data.shape[0]
    x = data
    x = self.preprocess(x)
    if self.window > 1:
      x = self.sliding_window(x)
    self.alloc_intervals[idi][2][o:o + l] = x

  def init_seq_order(self, epoch=None):
    """
    :type epoch: int|None
    Initialize lists:
      self.seq_index  # sorted seq idx
    """
    if epoch is not None:
      # Use some deterministic random seed.
      self.rnd_seed = epoch

    seq_index = list(range(self.num_seqs)); """ :type: list[int]. the real seq idx after sorting """
    if self.batching == 'default':
      pass  # Keep order as-is.
    elif self.batching == 'sorted':
      zipped = zip(seq_index, self.seq_lengths); """ :type: list[list[int]] """
      zipped.sort(key=lambda x: x[1])  # sort by length
      seq_index = [y[0] for y in zipped]
    elif self.batching == 'random':
      # Keep this deterministic! Use fixed seed.
      rnd = Random(self.rnd_seed)
      rnd.shuffle(seq_index)
    else:
      assert False, "invalid batching specified: " + self.batching

    if self.seq_index == seq_index:
      # Ignore if the order did not changed from the earlier order.
      # This avoids reloading of the cache.
      return

    if epoch is not None:
      # Give some hint to the user in case he is wondering why the cache is reloading.
      print >> log.v4, "Reinitialize dataset seq order for epoch %i." % epoch
    self.seq_index = seq_index

    self.init_seqs()

  def init_seqs(self):
    """
    Initialize lists:
      self.seq_start
      self.alloc_intervals
    """
    self.seq_start = [0]  # idx like in seq_index, *not* real idx
    num_cached = 0
    cached_bytes = 0
    for i in xrange(self.num_seqs):
      ids = self.seq_index[i]
      self.seq_start.append(self.seq_start[-1] + self.seq_lengths[ids])
      if self.isInitialized and i == num_cached:
        nbytes = self.seq_lengths[ids] * self.nbytes
        if self.cache_byte_size_limit_at_start >= cached_bytes + nbytes:
          num_cached = i + 1
          cached_bytes += nbytes
    self.alloc_intervals = \
      [(0, 0, numpy.zeros((1, self.num_inputs * self.window), dtype=theano.config.floatX)),
       (self.num_seqs, self.num_seqs, numpy.zeros((1, self.num_inputs * self.window), dtype=theano.config.floatX))]
    # self.alloc_intervals[i] is (idx start, idx end, data), where
    # idx start/end is the sorted seq idx start/end, end exclusive,
    # and data is a numpy.array.

    self.num_seqs_cached_at_start = num_cached
    self.cached_bytes_at_start = cached_bytes
    if num_cached > 0:
      assert self.isInitialized
      self.load_seqs(0, num_cached, free=False)

  def initialize(self):
    """
    Does the main initialization.
    :returns the cache byte size left over if we cache the whole dataset.
    """
    assert self.num_inputs > 0
    assert self.num_timesteps > 0
    self.isInitialized = True  # About

    self.nbytes = numpy.array([], dtype=theano.config.floatX).itemsize * (self.num_inputs * self.window + 1 + 1)

    if self.window > 1:
      if int(self.window) % 2 == 0: self.window += 1
      self.zpad = numpy.zeros((int(self.window) / 2, self.num_inputs), dtype=theano.config.floatX)
    self.targets = numpy.zeros((self.num_timesteps, ), dtype=theano.config.floatX) - 1  # Init with invalid values.

    self.init_seq_order()

    # Create Theano shared vars.
    self.x = theano.shared(numpy.zeros((1, 1, 1), dtype=theano.config.floatX), borrow=True)
    self.t = theano.shared(numpy.zeros((1, 1), dtype=theano.config.floatX), borrow=True)
    self.y = T.cast(self.t, 'int32')
    self.cp = theano.shared(numpy.zeros((1, 1), dtype=theano.config.floatX), borrow=True)
    self.c = T.cast(self.cp, 'int32')
    self.i = theano.shared(numpy.zeros((1, 1), dtype='int8'), borrow=True)

    # Calculate cache sizes.
    temp_cache_size_bytes = \
      max(0, self.cache_byte_size_total_limit) - self.cached_bytes_at_start
    extra_bytes = temp_cache_size_bytes if self.num_seqs_cached_at_start == self.num_seqs else 0
    self.cache_num_frames_free = temp_cache_size_bytes / self.nbytes

    print >> log.v4, "cached %i seqs" % self.num_seqs_cached_at_start, \
                     "%s GB" % (self.cached_bytes_at_start / float(1024 * 1024 * 1024)), \
                     ("(fully loaded, %s GB left over)" if extra_bytes else "(%s GB free)") % \
                     max(temp_cache_size_bytes / float(1024 * 1024 * 1024), 0)

    return extra_bytes

  def calculate_priori(self):
    priori = numpy.zeros((self.num_outputs,), dtype = theano.config.floatX)
    for i in xrange(self.num_seqs):
      self.load_seqs(i, i + 1)
      for t in self.targets[self.seq_start[i] : self.seq_start[i] + self.seq_lengths[self.seq_index[i]]]:
        priori[t] += 1
    return numpy.array(priori / self.num_timesteps, dtype = theano.config.floatX)

  def generate_batches(self, recurrent_net, batch_size, batch_step, max_seqs=-1):
    """
    :type recurrent_net: bool
    :type batch_size: int
    :type batch_step: int
    :type max_seqs: int
    :rtype: BatchSetGenerator
    """
    if max_seqs == -1: max_seqs = self.num_seqs
    if batch_step == -1: batch_step = batch_size

    def generator():
      batch = Batch([0, 0])
      s = 0
      while s < self.num_seqs:
        length = self.seq_lengths[self.seq_index[s]]
        if recurrent_net:
          if length > batch_size:
            print >> log.v4, "warning: sequence length (" + str(length) + ") larger than limit (" + str(batch_size) + ")"
          dt, ds = batch.try_sequence(length)
          if ds == 1:
            batch.add_sequence(length)
          else:
            if dt * ds > batch_size or ds > max_seqs:
              yield batch
              s = s - ds + min(batch_step, ds)
              batch = Batch([s, 0])
              length = self.seq_lengths[self.seq_index[s]]
            batch.add_sequence(length)
        else:
          while length > 0:
            num_frames = min(length, batch_size - batch.data_shape[0])
            if num_frames == 0 or batch.nseqs > max_seqs:
              yield batch
              batch = Batch([s, self.seq_lengths[self.seq_index[s]] - length])
              num_frames = min(length, batch_size)
            batch.add_frames(num_frames)
            length -= min(num_frames, batch_step)
          if s != self.num_seqs - 1: batch.nseqs += 1
        s += 1
      yield batch

    return BatchSetGenerator(self, generator())
