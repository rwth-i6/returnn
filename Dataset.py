#! /usr/bin/python2.7

__author__ = "Patrick Doetsch"
__copyright__ = "Copyright 2015"
__credits__ = ["Patrick Doetsch", "Paul Voigtlaender"]
__license__ = "RWTHASR"
__version__ = "0.9"
__maintainer__ = "Patrick Doetsch"
__email__ = "doetsch@i6.informatik.rwth-aachen.de"

import numpy
import random
import h5py
from Log import log
import theano
import theano.tensor as T
import gc
from random import Random


class Dataset(object):
  @classmethod
  def load_data(cls, config, cache_size, files_config_key=None, chunking="chunking", batching="batching"):
    """
    :type config: Config.Config
    :type cache_size: int
    :type files_config_key: str
    :type chunking: str
    :type batching: str
    """
    window = config.int('window', 1)
    if chunking == "chunking": chunking = config.value("chunking", "0")
    if batching == "batching": batching = config.value("batching", 'default')
    data = cls(window, cache_size, chunking, batching)
    extra = 0
    if files_config_key:
      if config.has(files_config_key):
        for f in config.list(files_config_key):
          data.add_file(f)
        extra = data.initialize()
      else:
        return None, 0
    return data, extra

  def __init__(self, window=1, cache_size=0, chunking="0", batching='default'):
    self.files = []; """ :type: list[str] """
    self.num_inputs = 0
    self.num_outputs = 0
    self.window = window
    self.batching = batching
    temp_cache_amount = 0.5
    self.temp_cache_size = 1
    self.cache_size = int((1.0 - temp_cache_amount) * cache_size)
    if self.cache_size < 0: self.cache_size = 1
    else: self.temp_cache_size = cache_size - self.cache_size
    self.num_cached = 0
    self.cached_bytes = 0
    self.seq_start = [0]  # uses sorted seq idx, see set_batching()
    self.seq_shift = [0]
    self.rnd = Random(0)  # Use fixed seed. Call rnd.seed() later for a different one. Make this deterministic!
    self.rnd_seed = 0  # See init_seq_order().
    self.file_start = [0]
    self.file_seq_start = []; """ :type: list[list[int]] """
    self.timestamps = []
    self.file_index = []; """ :type: list[int] """
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
    i = 0
    selection = []; """ :type: list[int] """
    modify = self._insert_alloc_interval if invert else self._remove_alloc_interval
    while i < len(self.alloc_intervals) - invert:
      ni = self.alloc_intervals[i + invert][1 - invert]
      if ni >= self.num_cached:
        ci = max(self.num_cached, self.alloc_intervals[i][invert])
        flag = ( (ci <= start < ni), (ci < end <= ni), (ci < start and ni <= start) or (ci >= end and ni > end) )
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
    s = 0
    e = len(self.alloc_intervals)
    while s < e:
      i = (s + e) / 2
      if self.alloc_intervals[i][0] <= start < self.alloc_intervals[i][1]:
        return self.alloc_intervals[i][0] < end <= self.alloc_intervals[i][1]
      elif self.alloc_intervals[i][0] <= start:
        if s == i: return False
        s = i
      else:
        if e == i: return False
        e = i
    return False

  def alloc_interval_index(self, ids):
    """
    :param int ids: index of sorted seq idx
    :return index in self.alloc_intervals
    :rtype: int
    """
    s = 0
    e = len(self.alloc_intervals)
    while s < e:
      i = (s + e) / 2
      if self.alloc_intervals[i][0] <= ids < self.alloc_intervals[i][1]:
        return i
      elif self.alloc_intervals[i][0] <= ids:
        if s == i: return -1
        s = i
      else:
        if e == i: return -1
        e = i
    return -1

  def delete(self, nframes):
    """
    :param int nframes: how much frames to delete max.
      Note that this limit is not strict. We can end up
      deleting more than nframes.
    :return: number of frames deleted
    :rtype: int
    """
    start = self.num_seqs
    deleted = 0
    end = 0
    i = 0
    while deleted < nframes and i < len(self.alloc_intervals):
      ai = self.alloc_intervals[i]
      if ai[1] > self.num_cached and ai[0] < ai[1]:
        s = max(ai[0], self.num_cached)
        start = min(start, s)
        end = max(end, ai[1])
        deleted += sum([self.seq_lengths[self.seq_index[i]] for i in self.remove_alloc_interval(s, ai[1])])
      i += 1
    return deleted

  def load_seqs(self, start, end, free=True, fill=True):
    """
    Load data sequences.
    As a side effect, will modify / fill-up:
      self.alloc_intervals
      self.targets

    :param int start: start sorted seq idx
    :param int end: end sorted seq idx
    :param bool free: free
    :param bool fill: fill
    """
    if self.is_cached(start, end): return
    if self.cache_size > 0 and free:
      weight = self.seq_start[end] - self.seq_start[start]
      if self.temp_cache_size < weight:
        self.temp_cache_size += self.delete(weight - self.temp_cache_size)
        gc.collect()
      self.temp_cache_size -= weight
      if fill:
        self.temp_cache_size += self.delete(self.num_timesteps)
        gc.collect()
        while end < self.num_seqs:
          ids = self.seq_index[end]
          weight = self.seq_lengths[ids]
          if self.temp_cache_size - weight < 0:
            break
          self.temp_cache_size -= weight
          end += 1
        self.load_seqs(start, end, free=False)
        if end == self.num_seqs:
          end = self.num_cached
          while end < start:
            weight = self.seq_lengths[self.seq_index[end]]
            if self.temp_cache_size - weight < 0:
              break
            self.temp_cache_size -= weight
            end += 1
          if end != self.num_cached:
            self.load_seqs(self.num_cached, end, free=False)
        return
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
    if epoch is not None: self.rnd_seed = epoch  # Use some deterministic random seed.
    self.seq_index = list(range(self.num_seqs)); """ :type: list[int]. the real seq idx after sorting """
    if self.batching == 'default':
      pass  # Keep order as-is.
    elif self.batching == 'sorted':
      zipped = zip(self.seq_index, self.seq_lengths); """ :type: list[list[int]] """
      zipped.sort(key = lambda x: x[1])  # sort by length
      self.seq_index = [y[0] for y in zipped]
    elif self.batching == 'random':
      # Keep this deterministic!
      self.rnd.seed(self.rnd_seed)
      self.rnd.shuffle(self.seq_index)
    else:
      assert False, "invalid batching specified: " + self.batching
    self.init_seqs()

  def init_seqs(self):
    """
    Initialize lists:
      self.seq_start
      self.alloc_intervals
    """
    self.seq_start = [0]  # idx like in seq_index, *not* real idx
    self.transcription_start = [0]
    self.cached_bytes = 0
    num_cached = -1
    for i in xrange(self.num_seqs):
      ids = self.seq_index[i]
      self.seq_start.append(self.seq_start[-1] + self.seq_lengths[ids])
      if self.isInitialized:
        nbytes = self.seq_lengths[ids] * self.nbytes
        if num_cached < 0:
          if 0 < self.cache_size < self.cached_bytes + nbytes:
            num_cached = i
          else:
            self.cached_bytes += nbytes
    self.temp_cache_size += self.cached_bytes
    self.alloc_intervals = \
      [(0, 0, numpy.zeros((1, self.num_inputs * self.window), dtype=theano.config.floatX)),
       (self.num_seqs, self.num_seqs, numpy.zeros((1, self.num_inputs * self.window), dtype=theano.config.floatX))]
    # self.alloc_intervals[i] is (idx start, idx end, data), where
    # idx start/end is the sorted seq idx start/end, end exclusive,
    # and data is a numpy.array.
    self.temp_cache_size -= self.cached_bytes
    if self.isInitialized and num_cached > 0:
      self.load_seqs(0, num_cached, free=False)
      self.num_cached = num_cached

  def initialize(self):
    self.isInitialized = True
    self.nbytes = numpy.array([], dtype=theano.config.floatX).itemsize * (self.num_inputs * self.window + 1 + 1)
    if self.window > 1:
      if int(self.window) % 2 == 0: self.window += 1
      self.zpad = numpy.zeros((int(self.window) / 2, self.num_inputs), dtype = theano.config.floatX)
    self.targets = numpy.zeros((self.num_timesteps, ), dtype = theano.config.floatX)
    self.temp_cache_size += self.cache_size
    self.init_seq_order()
    self.temp_cache_size += self.cache_size - self.cached_bytes
    print >> log.v4, "cached", self.num_cached, "seqs", self.cached_bytes / float(1024 * 1024 * 1024), "GB (" + str(max(self.temp_cache_size / float(1024 * 1024 * 1024), 0)), "GB temp)"
    extra = self.temp_cache_size if self.num_cached == self.num_seqs else 0
    self.temp_cache_size /= self.nbytes
    self.x = theano.shared(numpy.zeros((1, 1, 1), dtype = theano.config.floatX), borrow=True)
    self.t = theano.shared(numpy.zeros((1, 1), dtype = theano.config.floatX), borrow=True)
    self.y = T.cast(self.t, 'int32')
    self.cp = theano.shared(numpy.zeros((1, 1), dtype = theano.config.floatX), borrow=True)
    self.c = T.cast(self.cp, 'int32')
    self.i = theano.shared(numpy.zeros((1, 1), dtype = 'int8'), borrow=True)
    self.theano_init = True
    return extra

  def calculate_priori(self):
    priori = numpy.zeros((self.num_outputs,), dtype = theano.config.floatX)
    for i in xrange(self.num_seqs):
      self.load_seqs(i, i + 1)
      for t in self.targets[self.seq_start[i] : self.seq_start[i] + self.seq_lengths[self.seq_index[i]]]:
        priori[t] += 1
    return numpy.array(priori / self.num_timesteps, dtype = theano.config.floatX)
