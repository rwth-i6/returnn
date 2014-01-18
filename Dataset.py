#! /usr/bin/python2.7

import numpy
import random
from numpy.lib.stride_tricks import as_strided as ast
from scipy.io.netcdf import NetCDFFile
from Log import log
from Util import cmd
import theano
import theano.tensor as T
import gc

class Dataset:
  def __init__(self, window = 1, cache_size = 0, chunking = "0", batching = 'default'):
    self.files = []
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
    self.seq_start = [0]
    self.seq_shift = [0]
    self.file_start = [0]
    self.file_seq_start = []
    self.timestamps = []
    self.file_index = []
    self.seq_lengths = []
    self.labels = []
    self.tags = []
    self.num_seqs = 0
    self.num_timesteps = 0    
    self.chunk_size = int(chunking.split(':')[0])
    if ':' in chunking:
      self.chunk_step = int(chunking.split(':')[1])
      assert self.chunk_step > 0, "chunking step must be positive" 
    else:
      self.chunk_step = self.chunk_size
    assert self.chunk_size >= 0, "chunk size must not be negative"
    
  def read_tags(self, filename):
    # super frustrating resignation solution because there is no way to read ncChars in python
    call = "ncdump -v seqTags " + filename + " | grep \"^  \\\"\" | cut -d \\\" -f 2"
    return cmd(call)

  def read_labels(self, filename):
    call = "ncdump -v labels " + filename + " | grep \"^  \\\"\" | cut -d \\\" -f 2"
    return cmd(call)
  
  def add_file(self, filename):
    labels = self.read_labels(filename)
    if not self.labels:
      self.labels = labels
    assert len(self.labels) == len(labels), "expected " + str(len(self.labels)) + " got " + str(len(labels))
    tags = self.read_tags(filename)
    self.files.append(filename)
    nc = NetCDFFile(filename, 'r')
    seq_start = [0]
    if nc.variables.has_key('times'):
      self.timestamps.extend(nc.variables['times'].data.tolist())
    for l,t in zip(nc.variables['seqLengths'].data.tolist(), tags):
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
    self.num_seqs += nseqs #nc.dimensions['numSeqs']
    self.file_index.extend([len(self.files) - 1] * nseqs)
    self.file_start.append(self.file_start[-1] + nseqs)
    self.num_timesteps = sum(self.seq_lengths) #nc.dimensions['numTimesteps']
    if self.num_inputs == 0:
      self.num_inputs = nc.dimensions['inputPattSize']
    assert self.num_inputs == nc.dimensions['inputPattSize']
    if self.num_outputs == 0:
      self.num_outputs = nc.dimensions['numLabels']
    assert self.num_outputs == nc.dimensions['numLabels'] 
    nc.close()
    
  def sliding_window(self, xr):
    x = numpy.concatenate([self.zpad, xr, self.zpad])
    return ast(x,
               shape = (x.shape[0] - self.window + 1, 1, self.window, self.num_inputs),
               strides = (x.strides[0], x.strides[1] * self.num_inputs) + x.strides).reshape((xr.shape[0], self.num_inputs * self.window))
    
  def preprocess(self, seq):
    return seq
    
  def _insert_alloc_interval(self, pos, value): 
    ci = self.alloc_intervals[pos][1]
    ni = self.alloc_intervals[pos + 1][0]
    xc = self.alloc_intervals[pos][2]
    xn = self.alloc_intervals[pos + 1][2]
    if value[0] == ci and value[1] == ni:
      self.alloc_intervals.insert(pos, [self.alloc_intervals[pos][0],
                                        self.alloc_intervals[pos + 1][1],
                                        numpy.concatenate([xc, numpy.zeros((self.seq_start[ni] - self.seq_start[ci], self.num_inputs * self.window), dtype = theano.config.floatX), xn])])
      del self.alloc_intervals[pos + 1]
      del self.alloc_intervals[pos + 1]
      return 0
    elif value[0] == ci:
      self.alloc_intervals.insert(pos, [self.alloc_intervals[pos][0],
                                        value[1],
                                        numpy.concatenate([xc, numpy.zeros((self.seq_start[value[1]] - self.seq_start[ci], self.num_inputs * self.window), dtype = theano.config.floatX)])])
      del self.alloc_intervals[pos + 1]
      return 0
    elif value[1] == ni:
      self.alloc_intervals.insert(pos + 1, [value[0],
                                            self.alloc_intervals[pos + 1][1],
                                            numpy.concatenate([numpy.zeros((self.seq_start[ni] - self.seq_start[value[0]], self.num_inputs * self.window), dtype = theano.config.floatX), xc])])
      del self.alloc_intervals[pos + 2]
      return 0
    else:
      self.alloc_intervals.insert(pos + 1, value + [numpy.zeros((self.seq_start[value[1]] - self.seq_start[value[0]], self.num_inputs * self.window), dtype = theano.config.floatX)])
      return 1
    
  def _remove_alloc_interval(self, pos, value):
    ci = self.alloc_intervals[pos][0]
    ni = self.alloc_intervals[pos][1]
    xi = self.alloc_intervals[pos][2]
    if value[0] == ci and value[1] == ni:
      del self.alloc_intervals[pos]
      return -1
    elif value[0] == ci:
      self.alloc_intervals.insert(pos, [value[1], ni, xi[self.seq_start[value[1]] - self.seq_start[ci]:]])
      del self.alloc_intervals[pos + 1]
      return 0
    elif value[1] == ni:
      self.alloc_intervals.insert(pos, [ci, value[0], xi[:self.seq_start[value[0]] - self.seq_start[ci]]])
      del self.alloc_intervals[pos + 1]
      return 0
    else:
      self.alloc_intervals.insert(pos, [value[1], ni, xi[self.seq_start[value[1]] - self.seq_start[ci]:]])
      self.alloc_intervals.insert(pos, [ci, value[0], xi[:self.seq_start[value[0]] - self.seq_start[ci]]])
      del self.alloc_intervals[pos + 2]
      return 1
    
  def _modify_alloc_intervals(self, start, end, invert):
    i = 0
    selection = []
    modify = self._insert_alloc_interval if invert else self._remove_alloc_interval
    while i < len(self.alloc_intervals) - invert:
      ni = self.alloc_intervals[i + invert][1 - invert]
      if ni >= self.num_cached:
        ci = max(self.num_cached, self.alloc_intervals[i][invert])
        flag = [ (start >= ci and start < ni), (end > ci and end <= ni), (ci < start and ni <= start) or (ci >= end and ni > end) ]
        if not flag[0] and not flag[1]:
          if not flag[2]:
            selection.extend(range(ci, ni))
            i += modify(i, [ci, ni])
        elif flag[1]:
          v = [ start if flag[0] else ci, end]
          selection.extend(range(v[0], v[1]))
          i += modify(i, v)
          break
        elif flag[0]:
          selection.extend(range(start, ni))
          i += modify(i, [start, ni])
      i += 1
    if self.alloc_intervals[0][0] != 0:
      self.alloc_intervals.insert(0, [0,0, numpy.zeros((1, self.num_inputs * self.window), dtype = theano.config.floatX)])
    if self.alloc_intervals[-1][1] != self.num_seqs:
      self.alloc_intervals.append([self.num_seqs,self.num_seqs, numpy.zeros((1, self.num_inputs * self.window), dtype = theano.config.floatX)])
    return selection
  
  def insert_alloc_interval(self, start, end):
    return self._modify_alloc_intervals(start, end, True)
  def remove_alloc_interval(self, start, end):
    return self._modify_alloc_intervals(start, end, False)
  
  def is_cached(self, start, end):
    s = 0
    e = len(self.alloc_intervals)
    while s < e:
      i = (s + e) / 2
      if self.alloc_intervals[i][0] <= start and start < self.alloc_intervals[i][1]:
        return self.alloc_intervals[i][0] < end and end <= self.alloc_intervals[i][1]
      elif self.alloc_intervals[i][0] <= start:
        if s == i: return False
        s = i
      else:
        if e == i: return False
        e = i
    return False
  
  def alloc_interval_index(self, ids):
    s = 0
    e = len(self.alloc_intervals)
    while s < e:
      i = (s + e) / 2
      if self.alloc_intervals[i][0] <= ids and ids < self.alloc_intervals[i][1]:
        return i
      elif self.alloc_intervals[i][0] <= ids:
        if s == i: return -1
        s = i
      else:
        if e == i: return -1
        e = i
    return -1
  
  def delete(self, nframes):
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
        deleted += sum([self.seq_lengths[self.seq_index[i]] for i in self.remove_alloc_interval(s, ai[1])]) # len(self.remove_alloc_interval(s, ai[1])) 
      i += 1
    return deleted
  
  def load_seqs(self, start, end, free = True, fill = True):
    if self.is_cached(start, end): return
    if self.cache_size > 0 and free:
      weight = self.seq_start[end] - self.seq_start[start] #sum([self.seq_lengths[self.seq_index[i]] for i in xrange(start, end)])
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
        self.load_seqs(start, end, False)
        if end == self.num_seqs:
          end = self.num_cached
          while end < start:
            weight = self.seq_lengths[self.seq_index[end]]
            if self.temp_cache_size - weight < 0:
              break
            self.temp_cache_size -= weight
            end += 1
          if end != self.num_cached:
            self.load_seqs(self.num_cached, end, False)
        return
    selection = self.insert_alloc_interval(start, end)
    assert len(selection) <= end - start, "DEBUG: more sequences requested (" + str(len(selection)) + ") as required (" + str(end-start) + ")"
    file_info = [ [] for l in xrange(len(self.files)) ]
    for idc in selection:
      ids = self.seq_index[idc]
      file_info[self.file_index[ids]].append((idc,ids))
    for i in xrange(len(self.files)):
      if len(file_info[i]) == 0:
        continue
      print >> log.v4, "loading file", self.files[i]
      nc = NetCDFFile(self.files[i], 'r')
      inputs = nc.variables['inputs'].data
      targs = nc.variables['targetClasses'].data
      for idc, ids in file_info[i]:
        s = ids - self.file_start[i]
        p = self.file_seq_start[i][s]
        idi = self.alloc_interval_index(idc)
        o = self.seq_start[idc] - self.seq_start[self.alloc_intervals[idi][0]]
        l = self.seq_lengths[ids]
        y = targs[p : p + l]
        x = inputs[p : p + l]
        x = self.preprocess(x)
        if self.window > 1:
          x = self.sliding_window(x) 
        self.alloc_intervals[idi][2][o:o + l] = x
        self.targets[self.seq_start[idc] : self.seq_start[idc] + l] = y
      nc.close()
    gc.collect()
    
  def set_batching(self, batching):
    self.batching = batching
    self.seq_index = range(self.num_seqs)
    if self.batching == 'sorted':
      zipped = zip(self.seq_index, self.seq_lengths)
      zipped.sort(key = lambda x:x[1])
      self.seq_index = [ y[0] for y in zipped ]
    elif self.batching == 'random':
      random.shuffle(self.seq_index)
    else: assert self.batching == 'default', "invalid batching specified: " + self.batching
    self.seq_start = [0]
    self.cached_bytes = 0
    num_cached = self.num_seqs
    for i in xrange(self.num_seqs):
      ids = self.seq_index[i]
      self.seq_start.append(self.seq_start[-1] + self.seq_lengths[ids])
      nbytes = self.seq_lengths[ids] * self.nbytes
      if num_cached == self.num_seqs:
        if self.cache_size > 0 and self.cached_bytes + nbytes > self.cache_size:
          num_cached = i
        else: self.cached_bytes += nbytes
    self.temp_cache_size += self.cached_bytes
    self.alloc_intervals = [[0,0,numpy.zeros((1, self.num_inputs * self.window), dtype = theano.config.floatX)],
                            [self.num_seqs,self.num_seqs, numpy.zeros((1, self.num_inputs * self.window), dtype = theano.config.floatX)]]
    self.temp_cache_size -= self.cached_bytes
    if num_cached > 0:
      self.load_seqs(0, num_cached, free = False)
    self.num_cached = num_cached
 
  def initialize(self):
    self.nbytes = numpy.array([], dtype=theano.config.floatX).itemsize * (self.num_inputs * self.window + 1 + 1)
    if self.window > 1:
      if int(self.window) % 2 == 0: self.window += 1
      self.zpad = numpy.zeros((int(self.window) / 2, self.num_inputs), dtype = theano.config.floatX)
    self.targets = numpy.zeros((self.num_timesteps, ), dtype = theano.config.floatX)
    self.temp_cache_size += self.cache_size
    self.set_batching(self.batching)
    self.temp_cache_size += self.cache_size - self.cached_bytes
    print >> log.v4, "cached", self.num_cached, "seqs", self.cached_bytes / float(1024 * 1024 * 1024), "GB (" + str(max(self.temp_cache_size / float(1024 * 1024 * 1024), 0)), "GB temp)"
    extra = self.temp_cache_size if self.num_cached == self.num_seqs else 0
    self.temp_cache_size = self.temp_cache_size / self.nbytes
    self.x = theano.shared(numpy.zeros((1, 1, 1), dtype = theano.config.floatX), borrow=True)
    self.t = theano.shared(numpy.zeros((1, 1), dtype = theano.config.floatX), borrow=True)
    self.y = T.cast(self.t, 'int32')
    self.i = theano.shared(numpy.zeros((1, 1), dtype = 'int8'), borrow=True)
    self.theano_init = True
    return extra
