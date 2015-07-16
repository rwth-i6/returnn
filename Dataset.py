#! /usr/bin/python2.7

__author__ = "Patrick Doetsch"
__copyright__ = "Copyright 2015"
__credits__ = ["Patrick Doetsch", "Paul Voigtlaender"]
__license__ = "RWTHASR"
__version__ = "0.9"
__maintainer__ = "Patrick Doetsch"
__email__ = "doetsch@i6.informatik.rwth-aachen.de"

from threading import RLock
from random import Random

import sys
import numpy
import theano

from Log import log
from EngineBatch import Batch, BatchSetGenerator


class Dataset(object):

  @classmethod
  def from_config(cls, config, chunking=None, seq_ordering=None, shuffle_frames_of_nseqs=None, **kwargs):
    """
    :type config: Config.Config
    :type chunking: str
    :type seq_ordering: str
    :rtype: Dataset
    """
    window = config.int('window', 1)
    if chunking is None:
      chunking = config.value("chunking", "0")
    if seq_ordering is None:
      seq_ordering = config.value("batching", 'default')
    if shuffle_frames_of_nseqs is None:
      shuffle_frames_of_nseqs = config.int('shuffle_frames_of_nseqs', 0)
    return cls(window=window, chunking=chunking,
               seq_ordering=seq_ordering, shuffle_frames_of_nseqs=shuffle_frames_of_nseqs,
               **kwargs)

  def __init__(self, window=1, chunking="0", seq_ordering='default', shuffle_frames_of_nseqs=0):
    self.lock = RLock()  # Used when manipulating our data potentially from multiple threads.
    self.num_inputs = 0
    self.num_outputs = 0
    self.window = window
    self.seq_ordering = seq_ordering  # "default", "sorted" or "random". See self.get_seq_order_for_epoch().
    self.timestamps = []
    self.labels = {}; """ :type: list[str] """
    self.nbytes = 0
    self.num_running_chars = 0  # CTC running chars.
    self._num_timesteps = 0
    self._num_seqs = 0
    self.chunk_size = int(chunking.split(':')[0])
    if ':' in chunking:
      self.chunk_step = int(chunking.split(':')[1])
      assert self.chunk_step > 0, "chunking step must be positive"
    else:
      self.chunk_step = self.chunk_size
    assert self.chunk_size >= 0, "chunk size must not be negative"
    self.shuffle_frames_of_nseqs = shuffle_frames_of_nseqs

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

  def is_cached(self, start, end):
    """
    :param int start: like in load_seqs(), sorted seq idx
    :param int end: like in load_seqs(), sorted seq idx
    :rtype: bool
    :returns whether we have the full range (start,end) of sorted seq idx.
    """
    if start == end: return True  # Empty.
    assert start < end
    return False

  def get_seq_length(self, sorted_seq_idx):
    """
    :type sorted_seq_idx: int
    :rtype: int
    """
    raise NotImplementedError

  def get_num_timesteps(self):
    assert self._num_timesteps > 0
    return self._num_timesteps

  def get_num_codesteps(self):
    assert self._num_codesteps > 0
    return self._num_codesteps

  def load_seqs(self, start, end):
    """
    Load data sequences, such that self.get_data() & friends can return the data.
    :param int start: start sorted seq idx
    :param int end: end sorted seq idx
    """
    assert start >= 0
    assert start <= end
    if self.is_cached(start, end): return

    if self.shuffle_frames_of_nseqs > 0:
      # We always load N seqs at once and shuffle all their frames.
      start, end = self._load_seqs_superset(start, end)
      self._load_seqs(start, end)
      while start < end:
        self._shuffle_seqs(start, start + self.shuffle_frames_of_nseqs)
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
    thus start/end will be aligned to self.shuffle_frames_of_nseqs.
    """
    assert start <= end
    assert start < self.num_seqs
    if self.shuffle_frames_of_nseqs > 0:
      m = self.shuffle_frames_of_nseqs
      start -= start % m
      end += (m - (end % m)) % m
    return start, end

  def _shuffle_seqs(self, start, end):
    raise NotImplementedError

  def _load_seqs(self, start, end):
    """
    Load data sequences.
    :param int start: start sorted seq idx
    :param int end: end sorted seq idx. might be outside
    """
    raise NotImplementedError

  def get_seq_order_for_epoch(self, epoch, num_seqs, get_seq_len=None):
    """
    :returns the order for the given epoch.
    This is mostly a static method, except that is depends on the configured type of ordering,
     such as 'default' (= as-is), 'sorted' or 'random'. 'sorted' also uses the sequence length.
    :param int epoch: for 'random', this determines the random seed
    :type num_seqs: int
    :param get_seq_len: function (originalSeqIdx: int) -> int
    :rtype: list[int]
    """
    assert num_seqs > 0
    seq_index = list(range(num_seqs)); """ :type: list[int]. the real seq idx after sorting """
    if self.seq_ordering == 'default':
      pass  # Keep order as-is.
    elif self.seq_ordering == 'sorted':
      assert get_seq_len
      seq_index.sort(key=lambda x: get_seq_len(x)[0])  # sort by length
    elif self.seq_ordering == 'random':
      # Keep this deterministic! Use fixed seed.
      rnd_seed = epoch or 1
      rnd = Random(rnd_seed)
      rnd.shuffle(seq_index)
    else:
      assert False, "invalid batching specified: " + self.seq_ordering
    return seq_index

  def init_seq_order(self, epoch=None):
    """
    :type epoch: int|None
    This is called when we start a new epoch, or at initialization.
    Call this when you reset the seq list.
    """
    pass

  def initialize(self):
    """
    Does the main initialization before it can be used.
    This needs to be called before self.load_seqs() can be used.
    """
    # We expect that the following attributes are already set elsewhere, by a derived class.
    assert self.num_inputs > 0
    assert self.num_outputs
    assert self.window > 0

    if int(self.window) % 2 == 0:
      self.window += 1

    self.nbytes = numpy.array([], dtype=theano.config.floatX).itemsize * (self.num_inputs * self.window + 1 + 1)

    if self.window > 1:
      self.zpad = numpy.zeros((int(self.window) / 2, self.num_inputs), dtype=theano.config.floatX)

    self.init_seq_order()

  def get_times(self, sorted_seq_idx):
    raise NotImplementedError

  def get_targets(self, target, sorted_seq_idx):
    raise NotImplementedError

  def get_ctc_targets(self, sorted_seq_idx):
    raise NotImplementedError

  def get_tag(self, sorted_seq_idx):
    return "seq-%i" % sorted_seq_idx

  def has_ctc_targets(self):
    return False

  def get_max_ctc_length(self):
    return 0

  def get_complete_frac(self, seq_idx):
    """
    :return: Returns a fraction (float in [0,1], always > 0) of how far we have advanced
      for this seq in the dataset.
      This does not have to be exact. This is only for the user.
    """
    assert self.num_seqs > 0
    return float(seq_idx + 1) / self.num_seqs

  @property
  def num_seqs(self):
    raise NotImplementedError

  def get_target_dim(self, target):
    raise NotImplementedError

  def is_less_than_num_seqs(self, n):
    # We keep this dynamic so that other implementations which don't know the num of seqs
    # in advance can handle this somehow.
    return n < self.num_seqs

  def calculate_priori(self):
    priori = numpy.zeros((self.num_outputs,), dtype=theano.config.floatX)
    i = 0
    while self.is_less_than_num_seqs(i):
      self.load_seqs(i, i + 1)
      for t in self.get_targets(i):
        priori[t] += 1
      i += 1
    return numpy.array(priori / self.get_num_timesteps(), dtype=theano.config.floatX)

  def iterate_seqs(self):
    """
    Takes chunking into consideration.
    :return: index, and seq start, seq end
    :rtype: list[(int,int,int)]
    """
    s = 0
    while self.is_less_than_num_seqs(s):
      length = self.get_seq_length(s)
      if self.chunk_size == 0:
        yield (s, numpy.array([0,0]), length)
      else:
        t = 0
        while t < length[0]:
          l = min(t + self.chunk_size, length[0])
          yield (s, numpy.array([t,t]), numpy.array([l,l]))
          t += self.chunk_step
      s += 1

  def _generate_batches(self, recurrent_net, batch_size, max_seqs=-1):
    """
    :param bool recurrent_net: If True, the batch might have a batch seq dimension > 1.
      Otherwise, the batch seq dimension is always 1 and multiple seqs will be concatenated.
    :param int batch_size: Max number of frames in one batch.
    :param int max_seqs: Max number of seqs per batch.
    """
    if batch_size == 0: batch_size = sys.maxint
    assert batch_size > 0
    if max_seqs == -1: max_seqs = float('inf')
    assert max_seqs > 0
    if not recurrent_net:
      assert self.chunk_size == 0, "Chunking not supported for non-recurrent net"

    batch = Batch()
    for seq_idx, t_start, t_end in self.iterate_seqs():
      if recurrent_net:
        length = t_end - t_start
        if max(length) > batch_size:
          print >> log.v4, "warning: sequence length (%i) larger than limit (%i)" % (max(length), batch_size)
        dt, ds = batch.try_sequence_as_slice(length)
        if ds > 1 and (dt * ds > batch_size or ds > max_seqs):
          yield batch
          batch = Batch()
        batch.add_sequence_as_slice(seq_idx=seq_idx, seq_start_frame=t_start, length=length)
      else:  # Not recurrent.
        while t_start[0] < t_end[0]:
          length = t_end[0] - t_start[0]
          num_frames = min(length[0], batch_size - batch.get_all_slices_num_frames())
          batch.add_frames(seq_idx=seq_idx, seq_start_frame=t_start, length=num_frames)
          if batch.get_all_slices_num_frames() >= batch_size or batch.get_num_seqs() > max_seqs:
            yield batch
            batch = Batch()
          t_start += num_frames

    if batch.get_all_slices_num_frames() > 0:
      yield batch

  def generate_batches(self, recurrent_net, batch_size, max_seqs=-1):
    """
    :type recurrent_net: bool
    :type batch_size: int
    :type max_seqs: int
    :rtype: BatchSetGenerator
    """
    return BatchSetGenerator(self, self._generate_batches(recurrent_net, batch_size, max_seqs))


