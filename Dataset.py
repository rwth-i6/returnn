#! /usr/bin/python2.7

from __future__ import print_function

__author__ = "Patrick Doetsch"
__copyright__ = "Copyright 2015"
__credits__ = ["Patrick Doetsch", "Paul Voigtlaender"]
__license__ = "RWTHASR"
__version__ = "0.9"
__maintainer__ = "Patrick Doetsch"
__email__ = "doetsch@i6.informatik.rwth-aachen.de"

from threading import RLock
from random import Random, random

import sys
import os
import numpy
import functools

from Log import log
from EngineBatch import Batch, BatchSetGenerator
from Util import try_run, NumbersDict, unicode


class Dataset(object):

  @staticmethod
  def kwargs_update_from_config(config, kwargs):
    """
    :type config: Config.Config
    :type kwargs: dict[str]
    """
    def set_or_remove(key, value):
      if key in kwargs and kwargs[key] is None:
        del kwargs[key]
      if value is not None and key not in kwargs:
        kwargs[key] = value
    set_or_remove("window", config.int('window', 0) or None)
    set_or_remove("context_window", config.typed_value("context_window"))
    set_or_remove("chunking", config.opt_typed_value("chunking", None))
    set_or_remove("seq_ordering", config.value("batching", None))
    set_or_remove("shuffle_frames_of_nseqs", config.int('shuffle_frames_of_nseqs', 0) or None)
    set_or_remove("min_chunk_size", config.int('min_chunk_size', 0) or None)

  @classmethod
  def from_config(cls, config, **kwargs):
    """
    :type config: Config.Config
    :param dict[str] kwargs: passed on to __init__
    :rtype: Dataset
    """
    cls.kwargs_update_from_config(config, kwargs)
    return cls(**kwargs)

  def __init__(self, name=None,
               window=1, context_window=None, chunking=None,
               seq_ordering='default', partition_epoch=None,
               shuffle_frames_of_nseqs=0, min_chunk_size=0,
               estimated_num_seqs=None,):
    """
    :param str name: e.g. "train" or "eval"
    :param int window: features will be of dimension window * feature_dim, as we add a context-window around.
      not all datasets support this option.
    :param None|int|dict|NumbersDict context_window: will add this context for each chunk
    :param None|str|int|(int,int)|dict|(dict,dict) chunking: "chunk_size:chunk_step"
    :param str seq_ordering: "batching"-option in config. e.g. "default", "sorted" or "random".
      See self.get_seq_order_for_epoch() for more details.
    :param int|None partition_epoch:
    :param int shuffle_frames_of_nseqs: shuffles the frames. not always supported
    :param None|int estimated_num_seqs: for progress reporting in case the real num_seqs is unknown
    """
    self.name = name or ("dataset_id%s" % id(self))
    self.lock = RLock()  # Used when manipulating our data potentially from multiple threads.
    self.num_inputs = 0  # usually not used, but num_outputs instead, which is more generic
    self.num_outputs = None; " :type: dict[str,(int,int)] "  # tuple is num-classes, len(shape).
    self.window = window
    self.seq_ordering = seq_ordering  # "default", "sorted" or "random". See self.get_seq_order_for_epoch().
    self.partition_epoch = partition_epoch or 1
    self.timestamps = None
    self.labels = {}; """ :type: dict[str,list[str]] """
    self.nbytes = 0
    self.num_running_chars = 0  # CTC running chars.
    self._num_timesteps = 0
    self._num_codesteps = None; " :type: int "  # Num output frames, could be different from input, seq2seq, ctc.
    self._num_seqs = 0
    self._estimated_num_seqs = estimated_num_seqs
    self.min_chunk_size = min_chunk_size
    if isinstance(chunking, str):
      if ":" in chunking:
        chunking = tuple(map(int, chunking.split(":")))
      else:
        chunking = int(chunking)
    if not isinstance(chunking, (tuple, list)):
      chunking = (chunking, None)
    chunk_size, chunk_step = chunking
    if chunk_size is None:
      chunk_size = 0
    assert isinstance(chunk_size, (int, dict, NumbersDict))
    chunk_size = NumbersDict(chunk_size)
    assert chunk_size == 0 or chunk_size.min_value() > 0, "chunk size must not be negative"
    self.chunk_size = chunk_size
    if chunk_step in (None, 0):
      chunk_step = self.chunk_size
    assert isinstance(chunk_step, (int, dict, NumbersDict))
    chunk_step = NumbersDict(chunk_step)
    if self.chunk_size != 0:
      assert sorted(chunk_step.keys()) == sorted(chunk_size.keys())
      assert chunk_step.max_value() > 0, "chunking step must be positive (for some key)"
    self.chunk_step = chunk_step
    if context_window is None:
      context_window = NumbersDict(0)
    elif isinstance(context_window, int):
      context_window = NumbersDict(broadcast_value=0, numbers_dict={"data": context_window})
    elif isinstance(context_window, dict):
      context_window = NumbersDict(broadcast_value=0, numbers_dict=context_window)
    assert isinstance(context_window, NumbersDict)
    self.context_window = context_window
    self.shuffle_frames_of_nseqs = shuffle_frames_of_nseqs
    self.epoch = None

  def __repr__(self):
    return "<%s %r epoch=%s>" % (
      self.__class__.__name__,
      getattr(self, "name", "<unknown>"),
      getattr(self, "epoch", "<unknown>"))

  def sliding_window(self, xr):
    """
    :type xr: numpy.ndarray
    :rtype: numpy.ndarray
    """
    from numpy.lib.stride_tricks import as_strided
    x = numpy.concatenate([self.zpad, xr, self.zpad])
    return as_strided(
      x,
      shape=(x.shape[0] - self.window + 1, 1, self.window, self.num_inputs),
      strides=(x.strides[0], x.strides[1] * self.num_inputs) + x.strides
      ).reshape((xr.shape[0], self.num_inputs * self.window))

  def preprocess(self, seq):
    """
    :type seq: numpy.ndarray
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
    if start == end:
      return True  # Empty.
    assert start < end
    return False

  def get_seq_length_2d(self, sorted_seq_idx):
    """
    :type sorted_seq_idx: int
    :rtype: numpy.array[int,int]
    :returns the len of the input features and the len of the target sequence.
    For multiple target seqs, it is expected that they all have the same len.
    We support different input/target len for seq2seq/ctc and other models.
    Note: This is deprecated, better use get_seq_length().
    """
    l = self.get_seq_length(sorted_seq_idx)
    targets = self.get_target_list()
    if targets:
      return numpy.array([l["data"], l[targets[0]]])
    else:
      return numpy.array([l["data"], 0])

  def get_seq_length(self, seq_idx):
    """
    :rtype: NumbersDict
    """
    input_len, output_len = self.get_seq_length_2d(seq_idx)
    d = {"data": input_len}
    d.update({k: output_len for k in self.get_target_list()})
    return NumbersDict(d)

  def get_num_timesteps(self):
    assert self._num_timesteps > 0
    return self._num_timesteps

  def get_num_codesteps(self):
    if self._num_codesteps is None:
      return [self.get_num_timesteps()]
    return self._num_codesteps

  def load_seqs(self, start, end):
    """
    Load data sequences, such that self.get_data() & friends can return the data.
    :param int start: start sorted seq idx, inclusive
    :param int end: end sorted seq idx, exclusive
    """
    assert start >= 0
    assert start <= end
    if self.is_cached(start, end): return

    if self.shuffle_frames_of_nseqs > 0:
      # We always load N seqs at once and shuffle all their frames.
      start, end = self._get_load_seqs_superset(start, end)
      self._load_seqs(start, end)
      while start < end:
        self._shuffle_frames_in_seqs(start, start + self.shuffle_frames_of_nseqs)
        start += self.shuffle_frames_of_nseqs
    else:
      self._load_seqs(start, end)

  def _get_load_seqs_superset(self, start, end):
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

  def _shuffle_frames_in_seqs(self, start, end):
    raise NotImplementedError

  def _load_seqs(self, start, end):
    """
    Load data sequences.
    :param int start: inclusive seq idx start
    :param int end: exclusive seq idx end. can be more than num_seqs
    If end > num_seqs, will not load them.
    """
    raise NotImplementedError

  def get_seq_order_for_epoch(self, epoch, num_seqs, get_seq_len=None):
    """
    Returns the order of the given epoch.
    This is mostly a static method, except that is depends on the configured type of ordering,
    such as 'default' (= as-is), 'sorted' or 'random'. 'sorted' also uses the sequence length.

    :param int epoch: for 'random', this determines the random seed
    :param int num_seqs:
    :param ((int) -> int)|None get_seq_len: function (originalSeqIdx: int) -> int
    :return: the order for the given epoch. such that seq_idx -> underlying idx
    :rtype: list[int]
    """
    partition_epoch = self.partition_epoch or 1
    if not epoch:
      epoch = 1
    full_epoch = epoch
    if partition_epoch > 1:
      full_epoch = (epoch - 1) // partition_epoch + 1
    assert num_seqs > 0
    seq_index = list(range(num_seqs)); """ :type: list[int]. the real seq idx after sorting """
    if self.seq_ordering == 'default':
      pass  # Keep order as-is.
    elif self.seq_ordering == 'sorted':
      assert get_seq_len
      seq_index.sort(key=get_seq_len)  # sort by length, starting with shortest
    elif self.seq_ordering == "sorted_reverse":
      assert get_seq_len
      seq_index.sort(key=get_seq_len, reverse=True)  # sort by length, in reverse, starting with longest
    elif self.seq_ordering.startswith('laplace'):
      assert get_seq_len
      tmp = self.seq_ordering.split(':')
      bins = int(tmp[1]) if len(tmp) > 1 else 2
      nth = int(tmp[2]) if len(tmp) > 2 else 1
      rnd_seed = ((full_epoch - 1) // nth + 1) if full_epoch else 1
      rnd = Random(rnd_seed)
      rnd.shuffle(seq_index)
      out_index = []
      for i in range(bins):
        if i == bins - 1:
          part = seq_index[i * len(seq_index) // bins:][:]
        else:
          part = seq_index[i * len(seq_index) // bins:(i + 1) * len(seq_index) // bins][:]
        part.sort(key=get_seq_len, reverse=(i % 2 == 1))
        out_index += part
      seq_index = out_index
    elif self.seq_ordering.startswith('random'):
      tmp = self.seq_ordering.split(':')
      nth = int(tmp[1]) if len(tmp) > 1 else 1
      # Keep this deterministic! Use fixed seed.
      rnd_seed = (full_epoch - 1) / nth + 1
      rnd = Random(rnd_seed)
      rnd.shuffle(seq_index)
    else:
      assert False, "invalid batching specified: " + self.seq_ordering
    if partition_epoch > 1:
      current_partition = ((epoch or 1) - 1) % partition_epoch
      seqs_per_epoch = num_seqs // partition_epoch
      partition_sizes = ([seqs_per_epoch + 1] * (num_seqs % partition_epoch) +
                         [seqs_per_epoch] * (partition_epoch - num_seqs % partition_epoch))
      assert sum(partition_sizes) == num_seqs and len(partition_sizes) == partition_epoch
      partitions = functools.reduce(lambda a, x: a + [a[-1] + x], partition_sizes, [0])  # cumulative sum
      assert len(partitions) == partition_epoch + 1
      seq_index = seq_index[partitions[current_partition]:partitions[current_partition + 1]]
      assert len(seq_index) == partition_sizes[current_partition]
    return seq_index

  def init_seq_order(self, epoch=None, seq_list=None):
    """
    :type epoch: int|None
    :param list[str] | None seq_list: In case we want to set a predefined order.
    :rtype: bool
    :returns whether the order changed (True is always safe to return)

    This is called when we start a new epoch, or at initialization.
    Call this when you reset the seq list.
    """
    self.epoch = epoch
    self.rnd_seq_drop = Random(epoch or 1)
    return False

  def _base_init(self):
    self.nbytes = 0
    self.zpad = None
    # We expect that the following attributes are already set elsewhere, by a derived class.
    assert self.num_outputs
    if not self.num_inputs:
      assert not self.window or self.window in (0, 1) or "data" in self.num_outputs
      return
    assert self.num_inputs > 0
    assert self.window > 0

    if int(self.window) % 2 == 0:
      self.window += 1

    self.nbytes = numpy.array([], dtype=numpy.float32).itemsize * (self.num_inputs * self.window + 1 + 1)

    if self.window > 1:
      self.zpad = numpy.zeros((int(self.window) // 2, self.num_inputs), dtype=numpy.float32)

  def initialize(self):
    """
    Does the main initialization before it can be used.
    This needs to be called before self.load_seqs() can be used.
    """
    self._base_init()
    self.init_seq_order()

  def get_times(self, sorted_seq_idx):
    raise NotImplementedError

  def get_data(self, seq_idx, key):
    """
    :param int seq_idx: sorted seq idx
    :param str key: data-key, e.g. "data" or "classes"
    :rtype: numpy.ndarray
    :returns features or targets: format 2d (time,feature) (float)
    """
    # Fallback implementation for old-style subclasses.
    if key == "data":
      return self.get_input_data(seq_idx)
    else:
      return self.get_targets(key, seq_idx)

  def get_input_data(self, sorted_seq_idx):
    """
    :type sorted_seq_idx: int
    :rtype: numpy.ndarray
    :returns features: format 2d (time,feature) (float)
    """
    raise NotImplementedError

  def get_targets(self, target, sorted_seq_idx):
    """
    :type sorted_seq_idx: int
    :rtype: numpy.ndarray
    :returns targets: format 1d (time) (int: idx of output-feature)
    """
    # For new-style subclasses, which just provide get_data.
    return self.get_data(sorted_seq_idx, target)

  def get_ctc_targets(self, sorted_seq_idx):
    raise NotImplementedError

  def get_data_slice(self, seq_idx, key, start_frame, end_frame):
    if "[sparse:" in key and (start_frame > 0 or end_frame < self.get_seq_length(seq_idx)[key]):
      return self._get_data_slice_sparse(seq_idx, key, start_frame, end_frame)
    data = self.get_data(seq_idx, key)
    return data[start_frame:end_frame]

  def _get_data_slice_sparse(self, seq_idx, key, start_frame, end_frame):
    key_prefix = key[:key.index("[")]
    sparse_info = key[key.index("[") + 1:key.index("]")].split(":")
    assert len(sparse_info) == 4
    assert tuple(sparse_info[0:3]) == ("sparse", "coo", "2")
    s0 = self.get_data(seq_idx, "%s[sparse:coo:2:0]" % key_prefix)
    assert s0 is not None
    from NativeOp import sparse_splice_offset_numpy
    s0_start = sparse_splice_offset_numpy(s0, start_frame)
    s0_end = sparse_splice_offset_numpy(s0, end_frame)
    if sparse_info[-1] == "0":
      return s0[s0_start:s0_end] - start_frame
    else:
      data = self.get_data(seq_idx, key)
      return data[s0_start:s0_end]

  def get_tag(self, sorted_seq_idx):
    """
    :param int sorted_seq_idx:
    :rtype: str
    """
    return "seq-%i" % sorted_seq_idx

  def have_corpus_seq_idx(self):
    """
    :rtype: bool
    :return: whether you can call self.get_corpus_seq_idx()
    """
    return False

  def get_corpus_seq_idx(self, seq_idx):
    """
    :param int seq_idx: sorted sequence index from the current epoch, depending on seq_ordering
    :return: the sequence index as-is in the original corpus. only defined if self.have_corpus_seq_idx()
    :rtype: int
    """
    if self.seq_ordering == "default":
      return seq_idx
    assert self.have_corpus_seq_idx()
    raise NotImplemented

  def has_ctc_targets(self):
    return False

  def get_max_ctc_length(self):
    return 0

  @classmethod
  def generic_complete_frac(cls, seq_idx, num_seqs):
    """
    :param int seq_idx: idx
    :param int|None num_seqs: None if not available
    :return: Returns a fraction (float in [0,1], always > 0) of how far we have advanced
      for this seq in the dataset.
      This does not have to be exact. This is only for the user.
    """
    if num_seqs:
      return min(float(seq_idx + 1) / num_seqs, 1.0)
    else:
      # We don't know. So:
      # Some monotonic increasing function in [0,1] which never reaches 1.
      import math
      return max(1.e-10, 1.0 - math.exp(-seq_idx * 1000))

  def get_complete_frac(self, seq_idx):
    """
    :return: Returns a fraction (float in [0,1], always > 0) of how far we have advanced
      for this seq in the dataset.
      This does not have to be exact. This is only for the user.
    """
    try:
      num_seqs = self.num_seqs
    except Exception:  # num_seqs not always available
      try:
        num_seqs = self.estimated_num_seqs
      except Exception:  # also not always available
        num_seqs = None  # ignore
    return self.generic_complete_frac(seq_idx, num_seqs)

  @property
  def num_seqs(self):
    raise NotImplementedError

  @property
  def estimated_num_seqs(self):
    try:
      return self.num_seqs
    except Exception:  # might not be available
      pass
    if self._estimated_num_seqs is not None:
      return self._estimated_num_seqs
    return None

  def get_data_keys(self):
    """
    :return: all available data keys (for get_data and all other functions)
    :rtype: list[str]
    """
    return ["data"] + self.get_target_list()

  def get_target_list(self):
    """
    :return: subset of :func:`get_data_keys`. target keys are usually not available during inference
    :rtype: list[str]
    """
    return ["classes"]

  def get_data_dim(self, key):
    """
    :param str key: e.g. "data" or "classes"
    :return: number of classes, no matter if sparse or not
    :rtype: int
    """
    if key in self.num_outputs:
      # num_outputs should have the correct dimension, even for key "data" with self.window > 1.
      return self.num_outputs[key][0]
    if self.window > 1 and key == "data":
      assert self.num_inputs
      return self.num_inputs * self.window
    return 1  # unknown

  def get_data_dtype(self, key):
    """
    :param str key: e.g. "data" or "classes"
    :return: dtype as str, e.g. "int32" or "float32"
    :rtype: str
    """
    if self.is_data_sparse(key):
      return "int32"
    return "float32"

  def is_data_sparse(self, key):
    """
    :param str key: e.g. "data" or "classes"
    :return: whether the data is sparse
    :rtype: bool
    """
    if key in self.num_outputs:
      return self.num_outputs[key][1] == 1
    if key == "data":
      return False
    return True

  def get_data_shape(self, key):
    """
    :returns get_data(*, key).shape[1:], i.e. num-frames excluded
    :rtype: list[int]
    """
    if self.is_data_sparse(key):
      return []
    return [self.get_data_dim(key)]

  def have_seqs(self):
    """
    :return: whether num_seqs > 0
    :rtype: bool
    """
    return self.is_less_than_num_seqs(0)

  def len_info(self):
    """
    :rtype: str
    :returns a string to present the user as information about our len.
    Depending on our implementation, we can give some more or some less information.
    """
    return ", ".join([self.__class__.__name__,
                      "sequences: %s" % try_run(lambda: self.num_seqs, default="unknown"),
                      "frames: %s" % try_run(self.get_num_timesteps, default="unknown")])

  def is_less_than_num_seqs(self, n):
    """
    :type n: int
    :rtype: bool
    :returns whether n < num_seqs. In case num_seqs is not known in advance, it will wait
    until it knows that n is behind the end or that we have the seq.
    """
    # We keep this dynamic so that other implementations which don't know the num of seqs
    # in advance can handle this somehow.
    return n < self.num_seqs

  def can_serialize_data(self, key):
    """
    :param str key: e.g. "classes"
    :rtype: bool
    """
    return key in self.labels

  def serialize_data(self, key, data):
    """
    :param str key: e.g. "classes". self.labels[key] should be set
    :param numpy.ndarray data: 1D
    """
    return " ".join(map(self.labels[key].__getitem__, data))

  def calculate_priori(self, target="classes"):
    priori = numpy.zeros((self.num_outputs[target][0],), dtype=numpy.float32)
    i = 0
    while self.is_less_than_num_seqs(i):
      self.load_seqs(i, i + 1)
      for t in self.get_targets(target, i):
        priori[t] += 1
      i += 1
    return numpy.array(priori / self.get_num_timesteps(), dtype=numpy.float32)

  def iterate_seqs(self, chunk_size=None, chunk_step=None, used_data_keys=None):
    """
    Takes chunking into consideration.
    :param int|NumbersDict chunk_size:
    :param int|NumbersDict chunk_step:
    :param set(str)|None used_data_keys:
    :return: generator which yields tuples (seq index, seq start, seq end)
    :rtype: list[(int,NumbersDict,NumbersDict)]
    """
    if chunk_size is None:
      chunk_size = self.chunk_size
    if chunk_step is None:
      chunk_step = self.chunk_step
    chunk_size = NumbersDict(chunk_size)
    chunk_step = NumbersDict(chunk_step)
    s = 0
    while self.is_less_than_num_seqs(s):
      length = self.get_seq_length(s)
      if chunk_size == 0:
        yield (s, length.constant_like(0), length)
      else:
        default_key = "data"
        if used_data_keys is not None:
          length = NumbersDict({k: length[k] for k in used_data_keys})
          if default_key not in used_data_keys:
            default_key = sorted(used_data_keys)[0]
          if chunk_step[default_key] == 0:  # allow some keys with zero chunk-step
            assert chunk_step.max_value() > 0
            default_key = [key for key in sorted(used_data_keys) if chunk_step[key] > 0][0]
        assert chunk_step[default_key] > 0
        t = length.constant_like(0)
        # There are usually the 'data' (input) and 'classes' (targets) data-keys in `length` but there can be others.
        # We expect them all of the same length so that we can do chunking.
        # In case that some length is 0 or 1,
        # we treat it special and always return the full seq repeated for every chunk.
        keys_with_full_seqs = []
        for key in length.keys():
          if chunk_step[key] == chunk_step[default_key]:
            if length[key] == length[default_key]:
              continue  # ok
          if length[key] <= 1:  # special case as explained above
            keys_with_full_seqs.append(key)
            continue
          if chunk_step[key] == chunk_step[default_key]:
            raise Exception("Chunking with multiple data-keys of different length: %r" % length)
          else:
            nr_of_full_chunks_key = (length[key] - chunk_size[key]) // chunk_step[key] + 1
            nr_of_full_chunks_default_key = (length[default_key] - chunk_size[default_key]) // chunk_step[default_key] + 1
            assert nr_of_full_chunks_key == nr_of_full_chunks_default_key
        while length[default_key] > t[default_key]:
          chunk_start = NumbersDict(t)
          chunk_end = NumbersDict.min([t + chunk_size, length])
          for key in keys_with_full_seqs:
            chunk_start[key] = 0
            chunk_end[key] = length[key]
          if length.value is None:
            chunk_start.value = None
            chunk_end.value = None
          yield (s, chunk_start, chunk_end)
          t += chunk_step
          if length[default_key] - t[default_key] <= self.min_chunk_size:
            break
      s += 1

  def _get_context_window_left_right(self):
    """
    :return: (ctx_left, ctx_right)
    :rtype: None|(NumbersDict,NumbersDict)
    """
    if self.context_window:
      # One less because the original frame also counts, and context_window=1 means that we just have that single frame.
      # ctx_total is how much frames we add additionally.
      ctx_total = NumbersDict.max([self.context_window, 1]) - 1
      # In case ctx_total is odd / context_window is even, we have to decide where to put one more frame.
      # To keep it consistent with e.g. 1D convolution with a kernel of even size, we add one more to the right.
      # See test_tfconv1d_evensize().
      ctx_left = ctx_total // 2
      ctx_right = ctx_total - ctx_left
      return ctx_left, ctx_right
    else:
      return None

  def get_start_end_frames_full_seq(self, seq_idx):
    """
    :param int seq_idx:
    :return: (start,end) frame, taking context_window into account
    :rtype: (NumbersDict,NumbersDict)
    """
    end = self.get_seq_length(seq_idx)
    start = end.constant_like(0)
    ctx_lr = self._get_context_window_left_right()
    if ctx_lr:
      start -= ctx_lr[0]
      end += ctx_lr[1]
    return start, end

  def _generate_batches(self, recurrent_net,
                        batch_size, max_seqs=-1, max_seq_length=sys.maxsize,
                        seq_drop=0.0,
                        used_data_keys=None):
    """
    :param bool recurrent_net: If True, the batch might have a batch seq dimension > 1.
      Otherwise, the batch seq dimension is always 1 and multiple seqs will be concatenated.
    :param int batch_size: Max number of frames in one batch.
    :param int max_seqs: Max number of seqs per batch.
    :param int|dict[str,int]|NumbersDict max_seq_length:
    :param set(str)|None used_data_keys:
    """
    if batch_size == 0:
      batch_size = sys.maxsize
    assert batch_size > 0
    if max_seqs == -1:
      max_seqs = float('inf')
    if not max_seq_length:
      max_seq_length = sys.maxsize
    if isinstance(max_seq_length, int) and max_seq_length < 0:
      max_seq_length = {"classes": -max_seq_length}
    max_seq_length = NumbersDict(max_seq_length)
    assert max_seqs > 0
    assert seq_drop <= 1.0
    chunk_size = self.chunk_size
    chunk_step = self.chunk_step
    if not recurrent_net:
      if chunk_size != 0:
        print("Non-recurrent network, chunk size %s:%s ignored" % (chunk_size, chunk_step), file=log.v4)
        chunk_size = 0
    batch = Batch()
    ctx_lr = self._get_context_window_left_right()
    for seq_idx, t_start, t_end in self.iterate_seqs(
          chunk_size=chunk_size, chunk_step=chunk_step, used_data_keys=used_data_keys):
      if ctx_lr:
        t_start -= ctx_lr[0]
        t_end += ctx_lr[1]
      if recurrent_net:
        length = t_end - t_start
        if length.any_compare(max_seq_length, (lambda a, b: a > b)):
          continue
        if length.max_value() > batch_size:
          print("warning: sequence length (%i) larger than limit (%i)" % (length.max_value(), batch_size), file=log.v4)
        if self.rnd_seq_drop.random() < seq_drop:
          continue
        dt, ds = batch.try_sequence_as_slice(length)
        if ds > 1 and ((dt * ds).max_value() > batch_size or ds > max_seqs):
          yield batch
          batch = Batch()
        batch.add_sequence_as_slice(seq_idx=seq_idx, seq_start_frame=t_start, length=length)
      else:  # Not recurrent.
        while t_start.max_value() < t_end.max_value():
          length = t_end - t_start
          num_frames = NumbersDict.min([length, batch_size - batch.get_all_slices_num_frames()])
          assert num_frames.max_value() > 0
          batch.add_frames(seq_idx=seq_idx, seq_start_frame=t_start, length=num_frames)
          if batch.get_all_slices_num_frames() >= batch_size or batch.get_num_seqs() > max_seqs:
            yield batch
            batch = Batch()
          t_start += num_frames

    if batch.get_all_slices_num_frames() > 0:
      yield batch

  def batch_set_generator_cache_whole_epoch(self):
    """
    The BatchSetGenerator can cache the list of batches which we generated across epochs.
    See self.generate_batches() and self._generate_batches().
    In many cases, the dataset does not support this, and in that case,
    it is not needed to enable this cache and waste memory.
    Caching it together with option shuffle_batches could also mean that
    there will be self.load_seqs() calls with non-monotonic seq-idxs.
    The only dataset currently which enables this is CachedDataset and thus HDFDataset.

    :return: whether we should enable this cache
    :rtype: bool
    """
    return False

  def generate_batches(self, shuffle_batches=False, **kwargs):
    """
    :param bool shuffle_batches:
    :param kwargs: will be passed to :func:`_generate_batches`
    :rtype: BatchSetGenerator
    """
    return BatchSetGenerator(
      dataset=self,
      generator=self._generate_batches(**kwargs),
      shuffle_batches=shuffle_batches,
      cache_whole_epoch=self.batch_set_generator_cache_whole_epoch())

  @classmethod
  def index_shape_for_batches(cls, batches, data_key="data"):
    shape = [0, 0]  # time,batch
    for batch in batches:
      shape = [max(shape[0], batch.max_num_frames_per_slice[data_key]), shape[1] + batch.num_slices]
    return shape


class DatasetSeq:
  def __init__(self, seq_idx, features, targets=None, ctc_targets=None, seq_tag=None):
    """
    :param int seq_idx: sorted seq idx in the Dataset
    :param numpy.ndarray|dict[str,numpy.ndarray] features: format 2d (time,feature) (float)
    :param dict[str,numpy.ndarray]|numpy.ndarray|None targets: name -> format 1d (time) (idx of output-feature)
    :param numpy.ndarray|None ctc_targets: format 1d (time) (idx of output-feature)
    :param str seq_tag: sequence name / tag
    """
    assert isinstance(seq_idx, int)
    self.seq_idx = seq_idx
    self.seq_tag = seq_tag or ("seq-%i" % seq_idx)
    if not isinstance(features, dict):
      assert isinstance(features, numpy.ndarray)
      features = {"data": features}
      if targets is None:
        targets = {}
      if isinstance(targets, numpy.ndarray):  # old format
        targets = {"classes": targets}
      assert isinstance(targets, dict)
      features.update(targets)
      targets = None
    assert isinstance(features, dict) and targets is None
    for v in features.values():
      assert isinstance(v, numpy.ndarray)
    self.features = features
    self.ctc_targets = ctc_targets

  @property
  def num_frames(self):
    """
    :rtype: NumbersDict
    """
    d = {k: (v.shape[0] if v.ndim >= 1 else 1)
         for (k, v) in self.features.items()}
    return NumbersDict(d)

  def get_data(self, key):
    return self.features[key]

  def get_data_keys(self):
    return self.features.keys()

  def __repr__(self):
    return "<DataCache seq_idx=%i>" % self.seq_idx


def get_dataset_class(name):
  from importlib import import_module
  # Only those modules which make sense to be loaded by the user,
  # because this function is only used for such cases.
  mod_names = ["HDFDataset", "SprintDataset", "GeneratingDataset", "NumpyDumpDataset", "MetaDataset", "LmDataset", "StereoDataset", "RawWavDataset"]
  for mod_name in mod_names:
    mod = import_module(mod_name)
    if name in vars(mod):
      clazz = getattr(mod, name)
      assert issubclass(clazz, Dataset)
      return clazz
  return None


def init_dataset(kwargs, extra_kwargs=None, default_kwargs=None):
  """
  :param dict[str]|str|(()->dict[str]) kwargs:
  :param dict[str]|None extra_kwargs:
  :param dict[str]|None default_kwargs:
  :rtype: Dataset
  """
  assert kwargs
  if callable(kwargs):
    return init_dataset(kwargs(), extra_kwargs=extra_kwargs, default_kwargs=default_kwargs)
  if isinstance(kwargs, (str, unicode)):
    if kwargs.startswith("{"):
      kwargs = eval(kwargs)
    else:
      config_str = kwargs
      kwargs = {}
      if default_kwargs:
        kwargs.update(default_kwargs)
      if extra_kwargs:
        kwargs.update(extra_kwargs)
      return init_dataset_via_str(config_str=config_str, **kwargs)
  assert isinstance(kwargs, dict)
  kwargs = kwargs.copy()
  assert "class" in kwargs
  clazz_name = kwargs.pop("class")
  clazz = get_dataset_class(clazz_name)
  if not clazz:
    raise Exception("Dataset class %r not found" % clazz_name)
  if default_kwargs:
    for key, value in default_kwargs.items():
      kwargs.setdefault(key, value)
  if extra_kwargs:
    kwargs.update(extra_kwargs)
  obj = clazz(**kwargs)
  assert isinstance(obj, Dataset)
  obj.initialize()
  return obj


def init_dataset_via_str(config_str, config=None, cache_byte_size=None, **kwargs):
  """
  :param str config_str: hdf-files, or "LmDataset:..." or so
  :param Config.Config|None config: optional, only for "sprint:..."
  :param int|None cache_byte_size: optional, only for HDFDataset
  :rtype: Dataset
  """
  kwargs = kwargs.copy()
  if 'window' not in kwargs and config and config.has('window'):
    kwargs['window'] = config.int('window', 1)
  from HDFDataset import HDFDataset
  if config_str.startswith("sprint:"):
    kwargs["sprintConfigStr"] = config_str[len("sprint:"):]
    assert config, "need config for dataset in 'sprint:...' format. or use 'ExternSprintDataset:...' instead"
    sprintTrainerExecPath = config.value("sprint_trainer_exec_path", None)
    assert sprintTrainerExecPath, "specify sprint_trainer_exec_path in config"
    kwargs["sprintTrainerExecPath"] = sprintTrainerExecPath
    from SprintDataset import ExternSprintDataset
    cls = ExternSprintDataset
  elif config_str.startswith("config:"):
    from Config import get_global_config
    if not config:
      config = get_global_config()
    data = eval(config_str[len("config:"):], config.typed_dict, config.typed_dict)
    return init_dataset(data)
  elif ":" in config_str:
    kwargs.update(eval("dict(%s)" % config_str[config_str.find(":") + 1:]))
    class_name = config_str[:config_str.find(":")]
    cls = get_dataset_class(class_name)
  else:
    if cache_byte_size is not None:
      kwargs["cache_byte_size"] = cache_byte_size
    cls = HDFDataset
  if config:
    data = cls.from_config(config, **kwargs)
  else:
    data = cls(**kwargs)
  if isinstance(data, HDFDataset):
    for f in config_str.split(","):
      if f:
        assert os.path.exists(f)
        data.add_file(f)
  data.initialize()
  return data


def convert_data_dims(data_dims, leave_dict_as_is=False):
  """
  This converts what we called num_outputs originally,
  from the various formats which were allowed in the past
  (just an int, or dict[str,int]) into the format which we currently expect.
  In all cases, the output will be a new copy of the dict.

  :param int|dict[str,int|(int,int)|dict] data_dims: what we called num_outputs originally
  :param bool leave_dict_as_is:
  :rtype: dict[str,(int,int)|dict]
  :returns dict data-key -> (data-dimension, len(shape) (1 ==> sparse))
   (or potentially data-key -> dict, if leave_dict_as_is is True; for TensorFlow)
  """
  if isinstance(data_dims, int):
    data_dims = {"classes": data_dims}
  assert isinstance(data_dims, dict)
  data_dims = data_dims.copy()
  for k, v in list(data_dims.items()):
    if isinstance(v, int):
      v = [v, 2 if k == "data" else 1]
      data_dims[k] = v
    if isinstance(v, dict) and leave_dict_as_is:
      continue
    assert isinstance(v, (tuple, list))
    assert len(v) == 2
    assert isinstance(v[0], int)
    assert isinstance(v[1], int)
    assert 1 <= v[1] <= 2
  return data_dims


def shapes_for_batches(batches, data_keys, dataset=None, extern_data=None, enforce_min_len1=False):
  """
  :param list[EngineBatch.Batch] batches:
  :param list[str] data_keys:
  :param Dataset dataset:
  :param TFNetwork.ExternData extern_data: detailed data description. only used for TensorFlow
  :param bool enforce_min_len1:
  :rtype: dict[str,list[int]] | None
  """
  assert dataset or extern_data
  all_data_keys = set(data_keys) | {"data"}

  # The final device.data.shape is in format (time,batch,feature) in case of Theano.
  shape = [NumbersDict(0), 0]  # time,batch
  for batch in batches:
    shape = [NumbersDict.max([shape[0], batch.max_num_frames_per_slice]), shape[1] + batch.num_slices]
  if shape[1] == 0:
    return None
  assert shape[0].max_value() > 0
  # Theano has some buggy behaviour with tensors with some shape of zero.
  # We will just use one dummy frame in that case.
  # The index will stay zero in that case. (see EngineUtil.assign_dev_data())
  # However, also see the OutputLayer.output_index() behavior for forwarding.
  if not extern_data or enforce_min_len1:  # not needed if TensorFlow is used
    for k in all_data_keys:
      shape[0][k] = max(shape[0][k], 1)

  if extern_data:
    d = {}
    for k in all_data_keys:
      data_shape = list(extern_data.data[k].batch_shape)
      data_shape[extern_data.data[k].batch_dim_axis] = shape[1]
      if extern_data.data[k].have_time_axis():
        data_shape[extern_data.data[k].time_dim_axis] = shape[0][k]
      assert all([n is not None for n in data_shape])
      d[k] = data_shape
  else:  # shape via dataset
    d = {k: [shape[0][k], shape[1]] for k in all_data_keys}
    for k in all_data_keys:
      d[k] += dataset.get_data_shape(k)
  return d


def set_config_num_inputs_outputs_from_dataset(config, dataset):
  """
  :param Config.Config config:
  :param Dataset dataset:
  """
  config.set("num_inputs", dataset.num_inputs)
  from Util import BackendEngine
  if BackendEngine.is_tensorflow_selected():
    # TF supports more fine-grained specification,
    # however the dataset does not store that in num_outputs.
    from TFNetwork import ExternData
    config.set("num_outputs", {
      key: ExternData.data_kwargs_from_dataset_key(dataset=dataset, key=key)
      for key in dataset.get_data_keys()})
  else:
    config.set("num_outputs", dataset.num_outputs)
