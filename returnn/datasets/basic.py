
"""
This defines the base dataset class :class:`Dataset`.
"""

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
import typing

from returnn.log import log
from returnn.engine.batch import Batch, BatchSetGenerator
from returnn.util.basic import PY3, try_run, NumbersDict, unicode, OptionalNotImplementedError


class Dataset(object):
  """
  Base class for any dataset. This defines the dataset API.
  """

  @staticmethod
  def kwargs_update_from_config(config, kwargs):
    """
    :type config: Config.Config
    :type kwargs: dict[str]
    """
    def set_or_remove(key, value):
      """
      :param str key:
      :param value:
      """
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
    set_or_remove("chunking_variance", config.float("chunking_variance", 0))

  @staticmethod
  def get_default_kwargs_eval(config):
    """
    :param Config.Config config:
    :rtype: dict[str]
    """
    # For dev/eval, by default, we should not do chunking (i.e. chunking = "0").
    chunking = "0"
    if config.value("on_size_limit", "ignore") == "chunk":
      chunking = config.value("batch_size", "0")
    elif config.value('chunking', "0") == "1":  # MLP mode
      chunking = "1"
    elif config.bool('chunk_eval', False):
      chunking = config.value('chunking', "0")
    return dict(chunking=chunking, seq_ordering="sorted", shuffle_frames_of_nseqs=0)

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
               seq_ordering='default', random_seed_offset=None,
               partition_epoch=None, repeat_epoch=None,
               seq_list_filter_file=None, unique_seq_tags=False,
               seq_order_seq_lens_file=None,
               shuffle_frames_of_nseqs=0, min_chunk_size=0, chunking_variance=0,
               estimated_num_seqs=None):
    """
    :param str name: e.g. "train" or "eval"
    :param int window: features will be of dimension window * feature_dim, as we add a context-window around.
      not all datasets support this option.
    :param None|int|dict|NumbersDict|(dict,dict) context_window: will add this context for each chunk
    :param None|str|int|(int,int)|dict|(dict,dict) chunking: "chunk_size:chunk_step"
    :param str seq_ordering: "batching"-option in config. e.g. "default", "sorted" or "random".
      See self.get_seq_order_for_epoch() for more details.
    :param int|None random_seed_offset:
    :param int|None partition_epoch:
    :param int|None repeat_epoch: Repeat the sequences in an epoch this many times. Useful to scale the dataset
      relative to other datasets, e.g. when used in CombinedDataset. Not allowed to be used in combination with
      partition_epoch.
    :param str|None seq_list_filter_file: defines a subset of sequences (by tag) to use
    :param bool unique_seq_tags: uniquify seqs with same seq tags in seq order
    :param str|None seq_order_seq_lens_file: for seq order, use the seq length given by this file
    :param int shuffle_frames_of_nseqs: shuffles the frames. not always supported
    :param None|int estimated_num_seqs: for progress reporting in case the real num_seqs is unknown
    """
    self.name = name or ("dataset_id%s" % id(self))
    self.lock = RLock()  # Used when manipulating our data potentially from multiple threads.
    self.rnd_seq_drop = None  # type: typing.Optional[Random]
    self.num_inputs = 0  # usually not used, but num_outputs instead, which is more generic
    self.num_outputs = None  # type: typing.Optional[typing.Dict[str,typing.Tuple[int,int]]]  # tuple is num-classes, len(shape).  # nopep8
    self.window = window
    self.seq_ordering = seq_ordering  # "default", "sorted" or "random". See self.get_seq_order_for_epoch().
    if random_seed_offset is None:
      random_seed_offset = self._get_default_random_seed_offset()
    self.random_seed_offset = random_seed_offset
    self.partition_epoch = partition_epoch or 1
    self.repeat_epoch = repeat_epoch or 1
    self.seq_tags_filter = set(self._load_seq_list_file(seq_list_filter_file)) if seq_list_filter_file else None
    self.unique_seq_tags = unique_seq_tags
    self._seq_order_seq_lens_file = seq_order_seq_lens_file
    self._seq_order_seq_lens_by_idx = None
    # There is probably no use case for combining the two, so avoid potential misconfiguration.
    assert self.partition_epoch == 1 or self.repeat_epoch == 1, (
      "Combining partition_epoch and repeat_epoch is prohibited.")
    self.labels = {}  # type: typing.Dict[str,typing.List[str]]
    self.weights = {}
    self._num_timesteps = 0
    self._num_seqs = 0
    self._estimated_num_seqs = estimated_num_seqs
    self.min_chunk_size = min_chunk_size
    self.chunking_variance = chunking_variance
    self.chunk_size, self.chunk_step = self._parse_chunking(chunking)
    if isinstance(context_window, (tuple, list)):
      assert len(context_window) == 2
      for elem in context_window:
        assert isinstance(elem, dict)
      # assume that first element of tuple|list is for left context and second element for right context
      self.ctx_left = NumbersDict(numbers_dict=context_window[0])
      self.ctx_right = NumbersDict(numbers_dict=context_window[1])
    else:
      if context_window is None:
        context_window = NumbersDict()
      elif isinstance(context_window, int):
        context_window = NumbersDict(numbers_dict={"data": context_window})
      elif isinstance(context_window, dict):
        context_window = NumbersDict(numbers_dict=context_window)
      assert isinstance(context_window, NumbersDict)
      # ctx_total is how much frames we add additionally.
      # One less because the original frame also counts, and context_window=1 means that we just have that single frame.
      ctx_total = NumbersDict.max([context_window, 1]) - 1
      # In case ctx_total is odd / context_window is even, we have to decide where to put one more frame.
      # To keep it consistent with e.g. 1D convolution with a kernel of even size, we add one more to the right.
      # See test_tfconv1d_evensize().
      self.ctx_left = ctx_total // 2
      self.ctx_right = ctx_total - self.ctx_left
    assert isinstance(self.ctx_left, NumbersDict)
    assert isinstance(self.ctx_right, NumbersDict)
    self.shuffle_frames_of_nseqs = shuffle_frames_of_nseqs
    self.epoch = None

  def __repr__(self):
    return "<%s %r epoch=%s>" % (
      self.__class__.__name__,
      getattr(self, "name", "<unknown>"),
      getattr(self, "epoch", "<unknown>"))

  @staticmethod
  def _get_default_random_seed_offset():
    """
    :return: 0 usually
    :rtype: int
    """
    from returnn.config import get_global_config
    config = get_global_config(raise_exception=False)
    if not config:
      return 0
    if config.is_true("use_horovod"):
      import returnn.tf.horovod
      if returnn.tf.horovod.get_ctx().is_dataset_distribution_random_seed_offset():
        return returnn.tf.horovod.get_ctx().rank() * 16127
    return 0

  @staticmethod
  def _parse_chunking(chunking):
    """
    :param None|str|int|(int,int)|dict|(dict,dict)|(NumbersDict,NumbersDict) chunking:
      as it comes from the config / from the user
    :return: chunk_size, chunk_step
    :rtype: (NumbersDict,NumbersDict)
    """
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
    if chunk_step in (None, 0):
      chunk_step = chunk_size
    assert isinstance(chunk_step, (int, dict, NumbersDict))
    chunk_step = NumbersDict(chunk_step)
    if chunk_size != 0:
      assert sorted(chunk_step.keys()) == sorted(chunk_size.keys())
      assert chunk_step.max_value() > 0, "chunking step must be positive (for some key)"
    return chunk_size, chunk_step

  @staticmethod
  def _load_seq_list_file(filename, use_cache_manager=False, expect_list=True):
    """
    :param str filename:
    :param bool use_cache_manager:
    :param bool expect_list:
    :rtype: list[str]|dict[str,list[str]]
    """
    if use_cache_manager:
      import returnn.util.basic
      filename = returnn.util.basic.cf(filename)
    if filename.endswith(".pkl"):
      import pickle
      seq_list = pickle.load(open(filename, 'rb'))
      if expect_list:
        assert isinstance(seq_list, list)
    elif filename.endswith(".gz"):
      import gzip
      seq_list = gzip.open(filename, "rt").read().splitlines()
    else:
      seq_list = open(filename).read().splitlines()
    return seq_list

  def _sliding_window(self, xr):
    """
    :type xr: numpy.ndarray
    :rtype: numpy.ndarray
    """
    # noinspection PyProtectedMember
    from numpy.lib.stride_tricks import as_strided
    x = numpy.concatenate([self.zpad, xr, self.zpad])
    return as_strided(
      x,
      shape=(x.shape[0] - self.window + 1, 1, self.window, self.num_inputs),
      strides=(x.strides[0], x.strides[1] * self.num_inputs) + x.strides
      ).reshape((xr.shape[0], self.num_inputs * self.window))

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

  def get_seq_length(self, seq_idx):
    """
    :param int seq_idx:
    :rtype: NumbersDict
    :returns the len of the input features and the len of the target sequence.
    """
    raise NotImplementedError

  def get_estimated_seq_length(self, seq_idx):
    """
    In contrast to self.get_seq_length(), this method is designed to work for sequences that have not been loaded yet
    via self.load_seqs().
    Used by meta-datasets for sequence ordering. Currently we only provide one number, i.e. do not give different
    estimates for the different data keys (as in get_seq_length()). It is up to the dataset what this number represents
    and how it is computed.

    :param int seq_idx: for current epoch, not the corpus seq idx
    :rtype: int
    :returns sequence length estimate (for sorting)
    """
    raise OptionalNotImplementedError

  def get_num_timesteps(self):
    """
    :rtype: int
    """
    assert self._num_timesteps > 0
    return self._num_timesteps

  def load_seqs(self, start, end):
    """
    Load data sequences, such that self.get_data() & friends can return the data.

    :param int start: start sorted seq idx, inclusive
    :param int end: end sorted seq idx, exclusive
    """
    assert start >= 0
    assert start <= end
    if self.is_cached(start, end):
      return

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
    raise OptionalNotImplementedError

  def _load_seqs(self, start, end):
    """
    Load data sequences.
    :param int start: inclusive seq idx start
    :param int end: exclusive seq idx end. can be more than num_seqs
    If end > num_seqs, will not load them.
    """
    raise NotImplementedError

  def _get_seq_order_seq_lens_by_idx(self, seq_idx):
    """
    :param int seq_idx:
    :rtype: int
    """
    if not self._seq_order_seq_lens_by_idx:
      assert self._seq_order_seq_lens_file
      if self._seq_order_seq_lens_file.endswith(".gz"):
        import gzip
        raw = gzip.GzipFile(self._seq_order_seq_lens_file, "rb").read()
      else:
        raw = open(self._seq_order_seq_lens_file, "rb").read()
      seq_lens = eval(raw)
      assert isinstance(seq_lens, dict)
      all_tags = self.get_all_tags()
      self._seq_order_seq_lens_by_idx = [seq_lens[tag] for tag in all_tags]
    return self._seq_order_seq_lens_by_idx[seq_idx]

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
    repeat_epoch = self.repeat_epoch or 1
    if not epoch:
      epoch = 1
    full_epoch = epoch
    if partition_epoch > 1:
      full_epoch = (epoch - 1) // partition_epoch + 1
    assert num_seqs > 0
    seq_index = list(range(num_seqs))  # type: typing.List[int]  # the real seq idx after sorting
    if self._seq_order_seq_lens_file:
      get_seq_len = self._get_seq_order_seq_lens_by_idx
    if self.seq_ordering == 'default':
      pass  # Keep order as-is.
    elif self.seq_ordering.startswith("default_every_n:"):
      # This order is useful if you have "initial_state": "keep_over_epoch",
      # where num == max_seqs, batch_size = inf, max_seq_len = inf, chunking = None.
      _, num = self.seq_ordering.split(":")
      num = int(num)
      seq_index = numpy.arange(num_seqs // num, dtype="int64").repeat(num)
      for i in range(1, num):
        seq_index[i::num] += i * (num_seqs // num)
      seq_index = list(seq_index)
    elif self.seq_ordering == 'reverse':
      seq_index = list(reversed(seq_index))
    elif self.seq_ordering == 'sorted':
      assert get_seq_len
      seq_index.sort(key=get_seq_len)  # sort by length, starting with shortest
    elif self.seq_ordering == "sorted_reverse":
      assert get_seq_len
      seq_index.sort(key=get_seq_len, reverse=True)  # sort by length, in reverse, starting with longest
    elif self.seq_ordering.startswith('sort_bin_shuffle'):
      # Shuffle seqs, sort by length, and shuffle bins (then shuffle seqs within each bin if sort_bin_shuffle_x2).
      assert get_seq_len
      tmp = self.seq_ordering.split(':')[1:]
      # Keep this deterministic! Use fixed seed.
      if len(tmp) <= 1:
        nth = 1
      else:
        nth = int(tmp[1])
      rnd_seed = ((full_epoch - 1) // nth + 1) if full_epoch else 1
      rnd = Random(rnd_seed + self.random_seed_offset)
      rnd.shuffle(seq_index)  # Shuffle sequences.
      seq_index.sort(key=get_seq_len)  # Sort by length, starting with shortest.
      if len(tmp) == 0:
        bins = 2
      else:
        if tmp[0].startswith("."):  # starting with "." -> approx chunk size (num of seqs in one bin)
          bins = max(num_seqs // int(tmp[0][1:]), 2)
        else:  # the number of bins
          bins = int(tmp[0])
      bin_ids = list(range(bins))
      rnd.shuffle(bin_ids)  # Shuffle bins.
      out_index = []
      for i in bin_ids:
        if i == bins - 1:
          part = seq_index[i * len(seq_index) // bins:][:]
        else:
          part = seq_index[i * len(seq_index) // bins:(i + 1) * len(seq_index) // bins][:]
        if self.seq_ordering.startswith('sort_bin_shuffle_x2'):
          rnd.shuffle(part)  # Shuffle within the bin.
        out_index += part
      seq_index = out_index
    elif self.seq_ordering.startswith('laplace'):
      assert get_seq_len
      tmp = self.seq_ordering.split(':')[1:]
      if len(tmp) == 0:
        bins = 2
      else:
        if tmp[0].startswith("."):  # starting with "." -> approx chunk size (num of seqs in one bin)
          bins = max(num_seqs // int(tmp[0][1:]), 2)
        else:  # the number of bins
          bins = int(tmp[0])
      if len(tmp) <= 1:
        nth = 1
      else:
        nth = int(tmp[1])
      rnd_seed = ((full_epoch - 1) // nth + 1) if full_epoch else 1
      rnd = Random(rnd_seed + self.random_seed_offset)
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
      rnd = Random(rnd_seed + self.random_seed_offset)
      rnd.shuffle(seq_index)
    else:
      assert False, "invalid batching specified: " + self.seq_ordering
    if self.unique_seq_tags:
      # Note: This is as generic as possible, but requires that get_all_tags is implemented.
      all_seq_tags = self.get_all_tags()
      used_seq_tags = set()
      seq_index = [
        i for i in seq_index
        if (all_seq_tags[i] not in used_seq_tags, used_seq_tags.add(all_seq_tags[i]))[0]]
    if partition_epoch > 1:
      seq_index = self._apply_partition_epoch(seq_index, partition_epoch, epoch)
    if repeat_epoch > 1:
      seq_index = seq_index * repeat_epoch
    if self.seq_tags_filter is not None:
      # Note: This is as generic as possible, but requires that get_all_tags is implemented.
      assert seq_index
      all_seq_tags = self.get_all_tags()
      assert len(all_seq_tags) == num_seqs == self.get_total_num_seqs(), "%r vs %r vs %r" % (
        len(all_seq_tags), num_seqs, self.get_total_num_seqs())
      old_seq_index = seq_index
      seq_index = [i for i in seq_index if all_seq_tags[i] in self.seq_tags_filter]
      assert seq_index, "%s: empty after applying seq_list_filter_file. Example filter tags: %r, used tags: %r" % (
        self, sorted(self.seq_tags_filter)[:3], [all_seq_tags[i] for i in old_seq_index[:3]])
    return seq_index

  @classmethod
  def _apply_partition_epoch(cls, seq_index, partition_epoch, epoch):
    """
    :param list[int] seq_index: full list of ordered sequence indices
    :param int partition_epoch: number of partitions seq_index should be split into
    :param int|None epoch: current epoch
    :return: partition of seq_index for current epoch
    :rtype: list[int]
    """
    num_seqs = len(seq_index)
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

  def _get_random_seed_for_epoch(self, epoch):
    """
    :param int|None epoch:
    :rtype: int
    """
    partition_epoch = self.partition_epoch or 1
    full_epoch = epoch or 1
    if partition_epoch > 1:
      full_epoch = (full_epoch - 1) // partition_epoch + 1
    return full_epoch + self.random_seed_offset

  def init_seq_order(self, epoch=None, seq_list=None, seq_order=None):
    """
    :type epoch: int|None
    :param list[str]|None seq_list: List of sequence tags, to set a predefined order.
    :param list[int]|None seq_order: List of corpus sequence indices, to set a predefined order. Only possible
      if the dataset has such indices (see self.have_corpus_seq_idx()).
    :rtype: bool
    :returns whether the order changed (True is always safe to return)

    This is called when we start a new epoch, or at initialization.
    Call this when you reset the seq list.
    """
    self.epoch = epoch
    self.rnd_seq_drop = Random(self._get_random_seed_for_epoch(epoch=epoch))
    return False

  def finish_epoch(self):
    """
    This would get called at the end of the epoch (currently optional only).
    After this, further calls to :func:`get_data` or :func:`load_seqs` are invalid,
    until a new call to :func:`init_seq_order` follows.
    """
    self.epoch = None

  def get_current_seq_order(self):
    """
    :return: many datasets use self.get_seq_order_for_epoch. this function would return the current seq order
      for the current epoch, after self.init_seq_order was called.
      Not all datasets implement this.
    :rtype: list[int]
    """
    raise OptionalNotImplementedError

  def _base_init(self):
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
    """
    :param int sorted_seq_idx:
    """
    raise OptionalNotImplementedError

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
    :param str target: data key
    :type sorted_seq_idx: int
    :rtype: numpy.ndarray
    :returns targets: format 1d (time) (int: idx of output-feature)
    """
    # For new-style subclasses, which just provide get_data.
    return self.get_data(sorted_seq_idx, target)

  def get_ctc_targets(self, sorted_seq_idx):
    """
    Warning: This is deprecated/obsolete.

    :param int sorted_seq_idx:
    :rtype: numpy.ndarray|None
    """
    return None

  def get_data_slice(self, seq_idx, key, start_frame, end_frame):
    """
    :param int seq_idx:
    :param str key:
    :param int start_frame:
    :param int end_frame:
    :return: x[start_frame:end_frame], with x = get_data(seq_idx, key)
    :rtype: numpy.ndarray
    """
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
    from returnn.native_op import sparse_splice_offset_numpy
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

  def get_all_tags(self):
    """
    :return: list of all seq tags, of the whole dataset, without partition epoch.
      Note that this is not possible with all datasets.
    :rtype: list[str]
    """
    old_partition_epoch = self.partition_epoch
    try:
      all_tags = [None] * self.num_seqs  # type: typing.List[typing.Union[None,str]]
      for seq_idx in range(self.num_seqs):
        all_tags[seq_idx] = self.get_tag(seq_idx)
      return all_tags
    finally:
      self.partition_epoch = old_partition_epoch

  def get_total_num_seqs(self):
    """
    :return: total number of seqs, without partition epoch.
      Should be the same as len(self.get_all_tags()).
      Note that this is not possible with all datasets.
    :rtype: int
    """
    if self.partition_epoch == 1:
      # Note: self.num_seqs might not always be set, or even be correct...
      return self.num_seqs
    raise NotImplementedError("%s: get_total_num_seqs with partition epoch %i" % (self, self.partition_epoch))

  def have_corpus_seq_idx(self):
    """
    :rtype: bool
    :return: whether you can call self.get_corpus_seq_idx()
    """
    return False

  def get_corpus_seq_idx(self, seq_idx):
    """
    :param int seq_idx: sorted sequence index from the current epoch, depending on seq_ordering
    :return: the sequence index as-is in the original corpus (as if you would have sorting="default").
      only defined if self.have_corpus_seq_idx()
    :rtype: int
    """
    if self.seq_ordering == "default":
      return seq_idx
    assert self.have_corpus_seq_idx()
    raise NotImplemented

  def has_ctc_targets(self):
    """
    :return: whether we have get_ctc_targets implemented
    :rtype: bool
    """
    return False

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
    :param int seq_idx:
    :return: Returns a fraction (float in [0,1], always > 0) of how far we have advanced
      for this seq in the dataset.
      This does not have to be exact. This is only for the user.
    :rtype: float
    """
    # noinspection PyBroadException
    try:
      num_seqs = self.num_seqs
    except Exception:  # num_seqs not always available
      # noinspection PyBroadException
      try:
        num_seqs = self.estimated_num_seqs
      except Exception:  # also not always available
        num_seqs = None  # ignore
    return self.generic_complete_frac(seq_idx, num_seqs)

  @property
  def num_seqs(self):
    """
    :rtype: int
    """
    raise NotImplementedError

  @property
  def estimated_num_seqs(self):
    """
    :return: estimated num seqs. does not have to be exact
    :rtype: int|None
    """
    # noinspection PyBroadException
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
    # Note: We cannot call get_data_dtype, as we would maybe result in infinite recursion.
    if key in self.num_outputs:
      return self.num_outputs[key][1] <= 1
    if key == "data":
      return False
    return True

  def get_data_shape(self, key):
    """
    :returns get_data(*, key).shape[1:], i.e. num-frames excluded
    :rtype: list[int]
    """
    if key in self.num_outputs:
      if self.num_outputs[key][1] <= 1:
        return []
      res_shape = [None] * (self.num_outputs[key][1] - 1)  # type: typing.List[typing.Union[None,int]]
      if not self.is_data_sparse(key):
        res_shape[-1] = self.get_data_dim(key)
      return res_shape
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
    :param numpy.ndarray data: 0D or 1D
    :rtype: str
    """
    labels = self.labels[key]
    if data.ndim == 0:
      data = numpy.expand_dims(data, axis=0)
    assert data.ndim == 1
    if len(labels) < 1000 and all([len(label) == 1 for label in labels]):
      # are these actually raw bytes? -> assume utf8
      if all([ord(label) <= 255 for label in labels]):
        try:
          if PY3:
            return bytes([ord(labels[c]) for c in data]).decode("utf8")
          else:
            return b"".join([bytes(labels[c]) for c in data]).decode("utf8")
        except UnicodeDecodeError:
          pass  # pass on to default case
      return "".join(map(labels.__getitem__, data))
    else:
      return " ".join(map(labels.__getitem__, data))

  def calculate_priori(self, target="classes"):
    """
    :param str target:
    :rtype: numpy.ndarray
    """
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
    chunk_size_orig = chunk_size.copy()
    chunk_step_orig = chunk_step.copy()

    s = 0
    while self.is_less_than_num_seqs(s):
      length = self.get_seq_length(s)
      if chunk_size == 0:
        yield s, NumbersDict.constant_like(0, numbers_dict=length), length
      else:
        default_key = "data"
        if used_data_keys is not None:
          length = NumbersDict({k: length[k] for k in used_data_keys})
          if default_key not in used_data_keys:
            default_key = sorted(used_data_keys)[0]
          if chunk_step[default_key] == 0:  # allow some keys with zero chunk-step
            assert chunk_step.max_value() > 0
            default_key = [key for key in sorted(used_data_keys) if chunk_step[key] > 0][0]
          if self.chunking_variance > 0:
            chunking_variance = 1. - self.rnd_seq_drop.random() * self.chunking_variance
            for k in used_data_keys:
              chunk_size[k] = max(int(chunk_size_orig[k] * chunking_variance), 1)
              chunk_step[k] = max(int(chunk_step_orig[k] * chunking_variance), 1)
            # In case there are data keys with different chunk sizes,
            # make sure we keep the original ratio.
            smallest_key = [
              k for k in sorted(used_data_keys, key=lambda key: (chunk_step_orig[key], key))
              if chunk_step_orig[k] > 0][0]
            for k in used_data_keys:
              if chunk_size_orig[k] > chunk_size_orig[smallest_key]:
                if chunk_size_orig[k] % chunk_size_orig[smallest_key] == 0:
                  ratio = chunk_size_orig[k] // chunk_size_orig[smallest_key]
                  chunk_size[k] = chunk_size[smallest_key] * ratio
                  chunk_step[k] = chunk_step[smallest_key] * ratio
        assert chunk_step[default_key] > 0
        t = NumbersDict.constant_like(0, numbers_dict=length)
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
            limit = limit_default = 1
            if self.min_chunk_size == chunk_size[default_key]:
              limit = chunk_size[key]
              limit_default = chunk_size[default_key]
            nr_of_chunks = (length[key] - limit) // chunk_step[key] + 1
            nr_of_chunks_default = (length[default_key] - limit_default) // chunk_step[default_key] + 1
            assert nr_of_chunks == nr_of_chunks_default, (
              "%s: iterate seqs with chunking: length %r, chunk size/step %r/%r (min %r), key %r (default %r)" % (
                self, length, chunk_size, chunk_step, self.min_chunk_size, key, default_key))
        while length[default_key] > t[default_key]:
          chunk_start = NumbersDict(t)
          chunk_end = NumbersDict.min([t + chunk_size, length])
          for key in keys_with_full_seqs:
            chunk_start[key] = 0
            chunk_end[key] = length[key]
          if length.value is None:
            chunk_start.value = None
            chunk_end.value = None
          yield s, chunk_start, chunk_end
          t += chunk_step
          if length[default_key] - t[default_key] <= self.min_chunk_size:
            break
      s += 1

  def get_start_end_frames_full_seq(self, seq_idx):
    """
    :param int seq_idx:
    :return: (start,end) frame, taking context_window into account
    :rtype: (NumbersDict,NumbersDict)
    """
    end = self.get_seq_length(seq_idx)
    start = NumbersDict.constant_like(0, numbers_dict=end)
    start -= self.ctx_left
    end += self.ctx_right
    return start, end

  def sample(self, seq_idx):
    """
    :param int seq_idx:
    :rtype: bool
    """
    if seq_idx in self.weights:
      weight = self.weights[seq_idx]
      return weight[0] >= weight[1]
    return True

  def update_weights(self, seqs, weights):
    """
    :param list[EngineBatch.BatchSeqCopyPart] seqs:
    :param list[float] weights:
    """
    for seq, weight in zip(seqs, weights):
      self.weights[seq.seq_idx] = [weight, 0]

  def _generate_batches(self, recurrent_net,
                        batch_size, max_seqs=-1, max_seq_length=sys.maxsize,
                        max_pad_size=None,
                        min_seq_length=0, pruning=0.0,
                        seq_drop=0.0, max_total_num_seqs=-1,
                        used_data_keys=None):
    """
    :param bool recurrent_net: If True, the batch might have a batch seq dimension > 1.
      Otherwise, the batch seq dimension is always 1 and multiple seqs will be concatenated.
    :param int|dict[str,int]|NumbersDict batch_size: Max number of frames in one batch.
    :param int|dict[str,int]|NumbersDict max_pad_size: Max number of zero-padded frames in one batch.
    :param int max_seqs: Max number of seqs per batch.
    :param int max_total_num_seqs:
    :param int|dict[str,int]|NumbersDict max_seq_length:
    :param set(str)|None used_data_keys:
    """
    if not batch_size:
      batch_size = sys.maxsize
    batch_size = NumbersDict(batch_size)
    assert not batch_size.any_compare(NumbersDict(0), (lambda a, b: a <= b))
    max_pad_size = NumbersDict(max_pad_size)
    if max_seqs == -1:
      max_seqs = float('inf')
    if not max_seq_length:
      max_seq_length = sys.maxsize
    if isinstance(max_seq_length, int) and max_seq_length < 0:
      max_seq_length = {"classes": -max_seq_length}
    max_seq_length = NumbersDict(max_seq_length)
    min_seq_length = NumbersDict(min_seq_length)
    assert max_seqs > 0
    assert seq_drop <= 1.0
    if not max_total_num_seqs or max_total_num_seqs < 0:
      max_total_num_seqs = float("inf")
    chunk_size = self.chunk_size
    chunk_step = self.chunk_step
    if not recurrent_net:
      if chunk_size != 0:
        print("Non-recurrent network, chunk size %s:%s ignored" % (chunk_size, chunk_step), file=log.v4)
        chunk_size = 0
    batch = Batch()
    total_num_seqs = 0
    last_seq_idx = -1
    avg_weight = sum([v[0] for v in self.weights.values()]) / (len(self.weights.keys()) or 1)
    for idx in self.weights:
      self.weights[idx][1] = random() * avg_weight * pruning
      self.weights[idx][0] *= (1. + pruning)
    for seq_idx, t_start, t_end in self.iterate_seqs(
          chunk_size=chunk_size, chunk_step=chunk_step, used_data_keys=used_data_keys):
      if not self.sample(seq_idx):
        continue
      if total_num_seqs > max_total_num_seqs:
        break
      t_start -= self.ctx_left
      t_end += self.ctx_right
      if recurrent_net:
        length = t_end - t_start
        if length.any_compare(max_seq_length, (lambda a, b: a > b)):
          continue
        if length.any_compare(min_seq_length, (lambda a, b: a < b)):
          continue
        if length.any_compare(batch_size, (lambda a, b: a > b)):
          print("warning: sequence length (%r) larger than limit (%r)" % (length, batch_size), file=log.v4)
        if self.rnd_seq_drop.random() < seq_drop:
          continue
        dt, ds = batch.try_sequence_as_slice(length)
        if batch.num_slices >= 1:
          if (dt * ds).any_compare(batch_size, (lambda a, b: a > b)):
            yield batch
            batch = Batch()
          elif ds > max_seqs:
            yield batch
            batch = Batch()
          elif (dt * ds - batch.get_total_num_frames() - length).any_compare(max_pad_size, (lambda a, b: a > b)):
            yield batch
            batch = Batch()
        batch.add_sequence_as_slice(seq_idx=seq_idx, seq_start_frame=t_start, length=length)
      else:  # Not recurrent.
        while t_start.max_value() < t_end.max_value():
          length = t_end - t_start
          num_frames = NumbersDict.min(
            [length, batch_size.copy_like(length) - batch.get_all_slices_num_frames().copy_like(length)])
          assert num_frames.max_value() > 0
          batch.add_frames(seq_idx=seq_idx, seq_start_frame=t_start, length=num_frames)
          if (batch.get_all_slices_num_frames().any_compare(batch_size, (lambda a, b: a >= b))
                  or batch.get_num_seqs() > max_seqs):
            yield batch
            batch = Batch()
          t_start += num_frames
      if seq_idx != last_seq_idx:
        last_seq_idx = seq_idx
        total_num_seqs += 1

    if batch.get_all_slices_num_frames().max_value() > 0:
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
    """
    :param list[EngineBatch.Batch] batches:
    :param str data_key:
    :return: shape as (time, batch)
    :rtype: (int, int)
    """
    shape = [0, 0]  # time, batch
    for batch in batches:
      shape = [max(shape[0], batch.max_num_frames_per_slice[data_key]), shape[1] + batch.num_slices]
    return tuple(shape)


class DatasetSeq:
  """
  Encapsulates all data for one sequence.
  """

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
    """
    :param str key:
    :rtype: numpy.ndarray
    """
    return self.features[key]

  def get_data_keys(self):
    """
    :rtype: set[str]
    """
    return self.features.keys()

  def __repr__(self):
    return "<DataCache seq_idx=%i>" % self.seq_idx


def get_dataset_class(name):
  """
  :param str name:
  :rtype: type[Dataset]
  """
  from importlib import import_module
  # Only those modules which make sense to be loaded by the user,
  # because this function is only used for such cases.
  mod_names = [
    "hdf", "sprint", "generating", "numpy_dump",
    "meta", "lm", "stereo", "raw_wav"]
  for mod_name in mod_names:
    mod = import_module("returnn.datasets.%s" % mod_name)
    if name in vars(mod):
      clazz = getattr(mod, name)
      assert issubclass(clazz, Dataset)
      return clazz
  return None


def init_dataset(kwargs, extra_kwargs=None, default_kwargs=None):
  """
  :param dict[str]|str|(()->dict[str])|Dataset kwargs:
  :param dict[str]|None extra_kwargs:
  :param dict[str]|None default_kwargs:
  :rtype: Dataset
  """
  assert kwargs
  if isinstance(kwargs, Dataset):
    return kwargs
  if callable(kwargs):
    return init_dataset(kwargs(), extra_kwargs=extra_kwargs, default_kwargs=default_kwargs)
  if isinstance(kwargs, (str, unicode)):
    if kwargs.startswith("{"):
      kwargs = eval(kwargs)
    elif kwargs.startswith("config:"):
      from returnn.config import get_global_config
      config = get_global_config()
      data = eval(kwargs[len("config:"):], config.typed_dict, config.typed_dict)
      return init_dataset(data, extra_kwargs=extra_kwargs, default_kwargs=default_kwargs)
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
  from returnn.datasets.hdf import HDFDataset
  if config_str.startswith("sprint:"):
    kwargs["sprintConfigStr"] = config_str[len("sprint:"):]
    assert config, "need config for dataset in 'sprint:...' format. or use 'ExternSprintDataset:...' instead"
    sprint_trainer_exec_path = config.value("sprint_trainer_exec_path", None)
    assert sprint_trainer_exec_path, "specify sprint_trainer_exec_path in config"
    kwargs["sprintTrainerExecPath"] = sprint_trainer_exec_path
    from returnn.datasets.sprint import ExternSprintDataset
    cls = ExternSprintDataset
  elif config_str.startswith("config:"):
    from returnn.config import get_global_config
    if not config:
      config = get_global_config()
    data = eval(config_str[len("config:"):], config.typed_dict, config.typed_dict)
    return init_dataset(data, extra_kwargs=kwargs)
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
      v = (v, 2 if k == "data" else 1)
      data_dims[k] = v
    if isinstance(v, dict) and leave_dict_as_is:
      continue
    assert isinstance(v, (tuple, list))
    data_dims[k] = tuple(v)
    assert len(v) == 2
    assert isinstance(v[0], int)
    assert isinstance(v[1], int)
    assert 1 <= v[1]
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
  all_data_keys = set(data_keys)

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
      assert all([n is not None for n in data_shape]), "data %r" % extern_data.data[k]
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
  from returnn.util import BackendEngine
  if BackendEngine.is_tensorflow_selected():
    # TF supports more fine-grained specification,
    # however the dataset does not store that in num_outputs.
    from returnn.tf.network import ExternData
    config.set("extern_data", {
      key: ExternData.data_kwargs_from_dataset_key(dataset=dataset, key=key)
      for key in dataset.get_data_keys()})
  else:
    config.set("num_inputs", dataset.num_inputs)
    config.set("num_outputs", dataset.num_outputs)
