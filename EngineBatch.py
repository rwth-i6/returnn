
"""
Defines :class:`BatchSeqCopyPart` and other batch related helpers.
This is shared across different backends.
"""

import random
import typing
from Util import NumbersDict


class BatchSeqCopyPart:
  """
  A batch used for training in CRNN can consist of several parts from sequences,
   ordered in various ways. The dataset, depending on the configuration, can
   generate these. For the non-recurrent case, we usually concatenate
   them together into one slice. For the recurrent case, we have a single
   slice per sequence, or even multiple slices for a sequence in case of chunking.
  This class represents one single such part and where it is going to
   be stored in the batch.
  """

  def __init__(self, seq_idx, seq_start_frame, seq_end_frame,
               batch_slice, batch_frame_offset):
    """
    :type seq_idx: int
    :type seq_start_frame: NumbersDict | int
    :type seq_end_frame: NumbersDict | int
      Frame idx are input seq, output seq.
    :type batch_slice: int
    :type batch_frame_offset: int | NumbersDict
    """
    self.seq_idx = seq_idx
    self.seq_start_frame = NumbersDict(seq_start_frame)
    self.seq_end_frame = NumbersDict(seq_end_frame)
    self.batch_slice = batch_slice
    self.batch_frame_offset = NumbersDict(batch_frame_offset)
    assert self.seq_start_frame.has_values()
    assert self.seq_end_frame.has_values()
    assert self.batch_frame_offset.has_values()

  @property
  def frame_length(self):
    """
    :rtype: NumbersDict
    """
    return self.seq_end_frame - self.seq_start_frame

  def __repr__(self):
    keys = ("seq_idx", "seq_start_frame", "seq_end_frame", "batch_slice", "batch_frame_offset")
    return "<BatchSeqCopyPart %s>" % " ".join(["%s=%r" % (k, getattr(self, k)) for k in keys])


class Batch:
  """
  A batch can consists of several sequences (= segments).
  This is basically just a list of BatchSeqCopyPart.
  """

  def __init__(self):
    self.max_num_frames_per_slice = NumbersDict(0)
    self.num_slices = 0
    # original data_shape = [0, 0], format (time,batch/slice)
    #          data_shape = [max_num_frames_per_slice, num_slices]
    self.seqs = []  # type: typing.List[BatchSeqCopyPart]

  def __repr__(self):
    return "<Batch start_seq:%r, len(seqs):%i>" % (self.start_seq, len(self.seqs))

  def try_sequence_as_slice(self, length):
    """
    :param NumbersDict length: number of (time) frames
    :return: new shape which covers the old shape and one more data-batch, format (time,batch)
    :rtype: (NumbersDict,int)
    """
    return [NumbersDict.max([self.max_num_frames_per_slice, length]), self.num_slices + 1]

  def add_sequence_as_slice(self, seq_idx, seq_start_frame, length):
    """
    Adds one data-batch in an additional slice.

    :param int seq_idx:
    :param NumbersDict|int seq_start_frame:
    :param NumbersDict length: number of (time) frames
    """
    self.max_num_frames_per_slice, self.num_slices = self.try_sequence_as_slice(length)
    self.seqs += [BatchSeqCopyPart(seq_idx=seq_idx,
                                   seq_start_frame=seq_start_frame,
                                   seq_end_frame=seq_start_frame + length,
                                   batch_slice=self.num_slices - 1,
                                   batch_frame_offset=0)]

  def add_frames(self, seq_idx, seq_start_frame, length, frame_dim_corresponds=True):
    """
    Adds frames to all data-batches.
    Will add one data-batch if we don't have one yet.

    :param int seq_idx:
    :param NumbersDict|int seq_start_frame:
    :param NumbersDict length: number of (time) frames
    :param bool frame_dim_corresponds: if the batch frame offset should always be the same (max value) for all keys
    """
    batch_frame_offset = self.max_num_frames_per_slice
    if frame_dim_corresponds:
      batch_frame_offset = NumbersDict(batch_frame_offset.max_value())
      self.max_num_frames_per_slice = NumbersDict(self.max_num_frames_per_slice.max_value())
    self.max_num_frames_per_slice += length
    self.num_slices = max(self.num_slices, 1)
    self.seqs += [BatchSeqCopyPart(seq_idx=seq_idx,
                                   seq_start_frame=seq_start_frame,
                                   seq_end_frame=seq_start_frame + length,
                                   batch_slice=0,
                                   batch_frame_offset=batch_frame_offset)]

  def init_with_one_full_sequence(self, seq_idx, dataset):
    """
    :param int seq_idx:
    :param Dataset.Dataset dataset:
    """
    assert not self.seqs
    start, end = dataset.get_start_end_frames_full_seq(seq_idx)
    self.add_frames(seq_idx=seq_idx, seq_start_frame=start, length=end - start)

  def get_all_slices_num_frames(self):
    """
    Note that this is only an upper limit in case of data_shape[1] > 1
    because data_shape[0] is the max frame len of all seqs.

    :return: related to the data-key with max length
    :rtype: NumbersDict
    """
    return self.max_num_frames_per_slice * self.num_slices

  def get_total_num_frames(self):
    """
    :rtype: NumbersDict
    """
    return sum([s.frame_length for s in self.seqs])

  @property
  def start_seq(self):
    """
    :rtype: int|None
    """
    if not self.seqs:
      return None
    return min([s.seq_idx for s in self.seqs])

  @property
  def end_seq(self):
    """
    :rtype: int|None
    """
    if not self.seqs:
      return None
    return max([s.seq_idx for s in self.seqs]) + 1

  def get_num_seqs(self):
    """
    :rtype: int
    """
    if not self.seqs:
      return 0
    return self.end_seq - self.start_seq


class BatchSetGenerator:
  """
  This will give you the next batches (list[Batch]) such that you can use them for assign_dev_data().
  We get those batches from a generator, i.e. lazily on-the-fly. This is the whole point of BatchSetGenerator
  - that we must not know the whole list of batches in advance.
  As assign_dev_data() can fail for various reasons, we buffer the list of batches and
  you call self.advance() explicitly to go forward to next batches.
  """

  def __init__(self, dataset, generator, shuffle_batches=False, cache_whole_epoch=True):
    """
    :type dataset: Dataset.Dataset
    :type generator: typing.Generator[Batch]|typing.Iterator[Batch]
    :param bool shuffle_batches:
    :param bool cache_whole_epoch:
    """
    self.dataset = dataset
    self.generator = generator
    self.shuffle_batches = shuffle_batches
    # In some cases, it might be faster to cache the list of batches.
    self.cache_whole_epoch = cache_whole_epoch
    self.cache = []  # type: typing.List[Batch]
    self.buffer = []  # type: typing.List[Batch]
    self.last_batch = None  # type: typing.Optional[Batch]
    self.reached_end = False
    random.seed(1234)
    self._reset()

  def _reset(self):
    self.buffer = self.cache[:]
    if self.shuffle_batches:
      random.shuffle(self.buffer)
    self.cache_active = self.reached_end
    self.reached_end = False
    self.last_batch = None  # type: typing.Optional[Batch]
    self.current_batch_idx = 0

  def reset(self):
    """
    Call this after one epoch to reuse the previously cached batches.
    """
    assert self.cache_whole_epoch
    self._reset()

  def _read_next(self):
    if self.reached_end:
      return False
    try:
      batch = next(self.generator)
    except StopIteration:
      self.reached_end = True
      return False
    else:
      self.buffer += [batch]
      if self.cache_whole_epoch and not self.cache_active:
        self.cache += [batch]
      return True

  def _read_next_up_to_n(self, n):
    for i in range(n):
      if len(self.buffer) >= n:
        break
      if not self._read_next():
        break

  def peek_next_n(self, n):
    """
    :rtype: list[Batch]
    :returns it might return less. There is no way to know in advance.
    If self.has_more() is True, it will at least return one.
    """
    self._read_next_up_to_n(n)
    return self.buffer[:n]

  def advance(self, n):
    """
    :type n: int
    """
    assert n > 0
    self._read_next_up_to_n(n)
    assert n <= len(self.buffer)
    self.last_batch = self.buffer[n - 1]
    self.buffer = self.buffer[n:]
    self.current_batch_idx += n

  def completed_frac(self):
    """
    :rtype: float
    :returns 0-1, >0
    """
    if self.cache_active:
      return self.dataset.generic_complete_frac(self.current_batch_idx, len(self.cache))
    if not self.last_batch:
      return self.dataset.generic_complete_frac(0, None)
    # We cannot use the batch idx because we don't know the number
    # of batches in advance. Thus, we use the seq idx instead.
    # It's good enough.
    return self.dataset.get_complete_frac(self.last_batch.start_seq)

  def has_more(self):
    """
    This would also try to advance further in the dataset, thus it might block.
    If it returns False, no more data is available in the dataset.

    :rtype: bool
    """
    if len(self.buffer) > 0:
      return True
    return self._read_next()

  def get_current_batch_idx(self):
    """
    :rtype: int
    """
    return self.current_batch_idx
