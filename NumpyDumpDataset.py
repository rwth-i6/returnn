
"""
Provides :class:`NumpyDumpDataset`.
"""

from Dataset import Dataset, DatasetSeq
import os
import numpy
import typing


class NumpyDumpDataset(Dataset):
  """
  For ``tools/dump-dataset.py --type=numpy``.
  """

  file_format_data = "%i.data"
  file_format_targets = "%i.targets"

  def __init__(self, prefix, postfix=".txt.gz",
               start_seq=0, end_seq=None,
               num_inputs=None, num_outputs=None, **kwargs):
    super(NumpyDumpDataset, self).__init__(**kwargs)
    self.file_format_data = prefix + self.file_format_data + postfix
    self.file_format_targets = prefix + self.file_format_targets + postfix
    self.start_seq = start_seq
    self._init_num_seqs(end_seq)
    self._seq_index = None
    self.cached_seqs = []  # type: typing.List[DatasetSeq]
    self.num_inputs = num_inputs
    self.num_outputs = num_outputs
    assert num_inputs and num_outputs

  def _init_num_seqs(self, end_seq=None):
    last_seq = None
    i = self.start_seq
    while True:
      if end_seq is not None and i >= end_seq:
        break
      if not os.path.exists(self.file_format_data % i):
        break
      if not os.path.exists(self.file_format_targets % i):
        break
      last_seq = i
      i += 1
    if end_seq is None:
      assert last_seq is not None, "None found. Check %s." % (self.file_format_data % self.start_seq)
      end_seq = last_seq
    else:
      assert last_seq == end_seq - 1, "Check %s." % (self.file_format_data % end_seq)
    assert end_seq > self.start_seq
    self._num_seqs = end_seq - self.start_seq

  def _load_numpy_seq(self, seq_idx):
    """
    :param int seq_idx:
    """
    real_idx = self._seq_index[seq_idx]
    features = numpy.loadtxt(self.file_format_data % real_idx)
    targets = numpy.loadtxt(self.file_format_targets % real_idx)
    assert features.ndim == 2
    assert features.shape[1] == self.num_inputs
    assert targets.ndim == 1
    self._add_cache_seq(seq_idx, features, targets)

  # ------------ Dataset API --------------

  def init_seq_order(self, epoch=None, seq_list=None):
    """
    :param int|None epoch:
    :param list[str]|None seq_list:
    :rtype: bool
    """
    super(NumpyDumpDataset, self).init_seq_order(epoch=epoch, seq_list=seq_list)
    if seq_list:
      raise NotImplementedError
    if self.seq_ordering == "sorted":  # not supported atm
      self.seq_ordering = "default"
    self._seq_index = [i + self.start_seq for i in self.get_seq_order_for_epoch(epoch, self.num_seqs)]
    self.cached_seqs[:] = []
    return True

  def _load_seqs(self, start, end):
    """
    :param int start:
    :param int end:
    """
    self._cleanup_old_seq_cache(start)
    for i in range(start, end):
      if not self._have_cache_seq(i):
        self._load_numpy_seq(i)

  def get_input_data(self, seq_idx):
    """
    :param int seq_idx:
    :rtype: numpy.ndarray
    """
    return self._get_cache_seq(seq_idx).features

  def get_targets(self, target, seq_idx):
    """
    :param str target:
    :param int seq_idx:
    :rtype: numpy.ndarray
    """
    return self._get_cache_seq(seq_idx).targets.get(target, None)

  def get_ctc_targets(self, seq_idx):
    """
    Not implemented.

    :param int seq_idx:
    """
    assert False, "No CTC targets."

  def get_seq_length(self, seq_idx):
    """
    :param int seq_idx:
    :rtype: Util.NumbersDict
    """
    # This is different from the other get_* functions.
    # load_seqs() might not have been called before.
    if not self._have_cache_seq(seq_idx):
      self._load_numpy_seq(seq_idx)
    return self._get_cache_seq(seq_idx).num_frames

  @property
  def num_seqs(self):
    """
    :rtype: int
    """
    return self._num_seqs

  def len_info(self):
    """
    :rtype: str
    """
    return "%s, %i seqs" % (self.__class__.__name__, self.num_seqs)

  # ------------ Seq cache management -----------

  def _cleanup_old_seq_cache(self, seq_end):
    i = 0
    while i < len(self.cached_seqs):
      if self.cached_seqs[i].seq_idx >= seq_end:
        break
      i += 1
    del self.cached_seqs[:i]

  def _get_cache_seq(self, seq_idx, error_not_found=True):
    for data in self.cached_seqs:
      if data.seq_idx == seq_idx:
        return data
    if error_not_found:
      raise Exception("seq %i not loaded" % seq_idx)
    else:
      return None

  def _have_cache_seq(self, seq_idx):
    return self._get_cache_seq(seq_idx, error_not_found=False) is not None

  def _get_cache_last_seq_idx(self):
    if self.cached_seqs:
      return self.cached_seqs[-1].seq_idx
    else:
      return -1

  def _add_cache_seq(self, seq_idx, features, targets):
    last_seq_idx = self._get_cache_last_seq_idx()
    assert seq_idx == last_seq_idx + 1
    self.cached_seqs += [DatasetSeq(seq_idx, features, targets)]

