
"""
This file is going to be imported by Debug.debug_shell() and available as interactive commands.
"""

import sys
import numpy
import h5py


def find_obj_in_stack(cls, stack=None, all_threads=True):
  """
  :param type cls:
  :param types.FrameType|traceback.FrameSummary stack:
  :param bool all_threads:
  :return: obj
  """
  if all_threads:
    assert stack is None
    # noinspection PyProtectedMember,PyUnresolvedReferences
    for tid, stack in sys._current_frames().items():
      obj = find_obj_in_stack(cls=cls, stack=stack, all_threads=False)
      if obj is not None:
        return obj
    return None

  assert not all_threads
  if stack is None:
    # noinspection PyProtectedMember,PyUnresolvedReferences
    stack = sys._getframe()
    assert stack, "could not get stack"

  import inspect
  isframe = inspect.isframe

  _tb = stack
  while _tb is not None:
    if isframe(_tb):
      f = _tb
    else:
      f = _tb.tb_frame

    for obj in f.f_locals.values():
      if isinstance(obj, cls):
        return obj

    if isframe(_tb):
      _tb = _tb.f_back
    else:
      _tb = _tb.tb_next

  return None


class SimpleHdf:
  """
  Simple HDF writer.
  """

  def __init__(self, filename):
    self.hdf = h5py.File(filename)
    self.seq_tag_to_idx = {name: i for (i, name) in enumerate(self.hdf["seqTags"])}
    self.num_seqs = len(self.hdf["seqTags"])
    assert self.num_seqs == len(self.seq_tag_to_idx), "not unique seq tags"
    seq_lens = self.hdf["seqLengths"]
    if seq_lens.ndim == 2:
      seq_lens = seq_lens[:, 0]
    assert self.num_seqs == len(seq_lens)
    self.seq_starts = [0] + list(numpy.cumsum(seq_lens))
    total_len = self.seq_starts[-1]
    inputs_len = self.hdf["inputs"].shape[0]
    assert total_len == inputs_len, "time-dim does not match: %i vs %i" % (total_len, inputs_len)
    assert self.seq_starts[-1] == self.hdf["targets/data/classes"].shape[0]

  def get_seq_tags(self):
    """
    :rtype: list[str]
    """
    return self.hdf["seqTags"]

  def get_data(self, seq_idx):
    """
    :param int seq_idx:
    :rtype: numpy.ndarray
    """
    seq_t0, seq_t1 = self.seq_starts[seq_idx:seq_idx + 2]
    return self.hdf["inputs"][seq_t0:seq_t1]

  def get_targets(self, seq_idx):
    """
    :param int seq_idx:
    :rtype: numpy.ndarray
    """
    seq_t0, seq_t1 = self.seq_starts[seq_idx:seq_idx + 2]
    return self.hdf["targets/data/classes"][seq_t0:seq_t1]

  def get_data_dict(self, seq_idx):
    """
    :param int seq_idx:
    :rtype: dict[str,numpy.ndarray]
    """
    return {"data": self.get_data(seq_idx), "classes": self.get_targets(seq_idx)}
