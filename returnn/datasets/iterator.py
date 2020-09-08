from returnn.datasets.basic import DatasetSeq
from returnn.datasets.cached2 import CachedDataset2


class IteratorDatasetBase(CachedDataset2):

  def __init__(self, name=None, num_outputs=None):
    super(CachedDataset2, self).__init__(
      name=name
    )

    self.num_outputs = num_outputs or {}

    self.last_sequence_idx = -1
    self.last_sequence_buffer = None

  def __next__(self):
    """
    :return: next sequence of the dataset.
    :rtype DatasetSeq
    """
    raise NotImplementedError

  def seek_epoch(self, epoch=None):
    """
    Can be used to reset the generator to start a specific epoch

    :param int epoch:
    :return:
    """
    pass

  # Internal Functions, do not override

  def init_seq_order(self, epoch=None, seq_list=None, seq_order=None):
    """

    :param int|None epoch:
    :param list[str]|None seq_list: List of sequence tags, to set a predefined order.
    :param list[int]|None seq_order: List of corpus sequence indices, to set a predefined order.
    :rtype: bool
    :returns whether the order changed (True is always safe to return)
    """
    super(IteratorDatasetBase, self).init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)

    if seq_list or seq_order:
      raise NotImplementedError

    self.seek_epoch(epoch)

    if self.last_sequence_buffer is None:
      self.last_sequence_idx = 0
      try:
        self.last_sequence_buffer = next(self)
      except StopIteration:
        assert False, "The dataset has not data"

  def finish_epoch(self):
    self.last_sequence_idx = 0
    self.last_sequence_buffer = None

  def _collect_single_seq(self, seq_idx):
    """

    :param int seq_idx: sorted seq idx
    :return:
    """
    assert seq_idx == self.last_sequence_idx
    try:
      next_seq = next(self)
      self.last_sequence_idx += 1
      current_seq = self.last_sequence_buffer
      self.last_sequence_buffer = next_seq
    except StopIteration:
      current_seq = self.last_sequence_buffer

    return current_seq

  def is_less_than_num_seqs(self, n):
    """
    :param int n:
    :rtype: bool
    """
    return n <= self.last_sequence_idx

  def get_tag(self, sorted_seq_idx):
    """

    :param sorted_seq_idx:
    :return:
    """
    raise NotImplementedError

  def get_current_seq_order(self):
    """
    :rtype: list[int]
    """
    raise NotImplementedError

  def get_data_dim(self, key):
    """
    :param str key: e.g. "data" or "classes"
    :return: number of classes, no matter if sparse or not
    :rtype: int
    """
    if key in self.num_outputs:
      return self.num_outputs[key].get("dim", 1)
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
      return self.num_outputs[key].get("sparse", False)
    return False

  def get_data_shape(self, key):
    """
    :returns get_data(*, key).shape[1:], i.e. num-frames excluded
    :rtype: list[int]
    """
    if key in self.num_outputs:
      if "shape" in self.num_outputs[key].keys():
        if self.num_outputs[key]["shape"][0] is None:
          return self.num_outputs[key]["shape"][1:]
        else:
          assert False, "data shape has no time axis, calling get_data_shape is not possible"
    if self.is_data_sparse(key):
      return []
    return [self.get_data_dim(key)]
