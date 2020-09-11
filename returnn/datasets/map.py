"""
Provides :class:`MapDatasetBase`
"""

from returnn.datasets.basic import DatasetSeq
from returnn.datasets.cached2 import CachedDataset2
from returnn.util.basic import OptionalNotImplementedError


class MapDatasetBase(CachedDataset2):
  """
  This dataset can be used as template to implement user-side Datasets, where the data can be access in arbitrary order.
  For global sorting, the length information needs to be known beforehand.
  """

  def __init__(self, name=None, num_outputs=None, seq_ordering='default', random_seed_offset=None,
               partition_epoch=None):
    """

    :param str name: name of the dataset
    :param dict num_outputs: definition of the data as constructor parameters of a Data object
    :param str seq_ordering: string to define the sequence ordering
    :param int|None random_seed_offset: provide a seed offset for the sequence ordering
    :param int|None partition_epoch:
    """
    super(MapDatasetBase, self).__init__(
      name=name,
      seq_ordering=seq_ordering,
      random_seed_offset=random_seed_offset,
      partition_epoch=partition_epoch
    )

    self.num_outputs = num_outputs or {}
    self._seq_order = None

  def __len__(self):
    """
    :return: total number of sequences in the dataset
    :rtype: int
    """
    raise NotImplementedError

  def __getitem__(self, seq_idx):
    """
    This function does the actual data loading, the order can be arbitrary.

    :param int seq_idx:
    :return: The content of a single dataset entry
    :rtype dict[str,numpy.array]
    """
    raise NotImplementedError

  def get_seq_len(self, seq_idx):
    """
    This optional function provides the sequence length for the `seq_ordering` parameter.
    If not specified only a limited set of options is available.

    :param seq_idx:
    :return:
    :rtype: int|None
    """
    raise OptionalNotImplementedError

  def get_seq_tag(self, seq_idx):
    """
    Optionally return a sequence tag.

    :param int seq_idx:
    :return:
    :rtype str|None
    """
    return "seq-%i" % seq_idx

  def get_seq_order(self, epoch=None):
    """
    Override to implement a custom sequence order for a given epoch.
    The number of sequences can be less than the total number.
    This will override the effects of `partition_epoch` and `seq_ordering`

    :param epoch:
    :return: seq_order
    :rtype list[int]
    """
    raise OptionalNotImplementedError

  # -----------------------------------
  # Internal Functions, do not override
  # -----------------------------------

  @property
  def num_seqs(self):
    """
    :rtype: int
    """
    return len(self)

  def init_seq_order(self, epoch=None, seq_list=None, seq_order=None):
    """

    :param int|None epoch:
    :param list[str]|None seq_list: List of sequence tags, to set a predefined order.
    :param list[int]|None seq_order: List of corpus sequence indices, to set a predefined order.
    :rtype: bool
    :returns whether the order changed (True is always safe to return)
    """
    super(MapDatasetBase, self).init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)

    if seq_list or seq_order:
      raise NotImplementedError

    try:
      self._seq_order = self.get_seq_order_for_epoch(
        epoch=epoch, num_seqs=len(self), get_seq_len=self.get_seq_len)
    except OptionalNotImplementedError:
      # only support seq_ordering that need no length here
      assert self.seq_ordering in ["default", "reverse", "random"]
      self._seq_order = self.get_seq_order_for_epoch(
        epoch=epoch, num_seqs=len(self), get_seq_len=None)

  def _collect_single_seq(self, seq_idx):
    """

    :param int seq_idx: sorted seq idx
    :return:
    """
    corpus_seq_idx = self.get_corpus_seq_idx(seq_idx)
    return DatasetSeq(seq_idx, features=self.__getitem__(corpus_seq_idx), seq_tag=self.get_seq_tag(corpus_seq_idx))

  def get_current_seq_order(self):
    """
    :rtype: list[int]
    """
    assert self._seq_order is not None
    return self._seq_order

  def get_tag(self, sorted_seq_idx):
    """

    :param sorted_seq_idx:
    :return:
    """
    seq_tag = self.get_seq_tag(self.get_corpus_seq_idx(sorted_seq_idx))
    return seq_tag

  def get_corpus_seq_idx(self, sorted_seq_idx):
    """
    :param int sorted_seq_idx:
    :return corpus_seq_idx
    :rtype: int
    """
    return self._seq_order[sorted_seq_idx]

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
