from returnn.datasets.basic import DatasetSeq
from returnn.datasets.cached2 import CachedDataset2


class MapDataset(CachedDataset2):
  """
  This dataset can be used to implement user-side Datasets


  """

  def __init__(self, name=None, seq_ordering='default', random_seed_offset=None,
               partition_epoch=None):
    """

    :param name:
    :param seq_ordering:
    :param random_seed_offset:
    :param partition_epoch:
    """
    super(MapDataset, self).__init__(
      name=name,
      seq_ordering=seq_ordering,
      random_seed_offset=random_seed_offset,
      partition_epoch=partition_epoch
    )

    self.num_outputs = {}
    self._seq_order = None

  def __len__(self):
    """
    :return: total number of sequences in the dataset
    :rtype: int
    """
    raise NotImplementedError

  def __getitem__(self, seq_idx):
    """

    :param seq_idx int:
    :return: A single dataset entry
    :rtype dict[str,numpy.array]
    """
    raise NotImplementedError

  def get_seq_len(self, seq_idx):
    """

    :param seq_idx:
    :return:
    :rtype: int|None
    """
    return None

  def get_seq_tag(self, seq_idx):
    """

    :param seq_idx int:
    :return:
    :rtype str|None
    """
    return None


  # Internal Functions, do not override

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
    super(MapDataset, self).init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)

    if seq_list or seq_order:
      raise NotImplementedError

    if self.get_seq_len(0) is None:
      # only support seq_ordering that need no length here
      assert self.seq_ordering in ["default", "reverse", "random"]
      self._seq_order = self.get_seq_order_for_epoch(
        epoch=epoch, num_seqs=len(self), get_seq_len=None)
    else:
      self._seq_order = self.get_seq_order_for_epoch(
        epoch=epoch, num_seqs=len(self), get_seq_len=self.get_seq_len)

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
    seq_len = self.get_seq_tag(self.get_corpus_seq_idx(sorted_seq_idx))
    if seq_len is None:
      seq_len = super().get_tag(sorted_seq_idx)
    return seq_len

  def get_corpus_seq_idx(self, sorted_seq_idx):
    """
    :param int sorted_seq_idx:
    :return corpus_seq_idx
    :rtype: int
    """
    return self._seq_order[sorted_seq_idx]

