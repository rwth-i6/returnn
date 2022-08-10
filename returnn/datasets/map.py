"""
Provides :class:`MapDatasetBase`
"""

from returnn.datasets.basic import DatasetSeq
from returnn.datasets.cached2 import CachedDataset2
from returnn.util.basic import OptionalNotImplementedError


class MapDatasetBase(object):
  """
  This dataset can be used as template to implement user-side Datasets, where the data can be access in arbitrary order.
  For global sorting, the length information needs to be known beforehand, see get_seq_len.
  """

  def __init__(self, data_types=None):
    """
    :param dict[str,dict] data_types: data_key -> constructor parameters of Data object, for all data streams the
      dataset provides (inputs and targets). E.g. {'data': {'dim': 1000, 'sparse': True, ...}, 'classes': ...}.
    """
    self.data_types = data_types or {}

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
    :rtype: dict[str,numpy.array]
    """
    raise NotImplementedError

  def get_seq_len(self, seq_idx):
    """
    This optional function provides the sequence length for the `seq_ordering` parameter.
    If not specified only a limited set of options is available.

    :param int seq_idx:
    :return: sequence length
    :rtype: int
    """
    raise OptionalNotImplementedError

  def get_seq_tag(self, seq_idx):
    """
    :param int seq_idx:
    :return: tag for the sequence of the given index, default is 'seq-{seq_idx}'.
    :rtype: str
    """
    return "seq-%i" % seq_idx

  def get_seq_order(self, epoch=None):
    """
    Override to implement a dataset specific sequence order for a given epoch.
    The number of sequences can be less than the total number.
    This will override the effects of `partition_epoch` and `seq_ordering` when using MapDatasetWrapper.

    :param int epoch:
    :return: sequence order (list of sequence indices)
    :rtype: list[int]
    """
    raise OptionalNotImplementedError


class MapDatasetWrapper(CachedDataset2):
  """
  Takes a MapDataset and turns it into a returnn.datasets.Dataset by providing the required class methods.
  """
  def __init__(self, map_dataset, **kwargs):
    """
    :param MapDatasetBase|function map_dataset: the MapDataset to be wrapped
    """
    super(MapDatasetWrapper, self).__init__(**kwargs)

    if callable(map_dataset):
      map_dataset = map_dataset(**kwargs)
    assert isinstance(map_dataset, MapDatasetBase)

    self._dataset = map_dataset
    self._seq_order = None

  @property
  def num_seqs(self):
    """
    :returns number of sequences in the current epoch
    :rtype: int
    """
    if self._seq_order is None:
      raise NotImplementedError("'num_seqs' is only known after calling init_seq_order().")
    return len(self._seq_order)

  def init_seq_order(self, epoch=None, seq_list=None, seq_order=None):
    """
    :param int|None epoch:
    :param list[str]|None seq_list: List of sequence tags, to set a predefined order.
    :param list[int]|None seq_order: List of corpus sequence indices, to set a predefined order.
    :rtype: bool
    :returns whether the order changed (True is always safe to return)
    """
    super(MapDatasetWrapper, self).init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)

    if seq_list is not None or seq_order is not None:
      raise NotImplementedError

    try:
      self._seq_order = self._dataset.get_seq_order(epoch=epoch)
    except OptionalNotImplementedError:
      try:
        self._seq_order = self.get_seq_order_for_epoch(
          epoch=epoch, num_seqs=len(self._dataset), get_seq_len=self._dataset.get_seq_len)
      except OptionalNotImplementedError:
        # only support seq_ordering that need no length here
        assert self.seq_ordering in ["default", "reverse", "random"]
        self._seq_order = self.get_seq_order_for_epoch(
          epoch=epoch, num_seqs=len(self._dataset), get_seq_len=None)

    return True

  def _collect_single_seq(self, seq_idx):
    """
    :param int seq_idx: sorted seq idx
    :return:
    """
    corpus_seq_idx = self.get_corpus_seq_idx(seq_idx)
    return DatasetSeq(
      seq_idx, features=self._dataset[corpus_seq_idx], seq_tag=self._dataset.get_seq_tag(corpus_seq_idx))

  def get_current_seq_order(self):
    """
    :rtype: typing.Sequence[int]
    """
    assert self._seq_order is not None
    return self._seq_order

  def get_tag(self, sorted_seq_idx):
    """
    :param sorted_seq_idx:
    :return:
    """
    seq_tag = self._dataset.get_seq_tag(self.get_corpus_seq_idx(sorted_seq_idx))
    return seq_tag

  def get_corpus_seq_idx(self, sorted_seq_idx):
    """
    :param int sorted_seq_idx:
    :return corpus_seq_idx
    :rtype: int
    """
    return self._seq_order[sorted_seq_idx]

  def have_corpus_seq_idx(self):
    """
    :rtype: bool
    :return: whether you can call self.get_corpus_seq_idx()
    """
    return True

  def get_data_dim(self, key):
    """
    :param str key: e.g. "data" or "classes"
    :return: number of classes, no matter if sparse or not
    :rtype: int
    """
    if key in self._dataset.data_types:
      return self._dataset.data_types[key].get("dim", 1)
    return 1  # unknown

  def get_data_dtype(self, key):
    """
    :param str key: e.g. "data" or "classes"
    :return: dtype as str, e.g. "int32" or "float32"
    :rtype: str
    """
    if key in self._dataset.data_types and "dtype" in self._dataset.data_types[key]:
      return self._dataset.data_types[key]["dtype"]
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
    if key in self._dataset.data_types:
      return self._dataset.data_types[key].get("sparse", False)
    return False

  def get_data_shape(self, key):
    """
    :returns get_data(*, key).shape[1:], i.e. num-frames excluded
    :rtype: list[int]
    """
    if key in self._dataset.data_types:
      if "shape" in self._dataset.data_types[key].keys():
        if self._dataset.data_types[key]["shape"][0] is None:
          return self._dataset.data_types[key]["shape"][1:]
        else:
          assert False, "data shape has no time axis, calling get_data_shape is not possible"
    if self.is_data_sparse(key):
      return []
    return [self.get_data_dim(key)]


class FromListDataset(MapDatasetBase):
  """
  Simple implementation of a MapDataset where all data is given in a list.
  """
  def __init__(self, data_list, sort_data_key=None, **kwargs):
    """
    :param list[dict[str,numpy.ndarray]] data_list: sequence data as a dict data_key -> data for all sequences.
    :param str sort_data_key: Sequence length will be determined from data of this data_key.
    """
    self._data_list = data_list
    self._sort_data_key = sort_data_key
    super(FromListDataset, self).__init__(**kwargs)

  def __len__(self):
    """
    :return: total number of sequences in the dataset
    :rtype: int
    """
    return len(self._data_list)

  def __getitem__(self, seq_idx):
    """
    :param int seq_idx:
    :return: The content of a single dataset entry
    :rtype dict[str,numpy.array]
    """
    return self._data_list[seq_idx]

  def get_seq_len(self, seq_idx):
    """
    :param seq_idx:
    :return: length of data for 'sort_data_key'
    :rtype: int
    """
    assert self._sort_data_key, "Specify which data key should be used for sequence sorting via 'sort_data_key'."
    return len(self._data_list[seq_idx][self._sort_data_key])
