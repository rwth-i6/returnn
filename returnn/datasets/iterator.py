from returnn.datasets.basic import DatasetSeq
from returnn.datasets.cached2 import CachedDataset2

class IteratorDatasetBase(CachedDataset2):

  def __init__(self, **kwargs):
    super(CachedDataset2, self).__init__(**kwargs)

    self.num_outputs = {}


  def __next__(self):
    """
    :return: next sequence of the dataset.
    :rtype DatasetSeq
    """
    raise NotImplementedError


  def get_buffer_limit(self, epoch=None):
    """

    :param int epoch:
    :return: maximum number of sequences to cache, -1 for all till iterator ends
    :rtype int
    """
    return 1

  def get_buffer_seq_order(self, seq_idx_list):
    """
    Override to implement a local reordering

    :param list[int] seq_idx_list: idx list of buffer elements for reordering
    :return:
    """
    return seq_idx_list


  def start_at_epoch(self, epoch=None):
    """
    Can be used to reset the generator to start a specific epoch

    :param int epoch:
    :return:
    """
    pass

  def start_at_idx(self, seq_idx=None):
    """
    Can be used to set the generator at a specific running index

    :param seq_idx:
    :return:
    """
    pass





class Test(IteratorDatasetBase)


