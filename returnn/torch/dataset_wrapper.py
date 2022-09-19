"""
Code to create PyTorch datasets that can be used with the PyTorch DataLoader.
"""

from torch.utils.data import IterableDataset


# noinspection PyAbstractClass
class DatasetWrapper(IterableDataset):
  """
  Converts a RETURNN dataset into a PyTorch IterableDataset.
  """

  def __init__(self, returnn_dataset):
    """
    :param returnn.datasets.basic.Dataset returnn_dataset: dataset to be wrapped
    """
    self._dataset = returnn_dataset

  def __iter__(self):
    """
    :return: generator providing data samples in the form of a dict data_key -> data
    :rtype: Iterable[dict[str, numpy.ndarray]]
    """
    data_keys = self._dataset.get_data_keys()

    seq_index = 0
    while self._dataset.is_less_than_num_seqs(seq_index):
      self._dataset.load_seqs(seq_index, seq_index + 1)

      data = {data_key: self._dataset.get_data(seq_index, data_key) for data_key in data_keys}

      yield data

      seq_index += 1
