"""
Wrapper for RETURNN datasets
"""

from __future__ import annotations
import torch.utils.data


class IterableDatasetWrapper(torch.utils.data.IterableDataset):
    """
    Converts a RETURNN dataset into a PyTorch IterableDataset.
    """

    def __init__(self, returnn_dataset, epoch):
        """
        :param returnn.datasets.basic.Dataset returnn_dataset: dataset to be wrapped
        :param int epoch:
        """
        self._dataset = returnn_dataset
        self._epoch = epoch

    def __iter__(self):
        """
        :return: generator providing data samples in the form of a dict data_key -> data
        :rtype: Iterable[dict[str, numpy.ndarray]]
        """
        self._dataset.init_seq_order(epoch=self._epoch)

        data_keys = self._dataset.get_data_keys()

        seq_index = 0
        while self._dataset.is_less_than_num_seqs(seq_index):
            self._dataset.load_seqs(seq_index, seq_index + 1)
            data = {data_key: self._dataset.get_data(seq_index, data_key) for data_key in data_keys}
            yield data
            seq_index += 1

    def __getitem__(self, index):
        raise Exception(f"{self.__class__.__name__}.__getitem__ not supported")


class MapStyleDatasetPerEpochWrapper(torch.utils.data.Dataset):
    """
    Converts a RETURNN dataset into a PyTorch map-style Dataset.
    """

    def __int__(self, returnn_dataset, epoch):
        """
        :param returnn.datasets.basic.Dataset returnn_dataset: dataset to be wrapped
        :param int epoch:
        """
        assert returnn_dataset.have_corpus_seq_idx() and returnn_dataset.have_get_corpus_seq()
        self._dataset = returnn_dataset
        self._epoch = epoch
        self._dataset.init_seq_order(epoch=self._epoch)

    def __len__(self):
        """
        :return: number of data samples in the dataset
        :rtype: int
        """
        return self._dataset.num_seqs

    def __getitem__(self, index):
        """
        :param int index:
        :return: data sample in the form of a dict data_key -> data
        :rtype: dict[str, numpy.ndarray]
        """
        corpus_seq_idx = self._dataset.get_corpus_seq_idx(index)
        seq = self._dataset.get_corpus_seq(corpus_seq_idx)
        return seq.features


class MapStyleDatasetFullWrapper(torch.utils.data.Dataset):
    """
    Converts a RETURNN dataset into a PyTorch map-style Dataset.
    """

    def __int__(self, returnn_dataset, epoch):
        """
        :param returnn.datasets.basic.Dataset returnn_dataset: dataset to be wrapped
        """
        assert returnn_dataset.have_get_corpus_seq()
        self._dataset = returnn_dataset

    def __len__(self):
        """
        :return: number of data samples in the dataset
        :rtype: int
        """
        return self._dataset.get_total_num_seqs()

    def __getitem__(self, index):
        """
        :param int index:
        :return: data sample in the form of a dict data_key -> data
        :rtype: dict[str, numpy.ndarray]
        """
        seq = self._dataset.get_corpus_seq(index)
        return seq.features
