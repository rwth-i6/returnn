"""
Wrapper for RETURNN datasets.

We make use of TorchData data pipelines.
"""

from __future__ import annotations
from typing import Callable, Optional, Iterable, Dict
import numpy
import torch.utils.data
from returnn.datasets.basic import Dataset as ReturnnDataset
from returnn.datasets.util.strings import str_to_numpy_array

ResetCallbackT = Callable[[], None]


class ReturnnDatasetResetDefaultEpochCounterCallback:
    """
    Default for reset_callback.
    Has an internal counter for the epoch, starting at epoch 1 (RETURNN convention).
    """

    def __init__(self, dataset: ReturnnDataset):
        self.dataset = dataset
        self.epoch = 0  # next __call__ will increment, thus we start at epoch 1

    def __call__(self):
        # dataset is likely a copy of the original dataset, either in the main process or in a worker process
        self.epoch += 1
        self.dataset.init_seq_order(epoch=self.epoch)


class ReturnnDatasetResetMpSharedEpochCallback:
    """
    Can be used as reset_callback.
    """

    def __init__(self, dataset: ReturnnDataset, epoch_mp_shared: torch.multiprocessing.Value):
        self.dataset = dataset
        self.epoch_mp_shared = epoch_mp_shared

    def __call__(self):
        # dataset is likely a copy of the original dataset, either in the main process or in a worker process
        # Use epoch_mp_shared to get the current epoch correctly in worked processes
        epoch = self.epoch_mp_shared.value
        self.dataset.init_seq_order(epoch=epoch)


class ReturnnDatasetIterDataPipe(torch.utils.data.IterDataPipe):
    """
    Converts a RETURNN dataset into a PyTorch IterableDataset.
    """

    def __init__(self, returnn_dataset: ReturnnDataset, *, reset_callback: Optional[ResetCallbackT] = None):
        """
        :param returnn_dataset: dataset to be wrapped
        :param reset_callback: callback function to be called when the dataset is reset, e.g. to init the epoch.
            ReturnnDatasetResetDefaultEpochCounterCallback(returnn_dataset) is the default.
        """
        self._dataset = returnn_dataset
        if not reset_callback:
            reset_callback = ReturnnDatasetResetDefaultEpochCounterCallback(returnn_dataset)
        self._reset_callback = reset_callback

    def reset(self):
        """
        :return:
        """
        self._reset_callback()

    def __iter__(self) -> Iterable[Dict[str, numpy.ndarray]]:
        """
        :return: generator providing data samples in the form of a dict data_key -> data
        """
        data_keys = self._dataset.get_data_keys()

        seq_index = 0
        while self._dataset.is_less_than_num_seqs(seq_index):
            self._dataset.load_seqs(seq_index, seq_index + 1)
            data = {data_key: self._dataset.get_data(seq_index, data_key) for data_key in data_keys}
            data["seq_tag"] = str_to_numpy_array(self._dataset.get_tag(seq_index))
            yield data
            seq_index += 1

    def __getitem__(self, index):
        raise Exception(f"{self.__class__.__name__}.__getitem__ not supported")


class ReturnnDatasetPerEpochMapDataPipe(torch.utils.data.MapDataPipe):
    """
    Converts a RETURNN dataset into a PyTorch map-style Dataset.
    """

    def __int__(self, returnn_dataset: ReturnnDataset, *, reset_callback: Optional[ResetCallbackT] = None):
        """
        :param returnn_dataset: dataset to be wrapped
        :param reset_callback: callback function to be called when the dataset is reset, e.g. to init the epoch.
            ReturnnDatasetResetDefaultEpochCounterCallback(returnn_dataset) is the default.
        """
        assert returnn_dataset.have_corpus_seq_idx() and returnn_dataset.have_get_corpus_seq()
        self._dataset = returnn_dataset
        if not reset_callback:
            reset_callback = ReturnnDatasetResetDefaultEpochCounterCallback(returnn_dataset)
        self._reset_callback = reset_callback

    def reset(self):
        """
        :return:
        """
        self._reset_callback()

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


class ReturnnDatasetFullMapDataPipe(torch.utils.data.MapDataPipe):
    """
    Converts a RETURNN dataset into a PyTorch map-style Dataset.
    This is over the full dataset, using the default ordering.
    RETURNN-dataset-side sorting/shuffling is not supported here.
    Sorting/shuffling is intended to be done in the further PyTorch data pipeline.
    """

    def __int__(self, returnn_dataset: ReturnnDataset):
        """
        :param returnn_dataset: dataset to be wrapped
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
