"""
Wrapper for RETURNN datasets
"""

import torch.utils.data


class DatasetWrapper(torch.utils.data.IterableDataset):
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
