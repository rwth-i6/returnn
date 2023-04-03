"""
Code to create PyTorch datasets that can be used with the PyTorch DataLoader.

We make use of TorchData data pipelines.

Most functionality is implemented as a dataset/datapipe, as this seems to be the common way in PyTorch,
as it is also commonly done in Fairseq:
    https://github.com/facebookresearch/fairseq/tree/main/fairseq/data
    https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/subsample_dataset.py

This is also the intended way for TorchData.

We potentially could also implement some functionality as part of the data loader (v1),
but DataLoader2 suggests to decouple this, as we do here.

We also have :class:`ChunkShuffleDataset` on RETURNN dataset level.
However, having this separate pure PyTorch implementation is useful to allow to use
other PyTorch datasets more directly, including also HuggingFace datasets.
"""

from __future__ import annotations
from typing import List, Dict
import sys
from copy import deepcopy

import numpy
import torch
import torch.utils.data

from returnn.util.basic import NumbersDict


def create_tensor(array: numpy.ndarray) -> torch.Tensor:
    """
    Adjust non-supported dtypes

    :param array: numpy array to be converted
    """
    # The only supported PyTorch dtypes are:
    # float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool.
    if array.dtype == numpy.uint32:
        array = numpy.asarray(array, dtype=numpy.int64)
    return torch.tensor(array)


def collate_batch(batch: List[Dict[str, numpy.ndarray]]) -> Dict[str, torch.Tensor]:
    """
    :param batch:
    """
    assert isinstance(batch, list)
    assert batch, "batch is empty?"
    assert isinstance(batch[0], dict)
    data_keys = list(batch[0].keys())

    res = {}
    for key in data_keys:
        ls = [create_tensor(sample[key]) for sample in batch]
        padded = torch.nn.utils.rnn.pad_sequence(ls, batch_first=True, padding_value=0)
        res[key] = padded
        res["%s:seq_len" % key] = torch.tensor([v.shape[0] for v in ls])

    return res


class ChunkingIterDataPipe(torch.utils.data.IterDataPipe):
    """
    Splits each sequence in the given dataset into chunks according to the 'chunking' config option.
    So it transforms one sequences into multiple sequences.
    """

    def __init__(self, dataset: torch.utils.data.IterableDataset, chunking):
        """
        :param dataset: dataset to apply chunking to
        :param None|int|(int,int)|dict|(dict,dict) chunking: tuple (chunk_size, chunk_step).
            If given as single value,
            value will be used for both.
            Both chunk_size and chunk_step can be given as a dict data_key -> size/step.
            This can be used to apply chunking to only a subset of all data keys,
            or to use different chunking for different
            data keys.
            (The number of resulting chunks has to be match though for all given data keys, i.e. sequence lengths
            have to be considered.)
        """
        super().__init__()
        from returnn.datasets.basic import Dataset as ReturnnDataset

        self._dataset = dataset
        # noinspection PyProtectedMember
        self._chunk_size, self._chunk_step, custom_chunk_func = ReturnnDataset._parse_chunking(chunking)
        assert not custom_chunk_func, f"Custom chunking function not supported, {chunking!r}"

    def __iter__(self):
        """
        :return: generator providing chunks in the form of a dict data_key -> data chunk
        :rtype: Iterable[dict[str, numpy.ndarray]]
        """
        chunking_data_keys = list(self._chunk_size.keys())

        for data_dict in self._dataset:

            if not chunking_data_keys:
                chunking_data_keys = list(data_dict.keys())  # use all if not configured separately
                assert chunking_data_keys, "Dataset produced sequence without any data."

            data_chunks = {}
            num_chunks = None

            for data_key in chunking_data_keys:
                chunk_size = self._chunk_size[data_key]
                chunk_step = self._chunk_step[data_key]

                data = data_dict[data_key]
                chunks = [
                    data[start_index : start_index + chunk_size] for start_index in range(0, len(data), chunk_step)
                ]

                if num_chunks is None:
                    num_chunks = len(chunks)
                else:
                    assert num_chunks == len(
                        chunks
                    ), "Chunking resulted in different number of chunks for different data keys."

                data_chunks[data_key] = chunks

            assert num_chunks, "Bug: no chunk produced from current sequence."
            for chunk_index in range(num_chunks):
                chunk_data = {data_key: data_chunks[data_key][chunk_index] for data_key in data_chunks.keys()}

                # If chunking is configured using a dict,
                # i.e. with explicit data keys, there might be remaining data keys
                # for which we yield the full sequence in each chunk.
                non_chunked_data = {
                    data_key: data for data_key, data in data_dict.items() if data_key not in chunk_data
                }
                if non_chunked_data:
                    chunk_data.update(deepcopy(non_chunked_data))

                yield chunk_data

    def __getitem__(self, index):
        raise Exception(f"{self.__class__.__name__}.__getitem__ not supported")

    @staticmethod
    def _parse_chunking(chunking):
        """
        Similar to returnn.datasets.basic.Dataset._parse_chunking().

        :param None|int|(int,int)|dict|(dict,dict) chunking: see __init__()
        :return: chunk_size, chunk_step
        :rtype: (NumbersDict,NumbersDict)
        """
        if not isinstance(chunking, (tuple, list)):
            chunking = (chunking, None)
        chunk_size, chunk_step = chunking
        if chunk_size is None:
            chunk_size = 0
        assert isinstance(chunk_size, (int, dict))
        chunk_size = NumbersDict(chunk_size)
        assert chunk_size.min_value() > 0, "chunk size must not be negative"
        if chunk_step in (None, 0):
            chunk_step = chunk_size
        assert isinstance(chunk_step, (int, dict, NumbersDict))
        chunk_step = NumbersDict(chunk_step)
        assert sorted(chunk_step.keys()) == sorted(chunk_size.keys())
        assert chunk_step.min_value() > 0, "chunking step must be positive"
        return chunk_size, chunk_step


# noinspection PyAbstractClass
class BatchingIterDataPipe(torch.utils.data.IterDataPipe):
    """
    Converts a dataset yielding sequences (dict data_key -> array per sequence) into a dataset yielding lists of
    these sequences, i.e. batches.
    Sequences are grouped in-order according to the 'max_tokens' and 'max_seqs' batch size
    limits.
    Note, that batches are not yet merged into a single (padded) data array here, this happens in 'collate_batch()'.
    """

    def __init__(self, dataset: torch.utils.data.IterableDataset, batch_size=1, max_seqs=None):
        """
        :param dataset: dataset to apply batching to
        :param int|dict[str,int]|None batch_size: Maximum number of time steps (e.g. audio frames / words) in one
            batch (padding included).
            If given as a dict data_key -> value, sets different individual limits per data key.
            If None, no limit.
        :param int|None max_seqs: maximum number of sequences in a batch,
            None means unlimited (also -1 to match TF backend)
        """
        super().__init__()
        self._dataset = dataset
        self._max_batch_size = NumbersDict(sys.maxsize if batch_size is None else batch_size)
        self._max_seqs = sys.maxsize if (max_seqs is None or max_seqs == -1) else max_seqs

        assert self._max_batch_size.min_value() > 0
        assert self._max_seqs > 0

    def __iter__(self):
        """
        :return: generator providing batches in the form of lists of sequences, where each sequence is a dict
          data_key -> data_array.
        :rtype: Iterable[list[dict[str, numpy.ndarray]]]
        """
        current_batch = []
        current_max_sequence_lengths = NumbersDict(0)  # data_key -> length of longest sequence in current batch

        for data_dict in self._dataset:
            if len(current_batch) == self._max_seqs:
                yield current_batch
                current_batch = []
                current_max_sequence_lengths = NumbersDict(0)

            # TODO: This assumes all data has time as first dimension. Currently we can't know better..
            sequence_lengths = NumbersDict({data_key: data.shape[0] for data_key, data in data_dict.items()})

            max_sequence_lengths_if_included = NumbersDict.max([current_max_sequence_lengths, sequence_lengths])
            batch_size_if_included = max_sequence_lengths_if_included * (len(current_batch) + 1)  # including padding

            if current_batch and batch_size_if_included.any_compare(self._max_batch_size, (lambda a, b: a > b)):
                yield current_batch
                current_batch = [data_dict]
                current_max_sequence_lengths = sequence_lengths
            else:
                current_batch.append(data_dict)
                current_max_sequence_lengths = max_sequence_lengths_if_included

        if current_batch:
            yield current_batch
