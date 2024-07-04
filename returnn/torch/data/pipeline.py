"""
Code to create PyTorch datasets that can be used with the PyTorch DataLoader.

We make use of torch.utils.data.IterDataPipe data pipelines.
(We used TorchData before but migrated back to pure PyTorch. https://github.com/rwth-i6/returnn/issues/1382)

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
from typing import Optional, Any, Union, List, Dict, Callable
import sys
from copy import deepcopy

import numpy
import torch
import torch.utils.data

from returnn.util.basic import NumbersDict


def create_tensor(array: numpy.ndarray) -> Union[torch.Tensor, numpy.ndarray]:
    """
    Adjust non-supported dtypes

    :param array: numpy array to be converted
    """
    # The only supported PyTorch dtypes are:
    # float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool.
    if array.dtype.kind in "UO":  # string (unicode) or object
        return array  # keep as-is. e.g. seq_tag
    if array.dtype == numpy.uint32:
        array = numpy.asarray(array, dtype=numpy.int64)
    elif array.dtype == numpy.uint16:
        array = numpy.asarray(array, dtype=numpy.int32)
    return torch.tensor(array)


def collate_batch(batch: List[Dict[str, numpy.ndarray]]) -> Dict[str, Union[torch.Tensor, numpy.ndarray]]:
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
        if not ls:
            raise ValueError("batch is empty?")
        if isinstance(ls[0], torch.Tensor):
            if ls[0].ndim > 0:
                padded = torch.nn.utils.rnn.pad_sequence(ls, batch_first=True, padding_value=0)
                res[key] = padded
                res["%s:seq_len" % key] = torch.tensor([v.shape[0] for v in ls], dtype=torch.int32)
            else:
                res[key] = torch.stack(ls, dim=0)
        elif isinstance(ls[0], numpy.ndarray):
            padded = numpy.stack(ls, axis=0)
            res[key] = padded

    return res


class ChunkingIterDataPipe(torch.utils.data.IterDataPipe):
    """
    Splits each sequence in the given dataset into chunks according to the 'chunking' config option.
    So it transforms one sequences into multiple sequences.
    """

    def __init__(self, dataset: torch.utils.data.IterableDataset, chunking, *, min_chunk_size=0):
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
        self._min_chunk_size = NumbersDict(min_chunk_size)

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
                chunking_data_key_black_list = ["seq_tag"]
                for key in chunking_data_key_black_list:
                    if key in chunking_data_keys:
                        chunking_data_keys.remove(key)
                assert chunking_data_keys, "Dataset produced sequence without any data."

            data_chunks = {}
            num_chunks = None

            for data_key in chunking_data_keys:
                chunk_size = self._chunk_size[data_key]
                chunk_step = self._chunk_step[data_key]
                min_chunk_size = self._min_chunk_size[data_key]

                data = data_dict[data_key]
                chunks = [
                    data[start_index : start_index + chunk_size]
                    for start_index in range(0, len(data), chunk_step)
                    if len(data[start_index : start_index + chunk_size]) >= min_chunk_size
                ]

                if num_chunks is None:
                    num_chunks = len(chunks)
                else:
                    assert num_chunks == len(
                        chunks
                    ), "Chunking resulted in different number of chunks for different data keys."

                data_chunks[data_key] = chunks

            if num_chunks == 0:
                continue
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
            sequence_lengths = NumbersDict(
                {data_key: data.shape[0] for data_key, data in data_dict.items() if data.shape}
            )

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


class LenFilterDataPipe(torch.utils.data.IterDataPipe):
    """
    Removes sequences which are either too long or too short from a dataset
    Returns dataset yielding list of data lengths within the defined range
    """

    def __init__(
        self,
        dataset: torch.utils.data.IterableDataset,
        min_seq_length: Union[int, NumbersDict] = None,
        max_seq_length: Union[int, NumbersDict] = None,
    ):
        """
        :param dataset: dataset to apply the filter to
        :param min_seq_length: minimum sequence length either in general or per data_key via dict
        :param max_seq_length: maximum sequence length either in general or per data_key via dict
        """
        super().__init__()
        self._dataset = dataset
        self._min_seq_length = NumbersDict(0 if min_seq_length is None else min_seq_length)
        self._max_seq_length = NumbersDict(sys.maxsize if max_seq_length is None else max_seq_length)

    def __iter__(self):
        """
        :return: generator providing filtered data where each sequence is a dict
          data_key -> data_array.
        :rtype: Iterable[dict[str, numpy.ndarray]]
        """
        for data_dict in self._dataset:
            # TODO: This assumes all data has time as first dimension. Currently we can't know better..
            sequence_lengths = NumbersDict(
                {data_key: data.shape[0] for data_key, data in data_dict.items() if data.shape}
            )
            if sequence_lengths.any_compare(self._min_seq_length, lambda a, b: a < b):
                continue
            if sequence_lengths.any_compare(self._max_seq_length, lambda a, b: a > b):
                continue
            yield data_dict

    def __getitem__(self, index):
        raise Exception(f"{self.__class__.__name__}.__getitem__ not supported")


def create_data_loader_from_batches(
    batches_dataset: torch.utils.data.Dataset, loader_opts: Optional[Dict[str, Any]] = None
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader based on dataset over batches, e.g. via :class:`BatchingIterDataPipe`.
    """
    if loader_opts is None:
        loader_opts: Dict[str, Any] = {}
    else:
        loader_opts = loader_opts.copy()

    # Make sure to use workers unless specified otherwise to ensure reasonable GPU
    # utilization and to work around some issues wrt. overflowing ulimits when
    # workers are non-persistent.
    #
    # See for context:
    # - https://github.com/rwth-i6/returnn/issues/1560
    # - https://github.com/pytorch/pytorch/issues/129868
    loader_opts.setdefault("num_workers", 1)

    if loader_opts.get("num_workers"):
        loader_opts.setdefault("persistent_workers", True)
        loader_opts["worker_init_fn"] = _DataLoaderWorkerInitFunc(
            other_worker_init_fn=loader_opts.get("worker_init_fn")
        )
        # We don't want to use the default fork start method
        # (https://github.com/rwth-i6/returnn/issues/1494 and potentially lots of other issues with fork).
        # We cannot use the standard spawn start method, as DataLoader would start daemonic processes,
        # which is incompatible with some of our datasets
        # (https://github.com/rwth-i6/returnn/issues/1495).
        loader_opts.setdefault("multiprocessing_context", "spawn_non_daemonic")
        if loader_opts.get("multiprocessing_context") == "spawn_non_daemonic":
            from returnn.config import SubProcCopyGlobalConfigPreInitFunc
            from returnn.util.multi_proc_non_daemonic_spawn import NonDaemonicSpawnContext

            loader_opts["multiprocessing_context"] = NonDaemonicSpawnContext(
                # It's important that this init function is called even before the unpickling of the dataset,
                # as that unpickling might depend on the right context, e.g. having a global RETURNN config.
                # See _DataLoaderWorkerPreInitFunc below, https://github.com/rwth-i6/returnn/issues/1495.
                process_pre_init_func=SubProcCopyGlobalConfigPreInitFunc()
            )

    return torch.utils.data.DataLoader(
        batches_dataset,
        collate_fn=collate_batch,
        # Batching is already done by BatchingIterDataPipe.
        batch_size=None,
        # Explicitly not use the following opts, which are not supported and/or do not make sense
        # for an iterable-style dataset.
        shuffle=None,
        sampler=None,
        batch_sampler=None,
        # User-defined
        **loader_opts,
    )


class _DataLoaderWorkerInitFunc:
    def __init__(self, *, other_worker_init_fn: Optional[Callable] = None):
        self.other_worker_init_fn = other_worker_init_fn

    def __call__(self, worker_id: int):
        if sys.platform == "linux":
            with open("/proc/self/comm", "w") as f:
                f.write(f"TDL worker {worker_id}")

        if self.other_worker_init_fn:
            self.other_worker_init_fn(worker_id)
