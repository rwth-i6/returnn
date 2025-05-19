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
import bisect
import itertools
from typing import Optional, Any, Sequence, Tuple, Union, List, Dict, Callable
import sys
from copy import deepcopy

import numpy
import torch
import torch.utils.data

from returnn.config import Config
from returnn.log import log
from returnn.util.basic import NumbersDict, get_fwd_compat_kwargs


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
        if key in ("num_seqs", "epoch"):
            res[key] = batch[0][key]  # it should always be the same
            continue
        elif key == "complete_frac":
            res[key] = max(sample[key] for sample in batch)
            continue
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
                chunking_data_key_black_list = ["seq_tag", "seq_idx", "num_seqs", "epoch", "complete_frac"]
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
                    assert num_chunks == len(chunks), (
                        "Chunking resulted in different number of chunks for different data keys."
                    )

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
        :param int|dict[str,int]|None|function batch_size: Maximum number of time steps (e.g. audio frames / words)
            in one batch (padding included).
            If given as a dict data_key -> value, sets different individual limits per data key.
            If None, no limit.
            Can also be a callable with kwargs epoch, seq_idx, epoch_continuous, **_other_kwargs,
            returning the batch size.
        :param int|None|function max_seqs: maximum number of sequences in a batch,
            None means unlimited (also -1 to match TF backend).
            Can also be a callable with kwargs epoch, seq_idx, epoch_continuous, **_other_kwargs,
            returning the max seqs.
        """
        super().__init__()
        self._dataset = dataset
        self._max_batch_size = self._parse_batch_size(batch_size)
        self._max_seqs = self._parse_max_seqs(max_seqs)

        if not callable(self._max_batch_size):
            assert isinstance(self._max_batch_size, NumbersDict) and self._max_batch_size.min_value() > 0
        if not callable(self._max_seqs):
            assert isinstance(self._max_seqs, int) and self._max_seqs > 0

    @staticmethod
    def _parse_batch_size(
        batch_size: Union[int, Dict[str, int], NumbersDict, None, Callable],
        *,
        data_dict: Optional[Dict[str, Any]] = None,
    ) -> Union[NumbersDict, Callable]:
        """
        :param batch_size: see __init__()
        :return: batch_size
        """
        if callable(batch_size):
            if data_dict:
                batch_size = batch_size(**BatchingIterDataPipe._get_user_func_kwargs_from_data_dict(data_dict))
            else:
                return batch_size
        return NumbersDict(sys.maxsize if batch_size is None else batch_size)

    @staticmethod
    def _parse_max_seqs(
        max_seqs: Union[int, None, Callable], *, data_dict: Optional[Dict[str, Any]] = None
    ) -> Union[int, Callable]:
        """
        :param max_seqs: see __init__()
        :return: max_seqs
        """
        if callable(max_seqs):
            if data_dict:
                max_seqs = max_seqs(**BatchingIterDataPipe._get_user_func_kwargs_from_data_dict(data_dict))
            else:
                return max_seqs
        return sys.maxsize if (max_seqs is None or max_seqs == -1) else max_seqs

    @staticmethod
    def _get_user_func_kwargs_from_data_dict(data_dict: Dict[str, Any]) -> Dict[str, Any]:
        epoch = int(data_dict["epoch"])
        seq_idx = int(data_dict["seq_idx"])
        num_seqs = int(data_dict["num_seqs"])  # >=1 if known, otherwise -1
        complete_frac = float(data_dict["complete_frac"])  # >= 0 if known, otherwise -1
        epoch_continuous = (
            epoch - 1 + complete_frac
            if complete_frac >= 0.0
            else (epoch - 1 + (seq_idx + 1) / num_seqs)
            if num_seqs > 0
            else None
        )
        return {"epoch": epoch, "epoch_continuous": epoch_continuous, "seq_idx": seq_idx, **get_fwd_compat_kwargs()}

    def __iter__(self):
        """
        :return: generator providing batches in the form of lists of sequences, where each sequence is a dict
          data_key -> data_array.
        :rtype: Iterable[list[dict[str, numpy.ndarray]]]
        """
        current_batch = []
        current_max_sequence_lengths = NumbersDict(0)  # data_key -> length of longest sequence in current batch

        for data_dict in self._dataset:
            max_seqs = self._parse_max_seqs(self._max_seqs, data_dict=data_dict)
            max_batch_size = self._parse_batch_size(self._max_batch_size, data_dict=data_dict)
            assert isinstance(max_seqs, int) and max_seqs > 0
            assert isinstance(max_batch_size, NumbersDict) and max_batch_size.min_value() > 0

            if len(current_batch) >= max_seqs:
                yield current_batch
                current_batch = []
                current_max_sequence_lengths = NumbersDict(0)

            # TODO: This assumes all data has time as first dimension. Currently we can't know better..
            sequence_lengths = NumbersDict(
                {data_key: data.shape[0] for data_key, data in data_dict.items() if data.shape}
            )

            max_sequence_lengths_if_included = NumbersDict.max([current_max_sequence_lengths, sequence_lengths])
            batch_size_if_included = max_sequence_lengths_if_included * (len(current_batch) + 1)  # including padding

            if current_batch and batch_size_if_included.any_compare(max_batch_size, (lambda a, b: a > b)):
                yield current_batch
                current_batch = [data_dict]
                current_max_sequence_lengths = sequence_lengths
            else:
                current_batch.append(data_dict)
                current_max_sequence_lengths = max_sequence_lengths_if_included

        if current_batch:
            yield current_batch


class BucketOrderingIterDataPipe(torch.utils.data.IterDataPipe):
    """
    Converts a dataset yielding sequences (dict data_key -> array per sequence) into
    a dataset yielding lists of these sequences, i.e. batches. Can be used instead of
    the default ordered batching via the `custom_batching` config key.

    Sequences are grouped into a distinct set of length buckets. Each bucket has a
    limit of how many sequences it can buffer before being sent off to training.

    Note, that batches are not yet merged into a single (padded) data array here,
    this happens in 'collate_batch()'.
    """

    def __init__(
        self,
        dataset: torch.utils.data.IterableDataset,
        *,
        buckets: Sequence[Tuple[int, int]],
        length_key: str,
        random_bucket_prob: float = 0.0,
        seed: Optional[int] = None,
    ):
        """
        :param dataset: dataset to apply bucket batching to
        :param buckets: Bucket configuration as tuples of seq length and max number of seqs in that bucket.
            Segments longer than the largest size limit configured in the buckets are dropped. To avoid dropping
            any segments make sure your largest bucket allows segments larger than your longest training segment.
        :param length_key: data key to take as length measure
        :param random_bucket_prob: Probability of putting a segment not into the best-fitting bucket, but into
            a randomly chosen still-fitting bucket.
            This increases seq length variation within the buckets at the cost of slighly more padding.
        :param seed: random seed
        """
        self._dataset = dataset
        self._length_key = length_key
        assert random_bucket_prob >= 0.0
        self._random_bucket_prob = random_bucket_prob
        self._rng = numpy.random.RandomState()
        self._seed = seed % (2**32) if seed is not None else None

        assert buckets, "empty bucket batching configuration"
        if not all(size > 0 and max_seqs > 0 for size, max_seqs in buckets):
            raise ValueError(f"bucket sizes and max seqs in bucket must be positive")
        self._max_seq_lens, self._max_bucket_sizes = zip(*sorted(buckets))
        assert len(set(self._max_seq_lens)) == len(self._max_seq_lens), "seq len boundaries must all be unique"

    def __iter__(self):
        """:return: generator applying bucket ordering on the data"""
        buckets: List[List[Dict[str, numpy.ndarray]]] = [[] for _ in range(len(self._max_seq_lens))]
        buckets_full_counter = [0 for _ in range(len(buckets))]

        for data_dict in self._dataset:
            data_dict: Dict[str, numpy.ndarray]
            data_size = data_dict[self._length_key].shape[0]
            bucket_idx = bisect.bisect_left(self._max_seq_lens, data_size)
            if bucket_idx >= len(self._max_seq_lens):
                # seg is too long, drop it
                continue
            if (
                self._random_bucket_prob > 0.0
                and bucket_idx < len(self._max_seq_lens) - 1
                and self._rng.rand() < self._random_bucket_prob
            ):
                bucket_idx = self._rng.randint(bucket_idx, len(self._max_bucket_sizes))
            buckets[bucket_idx].append(data_dict)
            if len(buckets[bucket_idx]) >= self._max_bucket_sizes[bucket_idx]:
                yield buckets[bucket_idx]
                buckets[bucket_idx] = []
                buckets_full_counter[bucket_idx] += 1

        non_empty_buckets = [b for b in buckets if b]
        yield from non_empty_buckets

        # we do not count the buckets w/ leftover data as they were not completely filled
        description_str = ", ".join(f"{limit}: {cnt}" for limit, cnt in zip(self._max_seq_lens, buckets_full_counter))
        print(f"Batching buckets full: {description_str}", file=log.v4)

    def __getitem__(self, index):
        raise Exception(f"{self.__class__.__name__}.__getitem__ is not supported")

    def set_seed(self, seed: int) -> BucketOrderingIterDataPipe:
        """
        Sets the seed for the next invocation of ``__iter__``, for compatibility with
        ``torch.utils.data.graph_settings.apply_random_seed``.
        """
        self._seed = seed % (2**32)  # seed must be within [0, 2**32) for seeding RandomState
        return self

    def reset(self):
        """resets the internal state of the data pipe"""
        if self._seed is None:
            self._seed = int(2**31 + torch.empty((), dtype=torch.int32).random_().item())
        self._rng.seed(self._seed)
        self._seed = None


def get_batching_iterable_dataset_from_config(
    *, dataset: torch.utils.data.IterableDataset, config: Config, train: bool
) -> torch.utils.data.IterableDataset:
    """
    Batches the segments in the given dataset given the config settings.

    :param dataset: dataset whose segments to group into batches
    :param config: RETURNN config
    :param train: whether we are in training or in inference mode
    """
    torch_batching = config.typed_value("torch_batching", None)
    if torch_batching is None:
        batch_size = config.typed_value("batch_size", -1)
        batch_size = config.typed_value(f"batch_size_{'train' if train else 'dev'}", batch_size)
        assert batch_size != -1, f"batch_size or batch_size_{'train' if train else 'dev'} not defined in config"
        max_seqs = config.typed_value("max_seqs", -1)
        batches_dataset = BatchingIterDataPipe(dataset, batch_size=batch_size, max_seqs=max_seqs)
        return batches_dataset

    if isinstance(torch_batching, dict):
        assert "class" in torch_batching
        batching_args = torch_batching.copy()
        type_name = batching_args.pop("class")
        assert isinstance(type_name, str)
        cls = globals()[type_name]
    elif callable(torch_batching):
        # callables need to be forward compatible
        batching_args = get_fwd_compat_kwargs()
        batching_args["train"] = train
        cls = torch_batching
    else:
        raise ValueError(
            f"custom_batching must either be a dict containing a `class` key naming a type, a type or a callable."
        )
    batches_dataset = cls(dataset, **batching_args)
    assert isinstance(batches_dataset, torch.utils.data.IterableDataset)
    return batches_dataset


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


class ShufflingDataPipe(torch.utils.data.IterDataPipe):
    """
    Data pipe that is similar to ``torch.utils.data.datapipes.iter.Shuffler``,
    but it will keep certain data keys of the batches in order while shuffling the rest.

    Used for e.g. ``complete_frac`` and ``seq_idx``.
    """

    def __init__(
        self,
        dataset: torch.utils.data.IterableDataset,
        *,
        buffer_size: int,
        monotonic_data_keys: Sequence[str],
        seed: Optional[int] = None,
    ):
        """
        :param dataset: batches dataset to shuffle
        :param buffer_size: buffer size for shuffling
        :param monotonic_data_keys: data keys that will be excluded from shuffling/keep their order
        :param seed: random seed
        """
        super().__init__()

        self._dataset = dataset
        self._buffer: List[List[Dict[str, Any]]] = []
        self._next_buffer: List[List[Dict[str, Any]]] = []
        assert buffer_size > 0
        self._buffer_size = buffer_size
        self._monotonic_data_keys = monotonic_data_keys
        self._rng = numpy.random.RandomState()
        self._seed = seed % (2**32) if seed is not None else None

    def __iter__(self):
        # The implementation is very similar to the PostprocessingDataset's combinator LaplaceOrdering.

        data_iter = iter(self._dataset)

        self._buffer.extend(itertools.islice(data_iter, self._buffer_size))
        has_ended = False

        while True:
            # Make sure to not reorder the monotonic values from self._monotonic_data_keys.
            # These can contain things like complete_frac, which should be kept in order.
            ordered_data = {
                key: [data_dict[key] for batch in self._buffer for data_dict in batch]
                for key in self._monotonic_data_keys
            }
            self._rng.shuffle(self._buffer)
            for key in self._monotonic_data_keys:
                data_dicts = [data_dict for batch in self._buffer for data_dict in batch]
                assert len(data_dicts) == len(ordered_data[key])
                for ordered_value, data_dict in zip(ordered_data[key], data_dicts):
                    data_dict[key] = ordered_value

            for item in self._buffer:
                yield item

                try:
                    if not has_ended:
                        self._next_buffer.append(next(data_iter))
                except StopIteration:
                    has_ended = True

            if len(self._buffer) < self._buffer_size:
                assert has_ended and not self._next_buffer
                break

            self._buffer.clear()
            self._buffer, self._next_buffer = self._next_buffer, self._buffer

    def set_seed(self, seed: int) -> ShufflingDataPipe:
        """
        Sets the seed for the next invocation of ``__iter__``, for compatibility with
        ``torch.utils.data.graph_settings.apply_random_seed``.
        """
        self._seed = seed % (2**32)  # seed must be within [0, 2**32) for seeding RandomState
        return self

    def reset(self):
        """resets the internal state of the data pipe"""
        self._buffer.clear()
        self._next_buffer.clear()
        if self._seed is None:
            self._seed = int(2**31 + torch.empty((), dtype=torch.int32).random_().item())
        self._rng.seed(self._seed)
        self._seed = None

    def __getstate__(self):
        state = (
            self._dataset,
            self._buffer,
            self._next_buffer,
            self._buffer_size,
            self._monotonic_data_keys,
            self._rng.get_state(),
            self._seed,
        )
        if torch.utils.data.IterDataPipe.getstate_hook is not None:
            return torch.utils.data.IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        (
            self._dataset,
            self._buffer,
            self._next_buffer,
            self._buffer_size,
            self._monotonic_data_keys,
            rng_state,
            self._seed,
        ) = state
        self._rng = numpy.random.RandomState()
        self._rng.set_state(rng_state)

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
