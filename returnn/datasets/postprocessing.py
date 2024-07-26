"""
Provides :class:`PostprocessingDataset`.
"""

from __future__ import annotations

from numpy import ndarray
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from returnn.datasets.basic import DatasetSeq
from returnn.tensor import Tensor, TensorDict
from .basic import Dataset, init_dataset
from .cached2 import CachedDataset2
from .util.strings import str_to_numpy_array

__all__ = ["PostprocessingDataset"]


class PostprocessingDataset(CachedDataset2):
    """
    A dataset that allows for generic post-processing of data from another dataset
    using a function on the segment level and on the level of multiply segments via
    an iterator.

    This allows integrating various data augmentation techniques like e.g. Mixup, or
    speed perturbation into the data loading pipeline.

    The integration into the data loading pipeline makes it easy to distribute the
    data processing work across multiple CPU cores using `MultiProcDataset` and in
    turn frees the GPU from data preprocessing tasks.

    Both functions can be specified at the same time, and their results will be
    combined by first applying the segment-level function and then the
    multiple-segments-level function.

    Example usage:
        train = {
            "class": "PostprocessingDataset",
            "dataset": { "class": ".." },
            # at least one of them:
            "map_seq": segment_mapping_function,
            "map_seq_stream": multiple_segment_mapping_function,
        }
    """

    def __init__(
        self,
        dataset: Dict[str, Any],
        map_seq: Optional[Callable[[TensorDict], TensorDict]] = None,
        map_seq_stream: Optional[Callable[[Iterator[TensorDict]], Iterator[TensorDict]]] = None,
        _meta_info_cache: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        :param dataset: inner dataset to be post-processed
        :param map_seq: post processor function operating on the single-segment level.
        :param map_seq_stream: post processor function operating on the multiple segment level via an iterator.
            Allows merging multiple segments into one, or generating multiple output segments from one input segment.
            When both functions are specified, the segment-level function is applied first to every segment, and the
            results are passed to the iterator-level function.
        :param _meta_info_cache: internal usage
        :param kwargs: see :class:`CachedDataset2`, :class:`Dataset`
        """
        super().__init__(**kwargs)

        assert (
            self.seq_ordering == "default"
        ), f"specify seq_ordering in wrapped dataset, not in {self.__class__.__name__}"
        assert (
            map_seq is not None or map_seq_stream is not None
        ), "need to either define map_seq or map_seq_stream functions"

        self._dataset_def = dataset
        self._map_seq = map_seq or self._identity
        self._map_seq_stream = map_seq_stream or self._identity

        self._dataset: Optional[Dataset] = None
        self._data_keys: Optional[List[str]] = None
        self._data_iter: Optional[Iterator[Tuple[int, TensorDict]]] = None
        self._data_iter_seq_idx = -1

        if _meta_info_cache:
            self._estimated_num_seqs = _meta_info_cache["_estimated_num_seqs"]
            self.labels = _meta_info_cache["labels"]
            self.num_inputs = _meta_info_cache["num_inputs"]
            self.num_outputs = _meta_info_cache["num_outputs"]

    def initialize(self):
        """init"""
        self._lazy_init_num_outputs()
        super().initialize()

    @property
    def _meta_info_cache(self) -> Optional[Dict[str, Any]]:
        if self.num_outputs is None:
            return None
        return {
            "_estimated_num_seqs": self._estimated_num_seqs,
            "labels": self.labels,
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
        }

    def _lazy_init_num_outputs(self):
        if self.num_outputs is not None:
            return

        dataset = init_dataset(self._dataset_def, parent_dataset=self)
        dataset.init_seq_order(epoch=1)
        self._estimated_num_seqs = dataset.estimated_num_seqs
        tdict = next(self._build_dataset_iter(dataset))

        # TODO: is this correct?
        self.num_inputs = tdict.data["data"].feature_dim.size
        self.num_outputs = {k: (t.feature_dim_or_sparse_dim.size, t.ndim) for k, t in tdict.data.items()}

    def init_seq_order(self, epoch: Optional[int] = None, seq_list=None, seq_order=None):
        super().init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)

        if epoch is None:
            self._num_seqs = 0
            return True

        if self._dataset is None:
            self._dataset = init_dataset(self._dataset_def, parent_dataset=self)
        self._dataset.init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)
        self._data_iter = enumerate(self._build_dataset_iter(self._dataset))
        self._data_iter_seq_idx = -1
        self._lazy_init_num_outputs()
        return True

    def _collect_single_seq(self, seq_idx: int) -> DatasetSeq | None:
        assert seq_idx > self._data_iter_seq_idx, "collecting seqs must be done strictly monotonically"
        self._data_iter_seq_idx = seq_idx

        while True:
            try:
                loaded_seq_idx, tensor_dict = next(self._data_iter)
            except StopIteration:
                return None
            if loaded_seq_idx != seq_idx:
                continue
            seq = DatasetSeq(features=tensor_dict.as_raw_tensor_dict(), seq_idx=seq_idx)
            return seq

    def _build_dataset_iter(self, dataset: Dataset) -> Iterator[TensorDict]:
        """
        :return: an iterator applying both the segment level and across-segment transformations on the given dataset
        """
        return self._map_seq_stream(map(self._map_seq, self._iterate_dataset(dataset)))

    @staticmethod
    def _data_dict_to_tensor_dict(data_dict: Dict[str, ndarray]) -> TensorDict:
        """
        :return: the given data dict converted to a TensorDict class
        """
        tdict = TensorDict({k: Tensor(name=k, dtype=v.dtype, raw_tensor=v) for k, v in data_dict.items()})
        raise NotImplementedError
        return tdict

    @staticmethod
    def _iterate_dataset(dataset: Dataset) -> Iterator[TensorDict]:
        """
        :return: generator providing data samples in the form of a TensorDict
        """
        data_keys = dataset.get_data_keys()

        seq_index = 0
        while dataset.is_less_than_num_seqs(seq_index):
            dataset.load_seqs(seq_index, seq_index + 1)
            data = {data_key: dataset.get_data(seq_index, data_key) for data_key in data_keys}
            data["seq_tag"] = str_to_numpy_array(dataset.get_tag(seq_index))
            yield PostprocessingDataset._data_dict_to_tensor_dict(data)
            seq_index += 1

    @staticmethod
    def _identity(x):
        return x
