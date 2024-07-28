"""
Provides :class:`PostprocessingDataset`.
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from returnn.datasets.basic import DatasetSeq
from returnn.tensor import Tensor, TensorDict
from returnn.tensor.dim import Dim
from returnn.util.basic import NumbersDict
from .basic import Dataset, init_dataset
from .cached2 import CachedDataset2
from .util.strings import str_to_numpy_array

__all__ = ["PostprocessingDataset"]

one_element_dim = Dim(1)


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
            "dataset": {
                "class": "HDFDataset",
                "files": ["/path/to/data.hdf"],
            },
            # at least one of them:
            "map_seq": map_seq,  # (data: TensorDict) -> TensorDict
            "map_seq_stream": map_seqs,  # (iter: Iterator[TensorDict]) -> Iterator[TensorDict]
        }
    """

    def __init__(
        self,
        dataset: Dict[str, Any],
        map_seq: Optional[Callable[[TensorDict], TensorDict]] = None,
        map_seq_stream: Optional[Callable[[Iterator[TensorDict]], Iterator[TensorDict]]] = None,
        map_outputs: Optional[Dict[str, Any]] = None,
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
        :param output: Type and axis specification of the outputs of the mapping functions,
            like extern_data and model_outputs.
            To simplify the common case when no shapes change, this value can be left unspecified. The dataset then
            assumes the same data layout as returned by the wrapped dataset.
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
        self._map_outputs = map_outputs

        self._dataset: Optional[Dataset] = None
        self._data_keys: Optional[List[str]] = None
        self._data_iter: Optional[Iterator[Tuple[int, TensorDict]]] = None
        self._data_iter_seq_idx = -1
        self._default_input: Optional[str] = None
        self._dim_cache: Dict[str, Tuple[List[Dim], Optional[Dim]]] = {}

        if _meta_info_cache:
            self._default_input = _meta_info_cache["_default_input"]
            self._dim_cache = _meta_info_cache["_dim_cache"]
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
            "_default_input": self._default_input,
            "_dim_cache": self._dim_cache,
            "_estimated_num_seqs": self._estimated_num_seqs,
            "labels": self.labels,
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
        }

    def _lazy_init_num_outputs(self):
        if self.num_outputs is not None:
            return

        dataset = init_dataset(self._dataset_def, parent_dataset=self)
        self.labels = dataset.labels
        self._default_input = (
            "data" if not self._map_outputs or "data" in self._map_outputs else next(self._map_outputs.keys())
        )
        self._estimated_num_seqs = dataset.estimated_num_seqs

        if self._map_outputs is not None:
            tdict = TensorDict(self._map_outputs)
            self.num_outputs = {
                k: (t.sparse_dim.size if t.sparse_dim else t.shape[-1] or 1, t.ndim) for k, t in tdict.data.items()
            }
            self.num_inputs = self.num_outputs[self._default_input][0]
        else:
            self.num_inputs = dataset.num_inputs
            self.num_outputs = dataset.num_outputs

    def init_seq_order(self, epoch: Optional[int] = None, seq_list=None, seq_order=None):
        """
        :param epoch:
        :param seq_list:
        :param seq_order:
        :return: whether the order changed (True is always safe to return)
        """
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

    def _collect_single_seq(self, seq_idx: int) -> Optional[DatasetSeq]:
        assert seq_idx > self._data_iter_seq_idx, "collecting seqs must be done strictly monotonically"
        self._data_iter_seq_idx = seq_idx

        while True:
            try:
                loaded_seq_idx, tensor_dict = next(self._data_iter)
            except StopIteration:
                return None
            if loaded_seq_idx != seq_idx:
                continue
            seq = DatasetSeq(features=tensor_dict.as_raw_tensor_dict(exclude_sequence_lengths=True), seq_idx=seq_idx)
            return seq

    def _build_dataset_iter(self, dataset: Dataset) -> Iterator[TensorDict]:
        """
        :return: an iterator applying both the segment level and across-segment transformations on the given dataset
        """
        return self._map_seq_stream(map(self._map_seq, self._iterate_dataset(dataset)))

    def _data_dict_to_tensor_dict(
        self, dataset: Dataset, data_dict: Dict[str, ndarray], seq_len: NumbersDict
    ) -> TensorDict:
        """
        :return: the given data dict converted to a TensorDict class
        """

        def _make_tensor(name: str, data: ndarray) -> Tensor:
            dims = None
            sparse_dim = None
            dtype = data.dtype

            if data.dtype.name.startswith("str"):
                # TODO: is this correct?
                dims = []
                dtype = "string"
            elif dims is None:
                dims, sparse_dim = self._dim_cache.get(name, (None, None))
                if dims is None:
                    feature_dims = [
                        Dim(dimension=v, name=f"{name}_dim{i + 1}") for i, v in enumerate(dataset.get_data_shape(name))
                    ]
                    dims = [Dim(dimension=None, name=f"{name}_num_frames"), *feature_dims]
                    if dataset.is_data_sparse(name):
                        sparse_dim = Dim(
                            dimension=dataset.get_data_dim(name) if dataset.is_data_sparse(name) else None,
                            name=f"{name}_sparse",
                        )
                    self._dim_cache[name] = (dims, sparse_dim)

            try:
                return Tensor(name, dims=dims, dtype=dtype, sparse_dim=sparse_dim, raw_tensor=data)
            except Exception as exc:
                raise Exception(
                    "Could not convert from mapping function output to `TensorDict`, "
                    f"do the data shapes {data.shape} match up with the ones declared "
                    f"in `map_output` (inferred to: {dims}, sparse={sparse_dim})?"
                ) from exc

        return TensorDict({k: _make_tensor(k, v) for k, v in data_dict.items()})

    def _iterate_dataset(self, dataset: Dataset) -> Iterator[TensorDict]:
        """
        :return: generator providing data samples in the form of a TensorDict
        """
        data_keys = dataset.get_data_keys()

        seq_index = 0
        while dataset.is_less_than_num_seqs(seq_index):
            dataset.load_seqs(seq_index, seq_index + 1)
            data = {data_key: dataset.get_data(seq_index, data_key) for data_key in data_keys}
            data_len = dataset.get_seq_length(seq_index)
            data["seq_tag"] = str_to_numpy_array(dataset.get_tag(seq_index))
            yield self._data_dict_to_tensor_dict(dataset, data, data_len)
            seq_index += 1

    @staticmethod
    def _identity(x):
        return x
