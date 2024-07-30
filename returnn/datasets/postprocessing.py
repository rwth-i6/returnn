"""
Provides :class:`PostprocessingDataset`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from returnn.datasets.basic import DatasetSeq
from returnn.datasets.util.vocabulary import Vocabulary
from returnn.tensor import Tensor, TensorDict
from returnn.tensor.dim import Dim
from .basic import init_dataset
from .cached2 import CachedDataset2

__all__ = ["PostprocessingDataset"]


class PostprocessingDataset(CachedDataset2):
    """
    A dataset that allows for generic post-processing of data from another dataset
    using a function on the segment level and on the level of multiple segments via
    an iterator.

    This allows integrating various data augmentation techniques like e.g. Mixup,
    SpecAugment or speed perturbation into the data loading pipeline.

    The integration into the data loading pipeline makes it easy to distribute the
    data processing work across multiple CPU cores using `MultiProcDataset` and in
    turn frees the GPU from data preprocessing tasks.

    Example usage::

        from returnn.tensor.dim import Dim, DimTypes

        time_dim = Dim(None, kind=DimTypes.Spatial)
        new_data_dim = Dim(128)

        train = {
            "class": "PostprocessingDataset",
            "dataset": {
                "class": "HDFDataset",
                "files": ["/path/to/data.hdf"],
            },
            # one of them, but not both:
            "map_seq": map_seq,  # (data: TensorDict) -> TensorDict
            "map_seq_stream": map_seqs,  # (iter: Iterator[TensorDict]) -> Iterator[TensorDict]
            # only required when data shapes change wrt. the wrapped dataset:
            "map_outputs": {
                "data": {"dims": [time_dim, new_data_dim]},
            },
        }
    """

    def __init__(
        self,
        dataset: Dict[str, Any],
        map_seq: Optional[Union[Callable[[TensorDict], TensorDict]]] = None,
        map_seq_stream: Optional[Callable[[Iterator[TensorDict]], Iterator[TensorDict]]] = None,
        map_outputs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        :param dataset: inner dataset to be post-processed
        :param map_seq: post processor function operating on the single-segment level.
            To avoid confusion on the order of how the processing functions are applied to the data, only one of
            `map_seq` and `map_seq_stream` can be specified at a time.
        :param map_seq_stream: post processor function operating on the multiple segment level via an iterator.
            Allows merging multiple segments into one, or generating multiple output segments from one input segment.
            To avoid confusion on the order of how the processing functions are applied to the data, only one of
            `map_seq` and `map_seq_stream` can be specified at a time.
        :param map_outputs: Type and axis specification of the outputs of the mapping functions,
            like extern_data and model_outputs.
            To simplify the common case when no shapes change, this value can be left unspecified. The dataset then
            assumes the same data layout as returned by the wrapped dataset.
            Example: `map_outputs={"data": {"dim": 42}}`
        :param kwargs: see :class:`CachedDataset2`, :class:`Dataset`
        """
        super().__init__(**kwargs)

        if self.seq_ordering != "default":
            raise ValueError(f"{self}: specify seq_ordering in wrapped dataset, not in {self.__class__.__name__}")
        if map_seq is None and map_seq_stream is None:
            raise ValueError(f"{self}: need to either set map_seq or map_seq_stream")
        if map_seq and map_seq_stream:
            raise ValueError(f"{self}: cannot set both map_seq and map_seq_stream")

        self._dataset_def = dataset
        self._map_seq = map_seq
        self._map_seq_stream = map_seq_stream
        self._map_outputs = map_outputs

        self._dataset = init_dataset(self._dataset_def, parent_dataset=self)
        if self._map_seq_stream is None:
            # if the stream mapper is set, the num_seqs may change and the estimation is less accurate
            self._estimated_num_seqs = self._dataset.estimated_num_seqs
        self._data_iter: Optional[Iterator[Tuple[int, TensorDict]]] = None

        self._in_tensor_dict_template = TensorDict(
            {name: self._make_tensor_template_from_input(name) for name in self._dataset.get_data_keys()}
        )
        self._out_tensor_dict_template = (
            TensorDict(self._map_outputs) if self._map_outputs is not None else self._in_tensor_dict_template
        )
        self.num_outputs = {
            k: (t.sparse_dim.size if t.sparse_dim else t.shape[-1] if len(t.shape) > 0 else 1, t.ndim)
            for k, t in self._out_tensor_dict_template.data.items()
        }
        self._default_input = "data" if "data" in self.num_outputs else next(iter(self.num_outputs.keys()))
        self.num_inputs = self.num_outputs[self._default_input][0]

        self.labels = {}
        for k, t in self._out_tensor_dict_template.data.items():
            if t.vocab:
                self.labels[k] = t.vocab.labels
            elif t.sparse_dim:  # sparse_dim but not vocab
                self.labels[k] = list(map(str, range(t.sparse_dim.dimension)))  # dummy labels

    def init_seq_order(
        self, epoch: Optional[int] = None, seq_list: Optional[List[str]] = None, seq_order: Optional[List[int]] = None
    ):
        """
        :param epoch:
        :param seq_list:
        :param seq_order:
        :return: whether the order changed (True is always safe to return)
        """
        super().init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)

        if epoch is None and seq_list is None and seq_order is None:
            self._num_seqs = 0
            return True

        assert self._dataset is not None
        self._dataset.init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)
        self._data_iter = enumerate(self._build_mapping_iter())
        return True

    def _collect_single_seq(self, seq_idx: int) -> Optional[DatasetSeq]:
        while True:
            try:
                loaded_seq_idx, tensor_dict = next(self._data_iter)
            except StopIteration:
                return None
            assert loaded_seq_idx <= seq_idx, "_collect_single_seq must be done monotonically"
            if loaded_seq_idx != seq_idx:
                continue
            seq = DatasetSeq(features={k: t.raw_tensor for k, t in tensor_dict.data.items()}, seq_idx=seq_idx)
            return seq

    def _build_mapping_iter(self) -> Iterator[TensorDict]:
        """
        :return: an iterator applying both the segment level and across-segment transformations on the given dataset
        """

        def _validate_tensor_dict_iter(inner: Iterator[TensorDict]) -> Iterator[TensorDict]:
            for t_dict in inner:
                for data_key, out_t in self._out_tensor_dict_template.data.items():
                    in_t = t_dict.data[data_key]
                    assert (
                        in_t.ndim == out_t.batch_ndim
                        and in_t.dtype == out_t.dtype
                        and all(d.dimension in (d_, None) for (d, d_) in zip(in_t.dims, out_t.shape))
                    )
                yield t_dict

        data_iter = self._iterate_dataset()
        if self._map_seq_stream is not None:
            data_iter = self._map_seq_stream(data_iter)
            assert isinstance(
                data_iter, Iterator
            ), f"map_seq_stream must produce an {Iterator.__name__}, but produced {type(data_iter).__name__}"
        return _validate_tensor_dict_iter(data_iter)

    def _iterate_dataset(self) -> Iterator[TensorDict]:
        """
        :return: generator providing data samples in the form of a TensorDict
        """
        data_keys = self._dataset.get_data_keys()

        seq_index = 0
        while self._dataset.is_less_than_num_seqs(seq_index):
            self._dataset.load_seqs(seq_index, seq_index + 1)
            tensor_dict = self._in_tensor_dict_template.copy_template()
            for data_key in data_keys:
                tensor_dict.data[data_key].raw_tensor = self._dataset.get_data(seq_index, data_key)
            if self._map_seq is not None:
                tensor_dict = self._map_seq(tensor_dict)
                assert isinstance(
                    tensor_dict, TensorDict
                ), f"map_seq must produce a {TensorDict.__name__}, but produced {type(tensor_dict).__name__}"
            yield tensor_dict
            seq_index += 1

    def _make_tensor_template_from_input(self, data_key: str) -> Tensor:
        dtype = self._dataset.get_data_dtype(data_key)
        if dtype == "string":
            dims = []
        else:
            feature_dims = [
                Dim(dimension=dim, name=f"{data_key}_dim{i + 1}")
                for i, dim in enumerate(self._dataset.get_data_shape(data_key))
            ]
            dims = [Dim(dimension=None, name=f"{data_key}_frame"), *feature_dims]
        sparse_dim = None
        if self._dataset.is_data_sparse(data_key):
            sparse_dim = Dim(dimension=self._dataset.get_data_dim(data_key), name=f"{data_key}_sparse")
            if data_key in self._dataset.labels:
                sparse_dim.vocab = Vocabulary.create_vocab_from_labels(self._dataset.labels[data_key])
        return Tensor(data_key, dims=dims, dtype=dtype, sparse_dim=sparse_dim)
