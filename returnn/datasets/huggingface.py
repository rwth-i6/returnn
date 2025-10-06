"""
HuggingFace dataset wrapper

See https://github.com/rwth-i6/returnn/issues/1257 for some initial discussion.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union, Any, Callable, Sequence, Dict, List
import numpy
from returnn.tensor import Tensor
from .basic import DatasetSeq
from .cached2 import CachedDataset2
from .util.vocabulary import Vocabulary

if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    import datasets


class HuggingfaceDataset(CachedDataset2):
    """
    HuggingFace dataset wrapper.
    """

    def __init__(
        self,
        dataset_opts: Union[Dict[str, Any], str],
        *,
        map_func: Optional[Callable[[datasets.Dataset], datasets.Dataset]] = None,
        data_key: str = "data",
        seq_tag_column: Optional[str] = "id",
        data_format: Dict[str, Dict[str, Any]],
        **kwargs,
    ):
        """
        :param dataset_opts: either a dict of options for :func:`datasets.load_dataset`
            or a path to a local dataset for :func:`datasets.load_from_disk`
        :param map_func: optional function to apply to the dataset after loading
        :param data_key: key (column name) in the dataset to use as input data
        :param seq_tag_column: key (column name) in the dataset to use as sequence tag.
            If None, will use the sequence index as tag.
        :param data_format:
            For each column name (data key), specify the format,
            as a dict with entries for "dim", "ndim", "shape", and/or "dtype",
            compatible to :class:`Tensor`.
        """
        super().__init__(**kwargs)

        self.dataset_opts = dataset_opts
        self.map_func = map_func

        self.hf_dataset: Optional[datasets.Dataset] = None
        self.data_key = data_key
        self.seq_tag_column: Optional[str] = seq_tag_column
        self.data_format: Dict[str, Tensor] = {k: _make_tensor_template(v, k) for k, v in data_format.items()}

        self.labels = {k: data.vocab.labels for k, data in self.data_format.items() if data.vocab}
        self.num_outputs = {k: (data.dim, data.ndim) for k, data in self.data_format.items()}
        self._seq_order: Optional[Sequence[int]] = None

    def _lazy_init(self):
        if self.hf_dataset is not None:
            return

        # Load the dataset
        # noinspection PyUnresolvedReferences,PyPackageRequirements
        import datasets

        if isinstance(self.dataset_opts, dict):
            self.hf_dataset = datasets.load_dataset(**self.dataset_opts)
        else:
            self.hf_dataset = datasets.load_from_disk(self.dataset_opts)
        assert isinstance(self.hf_dataset, datasets.Dataset), (
            f"{self}: Expected single dataset, got {type(self.hf_dataset)} {self.hf_dataset}. Specify split if needed."
        )
        if self.map_func is not None:
            self.hf_dataset = self.map_func(self.hf_dataset)
        if self.seq_tag_column:
            assert self.seq_tag_column in self.hf_dataset.features, (
                f"{self}: seq_tag_column {self.seq_tag_column} not in dataset features {self.hf_dataset.features}"
            )
            assert self.hf_dataset.features[self.seq_tag_column].dtype in ("string", "int64"), (
                f"{self}: seq_tag_column {self.seq_tag_column} must be of dtype string or int64,"
                f" got {self.hf_dataset.features[self.seq_tag_column].dtype}"
            )

        self.hf_dataset.set_format("numpy")

        for key, user_format in self.data_format.items():
            feature = self.hf_dataset.features[key]
            inferred_format = _infer_data_format_for_feature(feature, f"{self}: column {key}: ")
            for key_ in ["dim", "ndim", "dtype"]:
                assert getattr(user_format, key_) == inferred_format[key_], (
                    f"{self}: column {key}, user-specified {user_format}, {key_}:"
                    f" user-specified {getattr(user_format, key_)} does not match inferred {inferred_format[key_]}"
                )
            if "vocab" in inferred_format and not user_format.vocab:
                assert user_format.sparse, f"{self}: column {key}: user_format expected to be sparse, got {user_format}"
                user_format.sparse_dim.vocab = Vocabulary.create_vocab(**inferred_format["vocab"])
                self.labels[key] = user_format.vocab.labels

    def get_data_keys(self) -> List[str]:
        """:return: list of data keys"""
        return list(self.data_format.keys())

    def get_target_list(self) -> List[str]:
        """:return: list of target keys"""
        return self.get_data_keys()  # it's somewhat arbitrary...

    def get_data_shape(self, key: str) -> List[int]:
        """:return: data shape for the given key"""
        return list(self.data_format[key].shape)

    def get_data_dim(self, key: str) -> int:
        """:return: data dimension for the given key"""
        return self.data_format[key].dim

    def is_data_sparse(self, key: str) -> bool:
        """:return: whether the data is sparse for the given key"""
        return self.data_format[key].sparse

    def get_data_dtype(self, key: str) -> str:
        """:return: dtype"""
        return self.data_format[key].dtype

    def _get_seq_len(self, seq_idx: int):
        return len(self.hf_dataset[seq_idx][self.data_key])  # noqa

    @property
    def num_seqs(self) -> int:
        """:return: number of sequences"""
        assert self._seq_order is not None, "num_seqs is only known after calling init_seq_order()"
        return len(self._seq_order)

    def get_tag(self, sorted_seq_idx: int) -> str:
        """:return: tag of the sequence"""
        corpus_seq_idx = self.get_corpus_seq_idx(sorted_seq_idx)
        dataset_item = self.hf_dataset[corpus_seq_idx]
        return self._get_seq_tag(corpus_seq_idx, dataset_item)

    def get_all_tags(self) -> List[str]:
        """:return: all tags"""
        self._lazy_init()
        if self.seq_tag_column:
            return list(self.hf_dataset[self.seq_tag_column])
        else:
            return [f"seq-{i}" for i in range(self.hf_dataset.num_rows)]

    def init_seq_order(
        self,
        epoch: Optional[int] = None,
        seq_list: Optional[Sequence[str]] = None,
        seq_order: Optional[Sequence[int]] = None,
    ) -> bool:
        """
        :param epoch:
        :param seq_list: List of sequence tags, to set a predefined order.
        :param seq_order: List of corpus sequence indices, to set a predefined order.
        :returns whether the order changed (True is always safe to return)
        """
        super().init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)

        if seq_order is not None:
            self._seq_order = seq_order
        elif seq_list is not None:
            all_tags = self.get_all_tags()
            self._seq_order = [all_tags.index(tag) for tag in seq_list]
        elif epoch is None:
            self._seq_order = ()
            return True
        else:
            self._lazy_init()
            self._seq_order = self.get_seq_order_for_epoch(
                epoch=epoch, num_seqs=self.hf_dataset.num_rows, get_seq_len=self._get_seq_len
            )
        return True

    def _collect_single_seq(self, seq_idx: int) -> DatasetSeq:
        corpus_seq_idx = self.get_corpus_seq_idx(seq_idx)

        def _ensure_numpy(x):
            if not isinstance(x, numpy.ndarray):
                return numpy.array(x)
            return x

        dataset_item = self.hf_dataset[corpus_seq_idx]
        seq_tag = self._get_seq_tag(corpus_seq_idx, dataset_item)
        features = {f: _ensure_numpy(dataset_item[f]) for f in self.data_format}
        return DatasetSeq(seq_idx, features=features, seq_tag=seq_tag)

    def _get_seq_tag(self, corpus_seq_idx: int, dataset_item: Dict[str, Any]) -> str:
        if self.seq_tag_column:
            seq_tag = dataset_item[self.seq_tag_column]
            assert isinstance(seq_tag, (str, int, numpy.int64)), f"got {type(seq_tag)} {seq_tag!r}"
            if not isinstance(seq_tag, str):
                seq_tag = str(seq_tag)
        else:
            seq_tag = f"seq-{corpus_seq_idx}"
        return seq_tag

    def get_current_seq_order(self) -> Sequence[int]:
        """:return: list of corpus seq idx"""
        assert self._seq_order is not None
        return self._seq_order

    def get_corpus_seq_idx(self, sorted_seq_idx: int) -> int:
        """:return: corpus seq idx"""
        return int(self._seq_order[sorted_seq_idx])

    def can_serialize_data(self, key: str):
        """:return: whether we can serialize"""
        return True

    def serialize_data(self, key: str, data: numpy.ndarray) -> str:
        """serialize"""
        if key in self.labels:
            return super().serialize_data(key, data)
        if isinstance(data, numpy.ndarray):
            data = data.tolist()
        return data


def _infer_data_format_for_feature(
    feature: Union[
        datasets.features.Sequence,
        datasets.features.ClassLabel,
        datasets.features.Value,
        datasets.features.Array2D,
        datasets.features.Array3D,
        datasets.features.Array4D,
        datasets.features.Audio,
    ],
    exc_prefix: str = "",
) -> Dict[str, Any]:
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    import datasets

    labels = None
    num_classes = None
    num_dims = 0
    while isinstance(feature, datasets.features.Sequence):
        num_dims += 1
        if feature.length != -1:
            num_classes = feature.length
        feature = feature.feature
    if isinstance(feature, datasets.features.ClassLabel):
        labels = feature.names
        dtype = feature.dtype
        num_classes = feature.num_classes  # noqa
    elif isinstance(feature, datasets.features.Value):
        dtype = feature.dtype
    elif isinstance(feature, (datasets.features.Array2D, datasets.features.Array3D, datasets.features.Array4D)):
        dtype = feature.dtype
        num_classes = feature.shape[-1]
        num_dims += len(feature.shape)
    elif isinstance(feature, datasets.features.Audio):
        if feature.decode:
            dtype = "float32"  # samples
        else:
            dtype = "uint8"  # bytes
        num_dims += 1  # time axis
    else:
        assert False, f"{exc_prefix}unsupported feature type {type(feature)} {feature}"

    d = {"dim": num_classes, "ndim": num_dims, "dtype": dtype}
    if labels:
        d["sparse"] = True
        d["vocab"] = {"vocab_file": None, "labels": labels, "unknown_label": None}
    return d


def _make_tensor_template(data: Union[Dict[str, Any], Tensor], name: str) -> Tensor:
    if isinstance(data, Tensor):
        data = data.copy(name)
    else:
        assert isinstance(data, dict)
        data = Tensor(name, batch_dim_axis=None, **data)
    assert data.batch_dim_axis is None
    return data
