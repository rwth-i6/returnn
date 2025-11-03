"""
HuggingFace dataset wrapper

See https://github.com/rwth-i6/returnn/issues/1257 for some initial discussion.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union, Any, Callable, Sequence, Dict, List
import os
import re
import numpy
from returnn.tensor import Tensor
from returnn.util import file_cache
from .basic import DatasetSeq
from .cached2 import CachedDataset2
from .util.vocabulary import Vocabulary
from .util.strings import str_to_numpy_array

if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    import datasets


class HuggingFaceDataset(CachedDataset2):
    """
    HuggingFace dataset wrapper.
    """

    def __init__(
        self,
        dataset_opts: Union[
            Dict[str, Any],
            str,
            os.PathLike,
            Sequence[Union[str, os.PathLike]],
            Callable[[], Union[Dict[str, Any], str, os.PathLike, Sequence[Union[str, os.PathLike]], datasets.Dataset]],
        ],
        *,
        use_file_cache: bool = False,
        map_func: Optional[Callable[[datasets.Dataset], datasets.Dataset]] = None,
        rename_columns: Optional[Dict[str, str]] = None,
        cast_columns: Optional[Dict[str, Dict[str, Any]]] = None,
        data_format: Dict[str, Dict[str, Any]],
        seq_tag_column: Optional[str] = "id",
        sorting_seq_len_column_data: Optional[str] = None,
        sorting_seq_len_column: Optional[str] = None,
        **kwargs,
    ):
        """
        :param dataset_opts: either a dict of options for :func:`datasets.load_dataset`
            or a path to a local dataset for :func:`datasets.load_from_disk`,
            or a list of Arrow filenames to load with :func:`datasets.Dataset.from_file` and concatenate.
            It can also be a callable returning one of the above,
            or returning a :class:`datasets.Dataset` directly.
        :param use_file_cache: if True, will cache the dataset files on local disk using :mod:`file_cache`.
            This only works for dataset_opts which is a str or list of str (or callable returning that).
        :param map_func: optional function to apply to the dataset after loading
        :param rename_columns: if given, will rename these columns
        :param cast_columns: if given, will cast these columns to the specified types.
            This is useful if the dataset has not the expected types.
            See :func:`datasets.Dataset.cast` for details.
            You can also e.g. enforce some sample_rate for audio, etc.
        :param data_format:
            For each column name (data key), specify the format,
            as a dict with entries for "dim", "ndim", "shape", and/or "dtype",
            compatible to :class:`Tensor`.
            It can be a subset of the available columns.
            If "vocab" is specified, and the underlying HF datasets column is of dtype "string",
            it will automatically tokenize the string using the vocab.
        :param seq_tag_column: key (column name) in the dataset to use as sequence tag.
            If None, will use the sequence index as tag.
        :param sorting_seq_len_column_data: key (column name) in the dataset to use for sorting by sequence length.
            It will take len(dataset[sorting_seq_len_column_data]) as sequence length (only for sorting/shuffling).
        :param sorting_seq_len_column: key (column name) in the dataset to use for sorting by sequence length.
            It will take the value of dataset[sorting_seq_len_column] as sequence length (only for sorting/shuffling).
            E.g. some datasets provide "duration", "duration_ms", "wav_filesize" or similar such information
            which can be used.
        """
        super().__init__(**kwargs)

        self.dataset_opts = dataset_opts
        self.use_file_cache = use_file_cache
        self.map_func = map_func
        self.rename_columns = rename_columns
        self.cast_columns = cast_columns

        self.data_format: Dict[str, Tensor] = {k: _make_tensor_template(v, k) for k, v in data_format.items()}
        self.seq_tag_column: Optional[str] = seq_tag_column
        self.sorting_seq_len_column_data = sorting_seq_len_column_data
        self.sorting_seq_len_column = sorting_seq_len_column

        self.labels = {k: data.vocab.labels for k, data in self.data_format.items() if data.vocab}
        self.num_outputs = {k: (data.dim or 1, data.ndim) for k, data in self.data_format.items()}

        self.hf_dataset: Optional[datasets.Dataset] = None  # lazily loaded, _lazy_init
        self._seq_order: Optional[Sequence[int]] = None  # init_seq_order
        self._seq_tags: Optional[List[str]] = None  # get_all_tags cache

    def _lazy_init(self):
        if self.hf_dataset is not None:
            return

        # Load the dataset
        # noinspection PyUnresolvedReferences,PyPackageRequirements
        import datasets

        dataset_opts = self.dataset_opts
        if callable(dataset_opts):
            dataset_opts = dataset_opts()
        if self.use_file_cache:
            assert isinstance(dataset_opts, (str, os.PathLike, list, tuple)), (
                f"{self}: with use_file_cache, dataset_opts must be str or list of str, got {type(dataset_opts)}"
            )
            if isinstance(dataset_opts, (str, os.PathLike)):
                dataset_opts = get_arrow_shard_files_from_hf_dataset_dir(dataset_opts)
            assert isinstance(dataset_opts, (list, tuple))
            cache = file_cache.get_instance()
            dataset_opts = [cache.get_file(os.fspath(fn)) for fn in dataset_opts]
            self.set_file_cache(cache)
        if isinstance(dataset_opts, dict):
            self.hf_dataset = datasets.load_dataset(**dataset_opts)
        elif isinstance(dataset_opts, str):
            self.hf_dataset = datasets.load_from_disk(dataset_opts)
        elif isinstance(dataset_opts, (list, tuple)):
            self.hf_dataset = datasets.concatenate_datasets([datasets.Dataset.from_file(fn) for fn in dataset_opts])
        elif isinstance(dataset_opts, datasets.Dataset):
            self.hf_dataset = dataset_opts
        else:
            raise TypeError(f"{self}: invalid dataset_opts type {type(dataset_opts)}")
        assert isinstance(self.hf_dataset, datasets.Dataset), (
            f"{self}: Expected single dataset, got {type(self.hf_dataset)} {self.hf_dataset}. Specify split if needed."
        )

        if self.map_func is not None:
            self.hf_dataset = self.map_func(self.hf_dataset)

        if self.rename_columns:
            self.hf_dataset = self.hf_dataset.rename_columns(self.rename_columns)

        if self.cast_columns:
            # Note: prefer cast_column, as this can avoid using `map`, i.e. be faster.
            for key, column_format in self.cast_columns.items():
                assert key in self.hf_dataset.features, (
                    f"{self}: cast_column {key} not in dataset features {self.hf_dataset.features}"
                )
                feat = datasets.features.features.generate_from_dict(column_format)
                self.hf_dataset = self.hf_dataset.cast_column(key, feat)

        if self.seq_tag_column:
            assert self.seq_tag_column in self.hf_dataset.features, (
                f"{self}: seq_tag_column {self.seq_tag_column} not in dataset features {self.hf_dataset.features}"
            )
            assert self.hf_dataset.features[self.seq_tag_column].dtype in ("string", "int64"), (
                f"{self}: seq_tag_column {self.seq_tag_column} must be of dtype string or int64,"
                f" got {self.hf_dataset.features[self.seq_tag_column].dtype}"
            )

        selected_columns = list(self.data_format.keys())
        if self.seq_tag_column and self.seq_tag_column not in selected_columns:
            selected_columns.append(self.seq_tag_column)
        if self.sorting_seq_len_column and self.sorting_seq_len_column not in selected_columns:
            selected_columns.append(self.sorting_seq_len_column)
        if self.sorting_seq_len_column_data and self.sorting_seq_len_column_data not in selected_columns:
            selected_columns.append(self.sorting_seq_len_column_data)
        self.hf_dataset = self.hf_dataset.select_columns(selected_columns)

        self.hf_dataset.set_format("numpy")

        for key, user_format in self.data_format.items():
            feature = self.hf_dataset.features[key]
            inferred_format = _infer_data_format_for_feature(feature, f"{self}: column {key}: ")
            if user_format.vocab and inferred_format["dtype"] == "string":
                pass  # allow to auto-tokenize strings when vocab is specified
            else:
                for key_ in ["dtype", "ndim", "dim"]:
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

    def _get_seq_len(self, seq_idx: int) -> Union[int, float]:
        if self._seq_order_seq_lens_by_idx is not None:
            self._get_seq_len = self._seq_order_seq_lens_by_idx.__getitem__  # faster
            return self._seq_order_seq_lens_by_idx[seq_idx]
        assert not self._seq_order_seq_lens_file  # not expected to call this
        if self.sorting_seq_len_column:
            self._seq_order_seq_lens_by_idx = numpy.array(self.hf_dataset[self.sorting_seq_len_column])
            self._get_seq_len = self._seq_order_seq_lens_by_idx.__getitem__  # faster
            v = self._seq_order_seq_lens_by_idx[seq_idx]
            return int(v)  # noqa
        if self.sorting_seq_len_column_data:
            v = self.hf_dataset[seq_idx][self.sorting_seq_len_column_data]
            return len(v)  # noqa
        raise ValueError(
            f"{self}: sorting/shuffling by seq len not configured,"
            f" need sorting_seq_len_column or sorting_seq_len_column_data"
        )

    @property
    def num_seqs(self) -> int:
        """:return: number of sequences"""
        assert self._seq_order is not None, "num_seqs is only known after calling init_seq_order()"
        return len(self._seq_order)

    def get_tag(self, sorted_seq_idx: int) -> str:
        """:return: tag of the sequence"""
        corpus_seq_idx = self.get_corpus_seq_idx(sorted_seq_idx)
        self._lazy_init()
        dataset_item = self.hf_dataset[corpus_seq_idx]
        return self._get_seq_tag(corpus_seq_idx, dataset_item)

    def get_all_tags(self) -> List[str]:
        """:return: all tags"""
        if self._seq_tags is not None:
            return self._seq_tags
        self._lazy_init()
        if self.seq_tag_column:
            res = list(map(str, self.hf_dataset[self.seq_tag_column]))
        else:
            res = [f"seq-{i}" for i in range(self.hf_dataset.num_rows)]
        self._seq_tags = res
        return res

    def get_total_num_seqs(self, *, fast: bool = False) -> int:
        """:return: total number of sequences in the dataset"""
        if fast:
            return super().get_total_num_seqs(fast=True)
        self._lazy_init()
        return self.hf_dataset.num_rows

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
        else:
            self._lazy_init()
            self._seq_order = self.get_seq_order_for_epoch(
                epoch=epoch, num_seqs=self.hf_dataset.num_rows, get_seq_len=self._get_seq_len
            )
        return True

    def _collect_single_seq(self, seq_idx: int) -> DatasetSeq:
        # noinspection PyUnresolvedReferences,PyPackageRequirements
        import datasets

        corpus_seq_idx = self.get_corpus_seq_idx(seq_idx)

        def _ensure_numpy(k, x):
            if isinstance(x, numpy.ndarray):  # fast path
                return x
            if isinstance(x, str):
                if self.data_format[k].dtype == "string":
                    return str_to_numpy_array(x)
                if self.data_format[k].vocab:
                    return numpy.array(self.data_format[k].vocab.get_seq(x), dtype=self.data_format[k].dtype)
                raise ValueError(f"{self}: column {k}: cannot convert string {x!r} to numpy array")
            feat = self.hf_dataset.features[k]
            if isinstance(feat, datasets.features.Audio):
                # In HF datasets 3, this is just a dict.
                # In HF datasets 4, this can also be a datasets.features._torchcodec.AudioDecoder.
                assert isinstance(x, dict) or x.__class__.__name__ == "AudioDecoder"
                if feat.decode:
                    x = x["array"]
                else:
                    x = x["bytes"]
            if isinstance(x, numpy.ndarray):  # fast path
                return x
            if isinstance(x, (bytes, bytearray)):
                return numpy.frombuffer(x, dtype=self.data_format[k].dtype)
            return numpy.array(x)

        self._lazy_init()
        dataset_item = self.hf_dataset[corpus_seq_idx]
        seq_tag = self._get_seq_tag(corpus_seq_idx, dataset_item)
        features = {k: _ensure_numpy(k, dataset_item[k]) for k in self.data_format}
        return DatasetSeq(seq_idx, features=features, seq_tag=seq_tag)

    def _get_seq_tag(self, corpus_seq_idx: int, dataset_item: Dict[str, Any]) -> str:
        if self.seq_tag_column:
            seq_tag = dataset_item[self.seq_tag_column]
            assert isinstance(seq_tag, (str, int, numpy.int64)), f"got {type(seq_tag)} {seq_tag!r}"
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


def get_arrow_shard_files_from_hf_dataset_dir(hf_data_dir: Union[str, os.PathLike]) -> List[str]:
    """
    Given some HF datasets directory (created via :func:`datasets.save_to_disk`),
    return the list of Arrow shard files (``data-*-of-*.arrow``).
    This also verifies that the directory looks like a valid HF datasets directory.
    The order of the returned list is by shard index.
    Note that this does not load the dataset, just inspects the directory structure.

    :param hf_data_dir: directory
    :return: list of Arrow shard files
    """
    hf_data_dir = os.fspath(hf_data_dir)
    content = os.listdir(hf_data_dir)
    assert "state.json" in content, f"not a valid HF datasets dir: {hf_data_dir!r}"
    assert "dataset_info.json" in content, f"not a valid HF datasets dir: {hf_data_dir!r}"
    pat = re.compile("^(.*)-([0-9]+)-of-([0-9]+).arrow$")
    content = [pat.match(fn) for fn in content]
    content = [m for m in content if m]
    assert content, f"no matching .arrow files in {hf_data_dir!r} found, expected *-*-of-*.arrow"
    prefix = content[0].group(1)
    assert all(m.group(1) == prefix for m in content), (
        f"mismatching prefix in {hf_data_dir!r}, expected {prefix}, got {[m.group(1) for m in content]}"
    )
    num_shards = int(content[0].group(3))
    assert all(int(m.group(3)) == num_shards for m in content), (
        f"mismatching number of shards in {hf_data_dir!r}, expected {num_shards}, got {[m.group(3) for m in content]}"
    )
    assert len(content) == num_shards, f"expected {num_shards} shard files in {hf_data_dir!r}, got {content}"
    content_by_idx = {int(m.group(2)): m for m in content}
    assert set(content_by_idx.keys()) == set(range(num_shards)), (
        f"expected shard indices 0..{num_shards - 1} in {hf_data_dir!r}, got {sorted(content_by_idx.keys())}"
    )
    return [hf_data_dir + "/" + content_by_idx[i].group(0) for i in range(num_shards)]


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
        feature: datasets.features.List  # typing for HF datasets 4
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
