"""
HuggingFace dataset wrapper

See https://github.com/rwth-i6/returnn/issues/1257 for some initial discussion.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union, Any, Callable, Sequence, Dict, List
import numpy
from .basic import DatasetSeq
from .cached2 import CachedDataset2
from returnn.util.basic import OptionalNotImplementedError

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
        seq_tag_key: Optional[str] = "id",
        features=None,
        **kwargs,
    ):
        """
        :param dataset_opts: either a dict of options for :func:`datasets.load_dataset`
            or a path to a local dataset for :func:`datasets.load_from_disk`
        :param map_func: optional function to apply to the dataset after loading
        :param data_key: key (column name) in the dataset to use as input data
        :param seq_tag_key: key (column name) in the dataset to use as sequence tag.
            If None, will use the sequence index as tag.
        :param features: list of keys in the dataset to use as features (if None, use all except seq_tag_key)
        """
        super().__init__(**kwargs)

        self.dataset_opts = dataset_opts
        self.map_func = map_func

        self.hf_dataset: Optional[datasets.Dataset] = None
        self.data_key = data_key
        self.seq_tag_key = seq_tag_key
        self.feature_keys = features
        self.data_dtype: Dict[str, str] = {}

        self._seq_order = None

    def initialize(self):
        """initialize"""
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
        if self.seq_tag_key:
            assert self.seq_tag_key in self.hf_dataset.features, (
                f"{self}: seq_tag_key {self.seq_tag_key} not in dataset features {self.hf_dataset.features}"
            )
            assert self.hf_dataset.features[self.seq_tag_key].dtype == "string"
        if self.feature_keys is None:
            self.feature_keys = list(self.hf_dataset.features.keys())
            if self.seq_tag_key in self.feature_keys:
                self.feature_keys.remove(self.seq_tag_key)

        self.hf_dataset.set_format("numpy")

        if self.seq_tag_key is not None:
            assert self.seq_tag_key in self.hf_dataset.column_names

        self.labels = {}
        self.num_outputs = {}
        for key in self.feature_keys:
            feature = self.hf_dataset.features[key]
            num_classes = 1
            spatial_dims = 0
            while type(feature) is datasets.features.Sequence:
                spatial_dims += 1
                if feature.length != -1:
                    num_classes = feature.length
                feature = feature.feature
            if type(feature) is datasets.features.ClassLabel:
                self.labels[key] = feature.names
                dtype = feature.dtype
                num_classes = feature.num_classes
            elif type(feature) is datasets.features.Value:
                dtype = feature.dtype
            elif isinstance(feature, (datasets.features.Array2D, datasets.features.Array3D, datasets.features.Array4D)):
                dtype = feature.dtype
                num_classes = feature.shape[-1]
                spatial_dims += len(feature.shape)
            else:
                assert False, f"Unsupported feature type {type(feature)}"

            len_shape = spatial_dims
            self.num_outputs[key] = (num_classes, len_shape)

            self.data_dtype[key] = dtype

        super().initialize()

    def get_data_dim(self, key: str) -> int:
        """:return: data dimension for the given key"""
        if key in self.num_outputs:
            return self.num_outputs[key][0]
        return super().get_data_dim(key)

    def get_data_dtype(self, key: str) -> str:
        return self.data_dtype[key]

    def _get_seq_len(self, seq_idx: int):
        return len(self.hf_dataset[seq_idx][self.data_key])

    @property
    def num_seqs(self) -> int:
        """:return: number of sequences"""
        assert self._seq_order is not None, "num_seqs is only known after calling init_seq_order()"
        return len(self._seq_order)

    def get_tag(self, sorted_seq_idx: int) -> str:
        """:return: tag of the sequence"""
        return self.hf_dataset[self.get_corpus_seq_idx(sorted_seq_idx)][self.seq_tag_key]

    def get_all_tags(self) -> List[str]:
        """:return: all tags"""
        return list(self.hf_dataset[self.seq_tag_key])

    def init_seq_order(self, epoch: Optional[int] = None, seq_list=None, seq_order=None):
        """
        :param int|None epoch:
        :param list[str]|None seq_list: List of sequence tags, to set a predefined order.
        :param list[int]|None seq_order: List of corpus sequence indices, to set a predefined order.
        :rtype: bool
        :returns whether the order changed (True is always safe to return)
        """
        super().init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)

        if seq_order:
            self._seq_order = seq_order
            # TODO can we return False?
            return True

        if seq_list:
            all_tags = self.get_all_tags()
            self._seq_order = [all_tags.index(tag) for tag in seq_list]
            # TODO can we return False?
            return True

        try:
            self._seq_order = self.get_seq_order_for_epoch(
                epoch=epoch, num_seqs=self.hf_dataset.num_rows, get_seq_len=self._get_seq_len
            )
        except OptionalNotImplementedError:
            # only support seq_ordering that need no length here
            assert self.seq_ordering in ["default", "reverse", "random"]
            self._seq_order = self.get_seq_order_for_epoch(
                epoch=epoch, num_seqs=self.hf_dataset.num_rows, get_seq_len=None
            )

        return True

    def _collect_single_seq(self, seq_idx: int) -> DatasetSeq:
        corpus_seq_idx = self.get_corpus_seq_idx(seq_idx)

        def _ensure_numpy(x):
            if not isinstance(x, numpy.ndarray):
                return numpy.array(x)
            return x

        dataset_item = self.hf_dataset[corpus_seq_idx]
        if self.seq_tag_key:
            seq_tag = dataset_item[self.seq_tag_key]
            assert isinstance(seq_tag, str)
        else:
            seq_tag = f"seq-{corpus_seq_idx}"
        features = {f: _ensure_numpy(dataset_item[f]) for f in self.feature_keys}
        return DatasetSeq(seq_idx, features=features, seq_tag=seq_tag)

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
