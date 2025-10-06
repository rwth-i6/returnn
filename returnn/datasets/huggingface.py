"""
HuggingFace dataset wrapper

See https://github.com/rwth-i6/returnn/issues/1257 for some initial discussion.
"""

from __future__ import annotations
from typing import Optional, Sequence, Dict, List
import numpy
from .basic import DatasetSeq
from .cached2 import CachedDataset2
from returnn.util.basic import OptionalNotImplementedError


class HuggingfaceDataset(CachedDataset2):
    """
    HuggingFace dataset wrapper.
    """

    def __init__(
        self,
        *,
        dataset_opts,
        map_func=None,
        map_func_args=None,
        data_key="data",
        seq_tag_key="id",
        features=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._seq_order = None

        self.dataset_opts = dataset_opts

        if map_func_args is not None:
            map_func = map_func(**map_func_args)
        self.map_func = map_func
        self.dataset = None
        self.data_key = data_key
        self.seq_tag_key = seq_tag_key

        self.feature_keys = features

        self.data_dtype: Dict[str, str] = {}

    def initialize(self):
        """initialize"""
        # Load the dataset
        # noinspection PyUnresolvedReferences,PyPackageRequirements
        import datasets

        if isinstance(self.dataset_opts, dict):
            self.dataset = datasets.load_dataset(**self.dataset_opts)
        else:
            self.dataset = datasets.load_from_disk(self.dataset_opts)
            assert isinstance(self.dataset, datasets.Dataset)
        if self.map_func is not None:
            self.dataset = self.map_func(self.dataset)
        if self.feature_keys is None:
            self.feature_keys = list(self.dataset.features.keys())
            if self.seq_tag_key is not None and self.seq_tag_key in self.feature_keys:
                self.feature_keys.remove(self.seq_tag_key)
            else:
                assert False, "Dataset does not have a seq_tag"

        self.dataset.set_format("numpy")

        if self.seq_tag_key is not None:
            assert self.seq_tag_key in self.dataset.column_names

        self.labels = {}
        self.num_outputs = {}
        for key in self.feature_keys:
            feature = self.dataset.features[key]
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
        return len(self.dataset[seq_idx][self.data_key])

    @property
    def num_seqs(self) -> int:
        """:return: number of sequences"""
        assert self._seq_order is not None, "num_seqs is only known after calling init_seq_order()"
        return len(self._seq_order)

    def get_tag(self, sorted_seq_idx: int) -> str:
        """:return: tag of the sequence"""
        return self.dataset[int(self.get_corpus_seq_idx(sorted_seq_idx))][self.seq_tag_key]

    def get_all_tags(self) -> List[str]:
        """:return: all tags"""
        return list(self.dataset[self.seq_tag_key])

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
                epoch=epoch, num_seqs=self.dataset.num_rows, get_seq_len=self._get_seq_len
            )
        except OptionalNotImplementedError:
            # only support seq_ordering that need no length here
            assert self.seq_ordering in ["default", "reverse", "random"]
            self._seq_order = self.get_seq_order_for_epoch(
                epoch=epoch, num_seqs=self.dataset.num_rows, get_seq_len=None
            )

        return True

    def _collect_single_seq(self, seq_idx: int) -> DatasetSeq:
        corpus_seq_idx = self.get_corpus_seq_idx(seq_idx)

        def ensure_numpy(x):
            if not isinstance(x, numpy.ndarray):
                return numpy.array(x)
            return x

        dataset_item = self.dataset[int(corpus_seq_idx)]
        features = {f: ensure_numpy(dataset_item[f]) for f in self.feature_keys}
        return DatasetSeq(seq_idx, features=features, targets=None, seq_tag=dataset_item[self.seq_tag_key])

    def get_current_seq_order(self) -> Sequence[int]:
        """
        :return: list of corpus seq idx
        """
        assert self._seq_order is not None
        return self._seq_order

    def get_corpus_seq_idx(self, sorted_seq_idx: int) -> int:
        """:return: corpus seq idx"""
        return self._seq_order[sorted_seq_idx]

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
