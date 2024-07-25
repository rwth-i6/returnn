__all__ = ["PostprocessingDataset"]

from dataclasses import dataclass
from numpy import ndarray
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from returnn.tensor import TensorDict
from returnn.util.basic import NumbersDict
from .basic import Dataset, init_dataset
from .util.strings import str_to_numpy_array


@dataclass
class LoadedData:
    tensors: TensorDict
    seq_idx: int


Postprocessor = Callable[[TensorDict], TensorDict]
StreamingPostprocessor = Callable[[Iterator[TensorDict]], Iterator[TensorDict]]


class PostprocessingDataset(Dataset):
    def __init__(
        self,
        dataset: Dict[str, Any],
        map_seq: Optional[Postprocessor] = None,
        map_seq_stream: Optional[StreamingPostprocessor] = None,
        _meta_info_cache: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
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
        self._num_seqs = None

        self._added_data: List[LoadedData] = []
        self._expected_load_seq_start = 0
        self._dataset: Optional[Dataset] = None
        self._data_keys: Optional[List[str]] = None
        self._data_iter: Optional[Iterator[Tuple[int, TensorDict]]] = None
        self._max_seq_idx = -1
        self._reached_final_seq = False

        if _meta_info_cache:
            self._data_keys = _meta_info_cache["_data_keys"]
            self.num_inputs = _meta_info_cache["num_inputs"]
            self.num_outputs = _meta_info_cache["num_outputs"]

    @property
    def num_seqs(self) -> int:
        if self._num_seqs is not None:
            return self._num_seqs
        if self.epoch is None:
            return 0
        raise NotImplementedError

    @property
    def _meta_info_cache(self) -> Optional[Dict[str, Any]]:
        if self._data_keys is None:
            return None
        return {
            "_data_keys": self._data_keys,
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
        }

    def _lazy_init_num_outputs(self):
        if self._data_keys is not None:
            return

        dataset = init_dataset(self._dataset_def, parent_dataset=self)
        dataset.init_seq_order(epoch=1)
        data = next(self._map_seq_stream(map(self._map_seq, self._iterate_dataset(dataset))))

        self._data_keys = sorted(data.data.keys())
        raise NotImplementedError  # TODO: num_outputs & friends
        self.num_outputs = None
        self.num_inputs = None

    def init_seq_order(self, epoch: Optional[int] = None, seq_list=None, seq_order=None):
        super().init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)

        if epoch is None:
            self._num_seqs = 0
            return True

        if self._dataset is None:
            self._dataset = init_dataset(self._dataset_def, parent_dataset=self)
        self._dataset.init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)

        self._added_data = []
        self._expected_load_seq_start = 0
        self._data_iter = enumerate(self._map_seq_stream(map(self._map_seq, self._iterate_dataset(self._dataset))))
        self._max_seq_idx = -1
        self._num_seqs = None
        self._reached_final_seq = False

        self._lazy_init_num_outputs()

        return True

    def _cleanup_old_seqs(self, seq_idx_end: int):
        i = next((i for i, el in enumerate(self._added_data) if el.seq_idx < seq_idx_end), 0)
        del self._added_data[:i]

    def _get_seq(self, seq_idx: int) -> Optional[LoadedData]:
        return next((data for data in self._added_data if data.seq_idx == seq_idx), None)

    def is_less_than_num_seqs(self, n: int):
        if n < self._expected_load_seq_start:
            return True
        if self.epoch is None:
            return False
        try:
            return super().is_less_than_num_seqs(n)
        except NotImplementedError:  # we only define num_seqs once we reach the end of the dataset
            assert n >= self._expected_load_seq_start
            self._load_seqs(self._expected_load_seq_start, n + 1)
            if self._get_seq(n) is not None:
                return True
            # We reached the end.
            assert n >= self._num_seqs
            return False

    def _load_seqs(self, start: int, end: int):
        # implementation similar to CachedDataset2
        assert start >= self._expected_load_seq_start
        if start > self._expected_load_seq_start:
            # Cleanup old data.
            self._cleanup_old_seqs(start)
            self._expected_load_seq_start = start

        while not self._added_data or self._added_data[-1].seq_idx < end:
            try:
                seq_idx, tensor_dict = next(self._data_iter)
            except StopIteration:
                self._reached_final_seq = True
                self._num_seqs = self._max_seq_idx + 1
                break

            loaded_data_keys = sorted(tensor_dict.data.keys())
            assert (
                loaded_data_keys == self._data_keys
            ), f"mismatch in data keys returned from data iterator, expected {self._data_keys} but received {loaded_data_keys}"
            self._added_data.append(LoadedData(tensors=tensor_dict, seq_idx=seq_idx))
            self._max_seq_idx = seq_idx

    def get_data(self, seq_idx: int, key: str) -> ndarray:
        return self._get_seq(seq_idx).tensors[key].raw_tensor

    def get_data_keys(self):
        self._lazy_init_num_outputs()
        return self._data_keys

    def get_seq_length(self, seq_idx: int) -> NumbersDict:
        assert seq_idx >= self._expected_load_seq_start
        self.load_seqs(self._expected_load_seq_start, seq_idx + 1)
        raise NotImplementedError
        return self._get_seq(seq_idx).tensors.something

    @staticmethod
    def _data_dict_to_tensor_dict(data_dict: Dict[str, ndarray]) -> TensorDict:
        """
        :return: the given data dict converted to a TensorDict class
        """
        raise NotImplementedError

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
