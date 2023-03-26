"""
Provides :class:`CachedDataset2`.
"""

from __future__ import annotations
import typing
from typing import Optional
from threading import Condition
from .basic import Dataset, DatasetSeq

try:
    # noinspection PyCompatibility
    from _thread import interrupt_main
except ImportError:
    # noinspection PyUnresolvedReferences,PyCompatibility
    from thread import interrupt_main


class CachedDataset2(Dataset):
    """
    Somewhat like CachedDataset, but different.
    Simpler in some sense. And more generic. Caching might be worse.

    If you derive from this class:
    - you must override `_collect_single_seq`
    - you must set `num_inputs` (dense-dim of "data" key) and `num_outputs` (dict key -> dim, ndim-1)
    - you should set `labels`
    - handle seq ordering by overriding `init_seq_order`
    - you can set `_estimated_num_seqs`
    - you can set `_num_seqs` or `_num_timesteps` if you know them in advance
    """

    def __init__(self, **kwargs):
        super(CachedDataset2, self).__init__(**kwargs)
        self._num_timesteps = None
        self.epoch = None
        self.reached_final_seq = False
        self.added_data = []  # type: typing.List[DatasetSeq]
        self.expected_load_seq_start = 0
        self._num_timesteps_accumulated = 0

    def init_seq_order(self, epoch=None, seq_list=None, seq_order=None):
        """
        :param int|None epoch:
        :param list[str]|None seq_list: List of sequence tags, to set a predefined order.
        :param list[int]|None seq_order: List of corpus sequence indices, to set a predefined order. Only possible
          if the dataset has such indices (see self.have_corpus_seq_idx()).
        :rtype: bool
        :returns whether the order changed (True is always safe to return)

        This is called when we start a new epoch, or at initialization.
        Call this when you reset the seq list.
        """
        super(CachedDataset2, self).init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)
        if not epoch:
            epoch = 1
        self.expected_load_seq_start = 0
        self.reached_final_seq = False
        self.added_data = []
        self._num_timesteps_accumulated = 0
        self._num_seqs = None
        self.epoch = epoch
        return True

    def _cleanup_old_seqs(self, seq_idx_end):
        """
        :param int seq_idx_end:
        """
        i = 0
        while i < len(self.added_data):
            if self.added_data[i].seq_idx >= seq_idx_end:
                break
            i += 1
        del self.added_data[:i]

    def _get_seq(self, seq_idx):
        """
        :param int seq_idx:
        :rtype: DatasetSeq|None
        """
        for data in self.added_data:
            if data.seq_idx == seq_idx:
                return data
        return None

    def is_cached(self, start, end):
        """
        :param int start:
        :param int end:
        :rtype: bool
        """
        # Always False, to force that we call self._load_seqs().
        # This is important for our buffer management.
        return False

    @property
    def num_seqs(self):
        """
        :rtype: int
        """
        if self._num_seqs is not None:
            return self._num_seqs
        raise NotImplementedError

    def _load_seqs(self, start, end):
        """
        :param int start: inclusive seq idx start
        :param int end: exclusive seq idx end. can be more than num_seqs
        If end > num_seqs, will not load them.
        """
        # We expect that start increase monotonic on each call
        # for not-yet-loaded data.
        # This will already be called with _load_seqs_superset indices.
        assert start >= self.expected_load_seq_start
        if start > self.expected_load_seq_start:
            # Cleanup old data.
            self._cleanup_old_seqs(start)
            self.expected_load_seq_start = start
        if self.added_data:
            start = max(self.added_data[-1].seq_idx + 1, start)
        seqs = [self._collect_single_seq(seq_idx=seq_idx) for seq_idx in range(start, end)]
        seqs = list(filter(None, seqs))  # We might not know the num seqs in advance.
        self._num_timesteps_accumulated += sum([seq.num_frames for seq in seqs])
        self.added_data += seqs

    def is_less_than_num_seqs(self, n):
        """
        :param int n:
        :rtype: int
        """
        if n < self.expected_load_seq_start:
            return True
        # noinspection PyBroadException
        try:
            return super(CachedDataset2, self).is_less_than_num_seqs(n)
        except Exception:  # can fail, e.g. if self.num_seqs is not defined
            assert n >= self.expected_load_seq_start
            self._load_seqs(self.expected_load_seq_start, n + 1)
            if self._get_seq(n) is not None:
                return True
            # We reached the end.
            assert self.added_data, "Not a single seq was loaded?"
            self._num_seqs = self.added_data[-1].seq_idx + 1
            assert n >= self._num_seqs
            self.reached_final_seq = True
            return False

    def _collect_single_seq(self, seq_idx: int) -> Optional[DatasetSeq]:
        """
        :param seq_idx:
        :returns DatasetSeq or None if seq_idx >= num_seqs.
        """
        raise NotImplementedError

    def get_num_timesteps(self):
        """
        :rtype: int
        """
        if self._num_timesteps is not None:
            return self._num_timesteps
        else:
            assert self.reached_final_seq
            return self._num_timesteps_accumulated

    def _load_something(self):
        if self.added_data:
            return
        self.load_seqs(self.expected_load_seq_start, self.expected_load_seq_start + 1)

    def get_seq_length(self, sorted_seq_idx):
        """
        :type sorted_seq_idx: int
        :rtype: returnn.util.NumbersDict
        """
        # get_seq_length() can be called before the seq is loaded via load_seqs().
        # Thus, we just call load_seqs() ourselves here.
        assert sorted_seq_idx >= self.expected_load_seq_start
        self.load_seqs(self.expected_load_seq_start, sorted_seq_idx + 1)
        return self._get_seq(sorted_seq_idx).num_frames

    def get_data(self, seq_idx, key):
        """
        :param int seq_idx:
        :param str key:
        :rtype: numpy.ndarray
        """
        return self._get_seq(seq_idx).features[key]

    def get_input_data(self, seq_idx):
        """
        :param int seq_idx:
        :rtype: numpy.ndarray
        """
        return self.get_data(seq_idx, "data")

    def get_targets(self, target, seq_idx):
        """
        :param str target:
        :param int seq_idx:
        :rtype: numpy.ndarray
        """
        return self.get_data(seq_idx, target)

    def get_tag(self, sorted_seq_idx):
        """
        :param int sorted_seq_idx:
        :rtype: str
        """
        # get_tag() can be called before the seq is loaded via load_seqs().
        # Thus, we just call load_seqs() ourselves here.
        self.load_seqs(self.expected_load_seq_start, sorted_seq_idx + 1)
        return self._get_seq(sorted_seq_idx).seq_tag

    def get_data_keys(self):
        """
        :rtype: list[str]
        """
        self._load_something()
        return sorted(self.added_data[0].get_data_keys())

    def get_target_list(self):
        """
        Target data keys are usually not available during inference.
        Overwrite this if your dataset is more custom.
        """
        keys = list(self.get_data_keys())
        if "data" in keys:
            keys.remove("data")
        return keys

    def is_data_sparse(self, key):
        """
        :param str key: e.g. "data" or "classes"
        :rtype: bool
        """
        if key in self.num_outputs:
            return self.num_outputs[key][1] == 1
        self._load_something()
        return len(self.added_data[0].features[key].shape) == 1

    def get_data_dim(self, key):
        """
        :param str key: e.g. "data" or "classes"
        :rtype: int
        :return: number of classes, no matter if sparse or not
        """
        if key in self.num_outputs:
            d = self.num_outputs[key][0]
            if self.added_data and not self.is_data_sparse(key):
                assert self.added_data[0].get_data(key).shape[1] == d
            return d
        self._load_something()
        if len(self.added_data[0].get_data(key).shape) == 1:
            return super(CachedDataset2, self).get_data_dim(key)  # unknown
        assert len(self.added_data[0].get_data(key).shape) == 2
        return self.added_data[0].get_data(key).shape[1]

    def get_data_dtype(self, key):
        """
        :param str key:
        :rtype: str
        """
        self._load_something()
        return str(self.added_data[0].get_data(key).dtype)


class SingleStreamPipeDataset(CachedDataset2):
    """
    Producer: Gets data from somewhere / an external source, running in some thread.
    Consumer: The thread / code which calls load_seqs and get_data here.
    """

    def __init__(self, dim, ndim, sparse=False, dtype="float32"):
        """
        :param int dim:
        :param int ndim:
        :param bool sparse:
        :param str dtype:
        """
        super(SingleStreamPipeDataset, self).__init__()
        self.num_inputs = dim
        self.num_outputs = {"data": [dim, ndim]}
        self.sparse = sparse
        self.dtype = dtype
        self.condition = Condition()
        self.producer_seq_idx = 0
        self.producer_data = []
        self.producer_finished = False

    def is_data_sparse(self, key):
        """
        :param str key:
        :rtype: bool
        """
        return self.sparse

    def get_data_dtype(self, key):
        """
        :param str key:
        :rtype: str
        """
        return self.dtype

    def init_seq_order(self, epoch=None, seq_list=None, seq_order=None):
        """
        :param int epoch:
        :param list[str]|None seq_list:
        :param list[int]|None seq_order:
        :rtype: bool
        """
        assert not seq_list and not seq_order
        super(SingleStreamPipeDataset, self).init_seq_order(epoch=epoch)
        with self.condition:
            self.producer_seq_idx = 0
            self.producer_data.clear()
            self.producer_finished = False
        return True

    def producer_add_data(self, data, seq_tag=None):
        """
        :param numpy.ndarray data:
        :param str|None seq_tag:
        """
        with self.condition:
            if seq_tag is None:
                seq_tag = "seq-%i" % self.producer_seq_idx
            seq = DatasetSeq(features=data, seq_idx=self.producer_seq_idx, seq_tag=seq_tag)
            self.producer_seq_idx += 1
            self.producer_data.append(seq)
            self.condition.notify()

    def producer_set_finished(self):
        """
        Mark finished.
        """
        with self.condition:
            self.producer_finished = True
            self.condition.notify()

    def _collect_single_seq(self, seq_idx):
        """
        :type seq_idx: int
        :rtype: DatasetSeq | None
        :returns DatasetSeq or None if seq_idx >= num_seqs.
        """
        with self.condition:
            while True:
                if self.producer_data:
                    seq = self.producer_data.pop(0)
                    assert isinstance(seq, DatasetSeq)
                    assert seq.seq_idx == seq_idx
                    return seq
                if self.producer_finished:
                    return None
                self.condition.wait()
