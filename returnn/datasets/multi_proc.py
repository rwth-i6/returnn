"""
Multi-processing dataset
"""

from __future__ import annotations
from collections import deque
from typing import Optional, Any, Dict, List
import sys
import gc
import multiprocessing as mp
from returnn.util import better_exchook
from returnn.util.basic import try_run
from returnn.config import SubProcCopyGlobalConfigPreInitFunc
from returnn.util.multi_proc_non_daemonic_spawn import NonDaemonicSpawnContext
from .basic import init_dataset, extend_dataset_dict_from_parent_dataset, Dataset, DatasetSeq
from .cached2 import CachedDataset2

# noinspection PyProtectedMember
from multiprocessing.connection import Connection as mpConnection


class MultiProcDataset(CachedDataset2):
    """
    Dataset which uses multi-processing to load the data from another dataset.

    To get deterministic behavior, it will use round-robin scheduling.

    There is one process just for generating the sequence order, i.e. list of sequences.
    Then there are ``num_workers`` processes which will load the data for the shard of the sequences.
    This means, one epoch (or subepoch) is exactly as in the original dataset.
    """

    def __init__(
        self,
        dataset: Dict[str, Any],
        num_workers: int,
        buffer_size: int,
        sharding_method: str = "seq_order",
        _meta_info_cache: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        :param dataset: the dataset to use
        :param num_workers: number of workers to use
        :param buffer_size: buffer size for each worker, amount of seqs to prefetch
        :param sharding_method: which method to use for sharding the data across the worker procs.
            The default is ``seq_order``, which fetches the full list of seq indices,
            and then distributes shards of that to the other workers.
            Can also be set to ``dedicated`` to enable a worker-index based sharding method.
            This is compatible with more types of datasets, in particular those
            that do not know their total number of segments upfront.
        :param _meta_info_cache: for internal use
        """
        super().__init__(**kwargs)
        assert num_workers > 0 and buffer_size > 0
        dataset = dataset.copy()
        for k, v in kwargs.items():
            dataset.setdefault(k, v)
        dataset = extend_dataset_dict_from_parent_dataset(dataset, parent_dataset=self)
        self.dataset = dataset
        self.num_workers = num_workers
        self.buffer_size = buffer_size
        allowed_sharding_methods = ["seq_order", "dedicated"]
        if sharding_method not in allowed_sharding_methods:
            raise ValueError(
                f"invalid sharding_method '{sharding_method}', must be {' or '.join(allowed_sharding_methods)}"
            )
        self._sharding_method = sharding_method
        self._data_keys = None
        self._data_dtypes: Optional[Dict[str, str]] = None
        self._num_seqs = None
        self._total_num_seqs = None
        self._all_tags = None

        self._worker_parent_conns = None  # type: Optional[List[mpConnection]]
        self._seq_order_proc_parent_conn = None  # type: Optional[mpConnection]
        self._seq_order_proc = None  # type: Optional[mp.Process]
        self._worker_procs = None  # type: Optional[List[mp.Process]]
        self._cur_max_complete_frac: Optional[float] = None

        if _meta_info_cache:
            # This allows to skip the lazy init in self.initialize().
            # This can be used when pickling/deepcopying an instance of this dataset,
            # see Dataset.__reduce__.
            self.num_inputs = _meta_info_cache["num_inputs"]
            self.num_outputs = _meta_info_cache["num_outputs"]
            self._total_num_seqs = _meta_info_cache["total_num_seqs"]
            self.labels = _meta_info_cache["labels"]
            self._data_keys = _meta_info_cache["data_keys"]
            self._data_dtypes = _meta_info_cache["data_dtypes"]

    def initialize(self):
        """init"""
        if not self.num_outputs:
            self._lazy_init()
        super().initialize()

    @property
    def _meta_info_cache(self):
        if not self.num_outputs:
            return None
        return {
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "labels": self.labels,
            "total_num_seqs": self._total_num_seqs,
            "all_tags": self._all_tags,
            "data_keys": self._data_keys,
            "data_dtypes": self._data_dtypes,
        }

    def _lazy_init(self):
        if self._worker_procs:
            return

        _mp = NonDaemonicSpawnContext(process_pre_init_func=SubProcCopyGlobalConfigPreInitFunc())

        seq_order_to_worker = []  # type: List[mpConnection]
        worker_from_seq_order = []  # type: List[mpConnection]
        if self._sharding_method == "seq_order":
            # Seq order proc (first worker) directly sends the seq order to each (other) worker.
            for i in range(self.num_workers - 1):
                reader, writer = _mp.Pipe(duplex=False)
                seq_order_to_worker.append(writer)
                worker_from_seq_order.append(reader)

        worker_parent_conns = []  # type: List[mpConnection]
        worker_child_conns = []  # type: List[mpConnection]
        for i in range(self.num_workers):
            parent_conn, child_conn = _mp.Pipe()
            worker_parent_conns.append(parent_conn)
            worker_child_conns.append(child_conn)

        worker_procs = []
        for i in range(self.num_workers):
            if self._sharding_method == "seq_order":
                sub_dataset = self.dataset
                args = (
                    i,
                    sub_dataset,
                    self.buffer_size,
                    worker_child_conns[i],
                    worker_from_seq_order[i - 1] if i > 0 else None,
                    seq_order_to_worker if i == 0 else None,
                    self._sharding_method,
                )
            elif self._sharding_method == "dedicated":
                sub_dataset = {**self.dataset, "_num_shards": self.num_workers, "_shard_index": i}
                args = (
                    i,
                    sub_dataset,
                    self.buffer_size,
                    worker_child_conns[i],
                    None,
                    None,
                    self._sharding_method,
                )
            else:
                raise ValueError(f"{self}: unknown sharding_method: {self._sharding_method}")
            worker_proc = _mp.Process(
                name=f"{self.name} worker proc {i + 1}/{self.num_workers}",
                target=self._worker_proc_loop,
                args=args,
                daemon=True,
            )
            worker_proc.start()
            worker_procs.append(worker_proc)
            # Make sure the child connection is closed here.
            # It stays open in the child, until the child dies.
            # When that happens, now any consecutive read on the pipe
            # should yield an exception -- which is what we want,
            # otherwise it would just hang.
            worker_child_conns[i].close()

        self._seq_order_proc_parent_conn = worker_parent_conns[0]  # type: mpConnection
        self._worker_parent_conns = worker_parent_conns
        self._worker_procs = worker_procs

        self._seq_order_proc_parent_conn.send(("init", {}))
        msg, self.num_inputs = self._seq_order_proc_parent_conn.recv()
        assert msg == "num_inputs"
        msg, self.num_outputs = self._seq_order_proc_parent_conn.recv()
        assert msg == "num_outputs"
        msg, self.labels = self._seq_order_proc_parent_conn.recv()
        assert msg == "labels"

    def _lazy_init_data_keys(self):
        if self._data_keys is not None:
            return

        self._lazy_init()  # indempotent, will not run twice if not needed

        self._seq_order_proc_parent_conn.send(("get_data_keys", {}))
        msg, self._data_keys = self._seq_order_proc_parent_conn.recv()
        assert msg == "data_keys"

    def _lazy_init_data_dtypes(self):
        if self._data_dtypes is not None:
            return

        self._lazy_init()  # indempotent, will not run twice if not needed

        self._seq_order_proc_parent_conn.send(("get_data_dtypes", {}))
        msg, self._data_dtypes = self._seq_order_proc_parent_conn.recv()
        assert msg == "data_dtypes"

    def __del__(self):
        if self._worker_procs:
            got_exception = False
            for worker_parent_conn in self._worker_parent_conns:
                # noinspection PyBroadException
                try:
                    worker_parent_conn.send(("exit", {}))
                except Exception:
                    got_exception = True
            if not got_exception:
                for worker_proc in self._worker_procs:
                    try_run(worker_proc.join)

    @staticmethod
    def _worker_proc_loop(
        worker_index: int,
        dataset_dict: Dict[str, Any],
        buffer_size: int,
        parent_conn: mpConnection,
        seq_order_conn: Optional[mpConnection],
        other_worker_conns: Optional[List[mpConnection]],
        sharding_method: str,
    ):
        if sys.platform == "linux":
            with open("/proc/self/comm", "w") as f:
                f.write(f"MPD worker {worker_index}")
        better_exchook.setup_all()

        dataset: Optional[Dataset] = None

        got_init_seq_order = False
        cache: deque[DatasetSeq] = deque()
        next_seq_idx = 0

        # noinspection PyShadowingNames
        def _add_to_cache():
            nonlocal next_seq_idx
            if len(cache) >= buffer_size:
                return False
            if not dataset.is_less_than_num_seqs(next_seq_idx):
                return False
            dataset.load_seqs(next_seq_idx, next_seq_idx + 1)
            seq_tag = dataset.get_tag(next_seq_idx)
            features = {data_key: dataset.get_data(next_seq_idx, data_key) for data_key in dataset.get_data_keys()}
            complete_frac = dataset.get_complete_frac(next_seq_idx, allow_only_lr_suitable=True)
            res = DatasetSeq(seq_idx=next_seq_idx, seq_tag=seq_tag, features=features, complete_frac=complete_frac)
            cache.append(res)
            next_seq_idx += 1
            return True

        # noinspection PyShadowingNames
        def _get_from_cache(seq_idx: int) -> Optional[DatasetSeq]:
            if not cache:
                return None
            if seq_idx > cache[-1].seq_idx:
                return None
            for seq in cache:
                assert seq.seq_idx <= seq_idx
                if seq.seq_idx == seq_idx:
                    return seq
            assert False

        # noinspection PyShadowingNames
        def _get(seq_idx: int) -> Optional[DatasetSeq]:
            if cache and seq_idx < cache[0].seq_idx:
                raise Exception(
                    f"requested seq idx {seq_idx} is smaller than cache start {cache[0].seq_idx}, cannot go backwards"
                )
            res = _get_from_cache(seq_idx)
            if res:
                return res
            if not dataset.is_less_than_num_seqs(seq_idx):
                return None
            assert next_seq_idx <= seq_idx
            while True:
                if not _add_to_cache():
                    raise Exception(
                        f"buffer too small, requested seq idx {seq_idx},"
                        f" cache starts at {cache[0].seq_idx if cache else None}"
                    )
                assert cache[-1].seq_idx <= seq_idx
                if cache[-1].seq_idx == seq_idx:
                    return cache[-1]

        try:
            while True:
                if got_init_seq_order:
                    while not parent_conn.poll():
                        if not _add_to_cache():
                            break
                msg, kwargs = parent_conn.recv()
                if msg == "exit":
                    break
                elif msg == "get_data_seq":
                    seq_idx = kwargs["seq_idx"]
                    while cache and cache[0].seq_idx < seq_idx:
                        cache.popleft()
                    res = _get(seq_idx)
                    parent_conn.send(("data_seq", res))
                elif msg == "init":
                    assert worker_index == 0
                    if dataset is None:
                        dataset = init_dataset(dataset_dict)
                    parent_conn.send(("num_inputs", dataset.num_inputs))
                    parent_conn.send(("num_outputs", dataset.num_outputs))
                    parent_conn.send(("labels", dataset.labels))
                elif msg == "get_total_num_seqs":
                    assert dataset
                    try:
                        total_num_seqs = dataset.get_total_num_seqs()
                        assert isinstance(total_num_seqs, int)
                    except NotImplementedError as exc:
                        total_num_seqs = NotImplementedError(f"{exc} in {dataset}")
                    parent_conn.send(("total_num_seqs", total_num_seqs))
                elif msg == "get_all_tags":
                    assert dataset
                    try:
                        all_tags = dataset.get_all_tags()
                        assert isinstance(all_tags, list)
                    except NotImplementedError as exc:
                        all_tags = NotImplementedError(f"{exc} in {dataset}")
                    parent_conn.send(("all_tags", all_tags))
                elif msg == "get_data_keys":
                    assert dataset
                    data_keys = dataset.get_data_keys()
                    parent_conn.send(("data_keys", data_keys))
                elif msg == "get_data_dtypes":
                    assert dataset
                    data_keys = dataset.get_data_keys()
                    parent_conn.send(("data_dtypes", {k: dataset.get_data_dtype(k) for k in data_keys}))
                elif msg == "init_seq_order":
                    if dataset is None:
                        dataset = init_dataset(dataset_dict)
                    if sharding_method == "dedicated":
                        dataset.init_seq_order(**kwargs)
                        try:
                            num_seqs = dataset.num_seqs
                        except NotImplementedError:
                            num_seqs = None
                        parent_conn.send(("num_seqs", num_seqs))
                    elif sharding_method == "seq_order":
                        if worker_index == 0:
                            # We are responsible to get the seq order and distrib it to all the other workers.
                            assert other_worker_conns is not None
                            dataset.init_seq_order(**kwargs)
                            try:
                                seq_order = dataset.get_current_seq_order()
                            except Exception as exc:
                                raise Exception(
                                    f"{MultiProcDataset.__name__}: `get_current_seq_order()` failed on {dataset}. "
                                    f'Consider trying {MultiProcDataset.__name__}\'s "sharding_method": "dedicated", '
                                    "which uses a different method for distributing the segments across workers."
                                ) from exc
                            for i, worker_conn in enumerate(other_worker_conns):
                                worker_conn.send(("seq_order_shard", seq_order[i + 1 :: len(other_worker_conns) + 1]))
                            parent_conn.send(("num_seqs", len(seq_order)))
                            # Now reset seq order for ourself (as the role of a normal worker).
                            kwargs["seq_order"] = seq_order[0 :: len(other_worker_conns) + 1]
                            kwargs.pop("seq_list", None)
                            dataset.init_seq_order(**kwargs)
                        else:
                            assert seq_order_conn is not None
                            msg_, seq_order = seq_order_conn.recv()
                            assert msg_ == "seq_order_shard"
                            dataset.init_seq_order(seq_order=seq_order, **kwargs)
                    else:
                        raise ValueError(f"{MultiProcDataset.__name__}: unknown sharding_method: {sharding_method}")
                    got_init_seq_order = True
                    next_seq_idx = 0
                    cache.clear()
                elif msg == "finish_epoch":
                    got_init_seq_order = False
                    next_seq_idx = 0
                    cache.clear()
                    if dataset:
                        dataset.finish_epoch(**kwargs)
                    if kwargs["free_resources"]:
                        dataset = None
                        gc.collect()
                else:
                    raise Exception(f"unknown msg {msg!r}")
        except KeyboardInterrupt:  # when parent dies
            pass
        except EOFError:  # when parent dies
            pass

    def init_seq_order(self, epoch=None, seq_list=None, seq_order=None):
        """
        :type epoch: int|None
        :param list[str]|None seq_list: List of sequence tags, to set a predefined order.
        :param list[int]|None seq_order: List of corpus sequence indices, to set a predefined order. Only possible
          if the dataset has such indices (see self.have_corpus_seq_idx()).
        :rtype: bool
        :returns whether the order changed (True is always safe to return)
        """
        super().init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)

        if epoch is None and seq_list is None and seq_order is None:
            self._num_seqs = 0
            return True

        self._lazy_init()
        self._cur_max_complete_frac = 0.0

        if self._sharding_method == "dedicated":
            for worker_conn in self._worker_parent_conns:
                worker_conn.send(("init_seq_order", {"epoch": epoch, "seq_list": seq_list, "seq_order": seq_order}))
            num_child_seqs = []
            for worker_conn in self._worker_parent_conns:
                msg, num_seqs = worker_conn.recv()
                assert msg == "num_seqs"
                num_child_seqs.append(num_seqs)
            if all(num_s is None for num_s in num_child_seqs):
                self._num_seqs = None
            elif all(num_s is not None for num_s in num_child_seqs):
                self._num_seqs = sum(num_child_seqs, 0)
            else:
                raise ValueError(f"heterogenous num_seqs in child datasets: {num_child_seqs}")
        elif self._sharding_method == "seq_order":
            self._seq_order_proc_parent_conn.send(
                ("init_seq_order", {"epoch": epoch, "seq_list": seq_list, "seq_order": seq_order})
            )
            for worker_conn in self._worker_parent_conns[1:]:
                worker_conn.send(("init_seq_order", {"epoch": epoch}))
            msg, num_seqs = self._seq_order_proc_parent_conn.recv()
            assert msg == "num_seqs"
            self._num_seqs = num_seqs
        else:
            raise ValueError(f"{self}: unknown sharding_method: {self._sharding_method}")

        return True

    def _collect_single_seq(self, seq_idx: int) -> Optional[DatasetSeq]:
        worker_idx = seq_idx % self.num_workers
        worker = self._worker_parent_conns[worker_idx]
        worker.send(("get_data_seq", {"seq_idx": seq_idx // self.num_workers}))
        msg, data = worker.recv()
        assert msg == "data_seq"
        if data is None:
            return None
        assert isinstance(data, DatasetSeq)
        # The complete_frac values from the subprocesses are not necessarily monotonic
        # due to rounding errors in the sharding and such.
        # We therefore fix them up here. This is valid due to monotonicity of `seq_idx`.
        max_comp_frac = max(data.complete_frac, self._cur_max_complete_frac)
        data.complete_frac = max_comp_frac
        self._cur_max_complete_frac = max_comp_frac
        data.seq_idx = seq_idx
        return data

    def get_total_num_seqs(self, *, fast: bool = False) -> int:
        """total num seqs"""
        if self._total_num_seqs is None:
            worker = self._seq_order_proc_parent_conn
            worker.send(("get_total_num_seqs", {}))
            msg, self._total_num_seqs = worker.recv()
            assert msg == "total_num_seqs" and self._total_num_seqs is not None
        if isinstance(self._total_num_seqs, int):
            return self._total_num_seqs
        elif isinstance(self._total_num_seqs, Exception):
            raise self._total_num_seqs
        else:
            raise TypeError(f"invalid type {type(self._total_num_seqs)} for total_num_seqs")

    def get_all_tags(self):
        """all tags"""
        if self._all_tags is None:
            worker = self._seq_order_proc_parent_conn
            worker.send(("get_all_tags", {}))
            msg, self._all_tags = worker.recv()
            assert msg == "all_tags" and self._all_tags is not None
        if isinstance(self._all_tags, list):
            return self._all_tags
        elif isinstance(self._all_tags, Exception):
            raise self._all_tags
        else:
            raise TypeError(f"invalid type {type(self._all_tags)} for all_tags")

    def finish_epoch(self, *, free_resources: bool = False):
        """finish epoch"""
        super().finish_epoch(free_resources=free_resources)
        for worker_parent_conn in self._worker_parent_conns:
            worker_parent_conn.send(("finish_epoch", {"free_resources": free_resources}))

    def get_data_keys(self) -> List[str]:
        """data keys"""
        if self._data_keys is None:
            self._lazy_init_data_keys()
            assert self._data_keys is not None
        return self._data_keys

    def get_data_dtype(self, key: str) -> str:
        """:return: dtype of `key`"""
        if self._data_dtypes is None:
            self._lazy_init_data_dtypes()
            assert self._data_dtypes is not None
        return self._data_dtypes[key]

    def is_data_sparse(self, key: str) -> bool:
        """:return: whether `key` is sparse"""
        if self.num_outputs is None:
            self._lazy_init()
            assert self.num_outputs is not None
        return super().is_data_sparse(key)
