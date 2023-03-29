"""
Multi-processing dataset
"""

from __future__ import annotations
from typing import Optional, Any, Dict, List
from .basic import init_dataset, DatasetSeq
from .cached2 import CachedDataset2
import multiprocessing as mp

# noinspection PyProtectedMember
from multiprocessing.connection import Connection as mpConnection

_mp = mp.get_context("spawn")


class MultiProcDataset(CachedDataset2):
    """
    Dataset which uses multi-processing to load the data from another dataset.

    To get deterministic behavior, it will use round-robin scheduling.
    """

    def __init__(self, dataset: Dict[str, Any], num_workers: int, buffer_size: int, **kwargs):
        """
        :param dataset: the dataset to use
        :param num_workers: number of workers to use
        :param buffer_size: buffer size for each worker, amount of seqs to prefetch
        """
        super().__init__(**kwargs)
        assert num_workers > 0 and buffer_size > 0
        dataset = dataset.copy()
        for k, v in kwargs.items():
            if k not in dataset:
                dataset[k] = v
        self.dataset = dataset
        self.num_workers = num_workers
        self.buffer_size = buffer_size
        self._data_keys = None
        self._num_seqs = None
        self._total_num_seqs = None

        self._worker_parent_conns = None  # type: Optional[List[mpConnection]]
        self._seq_order_proc_parent_conn = None  # type: Optional[mpConnection]
        self._seq_order_proc = None  # type: Optional[mp.Process]
        self._worker_procs = None  # type: Optional[List[mp.Process]]

    def initialize(self):
        """init"""
        if not self._worker_procs:
            # Seq order proc directly sends the seq order to each worker.
            seq_order_to_worker = []  # type: List[mpConnection]
            worker_from_seq_order = []  # type: List[mpConnection]
            for i in range(self.num_workers):
                reader, writer = _mp.Pipe(duplex=False)
                seq_order_to_worker.append(writer)
                worker_from_seq_order.append(reader)

            worker_parent_conns = []  # type: List[mpConnection]
            worker_child_conns = []  # type: List[mpConnection]
            for i in range(self.num_workers):
                parent_conn, child_conn = _mp.Pipe()
                worker_parent_conns.append(parent_conn)
                worker_child_conns.append(child_conn)

            seq_order_proc_parent_conn, seq_order_proc_child_conn = _mp.Pipe()
            seq_order_proc = _mp.Process(
                name=f"{self.name} seq order proc",
                target=self._seq_order_proc_loop,
                args=(self.dataset, seq_order_proc_child_conn, seq_order_to_worker),
            )
            seq_order_proc.start()

            worker_procs = []
            for i in range(self.num_workers):
                worker_proc = _mp.Process(
                    name=f"{self.name} worker proc {i + 1}/{self.num_workers}",
                    target=self._worker_proc_loop,
                    args=(self.dataset, self.buffer_size, worker_child_conns[i], worker_from_seq_order[i]),
                )
                worker_proc.start()
                worker_procs.append(worker_proc)

            self._seq_order_proc_parent_conn = seq_order_proc_parent_conn  # type: mpConnection
            self._seq_order_proc = seq_order_proc
            self._worker_parent_conns = worker_parent_conns
            self._worker_procs = worker_procs

            self._seq_order_proc_parent_conn.send(("init", {}))
            msg, self.num_inputs = self._seq_order_proc_parent_conn.recv()
            assert msg == "num_inputs"
            msg, self.num_outputs = self._seq_order_proc_parent_conn.recv()
            assert msg == "num_outputs"
            msg, self._total_num_seqs = self._seq_order_proc_parent_conn.recv()
            assert msg == "total_num_seqs"
            msg, self.labels = self._seq_order_proc_parent_conn.recv()
            assert msg == "labels"

        super().initialize()

    def __del__(self):
        if self._seq_order_proc:
            self._seq_order_proc_parent_conn.send(("exit", {}))
            self._seq_order_proc.join()
        if self._worker_procs:
            for worker_parent_conn in self._worker_parent_conns:
                worker_parent_conn.send(("exit", {}))
            for worker_proc in self._worker_procs:
                worker_proc.join()

    @staticmethod
    def _seq_order_proc_loop(dataset_dict: Dict[str, Any], parent: mpConnection, workers: List[mpConnection]):
        num_workers = len(workers)
        dataset = init_dataset(dataset_dict)
        try:
            while True:
                msg, kwargs = parent.recv()
                if msg == "exit":
                    break
                elif msg == "init":
                    parent.send(("num_inputs", dataset.num_inputs))
                    parent.send(("num_outputs", dataset.num_outputs))
                    try:
                        total_num_seqs = dataset.get_total_num_seqs()
                    except NotImplementedError:
                        total_num_seqs = None
                    parent.send(("total_num_seqs", total_num_seqs))
                    parent.send(("labels", dataset.labels))
                elif msg == "init_seq_order":
                    dataset.init_seq_order(**kwargs)
                    seq_order = dataset.get_current_seq_order()
                    for i, worker in enumerate(workers):
                        worker.send(("seq_order_shard", seq_order[i::num_workers]))
                    parent.send(("num_seqs", len(seq_order)))
                else:
                    raise Exception(f"unknown msg {msg!r}")
        except KeyboardInterrupt:  # when parent dies
            pass

    @staticmethod
    def _worker_proc_loop(
        dataset_dict: Dict[str, Any], buffer_size: int, parent: mpConnection, seq_order: mpConnection
    ):
        dataset = init_dataset(dataset_dict)

        got_init_seq_order = False
        cache = []  # type: List[DatasetSeq]
        next_seq_idx = 0

        # noinspection PyShadowingNames
        def _add_to_cache():
            nonlocal next_seq_idx
            if len(cache) >= buffer_size:
                return False
            if not dataset.is_less_than_num_seqs(next_seq_idx):
                return False
            seq_tag = dataset.get_tag(next_seq_idx)
            features = {data_key: dataset.get_data(next_seq_idx, data_key) for data_key in dataset.get_data_keys()}
            res = DatasetSeq(seq_idx=next_seq_idx, seq_tag=seq_tag, features=features)
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
                    while not parent.poll():
                        if not _add_to_cache():
                            break
                msg, kwargs = parent.recv()
                if msg == "exit":
                    break
                elif msg == "get_data_seq":
                    seq_idx = kwargs["seq_idx"]
                    while cache and cache[0].seq_idx < seq_idx:
                        cache.pop(0)
                    res = _get(seq_idx)
                    parent.send(("data_seq", res))
                elif msg == "init_seq_order":
                    msg_, seq_order_ = seq_order.recv()
                    assert msg_ == "seq_order_shard"
                    dataset.init_seq_order(seq_order=seq_order_, **kwargs)
                    got_init_seq_order = True
                    next_seq_idx = 0
                    cache[:] = []
                else:
                    raise Exception(f"unknown msg {msg!r}")
        except KeyboardInterrupt:  # when parent dies
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

        if epoch is not None or seq_list is not None or seq_order is not None:
            self._seq_order_proc_parent_conn.send(
                ("init_seq_order", {"epoch": epoch, "seq_list": seq_list, "seq_order": seq_order})
            )
            for i in range(self.num_workers):
                self._worker_parent_conns[i].send(("init_seq_order", {"epoch": epoch}))
            msg, num_seqs = self._seq_order_proc_parent_conn.recv()
            assert msg == "num_seqs"
            self._num_seqs = num_seqs
        else:
            self._num_seqs = 0

        return True

    def _collect_single_seq(self, seq_idx: int) -> Optional[DatasetSeq]:
        if seq_idx >= self._num_seqs:
            return None
        worker_idx = seq_idx % self.num_workers
        worker = self._worker_parent_conns[worker_idx]
        worker.send(("get_data_seq", {"seq_idx": seq_idx // self.num_workers}))
        msg, data = worker.recv()
        assert msg == "data_seq"
        if data is None:
            return None
        assert isinstance(data, DatasetSeq)
        data.seq_idx = seq_idx
        return data

    @property
    def num_seqs(self) -> int:
        """num seqs"""
        return self._num_seqs

    def get_total_num_seqs(self) -> int:
        """total num seqs"""
        if self._total_num_seqs is not None:
            return self._total_num_seqs
        raise NotImplementedError
