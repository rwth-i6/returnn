"""
Main loop of a worker proc which loads the seqs of a dataset for the parent proc,
as used by :class:`MultiProcDataset` and :class:`DistributeFilesDataset`.
"""

from __future__ import annotations
from typing import Optional, Any, Callable, Dict, Tuple
from collections import deque
import threading
from returnn.datasets.basic import Dataset, DatasetSeq

# noinspection PyProtectedMember
from multiprocessing.connection import Connection as mpConnection

__all__ = ["WorkerProcLoop", "get_data_seq"]


class WorkerProcLoop:
    """
    The main thread owns the dataset: it handles the msgs from the parent and prefetches up to ``buffer_size`` seqs.
    A receiving thread replies to ``get_data_seq`` directly when the seq is cached
    and no other msg is pending or being handled by the main thread,
    and passes on all other msgs to the main thread.
    """

    def __init__(self, *, name: str, parent_conn: mpConnection, buffer_size: int, dataset: Optional[Dataset]):
        """
        :param name: of the worker, for the receiving thread
        :param parent_conn: msgs from and replies to the parent
        :param buffer_size: max number of cached seqs
        :param dataset: can also be set later, but only by the main thread
        """
        self.name = name
        self.parent_conn = parent_conn
        self.buffer_size = buffer_size
        self.dataset = dataset
        self._got_init_seq_order = False
        self._got_seq_end = False
        self._cache: deque[DatasetSeq] = deque()
        self._next_seq_idx = 0
        self._cond = threading.Condition()  # guards _cache, _msg_queue, _num_unhandled_msgs
        self._msg_queue: deque[Tuple[str, Dict[str, Any]]] = deque()
        self._num_unhandled_msgs = 0  # in _msg_queue or being handled by the main thread

    def on_init_seq_order(self):
        """
        Call after ``dataset.init_seq_order``. Prefetches from seq 0 on.
        """
        self._got_init_seq_order = True
        self._got_seq_end = False
        self._next_seq_idx = 0
        self._cache.clear()

    def on_finish_epoch(self):
        """
        Stops prefetching until the next :func:`on_init_seq_order`.
        """
        self._got_init_seq_order = False
        self._next_seq_idx = 0
        self._cache.clear()

    def run(self, handle_msg: Callable[[str, Dict[str, Any]], None]):
        """
        Runs until the ``exit`` msg or until the parent dies.

        :param handle_msg: called in the main thread for all msgs except ``get_data_seq`` and ``exit``.
            It replies via ``parent_conn``.
        """
        threading.Thread(target=self._serve_parent, name=f"{self.name} serve parent", daemon=True).start()

        try:
            while True:
                with self._cond:
                    while not self._msg_queue and not (
                        self._got_init_seq_order and not self._got_seq_end and len(self._cache) < self.buffer_size
                    ):
                        self._cond.wait()
                    msg, kwargs = self._msg_queue.popleft() if self._msg_queue else (None, None)
                if msg is None:
                    if not self._add_to_cache():
                        self._got_seq_end = True
                    continue
                if msg == "exit":
                    break
                elif msg == "get_data_seq":
                    seq_idx = kwargs["seq_idx"]
                    while self._cache and self._cache[0].seq_idx < seq_idx:
                        self._cache.popleft()
                    res = self._get(seq_idx)
                    self.parent_conn.send(("data_seq", res))
                else:
                    handle_msg(msg, kwargs)
                with self._cond:
                    self._num_unhandled_msgs -= 1
        except KeyboardInterrupt:  # when parent dies
            pass
        except EOFError:  # when parent dies
            pass

    def _add_to_cache(self) -> bool:
        if len(self._cache) >= self.buffer_size:
            return False
        if not self.dataset.is_less_than_num_seqs(self._next_seq_idx):
            return False
        res = get_data_seq(self.dataset, self._next_seq_idx)
        with self._cond:
            self._cache.append(res)
        self._next_seq_idx += 1
        return True

    def _get_from_cache(self, seq_idx: int) -> Optional[DatasetSeq]:
        if not self._cache:
            return None
        if seq_idx > self._cache[-1].seq_idx:
            return None
        for seq in self._cache:
            assert seq.seq_idx <= seq_idx
            if seq.seq_idx == seq_idx:
                return seq
        assert False

    def _get(self, seq_idx: int) -> Optional[DatasetSeq]:
        if self._cache and seq_idx < self._cache[0].seq_idx:
            raise Exception(
                f"requested seq idx {seq_idx} is smaller than cache start {self._cache[0].seq_idx}, cannot go backwards"
            )
        res = self._get_from_cache(seq_idx)
        if res:
            return res
        if not self.dataset.is_less_than_num_seqs(seq_idx):
            return None
        assert self._next_seq_idx <= seq_idx
        while True:
            if not self._add_to_cache():
                raise Exception(
                    f"buffer too small, requested seq idx {seq_idx},"
                    f" cache starts at {self._cache[0].seq_idx if self._cache else None}"
                )
            assert self._cache[-1].seq_idx <= seq_idx
            if self._cache[-1].seq_idx == seq_idx:
                return self._cache[-1]

    def _serve_parent(self):
        try:
            while True:
                parent_msg, parent_kwargs = self.parent_conn.recv()
                if parent_msg == "exit":
                    break
                cached_seq = None
                with self._cond:
                    if parent_msg == "get_data_seq" and self._num_unhandled_msgs == 0:
                        requested_seq_idx = parent_kwargs["seq_idx"]
                        if self._cache and self._cache[0].seq_idx <= requested_seq_idx:
                            cached_seq = self._get_from_cache(requested_seq_idx)
                        if cached_seq:
                            while self._cache[0].seq_idx < requested_seq_idx:
                                self._cache.popleft()
                            self._cond.notify()
                    if not cached_seq:
                        self._msg_queue.append((parent_msg, parent_kwargs))
                        self._num_unhandled_msgs += 1
                        self._cond.notify()
                if cached_seq:
                    self.parent_conn.send(("data_seq", cached_seq))
        except EOFError:  # when parent dies
            pass
        finally:
            with self._cond:
                self._msg_queue.append(("exit", {}))
                self._cond.notify()


def get_data_seq(dataset: Dataset, seq_idx: int) -> DatasetSeq:
    """
    :param dataset: after ``init_seq_order``
    :param seq_idx: in the current epoch
    :return: the loaded seq with all data keys
    """
    dataset.load_seqs(seq_idx, seq_idx + 1)
    seq_tag = dataset.get_tag(seq_idx)
    features = {data_key: dataset.get_data(seq_idx, data_key) for data_key in dataset.get_data_keys()}
    complete_frac = dataset.get_complete_frac(seq_idx, allow_only_lr_suitable=True)
    return DatasetSeq(seq_idx=seq_idx, seq_tag=seq_tag, features=features, complete_frac=complete_frac)
