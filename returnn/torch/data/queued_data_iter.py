"""
Data iterator on a data loader which an additional queue
"""

from __future__ import annotations

import gc
import time
from typing import Union, TypeVar, Iterator, Iterable, Sequence
import torch

from returnn.util import basic as util

T = TypeVar("T")


class QueuedDataIter:
    """
    Wraps another iterator, and also has a queue which allows to put items back in.
    """

    def __init__(self, original_iter: Union[Iterable[T], Iterator[T]]):
        self._iter = iter(original_iter)
        self._trigger_gc = False
        self._queue = []

    def extend(self, items: Sequence[T]):
        """
        Extend the queue. The next items by :func:`__next__` will be the ones from ``items``, in the same order.
        """
        # reversed because we pop from behind.
        self._queue.extend(reversed(items))

    def setup_gc_trigger(self):
        """
        When called, then the next ``next(self)`` call will also do garbage collection (gc),
        with the intention to free all CUDA memory.

        Also see:
        https://github.com/pytorch/pytorch/issues/18853
        https://github.com/pytorch/pytorch/issues/27600
        https://pytorch.org/docs/stable/notes/faq.html#my-out-of-memory-exception-handler-can-t-allocate-memory
        https://github.com/Lightning-AI/pytorch-lightning/blob/7a4b0fc4331633cdf00b88776689e8a84ef96cb4/src/lightning/pytorch/utilities/memory.py#L83
        """
        self._trigger_gc = True

    def __iter__(self):
        return self

    def __next__(self) -> T:
        if self._trigger_gc:
            _garbage_collect()
            self._trigger_gc = False
        if self._queue:
            return self._queue.pop(-1)
        return next(self._iter)


def _garbage_collect():
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        stats = [
            f"alloc {util.human_bytes_size(torch.cuda.memory_allocated())}",
            f"reserved {util.human_bytes_size(torch.cuda.memory_reserved())}",
        ]
        print(f"CUDA memory usage after triggered GC:", " ".join(stats))
