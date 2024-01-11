"""
Data iterator on a data loader which an additional queue
"""

from __future__ import annotations

from typing import Union, TypeVar, Iterator, Iterable, Sequence

T = TypeVar("T")


class QueuedDataIter:
    """
    Wraps another iterator, and also has a queue which allows to put items back in.
    """

    def __init__(self, original_iter: Union[Iterable[T], Iterator[T]]):
        self._iter = iter(original_iter)
        self._queue = []

    def extend(self, items: Sequence[T]):
        """
        Extend the queue. The next items by :func:`__next__` will be the ones from ``items``, in the same order.
        """
        # reversed because we pop from behind.
        self._queue.extend(reversed(items))

    def __iter__(self):
        return self

    def __next__(self) -> T:
        if self._queue:
            return self._queue.pop(-1)
        return next(self._iter)
