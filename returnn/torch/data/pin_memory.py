"""
Pinning the batches of a DataLoader in a RETURNN-owned background thread,
coordinated with the CUDA graph captures.

The DataLoader option ``pin_memory=True`` (with ``num_workers > 0``) pins in its own thread,
concurrently with the training steps.
A pinned host memory allocation during a CUDA graph capture
(the model step, see :mod:`returnn.torch.util.graph_capture`,
or the optimizer step, see :mod:`returnn.torch.util.optimizer_step`)
invalidates the capture (``cudaErrorStreamCaptureInvalidated``).
The DataLoader pin thread cannot be paused (it only has a stop-forever event).

:class:`PinMemoryDataLoader` wraps a DataLoader without ``pin_memory``
and pins the batches in its own thread instead,
holding :data:`returnn.torch.util.capture_lock.capture_lock` while pinning
and while releasing its reference to a handed-over batch,
which every capture holds for the whole capture.
So pinning waits for a capture (and a capture waits for the current pinning),
but runs concurrently with the ordinary steps (graph replays).
"""

from __future__ import annotations
from typing import Optional, Union, Any, Iterable, Iterator, Tuple
import threading
import queue
import weakref
import torch

from returnn.torch.util.capture_lock import capture_lock

__all__ = ["PinMemoryDataLoader", "PinMemoryIter"]


class PinMemoryDataLoader:
    """
    Wraps a DataLoader (created without ``pin_memory``):
    iterating gives the same batches, with all tensors pinned, see :class:`PinMemoryIter`.
    """

    def __init__(self, data_loader: Iterable[Any], *, device: Union[str, torch.device], queue_size: int = 2):
        """
        :param data_loader: e.g. a DataLoader with ``pin_memory=False``
        :param device: CUDA device the batches are copied to
        :param queue_size: number of pinned batches the thread prepares ahead of the consumer
        """
        self.data_loader = data_loader
        device = torch.device(device)
        assert device.type == "cuda", f"{self}: expected a CUDA device, got {device}"
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.queue_size = queue_size
        self._cur_iter: Optional[weakref.ref[PinMemoryIter]] = None

    def __iter__(self) -> PinMemoryIter:
        # A DataLoader with persistent workers reuses its iterator for the next iter().
        # The previous thread must not use it concurrently anymore.
        prev_iter = self._cur_iter() if self._cur_iter is not None else None
        if prev_iter is not None:
            prev_iter.close()
        it = PinMemoryIter(iter(self.data_loader), device=self.device, queue_size=self.queue_size)
        self._cur_iter = weakref.ref(it)  # not keeping it alive: its thread stops when the consumer drops it
        return it


class PinMemoryIter:
    """
    Iterates over the source iterator in a background thread, pins all tensors of each batch
    (while holding :data:`capture_lock`), and yields the pinned batches in the same order.
    Exceptions of the source iterator (or of the pinning) are re-raised in the consumer.
    """

    def __init__(self, src_iter: Iterator[Any], *, device: torch.device, queue_size: int = 2):
        """
        :param src_iter: e.g. a DataLoader iterator. Only used by the background thread from here on.
        :param device: CUDA device, set as the current device of the thread
        :param queue_size: number of pinned batches the thread prepares ahead of the consumer
        """
        assert queue_size >= 1
        self._queue: queue.Queue[Tuple[str, Any]] = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._finished = False
        # The thread gets no reference to self, so self is freed (and closed) when the consumer drops it.
        self._thread = threading.Thread(
            target=_pin_loop,
            name="RETURNN pin memory",
            kwargs=dict(src_iter=src_iter, device=device, out_queue=self._queue, stop_event=self._stop_event),
            daemon=True,
        )
        self._thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        if self._finished:
            raise StopIteration
        while True:
            try:
                kind, value = self._queue.get(timeout=_PollInterval)
                break
            except queue.Empty:
                if not self._thread.is_alive() and self._queue.empty():
                    # should not happen, the thread always sends "end" or "error" before it stops
                    self._finished = True
                    raise RuntimeError(f"{self}: pin memory thread stopped unexpectedly")
        if kind == "batch":
            return value
        self._finished = True
        self._thread.join()
        if kind == "error":
            raise value
        assert kind == "end"
        raise StopIteration

    def close(self):
        """
        Stops the thread and waits for it.
        The thread stops after its current source ``next()`` returned (it does not interrupt the source).
        Pinned batches still in the queue are dropped.
        """
        self._finished = True
        self._stop_event.set()
        while self._thread.is_alive():
            _drain(self._queue)  # unblocks a pending put
            self._thread.join(timeout=_PollInterval)
        _drain(self._queue)

    def __del__(self):
        # also at interpreter shutdown, when the thread might already be gone
        if getattr(self, "_thread", None) is not None:
            self.close()


_PollInterval = 0.1  # seconds, for the stop checks of the blocking queue ops


def _pin_loop(*, src_iter: Iterator[Any], device: torch.device, out_queue: queue.Queue, stop_event: threading.Event):
    # the thread's current device, else pinning would create a CUDA context on the default device
    torch.cuda.set_device(device)
    while not stop_event.is_set():
        try:
            batch = next(src_iter)
        except StopIteration:
            msg = ("end", None)
        except Exception as exc:  # forwarded, re-raised in the consumer
            msg = ("error", exc)
        else:
            try:
                msg = ("batch", _pin_batch(batch))
            except Exception as exc:  # forwarded, re-raised in the consumer
                msg = ("error", exc)
            del batch
        while not stop_event.is_set():
            try:
                out_queue.put(msg, timeout=_PollInterval)
                break
            except queue.Full:
                continue
        kind = msg[0]
        # Once published, the consumer may have already copied from the batch and dropped it,
        # so this can be the last reference: freeing a pinned block used by an async copy records CUDA events.
        # So release it holding the capture lock (not held while blocking on the queue above).
        with capture_lock:
            del msg
        if kind != "batch":
            return


def _pin_batch(batch: Any) -> Any:
    """
    :return: batch with all tensors pinned, allocated while holding the capture lock
        (a pinned host allocation during a CUDA graph capture invalidates the capture)
    """
    with capture_lock:
        return _pin(batch)


def _pin(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x if x.is_pinned() else x.pin_memory()
    if type(x) is dict:
        return {k: _pin(v) for k, v in x.items()}
    if type(x) in (list, tuple):
        return type(x)(_pin(v) for v in x)
    return x  # e.g. numpy arrays (strings), Python scalars: as they are, as in the DataLoader pinning


def _drain(q: queue.Queue):
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return
