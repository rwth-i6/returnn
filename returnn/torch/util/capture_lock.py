"""
Process-wide lock around CUDA graph captures.

A capture in the default "global" mode (``torch.cuda.graph``) is invalidated
by capture-unsafe CUDA calls from *any* thread of the process,
e.g. a pinned host memory allocation or the pinned host allocator's event handling.
Background threads which do such work (e.g. pinning the next batches, see :mod:`returnn.torch.data.pin_memory`)
hold :data:`capture_lock` while doing it, and every capture holds it for the whole capture
(see :func:`cuda_graph_capture`), so they never overlap.
Graph replays do not take the lock, so the background work still overlaps with the training steps.

Also: the replay of a graph with collectives, see :func:`replay`.
"""

from __future__ import annotations
import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import List
import torch

__all__ = ["capture_lock", "cuda_graph_capture", "replay"]


# reentrant: code running while the lock is held (e.g. a capture) may take it again
capture_lock = threading.RLock()


@contextmanager
def cuda_graph_capture(graph: torch.cuda.CUDAGraph):
    """
    Like ``torch.cuda.graph(graph)``, while holding :data:`capture_lock`.
    Every capture in RETURNN goes through this.
    Notes on the graph whether the capture issued collectives, for :func:`replay`.

    :param graph:
    """
    counts = _collective_counts()
    with capture_lock, torch.cuda.graph(graph):
        yield
    graph.returnn_has_collectives = _collective_counts() != counts


def replay(graph: torch.cuda.CUDAGraph):
    """
    ``graph.replay()``, for a graph captured by :func:`cuda_graph_capture`.
    If it has collectives, also wait until it is done, at most the timeout of the process group
    (``torch_distributed`` ``timeout_sec``), then end the process, as the NCCL watchdog does:
    it does not watch collectives inside a CUDA graph, so a rank which never joins them (e.g. it crashed)
    would block the next host sync forever. Aborting the communicator then blocks as well (torch 2.14),
    and so would an exception (in the cleanup).
    The wait makes the replay synchronous for the host (only for graphs with collectives).

    :param graph:
    """
    graph.replay()
    if not getattr(graph, "returnn_has_collectives", False):
        return
    event = getattr(graph, "returnn_done_event", None)
    if event is None:
        event = graph.returnn_done_event = torch.cuda.Event()
    event.record()
    from returnn.torch.distributed import get_ctx

    ctx = get_ctx()
    timeout_sec = ctx.timeout_sec if ctx is not None else 1800.0
    deadline = time.monotonic() + timeout_sec
    while not event.query():
        if time.monotonic() > deadline:
            from returnn.log import log

            msg = (
                f"CUDA graph replay with collectives did not finish within {timeout_sec} s"
                f" (torch_distributed timeout_sec), a rank is missing in a collective. Ending the process."
            )
            print(msg, file=log.v1)
            print(msg, file=sys.stderr, flush=True)
            log.flush()
            os._exit(1)
        time.sleep(1e-4)


def _collective_counts() -> List[int]:
    """
    :return: per process group, the number of its collectives so far (also the ones captured into a graph)
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return []
    # noinspection PyProtectedMember
    from torch.distributed.distributed_c10d import _world

    counts = []
    for group in list(_world.pg_map):
        try:
            counts.append(group._get_sequence_number_for_group())
        except RuntimeError:  # e.g. the communicator is not created yet
            counts.append(0)
    return counts
