"""
Process-wide lock around CUDA graph captures.

A capture in the default "global" mode (``torch.cuda.graph``) is invalidated
by capture-unsafe CUDA calls from *any* thread of the process,
e.g. a pinned host memory allocation or the pinned host allocator's event handling.
Background threads which do such work (e.g. pinning the next batches, see :mod:`returnn.torch.data.pin_memory`)
hold :data:`capture_lock` while doing it, and every capture holds it for the whole capture
(see :func:`cuda_graph_capture`), so they never overlap.
Graph replays do not take the lock, so the background work still overlaps with the training steps.
"""

from __future__ import annotations
import threading
from contextlib import contextmanager
import torch

__all__ = ["capture_lock", "cuda_graph_capture"]


# reentrant: code running while the lock is held (e.g. a capture) may take it again
capture_lock = threading.RLock()


@contextmanager
def cuda_graph_capture(graph: torch.cuda.CUDAGraph):
    """
    Like ``torch.cuda.graph(graph)``, while holding :data:`capture_lock`.
    Every capture in RETURNN goes through this.

    :param graph:
    """
    with capture_lock, torch.cuda.graph(graph):
        yield
