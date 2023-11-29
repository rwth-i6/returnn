"""
torch.distributed utils
"""

from __future__ import annotations
from typing import Optional, Any, Dict
import os
import socket
import itertools

import torch

from returnn.config import Config


class DistributedContext:
    """
    This class setups some helper functions for torch distributed training
    """

    def __init__(self, options: Dict[str, Any]):
        import torch.distributed as dist

        # when no backend is specified, both gloo and nccl backends will be created
        # the gloo backend will be used for collectives with CPU tensors and
        # the nccl backend will be used for collectives with CUDA tensors
        dist.init_process_group(backend=None)

        self._opts = options
        self._local_rank = int(os.environ["LOCAL_RANK"])
        self._local_size = int(os.environ["LOCAL_WORLD_SIZE"])
        self._rank = dist.get_rank()
        self._size = dist.get_world_size()

        print(
            "Torch distributed initialized. Hostname %s, pid %i, rank %i / size %i, local rank %s / local size %s."
            % (socket.gethostname(), os.getpid(), self._rank, self._size, self._local_rank, self._local_size)
        )

    def local_rank(self) -> int:
        """local rank"""
        return self._local_rank

    def local_size(self) -> int:
        """local size"""
        return self._local_size

    def rank(self) -> int:
        """global rank"""
        return self._rank

    def size(self) -> int:
        """global size"""
        return self._size


_is_set_up = False
_ctx = None  # type: Optional[DistributedContext]


def get_ctx(config=None):
    """
    :param Config|None config:
    :returns: the global context if Torch distributed is enabled, or None otherwise.
      If we did not setup the context yet, it will automatically create it.
    :rtype: DistributedContext|None
    """
    global _is_set_up, _ctx
    if _is_set_up:
        return _ctx
    if not config:
        from returnn.config import get_global_config

        config = get_global_config(raise_exception=False)
        if not config:
            return None
    _is_set_up = True
    opts = config.typed_value("torch_distributed")
    if opts is None:
        return None
    _ctx = DistributedContext(opts)
    return _ctx


def get_device_ids():
    """
    It depends on the specific setup what to return here,
    how CUDA_VISIBLE_DEVICES is set up, etc.
    This is currently a reasonable assumption,
    but we might extend the logic later,
    or make it configurable.
    """
    return [get_local_rank()]


def get_local_rank():
    """
    torch.distributed does not seem to provide a function for this.
    Via mpirun (OpenMPI), this env variable would be set.
    It should fail with an error otherwise.
    """
    return int(os.environ["LOCAL_RANK"])


def _find_tensors(obj):
    """
    Recursively find all tensors contained in the specified object,
    cf. torch.nn.parallel.distributed._find_tensors
    """
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return itertools.chain(*map(_find_tensors, obj))
    if isinstance(obj, dict):
        return itertools.chain(*map(_find_tensors, obj.values()))
    return []
