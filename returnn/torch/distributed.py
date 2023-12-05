"""
torch.distributed utils
"""

from __future__ import annotations
from typing import Optional, Any, Dict
import os
import socket
import logging

import torch
from torch.nn.parallel import DistributedDataParallel

from returnn.config import Config

_logger = logging.getLogger("returnn.torch.distributed")


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

        _logger.info(
            "Torch distributed initialized. Hostname %s, pid %i, rank %i / size %i, local rank %s / local size %s."
            % (socket.gethostname(), os.getpid(), self._rank, self._size, self._local_rank, self._local_size)
        )

        self._reduce_type = self._opts.get("reduce_type", "grad")
        self._param_sync_step: Optional[int] = self._opts.get("param_sync_step", None)
        if self._reduce_type == "param":
            assert isinstance(self._param_sync_step, int) and self._param_sync_step > 0, (
                f"reduce_type param: param_sync_step must be a positive int,"
                f" got {self._param_sync_step!r} ({type(self._param_sync_step).__name__})"
            )
            _logger.info(f"reduce_type param: param_sync_step {self._param_sync_step}")
        elif self._reduce_type == "grad":
            _logger.info("reduce_type grad")
        else:
            raise ValueError(f"invalid reduce_type {self._reduce_type!r}")

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

    def get_param_sync_step(self) -> Optional[int]:
        """param sync step"""
        return self._param_sync_step

    def maybe_make_distributed_module(self, module: torch.nn.Module) -> Optional[DistributedDataParallel]:
        """
        Maybe make a wrapped distributed module.

        :param module: original module
        :return: potentially wrapped module
        """
        if self._reduce_type == "param":
            return None
        cls = self._opts.get("class", DistributedDataParallel)
        if cls is not DistributedDataParallel:
            _logger.warning(f"Using custom class {cls} instead of DistributedDataParallel, might be unsupported.")
        kwargs = self._opts.get("options", {})
        return cls(
            module=module,
            device_ids=[self.local_rank()],
            **kwargs,
        )

    def step_after_param_update(self, *, module: torch.nn.Module, epoch_step_idx: int):
        """one train step"""
        if self._reduce_type == "param" and ((epoch_step_idx % self._param_sync_step) == (self._param_sync_step - 1)):
            _sync_params_avg(module=module)


_is_set_up = False
_ctx = None  # type: Optional[DistributedContext]


def get_ctx(config=None) -> Optional[DistributedContext]:
    """
    :param Config|None config:
    :returns: the global context if Torch distributed is enabled, or None otherwise.
      If we did not setup the context yet, it will automatically create it.
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

    assert isinstance(opts, dict)
    _ctx = DistributedContext(opts)

    if _ctx.get_param_sync_step():
        # Just a sanity check.
        # Note that we could also instead just count the actual param update steps.
        # However, counting param update steps might be more variable in the future,
        # and also this behavior would be different from our TF implementation,
        # and in any case, it is probably more expected
        # that the param_sync_step is about the step (mini batch), not param update.
        accum_grad_multiple_step = config.int("accum_grad_multiple_step", 1)
        assert _ctx.get_param_sync_step() % accum_grad_multiple_step == 0, (
            f"{_ctx}: param_sync_step {_ctx.get_param_sync_step()}"
            f" must be a multiple of accum_grad_multiple_step {accum_grad_multiple_step}"
        )

    return _ctx


def _sync_params_avg(*, module: torch.nn.Module):
    import torch.distributed as dist

    # Older PyTorch versions do not have ReduceOp.AVG.
    reduce_op = getattr(dist.ReduceOp, "AVG", dist.ReduceOp.SUM)

    for param in module.parameters():
        dist.all_reduce(param.data, op=reduce_op)
        if reduce_op == dist.ReduceOp.SUM:
            param.data /= dist.get_world_size()
