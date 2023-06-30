"""
torch.distributed utils
"""

from __future__ import annotations
import itertools
from typing import Optional
import os
import socket

from contextlib import contextmanager
import torch
from torch.distributed.algorithms.join import Join

from returnn.config import Config
import returnn.frontend as rf


class DistributedContext:
    """
    This class setups some helper functions for torch distributed training
    """

    def __init__(self, config):
        """
        :param Config config:
        """
        import torch.distributed as dist

        # when no backend is specified, both gloo and nccl backends will be created
        # the gloo backend will be used for collectives with CPU tensors and
        # the nccl backend will be used for collectives with CUDA tensors
        dist.init_process_group(backend=None)

        self._config = config
        self._local_rank = os.environ["LOCAL_RANK"]
        self._local_size = os.environ["LOCAL_WORLD_SIZE"]
        self._rank = dist.get_rank()
        self._size = dist.get_world_size()

        print(
            "Torch distributed initialized. Hostname %s, pid %i, rank %i / size %i, local rank %s / local size %s."
            % (socket.gethostname(), os.getpid(), self._rank, self._size, self._local_rank, self._local_size)
        )

    def local_rank(self):
        """
        :rtype: int
        """
        return self._local_rank

    def rank(self):
        """
        :rtype: int
        """
        return self._rank

    def size(self):
        """
        :rtype: int
        """
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
    if config.typed_value("torch_distributed") is None:
        return None
    _ctx = DistributedContext(config=config)
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


@contextmanager
def ddp_train_forward_ctx(pt_model):
    """
    the original (unwrapped) module is passed to the train step, therefore here we set up the right context
    as what DistributedDataParallel.forward does internally
    """
    if torch.is_grad_enabled() and pt_model.require_backward_grad_sync:
        assert pt_model.logger is not None
        pt_model.logger.set_runtime_stats_and_log()
        pt_model.num_iterations += 1
        pt_model.reducer.prepare_for_forward()

    with torch.autograd.profiler.record_function("DistributedDataParallel.forward"):
        if torch.is_grad_enabled() and pt_model.require_backward_grad_sync:
            assert pt_model.logger is not None
            pt_model.logger.set_runtime_stats_and_log()
            pt_model.num_iterations += 1
            pt_model.reducer.prepare_for_forward()

        work = Join.notify_join_context(pt_model)
        if work:
            # noinspection PyProtectedMember
            pt_model.reducer._set_forward_pass_work_handle(work, pt_model._divide_by_initial_world_size)

        # noinspection PyProtectedMember
        if torch.is_grad_enabled() and pt_model.reducer._rebuild_buckets():
            pt_model._has_rebuilt_buckets = True

        # noinspection PyProtectedMember
        if pt_model._check_sync_bufs_pre_fwd():
            # noinspection PyProtectedMember
            pt_model._sync_buffers()

        # noinspection PyProtectedMember
        if pt_model._join_config.enable:
            # Notify joined ranks whether they should sync in backwards pass or not.
            # noinspection PyProtectedMember
            pt_model._check_global_requires_backward_grad_sync(is_joined_rank=False)

        # noinspection PyProtectedMember
        with pt_model._inside_ddp_forward():
            yield

        # noinspection PyProtectedMember
        if pt_model._check_sync_bufs_post_fwd():
            # noinspection PyProtectedMember
            pt_model._sync_buffers()

        if torch.is_grad_enabled() and pt_model.require_backward_grad_sync:
            pt_model.require_forward_param_sync = True
            # We'll return the output object verbatim since it is a freeform
            # object. We need to find any tensors in this object, though,
            # because we need to figure out which parameters were used during
            # this forward pass, to ensure we short circuit reduction for any
            # unused parameters. Only if `find_unused_parameters` is set.
            if pt_model.find_unused_parameters and not pt_model.static_graph:
                # Do not need to populate this for static graph.
                train_ctx = rf.get_run_ctx()
                loss = list(train_ctx.losses.values())[0].loss.raw_tensor
                # noinspection PyProtectedMember
                pt_model.reducer.prepare_for_backward(list(_find_tensors(loss)))
            else:
                pt_model.reducer.prepare_for_backward([])
        else:
            pt_model.require_forward_param_sync = False
