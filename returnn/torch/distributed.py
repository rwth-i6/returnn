"""
torch.distributed utils
"""

from __future__ import annotations
import ast
import logging
import os
import socket
from datetime import timedelta
from typing import Optional, Union, Any, Dict

import torch
from torch.nn.parallel import DistributedDataParallel

from returnn.config import Config
from returnn.util.basic import BehaviorVersion, CollectionReadCheckCovered

_logger = logging.getLogger("returnn.torch.distributed")


class DistributedContext:
    """
    This class setups some helper functions for torch distributed training
    """

    def __init__(self, options: Dict[str, Any]):
        self._opts = CollectionReadCheckCovered(options)

        # Subprocesses have issues initializing torch.distributed process groups.
        #
        # We therefore pass rank/size information of the process group via an env
        # variable that is automatically inherited in any created subprocess.
        env_var_name = "_RETURNN_TORCH_DISTRIBUTED_INIT_INFO"
        prev_init_info = os.environ.get(env_var_name)
        if prev_init_info:
            self.prev_init_info = ast.literal_eval(prev_init_info)
            self._rank = self.prev_init_info["rank"]
            self._size = self.prev_init_info["size"]
        else:
            import torch.distributed as dist

            # When no backend is specified, we set gloo for CPU tensors and nccl for CUDA tensors as backend.
            # torch 2.6.0 and onwards require explicitly setting the backends.
            # See https://github.com/rwth-i6/returnn/issues/1724 for discussion.
            dist.init_process_group(
                backend=self._opts.get("backend", "cpu:gloo,cuda:nccl"),
                timeout=timedelta(seconds=self._opts.get("timeout_sec", 1800)),
            )
            self._rank = dist.get_rank()
            self._size = dist.get_world_size()
            os.environ[env_var_name] = repr({"rank": self._rank, "size": self._size})

        self._local_rank = int(os.environ["LOCAL_RANK"])
        self._local_size = int(os.environ["LOCAL_WORLD_SIZE"])

        _logger.info(
            "Torch distributed initialized. Hostname %s, pid %i, rank %i / size %i, local rank %s / local size %s."
            % (socket.gethostname(), os.getpid(), self._rank, self._size, self._local_rank, self._local_size)
        )

        self._reduce_type = self._opts.get("reduce_type", "grad")
        self._param_sync_step: Optional[int] = self._opts.get("param_sync_step", None)
        self._sync_optimizer_state = False
        if self._reduce_type == "param":
            assert isinstance(self._param_sync_step, int) and self._param_sync_step > 0, (
                f"reduce_type param: param_sync_step must be a positive int,"
                f" got {self._param_sync_step!r} ({type(self._param_sync_step).__name__})"
            )
            self._sync_optimizer_state = self._opts.get("sync_optimizer_state", False)
            _logger.info(
                f"reduce_type param: param_sync_step {self._param_sync_step},"
                f" sync_optimizer_state {self._sync_optimizer_state}"
            )
        elif self._reduce_type == "grad":
            _logger.info("reduce_type grad")
        elif self._reduce_type == "grad_explicit":
            _logger.info("reduce_type grad_explicit")
        else:
            raise ValueError(f"invalid reduce_type {self._reduce_type!r}")

        self._sync_complete_frac: Optional[bool] = self._opts.get("sync_complete_frac", None)
        if self._sync_complete_frac is None:
            self._sync_complete_frac = BehaviorVersion.get() >= 33

        self._dataset_distribution = get_dataset_distribution(self._opts)

        self._check_no_unknown_opts()

    def __repr__(self):
        return f"<{self.__class__.__name__} size={self._size} reduce_type={self._reduce_type}>"

    def _check_no_unknown_opts(self):
        # We check that all opts in self._opts have been used.
        # This function here is called at the end in __init__,
        # and not all opts are used yet, so read them now,
        # such that the check in the end works.
        self._opts.get("backend")
        self._opts.get("timeout_sec")
        if self._reduce_type == "grad":
            self._opts.get("class")
            self._opts.get("options")
        if self._reduce_type == "param":
            self._opts.get("sync_on_cpu")

        self._opts.assert_all_read()

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

    def dataset_distribution(self) -> str:
        """:return: "random_seed_offset" or "shard", see :func:`get_dataset_distribution`"""
        return self._dataset_distribution

    def maybe_make_distributed_module(self, module: torch.nn.Module) -> Optional[DistributedDataParallel]:
        """
        Maybe make a wrapped distributed module.

        :param module: original module
        :return: potentially wrapped module
        """
        if self._reduce_type in ("param", "grad_explicit"):
            return None
        assert self._reduce_type == "grad"
        cls = self._opts.get("class", DistributedDataParallel)
        if cls is not DistributedDataParallel:
            _logger.warning(f"Using custom class {cls} instead of DistributedDataParallel, might be unsupported.")
        kwargs = self._opts.get("options", {})
        device_ids = [self.local_rank()]
        param = next(module.parameters(), None)
        if param is not None and param.device.type == "cpu":
            device_ids = None  # DistributedDataParallel takes no device ids for a CPU module (e.g. gloo)
        return cls(
            module=module,
            device_ids=device_ids,
            **kwargs,
        )

    def reduce_type(self) -> str:
        """reduce type"""
        return self._reduce_type

    def maybe_reduce_grads(self, *, module: torch.nn.Module):
        """
        Average the grads over all ranks, for reduce_type "grad_explicit". No-op otherwise.

        DistributedDataParallel does this via autograd hooks during backward.
        A compiled or captured step returns the grads instead, so those hooks never fire.
        Call this once the grads are complete, before the optimizer step.

        :param module: to take the params (and their grads) from
        """
        if self._reduce_type != "grad_explicit":
            return
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        if not grads:
            return
        # One flat buffer: thousands of small allreduces would be latency-bound.
        # noinspection protected-member
        flat = torch._utils._flatten_dense_tensors(grads)
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat /= self._size
        # noinspection protected-member
        for grad, reduced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
            grad.copy_(reduced)

    def sync_complete_frac(self) -> bool:
        """
        :return: whether the train loop replaces the rank-local ``complete_frac`` by its mean over the ranks
            in each step where it syncs (:func:`should_sync_now`),
            so that ``epoch_continuous`` (e.g. for ``dynamic_learning_rate``) is the same on every rank
        """
        return self._sync_complete_frac

    def should_sync_now(self, *, epoch_step_idx: int) -> bool:
        """
        :param epoch_step_idx: current step index
        :return: whether to sync the training processes in this step
        """
        if self._reduce_type in ("grad", "grad_explicit"):
            return True
        elif self._reduce_type == "param":
            return (epoch_step_idx % self._param_sync_step) == (self._param_sync_step - 1)
        else:
            raise ValueError(f"invalid reduce_type {self._reduce_type}")

    def step_after_param_update(
        self, *, module: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch_step_idx: int
    ):
        """
        Average the params over all ranks, for reduce_type "param" in the steps where it syncs.
        With the option ``sync_optimizer_state`` (default off), also the floating point optimizer state
        (e.g. the Adam moments); otherwise each rank keeps its own.

        :param module: to take the params from
        :param optimizer: to take the state from
        :param epoch_step_idx: current step index
        """
        if self._reduce_type == "param" and self.should_sync_now(epoch_step_idx=epoch_step_idx):
            _sync_params_avg(
                module=module,
                optimizer=optimizer,
                sync_on_cpu=self._opts.get("sync_on_cpu", False),
                sync_optimizer_state=self._sync_optimizer_state,
            )


_is_set_up = False
_ctx = None  # type: Optional[DistributedContext]


def get_ctx(config: Optional[Config] = None) -> Optional[DistributedContext]:
    """
    :param config:
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
    return _ctx


def get_dataset_distribution(opts: Union[Dict[str, Any], CollectionReadCheckCovered]) -> str:
    """
    :param opts: the ``torch_distributed`` config options
    :return: ``dataset_distribution`` option: how the train dataset is distributed over the ranks.
        "random_seed_offset" (default): every rank goes through all the data, with its own random seed.
        "shard": every rank gets its own shard of the data (``Dataset.get_default_kwargs_train``),
        with the same random seed on every rank.
    """
    dataset_distribution = opts.get("dataset_distribution", "random_seed_offset")
    if dataset_distribution not in ("random_seed_offset", "shard"):
        raise ValueError(
            f"torch_distributed dataset_distribution: expected 'random_seed_offset' or 'shard',"
            f" got {dataset_distribution!r}"
        )
    return dataset_distribution


@torch.no_grad()
def _sync_params_avg(
    *,
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sync_on_cpu: bool = False,
    sync_optimizer_state: bool = False,
):
    import torch.distributed as dist

    tensors = [param.data for param in module.parameters()]
    if sync_optimizer_state:
        for param in module.parameters():
            state = optimizer.state.get(param, {})
            for key in sorted(state):
                if not isinstance(state[key], torch.Tensor) or not state[key].is_floating_point():
                    continue
                if key == "step":
                    # The update count (the torch optimizers keep it as a tensor). The ranks can differ in it
                    # when they got different numbers of batches in an epoch; its mean would be no count.
                    step_cpu = state[key].to(torch.device("cpu"))
                    dist.all_reduce(step_cpu, op=dist.ReduceOp.MAX)
                    state[key].copy_(step_cpu)
                else:
                    tensors.append(state[key])

    if sync_on_cpu:
        for tensor in tensors:
            # Separately move each tensor to CPU (instead of all at once), to safe CPU memory.
            tensor_cpu = tensor.to(torch.device("cpu"))
            # On CPU, we are likely using Gloo, and Gloo does not support AVG
            dist.all_reduce(tensor_cpu, op=dist.ReduceOp.SUM)
            tensor_cpu /= dist.get_world_size()
            tensor.copy_(tensor_cpu)
        return

    if dist.get_backend() == "gloo":
        # Gloo does not support AVG
        reduce_op = dist.ReduceOp.SUM
    else:
        if hasattr(dist.ReduceOp, "AVG"):
            reduce_op = dist.ReduceOp.AVG
        else:
            # Older PyTorch versions do not have ReduceOp.AVG.
            reduce_op = dist.ReduceOp.SUM

    for tensor in tensors:
        dist.all_reduce(tensor, op=reduce_op)
        if reduce_op == dist.ReduceOp.SUM:
            tensor /= dist.get_world_size()
