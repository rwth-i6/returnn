"""
torch.distributed utils
"""

from __future__ import annotations
import ast
import logging
import os
import socket
from datetime import timedelta
from typing import Optional, Union, Any, Tuple, List, Dict

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
        if self._reduce_type == "param":
            assert isinstance(self._param_sync_step, int) and self._param_sync_step > 0, (
                f"reduce_type param: param_sync_step must be a positive int,"
                f" got {self._param_sync_step!r} ({type(self._param_sync_step).__name__})"
            )
            _logger.info(f"reduce_type param: param_sync_step {self._param_sync_step}")
        elif self._reduce_type == "grad":
            _logger.info("reduce_type grad")
        elif self._reduce_type == "grad_explicit":
            _logger.info("reduce_type grad_explicit")
        else:
            raise ValueError(f"invalid reduce_type {self._reduce_type!r}")

        self._sync_complete_frac: Optional[bool] = self._opts.get("sync_complete_frac", None)
        if self._sync_complete_frac is None:
            self._sync_complete_frac = BehaviorVersion.get() >= 33

        self._sync_at_epoch_end: Optional[bool] = self._opts.get("sync_at_epoch_end", None)
        if self._sync_at_epoch_end is None:
            self._sync_at_epoch_end = BehaviorVersion.get() >= 38

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
        _all_reduce_avg_flat([p.grad for p in module.parameters() if p.grad is not None])

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

    def step_after_param_update(self, *, module: torch.nn.Module, epoch_step_idx: int):
        """one train step"""
        if self._reduce_type == "param" and self.should_sync_now(epoch_step_idx=epoch_step_idx):
            _sync_params_avg(module=module, sync_on_cpu=self._opts.get("sync_on_cpu", False))

    def sync_at_epoch_end(self) -> bool:
        """
        :return: whether :func:`step_after_epoch` syncs the params, the buffers, the optimizer state and the step
            counts over all ranks at the end of every train epoch
            (option ``sync_at_epoch_end``, default on since behavior version 38).
            Only matters for reduce_type "param", where the ranks hold different params and optimizer states
            between the syncs. Then the checkpoint (written by rank 0) and the eval are the average,
            and every rank starts the next epoch with the state of the checkpoint,
            so resuming from the checkpoint continues with the same model and optimizer state as the uninterrupted
            training (the per-rank random states and data positions are not part of the checkpoint either way).
        """
        return self._sync_at_epoch_end

    def step_after_epoch(
        self, *, module: torch.nn.Module, optimizer: torch.optim.Optimizer, global_train_step: int
    ) -> int:
        """
        Make the state which goes into the checkpoint the same on every rank, see :func:`sync_at_epoch_end`.
        Call this at the end of the train epoch, before the checkpoint is written.

        Floating point tensors (params, buffers, optimizer state) take the mean over the ranks.
        Counts take the max: the train step, the ``step`` tensors of the optimizer state and integer tensors
        (e.g. ``num_batches_tracked`` of a batch norm). The ranks can differ in them after an epoch in which
        they got different numbers of batches (a rank out of data stops at its step, the others at their next sync),
        and every rank must count on as the checkpoint does.
        Other optimizer state values (no tensors) are taken from rank 0, which writes the checkpoint.

        :param module: to take the params and buffers from
        :param optimizer: to take the state from
        :param global_train_step: of this rank
        :return: the global train step to continue with, the same on every rank
        """
        import torch.distributed as dist

        if self._reduce_type != "param" or not self._sync_at_epoch_end:
            return global_train_step
        tensors = []  # mean
        counters = [torch.tensor([global_train_step], dtype=torch.int64)]  # max
        objects = []  # from rank 0
        for tensor in list(module.parameters()) + list(module.buffers()):
            (tensors if tensor.is_floating_point() else counters).append(tensor.data)
        for state in optimizer.state.values():
            for key in sorted(state):
                value = state[key]
                if not isinstance(value, torch.Tensor):
                    objects.append((state, key))
                elif key == "step" or not value.is_floating_point():
                    counters.append(value)
                else:
                    tensors.append(value)
        _all_reduce_avg_flat(tensors, sync_on_cpu=self._opts.get("sync_on_cpu", False))
        for counter in counters:
            counter_cpu = counter.to(torch.device("cpu"))
            dist.all_reduce(counter_cpu, op=dist.ReduceOp.MAX)
            counter.copy_(counter_cpu)
        if objects:
            values = [state[key] for state, key in objects]
            dist.broadcast_object_list(values, src=0, device=torch.device("cpu"))
            for (state, key), value in zip(objects, values):
                state[key] = value
        return int(counters[0])


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


_FLAT_REDUCE_BUCKET_BYTES = 256 * 1024**2


@torch.no_grad()
def _all_reduce_avg_flat(
    tensors: List[torch.Tensor], *, sync_on_cpu: bool = False, bucket_bytes: int = _FLAT_REDUCE_BUCKET_BYTES
):
    """
    Averages the tensors in place over all ranks.

    Flat buffers instead of one allreduce per tensor: thousands of small allreduces would be latency-bound.
    A flat buffer is a copy, so buckets of at most ``bucket_bytes`` bound the extra memory.
    One bucket holds one dtype on one device: the flat buffer is a ``cat``, which would promote mixed dtypes.

    :param tensors: any dtypes and devices
    :param sync_on_cpu: allreduce a CPU copy of each bucket (e.g. with Gloo for a CUDA module)
    :param bucket_bytes: of one flat buffer; a tensor larger than that gets a bucket of its own
    """
    buckets: Dict[Tuple[torch.device, torch.dtype], List[List[torch.Tensor]]] = {}
    for tensor in tensors:
        key_buckets = buckets.setdefault((tensor.device, tensor.dtype), [[]])
        if (
            key_buckets[-1]
            and sum(t.numel() for t in key_buckets[-1]) + tensor.numel() > bucket_bytes // tensor.element_size()
        ):
            key_buckets.append([])
        key_buckets[-1].append(tensor)
    for key_buckets in buckets.values():
        for bucket in key_buckets:
            # One flat buffer at a time: the function scope frees it before the next one is allocated.
            _all_reduce_avg_bucket(bucket, sync_on_cpu=sync_on_cpu)


def _all_reduce_avg_bucket(bucket: List[torch.Tensor], *, sync_on_cpu: bool):
    import torch.distributed as dist

    # noinspection protected-member
    flat = torch._utils._flatten_dense_tensors(bucket)
    flat_reduce = flat.cpu() if sync_on_cpu else flat
    # SUM then divide, not AVG: NCCL AVG pre-scales by 1/N rounded to the dtype, a bias for non-power-of-two N.
    dist.all_reduce(flat_reduce, op=dist.ReduceOp.SUM)
    flat_reduce /= dist.get_world_size()
    if flat_reduce is not flat:
        flat.copy_(flat_reduce)
    # noinspection protected-member
    for tensor, reduced in zip(bucket, torch._utils._unflatten_dense_tensors(flat, bucket)):
        tensor.copy_(reduced)


@torch.no_grad()
def _sync_params_avg(*, module: torch.nn.Module, sync_on_cpu: bool = False):
    _all_reduce_avg_flat([param.data for param in module.parameters()], sync_on_cpu=sync_on_cpu)
