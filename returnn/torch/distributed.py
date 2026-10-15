"""
torch.distributed utils
"""

from __future__ import annotations
import ast
import logging
import os
import socket
from datetime import timedelta
from typing import Optional, Union, Any, Sequence, Tuple, List, Dict

import torch
import tree
from torch.nn.parallel import DistributedDataParallel

from returnn.config import Config
from returnn.util.basic import BehaviorVersion, CollectionReadCheckCovered

_logger = logging.getLogger("returnn.torch.distributed")


class DistributedContext:
    """
    This class setups some helper functions for torch distributed training.

    How the ranks are synchronized is either the legacy ``reduce_type``
    ("grad": :class:`DistributedDataParallel`, "grad_explicit": grads averaged before the optimizer step,
    "param": params averaged every ``param_sync_step`` steps),
    or a list ``sync`` of nested levels, inner to outer, e.g.::

        "sync": [
            {"group": "node", "type": "grad"},                  # every step, among the ranks of a node
            {"group": "world", "type": "param", "every": 100},  # across the nodes
        ]

    Each level has a ``group`` ("world", "node" (the ranks on one host), an int (contiguous blocks of that many ranks),
    or an explicit list of rank lists), a ``type`` ("grad" or "param"),
    and ``every`` (period in optimizer update steps, counted per epoch, default 1).
    Every group of a level is a union of groups of the inner level, and every period a multiple of the inner one.
    A "grad" level runs in every step, so it has to be the innermost. Of the "param" levels firing in a step,
    only the outermost runs, which subsumes the inner averages.
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
            hostnames = self.prev_init_info["hostnames"]
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
            hostnames = None  # gathered below only when a sync level needs them

        self._local_rank = int(os.environ["LOCAL_RANK"])
        self._local_size = int(os.environ["LOCAL_WORLD_SIZE"])

        _logger.info(
            "Torch distributed initialized. Hostname %s, pid %i, rank %i / size %i, local rank %s / local size %s."
            % (socket.gethostname(), os.getpid(), self._rank, self._size, self._local_rank, self._local_size)
        )

        self._reduce_type: Optional[str] = self._opts.get("reduce_type", None)
        self._param_sync_step: Optional[int] = self._opts.get("param_sync_step", None)
        sync_spec = self._opts.get("sync", None)
        if sync_spec is not None:
            assert self._reduce_type is None, (
                f"torch_distributed: sync and reduce_type are exclusive, got reduce_type {self._reduce_type!r}"
            )
            self._reduce_type = "sync"
        elif self._reduce_type is None:
            self._reduce_type = "grad"
        if self._reduce_type == "param":
            assert isinstance(self._param_sync_step, int) and self._param_sync_step > 0, (
                f"reduce_type param: param_sync_step must be a positive int,"
                f" got {self._param_sync_step!r} ({type(self._param_sync_step).__name__})"
            )
            _logger.info(f"reduce_type param: param_sync_step {self._param_sync_step}")
            sync_spec = [
                {
                    "group": "world",
                    "type": "param",
                    "every": self._param_sync_step,
                    "sync_on_cpu": self._opts.get("sync_on_cpu", False),
                }
            ]
        elif self._reduce_type == "grad":
            _logger.info("reduce_type grad")
            sync_spec = []
        elif self._reduce_type == "grad_explicit":
            _logger.info("reduce_type grad_explicit")
            sync_spec = [{"group": "world", "type": "grad"}]
        elif self._reduce_type == "sync":
            pass
        else:
            raise ValueError(f"invalid reduce_type {self._reduce_type!r}")
        if not prev_init_info:
            if any(isinstance(level_spec, dict) and level_spec.get("group") == "node" for level_spec in sync_spec):
                hostnames = _all_gather_hostnames(size=self._size)
            os.environ[env_var_name] = repr({"rank": self._rank, "size": self._size, "hostnames": hostnames})
        self._sync_levels = _parse_sync_levels(
            sync_spec, size=self._size, local_size=self._local_size, hostnames=hostnames
        )
        if self._reduce_type == "sync":
            assert self._sync_levels, "torch_distributed: sync must have at least one level"
            _logger.info(f"sync levels: {self._sync_levels}")
        if not prev_init_info:
            # The subprocesses never sync; the process groups exist only in the main process.
            self._init_sync_groups()

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

        self._opts.assert_all_read()

    def _init_sync_groups(self):
        import torch.distributed as dist

        for level in self._sync_levels:
            if level.group_spec == "world":
                continue  # the default group
            for ranks in level.group_ranks:
                # Every rank creates every group (a collective), and keeps the one it belongs to.
                group = dist.new_group(ranks)
                if self._rank in ranks:
                    level.group = group

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
        if self._reduce_type != "grad":
            return None
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
        """reduce type ("grad", "grad_explicit", "param", or "sync" when configured via ``sync``)"""
        return self._reduce_type

    def has_grad_sync(self) -> bool:
        """
        :return: whether :func:`maybe_reduce_grads` averages the grads
            (reduce_type "grad_explicit", or a "grad" sync level)
        """
        return any(level.sync_type == "grad" for level in self._sync_levels)

    def maybe_reduce_grads(self, *, module: torch.nn.Module):
        """
        Average the grads over the ranks of the "grad" sync level (reduce_type "grad_explicit": all ranks).
        No-op without such a level.

        DistributedDataParallel does this via autograd hooks during backward.
        A compiled or captured step returns the grads instead, so those hooks never fire.
        Call this once the grads are complete, before the optimizer step.

        :param module: to take the params (and their grads) from
        """
        if not self.has_grad_sync():
            return
        level = self._sync_levels[0]
        assert level.sync_type == "grad"
        _all_reduce_avg_flat([p.grad for p in module.parameters() if p.grad is not None], group=level.group)

    def sync_complete_frac(self) -> bool:
        """
        :return: whether the train loop replaces the rank-local ``complete_frac`` by its mean over the ranks
            in each step where it syncs (:func:`should_sync_now`),
            so that ``epoch_continuous`` (e.g. for ``dynamic_learning_rate``) is the same on every rank
        """
        return self._sync_complete_frac

    def should_sync_now(self, *, epoch_update_step_idx: int, update_step: bool) -> bool:
        """
        :param epoch_update_step_idx: optimizer updates so far in this epoch
        :param update_step: whether the current step ends with an optimizer update (grad accumulation complete)
        :return: whether to sync the training processes in this step
        """
        if self._reduce_type == "grad":
            return True
        return any(
            level.sync_type == "grad" or (update_step and level.fires(epoch_update_step_idx=epoch_update_step_idx))
            for level in self._sync_levels
        )

    def step_after_param_update(self, *, module: torch.nn.Module, epoch_update_step_idx: int):
        """
        Average the params over the outermost "param" sync level firing after this optimizer update, if any.
        Call this right after an optimizer update.

        :param module: to take the params from
        :param epoch_update_step_idx: optimizer updates before this one in this epoch
        """
        for level in reversed(self._sync_levels):
            if level.sync_type == "param" and level.fires(epoch_update_step_idx=epoch_update_step_idx):
                _sync_state(module=module, sync_on_cpu=level.sync_on_cpu, group=level.group)
                return

    def sync_at_epoch_end(self) -> bool:
        """
        :return: whether :func:`step_after_epoch` syncs the params, the buffers, the optimizer state and the step
            counts over all ranks at the end of every train epoch
            (option ``sync_at_epoch_end``, default on since behavior version 38).
            Only matters with a "param" sync level (reduce_type "param", or ``sync`` with one), where the ranks hold
            different params and optimizer states between the syncs. Then the checkpoint (written by rank 0) and the eval are the average,
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
        if not self._sync_at_epoch_end or not any(level.sync_type == "param" for level in self._sync_levels):
            return global_train_step
        global_train_step_t = torch.tensor([global_train_step], dtype=torch.int64)
        _sync_state(
            module=module,
            optimizer=optimizer,
            buffers=True,
            counters=[global_train_step_t],
            sync_on_cpu=any(level.sync_on_cpu for level in self._sync_levels),
        )
        return int(global_train_step_t)


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


class _SyncLevel:
    def __init__(
        self,
        *,
        group_spec: Union[str, int, Sequence[Sequence[int]]],
        group_ranks: List[List[int]],
        sync_type: str,
        every: int,
        sync_on_cpu: bool,
    ):
        self.group_spec = group_spec
        self.group_ranks = group_ranks
        self.sync_type = sync_type
        self.every = every
        self.sync_on_cpu = sync_on_cpu
        self.group: Optional[torch.distributed.ProcessGroup] = None  # None: the default group ("world")

    def __repr__(self):
        return f"<{self.__class__.__name__} group={self.group_spec!r} type={self.sync_type!r} every={self.every}>"

    def fires(self, *, epoch_update_step_idx: int) -> bool:
        """
        :param epoch_update_step_idx: optimizer updates before the current one in this epoch
        :return: whether this level syncs after the current optimizer update
        """
        return (epoch_update_step_idx % self.every) == (self.every - 1)


def _parse_sync_levels(
    spec: Sequence[Dict[str, Any]], *, size: int, local_size: int, hostnames: Optional[Sequence[str]]
) -> List[_SyncLevel]:
    """
    :param spec: the ``sync`` option, see :class:`DistributedContext`
    :param size: world size
    :param local_size: ranks per host
    :param hostnames: per rank, needed for group "node"
    :return: the levels, inner to outer, groups and periods checked
    """
    assert isinstance(spec, (list, tuple)), f"torch_distributed sync: expected a list of levels, got {spec!r}"
    levels = []
    for idx, level_spec in enumerate(spec):
        assert isinstance(level_spec, dict), f"torch_distributed sync level {idx}: expected a dict, got {level_spec!r}"
        assert "group" in level_spec and "type" in level_spec, (
            f"torch_distributed sync level {idx}: group and type are required, got {level_spec!r}"
        )
        opts = CollectionReadCheckCovered(level_spec)
        group_spec = opts["group"]
        sync_type = opts["type"]
        every = opts.get("every", 1)
        sync_on_cpu = opts.get("sync_on_cpu", False) if sync_type == "param" else False
        opts.assert_all_read()
        assert sync_type in ("grad", "param"), f"torch_distributed sync level {idx}: invalid type {sync_type!r}"
        assert isinstance(every, int) and every > 0, (
            f"torch_distributed sync level {idx}: every must be a positive int, got {every!r}"
        )
        if sync_type == "grad":
            assert idx == 0, (
                f"torch_distributed sync level {idx}: a grad level runs every step, so it must be the first"
            )
            assert every == 1, f"torch_distributed sync level {idx}: a grad level runs every step, got every {every}"
        elif levels and levels[-1].sync_type == "param":
            assert every % levels[-1].every == 0, (
                f"torch_distributed sync level {idx}: every {every} is not a multiple of the inner {levels[-1].every}"
            )
        group_ranks = _make_group_ranks(group_spec, size=size, hostnames=hostnames)
        if group_spec == "node":
            for ranks in group_ranks:
                assert len(ranks) == local_size, (
                    f"torch_distributed sync level {idx}: ranks {ranks} share a host, but local size is {local_size}"
                )
        if levels:
            for ranks in levels[-1].group_ranks:
                assert any(set(ranks).issubset(outer_ranks) for outer_ranks in group_ranks), (
                    f"torch_distributed sync level {idx}: group {ranks} of the inner level {levels[-1]}"
                    f" is not inside one group {group_ranks}"
                )
        levels.append(
            _SyncLevel(
                group_spec=group_spec,
                group_ranks=group_ranks,
                sync_type=sync_type,
                every=every,
                sync_on_cpu=sync_on_cpu,
            )
        )
    return levels


def _make_group_ranks(
    group_spec: Union[str, int, Sequence[Sequence[int]]], *, size: int, hostnames: Optional[Sequence[str]]
) -> List[List[int]]:
    """
    :param group_spec: "world", "node", an int (contiguous blocks of that many ranks), or a partition of the ranks
    :param size: world size
    :param hostnames: per rank, needed for "node"
    :return: the groups, each a sorted list of ranks, together a partition of all ranks
    """
    if group_spec == "world":
        return [list(range(size))]
    elif group_spec == "node":
        assert hostnames is not None and len(hostnames) == size
        groups_by_host = {}
        for rank, hostname in enumerate(hostnames):
            groups_by_host.setdefault(hostname, []).append(rank)
        return sorted(groups_by_host.values())
    elif isinstance(group_spec, int):
        assert group_spec > 0 and size % group_spec == 0, (
            f"torch_distributed sync: group size {group_spec} does not divide the world size {size}"
        )
        return [list(range(start, start + group_spec)) for start in range(0, size, group_spec)]
    elif isinstance(group_spec, (list, tuple)):
        groups = [sorted(ranks) for ranks in group_spec]
        assert sorted(sum(groups, [])) == list(range(size)), (
            f"torch_distributed sync: groups {group_spec!r} are not a partition of the ranks 0..{size - 1}"
        )
        return groups
    else:
        raise TypeError(f"torch_distributed sync: invalid group {group_spec!r}")


def _all_gather_hostnames(*, size: int) -> List[str]:
    # A fixed-size CPU tensor keeps this on the CPU backend (gloo),
    # so no CUDA context is created before the device is selected.
    hostname = socket.gethostname().encode("utf8")
    assert len(hostname) <= 255
    buf = torch.zeros((256,), dtype=torch.uint8)
    buf[0] = len(hostname)
    buf[1 : 1 + len(hostname)] = torch.tensor(list(hostname), dtype=torch.uint8)
    bufs = [torch.empty_like(buf) for _ in range(size)]
    torch.distributed.all_gather(bufs, buf)
    return [bytes(b[1 : 1 + int(b[0])].tolist()).decode("utf8") for b in bufs]


_FLAT_REDUCE_BUCKET_BYTES = 256 * 1024**2


@torch.no_grad()
def _all_reduce_avg_flat(
    tensors: List[torch.Tensor],
    *,
    sync_on_cpu: bool = False,
    bucket_bytes: int = _FLAT_REDUCE_BUCKET_BYTES,
    group: Optional[torch.distributed.ProcessGroup] = None,
):
    """
    Averages the tensors in place over the ranks of the group.

    Flat buffers instead of one allreduce per tensor: thousands of small allreduces would be latency-bound.
    A flat buffer is a copy, so buckets of at most ``bucket_bytes`` bound the extra memory.
    One bucket holds one dtype on one device: the flat buffer is a ``cat``, which would promote mixed dtypes.

    :param tensors: any dtypes and devices
    :param sync_on_cpu: allreduce a CPU copy of each bucket (e.g. with Gloo for a CUDA module)
    :param bucket_bytes: of one flat buffer; a tensor larger than that gets a bucket of its own
    :param group: the ranks to average over, None: all
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
            _all_reduce_avg_bucket(bucket, sync_on_cpu=sync_on_cpu, group=group)


def _all_reduce_avg_bucket(
    bucket: List[torch.Tensor], *, sync_on_cpu: bool, group: Optional[torch.distributed.ProcessGroup]
):
    import torch.distributed as dist

    # noinspection protected-member
    flat = torch._utils._flatten_dense_tensors(bucket)
    flat_reduce = flat.cpu() if sync_on_cpu else flat
    # SUM then divide, not AVG: NCCL AVG pre-scales by 1/N rounded to the dtype, a bias for non-power-of-two N.
    dist.all_reduce(flat_reduce, op=dist.ReduceOp.SUM, group=group)
    flat_reduce /= dist.get_world_size(group=group)
    if flat_reduce is not flat:
        flat.copy_(flat_reduce)
    # noinspection protected-member
    for tensor, reduced in zip(bucket, torch._utils._unflatten_dense_tensors(flat, bucket)):
        tensor.copy_(reduced)


@torch.no_grad()
def _sync_state(
    *,
    module: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    buffers: bool = False,
    counters: Sequence[torch.Tensor] = (),
    sync_on_cpu: bool = False,
    group: Optional[torch.distributed.ProcessGroup] = None,
):
    """
    Makes the state the same on all ranks of the group, one rule per kind:
    floating point tensors take the mean (the params, the buffers, the optimizer state such as the Adam moments),
    counts the max (``counters``, the ``step`` tensors of the optimizer state, integer tensors),
    other optimizer state values come from the first rank of the group.
    The ranks can differ in the counts after an epoch in which they got different numbers of batches
    (a rank out of data stops at its step, the others at their next sync); the mean of a count would be no count.

    :param module: its params, and its buffers with ``buffers``
    :param optimizer: its state, if given: a flat dict of tensors per param for the torch optimizers,
        any nesting a custom optimizer keeps is fine (dm-tree)
    :param buffers: also the buffers of the module (e.g. the batch norm running stats)
    :param counters: further counts, e.g. the train step, in place
    :param sync_on_cpu: allreduce CPU copies (e.g. with Gloo for a CUDA module)
    :param group: the ranks to sync over, None: all
    """
    import torch.distributed as dist

    tensors = []  # mean
    counters = list(counters)  # max
    objects = []  # (param, leaf index) from the first rank
    for tensor in list(module.parameters()) + (list(module.buffers()) if buffers else []):
        (tensors if tensor.is_floating_point() else counters).append(tensor.data)
    if optimizer is not None:
        for param, state in optimizer.state.items():
            for idx, (path, value) in enumerate(tree.flatten_with_path(state)):
                if not isinstance(value, torch.Tensor):
                    objects.append((param, idx))
                elif path[-1] == "step" or not value.is_floating_point():
                    counters.append(value)
                else:
                    tensors.append(value)
    _all_reduce_avg_flat(tensors, sync_on_cpu=sync_on_cpu, group=group)
    for counter in counters:
        counter_cpu = counter.to(torch.device("cpu"))
        dist.all_reduce(counter_cpu, op=dist.ReduceOp.MAX, group=group)
        counter.copy_(counter_cpu)
    if objects:
        values = [tree.flatten(optimizer.state[param])[idx] for param, idx in objects]
        src = 0 if group is None else dist.get_global_rank(group, 0)
        dist.broadcast_object_list(values, src=src, group=group, device=torch.device("cpu"))
        for (param, idx), value in zip(objects, values):
            flat = tree.flatten(optimizer.state[param])
            flat[idx] = value
            optimizer.state[param] = tree.unflatten_as(optimizer.state[param], flat)
