"""
torch.distributed utils
"""

from __future__ import annotations
import ast
import logging
import os
import socket
from datetime import timedelta
from typing import Optional, Union, Any, Sequence, List, Dict

import torch
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
    or an explicit list of rank lists), a ``type`` ("grad" or "param"), and ``every`` (period in steps, default 1).
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
        self._sync_levels = _parse_sync_levels(sync_spec, size=self._size)
        if self._reduce_type == "sync":
            assert self._sync_levels, "torch_distributed: sync must have at least one level"
            _logger.info(f"sync levels: {self._sync_levels}")
        if not prev_init_info:
            # The subprocesses never sync; the process groups exist only in the main process.
            self._init_sync_groups()

        self._sync_complete_frac: Optional[bool] = self._opts.get("sync_complete_frac", None)
        if self._sync_complete_frac is None:
            self._sync_complete_frac = BehaviorVersion.get() >= 33

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

        hostnames = None
        for level in self._sync_levels:
            if level.group_spec == "node":
                if hostnames is None:
                    hostnames = _all_gather_hostnames(size=self._size)
                level.group_ranks = _make_group_ranks("node", size=self._size, hostnames=hostnames)
                for ranks in level.group_ranks:
                    assert len(ranks) == self._local_size, (
                        f"{self}: {level}: ranks {ranks} share a host, but local size is {self._local_size}"
                    )
        _check_sync_levels_nested(self._sync_levels)
        for level in self._sync_levels:
            assert level.group_ranks is not None
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
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        _all_reduce_avg_flat(grads, level=level)

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
        if self._reduce_type == "grad":
            return True
        return any(level.fires(epoch_step_idx=epoch_step_idx) for level in self._sync_levels)

    def step_after_param_update(self, *, module: torch.nn.Module, epoch_step_idx: int):
        """
        Average the params over the outermost "param" sync level firing in this step, if any.

        :param module: to take the params from
        :param epoch_step_idx: current step index
        """
        for level in reversed(self._sync_levels):
            if level.sync_type == "param" and level.fires(epoch_step_idx=epoch_step_idx):
                _sync_params_avg(module=module, level=level)
                return


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


class _SyncLevel:
    def __init__(
        self,
        *,
        group_spec: Union[str, int, Sequence[Sequence[int]]],
        group_ranks: Optional[List[List[int]]],
        sync_type: str,
        every: int,
        sync_on_cpu: bool,
    ):
        self.group_spec = group_spec
        self.group_ranks = group_ranks  # None for "node" until the hostnames are gathered
        self.sync_type = sync_type
        self.every = every
        self.sync_on_cpu = sync_on_cpu
        self.group: Optional[torch.distributed.ProcessGroup] = None  # None: the default group ("world")

    def __repr__(self):
        return f"<{self.__class__.__name__} group={self.group_spec!r} type={self.sync_type!r} every={self.every}>"

    def fires(self, *, epoch_step_idx: int) -> bool:
        """
        :param epoch_step_idx: current step index
        :return: whether this level syncs in this step
        """
        return (epoch_step_idx % self.every) == (self.every - 1)

    def group_size(self) -> int:
        """
        :return: number of ranks in the group of this rank
        """
        return torch.distributed.get_world_size(group=self.group)


def _parse_sync_levels(spec: Sequence[Dict[str, Any]], *, size: int) -> List[_SyncLevel]:
    assert isinstance(spec, (list, tuple)), f"torch_distributed sync: expected a list of levels, got {spec!r}"
    levels = []
    for idx, level_spec in enumerate(spec):
        assert isinstance(level_spec, dict), f"torch_distributed sync level {idx}: expected a dict, got {level_spec!r}"
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
        group_ranks = None if group_spec == "node" else _make_group_ranks(group_spec, size=size, hostnames=None)
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


def _check_sync_levels_nested(levels: Sequence[_SyncLevel]):
    for inner, outer in zip(levels[:-1], levels[1:]):
        for ranks in inner.group_ranks:
            assert any(set(ranks).issubset(outer_ranks) for outer_ranks in outer.group_ranks), (
                f"torch_distributed sync: group {ranks} of {inner} is not inside one group of {outer}"
            )


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


@torch.no_grad()
def _all_reduce_avg_flat(tensors: List[torch.Tensor], *, level: _SyncLevel):
    if not tensors:
        return
    # One flat buffer: thousands of small allreduces would be latency-bound.
    # noinspection protected-member
    flat = torch._utils._flatten_dense_tensors(tensors)
    torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM, group=level.group)
    flat /= level.group_size()
    # noinspection protected-member
    for tensor, reduced in zip(tensors, torch._utils._unflatten_dense_tensors(flat, tensors)):
        tensor.copy_(reduced)


@torch.no_grad()
def _sync_params_avg(*, module: torch.nn.Module, level: _SyncLevel):
    import torch.distributed as dist

    if level.sync_on_cpu:
        for param in module.parameters():
            # Separately move each param to CPU (instead of the whole module), to safe CPU memory.
            param_cpu = param.to(torch.device("cpu"))
            # On CPU, we are likely using Gloo, and Gloo does not support AVG
            dist.all_reduce(param_cpu.data, op=dist.ReduceOp.SUM, group=level.group)
            param_cpu.data /= level.group_size()
            param.data = param_cpu.to(param.device)
        return

    _all_reduce_avg_flat([param.data for param in module.parameters()], level=level)
