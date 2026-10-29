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

import numpy
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
    A "param" level also syncs the optimizer state with ``sync_optimizer_state`` (default off).
    Every group of a level is a union of groups of the inner level, and every period a multiple of the inner one.
    A "grad" level runs in every step, so it has to be the innermost. Of the "param" levels firing in a step,
    only the outermost averages the params, which subsumes the inner averages; the optimizer state is averaged
    over the outermost firing level with ``sync_optimizer_state``.

    ``grad_reduce_dtype`` (e.g. "bfloat16", default: the grad dtype): the grad sync allreduces in this dtype,
    which halves the communication of float32 grads; the averaged grads are then rounded to it
    (the division by a non-power-of-two rank count too).
    Measured for bfloat16 against the float32 allreduce (random normal grads): relative RMS error of the averaged
    grads 0.24 % at 2 ranks, 0.33 % at 3 ranks.

    ``grad_reduce_bucket_mb`` (default 256, with ``grad_reduce_in_graph`` 64): the grad sync allreduces the grads
    in buckets of at most this size, one allreduce per bucket, see :class:`_FlatReducer`.
    In the graph, the bucket of the first params is only complete at the end of the backward, i.e. not overlapped,
    and every bucket waits for the previous one: measured best with 64 (vs 16 and 256) for 109M and 656M params
    on 2 GPUs over PCIe.

    ``grad_reduce_in_graph`` (default False): the grad allreduce runs inside the compiled and captured train step
    (``torch_cuda_graph`` with ``compile``), overlapped with the rest of the backward,
    instead of after the step, see :class:`InGraphGradReducer` (also for the requirements and the numerics).
    Needs a grad sync over all ranks (e.g. reduce_type "grad_explicit").
    """

    def __init__(self, options: Dict[str, Any]):
        self._opts = CollectionReadCheckCovered(options)
        # of the process groups, also of the collectives in a CUDA graph (see returnn.torch.util.capture_lock.replay)
        self.timeout_sec = float(self._opts.get("timeout_sec", 1800))

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
                timeout=timedelta(seconds=self.timeout_sec),
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
            sync_spec = [
                {
                    "group": "world",
                    "type": "param",
                    "every": self._param_sync_step,
                    "sync_on_cpu": self._opts.get("sync_on_cpu", False),
                    "sync_optimizer_state": self._opts.get("sync_optimizer_state", False),
                }
            ]
            _logger.info(f"reduce_type param: {sync_spec[0]}")
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
        # for the control sync over all ranks (DistributedDataParallel, first step, callable grad accumulation)
        self._world_level = _parse_sync_levels(
            [{"group": "world", "type": "grad"}], size=self._size, local_size=self._local_size, hostnames=None
        )[0]
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

        # What happens at the epoch end when the ranks have different numbers of batches.
        # "stop": a rank out of data ends the epoch for its group (and its next outer control syncs),
        # the other ranks drop their remaining batches.
        # "join": a rank out of data takes part in all collectives with zero contribution until all are out.
        self._epoch_end = self._opts.get("epoch_end", "stop")
        assert self._epoch_end in ("stop", "join"), f"torch_distributed: invalid epoch_end {self._epoch_end!r}"
        if self._epoch_end == "join" and not self._sync_complete_frac:
            # a rank out of data would keep its epoch_continuous while the others advance: another learning rate,
            # so the params of the ranks drift apart
            raise ValueError("torch_distributed: epoch_end 'join' needs sync_complete_frac=True")
        # epoch_end "join": per sync level, whether this rank did an optimizer update since the last sync there
        self._updated_since_sync = [False] * len(self._sync_levels)

        # the grad allreduce, see maybe_reduce_grads; its flat buffer is kept across the steps
        grad_reduce_dtype = self._opts.get("grad_reduce_dtype", None)
        if grad_reduce_dtype is not None:
            assert self.has_grad_sync(), "torch_distributed: grad_reduce_dtype needs a grad sync level"
            grad_reduce_dtype = getattr(torch, grad_reduce_dtype)
            assert isinstance(grad_reduce_dtype, torch.dtype), "torch_distributed: invalid grad_reduce_dtype"
        self._grad_reduce_in_graph = bool(self._opts.get("grad_reduce_in_graph", False))
        if self._grad_reduce_in_graph:
            # The step of a rank out of data (epoch_end "join") must run the same in-graph allreduce
            # with the same divisor: one group, the one of the control sync, which counts the ranks with data.
            assert self.has_grad_sync() and self._sync_levels[0].group_spec == "world", (
                "torch_distributed: grad_reduce_in_graph needs a grad sync over all ranks,"
                " e.g. reduce_type 'grad_explicit'"
            )
        bucket_mb = self._opts.get("grad_reduce_bucket_mb", 64 if self._grad_reduce_in_graph else 256)
        self._grad_reducer = _FlatReducer(dtype=grad_reduce_dtype, bucket_bytes=int(bucket_mb * 1024**2))

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

    def destroy_process_groups(self):
        """
        Destroy and free the process groups: the ones of the sync levels and the default group.
        Call this once at the end, instead of :func:`torch.distributed.destroy_process_group`.
        """
        import torch.distributed as dist

        if not dist.is_initialized():
            return  # a subprocess, which shares the init info but has no process groups
        groups: List[Optional[dist.ProcessGroup]] = []
        for level in self._sync_levels:
            if level.group is not None:
                groups.append(level.group)
                level.group = None
        groups.append(None)  # the default group, last
        _destroy_process_groups(groups)

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
        self._grad_reducer.reduce([p.grad for p in module.parameters() if p.grad is not None], group=level.group)

    def grad_reduce_in_graph(self) -> bool:
        """
        :return: whether the grads are averaged inside the captured train step (option ``grad_reduce_in_graph``),
            see :func:`make_in_graph_grad_reducer`
        """
        return self._grad_reduce_in_graph

    def make_in_graph_grad_reducer(self, *, device: torch.device) -> Optional[InGraphGradReducer]:
        """
        :param device: of the grads
        :return: for the captured train step with ``grad_reduce_in_graph``, else None
        """
        if not self._grad_reduce_in_graph:
            return None
        return InGraphGradReducer(
            dtype=self._grad_reducer.dtype,
            group=self._sync_levels[0].group,
            device=device,
            bucket_bytes=self._grad_reducer.bucket_bytes,
        )

    def epoch_end_join(self) -> bool:
        """
        :return: whether a rank out of data takes part with zero contribution until all ranks are out
            (option ``epoch_end``: "join"; default "stop")
        """
        return self._epoch_end == "join"

    def is_outermost_control_level(self, level: _SyncLevel) -> bool:
        """
        :param level: of a control sync, see :func:`control_sync_levels`
        :return: whether no rank outside of its group can still have data, i.e. its result ends the epoch
            for epoch_end "join"
        """
        return level is self._world_level or (bool(self._sync_levels) and level is self._sync_levels[-1])

    def reduce_grads_join(self, *, module: torch.nn.Module, num_batches: int, accum_grad_multiple_step: int) -> float:
        """
        epoch_end "join": for an update step while some rank of the grad sync group is out of data,
        instead of :func:`maybe_reduce_grads` or the DistributedDataParallel reduce (all ranks then).
        Sums the grads over the group (zeros from a rank without batches in this grad accumulation),
        divided by the number of batches over ``accum_grad_multiple_step``,
        i.e. the mean over the contributing ranks, not over the group size.
        Afterwards, all ranks of the group hold the same grads (None where no rank has one),
        so the optimizer step (clipping, grad scaler, weight decay, moments) is the same on all of them.

        :param module: to take the params (and their grads) from
        :param num_batches: batches of this rank in the current grad accumulation (0 when out of data)
        :param accum_grad_multiple_step:
        :return: number of batches in the group in the current grad accumulation. 0: no update.
        """
        group = self._sync_levels[0].group if self.has_grad_sync() else None  # DistributedDataParallel: all ranks
        params = [p for p in module.parameters() if p.requires_grad]
        if not params:
            return 0.0
        counts = [num_batches > 0 and p.grad is not None for p in params] + [num_batches]
        counts = torch.tensor(counts, dtype=torch.float32, device=params[0].device)
        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM, group=group)
        counts = counts.tolist()
        grads = []
        for param, count in zip(params, counts[:-1]):
            if not count:  # no rank of the group has a grad for it
                param.grad = None
                continue
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            elif not num_batches:  # e.g. the static grad buffers under graph capture
                param.grad.zero_()
            grads.append(param.grad)
        if counts[-1]:
            self._grad_reducer.reduce(grads, group=group, div=counts[-1] / accum_grad_multiple_step)
        return counts[-1]

    def sync_complete_frac(self) -> bool:
        """
        :return: whether the train loop replaces the rank-local ``complete_frac`` by its mean over the ranks
            of the control sync group in each step where it syncs (:func:`control_sync_levels`),
            so that ``epoch_continuous`` (e.g. for ``dynamic_learning_rate``) is the same on every rank of the group
        """
        return self._sync_complete_frac

    def control_sync_levels(
        self, *, epoch_update_step_idx: int, update_step: bool, world: bool, no_data: bool
    ) -> List[_SyncLevel]:
        """
        The control sync (all ranks of a group agree whether every one of them still has data, and on the progress)
        follows the sync levels: it runs on the group of the outermost level firing in this step.
        A group which ran out of data joins the next control sync of every outer level in turn,
        so the stop reaches all ranks, and every collective stays matched.

        :param epoch_update_step_idx: optimizer updates so far in this epoch
        :param update_step: whether the current step ends with an optimizer update (grad accumulation complete)
        :param world: sync over all ranks in this step in any case (first step of the epoch,
            or a callable ``accum_grad_multiple_step``, whose value for the step is only known after the step)
        :param no_data: this rank has no more data. Then it joins the control sync the other ranks of its group
            do next: the one of this step if there is one, else the one of the next firing level.
        :return: the levels to sync on in this step, inner to outer: the first is the control sync of this step,
            the rest are the ones to join when there is no data. Empty: no control sync in this step.
        """
        if world or self._reduce_type == "grad":
            return [self._world_level]
        # The choice depends only on the state the ranks of a group share (the pending update, whether this step
        # completes it), never on this rank's data, so that all of them enter the same collective.
        firing = [
            idx
            for idx, level in enumerate(self._sync_levels)
            if level.sync_type == "grad" or (update_step and level.fires(epoch_update_step_idx=epoch_update_step_idx))
        ]
        if firing:
            return _control_sync_chain(self._sync_levels, max(firing), epoch_update_step_idx)
        if not no_data:
            return []
        # No control sync in this step. The next one of this rank's group: after the next update which fires a level.
        if update_step:
            epoch_update_step_idx += 1  # this update completes now and fires nothing
        while not any(level.fires(epoch_update_step_idx=epoch_update_step_idx) for level in self._sync_levels):
            epoch_update_step_idx += 1
        firing = [
            idx
            for idx, level in enumerate(self._sync_levels)
            if level.fires(epoch_update_step_idx=epoch_update_step_idx)
        ]
        return _control_sync_chain(self._sync_levels, max(firing), epoch_update_step_idx)

    def step_after_param_update(
        self,
        *,
        module: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch_update_step_idx: int,
        updated: bool = True,
    ):
        """
        Average the params over the outermost "param" sync level firing after this optimizer update, if any,
        and the optimizer state over the outermost firing level with ``sync_optimizer_state`` (default off),
        see :func:`_sync_state`; without such a level each rank keeps its own.
        Call this right after an optimizer update.

        :param module: to take the params from
        :param optimizer: to take the state from
        :param epoch_update_step_idx: optimizer updates before this one in this epoch
        :param updated: whether this rank did the update (epoch_end "join": a rank out of data counts the updates
            of the others without one). The averages then skip the ranks without an update since the last sync there.
        """
        if updated:
            self._updated_since_sync = [True] * len(self._sync_levels)
        firing = [
            level
            for level in self._sync_levels
            if level.sync_type == "param" and level.fires(epoch_update_step_idx=epoch_update_step_idx)
        ]
        if not firing:
            return
        params_level = firing[-1]
        opt_levels = [level for level in firing if level.sync_optimizer_state]
        opt_level = opt_levels[-1] if opt_levels else None
        _sync_state(
            module=module,
            optimizer=optimizer if opt_level is params_level else None,
            sync_on_cpu=params_level.sync_on_cpu,
            group=params_level.group,
            weight=self._join_weight(params_level),
        )
        if opt_level is not None and opt_level is not params_level:
            _sync_state(
                module=module,
                optimizer=optimizer,
                params=False,
                sync_on_cpu=opt_level.sync_on_cpu,
                group=opt_level.group,
                weight=self._join_weight(opt_level),
            )
        idx = self._sync_levels.index(params_level)
        self._updated_since_sync[: idx + 1] = [False] * (idx + 1)

    def _join_weight(self, level: _SyncLevel) -> Optional[float]:
        """
        :return: epoch_end "join": weight of this rank in the average over the group of the level,
            0 without an own update since the last sync there; else None (plain mean)
        """
        if not self.epoch_end_join():
            return None
        return float(self._updated_since_sync[self._sync_levels.index(level)])

    def sync_at_epoch_end(self) -> bool:
        """
        :return: whether :func:`step_after_epoch` syncs the params, the buffers, the optimizer state and the step
            counts over all ranks at the end of every train epoch
            (option ``sync_at_epoch_end``, default on since behavior version 38).
            Only matters with a "param" sync level (reduce_type "param", or ``sync`` with one), where the ranks hold
            different params and optimizer states between the syncs.
            Then the checkpoint (written by rank 0) and the eval are the average,
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
        param_levels = [level for level in self._sync_levels if level.sync_type == "param"]
        _sync_state(
            module=module,
            optimizer=optimizer,
            buffers=True,
            counters=[global_train_step_t],
            sync_on_cpu=any(level.sync_on_cpu for level in self._sync_levels),
            weight=self._join_weight(param_levels[-1]),
        )
        self._updated_since_sync = [False] * len(self._sync_levels)
        return int(global_train_step_t)


def _destroy_process_groups(groups: List[Optional[torch.distributed.ProcessGroup]]):
    """
    :func:`torch.distributed.destroy_process_group` on each group, and free them.

    Torch 2.0 to 2.7 free a process group with the GIL held, and join its gloo worker threads,
    while a worker thread can still need the GIL to release the last collective (its tensors): a deadlock.
    The gloo backend object is freed without the GIL, so it is held while the groups are freed, and freed last.

    :param groups: None is the default group, which must come last.
        Takes over the references: the list is cleared.
        A group referenced elsewhere (e.g. the default group by :class:`DistributedDataParallel`)
        is freed later by its holder, as without this function.
    """
    import torch.distributed as dist

    backends = []
    if (2, 0) <= torch.__version__ < (2, 8):  # torch 2.8 binds ProcessGroup with a destructor without the GIL
        backends = [
            (dist.group.WORLD if group is None else group)._get_backend(torch.device("cpu"))
            for group in groups
            if "gloo" in str(dist.get_backend(group))
        ]
    for i in range(len(groups)):  # no loop variable left, which would hold the last group
        dist.destroy_process_group(groups[i])
    groups.clear()
    backends.clear()


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
        sync_optimizer_state: bool,
    ):
        self.group_spec = group_spec
        self.group_ranks = group_ranks
        self.sync_type = sync_type
        self.every = every
        self.sync_on_cpu = sync_on_cpu
        self.sync_optimizer_state = sync_optimizer_state
        self.group: Optional[torch.distributed.ProcessGroup] = None  # None: the default group ("world")

    def __repr__(self):
        flags = ""
        if self.sync_type == "param":
            flags = f" sync_on_cpu={self.sync_on_cpu} sync_optimizer_state={self.sync_optimizer_state}"
        return (
            f"<{self.__class__.__name__} group={self.group_spec!r} type={self.sync_type!r} every={self.every}{flags}>"
        )

    def fires(self, *, epoch_update_step_idx: int) -> bool:
        """
        :param epoch_update_step_idx: optimizer updates before the current one in this epoch
        :return: whether this level syncs after the current optimizer update
        """
        return (epoch_update_step_idx % self.every) == (self.every - 1)

    def group_size(self) -> int:
        """
        :return: number of ranks in the group of this rank
        """
        return torch.distributed.get_world_size(group=self.group)


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
        sync_optimizer_state = opts.get("sync_optimizer_state", False) if sync_type == "param" else False
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
                sync_optimizer_state=sync_optimizer_state,
            )
        )
    return levels


def _control_sync_chain(levels: Sequence[_SyncLevel], idx: int, epoch_update_step_idx: int) -> List[_SyncLevel]:
    """
    :param levels: the sync levels, inner to outer
    :param idx: the level of the control sync in this step
    :param epoch_update_step_idx: the pending optimizer update in this step
    :return: that level, then the levels of the control syncs the ranks outside its group do next, one after the other:
        after every stop, the next optimizer update after which a level further out fires, and the outermost of those.
        A group which ran out of data joins exactly these.
    """
    chain = [levels[idx]]
    while idx < len(levels) - 1:
        while not any(level.fires(epoch_update_step_idx=epoch_update_step_idx) for level in levels[idx + 1 :]):
            epoch_update_step_idx += 1
        idx = max(
            i for i in range(idx + 1, len(levels)) if levels[i].fires(epoch_update_step_idx=epoch_update_step_idx)
        )
        chain.append(levels[idx])
    return chain


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
    div: Optional[float] = None,
):
    """
    Averages the tensors in place over the ranks of the group, see :class:`_FlatReducer`.

    :param tensors: any dtypes and devices
    :param sync_on_cpu: allreduce a CPU copy of each bucket (e.g. with Gloo for a CUDA module)
    :param bucket_bytes: of one flat buffer; a tensor larger than that gets a bucket of its own
    :param group: the ranks to average over, None: all
    :param div: divide the sum by this instead of the group size
    """
    _FlatReducer(sync_on_cpu=sync_on_cpu, bucket_bytes=bucket_bytes).reduce(tensors, group=group, div=div)


class _FlatReducer:
    """
    Averages tensors in place over the ranks of a group, bucket by bucket through a flat buffer:
    copy the tensors of the bucket in, allreduce, divide, copy back; each of these one op.
    Flat buffers instead of one allreduce per tensor: thousands of small allreduces would be latency-bound.
    A flat buffer is a copy, so buckets of at most ``bucket_bytes`` bound the extra memory;
    the buckets of one device and dtype share one buffer.
    One bucket holds one dtype on one device, see :func:`_bucket_layout`.
    The buffer, the bucket layout and the views into the buffer are kept for the next :func:`reduce`
    with the same dtypes, shapes and devices (e.g. the grads in every step).
    """

    def __init__(
        self,
        *,
        dtype: Optional[torch.dtype] = None,
        sync_on_cpu: bool = False,
        bucket_bytes: int = _FLAT_REDUCE_BUCKET_BYTES,
    ):
        """
        :param dtype: allreduce in this dtype (e.g. bfloat16: half the communication), None: the dtype of the tensor
        :param sync_on_cpu: allreduce in a CPU buffer (e.g. with Gloo for a CUDA module)
        :param bucket_bytes: of one flat buffer; a tensor larger than that gets a bucket of its own
        """
        self.dtype = dtype
        self._sync_on_cpu = sync_on_cpu
        self.bucket_bytes = bucket_bytes
        self._key: Optional[List[Tuple[torch.dtype, torch.Size, torch.device]]] = None
        self._buckets: List[Tuple[List[int], torch.Tensor, List[torch.Tensor]]] = []  # tensor indices, flat, views

    @torch.no_grad()
    def reduce(
        self,
        tensors: List[torch.Tensor],
        *,
        group: Optional[torch.distributed.ProcessGroup] = None,
        div: Optional[float] = None,
    ):
        """
        :param tensors: any dtypes and devices, averaged in place
        :param group: the ranks to average over, None: all
        :param div: divide the sum by this instead of the group size
        """
        import torch.distributed as dist

        key = [(t.dtype, t.shape, t.device) for t in tensors]
        if key != self._key:
            self._build(tensors)
            self._key = key
        if div is None:
            div = dist.get_world_size(group=group)
        for indices, flat, views in self._buckets:
            bucket = [tensors[i] for i in indices]
            torch._foreach_copy_(views, bucket)
            # SUM then divide, not AVG: NCCL AVG pre-scales by 1/N rounded to the dtype, a bias for non-power-of-two N.
            dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
            flat /= div
            torch._foreach_copy_(bucket, views)

    def _build(self, tensors: List[torch.Tensor]):
        buckets = _bucket_layout(
            tensors, dtype=self.dtype, sync_on_cpu=self._sync_on_cpu, bucket_bytes=self.bucket_bytes
        )
        sizes: Dict[Tuple[torch.device, torch.dtype], int] = {}  # key -> max bucket numel
        for key, _, numel in buckets:
            sizes[key] = max(sizes.get(key, 0), numel)
        buffers = {key: torch.empty((size,), dtype=key[1], device=key[0]) for key, size in sizes.items()}
        self._buckets = []
        for key, indices, numel in buckets:
            flat = buffers[key][:numel]
            views = list(torch.split(flat, [tensors[i].numel() for i in indices]))
            views = [view.view(tensors[i].shape) for view, i in zip(views, indices)]
            self._buckets.append((indices, flat, views))


def _bucket_layout(
    tensors: List[torch.Tensor],
    *,
    dtype: Optional[torch.dtype],
    sync_on_cpu: bool = False,
    bucket_bytes: int = _FLAT_REDUCE_BUCKET_BYTES,
) -> List[Tuple[Tuple[torch.device, torch.dtype], List[int], int]]:
    """
    The buckets of :class:`_FlatReducer`: per key (device, dtype), in the order of the first tensor of each key,
    the tensors of that key in order, in buckets of at most ``bucket_bytes``, a larger tensor alone.
    (So with interleaved dtypes still one allreduce per dtype as long as they fit, not one per run of a dtype.)

    :return: per bucket: key, indices of its tensors (ascending), numel
    """
    buckets: Dict[Tuple[torch.device, torch.dtype], List[List[Any]]] = {}  # key -> [[indices, numel], ...]
    for i, tensor in enumerate(tensors):
        key = (torch.device("cpu") if sync_on_cpu else tensor.device, dtype or tensor.dtype)
        key_buckets = buckets.setdefault(key, [])
        max_numel = bucket_bytes // torch.tensor([], dtype=key[1]).element_size()
        if not key_buckets or key_buckets[-1][1] + tensor.numel() > max_numel:
            key_buckets.append([[], 0])
        key_buckets[-1][0].append(i)
        key_buckets[-1][1] += tensor.numel()
    return [(key, indices, numel) for key, key_buckets in buckets.items() for indices, numel in key_buckets]


class InGraphGradReducer:
    """
    The grad allreduce (reduce_type "grad_explicit", option ``grad_reduce_in_graph``)
    inside the traced, compiled and captured train step, see :mod:`returnn.torch.util.graph_capture`,
    instead of :func:`DistributedContext.maybe_reduce_grads` after it.

    The grads go into the same buckets as with :class:`_FlatReducer`, one functional allreduce
    (``torch.ops._c10d_functional``) per bucket, the bucket of the last params first: the backward completes the grads
    of the last params first. Inductor then issues each allreduce as soon as its bucket is complete
    (the "raise_comms" reordering, see :func:`compile_config`). It keeps the collectives of a group in their order,
    each one waits for the previous one, and the waits are as late as that allows (simple_overlap, torch 2.14).
    NCCL runs the allreduces on its own stream, which the CUDA graph captures as a branch of the step,
    so the communication overlaps with the rest of the backward; only the bucket of the first params,
    which is complete at the end of the backward, cannot (hence smaller buckets, ``grad_reduce_bucket_mb``).

    The divisor and whether this rank contributes are a device input of the step (:func:`set_contributing`),
    as a rank out of data (epoch_end "join") runs the same step, with the same collectives,
    on its last batch with a zero contribution (see :func:`GraphCapturedTrainStep.run_idle_step`).
    Every rank runs the same compiled graph, so the collectives match without further assumptions.
    A rank must not run out of data before the step is compiled (all ranks raise then).
    The NCCL communicator is created by the eager runs of the compile, before the capture.
    The NCCL watchdog does not watch collectives inside a CUDA graph, so the step waits for its end itself,
    see :func:`returnn.torch.util.capture_lock.replay`.
    Tested with torch 2.14.1 and NCCL 2.30.7.

    Same numerics as :class:`_FlatReducer`: the sum, multiplied by the float32 reciprocal of the divisor
    (as the eager division does), i.e. bitwise the same averaged grads from the same local grads.
    One exception: with a ``dtype`` below the grad dtype (e.g. bfloat16), Inductor fuses the multiplication with
    the conversion to the grad dtype and does not round the product to ``dtype`` in between,
    so with a divisor which is not a power of two (e.g. 3 ranks) the result is more precise than the one of
    :class:`_FlatReducer`, i.e. differs in the last bit of ``dtype``.
    (The local grads of two compiles of a step can differ in the last bits anyway, also without this option:
    Inductor picks some reduction kernel configs by timing.)
    """

    def __init__(
        self,
        *,
        dtype: Optional[torch.dtype],
        group: Optional[torch.distributed.ProcessGroup],
        device: torch.device,
        bucket_bytes: int = _FLAT_REDUCE_BUCKET_BYTES,
    ):
        """
        :param dtype: allreduce in this dtype, None: the grad dtype
        :param group: None: all ranks
        :param device: of the grads
        :param bucket_bytes: see :class:`_FlatReducer`
        """
        import torch.distributed as dist

        self.dtype = dtype
        self.bucket_bytes = bucket_bytes
        self.group_name = (group or dist.group.WORLD).group_name
        self.group_size = dist.get_world_size(group=group)
        # whether this rank contributes (1. or 0.), the reciprocal of the number of contributing ranks
        self.scale = torch.zeros((2,), dtype=torch.float32, device=device)
        self._scale_host: Optional[Tuple[bool, int]] = None
        self.set_contributing(True)

    def set_contributing(self, contribute: bool, *, num_ranks: Optional[int] = None):
        """
        Set the device input of the next step.

        :param contribute: whether this rank has a batch, else (epoch_end "join") it only takes part in the allreduce
        :param num_ranks: with a batch, of the group, default all
        """
        value = (contribute, num_ranks or self.group_size)
        if value != self._scale_host:
            # as the eager division by a Python number: the float32 reciprocal, see _FlatReducer.reduce
            inv = numpy.float32(1.0) / numpy.float32(value[1])
            self.scale.copy_(torch.tensor([float(contribute), inv], dtype=torch.float32))
            self._scale_host = value

    @staticmethod
    def compile_config() -> Dict[str, Any]:
        """
        :return: torch._inductor.config entries while the train step compiles:
            each bucket allreduce together with the compute it needs (its grads) goes as early as possible,
            in their order (raise_comms), after Inductor's reordering for peak memory, which would otherwise
            delay the grads of a bucket (e.g. the weight grad matmuls) up to the end of the backward
        """
        return {"reorder_for_compute_comm_overlap": True, "reorder_for_compute_comm_overlap_passes": ["raise_comms"]}

    def reduce(self, grads: List[torch.Tensor], scale: torch.Tensor) -> List[torch.Tensor]:
        """
        Traced inside the train step.

        :param grads: of this rank
        :param scale: :attr:`scale`, as a trace input
        :return: grads averaged over the contributing ranks
        """
        res = list(grads)
        buckets = _bucket_layout(grads, dtype=self.dtype, bucket_bytes=self.bucket_bytes)
        # the bucket with the last params first, as the backward completes them
        for (_, dtype), indices, _ in sorted(buckets, key=lambda bucket: bucket[1][0], reverse=True):
            bucket = [grads[i] for i in indices]
            # Each grad at a 16-byte aligned offset: in the grad dtype, the grads are views into the reduced buffer,
            # the inputs of e.g. the compiled optimizer step, which would copy misaligned ones in every call.
            align = 16 // torch.tensor([], dtype=dtype).element_size()
            sizes = [-(-grad.numel() // align) * align for grad in bucket]
            flat = torch.cat(
                [
                    torch.nn.functional.pad(grad.reshape(-1).to(dtype), (0, size - grad.numel()))
                    if size > grad.numel()
                    else grad.reshape(-1).to(dtype)
                    for grad, size in zip(bucket, sizes)
                ]
            )
            flat = torch.where(scale[0] > 0, flat, 0)  # no contribution (but NaN-safe)
            flat = torch.ops._c10d_functional.all_reduce(flat, "sum", self.group_name)
            flat = torch.ops._c10d_functional.wait_tensor(flat) * scale[1]
            for i, view, grad in zip(indices, torch.split(flat, sizes), bucket):
                res[i] = view[: grad.numel()].view(grad.shape).to(grad.dtype)
        return res


@torch.no_grad()
def _sync_state(
    *,
    module: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    params: bool = True,
    buffers: bool = False,
    counters: Sequence[torch.Tensor] = (),
    sync_on_cpu: bool = False,
    group: Optional[torch.distributed.ProcessGroup] = None,
    weight: Optional[float] = None,
):
    """
    Makes the state the same on all ranks of the group, one rule per kind:
    floating point tensors take the mean (the params, the buffers, the optimizer state such as the Adam moments),
    counts the max (``counters``, the ``step`` tensors of the optimizer state, integer tensors),
    other optimizer state values come from the first rank of the group.
    The ranks can differ in the counts after an epoch in which they got different numbers of batches
    (a rank out of data stops at its step, the others at their next sync); the mean of a count would be no count.

    :param module: its params with ``params``, and its buffers with ``buffers``
    :param optimizer: its state, if given: a flat dict of tensors per param for the torch optimizers,
        any nesting a custom optimizer keeps is fine (dm-tree)
    :param params: the params of the module
    :param buffers: also the buffers of the module (e.g. the batch norm running stats)
    :param counters: further counts, e.g. the train step, in place
    :param sync_on_cpu: allreduce CPU copies (e.g. with Gloo for a CUDA module)
    :param group: the ranks to sync over, None: all
    :param weight: of this rank in the mean (epoch_end "join": 0 for a rank without an update since the last sync).
        If the weights sum to 0 over the group, or None: plain mean.
    """
    import torch.distributed as dist

    tensors = []  # mean
    counters = list(counters)  # max
    objects = []  # (param, leaf index) from the first rank
    for tensor in (list(module.parameters()) if params else []) + (list(module.buffers()) if buffers else []):
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
    div = None
    if weight is not None:
        total = torch.tensor([weight], dtype=torch.float64)
        dist.all_reduce(total, op=dist.ReduceOp.SUM, group=group)
        if total.item() > 0:
            div = total.item()
            if weight != 1.0 and tensors:
                torch._foreach_mul_(tensors, weight)
    _all_reduce_avg_flat(tensors, sync_on_cpu=sync_on_cpu, group=group, div=div)
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
