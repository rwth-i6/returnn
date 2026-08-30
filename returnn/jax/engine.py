"""
Engine for the JAX backend.

The RF-side API is the same as for the other backends
(``get_model`` / ``train_step`` in the config, losses via ``rf.get_run_ctx().mark_as_loss``).

The step differs: JAX has no gradient tape, so the loss must be a function of the parameters.
The step binds the parameter arrays into the ``rf.Parameter`` objects for one call,
and ``jax.value_and_grad`` differentiates through it. Whole step jit-able, see ``jax_jit``.

Everything a compiled step depends on is an argument of it, not a Python global
(a global is read once at trace time, e.g. every step would draw the same dropout masks).
The epoch is a static argument, so config code can branch on it, at one recompile per epoch.
"""

from __future__ import annotations
from typing import Optional, Union, Any, Dict, List, Tuple
import os
import queue as _queue
import socket
import threading as _threading
import time
import shutil

import jax
import jax.numpy as jnp
import numpy

from returnn.config import Config
from returnn.datasets.basic import Dataset
from returnn.engine.base import EngineBase
from returnn.forward_iface import ForwardCallbackIface
from returnn.log import log
from returnn.tensor import Tensor, Dim, TensorDict, batch_dim
from returnn.util import basic as util
import returnn.frontend as rf

from .data import (
    iter_dataset_batches,
    fill_extern_data,
    reset_extern_data_dims,
    pad_raws_to_bucket,
    batch_to_jax_raws,
)
from returnn.datasets.packing import packed_batch_config, packed_batch_key_opts

# noinspection PyProtectedMember
from .frontend._backend import JaxBackend, _device_from_str
from .updater import Updater, global_grad_norm
from . import checkpoint as _checkpoint


__all__ = ["Engine"]


class Engine(EngineBase):
    """
    JAX engine: training and evaluation.
    """

    def __init__(self, config: Config):
        """
        :param config:
        """
        super().__init__(config=config)
        self.model: Optional[rf.Module] = None
        self.extern_data: Optional[TensorDict] = None
        self.train_dataset: Optional[Dataset] = None
        self.eval_datasets: Dict[str, Dataset] = {}
        self.learning_rate: float = config.float("learning_rate", 1.0)
        self._device: Optional[str] = config.value("device", None)
        if not self._device:
            # RF creates its own tensors (constants, ranges, random) on rf.get_default_device(),
            # JAX everything else on its default device, and mixing the two is an error.
            self._device = jax.devices()[0].platform
        rf.set_default_device(self._device)
        self._updater: Optional[Updater] = None
        self._opt_state: Any = None
        self._train_step_func = None
        # Config hooks around an epoch, e.g. to switch a dataset's parameters per epoch.
        # They run on the host, outside the step, so they may do whatever Python they like.
        self._epoch_start_func = config.typed_value("epoch_start")
        self._epoch_end_func = config.typed_value("epoch_end")
        # A nonfinite train score means the run is producing garbage from here on;
        # the default is to stop,
        # so that a broken run does not spend its whole budget writing NaN checkpoints.
        self._stop_on_nonfinite_train_score = config.bool("stop_on_nonfinite_train_score", True)
        self._save_model_epoch_interval = config.int("save_interval", 1)
        self._final_epoch = config.int("num_epochs", 1)
        self._forward_step_func = None
        self._forward_step_expected_outputs: Optional[TensorDict] = None
        self._params: List[rf.Parameter] = []
        self._train_param_idx: List[int] = []
        self._value_and_grad = None
        self._batch_opts = _batch_opts_from_config(config)
        self._jit_opts = _jit_opts_from_config(config)
        self._rng_key = None
        self._jitted_step = None
        self._compiled_steps: Dict[Any, Any] = {}  # input signature -> executable, see _run_compiled_step
        # Traced by default: as a static argument it is one full recompile of every shape
        # per epoch (measured 204 sec x 3 shapes x 100 subepochs).
        self._static_argnums: Tuple[int, ...] = (7,) if (self._jit_opts or {}).get("epoch_static") else ()
        # Numeric defaults stay JAX's own -- notably, float32 matmuls run in TF32 on GPU,
        # which PyTorch does not do by default. We do not override that here:
        # each backend follows its own conventions.
        # Set this option to "highest" if you want a run to match a PyTorch baseline numerically.
        precision = config.value("jax_default_matmul_precision", None)
        if precision:
            jax.config.update("jax_default_matmul_precision", precision)
            print(f"JAX default matmul precision: {precision}", file=log.v3)
        # Mixed precision: the compute dtype of matmul/conv, scoped to the step.
        # Parameters, gradients and the optimizer stay float32, see returnn.frontend.amp.
        # Persistent compilation cache: a bucket costs ~35 s, and the whole grid is compiled
        # at startup, again after every resubmission (the 11.9h SLURM limit splits a run).
        # Located like the other RETURNN caches, via util.get_cache_dir() (RETURNN_CACHE_DIR),
        # whose default is node-local: point it at a shared filesystem to survive a resubmission.
        cache_dir = config.value("jax_compilation_cache_dir", None) or f"{util.get_cache_dir()}/returnn_jax_compile"
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        # default 1s: our step compiles take far longer, so cache all of them
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
        print(f"JAX compilation cache: {cache_dir}", file=log.v3)
        if self._jit_opts is not None:
            print(f"JAX engine: compiled step, {self._jit_opts}", file=log.v3)
        # Diagnostics, the same ones the PyTorch engine has, so the logs of a JAX run and a torch
        # run parse identically (the throughput tooling reads these lines).
        self._log_batch_size = config.bool("log_batch_size", False) and log.verbose[5]
        # How many batches the input pipeline runs ahead of the step.
        # Default 2: enough to cover one step's worth of device time,
        # and it bounds the extra host memory to 2 batches.
        # 0 restores the serial loop, for comparing.
        self._data_prefetch = config.int("jax_data_prefetch", 2)
        # Declared bounds instead of the bucket grid:
        # every batch is brought to one signature,
        # rather than rounded up to the nearest of many declared shapes.
        # Same options as the TF engine's tf_static_shapes and the PyTorch one's torch_cuda_graph.
        self._static_shapes_opts = _static_shapes_opts_from_config(config)
        if self._static_shapes_opts is not None:
            print(f"JAX engine: static shapes, {self._static_shapes_opts}", file=log.v3)
        # packed_tensors: the model computes on sum(lens) frames instead of n_seqs * max_len.
        # The packing itself happens inside the step (one gather per key),
        # so the batching layer only has to budget by content
        # -- that is what packed_batch_size does.
        self._packing = packed_batch_config()
        if self._packing is not None:
            print(f"JAX engine: packed tensors, {self._packing}", file=log.v3)
            if self._jit_opts is not None and self._static_shapes_opts is None:
                # A compiled step needs a static shape for the packed buffer,
                # i.e. a declared total per packed key.
                # Loud, not silent:
                # tracing rf.pack without one would fail deep inside the model instead.
                raise NotImplementedError(
                    "JAX engine: packed_tensors with jax_jit needs jax_static_shapes"
                    " (packed_total_bound per packed key), or drop jax_jit"
                )
        amp = config.value("jax_amp", None)
        self._amp_policy: Optional[rf.AmpPolicy] = rf.AmpPolicy(compute_dtype=amp) if amp else None
        if self._amp_policy:
            print(f"JAX engine: mixed precision, compute dtype {amp}", file=log.v3)

    def init_train_from_config(
        self,
        config: Optional[Config] = None,
        train_data: Optional[Dataset] = None,
        dev_data: Optional[Dataset] = None,
        eval_data: Optional[Dataset] = None,
    ):
        """
        :param config:
        :param train_data:
        :param dev_data:
        :param eval_data:
        """
        assert config is self.config or config is None
        config = self.config
        self.train_dataset = train_data
        self.eval_datasets = {name: ds for name, ds in [("dev", dev_data), ("eval", eval_data)] if ds is not None}
        if config.has("eval_datasets"):
            from returnn.datasets import init_dataset

            for name, dataset_opts in config.typed_value("eval_datasets", {}).items():
                self.eval_datasets[name] = init_dataset(dataset_opts, default_kwargs={"name": name})
        self.learning_rate_control = _load_learning_rate_control(config)

        from returnn.torch.data.extern_data import extern_data_template_from_config_opts

        self.extern_data = extern_data_template_from_config_opts(config.typed_value("extern_data"))
        if self._jit_opts is not None:
            _check_time_multiple(self._jit_opts["time_multiple"], extern_data=self.extern_data)
            host_only = [key for key, data in self.extern_data.data.items() if data.dtype == "string"]
            if host_only:
                print(f"JAX engine: compiled step, {host_only} is not available inside it", file=log.v3)

        self._train_step_func = config.typed_value("train_step")
        assert self._train_step_func, "train_step not defined in config"

        _check_config_opts_supported(config)
        self.model_filename = config.value("model", None)
        self.epoch = self.get_train_start_epoch(config)
        if self.global_train_step is None:
            self.global_train_step = 0
        self._create_model(epoch=self.epoch, step=self.global_train_step)
        self._updater = Updater(
            config=config,
            model=self.model,
            param_names=[self._param_names[i] for i in self._train_param_idx],
        )
        self._init_step_func()
        # Continue from an existing checkpoint if there is one (or one was configured via `load`).
        # Without this the engine would silently restart from scratch and overwrite it.
        load_epoch, load_filename = self.get_epoch_model(config)
        if load_filename:
            self._load_model(filename=load_filename)
            print(f"Continuing after epoch {load_epoch} ({load_filename})", file=log.v3)
        # After the checkpoint, so an explicitly preloaded part wins over what the checkpoint has.
        self._preload_from_files(is_first_train_epoch=not load_filename)
        print(
            f"JAX engine: starting at epoch {self.epoch}, device {_device_description()},"
            f" host {socket.gethostname()}, devices {jax.devices()}",
            file=log.v3,
        )
        if self._jit_opts is not None:
            self._precompile_buckets()

    def train(self):
        """
        Train for the configured number of epochs.
        """
        assert self.train_dataset is not None, "no train dataset"
        final_epoch = self._final_epoch
        while self.epoch <= final_epoch:
            self.init_train_epoch()
            self.train_epoch()
            self.epoch += 1
        print("Finished training.", file=log.v3)

    def init_train_epoch(self):
        """
        Learning rate and random seed for this (sub)epoch.
        """
        self.learning_rate = self.learning_rate_control.get_learning_rate_for_epoch(self.epoch)
        # The PyTorch engine's fields, above all the learning rate the epoch actually ran at:
        # the schedule makes it a function of the step, so it is otherwise not recoverable.
        self.learning_rate_control.epoch_data[self.epoch].meta.update(
            {
                "global_train_step": self.global_train_step,
                "effective_learning_rate": self._updater.get_effective_learning_rate(
                    learning_rate=self.learning_rate,
                    global_train_step=self.global_train_step,
                    epoch=self.epoch,
                    epoch_continuous=self.epoch - 1,
                ),
                "returnn": util.describe_returnn_version(),
                "jax": jax.__version__,
                "time": time.strftime("%Y-%m-%d-%H-%M-%S (UTC%z)"),
                "hostname": socket.gethostname(),
                # The GPU model, as the PyTorch engine records it:
                # rf.get_default_device() only says "gpu", which cannot tell two runs apart.
                "device": _device_description(),
                "cpu": util.get_cpu_model_name(),
            }
        )
        # Same seeding logic as the other engines: the epoch and step take part,
        # so dropout and dataset shuffling differ per epoch but stay reproducible.
        random_seed = self.config.int("random_seed", 42)
        rf.set_random_seed(
            (self.epoch * 193939 + self.global_train_step * 19937 + random_seed * 27644437 + 479001599) % (2**31)
        )

    def train_epoch(self):
        """
        One epoch over the train dataset.
        """
        print(f"start epoch {self.epoch} with learning rate {self.learning_rate} ...", file=log.v3)
        self._on_epoch_start(dataset_name="train")
        start_time = time.time()
        # summed losses and summed normalization factors, divided into a score at the end
        accumulated: Dict[str, float] = {}
        accumulated_norms: Dict[str, float] = {}
        num_steps = 0
        # The RNG stream goes through the step as a value, so take it out of the backend here
        # and put the advanced one back at the end of the epoch.
        # noinspection PyProtectedMember
        self._rng_key = self._commit_one(JaxBackend._get_rng_key_())

        pending_losses: Optional[Dict[str, Any]] = None  # of the step still running on the device
        pending_log: Optional[Dict[str, Any]] = None  # what to log about it, read at the same time
        pending_bucket: Optional[Tuple[int, ...]] = None  # the shape that step ran on
        bucket_stats: Dict[Tuple[int, ...], List[Any]] = {}  # shape -> [n steps, summed device wait]
        # Where the epoch's wall-clock goes.
        # Dispatch is async, so the loss read below is the only place the host waits:
        # t_dev_wait is device time the host could not hide, t_data is the input pipeline.
        t_data = t_dev_wait = 0.0
        t_mark = t_step_done = time.time()
        for batch_raws, complete_frac in _prefetch(
            self._iter_batches(self.train_dataset, train=True), buffer_size=self._data_prefetch
        ):
            t_data += time.time() - t_mark  # the generator ran between the iterations
            # The previous step's losses, not the one just issued.
            # Dispatch is async, so reading any earlier would serialize
            # the host pipeline (0.16 s) and the step (0.105 s).
            if pending_losses is not None:
                _t0 = time.time()
                _accumulate_losses(pending_losses, sums=accumulated, norms=accumulated_norms)
                _waited = time.time() - _t0  # blocks until the previous step finished
                t_dev_wait += _waited
                # Per-bucket cost, attributed to the shape it waited on.
                # Not recoverable from epoch totals (the bucket counts are collinear),
                # and it needs no extra sync: this read already blocks.
                if pending_bucket is not None:
                    entry = bucket_stats.setdefault(pending_bucket, [0, 0.0])
                    entry[0] += 1
                    entry[1] += _waited
                if pending_log:
                    _now = time.time()
                    # completion to completion, i.e. the throughput figure,
                    # which is also what the torch engine's sec/step measures
                    pending_log["sec_per_step"] = _now - t_step_done
                    pending_log["start_elapsed"] = _now - start_time
                    pending_log["mem_usage"] = _device_peak_bytes()
                    t_step_done = _now
                    print(_format_step_log(self.epoch, pending_losses, pending_log), file=log.v5)
                del pending_losses, pending_log, pending_bucket
            # The learning rate of the step: the epoch-level value, put through the config's schedule if it has one.
            learning_rate = self._updater.get_effective_learning_rate(
                learning_rate=self.learning_rate,
                global_train_step=self.global_train_step,
                epoch=self.epoch,
                epoch_continuous=(self.epoch - 1 + complete_frac) if complete_frac is not None else None,
            )
            step_raws = self._step_raws(batch_raws)
            pending_bucket = _bucket_key(step_raws) if self._jit_opts else None
            train_raws, other_raws, self._opt_state, self._rng_key, loss, losses, grad_norm = self._train_step(
                [self._params[i].raw_tensor for i in self._train_param_idx],
                [self._params[i].raw_tensor for i in self._other_param_idx],
                step_raws,
                self._opt_state,
                self._rng_key,
                jnp.asarray(learning_rate, dtype=jnp.float32),
                jnp.asarray(self.global_train_step, dtype=jnp.int32),
                self.epoch if self._static_argnums else jnp.asarray(self.epoch, dtype=jnp.int32),
            )
            for idx, raw in zip(self._train_param_idx + self._other_param_idx, list(train_raws) + list(other_raws)):
                self._params[idx].raw_tensor = raw
            # The dims of the templates hold what the step filled in; after a compiled step that is
            # a tracer of a finished trace, which any later use would fail on.
            reset_extern_data_dims(self.extern_data)
            # Everything to log about this step, read back at the deferred point below so that
            # reading it costs no extra device sync (see the comment there).
            pending_losses = losses
            pending_log = {"step": num_steps}
            if self._updater.log_grad_norm_p:
                pending_log[f"grad_norm:p{self._updater.log_grad_norm_p:g}"] = grad_norm
            if self._log_batch_size:
                pending_log.update(_batch_size_info(batch_raws))
            num_steps += 1
            self.global_train_step += 1
            t_mark = time.time()
            if num_steps % 100 == 0:
                print(f"ep {self.epoch} step {num_steps}, loss {float(loss):.5f}", file=log.v4)

        if pending_losses is not None:  # the last step of the epoch
            _t0 = time.time()
            _accumulate_losses(pending_losses, sums=accumulated, norms=accumulated_norms)
            _waited = time.time() - _t0  # read ONCE, so the two accumulators agree
            t_dev_wait += _waited
            if pending_bucket is not None:
                entry = bucket_stats.setdefault(pending_bucket, [0, 0.0])
                entry[0] += 1
                entry[1] += _waited
            if pending_log:
                _now = time.time()
                pending_log["sec_per_step"] = _now - t_step_done
                pending_log["start_elapsed"] = _now - start_time
                pending_log["mem_usage"] = _device_peak_bytes()
                print(_format_step_log(self.epoch, pending_losses, pending_log), file=log.v5)
        JaxBackend._rng_key = self._rng_key
        elapsed = time.time() - start_time
        scores = {name: value / accumulated_norms[name] for name, value in accumulated.items()}
        print(
            f"epoch {self.epoch} finished: {num_steps} steps, {_format_scores(scores)}, {elapsed:.1f} sec", file=log.v3
        )
        # host vs device: the split that says which side is worth optimizing
        print(
            f"epoch {self.epoch} time split:"
            f" data {t_data:.1f}s ({100 * t_data / max(elapsed, 1e-9):.0f}%),"
            f" device-wait {t_dev_wait:.1f}s ({100 * t_dev_wait / max(elapsed, 1e-9):.0f}%),"
            f" other {elapsed - t_data - t_dev_wait:.1f}s"
            f" -- per step: {t_data / max(num_steps, 1) * 1000:.1f} ms data,"
            f" {t_dev_wait / max(num_steps, 1) * 1000:.1f} ms device-wait",
            file=log.v3,
        )
        # Per-shape cost.
        # The device wait is a lower bound (the part the host could not hide),
        # so it understates all shapes equally and still compares them honestly.
        if bucket_stats:
            ranked = sorted(bucket_stats.items(), key=lambda kv: -kv[1][1])
            print(
                f"epoch {self.epoch} per-shape device wait:"
                + "".join(
                    f"\n  {key} n {n} mean {total / n * 1000:.1f} ms total {total:.1f} s" for key, (n, total) in ranked
                ),
                file=log.v3,
            )
        # train_loss_, not train_score_:
        # the key names are the PyTorch engine's,
        # so that a run of either engine can be read by the same downstream code
        # and compared key by key.
        self.learning_rate_control.set_epoch_error(self.epoch, {f"train_loss_{k}": v for k, v in scores.items()})
        self.learning_rate_control.epoch_data[self.epoch].meta.update(
            {
                "epoch_num_train_steps": num_steps,
                "epoch_train_time_secs": round(elapsed),
                "global_train_step_end": self.global_train_step,
            }
        )
        self._report_dev_memory_stats()
        if self._stop_on_nonfinite_train_score:
            nonfinite = {name: value for name, value in scores.items() if not numpy.isfinite(value)}
            if nonfinite:
                print(f"Model seems broken, got inf or nan score: {nonfinite}", file=log.v1)
                self._report_nonfinite_params()
                raise Exception(f"epoch {self.epoch}: nonfinite train score {nonfinite}")
        self._on_epoch_end(dataset_name="train")
        self._maybe_stop_for_resubmission(time.time() - start_time)
        # save_interval: the last epoch is always saved, whatever the interval, so that a run
        # always ends with a usable checkpoint.
        if self.epoch % self._save_model_epoch_interval == 0 or self.epoch == self._final_epoch:
            self._save_model()
        self.eval_model()
        self.learning_rate_control.save()
        if self.config.bool_or_other("cleanup_old_models", None):
            self.cleanup_old_models()

    def _on_epoch_start(self, *, dataset_name: str):
        """
        :param dataset_name: which dataset the epoch is about to run over
        """
        if self._epoch_start_func:
            self._epoch_start_func(
                epoch=self.epoch,
                step=self.global_train_step,
                model=self.model,
                dataset_name=dataset_name,
                **util.get_fwd_compat_kwargs(),
            )

    def _on_epoch_end(self, *, dataset_name: str):
        """
        :param dataset_name: which dataset the epoch just ran over
        """
        if self._epoch_end_func:
            self._epoch_end_func(
                epoch=self.epoch,
                step=self.global_train_step,
                model=self.model,
                dataset_name=dataset_name,
                **util.get_fwd_compat_kwargs(),
            )

    def _report_nonfinite_params(self):
        """
        Which parameters hold inf/nan, for a broken run. Whether any do at all is the first thing
        that separates a diverged model from a single bad batch.
        """
        count = 0
        for name, param in zip(self._param_names, self._params):
            raw = param.raw_tensor
            got_nan, got_inf = bool(jnp.isnan(raw).any()), bool(jnp.isinf(raw).any())
            if got_nan or got_inf:
                what = "/".join(s for s, b in [("nan", got_nan), ("inf", got_inf)] if b)
                print(f"  {name} {param}: {what}", file=log.v1)
                count += 1
        if not count:
            print("(No inf/nan in model parameters.)", file=log.v1)

    def eval_model(self):
        """
        Run over the eval datasets and report the scores.
        """
        for name, dataset in self.eval_datasets.items():
            self._on_epoch_start(dataset_name=name)
            accumulated: Dict[str, float] = {}
            accumulated_norms: Dict[str, float] = {}
            num_steps = 0
            for batch_raws in self._iter_batches(dataset, train=False):
                _, (losses, _) = self._forward(
                    [self._params[i].raw_tensor for i in self._train_param_idx],
                    [self._params[i].raw_tensor for i in self._other_param_idx],
                    self._step_raws(batch_raws),
                    self.global_train_step,
                    self.epoch,
                    False,
                )
                _accumulate_losses(losses, sums=accumulated, norms=accumulated_norms)
                num_steps += 1
            scores = {f"{name}_loss_{k}": v / accumulated_norms[k] for k, v in accumulated.items()}
            print(f"epoch {self.epoch} {name}: {_format_scores(scores)}", file=log.v3)
            self.learning_rate_control.set_epoch_error(self.epoch, scores)
            self._on_epoch_end(dataset_name=name)

    def get_model(self) -> rf.Module:
        """
        :return: the model
        """
        return self.model

    def _iter_batches(self, dataset: Dataset, *, train: bool):
        """
        :param dataset:
        :param train: whether this is the train dataset. Affects the batch options read from the config,
            and only the train iterator carries the complete-frac the LR schedule needs.
        :return: iterator over batches as raw dicts of JAX arrays (the templates are filled in
            inside the step, which is what makes the step compilable),
            or over ``(batch, complete_frac)`` when train
        """
        batch_size = (
            self._batch_opts["batch_size"]
            if train
            else self._batch_opts.get("eval_batch_size") or self._batch_opts["batch_size"]
        )
        return iter_dataset_batches(
            dataset,
            extern_data=self.extern_data,
            batch_size=batch_size,
            max_seqs=self._batch_opts["max_seqs"],
            epoch=self.epoch,
            device=self._device,
            with_complete_frac=train,
            as_raws=True,
            static_shapes=self._static_shapes_opts,
            packed_keys=self._packed_keys(),
            time_multiple=self._jit_opts["time_multiple"] if self._jit_opts else 0,
            **{k: v for k, v in self._batch_opts.items() if k not in ("batch_size", "eval_batch_size", "max_seqs")},
        )

    def _report_dev_memory_stats(self):
        """
        Device memory after the epoch, as the PyTorch engine reports it.

        ``peak_bytes_in_use`` is what decides whether a batch size fits;
        a gap to ``bytes_in_use`` (the live set) is the allocator holding freed blocks.
        """
        for dev in jax.local_devices():
            stats = dev.memory_stats()  # None on backends without an allocator (cpu)
            if not stats:
                continue
            parts = [f"dev {dev.id}"]
            for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_reservable_limit"):
                if key in stats:
                    parts.append(f"{key} {stats[key] / (1024**3):.2f} GB")
            if "num_allocs" in stats:
                parts.append(f"num_allocs {stats['num_allocs']}")
            print(f"epoch {self.epoch} device memory: {', '.join(parts)}", file=log.v3)

    def _maybe_stop_for_resubmission(self, last_epoch_wall_sec: float):
        """
        :param last_epoch_wall_sec: how long the epoch just finished took

        With less wall-time left than the (safety-scaled) last epoch needed, stop now:
        SIGINT reaches the Sisyphus worker as KeyboardInterrupt, so the job gets resubmitted,
        instead of the time limit killing an epoch half-way.
        Same as the PyTorch engine, minus torchelastic (we are one process).
        See https://github.com/rwth-i6/returnn/issues/1818.
        """
        import os
        import signal
        from returnn.util.basic import slurm_time_left_sec

        if not self.config.bool("stop_for_resubmission_when_low_time_left", False):
            return
        time_left = slurm_time_left_sec()
        if time_left is None:
            return  # not under SLURM, or the query failed -- nothing to decide on
        safety = self.config.float("stop_for_resubmission_safety_factor", 1.2)
        needed = last_epoch_wall_sec * safety
        if time_left >= needed:
            return
        print(
            f"stop_for_resubmission_when_low_time_left:"
            f" SLURM time_left={time_left}s, last epoch wall={last_epoch_wall_sec:.1f}s,"
            f" needed (x{safety})={needed:.1f}s -- stopping early so sisyphus can resubmit.",
            file=log.v1,
        )
        os.kill(-os.getpgrp(), signal.SIGINT)

    def _bucket_for(self, batch_raws: Dict[str, Any]) -> Dict[str, int]:
        """
        :param batch_raws: one batch
        :return: the first declared bucket the batch fits into
        :raise ValueError: when none does -- the batch cannot be run without compiling a program
            that was not declared, which is exactly what buckets exist to prevent
        """
        shapes = {key: value.shape for key, value in batch_raws.items() if hasattr(value, "shape")}
        for bucket in self._jit_opts["buckets"]:
            if all(
                value.shape[0] <= bucket["batch_dim"]
                and all(
                    value.shape[axis] <= int(bucket[key])
                    for axis, dim in enumerate(self.extern_data.data[key].dims[1:], start=1)
                    if dim.is_dynamic()
                )
                for key, value in batch_raws.items()
                if key in self.extern_data.data and hasattr(value, "shape") and value.ndim
            ):
                return bucket
        raise ValueError(
            f"JAX engine: no jax_jit bucket fits this batch.\n  batch: {shapes}\n  buckets:"
            + "".join(f"\n    {bucket}" for bucket in self._jit_opts["buckets"])
            + "\nDeclare a bucket that covers it, or bound the batching so it cannot occur."
        )

    def _precompile_buckets(self):
        """
        Compile one executable per declared bucket, before training starts.

        A compile costs minutes, so it is paid at startup, not in the middle of an epoch.
        """
        buckets = self._jit_opts.get("buckets")
        if not buckets:
            return
        start = time.time()
        timed: List[Tuple[Dict[str, int], Dict[str, Any]]] = []  # (bucket, its dummy batch), for _time_buckets
        for bucket in buckets:
            # Through the real data path (batch_to_jax_raws + _step_raws), not hand-built arrays:
            # the signature includes device commitment, and hand-built ones matched no real batch.
            raw = {"batch_dim": bucket["batch_dim"]}
            for key, template in self.extern_data.data.items():
                if template.dtype == "string":
                    continue
                shape, has_dyn = [bucket["batch_dim"]], False
                for dim in template.dims[1:]:
                    if dim.is_dynamic():
                        shape.append(int(bucket[key]))
                        has_dyn = True
                    else:
                        shape.append(dim.dimension)
                raw[key] = numpy.zeros(shape, dtype=template.dtype)
                if has_dyn:
                    raw[f"{key}_seq_lens"] = numpy.zeros((bucket["batch_dim"],), dtype="int32")
            raws = self._step_raws(
                batch_to_jax_raws(
                    raw,
                    extern_data=self.extern_data,
                    device=self._device,
                    time_multiple=self._jit_opts["time_multiple"],
                )
            )
            # noinspection PyProtectedMember
            self._run_compiled_step(
                [self._params[i].raw_tensor for i in self._train_param_idx],
                [self._params[i].raw_tensor for i in self._other_param_idx],
                raws,
                self._opt_state,
                self._rng_key if self._rng_key is not None else self._commit_one(JaxBackend._get_rng_key_()),
                jnp.asarray(0.0, dtype=jnp.float32),  # lr 0: this call must not change the model
                jnp.asarray(0, dtype=jnp.int32),
                self.epoch if self._static_argnums else jnp.asarray(self.epoch, dtype=jnp.int32),
                _compile_only=True,
            )
            timed.append((bucket, raws))
        print(
            f"JAX engine: compiled {len(buckets)} bucket programs in {time.time() - start:.1f} sec",
            file=log.v3,
        )
        self._time_buckets(timed)

    def _time_buckets(self, timed: List[Tuple[Dict[str, int], Dict[str, Any]]]):
        """
        :param timed: (bucket, its dummy batch) per compiled program

        Time each compiled program on its own dummy batch, blocking on the result:
        the pure device cost of one step per shape, with no data pipeline and no host overlap.
        Per-shape cost is not recoverable from epoch totals (the bucket counts are collinear),
        and the per-step device wait during training is only a lower bound.

        It cannot disturb training: parameters and optimizer state go in as copies
        (the step donates its input buffers), the learning rate is 0, and the outputs are dropped.
        """
        reps = self.config.int("jax_time_buckets", 2)
        if not reps or not timed:
            return
        rows = []
        for bucket, raws in timed:
            # copies: the compiled step donates args 0, 1 and 3, i.e. it deletes what it is given
            train_raws = [jnp.array(self._params[i].raw_tensor) for i in self._train_param_idx]
            other_raws = [jnp.array(self._params[i].raw_tensor) for i in self._other_param_idx]
            best = None
            for rep in range(reps + 1):  # one warmup, then the timed reps
                # noinspection PyProtectedMember
                args = (
                    [jnp.array(x) for x in train_raws],
                    [jnp.array(x) for x in other_raws],
                    raws,
                    jax.tree_util.tree_map(jnp.array, self._opt_state),
                    self._rng_key if self._rng_key is not None else self._commit_one(JaxBackend._get_rng_key_()),
                    jnp.asarray(0.0, dtype=jnp.float32),  # lr 0: this must not change the model
                    jnp.asarray(0, dtype=jnp.int32),
                    self.epoch if self._static_argnums else jnp.asarray(self.epoch, dtype=jnp.int32),
                )
                t0 = time.time()
                jax.block_until_ready(self._run_compiled_step(*args))
                took = time.time() - t0
                if rep and (best is None or took < best):
                    best = took  # min over reps: the least contaminated by anything else on the device
            volume = bucket["batch_dim"] * max((int(v) for k, v in bucket.items() if k != "batch_dim"), default=1)
            rows.append((bucket, best, volume))
        print(
            "JAX engine: per-shape step cost (pure device, min of"
            f" {reps} reps, no data pipeline):"
            + "".join(
                f"\n  {bucket} {sec * 1000:8.1f} ms  {vol / 1e6:7.1f}M padded  {sec / (vol / 1e6) * 1000:6.2f} ms/M"
                for bucket, sec, vol in rows
            ),
            file=log.v3,
        )

    def _run_compiled_step(self, *args, _compile_only: bool = False):
        """
        :param args: the arguments of one train step
        :param _compile_only: only compile the executable for these arguments, do not run it
            (see :func:`_precompile_buckets`)
        :return: what the step returns, or None when only compiling

        Compiles explicitly (lower + compile) and keeps one executable per input signature,
        so that a compile is a visible, counted event rather than ``jax.jit``'s implicit cache.
        With ``jax_jit`` buckets, an uncompiled signature is an error:
        the buckets are the set of programs, so a shape outside them means the padding failed.
        """
        signature = _step_signature(args)
        compiled = self._compiled_steps.get(signature)
        if compiled is None:
            if self._jit_opts.get("buckets") and not _compile_only:
                raise RuntimeError(
                    f"JAX engine: a batch reached the step with a signature no bucket program covers."
                    f"\n  got: {signature}\nCompiled bucket programs:"
                    + "".join(f"\n  {known}" for known in self._compiled_steps)
                )
            start = time.time()
            compiled = self._jitted_step.lower(*args).compile()
            self._compiled_steps[signature] = compiled
            print(
                f"JAX engine: compiled the step in {time.time() - start:.1f} sec"
                f" (compile #{len(self._compiled_steps)}, for {signature})",
                file=log.v3,
            )
        if _compile_only:
            return None
        # A compiled executable has the static arguments baked in (that is what static means),
        # so they are not passed again,
        # unlike the jitted function, which still takes them.
        return compiled(*[arg for i, arg in enumerate(args) if i not in self._static_argnums])

    def _commit_one(self, raw: Any) -> Any:
        """
        :param raw: a JAX array, or anything else (left alone)
        :return: the same, placed on this engine's device

        Whether an array is committed to a device is part of the input signature,
        so mixing committed and uncommitted parameters compiles the whole step twice.
        """
        if not isinstance(raw, jax.Array):
            return raw
        device = _device_from_str(self._device) if self._device else None
        return jax.device_put(raw, device) if device is not None else raw

    def _commit_to_device(self, raws: List[Any], *, into: Optional[List[rf.Parameter]] = None):
        """
        :param raws: JAX arrays
        :param into: if given, the parameters to write the committed arrays back into
        """
        committed = [self._commit_one(raw) for raw in raws]
        if into is not None:
            for param, raw in zip(into, committed):
                param.raw_tensor = raw

    def _step_raws(self, batch_raws: Dict[str, Any]) -> Dict[str, Any]:
        """
        :param batch_raws: one batch, as the data pipeline yields it
        :return: what the step gets.
            Entries which are not device arrays -- the string ones, i.e. seq_tag --
            cannot be arguments of a compiled function,
            so the compiled step does not get them at all;
            passing them would bake the first batch's values into the trace.
        """
        if self._jit_opts is None:
            return batch_raws
        batch_raws = {key: value for key, value in batch_raws.items() if not _is_host_only(value)}
        if self._jit_opts.get("buckets"):
            batch_raws = pad_raws_to_bucket(
                batch_raws, extern_data=self.extern_data, bucket=self._bucket_for(batch_raws)
            )
        return batch_raws

    def _pack_extern_data(self, extern_data: TensorDict):
        """
        :param extern_data: modified in place

        Repack the padded batch into packed storage (``packed_tensors``),
        in the step rather than in the data pipeline: one gather per key, on the device.
        The dims stay the same, only the storage behind them (``PackedRawTensor``),
        so the config's model code and RF do not have to know.
        """
        if self._packing is None:
            return
        for key, value in list(extern_data.data.items()):
            opts = packed_batch_key_opts(self._packing, key)
            if opts is None:
                continue  # this key opted out, it stays padded
            if len(value.dims) < 2 or not value.dims[1].is_dynamic():
                continue  # nothing to pack
            packed = rf.pack(value, dims=[batch_dim, value.dims[1]])
            new_value = value.copy()
            new_value.raw_tensor = packed.raw_tensor
            extern_data.data[key] = new_value

    def _packed_keys(self) -> List[str]:
        """
        :return: extern-data keys stored packed:
            those `packed_tensors` opts in, with a dynamic spatial dim.
            Empty when packing is off.
        """
        if self._packing is None:
            return []
        keys = []
        for key, value in self.extern_data.data.items():
            if packed_batch_key_opts(self._packing, key) is None:
                continue
            if len(value.dims) < 2 or not value.dims[1].is_dynamic():
                continue
            keys.append(key)
        return keys

    def _import_packed_extern_data(self, extern_data: TensorDict, batch_raws: Dict[str, Any]):
        """
        :param extern_data: modified in place
        :param batch_raws: holds ``<key>:packed``, the flat buffer built on the host

        The seqs arrive already concatenated (see :func:`returnn.jax.data.staticize_raws`),
        so nothing is packed here -- the buffer is imported as packed storage.
        The padded array was never built nor transferred, which is the point.
        """
        # the templates, not the filled dict:
        # a packed key has no padded entry, so fill_extern_data skipped it, dims included
        batch_bound = self._static_shapes_opts["batch_size_bound"]
        capacities = self._static_shapes_opts.get("dim_capacity") or {}
        for key, value in self.extern_data.data.items():
            flat = batch_raws.get(f"{key}:packed")
            if flat is None:
                continue
            value = value.copy_template()
            spatial: Dim = value.dims[1]
            if batch_dim.dyn_size_ext is None:
                batch_name: str = batch_dim.name or "batch"
                batch_dim.dyn_size_ext = Tensor(batch_name, dims=[], dtype="int32")
            # The batch is full at the bound, with zero-length filler seqs,
            # the regime the torch bound path uses (its CTC returns loss 0 for them).
            # Carrying the true count instead would make the batch dim need masking,
            # which ops like search_sorted refuse.
            batch_dim.dyn_size_ext.raw_tensor = jnp.asarray(batch_bound, dtype=jnp.int32)
            batch_dim.capacity = batch_bound
            if spatial.dyn_size_ext is None:
                spatial.dyn_size_ext = Tensor(spatial.name or "time", dims=[batch_dim], dtype="int32")
            spatial.dyn_size_ext.raw_tensor = batch_raws[f"{key}_seq_lens"]
            spatial.capacity = capacities.get(key, spatial.capacity)
            packed_dim = Dim(int(flat.shape[0]), name=f"{spatial.name or key}:packed")
            inner = Tensor(
                f"{key}_packed",
                dims=[packed_dim] + list(value.dims[2:]),
                dtype=value.dtype,
                sparse_dim=value.sparse_dim,
                raw_tensor=flat,
            )
            imported = rf.pack_import(
                inner,
                batch_dim=batch_dim,
                spatial_dim=spatial,
                packed_dim=packed_dim,
                feature_dim=value.feature_dim,
            )
            new_value = value.copy()
            new_value.raw_tensor = imported.raw_tensor
            extern_data.data[key] = new_value

    def _create_model(self, *, epoch: int, step: int):
        """
        :param epoch:
        :param step:
        """
        random_seed = self.config.int("random_seed", 42)
        rf.set_random_seed((epoch * 193939 + step * 19937 + random_seed * 27644437 + 479001599) % (2**31))
        get_model_func = self.config.typed_value("get_model")
        assert get_model_func, "get_model not defined in config"
        sentinel_kw = util.get_fwd_compat_kwargs()
        model = get_model_func(epoch=epoch, step=step, **sentinel_kw)
        assert isinstance(model, rf.Module), f"get_model returned {model!r}, expected an rf.Module"
        self.model = model
        self._param_names = [name for name, _ in model.named_parameters()]
        self._params = [param for _, param in model.named_parameters()]
        # jax_trainable, not param.trainable:
        # the latter is the value as given, and None means not trainable only for aux params.
        # See JaxBackend.set_parameter_trainable.
        # noinspection PyUnresolvedReferences
        self._train_param_idx = [i for i, param in enumerate(self._params) if param.jax_trainable]
        self._other_param_idx = [i for i in range(len(self._params)) if i not in set(self._train_param_idx)]
        self._commit_to_device([param.raw_tensor for param in self._params], into=self._params)
        num_params = sum(int(numpy.prod(p.batch_shape)) for p in self._params)
        print(
            f"net params #: {num_params} ({len(self._train_param_idx)} of {len(self._params)} trainable)", file=log.v2
        )

    def _init_step_func(self):
        """
        Build the differentiable step, and the compiled step around it.
        See the module docstring for why they have this shape.
        """

        def _forward(train_raws, other_raws, batch_raws, step, epoch, train_flag):
            """
            :return: (total loss, (the individual losses, the non-trainable params after the step))
            """
            orig = [p.raw_tensor for p in self._params]
            for idx, raw in zip(self._train_param_idx + self._other_param_idx, list(train_raws) + list(other_raws)):
                self._params[idx].raw_tensor = raw
            try:
                extern_data = fill_extern_data(self.extern_data, batch_raws)
                if self._static_shapes_opts is not None:
                    # the sequences arrived already concatenated, so import instead of pack
                    self._import_packed_extern_data(extern_data, batch_raws)
                else:
                    self._pack_extern_data(extern_data)
                # traced values: wrap them, the run ctx takes an int or a Tensor.
                # The dtype comes from the array, not assumed: a plain Python int becomes int64
                # under x64, which this backend enables.
                if not isinstance(step, int):
                    step = Tensor("step", dims=(), dtype=JaxBackend.get_dtype_name_raw(step), raw_tensor=step)
                if not isinstance(epoch, int):
                    epoch = Tensor("epoch", dims=(), dtype=JaxBackend.get_dtype_name_raw(epoch), raw_tensor=epoch)
                rf.init_train_step_run_ctx(train_flag=train_flag, step=step, epoch=epoch)
                sentinel_kw = util.get_fwd_compat_kwargs()
                # Static shapes need a capacity per dynamic dim,
                # and the derived ones (subsampled time, attention kv)
                # must derive it from the dims they come from.
                with rf.set_static_traceable_ctx(self._jit_opts is not None), rf.set_amp_policy_ctx(self._amp_policy):
                    assert callable(self._train_step_func)
                    self._train_step_func(model=self.model, extern_data=extern_data, **sentinel_kw)
                    run_ctx = rf.get_run_ctx()
                    total = run_ctx.total_loss()
                    # Summed loss and norm factor, not the mean, as the PyTorch engine does it:
                    # averaging per-batch means would weight 200 short seqs like 51 long ones.
                    losses = {
                        name: (loss.get_summed_loss().raw_tensor, _inv_norm_factor_raw(loss))
                        for name, loss in run_ctx.losses.items()
                    }
                # Non-trainable parameters can be written by the step -- rf.BatchNorm's running statistics
                # are the case that matters -- so they leave this function as values.
                # The restore below undoes the write, and under jit it would happen at trace time only.
                new_other = [self._params[idx].raw_tensor for idx in self._other_param_idx]
                total_raw = total.raw_tensor if isinstance(total, Tensor) else total
                return total_raw, (losses, new_other)
            finally:
                for param, raw in zip(self._params, orig):
                    param.raw_tensor = raw

        value_and_grad = jax.value_and_grad(_forward, has_aux=True)

        def _train_step(train_raws, other_raws, batch_raws, opt_state, rng_key, learning_rate, step, epoch):
            """
            One update: forward, gradients, optimizer. The whole thing is one compiled unit,
            so XLA fuses across the model, the losses and the optimizer.
            """
            # The RNG stream in and out, see the module docstring.
            # noinspection PyProtectedMember
            prev_key, JaxBackend._rng_key = JaxBackend._rng_key, rng_key
            try:
                (loss, (losses, other_raws)), grads = value_and_grad(
                    train_raws, other_raws, batch_raws, step, epoch, True
                )
                # noinspection PyProtectedMember
                rng_key = JaxBackend._rng_key
            finally:
                JaxBackend._rng_key = prev_key
            # Pre-clip, like the PyTorch engine reports it: computed before the optimizer runs
            grad_norm = (
                global_grad_norm(grads, p=self._updater.log_grad_norm_p)
                if self._updater.log_grad_norm_p
                else jnp.zeros((), dtype=jnp.float32)
            )
            train_raws, opt_state = self._updater.step(
                params=train_raws, grads=grads, opt_state=opt_state, learning_rate=learning_rate
            )
            return train_raws, other_raws, opt_state, rng_key, loss, losses, grad_norm

        self._forward = _forward
        # Static: only the epoch; the step number would recompile every step.
        # Donated: parameters and optimizer state, so XLA writes into those buffers, not a copy.
        # Donated buffers are deleted on return, so the caller holds exactly one reference each.
        # Not donated: the batch and the RNG key.
        if self._jit_opts is None:
            self._train_step = _train_step
        else:
            self._jitted_step = jax.jit(_train_step, static_argnums=self._static_argnums, donate_argnums=(0, 1, 3))
            # Not the jitted function directly: that compiles implicitly, mid-epoch and silently,
            # and a compile of this step costs minutes. See :func:`_run_compiled_step`.
            self._train_step = self._run_compiled_step
        self._opt_state = self._updater.init([self._params[i].raw_tensor for i in self._train_param_idx])
        self._opt_state = jax.tree_util.tree_map(self._commit_one, self._opt_state)

    def _save_model(self):
        """
        Save the model of the current epoch.
        """
        if not self.model_filename:
            print("No 'model' in the config, not saving.", file=log.v4)
            return
        filename = self.get_epoch_model_filename()
        _checkpoint.save_checkpoint(
            self.model,
            filename + util.get_model_filename_postfix(),
            step=self.global_train_step,
            epoch=self.epoch,
        )
        # The optimizer state goes next to it, as for PyTorch, so that a continued run
        # keeps the moments instead of restarting the optimizer.
        _checkpoint.save_opt_state(self._opt_state, filename + ".opt" + util.get_model_filename_postfix())

    @staticmethod
    def delete_model(filename: str) -> int:
        """
        :param filename: without the postfix
        :return: accumulated file size in bytes of the deleted files

        Used by :func:`EngineBase.cleanup_old_models`.
        A JAX checkpoint is the ``.orbax`` directory plus the optimizer state next to it,
        so this removes trees, not files.
        """
        postfix = util.get_model_filename_postfix()
        count_bytes = 0
        for fname in (filename + postfix, filename + ".opt" + postfix):
            if not os.path.exists(fname):
                continue
            for root, _dirs, files in os.walk(fname):
                count_bytes += sum(os.stat(os.path.join(root, f)).st_size for f in files)
            shutil.rmtree(fname)
        assert count_bytes > 0, f"delete_model: nothing to delete for {filename!r}"
        return count_bytes

    def init_network_from_config(self, config: Optional[Config] = None):
        """
        :param config:

        Init for the forward / search task, i.e. everything :func:`forward_with_callback` needs:
        the model with its checkpoint loaded, and the ``forward_step`` of the config.
        No optimizer and no learning-rate control -- nothing here updates parameters.
        """
        assert config is self.config or config is None
        config = self.config

        from returnn.torch.data.extern_data import extern_data_template_from_config_opts

        self.extern_data = extern_data_template_from_config_opts(config.typed_value("extern_data"))
        self._forward_step_func = config.typed_value("forward_step")
        assert self._forward_step_func, "forward_step not defined in config"
        model_outputs = config.typed_value("model_outputs")
        # The declared outputs, if any:
        # they make a missing mark_as_output an error (run_ctx.check_outputs_complete).
        # Built plainly, as extern_data_template_from_config_opts would add a seq_tag entry.
        self._forward_step_expected_outputs = None
        if model_outputs is not None:
            self._forward_step_expected_outputs = TensorDict()
            self._forward_step_expected_outputs.update(model_outputs, auto_convert=True)

        self.model_filename = config.value("model", None)
        epoch, load_filename = self.get_epoch_model(config)
        self.epoch = epoch or 1
        if self.global_train_step is None:
            self.global_train_step = 0
        self._create_model(epoch=self.epoch, step=self.global_train_step)
        if load_filename:
            self._load_model(filename=load_filename, with_opt_state=False)
            print(f"Loaded model {load_filename} (epoch {epoch})", file=log.v3)
        else:
            print("No model checkpoint to load, using the initial parameters.", file=log.v3)
        self._preload_from_files(is_first_train_epoch=False)
        print(f"JAX engine: forward, device {self._device}, devices {jax.devices()}", file=log.v3)

    def forward_with_callback(
        self,
        *,
        dataset: Dataset,
        callback: ForwardCallbackIface,
        dataset_init_epoch: bool = True,
        allow_skipping_seqs: bool = False,
    ):
        """
        :param dataset: to forward over
        :param callback: gets every sequence's outputs, one at a time
        :param dataset_init_epoch: whether to sort the dataset for us (as the PyTorch engine does)
        :param allow_skipping_seqs: whether min/max_seq_length may drop sequences here

        The step runs eager, also with ``jax_jit`` set for training:
        forward outputs (a beam search above all) have value-dependent lengths,
        and bounding them is the same work as the packed/bucketed training path.
        """
        assert isinstance(dataset, Dataset)
        assert isinstance(callback, ForwardCallbackIface)
        assert self.model is not None, "call init_network_from_config first"

        if dataset_init_epoch and self.config.bool("sort_dataset", True):
            if dataset.seq_ordering != "sorted_reverse" and dataset.supports_seq_order_sorting():
                # reverse, so the largest batch is the first one: it either fits or fails at once
                print("Dataset supports sorting, i.e. it will be sorted for optimal performance.", file=log.v3)
                dataset.seq_ordering = "sorted_reverse"
        if not allow_skipping_seqs:
            for key in ("min_seq_length", "max_seq_length"):
                assert not self._batch_opts.get(key), (
                    f"{key} {self._batch_opts[key]} would DROP sequences from the forward output."
                    f" Set allow_skipping_seqs=True if that is really wanted."
                )

        from returnn.torch.data.extern_data import get_batch_dim_from_extern_data

        batch_dim_ = get_batch_dim_from_extern_data(self.extern_data)
        report_prefix = f"ep {self.epoch} {dataset.name} forward"
        start_time = time.time()
        compute_time = 0.0
        step_idx = 0
        callback.init(model=self.model)
        for batch_raws in self._iter_batches(dataset, train=False):
            step_begin = time.time()
            if self._forward_step_expected_outputs is not None:
                # also resets the dyn dims the previous step set on them
                self._forward_step_expected_outputs.reset_content()
            outputs = self._forward_pass(batch_raws)
            # One host read per step: the sizes of the output dims. Everything below indexes
            # per sequence, which needs them as numbers anyway.
            for batch_idx in range(batch_dim_.get_dim_value()):
                # seq_tag is a numpy string array in the raw dict, not a Tensor
                # noinspection PyUnresolvedReferences
                seq_tag = batch_raws["seq_tag"][batch_idx]
                outputs_per_seq = TensorDict()
                for key, value in outputs.data.items():
                    outputs_per_seq.data[key] = _tensor_of_seq_numpy(value, batch_idx=batch_idx, batch_dim=batch_dim_)
                callback.process_seq(seq_tag=seq_tag, outputs=outputs_per_seq)
            compute_time += time.time() - step_begin
            step_idx += 1
        callback.finish()

        elapsed = time.time() - start_time
        print(
            f"{report_prefix}: {step_idx} steps, {util.hms(elapsed)} elapsed"
            f" ({compute_time / elapsed * 100.0:.1f}% computing time)",
            file=log.v3,
        )
        self._report_dev_memory_stats()

    def _forward_pass(self, batch_raws: Dict[str, Any]) -> TensorDict:
        """
        :param batch_raws: one batch, as the data pipeline yields it
        :return: what the step marked as output, still batched, raw tensors on the device

        Same parameter binding as the train step, without the gradient transform around it.
        """
        orig = [p.raw_tensor for p in self._params]
        try:
            extern_data = fill_extern_data(self.extern_data, batch_raws)
            rf.init_forward_step_run_ctx(
                expected_outputs=self._forward_step_expected_outputs, step=self.global_train_step, epoch=self.epoch
            )
            sentinel_kw = util.get_fwd_compat_kwargs()
            with rf.set_amp_policy_ctx(self._amp_policy):
                assert callable(self._forward_step_func)
                self._forward_step_func(model=self.model, extern_data=extern_data, **sentinel_kw)
            run_ctx = rf.get_run_ctx()
            run_ctx.check_outputs_complete()
            return run_ctx.outputs
        finally:
            for param, raw in zip(self._params, orig):
                param.raw_tensor = raw

    def _preload_from_files(self, *, is_first_train_epoch: bool):
        """
        :param is_first_train_epoch: whether no checkpoint was loaded, i.e. training starts fresh

        ``preload_from_files``: initialize (parts of) the model from other checkpoints.
        Same option shape and reversed-sorted order as the PyTorch engine.
        Per entry: ``filename`` (an ``.orbax`` directory or a PyTorch ``.pt``),
        ``prefix``, ``init_for_train``, ``ignore_missing``,
        ``ignore_params`` / ``ignore_params_prefixes``.
        """
        opts_dict = self.config.typed_value("preload_from_files", None)
        if not opts_dict:
            return
        params = dict(self.model.named_parameters())
        for key, opts in reversed(sorted(opts_dict.items())):
            if not isinstance(opts, dict) or "filename" not in opts:
                raise ValueError(f"preload_from_files {key!r}: expected a dict with 'filename', got {opts!r}")
            init_for_train = opts.get("init_for_train", False)
            if init_for_train:
                # "always" also on a continued run; True only when starting fresh
                if init_for_train != "always" and not is_first_train_epoch:
                    continue
            else:
                continue  # for recognition; this engine only trains so far
            filename = opts["filename"]
            print(f"Pre-load weights for key {key!r} from {filename}", file=log.v3)
            if filename.endswith(".pt"):
                loaded = _checkpoint.load_torch_checkpoint(filename)
            else:
                loaded = _checkpoint.load_checkpoint(filename)

            prefix = opts.get("prefix", "")
            if prefix:
                loaded = {name[len(prefix) :]: v for name, v in loaded.items() if name.startswith(prefix)}
            ignore = set(opts.get("ignore_params", ()))
            ignore_prefixes = tuple(opts.get("ignore_params_prefixes", ()))
            if ignore or ignore_prefixes:
                loaded = {
                    name: v for name, v in loaded.items() if name not in ignore and not name.startswith(ignore_prefixes)
                }

            missing = [name for name in params if name not in loaded]
            if missing and not opts.get("ignore_missing", False):
                raise ValueError(
                    f"preload_from_files {key!r}: {len(missing)} parameter(s) not in {filename}:"
                    f" {missing[:10]}{' ...' if len(missing) > 10 else ''}."
                    f" Set ignore_missing=True to initialize them normally instead."
                )
            _checkpoint.set_model_params(self.model, loaded, allow_missing=True)
            print(f"  loaded {len(loaded)} parameter(s), {len(missing)} left at their init", file=log.v3)
        # the params were replaced in place; re-commit so the step sees them on the device
        self._commit_to_device([p.raw_tensor for p in self._params], into=self._params)

    def _load_model(self, *, filename: str, with_opt_state: bool = True):
        """
        :param filename: without the ``.orbax`` postfix, as :func:`EngineBase.get_epoch_model` returns it
        :param with_opt_state: whether to also restore the optimizer state next to it.
            False for forward / search, which have no optimizer to restore it into.
        """
        postfix = util.get_model_filename_postfix()
        if filename.endswith(postfix):
            filename = filename[: -len(postfix)]
        _checkpoint.set_model_params(self.model, _checkpoint.load_checkpoint(filename + postfix))
        print(f"Loaded model {filename + postfix}", file=log.v3)
        self._commit_to_device([p.raw_tensor for p in self._params], into=self._params)
        opt_filename = filename + ".opt" + postfix
        if with_opt_state and os.path.exists(opt_filename):
            self._opt_state = _checkpoint.load_opt_state(self._opt_state, opt_filename)
            print(f"Loaded optimizer state {opt_filename}", file=log.v3)


def _bucket_key(step_raws: Dict[str, Any]) -> Tuple[int, ...]:
    """
    :param step_raws: what goes into the step, after padding to a bucket
    :return: the shapes it runs on, flattened, as a hashable key

    The shapes after bucket padding, i.e. what the compiled program actually sees -- which is also
    what selects the executable, so one key is one program.
    """
    key: List[int] = []
    for name in sorted(step_raws):
        value = step_raws[name]
        if hasattr(value, "shape"):
            key.extend(int(d) for d in value.shape)
    return tuple(key)


def _device_description() -> str:
    """
    :return: what the run computes on, e.g. "NVIDIA H100 80GB HBM3 (cuda)"

    ``device_kind`` is the GPU model;
    ``rf.get_default_device()`` only says "gpu",
    which is not enough to tell whether two runs shared hardware
    -- and a cross-node comparison that nobody noticed was cross-node is worthless.
    """
    devices = jax.local_devices()
    if not devices:
        return "none"
    dev = devices[0]
    kind = getattr(dev, "device_kind", None) or dev.platform
    return f"{kind} ({dev.platform})" + (f" x{len(devices)}" if len(devices) > 1 else "")


# noinspection PyShadowingNames
def _tensor_of_seq_numpy(x: Tensor, *, batch_idx: int, batch_dim: Dim) -> Tensor:
    """
    :param x: batched, with ``batch_dim`` among its dims
    :param batch_idx: which sequence
    :param batch_dim:
    :return: that one sequence, as a NumPy tensor, padding cut off, without the batch dim

    NumPy and not a JAX array:
    the callback interface is backend-neutral
    -- the same callback is used by the PyTorch and TF engines --
    so what it gets must not be a device array.
    """
    if batch_dim not in x.dims:
        raise Exception(f"Expected {batch_dim} in {x}.")
    if x.dims.index(batch_dim) != 0:
        x = x.copy_move_axis(x.dims.index(batch_dim), 0)

    kwargs = x.copy_template_excluding_axis(0).get_kwargs()
    kwargs["dims"] = [_dim_of_seq(dim, batch_idx=batch_idx, batch_dim=batch_dim) for dim in kwargs["dims"]]
    y = Tensor(**kwargs)

    raw = numpy.asarray(x.raw_tensor)
    # a scalar per sequence stays an ndarray, so the callback sees the same type either way
    raw = raw[batch_idx] if x.batch_ndim > 1 else raw[batch_idx : batch_idx + 1].reshape(())
    if any(d is not d_ for d, d_ in zip(x.dims[1:], y.dims)):  # any dim replaced above?
        raw = raw[tuple(slice(None, dim.get_dim_value()) for dim in y.dims)]  # cut the padding
    y.raw_tensor = raw
    return y


# noinspection PyShadowingNames
def _dim_of_seq(dim: Dim, *, batch_idx: int, batch_dim: Dim) -> Dim:
    """
    :param dim:
    :param batch_idx:
    :param batch_dim:
    :return: the dim as it applies to one sequence

    A dynamic dim's size is itself a tensor over the batch (``[B]``), and the callback gets one
    sequence at a time, so that size has to lose its batch dim too.
    """
    if dim.dyn_size_ext is None or batch_dim not in dim.dyn_size_ext.dims:
        return dim
    new_dim = dim.copy()
    new_dim.dyn_size_ext = _tensor_of_seq_numpy(dim.dyn_size_ext, batch_idx=batch_idx, batch_dim=batch_dim)
    return new_dim


def _static_shapes_opts_from_config(config: Config) -> Optional[Dict[str, Any]]:
    """
    :param config:
    :return: ``jax_static_shapes``: ``batch_size_bound`` (int),
        and optionally ``dim_capacity`` and ``packed_total_bound``, both per data key.

    Same option names as the TF engine's ``tf_static_shapes``
    and the PyTorch engine's ``torch_cuda_graph``,
    so one set of declared bounds carries across backends.
    """
    opts = config.typed_value("jax_static_shapes", None)
    if not opts:
        return None
    assert isinstance(opts, dict), f"jax_static_shapes: expected a dict, got {opts!r}"
    allowed = {"batch_size_bound", "dim_capacity", "packed_total_bound"}
    unknown = set(opts) - allowed
    assert not unknown, f"jax_static_shapes: unknown options {sorted(unknown)}, allowed {sorted(allowed)}"
    assert isinstance(opts.get("batch_size_bound"), int), f"jax_static_shapes: batch_size_bound required, got {opts!r}"
    return opts


def _batch_opts_from_config(config: Config) -> Dict[str, Any]:
    """
    :param config:
    :return: the batching options this engine supports
    """
    packed_batch_size = config.typed_value("packed_batch_size", None)
    opts = {
        # With packed_batch_size the content is what is budgeted,
        # and a padded budget on top of it would cut batches short for no reason,
        # so batch_size defaults to unlimited there.
        "batch_size": config.typed_value("batch_size", None)
        or (None if packed_batch_size is not None else config.int("batch_size", 10000)),
        "eval_batch_size": config.typed_value("eval_batch_size", None),
        "max_seqs": config.int("max_seqs", -1),
    }
    if packed_batch_size is not None:
        opts["packed_batch_size"] = packed_batch_size
    # Further options of the shared batching layer (Dataset.generate_batches), passed through as given.
    for key in ("max_seq_length", "min_seq_length", "max_pad_size", "max_total_num_seqs", "seq_drop"):
        value = config.typed_value(key, None)
        if value is not None:
            opts[key] = value
    return opts


# Config options which other engines implement and this one does not (yet),
# mapped to the value at which they are a no-op.
# Rejected rather than ignored: silently dropping e.g. accum_grad_multiple_step
# changes what the config means.
_UnsupportedConfigOpts = {
    "apply_cleanup_old_models_to_optim_states": False,
    "calculate_exp_loss": False,
    "chunking": None,
    "min_chunk_size": None,
    "debug_shell_before_train_loop": False,
    "default_float_dtype": None,
    "forward_auto_split_batch_on_oom": False,
    "grad_scaler": None,
    "load_model_post_hooks": None,
    "online_shuffle_batches": None,
    "pretrain": None,
    "reset_dev_memory_caches": False,
    "tensorboard_opts": None,
    "use_tensorboard": False,
    # backend-specific options are named after the backend, as `torch_...` is on PyTorch
    "jax_distributed": None,
    "jax_log_memory_usage": False,
    "jax_profile": None,
}

# PyTorch-specific options, with the JAX name they correspond to (None: no equivalent).
# A config copied from a PyTorch setup carries these, and ignoring them silently
# -- `torch_amp` above all -- is exactly what this check exists to prevent.
_TorchOnlyConfigOpts = {
    "torch_amp": "jax_amp",  # implemented, but under its own name
    "torch_cuda_graph": "jax_jit",
    "torch_dataloader_opts": None,  # this engine uses the shared batching, which has no worker pool
    "torch_distributed": "jax_distributed",
    "torch_log_memory_usage": "jax_log_memory_usage",
    "torch_profile": "jax_profile",
}


def _check_config_opts_supported(config: Config):
    """
    :param config:
    :raise NotImplementedError: if the config sets an option this engine would ignore
    """

    # noinspection PyShadowingNames
    def _value_if_set(key: str, noop_value: Any = None) -> Optional[Any]:
        """:return: the configured value, or None if unset or at its no-op value"""
        if not config.has(key):
            return None
        value = config.typed_value(key, None)
        if value is None:  # not in the typed dict, e.g. an old-style config file
            value = config.value(key, None)
        if value is None or value is False or value in ((), [], {}, ""):
            return None
        if value == noop_value or value == str(noop_value):
            return None
        return value

    unsupported = []
    for key, noop_value in sorted(_UnsupportedConfigOpts.items()):
        value = _value_if_set(key, noop_value)
        if value is not None:
            unsupported.append(f"{key} = {value!r}")
    for key, jax_name in sorted(_TorchOnlyConfigOpts.items()):
        value = _value_if_set(key)
        if value is not None:
            hint = ""
            if jax_name:
                hint = f", the JAX engine reads {jax_name}"
                if jax_name in _UnsupportedConfigOpts:
                    hint += " (not implemented either)"
            unsupported.append(f"{key} = {value!r} is PyTorch specific" + hint)
    if unsupported:
        raise NotImplementedError(
            "JAX engine: the config sets options which this engine does not implement:\n  "
            + "\n  ".join(unsupported)
            + "\nThey would otherwise be ignored silently, which would change what the config means."
        )


def _check_time_multiple(time_multiple: Union[int, Dict[str, int]], *, extern_data: TensorDict):
    """
    :param time_multiple: as configured, see :func:`returnn.jax.data.batch_to_jax_raws`
    :param extern_data: templates
    :raise NotImplementedError: when it cannot be applied unambiguously

    One number for all keys is ambiguous when the keys have different axes
    (the multiple is in the unit of the axis it pads),
    and different numbers for keys sharing a dim would need two capacities for that dim.
    """
    dim_of_key = {}
    for key, data in extern_data.data.items():
        dyn = [dim for i, dim in enumerate(data.dims) if i > 0 and dim.is_dynamic()]
        if dyn:
            dim_of_key[key] = dyn[0]
    if not dim_of_key:
        return
    if not isinstance(time_multiple, dict):
        if time_multiple > 1 and len({id(dim) for dim in dim_of_key.values()}) > 1:
            raise NotImplementedError(
                f"JAX engine: jax_jit time_multiple {time_multiple} is one number for the keys"
                f" {sorted(dim_of_key)}, whose axes are different dims and different units."
                f" Give it per key, e.g. {{{', '.join(repr(k) + ': ...' for k in sorted(dim_of_key))}}}."
            )
        return
    unknown = set(time_multiple) - set(extern_data.data)
    if unknown:
        raise NotImplementedError(f"JAX engine: jax_jit time_multiple for unknown data keys {sorted(unknown)}")
    by_dim = {}
    for key, dim in dim_of_key.items():
        by_dim.setdefault(id(dim), []).append((key, time_multiple.get(key, 0)))
    for entries in by_dim.values():
        if len({multiple for _, multiple in entries}) > 1:
            raise NotImplementedError(
                f"JAX engine: jax_jit time_multiple differs for keys sharing one dim: {sorted(entries)}."
                f" They are padded to the same extent, so the dim cannot have two capacities."
            )


def _step_signature(args) -> Any:
    """
    :param args: the arguments of one step
    :return: a hashable key holding everything a compiled executable is specialized on:
        the pytree structure,
        and per array its shape, dtype and whether it is committed to a device
        (which is part of the signature too --
        an uncommitted array and a committed one compile separately).
    """
    leaves, structure = jax.tree_util.tree_flatten(args)
    return (
        str(structure),
        tuple(
            (tuple(leaf.shape), str(leaf.dtype), bool(getattr(leaf, "committed", False)))
            if isinstance(leaf, jax.Array)
            else (type(leaf).__name__, repr(leaf) if isinstance(leaf, (int, float, bool, str)) else None)
            for leaf in leaves
        ),
    )


def _prefetch(iterable, *, buffer_size: int):
    """
    :param iterable: the batch iterator
    :param buffer_size: how many batches to run ahead; 0 disables
    :return: the same items, produced by a background thread

    The step is issued asynchronously, so the host is free while the device works,
    but only if something else builds the next batch.
    A thread, not a process: the dataset reads and the array conversion release the GIL.
    """
    if not buffer_size:
        yield from iterable
        return

    queue: "_queue.Queue" = _queue.Queue(maxsize=buffer_size)
    end = object()

    # noinspection PyShadowingNames
    def _produce():
        try:
            for item in iterable:
                queue.put(item)
        except BaseException as exc:  # noqa: BLE001  # re-raised in the consumer below
            queue.put(exc)
        else:
            queue.put(end)

    thread = _threading.Thread(target=_produce, name="jax-data-prefetch", daemon=True)
    thread.start()
    while True:
        item = queue.get()
        if item is end:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def _batch_size_info(batch_raws: Dict[str, Any]) -> Dict[str, Any]:
    """
    :param batch_raws: one batch, as the data pipeline yields it
    :return: num_seqs, and per data key the max and the summed (content) seq len

    The PyTorch engine's ``log_batch_size`` keys, so the throughput tooling parses either log.
    Sums stay device scalars:
    they are read with the step's losses, so they cost no extra sync.
    """
    info: Dict[str, Any] = {}
    for key, value in sorted(batch_raws.items()):
        if not key.endswith("_seq_lens"):
            continue
        name = key[: -len("_seq_lens")]
        data = batch_raws.get(name)
        if data is not None and getattr(data, "ndim", 0) > 1:
            info.setdefault("num_seqs", int(value.shape[0]))
            info[f"max_size:{name}"] = int(data.shape[1])
            # the content, i.e. what a packed step would compute on; num_seqs * max_size - this is
            # exactly the padding the bucket regime adds
            info[f"sum_size:{name}"] = jnp.sum(value)
            continue
        packed = batch_raws.get(f"{name}:packed")
        if packed is None or not getattr(packed, "ndim", 0):
            continue
        # packed: one flat buffer, so its length is the padded extent.
        # The lens are zero-padded to the batch bound,
        # so the non-zero count is the real seq count.
        info.setdefault("num_seqs", jnp.count_nonzero(value))
        info[f"max_size:{name}"] = int(packed.shape[0])
        info[f"sum_size:{name}"] = jnp.sum(value)
    return info


def _inv_norm_factor_raw(loss: Any) -> jax.Array:
    """
    :param loss: a :class:`LossHolder` of the run ctx
    :return: its normalization factor as a device scalar

    A loss without dynamic axes yields a plain int, which the compiled step would bake in.
    """
    inv_norm = loss.get_inv_norm_factor()
    if isinstance(inv_norm, Tensor):
        return inv_norm.raw_tensor
    return jnp.asarray(inv_norm, dtype=jnp.float32)


def _accumulate_losses(losses: Dict[str, Any], *, sums: Dict[str, float], norms: Dict[str, float]) -> None:
    """
    :param losses: of one step, each a (summed loss, normalization factor) pair
    :param sums: accumulated summed losses, updated in place
    :param norms: accumulated normalization factors, updated in place
    """
    for name, (loss_sum, inv_norm) in losses.items():
        sums[name] = sums.get(name, 0.0) + float(loss_sum)
        norms[name] = norms.get(name, 0.0) + float(inv_norm)


def _format_step_log(epoch: int, losses: Dict[str, Any], extra: Dict[str, Any]) -> str:
    """
    :param epoch:
    :param losses: of that step, each a (summed loss, normalization factor) pair
    :param extra: ``step`` plus whatever diagnostics were collected
    :return: one log line, in the same shape as the PyTorch engine's per-step line
    """
    trailing = ("step", "mem_usage", "sec_per_step", "start_elapsed")
    parts = [f"ep {epoch} train, step {extra['step']}"]
    parts += [f"{name} {float(loss_sum) / float(inv_norm):.3f}" for name, (loss_sum, inv_norm) in losses.items()]
    parts += [
        f"{name} {int(value) if name.startswith(('num_seqs', 'max_size', 'sum_size')) else float(value):.0f}"
        if name.startswith(("num_seqs", "max_size", "sum_size"))
        else f"{name} {float(value):.3f}"
        for name, value in extra.items()
        if name not in trailing
    ]
    # spelled and ordered like the torch engine's line, so one parser reads either log
    if extra.get("mem_usage") is not None:
        label, peak = extra["mem_usage"]
        parts.append(f"mem_usage:{label} {util.human_bytes_size(peak)}")
    if extra.get("sec_per_step") is not None:
        parts.append(f"{extra['sec_per_step']:.3f} sec/step")
    if extra.get("start_elapsed") is not None:
        parts.append(f"elapsed {util.hms(extra['start_elapsed'])}")
    return ", ".join(parts)


def _device_peak_bytes() -> Optional[Tuple[str, int]]:
    """
    :return: (device label, peak bytes in use), or None where the platform does not report it

    The counterpart of torch's ``max_memory_allocated``, labelled as the torch engine labels it
    (RF calls the JAX GPU device "cuda" as well), so one parser reads either log.
    A host-side counter read, no device sync.
    """
    try:
        dev = jax.local_devices()[0]
        stats = dev.memory_stats()
    except (RuntimeError, IndexError, AttributeError):
        return None
    if not stats or stats.get("peak_bytes_in_use") is None:
        return None
    return ("cuda" if dev.platform == "gpu" else dev.platform), stats["peak_bytes_in_use"]


def _is_host_only(value: Any) -> bool:
    """
    :param value:
    :return: whether this is data which stays on the host, i.e. is no device array.
        Only the string entries are (JAX has no string arrays, see :func:`batch_to_jax_raws`).
    """
    return isinstance(value, numpy.ndarray) and value.dtype.kind in ("U", "S", "O")


def _jit_opts_from_config(config: Config) -> Optional[Dict[str, Any]]:
    """
    :param config: ``jax_jit``: True, or the options as a dict
    :return: the options of the compiled step, or None when it is off
    """
    opts = config.typed_value("jax_jit", None)
    if opts is None:
        opts = config.bool("jax_jit", False)
    if not opts:
        return None
    if opts is True:
        opts = {}
    if not isinstance(opts, dict):
        raise TypeError(f"JAX engine: expected jax_jit True or a dict, got {opts!r}")
    opts = dict(opts)
    # A compiled step is specialized per input shape,
    # and the padded time extent of a batch is different almost every time,
    # so without this every step would trigger a compile.
    time_multiple = opts.pop("time_multiple", 0)
    if not isinstance(time_multiple, dict):
        time_multiple = int(time_multiple)
    else:
        time_multiple = {key: int(value) for key, value in time_multiple.items()}
    buckets = opts.pop("buckets", None)
    if buckets is not None:
        if not isinstance(buckets, (list, tuple)) or not all(isinstance(b, dict) for b in buckets):
            raise TypeError(f"JAX engine: jax_jit buckets must be a list of dicts, got {buckets!r}")
        buckets = [{key: int(value) for key, value in bucket.items()} for bucket in buckets]
    res = {
        "time_multiple": time_multiple,
        "buckets": buckets,
        # only for configs whose step branches on the epoch in Python: it costs a full recompile
        # of every input shape at every epoch
        "epoch_static": bool(opts.pop("epoch_static", False)),
    }
    if opts:
        raise NotImplementedError(f"JAX engine: jax_jit options not supported: {sorted(opts)}")
    return res


def _load_learning_rate_control(config: Config):
    """
    :param config:
    :return: the learning-rate control, shared with the other engines
    """
    from returnn.learning_rate_control import load_learning_rate_control_from_config

    return load_learning_rate_control_from_config(config)


def _format_scores(scores: Dict[str, float]) -> str:
    """
    :param scores:
    :return: one line, for the log
    """
    return ", ".join(f"{name} {value:.5f}" for name, value in sorted(scores.items()))
