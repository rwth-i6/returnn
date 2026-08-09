"""
Engine for the pure TF backend of the RETURNN frontend (``backend = "tensorflow"``).

The RF-side API is the same as for the other backends
(``get_model`` / ``train_step`` in the config, losses via ``rf.get_run_ctx().mark_as_loss``),
so a config that trains on PyTorch trains here as well.

What differs is that TF runs in graph mode (``returnn/tf/compat.py`` disables eager):
the whole step -- extern data, model, losses, gradients, optimizer -- is built ONCE as a graph,
and every step is a ``session.run`` of it with the batch fed into placeholders.
That is why this engine builds nothing per step, and why it is not a copy of the PyTorch engine.

It deliberately does not build a :class:`returnn.tf.network.TFNetwork`:
that class is the net-dict model representation (layers, losses, params),
all of which the RF path has itself. The parts of the TF stack which are not net-dict specific
are reused as they are: the :class:`returnn.tf.updater.Updater`, the batching
(:func:`returnn.engine.batch.batch_to_raw_dict`) and the learning-rate control.
"""

from __future__ import annotations
from typing import Optional, Any, Dict, List, Tuple
import copy
import inspect
import os
import time

import numpy
import tensorflow as tf

from returnn.config import Config
from returnn.datasets.basic import Dataset, init_dataset
from returnn.engine.base import EngineBase
from returnn.forward_iface import ForwardCallbackIface
from returnn.engine.batch import batch_to_raw_dict
from returnn.log import log
from returnn.tensor import Tensor, TensorDict, Dim, batch_dim
from returnn.util import basic as util
import returnn.frontend as rf
import returnn.tf.compat as tf_compat
import returnn.tf.util.basic as tf_util
from returnn.tf.updater import Updater
from returnn.tf.frontend_low_level import TFBackend


__all__ = ["Engine"]


class Engine(EngineBase):
    """
    TF engine for RF models: training and evaluation.
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
        self.session: Optional[tf_compat.v1.Session] = None
        self._graph: Optional[tf.Graph] = None
        self._updater: Optional[Updater] = None
        self._train_step_func = None
        self._batch_opts = _batch_opts_from_config(config)
        self._save_model_epoch_interval = config.int("save_interval", 1)
        self._amp_policy = _amp_policy_from_config(config)
        self._log_batch_size = config.bool("log_batch_size", False)
        self._log_memory_usage = config.bool("tf_log_memory_usage", False)
        self._extra_fetches: Dict[str, tf.Tensor] = {}  # per step, for the log only
        self._loss: Optional[tf.Tensor] = None  # the objective, per step
        self._losses: Dict[str, tf.Tensor] = {}  # per-loss mean, for the log
        self._optim_op: Optional[tf.Operation] = None
        self._global_train_step_var: Optional[tf.Variable] = None
        self._train_flag: Optional[tf.Tensor] = None  # fed False for eval
        self._dyn_lr_func: Optional[Any] = None  # dynamic_learning_rate, applied per step
        self._forward_step_func = None
        self._forward_outputs: Optional[TensorDict] = None  # templates of the marked outputs
        self._forward_fetches: Dict[str, tf.Tensor] = {}
        self._forward_dim_fetches: Dict[Dim, str] = {}  # dyn dim -> its key in _forward_fetches
        self._step_placeholder: Optional[tf.Tensor] = None
        self._saver: Optional[tf_compat.v1.train.Saver] = None
        self._data_keys: List[str] = []
        self._fed_dims: List[Tuple[str, Dim]] = []  # (data key, dim) per fed dyn-size placeholder

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
        # Extra eval datasets named in the config, e.g. "devtrain". Same as the torch engine reads them.
        for dataset_name, dataset_opts in (config.typed_value("eval_datasets", None) or {}).items():
            self.eval_datasets[dataset_name] = init_dataset(dataset_opts, default_kwargs={"name": dataset_name})
        self.learning_rate_control = _load_learning_rate_control(config)

        from returnn.torch.data.extern_data import extern_data_template_from_config_opts

        self.extern_data = extern_data_template_from_config_opts(config.typed_value("extern_data"))
        self._train_step_func = config.typed_value("train_step")
        assert self._train_step_func, "train_step not defined in config"

        _check_config_opts_supported(config)
        self.model_filename = config.value("model", None)
        self.epoch = self.get_train_start_epoch(config)
        if self.global_train_step is None:
            self.global_train_step = 0

        self._graph = tf_compat.v1.Graph()
        with self._graph.as_default():
            self.session = tf_compat.v1.Session(graph=self._graph)
            self._create_placeholders()
            self._create_model(epoch=self.epoch, step=self.global_train_step)
            # The step graph is built once, so mixed precision is a property of the BUILD here,
            # not of a step: the casts end up in the graph and every session.run uses them.
            with rf.set_amp_policy_ctx(self._amp_policy):
                self._init_step_func()
            self.session.run(tf_compat.v1.global_variables_initializer())
            self._updater.init_optimizer_vars(self.session)
        # Continue from an existing checkpoint if there is one (or one was configured via `load`).
        # Without this the engine would silently restart from scratch and overwrite it.
        load_epoch, load_filename = self.get_epoch_model(config)
        if load_filename:
            self._load_model(filename=load_filename)
            print(f"Continuing from epoch {load_epoch} ({load_filename})", file=log.v3)
        print(f"TF engine: starting at epoch {self.epoch}", file=log.v3)

    def train(self):
        """
        Train for the configured number of epochs.
        """
        assert self.train_dataset is not None, "no train dataset"
        final_epoch = self.config_get_final_epoch(self.config)
        while self.epoch <= final_epoch:
            self.init_train_epoch()
            self.train_epoch()
            self.epoch += 1
        print("Finished training.", file=log.v3)

    def init_train_epoch(self):
        """
        Learning rate for this (sub)epoch.
        """
        self.learning_rate = self.learning_rate_control.get_learning_rate_for_epoch(self.epoch)
        self._updater.set_learning_rate(self.learning_rate, session=self.session)

    def train_epoch(self):
        """
        One epoch over the train dataset, then save the model.
        """
        print(f"start epoch {self.epoch} with learning rate {self.learning_rate} ...", file=log.v3)
        start_time = time.time()
        accumulated: Dict[str, float] = {}
        num_steps = 0
        fetches = {"loss": self._loss, "optim": self._optim_op}
        fetches.update({f"loss:{name}": value for name, value in self._losses.items()})
        fetches.update(self._extra_fetches)

        for feed_dict, complete_frac, _ in self._iter_batches(self.train_dataset, train=True):
            feed_dict[self._step_placeholder] = self.global_train_step
            if self._dyn_lr_func is not None:
                self._updater.set_learning_rate(
                    float(
                        self._dyn_lr_func(
                            global_train_step=self.global_train_step,
                            epoch=self.epoch,
                            epoch_continuous=(self.epoch - 1 + complete_frac) if complete_frac is not None else None,
                            learning_rate=self.learning_rate,
                            **util.get_fwd_compat_kwargs(),
                        )
                    ),
                    session=self.session,
                )
            res = self.session.run(fetches, feed_dict=feed_dict)
            # the extra fetches are diagnostics of the step, not scores of the epoch
            scores = {k: float(v) for k, v in res.items() if k != "optim" and k not in self._extra_fetches}
            for key, value in scores.items():
                accumulated[key] = accumulated.get(key, 0.0) + value
            if log.verbose[5]:
                info = [f"ep {self.epoch} train, step {num_steps}"]
                info += [f"{k} {v:.5f}" for k, v in sorted(scores.items())]
                info += [_format_extra_fetch(k, res[k]) for k in sorted(self._extra_fetches)]
                if self._log_batch_size:
                    info += [f"{k} {v}" for k, v in self._batch_size_info(feed_dict).items()]
                if complete_frac is not None:
                    info += [f"complete {complete_frac * 100:.2f}%"]
                print(", ".join(info), file=log.v5)
            num_steps += 1
            self.global_train_step += 1

        assert num_steps > 0, f"no data in epoch {self.epoch}"
        scores = {key: value / num_steps for key, value in accumulated.items()}
        print(
            f"epoch {self.epoch} score: {_format_scores(scores)},"
            f" {num_steps} steps, {time.time() - start_time:.1f} sec",
            file=log.v3,
        )
        self.learning_rate_control.set_epoch_error(self.epoch, {f"train_{k}": v for k, v in scores.items()})
        self.learning_rate_control.save()
        if self.epoch % self._save_model_epoch_interval == 0 or self.epoch == self.config_get_final_epoch(self.config):
            self._save_model()
        if self.config.bool_or_other("cleanup_old_models", None):
            self.cleanup_old_models()
        for name, dataset in self.eval_datasets.items():
            self.eval_model(name, dataset)
        self._maybe_stop_for_resubmission(time.time() - start_time)

    def eval_model(self, name: str, dataset: Dataset):
        """
        :param name: e.g. "dev"
        :param dataset:
        """
        accumulated: Dict[str, float] = {}
        num_steps = 0
        fetches = {"loss": self._loss}
        fetches.update({f"loss:{key}": value for key, value in self._losses.items()})
        for feed_dict, _, _ in self._iter_batches(dataset, train=False):
            feed_dict[self._train_flag] = False  # no dropout etc. in eval
            feed_dict[self._step_placeholder] = self.global_train_step
            res = self.session.run(fetches, feed_dict=feed_dict)
            for key, value in res.items():
                accumulated[key] = accumulated.get(key, 0.0) + float(value)
            num_steps += 1
        if not num_steps:
            return
        scores = {key: value / num_steps for key, value in accumulated.items()}
        print(f"{name} epoch {self.epoch} score: {_format_scores(scores)}", file=log.v3)
        self.learning_rate_control.set_epoch_error(self.epoch, {f"{name}_{k}": v for k, v in scores.items()})
        self.learning_rate_control.save()

    def init_network_from_config(self, config: Optional[Config] = None):
        """
        :param config:

        Build the graph for the "forward" task, the way :func:`init_train_from_config` does for
        training. Named after the net-dict engine's method because :mod:`returnn.__main__`
        calls it by that name for this task.
        """
        assert config is self.config or config is None
        config = self.config

        from returnn.torch.data.extern_data import extern_data_template_from_config_opts

        self.extern_data = extern_data_template_from_config_opts(config.typed_value("extern_data"))
        self._forward_step_func = config.typed_value("forward_step")
        assert self._forward_step_func, "forward_step not defined in config"
        _check_config_opts_supported(config)
        self.model_filename = config.value("model", None)
        load_epoch, load_filename = self.get_epoch_model(config)
        self.epoch = load_epoch or 1

        self._graph = tf_compat.v1.Graph()
        with self._graph.as_default():
            self.session = tf_compat.v1.Session(graph=self._graph)
            self._create_placeholders()
            self._create_model(epoch=self.epoch, step=0)
            with rf.set_amp_policy_ctx(self._amp_policy):
                self._init_forward_func()
            self.session.run(tf_compat.v1.global_variables_initializer())
        assert load_filename, "forward task: no checkpoint to load (set `load` or `model` in the config)"
        self._load_model(filename=load_filename)
        print(f"TF engine: forward with epoch {self.epoch} ({load_filename})", file=log.v3)

    def _init_forward_func(self):
        """
        Build the forward graph: run ``forward_step`` once and keep what it marked as output.
        """
        rf.init_forward_step_run_ctx(epoch=self.epoch, step=0)
        self._forward_step_func(model=self.model, extern_data=self.extern_data, **util.get_fwd_compat_kwargs())
        outputs = rf.get_run_ctx().outputs
        assert outputs.data, "forward_step did not mark any output"
        expected = self.config.typed_value("model_outputs")
        if expected is not None:
            expected_dict = TensorDict()
            expected_dict.update(expected, auto_convert=True)
            if set(expected_dict.data) != set(outputs.data):
                raise ValueError(
                    f"model_outputs declares {sorted(expected_dict.data)}"
                    f" but forward_step marked {sorted(outputs.data)}"
                )
        self._forward_outputs = outputs
        self._forward_fetches = {key: value.raw_tensor for key, value in outputs.data.items()}
        # the dynamic sizes come along: they are what cuts each sequence out of the padded batch
        for value in outputs.data.values():
            for dim in value.dims:
                size = dim.dyn_size_ext
                if dim in self._forward_dim_fetches or size is None or not size.dims or size.raw_tensor is None:
                    continue
                key = f"size:{len(self._forward_dim_fetches)}"
                self._forward_dim_fetches[dim] = key
                self._forward_fetches[key] = size.raw_tensor

    def forward_with_callback(
        self, *, dataset: Dataset, callback: ForwardCallbackIface, dataset_init_epoch: bool = True
    ):
        """
        :param dataset:
        :param callback:
        :param dataset_init_epoch: whether to call ``dataset.init_seq_order`` here
        """
        assert self._forward_fetches, "forward_with_callback: init_network_from_config was not called"
        if dataset_init_epoch:
            dataset.init_seq_order(epoch=self.epoch)
        callback.init(model=self.model)
        num_seqs = 0
        start_time = time.time()
        for feed_dict, _, seq_tags in self._iter_batches(dataset, train=False, init_seq_order=False):
            res = self.session.run(self._forward_fetches, feed_dict=feed_dict)
            sizes = {dim: res[key] for dim, key in self._forward_dim_fetches.items()}
            for seq_idx, seq_tag in enumerate(seq_tags):
                out = TensorDict()
                for key, template in self._forward_outputs.data.items():
                    out.data[key] = _seq_from_batch(template, res[key], seq_idx, sizes)
                callback.process_seq(seq_tag=seq_tag, outputs=out)
                num_seqs += 1
        callback.finish()
        print(f"forward: {num_seqs} seqs, {time.time() - start_time:.1f} sec", file=log.v3)

    def get_model(self) -> rf.Module:
        """
        :return: the model
        """
        return self.model

    def _create_placeholders(self):
        """
        One placeholder per extern data entry and per dynamic dim, fed by :func:`_iter_batches`.

        The string entries (``seq_tag``) are skipped: they are meta info of the batch,
        not model inputs, and TF placeholders are not how they would be passed anyway.
        """
        self._data_keys = sorted(key for key, value in self.extern_data.data.items() if value.dtype != "string")
        # The dims come from the config, so they are shared between engines, and they cache derived
        # tensors (seq masks, dim math). No tensor may cross graphs, so drop all of that first --
        # otherwise a second engine in the same process mixes graphs.
        batch_dim.reset_raw()
        for key in self._data_keys:
            for dim in self.extern_data.data[key].dims:
                dim.reset_raw()
        for key in self._data_keys:
            value = self.extern_data.data[key]
            value.raw_tensor = TFBackend.create_placeholder_raw(value)
            for dim in value.dims:
                if dim == batch_dim or not dim.is_dynamic():
                    continue
                if dim.dyn_size_ext is None:
                    # A dim declared via `dim_tags` in the config carries no size template at all
                    # (Dim(None, name=...)); only a dim built as Dim(Tensor(...)) does.
                    # The PyTorch engine never notices, because its data pipeline attaches the
                    # sizes per batch. Here the size is a fed placeholder, so the template has to
                    # exist first -- without it the dim stays size-less and anything that masks
                    # (copy_masked, the seq masks) fails on dyn_size_ext being None.
                    dim.dyn_size_ext = Tensor(dim.name or "size", dims=[batch_dim], dtype="int32")
                if dim.dyn_size_ext.raw_tensor is None:
                    dim.dyn_size_ext.raw_tensor = TFBackend.create_placeholder_raw(dim.dyn_size_ext)
                    self._fed_dims.append((key, dim))
        # The batch dim gets its size from the data itself. The net-dict path derives this from
        # its BatchInfo, which the RF path does not build, but masked reduces need it.
        data = self.extern_data.data[self._data_keys[0]]
        batch_dim.dyn_size_ext = Tensor("batch", dims=(), dtype="int32")
        batch_dim.dyn_size_ext.raw_tensor = tf.shape(data.raw_tensor)[0]

    def _iter_batches(self, dataset: Dataset, *, train: bool, init_seq_order: bool = True):
        """
        :param dataset:
        :param train: whether this is the train dataset (affects only the batch options from the config)
        :param init_seq_order: whether to call ``dataset.init_seq_order`` here
        :return: iterator over (feed dict for the placeholders, complete_frac or None, seq tags)

        ``complete_frac`` is what the other engines derive ``epoch_continuous`` from,
        and it comes from the dataset rather than from a batch count:
        the number of batches per epoch is not known before the epoch is over,
        and a resumed run would never learn it for the epoch it resumes in.
        It is None when the dataset cannot tell (then a schedule using it must say so itself).
        """
        batch_size = (
            self._batch_opts["batch_size"]
            if train
            else self._batch_opts.get("eval_batch_size") or self._batch_opts["batch_size"]
        )
        if init_seq_order:
            dataset.init_seq_order(epoch=self.epoch)
        batches = dataset.generate_batches(
            recurrent_net=True,
            batch_size=batch_size,
            max_seqs=self._batch_opts["max_seqs"],
            **{k: v for k, v in self._batch_opts.items() if k not in ("batch_size", "eval_batch_size", "max_seqs")},
        )
        while batches.has_more():
            (batch,) = batches.peek_next_n(1)
            raw = batch_to_raw_dict(batch, dataset=dataset, extern_data=self.extern_data, data_keys=self._data_keys)
            feed_dict = {self.extern_data.data[key].raw_tensor: raw[key] for key in self._data_keys}
            for key, dim in self._fed_dims:
                feed_dict[dim.dyn_size_ext.raw_tensor] = raw[f"{key}_seq_lens"]
            # allow_only_lr_suitable: a dataset which cannot give a monotonic fraction says None here
            # rather than a number an LR schedule would step backwards on.
            complete_frac = dataset.get_complete_frac(batch.end_seq - 1, allow_only_lr_suitable=True)
            # in the order the rows sit in the batch, which is what a per-sequence callback needs
            seq_tags, seen = [], set()
            for part in batch.seqs:
                if part.seq_idx in seen:
                    continue  # chunking can split one sequence over several parts
                seen.add(part.seq_idx)
                seq_tags.append(dataset.get_tag(part.seq_idx))
            yield feed_dict, complete_frac, seq_tags
            batches.advance(1)

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
        # The variables come after the model, so that they can be named by the module hierarchy.
        with TFBackend.deferred_parameter_creation():
            model = get_model_func(epoch=epoch, step=step, **sentinel_kw)
            assert isinstance(model, rf.Module), f"get_model returned {model!r}, expected an rf.Module"
        TFBackend.create_parameters(model)
        self.model = model
        params = dict(model.named_parameters())
        num_params = sum(int(numpy.prod([d.dimension for d in p.dims])) for p in params.values())
        print(f"net params #: {num_params} ({len(params)} params)", file=log.v2)
        self._saver = tf_compat.v1.train.Saver(
            {name: TFBackend.get_parameter_variable(p) for name, p in params.items()}
        )

    def _init_step_func(self):
        """
        Build the step graph: the model outputs, the losses, and the optimizer op.
        """
        sentinel_kw = util.get_fwd_compat_kwargs()
        # The train flag is a placeholder, not a Python bool: the graph is built once and eval must
        # run it WITHOUT dropout etc. RunCtx supports a dynamic flag exactly for graph backends,
        # and the net-dict engine feeds its train flag the same way.
        self._train_flag = tf_compat.v1.placeholder_with_default(True, shape=(), name="train_flag")
        self._step_placeholder = tf_compat.v1.placeholder_with_default(
            tf.constant(0, dtype="int64"), shape=(), name="global_train_step"
        )
        rf.init_train_step_run_ctx(
            train_flag=rf.convert_to_tensor(self._train_flag, dims=(), dtype="bool"),
            step=self._step_placeholder,
            epoch=self.epoch,
        )
        self._train_step_func(model=self.model, extern_data=self.extern_data, **sentinel_kw)
        run_ctx = rf.get_run_ctx()
        assert run_ctx.losses, "train_step did not mark any loss"
        total = run_ctx.total_loss()
        self._loss = total.raw_tensor if isinstance(total, Tensor) else total
        self._losses = {name: loss.get_mean_loss().raw_tensor for name, loss in run_ctx.losses.items()}

        self._global_train_step_var = tf.Variable(
            self.global_train_step, dtype="int64", trainable=False, name="global_step"
        )
        # The whole LR schedule of a setup can live in dynamic_learning_rate, so it must be applied.
        # The TF Updater has an in-graph path for it, but that one only passes global_train_step,
        # while such a function may also want epoch / epoch_continuous
        # (e.g. dyn_lr_piecewise_linear with learning_rate_piecewise_by_epoch_continuous asserts on it),
        # and it computes on plain floats, not on tensors.
        # So apply it per step in Python, as the PyTorch engine does,
        # and hide it from the Updater so its in-graph path stays out of the way.
        self._dyn_lr_func = self.config.typed_value("dynamic_learning_rate", None)
        updater_config = self.config
        if self._dyn_lr_func is not None:
            if not callable(self._dyn_lr_func):
                raise NotImplementedError(f"dynamic_learning_rate {self._dyn_lr_func!r} is not callable")
            signature = inspect.signature(self._dyn_lr_func)
            assert any(arg.kind == inspect.Parameter.VAR_KEYWORD for arg in signature.parameters.values()), (
                "please specify **kwargs in dynamic_learning_rate for future compatibility"
            )
            if "network" in signature.parameters:
                raise ValueError("TF RF engine: dynamic_learning_rate with network is net-dict specific")
            print("Using dynamic learning rate scheduler that updates based on global train steps", file=log.v2)
            updater_config = copy.copy(self.config)
            updater_config.typed_dict = dict(self.config.typed_dict)
            updater_config.typed_dict.pop("dynamic_learning_rate", None)
        self._updater = Updater(
            config=updater_config,
            initial_learning_rate=self.learning_rate,
            objective=self._loss,
            global_train_step_var=self._global_train_step_var,
        )
        self._updater.set_trainable_vars(
            [TFBackend.get_parameter_variable(p) for _, p in self.model.named_parameters() if p.trainable is not False]
        )
        self._optim_op = self._updater.get_optim_op()
        if self._updater.log_grad_norm_tensor is not None:
            self._extra_fetches["grad_norm:p2"] = self._updater.log_grad_norm_tensor
        if self._log_memory_usage:
            # An in-graph op, so it has to be added here rather than read out per step.
            self._extra_fetches["mem_usage:GPU:0"] = tf_util.mem_usage_for_dev("/device:GPU:0")

    def _save_model(self):
        """
        Save the model of the current epoch.
        """
        if not self.model_filename:
            print("No 'model' in the config, not saving.", file=log.v4)
            return
        filename = self.get_epoch_model_filename()
        self._saver.save(self.session, filename)
        print(f"Saved model {filename}", file=log.v3)

    def _batch_size_info(self, feed_dict: Dict[Any, numpy.ndarray]) -> Dict[str, int]:
        """
        :param feed_dict: as :func:`_iter_batches` yields it
        :return: for the log: per data key the padded size and the used (summed seq len)

        Read off the fed arrays rather than off the batch: what matters for the log
        is what actually goes into the graph, padding included.
        """
        info = {"num_seqs": int(feed_dict[self.extern_data.data[self._data_keys[0]].raw_tensor].shape[0])}
        for key in self._data_keys:
            info[f"batch_size:{key}"] = int(numpy.prod(feed_dict[self.extern_data.data[key].raw_tensor].shape))
        for key, dim in self._fed_dims:
            info[f"seq_len:{key}"] = int(numpy.sum(feed_dict[dim.dyn_size_ext.raw_tensor]))
        return info

    def _maybe_stop_for_resubmission(self, last_epoch_wall_sec: float):
        """
        :param last_epoch_wall_sec: wall time of the epoch that just finished

        Stop now if the SLURM wall time left is less than the (safety-scaled) last epoch,
        so that sisyphus resubmits instead of the job dying mid-epoch and losing it.
        Same mechanism and the same reasoning as the PyTorch engine's, minus the torchelastic case
        (this engine is one process): SIGINT to our own process group reaches the sisyphus task
        worker as a KeyboardInterrupt, which it does not catch,
        so the job counts as interrupted rather than failed.
        See https://github.com/rwth-i6/returnn/issues/1818.
        """
        import signal
        from returnn.util.basic import slurm_time_left_sec

        if not self.config.bool("stop_for_resubmission_when_low_time_left", False):
            return
        time_left = slurm_time_left_sec()
        if time_left is None:
            return  # not in SLURM, or the squeue query failed
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

    @staticmethod
    def delete_model(filename: str) -> int:
        """
        :param filename: as :func:`EngineBase.get_epoch_model` returns it, without extension
        :return: accumulated size in bytes of the deleted files

        For :func:`EngineBase.cleanup_old_models`. A TF checkpoint is a set of files
        (``.index``, ``.meta``, ``.data-*``) sharing that prefix.
        """
        from glob import glob

        count_bytes = 0
        assert os.path.exists(filename + ".index"), f"delete_model: no checkpoint {filename}"
        for fn in glob(filename + "*"):
            fn_ext = os.path.splitext(fn)[1]
            if fn_ext not in (".index", ".meta") and not fn_ext.startswith(".data"):
                continue
            count_bytes += os.stat(fn).st_size
            os.remove(fn)
        assert count_bytes > 0
        return count_bytes

    def _load_model(self, *, filename: str):
        """
        :param filename: as :func:`EngineBase.get_epoch_model` returns it
        """
        self._saver.restore(self.session, filename)
        print(f"Loaded model {filename}", file=log.v3)


_UnsupportedConfigOpts = {
    "accum_grad_multiple_step": 1,
    "apply_cleanup_old_models_to_optim_states": False,
    "calculate_exp_loss": False,
    "chunking": None,
    "min_chunk_size": None,
    "debug_shell_before_train_loop": False,
    "default_float_dtype": None,
    "epoch_end": None,
    "epoch_start": None,
    "forward_auto_split_batch_on_oom": False,
    "grad_scaler": None,
    "load_model_post_hooks": None,
    "online_shuffle_batches": None,
    "preload_from_files": None,
    "pretrain": None,
    "reset_dev_memory_caches": False,
    "sort_dataset": None,
    "stop_on_nonfinite_train_score": None,
    "tensorboard_opts": None,
    "use_tensorboard": False,
    "use_train_proc_manager": False,
    # backend-specific options are named after the backend, as `torch_...` is on PyTorch
    "tf_distributed": None,
    "tf_jit": None,  # the XLA-compiled step
    "tf_profile": None,
}

# PyTorch-specific options, with the TF name they correspond to (None: no equivalent).
# A config copied from a PyTorch setup carries these, and ignoring them silently
# -- `torch_amp` above all -- is exactly what this check exists to prevent.
_TorchOnlyConfigOpts = {
    "torch_amp": "tf_amp",
    "torch_cuda_graph": "tf_jit",
    "torch_dataloader_opts": None,  # this engine uses the shared batching, which has no worker pool
    "torch_distributed": "tf_distributed",
    "torch_log_memory_usage": "tf_log_memory_usage",
    "torch_profile": "tf_profile",
}


def _check_config_opts_supported(config: Config):
    """
    :param config:
    :raise NotImplementedError: if the config sets an option this engine would ignore

    Same check as the JAX engine's (`returnn/jax/engine.py`), for the same reason:
    a config written for another backend carries options which this engine does not read,
    and silently ignoring them changes what the config means.
    """

    # noinspection PyShadowingNames  -- local helper, shadowing the outer key/value is intended
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
    for key, tf_name in sorted(_TorchOnlyConfigOpts.items()):
        value = _value_if_set(key)
        if value is not None:
            unsupported.append(
                f"{key} = {value!r} is PyTorch specific"
                + (f", the TF engine reads {tf_name} (not implemented either)" if tf_name else "")
            )
    if unsupported:
        raise NotImplementedError(
            "TF engine: the config sets options which this engine does not implement:\n  "
            + "\n  ".join(unsupported)
            + "\nThey would otherwise be ignored silently, which would change what the config means."
        )


def _seq_from_batch(template: Tensor, raw: numpy.ndarray, seq_idx: int, sizes: Dict[Dim, numpy.ndarray]) -> Tensor:
    """
    :param template: a marked output, batched
    :param raw: its value for the whole batch
    :param seq_idx: the row in the batch
    :param sizes: per dynamic dim, its size per sequence
    :return: that row, without the batch dim and with the padding cut off

    The callback gets one sequence at a time, so the dynamic dims become static here
    (their value for this sequence), as they do in the PyTorch engine.
    """
    if batch_dim not in template.dims:
        raise Exception(f"forward output {template} has no batch dim")
    batch_axis = template.dims.index(batch_dim)
    value = numpy.take(raw, seq_idx, axis=batch_axis)
    dims = []
    for axis, dim in enumerate([d for i, d in enumerate(template.dims) if i != batch_axis]):
        lens = sizes.get(dim)
        if lens is None:
            dims.append(dim)
            continue
        size = int(lens[seq_idx])
        value = value[(slice(None),) * axis + (slice(0, size),)]
        dims.append(Dim(size, name=dim.name or "spatial"))
    out = Tensor(template.name, dims=dims, dtype=template.dtype, sparse_dim=template.sparse_dim)
    out.raw_tensor = value
    return out


def _format_extra_fetch(key: str, value: Any) -> str:
    """
    :param key: e.g. "grad_norm:p2" or "mem_usage:GPU:0"
    :param value:
    :return: one entry for the step log line
    """
    if key.startswith("mem_usage:"):
        return f"{key} {util.human_bytes_size(int(value))}"
    return f"{key} {float(value):.5f}"


def _amp_policy_from_config(config: Config) -> Optional[rf.AmpPolicy]:
    """
    :param config:
    :return: the mixed-precision policy from ``tf_amp``, or None

    ``tf_amp`` is the TF counterpart of ``torch_amp``, and takes the compute dtype
    ("bfloat16", "float16") or a dict with a ``dtype`` key.
    Unlike PyTorch, TF has no autocast, so the casts are placed by RF itself
    (see :mod:`returnn.frontend.amp`); the parameters and the optimizer stay float32 either way.
    """
    opts = config.typed_value("tf_amp", None)
    if opts is None:
        return None
    if isinstance(opts, str):
        return rf.AmpPolicy(compute_dtype=opts)
    if isinstance(opts, dict):
        opts = dict(opts)
        dtype = opts.pop("dtype", None)
        if not dtype:
            raise ValueError(f"tf_amp {opts!r}: no dtype")
        if opts:
            raise NotImplementedError(f"tf_amp: unsupported options {sorted(opts)}")
        return rf.AmpPolicy(compute_dtype=dtype)
    raise TypeError(f"tf_amp {opts!r}: expected a dtype name or a dict")


def _batch_opts_from_config(config: Config) -> Dict[str, Any]:
    """
    :param config:
    :return: the batching options this engine supports
    """
    opts = {
        "batch_size": config.typed_value("batch_size", None) or config.int("batch_size", 10000),
        "eval_batch_size": config.typed_value("eval_batch_size", None),
        "max_seqs": config.int("max_seqs", -1),
    }
    # Further options of the shared batching layer (Dataset.generate_batches), passed through as given.
    for key in ("max_seq_length", "min_seq_length", "max_pad_size", "max_total_num_seqs", "seq_drop"):
        value = config.typed_value(key, None)
        if value is not None:
            opts[key] = value
    return opts


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
