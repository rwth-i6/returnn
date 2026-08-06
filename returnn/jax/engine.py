"""
Engine for the JAX backend.

The RF-side API is the same as for the other backends
(``get_model`` / ``train_step`` in the config, losses via ``rf.get_run_ctx().mark_as_loss``),
so a config that trains on PyTorch trains here as well.

What differs is the step itself, and it is the reason this is not a copy of the PyTorch engine:
JAX has no gradient tape, so the loss must be a FUNCTION of the parameters.
The step therefore binds the parameter arrays into the ``rf.Parameter`` objects for the duration
of one call, and ``jax.value_and_grad`` differentiates through it,
which also makes the whole step jit-able as one unit.
"""

from __future__ import annotations
from typing import Optional, Any, Dict, List
import os
import time

import jax
import numpy

from returnn.config import Config
from returnn.datasets.basic import Dataset
from returnn.engine.base import EngineBase
from returnn.log import log
from returnn.tensor import Tensor, TensorDict
from returnn.util import basic as util
import returnn.frontend as rf

from .data import iter_dataset_batches
from .updater import Updater
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
        self._updater: Optional[Updater] = None
        self._opt_state: Any = None
        self._train_step_func = None
        self._params: List[rf.Parameter] = []
        self._train_param_idx: List[int] = []
        self._value_and_grad = None
        self._batch_opts = _batch_opts_from_config(config)
        # Numeric defaults stay JAX's own -- notably, float32 matmuls run in TF32 on GPU,
        # which PyTorch does not do by default. We do not override that here:
        # each backend follows its own conventions.
        # Set this option to "highest" if you want a run to match a PyTorch baseline numerically.
        precision = config.value("jax_default_matmul_precision", None)
        if precision:
            jax.config.update("jax_default_matmul_precision", precision)
            print(f"JAX default matmul precision: {precision}", file=log.v3)

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
        self.learning_rate_control = _load_learning_rate_control(config)

        from returnn.torch.data.extern_data import extern_data_template_from_config_opts

        self.extern_data = extern_data_template_from_config_opts(config.typed_value("extern_data"))

        self._train_step_func = config.typed_value("train_step")
        assert self._train_step_func, "train_step not defined in config"

        self.model_filename = config.value("model", None)
        self.epoch = self.get_train_start_epoch(config)
        if self.global_train_step is None:
            self.global_train_step = 0
        self._create_model(epoch=self.epoch, step=self.global_train_step)
        self._updater = Updater(config=config)
        self._init_step_func()
        # Continue from an existing checkpoint if there is one (or one was configured via `load`).
        # Without this the engine would silently restart from scratch and overwrite it.
        load_epoch, load_filename = self.get_epoch_model(config)
        if load_filename:
            self._load_model(filename=load_filename)
            print(f"Continuing after epoch {load_epoch} ({load_filename})", file=log.v3)
        print(f"JAX engine: starting at epoch {self.epoch}, devices {jax.devices()}", file=log.v3)

    def train(self):
        """
        Train for the configured number of epochs.
        """
        assert self.train_dataset is not None, "no train dataset"
        final_epoch = self.config.int("num_epochs", 1)
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
        start_time = time.time()
        accumulated: Dict[str, float] = {}
        num_steps = 0
        raws = [p.raw_tensor for p in self._params]

        for extern_data in self._iter_batches(self.train_dataset, train=True):
            (loss, losses), grads = self._value_and_grad(
                [raws[i] for i in self._train_param_idx], extern_data, self.global_train_step
            )
            train_raws, self._opt_state = self._updater.step(
                params=[raws[i] for i in self._train_param_idx],
                grads=grads,
                opt_state=self._opt_state,
                learning_rate=self.learning_rate,
            )
            for i, raw in zip(self._train_param_idx, train_raws):
                raws[i] = raw
            for name, value in losses.items():
                accumulated[name] = accumulated.get(name, 0.0) + float(value)
            num_steps += 1
            self.global_train_step += 1
            if num_steps % 100 == 0:
                print(f"ep {self.epoch} step {num_steps}, loss {float(loss):.5f}", file=log.v4)

        for param, raw in zip(self._params, raws):
            param.raw_tensor = raw
        scores = {name: value / max(num_steps, 1) for name, value in accumulated.items()}
        print(
            f"epoch {self.epoch} finished: {num_steps} steps, {_format_scores(scores)},"
            f" {time.time() - start_time:.1f} sec",
            file=log.v3,
        )
        self.learning_rate_control.set_epoch_error(self.epoch, {f"train_score_{k}": v for k, v in scores.items()})
        self._save_model()
        self.eval_model()
        self.learning_rate_control.save()

    def eval_model(self):
        """
        Run over the eval datasets and report the scores.
        """
        for name, dataset in self.eval_datasets.items():
            accumulated: Dict[str, float] = {}
            num_steps = 0
            for extern_data in self._iter_batches(dataset, train=False):
                _, losses = self._loss_func(
                    [p.raw_tensor for p in self._params], extern_data, self.global_train_step, train_flag=False
                )
                for loss_name, value in losses.items():
                    accumulated[loss_name] = accumulated.get(loss_name, 0.0) + float(value)
                num_steps += 1
            scores = {f"{name}_score_{k}": v / max(num_steps, 1) for k, v in accumulated.items()}
            print(f"epoch {self.epoch} {name}: {_format_scores(scores)}", file=log.v3)
            self.learning_rate_control.set_epoch_error(self.epoch, scores)

    def get_model(self) -> rf.Module:
        """
        :return: the model
        """
        return self.model

    def _iter_batches(self, dataset: Dataset, *, train: bool):
        """
        :param dataset:
        :param train: whether this is the train dataset (affects only the batch options read from the config)
        :return: iterator over batches as TensorDicts of JAX arrays
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
        )

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
        self._params = [param for _, param in model.named_parameters()]
        self._train_param_idx = [i for i, param in enumerate(self._params) if param.trainable is not False]
        num_params = sum(int(numpy.prod(p.batch_shape)) for p in self._params)
        print(
            f"net params #: {num_params} ({len(self._train_param_idx)} of {len(self._params)} trainable)", file=log.v2
        )

    def _init_step_func(self):
        """
        Build the differentiable step. See the module docstring for why it has this shape.
        """

        def _loss_func(train_raws, extern_data: TensorDict, step, train_flag: bool = True):
            orig = [p.raw_tensor for p in self._params]
            for idx, raw in zip(self._train_param_idx, train_raws):
                self._params[idx].raw_tensor = raw
            try:
                rf.init_train_step_run_ctx(train_flag=train_flag, step=step, epoch=self.epoch)
                sentinel_kw = util.get_fwd_compat_kwargs()
                self._train_step_func(model=self.model, extern_data=extern_data, **sentinel_kw)
                run_ctx = rf.get_run_ctx()
                total = run_ctx.total_loss()
                losses = {name: loss.get_mean_loss().raw_tensor for name, loss in run_ctx.losses.items()}
                return total.raw_tensor if isinstance(total, Tensor) else total, losses
            finally:
                for param, raw in zip(self._params, orig):
                    param.raw_tensor = raw

        self._loss_func = _loss_func
        self._value_and_grad = jax.value_and_grad(_loss_func, has_aux=True)
        self._opt_state = self._updater.init([self._params[i].raw_tensor for i in self._train_param_idx])

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

    def _load_model(self, *, filename: str):
        """
        :param filename: without the ``.npz`` postfix, as :func:`EngineBase.get_epoch_model` returns it
        """
        postfix = util.get_model_filename_postfix()
        if filename.endswith(postfix):
            filename = filename[: -len(postfix)]
        _checkpoint.set_model_params(self.model, _checkpoint.load_checkpoint(filename + postfix))
        print(f"Loaded model {filename + postfix}", file=log.v3)
        opt_filename = filename + ".opt" + postfix
        if os.path.exists(opt_filename):
            self._opt_state = _checkpoint.load_opt_state(self._opt_state, opt_filename)
            print(f"Loaded optimizer state {opt_filename}", file=log.v3)


def _batch_opts_from_config(config: Config) -> Dict[str, Any]:
    """
    :param config:
    :return: the batching options this engine supports
    """
    return {
        "batch_size": config.typed_value("batch_size", None) or config.int("batch_size", 10000),
        "eval_batch_size": config.typed_value("eval_batch_size", None),
        "max_seqs": config.int("max_seqs", -1),
    }


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
