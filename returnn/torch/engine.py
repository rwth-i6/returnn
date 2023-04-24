"""
Main engine for PyTorch
"""

from __future__ import annotations
from typing import Optional, Union, Callable, Dict

import os
import torch
import torch.utils.data.datapipes as dp
from torchdata.dataloader2 import DataLoader2
from random import random

from returnn.config import Config
from returnn.log import log
from returnn.engine.base import EngineBase
import returnn.frontend as rf
from returnn.tensor import TensorDict, Tensor
from returnn.datasets.basic import init_dataset, Dataset
from returnn.util import basic as util
from returnn.util import NumbersDict
from .updater import Updater
from .data import pipeline as data_pipeline
from .data import returnn_dataset_wrapper
from .frontend.bridge import rf_module_to_pt_module


class Engine(EngineBase):
    """
    PyTorch engine
    """

    def __init__(self, config: Config):
        """
        :param config:
        """
        super(Engine, self).__init__(config=config)
        rf.select_backend_torch()
        self.model_filename = self.config.value("model", None)
        self._mp_manager = torch.multiprocessing.Manager()
        self._epoch_mp_shared = self._mp_manager.Value("i", 0)
        self.train_dataset = None  # type: Optional[Dataset]
        self.eval_datasets = {}
        self.extern_data = None  # type: Optional[TensorDict]
        self._train_dataloader = None  # type: Optional[DataLoader2]
        self._eval_dataloaders = {}  # type: Dict[str, DataLoader2]

        self._start_epoch = None  # type: Optional[int]
        self._final_epoch = None  # type: Optional[int]
        self._orig_model = None  # type: Optional[Union[rf.Module, torch.nn.Module]]
        self._pt_model = None  # type: Optional[torch.nn.Module]
        self._train_step_func = None  # type: Optional[Callable]
        self._save_model_epoch_interval = 1
        self._updater = None  # type: Optional[Updater]

        self._device = _get_device_from_config(config)
        print("Using device:", self._device, file=log.v2)

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
        assert config is self.config
        super().init_train_from_config(config=config)
        self.train_dataset = train_data
        self.eval_datasets.clear()
        if dev_data:
            self.eval_datasets["dev"] = dev_data
        if eval_data:
            self.eval_datasets["eval"] = eval_data
        if config.has("eval_datasets"):
            for dataset_name, dataset_opts in config.typed_value("eval_datasets", {}).items():
                self.eval_datasets[dataset_name] = init_dataset(dataset_opts, default_kwargs={"name": dataset_name})

        extern_data = TensorDict()
        extern_data_dict = self.config.typed_value("extern_data")
        extern_data.update(extern_data_dict, auto_convert=True)
        self.extern_data = extern_data

        self._train_dataloader = self._create_data_loader(train_data) if train_data else None
        for dataset_name, dataset in self.eval_datasets.items():
            self._eval_dataloaders[dataset_name] = self._create_data_loader(dataset)

        self._start_epoch = self.get_train_start_epoch(self.config)
        self._final_epoch = self.config_get_final_epoch(self.config)

        self._load_model(epoch=self._start_epoch)
        self._save_model_epoch_interval = config.int("save_interval", 1)

        self._updater = Updater(self.config, self._pt_model, self.learning_rate)
        self._updater.create_optimizer()
        if self._start_epoch > 1:
            self._load_optimizer(self._start_epoch)

        self._train_step_func = self.config.typed_value("train_step")
        assert self._train_step_func, "train_step not defined"

    def train(self):
        """
        Main training loop.
        """

        print("Starting training at epoch {}.".format(self._start_epoch), file=log.v3)
        assert self._pt_model is not None, "Model not initialized, call init_train_from_config()."

        self.epoch = self._start_epoch
        self._epoch_mp_shared.value = self.epoch
        while self.epoch <= self._final_epoch:
            self.init_train_epoch()
            self.train_epoch()

            self.epoch += 1
            self._epoch_mp_shared.value = self.epoch

        print("Finished training at epoch {}.".format(self.epoch), file=log.v3)

    def init_train_epoch(self):
        """
        init train (sub)epoch. LR etc
        """
        self.learning_rate = self.learning_rate_control.get_learning_rate_for_epoch(self.epoch)

        # Update learning rate
        self._updater.set_learning_rate(self.learning_rate)

    def train_epoch(self):
        """
        train one (sub)epoch
        """
        print("start", self.get_epoch_str(), "with learning rate", self.learning_rate, "...", file=log.v4)

        self._pt_model.train()

        accumulated_losses_dict = NumbersDict()
        accumulated_inv_norm_factors_dict = NumbersDict()
        step_idx = 0
        for data in self._train_dataloader:
            self._run_step(data, train_flag=True)

            train_ctx = rf.get_run_ctx()
            total_loss = train_ctx.total_loss()
            losses_dict = NumbersDict(
                {
                    name: float(loss.get_summed_loss().raw_tensor.detach().cpu().numpy())
                    for name, loss in train_ctx.losses.items()
                }
            )
            inv_norm_factors_dict = NumbersDict(
                {name: float(_to_raw(loss.get_inv_norm_factor())) for name, loss in train_ctx.losses.items()}
            )

            self._updater.get_optimizer().zero_grad()
            total_loss.raw_tensor.backward()
            self._updater.get_optimizer().step()

            accumulated_losses_dict += losses_dict
            accumulated_inv_norm_factors_dict += inv_norm_factors_dict
            _print_process(
                f"ep {self.epoch} train",
                step=step_idx,
                eval_info=dict(losses_dict / inv_norm_factors_dict),
            )

            step_idx += 1
            self.global_train_step += 1

        print("Trained %i steps" % step_idx)

        accumulated_losses_dict = accumulated_losses_dict / accumulated_inv_norm_factors_dict
        self.learning_rate_control.set_epoch_error(
            self.epoch, {f"train_loss_{k}": v for k, v in accumulated_losses_dict.items()}
        )
        self.learning_rate_control.save()

        print(f"Total train loss:", _format_score(dict(accumulated_losses_dict)), file=log.v3)

        if self.epoch % self._save_model_epoch_interval == 0 or self.epoch == self._final_epoch:
            self._save_model()
            self._save_optimizer()

        self.eval_model()

    def eval_model(self):
        """
        Runs model on all eval datasets and calculates the loss.
        """
        self._pt_model.eval()

        eval_dump_str = []
        score_keys = None
        error_keys = None

        for dataset_name, dataset in self.eval_datasets.items():
            print(f"Evaluating dataset {dataset_name!r}", file=log.v3)

            data_loader = self._eval_dataloaders[dataset_name]

            accumulated_losses_dict = NumbersDict()
            accumulated_inv_norm_factors_dict = NumbersDict()
            step_idx = 0

            with torch.no_grad():
                for data in data_loader:

                    self._run_step(data)
                    train_ctx = rf.get_run_ctx()

                    if score_keys is None:
                        score_keys = [name for name, loss in train_ctx.losses.items() if not loss.as_error]
                        error_keys = [name for name, loss in train_ctx.losses.items() if loss.as_error]

                    losses_dict = NumbersDict(
                        {
                            name: float(loss.get_summed_loss().raw_tensor.detach().cpu().numpy())
                            for name, loss in train_ctx.losses.items()
                        }
                    )
                    inv_norm_factors_dict = NumbersDict(
                        {name: float(_to_raw(loss.get_inv_norm_factor())) for name, loss in train_ctx.losses.items()}
                    )

                    accumulated_losses_dict += losses_dict
                    accumulated_inv_norm_factors_dict += inv_norm_factors_dict
                    _print_process(
                        f"ep {self.epoch} {dataset_name} eval",
                        step=step_idx,
                        eval_info=dict(losses_dict / inv_norm_factors_dict),
                    )
                    step_idx += 1

            assert step_idx > 0, f"No data in dataset {dataset_name!r}."
            accumulated_losses_dict = accumulated_losses_dict / accumulated_inv_norm_factors_dict

            self.learning_rate_control.set_epoch_error(
                self.epoch, {f"{dataset_name}_loss_{k}": v for k, v in accumulated_losses_dict.items()}
            )
            self.learning_rate_control.save()

            # Same format as the TF engine.
            eval_dump_str += [
                "%s: score %s error %s"
                % (
                    dataset_name,
                    _format_score({name: accumulated_losses_dict[name] for name in score_keys}),
                    _format_score({name: accumulated_losses_dict[name] for name in error_keys}),
                )
            ]

        print(" ".join(eval_dump_str), file=log.v1)

    def _create_data_loader(self, dataset: Dataset) -> DataLoader2:
        """
        :param dataset: RETURNN dataset
        :return: PyTorch data loader created from given RETURNN dataset
        """
        # Make sure that _dataset_reset does not keep a ref to `self`,
        # otherwise it would trigger to pickle `self` and all its members.
        dataset_reset = returnn_dataset_wrapper.ReturnnDatasetResetMpSharedEpochCallback(
            dataset=dataset, epoch_mp_shared=self._epoch_mp_shared
        )

        wrapped_dataset = returnn_dataset_wrapper.ReturnnDatasetIterDataPipe(dataset, reset_callback=dataset_reset)

        chunking = self.config.typed_value("chunking", None)
        if chunking:
            wrapped_dataset = data_pipeline.ChunkingIterDataPipe(wrapped_dataset, chunking)

        batch_size = self.config.typed_value("batch_size", 1)
        max_seqs = self.config.int("max_seqs", -1)
        batches_dataset = data_pipeline.BatchingIterDataPipe(wrapped_dataset, batch_size=batch_size, max_seqs=max_seqs)
        batches_dataset = dp.iter.Collator(batches_dataset, collate_fn=data_pipeline.collate_batch)

        try:
            return DataLoader2(batches_dataset)
        except TypeError as exc:
            try:
                # noinspection PyPackageRequirements
                import dill
            except ImportError:
                raise ModuleNotFoundError("Possible type error in DataLoader2 due to missing module 'dill'") from exc
            raise

    def _run_step(self, extern_data_raw: Dict[str, torch.Tensor], *, train_flag: bool = False):
        """
        :param dict[str, torch.Tensor] extern_data_raw: model inputs for the step
        """
        assert isinstance(extern_data_raw, dict) and extern_data_raw
        batch_dim = next(iter(self.extern_data.data.values())).dims[0]
        batch_dim.dyn_size_ext = None  # if it is dynamic, reset now, and set it below
        extern_data = TensorDict()
        for k, data in self.extern_data.data.items():
            data = data.copy_template()
            raw_tensor = extern_data_raw[k].to(self._device)
            data.dtype = str(raw_tensor.dtype).split(".")[-1]  # just overwrite for now...
            data.raw_tensor = raw_tensor

            if batch_dim.size is None and batch_dim.dyn_size_ext is None:
                batch_dim.dyn_size_ext = Tensor(batch_dim.name or "batch", dims=[], dtype="int32")
                batch_dim.dyn_size_ext.raw_tensor = torch.tensor(extern_data_raw[k].shape[0], dtype=torch.int32)

            if k + ":seq_len" in extern_data_raw:
                # Sequence lengths have to be on CPU for the later call to rnn.pack_padded_sequence
                size = extern_data_raw[k + ":seq_len"].cpu()
                size_dtype = str(size.dtype).split(".")[-1]
                if data.dims[1].dyn_size_ext is None:
                    data.dims[1].dyn_size_ext = Tensor(data.dims[1].name or "time", dims=[batch_dim], dtype=size_dtype)
                data.dims[1].dyn_size_ext.dtype = size_dtype
                data.dims[1].dyn_size_ext.raw_tensor = size

            extern_data.data[k] = data

        rf.init_train_step_run_ctx(train_flag=train_flag)

        sentinel_kw = {"__fwd_compatible_random_arg_%i" % int(random() * 100): None}
        self._train_step_func(model=self._orig_model, extern_data=extern_data, **sentinel_kw)

    def _load_model(self, *, epoch: int):
        """
        Sets self._model to a torch.nn.Module.

        :param epoch:
        """
        checkpoint_state = None
        if epoch > 1:
            filename = self.get_epoch_model_filename(epoch=epoch - 1) + util.get_model_filename_postfix()
            print("Load model %s" % (filename,), file=log.v4)
            checkpoint_state = torch.load(filename)
            assert checkpoint_state["epoch"] == epoch - 1
            step = checkpoint_state["step"]
        else:
            step = 0
        self.global_train_step = step

        # See :mod:`rf.rand` docstring for an explanation of this logic.
        random_seed = self.config.int("random_seed", 42)
        random_seed = (epoch * 193939 + step * 19937 + random_seed * 27644437 + 479001599) % (2**31)
        rf.set_random_seed(random_seed)

        get_model_func = self.config.typed_value("get_model")
        assert get_model_func, "get_model not defined"
        sentinel_kw = {"__fwd_compatible_random_arg_%i" % int(random() * 100): None}
        model = get_model_func(epoch=epoch, step=step, **sentinel_kw)
        self._orig_model = model
        if isinstance(model, rf.Module):
            self._pt_model = rf_module_to_pt_module(model)
        elif isinstance(model, torch.nn.Module):
            self._pt_model = model
        else:
            raise TypeError(f"get_model returned {model} of type {type(model)}, expected rf.Module or torch.nn.Module")
        assert isinstance(self._pt_model, torch.nn.Module)
        print("Model:", self._pt_model, file=log.v4)

        if checkpoint_state is not None:
            self._pt_model.load_state_dict(checkpoint_state["model"])
        preload_from_files = self.config.typed_value("preload_from_files", {})
        if preload_from_files:
            # see `preload_from_files` in tf engine and `returnn.tf.network.CustomCheckpointLoader`
            is_training = self.config.value("task", "train") == "train"
            is_first_train_epoch = epoch == 1 and (
                is_training or self.config.value("task", "train") == "initialize_model"
            )
            # We use the reversed sorted order here to achieve consistent behavior with the TF engine.
            # There, the keys are used in sorted order but if a variable is loaded,
            # it will not be considered anymore afterward.
            # So the first occurrence is used.
            # Here, we overwrite variables even if they have been loaded before.
            # In order to get consistent behavior, we use the reversed order.
            for preload_key, opts in reversed(sorted(preload_from_files.items())):
                assert isinstance(opts, dict) and "filename" in opts
                if opts.get("init_for_train", False):
                    if not is_first_train_epoch:
                        continue
                else:  # default: init for recog
                    if is_training:
                        continue
                print(f"Pre-load weights for key '{preload_key}' from {opts['filename']}", file=log.v3)
                preload_model_state = torch.load(opts["filename"])
                if opts.get("checkpoint_key", "model") is not None:
                    # This can be used if an external checkpoint saves a checkpoint a different structure that just the
                    # model state dict. E.g., if a checkpoint is created using
                    # `torch.save({"model": model.state_dict(), "optimizer": optimizer.state)_dict(), ...})`
                    # we can set checkpoint_key = "model" to load the model.
                    # Currently, this only supports single level dicts, but it could be extended if needed.
                    preload_model_state = preload_model_state[opts.get("checkpoint_key", "model")]
                if opts.get("prefix", ""):
                    # Only params with this prefix should be loaded.
                    # They are expected to be in the checkpoint without this prefix.
                    # By adding the prefix to all params,
                    # we make sure that only params matching this condition are loaded.
                    # This is in line with the behavior of the TF engine.
                    preload_model_state = {opts["prefix"] + key: value for key, value in preload_model_state.items()}
                ignore_params = opts.get("ignore_params", [])
                ignore_params_prefixes = opts.get("ignore_params_prefixes", [])
                for key in list(preload_model_state.keys()):
                    if key in ignore_params or any(
                        [key.startswith(ignore_key) for ignore_key in ignore_params_prefixes]
                    ):
                        print(f"Ignoring variable {key}", file=log.v3)
                        preload_model_state.pop(key)
                for new_name, name_in_checkpoint in opts.get("var_name_mapping", {}).items():
                    preload_model_state[new_name] = preload_model_state.pop(name_in_checkpoint)
                missing_keys, _ = self._pt_model.load_state_dict(preload_model_state, strict=False)
                if missing_keys and not opts.get("ignore_missing", False):
                    prefix_keys = [key for key in self._pt_model.state_dict() if key.startswith(opts.get("prefix", ""))]
                    missing_prefix_keys = set(prefix_keys).intersection(set(missing_keys))
                    assert not missing_prefix_keys, f"Missing keys and ignore_missing=False: {missing_prefix_keys}"
                print(f"Missing keys: {missing_keys}", file=log.v4)

        self._pt_model.to(self._device)

    def _save_model(self):
        """
        Saves the state of self._model to file.
        """
        filename = self.get_epoch_model_filename() + util.get_model_filename_postfix()
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        print("Save model under %s" % (filename,), file=log.v4)
        torch.save(
            {"model": self._pt_model.state_dict(), "epoch": self.epoch, "step": self.global_train_step}, filename
        )

    def _load_optimizer(self, epoch):
        """
        Loads a torch.optim.Optimizer from disk and uses it as the optimizer.
        This function is a wrapper to Updater.load_optimizer().

        :param int epoch: Epoch from which to load the optimizer state.
        """
        filename = self.get_epoch_model_filename(epoch=epoch - 1) + ".opt" + util.get_model_filename_postfix()
        self._updater.load_optimizer(filename)

    def _save_optimizer(self):
        """
        Saves the optimizer state to a file.
        This function is a wrapper to Updater.save_optimizer().
        """
        filename = self.get_epoch_model_filename() + ".opt" + util.get_model_filename_postfix()
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        self._updater.save_optimizer(filename)

        # keep only the last two optimizer states (two in case one file gets corrupted)
        clean_epoch = self.epoch - 2
        if clean_epoch > 0:
            filename = self.get_epoch_model_filename(epoch=clean_epoch) + ".opt" + util.get_model_filename_postfix()
            if os.path.isfile(filename):
                os.unlink(filename)


def _to_raw(n: Union[int, float, Tensor]):
    if isinstance(n, (int, float)):
        return n
    if isinstance(n, Tensor):
        return n.raw_tensor.detach().cpu().numpy()
    raise TypeError(f"Unexpected {n} of type {type(n)}")


def _print_process(report_prefix, step, eval_info):
    """
    Similar but simplified from TF engine _print_process.

    :param str report_prefix:
    :param int step:
    :param dict[str] eval_info: via :func:`_collect_eval_info`
    :return: nothing, will be printed to log
    """
    if log.verbose[5]:
        info = [report_prefix, "step %i" % step]
        if eval_info:  # Such as score.
            info += ["%s %s" % item for item in sorted(eval_info.items())]
        print(", ".join(filter(None, info)), file=log.v5)


def _format_score(score: Dict[str, float]) -> str:
    """
    Like the TF engine format_score.

    :param score:
    :return: score(s) as str
    """
    if not score:
        return "None"
    if len(score) == 1:
        return str(list(score.values())[0])
    return " ".join(["%s %s" % (key.split(":", 2)[-1], str(score[key])) for key in sorted(score.keys())])


def _get_gpu_device() -> Optional[str]:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return None


def _get_device_from_config(config: Config) -> str:
    device = config.value("device", None)
    if not device:
        device = _get_gpu_device()
        if device:
            return device
        return "cpu"
    if device == "gpu":
        device = _get_gpu_device()
        if not device:
            raise Exception("No GPU device found, but config requested 'gpu' device.")
    return device
