"""
Main engine for PyTorch
"""

from __future__ import annotations
from typing import Optional, Any, Union, Callable, Dict
from contextlib import nullcontext

import sys
import gc
import os
import torch
import time
import torch.distributed
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch import autocast
from torch.cuda import amp
from random import random

import returnn
from returnn.config import Config
from returnn.log import log
from returnn.engine.base import EngineBase
import returnn.frontend as rf
from returnn.tensor import TensorDict, Tensor, Dim
from returnn.datasets.basic import init_dataset, Dataset
from returnn.util import basic as util
from returnn.util import NumbersDict
from returnn.util.basic import hms, NotSpecified
from returnn.util.result_with_reason import ResultWithReason
from returnn.util.debug import debug_shell
from returnn.forward_iface import ForwardCallbackIface

from .updater import Updater
from .data import pipeline as data_pipeline
from .data import returnn_dataset_wrapper
from .data import extern_data as extern_data_util
from .frontend.bridge import rf_module_to_pt_module
from .util import diagnose_gpu


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
        if util.BackendEngine.selected_engine is None:
            util.BackendEngine.select_engine(default_fallback_engine=util.BackendEngine.Torch, config=self.config)
        self.model_filename = self.config.value("model", None)
        self._mp_manager = torch.multiprocessing.Manager()
        self._epoch_mp_shared = self._mp_manager.Value("i", 0)
        self.train_dataset = None  # type: Optional[Dataset]
        self.eval_datasets = {}
        self.extern_data = None  # type: Optional[TensorDict]
        self._train_dataloader = None  # type: Optional[DataLoader]
        self._eval_dataloaders = {}  # type: Dict[str, DataLoader]

        self._start_epoch = None  # type: Optional[int]
        self._final_epoch = None  # type: Optional[int]
        self._min_seq_length = config.typed_value("min_seq_length", None) or config.int(
            "min_seq_length", None
        )  # type: Union[int,float,Dict[str,int],NumbersDict]
        self._max_seq_length = config.typed_value("max_seq_length", None) or config.int(
            "max_seq_length", None
        )  # type: Union[int,float,Dict[str,int],NumbersDict]
        self._orig_model = None  # type: Optional[Union[rf.Module, torch.nn.Module]]
        self._pt_model = None  # type: Optional[torch.nn.Module]
        self._train_step_func = None  # type: Optional[Callable]
        self._forward_step_func = self.config.typed_value("forward_step")  # type: Optional[Callable]
        self._forward_step_expected_outputs = None  # type: Optional[TensorDict]
        if self.config.typed_value("model_outputs") is not None:
            self._forward_step_expected_outputs = TensorDict()
            self._forward_step_expected_outputs.update(self.config.typed_value("model_outputs"), auto_convert=True)
        self._save_model_epoch_interval = 1
        self._updater = None  # type: Optional[Updater]

        self._use_autocast = False
        self._autocast_dtype = None  # type: Optional[str]
        self._grad_scaler = None  # type: Optional[amp.GradScaler]

        dev_ = get_device_from_config_opt(config.value("device", None))
        self._device = dev_.result
        print("Using device:", self._device, f"({dev_.reason or '?'})", file=log.v2)

        self._use_torch_distributed = False
        self._torch_distributed_class = None  # type: Optional[Callable]
        self._torch_distributed_options = None  # type: Optional[dict]
        self._ddp_pt_model = None  # type: Optional[torch.nn.Module]
        self._accum_grad_multiple_step = config.int("accum_grad_multiple_step", 1)

        torch_distributed = config.typed_value("torch_distributed")
        if torch_distributed is not None:
            self._use_torch_distributed = True
            self._torch_distributed_class = torch_distributed.get("class", None)
            self._torch_distributed_options = torch_distributed.get("options", None)

            import returnn.torch.distributed

            torch_distributed = returnn.torch.distributed.get_ctx(config=config)
            local_rank = torch_distributed.local_rank()
            print(f"Start running torch distributed training on local rank {local_rank}.", file=log.v2)
            assert self._device == "cuda", f"torch distributed: unexpected device {self._device!r}"
            self._device = f"cuda:{local_rank}"

        # Theano and TensorFlow print sth like: Using gpu device 2: GeForce GTX 980 (...)
        # Print in a similar format so that some scripts which grep our stdout work just as before.
        if self._device.startswith("cuda"):
            diagnose_gpu.print_using_cuda_device_report(self._device, file=log.v2)

        self._log_memory_usage = config.bool("torch_log_memory_usage", False)

        amp_options = self.config.opt_typed_value("torch_amp")
        grad_scaler_opts = self.config.typed_value("grad_scaler", NotSpecified)
        if amp_options is not None:
            self._use_autocast = True
            if isinstance(amp_options, dict):
                amp_options = util.CollectionReadCheckCovered(amp_options)
                dtype = amp_options.get("dtype", None)
                grad_scaler_opts = amp_options.get("grad_scaler", grad_scaler_opts)
                amp_options.assert_all_read()
            elif isinstance(amp_options, str):
                dtype = amp_options
            else:
                raise TypeError(f"Invalid type for torch_amp: {type(amp_options)}")
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype)
            assert isinstance(dtype, torch.dtype) or dtype is None
            print(f"Using autocast (automatic mixed precision (AMP)) with dtype {dtype}", file=log.v2)
            self._autocast_dtype = dtype

        if grad_scaler_opts is NotSpecified:
            grad_scaler_opts = {} if self._use_autocast else None
        if grad_scaler_opts is not None:
            assert isinstance(grad_scaler_opts, dict)
            print("Using GradScaler with options:", grad_scaler_opts, file=log.v2)
            self._grad_scaler = amp.GradScaler(**grad_scaler_opts)

    def init_network_from_config(self, config: Optional[Config] = None):
        """init model"""
        assert config is self.config or not config
        super().init_network_from_config(config=config)

        extern_data_dict = self.config.typed_value("extern_data")
        assert extern_data_dict, "extern_data is not specified in config"
        self.extern_data = extern_data_util.extern_data_template_from_config_opts(extern_data_dict)

        self._load_model()

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
        assert config is self.config or not config
        config = self.config
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

        self._train_dataloader = self._create_data_loader(train_data) if train_data else None
        for dataset_name, dataset in self.eval_datasets.items():
            self._eval_dataloaders[dataset_name] = self._create_data_loader(dataset)

        self._start_epoch = self.get_train_start_epoch(self.config)
        self._final_epoch = self.config_get_final_epoch(self.config)

        self.init_network_from_config(config=config)

        self._save_model_epoch_interval = config.int("save_interval", 1)

        if self._use_torch_distributed:
            from returnn.torch.distributed import get_device_ids

            # wrap the model use torch distributed class
            self._ddp_pt_model = self._torch_distributed_class(
                self._pt_model, device_ids=get_device_ids(), **self._torch_distributed_options
            )
        self._updater = Updater(
            config=self.config, network=self._pt_model, device=self._device, initial_learning_rate=self.learning_rate
        )
        self._updater.create_optimizer()
        if self._start_epoch > 1:
            self._load_optimizer()

        self._train_step_func = self.config.typed_value("train_step")
        assert self._train_step_func, "train_step not defined"

    def train(self):
        """
        Main training loop.
        """
        assert self._pt_model is not None, "Model not initialized, call init_train_from_config()."
        assert self.train_dataset is not None, "Train dataset missing, call init_train_from_config() with train_data."

        if self._start_epoch > self._final_epoch:
            print(f"Already trained until final epoch {self._final_epoch}, nothing to do.", file=log.v3)
            return

        print(
            f"Starting training at epoch {self._start_epoch}, global train step {self.global_train_step}", file=log.v3
        )
        self.epoch = self._start_epoch - 1
        self._check_epoch_missing_eval()
        while self.epoch + 1 <= self._final_epoch:
            self.epoch += 1
            self._epoch_mp_shared.value = self.epoch

            self.init_train_epoch()
            self.train_epoch()

        print(f"Finished training at epoch {self.epoch}, global train step {self.global_train_step}", file=log.v3)

    def init_train_epoch(self):
        """
        init train (sub)epoch. LR etc
        """
        self.learning_rate = self.learning_rate_control.get_learning_rate_for_epoch(self.epoch)

        # Update learning rate
        self._updater.set_learning_rate(self.learning_rate)
        self._updater.set_current_train_step(global_train_step=self.global_train_step, epoch=self.epoch)

        self.learning_rate_control.epoch_data[self.epoch].meta.update(
            {
                "global_train_step": self.global_train_step,
                "effective_learning_rate": self._updater.get_effective_learning_rate(),
            }
        )

    def _reset_dev_memory_stats(self):
        dev = torch.device(self._device)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        self._maybe_report_dev_memory_stats()

    def _maybe_report_dev_memory_stats(self):
        if not self._log_memory_usage:
            return
        dev = torch.device(self._device)
        if dev.type == "cuda":
            stats = [
                f"alloc cur {util.human_bytes_size(torch.cuda.memory_allocated(dev))}",
                f"alloc peak {util.human_bytes_size(torch.cuda.max_memory_allocated(dev))}",
                f"reserved cur {util.human_bytes_size(torch.cuda.memory_reserved(dev))}",
                f"reserved peak {util.human_bytes_size(torch.cuda.max_memory_reserved(dev))}",
            ]
            print(f"Memory usage ({self._device}):", " ".join(stats), file=log.v1)

    def train_epoch(self):
        """
        train one (sub)epoch
        """
        print(
            "start",
            self.get_epoch_str(),
            "global train step",
            self.global_train_step,
            "with effective learning rate",
            self._updater.get_effective_learning_rate(),
            "...",
            file=log.v3,
        )

        accumulated_losses_dict = NumbersDict()
        accumulated_inv_norm_factors_dict = NumbersDict()
        step_idx = 0
        epoch_start_time = time.time()

        data_iter = iter(self._train_dataloader)
        elapsed_computation_time = 0

        self._pt_model.train()
        self._reset_dev_memory_stats()

        if self.config.bool("debug_shell_before_train_loop", False):
            print("debug_shell_before_train_loop", file=log.v1)
            debug_shell(user_ns=locals(), user_global_ns=globals(), exit_afterwards=False)

        while True:
            with torch.no_grad():
                extern_data_raw = next(data_iter, None)

            step_begin_time = time.time()

            _has_data = torch.tensor([extern_data_raw is not None], dtype=torch.int8)
            if self._use_torch_distributed:
                # use all reduce to check if all workers have data, if at least one worker does not have data,
                # all workers finish this epoch
                torch.distributed.all_reduce(_has_data, op=torch.distributed.ReduceOp.MIN)
            if not _has_data[0]:
                break

            # clear the gradients when every gradient accumulation loop starts
            if step_idx % self._accum_grad_multiple_step == 0:
                self._updater.get_optimizer().zero_grad()

            extern_data = extern_data_util.raw_dict_to_extern_data(
                extern_data_raw, extern_data_template=self.extern_data, device=self._device
            )
            self._run_step(extern_data, train_flag=True, train_func=True)

            train_ctx = rf.get_run_ctx()
            total_loss = train_ctx.total_loss()
            losses_dict = NumbersDict(
                {
                    name: float(loss.get_summed_loss().raw_tensor.detach().cpu().numpy())
                    if self._device != "meta"
                    else float("nan")
                    for name, loss in train_ctx.losses.items()
                }
            )
            inv_norm_factors_dict = NumbersDict(
                {name: float(_to_raw(loss.get_inv_norm_factor())) for name, loss in train_ctx.losses.items()}
            )

            with self._ddp_pt_model.no_sync() if self._use_torch_distributed and (
                step_idx % self._accum_grad_multiple_step
            ) != (self._accum_grad_multiple_step - 1) else nullcontext():
                if self._grad_scaler is not None:
                    self._grad_scaler.scale(total_loss.raw_tensor).backward()
                else:
                    total_loss.raw_tensor.backward()

            # only update the weights when every gradient accumulation loop ends
            if (step_idx % self._accum_grad_multiple_step) == (self._accum_grad_multiple_step - 1):
                self._updater.update_params(grad_scaler=self._grad_scaler)

            elapsed_computation_time += time.time() - step_begin_time

            accumulated_losses_dict += losses_dict
            accumulated_inv_norm_factors_dict += inv_norm_factors_dict
            _print_process(
                f"ep {self.epoch} train",
                step=step_idx,
                eval_info=dict(losses_dict / inv_norm_factors_dict),
                log_memory_usage_device=self._device if self._log_memory_usage else None,
            )

            step_idx += 1
            self.global_train_step += 1
            self._updater.set_current_train_step(global_train_step=self.global_train_step, epoch=self.epoch)

        elapsed = time.time() - epoch_start_time
        elapsed_computation_percentage = elapsed_computation_time / elapsed
        print(
            "Trained %i steps, %s elapsed (%.1f%% computing time)"
            % (step_idx, hms(elapsed), (elapsed_computation_percentage * 100.0)),
            file=log.v3,
        )

        self.learning_rate_control.epoch_data[self.epoch].meta.update(
            {
                "epoch_num_train_steps": step_idx,
                "epoch_train_time_secs": round(elapsed),
            }
        )

        accumulated_losses_dict = accumulated_losses_dict / accumulated_inv_norm_factors_dict
        self.learning_rate_control.set_epoch_error(
            self.epoch, {f"train_loss_{k}": v for k, v in accumulated_losses_dict.items()}
        )
        if self._do_save():
            self.learning_rate_control.save()

        print(f"Total train loss:", _format_score(dict(accumulated_losses_dict)), file=log.v3)

        self._maybe_report_dev_memory_stats()

        if self.epoch % self._save_model_epoch_interval == 0 or self.epoch == self._final_epoch:
            if self.model_filename:
                self._save_model()
                self._save_optimizer()
            else:
                print("Not saving model, `model` not specified.", file=log.v3)

        self.eval_model()
        if self.config.bool_or_other("cleanup_old_models", None):
            self.cleanup_old_models()

    def _do_save(self):
        if self._device == "meta":
            return False
        if not super()._do_save():
            return False
        return True

    def eval_model(self, *, skip_already_evaluated: bool = False):
        """
        Runs model on all eval datasets and calculates the loss.
        """
        self._pt_model.eval()
        self._reset_dev_memory_stats()

        eval_dump_str = []
        score_keys = None
        error_keys = None

        for dataset_name, dataset in self.eval_datasets.items():
            if skip_already_evaluated and self._is_dataset_evaluated(name=dataset_name):
                continue

            data_loader = self._eval_dataloaders[dataset_name]

            if self._use_torch_distributed and torch.distributed.get_rank() != 0:
                # We need to make sure the data loader iterator was created for proper synchronization.
                # However, now we only want to do evaluation on rank 0 for simplicity.
                iter(data_loader)
                continue
            print(f"Evaluating dataset {dataset_name!r}", file=log.v3)

            accumulated_losses_dict = NumbersDict()
            accumulated_inv_norm_factors_dict = NumbersDict()
            step_idx = 0

            with torch.no_grad():
                for extern_data_raw in data_loader:
                    extern_data = extern_data_util.raw_dict_to_extern_data(
                        extern_data_raw, extern_data_template=self.extern_data, device=self._device
                    )

                    self._run_step(extern_data, train_func=True)
                    train_ctx = rf.get_run_ctx()

                    if score_keys is None:
                        score_keys = [name for name, loss in train_ctx.losses.items() if not loss.as_error]
                        error_keys = [name for name, loss in train_ctx.losses.items() if loss.as_error]

                    losses_dict = NumbersDict(
                        {
                            name: float(loss.get_summed_loss().raw_tensor.detach().cpu().numpy())
                            if self._device != "meta"
                            else float("nan")
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
                        log_memory_usage_device=self._device if self._log_memory_usage else None,
                    )
                    step_idx += 1

            assert step_idx > 0, f"No data in dataset {dataset_name!r}."
            accumulated_losses_dict = accumulated_losses_dict / accumulated_inv_norm_factors_dict

            self.learning_rate_control.set_epoch_error(
                self.epoch, {f"{dataset_name}_loss_{k}": v for k, v in accumulated_losses_dict.items()}
            )
            if self._do_save():
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

        if not self._use_torch_distributed or torch.distributed.get_rank() == 0:
            print(" ".join(eval_dump_str) if eval_dump_str else "(No evaluations.)", file=log.v1)

        self._maybe_report_dev_memory_stats()

    def _create_data_loader(self, dataset: Dataset) -> DataLoader:
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
        if (self._min_seq_length is not None) or (self._max_seq_length is not None):
            wrapped_dataset = data_pipeline.LenFilterDataPipe(
                wrapped_dataset, min_seq_length=self._min_seq_length, max_seq_length=self._max_seq_length
            )
        chunking = self.config.typed_value("chunking", None)
        min_chunk_size = self.config.typed_value("min_chunk_size", 0)
        if chunking:
            wrapped_dataset = data_pipeline.ChunkingIterDataPipe(
                wrapped_dataset, chunking, min_chunk_size=min_chunk_size
            )

        assert self.config.typed_value("batch_size") is not None, "batch_size not defined in config"
        batch_size = self.config.typed_value("batch_size", 1)
        max_seqs = self.config.int("max_seqs", -1)
        batches_dataset = data_pipeline.BatchingIterDataPipe(wrapped_dataset, batch_size=batch_size, max_seqs=max_seqs)

        loader_opts = self.config.typed_value("torch_dataloader_opts") or {}
        assert isinstance(loader_opts, dict), f"config torch_dataloader_opts, expected dict, got {type(loader_opts)}"
        if loader_opts.get("num_workers"):
            loader_opts.setdefault("persistent_workers", True)
            loader_opts.setdefault("worker_init_fn", _data_loader_worker_init_func)

        return DataLoader(
            batches_dataset,
            collate_fn=data_pipeline.collate_batch,
            # Batching is already done by BatchingIterDataPipe.
            batch_size=None,
            # Explicitly not use the following opts, which are not supported and/or do not make sense
            # for an iterable-style dataset.
            shuffle=None,
            sampler=None,
            batch_sampler=None,
            # User-defined
            **loader_opts,
        )

    def _run_step(self, extern_data: TensorDict, *, train_flag: bool = False, train_func: bool):
        """
        :param extern_data: model inputs for the step
        """
        if train_func:
            assert self._train_step_func is not None
            rf.init_train_step_run_ctx(train_flag=train_flag, step=self.global_train_step)
        else:
            assert self._forward_step_func is not None, "define forward_step in the config"
            rf.init_forward_step_run_ctx(
                expected_outputs=self._forward_step_expected_outputs, step=self.global_train_step
            )

        from returnn.torch.distributed import ddp_train_forward_ctx

        with autocast(
            device_type=self._device, dtype=self._autocast_dtype
        ) if self._use_autocast else nullcontext(), ddp_train_forward_ctx(pt_model=self._ddp_pt_model) if isinstance(
            self._ddp_pt_model, DistributedDataParallel
        ) else nullcontext(), rf.set_default_device_ctx(
            self._device
        ):
            sentinel_kw = {"__fwd_compatible_random_arg_%i" % int(random() * 100): None}
            if train_func:
                self._train_step_func(model=self._orig_model, extern_data=extern_data, **sentinel_kw)
            else:
                self._forward_step_func(model=self._orig_model, extern_data=extern_data, **sentinel_kw)

    def _load_model(self):
        """
        Sets self._model to a torch.nn.Module.
        """
        # Check existing model. This takes `load` and `load_epoch` into account,
        # and also whether we are in train or eval mode.
        epoch, model_epoch_filename = self.get_epoch_model(self.config)
        step = None

        filename = None
        checkpoint_state = None
        if model_epoch_filename:
            filename = model_epoch_filename + util.get_model_filename_postfix()
            print("Load model %s" % (filename,), file=log.v4)
            checkpoint_state = torch.load(filename, map_location=self._device)
            if epoch is None:
                epoch = checkpoint_state["epoch"]
            step = checkpoint_state["step"]
            print(f"  epoch {epoch}, global train step {step}", file=log.v4)
            # The checkpoint was saved when the step was already increased (but not the epoch yet).
            # Restore the last step.
            # Below, we will increase the step again in case we are training.
            step -= 1

        is_training = self.config.value("task", "train") == "train"
        is_first_train_epoch = not epoch and (is_training or self.config.value("task", "train") == "initialize_model")

        if not model_epoch_filename:
            step = 0
            epoch = self._start_epoch or 1

        self._create_model(epoch=epoch, step=step)

        if checkpoint_state is not None:
            missing_keys, unexpected_keys = self._pt_model.load_state_dict(checkpoint_state["model"], strict=False)
            if missing_keys:
                raise Exception(
                    "\n".join(
                        [
                            f"While loading model {filename}:",
                            "Unexpected key(s) in state_dict: " + ", ".join(map(repr, unexpected_keys)),
                            "Missing key(s) in state_dict: " + ", ".join(map(repr, missing_keys)),
                            "Any missing key is an error.",
                        ]
                    )
                )
            if unexpected_keys:
                print(
                    f"Note: While loading {filename}, unexpected key(s) in state_dict: "
                    + ", ".join(map(repr, unexpected_keys)),
                    file=log.v4,
                )
        preload_from_files = self.config.typed_value("preload_from_files", {})
        if preload_from_files:
            # see `preload_from_files` in tf engine and `returnn.tf.network.CustomCheckpointLoader`
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

        # https://github.com/rwth-i6/returnn/issues/1345
        del checkpoint_state
        gc.collect()
        self._pt_model.to(self._device)

        if model_epoch_filename and is_training:
            # We loaded a model from a checkpoint, and the epoch and step were taken from it.
            # The epoch was just finished, so we need to increment it.
            # We decremented the step above.
            epoch += 1
            step += 1
        self.epoch = epoch  # in training, this will be reset to start_epoch
        self.global_train_step = step

    def _create_model(self, *, epoch: int, step: int):
        """
        Set up self._pt_model and self._orig_model
        by calling get_model from the config.

        Note on the `epoch` and `step` args:
        In case we are loading a model:
          This is the epoch and step of the model we are loading.
        In case we are initializing a model:
          Epoch starts at 1, step starts at 0.
        The step is the global train step, i.e. the number of train steps we have done so far over all epochs,
        so it does not reset to 0 at each epoch.
        In a checkpoint, we stored the epoch of the most recent epoch we just finished.
        We stored the global train step after we already incremented it (that's why you have step -= 1 above).
        The checkpoint is always stored when we just have finished the epoch.

        :param epoch:
        :param step:
        """
        # See :mod:`rf.rand` docstring for an explanation of this logic.
        random_seed = self.config.int("random_seed", 42)
        random_seed = (epoch * 193939 + step * 19937 + random_seed * 27644437 + 479001599) % (2**31)
        rf.set_random_seed(random_seed)

        get_model_func = self.config.typed_value("get_model")
        assert get_model_func, "get_model not defined in config"
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
        num_params = sum([parameter.numel() for parameter in self._pt_model.parameters()])
        print(f"net params #: {num_params}", file=log.v2)

    def get_pt_model(self) -> Optional[torch.nn.Module]:
        """
        :return: PyTorch Module. in case this is using RF, it will return the wrapped module
        """
        return self._pt_model

    def _save_model(self):
        """
        Saves the state of self._model to file.
        """
        if not self._do_save():
            return
        filename = self.get_epoch_model_filename() + util.get_model_filename_postfix()
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        print("Save model under %s" % (filename,), file=log.v4)
        # First write to a temp-file, to be sure that writing happens without errors,
        # and only afterward rename to the target file.
        tmp_filename = filename + ".tmp_write"
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)
        torch.save(
            {
                "model": self._pt_model.state_dict(),
                "epoch": self.epoch,
                "step": self.global_train_step,
                "effective_learning_rate": self._updater.get_effective_learning_rate() if self._updater else None,
                "returnn_version": returnn.__long_version__,
            },
            tmp_filename,
        )
        os.rename(tmp_filename, filename)

    def get_pt_optimizer(self) -> Optional[torch.optim.Optimizer]:
        """
        :return: PyTorch optimizer
        """
        if not self._updater:
            return None
        return self._updater.get_optimizer()

    def _load_optimizer(self):
        """
        Loads a torch.optim.Optimizer from disk and uses it as the optimizer.
        This function is a wrapper to Updater.load_optimizer().
        """
        filename = (
            self.get_epoch_model_filename(epoch=self._start_epoch - 1) + ".opt" + util.get_model_filename_postfix()
        )
        self._updater.load_optimizer(filename)

    def _save_optimizer(self):
        """
        Saves the optimizer state to a file.
        This function is a wrapper to Updater.save_optimizer().
        """
        if not self._do_save():
            return
        filename = self.get_epoch_model_filename() + ".opt" + util.get_model_filename_postfix()
        self._updater.save_optimizer(filename)

        # keep only the last two optimizer states (two in case one file gets corrupted)
        clean_epoch = self.epoch - 2
        if clean_epoch > 0:
            filename = self.get_epoch_model_filename(epoch=clean_epoch) + ".opt" + util.get_model_filename_postfix()
            if os.path.isfile(filename):
                os.unlink(filename)

    def forward_with_callback(self, *, dataset: Dataset, callback: ForwardCallbackIface):
        """forward"""
        assert isinstance(dataset, Dataset)
        assert isinstance(callback, ForwardCallbackIface)

        epoch_start_time = time.time()
        elapsed_computation_time = 0.0

        self._pt_model.eval()
        self._reset_dev_memory_stats()

        if dataset.supports_seq_order_sorting():
            # We can sort it. Sort it in reverse to make sure that we have enough memory right at the beginning.
            print("Dataset supports sorting, i.e. it will be sorted for optimal performance.", file=log.v3)
            dataset.seq_ordering = "sorted_reverse"
        else:
            print(
                "Dataset does not support sorting, i.e. it will not be sorted for optimal performance.",
                file=log.v3,
            )

        assert (self._min_seq_length is None) and (self._max_seq_length is None), (
            f"min_seq_length {self._min_seq_length}, max_seq_length {self._max_seq_length} not allowed,"
            f" we want to keep all source sentences."
        )

        data_loader = self._create_data_loader(dataset)
        batch_dim = extern_data_util.get_batch_dim_from_extern_data(self.extern_data)

        def _get_tensor_wo_batch_numpy(x: Tensor) -> Tensor:
            if batch_dim not in x.dims:
                raise Exception(f"Expected {batch_dim} in {x}.")
            if x.dims.index(batch_dim) != 0:
                x = x.copy_move_axis(x.dims.index(batch_dim), 0)

            y_kwargs = x.copy_template_excluding_axis(0).get_kwargs()
            y_kwargs["dims"] = [_get_dim_tag_wo_batch(dim) for dim in y_kwargs["dims"]]
            y = Tensor(**y_kwargs)

            if x.batch_ndim > 1:
                raw = x.raw_tensor[batch_idx]
            else:
                # Keep it as ndarray.
                raw = x.raw_tensor[batch_idx : batch_idx + 1].reshape(())
            # Convert it to Numpy array.
            # Note that users might also want to get the PyTorch tensors instead.
            # They might even want to get the whole batched tensor.
            # If that use cases comes up at some later point,
            # we can introduce it as an option, sth like "forward_raw_tensors" or so.
            # Currently, this callback interface is intended to also be used by other backends,
            # and then the user can always assume Numpy arrays.
            if isinstance(raw, torch.Tensor):  # might already be numpy array
                raw = raw.detach().cpu().numpy()
            y.raw_tensor = raw
            return y

        def _get_dim_tag_wo_batch(dim: Dim) -> Dim:
            """
            This is for dim tags with dyn_size_ext which include the batch_dim,
            e.g. the standard [batch] sizes.
            In the callback, we pass each sequence without the batch dim,
            so we must adapt the dim tags.
            """
            if not dim.dyn_size_ext:
                return dim
            if batch_dim not in dim.dyn_size_ext.dims:
                return dim
            new_dim = dim.copy()
            new_dim.dyn_size_ext = _get_tensor_wo_batch_numpy(dim.dyn_size_ext)
            return new_dim

        with torch.no_grad():
            callback.init(model=self._orig_model)

            step_idx = 0
            for extern_data_raw in data_loader:
                step_begin_time = time.time()
                extern_data = extern_data_util.raw_dict_to_extern_data(
                    extern_data_raw, extern_data_template=self.extern_data, device=self._device
                )
                self._run_step(extern_data, train_func=False)
                ctx = rf.get_run_ctx()
                ctx.check_outputs_complete()

                model_outputs = ctx.outputs
                for batch_idx in range(batch_dim.get_dim_value()):
                    seq_tag = extern_data["seq_tag"].raw_tensor[batch_idx].item()
                    model_outputs_per_batch = TensorDict()
                    for k, v in model_outputs.data.items():
                        model_outputs_per_batch.data[k] = _get_tensor_wo_batch_numpy(v)
                    callback.process_seq(seq_tag=seq_tag, outputs=model_outputs_per_batch)

                elapsed_computation_time += time.time() - step_begin_time
                _print_process(
                    f"ep {self.epoch} {dataset.name} forward",
                    step=step_idx,
                    eval_info=None,
                    log_memory_usage_device=self._device if self._log_memory_usage else None,
                )
                step_idx += 1

            callback.finish()

        elapsed = time.time() - epoch_start_time
        elapsed_computation_percentage = elapsed_computation_time / elapsed
        print(
            "Forward %i steps, %s elapsed (%.1f%% computing time)"
            % (step_idx, hms(elapsed), (elapsed_computation_percentage * 100.0)),
            file=log.v3,
        )

        self._maybe_report_dev_memory_stats()

    @staticmethod
    def delete_model(filename):
        """
        :param str filename:
        :return: accumulated file-size in bytes of deleted files
        :rtype: int
        """
        # This assumes PyTorch models here.
        # They consist of a file with the extension ".pt" and potentially an optimizer state with extension ".opt.pt"

        count_bytes = 0
        assert os.path.exists(filename + ".pt")
        count_bytes += os.stat(filename + ".pt").st_size
        os.remove(filename + ".pt")
        if os.path.exists(filename + "opt.pt"):
            count_bytes += os.stat(filename + "opt.pt").st_size
            os.remove(filename + "opt.pt")
        assert count_bytes > 0
        return count_bytes

    def _check_epoch_missing_eval(self):
        """
        Checks if there are outstanding tasks (eval_model) for the epoch,
        and executes them.
        """
        if not self.epoch:
            return
        if self.learning_rate_control.filename:
            for name, dataset in self.eval_datasets.items():
                if not self._is_dataset_evaluated(name=name):
                    # This can happen when we have a previous model but did not test it yet.
                    print(f"Last epoch model not yet evaluated on {name}. Doing that now.", file=log.v3)
                    self.eval_model(skip_already_evaluated=True)
                    break


def _to_raw(n: Union[int, float, Tensor]):
    if isinstance(n, (int, float)):
        return n
    if isinstance(n, Tensor):
        return n.raw_tensor.detach().cpu().numpy()
    raise TypeError(f"Unexpected {n} of type {type(n)}")


def _print_process(
    report_prefix: str,
    step: int,
    eval_info: Optional[Dict[str, Any]] = None,
    log_memory_usage_device: Optional[str] = None,
):
    """
    Similar but simplified from TF engine _print_process.

    :param report_prefix:
    :param step:
    :param eval_info:
    :param log_memory_usage_device: if given, will log memory usage (peak allocated memory)
    :return: nothing, will be printed to log
    """
    if log.verbose[5]:  # report every minibatch
        info = [report_prefix, "step %i" % step]
        if eval_info:  # Such as score.
            info += ["%s %s" % (k, _format_value(v)) for k, v in sorted(eval_info.items())]
        if log_memory_usage_device:
            dev = torch.device(log_memory_usage_device)
            if dev.type == "cuda":
                info += [
                    f"mem_usage:{log_memory_usage_device} {util.human_bytes_size(torch.cuda.max_memory_allocated())}"
                ]
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
        return _format_value(list(score.values())[0])
    return " ".join(["%s %s" % (key.split(":", 2)[-1], _format_value(score[key])) for key in sorted(score.keys())])


def _format_value(v: Any) -> str:
    if isinstance(v, float):
        if abs(v) > 1.0e3 or abs(v) < 1.0e-3:
            return f"{v:.3e}"
        else:
            return f"{v:.3f}"
    return str(v)


def _get_gpu_device() -> Optional[str]:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return None


def get_device_from_config_opt(device: Optional[str]) -> ResultWithReason[str]:
    """
    :param device: as in config
    :return: resolved device
    """
    if os.environ.get("PT_DEVICE"):
        return ResultWithReason(os.environ["PT_DEVICE"], "PT_DEVICE env var")
    if not device:
        device = _get_gpu_device()
        if device:
            return ResultWithReason(device, "GPU automatically selected")
        return ResultWithReason("cpu", "no GPU found")
    reason = "config"
    if device == "gpu":
        device = _get_gpu_device()
        if not device:
            reasons = diagnose_gpu.diagnose_no_gpu()
            raise Exception("No GPU device found, but config requested 'gpu' device.\n" + "\n".join(reasons))
        reason = "'gpu' in config"
    return ResultWithReason(device, reason)


def _data_loader_worker_init_func(worker_id: int):
    if sys.platform == "linux":
        with open("/proc/self/comm", "w") as f:
            f.write(f"TDL worker {worker_id}")
