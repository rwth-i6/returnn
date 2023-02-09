"""
Main engine for PyTorch
"""

from typing import Optional, Callable, Dict

import torch
import torch.utils.data.datapipes as dp
import numpy
import os
from torchdata.dataloader2 import DataLoader2
from random import random

from returnn.config import Config
from returnn.log import log
from returnn.engine.base import EngineBase
from returnn.learning_rate_control import load_learning_rate_control_from_config, LearningRateControl
from returnn.datasets.basic import init_dataset, Dataset
from returnn.torch.updater import Updater
from returnn.util import basic as util
from returnn.util import NumbersDict
from .data import pipeline as data_pipeline
from .data import returnn_dataset_wrapper


class Engine(EngineBase):
    """
    PyTorch engine
    """

    def __init__(self, config: Config):
        """
        :param config:
        """
        super(Engine, self).__init__()
        self.config = config
        self.model_filename = self.config.value("model", None)
        self._mp_manager = torch.multiprocessing.Manager()
        self._epoch_mp_shared = self._mp_manager.Value("i", 0)
        self.train_dataset = None  # type: Optional[Dataset]
        self.eval_datasets = {}
        self._train_dataloader = None  # type: Optional[DataLoader2]
        self._eval_dataloaders = {}  # type: Dict[str, DataLoader2]

        self._start_epoch = None  # type: Optional[int]
        self._final_epoch = None  # type: Optional[int]
        self._model = None  # type: Optional[torch.nn.Module]
        self._train_step_func = None  # type: Optional[Callable]
        self._learning_rate = 0.0
        self._learning_rate_control = None  # type: Optional[LearningRateControl]
        self._save_model_epoch_interval = 1
        self._updater = None  # type: Optional[Updater]

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

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

        self._start_epoch, _ = self.get_train_start_epoch_batch(self.config)
        self._final_epoch = self.config_get_final_epoch(self.config)

        self._load_model(epoch=self._start_epoch)
        self._learning_rate_control = load_learning_rate_control_from_config(config)
        self._learning_rate = self._learning_rate_control.get_learning_rate_for_epoch(self._start_epoch)
        self._save_model_epoch_interval = config.int("save_interval", 1)

        self._updater = Updater(self.config, self._model, self._learning_rate)
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
        assert self._model, "Model not initialized, call init_train_from_config()."

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
        self._learning_rate = self._learning_rate_control.get_learning_rate_for_epoch(self.epoch)

        # Update learning rate
        self._updater.set_learning_rate(self._learning_rate)

    def train_epoch(self):
        """
        train one (sub)epoch
        """
        print("start", self.get_epoch_str(), "with learning rate", self._learning_rate, "...", file=log.v4)

        self._model.train()

        step_idx = 0
        for data in self._train_dataloader:
            loss, _ = self._run_step(data)

            self._updater.get_optimizer().zero_grad()
            loss.backward()
            self._updater.get_optimizer().step()

            print("step %i, loss: %f" % (step_idx, loss.detach().cpu().numpy()), file=log.v4)

            step_idx += 1

        print("Trained %i steps" % step_idx)

        if self.epoch % self._save_model_epoch_interval == 0 or self.epoch == self._final_epoch:
            self._save_model()
            self._save_optimizer()

        self.eval_model()

    def eval_model(self):
        """
        Runs model on all eval datasets and calculates the loss.
        """
        self._model.eval()

        for dataset_name, dataset in self.eval_datasets.items():
            print(f"Evaluating dataset {dataset_name!r}'", file=log.v3)

            data_loader = self._eval_dataloaders[dataset_name]

            accumulated_loss = 0.0
            accumulated_losses_dict = NumbersDict()
            step_idx = 0

            with torch.no_grad():
                for data in data_loader:

                    total_loss, losses_dict = self._run_step(data)
                    total_loss = total_loss.detach().cpu().numpy()
                    losses_dict = {
                        dataset_name + "_score_" + name: float(loss.detach().cpu().numpy())
                        for name, loss in losses_dict.items()
                    }
                    print("step %i, loss: %f" % (step_idx, total_loss), file=log.v4)

                    accumulated_loss += total_loss
                    accumulated_losses_dict += NumbersDict(losses_dict)
                    step_idx += 1

            assert step_idx > 0, "No data in dataset '{}'.".format(dataset_name)
            accumulated_loss = accumulated_loss / step_idx
            accumulated_losses_dict = accumulated_losses_dict / step_idx

            self._learning_rate_control.set_epoch_error(self.epoch, dict(accumulated_losses_dict))

            print("Total loss for '{}': {:.6}".format(dataset_name, accumulated_loss), file=log.v3)

        self._learning_rate_control.save()

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

        return DataLoader2(batches_dataset)

    def _run_step(self, data):
        """
        :param dict[str, numpy.ndarray] data: model inputs for the step
        :return: total loss (weighted sum) calculated for the step, and individual losses as a name -> value mapping
        :rtype: tuple[torch.Tensor, dict[str, torch.Tensor]]
        """
        assert isinstance(data, dict) and data
        data = {
            k: v.cpu() if k.endswith(":seq_len") else v.to(self._device) for (k, v) in data.items()
        }  # Sequence lengths have to be on CPU for the later call to rnn.pack_padded_sequence

        train_ctx = TrainCtx()
        sentinel_kw = {"__fwd_compatible_random_arg_%i" % int(random() * 100): None}
        self._train_step_func(model=self._model, data=data, train_ctx=train_ctx, **sentinel_kw)
        losses_dict = train_ctx.losses
        total_loss = train_ctx.total_loss()

        return total_loss, losses_dict

    def _load_model(self, epoch):
        """
        Sets self._model to a torch.nn.Module.

        :param int epoch:
        """
        get_model_func = self.config.typed_value("get_model")
        assert get_model_func, "get_model not defined"
        sentinel_kw = {"__fwd_compatible_random_arg_%i" % int(random() * 100): None}
        self._model = get_model_func(**sentinel_kw)
        assert isinstance(self._model, torch.nn.Module)

        if epoch > 1:
            filename = self.get_epoch_model_filename(epoch=epoch - 1) + util.get_model_filename_postfix()
            print("Load model %s" % (filename,), file=log.v4)
            model_state = torch.load(filename)
            self._model.load_state_dict(model_state)

        self._model.to(self._device)

    def _save_model(self):
        """
        Saves the state of self._model to file.
        """
        filename = self.get_epoch_model_filename() + util.get_model_filename_postfix()
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        print("Save model under %s" % (filename,), file=log.v4)
        torch.save(self._model.state_dict(), filename)

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
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        self._updater.save_optimizer(filename)


class TrainCtx:
    """
    train ctx
    """

    def __init__(self):
        self.losses = {}  # type: Dict[str, torch.Tensor]
        self.loss_scales = {}  # type: Dict[str, float]

    def mark_as_loss(self, name: str, loss: torch.Tensor, scale: float = 1.0):
        """
        Can be called several times. Total loss will be weighted sum according to 'scale' parameters.
        The total loss is used for backpropagation, so the scale is only relevant for backprop,
        not for reporting.

        :param name:
        :param loss: e.g. the output of nn.CrossEntropyLoss
        :param scale: optional factor for this loss. only relevant for backprop.
        """
        assert isinstance(name, str)
        self.losses[name] = loss
        self.loss_scales[name] = scale

    def total_loss(self) -> torch.Tensor:
        """
        :return: total loss, as it is used for backpropagation
        """
        assert self.losses, "call train_ctx.mark_as_loss"
        return sum([self.losses[name] * self.loss_scales[name] for name in self.losses], torch.zeros(()))
