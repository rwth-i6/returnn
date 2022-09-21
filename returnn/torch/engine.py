"""
Main engine for PyTorch
"""

import torch
from torch.utils.data import DataLoader
from random import random

from returnn.log import log
from returnn.engine.base import EngineBase
from returnn.datasets.basic import init_dataset
from . import data_pipeline


class Engine(EngineBase):
  """
  PyTorch engine
  """

  def __init__(self, config):
    """
    :param returnn.config.Config config:
    """
    super(Engine, self).__init__()
    self.config = config
    self.train_dataset = None
    self.eval_datasets = {}
    self.learning_rate = config.float("learning_rate", 1.)  # TODO LR control...

  def init_train_from_config(self, config=None, train_data=None, dev_data=None, eval_data=None):
    """
    :param returnn.config.Config|None config:
    :param returnn.datasets.basic.Dataset|None train_data:
    :param returnn.datasets.basic.Dataset|None dev_data:
    :param returnn.datasets.basic.Dataset|None eval_data:
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

  def train(self):
    """
    Main training loop.
    """
    start_epoch, _ = self.get_train_start_epoch_batch(self.config)
    final_epoch = self.config_get_final_epoch(self.config)

    print("Starting training at epoch {}.".format(start_epoch), file=log.v3)

    self.epoch = start_epoch
    while self.epoch <= final_epoch:
      self.init_train_epoch()
      self.train_epoch()

      self.epoch += 1

    print("Finished training at epoch {}.".format(self.epoch), file=log.v3)

  def init_train_epoch(self):
    """
    init train (sub)epoch. LR etc
    """
    pass

  def train_epoch(self):
    """
    train one (sub)epoch
    """
    print("start", self.get_epoch_str(), "with learning rate", self.learning_rate, "...", file=log.v4)

    train_data = data_pipeline.DatasetWrapper(self.train_dataset, epoch=self.epoch)

    batch_max_seqs = self.config.int('max_seqs', 1)  # TODO wrong default, actually -1, no limit (limit via batch_size)
    data_loader = DataLoader(
      train_data,
      batch_size=batch_max_seqs,
      collate_fn=data_pipeline.collate_batch,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    get_model_func = self.config.typed_value("get_model")
    assert get_model_func, "get_model not defined"
    sentinel_kw = {"__fwd_compatible_random_arg_%i" % int(random() * 100): None}
    model = get_model_func(epoch=self.epoch, **sentinel_kw)
    assert isinstance(model, torch.nn.Module)
    model.to(device)
    model.train()
    train_step_func = self.config.typed_value("train_step")
    assert train_step_func, "train_step not defined"
    optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)

    step_idx = 0
    for data in data_loader:
      assert isinstance(data, dict) and data
      data = {k: v.to(device) for (k, v) in data.items()}

      train_ctx = TrainCtx()
      train_step_func(model=model, data=data, train_ctx=train_ctx, **sentinel_kw)
      loss = train_ctx.total_loss()

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      print("step %i, loss: %f" % (step_idx, loss.detach().cpu().numpy()), file=log.v4)

      step_idx += 1

    print("Trained %i steps" % step_idx)


class TrainCtx:
  """
  train ctx
  """

  def __init__(self):
    self.loss = None

  def mark_as_loss(self, loss):
    """
    :param torch.Tensor loss:
    """
    assert self.loss is None, "don't call multiple times"
    self.loss = loss

  def total_loss(self):
    """
    :rtype: torch.Tensor
    """
    assert self.loss is not None, "call train_ctx.mark_as_loss"
    return self.loss
