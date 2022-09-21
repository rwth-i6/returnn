"""
Main engine for PyTorch
"""

from torch.utils.data import DataLoader

from returnn.log import log
from returnn.engine.base import EngineBase
from returnn.datasets.basic import init_dataset
from returnn.torch.dataset_wrapper import DatasetWrapper


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

    train_data = DatasetWrapper(self.train_dataset, epoch=self.epoch)

    batch_max_seqs = self.config.int('max_seqs', 1)  # TODO wrong default, actually -1, no limit (limit via batch_size)
    data_loader = DataLoader(train_data, batch_size=batch_max_seqs)  # TODO: implement batching

    for data in data_loader:
      assert data  # TODO: only iterates through dataset so far
