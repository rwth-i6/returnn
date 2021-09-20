
"""
Provides :class:`EngineBase`.
"""

from __future__ import print_function

import os
import sys
import typing
from returnn.log import log
from returnn.pretrain import Pretrain
from returnn.util import basic as util


class EngineBase(object):
  """
  Base class for a backend engine, such as :class:`TFEngine.Engine`.
  """

  def __init__(self):
    self.epoch = 0
    self.pretrain = None  # type: typing.Optional[Pretrain]
    self.model_filename = None  # type: typing.Optional[str]

  @classmethod
  def config_get_final_epoch(cls, config):
    """
    :param Config.Config config:
    :rtype: int
    """
    num_epochs = config.int('num_epochs', 5)
    if config.has("load_epoch"):
      num_epochs = max(num_epochs, config.int("load_epoch", 0))
    return num_epochs

  @classmethod
  def get_existing_models(cls, config):
    """
    :param Config.Config config:
    :return: dict epoch -> model filename
    :rtype: dict[int,str]
    """
    model_filename = config.value('model', '')
    if not model_filename:
      return []
    # Automatically search the filesystem for existing models.
    file_list = {}
    for epoch in range(1, cls.config_get_final_epoch(config) + 1):
      for is_pretrain in [False, True]:
        fn = cls.epoch_model_filename(model_filename, epoch, is_pretrain)
        if os.path.exists(fn):
          file_list[epoch] = fn
          break
        if util.BackendEngine.is_tensorflow_selected():
          if os.path.exists(fn + ".index"):
            file_list[epoch] = fn
            break
    return file_list

  @classmethod
  def get_epoch_model(cls, config):
    """
    :type config: Config.Config
    :returns (epoch, modelFilename)
    :rtype: (int|None, str|None)
    """
    start_epoch_mode = config.value('start_epoch', 'auto')
    if start_epoch_mode == 'auto':
      start_epoch = None
    else:
      start_epoch = int(start_epoch_mode)
      assert start_epoch >= 1

    load_model_epoch_filename = util.get_checkpoint_filepattern(config.value('load', ''))
    if load_model_epoch_filename:
      assert os.path.exists(load_model_epoch_filename + util.get_model_filename_postfix())

    import_model_train_epoch1 = util.get_checkpoint_filepattern(config.value('import_model_train_epoch1', ''))
    if import_model_train_epoch1:
      assert os.path.exists(import_model_train_epoch1 + util.get_model_filename_postfix())

    existing_models = cls.get_existing_models(config)
    load_epoch = config.int("load_epoch", -1)
    if load_model_epoch_filename:
      if load_epoch <= 0:
        load_epoch = util.model_epoch_from_filename(load_model_epoch_filename)
    else:
      if load_epoch > 0:  # ignore if load_epoch == 0
        assert load_epoch in existing_models
        load_model_epoch_filename = existing_models[load_epoch]
        assert util.model_epoch_from_filename(load_model_epoch_filename) == load_epoch

    # Only use this when we don't train.
    # For training, we first consider existing models before we take the 'load' into account when in auto epoch mode.
    # In all other cases, we use the model specified by 'load'.
    if load_model_epoch_filename and (config.value('task', 'train') != 'train' or start_epoch is not None):
      if config.value('task', 'train') == 'train' and start_epoch is not None:
        # Ignore the epoch. To keep it consistent with the case below.
        epoch = None
      else:
        epoch = load_epoch
      epoch_model = (epoch, load_model_epoch_filename)

    # In case of training, always first consider existing models.
    # This is because we reran RETURNN training, we usually don't want to train from scratch
    # but resume where we stopped last time.
    elif existing_models:
      epoch_model = sorted(existing_models.items())[-1]
      if load_model_epoch_filename:
        print("note: there is a 'load' which we ignore because of existing model", file=log.v4)

    elif config.value('task', 'train') == 'train' and import_model_train_epoch1 and start_epoch in [None, 1]:
      epoch_model = (0, import_model_train_epoch1)

    # Now, consider this also in the case when we train, as an initial model import.
    elif load_model_epoch_filename:
      # Don't use the model epoch as the start epoch in training.
      # We use this as an import for training.
      epoch_model = (load_epoch, load_model_epoch_filename)

    else:
      epoch_model = (None, None)

    if start_epoch == 1:
      if epoch_model[0]:  # existing model
        print("warning: there is an existing model: %s" % (epoch_model,), file=log.v4)
        epoch_model = (None, None)
    elif (start_epoch or 0) > 1:
      if epoch_model[0]:
        if epoch_model[0] != start_epoch - 1:
          print("warning: start_epoch %i but there is %s" % (start_epoch, epoch_model), file=log.v4)
        epoch_model = start_epoch - 1, existing_models[start_epoch - 1]

    return epoch_model

  @classmethod
  def get_train_start_epoch_batch(cls, config):
    """
    We will always automatically determine the best start (epoch,batch) tuple
    based on existing model files.
    This ensures that the files are present and enforces that there are
    no old outdated files which should be ignored.
    Note that epochs start at idx 1 and batches at idx 0.
    :type config: Config.Config
    :returns (epoch,batch)
    :rtype (int,int)
    """
    start_batch_mode = config.value('start_batch', 'auto')
    if start_batch_mode == 'auto':
      start_batch_config = None
    else:
      start_batch_config = int(start_batch_mode)
    last_epoch, _ = cls.get_epoch_model(config)
    if last_epoch is None:
      start_epoch = 1
      start_batch = start_batch_config or 0
    elif start_batch_config is not None:
      # We specified a start batch. Stay in the same epoch, use that start batch.
      start_epoch = last_epoch
      start_batch = start_batch_config
    else:
      # Start with next epoch.
      start_epoch = last_epoch + 1
      start_batch = 0
    return start_epoch, start_batch

  @classmethod
  def epoch_model_filename(cls, model_filename, epoch, is_pretrain):
    """
    :type model_filename: str
    :type epoch: int
    :type is_pretrain: bool
    :rtype: str
    """
    if sys.platform == "win32" and model_filename.startswith("/tmp/"):
      import tempfile
      model_filename = tempfile.gettempdir() + model_filename[len("/tmp"):]
    return model_filename + (".pretrain" if is_pretrain else "") + ".%03d" % epoch

  def get_epoch_model_filename(self, epoch=None):
    """
    :param int|None epoch:
    :return: filename, excluding TF specific postfix
    :rtype: str
    """
    if not epoch:
      epoch = self.epoch
    return self.epoch_model_filename(self.model_filename, epoch, self.is_pretrain_epoch(epoch=epoch))

  def get_epoch_str(self):
    """
    :return: e.g. "epoch 3", or "pretrain epoch 5"
    :rtype: str
    """
    return ("pretrain " if self.is_pretrain_epoch() else "") + "epoch %s" % self.epoch

  def is_pretrain_epoch(self, epoch=None):
    """
    :param int|None epoch:
    :return: whether this epoch is covered by the pretrain logic
    :rtype: bool
    """
    if not epoch:
      epoch = self.epoch
    return self.pretrain and epoch <= self.pretrain.get_train_num_epochs()

  def is_first_epoch_after_pretrain(self):
    """
    :return: whether the current epoch is the first epoch right after pretraining
    :rtype: bool
    """
    return self.pretrain and self.epoch == self.pretrain.get_train_num_epochs() + 1
