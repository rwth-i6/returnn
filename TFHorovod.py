"""
Here we encapsulate some common Horovod functions.

Note that you are supposed to be able to import this module even if Horovod is not installed.

The usage of this module / global context is also considered optional at this point.
Horovod is enabled <==> ``use_horovod`` is enabled in the config.

Also see :mod:`TFDistributed`.
"""

import os
import socket
import typing
from Config import Config


class HorovodContext:
  """
  This setups some helper functions.
  """
  def __init__(self, config):
    """
    :param Config config:
    """
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    import horovod.tensorflow as hvd
    hvd.init()
    print(
      "Horovod initialized. Hostname %s, pid %i, rank %i / size %i, local rank %i / local size %i." % (
        socket.gethostname(), os.getpid(), hvd.rank(), hvd.size(), hvd.local_rank(), hvd.local_size()))
    self._config = config
    self._hvd_mod = hvd
    self._local_rank = hvd.local_rank()
    self._local_size = hvd.local_size()
    self._rank = hvd.rank()
    self._size = hvd.size()
    self._reduce_type = self._config.value("horovod_reduce_type", "grad")
    assert self._reduce_type in {"grad", "param"}
    self._param_sync_step = config.int("horovod_param_sync_step", 1)
    self._dataset_distribution = self._config.value("horovod_dataset_distribution", "shard")
    assert self._dataset_distribution in {"shard", "random_seed_offset"}

  def should_sync_every_step(self):
    """
    :return: whether we should sync every step.
      This is both for the signal for more data, and also loss/error/score reduction.
    :rtype: bool
    """
    if self.is_dataset_distribution_random_seed_offset() and self.is_reduce_type_param():
      # In this setting, we don't need a sync every step.
      # Thus avoid it to have faster more async training.
      return False
    # By default, we should sync every step.
    return True

  def get_reduce_type(self):
    """
    :rtype: str
    """
    return self._reduce_type

  def is_reduce_type_grad(self):
    """
    :rtype: bool
    """
    return self._reduce_type == "grad"

  def is_reduce_type_param(self):
    """
    :rtype: bool
    """
    return self._reduce_type == "param"

  def get_param_sync_step(self):
    """
    :rtype: int
    """
    assert self.is_reduce_type_param()
    return self._param_sync_step

  def is_dataset_distribution_shard(self):
    """
    :rtype: bool
    """
    return self._dataset_distribution == "shard"

  def get_dataset_shard_batch_slice(self):
    """
    :rtype: slice
    """
    assert self.is_dataset_distribution_shard()
    return slice(self.rank(), None, self.size())

  def is_dataset_distribution_random_seed_offset(self):
    """
    :rtype: bool
    """
    return self._dataset_distribution == "random_seed_offset"

  def rank(self):
    """
    :rtype: int
    """
    return self._rank

  def size(self):
    """
    :rtype: int
    """
    return self._size

  def local_rank(self):
    """
    :rtype: int
    """
    return self._local_rank

  def local_size(self):
    """
    :rtype: int
    """
    return self._local_size


_is_set_up = False
_ctx = None  # type: typing.Optional[HorovodContext]


def get_ctx(config=None):
  """
  :param Config|None config:
  :returns: the global context if Horovod is enabled, or None otherwise.
    If we did not setup the context yet, it will automatically create it.
  :rtype: HorovodContext|None
  """
  global _is_set_up, _ctx
  if _is_set_up:
    return _ctx
  if not config:
    from Config import get_global_config
    config = get_global_config()
  _is_set_up = True
  if not config.is_true("use_horovod"):
    return None
  _ctx = HorovodContext(config=config)
  return _ctx
