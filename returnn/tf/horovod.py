"""
Here we encapsulate some common Horovod functions.

Note that you are supposed to be able to import this module even if Horovod is not installed.

The usage of this module / global context is also considered optional at this point.
Horovod is enabled <==> ``use_horovod`` is enabled in the config.

For relevant further config options, see the code of :class:`HorovodContext` below.
Most importantly:

* ``horovod_dataset_distribution``, recommended value ``"random_seed_offset"``, default value ``"shard"``
* ``horovod_reduce_type``, recommended value ``"param"``, default value ``"grad"``
* ``horovod_param_sync_step``, recommended value ``100``, default value ``1``
* ``horovod_param_sync_time_diff``, alternative to ``horovod_param_sync_step``, e.g. ``100.`` (secs), default ``None``

Also see :ref:`multi_gpu`.
Also see :mod:`TFDistributed`.
"""

import os
import socket
import typing
from returnn.config import Config


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
    # Note: Rather than reading the config here, we will instead read it in the functions below on-the-fly.
    # This allows e.g. the pretraining or custom get_network to temporarily overwrite the behavior for some epoch.

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
    reduce_type = self._config.value("horovod_reduce_type", "grad")
    assert reduce_type in {"grad", "param"}
    return reduce_type

  def is_reduce_type_grad(self):
    """
    :rtype: bool
    """
    return self.get_reduce_type() == "grad"

  def is_reduce_type_param(self):
    """
    :rtype: bool
    """
    return self.get_reduce_type() == "param"

  def get_param_sync_time_diff(self):
    """
    :rtype: float|None
    """
    assert self.is_reduce_type_param()
    return self._config.float("horovod_param_sync_time_diff", None)

  def get_param_sync_step(self):
    """
    :rtype: int
    """
    assert self.is_reduce_type_param()
    return self._config.int("horovod_param_sync_step", 1)

  def get_dataset_distribution_type(self):
    """
    :rtype: str
    """
    dataset_distribution = self._config.value("horovod_dataset_distribution", "shard")
    assert dataset_distribution in {"shard", "random_seed_offset"}
    return dataset_distribution

  def is_dataset_distribution_shard(self):
    """
    :rtype: bool
    """
    return self.get_dataset_distribution_type() == "shard"

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
    return self.get_dataset_distribution_type() == "random_seed_offset"

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
    from returnn.config import get_global_config
    config = get_global_config(raise_exception=False)
    if not config:
      return None
  _is_set_up = True
  if not config.is_true("use_horovod"):
    return None
  _ctx = HorovodContext(config=config)
  return _ctx
