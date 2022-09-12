
"""
Provides the learning rate scheduling logic.
The base class is :class:`LearningRateControl`.
"""

from __future__ import print_function

import os
import typing
from returnn.util.basic import better_repr, simple_obj_repr, ObjAsDict, unicode
from returnn.log import log
import numpy


class LearningRateControl(object):
  """
  Base class for learning rate control / scheduling.
  """

  need_error_info = True

  class EpochData:
    """
    Encapsulates all relevant information for one epoch,
    needed to perform learning rate scheduling,
    such as the individual scores (cv or train; cross-entropy or frame-error or whatever).
    """

    # Need to keep the non-PEP8 name for compatibility, because we store the repr of the object.
    # noinspection PyPep8Naming
    def __init__(self, learningRate, error=None):
      """
      :type learningRate: float
      :type error: dict[str,float] | None
      """
      self.learning_rate = learningRate
      if isinstance(error, float):  # Old format.
        error = {"old_format_score": error}
      if error is None:
        error = {}
      self.error = error

    def __repr__(self):
      # This is being used for serialization, and we want some forward/backward compatibility,
      # so we should try to keep this consistent.
      return "EpochData(learningRate=%s, error=%s)" % (
        better_repr(self.learning_rate), better_repr(self.error))

  @classmethod
  def load_initial_kwargs_from_config(cls, config):
    """
    :type config: returnn.config.Config
    :rtype: dict[str]
    """
    return {
      "default_learning_rate": config.float('learning_rate', 1.0),
      "min_learning_rate": config.float('min_learning_rate', 0.0),
      "default_learning_rates": config.typed_value('learning_rates') or config.float_list('learning_rates'),
      "error_measure_key": (
        config.typed_value('learning_rate_control_error_measure')
        or config.value('learning_rate_control_error_measure', None)),
      "relative_error_also_relative_to_learning_rate": (
        config.bool('learning_rate_control_relative_error_relative_lr', False)),
      "min_num_epochs_per_new_learning_rate": config.int("learning_rate_control_min_num_epochs_per_new_lr", 0),
      "relative_error_div_by_old": config.bool('newbob_relative_error_div_by_old', False),
      "learning_rate_decay": config.typed_value(
        'learning_rate_decay', config.opt_typed_value('newbob_learning_rate_decay', 0.5)),
      "learning_rate_growth": config.typed_value(
        'learning_rate_growth', config.opt_typed_value('newbob_learning_rate_growth', 1.0)),
      "filename": config.value('learning_rate_file', None),
    }

  @classmethod
  def load_initial_from_config(cls, config):
    """
    :type config: returnn.config.Config
    :rtype: LearningRateControl
    """
    kwargs = cls.load_initial_kwargs_from_config(config)
    return cls(**kwargs)

  def __init__(self, default_learning_rate, min_learning_rate=0.0, default_learning_rates=None,
               error_measure_key=None,
               relative_error_also_relative_to_learning_rate=False,
               min_num_epochs_per_new_learning_rate=0,
               relative_error_div_by_old=False,
               learning_rate_decay=1.0,
               learning_rate_growth=1.0,
               filename=None):
    """
    :param float default_learning_rate: default learning rate. usually for epoch 1
    :param list[float] | dict[int,float] default_learning_rates: learning rates
    :param str|list[str]|None error_measure_key: for get_epoch_error_value() the key for EpochData.error which is a dict
    :param int min_num_epochs_per_new_learning_rate: if the lr was recently updated, use it for at least N epochs
    :param bool relative_error_div_by_old: if True, compute relative error as (new - old) / old.
    :param float|(float)->float learning_rate_decay:
    :param float|(float)->float learning_rate_growth:
    :param str filename: load from and save to file
    """
    self.epoch_data = {}  # type: typing.Dict[int,LearningRateControl.EpochData]
    self.filename = filename
    if filename:
      if os.path.exists(filename):
        print("Learning-rate-control: loading file %s" % filename, file=log.v4)
        # Load now, such that default_learning_rates is correctly handled.
        self.load()
      else:
        print("Learning-rate-control: file %s does not exist yet" % filename, file=log.v4)
    else:
      print("Learning-rate-control: no file specified, not saving history (no proper restart possible)", file=log.v4)
    self.default_learning_rate = default_learning_rate
    self.min_learning_rate = min_learning_rate
    if default_learning_rates:
      if isinstance(default_learning_rates, list):
        default_learning_rates = {i + 1: v for (i, v) in enumerate(default_learning_rates)}
      if isinstance(default_learning_rates, (str, unicode)):
        default_learning_rates = eval(default_learning_rates)
      assert isinstance(default_learning_rates, dict)
      for epoch, v in default_learning_rates.items():
        self.set_default_learning_rate_for_epoch(epoch, v)
    self.default_learning_rates = default_learning_rates
    self.error_measure_key = error_measure_key
    self.relative_error_also_relative_to_learning_rate = relative_error_also_relative_to_learning_rate
    self.min_num_epochs_per_new_learning_rate = min_num_epochs_per_new_learning_rate
    self.relative_error_div_by_old = relative_error_div_by_old
    self.learning_rate_decay = learning_rate_decay
    self.learning_rate_growth = learning_rate_growth

  __repr__ = simple_obj_repr

  def __str__(self):
    epochs = sorted(self.epoch_data.keys())
    if len(epochs) > 6:
      epoch_str = ", ".join(
        ["%i: %s" % (epoch, self.epoch_data[epoch]) for epoch in epochs[:3]] +
        ["..."] +
        ["%i: %s" % (epoch, self.epoch_data[epoch]) for epoch in epochs[-3:]])
    else:
      epoch_str = ", ".join(["%i: %s" % (epoch, self.epoch_data[epoch]) for epoch in epochs])
    return "%r, epoch data: %s, error key: %s" % (self, epoch_str, self.get_error_key(epoch=1))

  @staticmethod
  def _calc_learning_rate_update(learning_rate, update):
    """
    :param float learning_rate:
    :param None|float|(float)->float update: factor, or generic func
    :return: lr with update applied (e.g. decay factor)
    :rtype: float
    """
    if update is None:
      return learning_rate
    if isinstance(update, float):
      return learning_rate * update
    assert callable(update)
    learning_rate = update(learning_rate)
    assert isinstance(learning_rate, float)
    return learning_rate

  def _calc_learning_rate_decay(self, learning_rate):
    """
    :param float learning_rate:
    :return: lr with decay applied
    :rtype: float
    """
    return self._calc_learning_rate_update(learning_rate, update=self.learning_rate_decay)

  def _calc_learning_rate_growth(self, learning_rate):
    """
    :param float learning_rate:
    :return: lr with growth applied
    :rtype: float
    """
    return self._calc_learning_rate_update(learning_rate, update=self.learning_rate_growth)

  def calc_learning_rate_decay_or_grow(self, learning_rate, decay, grow=None):
    """
    :param float learning_rate:
    :param bool decay:
    :param bool|None grow: default is not decay
    :return: lr with decay or growth applied
    :rtype: float
    """
    assert isinstance(decay, bool)
    if grow is None:
      grow = not decay
    assert isinstance(grow, bool)
    assert not (grow and decay)  # not sure if this makes sense...
    if decay:
      learning_rate = self._calc_learning_rate_decay(learning_rate)
      if learning_rate < self.min_learning_rate:
        learning_rate = self.min_learning_rate
    if grow:
      learning_rate = self._calc_learning_rate_growth(learning_rate)
    return learning_rate

  def calc_learning_rate_for_epoch(self, epoch):
    """
    :type epoch: int
    :returns learning rate
    :rtype: float
    """
    raise NotImplementedError

  def calc_new_learning_rate_for_epoch(self, epoch):
    """
    :param int epoch:
    :return: new learning rate for this epoch
    :rtype: float
    """
    if self.min_num_epochs_per_new_learning_rate > 1:
      last_lrs = [self.epoch_data[e].learning_rate
                  for e in self._last_epochs_for_epoch(epoch, num_epochs=self.min_num_epochs_per_new_learning_rate)]
      if len(set(last_lrs)) >= 2 or 0 < len(last_lrs) < self.min_num_epochs_per_new_learning_rate:
        return last_lrs[-1]
    learning_rate = self.calc_learning_rate_for_epoch(epoch)
    return learning_rate

  def _last_epochs_for_epoch(self, epoch, num_epochs):
    """
    :param int epoch:
    :param int num_epochs:
    :return: last N epochs where we have some epoch data
    :rtype: list[int]
    """
    last_epochs = sorted([e for e in self.epoch_data.keys() if e < epoch])
    if not last_epochs:
      return []
    last_epochs = last_epochs[-num_epochs:]
    return last_epochs

  def get_learning_rate_for_epoch(self, epoch):
    """
    :type epoch: int
    :rtype: float
    """
    assert epoch >= 1
    if epoch in self.epoch_data:
      return self.epoch_data[epoch].learning_rate
    learning_rate = self.calc_new_learning_rate_for_epoch(epoch)
    self.set_default_learning_rate_for_epoch(epoch, learning_rate)
    return learning_rate

  def set_default_learning_rate_for_epoch(self, epoch, learning_rate):
    """
    :type epoch: int
    :type learning_rate: float
    """
    if epoch in self.epoch_data:
      if not self.epoch_data[epoch].learning_rate:
        self.epoch_data[epoch].learning_rate = learning_rate
    else:
      self.epoch_data[epoch] = self.EpochData(learning_rate)

  def get_last_epoch(self, epoch):
    """
    :param int epoch:
    :return: last epoch before ``epoch`` where we have some epoch data
    :rtype: int
    """
    epochs = sorted([e for e in self.epoch_data.keys() if e < epoch])
    if not epochs:
      return None
    return epochs[-1]

  def get_most_recent_learning_rate(self, epoch, exclude_current=True):
    """
    :param int epoch:
    :param bool exclude_current:
    :return: most learning rate before or including ``epoch``
    :rtype: float
    """
    for e, data in reversed(sorted(self.epoch_data.items())):
      assert isinstance(data, LearningRateControl.EpochData)
      if e > epoch:
        continue
      if exclude_current and e == epoch:
        continue
      if data.learning_rate is None:
        continue
      return data.learning_rate
    return self.default_learning_rate

  def calc_relative_error(self, old_epoch, new_epoch):
    """
    :param int old_epoch:
    :param int new_epoch:
    :return: relative error between old epoch and new epoch
    :rtype: float
    """
    old_key, old_error = self.get_epoch_error_key_value(old_epoch)
    new_key, new_error = self.get_epoch_error_key_value(new_epoch)
    if old_error is None or new_error is None:
      return None
    if old_key != new_key:
      return None
    if self.relative_error_div_by_old:
      relative_error = (new_error - old_error) / abs(old_error)
    else:
      relative_error = (new_error - old_error) / abs(new_error)
    if self.relative_error_also_relative_to_learning_rate:
      learning_rate = self.get_most_recent_learning_rate(new_epoch, exclude_current=False)
      if learning_rate > 0:
        # If the learning rate is lower than the initial learning rate,
        # the relative error is also expected to be lower, so correct for that here.
        relative_error /= learning_rate / self.default_learning_rate
    return relative_error

  def set_epoch_error(self, epoch, error):
    """
    :type epoch: int
    :type error: dict[str,float|dict[str,float]]
    """
    if epoch not in self.epoch_data:
      print("Learning rate not set for epoch %i. Assuming default." % epoch, file=log.v4)
      self.get_learning_rate_for_epoch(epoch)  # This will set it.
    assert isinstance(error, dict)
    error = error.copy()
    for k, v in list(error.items()):
      if isinstance(v, dict):  # like error = {"dev_score": {"cost:output1": .., "cost:output2": ...}, ...}
        del error[k]
        if len(v) == 1:
          error[k] = list(v.values())[0]
          continue
        for k1, v1 in v.items():
          if ":" in k1:
            k1 = k1[k1.index(":") + 1:]
          error[k + "_" + k1] = v1
    for v in error.values():
      assert isinstance(v, float)
    self.epoch_data[epoch].error.update(error)
    if epoch == 1:
      print("Learning-rate-control: error key %r from %r" % (self.get_error_key(epoch), error), file=log.v4)

  def get_error_key(self, epoch):
    """
    :param int epoch:
    :return: key which we should look in scores/errors, for this epoch
    :rtype: str
    """
    if epoch not in self.epoch_data:
      if isinstance(self.error_measure_key, list):
        return self.error_measure_key[0]
      assert isinstance(self.error_measure_key, (str, type(None)))
      return self.error_measure_key
    epoch_data = self.epoch_data[epoch]
    if not epoch_data.error:
      return None
    if len(epoch_data.error) == 1 and "old_format_score" in epoch_data.error:
      return "old_format_score"
    keys = []
    if isinstance(self.error_measure_key, list):
      for key in self.error_measure_key:
        keys += [key, key + "_output"]  # for multiple outputs, try default output
    elif isinstance(self.error_measure_key, str):
      keys += [self.error_measure_key, self.error_measure_key + "_output"]
    else:
      assert self.error_measure_key is None
    keys += ["dev_score", "dev_score_output"]
    for key in keys:
      if key in epoch_data.error:
        return key
    for key in sorted(epoch_data.error.keys()):
      if key == "dev_score_output/output" or key.startswith("dev_score_output/output_"):
        return key
    for key in sorted(epoch_data.error.keys()):
      if key.startswith("dev_score_output/"):
        return key
    for key in sorted(epoch_data.error.keys()):
      if key.startswith("dev_"):
        return key
    for key in ["train_score", "train_score_output"]:
      if key in epoch_data.error:
        return key
    return min(epoch_data.error.keys())

  def get_epoch_error_dict(self, epoch):
    """
    :param int epoch:
    :rtype: dict[str,float]
    """
    if epoch not in self.epoch_data:
      return {}
    return self.epoch_data[epoch].error

  def get_epoch_error_value(self, epoch):
    """
    :param int epoch:
    :return: error/score for the specific epoch, given the error-key, see :func:`get_error_key`
    :rtype: float
    """
    error = self.get_epoch_error_dict(epoch)
    if not error:
      return None
    key = self.get_error_key(epoch)
    assert key
    assert key in error, (
      "%r not in %r. fix %r in config. set it to %r or so." % (
        key, error, 'learning_rate_control_error_measure', 'dev_error'))
    return error[key]

  def get_epoch_error_key_value(self, epoch):
    """
    :param int epoch:
    :return: key, error
    :rtype: (str, float)
    """
    error = self.get_epoch_error_dict(epoch)
    if not error:
      return None, None
    key = self.get_error_key(epoch)
    assert key
    assert key in error, (
      "%r not in %r. fix %r in config. set it to %r or so." %
      (key, error, 'learning_rate_control_error_measure', 'dev_error'))
    return key, error[key]

  def get_last_best_epoch(self, last_epoch, first_epoch=1, filter_score=float("inf"), only_last_n=-1,
                          min_score_dist=0.0):
    """
    :param int first_epoch: will check all epochs >= first_epoch
    :param int last_epoch: inclusive. will check all epochs <= last_epoch
    :param float filter_score: all epochs which values over this score are not considered
    :param int only_last_n: if set (>=1), from the resulting list, we consider only the last only_last_n
    :param float min_score_dist: filter out epochs where the diff to the most recent is not big enough
    :return: the last best epoch. to get the details then, you might want to use getEpochErrorDict.
    :rtype: int|None
    """
    if first_epoch > last_epoch:
      return None
    values = [(self.get_epoch_error_key_value(ep), ep) for ep in range(first_epoch, last_epoch + 1)]
    # Note that the order of the checks here is a bit arbitrary but I had some thoughts on it.
    # Changing the order will also slightly change the behavior, so be sure it make sense.
    values = [((key, v), ep) for ((key, v), ep) in values if v is not None]
    if not values:
      return None
    last_key, latest_score = values[-1][0]
    values = [(v, ep) for ((key, v), ep) in values if key == last_key]  # only same key
    values = [(v, ep) for (v, ep) in values if v <= filter_score]
    if not values:
      return None
    if only_last_n >= 1:
      values = values[-only_last_n:]
    values = [(v, ep) for (v, ep) in values if v + min_score_dist < latest_score]
    if not values:
      return None
    return min(values)[1]

  def save(self):
    """
    Save the current epoch data to file (self.filename).
    """
    if not self.filename:
      return
    # First write to a temp-file, to be sure that the write happens without errors.
    # Otherwise, it could happen that we delete the old existing file, then
    # some error happens (e.g. disk quota), and we loose the newbob data.
    # Loosing that data is very bad because it basically means that we have to redo all the training.
    tmp_filename = self.filename + ".new_tmp"
    f = open(tmp_filename, "w")
    f.write(better_repr(self.epoch_data))
    f.write("\n")
    f.close()
    os.rename(tmp_filename, self.filename)

  def load(self):
    """
    Loads the saved epoch data from file (self.filename).
    """
    s = open(self.filename).read()
    self.epoch_data = eval(s, {"nan": float("nan"), "inf": float("inf")}, ObjAsDict(self))


class ConstantLearningRate(LearningRateControl):
  """
  Just a constant learning rate.
  """

  need_error_info = False

  def calc_learning_rate_for_epoch(self, epoch):
    """
    Dummy constant learning rate. Returns initial learning rate.
    :type epoch: int
    :returns learning rate
    :rtype: float
    """
    while True:
      last_epoch = self.get_last_epoch(epoch)
      if last_epoch is None:
        return self.default_learning_rate
      learning_rate = self.epoch_data[last_epoch].learning_rate
      if learning_rate is None:
        epoch = last_epoch
        continue
      return learning_rate


class NewbobRelative(LearningRateControl):
  """
  If relative diff between old and new error is over some threshold, decay learning rate.
  """

  @classmethod
  def load_initial_kwargs_from_config(cls, config):
    """
    :type config: returnn.config.Config
    :rtype: dict[str]
    """
    kwargs = super(NewbobRelative, cls).load_initial_kwargs_from_config(config)
    kwargs.update({
      "relative_error_threshold": config.float('newbob_relative_error_threshold', -0.01),
    })
    return kwargs

  def __init__(self, relative_error_threshold, **kwargs):
    """
    :type relative_error_threshold: float
    """
    super(NewbobRelative, self).__init__(**kwargs)
    self.relative_error_threshold = relative_error_threshold

  def calc_learning_rate_for_epoch(self, epoch):
    """
    Newbob+ on train data.
    :type epoch: int
    :returns learning rate
    :rtype: float
    """
    last_epoch = self.get_last_epoch(epoch)
    if last_epoch is None:
      return self.default_learning_rate
    learning_rate = self.epoch_data[last_epoch].learning_rate
    if learning_rate is None:
      return self.default_learning_rate
    last2_epoch = self.get_last_epoch(last_epoch)
    if last2_epoch is None:
      return learning_rate
    relative_error = self.calc_relative_error(last2_epoch, last_epoch)
    if relative_error is None:
      return learning_rate
    learning_rate = self.calc_learning_rate_decay_or_grow(
      learning_rate, decay=relative_error > self.relative_error_threshold)
    return learning_rate


class NewbobAbs(LearningRateControl):
  """
  If absolute diff between old and new error is over some threshold, decay learning rate.
  """

  @classmethod
  def load_initial_kwargs_from_config(cls, config):
    """
    :type config: returnn.config.Config
    :rtype: dict[str]
    """
    kwargs = super(NewbobAbs, cls).load_initial_kwargs_from_config(config)
    kwargs.update({
      "error_threshold": config.float('newbob_error_threshold', -0.01),
    })
    return kwargs

  def __init__(self, error_threshold, **kwargs):
    """
    :type error_threshold: float
    """
    super(NewbobAbs, self).__init__(**kwargs)
    self.error_threshold = error_threshold

  def calc_learning_rate_for_epoch(self, epoch):
    """
    Newbob+ on train data.

    :type epoch: int
    :returns learning rate
    :rtype: float
    """
    last_epoch = self.get_last_epoch(epoch)
    if last_epoch is None:
      return self.default_learning_rate
    learning_rate = self.epoch_data[last_epoch].learning_rate
    if learning_rate is None:
      return self.default_learning_rate
    last2_epoch = self.get_last_epoch(last_epoch)
    if last2_epoch is None:
      return learning_rate
    old_key, old_error = self.get_epoch_error_key_value(last2_epoch)
    new_key, new_error = self.get_epoch_error_key_value(last_epoch)
    if old_error is None or new_error is None:
      return learning_rate
    if old_key != new_key:
      return learning_rate
    error_diff = new_error - old_error
    learning_rate = self.calc_learning_rate_decay_or_grow(learning_rate, decay=error_diff > self.error_threshold)
    return learning_rate


class NewbobMultiEpoch(LearningRateControl):
  """
  Like :class:`NewbobRelative`, but looks at the average relative error over multiple epochs.
  This is useful together with ``partition_epoch`` from :class:`Dataset`.
  """

  @classmethod
  def load_initial_kwargs_from_config(cls, config):
    """
    :type config: returnn.config.Config
    :rtype: dict[str]
    """
    kwargs = super(NewbobMultiEpoch, cls).load_initial_kwargs_from_config(config)
    kwargs.update({
      "num_epochs": config.int("newbob_multi_num_epochs", 5),
      "update_interval": config.int("newbob_multi_update_interval", config.int("newbob_multi_num_epochs", 5)),
      "relative_error_threshold": config.float('newbob_relative_error_threshold', -0.01),
    })
    return kwargs

  def __init__(self, num_epochs, update_interval, relative_error_threshold, **kwargs):
    """
    :param int num_epochs:
    :param int update_interval:
    :param float relative_error_threshold:
    """
    super(NewbobMultiEpoch, self).__init__(**kwargs)
    self.num_epochs = num_epochs
    assert self.num_epochs >= 1
    self.update_interval = update_interval
    assert self.update_interval >= 1
    self.relative_error_threshold = relative_error_threshold

  def _calc_mean_relative_error(self, epochs):
    """
    :param list[int] epochs:
    :return: mean of relative errors
    :rtype: float|None
    """
    assert len(epochs) >= 2
    errors = [self.calc_relative_error(epochs[i], epochs[i + 1]) for i in range(len(epochs) - 1)]
    if any([e is None for e in errors]):
      return None
    return float(numpy.mean(errors))

  def _calc_recent_mean_relative_error(self, epoch):
    """
    :param int epoch:
    :return: recent mean of relative errors
    :rtype: float|None
    """
    # Take one more than numEpochs because we are looking at the diffs.
    last_epochs = self._last_epochs_for_epoch(epoch, num_epochs=self.num_epochs + 1)
    if not last_epochs:
      return None
    # We could also use the self.numEpochs limit here. But maybe this is better.
    if len(last_epochs) <= 1:
      return None
    return self._calc_mean_relative_error(last_epochs)

  def calc_learning_rate_for_epoch(self, epoch):
    """
    Newbob+ on train data.
    :type epoch: int
    :returns learning rate
    :rtype: float
    """
    learning_rate = self.get_most_recent_learning_rate(epoch)
    # We start counting epochs at 1.
    if self.update_interval > 1 and epoch % self.update_interval != 1:
      return learning_rate
    mean_relative_error = self._calc_recent_mean_relative_error(epoch)
    if mean_relative_error is None:
      return learning_rate
    learning_rate = self.calc_learning_rate_decay_or_grow(
      learning_rate, decay=mean_relative_error > self.relative_error_threshold)
    return learning_rate


def learning_rate_control_type(type_name):
  """
  :param str type_name:
  :rtype: type[LearningRateControl]|LearningRateControl
  """
  if type_name == "constant":
    return ConstantLearningRate
  elif type_name in ("newbob", "newbob_rel", "newbob_relative"):  # Old setups expect the relative version.
    return NewbobRelative
  elif type_name == "newbob_abs":
    return NewbobAbs
  elif type_name == "newbob_multi_epoch":
    return NewbobMultiEpoch
  else:
    assert False, "unknown learning-rate-control type %s" % type_name


def load_learning_rate_control_from_config(config):
  """
  :type config: returnn.config.Config
  :rtype: LearningRateControl
  """
  control_type = config.value("learning_rate_control", "constant")
  cls = learning_rate_control_type(control_type)
  return cls.load_initial_from_config(config)


def demo():
  """
  Demo run. Given some learning rate file (with scores / existing lrs), will calculate how lrs would have been set,
  given some config.
  """
  from returnn.util import better_exchook
  better_exchook.install()
  import returnn.__main__ as rnn
  import sys
  if len(sys.argv) <= 1:
    print("usage: python %s [config] [other options] [++check_learning_rates 1]" % __file__)
    print(
      ("example usage: "
       "python %s ++learning_rate_control newbob ++learning_rate_file newbob.data ++learning_rate 0.001") % __file__)
  rnn.init_config(command_line_options=sys.argv[1:])
  # noinspection PyProtectedMember
  rnn.config._hack_value_reading_debug()
  rnn.config.update({"log": []})
  rnn.init_log()
  rnn.init_backend_engine()
  check_lr = rnn.config.bool("check_learning_rates", False)
  from returnn.pretrain import pretrain_from_config
  pretrain = pretrain_from_config(rnn.config)
  first_non_pretrain_epoch = 1
  pretrain_learning_rate = None
  if pretrain:
    first_non_pretrain_epoch = pretrain.get_train_num_epochs() + 1
  log.initialize(verbosity=[5])
  control = load_learning_rate_control_from_config(rnn.config)
  print("LearningRateControl: %r" % control)
  if not control.epoch_data:
    print("No epoch data so far.")
    return
  first_epoch = min(control.epoch_data.keys())
  if first_epoch != 1:
    print("Strange, first epoch from epoch data is %i." % first_epoch)
  print("Error key: %s from %r" % (control.get_error_key(epoch=first_epoch), control.epoch_data[first_epoch].error))
  if pretrain:
    pretrain_learning_rate = rnn.config.float('pretrain_learning_rate', control.default_learning_rate)
  max_epoch = max(control.epoch_data.keys())
  for epoch in range(1, max_epoch + 2):  # all epochs [1..max_epoch+1]
    old_learning_rate = None
    if epoch in control.epoch_data:
      old_learning_rate = control.epoch_data[epoch].learning_rate
    if epoch < first_non_pretrain_epoch:
      learning_rate = pretrain_learning_rate
      s = "Pretrain epoch %i, fixed learning rate: %s (was: %s)" % (epoch, learning_rate, old_learning_rate)
    elif 1 < first_non_pretrain_epoch == epoch:
      learning_rate = control.default_learning_rate
      s = "First epoch after pretrain, epoch %i, fixed learning rate: %s (was %s)" % (
        epoch, learning_rate, old_learning_rate)
    else:
      learning_rate = control.calc_new_learning_rate_for_epoch(epoch)
      s = "Calculated learning rate for epoch %i: %s (was: %s)" % (epoch, learning_rate, old_learning_rate)
    if learning_rate < control.min_learning_rate:
      learning_rate = control.min_learning_rate
      s += ", clipped to %s" % learning_rate
    s += ", previous relative error: %s" % control.calc_relative_error(epoch - 2, epoch - 1)
    if hasattr(control, "_calc_recent_mean_relative_error"):
      # noinspection PyProtectedMember
      s += ", previous mean relative error: %s" % control._calc_recent_mean_relative_error(epoch)
    print(s)
    if check_lr and old_learning_rate is not None:
      if old_learning_rate != learning_rate:
        print("Learning rate is different in epoch %i!" % epoch)
        sys.exit(1)
    # Overwrite new learning rate so that the calculation for further learning rates stays consistent.
    if epoch in control.epoch_data:
      control.epoch_data[epoch].learning_rate = learning_rate
    else:
      control.epoch_data[epoch] = control.EpochData(learningRate=learning_rate)
  print("Finished, last stored epoch was %i." % max_epoch)


if __name__ == "__main__":
  demo()
