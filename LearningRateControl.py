
from __future__ import print_function

import os
from Util import betterRepr, simpleObjRepr, ObjAsDict, unicode
from Log import log
import numpy


class LearningRateControl(object):

  need_error_info = True

  class EpochData:
    def __init__(self, learningRate, error=None):
      """
      :type learningRate: float
      :type error: dict[str,float] | None
      """
      self.learningRate = learningRate
      if isinstance(error, float):  # Old format.
        error = {"old_format_score": error}
      if error is None:
        error = {}
      self.error = error

    __repr__ = simpleObjRepr

  @classmethod
  def load_initial_kwargs_from_config(cls, config):
    """
    :type config: Config.Config
    :rtype: dict[str]
    """
    return {
      "defaultLearningRate": config.float('learning_rate', 1.0),
      "minLearningRate": config.float('min_learning_rate', 0.0),
      "defaultLearningRates": config.typed_value('learning_rates') or config.float_list('learning_rates'),
      "errorMeasureKey": config.typed_value('learning_rate_control_error_measure')
                         or config.value('learning_rate_control_error_measure', None),
      "relativeErrorAlsoRelativeToLearningRate": config.bool('learning_rate_control_relative_error_relative_lr', False),
      "minNumEpochsPerNewLearningRate": config.int("learning_rate_control_min_num_epochs_per_new_lr", 0),
      "relativeErrorDivByOld": config.bool('newbob_relative_error_div_by_old', False),
      "filename": config.value('learning_rate_file', None),
    }

  @classmethod
  def load_initial_from_config(cls, config):
    """
    :type config: Config.Config
    :rtype: LearningRateControl
    """
    kwargs = cls.load_initial_kwargs_from_config(config)
    return cls(**kwargs)

  def __init__(self, defaultLearningRate, minLearningRate=0.0, defaultLearningRates=None,
               errorMeasureKey=None,
               relativeErrorAlsoRelativeToLearningRate=False,
               minNumEpochsPerNewLearningRate=0,
               relativeErrorDivByOld=False,
               filename=None):
    """
    :param float defaultLearningRate: default learning rate. usually for epoch 1
    :param list[float] | dict[int,float] defaultLearningRates: learning rates
    :param str|list[str]|None errorMeasureKey: for getEpochErrorValue() the selector for EpochData.error which is a dict
    :param int minNumEpochsPerNewLearningRate: if the lr was recently updated, use it for at least N epochs
    :param bool relativeErrorDivByOld: if True, compute relative error as (new - old) / old.
    :param str filename: load from and save to file
    """
    self.epochData = {}  # type: dict[int,LearningRateControl.EpochData]
    self.defaultLearningRate = defaultLearningRate
    self.minLearningRate = minLearningRate
    if defaultLearningRates:
      if isinstance(defaultLearningRates, list):
        defaultLearningRates = {i + 1: v for (i, v) in enumerate(defaultLearningRates)}
      if isinstance(defaultLearningRates, (str, unicode)):
        defaultLearningRates = eval(defaultLearningRates)
      assert isinstance(defaultLearningRates, dict)
      for epoch, v in defaultLearningRates.items():
        self.setDefaultLearningRateForEpoch(epoch, v)
    self.defaultLearningRates = defaultLearningRates
    self.errorMeasureKey = errorMeasureKey
    self.relativeErrorAlsoRelativeToLearningRate = relativeErrorAlsoRelativeToLearningRate
    self.minNumEpochsPerNewLearningRate = minNumEpochsPerNewLearningRate
    self.relativeErrorDivByOld = relativeErrorDivByOld
    self.filename = filename
    if filename:
      if os.path.exists(filename):
        print("Learning-rate-control: loading file %s" % filename, file=log.v4)
        self.load()
      else:
        print("Learning-rate-control: file %s does not exist yet" % filename, file=log.v4)
    else:
      print("Learning-rate-control: no file specified, not saving history (no proper restart possible)", file=log.v4)

  __repr__ = simpleObjRepr

  def __str__(self):
    return "%r, epoch data: %s, error key: %s" % \
           (self, ", ".join(["%i: %s" % (epoch, self.epochData[epoch])
                             for epoch in sorted(self.epochData.keys())]),
            self.getErrorKey(epoch=1))

  def calcLearningRateForEpoch(self, epoch):
    """
    :type epoch: int
    :returns learning rate
    :rtype: float
    """
    raise NotImplementedError

  def calcNewLearnignRateForEpoch(self, epoch):
    if self.minNumEpochsPerNewLearningRate > 1:
      lastLrs = [self.epochData[e].learningRate
                 for e in self._lastEpochsForEpoch(epoch, numEpochs=self.minNumEpochsPerNewLearningRate)]
      if len(set(lastLrs)) >= 2 or epoch < self.minNumEpochsPerNewLearningRate:
        return lastLrs[-1]
    learningRate = self.calcLearningRateForEpoch(epoch)
    if learningRate < self.minLearningRate:
      return self.minLearningRate
    return learningRate

  def _lastEpochsForEpoch(self, epoch, numEpochs):
    lastEpochs = sorted([e for e in self.epochData.keys() if e < epoch])
    if not lastEpochs:
      return []
    lastEpochs = lastEpochs[-numEpochs:]
    return lastEpochs

  def getLearningRateForEpoch(self, epoch):
    """
    :type epoch: int
    :rtype: float
    """
    assert epoch >= 1
    if epoch in self.epochData: return self.epochData[epoch].learningRate
    learningRate = self.calcNewLearnignRateForEpoch(epoch)
    self.setDefaultLearningRateForEpoch(epoch, learningRate)
    return learningRate

  def setDefaultLearningRateForEpoch(self, epoch, learningRate):
    """
    :type epoch: int
    :type learningRate: float
    """
    if epoch in self.epochData:
      if not self.epochData[epoch].learningRate:
        self.epochData[epoch].learningRate = learningRate
    else:
      self.epochData[epoch] = self.EpochData(learningRate)

  def getLastEpoch(self, epoch):
    epochs = sorted([e for e in self.epochData.keys() if e < epoch])
    if not epochs:
      return None
    return epochs[-1]

  def getMostRecentLearningRate(self, epoch, excludeCurrent=True):
    for e, data in reversed(sorted(self.epochData.items())):
      if e > epoch: continue
      if excludeCurrent and e == epoch: continue
      if data.learningRate is None: continue
      return data.learningRate
    return self.defaultLearningRate

  def calcRelativeError(self, oldEpoch, newEpoch):
    oldKey, oldError = self.getEpochErrorKeyValue(oldEpoch)
    newKey, newError = self.getEpochErrorKeyValue(newEpoch)
    if oldError is None or newError is None:
      return None
    if oldKey != newKey:
      return None
    if self.relativeErrorDivByOld:
      relativeError = (newError - oldError) / abs(oldError)
    else:
      relativeError = (newError - oldError) / abs(newError)
    if self.relativeErrorAlsoRelativeToLearningRate:
      learningRate = self.getMostRecentLearningRate(newEpoch, excludeCurrent=False)
      # If the learning rate is lower than the initial learning rate,
      # the relative error is also expected to be lower, so correct for that here.
      relativeError /= learningRate / self.defaultLearningRate
    return relativeError

  def setEpochError(self, epoch, error):
    """
    :type epoch: int
    :type error: dict[str,float|dict[str,float]]
    """
    if epoch not in self.epochData:
      print("Learning rate not set for epoch %i. Assuming default." % epoch, file=log.v4)
      self.getLearningRateForEpoch(epoch)  # This will set it.
    assert isinstance(error, dict)
    error = error.copy()
    for k, v in list(error.items()):
      if isinstance(v, dict):  # like error = {"dev_score": {"cost:output1": .., "cost:output2": ...}, ...}
        del error[k]
        if len(v) == 1:
          error[k] = list(v.values())[0]
          continue
        for k1, v1 in v.items():
          if ":" in k1: k1 = k1[k1.index(":") + 1:]
          error[k + "_" + k1] = v1
    for v in error.values():
      assert isinstance(v, float)
    self.epochData[epoch].error.update(error)
    if epoch == 1:
      print("Learning-rate-control: error key %r from %r" % (self.getErrorKey(epoch), error), file=log.v4)

  def getErrorKey(self, epoch):
    if epoch not in self.epochData:
      if isinstance(self.errorMeasureKey, list):
        return self.errorMeasureKey[0]
      assert isinstance(self.errorMeasureKey, (str, type(None)))
      return self.errorMeasureKey
    epoch_data = self.epochData[epoch]
    if not epoch_data.error:
      return None
    if len(epoch_data.error) == 1 and "old_format_score" in epoch_data.error:
      return "old_format_score"
    keys = []
    if isinstance(self.errorMeasureKey, list):
      for key in self.errorMeasureKey:
        keys += [key, key + "_output"]  # for multiple outputs, try default output
    elif isinstance(self.errorMeasureKey, str):
      keys += [self.errorMeasureKey, self.errorMeasureKey + "_output"]
    else:
      assert self.errorMeasureKey is None
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

  def getEpochErrorDict(self, epoch):
    if epoch not in self.epochData:
      return {}
    return self.epochData[epoch].error

  def getEpochErrorValue(self, epoch):
    error = self.getEpochErrorDict(epoch)
    if not error:
      return None
    key = self.getErrorKey(epoch)
    assert key
    assert key in error, "%r not in %r. fix %r in config. set it to %r or so." % \
                         (key, error, 'learning_rate_control_error_measure', 'dev_error')
    return error[key]

  def getEpochErrorKeyValue(self, epoch):
    error = self.getEpochErrorDict(epoch)
    if not error:
      return None, None
    key = self.getErrorKey(epoch)
    assert key
    assert key in error, "%r not in %r. fix %r in config. set it to %r or so." % \
                         (key, error, 'learning_rate_control_error_measure', 'dev_error')
    return key, error[key]

  def getLastBestEpoch(self, last_epoch, first_epoch=1, filter_score=float("inf"), only_last_n=-1, min_score_dist=0.0):
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
    values = [(self.getEpochErrorKeyValue(ep), ep) for ep in range(first_epoch, last_epoch + 1)]
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
    if not self.filename: return
    # First write to a temp-file, to be sure that the write happens without errors.
    # Otherwise, it could happen that we delete the old existing file, then
    # some error happens (e.g. disk quota), and we loose the newbob data.
    # Loosing that data is very bad because it basically means that we have to redo all the training.
    tmp_filename = self.filename + ".new_tmp"
    f = open(tmp_filename, "w")
    f.write(betterRepr(self.epochData))
    f.write("\n")
    f.close()
    os.rename(tmp_filename, self.filename)

  def load(self):
    s = open(self.filename).read()
    self.epochData = eval(s, {"nan": float("nan"), "inf": float("inf")}, ObjAsDict(self))


class ConstantLearningRate(LearningRateControl):

  need_error_info = False

  def calcLearningRateForEpoch(self, epoch):
    """
    Dummy constant learning rate. Returns initial learning rate.
    :type epoch: int
    :returns learning rate
    :rtype: float
    """
    while True:
      lastEpoch = self.getLastEpoch(epoch)
      if lastEpoch is None:
        return self.defaultLearningRate
      learningRate = self.epochData[lastEpoch].learningRate
      if learningRate is None:
        epoch = lastEpoch
        continue
      return learningRate


class NewbobRelative(LearningRateControl):

  @classmethod
  def load_initial_kwargs_from_config(cls, config):
    """
    :type config: Config.Config
    :rtype: dict[str]
    """
    kwargs = super(NewbobRelative, cls).load_initial_kwargs_from_config(config)
    kwargs.update({
      "relativeErrorThreshold": config.float('newbob_relative_error_threshold', -0.01),
      "learningRateDecayFactor": config.float('newbob_learning_rate_decay', 0.5)})
    return kwargs

  def __init__(self, relativeErrorThreshold, learningRateDecayFactor, **kwargs):
    """
    :param float defaultLearningRate: learning rate for epoch 1+2
    :type relativeErrorThreshold: float
    :type learningRateDecayFactor: float
    :type filename: str
    """
    super(NewbobRelative, self).__init__(**kwargs)
    self.relativeErrorThreshold = relativeErrorThreshold
    self.learningRateDecayFactor = learningRateDecayFactor

  def calcLearningRateForEpoch(self, epoch):
    """
    Newbob+ on train data.
    :type epoch: int
    :returns learning rate
    :rtype: float
    """
    lastEpoch = self.getLastEpoch(epoch)
    if lastEpoch is None:
      return self.defaultLearningRate
    learningRate = self.epochData[lastEpoch].learningRate
    if learningRate is None:
      return self.defaultLearningRate
    last2Epoch = self.getLastEpoch(lastEpoch)
    if last2Epoch is None:
      return learningRate
    relativeError = self.calcRelativeError(last2Epoch, lastEpoch)
    if relativeError is None:
      return learningRate
    if relativeError > self.relativeErrorThreshold:
      learningRate *= self.learningRateDecayFactor
    return learningRate


class NewbobAbs(LearningRateControl):

  @classmethod
  def load_initial_kwargs_from_config(cls, config):
    """
    :type config: Config.Config
    :rtype: dict[str]
    """
    kwargs = super(NewbobAbs, cls).load_initial_kwargs_from_config(config)
    kwargs.update({
      "errorThreshold": config.float('newbob_error_threshold', -0.01),
      "learningRateDecayFactor": config.float('newbob_learning_rate_decay', 0.5)})
    return kwargs

  def __init__(self, errorThreshold, learningRateDecayFactor, **kwargs):
    """
    :type errorThreshold: float
    :type learningRateDecayFactor: float
    """
    super(NewbobAbs, self).__init__(**kwargs)
    self.errorThreshold = errorThreshold
    self.learningRateDecayFactor = learningRateDecayFactor

  def calcLearningRateForEpoch(self, epoch):
    """
    Newbob+ on train data.
    :type epoch: int
    :returns learning rate
    :rtype: float
    """
    lastEpoch = self.getLastEpoch(epoch)
    if lastEpoch is None:
      return self.defaultLearningRate
    learningRate = self.epochData[lastEpoch].learningRate
    if learningRate is None:
      return self.defaultLearningRate
    last2Epoch = self.getLastEpoch(lastEpoch)
    if last2Epoch is None:
      return learningRate
    oldKey, oldError = self.getEpochErrorKeyValue(last2Epoch)
    newKey, newError = self.getEpochErrorKeyValue(lastEpoch)
    if oldError is None or newError is None:
      return learningRate
    if oldKey != newKey:
      return learningRate
    errorDiff = newError - oldError
    if errorDiff > self.errorThreshold:
      learningRate *= self.learningRateDecayFactor
    return learningRate


class NewbobMultiEpoch(LearningRateControl):

  @classmethod
  def load_initial_kwargs_from_config(cls, config):
    """
    :type config: Config.Config
    :rtype: dict[str]
    """
    kwargs = super(NewbobMultiEpoch, cls).load_initial_kwargs_from_config(config)
    kwargs.update({
      "numEpochs": config.int("newbob_multi_num_epochs", 5),
      "updateInterval": config.int("newbob_multi_update_interval", config.int("newbob_multi_num_epochs", 5)),
      "relativeErrorThreshold": config.float('newbob_relative_error_threshold', -0.01),
      "learningRateDecayFactor": config.float('newbob_learning_rate_decay', 0.5),
      "learningRateGrowthFactor": config.float('newbob_learning_rate_growth', 1.0),
      })
    return kwargs

  def __init__(self, numEpochs,  updateInterval,
               relativeErrorThreshold, learningRateDecayFactor, learningRateGrowthFactor=1.0, **kwargs):
    """
    :param float defaultLearningRate: learning rate for epoch 1+2
    :param int numEpochs:
    :param int updateInterval:
    :param float relativeErrorThreshold:
    :param float learningRateDecayFactor:
    :param int filename:
    """
    super(NewbobMultiEpoch, self).__init__(**kwargs)
    self.numEpochs = numEpochs
    assert self.numEpochs >= 1
    self.updateInterval = updateInterval
    assert self.updateInterval >= 1
    self.relativeErrorThreshold = relativeErrorThreshold
    self.learningRateDecayFactor = learningRateDecayFactor
    self.learningRateGrowthFactor = learningRateGrowthFactor

  def _calcMeanRelativeError(self, epochs):
    """
    :param list[int] epochs:
    :return: mean of relative errors
    :rtype: float|None
    """
    assert len(epochs) >= 2
    errors = [self.calcRelativeError(epochs[i], epochs[i + 1]) for i in range(len(epochs) - 1)]
    if any([e is None for e in errors]):
      return None
    return numpy.mean(errors)

  def _calcRecentMeanRelativeError(self, epoch):
    """
    :param int epoch:
    :return: recent mean of relative errors
    :rtype: float|None
    """
    # Take one more than numEpochs because we are looking at the diffs.
    lastEpochs = self._lastEpochsForEpoch(epoch, numEpochs=self.numEpochs + 1)
    if not lastEpochs:
      return None
    # We could also use the self.numEpochs limit here. But maybe this is better.
    if len(lastEpochs) <= 1:
      return None
    return self._calcMeanRelativeError(lastEpochs)

  def calcLearningRateForEpoch(self, epoch):
    """
    Newbob+ on train data.
    :type epoch: int
    :returns learning rate
    :rtype: float
    """
    learningRate = self.getMostRecentLearningRate(epoch)
    # We start counting epochs at 1.
    if self.updateInterval > 1 and epoch % self.updateInterval != 1:
      return learningRate
    meanRelativeError = self._calcRecentMeanRelativeError(epoch)
    if meanRelativeError is None:
      return learningRate
    if meanRelativeError > self.relativeErrorThreshold:
      learningRate *= self.learningRateDecayFactor
    else:
      learningRate *= self.learningRateGrowthFactor
    return learningRate


def learningRateControlType(typeName):
  if typeName == "constant":
    return ConstantLearningRate
  elif typeName in ("newbob", "newbob_rel", "newbob_relative"):  # Old setups expect the relative version.
    return NewbobRelative
  elif typeName == "newbob_abs":
    return NewbobAbs
  elif typeName == "newbob_multi_epoch":
    return NewbobMultiEpoch
  else:
    assert False, "unknown learning-rate-control type %s" % typeName


def loadLearningRateControlFromConfig(config):
  """
  :type config: Config.Config
  :rtype: LearningRateControl
  """
  controlType = config.value("learning_rate_control", "constant")
  cls = learningRateControlType(controlType)
  return cls.load_initial_from_config(config)


def demo():
  import better_exchook
  better_exchook.install()
  import rnn
  import sys
  if len(sys.argv) <= 1:
    print("usage: python %s [config] [other options] [++check_learning_rates 1]" % __file__)
    print("example usage: python %s ++learning_rate_control newbob ++learning_rate_file newbob.data ++learning_rate 0.001" % __file__)
  rnn.initConfig(commandLineOptions=sys.argv[1:])
  rnn.config._hack_value_reading_debug()
  rnn.config.update({"log": []})
  rnn.initLog()
  rnn.initBackendEngine()
  check_lr = rnn.config.bool("check_learning_rates", False)
  from Pretrain import pretrainFromConfig
  pretrain = pretrainFromConfig(rnn.config)
  first_non_pretrain_epoch = 1
  pretrain_learning_rate = None
  if pretrain:
    first_non_pretrain_epoch = pretrain.get_train_num_epochs() + 1
  log.initialize(verbosity=[5])
  control = loadLearningRateControlFromConfig(rnn.config)
  print("LearningRateControl: %r" % control)
  if not control.epochData:
    print("No epoch data so far.")
    return
  firstEpoch = min(control.epochData.keys())
  if firstEpoch != 1:
    print("Strange, first epoch from epoch data is %i." % firstEpoch)
  print("Error key: %s from %r" % (control.getErrorKey(epoch=firstEpoch), control.epochData[firstEpoch].error))
  if pretrain:
    pretrain_learning_rate = rnn.config.float('pretrain_learning_rate', control.defaultLearningRate)
  maxEpoch = max(control.epochData.keys())
  for epoch in range(1, maxEpoch + 2):  # all epochs [1..maxEpoch+1]
    oldLearningRate = None
    if epoch in control.epochData:
      oldLearningRate = control.epochData[epoch].learningRate
    if epoch < first_non_pretrain_epoch:
      learningRate = pretrain_learning_rate
      s = "Pretrain epoch %i, fixed learning rate: %s (was: %s)" % (epoch, learningRate, oldLearningRate)
    elif first_non_pretrain_epoch > 1 and epoch == first_non_pretrain_epoch:
      learningRate = control.defaultLearningRate
      s = "First epoch after pretrain, epoch %i, fixed learning rate: %s (was %s)" % (epoch, learningRate, oldLearningRate)
    else:
      learningRate = control.calcNewLearnignRateForEpoch(epoch)
      s = "Calculated learning rate for epoch %i: %s (was: %s)" % (epoch, learningRate, oldLearningRate)
    if learningRate < control.minLearningRate:
      learningRate = control.minLearningRate
      s += ", clipped to %s" % learningRate
    s += ", previous relative error: %s" % control.calcRelativeError(epoch - 2, epoch - 1)
    if hasattr(control, "_calcRecentMeanRelativeError"):
      s += ", previous mean relative error: %s" % control._calcRecentMeanRelativeError(epoch)
    print(s)
    if check_lr and oldLearningRate is not None:
      if oldLearningRate != learningRate:
        print("Learning rate is different in epoch %i!" % epoch)
        sys.exit(1)
    # Overwrite new learning rate so that the calculation for further learning rates stays consistent.
    if epoch in control.epochData:
      control.epochData[epoch].learningRate = learningRate
    else:
      control.epochData[epoch] = control.EpochData(learningRate=learningRate)
  print("Finished, last stored epoch was %i." % maxEpoch)


if __name__ == "__main__":
  demo()
