
import os
from Util import betterRepr, simpleObjRepr, ObjAsDict
from Log import log


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
      "initialLearningRate": config.float('learning_rate', 1.0),
      "initialLearningRates": config.typed_value('learning_rates') or config.float_list('learning_rates'),
      "errorMeasureKey": config.value('learning_rate_control_error_measure', None),
      "filename": config.value('learning_rate_file', None)}

  @classmethod
  def load_initial_from_config(cls, config):
    """
    :type config: Config.Config
    :rtype: LearningRateControl
    """
    kwargs = cls.load_initial_kwargs_from_config(config)
    return cls(**kwargs)

  def __init__(self, initialLearningRate, initialLearningRates=None, errorMeasureKey=None, filename=None):
    """
    :param float initialLearningRate: learning rate for epoch 1
    :param list[float] | dict[int,float] initialLearningRates: learning rates
    :param str errorMeasureKey: for getEpochErrorValue() the selector for EpochData.error which is a dict
    :param str filename: load from and save to file
    """
    self.epochData = {1: self.EpochData(initialLearningRate)}
    if initialLearningRates:
      if isinstance(initialLearningRates, list):
        initialLearningRates = {i + 1: v for (i, v) in enumerate(initialLearningRates)}
      assert isinstance(initialLearningRates, dict)
      for epoch, v in initialLearningRates.items():
        self.setLearningRateForEpoch(epoch, v)
    self.initialLearningRates = initialLearningRates
    self.initialLearningRate = initialLearningRate
    self.errorMeasureKey = errorMeasureKey
    self.filename = filename
    if filename:
      if os.path.exists(filename):
        print >>log.v4, "Learning-rate-control: loading file %s" % filename
        self.load()
      else:
        print >>log.v4, "Learning-rate-control: file %s does not exist yet" % filename
    else:
      print >>log.v4, "Learning-rate-control: no file specified, not saving history (no proper restart possible)"

  __repr__ = simpleObjRepr

  def __str__(self):
    return "%r, epoch data: %s" % \
           (self, ", ".join(["%i: %s" % (epoch, self.epochData[epoch])
                             for epoch in sorted(self.epochData.keys())]))

  def calcLearningRateForEpoch(self, epoch):
    """
    :type epoch: int
    :returns learning rate
    :rtype: float
    """
    raise NotImplementedError

  def getLearningRateForEpoch(self, epoch):
    """
    :type epoch: int
    :rtype: float
    """
    assert epoch >= 1
    if epoch in self.epochData: return self.epochData[epoch].learningRate
    learningRate = self.calcLearningRateForEpoch(epoch)
    self.setLearningRateForEpoch(epoch, learningRate)
    return learningRate

  def setLearningRateForEpoch(self, epoch, learningRate):
    """
    :type epoch: int
    :type learningRate: float
    """
    if epoch in self.epochData:
      self.epochData[epoch].learningRate = learningRate
    else:
      self.epochData[epoch] = self.EpochData(learningRate)

  def getLastEpoch(self, epoch):
    epochs = sorted([e for e in self.epochData.keys() if e < epoch])
    if not epochs:
      return None
    return epochs[-1]

  def setEpochError(self, epoch, error):
    """
    :type epoch: int
    :type error: dict[str,float|dict[str,float]]
    """
    if epoch not in self.epochData:
      print >> log.v4, "Learning rate not set for epoch %i. Assuming default." % epoch
      self.getLearningRateForEpoch(epoch)  # This will set it.
    assert isinstance(error, dict)
    error = error.copy()
    for k, v in list(error.items()):
      if isinstance(v, dict):  # like error = {"dev_score": {"cost:output1": .., "cost:output2": ...}, ...}
        del error[k]
        if len(v) == 1:
          error[k] = v.values()[0]
          continue
        for k1, v1 in v.items():
          if ":" in k1: k1 = k1[k1.index(":") + 1:]
          error[k + "_" + k1] = v1
    for v in error.values():
      assert isinstance(v, float)
    self.epochData[epoch].error.update(error)

  def getErrorKey(self, epoch):
    if epoch not in self.epochData:
      return self.errorMeasureKey
    epoch_data = self.epochData[epoch]
    if not epoch_data.error:
      return None
    if len(epoch_data.error) == 1 and "old_format_score" in epoch_data.error:
      return "old_format_score"
    if self.errorMeasureKey:
      if self.errorMeasureKey not in epoch_data.error:
        if self.errorMeasureKey + "_output" in epoch_data.error:  # for multiple outputs, try default output
          return self.errorMeasureKey + "_output"
      return self.errorMeasureKey
    for key in ["dev_score", "train_score"]:  # To keep old setups producing the same behavior, keep this order.
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
    self.epochData = eval(s, {}, ObjAsDict(self))


class ConstantLearningRate(LearningRateControl):

  need_error_info = False

  def calcLearningRateForEpoch(self, epoch):
    """
    Dummy constant learning rate. Returns initial learning rate.
    :type epoch: int
    :returns learning rate
    :rtype: float
    """
    return self.initialLearningRate


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
    :param float initialLearningRate: learning rate for epoch 1+2
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
      return self.initialLearningRate
    learningRate = self.epochData[lastEpoch].learningRate
    if learningRate is None:
      return self.initialLearningRate
    last2Epoch = self.getLastEpoch(lastEpoch)
    if last2Epoch is None:
      return learningRate
    oldError = self.getEpochErrorValue(last2Epoch)
    newError = self.getEpochErrorValue(lastEpoch)
    if oldError is None or newError is None:
      return learningRate
    relativeError = (newError - oldError) / abs(newError)
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
      return self.initialLearningRate
    learningRate = self.epochData[lastEpoch].learningRate
    if learningRate is None:
      return self.initialLearningRate
    last2Epoch = self.getLastEpoch(lastEpoch)
    if last2Epoch is None:
      return learningRate
    oldError = self.getEpochErrorValue(last2Epoch)
    newError = self.getEpochErrorValue(lastEpoch)
    if oldError is None or newError is None:
      return learningRate
    errorDiff = newError - oldError
    if errorDiff > self.errorThreshold:
      learningRate *= self.learningRateDecayFactor
    return learningRate


def learningRateControlType(typeName):
  if typeName == "constant":
    return ConstantLearningRate
  elif typeName in ("newbob", "newbob_rel", "newbob_relative"):  # Old setups expect the relative version.
    return NewbobRelative
  elif typeName == "newbob_abs":
    return NewbobAbs
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

