
import os
from Util import betterRepr, simpleObjRepr, ObjAsDict


class LearningRateControl(object):

  class EpochData:
    def __init__(self, learningRate, error=None):
      """
      :type learningRate: float
      :type error: float | None
      """
      self.learningRate = learningRate
      self.error = error

    __repr__ = simpleObjRepr

  @classmethod
  def load_initial_kwargs_from_config(cls, config):
    """
    :type config: Config.Config
    :rtype: dict[str]
    """
    return {
      "initialLearningRate": config.float('learning_rate', 0.01),
      "filename": config.value('learning_rate_file', None)}

  @classmethod
  def load_initial_from_config(cls, config):
    """
    :type config: Config.Config
    :rtype: LearningRateControl
    """
    kwargs = cls.load_initial_kwargs_from_config(config)
    return cls(**kwargs)

  def __init__(self, initialLearningRate, filename=None):
    """
    :param float initialLearningRate: learning rate for epoch 1
    :param str filename: load from and save to file
    """
    self.epochData = {1: self.EpochData(initialLearningRate)}
    self.filename = filename
    if filename and os.path.exists(filename):
      self.load()

  @property
  def initialLearningRate(self):
    return self.epochData[1].learningRate

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
    :type error: float
    """
    assert epoch in self.epochData, "You did not called getLearningRateForEpoch(%i)?" % epoch
    assert isinstance(error, float)
    self.epochData[epoch].error = error

  def save(self):
    if not self.filename: return
    f = open(self.filename, "w")
    f.write(betterRepr(self.epochData))
    f.write("\n")
    f.close()

  def load(self):
    s = open(self.filename).read()
    self.epochData = eval(s, {}, ObjAsDict(self))


class ConstantLearningRate(LearningRateControl):

  def calcLearningRateForEpoch(self, epoch):
    """
    Dummy constant learning rate. Returns initial learning rate.
    :type epoch: int
    :returns learning rate
    :rtype: float
    """
    return self.initialLearningRate


class Newbob(LearningRateControl):

  @classmethod
  def load_initial_kwargs_from_config(cls, config):
    """
    :type config: Config.Config
    :rtype: dict[str]
    """
    kwargs = super(Newbob, cls).load_initial_kwargs_from_config(config)
    kwargs.update({
      "relativeErrorThreshold": config.float('newbob_relative_error_threshold', -0.01),
      "learningRateDecayFactor": config.float('newbob_learning_rate_decay', 0.5)})
    return kwargs

  def __init__(self, initialLearningRate, relativeErrorThreshold, learningRateDecayFactor, filename=None):
    """
    :param float initialLearningRate: learning rate for epoch 1+2
    :type relativeErrorThreshold: float
    :type learningRateDecayFactor: float
    :type filename: str
    """
    super(Newbob, self).__init__(initialLearningRate=initialLearningRate, filename=filename)
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
    last2Epoch = self.getLastEpoch(lastEpoch)
    if last2Epoch is None:
      return learningRate
    oldError = self.epochData[last2Epoch].error
    newError = self.epochData[lastEpoch].error
    relativeError = (newError - oldError) / abs(newError)
    if relativeError > self.relativeErrorThreshold:
      learningRate *= self.learningRateDecayFactor
    return learningRate


def learningRateControlType(typeName):
  if typeName == "constant":
    return ConstantLearningRate
  elif typeName == "newbob":
    return Newbob
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

