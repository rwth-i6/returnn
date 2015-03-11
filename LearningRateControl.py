

class LearningRateControl(object):

  class EpochData:
    def __init__(self, learningRate, error=None):
      """
      :type learningRate: float
      :type error: float | None
      """
      self.learningRate = learningRate
      self.error = error

  @classmethod
  def load_initial_kwargs_from_config(cls, config):
    """
    :type config: Config.Config
    :rtype: dict[str]
    """
    return {"initialLearningRate": config.float('learning_rate', 0.01)}

  @classmethod
  def load_initial_from_config(cls, config):
    """
    :type config: Config.Config
    :rtype: LearningRateControl
    """
    kwargs = cls.load_initial_kwargs_from_config(config)
    return cls(**kwargs)

  def __init__(self, initialLearningRate):
    """
    :param float initialLearningRate: learning rate for epoch 1
    """
    self.epochData = {1: self.EpochData(initialLearningRate)}

  @property
  def initialLearningRate(self):
    return self.epochData[1].learningRate

  def __repr__(self):
    import inspect
    return self.__class__.__name__ + "(%s)" % \
      ", ".join(["%s=%r" % (arg, getattr(self, arg)) for arg in inspect.getargspec(self.__init__).args[1:]])

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
    self.epochData[epoch] = self.EpochData(learningRate)
    return learningRate

  def setEpochError(self, epoch, error):
    """
    :type epoch: int
    :type error: float
    """
    assert epoch in self.epochData, "You did not called getLearningRateForEpoch(%i)?" % epoch
    assert self.epochData[epoch].error is None, "Error expected to not be set yet."
    assert isinstance(error, float)
    self.epochData[epoch].error = error

  def save(self, filename):
    import pickle
    pickle.dump(self, open(filename, "w"))

  @staticmethod
  def load(filename):
    import pickle
    return pickle.load(open(filename))


class ConstantLearningRate(LearningRateControl):

  def calcLearningRateForEpoch(self, epoch):
    """
    Dummy constant learning rate. Returns initial learning rate.
    :type epoch: int
    :returns learning rate
    :rtype: float
    """
    return self.epochData[1].learningRate


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

  def __init__(self, initialLearningRate, relativeErrorThreshold, learningRateDecayFactor):
    """
    :param float initialLearningRate: learning rate for epoch 1+2
    :type relativeErrorThreshold: float
    :type learningRateDecayFactor: float
    """
    super(Newbob, self).__init__(initialLearningRate=initialLearningRate)
    self.relativeErrorThreshold = relativeErrorThreshold
    self.learningRateDecayFactor = learningRateDecayFactor

  def calcLearningRateForEpoch(self, epoch):
    """
    Newbob+ on train data.
    :type epoch: int
    :returns learning rate
    :rtype: float
    """
    learningRate = self.epochData[epoch - 1].learningRate
    if epoch == 2:
      return learningRate
    oldError = self.epochData[epoch - 2].error
    newError = self.epochData[epoch - 1].error
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

