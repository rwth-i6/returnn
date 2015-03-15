
import theano
import numpy
import os


class Updater:

  @classmethod
  def initFromConfig(cls, config):
    import rnn
    kwargs = {
      "updateOnDevice": rnn.isUpdateOnDevice(config),
      "adagrad": config.bool('adagrad', False),
      "adadelta": config.bool('adadelta', False),
      "momentum": config.float("momentum", 0)}
    return cls(**kwargs)

  def __init__(self, momentum, adagrad, adadelta, updateOnDevice):
    """
    :type momentum: float
    :type adagrad: bool
    :type adadelta: bool
    :type updateOnDevice: bool
    """
    self.momentum = momentum
    self.adagrad = adagrad
    self.adadelta = adadelta #TODO use string for training method instead of flags
    self.updateOnDevice = updateOnDevice
    self.pid = -1

  def initVars(self, network, net_param_deltas):
    """
    Initializes the Theano shared variables.
    This should be called in the process where you want to do the updating.
    All further calls must be from the same process.
    The network.gparams must be created in the same process.
    :type network: Network.LayerNetwork
    :type net_param_deltas: dict[theano.compile.sharedvalue.SharedVariable,theano.Variable] | None
    """
    assert not self.isInitialized
    self.pid = os.getpid()
    self.network = network
    if self.updateOnDevice:
      assert net_param_deltas is not None
      self.net_param_deltas = net_param_deltas
    else:
      assert net_param_deltas is None
      self.net_param_deltas = {p: theano.shared(value=numpy.zeros(p.get_value().shape, dtype=theano.config.floatX))
                               for p in network.gparams}
      " :type: dict[theano.compile.sharedvalue.SharedVariable,theano.compile.sharedvalue.SharedVariable] "
    self.learning_rate_var = theano.shared(value=numpy.cast[theano.config.floatX](0))
    " :type: theano.compile.sharedvalue.SharedVariable "

    if self.momentum > 0:
      self.deltas = {p: theano.shared(
                     value=numpy.zeros(p.get_value().shape, dtype=theano.config.floatX), borrow=True,
                     name="deltas_%s" % p)
                     for p in self.network.gparams}
    if self.adagrad:
      self.sqrsum = {p: theano.shared(
                     value=numpy.zeros(p.get_value().shape, dtype=theano.config.floatX), borrow=True,
                     name="sqrsum_%s " % p)
                     for p in self.network.gparams}
    if self.adadelta:
      self.eg2 = {p: theano.shared(value=numpy.zeros(p.get_value().shape, dtype=theano.config.floatX))
                  for p in self.network.gparams} #E[g^2]
      self.edx2 = {p: theano.shared(value=numpy.zeros(p.get_value().shape, dtype=theano.config.floatX))
                  for p in self.network.gparams} #E[\delta x^2]
      self.dx = {p: theano.shared(value=numpy.zeros(p.get_value().shape, dtype=theano.config.floatX))
                  for p in self.network.gparams} #\delta x

  @property
  def isInitialized(self):
    return self.pid >= 0

  def setNetParamDeltas(self, net_param_deltas):
    assert self.pid == os.getpid()
    assert not self.updateOnDevice
    for p in self.network.gparams:
      self.net_param_deltas[p].set_value(net_param_deltas[p], borrow=True)

  def getUpdateList(self):
    assert self.pid == os.getpid()
    updates = []
    " :type: list[(theano.SharedVariable, theano.Variable)] "
    for param in self.network.gparams:
      upd = - self.learning_rate_var * self.net_param_deltas[param]
      if self.momentum > 0:
        upd += self.momentum * self.deltas[param]
        updates.append((self.deltas[param], upd))
      if self.adagrad:
        updates.append((self.sqrsum[param], self.sqrsum[param] + self.net_param_deltas[param] ** 2))
        upd = upd * 0.1 / (0.1 + (self.sqrsum[param] + self.net_param_deltas[param] ** 2) ** 0.5)
      updates.append((param, param + upd))

    return updates

  def setLearningRate(self, learning_rate):
    """
    :type learning_rate: float
    """
    assert self.pid == os.getpid()
    self.learning_rate_var.set_value(learning_rate)

  def getUpdateFunction(self):
    assert self.pid == os.getpid()
    updates = self.getUpdateList()
    return theano.function(inputs=[], updates=updates, name="updater")
