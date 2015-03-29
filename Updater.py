
import theano
import numpy
import os
from Log import log
import theano.tensor as T

class Updater:

  @classmethod
  def initFromConfig(cls, config):
    import rnn
    kwargs = {
      "updateOnDevice": rnn.isUpdateOnDevice(config),
      "gradient_clip": config.float('gradient_clip', -1),
      "adagrad": config.bool('adagrad', False),
      "adadelta": config.bool('adadelta', False),
      "adadelta_decay": config.float('adadelta_decay', 0.90),
      "adadelta_offset": config.float('adadelta_offset', 1e-6),
      "momentum": config.float("momentum", 0)}
    return cls(**kwargs)

  def __init__(self, momentum, gradient_clip, adagrad, adadelta, adadelta_decay, adadelta_offset, updateOnDevice):
    """
    :type momentum: float
    :type gradient_clip: float
    :type adagrad: bool
    :type adadelta: bool
    :type updateOnDevice: bool
    """
    self.momentum = momentum
    self.gradient_clip = gradient_clip
    self.adagrad = adagrad
    self.adadelta = adadelta
    self.adadelta_decay = adadelta_decay
    self.adadelta_offset = adadelta_offset
    self.updateOnDevice = updateOnDevice
    self.pid = -1
    assert not (self.adagrad and self.adadelta)
    if self.adadelta:
      self.momentum = 0.0
      print >> log.v3, "using adadelta with decay", self.adadelta_decay, ", offset", self.adadelta_offset
    if self.adagrad:
      print >> log.v3, "using adagrad"
    if self.momentum:
      print >> log.v3, "using momentum %f" % self.momentum
    if self.gradient_clip > 0:
      print >> log.v3, "using gradient clipping %f" % self.gradient_clip

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
      self.net_train_param_deltas = net_param_deltas
    else:
      assert net_param_deltas is None
      self.net_train_param_deltas = {p: theano.shared(numpy.zeros(p.get_value(borrow=True,
                                                                              return_internal_type=True).shape,
                                                                  dtype=theano.config.floatX))
                                     for p in network.train_params_vars}
      " :type: dict[theano.compile.sharedvalue.SharedVariable,theano.compile.sharedvalue.SharedVariable] "
    self.learning_rate_var = theano.shared(value=numpy.cast[theano.config.floatX](0))
    " :type: theano.compile.sharedvalue.SharedVariable "

    if self.momentum > 0:
      self.deltas = {p: theano.shared(
                     value=numpy.zeros(p.get_value(borrow=True, return_internal_type=True).shape,
                                       dtype=theano.config.floatX), borrow=True,
                     name="deltas_%s" % p)
                     for p in self.network.train_params_vars}
    if self.adagrad:
      self.sqrsum = {p: theano.shared(
                     value=numpy.zeros(p.get_value(borrow=True, return_internal_type=True).shape,
                                       dtype=theano.config.floatX), borrow=True,
                     name="sqrsum_%s " % p)
                     for p in self.network.train_params_vars}
    if self.adadelta:
      # http://arxiv.org/pdf/1212.5701v1.pdf
      self.eg2 = {p: theano.shared(value=numpy.zeros(p.get_value(borrow=True, return_internal_type=True).shape,
                                                     dtype=theano.config.floatX))
                  for p in self.network.train_params_vars} #E[g^2]
      self.edx2 = {p: theano.shared(value=numpy.zeros(p.get_value(borrow=True, return_internal_type=True).shape,
                                                      dtype=theano.config.floatX))
                  for p in self.network.train_params_vars} #E[\delta x^2]
      self.dx = {p: theano.shared(value=numpy.zeros(p.get_value(borrow=True, return_internal_type=True).shape,
                                                    dtype=theano.config.floatX))
                  for p in self.network.train_params_vars} #\delta x

  @property
  def isInitialized(self):
    return self.pid >= 0

  def setNetParamDeltas(self, net_param_deltas):
    assert self.pid == os.getpid()
    assert not self.updateOnDevice
    for p in self.network.train_params_vars:
      self.net_train_param_deltas[p].set_value(net_param_deltas[p], borrow=True)

  def getUpdateList(self):
    assert self.pid == os.getpid()
    updates = []
    " :type: list[(theano.SharedVariable, theano.Variable)] "
    for param in self.network.train_params_vars:
      deltas = self.net_train_param_deltas[param]  # usually the gradients
      if self.gradient_clip > 0:
        # Note that there is also theano.gradient.grad_clip, which would clip it already
        # at the backprop step and which would affect also other dependent gradients.
        # However, this is simpler for now.
        # Also note that this is yet without the learning rate factor -
        # this might be different to other gradient clipping implementations.
        deltas = T.clip(deltas, -self.gradient_clip, self.gradient_clip)
      upd = - self.learning_rate_var * deltas
      if self.momentum > 0:
        upd += self.momentum * self.deltas[param]
        updates.append((self.deltas[param], upd))
      if self.adagrad:
        updates.append((self.sqrsum[param], self.sqrsum[param] + deltas ** 2))
        upd = upd * 0.1 / (0.1 + (self.sqrsum[param] + deltas ** 2) ** 0.5)
      if self.adadelta:
        # http://arxiv.org/pdf/1212.5701v1.pdf
        decay = self.adadelta_decay
        offset = self.adadelta_offset
        g = deltas
        g2 = g ** 2
        eg2_new = decay * self.eg2[param] + (1 - decay) * g2
        dx_new = - T.sqrt(self.edx2[param] + offset) / T.sqrt(eg2_new + offset) * g
        edx2_new = decay * self.edx2[param] + (1 - decay) * dx_new ** 2
        updates.append((self.eg2[param], eg2_new))
        updates.append((self.edx2[param], edx2_new))
        updates.append((self.dx[param], dx_new))
        upd = dx_new
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
