
import theano
import numpy
import os
from Log import log
from math import sqrt
import theano.tensor as T

class Updater:

  @classmethod
  def initFromConfig(cls, config):
    import rnn
    kwargs = {
      "gradient_clip": config.float('gradient_clip', -1),
      "adagrad": config.bool('adagrad', False),
      "adadelta": config.bool('adadelta', False),
      "varreg" : config.bool('varreg', False),
      "adadelta_decay": config.float('adadelta_decay', 0.90),
      "adadelta_offset": config.float('adadelta_offset', 1e-6),
      "momentum": config.float("momentum", 0)}
    return cls(**kwargs)

  def __init__(self, momentum, gradient_clip, adagrad, adadelta, adadelta_decay, adadelta_offset, varreg):
    """
    :type momentum: float
    :type gradient_clip: float
    :type adagrad: bool
    :type adadelta: bool
    """
    self.rng = numpy.random.RandomState(0101)
    self.momentum = momentum
    self.gradient_clip = gradient_clip
    self.varreg = varreg
    self.adagrad = adagrad
    self.adadelta = adadelta
    self.adadelta_decay = adadelta_decay
    self.adadelta_offset = adadelta_offset
    self.pid = -1
    if self.adadelta or self.adagrad:
      self.varreg = False
    assert not (self.adagrad and self.adadelta)
    if self.adadelta:
      self.momentum = 0.0
      print >> log.v4, "using adadelta with decay", self.adadelta_decay, ", offset", self.adadelta_offset
    if self.adagrad:
      print >> log.v4, "using adagrad"
    if self.momentum:
      print >> log.v4, "using momentum %f" % self.momentum
    if self.gradient_clip > 0:
      print >> log.v4, "using gradient clipping %f" % self.gradient_clip

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
    if net_param_deltas is not None:
      self.net_train_param_deltas = net_param_deltas
    else:
      self.net_train_param_deltas = {k: {p : theano.shared(numpy.zeros(p.get_value(borrow=True,
                                                                              return_internal_type=True).shape,
                                                                  dtype=theano.config.floatX))
                                     for p in network.train_params_vars} for k in network.cost}
      " :type: dict[theano.compile.sharedvalue.SharedVariable,theano.compile.sharedvalue.SharedVariable] "
    self.learning_rate_var = theano.shared(value=numpy.cast[theano.config.floatX](0))
    " :type: theano.compile.sharedvalue.SharedVariable "

    if self.momentum > 0:
      self.deltas = { k : {p: theano.shared(
                     value=numpy.zeros(p.get_value(borrow=True, return_internal_type=True).shape,
                                       dtype=theano.config.floatX), borrow=True,
                     name="deltas_%s" % p)
                     for p in network.train_params_vars} for k in self.network.cost}
    if self.adagrad:
      self.accu = {}
      for p in self.net_train_param_deltas[self.net_train_param_deltas.keys()[0]]:
        shape = p.get_value(borrow=True, return_internal_type=True).shape
        scale = numpy.sqrt(12. / numpy.sum(shape))
        values = numpy.asarray(self.rng.normal(loc=0.0, scale=scale, size=shape), dtype=theano.config.floatX)
        values = p.get_value()
        values = numpy.zeros(shape, dtype=theano.config.floatX)
        self.accu[p] = theano.shared(value=values)

      #self.accu = {p: theano.shared(
      #               value=numpy.zeros(p.get_value(borrow=True, return_internal_type=True).shape,
      #                                 dtype=theano.config.floatX), borrow=True, name="accu_%s " % p)
      #            for p in self.network.train_params_vars}
      #self.sqrsum = {p: theano.shared(
      #               value=numpy.zeros(p.get_value(borrow=True, return_internal_type=True).shape,
      #                                 dtype=theano.config.floatX), borrow=True,
      #               name="sqrsum_%s " % p)
      #               for p in self.network.train_params_vars}
    if self.adadelta or self.varreg:
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
    for k in net_param_deltas:
      for p in net_param_deltas[k]:
        self.net_train_param_deltas[k][p].set_value(net_param_deltas[k][p], borrow=True)

  def norm_constraint(self, tensor_var, max_norm, norm_axes=None, epsilon=1e-7):
    ndim = tensor_var.ndim

    if norm_axes is not None:
        sum_over = tuple(norm_axes)
    elif ndim == 2:  # DenseLayer
        sum_over = (0,)
    elif ndim in [3, 4, 5]:  # Conv{1,2,3}DLayer
        sum_over = tuple(range(1, ndim))
    else:
        sum_over = (0,)

    dtype = numpy.dtype(theano.config.floatX).type
    norms = T.sqrt(T.sum(T.sqr(tensor_var), axis=sum_over, keepdims=True))
    target_norms = T.clip(norms, 0, dtype(max_norm))
    constrained_output = \
        (tensor_var * (target_norms / (dtype(epsilon) + norms)))

    return constrained_output

  def getUpdateList(self):
    assert self.pid == os.getpid()
    updates = []
    " :type: list[(theano.SharedVariable, theano.Variable)] "
    for param in self.network.train_params_vars:
      upd = 0
      for target in self.net_train_param_deltas:
        if not param in self.net_train_param_deltas[target]:
          continue
        deltas = self.net_train_param_deltas[target][param]  # usually the gradients
        if self.gradient_clip > 0:
          # Note that there is also theano.gradient.grad_clip, which would clip it already
          # at the backprop step and which would affect also other dependent gradients.
          # However, this is simpler for now.
          # Also note that this is yet without the learning rate factor -
          # this might be different to other gradient clipping implementations.
          deltas = T.clip(deltas, -self.gradient_clip, self.gradient_clip)
        if self.momentum > 0:
          upd += self.momentum * self.deltas[target][param]
        if self.adagrad:
          epsilon = 1e-6
          accu_new = self.accu[param] + deltas ** 2
          updates.append((self.accu[param], accu_new))
          upd += -self.learning_rate_var * deltas / T.sqrt(accu_new + epsilon)
          #updates.append((self.sqrsum[param], self.sqrsum[param] + deltas ** 2))
          #upd = upd * 0.1 / (0.1 + (self.sqrsum[param] + deltas ** 2) ** 0.5)
        elif self.varreg:
          offset = 6.0 / param.layer.get_branching()
          decay = 1.0 - 1.0 / sqrt(param.layer.get_branching())
          g = deltas
          g2 = g ** 2
          #z_eg2 = T.sum(self.eg2[param])
          #z_g2 = T.sum(g2)
          #z = z_eg2 + z_g2
          #eg2_new = self.eg2[param] * z_eg2 / z + g2 * z_g2 / z
          eg2_new = decay * self.eg2[param] + (1 - decay) * g2
          dx_new = - g * T.sqrt(self.edx2[param] + offset) / T.sqrt(eg2_new + offset)
          edx2_new = decay * self.edx2[param] + (1 - decay) * dx_new ** 2
          #r_edx2 = T.sum(self.edx2[param], axis = 0)
          #r_dx_new = T.sum(dx_new**2, axis = 0)
          #r = r_edx2 + r_dx_new
          #edx2_new = self.edx2[param] * z_eg2 / z + (dx_new**2) * z_g2 / z
          #edx2_new = self.edx2[param] * r_edx2 / r + (dx_new**2) * r_dx_new / r
          updates.append((self.eg2[param], eg2_new))
          updates.append((self.edx2[param], edx2_new))
          updates.append((self.dx[param], dx_new))
          upd += self.learning_rate_var * dx_new
        elif self.adadelta:
          # http://arxiv.org/pdf/1212.5701v1.pdf
          decay = self.adadelta_decay
          offset = self.adadelta_offset
          g = deltas
          g2 = g ** 2
          eg2_new = decay * self.eg2[param] + (1 - decay) * g2
          dx_new = - g * T.sqrt(self.edx2[param] + offset) / T.sqrt(eg2_new + offset)
          edx2_new = decay * self.edx2[param] + (1 - decay) * dx_new ** 2
          updates.append((self.eg2[param], eg2_new))
          updates.append((self.edx2[param], edx2_new))
          updates.append((self.dx[param], dx_new))
          upd += self.learning_rate_var * dx_new
        else:
          upd += - self.learning_rate_var * deltas
      if self.momentum > 0:
        updates.append((self.deltas[target][param], upd))
      if upd:
        #updates.append((param, self.norm_constraint(param + upd, 1.0)))
        updates.append((param, param + upd))

    return updates

  def setLearningRate(self, learning_rate):
    """
    :type learning_rate: float
    """
    assert self.pid == os.getpid()
    self.learning_rate_var.set_value(learning_rate)

  def update(self):
    assert self.pid == os.getpid()
    updates = self.getUpdateList()
    updater = theano.function(inputs=[], updates=updates, name="updater")
    return updater()
