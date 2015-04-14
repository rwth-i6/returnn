
import numpy
from theano import tensor as T
import theano
from BestPathDecoder import BestPathDecodeOp
from CTC import CTCOp
from NetworkLayer import Layer
from SprintErrorSignals import SprintErrorSigOp


class OutputLayer(Layer):
  def __init__(self, sources, index, n_out, L1=0.0, L2=0.0, loss='ce', dropout=0.0, mask="unity", layer_class="softmax", name=""):
    """
    :param list[NetworkLayer.SourceLayer] sources: list of source layers
    :param theano.Variable index: index for batches
    :param int n_out: output dim
    :param float L1: l1-param-norm regularization
    :param float L2: l2-param-norm regularization
    :param str loss: e.g. 'ce'
    :type dropout: float
    :param str mask: "unity" or "dropout"
    :param str layer_class: name of layer type, e.g. "hidden"
    :param str name: custom layer name, e.g. "hidden_2"
    """
    super(OutputLayer, self).__init__(sources, n_out, L1, L2, layer_class, mask, dropout, name = name)
    self.z = self.b
    self.W_in = [self.add_param(self.create_forward_weights(source.attrs['n_out'], n_out,
                                                            name="W_in_%s_%s" % (source.name, self.name)),
                                "W_in_%s_%s" % (source.name, self.name))
                 for source in sources]
    assert len(sources) == len(self.masks) == len(self.W_in)
    for source, m, W in zip(sources, self.masks, self.W_in):
      if mask == "unity":
        self.z += T.dot(source.output, W)
      else:
        self.z += T.dot(self.mass * m * source.output, W)
    self.set_attr('from', ",".join([s.name for s in sources]))
    self.index = index
    self.i = (index.flatten() > 0).nonzero()
    self.loss = loss.encode("utf8")
    self.attrs['loss'] = self.loss
    if self.loss == 'priori': self.priori = theano.shared(value = numpy.ones((n_out,), dtype=theano.config.floatX), borrow=True)

  def create_bias(self, n, prefix='b'):
    name = "%s_%s" % (prefix, self.name)
    assert n > 0
    bias = numpy.log(1.0 / n)  # More numerical stable.
    value = numpy.zeros((n,), dtype=theano.config.floatX) + bias
    return theano.shared(value=value, borrow=True, name=name)

  def entropy(self):
    """
    :rtype: theano.Variable
    """
    return -T.sum(self.p_y_given_x[self.i] * T.log(self.p_y_given_x[self.i]))

  def errors(self, y):
    """
    :type y: theano.Variable
    :rtype: theano.Variable
    """
    if y.dtype.startswith('int'):
      return T.sum(T.neq(self.y_pred[self.i], y[self.i]))
    else: raise NotImplementedError()


class FramewiseOutputLayer(OutputLayer):
  def __init__(self, sources, index, n_out, L1=0.0, L2=0.0, loss='ce', dropout=0.0, mask="unity", layer_class="softmax", name=""):
    super(FramewiseOutputLayer, self).__init__(sources, index, n_out, L1, L2, loss, dropout, mask, layer_class, name=name)
    self.initialize()

  def initialize(self):
    #self.y_m = T.reshape(self.z, (self.z.shape[0] * self.z.shape[1], self.z.shape[2]), ndim = 2)
    self.y_m = self.z.dimshuffle(2,0,1).flatten(ndim = 2).dimshuffle(1,0)
    #T.reshape(self.z, (self.z.shape[0] * self.z.shape[1], self.z.shape[2]), ndim = 2)
    if self.loss == 'ce': self.p_y_given_x = T.nnet.softmax(self.y_m) # - self.y_m.max(axis = 1, keepdims = True))
    #if self.loss == 'ce':
    #  y_mmax = self.y_m.max(axis = 1, keepdims = True)
    #  y_mmin = self.y_m.min(axis = 1, keepdims = True)
    #  self.p_y_given_x = T.nnet.softmax(self.y_m - (0.5 * (y_mmax - y_mmin) + y_mmin))
    elif self.loss == 'sse': self.p_y_given_x = self.y_m
    elif self.loss == 'priori': self.p_y_given_x = T.nnet.softmax(self.y_m) / self.priori
    else: assert False, "invalid loss: " + self.loss
    self.y_pred = T.argmax(self.p_y_given_x, axis=-1)

  def cost(self, y):
    known_grads = None
    if self.loss == 'ce' or self.loss == 'priori':
      pcx = self.p_y_given_x[self.i, y[self.i]]
      pcx = T.clip(pcx, 1.e-20, 1.e20)  # For pcx near zero, the gradient will likely explode.
      return -T.sum(T.log(pcx)), known_grads
    elif self.loss == 'sse':
      y_f = T.cast(T.reshape(y, (y.shape[0] * y.shape[1]), ndim=1), 'int32')
      y_oh = T.eq(T.shape_padleft(T.arange(self.attrs['n_out']), y_f.ndim), T.shape_padright(y_f, 1))
      return T.mean(T.sqr(self.p_y_given_x[self.i] - y_oh[self.i])), known_grads
    else:
      assert False, "unknown loss: %s" % self.loss


class SequenceOutputLayer(OutputLayer):
  def __init__(self, sources, index, n_out, L1=0.0, L2=0.0, loss='ce', dropout=0.0, mask="unity", layer_class="softmax", name="", prior_scale=0.0, log_prior=None, ce_smoothing=0.0):
    super(SequenceOutputLayer, self).__init__(sources, index, n_out, L1, L2, loss, dropout, mask, layer_class, name = name)
    self.prior_scale = prior_scale
    self.log_prior = log_prior
    self.ce_smoothing = ce_smoothing
    self.initialize()

  def initialize(self):
    assert self.loss in ('ctc', 'ce_ctc', 'sprint', 'sprint_smoothed'), 'invalid loss: ' + self.loss
    self.y_m = T.reshape(self.z, (self.z.shape[0] * self.z.shape[1], self.z.shape[2]), ndim = 2)
    p_y_given_x = T.nnet.softmax(self.y_m)
    self.y_pred = T.argmax(p_y_given_x, axis = -1)
    self.p_y_given_x = T.reshape(T.nnet.softmax(self.y_m), self.z.shape)

  def cost(self, y):
    y_f = T.cast(T.reshape(y, (y.shape[0] * y.shape[1]), ndim = 1), 'int32')
    known_grads = None
    if self.loss == 'sprint':
      err, grad = SprintErrorSigOp()(self.p_y_given_x, T.sum(self.index, axis=0))
      known_grads = {self.z: grad}
      return err.sum(), known_grads
    elif self.loss == 'sprint_smoothed':
      assert self.log_prior is not None
      err, grad = SprintErrorSigOp()(self.p_y_given_x, T.sum(self.index, axis=0))
      err *= (1.0 - self.ce_smoothing)
      err = err.sum()
      grad *= (1.0 - self.ce_smoothing)
      y_m_prior = T.reshape(self.z + self.prior_scale * self.log_prior, (self.z.shape[0] * self.z.shape[1], self.z.shape[2]), ndim=2)
      p_y_given_x_prior = T.nnet.softmax(y_m_prior)
      pcx = p_y_given_x_prior[(self.i > 0).nonzero(), y_f[(self.i > 0).nonzero()]]
      ce = self.ce_smoothing * (-1.0) * T.sum(T.log(pcx))
      err += ce
      known_grads = {self.z: grad + T.grad(ce, self.z)}
      return err, known_grads
    elif self.loss == 'ctc':
      err, grad, priors = CTCOp()(self.p_y_given_x, y, T.sum(self.index, axis=0))
      known_grads = {self.z: grad}
      return err.sum(), known_grads, priors.sum(axis=0)
    elif self.loss == 'ce_ctc':
      y_m = T.reshape(self.z, (self.z.shape[0] * self.z.shape[1], self.z.shape[2]), ndim=2)
      p_y_given_x = T.nnet.softmax(y_m)
      #pcx = p_y_given_x[(self.i > 0).nonzero(), y_f[(self.i > 0).nonzero()]]
      pcx = p_y_given_x[self.i, y[self.i]]
      ce = -T.sum(T.log(pcx))
      return ce, known_grads

  def errors(self, y):
    if self.loss in ('ctc', 'ce_ctc'):
      return T.sum(BestPathDecodeOp()(self.p_y_given_x, y, T.sum(self.index, axis=0)))
    else:
      return super(SequenceOutputLayer, self).errors(y)
