#! /usr/bin/python2.7

import numpy
import theano
import theano.tensor as T
import json
import inspect
import h5py
from Util import hdf5_dimension, simpleObjRepr
from math import sqrt
from CTC import CTCOp
from Log import log
from BestPathDecoder import BestPathDecodeOp
from SprintErrorSignals import SprintErrorSigOp


"""
        Activation functions
"""

def relu(z):
  return (T.sgn(z) + 1) * z * 0.5

def identity(z):
  return z

def softsign(z):
  return z / (1.0 + abs(z))

def softsquare(z):
  return 1 / (1.0 + z * z)

def maxout(z):
  return T.max(z, axis=0)

def constant_one():
  return 1

def constant_zero():
  return 0


ActivationFunctions = {
  'logistic': T.nnet.sigmoid,
  'sigmoid': T.nnet.sigmoid,  # alias
  'tanh': T.tanh,
  'relu': relu,
  'identity': identity,
  'one': constant_one,
  'zero': constant_zero,
  'softsign': softsign,
  'softsquare': softsquare,
  'maxout': maxout,
  'sin': T.sin,
  'cos': T.cos
}

def strtoact(act):
  """
  :param str act: activation function name
  :rtype: theano.Op
  """
  assert ActivationFunctions.has_key(act), "invalid activation function: " + act
  return ActivationFunctions[act]


"""
        META LAYER
"""
class Container(object):
  def __init__(self, layer_class, name=""):
    """
    :param str layer_class: name of layer type, e.g. "hidden"
    :param str name: custom layer name, e.g. "hidden_2"
    """
    self.params = {}; """ :type: dict[str,theano.compile.sharedvalue.SharedVariable] """
    self.attrs = {}; """ :type: dict[str,str|float|int|bool] """
    self.layer_class = layer_class.encode("utf8")
    self.name = name.encode("utf8")

  def save(self, head):
    """
    :type head: h5py.File
    """
    grp = head.create_group(self.name)
    grp.attrs['class'] = self.layer_class
    for p in self.params.keys():
      value = self.params[p].get_value()
      dset = grp.create_dataset(p, value.shape, dtype='f')
      dset[...] = value
    for p in self.attrs.keys():
      try:
        grp.attrs[p] = self.attrs[p]
      except TypeError:
        print >> log.v3, "invalid type of attribute", "\"" + p + "\"", "(" + str(type(self.attrs[p])) + ")", "in layer", self.name

  def load(self, head):
    """
    :type head: h5py.File
    """
    grp = head[self.name]
    assert grp.attrs['class'] == self.layer_class, "invalid layer class (expected " + self.layer_class + " got " + grp.attrs['class'] + ")"
    for p in grp:
      assert self.params[p].get_value(borrow=True, return_internal_type=True).shape == grp[p].shape, \
        "invalid layer parameter shape for parameter " + p + " of layer " + self.name + \
        " (expected  " + str(self.params[p].get_value(borrow=True, return_internal_type=True).shape) + \
        " got " + str(grp[p].shape) + ")"
      array = grp[p][...]
      assert not (numpy.isinf(array).any() or numpy.isnan(array).any())
      self.params[p].set_value(array)
    for p in self.attrs.keys():
      self.attrs[p] = grp.attrs.get(p, None)

  def num_params(self):
    return sum([numpy.prod(v.get_value(borrow=True, return_internal_type=True).shape[0:]) for v in self.params.values()])

  def get_params_dict(self):
    """
    :rtype: dict[str,numpy.ndarray|theano.sandbox.cuda.CudaNdArray]
    """
    return {p: v.get_value(borrow=True, return_internal_type=True) for (p, v) in self.params.items()}

  def set_params_by_dict(self, params):
    """
    :type params: dict[str,numpy.ndarray|theano.sandbox.cuda.CudaNdArray]
    """
    for p, v in params.items():
      self_param_shape = self.params[p].get_value(borrow=True, return_internal_type=True).shape
      assert self_param_shape == v.shape, "In %s, param %s shape does not match. Expected %s, got %s." % \
                                          (self, p, self_param_shape, v.shape)
      self.params[p].set_value(v, borrow=True)

  def get_params_vars(self):
    """
    :returns list of shared vars in a well-defined order
    """
    return [v for (k, v) in sorted(self.params.items())]

  def add_param(self, param, name=""):
    """
    :type param: T
    :type name: str
    :rtype: T
    """
    if name == "": name = "param_%d" % len(self.params)
    self.params[name] = param
    return param

  def set_attr(self, name, value):
    self.attrs[name] = value

  def create_bias(self, n, prefix = 'b'):
    name = "%s_%s" % (prefix, self.name)
    return theano.shared(value=numpy.zeros((n,), dtype=theano.config.floatX), borrow=True, name=name)

  def create_random_weights(self, n, m, s, name = None):
    if name is None: name = self.name
    values = numpy.asarray(self.rng.normal(loc=0.0, scale=s, size=(n, m)), dtype=theano.config.floatX)
    return theano.shared(value=values, borrow=True, name=name)

  def create_uniform_weights(self, n, m, p = 0, name = None):
    if name is None: name = 'W_' + self.name
    if p == 0: p = n + m
    #values = numpy.asarray(self.rng.uniform(low = - 1 / sqrt(p), high = 1 / sqrt(p), size=(n, m)), dtype=theano.config.floatX)
    values = numpy.asarray(self.rng.uniform(low = - sqrt(6) / sqrt(p), high = sqrt(6) / sqrt(p), size=(n, m)), dtype=theano.config.floatX)
    return theano.shared(value = values, borrow = True, name = name)

  def create_forward_weights(self, n, m, name = None):
    n_in = n + m
    scale = numpy.sqrt(12. / (n_in))
    return self.create_random_weights(n, m, scale, name)

  def to_json(self):
    attrs = self.attrs.copy()
    for k in attrs.keys():
      if isinstance(attrs[k], numpy.bool_):
        attrs[k] = True if attrs[k] else False
    if 'from' in attrs:
      if attrs['from'] == 'data':
        attrs.pop('from', None)
      else:
        attrs['from'] = attrs['from'].split(',')
    return attrs

class Layer(Container):
  @classmethod
  def initialize_rng(cls):
    cls.rng = numpy.random.RandomState(1234)

  def __init__(self, sources, n_out, L1, L2, layer_class, mask="unity", dropout=0.0, name=""):
    """
    :param list[SourceLayer] sources: list of source layers
    :param int n_out: output dim
    :param float L1: l1-param-norm regularization
    :param float L2: l2-param-norm regularization
    :param str layer_class: name of layer type, e.g. "hidden"
    :param str mask: "unity" or "dropout"
    :type dropout: float
    :param str name: custom layer name, e.g. "hidden_2"
    """
    super(Layer, self).__init__(layer_class, name=name)
    self.sources = sources
    self.num_sources = len(sources)
    self.set_attr('mask', mask)
    self.set_attr('dropout', dropout)
    self.set_attr('n_out', n_out)
    self.set_attr('L1', L1)
    self.set_attr('L2', L2)
    self.b = self.add_param(self.create_bias(n_out), 'b_%s'%self.name)
    self.mass = T.constant(1., name = "mass_%s" % self.name)
    if mask == "unity":
      self.masks = [None] * len(self.sources)
    elif mask == "dropout":
      #TODO we can do some optimization if dropout == 0 for this layer

      # if we apply this mass during training then we don't need any mask or mass for testing
      # the expected weight should be 1
      # E[x] = mass * (1-dropout)
      # so mass has to be 1 / (1 - dropout)
      self.mass = T.constant(1.0 / (1.0 - dropout))
      srng = theano.tensor.shared_randomstreams.RandomStreams(self.rng.randint(1234))
      self.masks = [T.cast(srng.binomial(n=1, p=1 - dropout, size=(s.attrs['n_out'],)), theano.config.floatX) for s in self.sources]

      #this actually looked like dropconnect applied to the recurrent part, but I want to try dropout for the inputs
      #self.mask = T.cast(srng.binomial(n=1, p=1-dropout, size=(self.attrs['n_out'], self.attrs['n_out'])), theano.config.floatX)
    else:
      assert False, "invalid mask: %s" % mask

  def concat_units(self, other, axis = 1):
    assert other.layer_class == self.layer_class, "unable to concatenate %s (%s) to %s (%s)" % (other.name, other.layer_class, self.name, self.layer_class)
    for p in other.params.keys():
      if p != 'b':
        self.params[p].set_value(numpy.concatenate((self.params[p].get_value(), other.params[p].get_value()), axis = min(len(self.params[p].get_value().shape) - 1, axis)))
    if axis == 1: self.set_attr('n_out', self.attrs['n_out'] + other.arrs['n_out'])

  def to_json(self):
    attrs = super(Layer, self).to_json()
    attrs['class'] = self.layer_class
    return attrs

class SourceLayer(Container):
  def __init__(self, n_out, x_out, name = ""):
    super(SourceLayer, self).__init__('source',  name = name)
    self.output = x_out
    self.set_attr('n_out', n_out)

"""
        OUTPUT LAYERS
"""

class OutputLayer(Layer):
  def __init__(self, sources, index, n_out, L1=0.0, L2=0.0, loss='ce', dropout=0.0, mask="unity", layer_class="softmax", name=""):
    """
    :param list[SourceLayer] sources: list of source layers
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
      y_oh = T.eq(T.shape_padleft(T.arange(self.attr['n_out']), y_f.ndim), T.shape_padright(y_f, 1))
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

"""
        HIDDEN LAYERS
"""

class HiddenLayer(Layer):
  recurrent = False

  def __init__(self, sources, n_out, L1=0.0, L2=0.0, activation=T.tanh, dropout=0.0, mask="unity", connection="full", layer_class="hidden", name=""):
    """
    :param list[SourceLayer] sources: list of source layers
    :type n_out: int
    :type L1: float
    :type L2: float
    :type activation: theano.Op
    :type dropout: float
    :param str mask: mask
    :param str connection: unused
    :param str layer_class: layer class name
    :param str name: name
    """
    super(HiddenLayer, self).__init__(sources, n_out, L1, L2, layer_class, mask, dropout, name=name)
    self.activation = activation
    self.W_in = [self.add_param(self.create_forward_weights(s.attrs['n_out'],
                                                            self.attrs['n_out'],
                                                            name=self.name + "_" + s.name),
                                "W_in_%s_%s" % (s.name, self.name))
                 for s in sources]
    self.set_attr('from', ",".join([s.name for s in sources]))

class ForwardLayer(HiddenLayer):
  def __init__(self, sources, n_out, L1 = 0.0, L2 = 0.0, activation = T.tanh, dropout = 0, mask = "unity", layer_class = "hidden", name = ""):
    super(ForwardLayer, self).__init__(sources, n_out, L1, L2, activation, dropout, mask, layer_class = layer_class, name = name)
    z = self.b
    assert len(sources) == len(self.masks) == len(self.W_in)
    for s, m, W_in in zip(sources, self.masks, self.W_in):
      W_in.set_value(self.create_uniform_weights(s.attrs['n_out'], n_out, s.attrs['n_out'] + n_out, "W_in_%s_%s"%(s.name, self.name)).get_value())
      if mask == "unity":
        z += T.dot(s.output, W_in)
      else:
        z += T.dot(self.mass * m * s.output, W_in)
    self.output = z if self.activation is None else self.activation(z)

class ConvPoolLayer(ForwardLayer):
  def __init__(self, sources, n_out, L1 = 0.0, L2 = 0.0, activation = T.tanh, dropout = 0, mask = "unity", layer_class = "convpool", name = ""):
    super(ConvPoolLayer, self).__init__(sources, n_out, L1, L2, activation, dropout, mask, layer_class = layer_class, name = name)

class RecurrentLayer(HiddenLayer):
  recurrent = True

  def __init__(self, sources, index, n_out, L1 = 0.0, L2 = 0.0, activation = T.tanh, reverse = False, truncation = -1, compile = True, dropout = 0, mask = "unity", projection = None, layer_class = "recurrent", name = ""):
    super(RecurrentLayer, self).__init__(sources, n_out, L1, L2, activation, dropout, mask, layer_class = layer_class, name = name)
    self.act = self.create_bias(n_out)
    n_in = sum([s.attrs['n_out'] for s in sources])
    if projection:
      self.W_re = self.create_random_weights(projection, n_out, n_in, "W_re_%s"%self.name) #self.create_recurrent_weights(self.attrs['n_in'], n_out)
      self.W_proj = self.create_forward_weights(n_out, projection)
      self.add_param(self.W_proj, 'W_proj_%s'%self.name)
    else:
      self.W_re = self.create_random_weights(n_out, n_out, n_in, "W_re_%s"%self.name) #self.create_recurrent_weights(self.attrs['n_in'], n_out)
      self.W_proj = None
    for s, W in zip(sources, self.W_in):
      W.set_value(self.create_random_weights(s.attrs['n_out'], self.attrs['n_out'], n_in, "W_in_%s_%s"%(s.name,self.name)).get_value())
    self.add_param(self.W_re, 'W_re_%s'%self.name)
    self.index = index
    self.o = theano.shared(value = numpy.ones((n_out,), dtype='int8'), borrow=True)
    self.set_attr('reverse', reverse)
    self.set_attr('truncation', truncation)
    if projection: self.set_attr('projection', projection)
    if compile: self.compile()

  def compile(self):
    def step(x_t, i_t, h_p):
      h_pp = T.dot(h_p, self.W_re) if self.W_proj else h_p
      i = T.outer(i_t, self.o)
      z = T.dot(h_pp, self.W_re) + self.b
      for i in range(len(self.sources)):
        z += T.dot(self.mass * self.masks[i] * x_t[i], self.W_in[i])
      #z = (T.dot(x_t, self.mass * self.mask * self.W_in) + self.b) * T.nnet.sigmoid(T.dot(h_p, self.W_re))
      h_t = (z if self.activation is None else self.activation(z))
      return h_t * i
    self.output, _ = theano.scan(step,
                                 name="scan_%s" % self.name,
                                 go_backwards=self.attrs['reverse'],
                                 truncate_gradient=self.attrs['truncation'],
                                 sequences = [T.stack(self.sources), self.index],
                                 outputs_info = [T.alloc(self.act, self.sources[0].output.shape[1], self.attrs['n_out'])])
    self.output = self.output[::-(2 * self.attrs['reverse'] - 1)]

  def create_recurrent_weights(self, n, m):
    nin = n + m + m + m
    return self.create_random_weights(n, m, nin), self.create_random_weights(m, m, nin)

class LstmLayer(RecurrentLayer):
  def __init__(self, sources, index, n_out, L1 = 0.0, L2 = 0.0, activation = T.nnet.sigmoid, reverse = False, truncation = -1, sharpgates = 'none' , dropout = 0, mask = "unity", projection = None, layer_class = "lstm", name = ""):
    super(LstmLayer, self).__init__(sources, index, n_out * 4, L1, L2, activation, reverse, truncation, False, dropout, mask, projection, layer_class = layer_class, name = name)
    if not isinstance(activation, (list, tuple)):
      activation = [T.tanh, T.nnet.sigmoid, T.nnet.sigmoid, T.nnet.sigmoid, T.tanh]
    else: assert len(activation) == 5, "lstm activations have to be specified as 5 tuple (input, ingate, forgetgate, outgate, output)"
    self.set_attr('sharpgates', sharpgates)
    CI, GI, GF, GO, CO = activation #T.tanh, T.nnet.sigmoid, T.nnet.sigmoid, T.nnet.sigmoid, T.tanh
    n_in = sum([s.attrs['n_out'] for s in sources])
    n_re = projection if projection != None else n_out
    #self.state = self.create_bias(n_out, 'state')
    #self.act = self.create_bias(n_re, 'act')
    self.b.set_value(numpy.zeros((n_out * 3 + n_re,), dtype = theano.config.floatX))
    if projection:
      W_proj = self.create_uniform_weights(n_out, n_re, n_in + n_out + n_re, "W_proj_%s"%self.name)
      self.W_proj.set_value(W_proj.get_value())
    W_re = self.create_uniform_weights(n_re, n_out * 3 + n_re, n_in + n_re + n_out * 3 + n_re, "W_re_%s"%self.name)
    self.W_re.set_value(W_re.get_value())
    for s, W in zip(sources, self.W_in):
      W.set_value(self.create_uniform_weights(s.attrs['n_out'], n_out * 3 + n_re, s.attrs['n_out'] + n_out  + n_out * 3 + n_re, "W_in_%s_%s"%(s.name,self.name)).get_value(borrow=True, return_internal_type=True), borrow = True)
    self.o.set_value(numpy.ones((n_out,), dtype='int8')) #TODO what is this good for?
    if projection:
      self.set_attr('n_out', projection)
    else:
      self.set_attr('n_out', self.attrs['n_out'] / 4)
    if sharpgates == 'global': self.sharpness = self.create_uniform_weights(3, n_out)
    elif sharpgates == 'shared':
      if not hasattr(LstmLayer, 'sharpgates'):
        LstmLayer.sharpgates = self.create_bias(3)
        self.add_param(LstmLayer.sharpgates, 'gate_scaling')
      self.sharpness = LstmLayer.sharpgates
    elif sharpgates == 'single':
      if not hasattr(LstmLayer, 'sharpgates'):
        LstmLayer.sharpgates = self.create_bias(1)
        self.add_param(LstmLayer.sharpgates, 'gate_scaling')
      self.sharpness = LstmLayer.sharpgates
    else: self.sharpness = theano.shared(value = numpy.zeros((3,), dtype=theano.config.floatX), borrow=True, name = 'lambda')
    self.sharpness.set_value(numpy.ones(self.sharpness.get_value().shape, dtype = theano.config.floatX))
    if sharpgates != 'none' and sharpgates != "shared" and sharpgates != "single": self.add_param(self.sharpness, 'gate_scaling')

    def step(*args):
      x_ts = args[:self.num_sources]
      i_t = args[self.num_sources]
      s_p = args[self.num_sources + 1]
      h_p = args[self.num_sources + 2]
      if any(self.masks):
        masks = args[self.num_sources + 3:]
      else:
        masks = [None] * len(self.W_in)

      i = T.outer(i_t, T.alloc(numpy.cast['int8'](1), n_out))
      j = i if not self.W_proj else T.outer(i_t, T.alloc(numpy.cast['int8'](1), n_re))

      z = T.dot(h_p, self.W_re) + self.b
      for x_t, m, W in zip(x_ts, masks, self.W_in):
        if self.attrs['mask'] == "unity":
          z += T.dot(x_t, W)
        else:
          z += T.dot(self.mass * m * x_t, W)

      if sharpgates != 'none':
        ingate = GI(self.sharpness[0] * z[:, n_out: 2 * n_out])
        forgetgate = GF(self.sharpness[1] * z[:, 2 * n_out:3 * n_out])
        outgate = GO(self.sharpness[2] * z[:, 3 * n_out:])
      else:
        ingate = GI(z[:, n_out: 2 * n_out])
        forgetgate = GF(z[:, 2 * n_out:3 * n_out])
        outgate = GO(z[:, 3 * n_out:])
      input = CI(z[:, :n_out])
      s_i = input * ingate + s_p * forgetgate
      s_t = s_i if not self.W_proj else T.dot(s_i, self.W_proj)
      h_t = CO(s_t) * outgate
      return s_i * i, h_t * j


    [state, act], _ = theano.scan(step,
                                  name = "scan_%s"%self.name,
                                  truncate_gradient = self.attrs['truncation'],
                                  go_backwards = self.attrs['reverse'],
                                  sequences = [ s.output for s in self.sources ] + [self.index],
                                  non_sequences = self.masks if any(self.masks) else [],
                                  outputs_info = [ T.alloc(numpy.cast[theano.config.floatX](0), self.sources[0].output.shape[1], n_out),
                                                   T.alloc(numpy.cast[theano.config.floatX](0), self.sources[0].output.shape[1], n_re), ])

    self.output = act[::-(2 * self.attrs['reverse'] - 1)]

#faster but needs much more memory
class OptimizedLstmLayer(RecurrentLayer):
  def __init__(self, sources, index, n_out, L1 = 0.0, L2 = 0.0, activation = T.nnet.sigmoid, reverse = False, truncation = -1, sharpgates = 'none' , dropout = 0, mask = "unity", projection = None, layer_class = "lstm", name = ""):
    super(LstmLayer, self).__init__(sources, index, n_out * 4, L1, L2, activation, reverse, truncation, False, dropout, mask, projection, layer_class = layer_class, name = name)
    if not isinstance(activation, (list, tuple)):
      activation = [T.tanh, T.nnet.sigmoid, T.nnet.sigmoid, T.nnet.sigmoid, T.tanh]
    else: assert len(activation) == 5, "lstm activations have to be specified as 5 tuple (input, ingate, forgetgate, outgate, output)"
    self.set_attr('sharpgates', sharpgates)
    CI, GI, GF, GO, CO = activation #T.tanh, T.nnet.sigmoid, T.nnet.sigmoid, T.nnet.sigmoid, T.tanh
    n_in = sum([s.attrs['n_out'] for s in sources])
    n_re = projection if projection != None else n_out
    #self.state = self.create_bias(n_out, 'state')
    #self.act = self.create_bias(n_re, 'act')
    self.b.set_value(numpy.zeros((n_out * 3 + n_re,), dtype = theano.config.floatX))
    if projection:
      W_proj = self.create_uniform_weights(n_out, n_re, n_in + n_out + n_re, "W_proj_%s"%self.name)
      self.W_proj.set_value(W_proj.get_value())
    W_re = self.create_uniform_weights(n_re, n_out * 3 + n_re, n_in + n_re + n_out * 3 + n_re, "W_re_%s"%self.name)
    self.W_re.set_value(W_re.get_value())
    for s, W in zip(sources, self.W_in):
      W.set_value(self.create_uniform_weights(s.attrs['n_out'], n_out * 3 + n_re, s.attrs['n_out'] + n_out  + n_out * 3 + n_re, "W_in_%s_%s"%(s.name,self.name)).get_value(borrow=True, return_internal_type=True), borrow = True)
    self.o.set_value(numpy.ones((n_out,), dtype='int8'))
    if projection:
      self.set_attr('n_out', projection)
    else:
      self.set_attr('n_out', self.attrs['n_out'] / 4)
    if sharpgates == 'global': self.sharpness = self.create_uniform_weights(3, n_out)
    elif sharpgates == 'shared':
      if not hasattr(LstmLayer, 'sharpgates'):
        LstmLayer.sharpgates = self.create_bias(3)
        self.add_param(LstmLayer.sharpgates, 'gate_scaling')
      self.sharpness = LstmLayer.sharpgates
    elif sharpgates == 'single':
      if not hasattr(LstmLayer, 'sharpgates'):
        LstmLayer.sharpgates = self.create_bias(1)
        self.add_param(LstmLayer.sharpgates, 'gate_scaling')
      self.sharpness = LstmLayer.sharpgates
    else: self.sharpness = theano.shared(value = numpy.zeros((3,), dtype=theano.config.floatX), borrow=True, name = 'lambda')
    self.sharpness.set_value(numpy.ones(self.sharpness.get_value().shape, dtype = theano.config.floatX))
    if sharpgates != 'none' and sharpgates != "shared" and sharpgates != "single": self.add_param(self.sharpness, 'gate_scaling')

    z = self.b
    for x_t, m, W in zip(self.sources, self.masks, self.W_in):
      if self.attrs['mask'] == "unity":
        z += T.dot(x_t.output, W)
      else:
        z += T.dot(self.mass * m * x_t.output, W)

    def step(z, i_t, s_p, h_p):
      z += T.dot(h_p, self.W_re)
      i = T.outer(i_t, T.alloc(numpy.cast['int8'](1), n_out))
      j = i if not self.W_proj else T.outer(i_t, T.alloc(numpy.cast['int8'](1), n_re))
      if sharpgates != 'none':
        ingate = GI(self.sharpness[0] * z[:,n_out: 2 * n_out])
        forgetgate = GF(self.sharpness[1] * z[:,2 * n_out:3 * n_out])
        outgate = GO(self.sharpness[2] * z[:,3 * n_out:])
      else:
        ingate = GI(z[:,n_out: 2 * n_out])
        forgetgate = GF(z[:,2 * n_out:3 * n_out])
        outgate = GO(z[:,3 * n_out:])
      input = CI(z[:,:n_out])
      s_i = input * ingate + s_p * forgetgate
      s_t = s_i if not self.W_proj else T.dot(s_i, self.W_proj)
      h_t = CO(s_t) * outgate
      return s_i * i, h_t * j

    [state, act], _ = theano.scan(step,
                                  name = "scan_%s"%self.name,
                                  truncate_gradient = self.attrs['truncation'],
                                  go_backwards = self.attrs['reverse'],
                                  sequences = [ z, self.index ],
                                  outputs_info = [ T.alloc(numpy.cast[theano.config.floatX](0), self.sources[0].output.shape[1], n_out),
                                                   T.alloc(numpy.cast[theano.config.floatX](0), self.sources[0].output.shape[1], n_re), ])

    self.output = act[::-(2 * self.attrs['reverse'] - 1)]

class NormalizedLstmLayer(RecurrentLayer):
  def __init__(self, sources, index, n_out, L1 = 0.0, L2 = 0.0, activation = T.nnet.sigmoid, reverse = False, truncation = -1, sharpgates = 'none' , dropout = 0, mask = "unity", projection = None, layer_class = "lstm", name = ""):
    super(NormalizedLstmLayer, self).__init__(sources, index, n_out * 4, L1, L2, activation, reverse, truncation, False, dropout, mask, projection, layer_class = layer_class, name = name)
    if not isinstance(activation, (list, tuple)):
      activation = [T.tanh, T.nnet.sigmoid, T.nnet.sigmoid, T.nnet.sigmoid, T.tanh]
    else: assert len(activation) == 5, "lstm activations have to be specified as 5 tuple (input, ingate, forgetgate, outgate, output)"
    self.set_attr('sharpgates', sharpgates)
    CI, GI, GF, GO, CO = activation #T.tanh, T.nnet.sigmoid, T.nnet.sigmoid, T.nnet.sigmoid, T.tanh
    n_in = sum([s.attrs['n_out'] for s in sources])
    n_re = projection if projection != None else n_out
    #self.state = self.create_bias(n_out, 'state')
    #self.act = self.create_bias(n_re, 'act')
    self.b.set_value(numpy.zeros((n_out * 3 + n_re,), dtype = theano.config.floatX))
    if projection:
      W_proj = self.create_uniform_weights(n_out, n_re, n_in + n_out + n_re, "W_proj_%s"%self.name)
      self.W_proj.set_value(W_proj.get_value())
    W_re = self.create_uniform_weights(n_re, n_out * 3 + n_re, n_in + n_re + n_out * 3 + n_re, "W_re_%s"%self.name)
    self.W_re.set_value(W_re.get_value())
    for s, W in zip(sources, self.W_in):
      W.set_value(self.create_uniform_weights(s.attrs['n_out'], n_out * 3 + n_re, s.attrs['n_out'] + n_out  + n_out * 3 + n_re, "W_in_%s_%s"%(s.name,self.name)).get_value(borrow=True, return_internal_type=True), borrow = True)
    self.o.set_value(numpy.ones((n_out,), dtype='int8'))
    if projection:
      self.set_attr('n_out', projection)
    else:
      self.set_attr('n_out', self.attrs['n_out'] / 4)
    if sharpgates == 'global': self.sharpness = self.create_uniform_weights(3, n_out)
    elif sharpgates == 'shared':
      if not hasattr(LstmLayer, 'sharpgates'):
        LstmLayer.sharpgates = self.create_bias(3)
        self.add_param(LstmLayer.sharpgates, 'gate_scaling')
      self.sharpness = LstmLayer.sharpgates
    elif sharpgates == 'single':
      if not hasattr(LstmLayer, 'sharpgates'):
        LstmLayer.sharpgates = self.create_bias(1)
        self.add_param(LstmLayer.sharpgates, 'gate_scaling')
      self.sharpness = LstmLayer.sharpgates
    else: self.sharpness = theano.shared(value = numpy.zeros((3,), dtype=theano.config.floatX), borrow=True, name = 'lambda')
    self.sharpness.set_value(numpy.ones(self.sharpness.get_value().shape, dtype = theano.config.floatX))
    if sharpgates != 'none' and sharpgates != "shared" and sharpgates != "single": self.add_param(self.sharpness, 'gate_scaling')

    #for x_t, m, W in zip(self.sources, self.masks, self.W_in):
    #  if self.attrs['mask'] == "unity":
    #    z += T.dot(x_t.output, W)
    #  else:
    #    z += T.dot(self.mass * m * x_t.output, W)
    assert len(self.sources) == 1
    assert self.attrs['mask'] == "unity"

    n_cells = n_out
    gamma_val = theano.shared(value=numpy.ones((4 * n_cells,), dtype=theano.config.floatX), borrow=True, name='gamma_%s' % self.name)
    self.gamma = self.add_param(gamma_val, 'gamma_%s' % self.name)
    delta_val = theano.shared(value=numpy.ones((4 * n_cells,), dtype=theano.config.floatX), borrow=True, name='delta_%s' % self.name)
    self.delta = self.add_param(delta_val, 'delta_%s' % self.name)
    x = self.sources[0].output
    W = self.W_in[0]
    zx = T.dot(x, W)
    epsilon = 1e-5
    mean = T.concatenate([T.mean(zx[:,:,0*n_cells:1*n_cells], axis=[0,1]), T.mean(zx[:,:,1*n_cells:2*n_cells], axis=[0,1]), T.mean(zx[:,:,2*n_cells:3*n_cells], axis=[0,1]), T.mean(zx[:,:,3*n_cells:4*n_cells], axis=[0,1])], axis=0)
    std = T.sqrt(T.concatenate([T.var(zx[:,:,0*n_cells:1*n_cells], axis=[0,1]), T.var(zx[:,:,1*n_cells:2*n_cells], axis=[0,1]), T.var(zx[:,:,2*n_cells:3*n_cells], axis=[0,1]), T.var(zx[:,:,3*n_cells:4*n_cells], axis=[0,1])], axis=0) + epsilon)
    zx_norm = (zx - mean) / std
    zxs = self.gamma * zx_norm

    def step(zx_t, i_t, s_p, h_p):
      zv_t = T.dot(h_p, self.W_re)
      mean = T.concatenate([T.mean(zv_t[:,0*n_cells:1*n_cells], axis=0), T.mean(zv_t[:,1*n_cells:2*n_cells], axis=0), T.mean(zv_t[:,2*n_cells:3*n_cells], axis=0), T.mean(zv_t[:,3*n_cells:4*n_cells], axis=0)], axis=0)
      std = T.sqrt(T.concatenate([T.var(zv_t[:,0*n_cells:1*n_cells], axis=0), T.var(zv_t[:,1*n_cells:2*n_cells], axis=0), T.var(zv_t[:,2*n_cells:3*n_cells], axis=0), T.var(zv_t[:,3*n_cells:4*n_cells], axis=0)], axis=0) + epsilon)
      zv_t_norm = (zv_t - mean) / std
      zvs_t = self.delta * zv_t_norm
      z = zx_t + zvs_t + self.b

      i = T.outer(i_t, T.alloc(numpy.cast['int8'](1), n_out))
      j = i if not self.W_proj else T.outer(i_t, T.alloc(numpy.cast['int8'](1), n_re))
      if sharpgates != 'none':
        ingate = GI(self.sharpness[0] * z[:,n_out: 2 * n_out])
        forgetgate = GF(self.sharpness[1] * z[:,2 * n_out:3 * n_out])
        outgate = GO(self.sharpness[2] * z[:,3 * n_out:])
      else:
        ingate = GI(z[:,n_out: 2 * n_out])
        forgetgate = GF(z[:,2 * n_out:3 * n_out])
        outgate = GO(z[:,3 * n_out:])
      input = CI(z[:,:n_out])
      s_i = input * ingate + s_p * forgetgate
      s_t = s_i if not self.W_proj else T.dot(s_i, self.W_proj)
      h_t = CO(s_t) * outgate
      return s_i * i, h_t * j

    [state, act], _ = theano.scan(step,
                                  name = "scan_%s"%self.name,
                                  truncate_gradient = self.attrs['truncation'],
                                  go_backwards = self.attrs['reverse'],
                                  sequences = [ zxs, self.index ],
                                  outputs_info = [ T.alloc(numpy.cast[theano.config.floatX](0), self.sources[0].output.shape[1], n_out),
                                                   T.alloc(numpy.cast[theano.config.floatX](0), self.sources[0].output.shape[1], n_re), ])

    self.output = act[::-(2 * self.attrs['reverse'] - 1)]

class WLstmLayer(RecurrentLayer):
  def __init__(self, sources, index, n_out, L1 = 0.0, L2 = 0.0, activation = T.nnet.sigmoid, reverse = False, truncation = -1, sharpgates = 'none' , dropout = 0, mask = "unity", projection = None, layer_class = "lstm", name = ""):
    super(WLstmLayer, self).__init__(sources, index, n_out, L1, L2, activation, reverse, truncation, False, dropout, mask, projection, layer_class = layer_class, name = name)
    if not isinstance(activation, (list, tuple)):
      activation = [T.tanh, T.nnet.sigmoid, T.nnet.sigmoid, T.nnet.sigmoid, T.tanh]
    else: assert len(activation) == 5, "lstm activations have to be specified as 5 tuple (input, ingate, forgetgate, outgate, output)"
    self.set_attr('sharpgates', sharpgates)
    CI, GI, GF, GO, CO = activation #T.tanh, T.nnet.sigmoid, T.nnet.sigmoid, T.nnet.sigmoid, T.tanh
    self.state = self.create_bias(n_out, 'state')
    self.act = self.create_bias(n_out, 'act')
    n_in = sum([s.attrs['n_out'] for s in sources])

    #W_re = self.create_uniform_weights(n_out, n_out * 4, n_in + n_out  + n_out * 4, "W_re_%s"%self.name)
    #self.W_re.set_value(W_re.get_value())
    self.W_re_input = self.add_param(self.create_uniform_weights(n_out, n_out, n_in + n_out  + n_out, "W_re_input_%s"%self.name), "W_re_input_%s"%self.name)
    self.W_re_forget = self.add_param(self.create_uniform_weights(n_out, n_out, n_in + n_out  + n_out, "W_re_forget_%s"%self.name), "W_re_forget_%s"%self.name)
    self.W_re_output = self.add_param(self.create_uniform_weights(n_out, n_out, n_in + n_out  + n_out, "W_re_output_%s"%self.name), "W_re_output_%s"%self.name)

    self.b_input = self.add_param(self.create_bias(n_out, 'b_input'), 'b_input')
    self.b_forget = self.add_param(self.create_bias(n_out, 'b_forget'), 'b_forget')
    self.b_output = self.add_param(self.create_bias(n_out, 'b_output'), 'b_output')

    self.W_input = []
    self.W_forget = []
    self.W_output = []
    for s, W in zip(sources, self.W_in):
      W.set_value(self.create_uniform_weights(s.attrs['n_out'], n_out, s.attrs['n_out'] + n_out  + n_out, "W_in_%s_%s"%(s.name,self.name)).get_value())
      self.W_input.append(self.create_uniform_weights(s.attrs['n_out'], n_out, s.attrs['n_out'] + n_out  + n_out, "W_input_%s_%s"%(s.name,self.name)))
      self.add_param(self.W_input[-1], "W_input_%s_%s"%(s.name,self.name))
      self.W_forget.append(self.create_uniform_weights(s.attrs['n_out'], n_out, s.attrs['n_out'] + n_out  + n_out, "W_forget_%s_%s"%(s.name,self.name)))
      self.add_param(self.W_forget[-1], "W_forget_%s_%s"%(s.name,self.name))
      self.W_output.append(self.create_uniform_weights(s.attrs['n_out'], n_out, s.attrs['n_out'] + n_out  + n_out, "W_output_%s_%s"%(s.name,self.name)))
      self.add_param(self.W_output[-1], "W_output_%s_%s"%(s.name,self.name))

    #for s, W in zip(sources, self.W_in):
    #  W.set_value(self.create_uniform_weights(s.attrs['n_out'], n_out * 4, s.attrs['n_out'] + n_out  + n_out * 4, "W_in_%s_%s"%(s.name,self.name)).get_value())
    self.o.set_value(numpy.ones((n_out,), dtype='int8')) #theano.config.floatX))
    #self.set_attr('n_out', self.attrs['n_out'] / 4)
    if sharpgates == 'global': self.sharpness = self.create_uniform_weights(3, n_out)
    elif sharpgates == 'shared':
      if not hasattr(LstmLayer, 'sharpgates'):
        LstmLayer.sharpgates = self.create_bias(3)
        self.add_param(LstmLayer.sharpgates, 'gate_scaling')
      self.sharpness = LstmLayer.sharpgates
    elif sharpgates == 'single':
      if not hasattr(LstmLayer, 'sharpgates'):
        LstmLayer.sharpgates = self.create_bias(1)
        self.add_param(LstmLayer.sharpgates, 'gate_scaling')
      self.sharpness = LstmLayer.sharpgates
    else: self.sharpness = theano.shared(value = numpy.zeros((3,), dtype=theano.config.floatX), borrow=True, name = 'lambda') #self.create_bias(3)
    self.sharpness.set_value(numpy.ones(self.sharpness.get_value().shape, dtype = theano.config.floatX))
    if sharpgates != 'none' and sharpgates != "shared" and sharpgates != "single": self.add_param(self.sharpness, 'gate_scaling')

    z_in = self.b
    for x_t, m, W in zip(self.sources, self.masks, self.W_in):
      if self.attrs['mask'] == "unity":
        z_in += T.dot(x_t.output, W)
      else:
        z_in += T.dot(self.mass * m * x_t.output, W)

    z_input = self.b_input
    for x_t, m, W in zip(self.sources, self.masks, self.W_input):
      if self.attrs['mask'] == "unity":
        z_input += T.dot(x_t.output, W)
      else:
        z_input += T.dot(self.mass * m * x_t.output, W)

    z_forget = self.b_forget
    for x_t, m, W in zip(self.sources, self.masks, self.W_forget):
      if self.attrs['mask'] == "unity":
        z_forget += T.dot(x_t.output, W)
      else:
        z_forget += T.dot(self.mass * m * x_t.output, W)

    z_output = self.b_output
    for x_t, m, W in zip(self.sources, self.masks, self.W_output):
      if self.attrs['mask'] == "unity":
        z_output += T.dot(x_t.output, W)
      else:
        z_output += T.dot(self.mass * m * x_t.output, W)

    def sstep(z, i_t, s_p, h_p):
      h_pp = T.dot(h_p, self.W_re) if self.W_proj else h_p
      z += T.dot(h_pp, self.W_re)
      i = T.outer(i_t, self.o)
      partition = z.shape[1] / 4
      if sharpgates != 'none':
        ingate = GI(self.sharpness[0] * z[:,partition: 2 * partition])
        forgetgate = GF(self.sharpness[1] * z[:,2 * partition:3 * partition])
        outgate = GO(self.sharpness[2] * z[:,3 * partition:4 * partition])
      else:
        ingate = GI(z[:,partition: 2 * partition])
        forgetgate = GF(z[:,2 * partition:3 * partition])
        outgate = GO(z[:,3 * partition:4 * partition])
      input = CI(z[:,:partition])
      s_t = input * ingate + s_p * forgetgate
      h_t = CO(s_t) * outgate
      return s_t * i, h_t * i

    def step(z_in, z_input, z_forget, z_output, i_t, s_p, h_p):
      z_in += T.dot(h_p, self.W_re)
      z_input += T.dot(h_p, self.W_re_input)
      z_forget += T.dot(h_p, self.W_re_forget)
      z_output += T.dot(h_p, self.W_re_output)
      input = CI(z_in)
      ingate = GI(z_input)
      forgetgate = GF(z_forget)
      outgate = GO(z_output)
      s_t = input * ingate + s_p * forgetgate
      h_t = CO(s_t) * outgate
      i = T.outer(i_t, self.o)
      return s_t * i, h_t * i

    #partition = z.shape[1] / 4
    [self.state, self.act], _ = theano.scan(step,
                                          name = "scan_%s"%self.name,
                                          truncate_gradient = self.attrs['truncation'],
                                          go_backwards = self.attrs['reverse'],
                                          #sequences = [ z, self.index ],
                                          sequences = [ z_in, z_input, z_forget, z_output, self.index ],
                                          outputs_info = [ T.alloc(self.state, self.sources[0].output.shape[1], self.attrs['n_out']),
                                                           T.alloc(self.act, self.sources[0].output.shape[1], self.attrs['n_out']), ])
    self.output = self.act[::-(2 * self.attrs['reverse'] - 1)]

class XLstmLayer(RecurrentLayer):
  def __init__(self, sources, index, n_out, L1 = 0.0, L2 = 0.0, activation = T.nnet.sigmoid, reverse = False, truncation = -1, sharpgates = 'none' , dropout = 0, mask = "unity", projection = None, layer_class = "lstm", name = ""):
    super(LstmLayer, self).__init__(sources, index, n_out * 4, L1, L2, activation, reverse, truncation, False, dropout, mask, projection, layer_class = layer_class, name = name)
    if not isinstance(activation, (list, tuple)):
      activation = [T.tanh, T.nnet.sigmoid, T.nnet.sigmoid, T.nnet.sigmoid, T.tanh]
    else: assert len(activation) == 5, "lstm activations have to be specified as 5 tuple (input, ingate, forgetgate, outgate, output)"
    self.set_attr('sharpgates', sharpgates)
    CI, GI, GF, GO, CO = activation #T.tanh, T.nnet.sigmoid, T.nnet.sigmoid, T.nnet.sigmoid, T.tanh
    self.state = self.create_bias(n_out, 'state')
    self.act = self.create_bias(n_out, 'act')
    n_in = sum([s.attrs['n_out'] for s in sources]) / 2
    W_re = self.create_uniform_weights(n_out, n_out * 4, n_in + n_out  + n_out * 4, "W_re_%s"%self.name)
    self.W_re.set_value(W_re.get_value())
    del self.params['b_%s' % self.name]
    for s, W in zip(sources, self.W_in):
      del self.params["W_in_%s_%s"%(s.name, self.name)]
      #W.set_value(self.create_uniform_weights(s.attrs['n_out'], n_out * 4, s.attrs['n_out'] + n_out  + n_out * 4).get_value())
    self.o.set_value(numpy.ones((n_out,), dtype='int8')) #theano.config.floatX))
    self.set_attr('n_out', self.attrs['n_out'] / 4)
    if sharpgates == 'global': self.sharpness = self.create_uniform_weights(3, n_out)
    elif sharpgates == 'shared':
      if not hasattr(LstmLayer, 'sharpgates'):
        LstmLayer.sharpgates = self.create_bias(3)
        self.add_param(LstmLayer.sharpgates, 'gate_scaling')
      self.sharpness = LstmLayer.sharpgates
    elif sharpgates == 'single':
      if not hasattr(LstmLayer, 'sharpgates'):
        LstmLayer.sharpgates = self.create_bias(1)
        self.add_param(LstmLayer.sharpgates, 'gate_scaling')
      self.sharpness = LstmLayer.sharpgates
    else: self.sharpness = theano.shared(value = numpy.zeros((3,), dtype=theano.config.floatX), borrow=True, name = 'lambda') #self.create_bias(3)
    self.sharpness.set_value(numpy.ones(self.sharpness.get_value().shape, dtype = theano.config.floatX))
    if sharpgates != 'none' and sharpgates != "shared" and sharpgates != "single": self.add_param(self.sharpness, 'gate_scaling')

    # z = self.b
    # for x_t, W in zip(self.sources, self.W_in):
    #   if self.attrs['mask'] == "unity":
    #     z += T.dot(x_t.output, W)
    #   else:
    #     z += T.dot(x_t.output, self.mass * mask * W)

    x_t = sources[0].output
    partition = x_t.shape[2] / 2
    if self.attrs['reverse']:
      z = x_t[:,:,partition : 2 * partition]
    else:
      z = x_t[:,:,:partition]

    def step(q, i_t, s_p, h_p):
      h_pp = T.dot(h_p, self.W_re) if self.W_proj else h_p
      q += T.dot(h_pp, self.W_re)
      #partition = q.shape[1] / 2
      #if self.attrs['reverse']:
      #  z = q[:,:partition]
      #else:
      #  z = q[:,partition:]
      z = q
      i = T.outer(i_t, self.o)
      partition = z.shape[1] / 4
      if sharpgates != 'none':
        ingate = GI(self.sharpness[0] * z[:,partition: 2 * partition])
        forgetgate = GF(self.sharpness[1] * z[:,2 * partition:3 * partition])
        outgate = GO(self.sharpness[2] * z[:,3 * partition:4 * partition])
      else:
        ingate = GI(z[:,partition: 2 * partition])
        forgetgate = GF(z[:,2 * partition:3 * partition])
        outgate = GO(z[:,3 * partition:4 * partition])
      input = CI(z[:,:partition])
      s_t = input * ingate + s_p * forgetgate
      h_t = CO(s_t) * outgate
      return s_t * i, h_t * i

    [self.state, self.act], _ = theano.scan(step,
                                          name = "scan_%s"%self.name,
                                          truncate_gradient = self.attrs['truncation'],
                                          go_backwards = self.attrs['reverse'],
                                          sequences = [ z, self.index ],
                                          outputs_info = [ T.alloc(self.state, self.sources[0].output.shape[1], self.attrs['n_out']),
                                                           T.alloc(self.act, self.sources[0].output.shape[1], self.attrs['n_out']), ])
    self.output = self.act[::-(2 * self.attrs['reverse'] - 1)]

  def create_lstm_weights(self, n, m):
    n_in = n + 4 * m + m + 4 * m
    #scale = numpy.sqrt(12. / (n_in))
    #return self.create_random_weights(n, m * 4, scale), self.create_random_weights(m, m * 4, scale)
    #return self.create_uniform_weights(n, m * 4, n + m), self.create_uniform_weights(m, m * 4, n + m)
    return self.create_uniform_weights(n, m * 4, n + m + m * 4), self.create_uniform_weights(m, m * 4, n + m + m * 4)

  def concat_units(self, other, axis = 1):
    assert other.layer_class == self.layer_class, "unable to concatenate %s (%s) to %s (%s)" % (other.name, other.layer_class, self.name, self.layer_class)
    special_names = [ self.W_re.name ] + [ W_in.name for W_in in self.W_in ] #+ [] if not self.projection else [ self.projection.name ]
    for p in other.params.keys():
      paxis = min(len(self.params[p].get_value().shape) - 1, axis)
      if self.params[p].name in special_names:
        sshape = self.params[p].get_value().shape
        oshape = other.params[p].get_value().shape
        pself = self.params[p].get_value().reshape((sshape[0], sshape[1] / 4, 4))
        pother = other.params[p].get_value().reshape((oshape[0], oshape[1] / 4, 4))
        if p == "W_re":
          dim = pself.shape[0] + pother.shape[0]
          pconcat = numpy.zeros((dim, dim, 4), dtype = theano.config.floatX)
          pconcat[:pself.shape[0],:pself.shape[1],:] = pself
          pconcat[pself.shape[0]:pself.shape[0] + pother.shape[0],pself.shape[1]:pself.shape[1] + pother.shape[1],:] = pother
          concatenation = pconcat.reshape((pconcat.shape[0], pconcat.shape[1] * 4))
        else:
          pconcat = numpy.concatenate((pself, pother), axis = paxis)
          concatenation = pconcat.reshape((pconcat.shape[0], pconcat.shape[1] * 4))
      else:
        concatenation = numpy.concatenate((self.params[p].get_value(), other.params[p].get_value()), axis = paxis)
      self.params[p].set_value(concatenation)
    if axis == 1: self.set_attr('n_out', self.attrs['n_out'] + other.attrs['n_out'])

class MaxLstmLayer(RecurrentLayer):
  def __init__(self, sources, index, n_out, L1 = 0.0, L2 = 0.0, activation = T.nnet.sigmoid, reverse = False, truncation = -1, sharpgates = 'none' , dropout = 0, mask = "unity", projection = None, n_cores = 2, layer_class = "maxlstm", name = ""):
    super(MaxLstmLayer, self).__init__(sources, index, n_out * (2 + n_cores * 2), L1, L2, activation, reverse, truncation, False, dropout, mask, projection, layer_class = layer_class, name = name)
    if not isinstance(activation, (list, tuple)):
      activation = [T.tanh, T.nnet.sigmoid, T.nnet.sigmoid, T.nnet.sigmoid, T.tanh]
    else: assert len(activation) == 5, "lstm activations have to be specified as 5 tuple (input, ingate, forgetgate, outgate, output)"
    self.set_attr('sharpgates', sharpgates)
    self.set_attr('n_cores', n_cores)
    CI, GI, GF, GO, CO = activation #T.tanh, T.nnet.sigmoid, T.nnet.sigmoid, T.nnet.sigmoid, T.tanh
    self.act = self.create_uniform_weights(n_out, n_cores)
    self.state = self.create_uniform_weights(n_out, n_cores)
    n_in = sum([s.attrs['n_out'] for s in sources])
    W_re = self.create_uniform_weights(n_out, n_out * (2 + n_cores * 2), n_in + n_out  + n_out * (2 + n_cores * 2))
    self.W_re.set_value(W_re.get_value())
    for s, W in zip(sources, self.W_in):
      W.set_value(self.create_uniform_weights(s.attrs['n_out'], n_out * 4, s.attrs['n_out'] + n_out + n_out * (2 + n_cores * 2)).get_value())
    self.o.set_value(numpy.ones((n_out,), dtype=theano.config.floatX))
    self.set_attr('n_out', self.attrs['n_out'] / (2 + n_cores * 2))
    if sharpgates == 'global': self.sharpness = self.create_uniform_weights(3, n_out)
    elif sharpgates == 'shared':
      if not hasattr(LstmLayer, 'sharpgates'):
        LstmLayer.sharpgates = self.create_bias(3)
        self.add_param(LstmLayer.sharpgates, 'gate_scaling')
      self.sharpness = LstmLayer.sharpgates
    elif sharpgates == 'single':
      if not hasattr(LstmLayer, 'sharpgates'):
        LstmLayer.sharpgates = self.create_bias(1)
        self.add_param(LstmLayer.sharpgates, 'gate_scaling')
      self.sharpness = LstmLayer.sharpgates
    else: self.sharpness = theano.shared(value = numpy.zeros((3,), dtype=theano.config.floatX), borrow=True, name = 'lambda') #self.create_bias(3)
    self.sharpness.set_value(numpy.ones(self.sharpness.get_value().shape, dtype = theano.config.floatX))
    if sharpgates != 'none' and sharpgates != "shared" and sharpgates != "single": self.add_param(self.sharpness, 'gate_scaling')

    def step(*args):
      x_ts = args[:self.num_sources]
      i_t = args[self.num_sources]
      s_p = args[self.num_sources + 1]
      h_p = args[self.num_sources + 2]
      mask = args[self.num_sources + 3]
      return s_p, h_p
      i = T.outer(i_t, self.o)
      h_pp = T.dot(h_p, self.W_re) if self.W_proj else h_p
      z = T.dot(h_pp, self.W_re) + self.b
      for x_t, m, W in zip(x_ts, self.masks, self.W_in):
        #TODO why here is no check of the mask as in the other layers?
        z += T.dot(self.mass * m * x_t, W)
      partition = z.shape[1] / (2 + self.attrs['n_cores'] * 2)
      #input = CI(z[:,:partition])
      input = CI(T.tile(z[:,:partition], (1, self.attrs['n_cores'])))
      #input = T.stack([CI(z[:,:partition])] * self.attrs['n_cores'])
      ingate = T.reshape(GI(self.sharpness[0] * z[:,partition:partition + partition * self.attrs['n_cores']]), (z.shape[0], partition, self.attrs['n_cores']))
      forgetgate = T.reshape(GF(self.sharpness[1] * z[:,partition + partition * self.attrs['n_cores']:partition + 2 * partition * self.attrs['n_cores']]), (z.shape[0], partition, self.attrs['n_cores']))
      s_t = input * ingate + s_p * forgetgate
      #outgate = GO(self.sharpness[2] * z[:,-partition:])
      outgate = CI(T.tile(GO(self.sharpness[2] * z[:,-partition:]), (1, self.attrs['n_cores'])))
      #outgate = T.stack([ GO(self.sharpness[2] * z[:,-partition:]) ] * self.attrs['n_cores'])
      h_t = CO(s_t) * outgate
      return s_t * i, h_t * i

    [state, self.output], _ = theano.scan(step,
                                          truncate_gradient = self.truncation,
                                          go_backwards = self.reverse,
                                          #sequences = [T.stack(*[ s.output for s in self.sources]), self.index],
                                          sequences = [ s.output for s in self.sources ] + [self.index],
                                          non_sequences = self.masks,
                                          outputs_info = [ T.alloc(self.state, self.sources[0].output.shape[1], self.attrs['n_out'], self.attrs['n_cores']),
                                                           T.alloc(self.act, self.sources[0].output.shape[1], self.attrs['n_out'], self.attrs['n_cores']), ])
    self.output = T.max(self.output, axis = 2)
    self.output = self.output[::-(2 * self.reverse - 1)]

  def create_lstm_weights(self, n, m):
    n_in = n + 4 * m + m + 4 * m
    #scale = numpy.sqrt(12. / (n_in))
    #return self.create_random_weights(n, m * 4, scale), self.create_random_weights(m, m * 4, scale)
    #return self.create_uniform_weights(n, m * 4, n + m), self.create_uniform_weights(m, m * 4, n + m)
    return self.create_uniform_weights(n, m * 4, n + m + m * 4), self.create_uniform_weights(m, m * 4, n + m + m * 4)

class GateLstmLayer(RecurrentLayer):
  def __init__(self, source, index, n_in, n_out, activation = T.nnet.sigmoid, reverse = False, truncation = -1, sharpgates = 'none' , dropout = 0, mask = "unity", name = "lstm"):
    super(GateLstmLayer, self).__init__(source, index, n_in, n_out * 4, activation, reverse, truncation, False, dropout, mask, name = name)
    if not isinstance(activation, (list, tuple)):
      activation = [T.tanh, T.nnet.sigmoid, T.nnet.sigmoid, T.nnet.sigmoid, T.tanh]
    else: assert len(activation) == 5, "lstm activations have to be specified as 5 tuple (input, ingate, forgetgate, outgate, output)"
    CI, GI, GF, GO, CO = activation #T.tanh, T.nnet.sigmoid, T.nnet.sigmoid, T.nnet.sigmoid, T.tanh
    self.act = self.create_bias(n_out)
    self.state = self.create_bias(n_out)
    W_in, W_re = self.create_lstm_weights(n_in, n_out)
    if CI == T.nnet.sigmoid or CO == T.nnet.sigmoid:
      self.W_in.set_value(W_in.get_value()) # * 0.5) # * 0.000001)
      self.W_re.set_value(W_re.get_value()) # * 0.5) # * 0.000001)
    else:
      self.W_in.set_value(W_in.get_value())
      self.W_re.set_value(W_re.get_value())
    self.o.set_value(numpy.ones((n_out,), dtype=theano.config.floatX))
    self.set_attr('n_out', self.attrs['n_out'] / 4)
    if sharpgates == 'global': self.sharpness = self.create_uniform_weights(3, n_out)
    elif sharpgates == 'shared':
      if not hasattr(LstmLayer, 'sharpgates'):
        LstmLayer.sharpgates = self.create_bias(3)
        self.add_param(LstmLayer.sharpgates)
      self.sharpness = LstmLayer.sharpgates
    elif sharpgates == 'single':
      if not hasattr(LstmLayer, 'sharpgates'):
        LstmLayer.sharpgates = self.create_bias(1)
        self.add_param(LstmLayer.sharpgates)
      self.sharpness = LstmLayer.sharpgates
    else: self.sharpness = self.create_bias(3)
    self.sharpness.set_value(numpy.ones(self.sharpness.get_value().shape, dtype = theano.config.floatX))
    if sharpgates != 'none' and sharpgates != "shared" and sharpgates != "single": self.add_param(self.sharpness)

    self.ingate = self.create_bias(n_out)
    self.forgetgate = self.create_bias(n_out)
    self.outgate = self.create_bias(n_out)

    def step(x_t, i_t, s_p, h_p, ig_p, fg_p, og_p, mask):
      i = T.outer(i_t, self.o)
      z = T.dot(self.mass * mask * x_t, self.W_in) + T.dot(h_p, self.W_re) + self.b
      partition = z.shape[1] / 4
      input = CI(z[:,:partition])
      ingate = GI(self.sharpness[0] * z[:,partition: 2 * partition])
      forgetgate = GF(self.sharpness[1] * z[:,2 * partition:3 * partition])
      s_t = input * ingate + s_p * forgetgate
      outgate = GO(self.sharpness[2] * z[:,3 * partition:4 * partition])
      h_t = CO(s_t) * outgate
      return s_t * i, h_t * i, ingate * i, forgetgate * i, outgate * i

    [state, self.output, self.input_gate, self.forget_gate, self.output_gate], _ = theano.scan(step,
                                                                                   truncate_gradient = self.truncation,
                                                                                   go_backwards = self.reverse,
                                                                                   sequences = [self.source, self.index],
                                                                                   non_sequences = [self.mask],
                                                                                   outputs_info = [ T.alloc(self.state, self.source.shape[1], n_out),
                                                                                                    T.alloc(self.act, self.source.shape[1], n_out),
                                                                                                    T.alloc(self.ingate, self.source.shape[1], n_out),
                                                                                                    T.alloc(self.forgetgate, self.source.shape[1], n_out),
                                                                                                    T.alloc(self.outgate, self.source.shape[1], n_out) ])
    self.output = self.output[::-(2 * self.reverse - 1)]
    self.input_gate = self.input_gate[::-(2 * self.reverse - 1)]
    self.forget_gate = self.forget_gate[::-(2 * self.reverse - 1)]
    self.output_gate = self.output_gate[::-(2 * self.reverse - 1)]

  def create_lstm_weights(self, n, m):
    n_in = n + 4 * m + m + 4 * m
    #scale = numpy.sqrt(12. / (n_in))
    #return self.create_random_weights(n, m * 4, scale), self.create_random_weights(m, m * 4, scale)
    #return self.create_uniform_weights(n, m * 4, n + m), self.create_uniform_weights(m, m * 4, n + m)
    return self.create_uniform_weights(n, m * 4, n + m + m * 4), self.create_uniform_weights(m, m * 4, n + m + m * 4)

class LstmPeepholeLayer(LstmLayer):
  def __init__(self, source, index, n_in, n_out, activation = T.nnet.sigmoid, reverse = False, truncation = -1, dropout = 0, mask = "unity", name = "lstm"):
    super(LstmPeepholeLayer, self).__init__(source, index, n_in, n_out, activation, reverse, truncation, dropout, mask, name = name)
    self.peeps_in = self.create_peeps(n_out)
    self.peeps_forget = self.create_peeps(n_out)
    self.peeps_out = self.create_peeps(n_out)
    self.add_param(self.peeps_in)
    self.add_param(self.peeps_forget)
    self.add_param(self.peeps_out)

    def peep(x_t, i_t, s_p, h_p, mask):
      i = T.outer(i_t, self.o)
      z = T.dot(x_t, self.mass * mask * self.W_in) + T.dot(h_p, self.W_re) + self.b
      partition = z.shape[1] / 4
      CI = T.tanh
      CO = T.tanh
      G = T.nnet.sigmoid
      pi = s_p * self.peeps_in
      pf = s_p * self.peeps_forget
      input = CI(z[:,:partition])
      ingate = G(z[:,partition: 2 * partition] + pi)
      forgetgate = G(z[:,2 * partition:3 * partition] + pf)
      s_t = input * ingate + s_p * forgetgate
      po = s_t * self.peeps_out
      outgate = G(z[:,3 * partition:4 * partition] + po)
      h_t = CO(s_t) * outgate
      return s_t * i, h_t * i

    [pstate, peep_output], _ = theano.scan(peep,
                                       truncate_gradient = 0,
                                       go_backwards = self.reverse,
                                       sequences=[self.source, self.index],
                                       non_sequences=[self.mask],
                                       outputs_info=[ T.alloc(self.state, self.source.shape[1], n_out),
                                                      T.alloc(self.act, self.source.shape[1], n_out), ])
    self.output = 0.5 * (peep_output + self.output)
    self.output = self.output[::-(2 * self.reverse - 1)]

  def create_peeps(self, n):
    values = numpy.asarray(self.rng.normal(loc=0.0,
                                           scale=numpy.sqrt(.6/(4 * self.n_out)),
                                           size=(n, )), dtype=theano.config.floatX)
    return theano.shared(value=values, borrow=True)


"""
        Network Topology Description
"""

class LayerNetworkDescription:

  def __init__(self, num_inputs, num_outputs,
               hidden_info,
               loss, L1_reg, L2_reg, dropout=(),
               bidirectional=True, sharpgates='none',
               truncation=-1, entropy=0):
    """
    :type num_inputs: int
    :type num_outputs: int
    :param list[(str,int,(str,theano.Op)|list[(str,theano.Op)],str)] hidden_info: list of
      (layer_type, size, activation, name)
    :param str loss: loss type, "ce", "ctc" etc
    :type L1_reg: float
    :type L2_reg: float
    :type dropout: list[float]
    :type bidirectional: bool
    :param str sharpgates: see LSTM layers
    :param int truncation: number of steps to use in truncated BPTT or -1. see theano.scan
    :param float entropy: ...
    """
    assert len(dropout) == len(hidden_info) + 1
    self.num_inputs = num_inputs
    self.num_outputs = num_outputs
    self.hidden_info = list(hidden_info)
    self.loss = loss
    self.L1_reg = L1_reg
    self.L2_reg = L2_reg
    self.dropout = list(dropout)
    self.bidirectional = bidirectional
    self.sharpgates = sharpgates
    self.truncation = truncation
    self.entropy = entropy

  def __eq__(self, other):
    return self.init_args() == getattr(other, "init_args", lambda: {})()

  def __ne__(self, other):
    return not self == other

  def init_args(self):
    return {arg: getattr(self, arg) for arg in inspect.getargspec(self.__init__).args[1:]}

  __repr__ = simpleObjRepr

  def copy(self):
    args = self.init_args()
    return self.__class__(**args)

  @classmethod
  def from_config(cls, config):
    """
    :type config: Config.Config
    :returns dict
    """
    num_inputs, num_outputs = cls.num_inputs_outputs_from_config(config)
    loss = cls.loss_from_config(config)
    hidden_size = config.int_list('hidden_size')
    assert len(hidden_size) > 0, "no hidden layers specified"
    hidden_type = config.list('hidden_type')
    assert len(hidden_type) <= len(hidden_size), "too many hidden layer types"
    hidden_name = config.list('hidden_name')
    assert len(hidden_name) <= len(hidden_size), "too many hidden layer names"
    if len(hidden_type) != len(hidden_size):
      n_hidden_type = len(hidden_type)
      for i in xrange(len(hidden_size) - len(hidden_type)):
        if n_hidden_type == 1:
          hidden_type.append(hidden_type[0])
        else:
          hidden_type.append("forward")
    if len(hidden_name) != len(hidden_size):
      for i in xrange(len(hidden_size) - len(hidden_name)):
        hidden_name.append("_")
    for i, name in enumerate(hidden_name):
      if name == "_": hidden_name[i] = "hidden_%d" % i
    L1_reg = config.float('L1_reg', 0.0)
    L2_reg = config.float('L2_reg', 0.0)
    bidirectional = config.bool('bidirectional', True)
    truncation = config.int('truncation', -1)
    actfct = config.list('activation')
    dropout = config.list('dropout', [0.0])
    sharpgates = config.value('sharpgates', 'none')
    entropy = config.float('entropy', 0.0)
    if len(actfct) < len(hidden_size):
      for i in xrange(len(hidden_size) - len(actfct)):
        actfct.append("logistic")
    if len(dropout) < len(hidden_size) + 1:
      for i in xrange(len(hidden_size) + 1 - len(dropout)):
        dropout.append(0.0)
    dropout = [float(d) for d in dropout]
    hidden_info = []; """ :type: list[(str,int,(str,theano.Op)|list[(str,theano.Op)],str)] """
    """
    That represents (layer_type, size, activation, name),
    where activation is either a list of activation functions or a single one.
    Such activation function is a tuple (str,theano.Op).
    name is a custom name for the layer, such as "hidden_2".
    """
    for i in xrange(len(hidden_size)):
      if ':' in actfct[i]:
        acts = []; """ :type: list[(str,theano.Op)] """
        for a in actfct[i].split(':'):
          acts.append((a, strtoact(a)))
      else:
        acts = (actfct[i], strtoact(actfct[i]))
      """
      hidden_name[i]: custom name of the hidden layer, such as "hidden_2"
      hidden_type[i]: e.g. 'forward'
      acts: activation function, e.g. ("tanh", T.tanh)
      """
      hidden_info.append((hidden_type[i], hidden_size[i], acts, hidden_name[i]))

    return cls(num_inputs=num_inputs, num_outputs=num_outputs,
               hidden_info=hidden_info,
               loss=loss, L1_reg=L1_reg, L2_reg=L2_reg, dropout=dropout,
               bidirectional=bidirectional, sharpgates=sharpgates,
               truncation=truncation, entropy=entropy)

  @classmethod
  def loss_from_config(cls, config):
    """
    :type config: Config.Config
    :rtype: str
    """
    return config.value('loss', 'ce')

  @classmethod
  def num_inputs_outputs_from_config(cls, config):
    """
    :type config: Config.Config
    :rtype: (int,int)
    """
    num_inputs = config.int('num_inputs', 0)
    num_outputs = config.int('num_outputs', 0)
    if config.list('train'):
      _num_inputs = hdf5_dimension(config.list('train')[0], 'inputPattSize') * config.int('window', 1)
      _num_outputs = hdf5_dimension(config.list('train')[0], 'numLabels')
      if num_inputs: assert num_inputs == _num_inputs
      if num_outputs: assert num_outputs == _num_outputs
      num_inputs = _num_inputs
      num_outputs = _num_outputs
    assert num_inputs and num_outputs, "provide num_inputs/num_outputs directly or via train"
    loss = cls.loss_from_config(config)
    if loss in ('ctc', 'ce_ctc') or config.bool('add_blank', False):
      num_outputs += 1  # add blank
    return num_inputs, num_outputs


LayerClasses = {
  'forward': ForwardLayer,  # used in crnn.config format
  'hidden': ForwardLayer,  # used in JSON format
  'recurrent': RecurrentLayer,
  'lstm': LstmLayer,
  'lstm_opt': OptimizedLstmLayer,
  'lstm_norm': NormalizedLstmLayer,
  'gatelstm': GateLstmLayer,
  'peep_lstm': LstmPeepholeLayer,
  'maxlstm': MaxLstmLayer
}


def get_layer_class(name):
  if name in LayerClasses:
    return LayerClasses[name]
  assert False, "invalid layer type: " + name


"""
        NETWORKS
"""

class LayerNetwork(object):
  def __init__(self, n_in, n_out, mask="unity"):
    """
    :param int n_in: input dim of the network
    :param int n_out: output dim of the network
    :param str mask: e.g. "unity"
    """
    self.x = T.tensor3('x'); """ :type: theano.Variable """
    self.y = T.ivector('y'); """ :type: theano.Variable """
    self.c = T.imatrix('c'); """ :type: theano.Variable """
    self.i = T.bmatrix('i'); """ :type: theano.Variable """
    Layer.initialize_rng()
    self.n_in = n_in
    self.n_out = n_out
    self.mask = mask
    self.hidden = {}; """ :type: dict[str,ForwardLayer|RecurrentLayer] """
    self.train_params_vars = []; """ :type: list[theano.compile.sharedvalue.SharedVariable] """
    self.description = None; """ :type: LayerNetworkDescription | None """
    self.train_param_args = None; """ :type: dict[str] """

  @classmethod
  def from_config_topology(cls, config, mask="unity"):
    """
    :type config: Config.Config
    :param str mask: e.g. "unity" or "dropout"
    :rtype: LayerNetwork
    """
    if config.network_topology_json is not None:
      num_inputs, num_outputs = LayerNetworkDescription.num_inputs_outputs_from_config(config)
      return cls.from_json(config.network_topology_json, num_inputs, num_outputs)

    description = LayerNetworkDescription.from_config(config)
    return cls.from_description(description, mask)

  @classmethod
  def from_description(cls, description, mask="unity"):
    """
    :type description: LayerNetworkDescription
    :param str mask: e.g. "unity" or "dropout"
    :rtype: LayerNetwork
    """
    network = cls(description.num_inputs, description.num_outputs, mask)
    network.initialize(description)
    return network

  @classmethod
  def from_json(cls, json_content, n_in, n_out, mask=None):
    """
    :type json_content: str
    :type n_in: int
    :type n_out: int
    :type mask: str
    :rtype: LayerNetwork
    """
    network = cls(n_in, n_out, mask)
    try:
      topology = json.loads(json_content)
    except ValueError:
      print >> log.v4, "----- BEGIN JSON CONTENT -----"
      print >> log.v4, json_content
      print >> log.v4, "------ END JSON CONTENT ------"
      assert False, "invalid json content"
    if hasattr(LstmLayer, 'sharpgates'):
      del LstmLayer.sharpgates
    network.L1 = T.constant(0)
    network.L2 = T.constant(0)
    network.recurrent = False
    def traverse(content, layer_name, network):
      source = []
      obj = content[layer_name].copy()
      act = obj.pop('activation', 'logistic')
      cl = obj.pop('class', None)
      if not obj.has_key('from'):
        source = [SourceLayer(network.n_in, network.x, name = 'data')]
      else:
        for prev in obj['from']:
          if not network.hidden.has_key(prev):
            traverse(content, prev, network)
          source.append(network.hidden[prev])
      obj.pop('from', None)
      params = { 'sources': source }
      params.update(obj)
      if cl == 'softmax':
        network.make_classifier(params['sources'], params['loss'])
      else:
        layer_class = get_layer_class(cl)
        params.update({'activation': strtoact(act), 'name': layer_name})
        if layer_class.recurrent:
          network.recurrent = True
          params['index'] = network.i
        network.add_layer(layer_name, layer_class(**params), act)
    traverse(topology, 'output', network)
    return network

  @classmethod
  def from_hdf_model_topology(cls, model, mask="unity"):
    """
    :type model: h5py.File
    :param str mask: e.g. "unity"
    :rtype: LayerNetwork
    """
    grp = model['training']
    if mask is None: mask = grp.attrs['mask']
    network = cls(model.attrs['n_in'], model.attrs['n_out'], mask)
    network.L1 = T.constant(0)
    network.L2 = T.constant(0)
    network.recurrent = False
    def traverse(model, layer_name, network):
      if 'from' in model[layer_name].attrs and model[layer_name].attrs['from'] != 'data':
        x_in = []
        for s in model[layer_name].attrs['from'].split(','):
          if s == 'data':
            x_in.append(SourceLayer(network.n_in, network.x, name = 'data'))
          else:
            if not network.hidden.has_key(s):
              traverse(model, s, network)
            x_in.append(network.hidden[s])
      else:
        x_in = [ SourceLayer(network.n_in, network.x, name = 'data') ]
      if layer_name != model.attrs['output']:
        cl = model[layer_name].attrs['class']
        act = model[layer_name].attrs['activation']
        params = { 'sources': x_in,
                   'n_out': model[layer_name].attrs['n_out'],
                   'activation': strtoact(act),
                   'dropout': model[layer_name].attrs['dropout'],
                   'name': layer_name,
                   'mask': model[layer_name].attrs['mask'] }
        layer_class = get_layer_class(cl)
        if layer_class.recurrent:
          network.recurrent = True
          params['index'] = network.i
          for p in ['truncation', 'projection', 'reverse', 'sharpgates']:
            if p in model[layer_name].attrs:
              params[p] = model[layer_name].attrs[p]
        network.add_layer(layer_name, layer_class(**params), act)
    output = model.attrs['output']
    traverse(model, output, network)
    sources = [ network.hidden[s] for s in model[output].attrs['from'].split(',') ]
    loss = 'ce' if not 'loss' in model[output].attrs else model[output].attrs['loss']
    network.make_classifier(sources, loss, model[output].attrs['dropout'])
    return network

  def add_layer(self, name, layer, activation):
    """
    :type name: str
    :type layer: HiddenLayer
    :param str activation: activation function name
    """
    self.hidden[name] = layer
    self.hidden[name].set_attr('activation', activation)
    if isinstance(layer, RecurrentLayer):
      if layer.attrs['L1'] > 0.0: self.L1 += layer.attrs['L1'] * abs(layer.W_re.sum())
      if layer.attrs['L2'] > 0.0: self.L2 += layer.attrs['L2'] * (layer.W_re ** 2).sum()
    for W in layer.W_in:
      if layer.attrs['L1'] > 0.0: self.L1 += layer.attrs['L1'] * abs(W.sum())
      if layer.attrs['L2'] > 0.0: self.L2 += layer.attrs['L2'] * (W ** 2).sum()

  def make_classifier(self, sources, loss, dropout=0, mask="unity"):
    """
    :param list[Layer] sources: source layers
    :param str loss: loss type, "ce", "ctc" etc
    :type dropout: float
    :param str mask: "unity" or "dropout"
    """
    self.loss = loss
    if loss in ('ctc', 'ce_ctc', 'sprint', 'sprint_smoothed'):
      self.output = SequenceOutputLayer(sources=sources, index=self.i, n_out=self.n_out, loss=loss, dropout=dropout, mask=mask, name="output")
    else:
      self.output = FramewiseOutputLayer(sources=sources, index=self.i, n_out=self.n_out, loss=loss, dropout=dropout, mask=mask, name="output")
    for W in self.output.W_in:
      if self.output.attrs['L1'] > 0.0: self.L1 += self.output.attrs['L1'] * abs(W.sum())
      if self.output.attrs['L2'] > 0.0: self.L2 += self.output.attrs['L2'] * (W ** 2).sum()
    self.declare_train_params()
    targets = self.c if self.loss == 'ctc' else self.y
    error_targets = self.c if self.loss in ('ctc','ce_ctc') else self.y
    self.errors = self.output.errors(error_targets)
    cost = self.output.cost(targets)
    self.cost, self.known_grads = cost[:2]
    if len(cost) > 2:
      self.ctc_priors = cost[2]
      assert self.ctc_priors is not None
    else:
      self.ctc_priors = None
    self.objective = self.cost + self.L1 + self.L2 #+ entropy * self.output.entropy()
    #if hasattr(LstmLayer, 'sharpgates'):
      #self.objective += entropy * (LstmLayer.sharpgates ** 2).sum()
    #self.jacobian = T.jacobian(self.output.z, self.x)

  def get_params_vars(self, hidden_layer_selection, with_output):
    """
    :type hidden_layer_selection: list[str]
    :type with_output: bool
    :rtype: list[theano.compile.sharedvalue.SharedVariable]
    :returns list (with well-defined order) of shared variables
    """
    params = []
    """ :type: list[theano.compile.sharedvalue.SharedVariable] """
    for name in sorted(hidden_layer_selection):
      params += self.hidden[name].get_params_vars()
    if with_output:
      params += self.output.get_params_vars()
    return params

  def get_all_params_vars(self):
    return self.get_params_vars(**self.get_train_param_args_default())

  def get_train_param_args_default(self):
    """
    :returns default kwargs for self.get_params(), which returns all params with this.
    """
    return {
      "hidden_layer_selection": sorted(self.hidden.keys()),  # Use all.
      "with_output": True
    }

  def declare_train_params(self, **kwargs):
    """
    Kwargs as in self.get_params(), or default values.
    """
    # Set default values, also for None.
    for key, value in self.get_train_param_args_default().items():
      if kwargs.get(key, None) is None:
        kwargs[key] = value
    # Force a unique representation of kwargs.
    kwargs["hidden_layer_selection"] = sorted(kwargs["hidden_layer_selection"])
    self.train_param_args = kwargs
    self.train_params_vars = self.get_params_vars(**kwargs)

  def initialize(self, description):
    """
    :type description: LayerNetworkDescription
    Initializes the network based the description.
    """
    assert description.num_inputs == self.n_in
    assert description.num_outputs == self.n_out
    assert len(description.hidden_info) > 0
    self.description = description
    n_in = self.n_in
    x_in = self.x
    self.L1 = T.constant(0)
    self.L2 = T.constant(0)
    self.L1_reg = description.L1_reg
    self.L2_reg = description.L2_reg
    self.recurrent = False
    self.bidirectional = description.bidirectional
    if hasattr(LstmLayer, 'sharpgates'):
      del LstmLayer.sharpgates
    last_layer = None
    # create forward layers
    for info, drop in zip(description.hidden_info, description.dropout[:-1]):
      #params = { 'source': x_in, 'n_in': n_in, 'n_out': info[1], 'activation': info[2][1], 'dropout': drop, 'name': info[3], 'mask': self.mask }
      srcs = [SourceLayer(n_out=n_in, x_out=x_in, name='')]
      params = { 'sources': srcs, 'n_out': info[1], 'activation': info[2][1], 'dropout': drop, 'name': info[3], 'mask': self.mask }
      act = info[2][0]
      layer_class = get_layer_class(info[0])
      if layer_class.recurrent:
        self.recurrent = True
        params['index'] = self.i
        params['truncation'] = description.truncation
        if self.bidirectional:
          params['name'] = info[3] + "_fw"
        name = params['name']
        if 'sharpgates' in inspect.getargspec(layer_class.__init__).args[1:]:
          params['sharpgates'] = description.sharpgates
      name = params['name']; """ :type: str """
      self.add_layer(name, layer_class(**params), act)
      last_layer = self.hidden[name]
      n_in = info[1]
      x_in = last_layer.output
    sources = [last_layer]
    # create backward layers
    assert self.recurrent or not self.bidirectional, "non-recurrent networks can not be bidirectional"
    if self.bidirectional:
      last_layer = None
      n_in = self.n_in
      x_in = self.x
      for info, drop in zip(description.hidden_info, description.dropout[:-1]):
        #params = { 'source': x_in, 'n_in': n_in, 'n_out': info[1], 'activation': info[2][1], 'dropout': drop, 'name': info[3] + "_bw", 'mask': self.mask }
        srcs = [SourceLayer(n_out=n_in, x_out=x_in, name='')]
        params = { 'sources': srcs, 'n_out': info[1], 'activation': info[2][1], 'dropout': drop, 'name': info[3] + "_bw", 'mask': self.mask }
        act = info[2][0]
        layer_class = get_layer_class(info[0])
        if layer_class.recurrent:
          params['index'] = self.i
          params['truncation'] = description.truncation
          if 'sharpgates' in inspect.getargspec(layer_class.__init__).args[1:]:
            params['sharpgates'] = description.sharpgates
          params['reverse'] = True
        name = params['name']; """ :type: str """
        self.add_layer(name, layer_class(**params), act)
        last_layer = self.hidden[name]
        n_in = info[1]
        x_in = last_layer.output
      sources.append(last_layer)
    self.make_classifier(sources, description.loss, description.dropout[-1], self.mask)

  def num_params(self):
    return sum([self.hidden[h].num_params() for h in self.hidden]) + self.output.num_params()

  def get_params_dict(self):
    """
    :rtype: dict[str,dict[str,numpy.ndarray|theano.sandbox.cuda.CudaNdArray]]
    """
    params = {self.output.name: self.output.get_params_dict()}
    for h in self.hidden:
      params[h] = self.hidden[h].get_params_dict()
    return params

  def set_params_by_dict(self, params):
    """
    :type params: dict[str,dict[str,numpy.ndarray|theano.sandbox.cuda.CudaNdArray]]
    """
    self.output.set_params_by_dict(params[self.output.name])
    for h in self.hidden:
      self.hidden[h].set_params_by_dict(params[h])

  def save_hdf(self, model, epoch):
    """
    :type model: h5py.File
    :type epoch: int
    """
    grp = model.create_group('training')
    model.attrs['epoch'] = epoch
    model.attrs['output'] = self.output.name
    model.attrs['n_in'] = self.n_in
    model.attrs['n_out'] = self.n_out
    for h in self.hidden:
      self.hidden[h].save(model)
    self.output.save(model)

  def to_json(self):
    outattrs = self.output.attrs.copy()
    outattrs['from'] = outattrs['from'].split(',')
    out = { 'output' : outattrs }
    for h in self.hidden.keys():
      out[h] = self.hidden[h].to_json()
    return str(out)

  def load_hdf(self, model):
    """
    :type model: h5py.File
    :returns last epoch this was trained on
    :rtype: int
    """
    for name in self.hidden:
      if not name in model:
        print >> log.v2, "unable to load layer ", name
      else:
        self.hidden[name].load(model)
    self.output.load(model)
    return self.epoch_from_hdf_model(model)

  @classmethod
  def epoch_from_hdf_model(cls, model):
    """
    :type model: h5py.File
    :returns last epoch the model was trained on
    :rtype: int
    """
    epoch = model.attrs['epoch']
    return epoch

  @classmethod
  def epoch_from_hdf_model_filename(cls, model_filename):
    """
    :type model_filename: str
    :returns last epoch the model was trained on
    :rtype: int
    """
    model = h5py.File(model_filename, "r")
    epoch = cls.epoch_from_hdf_model(model)
    model.close()
    return epoch

