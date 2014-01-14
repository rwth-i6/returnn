#! /usr/bin/python2.7

import numpy
import theano
import theano.tensor as T
from Util import netcdf_dimension
from math import sqrt

"""
        META LAYER
"""
class Container(object):
  def __init__(self, name):
    self.params = []
    self.name = name
  
  @staticmethod
  def initialize():
    Layer.rng = numpy.random.RandomState(1234)
    
  def print_param(self, p):
    matrix = p.get_value().tolist()
    shape = numpy.shape(matrix)
    dim = len(shape)
    if dim == 2: return ";".join([",".join([str(x) for x in row]) for row in matrix])
    else: return ",".join([str(x) for x in matrix])
    
  def save(self, out):
    print >> out, self.name,
    for p in self.params:
      print >> out, self.print_param(p),
    print >> out, ''

  def load(self, layer):
    layer = layer.strip().split()
    assert layer[0] == self.name, "invalid layer name (expected " + self.name + " got " + layer[0] + ")"
    assert len(layer) - 1 == len(self.params), "invalid layer parameter count (expected " + str(len(self.params)) + " got " + str(len(layer) - 1) + ")"
    for l in xrange(len(layer) - 1):
      rows = layer[l + 1].split(';')
      if len(rows) == 1:
        value = numpy.array([float(x) for x in rows[0].split(',')], dtype=theano.config.floatX)
      else: # nr of rwos as n_in, nr of cols as n_out
        value = numpy.array([[float(x) for x in row.split(',')] for row in rows], dtype=theano.config.floatX)
      #######################################################################################################
      #if self.name == 'bisoftmax':
      #  mapping = [ int(line.strip()) for line in open('/home/creon/map.clean') ]
      #  if len(rows) == 1: valuex = numpy.array([value[i] for i in mapping], dtype = theano.config.floatX)
      #  else:
      #    valuex = numpy.zeros((value.shape[0], len(mapping)), dtype = theano.config.floatX)
      #    for i,j in enumerate(mapping):
      #      valuex[:, i] = value[:, j]
      #  value = valuex
      #######################################################################################################
      assert self.params[l].get_value().shape == value.shape, "invalid layer parameter shape (expected  " + str(self.params[l].get_value().shape) + " got " + str(value.shape) + ")"
      self.params[l].set_value(value)
        
  def num_params(self):
    return sum([numpy.prod(p.get_value().shape[0:]) for p in self.params])
  
  def get_params(self):
    return [ p.get_value() for p in self.params ]
    
  def set_params(self, params):
    for p,q in zip(self.params, params):
      p.set_value(q)
      
  def add_param(self, param):
    self.params.append(param)
    return param
      
  def create_bias(self, n):
    return theano.shared(value = numpy.zeros((n,), dtype=theano.config.floatX), borrow=True, name = 'b_' + self.name)
  
  def create_random_weights(self, n, m, s):
    values = numpy.asarray(self.rng.normal(loc = 0.0, scale = s, size=(n, m)), dtype=theano.config.floatX)
    return theano.shared(value = values, borrow = True, name = 'W_' + self.name)
  
  def create_uniform_weights(self, n, m, p = 0):
    if p == 0: p = n + m
    #values = numpy.asarray(self.rng.uniform(low = - 1 / sqrt(p), high = 1 / sqrt(p), size=(n, m)), dtype=theano.config.floatX)
    values = numpy.asarray(self.rng.uniform(low = - sqrt(6) / sqrt(p), high = sqrt(6) / sqrt(p), size=(n, m)), dtype=theano.config.floatX)
    return theano.shared(value = values, borrow = True, name = 'W_' + self.name)
  
  def create_forward_weights(self, n, m):
    n_in = n + m
    scale = numpy.sqrt(12. / (n_in))
    return self.create_random_weights(n, m, scale)
  
class Layer(Container):
  def __init__(self, n_in, n_out, name, mask = "unity", dropout = 0):
    super(Layer, self).__init__(name)
    self.n_in = n_in
    self.n_out = n_out
    self.b = self.add_param(self.create_bias(n_out))
    self.mass = T.constant(1.)
    if mask == "unity":
      self.mask = T.constant(1.)
      if dropout > 0:
        self.mass = T.constant(dropout)
    elif mask == "dropout":
      srng = theano.tensor.shared_randomstreams.RandomStreams(self.rng.randint(1234))
      self.mask = T.cast(srng.binomial(n=1, p=1-dropout, size=(n_in, n_out)), theano.config.floatX)
      
"""
        OUTPUT LAYERS
"""

class SoftmaxLayer(Layer):
  def __init__(self, source, index, n_in, n_out, dropout = 0, mask = "unity", name = "softmax"):
    super(SoftmaxLayer, self).__init__(n_in, n_out, name, mask, dropout)
    self.W_in = self.add_param(self.create_forward_weights(n_in, n_out))
    self.z = T.dot(source, self.mass * self.mask * self.W_in) + self.b
    self.i = index.flatten() #T.cast(T.reshape(index, (self.z.shape[0] * self.z.shape[1],)), 'int8')
    self.initialize_softmax()    
    
  def initialize_softmax(self):
    self.y_m = T.reshape(self.z, (self.z.shape[0] * self.z.shape[1], self.z.shape[2]), ndim = 2)
    #self.y_m = T.dimshuffle(T.flatten(T.dimshuffle(self.z, (2, 0, 1)), ndim = 2), (1, 0))
    #self.y_m = self.z.dimshuffle(2, 0, 1).flatten(ndim = 2).dimshuffle(1, 0)
    #self.y_m = T.reshape(self.z, (self.z.shape[0] * self.z.shape[1], self.z.shape[2]), ndim = 2)
    self.p_y_given_x = T.nnet.softmax(self.y_m)
    self.y_pred = T.argmax(self.p_y_given_x, axis = -1)

  def cost(self, y):
    y_f = T.cast(T.reshape(y, (y.shape[0] * y.shape[1]), ndim = 1), 'int32')
    #y_f = y.dimshuffle(2, 0, 1).flatten(ndim = 2).dimshuffle(1, 0)
    #y_f = y.flatten()
    pcx = self.p_y_given_x[(self.i > 0).nonzero(), y_f[(self.i > 0).nonzero()]]
    #return -T.sum(T.log(self.p_y_given_x)[(self.i > 0).nonzero(), y_f[(self.i > 0).nonzero()]])
    #return -T.sum(T.log(pcx))
    return -T.sum(T.log(pcx)) #* T.sum(self.p_y_given_x[(self.i > 0).nonzero()] * T.log(self.p_y_given_x[(self.i > 0).nonzero()]))
  
  def entropy(self):
    return -T.sum(self.p_y_given_x[(self.i > 0).nonzero()] * T.log(self.p_y_given_x[(self.i > 0).nonzero()]))

  def errors(self, y):
    y_f = y.flatten() #T.cast(T.reshape(y, (y.shape[0] * y.shape[1]), ndim = 1), 'int32') #y_f = y.flatten(ndim=1)
    if y_f.dtype.startswith('int'):
      return T.sum(T.neq(self.y_pred[(self.i > 0).nonzero()], y_f[(self.i > 0).nonzero()]))
    else: raise NotImplementedError()
    
class BidirectionalSoftmaxLayer(SoftmaxLayer):  
  def __init__(self, forward, backward, index, n_in, n_out, dropout = 0, mask = "unity", name = "bisoftmax"):
    super(BidirectionalSoftmaxLayer, self).__init__(forward, index, n_in, n_out, dropout, mask, name = name)
    self.W_reverse = self.add_param(self.create_forward_weights(n_in, n_out))
    self.z += T.dot(backward, self.mass * self.mask * self.W_reverse)
    self.initialize_softmax()
    
"""
        HIDDEN LAYERS
"""

class HiddenLayer(Layer):
  def __init__(self, source, n_in, n_out, activation, dropout = 0, mask = "unity", name = "hidden"):
    super(HiddenLayer, self).__init__(n_in, n_out, name, mask, dropout)
    self.n_in = n_in
    self.n_out = n_out
    self.activation = activation
    self.W_in = self.add_param(self.create_forward_weights(n_in, n_out))
    self.source = source
    
class ForwardLayer(HiddenLayer):
  def __init__(self, source, n_in, n_out, activation=T.tanh, dropout = 0, mask = "unity", name = "hidden"):
    super(ForwardLayer, self).__init__(source, n_in, n_out, activation, dropout, mask, name = name)
    #values = numpy.asarray(self.rng.uniform(low = -numpy.sqrt(6. / (n_in + n_out)),
    #                                        high = numpy.sqrt(6. / (n_in + n_out)),
    #                                        size = (n_in, n_out)), dtype=theano.config.floatX)
    #if self.activation == theano.tensor.nnet.sigmoid: values *= 4
    self.W_in.set_value(self.create_uniform_weights(n_in, n_out).get_value())
    z = T.dot(source, self.mass * self.mask * self.W_in) + self.b
    self.output = (z if self.activation is None else self.activation(z))
    
class RecurrentLayer(HiddenLayer):
  def __init__(self, source, index, n_in, n_out, activation = T.tanh, reverse = False, truncation = -1, compile = True, dropout = 0, mask = "unity", name = "recurrent"):
    super(RecurrentLayer, self).__init__(source, n_in, n_out, activation, dropout, mask, name = name)
    self.act = self.create_bias(n_out)
    W_in, self.W_re = self.create_recurrent_weights(n_in, n_out)
    self.W_in.set_value(W_in.get_value())
    self.add_param(self.W_re)
    self.index = T.cast(index, theano.config.floatX)
    self.o = theano.shared(value = numpy.ones((n_out,), dtype=theano.config.floatX), borrow=True)
    self.reverse = reverse
    self.truncation = truncation
    if compile: self.compile()
  
  def compile(self):
    def step(x_t, i_t, h_p):
      i = T.outer(i_t, self.o)
      z = T.dot(x_t, self.mass * self.mask * self.W_in) + T.dot(h_p, self.W_re) + self.b
      h_t = (z if self.activation is None else self.activation(z))
      return h_t * i
    
    self.output, _ = theano.scan(step,
                                 go_backwards = self.reverse,
                                 truncate_gradient = self.truncation,
                                 sequences = [self.source, self.index],
                                 outputs_info = [self.act])
    self.output = self.output[::-(2 * self.reverse - 1)]
    
  def create_recurrent_weights(self, n, m):
    nin = n + m + m + m
    return self.create_random_weights(n, m, nin), self.create_random_weights(m, m, nin)
   
class LstmLayer(RecurrentLayer):
  def __init__(self, source, index, n_in, n_out, activation = T.nnet.sigmoid, reverse = False, truncation = -1, sharpgates = 'false' , dropout = 0, mask = "unity", name = "lstm"):
    super(LstmLayer, self).__init__(source, index, n_in, n_out * 4, activation, reverse, truncation, False, dropout, mask, name = name)
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
    self.n_out /= 4
    
    if sharpgates == 'global': self.sharpness = self.create_uniform_weights(3, n_out)
    else: self.sharpness = self.create_bias(3)
    self.sharpness.set_value(numpy.ones(self.sharpness.get_value().shape, dtype = theano.config.floatX))
    if sharpgates != 'none': self.add_param(self.sharpness)
    
    def step(x_t, i_t, s_p, h_p, mask):
      i = T.outer(i_t, self.o)
      z = T.dot(x_t, self.mass * mask * self.W_in) + T.dot(h_p, self.W_re) + self.b
      #z = T.dot(x_t, self.mass * mask * self.W_in) + T.max(abs(T.dot(h_p, self.W_re)), axis = 0) * T.sgn(T.dot(h_p, self.W_re)) + self.b
      #z = T.dot(x_t, self.mass * mask * self.W_in) + T.sum(T.dot(h_p, self.W_re), axis = 0) + self.b
      #z_in = T.max(0, T.dot(x_t, self.mass * mask * self.W_in) + self.b)
      #z_re = T.dot(h_p, self.W_re)
      #z = T.tanh(z_in) + T.tanh(z_re)
      partition = z.shape[1] / 4
      input = CI(z[:,:partition])
      ingate = GI(self.sharpness[0] * z[:,partition: 2 * partition])
      forgetgate = GF(self.sharpness[1] * z[:,2 * partition:3 * partition])
      s_t = input * ingate + s_p * forgetgate
      outgate = GO(self.sharpness[2] * z[:,3 * partition:4 * partition])
      h_t = CO(s_t) * outgate
      return s_t * i, h_t * i
    
    [state, self.output], _ = theano.scan(step,
                                          truncate_gradient = self.truncation,
                                          go_backwards = self.reverse,
                                          sequences = [self.source, self.index],
                                          non_sequences = [self.mask],
                                          outputs_info = [ T.alloc(self.state, self.source.shape[1], n_out),
                                                           T.alloc(self.act, self.source.shape[1], n_out), ])
    self.output = self.output[::-(2 * self.reverse - 1)]
  
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
        NETWORKS
"""
        
class LayerNetwork(object):
  def __init__(self, n_in, n_out, mask = "unity"):
    self.x = T.tensor3('x')
    self.y = T.imatrix('y')
    self.i = T.bmatrix('i')
    Layer.initialize()
    self.hidden_info = []
    self.n_in = n_in
    self.n_out = n_out
    self.mask = mask
  
  @classmethod
  def from_config(cls, config, mask = "unity"):
    num_inputs = netcdf_dimension(config.list('train')[0], 'inputPattSize') * config.int('window', 1)
    num_outputs = netcdf_dimension(config.list('train')[0], 'numLabels')
    hidden_size = config.int_list('hidden_size')
    assert len(hidden_size) > 0, "no hidden layers specified"
    hidden_type = config.list('hidden_type')
    assert len(hidden_type) <= len(hidden_size), "too many hidden layer types"
    if len(hidden_type) != len(hidden_size):
      n_hidden_type = len(hidden_type) 
      for i in xrange(len(hidden_size) - len(hidden_type)):
        if n_hidden_type == 1:
          hidden_type.append(hidden_type[0])
        else:  
          hidden_type.append("forward")
    task = config.value('task', 'train')
    L1_reg = config.float('L1_reg', 0.0)
    L2_reg = config.float('L2_reg', 0.0)
    bidirectional = config.bool('bidirectional', True)
    truncation = config.int('truncation', -1)
    actfct = config.list('activation')
    dropout = config.list('dropout', [0.0])
    sharpgates = config.value('sharpgates', 'none')
    loss = config.value('loss', 'loglik')
    entropy = config.float('entropy', 0.0)
    if len(actfct) < len(hidden_size):
      for i in xrange(len(hidden_size) - len(actfct)):
        actfct.append("logistic")
    if len(dropout) < len(hidden_size) + 1:
      for i in xrange(len(hidden_size) + 1 - len(dropout)):
        dropout.append(0.0)
    dropout = [float(d) for d in dropout]
    network = cls(num_inputs, num_outputs, mask)
    activations = { 'logistic' : T.nnet.sigmoid,
                    'tanh' : T.tanh,
                    'relu': lambda z : (T.sgn(z) + 1) * z * 0.5,
                    'identity' : lambda z : z,
                    'one' : lambda z : 1,
                    'zero' : lambda z : 0,
                    'softsign': lambda z : z / (1.0 + abs(z)),
                    'softsquare': lambda z : 1 / (1.0 + z * z),
                    'maxout': lambda z : T.max(z, axis = 0),
                    'sin' : T.sin,
                    'cos' : T.cos }
    for i in xrange(len(hidden_size)):
      if ':' in actfct[i]:
        acts = []
        for a in actfct[i].split(':'):
          assert activations.has_key(a), "invalid activation function: " + a
          acts.append(activations[a])
      else:
        assert activations.has_key(actfct[i]), "invalid activation function: " + actfct[i]
        acts = activations[actfct[i]]
      network.add_hidden(hidden_size[i], hidden_type[i], acts)
    if task == 'pretrain':
      loss = 'layer'
      task = 'train'
      network.add_hidden(config.int('pretrain_layer_size', hidden_size[-1]),
                         config.value('pretrain_layer_type', hidden_type[-1]),
                         config.value('pretrain_layer_activation', actfct[-1]))
    network.initialize(loss, L1_reg, L2_reg, dropout, bidirectional, truncation, sharpgates, entropy)
    return network
    
  def add_hidden(self, size, layer_type = 'forward', activation = T.tanh):
    self.hidden_info.append((layer_type, size, activation))
  
  def initialize(self, loss, L1_reg, L2_reg, dropout = 0, bidirectional = True, truncation = -1, sharpgates = 'none', entropy = 0):
    self.hidden = []
    self.reverse_hidden = []
    self.params = []
    n_in = self.n_in
    x_in = self.x
    L1 = T.constant(0)
    L2 = T.constant(0)
    self.recurrent = False
    # create forward layers
    for info, drop in zip(self.hidden_info, dropout[:-1]):
      if info[0] == 'forward':
        self.hidden.append(ForwardLayer(source = x_in, n_in = n_in, n_out = info[1], activation = info[2], dropout = drop, mask = self.mask))
      else:
        self.recurrent = True
        if info[0] == 'recurrent':
          self.hidden.append(RecurrentLayer(source = x_in, index = self.i, n_in = n_in, n_out = info[1], activation = info[2], truncation = truncation, dropout = drop, mask = self.mask))
        elif info[0] == 'lstm':
          self.hidden.append(LstmLayer(source = x_in, index = self.i, n_in = n_in, n_out = info[1], activation = info[2], truncation = truncation, dropout = drop, mask = self.mask))
        elif info[0] == 'peep_lstm':
          self.hidden.append(LstmPeepholeLayer(source = x_in, index = self.i, n_in = n_in, n_out = info[1], activation = info[2], truncation = truncation, dropout = drop, mask = self.mask))
        else: assert False, "invalid layer type: " + info[0]
        L1 += abs(self.hidden[-1].W_re.sum())
        L2 += (self.hidden[-1].W_re ** 2).sum()
      L1 += abs(self.hidden[-1].W_in.sum())
      L2 += (self.hidden[-1].W_in ** 2).sum()
      n_in = info[1]
      x_in = self.hidden[-1].output
      self.params = self.params + self.hidden[-1].params
    # create backward layers
    self.bidirectional = bidirectional and self.recurrent
    if self.bidirectional:
      n_in = self.n_in
      x_in = self.x
      for info, drop in zip(self.hidden_info, dropout[:-1]):
        if info[0] == 'forward':
          self.reverse_hidden.append(ForwardLayer(source = x_in, n_in = n_in, n_out = info[1], activation = info[2], dropout = drop, mask = self.mask))
        else:
          if info[0] == 'recurrent':
            self.reverse_hidden.append(RecurrentLayer(rng = self.rng, source = x_in, index = self.i, n_in = n_in, n_out = info[1], activation = info[2], reverse = True, truncation = truncation, dropout = drop, mask = self.mask))
          elif info[0] == 'lstm':
            self.reverse_hidden.append(LstmLayer(source = x_in, index = self.i, n_in = n_in, n_out = info[1], activation = info[2], reverse = True, truncation = truncation, dropout = drop, mask = self.mask))
          elif info[0] == 'peep_lstm':
            self.reverse_hidden.append(LstmPeepholeLayer(source = x_in, index = self.i, n_in = n_in, n_out = info[1], activation = info[2], reverse = True, truncation = truncation, dropout = drop, mask = self.mask))
          else: assert False, "invalid layer type: " + info[0]
          L1 += abs(self.reverse_hidden[-1].W_re.sum())
          L2 += (self.reverse_hidden[-1].W_re ** 2).sum()
        L1 += abs(self.reverse_hidden[-1].W_in.sum())
        L2 += (self.reverse_hidden[-1].W_in ** 2).sum()
        n_in = info[1]
        x_in = self.reverse_hidden[-1].output
        self.params = self.params + self.reverse_hidden[-1].params
    self.gparams = self.params[:]
    # create output layer
    self.loss = loss
    if loss == 'loglik':
      if self.bidirectional:
        self.output = BidirectionalSoftmaxLayer(forward = self.hidden[-1].output, backward = self.reverse_hidden[-1].output, index = self.i, n_in = n_in, n_out = self.n_out, dropout = dropout[-1], mask = self.mask)
        #self.output = BidirectionalContextSoftmaxLayer(rng = self.rng, forward = self.hidden[-1].output, backward = self.reverse_hidden[-1].output, index = self.i, n_in = n_in, n_out = self.n_out, n_ctx = self.n_out)
        #self.output = BidirectionalStaticContextSoftmaxLayer(rng = self.rng, forward = self.hidden[-1].output, backward = self.reverse_hidden[-1].output, index = self.i, n_in = n_in, n_out = self.n_out, n_ctx = self.n_out)
        #self.output = RecurrentBidirectionalSoftmaxLayer(rng = self.rng, forward = self.hidden[-1].output, backward = self.reverse_hidden[-1].output, index = self.i, n_in = n_in, n_out = self.n_out)
        #self.output = BidirectionalLstmSoftmaxLayer(rng = self.rng, forward = self.hidden[-1].output, backward = self.reverse_hidden[-1].output, index = self.i, n_in = n_in, n_out = self.n_out)
      else:
        self.output = SoftmaxLayer(source = x_in, index = self.i, n_in = n_in, n_out = self.n_out, dropout = dropout[-1], mask = self.mask)
    elif loss == 'layer':
      if self.bidirectional:
        self.output = BidirectionalSoftmaxLayer(forward = self.hidden[-1].output, backward = self.reverse_hidden[-1].output, index = self.i, n_in = n_in, n_out = self.n_out, dropout = dropout[-1], mask = self.mask)
      else:
        self.output = SoftmaxLayer(source = x_in, index = self.i, n_in = n_in, n_out = self.n_out, dropout = dropout[-1], mask = self.mask)
      self.gparams = self.hidden[-1].params + (self.reverse_hidden[-1].params if self.bidirectional else [])
    else: assert False, "invalid loss: " + loss
    L1 += abs(self.output.W_in.sum())
    L2 += (self.output.W_in ** 2).sum()
    if self.bidirectional:
      L1 += abs(self.output.W_reverse.sum())
      L2 += (self.output.W_reverse ** 2).sum()
    self.params += self.output.params
    self.gparams += self.output.params
    self.cost = self.output.cost(self.y)
    self.objective = self.cost + L1_reg * L1 + L2_reg * L2 + entropy * self.output.entropy()
    self.errors = self.output.errors(self.y)
    #self.jacobian = T.jacobian(self.output.z, self.x)
  
  def num_params(self):
    return sum([h.num_params() for h in self.hidden]) + sum([h.num_params() for h in self.reverse_hidden]) + self.output.num_params()
  
  def get_params(self):
    params = [self.output.get_params()]
    for h in self.hidden + self.reverse_hidden:
      params.append(h.get_params())
    return params
  
  def set_params(self, params):
    self.output.set_params(params[0])
    for p,h in zip(params[1:], self.hidden + self.reverse_hidden):
      h.set_params(p)
  
  def save(self, name, epoch):
    model = open(name, 'w')
    print >> model, epoch, self.bidirectional
    if self.bidirectional:
      for g,h in zip(self.hidden, self.reverse_hidden):
        g.save(model)
        h.save(model)
    else:
      for h in self.hidden:
        h.save(model)
    self.output.save(model)
    model.close()
  
  def load(self, name):
    model = [ line.strip() for line in open(name, 'r').readlines() if len(line.strip().split()) > 0 ]
    desc = model[0].strip().split()
    epoch = int(desc[0])
    self.bidirectional = (len(desc) > 1) and (desc[1].lower() == "true")
    model = model[1:]
    num_hidden = len(self.hidden) - (self.loss == 'layer') 
    assert len(model) == self.bidirectional * num_hidden + num_hidden + 1
    for i in xrange(num_hidden):
      self.hidden[i].load(model[i + self.bidirectional * i].strip())
      if self.bidirectional:
        self.reverse_hidden[i].load(model[2 * i + 1].strip())
    self.output.load(model[-1].strip())
    return epoch
