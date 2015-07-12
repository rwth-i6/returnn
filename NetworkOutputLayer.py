
import numpy
from theano import tensor as T
import theano
from BestPathDecoder import BestPathDecodeOp
from CTC import CTCOp
from NetworkLayer import Layer
from SprintErrorSignals import SprintErrorSigOp
from NetworkRecurrentLayer import RecurrentLayer

class OutputLayer(Layer):
  def __init__(self, index, loss, y, **kwargs):
    """
    :param theano.Variable index: index for batches
    :param str loss: e.g. 'ce'
    """
    kwargs.setdefault("layer_class", "softmax")
    super(OutputLayer, self).__init__(**kwargs)
    self.z = self.b
    self.y = y
    self.W_in = [self.add_param(self.create_forward_weights(source.attrs['n_out'], self.attrs['n_out'],
                                                            name="W_in_%s_%s" % (source.name, self.name)),
                                "W_in_%s_%s" % (source.name, self.name))
                 for source in self.sources]
    assert len(self.sources) == len(self.masks) == len(self.W_in)
    for source, m, W in zip(self.sources, self.masks, self.W_in):
      if m is None:
        self.z += T.dot(source.output, W)
      else:
        self.z += T.dot(self.mass * m * source.output, W)
    self.set_attr('from', ",".join([s.name for s in self.sources]))
    self.index = index
    self.i = (index.flatten() > 0).nonzero()
    self.loss = loss.encode("utf8")
    self.attrs['loss'] = self.loss
    if self.loss == 'priori':
      self.priori = theano.shared(value=numpy.ones((self.attrs['n_out'],), dtype=theano.config.floatX), borrow=True)
    self.output = self.z

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

  def errors(self):
    """
    :type y: theano.Variable
    :rtype: theano.Variable
    """
    if self.y.dtype.startswith('int'):
      if self.y.type == T.ivector().type:
        return T.sum(T.neq(self.y_pred[self.i], self.y[self.i]))
      else:
        return T.sum(T.neq(self.y_pred[self.i], T.argmax(self.y[self.i], axis = -1)))
    else:
      raise NotImplementedError()


class FramewiseOutputLayer(OutputLayer):
  def __init__(self, **kwargs):
    super(FramewiseOutputLayer, self).__init__(**kwargs)
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
    self.output = self.p_y_given_x

  def cost(self):
    known_grads = None
    if self.loss == 'ce' or self.loss == 'priori':
      if self.y.type == T.ivector().type:
        logpcx, pcx = T.nnet.crossentropy_softmax_1hot(x=self.y_m[self.i], y_idx=self.y[self.i]) #T.log(self.p_y_given_x[self.i, y[self.i]])
        #pcx = T.log(T.clip(pcx, 1.e-20, 1.e20))  # For pcx near zero, the gradient will likely explode.
      else:
        logpcx = -T.dot(T.log(T.clip(self.p_y_given_x[self.i], 1.e-20, 1.e20)), self.y[self.i].T)
      #pcx = self.p_y_given_x[:, y[self.i]]
      return T.sum(logpcx), known_grads
    elif self.loss == 'sse':
      y_f = T.cast(T.reshape(self.y, (self.y.shape[0] * self.y.shape[1]), ndim=1), 'int32')
      y_oh = T.eq(T.shape_padleft(T.arange(self.attrs['n_out']), y_f.ndim), T.shape_padright(y_f, 1))
      return T.mean(T.sqr(self.p_y_given_x[self.i] - y_oh[self.i])), known_grads
    else:
      assert False, "unknown loss: %s" % self.loss


class SequenceOutputLayer(OutputLayer):
  def __init__(self, prior_scale=0.0, log_prior=None, ce_smoothing=0.0, **kwargs):
    super(SequenceOutputLayer, self).__init__(**kwargs)
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
    y_f = T.cast(T.reshape(self.y, (self.y.shape[0] * self.y.shape[1]), ndim = 1), 'int32')
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

  def errors(self):
    if self.loss in ('ctc', 'ce_ctc'):
      return T.sum(BestPathDecodeOp()(self.p_y_given_x, self.y, T.sum(self.index, axis=0)))
    else:
      return super(SequenceOutputLayer, self).errors(self.y)

class LstmOutputLayer(RecurrentLayer):
  def __init__(self, n_out, n_units, y, sharpgates='none', encoder = None, loss = 'cedec', loop = 'hard', n_dec = 0, **kwargs):
    kwargs.setdefault("layer_class", "lstm_softmax")
    kwargs.setdefault("activation", "sigmoid")
    kwargs["compile"] = False
    kwargs["n_out"] = n_units * 4
    super(LstmOutputLayer, self).__init__(**kwargs)
    self.set_attr('loss', loss.encode("utf8"))
    self.set_attr('n_out', n_out)
    self.set_attr('n_units', n_units)
    if loop == True:
      loop = 'hard' if self.train_flag else 'soft'
    self.set_attr('loop', loop)
    if n_dec: self.set_attr('n_dec', n_dec)
    self.y = y
    if not y: loop = False
    if encoder:
      self.set_attr('encoder', encoder.name)
    projection = kwargs.get("projection", None)
    if not isinstance(self.activation, (list, tuple)):
      self.activation = [T.tanh, T.nnet.sigmoid, T.nnet.sigmoid, T.nnet.sigmoid, T.tanh]
    else:
      assert len(self.activation) == 5, "lstm activations have to be specified as 5 tuple (input, ingate, forgetgate, outgate, output)"
    self.set_attr('sharpgates', sharpgates)
    CI, GI, GF, GO, CO = self.activation #T.tanh, T.nnet.sigmoid, T.nnet.sigmoid, T.nnet.sigmoid, T.tanh
    n_in = sum([s.attrs['n_out'] for s in self.sources])
    n_re = projection if projection is not None else n_units
    #self.state = self.create_bias(n_out, 'state')
    #self.act = self.create_bias(n_re, 'act')
    self.b.set_value(numpy.zeros((n_units * 3 + n_re,), dtype = theano.config.floatX))
    if projection:
      W_proj = self.create_random_uniform_weights(n_units, n_re, n_in + n_units + n_re, name="W_proj_%s" % self.name)
      self.W_proj.set_value(W_proj.get_value())
    W_re = self.create_random_uniform_weights(n_re, n_units * 3 + n_re, n_in + n_re + n_units * 3 + n_re,
                                              name="W_re_%s" % self.name)
    self.W_re.set_value(W_re.get_value())
    for s, W in zip(self.sources, self.W_in):
      W.set_value(self.create_random_uniform_weights(s.attrs['n_out'], n_units * 3 + n_re,
                                                     s.attrs['n_out'] + n_units + n_units * 3 + n_re,
                                                     name="W_in_%s_%s" % (s.name, self.name)).get_value(borrow=True, return_internal_type=True), borrow = True)

    self.W_cls = self.add_param(self.create_forward_weights(self.attrs['n_units'], self.attrs['n_out'],
                                                            name="W_cls_%s_%s" % (self.name, self.name)),
                                "W_cls_%s_%s" % (self.name, self.name))
    self.W_rec = self.add_param(self.create_random_uniform_weights(self.attrs['n_out'], self.attrs['n_units'] * 3 + n_re, n_in + n_re + self.attrs['n_units'] * 3 + n_re,
                                              name="W_rec_%s" % self.name),
                                "W_rec_%s" % (self.name))

    self.o.set_value(numpy.ones((n_units,), dtype='int8'))
    self.sharpness = theano.shared(value = numpy.zeros((3,), dtype=theano.config.floatX), borrow=True, name='lambda')
    self.sharpness.set_value(numpy.ones(self.sharpness.get_value().shape, dtype = theano.config.floatX))
    if sharpgates != 'none' and sharpgates != "shared" and sharpgates != "single":
      self.add_param(self.sharpness, 'gate_scaling')

    assert self.attrs['optimization'] in ['memory', 'speed']
    if self.attrs['optimization'] == 'speed':
      z = self.b
      for x_t, m, W in zip(self.sources, self.masks, self.W_in):
        if x_t.attrs['sparse']:
          z += W[T.cast(x_t.output[:,:,0], 'int32')]
        elif m is None:
          z += T.dot(x_t.output, W)
        else:
          z += T.dot(self.mass * m * x_t.output, W)

    def step(z, i_t, s_p, h_p):
      z += T.dot(h_p, self.W_re)
      if self.attrs['optimization'] == 'memory':
        z += self.b
        for x_t, m, W in zip(self.sources, self.masks, self.W_in):
          if x_t.attrs['sparse']:
            z += W[T.cast(x_t.output[:,:,0], 'int32')]
          elif m is None:
            z += T.dot(x_t.output, W)
          else:
            z += T.dot(self.mass * m * x_t.output, W)
      #z += T.dot(T.nnet.softmax(T.dot(h_p, self.W_cls)), self.W_rec) + T.dot(h_p, self.W_re)
      if self.attrs['loop'] == 'soft' or (self.attrs['loop'] != 'none' and not self.train_flag):
        z += self.W_rec[T.argmax(T.dot(h_p, self.W_cls), axis = -1)]
      i = T.outer(i_t, T.alloc(numpy.cast['int8'](1), n_units))
      j = i if not self.W_proj else T.outer(i_t, T.alloc(numpy.cast['int8'](1), n_re))
      ingate = GI(z[:,n_units: 2 * n_units])
      forgetgate = GF(z[:,2 * n_units:3 * n_units])
      outgate = GO(z[:,3 * n_units:])
      input = CI(z[:,:n_units])
      s_i = input * ingate + s_p * forgetgate
      s_t = s_i if not self.W_proj else T.dot(s_i, self.W_proj)
      h_t = CO(s_t) * outgate
      return theano.gradient.grad_clip(s_i * i, -50, 50), h_t * j

    def nstep(z_batch, i_t, s_batch, h_batch):
      #t_t = T.switch(T.eq(T.sum(i_t), 0), i_t + 1, i_t)
      #j_t = (t_t > 0).nonzero()
      j_t = (i_t.flatten() > 0).nonzero()
      #j_t = i_t
      z = z_batch[j_t]
      s_p = s_batch[j_t]
      h_p = h_batch[j_t]

      z += T.dot(h_p, self.W_re)
      #z += T.dot(T.nnet.softmax(T.dot(h_p, self.W_cls)), self.W_rec) + T.dot(h_p, self.W_re)
      if self.attrs['loop'] == 'soft' or (self.attrs['loop'] != 'none' and not self.train_flag):
        z += self.W_rec[T.argmax(T.dot(h_p, self.W_cls), axis = -1)]
      ingate = GI(z[:,n_units: 2 * n_units])
      forgetgate = GF(z[:,2 * n_units:3 * n_units])
      outgate = GO(z[:,3 * n_units:])
      input = CI(z[:,:n_units])
      s_i = input * ingate + s_p * forgetgate
      s_t = s_i if not self.W_proj else T.dot(s_i, self.W_proj)
      h_t = CO(s_t) * outgate
      #s_q = T.zeros_like(s_p, dtype = theano.config.floatX)
      #s_out = T.inc_subtensor(s_q[j_t], s_i)
      #h_q = T.zeros_like(h_p, dtype = theano.config.floatX)
      #h_out = T.inc_subtensor(h_q[j_t], h_t)
      s_out = T.set_subtensor(s_p[j_t], s_i)
      h_out = T.set_subtensor(h_p[j_t], h_t)
      return theano.gradient.grad_clip(s_out, -50, 50), h_out

    self.out_dec = self.index.shape[0] #encoder.output.shape[0] if encoder else self.sources[0].output.shape[0]
    if encoder and 'n_dec' in encoder.attrs:
      self.out_dec = encoder.out_dec
    for s in xrange(self.attrs['sampling']):
      index = self.index[s::self.attrs['sampling']]
      sequences = z
      if encoder:
        n_dec = self.out_dec
        if 'n_dec' in self.attrs:
          n_dec = self.attrs['n_dec']
          index = T.alloc(numpy.cast[numpy.int8](1), n_dec, encoder.index.shape[1])
        outputs_info = [ encoder.state[-1],
                         encoder.act[-1] ]
        sequences = T.alloc(numpy.cast[theano.config.floatX](0), n_dec, encoder.output.shape[1], n_units * 3 + n_re) + self.b
        if self.attrs['loop'] == 'hard' and self.train_flag:
          sequences = T.inc_subtensor(sequences[1:], self.W_rec[self.y.reshape((n_dec, encoder.output.shape[1]), ndim=2)][:-1])
      else:
        outputs_info = [ T.alloc(numpy.cast[theano.config.floatX](0), self.sources[0].output.shape[1], n_units),
                         T.alloc(numpy.cast[theano.config.floatX](0), self.sources[0].output.shape[1], n_re) ]
      [state, act], _ = theano.scan(step,
                                    name = "scan_%s"%self.name,
                                    truncate_gradient = self.attrs['truncation'],
                                    go_backwards = self.attrs['reverse'],
                                    sequences = [ sequences[s::self.attrs['sampling']], index ],
                                    outputs_info = outputs_info)
      if self.attrs['sampling'] > 1: # time batch dim
        if s == 0:
          totact = T.repeat(act, self.attrs['sampling'], axis = 0)[:n_dec]
        else:
          totact = T.set_subtensor(totact[s::self.attrs['sampling']], act)
      else:
        totact = act
    self.state = state
    self.act = totact[::-(2 * self.attrs['reverse'] - 1)]
    self.lstm_output = totact[::-(2 * self.attrs['reverse'] - 1)]

    self.y_m = T.dot(self.lstm_output, self.W_cls).dimshuffle(2,0,1).flatten(ndim = 2).dimshuffle(1,0)
    self.y_pred = T.argmax(self.y_m, axis=-1)
    self.output = T.argmax(self.lstm_output, axis=-1, keepdims=True) #T.argmax(self.y_m, axis=-1, keepdims=True).dimshuffle(1,0).reshape(self.lstm_output.shape, ndim = 3).dimshuffle(1,2,0) #.argmax(self.p_y_given_x, axis=-1, keepdim)
    self.attrs['sparse'] = True
    self.j = (self.index.flatten() > 0).nonzero()


  def cost(self):
    known_grads = None
    if self.attrs['loss'] == 'cedec' or self.attrs['loss'] == 'priori':
      if self.y.type == T.ivector().type:
        logpcx, pcx = T.nnet.crossentropy_softmax_1hot(x=self.y_m[self.j], y_idx=self.y[self.j])
        #logpcx = T.nnet.crossentropy_softmax_1hot(x=self.y_m[self.j], y_idx=y[self.j]) #T.log(self.p_y_given_x[self.i, y[self.i]])
        #pcx = T.log(T.clip(pcx, 1.e-20, 1.e20))  # For pcx near zero, the gradient will likely explode.
      else:
        logpcx = -T.dot(T.log(T.clip(self.p_y_given_x[self.j], 1.e-20, 1.e20)), self.y[self.j].T)
      #pcx = self.p_y_given_x[:, y[self.i]]
      return T.sum(logpcx), known_grads
    else:
      assert False, "unknown loss: %s" % self.attrs['loss']

  def errors(self):
    """
    :type y: theano.Variable
    :rtype: theano.Variable
    """
    if self.y.dtype.startswith('int'):
      if self.y.type == T.ivector().type:
        return T.sum(T.neq(self.y_pred[self.j], self.y[self.j]))
      else:
        return T.sum(T.neq(self.y_pred[self.j], T.argmax(self.y[self.j], axis = -1)))
    else:
      raise NotImplementedError()
