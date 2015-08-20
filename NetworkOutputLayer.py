
import numpy
from theano import tensor as T
import theano
from BestPathDecoder import BestPathDecodeOp
from CTC import CTCOp
from NetworkBaseLayer import Layer
from SprintErrorSignals import SprintErrorSigOp
from NetworkRecurrentLayer import RecurrentLayer


#from Accumulator import AccumulatorOpInstance

#def step(*args): # requires same amount of memory
#  xs = args[:(len(args)-1)/2]
#  ws = args[(len(args)-1)/2:-1]
#  b = args[-1]
#  out = b
#  for w,x in zip(ws,xs):
#    out += T.dot(x,w)
#  return out

class OutputLayer(Layer):
  layer_class = "softmax"

  def __init__(self, loss, y, copy_input=None, **kwargs):
    """
    :param theano.Variable index: index for batches
    :param str loss: e.g. 'ce'
    """
    super(OutputLayer, self).__init__(**kwargs)
    self.y = y
    if copy_input:
      self.set_attr("copy_input", copy_input.name)
    if not copy_input:
      self.z = self.b
      self.W_in = [self.add_param(self.create_forward_weights(source.attrs['n_out'], self.attrs['n_out'],
                                                              name="W_in_%s_%s" % (source.name, self.name)))
                   for source in self.sources]

      assert len(self.sources) == len(self.masks) == len(self.W_in)
      assert len(self.sources) > 0
      for source, m, W in zip(self.sources, self.masks, self.W_in):
        if source.attrs['sparse']:
          self.z += W[T.cast(source.output[:,:,0], 'int32')]
        elif m is None:
          self.z += self.dot(source.output, W)
        else:
          self.z += self.dot(self.mass * m * source.output, W)
    else:
      self.z = copy_input.output
    assert self.z.ndim == 3

    #xs = [s.output for s in self.sources]
    #self.z = AccumulatorOpInstance(*[self.b] + xs + self.W_in)
    #outputs_info = None #[ T.alloc(numpy.cast[theano.config.floatX](0), index.shape[1], self.attrs['n_out']) ]

    #self.z, _ = theano.scan(step,
    #                        sequences = [s.output for s in self.sources],
    #                        non_sequences = self.W_in + [self.b])

    self.set_attr('from', ",".join([s.name for s in self.sources]))
    if self.y.dtype.startswith('int'):
      self.i = (self.index.flatten() > 0).nonzero()
    elif self.y.dtype.startswith('float'):
      self.i = (self.index.dimshuffle(0,1,'x').repeat(self.z.shape[2],axis=2).flatten() > 0).nonzero()
    self.j = ((T.constant(1.0) - self.index.flatten()) > 0).nonzero()
    self.loss = loss.encode("utf8")
    self.attrs['loss'] = self.loss
    if self.loss == 'priori':
      self.priori = theano.shared(value=numpy.ones((self.attrs['n_out'],), dtype=theano.config.floatX), borrow=True)
    #self.make_output(self.z, collapse = False)
    self.output = self.make_consensus(self.z) if self.depth > 1 else self.z

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
        return T.sum(T.neq(T.argmax(self.y_m[self.i], axis=-1), self.y[self.i]))
      else:
        return T.sum(T.neq(T.argmax(self.y_m[self.i], axis=-1), T.argmax(self.y[self.i], axis = -1)))
    elif self.y.dtype.startswith('float'):
      return T.sum(T.sqr(self.y_m[self.i] - self.y.flatten()[self.i]))
      #return T.sum(T.sum(T.sqr(self.y_m - self.y.reshape(self.y_m.shape)), axis=1)[self.i])
      #return T.sum(T.sqr(self.y_m[self.i] - self.y.reshape(self.y_m.shape)[self.i]))
      #return T.sum(T.sum(T.sqr(self.z - (self.y.reshape((self.index.shape[0], self.index.shape[1], self.attrs['n_out']))[:self.z.shape[0]])), axis=2).flatten()[self.i])
      #return T.sum(T.sqr(self.y_m[self.i] - (self.y.reshape((self.index.shape[0], self.index.shape[1], self.attrs['n_out']))[:self.z.shape[0]]).reshape(self.y_m.shape)[self.i]))
      #return T.sum(T.sqr(self.y_m[self.i] - self.y.reshape(self.y_m.shape)[self.i]))
    else:
      raise NotImplementedError()

class FramewiseOutputLayer(OutputLayer):
  def __init__(self, **kwargs):
    super(FramewiseOutputLayer, self).__init__(**kwargs)
    self.initialize()

  def initialize(self):
    #self.y_m = self.output.dimshuffle(2,0,1).flatten(ndim = 2).dimshuffle(1,0)
    nreps = T.switch(T.eq(self.output.shape[0], 1), self.index.shape[0], 1)
    output = self.output.repeat(nreps,axis=0)
    self.y_m = output.flatten() if self.y.dtype.startswith('float') else output.reshape((output.shape[0]*output.shape[1],output.shape[2]))
    if self.loss == 'ce' or self.loss == 'entropy': self.p_y_given_x = T.nnet.softmax(self.y_m) # - self.y_m.max(axis = 1, keepdims = True))
    #if self.loss == 'ce':
    #  y_mmax = self.y_m.max(axis = 1, keepdims = True)
    #  y_mmin = self.y_m.min(axis = 1, keepdims = True)
    #  self.p_y_given_x = T.nnet.softmax(self.y_m - (0.5 * (y_mmax - y_mmin) + y_mmin))
    elif self.loss == 'sse': self.p_y_given_x = self.y_m
    elif self.loss == 'priori': self.p_y_given_x = T.nnet.softmax(self.y_m) / self.priori
    else: assert False, "invalid loss: " + self.loss
    self.y_pred = T.argmax(self.y_m[self.i], axis=-1, keepdims=True)
    self.output = self.p_y_given_x

  def cost(self):
    known_grads = None
    if self.loss == 'ce' or self.loss == 'priori':
      if self.y.type == T.ivector().type:
        # Use crossentropy_softmax_1hot to have a more stable and more optimized gradient calculation.
        # Theano fails to use it automatically; I guess our self.i indexing is too confusing.
        nll, pcx = T.nnet.crossentropy_softmax_1hot(x=self.y_m[self.i], y_idx=self.y[self.i])
        #nll, pcx = T.nnet.crossentropy_softmax_1hot(x=self.y_m, y_idx=self.y)
        #nll = T.set_subtensor(nll[self.j], T.constant(0.0))
      else:
        nll = -T.dot(T.log(T.clip(self.p_y_given_x[self.i], 1.e-38, 1.e20)), self.y[self.i].T)
      return T.sum(nll), known_grads
    elif self.loss == 'entropy':
      h_e = T.exp(self.y_m) #(TB)
      pcx = T.clip((h_e / T.sum(h_e, axis=1, keepdims=True)).reshape((self.index.shape[0],self.index.shape[1],self.attrs['n_out'])), 1.e-6, 1.e6) # TBD
      ee = self.index * -T.sum(pcx * T.log(pcx)) # TB
      #nll, pcxs = T.nnet.crossentropy_softmax_1hot(x=self.y_m[self.i], y_idx=self.y[self.i])
      nll, _ = T.nnet.crossentropy_softmax_1hot(x=self.y_m, y_idx=self.y) # TB
      ce = nll.reshape(self.index.shape) * self.index # TB
      y = self.y.reshape(self.index.shape) * self.index # TB
      f = T.any(T.gt(y,0), axis=0) # B
      return T.sum(f * T.sum(ce, axis=0) + (1-f) * T.sum(ee, axis=0)), known_grads
      #return T.sum(T.switch(T.gt(T.sum(y,axis=0),0), T.sum(ce, axis=0), -T.sum(ee, axis=0))), known_grads
      #return T.switch(T.gt(T.sum(self.y_m[self.i]),0), T.sum(nll), -T.sum(pcx * T.log(pcx))), known_grads
    elif self.loss == 'priori':
      pcx = self.p_y_given_x[self.i, self.y[self.i]]
      pcx = T.clip(pcx, 1.e-38, 1.e20)  # For pcx near zero, the gradient will likely explode.
      return -T.sum(T.log(pcx)), known_grads
    elif self.loss == 'sse':
      if self.y.dtype.startswith('int'):
        y_f = T.cast(T.reshape(self.y, (self.y.shape[0] * self.y.shape[1]), ndim=1), 'int32')
        y_oh = T.eq(T.shape_padleft(T.arange(self.attrs['n_out']), y_f.ndim), T.shape_padright(y_f, 1))
        return T.mean(T.sqr(self.p_y_given_x[self.i] - y_oh[self.i])), known_grads
      else:
        #return T.sum(T.sum(T.sqr(self.y_m - self.y.reshape(self.y_m.shape)), axis=1)[self.i]), known_grads
        return T.sum(T.sqr(self.y_m[self.i] - self.y.flatten()[self.i])), known_grads
        #return T.sum(T.sum(T.sqr(self.z - (self.y.reshape((self.index.shape[0], self.index.shape[1], self.attrs['n_out']))[:self.z.shape[0]])), axis=2).flatten()[self.i]), known_grads
        #y_z = T.set_subtensor(T.zeros((self.index.shape[0],self.index.shape[1],self.attrs['n_out']), dtype='float32')[:self.z.shape[0]], self.z).flatten()
        #return T.sum(T.sqr(y_z[self.i] - self.y[self.i])), known_grads
        #return T.sum(T.sqr(self.y_m - self.y[:self.z.shape[0]*self.index.shape[1]]).flatten()[self.i]), known_grads
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
    assert self.loss in ('ctc', 'ce_ctc', 'ctc2', 'sprint', 'sprint_smoothed'), 'invalid loss: ' + self.loss
    self.y_m = T.reshape(self.z, (self.z.shape[0] * self.z.shape[1], self.z.shape[2]), ndim = 2)
    p_y_given_x = T.nnet.softmax(self.y_m)
    self.y_pred = T.argmax(p_y_given_x, axis = -1)
    self.p_y_given_x = T.reshape(T.nnet.softmax(self.y_m), self.z.shape)

  def cost(self, y):
    """
    :param y: shape (time*batch,) -> label
    :return: error scalar, known_grads dict
    """
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
    elif self.loss == 'ctc2':
      from NetworkCtcLayer import ctc_cost, uniq_with_lengths, log_sum
      max_time = self.z.shape[0]
      num_batches = self.z.shape[1]
      time_mask = self.index.reshape((max_time, num_batches))
      y_batches = y.reshape((max_time, num_batches))
      targets, seq_lens = uniq_with_lengths(y_batches, time_mask)
      log_pcx = self.z - log_sum(self.z, axis=0, keepdims=True)
      err = ctc_cost(log_pcx, time_mask, targets, seq_lens)
      return err, known_grads

  def errors(self):
    if self.loss in ('ctc', 'ce_ctc'):
      return T.sum(BestPathDecodeOp()(self.p_y_given_x, self.y, T.sum(self.index, axis=0)))
    else:
      return super(SequenceOutputLayer, self).errors(self.y)

class LstmOutputLayer(RecurrentLayer):
  def __init__(self, n_out, n_units, y, sharpgates='none', encoder = None, loss = 'cedec', loop = -1, n_dec = 0, n_proto = 0, **kwargs):
    kwargs.setdefault("layer_class", "lstm_softmax")
    kwargs.setdefault("activation", "sigmoid")
    kwargs["compile"] = False
    kwargs["n_out"] = n_units * 4
    super(LstmOutputLayer, self).__init__(**kwargs)
    self.set_attr('loss', loss.encode("utf8"))
    self.set_attr('n_out', n_out)
    self.set_attr('n_units', n_units)
    self.set_attr('n_classes', n_out)
    self.set_attr('n_proto', n_proto)
    #if loop == True:
    #  loop = 'hard' if self.train_flag else 'soft'
    self.set_attr('loop', loop)
    if n_dec: self.set_attr('n_dec', n_dec)
    self.y = y
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
    n_re = n_units
    #self.state = self.create_bias(n_out, 'state')
    #self.act = self.create_bias(n_re, 'act')
    self.loop = loop
    self.b.set_value(numpy.zeros((n_units * 3 + n_re,), dtype = theano.config.floatX))
    if projection:
      n_re = projection
      W_proj = self.create_random_uniform_weights(n_units, projection, projection + n_units, name="W_proj_%s" % self.name)
      self.W_proj.set_value(W_proj.get_value())
    W_re = self.create_random_uniform_weights(n_re, n_units * 4, n_in + n_units * 4,
                                              name="W_re_%s" % self.name)
    self.W_re.set_value(W_re.get_value())
    for s, W in zip(self.sources, self.W_in):
      W.set_value(self.create_random_uniform_weights(s.attrs['n_out'], n_units * 4,
                                                     s.attrs['n_out'] + n_units + n_units * 4,
                                                     name="W_in_%s_%s" % (s.name, self.name)).get_value(borrow=True, return_internal_type=True), borrow = True)

    if self.attrs['n_proto']:
      self.W_pro = self.add_param(self.create_random_uniform_weights(self.attrs['n_units'], self.attrs['n_proto'], n_units + self.attrs['n_proto'], name="W_cls_%s_%s" % (self.name, self.name)),
                                  "W_pro_%s_%s" % (self.name, self.name))
      self.W_cls = self.add_param(self.create_random_uniform_weights(self.attrs['n_proto'], self.attrs['n_classes'], self.attrs['n_proto'] + self.attrs['n_classes'], name="W_cls_%s_%s" % (self.name, self.name)),
                                  "W_cls_%s_%s" % (self.name, self.name))
    else:
      self.W_cls = self.add_param(self.create_random_uniform_weights(self.attrs['n_units'], self.attrs['n_classes'], self.attrs['n_units'] + self.attrs['n_classes'], name="W_cls_%s_%s" % (self.name, self.name)),
                                  "W_cls_%s_%s" % (self.name, self.name))

    if self.attrs['loop'] != 0:
      if self.attrs['loop'] != -1:
        self.W_loop = self.add_param(self.create_random_uniform_weights(loop, self.attrs['n_units'] * 4, self.attrs['n_units'] * 4 + loop + n_in, name="W_loop_%s" % self.name))
      else:
        loop = self.attrs['n_units'] * 4
      self.W_rec = self.add_param(self.create_random_uniform_weights(self.attrs['n_classes'], loop, loop + n_in, name="W_rec_%s" % self.name), "W_rec_%s" % (self.name))

    self.o.set_value(numpy.ones((n_units,), dtype='int8'))
    self.sharpness = theano.shared(value = numpy.zeros((3,), dtype=theano.config.floatX), borrow=True, name='lambda')
    self.sharpness.set_value(numpy.ones(self.sharpness.get_value().shape, dtype = theano.config.floatX))
    if sharpgates != 'none' and sharpgates != "shared" and sharpgates != "single":
      self.add_param(self.sharpness, 'gate_scaling')

    z = self.b
    for x_t, m, W in zip(self.sources, self.masks, self.W_in):
      if x_t.attrs['sparse']:
        z += W[T.cast(x_t.output[:,:,0], 'int32')]
      elif m is None:
        z += T.dot(x_t.output, W)
      else:
        z += T.dot(self.mass * m * x_t.output, W)

    def index_step(z_batch, i_t, s_batch, h_batch):
      q_t = i_t #T.switch(T.any(i_t), i_t, T.ones_like(i_t))
      j_t = (q_t > 0).nonzero()
      s_p = s_batch[j_t]
      h_p = h_batch[j_t]
      z = z_batch[j_t]
      z += T.dot(h_p, self.W_re)
      if self.attrs['loop'] == 'soft' or (self.attrs['loop'] != 'none' and not self.train_flag):
        z += self.W_rec[T.argmax(T.dot(h_p if not projection else T.dot(self.W_proj, h_p), self.W_cls), axis = -1)]
      ingate = GI(z[:,n_units: 2 * n_units])
      forgetgate = GF(z[:,2 * n_units:3 * n_units])
      outgate = GO(z[:,3 * n_units:])
      input = CI(z[:,:n_units])
      s_i = input * ingate + s_p * forgetgate
      s_t = s_i #if not self.W_proj else T.dot(s_i, self.W_proj)
      h_t = CO(s_t) * outgate
      s_out = T.set_subtensor(s_batch[j_t], s_i)
      h_out = T.set_subtensor(h_batch[j_t], h_t)
      return theano.gradient.grad_clip(s_out, -50, 50), h_out

    def step(z, i_t, s_p, h_p):
      h_q = h_p if not self.attrs['projection'] else T.dot(h_p, self.W_proj)
      h_r = h_q if not self.attrs['n_proto'] else T.dot(h_q, self.W_pro)
      z += T.dot(h_q, self.W_re)
      #z += T.dot(T.nnet.softmax(T.dot(h_p, self.W_cls)), self.W_rec) + T.dot(h_p, self.W_re)
      if self.attrs['loop'] != 0 and not self.train_flag:
        if self.attrs['loop'] == -1: # direct loop
          z += self.W_rec[T.argmax(T.dot(h_r, self.W_cls), axis = -1)]
        else: # projected loop:
          z += T.dot(self.W_rec[T.argmax(T.dot(h_r, self.W_cls), axis = -1)], self.W_loop)
      i = T.outer(i_t, T.alloc(numpy.cast['int8'](1), n_units))
      j = i #if not self.attrs['projection'] else T.outer(i_t, T.alloc(numpy.cast['int8'](1), n_re))
      ingate = GI(z[:,n_units: 2 * n_units])
      forgetgate = GF(z[:,2 * n_units:3 * n_units])
      outgate = GO(z[:,3 * n_units:])
      input = CI(z[:,:n_units])
      s_i = input * ingate + s_p * forgetgate
      s_t = s_i #if not self.W_proj else T.dot(s_i, self.W_proj)
      h_t = CO(s_t) * outgate
      return theano.gradient.grad_clip(s_i * i + s_p * (1-i), -50, 50), h_t * j + h_p * (1-j)

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
        sequences = T.alloc(numpy.cast[theano.config.floatX](0), n_dec, encoder.output.shape[1], n_units * 4) + self.b
        if self.attrs['loop'] != 0 and self.train_flag:
          if self.attrs['loop'] == -1:
            sequences = T.inc_subtensor(sequences[1:], self.W_rec[self.y.reshape((n_dec, encoder.output.shape[1]), ndim=2)][:-1])
          else:
            sequences = T.inc_subtensor(sequences[1:], T.dot(self.W_rec[self.y.reshape((n_dec, encoder.output.shape[1]), ndim=2)], self.W_loop)[:-1])
      else:
        outputs_info = [ T.alloc(numpy.cast[theano.config.floatX](0), self.sources[0].output.shape[1], n_units),
                         T.alloc(numpy.cast[theano.config.floatX](0), self.sources[0].output.shape[1], n_units) ]
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
    self.softmax_input = self.lstm_output if self.attrs['n_proto'] <= 0 else T.dot(self.lstm_output, self.W_pro)

    self.y_m = T.dot(self.softmax_input, self.W_cls).dimshuffle(2,0,1).flatten(ndim = 2).dimshuffle(1,0)
    self.y_pred = T.argmax(self.y_m, axis=-1)
    self.output = T.argmax(T.dot(self.softmax_input, self.W_cls), axis=-1, keepdims=True) #T.argmax(self.y_m, axis=-1, keepdims=True).dimshuffle(1,0).reshape(self.lstm_output.shape, ndim = 3).dimshuffle(1,2,0) #.argmax(self.p_y_given_x, axis=-1, keepdim)
    self.attrs['sparse'] = True
    self.j = (self.index.flatten() > 0).nonzero()

  def get_branching(self):
    return sum([W.get_value().shape[0] for W in self.W_in]) + 1 + self.attrs['n_units'] + self.attrs['n_classes']

  def get_energyo(self):
    energy =  self.b / (4 * self.attrs['n_units'])
    for W in self.W_in:
      energy += T.sum(W, axis = 0)
    energy += T.sum(self.W_re, axis = 0)
    return energy

  def get_energy(self):
    energy =  abs(self.b) / (4 * self.attrs['n_out'])
    for W in self.W_in:
      energy += T.sum(abs(W), axis = 0)
    energy += T.sum(abs(self.W_re), axis = 0)
    return energy

  def make_constraints(self):
    if self.attrs['varreg'] > 0.0:
      # input: W_in, W_re, b
      energy = self.get_energy()
      #self.constraints = self.attrs['varreg'] * T.mean((energy - 6.0)**2) #(6.0 - T.var(energy, axis = 0))**2) #(T.sqrt(T.var(energy)) - T.sqrt(6.0))**2
      self.constraints =  self.attrs['varreg'] * (T.mean(energy) - T.sqrt(6.)) #T.mean((energy - 6.0)**2) # * T.var(energy) #(T.sqrt(T.var(energy)) - T.sqrt(6.0))**2
    return super(LstmOutputLayer, self).make_constraints()

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
