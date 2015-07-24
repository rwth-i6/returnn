
import numpy
from theano import tensor as T
import theano
from NetworkRecurrentLayer import RecurrentLayer
from ActivationFunctions import strtoact
from FastLSTM import LSTMOpInstance

class LstmLayer(RecurrentLayer):
  def __init__(self, n_out, sharpgates='none', **kwargs):
    kwargs.setdefault("layer_class", "lstm")
    kwargs.setdefault("activation", "sigmoid")
    kwargs.setdefault("compile", False)
    projection = kwargs.get("projection", None)
    n_re = projection if projection is not None else n_out
    W_in_m = n_out * 3 + n_re  # output dim of W_in and dim of bias. see step()
    kwargs["n_out"] = W_in_m
    super(LstmLayer, self).__init__(**kwargs)
    self.set_attr('n_out', n_out)
    if not isinstance(self.activation, (list, tuple)):
      self.activation = [T.tanh, T.nnet.sigmoid, T.nnet.sigmoid, T.nnet.sigmoid, T.tanh]
    else:
      assert len(self.activation) == 5, "lstm activations have to be specified as 5 tuple (input, ingate, forgetgate, outgate, output)"
    self.set_attr('sharpgates', sharpgates)
    CI, GI, GF, GO, CO = self.activation
    n_in = sum([s.attrs['n_out'] for s in self.sources])
    self.b.set_value(numpy.zeros((n_out * 3 + n_re,), dtype=theano.config.floatX))
    if projection:
      W_proj = self.create_random_uniform_weights(n_out, n_re, n_in + n_out + n_re, name="W_proj_%s" % self.name)
      self.W_proj.set_value(W_proj.get_value())
    W_re = self.create_random_uniform_weights(n=n_re, m=W_in_m, p=n_in + n_re + W_in_m,
                                              name="W_re_%s" % self.name)
    self.W_re.set_value(W_re.get_value())
    assert len(self.sources) == len(self.W_in)
    for s, W in zip(self.sources, self.W_in):
      W.set_value(self.create_random_uniform_weights(n=s.attrs['n_out'], m=W_in_m,
                                                     p=s.attrs['n_out'] + n_out + W_in_m,
                                                     name=W.name).get_value(borrow=True, return_internal_type=True), borrow=True)
    self.o.set_value(numpy.ones((n_out,), dtype='int8')) #TODO what is this good for?
    if sharpgates == 'global':
      self.sharpness = self.create_random_uniform_weights(3, n_out)
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
    else:
      self.sharpness = theano.shared(value = numpy.zeros((3,), dtype=theano.config.floatX), borrow=True, name = 'lambda')
    self.sharpness.set_value(numpy.ones(self.sharpness.get_value().shape, dtype = theano.config.floatX))
    if sharpgates != 'none' and sharpgates != "shared" and sharpgates != "single":
      self.sharpness = self.add_param(self.sharpness, 'gate_scaling')

    #set default value if not set
    if not 'optimization' in self.attrs:
      self.attrs['optimization'] = 'speed'
    assert self.attrs['optimization'] in ['memory', 'speed']
    if self.attrs['optimization'] == 'speed':
      z = self.b
      for x_t, m, W in zip(self.sources, self.masks, self.W_in):
        if x_t.attrs['sparse']:
          z += W[T.cast(x_t.output[:,:,0], 'int32')]
        elif m is None:
          z += T.tensordot(x_t.output, W, [[2],[2]])
          #z += T.dot(x_t.output, W)
        else:
          z += T.dot(self.mass * m * x_t.output, W)
    else:
      z = 0 if not self.sources else T.concatenate([x.output for x in self.sources], axis = -1)

    def step(z, i_t, s_p, h_p):
      if self.attrs['optimization'] == 'memory':
        offset = 0
        y = 0
        for x_t, m, W in zip(self.sources, self.masks, self.W_in):
          xin = z[:,:,offset:x_t.output.shape[2]]
          offset += x_t.output.shape[2]
          if x_t.attrs['sparse']:
            y += W[T.cast(xin[:,:,0], 'int32')]
          elif m is None:
            y += T.dot(xin, W)
          else:
            y += T.dot(self.mass * m * xin, W)
        z = y + self.b
      z += T.dot(h_p, self.W_re)
      i = T.outer(i_t, T.alloc(numpy.cast['int8'](1), n_out))
      j = i if not self.W_proj else T.outer(i_t, T.alloc(numpy.cast['int8'](1), n_re))
      ingate = GI(z[:,n_out: 2 * n_out])
      forgetgate = GF(z[:,2 * n_out:3 * n_out])
      outgate = GO(z[:,3 * n_out:])
      input = CI(z[:,:n_out])
      s_i = input * ingate + s_p * forgetgate
      s_t = s_i if not self.W_proj else T.dot(s_i, self.W_proj)
      h_t = CO(s_t) * outgate
      return theano.gradient.grad_clip(s_i * i, -50, 50), h_t * j

    def osstep(*args):
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
      assert len(x_ts) == len(masks) == len(self.W_in)
      for x_t, m, W in zip(x_ts, masks, self.W_in):
        if m is None:
          z += T.dot(x_t, W)
        else:
          z += T.dot(self.mass * m * x_t, W)

      if sharpgates != 'none':
        ingate = GI(self.sharpness[0] * z[:, n_out:2 * n_out])
        forgetgate = GF(self.sharpness[1] * z[:, 2 * n_out:3 * n_out])
        outgate = GO(self.sharpness[2] * z[:, 3 * n_out:])
      else:
        ingate = GI(z[:, n_out:2 * n_out])
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
                                  sequences = [ s.output[::self.attrs['sampling']] for s in self.sources ] + [self.index[::self.attrs['sampling']]],
                                  non_sequences = self.masks if any(self.masks) else [],
                                  outputs_info = [ T.alloc(numpy.cast[theano.config.floatX](0), self.sources[0].output.shape[1] / self.attrs['sampling'], n_out),
                                                   T.alloc(numpy.cast[theano.config.floatX](0), self.sources[0].output.shape[1] / self.attrs['sampling'], n_re), ])
    if self.attrs['sampling'] > 1:
      act = T.repeat(act, self.attrs['sampling'], axis = 1)[::self.sources[0].output.shape[1]]
    self.output = act[::-(2 * self.attrs['reverse'] - 1)]


#faster but needs much more memory
class OptimizedLstmLayer(RecurrentLayer):
  def __init__(self, n_out, sharpgates='none', encoder = None, n_dec = 0, **kwargs):
    kwargs.setdefault("layer_class", "lstm_opt")
    kwargs.setdefault("activation", "sigmoid")
    kwargs["compile"] = False
    kwargs["n_out"] = n_out * 4
    super(OptimizedLstmLayer, self).__init__(**kwargs)
    self.set_attr('n_out', n_out)
    if n_dec: self.set_attr('n_dec', n_dec)
    if encoder:
      self.set_attr('encoder', ",".join([e.name for e in encoder]))
    projection = kwargs.get("projection", None)
    if not isinstance(self.activation, (list, tuple)):
      self.activation = [T.tanh, T.nnet.sigmoid, T.nnet.sigmoid, T.nnet.sigmoid, T.tanh]
    else:
      assert len(self.activation) == 5, "lstm activations have to be specified as 5 tuple (input, ingate, forgetgate, outgate, output)"
    self.set_attr('sharpgates', sharpgates)
    CI, GI, GF, GO, CO = self.activation # T.tanh, T.nnet.sigmoid, T.nnet.sigmoid, T.nnet.sigmoid, T.tanh
    n_in = sum([s.attrs['n_out'] for s in self.sources])
    #self.state = self.create_bias(n_out, 'state')
    #self.act = self.create_bias(n_re, 'act')
    if self.depth > 1:
      value = numpy.zeros((self.depth, n_out * 4), dtype = theano.config.floatX)
      value[:,2 * n_out:3 * n_out] = 1
    else:
      value = numpy.zeros((n_out * 4, ), dtype = theano.config.floatX)
      value[2 * n_out:3 * n_out] = 1
    self.b.set_value(value)
    n_re = n_out
    if projection:
      n_re = projection
      W_proj = self.create_random_uniform_weights(n_out, projection, projection + n_out, name="W_proj_%s" % self.name)
      self.W_proj.set_value(W_proj.get_value())
      #self.set_attr('n_out', projection)
    W_re = self.create_random_uniform_weights(n_re, n_out * 4, n_in + n_out * 4,
                                              name="W_re_%s" % self.name)
    self.W_re.set_value(W_re.get_value())
    for s, W in zip(self.sources, self.W_in):
      W.set_value(self.create_random_uniform_weights(s.attrs['n_out'], n_out * 4,
                                                     s.attrs['n_out'] + n_out + n_out * 4,
                                                     name="W_in_%s_%s" % (s.name, self.name)).get_value(), borrow = True)

    if sharpgates == 'global':
      self.sharpness = self.create_random_uniform_weights(3, n_out)
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
    else:
      self.sharpness = theano.shared(value = numpy.zeros((3,), dtype=theano.config.floatX), borrow=True, name='lambda')
    self.sharpness.set_value(numpy.ones(self.sharpness.get_value().shape, dtype = theano.config.floatX))
    if sharpgates != 'none' and sharpgates != "shared" and sharpgates != "single":
      self.add_param(self.sharpness, 'gate_scaling')

    z = self.b
    for x_t, m, W in zip(self.sources, self.masks, self.W_in):
      if x_t.attrs['sparse']:
        z += W[T.cast(x_t.output[:,:,0], 'int32')]
      elif m is None:
        #z += T.tensordot(source.output, W, [[2],[0]])
        #z += T.dot(x_t.output.dimshuffle(0,1,'x',2), W)
        z += self.dot(x_t.output, W) #, [[2],[0]]) #.reshape((x_t.output.shape[0], x_t.output.shape[1], self.depth, 4 * n_out), ndim = 4) # tbd4m
        #z += T.tensordot(x_t.output, W, [[0], [0]])
      else:
        z += self.dot(self.mass * m * x_t.output, W)

    #self.set_attr('n_out', self.attrs['n_out'] * 4)
    #self.output = T.sum(z, axis=2) #.reshape((x_t.output.shape[0], x_t.output.shape[1], 4 * n_out), ndim = 3)
    #self.output = self.sources[0].output
    #return

    def index_step(z_batch, i_t, s_batch, h_batch): # why is this slower :(
      q_t = i_t #T.switch(T.any(i_t), i_t, T.ones_like(i_t))
      j_t = (q_t > 0).nonzero()
      s_p = s_batch[j_t]
      h_p = h_batch[j_t]
      z = z_batch[j_t]
      z += T.dot(h_p, self.W_re)
      ingate = GI(z[:,n_out: 2 * n_out])
      forgetgate = GF(z[:,2 * n_out:3 * n_out])
      outgate = GO(z[:,3 * n_out:])
      input = CI(z[:,:n_out])
      s_i = input * ingate + s_p * forgetgate
      s_t = s_i if not self.W_proj else T.dot(s_i, self.W_proj)
      h_t = CO(s_t) * outgate
      s_out = T.set_subtensor(s_batch[j_t], s_i)
      h_out = T.set_subtensor(h_batch[j_t], h_t)
      return theano.gradient.grad_clip(s_out, -50, 50), h_out

    def step(z, i_t, s_p, h_p, W_re):
      h_r = h_p if self.depth == 1 else self.make_consensus(h_p, axis = 1) # bdm -> bm
      if self.attrs['projection']:
        idx = T.argmax(GO(self.dot(h_r, self.W_proj)), -1)
        h_x = self.dot(GO(self.dot(h_r, self.W_proj)), W_re)
        #h_x = W_re[idx,:]
      else:
        h_x = self.dot(h_r, W_re) if self.depth == 1 else self.make_consensus(self.dot(h_r, W_re), axis = 1)
        #T.max(GO(T.dot(T.sum(h_p, axis = -1), self.W_proj))) #T.max(GO(T.tensordot(h_p, self.W_proj, [[2], [2]])), axis = -1)
      z += h_x
      if len(self.W_in) == 0:
        z += self.b
      if self.depth > 1:
        i = T.outer(i_t, T.alloc(numpy.cast['float32'](1), n_out)).dimshuffle(0, 'x', 1).repeat(self.depth, axis=1)
        ingate = GI(z[:,:,n_out: 2 * n_out]) # bdm
        forgetgate = GF(z[:,:,2 * n_out:3 * n_out]) # bdm
        outgate = GO(z[:,:,3 * n_out:]) # bdm
        input = CI(z[:,:,:n_out]) # bdm
      else:
        i = T.outer(i_t, T.alloc(numpy.cast['float32'](1), n_out))
        #ingate = GI(z[:,n_out: 2 * n_out])
        #forgetgate = GF(z[:,2 * n_out:3 * n_out])
        #outgate = GO(z[:,3 * n_out:])
        #input = CI(z[:,:n_out]) # bdm
        ingate = GI(z[:,: n_out])
        forgetgate = GF(z[:,1 * n_out:2 * n_out])
        outgate = GO(z[:,2 * n_out:3 * n_out])
        input = CI(z[:,3 * n_out:]) # bdm

      #s_i = input * ingate + s_p * forgetgate
      s_t = (input * ingate + s_p * forgetgate) # bdm  #if not self.W_proj else T.dot(s_i, self.W_proj)
      #h_t = T.max(CO(s_t) * outgate, axis = -1, keepdims = False) #T.max(CO(s_t) * outgate, axis=-1, keepdims=True) #T.max(CO(s_t) * outgate, axis = -1, keepdims = True)
      h_t = CO(s_t) * outgate
      #return s_t, h_t
      return theano.gradient.grad_clip(s_t, -50, 50), h_t
      #return theano.gradient.grad_clip(s_t * i + s_p * (1-i), -50, 50), h_t * i + h_p * (1-i)


    self.out_dec = self.index.shape[0]
    if encoder and 'n_dec' in encoder[0].attrs:
      self.out_dec = encoder[0].out_dec
    for s in xrange(self.attrs['sampling']):
      index = self.index[s::self.attrs['sampling']]
      sequences = z #T.unbroadcast(z, 3)
      if encoder:
        n_dec = self.out_dec
        if 'n_dec' in self.attrs:
          n_dec = self.attrs['n_dec']
          #index = T.alloc(numpy.cast[numpy.int8](1), n_dec, encoder.output.shape[1]) #index[:n_dec] #T.alloc(numpy.cast[numpy.int8](1), n_dec, encoder.output.shape[1])
          index = T.alloc(numpy.cast[numpy.int8](1), n_dec, encoder.index.shape[1])
        outputs_info = [ T.concatenate([e.state[-1] for e in encoder], axis = -1), T.concatenate([e.act[-1] for e in encoder], axis = -1) ]
        if len(self.W_in) == 0:
          if self.depth == 1:
            sequences = T.alloc(numpy.cast[theano.config.floatX](0), n_dec, encoder[0].output.shape[1], n_out * 4)
          else:
            sequences = T.alloc(numpy.cast[theano.config.floatX](0), n_dec, encoder[0].output.shape[1], self.depth, n_out * 4)
      else:
        if self.depth > 1:
          outputs_info = [ T.alloc(numpy.cast[theano.config.floatX](0), self.sources[0].output.shape[1], self.depth, n_out),
                           T.alloc(numpy.cast[theano.config.floatX](0), self.sources[0].output.shape[1], self.depth, n_out) ]
        else:
          outputs_info = [ T.alloc(numpy.cast[theano.config.floatX](0), self.sources[0].output.shape[1], n_out),
                           T.alloc(numpy.cast[theano.config.floatX](0), self.sources[0].output.shape[1], n_out) ]

      [state, act], _ = theano.scan(step,
                                    #strict = True,
                                    name = "scan_%s"%self.name,
                                    truncate_gradient = self.attrs['truncation'],
                                    go_backwards = self.attrs['reverse'],
                                    sequences = [ sequences[s::self.attrs['sampling']], T.cast(index, theano.config.floatX) ],
                                    outputs_info = outputs_info,
                                    non_sequences = [self.W_re])
      if self.attrs['sampling'] > 1: # time batch dim
        if s == 0:
          totact = T.repeat(act, self.attrs['sampling'], axis = 0)[:self.sources[0].output.shape[0]]
        else:
          totact = T.set_subtensor(totact[s::self.attrs['sampling']], act)
      else:
        totact = act
    self.state = state #[::-(2 * self.attrs['reverse'] - 1)]
    self.act = totact #[::-(2 * self.attrs['reverse'] - 1)] # tbdm
    self.make_output(self.act[::-(2 * self.attrs['reverse'] - 1)]) # if not self.attrs['projection'] else GO(self.dot(self.act, self.W_proj)))
    #self.output = T.sum(self.act, axis=2)
    #self.output = self.sources[0].output

  def get_branching(self):
    return sum([W.get_value().shape[0] for W in self.W_in]) + 1 + self.attrs['n_out']

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
      #self.constraints = self.attrs['varreg'] * (2.0 * T.sqrt(T.var(energy)) - 6.0)**2
      self.constraints =  self.attrs['varreg'] * (T.mean(energy) - T.sqrt(6.)) #T.mean((energy - 6.0)**2) # * T.var(energy) #(T.sqrt(T.var(energy)) - T.sqrt(6.0))**2
    return super(OptimizedLstmLayer, self).make_constraints()

#warning: highly experimental (not recommended for productive use yet)
#showed a speedup from 81 sec/epoch to 11 sec/epoch on demo dataset
#TODO:
# 1) !! use index vector for different sequence lengths
# 2) make sure implementations are compatible
# 3) support multiple sources, dropout, sharpgates, ...
# 4) use two CUDA streams for concurrent bidirectional execution
class FastLstmLayer(RecurrentLayer):
  def __init__(self, n_out, sharpgates='none', encoder = None, n_dec = 0, **kwargs):
    kwargs.setdefault("layer_class", "lstm_fast")
    kwargs.setdefault("activation", "sigmoid")
    kwargs["compile"] = False
    kwargs["n_out"] = n_out * 4
    super(FastLstmLayer, self).__init__(**kwargs)
    self.set_attr('n_out', n_out)
    value = numpy.zeros((n_out * 4, ), dtype = theano.config.floatX)
    self.b.set_value(value)
    n_re = n_out
    n_in = sum([s.attrs['n_out'] for s in self.sources])
    assert len(self.sources) == 1
    W_re = self.create_random_uniform_weights(n_re, n_out * 4, n_in + n_out * 4,
                                              name="W_re_%s" % self.name)
    self.W_re.set_value(W_re.get_value())
    for s, W in zip(self.sources, self.W_in):
      W.set_value(self.create_random_uniform_weights(s.attrs['n_out'], n_out * 4,
                                                     s.attrs['n_out'] + n_out + n_out * 4,
                                                     name="W_in_%s_%s" % (s.name, self.name)).get_value(), borrow = True)

    initial_state = T.alloc(numpy.cast[theano.config.floatX](0), self.sources[0].output.shape[1], n_out)
    self.act = LSTMOpInstance(self.sources[0].output[::-(2 * self.attrs['reverse'] - 1)], self.W_in[0], self.W_re, initial_state, self.b, self.index)[0]
    self.make_output(self.act[::-(2 * self.attrs['reverse'] - 1)])


class GRULayer(RecurrentLayer):
  def __init__(self, n_out, encoder = None, mode = "cho", n_dec = 0, **kwargs):
    kwargs.setdefault("layer_class", "gru")
    kwargs.setdefault("activation", "sigmoid")
    kwargs["compile"] = False
    kwargs["n_out"] = n_out * 3
    super(GRULayer, self).__init__(**kwargs)
    self.set_attr('n_out', n_out)
    self.set_attr('mode', mode)
    self.mode = mode
    if n_dec: self.set_attr('n_dec', n_dec)
    if encoder:
      self.set_attr('encoder', ",".join([e.name for e in encoder]))
    projection = kwargs.get("projection", None)
    if self.depth > 1:
      value = numpy.zeros((self.depth, n_out * 3), dtype = theano.config.floatX)
    else:
      value = numpy.zeros((n_out * 3, ), dtype = theano.config.floatX)
    self.b.set_value(value)
    n_re = n_out
    if projection:
      n_re = projection
      W_proj = self.create_random_uniform_weights(n_out, projection, projection + n_out, name="W_proj_%s" % self.name)
      self.W_proj.set_value(W_proj.get_value())
      #self.set_attr('n_out', projection)
    W_reset = self.create_random_uniform_weights(n_re, n_out, n_re + n_out * 3, name="W_re_%s" % self.name)
    self.W_reset = self.add_param(W_reset, "W_reset_%s" % self.name)
    W_re = self.create_random_uniform_weights(n_re, n_out * 2, n_re + n_out * 3, name="W_re_%s" % self.name)
    self.W_re.set_value(W_re.get_value())
    for s, W in zip(self.sources, self.W_in):
      W.set_value(self.create_random_uniform_weights(s.attrs['n_out'], n_out * 3,
                                                     s.attrs['n_out'] + n_out + n_out * 3,
                                                     name="W_in_%s_%s" % (s.name, self.name)).get_value(), borrow = True)
    z = self.b
    for x_t, m, W in zip(self.sources, self.masks, self.W_in):
      if x_t.attrs['sparse']:
        z += W[T.cast(x_t.output[:,:,0], 'int32')]
      elif m is None:
        z += self.dot(x_t.output, W)
      else:
        z += self.dot(self.mass * m * x_t.output, W)

    #if self.mode == 'cho':
    #  CI, GR, GU = [T.tanh, T.nnet.sigmoid, T.nnet.sigmoid]
    #else:
    CI, GR, GU = [T.tanh, T.nnet.sigmoid, T.nnet.sigmoid]

    def step(z, i_t, h_p, W_re):
      h_i = h_p if self.depth == 1 else self.make_consensus(h_p, axis = 1)
      i = T.outer(i_t, T.alloc(numpy.cast['float32'](1), n_out))
      if len(self.W_in) == 0:
        z += self.b
      h_x = self.dot(h_i, W_re) if self.depth == 1 else self.make_consensus(self.dot(h_i, W_re), axis = 1)
      r_t = GR(z[:,:n_out] + h_x[:,:n_out])
      h_r = self.dot(r_t * h_i, W_reset)
      z_t = GU(z[:,n_out:2*n_out] + h_x[:,n_out:2*n_out])
      h_cand = CI(z[:,2*n_out:] + h_r)
      h_t = z_t * h_i + (1 - z_t) * h_cand
      return h_t * i + h_i * (1-i)


    self.out_dec = self.index.shape[0]
    if encoder and 'n_dec' in encoder[0].attrs:
      self.out_dec = encoder[0].out_dec
    for s in xrange(self.attrs['sampling']):
      index = self.index[s::self.attrs['sampling']]
      sequences = z #T.unbroadcast(z, 3)
      if encoder:
        n_dec = self.out_dec
        if 'n_dec' in self.attrs:
          n_dec = self.attrs['n_dec']
          index = T.alloc(numpy.cast[numpy.int8](1), n_dec, encoder[0].index.shape[1])
        outputs_info = [ T.concatenate([e.act[-1] for e in encoder], axis = -1) ]
        if len(self.W_in) == 0:
          if self.depth == 1:
            sequences = T.alloc(numpy.cast[theano.config.floatX](0), n_dec, encoder[0].output.shape[1], n_out * 3)
          else:
            sequences = T.alloc(numpy.cast[theano.config.floatX](0), n_dec, encoder[0].output.shape[1], self.depth, n_out * 3)
      else:
        if self.depth > 1:
          outputs_info = [ T.alloc(numpy.cast[theano.config.floatX](0), self.sources[0].output.shape[1], self.depth, n_out) ]
        else:
          outputs_info = [ T.alloc(numpy.cast[theano.config.floatX](0), self.sources[0].output.shape[1], n_out) ]

      act, _ = theano.scan(step,
                            #strict = True,
                            name = "scan_%s"%self.name,
                            truncate_gradient = self.attrs['truncation'],
                            go_backwards = self.attrs['reverse'],
                            sequences = [ sequences[s::self.attrs['sampling']], T.cast(index, theano.config.floatX) ],
                            outputs_info = outputs_info,
                            non_sequences = [self.W_re])
      if self.attrs['sampling'] > 1: # time batch dim
        if s == 0:
          totact = T.repeat(act, self.attrs['sampling'], axis = 0)[:self.sources[0].output.shape[0]]
        else:
          totact = T.set_subtensor(totact[s::self.attrs['sampling']], act)
      else:
        totact = act
    self.act = totact #[::-(2 * self.attrs['reverse'] - 1)] # tbdm
    self.make_output(self.act[::-(2 * self.attrs['reverse'] - 1)])

class SRULayer(RecurrentLayer):
  def __init__(self, n_out, encoder = None, psize = 0, pact = 'relu', pdepth = 1, carry_time = False, n_dec = 0, **kwargs):
    kwargs.setdefault("layer_class", "sru")
    kwargs.setdefault("activation", "sigmoid")
    kwargs["compile"] = False
    kwargs["n_out"] = n_out * 3
    super(SRULayer, self).__init__(**kwargs)
    self.set_attr('n_out', n_out)
    self.set_attr('psize', psize)
    self.set_attr('pact', pact)
    self.set_attr('pdepth', pdepth)
    self.set_attr('carry_time', carry_time)
    if encoder:
      self.set_attr('encoder', ",".join([e.name for e in encoder]))
    pact = strtoact(pact)
    if n_dec: self.set_attr('n_dec', n_dec)
    if self.depth > 1:
      value = numpy.zeros((self.depth, n_out * 3), dtype = theano.config.floatX)
      value[:,n_out:2*n_out] = 1
    else:
      value = numpy.zeros((n_out * 3, ), dtype = theano.config.floatX)
      value[n_out:2*n_out] = 1
    #self.b.set_value(value)
    self.b = theano.shared(value=numpy.zeros((n_out * 3,), dtype=theano.config.floatX), borrow=True, name="b_%s"%self.name) #self.create_bias()
    self.params["b_%s"%self.name] = self.b
    n_re = n_out #psize if psize else n_out
    if self.attrs['consensus'] == 'flat':
      n_re *= self.depth
    self.Wp = []
    if psize:
      self.Wp = [ self.add_param(self.create_random_uniform_weights(n_re, psize, n_re + psize, name = "Wp_0_%s"%self.name, depth=1), name = "Wp_0_%s"%self.name) ]
      for i in xrange(1, pdepth):
        self.Wp.append(self.add_param(self.create_random_uniform_weights(psize, psize, psize + psize, name = "Wp_%d_%s"%(i, self.name), depth=1), name = "Wp_%d_%s"%(i, self.name)))
      W_re = self.create_random_uniform_weights(psize, n_out * 3, n_re + n_out * 3, name="W_re_%s" % self.name)
    else:
      W_re = self.create_random_uniform_weights(n_re, n_out * 3, n_re + n_out * 3, name="W_re_%s" % self.name)
    #self.params["W_re_%s" % self.name] = W_re
    #self.W_re = W_re
    self.W_re.set_value(W_re.get_value())
    self.W_in = []
    for s in self.sources:
      W = self.create_random_uniform_weights(s.attrs['n_out'], n_out * 3,
                                             s.attrs['n_out'] + n_out * 3,
                                             name="W_in_%s_%s" % (s.name, self.name), depth = 1)
      self.W_in.append(W)
      self.params["W_in_%s_%s" % (s.name, self.name)] = W
    z = self.b
    for x_t, m, W in zip(self.sources, self.masks, self.W_in):
      if x_t.attrs['sparse']:
        z += W[T.cast(x_t.output[:,:,0], 'int32')]
      elif m is None:
        z += T.dot(x_t.output, W)
      else:
        z += self.dot(self.mass * m * x_t.output, W)

    if self.depth > 1:
      z = z.dimshuffle(0,1,'x',2).repeat(self.depth, axis=2)

    x = T.concatenate([s.output for s in self.sources], axis = -1)
    if carry_time:
      assert sum([s.attrs['n_out'] for s in self.sources]) == self.attrs['n_out'], "input / output dimensions do not match in %s. input %d, output %d" % (self.name, sum([s.attrs['n_out'] for s in self.sources]), self.attrs['n_out'])
      name = 'W_RT_%s'%self.name
      W_rt = self.add_param(self.create_random_uniform_weights(self.attrs['n_out'], self.attrs['n_out'], name=name), name=name)

    CI, GR, GU = [T.tanh, T.nnet.sigmoid, T.nnet.sigmoid]

    def step(x, z, i_t, h_p, W_re):
      h_i = h_p if self.depth == 1 else self.make_consensus(h_p, axis = 1)
      i = T.outer(i_t, T.alloc(numpy.cast['float32'](1), n_out)) if self.depth == 1 else T.outer(i_t, T.alloc(numpy.cast['float32'](1), n_out)).dimshuffle(0, 'x', 1).repeat(self.depth, axis=1)
      if not self.W_in:
        z += self.b
      for W in self.Wp:
        h_i = pact(T.dot(h_i, W))
      h_x = self.dot(h_i, W_re)
      #if self.depth > 1:
      #  h_x = self.make_consensus(h_x, axis = 1)
      #h_x = h_q # if self.depth == 1 else self.make_consensus(h_q, axis = 1)
      if self.depth == 1:
        z_t = GU(z[:,:n_out] + h_x[:,:n_out])
        r_t = GR(z[:,n_out:2*n_out] + h_x[:,n_out:2*n_out])
        h_c = CI(z[:,2*n_out:] + r_t * h_x[:,2*n_out:])
      else:
        z_t = GU(z[:,:,:n_out] + h_x[:,:,:n_out])
        r_t = GR(z[:,:,n_out:2*n_out] + h_x[:,:,n_out:2*n_out])
        h_c = CI(z[:,:,2*n_out:] + r_t * h_x[:,:,2*n_out:])
      h_t = z_t * h_p + (1 - z_t) * h_c
      if carry_time:
        Tr = T.nnet.sigmoid(self.dot(x, W_rt))
        h_t = Tr * h_t + (1 - Tr) * x
      return h_t * i + h_p * (1 - i)

    self.out_dec = self.index.shape[0]
    if encoder and 'n_dec' in encoder[0].attrs:
      self.out_dec = encoder[0].out_dec
    for s in xrange(self.attrs['sampling']):
      index = self.index[s::self.attrs['sampling']]
      sequences = z #T.unbroadcast(z, 3)
      if encoder:
        n_dec = self.out_dec
        if 'n_dec' in self.attrs:
          n_dec = self.attrs['n_dec']
          index = T.alloc(numpy.cast[numpy.int8](1), n_dec, encoder.index.shape[1])
        outputs_info = [ T.concatenate([e.act[-1] for e in encoder], axis = -1) ]
        if len(self.W_in) == 0:
          if self.depth == 1:
            sequences = T.alloc(numpy.cast[theano.config.floatX](0), n_dec, encoder[0].output.shape[1], n_out * 3)
          else:
            sequences = T.alloc(numpy.cast[theano.config.floatX](0), n_dec, encoder[0].output.shape[1], self.depth, n_out * 3)
      else:
        if self.depth > 1:
          outputs_info = [ T.alloc(numpy.cast[theano.config.floatX](0), self.sources[0].output.shape[1], self.depth, n_out) ]
        else:
          outputs_info = [ T.alloc(numpy.cast[theano.config.floatX](0), self.sources[0].output.shape[1], n_out) ]

      act, _ = theano.scan(step,
                          #strict = True,
                          name = "scan_%s"%self.name,
                          truncate_gradient = self.attrs['truncation'],
                          go_backwards = self.attrs['reverse'],
                          sequences = [ x[s::self.attrs['sampling']], sequences[s::self.attrs['sampling']], T.cast(index, theano.config.floatX) ],
                          outputs_info = outputs_info,
                          non_sequences = [self.W_re])
      if self.attrs['sampling'] > 1: # time batch dim
        if s == 0:
          totact = T.repeat(act, self.attrs['sampling'], axis = 0)[:self.sources[0].output.shape[0]]
        else:
          totact = T.set_subtensor(totact[s::self.attrs['sampling']], act)
      else:
        totact = act
    self.act = totact #[::-(2 * self.attrs['reverse'] - 1)] # tbdm
    self.make_output(self.act[::-(2 * self.attrs['reverse'] - 1)])

class SRALayer(RecurrentLayer):
  def __init__(self, n_out, encoder = None, psize = 0, pact = 'relu', pdepth = 1, n_dec = 0, **kwargs):
    kwargs.setdefault("layer_class", "sra")
    kwargs.setdefault("activation", "sigmoid")
    kwargs["compile"] = False
    kwargs["n_out"] = n_out * 2
    super(SRALayer, self).__init__(**kwargs)
    self.set_attr('psize', psize)
    self.set_attr('pact', pact)
    self.set_attr('pdepth', pdepth)
    self.set_attr('n_out', n_out)
    if encoder:
      self.set_attr('encoder', ",".join([e.name for e in encoder]))
    pact = strtoact(pact)
    if n_dec: self.set_attr('n_dec', n_dec)
    if False and self.depth > 1:
      value = numpy.zeros((self.depth, n_out), dtype = theano.config.floatX)
      value[:,n_out:] = 1
    else:
      value = numpy.zeros((n_out, ), dtype = theano.config.floatX)
      value[n_out:] = 1
    #self.b.set_value(value)
    self.b = theano.shared(value=numpy.zeros((n_out,), dtype=theano.config.floatX), borrow=True, name="b_%s"%self.name) #self.create_bias()
    self.params["b_%s"%self.name] = self.b
    n_re = n_out #psize if psize else n_out
    if self.attrs['consensus'] == 'flat':
      n_re *= self.depth
    self.Wp = []
    if psize:
      self.Wp = [ self.add_param(self.create_random_uniform_weights(n_re, psize, n_re + psize, name = "Wp_0_%s"%self.name), name = "Wp_0_%s"%self.name) ]
      for i in xrange(1, pdepth):
        self.Wp.append(self.add_param(self.create_random_uniform_weights(psize * self.depth, psize, psize + psize, name = "Wp_%d_%s"%(i, self.name)), name = "Wp_%d_%s"%(i, self.name)))
      W_re = self.create_random_uniform_weights(psize * self.depth, n_out * 2, n_re + n_out * 2, name="W_re_%s" % self.name)
    else:
      W_re = self.create_random_uniform_weights(n_re, n_out * 2, n_re + n_out * 2, name="W_re_%s" % self.name)
    #self.params["W_re_%s" % self.name] = W_re
    #self.W_re = W_re
    self.W_re.set_value(W_re.get_value())
    self.W_in = []
    for s in self.sources:
      W = self.create_random_uniform_weights(s.attrs['n_out'], n_out,
                                             s.attrs['n_out'] + n_out,
                                             name="W_in_%s_%s" % (s.name, self.name), depth = 1)
      self.W_in.append(W)
      self.params["W_in_%s_%s" % (s.name, self.name)] = W
    z = self.b
    for x_t, m, W in zip(self.sources, self.masks, self.W_in):
      if x_t.attrs['sparse']:
        z += W[T.cast(x_t.output[:,:,0], 'int32')]
      elif m is None:
        z += T.dot(x_t.output, W)
      else:
        z += self.dot(self.mass * m * x_t.output, W)

    if not self.W_in and self.depth > 1:
      z = z.dimshuffle(0,1,'x',2).repeat(self.depth, axis=2)
    #if self.mode == 'cho':
    #  CI, GR, GU = [T.tanh, T.nnet.sigmoid, T.nnet.sigmoid]
    #else:
    #CI, GR, GU = [T.tanh, T.tanh, T.nnet.sigmoid]
    CI, GR, GU = [T.tanh, T.tanh, T.nnet.sigmoid]

    self.sp = self.add_param(theano.shared(value=numpy.asarray(self.rng.uniform(low=-1.0, high=1.0, size=(n_out,)), dtype=theano.config.floatX), borrow=True, name="sp_%s"%self.name), name="sp_%s"%self.name)

    def step(z, i_t, s_p, h_p):
      h_q = h_p #T.concatenate([CI(s_p), h_p], axis = -1)
      h_i = h_q if self.depth == 1 else self.make_consensus(h_q, axis = 1)
      i = T.outer(i_t, T.alloc(numpy.cast['float32'](1), n_out)) if self.depth == 1 else T.outer(i_t, T.alloc(numpy.cast['float32'](1), n_out)).dimshuffle(0, 'x', 1).repeat(self.depth, axis=1)
      if not self.W_in:
        z += self.b
      for W in self.Wp:
        h_i = self.make_consensus(pact(self.dot(h_i, W)), axis = 1)
      h_x = self.dot(h_i, self.W_re)
      #h_x = h_q # if self.depth == 1 else self.make_consensus(h_q, axis = 1)
      if self.depth == 1:
        u_t = GU(h_x[:,:n_out])
        r_t = GR(h_x[:,n_out:])
      else:
        u_t = GU(h_x[:,:,:n_out])
        r_t = GR(h_x[:,:,n_out:])
      s_t = r_t * s_p + r_t * self.sp #s_p  #+ 1 - u_t
      h_t = CI(u_t * z + s_t)
      return s_t * i + s_p * (1 - i), h_t * i + h_p * (1 - i)

    self.out_dec = self.index.shape[0]
    if encoder and 'n_dec' in encoder[0].attrs:
      self.out_dec = encoder[0].out_dec
    for s in xrange(self.attrs['sampling']):
      index = self.index[s::self.attrs['sampling']]
      sequences = z #T.unbroadcast(z, 3)
      if encoder:
        n_dec = self.out_dec
        if 'n_dec' in self.attrs:
          n_dec = self.attrs['n_dec']
          index = T.alloc(numpy.cast[numpy.int8](1), n_dec, encoder.index.shape[1])
        outputs_info = [ T.concatenate([e.state[-1] for e in encoder], axis = 1), T.concatenate([e.act[-1] for e in encoder], axis = 1) ]
        if len(self.W_in) == 0:
          if self.depth == 1:
            sequences = T.alloc(numpy.cast[theano.config.floatX](0), n_dec, encoder[0].output.shape[1], n_out)
          else:
            sequences = T.alloc(numpy.cast[theano.config.floatX](0), n_dec, encoder[0].output.shape[1], self.depth, n_out)
      else:
        if self.depth == 1:
          outputs_info = [ T.alloc(numpy.cast[theano.config.floatX](0), self.sources[0].output.shape[1], n_out), T.alloc(numpy.cast[theano.config.floatX](0), self.sources[0].output.shape[1], n_out) ]
          #outputs_info = [ T.alloc(numpy.cast[theano.config.floatX](0), self.sources[0].output.shape[1], n_out), T.alloc(numpy.cast[theano.config.floatX](0), self.sources[0].output.shape[1], n_out) ]
        else:
          #outputs_info = [ T.alloc(numpy.cast[theano.config.floatX](1), self.sources[0].output.shape[1], self.depth, n_out), T.alloc(numpy.cast[theano.config.floatX](0), self.sources[0].output.shape[1], self.depth, n_out) ]
          outputs_info = [ T.alloc(numpy.cast[theano.config.floatX](0), self.sources[0].output.shape[1], self.depth, n_out), T.alloc(numpy.cast[theano.config.floatX](0), self.sources[0].output.shape[1], self.depth, n_out) ]

      [state, act], _ = theano.scan(step,
                          #strict = True,
                          name = "scan_%s"%self.name,
                          truncate_gradient = self.attrs['truncation'],
                          go_backwards = self.attrs['reverse'],
                          sequences = [ sequences[s::self.attrs['sampling']], T.cast(index, theano.config.floatX) ],
                          outputs_info = outputs_info)
      if self.attrs['sampling'] > 1: # time batch dim
        if s == 0:
          totact = T.repeat(act, self.attrs['sampling'], axis = 0)[:self.sources[0].output.shape[0]]
        else:
          totact = T.set_subtensor(totact[s::self.attrs['sampling']], act)
      else:
        totact = act
    self.state = state
    self.act = totact #[::-(2 * self.attrs['reverse'] - 1)] # tbdm
    self.make_output(self.act)
    #self.make_output(T.concatenate([CI(self.state[::-(2 * self.attrs['reverse'] - 1)]), self.act[::-(2 * self.attrs['reverse'] - 1)]], axis = -1))
