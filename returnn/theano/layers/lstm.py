
import numpy
import json
from theano import tensor as T
import theano
from .hidden import HiddenLayer, _NoOpLayer
from returnn.theano.activation_functions import strtoact


class RecurrentLayer(HiddenLayer):
  recurrent = True
  layer_class = "recurrent"

  def __init__(self, reverse=False, truncation=-1, compile=True, projection=0, sampling=1, **kwargs):
    kwargs.setdefault("activation", "tanh")
    super(RecurrentLayer, self).__init__(**kwargs)
    self.set_attr('reverse', reverse)
    self.set_attr('truncation', truncation)
    self.set_attr('sampling', sampling)
    self.set_attr('projection', projection)
    n_in = sum([s.attrs['n_out'] for s in self.sources])
    n_out = self.attrs['n_out']
    self.act = self.create_bias(n_out)
    if projection:
      n_re_in = projection
    else:
      n_re_in = n_out
    self.W_re = self.add_param(self.create_random_normal_weights(n=n_re_in, m=n_out, scale=n_in,
                                                                 name="W_re_%s" % self.name))
    if projection:
      self.W_proj = self.add_param(self.create_forward_weights(n_out, projection, name='W_proj_%s' % self.name))
    else:
      self.W_proj = None
    #for s, W in zip(self.sources, self.W_in):
    #  W.set_value(self.create_random_normal_weights(n=s.attrs['n_out'], m=n_out, scale=n_in,
    #                                                name=W.name).get_value())
    self.o = theano.shared(value = numpy.ones((n_out,), dtype='int8'), borrow=True)
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
    return self.create_random_normal_weights(n, m, nin), self.create_random_normal_weights(m, m, nin)

class LstmLayer(RecurrentLayer):
  layer_class = "lstm"

  def __init__(self, n_out, sharpgates='none', **kwargs):
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
  layer_class = "lstm_opt"

  def __init__(self, n_out, sharpgates='none', encoder = None, n_dec = 0, **kwargs):
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
      value[2 * n_out:3 * n_out] = 0
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
      return s_t, h_t
      #return theano.gradient.grad_clip(s_t, -50, 50), h_t
      #return theano.gradient.grad_clip(s_t * i + s_p * (1-i), -50, 50), h_t * i + h_p * (1-i)


    self.out_dec = self.index.shape[0]
    if encoder and 'n_dec' in encoder[0].attrs:
      self.out_dec = encoder[0].out_dec
    for s in range(self.attrs['sampling']):
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
          outputs_info = [ T.alloc(numpy.cast[theano.config.floatX](0), self.index.shape[1], n_out),
                           T.alloc(numpy.cast[theano.config.floatX](0), self.index.shape[1], n_out) ]

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


class SimpleLstmLayer(RecurrentLayer):
  layer_class = "lstm_simple"

  def __init__(self, n_out, sharpgates='none', encoder = None, n_dec = 0, **kwargs):
    kwargs.setdefault("activation", "sigmoid")
    kwargs["compile"] = False
    kwargs["n_out"] = n_out * 4
    super(SimpleLstmLayer, self).__init__(**kwargs)
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

    X = self.sources[0].output[::-(2 * self.attrs['reverse'] - 1)]
    W = self.W_in[0]
    def _step(x_t, c_tm1, y_tm1):
      z_t = T.dot(x_t, W) + T.dot(y_tm1, self.W_re) + self.b
      partition = z_t.shape[1] / 4
      ingate = T.nnet.sigmoid(z_t[:,:partition])
      forgetgate = T.nnet.sigmoid(z_t[:,partition:2*partition])
      outgate = T.nnet.sigmoid(z_t[:,2*partition:3*partition])
      input = T.tanh(z_t[:,3*partition:4*partition])
      c_t = forgetgate * c_tm1 + ingate * input
      y_t = outgate * T.tanh(c_t)
      return c_t, y_t

    [self.state, self.act], _ = theano.scan(_step, sequences=[X],
                            outputs_info=[initial_state,
                                          initial_state])

    self.make_output(self.act[::-(2 * self.attrs['reverse'] - 1)])


def make_lstm_step(n_cells, W_re,
                   W_out_proj=None, W_re_proj=None,
                   W_peep_i=None, W_peep_f=None, W_peep_o=None,
                   grad_clip=None, CI=None, CO=None, G=None):
  # W_re: recurrent matrix. (n_out,n_cells*4)
  # W_out_proj: (n_cells,n_out) or None
  # W_re_proj: (n_out,n_proj) or None
  # W_peep_*: (n_cells,) or None
  if not CI: CI = T.tanh
  if not CO: CO = T.tanh
  if not G: G = T.nnet.sigmoid

  def lstm_step(z_t, i_t, s_p, h_p):
    # z_t: current input. (batch,n_cells*4)
    # i_t: 0 or 1 (via index). (batch,)
    # s_p: previous cell state. (batch,n_cells)
    # h_p: previous hidden out. (batch,n_out)
    i_t_bc = i_t.dimshuffle(0, 'x')
    if W_re_proj:
      h_p = T.dot(h_p, W_re_proj)
    z_t += T.dot(h_p, W_re)
    z_t *= i_t_bc
    input = CI(z_t[:,:n_cells])
    if W_peep_i or W_peep_f or W_peep_o:
      ingate = z_t[:,n_cells:2 * n_cells]
      forgetgate = z_t[:,2 * n_cells:3 * n_cells]
      if W_peep_i: ingate += s_p * W_peep_i.dimshuffle('x', 0) * i_t_bc
      if W_peep_f: forgetgate += s_p * W_peep_f.dimshuffle('x', 0) * i_t_bc
      s_t = input * G(ingate) + s_p * G(forgetgate)
      outgate = z_t[:,3 * n_cells:]
      if W_peep_o: outgate += s_t * W_peep_o.dimshuffle('x', 0) * i_t_bc
      h_t = CO(s_t) * G(outgate)
    else:  # no peepholes. simplified and faster
      gates = G(z_t[:,n_cells:])
      ingate = gates[:,:n_cells]
      forgetgate = gates[:,n_cells:2 * n_cells]
      outgate = gates[:,2 * n_cells:]
      s_t = input * ingate + s_p * forgetgate
      h_t = CO(s_t) * outgate
    if W_out_proj:
      h_t = T.dot(h_t, W_out_proj)
    s_t *= i_t_bc
    h_t *= i_t_bc
    if grad_clip:
      s_t = theano.gradient.grad_clip(s_t, -grad_clip, grad_clip)
      h_t = theano.gradient.grad_clip(h_t, -grad_clip, grad_clip)
    return s_t, h_t

  return lstm_step


def lstm(z, i, W_re, W_out_proj=None, W_re_proj=None, W_peep_i=None, W_peep_f=None, W_peep_o=None,
         CI=None, CO=None, G=None,
         grad_clip=None, direction=1):
  # z: (n_time,n_batch,n_cells*4)
  # i: (n_time,n_batch)
  # W_re: (n_out,n_cells*4)
  # W_out_proj: (n_cells,n_out) or None
  # W_re_proj: (n_out,n_proj) or None
  # W_peep_*: (n_cells,) or None
  n_batch = z.shape[1]
  assert W_re.ndim == 2
  n_cells = W_re.shape[1] // 4
  n_out = W_re.shape[0]  # normally the same as n_cells, but with W_proj, can be different
  if W_re_proj:
    n_out = W_re_proj.shape[0]
  i = T.cast(i, dtype="float32")  # so that it can run on gpu
  if grad_clip:
    grad_clip = numpy.float32(grad_clip)
  lstm_step = make_lstm_step(
    n_cells=n_cells, W_re=W_re,
    W_out_proj=W_out_proj, W_re_proj=W_re_proj, W_peep_i=W_peep_i, W_peep_f=W_peep_f, W_peep_o=W_peep_o,
    CI=CI, CO=CO, G=G,
    grad_clip=grad_clip)

  s_initial = T.zeros((n_batch, n_cells), dtype="float32")
  h_initial = T.zeros((n_batch, n_out), dtype="float32")
  go_backwards = {1:False, -1:True}[direction]
  (s, h), _ = theano.scan(lstm_step,
                          sequences=[z, i], go_backwards=go_backwards,
                          outputs_info=[s_initial, h_initial])
  h = h[::direction]
  return h


class Lstm2Layer(HiddenLayer):
  recurrent = True
  layer_class = "lstm2"

  def __init__(self, n_out, n_cells=None, n_proj=None, peepholes=False, direction=1, activation=None, grad_clip=None, truncation=None, **kwargs):
    if not n_cells: n_cells = n_out
    # It's a hidden layer, thus this will create the feed forward layer for the LSTM for the input.
    super(Lstm2Layer, self).__init__(n_out=n_cells * 4, **kwargs)
    self.set_attr('n_out', n_out)
    self.set_attr('n_cells', n_cells)
    if n_proj: self.set_attr('n_proj', n_proj)
    self.set_attr('peepholes', peepholes)
    self.set_attr('direction', direction)
    if grad_clip: self.set_attr('grad_clip', grad_clip)
    if activation: self.set_attr('activation', activation)

    n_re_in = n_out
    if n_proj:
      # Applied before recurrent matrix.
      self.W_re_proj = self.add_param(self.create_recurrent_weights(n=n_out, m=n_proj, name="W_re_proj_%s" % self.name))
      n_re_in = n_proj
    else:
      self.W_re_proj = None
    self.W_re = self.add_param(self.create_recurrent_weights(n=n_re_in, m=n_cells * 4, name="W_re_%s" % self.name))
    if n_out != n_cells:
      # Applied before output.
      self.W_out_proj = self.add_param(self.create_forward_weights(n_cells, n_out, name='W_proj_%s' % self.name))
    else:
      self.W_out_proj = None
    if peepholes:
      self.W_peepholes = [
        self.add_param(self.create_random_uniform_weights2(n_cells, name="W_peep_%s_%s" % (g, self.name)))
        for g in "ifo"]
    else:
      self.W_peepholes = [None] * 3

    CI, CO, G = [T.tanh, T.tanh, T.nnet.sigmoid]
    if activation:
      act_f = strtoact(activation)
      if isinstance(act_f, list):
        if len(act_f) == 2:
          CI, CO = act_f
        elif len(act_f) == 3:
          CI, CO, G = act_f
        else:
          assert False, "invalid number of activation funcs: %r" % act_f
      else:
        CI = CO = act_f

    z = self.get_linear_forward_output()
    h = lstm(z=z, i=self.index, W_re=self.W_re,
             W_out_proj=self.W_out_proj, W_re_proj=self.W_re_proj,
             W_peep_i=self.W_peepholes[0], W_peep_f=self.W_peepholes[1], W_peep_o=self.W_peepholes[2],
             CI=CI, CO=CO, G=G,
             grad_clip=grad_clip, direction=direction)
    self.make_output(h)


class Lstm3Layer(HiddenLayer):
  """
  Like lstm2 but even simpler.
  """
  recurrent = True
  layer_class = "lstm3"

  def __init__(self, n_out, direction=1, grad_clip=None, **kwargs):
    n_cells = n_out
    n_re_in = n_out
    # It's a hidden layer, thus this will create the feed forward layer for the LSTM for the input.
    super(Lstm3Layer, self).__init__(n_out=n_cells * 4, **kwargs)
    self.set_attr('n_out', n_out)
    self.set_attr('n_cells', n_cells)
    self.set_attr('direction', direction)
    if grad_clip:
      self.set_attr('grad_clip', grad_clip)
      grad_clip = numpy.float32(grad_clip)
    CI, CO, G = [T.tanh, T.tanh, T.nnet.sigmoid]

    z = self.get_linear_forward_output()
    self.W_re = self.add_param(self.create_recurrent_weights(n=n_re_in, m=n_cells * 4, name="W_re_%s" % self.name))

    def lstm_step(z_t, i_t, s_p, h_p):
      # z_t: current input. (batch,n_cells*4)
      # i_t: 0 or 1 (via index). (batch,)
      # s_p: previous cell state. (batch,n_cells)
      # h_p: previous hidden out. (batch,n_out)
      i_t_bc = i_t.dimshuffle(0, 'x')
      z_t += T.dot(h_p, self.W_re)
      z_t *= i_t_bc
      input = CI(z_t[:, :n_cells])
      gates = G(z_t[:, n_cells:])
      ingate = gates[:, :n_cells]
      forgetgate = gates[:, n_cells:2 * n_cells]
      outgate = gates[:, 2 * n_cells:]
      s_t = input * ingate + s_p * forgetgate
      h_t = CO(s_t) * outgate
      s_t *= i_t_bc
      h_t *= i_t_bc
      if grad_clip:
        s_t = theano.gradient.grad_clip(s_t, -grad_clip, grad_clip)
        h_t = theano.gradient.grad_clip(h_t, -grad_clip, grad_clip)
      return s_t, h_t

    i = T.cast(self.index, dtype="float32")  # so that it can run on gpu
    assert z.ndim == 3  # (time,batch,dim)
    n_batch = z.shape[1]
    s_initial = T.zeros((n_batch, n_cells), dtype="float32")
    h_initial = T.zeros((n_batch, n_out), dtype="float32")
    go_backwards = {1: False, -1: True}[direction]
    (s, h), _ = theano.scan(lstm_step,
                            sequences=[z, i], go_backwards=go_backwards,
                            outputs_info=[s_initial, h_initial])
    h = h[::direction]
    self.make_output(h)


class LayerNormLstmLayer(HiddenLayer):
  """
  Layer Normalization, https://arxiv.org/abs/1607.06450
  """
  recurrent = True
  layer_class = "ln_lstm"

  def __init__(self, n_out, direction=1, grad_clip=None, **kwargs):
    n_cells = n_out
    n_re_in = n_out
    # It's a hidden layer, thus this will create the feed forward layer for the LSTM for the input.
    super(LayerNormLstmLayer, self).__init__(n_out=n_cells * 4, **kwargs)
    self.set_attr('n_out', n_out)
    self.set_attr('n_cells', n_cells)
    self.set_attr('direction', direction)
    if grad_clip:
      self.set_attr('grad_clip', grad_clip)
      grad_clip = numpy.float32(grad_clip)
    C, G = [T.tanh, T.nnet.sigmoid]

    from returnn.theano.util import layer_normalization
    def _sliced_layer_norm(z, scale, idx):
      dimstart = n_cells * idx
      dimend = dimstart + n_cells
      zdims = (slice(None),) * (z.ndim - 1) + (slice(dimstart, dimend),)
      return layer_normalization(z[zdims], bias=None, scale=scale[dimstart:dimend])
    def sliced_layer_norm(z, scale):
      slices = [_sliced_layer_norm(z, scale=scale, idx=i) for i in range(4)]
      return T.concatenate(slices, axis=z.ndim - 1)

    z = self.get_linear_forward_output(with_bias=False)
    self.ln_zi_scale = self.add_param(self.create_bias(n=n_cells * 4, name="ln_zi_scale_%s" % self.name, init_eval_str="zeros() + 1"))
    z = sliced_layer_norm(z, scale=self.ln_zi_scale)
    z += self.b
    self.W_re = self.add_param(self.create_recurrent_weights(n=n_re_in, m=n_cells * 4, name="W_re_%s" % self.name))
    self.ln_zr_scale = self.add_param(self.create_bias(n=n_cells * 4, name="ln_zr_scale_%s" % self.name, init_eval_str="zeros() + 1"))
    self.ln_s_bias = self.add_param(self.create_bias(n=n_cells, name="ln_s_bias_%s" % self.name, init_eval_str="zeros()"))
    self.ln_s_scale = self.add_param(self.create_bias(n=n_cells, name="ln_s_scale_%s" % self.name, init_eval_str="zeros() + 1"))


    def lstm_step(z_t, i_t, s_p, h_p):
      # z_t: current input. (batch,n_cells*4)
      # i_t: 0 or 1 (via index). (batch,)
      # s_p: previous cell state. (batch,n_cells)
      # h_p: previous hidden out. (batch,n_out)
      i_t_bc = i_t.dimshuffle(0, 'x')
      z_t += sliced_layer_norm(T.dot(h_p, self.W_re), scale=self.ln_zr_scale)
      z_t *= i_t_bc
      input = C(z_t[:, :n_cells])
      gates = G(z_t[:, n_cells:])
      igate = gates[:, :n_cells]
      fgate = gates[:, n_cells:2 * n_cells]
      ogate = gates[:, 2 * n_cells:]
      s_t = input * igate + s_p * fgate
      s_t_ = layer_normalization(s_t, bias=self.ln_s_bias, scale=self.ln_s_scale)
      h_t = C(s_t_) * ogate
      s_t *= i_t_bc
      h_t *= i_t_bc
      if grad_clip:
        s_t = theano.gradient.grad_clip(s_t, -grad_clip, grad_clip)
        h_t = theano.gradient.grad_clip(h_t, -grad_clip, grad_clip)
      return s_t, h_t

    i = T.cast(self.index, dtype="float32")  # so that it can run on gpu
    assert z.ndim == 3  # (time,batch,dim)
    n_batch = z.shape[1]
    s_initial = T.zeros((n_batch, n_cells), dtype="float32")
    h_initial = T.zeros((n_batch, n_out), dtype="float32")
    go_backwards = {1: False, -1: True}[direction]
    (s, h), _ = theano.scan(lstm_step,
                            sequences=[z, i], go_backwards=go_backwards,
                            outputs_info=[s_initial, h_initial])
    h = h[::direction]
    self.make_output(h)


class NativeLstmLayer(HiddenLayer):
  recurrent = True
  layer_class = "native_lstm"

  def __init__(self, n_out, direction=1, truncation=None, **kwargs):
    n_cells = n_out
    # It's a hidden layer, thus this will create the feed forward layer for the LSTM for the input.
    super(NativeLstmLayer, self).__init__(n_out=n_cells * 4, **kwargs)
    self.set_attr('n_out', n_out)
    self.set_attr('direction', direction)

    n_re_in = n_out
    self.W_re = self.add_param(self.create_recurrent_weights(n=n_re_in, m=n_cells * 4, name="W_re_%s" % self.name))

    z = self.get_linear_forward_output()
    assert z.ndim == 3

    from returnn.native_op import LstmGenericBase
    lstm_op = LstmGenericBase().make_theano_op()
    op_out = lstm_op(*LstmGenericBase.map_layer_inputs_to_op(z[::direction], self.W_re, self.index[::direction]))
    from returnn.theano.util import make_var_tuple
    out = LstmGenericBase.map_layer_output_from_op(*make_var_tuple(op_out))
    self.make_output(out[::direction])


class GenericLstmLayer(_NoOpLayer):
  """
  LSTM implementation which allows a custom input+recurrent function (n_in + n_out -> n_cells * 4)
  and a custom output function (n_cells -> n_out) which is identity by default.
  You specify it as a sub layer.
  """
  recurrent = True
  layer_class = "generic_lstm"

  def __init__(self, n_out, sublayer, out_sublayer=None, n_cells=None,
               activation=None,
               direction=1, grad_clip=None, truncation=None, **kwargs):
    super(GenericLstmLayer, self).__init__(**kwargs)
    self.set_attr('n_out', n_out)
    if n_cells:
      self.set_attr('n_cells', n_cells)
    else:
      n_cells = n_out
    self.set_attr('direction', direction)
    if grad_clip:
      self.set_attr('grad_clip', grad_clip)
      grad_clip = numpy.float32(grad_clip)
    if isinstance(sublayer, (str, unicode)):
      sublayer = json.loads(sublayer)
    if isinstance(out_sublayer, (str, unicode)):
      out_sublayer = json.loads(out_sublayer)
    assert isinstance(sublayer, dict)
    self.set_attr('sublayer', sublayer.copy())
    if out_sublayer:
      assert isinstance(out_sublayer, dict)
      self.set_attr('out_sublayer', out_sublayer.copy())
    if activation:
      self.set_attr('activation', activation)

    from NetworkHiddenLayer import concat_sources
    x, n_in = concat_sources(self.sources, masks=self.masks, mass=self.mass, unsparse=True)  # (n_time,n_batch,n_in)
    n_time = x.shape[0]
    n_batch = x.shape[1]

    from returnn.theano.layers.base import SourceLayer
    from NetworkLayer import get_layer_class
    def make_sublayer(x_in, x_re, index, name):
      layer_opts = sublayer.copy()
      cl = layer_opts.pop("class")
      layer_class = get_layer_class(cl)
      s1_layer = SourceLayer(name="%s_source_in" % name, n_out=n_in, x_out=x_in, index=index)
      s2_layer = SourceLayer(name="%s_source_re" % name, n_out=n_out, x_out=x_re, index=index)
      layer = layer_class(sources=[s1_layer, s2_layer], index=index, name=name, n_out=n_cells * 4,
                          network=self.network, **layer_opts)
      self.sublayer = layer
      return layer.output
    self.sublayer = None
    def make_out_sublayer(h, index, name):
      if not out_sublayer: return h
      layer_opts = out_sublayer.copy()
      cl = layer_opts.pop("class")
      layer_class = get_layer_class(cl)
      s_layer = SourceLayer(name="%s_source_h" % name, n_out=n_cells, x_out=h, index=index)
      layer = layer_class(sources=[s_layer], index=index, name=name, n_out=n_out,
                          network=self.network, **layer_opts)
      self.out_sublayer = layer
      return layer.output
    self.out_sublayer = None

    CI, CO, GF = [T.tanh, T.tanh, T.nnet.sigmoid]
    if activation:
      act_f = strtoact(activation)
      if isinstance(act_f, list):
        if len(act_f) == 2:
          CI, CO = act_f
        elif len(act_f) == 3:
          CI, CO, GF = act_f
        else:
          assert False, "invalid number of activation funcs: %r" % act_f
      else:
        CI = CO = act_f

    def lstm_step(x_t, i_t, s_p, h_p):
      # x_t: current input. (dummy,batch,n_in)
      # i_t: 0 or 1 (via index). (dummy,batch,)
      # s_p: previous cell state. (batch,n_cells)
      # h_p: previous out. (dummy,batch,n_out)
      z_t = make_sublayer(x_in=x_t, x_re=h_p, index=i_t, name="%s_sublayer" % self.name)
      z_t = z_t[0]  # remove dummy dimension. (batch,n_cells*4)
      gates = GF(z_t[:, :3 * n_cells])
      u = CI(z_t[:, 3 * n_cells:])
      igate = gates[:, :n_cells]
      fgate = gates[:, n_cells:2 * n_cells]
      ogate = gates[:, 2 * n_cells:]
      s_t = u * igate + s_p * fgate
      h_t = s_t
      h_t = h_t.reshape((1, n_batch, n_cells))  # dummy,batch,n_cells
      h_t = T.patternbroadcast(h_t, (False, False, False))  # might a be Theano bug
      h_t = make_out_sublayer(h_t, index=i_t, name="%s_out_sublayer" % self.name)
      h_t = CO(h_t) * ogate.dimshuffle('x', 0, 1)  # dummy,batch,n_out
      s_t *= i_t[0].dimshuffle(0, 'x')  # batch,n_cells
      h_t *= i_t.dimshuffle(0, 1, 'x')  # dummy,batch,n_out
      if grad_clip:
        s_t = theano.gradient.grad_clip(s_t, -grad_clip, grad_clip)
        h_t = theano.gradient.grad_clip(h_t, -grad_clip, grad_clip)
      return s_t, h_t

    # i: (n_time,n_batch)
    i = T.cast(self.index, dtype="float32")  # so that it can run on gpu
    # Add extra dummy dimension. Used for sublayer.
    x = x.reshape((n_time, 1, n_batch, n_in))
    x = T.patternbroadcast(x, (False, False, False, False))  # might be a Theano bug
    i = i.reshape((n_time, 1, n_batch))
    i = T.patternbroadcast(i, (False, False, False))  # might be a Theano bug
    s_initial = T.zeros((n_batch, n_cells), dtype="float32")
    h_initial = T.zeros((1, n_batch, n_out), dtype="float32")
    h_initial = T.patternbroadcast(h_initial, (False, False, False))
    go_backwards = {1:False, -1:True}[direction]
    (s, h), _ = theano.scan(lstm_step,
                            sequences=[x, i], go_backwards=go_backwards,
                            non_sequences=[],
                            outputs_info=[s_initial, h_initial])
    h = h[:, 0]  # remove dummy dimension
    self.act = [h, s]
    h = h[::direction]
    self.make_output(h)

    self.params.update({"sublayer." + name: param for (name, param) in self.sublayer.params.items()})
    if self.out_sublayer:
      self.params.update({"out_sublayer." + name: param for (name, param) in self.out_sublayer.params.items()})


class AssociativeLstmLayer(HiddenLayer):
  """
  Associative Long Short-Term Memory
  http://arxiv.org/abs/1602.03032
  """
  recurrent = True
  layer_class = "associative_lstm"

  def __init__(self, n_out, n_copies, activation="tanh", direction=1, grad_clip=None, **kwargs):
    n_cells = n_out
    assert n_cells % 2 == 0  # complex numbers, split real/imag
    n_complex_cells = n_cells / 2
    # {input,forget,out}-gate have n_complex_cells dim.
    # {input,output}-key for holographic memory have n_cells dim.
    # update u (earlier called net-input) has n_cells dim.
    n_z = n_complex_cells * 3 + n_cells * 2 + n_cells
    # It's a hidden layer, thus this will create the feed forward layer for the LSTM for the input.
    super(AssociativeLstmLayer, self).__init__(n_out=n_z, **kwargs)
    self.set_attr('n_out', n_out)
    self.set_attr('n_copies', n_copies)
    self.set_attr('direction', direction)
    self.set_attr('activation', activation)
    if grad_clip:
      self.set_attr('grad_clip', grad_clip)
      grad_clip = numpy.float32(grad_clip)

    self.W_re = self.add_param(self.create_random_uniform_weights(n=n_out, m=n_z, name="W_re_%s" % self.name))
    static_rng = numpy.random.RandomState(1234)
    def make_permut():
      p = numpy.zeros((n_copies, n_cells), dtype="int32")
      for i in range(n_copies):
        p[i, :n_complex_cells] = static_rng.permutation(n_complex_cells)
        # Same permutation for imaginary part.
        p[i, n_complex_cells:] = p[i, :n_complex_cells] + n_complex_cells
      return T.constant(p)
    P = make_permut()  # (n_copies,n_cells) -> list of indices

    # Some defaults.
    CI, CO, RI, RO = [T.tanh] * 4  # original complex_bound, but tanh works better?
    G = T.nnet.sigmoid

    actf = strtoact(activation)
    if isinstance(actf, list):
      if len(actf) == 4:
        CI, CO, RI, RO = actf
      elif len(actf) == 5:
        CI, CO, RI, RO, G = actf
      else:
        assert False, "invalid number of activation functions: %s, %s, %s" % (len(actf), activation, actf)
    else:
      CI, CO, RI, RO = [actf] * 4  # Not for the gates.

    def lstm_step(z_t, i_t, s_p, h_p, W_re):
      # z_t: current input. (batch,n_z)
      # i_t: 0 or 1 (via index). (batch,)
      # s_p: previous cell state. (batch,n_copies,n_cells)
      # h_p: previous hidden out. (batch,n_out)
      # W_re: recurrent matrix. (n_out,n_z)
      i_t_bc = i_t.dimshuffle(0, 'x')
      z_t += T.dot(h_p, W_re)
      z_t *= i_t_bc
      gates = G(z_t[:, 0:n_complex_cells * 3])
      meminkey = RI(z_t[:, 3 * n_complex_cells:3 * n_complex_cells + n_cells])  # (batch,n_cells)
      memoutkey = RO(z_t[:, 3 * n_complex_cells + n_cells:3 * n_complex_cells + 2 * n_cells])  # (batch,n_cells)
      u = CI(z_t[:, 3 * n_complex_cells + 2 * n_cells:])
      ingate2 = T.tile(gates[:, 0:n_complex_cells], (1, 2))
      forgetgate2 = T.tile(gates[:, n_complex_cells:2 * n_complex_cells], (1, 2))
      outgate2 = T.tile(gates[:, 2 * n_complex_cells:], (1, 2))
      meminkeyP = meminkey[:, P]  # (batch,n_copies,n_cells)
      memoutkeyP = memoutkey[:, P]  # (batch,n_copies,n_cells)
      u_gated = u * ingate2  # (batch,n_cells)
      u_gated_bc = u_gated.dimshuffle(0, 'x', 1)  # (batch,n_copies,n_cells)
      forgetgate2_bc = forgetgate2.dimshuffle(0, 'x', 1)  # (batch,n_copies,n_cells)
      from returnn.theano.util import complex_elemwise_mult
      s_t = complex_elemwise_mult(meminkeyP, u_gated_bc) + s_p * forgetgate2_bc  # (batch,n_copies,n_cells)
      readout_avg = T.mean(complex_elemwise_mult(memoutkeyP, s_t), axis=1)  # (batch,n_cells)
      h_t = CO(readout_avg) * outgate2
      s_t *= i_t.dimshuffle(0, 'x', 'x')
      h_t *= i_t_bc
      if grad_clip:
        s_t = theano.gradient.grad_clip(s_t, -grad_clip, grad_clip)
        h_t = theano.gradient.grad_clip(h_t, -grad_clip, grad_clip)
      return s_t, h_t

    z = self.get_linear_forward_output()  # (n_time,n_batch,n_z)
    n_batch = z.shape[1]
    assert self.W_re.ndim == 2
    # i: (n_time,n_batch)
    i = T.cast(self.index, dtype="float32")  # so that it can run on gpu

    s_initial = T.zeros((n_batch, n_copies, n_cells), dtype="float32")
    h_initial = T.zeros((n_batch, n_out), dtype="float32")
    go_backwards = {1:False, -1:True}[direction]
    (s, h), _ = theano.scan(lstm_step,
                            sequences=[z, i], go_backwards=go_backwards,
                            non_sequences=[self.W_re],
                            outputs_info=[s_initial, h_initial])
    self.act = [h, s]
    h = h[::direction]
    self.make_output(h)


class LstmHalfGatesLayer(HiddenLayer):
  recurrent = True
  layer_class = "lstm_half_gates"

  def __init__(self, n_out, direction=1, activation='tanh', grad_clip=None, **kwargs):
    n_cells = n_out
    assert n_cells % 2 == 0  # complex numbers, split real/imag
    n_complex_cells = n_cells / 2
    # {input,forget,out}-gate have n_complex_cells dim.
    # update u (earlier called net-input) has n_cells dim.
    n_z = n_complex_cells * 3 + n_cells
    # It's a hidden layer, thus this will create the feed forward layer for the LSTM for the input.
    super(LstmHalfGatesLayer, self).__init__(n_out=n_z, **kwargs)
    self.set_attr('n_out', n_out)
    self.set_attr('direction', direction)
    self.set_attr('activation', activation)
    if grad_clip:
      self.set_attr('grad_clip', grad_clip)
      grad_clip = numpy.float32(grad_clip)

    self.W_re = self.add_param(self.create_random_uniform_weights(n=n_out, m=n_z, name="W_re_%s" % self.name))
    from returnn.theano.util import complex_elemwise_mult, complex_bound

    # Some defaults.
    CI, CO = [T.tanh] * 2
    GI, GF, GO = [T.nnet.sigmoid] * 3

    actf = strtoact(activation)
    if isinstance(actf, list):
      if len(actf) == 2:
        CI, CO = actf
      elif len(actf) == 5:
        CI, CO, GI, GF, GO = actf
      else:
        assert False, "invalid number of activation functions: %s, %s, %s" % (len(actf), activation, actf)
    else:
      CI, CO = [actf] * 2  # Not for the gates.

    def lstm_step(z_t, i_t, s_p, h_p, W_re):
      # z_t: current input. (batch,n_z)
      # i_t: 0 or 1 (via index). (batch,)
      # s_p: previous cell state. (batch,n_cells)
      # h_p: previous hidden out. (batch,n_out)
      # W_re: recurrent matrix. (n_out,n_z)
      i_t_bc = i_t.dimshuffle(0, 'x')
      z_t += T.dot(h_p, W_re)
      z_t *= i_t_bc
      ingate = GI(z_t[:, 0:n_complex_cells])
      forgetgate = GF(z_t[:, n_complex_cells:2 * n_complex_cells])
      outgate = GO(z_t[:, 2 * n_complex_cells:3 * n_complex_cells])
      u = CI(z_t[:, 3 * n_complex_cells:])
      ingate2 = T.tile(ingate, (1, 2))
      forgetgate2 = T.tile(forgetgate, (1, 2))
      outgate2 = T.tile(outgate, (1, 2))
      u_gated = u * ingate2  # (batch,n_cells)
      s_t = u_gated + s_p * forgetgate2  # (batch,n_cells)
      h_t = CO(s_t) * outgate2
      s_t *= i_t_bc
      h_t *= i_t_bc
      if grad_clip:
        s_t = theano.gradient.grad_clip(s_t, -grad_clip, grad_clip)
        h_t = theano.gradient.grad_clip(h_t, -grad_clip, grad_clip)
      return s_t, h_t

    z = self.get_linear_forward_output()  # (n_time,n_batch,n_z)
    n_batch = z.shape[1]
    assert self.W_re.ndim == 2
    # i: (n_time,n_batch)
    i = T.cast(self.index, dtype="float32")  # so that it can run on gpu

    s_initial = T.zeros((n_batch, n_cells), dtype="float32")
    h_initial = T.zeros((n_batch, n_out), dtype="float32")
    go_backwards = {1:False, -1:True}[direction]
    (s, h), _ = theano.scan(lstm_step,
                            sequences=[z, i], go_backwards=go_backwards,
                            non_sequences=[self.W_re],
                            outputs_info=[s_initial, h_initial])
    self.act = [h, s]
    h = h[::direction]
    self.make_output(h)


class LstmProjGatesLayer(HiddenLayer):
  recurrent = True
  layer_class = "lstm_proj_gates"

  def __init__(self, n_out, n_gate_proj, direction=1, activation='relu', grad_clip=None, **kwargs):
    n_cells = n_out
    # {input,forget,out}-gate have n_gate_proj dim.
    # update u (earlier called net-input) has n_cells dim.
    n_z = n_gate_proj * 3 + n_cells
    # It's a hidden layer, thus this will create the feed forward layer for the LSTM for the input.
    super(LstmProjGatesLayer, self).__init__(n_out=n_z, **kwargs)
    self.set_attr('n_out', n_out)
    self.set_attr('n_gate_proj', n_gate_proj)
    self.set_attr('direction', direction)
    self.set_attr('activation', activation)
    if grad_clip:
      self.set_attr('grad_clip', grad_clip)
      grad_clip = numpy.float32(grad_clip)

    self.W_re = self.add_param(self.create_random_uniform_weights(n=n_out, m=n_z, name="W_re_%s" % self.name))
    self.W_igate_proj = self.add_param(
      self.create_random_uniform_weights(n=n_gate_proj, m=n_cells, name="W_igate_proj_%s" % self.name))
    self.W_fgate_proj = self.add_param(
      self.create_random_uniform_weights(n=n_gate_proj, m=n_cells, name="W_fgate_proj_%s" % self.name))
    self.W_ogate_proj = self.add_param(
      self.create_random_uniform_weights(n=n_gate_proj, m=n_cells, name="W_ogate_proj_%s" % self.name))

    from ActivationFunctions import relu

    # Some defaults.
    PG = relu
    CI, CO = [T.tanh] * 2
    GI, GF, GO = [T.nnet.sigmoid] * 3

    actf = strtoact(activation)
    if isinstance(actf, list):
      if len(actf) == 3:
        PG, CI, CO = actf
      elif len(actf) == 6:
        PG, CI, CO, GI, GF, GO = actf
      else:
        assert False, "invalid number of activation functions: %s, %s, %s" % (len(actf), activation, actf)
    else:
      PG = actf  # This is what this layer is about.

    def lstm_step(z_t, i_t, s_p, h_p):
      # z_t: current input. (batch,n_z)
      # i_t: 0 or 1 (via index). (batch,)
      # s_p: previous cell state. (batch,n_cells)
      # h_p: previous hidden out. (batch,n_out)
      i_t_bc = i_t.dimshuffle(0, 'x')
      z_t += T.dot(h_p, self.W_re)
      z_t *= i_t_bc

      igate_in = PG(z_t[:, :n_gate_proj])
      fgate_in = PG(z_t[:, n_gate_proj:2 * n_gate_proj])
      ogate_in = PG(z_t[:, 2 * n_gate_proj:3 * n_gate_proj])
      u = CI(z_t[:, 3 * n_gate_proj:])

      igate = GI(T.dot(igate_in, self.W_igate_proj))
      fgate = GF(T.dot(fgate_in, self.W_fgate_proj))
      ogate = GO(T.dot(ogate_in, self.W_ogate_proj))

      s_t = u * igate + s_p * fgate
      h_t = CO(s_t) * ogate
      s_t *= i_t_bc
      h_t *= i_t_bc
      if grad_clip:
        s_t = theano.gradient.grad_clip(s_t, -grad_clip, grad_clip)
        h_t = theano.gradient.grad_clip(h_t, -grad_clip, grad_clip)
      return s_t, h_t

    z = self.get_linear_forward_output()  # (n_time,n_batch,n_z)
    n_batch = z.shape[1]
    assert self.W_re.ndim == 2
    # i: (n_time,n_batch)
    i = T.cast(self.index, dtype="float32")  # so that it can run on gpu

    s_initial = T.zeros((n_batch, n_cells), dtype="float32")
    h_initial = T.zeros((n_batch, n_out), dtype="float32")
    go_backwards = {1:False, -1:True}[direction]
    (s, h), _ = theano.scan(lstm_step,
                            sequences=[z, i], go_backwards=go_backwards,
                            non_sequences=[],
                            outputs_info=[s_initial, h_initial])
    self.act = [h, s]
    h = h[::direction]
    self.make_output(h)


class LstmComplexLayer(HiddenLayer):
  recurrent = True
  layer_class = "lstm_complex"

  def __init__(self, n_out, direction=1, activation='tanh', use_complex="1:1:1:1", grad_clip=None, **kwargs):
    n_cells = n_out
    assert n_cells % 2 == 0  # complex numbers, split real/imag
    n_complex_cells = n_cells / 2
    # {input,forget,out}-gate have n_cells dim.
    # update u (earlier called net-input) has n_cells dim.
    n_z = n_cells * 3 + n_cells
    # It's a hidden layer, thus this will create the feed forward layer for the LSTM for the input.
    super(LstmComplexLayer, self).__init__(n_out=n_z, **kwargs)
    self.set_attr('n_out', n_out)
    self.set_attr('direction', direction)
    self.set_attr('activation', activation)
    self.set_attr('use_complex', use_complex)
    if grad_clip:
      self.set_attr('grad_clip', grad_clip)
      grad_clip = numpy.float32(grad_clip)
    use_complex_t = map(int, use_complex.split(":"))
    assert len(use_complex_t) == 4
    from returnn.theano.util import complex_dot, complex_elemwise_mult

    n_re = n_z
    rec_dot = T.dot
    if use_complex_t[0]:  # Complex dot multiplication.
      n_re /= 2
      rec_dot = complex_dot
    self.W_re = self.add_param(self.create_random_uniform_weights(n=n_out, m=n_re, name="W_re_%s" % self.name))

    # Some defaults.
    CI, CO = [T.tanh] * 2
    GI, GF, GO = [T.nnet.sigmoid] * 3
    igate_mult, fgate_mult, ogate_mult = [T.mul] * 3

    actf = strtoact(activation)
    if isinstance(actf, list):
      if len(actf) == 2:
        CI, CO = actf
      elif len(actf) == 5:
        CI, CO, GI, GF, GO = actf
      else:
        assert False, "invalid number of activation functions: %s, %s, %s" % (len(actf), activation, actf)
    else:
      CI, CO = [actf] * 2

    if use_complex_t[1]: igate_mult = complex_elemwise_mult
    if use_complex_t[2]: fgate_mult = complex_elemwise_mult
    if use_complex_t[3]: ogate_mult = complex_elemwise_mult

    def lstm_step(z_t, i_t, s_p, h_p):
      # z_t: current input. (batch,n_z)
      # i_t: 0 or 1 (via index). (batch,)
      # s_p: previous cell state. (batch,n_cells)
      # h_p: previous hidden out. (batch,n_out)
      i_t_bc = i_t.dimshuffle(0, 'x')

      z_t += rec_dot(h_p, self.W_re)
      z_t *= i_t_bc

      igate = GI(z_t[:, :n_cells])
      fgate = GF(z_t[:, n_cells:2 * n_cells])
      ogate = GO(z_t[:, 2 * n_cells:3 * n_cells])
      u = CI(z_t[:, 3 * n_cells:])

      s_t = igate_mult(u, igate) + fgate_mult(s_p, fgate)
      h_t = ogate_mult(CO(s_t), ogate)
      s_t *= i_t_bc
      h_t *= i_t_bc
      if grad_clip:
        s_t = theano.gradient.grad_clip(s_t, -grad_clip, grad_clip)
        h_t = theano.gradient.grad_clip(h_t, -grad_clip, grad_clip)
      return s_t, h_t

    z = self.get_linear_forward_output()  # (n_time,n_batch,n_z)
    n_batch = z.shape[1]
    assert self.W_re.ndim == 2
    # i: (n_time,n_batch)
    i = T.cast(self.index, dtype="float32")  # so that it can run on gpu

    s_initial = T.zeros((n_batch, n_cells), dtype="float32")
    h_initial = T.zeros((n_batch, n_out), dtype="float32")
    go_backwards = {1:False, -1:True}[direction]
    (s, h), _ = theano.scan(lstm_step,
                            sequences=[z, i], go_backwards=go_backwards,
                            non_sequences=[],
                            outputs_info=[s_initial, h_initial])
    self.act = [h, s]
    h = h[::direction]
    self.make_output(h)


class ActLstmLayer(HiddenLayer):
  """
  Adaptive Computation Time for Recurrent Neural Networks, Graves
  """
  recurrent = True
  layer_class = "act_lstm"

  def __init__(self, n_out, n_max_calc_steps=10,
               time_penalty=0.01, time_penalty_type="linear_p",
               total_halt_penalty=0.0, total_halt_penalty_type="inv",
               direction=1, eps=0.01, grad_clip=None,
               unroll_inner_scan=False, **kwargs):
    n_out_orig = n_out
    n_out += 1  # the halting unit
    n_cells = n_out
    # {input,forget,out}-gate + update
    n_z = n_cells * 4
    # It's a hidden layer, thus this will create the feed forward layer for the LSTM for the input.
    super(ActLstmLayer, self).__init__(n_out=n_z, **kwargs)
    self.set_attr('n_out', n_out_orig)
    self.set_attr('n_max_calc_steps', n_max_calc_steps)
    self.set_attr('time_penalty', time_penalty)
    self.set_attr('time_penalty_type', time_penalty_type)
    self.set_attr('total_halt_penalty', total_halt_penalty)
    self.set_attr('total_halt_penalty_type', total_halt_penalty_type)
    self.set_attr('direction', direction)
    if grad_clip:
      self.set_attr('grad_clip', grad_clip)
      grad_clip = numpy.float32(grad_clip)
    self.set_attr('eps', eps)
    if unroll_inner_scan:
      self.set_attr('unroll_inner_scan', unroll_inner_scan)
    self.W_re = self.add_param(self.create_random_uniform_weights(n=n_out, m=n_z, name="W_re_%s" % self.name))
    self.W_delay = self.add_param(self.create_random_uniform_weights(n=1, m=n_z, p=n_out + n_z + 1, name="W_delay_%s" % self.name))

    z = self.get_linear_forward_output()  # (n_time,n_batch,n_z)
    n_batch = z.shape[1]

    CI, CO = [T.tanh] * 2
    G = T.nnet.sigmoid

    assert 0 < eps < 1
    hs_limit = numpy.float32(1.0 - eps)
    assert 0 < hs_limit < 1

    def lstm_step(z_t, i_t, s_p, h_p):
      # z_t: current input. (batch,n_z)
      # i_t: 0 or 1 (via index). (batch,)
      # s_p: previous cell state. (batch,n_cells)
      # h_p: previous hidden out. (batch,n_out)
      i_t_bc = i_t.dimshuffle(0, 'x')  # batch,x
      z_t += T.dot(h_p, self.W_re)
      z_t *= i_t_bc

      gates = G(z_t[:, :n_cells * 3])
      u = CI(z_t[:, 3 * n_cells:])
      igate = gates[:, :n_cells]
      fgate = gates[:, n_cells:2 * n_cells]
      ogate = gates[:, 2 * n_cells:]

      s_t = u * igate + s_p * fgate
      h_t = CO(s_t) * ogate
      s_t *= i_t_bc
      h_t *= i_t_bc
      if grad_clip:
        s_t = theano.gradient.grad_clip(s_t, -grad_clip, grad_clip)
        h_t = theano.gradient.grad_clip(h_t, -grad_clip, grad_clip)
      return s_t, h_t

    def inner_step(s_p, h_p, hs_p, delay_p, z_t, i_t):
      delay_t = delay_p + numpy.float32(1)
      not_last_state = T.lt(delay_t, numpy.float32(n_max_calc_steps - 0.1))
      z_t += T.dot(delay_p.dimshuffle('x', 'x'), self.W_delay)  # (n_batch,n_z)
      s_t, h_t = lstm_step(z_t, i_t, s_p, h_p)
      # We assume that tanh was applied to h_t.
      hp_t = (h_t[:, -1] + numpy.float32(1)) / numpy.float32(2)  # halting unit, (n_batch)
      hs_t = hs_p + hp_t  # (n_batch)
      # p_t can have 4 states:
      #   1) = hp_t,      if hs_t < 1 - eps and not last state
      #   2) = 1 - hs_p,  if hs_t < 1 - eps and last state
      #   3) = 1 - hs_p,  if hs_t >= 1 - eps and hs_p < 1 - eps  (R, remainder)
      #   4) = 0,         otherwise
      p_t = T.switch(hs_t < hs_limit,
                     T.switch(not_last_state,
                              hp_t,
                              numpy.float32(1) - hs_p),
                     T.switch(hs_p < hs_limit,
                              numpy.float32(1) - hs_p,
                              numpy.float32(0)))
      stop_cond = T.min((hs_p >= hs_limit) * i_t)
      if time_penalty_type == "linear":
        # Note: This is not the time penalty as in the paper.
        # However, I think the one in the paper is not smooth.
        # This one is even simpler and should be differentiable.
        tpi_t = delay_t * hp_t
      elif time_penalty_type == "linear_p":
        tpi_t = delay_t * p_t  # this yields actually the expected value of N(t)
      elif time_penalty_type == "sqrt":
        tpi_t = T.sqrt(delay_t) * hp_t
      else:
        assert False, "invalid time_penalty_type %r" % time_penalty_type
      if total_halt_penalty:
        if total_halt_penalty_type == "inv":
          tpi_t += T.inv(hp_t) * numpy.float32(total_halt_penalty)
        elif total_halt_penalty_type == "linear":
          tpi_t -= hp_t * numpy.float32(total_halt_penalty)
        else:
          assert False, "invalid total_halt_penalty_type %r" % total_halt_penalty_type
      return [s_t, h_t, hs_t, p_t, tpi_t, delay_t], {}, theano.scan_module.until(stop_cond)

    def outer_step(z_t, i_t, s_p, h_p):
      n_batch = z_t.shape[0]
      delay_initial = T.zeros((), dtype="float32")  # counter
      hs_initial = T.zeros((n_batch,), dtype="float32")  # sum(hp_t), the halting units summed up
      p_initial = None  # halting probability. hp_t or R or 0
      tpi_initial = None  # inner time penalty
      inner_scan = theano.scan
      if unroll_inner_scan:
        from returnn.theano.util import unroll_scan as inner_scan
      (s, h, hs, p, tpi, _), _ = inner_scan(
        inner_step,
        n_steps=n_max_calc_steps,
        outputs_info=[s_p, h_p, hs_initial, p_initial, tpi_initial, delay_initial],
        non_sequences=[z_t, i_t])
      p_bc = p.dimshuffle(0, 1, 'x')  # (calcstep,n_batch,x)
      s_t = T.sum(s * p_bc, axis=0)
      h_t = T.sum(h * p_bc, axis=0)
      return s_t, h_t, T.sum(tpi)

    # i: (n_time,n_batch)
    i = T.cast(self.index, dtype="float32")  # so that it can run on gpu
    s_initial = T.zeros((n_batch, n_cells), dtype="float32")  # lstm cell state
    h_initial = T.zeros((n_batch, n_out), dtype="float32")  # lstm hidden out
    tp_initial = None  # time penalty
    go_backwards = {1:False, -1:True}[direction]
    (s, h, tp), _ = theano.scan(outer_step,
                                sequences=[z, i], go_backwards=go_backwards,
                                non_sequences=[],
                                outputs_info=[s_initial, h_initial, tp_initial])
    h = h[:, :, :-1]  # remove halting unit
    self.act = [h, s]
    h = h[::direction]
    self.make_output(h)
    self.constraints += T.sum(tp) * numpy.float32(time_penalty)


class GRULayer(RecurrentLayer):
  layer_class = "gru"

  def __init__(self, n_out, encoder = None, mode = "cho", n_dec = 0, **kwargs):
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
    for s in range(self.attrs['sampling']):
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
  layer_class = "sru"

  def __init__(self, n_out, encoder = None, psize = 0, pact = 'relu', pdepth = 1, carry_time = False, n_dec = 0, **kwargs):
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
      for i in range(1, pdepth):
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
    for s in range(self.attrs['sampling']):
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
  layer_class = "sra"

  def __init__(self, n_out, encoder = None, psize = 0, pact = 'relu', pdepth = 1, n_dec = 0, **kwargs):
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
      for i in range(1, pdepth):
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
    for s in range(self.attrs['sampling']):
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
