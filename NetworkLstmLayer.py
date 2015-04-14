
import numpy
from theano import tensor as T
import theano
from NetworkRecurrentLayer import RecurrentLayer


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
      W_proj = self.create_random_uniform_weights(n_out, n_re, n_in + n_out + n_re, "W_proj_%s"%self.name)
      self.W_proj.set_value(W_proj.get_value())
    W_re = self.create_random_uniform_weights(n_re, n_out * 3 + n_re, n_in + n_re + n_out * 3 + n_re, "W_re_%s"%self.name)
    self.W_re.set_value(W_re.get_value())
    for s, W in zip(sources, self.W_in):
      W.set_value(self.create_random_uniform_weights(s.attrs['n_out'], n_out * 3 + n_re, s.attrs['n_out'] + n_out  + n_out * 3 + n_re, "W_in_%s_%s"%(s.name,self.name)).get_value(borrow=True, return_internal_type=True), borrow = True)
    self.o.set_value(numpy.ones((n_out,), dtype='int8')) #TODO what is this good for?
    if projection:
      self.set_attr('n_out', projection)
    else:
      self.set_attr('n_out', self.attrs['n_out'] / 4)
    if sharpgates == 'global': self.sharpness = self.create_random_uniform_weights(3, n_out)
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
      W_proj = self.create_random_uniform_weights(n_out, n_re, n_in + n_out + n_re, "W_proj_%s"%self.name)
      self.W_proj.set_value(W_proj.get_value())
    W_re = self.create_random_uniform_weights(n_re, n_out * 3 + n_re, n_in + n_re + n_out * 3 + n_re, "W_re_%s"%self.name)
    self.W_re.set_value(W_re.get_value())
    for s, W in zip(sources, self.W_in):
      W.set_value(self.create_random_uniform_weights(s.attrs['n_out'], n_out * 3 + n_re, s.attrs['n_out'] + n_out  + n_out * 3 + n_re, "W_in_%s_%s"%(s.name,self.name)).get_value(borrow=True, return_internal_type=True), borrow = True)
    self.o.set_value(numpy.ones((n_out,), dtype='int8'))
    if projection:
      self.set_attr('n_out', projection)
    else:
      self.set_attr('n_out', self.attrs['n_out'] / 4)
    if sharpgates == 'global': self.sharpness = self.create_random_uniform_weights(3, n_out)
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
      W_proj = self.create_random_uniform_weights(n_out, n_re, n_in + n_out + n_re, "W_proj_%s"%self.name)
      self.W_proj.set_value(W_proj.get_value())
    W_re = self.create_random_uniform_weights(n_re, n_out * 3 + n_re, n_in + n_re + n_out * 3 + n_re, "W_re_%s"%self.name)
    self.W_re.set_value(W_re.get_value())
    for s, W in zip(sources, self.W_in):
      W.set_value(self.create_random_uniform_weights(s.attrs['n_out'], n_out * 3 + n_re, s.attrs['n_out'] + n_out  + n_out * 3 + n_re, "W_in_%s_%s"%(s.name,self.name)).get_value(borrow=True, return_internal_type=True), borrow = True)
    self.o.set_value(numpy.ones((n_out,), dtype='int8'))
    if projection:
      self.set_attr('n_out', projection)
    else:
      self.set_attr('n_out', self.attrs['n_out'] / 4)
    if sharpgates == 'global': self.sharpness = self.create_random_uniform_weights(3, n_out)
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
    self.W_re_input = self.add_param(self.create_random_uniform_weights(n_out, n_out, n_in + n_out  + n_out, "W_re_input_%s"%self.name), "W_re_input_%s"%self.name)
    self.W_re_forget = self.add_param(self.create_random_uniform_weights(n_out, n_out, n_in + n_out  + n_out, "W_re_forget_%s"%self.name), "W_re_forget_%s"%self.name)
    self.W_re_output = self.add_param(self.create_random_uniform_weights(n_out, n_out, n_in + n_out  + n_out, "W_re_output_%s"%self.name), "W_re_output_%s"%self.name)

    self.b_input = self.add_param(self.create_bias(n_out, 'b_input'), 'b_input')
    self.b_forget = self.add_param(self.create_bias(n_out, 'b_forget'), 'b_forget')
    self.b_output = self.add_param(self.create_bias(n_out, 'b_output'), 'b_output')

    self.W_input = []
    self.W_forget = []
    self.W_output = []
    for s, W in zip(sources, self.W_in):
      W.set_value(self.create_random_uniform_weights(s.attrs['n_out'], n_out, s.attrs['n_out'] + n_out  + n_out, "W_in_%s_%s"%(s.name,self.name)).get_value())
      self.W_input.append(self.create_random_uniform_weights(s.attrs['n_out'], n_out, s.attrs['n_out'] + n_out  + n_out, "W_input_%s_%s"%(s.name,self.name)))
      self.add_param(self.W_input[-1], "W_input_%s_%s"%(s.name,self.name))
      self.W_forget.append(self.create_random_uniform_weights(s.attrs['n_out'], n_out, s.attrs['n_out'] + n_out  + n_out, "W_forget_%s_%s"%(s.name,self.name)))
      self.add_param(self.W_forget[-1], "W_forget_%s_%s"%(s.name,self.name))
      self.W_output.append(self.create_random_uniform_weights(s.attrs['n_out'], n_out, s.attrs['n_out'] + n_out  + n_out, "W_output_%s_%s"%(s.name,self.name)))
      self.add_param(self.W_output[-1], "W_output_%s_%s"%(s.name,self.name))

    #for s, W in zip(sources, self.W_in):
    #  W.set_value(self.create_uniform_weights(s.attrs['n_out'], n_out * 4, s.attrs['n_out'] + n_out  + n_out * 4, "W_in_%s_%s"%(s.name,self.name)).get_value())
    self.o.set_value(numpy.ones((n_out,), dtype='int8')) #theano.config.floatX))
    #self.set_attr('n_out', self.attrs['n_out'] / 4)
    if sharpgates == 'global': self.sharpness = self.create_random_uniform_weights(3, n_out)
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
    W_re = self.create_random_uniform_weights(n_out, n_out * 4, n_in + n_out  + n_out * 4, "W_re_%s"%self.name)
    self.W_re.set_value(W_re.get_value())
    del self.params['b_%s' % self.name]
    for s, W in zip(sources, self.W_in):
      del self.params["W_in_%s_%s"%(s.name, self.name)]
      #W.set_value(self.create_uniform_weights(s.attrs['n_out'], n_out * 4, s.attrs['n_out'] + n_out  + n_out * 4).get_value())
    self.o.set_value(numpy.ones((n_out,), dtype='int8')) #theano.config.floatX))
    self.set_attr('n_out', self.attrs['n_out'] / 4)
    if sharpgates == 'global': self.sharpness = self.create_random_uniform_weights(3, n_out)
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
    return self.create_random_uniform_weights(n, m * 4, n + m + m * 4), self.create_random_uniform_weights(m, m * 4, n + m + m * 4)

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
    self.act = self.create_random_uniform_weights(n_out, n_cores)
    self.state = self.create_random_uniform_weights(n_out, n_cores)
    n_in = sum([s.attrs['n_out'] for s in sources])
    W_re = self.create_random_uniform_weights(n_out, n_out * (2 + n_cores * 2), n_in + n_out  + n_out * (2 + n_cores * 2))
    self.W_re.set_value(W_re.get_value())
    for s, W in zip(sources, self.W_in):
      W.set_value(self.create_random_uniform_weights(s.attrs['n_out'], n_out * 4, s.attrs['n_out'] + n_out + n_out * (2 + n_cores * 2)).get_value())
    self.o.set_value(numpy.ones((n_out,), dtype=theano.config.floatX))
    self.set_attr('n_out', self.attrs['n_out'] / (2 + n_cores * 2))
    if sharpgates == 'global': self.sharpness = self.create_random_uniform_weights(3, n_out)
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
    return self.create_random_uniform_weights(n, m * 4, n + m + m * 4), self.create_random_uniform_weights(m, m * 4, n + m + m * 4)


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
    if sharpgates == 'global': self.sharpness = self.create_random_uniform_weights(3, n_out)
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
    return self.create_random_uniform_weights(n, m * 4, n + m + m * 4), self.create_random_uniform_weights(m, m * 4, n + m + m * 4)


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
