
import numpy
from theano import tensor as T
import theano
from NetworkHiddenLayer import HiddenLayer
from NetworkBaseLayer import Container, Layer
from ActivationFunctions import strtoact
from math import sqrt
from OpLSTM import LSTMOpInstance
from FastLSTM import LSTMOp2Instance

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


class Unit(Container):
  def __init__(self, n_units, depth, n_in, n_out, n_re, n_act):
    # number of cells, depth, cell fan in, cell fan out, recurrent fan in, number of outputs
    self.n_units, self.depth, self.n_in, self.n_out, self.n_re, self.n_act = n_units, depth, n_in, n_out, n_re, n_act
    self.slice = T.constant(self.n_units)
    self.params = {}

  def scan(self, step, x, z, non_sequences, i, outputs_info, W_re, W_in, b, go_backwards = False, truncate_gradient = -1):
    try:
      xc = z if not x else T.concatenate([s.output for s in x], axis = -1)
    except Exception:
      xc = z if not x else T.concatenate(x, axis = -1)
    outputs, _ = theano.scan(step,
                             #strict = True,
                             truncate_gradient = truncate_gradient,
                             go_backwards = go_backwards,
                             sequences = [xc,z,i],
                             non_sequences = non_sequences,
                             outputs_info = outputs_info)
    return outputs


class VANILLA(Unit):
  def __init__(self, n_units, depth):
    super(VANILLA, self).__init__(n_units, depth, n_units, n_units, n_units, 1)

  def step(self, x_t, z_t, z_p, h_p):
    return [ T.tanh(z_t + z_p) ]


class LSTME(Unit):
  def __init__(self, n_units, depth):
    super(LSTME, self).__init__(n_units, depth, n_units * 4, n_units, n_units * 4, 2)

  def step(self, x_t, z_t, z_p, h_p, s_p):
    CI, GI, GF, GO, CO = [T.tanh, T.nnet.sigmoid, T.nnet.sigmoid, T.nnet.sigmoid, T.tanh]
    z = z_t + z_p
    u_t = GI(z[:,0 * self.slice:1 * self.slice])
    r_t = GF(z[:,1 * self.slice:2 * self.slice])
    b_t = GO(z[:,2 * self.slice:3 * self.slice])
    a_t = CI(z[:,3 * self.slice:])
    s_t = (a_t * u_t) # + s_p * r_t)
    h_t = CO(s_t) * b_t
    #return [ h_t, theano.gradient.grad_clip(s_t, -50, 50) ]
    return [ h_t, s_t ]


class LSTM(Unit):
  def __init__(self, n_units, depth):
    super(LSTM, self).__init__(n_units, depth, n_units * 4, n_units, n_units * 4, 2)

  def scan(self, step, x, z, non_sequences, i, outputs_info, W_re, W_in, b, go_backwards = False, truncate_gradient = -1):
    try:
      XS = [S.output[::-(2 * go_backwards - 1)] for S in x]
    except Exception:
      XS = [S[::-(2 * go_backwards - 1)] for S in x]
    result = LSTMOp2Instance(*([W_re, outputs_info[1], b, i[::-(2 * go_backwards - 1)]] + XS + W_in))
    return [ result[0], result[2].dimshuffle('x',0,1) ]


class LSTMP(Unit):
  def __init__(self, n_units, depth):
    super(LSTMP, self).__init__(n_units, depth, n_units * 4, n_units, n_units * 4, 2)

  def scan(self, step, x, z, non_sequences, i, outputs_info, W_re, W_in, b, go_backwards = False, truncate_gradient = -1):
    result = LSTMOpInstance(z[::-(2 * go_backwards - 1)], W_re, outputs_info[1], i[::-(2 * go_backwards - 1)])
    return [ result[0], result[2].dimshuffle('x',0,1) ]


class GRU(Unit):
  def __init__(self, n_units, depth):
    super(GRU, self).__init__(n_units, depth, n_units * 3, n_units, n_units * 2, 1)
    l = sqrt(6.) / sqrt(n_units * 3)
    rng = numpy.random.RandomState(1234)
    if depth > 1: values = numpy.asarray(rng.uniform(low=-l, high=l, size=(n_units, depth, n_units)), dtype=theano.config.floatX)
    else: values = numpy.asarray(rng.uniform(low=-l, high=l, size=(n_units, n_units)), dtype=theano.config.floatX)
    self.W_reset = theano.shared(value=values, borrow=True, name = "W_reset")
    self.params['W_reset'] = self.W_reset

  def step(self, x_t, z_t, z_p, h_p):
    CI, GR, GU = [T.tanh, T.nnet.sigmoid, T.nnet.sigmoid]
    u_t = GU(z_t[:,:self.slice] + z_p[:,:self.slice])
    r_t = GR(z_t[:,self.slice:2*self.slice] + z_p[:,self.slice:2*self.slice])
    h_c = CI(z_t[:,2*self.slice:] + self.dot(r_t * h_p, self.W_reset))
    return [ u_t * h_p + (1 - u_t) * h_c ]


class SRU(Unit):
  def __init__(self, n_units, depth):
    super(SRU, self).__init__(n_units, depth, n_units * 3, n_units, n_units * 3, 1)

  def step(self, x_t, z_t, z_p, h_p):
    CI, GR, GU = [T.tanh, T.nnet.sigmoid, T.nnet.sigmoid]
    u_t = GU(z_t[:,:self.slice] + z_p[:,:self.slice])
    r_t = GR(z_t[:,self.slice:2*self.slice] + z_p[:,self.slice:2*self.slice])
    h_c = CI(z_t[:,2*self.slice:] + r_t * z_p[:,2*self.slice:])
    return [ u_t * h_p + (1 - u_t) * h_c ]


class RecurrentUnitLayer(Layer):
  recurrent = True
  layer_class = "rec"

  def __init__(self,
               n_out, # number of cells
               direction = 1, # forward (1), backward (-1) or bidirectional (0)
               truncation = -1, # truncate the gradient after this amount of time steps
               sampling = 1, # scan every nth frame only
               encoder = None, # list of encoder layers
               psize = 0, # size of projection
               pact = 'relu', # activation of projection
               pdepth = 1, # depth of projection
               carry_time = False, # carry gate on inputs
               unit = 'lstm', # cell type
               n_dec = 0, # number of time steps to decode
               attention = False, # soft-attention
               base = None,
               lm = False, # language model
               depth = 1,
               **kwargs):
    # if on cpu, we need to fall back to the theano version of the LSTM Op
    unit_given = unit
    if str(theano.config.device).startswith('cpu') and (unit == 'lstm' or unit == 'lstmp'):
      #print "%s: falling back to theano cell implementation" % kwargs['name']
      unit = "lstme"
    unit = eval(unit.upper())(n_out, depth)
    kwargs.setdefault("n_out", unit.n_out)
    kwargs.setdefault("depth", depth)
    kwargs.pop("activation", None)
    super(RecurrentUnitLayer, self).__init__(**kwargs)
    self.set_attr('from', ",".join([s.name for s in self.sources]))
    self.set_attr('n_out', n_out)
    self.set_attr('unit', unit_given.encode("utf8"))
    self.set_attr('psize', psize)
    self.set_attr('pact', pact)
    self.set_attr('pdepth', pdepth)
    self.set_attr('truncation', truncation)
    self.set_attr('sampling', sampling)
    self.set_attr('direction', direction)
    self.set_attr('carry_time', carry_time)
    self.set_attr('attention', attention)
    self.set_attr('lm', lm)
    if encoder:
      self.set_attr('encoder', ",".join([e.name for e in encoder]))
    if base:
      self.set_attr('base', ",".join([b.name for b in base]))
    else:
      base = encoder
    pact = strtoact(pact)
    if n_dec:
      self.set_attr('n_dec', n_dec)
    if direction == 0:
      self.depth *= 2
    # initialize recurrent weights
    W_re = None
    if unit.n_re > 0:
      n_re = unit.n_out
      if self.attrs['consensus'] == 'flat':
        n_re *= self.depth
      self.Wp = []
      if psize:
        self.Wp = [ self.add_param(self.create_random_uniform_weights(n_re, psize, n_re + psize, name = "Wp_0_%s"%self.name, depth=1), name = "Wp_0_%s"%self.name) ]
        for i in xrange(1, pdepth):
          self.Wp.append(self.add_param(self.create_random_uniform_weights(psize, psize, psize + psize, name = "Wp_%d_%s"%(i, self.name), depth=1), name = "Wp_%d_%s"%(i, self.name)))
        W_re = self.create_random_uniform_weights(psize, unit.n_re, psize + unit.n_re, name="W_re_%s" % self.name)
      else:
        W_re = self.create_random_uniform_weights(n_re, unit.n_re, n_re + unit.n_re + unit.n_in, name="W_re_%s" % self.name)
      self.add_param(W_re, W_re.name)
    # initialize forward weights
    if self.depth > 1:
      value = numpy.zeros((self.depth, unit.n_in), dtype = theano.config.floatX)
    else:
      value = numpy.zeros((unit.n_in, ), dtype = theano.config.floatX)
      value[unit.n_out:2*unit.n_out] = -1
    self.b = theano.shared(value=value, borrow=True, name="b_%s"%self.name) #self.create_bias()
    self.params["b_%s"%self.name] = self.b
    self.W_in = []
    for s in self.sources:
      W = self.create_random_uniform_weights(s.attrs['n_out'], unit.n_in,
                                             s.attrs['n_out'] + unit.n_in + unit.n_re,
                                             name="W_in_%s_%s" % (s.name, self.name), depth = 1)
      self.W_in.append(W)
      self.params["W_in_%s_%s" % (s.name, self.name)] = W
    # make input
    z = self.b if self.W_in else 0
    for x_t, m, W in zip(self.sources, self.masks, self.W_in):
      if x_t.attrs['sparse']:
        z += W[T.cast(x_t.output[:,:,0], 'int32')]
      elif m is None:
        z += T.dot(x_t.output, W)
        #z, _ = theano.foldl(lambda x, z: z + T.dot(x,W), sequences = [x_t.output], outputs_info = [z])
      else:
        z += self.dot(self.mass * m * x_t.output, W)
    if self.depth > 1:
      z = z.dimshuffle(0,1,'x',2).repeat(self.depth, axis=2)
    num_batches = self.index.shape[1]
    if direction == 0:
      z = T.set_subtensor(z[:,:,depth:,:], z[::-1,:,:depth,:])
      #q = T.alloc(numpy.cast[theano.config.floatX](0), z.shape[0], num_batches*2, z.shape[2])
      #z = z.repeat(2, axis = 1)
      #q = T.set_subtensor(q[:,:num_batches,:], z)
      #q = T.set_subtensor(q[:,num_batches:,:], z)
      #z = T.set_subtensor(z[:,num_batches:,:], z[::-1,:num_batches,:])
      #num_batches *= 2
      #z = q
    if carry_time:
      assert sum([s.attrs['n_out'] for s in self.sources]) == self.attrs['n_out'], "input / output dimensions do not match in %s. input %d, output %d" % (self.name, sum([s.attrs['n_out'] for s in self.sources]), self.attrs['n_out'])
      W_cr = self.add_param(self.create_random_uniform_weights(self.attrs['n_out'], self.attrs['n_out'], name='W_carry_%s'%self.name), name='W_carry_%s'%self.name)

    non_sequences = []
    if self.attrs['attention']:
      assert encoder, "attention networks are only defined for decoder networks"
      n_in = 0 #numpy.sum([s.attrs['n_out'] for s in self.sources])
      src = []
      for e in base:
        src += [s.output for s in e.sources]
        n_in += sum([s.attrs['n_out'] for s in e.sources])
      self.xc = T.concatenate(src, axis=-1)
      l = sqrt(6.) / sqrt(self.attrs['n_out'] + n_in)

      values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(n_in, 1)), dtype=theano.config.floatX)
      self.W_att_xc = theano.shared(value=values, borrow=True, name = "W_att_xc")
      self.add_param(self.W_att_xc, name = "W_att_xc")
      values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(self.attrs['n_out'], 1)), dtype=theano.config.floatX)
      self.W_att_re = theano.shared(value=values, borrow=True, name = "W_att_re")
      self.add_param(self.W_att_re, name = "W_att_re")
      self.zc = T.dot(self.xc, self.W_att_xc).reshape((self.xc.shape[0], self.xc.shape[1]))

      #values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(n_in + self.attrs['n_out'], 1)), dtype=theano.config.floatX)
      #self.W_att = theano.shared(value=values, borrow=True, name = "W_att")
      #self.add_param(self.W_att, name = "W_att")
      values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(n_in, self.attrs['n_out'] * 4)), dtype=theano.config.floatX)
      self.W_att_in = theano.shared(value=values, borrow=True, name = "W_att_in")
      self.add_param(self.W_att_in, name = "W_att_in")

      non_sequences += [self.xc, self.zc]

    if self.attrs['lm']:
      if not 'target' in self.attrs:
        self.attrs['target'] = 'classes'
      l = sqrt(6.) / sqrt(unit.n_out + self.y_in[self.attrs['target']].n_out)
      values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(unit.n_out, self.y_in[self.attrs['target']].n_out)), dtype=theano.config.floatX)
      self.W_lm_in = theano.shared(value=values, borrow=True, name = "W_lm_in")
      self.add_param(self.W_lm_in, name = "W_lm_in")
      l = sqrt(6.) / sqrt(unit.n_in + self.y_in[self.attrs['target']].n_out)
      values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(self.y_in[self.attrs['target']].n_out, unit.n_in)), dtype=theano.config.floatX)
      self.W_lm_out = theano.shared(value=values, borrow=True, name = "W_lm_out")
      self.add_param(self.W_lm_out, name = "W_lm_out")

      srng = theano.tensor.shared_randomstreams.RandomStreams(self.rng.randint(1234))
      lmmask = T.cast(srng.binomial(n=1, p=0.05, size=self.index.shape), theano.config.floatX).dimshuffle(0,1,'x').repeat(unit.n_in,axis=2) * int(self.train_flag)
      #lmflag = T.any(int(self.train_flag) * self.y_in[self.attrs['target']].reshape(self.index.shape), axis=0) # B

    self.out_dec = self.index.shape[0]
    #if encoder and 'n_dec' in encoder[0].attrs:
    #  self.out_dec = encoder[0].out_dec
    # scan over sequence
    for s in xrange(self.attrs['sampling']):
      index = self.index[s::self.attrs['sampling']]
      sequences = z
      sources = self.sources
      if encoder:
        n_dec = self.out_dec
        if 'n_dec' in self.attrs:
          n_dec = self.attrs['n_dec']
          index = T.alloc(numpy.cast[numpy.int8](1), n_dec, encoder[0].index.shape[1])
        #outputs_info = [ T.alloc(numpy.cast[theano.config.floatX](0), num_batches, unit.n_out) for i in xrange(unit.n_act) ]
        #offset = 0
        #for i in xrange(len(encoder)):
        #  for j in xrange(unit.n_act):
        #    outputs_info[j] = T.set_subtensor(outputs_info[j][:,offset:offset+encoder[i].attrs['n_out']], encoder[i].act[j][-1])
        #  offset += encoder[i].attrs['n_out']
        outputs_info = [ T.concatenate([e.act[i][-1] for e in encoder], axis = -1) for i in xrange(unit.n_act) ]
        if len(self.W_in) == 0:
          if self.depth == 1:
            if self.attrs['lm']:
              y = self.y_in[self.attrs['target']] #.reshape(self.index.shape)
              n_cls = self.y_in[self.attrs['target']].n_out
              y_t = self.W_lm_out[y].reshape((index.shape[0],index.shape[1],unit.n_in))[:-1] # (T-1)BD
              sequences = T.concatenate([self.W_lm_out[0].dimshuffle('x','x',0).repeat(self.index.shape[1],axis=1), y_t], axis=0) * lmmask
              outputs_info.append(T.eye(n_cls, 1).flatten().dimshuffle('x',0).repeat(index.shape[1],0))
            else:
              sequences = T.alloc(numpy.cast[theano.config.floatX](0), n_dec, num_batches, unit.n_in)
          else:
            sequences = T.alloc(numpy.cast[theano.config.floatX](0), n_dec, num_batches, self.depth, unit.n_in)
      else:
        if self.depth == 1:
          outputs_info = [ T.alloc(numpy.cast[theano.config.floatX](0), num_batches, unit.n_out) for i in xrange(unit.n_act) ]
        else:
          outputs_info = [ T.alloc(numpy.cast[theano.config.floatX](0), num_batches, self.depth, unit.n_out) for i in xrange(unit.n_act) ]

      def step(x_t, z_t, i_t, *args):
        if self.attrs['attention']:
          xc = args[-2]
          zc = args[-1]
          args = args[:-2]
        if self.attrs['lm']:
          c_p = args[-1]
          args = args[:-1]
        h_p = args[0]
        if self.depth == 1:
          i = T.outer(i_t, T.alloc(numpy.cast['float32'](1), n_out))
        else:
          h_p = self.make_consensus(h_p, axis = 1)
          i = T.outer(i_t, T.alloc(numpy.cast['float32'](1), n_out)).dimshuffle(0, 'x', 1).repeat(self.depth, axis=1)
        for W in self.Wp:
          h_p = pact(T.dot(h_p, W))
        if self.attrs['lm']:
          h_e = T.exp(T.dot(h_p, self.W_lm_in))
          c_t = h_e / T.sum(h_e, axis=1, keepdims=True)
          #z_t += T.dot(c_p, self.W_lm_out) * T.all(T.eq(z_t,0),axis=1,keepdims=True)
          z_t += self.W_lm_out[T.argmax(c_p,axis=1)] * T.all(T.eq(z_t,0),axis=1,keepdims=True)
        if not self.W_in:
          z_t += self.b
        z_p = self.dot(h_p, W_re)
        if self.depth > 1:
          assert False # this is broken
          sargs = [arg.dimshuffle(0,1,2) for arg in args]
          act = [ act.dimshuffle(0,2,1) for act in unit.step(x_t.dimshuffle(1,0), z_t.dimshuffle(0,2,1), z_p.dimshuffle(0,2,1), *sargs) ]
        else:
          if self.attrs['attention'] and unit_given != 'lstm' and unit_given != 'lstmp':
            #f_t = T.dot(T.concatenate([h_p.dimshuffle('x',0,1).repeat(xc.shape[0], axis=0),xc], axis = 2), self.W_att).reshape((xc.shape[0],h_p.shape[0]))
            f_t = zc + T.dot(h_p, self.W_att_re).flatten() # (time,batch)
            #f_t = z_t + T.dot(h_p, self.W_att_re)
            #f_t = T.dot(T.concatenate([z_p.dimshuffle('x',0,1).repeat(self.xc.shape[0], axis=0),self.xc], axis = 2), self.W_attention).reshape((self.xc.shape[0],z_p.shape[0])).dimshuffle(1,0)
            f_e = T.exp(f_t)
            w_t = (f_e / T.sum(f_e, axis=0)).dimshuffle(0,1,'x') # T.nnet.softmax gives weird results when cudnn is installed
            #w_t = T.tanh(f_t).dimshuffle(0,1,'x') #T.nnet.softmax(f_t.dimshuffle(1, 0)).dimshuffle(1,0,'x')
            #w_t = T.nnet.softmax(f_t.dimshuffle(1, 0)).dimshuffle(1,0,'x')
            #w_t = (f_t / f_t.norm(L=1,axis=0)).dimshuffle(0,1,'x') #T.nnet.sigmoid(f_t).dimshuffle(0,1,'x')
            #w_t = f_t.dimshuffle(0,1,'x')
            #z_t = T.dot(self.xc.dimshuffle(1,2,0), w_t).dimshuffle(2,0,1).reshape(z_p.shape)
            z_t += T.dot(T.sum(xc * w_t, axis=0, keepdims=False), self.W_att_in) #T.tensordot(xc.dimshuffe(2,1,0), w_t, [[2], [2]]) # (batch, dim)
            #z_t = T.dot(xc.dimshuffle(1,2,0), w_t).reshape(z_t.shape)
          act = unit.step(x_t, z_t, z_p, *args)
        if carry_time:
          c_t = T.nnet.sigmoid(self.dot(x_t, W_cr))
          for j in xrange(unit.n_act):
            act[j] = c_t * act[j] + (1 - c_t) * x_t
        return [ act[j] * i + args[j] * (1 - i) for j in xrange(unit.n_act) ] + ([c_t] if self.attrs['lm'] else [])

      index_f = T.cast(index, theano.config.floatX)
      outputs = unit.scan(step,
                          sources,
                          sequences[s::self.attrs['sampling']],
                          non_sequences,
                          index_f,
                          outputs_info,
                          W_re,
                          self.W_in,
                          self.b,
                          direction == -1,
                          self.attrs['truncation'])

      if not isinstance(outputs, list):
        outputs = [outputs]

      if self.attrs['lm'] and self.train_flag:

        #self.y_m = outputs[-1].reshape((outputs[-1].shape[0]*outputs[-1].shape[1],outputs[-1].shape[2])) # (TB)C
        j = (self.index[:-1].flatten() > 0).nonzero() # (TB)
        #y_f = T.extra_ops.to_one_hot(T.reshape(self.y_in[self.attrs['target']], (self.y_in[self.attrs['target']].shape[0] * self.y_in[self.attrs['target']].shape[1]), ndim=1), n_cls) # (TB)C
        #y_t = T.dot(T.extra_ops.to_one_hot(y,n_cls), self.W_lm_out).reshape((index.shape[0],index.shape[1],unit.n_in))[:-1] # TBD
        #self.constraints += -T.sum(T.log(self.y_m[j,y_f[j]]))
        #self.constraints += T.mean(T.sqr(self.y_m[j] - y_f[j]))

        h_y = (self.y_in[self.attrs['target']].reshape(index.shape)).flatten()
        h_e = T.dot(outputs[0][::direction or 1], self.W_lm_in)
        h_f = T.exp(h_e.reshape((h_e.shape[0]*h_e.shape[1],h_e.shape[2])))[j]
        self.constraints += self.index.shape[0] * T.sum(-T.log((h_f / T.sum(h_f,axis=1,keepdims=True))[:,h_y[j]]))
        #nll, pcx = T.nnet.crossentropy_softmax_1hot(x=h_f[j,self.y_in[self.attrs['target']][j]], y_idx=)
        #self.constraints += T.sum(nll)
        outputs = outputs[:-1]
      if self.attrs['sampling'] > 1:
        if s == 0:
          #self.act = [ T.repeat(act, self.attrs['sampling'], axis = 0)[:self.sources[0].output.shape[0]] for act in outputs ]
          self.act = [  T.alloc(numpy.cast['float32'](0), self.index.shape[0], self.index.shape[1], n_out) for act in outputs ]
        self.act = [ T.set_subtensor(tot[s::self.attrs['sampling']], act) for tot,act in zip(self.act, outputs) ]
      else:
        self.act = outputs
    self.make_output(self.act[0][::direction or 1])
    self.params.update(unit.params)
