
import numpy
from theano import tensor as T
import theano
from NetworkHiddenLayer import HiddenLayer
from NetworkBaseLayer import Container, Layer
from ActivationFunctions import strtoact
from math import sqrt
from OpLSTM import LSTMOpInstance
from FastLSTM import LSTMOp2Instance
import RecurrentTransform
import json


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
    self.slice = T.constant(self.n_units, dtype='int32')
    self.params = {}

  def set_parent(self, parent):
    """
    :type parent: RecurrentUnitLayer
    """
    self.parent = parent

  def scan(self, step, x, z, non_sequences, i, outputs_info, W_re, W_in, b, go_backwards=False, truncate_gradient=-1):
    self.outputs_info = outputs_info
    self.non_sequences = non_sequences
    self.W_re = W_re
    self.W_in = W_in
    self.b = b
    self.go_backwards = go_backwards
    self.truncate_gradient = truncate_gradient
    #z = T.inc_subtensor(z[-1 if go_backwards else 0], T.dot(outputs_info[0],W_re))
    try:
      self.xc = z if not x else T.concatenate([s.output for s in x], axis = -1)
    except Exception:
      self.xc = z if not x else T.concatenate(x, axis = -1)

    outputs, _ = theano.scan(step,
                             #strict = True,
                             truncate_gradient = truncate_gradient,
                             go_backwards = go_backwards,
                             sequences = [self.xc,z,i],
                             non_sequences = non_sequences,
                             outputs_info = outputs_info)
    return outputs


class VANILLA(Unit):
  def __init__(self, n_units, depth, **kwargs):
    super(VANILLA, self).__init__(n_units, depth, n_units, n_units, n_units, 1)

  def step(self, i_t, x_t, z_t, z_p, h_p):
    return [ T.tanh(z_t + z_p) ]


class LSTME(Unit):
  def __init__(self, n_units, depth, **kwargs):
    super(LSTME, self).__init__(n_units, depth, n_units * 4, n_units, n_units * 4, 2)

  def step(self, i_t, x_t, z_t, z_p, h_p, s_p):
    CI, GI, GF, GO, CO = [T.tanh, T.nnet.sigmoid, T.nnet.sigmoid, T.nnet.sigmoid, T.tanh]
    z = z_t + z_p
    u_t = GI(z[:,0 * self.slice:1 * self.slice]) # input gate
    r_t = GF(z[:,1 * self.slice:2 * self.slice]) # forget gate
    b_t = GO(z[:,2 * self.slice:3 * self.slice]) # output gate
    a_t = CI(z[:,3 * self.slice:4 * self.slice]) # net input
    s_t = a_t * u_t + s_p * r_t
    h_t = CO(s_t) * b_t
    #return [ h_t, theano.gradient.grad_clip(s_t, -50, 50) ]
    return [ h_t, s_t ]


class LSTM(Unit):
  def __init__(self, n_units, depth, **kwargs):
    super(LSTM, self).__init__(n_units, depth, n_units * 4, n_units, n_units * 4, 2)

  def scan(self, step, x, z, non_sequences, i, outputs_info, W_re, W_in, b, go_backwards = False, truncate_gradient = -1):
    try:
      XS = [S.output[::-(2 * go_backwards - 1)] for S in x]
    except Exception:
      XS = [S[::-(2 * go_backwards - 1)] for S in x]
    result = LSTMOp2Instance(*([W_re, outputs_info[1], b, i[::-(2 * go_backwards - 1)]] + XS + W_in))
    return [ result[0], result[2].dimshuffle('x',0,1) ]


class LSTMP(Unit):
  def __init__(self, n_units, depth, **kwargs):
    super(LSTMP, self).__init__(n_units, depth, n_units * 4, n_units, n_units * 4, 2)

  def scan(self, step, x, z, non_sequences, i, outputs_info, W_re, W_in, b, go_backwards = False, truncate_gradient = -1):
    z = T.inc_subtensor(z[-1 if go_backwards else 0], T.dot(outputs_info[0],W_re))
    result = LSTMOpInstance(z[::-(2 * go_backwards - 1)], W_re, outputs_info[1], i[::-(2 * go_backwards - 1)])
    return [ result[0], result[2].dimshuffle('x',0,1) ]


class LSTMC(Unit):
  def __init__(self, n_units, depth, **kwargs):
    super(LSTMC, self).__init__(n_units, depth, n_units * 4, n_units, n_units * 4, 2)

  def scan(self, step, x, z, non_sequences, i, outputs_info, W_re, W_in, b, go_backwards = False, truncate_gradient = -1):
    assert self.parent.recurrent_transform
    import OpLSTMCustom
    op = OpLSTMCustom.register_func(self.parent.recurrent_transform)
    custom_vars = self.parent.recurrent_transform.get_sorted_custom_vars()
    initial_state_vars = self.parent.recurrent_transform.get_sorted_state_vars_initial()

    # See OpLSTMCustom.LSTMCustomOp.
    # Inputs args are: Z, c, y0, i, W_re, custom input vars, initial state vars
    # Results: (output) Y, (gates and cell state) H, (final cell state) d, state vars sequences
    op_res = op(z[::-(2 * go_backwards - 1)],
                outputs_info[1], outputs_info[0], i[::-(2 * go_backwards - 1)], W_re, *(custom_vars + initial_state_vars))
    result = [ op_res[0], op_res[2].dimshuffle('x',0,1) ] + op_res[3:]
    assert len(result) == len(outputs_info)
    return result


class LSTMD(Unit):
  def __init__(self, n_units, depth, **kwargs):
    super(LSTMD, self).__init__(n_units, depth, n_units * 4, n_units, n_units * 4, 2)

  def scan(self, step, x, z, non_sequences, i, outputs_info, W_re, W_in, b, go_backwards = False, truncate_gradient = -1):
    assert self.parent.recurrent_transform
    import NetworkTwoDLayers
    NetworkTwoDLayers.TwoDLSTMConvLayer(x,)
    op = OpLSTMCustom.register_func(self.parent.recurrent_transform)
    custom_vars = self.parent.recurrent_transform.get_sorted_custom_vars()
    initial_state_vars = self.parent.recurrent_transform.get_sorted_state_vars_initial()

    # See OpLSTMCustom.LSTMCustomOp.
    # Inputs args are: Z, c, y0, i, W_re, custom input vars, initial state vars
    # Results: (output) Y, (gates and cell state) H, (final cell state) d, state vars sequences
    op_res = op(z[::-(2 * go_backwards - 1)],
                outputs_info[1], outputs_info[0], i[::-(2 * go_backwards - 1)], W_re, *(custom_vars + initial_state_vars))
    result = [ op_res[0], op_res[2].dimshuffle('x',0,1) ] + op_res[3:]
    assert len(result) == len(outputs_info)
    return result


class LSTMQ(Unit):
  def __init__(self, n_units, depth, **kwargs):
    super(LSTMQ, self).__init__(n_units, depth, n_units * 4, n_units, n_units * 4, 2)

  def step(self, i_t, x_t, z_t, z_p, h_p, s_p):
    #result = LSTMOpCellInstance(z_t+z_p,self.W_re,s_p,i_t)
    #return [ result[0], result[2] ]

    #proxy = LSTMP(self.n_units, self.depth)
    #res = proxy.scan(self.step, x_t.dimshuffle('x',0,1), z_t.dimshuffle('x',0,1), self.non_sequences, i_t.dimshuffle('x',0), [h_p,s_p], self.W_re, self.W_in, self.b, self.go_backwards, self.truncate_gradient)

    res = LSTMOpInstance((z_t+z_p).dimshuffle('x',0,1), self.W_re, s_p, i_t.dimshuffle('x',0))
    return [res[0][-1],res[2]]


class GRU(Unit):
  def __init__(self, n_units, depth, **kwargs):
    super(GRU, self).__init__(n_units, depth, n_units * 3, n_units, n_units * 2, 1)
    l = sqrt(6.) / sqrt(n_units * 3)
    rng = numpy.random.RandomState(1234)
    if depth > 1: values = numpy.asarray(rng.uniform(low=-l, high=l, size=(n_units, depth, n_units)), dtype=theano.config.floatX)
    else: values = numpy.asarray(rng.uniform(low=-l, high=l, size=(n_units, n_units)), dtype=theano.config.floatX)
    self.W_reset = theano.shared(value=values, borrow=True, name = "W_reset")
    self.params['W_reset'] = self.W_reset

  def step(self, i_t, x_t, z_t, z_p, h_p):
    CI, GR, GU = [T.tanh, T.nnet.sigmoid, T.nnet.sigmoid]
    u_t = GU(z_t[:,:self.slice] + z_p[:,:self.slice])
    r_t = GR(z_t[:,self.slice:2*self.slice] + z_p[:,self.slice:2*self.slice])
    h_c = CI(z_t[:,2*self.slice:] + self.dot(r_t * h_p, self.W_reset))
    return u_t * h_p + (1 - u_t) * h_c


class SRU(Unit):
  def __init__(self, n_units, depth, **kwargs):
    super(SRU, self).__init__(n_units, depth, n_units * 3, n_units, n_units * 3, 1)

  def step(self, i_t, x_t, z_t, z_p, h_p):
    CI, GR, GU = [T.tanh, T.nnet.sigmoid, T.nnet.sigmoid]
    u_t = GU(z_t[:,:self.slice] + z_p[:,:self.slice])
    r_t = GR(z_t[:,self.slice:2*self.slice] + z_p[:,self.slice:2*self.slice])
    h_c = CI(z_t[:,2*self.slice:3*self.slice] + r_t * z_p[:,2*self.slice:3*self.slice])
    return  u_t * h_p + (1 - u_t) * h_c


class RecurrentComponent(Container):
  def __init__(self):
    self.params = {}

  def init_state(self):
    return []

  def scan(self, step, x, z, non_sequences, i, outputs_info, W_re, W_in, b, go_backwards = False, truncate_gradient = -1):
    self.outputs_info = outputs_info
    self.non_sequences = non_sequences
    self.W_re = W_re
    self.W_in = W_in
    self.b = b
    self.go_backwards = go_backwards
    self.truncate_gradient = truncate_gradient
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

class RecurrentUnitLayer(Layer):
  recurrent = True
  layer_class = "rec"

  def __init__(self,
               n_out, # number of cells
               n_units = None,  # when initialized via Network.from_hdf_model_topology
               direction = 1, # forward (1), backward (-1) or bidirectional (0)
               truncation = -1, # truncate the gradient after this amount of time steps
               sampling = 1, # scan every nth frame only
               encoder = None, # list of encoder layers
               psize = 0, # size of projection
               pact = 'relu', # activation of projection
               pdepth = 1, # depth of projection
               unit = 'lstm', # cell type
               n_dec = 0, # number of time steps to decode
               attention = "none", # soft attention (none, input, time) # deprecated
               recurrent_transform = "none",
               recurrent_transform_attribs = "{}",
               attention_template = None,
               attention_distance = 'l2',
               attention_step = "focus",
               attention_beam = 0, # soft attention context window
               base = None,
               lm = False, # language model
               force_lm = False, # assumes y to be given during test
               droplm = 1.0, # language model drop during training
               dropconnect = 0.0, # recurrency dropout
               depth = 1,
               bias_random_init_forget_shift=0.0,
               **kwargs):
    unit_given = unit
    if unit == 'lstm':  # auto selection
      if str(theano.config.device).startswith('cpu'):
        unit = 'lstme'
      elif recurrent_transform == 'none' and (not lm or droplm == 0.0):
        unit = 'lstmp'
      else:
        unit = 'lstmc'
    elif unit in ("lstmc", "lstmp") and str(theano.config.device).startswith('cpu'):
      unit = "lstme"
    kwargs.setdefault("depth", depth)
    kwargs.setdefault("n_out", n_out)
    if n_units is not None:
      assert n_units == n_out
    super(RecurrentUnitLayer, self).__init__(**kwargs)
    self.set_attr('from', ",".join([s.name for s in self.sources]) if self.sources else "null")
    self.set_attr('n_out', n_out)
    self.set_attr('unit', unit_given.encode("utf8"))
    self.set_attr('psize', psize)
    self.set_attr('pact', pact)
    self.set_attr('pdepth', pdepth)
    self.set_attr('truncation', truncation)
    self.set_attr('sampling', sampling)
    self.set_attr('direction', direction)
    self.set_attr('lm', lm)
    self.set_attr('force_lm', force_lm)
    self.set_attr('droplm', droplm)
    self.set_attr('dropconnect', dropconnect)
    if bias_random_init_forget_shift:
      self.set_attr("bias_random_init_forget_shift", bias_random_init_forget_shift)
    self.set_attr('attention', attention.encode("utf8") if attention else None)
    self.set_attr('attention_beam', attention_beam)
    self.set_attr('recurrent_transform', recurrent_transform.encode("utf8"))
    if isinstance(recurrent_transform_attribs, str):
      recurrent_transform_attribs = json.loads(recurrent_transform_attribs)
    if attention_template is not None:
      self.set_attr('attention_template', attention_template)
    self.set_attr('recurrent_transform_attribs', recurrent_transform_attribs)
    self.set_attr('attention_distance', attention_distance)
    self.set_attr('attention_step', attention_step)
    if lm: # TODO hack
      recurrent_transform += "_lm"
    if encoder:
      self.set_attr('encoder', ",".join([e.name for e in encoder]))
    if base:
      self.set_attr('base', ",".join([b.name for b in base]))
    else:
      base = encoder
    self.base = base
    self.set_attr('n_units', n_out)
    unit = eval(unit.upper())(**self.attrs)
    assert isinstance(unit, Unit)
    self.unit = unit
    kwargs.setdefault("n_out", unit.n_out)
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
        self.Wp = [ self.add_param(self.create_random_uniform_weights(n_re, psize, name="Wp_0_%s"%self.name, depth=1)) ]
        for i in xrange(1, pdepth):
          self.Wp.append(self.add_param(self.create_random_uniform_weights(psize, psize, name="Wp_%d_%s"%(i, self.name), depth=1)))
        W_re = self.create_random_uniform_weights(psize, unit.n_re, name="W_re_%s" % self.name)
      else:
        W_re = self.create_random_uniform_weights(n_re, unit.n_re, name="W_re_%s" % self.name)
      self.add_param(W_re)
    # initialize forward weights
    if self.depth > 1:
      bias_init_value = numpy.zeros((self.depth, unit.n_in), dtype = theano.config.floatX)
    else:
      bias_init_value = numpy.zeros((unit.n_in, ), dtype = theano.config.floatX)
    if bias_random_init_forget_shift:
      assert self.depth == 1
      assert unit.n_units * 4 == unit.n_in  # (input gate, forget gate, output gate, net input)
      bias_init_value[unit.n_units:2 * unit.n_units] += bias_random_init_forget_shift
    #self.b = theano.shared(value=value, borrow=True, name="b_%s"%self.name) #self.create_bias()
    #self.params["b_%s"%self.name] = self.b
    self.b.set_value(bias_init_value)
    self.W_in = []
    for s in self.sources:
      W = self.create_random_uniform_weights(s.attrs['n_out'], unit.n_in,
                                             s.attrs['n_out'] + unit.n_in + unit.n_re,
                                             name="W_in_%s_%s" % (s.name, self.name), depth = 1)
      self.W_in.append(W)
      self.add_param(W)
    # make input
    z = self.b
    for x_t, m, W in zip(self.sources, self.masks, self.W_in):
      if x_t.attrs['sparse']:
        if x_t.output.ndim == 3: out_dim = x_t.output.shape[2]
        elif x_t.output.ndim == 2: out_dim = 1
        else: assert False, x_t.output.ndim
        #z += W[T.cast(s.output, 'int32')].reshape((x_t.output.shape[0],x_t.output.shape[1],out_dim * W.shape[1]))
        if x_t.output.ndim == 3:
          z += W[T.cast(x_t.output[:,:,0], 'int32')]
        elif x_t.output.ndim == 2:
          z += W[T.cast(x_t.output, 'int32')]
        else:
          assert False, x_t.output.ndim
      elif m is None:
        z += T.dot(x_t.output, W)
      else:
        z += self.dot(self.mass * m * x_t.output, W)
    #z = z * T.cast(self.index.dimshuffle(0,1,'x').repeat(unit.n_in,axis=2),'float32')
    if self.depth > 1:
      assert False
      z = z.dimshuffle(0,1,'x',2).repeat(self.depth, axis=2)
    num_batches = self.index.shape[1]
    self.num_batches = num_batches
    if direction == 0:
      assert False # this is broken
      z = T.set_subtensor(z[:,:,depth:,:], z[::-1,:,:depth,:])

    non_sequences = []
    if self.attrs['lm']:
      if not 'target' in self.attrs:
        self.attrs['target'] = 'classes'
      l = sqrt(6.) / sqrt(unit.n_out + self.y_in[self.attrs['target']].n_out)

      values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(unit.n_out, self.y_in[self.attrs['target']].n_out)), dtype=theano.config.floatX)
      self.W_lm_in = theano.shared(value=values, borrow=True, name = "W_lm_in_"+self.name)
      self.add_param(self.W_lm_in)
      l = sqrt(6.) / sqrt(unit.n_in + self.y_in[self.attrs['target']].n_out)
      values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(self.y_in[self.attrs['target']].n_out, unit.n_in)), dtype=theano.config.floatX)
      self.W_lm_out = theano.shared(value=values, borrow=True, name = "W_lm_out_"+self.name)
      self.add_param(self.W_lm_out)
      if self.attrs['droplm'] == 0.0 and (self.train_flag or force_lm):
        self.lmmask = 1
        recurrent_transform = recurrent_transform[:-3]
      elif self.attrs['droplm'] < 1.0 and (self.train_flag or force_lm):
        srng = theano.tensor.shared_randomstreams.RandomStreams(self.rng.randint(1234))
        self.lmmask = T.cast(srng.binomial(n=1, p=1.0 - self.attrs['droplm'], size=self.index.shape), theano.config.floatX).dimshuffle(0,1,'x').repeat(unit.n_in,axis=2)
      else:
        self.lmmask = T.zeros_like(self.index, dtype='float32').dimshuffle(0,1,'x').repeat(unit.n_in,axis=2)

    if self.attrs['dropconnect'] > 0.0:
      srng = theano.tensor.shared_randomstreams.RandomStreams(self.rng.randint(1234))
      connectmask = T.cast(srng.binomial(n=1, p=1.0 - self.attrs['dropconnect'], size=(unit.n_out,)), theano.config.floatX)
      connectmass = T.constant(1.0 / (1.0 - self.attrs['dropconnect']), dtype='float32')
      non_sequences += [connectmask, connectmass]

    if not attention:
      attention = "none"
    if attention == "default":
      attention = "attention_dot"
    if attention == 'input': # attention is just a sequence dependent bias (lstmp compatible)
      src = []
      src_names = []
      n_in = 0
      for e in base:
        src_base = [ s for s in e.sources if s.name not in src_names ]
        src_names += [ s.name for s in e.sources ]
        src += [s.output for s in src_base]
        n_in += sum([s.attrs['n_out'] for s in src_base])
      self.xc = T.concatenate(src, axis=2)
      l = sqrt(6.) / sqrt(self.attrs['n_out'] + n_in)
      values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(n_in, 1)), dtype=theano.config.floatX)
      self.W_att_xc = theano.shared(value=values, borrow=True, name = "W_att_xc")
      self.add_param(self.W_att_xc)
      values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(n_in, self.attrs['n_out'] * 4)), dtype=theano.config.floatX)
      self.W_att_in = theano.shared(value=values, borrow=True, name = "W_att_in")
      self.add_param(self.W_att_in)
      zz = T.exp(T.tanh(T.dot(self.xc, self.W_att_xc))) # TB1
      self.zc = T.dot(T.sum(self.xc * (zz / T.sum(zz, axis=0, keepdims=True)).repeat(self.xc.shape[2],axis=2), axis=0, keepdims=True), self.W_att_in)
    elif attention != "none":
      assert recurrent_transform == "none"
      recurrent_transform = attention
    recurrent_transform_inst = RecurrentTransform.transform_classes[recurrent_transform](layer=self)
    assert isinstance(recurrent_transform_inst, RecurrentTransform.RecurrentTransformBase)
    self.recurrent_transform = recurrent_transform_inst

    self.out_dec = self.index.shape[0]
    # scan over sequence
    for s in xrange(self.attrs['sampling']):
      index = self.index[s::self.attrs['sampling']]
      sequences = z
      sources = self.sources
      n_dec = self.out_dec
      if encoder:
        if 'n_dec' in self.attrs:
          n_dec = self.attrs['n_dec']
          index = T.alloc(numpy.cast[numpy.int8](1), n_dec, self.index.shape[1])
        outputs_info = [ T.concatenate([e.act[i][-1] for e in encoder], axis=-1) for i in xrange(unit.n_act) ]
        #outputs_info = [T.alloc(numpy.cast[theano.config.floatX](0), num_batches, unit.n_out)] + [ T.concatenate([e.act[i][-1] for e in encoder], axis=1) for i in xrange(1,unit.n_act) ]
        if self.depth == 1:
          sequences += T.alloc(numpy.cast[theano.config.floatX](0), n_dec, num_batches, unit.n_in) + (self.zc if attention == 'input' else 0)
        else:
          sequences += T.alloc(numpy.cast[theano.config.floatX](0), n_dec, num_batches, self.depth, unit.n_in) + (self.zc if attention == 'input' else 0)
      else:
        if self.depth == 1:
          outputs_info = [ T.alloc(numpy.cast[theano.config.floatX](0), num_batches, unit.n_out) for a in xrange(unit.n_act) ]
        else:
          assert False
          outputs_info = [ T.alloc(numpy.cast[theano.config.floatX](0), num_batches, self.depth, unit.n_out) for i in xrange(unit.n_act) ]

      if False and self.attrs['lm'] and self.attrs['droplm'] < 1.0:
        y = self.y_in[self.attrs['target']].flatten() #.reshape(self.index.shape)
        n_cls = self.y_in[self.attrs['target']].n_out
        #real_weight = T.constant(1.0 - (self.attrs['droplm'] if self.train_flag else 1.0), dtype='float32')
        #sequences = T.concatenate([self.W_lm_out[0].dimshuffle('x','x',0).repeat(self.index.shape[1],axis=1), y_t], axis=0) * real_weight + self.b #* lmmask * float(int(self.train_flag)) + self.b
        if direction == 1:
          y_t = self.W_lm_out[y].reshape((index.shape[0],index.shape[1],unit.n_in))[:-1] # (T-1)BD
          #sequences += T.concatenate([self.W_lm_out[0].dimshuffle('x','x',0).repeat(num_batches,axis=1), y_t], axis=0) * self.lmmask
        else:
          y_t = self.W_lm_out[y].reshape((index.shape[0],index.shape[1],unit.n_in))[1:] # (T-1)BD
          #sequences += T.concatenate([y_t, self.W_lm_out[0].dimshuffle('x','x',0).repeat(num_batches,axis=1)], axis=0) * self.lmmask
        #sequences += T.concatenate([self.W_lm_out[0].dimshuffle('x','x',0).repeat(self.index.shape[1],axis=1), y_t], axis=0) * self.lmmask
        #sequences += self.W_lm_out[self.y_in[self.attrs['target']]].reshape((index.shape[0],index.shape[1],unit.n_in)) #T.concatenate([self.W_lm_out[0].dimshuffle('x','x',0).repeat(self.index.shape[1],axis=1), y_t], axis=0) * self.lmmask
        #sequences = T.alloc(numpy.cast[theano.config.floatX](0), n_dec, num_batches, unit.n_in) + self.b + (self.zc if attention == 'input' else 0)
        #outputs_info.append(T.eye(n_cls, 1).flatten().dimshuffle('x',0).repeat(index.shape[1],0))
      if sequences == self.b:
        sequences += T.alloc(numpy.cast[theano.config.floatX](0), n_dec, num_batches, unit.n_in) + (self.zc if self.attrs['attention'] == 'input' else 0)

      if self.recurrent_transform:
        outputs_info += self.recurrent_transform.get_sorted_state_vars_initial()

      def stepr(x_t, z_t, i_t, *args):
        mask,mass = 0,0
        if self.attrs['dropconnect'] > 0.0:
          mask = args[-2]
          mass = args[-1]
          args = args[:-2]
        if self.recurrent_transform:
          attention_non_seq_inputs = args[len(args) - len(self.recurrent_transform.input_vars):]
          args = args[:len(args) - len(self.recurrent_transform.input_vars)]
        if self.attrs['lm']:
          c_p = args[-1]
          args = args[:-1]
        h_p = args[0]
        if self.depth == 1:
          #i = i_t.dimshuffle(0,'x').repeat(self.attrs['n_out'],axis=1)
          i = T.outer(i_t, T.alloc(numpy.cast['float32'](1), self.attrs['n_out']))
        else:
          assert False
          h_p = self.make_consensus(h_p, axis = 1)
          i = T.outer(i_t, T.alloc(numpy.cast['float32'](1), self.attrs['n_out'])).dimshuffle(0, 'x', 1).repeat(self.depth, axis=1)
        for W in self.Wp:
          h_p = pact(T.dot(h_p, W))
        result = []
        if self.attrs['lm']:
          h_e = T.exp(T.dot(h_p, self.W_lm_in))
          c_t = h_e / T.sum(h_e, axis=1, keepdims=True)
          result.append(c_t)
          #z_t += T.dot(c_p, self.W_lm_out) * T.all(T.eq(z_t,0),axis=1,keepdims=True)
          z_t += self.W_lm_out[T.argmax(c_p,axis=1)] * T.all(T.eq(z_t,0),axis=1,keepdims=True)
        if mass:
          z_p = T.dot(h_p * mass * mask, W_re)
        else:
          z_p = T.dot(h_p, W_re)
        updates = {}
        if self.depth > 1:
          assert False # this is broken
          sargs = [arg.dimshuffle(0,1,2) for arg in args]
          act = [ act.dimshuffle(0,2,1) for act in unit.step(x_t.dimshuffle(1,0), z_t.dimshuffle(0,2,1), z_p.dimshuffle(0,2,1), *sargs) ]
        else:
          if self.recurrent_transform:
            z_r, r_updates = self.recurrent_transform.step(h_p)
            z_t += z_r
            updates.update(r_updates)
        act = unit.step(i_t, x_t, z_t, z_p, *args)
        #return [ act[0] * i ] + [ act[j] * i + theano.gradient.grad_clip(args[j] * (T.ones_like(i)-i),-0.00000001,0.00000001) for j in xrange(1,unit.n_act) ] + result
        #return [ act[0] * i ] + [ T.switch(T.gt(i,T.zeros_like(i)),act[j], args[j]) for j in xrange(1,unit.n_act) ] + result
        return [act[0]*i] + [ act[j] * i + args[j] * (1-i) for j in xrange(1,unit.n_act) ] + result, updates

      o_h = T.as_tensor(numpy.ones((unit.n_units,), dtype='float32'))
      def stepk(x_t, z_t, i_t, *args):
        z_p = T.dot(args[0], W_re)
        #i_x = i_t.dimshuffle(0,'x').repeat(z_p.shape[1],axis=1)
        act = unit.step(i_t, x_t, z_t, z_p, *args)
        i_a = i_t.dimshuffle(0,'x').repeat(unit.n_units,axis=1)
        #i_a = T.outer(i_t, o_h) # T.alloc(numpy.cast['float32'](1), unit.n_units))
        #return [ act[0] * i ] + [  act[j] * i + theano.gradient.grad_clip(args[j] * (i-1),0,0) for j in xrange(1,unit.n_act) ]
        return [ act[0] * i_a ] + [  act[j] * i_a for j in xrange(1,unit.n_act) ]
        #return [ theano.gradient.grad_clip(act[0] * i,T.sum(i*500),T.sum(i*500)) ] + [  act[j] * i + theano.gradient.grad_clip(args[j] * (i-1),0,0) for j in xrange(1,unit.n_act) ]
        #return [ T.switch(T.lt(i,T.ones_like(i)), theano.gradient.grad_clip(args[a], 0, 0), act[a]) for a in xrange(unit.n_act) ]

      o_output = T.as_tensor(numpy.ones((unit.n_units,), dtype='float32'))
      o_h = T.as_tensor(numpy.ones((unit.n_units,), dtype='float32'))
      def step(x_t, z_t, i_t, y_p, c_p, *other_args):
        # See Unit.scan() for seqs.
        # args: seqs (x_t = unit.xc, z_t, i_t), outputs (# unit.n_act, y_p, c_p, ...), non_seqs (none)
        other_outputs = []
        if self.recurrent_transform:
          state_vars = other_args[:len(self.recurrent_transform.state_vars)]
          self.recurrent_transform.set_sorted_state_vars(state_vars)
          z_r, r_updates = self.recurrent_transform.step(y_p)
          z_t += z_r
          for v in self.recurrent_transform.get_sorted_state_vars():
            other_outputs += [r_updates[v]]
        z_t += T.dot(y_p, W_re)
        partition = z_t.shape[1] / 4
        ingate = T.nnet.sigmoid(z_t[:,:partition])
        forgetgate = T.nnet.sigmoid(z_t[:,partition:2*partition])
        outgate = T.nnet.sigmoid(z_t[:,2*partition:3*partition])
        input = T.tanh(z_t[:,3*partition:4*partition])
        c_t = forgetgate * c_p + ingate * input
        y_t = outgate * T.tanh(c_t)
        i_output = T.outer(i_t, o_output)
        i_h = T.outer(i_t, o_h)
        # return: next outputs (# unit.n_act, y_t, c_t, ...)
        return (y_t * i_output, c_t * i_h + c_p * (1 - i_h)) + tuple(other_outputs)

      index_f = T.cast(index, theano.config.floatX)
      unit.set_parent(self)
      #if unit_given in ['lstm', 'lstmp', 'lstmc'] and isinstance(unit,LSTME):
      #  print "hi"
      #  sequences = T.inc_subtensor(sequences[-1 if direction == -1 else 0], T.dot(outputs_info[0],W_re))
      #  outputs_info[0] = T.alloc(numpy.cast[theano.config.floatX](0), num_batches, unit.n_out)
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

      if self.recurrent_transform:
        self.recurrent_transform_state_var_seqs = outputs[-len(self.recurrent_transform.state_vars):]

      if self.attrs['sampling'] > 1:
        if s == 0:
          #self.act = [ T.repeat(act, self.attrs['sampling'], axis = 0)[:self.sources[0].output.shape[0]] for act in outputs ]
          self.act = [ T.alloc(numpy.cast['float32'](0), self.index.shape[0], self.index.shape[1], n_out) for act in outputs ]
        self.act = [ T.set_subtensor(tot[s::self.attrs['sampling']], act) for tot,act in zip(self.act, outputs) ]
      else:
        self.act = outputs[:unit.n_act]
        if len(outputs) > unit.n_act:
          self.aux = outputs[unit.n_act:]
    #T.set_subtensor(self.act[0][(self.index > 0).nonzero()], T.zeros_like(self.act[0][(self.index > 0).nonzero()]))
    #T.set_subtensor(self.act[1][(self.index > 0).nonzero()], T.zeros_like(self.act[1][(self.index > 0).nonzero()]))
    #jindex = self.index.dimshuffle(0,1,'x').repeat(unit.n_out,axis=2)
    #self.act[0] = T.switch(T.lt(jindex, T.ones_like(jindex)), T.zeros_like(self.act[0]), self.act[0])
    #self.act[1] = T.switch(T.eq(jindex, T.zeros_like(jindex)), T.zeros_like(self.act[1]), self.act[1])
    self.make_output(self.act[0][::direction or 1])
    self.params.update(unit.params)

  def cost(self):
    """
    :rtype: (theano.Variable | None, dict[theano.Variable,theano.Variable] | None)
    :returns: cost, known_grads
    """
    if self.recurrent_transform:
      return self.recurrent_transform.cost(), None
    return None, None


class RecurrentChunkLayer(Layer):
  recurrent = True
  layer_class = "chunkrec"

  def __init__(self,
               n_out, # number of cells
               chunk_size,
               n_units = None,  # when initialized via Network.from_hdf_model_topology
               direction = 1, # forward (1), backward (-1) or bidirectional (0)
               truncation = -1, # truncate the gradient after this amount of time steps
               sampling = 1, # scan every nth frame only
               encoder = None, # list of encoder layers
               psize = 0, # size of projection
               pact = 'relu', # activation of projection
               pdepth = 1, # depth of projection
               unit = 'lstm', # cell type
               n_dec = 0, # number of time steps to decode
               attention = "none", # soft attention (none, input, time) # deprecated
               recurrent_transform = "none",
               recurrent_transform_attribs = "{}",
               attention_sigma = 1.0,
               attention_beam = 3, # soft attention context window
               base = None,
               lm = False, # language model
               force_lm = False, # assumes y to be given during test
               droplm = 1.0, # language model drop during training
               dropconnect = 0.0, # recurrency dropout
               depth = 1,
               **kwargs):
    unit_given = unit
    if unit == 'lstm':  # auto selection
      if str(theano.config.device).startswith('cpu'):
        unit = 'lstme'
      elif recurrent_transform == 'none' and (not lm or droplm == 0.0):
        unit = 'lstmp'
      else:
        unit = 'lstmc'
    elif unit in ("lstmc", "lstmp") and str(theano.config.device).startswith('cpu'):
      unit = "lstme"
    kwargs.setdefault("depth", depth)
    kwargs.setdefault("n_out", n_out)
    if n_units is not None:
      assert n_units == n_out
    super(RecurrentChunkLayer, self).__init__(**kwargs)
    self.set_attr('from', ",".join([s.name for s in self.sources]) if self.sources else "null")
    self.set_attr('unit', unit_given.encode("utf8"))
    self.set_attr('psize', psize)
    self.set_attr('pact', pact)
    self.set_attr('pdepth', pdepth)
    self.set_attr('truncation', truncation)
    self.set_attr('sampling', sampling)
    self.set_attr('direction', direction)
    self.set_attr('lm', lm)
    self.set_attr('force_lm', force_lm)
    self.set_attr('droplm', droplm)
    self.set_attr('dropconnect', dropconnect)
    self.set_attr('attention', attention.encode("utf8") if attention else None)
    self.set_attr('attention_beam', attention_beam)
    self.set_attr('recurrent_transform', recurrent_transform.encode("utf8"))
    self.set_attr('attention_linear_support', attention_linear_support)
    if isinstance(recurrent_transform_attribs, str):
      recurrent_transform_attribs = json.loads(recurrent_transform_attribs)
    self.set_attr('recurrent_transform_attribs', recurrent_transform_attribs)
    self.set_attr('attention_sigma', attention_sigma)
    if lm: # TODO hack
      recurrent_transform += "_lm"
    if encoder:
      self.set_attr('encoder', ",".join([e.name for e in encoder]))
    if base:
      self.set_attr('base', ",".join([b.name for b in base]))
    else:
      base = encoder
    self.base = base
    self.set_attr('n_units', n_out)
    unit = eval(unit.upper())(**self.attrs)
    assert isinstance(unit, Unit)
    self.unit = unit
    kwargs.setdefault("n_out", unit.n_out)
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
        self.Wp = [ self.add_param(self.create_random_uniform_weights(n_re, psize, n_re + psize, name = "Wp_0_%s"%self.name, depth=1)) ]
        for i in xrange(1, pdepth):
          self.Wp.append(self.add_param(self.create_random_uniform_weights(psize, psize, psize + psize, name = "Wp_%d_%s"%(i, self.name), depth=1)))
        W_re = self.create_random_uniform_weights(psize, unit.n_re, psize + unit.n_re, name="W_re_%s" % self.name)
      else:
        W_re = self.create_random_uniform_weights(n_re, unit.n_re, n_re + unit.n_re, name="W_re_%s" % self.name)
      self.add_param(W_re)
    # initialize forward weights
    if self.depth > 1:
      value = numpy.zeros((self.depth, unit.n_in), dtype = theano.config.floatX)
    else:
      value = numpy.zeros((unit.n_in, ), dtype = theano.config.floatX)
      value[unit.n_units:2*unit.n_units] = 0
    #self.b = theano.shared(value=value, borrow=True, name="b_%s"%self.name) #self.create_bias()
    #self.params["b_%s"%self.name] = self.b
    self.b.set_value(value)
    self.W_in = []
    for s in self.sources:
      W = self.create_random_uniform_weights(s.attrs['n_out'], unit.n_in,
                                             s.attrs['n_out'] + unit.n_in + unit.n_re,
                                             name="W_in_%s_%s" % (s.name, self.name), depth = 1)
      self.W_in.append(W)
      self.add_param(W)
    # make input
    z = self.b.dimshuffle('x','x',0).repeat(self.index.shape[0],axis=0).repeat(self.index.shape[1],axis=1)
    for x_t, m, W in zip(self.sources, self.masks, self.W_in):
      if x_t.attrs['sparse']:
        if x_t.output.ndim == 3: out_dim = x_t.output.shape[2]
        elif x_t.output.ndim == 2: out_dim = 1
        else: assert False, x_t.output.ndim
        #z += W[T.cast(s.output, 'int32')].reshape((x_t.output.shape[0],x_t.output.shape[1],out_dim * W.shape[1]))
        if x_t.output.ndim == 3:
          z += W[T.cast(x_t.output[:,:,0], 'int32')]
        elif x_t.output.ndim == 2:
          z += W[T.cast(x_t.output, 'int32')]
        else:
          assert False, x_t.output.ndim
      elif m is None:
        z += T.dot(x_t.output, W)
      else:
        z += self.dot(self.mass * m * x_t.output, W)
    num_batches = self.index.shape[1]
    non_sequences = []
    sequences = T.alloc(numpy.cast[theano.config.floatX](0), self.index.shape[0], self.index.shape[1], unit.n_in) + z
    calloc = T.alloc(numpy.cast[theano.config.floatX](0), self.index.shape[0] + chunk_size - (self.index.shape[0] % chunk_size), sequences.shape[1], sequences.shape[2])
    sequences = T.set_subtensor(calloc[:self.index.shape[0]],sequences)
    ialloc = T.alloc(numpy.cast['int32'](1), self.index.shape[0] + chunk_size - (self.index.shape[0] % chunk_size), self.index.shape[1])
    index = T.set_subtensor(ialloc[:self.index.shape[0]],self.index)

    num_chunks = sequences.shape[0] / chunk_size
    num_batches = num_batches * num_chunks
    sequences = sequences.reshape((chunk_size,num_batches,sequences.shape[2]))
    index = index.reshape((chunk_size,num_batches))

    self.num_batches = num_batches

    if self.attrs['lm']:
      if not 'target' in self.attrs:
        self.attrs['target'] = 'classes'
      l = sqrt(2.) / sqrt(unit.n_out + self.y_in[self.attrs['target']].n_out + self.y_in[self.attrs['target']].n_in)
      values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(unit.n_out, self.y_in[self.attrs['target']].n_out)), dtype=theano.config.floatX)
      self.W_lm_in = theano.shared(value=values, borrow=True, name = "W_lm_in_"+self.name)
      self.add_param(self.W_lm_in)
      #l = sqrt(6.) / sqrt(unit.n_in + self.y_in[self.attrs['target']].n_out)
      values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(self.y_in[self.attrs['target']].n_out, unit.n_in)), dtype=theano.config.floatX)
      self.W_lm_out = theano.shared(value=values, borrow=True, name = "W_lm_out_"+self.name)
      self.add_param(self.W_lm_out)
      if self.attrs['droplm'] == 0.0:
        self.lmmask = 1
        recurrent_transform = recurrent_transform[:-3]
      elif self.attrs['droplm'] < 1.0 and (self.train_flag or force_lm):
        srng = theano.tensor.shared_randomstreams.RandomStreams(self.rng.randint(1234))
        self.lmmask = T.cast(srng.binomial(n=1, p=1.0 - self.attrs['droplm'], size=index.shape), theano.config.floatX).dimshuffle(0,1,'x').repeat(unit.n_in,axis=2)
      else:
        self.lmmask = T.zeros_like(sequences, dtype='float32')

    if not attention:
      attention = "none"
    if attention == "default":
      attention = "attention_dot"
    if attention == 'input': # attention is just a sequence dependent bias (lstmp compatible)
      src = []
      src_names = []
      n_in = 0
      for e in base:
        src_base = [ s for s in e.sources if s.name not in src_names ]
        src_names += [ s.name for s in e.sources ]
        src += [s.output for s in src_base]
        n_in += sum([s.attrs['n_out'] for s in src_base])
      self.xc = T.concatenate(src, axis=2)
      l = sqrt(6.) / sqrt(self.attrs['n_out'] + n_in)
      values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(n_in, 1)), dtype=theano.config.floatX)
      self.W_att_xc = theano.shared(value=values, borrow=True, name = "W_att_xc")
      self.add_param(self.W_att_xc)
      values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(n_in, self.attrs['n_out'] * 4)), dtype=theano.config.floatX)
      self.W_att_in = theano.shared(value=values, borrow=True, name = "W_att_in")
      self.add_param(self.W_att_in)
      zz = T.exp(T.dot(self.xc, self.W_att_xc)) # TB1
      self.zc = T.dot(T.sum(self.xc * (zz / T.sum(zz, axis=0, keepdims=True)).repeat(self.xc.shape[2],axis=2), axis=0, keepdims=True), self.W_att_in)
    elif attention != "none":
      assert recurrent_transform == "none"
      recurrent_transform = attention
    recurrent_transform_inst = RecurrentTransform.transform_classes[recurrent_transform](layer=self)
    assert isinstance(recurrent_transform_inst, RecurrentTransform.RecurrentTransformBase)
    self.recurrent_transform = recurrent_transform_inst
    self.out_dec = self.index.shape[0]
    # scan over sequence
    sources = self.sources
    if encoder:
      outputs_info = [ T.concatenate([e.act[i][-1] for e in encoder], axis=1) for i in xrange(unit.n_act) ]
    else:
      outputs_info = [ T.alloc(numpy.cast[theano.config.floatX](0), num_batches, unit.n_out) for a in xrange(unit.n_act) ]

    if self.attrs['lm'] and self.attrs['droplm'] < 1.0:
      yalloc = T.alloc(numpy.cast['int32'](0), (self.index.shape[0] + chunk_size - (self.index.shape[0] % chunk_size)) * self.index.shape[1])
      y = T.set_subtensor(yalloc[:self.index.shape[0] * self.index.shape[1]],self.y_in[self.attrs['target']].flatten())
      y_t = self.W_lm_out[y].reshape((chunk_size,num_batches,unit.n_in))[:-1] # (T-1)BD
      if direction == 1:
        sequences += T.concatenate([self.W_lm_out[0].dimshuffle('x','x',0).repeat(num_batches,axis=1), y_t], axis=0) * self.lmmask
      else:
        sequences += T.concatenate([y_t, self.W_lm_out[0].dimshuffle('x','x',0).repeat(num_batches,axis=1)], axis=0) * self.lmmask

    if self.recurrent_transform:
      outputs_info += self.recurrent_transform.get_sorted_state_vars_initial()
    if attention == "input":
      sequences += self.zc

    o_output = T.as_tensor(numpy.ones((unit.n_units,), dtype='float32'))
    o_h = T.as_tensor(numpy.ones((unit.n_units,), dtype='float32'))
    def step(x_t, z_t, i_t, y_p, c_p, *other_args):
      # See Unit.scan() for seqs.
      # args: seqs (x_t = unit.xc, z_t, i_t), outputs (# unit.n_act, y_p, c_p, ...), non_seqs (none)
      other_outputs = []
      if self.recurrent_transform:
        state_vars = other_args[:len(self.recurrent_transform.state_vars)]
        self.recurrent_transform.set_sorted_state_vars(state_vars)
        z_r, r_updates = self.recurrent_transform.step(y_p)
        z_t += z_r
        for v in self.recurrent_transform.get_sorted_state_vars():
          other_outputs += [r_updates[v]]
      z_t += T.dot(y_p, W_re)
      partition = z_t.shape[1] / 4
      ingate = T.nnet.sigmoid(z_t[:,:partition])
      forgetgate = T.nnet.sigmoid(z_t[:,partition:2*partition])
      outgate = T.nnet.sigmoid(z_t[:,2*partition:3*partition])
      input = T.tanh(z_t[:,3*partition:4*partition])
      c_t = forgetgate * c_p + ingate * input
      y_t = outgate * T.tanh(c_t)
      i_output = T.outer(i_t, o_output)
      i_h = T.outer(i_t, o_h)
      return (y_t * i_output, c_t * i_h + c_p * (1 - i_h)) + tuple(other_outputs)

    unit.set_parent(self)
    index_f = T.cast(index, theano.config.floatX)

    outputs = unit.scan(step,
                        sources,
                        sequences,
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

    #self.index = index[-1].reshape((num_chunks,num_batches/num_chunks))
    for i in xrange(unit.n_act):
      outputs[i] = outputs[i].reshape((num_chunks * outputs[i].shape[0],outputs[i].shape[1]/num_chunks,outputs[i].shape[2]))[::direction or 1][:self.index.shape[0]][::direction or 1]

    if self.recurrent_transform:
      self.recurrent_transform_state_var_seqs = outputs[-len(self.recurrent_transform.state_vars):]
    self.act = outputs
    self.make_output(self.act[0][::direction or 1])
    self.params.update(unit.params)

  def cost(self):
    """
    :rtype: (theano.Variable | None, dict[theano.Variable,theano.Variable] | None)
    :returns: cost, known_grads
    """
    if self.recurrent_transform:
      return self.recurrent_transform.cost(), None
    return None, None
