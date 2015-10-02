
from math import sqrt, pi
import theano
import theano.tensor as T
import theano.sandbox.cuda as cuda
import numpy


class RecurrentTransformBase(object):
  name = None

  def __init__(self, force_gpu=False, layer=None):
    """
    :type layer: NetworkRecurrentLayer.RecurrentUnitLayer
    """
    if force_gpu:
      self.tt = cuda
    else:
      self.tt = T
    self.layer = layer
    self.input_vars = {}  # used as non_sequences for theano.scan(), i.e. as input for the step() function
    self.state_vars = {}  # updated in each step()
    self.custom_vars = {}

  def init_vars(self):
    pass

  def create_vars(self):
    pass

  def add_param(self, v):
    assert v.name
    if self.layer:
      self.layer.add_param(v, v.name + "_" + self.name)
    self.add_var(v)
    return v

  def add_input(self, v):
    assert v.name
    self.input_vars[v.name] = v
    self.add_var(v)
    return v

  def add_state_var(self, v):
    assert v.name
    self.state_vars[v.name] = v
    self.add_var(v)
    return v

  def add_var(self, v):
    assert v.name
    self.custom_vars[v.name] = v
    return v

  def get_sorted_non_sequence_inputs(self):
    return [v for (k, v) in sorted(self.input_vars.items())]

  def get_sorted_custom_vars(self):
    return [v for (k, v) in sorted(self.custom_vars.items())]

  def get_sorted_state_vars(self):
    return [v for (k, v) in sorted(self.state_vars.items())]

  def function_for_custom_op(self):
    """
    :return: (y_p, z_re, custom_vars)
    :rtype: (theano.Variable,theano.Variable,list[theano.Variable],theano.Variable,list[theano.Variable])
    """
    assert not self.layer
    assert self.tt is cuda
    y_p = self.tt.fmatrix("y_p")
    self.init_vars()
    z_re, updates = self.step(y_p)
    return y_p, z_re, self.get_sorted_custom_vars(), updates

  def step(self, y_p):
    """
    :param theano.Variable y_p: output of last time-frame. 2d (batch,dim)
    :return: z_re
    :rtype: theano.Variable
    """
    raise NotImplementedError


class AttentionTest(RecurrentTransformBase):
  name = "test"

  def init_vars(self):
    self.W_att_in = self.add_param(self.tt.fmatrix("W_att_in"))

  def step(self, y_p):
    z_re = T.dot(y_p, self.W_att_in)
    return z_re, {}


class AttentionInput(RecurrentTransformBase): # TODO: this isn't a recurrent transform
  """
  attention is just a sequence dependent bias (lstmp compatible)
  """

  name = "input"

  def create_vars(self):
    layer = self.layer
    base = layer.base
    assert base, "attention networks are only defined for decoder networks"
    # TODO ...
    src = []
    src_names = []
    n_in = 0
    for e in base:
      src_base = [ s for s in e.sources if s.name not in src_names ]
      src_names += [ s.name for s in e.sources ]
      src += [s.output for s in src_base]
      n_in += sum([s.attrs['n_out'] for s in src_base])
    self.xc = T.concatenate(src, axis=2)
    l = sqrt(6.) / sqrt(self.layer.attrs['n_out'] + n_in)
    values = numpy.asarray(self.layer.rng.uniform(low=-l, high=l, size=(n_in, 1)), dtype=theano.config.floatX)
    self.W_att_xc = theano.shared(value=values, borrow=True, name = "W_att_xc")
    self.add_param(self.W_att_xc)
    values = numpy.asarray(self.layer.rng.uniform(low=-l, high=l, size=(n_in, self.layer.attrs['n_out'] * 4)), dtype=theano.config.floatX)
    self.W_att_in = theano.shared(value=values, borrow=True, name = "W_att_in")
    self.add_param(self.W_att_in)
    zz = T.exp(T.dot(self.xc, self.W_att_xc)) # TB1
    self.zc = T.dot(T.sum(self.xc * (zz / T.sum(zz, axis=0, keepdims=True)).repeat(self.xc.shape[2],axis=2), axis=0, keepdims=True), self.W_att_in)

  def step(self, y_p):
    # TODO
    return T.unbroadcast(T.constant(numpy.array([[0]]), dtype="float32"), 0, 1), {}


class AttentionDot(RecurrentTransformBase):
  """
  attention over dot product of base outputs and time dependent activation
  """
  name = "attention_dot"

  def init_vars(self):
    self.B = self.add_input(self.tt.ftensor3("B"))  # base (output of encoder). (time,batch,encoder-dim)
    self.W_att_in = self.add_param(self.tt.fmatrix("W_att_in"))
    self.W_att_re = self.add_param(self.tt.fmatrix("W_att_re"))

  def create_vars(self):
    layer = self.layer
    base = layer.base
    assert base, "attention networks are only defined for decoder networks"
    unit = layer.unit

    # if attention_step > 0:
    #   if attention_beam == 0:
    #     attention_beam = attention_step
    # elif attention_step == -1:
    #   assert attention_beam > 0
    #   self.index_range = T.arange(self.index.shape[0], dtype='float32').dimshuffle(0,'x','x').repeat(self.index.shape[1],axis=1)
    # else:
    #   assert attention_beam == 0

    # if self.attrs['attention'] != 'none' and attention_step != 0:
    #   outputs_info.append(T.alloc(numpy.cast['int32'](0), index.shape[1])) # focus (B)
    #   outputs_info.append(T.cast(T.alloc(numpy.cast['int32'](0), index.shape[1]) + attention_beam,'int32')) # beam (B)

    n_in = sum([e.attrs['n_out'] for e in base])
    src = [e.output for e in base]
    l = sqrt(6.) / sqrt(layer.attrs['n_out'] + n_in + unit.n_re)

    self.xb = layer.add_param(layer.create_bias(n_in, name='b_att'))
    self.B = T.concatenate(src, axis=2) + self.xb  # == B
    self.B.name = "B"
    self.add_input(self.B)
    #if n_in != unit.n_out:
    #  values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(n_in, unit.n_units)), dtype=theano.config.floatX)
    #  self.W_att_proj = theano.shared(value=values, borrow=True, name = "W_att_proj")
    #  self.add_param(self.W_att_proj)
    #  self.xc = T.dot(self.xc, self.W_att_proj)
    #  n_in = unit.n_units
    values = numpy.asarray(layer.rng.uniform(low=-l, high=l, size=(layer.attrs['n_out'], n_in)), dtype=theano.config.floatX)
    self.W_att_re = self.add_param(theano.shared(value=values, borrow=True, name = "W_att_re"))
    values = numpy.asarray(layer.rng.uniform(low=-l, high=l, size=(n_in, layer.attrs['n_out'] * 4)), dtype=theano.config.floatX)
    self.W_att_in = self.add_param(theano.shared(value=values, borrow=True, name = "W_att_in"))

  def step(self, y_p):

    # #att_z = zc
    # att_x = xc  # == self.B
    # if attention_step != 0:
    #   focus_i = T.switch(T.ge(focus + beam,xc.shape[0]), xc.shape[0], focus + beam)
    #   focus_j = T.switch(T.lt(focus - beam,0), 0, focus - beam)
    #   focus_end = T.max(focus_i)
    #   focus_start = T.min(focus_j)
    #   #att_z = zc[focus_start:focus_end]
    #   att_x = xc[focus_start:focus_end]

    #f_z = T.sum(B * T.tanh(T.dot(y_p, W_att_quadr)).dimshuffle('x',0,1).repeat(B.shape[0],axis=0), axis=2, keepdims=True)
    f_z = T.sum(self.B * T.tanh(T.dot(y_p, self.W_att_re)).dimshuffle('x',0,1).repeat(self.B.shape[0],axis=0) / T.cast(self.B.shape[0],'float32'), axis=2, keepdims=True)
    f_e = T.exp(f_z)
    w_t = f_e / T.sum(f_e, axis=0, keepdims=True)

    import theano.printing
    #w_t = theano.printing.Print("w_t", attrs=['argmax(axis=0)'])(w_t)
    #w_t = theano.printing.Print("w_t",global_fn=print_wt)(w_t)
    z_re = T.dot(T.sum(self.B * w_t, axis=0, keepdims=False), self.W_att_in)

    # if attention_step == -1:
    #   #focus = focus_start + T.cast(T.mean(w_t,axis=0).flatten() * (focus_end - focus_start), 'int32')
    #   focus = T.cast(T.sum(w_t*self.index_range[focus_start:focus_end],axis=0).flatten() + 1,'int32') #T.cast(T.sum(T.arange(attention_beam, dtype='float32').dimshuffle(0,'x').repeat(w_t.shape[1],axis=1) * w_t, axis=0), 'int32')
    #   beam = T.cast(T.max([0.5 * T.exp(-T.sum(T.log(w_t)*w_t,axis=0)).flatten(),T.ones_like(beam)],axis=0),'int32') #T.cast(2.0 * T.max(-T.log(w_t),axis=0).flatten() * (focus_end - focus_start),'int32')
    #   result = [focus,beam] + result
    # elif attention_step > 0:
    #   result = [focus+attention_step,beam] + result

    return z_re, {}


class AttentionRBF(AttentionDot):
  """
  attention over rbf kernel of base outputs and time dependent activation
  """
  name = "attention_rbf"

  def init_vars(self):
    super(AttentionRBF, self).init_vars()
    self.sigma = self.add_var(self.tt.fscalar('sigma'))

  def create_vars(self):
    super(AttentionRBF, self).create_vars()
    self.sigma = self.add_var(theano.shared(numpy.cast['float32'](self.layer.attrs['attention_sigma']), name="sigma"))

  def step(self, y_p):
    f_z = -T.sqrt(T.sum(T.sqr(self.B - T.tanh(T.dot(y_p, self.W_att_re)).dimshuffle('x',0,1).repeat(self.B.shape[0],axis=0)), axis=2, keepdims=True)) / self.sigma
    f_e = T.exp(f_z)
    w_t = f_e / T.sum(f_e, axis=0, keepdims=True)
    return T.dot(T.sum(self.B * w_t, axis=0, keepdims=False), self.W_att_in), {}


class AttentionTimeGauss(RecurrentTransformBase):
  name = "attention_time_gauss"

  def init_vars(self):
    self.B = self.add_input(self.tt.ftensor3("B"))  # base (output of encoder). (time,batch,encoder-dim)
    self.W_att_in = self.add_param(self.tt.fmatrix("W_att_in"))
    self.W_att_re = self.add_param(self.tt.fmatrix("W_att_re"))
    self.t_max = self.add_var(self.tt.fscalar("t_max"))

  def create_vars(self):
    layer = self.layer
    base = layer.base
    assert base, "attention networks are only defined for decoder networks"

    n_out = layer.attrs['n_out']
    n_in = sum([e.attrs['n_out'] for e in base])
    src = [e.output for e in base]

    self.xb = layer.add_param(layer.create_bias(n_in, name='b_att'))
    self.B = T.concatenate(src, axis=2) + self.xb  # == B
    self.B.name = "B"
    self.add_input(self.B)

    self.W_att_re = self.add_param(layer.create_random_uniform_weights(n=n_out, m=2, p=n_out, name="W_att_re"))
    self.W_att_in = self.add_param(layer.create_random_uniform_weights(n=n_in, m=n_out * 4, name="W_att_in"))

    self.t = self.add_state_var(theano.shared(numpy.cast['float32'](0), name="t"))
    self.t_max = self.add_var(theano.shared(numpy.cast['float32'](5), name="t_max"))

  def step(self, y_p):

    a = T.nnet.sigmoid(T.dot(y_p, self.W_att_re))  # (batch,2)
    dt = T.nnet.sigmoid(a[:, 0]) * self.t_max
    std = T.nnet.sigmoid(a[:, 1]) * 5

    t_old = self.t
    t = t_old + dt

    # gauss window
    idxs = T.arange(self.B.shape[0]).dimshuffle(0, 'x')
    f_e = T.exp(((t - idxs) ** 2) / (2 * std ** 2))
    norm = T.constant(1.0, dtype="float32") / (std * T.constant(sqrt(2 * pi), dtype="float32"))
    w_t = f_e * norm

    z_re = T.dot(T.sum(self.B * w_t, axis=0, keepdims=False), self.W_att_in)
    return z_re, {self.t: t}



transforms = {}

def _setup():
  import inspect
  for clazz in globals().values():
    if not inspect.isclass(clazz): continue
    if not issubclass(clazz, RecurrentTransformBase): continue
    if clazz.name is None: continue
    transforms[clazz.name] = clazz

_setup()
