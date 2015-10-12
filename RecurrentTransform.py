
from math import sqrt, pi
import theano
import theano.tensor as T
import theano.sandbox.cuda as cuda
import numpy


class RecurrentTransformBase(object):
  name = None

  def __init__(self, force_gpu=False, layer=None, for_custom=False):
    """
    :type layer: NetworkRecurrentLayer.RecurrentUnitLayer
    """
    self.force_gpu = force_gpu
    if force_gpu:
      self.tt = cuda
    else:
      self.tt = T
    self.layer = layer
    self.input_vars = {}  # used as non_sequences for theano.scan(), i.e. as input for the step() function
    self.state_vars = {}  # updated in each step()
    self.state_vars_initial = {}
    self.custom_vars = {}
    self.for_custom = for_custom
    if not for_custom:
      transforms_by_id[id(self)] = self
      self.create_vars()

  def _create_var_for_custom(self, base_var):
    var = self._create_symbolic_var(base_var)
    setattr(self, var.name, var)
    return var

  def _create_symbolic_var(self, base_var):
    if self.force_gpu:
      base_type_class = cuda.CudaNdarrayType
    else:
      base_type_class = T.TensorType
    dtype = base_var.dtype
    ndim = base_var.ndim
    type_inst = base_type_class(dtype=dtype, broadcastable=(False,) * ndim)
    name = base_var.name
    var = type_inst(name)
    return var

  def create_vars_for_custom(self):
    """
    Called via CustomLSTMFunctions.
    """
    assert self.for_custom
    assert self.tt is cuda
    self.y_p = self.tt.fmatrix("y_p")

    layer_transform_instance = self.layer.recurrent_transform   # this is a different instance
    assert isinstance(layer_transform_instance, RecurrentTransformBase)
    assert layer_transform_instance.layer is self.layer
    for k, v in layer_transform_instance.custom_vars.items():
      assert getattr(layer_transform_instance, k) is v
      assert v.name == k
      self.add_var(self._create_var_for_custom(v))
    for k, v in layer_transform_instance.state_vars.items():
      assert getattr(layer_transform_instance, k) is v
      assert v.name == k
      self.add_state_var(self._create_var_for_custom(v))

  def init_vars(self):
    pass

  def create_vars(self):
    """
    Called for regular theano.scan().
    """
    pass

  def add_param(self, v):
    assert v.name
    if not self.for_custom:
      self.layer.add_param(v, v.name + "_" + self.name)
    self.add_var(v)
    return v

  def add_input(self, v, name=None):
    if name: v.name = name
    assert v.name
    self.input_vars[v.name] = v
    self.add_var(v)
    return v

  def add_state_var(self, initial_value, name=None):
    if name: initial_value.name = name
    assert initial_value.name
    sym_var = self._create_symbolic_var(initial_value)
    self.state_vars_initial[initial_value.name] = initial_value
    self.state_vars[initial_value.name] = sym_var
    return sym_var

  def add_var(self, v, name=None):
    if name: v.name = name
    assert v.name
    self.custom_vars[v.name] = v
    return v

  def get_sorted_non_sequence_inputs(self):
    return [v for (k, v) in sorted(self.input_vars.items())]

  def get_sorted_custom_vars(self):
    return [v for (k, v) in sorted(self.custom_vars.items())]

  def get_sorted_state_vars(self):
    return [v for (k, v) in sorted(self.state_vars.items())]

  def get_sorted_state_vars_initial(self):
    return [v for (k, v) in sorted(self.state_vars_initial.items())]

  def step(self, y_p):
    """
    :param theano.Variable y_p: output of last time-frame. 2d (batch,dim)
    :return: z_re
    :rtype: theano.Variable
    """
    raise NotImplementedError


class AttentionTest(RecurrentTransformBase):
  name = "test"

  def create_vars_for_custom(self):
    self.W_att_in = self.add_param(self.tt.fmatrix("W_att_in"))

  def step(self, y_p):
    z_re = T.dot(y_p, self.W_att_in)
    return z_re, {}


class DummyTransform(RecurrentTransformBase):
  name = "none"
  def step(self, y_p):
    return T.zeros((y_p.shape[0],y_p.shape[1]*4),dtype='float32'), {}


class AttentionBase(RecurrentTransformBase):
  """
  Attention base class
  """

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
    self.B = (T.concatenate(src, axis=2) + self.xb) * base[0].index.dimshuffle(0,1,'x').repeat(n_in,axis=2)  # == B
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


class AttentionDot(AttentionBase):
  """
  attention over dot product of base outputs and time dependent activation
  """
  name = "attention_dot"

  def create_vars(self):
    super(AttentionDot, self).create_vars()
    self.B = (self.B - T.mean(self.B, axis=0, keepdims=True)) / T.std(self.B,axis=0,keepdims=True)
    self.add_input(self.B, 'B')
    self.index = self.add_input(T.cast(self.layer.base[0].index, 'float32'), "index")

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
    w_t = f_e / T.sum(f_e, axis=0, keepdims=True) #- T.sum(T.ones_like(self.index)-self.index,axis=0,keepdims=True).dimshuffle(0,1,'x').repeat(f_e.shape[2],axis=2))

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


class AttentionRBF(AttentionBase):
  """
  attention over rbf kernel of base outputs and time dependent activation
  """
  name = "attention_rbf"
  def create_vars(self):
    super(AttentionRBF, self).create_vars()
    self.B = self.B - T.mean(self.B, axis=0, keepdims=True)
    self.add_input(self.B, 'B')
    self.sigma = self.add_var(theano.shared(numpy.cast['float32'](self.layer.attrs['attention_sigma']), name="sigma"))

  def step(self, y_p):
    f_z = -T.sqrt(T.sum(T.sqr(self.B - T.tanh(T.dot(y_p, self.W_att_re)).dimshuffle('x',0,1).repeat(self.B.shape[0],axis=0)), axis=2, keepdims=True)) / self.sigma
    f_e = T.exp(f_z)
    w_t = f_e / T.sum(f_e, axis=0, keepdims=True)
    return T.dot(T.sum(self.B * w_t, axis=0, keepdims=False), self.W_att_in), {}


class AttentionRBFLM(AttentionRBF):
  """
  attention over rbf kernel of base outputs and time dependent activation
  """
  name = "attention_rbf_lm"
  def create_vars(self):
    super(AttentionRBFLM, self).create_vars()
    #self.y_in = self.add_input(self.layer.y_in)
    self.W_lm_in = self.add_param(self.layer.W_lm_in)
    self.W_lm_out = self.add_param(self.layer.W_lm_out)
    #self.test_flag = self.add_var(theano.shared(value=numpy.asarray(1.0 if self.layer.train_flag else 0.0,dtype='float32'),name='test_flag')) #T.constant(0.0 if self.layer.train_flag else 1.0, 'float32'), 'train_flag')
    self.loop_weight = self.add_var(theano.shared(value=numpy.asarray(self.layer.attrs['droplm'] if self.layer.train_flag else 1.0,dtype='float32'),name='loop_weight'))
    #l = sqrt(6.) / sqrt(self.layer.unit.n_out + self.y_in[self.layer.attrs['target']].n_out)
    #values = numpy.asarray(self.layer.rng.uniform(low=-l, high=l, size=(self.layer.unit.n_out, self.layer.y_in[self.layer.attrs['target']].n_out)), dtype=theano.config.floatX)
    #self.W_lm_in = self.add_param(theano.shared(value=values, borrow=True, name = "W_lm_in"))
    #l = sqrt(6.) / sqrt(self.layer.unit.n_in + self.layer.y_in[self.layer.attrs['target']].n_out)
    #values = numpy.asarray(self.layer.rng.uniform(low=-l, high=l, size=(self.layer.y_in[self.layer.attrs['target']].n_out, self.layer.unit.n_in)), dtype=theano.config.floatX)
    #self.W_lm_out = self.add_param(theano.shared(value=values, borrow=True, name = "W_lm_out"))

  def step(self, y_p):
    z_re, updates = super(AttentionRBFLM, self).step(y_p)
    z_re += self.W_lm_out[T.argmax(T.dot(y_p,self.W_lm_in), axis=1)] * self.loop_weight #* self.test_flag
    return z_re, updates


class AttentionBeam(AttentionBase):
  """
  beam attention over rbf kernel of base outputs and time dependent activation
  """
  name = "attention_beam"

  def create_vars_for_custom(self):
    super(AttentionBeam, self).create_vars_for_custom()
    self.beam = self.add_var(self.tt.fscalar('beam'))
    self.focus = self.add_state_var(theano.shared(value=numpy.zeros((50,), dtype="float32"), name="focus"))
    #self.focus = self.add_state_var(self.tt.fvector('focus'))
    self.index_range = self.add_var(self.tt.fmatrix('index_range'))
    self.loc = self.add_var(self.tt.fvector('loc')) # T.alloc(numpy.cast['int32'](0), 1)

  def create_vars(self):
    super(AttentionBeam, self).create_vars()
    self.beam = self.add_var(theano.shared(numpy.cast['float32'](self.layer.attrs['attention_beam']), name="beam"))
    self.focus = self.add_state_var(theano.shared(value=numpy.zeros((50,), dtype="float32"), name="focus"))  # (batch,)
    self.loc = T.alloc(numpy.cast['int32'](0), self.layer.index.shape[1])
    #self.focus = self.add_state_var(T.alloc(numpy.cast['float32'](0), self.layer.index.shape[1]), "focus")
    self.index_range = self.add_var(T.arange(self.layer.index.shape[0], dtype='float32').dimshuffle(0,'x').repeat(self.layer.index.shape[1],axis=1), "index_range")

  def step(self, y_p):
    import theano.printing
    focus = T.cast(self.focus, 'int32')
    beam = T.cast(self.beam, 'int32')
    #self.loc = theano.printing.Print("loc")(self.loc)
    self.loc += 1
    focus = T.cast(self.loc, 'int32')
    focus = theano.printing.Print("focus")(focus)
    focus_i = T.switch(T.ge(focus + beam,self.B.shape[0]), self.B.shape[0], focus + beam) #+ self.loc
    focus_j = T.switch(T.lt(focus - 1,0), 0, focus - 1)
    focus_i = theano.printing.Print("focus_start")(focus_i)
    focus_j = theano.printing.Print("focus_end")(focus_j)
    focus_end = T.max(focus_i) #theano.printing.Print("focus_end", T.max(focus_i))
    focus_start = T.min(focus_j)
    att_x = self.B[focus_start:focus_end]

    f_z = -T.sqrt(T.sum(T.sqr(att_x - T.tanh(T.dot(y_p, self.W_att_re)).dimshuffle('x',0,1).repeat(att_x.shape[0],axis=0)), axis=2, keepdims=True)) #/ self.sigma
    f_e = T.exp(f_z)
    w_t = f_e / T.sum(f_e, axis=0, keepdims=True)

    #focus = T.cast(T.argmax(w_t,axis=0).dimshuffle(0) + focus_start, 'float32') #T.sum(w_t[:,:,0]*self.index_range[focus_start:focus_end],axis=0)
    #focus = T.cast(T.argmax(w_t,axis=0).dimshuffle(0) + focus_start, 'float32') #T.sum(w_t[:,:,0]*self.index_range[focus_start:focus_end],axis=0)
    #focus = T.sum(w_t.dimshuffle(0,1)*T.arange(w_t.shape[0],dtype='float32').dimshuffle(0,'x').repeat(w_t.shape[1],axis=1),axis=0) #+ T.cast(focus_start,'float32') # #T.cast(T.sum(T.arange(attention_beam, dtype='float32').dimshuffle(0,'x').repeat(w_t.shape[1],axis=1) * w_t, axis=0), 'int32')
    self.loc += 3.0 * T.sum(w_t.dimshuffle(0,1)*T.arange(w_t.shape[0],dtype='float32').dimshuffle(0,'x').repeat(w_t.shape[1],axis=1),axis=0) / T.cast(w_t.shape[0],'float32')
    #focus = self.focus + 1
    #focus = self.focus + 1
    #self.focus += 1
    #self.focus = theano.printing.Print("focus")(self.focus)
    #w_t = theano.printing.Print("w_t",global_fn=print_wt)(w_t)

    #self.beam = T.cast(T.max([0.5 * T.exp(-T.sum(T.log(w_t)*w_t,axis=0)).flatten(),T.ones_like(beam)],axis=0),'int32') #T.cast(2.0 * T.max(-T.log(w_t),axis=0).flatten() * (focus_end - focus_start),'int32')

    return T.dot(T.sum(att_x * w_t, axis=0, keepdims=False), self.W_att_in), {} # self.focus : focus }


class AttentionTimeGauss(RecurrentTransformBase):
  name = "attention_time_gauss"

  def create_vars(self):
    layer = self.layer
    base = layer.base
    assert base, "attention networks are only defined for decoder networks"

    n_out = layer.attrs['n_out']
    n_in = sum([e.attrs['n_out'] for e in base])
    src = [e.output for e in base]

    self.xb = layer.add_param(layer.create_bias(n_in, name='b_att'))
    self.B = T.concatenate(src, axis=2) + self.xb  # base (output of encoder). (time,batch,encoder-dim)
    self.B.name = "B"
    self.add_input(self.B)

    self.W_att_re = self.add_param(layer.create_random_uniform_weights(n=n_out, m=2, p=n_out, name="W_att_re"))
    self.W_att_in = self.add_param(layer.create_random_uniform_weights(n=n_in, m=n_out * 4, name="W_att_in"))

    self.t = self.add_state_var(T.zeros((self.B.shape[1],), dtype="float32"), name="t")  # (batch,)
    self.t_max = self.add_var(theano.shared(numpy.cast['float32'](5), name="t_max"))

  def step(self, y_p):
    # self.B is (time,batch,dim)
    a = T.nnet.sigmoid(T.dot(y_p, self.W_att_re))  # (batch,2)
    dt = T.nnet.sigmoid(a[:, 0]) * self.t_max  # (batch,)
    std = T.nnet.sigmoid(a[:, 1]) * 5  # (batch,)
    std_t_bc = std.dimshuffle('x', 0)

    t_old = self.t  # (batch,)
    t = t_old + dt
    t_bc = t.dimshuffle('x', 0)  # (time,batch)

    # gauss window
    idxs = T.cast(T.arange(self.B.shape[0]), dtype="float32").dimshuffle(0, 'x')  # (time,batch)
    f_e = T.exp(((t_bc - idxs) ** 2) / (2 * std_t_bc ** 2))  # (time,batch)
    norm = T.constant(1.0, dtype="float32") / (std_t_bc * T.constant(sqrt(2 * pi), dtype="float32"))  # (time,batch)
    w_t = f_e * norm
    w_t_bc = w_t.dimshuffle(0, 1, 'x')  # (time,batch,dim)

    z_re = T.dot(T.sum(self.B * w_t_bc, axis=0, keepdims=False), self.W_att_in)

    return z_re, {self.t: t}



transform_classes = {}
transforms_by_id = {}; ":type: dict[int,RecurrentTransformBase]"

def _setup():
  import inspect
  for clazz in globals().values():
    if not inspect.isclass(clazz): continue
    if not issubclass(clazz, RecurrentTransformBase): continue
    if clazz.name is None: continue
    transform_classes[clazz.name] = clazz

_setup()
