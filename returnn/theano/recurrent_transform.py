
from math import sqrt, pi
import theano
import theano.tensor as T
import theano.sandbox.cuda as cuda
import numpy
from returnn.theano.ops.multi_batch_beam import multi_batch_beam
from returnn.theano.activation_functions import elu
from theano.ifelse import ifelse


class RecurrentTransformBase(object):
  name = None

  def __init__(self, force_gpu=False, layer=None, for_custom=False):
    """
    :type layer: NetworkRecurrentLayer.RecurrentUnitLayer
    :param bool for_custom: When used with LSTMC + LSTMCustomOp, there are two instances of this class:
      One via the network initialization as part of the layer (for_custom == False)
      and another one via CustomLSTMFunctions (for_custom == True).
      The symbolic vars will look different. See self.create_vars_for_custom().
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
    if for_custom:
      self.create_vars_for_custom()
    else:
      transforms_by_id[id(self)] = self
      self.create_vars()

  def copy_for_custom(self, force_gpu=True):
    """
    :returns a new instance of this class for LSTMCustomOp
    """
    return self.__class__(force_gpu=force_gpu, for_custom=True, layer=self.layer)

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
    self.y_p = self.tt.fmatrix("y_p")

    layer_transform_instance = self.layer.recurrent_transform   # this is a different instance
    assert isinstance(layer_transform_instance, RecurrentTransformBase)
    assert layer_transform_instance.layer is self.layer
    for k, v in layer_transform_instance.custom_vars.items():
      assert getattr(layer_transform_instance, k) is v
      assert v.name == k
      self.custom_vars[k] = self._create_var_for_custom(v)
    self.state_vars_initial = None  # must not be used in custom op. we will get that from outside
    for k, v in layer_transform_instance.state_vars.items():
      assert getattr(layer_transform_instance, k) is v
      assert v.name == k
      self.state_vars[k] = self._create_var_for_custom(v)

  def init_vars(self):
    pass

  def create_vars(self):
    """
    Called for regular theano.scan().
    """
    pass

  def add_param(self, v, name = None, **kwargs):
    if name: v.name = name
    assert v.name
    if not self.for_custom:
      self.layer.add_param(v, v.name + "_" + self.name,**kwargs)
    self.add_var(v)
    return v

  def add_input(self, v, name=None):
    if name: v.name = name
    assert v.name, "missing name for input"
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

  def set_sorted_state_vars(self, state_vars):
    assert len(state_vars) == len(self.state_vars)
    for (k, v), v_new in zip(sorted(self.state_vars.items()), state_vars):
      assert getattr(self, k) is v
      assert v.name == k
      v_new.name = k
      self.state_vars[k] = v_new
      setattr(self, k, v_new)

  def get_state_vars_seq(self, state_var):
    assert state_var.name in self.state_vars
    idx = sorted(self.state_vars.keys()).index(state_var.name)
    return self.layer.unit.recurrent_transform_state_var_seqs[idx]

  def step(self, y_p):
    """
    :param theano.Variable y_p: output of last time-frame. 2d (batch,dim)
    :return: z_re, updates
    :rtype: (theano.Variable, dict[theano.Variable, theano.Variable])
    """
    raise NotImplementedError

  def cost(self):
    """
    :rtype: theano.Variable | None
    """
    return None


class AttentionTest(RecurrentTransformBase):
  name = "test"

  def create_vars(self):
    n_out = self.layer.attrs['n_out']
    n_in = sum([e.attrs['n_out'] for e in self.layer.base])
    self.W_att_in = self.add_param(self.layer.create_random_uniform_weights(n=n_out, m=n_in, name="W_att_in"))

  def step(self, y_p):
    z_re = T.dot(y_p, self.W_att_in)
    return z_re, {}


class DummyTransform(RecurrentTransformBase):
  name = "none"
  def step(self, y_p):
    return T.zeros((y_p.shape[0],y_p.shape[1]*4),dtype='float32'), {}


class DynamicTransform(RecurrentTransformBase):
  name = "rnn"
  def create_vars(self):
    self.W_re = self.add_var(self.layer.W_re, name="W_re")
  def step(self, y_p):
    return T.dot(y_p,self.W_re), {}


class BatchNormTransform(RecurrentTransformBase):
  name = "batch_norm"
  def create_vars(self):
    self.W_re = self.add_var(self.layer.W_re, name="W_re")
    dim = self.layer.unit.n_in
    self.sample_mean = self.add_param(theano.shared(numpy.zeros((dim,), 'float32')), "sample_mean")
    self.gamma = self.add_param(self.layer.shared(numpy.zeros((dim,), 'float32') + numpy.float32(0.1), "gamma"))
    #self.beta = self.add_param(self.layer.shared(numpy.zeros((dim,), 'float32'), "beta"))

  def batch_norm(self, h, use_shift=True, use_std=True, use_sample=0.0):
    x = h
    mean = T.mean(x, axis=0)
    std = T.std(x, axis=0)
    sample_std = T.sqrt(T.mean((x - self.sample_mean) ** 2, axis=0))
    if not self.layer.train_flag:
      use_sample = 1.0
    mean = T.constant(1. - use_sample, 'float32') * mean + T.constant(use_sample, 'float32') * self.sample_mean
    std = T.constant(1. - use_sample, 'float32') * std + T.constant(use_sample, 'float32') * sample_std
    mean = mean.dimshuffle('x', 0).repeat(h.shape[0], axis=0)
    std = std.dimshuffle('x', 0).repeat(h.shape[0], axis=0)
    bn = (h - mean) #/ (std + numpy.float32(1e-10))
    if use_std:
      bn *= self.gamma.dimshuffle('x', 0).repeat(h.shape[0], axis=0)
    #if use_shift:
    #  bn += self.beta
    return bn

  def step(self, y_p):
    #return T.dot(y_p,self.W_re), {}
    return self.batch_norm(T.dot(y_p,self.W_re)), {}


class LM(RecurrentTransformBase):
  name = "lm"

  def create_vars(self):
    self.W_lm_in = self.add_var(self.layer.W_lm_in, name="W_lm_in")
    self.W_lm_out = self.add_var(self.layer.W_lm_out, name="W_lm_out")
    self.lmmask = self.add_var(self.layer.lmmask, "lmmask")
    self.t = self.add_state_var(T.zeros((1,), dtype="float32"), name="t")
    y = self.layer.y_in[self.layer.attrs['target']].flatten()
    if self.layer.attrs['droplm'] < 1.0 and (self.layer.train_flag or self.layer.attrs['force_lm']):
      eos = T.unbroadcast(self.W_lm_out[0].dimshuffle('x','x',0),1).repeat(self.layer.index.shape[1],axis=1)
      if self.layer.attrs['direction'] == 1:
        y_t = self.W_lm_out[y].reshape((self.layer.index.shape[0],self.layer.index.shape[1],self.layer.unit.n_in))[:-1] # (T-1)BD
        self.cls = T.concatenate([eos, y_t], axis=0)
      else:
        y_t = self.W_lm_out[y].reshape((self.layer.index.shape[0],self.layer.index.shape[1],self.layer.unit.n_in))[1:] # (T-1)BD
        self.cls = T.concatenate([eos,y_t[::-1]], axis=0)
      self.add_input(self.cls, 'cls')

  def step(self, y_p):
    result = 0
    updates = {}
    p_re = T.nnet.softmax(T.dot(y_p, self.W_lm_in))
    if self.layer.attrs['droplm'] < 1.0 and (self.layer.train_flag or self.layer.attrs['force_lm']):
      mask = self.lmmask[T.cast(self.t[0],'int32')]
      if self.layer.attrs['attention_lm'] == "hard":
        result += self.W_lm_out[T.argmax(p_re, axis=1)] * (1. - mask) + self.cls[T.cast(self.t[0],'int32')] * mask
      else:
        result += T.dot(p_re,self.W_lm_out) * (1. - mask) + self.cls[T.cast(self.t[0],'int32')] * mask
    else:
      if self.layer.attrs['attention_lm'] == "hard":
        result += self.W_lm_out[T.argmax(p_re, axis=1)]
      else:
        result += T.dot(p_re,self.W_lm_out)
    updates[self.t] = self.t + 1
    return result, updates


class AttentionBase(RecurrentTransformBase):
  base=None
  name = "attention_base"

  @property
  def attrs(self):
    return { "_".join(k.split("_")[1:]) : self.layer.attrs[k].decode('utf-8') if isinstance(self.layer.attrs[k],bytes) else self.layer.attrs[k] for k in self.layer.attrs.keys() if k.startswith("attention_") }

  def create_vars(self):
    if self.base is None:
      self.base = self.layer.base
    self.n = self.add_state_var(T.zeros((self.layer.index.shape[1],), 'float32'), 'n')
    self.bound = self.add_input(T.cast(T.sum(self.layer.index,axis=0), 'float32'), 'bound')
    if self.attrs['norm'] == 'RNN':
      n_tmp = self.attrs['template']
      l = sqrt(6.) / sqrt(2 * n_tmp)
      values = numpy.asarray(self.layer.rng.uniform(low=-l, high=l, size=(n_tmp, n_tmp*4)), dtype=theano.config.floatX)
      self.N_re = self.add_param(self.layer.shared(value=values, borrow=True, name = "N_re"))
      values = numpy.asarray(self.layer.rng.uniform(low=-l, high=l, size=(n_tmp, 1)), dtype=theano.config.floatX)
      self.N_out = self.add_param(self.layer.shared(value=values, borrow=True, name = "N_out"))
    if self.attrs['distance'] == 'rnn':
      n_tmp = self.attrs['template']
      l = sqrt(6.) / sqrt(2 * n_tmp)
      values = numpy.asarray(self.layer.rng.uniform(low=-l, high=l, size=(n_tmp, n_tmp)), dtype=theano.config.floatX)
      self.A_re = self.add_param(self.layer.shared(value=values, borrow=True, name = "A_re"))
    if self.attrs['distance'] == 'transpose':
      n_tmp = self.attrs['template']
      l = sqrt(6.) / sqrt(2 * n_tmp)
      values = numpy.asarray(self.layer.rng.uniform(low=-l, high=l, size=(n_tmp,)), dtype=theano.config.floatX)
      self.W_T = self.add_param(self.layer.shared(value=values, name="W_T"))
    if self.attrs['lm'] != "none":
      self.W_lm_in = self.add_var(self.layer.W_lm_in, name="W_lm_in")
      self.b_lm_in = self.add_var(self.layer.b_lm_in, name="b_lm_in")
      self.W_lm_out = self.add_var(self.layer.W_lm_out, name="W_lm_out")
      self.drop_mask = self.add_var(self.layer.lmmask, "drop_mask")
      y = self.layer.y_in[self.layer.attrs['target']].flatten()
      nil = T.unbroadcast(self.W_lm_out[0].dimshuffle('x','x',0),1).repeat(self.layer.index.shape[1],axis=1)
      if self.layer.attrs['direction'] == 1:
        y_t = self.W_lm_out[y].reshape((self.layer.index.shape[0],self.layer.index.shape[1],self.layer.unit.n_in))[:-1] # (T-1)BD
        self.cls = T.concatenate([nil, y_t], axis=0)
      else:
        y_t = self.W_lm_out[y].reshape((self.layer.index.shape[0],self.layer.index.shape[1],self.layer.unit.n_in))[1:] # (T-1)BD
        self.cls = T.concatenate([nil,y_t[::-1]], axis=0)
      self.add_input(self.cls, 'cls')

  def default_updates(self):
    self.base = self.layer.base
    self.glimpses = [ [] ] * len(self.base)
    self.n_glm = max(self.attrs['glimpse'],1)
    return { self.n : self.n + numpy.float32(1) } #T.constant(1,'float32') }

  def step(self, y_p):
    result = 0
    self.glimpses = []
    updates = self.default_updates()
    if self.attrs['lm'] != "none":
      p_re = T.nnet.softmax(T.dot(y_p, self.W_lm_in) + self.b_lm_in)
      if self.layer.attrs['droplm'] < 1.0:
        mask = self.drop_mask[T.cast(self.n[0],'int32')]
        if self.attrs['lm'] == "hard":
          result += self.W_lm_out[T.argmax(p_re, axis=1)] * (1. - mask) + self.cls[T.cast(self.n[0],'int32')] * mask
        else:
          result += T.dot(p_re,self.W_lm_out) * (1. - mask) + self.cls[T.cast(self.n[0],'int32')] * mask
      else:
        if self.attrs['lm'] == "hard":
          result += self.W_lm_out[T.argmax(p_re, axis=1)]
        else:
          result += T.dot(p_re,self.W_lm_out)
    inp, upd = self.attend(y_p)
    updates.update(upd)
    return result + inp, updates

  def distance(self, C, H):
    dist = self.attrs['distance']
    if H.ndim == 2:
      H = H.dimshuffle('x', 0, 1).repeat(C.shape[0],axis=0)
    assert H.ndim == 3
    if dist == 'l2':
      dst = T.sqrt(T.sum((C - H) ** 2, axis=2))
    elif dist == 'logl2':
      dst = T.sqrt(T.sum((T.log((C + numpy.float32(1))/numpy.float32(2)) - T.log((H + numpy.float32(1))/numpy.float32(2))) ** 2, axis=2))
    elif dist == 'sqr':
      dst = T.mean((C - H) ** 2, axis=2)
    elif dist == 'dot':
      dst = T.sum(C * H, axis=2)
    elif dist == 'l1':
      dst = T.sum(T.abs_(C - H), axis=2)
    elif dist == 'cos': # use with template size > 32
      J = H / (T.sqrt(T.sum(H**2,axis=2,keepdims=True)) + T.constant(1e-5, 'float32'))
      K = C / (T.sqrt(T.sum(C**2,axis=2,keepdims=True)) + T.constant(1e-5, 'float32'))
      dst = T.sum(K * J, axis=2)
    elif dist == 'rnn':
      inp, _ = theano.scan(lambda x,p,W:elu(x+T.dot(p,W)), sequences = C, outputs_info = [H[0]], non_sequences=[self.A_re])
      dst = inp[-1]
    elif dist == 'transpose':
      dst = T.sum(self.W_T.dimshuffle('x','x',0).repeat(C.shape[0],axis=0).repeat(C.shape[1],axis=1) * T.tanh(C + H),axis=2)
    else:
      raise NotImplementedError()
    return dst #/ T.cast(H.shape[1],'float32')

  def beam(self, X, beam_idx=None):
    if not beam_idx:
      beam_idx = X.beam_idx
    input_shape = [X.shape[0] * X.shape[1]]
    if X.ndim == 3:
      input_shape.append(X.shape[2])
    Y = X.reshape(input_shape)[beam_idx].reshape((self.attrs['beam'],X.shape[1]))
    Y.beam_idx = beam_idx
    return Y

  def align(self, w_i, Q):
    dst = -T.log(w_i)
    inf = T.zeros_like(Q[0, 0]) + T.cast(1e10, 'float32') * T.gt(self.n, 0)
    big = T.cast(1e10, 'float32')
    n0 = T.eq(T.max(self.n), 0)
    D = -T.log(w_i)

    def dtw(i, q_p, b_p, Q, D, inf):
      i0 = T.eq(i, 0)
      # inf = T.cast(1e10,'float32') * T.cast(T.switch(T.eq(self.n,0), T.switch(T.eq(i,0), 0, 1), 1), 'float32')
      penalty = T.switch(T.and_(T.neg(n0), i0), big, T.constant(0.0, 'float32'))
      loop = T.constant(0.0, 'float32') + q_p
      forward = T.constant(0.0, 'float32') + T.switch(T.or_(n0, i0), 0, Q[i - 1])
      opt = T.stack([loop, forward])
      k_out = T.cast(T.argmin(opt, axis=0), 'int32')
      return opt[k_out, T.arange(opt.shape[1])] + D[i] + penalty, k_out

    output, _ = theano.scan(dtw, sequences=[T.arange(dst.shape[0], dtype='int32')], non_sequences=[Q, D, inf],
                            outputs_info=[T.zeros((dst.shape[1],), 'float32'), T.zeros((dst.shape[1],), 'int32')])
    return output[0], T.cast(output[1],'float32')

  def softmax(self, D, I):
    D = D * T.constant(self.attrs['sharpening'], 'float32')
    if self.attrs['norm'] == 'exp':
      D = D - D.mean(axis=0,keepdims=True) * I
      E = T.exp(-D)
      E = E / T.maximum(T.sum(E,axis=0,keepdims=True),T.constant(1e-20,'float32'))
    elif self.attrs['norm'] == 'linear':
      E = D * I
      E = numpy.float32(1) - E / T.maximum(T.sum(E,axis=0,keepdims=True),T.constant(1e-20,'float32'))
    elif self.attrs['norm'] == 'sigmoid':
      E = (numpy.float32(1) - T.tanh(D)**2) * I
    elif self.attrs['norm'] == 'lstm':
      n_out = self.attrs['template']
      def lstm(z, i_t, s_p, h_p):
        z += T.dot(h_p, self.N_re)
        i = T.outer(i_t, T.alloc(numpy.cast['int8'](1), n_out))
        ingate = T.nnet.sigmoid(z[:,n_out: 2 * n_out])
        forgetgate = T.nnet.sigmoid(z[:,2 * n_out:3 * n_out])
        outgate = T.nnet.sigmoid(z[:,3 * n_out:])
        input = T.tanh(z[:,:n_out])
        s_t = input * ingate + s_p * forgetgate
        h_t = T.tanh(s_t) * outgate
        return theano.gradient.grad_clip(s_t * i, -50, 50), h_t * i
      E, _ = theano.scan(lstm, sequences=[D,I], outputs_info=[T.zeros((n_out,), 'float32'), T.zeros((n_out,), 'int32')])
      E = T.nnet.sigmoid(T.dot(E,self.N_out))
    else:
      raise NotImplementedError()
    if self.attrs['nbest'] > 1:
      opt = T.minimum(self.attrs['nbest'], E.shape[0])
      score = (T.sort(E, axis=0)[-opt]).dimshuffle('x',0).repeat(E.shape[0],axis=0)
      E = T.switch(T.lt(E,score), T.zeros_like(E), E)
    return E


class AttentionList(AttentionBase):
  """
  attention over list of bases
  """
  name = "attention_list"
  def init(self, i):
    if self.attrs['beam'] > 0:
      img = 0
      for b in range(self.attrs['beam']):
        img += T.eye(self.custom_vars['C_%d' % i].shape[0],self.custom_vars['C_%d' % i].shape[0],b,dtype='float32')
      self.__setattr__("P_%d" % i, self.add_input(img, 'P_%d' %i))
    self.__setattr__("B_%d" % i, self.custom_vars['B_%d' % i])
    if self.attrs['memory'] > 0:
      self.__setattr__("M_%d" % i, self.state_vars['M_%d' % i])
      self.__setattr__("W_mem_in_%d" % i, self.custom_vars['W_mem_in_%d' % i])
      self.__setattr__("W_mem_write_%d" % i, self.custom_vars['W_mem_write_%d' % i])
    self.__setattr__("C_%d" % i, self.custom_vars['C_%d' % i])
    self.__setattr__("I_%d" % i, self.custom_vars['I_%d' % i])
    self.__setattr__("W_att_re_%d" % i, self.custom_vars['W_att_re_%d' % i])
    self.__setattr__("b_att_re_%d" % i, self.custom_vars['b_att_re_%d' % i])
    self.__setattr__("W_att_in_%d" % i, self.custom_vars['W_att_in_%d' % i])
    self.__setattr__("b_att_in_%d" % i, self.custom_vars['b_att_in_%d' % i])
    if 'b_att_bs_%d' % i in self.custom_vars.keys():
      self.__setattr__("W_att_bs_%d" % i, self.custom_vars['W_att_bs_%d' % i])
      self.__setattr__("b_att_bs_%d" % i, self.custom_vars['b_att_bs_%d' % i])
    shape = self.layer.base[i].output_index().shape
    if self.attrs['store']:
      self.__setattr__("att_%d" % i, self.add_state_var(T.zeros(shape,'float32'), "att_%d" % i))
    if self.attrs['smooth']:
      self.__setattr__("datt_%d" % i, self.add_state_var(T.zeros(shape, 'float32'), "datt_%d" % i))
    if self.attrs['momentum'] == "conv1d":
      self.__setattr__("F_%d" % i, self.custom_vars['F_%d' % i])
      self.__setattr__("U_%d" % i, self.custom_vars['U_%d' % i])
    elif self.attrs['momentum'] == "conv2d":
      self.__setattr__("F_%d" % i, self.custom_vars['F_%d' % i])
      self.__setattr__("U_%d" % i, self.custom_vars['U_%d' % i])
    elif self.attrs['momentum'] == "mono":
      self.__setattr__("D_in_%d" % i, self.custom_vars['D_in_%d' % i])
      self.__setattr__("D_out_%d" % i, self.custom_vars['D_out_%d' % i])
      self.__setattr__("Db_out_%d" % i, self.custom_vars['Db_out_%d' % i])
    if self.attrs['loss']:
      self.__setattr__("iatt_%d" % i, self.custom_vars['iatt_%d' % i])
      self.__setattr__("catt_%d" % i, self.add_state_var(T.zeros((shape[1],), 'float32'), "catt_%d" % i))

  def create_bias(self, n, name, i=-1):
    if i >= 0: name += '_%d' % i
    values = numpy.zeros((n,), dtype=theano.config.floatX)
    return self.add_param(self.layer.shared(value=values, borrow=True, name=name), name=name)

  def create_weights(self, n, m, name, i=-1):
    if i >= 0: name += '_%d' % i
    l = sqrt(6.) / sqrt(n + m)
    values = numpy.asarray(self.layer.rng.uniform(low=-l, high=l, size=(n, m)), dtype=theano.config.floatX)
    return self.add_param(self.layer.shared(value=values, borrow=True, name=name), name=name)

  def create_vars(self):
    super(AttentionList, self).create_vars()
    n_tmp = self.attrs['template']
    direction = self.layer.attrs['direction']
    #self.W_re = self.add_var(self.layer.W_re, name="W_re")
    for i,e in enumerate(self.base):
      # base output
      B = e.output[::direction]
      self.add_input(B, 'B_%d' % i)
      # mapping from base output to template size
      self.create_weights(self.layer.attrs['n_out'], n_tmp, "W_att_re", i)
      self.create_bias(n_tmp, "b_att_re", i)
      if e.attrs['n_out'] == n_tmp:
        self.add_input(e.output[::direction], 'C_%d' % i)
      else:
        W_att_bs = self.create_weights(e.attrs['n_out'], n_tmp, "W_att_bs", i)
        b_att_bs = self.create_bias(n_tmp, "b_att_bs", i)
        h_att = T.tanh(T.dot(B, W_att_bs) + b_att_bs)
        if self.attrs['bn']:
          h_att = self.layer.batch_norm(h_att, n_tmp, index = e.output_index())
        else:
          i_f = T.cast(e.output_index()[::self.layer.attrs['direction']],'float32').dimshuffle(0,1,'x').repeat(h_att.shape[2],axis=2)
          h_att = h_att - (h_att * i_f).sum(axis=0,keepdims=True) / T.sum(i_f,axis=0,keepdims=True)
        if self.attrs['memory'] > 0:
          self.add_state_var(T.zeros((self.attrs['memory'], n_tmp), 'float32'), 'M_%d' % i)
          self.create_weights(n_tmp, self.layer.unit.n_in, "W_mem_in", i)
          self.create_weights(n_tmp, self.attrs['memory'], "W_mem_write", i)
        self.add_input(h_att, 'C_%d' % i)
      self.add_input(T.cast(self.base[i].output_index()[::direction], 'float32'), 'I_%d' % i)
      # mapping from template size to cell input
      self.create_weights(e.attrs['n_out'], self.layer.unit.n_in, "W_att_in", i)
      self.create_bias(self.layer.unit.n_in, "b_att_in", i)
      if self.attrs['momentum'] == 'conv1d':
        context = 5
        values = numpy.ones((self.attrs['filters'], 1, context, 1), 'float32')
        self.add_param(self.layer.shared(value=values, borrow=True, name="F_%d" % i))
        l = sqrt(6.) / sqrt(self.layer.attrs['n_out'] + n_tmp + self.layer.unit.n_re)
        values = numpy.asarray(self.layer.rng.uniform(low=-l, high=l, size=(self.attrs['filters'], n_tmp)), dtype=theano.config.floatX)
        self.add_param(self.layer.shared(value=values, borrow=True, name="U_%d" % i))
      elif self.attrs['momentum'] == 'conv2d':
        context = 3
        values = numpy.ones((self.attrs['filters'], 1, 2, context), 'float32')
        self.add_param(self.layer.shared(value=values, borrow=True, name="F_%d" % i))
        l = sqrt(6.) / sqrt(self.attrs['filters'] + 1)
        values = numpy.asarray(self.layer.rng.uniform(low=-l, high=l, size=(self.attrs['filters'], 1)), dtype=theano.config.floatX)
        self.add_param(self.layer.shared(value=values, borrow=True, name="U_%d" % i))
      elif self.attrs['momentum'] == "mono":
        size = 500
        l = 0.01
        values = numpy.asarray(self.layer.rng.uniform(low=-l, high=l, size=(1, size)),
                               dtype=theano.config.floatX)
        self.add_param(self.layer.shared(value=values, borrow=True, name="D_in_%d" % i))
        values = numpy.asarray(self.layer.rng.uniform(low=-l, high=l, size=(size, 1)),
                               dtype=theano.config.floatX)
        self.add_param(self.layer.shared(value=values, borrow=True, name="D_out_%d" % i))
        self.add_param(self.layer.shared(value=numpy.zeros((1,),'float32'), borrow=True, name="Db_out_%d" % i))
      elif self.attrs['loss']:
        att = e.att - T.arange(e.att.shape[1]) * e.sources[0].index.shape[0] # NB
        self.add_input(T.cast(att,'float32'), 'iatt_%d' % i)
      self.init(i)

  def item(self, name, i):
    key = "%s_%d" % (name,i)
    return self.custom_vars[key] if key in self.custom_vars.keys() else self.state_vars[key]

  def get(self, y_p, i, g):
    W_att_re = self.item("W_att_re", i)
    b_att_re = self.item("b_att_re", i)
    B = self.item("B", i)
    C = self.item("C", i)
    I = self.item("I", i)
    beam_size = T.minimum(numpy.int32(abs(self.attrs['beam'])), C.shape[0])
    loc = T.cast(T.maximum(T.minimum(T.sum(I,axis=0) * self.n / self.bound - beam_size / 2, T.sum(I,axis=0) - beam_size), 0),'int32')
    if self.attrs['beam'] > 0:
      beam_idx = (self.custom_vars[('P_%d' % i)][loc].dimshuffle(1,0).flatten() > 0).nonzero()
      I = I.reshape((I.shape[0]*I.shape[1],))[beam_idx].reshape((beam_size,I.shape[1]))
      C = C.reshape((C.shape[0]*C.shape[1],C.shape[2]))[beam_idx].reshape((beam_size,C.shape[1],C.shape[2]))
      B = B.reshape((B.shape[0]*B.shape[1],B.shape[2]))[beam_idx].reshape((beam_size,B.shape[1],B.shape[2]))
    if self.attrs['template'] != self.layer.unit.n_out:
      z_p = T.dot(y_p, W_att_re) + b_att_re
    else:
      z_p = y_p
    if self.attrs['momentum'] == 'conv1d':
      from theano.tensor.nnet import conv
      att = self.item('att', i)
      F = self.item("F", i)
      v = T.dot(T.sum(conv.conv2d(border_mode='full',
        input=att.dimshuffle(1, 'x', 0, 'x'),
        filters=F).dimshuffle(2,3,0,1),axis=1)[F.shape[2]/2:-F.shape[2]/2+1],self.item("U",i))
      v = I * v / v.sum(axis=0,keepdims=True)
      z_p += T.sum(C * v,axis=0)
    if g > 0:
      z_p += self.glimpses[i][-1]
    h_p = T.tanh(z_p)
    return B, C, I, h_p, self.item("W_att_in", i), self.item("b_att_in", i)

  def attend(self, y_p):
    inp, updates = 0, {}
    for i in range(len(self.base)):
      for g in range(self.n_glm):
        B, C, I, H, W_att_in, b_att_in = self.get(y_p, i, g)
        z_i = self.distance(C, H)
        w_i = self.softmax(z_i, I)
        if self.attrs['momentum'] == 'conv2d':
          F = self.item('F',i)
          context = F.shape[3]
          padding = T.zeros((2,context/2,C.shape[1]),'float32')
          att = T.concatenate([padding, T.stack([self.item('att',i), w_i]), padding],axis=1) # 2TB
          v_i = T.nnet.sigmoid(T.dot(T.nnet.conv2d(border_mode='valid',
                              input=att.dimshuffle(2,'x',0,1), # B12T
                              filters=F).dimshuffle(3,0,2,1),self.item('U',i)).reshape((C.shape[0],C.shape[1])))
          w_i *= v_i
          w_i = w_i / w_i.sum(axis=0, keepdims=True)
        elif self.attrs['momentum'] == 'mono': # gating function
          idx = T.arange(z_i.shape[0],dtype='float32').dimshuffle(0,'x').repeat(w_i.shape[1],axis=1) # TB
          d_i = idx - T.sum(self.item('att', i) * idx,axis=0,keepdims=True)
          f_i = T.nnet.sigmoid(T.dot(T.tanh(T.dot(d_i.dimshuffle(0,1,'x'), self.item('D_in', i))), self.item("D_out", i)) + self.item('Db_out',i))[:,:,0]
          w_i = T.exp(-z_i) * f_i * I
          w_i = w_i / w_i.sum(axis=0, keepdims=True)
        self.glimpses[i].append(T.sum(C * w_i.dimshuffle(0,1,'x').repeat(C.shape[2],axis=2),axis=0))
      if self.attrs['smooth']:
        updates[self.state_vars['datt_%d' % i]] = w_i - self.state_vars['att_%d' % i]
      if self.attrs['store']:
        updates[self.state_vars['att_%d' % i]] = theano.gradient.disconnected_grad(w_i)
      if self.attrs['memory'] > 0:
        M = self.item('M',i)
        z_r = self.distance(M, H)
        w_m = self.softmax(z_r, T.ones_like(M[0]))
        inp += T.dot(T.sum(w_m*M,axis=0), self.item('W_mem_in',i))
        v_m = T.nnet.sigmoid(T.dot(H, self.item('W_mem_write', i))).dimshuffle('x',0, 1).repeat(M.shape[0],axis=0)
        mem = H.dimshuffle('x',0,1).repeat(self.attrs['memory'],axis=0)
        updates[self.state_vars['M_%d' % i]] = T.sum((numpy.float32(1) - v_m) * M.dimshuffle(0,'x',1).repeat(v_m.shape[1],axis=1) + v_m * mem,axis=1)
      if self.attrs['accumulator'] == 'rnn':
        def rnn(x_t, w_t, c_p):
          c = x_t * w_t + c_p * (numpy.float32(1.) - w_t)
          return T.switch(T.ge(c, 0), c, T.exp(c) - 1)
        zT, _ = theano.scan(rnn, sequences=[B,w_i.dimshuffle(0, 1, 'x').repeat(B.shape[2], axis=2)],
                           outputs_info = [T.zeros_like(B[0])])
        z = zT[-1]
      else:
        if self.attrs['nbest'] == 1:
          z = B[T.argmax(w_i,axis=0),T.arange(w_i.shape[1])]
        else:
          z = T.sum(B * w_i.dimshuffle(0, 1, 'x').repeat(B.shape[2], axis=2), axis=0)
      if self.attrs['loss']:
        updates[self.state_vars['catt_%d' % i]] = -T.sum(T.log(w_i[T.cast(self.item('iatt', i),'int32')[T.cast(self.n,'int32')],T.arange(w_i.shape[1],dtype='int32')]),axis=0)
      inp += T.dot(z, W_att_in) + b_att_in
    ifelse(T.eq(T.mod(self.n[0],self.attrs['ndec']),0), inp, T.zeros((self.n.shape[0],self.layer.attrs['n_out'] * 4),'float32'))
    return inp, updates

  def cost(self):
    val = 0
    if self.attrs['smooth']:
      penalty = T.constant(0,'float32')
      for i in range(len(self.base)):
        penalty += theano.tensor.extra_ops.cumsum(self.get_state_vars_seq(self.state_vars['datt_%d' % i]),axis=0)
      val += T.sum(T.maximum(penalty,T.zeros_like(penalty)))
    if self.attrs['loss']:
      for i in range(len(self.base)):
        val += T.sum(self.get_state_vars_seq(self.state_vars['catt_%d' % i]))
    return val


class AttentionAlign(AttentionBase):
  """
  alignment controlled attention
  """
  name = "attention_align"
  def create_vars(self):
    super(AttentionAlign, self).create_vars()
    assert len(self.base) == 1
    #assert self.base[0].layer_class.endswith('align')
    max_skip = self.base[0].attrs['max_skip']
    self.B = self.add_input(self.base[0].output, 'B')
    l = sqrt(6.) / sqrt(self.layer.attrs['n_out'] + max_skip)
    values = numpy.asarray(self.layer.rng.uniform(low=-l, high=l,
                           size=(self.base[0].attrs['n_out'], self.layer.unit.n_in)),
                           dtype=theano.config.floatX)
    self.W_att_in = self.add_param(self.layer.shared(value=values, borrow=True, name="W_att_in"), name="W_att_in")

    self.T_W = self.add_var(self.layer.T_W, name="T_W")
    self.T_b = self.add_var(self.layer.T_b, name="T_b")

    #y_t = T.dot(self.base[0].attention, T.arange(self.base[0].output.shape[0], dtype='float32'))  # NB
    #y_t = T.concatenate([T.zeros_like(y_t[:1]), y_t], axis=0)  # (N+1)B
    #y_t = y_t[1:] - y_t[:-1] # NB
    self.y_t = self.add_input(self.layer.y_t, "y_t")

    lens = T.sum(self.base[0].index,axis=0,dtype='float32')
    self.t = self.add_state_var(lens - numpy.float32(1), "t")
    nlens = T.sum(self.layer.index,axis=0,dtype='float32')
    self.ns = self.add_state_var(nlens - numpy.float32(1), "ns")
    #self.cost_sum = self.add_state_var(T.zeros((1,), 'float32'), "cost_sum")


  def attend(self, y_p):
    inp, updates = 0, {}
    z = T.dot(y_p,self.T_W) + self.T_b
    #idx = self.I[self.n[0]]
    #y_out = T.cast(self.y_t[self.n[0]],'int32')
    #nll, _ = T.nnet.crossentropy_softmax_1hot(x=z[idx], y_idx=y_out[idx])
    smooth = T.constant(self.attrs['smooth'], 'float32')
    #n = T.cast(self.n[0],'int32')
    n = T.cast(self.ns, 'int32')
    t = T.dot(T.nnet.softmax(z), T.arange(self.base[0].attrs['max_skip'],dtype='float32')) #+ numpy.float32(1)
    #t = T.cast(T.argmax(z,axis=1), 'float32' )
    t = smooth * self.y_t[n,T.arange(self.y_t.shape[1]),T.cast(self.t,'int32')] + (numpy.float32(1) - smooth) * t
    pos = T.cast(T.ceil(self.t), 'int32')
    inp = T.dot(self.B[pos,T.arange(pos.shape[0])], self.W_att_in)
    #updates[self.cost_sum] = T.sum(nll,dtype='float32').dimshuffle('x').repeat(1,axis=0)
    updates[self.t] = T.maximum(self.t - t, numpy.float32(0))
    updates[self.ns] = self.ns - numpy.float32(1)
    return inp, updates


class AttentionInverted(AttentionBase):
  """
  alignment controlled attention
  """
  name = "attention_inverted"
  def create_vars(self):
    super(AttentionInverted, self).create_vars()
    assert len(self.base) == 1
    assert self.base[0].layer_class.endswith('align')
    align = self.base[0]
    dir = -self.layer.attrs['direction']
    self.max_skip = numpy.int32(self.layer.base[0].attrs['max_skip'])
    p_in = T.concatenate([T.zeros_like(align.p_y_given_x[:self.max_skip]), align.p_y_given_x[::dir]], axis=0)
    x_in = T.concatenate([T.zeros_like(align.x_in[:self.max_skip]), align.x_in[::dir]], axis=0)
    a_in = T.concatenate([T.zeros_like(align.attention.dimshuffle(2,1,0)[:self.max_skip]),
                          align.attention.dimshuffle(2,1,0)[::dir]], axis=0)
    self.P = self.add_input(p_in, 'P')
    self.X = self.add_input(x_in, 'X')
    self.A = self.add_input(a_in, 'A')
    l = sqrt(6.) / sqrt(self.layer.attrs['n_out'] + self.layer.unit.n_in)
    values = numpy.asarray(self.layer.rng.uniform(low=-l, high=l,
                                                  size=(self.layer.attrs['n_out'], align.n_cls)),
                                                  dtype=theano.config.floatX)
    self.W_cls = self.add_param(self.layer.shared(value=values, borrow=True, name="W_cls"), name="W_cls")
    values = numpy.zeros((align.n_cls,), 'float32')
    self.b_cls = self.add_param(self.layer.shared(value=values, borrow=True, name='b_cls'), name='b_cls')
    values = numpy.asarray(self.layer.rng.uniform(low=-l, high=l,
                           size=(align.attrs['n_out'], self.layer.unit.n_in)),
                           dtype=theano.config.floatX)
    self.W_in = self.add_param(self.layer.shared(value=values, borrow=True, name="W_in"), name="W_in")

    lens = T.sum(self.base[0].index,axis=0,dtype='float32')
    self.t = self.add_state_var(lens - numpy.float32(self.max_skip), "t")
    self.max_skip = self.add_var(T.zeros((1,),'float32') + numpy.float32(self.max_skip),'max_skip')
    nlens = T.sum(self.layer.index,axis=0,dtype='float32')
    self.ns = self.add_state_var(nlens - numpy.float32(1), "ns")

  def attend(self, y_p):
    inp, updates = 0, {}
    c = T.nnet.softmax(T.dot(y_p, self.W_cls) + self.b_cls) # BC
    n = T.cast(self.ns - numpy.float32(1),'int32')[0]
    tau = T.cast(self.t,'int32')[0]
    max_skip = T.cast(self.max_skip, 'int32')[0]
    #max_skip = numpy.int32(self.layer.base[0].attrs['max_skip'])
    #max_skip = 12
    p = self.P[tau:tau + max_skip] # MBC
    x = self.X[tau:tau + max_skip]
    a = self.A[tau:tau + max_skip,T.arange(x.shape[1]),n] # MB
    a = self.A[:,T.arange(x.shape[1]),n]
    e = T.exp(T.sum(c.dimshuffle('x',0,1).repeat(p.shape[0],axis=0) * p, axis=2)) # MB
    e = e / e.sum(axis=0,keepdims=True)

    w = a

    #e = e.dimshuffle(0,1,'x').repeat(p.shape[2],axis=2)
    q = T.exp(p.max(axis=2) * w)
    q = q / q.sum(axis=0,keepdims=True)

    q = w
    from returnn.theano.util import print_to_file
    #q = print_to_file('q', q)
    dt = q.argmax(axis=0) - T.cast(self.t,'int32') #+ max_skip

    pos = T.argmax(self.A[:,T.arange(x.shape[1]),n],axis=0)
    inp = T.dot(self.X[pos,T.arange(x.shape[1])], self.W_in)
    #q = q.dimshuffle(0,1,'x').repeat(x.shape[2],axis=2)
    #inp = T.dot(T.sum(x * q, axis=0), self.W_in)
    #updates[self.t] = T.maximum(self.t - self.max_skip[0] + T.cast(dt, 'float32'), T.zeros_like(self.t))
    n = T.cast(self.ns - numpy.float32(1), 'int32')[0]
    updates[self.t] = T.cast(T.argmax(self.A[:,T.arange(x.shape[1]),n],axis=0),'float32')
    updates[self.ns] = self.ns - numpy.float32(1)
    return inp, updates

class AttentionSegment(AttentionBase):
  """
  alignment controlled attention over segments
  """
  name = "attention_segment"

  def create_bias(self, n, name, i=-1):
    if i >= 0: name += '_%d' % i
    values = numpy.zeros((n,), dtype=theano.config.floatX)
    return self.add_param(self.layer.shared(value=values, borrow=True, name=name), name=name)

  def create_weights(self, n, m, name, i=-1):
    if i >= 0: name += '_%d' % i
    l = sqrt(6.) / sqrt(n + m)
    values = numpy.asarray(self.layer.rng.uniform(low=-l, high=l, size=(n, m)), dtype=theano.config.floatX)
    return self.add_param(self.layer.shared(value=values, borrow=True, name=name), name=name)

  def create_vars(self):
    super(AttentionSegment, self).create_vars()
    assert len(self.base) == 1
    n_tmp = self.attrs['template']
    B = self.B = self.add_input(self.base[0].output[::self.layer.attrs['direction']], 'B')

    self.W_att_in = self.create_weights(self.base[0].attrs['n_out'], self.layer.unit.n_in, 'W_att_in')
    self.b_att_in = self.create_bias(self.layer.unit.n_in, 'b_att_in')
    self.epoch = self.add_input(T.cast(self.layer.network.epoch,'float32'),'epoch')
    if not self.layer.attrs['n_out'] == n_tmp:
      if self.layer.attrs['attention_alnpts']:
        self.W_att_re = self.create_weights(self.layer.attrs['n_out'], n_tmp, "W_att_re")
        self.b_att_re = self.create_bias(n_tmp, "b_att_re")
      self.W_att_dec = self.create_weights(self.layer.attrs['n_out'], n_tmp, "W_att_dec")
      self.b_att_dec = self.create_bias(n_tmp, "b_att_dec")

    if not self.base[0].attrs['n_out'] == n_tmp:
      self.W_att_bs = self.create_weights(self.base[0].attrs['n_out'], n_tmp, "W_att_bs")
      self.b_att_bs = self.create_bias(n_tmp, "b_att_bs")
      h_att = T.tanh(T.dot(B,self.W_att_bs) + self.b_att_bs)
    else:
      h_att = B
    self.I_dec = self.add_input(T.cast(self.base[0].output_index()[::self.layer.attrs['direction']],'float32'), 'I_dec')
    self.i_f = self.add_input(T.cast(self.base[0].output_index()[::self.layer.attrs['direction']],'float32').dimshuffle(0,1,'x').repeat(h_att.shape[2],axis=2),'i_f')
    if not self.layer.eval_flag:
      self.inv_att = self.add_input(T.cast(self.layer.aligner.attention.dimshuffle(2,1,0)[::self.layer.attrs['direction']].dimshuffle(2,1,0),'float32'),'inv_att')
      self.red_ind = self.add_input(T.cast(self.layer.aligner.reduced_index,'float32'),'red_ind')
      self.i_f = self.add_input(T.cast(self.base[0].output_index()[::self.layer.attrs['direction']],'float32').dimshuffle(0,1,'x').repeat(h_att.shape[2],axis=2),'i_f')
      self.index_att = self.add_input(self.make_index(self.inv_att,self.I_dec),'index_att') #NTB
    if not self.base[0].attrs['n_out'] == n_tmp:
      h_att = h_att - (h_att * self.i_f).sum(axis=0,keepdims=True) / T.sum(self.i_f,axis=0,keepdims=True)
      self.C = self.add_input(h_att, 'C')
    else:
      self.C = self.add_input(self.base[0].output[::self.layer.attrs['direction']], 'C')
    self.E = self.add_input(T.concatenate([e.output[::self.layer.attrs['direction']] for e in self.layer.encoder],axis=2), 'E')

  def make_index(self,inv_att,ind):
    att = inv_att.argmax(axis=2) #NB
    new_ind = T.zeros_like(ind).dimshuffle('x',0,1).repeat(att.shape[0],axis=0).dimshuffle(0,2,1) #NBT
    mask = T.arange(ind.shape[0]).dimshuffle('x',0).repeat(att.shape[0]*att.shape[1],axis=0).reshape((att.shape[0],att.shape[1],ind.shape[0])) #NBT
    flat_att = att.flatten().dimshuffle(0,'x').repeat(ind.shape[0],axis=1).reshape((att.shape[0],att.shape[1],ind.shape[0])) #NBT
    result = T.switch(mask>flat_att,new_ind,numpy.float32(1))
    result = T.switch(T.eq(flat_att,0),numpy.float32(0),result).dimshuffle(0,2,1)
    return T.cast(result,'float32')

  def calc_temperature(self,method="epoch",min_dist=None):
    att_epoch = numpy.float32(self.layer.attrs['attention_epoch'])
    att_step = numpy.float32(self.layer.attrs['attention_segstep'])
    att_offset = numpy.float32(self.layer.attrs['attention_offset'])
    att_scale = numpy.float32(self.layer.attrs['attention_scale'])
    temperature = T.cast(T.cast(self.epoch/att_epoch,'int32') * att_step + att_offset,'float32')
    if method == "epoch":
      temperature = T.minimum(temperature,numpy.float32(1.0))
    elif method == "min_dist":
      assert min_dist is not None
      temperature = T.maximum(T.exp(-min_dist),T.minimum(temperature,numpy.float32(1.0)))
    elif method == "entropy":
      assert min_dist is not None
      exp_min_dist = T.exp(att_scale/T.cast(min_dist,'float32'))
      temperature = numpy.float32(1) - T.minimum(exp_min_dist,numpy.float32(1.0))
    elif method == "entropy_direct":
      assert min_dist is not None
      exp_min_dist = T.exp(T.cast(min_dist,'float32')*numpy.float32(0.5))
      temperature = numpy.float32(1) - T.minimum(exp_min_dist,numpy.float32(1.0))
    elif method == "entropy_batch_avg":
      assert min_dist is not None
      avg_entropy = T.sum(min_dist,dtype='float32')/T.cast(min_dist.shape[0],'float32')
      exp_min_dist = T.exp(att_scale/T.cast(avg_entropy,'float32'))
      temperature = numpy.float32(1) - T.minimum(exp_min_dist,numpy.float32(1.0))
    elif method == "entropy_batch_min":
      assert min_dist is not None
      min_entropy = T.max(min_dist)
      exp_min_dist = T.exp(att_scale/T.cast(min_entropy,'float32'))
      temperature = numpy.float32(1) - T.minimum(exp_min_dist,numpy.float32(1.0))
    return temperature

  def attend(self, y_p):
    inp, updates = 0, {}
    n = T.cast(self.n[0],'int32')
    attend_on_alnpts = self.layer.attrs['attention_alnpts']
    att_method = self.layer.attrs['attention_method']
    if not attend_on_alnpts:
      #if not self.layer.eval_flag:
      if self.layer.train_flag:
        att_pts = self.inv_att.argmax(axis=2) + T.arange(self.inv_att.shape[1])*self.inv_att.shape[2] #NB
        curr_enc_pts = T.cast(att_pts[n],'int32') #B

        if self.layer.attrs['n_out'] == self.layer.attrs['attention_template']:
          dis_curr = y_p

        else:
          prev_dec_step = T.dot(y_p,self.W_att_dec) + self.b_att_dec #BD
          dis_curr = prev_dec_step
        curr_seg_index = T.switch(T.gt(self.index_att[n] - self.index_att[n-1],numpy.float32(0)),numpy.float32(1),numpy.float32(0)) #TB
        ind_curr = theano.ifelse.ifelse(n > 0, curr_seg_index,self.index_att[n])
        e1 = self.distance(self.C, T.tanh(dis_curr)) #TB
        att_w1 = self.softmax(e1, ind_curr)
        att_w2 = self.softmax(e1, self.I_dec)

        if att_method == 'min_dist':
          min_dist = T.min(e1,axis=0) #B
        elif att_method.startswith("entropy"):
          log_alpha = T.log(T.maximum(att_w2,numpy.float32(1e-7)))
          min_dist = T.sum(att_w2 * log_alpha,axis=0) #B
        else:
          min_dist = None
        temperature = self.calc_temperature(att_method,min_dist)
        temperature = theano.ifelse.ifelse(self.epoch > numpy.float32(self.layer.attrs['attention_epoch']),T.ones_like(temperature),temperature)
        att_w = (numpy.float32(1) - temperature) * att_w1 + temperature * att_w2

      else:
        if self.layer.attrs['n_out'] == self.layer.attrs['attention_template']:
          dis_curr = y_p
        else:
          dis_curr = T.dot(y_p,self.W_att_dec) + self.b_att_dec
        e1 = self.distance(self.C, T.tanh(dis_curr))
        att_w = self.softmax(e1,self.I_dec)

      z = T.sum(self.B * att_w.dimshuffle(0, 1, 'x').repeat(self.B.shape[2], axis=2), axis=0)

    else:
      if not self.layer.eval_flag:
        att_pts = self.inv_att.argmax(axis=2) + T.arange(self.inv_att.shape[1]) * self.inv_att.shape[2]  # NB
        if self.layer.attrs['n_out'] == self.layer.attrs['attention_template']:
          C = self.E.dimshuffle(1, 0, 2).reshape((self.E.shape[0] * self.E.shape[1], self.E.shape[2]))[att_pts] #NBD
          dis_curr = y_p
        else:
          C = T.dot(
            self.E.dimshuffle(1, 0, 2).reshape((self.E.shape[0] * self.E.shape[1], self.E.shape[2]))[att_pts],
            self.W_att_re) + self.b_att_re  # NBD
          prev_dec_step = T.dot(y_p,self.W_att_dec) + self.b_att_dec #BD
          dis_curr = prev_dec_step
        ind_curr = self.red_ind
        e1 = self.distance(C,T.tanh(dis_curr))
        att_w = self.softmax(e1,ind_curr)
        z = T.sum(C * att_w.dimshuffle(0, 1, 'x').repeat(C.shape[2], axis=2), axis=0)
      else:
        if self.layer.attrs['n_out'] == self.layer.attrs['attention_template']:
          dis_curr = y_p
        else:
          prev_dec_step = T.dot(y_p, self.W_att_dec) + self.b_att_dec  # BD
          dis_curr = prev_dec_step
        ind_curr = self.I_dec
        e1 = self.distance(self.C, T.tanh(dis_curr))
        att_w = self.softmax(e1, ind_curr)
        z = T.sum(self.C * att_w.dimshuffle(0, 1, 'x').repeat(self.C.shape[2], axis=2), axis=0)
    res = T.dot(z, self.W_att_in) + self.b_att_in
    inp = res
    return inp, updates

class AttentionTime(AttentionList):
  """
  Concatenate time-aligned base element into single list element
  """
  name = "attention_time"
  def make_base(self):
    self.base = [T.concatenate([b.output[::b.attrs['direction']] for b in self.layer.base], axis=2)]
    self.base[0].index = self.layer.base[0].index
    self.base[0].output = self.base[0]
    self.base[0].attrs = { 'n_out' : sum([b.attrs['n_out'] for b in self.layer.base]), 'direction' : 1 }

  def create_vars(self):
    self.make_base()
    super(AttentionTime, self).create_vars()

  def default_updates(self):
    self.make_base()
    self.glimpses = [ [] ] * len(self.base)
    self.n_glm = max(self.attrs['glimpse'],1)
    return { self.n : self.n + T.constant(1,'float32') }


class AttentionTree(AttentionList):
  """
  attention over hierarchy of bases in different time resolutions
  """
  name = "attention_tree"
  def attend(self, y_p):
    B = self.custom_vars['B_0']
    for g in range(self.n_glm):
      prev = []
      for i in range(len(self.base)-1,-1,-1):
        B, C, I, H, W_att_in, b_att_in = self.get(y_p, i, g)
        h_p = sum([h_p] + prev) / T.constant(len(self.base)-i,'float32')
        w = self.softmax(self.distance(C, h_p), I)
        prev.append(T.sum(C * w.dimshuffle(0,1,'x').repeat(C.shape[2],axis=2),axis=0))
        self.glimpses[i].append(prev[-1])
    return T.dot(T.sum(B * w.dimshuffle(0,1,'x').repeat(B.shape[2],axis=2),axis=0), self.custom_vars['W_att_in_0']), {}


class AttentionBin(AttentionList):
  """
  pruning of hypotheses in base[0] by attending over versions in time lower resolutions
  """
  name = "attention_bin"

  def attend(self, y_p):
    updates = self.default_updates()
    for g in range(self.attrs['glimpse']):
      for i in range(len(self.base)-1,-1,-1):
        factor = T.constant(self.base[i].attrs['factor'][0], 'int32') if i > 0 else 1
        B, C, I, H, W_att_in, b_att_in = self.get(y_p, i, g)
        if i == len(self.base) - 1:
          z_i = self.distance(C, H)
        else:
          length = T.cast(T.max(T.sum(I,axis=0))+1,'int32')
          ext = T.cast(T.minimum(ext/factor,T.min(length)),'int32')
          def pick(i_t, ext):
            pad = T.minimum(i_t+ext, B.shape[0]) - ext
            return T.concatenate([T.zeros((pad,), 'int8'), T.ones((ext,), 'int8'), T.zeros((B.shape[0]-pad-ext+1,), 'int8')], axis=0)
          idx, _ = theano.map(pick, sequences = [pos/factor], non_sequences = [ext])
          idx = (idx.dimshuffle(1,0)[:-1].flatten() > 0).nonzero()
          C = C.reshape((C.shape[0]*C.shape[1],C.shape[2]))[idx].reshape((ext,C.shape[1],C.shape[2]))
          z_i = self.distance(C, H)
          I = I.reshape((I.shape[0]*I.shape[1],))[idx].reshape((ext,I.shape[1]))
        if i > 0:
          pos = T.argmax(self.softmax(z_i,I),axis=0) * factor
          ext = factor
        else:
          w_i = self.softmax(z_i,I)
      B = B.reshape((B.shape[0]*B.shape[1],B.shape[2]))[idx].reshape((ext,B.shape[1],B.shape[2]))
      proto = T.sum(B * w_i.dimshuffle(0,1,'x').repeat(B.shape[2],axis=2),axis=0)
      for i in range(len(self.base)):
        self.glimpses[i].append(proto)
    return T.dot(proto, self.custom_vars['W_att_in_0']), updates


class AttentionTimeGauss(RecurrentTransformBase):
  name = "attention_time_gauss"

  def create_vars(self):
    layer = self.layer
    base = layer.base
    assert base, "attention networks are only defined for decoder networks"

    n_out = layer.attrs['n_out']
    n_in = sum([e.attrs['n_out'] for e in base])
    src = [e.output for e in base]

    if len(src) == 1:
      self.B = src[0]
    else:
      self.B = T.concatenate(src, axis=2)  # base (output of encoder). (time,batch,encoder-dim)
    self.add_input(self.B, name="B")
    self.B_index = self.layer.base[0].index  # not an input
    self.B_times = self.add_input(T.cast(T.sum(self.B_index, axis=0), dtype="float32"), "B_times")  # float32 for gpu

    self.W_att_re = self.add_param(layer.create_random_uniform_weights(n=n_out, m=2, p=n_out, name="W_att_re"))
    self.b_att_re = self.add_param(layer.create_bias(2, name='b_att_re'))
    self.W_att_in = self.add_param(layer.create_random_uniform_weights(n=n_in, m=n_out * 4, name="W_att_in"))
    self.W_state_in = self.add_param(layer.create_random_uniform_weights(n=3, m=n_out * 4, name="W_state_in"))

    self.c = self.add_state_var(T.constant(0, dtype="float32"), name="c")  # float32 for gpu
    self.t = self.add_state_var(T.zeros((self.B.shape[1],), dtype="float32"), name="t")  # (batch,)

  def step(self, y_p):
    # y_p is (batch,n_out)
    # B is (time,batch,n_in)
    # B_index is (time,batch)
    attribs = self.layer.attrs["recurrent_transform_attribs"]
    n_batch = self.B.shape[1]
    dt_min = T.constant(attribs.get("dt_min", 0.5), dtype="float32")
    dt_max = T.constant(attribs.get("dt_max", 1.5), dtype="float32")
    std_min = T.constant(attribs.get("std_min", 1), dtype="float32")
    std_max = T.constant(attribs.get("std_max", 2), dtype="float32")
    n_beam = T.constant(attribs.get("beam", 20), dtype="float32")
    B_times = self.B_times

    b = self.b_att_re.dimshuffle('x', 0)  # (batch,2)
    a = T.nnet.sigmoid(T.dot(y_p, self.W_att_re) + b)  # (batch,2)
    dt = dt_min + a[:, 0] * (dt_max - dt_min)  # (batch,)
    std = std_min + a[:, 1] * (std_max - std_min)  # (batch,)
    std_t_bc = std.dimshuffle('x', 0)  # (beam,batch)

    t = self.t  # (batch,). that's the old t, which starts at zero.
    t_bc = t.dimshuffle('x', 0)  # (beam,batch)

    t_round = T.round(t)  # (batch,). +0.5 so that a later int-cast will be like round().
    start_idxs = t_round - n_beam / numpy.float32(2)  # (batch,), beams, centered around t_int
    idxs_0 = T.arange(n_beam).dimshuffle(0, 'x')  # (beam,batch). all on cpu, but static, no round trip
    idxs = T.cast(idxs_0, dtype="float32") + start_idxs.dimshuffle('x', 0)  # (beam,batch). centered around t_int

    # gauss window
    f_e = T.exp(-(T.cast(t_bc - idxs, dtype="float32") ** 2) / (2 * std_t_bc ** 2))  # (beam,batch)
    norm = T.constant(1.0, dtype="float32") / (std_t_bc * T.constant(sqrt(2 * pi), dtype="float32"))  # (beam,batch)
    w_t = f_e * norm  # (beam,batch)
    w_t_bc = w_t.dimshuffle(0, 1, 'x')  # (beam,batch,n_in)

    B_beam = multi_batch_beam(self.B, start_idxs, B_times, n_beam, "wrap_around")
    att = T.sum(B_beam * w_t_bc, axis=0, keepdims=False)  # (batch,n_in)
    z_re = T.dot(att, self.W_att_in)  # (batch,n_out*4)

    t_frac = T.cast((self.t + 1) / (self.c.dimshuffle('x') + 1), dtype="float32")  # (batch,)
    t_frac_row = t_frac.reshape((n_batch, 1))  # (batch,1)
    state_t_frac = T.constant(1, dtype="float32").dimshuffle('x', 'x') - t_frac_row  # (batch,1)
    state = T.concatenate([state_t_frac, a], axis=1)  # (batch,3)
    z_re += T.dot(state, self.W_state_in)

    return z_re, {self.t: self.t + dt, self.c: self.c + 1}

  def cost(self):
    t_seq = self.get_state_vars_seq(self.t)  # (time,batch)
    # Get the last frame. -2 because the last update is not used.
    B_index = self.B_index
    B_times = T.sum(B_index, axis=0)
    #B_times = T.printing.Print("B_times")(B_times)
    B_last = B_times - 1  # last frame idx of the base seq
    O_index = self.layer.index
    O_times = T.sum(O_index, axis=0)
    #O_times = T.printing.Print("O_times")(O_times)
    O_last = O_times - 2  # last frame. one less because initial states are in extra vector.
    # We need an extra check for small batches, would crash otherwise.
    O_last_clipped = T.clip(O_last, 0, t_seq.shape[0] - 1)
    batches = T.arange(t_seq.shape[1])  # (batch,)
    t_last = T.switch(T.lt(O_last, 0),
                      self.state_vars_initial["t"],
                      t_seq[O_last_clipped[batches], batches])  # (batch,)
    #t_last = T.printing.Print("t_last")(t_last)
    return T.sum((t_last - B_last) ** 2)


def get_dummy_recurrent_transform(recurrent_transform_name, n_out=5, n_batches=2, n_input_t=2, n_input_dim=2):
  """
  :type recurrent_transform_name: str
  :rtype: RecurrentTransformBase
  This function is a useful helper for testing/debugging.
  """
  cls = transform_classes[recurrent_transform_name]
  from returnn.theano.layers.rec import RecurrentUnitLayer
  from returnn.theano.layers.base import SourceLayer
  if getattr(RecurrentUnitLayer, "rng", None) is None:
    RecurrentUnitLayer.initialize_rng()
  index = theano.shared(numpy.array([[1] * n_batches] * n_input_t, dtype="int8"), name="i")
  x_out = theano.shared(numpy.array([[[1.0] * n_input_dim] * n_batches] * n_input_t, dtype="float32"), name="x")
  layer = RecurrentUnitLayer(n_out=n_out, index=index, sources=[],
                             base=[SourceLayer(n_out=x_out.get_value().shape[2], x_out=x_out, index=index)],
                             recurrent_transform=recurrent_transform_name)
  assert isinstance(layer.recurrent_transform, cls)
  return layer.recurrent_transform


transform_classes = {}; ":type: dict[str,class]"
transforms_by_id = {}; ":type: dict[int,RecurrentTransformBase]"

def _setup():
  import inspect
  for clazz in globals().values():
    if not inspect.isclass(clazz): continue
    if not issubclass(clazz, RecurrentTransformBase): continue
    if clazz.name is None: continue
    transform_classes[clazz.name] = clazz

_setup()
