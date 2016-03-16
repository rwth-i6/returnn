
from math import sqrt, pi
import theano
import theano.tensor as T
import theano.sandbox.cuda as cuda
import numpy
from MultiBatchBeam import multi_batch_beam
from ActivationFunctions import elu


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

  def add_param(self, v, name = None):
    if name: v.name = name
    assert v.name
    if not self.for_custom:
      self.layer.add_param(v, v.name + "_" + self.name)
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
    return self.layer.recurrent_transform_state_var_seqs[idx]

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


class LM(RecurrentTransformBase):
  name = "lm"

  def create_vars(self):
    self.W_lm_in = self.add_var(self.layer.W_lm_in, name="W_lm_in")
    self.W_lm_out = self.add_var(self.layer.W_lm_out, name="W_lm_out")
    self.lmmask = self.add_var(self.layer.lmmask, "lmmask")
    self.t = self.add_state_var(T.zeros((1,), dtype="float32"), name="t")
    y = self.layer.y_in[self.layer.attrs['target']].flatten()
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
    if self.layer.attrs['droplm'] < 1.0:
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

  @property
  def attrs(self):
    return { "_".join(k.split("_")[1:]) : self.layer.attrs[k] for k in self.layer.attrs.keys() if k.startswith("attention_") }

  def create_vars(self):
    self.base = self.layer.base
    self.n = self.add_state_var(T.zeros((self.layer.index.shape[1],), 'float32'), 'n')
    self.bound = self.add_input(T.cast(T.sum(self.layer.index,axis=0), 'float32'), 'bound')
    if self.attrs['distance'] == 'rnn':
      n_tmp = self.attrs['template']
      l = sqrt(6.) / sqrt(2 * n_tmp)
      values = numpy.asarray(self.layer.rng.uniform(low=-l, high=l, size=(n_tmp, n_tmp)), dtype=theano.config.floatX)
      self.A_re = self.add_param(theano.shared(value=values, borrow=True, name = "A_re"))
    if self.attrs['lm'] != "none":
      self.W_lm_in = self.add_var(self.layer.W_lm_in, name="W_lm_in")
      self.W_lm_out = self.add_var(self.layer.W_lm_out, name="W_lm_out")
      self.drop_mask = self.add_var(self.layer.lmmask, "drop_mask")
      y = self.layer.y_in[self.layer.attrs['target']].flatten()
      eos = T.unbroadcast(self.W_lm_out[0].dimshuffle('x','x',0),1).repeat(self.layer.index.shape[1],axis=1)
      if self.layer.attrs['direction'] == 1:
        y_t = self.W_lm_out[y].reshape((self.layer.index.shape[0],self.layer.index.shape[1],self.layer.unit.n_in))[:-1] # (T-1)BD
        self.cls = T.concatenate([eos, y_t], axis=0)
      else:
        y_t = self.W_lm_out[y].reshape((self.layer.index.shape[0],self.layer.index.shape[1],self.layer.unit.n_in))[1:] # (T-1)BD
        self.cls = T.concatenate([eos,y_t[::-1]], axis=0)
      self.add_input(self.cls, 'cls')

  def default_updates(self):
    self.base = self.layer.base
    self.glimpses = [ [] ] * len(self.base)
    self.n_glm = max(self.attrs['glimpse'],1)
    return { self.n : self.n + T.constant(1,'float32') }

  def step(self, y_p):
    result = 0
    self.glimpses = []
    updates = self.default_updates()
    if self.attrs['lm'] != "none":
      p_re = T.nnet.softmax(T.dot(y_p, self.W_lm_in))
      if self.layer.attrs['droplm'] < 1.0:
        mask = self.drop_mask[T.cast(self.n[0],'int32')]
        if self.attrs['lm'] == "hard":
          result += self.W_lm_out[T.argmax(p_re, axis=1)] * (1. - mask) + self.cls[T.cast(self.t[0],'int32')] * mask
        else:
          result += T.dot(p_re,self.W_lm_out) * (1. - mask) + self.cls[T.cast(self.t[0],'int32')] * mask
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
    if dist == 'l2':
      dst = T.sqrt(T.sum((C - H.dimshuffle('x',0,1).repeat(C.shape[0],axis=0)) ** 2, axis=2))
      #return T.mean((C - H.dimshuffle('x',0,1).repeat(C.shape[0],axis=0)) ** 2, axis=2)
    elif dist == 'dot':
      dst = T.sum(C * H.dimshuffle('x',0,1).repeat(C.shape[0],axis=0), axis=2)
    elif dist == 'l1':
      dst = T.sum(abs(C - H.dimshuffle('x',0,1)), axis=2)
    elif dist == 'cos': # use with template size > 32
      J = H / T.sqrt(T.sum(H**2,axis=1,keepdims=True))
      K = C / T.sqrt(T.sum(C**2,axis=2,keepdims=True))
      dst = T.sum(K * J.dimshuffle('x',0,1).repeat(C.shape[0],axis=0), axis=2)
    elif dist == 'rnn':
      inp, _ = theano.scan(lambda x,p,W:elu(x+T.dot(p,W)), sequences = C, outputs_info = [H[0]], non_sequences=[self.A_re])
      dst = inp[-1]
    else:
      raise NotImplementedError()
    return dst * T.constant(self.attrs['sharpening'], 'float32') #/ T.cast(H.shape[1],'float32')

  def beam(self, X, beam_idx=None):
    if not beam_idx:
      beam_idx = X.beam_idx
    input_shape = [X.shape[0] * X.shape[1]]
    if X.ndim == 3:
      input_shape.append(X.shape[2])
    Y = X.reshape(input_shape)[beam_idx].reshape((self.attrs['beam'],X.shape[1]))
    Y.beam_idx = beam_idx
    return Y

  def softmax(self, D, I):
    D = D
    if self.attrs['norm'] == 'exp':
      E = T.exp(-D)
    elif self.attrs['norm'] == 'sigmoid':
      E = T.nnet.sigmoid(D)
    else:
      raise NotImplementedError()
    E = E * I
    if self.attrs['nbest'] > 0:
      opt = T.minimum(self.attrs['nbest'], E.shape[0])
      score = (T.sort(E, axis=0)[-opt]).dimshuffle('x',0).repeat(E.shape[0],axis=0)
      E = T.switch(T.lt(E,score), T.zeros_like(E), E)
    return E / T.maximum(T.sum(E,axis=0,keepdims=True),T.constant(1e-20,'float32'))


class AttentionList(AttentionBase):
  """
  attention over list of bases
  """
  name = "attention_list"
  def init(self, i):
    if self.attrs['beam'] > 0:
      img = 0
      for b in xrange(self.attrs['beam']):
        img += T.eye(self.custom_vars['C_%d' % i].shape[0],self.custom_vars['C_%d' % i].shape[0],b,dtype='float32')
      self.__setattr__("P_%d" % i, self.add_input(img, 'P_%d' %i))
    self.__setattr__("B_%d" % i, self.custom_vars['B_%d' % i])
    self.__setattr__("C_%d" % i, self.custom_vars['C_%d' % i])
    self.__setattr__("I_%d" % i, self.custom_vars['I_%d' % i])
    self.__setattr__("W_att_re_%d" % i, self.custom_vars['W_att_re_%d' % i])
    self.__setattr__("b_att_re_%d" % i, self.custom_vars['b_att_re_%d' % i])
    self.__setattr__("W_att_in_%d" % i, self.custom_vars['W_att_in_%d' % i])
    shape = self.layer.base[i].index.shape
    if self.attrs['store']:
      self.__setattr__("att_%d" % i, self.add_state_var(T.zeros(shape,'float32'), "att_%d" % i))
    if self.attrs['align']:
      self.__setattr__("Q_%d" % i, self.add_state_var(T.zeros(shape,'float32'), "Q_%d" % i))
      self.__setattr__("K_%d" % i, self.add_state_var(T.zeros(shape,'float32'), "B_%d" % i))

  def create_vars(self):
    super(AttentionList, self).create_vars()
    n_tmp = self.attrs['template']
    direction = self.layer.attrs['direction']
    for i,e in enumerate(self.base):
      # base output
      self.add_input(e.output[::direction], 'B_%d' % i)
      # mapping from base output to template size
      l = sqrt(6.) / sqrt(self.layer.attrs['n_out'] + n_tmp + self.layer.unit.n_re) # + self.base[i].attrs['n_out'])
      values = numpy.asarray(self.layer.rng.uniform(low=-l, high=l, size=(self.layer.attrs['n_out'], n_tmp)), dtype=theano.config.floatX)
      self.add_param(theano.shared(value=values, borrow=True, name = "W_att_re_%d" % i))
      values = numpy.zeros((n_tmp,),dtype='float32')
      self.add_param(theano.shared(value=values, borrow=True, name="b_att_re_%d" % i))
      values = numpy.asarray(self.layer.rng.uniform(low=-l, high=l, size=(e.attrs['n_out'], n_tmp)), dtype=theano.config.floatX)
      W_att_bs = self.layer.add_param(theano.shared(value=values, borrow=True, name = "W_att_bs_%d" % i))
      values = numpy.zeros((n_tmp,),dtype='float32')
      b_att_bs = self.layer.add_param(theano.shared(value=values, borrow=True, name="b_att_bs_%d" % i))
      C = T.tanh(T.dot(self.base[i].output[::direction], W_att_bs) + b_att_bs)
      c_i = T.cast(self.base[i].index.dimshuffle(0,1,'x').repeat(C.shape[2],axis=2),'float32')
      self.add_input(C - T.sum(C * c_i,axis=0) / T.sum(c_i,axis=0), 'C_%d' % i)
      self.add_input(T.cast(self.base[i].index[::direction], 'float32'), 'I_%d' % i)
      # mapping from template size to cell input
      l = sqrt(6.) / sqrt(self.layer.attrs['n_out'] + n_tmp + self.layer.unit.n_re)
      values = numpy.asarray(self.layer.rng.uniform(low=-l, high=l, size=(e.attrs['n_out'], self.layer.attrs['n_out'] * 4)), dtype=theano.config.floatX)
      self.add_param(theano.shared(value=values, borrow=True, name = "W_att_in_%d" % i))
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
    h_p = T.tanh(T.dot(y_p, W_att_re) + b_att_re) if g == 0 else self.glimpses[-1]
    return B, C, I, h_p, self.item("W_att_in", i)

  def attend(self, y_p):
    inp, updates = 0, {}
    for i in xrange(len(self.base)):
      for g in xrange(self.n_glm):
        B, C, I, H, W_att_in = self.get(y_p, i, g)
        z_i = self.distance(C, H)
        w_i = self.softmax(z_i, I)
        self.glimpses[i].append(T.sum(C * w_i.dimshuffle(0,1,'x').repeat(C.shape[2],axis=2),axis=0))
      if self.attrs['store']:
        updates[self.state_vars['att_%d'%i]] = w_i
      if self.attrs['align']:
        dst = -T.log(w_i)
        inf = T.cast(1e30,'float32')
        Q = self.item("Q", i)
        K = self.item("K", i)
        def dtw(i,q_p,b_p,Q,D):
          forward = T.constant(0.0, 'float32') + q_p # (t-1,n-1) -> (t,n)
          loop = T.constant(3.0, 'float32') + T.switch(T.gt(i,0),Q[i-1],T.zeros_like(Q[0])) # (t-1,n) -> (t,n)
          opt = T.stack([loop,forward])
          k_out = T.argmin(opt,axis=0)
          return opt[k_out] + D[i], k_out
        output, _ = theano.scan(dtw, sequences=[T.arange(dst.shape[0],'float32')], non_sequences=[Q,K,-T.log(w_i)],
                                outputs_info=[T.zeros_like(Q[0]),T.zeros_like(K[0])])
        updates[self.custom_vars['Q_%d'%i]] = output[0]
        updates[self.custom_vars['K_%d'%i]] = output[1]
      inp += T.dot(T.sum(B * w_i.dimshuffle(0,1,'x').repeat(B.shape[2],axis=2),axis=0), W_att_in)
    return inp, updates


class AttentionTime(AttentionList):
  """
  Concatenate time-aligned base element into single list element
  """
  name = "attention_time"
  def create_vars(self):
    super(AttentionTime,self).create_vars()
    self.base = [T.concatenate(b.output,axis=2) for b in self.layer.base]
    self.base[0].index = self.layer.base[0].index
    self.base[0].output = self.base[0]

  def default_updates(self):
    self.base = [T.concatenate(self.layer.base,axis=2)]
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
    for g in xrange(self.n_glm):
      prev = []
      for i in xrange(len(self.base)-1,-1,-1):
        _, C, I, h_p, _ = self.get(y_p, i, g)
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
    for g in xrange(self.attrs['glimpse']):
      for i in xrange(len(self.base)-1,-1,-1):
        factor = T.constant(self.base[i].attrs['factor'][0], 'int32') if i > 0 else 1
        B, C, I, h_p, _ = self.get(y_p, i, g)
        if i == len(self.base) - 1:
          z_i = self.distance(C, h_p)
        else:
          length = T.cast(T.max(T.sum(I,axis=0))+1,'int32')
          ext = T.cast(T.minimum(ext/factor,T.min(length)),'int32')
          def pick(i_t, ext):
            pad = T.minimum(i_t+ext, B.shape[0]) - ext
            return T.concatenate([T.zeros((pad,), 'int8'), T.ones((ext,), 'int8'), T.zeros((B.shape[0]-pad-ext+1,), 'int8')], axis=0)
          idx, _ = theano.map(pick, sequences = [pos/factor], non_sequences = [ext])
          idx = (idx.dimshuffle(1,0)[:-1].flatten() > 0).nonzero()
          C = C.reshape((C.shape[0]*C.shape[1],C.shape[2]))[idx].reshape((ext,C.shape[1],C.shape[2]))
          z_i = self.distance(C, h_p)
          I = I.reshape((I.shape[0]*I.shape[1],))[idx].reshape((ext,I.shape[1]))
        if i > 0:
          pos = T.argmax(self.softmax(z_i,I),axis=0) * factor
          ext = factor
        else:
          w_i = self.softmax(z_i,I)
      B = B.reshape((B.shape[0]*B.shape[1],B.shape[2]))[idx].reshape((ext,B.shape[1],B.shape[2]))
      proto = T.sum(B * w_i.dimshuffle(0,1,'x').repeat(B.shape[2],axis=2),axis=0)
      for i in xrange(len(self.base)):
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
  from NetworkRecurrentLayer import RecurrentUnitLayer
  from NetworkBaseLayer import SourceLayer
  if getattr(RecurrentUnitLayer, "rng", None) is None:
    RecurrentUnitLayer.initialize_rng()
  index = theano.shared(numpy.array([[1] * n_batches] * n_input_t, dtype="int8"), name="i")
  x_out = theano.shared(numpy.array([[[1.0] * n_input_dim] * n_batches] * n_input_t, dtype="float32"), name="x")
  layer = RecurrentUnitLayer(n_out=n_out, index=index, sources=[],
                             base=[SourceLayer(n_out=x_out.get_value().shape[2], x_out=x_out, index=index)],
                             attention=recurrent_transform_name)
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
