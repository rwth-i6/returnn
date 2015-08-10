from math import sqrt
import numpy
from theano import tensor as T
import theano
from Log import log

__author__ = 'az'


class Container(object):
  rng_seed = 1234

  @classmethod
  def initialize_rng(cls):
    cls.rng = numpy.random.RandomState(cls.rng_seed)

  def __init__(self, layer_class, name="", network=None,
               train_flag=False, depth=1, consensus="flat",
               forward_weights_init=None, bias_init=None,
               substitute_param_expr=None):
    """
    :param str layer_class: name of layer type, e.g. "hidden", "recurrent", "lstm" or so. see LayerClasses.
    :param str name: custom layer name, e.g. "hidden_2"
    :param Network.LayerNetwork network: the network which we will be part of
    :param str forward_weights_init: see self.create_forward_weights()
    :param str bias_init: see self.create_bias()
    """
    self.params = {}; """ :type: dict[str,theano.compile.sharedvalue.SharedVariable] """
    self.attrs = {}; """ :type: dict[str,str|float|int|bool] """
    self.layer_class = layer_class.encode("utf8")
    self.name = name.encode("utf8")
    self.train_flag = train_flag
    self.depth = depth
    self.set_attr('depth', depth)
    self.set_attr('consensus', consensus)
    self.network = network
    self.forward_weights_init = forward_weights_init or "random_normal()"
    self.bias_init = bias_init or "zeros()"
    self.substitute_param_expr = substitute_param_expr

  def dot(self, vec, mat):
    if self.depth == 1:
      return T.dot(vec, mat)
    else:
      return T.tensordot(vec, mat, 1)

  def save(self, head):
    """
    :type head: h5py.File
    """
    grp = head.create_group(self.name)
    grp.attrs['class'] = self.layer_class
    for p in self.params.keys():
      value = self.params[p].get_value()
      dset = grp.create_dataset(p, value.shape, dtype='f')
      dset[...] = value
    for p in self.attrs.keys():
      try:
        grp.attrs[p] = self.attrs[p]
      except TypeError:
        print >> log.v3, "warning: invalid type of attribute", "\"" + p + "\"", "(" + str(type(self.attrs[p])) + ")", "in layer", self.name

  def load(self, head):
    """
    :type head: h5py.File
    """
    grp = head[self.name]
    if grp.attrs['class'] != self.layer_class:
      from NetworkLayer import get_layer_class
      assert get_layer_class(grp.attrs['class']) is get_layer_class(self.layer_class), \
        "invalid layer class (expected " + self.layer_class + " got " + grp.attrs['class'] + ")"
    for p in grp:
      assert self.params[p].get_value(borrow=True, return_internal_type=True).shape == grp[p].shape, \
        "invalid layer parameter shape for parameter " + p + " of layer " + self.name + \
        " (expected  " + str(self.params[p].get_value(borrow=True, return_internal_type=True).shape) + \
        " got " + str(grp[p].shape) + ")"
      array = grp[p][...]
      assert not (numpy.isinf(array).any() or numpy.isnan(array).any())
      self.params[p].set_value(array)
    for p in self.attrs.keys():
      att = grp.attrs.get(p, None)
      if att != None:
        self.attrs[p] = att

  def num_params(self):
    return sum([numpy.prod(v.get_value(borrow=True, return_internal_type=True).shape[0:]) for v in self.params.values()])

  def get_params_dict(self):
    """
    :rtype: dict[str,numpy.ndarray|theano.sandbox.cuda.CudaNdArray]
    """
    return {p: v.get_value(borrow=True, return_internal_type=True) for (p, v) in self.params.items()}

  def set_params_by_dict(self, params):
    """
    :type params: dict[str,numpy.ndarray|theano.sandbox.cuda.CudaNdArray]
    """
    for p, v in params.items():
      self_param_shape = self.params[p].get_value(borrow=True, return_internal_type=True).shape
      assert self_param_shape == v.shape, "In %s, param %s shape does not match. Expected %s, got %s." % \
                                          (self, p, self_param_shape, v.shape)
      self.params[p].set_value(v, borrow=True)

  def get_params_vars(self):
    """
    :returns list of shared vars in a well-defined order
    """
    res = []
    for (k, v) in sorted(self.params.items()):
      v.layer = self
      res.append(v)
    return res

  def add_param(self, param, name=""):
    """
    :type param: T
    :type name: str
    :rtype: T
    """
    if not name:
      name = getattr(param, "name", None)
    if not name:
      name = "param_%d" % len(self.params)
    if self.substitute_param_expr:
      substitute = eval(self.substitute_param_expr, {"self": self, "name": name, "value": param})
      if substitute:
        return substitute
    self.params[name] = param
    return param

  def set_attr(self, name, value):
    self.attrs[name] = value

  def create_bias(self, n, prefix='b'):
    """
    :param int n: output dimension
    :rtype: theano.shared
    """
    name = "%s_%s" % (prefix, self.name)
    if self.depth > 1:
      size = (self.depth, n)
    else:
      size = (n,)
    def random_normal(scale, loc=0.0):
      return self.rng.normal(loc=loc, scale=scale, size=size)
    def random_uniform(l, loc=0.0):
      return self.rng.uniform(low=-l + loc, high=l + loc, size=size)
    eval_locals = {
      "n": n,
      "sqrt": numpy.sqrt,
      "log": numpy.log,
      "zeros": (lambda: numpy.zeros(size, dtype=theano.config.floatX)),
      "random_normal": random_normal,
      "random_uniform": random_uniform
    }
    values = eval(self.bias_init, eval_locals)
    values = numpy.asarray(values, dtype=theano.config.floatX)
    return theano.shared(value=values, borrow=True, name=name)

  def create_random_normal_weights(self, n, m, scale=None, name=None):
    if name is None: name = self.name
    if not scale:
      scale =  numpy.sqrt((n + m) / 12.)
    else:
      scale = numpy.sqrt(scale / 12.)
    if self.depth > 1:
      values = numpy.asarray(self.rng.normal(loc=0.0, scale=1.0 / scale, size=(n, self.depth, m)), dtype=theano.config.floatX)
    else:
      values = numpy.asarray(self.rng.normal(loc=0.0, scale=1.0 / scale, size=(n, m)), dtype=theano.config.floatX)
    return theano.shared(value=values, borrow=True, name=name) # broadcastable=(True, True, False))

  def create_random_uniform_weights(self, n, m, p=None, l=None, name=None, depth=None):
    if not depth: depth = self.depth
    if name is None: name = 'W_' + self.name
    assert not (p and l)
    if not p: p = n + m
    if not l: l = sqrt(6.) / sqrt(p)  # 1 / sqrt(p)
    if depth > 1:
      values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(n, depth, m)), dtype=theano.config.floatX)
    else:
      values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(n, m)), dtype=theano.config.floatX)
    return theano.shared(value=values, borrow=True, name=name) #, broadcastable=(True, True, False))

  def create_random_uniform_weights1(self, n, m, p=None, l=None, name=None):
    if name is None: name = 'W_' + self.name
    assert not (p and l)
    if not p: p = n + m
    if not l: l = sqrt(6.) / sqrt(p)  # 1 / sqrt(p)
    values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(n, m)), dtype=theano.config.floatX)
    return theano.shared(value=values, borrow=True, name=name) #, broadcastable=(True, True, False))

  def create_forward_weights(self, n, m, name=None):
    """
    :param int n: input dimension
    :param int m: output dimension
    :param str|None name: layer name
    :rtype: theano.shared
    """
    eval_locals = {
      "n": n,
      "m": m,
      "sqrt": numpy.sqrt,
      "random_normal": (lambda scale=None: self.create_random_normal_weights(n, m, scale=scale, name=name)),
      "random_uniform": (lambda l=None, p=None: self.create_random_uniform_weights(n, m, p=p, l=l, name=name))
    }
    return eval(self.forward_weights_init, eval_locals)

  @classmethod
  def guess_source_layer_name(cls, layer_name):
    # Any model created via NetworkDescription has SourceLayer with empty name as a source.
    # Guess the real source layer name from our name, if it matches the scheme, e.g. "hidden_N_fw".
    import re
    m = re.search("^.*?([0-9]+)[^0-9]*$", layer_name)
    if m:
      nr = int(m.group(1))
      if nr > 0:
        return "%s%i%s" % (layer_name[:m.start(1)], nr - 1, layer_name[m.end(1):])
    return None

  def to_json(self):
    attrs = self.attrs.copy()
    for k in attrs.keys():
      if isinstance(attrs[k], numpy.bool_):
        attrs[k] = True if attrs[k] else False
    if 'from' in attrs:
      if attrs['from'] == 'data':
        attrs.pop('from', None)
      elif attrs['from'] == '':
        guessed = self.guess_source_layer_name(self.name)
        if guessed:
          attrs['from'] = [guessed]
        else:
          attrs.pop('from', None)
      else:
        attrs['from'] = attrs['from'].split(',')
    return attrs


class SourceLayer(Container):
  def __init__(self, n_out, x_out, delay = 0, sparse = False, name=""):
    super(SourceLayer, self).__init__(layer_class='source', name=name)
    if not delay:
      self.output = x_out
    else:
      self.output = T.inc_subtensor(T.zeros_like(x_out)[delay:], x_out[:-delay])
    self.set_attr('n_out', n_out)
    self.set_attr('sparse', sparse)
    self.set_attr('delay', delay)

class Layer(Container):
  def __init__(self, sources, n_out, index, L1=0.0, L2=0.0, varreg=0.0, mask="unity", dropout=0.0, target=None, sparse = False, carry = False, **kwargs):
    """
    :param list[NetworkBaseLayer.Layer] sources: list of source layers
    :param int n_out: output dim of W_in and dim of bias
    :param float L1: l1-param-norm regularization
    :param float L2: l2-param-norm regularization
    :param str mask: "unity" or "dropout"
    :type dropout: float
    """
    super(Layer, self).__init__(**kwargs)
    self.index = index
    self.sources = sources; ":type: list[Layer]"
    self.num_sources = len(sources)
    if mask is None: mask = 'none'
    self.set_attr('mask', mask)
    self.set_attr('dropout', dropout)
    self.set_attr('sparse', sparse)
    self.set_attr('carry', carry)
    self.set_attr('n_out', n_out)
    self.set_attr('L1', L1)
    self.set_attr('L2', L2)
    self.set_attr('varreg', varreg)
    self.constraints = T.constant(0)
    if target:
      self.set_attr('target', target)
    self.b = self.add_param(self.create_bias(n_out), 'b_%s'%self.name)
    self.mass = T.constant(1., name = "mass_%s" % self.name, dtype='float32')
    self.masks = [None] * len(self.sources)
    assert mask in ['dropout', 'unity', 'none'], "invalid mask: %s" % mask
    if mask == "dropout" or (mask == 'none' and dropout > 0):
      assert 0.0 < dropout < 1.0
      # If we apply this mass during training then we don't need any mask or mass for testing.
      # The expected weight should be 1 in
      #   E[x] = mass * (1-dropout)
      # so mass has to be 1 / (1 - dropout).
      self.mass = T.constant(1.0 / (1.0 - dropout), dtype='float32')
      srng = theano.tensor.shared_randomstreams.RandomStreams(self.rng.randint(1234))
      if self.depth > 1:
        self.masks = [T.cast(srng.binomial(n=1, p=1 - dropout, size=(s.attrs['n_out'],self.depth)), theano.config.floatX) for s in self.sources]
      else:
        self.masks = [T.cast(srng.binomial(n=1, p=1 - dropout, size=(s.attrs['n_out'],)), theano.config.floatX) for s in self.sources]
      #this actually looked like dropconnect applied to the recurrent part, but I want to try dropout for the inputs
      #self.mask = T.cast(srng.binomial(n=1, p=1-dropout, size=(self.attrs['n_out'], self.attrs['n_out'])), theano.config.floatX)

  def concat_units(self, other, axis = 1):
    assert other.layer_class == self.layer_class, "unable to concatenate %s (%s) to %s (%s)" % (other.name, other.layer_class, self.name, self.layer_class)
    for p in other.params.keys():
      if p != 'b':
        self.params[p].set_value(numpy.concatenate((self.params[p].get_value(), other.params[p].get_value()), axis = min(len(self.params[p].get_value().shape) - 1, axis)))
    if axis == 1: self.set_attr('n_out', self.attrs['n_out'] + other.arrs['n_out'])

  def output_index(self):
    return self.index

  def add_param(self, param, name="", constraints=True):
    """
    :type param: T
    :type name: str
    :rtype: T
    """
    param = super(Layer, self).add_param(param, name)
    if constraints:
      if 'L1' in self.attrs and self.attrs['L1'] > 0:
        self.constraints += T.constant(self.attrs['L1'], name="L1", dtype='floatX') * abs(param).sum()
      if 'L2' in self.attrs and self.attrs['L2'] > 0:
        self.constraints += T.constant(self.attrs['L2'], name="L2", dtype='floatX') * (param**2).sum()
      if 'varreg' in self.attrs and self.attrs['varreg'] > 0:
        self.constraints += self.attrs['varreg'] * (1.0 * T.sqrt(T.var(param)) - 1.0 / numpy.sum(param.get_value().shape))**2
    return param

  def get_branching(self):
    return sum([W.get_value().shape[0] for W in self.W_in]) + 1

  def get_energy(self):
    energy =  self.b / self.attrs['n_out']
    for W in self.W_in:
      energy += T.sum(W, axis = 0)
    return energy

  def make_constraints(self):
    return self.constraints

  def make_consensus(self, networks, axis=2):
    cns = self.attrs['consensus']
    if cns == 'max':
      return T.max(networks, axis=axis)
    elif cns == 'min':
      return T.min(networks, axis=axis)
    elif cns == 'mean':
      return T.mean(networks, axis=axis)
    elif cns == 'flat':
      if self.depth == 1:
        return networks
      if axis == 2:
        return networks.flatten(ndim=3)
        #return T.reshape(networks, (networks.shape[0], networks.shape[1], T.prod(networks.shape[2:]) ))
      else:
        return networks.flatten(ndim=2) # T.reshape(networks, (networks.shape[0], T.prod(networks.shape[1:]) ))
    elif cns == 'sum':
      return T.sum(networks, axis=axis, acc_dtype=theano.config.floatX)
    elif cns == 'prod':
      return T.prod(networks, axis=axis)
    elif cns == 'var':
      return T.var(networks, axis=axis)
    elif cns == 'project':
      p = self.add_param(self.create_random_uniform_weights(self.attrs['n_out'], 1, self.attrs['n_out'] + self.depth + 1))
      return T.tensordot(p, networks, [[1], [axis]])
    elif cns == 'random':
      idx = self.rng.random_integers(size=(1,), low=0, high=self.depth)
      if axis == 0: return networks[idx]
      if axis == 1: return networks[:,idx]
      if axis == 2: return networks[:,:,idx]
      if axis == 3: return networks[:,:,:,idx]
      assert False, "axis too large"
    else:
      assert False, "consensus method unknown: " + cns

  def make_output(self, output, collapse = True):
    self.output = output
    if collapse and self.depth > 1:
      self.output = self.make_consensus(self.output)
      if self.attrs['consensus'] == 'flat':
        self.attrs['n_out'] *= self.depth
    if self.attrs['carry']:
      assert sum([s.attrs['n_out'] for s in self.sources]) == self.attrs['n_out'], "input / output dimensions do not match in %s. input %d, output %d" % (self.name, sum([s.attrs['n_out'] for s in self.sources]), self.attrs['n_out'])
      name = 'W_T_%s'%self.name
      if not name in self.params:
        self.add_param(self.create_random_uniform_weights(self.attrs['n_out'], self.attrs['n_out'], name=name), name=name)
      W = self.params[name]
      name = "b_T_%s"%self.name
      if not name in self.params:
        self.add_param(theano.shared(value=numpy.asarray(self.rng.uniform(low=-3, high=-1, size=(self.attrs['n_out'],)), dtype=theano.config.floatX), borrow=True, name=name), name=name)
      b = self.params[name]
      x = T.concatenate([s.output for s in self.sources], axis = -1)
      Tr = T.nnet.sigmoid(self.dot(x, W) + b)
      self.output = Tr * self.output + (1 - Tr) * x
    if self.attrs['sparse']:
      self.output = T.argmax(self.output, axis=-1, keepdims=True)

  def to_json(self):
    attrs = super(Layer, self).to_json()
    attrs['class'] = self.layer_class
    return attrs
