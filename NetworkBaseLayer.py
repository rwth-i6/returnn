
from math import sqrt
import numpy
import theano
from theano import tensor as T
from Log import log


class Container(object):
  rng_seed = 1234

  @classmethod
  def initialize_rng(cls):
    cls.rng = numpy.random.RandomState(cls.rng_seed)

  def __init__(self, layer_class, name="", network=None,
               forward_weights_init=None, bias_init=None, depth=1,
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
    self.depth = depth
    self.set_attr('depth', depth)
    self.network = network
    self.forward_weights_init = forward_weights_init or "random_normal()"
    self.bias_init = bias_init or "zeros()"
    self.substitute_param_expr = substitute_param_expr

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
        print >> log.v3, "invalid type of attribute", "\"" + p + "\"", "(" + str(type(self.attrs[p])) + ")", "in layer", self.name

  def load(self, head):
    """
    :type head: h5py.File
    """
    grp = head[self.name]
    if grp.attrs['class'] != self.layer_class:  # First check the name. Also covers softmax (output layer).
      # This check is needed e.g. for "forward" != "hidden".
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
      self.attrs[p] = grp.attrs.get(p, None)

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
    return [v for (k, v) in sorted(self.params.items())]

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
    def random_normal(scale, loc=0.0):
      return self.rng.normal(loc=loc, scale=scale, size=(n,))
    def random_uniform(l, loc=0.0):
      return self.rng.uniform(low=-l + loc, high=l + loc, size=(n,))
    eval_locals = {
      "n": n,
      "sqrt": numpy.sqrt,
      "log": numpy.log,
      "zeros": (lambda: numpy.zeros((n,), dtype=theano.config.floatX)),
      "random_normal": random_normal,
      "random_uniform": random_uniform
    }
    values = eval(self.bias_init, eval_locals)
    values = numpy.asarray(values, dtype=theano.config.floatX)
    assert values.shape == (n,self.depth)
    return theano.shared(value=values, borrow=True, name=name)

  def create_random_normal_weights(self, n, m, scale=None, name=None):
    if name is None: name = self.name
    if not scale: scale = numpy.sqrt(12. / (n + m))
    values = numpy.asarray(self.rng.normal(loc=0.0, scale=scale, size=(n, m, self.depth)), dtype=theano.config.floatX)
    return theano.shared(value=values, borrow=True, name=name)

  def create_random_uniform_weights(self, n, m, p=None, l=None, name=None):
    if name is None: name = 'W_' + self.name
    assert not (p and l)
    if not p: p = n + m
    if not l: l = sqrt(6) / sqrt(p)  # 1 / sqrt(p)
    values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(n, m, self.depth)), dtype=theano.config.floatX)
    return theano.shared(value=values, borrow=True, name=name)

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

  def to_json(self):
    attrs = self.attrs.copy()
    for k in attrs.keys():
      if isinstance(attrs[k], numpy.bool_):
        attrs[k] = True if attrs[k] else False
    if 'from' in attrs:
      if attrs['from'] == 'data':
        attrs.pop('from', None)
      else:
        attrs['from'] = attrs['from'].split(',')
    return attrs


class Layer(Container):
  def __init__(self, sources, n_out, L1=0.0, L2=0.0, mask="unity", dropout=0.0, **kwargs):
    """
    :param list[SourceLayer] sources: list of source layers
    :param int n_out: output dim of W_in and dim of bias
    :param float L1: l1-param-norm regularization
    :param float L2: l2-param-norm regularization
    :param str mask: "unity" or "dropout"
    :type dropout: float
    """
    super(Layer, self).__init__(**kwargs)
    self.sources = sources; ":type: list[SourceLayer]"
    self.num_sources = len(sources)
    self.set_attr('mask', mask)
    self.set_attr('dropout', dropout)
    self.set_attr('n_out', n_out)
    self.set_attr('L1', L1)
    self.set_attr('L2', L2)
    self.b = self.add_param(self.create_bias(n_out), 'b_%s'%self.name)
    self.mass = T.constant(1., name = "mass_%s" % self.name)
    if mask == "unity" or dropout == 0:
      self.masks = [None] * len(self.sources)
    elif mask == "dropout":
      assert 0.0 < dropout < 1.0
      # If we apply this mass during training then we don't need any mask or mass for testing.
      # The expected weight should be 1 in
      #   E[x] = mass * (1-dropout)
      # so mass has to be 1 / (1 - dropout).
      self.mass = T.constant(1.0 / (1.0 - dropout))
      srng = theano.tensor.shared_randomstreams.RandomStreams(self.rng.randint(1234))
      self.masks = [T.cast(srng.binomial(n=1, p=1 - dropout, size=(s.attrs['n_out'],)), theano.config.floatX) for s in self.sources]

      #this actually looked like dropconnect applied to the recurrent part, but I want to try dropout for the inputs
      #self.mask = T.cast(srng.binomial(n=1, p=1-dropout, size=(self.attrs['n_out'], self.attrs['n_out'])), theano.config.floatX)
    else:
      assert False, "invalid mask: %s" % mask

  def concat_units(self, other, axis = 1):
    assert other.layer_class == self.layer_class, "unable to concatenate %s (%s) to %s (%s)" % (other.name, other.layer_class, self.name, self.layer_class)
    for p in other.params.keys():
      if p != 'b':
        self.params[p].set_value(numpy.concatenate((self.params[p].get_value(), other.params[p].get_value()), axis = min(len(self.params[p].get_value().shape) - 1, axis)))
    if axis == 1: self.set_attr('n_out', self.attrs['n_out'] + other.arrs['n_out'])

  def to_json(self):
    attrs = super(Layer, self).to_json()
    attrs['class'] = self.layer_class
    return attrs

  def regularization_param_list(self):
    return [self.b]

  def param_regularization_objective(self):
    o = T.constant(0)
    l1 = self.attrs['L1']
    l1c = T.constant(l1, dtype=theano.config.floatX)
    l2 = self.attrs['L2']
    l2c = T.constant(l2, dtype=theano.config.floatX)
    for param in self.regularization_param_list():
      if l1 > 0.0:
        o += l1c * abs(param).sum()
      if l2 > 0.0:
        o += l2c * (param ** 2).sum()
    return o

  def make_output(self, output):
    output = T.max(output, axis=-1, keepdims=False)
    if self.attrs['sparse']:
      output = T.argmax(output, axis=-2, keepdims=True)
    self.output = output


class SourceLayer(Container):
  def __init__(self, n_out, x_out, name=""):
    super(SourceLayer, self).__init__(layer_class='source', name=name)
    self.output = x_out
    self.set_attr('n_out', n_out)
