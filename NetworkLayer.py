
from math import sqrt
import numpy
import theano
from theano import tensor as T
from Log import log


class Container(object):
  @classmethod
  def initialize_rng(cls):
    cls.rng = numpy.random.RandomState(1234)

  def __init__(self, layer_class, name=""):
    """
    :param str layer_class: name of layer type, e.g. "hidden"
    :param str name: custom layer name, e.g. "hidden_2"
    """
    self.params = {}; """ :type: dict[str,theano.compile.sharedvalue.SharedVariable] """
    self.attrs = {}; """ :type: dict[str,str|float|int|bool] """
    self.layer_class = layer_class.encode("utf8")
    self.name = name.encode("utf8")

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
    assert grp.attrs['class'] == self.layer_class, "invalid layer class (expected " + self.layer_class + " got " + grp.attrs['class'] + ")"
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
    if name == "": name = "param_%d" % len(self.params)
    self.params[name] = param
    return param

  def set_attr(self, name, value):
    self.attrs[name] = value

  def create_bias(self, n, prefix = 'b'):
    name = "%s_%s" % (prefix, self.name)
    return theano.shared(value=numpy.zeros((n,), dtype=theano.config.floatX), borrow=True, name=name)

  def create_random_normal_weights(self, n, m, s, name=None):
    if name is None: name = self.name
    values = numpy.asarray(self.rng.normal(loc=0.0, scale=s, size=(n, m)), dtype=theano.config.floatX)
    return theano.shared(value=values, borrow=True, name=name)

  def create_random_uniform_weights(self, n, m, p=0, l=0, name=None):
    if name is None: name = 'W_' + self.name
    assert not (p and l)
    if not p: p = n + m
    if not l: l = sqrt(6) / sqrt(p)  # 1 / sqrt(p)
    values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(n, m)), dtype=theano.config.floatX)
    return theano.shared(value=values, borrow=True, name=name)

  def create_forward_weights(self, n, m, name = None):
    #n_in = n + m
    #scale = numpy.sqrt(12. / (n_in))
    #return self.create_random_normal_weights(n, m, scale, name)
    return self.create_random_uniform_weights(n, m, name=name)

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
  def __init__(self, sources, n_out, L1, L2, layer_class, mask="unity", dropout=0.0, name=""):
    """
    :param list[SourceLayer] sources: list of source layers
    :param int n_out: output dim
    :param float L1: l1-param-norm regularization
    :param float L2: l2-param-norm regularization
    :param str layer_class: name of layer type, e.g. "hidden"
    :param str mask: "unity" or "dropout"
    :type dropout: float
    :param str name: custom layer name, e.g. "hidden_2"
    """
    super(Layer, self).__init__(layer_class, name=name)
    self.sources = sources
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


class SourceLayer(Container):
  def __init__(self, n_out, x_out, name = ""):
    super(SourceLayer, self).__init__('source',  name = name)
    self.output = x_out
    self.set_attr('n_out', n_out)
