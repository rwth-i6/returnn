
from __future__ import print_function

from math import sqrt
import numpy
from theano import tensor as T
import theano
from returnn.log import log
from returnn.theano.util import time_batch_make_flat, tiled_eye
from returnn.util.basic import as_str
import json


class Container(object):
  rng_seed = 1234
  layer_class = None

  @classmethod
  def initialize_rng(cls):
    cls.rng = numpy.random.RandomState(cls.rng_seed)

  def __init__(self, layer_class=None, name="", network=None,
               train_flag=False, eval_flag=False, depth=1, consensus="flat",
               forward_weights_init=None, bias_init=None, weight_clip=0.0, cost=None,
               recurrent_weights_init=None,
               substitute_param_expr=None):
    """
    :param str layer_class: name of layer type, e.g. "hidden", "recurrent", "lstm" or so. see LayerClasses.
    :param str name: custom layer name, e.g. "hidden_2"
    :param Network.LayerNetwork network: the network which we will be part of
    :param str forward_weights_init: see self.create_forward_weights()
    :param str bias_init: see self.create_bias()
    """
    self.params = {}; """ :type: dict[str,theano.compile.sharedvalue.SharedVariable] """
    self.attrs = {}; """ :type: dict[str,str|float|int|bool|dict] """
    self.device = None
    if layer_class:
      self.layer_class = as_str(layer_class.encode("utf8"))
    self.name = as_str(name.encode("utf8"))
    self.train_flag = train_flag
    self.eval_flag = eval_flag
    self.depth = depth
    if depth != 1:
      self.set_attr('depth', depth)
    if consensus != "flat":
      self.set_attr('consensus', consensus)
    self.network = network
    if forward_weights_init:
      self.set_attr("forward_weights_init", forward_weights_init)
    self.forward_weights_init = forward_weights_init or "random_normal()"
    if recurrent_weights_init:
      self.set_attr("recurrent_weights_init", recurrent_weights_init)
    self.recurrent_weights_init = recurrent_weights_init or "random_uniform()"
    if bias_init:
      self.set_attr("bias_init", bias_init)
    self.bias_init = bias_init or "zeros()"
    if substitute_param_expr:
      self.set_attr("substitute_param_expr", substitute_param_expr)
    self.substitute_param_expr = substitute_param_expr
    if weight_clip:
      self.set_attr('weight_clip', weight_clip)
    if cost:
      self.set_attr('cost', cost)

  def __repr__(self):
    return "<%s class:%s name:%s>" % (self.__class__, self.layer_class, self.name)

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
    for p, v in self.attrs.items():
      if isinstance(v, (dict, list, tuple)):
        v = json.dumps(v, sort_keys=True)
      try:
        grp.attrs[p] = v
      except TypeError:
        print("warning: invalid type of attribute %r (%s) in layer %s" % (p, type(v), self.name), file=log.v3)

  def load(self, head):
    """
    :type head: h5py.File
    """
    try:
      grp = head[self.name]
    except Exception:
      print("warning: unable to load parameters for layer", self.name, file=log.v3)
      return

    grp_class = as_str(grp.attrs['class'])
    if grp_class == "<unknown_softmax>":
      grp_class = "softmax"  # bug in some RETURNN version. can be ignored.
    if grp_class != self.layer_class:
      from .basic import get_layer_class
      if not get_layer_class(grp_class, raise_exception=False) is get_layer_class(self.layer_class):
        print("warning: invalid layer class (expected " + self.layer_class + " got " + grp.attrs['class'] + ")", file=log.v3)
    for p in self.params:
      if p not in grp:
        print("unable to load parameter %s in %s" % (p, self.name), file=log.v4)
    for p in grp:
      if p in self.params:
        if self.params[p].get_value(borrow=True, return_internal_type=True).shape == grp[p].shape:
          array = grp[p][...]
          assert not (numpy.isinf(array).any() or numpy.isnan(array).any())
          self.params[p].set_value(array)
        else:
          print("warning: invalid layer parameter shape for parameter " + p + " of layer " + self.name + \
            " (expected  " + str(self.params[p].get_value(borrow=True, return_internal_type=True).shape) + \
            " got " + str(grp[p].shape) + ")", file=log.v2)
          #assert self.params[p].get_value(borrow=True, return_internal_type=True).shape == grp[p].shape, \
          #  "invalid layer parameter shape for parameter " + p + " of layer " + self.name + \
          #  " (expected  " + str(self.params[p].get_value(borrow=True, return_internal_type=True).shape) + \
          #  " got " + str(grp[p].shape) + ")"
      else:
        print("unable to match parameter %s in %s" % (p, self.name), file=log.v4)
    #for p in self.attrs.keys():
    #  att = grp.attrs.get(p, None)
    #  if att != None:
    #    self.attrs[p] = att

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
    :type param: theano.SharedVariable
    :type name: str
    :rtype: theano.SharedVariable
    """
    if not name:
      name = getattr(param, "name", None)
    if not name:
      name = "param_%d" % len(self.params)
    if self.network:
      substitute = self.network.get_layer_param(layer_name=self.name, param_name=name, param=param)
      if substitute:
        return substitute
    if self.substitute_param_expr:
      substitute = eval(self.substitute_param_expr, {"self": self, "name": name, "value": param})
      if substitute:
        return substitute
    self.params[name] = param
    return param

  def set_attr(self, name, value):
    """
    :param str name: key name
    :param bool|int|float|str|list|dict value: value
    This will be stored in to_json() and save() (in HDF).
    More complex types like list or dict will be encoded as a JSON-str when saved to HDF.
    """
    self.attrs[name] = value

  def shared(self, value, name, borrow=True):
    if self.device is None:
      return theano.shared(value=value, borrow=borrow, name=name)
    return theano.shared(value=value, borrow=borrow, name=name, target=self.device)

  def create_bias(self, n, prefix='b', name="", init_eval_str=None):
    """
    :param int n: output dimension
    :rtype: theano.shared
    """
    if not name:
      name = "%s_%s" % (prefix, self.name)
      if name in self.params:
        name += "_%i" % len(self.params)
    if not init_eval_str:
      init_eval_str = self.bias_init
    if self.depth > 1:
      size = (self.depth, n)
    else:
      size = (n,)
    def random_normal(scale, loc=0.0):
      return self.rng.normal(loc=loc, scale=scale, size=size)
    def random_uniform(l, loc=0.0):
      return self.rng.uniform(low=-l + loc, high=l + loc, size=size)
    import returnn.config as config_mod
    import returnn.util.basic as util
    try:
      config = config_mod.get_global_config()
    except Exception:
      config = None
    else:
      config = util.DictAsObj(config.typed_dict)
    eval_locals = {
      "numpy": numpy,
      "rng": self.rng,
      "config": config,
      "self": self,
      "n": n,
      "name": name,
      "sqrt": numpy.sqrt,
      "log": numpy.log,
      "zeros": (lambda: numpy.zeros(size, dtype=theano.config.floatX)),
      "random_normal": random_normal,
      "random_uniform": random_uniform
    }
    values = eval(init_eval_str, eval_locals)
    values = numpy.asarray(values, dtype=theano.config.floatX)
    assert values.shape == (n,)
    return self.shared(values, name)

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
    return self.shared(values, name)

  def create_random_uniform_weights(self, n, m, p=None, p_add=None, l=None, name=None, depth=None):
    if not depth: depth = self.depth
    if name is None: name = 'W_' + self.name
    assert not (p and l)
    if not p: p = n + m
    if p_add: p += p_add
    if not l: l = sqrt(6.) / sqrt(p)  # 1 / sqrt(p)
    if depth > 1:
      values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(n, depth, m)), dtype=theano.config.floatX)
    else:
      values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(n, m)), dtype=theano.config.floatX)
    return self.shared(values, name)

  def create_random_uniform_weights1(self, n, m, p=None, l=None, name=None):
    if name is None: name = 'W_' + self.name
    assert not (p and l)
    if not p: p = n + m
    if not l: l = sqrt(6.) / sqrt(p)  # 1 / sqrt(p)
    values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(n, m)), dtype=theano.config.floatX)
    return self.shared(values, name)

  def create_random_uniform_weights2(self, n, m=None, name=None):
    if name is None: name = 'W_' + self.name
    l = sqrt(1. / n)
    shape = [n]
    if m: shape += [m]
    values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=shape), dtype=theano.config.floatX)
    return self.shared(values, name)

  def create_random_unitary_weights(self, n, m, name=None):
    x = self.rng.randn(n, m)
    u, s, v = numpy.linalg.svd(x, full_matrices=0)
    if u.shape == (n, m):
      x = u
    else:
      x = v
    assert x.shape == (n, m)
    x = x.astype(theano.config.floatX)
    return self.shared(x, name)

  def create_random_unitary_tiled_weights(self, n, m, name=None):
    if n > m:
      transpose = True
      n, m = m, n  # n < m
    else:  # n <= m
      transpose = False
    fac = ((m - 1) // n) + 1
    def make_tile():
      x = self.rng.randn(n, n)
      u, s, v = numpy.linalg.svd(x)
      assert u.shape == (n, n)
      return u
    x = numpy.concatenate([make_tile() for i in range(fac)], axis=1)
    assert x.shape == (n, fac * n)
    x = x[:, :m]
    assert x.shape == (n, m)
    if transpose:
      x = x.T
    x = x.astype(theano.config.floatX)
    return self.shared(x, name)

  def _create_eval_weights(self, n, m, name, default_name_prefix, init_eval_str):
    """
    :param int n: input dimension
    :param int m: output dimension
    :param str|None name: layer name
    :rtype: theano.shared
    """
    if not name: name = "%s_%s_%i" % (default_name_prefix, self.name, len(self.params))
    from returnn.config import get_global_config
    import returnn.util.basic as util
    try:
      config = get_global_config()
    except Exception:
      config = None
    else:
      config = util.DictAsObj(config.typed_dict)
    eval_locals = {
      "numpy": numpy,
      "theano": theano,
      "rng": self.rng,
      "config": config,
      "self": self,
      "n": n,
      "m": m,
      "name": name,
      "sqrt": numpy.sqrt,
      "eye": (lambda N=n, M=m: numpy.eye(N, M, dtype=theano.config.floatX)),
      "random_normal": (
      lambda scale=None, **kwargs: self.create_random_normal_weights(n, m, scale=scale, name=name, **kwargs)),
      "random_uniform": (
      lambda l=None, p=None, **kwargs: self.create_random_uniform_weights(n, m, p=p, l=l, name=name, **kwargs)),
      "random_unitary": (lambda **kwargs: self.create_random_unitary_weights(n, m, name=name, **kwargs)),
      "random_unitary_tiled": (lambda **kwargs: self.create_random_unitary_tiled_weights(n, m, name=name, **kwargs))
    }
    v = eval(init_eval_str, eval_locals)
    if isinstance(v, numpy.ndarray):
      v = numpy.asarray(v, dtype=theano.config.floatX)
      v = self.shared(v, name)
    assert isinstance(v, theano.compile.SharedVariable)
    assert v.ndim == 2
    vshape = v.get_value(borrow=True, return_internal_type=True).shape
    assert vshape == (n, m)
    return v

  def _create_eval_params(self, shape, name, init_eval_str=None):
    assert 1 <= len(shape) <= 2
    if len(shape) == 1:
      return self.create_bias(n=shape[0], init_eval_str=init_eval_str, name=name)
    else:
      return self._create_eval_weights(
        n=shape[0], m=shape[1],
        init_eval_str=init_eval_str or "random_uniform()",
        default_name_prefix="W", name=name)

  def create_forward_weights(self, n, m, name=None):
    """
    :param int n: input dimension
    :param int m: output dimension
    :param str|None name: layer name
    :rtype: theano.shared
    """
    return self._create_eval_weights(n=n, m=m, name=name, default_name_prefix="W", init_eval_str=self.forward_weights_init)

  def create_recurrent_weights(self, n, m, name=None):
    """
    :param int n: input dimension
    :param int m: output dimension
    :param str|None name: layer name
    :rtype: theano.shared
    """
    return self._create_eval_weights(n=n, m=m, name=name, default_name_prefix="W_re", init_eval_str=self.recurrent_weights_init)

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
      if isinstance(attrs[k], bytes):
        attrs[k] = attrs[k].decode("utf8")
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
  layer_class = "source"
  recurrent = False

  def __init__(self, n_out, x_out=None, delay=0, sparse=False, name="", network=None, eval_flag=False,
               data_key=None,  # if we don't want to use "data" but something else. via y_in
               # These will be given if we initialize via JSON.
               sources=None, dropout=0, train_flag=None, mask=None, index=None, y_in=None, dtype=None):
    super(SourceLayer, self).__init__(layer_class=self.layer_class, name=name)
    if data_key is not None:
      assert x_out is None
      assert network
      assert dtype
      network.use_target(target=data_key, dtype=dtype)
      x_out = network.y[data_key]
      n_out = network.n_out[data_key][0]
      index = network.j[data_key]
    if x_out is None:
      assert network is not None
      x_out = network.x
    assert not sources, 'specify `"from": "null"` in json'  # or just ignore?
    assert dropout == 0
    if getattr(x_out.tag, "test_value", None) is None:
      if not sparse:
        x_out.tag.test_value = numpy.random.rand(3,2,n_out).astype('float32')
    if index and getattr(index.tag, "test_value", None) is None:
      index.tag.test_value = numpy.ones((3,2), dtype='int8')
    if not delay:
      self.output = x_out
    else:
      self.output = T.inc_subtensor(T.zeros_like(x_out)[delay:], x_out[:-delay])
    self.set_attr('n_out', n_out)
    self.set_attr('sparse', sparse)
    self.set_attr('delay', delay)
    self.index = index
    self.device = 'cpu'
    self.eval_flag = eval_flag

  def make_constraints(self):
    return 0

  def cost(self):
    return None, None

  def errors(self):
    """
    :rtype: theano.Variable
    """
    return None

  def transfer_output(self, device):
    pass


class Layer(Container):
  recurrent = False

  def __init__(self, sources, n_out, index, y_in=None, target=None, target_index=None,
               sparse=False, cost_scale=1.0, input_scale=1.0,
               L1=0.0, L2=0.0, L2_eye=None, varreg=0.0,
               output_L2_reg=0.0, output_entropy_reg=0.0, output_entropy_exp_reg=0.0,
               with_bias=True,
               mask="unity", dropout=0.0, batch_drop=False, batch_norm=False, bn_use_sample=False, layer_drop=0.0, residual=False,
               carry=False,
               sparse_filtering=False, gradient_scale=1.0, trainable=True, device=None,
               dtype='float32',
               **kwargs):
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
    self.D = max([s.D for s in sources if isinstance(s,Layer)] + [0])
    if mask is None: mask = 'none'
    self.set_attr('mask', mask)
    self.set_attr('dropout', dropout)
    self.set_attr('sparse', sparse)
    self.set_attr('bn_use_sample', bn_use_sample)
    self.set_attr('sparse_filtering', sparse_filtering)
    if not trainable:
      self.set_attr('trainable', trainable)  # only store if not default
      self.gradient_scale = 0.0  # just to be sure
    else:
      self.gradient_scale = gradient_scale
    if gradient_scale != 1.0:
      self.set_attr('gradient_scale', gradient_scale)
    self.set_attr('layer_drop', layer_drop)
    assert not carry, "not supported anymore"
    self.set_attr('residual', residual)
    self.set_attr('n_out', n_out)
    self.set_attr('L1', L1)
    self.set_attr('L2', L2)
    if L2_eye:
      self.set_attr('L2_eye', L2_eye)
    self.device = device # if device else str(theano.config.device)
    for s in self.sources:
      s.transfer_output(self.device)
    self.set_attr('varreg', varreg)
    if output_L2_reg:
      self.set_attr('output_L2_reg', output_L2_reg)
    if output_entropy_reg:
      self.set_attr('output_entropy_reg', output_entropy_reg)
    if output_entropy_exp_reg:
      self.set_attr('output_entropy_exp_reg', output_entropy_exp_reg)
    self.set_attr('batch_norm', batch_norm)
    self.set_attr('input_scale', input_scale)
    if y_in is not None:
      self.y_in = {}
      for k in y_in:
        if not isinstance(y_in[k], T.Variable): continue
        self.y_in[k] = time_batch_make_flat(y_in[k])  # TODO: better not flatten here...
        self.y_in[k].n_out = getattr(y_in[k], "n_out", None)
    else:
      self.y_in = None
    self.constraints = T.constant(0)
    if target:
      self.set_attr('target', target)
    if target_index:
      self.set_attr('target_index', target_index)
      assert target_index in self.network.j
      self.index = index = self.network.j[target_index]
    if cost_scale != 1:
      self.set_attr("cost_scale", cost_scale)
    if with_bias:
      self.b = self.add_param(self.create_bias(n_out), 'b_%s'%self.name)
    else:
      self.set_attr('with_bias', False)
      self.b = numpy.float32(0)
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
      from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
      srng = RandomStreams(self.rng.randint(1234) + 1)
      if self.depth > 1:
        self.masks = [T.cast(srng.binomial(n=1, p=1 - dropout, size=(s.attrs['n_out'],self.depth)), theano.config.floatX) for s in self.sources]
      else:
        if batch_drop:
          self.masks = [T.cast(srng.binomial(n=1, p=1 - dropout, size=s.output.shape), theano.config.floatX) for s in self.sources]
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
    from theano.ifelse import ifelse
    index = self.index
    if self.sources:
      # In some cases, e.g. forwarding, the target index (for "classes") might have shape[0]==0.
      # Or shape[0]==1 with index[0]==0. See Dataset.shapes_for_batches().
      # Use source index in that case.
      have_zero = ifelse(T.lt(index.shape[0], 1), 1, T.cast(T.le(index.shape[0], 1) * T.eq(T.sum(index[0]), 0), 'int8'))
      index = ifelse(have_zero, self.sources[0].index, index)
    return index

  def find_data_layer(self):
    for l in self.sources:
      if isinstance(l, SourceLayer):
        return l
      if isinstance(l, Layer):
        s = l.find_data_layer()
        if s is not None:
          return s
    return None

  def add_param(self, param, name="", constraints=True,
                custom_update=None, custom_update_normalized=False, custom_update_exp_average=0,
                custom_update_condition=None, custom_update_accumulate_batches=None, live_update=None):
    """
    :type param: theano.SharedVariable
    :type name: str
    :rtype: theano.SharedVariable
    """
    param = super(Layer, self).add_param(param, name)
    param.live_update = live_update
    if custom_update:
      # Handled in Device and Updater.
      param.custom_update = custom_update
      param.custom_update_normalized = custom_update_normalized
      param.custom_update_exp_average = custom_update_exp_average
      param.custom_update_condition = custom_update_condition
      param.custom_update_accumulate_batches = custom_update_accumulate_batches
    if constraints:
      if 'L1' in self.attrs and self.attrs['L1'] > 0:
        self.constraints += T.constant(self.attrs['L1'], name="L1", dtype='floatX') * abs(param).sum()
      if 'L2' in self.attrs and self.attrs['L2'] > 0:
        self.constraints += T.constant(self.attrs['L2'], name="L2", dtype='floatX') * (param**2).sum()
      if self.attrs.get('L2_eye', 0) > 0:
        L2_eye = T.constant(self.attrs['L2_eye'], name="L2_eye", dtype='floatX')
        if param.ndim == 2:
          eye = tiled_eye(param.shape[0], param.shape[1], dtype=param.dtype)
          self.constraints += L2_eye * ((param - eye)**2).sum()
        else:  # standard L2
          self.constraints += L2_eye * (param**2).sum()
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
    c = self.constraints
    f32_index = T.cast(self.index, "float32")
    if self.attrs.get('output_L2_reg', 0.0):
      # Note: For the output layer, it might even make sense to use a negative factor. http://www.danielpovey.com/files/2016_interspeech_mmi.pdf
      assert self.output.ndim == 3
      l2 = f32_index * T.sum(T.sqr(self.output), axis=2)
      c += numpy.float32(self.attrs.get('output_L2_reg', 0.0)) * T.sum(l2)
    if self.attrs.get('output_entropy_reg', 0.0):
      assert self.output.ndim == 3
      epsilon = numpy.float32(1e-10)
      entropy = f32_index * (-T.sum(self.output * T.log(self.output + epsilon), axis=2))
      c += numpy.float32(self.attrs.get('output_entropy_reg', 0.0)) * T.sum(entropy)
    if self.attrs.get('output_entropy_exp_reg', 0.0):
      assert self.output.ndim == 3
      entropy = f32_index * (-T.sum(T.exp(self.output) * self.output, axis=2))
      c += numpy.float32(self.attrs.get('output_entropy_exp_reg', 0.0)) * T.sum(entropy)
    return c

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

  def batch_norm(self, h, dim, use_shift=True, use_std=True, use_sample=0.0, force_sample=False, index=None,
                 sample_mean=None, gamma=None, beta=None, depth_norm=False):
    x = h
    if h.ndim == 3:
      if index is None: index = self.index
      x = h.reshape((h.shape[0]*h.shape[1],h.shape[2]))[(index.flatten()>0).nonzero()]
    elif h.ndim == 4: # index is sizes here
      assert index is not None
      x = h.reshape((h.shape[0] * h.shape[1] * h.shape[2], h.shape[3]))
      #x = x[(T.gt(x,numpy.float32(0))>0).nonzero()]
    mean = T.mean(x,axis=0)
    std = T.sqrt(T.mean((x - mean)**2,axis=0))
    if sample_mean is None:
      sample_mean = self.add_param(theano.shared(numpy.zeros((dim,), 'float32'), '%s_%s_mean' % (self.name,h.name)),
                                   custom_update=mean,
                                   custom_update_normalized=True)
    self.sample_mean = sample_mean
    sample_std = T.sqrt(T.mean((x - sample_mean)**2,axis=0))
    if not self.train_flag and not force_sample:
      use_sample = 1.0
    mean = T.constant(1.-use_sample,'float32') * mean + T.constant(use_sample,'float32') * sample_mean
    std = T.constant(1.-use_sample,'float32') * std + T.constant(use_sample,'float32') * sample_std
    if h.ndim == 3:
      mean = mean.dimshuffle('x','x',0).repeat(h.shape[0],axis=0).repeat(h.shape[1],axis=1)
      std = std.dimshuffle('x', 'x', 0).repeat(h.shape[0],axis=0).repeat(h.shape[1],axis=1)
    elif h.ndim == 4:
      mean = mean.dimshuffle('x', 'x', 'x', 0).repeat(h.shape[0], axis=0).repeat(h.shape[1], axis=1).repeat(h.shape[2], axis=2)
      std = std.dimshuffle('x', 'x', 'x', 0).repeat(h.shape[0], axis=0).repeat(h.shape[1], axis=1).repeat(h.shape[2], axis=2)
    else:
      mean = mean.dimshuffle('x', 0).repeat(h.shape[0], axis=0)
      std = std.dimshuffle('x', 0).repeat(h.shape[0], axis=0)
    bn = (h - mean) / (std + numpy.float32(1e-10))
    if use_std:
      if gamma is None:
        gamma = self.add_param(self.shared(numpy.zeros((dim,), 'float32') + numpy.float32(0.1), "%s_%s_gamma" % (self.name,h.name)))
      self.gamma = gamma
      if h.ndim == 3:
        bn *= gamma.dimshuffle('x','x',0).repeat(h.shape[0],axis=0).repeat(h.shape[1],axis=1)
      elif h.ndim == 4:
        bn *= gamma.dimshuffle('x', 'x', 'x', 0).repeat(h.shape[0], axis=0).repeat(h.shape[1], axis=1).repeat(h.shape[2], axis=2)
      else:
        bn *= gamma.dimshuffle('x', 0).repeat(h.shape[0], axis=0)
    if use_shift:
      if beta is None:
        beta = self.add_param(self.shared(numpy.zeros((dim,), 'float32'), "%s_%s_beta" % (self.name,h.name)))
      self.beta = beta
      bn += beta
    if depth_norm:
      bn = bn / (T.sqrt(2)**self.D)
    return bn

  def make_output(self, output, collapse = True, sample_mean=None, gamma=None):
    self.output = output
    if collapse and self.depth > 1:
      self.output = self.make_consensus(self.output)
      if self.attrs['consensus'] == 'flat':
        self.attrs['n_out'] *= self.depth
    if self.attrs['batch_norm']:
      self.output = self.batch_norm(self.output, self.attrs['n_out'], sample_mean=sample_mean, gamma=gamma, use_sample=self.attrs['bn_use_sample'])
    if self.attrs['residual']:
      from .hidden import concat_sources
      z, n_in = concat_sources(self.sources, unsparse=True, expect_source=False)
      assert n_in == self.attrs['n_out']
      self.output += z
    if self.attrs['layer_drop'] > 0.0:
      # Stochastic Depth, http://arxiv.org/abs/1603.09382
      from .hidden import concat_sources
      z, n_in = concat_sources(self.sources, unsparse=True, expect_source=False)
      n_out = self.attrs['n_out']
      if n_in != n_out:
        print("Layer drop with additional projection %i -> %i" % (n_in, n_out), file=log.v4)
        if n_in > 0:
          self.W_drop = self.add_param(self.create_forward_weights(n_in, n_out, name="W_drop_%s" % self.name))
          z = T.dot(z, self.W_drop)
        else:
          z = 0
      if self.train_flag:
        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
        rng = RandomStreams(self.rng.randint(1234) + 1)
        import theano.ifelse
        drop = rng.binomial(n=1, p=self.attrs['layer_drop'], size=(1,), dtype='int8')[0]
        # drop = theano.printing.Print("drop")(drop)
        self.output = theano.ifelse.ifelse(drop, z, self.output)
      else:
        drop = self.attrs['layer_drop']
        self.output = numpy.float32(drop) * z + numpy.float32(1.0 - drop) * self.output
    if self.attrs['sparse']:
      self.output = T.argmax(self.output, axis=-1, keepdims=True)
    if self.attrs['sparse_filtering']:
      # https://dlacombejr.github.io/programming/2015/09/13/sparse-filtering-implemenation-in-theano.html
      fs = T.sqrt(self.output ** 2 + 1e-8)    # numerical stability
      l2fs = T.sqrt(T.sum(fs ** 2, axis=1))   # l2 norm of row
      nfs = fs / l2fs.dimshuffle(0, 'x')      # normalize rows
      l2fn = T.sqrt(T.sum(nfs ** 2, axis=0))  # l2 norm of column
      self.output = nfs / l2fn.dimshuffle('x', 0)   # normalize columns
    self.output.name = "%s.output" % self.name
    self._output = output

  def transfer_output(self, device):
    if device is None:
      device = str(theano.config.device)
      if self.device is None:
        return self.output
    if device != self.device:
      self.output = self._output.transfer(device) # requires Theano 0.8
    else:
      self.output = self._output

  def to_json(self):
    attrs = super(Layer, self).to_json()
    attrs['class'] = self.layer_class
    return attrs

  def cost(self):
    """
    :rtype: (theano.Variable | None, dict[theano.Variable,theano.Variable] | None)
    :returns: cost, known_grads
    """
    return None, None

  def cost_scale(self):
    """
    :rtype: theano.Variable
    """
    return T.constant(self.attrs.get("cost_scale", 1.0), dtype="float32")

  def errors(self):
    """
    :rtype: theano.Variable
    """
    return None
