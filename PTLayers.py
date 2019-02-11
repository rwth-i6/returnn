from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from PTDevice import floatX

from math import sqrt
import numpy
from Log import log
from Util import as_str
import json


class Container(nn.Module):
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
    self.params = {}
    self.attrs = {}
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
    if grp_class == "<unknown_softmax>": grp_class = "softmax"  # bug in some CRNN version. can be ignored.
    if grp_class != self.layer_class:
      from NetworkLayer import get_layer_class
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
    import Config, Util
    try:
      config = Config.get_global_config()
    except Exception:
      config = None
    else:
      config = Util.DictAsObj(config.typed_dict)
    eval_locals = {
      "numpy": numpy,
      "rng": self.rng,
      "config": config,
      "self": self,
      "n": n,
      "name": name,
      "sqrt": numpy.sqrt,
      "log": numpy.log,
      "zeros": (lambda: numpy.zeros(size, dtype=floatX)),
      "random_normal": random_normal,
      "random_uniform": random_uniform
    }
    values = eval(init_eval_str, eval_locals)
    values = numpy.asarray(values, dtype=floatX)
    assert values.shape == (n,)
    return self.shared(values, name)

  def create_random_normal_weights(self, n, m, scale=None, name=None):
    if name is None: name = self.name
    if not scale:
      scale =  numpy.sqrt((n + m) / 12.)
    else:
      scale = numpy.sqrt(scale / 12.)
    if self.depth > 1:
      values = numpy.asarray(self.rng.normal(loc=0.0, scale=1.0 / scale, size=(n, self.depth, m)), dtype=floatX)
    else:
      values = numpy.asarray(self.rng.normal(loc=0.0, scale=1.0 / scale, size=(n, m)), dtype=floatX)
    return self.shared(values, name)

  def create_random_uniform_weights(self, n, m, p=None, p_add=None, l=None, name=None, depth=None):
    if not depth: depth = self.depth
    if name is None: name = 'W_' + self.name
    assert not (p and l)
    if not p: p = n + m
    if p_add: p += p_add
    if not l: l = sqrt(6.) / sqrt(p)  # 1 / sqrt(p)
    if depth > 1:
      values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(n, depth, m)), dtype=floatX)
    else:
      values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(n, m)), dtype=floatX)
    return self.shared(values, name)

  def create_random_uniform_weights1(self, n, m, p=None, l=None, name=None):
    if name is None: name = 'W_' + self.name
    assert not (p and l)
    if not p: p = n + m
    if not l: l = sqrt(6.) / sqrt(p)  # 1 / sqrt(p)
    values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(n, m)), dtype=floatX)
    return self.shared(values, name)

  def create_random_uniform_weights2(self, n, m=None, name=None):
    if name is None: name = 'W_' + self.name
    l = sqrt(1. / n)
    shape = [n]
    if m: shape += [m]
    values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=shape), dtype=floatX)
    return self.shared(values, name)

  def create_random_unitary_weights(self, n, m, name=None):
    x = self.rng.randn(n, m)
    u, s, v = numpy.linalg.svd(x, full_matrices=0)
    if u.shape == (n, m):
      x = u
    else:
      x = v
    assert x.shape == (n, m)
    x = x.astype(floatX)
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
    x = x.astype(floatX)
    return self.shared(x, name)

  def _create_eval_weights(self, n, m, name, default_name_prefix, init_eval_str):
    """
    :param int n: input dimension
    :param int m: output dimension
    :param str|None name: layer name
    :rtype: theano.shared
    """
    if not name: name = "%s_%s_%i" % (default_name_prefix, self.name, len(self.params))
    import Config, Util
    try:
      config = Config.get_global_config()
    except Exception:
      config = None
    else:
      config = Util.DictAsObj(config.typed_dict)
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
      "eye": (lambda N=n, M=m: numpy.eye(N, M, dtype=floatX)),
      "random_normal": (
      lambda scale=None, **kwargs: self.create_random_normal_weights(n, m, scale=scale, name=name, **kwargs)),
      "random_uniform": (
      lambda l=None, p=None, **kwargs: self.create_random_uniform_weights(n, m, p=p, l=l, name=name, **kwargs)),
      "random_unitary": (lambda **kwargs: self.create_random_unitary_weights(n, m, name=name, **kwargs)),
      "random_unitary_tiled": (lambda **kwargs: self.create_random_unitary_tiled_weights(n, m, name=name, **kwargs))
    }
    v = eval(init_eval_str, eval_locals)
    if isinstance(v, numpy.ndarray):
      v = numpy.asarray(v, dtype=floatX)
      v = self.shared(v, name)
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
    inputs = 'inputs'
    self.output = network.y[inputs]
    n_out = network.n_out[inputs][0]
    index = network.j[inputs]

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

  def __init__(self, sources, n_out, index, y_in=None, target=None,
               cost_scale=1.0,
               dropout=0.0,
               trainable=True,
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
    self.set_attr('residual', residual)
    self.set_attr('n_out', n_out)
    self.set_attr('L1', L1)
    self.set_attr('L2', L2)
    if L2_eye:
      self.set_attr('L2_eye', L2_eye)
    self.device = device # if device else str(theano.config.device)
    for s in self.sources:
      s.transfer_output(self.device)
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

  def output_index(self):
    from theano.ifelse import ifelse
    index = self.index
    if self.sources:
      # In some cases, e.g. forwarding, the target index (for "classes") might have shape[0]==0.
      # Or shape[0]==1 with index[0]==0. See Dataset.shapes_for_batches().
      # Use source index in that case.
      have_zero = T.le(index.shape[0], 1) * T.eq(T.sum(index[0]), 0)
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
    if not delay:
      self.output = x_out
    else:
      self.output = T.inc_subtensor(T.zeros_like(x_out)[delay:], x_out[:-delay])
    self.set_attr('n_out', n_out)
    self.set_attr('delay', delay)
    self.index = index
    self.eval_flag = eval_flag

  def cost(self):
    return None, None

  def errors(self):
    return None

class OutputLayer(Layer):
  layer_class = "softmax"

  def __init__(self, loss, y):
    super(OutputLayer, self).__init__(**kwargs)
    self.W_in = nn.Parameter(torch.ones(self.sources[0].attrs['n_out'],self.attrs['n_out']))
    self.b = nn.Parameter(torch.zeros(self.attrs['n_out']))
    self.softmax = nn.Softmax()(x_in.mm(self.W_in) + self.b)

  def forward(self, x_in):
    z = x_in.mm(self.W_in) + self.b
    self.p_y_given_x = self.softmax(z)
    scores = torch.index_select(self.p_y_given_x.view(-1), self.index.nonzero().long().view(-1))
    return -torch.log(scores[self.y_in.view(-1)]).sum()
