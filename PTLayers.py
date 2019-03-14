from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

floatX = 'float32'

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

  def __init__(self, name, network):
    """
    :param str layer_class: name of layer type, e.g. "hidden", "recurrent", "lstm" or so. see LayerClasses.
    :param str name: custom layer name, e.g. "hidden_2"
    :param PTNetwork.LayerNetwork network: the network which we will be part of
    """
    super(Container, self).__init__()
    self.params = {}
    self.attrs = {}
    self.name = as_str(name.encode("utf8"))
    self.network = network

  def __repr__(self):
    return "<%s class:%s name:%s>" % (self.__class__, self.layer_class, self.name)

  def save(self, head):
    """
    :type head: h5py.File
    """
    grp = head.create_group(self.name)
    grp.attrs['class'] = self.layer_class
    for p in self.named_parameters():
      value = p[1].detach().cpu().numpy()
      dset = grp.create_dataset(p[0], value.shape, dtype='f')
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
    if grp_class != self.layer_class:
      from NetworkLayer import get_layer_class
      if not get_layer_class(grp_class, raise_exception=False) is get_layer_class(self.layer_class):
        print("warning: invalid layer class (expected " + self.layer_class + " got " + grp.attrs['class'] + ")", file=log.v3)
    parameters = {}
    for p in self.named_parameters():
      if p[0] not in grp:
        print("unable to load parameter %s in %s" % (p[0], self.name), file=log.v4)
      parameters[p[0]] = p[1]
    for p in grp:
      if p in parameters:
        if parameters[p].detach().cpu().numpy().shape == grp[p].shape:
          array = grp[p][...]
          assert not (numpy.isinf(array).any() or numpy.isnan(array).any())
          parameters[p].data.copy_(torch.from_numpy(array))
        else:
          print("warning: invalid layer parameter shape for parameter " + p + " of layer " + self.name + \
            " (expected  " + str(parameters[p].detach().cpu().numpy().shape) + \
            " got " + str(grp[p].shape) + ")", file=log.v2)
      else:
        print("unable to match parameter %s in %s" % (p, self.name), file=log.v4)

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


class Layer(Container):
  recurrent = False

  def __init__(self, dependencies=None, sources=None, dropout=0.0, **kwargs):
    """
    :param list[NetworkBaseLayer.Layer] sources: list of source layers
    :param int n_out: output dim of W_in and dim of bias
    """
    super(Layer, self).__init__(**kwargs)
    self.output = None
    self.deps = {}
    if dependencies is not None:
      self.deps = { dep : None for dep in dependencies }
    self.sources = sources

  def process(self):
    if self.output is not None:
      return self.output, self.index_out
    for k in self.deps.keys():
      self.deps[k] = k.process(x, i)
    assert self.sources, "require sources for " + self.name + str(self.output)
    sources = [None for i in range(len(self.sources))]
    for i in range(len(self.sources)):
      sources[i], self.index_in = self.sources[i].process()
    x = torch.cat(sources, dim=2)
    self.output, self.index_out = self.forward(x, self.index_in)
    return self.output, self.index_out

  def zero_grad(self):
    super(Layer, self).zero_grad()
    self.output = None

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
    return self.attrs.get("cost_scale", 1.0)

  def errors(self):
    """
    :rtype: theano.Variable
    """
    return None

class DataLayer(Layer):
  layer_class = "data"

  def __init__(self, source, **kwargs):
    super(DataLayer, self).__init__(name=source, **kwargs)
    self.attrs['source'] = source
    self.attrs['n_out'] = self.network.n_out[source]
    #self.output = self.network.data[source]
    #self.index_out = self.network.index[source]
    #print(self.name,self.index_out)

  def process(self):
    return self.network.data[self.attrs['source']], self.network.index[self.attrs['source']]

class LSTMLayer(Layer):
  layer_class = "rec"
  recurrent = True

  def __init__(self, n_out, direction=1, **kwargs):
    super(LSTMLayer, self).__init__(**kwargs)
    self.attrs['n_out'] = n_out
    self.attrs['direction'] = direction
    n_in = sum([x.attrs['n_out'] for x in self.sources])
    self.module = nn.LSTM(int(n_in), int(n_out), 1)

  def forward(self, x, i):
    self.module.flatten_parameters()
    if self.attrs['direction'] == -1: # ugh
      idx = torch.LongTensor([i for i in range(x.size(0)-1, -1, -1)])
      idx = idx.to(x.device)
      x = x.index_select(0,idx)
    x, _ = self.module(x)
    if self.attrs['direction'] == -1: # ugh
      x = x.index_select(0,idx)
    return x, i

class OutputLayer(Layer):
  layer_class = "softmax"

  def __init__(self, loss = 'ce', target = 'classes', **kwargs):
    super(OutputLayer, self).__init__(**kwargs)
    self.attrs['loss'] = loss
    self.attrs['target'] = target
    self.attrs['n_out'] = self.network.n_out[target]
    n_in = sum([x.attrs['n_out'] for x in self.sources])
    self.module = nn.Linear(n_in, self.attrs['n_out'])
    self.loss = nn.NLLLoss()

  def forward(self, x, i):
    self.index_out = i
    self.output = F.softmax(self.module(x), dim=2)
    return self.output, self.index_out

  def cost(self):
    y = self.network.data[self.attrs['target']]
    loss_function = nn.NLLLoss()
    i = self.index_out.view(-1).nonzero().long().view(-1)
    c = torch.log(torch.index_select(
          self.output.view(-1,self.output.size(2)), 0, i))
    y = torch.index_select(y.view(-1), 0, i)
    return self.loss(c, y.long())

  def errors(self):
    y = self.network.data[self.attrs['target']]
    i = self.index_out.view(-1).nonzero().long().view(-1)
    y = torch.index_select(y.view(-1), 0, i)
    c = torch.index_select(torch.argmax(self.output,dim=2).view(-1), 0, i)
    #print(c.size(0),self.output.size(0)*self.output.size(1),y.size(0))
    return (y.long() != c).float().sum() # * self.index.float().sum() / (self.logpcx.size(0) * self.logpcx.size(1))
