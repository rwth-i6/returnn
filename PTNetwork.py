#! /usr/bin/python2.7

backend = 'pytorch'

from __future__ import print_function

import json
import h5py

from NetworkDescription import LayerNetworkDescription
from PTLayers import OutputLayer
from Util import dict_joined, as_str
from Log import log


class LayerNetwork(object):
  def __init__(self, n_in=None, n_out=None, train_flag=False, eval_flag=False):
    if n_out is None:
      assert base_network is not None
      n_out = base_network.n_out
    else:
      assert n_out is not None
      n_out = n_out.copy()
    if n_in is None:
      assert "data" in n_out
      n_in = n_out["data"][0]
    if "data" not in n_out:
      data_dim = 3
      n_out["data"] = (n_in, data_dim - 1)  # small hack: support input-data as target
    else:
      assert 1 <= n_out["data"][1] <= 2  # maybe obsolete check...
      data_dim = n_out["data"][1] + 1  # one more because of batch-dim
    self.constraints = {}
    self.total_constraints = T.constant(0)
    self.n_in = n_in
    self.n_out = n_out
    self.hidden = {}; """ :type: dict[str,ForwardLayer|RecurrentLayer] """
    self.train_params_vars = []; """ :type: list[theano.compile.sharedvalue.SharedVariable] """
    self.description = None; """ :type: LayerNetworkDescription | None """
    self.train_param_args = None; """ :type: dict[str] """
    self.recurrent = False  # any of the from_...() functions will set this
    self.default_mask = mask
    self.sparse_input = sparse_input
    self.default_target = target
    self.train_flag = train_flag
    self.eval_flag = eval_flag
    self.output = {}; " :type: dict[str,FramewiseOutputLayer] "
    self.known_grads = {}; " :type: dict[theano.Variable,theano.Variable]"
    self.json_content = "{}"
    self.costs = {}
    self.total_cost = T.constant(0)
    self.objective = None
    self.update_step = 0
    self.errors = {}
    self.loss = None
    self.ctc_priors = None
    self.calc_step_base = None
    self.calc_steps = []
    self.base_network = base_network
    self.shared_params_network = shared_params_network

  @classmethod
  def from_config_topology(cls, config, mask=None, **kwargs):
    """
    :type config: Config.Config
    :param str mask: e.g. "unity" or None ("dropout"). "unity" is for testing.
    :rtype: LayerNetwork
    """
    json_content = cls.json_from_config(config, mask=mask)
    return cls.from_json_and_config(json_content, config, mask=mask, **kwargs)

  @classmethod
  def json_from_config(cls, config, mask=None):
    """
    :type config: Config.Config
    :param str mask: "unity", "none" or "dropout"
    :rtype: dict[str]
    """
    json_content = None
    if config.has("network") and config.is_typed("network"):
      json_content = config.typed_value("network")
      assert isinstance(json_content, dict)
      assert json_content
    elif config.network_topology_json:
      start_var = config.network_topology_json.find('(config:', 0) # e.g. ..., "n_out" : (config:var), ...
      while start_var > 0:
        end_var = config.network_topology_json.find(')', start_var)
        assert end_var > 0, "invalid variable syntax at " + str(start_var)
        var = config.network_topology_json[start_var+8:end_var]
        assert config.has(var), "could not find variable " + var
        config.network_topology_json = config.network_topology_json[:start_var] + config.value(var,"") + config.network_topology_json[end_var+1:]
        print("substituting variable %s with %s" % (var,config.value(var,"")), file=log.v4)
        start_var = config.network_topology_json.find('(config:', start_var+1)
      try:
        json_content = json.loads(config.network_topology_json)
      except ValueError as e:
        print("----- BEGIN JSON CONTENT -----", file=log.v3)
        print(config.network_topology_json, file=log.v3)
        print("------ END JSON CONTENT ------", file=log.v3)
        assert False, "invalid json content, %r" % e
      assert isinstance(json_content, dict)
      if 'network' in json_content:
        json_content = json_content['network']
      assert json_content
    if not json_content:
      if not mask:
        if sum(config.float_list('dropout', [0])) > 0.0:
          mask = "dropout"
      description = LayerNetworkDescription.from_config(config)
      json_content = description.to_json_content(mask=mask)
    return json_content

  @classmethod
  def from_description(cls, description, mask=None, **kwargs):
    """
    :type description: NetworkDescription.LayerNetworkDescription
    :param str mask: e.g. "unity" or None ("dropout")
    :rtype: LayerNetwork
    """
    json_content = description.to_json_content(mask=mask)
    network = cls.from_json(json_content, n_in=description.num_inputs, n_out=description.num_outputs, mask=mask, **kwargs)
    return network

  @classmethod
  def init_args_from_config(cls, config):
    """
    :rtype: dict[str]
    :returns the kwarg for cls.from_json()
    """
    num_inputs, num_outputs = LayerNetworkDescription.num_inputs_outputs_from_config(config)
    return {
      "n_in": num_inputs, "n_out": num_outputs,
      "sparse_input": config.bool("sparse_input", False),
      "target": config.value('target', 'classes')
    }

  def init_args(self):
    return {
      "n_in": self.n_in,
      "n_out": self.n_out,
      "mask": self.default_mask,
      "sparse_input": self.sparse_input,
      "target": self.default_target,
      "train_flag": self.train_flag,
      "eval_flag": self.eval_flag
    }

  @classmethod
  def from_json_and_config(cls, json_content, config, **kwargs):
    """
    :type config: Config.Config
    :type json_content: str | dict
    :rtype: LayerNetwork
    """
    network = cls.from_json(json_content, **dict_joined(kwargs, cls.init_args_from_config(config)))
    network.recurrent = network.recurrent or config.bool('recurrent', False)
    return network

  @classmethod
  def from_json(cls, json_content, n_in=None, n_out=None, network=None, **kwargs):
    """
    :type json_content: dict[str]
    :type n_in: int | None
    :type n_out: dict[str,(int,int)] | None
    :param LayerNetwork | None network: optional already existing instance
    :rtype: LayerNetwork
    """
    if network is None:
      network = cls(n_in=n_in, n_out=n_out, **kwargs)
      network.json_content = json.dumps(json_content, sort_keys=True)
    mask = network.default_mask
    sparse_input = network.sparse_input
    target = network.default_target
    train_flag = network.train_flag
    eval_flag = network.eval_flag
    templates = {}
    assert isinstance(json_content, dict)
    network.y['data'].n_out = network.n_out['data'][0]
    def traverse(content, layer_name, target, output_index, inherit=False):
      if layer_name in network.hidden:
        return network.hidden[layer_name].index
      if layer_name in network.output:
        return network.output[layer_name].index
      source = []
      obj = content[layer_name].copy()
      if 'inherit' in obj:
        if not obj['inherit'] in templates:
          traverse(content, obj['inherit'], target, output_index, True)
        template = templates[obj['inherit']].copy()
        for key in template.keys():
          if not key in obj.keys():
            obj[key] = template[key]
        del obj['inherit']
      templates[layer_name] = obj.copy()
      if inherit:
        return output_index
      cl = obj.pop('class', None)
      index = output_index
      if 'target' in obj:
        target = obj['target']
      dtype = obj.get("dtype", "int32")
      network.use_target(target, dtype=dtype)
      if not 'from' in obj and cl is not None:
        source = [SourceLayer(network.n_in, network.x, sparse=sparse_input, name='data', index=network.i)]
        index = network.i
      elif 'from' in obj and obj['from']:
        if not isinstance(obj['from'], list):
          obj['from'] = [ obj['from'] ]
        for prev in obj['from']:
          if prev == 'data':
            source.append(SourceLayer(network.n_in, network.x, sparse=sparse_input, name='data', index=network.i))
            index = network.i
          elif not prev in content.keys() and prev != "null":
            sparse = obj.pop('sparse_input', False)
            dtype = 'int32' if sparse else 'float32'
            source.append(SourceLayer(0, None, sparse=sparse, dtype=dtype, name='data', network=network, data_key=prev))
            index = source[-1].index
          elif prev != "null":
            index = traverse(content, prev, target, index)
            source.append(network.get_layer(prev))
      if 'encoder' in obj:
        encoder = []
        if not isinstance(obj['encoder'], list):
          obj['encoder'] = [obj['encoder']]
        for prev in obj['encoder']:
          traverse(content, prev, target, index)
          encoder.append(network.get_layer(prev))
        obj['encoder'] = encoder
      if 'base' in obj: # TODO(doetsch) string/layer transform should be smarter
        base = []
        if not isinstance(obj['base'], list):
          if ',' in obj['base']:
            obj['base'] = obj['base'].split(',')
          else:
            obj['base'] = [obj['base']]
        for prev in obj['base']:
          if prev == 'data':
            base.append(SourceLayer(network.n_in, network.x, sparse=sparse_input, name='data', index=network.i))
          else:
            traverse(content, prev, target, index)
            base.append(network.get_layer(prev))
        obj['base'] = base
      for key in [ 'copy_input', 'copy_output', 'aligner' ]:
        if key in obj:
          index = traverse(content, obj[key], target, index)
          obj[key] = network.get_layer(obj[key])
      if 'encoder' in obj and not source:
        index = output_index
      if 'target' in obj and obj['target'] != "null":
        index = network.j[obj['target']]
      obj.pop('from', None)
      params = { 'sources': source,
                 'dropout' : 0.0,
                 'name' : layer_name,
                 "train_flag": train_flag,
                 "eval_flag": eval_flag,
                 'network': network }
      params.update(obj)
      params["mask"] = mask # overwrite
      params['index'] = index
      params['y_in'] = network.y
      if cl:
        templates[layer_name]['class'] = cl
      if cl == 'softmax' or cl == 'decoder':
        if not 'target' in params:
          params['target'] = target
        if 'loss' in obj and obj['loss'] in ('ctc','hmm'):
          params['index'] = network.i
        elif target != "null":
          params['index'] = network.j[target] #output_index
        return network.make_classifier(**params)
      elif cl is not None:
        layer_class = get_layer_class(cl)
        params.update({'name': layer_name})
        if layer_class.recurrent:
          network.recurrent = True
        return network.add_layer(layer_class(**params)).index
    for layer_name in sorted(json_content):
      if layer_name in network.hidden or layer_name in network.output:
        continue
      if layer_name == "data":
        print("warning: layer with name 'data' will be ignored (this name is reserved)", file=log.v3)
        continue
      trg = target
      if 'target' in json_content[layer_name]:
        trg = json_content[layer_name]['target']
      if layer_name == 'output' or 'target' in json_content[layer_name] or json_content[layer_name].get("class", None) == "softmax":
        network.use_target(trg, dtype=json_content.get("dtype", json_content[layer_name].get('dtype',"int32")))
        if trg != "null": index = network.j[trg]
        else: index = network.i
        traverse(json_content, layer_name, trg, index)
    network.set_cost_constraints_and_objective()
    return network

  @classmethod
  def _n_in_out_from_hdf_model(cls, model):
    n_out_model = {}
    try:
      for k in model['n_out'].attrs:
        dim = 1 if not 'dim' in model['n_out'] else model['n_out/dim'].attrs[k]
        n_out_model[k] = [model['n_out'].attrs[k], dim]
    except Exception:
      n_out_model = {'classes': [model.attrs['n_out'], 1]}
    n_in_model = model.attrs['n_in']
    n_out_model.pop('data')
    return n_in_model, n_out_model

  @classmethod
  def from_hdf_model_topology(cls, model, **kwargs):
    """
    :type model: h5py.File
    :rtype: LayerNetwork
    """
    return cls.from_hdf(model=model, filename=None, load_params=False, **kwargs)

  @classmethod
  def from_hdf(cls, filename=None, model=None, load_params=True, **kwargs):
    """
    Gets the JSON from the hdf file, initializes the network and loads the network params.
    :param str|None filename: filename of hdf
    :param h5py.File|None model: hdf, if no filename is provided
    :param bool load_params: whether to load the params
    """
    if model is None:
      assert filename
      model = h5py.File(filename, "r")
      close_at_end = True
    else:
      assert not filename
      close_at_end = False
    assert "json" in model.attrs, "Maybe old network model where JSON was not stored. Use version before 2016-10-11."
    json_content_s = as_str(model.attrs['json'])
    assert json_content_s and json_content_s != "{}"
    json_content = json.loads(json_content_s)
    kwargs = kwargs.copy()
    if "n_out" not in kwargs:
      n_in, n_out = cls._n_in_out_from_hdf_model(model)
      n_out['__final'] = True
      kwargs["n_in"] = n_in
      kwargs["n_out"] = n_out
    network = cls.from_json(json_content, **kwargs)
    if load_params:
      network.load_hdf(model)
    if close_at_end:
      model.close()
    return network

  def use_target(self, target, dtype):
    if target in self.y: return
    if target == "null": return
    assert target in self.n_out
    ndim = self.n_out[target][1] + 1  # one more because of batch-dim
    self.y[target] = T.TensorType(dtype, (False,) * ndim)('y_%s' % target)
    self.y[target].n_out = self.n_out[target][0]
    self.j.setdefault(target, T.bmatrix('j_%s' % target))
    if getattr(self.y[target].tag, "test_value", None) is None:
      if ndim == 2:
        self.y[target].tag.test_value = numpy.zeros((3,2), dtype='int32')
      elif ndim == 3:
        self.y[target].tag.test_value = numpy.random.rand(3,2,self.n_out[target][0]).astype('float32')
    if getattr(self.j[target].tag, "test_value", None) is None:
      self.j[target].tag.test_value = numpy.ones((3,2), dtype="int8")

  def get_used_data_keys(self):
    return [k for k in sorted(self.j.keys()) if not k.endswith("[sparse:coo]")]

  def get_layer(self, layer_name):
    if layer_name in self.hidden:
      return self.hidden[layer_name]
    if layer_name in self.output:
      return self.output[layer_name]
    return None

  def get_all_layers(self):
    return sorted(self.hidden) + sorted(self.output)

  def add_layer(self, layer):
    """
    :type layer: NetworkHiddenLayer.Layer
    :rtype NetworkHiddenLayer.Layer
    """
    assert layer.name
    layer_errors = layer.errors()
    if isinstance(layer, OutputLayer) or layer.name == "output" or layer_errors is not None:
      is_output_layer = True
      self.output[layer.name] = layer
    else:
      is_output_layer = False
      self.hidden[layer.name] = layer
    if layer_errors is not None:
      self.errors[layer.name] = layer_errors
    if is_output_layer:
      if getattr(layer, "p_y_given_x", None) is None and layer.output:
        # Small little hack for layers which we use as output-layers whicgh don't set this.
        from TheanoUtil import time_batch_make_flat
        layer.p_y_given_x = layer.output
        layer.p_y_given_x_flat = time_batch_make_flat(layer.output)
      self.declare_train_params()
    return layer

  def add_cost_and_constraints(self, layer):
    self.constraints[layer.name] = layer.make_constraints()
    self.total_constraints += self.constraints[layer.name]
    cost = layer.cost()
    if cost[0]:
      self.costs[layer.name] = cost[0]
      self.total_cost += self.costs[layer.name] * layer.cost_scale()
    if cost[1]:
      self.known_grads.update(cost[1])

  def make_classifier(self, name='output', target='classes', **kwargs):
    """
    :param list[NetworkBaseLayer.Layer] sources: source layers
    :param str loss: loss type, "ce", "ctc" etc
    """
    if not "loss" in kwargs: kwargs["loss"] = "ce"
    self.loss = kwargs["loss"]
    layer_class = FramewiseOutputLayer
    dtype = kwargs.pop('dtype', 'int32')
    if target != "null" and target not in self.y:
      self.use_target(target, dtype=dtype)
    if target != "null":
      targets = self.y[target]
    else:
      targets = None
    if 'n_symbols' in kwargs:
      kwargs.setdefault('n_out', kwargs.pop('n_symbols'))
    elif target != "null":
      kwargs.setdefault('n_out', self.n_out[target][0])
    layer = layer_class(name=name, target=target, y=targets, dtype=dtype, **kwargs)
    self.add_layer(layer)
    return layer.index

  def set_cost_constraints_and_objective(self):
    for name, layer in sorted(self.hidden.items()) + sorted(self.output.items()):
      self.add_cost_and_constraints(layer)
    self.objective = self.total_cost + self.total_constraints

  def get_objective(self):
    return self.objective

  def get_params_vars(self, hidden_layer_selection, with_output):
    """
    :type hidden_layer_selection: list[str]
    :type with_output: bool
    :rtype: list[theano.compile.sharedvalue.SharedVariable]
    :returns list (with well-defined order) of shared variables
    """
    params = []
    """ :type: list[theano.compile.sharedvalue.SharedVariable] """
    for name in sorted(hidden_layer_selection):
      params += self.hidden[name].get_params_vars()
    if with_output:
      for name in self.output:
        params += self.output[name].get_params_vars()
    return params

  def get_all_params_vars(self):
    return self.get_params_vars(hidden_layer_selection=sorted(self.hidden.keys()), with_output=True)

  def get_train_param_args_default(self):
    """
    :returns default kwargs for self.get_params(), which returns all params with this.
    """
    return {
      "hidden_layer_selection": [name for (name, layer) in sorted(self.hidden.items())
                                 if layer.attrs.get("trainable", True)],  # Use all.
      "with_output": True
    }

  def declare_train_params(self, **kwargs):
    """
    Kwargs as in self.get_params(), or default values.
    """
    # Set default values, also for None.
    for key, value in self.get_train_param_args_default().items():
      if kwargs.get(key, None) is None:
        kwargs[key] = value
    # Force a unique representation of kwargs.
    kwargs["hidden_layer_selection"] = sorted(kwargs["hidden_layer_selection"])
    self.train_param_args = kwargs
    self.train_params_vars = self.get_params_vars(**kwargs)

  def num_params(self):
    return sum([self.hidden[h].num_params() for h in self.hidden]) + sum([self.output[k].num_params() for k in self.output])

  def get_params_dict(self):
    """
    :rtype: dict[str,dict[str,numpy.ndarray|theano.sandbox.cuda.CudaNdArray]]
    """
    params = { name: self.output[name].get_params_dict() for name in self.output }
    for h in self.hidden:
      params[h] = self.hidden[h].get_params_dict()
    return params

  def set_params_by_dict(self, params):
    """
    :type params: dict[str,dict[str,numpy.ndarray|theano.sandbox.cuda.CudaNdArray]]
    """
    for name in self.output:
      self.output[name].set_params_by_dict(params[name])
    for h in self.hidden:
      self.hidden[h].set_params_by_dict(params[h])

  def save_hdf(self, model, epoch):
    """
    :type model: h5py.File
    :type epoch: int
    """
    grp = model.create_group('training')
    model.attrs['json'] = self.json_content
    model.attrs['update_step'] = self.update_step
    model.attrs['epoch'] = epoch
    model.attrs['output'] = 'output' #self.output.keys
    model.attrs['n_in'] = self.n_in
    out = model.create_group('n_out')
    for k in self.n_out:
      out.attrs[k] = self.n_out[k][0]
    out_dim = out.create_group("dim")
    for k in self.n_out:
      out_dim.attrs[k] = self.n_out[k][1]
    for h in self.hidden:
      self.hidden[h].save(model)
    for k in self.output:
      self.output[k].save(model)

  def to_json_content(self):
    out = {}
    for name in self.output:
      out[name] = self.output[name].to_json()
    for h in self.hidden.keys():
      out[h] = self.hidden[h].to_json()
    return out

  def to_json(self):
    json_content = self.to_json_content()
    return json.dumps(json_content, sort_keys=True)

  def load_hdf(self, model):
    """
    :type model: h5py.File
    :returns last epoch this was trained on
    :rtype: int
    """
    for name in self.hidden:
      if not name in model:
        print("unable to load layer", name, file=log.v2)
      else:
        self.hidden[name].load(model)
    for name in self.output:
      self.output[name].load(model)
    return self.epoch_from_hdf_model(model)

  def print_network_info(self, name="Network"):
    print("%s layer topology:" % name, file=log.v2)
    print("  input #:", self.n_in, file=log.v2)
    for layer_name, layer in sorted(self.hidden.items()):
      print("  hidden %s %r #: %i" % (layer.layer_class, layer_name, layer.attrs["n_out"]), file=log.v2)
    if not self.hidden:
      print("  (no hidden layers)", file=log.v2)
    for layer_name, layer in sorted(self.output.items()):
      print("  output %s %r #: %i" % (layer.layer_class, layer_name, layer.attrs["n_out"]), file=log.v2)
    if not self.output:
      print("  (no output layers)", file=log.v2)
    print("net params #:", self.num_params(), file=log.v2)
    print("net trainable params:", self.train_params_vars, file=log.v2)
