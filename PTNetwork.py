#! /usr/bin/python3

from __future__ import print_function

import torch
import torch.nn as nn
import json
import h5py

from PTLayers import *
from Log import log

from Util import dict_joined, as_str
from NetworkDescription import LayerNetworkDescription

LayerClasses = {}

def _initLayerClasses():
  global LayerClasses
  from inspect import isclass
  import PTLayers
  mods = [PTLayers]
  for mod in mods:
    for _, clazz in vars(mod).items():
      if not isclass(clazz): continue
      layer_class = getattr(clazz, "layer_class", None)
      if not layer_class: continue
      LayerClasses[layer_class] = clazz

_initLayerClasses()

def get_layer_class(name, raise_exception=True):
  """
  :type name: str
  :rtype: type(NetworkHiddenLayer.HiddenLayer)
  """
  if name in LayerClasses:
    return LayerClasses[name]
  if name.startswith("config."):
    from Config import get_global_config
    config = get_global_config()
    cls = config.typed_value(name[len("config."):])
    import inspect
    if not inspect.isclass(cls):
      if raise_exception:
        raise Exception("get_layer_class: %s not found" % name)
      else:
        return None
    if cls.layer_class is None:
      # Will make Layer.save() (to HDF) work correctly.
      cls.layer_class = name
    return cls
  if raise_exception:
    raise Exception("get_layer_class: invalid layer type: %s" % name)
  return None

class LayerNetwork():
  def __init__(self, n_out=None):
    super(LayerNetwork, self).__init__()
    self.n_out = n_out
    self.hidden = {}; """ :type: dict[str,ForwardLayer|RecurrentLayer] """
    self.recurrent = False  # any of the from_...() functions will set this
    self.output = {}; " :type: dict[str,FramewiseOutputLayer] "
    self.json_content = "{}"
    self.dtype = {k: 'float32' for k in self.n_out}
    self.data = {k: None for k in self.n_out}
    self.index = {k: None for k in self.n_out}

  def zero_grad(self):
    for k in self.hidden:
      self.hidden[k].zero_grad()
    for k in self.output:
      self.output[k].zero_grad()

  def train(self):
    for k in self.hidden:
      self.hidden[k].train()
    for k in self.output:
      self.output[k].train()

  def eval(self):
    for k in self.hidden:
      self.hidden[k].eval()
    for k in self.output:
      self.output[k].eval()

  def exec(self):
    output = {}
    for k in self.output:
      output[k], _ = self.output[k].process()

  def cost(self):
    result = []
    for k in self.output:
      result.append(self.output[k].cost())
    return result

  def errors(self):
    result = []
    for k in self.output:
      result.append(self.output[k].errors())
    return result

  def to(self, device):
    for k in self.output:
      self.output[k].to(device)
    for k in self.hidden:
      self.hidden[k].to(device)

  def parameters(self):
    params = []
    for k in self.output:
      params += self.output[k].parameters()
    for k in self.hidden:
      params += self.hidden[k].parameters()
    return params

  def named_parameters(self):
    params = []
    for k in self.output:
      params += self.output[k].named_parameters()
    for k in self.hidden:
      params += self.hidden[k].named_parameters()
    return params

  def set_parameters(self, params):
    i = 0
    for k in self.output:
      for p in self.output[k].parameters():
        p.data.copy_(params[i].reshape(p.data.shape))
        i += 1
    for k in self.hidden:
      for p in self.hidden[k].parameters():
        p.data.copy_(params[i].reshape(p.data.shape))
        i += 1

  @classmethod
  def from_config_topology(cls, config, mask=None, **kwargs):
    """
    :type config: Config.Config
    :param str mask: e.g. "unity" or None ("dropout"). "unity" is for testing.
    :rtype: LayerNetwork
    """
    json_content = cls.json_from_config(config)
    return cls.from_json_and_config(json_content, config, **kwargs)

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
    return json_content

  @classmethod
  def init_args_from_config(cls, config):
    """
    :rtype: dict[str]
    :returns the kwarg for cls.from_json()
    """
    num_inputs, num_outputs = LayerNetworkDescription.num_inputs_outputs_from_config(config)
    return {
      "n_in": num_inputs, "n_out": num_outputs
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
    n_out = {k:n_out[k][0] for k in n_out}
    n_out['data'] = n_in
    network = cls(n_out=n_out, **kwargs)
    network.n_out['data'] = n_in
    network.dtype['data'] = 'float32' # TODO
    network.dtype['classes'] = 'int32'
    kwargs = {'network':network}
    network.data_keys = set([])
    def traverse(content, layer_name):
      if layer_name in network.hidden or layer_name in network.output:
        return
      sources = []
      obj = content[layer_name].copy()
      layer_class = obj.pop('class', None)
      if not 'from' in obj and layer_class is not None:
        sources = [DataLayer(source='data',network=network)]
        network.data_keys.add('data')
      elif 'from' in obj and obj['from']:
        if not isinstance(obj['from'], list):
          obj['from'] = [ obj['from'] ]
        for prev in obj['from']:
          if not prev in content.keys() and prev != "null":
            sources.append(DataLayer(source=prev,network=network))
            network.data_keys.add(prev)
          elif prev != "null":
            traverse(content, prev)
            sources.append(network.get_layer(prev))
      obj.pop('from', None)
      params = { 'sources': sources,
                 'dropout' : 0.0,
                 'name' : layer_name,
                 'network': network }
      params.update(obj)
      layer_class = get_layer_class(layer_class)
      network.recurrent = network.recurrent or layer_class.recurrent
      network.add_layer(layer_class(**params))
    for layer_name in json_content:
      assert layer_name != "data", "this layer name is not allowed"
      traverse(json_content, layer_name)
    for k in network.hidden:
      if 'target' in network.hidden[k].attrs:
        network.data_keys.add(network.hidden[k].attrs['target'])
    for k in network.output:
      if 'target' in network.output[k].attrs:
        network.data_keys.add(network.output[k].attrs['target'])
    return network

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
    if isinstance(layer, OutputLayer):
      self.output[layer.name] = layer
    else:
      self.hidden[layer.name] = layer
    #self.layers.append(layer)
    return layer

  def num_params(self):
    return sum([self.hidden[h].num_params() for h in self.hidden]) + sum([self.output[k].num_params() for k in self.output])

  def save_hdf(self, model, epoch):
    """
    :type model: h5py.File
    :type epoch: int
    """
    grp = model.create_group('training')
    model.attrs['json'] = self.json_content
    #model.attrs['update_step'] = self.update_step # TODO
    model.attrs['epoch'] = epoch
    model.attrs['output'] = 'output' #self.output.keys
    model.attrs['n_in'] = self.n_out['data']
    out = model.create_group('n_out')
    for k in self.n_out:
      out.attrs[k] = self.n_out[k]
    #out_dim = out.create_group("dim")
    #for k in self.n_out:
    #  out_dim.attrs[k] = self.n_out[k][1]
    for h in self.hidden:
      self.hidden[h].save(model)
    for k in self.output:
      self.output[k].save(model)

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

  @classmethod
  def epoch_from_hdf_model(cls, model):
    """
    :type model: h5py.File
    :returns last epoch the model was trained on
    :rtype: int
    """
    epoch = model.attrs['epoch']
    return epoch

  def print_network_info(self, name="Network"):
    print("%s layer topology:" % name, file=log.v2)
    print("  input #:", self.n_out['data'], file=log.v2)
    for layer_name, layer in sorted(self.hidden.items()):
      print("  hidden %s %r #: %i" % (layer.layer_class, layer_name, layer.attrs["n_out"]), file=log.v2)
    if not self.hidden:
      print("  (no hidden layers)", file=log.v2)
    for layer_name, layer in sorted(self.output.items()):
      print("  output %s %r #: %i" % (layer.layer_class, layer_name, layer.attrs["n_out"]), file=log.v2)
    if not self.output:
      print("  (no output layers)", file=log.v2)
    print("net params #:", self.num_params(), file=log.v2)
    #print("net trainable params:", self.train_params_vars, file=log.v2)

  def get_used_data_keys(self):
    return self.data_keys
