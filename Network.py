#! /usr/bin/python2.7

import json
import h5py

from NetworkDescription import LayerNetworkDescription
from NetworkBaseLayer import Layer, SourceLayer
from NetworkLayer import get_layer_class
from NetworkLstmLayer import *
from NetworkOutputLayer import FramewiseOutputLayer, SequenceOutputLayer, LstmOutputLayer
from Log import log

class LayerNetwork(object):
  def __init__(self, n_in, n_out):
    """
    :param int n_in: input dim of the network
    :param dict[str,(int,int)] n_out: output dim of the network
    """
    self.x = T.tensor3('x'); """ :type: theano.Variable """
    self.y = {} #T.ivector('y'); """ :type: theano.Variable """
    self.l = T.ivector('l'); """ :type: theano.Variable """
    self.c = {} #T.imatrix('c'); """ :type: theano.Variable """
    self.i = T.bmatrix('i'); """ :type: theano.Variable """
    self.j = T.bmatrix('j'); """ :type: theano.Variable """
    self.constraints = T.constant(0)
    Layer.initialize_rng()
    self.n_in = n_in
    self.n_out = n_out
    self.hidden = {}; """ :type: dict[str,ForwardLayer|RecurrentLayer] """
    self.train_params_vars = []; """ :type: list[theano.compile.sharedvalue.SharedVariable] """
    self.description = None; """ :type: LayerNetworkDescription | None """
    self.train_param_args = None; """ :type: dict[str] """
    self.recurrent = False  # any of the from_...() functions will set this
    self.output = {}; " :type: dict[str,FramewiseOutputLayer] "
    self.costs = {}
    self.total_cost = T.constant(0)
    self.errors = {}

    for target in self.n_out:
      if self.n_out[target][1] == 1:
        self.y[target] = T.ivector('y')
      else:
        self.y[target] = T.imatrix('y')
      self.y[target].n_out = self.n_out[target][0]

  @classmethod
  def from_config_topology(cls, config, mask=None, train_flag = False):
    """
    :type config: Config.Config
    :param str mask: e.g. "unity" or None ("dropout"). "unity" is for testing.
    :rtype: LayerNetwork
    """
    json_content = cls.json_from_config(config, mask=mask)
    return cls.from_json_and_config(json_content, config, mask=mask, train_flag=train_flag)

  @classmethod
  def json_from_config(cls, config, mask=None):
    """
    :type config: Config.Config
    :param str mask: "unity", "none" or "dropout"
    :rtype: dict[str]
    """
    json_content = None
    if config.has("network") and config.is_typed("network"):
      json_content = config.value()
    if config.network_topology_json:
      try:
        json_content = json.loads(config.network_topology_json)
      except ValueError as e:
        print >> log.v4, "----- BEGIN JSON CONTENT -----"
        print >> log.v4, config.network_topology_json
        print >> log.v4, "------ END JSON CONTENT ------"
        assert False, "invalid json content, %r" % e
      assert isinstance(json_content, dict)
    if not json_content:
      if not mask:
        if sum(config.float_list('dropout', [0])) > 0.0:
          mask = "dropout"
      description = LayerNetworkDescription.from_config(config)
      json_content = description.to_json_content(mask=mask)
    if 'network' in json_content:
      json_content = json_content['network']
    return json_content

  @classmethod
  def from_description(cls, description, mask=None, train_flag = False):
    """
    :type description: NetworkDescription.LayerNetworkDescription
    :param str mask: e.g. "unity" or None ("dropout")
    :rtype: LayerNetwork
    """
    json_content = description.to_json_content(mask=mask)
    network = cls.from_json(json_content, mask=mask, train_flag=train_flag,
                            n_in=description.num_inputs, n_out=description.num_outputs)
    return network

  @classmethod
  def json_init_args_from_config(cls, config):
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

  @classmethod
  def from_json_and_config(cls, json_content, config, mask=None, train_flag=False):
    """
    :type config: Config.Config
    :type json_content: str | dict
    :param str mask: e.g. "unity" or None ("dropout"). "unity" is for testing.
    :rtype: LayerNetwork
    """
    return cls.from_json(json_content, mask=mask, train_flag=train_flag,
                         **cls.json_init_args_from_config(config))

  @classmethod
  def from_json(cls, json_content, n_in, n_out, mask=None, sparse_input = False, target = 'classes', train_flag = False):
    """
    :type json_content: dict[str]
    :type n_in: int
    :type n_out: dict[str,(int,int)]
    :param str mask: e.g. "unity" or None ("dropout")
    :rtype: LayerNetwork
    """
    network = cls(n_in, n_out)
    assert isinstance(json_content, dict)
    network.recurrent = False
    if hasattr(LstmLayer, 'sharpgates'):
      del LstmLayer.sharpgates
    def traverse(content, layer_name, network, index):
      source = []
      obj = content[layer_name].copy()
      act = obj.pop('activation', 'logistic')
      cl = obj.pop('class', None)
      if not 'from' in obj:
        source = [SourceLayer(network.n_in, network.x, sparse = sparse_input, name = 'data')]
      elif obj['from']:
        if not isinstance(obj['from'], list):
          obj['from'] = [ obj['from'] ]
        for prev in obj['from']:
          if prev == 'data':
            source.append(SourceLayer(network.n_in, network.x, sparse = sparse_input, name = 'data'))
          elif prev != "null":
            if not network.hidden.has_key(prev) and not network.output.has_key(prev):
              index = traverse(content, prev, network, index)
            source.append(network.hidden[prev] if prev in network.hidden else network.output[prev])
      if 'encoder' in obj:
        encoder = []
        if not isinstance(obj['encoder'], list):
          obj['encoder'] = [obj['encoder']]
        for prev in obj['encoder']:
          if not network.hidden.has_key(prev) and not network.output.has_key(prev):
            traverse(content, prev, network, index)
          encoder.append(network.hidden[prev] if prev in network.hidden else network.output[prev])
        obj['encoder'] = encoder
      if 'base' in obj: # TODO(doetsch) string/layer transform should be smarter
        base = []
        if not isinstance(obj['base'], list):
          obj['base'] = [obj['base']]
        for prev in obj['base']:
          if not network.hidden.has_key(prev) and not network.output.has_key(prev):
            traverse(content, prev, network, index)
          base.append(network.hidden[prev] if prev in network.hidden else network.output[prev])
        obj['base'] = base
      if 'copy_input' in obj:
        index = traverse(content, obj['copy_input'], network, index)
        obj['copy_input'] = network.hidden[obj['copy_input']] if obj['copy_input'] in network.hidden else network.output[obj['copy_input']]
      if 'centroids' in obj:
        index = traverse(content, obj['centroids'], network, index)
        obj['centroids'] = network.hidden[obj['centroids']] if obj['centroids'] in network.hidden else network.output[obj['centroids']]

      obj.pop('from', None)
      params = { 'sources': source,
                 'dropout' : 0.0,
                 'name' : layer_name,
                 "train_flag": train_flag,
                 'network': network }
      params.update(obj)
      params["mask"] = mask # overwrite
      params['index'] = index if not 'encoder' in obj else network.j
      params['y_in'] = network.y
      if cl == 'softmax':
        if not 'target' in params:
          params['target'] = target
        return network.make_classifier(**params)
      else:
        layer_class = get_layer_class(cl)
        params.update({'activation': act, 'name': layer_name})
        if layer_class.recurrent:
          network.recurrent = True
        return network.add_layer(layer_class(**params))
    for layer_name in json_content:
      if layer_name == 'output' or 'target' in json_content[layer_name]:
        traverse(json_content, layer_name, network, network.i)
    return network

  @classmethod
  def from_hdf_model_topology(cls, model, input_mask=None, sparse_input = False, target = 'classes', train_flag = False):
    """
    :type model: h5py.File
    :param str mask: e.g. "unity"
    :rtype: LayerNetwork
    """
    grp = model['training']
    n_out = {}
    try:
      for k in model['n_out'].attrs:
        dim = 1 if not 'dim' in model['n_out'] else model['n_out/dim'].attrs[k]
        n_out[k] = [model['n_out'].attrs[k], 1]
    except Exception:
      n_out = {'classes':[model.attrs['n_out'],1]}
    network = cls(model.attrs['n_in'], n_out)
    network.recurrent = False
    def traverse(model, layer_name, network, index):
      mask = input_mask
      if not input_mask and 'mask' in model[layer_name].attrs:
        mask = model[layer_name].attrs['mask']
      if 'from' in model[layer_name].attrs and model[layer_name].attrs['from'] != 'data':
        x_in = []
        for s in model[layer_name].attrs['from'].split(','):
          if s == 'data':
            x_in.append(SourceLayer(network.n_in, network.x, sparse = sparse_input, name = 'data'))
          elif s != "null" and s != "": # this is allowed, recurrent states can be passed as input
            if not network.hidden.has_key(s):
              index = traverse(model, s, network, index)
            x_in.append(network.hidden[s])
          elif s == "":
            assert not s
            # Fix for old models via NetworkDescription.
            s = Layer.guess_source_layer_name(layer_name)
            if not s:
              # Fix for data input. Just like in NetworkDescription, so that param names are correct.
              x_in.append(SourceLayer(n_out=network.n_in, x_out=network.x, name=""))
            else:
              if not network.hidden.has_key(s):
                index = traverse(model, s, network, index)
              # Add just like in NetworkDescription, so that param names are correct.
              x_in.append(SourceLayer(n_out=network.hidden[s].attrs['n_out'], x_out=network.hidden[s].output, name=""))
      else:
        x_in = [ SourceLayer(network.n_in, network.x, sparse = sparse_input, name = 'data') ]
      if 'encoder' in model[layer_name].attrs:
        encoder = []
        for s in model[layer_name].attrs['encoder'].split(','):
          if s != "":
            if not network.hidden.has_key(s):
              traverse(model, s, network, index)
            encoder.append(network.hidden[s])
      if 'base' in model[layer_name].attrs: # TODO see json
        base = []
        for s in model[layer_name].attrs['base'].split(','):
          if s != "":
            if not network.hidden.has_key(s):
              traverse(model, s, network, index)
            base.append(network.hidden[s])
      if 'copy_input' in model[layer_name].attrs:
        index = traverse(model, model[layer_name].attrs['copy_input'], network, index)
        copy_input = network.hidden[model[layer_name].attrs['copy_input']]
      if 'centroids' in model[layer_name].attrs:
        index = traverse(model, model[layer_name].attrs['centroids'], network, index)
        centroids = network.hidden[model[layer_name].attrs['centroids']]
      cl = model[layer_name].attrs['class']
      if cl == 'softmax' or cl == "lstm_softmax":
        params = { 'dropout' : 0.0,
                   'name' : 'output',
                   'mask' : mask,
                   'train_flag' : train_flag }
        params.update(model[layer_name].attrs)
        if 'encoder' in model[layer_name].attrs:
          params['encoder'] = encoder #network.hidden[model[layer_name].attrs['encoder']] if model[layer_name].attrs['encoder'] in network.hidden else network.output[model[layer_name].attrs['encoder']]
        if 'base' in model[layer_name].attrs:
          params['base'] = base
        if 'centroids' in model[layer_name].attrs:
          params['centroids'] = centroids
        if 'copy_input' in model[layer_name].attrs:
          params['copy_input'] = copy_input
        if not 'target' in params:
          params['target'] = target
        params['sources'] = x_in
        params.pop('from', None)
        params.pop('class', None)
        network.make_classifier(**params)
      else:
        try:
          act = model[layer_name].attrs['activation']
        except Exception:
          act = 'logistic'
        params = { 'sources': x_in,
                   'n_out': model[layer_name].attrs['n_out'],
                   'activation': act,
                   'dropout': model[layer_name].attrs['dropout'] if train_flag else 0.0,
                   'name': layer_name,
                   'mask': mask,
                   'train_flag' : train_flag,
                   'carry' : model[layer_name].attrs['carry'],
                   'depth' : model[layer_name].attrs['depth'],
                   'network': network,
                   'index' : index if not 'encoder' in model[layer_name].attrs else network.j }
        params['y_in'] = network.y
        layer_class = get_layer_class(cl)
        for p in ['truncation', 'projection', 'reverse', 'sharpgates', 'sampling', 'carry_time', 'unit', 'direction', 'psize', 'pact', 'pdepth', 'attention', 'L1', 'L2', 'lm', 'dual', 'acts', 'acth', 'filename', 'dset', 'entropy_weight', "droplm", "dropconnect"]: # uugh i hate this so much
          if p in model[layer_name].attrs.keys():
            params[p] = model[layer_name].attrs[p]
        if 'encoder' in model[layer_name].attrs:
          params['encoder'] = encoder #network.hidden[model[layer_name].attrs['encoder']] if model[layer_name].attrs['encoder'] in network.hidden else network.output[model[layer_name].attrs['encoder']]
        if 'base' in model[layer_name].attrs:
          params['base'] = base
        if 'centroids' in model[layer_name].attrs:
          params['centroids'] = centroids
        if layer_class.recurrent:
          network.recurrent = True
        return network.add_layer(layer_class(**params))
    for layer_name in model:
      if layer_name == model.attrs['output'] or 'target' in model[layer_name].attrs:
        traverse(model, layer_name, network, network.i)
    return network

  def add_layer(self, layer):
    """
    :type layer: NetworkHiddenLayer.HiddenLayer
    """
    assert layer.name
    self.hidden[layer.name] = layer
    self.constraints += layer.make_constraints()
    return layer.output_index()

  def make_classifier(self, name='output', target='classes', **kwargs):
    """
    :param list[NetworkBaseLayer.Layer] sources: source layers
    :param str loss: loss type, "ce", "ctc" etc
    """
    if not "loss" in kwargs: kwargs["loss"] = "ce"
    self.loss = kwargs["loss"]
    if self.loss in ('ctc', 'ce_ctc', 'ctc2', 'sprint', 'sprint_smoothed'):
      layer_class = SequenceOutputLayer
    elif self.loss == 'cedec':
      layer_class = LstmOutputLayer
    else:
      layer_class = FramewiseOutputLayer

    if target != "null":
      if 'dtype' in kwargs and kwargs['dtype'].startswith('float'):
        if self.n_out[target][1] == 1:
          self.y[target] = T.fvector('y')
        else:
          self.y[target] = T.fmatrix('y')
        self.y[target].n_out = self.n_out[target][0]
    dtype = kwargs.pop('dtype', 'int32')

    if 'n_symbols' in kwargs:
      targets = T.ivector()
      kwargs.setdefault('n_out', kwargs.pop('n_symbols'))
    elif target != "null":
      kwargs.setdefault('n_out', self.n_out[target][0])
      targets = self.c if self.loss == 'ctc' else self.y[target]
    else:
      targets = None
    kwargs['index'] = self.j
    self.output[name] = layer_class(name=name, target=target, y=targets, **kwargs)
    self.output[name].set_attr('dtype', dtype)
    if target != "null":
      self.errors[name] = self.output[name].errors()
      self.declare_train_params()
      cost = self.output[name].cost()
      self.costs[name], self.known_grads = cost[:2]
      if len(cost) > 2:
        self.ctc_priors = cost[2]
        assert self.ctc_priors is not None
      else:
        self.ctc_priors = None
      loss_scale = T.constant(self.output[name].attrs.get("loss_scale", 1.0), dtype="float32")
      self.total_cost += self.costs[name] * loss_scale
      self.constraints += self.output[name].make_constraints()

  def get_objective(self):
    return self.total_cost + self.constraints

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
    return self.get_params_vars(**self.get_train_param_args_default())

  def get_train_param_args_default(self):
    """
    :returns default kwargs for self.get_params(), which returns all params with this.
    """
    return {
      "hidden_layer_selection": sorted(self.hidden.keys()),  # Use all.
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
      outattrs = self.output[name].attrs.copy()
      outattrs['from'] = outattrs['from'].split(',')
      outattrs['class'] = 'softmax'
      if ',' in outattrs:
        out[name] = outattrs.split(',')
      else:
        out[name] = outattrs
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
        print >> log.v2, "unable to load layer ", name
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

  @classmethod
  def epoch_from_hdf_model_filename(cls, model_filename):
    """
    :type model_filename: str
    :returns last epoch the model was trained on
    :rtype: int
    """
    model = h5py.File(model_filename, "r")
    epoch = cls.epoch_from_hdf_model(model)
    model.close()
    return epoch

