#! /usr/bin/python2.7

import json
import inspect

import theano.tensor as T
import h5py

from NetworkDescription import LayerNetworkDescription
from NetworkHiddenLayer import ForwardLayer
from NetworkLayer import Layer, SourceLayer
from NetworkLstmLayer import LstmLayer, OptimizedLstmLayer
from NetworkOutputLayer import FramewiseOutputLayer, SequenceOutputLayer
from NetworkRecurrentLayer import RecurrentLayer
from Log import log


LayerClasses = {
  'forward': ForwardLayer,  # used in crnn.config format
  'hidden': ForwardLayer,  # used in JSON format
  'recurrent': RecurrentLayer,
  'lstm': LstmLayer,
  'lstm_opt': OptimizedLstmLayer,
}


def get_layer_class(name):
  """
  :type name: str
  :rtype: type(NetworkHiddenLayer.HiddenLayer)
  """
  if name in LayerClasses:
    return LayerClasses[name]
  assert False, "invalid layer type: " + name


class LayerNetwork(object):
  def __init__(self, n_in, n_out, mask="unity"):
    """
    :param int n_in: input dim of the network
    :param int n_out: output dim of the network
    :param str mask: e.g. "unity"
    """
    self.x = T.tensor3('x'); """ :type: theano.Variable """
    self.y = {} #T.ivector('y'); """ :type: theano.Variable """
    self.l = T.ivector('l'); """ :type: theano.Variable """
    self.c = {} #T.imatrix('c'); """ :type: theano.Variable """
    self.i = T.bmatrix('i'); """ :type: theano.Variable """
    Layer.initialize_rng()
    self.n_in = n_in
    self.n_out = n_out
    self.mask = mask
    self.hidden = {}; """ :type: dict[str,ForwardLayer|RecurrentLayer] """
    self.train_params_vars = []; """ :type: list[theano.compile.sharedvalue.SharedVariable] """
    self.description = None; """ :type: LayerNetworkDescription | None """
    self.train_param_args = None; """ :type: dict[str] """
    self.objective = {}
    self.output = {}
    self.cost = {}
    self.errors = {}

  @classmethod
  def from_config_topology(cls, config, mask="unity"):
    """
    :type config: Config.Config
    :param str mask: e.g. "unity" or "dropout"
    :rtype: LayerNetwork
    """
    if config.network_topology_json is not None:
      num_inputs, num_outputs = LayerNetworkDescription.num_inputs_outputs_from_config(config)
      return cls.from_json(config.network_topology_json, num_inputs, num_outputs, mask, config.value('target', 'classes'))

    description = LayerNetworkDescription.from_config(config)
    return cls.from_description(description, mask)

  @classmethod
  def from_description(cls, description, mask="unity"):
    """
    :type description: NetworkDescription.LayerNetworkDescription
    :param str mask: e.g. "unity" or "dropout"
    :rtype: LayerNetwork
    """
    network = cls(description.num_inputs, description.num_outputs, mask)
    network.initialize(description)
    return network

  @classmethod
  def from_json(cls, json_content, n_in, n_out, mask=None, target = 'classes'):
    """
    :type json_content: str
    :type n_in: int
    :type n_out: int
    :type mask: str
    :rtype: LayerNetwork
    """
    network = cls(n_in, n_out, mask)
    if mask is None: mask = "none"
    try:
      topology = json.loads(json_content)
    except ValueError:
      print >> log.v4, "----- BEGIN JSON CONTENT -----"
      print >> log.v4, json_content
      print >> log.v4, "------ END JSON CONTENT ------"
      assert False, "invalid json content"
    if hasattr(LstmLayer, 'sharpgates'):
      del LstmLayer.sharpgates
    network.L1 = T.constant(0)
    network.L2 = T.constant(0)
    network.recurrent = False
    def traverse(content, layer_name, network):
      source = []
      obj = content[layer_name].copy()
      act = obj.pop('activation', 'logistic')
      cl = obj.pop('class', None)
      if not 'from' in obj:
        source = [SourceLayer(network.n_in, network.x, name = 'data')]
      else:
        for prev in obj['from']:
          if prev != "null":
            if not network.hidden.has_key(prev):
              traverse(content, prev, network)
            source.append(network.hidden[prev])
      if 'encoder' in obj:
        if not network.hidden.has_key(obj['encoder']):
          traverse(content, obj['encoder'], network)
        obj['encoder'] = network.hidden[obj['encoder']]
      obj.pop('from', None)
      params = { 'sources': source, 'dropout' : 0.0, 'name' : 'output', 'mask' : mask }
      params.update(obj)
      if cl == 'softmax':
        if not 'target' in params:
          params['target'] = target
        network.make_classifier(name = layer_name, sources=params['sources'], target=params['target'], loss=params['loss'],
                                dropout=params['dropout'], mask=mask)
      else:
        layer_class = get_layer_class(cl)
        params.update({'activation': act, 'name': layer_name})
        if layer_class.recurrent:
          network.recurrent = True
          params['index'] = network.i
        network.add_layer(layer_class(**params))
    for layer_name in topology:
      if layer_name == 'output' or 'target' in topology[layer_name]:
        traverse(topology, layer_name, network)
    return network

  @classmethod
  def from_hdf_model_topology(cls, model, mask="unity", target = 'classes'):
    """
    :type model: h5py.File
    :param str mask: e.g. "unity"
    :rtype: LayerNetwork
    """
    grp = model['training']
    if mask is None: mask = grp.attrs['mask']
    try:
      network = cls(model.attrs['n_in'], model['n_out'].attrs, mask)
    except:
      network = cls(model.attrs['n_in'], {'classes':model.attrs['n_out']}, mask)
    network.L1 = T.constant(0)
    network.L2 = T.constant(0)
    network.recurrent = False
    def traverse(model, layer_name, network):
      if 'from' in model[layer_name].attrs and model[layer_name].attrs['from'] != 'data':
        x_in = []
        for s in model[layer_name].attrs['from'].split(','):
          if s == 'data':
            x_in.append(SourceLayer(network.n_in, network.x, name = 'data'))
          elif s != "null":
            if not network.hidden.has_key(s):
              traverse(model, s, network)
            x_in.append(network.hidden[s])
      else:
        x_in = [ SourceLayer(network.n_in, network.x, name = 'data') ]
      if 'encoder' in model[layer_name].attrs:
        if not network.hidden.has_key(model[layer_name].attrs['encoder']):
          traverse(model, model[layer_name].attrs['encoder'], network)
      cl = model[layer_name].attrs['class']
      if cl == 'softmax':
        if not 'target' in model[layer_name].attrs:
          targ = target
        else:
          targ = model[layer_name].attrs['target']
        loss = 'ce' if not 'loss' in model[layer_name].attrs else model[layer_name].attrs['loss']
        network.make_classifier(name = layer_name, target=targ, sources=x_in, loss=loss,
                                dropout=model[layer_name].attrs['dropout'], mask=mask)
      else:
        act = model[layer_name].attrs['activation']
        params = { 'sources': x_in,
                   'n_out': model[layer_name].attrs['n_out'],
                   'activation': act,
                   'dropout': model[layer_name].attrs['dropout'],
                   'name': layer_name,
                   'mask': model[layer_name].attrs['mask'] }
        layer_class = get_layer_class(cl)
        if layer_class.recurrent:
          network.recurrent = True
          params['index'] = network.i
          for p in ['truncation', 'projection', 'reverse', 'sharpgates', 'sampling']:
            if p in model[layer_name].attrs:
              params[p] = model[layer_name].attrs[p]
          if 'encoder' in model[layer_name].attrs:
            params['encoder'] = network.hidden[model[layer_name].attrs['encoder']]
        network.add_layer(layer_class(**params))
    for layer_name in model:
      if layer_name == model.attrs['output'] or 'target' in model[layer_name].attrs:
        traverse(model, layer_name, network)
    return network

  def add_layer(self, layer):
    """
    :type layer: NetworkHiddenLayer.HiddenLayer
    """
    assert layer.name
    self.hidden[layer.name] = layer
    if isinstance(layer, RecurrentLayer):
      if layer.attrs['L1'] > 0.0: self.L1 += layer.attrs['L1'] * abs(layer.W_re.sum())
      if layer.attrs['L2'] > 0.0: self.L2 += layer.attrs['L2'] * (layer.W_re ** 2).sum()
    for W in layer.W_in:
      if layer.attrs['L1'] > 0.0: self.L1 += layer.attrs['L1'] * abs(W.sum())
      if layer.attrs['L2'] > 0.0: self.L2 += layer.attrs['L2'] * (W ** 2).sum()

  def make_classifier(self, name, target, **kwargs):
    """
    :param list[NetworkLayer.Layer] sources: source layers
    :param str loss: loss type, "ce", "ctc" etc
    """
    self.loss = kwargs["loss"]
    if self.loss in ('ctc', 'ce_ctc', 'sprint', 'sprint_smoothed'):
      layer_class = SequenceOutputLayer
    else:
      layer_class = FramewiseOutputLayer
    self.output[name] = layer_class(index=self.i, n_out=self.n_out[target], name=name, target=target, **kwargs)
    for W in self.output[name].W_in:
      if self.output[name].attrs['L1'] > 0.0: self.L1 += self.output[name].attrs['L1'] * abs(W.sum())
      if self.output[name].attrs['L2'] > 0.0: self.L2 += self.output[name].attrs['L2'] * (W ** 2).sum()
    self.declare_train_params()
    if not target in self.y:
      self.y[target] = T.ivector('y')
    targets = self.c if self.loss == 'ctc' else self.y[target]
    error_targets = self.c if self.loss in ('ctc','ce_ctc') else self.y[target]
    self.errors[target] = self.output[name].errors(error_targets)
    cost = self.output[name].cost(targets)
    self.cost[target], self.known_grads = cost[:2]
    if len(cost) > 2:
      self.ctc_priors = cost[2]
      assert self.ctc_priors is not None
    else:
      self.ctc_priors = None
    self.objective[target] = self.cost[target] + self.L1 + self.L2 #+ entropy * self.output.entropy()
    #if hasattr(LstmLayer, 'sharpgates'):
      #self.objective += entropy * (LstmLayer.sharpgates ** 2).sum()
    #self.jacobian = T.jacobian(self.output.z, self.x)

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

  def _make_layer(self, description, info, sources, reverse=False):
    """
    :type description: LayerNetworkDescription
    :type info: dict[str]
    :type sources: list[SourceLayer]
    :type reverse: bool
    :rtype: NetworkHiddenLayer.ForwardLayer | NetworkRecurrentLayer.RecurrentLayer
    """
    params = dict(description.default_layer_info)
    params.update(info)
    params["sources"] = sources
    params["mask"] = self.mask
    layer_class = get_layer_class(params["layer_class"])
    if layer_class.recurrent:
      self.recurrent = True
      params['index'] = self.i
      params['truncation'] = description.truncation
      if self.bidirectional:
        if not reverse:
          params['name'] += "_fw"
        else:
          params['name'] += "_bw"
          params['reverse'] = True
      if 'sharpgates' in inspect.getargspec(layer_class.__init__).args[1:]:
        params['sharpgates'] = description.sharpgates
    layer = layer_class(**params)
    self.add_layer(layer=layer)
    return layer

  def _make_output(self, description, sources):
    """
    :type description: LayerNetworkDescription
    :type sources: list[SourceLayer]
    """
    params = dict(description.default_layer_info)
    params.pop("layer_class", None)  # Makes no sense to use this default.
    params.update(description.output_info)
    params["sources"] = sources
    params["mask"] = self.mask
    self.make_classifier(**params)

  def initialize(self, description):
    """
    :type description: LayerNetworkDescription

    Initializes the network based the description.
    """
    assert description.num_inputs == self.n_in
    #assert description.num_outputs == self.n_out
    assert len(description.hidden_info) > 0
    self.description = description
    n_in = self.n_in
    x_in = self.x
    self.L1 = T.constant(0)
    self.L2 = T.constant(0)
    self.recurrent = False
    self.bidirectional = description.bidirectional
    if hasattr(LstmLayer, 'sharpgates'):
      del LstmLayer.sharpgates
    last_layer = None
    # create forward layers
    for info in description.hidden_info:
      last_layer = self._make_layer(description=description, info=info,
                                    sources=[SourceLayer(n_out=n_in, x_out=x_in, name='')])
      n_in = last_layer.attrs['n_out']
      x_in = last_layer.output
    sources = [last_layer]
    # create backward layers
    assert self.recurrent or not self.bidirectional, "non-recurrent networks can not be bidirectional"
    if self.bidirectional:
      last_layer = None
      n_in = self.n_in
      x_in = self.x
      for info in description.hidden_info:
        last_layer = self._make_layer(description=description, info=info,
                                      sources=[SourceLayer(n_out=n_in, x_out=x_in, name='')],
                                      reverse=True)
        n_in = last_layer.attrs['n_out']
        x_in = last_layer.output
      sources += [last_layer]
    self._make_output(description=description, sources=sources)

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
      out.attrs[k] = self.n_out[k]
    for h in self.hidden:
      self.hidden[h].save(model)
    for k in self.output:
      self.output[k].save(model)

  def to_json(self):
    out = {}
    for name in self.output:
      outattrs = self.output[name].attrs.copy()
      outattrs['from'] = outattrs['from'].split(',')
      out[name] = outattrs
    for h in self.hidden.keys():
      out[h] = self.hidden[h].to_json()
    return str(out)

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

