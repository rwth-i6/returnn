#! /usr/bin/python2.7

from __future__ import print_function

import h5py
import theano
from theano import tensor as T

from returnn.network_description import LayerNetworkDescription
from returnn.theano.layers.base import Layer, SourceLayer
from returnn.theano.layers.basic import get_layer_class
from returnn.theano.layers.lstm import *
from returnn.theano.layers.output import OutputLayer, FramewiseOutputLayer, SequenceOutputLayer, DecoderOutputLayer, UnsupervisedOutputLayer
from returnn.util.basic import dict_joined, as_str
from returnn.log import log


class LayerNetwork(object):
  def __init__(self, n_in=None, n_out=None,
               base_network=None, data_map=None, data_map_i=None,
               shared_params_network=None,
               mask=None, sparse_input=False, target='classes', train_flag=False, eval_flag=False):
    """
    :param int n_in: input dim of the network
    :param dict[str,(int,int)] n_out: output dim of the network.
      first int is num classes, second int is 1 if it is sparse, i.e. we will get the indices.
    :param dict[str,theano.Variable] data_map: if specified, this will be used for x/y (and it expects data_map_i)
    :param dict[str,theano.Variable] data_map_i: if specified, this will be used for i/j
    :param LayerNetwork|None base_network: optional base network where we will derive x/y/i/j/n_in/n_out from.
      data_map will have precedence over base_network.
    :param LayerNetwork|()->LayerNetwork|None shared_params_network: optional network where we will share params with.
      we will error if there is a param which cannot be shared.
    :param str mask: e.g. "unity" or None ("dropout")
    :param bool sparse_input: for SourceLayer
    :param str target: default target
    :param bool train_flag: marks that we are used for training
    :param bool eval_flag: marks that we are used for evaluation
    """
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
    if data_map is not None:
      assert data_map_i is not None
      self.y = data_map
      self.x = data_map["data"]
      self.j = data_map_i
      self.i = data_map_i["data"]
    elif base_network is not None:
      self.x = base_network.x
      self.y = base_network.y
      self.i = base_network.i
      self.j = base_network.j
    else:
      dtype = "float32" if data_dim >= 3 else "int32"
      self.x = T.TensorType(dtype, ((False,) * data_dim))('x')
      self.y = {"data": self.x}
      self.i = T.bmatrix('i'); """ :type: theano.Variable """
      self.j = {"data": self.i}
    if base_network is not None:
      self.epoch = base_network.epoch
      self.tags  = base_network.tags
    else:
      self.epoch = T.constant(0, name="epoch", dtype="int32")
      self.tags  = T.bmatrix('tags')
    self.constraints = {}
    self.total_constraints = T.constant(0)
    Layer.initialize_rng()
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
    :type config: returnn.config.Config
    :param str mask: e.g. "unity" or None ("dropout"). "unity" is for testing.
    :rtype: LayerNetwork
    """
    json_content = cls.json_from_config(config, mask=mask)
    from returnn.pretrain import find_pretrain_wrap_values, pretrain_from_config
    if find_pretrain_wrap_values(json_content):
      pretrain = pretrain_from_config(config=config)
      assert pretrain, "found Pretrain WrapEpochValue but no pretrain configured"
      json_content = pretrain.get_final_network_json()
    return cls.from_json_and_config(json_content, config, mask=mask, **kwargs)

  @classmethod
  def json_from_config(cls, config, mask=None):
    """
    :type config: returnn.config.Config
    :param str mask: "unity", "none" or "dropout"
    :rtype: dict[str]
    """
    from returnn.config import network_json_from_config
    return network_json_from_config(config=config, mask=mask)

  @classmethod
  def from_description(cls, description, mask=None, **kwargs):
    """
    :type description: NetworkDescription.LayerNetworkDescription
    :param str mask: e.g. "unity" or None ("dropout")
    :rtype: LayerNetwork
    """
    json_content = description.to_json_content(mask=mask)
    network = cls.from_json(
      json_content, n_in=description.num_inputs, n_out=description.num_outputs, mask=mask, **kwargs)
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
    :type config: returnn.config.Config
    :type json_content: str | dict
    :rtype: LayerNetwork
    """
    network = cls.from_json(json_content, **dict_joined(kwargs, cls.init_args_from_config(config)))
    network.recurrent = network.recurrent or config.bool('recurrent', False)
    return network

  def get_layer_param(self, layer_name, param_name, param):
    """
    Used by Container.add_param() to maybe substitute a parameter instead of creating a new shared var.
    :param str layer_name: the layer name where this param will be added
    :param str param_name: the name of the param
    :param theano.SharedVariable param: the already created shared var
    :rtype None | theano.Variable
    If we return None, Container.add_param() will continue as usual.
    """
    if self.shared_params_network:
      network = self.shared_params_network
      if callable(network):
        network = network()
      base_substitute = network.get_layer_param(layer_name=layer_name, param_name=param_name, param=param)
      if base_substitute: return base_substitute
      base_layer = network.get_layer(layer_name)
      assert base_layer, "%s not found in shared_params_network" % layer_name
      return base_layer.params.get(param_name, None)
    return None

  @classmethod
  def from_base_network(cls, base_network, json_content=None, share_params=False, base_as_calc_step=False, **kwargs):
    """
    :param LayerNetwork base_network: base network to derive from
    :param dict[str]|None json_content: JSON content for subnetwork. if None, will use from base network
    :param bool share_params: will use the same params as the base network
    :param bool base_as_calc_step: base is calc step 0. see below
    :param dict[str] kwargs: kwargs for __init__
    :rtype: LayerNetwork
    """
    if "n_out" in kwargs and "n_in" not in kwargs:
      kwargs["n_in"] = None
    network = cls(
      base_network=base_network,
      shared_params_network=base_network if share_params else None,
      **dict_joined(base_network.init_args(), kwargs))
    if base_as_calc_step:
      network.calc_step_base = base_network  # used by CalcStepLayer. see also get_calc_step()
    if json_content is None:
      json_content = base_network.to_json_content()
    cls.from_json(json_content, network=network)
    if share_params:
      trainable_params = network.get_all_params_vars()
      assert len(trainable_params) == 0
    return network

  def get_calc_step(self, i):
    """
    :param int i: calc step, 0 to n
    :rtype: LayerNetwork
    Used by CalcStepLayer. Will automatically create the requested calc step.
    Calc step 0 is the base network (calc_step_base).
    """
    if self.calc_step_base:
      return self.calc_step_base.get_calc_step(i)  # go up to the main network
    if i == 0: return self
    if i <= len(self.calc_steps):
      return self.calc_steps[i - 1]
    print("creating calc steps up to %i" % i, file=log.v4)
    while i > len(self.calc_steps):
      base_network = self
      if self.calc_steps: base_network = self.calc_steps[-1]
      subnetwork = self.from_base_network(
        base_network=base_network, share_params=True, base_as_calc_step=True)
      self.calc_steps += [subnetwork]
    return self.calc_steps[i - 1]

  def new_subnetwork(self, json_content, n_out, data_map, data_map_i):
    """
    :param dict[str,dict] json_content: subnetwork specification
    :param dict[str,list[int,int]] n_out: n_out info for subnetwork
    :param dict[str,theano.Variable] data_map: data
    :param dict[str,theano.Variable] data_map_i: indices for data
    :rtype: LayerNetwork
    The data input for the subnetwork is not derived from ourselves but specified
    explicitly through n_out & data_map.
    """
    return self.from_base_network(self, json_content=json_content,
                                  n_out=n_out, data_map=data_map, data_map_i=data_map_i)

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
    if hasattr(LstmLayer, 'sharpgates'):
      del LstmLayer.sharpgates
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
    if target == 'sizes' and not 'sizes' in self.n_out: #TODO(voigtlaender): fix data please
      self.n_out['sizes'] = [2,1]
    if self.base_network:
      self.base_network.use_target(target=target, dtype=dtype)
      if not self.y is self.base_network.y:
        self.y[target] = self.base_network.y[target]
      if not self.j is self.base_network.j:
        self.j[target] = self.base_network.j[target]
      if target not in self.n_out:
        self.n_out[target] = self.base_network.n_out[target]
      return
    if target.endswith("[sparse:coo]"):
      tprefix = target[:target.index("[")]
      ndim = self.n_out[target][1]  # expected (without batch), e.g. 2 if like (time,feature)
      # For each coordinate axe. Also with batch-dim.
      for i in range(ndim):
        self.y["%s[sparse:coo:%i:%i]" % (tprefix, ndim, i)] = T.TensorType("int32", (False,) * 2)('y_%s[sparse:coo:%i:%i]' % (tprefix, ndim, i))
      # And the data itself. Also with batch-dim.
      self.y["%s[sparse:coo:%i:%i]" % (tprefix, ndim, ndim)] = \
        T.TensorType(dtype, (False,) * 2)("y_%s[%i]" % (tprefix, ndim))
      # self.j will be used to get the list of keys we need to get from the dataset.
      for i in range(ndim + 1):
        self.j.setdefault("%s[sparse:coo:%i:%i]" % (tprefix, ndim, i), T.bmatrix('j_%s[sparse:coo:%i:%i]' % (tprefix, ndim, i)))
      # self.y[target] will be given to the OutputLayer.
      self.y[target] = tuple(self.y["%s[sparse:coo:%i:%i]" % (tprefix, ndim, i)] for i in range(ndim + 1))
      self.j[target] = self.j["data"]  # Not sure if this is the best we can do...
      return
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
        from returnn.theano.util import time_batch_make_flat
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
    if len(cost) > 2:
      if self.ctc_priors:
        print("multiple ctc_priors, second one from layer %s" % layer.name, file=log.v3)
      else:
        self.ctc_priors = cost[2]
      assert self.ctc_priors is not None

  def make_classifier(self, name='output', target='classes', **kwargs):
    """
    :param list[NetworkBaseLayer.Layer] sources: source layers
    :param str loss: loss type, "ce", "ctc" etc
    """
    if not "loss" in kwargs: kwargs["loss"] = "ce"
    self.loss = kwargs["loss"]
    if self.loss in ('ctc', 'ce_ctc', 'hmm', 'ctc2', 'sprint', 'viterbi', 'fast_bw', 'seg_fast_bw', 'lf_mmi', 'ctc_warp', 'inv', "ctc_rasr"):
      layer_class = SequenceOutputLayer
      # We must keep sequences as they are. Setting us as recurrent
      # will tell other code to leave seqs as they are (e.g. the dataset batch building).
      self.recurrent = True
    elif self.loss == 'decode':
      layer_class = DecoderOutputLayer
    elif self.loss == 'unsupervised':
      layer_class = UnsupervisedOutputLayer
    else:
      layer_class = FramewiseOutputLayer

    dtype = kwargs.pop('dtype', 'int32')
    if target != "null" and target not in self.y:
      self.use_target(target, dtype=dtype)
    if target != "null":
      targets = self.y[target]
    else:
      targets = None
    if self.loss == "ctc" and not '__final' in self.n_out:
      self.n_out[target][0] += 1
    elif self.loss == "hmm":
      self.n_out[target][0] = 2 * self.n_out[target][0] - 1  # silence has only 1 state
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

  def get_params_shared_flat_dict(self):
    """
    :rtype: dict[str,theano.shared]
    This will collect all vars of all layers in one dict.
    We extend the param name with our custom scheme.
    """
    params = {}
    for l_name, layer in list(self.output.items()) + list(self.hidden.items()):
      for p_name, param in layer.params.items():
        p_name = "%s.%s" % (l_name, p_name)
        assert p_name not in params
        params[p_name] = param
    return params

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
