
"""
Provides :class:`LayerNetworkDescription`.
"""

from __future__ import print_function

from Util import simple_obj_repr, hdf5_dimension, hdf5_group, hdf5_shape
from Log import log


class LayerNetworkDescription:

  """
  This class is used as a description to build up the LayerNetwork.
  The other options to build up a LayerNetwork are JSON or from a HDF model.
  """

  def __init__(self, num_inputs, num_outputs,
               hidden_info,
               output_info,
               default_layer_info,
               bidirectional=True, sharpgates='none',
               truncation=-1, entropy=0):
    """
    :type num_inputs: int
    :type num_outputs: dict[str,(int,int)]
    :param list[dict[str]] hidden_info: list of
      (layer_type, size, activation, name)
    :type output_info: dict[str]
    :type default_layer_info: dict[str]
    :type bidirectional: bool
    :param str sharpgates: see LSTM layers
    :param int truncation: number of steps to use in truncated BPTT or -1. see theano.scan
    :param float entropy: ...
    """
    self.num_inputs = num_inputs
    self.num_outputs = num_outputs
    self.hidden_info = list(hidden_info)
    self.output_info = output_info
    self.default_layer_info = default_layer_info
    self.bidirectional = bidirectional
    self.sharpgates = sharpgates
    self.truncation = truncation
    self.entropy = entropy

  def __eq__(self, other):
    return self.init_args() == getattr(other, "init_args", lambda: {})()

  def __ne__(self, other):
    return not self == other

  def init_args(self):
    import inspect
    return {arg: getattr(self, arg) for arg in inspect.getargspec(self.__init__).args[1:]}

  __repr__ = simple_obj_repr

  def copy(self):
    args = self.init_args()
    return self.__class__(**args)

  @classmethod
  def from_config(cls, config):
    """
    :type config: Config.Config
    :rtype: LayerNetworkDescription
    """
    num_inputs, num_outputs = cls.num_inputs_outputs_from_config(config)
    loss = cls.loss_from_config(config)
    hidden_size = config.int_list('hidden_size')
    assert len(hidden_size) > 0, "no hidden layers specified"
    hidden_type = config.list('hidden_type')
    assert len(hidden_type) <= len(hidden_size), "too many hidden layer types"
    hidden_name = config.list('hidden_name')
    assert len(hidden_name) <= len(hidden_size), "too many hidden layer names"
    if len(hidden_type) != len(hidden_size):
      n_hidden_type = len(hidden_type)
      for i in range(len(hidden_size) - len(hidden_type)):
        if n_hidden_type == 1:
          hidden_type.append(hidden_type[0])
        else:
          hidden_type.append("forward")
    if len(hidden_name) != len(hidden_size):
      for i in range(len(hidden_size) - len(hidden_name)):
        hidden_name.append("_")
    for i, name in enumerate(hidden_name):
      if name == "_": hidden_name[i] = "hidden_%d" % i
    L1_reg = config.float('L1_reg', 0.0)
    L2_reg = config.float('L2_reg', 0.0)
    bidirectional = config.bool('bidirectional', True)
    truncation = config.int('truncation', -1)
    actfct = config.list('activation')
    assert actfct, "need some activation function"
    dropout = config.list('dropout', [0.0])
    sharpgates = config.value('sharpgates', 'none')
    entropy = config.float('entropy', 0.0)
    if len(actfct) < len(hidden_size):
      for i in range(len(hidden_size) - len(actfct)):
        actfct.append(actfct[-1])
    if len(dropout) < len(hidden_size) + 1:
      assert len(dropout) > 0
      for i in range(len(hidden_size) + 1 - len(dropout)):
        dropout.append(dropout[-1])
    dropout = [float(d) for d in dropout]
    hidden_info = []; """ :type: list[dict[str]] """
    for i in range(len(hidden_size)):
      hidden_info.append({
        "layer_class": hidden_type[i],  # e.g. 'forward'
        "n_out": hidden_size[i],
        "activation": actfct[i],  # activation function, e.g. "tanh". see strtoact().
        "name": hidden_name[i],  # custom name of the hidden layer, such as "hidden_2"
        "dropout": dropout[i]
      })
    output_info = {"loss": loss, "dropout": dropout[-1]}
    default_layer_info = {
      "L1": L1_reg, "L2": L2_reg,
      "forward_weights_init": config.value("forward_weights_init", None),
      "bias_init": config.value("bias_init", None),
      "substitute_param_expr": config.value("substitute_param_expr", None)
    }

    return cls(num_inputs=num_inputs, num_outputs=num_outputs,
               hidden_info=hidden_info,
               output_info=output_info,
               default_layer_info=default_layer_info,
               bidirectional=bidirectional, sharpgates=sharpgates,
               truncation=truncation, entropy=entropy)

  @classmethod
  def loss_from_config(cls, config):
    """
    :type config: Config.Config
    :rtype: str
    """
    return config.value('loss', 'ce')

  @classmethod
  def tf_extern_data_types_from_config(cls, config):
    """
    :param Config.Config config:
    :return: dict data_key -> kwargs of Data
    :rtype: dict[str,dict[str]]
    """
    input_data_key = config.value('default_input', 'data')
    if config.has("extern_data"):
      data_dims = config.typed_dict["extern_data"]
      assert isinstance(data_dims, dict), "extern_data in config must be a dict"
      if config.has("num_inputs") or config.has("num_outputs"):
        print("Warning: Using extern_data and will ignore num_inputs/num_outputs in config.", file=log.v2)
    else:
      num_inputs, num_outputs = cls.num_inputs_outputs_from_config(config)
      data_dims = num_outputs.copy()
      sparse_input = config.bool("sparse_input", False)
      data_dims.setdefault(input_data_key, (num_inputs, 1 if sparse_input else 2))
    data = {}
    for key, data_type in data_dims.items():
      if isinstance(data_type, dict):
        data[key] = data_type
        continue
      assert isinstance(data_type, (list, tuple))
      dim, ndim = data_type
      init_args = {"dim": dim}
      if ndim == 1:
        init_args["shape"] = (None,)
        init_args["sparse"] = True
      elif ndim == 2:
        init_args["shape"] = (None, dim)
      else:
        assert ndim >= 3
        init_args["shape"] = (None,) * (ndim - 1) + (dim,)
      # In Returnn with Theano, we usually have the shape (time,batch,feature).
      # In TensorFlow, the default is (batch,time,feature).
      # This is also what we use here, i.e.:
      # batch_dim_axis=0, time_dim_axis=1. See TFEngine.DataProvider._get_next_batch().
      data[key] = init_args
    for key, v in data.items():
      if key == input_data_key:
        v.setdefault("available_for_inference", True)
      else:
        v.setdefault("available_for_inference", False)
    return data

  @classmethod
  def num_inputs_outputs_from_config(cls, config):
    """
    :type config: Config.Config
    :returns (num_inputs, num_outputs),
       where num_inputs is like num_outputs["data"][0],
       and num_outputs is a dict of data_key -> (dim, ndim),
         where data_key is e.g. "classes" or "data",
         dim is the feature dimension or the number of classes,
         and ndim is the ndim counted without batch-dim,
         i.e. ndim=1 means usually sparse data and ndim=2 means dense data.
    :rtype: (int,dict[str,(int,int)])
    """
    from Util import BackendEngine
    num_inputs = config.int('num_inputs', 0)
    target = config.value('target', 'classes')
    if config.is_typed('num_outputs'):
      num_outputs = config.typed_value('num_outputs')
      if not isinstance(num_outputs, dict):
        num_outputs = {target: num_outputs}
      num_outputs = num_outputs.copy()
      from Dataset import convert_data_dims
      num_outputs = convert_data_dims(num_outputs, leave_dict_as_is=BackendEngine.is_tensorflow_selected())
      if "data" in num_outputs:
        num_inputs = num_outputs["data"]
        if isinstance(num_inputs, (list, tuple)):
          num_inputs = num_inputs[0]
        elif isinstance(num_inputs, dict):
          if "dim" in num_inputs:
            num_inputs = num_inputs["dim"]
          else:
            num_inputs = num_inputs["shape"][-1]
        else:
          raise TypeError("data key %r" % num_inputs)
    elif config.has('num_outputs'):
      num_outputs = {target: [config.int('num_outputs', 0), 1]}
    else:
      num_outputs = None
    dataset = None
    if config.list('train') and ":" not in config.value('train', ''):
      dataset = config.list('train')[0]
    if not config.is_typed('num_outputs') and dataset:
      # noinspection PyBroadException
      try:
        _num_inputs = hdf5_dimension(dataset, 'inputCodeSize') * config.int('window', 1)
      except Exception:
        _num_inputs = hdf5_dimension(dataset, 'inputPattSize') * config.int('window', 1)
      # noinspection PyBroadException
      try:
        _num_outputs = {target: [hdf5_dimension(dataset, 'numLabels'), 1]}
      except Exception:
        _num_outputs = hdf5_group(dataset, 'targets/size')
        for k in _num_outputs:
          _num_outputs[k] = [_num_outputs[k], len(hdf5_shape(dataset, 'targets/data/' + k))]
      if num_inputs:
        assert num_inputs == _num_inputs
      if num_outputs:
        assert num_outputs == _num_outputs
      num_inputs = _num_inputs
      num_outputs = _num_outputs
    if not num_inputs and not num_outputs and config.has("load") and BackendEngine.is_theano_selected():
      from Network import LayerNetwork
      import h5py
      model = h5py.File(config.value("load", ""), "r")
      # noinspection PyProtectedMember
      num_inputs, num_outputs = LayerNetwork._n_in_out_from_hdf_model(model)
    assert num_inputs and num_outputs, "provide num_inputs/num_outputs directly or via train"
    return num_inputs, num_outputs

  @classmethod
  def _layer_param_to_json(cls, params):
    """
    :type params: dict[str]
    :rtype: dict[str]

    Some params are named differently in JSON than the real kwargs.
    Some are also obsolete.
    """
    if "name" in params:
      del params["name"]
    if "layer_class" in params:
      params["class"] = params["layer_class"]
      del params["layer_class"]
    for key, value in list(params.items()):
      if value is None:
        del params[key]
    return params

  def _layer_params(self, info, sources, mask, reverse=False):
    """
    :param dict[str] info: self.hidden_info[i]
    :param list[str] sources: 'from' entry
    :param None | str mask: mask
    :param bool reverse: reverse or not
    :rtype: dict[str]
    """
    import Util
    if Util.BackendEngine.is_theano_selected():
      from NetworkLayer import get_layer_class
    elif Util.BackendEngine.is_tensorflow_selected():
      from TFNetworkLayer import get_layer_class
    else:
      raise NotImplementedError
    params = dict(self.default_layer_info)
    params.update(info)
    params["from"] = sources
    if mask:
      params["mask"] = mask
    layer_class = get_layer_class(params["layer_class"])
    if layer_class.recurrent:
      params['truncation'] = self.truncation
      if self.bidirectional:
        if not reverse:
          params['name'] += "_fw"
        else:
          params['name'] += "_bw"
          params['reverse'] = True
      if 'sharpgates' in Util.getargspec(layer_class.__init__).args[1:]:
        params['sharpgates'] = self.sharpgates
    return params

  def _output_to_json(self, mask, sources):
    """
    :param list[str] sources: 'from' entry
    :param None | str mask: mask
    :rtype: dict[str]
    """
    params = dict(self.default_layer_info)
    params.pop("layer_class", None)  # Makes no sense to use this default.
    params.update(self.output_info)
    params["from"] = sources
    if mask:
      params["mask"] = mask
    params["class"] = "softmax"
    return self._layer_param_to_json(params)

  def to_json_content(self, mask=None):
    """
    :param None | str mask: mask
    :rtype: dict
    """
    content = {}

    # create forward layers
    last_source = "data"
    for info in self.hidden_info:
      layer = self._layer_params(info=info, mask=mask, sources=[last_source])
      layer_name = layer["name"]
      content[layer_name] = self._layer_param_to_json(layer)
      last_source = layer_name
    sources = [last_source]

    if self.bidirectional:
      # create backward layers
      last_source = "data"
      for info in self.hidden_info:
        layer = self._layer_params(info=info, mask=mask, sources=[last_source], reverse=True)
        layer_name = layer["name"]
        content[layer_name] = self._layer_param_to_json(layer)
        last_source = layer_name
      sources += [last_source]

    output = self._output_to_json(sources=sources, mask=mask)
    content["output"] = output
    return content
