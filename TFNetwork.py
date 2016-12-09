
import tensorflow as tf
import numpy
from Log import log
from TFNetworkLayer import Data, LayerBase, get_layer_class


class ExternData(object):

  def __init__(self, data=None, default_input="data", default_target="classes"):
    """
    :param None|dict[str,dict[str]] data: optional init kwargs for Data
    """
    self.data = {}  # type: dict[str,Data]
    self.default_input = default_input
    self.default_target = default_target
    if data:
      self.register_data_from_dict(data)

  def init_from_config(self, config):
    """
    :param Config.Config config:
    """
    from NetworkDescription import LayerNetworkDescription
    num_inputs, num_outputs = LayerNetworkDescription.num_inputs_outputs_from_config(config)
    data_dims = num_outputs.copy()
    data_dims.setdefault("data", (num_inputs, 2))
    sparse_input = config.bool("sparse_input", False)
    for key, (dim, ndim) in data_dims.items():
      init_args = {"name": key, "dim": dim}
      if ndim == 1:
        init_args["shape"] = (None,)
        init_args["sparse"] = True
      elif ndim == 2:
        init_args["shape"] = (None, dim)
      else:
        assert ndim >= 3
        init_args["shape"] = (None,) * (ndim - 1) + (dim,)
      if key == "data":
        init_args["sparse"] = sparse_input
      self.data[key] = Data(**init_args)
    self.default_target = config.value('target', 'classes')

  def register_data_from_dict(self, data):
    """
    :param dict[str,dict[str]] data: init kwargs for Data
    """
    for key, value in data.items():
      init_args = value.copy()
      init_args["name"] = key
      self.data[key] = Data(**init_args)

  def register_data(self, data):
    """
    :param ExternData.Data data:
    """
    assert data.name not in self.data
    self.data[data.name] = data

  def get_data(self, name):
    return self.data[name]

  def get_default_input_data(self):
    return self.data[self.default_input]

  def get_default_target_data(self):
    return self.data[self.default_target]

  def get_data_description(self):
    return ", ".join(["%s: %s" % (name, self.data[name].get_description(with_name=False))
                      for name in self.data.keys()])


class TFNetwork(object):
  def __init__(self, config=None, extern_data=None, rnd_seed=42):
    """
    :param Config.Config config:
    :param ExternData|None extern_data:
    :param int rnd_seed:
    """
    self.config = config
    if extern_data is None:
      extern_data = ExternData()
      if not config:
        from Config import get_global_config
        config = get_global_config()
      extern_data.init_from_config(config)
    self.extern_data = extern_data
    self.rnd_seed = rnd_seed
    self.layers_desc = {}  # type: dict[str,dict[str]]
    self.layers = {}  # type: dict[str,LayerBase]

  def construct_from(self, list_or_dict):
    """
    :param list[dict[str]] | dict[str,dict[str]] list_or_dict:
    """
    if isinstance(list_or_dict, (tuple, list)):
      self.construct_from_list(list_or_dict)
    elif isinstance(list_or_dict, dict):
      self.construct_from_dict(list_or_dict)
    else:
      raise Exception("unsupported: %r (type %r)" % (list_or_dict, type(list_or_dict)))

  def construct_from_list(self, net_list):
    """
    :param list[dict[str]] net_list: list of layer descriptions
    """
    net_dict = {}  # type: dict[str,dict[str]]
    for i, layer_desc in enumerate(net_list):
      layer_desc = layer_desc.copy()
      name = layer_desc.pop("name", None)
      if not name:
        if i == len(net_list) - 1:
          name = "output"
        else:
          name = "layer%i" % i
      if i == len(net_list) - 1:
        if "target" not in layer_desc:
          layer_desc["target"] = self.extern_data.default_target
      net_dict[name] = layer_desc
    self.construct_from_dict(net_dict)

  def construct_from_dict(self, net_dict):
    """
    :param dict[str,dict[str]] net_dict:
    """
    def _construct_layer(name):
      """
      :param str name:
      :param dict[str] layer_desc:
      """
      if name in self.layers:
        return self.layers[name]
      if name not in net_dict:
        if name == "data":
          layer_desc = {"class": "source", "from": []}
        else:
          raise Exception("layer not found: %r" % name)
      else:
        layer_desc = net_dict[name]
      self.layers_desc[name] = layer_desc
      layer_desc = layer_desc.copy()
      class_name = layer_desc.pop("class")
      layer_class = get_layer_class(class_name)
      layer_desc["name"] = name
      layer_desc["network"] = self
      src_names = layer_desc.pop("from", ["data"])
      if not isinstance(src_names, (list, tuple)):
        src_names = [src_names]
      layer_desc["sources"] = [
        _construct_layer(src_name)
        for src_name in src_names
        if not src_name == "none"]
      layer = layer_class(**layer_desc)
      self.layers[name] = layer
      return layer

    for name, layer_desc in sorted(net_dict.items()):
      if name == "output" or "target" in layer_desc or layer_desc.get("class") == "softmax":
        _construct_layer(name)

  def get_params(self):
    """
    :return: list of variables
    :rtype: list[tf.Variable]
    """
    l = []  # type: list[tf.Variable]
    for layer_name, layer in sorted(self.layers.items()):
      for param_name, param in sorted(layer.params.items()):
        assert isinstance(param, tf.Variable)
        l.append(param)
    return l

  def get_trainable_params(self):
    return self.get_params()  # at the moment just the same

  def get_num_params(self):
    num_params = 0
    params = self.get_params()
    for param in params:
      shape = param.get_shape().as_list()
      num_params += numpy.prod(shape)
    return num_params

  def load_params_from_file(self, filename):
    pass  # TODO...

  def print_network_info(self, name="Network"):
    print >> log.v2, "%s layer topology:" % name
    print >> log.v2, "  extern data #:", self.extern_data.get_data_description()
    for layer_name, layer in sorted(self.layers.items()):
      print >> log.v2, "  layer %s %r #: %i" % (layer.layer_class, layer_name, layer.attrs["n_out"])
    if not self.layers:
      print >> log.v2, "  (no layers)"
    print >> log.v2, "net params #:", self.get_num_params()
    print >> log.v2, "net trainable params:", self.get_trainable_params()
