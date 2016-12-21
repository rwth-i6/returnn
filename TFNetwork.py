
from __future__ import print_function

import tensorflow as tf
import sys
import numpy
from Log import log
from TFNetworkLayer import Data, LayerBase, get_layer_class
from TFUtil import reuse_name_scope


class ExternData(object):
  """
  This holds `Data` instances for every data-key of external data from the dataset,
  i.e. the description such as shape and sparsity, etc.
  """

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
      init_args = {"dim": dim}
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
      self.data[key] = Data(name=key, auto_create_placeholders=True, **init_args)
    self.default_target = config.value('target', 'classes')

  def register_data_from_dict(self, data):
    """
    :param dict[str,dict[str]] data: init kwargs for Data
    """
    for key, value in data.items():
      self.data[key] = Data(name=key, auto_create_placeholders=True, **value)

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
  def __init__(self, config=None, extern_data=None, rnd_seed=42, train_flag=False, eval_flag=False):
    """
    :param Config.Config config: only needed to init extern_data if not specified explicitely
    :param ExternData|None extern_data:
    :param int rnd_seed:
    :param bool train_flag: True if we want to use this model in training
    :param bool eval_flag: True if we want to use this model in evaluation
    """
    if extern_data is None:
      extern_data = ExternData()
      if not config:
        from Config import get_global_config
        config = get_global_config()
      extern_data.init_from_config(config)
    self.extern_data = extern_data
    self.rnd_seed = rnd_seed
    self.random = numpy.random.RandomState(rnd_seed)
    self.train_flag = train_flag
    self.eval_flag = eval_flag
    self.layers_desc = {}  # type: dict[str,dict[str]]
    self.layers = {}  # type: dict[str,LayerBase]
    self.loss_by_layer = {}  # type: dict[str,tf.Tensor]
    self.error_by_layer = {}  # type: dict[str,tf.Tensor]
    self.total_loss = None  # type: tf.Tensor
    self.total_constraints = None  # type: tf.Tensor
    self.total_objective = None  # type: tf.Tensor
    self.saver = None  # type: tf.train.Saver
    self.recurrent = False

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
      src_names = layer_desc.pop("from", ["data"])
      if not isinstance(src_names, (list, tuple)):
        src_names = [src_names]
      layer_desc["sources"] = [
        _construct_layer(src_name)
        for src_name in src_names
        if not src_name == "none"]
      return self._add_layer(name=name, layer_class=layer_class, **layer_desc)

    for name, layer_desc in sorted(net_dict.items()):
      if name == "output" or "target" in layer_desc:
        _construct_layer(name)

  def _add_layer(self, name, layer_class, **layer_desc):
    """
    :param str name:
    :param () -> LayerBase layer_class:
    """
    with reuse_name_scope(name):
      layer = layer_class(name=name, network=self, **layer_desc)
    self.layers[name] = layer
    if layer.recurrent:
      self.recurrent = True
    return layer

  def construct_objective(self):
    with tf.name_scope("objective"):
      self.total_loss = 0
      self.total_constraints = 0
      self.loss_by_layer.clear()
      self.error_by_layer.clear()
      for name, layer in sorted(self.layers.items()):
        with reuse_name_scope(layer.name):
          assert isinstance(layer, LayerBase)
          loss = layer.get_loss_value()
          error = layer.get_error_value()
          constraints = layer.get_constraints_value()
          if loss is not None:
            tf.scalar_summary("loss_%s" % layer.name, loss)
          if error is not None:
            tf.scalar_summary("error_%s" % layer.name, error)
        if loss is not None:
          self.loss_by_layer[name] = loss
          self.total_loss += loss
        if error is not None:
          self.error_by_layer[name] = error
        if constraints is not None:
          self.total_constraints += constraints
      tf.scalar_summary("loss", self.total_loss)
      tf.scalar_summary("constraints", self.total_constraints)
      self.total_objective = self.total_loss + self.total_constraints
      tf.scalar_summary("objective", self.total_objective)

  def get_all_losses(self):
    if self.total_objective is None:
      self.construct_objective()
    return self.loss_by_layer

  def get_all_errors(self):
    if self.total_objective is None:
      self.construct_objective()
    return self.error_by_layer

  def get_objective(self):
    if self.total_objective is None:
      self.construct_objective()
    return self.total_objective

  def get_params_list(self):
    """
    :return: list of variables
    :rtype: list[tf.Variable]
    """
    l = []  # type: list[tf.Variable]
    for layer_name, layer in sorted(self.layers.items()):
      assert isinstance(layer, LayerBase)
      for param_name, param in sorted(layer.params.items()):
        assert isinstance(param, tf.Variable)
        l.append(param)
    return l

  def get_params_nested_dict(self):
    """
    :return: dict: layer_name -> param_name -> variable
    :rtype: dict[str,dict[str,tf.Variable]]
    """
    l = {}  # type: dict[str,dict[str,tf.Variable]]
    for layer_name, layer in self.layers.items():
      assert isinstance(layer, LayerBase)
      l[layer_name] = {}  # type: dict[str,tf.Variable]
      for param_name, param in layer.params.items():
        assert isinstance(param, tf.Variable)
        l[layer_name][param_name] = param
    return l

  def get_trainable_params(self):
    # TODO...
    return self.get_params_list()  # at the moment just the same

  def declare_train_params(self, hidden_layer_selection=None, with_output=None):
    pass  # TODO...

  def get_num_params(self):
    num_params = 0
    params = self.get_params_list()
    for param in params:
      shape = param.get_shape().as_list()
      num_params += numpy.prod(shape)
    return num_params

  def initialize_params(self, session):
    """
    :param tf.Session session:
    """
    session.run(tf.initialize_variables(var_list=self.get_params_list()))

  def get_param_values_dict(self, session):
    """
    :param tf.Session session:
    :return: dict: layer_name -> param_name -> variable numpy array
    :rtype: dict[str,dict[str,numpy.ndarray]]
    """
    l = {}  # type: dict[str,dict[str,numpy.ndarray]]
    for layer_name, layer in self.layers.items():
      assert isinstance(layer, LayerBase)
      l[layer_name] = layer.get_param_values_dict(session)
    return l

  def set_param_values_by_dict(self, values_dict, session):
    """
    :param dict[str,dict[str,numpy.ndarray]] values_dict:
    :param tf.Session session:
    """
    for layer_name, layer_values_dict in values_dict.items():
      self.layers[layer_name].set_param_values_by_dict(values_dict=layer_values_dict, session=session)

  def _create_saver(self):
    # Saver for storing checkpoints of the model.
    self.saver = tf.train.Saver(var_list=self.get_params_list(), max_to_keep=sys.maxint)

  def save_params_to_file(self, filename, session):
    """
    Will save the model parameters to the filename.
    Note that the model parameters live inside the current TF session.
    :param str filename:
    :param tf.Session session:
    """
    if not self.saver:
      self._create_saver()
    # We add some extra logic to try again for DiskQuota and other errors.
    # This could save us multiple hours of computation.
    try_again_wait_time = 10
    while True:
      try:
        self.saver.save(sess=session, save_path=filename)
        break
      except IOError as e:
        import errno, time
        if e.errno in [errno.EBUSY, errno.EDQUOT, errno.EIO, errno.ENOSPC]:
          print("Exception while saving:", e, file=log.v3)
          print("Trying again in %s secs." % try_again_wait_time, file=log.v3)
          time.sleep(try_again_wait_time)
          continue
        raise

  def load_params_from_file(self, filename, session):
    """
    Will save the model parameters to the filename.
    Note that the model parameters live inside the current TF session.
    :param str filename:
    :param tf.Session session:
    """
    if not self.saver:
      self._create_saver()
    self.saver.restore(sess=session, save_path=filename)

  def print_network_info(self, name="Network"):
    print("%s layer topology:" % name, file=log.v2)
    print("  extern data #:", self.extern_data.get_data_description(), file=log.v2)
    for layer_name, layer in sorted(self.layers.items()):
      print("  layer %s %r #: %i" % (layer.layer_class, layer_name, layer.attrs["n_out"]), file=log.v2)
    if not self.layers:
      print("  (no layers)", file=log.v2)
    print("net params #:", self.get_num_params(), file=log.v2)
    print("net trainable params:", self.get_trainable_params(), file=log.v2)
