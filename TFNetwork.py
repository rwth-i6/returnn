
from __future__ import print_function

import tensorflow as tf
import sys
import numpy
from Log import log
from TFNetworkLayer import Data, LayerBase, get_layer_class
from TFUtil import reuse_name_scope, VariableAssigner


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
    data_dims = LayerNetworkDescription.tf_extern_data_types_from_config(config)
    for key, init_args in data_dims.items():
      # In Returnn with Theano, we usually have the shape (time,batch,feature).
      # In TensorFlow, the default is (batch,time,feature).
      # This is also what we use here, i.e.:
      # batch_dim_axis=0, time_dim_axis=1. See TFEngine.DataProvider._get_next_batch().
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

  def has_data(self, name):
    return name in self.data

  def get_data(self, name):
    return self.data[name]

  def get_default_input_data(self):
    return self.data[self.default_input]

  def get_default_target_data(self):
    return self.data[self.default_target]

  def get_data_description(self):
    return ", ".join(["%s: %s" % (name, self.data[name].get_description(with_name=False))
                      for name in self.data.keys()])

  def get_queue_args(self, with_batch_dim, fixed_batch_dim=None):
    """
    :param bool with_batch_dim:
    :param int|None fixed_batch_dim:
    :return: kwargs for tf.Queue.__init__
    :rtype: dict[str,list]
    """
    names = list(sorted(self.data.keys()))
    shapes = [self.data[name].shape for name in names]
    if with_batch_dim:
      shapes = [(fixed_batch_dim,) + shape for shape in shapes]
    dtypes = [self.data[name].dtype for name in names]
    # And add seq_lens for each.
    for name in list(names):
      names.append("%s_seq_lens" % name)
      shapes.append((fixed_batch_dim,) if with_batch_dim else ())
      dtypes.append(self.data[name].size_dtype)
    return {"names": names, "shapes": shapes, "dtypes": dtypes}


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
    self.used_data_keys = set()
    self.rnd_seed = rnd_seed
    self.random = numpy.random.RandomState(rnd_seed)
    self.train_flag = train_flag
    self.eval_flag = eval_flag
    self._selected_train_layers = None
    self.layers_desc = {}  # type: dict[str,dict[str]]
    self.layers = {}  # type: dict[str,LayerBase]
    self.loss_by_layer = {}  # type: dict[str,tf.Tensor]
    self.error_by_layer = {}  # type: dict[str,tf.Tensor]
    self.total_loss = None  # type: tf.Tensor
    self.total_constraints = None  # type: tf.Tensor
    self.total_objective = None  # type: tf.Tensor
    self.global_train_step = tf.Variable(
      name="global_step", initial_value=0, dtype="int64", collections=[tf.GraphKeys.GLOBAL_STEP], trainable=False)
    self.saver = None  # type: tf.train.Saver
    self.recurrent = False
    self._assigner_cache = {}  # type: dict[tf.Variable,VariableAssigner]

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
        elif name.startswith("data:"):
          layer_desc = {"class": "source", "data_key": name[len("data:"):], "from": []}
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
      return self.add_layer(name=name, layer_class=layer_class, **layer_desc)

    for name, layer_desc in sorted(net_dict.items()):
      if name == "output" or "target" in layer_desc or "is_output_layer" in layer_desc:
        _construct_layer(name)

  def add_layer(self, name, layer_class, **layer_desc):
    """
    :param str name:
    :param ()->LayerBase layer_class:
    """
    with reuse_name_scope(layer_class.cls_get_tf_scope_name(name)):
      layer = layer_class(name=name, network=self, **layer_desc)
    assert layer.output
    assert layer.output.placeholder is not None
    assert layer.output.size_placeholder is not None
    self.layers[name] = layer
    if layer.recurrent:
      self.recurrent = True
    return layer

  def get_extern_data(self, key, mark_data_key_as_used=True):
    """
    Returns Data and add the key to self.used_data_keys if mark_data_key_as_used.
    :param str key:
    :param bool mark_data_key_as_used:
    :rtype: Data
    """
    if mark_data_key_as_used:
      self.used_data_keys.add(key)
    return self.extern_data.get_data(key)

  def construct_objective(self):
    with tf.name_scope("objective"):
      self.total_loss = 0
      self.total_constraints = 0
      self.loss_by_layer.clear()
      self.error_by_layer.clear()
      for name, layer in sorted(self.layers.items()):
        assert isinstance(layer, LayerBase)
        with reuse_name_scope(layer.tf_scope_name):
          loss = layer.get_loss_value()
          error = layer.get_error_value()
          constraints = layer.get_constraints_value()
          if loss is not None:
            tf.summary.scalar("loss_%s" % layer.name, loss)
          if error is not None:
            tf.summary.scalar("error_%s" % layer.name, error)
        if loss is not None:
          self.loss_by_layer[name] = loss
          self.total_loss += loss
        if error is not None:
          self.error_by_layer[name] = error
        if constraints is not None:
          self.total_constraints += constraints
      tf.summary.scalar("loss", self.total_loss)
      tf.summary.scalar("constraints", self.total_constraints)
      self.total_objective = self.total_loss + self.total_constraints
      tf.summary.scalar("objective", self.total_objective)

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

  def get_used_targets(self):
    """
    :return: sorted list of targets
    :rtype: list[str]
    """
    targets = set()
    for layer in self.layers.values():
      if layer.target:
        targets.add(layer.target)
    return list(sorted(targets))

  def get_default_target(self):
    """
    :return: e.g. "classes"
    :rtype: str
    """
    targets = self.get_used_targets()
    default_target = self.extern_data.default_target
    if not targets:
      return default_target
    if len(targets) == 1:
      return targets[0]
    if default_target in targets:
      return default_target
    raise Exception("multiple targets %r and default_target %r not in list. set 'target' in config" %
                    (targets, default_target))

  def get_output_layers(self):
    """
    :rtype: list[LayerBase]
    """
    return [layer for (_, layer) in sorted(self.layers.items()) if layer.is_output_layer()]

  def get_default_output_layer_name(self):
    """
    :rtype: str|None
    :returns: default output layer name if there is one, or None
    """
    if "output" in self.layers:
      return "output"
    output_layers = self.get_output_layers()
    if len(output_layers) == 1:
      return output_layers[1]
    return None  # no sensible default

  def get_params_list(self):
    """
    :return: list of model variables, i.e. from all the layers, excluding auxiliary vars like global_step
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
      l[layer_name] = layer.params
    return l

  def get_trainable_params(self):
    """
    :return: list of variables
    :rtype: list[tf.Variable]
    """
    if not self._selected_train_layers:
      self.declare_train_params()
    l = []  # type: list[tf.Variable]
    for layer_name in sorted(self._selected_train_layers):
      layer = self.layers[layer_name]
      assert isinstance(layer, LayerBase)
      for param_name, param in sorted(layer.params.items()):
        assert isinstance(param, tf.Variable)
        l.append(param)
    return l

  def declare_train_params(self, hidden_layer_selection=None, with_output=None):
    if hidden_layer_selection is None:
      hidden_layer_selection = [name for (name, layer) in self.layers.items() if not layer.is_output_layer()]
    else:
      hidden_layer_selection = list(hidden_layer_selection)
    if with_output is None:
      with_output = True
    if with_output:
      hidden_layer_selection += [name for (name, layer) in self.layers.items() if layer.is_output_layer()]
    hidden_layer_selection = set(hidden_layer_selection)
    self._selected_train_layers = sorted(hidden_layer_selection)

  def get_num_params(self):
    """
    :return: number of model parameters, i.e. total dimension
    :rtype: int
    """
    num_params = 0
    params = self.get_params_list()
    for param in params:
      shape = param.get_shape().as_list()
      num_params += numpy.prod(shape)
    return num_params

  def initialize_params(self, session):
    """
    :param tf.Session session:
    Note: This will create a new node to the graph for each call!
    And it will overwrite also the already initialized variables.
    So you should call this only once after network construction and before you maybe load some of the params
    from external sources.
    If you know that you will load all params explicitly, you would not need to call this function.
    """
    with tf.name_scope("var_initializer"):
      initializer_op = tf.variables_initializer(var_list=self.get_params_list() + self.get_auxiliary_params())
    session.run(initializer_op)

  def get_var_assigner(self, var):
    """
    :param tf.Variable var:
    """
    if var in self._assigner_cache:
      return self._assigner_cache[var]
    with reuse_name_scope("var_assigner"):
      assigner = VariableAssigner(var)
    self._assigner_cache[var] = assigner
    return assigner

  def get_param_values_dict(self, session):
    """
    :param tf.Session session:
    :return: dict: layer_name -> param_name -> variable numpy array
    :rtype: dict[str,dict[str,numpy.ndarray]]
    Note that this excludes auxiliary params.
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
    Note that this excludes auxiliary params.
    """
    for layer_name, layer_values_dict in values_dict.items():
      self.layers[layer_name].set_param_values_by_dict(values_dict=layer_values_dict, session=session)

  def get_auxiliary_params(self):
    return [self.global_train_step]

  def get_params_serialized(self, session):
    """
    :param tf.Session session:
    :rtype: TFNetworkParamsSerialized
    """
    return TFNetworkParamsSerialized(
      values_dict=self.get_param_values_dict(session=session),
      global_train_step=self.get_global_train_step(session=session))

  def set_params_by_serialized(self, serialized, session):
    """
    :param TFNetworkParamsSerialized serialized:
    :param tf.Session session:
    """
    self.set_param_values_by_dict(serialized.values_dict, session=session)
    self.set_global_train_step(serialized.global_train_step, session=session)

  def set_global_train_step(self, step, session):
    """
    :param int step:
    :param tf.Session session:
    """
    self.get_var_assigner(self.global_train_step).assign(step, session=session)

  def get_global_train_step(self, session):
    """
    :param tf.Session session:
    :rtype: int
    """
    return self.global_train_step.eval(session=session)

  def _create_saver(self):
    # Saver for storing checkpoints of the model.
    # If we want to check for existence of variables in the checkpoint:
    # http://stackoverflow.com/questions/38218174/how-can-find-the-variable-names-that-saved-in-tensorflow-checkpoint
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/framework/python/framework/checkpoint_utils.py
    # http://stackoverflow.com/questions/38944238/tensorflow-list-variables-in-the-checkpoint
    with tf.name_scope("saver"):
      self.saver = tf.train.Saver(
        var_list=self.get_params_list() + self.get_auxiliary_params(), max_to_keep=2 ** 31 - 1)

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
    print("  extern data:", self.extern_data.get_data_description(), file=log.v2)
    print("  used data keys: %s" % list(sorted(self.used_data_keys)))
    for layer_name, layer in sorted(self.layers.items()):
      print("  layer %s %r #: %i" % (layer.layer_class, layer_name, layer.output.dim), file=log.v2)
    if not self.layers:
      print("  (no layers)", file=log.v2)
    print("net params #:", self.get_num_params(), file=log.v2)
    print("net trainable params:", self.get_trainable_params(), file=log.v2)


class TFNetworkParamsSerialized(object):
  """
  Holds all the params as numpy arrays, including auxiliary params.
  """
  def __init__(self, values_dict, global_train_step):
    """
    :param dict[str,dict[str,numpy.ndarray]] values_dict:
    :param int global_train_step:
    """
    self.values_dict = values_dict
    self.global_train_step = global_train_step
