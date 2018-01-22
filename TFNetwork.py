
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
    self.extra_added_keys = set()  # set[str]

  def __repr__(self):
    return "<ExternData data=%r>" % self.data

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

  def init_from_dataset(self, dataset):
    """
    :param Dataset.Dataset dataset:
    """
    target_keys = list(dataset.get_target_list())
    if target_keys:
      if "classes" in target_keys:
        self.default_target = "classes"
      else:
        self.default_target = target_keys[0]
    data_keys = list(dataset.get_data_keys())
    input_keys = [key for key in data_keys if key not in target_keys]
    if input_keys:
      if "data" in input_keys:
        self.default_input = "data"
      else:
        self.default_input = input_keys[0]
    for key in data_keys:
      if key in dataset.get_target_list():
        available_for_inference = False
      else:
        available_for_inference = True
      dim = dataset.get_data_dim(key)
      shape = [None] + list(dataset.get_data_shape(key))
      sparse = dataset.is_data_sparse(key)
      dtype = dataset.get_data_dtype(key)
      self.data[key] = Data(
        name=key, auto_create_placeholders=True, batch_dim_axis=0, time_dim_axis=1,
        shape=shape, dim=dim, sparse=sparse, dtype=dtype,
        available_for_inference=available_for_inference)

  def check_matched_dataset(self, dataset, used_data_keys=None):
    """
    :param Dataset.Dataset dataset:
    :param set[str]|list[str] used_data_keys:
    :return: nothing, will assert the check
    """
    if used_data_keys is None:
      used_data_keys = dataset.get_data_keys()
    base_err_msg = "%r num_outputs %r vs %r" % (dataset, dataset.num_outputs, self)
    for key in sorted(used_data_keys):
      if key in ["seq_idx", "seq_tag"]:
        continue  # special cases, ignored for now
      if key in self.extra_added_keys:
        continue
      data = self.data[key]
      data_sparse = dataset.is_data_sparse(key)
      assert data.sparse == data_sparse, "key %r sparse mismatch. %s" % (key, base_err_msg)
      data_dtype = dataset.get_data_dtype(key)
      assert data.dtype == data_dtype, "key %r dtype mismatch. %s" % (key, base_err_msg)
      data_dim = dataset.get_data_dim(key)
      assert data.dim == data_dim, "key %r dim mismatch. %s" % (key, base_err_msg)
      data_shape = tuple(dataset.get_data_shape(key))
      assert data.shape[1:] == data_shape, "key %r shape mismatch. %s" % (key, base_err_msg)

  def register_data_from_dict(self, data):
    """
    :param dict[str,dict[str]] data: init kwargs for Data
    """
    for key, value in data.items():
      self.data[key] = Data(name=key, auto_create_placeholders=True, **value)

  def register_data(self, data):
    """
    :param Data data: will use data.name as the key
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
      for axis in self.data[name].get_axes_with_size():
        names.append("%s/size%i" % (name, axis))
        shapes.append((fixed_batch_dim,) if with_batch_dim else ())
        dtypes.append(self.data[name].size_dtype)
    return {"names": names, "shapes": shapes, "dtypes": dtypes}

  def get_sorted_data_items(self):
    """
    :rtype: list[(str,Data)]
    """
    keys = sorted(self.data.keys())
    if self.default_input in self.data:
      # Move to front.
      keys.remove(self.default_input)
      keys.insert(0, self.default_input)
    return [(key, self.data[key]) for key in keys]


class TFNetwork(object):
  def __init__(self, config=None, extern_data=None, rnd_seed=42,
               train_flag=False, eval_flag=False, search_flag=False,
               parent_layer=None, parent_net=None,
               name=None):
    """
    :param Config.Config config: only needed to init extern_data if not specified explicitly
    :param ExternData|None extern_data:
    :param int rnd_seed:
    :param bool|tf.Tensor train_flag: True if we want to use this model in training, False if in eval, or dynamic
    :param bool eval_flag: whether to calculate losses. if train_flag is not False, this will be set to True
    :param bool search_flag: whether we perform a beam-search. see usage
    :param TFNetworkLayer.LayerBase|None parent_layer:
    :param TFNetwork parent_net:
    :param str name: only for debugging
    """
    if not name:
      from Util import try_get_caller_name
      name = "<network via %s>" % try_get_caller_name(fallback="<unknown>")
    self.name = name
    if not parent_net and parent_layer:
      parent_net = parent_layer.network
    if not config and parent_net:
      config = parent_net._config
    if extern_data is None:
      if not config:
        from Config import get_global_config
        config = get_global_config()
      extern_data = ExternData()
      extern_data.init_from_config(config)
    self.extern_data = extern_data
    self._config = config
    self.used_data_keys = set()  # type: set[str]  # keys from extern_data
    self.rnd_seed = rnd_seed
    self.random = numpy.random.RandomState(rnd_seed)
    assert isinstance(train_flag, (bool, tf.Tensor))
    self.train_flag = train_flag
    assert isinstance(eval_flag, bool)
    if train_flag is not False:  # True or dynamic
      eval_flag = True
    self.eval_flag = eval_flag
    self.search_flag = search_flag
    self.parent_layer = parent_layer
    self.parent_net = parent_net
    self._selected_train_layers = None
    self._constructing_layers = []  # type: list[str]
    self.layers_desc = {}  # type: dict[str,dict[str]]
    self.layers = {}  # type: dict[str,LayerBase]
    self.loss_by_layer = {}  # type: dict[str,tf.Tensor]
    self.error_by_layer = {}  # type: dict[str,tf.Tensor]
    self.total_loss = None  # type: tf.Tensor
    self.total_constraints = None  # type: tf.Tensor
    self.total_objective = None  # type: tf.Tensor
    if parent_net:
      self.global_train_step = parent_net.global_train_step
    else:
      self.global_train_step = tf.Variable(
        name="global_step", initial_value=0, dtype="int64", collections=[tf.GraphKeys.GLOBAL_STEP], trainable=False)
    self.saver = None  # type: tf.train.Saver
    self.extra_vars_to_save = []  # type: list[tf.Variable]
    self.recurrent = False
    self._assigner_cache = {}  # type: dict[tf.Variable,VariableAssigner]
    self.concat_sources_dropout_cache = {}  # type: dict[(tuple[LayerBase],float),Data]
    self._batch_dim = None  # see get_batch_dim

  def __repr__(self):
    s = "TFNetwork %r" % self.name
    if self.parent_layer:
      s += " parent_layer=%r" % self.parent_layer
    elif self.parent_net:
      s += " parent_net=%r" % self.parent_net
    if self.train_flag is True:
      s += " train"
    elif self.train_flag is not None:
      s += " train=%r" % self.train_flag
    if self.search_flag is True:
      s += " search"
    return "<%s>" % s

  def get_absolute_name_scope_prefix(self):
    """
    :return: scope, always with "/" at the end, or ""
    :rtype: str
    """
    if self.parent_layer:
      return self.parent_layer.get_absolute_name_scope_prefix()
    if self.parent_net:
      return self.parent_net.get_absolute_name_scope_prefix()
    return ""

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
    for name, layer_desc in sorted(net_dict.items()):
      assert isinstance(name, str)
      assert isinstance(layer_desc, dict)
      if layer_desc.get("register_as_extern_data"):
        self._construct_layer(net_dict, name)
    for name, layer_desc in sorted(net_dict.items()):
      assert isinstance(name, str)
      assert isinstance(layer_desc, dict)
      if name == "output" or "target" in layer_desc or "loss" in layer_desc or layer_desc.get("is_output_layer", False):
        self._construct_layer(net_dict, name)
    assert not self._constructing_layers

  def _construct_layer(self, net_dict, name, get_layer=None, add_layer=None):
    """
    :param dict[str,dict[str]] net_dict:
    :param str name: layer name
    :param ((str) -> LayerBase)|None get_layer: optional, for source layers, for transform_config_dict.
      by default, this wraps self._construct_layer().
    :param ((str, LayerBase, dict) -> LayerBase) | None add_layer: by default self.add_layer
    :rtype: LayerBase
    """
    if name in self.layers:
      return self.layers[name]
    if name in self._constructing_layers:
      raise NetworkConstructionDependencyLoopException(
        layer_name=name, constructing_layers=self._constructing_layers, net_dict=net_dict, network=self)
    if name not in net_dict:
      if name == "data":
        layer_desc = {"class": "source", "from": []}
      elif name.startswith("data:"):
        layer_desc = {"class": "source", "data_key": name[len("data:"):], "from": []}
      else:
        raise Exception("layer not found: %r" % name)
    else:
      layer_desc = net_dict[name]
    if not get_layer:
      def get_layer(src_name):
        return self._construct_layer(net_dict=net_dict, name=src_name)
    if not add_layer:
      add_layer = self.add_layer
    self.layers_desc[name] = layer_desc
    layer_desc = layer_desc.copy()
    class_name = layer_desc.pop("class")
    layer_class = get_layer_class(class_name)
    self._constructing_layers.append(name)
    try:
      layer_class.transform_config_dict(layer_desc, network=self, get_layer=get_layer)
    finally:
      self._constructing_layers.remove(name)
    return add_layer(name=name, layer_class=layer_class, **layer_desc)

  def _create_layer_layer_desc(self, name, layer_desc):
    """
    :param str name: layer name
    :param dict[str] layer_desc: opts
    :rtype: dict[str]
    """
    if self.search_flag:
      from TFNetworkLayer import SearchChoices
      layer_desc = SearchChoices.translate_to_common_search_beam(layer_desc)
    layer_desc = layer_desc.copy()
    assert "name" not in layer_desc
    assert "network" not in layer_desc
    layer_desc["name"] = name
    layer_desc["network"] = self
    return layer_desc

  def _create_layer(self, name, layer_class, **layer_desc):
    """
    This will create the layer given the layer_desc arguments.

    :param str name:
    :param (()->LayerBase)|LayerBase layer_class:
    :param layer_desc: contains the kwargs for the layer class.
      the args should have been transformed via layer_class.transform_config_dict before (see _construct_layer).
      must not contain "name" and "network", which will be automatically added here.
      should not contain "output", which will be initialized to layer_class.get_out_data_from_opts.
      the layer_class will usually then define the layer.output and its placeholder.
      there is one notable exception: the InternalLayer, where you predefine the output.
    :rtype: LayerBase
    """
    from Util import help_on_type_error_wrong_args
    layer_desc = self._create_layer_layer_desc(name=name, layer_desc=layer_desc)
    debug_print_layer_output_template = self.get_config().bool("debug_print_layer_output_template", False)
    debug_print_layer_output_shape = self.get_config().bool("debug_print_layer_output_shape", False)
    debug_add_check_numerics_on_output = self.get_config().bool("debug_add_check_numerics_on_output", False)  # also see debug_add_check_numerics_ops
    with reuse_name_scope(layer_class.cls_get_tf_scope_name(name)):
      try:
        if "output" not in layer_desc:
          layer_desc["output"] = layer_class.get_out_data_from_opts(**layer_desc)
        if debug_print_layer_output_template:
          print("layer %s%r output: %r" % (self.get_absolute_name_scope_prefix(), name, layer_desc["output"]))
        layer = layer_class(**layer_desc)
      except TypeError:
        help_on_type_error_wrong_args(cls=layer_class, kwargs=list(layer_desc.keys()))
        raise
      layer.post_init()
      if debug_print_layer_output_shape:
        layer.output.placeholder = tf.Print(
          layer.output.placeholder, [layer_class.cls_get_tf_scope_name(name), "shape:", tf.shape(layer.output.placeholder)],
          summarize=10, name="debug_print_layer_output_shape")
      if debug_add_check_numerics_on_output and layer.output.dtype.startswith("float"):
        print("debug_add_check_numerics_on_output: add for layer %r: %r" % (name, layer.output.placeholder))
        from TFUtil import identity_with_check_numerics
        layer.output.placeholder = identity_with_check_numerics(
          layer.output.placeholder,
          name="%s_identity_with_check_numerics" % layer_class.cls_get_tf_scope_name(name))
    assert layer.output
    assert layer.output.placeholder is not None
    layer.output.placeholder.set_shape(layer.output.batch_shape)
    assert layer.output.size_placeholder is not None
    return layer

  def add_layer(self, name, layer_class, **layer_desc):
    """
    This will construct the layer given the layer_desc arguments,
    and add it to the network.

    :param str name:
    :param (()->LayerBase)|LayerBase layer_class:
    :param layer_desc: contains the kwargs for the layer class.
      the args should have been transformed via layer_class.transform_config_dict before (see _construct_layer).
      must not contain "name" and "network", which will be automatically added here.
      should not contain "output", which will be initialized to layer_class.get_out_data_from_opts.
      the layer_class will usually then define the layer.output and its placeholder.
      there is one notable exception: the InternalLayer, where you predefine the output.
    """
    assert name not in self.layers
    layer = self._create_layer(name=name, layer_class=layer_class, **layer_desc)
    self.layers[name] = layer
    if layer.recurrent:
      self.recurrent = True
    return layer

  def get_extern_data(self, key, mark_data_key_as_used=True):
    """
    Returns Data and add the key to self.used_data_keys if mark_data_key_as_used.
    :param str key: e.g. "data" or "classes"
    :param bool mark_data_key_as_used:
    :rtype: Data
    """
    if mark_data_key_as_used:
      self.used_data_keys.add(key)
    if key == "seq_idx" and key not in self.extern_data.data:
      self.extern_data.data[key] = Data(name="seq_idx", shape=(), dtype="int32", sparse=False, auto_create_placeholders=True)
    if key == "seq_tag" and key not in self.extern_data.data:
      self.extern_data.data[key] = Data(name="seq_tag", shape=(), dtype="string", auto_create_placeholders=True)
    return self.extern_data.get_data(key)

  def get_seq_tags(self, mark_data_key_as_used=True):
    """
    :param bool mark_data_key_as_used: for extern_data
    :return: tensor of shape (batch,) of dtype string, via extern_data
    :rtype: tf.Tensor
    """
    return self.get_extern_data(key="seq_tag", mark_data_key_as_used=mark_data_key_as_used).placeholder

  def construct_objective(self):
    with tf.name_scope("objective"):
      self.total_loss = 0
      self.total_constraints = 0
      self.loss_by_layer.clear()
      self.error_by_layer.clear()
      for name, layer in sorted(self.layers.items()):
        assert isinstance(layer, LayerBase)
        with reuse_name_scope("loss"):
          with reuse_name_scope(layer.tf_scope_name):
            loss = layer.get_loss_value()
            error = layer.get_error_value()
            if loss is not None:
              tf.summary.scalar("loss_%s" % layer.name, loss * layer.get_loss_normalization_factor())
              if self.get_config().bool("calculate_exp_loss", False):
                tf.summary.scalar("exp_loss_%s" % layer.name, tf.exp(loss * layer.get_loss_normalization_factor()))
              if self.get_config().bool("debug_add_check_numerics_on_output", False):
                print("debug_add_check_numerics_on_output: add for layer loss %r: %r" % (name, layer.output.placeholder))
                from TFUtil import identity_with_check_numerics
                loss = identity_with_check_numerics(
                  loss,
                  name="%s_loss_identity_with_check_numerics" % layer.tf_scope_name)
            if error is not None:
              tf.summary.scalar("error_%s" % layer.name, error * layer.get_loss_normalization_factor())
        with reuse_name_scope("constraints"):
          with reuse_name_scope(layer.tf_scope_name):
            constraints = layer.get_constraints_value()

        with reuse_name_scope("loss"):
          if loss is not None:
            self.loss_by_layer[name] = loss
            if layer.loss_scale != 1:
              loss *= layer.loss_scale
            if self.total_loss is 0:
              self.total_loss = loss
            else:
              self.total_loss += loss
          if error is not None:
            self.error_by_layer[name] = error
        with reuse_name_scope("constraints"):
          if constraints is not None:
            if self.total_constraints is 0:
              self.total_constraints = constraints
            else:
              self.total_constraints += constraints

      tf.summary.scalar("loss", self.total_loss)
      tf.summary.scalar("constraints", self.total_constraints)
      self.total_objective = self.total_loss + self.total_constraints
      tf.summary.scalar("objective", self.total_objective)

  def maybe_construct_objective(self):
    if self.total_objective is None:
      self.construct_objective()

  def get_all_losses(self):
    self.maybe_construct_objective()
    return self.loss_by_layer

  def get_all_errors(self):
    """
    :rtype: dict[str|tf.Tensor]
    :return: layer-name -> error dict. contains only the layers which have some error value
    """
    self.maybe_construct_objective()
    return self.error_by_layer

  def get_objective(self):
    self.maybe_construct_objective()
    return self.total_objective

  def get_total_loss(self):
    """
    :rtype: int|tf.Tensor
    :return: 0 if no loss, or tf.Tensor
    """
    self.maybe_construct_objective()
    return self.total_loss

  def get_total_constraints(self):
    self.maybe_construct_objective()
    return self.total_constraints

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

  def get_default_output_layer(self, must_exist=True):
    """
    :param bool must_exist: if it does not exist, will raise an exception
    :rtype: LayerBase|None
    :return: the default output layer
    """
    name = self.get_default_output_layer_name()
    if not name:
      assert not must_exist, "default output layer does not exist"
      return None
    return self.layers[name]

  def get_layer(self, layer_name):
    """
    Normally just self.layers[layer_name] but with some extra logic added,
    such as resolving "base:" prefix to the parent network.

    :param str layer_name:
    :rtype: LayerBase
    """
    if layer_name.startswith("base:"):
      return self.parent_net.get_layer(layer_name[len("base:"):])
    return self.layers[layer_name]

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

  def get_saveable_params_list(self):
    """
    :return: list of model variables or SaveableObject, to save/restore
    :rtype: list[tf.Variable|tensorflow.python.training.saver.BaseSaverBuilder.SaveableObject]
    """
    l = []  # type: list[tf.Variable]
    for layer_name, layer in sorted(self.layers.items()):
      assert isinstance(layer, LayerBase)
      for param_name, param in sorted(layer.get_saveable_params_dict().items()):
        l.append(param)
    l += self.get_auxiliary_params()
    l += self.extra_vars_to_save
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
    trainable_vars_col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    assert isinstance(trainable_vars_col, list)
    l = []  # type: list[tf.Variable]
    for layer_name in sorted(self._selected_train_layers):
      layer = self.layers[layer_name]
      assert isinstance(layer, LayerBase)
      if not layer.trainable:
        continue
      for param_name, param in sorted(layer.params.items()):
        assert isinstance(param, tf.Variable)
        if param in trainable_vars_col:
          l.append(param)
          trainable_vars_col.remove(param)
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
    var_list = self.get_params_list() + self.get_auxiliary_params()
    with tf.name_scope("var_initializer"):
      initializer_op = tf.variables_initializer(var_list=var_list)
    session.run(initializer_op)
    for var in var_list:
      # Some of our code could set this, e.g. the SubnetworkLayer.
      custom_post_init = getattr(var, "custom_post_init", None)
      if custom_post_init:
        assert callable(custom_post_init)
        custom_post_init(session=session)

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
      if layer_values_dict:
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

  def reset_saver(self):
    """
    Resets the :class:`tf.train.Saver` object which will be used
    for :func:`load_params_from_file` and :func:`save_params_to_file`.
    Warning: Don't repeat that too often as it will always create new ops in the computation graph.
    """
    self.saver = None

  def _create_saver(self):
    # Saver for storing checkpoints of the model.
    with tf.name_scope("saver"):
      self.saver = tf.train.Saver(
        var_list=self.get_saveable_params_list(), max_to_keep=2 ** 31 - 1)

  def save_params_to_file(self, filename, session):
    """
    Will save the model parameters to the filename.
    Note that the model parameters live inside the current TF session.
    :param str filename:
    :param tf.Session session:
    """
    import os
    filename = os.path.abspath(filename)  # TF needs absolute path
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
    # Note:
    # If we want to check for existence of variables in the checkpoint:
    # http://stackoverflow.com/questions/38218174/how-can-find-the-variable-names-that-saved-in-tensorflow-checkpoint
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/framework/python/framework/checkpoint_utils.py
    # http://stackoverflow.com/questions/38944238/tensorflow-list-variables-in-the-checkpoint
    try:
      self.saver.restore(sess=session, save_path=filename)
    except tf.errors.NotFoundError as exc:
      print("load_params_from_file: some variables not found", file=log.v2)
      # First, the short version, we will try to automatically resolve this, similar to this:
      # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/tools/checkpoint_convert.py
      # Also see:
      # https://github.com/tensorflow/tensorflow/issues/11168
      # https://github.com/tensorflow/tensorflow/commit/92da8abfd35b93488ed7a55308b8f589ee23b622
      # https://github.com/tensorflow/tensorflow/commit/157370e5916b85c65958ed8383ae31d727228ed7
      # This map_list can be extended by all the mappings in checkpoint_convert.py.
      map_list = {"lstm_cell/biases": "lstm_cell/bias", "lstm_cell/weights": "lstm_cell/kernel"}
      reader = tf.train.NewCheckpointReader(filename)
      net_vars = [v for v in self.get_saveable_params_list() if isinstance(v, tf.Variable)]
      net_saveables = [v for v in self.get_saveable_params_list() if not isinstance(v, tf.Variable)]
      var_ckpt_names = set(reader.get_variable_to_shape_map())
      var_net_names = set([v.name[:-2] for v in net_vars] + [v.name for v in net_saveables])
      missing_var_names = [v for v in sorted(var_net_names) if v not in var_ckpt_names]
      obsolete_var_names = [v for v in sorted(var_ckpt_names) if v not in var_net_names]
      print("Variables to restore which are not in checkpoint:", missing_var_names, file=log.v2)
      if not missing_var_names:
        print("Strange, nothing missing?", file=log.v2)
        print("Original exception:", exc, file=log.v2)

      var_name_map = {}  # type: dict[str,()->numpy.ndarray]  # current name -> value-loader

      def make_load_renamed(old_name):
        def load_old():
          return reader.get_tensor(old_name)
        return load_old

      class make_load_cudnn_rnn:
        cudnn_postfix = "/cudnn/CudnnRNNParamsToCanonical:0"

        def __init__(self, prefix, target="lstm_block_wrapper/"):
          self.target = target
          self.keys = [target + "bias", target + "kernel"]
          self.prefix = prefix
          self.data = None

        def _load(self):
          from TFNetworkRecLayer import RecLayer
          self.data = RecLayer.convert_cudnn_canonical_to_lstm_block(
            reader=reader, prefix=self.prefix, target=self.target)

        def make_getter(self, key):
          def get():
            if self.data is None:
              self._load()
            return self.data[key]
          return get

        def get_lazy_dict(self):
          return {self.prefix + k: self.make_getter(self.prefix + k) for k in self.keys}

      for v in obsolete_var_names:
        for k_old, k_new in map_list.items():
          if v.endswith("/%s" % k_old):
            v2 = v[:-len(k_old)] + k_new
            if v2 in missing_var_names:
              var_name_map[v2] = make_load_renamed(old_name=v)
              break
        if v.endswith(make_load_cudnn_rnn.cudnn_postfix):
          var_name_map.update(
            make_load_cudnn_rnn(prefix=v[:-len(make_load_cudnn_rnn.cudnn_postfix) + 1]).get_lazy_dict())

      could_not_find_map_list = [v for v in missing_var_names if v not in var_name_map]
      if not could_not_find_map_list:
        # We can restore all.
        print("We found these corresponding variables in the checkpoint:", var_name_map, file=log.v2)
        print("Loading now...", file=log.v3)
        # Similar: from tensorflow.contrib.framework.python.ops import assign_from_checkpoint
        for v in self.get_saveable_params_list():
          assert isinstance(v, tf.Variable), "not yet implemented otherwise..."
          v_name = v.name[:-2]  # current name
          if v_name in var_ckpt_names:
            value = reader.get_tensor(v_name)
          else:
            value = var_name_map[v_name]()
          assigner = self.get_var_assigner(v)
          assigner.assign(value=value, session=session)
        print("Successfully loaded all variables. Any new save will use the updated variable names.", file=log.v3)

      else:
        print("Could not find mappings for these variables:", could_not_find_map_list, "var_name_map:", var_name_map)
        print("Error, some entry is missing in the checkpoint %r: %s: %s" % (filename, type(exc), exc), file=log.v1)
        print("All variables in checkpoint:")
        print(reader.debug_string())
        print("All variables to restore:")
        for v in net_vars + net_saveables:
          print(v)
        print()
        print("Variables to restore which are not in checkpoint:")
        for v in sorted(var_net_names):
          if v in var_ckpt_names:
            continue
          print(v)
        print()
        print("Variables in checkpoint which are not needed for restore:")
        for v in sorted(var_ckpt_names):
          if v in var_net_names:
            continue
          print(v)
        print()
        raise exc

  def print_network_info(self, name="Network"):
    print("%s layer topology:" % name, file=log.v2)
    print("  extern data:", self.extern_data.get_data_description(), file=log.v2)
    print("  used data keys: %s" % list(sorted(self.used_data_keys)), file=log.v2)
    for layer_name, layer in sorted(self.layers.items()):
      layer_dim = 'unknown' if layer.output.dim is None else '%i' % layer.output.dim
      print("  layer %s %r #: %s" % (layer.layer_class, layer_name, layer_dim), file=log.v2)
    if not self.layers:
      print("  (no layers)", file=log.v2)
    print("net params #:", self.get_num_params(), file=log.v2)
    print("net trainable params:", self.get_trainable_params(), file=log.v2)

  def cond_on_train(self, fn_train, fn_eval):
    """
    Uses fn_train() or fn_eval() base on self.train_flag.
    It will be a branched evaluation.

    :param ()->tf.Tensor fn_train:
    :param ()->tf.Tensor fn_eval:
    :return: fn_train() if self.train_flag else fn_eval()
    :rtype: tf.Tensor
    """
    from TFUtil import cond
    return cond(self.train_flag, fn_train, fn_eval)

  def get_search_choices(self, sources=None, src=None, base_search_choice=None, _visited=None):
    """
    Recursively searches through all sources,
    and if there is a ChoiceLayer / any layer with search_choices, returns it.
    Could also go to the parent network.
    If there are multiple, it assumes they are on the same search-sequence in the search-tree
    and it will return the last one.

    :param LayerBase|None src:
    :param LayerBase|None base_search_choice:
    :param list[LayerBase]|None sources:
    :param set[LayerBase]|None _visited: keep track about visited layers in case there are circular deps
    :return: (direct or indirect) source LayerBase which has search_choices, or None
    :rtype: LayerBase|None
    """
    if src is not None:
      assert isinstance(src, LayerBase)
      if src.search_choices:
        if src.search_choices.is_decided:
          return None
        return src
      assert base_search_choice is None
      base_search_choice = src
    if base_search_choice is not None:
      assert sources is None
      sources = base_search_choice.get_dep_layers()
    if _visited is None:
      _visited = set()
    assert sources is not None
    sources = [src for src in sources if src not in _visited]
    _visited.update(sources)
    layers = [self.get_search_choices(src=src, _visited=_visited) for src in sources]
    layers = [layer for layer in layers if layer is not None]  # type: list[LayerBase]
    if not layers:
      if self.parent_layer:
        return self.parent_layer.network.get_search_choices(sources=self.parent_layer.get_dep_layers())
      return None
    from TFNetworkLayer import SearchChoices
    from functools import cmp_to_key
    layers = sorted(layers, key=cmp_to_key(lambda l1, l2: SearchChoices.compare(l1.search_choices, l2.search_choices)))
    return layers[-1]

  def debug_search_choices(self, base_search_choice):
    """
    :param LayerBase base_search_choice:
    """
    print("debug search choices:")
    print("  base:", base_search_choice)
    print("  network:")
    for _, layer in sorted(self.layers.items()):
      print("    layer:", layer)

    class Visitor(set):
      def update(self, others):
        print("  visit: %r" % (others,))
        super(Visitor, self).update(others)

    self.get_search_choices(base_search_choice=base_search_choice, _visited=Visitor())

  def get_data_batch_dim(self):
    """
    Get the batch-dim size, i.e. amount of sequences in the current batch.
    Consider that the data tensor is usually of shape [batch, time, dim],
    this would return shape(data)[0].

    The code currently assumes that the batch-dim can be taken from the extern data.
    If it does not have that available for some reason (e.g. some subnetwork),
    it will try some alternative sources and assumes that they have the correct batch-dim.

    Note that the batch-dim usually stays always the same across the whole network
    and also every individual batch sequence will stay related.
    One notable exception of this is the choice layer, where the
    batch-dim will get expanded by the beam search if search is used,
    as well as in all following layers, until there is a decide layer.

    :return: int scalar tensor which states the batch-dim
    :rtype: int|tf.Tensor
    """
    from TFUtil import get_shape_dim, reuse_name_scope_of_tensor
    # First check parent because there we might get the true batch dim.
    if self.parent_net:
      return self.parent_net.get_data_batch_dim()
    if self._batch_dim is not None:
      return self._batch_dim
    for key, data in self.extern_data.get_sorted_data_items():
      assert isinstance(data, Data)
      if data.available_for_inference:
        self.used_data_keys.add(key)
        with reuse_name_scope_of_tensor(data.placeholder):
          batch_dim = get_shape_dim(data.placeholder, data.batch_dim_axis, name="batch_dim")
          self._batch_dim = batch_dim
          return batch_dim
    raise Exception("We cannot tell the batch dim.")

  def set_rec_step_info(self, i, end_flag=None, seq_lens=None):
    """
    Used by _SubnetworkRecCell.
    :param tf.Tensor i: scalar, int32, current step (time)
    :param tf.Tensor|None end_flag: (batch,), bool, says that the current sequence has ended
    :param tf.Tensor|None seq_lens: (batch,) int32, seq lens
    """
    from TFNetworkRecLayer import RecStepInfoLayer
    self.layers[":i"] = RecStepInfoLayer(
      name=":i", network=self, i=i, end_flag=end_flag, seq_lens=seq_lens)

  def have_rec_step_info(self):
    return ":i" in self.layers

  def get_rec_step_info(self):
    """
    Assumes that have_rec_step_info is True.
    :rtype: TFNetworkRecLayer.RecStepInfoLayer
    """
    from TFNetworkRecLayer import RecStepInfoLayer
    layer = self.layers[":i"]
    assert isinstance(layer, RecStepInfoLayer)
    return layer

  def get_rec_step_index(self):
    """
    Assumes that have_rec_step_info is True.

    :rtype: tf.Tensor
    :return: scalar, int32
    """
    return self.get_rec_step_info().step

  def get_config(self, consider_global_config=True, fallback_dummy_config=True):
    """
    :param bool consider_global_config: if no config is set, check for global config
    :param bool fallback_dummy_config: if no config, return a new empty Config, otherwise return None
    :rtype: Config.Config|None
    """
    from Config import Config, get_global_config
    if self._config:
      return self._config
    if self.parent_net:
      return self.parent_net.get_config(
        consider_global_config=consider_global_config, fallback_dummy_config=fallback_dummy_config)
    if consider_global_config:
      config = get_global_config(raise_exception=False)
      if config:
        return config
    if fallback_dummy_config:
      return Config()
    return None


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


class NetworkConstructionDependencyLoopException(Exception):
  """
  This is raised when there is a dependency loop in the network construction.
  """
  def __init__(self, network, layer_name, constructing_layers, net_dict):
    """
    :param TFNetwork network:
    :param str layer_name:
    :param list[str] constructing_layers:
    :param dict[str,dict[str]] net_dict:
    """
    msg = "Error: There is a dependency loop on layer %r." % layer_name
    msg += "\nConstruction stack (most recent first):"
    for l in reversed(constructing_layers):
      msg += "\n  %s" % l
    super(NetworkConstructionDependencyLoopException, self).__init__(msg)
    self.network = network
    self.layer_name = layer_name
    self.net_dict = net_dict
