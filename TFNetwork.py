
from __future__ import print_function

import tensorflow as tf
import sys
import numpy
import contextlib
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
               parent_layer=None, parent_net=None, extra_parent_net=None,
               name=None):
    """
    :param Config.Config config: only needed to init extern_data if not specified explicitly
    :param ExternData|None extern_data:
    :param int rnd_seed:
    :param bool|tf.Tensor train_flag: True if we want to use this model in training, False if in eval, or dynamic
    :param bool eval_flag: whether to calculate losses. if train_flag is not False, this will be set to True
    :param bool search_flag: whether we perform a beam-search. see usage
    :param TFNetworkLayer.LayerBase|None parent_layer:
    :param TFNetwork|None parent_net:
    :param TFNetwork|None extra_parent_net:
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
    self.extra_parent_net = extra_parent_net
    self.extra_net = None  # type: TFNetwork
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
    self.epoch_step = None
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
    if self.extra_net:
      s += " extra_net=%r" % self.extra_net
    if self.train_flag is True:
      s += " train"
    elif self.train_flag is not None:
      s += " train=%r" % self.train_flag
    if self.search_flag:
      s += " search"
    return "<%s>" % s

  def get_root_network(self):
    """
    :rtype: TFNetwork
    """
    if self.parent_net:
      return self.parent_net.get_root_network()
    return self

  def get_absolute_name_scope_prefix(self):
    """
    :return: scope, always with "/" at the end, or ""
    :rtype: str
    """
    if self.parent_layer:
      return self.parent_layer.get_absolute_name_scope_prefix()
    if self.parent_net:
      return self.parent_net.get_absolute_name_scope_prefix()
    if self.extra_parent_net:
      return self.extra_parent_net.get_absolute_name_scope_prefix()
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

  _LayerNamesToIgnore = ["#config", "#repetition"]

  def construct_from_dict(self, net_dict):
    """
    :param dict[str,dict[str]] net_dict:
    """
    for name, layer_desc in sorted(net_dict.items()):
      assert isinstance(name, str)
      if name in self._LayerNamesToIgnore:
        continue
      assert isinstance(layer_desc, dict)
      if layer_desc.get("register_as_extern_data"):
        self.construct_layer(net_dict, name)
    for name, layer_desc in sorted(net_dict.items()):
      assert isinstance(name, str)
      if name in self._LayerNamesToIgnore:
        continue
      assert isinstance(layer_desc, dict)
      if layer_desc.get("only_on_search") and not self.search_flag:
        continue
      if layer_desc.get("only_on_eval") and not self.eval_flag:
        continue
      if name == "output" or "target" in layer_desc or "loss" in layer_desc or layer_desc.get("is_output_layer", False):
        self.construct_layer(net_dict, name)
    assert not self._constructing_layers

  def construct_extra_net(self, net_dict, layer_list, search_flag=None):
    """
    The purpose is to create another net like `self` but with different flags,
    e.g. with `search_flag = True`.
    That `extra_net` can have different losses, which will be added.

    :param dict[str,dict[str]] net_dict:
    :param list[str] layer_list:
    :param bool|None search_flag:
    """
    if not self.extra_net:
      self.extra_net = TFNetwork(
        config=self._config, extern_data=self.extern_data, rnd_seed=self.random.randint(2 ** 31),
        train_flag=self.train_flag, eval_flag=self.eval_flag,
        search_flag=search_flag if search_flag is not None else self.search_flag,
        extra_parent_net=self)

    def extra_get_layer(layer_name):
      if layer_name in self.extra_net.layers:
        return self.extra_net.layers[layer_name]
      if layer_name in self.layers:
        return self.layers[layer_name]
      return self.extra_net.construct_layer(net_dict=net_dict, name=layer_name)

    for layer_name in layer_list:
      self.extra_net.construct_layer(net_dict=net_dict, name=layer_name, get_layer=extra_get_layer)

  def construct_layer(self, net_dict, name, get_layer=None, add_layer=None):
    """
    :param dict[str,dict[str]] net_dict:
    :param str name: layer name
    :param ((str) -> LayerBase)|None get_layer: optional, for source layers, for transform_config_dict.
      by default, this wraps self.construct_layer().
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
        raise LayerNotFound("layer %r not found in %r" % (name, self))
    else:
      layer_desc = net_dict[name]
    if not get_layer:
      def get_layer(src_name):
        return self.construct_layer(net_dict=net_dict, name=src_name)
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
      the args should have been transformed via layer_class.transform_config_dict before (see construct_layer).
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
    with reuse_name_scope(layer_class.cls_get_tf_scope_name(name)), self.register_network_scope():
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
          name="%s_identity_with_check_numerics_output" % layer_class.cls_get_tf_scope_name(name))
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
      the args should have been transformed via layer_class.transform_config_dict before (see construct_layer).
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
      layer_items = sorted(self.layers.items())
      if self.extra_net:
        extra_name_prefix = "extra"
        if self.extra_net.search_flag and not self.search_flag:
          extra_name_prefix += "_search"
        layer_items += [
          ("%s:%s" % (extra_name_prefix, name), layer)
          for (name, layer) in sorted(self.extra_net.layers.items())]
      for name, layer in layer_items:
        assert isinstance(name, str)
        assert isinstance(layer, LayerBase)
        tf_scope_name = layer.cls_get_tf_scope_name(name=name.replace(":", "/", 1))
        tf_flat_scope_name = layer.cls_get_tf_scope_name(name=name.replace(":", "_"))
        assert isinstance(layer, LayerBase)
        with reuse_name_scope("loss"):
          with reuse_name_scope(tf_scope_name):
            loss = layer.get_loss_value()
            error = layer.get_error_value()
            if loss is not None:
              tf.summary.scalar("loss_%s" % tf_flat_scope_name, loss * layer.get_loss_normalization_factor())
              if self.get_config().bool("calculate_exp_loss", False):
                tf.summary.scalar("exp_loss_%s" % tf_flat_scope_name, tf.exp(loss * layer.get_loss_normalization_factor()))
              if self.get_config().bool("debug_add_check_numerics_on_output", False):
                print("debug_add_check_numerics_on_output: add for layer loss %r: %r" % (name, layer.output.placeholder))
                from TFUtil import identity_with_check_numerics
                loss = identity_with_check_numerics(
                  loss, name="%s_identity_with_check_numerics_loss" % tf_flat_scope_name)
            if error is not None:
              tf.summary.scalar("error_%s" % tf_flat_scope_name, error * layer.get_loss_normalization_factor())
        with reuse_name_scope("constraints"):
          with reuse_name_scope(tf_scope_name):
            constraints = layer.get_constraints_value()

        with reuse_name_scope("loss"):
          if loss is not None:
            self.loss_by_layer[name] = loss
          if loss is not None and layer.loss_scale != 1:
            if not layer.loss_scale:
              loss = None
            else:
              loss *= layer.loss_scale
          if loss is not None:
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
    Raises :class:`LayerNotFound` if the layer is not found.

    :param str layer_name:
    :rtype: LayerBase
    """
    if layer_name in self.layers:
      return self.layers[layer_name]
    if layer_name.startswith("extra:") or layer_name.startswith("extra_search:"):
      if not self.extra_net:
        raise LayerNotFound("cannot get layer %r, no extra net for %r" % (layer_name, self))
      return self.extra_net.get_layer(layer_name[layer_name.find(":") + 1:])
    if self.extra_parent_net:
      return self.extra_parent_net.get_layer(layer_name)
    if layer_name.startswith("base:"):
      if not self.parent_net:
        raise LayerNotFound("cannot get layer %r, no parent net for %r" % (layer_name, self))
      return self.parent_net.get_layer(layer_name[len("base:"):])
    if layer_name not in self.layers:
      raise LayerNotFound("layer %r not found in %r" % (layer_name, self))
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

  def get_saveable_param_replace_dict(self):
    """
    :return: params and saveable_param_replace resolved, union of all layers
    :rtype: dict[str,tf.Variable|tensorflow.python.training.saver.BaseSaverBuilder.SaveableObject]
    """
    d = {}
    for layer_name, layer in sorted(self.layers.items()):
      assert isinstance(layer, LayerBase)
      d.update(layer.get_saveable_params_dict())
    return d

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
      if all(shape):
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
      # See :func:`set_custom_post_init`
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

  def set_param_values_by_dict(self, values_dict, ignore_non_existing=False, **kwargs):
    """
    :param dict[str,dict[str,numpy.ndarray]] values_dict:
    :param bool ignore_non_existing:
    :param kwargs: passed to :func:`LayerBase.set_param_values_by_dict`

    Note that this excludes auxiliary params.
    """
    for layer_name, layer_values_dict in values_dict.items():
      if layer_values_dict:
        if ignore_non_existing and layer_name not in self.layers:
          print("Will not set layer %r because it does not exist." % (layer_name,), file=log.v3)
          continue
        self.layers[layer_name].set_param_values_by_dict(values_dict=layer_values_dict, **kwargs)

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

  def set_params_by_serialized(self, serialized, session, **kwargs):
    """
    :param TFNetworkParamsSerialized serialized:
    :param tf.Session session:
    :param kwargs: passed to :func:`set_param_values_by_dict`
    """
    self.set_param_values_by_dict(serialized.values_dict, session=session, **kwargs)
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

  def get_epoch_step(self):
    """
    :return: int64
    :rtype: tf.Tensor
    """
    if self.parent_net:
      return self.parent_net.get_epoch_step()
    if self.epoch_step is not None:
      return self.epoch_step
    with reuse_name_scope("", absolute=True):
      self.epoch_step = tf.placeholder(name="epoch_step", shape=(), dtype=tf.int64)
    return self.epoch_step

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
      try:
        loader = CustomCheckpointLoader(
          filename=filename, saveable_params=self.get_saveable_params_list())
        if not loader.missing_var_names:
          print("Strange, nothing missing? Pre-loaded missing variables from other checkpoints?", file=log.v2)
        loader.load_now(session=session)
      except tf.errors.NotFoundError:
        print("Error, some entry is missing in the checkpoint %r: %s: %s" % (filename, type(exc), exc), file=log.v1)
        print("CustomCheckpointLoader was not able to recover.", file=log.v2)
        raise

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
    if self.extra_parent_net:
      return self.extra_parent_net.get_data_batch_dim()
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
    if self.extra_parent_net:
      return self.extra_parent_net.get_config(
        consider_global_config=consider_global_config, fallback_dummy_config=fallback_dummy_config)
    if consider_global_config:
      config = get_global_config(raise_exception=False)
      if config:
        return config
    if fallback_dummy_config:
      return Config()
    return None

  @classmethod
  def get_network_stack(cls):
    """
    :rtype: list[TFNetwork]
    """
    coll = tf.get_collection_ref("_RETURNN_network_stack")
    assert isinstance(coll, list)
    return coll

  @classmethod
  def get_current_network(cls):
    """
    :rtype: TFNetwork
    """
    coll = cls.get_network_stack()
    assert coll
    return coll[-1]

  @contextlib.contextmanager
  def register_network_scope(self):
    coll = self.get_network_stack()
    coll.append(self)
    try:
      yield
    finally:
      assert coll[-1] is self
      coll.pop(-1)


class TFNetworkParamsSerialized(object):
  """
  Holds all the params as numpy arrays, including auxiliary params.
  """
  def __init__(self, values_dict, global_train_step):
    """
    :param dict[str,dict[str,numpy.ndarray]] values_dict: dict: layer_name -> param_name -> variable numpy array
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


class LayerNotFound(Exception):
  """
  Via :func:`TFNetwork.get_layer`.
  """


def help_on_tf_exception(exception, feed_dict, meta_step_info, extern_data, file=sys.stdout):
  """
  :param tf.errors.OpError exception:
  :param dict[tf.Tensor,numpy.ndarray] feed_dict:
  :param dict[str] meta_step_info:
  :param ExternData extern_data:
  :param typing.IO[str] file:
  """
  from pprint import pprint
  import numpy
  from TFUtil import get_base_name
  print("Step meta information:", file=file)
  pprint(meta_step_info, stream=file)
  print("Feed dict:", file=file)
  if isinstance(feed_dict, dict):
    for key, value in sorted(feed_dict.items(), key=lambda item: item[0].name):
      assert isinstance(key, tf.Tensor)
      if isinstance(value, numpy.ndarray):
        v_minmax = numpy.min(value), numpy.max(value)
        info = "shape %s, dtype %s" % (value.shape, value.dtype)
        info += ", min/max %s/%s" % v_minmax
        if value.dtype.kind == "f":
          info += ", mean/stddev %s/%s" % (numpy.mean(value), numpy.std(value))
      else:
        v_minmax = -1, -1
        info = "type %r" % type(value)
      data = None
      if key.name.startswith("extern_data/"):
        data_key = get_base_name(key)
        if data_key in extern_data.data:
          data = extern_data.data[data_key]
          info += ", %s" % data
      print("  %r: %s" % (key, info), file=file)
      if data and data.sparse:
        if v_minmax[0] < 0 or v_minmax[1] >= data.dim:
          print("  WARNING, invalid label for data", data, file=file)
  else:
    pprint(feed_dict, stream=file)


class CustomCheckpointLoader:
  """
  This uses `tf.train.NewCheckpointReader`.
  It would do automatic conversions if needed, e.g. between different LSTM implementations.
  It tries to automatically resolve renames, similar to this:

    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/tools/checkpoint_convert.py

  Also see:

    https://github.com/tensorflow/tensorflow/issues/11168
    https://github.com/tensorflow/tensorflow/commit/92da8abfd35b93488ed7a55308b8f589ee23b622
    https://github.com/tensorflow/tensorflow/commit/157370e5916b85c65958ed8383ae31d727228ed7

  """

  def __init__(self, filename, saveable_params, params_prefix=""):
    """
    :param str filename: filepattern for NewCheckpointReader
    :param list[tf.Variable|tensorflow.python.training.saver.BaseSaverBuilder.SaveableObject] saveable_params:
    """
    self.saveable_params = []
    for param in saveable_params:
      custom_post_init = getattr(param, "custom_post_init", None)
      if not custom_post_init:
        self.saveable_params.append(param)
      else:
        print("Not loading pre-initialized variables %s" % param, file=log.v2)
    self.reader = tf.train.NewCheckpointReader(filename)
    self.params_prefix = params_prefix
    self.net_vars = [v for v in self.saveable_params if isinstance(v, tf.Variable)]
    self.net_saveables = [v for v in self.saveable_params if not isinstance(v, tf.Variable)]
    self.var_ckpt_names = set(self.reader.get_variable_to_shape_map())
    self.var_net_names = set([self._get_param_name(v) for v in self.saveable_params])
    self.missing_var_names = [v for v in sorted(self.var_net_names) if v not in self.var_ckpt_names]
    self.obsolete_var_names = [v for v in sorted(self.var_ckpt_names) if v not in self.var_net_names]

  def _get_param_name(self, v):
    """
    :param tf.Variable|tensorflow.python.training.saver.BaseSaverBuilder.SaveableObject v:
    :return:
    """
    if isinstance(v, tf.Variable):
      v_name = v.name[:-2]
    else:  # saveable
      v_name = v.name
    if self.params_prefix:
      assert v_name.startswith(self.params_prefix), "did not expect %r" % v
      v_name = v_name[len(self.params_prefix):]
    return v_name

  def get_variable_value_map(self):
    """
    :return: var -> numpy array
    :rtype: dict[tf.Variable,numpy.ndarray]
    """
    variable_values = {}
    if not self.missing_var_names:
      # Fast path.
      for v in self.saveable_params:
        assert isinstance(v, tf.Variable), "not yet implemented otherwise..."
        v_name = self._get_param_name(v)
        value = self.reader.get_tensor(v_name)
        variable_values[v] = value
      return variable_values

    reader = self.reader
    net_vars = self.net_vars
    net_saveables = self.net_saveables
    var_ckpt_names = self.var_ckpt_names
    var_net_names = self.var_net_names
    missing_var_names = self.missing_var_names
    obsolete_var_names = self.obsolete_var_names

    # This map_list can be extended by all the mappings in checkpoint_convert.py.
    # Old name (in checkpoint) -> new name (current variable name).
    map_list = {
      "lstm_cell/biases": "lstm_cell/bias",
      "lstm_cell/weights": "lstm_cell/kernel",
      "cudnn/params_canonical/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/bias": "lstm_block_wrapper/bias",
      "cudnn/params_canonical/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/kernel": "lstm_block_wrapper/kernel",
    }

    print("Variables to restore which are not in checkpoint:", missing_var_names, file=log.v2)

    var_name_map = {}  # type: dict[str,()->numpy.ndarray]  # current name -> value-loader

    def make_load_renamed(old_name):
      def load_old():
        return reader.get_tensor(old_name)

      return load_old

    def make_load_weights_nativelstm_to_basic(new_name):
      assert new_name.endswith("/lstm_cell/kernel")
      old_name1 = new_name[:-len("/lstm_cell/kernel")] + "/W_re"
      old_name2 = new_name[:-len("/lstm_cell/kernel")] + "/W"

      def load_native_lstm_weights():
        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        # BasicLSTM: i, j, f, o; Input: [inputs, h]
        # LstmGenericBase/NativeLstm: j, i, f, o
        # NativeLstm2: j, i, f, o
        W_re = reader.get_tensor(old_name1)  # (n_out,n_out*4)
        W_ff = reader.get_tensor(old_name2)  # (n_in,n_out*4)
        assert W_re.ndim == W_ff.ndim == 2 and W_re.shape[1] == W_ff.shape[1] and W_re.shape[1] // 4 == W_re.shape[0]
        W = numpy.concatenate([W_ff, W_re], axis=0)  # (n_in+n_out,n_out*4)
        W_j, W_i, W_f, W_o = numpy.split(W, 4, axis=1)
        W = numpy.concatenate([W_i, W_j, W_f, W_o], axis=1)
        return W

      return load_native_lstm_weights

    def make_load_bias_nativelstm_to_basic(new_name):
      assert new_name.endswith("/lstm_cell/bias")
      old_name = new_name[:-len("/lstm_cell/bias")] + "/b"

      def load_native_lstm_bias():
        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        # BasicLSTM: i, j, f, o; Input: [inputs, h]
        # LstmGenericBase/NativeLstm: j, i, f, o
        # NativeLstm2: j, i, f, o
        b = reader.get_tensor(old_name)  # (n_out*4,)
        assert b.ndim == 1
        b_j, b_i, b_f, b_o = numpy.split(b, 4, axis=0)
        b = numpy.concatenate([b_i, b_j, b_f, b_o], axis=0)
        return b

      return load_native_lstm_bias

    class make_load_cudnn_rnn:
      cudnn_postfix = "/cudnn/CudnnRNNParamsToCanonical:0"

      def __init__(self, prefix, target="lstm_block_wrapper/"):
        self.target = target
        self.keys = [target + "bias", target + "kernel"]
        self.prefix = prefix
        self.data = None

      def _load(sself):
        from TFNetworkRecLayer import RecLayer
        sself.data = RecLayer.convert_cudnn_canonical_to_lstm_block(
          reader=reader, prefix=sself.prefix, target=sself.target)

      def make_getter(self, key):
        def get():
          if self.data is None:
            self._load()
          return self.data[key]

        return get

      def get_lazy_dict(self):
        return {self.prefix + k: self.make_getter(self.prefix + k) for k in self.keys}

    # Here we try to make matches of missing vars and vars which seem to be obsolete.
    for v in missing_var_names:
      if v.endswith("/lstm_cell/kernel"):
        old_name1 = v[:-len("/lstm_cell/kernel")] + "/W_re"
        old_name2 = v[:-len("/lstm_cell/kernel")] + "/W"
        if old_name1 in obsolete_var_names and old_name2 in obsolete_var_names:
          var_name_map[v] = make_load_weights_nativelstm_to_basic(v)
      if v.endswith("/lstm_cell/bias"):
        old_name = v[:-len("/lstm_cell/bias")] + "/b"
        if old_name in obsolete_var_names:
          var_name_map[v] = make_load_bias_nativelstm_to_basic(v)
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
      for v in self.saveable_params:
        assert isinstance(v, tf.Variable), "not yet implemented otherwise..."
        v_name = self._get_param_name(v)  # current name
        if v_name in var_ckpt_names:
          value = reader.get_tensor(v_name)
        else:
          value = var_name_map[v_name]()
        variable_values[v] = value
      print("Successfully loaded all variables. Any new save will use the updated variable names.", file=log.v3)
      return variable_values

    else:
      print("Could not find mappings for these variables:", could_not_find_map_list, "var_name_map:", var_name_map,
            file=log.v3)
      print("All variables in checkpoint:", file=log.v3)
      print(reader.debug_string().decode("utf8"), file=log.v3)
      print("All variables to restore:", file=log.v3)
      for v in net_vars + net_saveables:
        print(v, file=log.v3)
      print(file=log.v3)
      print("Variables to restore which are not in checkpoint:", file=log.v3)
      for v in sorted(var_net_names):
        if v in var_ckpt_names:
          continue
        print(v, file=log.v3)
      print(file=log.v3)
      print("Variables in checkpoint which are not needed for restore:", file=log.v3)
      for v in sorted(var_ckpt_names):
        if v in var_net_names:
          continue
        print(v, file=log.v3)
      print(file=log.v3)
      print("Probably we can restore these:", file=log.v3)
      for v in sorted(var_name_map.keys()):
        print(v, file=log.v3)
      print(file=log.v3)
      raise tf.errors.NotFoundError(
        node_def=None, op=None,
        message="CustomCheckpointLoader. could_not_find_map_list: %r" % (could_not_find_map_list,))

  def load_now(self, session):
    """
    :param tf.Session session:
    :return: nothing, will assign the variables in the session
    """
    for var, value in self.get_variable_value_map().items():
      VariableAssigner(var=var).assign(value=value, session=session)

  def set_as_custom_init(self):
    var_value_map = self.get_variable_value_map()
    read_vars = set()

    def make_var_post_init(var):
      """
      :param tf.Variable var:
      :return: function
      :rtype: (tf.Session)->None
      """
      def var_post_init(session):
        """
        :param tf.Session session:
        """
        assert var not in read_vars, "Cannot initialize this twice. On purpose, to free memory."
        read_vars.add(var)
        value = var_value_map.pop(var)
        VariableAssigner(var).assign(value=value, session=session)

      return var_post_init

    for var in self.saveable_params:
      set_custom_post_init(var=var, func=make_var_post_init(var))


def set_custom_post_init(var, func):
  """
  It registers the provided `func` such that it gets called for this variable
  in TFNetwork.initialize_params().

  :param tf.Variable var:
  :param (tf.Session)->None func:
  """
  # This custom attribute is a big ugly but simple.
  # It's read in TFNetwork.initialize_params().
  var.custom_post_init = func
