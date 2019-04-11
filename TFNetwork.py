
"""
Defines the :class:`TFNetwork` and :class:`ExternData`.
"""

from __future__ import print_function

import tensorflow as tf
import sys
import numpy
import contextlib
import typing
from Log import log
from TFNetworkLayer import LayerBase, get_layer_class
from TFUtil import Data, DimensionTag, reuse_name_scope, VariableAssigner


class ExternData(object):
  """
  This holds `Data` instances for every data-key of external data from the dataset,
  i.e. the description such as shape and sparsity, etc.
  """

  def __init__(self, data=None, default_input="data", default_target="classes"):
    """
    :param None|dict[str,dict[str]] data: optional init kwargs for Data
    """
    self.data = {}  # type: typing.Dict[str,Data]
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

  @classmethod
  def data_kwargs_from_dataset_key(cls, dataset, key):
    """
    :param Dataset.Dataset dataset:
    :param str key:
    :rtype: dict[str]
    """
    if key in dataset.get_target_list():
      available_for_inference = False
    else:
      available_for_inference = True
    dim = dataset.get_data_dim(key)
    shape = [None] + list(dataset.get_data_shape(key))
    sparse = dataset.is_data_sparse(key)
    dtype = dataset.get_data_dtype(key)
    if not sparse and shape[-1] is None:
      dim = None  # overwrite. some datasets just would return some dummy int value
    return dict(
      batch_dim_axis=0, time_dim_axis=1,
      shape=shape, dim=dim, sparse=sparse, dtype=dtype,
      available_for_inference=available_for_inference)

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
      self.data[key] = Data(
        name=key, auto_create_placeholders=True,
        **self.data_kwargs_from_dataset_key(dataset=dataset, key=key))

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
      # If data.dim is None, it's ok to ignore.
      assert data.sparse == data_sparse or data.dim is None, "key %r sparse mismatch. %s" % (key, base_err_msg)
      data_dtype = dataset.get_data_dtype(key)
      assert data.dtype == data_dtype, "key %r dtype mismatch. %s" % (key, base_err_msg)
      data_dim = dataset.get_data_dim(key)
      # some datasets just would return some dummy int value, but ignore if data.dim is None
      assert data.dim == data_dim or data.dim is None, "key %r dim mismatch. %s" % (key, base_err_msg)
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
    """
    :param str name:
    :rtype: bool
    """
    return name in self.data

  def get_data(self, name):
    """
    :param str name:
    :rtype: Data
    """
    return self.data[name]

  def get_default_input_data(self):
    """
    :rtype: Data
    """
    return self.data[self.default_input]

  def get_default_target_data(self):
    """
    :rtype: Data
    """
    return self.data[self.default_target]

  def get_data_description(self):
    """
    :return: str describing the data
    :rtype: str
    """
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

  def get_all_dimension_tags(self, allow_same_feature_dim=False):
    """
    :param bool allow_same_feature_dim:
    :rtype: list[DimensionTag]
    """
    tags, _ = DimensionTag.get_all_dimension_tags(
      [data for _, data in self.get_sorted_data_items()],
      allow_same_feature_dim=allow_same_feature_dim)
    return tags


class _NetworkConstructionStack:
  """
  Used to keep the recursive construction state of :function:`TFNetwork.construct_layer`.
  """

  def __init__(self):
    self.layers = []  # type: typing.List[str]
    self.in_flat_construct_count = 0

  def append(self, layer_name):
    """
    :param str layer_name:
    """
    assert layer_name not in self.layers
    self.layers.append(layer_name)

  def remove(self, layer_name):
    """
    :param str layer_name:
    """
    self.layers.remove(layer_name)

  def flat_construct(self, initial):
    """
    :param _DelayedConstructionException initial:
    """
    self.in_flat_construct_count += 1
    queue = [initial]  # type: typing.List[_DelayedConstructionException]
    try:
      while queue:
        try:
          res = queue[-1].delayed_construction()
          if queue[-1] is initial:
            return res
          queue.pop(-1)
        except _DelayedConstructionException as delayed_exc:
          queue.append(delayed_exc)
    finally:
      self.in_flat_construct_count -= 1
    assert False, "we should not get here"


class TFNetwork(object):
  """
  The main neural network, i.e. collection of interconnected layers, i.e. computation graph with trainable params.
  """

  def __init__(self, config=None, extern_data=None, rnd_seed=None,
               train_flag=False, eval_flag=False, search_flag=False,
               parent_layer=None, parent_net=None, extra_parent_net=None,
               is_inside_rec_layer=None,
               name=None):
    """
    :param Config.Config config: only needed to init extern_data if not specified explicitly
    :param ExternData|None extern_data:
    :param int|None rnd_seed:
    :param bool|tf.Tensor train_flag: True if we want to use this model in training, False if in eval, or dynamic
    :param bool eval_flag: whether to calculate losses. if train_flag is not False, this will be set to True
    :param bool search_flag: whether we perform a beam-search. see usage
    :param TFNetworkLayer.LayerBase|None parent_layer:
    :param TFNetwork|None parent_net:
    :param TFNetwork|None extra_parent_net:
    :param bool is_inside_rec_layer: at template construction, use this
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
    self.used_data_keys = set()  # type: typing.Set[str]  # keys from extern_data
    if rnd_seed is None:
      if parent_net:
        rnd_seed = parent_net.random.randint(2 ** 31)
      else:
        rnd_seed = 42
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
    self._is_inside_rec_layer = is_inside_rec_layer
    self.extra_parent_net = extra_parent_net
    self.extra_net = None  # type: typing.Optional[TFNetwork]
    self._selected_train_layers = None
    self._construction_stack = _NetworkConstructionStack()
    self.layers_desc = {}  # type: typing.Dict[str,typing.Dict[str]]
    self.layers = {}  # type: typing.Dict[str,LayerBase]
    self.losses_dict = {}  # type: typing.Dict[str,LossHolder]
    self.total_loss = None  # type: typing.Optional[tf.Tensor]
    self.total_constraints = None  # type: typing.Optional[tf.Tensor]
    self.total_objective = None  # type: typing.Optional[tf.Tensor]
    if parent_net:
      self.global_train_step = parent_net.global_train_step
    else:
      self.global_train_step = tf.Variable(
        name="global_step", initial_value=0, dtype="int64", collections=[tf.GraphKeys.GLOBAL_STEP], trainable=False)
    self.epoch_step = None
    self.saver = None  # type: typing.Optional[tf.train.Saver]
    self.extra_vars_to_save = []  # type: typing.List[tf.Variable]
    self.recurrent = False
    self._assigner_cache = {}  # type: typing.Dict[tf.Variable,VariableAssigner]
    self.concat_sources_dropout_cache = {}  # type: typing.Dict[typing.Tuple[typing.Tuple[LayerBase,...],float,typing.Optional[typing.Tuple[typing.Optional[int],...]]],Data]  # nopep8
    self._batch_dim = None  # see get_batch_dim
    self._merge_all_summaries = None  # type: typing.Optional[tf.Tensor]

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
    net_dict = {}  # type: typing.Dict[str,typing.Dict[str]]
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
      if (name == "output"
              or layer_desc.get("loss", None)
              or layer_desc.get("is_output_layer", False)):
        self.construct_layer(net_dict, name)
    assert not self._construction_stack.layers

  def construct_extra_net(self, net_dict, layer_list, search_flag=None):
    """
    The purpose is to create another net like `self` but with different flags,
    e.g. with `search_flag = True`.
    That `extra_net` can have different losses, which will be added.
    It will not recreate any already existing layers.

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

    for layer_name in layer_list:
      # Always (re)create the specified layer in the layer_list.
      # However, any dependencies might resolve to the main net.
      self.extra_net.construct_layer(net_dict=net_dict, name=layer_name, check_existing=False)

    if self.extra_net.recurrent:
      self.recurrent = True

  def _flat_construction_enabled(self):
    """
    :return: whether to use flat construction algorithm in :func:`construct_layer`.
      Use this if you get stack overflow errors, such as:
        ``Fatal Python error: Cannot recover from stack overflow``
      or
        ``RuntimeError: maximum recursion depth exceeded``.
    :rtype: bool
    """
    return self.get_config().bool("flat_net_construction", False)

  def construct_layer(self, net_dict, name, get_layer=None, add_layer=None, check_existing=True):
    """
    :param dict[str,dict[str]] net_dict:
    :param str name: layer name
    :param ((str) -> LayerBase)|None get_layer: optional, for source layers, for transform_config_dict.
      By default, this wraps self.construct_layer().
      I.e. the name might be misleading, as this should return an existing layer,
      or construct it if it does not exist yet.
    :param ((str, LayerBase, dict) -> LayerBase) | None add_layer: by default self.add_layer
    :param bool check_existing: check self.get_layer. (self.layers will be checked in any case)
    :rtype: LayerBase
    """
    if name in self.layers:
      return self.layers[name]
    if check_existing and name != "data" and not name.startswith("data:"):
      try:
        return self.get_layer(name)
      except LayerNotFound:
        pass  # ok, we will try to construct it then
    if name in self._construction_stack.layers:
      raise NetworkConstructionDependencyLoopException(
        layer_name=name, constructing_layers=self._construction_stack.layers, net_dict=net_dict, network=self)
    if self._flat_construction_enabled():
      delayed_exc = _DelayedConstructionException(
        network=self, layer_name=name,
        other_kwargs=dict(net_dict=net_dict, get_layer=get_layer, add_layer=add_layer, check_existing=check_existing))
      if not self._construction_stack.in_flat_construct_count:
        return self._construction_stack.flat_construct(delayed_exc)
      if self._construction_stack.layers:
        raise delayed_exc
    if not get_layer:
      def get_layer(src_name):
        """
        :param str src_name:
        :rtype: LayerBase
        """
        return self.construct_layer(net_dict=net_dict, name=src_name)  # set get_layer to wrap construct_layer
    if name not in net_dict:
      layer_desc = None
      if name == "data":
        layer_desc = {"class": "source", "from": []}
      elif name.startswith("data:"):
        layer_desc = {"class": "source", "data_key": name[len("data:"):], "from": []}
      elif '/' in name:
        # it may be a hierarchical path to a sub-layer, which should have been found by get_layer()
        # but maybe it's not constructed yet, so try constructing the root layer
        root_layer = get_layer(name.split('/')[0])
        sub_layer = root_layer.get_sub_layer('/'.join(name.split('/')[1:]))  # get the sub-layer from the root-layer
        if sub_layer:
          return sub_layer
      if not layer_desc:
        raise LayerNotFound("layer %r not found in %r" % (name, self))
    else:
      layer_desc = net_dict[name]
    if not add_layer:
      add_layer = self.add_layer
    self.layers_desc[name] = layer_desc
    layer_desc = layer_desc.copy()
    class_name = layer_desc.pop("class")
    layer_class = get_layer_class(class_name)
    self._construction_stack.append(name)
    try:
      # This call would also resolve dependencies, and e.g. recursively then create them (via get_layer calls).
      layer_class.transform_config_dict(layer_desc, network=self, get_layer=get_layer)
    finally:
      self._construction_stack.remove(name)
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
    from pprint import pprint
    from Util import help_on_type_error_wrong_args
    from TFUtil import py_print
    layer_desc = self._create_layer_layer_desc(name=name, layer_desc=layer_desc)
    debug_print_layer_output_template = self.get_config().bool("debug_print_layer_output_template", False)
    debug_print_layer_output_shape = self.get_config().bool("debug_print_layer_output_shape", False)
    debug_add_check_numerics_on_output = self.get_config().bool(
      "debug_add_check_numerics_on_output", False)  # also see debug_add_check_numerics_ops
    with reuse_name_scope(layer_class.cls_get_tf_scope_name(name)), self.register_network_scope():
      try:
        if "output" not in layer_desc:
          layer_desc["output"] = layer_class.get_out_data_from_opts(**layer_desc)
        if debug_print_layer_output_template:
          print("layer %s/%r output: %r" % (self.name, name, layer_desc["output"]))
        assert isinstance(layer_desc["output"], Data)
        layer_desc["output"].sanity_check(ignore_placeholder=True)  # placeholder might be overwritten later
        layer = layer_class(**layer_desc)
        layer.post_init(layer_desc)
        layer.output.sanity_check()
      except TypeError:
        help_on_type_error_wrong_args(cls=layer_class, kwargs=list(layer_desc.keys()))
        print("TypeError creating layer %s/%r of class %s with opts:" % (self.name, name, layer_class.__name__))
        pprint(layer_desc)
        raise
      except Exception:
        print("Exception creating layer %s/%r of class %s with opts:" % (self.name, name, layer_class.__name__))
        pprint(layer_desc)
        raise
      if debug_print_layer_output_shape:
        layer.output.placeholder = py_print(
          layer.output.placeholder,
          [layer_class.cls_get_tf_scope_name(name), "shape:", str(layer.output), tf.shape(layer.output.placeholder)],
          summarize=10, name="debug_print_layer_output_shape")
      if (debug_add_check_numerics_on_output
              and layer.output.dtype.startswith("float") and not layer.allow_inf_in_output):
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
    if key in {"seq_idx", "seq_tag"} and self.parent_net:
      return self.parent_net.get_extern_data(key, mark_data_key_as_used=mark_data_key_as_used)
    if mark_data_key_as_used:
      self.used_data_keys.add(key)
    if key == "seq_idx" and key not in self.extern_data.data:
      self.extern_data.data[key] = Data(
        name="seq_idx", shape=(), dtype="int32", sparse=False, auto_create_placeholders=True)
    if key == "seq_tag" and key not in self.extern_data.data:
      self.extern_data.data[key] = Data(
        name="seq_tag", shape=(), dtype="string", auto_create_placeholders=True)
    return self.extern_data.get_data(key)

  def get_seq_tags(self, mark_data_key_as_used=True):
    """
    :param bool mark_data_key_as_used: for extern_data
    :return: tensor of shape (batch,) of dtype string, via extern_data
    :rtype: tf.Tensor
    """
    return self.get_extern_data(key="seq_tag", mark_data_key_as_used=mark_data_key_as_used).placeholder

  def get_losses_initialized(self, reduce_func=None, with_total=False):
    """
    :param ((tf.Tensor)->tf.Tensor)|None reduce_func: as in get_losses. e.g. TFUtil.identity
    :param bool with_total: whether to return total loss / constraints
    :return: loss name (e.g. "output" or "rec_layer/output" or so) -> LossHolder (initialized, i.e. layer set),
      and optionally total loss and total constraints (if with_total)
    :rtype: (dict[str,LossHolder], tf.Tensor|int|None, tf.Tensor|int|None)
    """
    if with_total:
      total_loss = 0
      total_constraints = 0
    else:
      total_loss = None
      total_constraints = None
    losses_dict = {}
    layer_items = sorted(self.layers.items())
    if self.extra_net:
      extra_name_prefix = "extra"
      if self.extra_net.search_flag and not self.search_flag:
        extra_name_prefix += "_search"
      layer_items += [
        ("%s/%s" % (extra_name_prefix, name), layer)
        for (name, layer) in sorted(self.extra_net.layers.items())]
    for name, layer in layer_items:
      assert isinstance(name, str)
      assert isinstance(layer, LayerBase)
      tf_scope_name = layer.cls_get_tf_scope_name(name=name)
      assert isinstance(layer, LayerBase)
      with reuse_name_scope("loss"):
        with reuse_name_scope(tf_scope_name):
          losses = layer.get_losses_initialized(reduce_func=reduce_func)
          for loss_obj in losses:
            assert loss_obj.name not in losses_dict, "layer %r loss name %r not unique" % (layer, loss_obj.name)
            losses_dict[loss_obj.name] = loss_obj
        if with_total:
          # Accumulate losses (outside of layer scope name).
          for loss_obj in losses:
            if loss_obj.get_loss_value_for_objective() is not None:
              if total_loss is 0:
                total_loss = loss_obj.get_loss_value_for_objective()
              else:
                total_loss += loss_obj.get_loss_value_for_objective()

      if with_total:
        with reuse_name_scope("constraints"):
          with reuse_name_scope(tf_scope_name):
            constraints = layer.get_constraints_value()
          if constraints is not None:
            if total_constraints is 0:
              total_constraints = constraints
            else:
              total_constraints += constraints

    return losses_dict, total_loss, total_constraints

  def _construct_objective(self):
    with tf.name_scope("objective"):
      losses_dict, total_loss, total_constraints = self.get_losses_initialized(with_total=True)
      self.losses_dict.clear()
      self.losses_dict.update(losses_dict)
      self.total_loss = total_loss
      self.total_constraints = total_constraints
      self.total_objective = total_loss + total_constraints
      tf.summary.scalar("loss", self.total_loss)
      tf.summary.scalar("constraints", self.total_constraints)
      tf.summary.scalar("objective", self.total_objective)

  def maybe_construct_objective(self):
    """
    Construct self.total_object.
    """
    if self.total_objective is None:
      self._construct_objective()

  def get_objective(self):
    """
    :rtype: int|tf.Tensor
    :return: 0 if no loss, or tf.Tensor, scalar. loss + constraints. will be used for the updater.
    """
    self.maybe_construct_objective()
    return self.total_objective

  def get_total_loss(self):
    """
    :rtype: int|tf.Tensor
    :return: 0 if no loss, or tf.Tensor, scalar. without constraints. will be used for the updater
    """
    self.maybe_construct_objective()
    return self.total_loss

  def get_total_constraints(self):
    """
    :rtype: int|tf.Tensor
    :return: 0 if no constraints, or tf.Tensor, scalar. will be used for the updater
    """
    self.maybe_construct_objective()
    return self.total_constraints

  def _get_all_merged_summaries(self):
    """
    :return: merged summaries, serialized string
    :rtype: tf.Tensor
    """
    # Note: This assumes that the summaries never change.
    # Both both training and evaluation on the CV dataset, this is the case.
    if self._merge_all_summaries is None:
      self._merge_all_summaries = tf.summary.merge_all()
    return self._merge_all_summaries

  def get_fetches_dict(self, config=None, should_train=None, should_eval=None, with_summary=False, with_size=False):
    """
    :param Config.Config|None config:
    :param bool|None should_train:
    :param bool|None should_eval:
    :param bool with_summary:
    :param bool with_size:
    :return: values and actions which should be calculated and executed in self.run() by the TF session for each step
    :rtype: dict[str,tf.Tensor|tf.Operation]
    """
    # Note that it is important that we do not recreate graph nodes for every call to this function.
    # Thus everything which we access here should be cached.
    import os
    import TFUtil
    if config is None:
      config = self.get_config()
    if should_train is None:
      should_train = self.train_flag
    if should_eval is None:
      should_eval = self.eval_flag

    def reduce_sum(x, name, average=False):
      """
      :param tf.Tensor x:
      :param str name:
      :param bool average:
      :return: sum(x) if horovod else x
      :rtype: tf.Tensor
      """
      if not config.is_true("use_horovod"):
        return x
      from TFUtil import global_tensor
      # noinspection PyUnresolvedReferences,PyPackageRequirements
      import horovod.tensorflow as hvd
      return global_tensor(
        lambda: hvd.allreduce(x, average=average),
        name="fetch_reduce_sum__" + name.replace(":", "__").replace("/", "_"))

    def inv_reduce_sum(x, name):
      """
      :param tf.Tensor x:
      :param str name:
      :return: reciprocal(sum(reciprocal(x))) if horovod else x
      :rtype: tf.Tensor
      """
      if not config.is_true("use_horovod"):
        return x
      from TFUtil import global_tensor
      return global_tensor(
        lambda: tf.reciprocal(reduce_sum(tf.reciprocal(x), name=name)),
        name="fetch_inv_reduce_sum__" + name.replace(":", "__").replace("/", "_"))

    d = {}
    if with_size:
      for key in self.used_data_keys:
        data = self.extern_data.get_data(key)
        for dim, v in data.size_placeholder.items():
          d["size:%s:%i" % (key, dim)] = v

    if should_train or should_eval:
      # These values are cached internally and the graph nodes are created on the first call.
      loss = self.get_objective()
      if loss is 0:
        loss = TFUtil.global_tensor(lambda: tf.constant(0.0), name="zero_loss")
      else:  # non-constant-zero loss
        assert self.losses_dict
      d["loss"] = reduce_sum(loss, name="loss", average=True)
      for loss_name, loss in self.losses_dict.items():
        if loss.get_only_on_eval() and should_train:
          continue
        if loss.get_loss_value_for_fetch() is not None:
          d["cost:%s" % loss_name] = reduce_sum(loss.get_loss_value_for_fetch(), name="cost:%s" % loss_name)
        if loss.get_error_value() is not None:
          d["error:%s" % loss_name] = reduce_sum(loss.get_error_value(), name="error:%s" % loss_name)
        d["loss_norm_factor:%s" % loss_name] = inv_reduce_sum(
          loss.get_norm_factor(), name="loss_norm_factor:%s" % loss_name)
      if with_size:
        for layer in self.layers.values():
          if layer.only_on_eval and should_train:
            continue
          # Maybe store additional size info of layer targets.
          if layer.target and layer.target.startswith("layer:"):
            target_data = layer.loss.target
            for dim, v in target_data.size_placeholder.items():
              d["size:%s:%i" % (layer.target, dim)] = v

    for layer in self.layers.values():
      for k, v in layer.stats.items():
        d["stats:%s:%s" % (layer.name, k)] = v

    if config.bool("tf_log_memory_usage", False):
      for dev in TFUtil.get_tf_list_local_devices():
        if dev.device_type != "GPU":
          # mem_usage_for_dev currently only works for GPU
          continue
        d["mem_usage:%s" % os.path.basename(dev.name.replace("/device:", "/"))] = TFUtil.mem_usage_for_dev(dev.name)

    if self.get_post_control_dependencies():
      d["post_control_dependencies"] = self.get_post_control_dependencies()

    if with_summary and self._get_all_merged_summaries() is not None:
      d["summary"] = self._get_all_merged_summaries()

    return d

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
      return output_layers[0].name
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
    if layer_name == "data" or layer_name.startswith("data:"):
      # Not created yet. Try to create it now.
      return self.construct_layer(name=layer_name, net_dict={}, check_existing=False)
    if '/' in layer_name:
      # this is probably a path to a sub-layer
      root_layer = self.get_layer(layer_name.split('/')[0])  # get the root-layer (first part of the path)
      sub_layer = root_layer.get_sub_layer('/'.join(layer_name.split('/')[1:]))  # get the sub-layer from the root-layer
      if sub_layer:  # get_sub_layer returns None by default (if sub-layer not found)
        return sub_layer
    if layer_name not in self.layers:
      raise LayerNotFound("layer %r not found in %r" % (layer_name, self))
    return self.layers[layer_name]

  def _get_all_layers(self):
    """
    :return: all layers, including extra net
    :rtype: list[LayerBase]
    """
    layers = []
    for (_, layer) in sorted(self.layers.items()):
      if layer not in layers:
        layers.append(layer)
    if self.extra_net:
      for layer in self.extra_net._get_all_layers():
        if layer not in layers:
          layers.append(layer)
    return layers

  def get_params_list(self):
    """
    :return: list of model variables, i.e. from all the layers, excluding auxiliary vars like global_step
    :rtype: list[tf.Variable]
    """
    ls = []  # type: typing.List[tf.Variable]
    for layer in self._get_all_layers():
      assert isinstance(layer, LayerBase)
      for param_name, param in sorted(layer.params.items()):
        assert isinstance(param, tf.Variable)
        if param in ls:  # could happen with reuse_params
          continue
        ls.append(param)
    return ls

  def get_saveable_param_replace_dict(self):
    """
    :return: params and saveable_param_replace resolved, union of all layers
    :rtype: dict[tf.Variable,tensorflow.python.training.saver.BaseSaverBuilder.SaveableObject]
    """
    d = {}
    for layer in self._get_all_layers():
      assert isinstance(layer, LayerBase)
      d.update(layer.saveable_param_replace)
    return d

  def get_saveable_params_list(self):
    """
    :return: list of model variables or SaveableObject, to save/restore
    :rtype: list[tf.Variable|tensorflow.python.training.saver.BaseSaverBuilder.SaveableObject]
    """
    ls = []  # type: typing.List[tf.Variable]
    for layer in self._get_all_layers():
      assert isinstance(layer, LayerBase)
      for param_name, param in sorted(layer.get_saveable_params_dict().items()):
        if param in ls:  # could happen with reuse_params
          continue
        ls.append(param)
    ls += self.get_auxiliary_params()
    ls += self.extra_vars_to_save
    return ls

  def get_trainable_params(self):
    """
    :return: list of variables
    :rtype: list[tf.Variable]
    """
    if self._selected_train_layers is None:
      self.declare_train_params()
    trainable_vars_col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    assert isinstance(trainable_vars_col, list)
    ls = []  # type: typing.List[tf.Variable]
    for layer_name in sorted(self._selected_train_layers):
      layer = self.layers[layer_name]
      assert isinstance(layer, LayerBase)
      for param_name, param in sorted(layer.params.items()):
        assert isinstance(param, tf.Variable)
        if param in trainable_vars_col:
          ls.append(param)
          trainable_vars_col.remove(param)
    if self.extra_net:
      for param in self.extra_net.get_trainable_params():
        if param not in ls:
          ls.append(param)
    return ls

  def declare_train_params(self, hidden_layer_selection=None, with_output=None):
    """
    :param list[str]|None hidden_layer_selection:
    :param bool|None with_output:
    """
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
    if self.extra_net:
      self.extra_net.declare_train_params()  # select all, currently...

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
    layers = {}  # type: typing.Dict[str,typing.Dict[str,numpy.ndarray]]
    for layer_name, layer in self.layers.items():
      assert isinstance(layer, LayerBase)
      layers[layer_name] = layer.get_param_values_dict(session)
    return layers

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
    """
    :rtype: list[tf.Variable]
    """
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
        import errno
        import time
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
    if any([layer.custom_param_importer for layer in self.layers.values()]):
      # Need to use CustomCheckpointLoader because only that handles custom_param_importer correctly.
      loader = CustomCheckpointLoader(
        filename=filename, saveable_params=self.get_saveable_params_list(), network=self)
      loader.load_now(session=session)
      return
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
          filename=filename, saveable_params=self.get_saveable_params_list(), network=self)
        if not loader.missing_var_names:
          print("Strange, nothing missing? Pre-loaded missing variables from other checkpoints?", file=log.v2)
        loader.load_now(session=session)
      except tf.errors.NotFoundError:
        print("Error, some entry is missing in the checkpoint %r: %s: %s" % (filename, type(exc), exc), file=log.v1)
        print("CustomCheckpointLoader was not able to recover.", file=log.v2)
        raise

  def print_network_info(self, name="Network"):
    """
    :param str name:
    :return: nothing, prints very brief net topology on log
    """
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
    layers = [layer for layer in layers if layer is not None]  # type: typing.List[LayerBase]
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
      """
      Wraps around `set`, to catch any `update` calls.
      """
      def update(self, others):
        """
        :param set others:
        """
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

  def is_inside_rec_layer(self):
    """
    :return: whether we are inside a :class:`RecLayer`. see :func:`get_rec_parent_layer`
    :rtype: bool
    """
    if self._is_inside_rec_layer is not None:
      return self._is_inside_rec_layer
    return self.get_rec_parent_layer() is not None

  def get_rec_parent_layer(self):
    """
    :return: if we are a subnet of a :class:`RecLayer`, will return the RecLayer instance
    :rtype: TFNetworkRecLayer.RecLayer|None
    """
    from TFNetworkRecLayer import RecLayer
    if isinstance(self.parent_layer, RecLayer):
      return self.parent_layer
    if self.parent_net:
      return self.parent_net.get_rec_parent_layer()
    return None

  def have_rec_step_info(self):
    """
    :rtype: bool
    """
    return self.get_rec_step_info(must_exist=False) is not None

  def get_rec_step_info(self, must_exist=True):
    """
    :param bool must_exist: if True, will throw exception if not available
    :rtype: TFNetworkRecLayer.RecStepInfoLayer|None
    """
    from TFNetworkRecLayer import RecStepInfoLayer, _SubnetworkRecCell
    # Fast path first. This also enables some simple debugging.
    if ":i" in self.layers and isinstance(self.layers[":i"], RecStepInfoLayer):
      return self.layers[":i"]
    rec_layer = self.get_rec_parent_layer()
    # the second condition is true if all layers have been optimized out of the rec layer
    if not rec_layer or len(rec_layer.cell.layers_in_loop) == 0:
      assert not must_exist, "%s: We expect to be the subnet of a RecLayer, but we are not." % self
      return None
    assert isinstance(rec_layer.cell, _SubnetworkRecCell)
    step_info_layer = rec_layer.cell.net.layers[":i"]
    assert isinstance(step_info_layer, RecStepInfoLayer)
    return step_info_layer

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

  @staticmethod
  def register_post_control_dependencies(deps):
    """
    Will register the control dependencies
    or globally for a session run on this network.
    This can e.g. be called inside `self.post_init`.
    We use UPDATE_OPS, as that is also e.g. used by batchnorm. See:
      https://github.com/tensorflow/tensorflow/issues/1122

    :param list[tf.Tensor|tf.Operation] deps:
    :return: nothing
    """
    ls = tf.get_collection_ref(tf.GraphKeys.UPDATE_OPS)
    assert isinstance(ls, list)
    ls.extend(deps)

  @staticmethod
  def get_post_control_dependencies():
    """
    :rtype: list[tf.Operation]
    """
    return tf.get_collection(tf.GraphKeys.UPDATE_OPS)

  @classmethod
  def get_network_stack(cls):
    """
    :rtype: list[TFNetwork]
    """
    from TFUtil import CollectionKeys
    coll = tf.get_collection_ref(CollectionKeys.RETURNN_NET_STACK)
    assert isinstance(coll, list)
    return coll

  @classmethod
  def get_current_network(cls, must_exist=True):
    """
    :param bool must_exist:
    :rtype: TFNetwork|None
    """
    coll = cls.get_network_stack()
    if must_exist:
      assert coll
    elif not coll:
      return None
    return coll[-1]

  @contextlib.contextmanager
  def register_network_scope(self):
    """
    Registers a ref to this network inside the current TF computation graph.
    """
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


class LossHolder:
  """
  This object just keeps a reference to the loss/error value,
  and does the necessary logic to collect it, and also the normalization logic.
  Every new computation (nodes in the computation graph) must be constructed on demand,
  to allow first to collect all possible losses without calculating them,
  and then calculating them in the right context (e.g. inside a while_loop, or so).
  """

  def __init__(self, name, loss, layer_output, reduce_func=None,
               layer=None, loss_value=None, error_value=None,
               norm_factor=None,
               only_on_eval=None,
               network=None):
    """
    After construction, you should call init() before usage, in case you do not provide `layer` here.

    :param str name: The name uniquely identifies the loss. Earlier, this was the same as the layer name.
      This is still true for simple cases,
      but for losses coming from a subnetwork or other extended losses,
      it can be something else.
      It could look like "output", or "output/sublayer".
    :param LayerBase layer:
      We can always point to a layer where this comes from (either in the subnet, or the parent layer).
    :param Data layer_output: template describing the layer output
    :param TFNetwork network: for which network to create this LossHolder. might be different from layer.network
    :param TFNetworkLayer.Loss loss:
    :param ((tf.Tensor)->tf.Tensor)|None reduce_func: if given, will overwrite the reduce func for the loss.
      By default, every loss_value and error_value is a scalar
      (sum or average over the batches, and over the frames for frame-wise losses).
      However, if you provide reduce_func = TFUtil.identity, you can get the unreduced tensor.
    :param tf.Tensor|None loss_value:
    :param tf.Tensor|None error_value:
    :param tf.Tensor norm_factor:
    :param bool only_on_eval:
    """
    if layer and not network:
      network = layer.network
    if layer and only_on_eval is None:
      only_on_eval = layer.only_on_eval
    assert name and loss and network  # these must always be provided
    if layer:
      assert isinstance(layer, LayerBase)
    if reduce_func:
      loss.reduce_func = reduce_func
    self.name = name
    self.loss = loss
    self.layer_output = layer_output
    self.reduce_func = reduce_func
    self._network = network
    self._layer = layer
    self._is_prepared = False
    self._loss_value = loss_value
    self._loss_value_for_fetch = None  # via self._prepare
    self._loss_value_for_objective = None  # via self._prepare
    self._error_value = error_value
    self._norm_factor = norm_factor
    self._only_on_eval = only_on_eval

  def __repr__(self):
    return "<LossHolder name=%r loss=%r>" % (self.name, self.loss)

  def init(self, layer):
    """
    It will just set the layer.
    The `LossHolder` is initialized if the layer is set.

    :param LayerBase layer:
    :return: self
    :rtype: LossHolder
    """
    self._layer = layer
    if self._only_on_eval is None:
      self._only_on_eval = layer.only_on_eval
    if self._network is None:
      self._network = layer.network
    return self

  def get_layer(self):
    """
    :return: layer. assumes that it is set
    :rtype: LayerBase
    """
    assert self._layer, "call init()"
    return self._layer

  def get_only_on_eval(self):
    """
    :return: only_on_eval flag. assumes that it is set
    :rtype: bool
    """
    assert self._only_on_eval is not None, "call init()"
    return self._only_on_eval

  def get_tf_name(self):
    """
    :return: name which can be used for a TF op, thus contains no "/" or other special chars
    :rtype: str
    """
    return LayerBase.cls_get_tf_scope_name(self.name.replace("/", "_"))

  def get_loss_value(self):
    """
    :return: loss value. scalar
    :rtype: tf.Tensor|None
    """
    self._prepare()
    return self._loss_value

  def get_loss_value_for_fetch(self):
    """
    :return: loss value for fetch. scalar. same as loss_value, but maybe with additional checks
    :rtype: tf.Tensor|None
    """
    self._prepare()
    return self._loss_value_for_fetch

  def get_loss_value_for_objective(self):
    """
    :return: loss value for objective. scalar. might be scaled (scale) and/or normalized (use_normalized_loss)
    :rtype: tf.Tensor|None
    """
    self._prepare()
    return self._loss_value_for_objective

  def get_error_value(self):
    """
    :return: error value for fetch. scalar
    :rtype: tf.Tensor|None
    """
    self._prepare()
    return self._error_value

  def get_norm_factor(self):
    """
    :return: norm factor for loss and error. scalar
    :rtype: tf.Tensor
    """
    self._prepare()
    return self._norm_factor

  def _normalized_value_per_seq(self, value, per_pos=False):
    """
    :param tf.Tensor|None value: (batch*time,) or (time*batch,)
    :param bool per_pos: one value per time position
    :return: if per_pos return (batch,time) else (batch,) or None if loss is None
    :rtype: tf.Tensor|None
    """
    if value is None:
      return None

    if per_pos:
      value = tf.reshape(value, tf.shape(self.loss.output.placeholder)[:2])  # (batch,time) or (time,batch)

      # We want output of the form (B,T)
      if self.loss.output.time_dim_axis == 0:
        from TFUtil import swapaxes
        value = swapaxes(value, 0, 1)  # resulting in (B,T,...)

      return value
    else:
      return self.loss.reduce_to_batch(value, normalize=True)

  def get_normalized_loss_value_per_seq(self, per_pos=False):
    """
    :param bool per_pos: one value per time position
    :return: if per_pos return (batch,time) else (batch,) or None if loss is None
    :rtype: tf.Tensor|None
    """
    self._prepare()
    return self._normalized_value_per_seq(self._loss_value, per_pos=per_pos)

  def get_normalized_error_value_per_seq(self, per_pos=False):
    """
    :param bool per_pos: one value per time position
    :return: if per_pos return (batch,time) else (batch,) or None if error is None
    :rtype: tf.Tensor|None
    """
    self._prepare()
    return self._normalized_value_per_seq(self._error_value, per_pos=per_pos)

  def _tf_summary(self):
    """
    This gets called inside a loss name scope of the layer.

    :return: nothing, will use tf.summary
    """
    if self._network.parent_net:
      return  # skip summaries. the root net should also do this
    name = self.get_tf_name()
    if self._loss_value is not None:
      # a loss value is typically a scalar but there are cases of sequence or position wise loss values (e.g. if
      #   the eval_output_file_per_seq option is used)
      tf.summary.tensor_summary("loss_%s" % name, self._loss_value * self._norm_factor)
      if self._network.get_config().bool("calculate_exp_loss", False):
        tf.summary.tensor_summary("exp_loss_%s" % name, tf.exp(self._loss_value * self._norm_factor))
    if self._error_value is not None:
      tf.summary.tensor_summary("error_%s" % name, self._error_value * self._norm_factor)

  def _prepare(self):
    """
    This gets called inside a loss name scope of the layer.

    :return: nothing, will prepare
    """
    if self._is_prepared:
      return
    assert self._layer, "call init()"
    self.loss.init_by_layer(layer=self._layer, layer_output_template=self.layer_output)
    if self._loss_value is None and self._error_value is None:
      with reuse_name_scope("loss"):
        if self._only_on_eval:
          # noinspection PyProtectedMember
          self._loss_value = self._layer._cond_only_on_eval_opt(self.loss.get_value, default_value=0.0)
        else:
          self._loss_value = self.loss.get_value()
      with reuse_name_scope("error"):
        if self._only_on_eval:
          # noinspection PyProtectedMember
          self._error_value = self._layer._cond_only_on_eval_opt(self.loss.get_error, default_value=0.0)
        else:
          self._error_value = self.loss.get_error()
      assert self._loss_value is not None or self._error_value is not None, (
        "layer %r loss %r return None for loss and error" % (self._layer, self.loss))
    if self._norm_factor is None:
      self._norm_factor = self.loss.get_normalization_factor()
    self._tf_summary()
    loss_value = self._loss_value
    if loss_value is not None:
      if self._network.get_config().bool("debug_add_check_numerics_on_output", False):
        print("debug_add_check_numerics_on_output: add for layer loss %r: %r" % (
          self._layer.name, self._layer.output.placeholder))
        from TFUtil import identity_with_check_numerics
        loss_value = identity_with_check_numerics(
          loss_value, name="%s_identity_with_check_numerics_loss" % self.get_tf_name())
    self._loss_value_for_fetch = loss_value
    if self.loss.scale != 1 and loss_value is not None:
      if not self.loss.scale:
        loss_value = None  # scale 0 means to not use this loss
      else:
        loss_value *= self.loss.scale
    if self.loss.use_normalized_loss and loss_value is not None:
      loss_value *= self._norm_factor
    self._loss_value_for_objective = loss_value
    self._is_prepared = True

  def copy_new_base(self, name=None, layer=None, network=None, reduce_func=None):
    """
    :param LayerBase layer:
    :param TFNetwork network:
    :param str name:
    :param ((tf.Tensor)->tf.Tensor)|None reduce_func:
    :return: new copy of LossHolder
    :rtype: LossHolder
    """
    if not layer:
      layer = self._layer
    if not network:
      network = self._network
    if not name:
      name = self.name
    loss_value = self._loss_value
    error_value = self._error_value
    if reduce_func is None:
      reduce_func = self.loss.reduce_func
    if reduce_func and reduce_func != self.loss.reduce_func:
      # Must recreate those.
      loss_value = None
      error_value = None
    return LossHolder(
      name=name, layer=layer, layer_output=self.layer_output, network=network,
      loss=self.loss, reduce_func=reduce_func,
      loss_value=loss_value, error_value=error_value,
      norm_factor=self._norm_factor, only_on_eval=self._only_on_eval)


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


class _DelayedConstructionException(Exception):
  """
  When we want to do a flat construction.
  """
  def __init__(self, network, layer_name, other_kwargs):
    """
    :param TFNetwork network:
    :param str layer_name:
    :param dict[str] other_kwargs:
    """
    self.network = network
    self.layer_name = layer_name
    self.other_kwargs = other_kwargs

  def __repr__(self):
    return "%s(layer_name=%r)" % (self.__class__.__name__, self.layer_name)

  def delayed_construction(self):
    """
    Call :func:`TFNetwork.construct_layer` again now.

    :rtype: LayerBase
    """
    print("Delayed flat layer construction:", self.layer_name, file=log.v5)
    return self.network.construct_layer(name=self.layer_name, **self.other_kwargs)


class LayerNotFound(Exception):
  """
  Via :func:`TFNetwork.get_layer`.
  """


# noinspection PyUnusedLocal
def help_on_tf_exception(exception, feed_dict, meta_step_info, extern_data, file=sys.stdout):
  """
  :param tf.errors.OpError|BaseException exception:
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
        info = "shape %s, dtype %s" % (value.shape, value.dtype)
        if value.size > 0:
          v_minmax = numpy.min(value), numpy.max(value)
          info += ", min/max %s/%s" % v_minmax
          if value.dtype.kind == "f":
            info += ", mean/stddev %s/%s" % (numpy.mean(value), numpy.std(value))
        else:
          v_minmax = 0, 0
          info += ", EMPTY"
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

  def __init__(self, filename, saveable_params, params_prefix="", load_if_prefix="", ignore_missing=False,
               network=None):
    """
    :param str filename: filepattern for NewCheckpointReader
    :param list[tf.Variable|tensorflow.python.training.saver.BaseSaverBuilder.SaveableObject] saveable_params:
    :param str params_prefix: expect that all vars in saveable_params have this prefix, and remove it
    :param str load_if_prefix: if given, only load variables with a name containing this string.
      the variables in the file are expected to have the same name but without this string.
    :param bool ignore_missing: any vars in the model, which are not found in the checkpoint, will be ignored.
      however, if there is no single var in the checkpoint, this is still an error.
    :param TFNetwork network:
    """
    self.filename = filename
    self.network = network
    self.ignore_missing = ignore_missing
    self.params_prefix = params_prefix
    self.load_if_prefix = load_if_prefix
    self.saveable_params = []
    for param in saveable_params:
      custom_post_init = getattr(param, "custom_post_init", None)
      if custom_post_init:
        print("Not loading pre-initialized variables %s" % param, file=log.v2)
        continue
      if load_if_prefix and self._get_param_name(param, assert_load_if_prefix_match=False) is None:
        continue
      self.saveable_params.append(param)
    assert self.saveable_params, "no saveable vars"
    self.reader = tf.train.NewCheckpointReader(filename)
    self.net_vars = [v for v in self.saveable_params if isinstance(v, tf.Variable)]
    self.net_saveables = [v for v in self.saveable_params if not isinstance(v, tf.Variable)]
    # All variables in the checkpoint:
    self.var_ckpt_names = set(self.reader.get_variable_to_shape_map())
    # All variables of the model to be loaded:
    self.var_net_names = set([self._get_param_name(v) for v in self.saveable_params])
    # Model variables missing in the checkpoint:
    self.missing_var_names = [v for v in sorted(self.var_net_names) if v not in self.var_ckpt_names]
    # Checkpoint variables which are not used in this model:
    self.obsolete_var_names = [v for v in sorted(self.var_ckpt_names) if v not in self.var_net_names]
    self.custom_param_importers = [
      self.CustomParamImporter(layer=layer, checkpoint_loader=self)
      for layer in network.layers.values() if layer.custom_param_importer] if network else []

  def __repr__(self):
    keys = ["filename", "params_prefix", "load_if_prefix", "ignore_missing", "network"]
    return "%s(%s)" % (
      self.__class__.__name__,
      ", ".join(["%s=%r" % (key, getattr(self, key, "<unset>")) for key in keys]))

  class CustomParamImporter:
    """
    Helper class for custom param loading.
    """

    def __init__(self, layer, checkpoint_loader):
      """
      :param LayerBase layer:
      :param CustomCheckpointLoader checkpoint_loader:
      """
      self.layer = layer
      self.prefix_param_name = layer.get_absolute_name_scope_prefix()
      self.checkpoint_param_names = []
      prefix = self.prefix_param_name
      # Collect checkpoint params, and remove them from the lists.
      for name in list(checkpoint_loader.var_ckpt_names):
        if name.startswith(prefix):
          checkpoint_loader.var_ckpt_names.remove(name)
          self.checkpoint_param_names.append(name[len(prefix):])
      # Don't treat any of them as missing.
      for name in list(checkpoint_loader.missing_var_names):
        if name.startswith(prefix):
          checkpoint_loader.missing_var_names.remove(name)
      # Also not obsolete.
      for name in list(checkpoint_loader.obsolete_var_names):
        if name.startswith(prefix):
          checkpoint_loader.obsolete_var_names.remove(name)
      # When we load the params, we need this.
      self.reader = checkpoint_loader.reader
      self.assigned = False

    def __repr__(self):
      return "<CustomParamImporter %r on layer %r>" % (self.layer.custom_param_importer, self.layer.name)

    # noinspection PyUnusedLocal
    def assign_var(self, var, session):
      """
      :param tf.Variable var:
      :param tf.Session session:
      """
      # This function gets called for every param of the layer.
      # However, the underlying custom_param_importer API
      # will assign all the layer params together,
      # so we want to call it exactly once.
      if self.assigned:
        return
      self.assigned = True
      values_dict = {
        name: self.reader.get_tensor(self.prefix_param_name + name)
        for name in self.checkpoint_param_names}
      self.reader = None  # Allow GC now, we do not need it anymore.
      print("Custom param import of layer %r with original params %r." % (
        self.layer, sorted(values_dict.keys())), file=log.v3)
      self.layer.set_param_values_by_dict(values_dict=values_dict, session=session)

  def _find_custom_param_importer(self, v_name):
    """
    :param str v_name:
    :rtype: CustomParamImporter|None
    """
    for importer in self.custom_param_importers:
      if v_name.startswith(importer.prefix_param_name):
        return importer
    return None

  def _get_param_name(self, v, assert_load_if_prefix_match=True):
    """
    :param tf.Variable|tensorflow.python.training.saver.BaseSaverBuilder.SaveableObject v:
    :param bool assert_load_if_prefix_match: only has an effect with self.load_if_prefix.
      if True, auto resolve load_if_prefix. if False and no match, return None.
    :return: var name. self.params_prefix removed if given
    :rtype: str|None
    """
    if isinstance(v, tf.Variable):
      v_name = v.name[:-2]
    else:  # saveable
      v_name = v.name
    if self.params_prefix:
      assert v_name.startswith(self.params_prefix), "did not expect %r" % v
      v_name = v_name[len(self.params_prefix):]
    if self.load_if_prefix:
      if self.load_if_prefix not in v_name:
        assert not assert_load_if_prefix_match, "var %r not expected with load_if_prefix %r" % (v, self.load_if_prefix)
        return None
      v_name = v_name.replace(self.load_if_prefix, "")
    return v_name

  class VariableValue:
    """
    Helper to assign some variable.
    """

    def __init__(self, value=None, custom_param_importer=None):
      """
      :param numpy.ndarray|None value:
      :param CustomCheckpointLoader.CustomParamImporter custom_param_importer:
      """
      assert value is not None or custom_param_importer
      self.value = value
      self.custom_param_importer = custom_param_importer

    def assign_var(self, var, session):
      """
      :param tf.Variable var:
      :param tf.Session session:
      """
      if self.value is not None:
        VariableAssigner(var=var).assign(value=self.value, session=session)
      else:
        self.custom_param_importer.assign_var(var=var, session=session)

  def get_variable_value_map(self):
    """
    :return: var -> numpy array
    :rtype: dict[tf.Variable,CustomCheckpointLoader.VariableValue]
    """
    variable_values = {}
    if not self.missing_var_names and not self.custom_param_importers:
      # Fast path.
      for v in self.saveable_params:
        assert isinstance(v, tf.Variable), "not yet implemented otherwise..."
        v_name = self._get_param_name(v)
        value = self.reader.get_tensor(v_name)
        variable_values[v] = self.VariableValue(value=value)
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
      "cudnn/params_canonical/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/bias": "lstm_fused_cell/bias",
      "cudnn/params_canonical/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/kernel": "lstm_fused_cell/kernel",
    }

    print("Variables to restore which are not in checkpoint:", missing_var_names, file=log.v2)

    var_name_map = {}  # type: typing.Dict[str,typing.Callable[[],numpy.ndarray]]  # current name -> value-loader

    # noinspection PyShadowingNames
    def make_load_renamed(old_name):
      """
      :param str old_name:
      :rtype: () -> numpy.ndarray
      """
      def load_old():
        """
        :rtype: numpy.ndarray
        """
        return reader.get_tensor(old_name)

      return load_old

    def make_load_weights_nativelstm_to_basic(new_name):
      """
      :param str new_name:
      :rtype: ()->numpy.ndarray
      """
      assert new_name.endswith("/lstm_cell/kernel")
      # noinspection PyShadowingNames
      old_name1 = new_name[:-len("/lstm_cell/kernel")] + "/W_re"
      # noinspection PyShadowingNames
      old_name2 = new_name[:-len("/lstm_cell/kernel")] + "/W"

      def load_native_lstm_weights():
        """
        :rtype: numpy.ndarray
        """
        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        # BasicLSTM: i, j, f, o; Input: [inputs, h]
        # LstmGenericBase/NativeLstm: j, i, f, o
        # NativeLstm2: j, i, f, o
        w_re = reader.get_tensor(old_name1)  # (n_out,n_out*4)
        w_ff = reader.get_tensor(old_name2)  # (n_in,n_out*4)
        assert w_re.ndim == w_ff.ndim == 2 and w_re.shape[1] == w_ff.shape[1] and w_re.shape[1] // 4 == w_re.shape[0]
        w = numpy.concatenate([w_ff, w_re], axis=0)  # (n_in+n_out,n_out*4)
        w_j, w_i, w_f, w_o = numpy.split(w, 4, axis=1)
        w = numpy.concatenate([w_i, w_j, w_f, w_o], axis=1)
        return w

      return load_native_lstm_weights

    def make_load_bias_nativelstm_to_basic(new_name):
      """
      :param str new_name:
      :rtype: ()->numpy.ndarray
      """
      assert new_name.endswith("/lstm_cell/bias")
      # noinspection PyShadowingNames
      old_name = new_name[:-len("/lstm_cell/bias")] + "/b"

      def load_native_lstm_bias():
        """
        :rtype: numpy.ndarray
        """
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

    class MakeLoadCudnnRnn:
      """
      Helper to load the CuDNN params.
      """

      cudnn_postfix = "/cudnn/CudnnRNNParamsToCanonical:0"

      def __init__(self, prefix, target="lstm_block_wrapper/"):
        self.target = target
        self.keys = [target + "bias", target + "kernel"]
        self.prefix = prefix
        self.data = None  # type: typing.Optional[typing.Dict[str,numpy.ndarray]]

      # noinspection PyMethodParameters
      def _load(sself):
        from TFNetworkRecLayer import RecLayer
        sself.data = RecLayer.convert_cudnn_canonical_to_lstm_block(
          reader=reader, prefix=sself.prefix, target=sself.target)

      def make_getter(self, key):
        """
        :param str key:
        :rtype: ()->numpy.ndarray
        """
        def get():
          """
          :rtype: numpy.ndarray
          """
          if self.data is None:
            self._load()
          return self.data[key]

        return get

      def get_lazy_dict(self):
        """
        :rtype: dict[str,()->numpy.ndarray]
        """
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
      if v.endswith(MakeLoadCudnnRnn.cudnn_postfix):
        var_name_map.update(
          MakeLoadCudnnRnn(prefix=v[:-len(MakeLoadCudnnRnn.cudnn_postfix) + 1]).get_lazy_dict())

    could_not_find_map_list = [v for v in missing_var_names if v not in var_name_map]
    if self.ignore_missing or not could_not_find_map_list:
      # We can restore all.
      print("We found these corresponding variables in the checkpoint:", var_name_map, file=log.v2)
      print("Custom param importers:", self.custom_param_importers, file=log.v2)
      print("Loading now...", file=log.v3)
      # Similar: from tensorflow.contrib.framework.python.ops import assign_from_checkpoint
      for v in self.saveable_params:
        v_name = self._get_param_name(v)  # current name
        custom_importer = self._find_custom_param_importer(v_name)
        if custom_importer:
          variable_values[v] = self.VariableValue(custom_param_importer=custom_importer)
        elif v_name in var_ckpt_names:
          variable_values[v] = self.VariableValue(value=reader.get_tensor(v_name))
        else:
          if self.ignore_missing and v_name not in var_name_map:
            print(
              "Warning, did not find match for var %r (%r, params_prefix %r, load_if_prefix %r) in checkpoint %r." % (
                v, v_name, self.params_prefix, self.load_if_prefix, self.filename), file=log.v3)
            continue
          variable_values[v] = self.VariableValue(value=var_name_map[v_name]())
      assert variable_values, "no vars to load; saveable vars are %r. load_if_prefix %r." % (
        self.saveable_params, self.load_if_prefix)
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
      if not var_name_map:
        print("(None)", file=log.v3)
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
      value.assign_var(var=var, session=session)

  def set_as_custom_init(self):
    """
    Make sure that this loader is used during initialization.
    """
    var_value_map = self.get_variable_value_map()
    read_vars = set()

    # noinspection PyShadowingNames
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
        value.assign_var(var=var, session=session)

      return var_post_init

    for var in self.saveable_params:
      if self.ignore_missing and var not in var_value_map:
        continue
      if self.load_if_prefix:
        print("%s registered for pre-loading via prefix %r." % (var.name, self.load_if_prefix), file=log.v2)
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
