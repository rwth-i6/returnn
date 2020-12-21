
"""
Defines the :class:`TFNetwork` and :class:`ExternData`.
"""

from __future__ import print_function

import tensorflow as tf
import sys
import re
import numpy
import contextlib
import typing
from returnn.log import log
from returnn.tf.layers.basic import LayerBase, get_layer_class
import returnn.tf.compat as tf_compat
import returnn.tf.util.basic as tf_util
from returnn.tf.util.basic import Data, DimensionTag, reuse_name_scope, VariableAssigner


class ExternData(object):
  """
  This holds `Data` instances for every data-key of external data from the dataset,
  i.e. the description such as shape and sparsity, etc.
  """

  def __init__(self, data=None, default_input="data", default_target="classes"):
    """
    :param None|dict[str,dict[str]] data: optional init kwargs for Data
    """
    self._config = None  # type: typing.Optional["Config.Config"]
    self.data = {}  # type: typing.Dict[str,Data]
    self.default_input = default_input
    self.default_target = default_target
    if data:
      self.register_data_from_dict(data)
    self.extra_added_keys = set()  # set[str]

  def __repr__(self):
    return "<ExternData data=%r>" % self.data

  def init_from_config(self, config, auto_create_placeholders=True):
    """
    :param Config.Config config:
    :param bool auto_create_placeholders:
    """
    self._config = config
    from returnn.network_description import LayerNetworkDescription
    data_dims = LayerNetworkDescription.tf_extern_data_types_from_config(config)
    for key, init_args in data_dims.items():
      # In Returnn with Theano, we usually have the shape (time,batch,feature).
      # In TensorFlow, the default is (batch,time,feature).
      # This is also what we use here, i.e.:
      # batch_dim_axis=0, time_dim_axis=1. See TFEngine.DataProvider._get_next_batch().
      self.data[key] = Data(name=key, auto_create_placeholders=auto_create_placeholders, **init_args)
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

  def init_from_dataset(self, dataset, auto_create_placeholders=True):
    """
    :param Dataset.Dataset dataset:
    :param bool auto_create_placeholders:
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
        name=key, auto_create_placeholders=auto_create_placeholders,
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
    try:
      return self.data[name]
    except KeyError:
      config_extern_data = "<unknown>"
      if self._config and self._config.has("extern_data"):
        config_extern_data = self._config.opt_typed_value("extern_data")
      raise KeyError(
        "ExternData: unknown key %r. available keys: %s. config: %s" % (
          name, list(self.data.keys()), config_extern_data))

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
      dict(allow_same_feature_dim=allow_same_feature_dim))
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
               absolute_name_prefix=None, name=None):
    """
    :param Config.Config config: only needed to init extern_data if not specified explicitly
    :param ExternData|None extern_data:
    :param int|None rnd_seed:
    :param bool|tf.Tensor train_flag: True if we want to use this model in training, False if in eval, or dynamic
    :param bool eval_flag: whether to calculate losses. if train_flag is not False, this will be set to True
    :param bool search_flag: whether we perform a beam-search. see usage
    :param returnn.tf.layers.base.LayerBase|None parent_layer:
    :param TFNetwork|None parent_net:
    :param TFNetwork|None extra_parent_net: we are on the same level (not really a child),
      but an "extra" net of extra_parent_net
    :param bool is_inside_rec_layer: at template construction, use this
    :param str|None absolute_name_prefix:
    :param str name: only for debugging
    """
    if not name:
      from returnn.util.basic import try_get_caller_name
      name = "<network via %s>" % try_get_caller_name(fallback="<unknown>")
    self.name = name
    if absolute_name_prefix:
      assert absolute_name_prefix.endswith("/")
    self._absolute_name_prefix = absolute_name_prefix
    if not parent_net and parent_layer:
      parent_net = parent_layer.network
    if not config and parent_net:
      config = parent_net._config
    if extern_data is None:
      if not config:
        from returnn.config import get_global_config
        config = get_global_config()
      extern_data = ExternData()
      extern_data.init_from_config(config)
    self.extern_data = extern_data
    self._config = config
    self.used_data_keys = set()  # type: typing.Set[str]  # keys from extern_data
    if rnd_seed is None:
      if parent_net:
        rnd_seed = parent_net.random.randint(2 ** 31)
      elif extra_parent_net:
        rnd_seed = extra_parent_net.random.randint(2 ** 31)
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
    self.extra_nets = {}  # type: typing.Dict[str,TFNetwork]
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
    elif extra_parent_net:
      self.global_train_step = extra_parent_net.global_train_step
    else:
      # Reuse mostly because some of the test cases currently work that way.
      with tf_compat.v1.variable_scope(
            tf_compat.v1.get_variable_scope(), reuse=getattr(tf_compat.v1, "AUTO_REUSE", None)):
        self.global_train_step = tf_compat.v1.get_variable(
          name="global_step", shape=(), dtype=tf.int64, initializer=tf_compat.v1.zeros_initializer(tf.int64),
          collections=[tf_compat.v1.GraphKeys.GLOBAL_STEP], trainable=False)
    self.epoch_step = None
    self.saver = None  # type: typing.Optional[tf.compat.v1.train.Saver]
    self.extra_vars_to_save = []  # type: typing.List[tf.Variable]
    self.recurrent = False
    self._assigner_cache = {}  # type: typing.Dict[tf.Variable,VariableAssigner]
    self.concat_sources_dropout_cache = {}  # type: typing.Dict[typing.Tuple[typing.Tuple[LayerBase,...],float,typing.Optional[typing.Tuple[typing.Optional[int],...]]],Data]  # nopep8
    self._batch_dim = None  # see get_data_batch_dim
    self._merge_all_summaries = None  # type: typing.Optional[tf.Tensor]
    self._graph_reset_callbacks = []  # type: typing.List[typing.Callable]
    self._run_opts = {}  # type: typing.Dict[str]
    self._run_finished_callbacks = []  # type: typing.List[typing.Callable]
    self._map_search_beam_to_search_choices = {}  # type: typing.Dict[tf_util.SearchBeam,"returnn.tf.layers.base.SearchChoices"]  # nopep8

  def __repr__(self):
    s = "TFNetwork %r" % self.name
    if self.parent_layer:
      s += " parent_layer=%r" % self.parent_layer
    elif self.parent_net:
      s += " parent_net=%r" % self.parent_net
    if self.extra_nets:
      s += " extra_nets=%r" % self.extra_nets
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
    if self.extra_parent_net:
      return self.extra_parent_net.get_root_network()
    return self

  def get_absolute_name_scope_prefix(self):
    """
    :return: TF scope name, always with "/" at the end, or ""
    :rtype: str
    """
    if self.parent_layer:
      return self.parent_layer.get_absolute_name_scope_prefix()
    if self.parent_net:
      return self.parent_net.get_absolute_name_scope_prefix()
    if self.extra_parent_net:
      return self.extra_parent_net.get_absolute_name_scope_prefix()
    return ""

  def get_absolute_name_prefix(self):
    """
    :return: name, always with "/" at the end, or ""
    :rtype: str
    """
    if self._absolute_name_prefix is not None:
      return self._absolute_name_prefix
    if self.parent_layer:
      return self.parent_layer.get_absolute_name() + "/"
    if self.parent_net:
      return self.parent_net.get_absolute_name_prefix()
    if self.extra_parent_net:
      prefixes = {net: prefix for (prefix, net) in self.extra_parent_net.extra_nets.items()}
      my_prefix = ("%s:" % prefixes[self]) if self in prefixes else ""
      return self.extra_parent_net.get_absolute_name_prefix() + my_prefix
    return ""

  def construct_from_dict(self, net_dict):
    """
    :param dict[str,dict[str]] net_dict:
    """
    self.layers_desc.update(net_dict)
    for name, layer_desc in sorted(net_dict.items()):
      assert isinstance(name, str)
      if name.startswith("#"):  # ignore this
        continue
      assert isinstance(layer_desc, dict)
      if layer_desc.get("register_as_extern_data"):
        self.construct_layer(net_dict, name)
    for name, layer_desc in sorted(net_dict.items()):
      assert isinstance(name, str)
      if name.startswith("#"):  # ignore this
        continue
      assert isinstance(layer_desc, dict)
      if layer_desc.get("only_on_search") and not self.search_flag:
        continue
      if layer_desc.get("only_on_eval") and not self.eval_flag:
        continue
      if (name == "output" or name.endswith(":output")
              or layer_desc.get("loss", None)
              or layer_desc.get("is_output_layer", False)):
        self.construct_layer(net_dict, name)
    assert not self._construction_stack.layers

  # Currently this pattern is very simple.
  # This pattern might be extended, when we want to make it more flexible.
  _extra_layer_name_prefix_pattern = re.compile("^(extra(\\.[A-Za-z0-9_.]+)?):")

  def _get_extra_net(self, search_flag=None, net_name=None, prefix_name=None, auto_create=True):
    """
    :param bool|None search_flag:
    :param str|None net_name:
    :param str|None prefix_name:
    :param bool auto_create:
    :return: (net, prefix_name)
    :rtype: (TFNetwork|None,str)
    """
    if search_flag is None and prefix_name:
      search_flag = ".search" in prefix_name  # currently very simple...
    if not prefix_name:
      assert search_flag is not None
      prefix_name = "extra.search" if search_flag else "extra"
    if prefix_name and not net_name:
      net_name = prefix_name
    if self.extra_parent_net:  # Also, only the root can have other extra nets.
      assert not auto_create
      return None, prefix_name
    if prefix_name not in self.extra_nets:
      if not auto_create:
        return None, prefix_name
      self.extra_nets[prefix_name] = TFNetwork(
        config=self._config, extern_data=self.extern_data, name=net_name,
        rnd_seed=self.random.randint(2 ** 31),
        train_flag=self.train_flag, eval_flag=self.eval_flag,
        search_flag=search_flag if search_flag is not None else self.search_flag,
        extra_parent_net=self)
    if search_flag is not None:
      assert self.extra_nets[prefix_name].search_flag == search_flag
    return self.extra_nets[prefix_name], prefix_name

  def construct_extra_net(self, net_dict, layer_list,
                          search_flag=None, dep_layers_in_extra=False,
                          net_name=None, prefix_name=None):
    """
    The purpose is to create another net like `self` but with different flags,
    e.g. with `search_flag = True`.
    That `extra_net` can have different losses, which will be added.
    Layers in ``layer_list`` will be explicitly re-created in the extra net.
    Other layers are taken from ``self``.

    The creation of the extra net and layers in the extra net can be triggered explicitly
    by referring to another layer as e.g. ``"extra.search:layer"``.
    When done this way, all the dependencies of it are created in self again;
    unless you explicitly have called another layer like ``"extra.search:dep"``.
    See :func:`test_extra_search` for an example.

    :param dict[str,dict[str]] net_dict:
    :param list[str] layer_list:
    :param bool|None search_flag:
    :param bool dep_layers_in_extra: layers not in layer_list, but which are not yet created,
      will be part of the extra net, not self.
    :param str|None net_name:
    :param str|None prefix_name: e.g. "extra.search", such that layers would be called like "extra.search:layer"
    :return: the layers created via layer_list (all in extra net)
    :rtype: list[LayerBase]
    """
    extra_net, prefix_name = self._get_extra_net(
      search_flag=search_flag, net_name=net_name, prefix_name=prefix_name)
    extra_net.layers_desc.update(net_dict)

    def get_layer(src_name):
      """
      :param str src_name:
      :rtype: LayerBase
      """
      explicit_extra_layer_name = "%s:%s" % (prefix_name, src_name)
      if explicit_extra_layer_name in net_dict:
        # This is a special marked layer, which is named like this to specify that it should explicitly
        # be used in this case.
        return extra_net.construct_layer(net_dict=net_dict, name=explicit_extra_layer_name, get_layer=get_layer)
      if dep_layers_in_extra:
        return extra_net.construct_layer(net_dict=net_dict, name=src_name, get_layer=get_layer)
      try:
        return extra_net.get_layer(src_name)
      except LayerNotFound:
        pass  # ok, we will try to construct it then
      # Create it in self, not in the extra net.
      return self.construct_layer(net_dict=net_dict, name=src_name, get_layer=get_layer)

    created_layers = []
    for layer_name in layer_list:
      # Always (re)create the specified layer in the layer_list (or return it if it already in the extra net).
      # However, any dependencies might resolve to the main net.
      created_layers.append(extra_net.construct_layer(
        net_dict=net_dict, name=layer_name, check_existing=False, get_layer=get_layer))

    if extra_net.recurrent:
      self.recurrent = True
    self.used_data_keys.update(extra_net.used_data_keys)
    return created_layers

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
    if '/' in name:
      # It may be a hierarchical path to a sub-layer, which should have been found by get_layer()
      # but maybe it's not constructed yet, so try constructing the root layer.
      root_layer = get_layer(name.split('/')[0])
      sub_layer = root_layer.get_sub_layer('/'.join(name.split('/')[1:]))  # get the sub-layer from the root-layer
      if sub_layer:
        return sub_layer
      # Pass on here. Maybe we find the layer. Otherwise it will fail below.
    layer_desc = None
    if self._extra_layer_name_prefix_pattern.match(name):
      prefix, name_ = name.split(":", 1)
      if self.extra_parent_net:
        extra_net, _ = self.extra_parent_net._get_extra_net(prefix_name=prefix, auto_create=False)
        if extra_net is not self:
          return self.extra_parent_net.construct_extra_net(net_dict=net_dict, layer_list=[name], prefix_name=prefix)[0]
        if name in net_dict:
          # We explicitly allow this, and want to construct it here in this extra net, from this layer desc.
          layer_desc = net_dict[name]
        # In any case, this layer should have the name without that prefix,
        # such that param-sharing etc works as expected.
        name = name_
      else:
        return self.construct_extra_net(net_dict=net_dict, layer_list=[name], prefix_name=prefix)[0]
    if not layer_desc:
      if name not in net_dict:
        if name == "data":
          layer_desc = {"class": "source"}
        elif name.startswith("data:"):
          layer_desc = {"class": "source", "data_key": name[len("data:"):]}
        elif name == ":i":  # via set_rec_step_info / RecStepInfoLayer
          # Note: This will fail at normal construction, but works for template construction.
          layer_desc = {"class": ":i"}
      else:
        layer_desc = net_dict[name]
    if not layer_desc:
      raise LayerNotFound("layer %r not found in %r" % (name, self), layer_name=name, network=self, net_dict=net_dict)
    if not add_layer:
      add_layer = self.add_layer
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

  def _create_layer_layer_desc(self, name, layer_desc, template=True):
    """
    This is called *after* :func:`LayerBase.transform_config_dict`
    and *before* :func:`LayerBase.get_out_data_from_opts`.

    :param str name: layer name
    :param dict[str] layer_desc: opts
    :param bool template: for template inference, we do not need the full logic
    :rtype: dict[str]
    """
    from returnn.tf.layers.basic import SearchChoices
    if not template:
      # This might not even work correctly during template construction.
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
    :param (()->LayerBase)|LayerBase|type[LayerBase] layer_class:
    :param layer_desc: contains the kwargs for the layer class.
      the args should have been transformed via layer_class.transform_config_dict before (see construct_layer).
      must not contain "name" and "network", which will be automatically added here.
      should not contain "output", which will be initialized to layer_class.get_out_data_from_opts.
      the layer_class will usually then define the layer.output and its placeholder.
      there is one notable exception: the InternalLayer, where you predefine the output.
    :rtype: LayerBase
    """
    from pprint import pprint
    from returnn.util.basic import help_on_type_error_wrong_args
    from returnn.tf.util.basic import py_print
    layer_desc = self._create_layer_layer_desc(name=name, layer_desc=layer_desc, template=False)
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
        output_template = layer_desc["output"]
        assert isinstance(output_template, Data), "%s %r layer_desc %r ['output'] is not a Data instance" % (
          layer_class.__name__, name, layer_desc)
        output_template.sanity_check(ignore_placeholder=True)  # placeholder might be overwritten later
        output_template_special_axes = output_template.get_special_axes_dict()
        if not output_template.available_for_inference and not self.eval_flag:
          from returnn.tf.layers.base import DataNotAvailableLayer
          return DataNotAvailableLayer(
            name=layer_desc['name'], network=layer_desc['network'], output=output_template,
            layer_class=layer_class, layer_desc=layer_desc)
        layer = layer_class(**layer_desc)
        layer.post_init(layer_desc)
        layer.output.sanity_check()
        # The axes should not have moved now.
        output_special_axes = layer.output.get_special_axes_dict()
        assert output_template_special_axes == output_special_axes, "%s %r: not equal: %r == %r, from data %r -> %r" % (
          layer_class.__name__, name,
          output_template_special_axes, output_special_axes,
          output_template, layer.output)
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
        from returnn.tf.util.basic import identity_with_check_numerics
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

  def get_used_data_keys(self, exclude_extra_added=True):
    """
    :param bool exclude_extra_added:
    :rtype: set[str]
    """
    used_data_keys = self.used_data_keys
    if exclude_extra_added:
      used_data_keys = used_data_keys.difference(self.extern_data.extra_added_keys)
    return used_data_keys

  def get_seq_tags(self, mark_data_key_as_used=True, beam=None):
    """
    :param bool mark_data_key_as_used: for extern_data
    :param returnn.tf.util.data.SearchBeam|None beam:
    :return: tensor of shape (batch,) of dtype string, via extern_data
    :rtype: tf.Tensor
    """
    data = self.get_extern_data(key="seq_tag", mark_data_key_as_used=mark_data_key_as_used)
    if beam:
      data = data.copy_extend_with_beam(beam)
    return data.placeholder

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
    losses_multi_dict = {}  # type: typing.Dict[str,typing.List[typing.Tuple[typing.Optional[str],LossHolder]]]
    layer_items = [(None, layer) for name, layer in sorted(self.layers.items())]
    if self.extra_nets:
      for extra_name_prefix, extra_net in sorted(self.extra_nets.items()):
        assert isinstance(extra_net, TFNetwork)
        layer_items += [(extra_name_prefix, layer) for (name, layer) in sorted(extra_net.layers.items())]
    for extra_name_prefix, layer in layer_items:
      assert isinstance(layer, LayerBase)
      assert not extra_name_prefix or isinstance(extra_name_prefix, str)
      if not extra_name_prefix:
        name = layer.name
      else:
        name = "%s/%s" % (extra_name_prefix, layer.name)
      tf_scope_name = layer.cls_get_tf_scope_name(name=name)
      assert isinstance(layer, LayerBase)
      with reuse_name_scope("loss"):
        with reuse_name_scope(tf_scope_name):
          losses = layer.get_losses_initialized(reduce_func=reduce_func)
          for loss_obj in losses:
            losses_multi_dict.setdefault(loss_obj.name, []).append((extra_name_prefix, loss_obj))
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

    losses_dict = {}  # type: typing.Dict[str,LossHolder]
    for loss_name, loss_holders in losses_multi_dict.items():
      assert len(loss_holders) >= 1
      if len(loss_holders) == 1:  # unique name
        assert loss_name not in losses_dict
        losses_dict[loss_name] = loss_holders[0][1]
      else:
        for extra_name_prefix, loss_holder in loss_holders:
          if not extra_name_prefix:
            name = loss_holder.name
          else:
            name = "%s:%s" % (extra_name_prefix, loss_holder.name)
          assert name not in losses_dict
          losses_dict[name] = loss_holder

    return losses_dict, total_loss, total_constraints

  def _construct_objective(self):
    with tf.name_scope("objective"):
      losses_dict, total_loss, total_constraints = self.get_losses_initialized(with_total=True)
      self.losses_dict.clear()
      self.losses_dict.update(losses_dict)
      self.total_loss = total_loss
      self.total_constraints = total_constraints
      self.total_objective = total_loss + total_constraints
      if not tf_util.get_current_control_flow_context():  # summaries cannot be used when in loop or cond
        tf_compat.v1.summary.scalar("loss", self.total_loss)
        tf_compat.v1.summary.scalar("constraints", self.total_constraints)
        tf_compat.v1.summary.scalar("objective", self.total_objective)

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
      self._merge_all_summaries = tf_compat.v1.summary.merge_all()
    return self._merge_all_summaries

  def get_fetches_dict(self, config=None, should_train=None, should_eval=None,
                       with_summary=False, with_size=False,
                       horovod_collected_reduce_inputs=None):
    """
    :param Config.Config|None config:
    :param bool|None should_train:
    :param bool|None should_eval:
    :param bool with_summary:
    :param bool with_size:
    :param dict[str,(tf.Tensor,tf.Tensor)]|None horovod_collected_reduce_inputs: will write into. see below
    :return: values and actions which should be calculated and executed in self.run() by the TF session for each step
    :rtype: dict[str,tf.Tensor|tf.Operation]
    """
    # Note that it is important that we do not recreate graph nodes for every call to this function.
    # Thus everything which we access here should be cached.
    import os
    if config is None:
      config = self.get_config()
    if should_train is None:
      should_train = self.train_flag
    if should_eval is None:
      should_eval = self.eval_flag
    use_horovod_reduction = False
    if config.is_true("use_horovod"):
      import returnn.tf.horovod
      if returnn.tf.horovod.get_ctx().should_sync_every_step():
        # Note: This logic should be in sync with the logic in _horovod_signal_have_more_data.
        use_horovod_reduction = True

    def reduce_sum(x, name, average=False):
      """
      :param tf.Tensor x:
      :param str name:
      :param bool average:
      :return: sum(x) if horovod else x
      :rtype: tf.Tensor
      """
      if not use_horovod_reduction:
        return x
      from returnn.tf.util.basic import global_tensor
      # noinspection PyUnresolvedReferences,PyPackageRequirements
      import horovod.tensorflow as hvd
      out = global_tensor(
        lambda: hvd.allreduce(x, average=average),
        name="horovod_fetch_reduce_sum__" + name.replace(":", "__").replace("/", "_"))
      if horovod_collected_reduce_inputs is not None and x.name not in horovod_collected_reduce_inputs:
        horovod_collected_reduce_inputs[x.name] = (x, out)
      return out

    def inv_reduce_sum(x, name):
      """
      :param tf.Tensor x:
      :param str name:
      :return: reciprocal(sum(reciprocal(x))) if horovod else x
      :rtype: tf.Tensor
      """
      if not use_horovod_reduction:
        return x
      return tf_compat.v1.reciprocal(reduce_sum(tf_compat.v1.reciprocal(x), name=name))

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
        loss = tf_util.global_tensor(lambda: tf.constant(0.0), name="zero_loss")
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
      for dev in tf_util.get_tf_list_local_devices():
        if dev.device_type != "GPU":
          # mem_usage_for_dev currently only works for GPU
          continue
        d["mem_usage:%s" % os.path.basename(dev.name.replace("/device:", "/"))] = tf_util.mem_usage_for_dev(dev.name)

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
    if self._extra_layer_name_prefix_pattern.match(layer_name):
      prefix, layer_name = layer_name.split(":", 1)
      extra_net, _ = self._get_extra_net(prefix_name=prefix, auto_create=False)
      if not extra_net:
        raise LayerNotFound(
          "cannot get layer %r, no extra net for %r" % (layer_name, self),
          layer_name=layer_name, network=self)
      return extra_net.get_layer(layer_name)
    if layer_name.startswith("base:"):
      if not self.parent_net:
        raise LayerNotFound(
          "cannot get layer %r, no parent net for %r" % (layer_name, self), layer_name=layer_name, network=self)
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
    if self.extra_parent_net:
      return self.extra_parent_net.get_layer(layer_name)
    if layer_name not in self.layers:
      raise LayerNotFound("layer %r not found in %r" % (layer_name, self), layer_name=layer_name, network=self)
    return self.layers[layer_name]

  def _get_all_layers(self):
    """
    :return: all layers, including extra net (but excluding layers which already exist in self)
    :rtype: list[LayerBase]
    """
    layers = []
    for (_, layer) in sorted(self.layers.items()):
      if layer not in layers:
        layers.append(layer)
    if self.extra_nets:
      for _, extra_net in sorted(self.extra_nets.items()):
        assert isinstance(extra_net, TFNetwork)
        for layer in extra_net._get_all_layers():
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
    trainable_vars_col = tf_compat.v1.get_collection(tf_compat.v1.GraphKeys.TRAINABLE_VARIABLES)
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
    if self.extra_nets:
      for _, extra_net in sorted(self.extra_nets.items()):
        assert isinstance(extra_net, TFNetwork)
        for param in extra_net.get_trainable_params():
          if param not in ls:
            ls.append(param)
    return ls

  def declare_train_params(self, hidden_layer_selection=None, with_output=None, global_trainable=None):
    """
    :param list[str]|None hidden_layer_selection:
    :param bool|None with_output:
    :param bool|None global_trainable:
    """
    if global_trainable is None:
      global_trainable = self.layers_desc.get("#trainable", True)
    if global_trainable:
      if hidden_layer_selection is None:
        hidden_layer_selection = [name for (name, layer) in self.layers.items() if not layer.is_output_layer()]
      else:
        hidden_layer_selection = list(hidden_layer_selection)
      if with_output is None:
        with_output = True
      if with_output:
        hidden_layer_selection += [name for (name, layer) in self.layers.items() if layer.is_output_layer()]
      hidden_layer_selection = set(hidden_layer_selection)
    else:
      hidden_layer_selection = set()
    self._selected_train_layers = sorted(hidden_layer_selection)
    if self.extra_nets:
      for _, extra_net in self.extra_nets.items():
        extra_net.declare_train_params(global_trainable=global_trainable)  # select all, currently...

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
    :param tf.compat.v1.Session session:

    Note: This will create a new node to the graph for each call!
    And it will overwrite also the already initialized variables.
    So you should call this only once after network construction and before you maybe load some of the params
    from external sources.
    If you know that you will load all params explicitly, you would not need to call this function.
    """
    var_list = self.get_params_list() + self.get_auxiliary_params()
    with tf.name_scope("var_initializer"):
      initializer_op = tf_compat.v1.variables_initializer(var_list=var_list)
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
    :param tf.compat.v1.Session session:
    :return: dict: layer_name -> param_name -> variable numpy array
    :rtype: dict[str,dict[str,numpy.ndarray]]
    Note that this excludes auxiliary params.
    """
    layers = {}  # type: typing.Dict[str,typing.Dict[str,numpy.ndarray]]
    for layer in self._get_all_layers():
      layers[layer.name] = layer.get_param_values_dict(session)
    return layers

  def set_param_values_by_dict(self, values_dict, ignore_non_existing=False, **kwargs):
    """
    :param dict[str,dict[str,numpy.ndarray]] values_dict:
    :param bool ignore_non_existing:
    :param kwargs: passed to :func:`LayerBase.set_param_values_by_dict`

    Note that this excludes auxiliary params.
    """
    layers = {layer.name: layer for layer in self._get_all_layers()}  # type: typing.Dict[str,LayerBase]
    for layer_name, layer_values_dict in values_dict.items():
      if layer_values_dict:
        if ignore_non_existing and layer_name not in layers:
          print("Will not set layer %r because it does not exist." % (layer_name,), file=log.v3)
          continue
        layers[layer_name].set_param_values_by_dict(values_dict=layer_values_dict, **kwargs)

  def get_auxiliary_params(self):
    """
    :rtype: list[tf.Variable]
    """
    return [self.global_train_step]

  def get_params_serialized(self, session):
    """
    :param tf.compat.v1.Session session:
    :rtype: TFNetworkParamsSerialized
    """
    return TFNetworkParamsSerialized(
      values_dict=self.get_param_values_dict(session=session),
      global_train_step=self.get_global_train_step(session=session))

  def set_params_by_serialized(self, serialized, session, **kwargs):
    """
    :param TFNetworkParamsSerialized serialized:
    :param tf.compat.v1.Session session:
    :param kwargs: passed to :func:`set_param_values_by_dict`
    """
    self.set_param_values_by_dict(serialized.values_dict, session=session, **kwargs)
    self.set_global_train_step(serialized.global_train_step, session=session)

  def set_global_train_step(self, step, session):
    """
    :param int step:
    :param tf.compat.v1.Session session:
    """
    self.get_var_assigner(self.global_train_step).assign(step, session=session)

  def get_global_train_step(self, session):
    """
    :param tf.compat.v1.Session session:
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
      self.epoch_step = tf_compat.v1.placeholder(name="epoch_step", shape=(), dtype=tf.int64)
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
      self.saver = tf_compat.v1.train.Saver(
        var_list=self.get_saveable_params_list(), max_to_keep=2 ** 31 - 1)

  def save_params_to_file(self, filename, session):
    """
    Will save the model parameters to the filename.
    Note that the model parameters live inside the current TF session.

    :param str filename:
    :param tf.compat.v1.Session session:
    """
    import os
    filename = os.path.abspath(filename)  # TF needs absolute path
    from returnn.util.basic import maybe_make_dirs
    maybe_make_dirs(os.path.dirname(filename))
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
    Will load the model parameters from the filename.
    Note that the model parameters live inside the current TF session.

    :param str filename:
    :param tf.compat.v1.Session session:
    """
    saveable_params = self.get_saveable_params_list()
    must_use_custom_checkpoint_loader = False
    if any([have_custom_post_init(param) for param in saveable_params]):
      # We must keep the behavior consistent.
      # CustomCheckpointLoader will not load any params with a custom init.
      must_use_custom_checkpoint_loader = True
    if any([layer.custom_param_importer for layer in self._get_all_layers()]):
      # Need to use CustomCheckpointLoader because only that handles custom_param_importer correctly.
      must_use_custom_checkpoint_loader = True
    if must_use_custom_checkpoint_loader:
      loader = CustomCheckpointLoader(
        filename=filename, saveable_params=saveable_params, network=self)
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
          filename=filename, saveable_params=saveable_params, network=self)
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
    print("  layers:", file=log.v2)
    for layer_name, layer in sorted(self.layers.items()):
      layer_dim = 'unknown' if layer.output.dim is None else '%i' % layer.output.dim
      print("    layer %s %r #: %s" % (layer.layer_class, layer_name, layer_dim), file=log.v2)
    if not self.layers:
      print("    (no layers)", file=log.v2)
    if self.extra_nets:
      for _, extra_net in sorted(self.extra_nets.items()):
        assert isinstance(extra_net, TFNetwork)
        print("  %r layers:" % extra_net.name, file=log.v2)
        for layer_name, layer in sorted(extra_net.layers.items()):
          layer_dim = 'unknown' if layer.output.dim is None else '%i' % layer.output.dim
          print("    layer %s %r #: %s" % (layer.layer_class, layer_name, layer_dim), file=log.v2)
        if not extra_net.layers:
          print("    (no layers)", file=log.v2)
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
    from returnn.tf.util.basic import cond
    return cond(self.train_flag, fn_train, fn_eval)

  def get_search_choices(self, sources=None, src=None, base_search_choice=None, _layer_to_search_choices=None,
                         debug_stream=None):
    """
    Recursively searches through all sources,
    and if there is a :class:`ChoiceLayer` / any layer with search_choices, returns it.
    Could also go to the parent network.
    If there are multiple, it assumes they are on the same search-sequence in the search-tree
    and it will return the last one.

    :param LayerBase|None src:
    :param LayerBase|None base_search_choice:
    :param list[LayerBase]|None sources:
    :param dict[LayerBase]|None _layer_to_search_choices: keep track of visited layers in case there are circular deps
    :param typing.TextIO|None debug_stream: if given, will print additional debug info into it
    :return: (direct or indirect) source LayerBase which has search_choices, or None
    :rtype: LayerBase|None
    """
    if src:
      # Note: Make sure we query from the current frame. Otherwise this would get ugly.
      # (Only for src; accept for base_search_choice, it should not matter.)
      assert src.get_normalized_layer() == src
    from returnn.tf.layers.basic import SearchChoices
    from functools import cmp_to_key
    from pprint import pformat
    if _layer_to_search_choices is None:
      _layer_to_search_choices = {}  # type: typing.Dict[LayerBase,typing.List[LayerBase]]
    normalized_to_layer = {}  # type: typing.Dict[LayerBase,LayerBase]
    layers = self._get_all_search_choices(
      sources=sources, src=src, base_search_choice=base_search_choice,
      _layer_to_search_choices=_layer_to_search_choices, _normalized_to_layer=normalized_to_layer)

    def full_trace_for_layer(layer, _layer_trace=None):
      """
      :param LayerBase layer: with search choices
      :param list[LayerBase]|None _layer_trace:
      :return: layers with search choices
      :rtype: list[LayerBase]
      """
      assert isinstance(layer, LayerBase) and isinstance(layer.search_choices, SearchChoices)
      if _layer_trace is None:
        _layer_trace = []  # type: typing.List[LayerBase]
      if layer not in _layer_trace:
        _layer_trace.append(layer)
      else:
        return _layer_trace
      if layer not in _layer_to_search_choices:  # happens if layer is a choice
        self._get_all_search_choices(
          base_search_choice=layer,
          _layer_to_search_choices=_layer_to_search_choices, _normalized_to_layer=normalized_to_layer)
      for dep in _layer_to_search_choices[layer]:
        full_trace_for_layer(dep, _layer_trace=_layer_trace)
      return _layer_trace

    def get_debug_dep_map():
      """
      :rtype: dict[str,list[str]]
      """
      relevant_map = {}
      for key, values in _layer_to_search_choices.items():
        relevant_map[key.get_absolute_name()] = [value.get_absolute_name() for value in values]
      return relevant_map

    def compare_layer(l1, l2):
      """
      Compares two layers with search_choices, to sort them.
      See also: :func:`SearchChoices.compare`.

      :param LayerBase l1:
      :param LayerBase l2:
      :return: 0 if equal, -1 if l1 <= l2, else 1 if l1 >= l2
      :rtype: int
      """
      assert isinstance(l1, LayerBase) and isinstance(l1.search_choices, SearchChoices)
      assert isinstance(l2, LayerBase) and isinstance(l2.search_choices, SearchChoices)
      l1n = l1.get_normalized_layer()
      l2n = l2.get_normalized_layer()
      if l1 != l1n and l2 != l2n:  # only in the case that we get normalized variants for both
        l1, l2 = l1n, l2n
      if l1 is l2:
        return 0
      l1trace_ = full_trace_for_layer(l1)
      l2trace_ = full_trace_for_layer(l2)
      if l1 in l2trace_ and l2 not in l1trace_:
        return -1
      if l2 in l1trace_ and l1 not in l2trace_:
        return 1
      raise Exception(
        ("get_search_choices src=%r base_search_choice=%r sources=%r.\n"
         "Search choices cannot be compared.\n"
         "Layer 1\n  %r\nchoice trace\n%s\n"
         "vs layer 2\n  %r\nchoice trace\n%s.\n"
         "Full dependency map:\n%s\n"
         "Relevant layers:\n%s\nNetwork:\n%s") % (
          src, base_search_choice, sources,
          l1, pformat(l1trace_), l2, pformat(l2trace_), pformat(get_debug_dep_map()), pformat(layers),
          pformat(self.layers)))

    if debug_stream:
      print("Relevant layers:\n%s" % pformat(layers), file=debug_stream)
      print("Full dependency map:\n%s" % pformat(get_debug_dep_map()), file=debug_stream)
    while base_search_choice in layers:
      layers.remove(base_search_choice)
    if not layers:
      return None
    layers = sorted(layers, key=cmp_to_key(compare_layer))
    return layers[-1]

  def _get_all_search_choices(self, sources=None, src=None, base_search_choice=None,
                              _layer_to_search_choices=None, _normalized_to_layer=None):
    """
    Recursively searches through all sources,
    and if there is a :class:`ChoiceLayer` / any layer with search_choices, returns it.
    Could also go to the parent network.
    If there are multiple, it assumes they are on the same search-sequence in the search-tree
    and it will return the last one.

    :param LayerBase|None src:
    :param LayerBase|None base_search_choice:
    :param list[LayerBase]|None sources:
    :param dict[LayerBase,list[LayerBase]]|None _layer_to_search_choices:
      tracks visited layers in case there are circular deps
    :param dict[LayerBase,LayerBase]|None _normalized_to_layer:
    :return: (direct or indirect) sources LayerBase which has search_choices
    :rtype: list[LayerBase]
    """
    if _layer_to_search_choices is None:
      _layer_to_search_choices = {}  # type: typing.Dict[LayerBase,typing.List[LayerBase]]
    if _normalized_to_layer is None:
      _normalized_to_layer = {}  # type: typing.Dict[LayerBase,LayerBase]
    if src is not None:
      assert isinstance(src, LayerBase)
      normalized_src = src.get_normalized_layer()
      if normalized_src != src:
        assert _normalized_to_layer.setdefault(normalized_src, src) == src  # Currently expecting that this is unique.
        if src.search_choices:
          assert normalized_src.search_choices, "normalized %s vs %s (choices %s)" % (
            normalized_src, src, src.search_choices)
      if src.search_choices:
        if src.search_choices.is_decided:
          return []
        return [src]
      assert base_search_choice is None
      base_search_choice = src
    if base_search_choice is not None:
      if base_search_choice in _layer_to_search_choices:
        return _layer_to_search_choices[base_search_choice]
      else:
        _layer_to_search_choices[base_search_choice] = []  # we visit it now
      normalized_base = base_search_choice.get_normalized_layer()
      if normalized_base != base_search_choice:
        # Currently expecting that this is unique.
        assert _normalized_to_layer.setdefault(normalized_base, base_search_choice) == base_search_choice
      assert sources is None
      sources = base_search_choice.get_dep_layers()
    assert sources is not None
    layers = []  # type: typing.List[LayerBase]
    for src_ in sources:
      src_choice_layers = self._get_all_search_choices(
        src=src_, _layer_to_search_choices=_layer_to_search_choices, _normalized_to_layer=_normalized_to_layer)
      for layer in src_choice_layers:
        if base_search_choice and layer not in _layer_to_search_choices[base_search_choice]:
          _layer_to_search_choices[base_search_choice].append(layer)
        if layer not in layers:
          layers.append(layer)
    if not layers:
      # Use parent layer if available.
      # Note that we should not mix layers from different context frames,
      # e.g. inside and outside a rec loop, as the search choices cannot be compared.
      if self.parent_layer and not self.is_inside_rec_layer():
        # noinspection PyProtectedMember
        return self.parent_layer.network._get_all_search_choices(sources=self.parent_layer.get_dep_layers())
      return []
    if base_search_choice is not None:
      normalized_base = base_search_choice.get_normalized_layer()
      if normalized_base != base_search_choice:  # from prev frame or so
        # Just make sure we visit these as well.
        normalized_choices = self._get_all_search_choices(
          base_search_choice=normalized_base,
          _layer_to_search_choices=_layer_to_search_choices, _normalized_to_layer=_normalized_to_layer)
        if normalized_choices == layers:
          # This looks independent. Return as is.
          return layers
        if any([layer.get_normalized_layer() == layer for layer in normalized_choices]):
          # Filter any "prev:..." layers away. This should always be correct.
          # Also, this is important to have the correct choice resolution for the prev layer (base_search_choice).
          normalized_choices = [layer for layer in normalized_choices if layer.get_normalized_layer() == layer]
          # Get corresponding "prev:..." layers.
          from pprint import pformat
          assert all([layer in _normalized_to_layer for layer in normalized_choices]), "\n".join([
            "No cur -> prev mapping for some layers.", "",
            "Base: %s" % base_search_choice, "", "Cur (normalized) base: %s" % normalized_base, "",
            "Prev choices:", pformat(layers), "", "Cur (normalized) choices:", pformat(normalized_choices), "",
            "Mapping:", pformat(_normalized_to_layer), ""])
          layers = [_normalized_to_layer[layer] for layer in normalized_choices]
          _layer_to_search_choices[base_search_choice] = layers
    return layers

  def debug_search_choices(self, base_search_choice):
    """
    :param LayerBase base_search_choice:
    :return: nothing, by intention, such that constructs like `assert ..., debug_search_choices(...) or (...)` work
    """
    print("debug search choices:")
    print("  base:", base_search_choice)
    print("  network:")
    for _, layer in sorted(self.layers.items()):
      print("    layer:", layer)

    class Visitor(dict):
      """
      Wraps around `dict`, to catch any `__setitem__` calls.
      """
      def __setitem__(self, key, value):
        """
        :param LayerBase key:
        :param value:
        """
        print("  visit: %r, search choices %r" % (key, key.search_choices))
        print("    sources: %s" % ", ".join([
          "%r search choices %r" % (dep.get_absolute_name(), dep.search_choices)
          for dep in key.get_dep_layers()] or ["None"]))
        super(Visitor, self).__setitem__(key, value)

    search_choices = self.get_search_choices(
      base_search_choice=base_search_choice, _layer_to_search_choices=Visitor(), debug_stream=sys.stdout)
    print("-> search choices:", search_choices)

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
    from returnn.tf.util.basic import get_shape_dim, reuse_name_scope_of_tensor
    if self._batch_dim is not None:
      return self._batch_dim
    # First check parent because there we might get the true batch dim.
    # (Do not check parent_layer, because that potentially includes a beam.)
    if self.parent_net:
      return self.parent_net.get_data_batch_dim()
    if self.extra_parent_net:
      return self.extra_parent_net.get_data_batch_dim()
    for key, data in self.extern_data.get_sorted_data_items():
      assert isinstance(data, Data)
      if data.available_for_inference:
        self.used_data_keys.add(key)
        with reuse_name_scope_of_tensor(data.placeholder):
          batch_dim = get_shape_dim(data.placeholder, data.batch_dim_axis, name="batch_dim")
          self._batch_dim = batch_dim
          return batch_dim
    raise Exception("We cannot tell the batch dim.")

  def set_rec_step_info(self, i, end_flag=None, end_flag_source=None, seq_lens=None):
    """
    Used by _SubnetworkRecCell.

    :param tf.Tensor i: scalar, int32, current step (time)
    :param tf.Tensor|None end_flag: (batch,), bool, says that the current sequence has ended
    :param LayerBase|None end_flag_source:
    :param tf.Tensor|None seq_lens: (batch,) int32, seq lens
    """
    from returnn.tf.layers.rec import RecStepInfoLayer
    self.layers[":i"] = RecStepInfoLayer(
      name=":i", network=self, i=i, end_flag=end_flag, end_flag_source=end_flag_source, seq_lens=seq_lens)

  def is_inside_rec_layer(self, inside_loop=True):
    """
    :param bool inside_loop: only True if the network is constructed within the loop (not moved out)
    :return: whether we are inside a :class:`RecLayer`. see :func:`get_rec_parent_layer`
    :rtype: bool
    """
    if self._is_inside_rec_layer is not None:
      return self._is_inside_rec_layer
    if self.get_rec_parent_layer(inside_loop=inside_loop) is not None:
      return True
    if self.parent_net:
      # This might still return True, in case of template construction, where _is_inside_rec_layer is set.
      return self.parent_net.is_inside_rec_layer(inside_loop=inside_loop)
    return False

  def get_rec_parent_layer(self, inside_loop=True):
    """
    :param bool inside_loop: only return if the network is constructed within the loop (not moved out)
    :return: if we are a subnet of a :class:`RecLayer`, will return the RecLayer instance
    :rtype: returnn.tf.layers.rec.RecLayer|None
    """
    from returnn.tf.layers.rec import RecLayer
    if isinstance(self.parent_layer, RecLayer):
      if inside_loop:
        # noinspection PyProtectedMember
        from returnn.tf.layers.rec import _SubnetworkRecCell
        assert isinstance(self.parent_layer.cell, _SubnetworkRecCell)
        if self is not self.parent_layer.cell.net:
          return None
      return self.parent_layer
    if self.parent_net:
      return self.parent_net.get_rec_parent_layer(inside_loop=inside_loop)
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
    # noinspection PyProtectedMember
    from returnn.tf.layers.rec import RecStepInfoLayer, _SubnetworkRecCell
    # Fast path first. This also enables some simple debugging.
    if ":i" in self.layers and isinstance(self.layers[":i"], RecStepInfoLayer):
      return self.layers[":i"]
    rec_layer = self.get_rec_parent_layer()
    # The second condition is true if all layers have been optimized out of the rec layer.
    # (We could handle this case though. It's just not implemented.)
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
    from returnn.config import Config, get_global_config
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
    ls = tf_compat.v1.get_collection_ref(tf_compat.v1.GraphKeys.UPDATE_OPS)
    assert isinstance(ls, list)
    ls.extend(deps)

  @staticmethod
  def get_post_control_dependencies():
    """
    :rtype: list[tf.Operation]
    """
    return tf_compat.v1.get_collection(tf_compat.v1.GraphKeys.UPDATE_OPS)

  def register_graph_reset_callback(self, cb):
    """
    Note: These callbacks are not called automatically.
    You explicitly have to call :func:`call_graph_reset_callbacks`.

    Note: We don't store this in the graph itself (e.g. via tf.get_collection),
    as we don't want to serialize this
    (which would also lead to an error, because it cannot be serialized).

    Note: Currently these callbacks might get called multiple times,
    so make sure that this is not a problem.
    Also make sure that the network/session is still in a valid state after this has been called,
    e.g. such that further session runs would still work correctly.

    Note: These callbacks will only be called if there was not any error.

    :param function|()->None cb:
    """
    self.get_root_network()._graph_reset_callbacks.append(cb)

  def get_graph_reset_callbacks(self):
    """
    :rtype: list[()->None]
    """
    return self.get_root_network()._graph_reset_callbacks

  def call_graph_reset_callbacks(self):
    """
    Calls any callbacks registered via :func:`register_graph_reset_callback`.
    """
    for cb in self.get_graph_reset_callbacks():
      cb()

  def set_run_opts(self, epoch, dataset_name):
    """
    The run options are valid during one loop over some dataset.

    Contrary to epoch_step, train_flag, etc, we do not provide these as TF placeholders,
    for convenience, because it is not needed right now.
    If it is needed, it probably is easier to introduce auxiliary TF variables (on CPU) instead
    and just set them once here.

    :param int epoch:
    :param str|None dataset_name:
    """
    root_net = self.get_root_network()
    root_net._run_opts = dict(epoch=epoch, dataset_name=dataset_name)

  def get_run_opts(self):
    """
    :rtype: dict[str]
    """
    opts = self.get_root_network()._run_opts
    assert opts, "set_run_opts not called?"
    return opts.copy()

  def register_run_finished_callback(self, cb):
    """
    :param function|()->None cb:
    """
    self.get_root_network()._run_finished_callbacks.append(cb)

  def set_run_finished(self, error_occurred=False):
    """
    Maybe calls any callbacks registered via :func:`register_run_finished_callback`
    (if no error occurred)
    and cleans up the run opts.

    :param bool error_occurred:
    """
    root_net = self.get_root_network()
    if not error_occurred:
      for cb in root_net._run_finished_callbacks:
        cb()
    root_net._run_finished_callbacks[:] = []
    root_net._run_opts.clear()

  @classmethod
  def get_network_stack(cls):
    """
    :rtype: list[TFNetwork]
    """
    from returnn.tf.util.basic import CollectionKeys
    coll = tf_compat.v1.get_collection_ref(CollectionKeys.RETURNN_NET_STACK)
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

  def get_search_choices_from_beam(self, beam):
    """
    Currently we have somewhat redundant information in
    :class:`returnn.tf.util.data.SearchBeam` (which is totally independent from other things in RETURNN (which is good))
    and
    :class:`returnn.tf.layers.base.SearchChoices` (which is more dependent on the RETURNN layers,
      and has some more info).
    The :class:`Data` (which is also independent from other things in RETURNN (which is also good))
    only knows about :class:`returnn.tf.util.data.SearchBeam`
    but not about :class:`returnn.tf.layers.base.SearchChoices`.
    Thus there are situations where we only have a ref to the former, but like to get a ref to the latter.

    Note that this might (hopefully) get cleaned up at some point...

    :param returnn.tf.util.data.SearchBeam beam:
    :rtype: returnn.tf.layers.base.SearchChoices|None
    """
    root_net = self.get_root_network()
    if root_net is not self:
      # Use the root network, to just use a single map where to look at.
      return root_net.get_search_choices_from_beam(beam)
    return self._map_search_beam_to_search_choices.get(beam, None)

  def register_search_choices_for_beam(self, beam, search_choices):
    """
    :param returnn.tf.util.data.SearchBeam beam:
    :param returnn.tf.layers.base.SearchChoices search_choices:
    """
    root_net = self.get_root_network()
    if root_net is not self:
      # Use the root network, to just use a single map where to look at.
      return root_net.register_search_choices_for_beam(beam, search_choices)
    self._map_search_beam_to_search_choices[beam] = search_choices


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
    :param returnn.tf.layers.base.Loss loss:
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

  def _normalized_value_per_seq(self, value):
    """
    :param tf.Tensor|None value: (batch*time,) or (time*batch,)
    :return: (batch,) or None if value is None
    :rtype: tf.Tensor|None
    """
    if value is None:
      return None
    return self.loss.reduce_to_batch(value, normalize=True)

  def get_normalized_loss_value_per_seq(self):
    """
    :return: (batch,) or None if loss is None
    :rtype: tf.Tensor|None
    """
    self._prepare()
    return self._normalized_value_per_seq(self._loss_value)

  def get_normalized_error_value_per_seq(self):
    """
    :return: (batch,) or None if error is None
    :rtype: tf.Tensor|None
    """
    self._prepare()
    return self._normalized_value_per_seq(self._error_value)

  def _value_per_pos(self, value):
    """
    :param tf.Tensor|None value: (batch*time,) or (time*batch,)
    :return: (batch,time) or None if value is None
    :rtype: tf.Tensor|None
    """
    if value is None:
      return None

    value = tf.reshape(value, tf.shape(self.loss.output.placeholder)[:2])  # (batch,time) or (time,batch)

    # We want output of the form (B,T)
    if self.loss.output.time_dim_axis == 0:
      from returnn.tf.util.basic import swapaxes
      value = swapaxes(value, 0, 1)  # resulting in (B,T,...)

    return value

  def get_loss_value_per_pos(self):
    """
    :return: (batch,time) or None if loss is None
    :rtype: tf.Tensor|None
    """
    self._prepare()
    return self._value_per_pos(self._loss_value)

  def get_error_value_per_pos(self):
    """
    :return: (batch,time) or None if error is None
    :rtype: tf.Tensor|None
    """
    self._prepare()
    return self._value_per_pos(self._error_value)

  def _tf_summary(self):
    """
    This gets called inside a loss name scope of the layer.

    :return: nothing, will use tf.summary
    """
    if self._network.parent_net:
      return  # skip summaries. the root net should also do this
    if tf_util.get_current_control_flow_context():  # summaries cannot be used when in loop or cond
      return
    name = self.get_tf_name()
    if self._loss_value is not None:
      # A loss value is typically a scalar but there are cases of sequence or position wise loss values
      # (e.g. if the eval_output_file_per_seq option is used).
      if self._loss_value.get_shape().ndims == 0:
        tf_compat.v1.summary.scalar("loss_%s" % name, self._loss_value * self._norm_factor)
        if self._network.get_config().bool("calculate_exp_loss", False):
          tf_compat.v1.summary.scalar("exp_loss_%s" % name, tf.exp(self._loss_value * self._norm_factor))
        if self._network.get_config().bool("debug_unnormalized_loss_summaries", False):
          tf_compat.v1.summary.scalar("unnormalized_loss_%s" % name, self._loss_value)
        if (self._network.get_config().bool("debug_objective_loss_summaries", False) and
                self._loss_value_for_objective is not None):
          tf_compat.v1.summary.scalar("objective_loss_%s" % name, self._loss_value_for_objective)
    if self._error_value is not None:
      if self._error_value.get_shape().ndims == 0:
        tf_compat.v1.summary.scalar("error_%s" % name, self._error_value * self._norm_factor)

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
    loss_value = self._loss_value
    if loss_value is not None:
      if self._network.get_config().bool("debug_add_check_numerics_on_output", False):
        print("debug_add_check_numerics_on_output: add for layer loss %r: %r" % (
          self._layer.name, self._layer.output.placeholder))
        from returnn.tf.util.basic import identity_with_check_numerics
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
    self._tf_summary()
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


class NetworkLayerException(Exception):
  """
  Some exception by the network, e.g. during construction.
  """
  def __init__(self, message, layer_name, network, net_dict=None):
    """
    :param str message:
    :param str layer_name:
    :param TFNetwork network:
    :param dict[str]|None net_dict:
    """
    super(NetworkLayerException, self).__init__(message)
    self.layer_name = layer_name
    self.network = network
    self.net_dict = net_dict or network.layers_desc


class NetworkConstructionDependencyLoopException(NetworkLayerException):
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
    for layer_name_ in reversed(constructing_layers):
      msg += "\n  %s" % layer_name_
    super(NetworkConstructionDependencyLoopException, self).__init__(
      msg, network=network, layer_name=layer_name, net_dict=net_dict)


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


class LayerNotFound(NetworkLayerException):
  """
  Via :func:`TFNetwork.get_layer`.
  """


def _help_data_or_array(value):
  """
  :param numpy.ndarray|bool|object value:
  :return: (info,(min,max))
  :rtype: (str,(int|float,int|float))
  """
  import numpy
  if isinstance(value, numpy.ndarray):
    info = "shape %s, dtype %s" % (value.shape, value.dtype)
    if value.size > 0:
      v_minmax = numpy.min(value), numpy.max(value)
      info += ", min/max %s/%s" % v_minmax
      if value.dtype.kind == "f":
        info += ", mean/stddev %s/%s" % (numpy.mean(value), numpy.std(value))
      if value.ndim <= 1:
        info += ", (%s)" % numpy.array2string(value)
    else:
      v_minmax = 0, 0
      info += ", EMPTY"
  elif isinstance(value, (numpy.floating, numpy.integer, numpy.bool_, float, int, bool, str, bytes)):
    v_minmax = 0, 1
    info = "%s(%s)" % (type(value).__name__, value)
  elif value is None:
    v_minmax = -1, -1
    info = "None"
  else:
    v_minmax = -1, -1
    info = "type %r" % type(value)
  return info, v_minmax


def help_on_tf_exception(
  session, exception, fetches, feed_dict=None,
  meta_step_info=None, extern_data=None,
  file=sys.stdout):
  """
  Generic debugging helper, on any TF exception (or even any other exception as well).
  Will try to provide as much helpful context information as possible.
  (This is not in :mod:`TFUtil` because it depends on `ExternData`, which is only defined here.)

  :param tf.compat.v1.Session session:
  :param tf.errors.OpError|BaseException exception:
  :param tf.Tensor|list[tf.Tensor]|dict[str,tf.Tensor]|object|None fetches:
  :param dict[tf.Tensor,numpy.ndarray]|None feed_dict:
  :param dict[str]|None meta_step_info:
  :param ExternData|None extern_data:
  :param typing.IO[str]|io.TextIOBase|io.StringIO file:
  """
  from pprint import pprint, pformat
  import traceback
  from returnn.tf.util.basic import get_base_name, find_ops_with_tensor_input, find_ops_path_output_to_input
  from tensorflow.python.util import nest
  if fetches is not None:
    fetches = nest.flatten(fetches)
  if isinstance(exception, tf.errors.OpError):
    op = exception.op
    print("Failing op:", repr(op), file=file)
    assert op is None or isinstance(op, tf.Operation)
    show_verbose_op_inputs = True
    if op and op.type == "Placeholder":
      # Likely this placeholder is not feeded, but used somewhere.
      # We assume that the usage of it is incorrect.
      # Try to give some hints where it was (incorrectly) used.
      using_ops = find_ops_with_tensor_input(op.outputs[0], fetches=fetches)
      print("Used by:", repr(using_ops), file=file)
      for op_ in using_ops:
        print("".join(traceback.format_list(op_.traceback)), file=file)
      if fetches:
        input_to_output_ops = find_ops_path_output_to_input(op.outputs[0], fetches=fetches)
        print("Input to output:", file=file)
        pprint(input_to_output_ops, stream=file)
      show_verbose_op_inputs = False
    if op and op.type.startswith("Horovod"):
      show_verbose_op_inputs = False
    if isinstance(exception, tf.errors.ResourceExhaustedError):
      show_verbose_op_inputs = False
    if op and op.inputs and show_verbose_op_inputs:
      # The exception occurred in the op, but that means that all the inputs to the op were correctly calculated.
      # It is probably helpful to calculate these again, and show their shape.
      try:
        # Note: In principle, we would just do `input_values = session.run(list(op.inputs), feed_dict=feed_dict)`.
        # However, this will not work if the op is inside a loop.
        # Thus, we use this workaround to fetch them nonetheless.
        # First find some value to fetch.
        assert fetches
        input_to_output_ops = find_ops_path_output_to_input(op.inputs[0], fetches=fetches)
        assert input_to_output_ops, "op.inputs[0] %r not in fetches\n%s" % (op.inputs[0], pformat(fetches))
        debug_fetch = None
        for x in input_to_output_ops:
          # noinspection PyProtectedMember
          if x._control_flow_context is None or x.type == "Exit":
            debug_fetch = x
            break
        assert debug_fetch is not None, "ops: %r, fetches: %r" % (input_to_output_ops, fetches)
        stop_at_ts = list(feed_dict.keys() if feed_dict else ())  # should not copy these
        for op_ in op.graph.get_operations():
          assert isinstance(op_, tf.Operation)
          # noinspection PyProtectedMember
          if op_._control_flow_context:
            continue
          for x in list(op_.inputs) + list(op_.outputs) + list(op.control_inputs):
            if isinstance(x, tf.Operation):
              continue
            assert isinstance(x, tf.Tensor)
            # noinspection PyProtectedMember
            if x.dtype._is_ref_dtype and x not in stop_at_ts:
              stop_at_ts.append(x)  # and also should not copy any variables/refs
        # Note: Some code in graph_editor, which is used in copy_graph, results in lots of spam about
        # tf.compat.v1.GraphKeys.VARIABLES deprecated usage (e.g. via get_predefined_collection_names or so).
        # We just do this ugly patch here, to work around the spam.
        tf_compat.v1.GraphKeys.VARIABLES = tf_compat.v1.GraphKeys.GLOBAL_VARIABLES
        from returnn.tf.util.basic import FetchHelper
        debug_fetch, fetch_helpers, op_copied = FetchHelper.copy_graph(
          debug_fetch, target_op=op, fetch_helper_tensors=list(op.inputs),
          stop_at_ts=stop_at_ts,
          verbose_stream=file)
        try:
          print("Execute again to debug the op inputs...", file=file)
          session.run(debug_fetch, feed_dict=feed_dict)
        except tf.errors.OpError as sub_exc:
          # We expect to get the same exception as the original one. The same op likely still fails.
          if sub_exc.op is op_copied:
            pass  # As expected.
          else:
            print("We tried to fetch the op inputs (%r) but got another exception:" % (list(op.inputs),), file=file)
            print(sub_exc, file=file)
            print("Maybe we still get some values via the fetch helpers, though...", file=file)
      except Exception as sub_exc:
        print("We tried to fetch the op inputs (%r) but got another exception:" % (list(op.inputs),), file=file)
        print(sub_exc, file=file)
        from returnn.util import better_exchook
        better_exchook.better_exchook(*sys.exc_info(), autodebugshell=False, file=file)
      else:
        print("Op inputs:", file=file)
        for input_t, fetch_helper in zip(op.inputs, fetch_helpers):
          info, _ = _help_data_or_array(fetch_helper.most_recent_value)
          print("  %r: %s" % (input_t, info), file=file)
    if op is None and isinstance(exception, tf.errors.InvalidArgumentError) and "Retval[0]" in exception.message:
      # E.g.: InvalidArgumentError: Retval[0] does not have value
      # Unfortunately, this TF exception does not give us any hint about the failing op.
      # Try to find it.
      if fetches is not None:
        # At least find out which of the fetches leads to the exception.
        found_fetch = None
        for fetch in fetches:
          try:
            session.run(fetch, feed_dict=feed_dict)
          except Exception as exc_:
            print("Exception for fetch %s: %s: %s" % (fetch, type(exc_).__name__, exc_), file=file)
            found_fetch = fetch
            break
        if found_fetch is not None:
          if isinstance(found_fetch, tf.Tensor):
            found_fetch = found_fetch.op
          assert isinstance(found_fetch, tf.Operation)
          # Try to go through op inputs.
          for fetch in list(found_fetch.control_inputs) + list(found_fetch.inputs):
            if isinstance(fetch, tf.Tensor) and fetch.op.type == "ScalarSummary":
              # Avoid error: Operation '...' has been marked as not fetchable
              fetch = tf_compat.v1.summary.merge([fetch])
            try:
              session.run(fetch, feed_dict=feed_dict)
            except Exception as exc_:
              print("Exception for fetch %s: %s: %s" % (fetch, type(exc_).__name__, exc_), file=file)
              input_to_output_ops = find_ops_path_output_to_input(list(feed_dict.keys()), fetches=fetch)
              print("Input to output op path:", file=file)
              pprint(input_to_output_ops, stream=file)
              if not input_to_output_ops:
                tf_util.print_graph_output(fetch, file=file)
              break
  print("Step meta information:", file=file)
  pprint(meta_step_info, stream=file)
  print("Feed dict:", file=file)
  if isinstance(feed_dict, dict):
    for key, value in sorted(feed_dict.items(), key=lambda item: item[0].name):
      assert isinstance(key, tf.Tensor)
      info, v_minmax = _help_data_or_array(value)
      data = None
      if key.name.startswith("extern_data/"):
        data_key = get_base_name(key)
        if extern_data and data_key in extern_data.data:
          data = extern_data.data[data_key]
          info += ", %s" % data
      print("  %r: %s" % (key, info), file=file)
      if data and data.sparse:
        if v_minmax[0] < 0 or v_minmax[1] >= data.dim:
          print("  WARNING, invalid label for data", data, file=file)
  elif feed_dict is None:
    print("None", file=file)
  else:
    print("(unexpected type %r)" % type(feed_dict), file=file)
    pprint(feed_dict, stream=file)


class CustomCheckpointLoader:
  """
  This uses `tf.train.NewCheckpointReader`.

  It would do automatic conversions if needed, e.g. between different LSTM implementations.
  However, be careful that for some LSTM implementation, there is an additional ``forget_bias``
  option, which is an additional scalar which gets added (not to the param, but to the forget value directly).
  When we convert the parameters, this is ignored, and you must take care about that explicitly
  to make sure you get the same results.

  It tries to automatically resolve renames, similar to this:

    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/tools/checkpoint_convert.py

  Also see:

    https://github.com/tensorflow/tensorflow/issues/11168
    https://github.com/tensorflow/tensorflow/commit/92da8abfd35b93488ed7a55308b8f589ee23b622
    https://github.com/tensorflow/tensorflow/commit/157370e5916b85c65958ed8383ae31d727228ed7

  """

  def __init__(self, filename, saveable_params, params_prefix="", load_if_prefix="", ignore_missing=False,
               ignore_params=(), ignore_params_prefixes=(),
               network=None):
    """
    :param str filename: filepattern for NewCheckpointReader
    :param list[tf.Variable|tensorflow.python.training.saver.BaseSaverBuilder.SaveableObject] saveable_params:
    :param str params_prefix: expect that all vars in saveable_params have this prefix, and remove it
    :param str load_if_prefix: if given, only load variables with a name containing this string.
      the variables in the file are expected to have the same name but without this string.
    :param bool ignore_missing: any vars in the model, which are not found in the checkpoint, will be ignored.
      however, if there is no single var in the checkpoint, this is still an error.
    :param typing.Container[str] ignore_params: these param (by name) will not be loaded
    :param typing.Iterable[str] ignore_params_prefixes: these param (by prefix name) will not be loaded
    :param TFNetwork network:
    """
    self.filename = filename
    self.network = network
    self.ignore_missing = ignore_missing
    self.params_prefix = params_prefix
    self.load_if_prefix = load_if_prefix
    self.saveable_params = []
    count = 0
    for param in saveable_params:
      param_name = self._get_param_name(param, assert_load_if_prefix_match=False)
      if load_if_prefix and param_name is None:
        continue
      if param_name in ignore_params:
        print("%s: Ignoring variable %s" % (self, param), file=log.v3)
        continue
      if any([param_name.startswith(prefix) for prefix in ignore_params_prefixes]):
        print("%s: Ignoring variable %s" % (self, param), file=log.v3)
        continue
      count += 1
      if have_custom_post_init(param):
        print("%s: Not loading pre-initialized variable %s" % (self, param), file=log.v3)
        continue
      self.saveable_params.append(param)
    assert count > 0, "%s: no saveable vars" % self
    self.reader = tf_compat.v1.train.NewCheckpointReader(filename)
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
      :param tf.compat.v1.Session session:
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
      :param CustomCheckpointLoader.CustomParamImporter|None custom_param_importer:
      """
      assert value is not None or custom_param_importer
      self.value = value
      self.custom_param_importer = custom_param_importer

    def assign_var(self, var, session):
      """
      :param tf.Variable var:
      :param tf.compat.v1.Session session:
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
      "lstm_cell/bias": "rnn/lstm_cell/bias",
      "lstm_cell/kernel": "rnn/lstm_cell/weights",
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

    # noinspection PyShadowingNames
    def make_load_weights_nativelstm_to_basic(new_name, postfix):
      """
      :param str new_name:
      :param str postfix: "/lstm_cell/kernel" or "/rnn/lstm_cell/kernel"
      :rtype: ()->numpy.ndarray
      """
      assert new_name.endswith(postfix)
      # noinspection PyShadowingNames
      old_name1 = new_name[:-len(postfix)] + "/W_re"
      # noinspection PyShadowingNames
      old_name2 = new_name[:-len(postfix)] + "/W"

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

    # noinspection PyShadowingNames
    def make_load_bias_nativelstm_to_basic(new_name, postfix):
      """
      :param str new_name:
      :param str postfix: "/lstm_cell/bias" or "/rnn/lstm_cell/bias"
      :rtype: ()->numpy.ndarray
      """
      assert new_name.endswith(postfix)
      # noinspection PyShadowingNames
      old_name = new_name[:-len(postfix)] + "/b"

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

    class MakeLoadBasicToNativeLstm:
      """
      BasicLSTM -> NativeLSTM converter.
      """
      def __init__(self, basic_kernel, basic_bias):
        """
        :param str basic_kernel:
        :param str basic_bias:
        """
        self.basic_kernel = basic_kernel
        self.basic_bias = basic_bias
        self._w_ff = None
        self._w_re = None
        self._bias = None

      def _calc(self):
        if self._w_ff is not None:
          return
        old_w_ff_re = reader.get_tensor(self.basic_kernel)  # (n_in+n_out,n_out*4)
        assert old_w_ff_re.ndim == 2
        old_bias = reader.get_tensor(self.basic_bias)  # (n_out*4,)
        assert old_bias.ndim == 1 and old_bias.shape[0] == old_w_ff_re.shape[1] and old_bias.shape[0] % 4 == 0
        n_out = old_bias.shape[0] // 4
        assert old_w_ff_re.shape[0] > n_out
        n_in = old_w_ff_re.shape[0] - n_out
        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        # BasicLSTM: i, j, f, o; Input: [inputs, h]
        # LstmGenericBase/NativeLstm: j, i, f, o
        # NativeLstm2: j, i, f, o
        old_w_ff_re_i, old_w_ff_re_j, old_w_ff_re_f, old_w_ff_re_o = numpy.split(old_w_ff_re, 4, axis=1)
        old_bias_i, old_bias_j, old_bias_f, old_bias_o = numpy.split(old_bias, 4, axis=0)
        new_w_ff_re = numpy.concatenate([old_w_ff_re_j, old_w_ff_re_i, old_w_ff_re_f, old_w_ff_re_o], axis=1)
        new_w_ff, new_w_re = numpy.split(new_w_ff_re, [n_in], axis=0)
        new_bias = numpy.concatenate([old_bias_j, old_bias_i, old_bias_f, old_bias_o], axis=0)
        self._w_ff = new_w_ff
        self._w_re = new_w_re
        self._bias = new_bias

      def get_w_re(self):
        """
        :rtype: numpy.ndarray
        """
        self._calc()
        return self._w_re

      def get_w(self):
        """
        :rtype: numpy.ndarray
        """
        self._calc()
        return self._w_ff

      def get_b(self):
        """
        :rtype: numpy.ndarray
        """
        self._calc()
        return self._bias

    class MakeLoadCudnnRnn:
      """
      Helper to load the CuDNN params.
      """

      cudnn_postfix = "/cudnn/CudnnRNNParamsToCanonical:0"

      # noinspection PyShadowingNames
      def __init__(self, prefix, target="lstm_block_wrapper/"):
        self.target = target
        self.keys = [target + "bias", target + "kernel"]
        self.prefix = prefix
        self.data = None  # type: typing.Optional[typing.Dict[str,numpy.ndarray]]

      # noinspection PyMethodParameters
      def _load(sself):
        from returnn.tf.layers.rec import RecLayer
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
      # Check NativeLSTM -> BasicLSTM.
      for postfix in [
        "/rnn/lstm_cell/kernel", "/lstm_cell/kernel", "/rnn/basic_lstm_cell/kernel", "/basic_lstm_cell/kernel"]:
        if v.endswith(postfix):
          old_name1 = v[:-len(postfix)] + "/W_re"
          old_name2 = v[:-len(postfix)] + "/W"
          if old_name1 in obsolete_var_names and old_name2 in obsolete_var_names:
            var_name_map[v] = make_load_weights_nativelstm_to_basic(v, postfix=postfix)
            break
      for postfix in [
        "/rnn/lstm_cell/bias", "/lstm_cell/bias", "/rnn/basic_lstm_cell/bias", "/basic_lstm_cell/bias"]:
        if v.endswith(postfix):
          old_name = v[:-len(postfix)] + "/b"
          if old_name in obsolete_var_names:
            var_name_map[v] = make_load_bias_nativelstm_to_basic(v, postfix=postfix)
      # Check BasicLSTM -> NativeLSTM.
      if v.endswith("/rec/W_re"):
        prefix = v[:-len("/rec/W_re")]
        cur_name_w = "%s/rec/W" % prefix
        cur_name_b = "%s/rec/b" % prefix
        old_name_kernel = "%s/rec/rnn/lstm_cell/kernel" % prefix
        old_name_bias = "%s/rec/rnn/lstm_cell/bias" % prefix
        if (
              old_name_kernel in obsolete_var_names and
              old_name_bias in obsolete_var_names and
              cur_name_w in missing_var_names and
              cur_name_b in missing_var_names):
          loader = MakeLoadBasicToNativeLstm(basic_kernel=old_name_kernel, basic_bias=old_name_bias)
          var_name_map[v] = loader.get_w_re
          var_name_map[cur_name_w] = loader.get_w
          var_name_map[cur_name_b] = loader.get_b
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
    :param tf.compat.v1.Session session:
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
      :rtype: (tf.compat.v1.Session)->None
      """
      def var_post_init(session):
        """
        :param tf.compat.v1.Session session:
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
  in :func:`TFNetwork.initialize_params`.

  :param tf.Variable var:
  :param (tf.compat.v1.Session)->None func:
  """
  # This custom attribute is a big ugly but simple.
  # It's read in TFNetwork.initialize_params().
  assert callable(func)
  var.custom_post_init = func


def have_custom_post_init(var):
  """
  :param tf.Variable var:
  :return: whether :func:`set_custom_post_init` was called on this var, i.e. we have custom init
  :rtype: bool
  """
  custom_post_init = getattr(var, "custom_post_init", None)
  if custom_post_init:
    assert callable(custom_post_init)
    return True
  return False
