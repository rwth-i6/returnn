
"""
This module contains the layer base class :class:`LayerBase`,
and many canonical basic layers.
"""

from __future__ import print_function

import tensorflow as tf
import contextlib
import typing
import TFUtil
from Util import unicode, NotSpecified, CollectionReadCheckCovered
from TFUtil import Data, OutputWithActivation, CustomUpdate, dimshuffle, swapaxes
from Log import log


class LayerBase(object):
  """
  This is the base class for all layers.
  Every layer by default has a list of source layers `sources` and defines `self.output` which is of type :class:`Data`.
  It shares some common functionality across all layers, such as explicitly defining the output format,
  some parameter regularization, and more.

  If you want to implement your own layer::

      class YourOwnLayer(_ConcatInputLayer):  # e.g. either _ConcatInputLayer or LayerBase as a base
          " some docstring "
          layer_class = "your_layer_name"

          def __init__(self, your_kwarg1, your_opt_kwarg2=None, **kwargs):
              " docstring, document the args! "
              super(YourOwnLayer, self).__init__(**kwargs)
              # Now we need to set self.output, which must be of type :class:`Data`.
              # It is set at this point to whatever we got from `selfget_out_data_from_opts()`,
              # so it is enough if we set self.output.placeholder and self.output.size_placeholder,
              # but we could also reset self.output.
              self.output.placeholder = self.input_data.placeholder + 42  # whatever you want to do
              # If you don't modify the sizes (e.g. sequence-length), just copy the input sizes.
              self.output.size_placeholder = self.input_data.size_placeholder.copy()

          @classmethod
          def get_out_data_from_opts(cls, **kwargs):
              " This is supposed to return a :class:`Data` instance as a template, given the arguments. "
              # example, just the same as the input:
              return get_concat_sources_data_template(kwargs["sources"], name="%s_output" % kwargs["name"])

  """

  layer_class = None  # type: typing.Optional[str]  # for get_layer_class()
  recurrent = False  # if the order in the time-dimension is relevant
  allow_inf_in_output = False

  # For compatibility, we have some parameter names (e.g. "L2") which do not conform to PEP8.
  # noinspection PyPep8Naming
  def __init__(self, name, network, output=None, n_out=NotSpecified, out_type=None, sources=(),
               target=None, _target_layers=None, loss=None, size_target=None,
               reuse_params=None,
               param_device=None,
               is_output_layer=None, only_on_eval=False, only_on_search=False,
               copy_output_loss_from_source_idx=None,
               batch_norm=False,
               L2=None, darc1=None,
               spatial_smoothing=0.0,
               param_variational_noise=None,
               updater_opts=None,
               initial_output=None,
               rec_previous_layer=None,
               collocate_with=None,
               trainable=True,
               custom_param_importer=None,
               register_as_extern_data=None):
    """
    Usually the arguments, when specified in the network dict,
    are going through :func:`transform_config_dict`, before they are passed to here.
    See :func:`TFNetwork.construct_from_dict`.

    :param str name:
    :param TFNetwork.TFNetwork network:
    :param Data output:
    :param NotSpecified|None|int n_out: output dim
    :param dict[str] out_type: kwargs for Data class. more explicit than n_out.
    :param list[LayerBase] sources: via self.transform_config_dict()
    :param str|list[str]|None target: if some loss is set, this is the target data-key,
      i.e. network.extern_data.get_data(target). alternatively, this also can be a layer name.
    :param dict[str,LayerBase]|None _target_layers: if target.startswith("layer:"), then this is target -> layer
    :param str|None size_target: like target but this is only used to set our output size in case of training
    :param Loss|None loss: via :func:`transform_config_dict`.
      Every layer can have one loss (of type :class:`Loss`), or none loss.
      In the net dict, it is specified as a string.
      In :class:`TFNetwork`, all losses from all layers will be collected.
      That is what :class:`TFUpdater.Updater` will use for training.
    :param ReuseParams|None reuse_params: if given, will opt reuse the params. see :func:`self.var_creation_scope`
    :param str|None param_device: e.g. "CPU", etc. any valid name for tf.device.
      see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/device_name_utils.h
    :param float|None L2: for constraints
    :param float|None darc1: for constraints. see Generalization in Deep Learning, https://arxiv.org/abs/1710.05468
    :param float|None spatial_smoothing: see :func:`TFUtil.spatial_smoothing_energy`
    :param float|None param_variational_noise: adds variational noise to the params during training
    :param dict[str]|None updater_opts: accepts similar opts as TFUpdater, e.g. "optimizer", "learning_rate", ...
    :param bool|None is_output_layer:
    :param bool only_on_eval: if True, this layer will only be calculated in eval
    :param bool only_on_search: if True, this layer will only be calculated when search is done
    :param int|None copy_output_loss_from_source_idx: if set, will copy output_loss from this source
    :param bool|dict batch_norm: see self.batch_norm()
    :param str|float initial_output: used for recurrent layer, see self.get_rec_initial_output()
    :param LayerBase|None rec_previous_layer: via the recurrent layer, layer (template) which represents the past of us
    :param list[LayerBase]|None collocate_with: in the rec layer, collocate with the specified other layers
    :param bool trainable: whether the parameters of this layer will be trained
    :param str|callable|None custom_param_importer: used by :func:`set_param_values_by_dict`
    :param str|None register_as_extern_data: registers output in network.extern_data
    """
    self.name = name
    self.network = network
    self._register_layer()
    self.kwargs = None  # type: typing.Optional[typing.Dict[str]] # set via self.post_init
    self.target = None
    self.targets = None
    if target:
      if isinstance(target, list):
        self.targets = target
        self.target = target[0]
      else:
        assert isinstance(target, str)
        self.targets = [target]
        self.target = target
    self._target_layers = _target_layers
    self.loss = loss
    if self.loss and self.loss.recurrent:
      self.recurrent = True
    if output:
      self.output = output
      if n_out is not NotSpecified:
        assert self.output.dim == n_out
      if isinstance(out_type, dict):
        if "shape" in out_type:
          assert self.output.shape == out_type["shape"]
        if "dim" in out_type:
          assert self.output.dim == out_type["dim"]
    else:
      self.output = self.get_out_data_from_opts(
        out_type=out_type, n_out=n_out,
        network=network, name=name, target=target, size_target=size_target,
        sources=sources, loss=loss)
    self.output_before_activation = None  # type: typing.Optional[OutputWithActivation]
    self.output_loss = None  # type: typing.Optional[tf.Tensor]
    if copy_output_loss_from_source_idx is not None:
      self.output_loss = sources[copy_output_loss_from_source_idx].output_loss
    self.rec_vars_outputs = {}  # type: typing.Dict[str,tf.Tensor]
    self.search_choices = None  # type: typing.Optional[SearchChoices]
    self._initial_output = initial_output
    self._rec_previous_layer = rec_previous_layer
    self.collocate_with = collocate_with or []
    self.post_init_hooks = []  # list of functions
    self.sources = sources
    self.params = {}  # type: typing.Dict[str,tf.Variable]
    self.saveable_param_replace = {}  # type:  typing.Dict[tf.Variable,typing.Union['tensorflow.python.training.saver.BaseSaverBuilder.SaveableObject',None]]  # see get_saveable_params_dict()  # nopep8
    self.reuse_params = reuse_params
    self.param_device = param_device
    self.L2 = L2
    self.darc1 = darc1
    self.spatial_smoothing = spatial_smoothing
    self.param_variational_noise = param_variational_noise
    self.updater_opts = CollectionReadCheckCovered(updater_opts or {})
    self._is_output_layer = is_output_layer
    self.only_on_eval = only_on_eval
    self.only_on_search = only_on_search
    self.use_batch_norm = batch_norm
    self.trainable = trainable
    self.custom_param_importer = custom_param_importer
    self.register_as_extern_data = register_as_extern_data
    # Stats will be collected by the engine.
    self.stats = {}  # type: typing.Dict[str,tf.Tensor]

  def post_init(self, layer_desc):
    """
    This gets called right after self.__init__().

    :param dict[str] layer_desc: kwargs as they are passed to self.__init__
    """
    self.kwargs = layer_desc
    if self.use_batch_norm:
      opts = {}
      if isinstance(self.use_batch_norm, dict):
        opts = self.use_batch_norm
      self.output.placeholder = self.batch_norm(self.output, **opts)
    if self.register_as_extern_data:
      self.network.extern_data.extra_added_keys.add(self.register_as_extern_data)
      self.network.extern_data.data[self.register_as_extern_data] = self.output
    for func in self.post_init_hooks:
      func()

  def __repr__(self):
    return "<%s %r out_type=%s>" % (
      self.__class__.__name__, self.name, self.output.get_description(with_name=False) if self.output else None)

  @classmethod
  def get_out_data_from_opts(cls, **kwargs):
    """
    Gets a Data template (i.e. shape etc is set but not the placeholder) for our __init__ args.
    The purpose of having this as a separate classmethod is to be able to infer the shape information
    without having to construct the layer.
    This function should not create any nodes in the computation graph.

    :param kwargs: all the same kwargs as for self.__init__()
    :return: Data template (placeholder not set)
    :rtype: Data
    """
    return cls._base_get_out_data_from_opts(**kwargs)

  @classmethod
  def _base_get_out_data_from_opts(cls, network, name, out_type=None, n_out=NotSpecified,
                                   target=None, _target_layers=None, size_target=None,
                                   sources=(), loss=None,
                                   **kwargs):
    """
    Called via BaseLayer.get_out_data_from_opts().

    :param TFNetwork.TFNetwork network:
    :param str name:
    :param dict[str]|None|(()->Data) out_type:
    :param int|None|NotSpecified n_out:
    :param str|list[str]|None target:
    :param dict[str,LayerBase]|None _target_layers: if target.startswith("layer:"), then this is target -> layer
    :param str|None size_target:
    :param list[LayerBase] sources:
    :param Loss|None loss:
    :param kwargs: remaining kwargs of self.__init__(), ignored here
    :return: Data template (placeholder not set)
    :rtype: Data
    """
    if callable(out_type):
      return out_type(
        network=network, name=name, n_out=n_out, target=target, size_target=size_target, sources=sources, loss=loss,
        **kwargs)
    if out_type is None:
      out_type = {}
    else:
      out_type = out_type.copy()
    out_type.setdefault("name", "%s_output" % name)
    if "dim" not in out_type and n_out is not NotSpecified:
      out_type["dim"] = n_out
    if "dim" not in out_type and target:
      out_type["dim"] = cls._static_get_target_value(
        target=target[0] if isinstance(target, list) else target, _target_layers=_target_layers,
        network=network, mark_data_key_as_used=False).dim
    if n_out is not NotSpecified:
      assert out_type["dim"] == n_out
    sources_data = None
    if sources and sources[0]:
      sources_data = sources[0].output.copy_template()
    if sources_data and not sources_data.sparse and not out_type.get("sparse", False):
      out_type.setdefault("dtype", sources_data.dtype)
    # You are supposed to set self.output.{batch_dim_axis,time_dim_axis} explicitly,
    # as well as check the inputs if they are as you would suggest.
    # However, a good default is often to use the same as the input.
    if all([k not in out_type for k in Data.SpecialAxesNames]):
      if sources_data:
        out_type.setdefault("batch_dim_axis", sources_data.batch_dim_axis)
        out_type.setdefault("time_dim_axis", sources_data.time_dim_axis)
        if not out_type.get("sparse", False) and sources_data.feature_dim_axis_or_unspecified is not NotSpecified:
          if sources_data.feature_dim_axis_or_unspecified is not None:
            out_type.setdefault("feature_dim_axis", sources_data.feature_dim_axis_or_unspecified)
          else:  # None
            if out_type.get("dim", None) is None:
              out_type.setdefault("feature_dim_axis", None)
      elif network.is_inside_rec_layer():
        out_type.setdefault("time_dim_axis", None)
    if "shape" not in out_type:
      if sources_data:
        if out_type.get("sparse", False):
          out_type.setdefault("shape", sources_data.shape_sparse)
        else:  # not sparse
          feature_dim_axis = out_type.get("feature_dim_axis", NotSpecified)
          if feature_dim_axis is NotSpecified:
            if sources_data.feature_dim_axis is not None:
              feature_dim_axis = sources_data.feature_dim_axis
            else:
              feature_dim_axis = -1
          default_shape = list(sources_data.shape_dense)
          default_shape.insert(sources_data.batch_dim_axis, None)
          default_shape[feature_dim_axis] = out_type["dim"]
          default_shape.pop(out_type.get("batch_dim_axis"))
          out_type.setdefault("shape", tuple(default_shape))
      elif network.is_inside_rec_layer():
        if out_type.get("sparse", False):
          out_type.setdefault("shape", ())
        else:
          out_type.setdefault("shape", (out_type.get("dim", None),))
    # Note: No special handling for feature_dim_axis here for now...
    beam_size = None
    for src in sources:
      if src:  # might be None if template construction
        beam_size = beam_size or src.output.beam_size
    out_type.setdefault("beam_size", beam_size)
    output = Data(**out_type)
    cls._post_init_output(
      output=output, network=network, target=target, size_target=size_target, _target_layers=_target_layers,
      sources=sources, **kwargs)
    return output

  # noinspection PyUnusedLocal
  @classmethod
  def _post_init_output(cls, output, network, target=None, size_target=None, _target_layers=None, sources=(), **kwargs):
    """
    :param Data output:
    :param TFNetwork.TFNetwork network:
    :param str|list[str]|None target:
    :param str|None size_target:
    :param dict[str,LayerBase]|None _target_layers: if target.startswith("layer:"), then this is target -> layer
    :param list[LayerBase] sources:
    """
    # You are supposed to set self.output.placeholder to the value which you want to return by the layer.
    # Normally you are also supposed to set self.output.size_placeholder explicitly, just like self.output.placeholder.
    # However, in many cases, this will just be {0: time-lengths} and the same as from the input.
    # We check for this case and preset it by that if possible.
    # If you want to have it different in your layer, just overwrite it.
    common_source = Data.get_common_data([s.output for s in sources if s])
    if not output.size_placeholder:
      if common_source and common_source.matches_var_dim_pattern(output):
        output.size_placeholder = common_source.size_placeholder.copy()
      elif target or size_target:
        if network.train_flag is not False:
          # TODO: In training, this is ok. Maybe as well as for eval but not clear.
          # In forward, mark_data_key_as_used=False should be used and anyway that target value is not available.
          output.size_placeholder = cls._static_get_target_value(
            target=(target[0] if (target and isinstance(target, list)) else target) or size_target,
            _target_layers=_target_layers,
            network=network, mark_data_key_as_used=network.train_flag is not False).size_placeholder.copy()
    if any([(src and not src.output.available_for_inference) for src in sources if src]):
      output.available_for_inference = False

  @classmethod
  def cls_get_tf_scope_name(cls, name):
    """
    :param str name: layer name
    :return: valid scope name, might be just name. see tf._VALID_SCOPE_NAME_REGEX and tf._VALID_OP_NAME_REGEX
    :rtype: str
    """
    from TFUtil import get_valid_scope_name_from_str
    return get_valid_scope_name_from_str(name)

  @classmethod
  def cls_layer_scope(cls, name):
    """
    Setup scope for layer. This can also be used when the layer does not yet exists.
    This is supposed to cover variable creations as well.
    Currently vars might be created when used within the rec-layer, but they are caught
    in a more generic way there, so we have not implemented yet any special logic here.

    :param str name: layer name
    :return: context manager object
    """
    @contextlib.contextmanager
    def layer_scope_ctx():
      """
      :return: context manager object
      """
      from TFUtil import reuse_name_scope
      with reuse_name_scope(cls.cls_get_tf_scope_name(name)) as scope:
        yield scope
    return layer_scope_ctx()

  @classmethod
  def get_global_layer_list(cls):
    """
    :rtype: list[LayerBase]
    """
    from TFUtil import CollectionKeys
    coll = tf.get_collection_ref(CollectionKeys.RETURNN_LAYERS)
    assert isinstance(coll, list)
    return coll

  @classmethod
  def get_recent_layer(cls):
    """
    :rtype: LayerBase
    """
    coll = cls.get_global_layer_list()
    assert coll
    return coll[-1]

  def _register_layer(self):
    self.get_global_layer_list().append(self)

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param TFNetwork.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
      The name `get_layer` might be misleading, as this should return an existing layer,
      or construct it if it does not exist yet.
      `network.get_layer` would just return an existing layer.

    Will modify `d` inplace such that it becomes the kwargs for `self.__init__()`.
    Mostly leaves `d` as-is.
    This is used by :func:`TFNetwork.construct_from_dict`.
    It resolves certain arguments,
    e.g. it resolves the `"from"` argument which is a list of strings,
    to make it the `"sources"` argument in kwargs, with a list of :class:`LayerBase` instances.
    Subclasses can extend/overwrite this.
    Usually the only reason to overwrite this is when some argument might be a reference to a layer
    which should be resolved.
    """
    src_names = d.pop("from", ["data"])
    if not isinstance(src_names, (list, tuple)):
      src_names = [src_names]
    d["sources"] = [
      get_layer(src_name)
      for src_name in src_names
      if not src_name == "none"]
    if "collocate_with" in d:
      collocate_with = d["collocate_with"]
      if not isinstance(collocate_with, (list, tuple)):
        collocate_with = [collocate_with]
      d["collocate_with"] = [get_layer(src_name) for src_name in collocate_with]
    if "reuse_params" in d:
      d["reuse_params"] = ReuseParams.from_config_dict(d["reuse_params"], network=network, get_layer=get_layer)
    if d.get("loss", None) and "target" not in d:
      target = get_loss_class(d["loss"]).get_default_target(network.extern_data)
      if target:
        d["target"] = target
    targets = None
    target_layers = {}
    assert "_target_layers" not in d
    if d.get("target", None):
      targets = d["target"]
      # we might have multiple targets, e.g. in choice layer, so convert to list
      if isinstance(targets, str):
        targets = [targets]
      if network.eval_flag:
        # _target_layers is a small workaround for further code which might not have access to the right get_layer.
        d["_target_layers"] = target_layers
        for target in targets:
          assert isinstance(target, str)
          # Not resolving this in the dict as target, but call get_layer to make it available.
          if target.startswith("layer:"):
            target_layers[target] = get_layer(target[len("layer:"):])
          else:
            # Note: This is a workaround for cases where we need to know about used data keys before the layer
            # itself is constructed (e.g. in _SubnetworkRecCell.get_output).
            # A nicer solution would be to not modify this here,
            # but instead lazily handle it in TFNetwork.get_extern_data,
            # such that we do not need to know in advance which data keys we need.
            # Also, if we are inside a rec layer, and doing search, we also cannot do that.
            if not network.is_inside_rec_layer() or not network.search_flag:
              network.used_data_keys.add(target)
    if "n_out" not in d and targets and network.eval_flag:
      # Must be done here now because loss might be set to None later.
      target = targets[0]  # guess using first target
      if target in target_layers:
        d["n_out"] = target_layers[target].output.dim
      else:
        d["n_out"] = cls._guess_n_out_from_target_and_opt_loss(
          network=network, target=target, loss_class_name=d.get("loss", None), get_layer=get_layer)
    if d.pop("loss_only_on_non_search", None) and network.search_flag:
      d.pop("loss", None)
      d.pop("loss_scale", None)
      d.pop("loss_opts", None)
    if d.get("loss", None):
      loss_opts = d.pop("loss_opts", None)
      if not loss_opts:
        loss_opts = {}
      # loss_scale: scale factor for loss (1.0 by default). DEPRECATED: use loss.scale instead, via loss_opts
      loss_scale = d.pop("loss_scale", 1.0)
      if loss_scale != 1.0:
        assert loss_opts.get("scale", 1.0) == 1.0, "do not use loss_scale and loss with 'scale' option together"
        loss_opts["scale"] = loss_scale
      d["loss"] = cls._make_loss(
        class_name=d.pop("loss", None), opts=loss_opts, network=network, get_layer=get_layer)
    else:
      assert "loss_scale" not in d, "loss not defined, do not set loss_scale"
      assert "loss_opts" not in d, "loss not defined, do not set loss_opts"

  @classmethod
  def _guess_n_out_from_target_and_opt_loss(cls, network, target, loss_class_name, get_layer):
    """
    :param TFNetwork.TFNetwork network:
    :param str target: e.g. "classes"
    :param str|None loss_class_name: e.g. "ce" or None
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    :return: n_out value
    :rtype: int
    """
    n_out = cls._static_get_target_value(
      target=target, network=network, mark_data_key_as_used=False, get_layer=get_layer).dim
    if loss_class_name:
      n_out = get_loss_class(loss_class_name).get_auto_output_layer_dim(n_out)
    return n_out

  @classmethod
  def _make_loss(cls, class_name, opts, network, get_layer):
    """
    :param str|None class_name:
    :param dict[str]|None opts:
    :param TFNetwork.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    :rtype: Loss|None
    """
    if not network.eval_flag:
      # Don't resolve the loss opts on purpose.
      # This might result in a smaller network because it might skip some get_layer calls.
      # This is what we want, i.e. we don't want to resolve layers which are only needed for the loss.
      return None
    if not class_name:
      return None
    if not opts:
      opts = {}
    opts = opts.copy()
    loss_class = get_loss_class(class_name)
    assert issubclass(loss_class, Loss)
    loss_class.transform_config_dict(opts, network=network, get_layer=get_layer)
    loss = loss_class(base_network=network, **opts)
    assert isinstance(loss, Loss)
    return loss

  @property
  def tf_scope_name(self):
    """
    :rtype: str
    :return: normally just self.name, but make it a valid TF scope name
    """
    return self.cls_get_tf_scope_name(name=self.name)

  def get_base_absolute_name_scope_prefix(self):
    """
    :return: e.g. "output/", always with "/" at end
    :rtype: str
    """
    return self.network.get_absolute_name_scope_prefix() + self.tf_scope_name + "/"

  def get_absolute_name_scope_prefix(self):
    """
    :return: e.g. "output/", always with "/" at end
    :rtype: str
    """
    return self.get_base_absolute_name_scope_prefix()

  def is_output_layer(self):
    """
    Some code differs between an output layer and other layers.
    It is a bit arbitrary what we define as output layer.
    This should be consistent with :func:`TFNetwork.construct_from_dict`.

    :rtype: bool
    """
    if self._is_output_layer is not None:
      return self._is_output_layer
    if self.loss:
      return True
    if self.name == "output":
      return True
    return False

  def get_dep_layers(self):
    """
    :return: list of layers this layer depends on.
      normally this is just self.sources but e.g. the attention layer in addition has a base, etc.
    :rtype: list[LayerBase]
    """
    layers = list(self.sources) + list(self.collocate_with)
    if self._target_layers:
      layers += [layer for _, layer in sorted(self._target_layers.items())]
    return layers

  def get_sub_layer(self, layer_name):
    """
    The default behavior for any layer is to return None.

    :param str layer_name: name of the sub_layer (right part of '/' separated path)
    :return: the sub_layer addressed in layer_name or None if no sub_layer exists
    :rtype: LayerBase|None
    """
    return None

  @classmethod
  def get_sub_layer_out_data_from_opts(cls, layer_name, parent_layer_kwargs):
    """
    Called by _TemplateLayer.get_sub_layer(). Gets a Data template for the sub-layer with name 'layer_name'.
    Also returns the network the sub-layer is in and the class type of the sub-layer. There is no good default
    behaviour here, as this heavily depends on how the current layer uses sub-layers.

    :param str layer_name: name of the sub_layer (right part of '/' separated path)
    :param dict[str] parent_layer_kwargs: kwargs for the parent layer (as kwargs in cls.get_out_data_from_opts())
    :return: Data template, network and the class type of the sub-layer
    :rtype: (Data, TFNetwork, type)|None
    """
    return None

  def get_search_choices(self):
    """
    :rtype: SearchChoices|None
    """
    if self.search_choices:
      return self.search_choices
    layer = self.network.get_search_choices(src=self)
    if layer:
      assert layer.search_choices
      return layer.search_choices
    return None

  def get_search_beam_size(self):
    """
    :return: beam size if there was a choice layer and we do search
    :rtype: int|None
    """
    if self.network.search_flag:
      choices = self.get_search_choices()
      if choices:
        return choices.beam_size
    return None

  def get_batch_dim(self):
    """
    The batch dim by this layer, not taken from our output but calculated.
    Normally it is self.network.get_batch_dim()
    but if we do search and there was a choice layer, it it multiplied by the beam size.
    :return: batch dim * beam size
    :rtype: tf.Tensor
    """
    batch_dim = self.network.get_data_batch_dim()
    beam_size = self.get_search_beam_size()
    if beam_size is not None:
      with tf.name_scope("batch_beam_dim"):
        batch_dim *= beam_size
    return batch_dim

  @contextlib.contextmanager
  def var_creation_scope(self, **kwargs):
    """
    This takes care of setting up a scope where variables can be created.

    :param kwargs: passed to variable_scope
    :return: yields the variable_scope
    """
    from TFUtil import get_current_var_scope_name, reuse_name_scope
    self_base_scope = self.get_base_absolute_name_scope_prefix()
    assert self_base_scope.endswith("/")
    cur_scope = get_current_var_scope_name()
    assert (cur_scope + "/").startswith(self_base_scope)
    # There are cases were a dummy layer was created already to create the variables,
    # e.g. see ReuseParams.LazyLayerResolver.
    kwargs = kwargs.copy()
    kwargs.setdefault("reuse", getattr(tf, "AUTO_REUSE", None))

    param_variational_noise = self.param_variational_noise
    if param_variational_noise is None:
      param_variational_noise = self.network.get_config().float("param_variational_noise", 0)
    if self.network.train_flag is False:  # if True or tf.Tensor, it will use cond_on_train below
      param_variational_noise = None
    need_custom_getter = bool(param_variational_noise)  # and param.dtype.is_floating
    kwargs_custom_getter = kwargs.get("custom_getter", None)

    def layer_custom_getter(getter, **getter_kwargs):
      """
      See TF docs :func:`_VariableStore.get_variable`.

      :param (...)->tf.Variable getter:
      :rtype: tf.Variable|tf.Tensor
      """
      if kwargs_custom_getter:
        param = kwargs_custom_getter(getter, **getter_kwargs)
      else:
        param = getter(**getter_kwargs)

      # Only apply this if we get a variable. Otherwise, maybe variational noise was already applied
      # (by some parent var scope), and we don't want to apply it twice.
      if param_variational_noise and param.dtype.is_floating and isinstance(param, tf.Variable):
        with tf.name_scope("param_variational_noise"):
          param = self.network.cond_on_train(
            fn_train=lambda: param + tf.random_normal(
              tf.shape(param), dtype=param.dtype.base_dtype,
              stddev=param_variational_noise,
              seed=self.network.random.randint(2 ** 31)),
            fn_eval=lambda: param)

      return param

    @contextlib.contextmanager
    def inner():
      """
      Var creation scope + variable scope.
      """
      if self.reuse_params:
        var_scope = self.reuse_params.get_variable_scope(base_layer=self)
      else:
        var_scope = tf.get_variable_scope()
      if need_custom_getter:
        kwargs["custom_getter"] = layer_custom_getter
      with reuse_name_scope(var_scope, **kwargs) as scope_:
        yield scope_

    if self.param_device:
      device_name = self.param_device
      if ":" not in device_name:
        device_name = "%s:*" % device_name
      if "/" not in device_name:
        device_name = "/device:%s" % device_name
      with tf.device(device_name):
        with inner() as scope:
          yield scope
    else:
      with inner() as scope:
        yield scope

  def add_param(self, param, custom_update=None, trainable=None, saveable=None, axes_split_info=None):
    """
    :param tf.Variable|tf.Tensor param:
    :param None|CustomUpdate custom_update: will be applied in training, instead of taking the gradient
    :param bool|None trainable:
    :param bool|None saveable:
    :param list[list[int]]|None axes_split_info: e.g. [[n],[n]*4] for LSTM matrices
    :return: param
    :rtype tf.Variable
    """
    _param = param
    if isinstance(param, tf.Tensor):
      # This can happen with a custom_getter in tf.get_variable(), e.g. via self.reuse_params.
      # Check if we can still find the original variable.
      from tensorflow.contrib import graph_editor
      import re
      possible_params = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=re.escape(self.get_absolute_name_scope_prefix()))
      if not possible_params:
        # None found. Just return as-is.
        return param
      all_ops = graph_editor.get_backward_walk_ops([param.op], inclusive=False, control_inputs=False)
      all_1st_tensors = [op.outputs[0] for op in all_ops if len(op.outputs) == 1]
      # noinspection PyProtectedMember
      possible_params = [p for p in possible_params if p._ref() in all_1st_tensors]
      if not possible_params:
        # Not found. Just return as-is.
        return param
      assert len(possible_params) == 1
      param = possible_params[0]
    assert isinstance(param, tf.Variable)
    if not self.trainable:
      trainable_collection_ref = param.graph.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
      if param in trainable_collection_ref:
        trainable_collection_ref.remove(param)
    if trainable is None:
      trainable = param in param.graph.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
    if saveable is None:
      saveable = True
    if custom_update:
      assert trainable
      custom_update.set_on_var(param)
    if axes_split_info:
      from TFUtil import set_param_axes_split_info
      set_param_axes_split_info(param, axes_split_info)
    if self.reuse_params:
      name_scope_prefix = self.reuse_params.get_absolute_name_scope_prefix(base_layer=self, param=param)
    else:
      name_scope_prefix = self.get_absolute_name_scope_prefix()
    assert param.name
    assert param.name[:len(name_scope_prefix)] == name_scope_prefix
    assert param.name[-2:] == ":0"
    param_name = param.name[len(name_scope_prefix):-2]
    if param_name not in self.params:
      self.params[param_name] = param
    else:
      assert self.params[param_name] is param
    if not saveable:
      self.saveable_param_replace[param] = None
    if getattr(param, "RETURNN_layer", None) is None:
      param.RETURNN_layer = self
    if getattr(param, "RETURNN_updater_opts", None) is None and self.updater_opts.truth_value:
      param.RETURNN_updater_opts = self.updater_opts
    # Note that any further postprocessing on the parameter should not be done here,
    # as we cannot guarantee that the result from this method is really used,
    # e.g. when we use official TF code such as the official LSTM cell.
    # The better way is to do it in self.var_creation_scope(), which also applies in those cases.
    return _param

  def set_param_values_by_dict(self, values_dict, session, ignore_wrong_shape=False, copy_param_mode=None):
    """
    :param dict[str,numpy.ndarray] values_dict:
    :param bool ignore_wrong_shape:
    :param str|None copy_param_mode:
    :param tf.Session session:
    """
    if callable(self.custom_param_importer):
      self.custom_param_importer(layer=self, values_dict=values_dict, session=session)
      return
    if self.custom_param_importer:
      copy_param_mode = self.custom_param_importer
    assert copy_param_mode in [None, "ifpossible", "subset"]
    if copy_param_mode:
      ignore_wrong_shape = True
    for param_name, values in values_dict.items():
      assert param_name in self.params, '%s: param %r unknown' % (self, param_name)
      param = self.params[param_name]
      assert isinstance(param, tf.Variable)
      shape = param.get_shape()
      assert isinstance(shape, tf.TensorShape)
      assert shape.is_fully_defined(), '%s: shape of param %r %r not fully defined?' % (self, param_name, param)
      param_shape = tuple(shape.as_list())
      if not ignore_wrong_shape:
        assert param_shape == values.shape, "var %r: shape %s != %s" % (param, shape.as_list(), values.shape)
      if param_shape != values.shape:
        if copy_param_mode == "subset":
          assert len(param_shape) == len(values.shape), "param %r ndim must match" % param
          new_values = session.run(param)  # use currently (randomly) initialized params as base
          param_axes_split_info = TFUtil.get_param_axes_split_info(param)
          if param_axes_split_info:
            TFUtil.check_param_axes_split_info(param.get_shape().as_list(), param_axes_split_info)
            old_axes_splits = TFUtil.transform_param_axes_split_info_to_new_shape(
              param_axes_split_info, values.shape)
            print("Param %r: transform old values of shape parts %r into new shape parts %r." % (
              param, old_axes_splits, param_axes_split_info), file=log.v3)
            values = TFUtil.copy_with_new_split_axes(
              old_axis_splits=old_axes_splits, new_axis_splits=param_axes_split_info,
              old_values=values, new_values=new_values)
          else:
            print("Param %r: transform old values of shape %r into new shape %r." % (
              param, values.shape, param_shape), file=log.v3)
            values = TFUtil.copy_with_new_split_axes(
              old_axis_splits=[[d] for d in values.shape],
              new_axis_splits=[[d] for d in param_shape],
              old_values=values, new_values=new_values)
        else:
          print(
            "Will not set param %r because its shape %s != %s." % (param, shape.as_list(), values.shape), file=log.v3)
          continue
      self.network.get_var_assigner(param).assign(values, session=session)

  def get_param_values_dict(self, session):
    """
    :param tf.Session session:
    :return: dict name -> values
    :rtype: dict[str,numpy.ndarray]
    """
    d = {}
    for param_name, param in self.get_saveable_params_dict().items():
      d[param_name] = param.eval(session)
    return d

  def get_saveable_params_dict(self):
    """
    :return: params and saveable_param_replace resolved
    :rtype: dict[str,tf.Variable|tensorflow.python.training.saver.BaseSaverBuilder.SaveableObject]
    """
    if not self.saveable_param_replace:
      return self.params.copy()
    d = {}
    for param_name, param in self.params.items():
      if param in self.saveable_param_replace:
        param = self.saveable_param_replace[param]
        if param is None:
          continue
      d[param_name] = param
    return d

  @staticmethod
  def _static_get_target_value(target, network, mark_data_key_as_used=True, _target_layers=None, get_layer=None):
    """
    :param str target:
    :param dict[str,LayerBase]|None _target_layers: if target.startswith("layer:"), then this is target -> layer
    :param TFNetwork.TFNetwork network:
    :param bool mark_data_key_as_used: forwarded self.network.get_extern_data()
    :param None|((str) -> LayerBase) get_layer: function to get or construct another layer
    :rtype: Data | None
    """
    if not target or target == "none":
      return None
    if _target_layers and target in _target_layers:
      return _target_layers[target].output
    if target.startswith("layer:"):
      if not get_layer:
        get_layer = network.get_layer
      return get_layer(target[len("layer:"):]).output
    assert network.extern_data.has_data(target), "target %r unknown" % target
    return network.get_extern_data(target, mark_data_key_as_used=mark_data_key_as_used)

  def _get_target_value(self, mark_data_key_as_used=True):
    """
    :param bool mark_data_key_as_used: forwarded self.network.get_extern_data()
    :rtype: Data | None
    """
    return self._static_get_target_value(
      target=self.target, _target_layers=self._target_layers,
      network=self.network, mark_data_key_as_used=mark_data_key_as_used)

  def _cond_only_on_eval_opt(self, on_eval_func, default_value):
    """
    :param ()->(tf.Tensor|None) on_eval_func:
    :param float|tf.Tensor default_value:
    :return: tensor (coming from tf.cond if needed) if on_eval_func returned a tensor, otherwise None
    :rtype: tf.Tensor|None
    """
    if not isinstance(default_value, tf.Tensor):
      default_value = tf.constant(default_value, name="only_on_eval_dummy_zero")

    class OnEval:
      """
      Closure.
      """
      have_output = True

      @classmethod
      def get_value(cls):
        """
        :rtype: tf.Tensor
        """
        res_ = on_eval_func()
        if res_ is None:
          cls.have_output = False
          return default_value  # Doesn't matter, will not be used anyway.
        return res_

    res = self.network.cond_on_train(lambda: default_value, OnEval.get_value)
    if not OnEval.have_output:
      return None
    return res

  @classmethod
  def get_losses(cls, name, network, output, loss=None, reduce_func=None, layer=None, **kwargs):
    """
    Losses will get constructed here.
    This gets called inside a loss name scope of the layer.
    When overriding this, make sure that it works both with `layer` set and unset.

    :param str name: layer name
    :param TFNetwork.TFNetwork network:
    :param Loss|None loss: argument just as for __init__
    :param Data output: the output (template) for the layer
    :param LayerBase|None layer:
      The real layer instance, if it exists at the current point.
      If not given, init() must be called at a later point.
    :param ((tf.Tensor)->tf.Tensor)|None reduce_func: if given, will overwrite the reduce func for the loss.
      By default, every loss_value and error_value is a scalar
      (sum or average over the batches, and over the frames for frame-wise losses).
      However, if you provide reduce_func = TFUtil.identity, you can get the unreduced tensor.
    :param kwargs: all the remaining __init__ args
    :return: the losses defined by this layer
    :rtype: list[TFNetwork.LossHolder]
    """
    if not loss:
      return []
    from TFNetwork import LossHolder
    return [LossHolder(
      name=name, network=network, loss=loss, layer_output=output, layer=layer, reduce_func=reduce_func)]

  def get_losses_initialized(self, reduce_func=None):
    """
    As self.get_losses, but here we return them all initialized (i.e. the layer is set).
    You should not override this method but rather :func:`get_losses`.

    :param ((tf.Tensor)->tf.Tensor)|None reduce_func: as in get_losses
    :return: the losses defined by this layer
    :rtype: list[TFNetwork.LossHolder]
    """
    return self.__class__.get_losses(reduce_func=reduce_func, layer=self, **self.kwargs)

  def get_params_l2_norm(self):
    """
    :return: scalar
    :rtype: tf.Tensor
    """
    return 2 * sum([tf.nn.l2_loss(param) for (name, param) in sorted(self.params.items())])

  def get_output_spatial_smoothing_energy(self):
    """
    :return: scalar. see :func:`TFUtil.spatial_smoothing_energy`
    :rtype: tf.Tensor
    """
    from TFUtil import spatial_smoothing_energy, flatten_with_seq_len_mask
    energy = spatial_smoothing_energy(self.output.placeholder, dim=self.output.dim)  # (batch,time)
    assert self.output.have_time_axis()
    energy = flatten_with_seq_len_mask(
      energy,
      seq_lens=self.output.size_placeholder[self.output.time_dim_axis_excluding_batch],
      time_major=self.output.is_time_major)  # (time')
    energy = tf.reduce_sum(energy)
    return energy

  def get_darc1(self):
    """
    DARC1, simplified Directly Approximately Regularizing Complexity (DARC), via
    Generalization in Deep Learning, https://arxiv.org/abs/1710.05468

    :return: scalar
    :rtype: tf.Tensor
    """
    with tf.name_scope("darc1"):
      if self.output_before_activation:
        x = self.output_before_activation.x
      else:
        x = self.output.placeholder
      mask = self.output.get_sequence_mask()  # (time,batch) or (batch,time), like output
      size = tf.size(mask)  # time * batch
      mask = tf.reshape(mask, (size,))  # (time*batch,)
      x = tf.reshape(x, (size,) + self.output.shape[1:])  # (time*batch,dim)
      x = tf.abs(x)
      x = tf.where(mask, x, tf.zeros_like(x))
      x = tf.reduce_sum(x, axis=0)  # (dim,)
      assert isinstance(x, tf.Tensor)
      assert x.get_shape().ndims == 1
      x = tf.reduce_max(x)  # scalar
      return x

  def get_constraints_value(self):
    """
    :return: None or scalar
    :rtype: tf.Tensor|None
    """
    c = 0
    if self.L2:
      c += self.L2 * self.get_params_l2_norm()
    if self.spatial_smoothing:
      c += self.spatial_smoothing * self.get_output_spatial_smoothing_energy()
    if self.darc1:
      c += self.darc1 * self.get_darc1()
    if c is 0:
      return None
    return c

  def batch_norm(self, data,
                 use_shift=True, use_std=True, use_sample=0.0, force_sample=False,
                 momentum=0.99, epsilon=1e-3,
                 sample_mean=None, sample_variance=None,
                 gamma=None, beta=None,
                 masked_time=True):
    """
    :param Data data:
    :param bool use_shift:
    :param bool use_std:
    :param float use_sample: defaults to 0.0 which is used in training
    :param bool force_sample: even in eval, use the use_sample factor
    :param float momentum: for the running average of sample_mean and sample_std
    :param float epsilon:
    :param tf.Tensor sample_mean:
    :param tf.Tensor sample_variance:
    :param tf.Tensor gamma:
    :param tf.Tensor beta:
    :param bool masked_time: flatten and mask input tensor
    :rtype: tf.Tensor

    http://arxiv.org/abs/1502.03167

    Also see:
      tf.nn.batch_normalization()
      https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/batch_norm.py
    """
    with tf.variable_scope("batch_norm"):
      if masked_time:
        x = data.get_placeholder_flattened(keep_dims=True)
        mean, variance = tf.nn.moments(x, axes=[0], keep_dims=True)
      else:
        x = data.placeholder
        mean, variance = tf.nn.moments(x, axes=data.get_axes(exclude_feature=True), keep_dims=True)
      if sample_mean is None:
        with self.var_creation_scope():
          sample_mean = self.add_param(tf.Variable(
            initial_value=tf.zeros(data.get_bc_spatial_batch_shape()),
            name="%s_%s_mean" % (self.name, data.name),
            trainable=False))
        # Use exponential moving average of batch mean.
        # Note: We could also use cumulative moving average. Our Theano implementation does that for inference.
        sample_mean = tf.assign_add(sample_mean, (mean - sample_mean) * momentum)
      if sample_variance is None:
        # Note: Our Theano implementation does not use a moving average for this.
        with self.var_creation_scope():
          sample_variance = self.add_param(tf.Variable(
            initial_value=tf.ones(data.get_bc_spatial_batch_shape()),
            name="%s_%s_variance" % (self.name, data.name),
            trainable=False))
        sample_variance = tf.assign_add(sample_variance, (variance - sample_variance) * momentum)
      # If train or if force_sample, use default use_sample=0.0, otherwise use_sample=1.0.
      use_sample = 1.0 + tf.cast(tf.logical_or(self.network.train_flag, force_sample), tf.float32) * (use_sample - 1.0)
      mean = (1. - use_sample) * mean + use_sample * sample_mean
      variance = (1. - use_sample) * variance + use_sample * sample_variance
      bn = (data.placeholder - mean) * tf.rsqrt(variance + epsilon)
      if use_std:
        if gamma is None:
          with self.var_creation_scope():
            gamma = self.add_param(tf.Variable(
              initial_value=tf.ones(data.get_bc_spatial_batch_shape()),
              name="%s_%s_gamma" % (self.name, data.name),
              trainable=True))
        bn *= gamma
      if use_shift:
        if beta is None:
          with self.var_creation_scope():
            beta = self.add_param(tf.Variable(
              initial_value=tf.zeros(data.get_bc_spatial_batch_shape()),
              name="%s_%s_beta" % (self.name, data.name),
              trainable=True))
        bn += beta
      return bn

  def get_hidden_state(self):
    """
    If this is a recurrent layer, this would return the hidden state.
    This is used e.g. for the RnnCellLayer class.

    :rtype: tf.Tensor | list[tf.Tensor] | None
    :return: optional tensor(s) with shape (time, batch, dim)
    """
    return None

  def get_last_hidden_state(self, key):
    """
    If this is a recurrent layer, this would return the last hidden state.
    Otherwise, we return None.

    :param int|str|None key: also the special key "*"
    :rtype: tf.Tensor | None
    :return: optional tensor with shape (batch, dim)
    """
    if key in self.rec_vars_outputs:
      return self.rec_vars_outputs[key]
    if key is None and len(self.rec_vars_outputs) == 1:
      return list(self.rec_vars_outputs.values())[0]
    assert not self.rec_vars_outputs
    return None

  def post_process_final_rec_vars_outputs(self, rec_vars_outputs, seq_len):
    """
    :param dict[str,tf.Tensor] rec_vars_outputs:
    :param tf.Tensor seq_len: shape (batch,)
    :rtype: dict[str,tf.Tensor]
    """
    return rec_vars_outputs

  @classmethod
  def get_rec_initial_output(cls, batch_dim, name, output, rec_layer, initial_output=None, **kwargs):
    """
    If this layer is used inside a recurrent layer, this function specifies the
    output of frame t=-1, if it is needed.
    As arguments, we get the usual layer arguments.
    batch_dim is added because it might be special because of beam search.

    Note: This could maybe share code with :func:`RnnCellLayer.get_rec_initial_state`.
    We could also add support to make the initial output be the output of another layer.

    :param tf.Tensor batch_dim: including beam size in beam search
    :param str name: layer name
    :param Data output: template
    :param TFNetworkRecLayer.RecLayer rec_layer:
    :param str|float|int|tf.Tensor|None initial_output:
    :rtype: tf.Tensor
    """
    import numpy
    v = initial_output
    data = output
    if isinstance(v, tf.Tensor):
      return v
    if v is None and data.sparse:
      raise Exception(
        ("You must explicitly provide an initial output value for sparse data %r." % data) +
        (" E.g. '%s': {'initial_output': 'zeros'}." % name))
    if v is None:
      v = "zeros"
    bc_shape = [(d if (d is not None) else 1) for d in data.batch_shape]
    # Some other code might not support automatic broadcasting in the batch-axis. (Earlier example: concat_in_time)
    # Thus we will automatically unbroadcast here (at least the batch axis).
    # Note that there still might be other axes which we do not unbroadcast here.
    # Thus, concat_in_time was fixed now, and maybe we actually do not need this anymore.
    shape = list(bc_shape)
    shape[data.batch_dim_axis] = batch_dim
    if isinstance(v, (float, int)):
      with tf.name_scope("init_%s_const" % name):
        from TFUtil import constant_with_shape
        return tf.cast(constant_with_shape(v, shape=shape), dtype=data.dtype)
    assert isinstance(v, str)
    if v == "zeros":
      return tf.zeros(shape, dtype=data.dtype, name="init_%s_zeros" % name)
    elif v == "ones":
      return tf.ones(shape, dtype=data.dtype, name="init_%s_ones" % name)
    elif v == "var":
      assert not data.sparse
      assert numpy.prod(bc_shape) == data.dim
      with rec_layer.var_creation_scope():
        x = tf.get_variable(
          "init_%s_var" % name, shape=(data.dim,), dtype=data.dtype, initializer=tf.zeros_initializer(dtype=data.dtype))
      x = tf.reshape(x, bc_shape, name="init_%s_var_bc" % name)
      x = tf.tile(x, [batch_dim if (i == data.batch_dim_axis) else 1 for i in range(data.batch_ndim)],
                  name="init_%s_var_batch_bc" % name)
      return x
    elif v == "apply(0)":
      # We will apply the layer for the input 0 and use this as the initial state.
      # This code might be a bit unstable.
      kwargs = kwargs.copy()
      sources = kwargs.pop("sources")
      zeroed_sources = []
      for src in sources:
        assert isinstance(src, LayerBase)
        src_output = src.output.copy()
        if src_output.placeholder is not None:
          zeroed_src_shape = tf.shape(src_output.placeholder)
          zeroed_src_shape = [zeroed_src_shape[i] for i in range(src_output.batch_ndim)]
        else:
          zeroed_src_shape = [(d if (d is not None) else 1) for d in src_output.batch_shape]
        if src_output.batch_dim_axis is not None:
          zeroed_src_shape[src_output.batch_dim_axis] = batch_dim
        src_output.placeholder = tf.zeros(
          zeroed_src_shape, dtype=src_output.dtype, name="init_%s_zeros" % src.name)
        src_output.sanity_check()
        zeroed_src = InternalLayer(name="%s_zeroed" % src.name, output=src_output, network=src.network)
        zeroed_sources.append(zeroed_src)
      layer = cls(name=name, output=output.copy(), sources=list(zeroed_sources), **kwargs)
      out = layer.output.placeholder
      out.set_shape(data.batch_shape)
      return out
    else:
      raise Exception("invalid initial output type %r for sub-layer %r" % (v, name))

  @classmethod
  def get_rec_initial_extra_outputs(cls, batch_dim, rec_layer, **kwargs):
    """
    :param tf.Tensor batch_dim: for this layer, might be with beam
    :param TFNetworkRecLayer.RecLayer rec_layer:
    :rtype: dict[str,tf.Tensor]
    """
    return {}

  @classmethod
  def get_rec_initial_extra_outputs_shape_invariants(cls, **kwargs):
    """
    :return: optional shapes for the tensors by get_rec_initial_extra_outputs
    :rtype: dict[str,tf.TensorShape]
    """
    return {}


class ReuseParams:
  """
  This is for parameter sharing, i.e. reusing existing `tf.Variable` objects in a new layer,
  instead of creating new variables.
  :func:`ReuseParams.from_config_dict` will be called via :func:`LayerBase.transform_config_dict`.
  """

  @classmethod
  def from_config_dict(cls, opts, network, get_layer):
    """
    This will be called via :func:`LayerBase.transform_config_dict` on the layer option `"reuse_params"`.

    :param str|dict[str]|None opts:
      If None, we will return None.
      If str, it will be interpret as a layer name.
      If dict, you can specify:
        "reuse_layer": layer name
        "map": dict where the keys are parameter names, and the values can be:
          A str would be interpret as a layer name.
          None would be interpret as the option `auto_create_missing`.
          A dict would specify :func:`ReuseParams.__init__` options.
            The option reuse_layer would be specified as a str, and represents a layer name.
    :param TFNetwork.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    :rtype: ReuseParams|None
    """
    if not opts:
      return None

    def optional_get_layer(layer_name):
      """
      :param str layer_name:
      :rtype: LayerBase|ReuseParams.LazyLayerResolver
      """
      from TFNetwork import NetworkConstructionDependencyLoopException
      try:
        return get_layer(layer_name)
      except NetworkConstructionDependencyLoopException:
        # This dependency loop is not seen as critical. We allow it to be done later.
        # So any template construction of this layer should work.
        return ReuseParams.LazyLayerResolver(layer_name=layer_name, network=network, get_layer=get_layer)

    if isinstance(opts, str):  # share the whole layer
      return ReuseParams(reuse_layer=optional_get_layer(opts))
    assert isinstance(opts, dict)
    opts = opts.copy()
    if "reuse_layer" in opts:  # share the whole layer (+ possibly some more params)
      opts["reuse_layer"] = optional_get_layer(opts["reuse_layer"])
    if "map" in opts:  # share specific parameters
      assert isinstance(opts["map"], dict), "reuse_params['map'] should be a dict but is %s" % (type(opts["map"]),)
      opts["map"] = opts["map"].copy()
      for key, value in sorted(opts["map"].items()):
        if isinstance(value, str):
          value = {"reuse_layer": optional_get_layer(value)}
        elif value is None:
          value = {"auto_create_missing": True}
        else:
          assert isinstance(value, dict)
          value = value.copy()
          if "reuse_layer" in value:
            value["reuse_layer"] = optional_get_layer(value["reuse_layer"])
        opts["map"][key] = ReuseParams(**value)
    return ReuseParams(**opts)

  class LazyLayerResolver:
    """
    Unfortunately this is a bit tricky and difficult to do right.
    We want to support it because it can happen that e.g. in training, this is layer resolving is not needed,
    and then in search, it is needed, due to different dependencies.
    See :func:`test_reuse_params_map_custom_dep_loop` for an example.
    The params depend on a layer which is not constructed yet and cannot be constructed yet
    because of a dependency loop.
    Thus, here we again try to create it, and if we still get the dependency loop,
    we create the reused-params-layer based on dummy inputs, such that the variables/parameters get created
    and can be used now. Then, later, we are going to recreate the reused-params-layer.
    """

    def __init__(self, layer_name, network, get_layer):
      """
      :param str layer_name:
      :param TFNetwork.TFNetwork network:
      :param ((str) -> LayerBase) get_layer:
      """
      self.layer_name = layer_name
      self.network = network
      self.get_layer_func = get_layer
      self.var_scope = tf.get_variable_scope()

    def get_layer(self):
      """
      :rtype: LayerBase
      """
      from TFNetwork import NetworkConstructionDependencyLoopException
      from TFUtil import reuse_name_scope
      with reuse_name_scope(self.var_scope):
        try:
          return self.get_layer_func(self.layer_name)
        except NetworkConstructionDependencyLoopException as exc:
          return self.create_dummy_layer(dep_loop_exception=exc)

    def create_dummy_layer(self, dep_loop_exception):
      """
      :param TFNetwork.NetworkConstructionDependencyLoopException dep_loop_exception:
      :rtype: LayerBase
      """
      print(
        ("ReuseParams: layer %r does not exist yet and there is a dependency loop, " +
         "thus creating it on dummy inputs now") % self.layer_name,
        file=log.v4)

      def opt_get_layer(layer_name):
        """
        :param str layer_name:
        :rtype: LayerBase|None
        """
        if layer_name in self.network.layers:
          return self.network.layers[layer_name]
        print("ReuseParams: non-existing layer %r, ignoring..." % layer_name, file=log.v4)
        return None

      def get_dummy_input_layer(layer_name):
        """
        :param str layer_name:
        :rtype: LayerBase
        """
        if layer_name in self.network.layers:
          return self.network.layers[layer_name]
        print("ReuseParams: creating dummy input %r" % layer_name, file=log.v4)
        layer_desc_ = dep_loop_exception.net_dict[layer_name].copy()
        class_name_ = layer_desc_.pop("class")
        layer_class_ = get_layer_class(class_name_)
        # noinspection PyProtectedMember
        layer_desc_ = self.network._create_layer_layer_desc(name=layer_name, layer_desc=layer_desc_)
        layer_class_.transform_config_dict(
          layer_desc_, network=self.network, get_layer=opt_get_layer)
        output = layer_class_.get_out_data_from_opts(**layer_desc_).copy()
        output.placeholder = tf.zeros(
          [d or 1 for d in output.batch_shape], dtype=output.dtype, name="%s_dummy" % output.name)
        output.sanity_check()
        return InternalLayer(name=layer_name, network=self.network, output=output)

      layer_desc = dep_loop_exception.net_dict[self.layer_name].copy()
      class_name = layer_desc.pop("class")
      layer_class = get_layer_class(class_name)
      layer_class.transform_config_dict(layer_desc, network=self.network, get_layer=get_dummy_input_layer)
      # noinspection PyProtectedMember
      return self.network._create_layer(name=self.layer_name, layer_class=layer_class, **layer_desc)

  # noinspection PyShadowingBuiltins
  def __init__(self, reuse_layer=None, map=None, custom=None, auto_create_missing=False):
    """
    :param LayerBase|ReuseParams.LazyLayerResolver|None reuse_layer:
    :param dict[str,ReuseParams]|None map:
    :param (**kwargs)->(tf.Tensor|tf.Variable) custom: see :func:`self.variable_custom_getter`
    :param bool auto_create_missing:
    """
    assert isinstance(reuse_layer, (LayerBase, ReuseParams.LazyLayerResolver)) or not reuse_layer
    self._reuse_layer = reuse_layer
    self.param_map = map
    self.custom_func = custom
    self.auto_create_missing = auto_create_missing

  @property
  def reuse_layer(self):
    """
    :rtype: LayerBase|None
    """
    if self._reuse_layer:
      if isinstance(self._reuse_layer, ReuseParams.LazyLayerResolver):
        self._reuse_layer = self._reuse_layer.get_layer()
      assert isinstance(self._reuse_layer, LayerBase)
      return self._reuse_layer
    return None

  def get_absolute_name_scope_prefix(self, base_layer, param):
    """
    :param LayerBase base_layer:
    :param tf.Variable param: e.g. "base_layer/rec/W"
    :return: e.g. "base_layer/" or "base_layer/rec/", always with "/" at end
    :rtype: str
    """
    abs_scope_prefix = base_layer.get_absolute_name_scope_prefix()  # e.g. "current_layer/" or "current_layer/rec/"
    assert abs_scope_prefix.endswith("/")
    from TFUtil import get_current_var_scope_name
    cur_scope = get_current_var_scope_name() + "/"  # e.g. "current_layer/rec/" or "current_layer/"
    assert cur_scope.startswith(abs_scope_prefix)
    assert param.name[-2:] == ":0"
    abs_param_name = param.name[:-2]
    param_name = abs_param_name.split("/")[-1]
    assert param_name
    assert abs_param_name.endswith("/" + param_name)
    if self.custom_func:  # Could be any base absolute name scope prefix, so just return what we have.
      return abs_param_name[:-len(param_name)]
    if self.param_map is not None and param_name in self.param_map:
      return self.param_map[param_name].get_absolute_name_scope_prefix(base_layer=base_layer, param=param)
    if self.reuse_layer and param_name in self.reuse_layer.params:
      reuse_layer_prefix = self.reuse_layer.get_absolute_name_scope_prefix()
      assert reuse_layer_prefix + param_name == abs_param_name
      return reuse_layer_prefix
    assert self.auto_create_missing
    base_layer_prefix = base_layer.get_absolute_name_scope_prefix()
    assert base_layer_prefix + param_name == abs_param_name
    return base_layer_prefix

  def get_variable_scope(self, base_layer, **kwargs):
    """
    :param LayerBase base_layer:
    :param kwargs: passed to tf.variable_scope
    :rtype: tf.VariableScope
    """
    def _variable_custom_getter(**kwargs_):
      return self.variable_custom_getter(base_layer=base_layer, **kwargs_)
    with tf.variable_scope(tf.get_variable_scope(), custom_getter=_variable_custom_getter, **kwargs) as scope:
      return scope

  def variable_custom_getter(self, getter, name, base_layer, **kwargs):
    """
    By TF docs, from :func:`_VariableStore.get_variable`:
    Callable that takes as a first argument the true getter,
    and allows overwriting the internal get_variable method.
    The signature of `custom_getter` should match that of this method,
    but the most future-proof version will allow for changes:
    `def custom_getter(getter, *args, **kwargs)`.  Direct access to
    all `get_variable` parameters is also allowed:
    `def custom_getter(getter, name, *args, **kwargs)`.  A simple identity
    custom getter that simply creates variables with modified names is:
    ```python
    def custom_getter(getter, name, *args, **kwargs):
      return getter(name + '_suffix', *args, **kwargs)
    ```
    In addition, we get the argument `base_scope_name`, via :func:`self.get_variable_scope`.

    :param (...)->tf.Variable getter:
    :param str name: absolute param name
    :param LayerBase base_layer: we expect that this is the prefix of ``name``
    :rtype: tf.Variable|tf.Tensor
    """
    abs_scope_prefix = base_layer.get_absolute_name_scope_prefix()
    assert not abs_scope_prefix or abs_scope_prefix.endswith("/")
    assert name.startswith(abs_scope_prefix)
    param_name = name[len(abs_scope_prefix):]  # e.g. "W" (not "rec/W")
    if self.custom_func:
      return self.custom_func(
        base_layer=base_layer, reuse_layer=self.reuse_layer, name=param_name, getter=getter, full_name=name, **kwargs)
    if self.param_map is not None:
      if not self.auto_create_missing:
        assert param_name in self.param_map
      if param_name in self.param_map:
        return self.param_map[param_name].variable_custom_getter(
          getter=getter, name=name, base_layer=base_layer, **kwargs)
    if self.reuse_layer:
      if not self.auto_create_missing:
        assert param_name in self.reuse_layer.params
      if param_name in self.reuse_layer.params:
        return self.reuse_layer.params[param_name]
    assert self.auto_create_missing
    return getter(name=name, **kwargs)


class SearchChoices(object):
  """
  In beam search, after expanding the beam and then selecting the N best (beam) (see :class:`ChoiceLayer`),
  when doing this multiple times, we need to keep reference where each beam came from,
  and what the current score is, etc.
  Also we could have multiple different such expansions & prunes via different :class:`ChoiceLayer`.
  This is what we keep track here.
  """

  def __init__(self, owner, src_beams=None, beam_size=None, is_decided=False):
    """
    :param LayerBase owner:
    :param tf.Tensor|None src_beams: (batch, beam) -> src beam index
    :param int|None beam_size:
    :param bool is_decided: by decide layer
    """
    self.owner = owner
    self._done_src_layer = False
    self._src_layer = None  # type: typing.Optional[LayerBase]
    self.src_beams = src_beams  # (batch, beam)
    self.beam_size = beam_size
    self.beam_scores = None  # type: typing.Optional[tf.Tensor]  # (batch, beam)
    self.is_decided = is_decided

  def __repr__(self):
    def short(v):
      """
      :param LayerBase|tf.Tensor|None v:
      :return: short repr
      :rtype: str
      """
      if isinstance(v, LayerBase):
        return repr(v.name)
      if isinstance(v, tf.Tensor):
        if v.get_shape().ndims is not None:
          return "shaped:(%s)" % ",".join(map(str, v.get_shape().as_list()))
        return "unknown-ndim"
      return repr(v)
    s = " beam_size=%r" % self.beam_size
    if self._done_src_layer:
      s += " src_layer=%s" % short(self._src_layer)
    s += " beam_scores=%s" % short(self.beam_scores)
    if self.is_decided:
      s += " is_decided"
    return "<SearchChoices owner=%s%s>" % (short(self.owner), s)

  @property
  def src_layer(self):
    """
    :return: The layer where we had the last search choices.
    :rtype: LayerBase
    """
    if not self._done_src_layer:
      self._src_layer = self.owner.network.get_search_choices(base_search_choice=self.owner)
      self._done_src_layer = True
    return self._src_layer

  def set_beam_scores_from_own_rec(self):
    """
    Assumes we have set self.owner, and uses those rec vars to set the beam scores.
    """
    self.set_beam_scores_from_rec(self.owner.rec_vars_outputs)

  def set_beam_scores_from_rec(self, rev_vars_outputs):
    """
    :param dict[str,tf.Tensor] rev_vars_outputs:
    """
    assert rev_vars_outputs.get("choice_scores", None) is not None
    self.beam_scores = rev_vars_outputs["choice_scores"]  # (batch, beam)
    if self.src_beams is not None:
      self.beam_scores.set_shape(self.src_beams.get_shape())

  def set_beam_scores(self, scores):
    """
    :param tf.Tensor scores: (batch, beam) -> log score
     """
    self.beam_scores = scores
    self.owner.rec_vars_outputs["choice_scores"] = scores

  def get_all_src_choices(self):
    """
    :return: all SearchChoices we depend on up to the root, including self
    :rtype: list[SearchChoices]
    """
    sources = [self]
    choice = self
    while True:
      src_layer = choice.src_layer
      if not src_layer:
        break
      assert isinstance(src_layer.search_choices, SearchChoices)
      choice = src_layer.search_choices
      if choice in sources:  # loop, can happen e.g. in RecLayer in the loop
        break
      sources.append(choice)
    return sources

  def __eq__(self, other):
    return self is other

  def __ne__(self, other):
    return self is not other

  @staticmethod
  def compare(self, other):
    """
    :param SearchChoices|None self:
    :param SearchChoices|None other:
    :return: 0 if equal, -1 if we are smaller, else 1
    :rtype: int
    """
    if self is other:
      return 0
    if self is None:
      return -1
    if other is None:
      return 1
    self_src_choices = self.get_all_src_choices()
    other_src_choices = other.get_all_src_choices()
    assert len(self_src_choices) != len(other_src_choices)
    if len(self_src_choices) < len(other_src_choices):
      res = -1
    else:
      res = 1
      self_src_choices, other_src_choices = other_src_choices, self_src_choices
    assert len(self_src_choices) < len(other_src_choices)
    for i in range(len(self_src_choices)):
      assert self_src_choices[-1 - i] == other_src_choices[-1 - i], (
        "cannot compare, they don't share the same search-tree")
    return res

  def __cmp__(self, other):
    return self.compare(self, other)

  def __lt__(self, other):
    return self.__cmp__(other) < 0

  def __gt__(self, other):
    return self.__cmp__(other) > 0

  def translate_to_this_search_beam(self, sources):
    """
    :param LayerBase|list[LayerBase]|dict[str,LayerBase|object]|tuple[LayerBase|object]|T sources:
    :return: sources but all layers transformed when needed
    :rtype: T
    """
    d = sources
    if isinstance(d, dict):
      return {k: self.translate_to_this_search_beam(v) for (k, v) in d.items()}
    if isinstance(d, (tuple, list)):
      from Util import make_seq_of_type
      return make_seq_of_type(type(d), [self.translate_to_this_search_beam(v) for v in d])
    if isinstance(d, LayerBase):
      if d.get_search_choices() == self:
        return d
      return SelectSearchSourcesLayer(sources=(d,), search_choices=self.owner, name=d.name, network=d.network)
    return d

  @classmethod
  def translate_to_common_search_beam(cls, sources):
    """
    :param list[LayerBase]|dict[str,LayerBase|object] sources:
    :return: sources but all layers transformed when needed
    :rtype: list[LayerBase]|dict[str,LayerBase|object]
    """
    from tensorflow.python.util import nest
    layers_flat = [v for v in nest.flatten(sources) if isinstance(v, LayerBase)]
    if len(layers_flat) <= 1:
      return sources
    from functools import cmp_to_key
    common_choices = max([layer.get_search_choices() for layer in layers_flat], key=cmp_to_key(cls.compare))
    if not common_choices:
      return sources
    return common_choices.translate_to_this_search_beam(sources)


class SourceLayer(LayerBase):
  """
  This gives access to some entry from network.extern_data (:class:`ExternData`).
  """
  layer_class = "source"

  def __init__(self, network, data_key=None, sources=(), **kwargs):
    """
    :param TFNetwork.TFNetwork network:
    :param str|None data_key:
    :param tuple sources:
    """
    if data_key is None:
      data_key = network.extern_data.default_input
    assert not sources, "source layer does not expect sources"
    data = network.get_extern_data(data_key, mark_data_key_as_used=True).copy()
    super(SourceLayer, self).__init__(network=network, **kwargs)
    if data.placeholder is None:
      raise Exception("%r: data %r:%r only exists as template. You can only use %r." % (
        self, data_key, data,
        {k: v for (k, v) in network.extern_data.data.items() if v.placeholder is not None}))
    self.output = data

  @classmethod
  def get_out_data_from_opts(cls, network, data_key=None, **kwargs):
    """
    :param TFNetwork.TFNetwork network:
    :param str|None data_key:
    :rtype: Data
    """
    if data_key is None:
      data_key = network.extern_data.default_input
    return network.get_extern_data(data_key, mark_data_key_as_used=False).copy()


@contextlib.contextmanager
def _name_scope_for_concat_src_layers(src_layers, postfix):
  """
  :param list[LayerBase] src_layers:
  :param str postfix:
  :return: yields scope via reuse_name_scope()
  """
  assert src_layers
  if len(src_layers) == 1:
    name_scope = src_layers[0].get_absolute_name_scope_prefix() + postfix
  else:
    base = src_layers[0].network.get_absolute_name_scope_prefix()
    name = "concat_" + "_".join([l.tf_scope_name for l in src_layers])
    name_scope = base + name + "/" + postfix
  from TFUtil import reuse_name_scope
  with reuse_name_scope(name_scope, absolute=True) as scope:
    yield scope


def concat_sources(src_layers):
  """
  :param list[LayerBase] src_layers:
  :return: data with placeholders set
  :rtype: Data
  """
  assert src_layers, "need source layers"
  if len(src_layers) == 1:
    return src_layers[0].output.copy()
  network = src_layers[0].network
  cache_key = (tuple(src_layers), 0.0, None)
  if cache_key in network.concat_sources_dropout_cache:
    return network.concat_sources_dropout_cache[cache_key].copy()
  data = get_concat_sources_data_template(src_layers)
  common_source = Data.get_common_data([s.output for s in src_layers])
  # Currently we assume that get_concat_sources_data_template will match Data.get_common_data (besides the dim).
  data.size_placeholder = common_source.size_placeholder.copy()  # to get right dimension tags
  layers_data = []
  with _name_scope_for_concat_src_layers(src_layers, "concat_sources"):
    data_dyn_shape = list(data.batch_shape)
    if any([d is None for d in data_dyn_shape]):
      # Currently we assume that get_concat_sources_data_template will match Data.get_common_data (besides the dim).
      assert common_source.batch_ndim == data.batch_ndim
      for axis in range(data.batch_ndim):
        if data_dyn_shape[axis] is None:
          data_dyn_shape[axis] = tf.shape(common_source.placeholder)[axis]
    for layer in src_layers:
      assert not layer.output.sparse, "sparse concat not supported"
      assert layer.output.dtype == data.dtype, "incompatible dtype with layer %r" % layer
      # unbroadcast is needed for tf.concat.
      layers_data.append(layer.output.copy_compatible_to(data, unbroadcast=True, data_dyn_shape=data_dyn_shape))
    data.placeholder = tf.concat(
      axis=data.feature_dim_axis,
      values=[l.placeholder for l in layers_data])
    axes_split_info = [None] * data.batch_ndim  # type: typing.List[typing.Optional[typing.List[int]]]
    axes_split_info[data.feature_dim_axis] = [l.dim for l in layers_data]
    TFUtil.set_param_axes_split_info(data.placeholder, axes_split_info)
    # Note: We will loose this info for any further op (e.g. dropout, activation, etc). Should be better...
    # Maybe instead in Data class?
    # Also note, even for tf.Variable, e.g. with weight noise, we might loose this?
  network.concat_sources_dropout_cache[cache_key] = data.copy()
  return data


def get_concat_sources_data_template(src_layers, name=None):
  """
  This just creates a template :class:`Data` instance,
  without creating any real TF tensors.
  :func:`concat_sources` (and related) are the equivalent functions
  which would create a :class:`Data` together with the tensor.

  :param list[LayerBase]|tuple[LayerBase] src_layers:
  :param str|None name: name of the Data
  :return: data with no placeholders set. it is always a copy or new instance, so safe to manipulate
  :rtype: Data
  """
  assert src_layers, "need source layers"
  if len(src_layers) == 1:
    return src_layers[0].output.copy(name=name)
  dim = 0
  beam_size = None
  common_source = Data.get_common_data([s.output for s in src_layers])
  for layer in src_layers:
    # Note: We do not perform much compatibility checks at this point,
    # as this is for a template only anyway.
    # The real checks are in concat_sources.
    assert not layer.output.sparse
    assert layer.output.dim is not None
    dim += layer.output.dim
    beam_size = beam_size or layer.output.beam_size
  shape = list(common_source.shape)
  shape[common_source.get_batch_axis_excluding_batch(common_source.feature_dim_axis)] = dim
  kwargs = common_source.get_kwargs()
  kwargs.update(dict(
    name=name or ("concat_" + "_".join([l.name for l in src_layers])),
    shape=shape,
    dim=dim,
    sparse=False,
    beam_size=beam_size))
  data = Data(**kwargs)
  return data


def concat_sources_with_opt_dropout(src_layers, dropout=0, dropout_noise_shape=None):
  """
  :param list[LayerBase] src_layers:
  :param float dropout: will be applied if train_flag is set
  :param tuple|list|dict|None dropout_noise_shape:
  :return: data with placeholders set
  :rtype: Data
  """
  assert src_layers, "need source layers"
  data = concat_sources(src_layers)
  network = src_layers[0].network
  if network.train_flag is False:
    # If we know that we are not training, we always disable dropout.
    dropout = 0
  if not dropout:
    return data.copy()
  assert not data.sparse, "need dense data when dropout is used; sources: %r" % (src_layers,)
  if isinstance(dropout_noise_shape, dict) or not dropout_noise_shape:
    # Default noise_shape behavior is like old for now:
    # All dynamic dimensions (batch,time) will use the same dropout-mask broadcasted.
    dropout_noise_shape = data.get_bc_shape(dropout_noise_shape)
  cache_key = (tuple(src_layers), float(dropout), tuple(dropout_noise_shape))
  if cache_key in network.concat_sources_dropout_cache:
    return network.concat_sources_dropout_cache[cache_key].copy()
  data = data.copy()
  assert 0.0 < dropout < 1.0
  with _name_scope_for_concat_src_layers(src_layers, "dropout_in_train"):
    import TFUtil
    data.placeholder = network.cond_on_train(
      fn_train=lambda: TFUtil.dropout(
        data.placeholder,
        keep_prob=1 - dropout,
        noise_shape=dropout_noise_shape,
        seed=network.random.randint(2 ** 31)),
      fn_eval=lambda: data.placeholder)
  network.concat_sources_dropout_cache[cache_key] = data.copy()
  return data


class _ConcatInputLayer(LayerBase):
  """
  Base layer which concatenates all incoming source layers in the feature dimension,
  and provides that as `self.input_data`, which is of type :class:`Data`.
  This is the most common thing what many layers do with the input sources.
  If there is only a single source, will not do anything.
  This layer also optionally can do dropout on the input.
  """

  def __init__(self, dropout=0, dropout_noise_shape=None, mask=None, **kwargs):
    """
    :param float dropout: 0.0 means to apply no dropout. dropout will only be applied during training
    :param dict[str|tuple,int|None] dropout_noise_shape: see :func:`TFUtil.get_bc_shape`
    :param str|None mask: "dropout" or "unity" or None. this is obsolete and only here for historical reasons
    """
    super(_ConcatInputLayer, self).__init__(**kwargs)
    assert mask in ['dropout', 'unity', None], "invalid mask: %r" % mask
    if mask == "unity":
      assert not dropout
    elif mask == "dropout":
      assert dropout > 0
    self.dropout = dropout
    self.input_data = None
    if self.sources:
      self.input_data = concat_sources_with_opt_dropout(
        self.sources, dropout=dropout, dropout_noise_shape=dropout_noise_shape)


class CopyLayer(_ConcatInputLayer):
  """
  This layer does nothing, it copies its input.
  If multiple sources are provided, they are concatenated in the feature-dim.
  """

  layer_class = "copy"

  def __init__(self, **kwargs):
    super(CopyLayer, self).__init__(**kwargs)
    self.output = self.input_data.copy(name="%s_output" % self.name)
    if len(self.sources) == 1:
      self.output_loss = self.sources[0].output_loss
      if not self.dropout:
        self.output_before_activation = self.sources[0].output_before_activation

  @classmethod
  def get_out_data_from_opts(cls, name, sources=(), out_type=None, n_out=NotSpecified, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param dict[str]|None out_type:
    :param int|None|NotSpecified n_out:
    :rtype: Data
    """
    if out_type or n_out is not NotSpecified:
      return super(CopyLayer, cls).get_out_data_from_opts(
        name=name, out_type=out_type, n_out=n_out, sources=sources, **kwargs)
    return get_concat_sources_data_template(sources, name="%s_output" % name)


class DropoutLayer(CopyLayer):
  """
  Just the same as :class:`CopyLayer`, because that one already supports dropout.
  """
  layer_class = "dropout"


class InternalLayer(LayerBase):
  """
  This is not supposed to be used by the user.
  It is used by some code to construct a wrapper layer or so.
  """


class WrappedInternalLayer(InternalLayer):
  """
  This is not supposed to be used by the user. Like :class:`InternalLayer`, only intended for internal usage.
  This layer is supposed to logically wrap another layer.
  """

  def __init__(self, base_layer, **kwargs):
    """
    :param LayerBase base_layer: the layer which we are wrapping
    """
    super(WrappedInternalLayer, self).__init__(**kwargs)
    self.base_layer = base_layer
    self.params.update(base_layer.params)  # maybe ReuseParams wants to access it or so

  def get_base_absolute_name_scope_prefix(self):
    """
    :rtype: str
    """
    return self.base_layer.get_base_absolute_name_scope_prefix()

  def get_absolute_name_scope_prefix(self):
    """
    :rtype: str
    """
    return self.base_layer.get_absolute_name_scope_prefix()


class ExtendWithBeamLayer(WrappedInternalLayer):
  """
  This is not supposed to be used by the user. Like :class:`InternalLayer`, only intended for internal usage.
  This layer is supposed to logically wrap another layer, and extend the output with a specific beam size.
  """

  def __init__(self, base_layer, beam_size, **kwargs):
    """
    :param LayerBase base_layer: the layer which we are wrapping
    :param int beam_size:
    """
    super(ExtendWithBeamLayer, self).__init__(base_layer=base_layer, **kwargs)
    self.output = base_layer.output.copy_extend_with_beam(beam_size)

  @classmethod
  def get_out_data_from_opts(cls, name, base_layer, beam_size, **kwargs):
    """
    :param str name:
    :param LayerBase base_layer:
    :param int beam_size:
    """
    return base_layer.output.copy_template(name="%s_output" % name).copy_extend_with_beam(beam_size)


class SelectSearchSourcesLayer(InternalLayer):
  """
  Selects the corresponding search beams from the source, given current search choices
  (determined by a layer).
  Like :class:`InternalLayer`, only for internal purpose at the moment.
  """

  def __init__(self, search_choices, **kwargs):
    """
    :param LayerBase search_choices:
    """
    if "output" not in kwargs:
      kwargs = kwargs.copy()
      kwargs["output"] = kwargs["sources"][0].output  # will be reset later
    from TFUtil import select_src_beams
    super(SelectSearchSourcesLayer, self).__init__(**kwargs)
    assert len(self.sources) == 1
    src = self.sources[0]
    self.search_choices_layer = search_choices
    search_choices = search_choices.get_search_choices()
    self.output = src.output.copy_as_batch_major()
    if search_choices:
      self.output = self.output.copy_extend_with_beam(search_choices.beam_size)
    src_search_choices = src.get_search_choices()
    if not search_choices or search_choices == src_search_choices or not src_search_choices:
      pass
    else:
      assert search_choices and search_choices != src_search_choices
      search_choices_seq = search_choices.get_all_src_choices()
      assert src_search_choices in search_choices_seq, "no common search base"
      search_choices_seq = search_choices_seq[:search_choices_seq.index(src_search_choices)]
      assert src_search_choices not in search_choices_seq

      def transform(v):
        """
        :param tuple|list|tf.Tensor|tf.TensorArray|T v:
        :rtype: T
        """
        if isinstance(v, (tuple, list)):
          from Util import make_seq_of_type
          return make_seq_of_type(type(v), [transform(v_) for v_ in v])
        assert isinstance(v, (tf.Tensor, tf.TensorArray))
        if isinstance(v, tf.Tensor) and v.get_shape().ndims == 0:
          return v  # leave scalars as-is
        for base_src_choices in reversed(search_choices_seq):
          assert isinstance(base_src_choices, SearchChoices)
          v = select_src_beams(v, src_beams=base_src_choices.src_beams)
        return v

      # It's possible that src.output.placeholder is not set, e.g. in a prev-layer where the
      # prev output is not needed, only the prev state. See _TemplateLayer.copy_as_prev_time_frame.
      if src.output.placeholder is not None:
        self.output.placeholder = transform(src.output.get_placeholder_as_batch_major())
      self.rec_vars_outputs = {k: transform(v) for (k, v) in src.rec_vars_outputs.items()}  # assumes batch-major

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    return super(SelectSearchSourcesLayer, self).get_dep_layers() + [self.search_choices_layer]

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param TFNetwork.TFNetwork network:
    :param get_layer:
    """
    super(SelectSearchSourcesLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["search_choices"] = get_layer(d["search_choices"])

  @classmethod
  def get_out_data_from_opts(cls, name, sources, search_choices, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param LayerBase search_choices:
    :rtype: Data
    """
    assert len(sources) == 1
    search_choices = search_choices.get_search_choices()
    data = sources[0].output.copy_as_batch_major()
    if data.beam_size:
      assert search_choices
      data = data.copy_extend_with_beam(search_choices.beam_size)
    elif search_choices:
      data = data.copy_extend_with_beam(search_choices.beam_size)
    return data


class ActivationLayer(CopyLayer):
  """
  This layer just applies an activation function.
  See :func:`TFUtil.get_activation_function` about supported functions.
  Also see :class:`EvalLayer` and :class:`CombineLayer` for similar layers.
  """

  layer_class = "activation"

  def __init__(self, activation, **kwargs):
    """
    :param str activation: e.g. "relu", "tanh", etc
    """
    super(ActivationLayer, self).__init__(**kwargs)
    x = self.input_data.placeholder
    if activation:
      from TFUtil import get_activation_function
      act_func = get_activation_function(activation)
      self.output_before_activation = OutputWithActivation(x, act_func=act_func)
    else:
      self.output_before_activation = OutputWithActivation(x)
    self.output.placeholder = self.output_before_activation.y


class BatchNormLayer(CopyLayer):
  """
  Implements batch-normalization (http://arxiv.org/abs/1502.03167) as a separate layer.
  """
  layer_class = "batch_norm"

  def __init__(self, **kwargs):
    """
    All kwargs which are present in our base class are passed to our base class.
    All remaining kwargs are used for self.batch_norm().
    """
    kwargs = kwargs.copy()
    from Util import getargspec
    batch_norm_kwargs = getargspec(self.batch_norm).args[1:]  # first is self, ignore
    batch_norm_opts = {key: kwargs.pop(key)
                       for key in batch_norm_kwargs
                       if key in kwargs}
    super(BatchNormLayer, self).__init__(batch_norm=batch_norm_opts or True, **kwargs)


class LayerNormLayer(_ConcatInputLayer):
  """
  Applies layer-normalization.
  """
  layer_class = "layer_norm"

  def __init__(self, epsilon=1e-6, **kwargs):
    super(LayerNormLayer, self).__init__(**kwargs)
    assert not self.input_data.sparse
    x = self.input_data.placeholder
    dim = self.input_data.dim
    axis = self.input_data.feature_dim_axis
    with self.var_creation_scope():
      scale = self.add_param(tf.get_variable("scale", [dim], initializer=tf.ones_initializer()))
      bias = self.add_param(tf.get_variable("bias", [dim], initializer=tf.zeros_initializer()))
    mean = tf.reduce_mean(x, axis=[axis], keep_dims=True, name="mean")
    variance = tf.reduce_mean(tf.square(x - mean), axis=[axis], keep_dims=True, name="variance")
    with tf.name_scope("normalized"):
      norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    self.output.placeholder = norm_x * scale + bias
    self.output.size_placeholder = self.input_data.size_placeholder.copy()

  @classmethod
  def get_out_data_from_opts(cls, sources, name, **kwargs):
    """
    :param list[LayerBase] sources:
    :param str name:
    :rtype: Data
    """
    return get_concat_sources_data_template(sources, name="%s_output" % name)


class SliceLayer(_ConcatInputLayer):
  """
  Slicing on the input, i.e. x[start:end:step] in some axis.
  See also :class:`SliceNdLayer`.
  """
  layer_class = "slice"

  def __init__(self, axis, slice_start=None, slice_end=None, slice_step=None, **kwargs):
    """
    :param int|str axis:
    :param str|None axis_kind: "T" for time, "B" for batch, "F" for feature
    :param int|None slice_start:
    :param int|None slice_end:
    :param int|None slice_step:
    """
    super(SliceLayer, self).__init__(**kwargs)
    axis = self.input_data.get_axis_from_description(axis)
    dim_slice = slice(slice_start, slice_end, slice_step)
    slices = [slice(None, None)] * axis + [dim_slice]
    axis_wo_batch = self.input_data.get_batch_axis_excluding_batch(axis)
    self.output.size_placeholder = self.input_data.size_placeholder.copy()
    if axis_wo_batch in self.output.size_placeholder:
      if slice_start:
        assert slice_start > 0
        self.output.size_placeholder[axis_wo_batch] = (
          tf.maximum(0, self.output.size_placeholder[axis_wo_batch] - slice_start))
      if slice_end is not None:
        if slice_end < 0:
          slice_end = tf.shape(self.input_data.placeholder)[axis] + slice_end
        self.output.size_placeholder[axis_wo_batch] = (
          tf.minimum(
            tf.shape(self.input_data.placeholder)[axis] - slice_end,
            self.output.size_placeholder[axis_wo_batch]))
      if slice_step:
        self.output.size_placeholder[axis_wo_batch] = (
          tf.ceil(tf.divide(self.output.size_placeholder[axis_wo_batch], slice_step)))
    self.output.placeholder = self.input_data.placeholder[slices]

  @classmethod
  def get_out_data_from_opts(
        cls, name, axis, sources=(),
        slice_start=None, slice_end=None, slice_step=None, **kwargs):
    """
    :param str name:
    :param str axis:
    :param list[LayerBase] sources:
    :param int|None slice_start:
    :param int|None slice_end:
    :param int|None slice_step:
    :rtype: Data
    """
    input_data = get_concat_sources_data_template(sources)
    axis = input_data.get_axis_from_description(axis)
    out_type = input_data.get_kwargs()
    out_type["name"] = "%s_output" % name
    axis_wo_batch = input_data.get_batch_axis_excluding_batch(axis)
    dim_slice = slice(slice_start, slice_end, slice_step)
    if axis_wo_batch is not None:
      out_type["shape"] = list(out_type["shape"])
      if out_type["shape"][axis_wo_batch] is not None:
        out_type["shape"][axis_wo_batch] = len(range(out_type["shape"][axis_wo_batch])[dim_slice])
    if not out_type["sparse"]:
      # Let Data() automatically infer "dim".
      out_type["dim"] = NotSpecified
    return Data(**out_type)


class SliceNdLayer(_ConcatInputLayer):
  """
  This takes out a slice-range from some axis,
  e.g. ``x[start:start + size]``.
  This layers allows a different start slice point for each batch,
  in contrast to :class:`SliceLayer`, and the start is variable.
  """
  layer_class = "slice_nd"

  def __init__(self, start, size, **kwargs):
    """
    :param LayerBase start:
    :param int size:
    """
    super(SliceNdLayer, self).__init__(**kwargs)
    from TFUtil import slice_nd, dimshuffle
    x = self.input_data.copy_as_batch_major()
    start = start.output.get_placeholder_as_batch_major()
    start = dimshuffle(start, [0, 'x'])  # (B, T, ...)
    axis = x.time_dim_axis
    assert axis == 1, "currently only time-axis==1 supported"
    slices = slice_nd(x.placeholder, tf.cast(start, tf.int32), size)  # (B,size, ...)

    self.output.size_placeholder = x.size_placeholder.copy()
    self.output.size_placeholder.pop(0, None)  # static time axis
    self.output.placeholder = slices

  @classmethod
  def get_out_data_from_opts(cls, name, sources=(), start=None, size=None, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param LayerBase|None start:
    :param int|None size:
    :rtype: Data
    """
    input_data = get_concat_sources_data_template(sources).copy_as_batch_major()
    in_shape = list(input_data.shape)
    shape = [size] + in_shape[1:]  # (B, size, ...) (w/o batch)
    out_type = input_data.get_kwargs()
    out_type["name"] = "%s_output" % name
    out_type["shape"] = shape
    out_type["batch_dim_axis"] = 0
    return Data(**out_type)

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param TFNetwork.TFNetwork network:
    :param get_layer:
    """
    super(SliceNdLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["start"] = get_layer(d["start"])


class LinearLayer(_ConcatInputLayer):
  """
  Linear/forward/fully-connected/1x1-conv layer.
  Does a linear transformation on the feature-dimension of the input
  with an optional bias term and an optional activation function.
  See also :class:`DotLayer`, :class:`ElemwiseProdLayer`, :class:`WeightedSumLayer`.
  """
  layer_class = "linear"

  def __init__(self, activation, with_bias=True, grad_filter=None, forward_weights_init="glorot_uniform",
               bias_init=0.0, use_transposed_weights=False, **kwargs):
    """
    :param str|None activation: e.g. "relu", or None
    :param bool with_bias:
    :param float|None grad_filter: if grad norm is higher than this threshold (before activation), the grad is removed
    :param str forward_weights_init: see :func:`TFUtil.get_initializer`
    :param str recurrent_weights_init: see :func:`TFUtil.get_initializer`
    :param str|float bias_init: see :func:`TFUtil.get_initializer`
    :param bool use_transposed_weights: If True, define the weight matrix with transposed dimensions (n_out, n_in).
    """
    super(LinearLayer, self).__init__(**kwargs)
    from TFUtil import get_initializer

    self.activation = activation
    self.with_bias = with_bias
    self.use_transposed_weights = use_transposed_weights

    input_data = self.input_data
    n_in = input_data.dim
    n_out = self.output.dim
    assert n_in and n_out, "%r and %r" % (input_data, self.output)

    with self.var_creation_scope():
      # Our Theano default: normal distribution, std_dev = sqrt(12. / (fan_in + fan_out))
      # glorot_normal = variance_scaling_initializer(scale=1.0, mode="fan_avg", distribution="normal")
      #  -> std_dev = sqrt(2. / (fan_in + fan_out)).
      #  Or use VarianceScaling(scale=6.0, mode="fan_avg", distribution="normal") to get the same as in Theano.
      fwd_weights_initializer = get_initializer(
        forward_weights_init, seed=self.network.random.randint(2 ** 31), eval_local_ns={"layer": self})
      if self.use_transposed_weights:
        weights_shape = (n_out, n_in)
      else:
        weights_shape = (n_in, n_out)

      weights = self.add_param(tf.get_variable(
        name="W", shape=weights_shape, dtype=tf.float32, initializer=fwd_weights_initializer))
      weights_ = weights

      if self.use_transposed_weights:
        weights = tf.transpose(weights)

      if self.with_bias:
        bias_initializer = get_initializer(
          bias_init, seed=self.network.random.randint(2 ** 31) if bias_init else 0, eval_local_ns={"layer": self})
        b = self.add_param(tf.get_variable(
          name="b", shape=(n_out,), dtype=tf.float32, initializer=bias_initializer))
      else:
        assert not bias_init
        b = None

    with tf.name_scope("linear"):
      from TFUtil import dot, to_int32_64, is_gpu_available, move_axis
      x = input_data.placeholder
      ndim = x.get_shape().ndims

      if self.input_data.sparse:
        # Maybe optionally we could also use tf.contrib.layers.safe_embedding_lookup_sparse().
        x = tf.nn.embedding_lookup(weights, to_int32_64(x))
        ndim += 1
      elif self.input_data.feature_dim_axis == self.input_data.batch_ndim - 1:
        x = dot(x, weights_, transpose_b=self.use_transposed_weights)
      elif self.input_data.is_batch_feature_major and is_gpu_available():  # CuDNN has a fast version for this
        # Use conv instead, it has optimized code for batch-feature major (only CuDNN).
        x_shape = None
        if self.input_data.batch_ndim > 3:
          x_shape = tf.shape(x)
          x_shape = [x_shape[i] for i in range(self.input_data.batch_ndim)]
          x = tf.reshape(x, [x_shape[0], n_in, tf.reduce_prod(x_shape[2:])])  # (B,n_in,x)
        x = tf.nn.conv1d(
          x,  # (B,n_in,x)
          filters=tf.expand_dims(weights, 0),  # (1,n_in,n_out)
          stride=1, padding='SAME', data_format="NCW")  # (B,n_out,x)
        if self.input_data.batch_ndim > 3:
          x = tf.reshape(x, x_shape[:1] + [n_out] + x_shape[2:])  # (B,n_out,...)
      else:
        print("%s: Warning: inefficient implementation for input %r." % (self, self.input_data), file=log.v2)
        x = move_axis(x, self.input_data.feature_dim_axis, -1)
        x = dot(x, weights_, transpose_b=self.use_transposed_weights)
        x = move_axis(x, -1, self.input_data.feature_dim_axis)
      assert x.get_shape().ndims == ndim

      if self.with_bias:
        if self.input_data.sparse or self.input_data.feature_dim_axis == self.input_data.batch_ndim - 1:
          x = tf.add(x, b, name="add_bias")
        else:
          b_bc_shape = (
            ([1] * self.input_data.feature_dim_axis) +
            [n_out] +
            ([1] * (self.input_data.batch_ndim - self.input_data.feature_dim_axis - 1)))
          assert len(b_bc_shape) == self.input_data.batch_ndim == x.get_shape().ndims
          b_bc = tf.reshape(b, b_bc_shape)
          x = tf.add(x, b_bc, name="add_bias")
        assert x.get_shape().ndims == ndim

    if grad_filter:
      x = TFUtil.filter_grad(
        x,
        threshold=grad_filter,
        axis=[i for i in range(input_data.batch_ndim) if i != input_data.batch_dim_axis])

    if self.activation:
      from TFUtil import get_activation_function
      act_func = get_activation_function(self.activation)
      self.output_before_activation = OutputWithActivation(x, act_func=act_func)
    else:
      self.output_before_activation = OutputWithActivation(x)
    x = self.output_before_activation.y

    assert self.output.batch_dim_axis == self.input_data.batch_dim_axis
    assert self.output.time_dim_axis == self.input_data.time_dim_axis
    self.output.placeholder = x


class SoftmaxLayer(LinearLayer):
  """
  Just a LinearLayer with activation="softmax" by default.
  """
  layer_class = "softmax"

  def __init__(self, activation="softmax", **kwargs):
    super(SoftmaxLayer, self).__init__(activation=activation, **kwargs)


class LengthLayer(LayerBase):
  """
  Returns the length of sources as (B,), via input size_placeholder.
  """
  layer_class = "length"

  def __init__(self, add_time_axis=False, dtype="int32", **kwargs):
    super(LengthLayer, self).__init__(**kwargs)
    assert len(self.sources) == 1, "%s: expects one source" % self
    out = tf.cast(self.sources[0].output.get_sequence_lengths(), dtype)
    if add_time_axis:
      out = tf.expand_dims(out, axis=self.output.time_dim_axis)
    self.output.placeholder = out

  @classmethod
  def get_out_data_from_opts(cls, name, sources, add_time_axis=False, dtype="int32", **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param bool add_time_axis:
    :param str dtype:
    :rtype: Data
    """
    if add_time_axis:
      shape = (1,)
      time_dim_axis = 1
    else:
      shape = ()
      time_dim_axis = None
    return Data(
      name="%s_length" % name,
      shape=shape,
      batch_dim_axis=0,
      time_dim_axis=time_dim_axis,
      dtype=dtype,
      sparse=False)


class SoftmaxOverSpatialLayer(_ConcatInputLayer):
  """
  This applies a softmax over spatial axis/axes (currently only time axis supported).
  E.g. when the input is of shape (B,T,dim), the output will be (B,T,dim).
  It automatically masks the frames outside the seq defined by the seq-len.
  In contrast to :class:`SoftmaxLayer`, this will not do a linear transformation.
  """
  layer_class = "softmax_over_spatial"

  def __init__(self, axis=None, energy_factor=None, window_start=None, window_size=None, use_time_mask=None, **kwargs):
    """
    :param str|None axis: which axis to do the softmax over
    :param float|None energy_factor: the energy will be scaled by this factor.
      This is like a temperature for the softmax.
      In Attention-is-all-you-need, this is set to 1/sqrt(base_ctx.dim).
    :param LayerBase|None window_start: Tensor of shape (B,1) indicating the window start
    :param int|None window_size:
    :param bool use_time_mask: if True, assumes dyn seq len, and use it for masking.
      By default, if dyn seq len exists, it uses it.
    """
    from TFUtil import where_bc
    super(SoftmaxOverSpatialLayer, self).__init__(**kwargs)
    energy_data = self.input_data
    assert energy_data.dtype.startswith("float")
    axis = self._get_axis_to_reduce(input_data=energy_data, axis=axis, exception_prefix=self)
    # tf.nn.softmax operates on the last axis.
    energy_data = energy_data.copy_move_axis(axis, -1)
    axis = energy_data.batch_ndim - 1
    energy = energy_data.placeholder
    energy_shape = tf.shape(energy, name="energy_shape")
    energy_shape = [energy_shape[i] for i in range(energy_data.batch_ndim)]
    assert energy_data.have_time_axis()
    # if the time-axis is static, we can skip the masking
    if use_time_mask is None:
      use_time_mask = energy_data.is_axis_dynamic(axis)
    if use_time_mask:
      assert energy_data.is_axis_dynamic(axis), "%s: use_time_mask True, dyn time axis expected" % self
      energy_mask = energy_data.get_sequence_mask_broadcast(axis=axis)
      if window_start is not None:
        assert window_size is not None, "set window_size explicitly"
        from TFUtil import nd_indices, expand_dims_unbroadcast
        # handle edge cases correctly:
        # 1. if the energy time-dim is less than `window_size`, we adjust the window size.
        # 2. for each seq, we adjust the window so that no elements after the seq-len are indexed.
        window_len = tf.minimum(window_size, energy_shape[axis])  # case 1.
        window_start = tf.squeeze(window_start.output.placeholder, axis=window_start.output.feature_dim_axis)  # (B,)
        window_start = tf.to_int32(window_start)
        window_start = tf.where(
          tf.greater(window_start + window_len, energy_shape[axis]),
          tf.ones_like(window_start) * (energy_shape[axis] - window_len),
          window_start)  # case 2.
        n_batch = energy_shape[energy_data.batch_dim_axis]
        indices = expand_dims_unbroadcast(tf.range(window_len), energy_data.batch_dim_axis, n_batch)
        # time-major: (W,B) + (1,B), batch-major: (B, W) + (1,B)
        indices += tf.expand_dims(tf.to_int32(window_start), axis=axis)
        idxs = nd_indices(indices)
        mask_shape = energy_shape[:2]  # (T, B)
        mask_shape[axis] = window_len
        energy_mask_window = tf.scatter_nd(idxs, tf.ones(shape=mask_shape), energy_shape[:2])
        energy_mask_window = tf.cast(energy_mask_window, tf.bool)
        energy_mask = tf.logical_and(energy_mask, energy_mask_window)
      energy = where_bc(energy_mask, energy, float("-inf"), name="energy_masked")
    if energy_factor:
      energy = tf.multiply(energy, energy_factor, name="energy_scaled")
    weights = tf.nn.softmax(energy)  # (...,T)
    self.output.placeholder = weights

  @classmethod
  def _get_axis_to_reduce(cls, input_data, axis, exception_prefix):
    """
    :param str|None axis:
    :param str|object exception_prefix:
    :rtype: int
    """
    if axis is None:
      assert input_data.have_time_axis(), "%s: requires that the input has a time dim" % exception_prefix
      axis = input_data.time_dim_axis
    else:
      axis = input_data.get_axis_from_description(axis, allow_int=False)
    return axis

  @classmethod
  def get_out_data_from_opts(cls, name, sources, axis=None, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param str|None axis:
    :rtype: Data
    """
    out = get_concat_sources_data_template(sources, name="%s_output" % name)
    axis = cls._get_axis_to_reduce(out, axis=axis, exception_prefix="%s %r" % (cls.__name__, name))
    return out.copy_move_axis(axis, -1)

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param TFNetwork.TFNetwork network:
    :param get_layer:
    """
    super(SoftmaxOverSpatialLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    if d.get("window_start", None):
      d["window_start"] = get_layer(d["window_start"])


class SeqLenMaskLayer(_ConcatInputLayer):
  """
  Masks some values away given the seq_len_source with mask_value.
  """
  layer_class = "seq_len_mask"

  def __init__(self, mask_value, axis="T", seq_len_source=None, **kwargs):
    """
    :param LayerBase|None seq_len_source: if not given, uses source
    :param str|int axis:
    :param float mask_value:
    """
    super(SeqLenMaskLayer, self).__init__(**kwargs)
    x = self.input_data.copy_as_batch_major()  # e.g. (B,T',T)
    axis = x.get_axis_from_description(axis)
    if not seq_len_source:
      seq_len_source = self.input_data
    else:
      seq_len_source = seq_len_source.output
    energy_mask = seq_len_source.copy_as_batch_major().get_sequence_mask()  # e.g. (B,T)
    from TFUtil import expand_multiple_dims
    energy_mask = expand_multiple_dims(
      energy_mask, [i for i in range(x.batch_ndim) if i not in [x.batch_dim_axis, axis]])  # e.g. (B,1,T) with axis=-1
    energy_mask = tf.logical_and(energy_mask, tf.ones_like(x.placeholder, dtype=energy_mask.dtype))
    x_ = tf.where(energy_mask, x.placeholder, mask_value * tf.ones_like(x.placeholder), "energy_masked")
    self.output.placeholder = x_
    self.output.size_placeholder = x.size_placeholder.copy()
    if mask_value in [float("-inf"), float("inf")]:
      self.allow_inf_in_output = True

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param TFNetwork.TFNetwork network:
    :param get_layer:
    """
    super(SeqLenMaskLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    if "seq_len_source" in d:
      d["seq_len_source"] = get_layer(d["seq_len_source"])

  @classmethod
  def get_out_data_from_opts(cls, name, sources, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    return get_concat_sources_data_template(sources, name="%s_output" % name).copy_as_batch_major()


class RangeInAxisLayer(LayerBase):
  """
  Assume that the input is e.g. (B,T,D), and you specify axis="T", you will get (B=1,T,D=1),
  where the specified axis is filled with ``tf.range``.
  (Currently always keep_dims.)
  """
  layer_class = "range_in_axis"
  recurrent = True  # if axis=="T", the time-dim order matters

  def __init__(self, axis, dtype="int32", **kwargs):
    """
    :param str axis:
    :param str dtype:
    """
    super(RangeInAxisLayer, self).__init__(**kwargs)
    axis = self.output.get_axis_from_description(axis)
    source = self.sources[0].output
    source_shape = tf.shape(source.placeholder)
    dim = source_shape[axis]
    out = tf.range(0, dim)
    out_shape = [dim if (i == axis) else 1 for i in range(self.output.batch_ndim)]
    out = tf.reshape(out, out_shape)  # add missing axes (keep_dims)
    out = tf.cast(out, dtype)
    self.output.placeholder = out
    axis_wo_b = source.get_batch_axis_excluding_batch(axis)
    self.output.size_placeholder = {i: size for (i, size) in source.size_placeholder.items() if i == axis_wo_b}

  @classmethod
  def get_out_data_from_opts(cls, name, sources, axis, dtype="int32", **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param str axis:
    :param str dtype:
    """
    assert len(sources) == 1, "%s layer %r requires single source" % (cls, name)
    out = sources[0].output.copy_template(name="%s_output" % name)
    axis = out.get_axis_from_description(axis)
    axis_wo_b = out.get_batch_axis_excluding_batch(axis)
    out.shape = tuple([d if (i == axis_wo_b) else 1 for (i, d) in enumerate(out.shape)])
    out.dtype = dtype
    return out


class BatchSoftmaxLayer(_ConcatInputLayer):
  """
  Softmax over spacial and feature axis
  """
  layer_class = "batch_softmax"

  def __init__(self, **kwargs):
    from TFUtil import sequence_mask
    super(BatchSoftmaxLayer, self).__init__(**kwargs)

    data = self.input_data.get_placeholder_as_batch_major()
    data_shape = tf.shape(data)
    data_flat = tf.reshape(data, [data_shape[0] * data_shape[1], -1])  # (B * T, D)
    # first mask all values that are not used with -inf, the mask is flat, otherwise we would have to tile it
    # (increasing the size by the number of features)
    mask = tf.reshape(sequence_mask(self.input_data.get_sequence_lengths()), [-1])  # (B * T,)
    data_flat = tf.where(mask, data_flat, tf.fill(tf.shape(data_flat), float("-inf")))
    data = tf.reshape(data_flat, [data_shape[0], -1])  # (B, T*D)
    data = tf.nn.softmax(data)
    data = tf.reshape(data, data_shape)  # (B, T, D)
    self.output.placeholder = data

  @classmethod
  def get_out_data_from_opts(cls, name, sources, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    return get_concat_sources_data_template(sources, name="%s_output" % name).copy_as_batch_major()


class ConstantLayer(LayerBase):
  """
  Output is a constant value.
  """
  layer_class = "constant"

  def __init__(self, sources, value=0, dtype=None, **kwargs):
    assert not sources, "constant layer cannot have sources"
    super(ConstantLayer, self).__init__(**kwargs)
    # Add batch-dim to the constant.
    self.output.placeholder = tf.expand_dims(tf.constant(value, dtype=dtype), axis=0)

  @classmethod
  def get_out_data_from_opts(cls, name, dtype="float32", **kwargs):
    """
    :param str name:
    :param str dtype:
    :rtype: Data
    """
    return Data(
      name="%s_const" % name, shape=(), batch_dim_axis=0, time_dim_axis=None, dtype=dtype)


class GatingLayer(_ConcatInputLayer):
  """
  Splits the output into two equal parts, applies the gate_activation (sigmoid by default)
  on the one part, some other activation (e.g. tanh) on the other part and then
  element-wise multiplies them.
  Thus, the output dimension is input-dimension / 2.
  """
  layer_class = "gating"

  def __init__(self, activation, gate_activation="sigmoid", **kwargs):
    super(GatingLayer, self).__init__(**kwargs)
    from TFUtil import get_activation_function
    act_func = get_activation_function(activation)
    gate_act_func = get_activation_function(gate_activation)
    a, b = tf.split(self.input_data.placeholder, 2, axis=self.input_data.batch_ndim - 1)
    self.output.placeholder = act_func(a) * gate_act_func(b)
    self.output.size_placeholder = self.input_data.size_placeholder.copy()

  @classmethod
  def get_out_data_from_opts(cls, name, sources, n_out=NotSpecified, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param int|None|NotSpecified n_out:
    :rtype: Data
    """
    input_data = get_concat_sources_data_template(sources)
    assert not input_data.sparse
    assert input_data.dim % 2 == 0
    dim = input_data.dim // 2
    if n_out is not NotSpecified:
      assert n_out == dim
    return Data(
      name="%s_output" % name,
      dtype=input_data.dtype,
      shape=input_data.shape[:-1] + (dim,),
      sparse=False,
      batch_dim_axis=input_data.batch_dim_axis,
      time_dim_axis=input_data.time_dim_axis,
      beam_size=input_data.beam_size)


class WindowLayer(_ConcatInputLayer):
  """
  Adds a window dimension.
  By default, uses the time axis and goes over it with a sliding window.
  The new axis for the window is created right after the time axis.
  Will always return as batch major mode.
  E.g. if the input is (batch, time, dim), the output is (batch, time, window_size, dim).
  If you want to merge the (window_size, dim) together to (window_size * dim,),
  you can use the MergeDimsLayer, e.g. {"class": "merge_dims", "axes": "except_time"}.

  This is not to take out a window from the time-dimension.
  See :class:`SliceLayer` or :class:`SliceNdLayer`.
  """
  layer_class = "window"
  recurrent = True  # we must not allow any shuffling in the time-dim or so

  def __init__(self, window_size, window_left=None, window_right=None, axis="T", padding="same", **kwargs):
    """
    :param int window_size:
    :param int|None window_left:
    :param int|None window_right:
    :param str|int axis: see Data.get_axis_from_description()
    :param str padding: "same" or "valid"
    :param kwargs:
    """
    super(WindowLayer, self).__init__(**kwargs)
    data = self.input_data.copy_as_batch_major()
    if axis == "T" and data.time_dim_axis is None:
      # Assume inside RecLayer.
      assert self._rec_previous_layer, "%s: expected to be used inside a RecLayer" % self
      assert padding == "same"
      prev_state = self._rec_previous_layer.rec_vars_outputs["state"]  # (batch,window,...)
      next_state = tf.concat(
        [prev_state[:, 1:], tf.expand_dims(data.placeholder, axis=1)], axis=1)  # (batch,window,...)
      self.rec_vars_outputs["state"] = next_state
      self.output.placeholder = next_state
    else:
      axis = data.get_axis_from_description(axis)
      from TFUtil import windowed_nd
      self.output.placeholder = windowed_nd(
        data.placeholder,
        window_size=window_size, window_left=window_left, window_right=window_right,
        padding=padding, time_axis=axis, new_window_axis=axis + 1)
    self.output.placeholder.set_shape(tf.TensorShape(self.output.batch_shape))
    self.output.size_placeholder = self.input_data.size_placeholder.copy()
    axis_wo_b = self.output.get_batch_axis_excluding_batch(axis)
    if axis_wo_b in self.output.size_placeholder:
      self.output.size_placeholder[axis_wo_b] = ConvLayer.calc_out_dim(
        in_dim=self.output.size_placeholder[axis_wo_b],
        filter_size=window_size, stride=1, dilation_rate=1, padding=padding)

  @classmethod
  def get_out_data_from_opts(cls, name, window_size, axis="T", sources=(), **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param int window_size:
    :param str axis:
    :rtype: Data
    """
    data = get_concat_sources_data_template(sources)
    data = data.copy_template(name="%s_output" % name)
    data = data.copy_as_batch_major()
    if axis == "T" and data.time_dim_axis is None:
      # Assume inside RecLayer.
      axis = 0
    else:
      axis = data.get_axis_from_description(axis)
    data = data.copy_add_spatial_dim(spatial_dim_axis=axis + 1, dim=window_size)  # add new axis right after
    return data

  # noinspection PyMethodOverriding
  @classmethod
  def get_rec_initial_extra_outputs(cls, batch_dim, rec_layer, window_size, axis="T", sources=(), **kwargs):
    """
    :param tf.Tensor batch_dim:
    :param TFNetworkRecLayer.RecLayer|LayerBase rec_layer:
    :param int window_size:
    :param str axis:
    :param list[LayerBase] sources:
    :rtype: dict[str,tf.Tensor]
    """
    data = get_concat_sources_data_template(sources)
    data = data.copy_as_batch_major()
    if axis == "T" and data.time_dim_axis is None:
      # Assume inside RecLayer.
      shape = list(data.batch_shape)
      shape[0] = batch_dim
      shape.insert(1, window_size)
      return {"state": tf.zeros(shape, dtype=data.dtype)}
    return {}


class CumsumLayer(_ConcatInputLayer):
  """
  Basically wraps tf.cumsum. Also supports that in the RecLayer.
  """
  layer_class = "cumsum"
  recurrent = True  # order matters

  def __init__(self, axis="T", additional_left_summand_per_element=None, reverse=False, **kwargs):
    """
    :param str axis: see :func:`Data.get_axis_from_description`
    :param str|int|float|None additional_left_summand_per_element: the order matters for tf.string
    :param bool reverse:
    """
    super(CumsumLayer, self).__init__(**kwargs)
    data = self.input_data
    x = data.placeholder
    if additional_left_summand_per_element is not None:
      x = additional_left_summand_per_element + x
    if axis == "T" and data.time_dim_axis is None:
      # Assume inside RecLayer.
      assert self._rec_previous_layer, "%s: expected to be used inside a RecLayer" % self
      assert not reverse
      prev_state = self._rec_previous_layer.rec_vars_outputs["state"]
      next_state = prev_state + x
      self.rec_vars_outputs["state"] = next_state
      self.output.placeholder = next_state
    else:
      axis = data.get_axis_from_description(axis)
      self.output.placeholder = tf.cumsum(x, axis=axis, reverse=reverse)
    self.output.placeholder.set_shape(data.placeholder.get_shape())
    self.output.placeholder.set_shape(tf.TensorShape(self.output.batch_shape))
    self.output.size_placeholder = self.input_data.size_placeholder.copy()

  @classmethod
  def get_out_data_from_opts(cls, name, sources, axis="T", **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param str axis:
    :rtype: Data
    """
    # Just same format.
    return get_concat_sources_data_template(sources, name="%s_output" % name)

  @classmethod
  def get_rec_initial_extra_outputs(cls, batch_dim, rec_layer, axis="T", sources=(), **kwargs):
    """
    :param tf.Tensor batch_dim:
    :param TFNetworkRecLayer.RecLayer|LayerBase rec_layer:
    :param str axis:
    :param list[LayerBase] sources:
    :rtype: dict[str,tf.Tensor]
    """
    data = get_concat_sources_data_template(sources)
    if axis == "T" and data.time_dim_axis is None:
      # Assume inside RecLayer.
      assert all(data.shape)
      return {"state": tf.zeros(data.get_batch_shape(batch_dim=batch_dim), dtype=data.dtype)}
    return {}


class PadLayer(_ConcatInputLayer):
  """
  Adds (e.g. zero) padding in some axis or axes.
  """
  layer_class = "pad"

  def __init__(self, axes, padding, value=None, mode="constant", **kwargs):
    """
    :param str|list[str] axes: e.g. "F" etc. see :func:`Dataset.get_axes_from_description`.
    :param list[(int,int)]|(int,int)|int padding: how much to pad left/right in each axis
    :param int|float value: what constant value to pad, with mode=="constant"
    :param str mode: "constant", "reflect" or "symmetric"
    """
    super(PadLayer, self).__init__(**kwargs)
    axes = self.input_data.get_axes_from_description(axes)
    padding = self._transform_padding(padding=padding, axes=axes)
    paddings = [(0, 0)] * len(range(self.input_data.batch_ndim))
    for i, a in enumerate(axes):
      paddings[a] = padding[i]
    mode = mode.upper()
    if mode == "CONSTANT":
      assert value is None or value == 0, "not yet implemented otherwise..."
    else:
      assert value is None
    self.output.placeholder = tf.pad(self.input_data.placeholder, paddings=paddings, mode=mode)
    self.output.size_placeholder = self.input_data.size_placeholder.copy()
    for a in axes:
      p = sum(paddings[a])
      a = self.input_data.get_batch_axis_excluding_batch(a)
      if a is None:
        continue
      if a not in self.output.size_placeholder:
        continue
      self.output.size_placeholder[a] += p

  @classmethod
  def _transform_padding(cls, padding, axes):
    """
    :param list[(int,int)]|(int,int)|int padding:
    :param list[int] axes:
    :rtype: list[(int,int)]
    """
    if isinstance(padding, (list, tuple)):
      if isinstance(padding[0], (list, tuple)):
        assert len(padding[0]) == 2
        assert len(padding) == len(axes)
      else:
        assert len(padding) == 2
        padding = [tuple(padding)] * len(axes)
    else:
      padding = [(padding, padding)] * len(axes)
    return padding

  @classmethod
  def get_out_data_from_opts(cls, name, axes, padding, sources=(), **kwargs):
    """
    :param str name:
    :param str|list[str] axes:
    :param list[(int,int)]|(int,int)|int padding:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    data = get_concat_sources_data_template(sources)
    data.name = "%s_output" % name
    axes = data.get_axes_from_description(axes)
    padding = cls._transform_padding(padding=padding, axes=axes)
    for i, a in enumerate(axes):
      a = data.get_batch_axis_excluding_batch(a)
      if a is None:
        continue
      if data.shape[a] is None:
        continue
      data.shape = data.shape[:a] + (data.shape[a] + sum(padding[i]),) + data.shape[a + 1:]
    return data


class MergeDimsLayer(_ConcatInputLayer):
  """
  Merges a list of axes into a single one. (Flatten the dims.)
  E.g. input is (batch, width, height, dim) and axes=(1,2), then we get (batch, width*height, dim).
  Or input is (batch, time, height, dim) and axes="except_time", then we get (batch, time, height*dim).
  See also :class:`CombineDimsLayer`.
  When batch and time got merged, :class:`SplitBatchTimeLayer` can undo this.
  """
  layer_class = "merge_dims"

  def __init__(self, axes, n_out=None, **kwargs):
    """
    :param str|list[str]|list[int] axes: see Data.get_axes_from_description(), e.g. "except_time"
    :param int|None n_out:
    """
    super(MergeDimsLayer, self).__init__(**kwargs)
    axes = self.input_data.get_axes_from_description(axes)
    axes = sorted(axes)
    merge_target_axis = self._get_target_axis(input_data=self.input_data, merge_axes=axes)
    x = self.input_data.placeholder
    if len(axes) > 1:
      axes = sorted(axes)
      # Transpose so that all axes are behind each other.
      perm = [i for i in range(self.input_data.batch_ndim) if i not in axes]
      # If batch axis included, move to front.
      # This is such that we can deterministically undo this later, e.g. in SplitBatchTimeLayer.
      if self.input_data.batch_dim_axis in axes:
        axes.remove(self.input_data.batch_dim_axis)
        axes.insert(0, self.input_data.batch_dim_axis)
      for i, a in enumerate(axes):
        perm.insert(merge_target_axis + i, a)
      x = tf.transpose(x, perm)
      # Now merge all dims with a reshape.
      shape = tf.shape(x)
      i0 = merge_target_axis
      i1 = i0 + len(axes)
      x = tf.reshape(
        x,
        shape=tf.concat([
          shape[:i0],
          tf.reduce_prod(shape[i0:i1], keep_dims=True),
          shape[i1:]], axis=0))
    if n_out is not None and not self.output.sparse:
      from TFUtil import check_input_dim
      x = check_input_dim(x, axis=-1, dim=n_out)
    self.output.placeholder = x
    self.output.size_placeholder = self._get_output_sizes(merge_axes=axes)

  @classmethod
  def _get_target_axis(cls, input_data, merge_axes):
    """
    :param Data input_data:
    :param list[int] merge_axes:
    :rtype: int
    """
    if input_data.feature_dim_axis in merge_axes:
      # We want it to become the new feature dim axis.
      # but we have to count only the dims prior to the feature-dim
      new_feature_dim_axis = input_data.feature_dim_axis
      new_feature_dim_axis -= sum([axis < input_data.feature_dim_axis for axis in merge_axes])
      return new_feature_dim_axis
    else:
      return min(merge_axes)

  @classmethod
  def _old_axis_to_new_axis(cls, input_data, merge_axes, old_axis):
    """
    :param Data input_data:
    :param list[int] merge_axes:
    :param int|None old_axis:
    :rtype: int|None
    """
    if old_axis is None:
      return old_axis
    target_axis = cls._get_target_axis(input_data=input_data, merge_axes=merge_axes)
    if old_axis in merge_axes:
      return target_axis
    new_axis = old_axis
    for i in range(input_data.batch_ndim):
      if i in merge_axes:
        new_axis -= 1
      if i == target_axis:
        new_axis += 1
      if i >= new_axis:
        break
    assert new_axis != target_axis
    return new_axis

  def _get_output_sizes(self, merge_axes):
    """
    :param list[int] merge_axes:
    :rtype: dict[int,tf.Tensor]
    """
    d = {}
    for i, v in sorted(self.input_data.size_placeholder.items()):
      axis = self.input_data.get_batch_axis(i)
      axis = self._old_axis_to_new_axis(input_data=self.input_data, merge_axes=merge_axes, old_axis=axis)
      if axis == self.output.batch_dim_axis:
        continue
      j = self.output.get_batch_axis_excluding_batch(axis)
      if j in d:
        d[j] *= v
      else:
        d[j] = v
    return d

  @classmethod
  def get_out_data_from_opts(cls, name, axes, sources=(), n_out=NotSpecified, out_type=None, **kwargs):
    """
    :param str name:
    :param str|list[str] axes:
    :param list[LayerBase] sources:
    :param int|None|NotSpecified n_out:
    :param None|dict[str] out_type:
    :rtype: Data
    """
    assert not out_type, "currently ignored"
    input_data = get_concat_sources_data_template(sources)
    data = input_data.copy(name="%s_output" % name)
    axes = input_data.get_axes_from_description(axes)
    if len(axes) <= 1:
      return data
    axes = sorted(axes)
    import numpy
    res_dim = None
    if all([data.batch_shape[i] is not None for i in axes]):
      res_dim = numpy.prod([data.batch_shape[i] for i in axes])
    if not data.sparse and data.feature_dim_axis in axes:  # will also merge the feature dim
      if res_dim is not None and n_out is not NotSpecified:
        assert res_dim == n_out
      elif res_dim is not None and n_out is NotSpecified:
        pass
      elif res_dim is None and n_out is not NotSpecified:
        res_dim = n_out
      data.dim = res_dim
    merge_target_axis = cls._get_target_axis(input_data=data, merge_axes=axes)
    if data.feature_dim_axis_or_unspecified is NotSpecified:
      new_feature_dim_axis = NotSpecified
    elif data.feature_dim_axis in axes and merge_target_axis != data.feature_dim_axis:
      if merge_target_axis == data.batch_dim_axis:
        new_feature_dim_axis = None
      else:
        new_feature_dim_axis = merge_target_axis
    else:
      new_feature_dim_axis = cls._old_axis_to_new_axis(
        input_data=input_data, merge_axes=axes, old_axis=input_data.feature_dim_axis)
    data.batch_dim_axis = cls._old_axis_to_new_axis(
      input_data=input_data, merge_axes=axes, old_axis=input_data.batch_dim_axis)
    new_shape = [d for (i, d) in enumerate(data.batch_shape) if i not in axes]
    new_shape.insert(merge_target_axis, res_dim)
    new_shape.pop(data.batch_dim_axis)
    data.shape = tuple(new_shape)
    data.time_dim_axis = cls._old_axis_to_new_axis(
      input_data=input_data, merge_axes=axes, old_axis=input_data.time_dim_axis)
    if data.time_dim_axis == data.batch_dim_axis:  # special case: batch and time got merged
      # Fallback to some sensible default.
      # Note: Not sure if this is good. Maybe we change that... You can always use ReinterpretDataLayer to be explicit.
      data.time_dim_axis = data.get_spatial_batch_axes()[0] if data.get_spatial_batch_axes() else None
    data.feature_dim_axis = new_feature_dim_axis
    return data


class SplitDimsLayer(_ConcatInputLayer):
  """
  Splits one axis into multiple axes.
  E.g. if you know that your feature-dim is composed by a window,
  i.e. the input is (batch, time, window * feature),
  you can set axis="F", dims=(window, -1),
  and you will get the output (batch, time, window, feature).
  Also see :class:`SplitBatchTimeLayer`.
  """
  layer_class = "split_dims"

  def __init__(self, axis, dims, **kwargs):
    """
    :param str axis: e.g. "F"
    :param tuple[int] dims: what the axis should be split into. e.g. (window, -1)
    """
    super(SplitDimsLayer, self).__init__(**kwargs)
    data = self.input_data
    if isinstance(axis, int):
      data = data.copy_as_batch_major()
    old_shape = tf.shape(data.placeholder)
    old_shape = [old_shape[i] for i in range(data.batch_ndim)]
    axis = data.get_axis_from_description(axis)
    new_shape = old_shape[:axis] + list(dims) + old_shape[axis + 1:]
    self.output.placeholder = tf.reshape(data.placeholder, shape=new_shape)
    self.output.size_placeholder = {
      (i if (data.get_batch_axis(i) < axis) else i + len(dims) - 1): v
      for (i, v) in data.size_placeholder.items()}
    assert axis != data.time_dim_axis, "size not yet implemented correctly..."

  @classmethod
  def _resolve_dims(cls, old_dim, new_dims):
    """
    :param int old_dim:
    :param tuple[int] new_dims:
    :return: new_dims with -1 resolved
    :rtype: tuple[int]
    """
    import numpy
    if all([d > 0 for d in new_dims]):
      new_dims_resolved = new_dims
    else:
      assert all([d != 0 or d == -1 or d > 0 for d in new_dims])
      assert len([d for d in new_dims if d == -1]) == 1
      new_pos_dims = [d for d in new_dims if d > 0]
      n = numpy.prod(new_pos_dims)
      assert old_dim % n == 0
      rem = old_dim // n
      new_dims_resolved = tuple([(d if (d > 0) else rem) for d in new_dims])
    assert numpy.prod(new_dims_resolved) == old_dim
    return new_dims_resolved

  @classmethod
  def get_out_data_from_opts(cls, name, axis, dims, sources=(), **kwargs):
    """
    :param str name:
    :param str|int axis:
    :param tuple[int] dims:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    data = get_concat_sources_data_template(sources)
    data.name = "%s_output" % name
    if isinstance(axis, int):
      data = data.copy_as_batch_major()
    axis = data.get_axis_from_description(axis)
    if data.batch_shape[axis] is not None:
      resolved_shape_dims = cls._resolve_dims(old_dim=data.batch_shape[axis], new_dims=dims)
    else:
      resolved_shape_dims = tuple([(d if (d >= 0) else None) for d in dims])
    if axis != data.batch_dim_axis:
      axis_wb = data.get_batch_axis_excluding_batch(axis)
      data.shape = data.shape[:axis_wb] + resolved_shape_dims + data.shape[axis_wb + 1:]
      if data.batch_dim_axis is not None and axis < data.batch_dim_axis:
        data.batch_dim_axis += len(dims) - 1
    else:  # axis == data.batch_dim_axis
      new_batch_shape = data.batch_shape[:axis] + resolved_shape_dims + data.batch_shape[axis + 1:]
      assert any([d == -1 for d in dims])
      new_batch_axis = data.batch_dim_axis + [d == -1 for d in dims].index(True)
      data.batch_dim_axis = new_batch_axis
      data.shape = new_batch_shape[:new_batch_axis] + new_batch_shape[new_batch_axis + 1:]
    assert axis != data.time_dim_axis, "size not yet implemented correctly..."
    if data.time_dim_axis is not None and axis < data.time_dim_axis:
      data.time_dim_axis += len(dims) - 1
    if data.feature_dim_axis is None and not data.sparse and any([d > 0 for d in dims]):
      # We want to have the last index where dims[index] > 0.
      i = len(dims) - list(reversed([d > 0 for d in dims])).index(True) - 1
      new_feature_dim_axis = axis + i
      if new_feature_dim_axis == data.batch_ndim - 1:
        data.feature_dim_axis = NotSpecified
      else:
        data.feature_dim_axis = new_feature_dim_axis
    if data.feature_dim_axis is not None:
      data.dim = data.batch_shape[data.feature_dim_axis]
    return data


class SplitBatchTimeLayer(_ConcatInputLayer):
  """
  A very specific layer which expects to get input of shape (batch * time, ...)
  and converts it into (batch, time, ...), where it recovers the seq-lens from some other layer.
  See :class:`SplitDimsLayer` for a more generic layer.
  """
  layer_class = "split_batch_time"

  def __init__(self, base, **kwargs):
    """
    :param LayerBase base: used to recover the seq-lens
    """
    super(SplitBatchTimeLayer, self).__init__(**kwargs)
    assert base.output.time_dim_axis is not None
    base_shape = tf.shape(base.output.placeholder)
    batch_dim = base_shape[base.output.batch_dim_axis]
    time_dim = tf.shape(base.output.placeholder)[base.output.time_dim_axis]
    seq_lens = base.output.get_sequence_lengths()
    assert self.input_data.batch_dim_axis == 0
    input_shape = tf.shape(self.input_data.placeholder)
    input_shape = [input_shape[i] for i in range(self.input_data.batch_ndim)]
    self.output.placeholder = tf.reshape(self.input_data.placeholder, shape=[batch_dim, time_dim] + input_shape[1:])
    self.output.size_placeholder = {i + 1: v for (i, v) in self.input_data.size_placeholder.items()}
    self.output.size_placeholder[0] = seq_lens

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param TFNetwork.TFNetwork network:
    :param get_layer:
    """
    super(SplitBatchTimeLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["base"] = get_layer(d.get("base", "data"))

  @classmethod
  def get_out_data_from_opts(cls, name, base, sources=(), **kwargs):
    """
    :param str name:
    :param LayerBase base:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    data = get_concat_sources_data_template(sources)
    data.name = "%s_output" % name
    assert data.batch_dim_axis == 0
    data.time_dim_axis = 1
    data.shape = (None,) + data.shape
    return data


class UnflattenNdLayer(_ConcatInputLayer):
  """
  Example:

    Assumes that the input is of shape (B,T,<Ds>) which represents flattened images,
    where each image is of size width * height.
    We additionally provide these image sizes (shape (B,2)), i.e. (width,height) tuples.
    We return the unflattened images of shape (B,W,H,<Ds>), where W/H are the max width/height.

  This basically wraps :func:`TFUtil.unflatten_nd`.
  """
  layer_class = "unflatten_nd"
  recurrent = True

  def __init__(self, sizes, num_axes, declare_same_sizes_as=None, **kwargs):
    """
    :param LayerBase sizes:
    :param int num_axes:
    :param dict[int,LayerBase] declare_same_sizes_as:
    """
    super(UnflattenNdLayer, self).__init__(**kwargs)
    input_data = self.input_data.copy_as_batch_major()
    sizes_data = sizes.output.copy_as_batch_major()
    assert sizes_data.batch_ndim == 2
    assert sizes_data.batch_shape[1] in (None, num_axes)  # also allow None...
    self.output.placeholder = TFUtil.unflatten_nd(input_data.placeholder, sizes_data.placeholder, num_axes=num_axes)
    self.output.size_placeholder = {i: sizes_data.placeholder[:, i] for i in range(num_axes)}
    if declare_same_sizes_as:
      for i, other in declare_same_sizes_as.items():
        assert 0 <= i < num_axes
        other_dim_tag = other.output.get_size_dim_tag(0)
        other_dim_tag.set_tag_on_size_tensor(self.output.size_placeholder[i])

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param TFNetwork.TFNetwork network:
    :param get_layer:
    """
    super(UnflattenNdLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    if "sizes" in d:  # check whether we need the param is later
      d["sizes"] = get_layer(d["sizes"])
    if d.get("declare_same_sizes_as", None):
      assert isinstance(d["declare_same_sizes_as"], dict)
      d["declare_same_sizes_as"] = {i: get_layer(name) for (i, name) in d["declare_same_sizes_as"].items()}

  @classmethod
  def get_out_data_from_opts(cls, name, sources, num_axes, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param int num_axes:
    :rtype: Data
    """
    input_data = get_concat_sources_data_template(sources).copy_as_batch_major()
    assert input_data.batch_ndim >= 2 and input_data.is_time_axis_dynamic()
    feature_dim_axis_or_unspecified = input_data.feature_dim_axis_or_unspecified
    if feature_dim_axis_or_unspecified is not NotSpecified:
      feature_dim_axis_or_unspecified -= input_data.batch_ndim
      assert feature_dim_axis_or_unspecified < 0
    res = Data(
      name="%s_output" % name,
      shape=((None,) * num_axes) + input_data.shape[1:],
      batch_dim_axis=0,
      time_dim_axis=1,
      feature_dim_axis=feature_dim_axis_or_unspecified,
      dtype=input_data.dtype)
    return res


class ExpandDimsLayer(_ConcatInputLayer):
  """
  Adds some axis.
  """
  layer_class = "expand_dims"

  def __init__(self, axis, dim=1, **kwargs):
    """
    :param str|int axis: axis to add, e.g. "F"|"feature" or "spatial"|"time"|"T".
      if this is an integer, the input data is first converted into batch-major mode,
      and then this is counted with batch-dim.
    :param int dim: dimension of new axis (1 by default)
    """
    super(ExpandDimsLayer, self).__init__(**kwargs)
    data = self.input_data
    if isinstance(axis, int):
      data = data.copy_as_batch_major()
    axis = self._get_axis(data=data, axis=axis)
    from TFUtil import expand_dims_unbroadcast
    self.output.placeholder = expand_dims_unbroadcast(data.placeholder, axis=axis, dim=dim)
    self.output.size_placeholder = {
      (i if (data.get_batch_axis(i) < axis) else i + 1): v
      for (i, v) in data.size_placeholder.items()}

  @classmethod
  def _get_axis(cls, data, axis):
    """
    This returns where to add the new axis.
    This is supposed to be after the specified axis, e.g. after the feature-dim.

    :param Data data:
    :param str|int axis: e.g. "F"|"feature" or "spatial"|"time"
    :return: axis as int for data.placeholder
    :rtype: int
    """
    if isinstance(axis, int):
      return axis
    assert isinstance(axis, str)
    axis = axis.lower()
    if axis in ["f", "feature"]:
      assert not data.sparse
      assert data.feature_dim_axis_or_unspecified is NotSpecified
      return data.batch_ndim
    elif axis in ["spatial", "time", "t"]:
      if data.sparse:
        return data.batch_ndim
      else:
        return data.batch_ndim - 1
    else:
      raise Exception("invalid axis %r" % axis)

  @classmethod
  def get_out_data_from_opts(cls, name, axis, dim=1, sources=(), **kwargs):
    """
    :param str name:
    :param str axis:
    :param int dim:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    init_axis = axis
    data = get_concat_sources_data_template(sources)
    data.name = "%s_output" % name
    if isinstance(axis, int):
      data = data.copy_as_batch_major()
    axis = cls._get_axis(data=data, axis=axis)
    if axis == data.batch_ndim and not data.sparse:
      assert data.feature_dim_axis_or_unspecified is NotSpecified
      data.dim = dim
    axis_wo_batch = data.get_batch_axis_excluding_batch(axis)
    data.shape = data.shape[:axis_wo_batch] + (dim,) + data.shape[axis_wo_batch:]
    if isinstance(init_axis, str):
      if init_axis.lower() in ["spatial", "time", "t"] and data.time_dim_axis is None:
        data.time_dim_axis = axis
    return data


class SwapAxesLayer(_ConcatInputLayer):
  """
  Swaps two axes. Basically a wrapper around :func:`TFUtil.swapaxes`.
  See also :class:`ReinterpretDataLayer`.
  """
  layer_class = "swap_axes"

  def __init__(self, axis1, axis2, **kwargs):
    """
    :param int|str axis1:
    :param int|str axis2:
    """
    super(SwapAxesLayer, self).__init__(**kwargs)
    from TFUtil import swapaxes
    axis1 = self.input_data.get_axis_from_description(axis1)
    axis2 = self.input_data.get_axis_from_description(axis2)
    self.output.placeholder = swapaxes(self.input_data.placeholder, axis1=axis1, axis2=axis2)
    axis1_wo_b = self.output.get_batch_axis_excluding_batch(axis1)
    axis2_wo_b = self.output.get_batch_axis_excluding_batch(axis2)
    self.output.size_placeholder = {
      self._translate_axis(i, axis1_wo_b, axis2_wo_b): v for (i, v) in self.input_data.size_placeholder.items()}

  @classmethod
  def _translate_axis(cls, axis_to_translate, axis1, axis2):
    """
    :param int|None axis_to_translate:
    :param int axis1:
    :param int axis2:
    :return: new axis
    :rtype: int|None
    """
    if axis_to_translate == axis1:
      return axis2
    if axis_to_translate == axis2:
      return axis1
    return axis_to_translate

  @classmethod
  def get_out_data_from_opts(cls, name, sources, axis1, axis2, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param int|str axis1:
    :param int|str axis2:
    :rtype: Data
    """
    out = get_concat_sources_data_template(sources, name="%s_output" % name)
    axis1 = out.get_axis_from_description(axis1)
    axis2 = out.get_axis_from_description(axis2)
    assert axis1 != axis2, "would be no-op. currently this is an error."
    assert axis1 != out.batch_dim_axis and axis2 != out.batch_dim_axis, "currently not supported..."
    axis1_wo_b = out.get_batch_axis_excluding_batch(axis1)
    axis2_wo_b = out.get_batch_axis_excluding_batch(axis2)
    shape = list(out.shape)
    shape[axis1_wo_b], shape[axis2_wo_b] = shape[axis2_wo_b], shape[axis1_wo_b]
    out.shape = tuple(shape)
    if not out.sparse:
      out.dim = out.shape[-1]
    out.time_dim_axis = cls._translate_axis(out.time_dim_axis, axis1, axis2)
    if out.feature_dim_axis_or_unspecified is not NotSpecified:
      out.feature_dim_axis = cls._translate_axis(out.feature_dim_axis, axis1, axis2)
    return out


class ReinterpretDataLayer(_ConcatInputLayer):
  """
  Acts like the :class:`CopyLayer` but reinterprets the role of some axes or data.
  """
  layer_class = "reinterpret_data"

  # noinspection PyUnusedLocal
  def __init__(self, switch_axes=None, size_base=None, set_axes=None,
               enforce_batch_major=False, enforce_time_major=False, increase_sparse_dim=None, **kwargs):
    """
    :param str|list[str] switch_axes: e.g. "bt" to switch batch and time axes
    :param LayerBase|None size_base:
    :param dict[str,int] set_axes:
    :param bool enforce_batch_major:
    :param bool enforce_time_major:
    """
    super(ReinterpretDataLayer, self).__init__(**kwargs)
    # All is done already in get_out_data_from_opts().

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param TFNetwork.TFNetwork network:
    :param get_layer:
    """
    super(ReinterpretDataLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    if d.get("size_base"):
      d["size_base"] = get_layer(d["size_base"])

  @classmethod
  def get_out_data_from_opts(cls, name, sources,
                             switch_axes=None, size_base=None, set_axes=None,
                             enforce_batch_major=False, enforce_time_major=False,
                             increase_sparse_dim=None, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param str|list[str] switch_axes: e.g. "bt" to switch batch and time axes
    :param LayerBase|None size_base:
    :param dict[str,int] set_axes:
    :param bool enforce_batch_major:
    :param bool enforce_time_major:
    :param int|None increase_sparse_dim: if sparse, add this to the dim
    """
    out = get_concat_sources_data_template(sources, name="%s_output" % name)
    assert not (enforce_batch_major and enforce_time_major)
    if enforce_batch_major:
      out = out.copy_as_batch_major()
    if enforce_time_major:
      out = out.copy_as_time_major()

    def map_axis_name(s):
      """
      :param str s:
      :rtype: str
      """
      if s.upper() == "B":
        return "batch_dim_axis"
      if s.upper() == "T":
        return "time_dim_axis"
      assert s in ["batch_dim_axis", "time_dim_axis"]
      return s

    if switch_axes:
      assert len(switch_axes) == 2
      axes_s = list(map(map_axis_name, switch_axes))
      axes = [getattr(out, s) for s in axes_s]
      for i in range(len(axes)):
        setattr(out, axes_s[i], axes[(i + 1) % len(axes)])
    if set_axes:
      assert enforce_batch_major or enforce_time_major
      for s, i in sorted(set_axes.items()):
        setattr(out, map_axis_name(s), i)
    if size_base:
      out.size_placeholder = size_base.output.size_placeholder.copy()
    if increase_sparse_dim:
      assert out.sparse
      out.dim += increase_sparse_dim
    return out


class ConvLayer(_ConcatInputLayer):
  """
  A generic convolution layer which supports 1D, 2D and 3D convolution.
  Pooling can be done in the separate "pool" layer.
  """

  layer_class = "conv"
  recurrent = True  # we must not allow any shuffling in the time-dim or so

  # noinspection PyUnusedLocal
  def __init__(self, n_out, filter_size, padding, strides=1, dilation_rate=1,
               input_expand_dims=0, input_add_feature_dim=False, input_split_feature_dim=None,
               auto_use_channel_first=False,
               with_bias=False,
               activation=None,
               forward_weights_init="glorot_uniform", bias_init=0.0,
               **kwargs):
    """
    :param int n_out: number of outgoing features
    :param tuple[int] filter_size: (width,), (height,width) or (depth,height,width) for 1D/2D/3D conv.
      the input data ndim must match, or you can add dimensions via input_expand_dims or input_add_feature_dim.
      it will automatically swap the batch-dim to the first axis of the input data.
    :param str padding: "same" or "valid"
    :param int|tuple[int] strides: strides for the spatial dims,
      i.e. length of this tuple should be the same as filter_size, or a single int.
    :param int input_expand_dims: number of dynamic dims to add to the input
    :param bool input_add_feature_dim: will add a dim at the end and use input-feature-dim == 1,
      and use the original input feature-dim as a spatial dim.
    :param bool auto_use_channel_first: convert the input to NCHW or not
    :param None|int input_split_feature_dim: if set, like input_add_feature_dim it will add a new feature dim
      which is of value input_split_feature_dim, and the original input feature dim
      will be divided by input_split_feature_dim, thus it must be a multiple of that value.
    :param bool with_bias: if True, will add a bias to the output features
    :param None|str activation: if set, will apply this function at the end
    """
    padding = padding.upper()
    assert padding in ["SAME", "VALID"], "no other padding supported at the moment"
    assert "out_type" not in kwargs, "don't set out_type explicitly for this layer"
    assert len(filter_size) in (1, 2, 3), "only 1D conv, 2D conv or 3D conv supported"
    super(ConvLayer, self).__init__(**kwargs)
    if isinstance(strides, int):
      strides = [strides] * len(filter_size)
    else:
      strides = list(strides)
    assert len(strides) == len(filter_size)
    if isinstance(dilation_rate, int):
      dilation_rate = [dilation_rate] * len(filter_size)
    else:
      dilation_rate = list(dilation_rate)
    assert len(dilation_rate) == len(filter_size)
    assert not self.input_data.sparse
    input_data = self.input_data.copy_as_batch_major()
    if input_expand_dims:
      for i in range(input_expand_dims):
        input_data = input_data.copy_add_spatial_dim()
    if input_split_feature_dim:
      # Split the feature dimension.
      input_data = input_data.copy_split_feature_dim(input_split_feature_dim)
    if input_add_feature_dim:
      # Add a feature dimension; any other static dims will be used as dynamic dims below.
      input_data = input_data.copy_add_feature_dim()
    if self.output.is_batch_feature_major:
      input_data = input_data.copy_as_batch_feature_major()
    else:
      input_data = input_data.copy_with_feature_dim_axis(-1)
    assert input_data.feature_dim_axis is not None, (
      "this should be our single input feature dim now. otherwise use input_add_feature_dim")
    assert len(input_data.get_spatial_axes()) == len(filter_size), (
      "filter-size-dimension does not match the input data. " +
      "this is %i-D conv but number of spatial dims is %i in the input %s. " % (
        len(filter_size), len(input_data.get_spatial_axes()), self.input_data.get_description()) +
      "consider using input_expand_dims or input_add_feature_dim.")
    filter_shape = list(filter_size) + [input_data.dim, n_out]
    from TFUtil import get_initializer
    with self.var_creation_scope():
      fwd_weights_initializer = get_initializer(
        forward_weights_init, seed=self.network.random.randint(2 ** 31), eval_local_ns={"layer": self})
      filters = self.add_param(tf.get_variable(name="W", shape=filter_shape, initializer=fwd_weights_initializer))
    data_format = None
    if input_data.is_batch_feature_major:
      assert self.output.is_batch_feature_major
      data_format = {1: "NCW", 2: "NCHW", 3: "NCDHW"}[len(filter_size)]
    y = tf.nn.convolution(
      input_data.placeholder, data_format=data_format,
      filter=filters,
      padding=padding, strides=strides, dilation_rate=dilation_rate)
    # y shape is [batch] + dynamic_dims + [n_out].
    if with_bias:
      with self.var_creation_scope():
        bias_initializer = get_initializer(
          bias_init, seed=self.network.random.randint(2 ** 31) if bias_init else 0, eval_local_ns={"layer": self})
        b = self.add_param(tf.get_variable(name="bias", shape=(n_out,), initializer=bias_initializer))
      y += b
    if activation:
      from TFUtil import get_activation_function
      act_func = get_activation_function(activation)
      self.output_before_activation = OutputWithActivation(y, act_func=act_func)
    else:
      self.output_before_activation = OutputWithActivation(y)
    y = self.output_before_activation.y
    self.output.placeholder = y
    self.output.size_placeholder = {
      i: input_data.size_placeholder[i]
      for i in input_data.get_spatial_axes()
      if i in input_data.size_placeholder}
    index_shift = self.output.get_spatial_axes()[0]
    for i in list(self.output.size_placeholder.keys()):
      self.output.size_placeholder[i] = self.calc_out_dim(
        in_dim=self.output.size_placeholder[i],
        filter_size=filter_size[i - index_shift], stride=strides[i - index_shift],
        dilation_rate=dilation_rate[i - index_shift], padding=padding)

  @classmethod
  def calc_out_dim(cls, in_dim, filter_size, stride, padding, dilation_rate=1):
    """
    :param int|tf.Tensor|T in_dim: dimension in some axis
    :param int filter_size: e.g. 2, for the corresponding axis
    :param int stride: e.g. 1, for the corresponding axis
    :param int dilation_rate: e.g. 1
    :param str padding: "valid" or "same"
    :return: the output dimension
    :rtype: T
    """
    def ceildiv(a, b):
      """
      :param int|tf.Tensor|T a:
      :param int|tf.Tensor|T b:
      :rtype: T
      """
      return -(-a // b)

    padding = padding.upper()
    # See tf.nn.convolution() documentation for more.
    if padding == "SAME":
      return ceildiv(in_dim, stride)
    elif padding == "VALID":
      max_func = tf.maximum if isinstance(in_dim, tf.Tensor) else max
      return max_func(ceildiv((in_dim - (filter_size - 1) * dilation_rate), stride), 0)
    else:
      raise Exception("invalid padding %r" % padding)

  # noinspection PyUnusedLocal
  @classmethod
  def _get_out_type_from_opts(cls, n_out, filter_size, padding, strides=1, dilation_rate=1, sources=(),
                              input_expand_dims=0, input_add_feature_dim=False, input_split_feature_dim=None,
                              auto_use_channel_first=False,
                              **kwargs):
    data = get_concat_sources_data_template(sources)
    shape = [None] * len(filter_size) + [n_out]
    if isinstance(strides, int):
      strides = [strides] * len(filter_size)
    else:
      assert isinstance(strides, (tuple, list))
      strides = list(strides)
    assert len(strides) == len(filter_size)
    if isinstance(dilation_rate, int):
      dilation_rate = [dilation_rate] * len(filter_size)
    else:
      assert isinstance(dilation_rate, (tuple, list))
      dilation_rate = list(dilation_rate)
    assert len(dilation_rate) == len(filter_size)
    padding = padding.upper()
    if input_expand_dims == 0 and not input_add_feature_dim and not input_split_feature_dim:
      # Maybe we have a chance to correctly define the output shapes.
      index_shift = data.get_spatial_axes()[0]
      for i in range(len(filter_size)):
        if data.shape[i + index_shift] is not None:
          shape[i] = cls.calc_out_dim(
            in_dim=data.shape[i + index_shift],
            filter_size=filter_size[i], stride=strides[i], dilation_rate=dilation_rate[i], padding=padding)
    feature_dim_axis = NotSpecified
    # Swap the dims if the input dim order doesn't fit the flag auto_use_channel_first.
    if (TFUtil.is_gpu_available() and auto_use_channel_first) or data.is_batch_feature_major:
      feature_dim_axis = 1
      shape = shape[-1:] + shape[:-1]
    return {
      "dim": n_out,
      "shape": shape,
      "batch_dim_axis": 0,
      "feature_dim_axis": feature_dim_axis,
      "sparse": False}

  @classmethod
  def get_out_data_from_opts(cls, **kwargs):
    """
    Via :func:`_get_out_type_from_opts`.

    :rtype: Data
    """
    out_type = cls._get_out_type_from_opts(**kwargs)
    return super(ConvLayer, cls).get_out_data_from_opts(out_type=out_type, **kwargs)


class PoolLayer(_ConcatInputLayer):
  """
  A generic N-D pooling layer.
  This would usually be done after a convolution for down-sampling.
  """

  layer_class = "pool"
  recurrent = True  # we should not shuffle in the time-dimension

  # noinspection PyUnusedLocal
  def __init__(self, mode, pool_size, padding="VALID", dilation_rate=1, strides=None,
               use_channel_first=False,
               **kwargs):
    """
    :param str mode: "max" or "avg"
    :param tuple[int] pool_size: shape of the window of each reduce
    :param str padding: "valid" or "same"
    :param tuple[int]|int dilation_rate:
    :param tuple[int]|int|None strides: in contrast to tf.nn.pool, the default (if it is None) will be set to pool_size
    :param bool use_channel_first: if set, will transform input to NCHW format
    """
    assert "n_out" not in kwargs
    assert "out_type" not in kwargs
    from TFUtil import check_input_dim, DimensionTag
    mode = mode.upper()
    assert mode in ["MAX", "AVG"]
    padding = padding.upper()
    assert padding in ["VALID", "SAME"]
    if isinstance(dilation_rate, int):
      dilation_rate = [dilation_rate] * len(pool_size)
    assert len(dilation_rate) == len(pool_size)
    if strides is None:
      strides = pool_size
    elif isinstance(strides, int):
      strides = [strides] * len(pool_size)
    assert len(strides) == len(pool_size)
    super(PoolLayer, self).__init__(**kwargs)
    if all([s == 1 for s in pool_size]) and all([s == 1 for s in strides]):
      # Identity function. Just copy and don't do anything.
      self.output = self.input_data.copy("%s_output" % self.name)
      return
    # We want to prepare the input data such that the batch-dim is the very first,
    # the feature-dim is the very last ("NHWC" format) or right after batch-dim ("NCHW"),
    # and all other dims are where we convolve over.
    if self.output.is_batch_feature_major:
      input_data = self.input_data.copy_as_batch_feature_major()
      x = check_input_dim(input_data.get_placeholder_as_batch_major(), 1, self.input_data.dim)
    else:
      input_data = self.input_data.copy_with_feature_dim_axis(-1)
      x = check_input_dim(input_data.get_placeholder_as_batch_major(), -1, self.input_data.dim)
    data_format = None
    if input_data.is_batch_feature_major:
      assert self.output.is_batch_feature_major
      data_format = {1: "NCW", 2: "NCHW", 3: "NCDHW"}[len(pool_size)]
    y = tf.nn.pool(
      x, window_shape=pool_size, pooling_type=mode, padding=padding,
      dilation_rate=dilation_rate, strides=strides, data_format=data_format)
    # y shape is [batch] + spatial_dims + [n_out].
    self.output.placeholder = y
    self.output.size_placeholder = {
      i: input_data.size_placeholder[i]
      for i in range(len(pool_size))
      if i in input_data.size_placeholder}
    index_shift = self.output.time_dim_axis_excluding_batch
    for i in list(self.output.size_placeholder.keys()):
      self.output.size_placeholder[i] = ConvLayer.calc_out_dim(
        in_dim=self.output.size_placeholder[i],
        filter_size=pool_size[i - index_shift], stride=strides[i - index_shift],
        dilation_rate=dilation_rate[i - index_shift], padding=padding)
      tag = DimensionTag(
        description="spatial:%i:%s" % (i, self.get_base_absolute_name_scope_prefix()[:-1]),
        kind=DimensionTag.Types.Spatial)
      tag.set_tag_on_size_tensor(self.output.size_placeholder[i])

  @classmethod
  def get_out_data_from_opts(cls, name, pool_size, strides=None, dilation_rate=1, sources=(), padding="VALID",
                             use_channel_first=False,
                             **kwargs):
    """
    :param str name:
    :param tuple[int]|list[int] pool_size:
    :param tuple[int]|list[int]|int strides:
    :param int|tuple[int]|list[int] dilation_rate:
    :param list[LayerBase] sources:
    :param str padding:
    :param bool use_channel_first:
    :rtype: Data
    """
    # y shape is [batch] + spatial_dims + [n_out].
    data = get_concat_sources_data_template(sources, name="%s_output" % name)
    shape = [None] * len(pool_size) + [data.dim]
    if strides is None:
      strides = pool_size
    if isinstance(strides, int):
      strides = [strides] * len(pool_size)
    else:
      strides = list(strides)
    assert len(strides) == len(pool_size)
    if isinstance(dilation_rate, int):
      dilation_rate = [dilation_rate] * len(pool_size)
    else:
      dilation_rate = list(dilation_rate)
    assert len(dilation_rate) == len(pool_size)
    if all([s == 1 for s in pool_size]) and all([s == 1 for s in strides]):
      # Identity function. Just copy and don't do anything.
      return data
    padding = padding.upper()
    index_shift = data.time_dim_axis_excluding_batch
    for i in range(len(pool_size)):
      if data.shape[i + index_shift] is not None:
        shape[i] = ConvLayer.calc_out_dim(
          in_dim=data.shape[i + index_shift],
          filter_size=pool_size[i], stride=strides[i], dilation_rate=dilation_rate[i], padding=padding)
    feature_dim_axis = NotSpecified
    # Swap the dims if use_channel_first is set.
    if TFUtil.is_gpu_available() and use_channel_first:
      feature_dim_axis = 1
      shape = shape[-1:] + shape[:-1]
    return Data(
      name="%s_output" % name,
      shape=tuple(shape),
      dim=data.dim,
      dtype=data.dtype,
      sparse=False,
      batch_dim_axis=0,
      feature_dim_axis=feature_dim_axis,
      beam_size=data.beam_size)


class ReduceLayer(_ConcatInputLayer):
  """
  This reduces some axis by using "sum" or "max".
  It's basically a wrapper around tf.reduce_sum or tf.reduce_max.
  """
  layer_class = "reduce"

  def __init__(self, mode, axes=None, axis=None, keep_dims=False, enforce_batch_dim_axis=None, use_time_mask=None,
               **kwargs):
    """
    :param str mode: "sum" or "max", "argmin", "min", "argmin", or "mean"
    :param int|list[int]|str axes: One axis or multiple axis to reduce.
      It accepts the special tokens "B"|"batch", "spatial", "spatial_except_time", or "F"|"feature",
      and it is strongly recommended to use some of these symbolic names.
      See :func:`Data.get_axes_from_description`.
    :param int|list[int]|str axis: for compatibility, can be used instead of ``axes``
    :param bool keep_dims: if dimensions should be kept (will be 1)
    :param int enforce_batch_dim_axis: will swap the batch-dim-axis of the input with the given axis.
      e.g. 0: will convert the input into batch-major format if not already like that.
      Note that this is still not enough in some cases, e.g. when the other axes are also not as expected.
      The strong recommendation is to use a symbolic axis description.
    :param bool use_time_mask: if we reduce over the time-dim axis, use the seq len info.
      By default, in that case, it will be True.
    """
    from TFUtil import expand_multiple_dims
    super(ReduceLayer, self).__init__(**kwargs)
    if axis is not None:
      print("reduce layer %r: option 'axis' is deprecated, use 'axes' instead" % kwargs["name"], file=log.v4)
      assert axes is None, "don't provide both 'axes' and 'axis', layer %r" % kwargs["name"]
      axes = axis
    if enforce_batch_dim_axis is None and self.need_enforce_batch_dim_axis(axes):
      enforce_batch_dim_axis = 0
    if "n_out" in kwargs:
      assert kwargs["n_out"] == self.output.dim
    assert "out_type" not in kwargs
    mode = mode.lower()
    assert mode in ["max", "argmax", "min", "argmin", "sum", "avg", "mean"]
    assert not self.input_data.sparse
    x = self.input_data
    if enforce_batch_dim_axis is not None and x.batch_dim_axis != enforce_batch_dim_axis:
      x = x.copy_with_batch_dim_axis(enforce_batch_dim_axis)
    axes = self.get_axes(axes, input_data=x)
    if use_time_mask is None:
      if x.time_dim_axis in axes:
        use_time_mask = True
      else:
        use_time_mask = False
    assert isinstance(use_time_mask, bool)
    if mode == "max":
      f = tf.reduce_max
    elif mode == "argmax":
      f = tf.argmax
    elif mode == "min":
      f = tf.reduce_min
    elif mode == "argmin":
      f = tf.argmin
    elif mode == "sum":
      f = tf.reduce_sum
    elif mode in ["avg", "mean"]:
      f = tf.reduce_mean
    else:
      raise Exception("invalid mode %r" % mode)
    x_ = x.placeholder
    # Check if we should ignore some frames, e.g. via masking.
    if use_time_mask:
      if f is tf.reduce_sum:
        # For sum, the fastest and simplest way is masking.
        for axis in axes:
          if axis == x.batch_dim_axis:
            continue
          axis_wo_b = x.get_batch_axis_excluding_batch(axis)
          if axis_wo_b not in x.size_placeholder:
            continue
          assert axis == x.time_dim_axis
          mask = x.get_sequence_mask()  # e.g. (B,T)
          mask = expand_multiple_dims(
            mask, [i for i in range(x.batch_ndim) if i not in [x.batch_dim_axis, axis]])  # e.g. (B,1,T) with axis=-1
          mask = tf.logical_and(mask, tf.ones_like(x_, dtype=mask.dtype))
          x_ = tf.where(mask, x_, tf.zeros_like(x.placeholder), "x_masked_axis_%i" % axis)
      elif f in (tf.reduce_min, tf.reduce_mean, tf.reduce_max):
        # Flattening.
        if x.time_dim_axis in axes:
          assert not keep_dims, "not yet implemented otherwise"
          assert x.batch_dim_axis in axes, "not yet implemented otherwise"
          axes = [a if (a < x.time_dim_axis) else (a - 1)
                  for a in axes if a != x.time_dim_axis]
          x = x.copy_time_flattened()
          x_ = x.placeholder
    if f in (tf.argmax, tf.argmin):
      assert len(axes) == 1, "For argmax/argmin, only one reduction axis is supported"
      y = tf.to_float(f(x_, axis=axes[0]))
      # argmax and argmin don't support keep_dims argument
      # so we emulate it manually
      if keep_dims:
        y = expand_multiple_dims(y, axes=axes)
    else:
      y = f(x_, axis=axes, keep_dims=keep_dims)
    y_dyn_sizes = x.size_placeholder.copy()
    if keep_dims:
      for i in axes:
        if i in y_dyn_sizes:
          del y_dyn_sizes[i]
    else:
      for i in reversed(sorted(axes)):
        if i in y_dyn_sizes:
          del y_dyn_sizes[i]
        y_dyn_sizes = {(j if (j < i) else (j - 1)): s
                       for (j, s) in list(y_dyn_sizes.items())}
    self.output.placeholder = y
    self.output.size_placeholder = y_dyn_sizes

  @classmethod
  def need_enforce_batch_dim_axis(cls, axes):
    """
    :param int|list[int]|str axes:
    :return: if any integer is in axes, thus we should have a fixed dimension layout
    :rtype: bool
    """
    if isinstance(axes, int):
      return True
    if isinstance(axes, str):
      return False
    assert isinstance(axes, (list, tuple))
    return any([cls.need_enforce_batch_dim_axis(a) for a in axes])

  @classmethod
  def get_axes(cls, axis, input_data):
    """
    :param axis: see self.__init__()
    :param Data input_data:
    :return: list of axes
    :rtype: list[int]
    """
    axis = input_data.get_axes_from_description(axis)
    assert len(axis) > 0, "no axis to reduce. input_data: %s" % (input_data,)
    return axis

  @classmethod
  def get_out_data_from_opts(cls, name, sources, axes=None, axis=None, keep_dims=False, enforce_batch_dim_axis=None,
                             **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param str|list[str]|None axes:
    :param str|None axis:
    :param bool keep_dims:
    :param int|None enforce_batch_dim_axis:
    :rtype: Data
    """
    if axis is not None:
      axes = axis
    if enforce_batch_dim_axis is None and cls.need_enforce_batch_dim_axis(axes):
      enforce_batch_dim_axis = 0
    x = get_concat_sources_data_template(sources)
    assert not x.sparse
    if enforce_batch_dim_axis is not None and x.batch_dim_axis != enforce_batch_dim_axis:
      x = x.copy_with_batch_dim_axis(enforce_batch_dim_axis)
    axes = cls.get_axes(axis=axes, input_data=x)
    y_shape = list(x.batch_shape)
    out_batch_dim_axis = x.batch_dim_axis
    out_feature_dim_axis = x.feature_dim_axis_or_unspecified
    out_time_dim_axis = x.time_dim_axis
    if keep_dims:
      for i in axes:
        y_shape[i] = 1
      del y_shape[x.batch_dim_axis]
    else:
      if out_batch_dim_axis in axes:
        out_batch_dim_axis = None
      if out_time_dim_axis in axes:
        out_time_dim_axis = NotSpecified
      if out_feature_dim_axis in axes:
        out_feature_dim_axis = NotSpecified
      for i in reversed(sorted(set(axes + [x.batch_dim_axis] if x.batch_dim_axis is not None else []))):
        del y_shape[i]
      for i in reversed(sorted(set(axes))):
        if out_batch_dim_axis and i < out_batch_dim_axis:
          out_batch_dim_axis -= 1
        if out_time_dim_axis and out_time_dim_axis is not NotSpecified and i < out_time_dim_axis:
          out_time_dim_axis -= 1
        if out_feature_dim_axis and out_feature_dim_axis is not NotSpecified and i < out_feature_dim_axis:
          out_feature_dim_axis -= 1
    return Data(
      name="%s_output" % name,
      shape=y_shape,
      batch_dim_axis=out_batch_dim_axis,
      time_dim_axis=out_time_dim_axis,
      feature_dim_axis=out_feature_dim_axis,
      dtype=x.dtype,
      sparse=False,
      beam_size=x.beam_size)


class ReduceOutLayer(_ConcatInputLayer):
  """
  Combination of :class:`SplitDimsLayer` applied to the feature dim
  and :class:`ReduceLayer` applied to the resulting feature dim.
  This can e.g. be used to do maxout.
  """
  layer_class = "reduce_out"

  def __init__(self, mode, num_pieces, **kwargs):
    """
    :param str mode: "sum" or "max" or "mean"
    :param int num_pieces: how many elements to reduce. The output dimension will be input.dim // num_pieces.
    """
    super(ReduceOutLayer, self).__init__(**kwargs)
    if mode == "max":
      f = tf.reduce_max
    elif mode == "sum":
      f = tf.reduce_sum
    elif mode in ["avg", "mean"]:
      f = tf.reduce_mean
    else:
      raise Exception("invalid mode %r" % mode)
    shape = tf.shape(self.input_data.placeholder)
    shape = [shape[i] for i in range(self.input_data.batch_ndim)]
    x = tf.reshape(self.input_data.placeholder, shape[:-1] + [self.output.dim, num_pieces])
    x.set_shape(tf.TensorShape(self.input_data.batch_shape[:-1] + (self.output.dim, num_pieces)))
    x = f(x, axis=self.input_data.batch_ndim, name="%s_out" % mode)
    x.set_shape(tf.TensorShape(self.output.batch_shape))
    self.output.placeholder = x
    self.output.size_placeholder = self.input_data.size_placeholder.copy()

  @classmethod
  def get_out_data_from_opts(cls, num_pieces, sources, name, **kwargs):
    """
    :param int num_pieces:
    :param list[LayerBase] sources:
    :param str name:
    :rtype: Data
    """
    out = get_concat_sources_data_template(sources, name="%s_output" % name)
    assert not out.sparse
    assert out.dim % num_pieces == 0
    out.dim //= num_pieces
    out.shape = out.shape[:-1] + (out.dim,)
    return out


class SqueezeLayer(_ConcatInputLayer):
  """
  Removes an axis with dimension 1.
  This is basically a wrapper around tf.squeeze.
  """
  layer_class = "squeeze"

  # noinspection PyUnusedLocal
  def __init__(self, axis, enforce_batch_dim_axis=None, allow_no_op=False, **kwargs):
    """
    :param int|list[int]|str axis: one axis or multiple axis to squeeze.
      this is counted with batch-dim, which by default is axis 0 (see enforce_batch_dim_axis).
      it also accepts the special tokens "B"|"batch", "spatial", "spatial_except_time", or "F"|"feature"
    :param int|None enforce_batch_dim_axis:
    :param bool allow_no_op:
    """
    super(SqueezeLayer, self).__init__(**kwargs)
    input_data = self.input_data
    if enforce_batch_dim_axis is not None and input_data.batch_dim_axis != enforce_batch_dim_axis:
      input_data = input_data.copy_with_batch_dim_axis(enforce_batch_dim_axis)
    axes = self._get_axes(axis, input_data=input_data)
    x = input_data.placeholder
    for i in reversed(sorted(axes)):
      x = tf.squeeze(x, axis=i)
    self.output.placeholder = x
    self.output.size_placeholder = {
      i - len([j for j in axes if j < input_data.get_batch_axis(i)]): size
      for (i, size) in input_data.size_placeholder.items()
      if input_data.get_batch_axis(i) not in axes}

  @classmethod
  def _get_axes(cls, axis, input_data):
    """
    :param int|list[int]|str axis: one axis or multiple axis to squeeze.
    :param Data input_data:
    :rtype: list[int]
    """
    if axis == "auto":
      return [i for (i, dim) in enumerate(input_data.batch_shape) if dim == 1]
    return input_data.get_axes_from_description(axis)

  @classmethod
  def get_out_data_from_opts(cls, axis, enforce_batch_dim_axis=None, allow_no_op=False, sources=(), **kwargs):
    """
    :param axis:
    :param int|None enforce_batch_dim_axis:
    :param bool allow_no_op:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    input_data = get_concat_sources_data_template(sources)
    if enforce_batch_dim_axis is not None:
      input_data = input_data.copy_with_batch_dim_axis(enforce_batch_dim_axis)
    if axis == "auto":
      axis = cls._get_axes(axis, input_data=input_data)
    if allow_no_op:
      if not cls._get_axes(axis, input_data=input_data):
        return input_data.copy("%s_output" % kwargs["name"])
    return ReduceLayer.get_out_data_from_opts(
      axis=axis, keep_dims=False, enforce_batch_dim_axis=enforce_batch_dim_axis, sources=sources, **kwargs)


class WeightedSumLayer(_ConcatInputLayer):
  """
  Calculates a weighted sum, either over a complete axis of fixed dimension, or over some window.
  Can also do that for multiple axes.
  The weights are a trainable parameter matrix.
  Similar would be to use :class:`ElemwiseProdLayer` and :class:`ReduceLayer`,
  or just a :class:`DotLayer` with a :class:`VariableLayer`.
  See also :class:`LinearLayer`.
  """
  layer_class = "weighted_sum"

  def __init__(self, axes, padding=None, size=None, keep_dims=None, **kwargs):
    """
    :param str|list[str] axes: the axes to do the weighted-sum over
    :param str padding: "valid" or "same", in case of keep_dims=True
    :param None|tuple[int] size: the kernel-size. if left away, the axes must be of fixed dimension,
      and we will use keep_dims=False, padding="valid" by default.
      Otherwise, if given, you must also provide padding and keep_dims=True by default.
    :param bool keep_dims: if False, the axes will be squeezed away. see also `size`.
    """
    super(WeightedSumLayer, self).__init__(**kwargs)
    import numpy
    axes, padding, size, keep_dims = self._resolve_opts(
      input_data=self.input_data, axes=axes, padding=padding, size=size, keep_dims=keep_dims)
    assert len(axes) in [1, 2, 3]  # not supported otherwise
    axes = list(sorted(axes))
    # We want to transpose the input such that we get the axes in order [all axes which are not in axes] + axes.
    other_axes = [i for i in range(self.input_data.batch_ndim) if i not in axes]
    perm = other_axes + axes
    x = tf.transpose(self.input_data.placeholder, perm=perm)
    # Now merge all other_axes together, and add one new axis at the end, so that we get the shape
    # [new_batch_dim] + [shape(x)[a] for a in axes] + [1].
    x_shape = tf.shape(x)
    new_batch_dim = tf.reduce_prod(x_shape[:len(other_axes)])
    axes_shape = [x_shape[i] for i in range(len(other_axes), self.input_data.batch_ndim)]
    x = tf.reshape(x, shape=[new_batch_dim] + axes_shape + [1])
    with self.var_creation_scope():
      filters = self.add_param(tf.get_variable(
        name="W", shape=size, initializer=tf.constant_initializer(1.0 / numpy.prod(size))))
    filters = tf.reshape(filters, shape=list(size) + [1, 1])
    y = tf.nn.convolution(x, filter=filters, padding=padding.upper())  # result: (new_batch_dim, ..., 1)
    if keep_dims:
      y_shape = tf.shape(y)
      # Now split new_batch_dim again into the other_axes.
      y = tf.reshape(
        y, shape=[x_shape[i] for i in range(len(other_axes))] +
                 [y_shape[i + 1] for i in range(len(axes))])  # [shape of other_axes] + [shape of axes]
      # And revert the initial axes permutation.
      inv_perm = numpy.argsort(perm)
      y = tf.transpose(y, perm=inv_perm)  # original shape with maybe some less in the axes if padding="valid"
      self.output.placeholder = y
      self.output.size_placeholder = self.input_data.size_placeholder.copy()
      if padding == "valid":
        assert self.input_data.time_dim_axis not in axes, "size not yet implemented correctly..."
    else:  # not keep_dims
      # We expect that all the remaining shape of the axes can be reduced/squeezed, i.e. is all 1.
      # Thus, we can reshape it to just the shape of the other_axes.
      y = tf.reshape(y, shape=[x_shape[i] for i in range(len(other_axes))])
      # The axes are all in the right order already now, so no need to permute/transpose the axes.
      # We are ready.
      self.output.placeholder = y
      self.output.size_placeholder = {
        i - len([a for a in axes if a < self.input_data.get_batch_axis(i)]): v
        for (i, v) in self.input_data.size_placeholder.items()
        if self.input_data.get_batch_axis(i) not in axes}

  @classmethod
  def _resolve_opts(cls, input_data, axes, padding=None, size=None, keep_dims=None):
    """
    :param Data input_data:
    :param str|list[str] axes:
    :param None|str padding:
    :param None|tuple[int] size:
    :param None|bool keep_dims:
    :return: (axes, padding, size, keep_dims)
    :rtype: (list[int], str, tuple[int], bool)
    """
    axes = input_data.get_axes_from_description(axes)
    if size is None:
      size = [None] * len(axes)
      for i, a in enumerate(axes):
        assert input_data.batch_shape[a] is not None
        size[i] = input_data.batch_shape[a]
      if keep_dims is None:
        keep_dims = False
      if padding is None:
        padding = "valid"
    else:
      assert isinstance(size, (list, tuple))
      assert len(size) == len(axes)
      if keep_dims is None:
        keep_dims = True
      assert padding is not None
    return axes, padding, tuple(size), keep_dims

  @classmethod
  def get_out_data_from_opts(cls, name, sources, axes, padding=None, size=None, keep_dims=None, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param str|list[str] axes:
    :param str|None padding:
    :param None|tuple[int] size:
    :param bool|None keep_dims:
    :rtype: Data
    """
    data = get_concat_sources_data_template(sources, name="%s_output" % name)
    assert not data.sparse
    axes, padding, size, keep_dims = cls._resolve_opts(
      input_data=data, axes=axes, padding=padding, size=size, keep_dims=keep_dims)
    padding = padding.lower()
    res_dims = [data.batch_shape[a] for a in axes]
    if padding == "same":
      pass
    elif padding == "valid":
      for i in range(len(axes)):
        if res_dims[i] is not None:
          res_dims[i] -= size[i] - 1
          assert res_dims[i] > 0
    else:
      raise Exception("invalid padding %r" % padding)
    if keep_dims:
      shape = list(data.shape)
      assert data.batch_dim_axis not in axes
      for i, a in enumerate(axes):
        shape[data.get_batch_axis_excluding_batch(a)] = res_dims[i]
      data.shape = tuple(shape)
    else:
      assert all([d == 1 for d in res_dims])
      assert data.batch_dim_axis not in axes
      data.shape = tuple([d for (i, d) in enumerate(data.shape) if data.get_batch_axis(i) not in axes])
    data.dim = data.shape[data.get_batch_axis_excluding_batch(data.feature_dim_axis)]
    return data


class ElemwiseProdLayer(_ConcatInputLayer):
  """
  Element-wise product in some axes.
  Microsoft calls this "static attention", in Deep Conv. NN with Layer-wise Context Expansion and Attention (LACE).
  The matrix/tensor to be used for the product are given as a trainable parameter.
  See also :class:`LinearLayer`.
  """
  layer_class = "elemwise_prod"

  def __init__(self, axes, size=None, **kwargs):
    """
    :param str|list[str] axes: e.g. "spatial", but all those axes must be of fixed dimension
    :param tuple[int] size: for double-checking, you can explicitly provide the size
    """
    super(ElemwiseProdLayer, self).__init__(**kwargs)
    axes = self.input_data.get_axes_from_description(axes)
    axes = list(sorted(axes))
    shape_size = [None] * len(axes)
    for i, a in enumerate(axes):
      assert self.input_data.batch_shape[a] is not None
      shape_size[i] = self.input_data.batch_shape[a]
    if size is None:
      size = shape_size
    else:
      assert isinstance(size, (list, tuple))
      assert tuple(size) == tuple(shape_size), "wrong size %r with input_data %r" % (size, self.input_data)
    with self.var_creation_scope():
      w = self.add_param(tf.get_variable(name="W", shape=size, initializer=tf.constant_initializer(1.0)))
    w_full_shape = [self.input_data.batch_shape[a] if (a in axes) else 1
                    for a in range(self.input_data.batch_ndim)]
    w = tf.reshape(w, shape=w_full_shape)
    self.output.placeholder = tf.multiply(self.input_data.placeholder, w)
    self.output.size_placeholder = self.input_data.size_placeholder.copy()

  @classmethod
  def get_out_data_from_opts(cls, name, sources, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    # Just the same as the input.
    return get_concat_sources_data_template(sources, name="%s_output" % name)


class PrefixInTimeLayer(CopyLayer):
  """
  Adds some prefix in time dimension.
  """
  layer_class = "prefix_in_time"
  recurrent = True

  def __init__(self, prefix=0.0, repeat=1, **kwargs):
    """
    :param float|str prefix: either some constant or another layer
    :param int repeat: how often to repeat the prefix
    """
    super(PrefixInTimeLayer, self).__init__(**kwargs)
    assert self.output.time_dim_axis is not None
    assert isinstance(prefix, (float, int)), "other layer src not yet supported"
    c = tf.constant(prefix, dtype=self.output.dtype)
    shape = [((self.output.batch_shape[i] or tf.shape(self.output.placeholder)[i])
              if (i != self.output.time_dim_axis)
              else repeat)
             for i in range(self.output.batch_ndim)]
    x = tf.ones(shape, dtype=self.output.dtype)
    self.output.placeholder = tf.concat([x * c, self.output.placeholder], axis=self.output.time_dim_axis)
    self.output.size_placeholder[self.output.time_dim_axis_excluding_batch] += repeat


class DotLayer(LayerBase):
  """
  This performs a dot-product of two sources.
  The underlying matmul expects shapes (shared..., I, J) * (shared..., J, K) -> (shared..., I, K).
  We say that J is the axis to be reduced,
  I is the var-dim of source 1, and K is the var-dim of source 2.
  I, J, K can also be multiple axes from the sources.
  The var-dims don't need to exist.
  All other axes (shared...) are expected to match.
  """
  layer_class = "dot"

  def __init__(self, red1=-1, red2=-2, var1=-2, var2=-1, add_var2_if_empty=True, debug=False, **kwargs):
    """
    :param str|int|tuple[str|int]|list[str|int] red1: reduce axes of first source
    :param str|int|tuple[str|int]|list[str|int] red2: reduce axes of second source
    :param str|int|tuple[str|int]|list[str|int]|None var1: var axes of first source
    :param str|int|tuple[str|int]|list[str|int]|None var2: var axes of second source
    :param bool add_var2_if_empty: if var2=None, add dim=1 at the end
    :param bool debug: will print debug shapes, etc.
    """
    from TFUtil import prod
    super(DotLayer, self).__init__(**kwargs)
    a_out = self.sources[0].output.copy_as_batch_major()
    b_out = self.sources[1].output.copy_as_batch_major()
    a_reduce_axes = a_out.get_axes_from_description(red1)
    b_reduce_axes = b_out.get_axes_from_description(red2)
    assert a_reduce_axes and b_reduce_axes
    a_var_axes = a_out.get_axes_from_description(var1)
    b_var_axes = b_out.get_axes_from_description(var2)
    assert not set(a_reduce_axes).intersection(a_var_axes)
    assert not set(b_reduce_axes).intersection(b_var_axes)
    a_rem_axes = [i for i in range(a_out.batch_ndim) if i not in a_var_axes + a_reduce_axes]
    b_rem_axes = [i for i in range(b_out.batch_ndim) if i not in b_var_axes + b_reduce_axes]
    transpose_a = bool(a_var_axes and a_reduce_axes[0] < a_var_axes[0])
    transpose_b = bool(b_var_axes and b_reduce_axes[0] > b_var_axes[0])
    # For A, if not transpose_a, we must reorder the axes as: a_rem_axes + a_var_axes + a_reduce_axes.
    # For A, if transpose_a, we must reorder the axes as: a_rem_axes + a_reduce_axes + a_var_axes.
    # For B, if not transpose_b, we must reorder the axes as: b_rem_axes + b_reduce_axes + b_var_axes.
    # For B, if transpose_b, we must reorder the axes as: b_rem_axes + b_var_axes + b_reduce_axes.
    # For matmul, all the first dims must match (batch dim etc), and for the remaining 2 dims,
    # we get (I, J) * (J, K) -> (I, K).
    # So we reshape such that we collapse all reduce-axes and var-axes into each a single axis.
    a = a_out.placeholder
    b = b_out.placeholder
    a_shape = tf.shape(a)
    b_shape = tf.shape(b)
    a_shape = [a_out.batch_shape[i] or a_shape[i] for i in range(a_out.batch_ndim)]
    b_shape = [b_out.batch_shape[i] or b_shape[i] for i in range(b_out.batch_ndim)]
    a_rem_dims = [a_shape[i] for i in a_rem_axes]
    b_rem_dims = [b_shape[i] for i in b_rem_axes]
    assert len(a_rem_axes) == len(b_rem_axes), "remaining shared (batch) axes do not match"
    assert all([
      isinstance(d1, tf.Tensor) or isinstance(d2, tf.Tensor) or d1 == d2
      for (d1, d2) in zip(a_rem_dims, b_rem_dims)])
    a_var_dims = [a_shape[i] for i in a_var_axes]
    b_var_dims = [b_shape[i] for i in b_var_axes]
    a_reduce_dims = [a_shape[i] for i in a_reduce_axes]
    b_reduce_dims = [b_shape[i] for i in b_reduce_axes]
    assert len(a_reduce_axes) == len(b_reduce_axes)
    assert all([
      isinstance(d1, tf.Tensor) or isinstance(d2, tf.Tensor) or d1 == d2
      for (d1, d2) in zip(a_reduce_dims, b_reduce_dims)])
    a_var_dim = prod(a_var_dims)
    b_var_dim = prod(b_var_dims)
    a_reduce_dim = prod(a_reduce_dims)
    b_reduce_dim = prod(b_reduce_dims)
    if debug:
      print("%s, red1=%r, red2=%r, var1=%r, var2=%r:" % (self, red1, red2, var1, var2), file=log.v3)
      print(" ", "a:", a_out, a, file=log.v3)
      print(
        " ", "a_rem_axes:", a_rem_axes, "dims:", a_rem_dims, "a_var_axes:", a_var_axes, "dims:", a_var_dims, a_var_dim,
        "a_reduce_axes:", a_reduce_axes, "dims:", a_reduce_dims, a_reduce_dim, "transpose_a:", transpose_a, file=log.v3)
      print(" ", "b:", b_out, b, file=log.v3)
      print(
        " ", "b_rem_axes:", b_rem_axes, "dims:", b_rem_dims, "b_var_axes:", b_var_axes, "dims:", b_var_dims, b_var_dim,
        "b_reduce_axes:", b_reduce_axes, "dims:", b_reduce_dims, b_reduce_dim, "transpose_b:", transpose_b, file=log.v3)
      with tf.control_dependencies([
            tf.assert_equal(
              a_rem_dims, b_rem_dims, data=[a_shape, b_shape, a_rem_dims, b_rem_dims], summarize=100),
            tf.assert_equal(
              a_reduce_dim, b_reduce_dim, data=[a_shape, b_shape, a_reduce_dims, b_reduce_dims], summarize=100)]):
        a = tf.identity(a)
    if not transpose_a:
      a = tf.transpose(a, a_rem_axes + a_var_axes + a_reduce_axes)
      a = tf.reshape(a, a_rem_dims + [a_var_dim, a_reduce_dim])
    else:
      a = tf.transpose(a, a_rem_axes + a_reduce_axes + a_var_axes)
      a = tf.reshape(a, a_rem_dims + [a_reduce_dim, a_var_dim])
    if not transpose_b:
      b = tf.transpose(b, b_rem_axes + b_reduce_axes + b_var_axes)
      b = tf.reshape(b, b_rem_dims + [b_reduce_dim, b_var_dim])
    else:
      b = tf.transpose(b, b_rem_axes + b_var_axes + b_reduce_axes)
      b = tf.reshape(b, b_rem_dims + [b_var_dim, b_reduce_dim])
    # `res` will be of shape: a_rem_dims + [a_var_dim, b_var_dim]
    res = tf.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)
    if not b_var_dims and add_var2_if_empty:
      b_var_dims.append(1)
      b_var_axes.append(None)
    res = tf.reshape(res, a_rem_dims + a_var_dims + b_var_dims)
    self.output.placeholder = res
    # Collect dynamic size info.
    self.output.size_placeholder = {}
    for axis1_wo_b in sorted(a_out.size_placeholder.keys()):
      axis_out_wb = self._axis1_to_output(axis1_wo_b + 1, a_rem_axes=a_rem_axes, a_var_axes=a_var_axes)
      if axis_out_wb is None:
        continue
      self.output.size_placeholder[axis_out_wb - 1] = a_out.size_placeholder[axis1_wo_b]
    for axis2_wo_b in sorted(b_out.size_placeholder.keys()):
      axis_out_wb = self._axis2_to_output(
        axis2_wo_b + 1, b_rem_axes=b_rem_axes, a_var_axes=a_var_axes, b_var_axes=b_var_axes)
      if axis_out_wb is None or axis_out_wb in self.output.size_placeholder:
        continue
      self.output.size_placeholder[axis_out_wb - 1] = b_out.size_placeholder[axis2_wo_b]

  @staticmethod
  def _axis1_to_output(axis, a_rem_axes, a_var_axes):
    # Output will be of shape a_rem_dims + [a_var_dim, b_var_dim].
    out_axes = a_rem_axes + a_var_axes  # remaining axes do not matter
    if axis not in out_axes:
      return None
    return out_axes.index(axis)

  @staticmethod
  def _axis2_to_output(axis, b_rem_axes, a_var_axes, b_var_axes):
    # Output will be of shape a_rem_dims + [a_var_dim, b_var_dim].
    out_axes = b_rem_axes + [None for _ in a_var_axes] + b_var_axes
    if axis not in out_axes:
      return None
    return out_axes.index(axis)

  @classmethod
  def get_out_data_from_opts(cls, name, sources, red1=-1, red2=-2, var1=-2, var2=-1, add_var2_if_empty=True, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param str|int|tuple[str|int]|list[str|int] red1: reduce axes of first source
    :param str|int|tuple[str|int]|list[str|int] red2: reduce axes of second source
    :param str|int|tuple[str|int]|list[str|int]|None var1: var axes of first source
    :param str|int|tuple[str|int]|list[str|int]|None var2: var axes of second source
    :param bool add_var2_if_empty:
    :rtype: Data
    """
    assert len(sources) == 2, "dot-layer %r: needs exactly two sources" % (name,)
    # See __init__.
    a_out = sources[0].output.copy_as_batch_major()
    b_out = sources[1].output.copy_as_batch_major()
    assert not a_out.beam_size or not b_out.beam_size or a_out.beam_size == b_out.beam_size
    a_reduce_axes = a_out.get_axes_from_description(red1)
    b_reduce_axes = b_out.get_axes_from_description(red2)
    assert a_reduce_axes and b_reduce_axes
    a_var_axes = a_out.get_axes_from_description(var1)
    b_var_axes = b_out.get_axes_from_description(var2)
    assert not set(a_reduce_axes).intersection(a_var_axes)
    assert not set(b_reduce_axes).intersection(b_var_axes)
    a_rem_axes = [i for i in range(a_out.batch_ndim) if i not in a_var_axes + a_reduce_axes]
    b_rem_axes = [i for i in range(b_out.batch_ndim) if i not in b_var_axes + b_reduce_axes]
    a_shape = a_out.batch_shape
    b_shape = b_out.batch_shape
    a_rem_dims = [a_shape[i] for i in a_rem_axes]
    a_var_dims = [a_shape[i] for i in a_var_axes]
    b_var_dims = [b_shape[i] for i in b_var_axes]
    time_dim_axis = None
    if a_out.time_dim_axis is not None:
      time_dim_axis = cls._axis1_to_output(a_out.time_dim_axis, a_rem_axes=a_rem_axes, a_var_axes=a_var_axes)
    if time_dim_axis is None and b_out.time_dim_axis is not None:
      time_dim_axis = cls._axis2_to_output(
        b_out.time_dim_axis, b_rem_axes=b_rem_axes, a_var_axes=a_var_axes, b_var_axes=b_var_axes)
    if time_dim_axis is None and (a_out.time_dim_axis is not None or b_out.time_dim_axis is not None):
      # We had some time dim axis before and reduced it now.
      # But maybe there are others, so let's automatically figure out.
      time_dim_axis = NotSpecified
    if not b_var_dims and add_var2_if_empty:
      b_var_dims.append(1)
    return Data(
      name="%s_output" % name,
      shape=tuple(a_rem_dims[1:] + a_var_dims + b_var_dims),
      batch_dim_axis=0,
      time_dim_axis=time_dim_axis,
      dtype=a_out.dtype,
      beam_size=a_out.beam_size or b_out.beam_size)


class ShiftAxisLayer(_ConcatInputLayer):
  """
  Shifts the dimensions in an axis around.
  This layer may change the axis-dimension.

  This name might be confusing. No axis will be shifted here. See :class:`SwapAxesLayer` for that.
  """
  layer_class = "shift_axis"

  def __init__(self, axis, amount, pad=True, adjust_size_info=True, **kwargs):
    """
    :param str|int axis: single axis to shift
    :param int amount: number of elements to shift
                   (<0 for left-shift, >0 for right-shift)
    :param bool pad: preserve shape by padding
    :param bool adjust_size_info: whether to adjust the size_placeholder
    """
    from TFUtil import single_strided_slice
    import numpy
    super(ShiftAxisLayer, self).__init__(**kwargs)
    assert isinstance(amount, int)
    axis = self.input_data.get_axis_from_description(axis)
    paddings = numpy.zeros(shape=(self.input_data.batch_ndim, 2))
    if amount < 0:  # left-shift
      shifted = single_strided_slice(self.input_data.placeholder, axis=axis, begin=-amount)
      paddings[axis] = [0, -amount]
    elif amount > 0:  # right-shift
      # discard `amount` values in the end of the axis
      shifted = single_strided_slice(self.input_data.placeholder, axis=axis, end=-amount)
      paddings[axis] = [amount, 0]
    else:
      assert False, "amount == 0 equals no operation"
    if pad:
      # insert missing values, so that the shape is preserved
      shifted = tf.pad(shifted, paddings)
    self.output.placeholder = shifted
    self.output.size_placeholder = self.input_data.size_placeholder.copy()
    axis_wob = self.input_data.get_batch_axis_excluding_batch(axis)
    if adjust_size_info and axis_wob in self.output.size_placeholder:
      # Note: Different logic than in get_out_data_from_opts() because this is about e.g. the seq lengths.
      if amount < 0:
        size_delta = amount
      else:  # amount > 0
        if pad:
          size_delta = amount
        else:
          size_delta = 0
      self.output.size_placeholder[axis_wob] = tf.clip_by_value(
        self.output.size_placeholder[axis_wob] + size_delta, 0, tf.shape(shifted)[axis])

  @classmethod
  def get_out_data_from_opts(cls, name, amount, axis, pad, sources=(), **kwargs):
    """
    :param str name:
    :param int amount:
    :param str axis:
    :param bool pad:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    out = get_concat_sources_data_template(sources, name="%s_output" % name)
    assert isinstance(amount, int)
    axis = out.get_axis_from_description(axis)
    axis_wob = out.get_batch_axis_excluding_batch(axis)
    if axis_wob is None:  # batch-axis
      return out  # not storing this information
    if pad:
      return out  # nothing in the shape will change
    if out.shape[axis_wob] is not None:
      shape = list(out.shape)
      shape[axis_wob] -= abs(amount)
      out.shape = tuple(shape)
    if axis == out.feature_dim_axis:
      out.dim = out.shape[axis_wob]
    return out


class ResizeLayer(_ConcatInputLayer):
  """
  Resizes the input, i.e. upsampling or downsampling.
  Supports different kinds, such as linear interpolation or nearest-neighbor.
  """
  layer_class = "resize"

  def __init__(self, factor, axis, kind="nn", fill_value=None, fill_dropout=None, **kwargs):
    """
    :param int factor:
    :param str|int axis: the axis to resize, counted with batch-dim. can also be "T" for time
    :param str kind: "linear", "nn"/"nearest_neighbor", "cubic", "fill"
    :param None|int|float fill_value: if kind=="fill"
    :param float fill_dropout: if set, will dropout in the same axis
    """
    super(ResizeLayer, self).__init__(**kwargs)
    # self.output.shape and self.output.batch_dim_axis are already set here via self.get_out_data_from_opts().
    axis = self.output.get_axis_from_description(axis)
    assert axis > 0, "batch-dim resize not supported"
    self.output.placeholder = self.input_data.copy_as_batch_major().placeholder
    self.output.size_placeholder = self.input_data.size_placeholder.copy()
    if (axis - 1) in self.output.size_placeholder:
      self.output.size_placeholder[axis - 1] *= factor

    # images expected shape: [batch, height, width, channels]
    remaining_axes = [i for i in range(self.output.batch_ndim) if i not in (0, axis)]
    x = dimshuffle(self.output.placeholder, [0, axis, 'x'] + remaining_axes)  # [batch,height,width] + remaining_axes
    shape = tf.shape(self.output.placeholder)
    shape = [shape[i] for i in range(self.output.batch_ndim)]
    remaining_shape = [shape[i] for i in remaining_axes]
    remaining_dim = tf.reduce_prod(remaining_shape) if remaining_axes else 1
    x = tf.reshape(x, [shape[0], shape[axis], 1, remaining_dim])  # [batch,height,width,channels]
    new_size = shape[axis] * factor
    if kind == "linear":
      x = tf.image.resize_bilinear(x, size=(new_size, 1))
    elif kind == "cubic":
      x = tf.image.resize_bicubic(x, size=(new_size, 1))
    elif kind in ["nn", "nearest_neighbor"]:
      x = tf.image.resize_nearest_neighbor(x, size=(new_size, 1))
    elif kind == "fill":
      if self.input_data.sparse:
        assert isinstance(fill_value, int)
        if fill_value < 0:
          fill_value += self.input_data.dim
          assert fill_value > 0
      else:
        assert isinstance(fill_value, (int, float))
      assert isinstance(factor, int) and factor > 1
      from TFUtil import constant_with_shape
      fill_tensor = constant_with_shape(
        fill_value, shape=[shape[0], shape[axis], factor - 1, remaining_dim], dtype=x.dtype)
      x = tf.concat([x, fill_tensor], axis=2)  # [batch,height,factor,channels]
      x.set_shape(tf.TensorShape((None, None, factor, None)))
    else:
      raise Exception("invalid kind %r for resizing" % kind)
    x = tf.reshape(x, [shape[0], new_size] + remaining_shape)  # [batch,new_size] + remaining_shape
    if fill_dropout:
      from TFUtil import expand_dims_unbroadcast
      # We are going to build a mask over the axis. This mask will be shared over all seqs in the batch.
      # Similar as in tf.nn.dropout. Build random_tensor as uniform [keep_prob, 1.0 + keep_prob).
      random_tensor = 1.0 - fill_dropout  # keep_prop
      random_tensor += tf.random_uniform(
        [shape[axis], factor - 1], seed=self.network.random.randint(2**31))  # (old_size, factor - 1)
      # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
      mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
      mask = tf.concat([tf.ones((shape[axis], 1), dtype=tf.bool), mask], axis=1)  # (old_size, factor)
      mask = tf.reshape(mask, (new_size,))  # (new_size,)
      new_size_dropped = tf.reduce_sum(tf.to_int32(mask))
      mask = expand_dims_unbroadcast(mask, axis=0, dim=shape[0])  # (batch,new_size)
      x = tf.boolean_mask(x, mask)  # [batch*new_size_dropped] + remaining_shape
      x = tf.reshape(x, [shape[0], new_size_dropped] + remaining_shape)  # [batch, new_size_dropped] + remaining_shape
      if (axis - 1) in self.output.size_placeholder:
        orig_mask = tf.sequence_mask(
          self.output.size_placeholder[axis - 1], maxlen=new_size, dtype=tf.bool)  # (batch,new_size)
        self.output.size_placeholder[axis - 1] = tf.reduce_sum(tf.to_int32(tf.logical_and(mask, orig_mask)), axis=1)
    if axis != 1:
      perm = [0] + remaining_axes
      perm.insert(axis, 1)
      x = tf.transpose(x, perm)
    self.output.placeholder = x

  @classmethod
  def get_out_data_from_opts(cls, factor, axis, sources, name, **kwargs):
    """
    :param int factor:
    :param str axis:
    :param list[LayerBase] sources:
    :param str name:
    :rtype: Data
    """
    out = get_concat_sources_data_template(sources).copy_as_batch_major()
    out.name = "%s_output" % name
    axis = out.get_axis_from_description(axis)
    assert axis > 0, "batch-dim resize not supported"
    if out.shape[axis - 1] is not None:
      out_shape = list(out.shape)
      out_shape[axis - 1] *= factor
      out.shape = tuple(out_shape)
    return out


class CombineDimsLayer(_ConcatInputLayer):
  """
  Combines multiple dimensions.
  See also :class:`MergeDimsLayer`. This is deprecated in favor of :class:`MergeDimsLayer`.
  """
  layer_class = "combine_dims"

  def __init__(self, axes, **kwargs):
    """
    :param int|list[int]|str axis: one axis or multiple axis to reduce.
      this is counted with batch-dim, which by default is axis 0 (see enforce_batch_dim_axis).
      it also accepts the special tokens "B"|"batch", "spatial", "spatial_except_time", or "F"|"feature"
    """
    super(CombineDimsLayer, self).__init__(**kwargs)
    axes = self.input_data.get_axes_from_description(axes)
    assert len(axes) >= 2
    shape = list(self.input_data.batch_shape)
    assert all([shape[i] for i in axes]), "all axes which are reduced must be defined"
    import numpy
    first_axis = min(axes)
    new_size = numpy.prod([shape[i] for i in axes])
    # self.output.shape should be already set via self.get_out_data_from_opts().
    assert self.output.time_dim_axis_excluding_batch not in axes, "not supported yet"
    # Transpose so that all axes to be combined start at axis `first_axis`.
    perm = list(range(len(shape)))
    for i, j in enumerate(axes):
      perm.pop(j)
      perm.insert(first_axis + i, j)
    x = tf.transpose(self.input_data.placeholder, perm)
    # Now combine the axes via reshaping.
    x_shape = tf.shape(x)
    x = tf.reshape(
      x,
      [x_shape[i] for i in range(first_axis)] +
      [new_size] +
      [x_shape[i] for i in range(first_axis + len(axes), len(shape))])
    self.output.placeholder = x
    self.output.size_placeholder = self.input_data.size_placeholder.copy()
    for i in axes:
      assert self.output.get_batch_axis_excluding_batch(i) not in self.output.size_placeholder

  @classmethod
  def get_out_data_from_opts(cls, axes, sources, **kwargs):
    """
    :param str|list[str] axes:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    out = get_concat_sources_data_template(sources)
    axes = out.get_axes_from_description(axes)
    assert len(axes) >= 2
    axes = sorted(axes)
    shape = list(out.batch_shape)
    assert all([shape[i] for i in axes]), "all axes which are reduced must be defined"
    import numpy
    shape[min(axes)] = numpy.prod([shape[i] for i in axes])
    for i in reversed(sorted(axes)[1:]):
      del shape[i]
    out.shape = tuple(shape[:out.batch_dim_axis] + shape[out.batch_dim_axis + 1:])
    out.dim = shape[-1]
    return out


class RemoveLayer(LayerBase):
  """
  Currently, assumes sparse data, and removes a specific symbol from the data.
  """
  layer_class = "remove"

  def __init__(self, symbol, **kwargs):
    """
    :param int symbol:
    """
    super(RemoveLayer, self).__init__(**kwargs)
    if symbol < 0:
      symbol += self.output.dim
      assert symbol > 0

    # I currently do not have a good idea how to make this efficient.
    in_data = self.sources[0].output.copy_as_batch_major()
    assert in_data.sparse
    assert in_data.batch_ndim == 2
    in_seqs = in_data.placeholder  # (batch,time)
    in_mask = tf.logical_and(tf.not_equal(in_seqs, symbol), in_data.get_sequence_mask_broadcast())  # (batch,time)
    out_seq_lens = tf.count_nonzero(in_mask, axis=1, dtype=tf.int32)  # (batch,)
    max_out_seq_len = tf.reduce_max(out_seq_lens)  # scalar
    from TFUtil import constant_with_shape
    zero_seq = constant_with_shape(0, shape=[max_out_seq_len], dtype=in_seqs.dtype)

    def body(args):
      """
      :param (tf.Tensor,tf.Tensor) args: seq, mask; both (time,)
      :return: out seq
      :rtype: tf.Tensor
      """
      seq, mask = args
      out_seq = tf.boolean_mask(seq, mask)
      out_seq = tf.concat([out_seq, zero_seq], axis=0)
      return out_seq[:max_out_seq_len]

    out_seqs = tf.map_fn(body, [in_seqs, in_mask], dtype=in_seqs.dtype)
    self.output.placeholder = out_seqs
    self.output.size_placeholder[0] = out_seq_lens

  @classmethod
  def get_out_data_from_opts(cls, name, sources=(), **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    assert len(sources) == 1, "%s layer %r: must have exactly one source" % (cls, name)
    assert sources[0].output.sparse, "%s layer %r: assumes sparse data" % (cls, name)
    out = sources[0].output.copy(name="%s_output" % name).copy_as_batch_major()
    out.shape = (None,) + out.shape[1:]  # must be dynamic
    return out


class CombineLayer(LayerBase):
  """
  Applies some binary operation on all sources, such as addition.
  Also see :class:`ActivationLayer`.
  """
  layer_class = "combine"

  # noinspection PyShadowingBuiltins
  def __init__(self, kind, sources, activation=None, with_bias=False,
               eval=None, eval_locals=None, eval_for_output_loss=False,
               **kwargs):
    """
    :param str kind: e.g. "average" or "add", or "eval"
    :param list[LayerBase] sources:
    :param str|None activation: if provided, activation function to apply, e.g. "tanh" or "relu"
    :param bool with_bias: if given, will add a bias
    :param str eval: for kind="eval", will eval this string. see :func:`_op_kind_eval`
    :param dict[str]|None eval_locals: locals for eval
    :param bool eval_for_output_loss: will do the same eval on layer.output_loss
    """
    assert sources
    super(CombineLayer, self).__init__(sources=sources, **kwargs)
    assert kind in ["average", "add", "sub", "mul", "eval"], (
      "%s: Invalid `kind` %r for this layer." % (self, kind))
    op = self._get_op(kind=kind, eval_str=eval, eval_locals=eval_locals)
    x = op(sources)
    if eval_for_output_loss:
      assert eval
      assert all([layer.output_loss is not None for layer in sources])
      self.output_loss = self._op_kind_eval(
        sources=[layer.output_loss for layer in sources],
        eval_str=eval, eval_locals=eval_locals)
    if with_bias:
      with self.var_creation_scope():
        b = self.add_param(tf.get_variable(
          name="b", shape=(self.output.dim,),
          initializer=tf.constant_initializer(value=0, dtype=tf.float32)))
      x += b
    if activation:
      from TFUtil import get_activation_function
      act_func = get_activation_function(activation)
      self.output_before_activation = OutputWithActivation(x, act_func=act_func)
    else:
      self.output_before_activation = OutputWithActivation(x)
    x = self.output_before_activation.y
    self.output.placeholder = x

  @classmethod
  def get_out_data_from_opts(cls, n_out=NotSpecified, out_type=None, sources=(), **kwargs):
    """
    :param int|None|NotSpecified n_out:
    :param dict[str]|None out_type:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    out_type_ = {}
    if sources:
      out_type_.update(Data.get_common_data([s.output for s in sources]).get_kwargs())
    if n_out is not NotSpecified:
      out_type_["dim"] = n_out
    out_type_["name"] = "%s_output" % kwargs["name"]
    if out_type:
      if isinstance(out_type, dict):
        out_type_.update(out_type)
      elif callable(out_type):
        out_type_ = out_type  # just overwrite
      else:
        raise TypeError("unexpected type of out_type %r" % (out_type,))
    return super(CombineLayer, cls).get_out_data_from_opts(n_out=n_out, out_type=out_type_, sources=sources, **kwargs)

  def _check_same_dense_dim(self, sources):
    """
    :param list[LayerBase] sources:
    """
    assert not self.output.sparse
    for source in sources:
      assert not source.output.sparse
      assert (source.output.dim == self.output.dim
              or source.output.dim == 1)  # constant layer broadcasting

  # Requires the same input shape and yield the same output shape.
  def _op_dense_fn(self, sources, fn):
    """
    :param list[LayerBase] sources:
    :param ((x1,x2) -> y) fn: function to perform on x1 and x2
    :rtype: tf.Tensor
    """
    self._check_same_dense_dim(sources)
    common_data = Data.get_common_data([s.output for s in sources])
    x = sources[0].output.copy_compatible_to(common_data).placeholder
    for source in sources[1:]:
      x2 = source.output.copy_compatible_to(common_data).placeholder
      x = fn(x, x2)
    return x

  def _op_kind_add(self, sources):
    """
    :param list[LayerBase] sources:
    :rtype: tf.Tensor
    """
    return self._op_dense_fn(sources, tf.add)

  def _op_kind_sub(self, sources):
    """
    :param list[LayerBase] sources:
    :rtype: tf.Tensor
    """
    return self._op_dense_fn(sources, tf.subtract)

  def _op_kind_mul(self, sources):
    """
    :param list[LayerBase] sources:
    :rtype: tf.Tensor
    """
    return self._op_dense_fn(sources, tf.multiply)

  def _op_kind_average(self, sources):
    """
    :param list[LayerBase] sources:
    :rtype: tf.Tensor
    """
    x = self._op_kind_add(sources)
    x /= len(sources)
    return x

  def _op_kind_eval(self, sources, eval_str, eval_locals=None):
    """
    :param list[LayerBase]|list[tf.Tensor] sources:
    :param str eval_str:
    :param dict[str]|None eval_locals:
    :rtype: tf.Tensor
    """
    used_sources = set()  # type: typing.Set[int]
    common_data = Data.get_common_data([s.output for s in sources])

    def source(i, auto_convert=True, enforce_batch_major=False, as_data=False):
      """
      :param int i: layer index
      :param bool auto_convert:
      :param bool enforce_batch_major: if True, return as batch-major
      :param bool as_data: if True, return the Data object
      :return: output placeholder from source i, compatible to source 0
      :rtype: tf.Tensor|Data
      """
      assert 0 <= i < len(sources)
      used_sources.add(i)
      if isinstance(sources[i], LayerBase):
        output = sources[i].output
        if auto_convert:
          output = output.copy_compatible_to(common_data)
        if enforce_batch_major:
          output = output.copy_as_batch_major()
        if as_data:
          return output
        return output.placeholder
      assert not as_data
      return sources[i]

    vs = vars(TFUtil).copy()
    vs.update({"tf": tf, "source": source, "self": self})
    vs.update(eval_locals or {})
    x = eval(eval_str, vs)
    assert sorted(used_sources) == list(range(len(sources))), (
      "not used sources: %r" % set(range(len(sources))).difference(used_sources))
    return x

  def _get_op(self, kind, eval_str=None, eval_locals=None):
    """
    :param str kind:
    :param str eval_str:
    :param dict[str]|None eval_locals:
    :rtype: (list[LayerBase]) -> tf.Tensor
    """
    op = getattr(self, "_op_kind_%s" % kind)
    if eval_str:
      assert kind == "eval"

      def wrap_eval_op(sources):
        """
        :param list[LayerBase] sources:
        :rtype: tf.Tensor
        """
        return self._op_kind_eval(sources, eval_str=eval_str, eval_locals=eval_locals)
      op = wrap_eval_op
    return op


class EvalLayer(CombineLayer):
  """
  Evaluates some string.
  The :class:`CombineLayer` provides this functionality, thus this is just a special case of it.
  Also see :class:`ActivationLayer`.
  """
  layer_class = "eval"

  # noinspection PyShadowingBuiltins
  def __init__(self, eval, **kwargs):
    """
    :param str eval: will eval this string. see :func:`_op_kind_eval`
    """
    super(EvalLayer, self).__init__(kind="eval", eval=eval, **kwargs)


class CompareLayer(LayerBase):
  """
  Compares (e.g. equality check) all the sources element-wise.
  """
  layer_class = "compare"

  def __init__(self, kind="equal", value=None, **kwargs):
    """
    :param str kind: e.g. "equal"
    :param float|int|None value: if specified, will also compare to this
    """
    super(CompareLayer, self).__init__(**kwargs)
    assert len(self.sources) >= 1
    if value is None:
      assert len(self.sources) >= 2
    op = getattr(tf, kind)  # e.g. tf.equal
    from TFUtil import swapaxes
    x = self.sources[0].output.placeholder
    batch_axis = self.sources[0].output.batch_dim_axis
    r_last = None
    if value is not None:
      r_last = op(x, value)
    for source in self.sources[1:]:
      x2 = source.output.placeholder
      if source.output.batch_dim_axis != batch_axis:
        x2 = swapaxes(x2, batch_axis, source.output.batch_dim_axis)
      r = op(x, x2)
      if r_last is not None:
        r = tf.logical_and(r_last, r)
      r_last = r
    self.output.placeholder = r_last

  @classmethod
  def get_out_data_from_opts(cls, n_out=NotSpecified, out_type=None, sources=(), **kwargs):
    """
    :param int|None|NotSpecified n_out:
    :param dict[str]|None out_type:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    if n_out is NotSpecified and not out_type:
      out_type = sources[0].output.get_kwargs()
      out_type["name"] = "%s_output" % kwargs["name"]
      if out_type.get("sparse", False):
        out_type["dim"] = 2  # True or False
      out_type["dtype"] = "bool"
      out_type["vocab"] = None
    return super(CompareLayer, cls).get_out_data_from_opts(n_out=n_out, out_type=out_type, sources=sources, **kwargs)


class SwitchLayer(LayerBase):
  """
  Wrapper around tf.where().
  Uses three inputs: condition, true_from and false_from.
  The output of this layer contains elements of true_from
  where condition is True, otherwise elements of false_from.
  condition has to be of dtype bool.
  true_from and false_from must have the same shape.
  """

  layer_class = "switch"

  def __init__(self, condition, true_from, false_from, **kwargs):
    """
    :param LayerBase condition:
    :param LayerBase true_from:
    :param LayerBase false_from:
    """
    super(SwitchLayer, self).__init__(**kwargs)

    self.condition = condition
    self.true_from = true_from
    self.false_from = false_from

    assert condition.output.dtype == "bool"
    assert true_from.output.shape == false_from.output.shape

    self.output.placeholder = tf.where(
      condition=condition.output.placeholder, x=true_from.output.placeholder, y=false_from.output.placeholder)

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param TFNetwork.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    d.setdefault("from", [])

    super(SwitchLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)

    d["condition"] = get_layer(d["condition"])
    d["true_from"] = get_layer(d["true_from"])
    d["false_from"] = get_layer(d["false_from"])

  @classmethod
  def get_out_data_from_opts(cls, true_from, name, **kwargs):
    """
    :param LayerBase true_from:
    :param str name:
    :rtype: Data
    """
    return true_from.output.copy(name="%s_output" % name)

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    return [self.condition, self.true_from, self.false_from]


class SubnetworkLayer(LayerBase):
  """
  You can define a whole subnetwork as a single layer by this class.

  The subnetwork will be specified by a ``dict[str,dict[str]]``, just like
  a normal network is specified in the config.

  The ``"output"`` layer of the subnetwork will be the output of this
  subnetwork-layer.

  With ``concat_sources=True`` (default),
    the input to this layer will be represented as the ``"data:data"`` or simply ``"data"``
    in the subnetwork,
  otherwise with ``concat_sources=False``,
    the input to this layer will be represented as ``"data:input_layer_name"``
    for each input, in the subnetwork.
  """

  layer_class = "subnetwork"
  recurrent = True  # we don't know. depends on the subnetwork.

  # noinspection PyShadowingNames
  def __init__(self, subnetwork, concat_sources=True, load_on_init=None, dropout=0, dropout_noise_shape=None, **kwargs):
    """
    :param dict[str,dict] subnetwork: subnetwork as dict (JSON content). must have an "output" layer-
    :param bool concat_sources: if we concatenate all sources into one, like it is standard for most other layers
    :param str|None load_on_init: if provided, for parameter initialization,
      we will load the given model file.
    :param float dropout: will be applied if train_flag is set
    :param tuple|list|dict|None dropout_noise_shape:
    """
    super(SubnetworkLayer, self).__init__(**kwargs)
    from TFNetwork import TFNetwork
    sub_extern_data = self._get_subnet_extern_data(
      base_network=self.network,
      sources=self.sources, concat_sources=concat_sources,
      dropout=dropout, dropout_noise_shape=dropout_noise_shape)
    net = TFNetwork(
      name="%s/%s:subnet" % (self.network.name, self.name),
      extern_data=sub_extern_data,
      train_flag=self.network.train_flag,
      search_flag=self.network.search_flag,
      parent_layer=self)
    if self._rec_previous_layer:
      # Make some rec_previous_layer for the subnet layers.
      subnetwork = subnetwork.copy()
      for layer_name in list(subnetwork.keys()):
        # The actual layer is not so important.
        # In some cases (e.g. RnnCellLayer), we just want rec_vars_outputs.
        dummy_rec_previous_layer = InternalLayer(
          name=layer_name, network=net, output=Data(name="dummy_rec_previous_layer(%s)" % layer_name, dim=1))
        dummy_rec_previous_layer.rec_vars_outputs.update({
          key[len(layer_name + "/"):]: value
          for (key, value) in self._rec_previous_layer.rec_vars_outputs.items()
          if key.startswith(layer_name + "/")})
        subnetwork[layer_name] = subnetwork[layer_name].copy()
        subnetwork[layer_name]["rec_previous_layer"] = dummy_rec_previous_layer
    net.construct_from_dict(subnetwork)
    self.subnetwork = net
    if self.network.eval_flag:
      self.subnetwork.maybe_construct_objective()
    self.output = net.get_default_output_layer().output
    # See _get_subnet_extern_data for which data keys we expect to be used.
    if concat_sources:
      assert sub_extern_data.default_input in self.subnetwork.used_data_keys, "%s: inputs are not used" % self
    else:
      for source in self.sources:
        assert source.name in self.subnetwork.used_data_keys, "%s: some input is not used" % self
    self._update_used_data_keys()
    for layer in net.layers.values():
      if layer.params:
        assert layer.trainable == self.trainable, "partly trainable subnetworks not yet supported"
      self.params.update({"%s/%s" % (layer.name, k): v for (k, v) in layer.params.items()})
      self.rec_vars_outputs.update({"%s/%s" % (layer.name, k): v for (k, v) in layer.rec_vars_outputs.items()})
    if load_on_init:
      print("loading initial weights from", load_on_init, file=log.v2)
      self_prefix = self.get_absolute_name_scope_prefix()  # with "/" at end
      from TFNetwork import CustomCheckpointLoader
      loader = CustomCheckpointLoader(
        filename=load_on_init, saveable_params=list(self.params.values()), params_prefix=self_prefix, network=net)
      loader.set_as_custom_init()

  # noinspection PyShadowingNames
  @classmethod
  def _get_subnet_extern_data(cls, base_network, sources, concat_sources, dropout=0, dropout_noise_shape=None):
    """
    :param TFNetwork.TFNetwork base_network:
    :param list[LayerBase] sources:
    :param bool concat_sources:
    :param float dropout: will be applied if train_flag is set
    :param tuple|list|dict|None dropout_noise_shape:
    :rtype: TFNetwork.ExternData
    """
    from TFNetwork import ExternData
    sub_extern_data = ExternData()
    if concat_sources:
      sub_extern_data.data[sub_extern_data.default_input] = (
        concat_sources_with_opt_dropout(
          sources, dropout=dropout, dropout_noise_shape=dropout_noise_shape))
    else:
      assert not dropout, "not implemented without concat_sources"
      for source in sources:
        assert isinstance(source, LayerBase)
        sub_extern_data.data[source.name] = source.output
    # Copy data (e.g. target classes) over from base network.
    for key, data in base_network.extern_data.data.items():
      sub_extern_data.data.setdefault(key, data)
    return sub_extern_data

  def _update_used_data_keys(self):
    # Maybe update self.network.used_data_keys.
    for key in self.subnetwork.used_data_keys:
      if self.subnetwork.extern_data.data[key] is self.network.extern_data.data.get(key, None):
        self.network.used_data_keys.add(key)

  # noinspection PyShadowingNames
  @classmethod
  def get_out_data_from_opts(cls, subnetwork, concat_sources=True, n_out=NotSpecified, out_type=None, **kwargs):
    """
    :param dict[str,dict[str]] subnetwork:
    :param bool concat_sources:
    :param int|None|NotSpecified n_out:
    :param dict[str]|None out_type:
    :rtype: Data
    """
    if n_out is not NotSpecified or out_type:
      return super(SubnetworkLayer, cls).get_out_data_from_opts(n_out=n_out, out_type=out_type, **kwargs)
    subnet = cls._construct_template_subnet(
      name=kwargs["name"], network=kwargs["network"],
      subnetwork=subnetwork,
      concat_sources=concat_sources, sources=kwargs["sources"])
    return subnet.layers["output"].output

  # noinspection PyShadowingNames
  @classmethod
  def _construct_template_subnet(cls, name, network, subnetwork, sources, concat_sources=True):
    """
    Very similar to _SubnetworkRecCell._construct_template, but simpler.

    :param str name:
    :param TFNetwork.TFNetwork network: parent net
    :param dict[str,dict[str]] subnetwork:
    :param list[LayerBase] sources:
    :param bool concat_sources:
    :rtype: TFNetwork.TFNetwork
    """
    assert "output" in subnetwork
    from TFNetwork import TFNetwork
    from TFNetworkRecLayer import _TemplateLayer
    # Placeholder, will not be used.
    sub_extern_data = cls._get_subnet_extern_data(
      base_network=network, sources=sources, concat_sources=concat_sources)
    subnet = TFNetwork(
      name="%s/%s:subnet" % (network.name, name),
      extern_data=sub_extern_data,
      train_flag=network.train_flag,
      search_flag=network.search_flag,
      parent_net=network,
      rnd_seed=0)  # seed 0, will not be used, as we only construct the templates

    # noinspection PyShadowingNames
    def add_templated_layer(name, layer_class, **layer_desc):
      """
      :param str name:
      :param type[LayerBase]|LayerBase layer_class:
      :param dict[str] layer_desc:
      :rtype: LayerBase
      """
      layer_ = _TemplateLayer(name=name, network=subnet)
      subnet.layers[name] = layer_
      layer_desc = layer_desc.copy()
      layer_desc["name"] = name
      layer_desc["network"] = subnet
      output = layer_class.get_out_data_from_opts(**layer_desc)
      layer_.init(layer_class=layer_class, output=output, **layer_desc)
      if layer_class.recurrent:
        subnet.recurrent = True
      return layer_

    # noinspection PyShadowingNames
    def get_templated_layer(name):
      """
      :param str name:
      :rtype: _TemplateLayer|LayerBase
      """
      if name in subnet.layers:
        layer_ = subnet.layers[name]
        return layer_
      if name.startswith("base:"):
        layer_ = network.get_layer(name[len("base:"):])
        return layer_
      return subnet.construct_layer(
        net_dict=subnetwork, name=name, get_layer=get_templated_layer, add_layer=add_templated_layer)

    try:
      get_templated_layer("output")
      assert "output" in subnet.layers

      for layer_name, layer in subnetwork.items():
        if network.eval_flag and layer.get("loss"):  # only collect losses if we need them
          get_templated_layer(layer_name)
      for layer_name, layer in subnetwork.items():
        if layer.get("is_output_layer"):
          get_templated_layer(layer_name)

    except Exception:
      print("%r: exception constructing template network (for deps and data shapes)" % cls)
      print("Template network so far:")
      from pprint import pprint
      pprint(subnet.layers)
      raise

    return subnet

  def get_constraints_value(self):
    """
    :rtype: tf.Tensor|None
    """
    v = self.subnetwork.get_total_constraints()
    if v is 0:
      return None
    return v

  @classmethod
  def get_losses(cls, name, network, output, loss=None, reduce_func=None, layer=None, **kwargs):
    """
    :param str name: layer name
    :param TFNetwork.TFNetwork network:
    :param Loss|None loss: argument just as for __init__
    :param Data output: the output (template) for the layer
    :param LayerBase|None layer:
    :param ((tf.Tensor)->tf.Tensor)|None reduce_func:
    :param kwargs: other layer kwargs
    :rtype: list[TFNetwork.LossHolder]
    """
    from TFNetwork import LossHolder
    from TFNetworkRecLayer import _TemplateLayer
    losses = super(SubnetworkLayer, cls).get_losses(
      name=name, network=network, output=output, loss=loss, layer=layer, reduce_func=reduce_func, **kwargs)
    if layer:
      assert isinstance(layer, SubnetworkLayer)
      subnet = layer.subnetwork
    else:
      subnet = cls._construct_template_subnet(
        name=name, network=network, subnetwork=kwargs["subnetwork"],
        sources=kwargs["sources"], concat_sources=kwargs["concat_sources"])
    for layer_name, sub_layer in sorted(subnet.layers.items()):
      if layer:
        assert isinstance(layer, SubnetworkLayer)
        sub_layer_class = sub_layer.__class__
        real_sub_layer = sub_layer
      else:
        real_sub_layer = None
        assert isinstance(sub_layer, _TemplateLayer)
        assert issubclass(sub_layer.layer_class_type, LayerBase)
        sub_layer_class = sub_layer.layer_class_type
      for loss in sub_layer_class.get_losses(reduce_func=reduce_func, layer=real_sub_layer, **sub_layer.kwargs):
        assert isinstance(loss, LossHolder)
        losses.append(loss.copy_new_base(
          network=network, name="%s/%s" % (name, loss.name)))
    return losses

  def get_last_hidden_state(self, key):
    """
    :param int|str|None key: also the special key "*"
    :rtype: tf.Tensor|None
    """
    h = self.subnetwork.get_default_output_layer().get_last_hidden_state(key=key)
    if h is not None:
      return h
    return super(SubnetworkLayer, self).get_last_hidden_state(key=key)

  # noinspection PyMethodOverriding
  @classmethod
  def get_rec_initial_extra_outputs(cls, batch_dim, rec_layer, subnetwork, **kwargs):
    """
    :param tf.Tensor batch_dim: for this layer, might be with beam
    :param TFNetworkRecLayer.RecLayer rec_layer:
    :param dict[str,dict[str]] subnetwork:
    :rtype: dict[str,tf.Tensor]
    """
    extra_outputs = {}
    for layer_name, layer_desc in subnetwork.items():
      layer_desc = layer_desc.copy()
      layer_class_name = layer_desc.pop("class")
      cl = get_layer_class(layer_class_name)
      # Note: This is not totally correct. We should call transform_config_dict.
      # But that will make it quite complicated...
      layer_desc["name"] = layer_name
      assert issubclass(cl, LayerBase)
      with cl.cls_layer_scope(layer_name):
        d = cl.get_rec_initial_extra_outputs(
          batch_dim=batch_dim, rec_layer=rec_layer, **layer_desc)
        for key, value in d.items():
          extra_outputs["%s/%s" % (layer_name, key)] = value
    return extra_outputs

  @classmethod
  def get_rec_initial_extra_outputs_shape_invariants(cls, subnetwork, **kwargs):
    """
    :param dict[str,dict[str]] subnetwork:
    :return: optional shapes for the tensors by get_rec_initial_extra_outputs
    :rtype: dict[str,tf.TensorShape]
    """
    # Very similar to get_rec_initial_extra_outputs.
    shape_invariants = {}
    for layer_name, layer_desc in subnetwork.items():
      layer_desc = layer_desc.copy()
      layer_class_name = layer_desc.pop("class")
      cl = get_layer_class(layer_class_name)
      # Note: This is not totally correct. We should call transform_config_dict.
      # But that will make it quite complicated...
      layer_desc["name"] = layer_name
      assert issubclass(cl, LayerBase)
      with cl.cls_layer_scope(layer_name):
        d = cl.get_rec_initial_extra_outputs_shape_invariants(**layer_desc)
        for key, value in d.items():
          shape_invariants["%s/%s" % (layer_name, key)] = value
    return shape_invariants


class VariableLayer(LayerBase):
  """
  Represents a variable. Can add batch/time dimension if wanted. Can be trainable.
  See defaults.
  """
  layer_class = "variable"

  def __init__(self, shape, dtype="float32", add_batch_axis=True, add_time_axis=False, trainable=True,
               init=0,
               **kwargs):
    """
    :param tuple[int]|list[int] shape:
    :param str dtype:
    :param bool add_batch_axis:
    :param bool add_time_axis:
    :param bool trainable:
    :param str|float|int init: see :func:`TFUtil.get_initializer`
    """
    super(VariableLayer, self).__init__(trainable=trainable, **kwargs)
    assert not self.sources, "%s: does not expect any sources" % self
    from TFUtil import get_initializer, expand_dims_unbroadcast
    initializer = get_initializer(init, seed=self.network.random.randint(2 ** 31), eval_local_ns={"layer": self})
    with self.var_creation_scope():
      var = self.add_param(tf.get_variable(
        name=self.name, shape=shape, dtype=dtype,
        initializer=initializer, trainable=trainable
      ))
      out = var
      if add_batch_axis:
        # Unbroadcast to not confuse some other layers
        batch_dim = self.get_batch_dim()
        out = expand_dims_unbroadcast(out, axis=self.output.batch_dim_axis, dim=batch_dim)
      if add_time_axis:
        out = tf.expand_dims(out, axis=self.output.time_dim_axis)
    self.output.placeholder = out

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param TFNetwork.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    # Overwrite default behavior for default sources.
    # Here: none by default.
    d.setdefault("from", [])
    super(VariableLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)

  @classmethod
  def get_out_data_from_opts(cls, name, shape, dtype="float32", add_batch_axis=True, add_time_axis=False, **kwargs):
    """
    :param str name:
    :param tuple[int]|list[int] shape:
    :param str dtype:
    :param bool add_batch_axis:
    :param bool add_time_axis:
    :rtype: Data
    """
    assert isinstance(shape, (list, tuple))
    assert len(shape) == 0 or all(shape)
    shape = list(shape)
    batch_dim_axis = 0 if add_batch_axis else None
    if add_time_axis:
      shape.insert(0, 1)
      if add_batch_axis:
        time_dim_axis = 1
      else:
        time_dim_axis = 0
    else:
      time_dim_axis = None
    return Data(
      name="%s_output" % name, shape=shape, dtype=dtype,
      dim=shape[-1] if shape else None,
      batch_dim_axis=batch_dim_axis, time_dim_axis=time_dim_axis)


class AccumulateMeanLayer(ReduceLayer):
  """
  Accumulates the mean of the input (in training) (over batch-dim and time-dim by default).
  It's similar to :class:`ReduceLayer`
  """
  layer_class = "accumulate_mean"

  def __init__(self, exp_average, axes="bt", initial_value=None, is_prob_distribution=None, **kwargs):
    """
    :param float exp_average: momentum in exponential average calculation
    :param int|list[str]|str axes: the axes to reduce. must contain batch and time.
    :param float initial_value: how to initialize the variable which accumulates the mean
    :param bool is_prob_distribution: if provided, better default for initial_value
    """
    super(AccumulateMeanLayer, self).__init__(mode="mean", keep_dims=False, axes=axes, **kwargs)
    assert self.output.batch_dim_axis is None
    assert all(self.output.batch_shape), "shape must be fixed. input: %r" % self.input_data
    shape = self.output.batch_shape

    if is_prob_distribution:
      assert len(shape) == 1
      if initial_value is None:
        initial_value = 1.0 / shape[0]
    if initial_value is not None:
      initial_value = tf.ones(shape) * initial_value
    else:
      initial_value = tf.zeros(shape)
    from TFUtil import CustomUpdateExpAverage
    v = self.add_param(
      tf.Variable(initial_value, name="mean"),
      custom_update=CustomUpdateExpAverage(average=self.output.placeholder, alpha=exp_average))
    self.output.placeholder = v

  @classmethod
  def get_out_data_from_opts(cls, axes="bt", **kwargs):
    """
    :param str axes:
    :rtype: Data
    """
    return super(AccumulateMeanLayer, cls).get_out_data_from_opts(axes=axes, **kwargs)


class FastBaumWelchLayer(_ConcatInputLayer):
  """
  Calls :func:`fast_baum_welch` or :func:`fast_baum_welch_by_sprint_automata`.
  We expect that our input are +log scores, e.g. use log-softmax.
  """
  layer_class = "fast_bw"
  recurrent = True

  def __init__(self, align_target, sprint_opts=None,
               input_type="log_prob",
               tdp_scale=1.0, am_scale=1.0, min_prob=0.0,
               staircase_seq_len_source=None,
               **kwargs):
    """
    :param str align_target: e.g. "sprint" or "staircase"
    :param dict[str] sprint_opts:
    :param str input_type: "log_prob" or "prob"
    :param float tdp_scale:
    :param float am_scale:
    :param float min_prob: clips the minimum prob (value in [0,1])
    :param LayerBase|None staircase_seq_len_source:
    """
    import numpy
    super(FastBaumWelchLayer, self).__init__(**kwargs)
    data = self.input_data.copy_as_time_major()
    # We want the scores in -log space.
    if input_type == "log_prob":
      am_scores = -data.placeholder
    elif input_type == "prob":
      if len(self.sources) == 1 and self.sources[0].output_before_activation:
        am_scores = -self.sources[0].output_before_activation.get_log_output()
        if self.sources[0].output.is_batch_major:
          from TFUtil import swapaxes
          am_scores = swapaxes(am_scores, 0, 1)
      else:
        from TFUtil import safe_log
        am_scores = -safe_log(data.placeholder)
    else:
      raise Exception("%s: invalid input_type %r" % (self, input_type))
    if min_prob > 0:
      am_scores = tf.minimum(am_scores, -numpy.log(min_prob))  # in -log space
    if am_scale != 1:
      am_scores *= am_scale
    if align_target == "sprint":
      from TFUtil import sequence_mask_time_major
      seq_mask = sequence_mask_time_major(data.get_sequence_lengths())
      from TFNativeOp import fast_baum_welch_by_sprint_automata
      seq_tags = self.network.get_seq_tags()
      fwdbwd, obs_scores = fast_baum_welch_by_sprint_automata(
        sprint_opts=sprint_opts,
        tdp_scale=tdp_scale,
        am_scores=am_scores,
        float_idx=seq_mask,
        tags=seq_tags)
    elif align_target == "staircase":
      from TFNativeOp import fast_baum_welch_staircase
      fwdbwd, obs_scores = fast_baum_welch_staircase(
        am_scores=am_scores, seq_lens=staircase_seq_len_source.output.get_sequence_lengths())
    else:
      raise Exception("%s: invalid align_target %r" % (self, align_target))
    loss = tf.reduce_sum(obs_scores[0])
    self.output_loss = loss
    bw = tf.exp(-fwdbwd)
    self.output.placeholder = bw
    self.output.size_placeholder = data.size_placeholder.copy()

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param TFNetwork.TFNetwork network:
    :param get_layer:
    """
    super(FastBaumWelchLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    if d.get("staircase_seq_len_source"):
      d["staircase_seq_len_source"] = get_layer(d["staircase_seq_len_source"])

  @classmethod
  def get_out_data_from_opts(cls, name, sources, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    return get_concat_sources_data_template(sources, name="%s_output" % name).copy_as_time_major()


class SyntheticGradientLayer(_ConcatInputLayer):
  """
  This is a generalized way to be able to replace the true gradient with any kind of predicted gradient.
  This enabled to implement the idea from here:
    Decoupled Neural Interfaces using Synthetic Gradients, https://arxiv.org/abs/1608.05343
  """
  layer_class = "synthetic_gradient"

  def __init__(self, gradient, meta_loss_scale=1.0, **kwargs):
    """
    :param LayerBase gradient:
    :param float meta_loss_scale:
    """
    super(SyntheticGradientLayer, self).__init__(**kwargs)
    from TFUtil import MetaLosses
    self.output.placeholder = MetaLosses.synthetic_gradient(
      x=self.input_data.placeholder,
      synthetic_grad_x=gradient.output.copy_compatible_to(self.input_data).placeholder,
      loss_name=self.get_absolute_name_scope_prefix() + "synthetic_grad",
      loss_source=self,
      loss_scale=meta_loss_scale)
    self.output.size_placeholder = self.input_data.size_placeholder.copy()

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param TFNetwork.TFNetwork network:
    :param get_layer:
    """
    super(SyntheticGradientLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["gradient"] = get_layer(d["gradient"])

  @classmethod
  def get_out_data_from_opts(cls, sources, name, **kwargs):
    """
    :param list[LayerBase] sources:
    :param str name:
    :rtype: Data
    """
    return get_concat_sources_data_template(sources, name="%s_output" % name)


class TikhonovRegularizationLayer(CopyLayer):
  """
  Adds the Tikhonov regularization as a meta-loss (see :class:`TFUtil.MetaLosses`).
  """
  layer_class = "tikhonov_regularization"

  def __init__(self, meta_loss_scale=1.0, **kwargs):
    """
    :param float meta_loss_scale:
    """
    super(TikhonovRegularizationLayer, self).__init__(**kwargs)
    with self.var_creation_scope():
      dummy_var = self.add_param(tf.get_variable(name="dummy", shape=(), dtype=tf.float32))
    from TFUtil import MetaLosses
    self.output.placeholder = MetaLosses.tikhonov_regularized(
      x=self.input_data.placeholder, dummy=dummy_var,
      loss_name=self.get_absolute_name_scope_prefix() + "tikhonov_reg",
      loss_source=self,
      loss_scale=meta_loss_scale)
    self.output.size_placeholder = self.input_data.size_placeholder.copy()


class AllophoneStateIdxParserLayer(LayerBase):
  """
  This is very much Sprint/RASR specific.
  We get allophone state indices and return (center, left_1, right_1, ..., state, boundary).
  The index is defined by NoTyingDense (ClassicStateTying.cc).
  In the Sprint config, this is via option --*.state-tying.type=no-tying-dense.
  """
  layer_class = "allophone_state_idx_parser"
  NumBoundaryClasses = 4  # 0: none, 1: start (@i), 2: end (@f), 3: start+end (@i@f)

  def __init__(self, num_phone_classes, num_states=3, context_len=1, **kwargs):
    """
    :param list[LayerBase] sources:
    :param int num_phone_classes: number of phonemes + 1, with special 0 phone == no context
    :param int num_states: number of HMM states
    :param int context_len: left/right context len
    """
    super(AllophoneStateIdxParserLayer, self).__init__(**kwargs)
    result = [None] * self.output.dim  # type: typing.List[typing.Optional[tf.Tensor]]
    code = self.sources[0].output.placeholder
    result[-1] = code % self.NumBoundaryClasses  # boundary
    code //= self.NumBoundaryClasses
    result[-2] = code % num_states  # state
    code //= num_states
    for i in range(2 * context_len + 1):
      result[2 * context_len - i] = code % num_phone_classes  # phone idx
      code //= num_phone_classes
    self.output.placeholder = tf.stack(result, axis=self.output.batch_ndim - 1)
    self.output.size_placeholder = self.sources[0].output.size_placeholder.copy()

  @classmethod
  def get_out_data_from_opts(cls, name, sources, context_len=1, n_out=NotSpecified, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param int context_len:
    :param int|None|NotSpecified n_out:
    :rtype: Data
    """
    assert len(sources) == 1, "%s: We expect exactly one source layer." % name
    dim = 3 + context_len * 2  # (center, left_1, right_1, ..., state, boundary)
    if n_out is not NotSpecified:
      assert dim == n_out
    return Data(
      name="%s_output" % name,
      shape=sources[0].output.shape + (dim,),
      dtype="int32", sparse=False, dim=dim,
      batch_dim_axis=sources[0].output.batch_dim_axis)


class FramewiseStatisticsLayer(LayerBase):
  """
  Collects various statistics (such as FER, etc) on the sources.
  The tensors will get stored in self.stats which will be collected by TFEngine.
  """
  layer_class = "framewise_statistics"

  def __init__(self, sil_label_idx, histogram_num_bins=20, **kwargs):
    super(FramewiseStatisticsLayer, self).__init__(**kwargs)
    self.output.placeholder = tf.constant(0, name="dummy")
    assert self.sources, "give me some sources"
    # Currently, a bit hardcoded.
    # We expect a framewise hard alignment, and calculate FER, CE, perplexity,
    # for all frames, frames without silence, and silence frames.
    from TFUtil import flatten_with_seq_len_mask
    import numpy
    source = self.sources[0]
    output = source.output
    target = source._get_target_value()
    assert target.sparse
    assert source.output_before_activation.act_func is tf.nn.softmax
    output_seq_lens = output.size_placeholder[0]
    output_before_softmax_flat = flatten_with_seq_len_mask(
      source.output_before_activation.x, output_seq_lens, time_major=output.is_time_major)
    target_seq_lens = target.size_placeholder[0]
    target_flat = flatten_with_seq_len_mask(target.placeholder, target_seq_lens, time_major=target.is_time_major)
    target_flat.set_shape(tf.TensorShape([tf.Dimension(None)]))
    loss_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_before_softmax_flat, labels=target_flat)
    flat_last_dim = output_before_softmax_flat.get_shape().ndims - 1
    assert flat_last_dim == 1
    output_flat = flatten_with_seq_len_mask(output.placeholder, output_seq_lens, time_major=output.is_time_major)
    output_flat_argmax = tf.cast(tf.argmax(output_before_softmax_flat, axis=flat_last_dim), "int32")
    frame_error = tf.not_equal(output_flat_argmax, target_flat)
    # target_flat is shape (time,) -> index.
    target_flat_exp = tf.stack([tf.range(tf.shape(target_flat)[0], dtype=tf.int32), target_flat], axis=1)
    true_label_prob = tf.gather_nd(output_flat, target_flat_exp)
    true_label_prob.set_shape(tf.TensorShape([tf.Dimension(None)]))
    true_label_prob_i32 = tf.clip_by_value(
      tf.cast(tf.round(true_label_prob * histogram_num_bins), tf.int32), 0, histogram_num_bins - 1)
    true_label_prob_histogram = tf.stack(
      [tf.equal(true_label_prob_i32, i) for i in range(histogram_num_bins)], axis=1)
    true_label_prob_histogram.set_shape(tf.TensorShape([tf.Dimension(None), tf.Dimension(histogram_num_bins)]))

    mask_no_sil = tf.not_equal(target_flat, sil_label_idx)
    mask_sil = tf.equal(target_flat, sil_label_idx)
    seq_len = tf.reduce_sum(target_seq_lens)
    seq_len_sil = tf.reduce_sum(tf.cast(mask_sil, tf.int32))
    seq_len_no_sil = tf.reduce_sum(tf.cast(mask_no_sil, tf.int32))

    with self.var_creation_scope():
      accumulated_seq_len = tf.Variable(
        initial_value=0, dtype=tf.int64, trainable=False, name="accumulated_seq_len")
      accumulated_seq_len_sil = tf.Variable(
        initial_value=0, dtype=tf.int64, trainable=False, name="accumulated_seq_len_sil")
    accumulated_seq_len = tf.assign_add(accumulated_seq_len, tf.cast(seq_len, tf.int64))
    accumulated_seq_len_sil = tf.assign_add(accumulated_seq_len_sil, tf.cast(seq_len_sil, tf.int64))
    accumulated_seq_len_no_sil = accumulated_seq_len - accumulated_seq_len_sil

    self.stats["batch_seq_length"] = seq_len
    self.stats["batch_seq_length_sil"] = seq_len_sil
    self.stats["batch_seq_length_no_sil"] = seq_len_no_sil
    self.stats["accumulated_seq_length"] = accumulated_seq_len
    self.stats["accumulated_seq_length_sil"] = accumulated_seq_len_sil
    self.stats["accumulated_seq_length_no_sil"] = accumulated_seq_len_no_sil

    for _k, _v in {
          "loss_ce": loss_ce,
          "frame_error": frame_error,
          "true_label_prob_histogram": true_label_prob_histogram}.items():
      for _k2 in ["", "_sil", "_no_sil"]:
        k = _k + _k2
        v = _v
        acc_seq_len = accumulated_seq_len
        if k.endswith("_no_sil"):
          v = tf.boolean_mask(v, mask_no_sil)
          acc_seq_len = accumulated_seq_len_no_sil
        elif k.endswith("_sil"):
          v = tf.boolean_mask(v, mask_sil)
          acc_seq_len = accumulated_seq_len_sil
        v_f32 = tf.cast(v, tf.float32)
        self.stats["batch_%s" % k] = tf.reduce_mean(v_f32, axis=0)
        if v.dtype.is_floating:
          acc_dtype = "float64"
        else:
          acc_dtype = "int64"
        acc_shape = v.get_shape().as_list()[1:]
        assert all(acc_shape)
        with self.var_creation_scope():
          acc_v = tf.Variable(
            name="accumulated_%s" % k,
            initial_value=numpy.zeros(acc_shape, dtype=acc_dtype), dtype=acc_dtype,
            trainable=False)
        acc_v = tf.assign_add(acc_v, tf.reduce_sum(tf.cast(v, acc_dtype), axis=0))
        self.stats["accumulated_%s" % k] = tf.cast(acc_v, tf.float64) / tf.cast(acc_seq_len, tf.float64)

    self.stats["batch_loss_perplexity"] = tf.exp(self.stats["batch_loss_ce"])
    self.stats["batch_loss_perplexity_sil"] = tf.exp(self.stats["batch_loss_ce_sil"])
    self.stats["batch_loss_perplexity_no_sil"] = tf.exp(self.stats["batch_loss_ce_no_sil"])
    self.stats["accumulated_loss_perplexity"] = tf.exp(self.stats["accumulated_loss_ce"])
    self.stats["accumulated_loss_perplexity_sil"] = tf.exp(self.stats["accumulated_loss_ce_sil"])
    self.stats["accumulated_loss_perplexity_no_sil"] = tf.exp(self.stats["accumulated_loss_ce_no_sil"])

  @classmethod
  def get_out_data_from_opts(cls, **kwargs):
    """
    :rtype: Data
    """
    # n_out=1 is a workaround for now. Our output should not be used. We have none.
    return Data(name="framewise_statistics_dummy_output", shape=(), dtype="int32", batch_dim_axis=None)


class PrintLayer(LayerBase):
  """
  Prints the sources to console/log, via :func:`TFUtil.py_print`.
  """
  layer_class = "print"

  def __init__(self, **kwargs):
    super(PrintLayer, self).__init__(**kwargs)
    from TFUtil import py_print
    with tf.name_scope("print_layer"):
      source = self.sources[0]
      output = py_print(source.output.placeholder, [source.output.placeholder], kwargs["name"], summarize=99)
      self.output.placeholder = output
      self.output.size_placeholder = source.output.size_placeholder.copy()

  @classmethod
  def get_out_data_from_opts(cls, name, sources, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    assert len(sources) == 1, "PrintLayer %r: expects exactly one source, but got: %r" % (name, sources)
    return sources[0].output.copy("%s_output" % name)


class HDFDumpLayer(LayerBase):
  """
  Dumps into HDF file, compatible to :class:`HDFDataset`.

  Common usage would be to add this to your network with "is_output_layer": True,
  such that you don't need to make other layers depend on it.
  """
  layer_class = "hdf_dump"

  def __init__(self, filename, dump_whole_batches=False, **kwargs):
    """
    :param str filename:
    :param bool dump_whole_batches: dumps the whole batch as a single sequence into the HDF
    """
    super(HDFDumpLayer, self).__init__(**kwargs)
    self.output = self.sources[0].output.copy("%s_output" % self.name)
    data = self.output.copy_as_batch_major()  # need batch-major for SimpleHDFWriter

    from HDFDataset import SimpleHDFWriter
    import atexit
    import numpy
    import sys
    self.filename = filename
    self.dump_whole_batches = dump_whole_batches
    self.num_seqs_written = 0
    ndim = data.ndim
    if dump_whole_batches:
      ndim = data.ndim - len(data.size_placeholder) + 1
    data_dim = None if data.sparse else data.dim
    self.hdf_writer = SimpleHDFWriter(filename=filename, dim=data_dim, ndim=ndim)
    atexit.register(self._at_exit)

    def py_write(data_np, tags, *sizes):
      """
      :param numpy.ndarray data_np: (B,...), this is data.placeholder
      :param list[bytes] tags:
      :param sizes:
      :return: unused
      """
      # noinspection PyBroadException
      try:
        n_batch = data_np.shape[0]
        assert len(sizes) == len(data.size_placeholder)
        seq_lens = {i: size for (i, size) in zip(sorted(data.size_placeholder.keys()), sizes)}
        extra = {}
        if self.dump_whole_batches:
          # The batch dim itself becomes another axis to dump.
          # We also want to store the individual seq lens.
          batch_seq_sizes = numpy.zeros((1, n_batch, len(seq_lens)), dtype="int32")
          for i, (axis, size) in enumerate(sorted(seq_lens.items())):
            batch_seq_sizes[0, :, i] = seq_lens[axis]
          extra["seq_sizes"] = batch_seq_sizes
          assert sorted(seq_lens.keys()) == list(range(len(seq_lens)))
          flat_len = numpy.prod(data_np.shape[:len(seq_lens) + 1])
          data_np = data_np.reshape((1, flat_len) + data_np.shape[len(seq_lens) + 1:])
          seq_lens = {0: numpy.array([flat_len], dtype="int32")}
          tags = [b"<->".join(tags)]
          n_batch = 1
        assert n_batch == data_np.shape[0] == len(tags)
        self.num_seqs_written += n_batch
        self.hdf_writer.insert_batch(inputs=data_np, seq_tag=tags, seq_len=seq_lens, extra=extra)
        return 0
      # TF does not print the stacktrace, so we do it instead.
      except Exception:
        sys.excepthook(*sys.exc_info())
        raise

    tf_write = tf.py_func(
      py_write,
      [data.placeholder, self.network.get_seq_tags()] + [size for (i, size) in sorted(data.size_placeholder.items())],
      tf.int64,
      stateful=True)

    self.network.register_post_control_dependencies([tf_write])

  def _at_exit(self):
    print("HDFDumpLayer, wrote %i seqs to file %r." % (self.num_seqs_written, self.filename))
    self.hdf_writer.close()

  @classmethod
  def get_out_data_from_opts(cls, name, sources, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    assert len(sources) == 1, "PrintLayer %r: expects exactly one source, but got: %r" % (name, sources)
    return sources[0].output.copy("%s_output" % name)


class ImageSummaryLayer(LayerBase):
  """
  Creates image summaries which can be viewed in TensorBoard.
  This layer expects the source to be in (T-decoder, T-encoder, B, 1).
  """
  layer_class = "image_summary"

  def __init__(self, max_outputs=3, **kwargs):
    """
    :param max_outputs: number of images to generate per step
    """
    super(ImageSummaryLayer, self).__init__(**kwargs)
    with tf.name_scope("image_summary"):
      input_data = self.sources[0].output
      img = tf.transpose(input_data.placeholder, [2, 0, 1, 3])  # (B, T-dec, T-enc, 1)
      tf.summary.image(kwargs["name"], img, max_outputs=max_outputs)
      self.output.placeholder = input_data.placeholder
      self.output.size_placeholder = input_data.size_placeholder.copy()

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace, the loss_opts
    :param TFNetwork.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    d["sources"] = [get_layer(d.pop("from"))]

  @classmethod
  def get_out_data_from_opts(cls, **kwargs):
    """
    :rtype: Data
    """
    assert "n_out" not in kwargs, "Don't set n_out explicitly in this layer"
    kwargs["n_out"] = kwargs["sources"][0].output.dim
    return super(ImageSummaryLayer, cls).get_out_data_from_opts(**kwargs)


class OfficialResNetLayer(_ConcatInputLayer):
  """
  Wrapper around extern/official_tf_resnet.

  This operates on NHWC (batch, height, width, channel) data, and returns ND, where D = num_classes.
  If you have (batch, time, width, channel) as input,
  you probably want to use :class:`WindowLayer` to get (batch,time,window,width,channel),
  and then :class:`MergeDimsLayer` to get (batch*time,window,width,channel),
  such that we would interpret window = height here.
  Then the output is (batch*time,D),
  and you can use :class:`SplitBatchTimeLayer` to get (batch,time,D).
  As you get logits, you can then use :class:`ActivationLayer` with softmax.
  """
  layer_class = "official_resnet"
  recurrent = True  # convolutional network

  def __init__(self, num_classes, final_size, num_filters, kernel_size,
               conv_stride, first_pool_size, first_pool_stride, first_kernel_size=3,
               block_sizes=None, block_strides=None, conv_time_dim=False,
               bottleneck=False, resnet_size=32, resnet_version=2, data_format=None, **kwargs):
    """
    :param int num_classes:
    :param final_size:
    :param num_filters:
    :param kernel_size:
    :param conv_stride:
    :param first_pool_size:
    :param first_pool_stride:
    :param first_kernel_size:
    :param block_sizes:
    :param block_strides:
    :param conv_time_dim:
    :param bottleneck:
    :param resnet_size:
    :param resnet_version:
    :param data_format:
    """
    if block_strides is None:
      block_strides = [1, 2, 2]
    if block_sizes is None:
      block_sizes = [5, 5, 5]

    import re
    # noinspection PyUnresolvedReferences
    from extern.official_tf_resnet.resnet_model import Model
    super(OfficialResNetLayer, self).__init__(**kwargs)

    self.model = Model(resnet_size=resnet_size, num_classes=num_classes,
                       num_filters=num_filters, conv_time_dim=conv_time_dim,
                       first_kernel_size=first_kernel_size, kernel_size=kernel_size,
                       conv_stride=conv_stride, first_pool_size=first_pool_size,
                       first_pool_stride=first_pool_stride, block_sizes=block_sizes,
                       final_size=final_size, block_strides=block_strides,
                       bottleneck=bottleneck, resnet_version=resnet_version,
                       data_format=data_format)

    # Model assumes always NHWC input format.
    inputs_data = self.input_data.copy_as_batch_major()
    assert (
      inputs_data.batch_ndim == 4 and
      inputs_data.batch_dim_axis == 0 and
      inputs_data.feature_dim_axis == 3)
    output = self.model.__call__(inputs=inputs_data.placeholder, training=self.network.train_flag)
    # Output is logits with [<batch_size>, self.num_classes].
    self.output.placeholder = output
    # Very generic way to collect all created params.
    # Also, see the usage of :func:`LayerBase.cls_layer_scope`, e.g. for initial vars.
    scope_name_prefix = tf.get_variable_scope().name + "/"  # e.g. "layer1/"
    params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=re.escape(scope_name_prefix))
    for p in params:
      if not p.name.startswith(scope_name_prefix):
        continue
      assert p.name.startswith(scope_name_prefix) and p.name.endswith(":0")
      self.params[p.name[len(scope_name_prefix):-2]] = p

    crop = self.model.time_dim_reduction
    self.output.size_placeholder = self.input_data.size_placeholder.copy()
    self.output.size_placeholder[0] -= crop

  @classmethod
  def get_out_data_from_opts(cls, name, num_classes, bottleneck, conv_time_dim, num_filters,
                             block_sizes=None, sources=(), **kwargs):
    """
    :param str name:
    :param int num_classes:
    :param bottleneck:
    :param conv_time_dim:
    :param num_filters:
    :param block_sizes:
    :param sources:
    :rtype: Data
    """
    return Data(
      name="%s_output" % name, shape=(1, num_classes), dtype="float32",
      batch_dim_axis=0, time_dim_axis=1)


# ------------------------------------------------------------------------------

class Loss(object):
  """
  Base class for all losses.
  """
  class_name = None  # type: str  # used by get_loss_class()
  recurrent = False  # if this is a frame-wise criteria, this will be False

  def __init__(self, base_network, use_flatten_frames=True, use_normalized_loss=False, scale=1.0):
    """
    :param TFNetwork.TFNetwork base_network:
    :param bool use_flatten_frames: will use :func:`TFUtil.flatten_with_seq_len_mask`
    :param bool use_normalized_loss: the loss used in optimization will be normalized
    :param float scale: additional scale factor for the loss
    """
    self.base_network = base_network
    self.use_flatten_frames = use_flatten_frames
    self.layer = None  # type: typing.Optional[LayerBase]
    # All are initialized in self.init().
    self.output = None  # type: typing.Optional[Data]
    self.output_with_activation = None  # type: typing.Optional[OutputWithActivation]
    self.output_seq_lens = None  # type: typing.Optional[tf.Tensor]
    self.target = None  # type: typing.Optional[Data]
    self.target_seq_lens = None  # type: typing.Optional[tf.Tensor]
    self.output_flat = None  # type: typing.Optional[tf.Tensor]
    self.output_before_softmax_flat = None  # type: typing.Optional[tf.Tensor]
    self.target_flat = None  # type: typing.Optional[tf.Tensor]
    # Maybe make configurable. For now, same as in our Theano behavior.
    self.loss_norm_factor = None  # type: typing.Optional[tf.Tensor]
    self.use_normalized_loss = use_normalized_loss
    self.scale = scale

  def _reduce_batch_time(self):
    """
    :return: In self.reduce_func, whether to expect that the loss is of shape (batch*time|time*batch,).
    :rtype: bool
    """
    if self.use_flatten_frames:
      return False  # we have used flatten_with_seq_len_mask. see self._flatten_or_merge
    if self.recurrent:
      return False
    if not self.output.have_time_axis():
      return False
    return True

  def _reduce_to_batch_time_with_mask(self, loss, normalize=False):
    """
    :param tf.Tensor loss: (batch*time,...) or (time*batch,...) depending if self.output is batch/time major
    :param bool normalize: for remaining dims. False -> use tf.reduce_sum, True -> use tf.reduce_mean
    :return: (batch*time,) or (time*batch,)
    :rtype: tf.Tensor
    """
    assert {self.output.batch_dim_axis, self.output.time_dim_axis} == {0, 1}
    if loss.get_shape().ndims > 1:
      reduce_func = tf.reduce_mean if normalize else tf.reduce_sum
      loss = reduce_func(loss, axis=list(range(1, loss.get_shape().ndims)))  # reduce remaining dims already
    mask = self.output.get_sequence_mask()  # e.g. (B,T)
    mask = tf.reshape(mask, [tf.shape(loss)[0]])
    loss = tf.where(mask, loss, tf.zeros_like(loss), "loss_masked")
    return loss

  def reduce_func(self, loss):
    """
    Reduces the frames. Currently the sum, and we do averaging later.
    We might change this logic at some point.
    Also, some code overwrites this function externally, e.g. with TFUtil.identity, to not do reducing.

    :param tf.Tensor loss: e.g. (batch*time,), or (time_flat,), or (batch*time,dim), etc
    :return: by default just a scalar. but this can be overwritten, to not reduce
    :rtype: tf.Tensor
    """
    if self._reduce_batch_time():
      # We expect to get (batch*time) or (time*batch) in the first dimension of the loss and the output.
      loss = self._reduce_to_batch_time_with_mask(loss)
    return tf.reduce_sum(loss)

  def reduce_to_batch(self, loss, normalize):
    """
    :param tf.Tensor loss: e.g. (batch*time,), or (time_flat,), or (batch*time,dim), etc
    :param bool normalize: reduce mean instead of reduce sum
    :return: (batch,)
    :rtype: tf.Tensor
    """
    if not self.recurrent and self.output.have_time_axis():
      assert not self.use_flatten_frames
      assert self._reduce_batch_time()
      # We expect to get (batch*time) or (time*batch) in the first dimension of the loss and the output.
      loss = self._reduce_to_batch_time_with_mask(loss, normalize=normalize)  # (batch*time,) or (time*batch,)
      loss.set_shape((None,))
      loss = tf.reshape(loss, tf.shape(self.output.placeholder)[:2])  # (batch,time) or (time,batch)
      loss = tf.reduce_sum(loss, axis=self.output.time_dim_axis)  # (batch,)
      if normalize:
        loss /= tf.to_float(self.output.get_sequence_lengths())
    else:
      # We expect that there is no time-dim, just a batch-dim. It could be like (batch,other_dims...) though.
      if loss.get_shape().ndims > 1:
        reduce_func = tf.reduce_mean if normalize else tf.reduce_sum
        loss = reduce_func(loss, axis=list(range(1, loss.get_shape().ndims)))  # (batch,)
    return loss

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace, the loss_opts
    :param TFNetwork.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer

    Will modify `d` such that it becomes the kwargs for `self.__init__()`.
    Mostly leaves `d` as-is.
    This is used by `LayerBase.transform_config_dict`.
    """

  def init_by_layer(self, layer, layer_output_template=None):
    """
    :param LayerBase|None layer:
    :param Data|None layer_output_template: maybe alternative template
    """
    if layer_output_template and layer_output_template.have_time_axis() and not layer.output.have_time_axis():
      # It could be that the layer is from inside a RecLayer loop, and does not have a time dim.
      # In that case, use our template instead.
      layer_output = layer_output_template
    else:
      # Use the real layer.output instead.
      layer_output = layer.output
    if layer is self.layer and self.output is layer_output:
      return
    # noinspection PyProtectedMember
    self.init(
      output=layer_output,
      output_with_activation=layer.output_before_activation,
      target=layer._get_target_value(),
      layer=layer)

  def _flatten_or_merge(self, x, seq_lens, time_major):
    """
    :param tf.Tensor x: (B,T,...) or (T,B,...)
    :param tf.Tensor seq_lens: (B,)
    :param bool time_major:
    :return: (B*T|T*B|B',...)
    :rtype: tf.Tensor
    """
    if self.use_flatten_frames:
      from TFUtil import flatten_with_seq_len_mask
      return flatten_with_seq_len_mask(x, seq_lens, time_major=time_major)
    x_shape = tf.shape(x)
    x_shape = [x_shape[i] for i in range(x.get_shape().ndims)]
    if time_major != self.output.is_time_major:
      # We expect at various places (eg. reduce_to_batch) that the loss is the same as self.output.
      from TFUtil import swapaxes
      x = swapaxes(x, 0, 1)  # (B,T,...) or (T,B,...)
    return tf.reshape(x, [x_shape[0] * x_shape[1]] + x_shape[2:], name="merge_batch_time")

  def init(self, output, output_with_activation=None, target=None, layer=None):
    """
    :param Data output: generated output
    :param OutputWithActivation|None output_with_activation:
    :param Data target: reference target from dataset
    :param LayerBase|None layer:
    """
    flatten_or_merge = self._flatten_or_merge
    with tf.name_scope("loss_init"):
      self.layer = layer
      if target:
        if output.beam_size:
          if target.beam_size != output.beam_size:
            target = target.copy_extend_with_beam(output.beam_size)
        else:
          assert not target.beam_size
      if output.feature_dim_axis is not None and output.feature_dim_axis != output.batch_ndim - 1:
        if output_with_activation:
          from TFUtil import move_axis
          output_with_activation = OutputWithActivation(x=move_axis(output_with_activation.x,
                                                                    output.feature_dim_axis, -1),
                                                        act_func=output_with_activation.act_func)
        output = output.copy_with_feature_dim_axis(-1)
      self.output = output
      self.output_with_activation = output_with_activation
      self.target = target
      # Flat variants are with batch,time collapsed into one, masked via seq_lens.
      self.output_flat = None
      self.output_before_softmax_flat = None
      self.target_flat = None
      self.output_seq_lens = None
      self.target_seq_lens = None
      self.loss_norm_factor = 1.0
      if self.output.have_time_axis():
        self.output_seq_lens = output.get_sequence_lengths()
        time_and_batch_dims = (self.output.time_dim_axis, self.output.batch_dim_axis)
        assert time_and_batch_dims in [(0, 1), (1, 0)], "output time-batch-dim unexpected: %s" % self.output
        if output_with_activation and output_with_activation.act_func is tf.nn.softmax:
          self.output_before_softmax_flat = flatten_or_merge(
            output_with_activation.x, self.output_seq_lens, time_major=output.is_time_major)
        else:
          self.output_flat = flatten_or_merge(output.placeholder, self.output_seq_lens, time_major=output.is_time_major)
          self.output_flat.set_shape(tf.TensorShape(output.shape))
        if target:
          assert target.have_time_axis()
          self.target_seq_lens = target.get_sequence_lengths()
          self.target_flat = flatten_or_merge(target.placeholder, self.target_seq_lens, time_major=target.is_time_major)
          self.loss_norm_factor = 1.0 / tf.cast(tf.reduce_sum(self.target_seq_lens), tf.float32)
        else:
          self.loss_norm_factor = 1.0 / tf.cast(tf.reduce_sum(self.output_seq_lens), tf.float32)
      else:  # no time axis
        if output_with_activation and output_with_activation.act_func is tf.nn.softmax:
          self.output_before_softmax_flat = output_with_activation.x
        else:
          self.output_flat = output.placeholder
        if self.output.have_batch_axis():
          self.loss_norm_factor = (
            1.0 / tf.cast(tf.shape(self.output.placeholder)[self.output.batch_dim_axis], tf.float32))
        else:
          self.loss_norm_factor = 1.0
        if target:
          assert not self.target.have_time_axis()
          self.target_flat = target.placeholder
      self._check_init()

  def _check_init(self):
    """
    Does some checks on self.target and self.output, e.g. if the dense shapes matches.
    You can overwrite this if those checks don't make sense for your derived loss class.
    """
    if not self.target:
      return
    assert self.target.ndim_dense == self.output.ndim_dense, (
      "Number of dimensions mismatch. Target: %s, output: %s" % (self.target, self.output))
    expected_output_dim = self.get_auto_output_layer_dim(self.target.dim)
    assert expected_output_dim == self.output.dim, (
      "Expected output dim is %i but the output has dim %r. " % (expected_output_dim, self.output.dim) +
      "Target: %s, output: %s" % (self.target, self.output))

  def get_error(self):
    """
    :return: frame error rate as a scalar value with the default self.reduce_func (see also self.get_value)
    :rtype: tf.Tensor
    """
    with tf.name_scope("loss_frame_error"):
      assert self.output.ndim_dense == self.target.ndim_dense
      from TFUtil import check_input_ndim, check_shape_equal
      output_flat = self.output_before_softmax_flat
      if output_flat is None:
        output_flat = self.output_flat
      output_flat = check_input_ndim(output_flat, ndim=2)
      last_dim = tf.rank(output_flat) - 1  # should be 1
      if self.target.sparse:
        target_label = check_input_ndim(self.target_flat, ndim=1)
      else:
        target_flat = check_shape_equal(self.target_flat, output_flat)
        target_label = tf.cast(tf.argmax(target_flat, axis=last_dim), tf.int32)
      output_label = tf.cast(tf.argmax(output_flat, axis=last_dim), target_label.dtype)
      not_equal = tf.not_equal(output_label, target_label)
      return self.reduce_func(tf.cast(not_equal, tf.float32))

  def get_value(self):
    """
    :return: self.reduce_func(loss), which is usually a scalar with the default as if does tf.reduce_sum.
      float32 value. it should *not* be normalized over frames,
      as this will be calculated in :func:`TFEngine.Runner._collect_eval_info`.
    :rtype: tf.Tensor|None
    """
    raise NotImplementedError

  def get_normalization_factor(self):
    """
    :return: factor as a float scalar, usually 1.0 / num_frames. see self.reduce_func.
    :rtype: tf.Tensor
    """
    assert self.loss_norm_factor is not None, "init not called?"
    return self.loss_norm_factor

  @classmethod
  def get_auto_output_layer_dim(cls, target_dim):
    """
    :param int target_dim:
    :return: normally just the same as target_dim. e.g. for CTC, we would add 1 for the blank label
    :rtype: int
    """
    return target_dim

  @classmethod
  def get_default_target(cls, extern_data):
    """
    :param TFNetwork.ExternData extern_data:
    :return: default target name, or None if this loss does not have a target
    :rtype: str|None
    """
    return extern_data.default_target


class CrossEntropyLoss(Loss):
  """
  Cross-Entropy loss. Basically sum(target * log(output)).
  """
  class_name = "ce"

  def __init__(self,
               focal_loss_factor=0.0,
               label_smoothing=0.0, label_smoothing_gaussian=False,
               debug_dump=False,
               safe_log_opts=None,
               use_fused=True,
               **kwargs):
    """
    :param float focal_loss_factor: see https://arxiv.org/abs/1708.02002. 0 means disabled
    :param float label_smoothing: 0.1 is a common default. see :func:`TFUtil.smoothing_cross_entropy`
    :param bool label_smoothing_gaussian: see :func:`TFUtil.smoothing_cross_entropy`
    :param bool debug_dump:
    :param dict[str] safe_log_opts: passed to :func:`safe_log`
    :param bool use_fused: if possible, use fused opts
    """
    super(CrossEntropyLoss, self).__init__(**kwargs)
    self.focal_loss_factor = focal_loss_factor
    self.label_smoothing = label_smoothing
    self.label_smoothing_gaussian = label_smoothing_gaussian
    self.debug_dump = debug_dump
    self.safe_log_opts = safe_log_opts or {}
    self.use_fused = use_fused

  def get_output_target_scores(self):
    """
    :return: shape (time_flat,), type float32
    :rtype: tf.Tensor
    """
    output_flat = self.output_flat
    if output_flat is None:
      output_flat = self.output.get_placeholder_time_flattened()
    target_flat_exp = tf.stack(
      [tf.range(tf.shape(self.target_flat)[0], dtype=tf.int32),
       tf.cast(self.target_flat, tf.int32)], axis=1)  # (time,2)
    out = tf.gather_nd(output_flat, target_flat_exp)
    return out

  def get_value(self):
    """
    :rtype: tf.Tensor
    """
    from TFUtil import to_int32_64, smoothing_cross_entropy, safe_log, py_print
    with tf.name_scope("loss_ce"):
      assert self.target.ndim_dense == self.output.ndim_dense
      if self.target.sparse:
        if self.use_fused and self.output_before_softmax_flat is not None:
          target_flat = self.target_flat
          if self.debug_dump:
            target_flat = py_print(target_flat, [target_flat], summarize=10000, message='target word IDs ')
            target_flat = py_print(target_flat, [tf.shape(target_flat)], message='sequence length ')
          if self.label_smoothing:
            out = smoothing_cross_entropy(
              logits=self.output_before_softmax_flat, labels=to_int32_64(target_flat), vocab_size=self.target.dim,
              label_smoothing=self.label_smoothing, gaussian=self.label_smoothing_gaussian)  # shape(labels)
          else:
            # This is really the standard case which we hope to get:
            out = tf.nn.sparse_softmax_cross_entropy_with_logits(
              logits=self.output_before_softmax_flat, labels=to_int32_64(target_flat))  # shape(labels)
          if self.debug_dump:
            out = py_print(out, [tf.exp(tf.negative(out))], summarize=10000, message='target prob ')
        else:
          assert not self.label_smoothing, "not implemented"
          print("Warning: using numerical unstable sparse Cross-Entropy loss calculation", file=log.v3)
          out = -safe_log(self.get_output_target_scores(), **self.safe_log_opts)
        if self.focal_loss_factor:
          out *= (1.0 - self.get_output_target_scores()) ** self.focal_loss_factor
        return self.reduce_func(out)
      else:  # not sparse
        assert not self.focal_loss_factor, "not implemented"
        assert not self.label_smoothing, "not implemented"
        assert not self.debug_dump, "not implemented"
        if self.use_fused and self.output_before_softmax_flat is not None:
          out = tf.nn.softmax_cross_entropy_with_logits(logits=self.output_before_softmax_flat, labels=self.target_flat)
          return self.reduce_func(out)
        else:
          print("Warning: using numerical unstable dense Cross-Entropy loss calculation", file=log.v3)
          out = self.target_flat * safe_log(self.output_flat, **self.safe_log_opts)
          return -self.reduce_func(out)


class BinaryCrossEntropyLoss(Loss):
  """
  Binary cross entropy.
  We expect the output as logits, not in probability space!
  Per frame: mean(target * log(sigmoid(output)) + (1 - target) * log(1 - sigmoid(output)))
  """
  class_name = "bin_ce"

  def get_value(self):
    """
    :rtype: tf.Tensor
    """
    assert not self.target.sparse, "sparse is not supported yet"
    assert self.target.dim == self.output.dim
    with tf.name_scope("loss_bin_ce"):
      out = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output_flat, labels=self.target_flat)
      return self.reduce_func(out * (1.0 / self.target.dim))


class GenericCELoss(Loss):
  """
  Some generalization of cross entropy.
  """
  class_name = "generic_ce"

  def __init__(self, **kwargs):
    super(GenericCELoss, self).__init__(**kwargs)

    # noinspection PyUnusedLocal
    def loss(z, y, grad_f, target):
      """
      :param tf.Tensor z:
      :param tf.Tensor y:
      :param grad_f:
      :param tf.Tensor target:
      :rtype: tf.Tensor
      """
      nlog_scores = -tf.log(tf.clip_by_value(y, 1.e-20, 1.e20))  # (time,dim)
      # target is shape (time,) -> index.
      target_exp = tf.stack([tf.range(tf.shape(target)[0], dtype=tf.int32), target], axis=1)  # (time,2)
      # Thus K == 2. gather_nd out will be (target_exp.shape[0],) = (time,).
      gathered = tf.gather_nd(nlog_scores, target_exp)   # (time,)
      return self.reduce_func(gathered)

    # noinspection PyUnusedLocal
    def loss_grad(op, grad):
      """
      :param tf.Operation op:
      :param tf.Tensor grad: grad for loss
      :return: grad for op.outputs
      """
      z, y, grad_f, target = op.inputs
      num_classes = tf.shape(z)[-1]
      bw = tf.one_hot(target, depth=num_classes)
      grad_z = grad_f * (y - bw)
      return grad_z, None, None, None  # for each input

    # We need to create the loss func here in __init__ to register it in the default graph as early as possible,
    # before we create the TF session.
    from TFUtil import custom_gradient
    self._loss_func = custom_gradient.register(
      [tf.float32, tf.float32, tf.float32, tf.int32], op=loss, grad_op=loss_grad)

  def get_value(self):
    """
    :rtype: tf.Tensor
    """
    # Should be generic for any activation function.
    # (Except when the labels are not independent, such as for softmax.)
    # See Theano NetworkOutputLayer.FramewiseOutputLayer.cost() with "generic_ce" loss.
    from TFUtil import flatten_with_seq_len_mask
    # activation function can be anything, e.g. exp or sigmoid, but not softmax, must be elemwise.
    assert self.output_with_activation
    x = self.output_with_activation.x
    y = self.output_with_activation.y
    grad_f, = tf.gradients(tf.log(y), x)
    assert grad_f is not None
    grad_f = flatten_with_seq_len_mask(grad_f, seq_lens=self.output_seq_lens, time_major=self.output.is_time_major)
    x = flatten_with_seq_len_mask(x, seq_lens=self.output_seq_lens, time_major=self.output.is_time_major)
    y = flatten_with_seq_len_mask(y, seq_lens=self.output_seq_lens, time_major=self.output.is_time_major)
    assert y.get_shape().ndims == 2
    y /= tf.reduce_sum(y, axis=1, keep_dims=True)
    assert self.output.dim == self.target.dim
    assert self.target.sparse
    return self._loss_func(x, y, grad_f, self.target_flat)


class CtcLoss(Loss):
  """
  Connectionist Temporal Classification (CTC) loss.
  Basically a wrapper around tf.nn.ctc_loss.
  """
  class_name = "ctc"
  recurrent = True

  def __init__(self, target_collapse_repeated=False, auto_clip_target_len=False, output_in_log_space=False,
               beam_width=100, ctc_opts=None,
               focal_loss_factor=0.0,
               use_native=False, use_viterbi=False, **kwargs):
    """
    :param bool target_collapse_repeated: like preprocess_collapse_repeated option for CTC. used for sparse_labels().
    :param bool auto_clip_target_len: see self._get_target_sparse_labels().
    :param bool output_in_log_space: False -> output expected in prob space. see self.get_output_logits
    :param int beam_width: used in eval
    :param dict[str]|None ctc_opts: other kwargs used for tf.nn.ctc_loss
    :param float focal_loss_factor: see https://arxiv.org/abs/1708.02002. 0 means disabled. generalized for CTC
    :param bool use_native: use our native implementation (:func:`TFNativeOp.ctc_loss`)
    :param bool use_viterbi: instead of full-sum, use only best path (via :func:`ctc_loss_viterbi`)
    """
    super(CtcLoss, self).__init__(**kwargs)
    self.target_collapse_repeated = target_collapse_repeated
    self.auto_clip_target_len = auto_clip_target_len
    self.output_in_log_space = output_in_log_space
    self._target_sparse_labels = None
    self._ctc_loss = None  # set in get_value
    self.beam_width = beam_width
    self.ctc_opts = ctc_opts
    self.focal_loss_factor = focal_loss_factor
    self.use_native = use_native
    self.use_viterbi = use_viterbi

  def init(self, **kwargs):
    """
    See super.
    """
    self._target_sparse_labels = None
    super(CtcLoss, self).init(**kwargs)

  def _get_target_sparse_labels(self):
    """
    :rtype: tf.SparseTensor
    """
    if self._target_sparse_labels is not None:
      return self._target_sparse_labels
    from TFUtil import sparse_labels
    target_seq_lens = self.target_seq_lens
    if self.auto_clip_target_len:
      # Not more than output_seq_lens, otherwise we can get an exception by the CTC algorithm
      # "Not enough time for target transition sequence".
      # One less to allow for at least one blank somewhere.
      target_seq_lens = tf.minimum(target_seq_lens, tf.maximum(self.output_seq_lens - 1, 0))
    assert self.target.is_batch_major
    labels = sparse_labels(self.target.placeholder, target_seq_lens,
                           collapse_repeated=self.target_collapse_repeated)
    self._target_sparse_labels = labels
    return labels

  def get_output_logits(self):
    """
    :return: outputs in log-space / logits
    :rtype: tf.Tensor
    """
    from TFUtil import safe_log
    if self.output_in_log_space:
      logits = self.output.placeholder
    else:
      # If not self.output_in_log_space, we expect the values in probability space.
      logits = self.output_with_activation
      if self.output_with_activation:
        logits = self.output_with_activation.get_logits()
      if logits is None:
        logits = safe_log(self.output.placeholder)
    assert logits.get_shape().ndims == 3  # (B,T,N) or (T,B,N)
    assert logits.get_shape().dims[2].value == self.target.dim + 1  # one more for blank
    return logits

  def get_soft_alignment(self):
    """
    Also called the Baum-Welch-alignment.
    This is basically p_t(s|x_1^T,w_1^N), where s are the output labels (including blank),
    and w are the real target labels.

    :return: shape (time, batch, dim)
    :rtype: tf.Tensor
    """
    assert self._ctc_loss is not None
    assert isinstance(self._ctc_loss, tf.Tensor)
    assert self._ctc_loss.op.type == "CTCLoss"
    # See grad definition of CTCLoss.
    # The op will calculate the gradient w.r.t. the logits.
    # I.e. with y = softmax(z), this is \partial loss / \partial z = y - soft_align.
    ctc_grad_z = self._ctc_loss.op.outputs[1]  # time major, i.e. (time, batch, dim)
    y = self.output.get_placeholder_as_time_major()  # (time, batch, dim)
    soft_align = y - ctc_grad_z
    soft_align.set_shape(tf.TensorShape((None, None, self.output.dim)))
    return soft_align

  def get_focal_loss_factor(self):
    """
    :return: shape (time, batch, dim)
    :rtype: tf.Tensor
    """
    y = self.output.get_placeholder_as_time_major()
    return (1.0 - y) ** self.focal_loss_factor

  def get_value(self):
    """
    :rtype: tf.Tensor
    """
    if not self.target.sparse:
      raise Exception("CTC target expected to be sparse (symbols)")
    with tf.name_scope("loss_ctc"):
      logits = self.get_output_logits()
      seq_lens = self.output_seq_lens
      labels = self._get_target_sparse_labels()
      # logits can be unnormalized. It will do softmax internally.
      if self.use_viterbi:
        assert not self.ctc_opts
        import TFNativeOp
        self._ctc_loss = TFNativeOp.ctc_loss_viterbi(
          logits=logits, logits_seq_lens=seq_lens, logits_time_major=self.output.is_time_major,
          targets=self.target.get_placeholder_as_batch_major(), targets_seq_lens=self.target_seq_lens)
      elif self.use_native:
        assert not self.ctc_opts
        import TFNativeOp
        self._ctc_loss = TFNativeOp.ctc_loss(
          logits=logits, logits_seq_lens=seq_lens, logits_time_major=self.output.is_time_major,
          targets=self.target.get_placeholder_as_batch_major(), targets_seq_lens=self.target_seq_lens)
      else:
        self._ctc_loss = tf.nn.ctc_loss(
          inputs=logits, labels=labels, sequence_length=seq_lens, time_major=self.output.is_time_major,
          **(self.ctc_opts or {}))
      loss = self._ctc_loss  # shape (batch,)
      if self.focal_loss_factor:
        # We are going up to (time,batch,dim), and we later use reduce_sum,
        # and we want the same normalization, thus we multiply that in now.
        loss /= tf.cast(self.output_seq_lens, tf.float32)
        loss /= tf.cast(self.output.dim, tf.float32)
        loss = tf.expand_dims(loss, axis=0)  # (time,batch)
        loss = tf.expand_dims(loss, axis=2)  # (time,batch,dim)
        loss *= self.get_focal_loss_factor()
        from TFUtil import flatten_with_seq_len_mask
        loss = flatten_with_seq_len_mask(loss, seq_lens=self.output_seq_lens, time_major=True)  # (time_flat,dim)
      return self.reduce_func(loss)

  def get_error(self):
    """
    :rtype: tf.Tensor
    """
    if not self.target.sparse:
      raise Exception("CTC target expected to be sparse (symbols)")
    with tf.name_scope("loss_ctc_error"):
      logits = self.get_output_logits()
      if not self.output.is_time_major:
        logits = tf.transpose(logits, [1, 0, 2])  # (B,T,N) => (T,B,N)
      seq_lens = self.output_seq_lens
      if self.beam_width > 1:
        assert not self.use_native, "use beam_width=1 with use_native"
        decoded = self.base_network.cond_on_train(
          lambda: tf.nn.ctc_greedy_decoder(inputs=logits, sequence_length=seq_lens)[0][0],
          lambda: tf.nn.ctc_beam_search_decoder(
            inputs=logits, sequence_length=seq_lens, beam_width=self.beam_width)[0][0])
      else:
        if self.use_native or self.use_viterbi:
          decoded = TFUtil.ctc_greedy_decode(logits=logits, seq_lens=seq_lens, time_major=True)
        else:
          decoded = tf.nn.ctc_greedy_decoder(inputs=logits, sequence_length=seq_lens)[0][0]
      labels = self._get_target_sparse_labels()
      error = tf.edit_distance(hypothesis=tf.cast(decoded, labels.dtype), truth=labels, normalize=False)
      return self.reduce_func(error)

  @classmethod
  def get_auto_output_layer_dim(cls, target_dim):
    """
    :rtype: int
    """
    return target_dim + 1  # one added for blank


class EditDistanceLoss(Loss):
  """
  Note that this loss is not differentiable, thus it's only for keeping statistics.
  """
  class_name = "edit_distance"
  recurrent = True

  def __init__(self, debug_print=False, label_map=None, ctc_decode=False, output_in_log_space=False, **kwargs):
    """
    :param bool debug_print: will tf.Print the sequence
    :param dict[int,int]|None label_map: before calculating the edit-distance, will apply this map
    :param bool ctc_decode: True -> expects dense output and does CTC decode, False -> expects sparse labels in output
    :param bool output_in_log_space: False -> dense output expected in prob space. see self.get_output_logits
    """
    super(EditDistanceLoss, self).__init__(**kwargs)
    self._output_sparse_labels = None
    self._target_sparse_labels = None
    self._debug_print = debug_print
    self._label_map = label_map
    self._ctc_decode = ctc_decode
    self._output_in_log_space = output_in_log_space
    if self._ctc_decode:
      self.get_auto_output_layer_dim = lambda dim: dim + 1

  def init(self, output, output_with_activation=None, target=None, **kwargs):
    """
    :param Data output: generated output
    :param OutputWithActivation|None output_with_activation:
    :param Data target: reference target from dataset
    """
    super(EditDistanceLoss, self).init(
      output=output, output_with_activation=output_with_activation, target=target, **kwargs)
    assert target.sparse
    if output.sparse:
      assert not self._ctc_decode
      assert output.dim == target.dim
      assert output.shape == target.shape
    else:
      assert self._ctc_decode
      assert output.dim == target.dim + 1
      assert output.shape == target.shape + (target.dim + 1,)
    self._output_sparse_labels = None
    self._target_sparse_labels = None

  # noinspection PyMethodMayBeStatic
  def _sparse_labels(self, output, seq_lens):
    """
    :param tf.Tensor output: batch-major, (batch, time) -> idx, of type int32
    :param tf.Tensor seq_lens: (batch,) -> seq-len
    :rtype: tf.SparseTensor
    """
    from TFUtil import sparse_labels
    return sparse_labels(output, seq_lens=seq_lens)

  def _map_labels(self, labels):
    """
    :param tf.SparseTensor labels:
    :rtype: tf.SparseTensor
    """
    if not self._label_map:
      return labels
    from TFUtil import map_labels
    return map_labels(labels, label_map=self._label_map)

  def get_output_logits(self):
    """
    :return: outputs in log-space / logits
    :rtype: tf.Tensor
    """
    from TFUtil import safe_log
    assert not self.output.sparse
    if self._output_in_log_space:
      logits = self.output.placeholder
    else:
      # If not self.output_in_log_space, we expect the values in probability space.
      logits = self.output_with_activation
      if self.output_with_activation:
        logits = self.output_with_activation.get_logits()
      if logits is None:
        logits = safe_log(self.output.placeholder)
    assert logits.get_shape().ndims == 3  # (B,T,N) or (T,B,N)
    assert logits.get_shape().dims[2].value == self.target.dim + 1  # one more for blank
    return logits

  def _ctc_decode_dense_output(self):
    assert not self.output.sparse
    logits = self.get_output_logits()
    if not self.output.is_time_major:
      logits = tf.transpose(logits, [1, 0, 2])  # (B,T,N) => (T,B,N)
    seq_lens = self.output_seq_lens
    # decoded = self.base_network.cond_on_train(
    #  lambda: tf.nn.ctc_greedy_decoder(inputs=logits, sequence_length=seq_lens)[0],
    #  lambda: tf.nn.ctc_beam_search_decoder(inputs=logits, sequence_length=seq_lens)[0]
    # )
    # TODO...
    decoded = tf.nn.ctc_beam_search_decoder(inputs=logits, sequence_length=seq_lens)[0][0]
    assert isinstance(decoded, tf.SparseTensor)
    return decoded

  def _get_output_sparse_labels(self):
    """
    :rtype: tf.SparseTensor
    """
    if self._output_sparse_labels is not None:
      return self._output_sparse_labels
    if self._ctc_decode:
      labels = self._ctc_decode_dense_output()
    else:
      labels = self._sparse_labels(self.output.get_placeholder_as_batch_major(), seq_lens=self.output_seq_lens)
    labels = self._map_labels(labels)
    self._output_sparse_labels = labels
    return labels

  def _get_target_sparse_labels(self):
    """
    :rtype: tf.SparseTensor
    """
    if self._target_sparse_labels is not None:
      return self._target_sparse_labels
    labels = self._sparse_labels(self.target.get_placeholder_as_batch_major(), seq_lens=self.target_seq_lens)
    labels = self._map_labels(labels)
    self._target_sparse_labels = labels
    return labels

  def _debug_print_out(self):
    """
    :rtype: list[str|tf.Tensor]
    """
    def get_first_seq(data):
      """
      :param Data data:
      :rtype: tf.Tensor
      """
      x = data.get_placeholder_as_batch_major()
      seq = x[0][:data.size_placeholder[0][0]]
      return seq
    output = get_first_seq(self.output)
    target = get_first_seq(self.target)
    from TFUtil import vocab_idx_repr
    return ["output", tf.size(output), vocab_idx_repr(output, self.output),
            "target", tf.size(target), vocab_idx_repr(target, self.output)]

  def get_error(self):
    """
    :rtype: tf.Tensor
    """
    output = self._get_output_sparse_labels()
    labels = self._get_target_sparse_labels()
    error = tf.edit_distance(hypothesis=output, truth=labels, normalize=False)
    if self._debug_print:
      from TFUtil import py_print
      error = py_print(error, self._debug_print_out(), summarize=10)
    return self.reduce_func(error)

  def get_value(self):
    """
    :rtype: None
    """
    return None


class BleuLoss(Loss):
  """
  Note that this loss is not differentiable, thus it's only for keeping statistics.
  Also, BLEU is a score, i.e. the higher, the better.
  Thus, to interpret it as a loss or error, we take the negative value.
  """
  class_name = "bleu"
  recurrent = True

  def __init__(self, **kwargs):
    super(BleuLoss, self).__init__(**kwargs)

  def init(self, output, output_with_activation=None, target=None, **kwargs):
    """
    :param Data output: generated output
    :param OutputWithActivation|None output_with_activation:
    :param Data target: reference target from dataset
    """
    super(BleuLoss, self).init(
      output=output, output_with_activation=output_with_activation, target=target, **kwargs)
    assert target.sparse
    assert output.sparse
    assert output.dim == target.dim
    assert output.shape == target.shape

  def get_error(self):
    """
    :rtype: tf.Tensor
    """
    from TFUtil import bleu_score
    score = bleu_score(
      hypothesis=self.output.get_placeholder_as_batch_major(), hyp_seq_lens=self.output.get_sequence_lengths(),
      truth=self.target.get_placeholder_as_batch_major(), truth_seq_lens=self.target.get_sequence_lengths())
    # Take negative, to make it a loss/error, which we want to minimize.
    return -self.reduce_func(score)

  def get_value(self):
    """
    :rtype: None
    """
    return None


class ExpectedLoss(Loss):
  """
  This loss uses another loss error or value and given the search beam scores, calculates the expected loss.
  Sometimes also called minimum Bayes risk.
  """
  class_name = "expected_loss"
  recurrent = True  # we don't know

  def __init__(self, loss, loss_kind,
               norm_scores=True, norm_scores_stop_gradient=True,
               divide_beam_size=True, subtract_average_loss=True,
               loss_correction_grad_only=False,
               **kwargs):
    """
    :param Loss loss:
    :param str loss_kind: "error" or "value". whether to use loss.get_error() or loss.get_value()
    :param bool norm_scores:
    :param bool norm_scores_stop_gradient:
    :param bool divide_beam_size:
    :param bool subtract_average_loss:
    :param bool loss_correction_grad_only:
    """
    super(ExpectedLoss, self).__init__(**kwargs)
    from TFUtil import identity
    self.losses = loss
    self.losses.reduce_func = identity  # see self.get_value()
    self.loss_kind = loss_kind
    self.norm_scores = norm_scores
    self.norm_scores_stop_gradient = norm_scores_stop_gradient
    self.divide_beam_size = divide_beam_size
    self.subtract_average_loss = subtract_average_loss
    self.loss_correction_grad_only = loss_correction_grad_only
    self.search_choices = None  # type: typing.Optional[SearchChoices]

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param TFNetwork.TFNetwork network:
    :param get_layer:
    """
    assert "loss" in d, "specify 'loss' in 'loss_opts' for the expected loss"
    assert isinstance(d["loss"], dict)
    opts = d["loss"].copy()
    assert isinstance(opts, dict)
    class_name = opts.pop("class")
    loss_class = get_loss_class(class_name)
    assert issubclass(loss_class, Loss)
    loss_class.transform_config_dict(opts, network=network, get_layer=get_layer)
    loss = loss_class(base_network=network, **opts)
    assert isinstance(loss, Loss)
    d["loss"] = loss

  def init(self, **kwargs):
    """
    Overwrites super. Get search choices.
    """
    super(ExpectedLoss, self).init(**kwargs)
    self.losses.init(**kwargs)
    assert isinstance(self.layer, LayerBase)
    self.search_choices = self.layer.get_search_choices()
    assert isinstance(self.search_choices, SearchChoices), "no search choices from layer %r" % self.layer

  def get_value(self):
    """
    :rtype: tf.Tensor
    """
    with tf.name_scope("expected_loss"):
      if self.loss_kind == "value":
        losses = self.losses.get_value()
      elif self.loss_kind == "error":
        losses = tf.to_float(self.losses.get_error())
      else:
        raise ValueError("invalid loss_kind %r" % self.loss_kind)
      assert losses is not None, "no value for loss_kind %r with loss %r" % (self.loss_kind, self.losses)
      beam_scores = self.search_choices.beam_scores  # (batch,beam), +log scores
      # We currently expect that `losses` is of shape (batch*beam,), as we set reduce_func = identity,
      # and that self.losses is a sequence criterion.
      # This does not work for frame-wise criteria yet where we get (batch*beam*time') flattened.
      losses = tf.reshape(losses, tf.shape(beam_scores), name="losses")  # (batch,beam)
      corrected_losses = losses
      if self.norm_scores:
        scores_norm_shift = tf.reduce_logsumexp(
          beam_scores, name="scores_norm_shift", axis=1, keep_dims=True)  # (batch,1)
        if self.norm_scores_stop_gradient:
          scores_norm_shift = tf.stop_gradient(scores_norm_shift)
        # Thus sum(value_weights) == 1.
        value_weights = tf.exp(beam_scores - scores_norm_shift)
      else:
        value_weights = tf.exp(beam_scores)
      if self.subtract_average_loss:
        # Gradient variance reduction for the gradient of the value-weights.
        # In case that the values also are differentiable, we don't want it to propagate through this.
        corrected_losses -= tf.stop_gradient(tf.reduce_mean(losses, axis=1, keep_dims=True, name="avg_loss"))
      weighted_losses = tf.reduce_sum(corrected_losses * value_weights, axis=1, name="weighted_loss")  # (batch,)
      if self.loss_correction_grad_only and corrected_losses is not losses:
        weighted_losses += tf.stop_gradient(tf.reduce_sum((losses - corrected_losses) * value_weights, axis=1))
      if self.divide_beam_size:
        weighted_losses /= tf.to_float(tf.shape(beam_scores)[-1])
      return self.reduce_func(weighted_losses)

  def get_error(self):
    """
    :rtype: None
    """
    return None


class DeepClusteringLoss(Loss):
  """
  Cost function used for deep clustering as described in
  [Hershey & Chen+, 2016]: "Deep clustering discriminative embeddings for segmentation and separation"
  """
  class_name = "deep_clustering"

  def __init__(self, embedding_dimension, nr_of_sources, **kwargs):
    """
    :param int embedding_dimension:
    :param int nr_of_sources:
    """
    super(DeepClusteringLoss, self).__init__(**kwargs)
    self._embedding_dimension = embedding_dimension
    self._nr_of_sources = nr_of_sources

  def _check_init(self):
    """
    Does some checks on self.target and self.output, e.g. if the dense shapes matches.
    You can overwrite this if those checks don't make sense for your derived loss class.
    """
    assert self.target.ndim_dense == self.output.ndim_dense, (
      "Number of dimensions mismatch. Target: %s, output: %s" % (self.target, self.output))
    expected_output_dim = self._embedding_dimension * (self.target.shape[1] // self._nr_of_sources)
    assert expected_output_dim == self.output.dim, (
      "Expected output dim is %i but the output has dim %r. " % (expected_output_dim, self.output.dim) +
      "Target: %s, output: %s" % (self.target, self.output))

  def get_error(self):
    """
    :return: frame error rate as a scalar value
    :rtype: tf.Tensor | None
    """
    return None

  def get_value(self):
    """
    :rtype: tf.Tensor
    """
    assert not self.target.sparse, "sparse is not supported yet"
    assert self.target.dim == self.output.dim
    with tf.name_scope("loss_deep_clustering"):
      # iterate through all chunks and compute affinity cost function for every chunk separately

      # noinspection PyUnusedLocal,PyShadowingNames
      def iterate_sequences(s, start, c):
        """
        :param tf.Tensor s:
        :param start:
        :param c:
        :rtype: tf.Tensor
        """
        return tf.less(s, tf.shape(self.output_seq_lens)[0])

      # noinspection PyShadowingNames
      def compute_cost(s, start, c):
        """
        :param tf.Tensor s: scalar, int32, seq idx
        :param tf.Tensor start: scalar, int32, time offset
        :param tf.Tensor c: scalar, float32, accumulated loss
        :return: new (s, start, c)
        :rtype: (tf.Tensor, tf.Tensor, tf.Tensor)
        """
        seq_length = self.output_seq_lens[s]  # scalar, int32
        # Note: This slice indexed access will be inefficient to do in every frame of the loop.
        #   It's better to use a tf.TensorArray.
        #   It would also be better/faster/easier to use self.output/self.target instead of the flat variants.
        #   It would also maybe be easier to use tf.foldl, tf.scan or some of the other variants instead
        #   of tf.while_loop.
        chunk_out = self.output_flat[start:(start + seq_length), :]  # (time, dim)
        chunk_target = self.target_flat[start:(start + seq_length), :]  # (time, dim)
        # convert network output into embedding vectors
        # Note: The first reshape is redundant if you reshape it right after again.
        v = tf.reshape(
          tf.reshape(
            chunk_out,
            (tf.shape(chunk_out)[0], tf.shape(chunk_out)[1] // self._embedding_dimension, self._embedding_dimension)),
          (tf.shape(chunk_out)[0] * (tf.shape(chunk_out)[1] // self._embedding_dimension), self._embedding_dimension))
        # convert targets into class vectors
        # Note: The first reshape is redundant if you reshape it right after again.
        y = tf.reshape(
          tf.reshape(
            chunk_target,
            (tf.shape(chunk_target)[0], tf.shape(chunk_target)[1] // self._nr_of_sources, self._nr_of_sources)),
          (tf.shape(chunk_target)[0] * (tf.shape(chunk_target)[1] // self._nr_of_sources), self._nr_of_sources))
        chunk_c = (
          tf.pow(tf.norm(tf.matmul(tf.transpose(v), v)), 2)
          - 2.0 * tf.pow(tf.norm(tf.matmul(tf.transpose(v), y)), 2)
          + tf.pow(tf.norm(tf.matmul(tf.transpose(y), y)), 2))
        # append chunk cost to cost tensor
        # Note: It's very inefficient to have a different shape in each frame of the loop.
        #   A tf.TensorArray should be used instead.
        #   As I see from the code, you are anyway reducing it at the end,
        #   so it would be even easier to just accumulate it here (just sum it).
        # Note: tf.cond can be slow. It should only be used if the arguments are very expensive to compute.
        #   Maybe tf.where might be better. But if you just sum it up, you don't have that problem here anyway.
        c = tf.cond(
          tf.greater(s, 0),
          lambda: tf.concat([c, tf.reshape(chunk_c, (1,))], axis=0),
          lambda: tf.reshape(chunk_c, (1,)))
        return tf.add(s, 1), tf.add(start, seq_length), c

      c = tf.constant(0.0)
      s = tf.constant(0, dtype=tf.int32)
      start = tf.constant(0, dtype=tf.int32)
      r = tf.while_loop(
        iterate_sequences, compute_cost, [s, start, c],
        shape_invariants=[s.get_shape(), start.get_shape(), tf.TensorShape([None])])
      return self.reduce_func(r[-1])


class L1Loss(Loss):
  """
  L1-distance loss. sum(target - output).
  """
  class_name = "l1"

  def get_value(self):
    """
    :rtype: tf.Tensor
    """
    assert not self.target.sparse, "sparse target values are not yet supported"
    with tf.name_scope("loss_l1"):
      return self.reduce_func(tf.abs(self.target_flat - self.output_flat))


class MeanSquaredError(Loss):
  """
  The generic mean squared error loss function
  """
  class_name = "mse"

  def get_value(self):
    """
    :rtype: tf.Tensor
    """
    assert not self.target.sparse, "sparse is not supported yet"
    with tf.name_scope("loss_mse"):
      if self.target_flat is not None:
        assert self.output_flat is not None
        out = tf.squared_difference(self.output_flat, self.target_flat)
        assert out.get_shape().ndims == 2
        out = self.reduce_func(tf.reduce_mean(out, axis=1))
      else:
        assert self.output is not None and self.target is not None
        out = tf.squared_difference(self.output, self.target)
        assert out.get_shape().ndims == 1
        out = self.reduce_func(out)
      return out


class ExternSprintLoss(Loss):
  """
  The loss is calculated by an extern Sprint instance.
  """
  class_name = "sprint"
  recurrent = True

  def __init__(self, sprint_opts, **kwargs):
    """
    :param dict[str] sprint_opts:
    """
    super(ExternSprintLoss, self).__init__(**kwargs)
    self.sprint_opts = sprint_opts
    from TFUtil import custom_gradient
    custom_gradient.register_generic_loss_and_error_signal()

  def get_value(self):
    """
    :rtype: tf.Tensor
    """
    with tf.name_scope("ExternSprintLoss"):
      seq_tags = self.base_network.get_seq_tags()
      assert self.output_with_activation.is_softmax_act_func()
      output_before_softmax = self.output_with_activation.get_logits()
      if not self.output.is_time_major:
        output_before_softmax = swapaxes(output_before_softmax, self.output.time_dim_axis, self.output.batch_dim_axis)
      output = self.output.get_placeholder_as_time_major()
      from TFSprint import get_sprint_loss_and_error_signal
      loss, error_signal = get_sprint_loss_and_error_signal(
        sprint_opts=self.sprint_opts,
        log_posteriors=tf.log(output),
        seq_lengths=self.output_seq_lens,
        seq_tags=seq_tags)
      loss = self.reduce_func(loss)
      from TFUtil import custom_gradient
      loss = custom_gradient.generic_loss_and_error_signal(loss=loss, x=output_before_softmax, grad_x=error_signal)
      return loss

  def get_error(self):
    """
    :rtype: tf.Tensor|None
    """
    if self.target is None:
      return None  # we don't have it
    # Use default frame-wise error to reference target.
    return super(ExternSprintLoss, self).get_error()


class FastBaumWelchLoss(Loss):
  """
  The loss is calculated via :func:`fast_baum_welch`.
  The automata are created by an extern Sprint instance.
  """
  class_name = "fast_bw"
  recurrent = True

  def __init__(self, sprint_opts, **kwargs):
    """
    :param dict[str] sprint_opts:
    """
    super(FastBaumWelchLoss, self).__init__(**kwargs)
    self.sprint_opts = sprint_opts
    from TFUtil import custom_gradient
    custom_gradient.register_generic_loss_and_error_signal()

  def get_value(self):
    """
    :rtype: tf.Tensor
    """
    with tf.name_scope("FastBaumWelchLoss"):
      seq_tags = self.base_network.get_seq_tags()
      assert self.output_with_activation.is_softmax_act_func()
      output_before_softmax = self.output_with_activation.get_logits()
      if not self.output.is_time_major:
        output_before_softmax = swapaxes(output_before_softmax, self.output.time_dim_axis, self.output.batch_dim_axis)
      output = self.output.get_placeholder_as_time_major()
      from TFUtil import sequence_mask_time_major
      seq_mask = sequence_mask_time_major(self.output_seq_lens)
      from TFNativeOp import fast_baum_welch_by_sprint_automata
      fwdbwd, obs_scores = fast_baum_welch_by_sprint_automata(
        sprint_opts=self.sprint_opts,
        am_scores=-tf.log(output),
        float_idx=seq_mask,
        tags=seq_tags)
      loss = self.reduce_func(obs_scores[0])
      bw = tf.exp(-fwdbwd)
      grad_x = (output - bw) * tf.expand_dims(seq_mask, 2)
      from TFUtil import custom_gradient
      loss = custom_gradient.generic_loss_and_error_signal(loss=loss, x=output_before_softmax, grad_x=grad_x)
      return loss

  def get_error(self):
    """
    :rtype: tf.Tensor|None
    """
    if self.target is None:
      return None  # we don't have it
    # Use default frame-wise error to reference target.
    return super(FastBaumWelchLoss, self).get_error()


class ViaLayerLoss(Loss):
  """
  The loss error signal and loss value is defined as the output of another layer.
  That way, you can define any custom loss.
  This could e.g. be used together with the fast_bw layer.
  """
  class_name = "via_layer"
  recurrent = True

  def __init__(self, error_signal_layer=None, align_layer=None, loss_wrt_to_act_in=False, **kwargs):
    """
    :param LayerBase error_signal_layer:
    :param LayerBase align_layer:
    :param bool|str loss_wrt_to_act_in: if True, we expect that the given output_with_activation is
      set, and the given error signal is w.r.t. the input of the specific activation function.
      A common example is the input to the softmax function, where the gradient is much more stable to define,
      e.g. `y - z` instead of `y/z` for cross entropy.
      If you specify a str, e.g. "softmax" or "log_softmax", there is an additional check
      that the used activation function is really that one.
    """
    super(ViaLayerLoss, self).__init__(**kwargs)
    self.error_signal_layer = error_signal_layer
    self.align_layer = align_layer
    self.loss_wrt_to_act_in = loss_wrt_to_act_in
    assert not (error_signal_layer and align_layer)
    assert error_signal_layer or align_layer
    layer = (error_signal_layer or align_layer)
    assert isinstance(layer, LayerBase)
    assert layer.output_loss is not None
    self._loss_value = layer.output_loss
    from TFUtil import custom_gradient
    custom_gradient.register_generic_loss_and_error_signal()

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace, the loss_opts
    :param TFNetwork.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    for key in ["error_signal_layer", "align_layer"]:
      if key in d:
        d[key] = get_layer(d[key])

  def get_value(self):
    """
    :rtype: tf.Tensor
    """
    with tf.name_scope("ViaLayerLoss"):
      if self.error_signal_layer:
        assert not self.align_layer
        error_signal = self.error_signal_layer.output.copy_compatible_to(self.output).placeholder
      else:
        assert self.align_layer
        error_signal = self.output.placeholder - self.align_layer.output.copy_compatible_to(self.output).placeholder
      if self.output.is_time_axis_dynamic():
        seq_mask_bc = self.output.get_sequence_mask_broadcast()
        error_signal = tf.where(
          tf.logical_and(seq_mask_bc, tf.ones_like(error_signal, dtype=tf.bool)),
          error_signal, tf.zeros_like(error_signal))
      if self.loss_wrt_to_act_in:
        assert self.output_with_activation, "activation unknown, via %r" % self.output
        if isinstance(self.loss_wrt_to_act_in, (str, unicode)):
          from TFUtil import get_activation_function
          assert self.output_with_activation.act_func is get_activation_function(self.loss_wrt_to_act_in)
        else:
          assert self.output_with_activation.act_func  # just check that there is some activation function
        grad_wrt = self.output_with_activation.x  # activation (e.g. softmax) input
      else:
        grad_wrt = self.output.placeholder
      from TFUtil import custom_gradient
      loss = custom_gradient.generic_loss_and_error_signal(
        loss=self._loss_value, x=grad_wrt, grad_x=error_signal)
      return loss

  def get_error(self):
    """
    :rtype: tf.Tensor|None
    """
    if self.target is None:
      return None  # we don't have it
    # Use default frame-wise error to reference target.
    return super(ViaLayerLoss, self).get_error()


class AsIsLoss(Loss):
  """
  Use the output as-is as the loss.
  """
  class_name = "as_is"

  def __init__(self, **kwargs):
    super(AsIsLoss, self).__init__(**kwargs)

  def get_value(self):
    """
    :rtype: tf.Tensor
    """
    assert self.output_flat is not None
    return self.reduce_func(self.output_flat)

  def get_error(self):
    """
    :rtype: None
    """
    return None  # not defined

  @classmethod
  def get_default_target(cls, extern_data):
    """
    :param TFNetwork.ExternData extern_data:
    :rtype: None
    """
    # We do not need any target.
    return None


class SamplingBasedLoss(Loss):
  """
  Implement two sampling based losses, sampled softmax (default) and noise contrastive estimation.
  https://www.tensorflow.org/api_docs/python/tf/nn/sampled_softmax_loss.
  https://www.tensorflow.org/api_docs/python/tf/nn/nce_loss.

  Must be used in an output linear layer with a weight matrix of shape (num_classes, dim).
  When using 'log_uniform' sampler (default), optimal performance is typically achieved with the vocabulary list sorted
  in decreasing order of frequency (https://www.tensorflow.org/api_docs/python/tf/random/log_uniform_candidate_sampler).
  """
  class_name = "sampling_loss"

  def __init__(self,
               num_sampled=128,
               num_splits=1,
               sampler="log_uniform",
               nce_loss=False,
               use_full_softmax=False,
               **kwargs):
    """
    :param int num_sampled: Number of classes to be sampled. For sampled softmax, this is the number of classes to be
      used to estimate the sampled softmax. For noise contrastive estimation, this is the number of noise samples.
    :param int num_splits: Number of different samples (each with 'num_sampled' classes) to be used per batch.
    :param str sampler: Specify sampling distribution ("uniform", "log_uniform", or "learned_unigram").
    :param bool nce_loss: If True, use noise contrastive estimation loss. Else (default), use the sampled softmax.
    :param bool use_full_softmax: If True, compute the full softmax instead of sampling (can be used for evaluation).
    """
    super(SamplingBasedLoss, self).__init__(**kwargs)
    assert num_sampled >= 1
    assert sampler in ["uniform", "log_uniform", "learned_unigram"], (
      "Sampler must be one of 'uniform', 'log_uniform', or 'learned_unigram'.")
    self.num_sampled = num_sampled
    self.num_splits = num_splits
    self.sampler = sampler
    self.use_full_softmax = use_full_softmax
    self.nce_loss = nce_loss

  def get_value(self):
    """
    :rtype: tf.Tensor
    """
    assert self.target.sparse
    assert isinstance(self.layer, LinearLayer)
    with tf.name_scope("loss_with_sampling"):
      def sampled_loss_fn():
        """
        Compute sampling based loss.

        :rtype: tf.Tensor
        """
        # Prepare shapes for 'tf.nn.sampled_softmax_loss' and 'tf.nn.nce_loss'.
        labels = self.target_flat  # (B*T|T*B|sum seq len=B',)
        batch_num = tf.shape(labels)[0]  # B'.
        labels = tf.reshape(labels, [-1, 1])  # (B',1).

        input_data = self.layer.input_data
        assert isinstance(input_data, Data)
        inputs = self._flatten_or_merge(
          input_data.placeholder,
          seq_lens=input_data.get_sequence_lengths(),
          time_major=input_data.is_time_major)  # (B',D).

        from tensorflow.python.framework import dtypes
        from tensorflow.python.ops import math_ops
        if labels.dtype != dtypes.int64:
          labels = math_ops.cast(labels, dtypes.int64)

        from tensorflow.python.ops import candidate_sampling_ops
        # Dictionary of available samplers in TensorFlow.
        sampler_dict = {"log_uniform": candidate_sampling_ops.log_uniform_candidate_sampler,
                        "uniform": candidate_sampling_ops.uniform_candidate_sampler,
                        "learned_unigram": candidate_sampling_ops.learned_unigram_candidate_sampler}
        sampler = sampler_dict[self.sampler]

        splits = []
        batch_part = batch_num // self.num_splits  # B'' = B' // num_splits.
        for split_nr in range(self.num_splits):
          if self.num_splits > 1:
            start_frame = split_nr * batch_part
            if split_nr == self.num_splits - 1:
              end_frame = batch_num
            else:
              end_frame = (split_nr + 1) * batch_part
            labels_ = labels[start_frame:end_frame]  # (B'',1).
            inputs_ = inputs[start_frame:end_frame]  # (B'',D).
          else:
            labels_ = labels
            inputs_ = inputs

          # 'sampled_values' is a tuple of (sampled_candidates, true_expected_count, sampled_expected_count).
          # See https://www.tensorflow.org/api_docs/python/tf/random/log_uniform_candidate_sampler.
          sampled_values = sampler(true_classes=labels_,
                                   num_true=1,
                                   num_sampled=self.num_sampled,
                                   unique=True,  # Sampling without replacement.
                                   range_max=self.target.dim)
          if self.nce_loss:
            loss_fn = tf.nn.nce_loss
          else:
            loss_fn = tf.nn.sampled_softmax_loss

          assert self.layer.params["W"].shape[0] == self.target.dim, "Expect weight matrix of shape [num_classes, dim]"
          out = loss_fn(weights=self.layer.params["W"],  # (num_classes,D).
                        biases=self.layer.params["b"],  # (num_classes).
                        labels=labels_,  # (B'',1).
                        inputs=inputs_,  # (B'',D).
                        num_sampled=self.num_sampled,
                        num_classes=self.target.dim,
                        num_true=1,
                        sampled_values=sampled_values,
                        remove_accidental_hits=True,
                        partition_strategy="div",
                        name="sampling_based_loss")  # (B'').
          splits.append(out)
        if len(splits) == 1:
          return splits[0]
        return tf.concat(splits, axis=0)  # (B').

      def full_softmax_fn():
        """
        Compute full softmax.

        :rtype: tf.Tensor
        """
        assert self.target.sparse is True
        if self.output_before_softmax_flat is not None:
          target_flat = self.target_flat
          from TFUtil import to_int32_64
          out = tf.nn.sparse_softmax_cross_entropy_with_logits(
             logits=self.output_before_softmax_flat, labels=to_int32_64(target_flat))
        else:
          print("Warning: using numerical unstable sparse Cross-Entropy loss calculation", file=log.v3)
          raise NotImplementedError
        return out

      if self.use_full_softmax:  # Used instead of slow 'cond_on_train(train_fn, eval_fn)'.
        return self.reduce_func(full_softmax_fn())
      else:
        return self.reduce_func(sampled_loss_fn())


class TripletLoss(Loss):
  """
  Triplet loss: loss = max(margin + d(x_a, x_s) - d(x_a, x_d), 0.0)
  Triplet loss is used for metric learning in a siamese/triplet network.
  It should be used as a part of CopyLayer with 3 inputs corresponding to
    x_a, x_s and x_d in a loss.
  Here we assume that x_a are anchor samples, x_s are samples where
    at each position i in a minibatch x_ai and x_si belong to the same class,
    while pairs x_ai and x_di belong to different classes.
  In this implementation the number of training examples is increased
  by extracting all possible same/different pairs within a minibatch.
  """
  class_name = "triplet_loss"

  def __init__(self, margin, multi_view_training=False, **kwargs):
    super(TripletLoss, self).__init__(**kwargs)
    """
    :param float margin: how much the distance between instances of the same class 
      should be smaller then distances between instances of different classes.
    :param bool multi_view_training: True if we have a pair of inputs (x_a, x_s, x_d) 
      extracted from two different data representations (i.e. acoustic and orthographic)
    """
    self.margin = margin
    self.multi_view = multi_view_training

  def init(self, output, output_with_activation=None, target=None, **kwargs):
    """
    :param Data output: generated output
    :param OutputWithActivation|None output_with_activation:
    :param Data target: reference target from dataset
    """
    super(TripletLoss, self).init(output=output, output_with_activation=output_with_activation, target=target, **kwargs)
    batch_size = tf.cast(tf.shape(self.output.placeholder)[self.output.batch_dim_axis], tf.float32)
    scale_factor = 2.0 if self.multi_view else 1.0
    self.loss_norm_factor = 1.0 / (scale_factor * 9.0 * batch_size)

  def get_value(self):
    """
    :rtype: tf.Tensor
    """
    if self.multi_view:
      with tf.name_scope("multi_view_loss"):
        out = self.output_flat
        assert self.output.dim % 6 == 0
        sources = tf.split(out, num_or_size_splits=6, axis=1)
        targets = self.target_flat
        aembeds_anchor = sources[0]
        aembeds_pair = sources[1]
        aembeds_diff = sources[2]
        # noinspection PyUnusedLocal
        cembeds_anchor = sources[3]
        cembeds_pair = sources[4]
        cembeds_diff = sources[5]
        embeds_1 = tf.concat(values=[aembeds_anchor, cembeds_pair, cembeds_diff], axis=0)
        embeds_2 = tf.concat(values=[aembeds_pair, cembeds_pair, aembeds_diff], axis=0)
        anchor_targets = targets[:, 0]
        pair_targets = targets[:, 1]
        diff_targets = targets[:, 2]
        labels = tf.concat(values=[anchor_targets, pair_targets, diff_targets], axis=0)
        loss_out = self._triplet_loss(embeds_1, labels) + self._triplet_loss(embeds_2, labels)
    else:
      with tf.name_scope("single_view_loss"):
        out = self.output_flat
        assert self.output.dim % 3 == 0
        sources = tf.split(out, num_or_size_splits=3, axis=1)
        targets = self.target_flat
        aembeds_anchor = sources[0]
        aembeds_pair = sources[1]
        aembeds_diff = sources[2]
        embeds = tf.concat(values=[aembeds_anchor, aembeds_pair, aembeds_diff], axis=0)
        anchor_targets = targets[:, 0]
        pair_targets = targets[:, 1]
        diff_targets = targets[:, 2]
        labels = tf.concat(values=[anchor_targets, pair_targets, diff_targets], axis=0)
        loss_out = self._triplet_loss(embeds, labels)

    return loss_out

  def _triplet_loss(self, embeds, labels):
    """
    param tf.Tensor embeds: shape (3*B,F); all embeddings concatenated in batch dim; float32;
    param tf.Tensor labels: shape (3*B,); all output labels concatenated in batch dim; int32
    """
    emb_norm = tf.nn.l2_normalize(embeds, axis=1, epsilon=1e-15)
    sim = tf.matmul(emb_norm, emb_norm, transpose_b=True)
    dist = 1.0 - sim
    labels = tf.expand_dims(labels, 0)
    labels = tf.cast(labels, tf.int32)
    prod = tf.matmul(tf.transpose(labels), labels)
    squer = tf.square(labels)
    same_mask = tf.equal(squer, prod)
    same_indices = tf.where(same_mask)
    diff_mask = tf.logical_not(same_mask)
    diff_indices = tf.where(diff_mask)

    with tf.name_scope("same_loss"):
      same_distances = tf.gather_nd(dist, same_indices)
      same_loss = 0.5 * tf.reduce_sum(same_distances, 0)

    with tf.name_scope("diff_loss"):
      diff_distances = tf.gather_nd(dist, diff_indices)
      # if the distance between embeddings is large than margin => assign zero
      diff_max = tf.maximum(self.margin - diff_distances, 0.0)
      diff_loss = 0.5 * tf.reduce_sum(diff_max, 0)

    return same_loss + diff_loss

  def _check_init(self):
    """
    Does some checks on self.target and self.output, e.g. if the dense shapes matches.
    """
    assert self.target.sparse
    assert self.output.dim % (6 if self.multi_view else 3) == 0

  def get_error(self):
    """
    Error is not defined for triplet_loss
    :return: None
    """
    return None


_LossClassDict = {}  # type: typing.Dict[str,typing.Type[Loss]]


def _init_loss_class_dict():
  from TFNetworkNeuralTransducer import NeuralTransducerLoss

  for v in globals().values():
    if isinstance(v, type) and issubclass(v, Loss) and v.class_name:
      assert v.class_name not in _LossClassDict
      _LossClassDict[v.class_name] = v

  # Outside loss functions
  for v in [NeuralTransducerLoss]:
    if isinstance(v, type) and issubclass(v, Loss) and v.class_name:
      assert v.class_name not in _LossClassDict
      _LossClassDict[v.class_name] = v
  for alias, v in {"sse_sigmoid": BinaryCrossEntropyLoss}.items():
    _LossClassDict[alias] = v


def get_loss_class(loss):
  """
  :param str loss: loss type such as "ce"
  :rtype: (() -> Loss) | type[Loss] | Loss
  """
  if not _LossClassDict:
    _init_loss_class_dict()
  if loss not in _LossClassDict:
    raise Exception("unknown loss class %r" % loss)
  return _LossClassDict[loss]


_LayerClassDictInitialized = False
_LayerClassDict = {}  # type: typing.Dict[str,typing.Type[LayerBase]]


def _init_layer_class_dict():
  global _LayerClassDictInitialized
  _LayerClassDictInitialized = True
  import TFNetworkRecLayer
  import TFNetworkSigProcLayer
  import TFNetworkSegModLayer
  import TFNetworkNeuralTransducer

  auto_register_layer_classes(list(globals().values()))
  for mod in [TFNetworkRecLayer, TFNetworkSigProcLayer, TFNetworkSegModLayer, TFNetworkNeuralTransducer]:
    auto_register_layer_classes(list(vars(mod).values()))

  for alias, v in {"forward": LinearLayer, "hidden": LinearLayer}.items():
    assert alias not in _LayerClassDict
    _LayerClassDict[alias] = v


def auto_register_layer_classes(vars_values):
  """
  Example usage::

      from TFNetworkLayer import auto_register_layer_classes
      auto_register_layer_classes('extern_private/your_stuff/CoolThingy.py')


  :param list|types.ModuleType|str vars_values: e.g. use list(globals().values()).
    str is considered as a module-filename
  :return: nothing
  """
  import inspect
  if isinstance(vars_values, str):
    from Util import generic_import_module
    vars_values = generic_import_module(vars_values)
  if inspect.ismodule(vars_values):
    vars_values = list(vars(vars_values).values())
  for v in vars_values:
    if isinstance(v, type) and issubclass(v, LayerBase) and v.layer_class:
      register_layer_class(v)


def register_layer_class(layer_class):
  """
  Registers a layer class such that it can be used in network construction.

  :param type[LayerBase] layer_class:
  :return: nothing
  """
  assert isinstance(layer_class, type) and issubclass(layer_class, LayerBase) and layer_class.layer_class
  assert _LayerClassDict.get(layer_class.layer_class, None) in [None, layer_class]
  _LayerClassDict[layer_class.layer_class] = layer_class


def get_layer_class(name):
  """
  :param str name: matches layer_class
  :rtype: (() -> LayerBase) | type[LayerBase] | LayerBase
  """
  if not _LayerClassDictInitialized:
    _init_layer_class_dict()
  if name not in _LayerClassDict:
    raise Exception("unknown layer class %r" % name)
  return _LayerClassDict[name]


def get_layer_class_name_list():
  """
  :rtype: list[str]
  """
  if not _LayerClassDictInitialized:
    _init_layer_class_dict()
  return sorted(_LayerClassDict.keys())

