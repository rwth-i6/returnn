
"""
This module contains the layer base class :class:`LayerBase`.
"""

from __future__ import print_function

import typing
import contextlib
import tensorflow as tf
from returnn.util.basic import NotSpecified, CollectionReadCheckCovered, BehaviorVersion
import returnn.tf.compat as tf_compat
import returnn.tf.util.basic as tf_util
from returnn.tf.util.data import Data, FeatureDim
from returnn.tf.util.basic import OutputWithActivation, CustomUpdate, reuse_name_scope
from returnn.log import log


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
              # It is set at this point to whatever we got from `self.get_out_data_from_opts()`,
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
  def __init__(self, name, network, output,
               n_out=NotSpecified, out_dim=None, out_type=None,
               out_shape=None,
               sources=(),
               in_dim=None,
               target=None, _target_layers=None, loss=None, size_target=None,
               reuse_params=None,
               name_scope=None,
               param_device=None,
               is_output_layer=None, only_on_eval=False, only_on_search=False,
               copy_output_loss_from_source_idx=None,
               batch_norm=False,
               L2=None, darc1=None,
               spatial_smoothing=0.0,
               param_variational_noise=None,
               updater_opts=None,
               initial_output=None,
               state=None,
               need_last=False,
               rec_previous_layer=None,
               encapsulate=False,
               collocate_with=None,
               trainable=None,
               custom_param_importer=None,
               register_as_extern_data=None,
               control_dependencies_on_output=None,
               debug_print_layer_output=None,
               _network=None, _name=None,
               _src_common_search_choices=None):
    """
    Usually the arguments, when specified in the network dict,
    are going through :func:`transform_config_dict`, before they are passed to here.
    See :func:`TFNetwork.construct_from_dict`.

    :param str name:
    :param returnn.tf.network.TFNetwork network:
    :param Data output: Set a specific output instead of using :func:`get_out_data_from_opts`
    :param NotSpecified|None|int n_out: output dim
    :param returnn.tf.util.data.Dim|None out_dim: output feature dim tag
    :param dict[str] out_type: kwargs for Data class. more explicit than n_out.
    :param set[returnn.tf.util.data.Dim|returnn.tf.util.data._MarkedDim]|tuple|list|None out_shape:
      verifies the output shape (dim tags). See :func:`Data.verify_out_shape`.
    :param list[LayerBase] sources: via self.transform_config_dict()
    :param returnn.tf.util.data.Dim|None in_dim: input feature dim tag
    :param str|list[str]|None target: if some loss is set, this is the target data-key,
      i.e. network.extern_data.get_data(target). alternatively, this also can be a layer name.
    :param dict[str,LayerBase]|None _target_layers: if target.startswith("layer:"), then this is target -> layer
    :param str|None size_target: like target but this is only used to set our output size in case of training
    :param Loss|None loss: via :func:`transform_config_dict`.
      Every layer can have one loss (of type :class:`Loss`), or none loss.
      In the net dict, it is specified as a string.
      In :class:`TFNetwork`, all losses from all layers will be collected.
      That is what :class:`TFUpdater.Updater` will use for training.
    :param ReuseParams|None reuse_params: if given, will opt reuse the params. see :func:`self.var_creation_scope`.
      See also the ``name_scope`` option as an alternative.
    :param str|None name_scope: If set, uses this custom (relative) name scope.
      If it starts with a "/", it will be the absolute name scope.
      It should not end with a "/".
      It can be empty, in which case it will not consume a new name scope.
      This can also be used for parameter sharing.
      The default is the layer name in most cases,
      but this logic is in :func:`get_absolute_name_scope_prefix` and :func:`TFNetwork.layer_creation_scope`.
    :param str|None param_device: e.g. "CPU", etc. any valid name for tf.device.
      see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/device_name_utils.h
    :param float|None L2: for constraints
    :param float|None darc1: for constraints. see Generalization in Deep Learning, https://arxiv.org/abs/1710.05468
    :param float|None spatial_smoothing: see :func:`returnn.tf.util.basic.spatial_smoothing_energy`
    :param float|None param_variational_noise: adds variational noise to the params during training
    :param dict[str]|None updater_opts: accepts similar opts as TFUpdater, e.g. "optimizer", "learning_rate", ...
    :param bool|None is_output_layer: triggers the construction of this layer in the root net.
      Inside a :class:`RecLayer`, it triggers the explicit accumulation of all frames.
      Also see the ``need_last`` option.
    :param bool only_on_eval: if True, this layer will only be calculated in eval
    :param bool only_on_search: if True, this layer will only be calculated when search is done
    :param int|None copy_output_loss_from_source_idx: if set, will copy output_loss from this source
    :param bool|dict batch_norm: see self.batch_norm()
    :param str|float initial_output: used for recurrent layer, see self.get_rec_initial_output()
    :param state: explicitly defines the rec state. initial_state would define the initial state (in the first frame)
    :param bool need_last: Inside :class:`RecLayer`, make sure that we can access the last frame.
      Similar to ``is_output_layer, but this is specifically about the last frame,
      i.e. it does not trigger accumulation.
    :param LayerBase|None rec_previous_layer: via the recurrent layer, layer (template) which represents the past of us.
      You would not explicitly set this in a config. This is automatically, internally, via :class:`RecLayer`.
    :param bool encapsulate: mostly relevant for SubnetworkLayer and similar:
      If True, all sub layers will be created,
        and covered in functions like :func:`get_rec_initial_extra_outputs`,
        and the logic in :func:`cls_get_sub_network` will not be used.
      If False, the logic in :func:`cls_get_sub_network` will be used.
    :param list[str]|None collocate_with: in the rec layer, collocate with the specified other layers
    :param bool|None trainable: whether the parameters of this layer will be trained.
      default (None) inherits from the parent layer if there is one, or otherwise True.
    :param str|callable|None custom_param_importer: used by :func:`set_param_values_by_dict`
    :param str|None register_as_extern_data: registers output in network.extern_data
    :param None|((LayerBase)->list[tf.Operation]) control_dependencies_on_output:
    :param None|bool|dict[str] debug_print_layer_output: same as global config option but per layer
    :param str _name: just for internal construction, should be the same as ``name``
    :param returnn.tf.network.TFNetwork _network: just for internal construction, should be the same as ``network``
    :param None|SearchChoices _src_common_search_choices: set via :func:`SearchChoices.translate_to_common_search_beam`
    """
    debug_print_layer_output  # noqa  # not used here but in TFNetwork._create_layer
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
    self.size_target = size_target
    self.loss = loss
    if self.loss and self.loss.recurrent:
      self.recurrent = True
    self.output = output
    if n_out is not NotSpecified:
      assert self.output.dim == n_out
    if isinstance(out_type, dict):
      if "shape" in out_type:
        assert self.output.shape == out_type["shape"]
      if "dim" in out_type:
        assert self.output.dim == out_type["dim"]
    if out_dim:
      # When this fails, the layer does not handle it correctly.
      # Note that layers using _base_get_out_data_from_opts should handle it correctly.
      assert out_dim in self.output.dim_tags_set_implicit, (
        "%s: out_dim handling not implemented correctly for this layer" % self)
    out_shape  # noqa  # not used here but in fixup_out_data
    self.output_before_activation = None  # type: typing.Optional[OutputWithActivation]
    self.output_loss = None  # type: typing.Optional[tf.Tensor]
    if copy_output_loss_from_source_idx is not None:
      self.output_loss = sources[copy_output_loss_from_source_idx].output_loss
    self.rec_vars_outputs = {}  # type: typing.Dict[str,tf.Tensor]
    self.search_choices = None  # type: typing.Optional[SearchChoices]
    self._src_common_search_choices = _src_common_search_choices
    self._initial_output = initial_output
    self.need_last = need_last
    self._rec_previous_layer = rec_previous_layer
    self._encapsulate = encapsulate
    self.collocate_with = collocate_with or []
    self.post_init_hooks = []  # list of functions
    self.sources = list(sources)
    if in_dim and len(sources) == 1:
      # Note that this check is somewhat incomplete
      # (does not check multiple sources, see _ConcatInputLayer)
      # and there is no guarantee that a specific layer really uses this correctly.
      assert sources[0].output.have_dim_tag(in_dim, unique=True), (
        "%s: in_dim %s not found or unique in input %s" % (self, in_dim, sources[0]))
    self.params = {}  # type: typing.Dict[str,tf.Variable]
    self.saveable_param_replace = {}  # type:  typing.Dict[tf.Variable,typing.Union['tensorflow.python.training.saver.BaseSaverBuilder.SaveableObject',None]]  # see get_saveable_params_dict()  # nopep8
    self.reuse_params = reuse_params
    self.name_scope = name_scope
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
    if trainable is None:
      trainable = self.network.parent_layer.trainable if self.network.parent_layer else True
    self.trainable = trainable
    self.custom_param_importer = custom_param_importer
    self.control_dependencies_on_output = control_dependencies_on_output
    self.register_as_extern_data = register_as_extern_data
    # Stats will be collected by the engine.
    self.stats = {}  # type: typing.Dict[str,tf.Tensor]
    self._set_prev_state(state)

  def _set_prev_state(self, state):
    if state is None:
      return
    from tensorflow.python.util import nest

    if not self._rec_previous_layer:
      # This is allowed when outside a rec layer.
      # We could use get_rec_initial_extra_outputs but self.kwargs are not available yet...
      # Instead, we rely on this heuristic for now.
      def _map_to_state_tensor_simple(state_layer):
        assert isinstance(state_layer, LayerBase)
        assert state_layer.output.have_batch_axis()
        assert state_layer.output.batch_ndim <= 2, "%s with state %s expects to operate a single step" % (self, state)
        return state_layer.output.copy_as_batch_major().placeholder

      self._rec_previous_layer = InternalLayer(
        name="prev-dummy:%s" % self.name, network=self.network, output=self.output)
      self._rec_previous_layer.rec_vars_outputs["state"] = nest.map_structure(
        _map_to_state_tensor_simple, state)
      return

    def _map_to_state_tensor(orig_state, state_layer):
      assert isinstance(orig_state, tf.Tensor)
      if state_layer is None:  # this is allowed, can be partial
        return orig_state
      assert isinstance(state_layer, LayerBase)
      assert orig_state.shape.as_list() == list(state_layer.output.batch_shape)
      return state_layer.output.placeholder

    if set(self._rec_previous_layer.rec_vars_outputs.keys()) == {"state"}:
      rec_prev_layer_state = self._rec_previous_layer.rec_vars_outputs["state"]
      nest.assert_same_structure(rec_prev_layer_state, state)
      self._rec_previous_layer.rec_vars_outputs["state"] = nest.map_structure(
        _map_to_state_tensor, rec_prev_layer_state, state)
      return
    raise NotImplementedError(
      "%s: explicit 'state' %r, internal states %r" % (self, state, self._rec_previous_layer.rec_vars_outputs))

  def post_init(self, layer_desc):
    """
    This gets called right after self.__init__().

    :param dict[str] layer_desc: kwargs as they are passed to self.__init__
    """
    self.kwargs = layer_desc.copy()
    assert "output" in self.kwargs
    self.kwargs.setdefault("name", self.name)
    if self.output.placeholder is not None:  # unset e.g. in DataNotAvailableLayer
      if self.use_batch_norm:
        opts = {}
        if isinstance(self.use_batch_norm, dict):
          opts = self.use_batch_norm
        self.output.placeholder = self.batch_norm(self.output, **opts)
      if self.control_dependencies_on_output:
        control_deps = self.control_dependencies_on_output(self)
        if not isinstance(control_deps, (list, tuple)):
          assert isinstance(control_deps, (tf.Operation, tf.Tensor))
          control_deps = [control_deps]
        assert all([isinstance(dep, (tf.Operation, tf.Tensor)) for dep in control_deps])
        if control_deps:
          with tf.control_dependencies(control_deps):
            self.output.placeholder = tf.identity(self.output.placeholder)
    if self.register_as_extern_data:
      self.network.extern_data.extra_added_keys.add(self.register_as_extern_data)
      self.network.extern_data.data[self.register_as_extern_data] = self.output
    for func in self.post_init_hooks:
      func()

  def __repr__(self):
    return "<%s %s%r out_type=%s>" % (
      self.__class__.__name__, self.network.get_absolute_name_prefix(), self.name,
      self.output.get_description(with_name=False) if self.output else None)

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
  def _base_get_out_data_from_opts(cls, network, name,
                                   out_type=None, out_dim=None, n_out=NotSpecified,
                                   out_shape=None,
                                   target=None, _target_layers=None, size_target=None,
                                   sources=(), in_dim=None, loss=None,
                                   **kwargs):
    """
    Called via BaseLayer.get_out_data_from_opts().

    :param returnn.tf.network.TFNetwork network:
    :param str name:
    :param dict[str]|None|(()->Data) out_type:
    :param returnn.tf.util.data.Dim|None out_dim:
    :param int|None|NotSpecified n_out:
    :param set[Dim|_MarkedDim]|tuple|list|None out_shape: verifies the output shape (dim tags).
    :param str|list[str]|None target:
    :param dict[str,LayerBase]|None _target_layers: if target.startswith("layer:"), then this is target -> layer
    :param str|None size_target:
    :param list[LayerBase] sources:
    :param Dim|None in_dim:
    :param Loss|None loss:
    :param kwargs: remaining kwargs of self.__init__(), ignored here
    :return: Data template (placeholder not set)
    :rtype: Data
    """
    if callable(out_type):
      return out_type(
        network=network, name=name, n_out=n_out, target=target, size_target=size_target, sources=sources, loss=loss,
        **kwargs)
    # Any template construction should be aware of that, and eventually resolve it.
    if out_type is None:
      out_type = {}  # type: typing.Dict[str]
    else:
      out_type = out_type.copy()
    out_type.setdefault("name", "%s_output" % name)
    if "dim" not in out_type and n_out is not NotSpecified:
      out_type["dim"] = n_out
    if "dim" not in out_type and target and not out_dim:
      out_type["dim"] = cls._static_get_target_value(
        target=target[0] if isinstance(target, list) else target, _target_layers=_target_layers,
        network=network, mark_data_key_as_used=False).dim
    if n_out is not NotSpecified:
      assert out_type["dim"] == n_out
    sources_data_list = [src.output for src in sources if src]
    if in_dim:
      assert len(sources_data_list) == 1, "%r: with specific in_dim %s, there must be a single source" % (name, in_dim)
      if sources_data_list[0].feature_dim_or_sparse_dim != in_dim:
        # Allow to specify some in_dim which is not the feature dim.
        # However, the follow-up code will expect it to be the feature dim, thus reassign it if possible.
        assert in_dim in sources_data_list[0].dim_tags
        axis = sources_data_list[0].get_axis_from_description(in_dim)
        sources_data_list = [sources_data_list[0].copy()]
        sources_data_list[0].feature_dim_axis = axis
    allow_broadcast_all_sources = NotSpecified
    if "shape" in out_type or "dim_tags" in out_type or out_shape is not None:
      allow_broadcast_all_sources = True
    sources_data = Data.get_common_data(
      sources_data_list, ignore_feature_dim=True,
      allow_broadcast_all_sources=allow_broadcast_all_sources, name="%s_sources" % name) if sources_data_list else None
    if sources_data and not sources_data.sparse and not out_type.get("sparse", False):
      out_type.setdefault("dtype", sources_data.dtype)
    # You are supposed to set self.output.{batch_dim_axis,time_dim_axis} explicitly,
    # as well as check the inputs if they are as you would suggest.
    # However, a good default is often to use the same as the input.
    if all([k not in out_type for k in Data.SpecialAxesNames + ("dim_tags", "shape")]):
      if sources_data:
        out_type.setdefault("batch_dim_axis", sources_data.batch_dim_axis)
        out_type.setdefault("time_dim_axis", sources_data.time_dim_axis)
        if (
              not out_type.get("sparse", False) and
              not out_type.get("sparse_dim", None) and
              sources_data.feature_dim_axis_or_unspecified is not NotSpecified):
          if sources_data.feature_dim_axis_or_unspecified is not None:
            out_type.setdefault("feature_dim_axis", sources_data.feature_dim_axis_or_unspecified)
          else:  # None
            if out_type.get("dim", None) is None:
              out_type.setdefault("feature_dim_axis", None)
      elif network.is_inside_rec_layer() and None not in out_type.get("shape", ()):
        out_type.setdefault("time_dim_axis", None)
    if "shape" not in out_type and "dim_tags" not in out_type:
      if sources_data:
        if out_type.get("sparse", False) or out_type.get("sparse_dim", None):
          out_type["dim_tags"] = sources_data.dim_tags_sparse
        else:  # not sparse
          feature_dim_axis = out_type.get("feature_dim_axis", NotSpecified)
          dim_tags = list(sources_data.dim_tags_sparse)
          if out_dim:
            feature_dim_tag = out_dim
          else:
            dim = out_type.get("dim", None)
            feature_dim_tag = FeatureDim("%s:feature-dense" % name, dim, auto_generated=True)
          if feature_dim_axis in (NotSpecified, None):
            if sources_data.feature_dim_axis is None:
              feature_dim_axis = len(dim_tags)
            else:
              feature_dim_axis = sources_data.feature_dim_axis
          dim_tags.insert(feature_dim_axis, feature_dim_tag)
          out_type["dim_tags"] = dim_tags
      elif network.is_inside_rec_layer():
        if out_type.get("sparse", False):
          out_type.setdefault("shape", ())
        else:
          out_type.setdefault("shape", (out_type.get("dim", None),))
    # Note: No special handling for feature_dim_axis here for now...
    if sources_data and sources_data.batch:
      out_type.setdefault("batch", sources_data.batch)
    if sources_data and sources_data.beam:
      out_type.setdefault("beam", sources_data.beam)
    if out_dim:
      out_type.setdefault("dim", out_dim.dimension)  # e.g. needed when sparse
    output = Data(**out_type)
    if not out_dim and sources_data and sources_data.feature_dim_or_sparse_dim and sources_data.dim == output.dim:
      # Special case: Input feature or sparse dim looks the same, so overtake it.
      out_dim = sources_data.feature_dim_or_sparse_dim
    if out_dim:
      assert out_dim.dimension == output.dim, (
        "Layer %r out_dim %s does not match Data via out_type %s" % (name, out_dim, output))
      if output.sparse:
        output.sparse_dim = out_dim
      else:
        output = output.copy_template_replace_dim_tag(axis=output.feature_dim_axis, new_dim_tag=out_dim)
    cls._post_init_output(
      output=output, network=network, target=target, size_target=size_target, _target_layers=_target_layers,
      sources=sources, **kwargs)
    return output

  # noinspection PyUnusedLocal
  @classmethod
  def _post_init_output(cls, output, network, target=None, size_target=None, _target_layers=None, sources=(),
                        _src_common_search_choices=None,
                        **kwargs):
    """
    :param Data output:
    :param returnn.tf.network.TFNetwork network:
    :param str|list[str]|None target:
    :param str|None size_target:
    :param dict[str,LayerBase]|None _target_layers: if target.startswith("layer:"), then this is target -> layer
    :param None|SearchChoices _src_common_search_choices: set via :func:`SearchChoices.translate_to_common_search_beam`
    :param list[LayerBase] sources:
    """
    # You are supposed to set self.output.placeholder to the value which you want to return by the layer.
    # Normally you are also supposed to set self.output.size_placeholder explicitly, just like self.output.placeholder.
    # However, in many cases, this will just be {0: time-lengths} and the same as from the input.
    # We check for this case and preset it by that if possible.
    # If you want to have it different in your layer, just overwrite it.
    common_source = Data.get_common_data(
      [s.output for s in sources if s], ignore_feature_dim=True, allow_broadcast_all_sources=True)
    if not output.size_placeholder:
      if network.eval_flag and size_target:
        output.size_placeholder = cls._static_get_target_value(
          target=size_target,
          _target_layers=_target_layers,
          search_choices=_src_common_search_choices,
          network=network, mark_data_key_as_used=network.eval_flag).size_placeholder.copy()
      elif common_source and common_source.matches_var_dim_pattern(output):
        output.size_placeholder = common_source.size_placeholder.copy() if common_source.size_placeholder else {}
      elif network.train_flag is not False and target:
        # TODO: In training, this is ok. Maybe as well as for eval but not clear.
        # In forward, mark_data_key_as_used=False should be used and anyway that target value is not available.
        output.size_placeholder = cls._static_get_target_value(
          target=(target[0] if (target and isinstance(target, list)) else target),
          _target_layers=_target_layers,
          search_choices=_src_common_search_choices,
          network=network, mark_data_key_as_used=network.train_flag is not False).size_placeholder.copy()
    if any([(src and not src.output.available_for_inference) for src in sources if src]):
      output.available_for_inference = False

  @classmethod
  def fixup_out_data(cls, output, network, out_shape=None, **kwargs):
    """
    This is called after get_out_data_from_opts, to fixup incomplete information.
    E.g. we can patch batch or beam information here
    but maybe also other things.

    Other layer classes might overwrite this but then should call this super method.
    Usually this should not be needed though.

    :param Data output:
    :param returnn.tf.network.TFNetwork network:
    :param set[Dim|_MarkedDim]|tuple|list|None out_shape: verifies the output shape (dim tags).
      See :func:`Data.verify_out_shape`.
    :rtype: Data
    """
    from tensorflow.python.util import nest
    from ..util.data import BatchInfo
    from ..network import ExternData
    if not output.batch:
      # In all cases set output.batch, even if the output has no batch dim,
      # as this is important in Data._adapt_batch_consistent_dim_tags().

      def _set_global_batch_by_data(data):
        """
        :param Data data:
        :rtype: returnn.tf.util.data.BatchInfo
        """
        assert data.placeholder is not None and not data.beam
        # Create dummy extern data with new global batch info.
        extern_data = ExternData()
        extern_data.data["_fixup_out_data_dummy_input_" + data.name] = data
        assert data.available_for_inference
        extern_data.init_batch_info()  # this should create it and also set it
        assert data.batch
        return data.batch

      # Some heuristic for now to fix missing batch info. We should try to fix get_out_data_from_opts though...
      dep_layers = [v for v in nest.flatten(kwargs) if isinstance(v, LayerBase)]
      dep_batches = [dep.output.batch for dep in dep_layers if dep.output.batch]
      dyn_dim_tags_with_batch = [
        dim_tag for dim_tag in output.dim_tags
        if dim_tag.dyn_size_ext and dim_tag.dyn_size_ext.have_batch_axis()]
      dim_tags_with_batch_info = [dim_tag for dim_tag in output.dim_tags if dim_tag.batch]
      if dep_batches:
        output.batch = BatchInfo.get_common_batch_info(dep_batches).copy_set_beam(output.beam)
      elif network.extern_data.get_batch_info(allow_none=True):
        output.batch = network.extern_data.get_batch_info().copy_set_beam(output.beam)
      elif network.parent_net and network.get_root_network().extern_data.get_batch_info(allow_none=True):
        output.batch = network.get_root_network().extern_data.get_batch_info().copy_set_beam(output.beam)
      elif dim_tags_with_batch_info:
        output.batch = dim_tags_with_batch_info[0].batch.copy_set_beam(output.beam)
      elif dyn_dim_tags_with_batch:
        for tag in dyn_dim_tags_with_batch:
          if tag.dyn_size_ext.batch:
            output.batch = tag.dyn_size_ext.batch.copy_set_beam(output.beam)
            break
          batch_dim_tag = tag.dyn_size_ext.dim_tags[tag.dyn_size_ext.batch_dim_axis]
          if batch_dim_tag.batch:
            output.batch = batch_dim_tag.batch
            break
        if not output.batch and dyn_dim_tags_with_batch[0].dyn_size_ext.have_batch_axis():
          output.batch = _set_global_batch_by_data(dyn_dim_tags_with_batch[0].dyn_size_ext)
      elif output.placeholder is not None and output.have_batch_axis():
        # No layers at all yet. This implies that the output must already have a placeholder.
        output.batch = _set_global_batch_by_data(output)
    if output.batch:
      output.batch = output.batch.copy_set_beam(output.beam)
    if output.control_flow_ctx != network.get_control_flow_ctx():
      x = output.placeholder
      output = output.copy_template_set_ctx(network.get_control_flow_ctx())
      if x is not None:
        # Some layers might just copy the input. But the input might have buggy ctx.
        # Just leave the placeholder as-is. Most layers should anyway reset this.
        output.placeholder = x
    if out_shape is not None:
      output.verify_out_shape(out_shape)
    return output

  @classmethod
  def get_global_layer_list(cls):
    """
    :rtype: list[LayerBase]
    """
    from returnn.tf.util.basic import CollectionKeys
    coll = tf_compat.v1.get_collection_ref(CollectionKeys.RETURNN_LAYERS)
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
    :param returnn.tf.network.TFNetwork network:
    :param returnn.tf.network.GetLayer|((str)->LayerBase) get_layer: function to get or construct another layer
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
    from .basic import get_loss_class
    from ..network import LayerNotFound
    BehaviorVersion.require(
      condition="from" in d,
      message='Missing "from" in layer definition: %s/%s' % (network.name, d.get("_name", "<UNKNOWN>")),
      version=1)
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
      d["collocate_with"] = collocate_with  # not get_layer: we don't really want this dependency
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
      # _target_layers is a small workaround for further code which might not have access to the right get_layer.
      d["_target_layers"] = target_layers
      for target in targets:
        assert isinstance(target, str)
        # Not resolving this in the dict as target, but call get_layer to make it available.
        if target.startswith("layer:"):
          target_layers[target] = get_layer(target[len("layer:"):])
        else:
          try:
            # Check whether the target itself is a layer
            target_layers[target] = get_layer(target)
          except LayerNotFound:
            # Note: This is a workaround for cases where we need to know about used data keys before the layer
            # itself is constructed (e.g. in _SubnetworkRecCell.get_output).
            # A nicer solution would be to not modify this here,
            # but instead lazily handle it in TFNetwork.get_extern_data,
            # such that we do not need to know in advance which data keys we need.
            # Also, if we are inside a rec layer, and doing search, we also cannot do that.
            if network.is_inside_rec_layer() and not network.search_flag:
              network.get_extern_data(target, mark_data_key_as_used=network.eval_flag)
            if not network.search_flag:
              # Also, there are cases when we want to have the target as an explicit layer dep,
              # e.g. when the target has a beam, to derive the search choices.
              target_layers[target] = get_layer("data:%s" % target)
    if d.get("initial_output", None):  # see get_rec_initial_output
      initial_output = d["initial_output"]
      if isinstance(initial_output, str):
        if initial_output not in ["zeros", "ones", "var", "keep_over_epoch", "keep_over_epoch_no_init", "apply(0)"]:
          # if initial_output is not a reserved keyword, assume it is a layer
          d['initial_output'] = get_layer(initial_output)
    if "n_out" not in d and "out_dim" not in d and targets:
      # Must be done here now because loss might be set to None later.
      target = targets[0]  # guess using first target
      guessed_out_dim = cls._guess_out_dim_from_target_and_opt_loss(
        network=network, target=target, target_layers=target_layers,
        loss_class_name=d.get("loss", None), get_layer=get_layer)
      if guessed_out_dim:
        if cls.layer_class in {"linear", "softmax"}:
          d["out_dim"] = guessed_out_dim
        else:
          # Many layers don't introduce a new out_dim (e.g. activation, copy, etc),
          # and setting out_dim would break many old configs.
          d["n_out"] = guessed_out_dim.dimension
    if "out_shape" in d:
      inside_rec_time_dim = network.get_inside_rec_time_dim(inside_loop=True)
      over_rec_time_dim = network.get_inside_rec_time_dim(inside_loop=False)
      if over_rec_time_dim and not inside_rec_time_dim:  # moved out of loop
        # noinspection PyProtectedMember
        from returnn.tf.util.data import OptionalDim, _MarkedDim
        out_shape = d["out_shape"]
        if not isinstance(out_shape, set):
          assert not out_shape, "out_shape %r must be empty if not a set" % (out_shape,)
        out_shape = set(out_shape)
        out_shape.add(OptionalDim(over_rec_time_dim))
        if over_rec_time_dim.dyn_size_ext:
          for tag in over_rec_time_dim.dyn_size_ext.dim_tags:
            if tag not in [d.tag if isinstance(d, _MarkedDim) else d for d in out_shape]:
              out_shape.add(OptionalDim(tag))
        d["out_shape"] = out_shape
    if d.pop("loss_only_on_non_search", None) and network.search_flag:
      d.pop("loss", None)
      d.pop("loss_scale", None)
      d.pop("loss_opts", None)
    if d.get("loss", None):
      loss_opts = d.pop("loss_opts", None)
      loss_opts = loss_opts.copy() if loss_opts else {}
      # loss_scale: scale factor for loss (1.0 by default). DEPRECATED: use loss.scale instead, via loss_opts
      loss_scale = d.pop("loss_scale", 1.0)
      if loss_scale != 1.0:
        if "scale" in loss_opts:
          assert loss_opts["scale"] == loss_scale, "do not use loss_scale and loss with 'scale' option together"
        loss_opts["scale"] = loss_scale
      d["loss"] = cls._make_loss(
        class_name=d.pop("loss", None), opts=loss_opts, network=network, get_layer=get_layer)
    else:
      if "loss_scale" in d and d["loss_scale"] is None:
        d.pop("loss_scale")
      if "loss_opts" in d and d["loss_opts"] is None:
        d.pop("loss_opts")
      assert "loss_scale" not in d, "loss not defined, do not set loss_scale"
      assert "loss_opts" not in d, "loss not defined, do not set loss_opts"
    root_ctx_net, prefix = d["_network"].get_root_ctx_network()
    rec_previous_layer = root_ctx_net.layers.get("prev:%s%s" % (prefix, d["_name"]))
    if rec_previous_layer:
      d["rec_previous_layer"] = rec_previous_layer
    if d.get("state", None) is not None:
      from tensorflow.python.util import nest
      d["state"] = nest.map_structure(get_layer, d["state"])

  @classmethod
  def _guess_out_dim_from_target_and_opt_loss(cls, network, target, target_layers, loss_class_name, get_layer):
    """
    :param returnn.tf.network.TFNetwork network:
    :param str target: e.g. "classes"
    :param dict[str,LayerBase] target_layers:
    :param str|None loss_class_name: e.g. "ce" or None
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    :return: out_dim value
    :rtype: returnn.tf.util.data.Dim|None
    """
    from .basic import get_loss_class
    if target in target_layers:
      target_data = target_layers[target].output
    else:
      target_data = cls._static_get_target_value(
        target=target, network=network, mark_data_key_as_used=False, get_layer=get_layer, _target_layers=target_layers)
      if not target_data:
        # dummy value during template construction. this would be corrected later
        return FeatureDim("dummy-unk-target-out", 1)
    out_dim = target_data.feature_dim_or_sparse_dim
    if not out_dim:
      return None
    if loss_class_name:
      out_dim = get_loss_class(loss_class_name).get_auto_output_layer_dim(out_dim)
    return out_dim

  @classmethod
  def _make_loss(cls, class_name, opts, network, get_layer, always_make=False):
    """
    :param str|None class_name:
    :param dict[str]|None opts:
    :param returnn.tf.network.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    :param bool always_make:
    :rtype: Loss|None
    """
    from .basic import get_loss_class
    if not network.eval_flag and not always_make:
      # Don't resolve the loss opts on purpose.
      # This might result in a smaller network because it might skip some get_layer calls.
      # This is what we want, i.e. we don't want to resolve layers which are only needed for the loss.
      return None
    if not class_name:
      assert not always_make
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

  def get_full_ctx_name(self):
    """
    :return: name w.r.t. root ctx network
    """
    _, prefix = self.network.get_root_ctx_network()
    return prefix + self.name

  @classmethod
  def cls_get_tf_scope_name(cls, name):
    """
    :param str name: layer name
    :return: valid scope name, might be just name. see tf._VALID_SCOPE_NAME_REGEX and tf._VALID_OP_NAME_REGEX
    :rtype: str
    """
    from returnn.tf.util.basic import get_valid_scope_name_from_str
    return get_valid_scope_name_from_str(name)

  @classmethod
  @contextlib.contextmanager
  def cls_setup_scope(cls, name, name_scope=None, **_kwargs):
    """
    :param str name:
    :param str|None name_scope:
    :param _kwargs: other layer kwargs after being transformed
    """
    scope = cls.cls_get_tf_scope_name(name)
    name_scope_abs = None
    if name_scope is not None:
      if name_scope == "":
        scope = tf_compat.v1.get_variable_scope()
      elif name_scope.startswith("/"):
        name_scope_abs = True
        scope = name_scope[1:]
      else:
        scope = name_scope
      if isinstance(scope, str):
        assert not scope.endswith("/"), "invalid name_scope %r" % name_scope
    with reuse_name_scope(scope, absolute=name_scope_abs):
      yield

  @property
  def tf_scope_name(self):
    """
    :rtype: str
    :return: normally just self.name, but make it a valid TF scope name.
      this is meant mostly to extend TF names. see func:`get_base_absolute_name_scope_prefix` otherwise.
    """
    if self.name_scope and not self.name_scope.startswith("/"):
      assert not self.name_scope.endswith("/")
      return self.name_scope
    return self.cls_get_tf_scope_name(name=self.name)

  def get_base_absolute_name_scope_prefix(self):
    """
    :return: e.g. "output/", always with "/" at end, or "". this is for the TF name scope or variable scope
    :rtype: str
    """
    if self.name_scope is not None:
      if self.name_scope == "":
        return self.network.get_absolute_name_scope_prefix()
      elif self.name_scope == "/":  # absolute, root
        return ""
      elif self.name_scope.startswith("/"):  # absolute
        assert not self.name_scope[1:].endswith("/")
        return self.name_scope[1:] + "/"
      else:
        assert not self.name_scope.endswith("/")
        return self.network.get_absolute_name_scope_prefix() + self.name_scope + "/"
    return self.network.get_absolute_name_scope_prefix() + self.tf_scope_name + "/"

  def get_absolute_name_scope_prefix(self):
    """
    :return: e.g. "output/", always with "/" at end, or "". this is for the TF name scope or variable scope.
      This is the same as :func:`get_base_absolute_name_scope_prefix` in most cases,
      but some layers like :class:`RecLayer` extend this by an additional postfix.
    :rtype: str
    """
    return self.get_base_absolute_name_scope_prefix()

  def get_absolute_name(self):
    """
    :return: e.g. "output" or "subnet/output". This is mostly for representation.
      See also :func:`get_absolute_name_scope_prefix`.
    :rtype: str
    """
    return self.network.get_absolute_name_prefix() + self.name

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
    if self.get_full_ctx_name() == "output":
      return True
    return False

  def get_dep_layers(self):
    """
    :return: list of layers this layer depends on.
      normally this is just self.sources but e.g. the attention layer in addition has a base, etc.
    :rtype: list[LayerBase]
    """
    layers = list(self.sources)
    if self._target_layers:
      layers += [layer for _, layer in sorted(self._target_layers.items())]
    return layers

  # noinspection PyUnusedLocal
  @classmethod
  def cls_get_sub_network(cls, name, network, layer_desc):
    """
    A layer class can override this to return a custom :class:`Subnetwork`,
    which just sets another namespace (and possibly variable sharing)
    for contained layers but otherwise shares the same construction logic
    via root network :func:`TFNetwork.construct_layer`.

    When not overriding this, a layer still can have sub layers
    via :func:`LayerBase.get_sub_layer`, but they belong to the root layer
    (collocated) and can not be decoupled.

    :param str name:
    :param returnn.tf.network.TFNetwork network:
    :param dict[str] layer_desc:
    :rtype: returnn.tf.network.Subnetwork|None
    """
    return None

  def get_sub_layer(self, layer_name):
    """
    The default behavior for any layer is to return None.
    Returned layers belong to the root layer (self).

    Also see :func:`LayerBase.cls_get_sub_network`.

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
    :return: Data template, class type of sub-layer, layer opts (transformed)
    :rtype: (Data, type, dict[str])|None
    """
    return None

  def get_sub_networks(self):
    """
    :return: All subnetworks, including those which might be in a different ctx.
      If this returns a non-empty list, we expect that all layers via get_sub_layers
      can be reached via the subnetworks.
    :rtype: list[returnn.tf.network.TFNetwork]
    """
    return []

  def get_sub_layers(self):
    """
    :return: All (direct) (non-temporary) sub layers, including those which might be in a different ctx.
      This is mostly intended to collect params.
    :rtype: list[LayerBase]
    """
    return []

  def get_search_choices(self):
    """
    :rtype: SearchChoices|None
    """
    if self.search_choices:
      return self.search_choices
    if self._src_common_search_choices:
      return self._src_common_search_choices
    if not self.output.beam:  # small optimization
      # Note that the logic to determine self.output.beam (via get_out_data_from_opts) is currently
      # independently implemented from the search choices (via get_search_choices),
      # and due to bugs, it could happen that self.output.beam is not set but actually we have search choices.
      # This will likely be caught at some later check.
      return None
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
    if self.output.beam:
      return self.output.beam.beam_size
    return None

  def get_normalized_layer(self):
    """
    :return: e.g. if prev layer in :class:`RecLayer`, return current layer
    :rtype: LayerBase
    """
    return self

  def get_batch_dim(self):
    """
    The batch dim by this layer, not taken from our output placeholder but calculated.
    Normally it is self.network.get_batch_dim()
    but if we do search and there was a choice layer, it it multiplied by the beam size.

    :return: batch dim * beam size
    :rtype: tf.Tensor|int
    """
    return self.get_batch_info().dim

  def get_batch_info(self):
    """
    :rtype: returnn.tf.util.data.BatchInfo
    """
    if self.output.batch and self.output.batch.beam == self.output.beam:
      return self.output.batch
    # Fallback.
    batch = self.network.get_global_batch_info()
    if self.output.beam:
      batch = batch.copy_extend_with_beam(self.output.beam)
    return batch

  @contextlib.contextmanager
  def var_creation_scope(self, **kwargs):
    """
    This takes care of setting up a scope where variables can be created.
    This handles multiple things:

     * the param sharing logic, to reuse existing variables from elsewhere
     * variational noise
     * Note: :func:`default_control_flow_ctx` is not needed for tf.get_variable.
       But it might be needed for other code which uses custom inits and tf.Variable,
       e.g. tf.random.Generator.
       However, always using this could be a problem if we use other input tensors inside this scope,
       so we do not enable this here.

    :param kwargs: passed to variable_scope
    :return: yields the variable_scope
    """
    from returnn.tf.util.basic import get_current_var_scope_name, reuse_name_scope
    from returnn.tf.util.basic import default_control_flow_ctx, reuse_name_scope_of_tensor
    self_base_scope = self.get_base_absolute_name_scope_prefix()
    assert self_base_scope.endswith("/") or self_base_scope == ""
    cur_scope = get_current_var_scope_name()
    assert (cur_scope + "/").startswith(self_base_scope)
    # There are cases were a dummy layer was created already to create the variables,
    # e.g. see ReuseParams.LazyLayerResolver.
    kwargs = kwargs.copy()
    kwargs.setdefault("reuse", getattr(tf_compat.v1, "AUTO_REUSE", None))

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
        with default_control_flow_ctx():  # make independent from loop/cond
          with reuse_name_scope_of_tensor(param, postfix="_variational_noise", add_tensor_name=True):
            param = self.network.cond_on_train(
              fn_train=lambda: param + tf_compat.v1.random_normal(
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
        var_scope = tf_compat.v1.get_variable_scope()
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
      # This can happen with a custom_getter in tf.compat.v1.get_variable(), e.g. via self.reuse_params.
      # Check if we can still find the original variable.
      from returnn.extern import graph_editor
      import re
      possible_params = tf_compat.v1.get_collection(
        tf_compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=re.escape(self.get_absolute_name_scope_prefix()))
      if not possible_params:
        # None found. Just return as-is.
        return param
      all_ops = graph_editor.get_backward_walk_ops([param.op], inclusive=False, control_inputs=False)
      all_1st_tensors = [op.outputs[0] for op in all_ops if len(op.outputs) == 1]
      # noinspection PyProtectedMember
      possible_params = [p for p in possible_params if tf_util.var_handle_or_ref(p) in all_1st_tensors]
      if not possible_params:
        # Not found. Just return as-is.
        return param
      assert len(possible_params) == 1
      param = possible_params[0]
    assert isinstance(param, tf.Variable)
    if not self.trainable:
      trainable_collection_ref = tf_compat.v1.get_collection_ref(tf_compat.v1.GraphKeys.TRAINABLE_VARIABLES)
      if param in trainable_collection_ref:
        trainable_collection_ref.remove(param)
    if trainable is None:
      trainable = param in tf_compat.v1.get_collection_ref(tf_compat.v1.GraphKeys.TRAINABLE_VARIABLES)
    if saveable is None:
      saveable = True
    if custom_update:
      assert trainable
      custom_update.set_on_var(param)
    if axes_split_info:
      tf_util.set_param_axes_split_info(param, axes_split_info)
    name_scope_prefix = self.get_absolute_name_scope_prefix()
    if self.reuse_params:
      if not param.name.startswith(name_scope_prefix):
        # We likely used an existing var from elsewhere. Return as-is.
        return _param
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
    :param tf.compat.v1.Session session:
    """
    if callable(self.custom_param_importer):
      self.custom_param_importer(layer=self, values_dict=values_dict, session=session)
      return
    if self.custom_param_importer:
      copy_param_mode = self.custom_param_importer
    if copy_param_mode == "reset":
      return  # just ignore. the random init was already done
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
          param_axes_split_info = tf_util.get_param_axes_split_info(param)
          if param_axes_split_info:
            tf_util.check_param_axes_split_info(param.get_shape().as_list(), param_axes_split_info)
            old_axes_splits = tf_util.transform_param_axes_split_info_to_new_shape(
              param_axes_split_info, values.shape, debug_name="param %r" % param.name)
            print("Param %r: transform old values of shape parts %r into new shape parts %r." % (
              param, old_axes_splits, param_axes_split_info), file=log.v3)
            values = tf_util.copy_with_new_split_axes(
              old_axis_splits=old_axes_splits, new_axis_splits=param_axes_split_info,
              old_values=values, new_values=new_values)
          else:
            print("Param %r: transform old values of shape %r into new shape %r." % (
              param, values.shape, param_shape), file=log.v3)
            values = tf_util.copy_with_new_split_axes(
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
    :param tf.compat.v1.Session session:
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
  def _static_get_target_value(
    target, network, mark_data_key_as_used=True,
    _target_layers=None, get_layer=None,
    search_choices=None
  ):
    """
    :param str target:
    :param dict[str,LayerBase]|None _target_layers: if target.startswith("layer:"), then this is target -> layer
    :param returnn.tf.network.TFNetwork network:
    :param bool mark_data_key_as_used: forwarded self.network.get_extern_data()
    :param None|((str) -> LayerBase) get_layer: function to get or construct another layer
    :param SearchChoices|None search_choices:
    :rtype: Data | None
    """
    if not target or target == "none":
      return None
    from .basic import SelectSearchSourcesLayer
    if _target_layers and target in _target_layers:
      return SelectSearchSourcesLayer.select_if_needed(
        _target_layers[target], search_choices=search_choices).output
    if target.startswith("layer:"):
      if not get_layer:
        get_layer = network.get_layer
      layer = get_layer(target[len("layer:"):])
      if not layer:  # some get_layer during temp construction might return None
        return None
      return SelectSearchSourcesLayer.select_if_needed(layer, search_choices=search_choices).output
    assert network.extern_data.has_data(target), "target %r unknown" % target
    data = network.get_extern_data(target, mark_data_key_as_used=mark_data_key_as_used)
    if search_choices:
      data = data.copy_extend_with_beam(search_choices.get_beam_info())
    return data

  def _get_target_value(self, target=None, mark_data_key_as_used=True, search_choices=NotSpecified):
    """
    :param str|None target:
    :param bool mark_data_key_as_used: forwarded self.network.get_extern_data()
    :param SearchChoices|NotSpecified|None search_choices:
    :rtype: Data | None
    """
    if target is None:
      target = self.target
    if search_choices is NotSpecified:
      search_choices = self.get_search_choices()
    return self._static_get_target_value(
      target=target, _target_layers=self._target_layers, search_choices=search_choices,
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
    :param returnn.tf.network.TFNetwork network:
    :param Loss|None loss: argument just as for __init__
    :param Data output: the output (template) for the layer
    :param LayerBase|None layer:
      The real layer instance, if it exists at the current point.
      If not given, init() must be called at a later point.
    :param ((tf.Tensor)->tf.Tensor)|None reduce_func: if given, will overwrite the reduce func for the loss.
      By default, every loss_value and error_value is a scalar
      (sum or average over the batches, and over the frames for frame-wise losses).
      However, if you provide reduce_func = returnn.tf.util.basic.identity, you can get the unreduced tensor.
    :param kwargs: all the remaining __init__ args
    :return: the losses defined by this layer
    :rtype: list[returnn.tf.network.LossHolder]
    """
    if not loss:
      return []
    from returnn.tf.network import LossHolder
    return [LossHolder(
      name=name, network=network, loss=loss, layer_output=output, layer=layer, reduce_func=reduce_func)]

  def get_losses_initialized(self, reduce_func=None):
    """
    As self.get_losses, but here we return them all initialized (i.e. the layer is set).
    You should not override this method but rather :func:`get_losses`.

    :param ((tf.Tensor)->tf.Tensor)|None reduce_func: as in get_losses
    :return: the losses defined by this layer
    :rtype: list[returnn.tf.network.LossHolder]
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
    :return: scalar. see :func:`returnn.tf.util.basic.spatial_smoothing_energy`
    :rtype: tf.Tensor
    """
    from returnn.tf.util.basic import spatial_smoothing_energy, flatten_with_seq_len_mask
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
                 momentum=NotSpecified, epsilon=1e-3,
                 update_sample_only_in_training=NotSpecified,
                 delay_sample_update=NotSpecified,
                 param_version=NotSpecified,
                 gamma_init=1.0, beta_init=0.0,
                 masked_time=NotSpecified):
    """
    :param Data data:
    :param bool use_shift:
    :param bool use_std:
    :param float use_sample: defaults to 0.0 which is used in training
    :param bool force_sample: even in eval, use the use_sample factor
    :param float momentum: for the running average of sample_mean and sample_std
    :param bool update_sample_only_in_training:
    :param bool delay_sample_update:
    :param int param_version: 0 or 1 or 2
    :param float epsilon:
    :param str|float gamma_init: see :func:`returnn.tf.util.basic.get_initializer`, for the scale
    :param str|float beta_init: see :func:`returnn.tf.util.basic.get_initializer`, for the mean
    :param bool masked_time: flatten and mask input tensor
    :rtype: tf.Tensor

    https://arxiv.org/abs/1502.03167

    With our default settings:

    - In training: use_sample=0, i.e. not using running average, using current batch mean/var.
    - Not in training (e.g. eval): use_sample=1, i.e. using running average, not using current batch mean/var.
    - The running average includes the statistics of the current batch.
    - The running average is also updated when not training.

    Also see:
      tf.nn.batch_normalization()
      https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/batch_norm.py
    """
    from returnn.util import BehaviorVersion
    # Note that the old defaults (behavior version <= 11) don't really make sense!
    # https://github.com/rwth-i6/returnn/issues/522
    if momentum is NotSpecified:
      momentum = 0.1 if BehaviorVersion.get() >= 12 else 0.99
    if update_sample_only_in_training is NotSpecified:
      update_sample_only_in_training = True if BehaviorVersion.get() >= 12 else False
    if delay_sample_update is NotSpecified:
      delay_sample_update = True if BehaviorVersion.get() >= 12 else False
    if param_version is NotSpecified:
      param_version = 2 if BehaviorVersion.get() >= 12 else 0
    BehaviorVersion.require(
      masked_time is not NotSpecified, message="batch_norm masked_time should be specified explicitly", version=12)
    if masked_time is NotSpecified:
      assert BehaviorVersion.get() <= 11
      masked_time = True

    with reuse_name_scope(self.get_absolute_name_scope_prefix() + "batch_norm", absolute=True):
      # Note about param_version:
      # We might later drop some earlier versions.
      # We just need to implement a conversion in CustomCheckpointLoader.
      if param_version == 0:
        param_name_prefix = "%s_%s_" % (self.name, data.name)
      elif param_version == 1:
        param_name_prefix = ""
      elif param_version == 2:
        param_name_prefix = "v2_"
      else:
        raise NotImplementedError("%s: batch_norm param_version %r" % (self, param_version))
      stats_shape = data.get_bc_spatial_batch_shape() if param_version <= 1 else [data.dim]

      with self.var_creation_scope():
        sample_mean = self.add_param(tf_compat.v1.get_variable(
          shape=stats_shape,
          initializer=tf_compat.v1.zeros_initializer(),
          name="%smean" % param_name_prefix,
          trainable=False))
      # Note: Our Theano implementation does not use a moving average for this.
      with self.var_creation_scope():
        sample_variance = self.add_param(tf_compat.v1.get_variable(
          shape=stats_shape,
          initializer=tf_compat.v1.ones_initializer(),
          name="%svariance" % param_name_prefix,
          trainable=False))
      if use_std:
        with self.var_creation_scope():
          from returnn.tf.util.basic import get_initializer
          gamma_initializer = get_initializer(
            gamma_init, seed=self.network.random.randint(2 ** 31) if gamma_init else 0, eval_local_ns={"layer": self})
          gamma = self.add_param(tf_compat.v1.get_variable(
            shape=stats_shape,
            initializer=gamma_initializer,
            name="%sgamma" % param_name_prefix,
            trainable=True))
      else:
        gamma = None
      if use_shift:
        with self.var_creation_scope():
          from returnn.tf.util.basic import get_initializer
          beta_initializer = get_initializer(
            beta_init, seed=self.network.random.randint(2 ** 31) if beta_init else 0, eval_local_ns={"layer": self})
          beta = self.add_param(tf_compat.v1.get_variable(
            shape=stats_shape,
            initializer=beta_initializer,
            name="%sbeta" % param_name_prefix,
            trainable=True))
      else:
        beta = None

      use_fused = (
        tf_util.tf_version_tuple() >= (2, 0, 0) and  # TF2 required for exponential_avg_factor
        param_version >= 2 and
        not masked_time and
        use_shift and use_std and
        use_sample == 0 and not force_sample)

      def _calc_batch_norm_fused(train_flag):
        """
        :param bool train_flag:
        :return: like data, optional grouped update op or no_op
        :rtype: (tf.Tensor, tf.Operation)
        """
        from returnn.util import basic as util
        x = data.placeholder
        x_shape = tf_util.get_shape(x)
        if data.feature_dim_axis == data.batch_ndim - 1:  # feature last
          data_format = "NHWC"
          x = tf.reshape(x, [-1, 1, 1, x_shape[-1]])
        else:  # feature not last
          data_format = "NCHW"
          x = tf.reshape(
            x,
            [util.prod(x_shape[:data.feature_dim_axis]),
             x_shape[data.feature_dim_axis],
             util.prod(x_shape[data.feature_dim_axis + 1:]), 1])
        bn_, sample_mean_, sample_variance_ = tf_compat.v1.nn.fused_batch_norm(
          x, scale=gamma, offset=beta, mean=sample_mean, variance=sample_variance,
          epsilon=epsilon, exponential_avg_factor=momentum,
          data_format=data_format, is_training=train_flag)
        bn_ = tf.reshape(bn_, x_shape)
        update_ops = []
        if train_flag:
          updated_sample_mean = tf_compat.v1.assign(sample_mean, sample_mean_)
          updated_sample_variance = tf_compat.v1.assign(sample_variance, sample_variance_)
          update_ops += [updated_sample_mean.op, updated_sample_variance.op]
        op_ = tf.group(*update_ops)
        return bn_, op_

      def _calc_batch_norm(train_flag):
        """
        :param bool train_flag:
        :return: like data, optional grouped update op or no_op
        :rtype: (tf.Tensor, tf.Operation)
        """
        update_sample = (not update_sample_only_in_training) or train_flag
        need_mean_var_cur_batch = update_sample or (use_sample != 1 and (force_sample or train_flag))

        if need_mean_var_cur_batch:
          data_ = data
          if masked_time:
            data_ = data.copy_time_flattened()
          mean_cur_batch, variance_cur_batch = tf_compat.v1.nn.moments(
            data_.placeholder, axes=data_.get_axes(exclude_feature=True))
          mean_cur_batch = tf.reshape(mean_cur_batch, stats_shape)
          variance_cur_batch = tf.reshape(variance_cur_batch, stats_shape)
        else:
          mean_cur_batch, variance_cur_batch = None, None

        # Use exponential moving average of batch mean.
        # Note: We could also use cumulative moving average. Our Theano implementation does that for inference.
        update_ops = []  # type: typing.List[tf.Operation]
        sample_mean_, sample_variance_ = sample_mean, sample_variance
        if update_sample:
          updated_sample_mean = tf_compat.v1.assign_add(sample_mean, (mean_cur_batch - sample_mean) * momentum)
          updated_sample_variance = tf_compat.v1.assign_add(
            sample_variance, (variance_cur_batch - sample_variance) * momentum)
          update_ops += [updated_sample_mean.op, updated_sample_variance.op]
          if not delay_sample_update:
            sample_mean_ = updated_sample_mean
            sample_variance_ = updated_sample_variance
        # If train or if force_sample, use default use_sample=0.0, otherwise use_sample=1.0.
        if force_sample or train_flag:
          if use_sample == 1:
            mean, variance = sample_mean_, sample_variance_
          elif use_sample == 0:
            mean, variance = mean_cur_batch, variance_cur_batch
          else:
            mean = (1. - use_sample) * mean_cur_batch + use_sample * sample_mean_
            variance = (1. - use_sample) * variance_cur_batch + use_sample * sample_variance_
        else:
          # use_sample = 1.
          mean, variance = sample_mean_, sample_variance_

        if param_version >= 2:
          mean = tf.reshape(mean, data.get_bc_spatial_batch_shape())
          variance = tf.reshape(variance, data.get_bc_spatial_batch_shape())
        bn_ = (data.placeholder - mean) * tf_compat.v1.rsqrt(tf_util.optional_add(variance, epsilon))
        op_ = tf.group(*update_ops)
        return bn_, op_

      if use_fused:
        bn, op = self.network.cond_on_train(lambda: _calc_batch_norm_fused(True), lambda: _calc_batch_norm_fused(False))
      else:
        bn, op = self.network.cond_on_train(lambda: _calc_batch_norm(True), lambda: _calc_batch_norm(False))
      if isinstance(op, tf.Tensor):
        op = op.op
      # Make sure we update after we calculated the batch norm.
      if not tf_compat.executing_eagerly():
        tf_util.add_control_input(op, control_input=bn.op)
      self.network.register_post_control_dependencies([op])
      if not use_fused:
        if use_std:
          if param_version >= 2:
            gamma = tf.reshape(gamma, data.get_bc_spatial_batch_shape())
          bn *= gamma
        if use_shift:
          if param_version >= 2:
            beta = tf.reshape(beta, data.get_bc_spatial_batch_shape())
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

    :param tf.Tensor batch_dim: including beam size in beam search
    :param str name: layer name
    :param Data output: template
    :param returnn.tf.layers.rec.RecLayer rec_layer:
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
    if data.have_batch_axis():
      shape[data.batch_dim_axis] = batch_dim
    if isinstance(v, (float, int)):
      with tf.name_scope("init_%s_const" % name):
        from returnn.tf.util.basic import constant_with_shape
        return tf.cast(constant_with_shape(v, shape=shape), dtype=data.dtype)
    if isinstance(v, LayerBase):
      v = v.output.copy_compatible_to(output)
      if output.beam:
        v = v.copy_extend_with_beam(output.beam)
      return v.placeholder
    assert isinstance(v, str)
    if v == "zeros":
      return tf.zeros(shape, dtype=data.dtype, name="init_%s_zeros" % name)
    elif v == "ones":
      return tf.ones(shape, dtype=data.dtype, name="init_%s_ones" % name)
    elif v == "var":
      assert not data.sparse
      assert numpy.prod(bc_shape) == data.dim
      with rec_layer.var_creation_scope():
        x = tf_compat.v1.get_variable(
          "init_%s_var" % name, shape=(data.dim,), dtype=data.dtype, initializer=tf.zeros_initializer())
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
          zeroed_src_shape = tf_util.get_shape(src_output.placeholder)
          zeroed_src_shape = [
            zeroed_src_shape[i] for i in range(src_output.batch_ndim)]  # type: typing.List[typing.Union[tf.Tensor,int]]
        else:
          zeroed_src_shape = []
          for i, d in enumerate(src_output.batch_shape):
            if d is None:
              if src_output.has_dynamic_size(i):
                d = tf.reduce_max(src_output.get_dynamic_size(i))
            if d is None:
              d = 1  # fallback dummy
            zeroed_src_shape.append(d)
        if src_output.batch_dim_axis is not None:
          zeroed_src_shape[src_output.batch_dim_axis] = batch_dim
        if not src_output.beam and output.beam:
          src_output = src_output.copy_extend_with_beam(output.beam)  # potentially for seq lengths
        src_output.placeholder = tf.zeros(
          zeroed_src_shape, dtype=src_output.dtype,
          name="init_%s_zeros" % tf_util.get_valid_scope_name_from_str(src.name))
        src_output.name += "_zeroed"
        src_output.sanity_check()
        if rec_layer.network.get_config().bool("debug_runtime_sanity_checks", False):
          with tf.name_scope(tf_util.get_valid_scope_name_from_str(src.name + "_zeroed")):
            src_output.placeholder = src_output.get_placeholder_with_runtime_sanity_checks()
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
    :param returnn.tf.layers.rec.RecLayer|LayerBase|None rec_layer: for the scope
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


class InternalLayer(LayerBase):
  """
  This is not supposed to be used by the user.
  It is used by some code to construct a wrapper layer or so.
  """


class DataNotAvailableLayer(InternalLayer):
  """
  This is a dummy layer that is created when the output template is flagged "not available for inference".
  The output template should be passed to the constructor to correctly forward the information
  in case any dependent output is exported with "register_as_extern_data".

  See :func:`returnn.tf.network._create_layer`
  """
  def __init__(self, layer_class, layer_desc, **kwargs):
    """
    :param type[LayerBase] layer_class:
    :param dict[str] layer_desc:
    """
    super(DataNotAvailableLayer, self).__init__(**kwargs)
    self.layer_class_ = layer_class
    self.layer_desc = layer_desc

  def get_sub_layer(self, layer_name):
    """
    :param str layer_name: name of the sub_layer (right part of '/' separated path)
    :rtype: LayerBase|None
    """
    cls = self.layer_class_
    assert issubclass(cls, LayerBase)
    res = cls.get_sub_layer_out_data_from_opts(layer_name=layer_name, parent_layer_kwargs=self.layer_desc)
    if not res:
      return None
    out, sub_layer_class, opts = res
    assert isinstance(out, Data)
    assert issubclass(sub_layer_class, LayerBase)
    return DataNotAvailableLayer(
      name="%s/%s" % (self.name, layer_name), network=self.network, output=out,
      layer_class=sub_layer_class, layer_desc=opts)


class WrappedInternalLayer(InternalLayer):
  """
  This is not supposed to be used by the user. Like :class:`InternalLayer`, only intended for internal usage.
  This layer is supposed to logically wrap another layer.
  """

  def __init__(self, base_layer, sources=None, **kwargs):
    """
    :param LayerBase base_layer: the layer which we are wrapping
    :param list[LayerBase]|None sources: by default [base_layer]. overwrite to explicitly specify the layer deps
    """
    if sources is None:
      sources = [base_layer]
    super(WrappedInternalLayer, self).__init__(sources=sources, **kwargs)
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
    :param returnn.tf.network.TFNetwork network:
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
      from returnn.tf.network import NetworkConstructionDependencyLoopException, LayerNotFound
      try:
        return get_layer(layer_name)
      except (NetworkConstructionDependencyLoopException, LayerNotFound):
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
          if value.get("reuse_layer", None):
            value["reuse_layer"] = optional_get_layer(value["reuse_layer"])
          if value.get("layer_output", None):
            value["layer_output"] = get_layer(value["layer_output"])  # not optional, we need it right away
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
      :param returnn.tf.network.TFNetwork network:
      :param ((str) -> LayerBase) get_layer:
      """
      self.layer_name = layer_name
      self.network = network
      self.get_layer_func = get_layer
      self.var_scope = tf_compat.v1.get_variable_scope()

    def __repr__(self):
      return "<%s layer %r, net %r>" % (self.__class__.__name__, self.layer_name, self.network)

    def get_layer(self):
      """
      :rtype: LayerBase
      """
      from returnn.tf.network import NetworkConstructionDependencyLoopException, LayerNotFound
      from returnn.tf.util.basic import reuse_name_scope
      with reuse_name_scope(self.var_scope):
        try:
          return self.get_layer_func(self.layer_name)
        except (NetworkConstructionDependencyLoopException, LayerNotFound):
          return self.create_dummy_layer()

    def create_dummy_layer(self):
      """
      :rtype: LayerBase
      """
      from .basic import get_layer_class
      print(
        ("ReuseParams: layer %r does not exist yet and there is a dependency loop, " +
         "thus creating it on dummy inputs now") % self.layer_name,
        file=log.v4)

      layer_name = self.layer_name
      network = self.network
      with_time_dim = False
      while layer_name.startswith("base:") and network.parent_net:
        if network.parent_layer and network.parent_layer.output.have_time_axis():
          with_time_dim = True
        layer_name = layer_name[len("base:"):]
        network = network.parent_net

      # noinspection PyShadowingNames
      def get_dummy_input_layer(layer_name):
        """
        :param str layer_name:
        :rtype: LayerBase
        """
        if layer_name in network.layers:
          return network.layers[layer_name]
        output = None
        net = network

        # noinspection PyShadowingNames
        def opt_get_layer(layer_name):
          """
          :param str layer_name:
          :rtype: LayerBase
          """
          if layer_name in net.layers:
            return net.layers[layer_name]
          print("ReuseParams: non-existing layer %r in %r, ignoring..." % (layer_name, net), file=log.v4)
          return InternalLayer(
            name=layer_name, network=net,
            output=Data(
              name="LazyLayerResolver_dummy_output_%s" % layer_name,
              shape=(None, 1) if with_time_dim else ()))

        if self.network.parent_net is network and self.network.parent_layer:
          if layer_name.startswith(self.network.parent_layer.name + "/"):
            net = self.network
            layer_name = layer_name[len(net.parent_layer.name) + 1:]
            if layer_name in net.layers:
              # Don't return layer, could be inside loop and that wont work.
              output = net.layers[layer_name].output.copy_template()
              if not output.have_time_axis() and with_time_dim:
                output = output.copy_template_adding_time_dim().copy_template_set_ctx(network.get_control_flow_ctx())
        if not output:
          layer_desc_ = net.layers_desc[layer_name].copy()
          class_name_ = layer_desc_.pop("class")
          layer_class_ = get_layer_class(class_name_)
          layer_desc_["_network"] = net
          layer_desc_["_name"] = layer_name
          layer_class_.transform_config_dict(layer_desc_, network=net, get_layer=opt_get_layer)
          # noinspection PyProtectedMember
          layer_desc_ = net._create_layer_layer_desc(name=layer_name, layer_desc=layer_desc_)
          output = layer_class_.get_out_data_from_opts(**layer_desc_).copy()
        output.beam = None
        output.placeholder = tf.zeros(
          [d or 1 for d in output.batch_shape], dtype=output.dtype, name="%s_dummy" % output.name)
        if not output.size_placeholder:
          output.size_placeholder = {}
        for i, dim in enumerate(output.shape):
          if dim is None and i not in output.size_placeholder:
            output.size_placeholder[i] = tf.ones([1], dtype=tf.int32, name="dummy_reuse_params_size")
        output.sanity_check()
        print("ReuseParams: creating dummy input %r with %r" % (layer_name, output), file=log.v4)
        return InternalLayer(name=layer_name, network=network, output=output)

      layer_desc = network.layers_desc[layer_name].copy()
      class_name = layer_desc.pop("class")
      layer_class = get_layer_class(class_name)
      layer_desc["_network"] = network
      layer_desc["_name"] = layer_name
      layer_class.transform_config_dict(layer_desc, network=network, get_layer=get_dummy_input_layer)
      with reuse_name_scope(network.get_absolute_name_scope_prefix()[:-1], absolute=True):
        # noinspection PyProtectedMember
        return network._create_layer(
          name=layer_name, layer_class=layer_class, **layer_desc)

  # noinspection PyShadowingBuiltins
  def __init__(self, reuse_layer=None, map=None, custom=None, auto_create_missing=False, layer_output=None, shape=None):
    """
    :param LayerBase|ReuseParams.LazyLayerResolver|None reuse_layer:
    :param dict[str,ReuseParams]|None map:
    :param (**kwargs)->(tf.Tensor|tf.Variable) custom: see :func:`self.variable_custom_getter`
    :param bool auto_create_missing:
    :param LayerBase|None layer_output:
    :param tuple[Dim]|None shape:
    """
    assert isinstance(reuse_layer, (LayerBase, ReuseParams.LazyLayerResolver)) or not reuse_layer
    self._reuse_layer = reuse_layer
    self.param_map = map
    self.custom_func = custom
    self.auto_create_missing = auto_create_missing
    self.layer_output = layer_output
    self.shape = shape

  def __repr__(self):
    return "<%s reuse_layer %r, map %r>" % (self.__class__.__name__, self._reuse_layer, self.param_map)

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

  def get_variable_scope(self, base_layer, **kwargs):
    """
    :param LayerBase base_layer:
    :param kwargs: passed to tf.compat.v1.variable_scope
    :rtype: tf.compat.v1.VariableScope
    """
    def _variable_custom_getter(**kwargs_):
      return self.variable_custom_getter(base_layer=base_layer, **kwargs_)
    with tf_compat.v1.variable_scope(
          tf_compat.v1.get_variable_scope(), custom_getter=_variable_custom_getter, **kwargs) as scope:
      return scope

  def variable_custom_getter(self, base_layer, name, shape, dtype, getter, **kwargs):
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

    :param LayerBase base_layer: we expect that this is the prefix of ``name``
    :param str name: absolute param name
    :param tuple[int] shape:
    :param tensorflow.DType dtype:
    :param (...)->tf.Variable getter:
    :rtype: tf.Variable|tf.Tensor
    """
    if self.shape is not None:
      assert shape == tuple(d.dimension for d in self.shape), "unexpected shape %r for param %r" % (shape, name)
    abs_scope_prefix = base_layer.get_absolute_name_scope_prefix()
    assert not abs_scope_prefix or abs_scope_prefix.endswith("/")
    assert name.startswith(abs_scope_prefix)
    param_name = name[len(abs_scope_prefix):]  # e.g. "W" (not "rec/W")
    if self.custom_func:
      return self.custom_func(
        base_layer=base_layer, reuse_layer=self.reuse_layer, full_name=name,
        name=param_name, shape=shape, dtype=dtype, getter=getter, **kwargs)
    if self.param_map is not None:
      if not self.auto_create_missing:
        assert param_name in self.param_map
      if param_name in self.param_map:
        return self.param_map[param_name].variable_custom_getter(
          base_layer=base_layer, name=name, shape=shape, dtype=dtype, getter=getter, **kwargs)
    if self.reuse_layer:
      if not self.auto_create_missing:
        assert param_name in self.reuse_layer.params
      if param_name in self.reuse_layer.params:
        return self.reuse_layer.params[param_name]
    if self.layer_output:
      if self.shape is not None:
        out = self.layer_output.output.copy_compatible_to(Data(name=name, dim_tags=self.shape, dtype=dtype.name))
        return out.placeholder
      assert tuple(shape) == self.layer_output.output.batch_shape
      return self.layer_output.output.placeholder
    assert self.auto_create_missing
    return getter(name=name, shape=shape, dtype=dtype, **kwargs)


class SearchChoices(object):
  """
  In beam search, after expanding the beam and then selecting the N best (beam) (see :class:`ChoiceLayer`),
  when doing this multiple times, we need to keep reference where each beam came from,
  and what the current score is, etc.
  Also we could have multiple different such expansions & prunes via different :class:`ChoiceLayer`.
  This is what we keep track here.
  """

  def __init__(self, owner, beam_size, is_decided=False, keep_raw=False):
    """
    :param LayerBase owner:
    :param int beam_size:
    :param bool is_decided: by :class:`DecideLayer`
    :param bool keep_raw: by :class:`DecideKeepBeamLayer`
    """
    assert beam_size is not None
    self.owner = owner
    self._done_src_layer = False
    self._src_layer = None  # type: typing.Optional[LayerBase]
    self.src_beams = None  # type: typing.Optional[tf.Tensor]  # src beam index, (batch, beam)
    self.beam_size = beam_size
    self.beam_scores = None  # type: typing.Optional[tf.Tensor]  # (batch, beam)
    self.is_decided = is_decided
    self.keep_raw = keep_raw
    if not owner.output.beam:
      assert beam_size == 1
    else:
      assert owner.output.beam.beam_size == beam_size
      owner.network.register_search_choices_for_beam(beam=owner.output.beam, search_choices=self)

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
    if self.keep_raw:
      s += " keep_raw"
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

  def set_beam_from_own_rec(self):
    """
    Assumes we have set self.owner, and uses those rec vars to set the beam scores.
    """
    self.set_beam_from_rec(self.owner.rec_vars_outputs)

  def set_beam_from_rec(self, rev_vars_outputs):
    """
    :param dict[str,tf.Tensor] rev_vars_outputs: e.g. via :class:`ChoiceLayer`
    """
    assert (
      rev_vars_outputs.get("choice_scores", None) is not None and
      rev_vars_outputs.get("choice_src_beams", None) is not None)
    self.beam_scores = rev_vars_outputs["choice_scores"]  # (batch, beam)
    self.src_beams = rev_vars_outputs["choice_src_beams"]  # (batch, beam)
    self.beam_scores.set_shape(self.src_beams.get_shape())

  def set_src_beams(self, src_beam_idxs):
    """
    :param tf.Tensor src_beam_idxs: source beam index, (batch, beam)
    """
    if isinstance(self.beam_size, int):
      src_beam_idxs.set_shape((None, self.beam_size))
    self.src_beams = src_beam_idxs
    self.owner.rec_vars_outputs["choice_src_beams"] = src_beam_idxs

  def set_beam_scores(self, scores):
    """
    :param tf.Tensor scores: (batch, beam) -> log score
     """
    if isinstance(self.beam_size, int):
      scores.set_shape((None, self.beam_size))
    self.beam_scores = scores
    self.owner.rec_vars_outputs["choice_scores"] = scores

  def get_src_choices_seq(self):
    """
    :return: all SearchChoices we depend on up to the root, including and starting with self
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

  def get_beam_info(self):
    """
    :rtype: returnn.tf.util.data.SearchBeam|None
    """
    if self.owner.output.beam is None:
      assert self.beam_size == 1
      return None
    assert self.owner.output.beam.beam_size == self.beam_size
    return self.owner.output.beam

  def __eq__(self, other):
    return self is other

  def __ne__(self, other):
    return self is not other

  @staticmethod
  def compare(self, other):
    """
    Also see :func:`TFNetwork.get_search_choices.compare_layer`, which is basically the same.

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
    if self.keep_raw or other.keep_raw:
      return 0
    self_norm_layer = self.owner.get_normalized_layer()
    other_norm_layer = other.owner.get_normalized_layer()
    if self_norm_layer != self.owner and other_norm_layer != other.owner:
      assert self_norm_layer.search_choices and other_norm_layer.search_choices
      return SearchChoices.compare(self=self_norm_layer.search_choices, other=other_norm_layer.search_choices)
    self_src_choices = self.get_src_choices_seq()
    other_src_choices = other.get_src_choices_seq()
    if self in other_src_choices and other not in self_src_choices:
      return -1
    if other in self_src_choices and self not in other_src_choices:
      return 1
    from pprint import pformat
    raise Exception("Cannot compare search choices\n %r,\n %r\nwhich have traces:\n%s,\n%s" % (
      self, other, pformat(self_src_choices), pformat(other_src_choices)))

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
    from .basic import SelectSearchSourcesLayer
    d = sources
    if isinstance(d, dict):
      return {k: self.translate_to_this_search_beam(v) for (k, v) in d.items()}
    if isinstance(d, (tuple, list)):
      from returnn.util.basic import make_seq_of_type
      return make_seq_of_type(type(d), [self.translate_to_this_search_beam(v) for v in d])
    if isinstance(d, LayerBase):
      return SelectSearchSourcesLayer.select_if_needed(d, search_choices=self)
    return d

  @classmethod
  def translate_to_common_search_beam(cls, layer_desc):
    """
    :param list[LayerBase]|dict[str,LayerBase|object] layer_desc:
    :return: sources but all layers transformed when needed
    :rtype: list[LayerBase]|dict[str,LayerBase|object]
    """
    assert "_src_common_search_choices" not in layer_desc  # do not set this manually
    from tensorflow.python.util import nest
    layers_flat = [v for v in nest.flatten(layer_desc) if isinstance(v, LayerBase)]
    if len(layers_flat) <= 1:
      return layer_desc
    search_choicess = []
    for layer in layers_flat:
      if not layer.output.beam:
        continue
      if layer.network.is_extra_internal_template_construction():
        continue
      search_choices = layer.get_search_choices()
      from pprint import pformat
      assert search_choices, "layer %r has beam %r but no search choices; from layer desc\n%s" % (
        layer, layer.output.beam, pformat(layer_desc))
      search_choicess.append(search_choices)
    if not search_choicess:
      return layer_desc
    from functools import cmp_to_key
    common_choices = max(search_choicess, key=cmp_to_key(cls.compare))
    layer_desc = layer_desc.copy()
    layer_desc["_src_common_search_choices"] = common_choices
    return common_choices.translate_to_this_search_beam(layer_desc)


class Loss(object):
  """
  Base class for all losses.
  """
  class_name = None  # type: str  # used by get_loss_class()
  recurrent = False  # if this is a frame-wise criteria, this will be False
  need_target = True

  def __init__(self, base_network,
               use_flatten_frames=True,
               use_normalized_loss=False,
               custom_norm_factor=None,
               custom_inv_norm_factor=None,
               scale=1.0):
    """
    :param returnn.tf.network.TFNetwork base_network:
    :param bool use_flatten_frames: will use :func:`returnn.tf.util.basic.flatten_with_seq_len_mask`
    :param bool use_normalized_loss: the loss used in optimization will be normalized
    :param float|function|None custom_norm_factor:
      The standard norm factor is 1/sum(target_seq_len) if the target has a time-axis,
      or 1/sum(output_seq_len) if there is no target and the output has a time-axis,
      or 1 otherwise. (See :func:`Loss.init` for details.)
      This is used for proper normalization of accumulated loss/error per epoch
      and also proper normalization per batch for reporting,
      no matter if use_normalized_loss is True or False.
      If you want to change this norm factor, you can set this.
      As a function, it takes (self=self, output=output, layer=layer) and returns a float scalar.
    :param LayerBase|None custom_inv_norm_factor: inverse of custom_norm_factor.
      Here we allow to pass a layer.
      Here we also allow to pass any shape and it will automatically be reduced via sum.
      So you could simply pass target_seq_len directly here.
      Basically, for all reporting, it uses sum(loss) * sum(custom_inv_norm_factor).
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
    # The loss_norm_factor is used by Runner._normalize_loss both for normalization per epoch and per batch.
    # It is e.g. set to 1/sum(target_seq_len), and logic of accumulation is handled in the Runner.
    self.loss_norm_factor = None  # type: typing.Optional[tf.Tensor]
    self.use_normalized_loss = use_normalized_loss  # for the optimizer, per batch
    self.custom_norm_factor = custom_norm_factor
    self.custom_inv_norm_factor = custom_inv_norm_factor
    if custom_inv_norm_factor:
      assert custom_norm_factor is None, "%s: do not provide both custom_norm_factor and custom_inv_norm_factor" % self
    self.scale = scale

  def __repr__(self):
    return "<%s %r>" % (self.__class__.__name__, self.layer or self.output)

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
    Also, some code overwrites this function externally, e.g. with returnn.tf.util.basic.identity, to not do reducing.

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
        loss /= tf.cast(self.output.get_sequence_lengths(), tf.float32)
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
    :param returnn.tf.network.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer

    Will modify `d` such that it becomes the kwargs for `self.__init__()`.
    Mostly leaves `d` as-is.
    This is used by `LayerBase.transform_config_dict`.
    """
    if d.get("custom_inv_norm_factor", None) is not None:
      d["custom_inv_norm_factor"] = get_layer(d["custom_inv_norm_factor"])

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

  def _flatten_or_merge(self, data):
    """
    :param Data data: (B,T,...) or (T,B,...)
    :return: (B*T|T*B|B',...)
    :rtype: tf.Tensor
    """
    x = data.placeholder
    if self.use_flatten_frames:
      return tf_util.flatten_with_seq_len_mask(x, data.get_sequence_lengths(), time_major=data.is_time_major)
    x_shape = tf_util.get_shape(x)
    if data.is_time_major != self.output.is_time_major:
      # We expect at various places (eg. reduce_to_batch) that the loss is the same as self.output.
      x = tf_util.swapaxes(x, 0, 1)  # (B,T,...) or (T,B,...)
    return tf.reshape(x, [x_shape[0] * x_shape[1]] + x_shape[2:], name="merge_batch_time")

  def init(self, output, output_with_activation=None, target=None, layer=None):
    """
    :param Data output: generated output
    :param OutputWithActivation|None output_with_activation:
    :param Data target: reference target from dataset
    :param LayerBase|None layer:
    """
    with tf.name_scope("loss_init"):
      self.layer = layer
      if target:
        if output.beam:
          if target.beam != output.beam:
            target = target.copy_extend_with_beam(output.beam)
        else:
          assert not target.beam
      if output.feature_dim_axis is not None and output.feature_dim_axis != output.batch_ndim - 1:
        if output_with_activation:
          from returnn.tf.util.basic import move_axis
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
        assert time_and_batch_dims in [(0, 1), (1, 0)], (
          "output time-batch-dim unexpected: %r (target %r)" % (self.output, self.target))
        if output_with_activation and output_with_activation.act_func is tf.nn.softmax:
          out_before_act = output.copy(name="%s_before_softmax" % output.name)
          out_before_act.placeholder = output_with_activation.x
          self.output_before_softmax_flat = self._flatten_or_merge(out_before_act)
        else:
          self.output_flat = self._flatten_or_merge(output)
          self.output_flat.set_shape(tf.TensorShape((None,) + output.shape[1:]))
        if target:
          assert target.have_time_axis()
          self.target_seq_lens = target.get_sequence_lengths()
          self.target_flat = self._flatten_or_merge(target)
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
      if self.custom_norm_factor is not None:
        if callable(self.custom_norm_factor):
          self.loss_norm_factor = self.custom_norm_factor(self=self, output=output, layer=layer)
        else:
          assert isinstance(self.custom_norm_factor, float)
          self.loss_norm_factor = self.custom_norm_factor
      if self.custom_inv_norm_factor:
        self.loss_norm_factor = 1.0 / tf.cast(tf.reduce_sum(self.custom_inv_norm_factor.output.placeholder), tf.float32)
      self._check_init()

  def _check_init(self):
    """
    Does some checks on self.target and self.output, e.g. if the dense shapes matches.
    You can overwrite this if those checks don't make sense for your derived loss class.
    """
    if not self.target:
      assert not self.need_target, "%s: did not get target" % self
      return
    assert self.target.placeholder is not None
    if self.output_before_softmax_flat is not None or self.output_flat is not None:
      assert self.target_flat is not None, (
        "%s: have flat output (%r) but not flat targets (%r)" % (self, self.output, self.target))
    assert self.target.ndim_dense == self.output.ndim_dense, (
      "Number of dimensions mismatch. Target: %s, output: %s" % (self.target, self.output))
    expected_output_dim = self.get_auto_output_layer_dim(self.target.feature_dim_or_sparse_dim)
    assert expected_output_dim.dimension == self.output.dim, (
      "Expected output dim is %r but the output has dim %r. " % (
        expected_output_dim, self.output.feature_dim_or_sparse_dim) +
      "Target: %s, output: %s" % (self.target, self.output))
    if self.base_network.get_config().bool("debug_runtime_sanity_checks", False):
      with tf.name_scope("Loss_debug_runtime_sanity_checks"):
        checks = [self.output.get_runtime_sanity_check_op(), self.target.get_runtime_sanity_check_op()]
        out_shape = tf.shape(self.output.placeholder)
        target_shape = tf.shape(self.target.placeholder)
        if self.output.have_batch_axis() and self.target.have_batch_axis():
          out_batch_dim = out_shape[self.output.batch_dim_axis]
          target_batch_dim = target_shape[self.target.batch_dim_axis]
          checks += [tf.Assert(
            tf.equal(out_batch_dim, target_batch_dim),
            ["Loss_debug_runtime_sanity_checks", "batch dim mismatch",
             "output:", str(self.output), "shape", out_shape,
             "target:", str(self.target), "shape", target_shape])]
        if not self.recurrent:  # framewise
          if self.output.have_time_axis() and self.target.have_time_axis():
            out_time_dim = out_shape[self.output.time_dim_axis]
            target_time_dim = target_shape[self.target.time_dim_axis]
            checks += [tf.Assert(
              tf.equal(out_time_dim, target_time_dim),
              ["Loss_debug_runtime_sanity_checks", "time dim mismatch",
               "output:", str(self.output), "shape", out_shape,
               "target:", str(self.target), "shape", target_shape])]
            if self.output.has_dynamic_size(self.output.time_dim_axis):
              assert self.target.has_dynamic_size(self.target.time_dim_axis)
              out_sizes = self.output.get_dynamic_size(self.output.time_dim_axis)
              target_sizes = self.target.get_dynamic_size(self.target.time_dim_axis)
              checks += [tf.Assert(
                tf.reduce_all(tf.equal(out_sizes, target_sizes)),
                ["Loss_debug_runtime_sanity_checks", "dyn seq len mismatch",
                 "output:", str(self.output), "shape", out_shape, "sizes", out_sizes,
                 "target:", str(self.target), "shape", target_shape, "sizes", target_sizes], summarize=20)]
        with tf.control_dependencies(checks):
          if self.target_flat is not None:
            self.target_flat = tf.identity(self.target_flat)
          else:
            self.target = self.target.copy()
            self.target.placeholder = tf.identity(self.target.placeholder)

  def get_error(self):
    """
    :return: frame error rate as a scalar value with the default self.reduce_func (see also self.get_value)
    :rtype: tf.Tensor
    """
    with tf.name_scope("loss_frame_error"):
      assert self.output.ndim_dense == self.target.ndim_dense
      from returnn.tf.util.basic import check_input_ndim, check_shape_equal
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
    return tf.convert_to_tensor(self.loss_norm_factor)

  @classmethod
  def get_auto_output_layer_dim(cls, target_dim):
    """
    :param returnn.tf.util.data.Dim target_dim:
    :return: normally just the same as target_dim. e.g. for CTC, we would add 1 for the blank label
    :rtype: returnn.tf.util.data.Dim
    """
    return target_dim

  @classmethod
  def get_default_target(cls, extern_data):
    """
    :param TFNetwork.ExternData extern_data:
    :return: default target name, or None if this loss does not have a target
    :rtype: str|None
    """
    if not cls.need_target:
      return None
    return extern_data.default_target
