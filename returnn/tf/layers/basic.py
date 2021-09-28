
"""
Many canonical basic layers.
"""

from __future__ import print_function

import tensorflow as tf
import contextlib
import typing
import returnn.tf.compat as tf_compat
import returnn.tf.util.basic as tf_util
from returnn.util.basic import unicode, NotSpecified
from returnn.tf.util.data import Data, SearchBeam, DimensionTag
from returnn.tf.util.basic import OutputWithActivation, dimshuffle, swapaxes
from returnn.log import log
from .base import LayerBase, Loss, InternalLayer, SearchChoices


class SourceLayer(LayerBase):
  """
  This gives access to some entry from network.extern_data (:class:`ExternData`).
  """
  layer_class = "source"

  def __init__(self, network, data_key=None, sources=(), **kwargs):
    """
    :param returnn.tf.network.TFNetwork network:
    :param str|None data_key:
    :param tuple sources:
    """
    if data_key is None:
      data_key = network.extern_data.default_input
    assert not sources, "source layer does not expect sources"
    data = network.get_extern_data(data_key, mark_data_key_as_used=True).copy()
    super(SourceLayer, self).__init__(network=network, **kwargs)
    # Note: No check on data.placeholder. We allow to behave similar as DataNotAvailableLayer.
    self.output = data
    if self.output.beam:
      # This can happen if register_as_extern_data was used on a layer with a beam.
      search_choices = network.get_search_choices_from_beam(self.output.beam)
      self.sources.append(search_choices.owner)

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param returnn.tf.network.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    d.setdefault("from", [])  # source does not make sense
    super(SourceLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)

  @classmethod
  def get_out_data_from_opts(cls, network, data_key=None, **kwargs):
    """
    :param returnn.tf.network.TFNetwork network:
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
    name = "concat_" + "_".join([layer.tf_scope_name for layer in src_layers])
    name_scope = base + name + "/" + postfix
  from returnn.tf.util.basic import reuse_name_scope
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
  # Currently we assume that get_concat_sources_data_template will match Data.get_common_data (besides the dim).
  common_source = Data.get_common_data([s.output for s in src_layers], ignore_feature_dim=True)
  data.size_placeholder = common_source.size_placeholder.copy()  # to get right dimension tags
  layers_data = []
  with _name_scope_for_concat_src_layers(src_layers, "concat_sources"):
    for layer in src_layers:
      assert not layer.output.sparse, "sparse concat not supported"
      assert layer.output.dtype == data.dtype, "incompatible dtype with layer %r" % layer
      # unbroadcast is needed for tf.concat.
      layers_data.append(layer.output.copy_compatible_to(data, unbroadcast=True, except_feature=True))
    data.placeholder = tf.concat(
      axis=data.feature_dim_axis,
      values=[layer_data.placeholder for layer_data in layers_data])
    axes_split_info = [None] * data.batch_ndim  # type: typing.List[typing.Optional[typing.List[int]]]
    axes_split_info[data.feature_dim_axis] = [layer_data.dim for layer_data in layers_data]
    tf_util.set_param_axes_split_info(data.placeholder, axes_split_info)
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
  from ..util.data import DimensionTag
  assert src_layers, "need source layers"
  if len(src_layers) == 1:
    return src_layers[0].output.copy_template(name=name)
  if not name:
    name = "concat_" + "_".join([layer.name for layer in src_layers])
  dim = 0
  common_source = Data.get_common_data([s.output for s in src_layers], ignore_feature_dim=True)
  for layer in src_layers:
    # Note: We do not perform much compatibility checks at this point,
    # as this is for a template only anyway.
    # The real checks are in concat_sources.
    assert not layer.output.sparse
    if layer.output.dim is not None:  # just ignore at this point if None (e.g. during template construction)
      dim += layer.output.dim
  return common_source.copy_template_replace_dim_tag(
    name=name,
    axis=common_source.feature_dim_axis,
    new_dim_tag=DimensionTag(
      kind=DimensionTag.Types.Feature, description=name + "_feature", dimension=dim))


def concat_sources_with_opt_dropout(src_layers, dropout=0, dropout_noise_shape=None, dropout_on_forward=False):
  """
  :param list[LayerBase] src_layers:
  :param float dropout: dropout rate that will be applied if train_flag is set or dropout_on_forward is enabled
  :param tuple|list|dict|None dropout_noise_shape: provide 1 for broadcasting or None otherwise for each axis.
  The default "None" will broadcast across all dynamic axes including the batch axis.
  Use {"*": None} to disable broadcasting for all axes.
  :param bool dropout_on_forward: apply dropout also during inference
  :return: data with placeholders set
  :rtype: Data
  """
  assert src_layers, "need source layers"
  data = concat_sources(src_layers)
  network = src_layers[0].network
  if network.train_flag is False and not dropout_on_forward:
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
    if dropout_on_forward:
      data.placeholder = tf_util.dropout(
          data.placeholder,
          keep_prob=1 - dropout,
          noise_shape=dropout_noise_shape,
          seed=network.random.randint(2 ** 31))
    else:
      data.placeholder = network.cond_on_train(
        fn_train=lambda: tf_util.dropout(
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

  def __init__(self, dropout=0, dropout_noise_shape=None, dropout_on_forward=False, mask=None, **kwargs):
    """
    :param float dropout: 0.0 means to apply no dropout. dropout will only be applied during training
    :param dict[str|tuple,int|None] dropout_noise_shape: see :func:`TFUtil.get_bc_shape`
    :param bool dropout_on_forward: apply dropout during inference
    :param str|None mask: "dropout" or "unity" or None. this is obsolete and only here for historical reasons
    """
    super(_ConcatInputLayer, self).__init__(**kwargs)
    assert mask in ['dropout', 'unity', None], "invalid mask: %r" % mask
    if mask == "unity":
      assert not dropout
    elif mask == "dropout":
      assert dropout > 0
    self.dropout = dropout
    self.input_data = None  # type: typing.Optional[Data]
    if self.sources:
      self.input_data = concat_sources_with_opt_dropout(
        self.sources, dropout=dropout, dropout_noise_shape=dropout_noise_shape, dropout_on_forward=dropout_on_forward)


class CopyLayer(_ConcatInputLayer):
  """
  This layer does nothing, it copies its input.
  If multiple sources are provided, they are concatenated in the feature-dim.
  """

  layer_class = "copy"

  def __init__(self, extra_deps=(), **kwargs):
    """
    :param list[LayerBase] extra_deps: Just add as an additional dependency, without really using it.
      This can have an effect though on the search beam, via :class:`SelectSearchSourcesLayer`.
      We only have this here for the :class:`CopyLayer` because the :func:`get_out_data_from_opts`
      must know about it and define the right beam.
      Also see the option ``collocate_with``, which is different in that it does *not* add a dependency.
    """
    super(CopyLayer, self).__init__(**kwargs)
    self.extra_deps = extra_deps
    self.output = self.input_data.copy(name="%s_output" % self.name)
    if len(self.sources) == 1:
      self.output_loss = self.sources[0].output_loss
      if not self.dropout:
        self.output_before_activation = self.sources[0].output_before_activation
    for src in self.sources:
      if src.allow_inf_in_output:
        self.allow_inf_in_output = True

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    return super(CopyLayer, self).get_dep_layers() + list(self.extra_deps)

  @classmethod
  def get_out_data_from_opts(cls, name, sources=(), extra_deps=(), out_type=None, n_out=NotSpecified, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param list[LayerBase] extra_deps:
    :param dict[str]|None out_type:
    :param int|None|NotSpecified n_out:
    :rtype: Data
    """
    # If all sources are defined, use them to get the exact out_type.
    out = get_concat_sources_data_template(sources, name="%s_output" % name)
    # Instead of checking or raising an exception, just overwrite, as this could be the template construction.
    if out_type or n_out is not NotSpecified:
      if not out_type:
        out_type = {}
      else:
        out_type = out_type.copy()
      if out.sparse:
        out_type["sparse"] = True  # otherwise the default get_out_data_from_opts would assume dense
      if n_out is not NotSpecified:
        out_type["dim"] = n_out
      elif out.dim is not None:
        out_type.setdefault("dim", out.dim)
      out = super(CopyLayer, cls).get_out_data_from_opts(
        name=name, out_type=out_type, n_out=n_out, sources=sources, **kwargs)
    out.beam = SearchBeam.get_combined_beam(out.beam, *[dep.output.beam for dep in extra_deps if dep])
    return out

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param returnn.tf.network.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    super(CopyLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    if "extra_deps" in d:
      extra_deps = d["extra_deps"]
      if not isinstance(extra_deps, (list, tuple)):
        extra_deps = [extra_deps]
      d["extra_deps"] = [get_layer(src_name) for src_name in extra_deps]


class DropoutLayer(CopyLayer):
  """
  Just the same as :class:`CopyLayer`, because that one already supports dropout.
  """
  layer_class = "dropout"


class ScaledGradientLayer(CopyLayer):
  """
  Just tf.identity in the forward pass.
  Scales the gradient by some factor in backprop.
  Can be used as gradient reversal layer (with negative factor).
  Uses :func:`TFUtil.scaled_gradient`, or :func:`tf.stop_gradient`
  """
  layer_class = "scaled_grad"

  def __init__(self, scale, **kwargs):
    """
    :param float scale: if 0., will use tf.stop_gradient
    """
    super(ScaledGradientLayer, self).__init__(**kwargs)
    from returnn.tf.util.basic import scaled_gradient
    if scale == 0.:
      self.output.placeholder = tf.stop_gradient(self.output.placeholder)
    else:
      self.output.placeholder = scaled_gradient(self.output.placeholder, scale=scale)


class SelectSearchSourcesLayer(InternalLayer):
  """
  Selects the corresponding search beams from the source, given current search choices
  (determined by a layer).
  Like :class:`InternalLayer`, only for internal purpose at the moment.
  """

  @classmethod
  def select_if_needed(cls, layer, search_choices):
    """
    :param LayerBase layer:
    :param SearchChoices|None search_choices:
    :rtype: LayerBase
    """
    assert isinstance(layer, LayerBase)
    if not search_choices:
      return layer
    if layer.network.is_extra_internal_template_construction():
      assert layer.output.placeholder is None  # we expect a template
      return layer
    layer_search_choices = layer.get_search_choices()
    if layer_search_choices and layer_search_choices.keep_raw:
      return layer
    if layer_search_choices == search_choices:
      assert layer.output.beam == search_choices.get_beam_info(), "%r != %r. %s" % (
        layer.output.beam, search_choices.get_beam_info(),
        layer.network.debug_search_choices(layer) or "debug search dumped")
      return layer
    if layer.output.batch_dim_axis is None:  # e.g. VariableLayer, ConstantLayer, or so
      return layer
    layer = SelectSearchSourcesLayer(sources=[layer], search_choices_layer=search_choices.owner)
    assert layer.output.beam == search_choices.get_beam_info(), "%r != %r. %s" % (
      layer.output.beam, search_choices.get_beam_info(),
      layer.network.debug_search_choices(layer) or "debug search dumped")
    return layer

  def __init__(self, search_choices_layer, sources, **kwargs):
    """
    :param LayerBase search_choices_layer:
    :param list[LayerBase] sources:
    """
    from returnn.tf.util.basic import select_src_beams, get_valid_scope_name_from_str, DimensionTag
    from pprint import pformat
    assert len(sources) == 1
    search_choices = search_choices_layer.get_search_choices()
    src = sources[0]
    kwargs = kwargs.copy()
    kwargs["sources"] = sources
    if "output" not in kwargs:
      kwargs["output"] = src.output  # will be reset later
    if "network" not in kwargs:
      kwargs["network"] = src.network
    if "name" not in kwargs:
      kwargs["name"] = src.name
    if "_src_common_search_choices" not in kwargs:
      kwargs["_src_common_search_choices"] = search_choices
    super(SelectSearchSourcesLayer, self).__init__(**kwargs)
    self.search_choices_layer = search_choices_layer
    self.used_search_choices_beams = False
    self.search_choices_from_layer = search_choices
    self.output = src.output.copy_as_batch_major()
    self.rec_vars_outputs = src.rec_vars_outputs.copy()
    src_search_choices = src.get_search_choices()
    self.transform_func = None  # type: typing.Optional[typing.Callable[[tf.Tensor],tf.Tensor]]
    self.search_choices_seq = None  # type: typing.Optional[typing.List[SearchChoices]]
    if not search_choices:
      assert not src_search_choices
      assert not self.output.beam
    elif search_choices == src_search_choices:
      pass
    elif not src_search_choices:
      assert not self.output.beam, ("no src %r search choices but beam?" % src, src.network.debug_search_choices(src))
      self.output = self.output.copy_extend_with_beam(search_choices.get_beam_info())
    else:
      assert search_choices and search_choices != src_search_choices
      search_choices_seq = search_choices.get_src_choices_seq()
      assert src_search_choices in search_choices_seq, self.network.debug_search_choices(self.search_choices_layer) or (
        ("%s: No common search base:\n"
         "from layer %s\n"
         "search choices %s,\n"
         "to layer %s\n"
         "search choices\n%s.") % (
          self, src, src_search_choices, self.search_choices_layer, pformat(search_choices_seq)))
      search_choices_seq = search_choices_seq[:search_choices_seq.index(src_search_choices)]
      assert src_search_choices not in search_choices_seq
      assert search_choices_seq
      self.output.beam = search_choices.get_beam_info()
      if self.output.batch:
        self.output.batch = self.output.batch.copy_set_beam(self.output.beam)

      def transform(v):
        """
        :param tuple|list|tf.Tensor|tf.TensorArray|T v:
        :rtype: T
        """
        if isinstance(v, (tuple, list)):
          from returnn.util.basic import make_seq_of_type
          return make_seq_of_type(type(v), [transform(v_) for v_ in v])
        assert isinstance(v, (tf.Tensor, tf.TensorArray))
        if isinstance(v, tf.Tensor) and v.get_shape().ndims == 0:
          return v  # leave scalars as-is
        if isinstance(v, tf.Tensor) and getattr(v, "_RETURNN_beam_expanded_base_data", None):
          # This tensor was just expanded by a beam. Selecting beams are not needed.
          return v
        for i, base_src_choices in enumerate(reversed(search_choices_seq)):
          assert isinstance(base_src_choices, SearchChoices)
          assert base_src_choices.src_beams is not None, (
            self.network.debug_search_choices(self.search_choices_layer) or (
              ("Cannot transform %r,\n"
               "search choices %r,\n"
               "to search choices %r.\n"
               "Missing beam idxs.") % (src, src_search_choices, search_choices_seq)))
          tag = DimensionTag.get_tag_from_size_tensor(v)
          v = select_src_beams(
            v, src_beams=base_src_choices.src_beams,
            name="%s_select_src_beams_%i_%s_%i_%s" % (
              get_valid_scope_name_from_str(self.name),
              i, get_valid_scope_name_from_str(base_src_choices.owner.name),
              len(search_choices_seq), get_valid_scope_name_from_str(search_choices.owner.name)))
          if tag:
            tag.set_tag_on_size_tensor(v, batch=self.output.batch.copy_set_beam(base_src_choices.get_beam_info()))
          self.used_search_choices_beams = True
        return v

      self.search_choices_seq = search_choices_seq
      self.transform_func = transform
      # It's possible that src.output.placeholder is not set, e.g. in a prev-layer where the
      # prev output is not needed, only the prev state. See _TemplateLayer.copy_as_prev_time_frame.
      src_output = src.output.copy_as_batch_major()
      if src_output.placeholder is not None:
        self.output.placeholder = transform(src_output.placeholder)
      if src_output.size_placeholder:
        self.output.size_placeholder = {i: transform(size) for (i, size) in src_output.size_placeholder.items()}
      self.rec_vars_outputs = {k: transform(v) for (k, v) in src.rec_vars_outputs.items()}  # assumes batch-major

    for src in self.sources:
      if src.allow_inf_in_output:
        self.allow_inf_in_output = True

  def __repr__(self):
    return "<%s %r %r out_type=%s>" % (
      self.__class__.__name__, self.name, self.search_choices_from_layer,
      self.output.get_description(with_name=False) if self.output else None)

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    dep_layers = super(SelectSearchSourcesLayer, self).get_dep_layers()
    if self.used_search_choices_beams:  # only in that case, it is really a dependency
      dep_layers.append(self.search_choices_layer)
    return dep_layers

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
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
    if data.beam:
      assert search_choices
      data = data.copy_extend_with_beam(search_choices.get_beam_info())
    elif search_choices:
      data = data.copy_extend_with_beam(search_choices.get_beam_info())
    return data


class ActivationLayer(_ConcatInputLayer):
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
      from returnn.tf.util.basic import get_activation_function
      act_func = get_activation_function(activation)
      if act_func in {tf.nn.softmax, tf.nn.log_softmax} and self.output.feature_dim_axis != self.output.batch_ndim - 1:
        # Make sure we use the right axis. Don't use OutputWithActivation.
        # noinspection PyArgumentList
        self.output.placeholder = act_func(x, axis=self.output.feature_dim_axis)
        self.output_before_activation = None
      else:
        self.output_before_activation = OutputWithActivation(x, act_func=act_func)
    else:
      self.output_before_activation = OutputWithActivation(x)
    if self.output_before_activation:
      self.output.placeholder = self.output_before_activation.y

  @classmethod
  def get_out_data_from_opts(cls, activation, **kwargs):
    """
    :param str activation:
    :rtype: Data
    """
    # Just the same as the input.
    # Use CopyLayer.get_out_data_from_opts for potential extra logic for out_type.
    out = CopyLayer.get_out_data_from_opts(**kwargs)
    # Modify dtype if needed based on activation function
    if activation == "abs" and out.dtype == "complex64":
      out.dtype = "float32"
    return out


class BatchNormLayer(CopyLayer):
  """
  Implements batch-normalization (http://arxiv.org/abs/1502.03167) as a separate layer.

  Also see :class:`NormLayer`.
  """
  layer_class = "batch_norm"

  def __init__(self, use_shift=NotSpecified, use_std=NotSpecified, use_sample=NotSpecified, force_sample=NotSpecified,
               momentum=NotSpecified, epsilon=NotSpecified,
               update_sample_only_in_training=NotSpecified,
               delay_sample_update=NotSpecified,
               param_version=NotSpecified,
               gamma_init=NotSpecified, beta_init=NotSpecified,
               masked_time=NotSpecified, **kwargs):
    """
    :param bool use_shift:
    :param bool use_std:
    :param float use_sample: defaults to 0.0 which is used in training
    :param bool force_sample: even in eval, use the use_sample factor
    :param float momentum: for the running average of sample_mean and sample_std
    :param bool update_sample_only_in_training:
    :param bool delay_sample_update:
    :param int param_version: 0 or 1
    :param float epsilon:
    :param str|float gamma_init: see :func:`TFUtil.get_initializer`, for the scale
    :param str|float beta_init: see :func:`TFUtil.get_initializer`, for the mean
    :param bool masked_time: flatten and mask input tensor

    The default settings for these variables are set in the function "batch_norm" of the LayerBase. If you do not want
    to change them you can leave them undefined here.
    With our default settings:

    - In training: use_sample=0, i.e. not using running average, using current batch mean/var.
    - Not in training (e.g. eval): use_sample=1, i.e. using running average, not using current batch mean/var.
    - The running average includes the statistics of the current batch.
    - The running average is also updated when not training.
    """
    local = locals()
    from returnn.util.basic import getargspec
    batch_norm_kwargs = getargspec(self.batch_norm).args[1:]  # first is self, ignore
    batch_norm_opts = {key: local[key] for key in batch_norm_kwargs if key in local and local[key] != NotSpecified}
    super(BatchNormLayer, self).__init__(batch_norm=batch_norm_opts or True, **kwargs)


class LayerNormLayer(_ConcatInputLayer):
  """
  Applies `layer-normalization <https://arxiv.org/abs/1607.06450>`__.

  Note that we *just* normalize over the feature-dim axis here.
  This is consistent to the default behavior of :class:`tf.keras.layers.LayerNormalization`
  and also how it is commonly used in many models, including Transformer.

  However, there are cases where it would be common to normalize over all axes except batch-dim,
  or all axes except batch and time.
  For a more generic variant, see :class:`NormLayer`.
  """
  layer_class = "layer_norm"

  def __init__(self, epsilon=1e-6, **kwargs):
    """
    :param float epsilon:
    """
    super(LayerNormLayer, self).__init__(**kwargs)
    assert not self.input_data.sparse
    x = self.input_data.placeholder
    dim = self.input_data.dim
    axis = self.input_data.feature_dim_axis
    with self.var_creation_scope():
      scale = self.add_param(tf_compat.v1.get_variable("scale", [dim], initializer=tf.ones_initializer()))
      bias = self.add_param(tf_compat.v1.get_variable("bias", [dim], initializer=tf.zeros_initializer()))
    mean = tf.reduce_mean(x, axis=[axis], keepdims=True, name="mean")
    variance = tf.reduce_mean(tf.square(x - mean), axis=[axis], keepdims=True, name="variance")
    with tf.name_scope("normalized"):
      norm_x = (x - mean) * tf_compat.v1.rsqrt(variance + epsilon)
    if axis != self.input_data.batch_ndim - 1:
      ndim = self.input_data.batch_ndim
      scale_bc = tf.reshape(scale, [dim if i == axis else 1 for i in range(ndim)])
      bias_bc = tf.reshape(bias, [dim if i == axis else 1 for i in range(ndim)])
      self.output.placeholder = norm_x * scale_bc + bias_bc
    else:
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


class NormLayer(_ConcatInputLayer):
  """
  Normalize over specified axes, e.g. time and/or feature axis.

  Note: For calculating a norm, see :class:`MathNormLayer` instead.

  In case of just feature (``axes="F"``),
  this corresponds to `layer normalization <https://arxiv.org/abs/1607.06450>`__ (see :class:`LayerNormLayer`).
  In case of time and feature (``axes="TF"``) for a 3D input,
  or more general all except batch (``axes="except_batch"``),
  this corresponds to `group normalization <https://arxiv.org/abs/1803.08494>`__ with G=1,
  or non-standard layer normalization.
  (The definition of layer-normalization is not clear on what axes should be normalized over.
  In many other frameworks, the default axis is just the last axis,
  which is usually the feature axis.
  However, in certain implementations and models,
  it is also common to normalize over all axes except batch.)

  The statistics are calculated just on the input.
  There are no running statistics (in contrast to batch normalization, see :class:`BatchNormLayer`).

  For some discussion on the definition of layer-norm vs group-norm,
  also see
  `here <https://stats.stackexchange.com/questions/485550/is-group-norm-with-g-1-equiv-to-layer-norm>`__
  and `here <https://github.com/tensorflow/addons/issues/2143>`__.
  """
  layer_class = "norm"

  def __init__(self, axes, param_shape="F", scale=True, bias=True, epsilon=1e-6, **kwargs):
    """
    :param str|list[str] axes: axes over which the mean and variance are computed, e.g. "F" or "TF"
    :param str|list[str]|tuple[str]|int|list[int]|tuple[int] param_shape: shape of the scale and bias parameters.
      You can also refer to (static) axes of the input, such as the feature-dim.
      This is also the default, i.e. a param-shape of [F], independent of the axes to normalize over.
    :param bool scale: add trainable scale parameters
    :param bool bias: add trainable bias parameters
    :param float epsilon: epsilon for numerical stability
    """
    super(NormLayer, self).__init__(**kwargs)
    assert not self.input_data.sparse
    x = self.input_data.placeholder
    assert isinstance(param_shape, str)  # not implemented otherwise yet
    param_axes = sorted(self.input_data.get_axes_from_description(param_shape))
    param_shape = [self.input_data.batch_shape[axis] for axis in param_axes]
    assert all(isinstance(dim, int) for dim in param_shape), "%s: only static param shape allowed" % self
    param_bc_shape = [dim if axis in param_axes else 1 for (axis, dim) in enumerate(self.input_data.batch_shape)]
    axes = self.input_data.get_axes_from_description(axes)

    mean = tf.reduce_mean(x, axis=axes, keepdims=True, name="mean")
    variance = tf.reduce_mean(tf.square(x - mean), axis=axes, keepdims=True, name="variance")
    with tf.name_scope("normalized"):
      norm_x = (x - mean) * tf_compat.v1.rsqrt(variance + epsilon)
    if scale:
      with self.var_creation_scope():
        scale_param = self.add_param(tf_compat.v1.get_variable("scale", param_shape, initializer=tf.ones_initializer()))
      norm_x *= tf.reshape(scale_param, param_bc_shape)
    if bias:
      with self.var_creation_scope():
        bias_param = self.add_param(tf_compat.v1.get_variable("bias", param_shape, initializer=tf.zeros_initializer()))
      norm_x += tf.reshape(bias_param, param_bc_shape)
    self.output.placeholder = norm_x
    self.output.size_placeholder = self.input_data.size_placeholder.copy()

  @classmethod
  def get_out_data_from_opts(cls, sources, name, **kwargs):
    """
    :param list[LayerBase] sources:
    :param str name:
    :rtype: Data
    """
    return get_concat_sources_data_template(sources, name="%s_output" % name)


class MathNormLayer(_ConcatInputLayer):
  """
  Calculates sum(abs(x) ** p) ** (1./p).
  """
  layer_class = "math_norm"

  def __init__(self, p, axes, keep_dims=False, **kwargs):
    """
    :param int|float p:
    :param str|list[str] axes:
    :param bool keep_dims:
    """
    super(MathNormLayer, self).__init__(**kwargs)
    x = self.input_data.copy()
    x.placeholder = tf.abs(x.placeholder) ** p
    self.output.placeholder = ReduceLayer.reduce(x, mode="sum", axes=axes, keep_dims=keep_dims) ** (1. / p)

  @classmethod
  def get_out_data_from_opts(cls, name, sources, axes, keep_dims=False, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param str|list[str] axes:
    :param bool keep_dims:
    :rtype: Data
    """
    return ReduceLayer.get_out_data_from_opts(name=name, sources=sources, axes=axes, keep_dims=keep_dims)


class SliceLayer(_ConcatInputLayer):
  """
  Slicing on the input, i.e. x[start:end:step] in some axis.
  See also :class:`SliceNdLayer`, for variable start.
  See also :class:`GatherLayer`, for one single position.

  Note that __getitem__ on a TF tensor (or also Numpy ND array) is more generic,
  and supports slices in multiple axes, as well as adding new dimensions, etc.
  It even allows to get boolean values, and then applies a boolean mask.
  See TF _slice_helper (== tf.Tensor.__getitem__) for a generic implementation,
  which calls tf.strided_slice.
  If we ever need such more generic support, we might consider adding a new layer,
  like ``GenericSliceLayer``, which gets a ``splice_spec``,
  just like ``_slice_helper`` (argument to ``__getitem__``).
  But any such a slice can already be constructed with multiple individual layers,
  which perform individual slices (per axis).

  We just support slicing in a single axis here, with optional striding (slice_step).
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
    size_placeholder = self.input_data.size_placeholder.copy()
    if axis_wo_batch in size_placeholder:
      if slice_start:
        assert slice_start > 0
        size_placeholder[axis_wo_batch] = (
          tf.maximum(0, size_placeholder[axis_wo_batch] - slice_start))
      if slice_end is not None:
        if slice_end >= 0:
          size_placeholder[axis_wo_batch] = (
            tf.minimum(slice_end, size_placeholder[axis_wo_batch]))
        else:  # slice_end < 0
          size_placeholder[axis_wo_batch] = (
            tf.maximum(0, size_placeholder[axis_wo_batch] + slice_end))
      if slice_step:
        size_placeholder[axis_wo_batch] = (
          tf.cast(tf_compat.v1.ceil(tf.divide(size_placeholder[axis_wo_batch], slice_step)), tf.int32))
      from returnn.tf.util.basic import DimensionTag
      if not DimensionTag.get_tag_from_size_tensor(size_placeholder[axis_wo_batch]):
        tag = DimensionTag(
          description="slice%i:%s" % (axis_wo_batch, self.get_absolute_name()),
          kind=DimensionTag.Types.Spatial, batch=self.output.batch)
        tag.set_tag_on_size_tensor(size_placeholder[axis_wo_batch])
    self.output.size_placeholder = size_placeholder
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
    from ..util.data import DimensionTag
    input_data = get_concat_sources_data_template(sources)
    axis = input_data.get_axis_from_description(axis)
    dim_tag = input_data.dim_tags[axis]
    dim_slice = slice(slice_start, slice_end, slice_step)
    new_dim = len(range(dim_tag.dimension)[dim_slice]) if dim_tag.dimension is not None else None
    new_dim_tag = DimensionTag(kind=dim_tag.kind, description="%s:slice" % name, dimension=new_dim)
    return input_data.copy_template_replace_dim_tag(axis=axis, new_dim_tag=new_dim_tag, name="%s_output" % name)


class SliceNdLayer(_ConcatInputLayer):
  """
  This takes out a slice-range from the time axis,
  e.g. ``x[start:start + size]``.
  If the input is of shape (B,T,F) and start is of shape (B,),
  then the output will be of shape (B,size,F).
  If the input is of shape (B,T,F) and start is of shape (B,T),
  then the output will be of shape (B,T,size,F).
  This layer allows a different start slice point for each batch,
  in contrast to :class:`SliceLayer`, and the start is variable.
  See also :class:`GatherNdLayer`.
  :class:`PrefixInTimeLayer` can recover the original shape (by zero-padding).
  """
  layer_class = "slice_nd"
  recurrent = True

  def __init__(self, start, size, min_size=None, **kwargs):
    """
    :param LayerBase start: (B,...)
    :param int|LayerBase|None size: if None, it uses the max possible size,
      and it becomes a dynamic axis.
    :param int|None min_size: if size is None, but we want to have a min-size
    """
    super(SliceNdLayer, self).__init__(**kwargs)
    from returnn.tf.util.basic import where_bc
    from returnn.tf.util.data import Data
    x = self.input_data.copy()
    seq_lens_data = x.get_time_dim_tag().dyn_size_ext  # (B,) or None
    self.start = start
    self.size = size
    start_data = start.output.copy()  # e.g. (B,) or (B,T)
    data_objs = [start_data]
    data_objs += [size.output] if isinstance(size, LayerBase) else []
    data_objs += [seq_lens_data] if isinstance(seq_lens_data, Data) else []
    common_data = Data.get_common_data(data_objs)
    start_data = start_data.copy_compatible_to(common_data, check_sparse=False)
    start_t = start_data.placeholder
    if size is None:
      if min_size is None:
        min_size = 0
      if seq_lens_data is None:
        assert isinstance(x.batch_shape[x.time_dim_axis], int)
        size_t = x.batch_shape[x.time_dim_axis] - start_t
      else:
        seq_lens_t = seq_lens_data.copy_compatible_to(common_data, check_sparse=False).placeholder
        size_t = seq_lens_t - start_t
      size = tf.maximum(tf.reduce_max(size_t), min_size)  # scalar
    elif isinstance(size, LayerBase):
      size_data = size.output.copy_compatible_to(common_data, check_sparse=False)
      size_t = size_data.placeholder
      min_size = 0
      size = tf.maximum(tf.reduce_max(size_t), min_size)  # scalar
    else:
      size_t = None
    # for each start index in start_data, we want to gather a slice
    # therefore, the output's first axes are the same as the ones from start_data
    # and the next axis will therefore be the slice axis
    slice_tag = self.output.dim_tags[start_data.batch_ndim]
    assert slice_tag.description.startswith("sliced-time:")
    if size_t is not None:
      # in this case, size is not known before runtime and becomes dynamic and we need to set dyn_size
      assert not isinstance(size, int)
      dyn_size = tf.maximum(size_t, min_size)  # (B,) or (B,T)
      dyn_size_ext = Data(
        name=("%s:dyn_size" % slice_tag.description),
        dtype=Data.size_dtype,
        placeholder=dyn_size,
        dim_tags=start_data.dim_tags,
        batch=slice_tag.batch,
        beam=slice_tag.batch.beam if slice_tag.batch else self.output.beam,
        control_flow_ctx=slice_tag.control_flow_ctx)
      slice_tag.dyn_size_ext = dyn_size_ext
      slice_tag.set_tag_on_size_tensor(dyn_size)
    gather_positions_data = start_data.copy_template(name="%s_gather_positions" % self.name)
    gather_positions_data = gather_positions_data.copy_add_dim_by_tag(
      slice_tag,
      unbroadcast=True,
      axis=start_data.batch_ndim)
    # [start+0, start+1, ...]
    gather_positions = tf.expand_dims(start_t, -1) + tf.range(0, size)  # e.g. (B, size) or (B, T, size)
    if seq_lens_data is not None:
      seq_lens_t = seq_lens_data.copy_compatible_to(
        gather_positions_data,
        check_sparse=False).placeholder
      pad_mask = tf.logical_or(  # shape like gather_positions
        tf.greater(gather_positions, seq_lens_t - 1),
        tf.less(gather_positions, 0))
      gather_positions = tf.clip_by_value(gather_positions, 0, seq_lens_t - 1)
    else:
      pad_mask = tf.logical_or(  # shape like gather_positions
        tf.greater(gather_positions, x.batch_shape[1] - 1),
        tf.less(gather_positions, 0))
      gather_positions = tf.clip_by_value(gather_positions, 0, x.batch_shape[1] - 1)
    if isinstance(self.size, LayerBase):
      pad_mask = tf.logical_or(tf.greater(gather_positions, tf.expand_dims(start_t + size_t - 1, -1)), pad_mask)
    pad_mask_data = gather_positions_data.copy_template(
      name="%s_gather_positions" % self.name,
      dtype="bool")
    pad_mask_data.placeholder = pad_mask
    gather_positions_data.placeholder = gather_positions
    position = InternalLayer(
      network=self.network,
      name="%s_internal" % gather_positions_data.name,
      output=gather_positions_data)
    gather_layer = GatherLayer(
      name="%s_gather" % self.name,
      network=self.network,
      output=self.output,
      sources=self.sources,
      position=position,
      axis=x.get_time_dim_tag())
    placeholder = gather_layer.output.placeholder
    # In principle, the padded frames are being ignored
    # (unless get_padding_info_dict_ref et al are used).
    # However, you can still end up with gradients for them
    # in unexpected ways.
    # Due to our gather implementation,
    # the gradient flow would go into wrong frames
    # and might lead to unexpected behavior.
    # So to be on the safe side, we do the masking here.
    pad_mask_data = pad_mask_data.copy_compatible_to(gather_layer.output, check_sparse=False, check_dtype=False)
    pad_mask = pad_mask_data.placeholder
    self.output.placeholder = where_bc(pad_mask, tf.zeros_like(placeholder), placeholder)

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    dep_layers = super(SliceNdLayer, self).get_dep_layers() + [self.start]
    if isinstance(self.size, LayerBase):
      dep_layers += [self.size]
    return dep_layers

  @classmethod
  def get_out_data_from_opts(cls, name, sources=(), start=None, size=None, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param LayerBase|None start:
    :param int|LayerBase|None size:
    :rtype: Data
    """
    from ..util.data import DimensionTag
    start_data = start.output.copy()
    input_data = sources[0].output.copy()
    gather_positions_data = start_data.copy_template(name="%s_gather_positions" % name)
    if isinstance(size, LayerBase):
      size = None
    # size might be None here in which case we set the dyn_size in __init__
    tag = DimensionTag(
      kind=DimensionTag.Types.Spatial,
      description="sliced-time:%s" % name,
      dimension=size)
    gather_positions_data = gather_positions_data.copy_add_dim_by_tag(tag, unbroadcast=True, axis=start_data.batch_ndim)
    position = InternalLayer(
      network=sources[0].network,
      name="%s_internal" % gather_positions_data.name,
      output=gather_positions_data)
    return GatherLayer.get_out_data_from_opts(
      name="%s_gather" % name,
      sources=sources,
      position=position,
      axis=input_data.get_time_dim_tag())

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param get_layer:
    """
    super(SliceNdLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["start"] = get_layer(d["start"])
    if isinstance(d["size"], str):
      d["size"] = get_layer(d["size"])


class GatherLayer(_ConcatInputLayer):
  """
  Gathers slices on a specified axis from the input layer using indices from a ``position`` layer.
  If the input is a layer of the shape ``[B,D,F1]``, and position of shape ``[B,F2]``, this will yield output of the
  shape ``[B,F2,F1]`` where

  ``output[b,f2,f1] = input[b,position[b,f2],f1]``

  (if ``D`` is the axis to gather from).
  In general, all shared axes of the input and the positions will be considered as batch-axes.

  The ``position`` argument can also be an ``int``.
  In this case, this simply gives ``input[position]`` one the specified ``axis``.

  It's basically a wrapper around ``tf.gather``.
  It provides the same functionality as the deprecated ``GatherNdLayer``, but is more generic.
  See also :class:`GatherNdLayer`.
  """
  layer_class = "gather"

  def __init__(self, position, axis, **kwargs):
    """
    :param LayerBase|int position: Layer containing the indices used to select the slices of the input from.
      If another layer, must be of type ``int32`` or ``int64``.
      Can also specify a constant ``int``.
    :param str axis: The axis into which we gather the indices into
    """
    super(GatherLayer, self).__init__(**kwargs)
    self.position = position

    input_data = self.input_data
    if isinstance(position, int):
      position_data = Data.from_tensor(tf.constant(position))
    else:
      position_data = position.output
    old_gather_axis = input_data.get_axis_from_description(axis, allow_int=False)  # might be moved later

    # determine all common axes of input_data and position_data
    common_axes_input, common_axes_position, input_axes, position_axes = (
      self._get_common_input_position_axes(input_data, position_data, old_gather_axis))
    batch_dims = len(common_axes_input)

    # move gather axis back if necessary s.t. common axes are always in front
    gather_axis = max(batch_dims, old_gather_axis)

    # (BatchAxes.., InputAxesBeforeGatherAxis, gather_axis, InputAxesAfterGatherAxis..)
    params = tf.transpose(
      input_data.placeholder,
      common_axes_input + input_axes[:gather_axis-batch_dims] + [old_gather_axis] + input_axes[gather_axis-batch_dims:])
    # (BatchAxes.., PositionAxes..)
    indices = tf.transpose(position_data.placeholder, common_axes_position + position_axes)
    # (BatchAxes.., InputAxesBeforeGatherAxis, PositionAxes.., InputAxesAfterGatherAxis..)
    self.output.placeholder = tf.gather(params=params, indices=indices, axis=gather_axis, batch_dims=batch_dims)

  @classmethod
  def _get_common_input_position_axes(cls, input_data, position_data, old_gather_axis):
    """
    Determine all common axes of input_data and position_data.
    All other axes will be specific axes then (except for the gather axis of ``input_data``).
    Will not change the order of the position axes.

    :param Data input_data:
    :param Data position_data:
    :param int old_gather_axis: gather axis of ``input_data`` counted with batch dim (before any transformation)
    :rtype: (list[int], list[int], list[int], list[int])
    :return: (common_axes_input, common_axes_position, specific_input_axes, specific_position_axes), all counted with
    batch dim.
    """
    is_equal_opts = dict(ignore_feature_dim=True, allow_same_spatial_dim=True, broadcast_matches=True)
    common_axes_pairs = [
      (input_axis, position_axis)
      for input_axis in range(input_data.batch_ndim) for position_axis in range(position_data.batch_ndim)
      if not input_axis == old_gather_axis
      if input_data.get_dim_tag(input_axis).is_equal(position_data.get_dim_tag(position_axis), **is_equal_opts)
    ]
    common_axes_input, common_axes_position = zip(*common_axes_pairs) if common_axes_pairs else ([], [])
    common_axes_input, common_axes_position = list(common_axes_input), list(common_axes_position)
    specific_input_axes = [
      axis for axis in range(input_data.batch_ndim) if axis not in common_axes_input and not axis == old_gather_axis]
    specific_position_axes = [axis for axis in range(position_data.batch_ndim) if axis not in common_axes_position]
    return common_axes_input, common_axes_position, specific_input_axes, specific_position_axes

  @classmethod
  def _translate_input_axis(
    cls, input_axis, old_gather_axis, common_axes_input, input_axes, position_axes):
    """
    :param int input_axis: batch axis of input_data
    :param int old_gather_axis: gather axis of ``input_data`` counted with batch dim
    :param list[int] common_axes_input:
    :param list[int] input_axes:
    :param list[int] position_axes:
    :rtype: int|None
    :return: batch axis of output
    """
    if input_axis in common_axes_input:
      return common_axes_input.index(input_axis)
    elif input_axis < old_gather_axis:
      return len(common_axes_input) + input_axes.index(input_axis)
    elif input_axis == old_gather_axis:
      return None  # the gather axis will not be present in the output
    else:
      return len(common_axes_input) + len(position_axes) + input_axes.index(input_axis)

  @classmethod
  def _translate_position_axis(
    cls, position_axis, old_gather_axis, common_axes_position, position_axes):
    """
    :param int position_axis: batch axis of position_data
    :param int old_gather_axis: gather axis of ``input_data`` counted with batch dim
    :param list[int] common_axes_position:
    :param list[int] position_axes:
    :rtype: int
    :return: batch axis of output
    """
    if position_axis in common_axes_position:
      return common_axes_position.index(position_axis)
    else:
      num_input_axes_before = max(0, old_gather_axis - len(common_axes_position))
      return len(common_axes_position) + num_input_axes_before + position_axes.index(position_axis)

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    if isinstance(self.position, LayerBase):
      return super(GatherLayer, self).get_dep_layers() + [self.position]
    else:
      return super(GatherLayer, self).get_dep_layers()

  @classmethod
  def get_out_data_from_opts(cls, name, sources, position, axis, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param LayerBase|int position:
    :param str axis:
    :rtype: Data
    """
    input_data = get_concat_sources_data_template(sources)
    old_gather_axis = input_data.get_axis_from_description(axis, allow_int=False)

    if isinstance(position, int):
      position_data = Data(name="constant_position", dtype="int32", shape=(), batch_dim_axis=None)
    else:
      position_data = position.output.copy_template()
    assert position_data.dtype in ["int32", "int64"]

    # determine all common axes of input_data and position_data
    common_axes_input, common_axes_position, input_axes, position_axes = (
      cls._get_common_input_position_axes(input_data, position_data, old_gather_axis))
    batch_dims = len(common_axes_input)

    # move gather axis back if necessary s.t. common axes are always in front
    gather_axis = max(batch_dims, old_gather_axis)

    # (BatchAxes.., InputAxesBeforeGatherAxis, PositionAxes.., InputAxesAfterGatherAxis..)
    dim_tags = (
      [input_data.dim_tags[ax] for ax in common_axes_input] +
      [input_data.dim_tags[ax] for ax in input_axes[:gather_axis-batch_dims]] +
      [position_data.dim_tags[ax] for ax in position_axes] +
      [input_data.dim_tags[ax] for ax in input_axes[gather_axis-batch_dims:]])
    out_type = input_data.get_kwargs(include_special_axes=False)
    out_type["name"] = "%s_output" % name
    out_type["dim_tags"] = dim_tags
    out_type["beam"] = SearchBeam.get_combined_beam(input_data.beam, position_data.beam)
    out_type["available_for_inference"] = input_data.available_for_inference and position_data.available_for_inference

    # Take axes from input_data if they exist there, otherwise from position_data
    for axis_kind in Data.SpecialAxesNames:
      input_axis, position_axis = getattr(input_data, axis_kind), getattr(position_data, axis_kind)
      if input_axis is not None and input_axis != old_gather_axis:
        out_type[axis_kind] = cls._translate_input_axis(
          input_axis, old_gather_axis, common_axes_input, input_axes, position_axes)
      elif position_axis is not None:
        out_type[axis_kind] = cls._translate_position_axis(
          position_axis, old_gather_axis, common_axes_position, position_axes)
      else:
        out_type[axis_kind] = None
    # feature_dim_axis needs to be handled differently if it is NotSpecified
    if (input_data.feature_dim_axis_or_unspecified is NotSpecified and
            position_data.feature_dim_axis_or_unspecified is NotSpecified):
      out_type["feature_dim_axis"] = NotSpecified
    elif input_data.feature_dim_axis_or_unspecified is NotSpecified:
      assert position_data.feature_dim_axis_or_unspecified is not NotSpecified
      if position_data.feature_dim_axis is not None:
        out_type["feature_dim_axis"] = cls._translate_position_axis(
          position_data.feature_dim_axis, old_gather_axis, common_axes_position, position_axes)
      else:
        out_type["feature_dim_axis"] = NotSpecified
    elif position_data.feature_dim_axis_or_unspecified is NotSpecified:
      assert input_data.feature_dim_axis_or_unspecified is not NotSpecified
      if input_data.feature_dim_axis is not None:
        out_type["feature_dim_axis"] = cls._translate_input_axis(
          input_data.feature_dim_axis, old_gather_axis, common_axes_input, input_axes, position_axes)
      else:
        out_type["feature_dim_axis"] = NotSpecified
    else:
      assert input_data.feature_dim_axis_or_unspecified is not NotSpecified
      assert position_data.feature_dim_axis_or_unspecified is not NotSpecified
      pass  # keep the logic as before

    # If not sparse, the feature dim axis could now originate from position, let Data figure this out
    if not out_type.get("sparse", False):
      out_type["dim"] = NotSpecified

    output_data = Data(**out_type)

    # Take size_placeholder from input_data if they exist there, otherwise from position_data
    size_placeholder = {}
    for input_axis, size in input_data.size_placeholder.items():
      input_axis = input_data.get_batch_axis(input_axis)
      if input_axis == old_gather_axis:
        continue
      output_axis = output_data.get_batch_axis_excluding_batch(
        cls._translate_input_axis(input_axis, old_gather_axis, common_axes_input, input_axes, position_axes))
      size_placeholder[output_axis] = size
    for position_axis, size in position_data.size_placeholder.items():
      position_axis = position_data.get_batch_axis(position_axis)
      output_axis = output_data.get_batch_axis_excluding_batch(cls._translate_position_axis(
        position_axis, old_gather_axis, common_axes_position, position_axes))
      size_placeholder.setdefault(output_axis, size)
    output_data.size_placeholder = size_placeholder

    return output_data

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param get_layer:
    """
    super(GatherLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)

    if not isinstance(d["position"], int):
      assert isinstance(d["position"], str), "Invalid 'position' type %s" % type(d["position"])
      d["position"] = get_layer(d["position"])


class GatherNdLayer(_ConcatInputLayer):
  """
  Warning: This layer is deprecated, use the more general :class:`GatherLayer` instead.
  :class:`GatherLayer` should be equivalent, but is more general (supports multiple batch dimensions, can specify gather
   axis) and its name is less misleading.

  This takes out a position from some axis, e.g. ``x[pos]``.
  This layers allows a different position for each batch.
  It's basically a wrapper around ``tf.gather`` (the name of this layer is misleading).
  See also :class:`GatherLayer` instead, which will replace this layer in the future.
  See also :class:`SliceNdLayer`.
  See also :class:`ScatterNdLayer`, which is the inverse operation.
  """
  layer_class = "gather_nd"

  def __init__(self, position, **kwargs):
    """
    :param LayerBase position: indices into first axis (excluding batch) of the input
    """
    super(GatherNdLayer, self).__init__(**kwargs)
    self.position = position
    from returnn.tf.util.basic import batch_gather
    x = self.input_data.copy_as_batch_major()
    position = position.output
    self.output.size_placeholder = position.size_placeholder.copy()
    for i in range(position.ndim, self.output.ndim):
      j = i - position.ndim + 1
      if j in x.size_placeholder:
        self.output.size_placeholder[i] = x.size_placeholder[j]
    if position.have_batch_axis():
      position = position.copy_as_batch_major()
    else:
      position = position.copy_add_batch_dim(batch_dim_axis=0, batch=self.input_data.batch)
    self.output.placeholder = batch_gather(x.placeholder, position.placeholder)  # (B,...)

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    return super(GatherNdLayer, self).get_dep_layers() + [self.position]

  @classmethod
  def get_out_data_from_opts(cls, name, sources, position, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param LayerBase position:
    :rtype: Data
    """
    input_data = get_concat_sources_data_template(sources).copy_as_batch_major()
    assert input_data.have_batch_axis()
    position_data = position.output.copy_template()
    if position_data.have_batch_axis():
      position_data = position_data.copy_as_batch_major()
    else:
      position_data = position_data.copy_add_batch_dim(batch_dim_axis=0, batch=input_data.batch)
    dim_tags = list(position_data.dim_tags) + list(input_data.dim_tags[2:])  # (B, ...) (w/o batch)
    out_type = position_data.get_kwargs(include_special_axes=False)
    out_type["name"] = "%s_output" % name
    out_type["dim_tags"] = dim_tags
    if position_data.time_dim_axis is None:
      if input_data.time_dim_axis is not None and input_data.time_dim_axis_excluding_batch >= 1:
        out_type["time_dim_axis"] = len(dim_tags) + input_data.time_dim_axis_excluding_batch - 2
    out_type["dim"] = input_data.dim
    out_type["sparse"] = input_data.sparse
    out_type["dtype"] = input_data.dtype
    out_type["beam"] = SearchBeam.get_combined_beam(position_data.beam, input_data.beam)
    return Data(**out_type)

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param get_layer:
    """
    super(GatherNdLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["position"] = get_layer(d["position"])


class ScatterNdLayer(_ConcatInputLayer):
  """
  The inverse of :class:`GatherNdLayer`.
  Mostly a wrapper for ``tf.scatter_nd``.

  The input to the layer are the ``updates``, the ``indices`` are via the ``position`` argument.
  The indices are into the newly constructed output dimension.
  The output shape is constructed via the common shape of the input, the position,
  and the the unique common axis (if not unique, we would need to introduce an option to specify it)
  is replaced by the given output dimension (currently via ``output_dim_via_time_from``).

  Examples::

    position (indices): (B,eTs)
    input (updates): (eTs,D) or (B,eTs,D) -> expanded to (B,eTs,D)
    output shape: (B,eT,D)

    position (indices): (B,dT,eTs)
    input (updates): (eTs,D) -> expanded to (B,dT,eTs,D)
    output shape: (B,dT,eT,D)

    position (indices): (dT,eTs)
    input (updates): (eTs,D) -> expanded to (dT,eTs,D)
    output shape: (dT,eTs,D)

    position (indices): (dT,eTs)
    input (updates): (B,eTs,D) -> expanded to (dT,eTs,B,D)
    output shape: (dT,eT,B,D)

  In all these examples, output_dim_via_time_from is (B,eT,F), and eTs gets replaced by eT.
  """
  layer_class = "scatter_nd"

  def __init__(self, position, position_axis, output_dim_via_time_from, filter_invalid_indices=False, **kwargs):
    """
    :param LayerBase position: indices into first axis (excluding batch) of the output
    :param str|int position_axis: axis in `position` to replace by the output-dim
    :param LayerBase output_dim_via_time_from: use the time-dim from this layer as the output-dim
    :param bool filter_invalid_indices: allow for indices <0 or >= output_dim, which will be discarded in the output
    """
    super(ScatterNdLayer, self).__init__(**kwargs)
    self.position = position
    common, output, replace_common_axis, input_extra_axes = self._get_axes(
      input_data=self.input_data, position=position.output, position_axis=position_axis,
      output_dim_via_time_from=output_dim_via_time_from.output)
    pos_v = position.output.placeholder
    pos_ndim = position.output.batch_ndim
    assert 0 <= replace_common_axis < pos_ndim
    pos_shape = [position.output.get_dim(i) for i in range(pos_ndim)]
    output_dim = output_dim_via_time_from.output.time_dimension()
    input_shape = pos_shape + [self.input_data.get_dim(i) for i in input_extra_axes]
    input_expanded = self.input_data.copy_compatible_to(common, unbroadcast=True)
    input_v = input_expanded.placeholder
    if filter_invalid_indices:
      mask = tf.logical_or(tf.less(pos_v, 0), tf.greater_equal(pos_v, output_dim))
      input_v = tf_util.where_bc(
        tf_util.expand_multiple_dims(mask, [-1] * (common.batch_ndim - position.output.batch_ndim)),
        0.0, input_v)
      # It does not matter what dummy indices we use, as we have 0.0 updates, but it must be valid (0 is valid).
      pos_v = tf.where(mask, tf.zeros_like(pos_v), pos_v)
    # Now we need to implement a similar logic as `TFUtil.nd_indices`, but more generic.
    idxs = [
      (tf.reshape(tf.range(pos_shape[i], dtype=pos_v.dtype), [1] * i + [pos_shape[i]] + [1] * (pos_ndim - i - 1))
       + tf.zeros_like(pos_v))
      if i != replace_common_axis else
      pos_v
      for i in range(pos_ndim)]
    nd_idxs = tf.stack(idxs, axis=-1)
    # updates.shape == indices.shape[:-1] + output_shape[indices.shape[-1]:]
    assert pos_ndim <= input_expanded.batch_ndim == self.output.batch_ndim
    output_shape = [
      input_shape[i] if i != replace_common_axis else output_dim
      for i in range(input_expanded.batch_ndim)]
    self.output.placeholder = tf.scatter_nd(
      indices=nd_idxs, updates=input_v, shape=output_shape)

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    return super(ScatterNdLayer, self).get_dep_layers() + [self.position]

  @classmethod
  def _get_axes(cls, input_data, position, position_axis, output_dim_via_time_from):
    """
    :param Data input_data: updates
    :param Data position: indices
    :param str|int position_axis: axis in `position` to replace by the output-dim
    :param Data output_dim_via_time_from:
    :rtype: (Data, Data, int, list[int])
    :return: common, output, axis, input_extra_axes
    """
    from returnn.tf.util.basic import DimensionTag
    # Construct `common` manually, not via Data.get_common_data, such that we can control the axis order.
    # We want the same axis from `position`, and all further axes should be added behind that.
    common = position.copy_template()
    common.dtype = input_data.dtype
    common.vocab = input_data.vocab
    common.sparse = input_data.sparse
    if common.sparse:
      common.dim = input_data.dim
    elif common.feature_dim_axis is not None:
      common.dim = common.batch_shape[common.feature_dim_axis]
    else:
      common.dim = None
    common.sanity_check()
    dim_tags, tags_dict = DimensionTag.get_all_dimension_tags(
      [common, input_data], dict(allow_same_feature_dim=True, treat_feature_as_spatial=True))
    common_dim_tags = tags_dict[common]
    input_extra_dim_tags = list(tags_dict[input_data])
    input_extra_axes = []
    for tag in dim_tags:
      if tag not in common_dim_tags:
        common = common.copy_add_dim_by_tag(tag, unbroadcast=True, axis=-1)
        input_extra_axes.append(input_extra_dim_tags.index(tag))
        input_extra_dim_tags[input_extra_axes[-1]] = None
        common_dim_tags.append(tag)
    position_axis = position.get_axis_from_description(position_axis)
    assert position_axis != position.batch_dim_axis
    if common.time_dim_axis is None:
      common.time_dim_axis = position_axis
    output_dim = output_dim_via_time_from.batch_shape[output_dim_via_time_from.time_dim_axis]
    output_size = output_dim_via_time_from.size_placeholder.get(
      output_dim_via_time_from.time_dim_axis_excluding_batch, None)
    output = common.copy_template_replace_dim(axis=position_axis, new_dim=output_dim, new_size=output_size)
    return common, output, position_axis, input_extra_axes

  @classmethod
  def get_out_data_from_opts(cls, name, sources, position, position_axis, output_dim_via_time_from, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param LayerBase position:
    :param str|int position_axis: axis in `position` to replace by the output-dim
    :param LayerBase output_dim_via_time_from:
    :rtype: Data
    """
    input_data = get_concat_sources_data_template(sources)
    common, output, replace_common_axis, input_extra_axes = cls._get_axes(
      input_data=input_data, position=position.output, position_axis=position_axis,
      output_dim_via_time_from=output_dim_via_time_from.output)
    return output.copy_template(name="%s_output" % name)

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param (str)->LayerBase get_layer:
    """
    super(ScatterNdLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["position"] = get_layer(d["position"])
    d["output_dim_via_time_from"] = get_layer(d["output_dim_via_time_from"])


class LinearLayer(_ConcatInputLayer):
  """
  Linear/forward/fully-connected/1x1-conv layer.
  Does a linear transformation on the feature-dimension of the input
  with an optional bias term and an optional activation function.
  See also :class:`DotLayer`, :class:`ElemwiseProdLayer`, :class:`WeightedSumLayer`.
  """
  layer_class = "linear"

  def __init__(self, activation=None, with_bias=True, grad_filter=None, forward_weights_init="glorot_uniform",
               bias_init=0.0, use_transposed_weights=False, **kwargs):
    """
    :param str|None activation: e.g. "relu", or None
    :param bool with_bias:
    :param float|None grad_filter: if grad norm is higher than this threshold (before activation), the grad is removed
    :param str forward_weights_init: see :func:`returnn.tf.util.basic.get_initializer`
    :param str recurrent_weights_init: see :func:`returnn.tf.util.basic.get_initializer`
    :param str|float bias_init: see :func:`returnn.tf.util.basic.get_initializer`
    :param bool use_transposed_weights: If True, define the weight matrix with transposed dimensions (n_out, n_in).
    """
    super(LinearLayer, self).__init__(**kwargs)
    from returnn.tf.util.basic import get_initializer

    self.activation = activation
    self.with_bias = with_bias
    self.use_transposed_weights = use_transposed_weights

    input_data = self.input_data
    n_in = input_data.dim
    n_out = self.output.dim
    assert n_in and n_out, "%r and %r" % (input_data, self.output)
    in_split_info = self._get_in_split_info()

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

      weights = self.add_param(tf_compat.v1.get_variable(
        name="W", shape=weights_shape, dtype=tf.float32, initializer=fwd_weights_initializer))
      weights_ = weights
      if in_split_info:
        tf_util.set_param_axes_split_info(
          weights, [[n_out], in_split_info] if self.use_transposed_weights else [in_split_info, [n_out]])

      if self.use_transposed_weights:
        weights = tf.transpose(weights)

      if self.with_bias:
        bias_initializer = get_initializer(
          bias_init, seed=self.network.random.randint(2 ** 31) if bias_init else 0, eval_local_ns={"layer": self})
        b = self.add_param(tf_compat.v1.get_variable(
          name="b", shape=(n_out,), dtype=tf.float32, initializer=bias_initializer))
      else:
        assert not bias_init
        b = None

    with tf.name_scope("linear"):
      from returnn.tf.util.basic import dot, to_int32_64, is_gpu_available_in_session, move_axis
      x = input_data.placeholder
      ndim = x.get_shape().ndims

      if self.input_data.sparse:
        # Maybe optionally we could also use tf.contrib.layers.safe_embedding_lookup_sparse().
        x = tf.nn.embedding_lookup(weights, to_int32_64(x))
        ndim += 1
      elif self.input_data.feature_dim_axis == self.input_data.batch_ndim - 1:
        x = dot(x, weights_, transpose_b=self.use_transposed_weights)
      elif self.input_data.is_batch_feature_major and is_gpu_available_in_session():  # CuDNN has fast version for this
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
      x = tf_util.filter_grad(
        x,
        threshold=grad_filter,
        axis=[i for i in range(input_data.batch_ndim) if i != input_data.batch_dim_axis])

    if self.activation:
      from returnn.tf.util.basic import get_activation_function
      act_func = get_activation_function(self.activation)
      if act_func in {tf.nn.softmax, tf.nn.log_softmax} and self.output.feature_dim_axis != self.output.batch_ndim - 1:
        # Make sure we use the right axis. Don't use OutputWithActivation.
        # noinspection PyArgumentList
        x = act_func(x, axis=self.output.feature_dim_axis)
        self.output_before_activation = None
      else:
        self.output_before_activation = OutputWithActivation(x, act_func=act_func)
    else:
      self.output_before_activation = OutputWithActivation(x)
    if self.output_before_activation:
      x = self.output_before_activation.y

    assert self.output.batch_dim_axis == self.input_data.batch_dim_axis
    assert self.output.time_dim_axis == self.input_data.time_dim_axis
    self.output.placeholder = x

  def _get_in_split_info(self):
    """
    :rtype: list[int]|None
    """
    n_in = self.input_data.dim
    in_split_info = []
    src_queue = list(self.sources)
    while src_queue:
      src = src_queue.pop(0)
      if isinstance(src, CopyLayer):  # special case, contains itself of multiple sources maybe
        src_queue = list(src.sources) + src_queue
        continue
      in_split_info.append(src.output.dim)
    if not all(in_split_info) or sum(in_split_info) != n_in:
      print(
        "%s: Warning: input split dims %r unclear for sources %r?" % (self, in_split_info, self.sources), file=log.v3)
      return None
    return in_split_info


class SoftmaxLayer(LinearLayer):
  """
  Just a LinearLayer with activation="softmax" by default.
  """
  layer_class = "softmax"

  def __init__(self, **kwargs):
    super(SoftmaxLayer, self).__init__(activation="softmax", **kwargs)


class LengthLayer(LayerBase):
  """
  Returns the length of sources as (B,), via input size_placeholder.
  """
  layer_class = "length"

  # noinspection PyUnusedLocal
  def __init__(self, axis="T", add_time_axis=False, dtype="int32", sparse=False, **kwargs):
    """
    :param str|DimensionTag axis:
    :param bool add_time_axis:
    :param str dtype:
    :param bool sparse:
    """
    super(LengthLayer, self).__init__(**kwargs)
    assert len(self.sources) == 1, "%s: expects one source" % self
    source = self.sources[0].output
    axis = source.get_axis_from_description(axis, allow_int=False)
    dim = source.dim_tags[axis]
    self.dim_tag = dim
    if add_time_axis:
      self.output.placeholder = tf.expand_dims(dim.dyn_size, axis=self.output.time_dim_axis)
    else:
      self.output.placeholder = dim.dyn_size_ext.placeholder

  @classmethod
  def get_out_data_from_opts(cls, name, sources, axis="T", add_time_axis=False, dtype="int32", sparse=False, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param str|DimensionTag axis:
    :param bool add_time_axis:
    :param str dtype:
    :param bool sparse:
    :rtype: Data
    """
    assert len(sources) == 1
    source = sources[0].output
    axis = source.get_axis_from_description(axis, allow_int=False)
    dim = source.dim_tags[axis]
    if add_time_axis:
      assert dim.dyn_size_ext and dim.dyn_size_ext.have_batch_axis() and dim.dyn_size_ext.batch_ndim == 1  # [B]
      return Data(
        name="%s_length" % name,
        shape=[1], batch_dim_axis=0, time_dim_axis=1,
        dtype=dtype, sparse=sparse, dim=None if sparse else NotSpecified)
    if not dim.dyn_size_ext:  # yet undefined
      return Data(
        name="%s_length" % name,
        shape=(), batch_dim_axis=0, time_dim_axis=None,
        dtype=dtype, sparse=sparse, dim=None if sparse else NotSpecified)
    return dim.dyn_size_ext


class SoftmaxOverSpatialLayer(_ConcatInputLayer):
  """
  This applies a softmax over spatial axis/axes (currently only time axis supported).
  E.g. when the input is of shape (B,T,dim), the output will be (B,T,dim).
  It automatically masks the frames outside the seq defined by the seq-len.
  In contrast to :class:`SoftmaxLayer`, this will not do a linear transformation.
  See :class:`SeqLenMaskLayer` if you just want to apply a masking.
  """
  layer_class = "softmax_over_spatial"

  def __init__(self, axis=None, energy_factor=None,
               start=None, window_start=None, window_size=None, use_time_mask=None,
               log_space=False, **kwargs):
    """
    :param str|None axis: which axis to do the softmax over
    :param float|None energy_factor: the energy will be scaled by this factor.
      This is like a temperature for the softmax.
      In Attention-is-all-you-need, this is set to 1/sqrt(base_ctx.dim).
    :param LayerBase|None start: Tensor of shape (B,) indicating the start frame
    :param LayerBase|int|None window_start: Layer with output of shape (B,) or (constant) int value indicating
      the window start.
    :param LayerBase|int|None window_size: Layer with output of shape (B,) or (constant) int value indicating
      the window size.
    :param bool use_time_mask: if True, assumes dyn seq len, and use it for masking.
      By default, if dyn seq len exists, it uses it.
    :param bool log_space: if True, returns in log space (i.e. uses log_softmax)
    """
    from returnn.tf.util.basic import where_bc, set_padding_info
    super(SoftmaxOverSpatialLayer, self).__init__(**kwargs)
    self.start = start
    self.window_start = window_start
    self.window_size = window_size
    energy_data = self.input_data
    assert energy_data.dtype.startswith("float")
    axis = self._get_axis_to_reduce(input_data=energy_data, axis=axis, exception_prefix=self)
    # tf.nn.softmax operates on the last axis.
    energy_data = energy_data.copy_move_axis(axis, -1)
    energy = energy_data.placeholder
    axis = energy_data.batch_ndim - 1
    # if the time-axis is static, we can skip the masking
    if use_time_mask is None:
      use_time_mask = energy_data.is_axis_dynamic(axis)
    if start or window_start is not None or window_size is not None:
      assert use_time_mask
    if use_time_mask:
      energy_mask = SeqLenMaskLayer.build_mask(
        energy_data,
        axis=axis,
        start=start.output if start else None,
        window_start=window_start.output if isinstance(window_start, LayerBase) else window_start,
        window_size=window_size.output if isinstance(window_size, LayerBase) else window_size)
      energy = where_bc(energy_mask, energy, float("-inf"), name="energy_masked")
    if energy_factor:
      energy = tf.multiply(energy, energy_factor, name="energy_scaled")
    self.output_before_activation = OutputWithActivation(
      energy, act_func=tf.nn.log_softmax if log_space else tf.nn.softmax)  # (...,T)
    self.output.placeholder = self.output_before_activation.y
    if use_time_mask:
      set_padding_info(self.output.placeholder, dim=self.output.dim_tags[axis], pad_value=0.)
    # Never allow inf in output, as softmax should remove all -inf values used for masking
    self.allow_inf_in_output = False

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    deps = super(SoftmaxOverSpatialLayer, self).get_dep_layers()
    if self.start:
      deps.append(self.start)
    if self.window_start:
      deps.append(self.window_start)
    if isinstance(self.window_size, LayerBase):
      deps.append(self.window_size)
    return deps

  @classmethod
  def _get_axis_to_reduce(cls, input_data, axis, exception_prefix):
    """
    :param Data input_data:
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
  def get_out_data_from_opts(cls, name, sources, axis=None, start=None, window_start=None, window_size=None, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param str|None axis:
    :param LayerBase|None start:
    :param LayerBase|None window_start:
    :param LayerBase|int|None window_size:
    :rtype: Data
    """
    out = get_concat_sources_data_template(sources, name="%s_output" % name)
    axis = cls._get_axis_to_reduce(out, axis=axis, exception_prefix="%s %r" % (cls.__name__, name))
    out = out.copy_move_axis(axis, -1)
    if isinstance(start, LayerBase):
      out.beam = SearchBeam.get_combined_beam(out.beam, start.output.beam)
    if isinstance(window_start, LayerBase):
      out.beam = SearchBeam.get_combined_beam(out.beam, window_start.output.beam)
    if isinstance(window_size, LayerBase):
      out.beam = SearchBeam.get_combined_beam(out.beam, window_size.output.beam)
    return out

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param get_layer:
    """
    super(SoftmaxOverSpatialLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    if d.get("start", None):
      d["start"] = get_layer(d["start"])
    if d.get("window_start", None):
      d["window_start"] = get_layer(d["window_start"])
    if d.get("window_size", None):
      if isinstance(d["window_size"], str):
        d["window_size"] = get_layer(d["window_size"])


class SeqLenMaskLayer(_ConcatInputLayer):
  """
  Masks some values away given the seq_len_source with mask_value.
  Also see :class:`SoftmaxOverSpatialLayer`.
  Also see :class:`SwitchLayer`, which can be used to apply a generic mask.
  """
  layer_class = "seq_len_mask"

  def __init__(self, mask_value, axis="T", seq_len_source=None,
               start=None, window_start=None, window_size=None, **kwargs):
    """
    :param LayerBase|None seq_len_source: if not given, uses source
    :param str|int axis:
    :param float mask_value:
    :param LayerBase|None start: Tensor of shape (B,) indicating the start frame
    :param LayerBase|None window_start: Tensor of shape (B,) indicating the window start
    :param LayerBase|int|None window_size:
    """
    from ..util.basic import set_padding_info
    super(SeqLenMaskLayer, self).__init__(**kwargs)
    self.seq_len_source = seq_len_source
    self.start = start
    self.window_start = window_start
    self.window_size = window_size
    assert isinstance(axis, str), "%s: use symbolic axis (e.g. 'T')" % self
    mask = self.build_mask(
      self.input_data,
      axis=axis,
      seq_len_source=seq_len_source.output if seq_len_source else None,
      start=start.output if start else None,
      window_start=window_start.output if isinstance(window_start, LayerBase) else window_start,
      window_size=window_size.output if isinstance(window_size, LayerBase) else window_size)
    from returnn.tf.util.basic import where_bc
    x_ = where_bc(mask, self.input_data.placeholder, mask_value)
    axis_ = self.input_data.get_axis_from_description(axis)
    set_padding_info(x_, dim=self.input_data.dim_tags[axis_], pad_value=mask_value)
    self.output.placeholder = x_
    if mask_value in [float("-inf"), float("inf")]:
      self.allow_inf_in_output = True

  @classmethod
  def build_mask(cls, x, axis="T", seq_len_source=None, start=None, window_start=None, window_size=None):
    """
    :param Data x:
    :param str|int axis:
    :param Data|None seq_len_source:
    :param Data|None start:
    :param Data|None window_start:
    :param Data|int|None window_size:
    :return: mask which is broadcastable to energy_data, thus you can e.g. use :func:`TFUtil.where_bc`
    :rtype: tf.Tensor
    """
    from returnn.tf.util.basic import get_shape
    energy = x.placeholder
    energy_shape = get_shape(energy)
    axis = x.get_axis_from_description(axis)
    assert x.is_axis_dynamic(axis), "%s: use_time_mask True, dyn time axis expected" % x
    if seq_len_source:
      energy_mask = seq_len_source.copy_compatible_to(x).get_sequence_mask_broadcast(axis=axis)
    else:
      energy_mask = x.get_sequence_mask_broadcast(axis=axis)
    if start:
      idxs_shape = [1] * x.batch_ndim  # type: typing.List[typing.Union[int,tf.Tensor]]
      idxs_shape[axis] = energy_shape[axis]
      idxs = tf.reshape(tf.range(energy_shape[axis]), idxs_shape)
      start_data = start.copy_compatible_to(
        x, check_sparse=False, check_dtype=False)  # adds dummy time-dim
      energy_mask = tf.logical_and(energy_mask, tf.greater_equal(idxs, start_data.placeholder))
    if window_start is not None:
      assert window_size, "set window_size explicitly"
      if not isinstance(window_start, Data):
        assert isinstance(window_start, int)
        window_start = Data.from_tensor(tf.constant(window_start))
      if not isinstance(window_size, Data):
        assert isinstance(window_size, int)
        window_size = Data.from_tensor(tf.constant(window_size))
      window_start = window_start.copy_compatible_to(
        x, check_sparse=False, check_dtype=False)  # adds dummy time-dim
      window_size = window_size.copy_compatible_to(
        x, check_sparse=False, check_dtype=False)  # adds dummy time-dim
      idxs_shape = [1] * x.batch_ndim  # type: typing.List[typing.Union[int,tf.Tensor]]
      idxs_shape[axis] = energy_shape[axis]
      idxs = tf.reshape(tf.range(energy_shape[axis]), idxs_shape)
      energy_mask = tf.logical_and(
        energy_mask,
        tf.logical_and(
          tf.greater_equal(idxs, window_start.placeholder),
          tf.less(idxs, window_start.placeholder + window_size.placeholder)
        ))
    return energy_mask

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    deps = super(SeqLenMaskLayer, self).get_dep_layers()
    if self.seq_len_source:
      deps.append(self.seq_len_source)
    if self.start:
      deps.append(self.start)
    if self.window_start:
      deps.append(self.window_start)
    if isinstance(self.window_size, LayerBase):
      deps.append(self.window_size)
    return deps

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param get_layer:
    """
    super(SeqLenMaskLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    if d.get("seq_len_source", None):
      d["seq_len_source"] = get_layer(d["seq_len_source"])
    if d.get("start", None):
      d["start"] = get_layer(d["start"])
    if d.get("window_start", None):
      d["window_start"] = get_layer(d["window_start"])
    if d.get("window_size", None):
      if isinstance(d["window_size"], str):
        d["window_size"] = get_layer(d["window_size"])

  @classmethod
  def get_out_data_from_opts(cls, name, sources, start=None, window_start=None, window_size=None, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param LayerBase|None start:
    :param LayerBase|None window_start:
    :param LayerBase|int|None window_size:
    :rtype: Data
    """
    out = get_concat_sources_data_template(sources, name="%s_output" % name)
    if isinstance(start, LayerBase):
      out.beam = SearchBeam.get_combined_beam(out.beam, start.output.beam)
    if isinstance(window_start, LayerBase):
      out.beam = SearchBeam.get_combined_beam(out.beam, window_start.output.beam)
    if isinstance(window_size, LayerBase):
      out.beam = SearchBeam.get_combined_beam(out.beam, window_size.output.beam)
    return out


class RandIntLayer(LayerBase):
  """
  Generates random numbers using ``tf.random.uniform``
  """
  layer_class = "rand_int"

  # noinspection PyUnusedLocal
  def __init__(self, shape, maxval, minval=0, dtype="int32", seed=None, **kwargs):
    """
    :param tuple[DimensionTag|int]|list[DimensionTag|int] shape: desired shape of output tensor
    :param int maxval: upper bound (exclusive) on range of random values
    :param int minval: lower bound (inclusive) on range of random values
    :param str dtype: type of the output. For random ints, int32 and int64 make sense, but could also be floats
    :param int|None seed: random seed
    """
    from returnn.tf.util.data import DimensionTag
    super(RandIntLayer, self).__init__(**kwargs)

    seed = seed if seed is not None else self.network.random.randint(2 ** 31)

    shape_parsed = []
    for ax, s in enumerate(shape):
      if isinstance(s, int):
        shape_parsed.append(s)
      elif isinstance(s, DimensionTag):
        if s.is_batch_dim():
          shape_parsed.append(self.get_batch_dim())
        elif isinstance(s.dimension, int):
          shape_parsed.append(s.dimension)
        elif s.dimension is None:
          assert s.dyn_size is not None
          shape_parsed.append(tf.reduce_max(s.dyn_size))
        else:
          raise Exception("%s: invalid dim tag %s" % (self, s))
      else:
        raise TypeError("%s: invalid dim %s" % (self, type(s)))
    shape_parsed = tuple(shape_parsed)

    self.output.placeholder = tf.random.uniform(shape_parsed, minval=minval, maxval=maxval, dtype=dtype, seed=seed)

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param (str)->LayerBase get_layer:
    """
    d.setdefault("from", [])
    super(RandIntLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)

  @classmethod
  def get_out_data_from_opts(cls, name, shape, maxval, minval=0, dtype="int32", **kwargs):
    """
    :param str name:
    :param tuple[DimensionTag|int]|list[DimensionTag|int] shape: desired shape of output tensor
    :param int maxval: upper bound (exclusive) on range of random values
    :param int minval: lower bound (inclusive) on range of random values
    :param str dtype: type of the output. For random ints, int32 and int64 make sense, but could also be floats
    :rtype: Data
    """
    from returnn.tf.util.data import DimensionTag

    shape_parsed = []
    batch_dim_axis = None
    dyn_axes_sizes = {}
    for ax, s in enumerate(shape):
      if isinstance(s, int):
        shape_parsed.append(s)
      else:
        assert isinstance(s, DimensionTag)
        if s.is_batch_dim():
          assert batch_dim_axis is None, "Cannot have multiple batch axes"
          batch_dim_axis = ax
        elif isinstance(s.dimension, int):
          shape_parsed.append(s.dimension)
        else:
          assert s.dimension is None
          shape_parsed.append(None)
          dyn_axes_sizes[ax] = s.dyn_size
    shape_parsed = tuple(shape_parsed)

    data = Data(name="%s_output" % name, shape=shape_parsed, dtype=dtype, batch_dim_axis=batch_dim_axis)

    if batch_dim_axis is not None:
      data.size_placeholder = {data.get_batch_axis_excluding_batch(i): size for i, size in dyn_axes_sizes.items()}
    else:
      assert not dyn_axes_sizes, "Cannot have dynamic axes without a batch axis"
      data.size_placeholder = {}

    return data


class RangeLayer(LayerBase):
  """
  Generic wrapper around ``tf.range``.
  See also :class:`RangeInAxisLayer`.
  """
  layer_class = "range"

  # noinspection PyUnusedLocal
  def __init__(self, limit, start=0, delta=1, dtype=None, sparse=False, **kwargs):
    """
    :param int|float limit:
    :param int|float start:
    :param int|float delta:
    :param str|None dtype:
    :param bool sparse:
    """
    super(RangeLayer, self).__init__(**kwargs)
    self.output.placeholder = tf.range(start=start, limit=limit, delta=delta, dtype=self.output.dtype)

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param (str)->LayerBase get_layer:
    """
    d.setdefault("from", [])
    super(RangeLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)

  @classmethod
  def get_out_data_from_opts(cls, name, limit, start=0, delta=1, dtype=None, sparse=False, **kwargs):
    """
    :param str name:
    :param int|float limit:
    :param int|float start:
    :param int|float delta:
    :param str|None dtype:
    :param bool sparse:
    :rtype: Data
    """
    if dtype is None:
      if any([float(arg) != int(arg) for arg in [start, limit, delta]]):
        dtype = "float32"
      else:
        dtype = "int32"
    dim = len(range(start, limit, delta))
    return Data(name="%s_output" % name, shape=(dim,), dim=dim, dtype=dtype, sparse=sparse, batch_dim_axis=None)


class RangeInAxisLayer(LayerBase):
  """
  Assume that the input is e.g. (B,T,D), and you specify axis="T", you will get (B=1,T,D=1),
  where the specified axis is filled with ``tf.range``.
  See also :class:`RangeLayer`.
  """
  layer_class = "range_in_axis"
  recurrent = True  # if axis=="T", the time-dim order matters

  # noinspection PyUnusedLocal
  def __init__(self, axis, dtype="int32", unbroadcast=False, keepdims=False, sparse=False, **kwargs):
    """
    :param str axis:
    :param str dtype:
    :param bool unbroadcast: DEPRECATED, unsupported, and not needed
    :param bool keepdims: DEPRECATED, unsupported, and not needed
    :param bool sparse:
    """
    super(RangeInAxisLayer, self).__init__(**kwargs)
    source = self.sources[0].output
    axis = source.get_axis_from_description(axis)
    axis_wo_b = source.get_batch_axis_excluding_batch(axis)
    from returnn.tf.util.basic import get_shape
    source_shape = get_shape(source.placeholder)
    out = tf.range(0, source_shape[axis], dtype=dtype)
    if unbroadcast:
      raise Exception("%s: do not use unbroadcast")
    if keepdims:
      raise Exception("%s: do not use keepdims")
    self.output.placeholder = out

  @classmethod
  def get_out_data_from_opts(cls, name, sources, axis, dtype="int32", sparse=False, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param str axis:
    :param str dtype:
    :param bool sparse:
    """
    assert len(sources) == 1, "%s layer %r requires single source" % (cls, name)
    source = sources[0].output
    axis = source.get_axis_from_description(axis)
    data_opts = source.get_kwargs(include_special_axes=False)
    dim_tags = [source.dim_tags[axis]]
    if not dim_tags[0].is_batch_dim():
      data_opts.pop("batch", None)
      data_opts.pop("beam", None)
    data_opts["name"] = "%s_output" % name
    data_opts["dim_tags"] = dim_tags
    data_opts["dtype"] = dtype
    data_opts["sparse"] = sparse
    if sparse:
      data_opts["dim"] = None
    else:
      data_opts.pop("dim", None)
    return Data(**data_opts)


class RangeFromLengthLayer(LayerBase):
  """
  Given some dynamic sequence lengths as input, this creates a tf.range over the implied dimension.
  As a side effect, this can create a new dyn dim tag for the given sequence lengths.
  This side effect can be the main functionality in certain use cases.
  See also :class:`RangeInAxisLayer`.

  Consider the example::

    y: {class: range_in_axis, from: x, axis: T}

  This is basically equivalent to::

    x_len: {class: length, from: x}
    y: {class: range_from_length, from: x_len}

  """
  layer_class = "range_from_length"
  recurrent = True

  # noinspection PyUnusedLocal
  def __init__(self, dtype="int32", sparse=False, **kwargs):
    """
    :param str axis:
    :param str dtype:
    :param bool sparse:
    """
    super(RangeFromLengthLayer, self).__init__(**kwargs)
    source = self.sources[0].output
    assert source.placeholder is self.output.dim_tags[0].dyn_size_ext.placeholder
    out = tf.range(0, tf.reduce_max(source.placeholder), dtype=dtype)
    self.output.placeholder = out

  @classmethod
  def get_out_data_from_opts(cls, name, sources, dtype="int32", sparse=False, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param str dtype:
    :param bool sparse:
    """
    assert len(sources) == 1, "%s layer %r requires single source" % (cls, name)
    source = sources[0].output
    dim_tag = None
    if source.placeholder is not None:
      dim_tag = DimensionTag.get_tag_from_size_tensor(source.placeholder)
    if not dim_tag:
      dim_tag = DimensionTag(
        kind=DimensionTag.Types.Spatial, description="%s_input_len" % name,
        batch=source.batch, control_flow_ctx=source.control_flow_ctx,
        dyn_size_ext=source)
      if source.placeholder is not None:
        dim_tag.set_tag_on_size_tensor(source.placeholder)
    return Data(name="%s_output" % name, dim_tags=[dim_tag], dtype=dtype, sparse=sparse, dim=None)


class BatchSoftmaxLayer(_ConcatInputLayer):
  """
  Softmax over spacial and feature axis
  """
  layer_class = "batch_softmax"

  def __init__(self, **kwargs):
    from returnn.tf.util.basic import sequence_mask
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

  # noinspection PyUnusedLocal
  def __init__(self, sources, value=0., dtype=None, with_batch_dim=False, **kwargs):
    """
    :param list[LayerBase] sources:
    :param int|float|bool value:
    :param str|None dtype:
    :param bool with_batch_dim:
    """
    assert not sources, "constant layer cannot have sources"
    super(ConstantLayer, self).__init__(**kwargs)
    value = tf.constant(value, dtype=self.output.dtype)
    if with_batch_dim:
      # Add batch-dim to the constant.
      from returnn.tf.util.basic import expand_dims_unbroadcast
      value = expand_dims_unbroadcast(value, axis=0, dim=self.get_batch_dim())
    self.output.placeholder = value

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param returnn.tf.network.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    d.setdefault("from", [])
    super(ConstantLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)

  @classmethod
  def get_out_data_from_opts(cls, name, value=0., dtype=None, with_batch_dim=False, **kwargs):
    """
    :param str name:
    :param int|float|bool value:
    :param str|None dtype:
    :param bool with_batch_dim:
    :rtype: Data
    """
    return Data.template_from_constant(value, name="%s_const" % name, dtype=dtype, with_batch_dim=with_batch_dim)


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
    from returnn.tf.util.basic import get_activation_function
    act_func = get_activation_function(activation)
    gate_act_func = get_activation_function(gate_activation)
    if len(self.sources) == 2 and self.sources[0].output.dim == self.sources[1].output.dim == self.output.dim:
      # Don't need to concat and then split; we can use the sources directly.
      a = self.sources[0].output.copy_compatible_to(self.output).placeholder
      b = self.sources[1].output.copy_compatible_to(self.output).placeholder
    else:
      a, b = tf.split(self.input_data.placeholder, 2, axis=self.input_data.feature_dim_axis)
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
    from ..util.data import DimensionTag
    input_data = get_concat_sources_data_template(sources)
    assert not input_data.sparse
    assert input_data.dim % 2 == 0
    dim = input_data.dim // 2
    new_dim_tag = DimensionTag(kind=DimensionTag.Types.Feature, description="%s:gating" % name, dimension=dim)
    if n_out is not NotSpecified:
      assert n_out == dim
    return Data(
      name="%s_output" % name,
      dtype=input_data.dtype,
      dim_tags=[
        new_dim_tag if i == input_data.feature_dim_axis else d
        for (i, d) in enumerate(input_data.dim_tags)],
      sparse=False,
      time_dim_axis=input_data.time_dim_axis,
      feature_dim_axis=input_data.feature_dim_axis_or_unspecified,
      batch=input_data.batch)


class WindowLayer(_ConcatInputLayer):
  """
  Adds a window dimension.
  By default, uses the time axis and goes over it with a sliding window.
  The new axis for the window is created right after the time axis.
  Will always return as batch major mode.
  E.g. if the input is (batch, time, dim), the output is (batch, time, window_size, dim).
  If you want to merge the (window_size, dim) together to (window_size * dim,),
  you can use the MergeDimsLayer, e.g. {"class": "merge_dims", "axes": "except_time"}.
  Use stride==window_size and window_right=window_size - 1 in combination with a
  MergeDimsLayer to achieve feature stacking with right-hand zero padding.

  This is not to take out a window from the time-dimension.
  See :class:`SliceLayer` or :class:`SliceNdLayer`.
  """
  layer_class = "window"
  recurrent = True  # we must not allow any shuffling in the time-dim or so

  def __init__(self, window_size, window_left=None, window_right=None, axis="T", padding="same", stride=1, **kwargs):
    """
    :param int window_size:
    :param int|None window_left:
    :param int|None window_right:
    :param str axis: see Data.get_axis_from_description()
    :param str padding: "same" or "valid"
    :param int stride: return only each Nth window
    :param kwargs:
    """
    super(WindowLayer, self).__init__(**kwargs)
    data = self.input_data.copy_as_batch_major()
    if axis == "T" and data.time_dim_axis is None:
      # Assume inside RecLayer.
      axis = None
      assert self._rec_previous_layer, "%s: expected to be used inside a RecLayer" % self
      assert padding == "same"
      prev_state = self._rec_previous_layer.rec_vars_outputs["state"]  # (batch,window,...)
      next_state = tf.concat(
        [prev_state[:, 1:], tf.expand_dims(data.placeholder, axis=1)], axis=1)  # (batch,window,...)
      self.rec_vars_outputs["state"] = next_state
      self.output.placeholder = next_state
    else:
      axis = data.get_axis_from_description(axis)
      from returnn.tf.util.basic import windowed_nd
      self.output.placeholder = windowed_nd(
        data.placeholder,
        window_size=window_size, window_left=window_left, window_right=window_right,
        padding=padding, time_axis=axis, new_window_axis=axis + 1, stride=stride)
    self.output.placeholder.set_shape(tf.TensorShape(self.output.batch_shape))
    self.output.size_placeholder = self.input_data.size_placeholder.copy()
    if axis is not None:
      axis_wo_b = self.output.get_batch_axis_excluding_batch(axis)
      if axis_wo_b in self.output.size_placeholder:
        size = self.output.size_placeholder[axis_wo_b]
        from ..util.basic import same_control_flow_ctx
        from ..util.data import DimensionTag
        with same_control_flow_ctx(size):
          size = ConvLayer.calc_out_dim(
            in_dim=size,
            filter_size=window_size, stride=stride, dilation_rate=1, padding=padding)
        DimensionTag(
          kind=DimensionTag.Types.Spatial, description="%s:window:%i" % (self.name, axis_wo_b),
          dimension=None, dyn_size=size, batch=self.output.batch,
          src_data=self.output, src_axis=axis)
        self.output.size_placeholder[axis_wo_b] = size

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
    :param returnn.tf.layers.rec.RecLayer|LayerBase rec_layer:
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
    :param returnn.tf.layers.rec.RecLayer|LayerBase rec_layer:
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

  def __init__(self, axes, padding, value=0, mode="constant", **kwargs):
    """
    :param str|list[str] axes: e.g. "F" etc. see :func:`Dataset.get_axes_from_description`.
    :param list[(int,int)]|(int,int)|int padding: how much to pad left/right in each axis
    :param int|float value: what constant value to pad, with mode=="constant"
    :param str mode: "constant", "reflect", "symmetric" and "replication"
    """
    from returnn.tf.util.data import DimensionTag
    super(PadLayer, self).__init__(**kwargs)
    axes = self.input_data.get_axes_from_description(axes)
    padding = self._transform_padding(padding=padding, axes=axes)
    paddings = [(0, 0)] * len(range(self.input_data.batch_ndim))
    for i, a in enumerate(axes):
      paddings[a] = padding[i]
    mode = mode.upper()
    if all(sum(p) == 0 for p in padding):
      self.output.placeholder = self.input_data.placeholder
    elif mode == "REPLICATION":
      self.output.placeholder = tf_util.pad_replicate(self.input_data.placeholder, axes, padding)
    else:
      self.output.placeholder = tf.pad(self.input_data.placeholder, paddings=paddings, mode=mode, constant_values=value)
    for a in axes:
      p = sum(paddings[a])
      in_tag = self.input_data.dim_tags[a]
      out_tag = self.output.dim_tags[a]
      a = self.input_data.get_batch_axis_excluding_batch(a)
      if a is None:
        continue
      if in_tag.dyn_size is None:
        continue
      if p == 0:
        continue
      size = in_tag.dyn_size
      with tf_util.same_control_flow_ctx(size):
        size = tf_util.simplify_add(size, p)
      size_tag = DimensionTag.get_tag_from_size_tensor(size)
      if not size_tag:
        out_tag.set_tag_on_size_tensor(size, batch=in_tag.batch)
      else:
        out_tag.declare_same_as(size_tag)

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
    from ..util.data import DimensionTag
    data = get_concat_sources_data_template(sources)
    data.name = "%s_output" % name
    axes = data.get_axes_from_description(axes)
    padding = cls._transform_padding(padding=padding, axes=axes)
    dim_tags = data.dim_tags
    for i, a in enumerate(axes):
      if sum(padding[i]) == 0:
        continue
      tag = dim_tags[a]
      dim = None if tag.dimension is None else (tag.dimension + sum(padding[i]))
      tag = DimensionTag(kind=tag.kind, description="%s_pad%i" % (name, i), dimension=dim, derived_from_tag=tag)
      dim_tags = dim_tags[:a] + (tag,) + dim_tags[a + 1:]
    return data.copy_template_new_dim_tags(dim_tags, keep_special_axes=True)


class MergeDimsLayer(_ConcatInputLayer):
  """
  Merges a list of axes into a single one. (Flatten the dims.)
  E.g. input is (batch, width, height, dim) and axes=(1,2), then we get (batch, width*height, dim).
  Or input is (batch, time, height, dim) and axes="except_time", then we get (batch, time, height*dim).
  See also :class:`CombineDimsLayer`.
  When batch and time got merged, :class:`SplitBatchTimeLayer` can undo this.
  When you want to merge batch and time, but remove the padding efficiently, i.e. flatten it,
  see :class:`FlattenBatchLayer`.
  """
  layer_class = "merge_dims"

  def __init__(self, axes, keep_order=False, n_out=None, **kwargs):
    """
    :param str|list[str]|list[int] axes: see Data.get_axes_from_description(), e.g. "except_time"
    :param bool keep_order: By default (for historical reasons), the axes are sorted, and then merged.
      Thus, the order of incoming axes will influence the result.
      E.g. inputs [B,S,F] and [B,F,S], with ``axes=["S","F"]``, will get different results,
      although the output shape is [B,S*F] in both cases.
      This is bad: In general, other layers in RETURNN might reorder the axes for various reasons,
      and all layers should behave in the same way, no matter the order.
      It is recommended to set ``keep_order=True``, such that the order defined in ``axes`` defines the behavior,
      and not the incoming axis order.
    :param int|None n_out:
    """
    super(MergeDimsLayer, self).__init__(**kwargs)
    if keep_order:
      assert isinstance(axes, (tuple, list)), "%s: unique axes %r required" % (self, axes)
      axes_ = []
      for axis in axes:
        axis_ = self.input_data.get_axes_from_description(axis, allow_int=False)
        assert len(axis_) <= 1, "%s: unique axes %r required, but got %r -> %r" % (self, axes, axis, axis_)
        axes_ += axis_
      axes = axes_
    else:
      axes = self.input_data.get_axes_from_description(axes)
      axes = sorted(axes)
    self._set_output_sizes(merge_axes=axes)
    merge_target_axis = self._get_target_axis(input_data=self.input_data, merge_axes=axes)
    x = self.input_data.placeholder
    if len(axes) > 1:
      # Transpose so that all axes are behind each other.
      perm = [i for i in range(self.input_data.batch_ndim) if i not in axes]
      # If batch axis included, move to front.
      # This is such that we can deterministically undo this later, e.g. in SplitBatchTimeLayer.
      if self.input_data.batch_dim_axis in axes and not keep_order:
        axes.remove(self.input_data.batch_dim_axis)
        axes.insert(0, self.input_data.batch_dim_axis)
      for i, a in enumerate(axes):
        perm.insert(merge_target_axis + i, a)
      x = tf.transpose(x, perm)
      # Now merge all dims with a reshape.
      from returnn.tf.util.basic import get_shape
      shape = get_shape(x)
      i0 = merge_target_axis
      i1 = i0 + len(axes)
      if all([isinstance(d, int) for d in shape[i0:i1]]):
        import numpy
        res_dim = int(numpy.prod(shape[i0:i1]))
      else:
        res_dim = tf.reduce_prod(shape[i0:i1])
      x = tf.reshape(
        x, shape=shape[:i0] + [res_dim] + shape[i1:])
    if n_out is not None and not self.output.sparse:
      from returnn.tf.util.basic import check_input_dim
      x = check_input_dim(x, axis=self.output.feature_dim_axis, dim=n_out)
    self.output.placeholder = x

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

  def _set_output_sizes(self, merge_axes):
    """
    :param list[int] merge_axes:
    """
    target_axis = self._get_target_axis(input_data=self.input_data, merge_axes=merge_axes)
    target_tag = self.output.dim_tags[target_axis]
    if target_tag.is_batch_dim():
      return  # should be handled already
    if target_tag.dimension is not None:  # static
      return  # should be handled already
    if target_tag.dyn_size_ext:
      return  # handled already

    out_size = None
    for in_axis in merge_axes:
      in_tag = self.input_data.dim_tags[in_axis]
      assert not in_tag.is_batch_dim()
      if in_tag.dimension is not None:
        if in_tag.dimension == 1:
          continue
        in_size = Data.from_tensor(tf.constant(in_tag.dimension, dtype=tf.int32))
      else:
        assert in_tag.dyn_size_ext
        in_size = in_tag.dyn_size_ext
      if not out_size:
        out_size = in_size
      else:
        new_data = Data.get_common_data([out_size, in_size])
        new_data.placeholder = (
          out_size.copy_compatible_to(new_data).placeholder
          * in_size.copy_compatible_to(new_data).placeholder)
        out_size = new_data
    if not out_size:
      out_size = Data.from_tensor(tf.constant(1, dtype=tf.int32))
    target_tag.dyn_size_ext = out_size

  @classmethod
  def get_out_data_from_opts(cls, name, axes, keep_order=False,
                             sources=(), n_out=NotSpecified, out_type=None, **kwargs):
    """
    :param str name:
    :param str|list[str] axes:
    :param bool keep_order:
    :param list[LayerBase] sources:
    :param int|None|NotSpecified n_out:
    :param None|dict[str] out_type:
    :rtype: Data
    """
    from ..util.data import DimensionTag
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
      res_dim = int(numpy.prod([data.batch_shape[i] for i in axes]))
    merge_dim_tags = [tag for (i, tag) in enumerate(data.dim_tags) if i in axes]
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
    if any(tag.is_batch_dim() for tag in merge_dim_tags):
      res_dim_tag_kind = DimensionTag.Types.Batch
    elif any(tag.is_feature_dim() for tag in merge_dim_tags):
      res_dim_tag_kind = DimensionTag.Types.Feature
    else:
      res_dim_tag_kind = DimensionTag.Types.Spatial
    res_dim_tag = DimensionTag(
      kind=res_dim_tag_kind, description="%s_merge_dims" % name,
      dimension=res_dim)
    new_dim_tags = [d for (i, d) in enumerate(data.dim_tags) if i not in axes]
    new_dim_tags.insert(merge_target_axis, res_dim_tag)

    data_opts = data.get_kwargs(include_special_axes=False)
    data_opts["dim_tags"] = new_dim_tags
    data_opts["feature_dim_axis"] = new_feature_dim_axis
    data = Data(**data_opts)

    data.time_dim_axis = cls._old_axis_to_new_axis(
      input_data=input_data, merge_axes=axes, old_axis=input_data.time_dim_axis)
    if data.time_dim_axis is not None and data.time_dim_axis in {data.batch_dim_axis, data.feature_dim_axis}:
      if input_data.time_dim_axis not in {input_data.batch_dim_axis, input_data.feature_dim_axis}:
        # Time got merged with feature or batch.
        data.time_dim_axis = None

    if input_data.batch_dim_axis in axes and data.batch:
      for axis in axes:
        if axis != input_data.batch_dim_axis:
          data.batch = data.batch.copy_extend_with_padded_or_fixed_dim_tag(
            dim_tag=input_data.get_dim_tag(axis),
            batch_major=(axis > input_data.batch_dim_axis) if keep_order else True)
    return data


class SplitLayer(_ConcatInputLayer):
  """
  Splits one axis into multiple parts, via tf.split.
  self.output is simply the input copied.
  Each part can be accessed via the sublayers "/%i".
  """
  layer_class = "split"

  def __init__(self, axis=None, num_splits=None, size_splits=None, **kwargs):
    """
    :param str|None axis: feature axis by default
    :param int|None num_splits:
    :param list[int]|None size_splits:
    """
    assert num_splits or size_splits, "%s: provide either num_splits or size_splits" % self
    super(SplitLayer, self).__init__(**kwargs)
    self.output = self.input_data
    self.axis, self.size_splits = self._get_axis_size_splits_num_splits(
      input_data=self.input_data, axis=axis, num_splits=num_splits, size_splits=size_splits,
      err_prefix=self)
    self.splits = tf.split(self.output.placeholder, self.size_splits, axis=self.axis)
    assert len(self.splits) == len(self.size_splits)
    self._sub_layers = {"%i" % i: self._make_split_layer(i) for i in range(len(self.splits))}

  @classmethod
  def _get_axis_size_splits_num_splits(cls, input_data, axis=None, num_splits=None, size_splits=None, err_prefix=None):
    """
    :param Data input_data:
    :param str|None axis: feature axis by default
    :param int|None num_splits:
    :param list[int]|None size_splits:
    :param object err_prefix:
    :return: axis, size_splits
    :rtype: (int, list[int])
    """
    assert num_splits or size_splits, "%s: provide either num_splits or size_splits" % err_prefix
    if axis is None:
      axis = "feature"
    axis = input_data.get_axis_from_description(axis, allow_int=False)
    dim = input_data.batch_shape[axis]
    assert isinstance(dim, int), "%s: expects static axis %s in %r" % (err_prefix, axis, input_data)
    if num_splits:
      assert dim % num_splits == 0, "%s: expects multiple of %i in dim %i in %r" % (
        err_prefix, num_splits, dim, input_data)
      size_splits = [dim // num_splits for _ in range(num_splits)]
    else:
      if not isinstance(size_splits, (list, tuple)):
        raise TypeError("%s: invalid type num_or_size_splits %r" % (err_prefix, size_splits))
      size_splits = list(size_splits)
      assert sum(size_splits) == dim, "%s: invalid num_or_size_splits %r for dim %i in %r" % (
        err_prefix, size_splits, dim, input_data)
    return axis, size_splits

  def _make_split_layer(self, idx):
    """
    :param int idx:
    :rtype: LayerBase
    """
    out = self._get_split_out_data(
      name=self.name, idx=idx, size_splits=self.size_splits, input_data=self.input_data, axis=self.axis)
    out.placeholder = self.splits[idx]
    out.sanity_check()
    return InternalLayer(name="%s/%i" % (self.name, idx), network=self.network, output=out, sources=self.sources)

  def get_sub_layer(self, layer_name):
    """
    :param str layer_name:
    :rtype: LayerBase|None
    """
    return self._sub_layers.get(layer_name, None)

  @classmethod
  def get_out_data_from_opts(cls, sources, **kwargs):
    """
    :param list[LayerBase] sources:
    :rtype: Data
    """
    return get_concat_sources_data_template(sources)

  @classmethod
  def get_sub_layer_out_data_from_opts(cls, layer_name, parent_layer_kwargs):
    """
    :param str layer_name: name of the sub_layer (right part of '/' separated path)
    :param dict[str] parent_layer_kwargs: kwargs for the parent layer (as kwargs in cls.get_out_data_from_opts())
    :return: Data template, network and the class type of the sub-layer
    :rtype: (Data, TFNetwork, type)|None
    """
    try:
      idx = int(layer_name)
    except ValueError:
      return None
    name = parent_layer_kwargs.get("name", "<unknown>")
    input_data = get_concat_sources_data_template(parent_layer_kwargs["sources"], name="%s_output" % name)
    axis, size_splits = cls._get_axis_size_splits_num_splits(
      input_data=input_data,
      axis=parent_layer_kwargs.get("axis", None),
      num_splits=parent_layer_kwargs.get("num_splits", None),
      size_splits=parent_layer_kwargs.get("size_splits", None),
      err_prefix="%s/%s" % (name, layer_name))
    out = cls._get_split_out_data(
      name=name, idx=idx, input_data=input_data, size_splits=size_splits, axis=axis)
    return out, parent_layer_kwargs["network"], InternalLayer

  @classmethod
  def _get_split_out_data(cls, name, input_data, size_splits, idx, axis):
    """
    :param str name:
    :param Data input_data:
    :param list[int] size_splits:
    :param int idx:
    :param int axis:
    :rtype: Data
    """
    from ..util.data import DimensionTag
    new_dim_tag = DimensionTag(
      kind=input_data.dim_tags[axis].kind, description="%s_split%i" % (name, idx),
      dimension=size_splits[idx])
    out = input_data.copy_template("%s/%i_output" % (name, idx))
    return out.copy_template_replace_dim_tag(axis=axis, new_dim_tag=new_dim_tag)


class SplitDimsLayer(_ConcatInputLayer):
  """
  Splits one axis into multiple axes.
  E.g. if you know that your feature-dim is composed by a window,
  i.e. the input is (batch, time, window * feature),
  you can set axis="F", dims=(window, -1),
  and you will get the output (batch, time, window, feature).

  If the split axis has a dynamic length,
  exactly one of the axes that we split into need to also have a dynamic length.
  You can e.g. use this to split the input dimension into smaller "chunks" of a fixed window size.
  E.g. you could have input (batch, time, feature) and set axis="T", dims=(-1, window),
  to get output (batch, split_time, window, feature).
  In this case, the exact sequence lengths are lost and everything is padded to multiples of the window size using
  the given padding value.
  Use :class:`ReinterpretDataLayer` to receive back the original sequence lengths after merging.

  Also see :class:`SplitBatchTimeLayer`.
  Also see :class:`MergeDimsLayer` which can undo this operation.
  """
  layer_class = "split_dims"

  def __init__(self, axis, dims, pad_to_multiples=None, pad_value=0, **kwargs):
    """
    :param str axis: e.g. "F"
    :param tuple[int]|list[int] dims: what the axis should be split into. e.g. (window, -1)
    :param bool|None pad_to_multiples: If true, input will be padded to the next multiple of the product of the
      static dims, such that splitting is actually possible.
      By default this is done iff the axis has a dynamic size
    :param int|float pad_value: What pad value to use for pad_to_multiples
    """
    super(SplitDimsLayer, self).__init__(**kwargs)
    data = self.input_data
    if isinstance(axis, int):
      data = data.copy_as_batch_major()
    axis = data.get_axis_from_description(axis)
    if pad_to_multiples is None:
      pad_to_multiples = data.is_axis_dynamic(axis)

    from returnn.tf.util.basic import get_shape
    old_shape = get_shape(data.placeholder)
    new_shape = old_shape[:axis] + list(dims) + old_shape[axis + 1:]
    assert len(new_shape) == len(self.output.batch_shape)
    for i in range(len(new_shape)):
      if new_shape[i] == -1 and self.output.batch_shape[i] is not None:
        new_shape[i] = self.output.batch_shape[i]

    import numpy
    new_pos_dims = [d for d in dims if d > 0]
    constant_size = int(numpy.prod(new_pos_dims))
    assert not data.is_axis_dynamic(axis) or pad_to_multiples or constant_size == 1
    if pad_to_multiples and constant_size != 1:
      assert len([d for d in dims if d == -1]) == 1
      old_size = old_shape[axis]
      pad_size = (-old_size) % constant_size

      paddings = [(0, 0)] * axis + [(0, pad_size)] + [(0, 0)] * (data.batch_ndim - axis - 1)
      data.placeholder = tf.pad(data.placeholder, paddings=paddings, constant_values=pad_value)

      axis_wo_batch = data.get_batch_axis_excluding_batch(axis + dims.index(-1))
      if axis_wo_batch in data.size_placeholder:
        # When there is support for this, this should already be created in get_out_data_from_opts
        # Note: currently get_out_data_from_opts already sets data.size_placeholder[axis_wo_batch] to the input size
        # (meaning we do not need to transform the axis here)
        from returnn.tf.util.data import DimensionTag
        dyn_size = -(-data.size_placeholder[axis_wo_batch] // constant_size)  # == ceildiv(size, constant_size)
        if not DimensionTag.get_tag_from_size_tensor(dyn_size):
          tag = DimensionTag(
            description="split-time:%i:%s" % (axis, self.get_absolute_name()),
            kind=DimensionTag.Types.Spatial, batch=self.output.batch)
          tag.set_tag_on_size_tensor(dyn_size)
        self.output.size_placeholder[axis_wo_batch] = dyn_size

    self.output.placeholder = tf.reshape(data.placeholder, shape=new_shape)

  @classmethod
  def _resolve_dims(cls, old_dim, new_dims, pad_to_multiples=False):
    """
    :param int old_dim:
    :param tuple[int]|list[int] new_dims:
    :param bool|None pad_to_multiples:
    :return: new_dims with -1 resolved
    :rtype: tuple[int]
    """
    import numpy
    if all([d > 0 for d in new_dims]):
      new_dims_resolved = new_dims
    else:
      assert all([d != 0 and (d == -1 or d > 0) for d in new_dims])
      assert len([d for d in new_dims if d == -1]) == 1
      new_pos_dims = [d for d in new_dims if d > 0]
      n = int(numpy.prod(new_pos_dims))
      if pad_to_multiples:
        old_dim += (-old_dim) % n
      assert old_dim % n == 0
      rem = old_dim // n
      new_dims_resolved = [(d if (d > 0) else rem) for d in new_dims]
    assert numpy.prod(new_dims_resolved) == old_dim
    return tuple(new_dims_resolved)

  @classmethod
  def _map_old_axis_to_new_axis(cls, split_axis, dims, old_axis, split_offset=None):
    """
    :param int split_axis:
    :param tuple[int] dims: might include -1
    :param int old_axis:
    :param int|None split_offset:
    :rtype: int
    """
    if old_axis < split_axis:
      return old_axis
    if old_axis > split_axis:
      return old_axis + len(dims) - 1
    assert old_axis == split_axis
    if -1 in dims:
      assert dims.count(-1) == 1
      return split_axis + dims.index(-1)
    assert split_offset is not None
    if split_offset < 0:
      split_offset += len(dims)
    assert 0 <= split_offset < len(dims)
    return split_axis + split_offset

  @classmethod
  def get_out_data_from_opts(cls, name, axis, dims, pad_to_multiples=None, sources=(), **kwargs):
    """
    :param str name:
    :param str|int axis:
    :param tuple[int] dims:
    :param bool|None pad_to_multiples:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    from ..util.data import DimensionTag
    input_data = get_concat_sources_data_template(sources)
    data = input_data.copy("%s_output" % name)
    if isinstance(axis, int):
      data = data.copy_as_batch_major()
    axis = data.get_axis_from_description(axis)
    if pad_to_multiples is None:
      pad_to_multiples = data.is_axis_dynamic(axis)

    if data.batch_shape[axis] is not None:
      resolved_shape_dims = cls._resolve_dims(
        old_dim=data.batch_shape[axis], new_dims=dims, pad_to_multiples=pad_to_multiples)
    else:
      resolved_shape_dims = tuple([(d if (d >= 0) else None) for d in dims])

    rem_dim_indices = [i for i, d in enumerate(dims) if d < 0]
    assert len(rem_dim_indices) <= 1, "only one entry in dims %r can be -1" % (dims,)
    rem_dim_idx = rem_dim_indices[0] if rem_dim_indices else (len(dims) - 1)
    axis_dim_tag = data.dim_tags[axis]
    if all(d == 1 for (i, d) in enumerate(dims) if i != rem_dim_idx):
      rem_dim = axis_dim_tag  # we can overtake the existing dim tag
    else:
      rem_dim = DimensionTag(
        kind=axis_dim_tag.kind,
        description="%s_split_dims%i_rem" % (name, rem_dim_idx),
        dimension=resolved_shape_dims[rem_dim_idx])
    resolved_dims = tuple(
      DimensionTag(
        kind=DimensionTag.Types.Spatial,
        description="%s_split_dims%i" % (name, i),
        dimension=resolved_shape_dims[i])
      if i != rem_dim_idx else rem_dim
      for i in range(len(dims)))
    new_dim_tags = data.dim_tags[:axis] + resolved_dims + data.dim_tags[axis + 1:]
    out = data.copy_template_new_dim_tags(new_dim_tags)
    if data.time_dim_axis is None:
      out.time_dim_axis = None
    if data.feature_dim_axis is not None:
      expected_out_feature_dim_axis = cls._map_old_axis_to_new_axis(
        split_axis=axis, dims=dims, old_axis=data.feature_dim_axis, split_offset=-1)
      if out.feature_dim_axis != expected_out_feature_dim_axis:  # maybe due to non-specified default behavior
        out.feature_dim_axis = expected_out_feature_dim_axis
        out.dim = out.batch_shape[out.feature_dim_axis]
    return out


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
    self.base = base
    assert base.output.time_dim_axis is not None
    base_shape = tf.shape(base.output.placeholder)
    batch_dim = base_shape[base.output.batch_dim_axis]
    time_dim = base_shape[base.output.time_dim_axis]
    input_data = self.input_data.copy_as_batch_major()
    from returnn.tf.util.basic import get_shape
    input_shape = get_shape(input_data.placeholder)
    self.output.placeholder = tf.reshape(input_data.placeholder, shape=[batch_dim, time_dim] + input_shape[1:])

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    return super(SplitBatchTimeLayer, self).get_dep_layers() + [self.base]

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
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
    data = get_concat_sources_data_template(sources).copy_as_batch_major()
    data.batch = base.output.batch
    data = data.copy_add_dim_by_tag(base.output.get_time_dim_tag(), axis=1, unbroadcast=True)
    return data.copy("%s_output" % name)


class FlattenBatchLayer(_ConcatInputLayer):
  """
  Merges one axis into the batch axis.
  If the axis has dynamic lengths, this would use flattening,
  i.e. recalculate the padding, i.e. the size changes.
  This basically wraps :func:`flatten_with_seq_len_mask` or :func:`flatten_with_seq_len_mask_time_major`.
  See also :class:`MergeDimsLayer`, which does not do flattening,
  i.e. the size stays the same.
  """
  layer_class = "flatten_batch"

  def __init__(self, axis="T", batch_major=True, **kwargs):
    """
    :param str axis:
    :param bool batch_major: if False, will flatten in time-major manner
    """
    super(FlattenBatchLayer, self).__init__(**kwargs)
    assert self.input_data.have_batch_axis()
    x = self.input_data
    axis = x.get_axis_from_description(axis, allow_int=False)
    assert axis != x.batch_dim_axis
    if x.is_axis_dynamic(axis):
      if batch_major:
        self.output.placeholder = tf_util.flatten_with_seq_len_mask(
          x.placeholder, seq_lens=x.get_dynamic_size(axis), batch_dim_axis=x.batch_dim_axis, time_dim_axis=axis)
      else:
        self.output.placeholder = tf_util.flatten_with_seq_len_mask_time_major(
          x.placeholder, seq_lens=x.get_dynamic_size(axis), batch_dim_axis=x.batch_dim_axis, time_dim_axis=axis)
    else:
      if batch_major:
        x = x.copy_as_batch_major()  # (B,...)
        x = x.copy_move_axis(axis, 1)  # (B,T,...)
      else:
        x = x.copy_move_axis(axis, 0)  # (T,...)
        x = x.copy_move_axis(x.batch_dim_axis, 1)  # (T,B,...)
      out_shape = tf_util.get_shape(x.placeholder)
      out_shape = [out_shape[0] * out_shape[1]] + out_shape[2:]
      self.output.placeholder = tf.reshape(x.placeholder, out_shape)

  @classmethod
  def get_out_data_from_opts(cls, sources, name, axis="T", batch_major=True, **kwargs):
    """
    :param list[LayerBase] sources:
    :param str name:
    :param str axis:
    :param bool batch_major: if False, will flatten in time-major manner
    :rtype: Data
    """
    out = get_concat_sources_data_template(sources, name="%s_output" % name)
    out = out.copy_as_batch_major()
    axis = out.get_axis_from_description(axis, allow_int=False)
    assert axis != out.batch_dim_axis
    dim_tag = out.get_dim_tag(axis)
    out = out.copy_template_excluding_axis(axis)
    if out.batch:
      out.batch = out.batch.copy_extend_with_packed_dim_tag(dim_tag, batch_major=batch_major)
    return out


class UnflattenNdLayer(_ConcatInputLayer):
  """
  This keeps the batch axis as-is, i.e. the flattening/unflattening did not happen on the batch axis.

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
    :param dict[int,LayerBase]|None declare_same_sizes_as:
    """
    super(UnflattenNdLayer, self).__init__(**kwargs)
    self.sizes = sizes
    self.declare_same_sizes_as = declare_same_sizes_as
    input_data = self.input_data.copy_as_batch_major()
    sizes_data = sizes.output.copy_as_batch_major()
    assert sizes_data.batch_ndim == 2
    assert sizes_data.batch_shape[1] in (None, num_axes)  # also allow None...
    self.output.placeholder = tf_util.unflatten_nd(input_data.placeholder, sizes_data.placeholder, num_axes=num_axes)
    size_placeholder = {i: sizes_data.placeholder[:, i] for i in range(num_axes)}
    if declare_same_sizes_as:
      for i, other in declare_same_sizes_as.items():
        assert 0 <= i < num_axes
        other_dim_tag = other.output.get_size_dim_tag(0)
        other_dim_tag.set_tag_on_size_tensor(size_placeholder[i], batch=self.output.batch, same_as_before=True)
    self.output.size_placeholder = size_placeholder

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    deps = super(UnflattenNdLayer, self).get_dep_layers()
    if self.sizes:
      deps.append(self.sizes)
    if self.declare_same_sizes_as:
      for i, other in sorted(self.declare_same_sizes_as.items()):
        deps.append(other)
    return deps

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
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
    from returnn.tf.util.basic import expand_dims_unbroadcast
    self.output.placeholder = expand_dims_unbroadcast(data.placeholder, axis=axis, dim=dim)

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
    from ..util.data import DimensionTag
    init_axis = axis
    data = get_concat_sources_data_template(sources)
    if isinstance(axis, int):
      data = data.copy_as_batch_major()
    axis = cls._get_axis(data=data, axis=axis)

    new_dim = DimensionTag(
      kind=DimensionTag.Types.Spatial, description="%s_expand_dims" % name,
      dimension=dim)
    data = data.copy_template(name="%s_output" % name)
    data = data.copy_add_dim_by_tag(new_dim, unbroadcast=True, axis=axis)
    if isinstance(init_axis, str):
      if init_axis.lower() in ["spatial", "time", "t"] and data.time_dim_axis is None:
        data.time_dim_axis = axis
    return data


class RepeatLayer(_ConcatInputLayer):
  """
  A wrapper around tf.repeat, but supports an additional batch axis for the durations
  The sum of the repetitions has to be non-zero for each sequence in the batch.

  This layer can only be used with Tensorflow 1.15.0 or newer.
  """
  layer_class = "repeat"

  def __init__(self, repetitions, axis="T", **kwargs):
    """
    :param LayerBase|int repetitions:
      number of repetitions for each sequence and position in target axis.
      Can be [B,T] or [T,B] or some subset of that shape
    :param str axis: (dynamic) axis for repetition (currently only time axis is supported)
    """
    super(RepeatLayer, self).__init__(**kwargs)
    self.repetitions = repetitions
    if isinstance(self.repetitions, int):
      repetitions_data = Data.from_tensor(tf.constant(self.repetitions))
    else:
      assert isinstance(self.repetitions, LayerBase)
      repetitions_data = self.repetitions.output
    input_axis = self.input_data.get_axis_from_description(axis, allow_int=False)
    assert repetitions_data.dtype in ["int32", "int64"]
    if repetitions_data.ndim == 0:  # no T dim: add a (un)broadcasted one.
      axis_dim_tag = self.input_data.get_dim_tag(input_axis)
      repetitions_data = repetitions_data.copy_add_dim_by_tag(axis_dim_tag, unbroadcast=True)
    repetitions_axis = repetitions_data.get_axis_from_description(axis, allow_int=False)
    assert repetitions_data.ndim == 1, "Repetitions %r must only have at most one non-batch axis" % repetitions
    assert repetitions_data.batch_shape[repetitions_axis] == self.input_data.batch_shape[input_axis], (
      "Axis mismatch between input (%i) and repetitions (%i)" % (
        self.input_data.batch_shape[input_axis], repetitions_data.batch_shape[repetitions_axis]))

    assert self.output.have_batch_axis() == (self.input_data.have_batch_axis() or repetitions_data.have_batch_axis())

    def copy_placeholder_with_batch_axis(data, other_batch):
      """
      Adds a batch dim by unbroadcasting if our output also has a batch dim,
      otherwise just expands the dims.

      :param Data data:
      :param BatchInfo|None other_batch: use this batch info if we should add some
      :rtype: tf.Tensor
      :return: the placeholder, with shape [B, T, ...], maybe with added batch dim.
      """
      if self.output.have_batch_axis():
        if data.have_batch_axis():
          data = data.copy_as_batch_major()
        else:
          data = data.copy_add_batch_dim(batch_dim_axis=0, batch=other_batch)  # will unbroadcast
        original_axis = data.get_axis_from_description(axis, allow_int=False)
        data = data.copy_move_axis(original_axis, 1)
        return data.placeholder  # [B, T, ...]
      else:
        assert not self.output.have_batch_axis()
        original_axis = data.get_axis_from_description(axis, allow_int=False)
        data = data.copy_move_axis(original_axis, 0)
        return tf.expand_dims(data.placeholder, 0)  # [B=1, T, ...]

    input_placeholder = copy_placeholder_with_batch_axis(
      self.input_data, other_batch=repetitions_data.batch)  # [B, T, ... ]
    repetitions_placeholder = copy_placeholder_with_batch_axis(
      repetitions_data, other_batch=self.input_data.batch)  # [B, T]

    # pad the target axis
    paddings = [(0, 1) if i == 1 else (0, 0) for i in range(self.input_data.ndim + 1)]
    padded_data = tf.pad(input_placeholder, paddings)  # [B, T+1, ...]
    # those are the sequence lengths after expansion
    target_seq_len = tf.reduce_sum(repetitions_placeholder, axis=1)  # [B], the new size_placeholder for 'T'
    # maximum sequence length after expansion
    max_duration = tf.reduce_max(target_seq_len)  # [] == T'
    # the new padding is the difference of the maximum to the new target
    target_padding_steps = tf.expand_dims(max_duration - target_seq_len, 1)  # [B, 1]
    # add the repetitions for the artificial padding position
    target_repetitions = tf.concat([repetitions_placeholder, target_padding_steps], axis=1)  # [B, T+1]
    # flatten batch and time
    shape = tf_util.get_shape(padded_data)
    flat_shape = [shape[0] * shape[1]] + shape[2:]
    reshaped_data = tf.reshape(padded_data, flat_shape)  # [B * T+1, F]
    reshaped_repetitions = tf.reshape(target_repetitions, (shape[0] * shape[1],))  # [B * T+1]
    # run the repetition
    repeated_data = tf.repeat(reshaped_data, reshaped_repetitions, axis=0)  # [B * T', ...]
    # unflatten the output
    target_shape = ([shape[0]] if self.output.have_batch_axis() else []) + [max_duration] + shape[2:]
    res = tf.reshape(repeated_data, target_shape)
    res.set_shape(self.output.batch_shape)
    self.output.placeholder = res  # [B, T', ...] or [T', ...]
    # set size placeholders
    output_axis = self.output.get_axis_from_description(axis)
    tag = self.output.dim_tags[output_axis]
    if tag.dimension is None:  # dynamic? dyn sizes needed?
      tag.set_tag_on_size_tensor(target_seq_len, batch=self.output.batch)

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    deps = super(RepeatLayer, self).get_dep_layers()
    if isinstance(self.repetitions, LayerBase):
      deps.append(self.repetitions)
    return deps

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param get_layer:
    """
    super(RepeatLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    if isinstance(d["repetitions"], str):
      d["repetitions"] = get_layer(d["repetitions"])

  @classmethod
  def get_out_data_from_opts(cls, name, axis, repetitions, sources=(), **kwargs):
    """
    :param str name:
    :param str axis:
    :param LayerBase|int repetitions:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    from ..util.data import DimensionTag
    data = get_concat_sources_data_template(sources, name="%s_output" % name)
    if data.have_batch_axis():
      data = data.copy_as_batch_major()
    elif isinstance(repetitions, LayerBase) and repetitions.output.have_batch_axis():
      data = data.copy_add_batch_dim(batch_dim_axis=0, batch=repetitions.output.batch)
    original_axis = data.get_axis_from_description(axis, allow_int=False)
    tag = data.dim_tags[original_axis]
    if tag.dimension is not None and isinstance(repetitions, int):
      new_dim = tag.dimension * repetitions
    else:
      new_dim = None
    data = data.copy_move_axis(original_axis, data.get_batch_axis(0))
    tag = DimensionTag(description="repeated:%s" % name, kind=tag.kind, dimension=new_dim)
    return data.copy_template_replace_dim_tag(axis=data.get_batch_axis(0), new_dim_tag=tag)


class TileLayer(_ConcatInputLayer):
  """
  A wrapper around tf.tile
  """
  layer_class = "tile"

  def __init__(self, multiples, **kwargs):
    """
    :param dict[str, int] multiples: number of multiples per axis (axis provided as str)
    """
    super(TileLayer, self).__init__(**kwargs)
    self.multiples = multiples
    input_data = self.input_data

    multiples_full = [1] * input_data.batch_ndim
    for axis, multiple in multiples.items():
      a = input_data.get_axis_from_description(axis, allow_int=False)
      if multiple != 1:
        assert a not in input_data.get_dynamic_axes(), "Tiling of dynamic axes not yet implemented"
        assert a != input_data.batch_dim_axis, "Tiling of batch axis not yet implemented"
      multiples_full[a] *= multiple

    self.output.placeholder = tf.tile(input_data.placeholder, multiples_full)

  @classmethod
  def get_out_data_from_opts(cls, name, multiples, sources=(), **kwargs):
    """
    :param str name:
    :param dict[str, int] multiples:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    from ..util.data import DimensionTag
    data = get_concat_sources_data_template(sources, name="%s_output" % name)
    dim_tags = list(data.dim_tags)
    for axis, multiple in multiples.items():
      axis = data.get_axis_from_description(axis, allow_int=False)
      tag = dim_tags[axis]
      dim = None if tag.dimension is None else (tag.dimension * multiple)
      tag = DimensionTag(kind=tag.kind, description="%s_tile" % name, dimension=dim)
      dim_tags[axis] = tag
    return data.copy_template_new_dim_tags(dim_tags, keep_special_axes=True)


class CastLayer(CopyLayer):
  """
  Cast to some other dtype.
  """
  layer_class = "cast"

  def __init__(self, dtype, output, **kwargs):
    """
    :param str dtype:
    :param Data output:
    """
    super(CastLayer, self).__init__(output=output, **kwargs)
    if self.output.dtype != dtype:
      self.output = output
      self.output.placeholder = tf.cast(self.input_data.placeholder, dtype)

  @classmethod
  def get_out_data_from_opts(cls, dtype, **kwargs):
    """
    :param str dtype:
    :rtype: Data
    """
    out = super(CastLayer, cls).get_out_data_from_opts(**kwargs).copy_template()
    out.dtype = dtype
    if out.sparse and (out.dtype.startswith("float") or out.dtype.startswith("complex")):
      out.sparse = False
      if out.feature_dim_axis is not None:
        out.dim = out.batch_shape[out.feature_dim_axis]
      else:
        out.dim = None
    return out


class SwapAxesLayer(_ConcatInputLayer):
  """
  Swaps two axes. Basically a wrapper around :func:`TFUtil.swapaxes`.
  Note that usually, this should not be needed, and it is recommended not to be used,
  as this will be unnecessarily inefficient.
  Normally, all RETURNN layers will automatically transpose the input data into whatever format they need.

  All axes always have a special meaning (e.g. feature dim or time dim)
  or dimension tag (e.g. for time axes, including dyn seq lengths).
  If you need to change the meaning (and not actually transpose / swap axes),
  you need to use :class:`ReinterpretDataLayer`.

  See also :class:`TransposeLayer` for a more generic variant.

  See also :class:`ReinterpretDataLayer`, which does not swap/transpose axes,
  but allows to reinterpret their meaning / dim tags.
  """
  layer_class = "swap_axes"

  def __init__(self, axis1, axis2, **kwargs):
    """
    :param int|str axis1:
    :param int|str axis2:
    """
    super(SwapAxesLayer, self).__init__(**kwargs)
    from returnn.tf.util.basic import swapaxes
    axis1 = self.input_data.get_axis_from_description(axis1)
    axis2 = self.input_data.get_axis_from_description(axis2)
    self.output.placeholder = swapaxes(self.input_data.placeholder, axis1=axis1, axis2=axis2)

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
    dim_tags = list(out.dim_tags)
    dim_tags[axis1], dim_tags[axis2] = dim_tags[axis2], dim_tags[axis1]
    opts = out.get_kwargs(include_special_axes=False)
    opts["dim_tags"] = dim_tags
    opts["time_dim_axis"] = cls._translate_axis(out.time_dim_axis, axis1, axis2)
    if out.feature_dim_axis_or_unspecified is not NotSpecified:
      opts["feature_dim_axis"] = cls._translate_axis(out.feature_dim_axis, axis1, axis2)
    return Data(**opts)


class TransposeLayer(_ConcatInputLayer):
  """
  Basically a wrapper around :func:`tf.transpose`.
  Note that usually, this should not be needed, and it is recommended not to be used,
  as this will be unnecessarily inefficient.
  Normally, all RETURNN layers will automatically transpose the input data into whatever format they need.

  All axes always have a special meaning (e.g. feature dim or time dim)
  or dimension tag (e.g. for time axes, including dyn seq lengths).
  If you need to change the meaning (and not actually transpose / swap axes),
  you need to use :class:`ReinterpretDataLayer`.

  See also :class:`ReinterpretDataLayer`, which does not transpose axes,
  but allows to reinterpret their meaning / dim tags.
  """
  layer_class = "transpose"

  def __init__(self, perm, **kwargs):
    """
    :param dict[str,str] perm: target axis -> source axis
    """
    super(TransposeLayer, self).__init__(**kwargs)
    self.perm = perm
    perm_ = self.get_perm_int(input_data=self.input_data, perm=perm)
    self.output.placeholder = tf.transpose(
      self.input_data.placeholder, [perm_[i] for i in range(self.input_data.batch_ndim)])

  @classmethod
  def transpose(cls, input_data, perm, name=None):
    """
    :param Data input_data:
    :param dict[str,str] perm:
    :param str|str name:
    :return: transposed data
    :rtype: Data
    """
    perm_ = cls.get_perm_int(input_data=input_data, perm=perm)
    perm__ = [perm_[i] for i in range(input_data.batch_ndim)]
    return input_data.copy_transpose(perm__).copy(name=name)

  @classmethod
  def get_perm_int(cls, input_data, perm):
    """
    :param Data input_data:
    :param dict[str,str] perm:
    :rtype: dict[int,int]
    """
    def _axis(a):
      """
      :param str a:
      :rtype: int
      """
      return input_data.get_axis_from_description(a, allow_int=False)
    perm_ = {_axis(i): _axis(j) for (i, j) in perm.items()}
    assert len(perm) == len(perm_) == len(set(perm_.values())), "data %s, perm %r invalid" % (input_data, perm)
    target_axes = set(perm_)
    source_axes = set(perm_.values())
    rem_target_axes = [i for i in range(input_data.batch_ndim) if i not in target_axes]
    rem_source_axes = [i for i in range(input_data.batch_ndim) if i not in source_axes]
    assert len(rem_target_axes) == len(rem_source_axes)
    perm_.update({i: j for (i, j) in zip(rem_target_axes, rem_source_axes)})
    assert len(perm_) == len(set(perm_.values())) == input_data.batch_ndim
    return perm_

  @classmethod
  def get_out_data_from_opts(cls, name, sources, perm, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param dict[str,str] perm: target axis -> source axis
    :rtype: Data
    """
    input_data = get_concat_sources_data_template(sources)
    return cls.transpose(input_data=input_data.copy_template(), perm=perm, name="%s_output" % name)


class ReinterpretDataLayer(_ConcatInputLayer):
  """
  Acts like the :class:`CopyLayer` but reinterprets the role of some axes or data.
  """
  layer_class = "reinterpret_data"

  # noinspection PyUnusedLocal
  def __init__(self, switch_axes=None, size_base=None, set_axes=None,
               set_dim_tags=None,
               enforce_batch_major=False, enforce_time_major=False,
               set_sparse=None, set_sparse_dim=NotSpecified, increase_sparse_dim=None,
               **kwargs):
    """
    :param str|list[str] switch_axes: e.g. "bt" to switch batch and time axes
    :param LayerBase|None size_base: copy the size_placeholder from the given layer
    :param dict[str,int|str] set_axes:
      This can be used to overwrite the special axes like time_dim_axis or feature_dim_axis.
      For that, use keys "B","T" or "F", and a value via :func:`Data.get_axis_from_description`.
    :param dict[str|DimensionTag,DimensionTag] set_dim_tags: axis -> new dim tag. assigns new dim tags.
      If the dim tag is yet undefined, this will not use same_dim_tags_as (declare_same_as)
      but create a new dim tag.
      This option is useful for generalized self attention (https://github.com/rwth-i6/returnn/issues/391).
    :param bool enforce_batch_major:
    :param bool enforce_time_major:
    :param bool|None set_sparse: if bool, set sparse value to this
    :param int|None|NotSpecified set_sparse_dim: set sparse dim to this. assumes that it is sparse
    :param int|None increase_sparse_dim: add this to the dim. assumes that it is sparse
    """
    from returnn.tf.util.basic import get_valid_scope_name_from_str
    super(ReinterpretDataLayer, self).__init__(**kwargs)
    self.size_base = size_base
    input_data = self.input_data
    if enforce_batch_major:
      input_data = input_data.copy_as_batch_major()
    if enforce_time_major:
      input_data = input_data.copy_as_time_major()
    if set_dim_tags:
      for axis, new_tag in set_dim_tags.items():
        axis_int = input_data.get_axis_from_description(axis, allow_int=False)
        old_tag = input_data.dim_tags[axis_int]
        assert new_tag.dimension == old_tag.dimension
        new_tag = new_tag.get_for_batch_ctx(input_data.batch, input_data.control_flow_ctx)
        if new_tag.dimension is None and not new_tag.dyn_size_ext:  # still undefined
          assert old_tag.dyn_size_ext
          new_dyn_size_ext = old_tag.dyn_size_ext.copy(name="%s_size" % (new_tag.description or "<unnamed>"))
          # Need to create new size tensor as long as we have get_tag_from_size_tensor.
          new_dyn_size_ext.placeholder = tf.identity(
            new_dyn_size_ext.placeholder, name=get_valid_scope_name_from_str(new_dyn_size_ext.name))
          new_tag.set_tag_on_size_tensor(new_dyn_size_ext.placeholder)
          new_tag.dyn_size_ext = new_dyn_size_ext
    self.output.placeholder = input_data.placeholder
    if len(self.sources) == 1:
      self.output_loss = self.sources[0].output_loss
      if not self.dropout:
        self.output_before_activation = self.sources[0].output_before_activation
    for src in self.sources:
      if src.allow_inf_in_output:
        self.allow_inf_in_output = True

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    deps = super(ReinterpretDataLayer, self).get_dep_layers()
    if self.size_base:
      deps.append(self.size_base)
    return deps

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param get_layer:
    """
    super(ReinterpretDataLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    if d.get("size_base"):
      d["size_base"] = get_layer(d["size_base"])

  @classmethod
  def get_out_data_from_opts(cls, name, sources,
                             switch_axes=None, size_base=None, set_axes=None,
                             set_dim_tags=None,
                             enforce_batch_major=False, enforce_time_major=False,
                             set_sparse=None, set_sparse_dim=NotSpecified, increase_sparse_dim=None,
                             **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param str|list[str] switch_axes: e.g. "bt" to switch batch and time axes
    :param LayerBase|None size_base: similar as size_target
    :param dict[str,int] set_axes:
    :param dict[str|DimensionTag,DimensionTag] set_dim_tags:
    :param bool enforce_batch_major:
    :param bool enforce_time_major:
    :param bool|None set_sparse: if bool, set sparse value to this
    :param int|None|NotSpecified set_sparse_dim: set sparse dim to this. assumes that it is sparse
    :param int|None increase_sparse_dim: add this to the dim. assumes that it is sparse
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
      if s.upper() == "F":
        return "feature_dim_axis"
      assert s in ["batch_dim_axis", "time_dim_axis", "feature_dim_axis"]
      return s

    if switch_axes:
      assert len(switch_axes) == 2
      axes_s = list(map(map_axis_name, switch_axes))
      axes = [getattr(out, s) for s in axes_s]
      for i in range(len(axes)):
        setattr(out, axes_s[i], axes[(i + 1) % len(axes)])
    if set_axes:
      for s, i in sorted(set_axes.items()):
        s = map_axis_name(s)
        if isinstance(i, int):
          assert enforce_batch_major or enforce_time_major, "%r: explicit set_axes %r" % (name, set_axes)
        i = out.get_axis_from_description(i)
        setattr(out, s, i)
        if s == "feature_dim_axis":
          out.dim = out.batch_shape[out.feature_dim_axis]
    if out.size_placeholder and size_base:  # size_placeholder might be None, e.g. via DataNotAvailableLayer
      assert size_base.output.size_placeholder
      assert len(out.size_placeholder) == len(size_base.output.size_placeholder)
      # Keep same indices. Assumes same order of spatial dims.
      out.size_placeholder = {
        i: size_base.output.size_placeholder[j]
        for (i, j) in zip(sorted(out.size_placeholder), sorted(size_base.output.size_placeholder))}
    if set_dim_tags:
      for axis, tag in set_dim_tags.items():
        axis_int = out.get_axis_from_description(axis)
        out = out.copy_template_replace_dim_tag(axis=axis_int, new_dim_tag=tag)
    if set_sparse is not None:
      assert isinstance(set_sparse, bool)
      out.sparse = set_sparse
      if not set_sparse:
        assert set_sparse_dim is NotSpecified
        if out.feature_dim_axis is None:
          out.dim = None
        else:
          out.dim = out.batch_shape[out.feature_dim_axis]
    old_dim = out.dim
    if set_sparse_dim is not NotSpecified:
      assert set_sparse_dim is None or isinstance(set_sparse_dim, int)
      out.dim = set_sparse_dim
    if increase_sparse_dim:
      assert out.sparse
      out.dim += increase_sparse_dim
    if old_dim != out.dim:
      out.vocab = None
    return out


class ConvLayer(_ConcatInputLayer):
  """
  A generic convolution layer which supports 1D, 2D and 3D convolution.
  Pooling can be done in the separate "pool" layer.
  """

  layer_class = "conv"
  recurrent = True  # we must not allow any shuffling in the time-dim or so

  # noinspection PyUnusedLocal,PyShadowingBuiltins
  def __init__(self, n_out, filter_size, padding, strides=1, dilation_rate=1,
               groups=1,
               input_expand_dims=0, input_add_feature_dim=False, input_split_feature_dim=None,
               auto_use_channel_first=False,
               with_bias=NotSpecified,
               activation=None,
               forward_weights_init="glorot_uniform", bias_init=0.0,
               filter=None, filter_perm=None, bias=None,
               **kwargs):
    """
    :param int n_out: number of outgoing features
    :param tuple[int] filter_size: (width,), (height,width) or (depth,height,width) for 1D/2D/3D conv.
      the input data ndim must match, or you can add dimensions via input_expand_dims or input_add_feature_dim.
      it will automatically swap the batch-dim to the first axis of the input data.
    :param str padding: "same" or "valid"
    :param int|tuple[int] strides: strides for the spatial dims,
      i.e. length of this tuple should be the same as filter_size, or a single int.
    :param int|tuple[int] dilation_rate: dilation for the spatial dims
    :param int groups: grouped convolution
    :param int input_expand_dims: number of dynamic dims to add to the input
    :param bool input_add_feature_dim: will add a dim at the end and use input-feature-dim == 1,
      and use the original input feature-dim as a spatial dim.
    :param None|int input_split_feature_dim: if set, like input_add_feature_dim it will add a new feature dim
      which is of value input_split_feature_dim, and the original input feature dim
      will be divided by input_split_feature_dim, thus it must be a multiple of that value.
    :param bool auto_use_channel_first: convert the input to NCHW or not
    :param bool|NotSpecified with_bias: if True, will add a bias to the output features. False by default
    :param None|str activation: if set, will apply this function at the end
    :param LayerBase|None filter: if given, will not create an own parameter, but use this as the filter
    :param dict[str,str]|None filter_perm: transposes the filter (input filter as layer)
    :param LayerBase|None bias: if given, will not create an own parameter, but use this as the bias
    """
    from returnn.tf.util.data import DimensionTag
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
    input_data = self._transform_input(
      self.input_data,
      input_expand_dims=input_expand_dims,
      input_split_feature_dim=input_split_feature_dim,
      input_add_feature_dim=input_add_feature_dim)
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
    n_in = input_data.dim
    if groups != 1:
      assert groups >= 1 and n_in % groups == 0 and n_out % groups == 0
    filter_shape = list(filter_size) + [n_in // groups, n_out]
    from returnn.tf.util.basic import get_initializer
    self.filter_layer = None
    if filter:
      self.filter_layer = filter
      filter_data = filter.output
      if filter_perm:
        filter_data = TransposeLayer.transpose(filter_data, perm=filter_perm, name="filter_transposed")
      assert filter_data.batch_shape == tuple(filter_shape)
      filters = filter_data.placeholder
    else:
      assert not filter_perm
      with self.var_creation_scope():
        fwd_weights_initializer = get_initializer(
          forward_weights_init, seed=self.network.random.randint(2 ** 31), eval_local_ns={"layer": self})
        filters = self.add_param(tf_compat.v1.get_variable(
          name="W", shape=filter_shape, initializer=fwd_weights_initializer))
    data_format = None
    if input_data.is_batch_feature_major:
      assert self.output.is_batch_feature_major
      data_format = {1: "NCW", 2: "NCHW", 3: "NCDHW"}[len(filter_size)]
    if groups > 1 and groups == n_in and len(filter_size) <= 2:  # depthwise conv
      x = input_data.placeholder
      if len(filter_size) == 1:
        filters = tf.reshape(filters, [filter_size[0], 1, n_in, n_out // n_in])  # [1,K,n_in,n_out//n_in]
        x = tf.expand_dims(x, axis=-1 if self.output.is_batch_feature_major else -2)  # [B,T,1,n_in]
        strides = strides + [1]
        dilation_rate = dilation_rate + [1]
      else:
        filters = tf.reshape(filters, list(filter_size) + [n_in, n_out // n_in])  # K+[n_in,n_out//n_in]
      y = tf.nn.depthwise_conv2d(
        x, data_format=data_format,
        filter=filters,
        padding=padding,
        strides=([1] + strides + [1]) if self.output.is_batch_feature_major else ([1, 1] + strides),
        dilations=dilation_rate)
      if len(filter_size) == 1:
        y = tf.squeeze(y, axis=-1 if self.output.is_batch_feature_major else -2)
        strides = strides[:-1]
        dilation_rate = dilation_rate[:-1]
    else:
      y = tf_compat.v1.nn.convolution(
        input_data.placeholder, data_format=data_format,
        filter=filters,
        padding=padding, strides=strides, dilation_rate=dilation_rate)
    # y shape is [batch] + dynamic_dims + [n_out].
    if with_bias is NotSpecified:
      with_bias = True if bias else False
    if bias:
      assert with_bias
    self.bias_layer = None
    if with_bias:
      if bias:
        self.bias_layer = bias
        b_ = bias.output.copy_compatible_to(self.output)
        y += b_.placeholder
      else:
        with self.var_creation_scope():
          bias_initializer = get_initializer(
            bias_init, seed=self.network.random.randint(2 ** 31) if bias_init else 0, eval_local_ns={"layer": self})
          b = self.add_param(tf_compat.v1.get_variable(name="bias", shape=(n_out,), initializer=bias_initializer))
        if input_data.is_batch_feature_major:
          y += tf.reshape(b, [1, n_out] + [1] * len(filter_size))
        else:
          y += b
    if activation:
      from returnn.tf.util.basic import get_activation_function
      act_func = get_activation_function(activation)
      self.output_before_activation = OutputWithActivation(y, act_func=act_func)
    else:
      self.output_before_activation = OutputWithActivation(y)
    y = self.output_before_activation.y
    self.output.placeholder = y
    index_shift = self.output.get_spatial_batch_axes()[0]
    for i, in_axis in enumerate(input_data.get_spatial_batch_axes()):
      in_tag = input_data.dim_tags[in_axis]
      if in_tag.dimension is None and in_tag.dyn_size is not None:
        size = in_tag.dyn_size
        with tf_util.same_control_flow_ctx(size):
          size = self.calc_out_dim(
            in_dim=size,
            filter_size=filter_size[i], stride=strides[i],
            dilation_rate=dilation_rate[i], padding=padding)
        out_tag = self.output.dim_tags[i + index_shift]
        size_tag = DimensionTag.get_tag_from_size_tensor(size)
        if not size_tag:
          out_tag.set_tag_on_size_tensor(size, batch=in_tag.batch)
        else:
          out_tag.declare_same_as(size_tag)

  @classmethod
  def _transform_input(cls, input_data, input_expand_dims, input_split_feature_dim, input_add_feature_dim):
    """
    :param Data input_data:
    :param int input_expand_dims: number of dynamic dims to add to the input
    :param bool input_add_feature_dim: will add a dim at the end and use input-feature-dim == 1,
      and use the original input feature-dim as a spatial dim.
    :param None|int input_split_feature_dim: if set, like input_add_feature_dim it will add a new feature dim
      which is of value input_split_feature_dim, and the original input feature dim
      will be divided by input_split_feature_dim, thus it must be a multiple of that value.
    :rtype: Data
    """
    input_data = input_data.copy_as_batch_major()
    if input_expand_dims:
      for i in range(input_expand_dims):
        input_data = input_data.copy_add_spatial_dim()
    if input_split_feature_dim:
      # Split the feature dimension.
      input_data = input_data.copy_split_feature_dim(input_split_feature_dim)
    if input_add_feature_dim:
      # Add a feature dimension; any other static dims will be used as dynamic dims below.
      input_data = input_data.copy_add_feature_dim()
    return input_data

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
      if isinstance(b, int) and b == 1:
        return a
      return -(-a // b)

    padding = padding.upper()
    # See tf.compat.v1.nn.convolution() documentation for more.
    if padding == "SAME":
      return ceildiv(in_dim, stride)
    elif padding == "VALID":
      return tf_util.simplify_nonzero_seq_length(
        ceildiv(
          (tf_util.simplify_sub(in_dim, (filter_size - 1) * dilation_rate)),
          stride))
    else:
      raise Exception("invalid padding %r" % padding)

  @classmethod
  def get_out_data_from_opts(
        cls, name, n_out, filter_size, padding, strides=1, dilation_rate=1, sources=(),
        input_expand_dims=0, input_add_feature_dim=False, input_split_feature_dim=None,
        auto_use_channel_first=False,
        **kwargs):
    """
    :param str name:
    :param int n_out:
    :param tuple[int] filter_size:
    :param str padding:
    :param int|tuple[int] strides:
    :param int|tuple[int] dilation_rate:
    :param list[LayerBase]|tuple[LayerBase] sources:
    :param int input_expand_dims: number of dynamic dims to add to the input
    :param bool input_add_feature_dim:
    :param None|int input_split_feature_dim:
    :param bool auto_use_channel_first:
    """
    input_data = get_concat_sources_data_template(sources)
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
    input_data = cls._transform_input(
      input_data,
      input_expand_dims=input_expand_dims,
      input_split_feature_dim=input_split_feature_dim,
      input_add_feature_dim=input_add_feature_dim)
    data = input_data.copy_as_batch_spatial_major()  # just to have the dim tags in order [B,S...,D]
    # Be relaxed about incorrect input data. Throw errors later. This can also work during template construction.
    old_spatial_dim_tags = data.dim_tags[1:-1]
    dim_tags = [data.dim_tags[0]]  # [B]
    for i in range(len(filter_size)):
      old_tag = old_spatial_dim_tags[i] if i < len(old_spatial_dim_tags) else None
      if old_tag and (filter_size[i] == strides[i] == 1 or (strides[i] == 1 and padding == "SAME")):
        dim_tags.append(old_tag)  # identity in this axis
        continue
      new_dim = None
      if old_tag and old_tag.dimension is not None:
        new_dim = ConvLayer.calc_out_dim(
          in_dim=old_tag.dimension,
          filter_size=filter_size[i], stride=strides[i], dilation_rate=dilation_rate[i], padding=padding)
      dim_tags.append(DimensionTag(
        kind=DimensionTag.Types.Spatial, description="%s:conv:s%i" % (name, i), dimension=new_dim,
        derived_from_tag=old_tag, undefined=not old_tag))
    dim_tags.append(DimensionTag(kind=DimensionTag.Types.Feature, description="%s:channel" % name, dimension=n_out))
    feature_dim_axis = NotSpecified
    # Swap the dims if the input dim order doesn't fit the flag auto_use_channel_first.
    if tf_util.is_gpu_available_in_session() and (auto_use_channel_first or input_data.is_batch_feature_major):
      feature_dim_axis = 1
      dim_tags = dim_tags[:1] + dim_tags[-1:] + dim_tags[1:-1]
    return Data(
      name="%s_output" % name, dim_tags=dim_tags, dim=n_out, feature_dim_axis=feature_dim_axis,
      batch=data.batch, beam=data.beam, control_flow_ctx=data.control_flow_ctx)

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    deps = super(ConvLayer, self).get_dep_layers()
    if self.filter_layer:
      deps.append(self.filter_layer)
    if self.bias_layer:
      deps.append(self.bias_layer)
    return deps

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param get_layer:
    """
    super(ConvLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    if d.get("filter"):
      d["filter"] = get_layer(d["filter"])
    if d.get("bias"):
      d["bias"] = get_layer(d["bias"])


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
    from returnn.tf.util.basic import check_input_dim, DimensionTag
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
    y = tf_compat.v1.nn.pool(
      x, window_shape=pool_size, pooling_type=mode, padding=padding,
      dilation_rate=dilation_rate, strides=strides, data_format=data_format)
    # y shape is [batch] + spatial_dims + [n_out].
    self.output.placeholder = y
    index_shift = self.output.get_spatial_batch_axes()[0]
    for i, in_axis in enumerate(input_data.get_spatial_batch_axes()):
      in_tag = input_data.dim_tags[in_axis]
      if in_tag.dimension is None and in_tag.dyn_size is not None:
        size = in_tag.dyn_size
        with tf_util.same_control_flow_ctx(size):
          size = ConvLayer.calc_out_dim(
            in_dim=size,
            filter_size=pool_size[i], stride=strides[i],
            dilation_rate=dilation_rate[i], padding=padding)
        out_tag = self.output.dim_tags[i + index_shift]
        size_tag = DimensionTag.get_tag_from_size_tensor(size)
        if not size_tag:
          out_tag.set_tag_on_size_tensor(size, batch=in_tag.batch)
        else:
          out_tag.declare_same_as(size_tag)

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
    from ..util.data import DimensionTag
    # y shape is [batch] + spatial_dims + [n_out].
    data = get_concat_sources_data_template(sources, name="%s_output" % name)
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
    if all(p == s == d == 1 for (p, s, d) in zip(pool_size, strides, dilation_rate)):
      # Identity function. Just copy and don't do anything.
      return data
    padding = padding.upper()
    data = data.copy_as_batch_spatial_major()  # just to have the dim tags in order [B,S...,D]
    dim_tags = list(data.dim_tags)
    for i in range(len(pool_size)):
      if pool_size[i] == strides[i] == 1 or (strides[i] == 1 and padding == "SAME"):
        continue  # identity in this axis
      new_dim = None
      if dim_tags[i + 1].dimension is not None:
        new_dim = ConvLayer.calc_out_dim(
          in_dim=dim_tags[i + 1].dimension,
          filter_size=pool_size[i], stride=strides[i], dilation_rate=dilation_rate[i], padding=padding)
      dim_tags[i + 1] = DimensionTag(
        kind=DimensionTag.Types.Spatial, description="%s:pool:s%i" % (name, i), dimension=new_dim)
    feature_dim_axis = NotSpecified
    # Swap the dims if use_channel_first is set.
    if tf_util.is_gpu_available_in_session() and use_channel_first:
      feature_dim_axis = 1
      dim_tags = dim_tags[:1] + dim_tags[-1:] + dim_tags[1:-1]
    return Data(
      name="%s_output" % name,
      dim_tags=dim_tags,
      dim=data.dim,
      dtype=data.dtype,
      feature_dim_axis=feature_dim_axis,
      batch=data.batch, beam=data.beam, control_flow_ctx=data.control_flow_ctx)


class DctLayer(_ConcatInputLayer):
  """
  Layer to perform DCT
  Wraps :func:`tf.signal.dct`. For further documentation on the input arguments, refer to
  https://www.tensorflow.org/api_docs/python/tf/signal/dct
  """

  layer_class = "dct"
  recurrent = True  # we should not shuffle in the time-dimension

  # noinspection PyShadowingBuiltins
  def __init__(self, type=2, n=None, norm=None, **kwargs):
    """
    :param int type: DCT type to perform. Must be 1, 2, 3, or 4
    :param int|None n: length of the transform
    :param str|None norm: normalization to apply. Must be None or "ortho"
    """
    assert n is None, 'Not implemented yet for n other than None'
    super(DctLayer, self).__init__(**kwargs)
    input_data = self.input_data.copy_as_batch_spatial_major()
    x = input_data.placeholder
    y = tf_compat.v1.spectral.dct(x, type=type, n=n, norm=norm)
    self.output.placeholder = y
    if n is None:
      self.output.size_placeholder = self.input_data.size_placeholder.copy()
    else:
      raise NotImplementedError

  @classmethod
  def get_out_data_from_opts(cls, name, sources, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    out = get_concat_sources_data_template(sources, name="%s_output" % name)
    out = out.copy_as_batch_spatial_major()
    return out


class TransposedConvLayer(_ConcatInputLayer):
  """
  Transposed convolution, sometimes also called deconvolution.
  See :func:`tf.nn.conv2d_transpose` (currently we support 1D/2D).
  """
  layer_class = "transposed_conv"
  recurrent = True

  # noinspection PyShadowingBuiltins
  def __init__(self, filter_size, activation, strides=None,
               padding="same",
               remove_padding=0,
               output_padding=None,
               with_bias=True,
               forward_weights_init="glorot_uniform", bias_init=0.0,
               filter=None, filter_perm=None, bias=None,
               **kwargs):
    """
    :param list[int] filter_size:
    :param list[int]|None strides: specifies the upscaling. by default, same as filter_size
    :param str padding: "same" or "valid"
    :param list[int]|int remove_padding:
    :param list[int|None]|int|None output_padding:
    :param bool with_bias: whether to add a bias. enabled by default.
      Note that the default is different from ConvLayer!
    :param str|None activation:
    :param forward_weights_init:
    :param bias_init:
    :param LayerBase|None filter: if given, will not create an own parameter, but use this as the filter
    :param dict[str,str]|None filter_perm: transposes the filter (input filter as layer)
    :param LayerBase|None bias: if given, will not create an own parameter, but use this as the bias
    """
    from returnn.tf.util.basic import get_initializer, get_activation_function, get_shape
    super(TransposedConvLayer, self).__init__(**kwargs)
    input_data = self.input_data.copy_as_batch_spatial_major()
    spatial_axes = input_data.get_spatial_axes()
    padding = padding.lower()
    if strides is None:
      strides = filter_size
    if not isinstance(remove_padding, (list, tuple)):
      remove_padding = [remove_padding] * len(spatial_axes)
    if not isinstance(output_padding, (list, tuple)):
      output_padding = [output_padding] * len(spatial_axes)
    assert len(spatial_axes) == len(filter_size) == len(strides) == len(remove_padding) == len(output_padding), (
      "%s: expected %i-D transposed-conv for input %r but got filter %r and strides %r" % (
        self, len(spatial_axes), input_data, filter_size, strides))
    assert len(spatial_axes) in [1, 2], "%s: %i-D not yet implemented..." % (self, len(spatial_axes))
    x = input_data.placeholder
    shape = get_shape(x)
    if len(spatial_axes) == 1:
      # TF supports only 2D transposed-conv, so wrap it.
      shape = shape[:2] + [1] + shape[2:]
      x = tf.reshape(x, shape)
      filter_size = list(filter_size) + [1]
      strides = list(strides) + [1]
      output_padding = list(output_padding) + [0]
    filter_shape = list(filter_size) + [self.output.dim, input_data.dim]  # transposed
    self.filter_layer = None
    if filter:
      self.filter_layer = filter
      filter_data = filter.output
      if filter_perm:
        filter_data = TransposeLayer.transpose(filter_data, perm=filter_perm, name="filter_transposed")
      if filter_data.batch_ndim < len(filter_shape) and len(spatial_axes) == 1:
        # Special case. Allow this.
        filter_data = filter_data.copy_add_spatial_dim(1)
      assert filter_data.batch_shape == tuple(filter_shape)
      filters = filter_data.placeholder
    else:
      with self.var_creation_scope():
        fwd_weights_initializer = get_initializer(
          forward_weights_init, seed=self.network.random.randint(2 ** 31), eval_local_ns={"layer": self})
        filters = self.add_param(tf_compat.v1.get_variable(
          # Use that name to easily identify the format later.
          # E.g. we might want to switch to some more canonical format at some point.
          name="W_native_transposed_conv", shape=filter_shape, initializer=fwd_weights_initializer))
    output_shape = (
      shape[:1] +
      [self.deconv_output_length(
        size, filter_size=filter_size[i], stride=strides[i],
        padding=padding, output_padding=output_padding[i])
       for (i, size) in enumerate(shape[1:-1])]
      + [self.output.dim])
    y = tf.nn.conv2d_transpose(
      x, filters, strides=[1] + list(strides) + [1], output_shape=output_shape, padding=padding.upper())
    if len(spatial_axes) == 1:
      y = tf.reshape(y, output_shape[:2] + [self.output.dim])
    if any(remove_padding):
      from returnn.tf.util.basic import single_strided_slice
      for i, p in enumerate(remove_padding):
        if p:
          assert isinstance(p, int)
          assert p > 0
          y = single_strided_slice(y, axis=i + 1, begin=p, end=-p)
    if bias:
      assert with_bias
    self.bias_layer = None
    if with_bias:
      if bias:
        self.bias_layer = bias
        b_ = bias.output.copy_compatible_to(self.output)
        y += b_.placeholder
      else:
        with self.var_creation_scope():
          bias_initializer = get_initializer(
            bias_init, seed=self.network.random.randint(2 ** 31) if bias_init else 0, eval_local_ns={"layer": self})
          b = self.add_param(tf_compat.v1.get_variable(
            name="bias", shape=(self.output.dim,), initializer=bias_initializer))
        y += b
    if activation:
      act_func = get_activation_function(activation)
      self.output_before_activation = OutputWithActivation(y, act_func=act_func)
    else:
      self.output_before_activation = OutputWithActivation(y)
    y = self.output_before_activation.y
    self.output.placeholder = y
    for idx, axis_wo_b in enumerate(input_data.get_spatial_axes()):
      axis = input_data.get_batch_axis(axis_wo_b)
      input_tag = input_data.dim_tags[axis]
      output_tag = self.output.dim_tags[axis]
      if input_tag.dimension is None:
        assert output_tag.dimension is None
        assert input_tag.dyn_size is not None
        size = input_tag.dyn_size
        with tf_util.same_control_flow_ctx(size):
          size = self.deconv_output_length(
            size,
            filter_size=filter_size[idx], stride=strides[idx],
            padding=padding, output_padding=output_padding[idx])
          r = remove_padding[idx]
          if r:
            assert isinstance(r, int)
            size = tf_util.simplify_add(size, -r * 2)
        output_tag.set_tag_on_size_tensor(size, batch=self.output.batch)

  @staticmethod
  def deconv_output_length(input_length,
                           filter_size,
                           padding,
                           output_padding=None,
                           stride=0,
                           dilation=1):
    """
    Determines output length of a transposed convolution given input length.
    Copied from conv_utils.deconv_output_length, adapted with simplification.

    :param T|int|tf.Tensor input_length:
    :param int filter_size:
    :param str padding: one of `"same"`, `"valid"`, `"full"`.
    :param int|None output_padding: amount of padding along the output dimension.
      Can be set to `None` in which case the output length is inferred.
    :param int stride:
    :param int dilation:
    :returns: The output length (integer)
    :rtype: T
    """
    assert padding in {'same', 'valid', 'full'}

    # Get the dilated kernel size
    filter_size = filter_size + (filter_size - 1) * (dilation - 1)

    if stride != 1:
      input_length = input_length * stride

    # Infer length if output padding is None, else compute the exact length
    if output_padding is None:
      if padding == 'valid':
        length = tf_util.simplify_add(input_length, max(filter_size - stride, 0))
      elif padding == 'full':
        length = tf_util.simplify_add(input_length, -(stride + filter_size - 2))
      elif padding == 'same':
        length = input_length
      else:
        length = None
    else:  # output_padding
      if padding == 'same':
        pad = filter_size // 2
      elif padding == 'valid':
        pad = 0
      elif padding == 'full':
        pad = filter_size - 1
      else:
        pad = None
      length = tf_util.simplify_add(input_length, -stride + filter_size - 2 * pad + output_padding)
    return length

  @classmethod
  def get_out_data_from_opts(cls, name, sources, n_out,
                             filter_size, strides=None,
                             padding="same",
                             remove_padding=0, output_padding=None, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param int n_out:
    :param list[int] filter_size:
    :param list[int]|None strides:
    :param str padding:
    :param list[int]|int remove_padding:
    :param list[int|None]|int|None output_padding:
    :rtype: Data
    """
    from ..util.data import DimensionTag
    out = get_concat_sources_data_template(sources)
    out = out.copy_as_batch_spatial_major()
    out = out.copy_template(name="%s_output" % name)
    assert out.have_feature_axis()
    out = out.copy_template_replace_dim(axis=out.feature_dim_axis, new_dim=n_out)
    dim_tags = list(out.dim_tags)
    if strides is None:
      strides = filter_size
    if isinstance(remove_padding, int):
      remove_padding = [remove_padding] * len(filter_size)
    if not isinstance(output_padding, (list, tuple)):
      output_padding = [output_padding] * len(filter_size)
    assert len(strides) == len(out.get_spatial_batch_axes()) == len(remove_padding) == len(output_padding), (
      "Expected strides for all spatial axes")
    for idx, axis in enumerate(out.get_spatial_batch_axes()):
      if not output_padding[idx] and remove_padding[idx] == 0 and strides[idx] == 1:
        if filter_size[idx] == 1 or padding.lower() == "same":
          continue
      tag = dim_tags[axis]
      dim = None
      if tag.dimension is not None:
        dim = cls.deconv_output_length(
          tag.dimension, filter_size=filter_size[idx], stride=strides[idx],
          padding=padding, output_padding=output_padding[idx]) - remove_padding[idx] * 2
      dim_tags[axis] = DimensionTag(
        kind=DimensionTag.Types.Spatial, description="%s_spatial%i_transposed_conv" % (name, idx),
        dimension=dim)
    return out.copy_template_new_dim_tags(dim_tags, keep_special_axes=True)

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    deps = super(TransposedConvLayer, self).get_dep_layers()
    if self.filter_layer:
      deps.append(self.filter_layer)
    if self.bias_layer:
      deps.append(self.bias_layer)
    return deps

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param get_layer:
    """
    super(TransposedConvLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    if d.get("filter"):
      d["filter"] = get_layer(d["filter"])
    if d.get("bias"):
      d["bias"] = get_layer(d["bias"])


class ReduceLayer(_ConcatInputLayer):
  """
  This reduces some axis by using "sum" or "max".
  It's basically a wrapper around tf.reduce_sum or tf.reduce_max.
  """
  layer_class = "reduce"

  def __init__(self, mode, axes=None, axis=None, keep_dims=False, enforce_batch_dim_axis=None, use_time_mask=None,
               **kwargs):
    """
    :param str mode: "sum" or "max", "argmin", "min", "argmax", "mean", "logsumexp"
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
    super(ReduceLayer, self).__init__(**kwargs)
    if axis is not None:
      assert axes is None, "don't provide both 'axes' and 'axis', layer %r" % kwargs["name"]
      axes = axis
    self.output.placeholder = self.reduce(
      input_data=self.input_data,
      mode=mode, axes=axes, keep_dims=keep_dims, enforce_batch_dim_axis=enforce_batch_dim_axis,
      use_time_mask=use_time_mask)

  @classmethod
  def reduce(cls, input_data, mode, axes=None, keep_dims=False, enforce_batch_dim_axis=None, use_time_mask=None):
    """
    :param Data input_data:
    :param str mode: "sum" or "max", "argmin", "min", "argmax", "mean", "logsumexp"
    :param int|list[int]|str axes: One axis or multiple axis to reduce.
      It accepts the special tokens "B"|"batch", "spatial", "spatial_except_time", or "F"|"feature",
      and it is strongly recommended to use some of these symbolic names.
      See :func:`Data.get_axes_from_description`.
    :param bool keep_dims: if dimensions should be kept (will be 1)
    :param int enforce_batch_dim_axis: will swap the batch-dim-axis of the input with the given axis.
      e.g. 0: will convert the input into batch-major format if not already like that.
      Note that this is still not enough in some cases, e.g. when the other axes are also not as expected.
      The strong recommendation is to use a symbolic axis description.
    :param bool use_time_mask: if we reduce over the time-dim axis, use the seq len info.
      By default, in that case, it will be True.
    :rtype: tf.Tensor
    """
    from returnn.tf.util.basic import expand_multiple_dims
    if enforce_batch_dim_axis is None and cls.need_enforce_batch_dim_axis(axes):
      enforce_batch_dim_axis = 0
    assert not input_data.sparse
    x = input_data
    if enforce_batch_dim_axis is not None and x.batch_dim_axis != enforce_batch_dim_axis:
      x = x.copy_with_batch_dim_axis(enforce_batch_dim_axis)
    axes = cls.get_axes(axes, input_data=x)
    if use_time_mask is None:
      if x.time_dim_axis in axes:
        use_time_mask = True
      else:
        use_time_mask = False
    assert isinstance(use_time_mask, bool)
    mode = mode.lower()
    if mode == "avg":  # alias
      mode = "mean"
    reduce_abs_funcs = {
      name: getattr(tf, "reduce_%s" % name) for name in ["max", "min", "sum", "logsumexp", "any", "all"]}
    reduce_rel_func = {"mean": tf.reduce_mean}
    arg_funcs = {name: getattr(tf, name) for name in ["argmax", "argmin"]}
    funcs = dict(list(reduce_abs_funcs.items()) + list(reduce_rel_func.items()) + list(arg_funcs.items()))
    assert mode in funcs, "invalid mode %r. choose from: %r" % (mode, funcs)
    f = funcs[mode]
    x_ = x.placeholder
    # Check if we should ignore some frames, e.g. via masking.
    if use_time_mask:
      if mode in reduce_abs_funcs or (mode in reduce_rel_func and axes == [x.time_dim_axis]):
        # For sum, the fastest and simplest way is masking.
        for axis in axes:
          if axis == x.batch_dim_axis:
            continue
          axis_wo_b = x.get_batch_axis_excluding_batch(axis)
          if axis_wo_b not in x.size_placeholder:
            continue
          assert axis == x.time_dim_axis
          mask = x.get_sequence_mask_broadcast(axis=axis)

          zeros = tf.zeros((), dtype=x.placeholder.dtype)
          # Cannot call x.placeholder.dtype.{min,max} in case input is e.g. a bool
          if x.placeholder.dtype.is_floating:
            replacement_value = {
              tf.reduce_mean: zeros,
              tf.reduce_sum: zeros,
              tf.reduce_logsumexp: zeros + x.placeholder.dtype.min,
              tf.reduce_min: zeros + x.placeholder.dtype.max,
              tf.reduce_max: zeros + x.placeholder.dtype.min}[f]
          elif x.placeholder.dtype.is_bool:
            replacement_value = {
              tf.reduce_any: zeros,
              tf.reduce_all: tf.ones((), dtype=x.placeholder.dtype)}[f]
          else:
            assert False

          x_ = tf_util.where_bc(mask, x_, replacement_value, "x_masked_axis_%i" % axis)
          if f == tf.reduce_mean:
            seq_len_bc = tf.reshape(
              x.get_sequence_lengths(),
              [1 if (i != x.batch_dim_axis) else -1 for i in range(x.batch_ndim)])  # (1,..B..,1)
            x_ = x_ / tf.cast(seq_len_bc, tf.float32)
            f = tf.reduce_sum
      elif f == tf.reduce_mean:
        # Flattening.
        if x.time_dim_axis in axes:
          assert not keep_dims, "not yet implemented otherwise"
          assert x.batch_dim_axis in axes, "not yet implemented otherwise"
          axes = [a if (a < x.time_dim_axis) else (a - 1)
                  for a in axes if a != x.time_dim_axis]
          x = x.copy_time_flattened()
          x_ = x.placeholder
    if mode in arg_funcs:
      assert len(axes) == 1, "For argmax/argmin, only one reduction axis is supported"
      y = f(x_, axis=axes[0], output_type=tf.int32)
      # argmax and argmin don't support keep_dims argument
      # so we emulate it manually
      if keep_dims:
        y = expand_multiple_dims(y, axes=axes)
    else:
      y = f(x_, axis=axes, keepdims=keep_dims)
    return y

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
  def get_out_data_from_opts(cls, name, sources, mode="", axes=None, axis=None, keep_dims=False,
                             enforce_batch_dim_axis=None,
                             **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param str mode: (default here "" because other code uses this function)
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
    out_size = x.size_placeholder.copy() if x.size_placeholder else {}
    if keep_dims:
      for i in axes:
        y_shape[i] = 1
      if x.batch_dim_axis is not None:
        del y_shape[x.batch_dim_axis]
      out_size = {i: size for (i, size) in out_size.items() if x.get_batch_axis(i) not in axes}
    else:
      if out_batch_dim_axis in axes:
        out_batch_dim_axis = None
      if out_time_dim_axis in axes:
        out_time_dim_axis = NotSpecified
      if out_feature_dim_axis in axes:
        out_feature_dim_axis = NotSpecified
      for i in reversed(sorted(set(axes + ([x.batch_dim_axis] if x.batch_dim_axis is not None else [])))):
        del y_shape[i]
      out_size = {x.get_batch_axis(i): size for (i, size) in out_size.items()}  # by batch-axis
      out_size = {i: size for (i, size) in out_size.items() if i not in axes}
      for i in reversed(sorted(set(axes))):
        if out_batch_dim_axis and i < out_batch_dim_axis:
          out_batch_dim_axis -= 1
        if out_time_dim_axis and out_time_dim_axis is not NotSpecified and i < out_time_dim_axis:
          out_time_dim_axis -= 1
        if out_feature_dim_axis and out_feature_dim_axis is not NotSpecified and i < out_feature_dim_axis:
          out_feature_dim_axis -= 1
        out_size = {(j - 1) if i < j else j: size for (j, size) in out_size.items()}
      out_size = {
        i if out_batch_dim_axis is None or out_batch_dim_axis >= i else i - 1: size
        for (i, size) in out_size.items()}  # by axis without batch-dim
    sparse_out = mode.lower().startswith("arg")
    if sparse_out:
      out_feature_dim_axis = None
    return Data(
      name="%s_output" % name,
      shape=y_shape,
      batch_dim_axis=out_batch_dim_axis,
      time_dim_axis=out_time_dim_axis,
      feature_dim_axis=out_feature_dim_axis,
      dtype="int32" if sparse_out else x.dtype,
      sparse=sparse_out,
      dim=x.batch_shape[axes[0]] if sparse_out else NotSpecified,
      size_placeholder=out_size,
      beam=x.beam)


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

  @classmethod
  def get_out_data_from_opts(cls, num_pieces, sources, name, **kwargs):
    """
    :param int num_pieces:
    :param list[LayerBase] sources:
    :param str name:
    :rtype: Data
    """
    from ..util.data import DimensionTag
    out = get_concat_sources_data_template(sources, name="%s_output" % name)
    assert out.have_feature_axis()
    assert not out.sparse
    assert out.dim % num_pieces == 0
    dim = out.dim // num_pieces
    tag = DimensionTag(
      kind=DimensionTag.Types.Feature, description="%s_reduce_out" % name,
      dimension=dim)
    return out.copy_template_replace_dim_tag(axis=out.feature_dim_axis, new_dim_tag=tag)


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
    :param int|list[int]|str axis:
    :param int|None enforce_batch_dim_axis:
    :param bool allow_no_op:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    input_data = get_concat_sources_data_template(sources)
    if enforce_batch_dim_axis is not None:
      input_data = input_data.copy_with_batch_dim_axis(enforce_batch_dim_axis)
    axis_indices = cls._get_axes(axis, input_data=input_data)
    if allow_no_op:
      if not axis_indices:
        return input_data.copy("%s_output" % kwargs["name"])
    # remove the axis in reversed order
    for axis_idx in sorted(axis_indices, reverse=True):
      input_data = input_data.copy_template_excluding_axis(axis_idx)
    return input_data


class StackLayer(LayerBase):
  """
  Stacks multiple inputs together using :func:`tf.stack`.
  """
  layer_class = "stack"

  def __init__(self, axis=None, **kwargs):
    """
    :param int|None axis: new axis.
      If not given, will use Data.get_default_new_axis_for_dim_tag(<spatial>),
      i.e. some reasonable default for a new spatial axis.
    """
    super(StackLayer, self).__init__(**kwargs)
    axis_, common_source = self._get_axis_and_common(self.sources)
    if axis is None:
      axis = axis_
    assert self.output.batch_shape[axis] == len(self.sources)
    sources_ = [
      src.output.copy_compatible_to(common_source, unbroadcast=True)
      for src in self.sources]
    self.output.placeholder = tf.stack([src.placeholder for src in sources_], axis=axis)

  @classmethod
  def _get_axis_and_common(cls, sources):
    """
    :param list[LayerBase] sources:
    :rtype: (int,Data)
    """
    from returnn.tf.util.basic import DimensionTag
    common_source = Data.get_common_data([src.output for src in sources]).copy_template()
    tag = DimensionTag(kind=DimensionTag.Types.Spatial, dimension=1)
    return common_source.get_default_new_axis_for_dim_tag(tag), common_source

  @classmethod
  def get_out_data_from_opts(cls, name, sources, axis=None, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param int|None axis:
    :rtype: Data
    """
    axis_, common_source = cls._get_axis_and_common(sources)
    if axis is None:
      axis = axis_
    out = common_source.copy_template(name="%s_output" % name)
    out = out.copy_add_spatial_dim(spatial_dim_axis=axis, dim=len(sources))
    return out


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
      filters = self.add_param(tf_compat.v1.get_variable(
        name="W", shape=size, initializer=tf.constant_initializer(1.0 / numpy.prod(size))))
    filters = tf.reshape(filters, shape=list(size) + [1, 1])
    y = tf_compat.v1.nn.convolution(x, filter=filters, padding=padding.upper())  # result: (new_batch_dim, ..., 1)
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
    from ..util.data import DimensionTag
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
      dim_tags = list(data.dim_tags)
      for i, a in enumerate(axes):
        dim_tags[a] = DimensionTag(
          kind=dim_tags[a].kind, description="%s:weighted-sum:%i" % (name, i), dimension=res_dims[i])
      data = data.copy_template_new_dim_tags(dim_tags, keep_special_axes=True)
    else:
      assert all([d == 1 for d in res_dims])
      dim_tags = [tag for a, tag in enumerate(data.dim_tags) if a not in axes]
      data = data.copy_template_new_dim_tags(dim_tags)
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
      w = self.add_param(tf_compat.v1.get_variable(name="W", shape=size, initializer=tf.constant_initializer(1.0)))
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


class PrefixInTimeLayer(_ConcatInputLayer):
  """
  Adds some prefix in time dimension.
  This is kind of the reverse of :class:`SliceNdLayer` does.
  """
  layer_class = "prefix_in_time"
  recurrent = True

  def __init__(self, prefix=0.0, repeat=1, size_base=None, **kwargs):
    """
    :param float|str prefix: either some constant or another layer
    :param int|LayerBase repeat: how often to repeat the postfix
    :param LayerBase|None size_base: copy seq-lens from here
    """
    from returnn.tf.util.basic import DimensionTag
    super(PrefixInTimeLayer, self).__init__(**kwargs)
    self.output = self.input_data.copy(name="%s_output" % self.name)
    assert self.output.time_dim_axis is not None
    assert isinstance(prefix, (float, int)), "other layer src not yet supported"
    self.repeat_layer = None
    if isinstance(repeat, LayerBase):
      self.repeat_layer = repeat
      assert repeat.output.ndim == 0 and repeat.output.have_batch_axis() == self.input_data.have_batch_axis()
      repeat = repeat.output.placeholder
      self.output = self.output.copy_as_batch_spatial_major()
    else:
      assert isinstance(repeat, int)
      assert repeat >= 0
    self.repeat = repeat
    c = tf.constant(prefix, dtype=self.output.dtype)
    seq_len = self.output.get_sequence_lengths()
    if size_base:
      new_seq_len = size_base.output.get_sequence_lengths()
    else:
      new_seq_len = seq_len + repeat
    if not DimensionTag.get_tag_from_size_tensor(new_seq_len):
      tag = DimensionTag(
        description="time-with-prefix:%s" % self.get_absolute_name(),
        kind=DimensionTag.Types.Spatial, batch=self.output.batch)
      tag.set_tag_on_size_tensor(new_seq_len)
    self.output.size_placeholder[self.output.time_dim_axis_excluding_batch] = new_seq_len
    max_repeat = repeat if isinstance(repeat, int) else tf.maximum(tf.reduce_max(repeat), 0)
    shape = [((self.output.batch_shape[i] or tf.shape(self.output.placeholder)[i])
              if (i != self.output.time_dim_axis)
              else max_repeat)
             for i in range(self.output.batch_ndim)]
    prefix_t = tf.ones(shape, dtype=self.output.dtype) * c
    x = tf.concat([prefix_t, self.output.placeholder], axis=self.output.time_dim_axis)
    if not isinstance(repeat, int):
      assert isinstance(repeat, tf.Tensor) and repeat.shape.ndims == 1  # (B,)
      max_new_seq_len = tf.reduce_max(new_seq_len)
      from returnn.tf.util.basic import slice_nd
      assert (self.output.batch_dim_axis, self.output.time_dim_axis) == (0, 1)
      x = slice_nd(x, start=max_repeat - repeat, size=max_new_seq_len)
    self.output.placeholder = x

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    deps = super(PrefixInTimeLayer, self).get_dep_layers()
    if self.repeat_layer:
      deps.append(self.repeat_layer)
    return deps

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param returnn.tf.network.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    super(PrefixInTimeLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    if isinstance(d.get("repeat", None), str):
      d["repeat"] = get_layer(d["repeat"])
    if d.get("size_base", None):
      d["size_base"] = get_layer(d["size_base"])

  @classmethod
  def get_out_data_from_opts(cls, name, sources, size_base=None, repeat=None, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param LayerBase|None size_base:
    :param LayerBase|int|None repeat:
    :rtype: Data
    """
    # Note: Time seq len is not correct...
    x = get_concat_sources_data_template(sources, name="%s_output" % name)
    if size_base:
      x.size_placeholder[x.time_dim_axis_excluding_batch] = size_base.output.get_sequence_lengths()
    if isinstance(repeat, LayerBase):
      x = x.copy_as_batch_spatial_major()
    return x


class PostfixInTimeLayer(_ConcatInputLayer):
  """
  Adds some postfix in time dimension.
  """
  layer_class = "postfix_in_time"
  recurrent = True

  def __init__(self, postfix=0.0, repeat=1, **kwargs):
    """
    :param float|int|LayerBase postfix: constant or other layer without time axis to use as postfix
    :param int repeat: how often to repeat the postfix
    """
    from returnn.tf.util.basic import DimensionTag
    super(PostfixInTimeLayer, self).__init__(**kwargs)
    self.output = self.input_data.copy(name="%s_output" % self.name)
    assert self.output.time_dim_axis is not None
    assert isinstance(postfix, (float, int, LayerBase))
    if isinstance(postfix, LayerBase):
      self.postfix_layer = postfix
      assert not postfix.output.have_time_axis(), 'Postfix layer with time axis not implemented yet'
      postfix = postfix.output.copy_compatible_to(self.output)
      assert self.output.time_dim_axis_excluding_batch not in postfix.size_placeholder
      c = postfix.placeholder
    else:
      self.postfix_layer = None
      c = tf.constant(postfix, dtype=self.output.dtype)
    added_shape = [
      ((self.output.batch_shape[i] or tf.shape(self.output.placeholder)[i])
       if (i != self.output.time_dim_axis)
       else repeat)
      for i in range(self.output.batch_ndim)]
    x = tf.concat(
      [self.output.placeholder, tf.zeros(added_shape, dtype=self.output.dtype)],
      axis=self.output.time_dim_axis)  # make enough space
    seq_len = self.output.size_placeholder[self.output.time_dim_axis_excluding_batch]  # (B,)
    seq_len_bc = tf.reshape(
      seq_len, [1 if (i != self.output.batch_dim_axis) else -1 for i in range(self.output.batch_ndim)])  # (1,..B..,1)
    time_idxs = tf.range(tf.shape(x)[self.output.time_dim_axis])
    time_idxs_bc = tf.reshape(
      time_idxs, [1 if (i != self.output.time_dim_axis) else -1 for i in range(self.output.batch_ndim)])  # (1,..T..,1)
    mask = tf.less(time_idxs_bc, seq_len_bc)
    from returnn.tf.util.basic import where_bc
    self.output.placeholder = where_bc(mask, x, c)
    new_seq_len = seq_len + repeat
    tag = DimensionTag(
      description="time-with-postfix:%s" % self.get_absolute_name(),
      kind=DimensionTag.Types.Spatial, batch=self.output.batch)
    tag.set_tag_on_size_tensor(new_seq_len)
    self.output.size_placeholder[self.output.time_dim_axis_excluding_batch] = new_seq_len

  @classmethod
  def get_out_data_from_opts(cls, name, sources, postfix=0.0, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param float|int|LayerBase postfix: constant or other layer without time axis to use as postfix
    :rtype: Data
    """
    # Note: Time seq len is not correct...
    out = get_concat_sources_data_template(sources, name="%s_output" % name)
    if isinstance(postfix, LayerBase):
      out.beam = SearchBeam.get_combined_beam(out.beam, postfix.output.beam)
    return out

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param get_layer:
    """
    super(PostfixInTimeLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    if d.get("postfix", None):
      postfix = d["postfix"]
      if isinstance(postfix, str):
        d["postfix"] = get_layer(postfix)

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    deps = super(PostfixInTimeLayer, self).get_dep_layers()
    if self.postfix_layer:
      deps.append(self.postfix_layer)
    return deps


class TimeChunkingLayer(_ConcatInputLayer):
  """
  Performs chunking in time. See :func:`TFNativeOp.chunk`.
  """
  layer_class = "time_chunking"
  recurrent = True

  def __init__(self, chunk_size, chunk_step, **kwargs):
    """
    :param int chunk_size:
    :param int chunk_step:
    """
    super(TimeChunkingLayer, self).__init__(**kwargs)
    self.chunk_size = chunk_size
    self.chunk_step = chunk_step
    from returnn.tf.native_op import chunk
    x = self.input_data.copy_as_time_major()
    index = tf.cast(x.get_sequence_mask(), tf.float32)
    out, oindex = chunk(x.placeholder, index=index, chunk_step=chunk_step, chunk_size=chunk_size)
    out.set_shape((None, None, self.output.dim))
    self.output.placeholder = out
    self.output.size_placeholder = {0: tf.reduce_sum(tf.cast(oindex, tf.int32), axis=0)}

  @classmethod
  def get_out_data_from_opts(cls, name, sources, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    data = get_concat_sources_data_template(sources, name="%s_output" % name).copy_as_time_major()
    assert data.batch_shape == (None, None, data.dim)
    return data


class TimeUnChunkingLayer(_ConcatInputLayer):
  """
  Performs chunking in time. See :func:`TFNativeOp.chunk`.
  """
  layer_class = "time_unchunking"
  recurrent = True

  def __init__(self, chunking_layer, **kwargs):
    """
    :param TimeChunkingLayer chunking_layer:
    """
    super(TimeUnChunkingLayer, self).__init__(**kwargs)
    assert isinstance(chunking_layer, TimeChunkingLayer)
    self.chunking_layer = chunking_layer
    chunk_size = chunking_layer.chunk_size
    chunk_step = chunking_layer.chunk_step
    orig_shape = tf.shape(chunking_layer.input_data.placeholder)
    n_time = orig_shape[chunking_layer.input_data.time_dim_axis]
    n_batch = orig_shape[chunking_layer.input_data.batch_dim_axis]
    from returnn.tf.native_op import unchunk
    x = self.input_data.copy_as_time_major()
    index = tf.cast(x.get_sequence_mask(), tf.float32)
    out, oindex, factors = unchunk(
      x.placeholder, index=index, chunk_step=chunk_step, chunk_size=chunk_size, n_time=n_time, n_batch=n_batch)
    out.set_shape((None, None, self.output.dim))
    self.output.placeholder = out
    self.output.size_placeholder = {0: chunking_layer.input_data.get_sequence_lengths()}

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    return super(TimeUnChunkingLayer, self).get_dep_layers() + [self.chunking_layer]

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param get_layer:
    """
    super(TimeUnChunkingLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    if "chunking_layer" in d:
      d["chunking_layer"] = get_layer(d["chunking_layer"])

  @classmethod
  def get_out_data_from_opts(cls, name, sources, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    data = get_concat_sources_data_template(sources, name="%s_output" % name).copy_as_time_major()
    assert data.batch_shape == (None, None, data.dim)
    return data


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
    from returnn.tf.util.basic import prod, get_shape, get_padding_info_dict_ref, mask_dyn_seq_len_nd
    super(DotLayer, self).__init__(**kwargs)
    a_out = self.sources[0].output.copy()
    b_out = self.sources[1].output.copy()
    a_reduce_axes = a_out.get_axes_from_description(red1)
    b_reduce_axes = b_out.get_axes_from_description(red2)
    assert a_reduce_axes and b_reduce_axes, "%s: sources %r, red1 %r, red2 %r" % (self, self.sources, red1, red2)
    a_var_axes = a_out.get_axes_from_description(var1)
    b_var_axes = b_out.get_axes_from_description(var2)
    assert not set(a_reduce_axes).intersection(a_var_axes), "%s: sources %r, red1 %r, red2 %r, var1 %r, var2 %r" % (
      self, self.sources, red1, red2, var1, var2)
    assert not set(b_reduce_axes).intersection(b_var_axes), "%s: sources %r, red1 %r, red2 %r, var1 %r, var2 %r" % (
      self, self.sources, red1, red2, var1, var2)
    a_rem_axes = [i for i in range(a_out.batch_ndim) if i not in a_var_axes + a_reduce_axes]
    b_rem_axes = [i for i in range(b_out.batch_ndim) if i not in b_var_axes + b_reduce_axes]
    assert len(a_rem_axes) == len(b_rem_axes)

    # ensure that a_rem_axes and b_rem_axes are in the same order
    map_a_to_b_rem_axes = b_out.find_matching_dim_map(a_out, a_rem_axes)
    assert all(b_axis in map_a_to_b_rem_axes.values() for b_axis in b_rem_axes)
    b_rem_axes = [map_a_to_b_rem_axes[a_axis] for a_axis in a_rem_axes]

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
    a_shape = get_shape(a)
    b_shape = get_shape(b)
    a_rem_dims = [a_shape[i] for i in a_rem_axes]
    b_rem_dims = [b_shape[i] for i in b_rem_axes]
    assert len(a_rem_axes) == len(b_rem_axes), "%s: remaining shared (batch) axes do not match. sources %r" % (
      self, self.sources)
    assert all([
      a_out.dim_tags[i1] == b_out.dim_tags[i2] or d1 == d2
      for (d1, d2, i1, i2) in zip(a_rem_dims, b_rem_dims, a_rem_axes, b_rem_axes)])
    a_var_dims = [a_shape[i] for i in a_var_axes]
    b_var_dims = [b_shape[i] for i in b_var_axes]
    a_reduce_dims = [a_shape[i] for i in a_reduce_axes]
    b_reduce_dims = [b_shape[i] for i in b_reduce_axes]
    assert len(a_reduce_axes) == len(b_reduce_axes)
    assert all([
      a_out.dim_tags[i1] == b_out.dim_tags[i2] or d1 == d2
      for (d1, d2, i1, i2) in zip(a_reduce_dims, b_reduce_dims, a_reduce_axes, b_reduce_axes)])
    a_var_dim = prod(a_var_dims)
    b_var_dim = prod(b_var_dims)
    a_reduce_dyn_axes = [i for i in a_reduce_axes if a_out.batch_shape[i] is None]
    b_reduce_dyn_axes = [i for i in b_reduce_axes if b_out.batch_shape[i] is None]
    assert len(a_reduce_dyn_axes) == len(b_reduce_dyn_axes)
    if a_reduce_dyn_axes:
      a_pad, b_pad = get_padding_info_dict_ref(a), get_padding_info_dict_ref(b)
      a_pad_values = [a_pad.get(a_out.dim_tags[i], None) for i in a_reduce_dyn_axes]
      b_pad_values = [b_pad.get(b_out.dim_tags[i], None) for i in b_reduce_dyn_axes]
      if set(a_pad_values) == {0}:
        self._info_reduce_mask = "source-0-already-masked"  # it's already masked as needed
      elif set(b_pad_values) == {0}:
        self._info_reduce_mask = "source-1-already-masked"  # it's already masked as needed
      else:
        # We need to apply a mask.
        # We don't need it on both a and b. We can either apply it on a or on b.
        # Use some very simple heuristic where the mask is maybe cheaper.
        if len(a_shape) < len(b_shape):
          a = mask_dyn_seq_len_nd(a_out, pad_value=0, axes=a_reduce_dyn_axes)
          self._info_reduce_mask = "mask-source-0"
        else:
          b = mask_dyn_seq_len_nd(b_out, pad_value=0, axes=b_reduce_dyn_axes)
          self._info_reduce_mask = "mask-source-1"
    else:
      self._info_reduce_mask = "none-dynamic"
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
            tf_compat.v1.assert_equal(
              a_rem_dims, b_rem_dims, data=[a_shape, b_shape, a_rem_dims, b_rem_dims], summarize=100),
            tf_compat.v1.assert_equal(
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
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param returnn.tf.network.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    super(DotLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    rec_time_dims = network.get_all_rec_time_dims()
    if rec_time_dims:
      assert len(d["sources"]) == 2, "dot-layer %r: needs exactly two sources" % (d["name"],)
      src1, src2 = d["sources"]
      assert isinstance(src1, LayerBase) and isinstance(src2, LayerBase)
      # Maybe we want to add some of the outer rec layer dims to the var1/var2 list,
      # or use those rec layer dims as further common dims (implicitly).
      dims1 = set(tag for tag in rec_time_dims if tag in src1.output.dim_tags)
      dims2 = set(tag for tag in rec_time_dims if tag in src2.output.dim_tags)
      # If the rec layer dim is the same as some other dim,
      # and was already explicitly specified in var1/var2 before,
      # skip it.
      var1 = d.get("var1", -2)  # the default should really not be used...
      var2 = d.get("var2", -1)  # the default should really not be used...
      var1_ = set([src1.output.dim_tags[i] for i in src1.output.get_axes_from_description(var1)])
      var2_ = set([src2.output.dim_tags[i] for i in src2.output.get_axes_from_description(var2)])
      dims1.difference_update(var1_)
      dims2.difference_update(var2_)
      # The common dims should be shared. The shared common dims are implicit, so nothing to do about them.
      dims_common = dims1.intersection(dims2)
      # Those are dims which should be added to var1/var2.
      dims1.difference_update(dims_common)
      dims2.difference_update(dims_common)

      def _add(dims, val, d_key):
        if not dims:
          return
        if val is None or val == "":
          val = []
        elif not isinstance(val, (tuple, list)):
          val = [val]
        d[d_key] = val + type(val)(dims)

      _add(dims1, var1, "var1")
      _add(dims2, var2, "var2")

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
    from ..util.data import DimensionTag, BatchInfo
    assert len(sources) == 2, "dot-layer %r: needs exactly two sources" % (name,)
    # See __init__.
    a_out = sources[0].output.copy()
    a_reduce_axes = a_out.get_axes_from_description(red1)
    b_out = sources[1].output.copy()
    assert not a_out.beam or not b_out.beam or a_out.beam == b_out.beam
    b_reduce_axes = b_out.get_axes_from_description(red2)
    assert a_reduce_axes and b_reduce_axes, "%s: sources %r, red1 %r, red2 %r" % (name, sources, red1, red2)
    a_var_axes = a_out.get_axes_from_description(var1)
    b_var_axes = b_out.get_axes_from_description(var2)
    assert not set(a_reduce_axes).intersection(a_var_axes)
    assert not set(b_reduce_axes).intersection(b_var_axes)
    a_rem_axes = [i for i in range(a_out.batch_ndim) if i not in a_var_axes + a_reduce_axes]
    b_rem_axes = [i for i in range(b_out.batch_ndim) if i not in b_var_axes + b_reduce_axes]
    assert len(a_rem_axes) == len(b_rem_axes)

    # ensure that a_rem_axes and b_rem_axes are in the same order
    map_a_to_b_rem_axes = b_out.find_matching_dim_map(a_out, a_rem_axes)
    assert all(b_axis in map_a_to_b_rem_axes.values() for b_axis in b_rem_axes)
    b_rem_axes = [map_a_to_b_rem_axes[a_axis] for a_axis in a_rem_axes]

    a_rem_dims = [a_out.dim_tags[i] for i in a_rem_axes]
    a_var_dims = [a_out.dim_tags[i] for i in a_var_axes]
    b_var_dims = [b_out.dim_tags[i] for i in b_var_axes]

    def find_axis(a_axis, b_axis):
      """
      :param int|None a_axis:
      :param int|None b_axis:
      :rtype: int|None|NotSpecified
      """
      axis = None
      if a_axis is not None:
        axis = cls._axis1_to_output(a_axis, a_rem_axes=a_rem_axes, a_var_axes=a_var_axes)
      if axis is None and b_axis is not None:
        axis = cls._axis2_to_output(b_axis, b_rem_axes=b_rem_axes, a_var_axes=a_var_axes, b_var_axes=b_var_axes)
      if axis is None and (a_axis is not None or b_axis is not None):
        # We had some time dim axis before and reduced it now.
        # But maybe there are others, so let's automatically figure out.
        # this should not happen for the batch_dim_axis, we chack for that outside this function
        axis = NotSpecified
      return axis

    time_dim_axis = find_axis(a_out.time_dim_axis, b_out.time_dim_axis)

    if not b_var_dims and add_var2_if_empty:
      b_var_dims.append(
        DimensionTag(kind=DimensionTag.Types.Spatial, description="%s:dot:dummy-var2" % name, dimension=1))

    dim_tags = list(a_rem_dims + a_var_dims + b_var_dims)
    return Data(
      name="%s_output" % name,
      dim_tags=dim_tags,
      time_dim_axis=time_dim_axis,
      dtype=a_out.dtype,
      batch=BatchInfo.get_common_batch_info([src.batch for src in (a_out, b_out)]),
      beam=SearchBeam.get_combined_beam(a_out.beam, b_out.beam))


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
    from returnn.tf.util.basic import single_strided_slice
    import numpy
    super(ShiftAxisLayer, self).__init__(**kwargs)
    assert isinstance(amount, int)
    axis = self.input_data.get_axis_from_description(axis)
    paddings = numpy.zeros(shape=(self.input_data.batch_ndim, 2), dtype="int32")
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
      new_size = tf.clip_by_value(
        self.output.size_placeholder[axis_wob] + size_delta, 0, tf.shape(shifted)[axis])
      from ..util.data import DimensionTag
      DimensionTag(
        kind=DimensionTag.Types.Spatial, description="shift_axis",
        dyn_size=new_size, batch=self.output.batch,
        src_data=self.output, src_axis=axis)
      self.output.size_placeholder[axis_wob] = new_size

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
    from ..util.data import DimensionTag
    out = get_concat_sources_data_template(sources, name="%s_output" % name)
    assert isinstance(amount, int)
    axis = out.get_axis_from_description(axis)
    tag = out.dim_tags[axis]
    dim = None if tag.dimension is None else max(0, tag.dimension - abs(amount))
    tag = DimensionTag(kind=tag.kind, description="%s_shift_axis" % name, dimension=dim)
    return out.copy_template_replace_dim_tag(axis=axis, new_dim_tag=tag)


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
    from returnn.tf.util.basic import DimensionTag
    super(ResizeLayer, self).__init__(**kwargs)
    # self.output.shape and self.output.batch_dim_axis are already set here via self.get_out_data_from_opts().
    axis = self.output.get_axis_from_description(axis)
    assert axis > 0, "batch-dim resize not supported"
    input_data = self.input_data.copy_as_batch_major()
    self.output.placeholder = input_data.placeholder
    self.output.size_placeholder = input_data.size_placeholder.copy()
    if (axis - 1) in self.output.size_placeholder:
      size = self.output.size_placeholder[axis - 1] * factor
      tag = DimensionTag(
        description="resize:%s" % self.get_absolute_name(),
        kind=DimensionTag.Types.Spatial, batch=self.output.batch)
      tag.set_tag_on_size_tensor(size)
      self.output.size_placeholder[axis - 1] = size

    # images expected shape: [batch, height, width, channels]
    remaining_axes = [i for i in range(self.output.batch_ndim) if i not in (0, axis)]
    x = dimshuffle(self.output.placeholder, [0, axis, 'x'] + remaining_axes)  # [batch,height,width] + remaining_axes
    from returnn.tf.util.basic import get_shape, optional_mul
    shape = get_shape(self.output.placeholder)
    remaining_shape = [shape[i] for i in remaining_axes]
    remaining_dim = optional_mul(*remaining_shape) if remaining_axes else 1
    x = tf.reshape(x, [shape[0], shape[axis], 1, remaining_dim])  # [batch,height,width,channels]
    new_size = shape[axis] * factor
    if kind == "linear":
      x = tf_compat.v1.image.resize_bilinear(x, size=(new_size, 1))
    elif kind == "cubic":
      x = tf_compat.v1.image.resize_bicubic(x, size=(new_size, 1))
    elif kind in ["nn", "nearest_neighbor"]:
      x = tf_compat.v1.image.resize_nearest_neighbor(x, size=(new_size, 1))
    elif kind == "fill":
      if self.input_data.sparse:
        assert isinstance(fill_value, int)
        if fill_value < 0:
          fill_value += self.input_data.dim
          assert fill_value > 0
      else:
        assert isinstance(fill_value, (int, float))
      assert isinstance(factor, int) and factor > 1
      from returnn.tf.util.basic import constant_with_shape
      fill_tensor = constant_with_shape(
        fill_value, shape=[shape[0], shape[axis], factor - 1, remaining_dim], dtype=x.dtype)
      x = tf.concat([x, fill_tensor], axis=2)  # [batch,height,factor,channels]
      x.set_shape(tf.TensorShape((None, None, factor, None)))
    else:
      raise Exception("invalid kind %r for resizing" % kind)
    x = tf.reshape(x, [shape[0], new_size] + remaining_shape)  # [batch,new_size] + remaining_shape
    if fill_dropout:
      from returnn.tf.util.basic import expand_dims_unbroadcast
      # We are going to build a mask over the axis. This mask will be shared over all seqs in the batch.
      # Similar to in tf.nn.dropout. Build random_tensor as uniform [keep_prob, 1.0 + keep_prob).
      random_tensor = 1.0 - fill_dropout  # keep_prop
      random_tensor += tf_compat.v1.random_uniform(
        [shape[axis], factor - 1], seed=self.network.random.randint(2**31))  # (old_size, factor - 1)
      # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
      mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
      mask = tf.concat([tf.ones((shape[axis], 1), dtype=tf.bool), mask], axis=1)  # (old_size, factor)
      mask = tf.reshape(mask, (new_size,))  # (new_size,)
      new_size_dropped = tf.reduce_sum(tf.cast(mask, tf.int32))
      mask = expand_dims_unbroadcast(mask, axis=0, dim=shape[0])  # (batch,new_size)
      x = tf.boolean_mask(x, mask)  # [batch*new_size_dropped] + remaining_shape
      x = tf.reshape(x, [shape[0], new_size_dropped] + remaining_shape)  # [batch, new_size_dropped] + remaining_shape
      if (axis - 1) in self.output.size_placeholder:
        orig_mask = tf.sequence_mask(
          self.output.size_placeholder[axis - 1], maxlen=new_size, dtype=tf.bool)  # (batch,new_size)
        size = tf.reduce_sum(tf.cast(tf.logical_and(mask, orig_mask), tf.int32), axis=1)
        tag = DimensionTag(
          description="resize:fill_dropout:%s" % self.get_absolute_name(),
          kind=DimensionTag.Types.Spatial, batch=self.output.batch)
        tag.set_tag_on_size_tensor(size)
        self.output.size_placeholder[axis - 1] = size
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
    from ..util.data import DimensionTag
    out = get_concat_sources_data_template(sources).copy_as_batch_major()
    out = out.copy_template(name="%s_output" % name)
    axis = out.get_axis_from_description(axis)
    assert axis != out.batch_dim_axis, "batch-dim resize not supported"
    tag = out.dim_tags[axis]
    dim = None if tag.dimension is None else (tag.dimension * factor)
    tag = DimensionTag(kind=tag.kind, description="%s_resize" % name, dimension=dim)
    return out.copy_template_replace_dim_tag(axis=axis, new_dim_tag=tag)


class CombineDimsLayer(MergeDimsLayer):
  """
  Combines multiple dimensions.
  See also :class:`MergeDimsLayer`. This is deprecated in favor of :class:`MergeDimsLayer`.
  """
  layer_class = "combine_dims"

  def __init__(self, **kwargs):
    """
    :param int|list[int]|str axes: one axis or multiple axis to reduce.
      this is counted with batch-dim, which by default is axis 0 (see enforce_batch_dim_axis).
      it also accepts the special tokens "B"|"batch", "spatial", "spatial_except_time", or "F"|"feature"
    """
    super(CombineDimsLayer, self).__init__(keep_order=True, **kwargs)

  @classmethod
  def get_out_data_from_opts(cls, **kwargs):
    """
    :rtype: Data
    """
    return super(CombineDimsLayer, cls).get_out_data_from_opts(keep_order=True, **kwargs)


class RemoveLayer(LayerBase):
  """
  Currently, assumes sparse data, and removes a specific symbol from the data.

  It is recommended to use :class:`MaskedComputationLayer` in combination with e.g.
  a :class:CompareLayer` instead, as this provides more flexibility.
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
    out_seq_lens = tf_compat.v1.count_nonzero(in_mask, axis=1, dtype=tf.int32)  # (batch,)
    max_out_seq_len = tf.reduce_max(out_seq_lens)  # scalar
    from returnn.tf.util.basic import constant_with_shape
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
    from ..util.data import DimensionTag
    assert len(sources) == 1, "%s layer %r: must have exactly one source" % (cls, name)
    assert sources[0].output.sparse, "%s layer %r: assumes sparse data" % (cls, name)
    out = sources[0].output.copy(name="%s_output" % name).copy_as_batch_major()
    dim_tag = DimensionTag(kind=DimensionTag.Types.Spatial, description="%s_removed_items", dimension=None)
    return out.copy_template_replace_dim_tag(axis=out.time_dim_axis, new_dim_tag=dim_tag)


class CombineLayer(LayerBase):
  """
  Applies a binary operation, such as addition, to all sources while accumulating the partial results.
  In the first step, the binary operation is performed on the first two sources.
  After the first step, the previous results is always the left-hand operator.

  Its basic working is similar to the `reduce` function used in functional programming.
  Also see :class:`ActivationLayer`, or :class:`CompareLayer`.
  """
  layer_class = "combine"

  # noinspection PyShadowingBuiltins
  def __init__(self, kind, sources, activation=None, with_bias=False,
               eval=None, eval_locals=None, eval_for_output_loss=False,
               **kwargs):
    """
    :param str kind:
      currently accepted values are `average`, `add`, `sub`, `mul`, `truediv`, `logical_and`, `logical_or`, or `eval`
    :param list[LayerBase] sources:
    :param str|None activation: if provided, activation function to apply, e.g. "tanh" or "relu"
    :param bool with_bias: if given, will add a trainable bias tensor
    :param str|callable eval: for kind="eval", will eval this string. or function. see :func:`_op_kind_eval`
    :param dict[str]|None eval_locals: locals for eval
    :param bool eval_for_output_loss: will do the same eval on layer.output_loss
    """
    super(CombineLayer, self).__init__(sources=sources, **kwargs)
    assert kind in ["average", "add", "sub", "mul", "truediv", "logical_and", "logical_or", "eval"], (
      "%s: Invalid `kind` %r for this layer." % (self, kind))
    op = self._get_op(kind=kind, eval_str=eval, eval_locals=eval_locals)
    x = op(sources)
    if eval_for_output_loss:
      assert eval
      assert all([layer.output_loss is not None for layer in sources])
      self.output_loss = self._op_kind_eval(
        sources=[InternalLayer(name=layer.name, network=self.network, output=Data.from_tensor(layer.output_loss))
                 for layer in sources],
        eval_str=eval, eval_locals=eval_locals)
    if with_bias:
      with self.var_creation_scope():
        b = self.add_param(tf_compat.v1.get_variable(
          name="b", shape=(self.output.dim,),
          initializer=tf.constant_initializer(value=0)))
      x += b
    if activation:
      from returnn.tf.util.basic import get_activation_function
      act_func = get_activation_function(activation)
      self.output_before_activation = OutputWithActivation(x, act_func=act_func)
    else:
      self.output_before_activation = OutputWithActivation(x)
    x = self.output_before_activation.y
    self.output.placeholder = x

  @classmethod
  def get_out_data_from_opts(cls, eval_locals=None, n_out=NotSpecified, out_type=None, sources=(), **kwargs):
    """
    :param dict[str]|None eval_locals: locals for eval, will also pass to out_type is out_type is a function
    :param int|None|NotSpecified n_out:
    :param dict[str]|None|(()->Data) out_type:
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
        if "shape" in out_type:
          out_type_.pop("dim_tags", None)
          out_type_.pop("batch_dim_axis", None)
          out_type_.pop("feature_dim_axis", None)
          out_type_.pop("time_dim_axis", None)
        out_type_.update(out_type)
      elif callable(out_type):
        def call_out_type_with_eval_locals(**out_type_kwargs):
          """
          :param out_type_kwargs:
          :rtype: Data
          """
          out_type_kwargs = out_type_kwargs.copy()
          out_type_kwargs.update(eval_locals or {})
          return out_type(**out_type_kwargs)
        out_type_ = call_out_type_with_eval_locals
      else:
        raise TypeError("unexpected type of out_type %r" % (out_type,))
    return super(CombineLayer, cls).get_out_data_from_opts(n_out=n_out, out_type=out_type_, sources=sources, **kwargs)

  @staticmethod
  def _op_dense_fn(sources, fn):
    """
    :param list[LayerBase] sources:
    :param ((x1,x2) -> y) fn: function to perform on x1 and x2
    :rtype: tf.Tensor
    """
    # Earlier we checked for the same dense dim.
    # Now, we completely rely on Data.get_common_data and Data.copy_compatible_to.
    # That should fail if they are not compatible. Otherwise it would add any needed broadcast dimensions.
    # All the dense element-wise functions should be able to deal with broadcasting.
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

  def _op_kind_truediv(self, sources):
    """
    :param list[LayerBase] sources:
    :rtype: tf.Tensor
    """
    return self._op_dense_fn(sources, tf.truediv)

  def _op_kind_average(self, sources):
    """
    :param list[LayerBase] sources:
    :rtype: tf.Tensor
    """
    x = self._op_kind_add(sources)
    x /= len(sources)
    return x

  def _op_kind_logical_and(self, sources):
    """
    :param list[LayerBase] sources:
    :rtype: tf.Tensor
    """
    return self._op_dense_fn(sources, tf.logical_and)

  def _op_kind_logical_or(self, sources):
    """
    :param list[LayerBase] sources:
    :rtype: tf.Tensor
    """
    return self._op_dense_fn(sources, tf.logical_or)

  def _op_kind_eval(self, sources, eval_str, eval_locals=None):
    """
    :param list[LayerBase]|list[tf.Tensor] sources:
    :param str|callable eval_str:
    :param dict[str]|None eval_locals:
    :rtype: tf.Tensor
    """
    used_sources = set()  # type: typing.Set[int]

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
          output = output.copy_compatible_to(self.output, check_dtype=False, check_sparse=False)
        if enforce_batch_major:
          output = output.copy_as_batch_major()
        if as_data:
          return output
        return output.placeholder
      assert not as_data
      return sources[i]

    vs = {}  # type: typing.Dict[str,object]
    if not callable(eval_str):
      vs.update(vars(tf_util))
      vs.update({"tf": tf})
    vs.update({"source": source, "self": self})
    vs.update(eval_locals or {})
    if callable(eval_str):
      x = eval_str(**vs)
    else:
      x = eval(eval_str, vs)
    assert sorted(used_sources) == list(range(len(sources))), (
      "not used sources: %r" % set(range(len(sources))).difference(used_sources))
    assert isinstance(x, tf.Tensor), "%r: eval %r did not return a tensor" % (self, eval_str)
    return x

  def _get_op(self, kind, eval_str=None, eval_locals=None):
    """
    :param str kind:
    :param str|callable eval_str:
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
  Also see :class:`ActivationLayer`, or :class:`CompareLayer`.

  The output type is defined as a broadcasted extension of all sources.
  You can overwrite it by (partially) specifying `out_type`.
  `out_type` can also be a generic Python function, returning a `Data` instance.
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
  Compares element-wise the tokens of all input sequences among themselves and/or with a specified given value.
  The comparisons are performed in a chain according to the order in which they are listed.

  Example::

      {"class": "compare", "from": ["i1", "i2"], "value": val, "kind": "less"}

  computes i1 < i2 < val and it is true only if the whole chain of operations is true.
  The final result is the logical "and" of all comparisons. Note that `value` is the last element to be compared to.

  A common example usage is the `end` layer in a rec subnetwork to specify the stopping criterion,
  e.g. the last generated token is equal to the end-of-sentence token::

      "output": {"class": "rec", "from": [], "unit": {
          .
          .
          .
          "end": {"class": "compare", "from": "output", "value": end_of_sentence_id}
      }, "target": "classes0"}

  """
  layer_class = "compare"

  def __init__(self, kind="equal", value=None, **kwargs):
    """
    :param str kind: which comparison operation to use, e.g. "equal", "greater", "less"
      or other supported TF comparison ops
    :param float|int|None value: if specified, will also compare to this
    """
    super(CompareLayer, self).__init__(**kwargs)
    assert len(self.sources) >= 1
    if value is None:
      assert len(self.sources) >= 2, "{} requires at least two elements to compare".format(self)
    op = getattr(tf, kind)  # e.g. tf.equal
    from returnn.tf.util.basic import opt_logical_and
    common_data = Data.get_common_data([s.output for s in self.sources])
    x = self.sources[0].output.copy_compatible_to(common_data).placeholder
    r_last = True
    for source in self.sources[1:]:
      x2 = source.output.copy_compatible_to(common_data).placeholder
      r_last = opt_logical_and(r_last, op(x, x2))
      x = x2
    if value is not None:
      r_last = opt_logical_and(r_last, op(x, value))
    self.output.placeholder = r_last

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
    elif out_type_.get("sparse", False):
      out_type_["dim"] = 2
    out_type_["dtype"] = "bool"
    out_type_["vocab"] = None
    out_type_["name"] = "%s_output" % kwargs["name"]
    if out_type:
      if isinstance(out_type, dict):
        out_type_.update(out_type)
      elif callable(out_type):
        out_type_ = out_type  # just overwrite
      else:
        raise TypeError("unexpected type of out_type %r" % (out_type,))
    return super(CompareLayer, cls).get_out_data_from_opts(n_out=n_out, out_type=out_type_, sources=sources, **kwargs)


class SwitchLayer(LayerBase):
  """
  Wrapper around ``tf.where()`` (or more generically :func:`TFUtil.where_bc`),
  or statically choose a single source if the condition is a callable (...)->bool.
  (``tf.cond`` is not useful here, as the sources would have been already constructed and computed.)

  This layer is also useful for applying any kind of generic masking to the frames.
  E.g. one could have a layer called "mask" computing a boolean mask for the values stored in another layer "input".
  Then use this layer with condition="mask", true_from="input", false_from=mask_value,
  to mask out all frames where the mask is false with the mask_value.

  See also :class:`CondLayer`.
  See also :class:`SeqLenMaskLayer` if you just want to mask using the sequence lengths.
  """
  layer_class = "switch"

  def __init__(self, condition, true_from, false_from, **kwargs):
    """
    :param LayerBase|bool condition: if callable, expected to be (...)->bool, and called in transform_config_dict
    :param LayerBase|float|int|None true_from:
    :param LayerBase|float|int|None false_from:
    """
    from returnn.tf.util.basic import where_bc
    super(SwitchLayer, self).__init__(**kwargs)
    assert not self.sources, "%s: you should use the explicit args" % self
    self.condition = condition
    self.true_from = true_from
    self.false_from = false_from

    def get_source_output(source, const_name):
      """
      :param LayerBase|float|int source:
      :param str const_name: if creating a new constant, use this name
      :rtype: Data
      """
      if isinstance(source, LayerBase):
        return source.output
      else:
        return Data.from_tensor(tf.constant(source, name=const_name))

    def get_source_allow_inf_in_output(source):
      """
      :param LayerBase|float|int source:
      :rtype: bool
      """
      if isinstance(source, LayerBase):
        return source.allow_inf_in_output
      else:
        return source in (float('inf'), float('-inf'))

    if isinstance(condition, bool):
      src = get_source_output(true_from if condition else false_from, const_name="const_output")
      self.output = src.copy("%s_output" % self.name)
      if get_source_allow_inf_in_output(true_from if condition else false_from):
        self.allow_inf_in_output = True
    else:
      assert isinstance(condition, LayerBase)
      assert condition.output.dtype == "bool"
      self.output.placeholder = where_bc(
        condition=condition.output.copy_compatible_to(self.output, check_dtype=False, check_sparse=False).placeholder,
        x=get_source_output(true_from, const_name="true_const_output").copy_compatible_to(self.output).placeholder,
        y=get_source_output(false_from, const_name="false_const_output").copy_compatible_to(self.output).placeholder)
      if get_source_allow_inf_in_output(true_from) or get_source_allow_inf_in_output(false_from):
        self.allow_inf_in_output = True

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param returnn.tf.network.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    d.setdefault("from", [])
    super(SwitchLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)

    def transform_source_name(source_name):
      """
      :param str source_name:
      :return: how the input layer should appear in the transformed config dict
      :rtype: LayerBase|float|int
      """
      if isinstance(source_name, str):
        return get_layer(source_name)
      return source_name

    if callable(d["condition"]):
      kwargs = d.copy()
      kwargs.update(dict(network=network, get_layer=get_layer))
      condition = d["condition"](**kwargs)
      assert isinstance(condition, bool)
      if condition:
        true_from = transform_source_name(d["true_from"])
        d.update(dict(condition=True, true_from=true_from, false_from=None))
      else:
        false_from = transform_source_name(d["false_from"])
        d.update(dict(condition=False, true_from=None, false_from=false_from))
    else:
      d["condition"] = get_layer(d["condition"])
      d["true_from"] = transform_source_name(d["true_from"])
      d["false_from"] = transform_source_name(d["false_from"])

  @classmethod
  def get_out_data_from_opts(cls, name, condition, true_from, false_from, **kwargs):
    """
    :param str name:
    :param LayerBase|bool condition:
    :param LayerBase|float|int|None true_from:
    :param LayerBase|float|int|None false_from:
    :rtype: Data
    """
    def get_source_template(source, source_name):
      """
      :param LayerBase|float|int source:
      :param str source_name:
      :rtype: Data
      """
      if isinstance(source, LayerBase):
        return source.output.copy_template(source_name)
      return Data.template_from_constant(source, name=source_name)

    if isinstance(condition, bool):
      return get_source_template(true_from if condition else false_from, source_name="%s_output" % name)
    true_data = get_source_template(true_from, source_name="%s_true" % name)
    false_data = get_source_template(false_from, source_name="%s_false" % name)
    out = Data.get_common_data([true_data, false_data, condition.output.copy_template()])
    out.dtype = true_data.dtype
    out.sparse = true_data.sparse
    if out.feature_dim_axis is not None:
      out.dim = out.batch_shape[out.feature_dim_axis]
    else:
      out.dim = true_data.dim
    out.vocab = true_data.vocab
    out.sanity_check()
    return out.copy(name="%s_output" % name)

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    if isinstance(self.condition, LayerBase):
      dep_layers = [self.condition, self.true_from, self.false_from]
    else:
      assert isinstance(self.condition, bool)
      dep_layers = [self.true_from] if self.condition else [self.false_from]
    # Filter out constants
    return [layer for layer in dep_layers if isinstance(layer, LayerBase)]


class CondLayer(LayerBase):
  """
  See also :class:`SwitchLayer`, which uses :func:`tf.where`.
  Here, we use `tf.cond` instead. I.e. the condition has to be a scalar bool,
  and only the corresponding true/false branch is computed.
  """
  layer_class = "cond"
  recurrent = True  # unclear

  def __init__(self, condition, true_layer, false_layer,
               _condition_network=None, _true_layer_network=None, _false_layer_network=None,
               **kwargs):
    """
    :param LayerBase|dict[str] condition:
    :param LayerBase|dict[str] true_layer:
    :param LayerBase|dict[str] false_layer:
    """
    import os
    from ..util.data import DimensionTag
    super(CondLayer, self).__init__(**kwargs)
    self._parent_scope = os.path.dirname(tf_compat.v1.get_variable_scope().name)
    self.condition_desc = condition
    self.condition_layer = self._make_layer("condition", self.condition_desc)
    self.true_layer_desc = true_layer
    self.true_layer = None  # type: typing.Optional[LayerBase]
    self.false_layer_desc = false_layer
    self.false_layer = None  # type: typing.Optional[LayerBase]
    assert self.condition_layer.output.batch_ndim == 0 and self.condition_layer.output.dtype == "bool"
    x, sizes = tf.cond(
      pred=self.condition_layer.output.placeholder,
      true_fn=self._true_fn,
      false_fn=self._false_fn)
    assert isinstance(x, tf.Tensor)
    self.output.placeholder = x
    assert len(sizes) == len(self.output.size_placeholder)
    for i, size in zip(sorted(self.output.size_placeholder.keys()), sizes):
      assert isinstance(size, tf.Tensor)
      assert size.shape.ndims == 1
      old_size = self.output.size_placeholder[i]
      old_tag = DimensionTag.get_tag_from_size_tensor(old_size)
      assert old_tag
      old_tag.set_tag_on_size_tensor(size, batch=self.output.batch, same_as_before=True)
      self.output.size_placeholder[i] = size

  def _cond_layer_return(self, layer):
    """
    :param LayerBase layer:
    :return: for tf.cond
    """
    out = layer.output
    out_template = self.output.copy_template()
    assert len(out.size_placeholder) == len(out_template.size_placeholder) <= 1  # not implemented yet...
    if len(out.size_placeholder) == len(out_template.size_placeholder) == 1:
      # Make sure that it is the same dynamic size, so that copy_compatible_to accepts it.
      out_template.size_placeholder[list(out_template.size_placeholder.keys())[0]] = (
        list(out.size_placeholder.values())[0])
    out = out.copy_compatible_to(out_template)
    return out.placeholder, [out.size_placeholder[i] for i in sorted(out_template.size_placeholder.keys())]

  def _true_fn(self):
    self.true_layer = self._make_layer("true_layer", self.true_layer_desc)
    return self._cond_layer_return(self.true_layer)

  def _false_fn(self):
    self.false_layer = self._make_layer("false_layer", self.false_layer_desc)
    return self._cond_layer_return(self.false_layer)

  def _make_layer(self, net_name, layer_desc):
    """
    :param str net_name:
    :param LayerBase|dict[str] layer_desc:
    :rtype: LayerBase
    """
    if isinstance(layer_desc, LayerBase):
      return layer_desc
    layer_desc = layer_desc.copy()
    layer_class = layer_desc.pop("class")
    assert issubclass(layer_class, LayerBase)
    extra_net = self.network.make_extra_net(
      prefix_name="extra._internal.%s(%s)" % (self.name, net_name),
      net_name="%s/%s(%s)" % (self.network.name, self.name, net_name),
      boundary=True)
    from returnn.tf.util.basic import reuse_name_scope
    with reuse_name_scope(self._parent_scope, absolute=True):
      # noinspection PyProtectedMember
      layer = extra_net._create_layer(
        name=self.name,  # use our name, such that we get the same name space
        layer_class=layer_class,
        **layer_desc)
    self.params.update(layer.params)
    return layer

  @classmethod
  def _transform_layer(cls, d, key, network, get_layer):
    """
    :param dict[str] d:
    :param str key: e.g. "true_layer"
    :param returnn.tf.network.TFNetwork network:
    :param (str)->LayerBase get_layer:
    :return: nothing, will replace inplace in ``d``
    """
    layer_desc = d[key]
    if isinstance(layer_desc, str):
      d[key] = get_layer(layer_desc)
      return
    assert isinstance(layer_desc, dict)
    name = d["_name"]
    extra_net = network.make_extra_net(
      prefix_name="extra._internal_template.%s(%s)" % (name, key), net_name="%s/%s(%s)" % (network.name, name, key),
      boundary=True, only_template=True)
    d["_%s_network" % key] = extra_net
    layer_desc = layer_desc.copy()
    class_name = layer_desc.pop("class")
    layer_class = get_layer_class(class_name)
    layer_desc["_network"] = extra_net
    layer_desc["_name"] = name
    layer_class.transform_config_dict(layer_desc, network=extra_net, get_layer=get_layer)
    layer_desc["class"] = layer_class  # will later pop again
    d[key] = layer_desc

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param (str)->LayerBase get_layer:
    """
    super(CondLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    cls._transform_layer(d, "condition", network=network, get_layer=get_layer)
    cls._transform_layer(d, "true_layer", network=network, get_layer=get_layer)
    cls._transform_layer(d, "false_layer", network=network, get_layer=get_layer)

  @classmethod
  def _get_out_data_from_layer(cls, layer, name, network):
    """
    :param LayerBase|dict[str] layer:
    :param str name:
    :param returnn.tf.network.TFNetwork network:
    :rtype: Data
    """
    if isinstance(layer, LayerBase):
      return layer.output
    assert isinstance(layer, dict)
    layer_desc = layer.copy()
    layer_class = layer_desc.pop("class")
    assert issubclass(layer_class, LayerBase)
    # noinspection PyProtectedMember
    layer_desc = network._create_layer_layer_desc(name=name, layer_desc=layer_desc)
    return layer_class.get_out_data_from_opts(**layer_desc)

  @classmethod
  def get_out_data_from_opts(cls, true_layer, false_layer, name, network, **kwargs):
    """
    :param LayerBase|dict[str] true_layer:
    :param LayerBase|dict[str] false_layer:
    :param str name:
    :param returnn.tf.network.TFNetwork network:
    :rtype: Data
    """
    # Only take one single out.
    # We allow that we get different size placeholders for each case.
    # So Data.get_common_data would not work.
    true_out = cls._get_out_data_from_layer(true_layer, name="%s/true" % name, network=network)
    return true_out

  def get_sub_layers(self):
    """
    :rtype: list[LayerBase]
    """
    layers = []
    for layer in [self.condition_layer, self.true_layer, self.false_layer]:
      if layer:
        layers.append(layer)
    return layers


class SearchSortedLayer(LayerBase):
  """
  Basically wraps :func:`tf.searchsorted`.

  Takes a tensor `sorted_sequence` that is sorted along one axis, and a tensor `values`.
  Will compute an output tensor with the same axes as `values`,
  where each entry is the index of the value within the sorted sequence.
  All (batch) axes of `sorted_sequence` except for the axis it is sorted along must be present in `values`.
  """
  layer_class = "search_sorted"
  recurrent = True  # do not shuffle the sorted_sequence

  def __init__(self, sorted_sequence, values, axis="T", side="left", **kwargs):
    """
    :param LayerBase sorted_sequence:
    :param LayerBase values: search values
    :param str axis: the axis along which `sorted_sequence` is sorted
    :param str side: "left" or "right".
      When one of the `values` exactly matches an element of the `sorted_sequence`,
      whether to choose the lower or higher index.
    """
    super(SearchSortedLayer, self).__init__(**kwargs)
    self.sorted_sequence = sorted_sequence  # e.g. [B,T]
    self.values = values  # e.g. [B,F]
    sorted_axis = sorted_sequence.output.get_axis_from_description(axis)  # = T
    side = side.lower()
    assert side in {"left", "right"}

    sorted_data = sorted_sequence.output  # e.g. [B,T]
    values_data = values.output  # e.g. [B,F]
    sorted_batch_axes = [ax for ax in range(sorted_data.batch_ndim) if ax != sorted_axis]  # = sorted B
    sorted_to_values_batch_axes = values_data.find_matching_dim_map(
      other=sorted_data, other_axes=sorted_batch_axes)  # sorted B -> values B
    values_batch_axes = [
      sorted_to_values_batch_axes[sorted_batch_ax] for sorted_batch_ax in sorted_batch_axes]  # = values B
    values_non_batch_axes = [ax for ax in range(values_data.batch_ndim) if ax not in values_batch_axes]  # = F
    assert len(values_non_batch_axes) == 1, 'not implemented'
    # move batch axes to front and align them between sorted_data and values
    transposed_sorted_data = sorted_data.copy_transpose(perm=sorted_batch_axes + [sorted_axis])  # [B,T]
    transposed_values_data = values_data.copy_transpose(perm=values_batch_axes + values_non_batch_axes)  # [B,F]
    x = transposed_sorted_data.placeholder  # [B,T]
    if transposed_sorted_data.is_axis_dynamic(axis=-1):
      from returnn.tf.util.basic import where_bc, sequence_mask
      seq_mask = transposed_sorted_data.get_sequence_mask_broadcast(axis=-1)
      x = where_bc(seq_mask, x, x.dtype.max)  # note: this is not correct if values contains x.dtype.max
    self.output.placeholder = tf.searchsorted(
      sorted_sequence=x, values=transposed_values_data.placeholder, side=side,
      out_type=self.output.dtype)

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    return super(SearchSortedLayer, self).get_dep_layers() + [self.sorted_sequence, self.values]

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param returnn.tf.network.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    d.setdefault("from", [])
    super(SearchSortedLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["sorted_sequence"] = get_layer(d["sorted_sequence"])
    d["values"] = get_layer(d["values"])

  @classmethod
  def get_out_data_from_opts(cls, sorted_sequence, values, axis, name, network, **kwargs):
    """
    :param LayerBase sorted_sequence:
    :param LayerBase values: search values
    :param str axis: the axis along which `sorted_sequence` is sorted
    :param str name:
    :param returnn.tf.network.TFNetwork network:
    :rtype: Data
    """
    sorted_data = sorted_sequence.output  # e.g. [B,T]
    values_data = values.output  # e.g. [B,F]
    sorted_axis = sorted_data.get_axis_from_description(axis)  # = T
    sorted_batch_axes = [ax for ax in range(sorted_data.batch_ndim) if ax != sorted_axis]  # = sorted B
    sorted_to_values_batch_axes = values_data.find_matching_dim_map(
      other=sorted_data, other_axes=sorted_batch_axes)  # sorted B -> values B
    values_batch_axes = [
      sorted_to_values_batch_axes[sorted_batch_ax] for sorted_batch_ax in sorted_batch_axes]  # = values B
    values_non_batch_axes = [ax for ax in range(values_data.batch_ndim) if ax not in values_batch_axes]  # = F
    if len(values_non_batch_axes) != 1:
      raise NotImplementedError(
        "%r %s: need exactly one axis in values %r that does not match any in sorted_sequence %r, but found: %r" % (
          cls, name, values_data, sorted_data, values_non_batch_axes))
    # move all batch axes to front, non batch axes to back
    output_data = values_data.copy("%s_output" % name).copy_transpose(perm=values_batch_axes + values_non_batch_axes)
    output_data.dtype = "int32"
    return output_data


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
    and also ``"data:0"`` to ``"data:<n-1>"`` for n inputs,
    for each input, in the subnetwork.
    The first input will also be simply available as ``"data:data"``/``"data"`.
  """

  layer_class = "subnetwork"
  recurrent = True  # we don't know. depends on the subnetwork.

  # noinspection PyShadowingNames
  def __init__(self, subnetwork,
               _subnet, _output,
               concat_sources=True, load_on_init=None,
               dropout=0, dropout_noise_shape=None,
               _parent_layer_cache=None, _from=None,
               **kwargs):
    """
    :param dict[str,dict] subnetwork: subnetwork as dict (JSON content). must have an "output" layer-
    :param bool concat_sources: if we concatenate all sources into one, like it is standard for most other layers
    :param str|dict[str]|None load_on_init: if provided, for parameter initialization,
      we will load the given model file. see :class:`CustomCheckpointLoader`.
    :param float dropout: will be applied if train_flag is set
    :param tuple|list|dict|None dropout_noise_shape:
    :param dict[str,LayerBase]|None _parent_layer_cache:
    :param returnn.tf.network.Subnetwork _subnet:
    :param LayerBase _output:
    """
    super(SubnetworkLayer, self).__init__(**kwargs)
    if _subnet.template:
      subnetwork = subnetwork.copy()
      _subnet = self.network.make_subnet(
        name=self.name,
        opts=dict(
          subnetwork=subnetwork,
          _from=_from, concat_sources=concat_sources,
          dropout=dropout, dropout_noise_shape=dropout_noise_shape,
          **kwargs))
      assert not _subnet.template, "%s, net %s, subnet %r" % (self, self.network, _subnet)
      self._update_for_rec_previous_layer(self._rec_previous_layer, subnetwork, _subnet.net)
      _subnet.construct_layer("output")
    net = _subnet.net
    if _parent_layer_cache:
      net.layers.update({"base:%s" % name: layer for (name, layer) in _parent_layer_cache.items()})
    self.subnetwork_ = _subnet
    self.subnetwork = net
    if self.network.eval_flag:
      self.subnetwork.maybe_construct_objective()
    self.output = net.get_default_output_layer().output
    self.update_params_from_subnet()
    self.update_rec_vars_outputs()
    self.load_on_init = load_on_init
    self.update_load_on_init()

  def update_params_from_subnet(self):
    """
    Update self.params.
    """
    # Very generic way to collect all created params.
    import re
    scope_name_prefix = self.get_absolute_name_scope_prefix()
    params = tf_compat.v1.get_collection(
      tf_compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=re.escape(scope_name_prefix))
    for p in params:
      assert p.name.startswith(scope_name_prefix) and p.name.endswith(":0")
      param_name = p.name[len(scope_name_prefix):-len(":0")]
      self.params[param_name] = p

      # Sublayers do not know whether the RecLayer is trainable.
      # If it is not, we need to mark all defined parameters as untrainable.
      if not self.trainable:
        trainable_collection_ref = p.graph.get_collection_ref(tf_compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        if p in trainable_collection_ref:
          trainable_collection_ref.remove(p)

  def update_rec_vars_outputs(self):
    """
    Update self.rec_vars_outputs.
    """
    for layer in self.subnetwork.layers.values():
      self.rec_vars_outputs.update({"%s/%s" % (layer.name, k): v for (k, v) in layer.rec_vars_outputs.items()})

  def update_load_on_init(self):
    """
    Handle load_on_init.
    """
    from returnn.tf.network import CustomCheckpointLoader
    load_on_init = self.load_on_init
    if load_on_init:
      if callable(load_on_init):
        load_on_init = load_on_init()
      print("loading initial weights from", load_on_init, file=log.v2)
      self_prefix = self.get_absolute_name_scope_prefix()  # with "/" at end
      opts = dict(saveable_params=list(self.params.values()), params_prefix=self_prefix, network=self.subnetwork)
      if isinstance(load_on_init, str):
        opts["filename"] = load_on_init
      elif isinstance(load_on_init, dict):
        opts.update(load_on_init)
      else:
        raise TypeError("%s: invalid load_on_init %r" % (self, load_on_init))
      loader = CustomCheckpointLoader(**opts)
      loader.set_as_custom_init()

  @classmethod
  def _update_for_rec_previous_layer(cls, rec_previous_layer, subnet_dict, subnet):
    """
    :param LayerBase|None rec_previous_layer:
    :param dict[str] subnet_dict:
    :param returnn.tf.network.TFNetwork subnet:
    """
    if not rec_previous_layer:
      return
    # Make some rec_previous_layer for the subnet layers.
    for layer_name in list(subnet_dict.keys()):
      # The actual layer is not so important.
      # In some cases (e.g. RnnCellLayer), we just want rec_vars_outputs.
      dummy_rec_previous_layer = InternalLayer(
        name=layer_name, network=subnet,
        sources=[rec_previous_layer],
        output=Data(
          name="dummy_rec_previous_layer(%s)" % layer_name, dim=1, shape=(1,),
          batch=rec_previous_layer.output.batch, beam=rec_previous_layer.output.beam))
      dummy_rec_previous_layer.rec_vars_outputs.update({
        key[len(layer_name + "/"):]: value
        for (key, value) in rec_previous_layer.rec_vars_outputs.items()
        if key.startswith(layer_name + "/")})
      if dummy_rec_previous_layer.rec_vars_outputs:
        subnet_dict[layer_name] = subnet_dict[layer_name].copy()
        subnet_dict[layer_name]["rec_previous_layer"] = dummy_rec_previous_layer

  # noinspection PyShadowingNames
  @classmethod
  def get_out_data_from_opts(cls, n_out=NotSpecified, out_type=None, **kwargs):
    """
    :param int|None|NotSpecified n_out:
    :param dict[str]|None out_type:
    :rtype: Data
    """
    output = kwargs["_output"]
    assert isinstance(output, LayerBase)
    return output.output.copy()

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param get_layer:
    """
    # Note that in the common cases, this transform_config_dict is actually *not* called,
    # because net.construct_layer directly gets the Subnetwork via cls_get_sub_network.
    # This is only called when we explicitly construct the layer itself.
    name = d["_name"]
    # Make sure the Subnetwork instance is created.
    # This works both in case of template construction or for real construction.
    # See Subnetwork for determining whether this is supposed to be a template or real.
    subnet = cls.cls_get_sub_network(network=network, name=name, layer_desc=d)
    cls._update_for_rec_previous_layer(d.get("rec_previous_layer"), d["subnetwork"], subnet.net)
    d["_subnet"] = subnet
    # In case of non-template construction, this will trigger the non-template construction of our "output" sublayer.
    d["_output"] = subnet.construct_layer("output", parent_get_layer=get_layer)
    d["_from"] = d.get("from", "data")  # cache this
    d["from"] = []  # disable now. we should get them in the template construction when needed
    super(SubnetworkLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)

  @classmethod
  def cls_get_sub_network(cls, name, network, layer_desc):
    """
    :param str name:
    :param returnn.tf.network.TFNetwork network:
    :param dict[str] layer_desc:
    :rtype: returnn.tf.network.Subnetwork|None
    """
    return network.make_subnet(name, opts=layer_desc)

  def get_sub_layer(self, layer_name):
    """
    :param str layer_name: name of the sub_layer (right part of '/' separated path)
    :return: the sub_layer addressed in layer_name or None if no sub_layer exists
    :rtype: LayerBase|None
    """
    from returnn.tf.network import LayerNotFound
    try:
      return self.subnetwork.get_layer(layer_name)
    except LayerNotFound:
      return None

  def get_sub_networks(self):
    """
    :rtype: list[returnn.tf.network.TFNetwork]
    """
    return [self.subnetwork]

  def get_sub_layers(self):
    """
    :rtype: list[LayerBase]
    """
    return self.subnetwork.get_all_layers_shallow()

  def get_dep_layers(self):
    """
    :return: list of layers this layer depends on.
      normally this is just self.sources but e.g. the attention layer in addition has a base, etc.
    :rtype: list[LayerBase]
    """
    return super(SubnetworkLayer, self).get_dep_layers() + [self.subnetwork.layers["output"]]

  def get_last_hidden_state(self, key):
    """
    :param int|str|None key: also the special key "*"
    :rtype: tf.Tensor|None
    """
    h = self.subnetwork.get_default_output_layer().get_last_hidden_state(key=key)
    if h is not None:
      return h
    return super(SubnetworkLayer, self).get_last_hidden_state(key=key)

  @classmethod
  def get_rec_initial_extra_outputs(cls, batch_dim, rec_layer, **kwargs):
    """
    :param tf.Tensor batch_dim: for this layer, might be with beam
    :param returnn.tf.layers.rec.RecLayer rec_layer:
    :rtype: dict[str,tf.Tensor]
    """
    from .rec import _TemplateLayer
    from returnn.tf.network import Subnetwork
    subnet_ = kwargs["_subnet"]
    assert isinstance(subnet_, Subnetwork)
    subnet = subnet_.net
    extra_outputs = {}
    for layer_name, sub_layer in subnet.layers.items():
      assert isinstance(sub_layer, _TemplateLayer)
      cl = sub_layer.layer_class_type
      layer_desc = sub_layer.kwargs
      assert issubclass(cl, LayerBase)
      with cl.cls_layer_scope(layer_name):
        d = cl.get_rec_initial_extra_outputs(
          batch_dim=batch_dim, rec_layer=rec_layer, **layer_desc)
        for key, value in d.items():
          extra_outputs["%s/%s" % (layer_name, key)] = value
    return extra_outputs

  @classmethod
  def get_rec_initial_extra_outputs_shape_invariants(cls, **kwargs):
    """
    :return: optional shapes for the tensors by get_rec_initial_extra_outputs
    :rtype: dict[str,tf.TensorShape]
    """
    # Very similar to get_rec_initial_extra_outputs.
    from .rec import _TemplateLayer
    from returnn.tf.network import Subnetwork
    subnet_ = kwargs["_subnet"]
    assert isinstance(subnet_, Subnetwork)
    subnet = subnet_.net
    shape_invariants = {}
    for layer_name, sub_layer in subnet.layers.items():
      assert isinstance(sub_layer, _TemplateLayer)
      cl = sub_layer.layer_class_type
      layer_desc = sub_layer.kwargs
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
    from returnn.tf.util.basic import get_initializer, expand_dims_unbroadcast
    initializer = get_initializer(init, seed=self.network.random.randint(2 ** 31), eval_local_ns={"layer": self})
    with self.var_creation_scope():
      var = self.add_param(tf_compat.v1.get_variable(
        name=self.name, shape=shape, dtype=dtype,
        initializer=initializer, trainable=trainable
      ))
      out = var
      if add_batch_axis:
        # Unbroadcast to not confuse some other layers
        batch_dim = self.output.get_batch_dim()
        out = expand_dims_unbroadcast(out, axis=self.output.batch_dim_axis, dim=batch_dim)
      if add_time_axis:
        out = tf.expand_dims(out, axis=self.output.time_dim_axis)
    self.output.placeholder = out

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param returnn.tf.network.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    # Overwrite default behavior for default sources.
    # Here: none by default.
    d.setdefault("from", [])
    super(VariableLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)

  @classmethod
  def get_out_data_from_opts(cls, name, network,
                             shape, dtype="float32", add_batch_axis=True, add_time_axis=False, **kwargs):
    """
    :param str name:
    :param returnn.tf.network.TFNetwork network:
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
      batch_dim_axis=batch_dim_axis,
      batch=network.get_global_batch_info() if add_batch_axis else None,
      time_dim_axis=time_dim_axis)


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
    from returnn.tf.util.basic import CustomUpdateExpAverage
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


class LossLayer(LayerBase):
  """
  This layers wraps a :class:`Loss` calculation as a layer.
  I.e. the loss will be calculated and returned by the layer.
  But this loss will not be used as a loss by the updater.
  If you want to use it as a loss, you can use the :class:`AsIsLoss`,
  i.e. write ``"loss": "as_is"``.

  Note that the loss options for the wrapped loss need to be provided via ``loss_opts_``,
  and it does not apply any reduce function.

  .. note::

    The ``LossLayer`` might be deprecated in the future in favor of implementing the losses as actual layers.

    If you want to define a loss inside the network, it is recommended to define it explicitly.
    An example could be:

    ``"se_loss": {"class": "eval", "eval": "(source(0) - source(1)) ** 2", "from": ["output", "data:classes"]}``

    Followed by an e.g. mean reduce if needed:

    ``"mse_loss": {"class": "reduce", "mode": "mean", "axis": "F", "from": "se_loss"}``


  """
  layer_class = "loss"
  recurrent = True  # we don't know. depends on the loss

  def __init__(self, loss_, target_=None, use_error=False, **kwargs):
    """
    ``loss_`` and related params have the postfix ``_`` to distinguish them
    from the loss options, which are used by the network and updater for training.
    Some of these (e.g. ``loss_opts_``) are handled in :func:`transform_config_dict`.

    :param Loss loss_:
    :param LayerBase|None target_:
    :param bool use_error: whether to output the loss error instead of the loss value
    """
    super(LossLayer, self).__init__(**kwargs)
    assert len(self.sources) == 1
    source = self.sources[0]
    self.loss_ = loss_
    self.target_ = target_
    self.loss_.reduce_func = tf_util.identity
    self.loss_.use_flatten_frames = False
    self.loss_.init(
      output=source.output,
      output_with_activation=source.output_before_activation,
      target=target_.output if target_ else None,
      layer=source)
    self.loss_value = self._make_output_value(self.loss_.get_value())
    self.error_value = self._make_output_value(self.loss_.get_error())
    if use_error:
      assert self.error_value is not None
      self.output.placeholder = self.error_value
    else:
      assert self.loss_value is not None
      self.output.placeholder = self.loss_value

  def _make_output_value(self, value):
    """
    :param tf.Tensor|None value: either loss value or error value,
      as it comes out from Loss.get_value or Loss.get_error.
      [B*T|T*B|B] just like source
    :return: shape as self.output, e.g. [B,T] or [T,B] ...
    :rtype: tf.Tensor|None
    """
    if value is None:
      return None
    # Note: This code is not complete. There are other cases.
    # E.g. MseLoss leaves more dimensions... In that case, we should reduce_sum on them.
    assert value.shape.ndims == 1  # [B*T|T*B|B] just like source
    source = self.sources[0]
    shape = tf_util.get_shape(source.output.placeholder)
    if source.output.have_feature_axis():
      del shape[source.output.feature_dim_axis]
    return tf.reshape(value, shape)

  def get_sub_layer(self, layer_name):
    """
    :param str layer_name: sub layer name
    :rtype: LayerBase|None
    """
    sub_layer_name = "%s/%s" % (self.name, layer_name)
    out = self.output.copy_template(name="%s_output" % sub_layer_name)
    if layer_name == "loss":
      assert self.loss_value is not None, "%s: loss not defined" % self
      out.placeholder = self.loss_value
      return InternalLayer(name=sub_layer_name, network=self.network, output=out)
    elif layer_name == "error":
      assert self.error_value is not None, "%s: error not defined" % self
      out.placeholder = self.error_value
      return InternalLayer(name=sub_layer_name, network=self.network, output=out)
    else:
      raise Exception("%s: invalid sub layer %r" % (self, layer_name))

  @classmethod
  def get_sub_layer_out_data_from_opts(cls, layer_name, parent_layer_kwargs):
    """
    :param str layer_name: sub layer name
    :param dict[str] parent_layer_kwargs:
    :rtype: (Data, TFNetwork, type)|None
    """
    if layer_name not in ["loss", "error"]:
      return None
    return cls.get_out_data_from_opts(**parent_layer_kwargs), parent_layer_kwargs["network"], InternalLayer  # same type

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    deps = super(LossLayer, self).get_dep_layers()
    if self.target_:
      deps.append(self.target_)
    return deps

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param get_layer:
    """
    super(LossLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    assert "loss_" in d
    if "target_" not in d:
      target = get_loss_class(d["loss_"]).get_default_target(network.extern_data)
      if target:
        d["target_"] = target
    if d.get("target_", None):
      target = d["target_"]
      if target.startswith("layer:"):
        target = get_layer(target[len("layer:"):])
      else:
        target = get_layer("data:%s" % target)
      d["target_"] = target
    d["loss_"] = cls._make_loss(
      class_name=d.pop("loss_", None), opts=d.pop("loss_opts_", {}),
      network=network, get_layer=get_layer, always_make=True)

  @classmethod
  def get_out_data_from_opts(cls, name, sources, target_=None, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param LayerBase|None target_:
    :rtype: Data
    """
    assert len(sources) == 1
    out = sources[0].output.copy_template(name="%s_output" % name)
    if out.have_feature_axis():
      out = out.copy_template_excluding_axis(out.feature_dim_axis)
    out.dtype = "float32"
    if target_:
      out.beam = SearchBeam.get_combined_beam(out.beam, target_.output.beam)
      out.available_for_inference = out.available_for_inference & target_.output.available_for_inference
    return out


class ForcedAlignmentLayer(_ConcatInputLayer):
  """
  Calculates a forced alignment, via Viterbi algorithm.
  """
  layer_class = "forced_align"

  def __init__(self, align_target, topology, input_type, **kwargs):
    """
    :param LayerBase align_target:
    :param str topology: e.g. "ctc" or "rna" (RNA is CTC without label loop)
    :param str input_type: "log_prob" or "prob"
    """
    from returnn.tf.native_op import get_ctc_fsa_fast_bw, fast_viterbi
    super(ForcedAlignmentLayer, self).__init__(**kwargs)
    self.align_target = align_target
    assert topology in ["ctc", "rna"], "%s no other topology implemented" % self
    logits_data = self.input_data.copy_as_time_major()
    logits = logits_data.placeholder
    assert logits.get_shape().ndims == 3 and logits.get_shape().dims[-1].value == logits_data.dim
    assert align_target.output.shape == (None,) and align_target.output.dim == logits_data.dim - 1
    if input_type == "log_prob":
      pass  # ok
    elif input_type == "prob":
      logits = tf_util.safe_log(logits)
    else:
      raise ValueError("%s: invalid input_type %r" % (self, input_type))

    edges, weights, start_end_states = get_ctc_fsa_fast_bw(
      targets=align_target.output.get_placeholder_as_batch_major(),
      seq_lens=align_target.output.get_sequence_lengths(),
      blank_idx=logits_data.dim - 1,
      label_loop=topology == "ctc")
    alignment, scores = fast_viterbi(
      am_scores=logits, am_seq_len=logits_data.get_sequence_lengths(),
      edges=edges, weights=weights, start_end_states=start_end_states)
    self.alignment = alignment
    self.scores = scores
    self.output.placeholder = alignment

  @classmethod
  def get_sub_layer_out_data_from_opts(cls, layer_name, parent_layer_kwargs):
    """
    :param str layer_name: sub layer name
    :param dict[str] parent_layer_kwargs:
    :rtype: (Data, TFNetwork, type)|None
    """
    if layer_name == "scores":
      return Data(name="align_scores", shape=(), dtype="float32"), parent_layer_kwargs["network"], InternalLayer
    return None

  def get_sub_layer(self, layer_name):
    """
    :param str layer_name:
    :rtype: LayerBase|None
    """
    if layer_name == "scores":
      return InternalLayer(
        name="%s_scores" % self.name, network=self.network,
        output=Data(name="%s_scores_output" % self.name, shape=(), dtype="float32", placeholder=self.scores))
    return None

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    return super(ForcedAlignmentLayer, self).get_dep_layers() + [self.align_target]

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param get_layer:
    """
    super(ForcedAlignmentLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["align_target"] = get_layer(d["align_target"])

  @classmethod
  def get_out_data_from_opts(cls, name, sources, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    src = get_concat_sources_data_template(sources, name="%s_output" % name).copy_as_time_major()
    opts = src.get_kwargs(include_special_axes=False)
    opts["dim_tags"] = (src.get_time_dim_tag(), src.dim_tags[src.batch_dim_axis])
    opts["dtype"] = "int32"
    opts["sparse"] = True
    return Data(**opts)


class FastBaumWelchLayer(_ConcatInputLayer):
  """
  Calls :func:`fast_baum_welch` or :func:`fast_baum_welch_by_sprint_automata`.
  We expect that our input are +log scores, e.g. use log-softmax.
  """
  layer_class = "fast_bw"
  recurrent = True

  def __init__(self, align_target, align_target_key=None,
               ctc_opts=None, sprint_opts=None,
               input_type="log_prob",
               tdp_scale=1.0, am_scale=1.0, min_prob=0.0,
               staircase_seq_len_source=None,
               **kwargs):
    """
    :param str align_target: e.g. "sprint" or "staircase"
    :param str|None align_target_key: e.g. "classes", used for e.g. align_target "ctc"
    :param dict[str] ctc_opts: used for align_target "ctc"
    :param dict[str] sprint_opts: used for Sprint (RASR) for align_target "sprint"
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
          from returnn.tf.util.basic import swapaxes
          am_scores = swapaxes(am_scores, 0, 1)
      else:
        from returnn.tf.util.basic import safe_log
        am_scores = -safe_log(data.placeholder)
    else:
      raise Exception("%s: invalid input_type %r" % (self, input_type))
    if min_prob > 0:
      am_scores = tf.minimum(am_scores, -numpy.log(min_prob))  # in -log space
    if am_scale != 1:
      am_scores *= am_scale
    if align_target == "ctc":
      # See :func:`TFNativeOp.ctc_loss` for reference.
      if ctc_opts is None:
        ctc_opts = {}
      targets_data = self._get_target_value(align_target_key)
      ctc_opts = ctc_opts.copy()
      ctc_opts.setdefault("blank_idx", targets_data.dim)
      assert targets_data.dim + 1 == data.dim
      ctc_merge_repeated = ctc_opts.pop("ctc_merge_repeated", NotSpecified)  # label_loop === ctc_merge_repeated
      if ctc_merge_repeated is not NotSpecified:
        assert "label_loop" not in ctc_opts
        ctc_opts["label_loop"] = ctc_merge_repeated
      from returnn.tf.native_op import get_ctc_fsa_fast_bw, fast_baum_welch
      edges, weights, start_end_states = get_ctc_fsa_fast_bw(
        targets=targets_data.get_placeholder_as_batch_major(),
        seq_lens=targets_data.get_sequence_lengths(),
        **ctc_opts)
      from returnn.tf.util.basic import sequence_mask_time_major
      seq_mask = sequence_mask_time_major(data.get_sequence_lengths())
      fwdbwd, obs_scores = fast_baum_welch(
        am_scores=am_scores, float_idx=seq_mask,
        edges=edges, weights=weights, start_end_states=start_end_states)
    elif align_target == "sprint":
      from returnn.tf.util.basic import sequence_mask_time_major
      seq_mask = sequence_mask_time_major(data.get_sequence_lengths())
      from returnn.tf.native_op import fast_baum_welch_by_sprint_automata
      seq_tags = self.network.get_seq_tags()
      fwdbwd, obs_scores = fast_baum_welch_by_sprint_automata(
        sprint_opts=sprint_opts,
        tdp_scale=tdp_scale,
        am_scores=am_scores,
        float_idx=seq_mask,
        tags=seq_tags)
    elif align_target == "staircase":
      from returnn.tf.native_op import fast_baum_welch_staircase
      fwdbwd, obs_scores = fast_baum_welch_staircase(
        am_scores=am_scores, seq_lens=staircase_seq_len_source.output.get_sequence_lengths())
    else:
      raise Exception("%s: invalid align_target %r" % (self, align_target))
    loss = obs_scores[0]  # [batch]
    self.output_loss = loss
    bw = tf.exp(-fwdbwd)
    self.output.placeholder = bw
    self.output.size_placeholder = data.size_placeholder.copy()

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
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
    from returnn.tf.util.basic import MetaLosses
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
    :param returnn.tf.network.TFNetwork network:
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
      dummy_var = self.add_param(tf_compat.v1.get_variable(name="dummy", shape=(), dtype=tf.float32))
    from returnn.tf.util.basic import MetaLosses
    self.output.placeholder = MetaLosses.tikhonov_regularized(
      x=self.input_data.placeholder, dummy=dummy_var,
      loss_name=self.get_absolute_name_scope_prefix() + "tikhonov_reg",
      loss_source=self,
      loss_scale=meta_loss_scale)
    self.output.size_placeholder = self.input_data.size_placeholder.copy()


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
    from returnn.tf.util.basic import flatten_with_seq_len_mask
    import numpy
    source = self.sources[0]
    output = source.output
    target = source._get_target_value()  # noqa
    assert target.sparse
    assert source.output_before_activation.act_func is tf.nn.softmax
    output_seq_lens = output.size_placeholder[0]
    output_before_softmax_flat = flatten_with_seq_len_mask(
      source.output_before_activation.x, output_seq_lens, time_major=output.is_time_major)
    target_seq_lens = target.size_placeholder[0]
    target_flat = flatten_with_seq_len_mask(target.placeholder, target_seq_lens, time_major=target.is_time_major)
    target_flat.set_shape(tf.TensorShape([None]))
    loss_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_before_softmax_flat, labels=target_flat)
    flat_last_dim = output_before_softmax_flat.get_shape().ndims - 1
    assert flat_last_dim == 1
    output_flat = flatten_with_seq_len_mask(output.placeholder, output_seq_lens, time_major=output.is_time_major)
    output_flat_argmax = tf.cast(tf.argmax(output_before_softmax_flat, axis=flat_last_dim), "int32")
    frame_error = tf.not_equal(output_flat_argmax, target_flat)
    # target_flat is shape (time,) -> index.
    target_flat_exp = tf.stack([tf.range(tf.shape(target_flat)[0], dtype=tf.int32), target_flat], axis=1)
    true_label_prob = tf.gather_nd(output_flat, target_flat_exp)
    true_label_prob.set_shape(tf.TensorShape([None]))
    true_label_prob_i32 = tf.clip_by_value(
      tf.cast(tf.round(true_label_prob * histogram_num_bins), tf.int32), 0, histogram_num_bins - 1)
    true_label_prob_histogram = tf.stack(
      [tf.equal(true_label_prob_i32, i) for i in range(histogram_num_bins)], axis=1)
    true_label_prob_histogram.set_shape(tf.TensorShape([None, histogram_num_bins]))

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
    accumulated_seq_len = tf_compat.v1.assign_add(accumulated_seq_len, tf.cast(seq_len, tf.int64))
    accumulated_seq_len_sil = tf_compat.v1.assign_add(accumulated_seq_len_sil, tf.cast(seq_len_sil, tf.int64))
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
        acc_v = tf_compat.v1.assign_add(acc_v, tf.reduce_sum(tf.cast(v, acc_dtype), axis=0))
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

  def __init__(self, summarize=99, extra_print_args=(), **kwargs):
    """
    :param int|None summarize: passed to :func:`py_print`
    :param list|tuple extra_print_args:
    """
    super(PrintLayer, self).__init__(**kwargs)
    from returnn.tf.util.basic import py_print
    with tf.name_scope("print_layer"):
      source = self.sources[0]
      print_args = [self.__class__.__name__, self.name, source.output.placeholder]
      print_args.extend(extra_print_args)
      output = py_print(source.output.placeholder, print_args, summarize=summarize)
      if not tf_util.get_current_control_flow_context():  # Only possible to globally register if not in cond/loop.
        self.network.register_post_control_dependencies([output])
      with tf.control_dependencies([output]):
        self.output.placeholder = tf.identity(source.output.placeholder)
      self.output.size_placeholder = source.output.size_placeholder.copy()
      if source.allow_inf_in_output:
        self.allow_inf_in_output = True

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

  The HDF will be written to disk under the specified filename, if there was no error,
  by default at graph reset, via :func:`TFNetwork.register_graph_reset_callback`.
  Or after the dataset iteration run loop, with dump_per_run,
  via :func:`TFNetwork.register_run_finished_callback`.

  Common usage would be to add this to your network with "is_output_layer": True,
  such that you don't need to make other layers depend on it.

  It currently uses :class:`SimpleHDFWriter` internally.
  """
  layer_class = "hdf_dump"

  def __init__(self, filename, extra=None, dump_whole_batches=False, labels=None,
               extend_existing_file=False, dump_per_run=False,
               **kwargs):
    """
    :param str|(()->str) filename:
    :param None|dict[str,LayerBase] extra:
    :param bool dump_whole_batches: dumps the whole batch as a single sequence into the HDF
    :param list[str]|None labels:
    :param bool extend_existing_file: True also means we expect that it exists
    :param bool dump_per_run: write via :func:`TFNetwork.register_run_finished_callback`
    """
    super(HDFDumpLayer, self).__init__(**kwargs)
    assert len(self.sources) == 1
    assert self.sources[0].output.have_time_axis()
    self.output = self.sources[0].output.copy("%s_output" % self.name)
    data = self.output.copy_as_batch_spatial_major()  # need batch-major for SimpleHDFWriter

    if labels is not None:
      assert len(labels) == data.dim
    elif data.vocab:
      labels = data.vocab.labels

    from returnn.datasets.hdf import SimpleHDFWriter
    import numpy
    import sys
    self.filename = filename
    if extra is None:
      extra = {}
    extra = {key: layer.output for (key, layer) in extra.items()}
    extra = {
      key: output.copy_add_batch_dim(0, batch=data.batch)
      if not output.have_batch_axis() else output
      for (key, output) in extra.items()}
    extra = {key: output.copy_as_batch_spatial_major() for (key, output) in extra.items()}
    self.extra = extra  # type: typing.Dict[str,Data]
    self.dump_whole_batches = dump_whole_batches
    self.num_seqs_written = 0
    ndim = data.ndim
    if dump_whole_batches:
      ndim = data.ndim - len(data.size_placeholder) + 1
    ndim_without_features = ndim - (0 if data.sparse or data.feature_dim_axis is None else 1)
    # Open the HDF writer lazily. We only want to start writing (or overwriting) once we really start.
    # E.g. when just building the graph for importing a model,
    # it should not touch (delete/overwrite) an existing file!
    self.hdf_writer = None

    def py_write(data_np, tags, sizes, *extras):
      """
      :param numpy.ndarray data_np: (B,...), this is data.placeholder
      :param list[bytes] tags:
      :param numpy.ndarray sizes: shape [num_sizes,size_placeholder[i]]
      :param numpy.ndarray extras:
      :return: unused
      """
      # noinspection PyBroadException
      try:
        if not self.hdf_writer:
          filename_ = self.filename
          if callable(filename_):
            assert dump_per_run  # does not make sense otherwise
            filename_ = filename_(**self.network.get_run_opts())
          assert isinstance(filename_, str)
          self.hdf_writer = SimpleHDFWriter(
            filename=filename_,
            extend_existing_file=extend_existing_file,
            dim=data.dim, ndim=ndim,
            labels=labels,
            extra_type={
              key: (
                value.dim,
                1 if self.dump_whole_batches else min(value.ndim - len(value.size_placeholder) + 1, 2),
                value.dtype)
              for (key, value) in self.extra.items()
            })
          if dump_per_run:
            self.network.register_run_finished_callback(self._maybe_close)
          else:
            self.network.register_graph_reset_callback(self._maybe_close)

        n_batch = data_np.shape[0]
        assert sizes.shape == (len(data.size_placeholder), n_batch) if data.size_placeholder else (0,)
        assert len(sizes) == len(data.size_placeholder)
        seq_lens = {i: size for (i, size) in zip(sorted(data.size_placeholder.keys()), sizes)}
        # There may be axes with a fixed length other than the batch and feature axes.
        # These have the indices 0, ..., (ndim-1), as the batch dimension is skipped.
        for dim in range(ndim_without_features):
          if dim not in seq_lens:
            seq_lens[dim] = numpy.array([data_np.shape[dim + 1]] * n_batch, dtype="int32")
        assert len(seq_lens) == ndim_without_features
        assert len(extras) == len(self.extra) * 2  # value + sizes
        # noinspection PyShadowingNames
        extra = {}
        for i, (key, extra_data) in enumerate(sorted(self.extra.items())):
          assert isinstance(key, str)
          assert isinstance(extra_data, Data)
          assert key not in extra
          assert "%s_seq_lens" % key not in extra
          value, value_sizes = extras[i], extras[len(self.extra) + i]
          assert isinstance(value, numpy.ndarray)
          assert extra_data.batch_dim_axis == 0 and value.ndim == extra_data.ndim + 1  # including batch-dim
          value_batch = value.shape[0]  # we allow a different beam size / different batch size
          value_seq_lens = {i: size for (i, size) in zip(sorted(extra_data.size_placeholder.keys()), value_sizes)}
          for dim in range(extra_data.ndim):
            if dim not in value_seq_lens:
              assert dim > max(list(value_seq_lens.keys()) + [-1])  # assume all dynamic axes came first
              value_seq_lens[dim] = numpy.array([value.shape[dim + 1]] * value_batch, dtype="int32")
          if value_seq_lens:
            batch_value_seq_sizes = numpy.zeros((value_batch, len(value_seq_lens)), dtype="int32")
            for i_, (axis, size) in enumerate(sorted(value_seq_lens.items())):
              assert isinstance(size, numpy.ndarray) and size.shape == (value_batch,)
              batch_value_seq_sizes[:, i_] = size
            extra["%s_seq_lens" % key] = batch_value_seq_sizes
          elif self.dump_whole_batches:
            # Have sth in there, such that we know explicitly the batch-dim.
            extra["%s_seq_lens" % key] = numpy.zeros((value_batch, 1), dtype="int32")
          if self.dump_whole_batches:
            value = numpy.reshape(value, (-1,))  # flatten all
          elif extra_data.ndim - 1 in extra_data.size_placeholder:  # last axis dynamic?
            value = numpy.reshape(value, (value_batch, -1))  # flatten all
          elif value.ndim > 2:
            value = numpy.reshape(value, (value_batch, -1, value.shape[-1]))  # flatten all except last
          extra[key] = value
        if seq_lens:
          batch_seq_sizes = numpy.zeros((n_batch, len(seq_lens)), dtype="int32")
          for i, (axis, size) in enumerate(sorted(seq_lens.items())):
            batch_seq_sizes[:, i] = size
          extra["seq_sizes"] = batch_seq_sizes
        elif self.dump_whole_batches:
          # Have sth in there, such that we know explicitly the batch-dim.
          extra["seq_sizes"] = numpy.zeros((n_batch, 1), dtype="int32")
        if self.dump_whole_batches:
          # The batch dim itself becomes another axis to dump.
          # We also want to store the individual seq lens.
          assert sorted(seq_lens.keys()) == list(range(len(seq_lens)))
          flat_len = numpy.prod(data_np.shape[:len(seq_lens) + 1])
          data_np = data_np.reshape((1, flat_len) + data_np.shape[len(seq_lens) + 1:])
          seq_lens = {0: numpy.array([flat_len], dtype="int32")}
          extra["seq_tags"] = numpy.array(tags)
          for key, value in list(extra.items()):
            value = numpy.expand_dims(value, 0)  # add outer dim
            extra[key] = value
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

    tf_write = tf_compat.v1.py_func(
      py_write,
      [data.placeholder,
       self.network.get_seq_tags(beam=data.beam),
       tf.convert_to_tensor([size for (i, size) in sorted(data.size_placeholder.items())])] +
      [value.placeholder for (key, value) in sorted(extra.items())] +
      [tf.convert_to_tensor([size for (i, size) in sorted(value.size_placeholder.items())])
       for (key, value) in sorted(extra.items())],
      tf.int64,  # return value is ignored
      stateful=True)

    import returnn.util.basic
    # The check covers multi-GPU and maybe dry-run.
    # Note that for multi-GPU, horovod_dataset_distribution "shard" would need some other treatment,
    # which is not yet implemented!
    # In that case, we could gather all the batches.
    if returnn.util.basic.should_write_to_disk(config=self.network.get_config()):
      self.network.register_post_control_dependencies([tf_write])

  def _maybe_close(self):
    if self.hdf_writer:
      print("HDFDumpLayer, wrote %i seqs to file %r." % (self.num_seqs_written, self.hdf_writer.filename))
      self.hdf_writer.close()
      self.hdf_writer = None

  @classmethod
  def get_out_data_from_opts(cls, name, sources, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    assert len(sources) == 1, "%s %r: expects exactly one source, but got: %r" % (cls.__name__, name, sources)
    return sources[0].output.copy(name="%s_output" % name)

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param returnn.tf.network.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    super(HDFDumpLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    if d.get("extra", None):
      extra = d["extra"]
      assert isinstance(extra, dict), "invalid in %r" % d
      d["extra"] = {key: get_layer(value) for (key, value) in extra.items()}


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
    :param returnn.tf.network.TFNetwork network:
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


# ------------------------------------------------------------------------------


class CrossEntropyLoss(Loss):
  """
  Cross-Entropy loss. Basically sum(target * log(output)).
  """
  class_name = "ce"
  need_target = True

  def __init__(self,
               focal_loss_factor=0.0,
               label_smoothing=0.0, label_smoothing_gaussian=False,
               debug_dump=False,
               safe_log_opts=None,
               use_fused=True,
               fake_upper_bound=None,
               **kwargs):
    """
    :param float focal_loss_factor: see https://arxiv.org/abs/1708.02002. 0 means disabled
    :param float label_smoothing: 0.1 is a common default. see :func:`TFUtil.smoothing_cross_entropy`
    :param bool label_smoothing_gaussian: see :func:`TFUtil.smoothing_cross_entropy`
    :param bool debug_dump:
    :param dict[str] safe_log_opts: passed to :func:`safe_log`
    :param bool use_fused: if possible, use fused opts
    :param float|None fake_upper_bound: uses :func:`TFUtil.minimum_with_identity_grad`.
      I.e. you will see a finite loss, but we use the original gradient (which should be safe).
    """
    super(CrossEntropyLoss, self).__init__(**kwargs)
    self.focal_loss_factor = focal_loss_factor
    self.label_smoothing = label_smoothing
    self.label_smoothing_gaussian = label_smoothing_gaussian
    self.debug_dump = debug_dump
    self.safe_log_opts = safe_log_opts or {}
    self.use_fused = use_fused
    self.fake_upper_bound = fake_upper_bound

  def get_output_target_scores(self):
    """
    :return: shape (time_flat,), type float32
    :rtype: tf.Tensor
    """
    output_flat = self.output_flat
    if output_flat is None:
      output_flat = tf.nn.softmax(self.output_before_softmax_flat)
    assert output_flat is not None
    target_flat_exp = tf.stack(
      [tf.range(tf.shape(self.target_flat)[0], dtype=tf.int32),
       tf.cast(self.target_flat, tf.int32)], axis=1)  # (time,2)
    out = tf.gather_nd(output_flat, target_flat_exp, name="ce_output_target_scores")
    return out

  def get_value(self):
    """
    :rtype: tf.Tensor
    """
    from returnn.tf.util.basic import to_int32_64, smoothing_cross_entropy, safe_log, py_print
    from returnn.tf.util.basic import minimum_with_identity_grad
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
          print("Warning: using numerical unstable sparse Cross-Entropy loss calculation (%s to %s)" % (
            self.output, self.target), file=log.v3)
          if self.label_smoothing:
            out = smoothing_cross_entropy(
              logits=safe_log(self.output_flat), logits_are_normalized=True,
              labels=to_int32_64(self.target_flat), vocab_size=self.target.dim,
              label_smoothing=self.label_smoothing, gaussian=self.label_smoothing_gaussian)  # shape(labels)
          else:
            out = -safe_log(self.get_output_target_scores(), **self.safe_log_opts)
        if self.focal_loss_factor:
          out *= (1.0 - self.get_output_target_scores()) ** self.focal_loss_factor
        if self.fake_upper_bound is not None:
          out = minimum_with_identity_grad(out, self.fake_upper_bound)
        return self.reduce_func(out)
      else:  # not sparse
        assert not self.focal_loss_factor, "not implemented"
        assert not self.label_smoothing, "not implemented"
        assert not self.debug_dump, "not implemented"
        if self.use_fused and self.output_before_softmax_flat is not None:
          out = tf.nn.softmax_cross_entropy_with_logits(logits=self.output_before_softmax_flat, labels=self.target_flat)
          if self.fake_upper_bound is not None:
            out = minimum_with_identity_grad(out, self.fake_upper_bound)
          return self.reduce_func(out)
        else:
          print("Warning: using numerical unstable dense Cross-Entropy loss calculation", file=log.v3)
          out = self.target_flat * safe_log(self.output_flat, **self.safe_log_opts)
          if self.fake_upper_bound is not None:
            out = minimum_with_identity_grad(out, self.fake_upper_bound)
          return -self.reduce_func(out)


class BinaryCrossEntropyLoss(Loss):
  """
  Binary cross entropy.
  We expect the output as logits, not in probability space!
  Per frame: mean(target * log(sigmoid(output)) + (1 - target) * log(1 - sigmoid(output)))
  """
  class_name = "bin_ce"

  def __init__(self, pos_weight=None, **kwargs):
    """
    :param float|None pos_weight: weight of positive labels, see tf.nn.weighted_cross_entropy_with_logits.
    """
    super(BinaryCrossEntropyLoss, self).__init__(**kwargs)
    self._pos_weight = pos_weight

  def _check_init(self):
    assert self.target
    assert self.target.batch_ndim == self.output.batch_ndim, (
      "Number of dimensions mismatch. Target: %s, output: %s" % (self.target, self.output))

  def get_value(self):
    """
    :rtype: tf.Tensor
    """
    with tf.name_scope("loss_bin_ce"):
      target_flat = tf.cast(self.target_flat, self.output_flat.dtype)
      if self._pos_weight is None:
        out = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output_flat, labels=target_flat)
      else:
        out = tf.nn.weighted_cross_entropy_with_logits(
          logits=self.output_flat, labels=target_flat, pos_weight=self._pos_weight)
      return self.reduce_func(out) * (1.0 / (self.output.dim or 1))

  def get_error(self):
    """
    :return: frame error rate as a scalar value with the default self.reduce_func (see also self.get_value)
    :rtype: tf.Tensor
    """
    with tf.name_scope("loss_frame_error"):
      targets_bool = tf.cast(self.target_flat, tf.float32) > 0.5
      output_bool = tf.greater(self.output_flat, 0.)  # logits
      not_equal = tf.not_equal(output_bool, targets_bool)
      return self.reduce_func(tf.cast(not_equal, tf.float32)) * (1.0 / (self.output.dim or 1))


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
      nlog_scores = -tf_compat.v1.log(tf.clip_by_value(y, 1.e-20, 1.e20))  # (time,dim)
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
    from returnn.tf.util.basic import custom_gradient
    self._loss_func = custom_gradient.register(
      [tf.float32, tf.float32, tf.float32, tf.int32], op=loss, grad_op=loss_grad)

  def get_value(self):
    """
    :rtype: tf.Tensor
    """
    # Should be generic for any activation function.
    # (Except when the labels are not independent, such as for softmax.)
    # See Theano NetworkOutputLayer.FramewiseOutputLayer.cost() with "generic_ce" loss.
    from returnn.tf.util.basic import flatten_with_seq_len_mask
    # activation function can be anything, e.g. exp or sigmoid, but not softmax, must be elemwise.
    assert self.output_with_activation
    x = self.output_with_activation.x
    y = self.output_with_activation.y
    grad_f, = tf.gradients(tf_compat.v1.log(y), x)
    assert grad_f is not None
    grad_f = flatten_with_seq_len_mask(grad_f, seq_lens=self.output_seq_lens, time_major=self.output.is_time_major)
    x = flatten_with_seq_len_mask(x, seq_lens=self.output_seq_lens, time_major=self.output.is_time_major)
    y = flatten_with_seq_len_mask(y, seq_lens=self.output_seq_lens, time_major=self.output.is_time_major)
    assert y.get_shape().ndims == 2
    y /= tf.reduce_sum(y, axis=1, keepdims=True)
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
               use_native=False, use_viterbi=False, **kwargs):
    """
    :param bool target_collapse_repeated: like preprocess_collapse_repeated option for CTC. used for sparse_labels().
    :param bool auto_clip_target_len: see self._get_target_sparse_labels().
    :param bool output_in_log_space: False -> output expected in prob space. see self.get_output_logits
    :param int beam_width: used in eval
    :param dict[str]|None ctc_opts: other kwargs used for tf.nn.ctc_loss
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
    if ctc_opts is None:
      ctc_opts = {}
    self.ctc_opts = ctc_opts
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
    from returnn.tf.util.basic import sparse_labels
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
    from returnn.tf.util.basic import safe_log
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
        from returnn.tf.native_op import ctc_loss_viterbi
        assert not self.ctc_opts
        self._ctc_loss = ctc_loss_viterbi(
          logits=logits, logits_seq_lens=seq_lens, logits_time_major=self.output.is_time_major,
          targets=self.target.get_placeholder_as_batch_major(), targets_seq_lens=self.target_seq_lens)
      elif self.use_native:
        from returnn.tf.native_op import ctc_loss
        self._ctc_loss = ctc_loss(
          logits=logits, logits_seq_lens=seq_lens, logits_time_major=self.output.is_time_major,
          targets=self.target.get_placeholder_as_batch_major(), targets_seq_lens=self.target_seq_lens,
          **self.ctc_opts)
      else:
        self._ctc_loss = tf_compat.v1.nn.ctc_loss(
          inputs=logits, labels=labels, sequence_length=seq_lens, time_major=self.output.is_time_major,
          **self.ctc_opts)
      loss = self._ctc_loss  # shape (batch,)
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
          decoded = tf_util.ctc_greedy_decode(logits=logits, seq_lens=seq_lens, time_major=True)
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
    from returnn.tf.util.basic import sparse_labels
    return sparse_labels(output, seq_lens=seq_lens)

  def _map_labels(self, labels):
    """
    :param tf.SparseTensor labels:
    :rtype: tf.SparseTensor
    """
    if not self._label_map:
      return labels
    from returnn.tf.util.basic import map_labels
    return map_labels(labels, label_map=self._label_map)

  def get_output_logits(self):
    """
    :return: outputs in log-space / logits
    :rtype: tf.Tensor
    """
    from returnn.tf.util.basic import safe_log
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
    from returnn.tf.util.basic import vocab_idx_repr
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
      from returnn.tf.util.basic import py_print
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
    from returnn.tf.util.basic import bleu_score
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
    from returnn.tf.util.basic import identity
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
    :param returnn.tf.network.TFNetwork network:
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
        losses = tf.cast(self.losses.get_error(), tf.float32)
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
          beam_scores, name="scores_norm_shift", axis=1, keepdims=True)  # (batch,1)
        if self.norm_scores_stop_gradient:
          scores_norm_shift = tf.stop_gradient(scores_norm_shift)
        # Thus sum(value_weights) == 1.
        value_weights = tf.exp(beam_scores - scores_norm_shift)
      else:
        value_weights = tf.exp(beam_scores)
      if self.subtract_average_loss:
        # Gradient variance reduction for the gradient of the value-weights.
        # In case that the values also are differentiable, we don't want it to propagate through this.
        corrected_losses -= tf.stop_gradient(tf.reduce_mean(losses, axis=1, keepdims=True, name="avg_loss"))
      weighted_losses = tf.reduce_sum(corrected_losses * value_weights, axis=1, name="weighted_loss")  # (batch,)
      if self.loss_correction_grad_only and corrected_losses is not losses:
        weighted_losses += tf.stop_gradient(tf.reduce_sum((losses - corrected_losses) * value_weights, axis=1))
      if self.divide_beam_size:
        weighted_losses /= tf.cast(tf.shape(beam_scores)[-1], tf.float32)
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
    with tf.name_scope("loss_mse"):
      if self.output_before_softmax_flat is not None:
        x = tf.nn.softmax(self.output_before_softmax_flat)
      else:
        assert self.output_flat is not None
        x = self.output_flat
      assert x.get_shape().ndims == 2
      assert self.target_flat is not None
      if self.target.sparse:
        y = tf.one_hot(self.target_flat, self.target.dim)
      else:
        y = self.target_flat
      assert y.get_shape().ndims == 2
      out = tf_compat.v1.squared_difference(x, y)
      assert out.get_shape().ndims == 2
      out = self.reduce_func(tf.reduce_mean(out, axis=1))
      return out


class MeanL1Loss(Loss):
  """
  Like MSE loss, but with absolute difference
  """
  class_name = "mean_l1"

  def get_value(self):
    """
    :rtype: tf.Tensor
    """
    assert not self.target.sparse, "sparse target values are not yet supported"
    with tf.name_scope("loss_mean_l1"):
      return self.reduce_func(tf.reduce_mean(tf.abs(self.target_flat - self.output_flat), axis=1))


class ExternSprintLoss(Loss):
  """
  The loss is calculated by an extern Sprint instance.
  """
  class_name = "sprint"
  recurrent = True
  need_target = False

  def __init__(self, sprint_opts, **kwargs):
    """
    :param dict[str] sprint_opts:
    """
    super(ExternSprintLoss, self).__init__(**kwargs)
    self.sprint_opts = sprint_opts
    from returnn.tf.util.basic import custom_gradient
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
      from returnn.tf.sprint import get_sprint_loss_and_error_signal
      loss, error_signal = get_sprint_loss_and_error_signal(
        sprint_opts=self.sprint_opts,
        log_posteriors=tf_compat.v1.log(output),
        seq_lengths=self.output_seq_lens,
        seq_tags=seq_tags)
      loss = self.reduce_func(loss)
      from returnn.tf.util.basic import custom_gradient
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
    from returnn.tf.util.basic import custom_gradient
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
      from returnn.tf.util.basic import sequence_mask_time_major
      seq_mask = sequence_mask_time_major(self.output_seq_lens)
      from returnn.tf.native_op import fast_baum_welch_by_sprint_automata
      fwdbwd, obs_scores = fast_baum_welch_by_sprint_automata(
        sprint_opts=self.sprint_opts,
        am_scores=-tf_compat.v1.log(output),
        float_idx=seq_mask,
        tags=seq_tags)
      loss = self.reduce_func(obs_scores[0])
      bw = tf.exp(-fwdbwd)
      grad_x = (output - bw) * tf.expand_dims(seq_mask, 2)
      from returnn.tf.util.basic import custom_gradient
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
  need_target = False

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
    from returnn.tf.util.basic import custom_gradient
    custom_gradient.register_generic_loss_and_error_signal()

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace, the loss_opts
    :param returnn.tf.network.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    for key in ["error_signal_layer", "align_layer"]:
      if key in d:
        d[key] = get_layer(d[key])

  def get_value(self):
    """
    :rtype: tf.Tensor
    """
    from returnn.tf.util.basic import where_bc
    with tf.name_scope("ViaLayerLoss"):
      if self.error_signal_layer:
        assert not self.align_layer
        error_signal = self.error_signal_layer.output.copy_compatible_to(self.output).placeholder
      else:
        assert self.align_layer
        error_signal = self.output.placeholder - self.align_layer.output.copy_compatible_to(self.output).placeholder
      if self.output.is_time_axis_dynamic():
        seq_mask_bc = self.output.get_sequence_mask_broadcast()
        error_signal = where_bc(seq_mask_bc, error_signal, 0.0)
      if self.loss_wrt_to_act_in:
        assert self.output_with_activation, "activation unknown, via %r" % self.output
        if isinstance(self.loss_wrt_to_act_in, (str, unicode)):
          from returnn.tf.util.basic import get_activation_function
          assert self.output_with_activation.act_func is get_activation_function(self.loss_wrt_to_act_in)
        else:
          assert self.output_with_activation.act_func  # just check that there is some activation function
        grad_wrt = self.output_with_activation.x  # activation (e.g. softmax) input
      else:
        grad_wrt = self.output.placeholder
      from returnn.tf.util.basic import custom_gradient
      loss = self._loss_value
      assert isinstance(loss, tf.Tensor)
      orig_loss_shape = tf.shape(loss)
      # We need to add broadcast dims to loss such that it can be broadcasted to grad_wrt/error_signal.
      if loss.shape.ndims == 0:  # scalar
        pass  # should be ok...
      elif loss.shape.ndims == 1:  # assume [batch]
        loss = tf.reshape(
          loss, [orig_loss_shape[0] if i == self.output.batch_dim_axis else 1 for i in range(self.output.batch_ndim)])
      else:
        # Just hope that is has the same prefix shape.
        loss = tf.reshape(
          loss, [orig_loss_shape[i] if i < loss.shape.ndims else 1 for i in range(self.output.batch_ndim)])
      loss = custom_gradient.generic_loss_and_error_signal(
        loss=loss, x=grad_wrt, grad_x=error_signal)
      loss = tf.reshape(loss, orig_loss_shape)
      return self.reduce_func(loss)

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
  need_target = False

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


class SearchScoreLoss(Loss):
  """
  Use the scores from :class:`SearchChoices`.
  """
  class_name = "search_score"
  need_target = False

  def reduce_to_batch(self, loss, normalize):
    """
    :param tf.Tensor loss: (batch,)
    :param bool normalize: reduce mean instead of reduce sum
    :return: (batch,)
    :rtype: tf.Tensor
    """
    return loss

  def get_value(self):
    """
    :rtype: tf.Tensor
    """
    assert self.layer
    search_choices = self.layer.get_search_choices()
    assert self.layer.network.search_flag and search_choices, "no search?"
    # Negative score, because we minimize the loss, i.e. maximize the score.
    return self.reduce_func(-search_choices.beam_scores)

  def get_error(self):
    """
    :rtype: None
    """
    return None  # not defined


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
               remove_accidental_hits=None,
               sampler_args=None,
               nce_log_norm_term=0.0,
               **kwargs):
    """
    :param int num_sampled: Number of classes to be sampled. For sampled softmax, this is the number of classes to be
      used to estimate the sampled softmax. For noise contrastive estimation, this is the number of noise samples.
    :param int num_splits: Number of different samples (each with 'num_sampled' classes) to be used per batch.
    :param str sampler: Specify sampling distribution ("uniform", "log_uniform", "learned_unigram" or "fixed_unigram").
    :param bool nce_loss: If True, use noise contrastive estimation loss. Else (default), use the sampled softmax.
    :param bool use_full_softmax: If True, compute the full softmax instead of sampling (can be used for evaluation).
    :param bool|None remove_accidental_hits: If True, remove sampled classes that equal one of the target classes.
      If not specified (None), the value is determined based on the choosen objective.
      For sampled softmax this should be set to True; for NCE the default is False.
      Set this to True in case of NCE training and the objective is equal to sampled logistic loss.
    :param dict[str] sampler_args: additional arguments for the candidate sampler.
      This is most relevant to the fixed_unigram sampler.
      See https://www.tensorflow.org/api_docs/python/tf/random/fixed_unigram_candidate_sampler for details.
    :param float nce_log_norm_term: The logarithm of the constant normalization term for NCE.
    """
    super(SamplingBasedLoss, self).__init__(**kwargs)
    assert num_sampled >= 1
    assert sampler in ["uniform", "log_uniform", "learned_unigram", "fixed_unigram"], (
      "Sampler must be one of 'uniform', 'log_uniform', 'learned_unigram' or 'fixed_unigram'.")
    self.num_sampled = num_sampled
    self.num_splits = num_splits
    self.sampler = sampler
    self.use_full_softmax = use_full_softmax
    self.nce_loss = nce_loss
    self.remove_accidental_hits = remove_accidental_hits
    if self.remove_accidental_hits is None:
      self.remove_accidental_hits = not self.nce_loss
    self.sampler_args = sampler_args
    if self.sampler_args is None:
      self.sampler_args = {}
    self.nce_log_norm_term = nce_log_norm_term

  def _nce_loss(self,
                weights,
                biases,
                labels,
                inputs,
                num_sampled,
                num_classes,
                num_true=1,
                sampled_values=None,
                remove_accidental_hits=False,
                partition_strategy="div",
                name=None,
                seed=None):
    """
    Returns the example-wise NCE losses for the batch.

    This is mostly a copy of
      https://github.com/tensorflow/tensorflow/blob/e19c354920c3b246dda6598229210a582caaa1a9/tensorflow/python/ops/nn_impl.py#L1729-L1837
    with an extension which introduces the subtraction by a constant normalization term.

    :param tf.Tensor weights: NCE parameters of shape [num_classes, dim].
    :param tf.Tensor biases: biases with shape [num_classes]
    :param tf.Tensor labels: The target classes of shape [batch_size, num_true]
    :param tf.Tensor inputs: The forward activations of the network of shape [batch_size, dim]
    :param int num_sampled: Number of sampled classes
    :param int num_classes: Number of possible classes
    :param int num_true: Number of target classes per training example
    :param (tf.Tensor, tf.Tensor, tf.Tensor)|None sampled_values: a tuple of
      (`sampled_candidates`, `true_expected_count`, `sampled_expected_count`)
      returned by a `*_candidate_sampler` function (if None, we default to `log_uniform_candidate_sampler`).
      sampled_candidates is of dtype int64.
    :param bool remove_accidental_hits: Remove sampled classes that equal on of the target classes
    :param str partition_strategy: A string specifying the partitioning strategy, relevant
      if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
      Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
    :param str|None name: Name for the operation
    :param int|None seed: Seed for sampling if no sampled_values are given
    :return: A [batch_size] tensor of example-wise NCE losses
    :rtype: tf.Tensor
    """

    from returnn.tf.util.basic import compute_sampled_logits

    logits, labels = compute_sampled_logits(weights=weights,
                                            biases=biases,
                                            labels=labels,
                                            inputs=inputs,
                                            num_sampled=num_sampled,
                                            num_classes=num_classes,
                                            num_true=num_true,
                                            sampled_values=sampled_values,
                                            subtract_log_q=True,
                                            remove_accidental_hits=remove_accidental_hits,
                                            partition_strategy=partition_strategy,
                                            name=name,
                                            seed=seed)
    logits -= self.nce_log_norm_term
    sampled_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                             logits=logits,
                                                             name="sampled_losses")
    return tf.reduce_sum(sampled_losses, axis=1)

  def get_value(self):
    """
    :rtype: tf.Tensor
    """
    assert self.target.sparse
    layer = self.layer
    assert isinstance(layer, LinearLayer)
    input_data = layer.input_data
    assert isinstance(input_data, Data)

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
                        "learned_unigram": candidate_sampling_ops.learned_unigram_candidate_sampler,
                        "fixed_unigram": candidate_sampling_ops.fixed_unigram_candidate_sampler}
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
                                   range_max=self.target.dim,
                                   **self.sampler_args)
          if self.nce_loss:
            loss_fn = self._nce_loss
          else:
            loss_fn = tf.nn.sampled_softmax_loss

          assert self.layer.params["W"].shape[0] == self.target.dim, "Expect weight matrix of shape [num_classes, dim]"
          out = loss_fn(weights=self.layer.params["W"].read_value(),  # (num_classes,D).
                        biases=self.layer.params["b"].read_value(),  # (num_classes).
                        labels=labels_,  # (B'',1).
                        inputs=inputs_,  # (B'',D).
                        num_sampled=self.num_sampled,
                        num_classes=self.target.dim,
                        num_true=1,
                        sampled_values=sampled_values,
                        remove_accidental_hits=self.remove_accidental_hits,
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
          from returnn.tf.util.basic import to_int32_64
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
  for v in globals().values():
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
  from . import rec
  from . import signal_processing
  from . import segmental_model

  auto_register_layer_classes(list(globals().values()))
  for mod in [rec, signal_processing, segmental_model]:
    auto_register_layer_classes(list(vars(mod).values()))

  for alias, v in {"forward": LinearLayer, "hidden": LinearLayer}.items():
    assert alias not in _LayerClassDict
    _LayerClassDict[alias] = v


def auto_register_layer_classes(vars_values):
  """
  Example usage::

      from returnn.tf.layers.basic import auto_register_layer_classes
      auto_register_layer_classes('extern_private/your_stuff/CoolThingy.py')


  :param list|types.ModuleType|str vars_values: e.g. use list(globals().values()).
    str is considered as a module-filename
  :return: nothing
  """
  import inspect
  if isinstance(vars_values, str):
    from returnn.util.basic import generic_import_module
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
