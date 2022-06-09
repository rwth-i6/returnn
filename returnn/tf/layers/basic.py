
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
from returnn.tf.util.data import Data, SearchBeam, Dim, FeatureDim, SpatialDim
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


def concat_sources(src_layers, out_dim=None, allow_broadcast_all_sources=NotSpecified):
  """
  :param list[LayerBase] src_layers:
  :param Dim|None out_dim:
  :param bool|NotSpecified allow_broadcast_all_sources:
  :return: data with placeholders set
  :rtype: Data
  """
  assert src_layers, "need source layers"
  if len(src_layers) == 1:
    data = src_layers[0].output.copy()
    if out_dim:
      if out_dim == data.feature_dim_or_sparse_dim:
        pass  # good
      elif out_dim in data.dim_tags:
        # We found out_dim in the input but it is not marked as the feature dim.
        # This is explicitly allowed. Follow-up code will expect this to be the feature-dim though,
        # So we mark it accordingly.
        assert not data.sparse
        axis = data.get_axis_from_description(out_dim)
        data.feature_dim_axis = axis
      else:
        raise Exception("%s not found in %s" % (out_dim, data))
    return data
  network = src_layers[0].network
  cache_key = (tuple(src_layers), out_dim, 0.0, None)
  if cache_key in network.concat_sources_dropout_cache:
    return network.concat_sources_dropout_cache[cache_key].copy()
  data = get_concat_sources_data_template(
    src_layers, out_dim=out_dim, allow_broadcast_all_sources=allow_broadcast_all_sources)
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


def get_concat_sources_data_template(src_layers, out_dim=None, allow_broadcast_all_sources=NotSpecified, name=None):
  """
  This just creates a template :class:`Data` instance,
  without creating any real TF tensors.
  :func:`concat_sources` (and related) are the equivalent functions
  which would create a :class:`Data` together with the tensor.

  :param list[LayerBase]|tuple[LayerBase] src_layers:
  :param Dim|None out_dim:
  :param bool|NotSpecified allow_broadcast_all_sources:
  :param str|None name: name of the Data
  :return: data with no placeholders set. it is always a copy or new instance, so safe to manipulate
  :rtype: Data
  """
  assert src_layers, "need source layers"
  if len(src_layers) == 1:
    data = src_layers[0].output.copy_template(name=name)
    if out_dim:
      assert out_dim == data.feature_dim_or_sparse_dim
    return data
  if not name:
    name = "concat_" + "_".join([layer.name for layer in src_layers])
  dim = None
  common_source = Data.get_common_data(
    [s.output for s in src_layers], ignore_feature_dim=True, allow_broadcast_all_sources=allow_broadcast_all_sources)
  for layer in src_layers:
    # Note: We do not perform much compatibility checks at this point,
    # as this is for a template only anyway.
    # The real checks are in concat_sources.
    if layer.output.have_feature_axis():  # just ignore at this point if None (e.g. during template construction)
      layer_dim = layer.output.feature_dim_or_sparse_dim
      if layer_dim.dimension is not None:  # maybe during template construction
        if dim is None:
          dim = layer_dim
        else:
          dim = dim + layer_dim
  if out_dim:
    assert out_dim.dimension == dim.dimension
  else:
    out_dim = dim
  return common_source.copy_template_replace_dim_tag(
    name=name,
    axis=common_source.feature_dim_axis,
    new_dim_tag=out_dim)


def concat_sources_with_opt_dropout(src_layers, out_dim=None,
                                    dropout=0, dropout_axis=None, dropout_noise_shape=None, dropout_on_forward=False,
                                    allow_broadcast_all_sources=NotSpecified):
  """
  Concatenates in the feature dim (see :func:`concat_sources`),
  and then optionally applies dropout.

  :param list[LayerBase] src_layers:
  :param Dim|None out_dim:
  :param float dropout: dropout rate that will be applied if train_flag is set or dropout_on_forward is enabled
  :param Dim|str|list[Dim|str]|None dropout_axis:
  :param tuple|list|dict[Dim|str|list[Dim|str]|tuple[Dim|str],int|str|None]|None dropout_noise_shape:
    provide 1 for broadcasting or None otherwise for each axis.
    The default "None" will broadcast across all dynamic axes including the batch axis.
    Use {"*": None} to disable broadcasting for all axes.
  :param bool dropout_on_forward: apply dropout also during inference
  :param bool|NotSpecified allow_broadcast_all_sources:
  :return: data with placeholders set
  :rtype: Data
  """
  assert src_layers, "need source layers"
  data = concat_sources(src_layers, out_dim=out_dim, allow_broadcast_all_sources=allow_broadcast_all_sources)
  network = src_layers[0].network
  if network.train_flag is False and not dropout_on_forward:
    # If we know that we are not training, we always disable dropout.
    dropout = 0
  if not dropout:
    return data.copy()
  assert not data.sparse, "need dense data when dropout is used; sources: %r" % (src_layers,)
  if dropout_axis is not None:
    dropout_axis = data.get_axes_from_description(dropout_axis, allow_int=False)
    assert not dropout_noise_shape, (
      "do not provide both dropout_axis %r and dropout_noise_shape %r" % (dropout_axis, dropout_noise_shape))
    dropout_noise_shape = [dim if i in dropout_axis else 1 for i, dim in enumerate(data.batch_shape)]
  if isinstance(dropout_noise_shape, dict) or not dropout_noise_shape:
    # Default noise_shape behavior is like old for now:
    # All dynamic dimensions (batch,time) will use the same dropout-mask broadcasted.
    dropout_noise_shape = data.get_bc_shape(dropout_noise_shape)
  cache_key = (tuple(src_layers), out_dim, float(dropout), tuple(dropout_noise_shape))
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

  def __init__(self, in_dim=None, out_shape=None,
               dropout=0, dropout_axis=None, dropout_noise_shape=None, dropout_on_forward=False,
               mask=None,
               **kwargs):
    """
    :param Dim|None in_dim:
    :param set[Dim|returnn.tf.util.data._MarkedDim]|tuple|list|None out_shape:
    :param float dropout: 0.0 means to apply no dropout. dropout will only be applied during training
    :param Dim|str|list[Dim|str]|None dropout_axis:
    :param dict[Dim|str|list[Dim|str]|tuple[Dim|str],int|str|None]|None dropout_noise_shape:
      see :func:`Data.get_bc_shape`
    :param bool dropout_on_forward: apply dropout during inference
    :param str|None mask: "dropout" or "unity" or None. this is obsolete and only here for historical reasons
    """
    super(_ConcatInputLayer, self).__init__(in_dim=in_dim, out_shape=out_shape, **kwargs)
    assert mask in ['dropout', 'unity', None], "invalid mask: %r" % mask
    if mask == "unity":
      assert not dropout
    elif mask == "dropout":
      assert dropout > 0
    self.dropout = dropout
    self.input_data = None  # type: typing.Optional[Data]
    if self.sources:
      self.input_data = concat_sources_with_opt_dropout(
        self.sources, out_dim=in_dim,
        dropout=dropout, dropout_axis=dropout_axis, dropout_noise_shape=dropout_noise_shape,
        dropout_on_forward=dropout_on_forward,
        allow_broadcast_all_sources=True if out_shape else NotSpecified)


class CopyLayer(_ConcatInputLayer):
  """
  This layer does nothing, it copies its input.
  If multiple sources are provided, they are concatenated in the feature-dim.
  """

  layer_class = "copy"

  def __init__(self, in_dim=None, out_dim=None, extra_deps=(), **kwargs):
    """
    :param Dim|None in_dim:
    :param Dim|None out_dim:
    :param list[LayerBase] extra_deps: Just add as an additional dependency, without really using it.
      This can have an effect though on the search beam, via :class:`SelectSearchSourcesLayer`.
      We only have this here for the :class:`CopyLayer` because the :func:`get_out_data_from_opts`
      must know about it and define the right beam.
      Also see the option ``collocate_with``, which is different in that it does *not* add a dependency.
    """
    if in_dim and out_dim:
      assert in_dim == out_dim
    in_dim = in_dim or out_dim
    out_dim = in_dim
    super(CopyLayer, self).__init__(in_dim=in_dim, out_dim=out_dim, **kwargs)
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
  def get_out_data_from_opts(cls, name, sources=(), extra_deps=(),
                             out_type=None, out_dim=None, n_out=NotSpecified, out_shape=None,
                             **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param list[LayerBase] extra_deps:
    :param dict[str]|None out_type:
    :param Dim|None out_dim:
    :param int|None|NotSpecified n_out:
    :param set[Dim|returnn.tf.util.data._MarkedDim]|tuple|list|None out_shape:
    :rtype: Data
    """
    # If all sources are defined, use them to get the exact out_type.
    out = get_concat_sources_data_template(
      sources, out_dim=out_dim, name="%s_output" % name,
      allow_broadcast_all_sources=True if out_shape else NotSpecified)
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
        name=name, out_type=out_type, n_out=n_out, out_dim=out_dim, out_shape=out_shape, sources=sources, **kwargs)
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


class ConcatLayer(LayerBase):
  """
  Concatenates the inputs in specified axes.
  This generalizes :class:`CopyLayer` which concatenates in the feature dim.
  """
  layer_class = "concat"

  def __init__(self, sources, allow_broadcast=False, out_dim=None, **kwargs):
    """
    :param list[(LayerBase,str|Dim)] sources:
    :param bool allow_broadcast:
    :param Dim|None out_dim:
    """
    sources, axes = zip(*sources)  # unzip
    super(ConcatLayer, self).__init__(sources=sources, **kwargs)
    sources_data = [layer.output for layer in sources]  # type: typing.List[Data]
    axes_int = [src.get_axis_from_description(axis) for (src, axis) in zip(sources_data, axes)]
    concat_dim_tags = [src.dim_tags[axis] for (src, axis) in zip(sources_data, axes_int)]  # type: typing.List[Dim]
    if not out_dim:
      out_dim = sum(concat_dim_tags)
    out_concat_axis = self.output.get_axis_from_description(out_dim)

    def _copy_compatible(x, axis):
      """
      :param Data x: input
      :param int axis:
      :rtype: Data
      """
      dummy_ref = self.output.copy_template()
      dummy_ref = dummy_ref.copy_template_replace_dim_tag(
        axis=out_concat_axis, new_dim_tag=x.dim_tags[axis])
      return x.copy_compatible_to(dummy_ref, add_dims=allow_broadcast, unbroadcast=False)

    sources_data = [_copy_compatible(src, axis) for (src, axis) in zip(sources_data, axes_int)]
    self.output.placeholder = tf_util.concat_with_opt_broadcast(
      [src.placeholder for src in sources_data], axis=out_concat_axis,
      allow_broadcast=[allow_broadcast] * len(sources_data))

  @classmethod
  def get_out_data_from_opts(cls, name, sources, out_dim=None, **kwargs):
    """
    :param str name:
    :param list[(LayerBase,str|Dim)] sources:
    :param Dim|None out_dim:
    :rtype: Data
    """
    assert sources
    sources, axes = zip(*sources)  # unzip
    axes_int = [layer.output.get_axis_from_description(axis) for (layer, axis) in zip(sources, axes)]
    concat_dim_tags = [
      layer.output.dim_tags[axis] for (layer, axis) in zip(sources, axes_int)]  # type: typing.List[Dim]
    if any(tag.dimension is None for tag in concat_dim_tags):
      dimension = None
    else:
      dimension = 0
      for tag in concat_dim_tags:
        dimension += tag.dimension
    if not out_dim:
      out_dim = sum(concat_dim_tags)
      assert isinstance(out_dim, Dim)
    else:
      sum(concat_dim_tags).declare_same_as(out_dim)
    assert out_dim.dimension == dimension

    def _as_common(x, axis):
      """
      :param Data x: input
      :param int axis:
      :rtype: Data
      """
      return x.copy_template_replace_dim_tag(axis=axis, new_dim_tag=out_dim)

    sources_data = [_as_common(layer.output, axis) for (layer, axis) in zip(sources, axes_int)]
    # Always allow broadcast here, for template construction. We will check it in __init__.
    return Data.get_common_data(sources_data, allow_broadcast_all_sources=True, name="%s_output" % name)

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param returnn.tf.network.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    sources_and_axes = d.pop("from")
    d["from"], axes = zip(*sources_and_axes)
    super(ConcatLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["sources"] = list(zip(d["sources"], axes))


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
  Uses :func:`returnn.tf.util.basic.scaled_gradient`, or :func:`tf.stop_gradient`
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
    from returnn.tf.util.basic import select_src_beams, get_valid_scope_name_from_str, Dim
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
          tag = Dim.get_tag_from_size_tensor(v)
          if tag:
            assert tag.dyn_size_ext.is_batch_major
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
      for src_tag, out_tag in zip(src_output.dim_tags, self.output.dim_tags):
        assert src_tag.dimension == out_tag.dimension
        if src_tag.is_batch_dim():
          assert out_tag.batch == self.output.batch
          continue
        if src_tag.dimension is not None:
          continue
        if out_tag.dyn_size_ext is None:
          if src_tag.dyn_size_ext.have_batch_axis():
            out_tag.dyn_size_ext = src_tag.dyn_size_ext.copy_template()
            out_tag.dyn_size_ext.beam = None
            out_tag.dyn_size_ext = out_tag.dyn_size_ext.copy_extend_with_beam(self.output.beam)
          else:
            out_tag.dyn_size_ext = src_tag.dyn_size_ext.copy()
        if out_tag.dyn_size_ext.placeholder is None:
          assert out_tag.dyn_size_ext.have_batch_axis() and out_tag.dyn_size_ext.is_batch_major
          out_tag.dyn_size_ext.placeholder = transform(src_tag.dyn_size_ext.placeholder)
        if out_tag.dyn_size_ext.have_batch_axis():
          assert out_tag.dyn_size_ext.batch == out_tag.batch == self.output.batch
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
    search_choices_ = search_choices.get_search_choices()
    data = sources[0].output.copy_template().copy_as_batch_major()
    if data.beam or search_choices_:
      assert search_choices_
      data = data.copy_extend_with_beam(search_choices_.get_beam_info())
    return data


class ActivationLayer(_ConcatInputLayer):
  """
  This layer just applies an activation function.
  See :func:`returnn.tf.util.basic.get_activation_function` about supported functions.
  Also see :class:`EvalLayer` and :class:`CombineLayer` for similar layers.
  """

  layer_class = "activation"

  def __init__(self, activation, opts=None, **kwargs):
    """
    :param str activation: e.g. "relu", "tanh", etc
    :param dict[str]|None opts: for activation function, e.g. eps for safe_log
    """
    super(ActivationLayer, self).__init__(**kwargs)
    x = self.input_data.copy_compatible_to(self.output, check_dtype=False).placeholder
    if activation:
      if "softmax" in activation:
        assert not opts  # do not set axis or anything. this handled automatically. we moved feature to last axis.
        if self.output.dim_tags[-1].is_dynamic():
          self.recurrent = True
      from returnn.tf.util.basic import get_activation_function
      act_func = get_activation_function(activation)
      self.output_before_activation = OutputWithActivation(x, act_func=act_func, act_func_opts=opts)
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
    if activation in ["abs", "angle"] and out.dtype == "complex64":
      out.dtype = "float32"
    if "softmax" in activation:
      # Make sure we use the right axis.
      out = out.copy_with_feature_last()
    return out


class BatchNormLayer(CopyLayer):
  """
  Implements batch-normalization (https://arxiv.org/abs/1502.03167) as a separate layer.

  Also see :class:`NormLayer`.
  """
  layer_class = "batch_norm"

  def __init__(self,
               in_dim=None,
               use_shift=NotSpecified, use_std=NotSpecified,
               use_sample=NotSpecified, force_sample=NotSpecified,
               momentum=NotSpecified, epsilon=NotSpecified,
               update_sample_only_in_training=NotSpecified,
               delay_sample_update=NotSpecified,
               param_version=NotSpecified,
               gamma_init=NotSpecified, beta_init=NotSpecified,
               masked_time=NotSpecified, **kwargs):
    """
    :param returnn.tf.util.data.Dim|None in_dim:
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

    The default settings for these variables are set in the function :func:`batch_norm` of :class:`LayerBase`.
    If you do not want to change them you can leave them undefined here.
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
    super(BatchNormLayer, self).__init__(batch_norm=batch_norm_opts or True, in_dim=in_dim, **kwargs)
    if in_dim:
      # should be the case via get_out_data_from_opts
      assert self.output.dim_tags[self.output.feature_dim_axis] == in_dim
    # batch norm is now applied via post_init


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

  def __init__(self, in_dim=None, out_dim=None, epsilon=1e-6, **kwargs):
    """
    :param Dim|None in_dim: axis to normalize over. feature-dim by default
    :param Dim|None out_dim: just the same as in_dim
    :param float epsilon:
    """
    super(LayerNormLayer, self).__init__(**kwargs)
    assert not self.input_data.sparse
    x = self.input_data.placeholder
    if not in_dim and out_dim:
      in_dim = out_dim
    if in_dim:
      if out_dim:
        assert in_dim == out_dim
      assert isinstance(in_dim, Dim)
      axis = self.input_data.get_axis_from_description(in_dim)
    else:
      axis = self.input_data.feature_dim_axis
    dim = self.input_data.batch_shape[axis]
    assert dim is not None, "%s: in_dim %i must be static in input %s" % (self, in_dim or axis, self.input_data)
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

  def __init__(self, axis=NotSpecified, axes=NotSpecified,
               param_shape=NotSpecified, scale=True, bias=True, epsilon=1e-6, **kwargs):
    """
    :param Dim|str|list[Dim|str] axis: axis or axes over which the mean and variance are computed, e.g. "F" or "TF"
    :param Dim|str|list[Dim|str] axes: axis or axes over which the mean and variance are computed, e.g. "F" or "TF"
    :param Dim|str|list[Dim|str]|tuple[Dim|str] param_shape: shape of the scale and bias parameters.
      You can also refer to (static) axes of the input, such as the feature-dim.
      This is also the default, i.e. a param-shape of [F], independent of the axes to normalize over.
    :param bool scale: add trainable scale parameters
    :param bool bias: add trainable bias parameters
    :param float epsilon: epsilon for numerical stability
    """
    if axis is not NotSpecified:
      assert axes is NotSpecified
      axes = axis
    super(NormLayer, self).__init__(**kwargs)
    assert not self.input_data.sparse
    x = self.input_data.placeholder
    if scale or bias:
      if param_shape is NotSpecified:
        param_shape = "F"
      if isinstance(param_shape, (list, tuple)):
        param_axes = [self.input_data.get_axis_from_description(a, allow_int=False) for a in param_shape]
      else:
        param_axes = [self.input_data.get_axis_from_description(param_shape, allow_int=False)]
      assert sorted(set(param_axes)) == sorted(param_axes), "%s: param_shape %r should be unique" % (self, param_shape)
      param_shape = [self.input_data.batch_shape[axis] for axis in param_axes]
      assert all(isinstance(dim, int) for dim in param_shape), "%s: only static param shape allowed" % self
      param_dim_tags = [self.input_data.dim_tags[axis] for axis in param_axes]
    else:
      assert param_shape is NotSpecified or not param_shape
      param_dim_tags = None
    axes = self.input_data.get_axes_from_description(axes)

    mean = tf.reduce_mean(x, axis=axes, keepdims=True, name="mean")
    variance = tf.reduce_mean(tf.square(x - mean), axis=axes, keepdims=True, name="variance")
    with tf.name_scope("normalized"):
      norm_x = (x - mean) * tf_compat.v1.rsqrt(variance + epsilon)
    if scale:
      with self.var_creation_scope():
        scale_param = self.add_param(tf_compat.v1.get_variable("scale", param_shape, initializer=tf.ones_initializer()))
      norm_x *= (
        Data(name="scale_param", dim_tags=param_dim_tags, placeholder=scale_param)
        .copy_compatible_to(self.output).placeholder)
    if bias:
      with self.var_creation_scope():
        bias_param = self.add_param(tf_compat.v1.get_variable("bias", param_shape, initializer=tf.zeros_initializer()))
      norm_x += (
        Data(name="bias_param", dim_tags=param_dim_tags, placeholder=bias_param)
        .copy_compatible_to(self.output).placeholder)
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

  def __init__(self, p, axis=NotSpecified, axes=NotSpecified, keep_dims=False, **kwargs):
    """
    :param int|float p:
    :param Dim|str|list[Dim|str] axis:
    :param Dim|str|list[Dim|str] axes:
    :param bool keep_dims:
    """
    if axis is not NotSpecified:
      assert axes is NotSpecified
      axes = axis
    super(MathNormLayer, self).__init__(**kwargs)
    x = self.input_data.copy()
    x.placeholder = tf.abs(x.placeholder) ** p
    self.output.placeholder = ReduceLayer.reduce(x, mode="sum", axes=axes, keep_dims=keep_dims) ** (1. / p)

  @classmethod
  def get_out_data_from_opts(cls, name, sources, axis=NotSpecified, axes=NotSpecified, keep_dims=False, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param Dim|str|list[Dim|str] axis:
    :param Dim|str|list[Dim|str] axes:
    :param bool keep_dims:
    :rtype: Data
    """
    if axis is not NotSpecified:
      assert axes is NotSpecified
      axes = axis
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

  def __init__(self, axis, slice_start=None, slice_end=None, slice_step=None, out_dim=None, **kwargs):
    """
    :param Dim|str axis:
    :param str|None axis_kind: "T" for time, "B" for batch, "F" for feature
    :param int|None slice_start:
    :param int|None slice_end:
    :param int|None slice_step:
    :param Dim|None out_dim:
    """
    out_dim  # noqa  # via get_out_data_from_opts
    super(SliceLayer, self).__init__(**kwargs)
    axis = self.input_data.get_axis_from_description(axis)
    dim_slice = slice(slice_start, slice_end, slice_step)
    slices = [slice(None, None)] * axis + [dim_slice]
    y = self.input_data.placeholder[slices]
    y.set_shape(self.output.batch_shape)  # can be necessary for slice_end>0
    self.output.placeholder = y

  @classmethod
  def get_out_data_from_opts(
        cls, name, axis, sources=(),
        slice_start=None, slice_end=None, slice_step=None,
        out_dim=None, **kwargs):
    """
    :param str name:
    :param Dim|str axis:
    :param list[LayerBase] sources:
    :param int|None slice_start:
    :param int|None slice_end:
    :param int|None slice_step:
    :param Dim|None out_dim:
    :rtype: Data
    """
    from ..util.data import Dim
    input_data = get_concat_sources_data_template(sources)
    axis = input_data.get_axis_from_description(axis)
    dim_tag = input_data.dim_tags[axis]
    out_dim_ = dim_tag
    if slice_start:
      assert slice_start >= 0
      out_dim_ = out_dim_.sub_left(slice_start)
    if slice_end is not None:
      if slice_end >= 0:
        out_dim_ = Dim(
          kind=dim_tag.kind, description="%s:slice-end" % name,
          dimension=slice_end - (slice_start or 0), auto_generated=True)
      else:  # slice_end < 0
        out_dim_ = out_dim_ - (-slice_end)
    if slice_step and slice_step != 1:
      out_dim_ = out_dim_.ceildiv_right(abs(slice_step))
    if out_dim:
      out_dim_.declare_same_as(out_dim)
    return input_data.copy_template_replace_dim_tag(axis=axis, new_dim_tag=out_dim_, name="%s_output" % name)


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

  def __init__(self, start, size, min_size=None, axis="T", out_spatial_dim=None, **kwargs):
    """
    :param LayerBase start: (B,...)
    :param int|LayerBase|Dim|None size:
      We assume that this is >=0. If this might not be the case, use ``min_size=0``.
      If None, it uses the max possible size, and it becomes a dynamic axis.
    :param int|None min_size: if size is None, but we want to have a min-size
    :param Dim|str axis:
    :param Dim|None out_spatial_dim:
    """
    out_spatial_dim  # noqa  # via get_out_data_from_opts
    super(SliceNdLayer, self).__init__(**kwargs)
    from returnn.tf.util.basic import where_bc
    from returnn.tf.util.data import Data
    x = self.input_data.copy()
    in_axis = x.get_axis_from_description(axis, allow_int=False)
    in_tag = x.dim_tags[in_axis]
    seq_lens_data = in_tag.dyn_size_ext  # (B,) or None
    self.start = start
    self.size = size
    start_data = start.output.copy()  # e.g. (B,) or (B,T)
    data_objs = [start_data]
    data_objs += [size.output] if isinstance(size, LayerBase) else []
    data_objs += [seq_lens_data] if isinstance(seq_lens_data, Data) else []
    common_data = Data.get_common_data(data_objs, name="%s_inputs")
    start_data = start_data.copy_compatible_to(common_data, check_sparse=False)
    start_t = start_data.placeholder
    if size is None:
      if seq_lens_data is None:
        assert isinstance(x.batch_shape[in_axis], int)
        size_t = x.batch_shape[in_axis] - start_t
      else:
        seq_lens_t = seq_lens_data.copy_compatible_to(common_data, check_sparse=False).placeholder
        size_t = seq_lens_t - start_t
      size_t = tf.maximum(size_t, min_size or 0)  # must make sure >=0 in any case
      size = tf.reduce_max(size_t)  # scalar
    elif isinstance(size, LayerBase):
      size_data = size.output.copy_compatible_to(common_data, check_sparse=False)
      size_t = size_data.placeholder  # assume already >=0
      if min_size:
        size_t = tf.maximum(size_t, min_size)
      size = tf.reduce_max(size_t)  # scalar
    elif isinstance(size, Dim):
      assert size.dyn_size_ext
      size_data = size.dyn_size_ext.copy_compatible_to(common_data, check_sparse=False)
      size_t = size_data.placeholder  # assume already >=0
      if min_size:
        size_t = tf.maximum(size_t, min_size)
      size = tf.reduce_max(size_t)  # scalar
    else:
      assert isinstance(size, int), "%s: invalid type %s for size" % (self, type(size))
      size_t = None
    # for each start index in start_data, we want to gather a slice
    # therefore, the output's first axes are the same as the ones from start_data
    # and the next axis will therefore be the slice axis
    slice_tag = self.output.dim_tags[start_data.batch_ndim]
    assert slice_tag.description.startswith("sliced-time:")
    if size_t is not None:
      # in this case, size is not known before runtime and becomes dynamic and we need to set dyn_size
      assert not isinstance(size, int)
      assert isinstance(size_t, tf.Tensor)
      dyn_size_ext = Data(
        name=("%s:dyn_size" % slice_tag.description),
        dtype=Data.size_dtype,
        placeholder=size_t,
        dim_tags=start_data.dim_tags,  # (B,) or (B,T)
        batch=slice_tag.batch,
        beam=slice_tag.batch.beam if slice_tag.batch else self.output.beam,
        control_flow_ctx=slice_tag.control_flow_ctx)
      slice_tag.dyn_size_ext = dyn_size_ext
      slice_tag.set_tag_on_size_tensor(size_t)
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
      axis=in_tag)
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
  def get_out_data_from_opts(cls, name, sources=(), start=None, size=None, axis="T", out_spatial_dim=None, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param LayerBase|None start:
    :param int|LayerBase|Dim|None size:
    :param Dim|str axis:
    :param Dim|None out_spatial_dim:
    :rtype: Data
    """
    start_data = start.output.copy()
    gather_positions_data = start_data.copy_template(name="%s_gather_positions" % name)
    if isinstance(size, LayerBase):
      size = None
    if isinstance(size, Dim):
      if out_spatial_dim:
        assert size == out_spatial_dim
      else:
        out_spatial_dim = size
    else:
      # size might be None here in which case we set the dyn_size in __init__
      assert size is None or isinstance(size, int)
      out_spatial_dim_ = Dim(
        kind=Dim.Types.Spatial,
        description="sliced-time:%s" % name,
        dimension=size, auto_generated=True)
      if out_spatial_dim:
        out_spatial_dim_.declare_same_as(out_spatial_dim)
      else:
        out_spatial_dim = out_spatial_dim_
    gather_positions_data = gather_positions_data.copy_add_dim_by_tag(
      out_spatial_dim, unbroadcast=True, axis=start_data.batch_ndim)
    position = InternalLayer(
      network=sources[0].network,
      name="%s_internal" % gather_positions_data.name,
      output=gather_positions_data)
    return GatherLayer.get_out_data_from_opts(
      name="%s_gather" % name,
      sources=sources,
      position=position,
      axis=axis)

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
    :param Dim|str axis: The axis into which we gather the indices into
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
    from returnn.util import BehaviorVersion
    is_equal_opts = dict(allow_same_spatial_dim=True)
    if BehaviorVersion.get() < 11:
      is_equal_opts["broadcast_matches"] = True
    all_dim_tags, tags_dict = Dim.get_all_dimension_tags([input_data, position_data], is_equal_opts=is_equal_opts)
    input_tags, pos_tags = tags_dict[input_data], tags_dict[position_data]
    specific_input_axes = [i for i, tag in enumerate(input_tags) if tag not in pos_tags and i != old_gather_axis]
    # note: we currently also allow the gather axis dim is present in the position data.
    specific_pos_axes = [
      i for i, tag in enumerate(pos_tags) if tag not in input_tags or tag == input_tags[old_gather_axis]]
    common_axes_input = [
      i for i, tag in enumerate(input_data.dim_tags) if tag in input_tags and tag in pos_tags and i != old_gather_axis]
    # order of common_axes_pos must match order of common_axes_input
    common_axes_map_input_to_pos = position_data.find_matching_dim_map(input_data, common_axes_input)
    common_axes_pos = [common_axes_map_input_to_pos[input_axis] for input_axis in common_axes_input]

    assert set(common_axes_input) | set(specific_input_axes) | {old_gather_axis} == set(range(input_data.batch_ndim))
    assert set(common_axes_pos) | set(specific_pos_axes) == set(range(position_data.batch_ndim))
    return common_axes_input, common_axes_pos, specific_input_axes, specific_pos_axes

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
    :param Dim|str axis:
    :rtype: Data
    """
    from returnn.tf.util.data import BatchInfo
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
    if old_gather_axis == input_data.batch_dim_axis:
      out_type["batch"] = position_data.batch
    else:
      out_type["batch"] = BatchInfo.get_common_batch_info([src.batch for src in (input_data, position_data)])

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
    if (
          input_data.feature_dim_axis_or_unspecified is NotSpecified and
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
    out_type["sparse_dim"] = input_data.sparse_dim
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

  Note that "nd" is maybe a bit misleading.
  While we operate on N-D tensors, the indices (``position``)
  are into a single new dimension.

  The input to the layer are the ``updates``, the ``indices`` are via the ``position`` argument.
  The indices are into the newly constructed output dimension.
  The output shape is constructed via the common shape of the input, the position,
  and the unique common axis (if not unique, we would need to introduce an option to specify it)
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

  def __init__(self, position, position_axis, output_dim_via_time_from=None, out_spatial_dim=None,
               filter_invalid_indices=False, **kwargs):
    """
    :param LayerBase position: indices into first axis (excluding batch) of the output
    :param Dim|str position_axis: axis in `position` to replace by the output-dim
    :param LayerBase|None output_dim_via_time_from: use the time-dim from this layer as the output-dim
    :param Dim|None out_spatial_dim:
    :param bool filter_invalid_indices: allow for indices <0 or >= output_dim, which will be discarded in the output
    """
    super(ScatterNdLayer, self).__init__(**kwargs)
    assert (out_spatial_dim or output_dim_via_time_from) and not (out_spatial_dim and output_dim_via_time_from), (
      "%s: provide either out_spatial_dim or output_dim_via_time_from but not both" % self)
    if not out_spatial_dim:
      out_spatial_dim = output_dim_via_time_from.output.get_time_dim_tag()
    assert out_spatial_dim.is_dim_known(), (
      "%s: out_spatial_dim %s must have a known (dynamic or static) dim" % (self, out_spatial_dim))
    self.position = position
    common, output, replace_common_axis, input_extra_axes = self._get_axes(
      input_data=self.input_data, position=position.output, position_axis=position_axis,
      out_spatial_dim=out_spatial_dim)
    pos_v = position.output.placeholder
    pos_ndim = position.output.batch_ndim
    assert 0 <= replace_common_axis < pos_ndim
    pos_shape = [position.output.get_dim(i) for i in range(pos_ndim)]
    if output_dim_via_time_from:
      output_dim = output_dim_via_time_from.output.time_dimension()
    else:
      output_dim = out_spatial_dim.get_dim_value()
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
    # Now we need to implement a similar logic as `returnn.tf.util.basic.nd_indices`, but more generic.
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
  def _get_axes(cls, input_data, position, position_axis, out_spatial_dim):
    """
    :param Data input_data: updates
    :param Data position: indices
    :param Dim|str position_axis: axis in `position` to replace by the output-dim
    :param Dim out_spatial_dim:
    :rtype: (Data, Data, int, list[int])
    :return: common, output, axis, input_extra_axes
    """
    from returnn.tf.util.basic import Dim
    # Construct `common` manually, not via Data.get_common_data, such that we can control the axis order.
    # We want the same axis from `position`, and all further axes should be added behind that.
    common = position.copy_template()
    common.dtype = input_data.dtype
    common.sparse_dim = input_data.sparse_dim
    common.sanity_check()
    dim_tags, tags_dict = Dim.get_all_dimension_tags(
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
    output = common.copy_template_replace_dim_tag(axis=position_axis, new_dim_tag=out_spatial_dim)
    return common, output, position_axis, input_extra_axes

  @classmethod
  def get_out_data_from_opts(cls, name, sources, position, position_axis,
                             output_dim_via_time_from=None, out_spatial_dim=None,
                             **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param LayerBase position:
    :param Dim|str position_axis: axis in `position` to replace by the output-dim
    :param LayerBase|None output_dim_via_time_from: use the time-dim from this layer as the output-dim
    :param Dim|None out_spatial_dim:
    :rtype: Data
    """
    input_data = get_concat_sources_data_template(sources)
    common, output, replace_common_axis, input_extra_axes = cls._get_axes(
      input_data=input_data, position=position.output, position_axis=position_axis,
      out_spatial_dim=out_spatial_dim if out_spatial_dim else output_dim_via_time_from.output.get_time_dim_tag())
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
    if d.get("output_dim_via_time_from", None):
      d["output_dim_via_time_from"] = get_layer(d["output_dim_via_time_from"])


class LinearLayer(_ConcatInputLayer):
  """
  Linear/forward/fully-connected/1x1-conv layer.
  Does a linear transformation on the feature-dimension of the input
  with an optional bias term and an optional activation function.
  See also :class:`DotLayer`, :class:`ElemwiseProdLayer`, :class:`WeightedSumLayer`.
  """
  layer_class = "linear"

  def __init__(self,
               activation=None, with_bias=True, grad_filter=None, forward_weights_init="glorot_uniform",
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
    :param str|Dim axis:
    :param bool add_time_axis: should not be used
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
      # You anyway should not use this, so it's ok to have only a single case supported here.
      self.output.placeholder = tf.expand_dims(dim.dyn_size, axis=self.output.time_dim_axis)
    elif dim.is_batch_dim():
      self.output.placeholder = source.get_batch_dim()
    elif dim.dimension is not None:  # static
      self.output.placeholder = tf.constant(dim.dimension, dtype=dtype, name="static_dim")
    else:
      self.output.placeholder = dim.dyn_size_ext.placeholder

  @classmethod
  def get_out_data_from_opts(cls, name, sources, axis="T", add_time_axis=False, dtype="int32", sparse=False, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param str|Dim axis:
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
      # You anyway should not use this, so it's ok to have only a single case supported here.
      assert dim.dyn_size_ext and dim.dyn_size_ext.have_batch_axis() and dim.dyn_size_ext.batch_ndim == 1  # [B]
      return Data(
        name="%s_length" % name,
        shape=[1], batch_dim_axis=0, time_dim_axis=1,
        dtype=dtype, sparse=sparse, dim=None if sparse else NotSpecified)
    if dim.is_batch_dim():
      return Data("%s_batch_dim" % name, dim_tags=(), dtype=dtype, sparse=sparse)
    if dim.dimension is not None:  # static
      return Data("%s_static_dim" % name, dim_tags=(), dtype=dtype, sparse=sparse)
    if not dim.dyn_size_ext:  # yet undefined
      return Data(
        name="%s_length" % name,
        shape=(), batch_dim_axis=0, time_dim_axis=None,
        dtype=dtype, sparse=sparse, dim=None if sparse else NotSpecified)
    return dim.dyn_size_ext.copy()


class SoftmaxOverSpatialLayer(_ConcatInputLayer):
  """
  This applies a softmax over spatial axis/axes (currently only time axis supported).
  E.g. when the input is of shape (B,T,dim), the output will be (B,T,dim).
  It automatically masks the frames outside the seq defined by the seq-len.
  In contrast to :class:`SoftmaxLayer`, this will not do a linear transformation.
  See :class:`SeqLenMaskLayer` if you just want to apply a masking.
  """
  layer_class = "softmax_over_spatial"
  recurrent = True  # can operate on the spatial dim

  def __init__(self, axis=None, energy_factor=None,
               start=None, window_start=None, window_size=None, use_time_mask=None,
               log_space=False, **kwargs):
    """
    :param Dim|str|None axis: which axis to do the softmax over. "T" by default
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
    if not energy_data.dim_tags[axis].is_dynamic():
      self.recurrent = False
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
        axis=axis, axis_allow_int=True,
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
    :param Dim|str|None axis:
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
    :param Dim|str|None axis:
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
    :param Dim|str axis:
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
    assert isinstance(axis, (Dim, str)), "%s: use symbolic axis (e.g. 'T') or Dim instance" % self
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
  def build_mask(cls, x, axis="T", axis_allow_int=NotSpecified,
                 seq_len_source=None, start=None, window_start=None, window_size=None):
    """
    :param Data x:
    :param Dim|str|int axis:
    :param bool|NotSpecified axis_allow_int:
      Some callers of this function would pass in an int for axis directly.
      In that case, explicitly set this to True.
    :param Data|None seq_len_source:
    :param Data|None start:
    :param Data|None window_start:
    :param Data|int|None window_size:
    :return: mask which is broadcastable to energy_data, thus you can e.g. use :func:`returnn.tf.util.basic.where_bc`
    :rtype: tf.Tensor
    """
    from returnn.tf.util.basic import get_shape
    energy = x.placeholder
    energy_shape = get_shape(energy)
    axis = x.get_axis_from_description(axis, allow_int=axis_allow_int)
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


class RandomStateInitLayer(LayerBase):
  """
  This calculates the initial state value for the state var
  of :class:`RandomLayer`.
  This depends on the algorithm and seed.
  """
  layer_class = "random_state_init"

  def __init__(self, algorithm=None, seed=None, out_dim=None, **kwargs):
    """
    :param str|tf.random.Algorithm|None algorithm: "philox", "three-fry", "auto-select". by default "philox".
      See :func:`tf.random.stateless_uniform` for some documentation.
      "auto-select" will automatically select the optimal algorithm based on the device,
      so it might select a different algorithm depending on the device.
      Note that the state shape is dependent on the device, so if you want that checkpoints are compatible
      across devices, do not use "auto-select".
      We take the default from :class:`tf.random.Generator`.
    :param int|typing.Sequence[int]|numpy.ndarray|None seed: if given, the state will deterministically depend on this
      (and the algorithm) and nothing else. If you have multiple random generators (state vars),
      make sure that you have different seeds for each!
      If None (default), the seed will be deterministically taken from the network random generator
      at construction time, which is usually a good idea. You still can change the global network seed.
    :param Dim|None out_dim: new dim tag for random state dim
    """
    out_dim  # noqa  # via get_out_data_from_opts
    super(RandomStateInitLayer, self).__init__(**kwargs)
    if seed is None:
      seed = self.network.random.randint(2 ** 31, size=self.output.shape)
    algorithm = self.select_algorithm(algorithm)
    self.algorithm = algorithm
    self.seed = seed
    from tensorflow.python.ops import stateful_random_ops
    self.output.placeholder = tf.convert_to_tensor(stateful_random_ops.create_rng_state(seed=seed, alg=algorithm))

  @classmethod
  def select_algorithm(cls, algorithm):
    """
    :param str|int|tf.random.Algorithm|None algorithm:
    :rtype: int
    """
    from tensorflow.python.ops import stateful_random_ops, stateless_random_ops
    if algorithm is None:
      # Note: No auto-select because we want that the checkpoints are compatible across devices,
      #   and thus it is required that the state var stays compatible (esp the shape).
      return stateful_random_ops.DEFAULT_ALGORITHM
    if isinstance(algorithm, int):
      return algorithm
    if isinstance(algorithm, tf.random.Algorithm):
      return algorithm.value
    if isinstance(algorithm, str):
      try:
        # noinspection PyUnresolvedReferences
        convert_alg_to_int = stateless_random_ops.convert_alg_to_int
      except AttributeError:
        # noinspection PyProtectedMember,PyUnresolvedReferences
        convert_alg_to_int = stateful_random_ops._convert_alg_to_int  # TF 2.6 or earlier
      return convert_alg_to_int(algorithm.lower())
    raise TypeError("algorithm %r" % (algorithm,))

  _state_size_dim_tags_cache = {}  # type: typing.Dict[int, Dim]

  @classmethod
  def get_out_data_from_opts(cls, name, algorithm=None, out_dim=None, **kwargs):
    """
    :param str name:
    :param str|None algorithm:
    :param Dim|None out_dim:
    :rtype: Data
    """
    from tensorflow.python.ops import stateful_random_ops
    algorithm = cls.select_algorithm(algorithm)
    algo_type = tf.random.Algorithm(algorithm)  # noqa
    if algorithm in cls._state_size_dim_tags_cache:
      algo_state_dim = cls._state_size_dim_tags_cache[algorithm]
    else:
      # noinspection PyProtectedMember
      state_size = stateful_random_ops._get_state_size(algorithm)
      algo_state_dim = Dim(kind=Dim.Types.Feature, description="random_%s_state" % algo_type.name, dimension=state_size)
      cls._state_size_dim_tags_cache[algorithm] = algo_state_dim
    if out_dim:
      algo_state_dim.declare_same_as(out_dim)
    return Data(name="%s_output" % name, dim_tags=[algo_state_dim], dtype=stateful_random_ops.STATE_TYPE)

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param get_layer:
    """
    d.setdefault("from", ())
    super(RandomStateInitLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)


class RandomLayer(LayerBase):
  """
  Generates random numbers from uniform or normal or truncated normal distribution.

  This uses the TensorFlow stateless random ops internally, i.e. all the state handling is explicit.
  The state var can be explicitly provided and initialized via :class:`RandomStateInitLayer`,
  or when not provided it will be automatically created.

  There are two possible distinct use cases:

  - For any randomness in the model, e.g. dropout. So each ``session.run`` step will produce a new random number
    and advance the random state.
  - To initialize parameters via the config, using :class:`VariableLayer` with the ``init_by_layer`` option.
    This will only be called once when initializing the parameters.
    For this use case, we do not want to keep a random state var.
    You can just pass ``static=False``.
    Alternatively you could also pass the output of a :class:`RandomStateInitLayer` as ``state``.
  """
  layer_class = "random"

  def __init__(self, shape, distribution,
               mean=None, stddev=None, bound=None, minval=None, maxval=None, dtype="float32",
               seed=None,
               algorithm=None, explicit_state=None, auto_update_state=None, static=None,
               **kwargs):
    """
    :param typing.Sequence[Dim|int] shape:
    :param str distribution: "uniform", "normal" or "truncated_normal"
    :param int|float|LayerBase|None mean:
    :param int|float|LayerBase|None stddev:
    :param int|float|LayerBase|None bound: for uniform, defining the range [-bound, bound)
    :param int|float|LayerBase|None minval: for uniform
    :param int|float|LayerBase|None maxval: for uniform
    :param str dtype:
    :param int|list[int]|numpy.ndarray|None seed: If not given, uses self.network.random.randint,
      i.e. then it is controlled by the global seed setting, and every layer would get its own seed.
      If you specify it explicitly, make sure every :class:`RandomLayer` uses a different seed,
      otherwise you would get the same random numbers everywhere.
    :param str|tf.random.Algorithm|None algorithm: see :class:`RandomStateInitLayer`
    :param LayerBase|None explicit_state: You can pass the state explicitly here.
      If not given, will be created automatically, and updated automatically.
      You could pass a :class:`VariableLayer` with initial value via :class:`RandomStateInitLayer`,
      or directly a :class:`RandomStateInitLayer`.
      If auto_update_state is True, it must be a variable,
      and every time a new random number is created, this variable is updated.
      Otherwise (default) it will not be updated automatically.
    :param bool|None auto_update_state: only used when you pass an explicit state
    :param bool|None static: if no state at all should be used. it just relies on the seed then.
    """
    def _attrib_value(x):
      """
      :param LayerBase|T x:
      :rtype: tf.Tensor|T
      """
      if isinstance(x, LayerBase):
        assert x.output.batch_shape == ()  # expect scalars
        return x.output.placeholder
      return x

    shape  # noqa  # handled in get_out_data_from_opts
    super(RandomLayer, self).__init__(**kwargs)
    algorithm_int = RandomStateInitLayer.select_algorithm(algorithm)
    # Note: tf.random.Generator itself is just a wrapper around a state variable
    #   and corresponding stateless random ops.
    #   It's cheap to create multiple instances of this class e.g. for different random distributions
    #   when we reuse the same state var.

    # Need to derive custom class from tf.random.Generator to not require a tf.Variable in certain cases below.
    # noinspection PyAbstractClass
    class _RndGeneratorCustomState(tf.random.Generator):
      # noinspection PyShadowingNames
      def _create_variable(self, state, dtype, **_kwargs):
        return tf.cast(state, dtype) if state.dtype != dtype else state

      def skip(self, delta):
        """returns state without actually changing it"""
        return self.state

    # noinspection PyAbstractClass
    class _RndGeneratorStaticSeed(_RndGeneratorCustomState):
      # noinspection PyShadowingNames
      def __init__(self, seed, alg):
        from tensorflow.python.ops import stateful_random_ops
        state_ = tf.convert_to_tensor(stateful_random_ops.create_rng_state(seed=seed, alg=alg))
        super(_RndGeneratorStaticSeed, self).__init__(state=state_, alg=alg)
        assert self.state is state_

        # The underlying implementation in more recent TF versions uses gen_stateless_random_ops_v2
        # which is good.
        # Earlier TF versions uses gen_stateful_random_ops which requires a tf.Variable
        # so we cannot use it.
        # In that case, we can still do some fallback here for the case with a static seed.
        if tf_util.tf_version_tuple() < (2, 6, 0):
          assert algorithm is None, "%s: custom algorithm not supported on older TF version" % self
          for func_name in ["normal", "truncated_normal", "uniform"]:
            func = getattr(tf.random, "stateless_" + func_name)
            setattr(self, func_name, lambda _func=func, **kwargs_: _func(seed=seed[:2], **kwargs_))

    self.explicit_state = explicit_state
    if explicit_state is None:
      if static is None or static is False:
        assert auto_update_state is None or auto_update_state is True, (
          "%s: without explicit state, we always auto-update" % self)
        with self.var_creation_scope(), tf_util.default_control_flow_ctx():
          if seed is None:
            seed = self.network.random.randint(2 ** 31, size=[32], dtype="uint32")
          gen = tf.random.Generator.from_seed(seed=seed, alg=algorithm_int)
          self.add_param(gen.state)
      else:  # static is True
        assert static is True
        assert auto_update_state is None or auto_update_state is False, (
          "%s: in static mode, we can not auto-update" % self)
        if seed is None:
          seed = self.network.random.randint(2 ** 31, size=[32], dtype="uint32")
        gen = _RndGeneratorStaticSeed(seed=seed, alg=algorithm_int)

    else:  # state is not None
      assert static is None or static is False, "%s: state is given, thus it is not static" % self
      assert seed is None, "%s: explicit state and seed are mutually exclusive" % self
      state_ = explicit_state.output.placeholder
      if auto_update_state is True:
        state_ = tf_util.get_variable_from_tensor(state_)
        if not isinstance(state_, tf.Variable):
          tf_util.print_graph_output(state_, max_depth=5)
          raise Exception("%s: explicit_state %s is not a tf.Variable but %s" % (self, explicit_state, state_))
        gen = tf.random.Generator.from_state(state_, alg=algorithm_int)
      else:
        assert auto_update_state is None or auto_update_state is False
        gen = _RndGeneratorCustomState.from_state(state_, alg=algorithm_int)
      assert gen.state is state_
    self.random_generator = gen
    self._distribution_attribs = (mean, stddev, bound, minval, maxval)
    mean, stddev, bound, minval, maxval = [_attrib_value(attrib) for attrib in self._distribution_attribs]
    shape_ = [d.get_dim_value() for d in self.output.dim_tags]
    if distribution == "uniform":
      if minval is not None or maxval is not None:
        assert maxval is not None
        assert mean is None and stddev is None and bound is None
        if minval is None:
          minval = 0
      elif bound is not None:
        assert mean is None and stddev is None
        minval, maxval = -bound, bound
      elif mean is not None or stddev is not None:
        if mean is None:
          mean = 0
        assert stddev is not None
        import math
        _b = math.sqrt(3.) * stddev
        minval, maxval = -_b + mean, _b + mean
      else:
        raise ValueError("%s: uniform distribution needs either mean, stddev, bound or minval, maxval" % self)
      out = gen.uniform(shape=shape_, minval=minval, maxval=maxval, dtype=dtype)
    elif distribution in {"normal", "truncated_normal"}:
      assert minval is None and maxval is None and bound is None
      if mean is None:
        mean = 0
      if stddev is None:
        stddev = 1
      if distribution == "normal":
        out = gen.normal(shape=shape_, mean=mean, stddev=stddev, dtype=dtype)
      elif distribution == "truncated_normal":
        out = gen.truncated_normal(shape=shape_, mean=mean, stddev=stddev, dtype=dtype)
      else:
        assert False, distribution  # should not get here
    else:
      raise ValueError("%s: unknown distribution %r (or not implemented yet)" % (self, distribution))
    self.output.placeholder = out

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    deps = super(RandomLayer, self).get_dep_layers()
    if self.explicit_state:
      deps.append(self.explicit_state)
    for attrib in self._distribution_attribs:
      if isinstance(attrib, LayerBase):
        deps.append(attrib)
    return deps

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param get_layer:
    """
    d.setdefault("from", ())
    super(RandomLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    for attrib in ("mean", "stddev", "bound", "minval", "maxval", "explicit_state"):
      if attrib in d and isinstance(d[attrib], str):
        d[attrib] = get_layer(d[attrib])
    # We need special care in case this is inside a RecLayer.
    # https://github.com/rwth-i6/returnn/issues/1044
    # When this layer stays inside the loop, all is fine.
    # When it is moved out, we must add the loop dim
    # such that we get different random values in each loop iteration.
    # When the loop dim is not defined yet because of a dynamic loop ("end" layer),
    # this is not possible, so we must make sure the layer stays inside the loop.
    inside_rec_time_dim = network.get_inside_rec_time_dim(inside_loop=True)
    over_rec_time_dim = network.get_inside_rec_time_dim(inside_loop=False)
    if over_rec_time_dim:
      collocate_with = list(d.get("collocate_with", None) or [])
      collocate_with.append("end")  # in case the "end" layer is used, it will be collocated
      d["collocate_with"] = collocate_with
    if over_rec_time_dim and not inside_rec_time_dim:  # moved out of loop
      d["shape"] = [over_rec_time_dim] + list(d["shape"])

  @classmethod
  def get_out_data_from_opts(cls, name, shape, dtype="float32", **kwargs):
    """
    :param str name:
    :param typing.Sequence[Dim|int] shape:
    :param str dtype:
    :rtype: Data
    """
    dim_tags = [
      d if isinstance(d, Dim) else SpatialDim("%s:dim%i" % (name, i), d, auto_generated=True)
      for i, d in enumerate(shape)]
    return Data(name="%s_output" % name, dim_tags=dim_tags, dtype=dtype)


class RandIntLayer(LayerBase):
  """
  Generates random integer numbers using ``tf.random.uniform``.
  It is recommended to use :class:`RandomLayer` instead.
  """
  layer_class = "rand_int"

  # noinspection PyUnusedLocal
  def __init__(self, shape, maxval, minval=0, dtype="int32", sparse_dim=None, seed=None, **kwargs):
    """
    :param tuple[Dim|int]|list[Dim|int] shape: desired shape of output tensor
    :param int|LayerBase maxval: upper bound (exclusive) on range of random values
    :param int|LayerBase minval: lower bound (inclusive) on range of random values
    :param str dtype: type of the output. For random ints, int32 and int64 make sense, but could also be floats
    :param Dim|None sparse_dim:
    :param int|None seed: random seed
    """
    super(RandIntLayer, self).__init__(**kwargs)
    seed = seed if seed is not None else self.network.random.randint(2 ** 31)
    batch = self.output.batch or self.get_batch_info()
    shape_ = [
      d.get_for_batch_ctx(batch, self.network.control_flow_ctx).get_dim_value()
      for d in self.output.dim_tags]
    self.minval = minval
    self.maxval = maxval
    if isinstance(minval, LayerBase):
      assert minval.output.batch_shape == ()  # only scalars supported
    if isinstance(maxval, LayerBase):
      assert maxval.output.batch_shape == ()  # only scalars supported
    self.output.placeholder = tf.random.uniform(
      shape_,
      minval=minval.output.placeholder if isinstance(minval, LayerBase) else minval,
      maxval=maxval.output.placeholder if isinstance(maxval, LayerBase) else maxval,
      dtype=dtype, seed=seed)

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    deps = super(RandIntLayer, self).get_dep_layers()
    if isinstance(self.minval, LayerBase):
      deps.append(self.minval)
    if isinstance(self.maxval, LayerBase):
      deps.append(self.maxval)
    return deps

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param (str)->LayerBase get_layer:
    """
    d.setdefault("from", [])
    super(RandIntLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    if isinstance(d.get("maxval", None), str):
      d["maxval"] = get_layer(d["maxval"])
    if isinstance(d.get("minval", None), str):
      d["minval"] = get_layer(d["minval"])

  @classmethod
  def get_out_data_from_opts(cls, name, network, shape, maxval, minval=0, dtype="int32", sparse_dim=None,
                             **kwargs):
    """
    :param str name:
    :param returnn.tf.network.TFNetwork network:
    :param tuple[Dim|int]|list[Dim|int] shape: desired shape of output tensor
    :param int|LayerBase maxval: upper bound (exclusive) on range of random values
    :param int|LayerBase minval: lower bound (inclusive) on range of random values
    :param str dtype: type of the output. For random ints, int32 and int64 make sense, but could also be floats
    :param Dim|None sparse_dim:
    :rtype: Data
    """
    from returnn.tf.util.data import Dim
    dim_tags = []
    batch = None
    for i, d in enumerate(shape):
      if isinstance(d, Dim):
        pass  # good
      elif isinstance(d, int):
        d = Dim(
          kind=Dim.Types.Spatial if i < len(shape) - 1 else Dim.Types.Feature,
          description="%s:static:%i" % (name, i), auto_generated=True,
          dimension=d)
      else:
        raise TypeError("Layer %r: invalid type %s in shape %r" % (name, type(d), shape))
      if not batch and d.batch:
        batch = d.batch
      dim_tags.append(d)
    if not batch:
      batch = network.get_global_batch_info()
    ctx = network.control_flow_ctx
    dim_tags = [d.get_for_batch_ctx(batch, ctx) for d in dim_tags]
    return Data(
      name="%s_output" % name, dim_tags=dim_tags, dtype=dtype, sparse_dim=sparse_dim,
      batch=batch, control_flow_ctx=ctx)


class RangeLayer(LayerBase):
  """
  Generic wrapper around ``tf.range``.
  See also :class:`RangeInAxisLayer`.
  """
  layer_class = "range"

  # noinspection PyUnusedLocal
  def __init__(self, limit, start=0, delta=1, dtype=None, sparse=False, out_spatial_dim=None, **kwargs):
    """
    :param int|float limit:
    :param int|float start:
    :param int|float delta:
    :param str|None dtype:
    :param bool sparse:
    :param Dim|None out_spatial_dim:
    """
    out_spatial_dim  # noqa  # used in get_out_data_from_opts
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
  def get_out_data_from_opts(cls, name, limit, start=0, delta=1, dtype=None, sparse=False, out_spatial_dim=None,
                             **kwargs):
    """
    :param str name:
    :param int|float limit:
    :param int|float start:
    :param int|float delta:
    :param str|None dtype:
    :param bool sparse:
    :param Dim|None out_spatial_dim:
    :rtype: Data
    """
    if dtype is None:
      if any([float(arg) != int(arg) for arg in [start, limit, delta]]):
        dtype = "float32"
      else:
        dtype = "int32"
    dim = len(range(start, limit, delta))
    tag = Dim(kind=Dim.Types.Spatial, dimension=dim, description="%s:range" % name, auto_generated=True)
    if out_spatial_dim:
      tag.declare_same_as(out_spatial_dim)
    sparse_dim = None
    if sparse:
      sparse_dim = SpatialDim("%s:range-indices" % name, auto_generated=True)
    return Data(name="%s_output" % name, dim_tags=[tag], dtype=dtype, sparse_dim=sparse_dim)


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
    source_shape_dim = tf_util.get_shape_dim(source.placeholder, axis)
    out = tf.range(0, source_shape_dim, dtype=dtype)
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
    data_opts.pop("sparse_dim", None)
    if sparse:
      data_opts["dim"] = None
      data_opts["sparse_dim"] = dim_tags[0]
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
  def __init__(self, dtype="int32", sparse=False, out_spatial_dim=None, **kwargs):
    """
    :param str axis:
    :param str dtype:
    :param bool sparse:
    :param Dim|None out_spatial_dim:
    """
    out_spatial_dim  # noqa  # used in get_out_data_from_opts
    super(RangeFromLengthLayer, self).__init__(**kwargs)
    source = self.sources[0].output
    if not self.output.dim_tags[0].is_batch_dim():
      assert source.placeholder is self.output.dim_tags[0].dyn_size_ext.placeholder
    out = tf.range(0, tf.reduce_max(source.placeholder), dtype=dtype)
    self.output.placeholder = out

  @classmethod
  def get_out_data_from_opts(cls, name, sources, dtype="int32", sparse=False, out_spatial_dim=None, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param str dtype:
    :param bool sparse:
    :param Dim|None out_spatial_dim:
    """
    assert len(sources) == 1, "%s layer %r requires single source" % (cls, name)
    source = sources[0].output
    assert source.dtype in {"int32", "int64"}, (
      "%s %r: expect int32/int64 input but got %s from %s" % (cls.__name__, name, source.dtype, source))
    dim_tag = None
    if source.placeholder is not None:
      dim_tag = Dim.get_tag_from_size_tensor(source.placeholder)
    if not dim_tag:
      dim_tag = Dim(
        kind=Dim.Types.Spatial, description="%s_input_len" % name, auto_generated=True,
        batch=source.batch, control_flow_ctx=source.control_flow_ctx,
        dyn_size_ext=source)
      if source.placeholder is not None:
        dim_tag.set_tag_on_size_tensor(source.placeholder)
    if out_spatial_dim:
      dim_tag.declare_same_as(out_spatial_dim)
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
  def __init__(self, sources, value=0., shape=None, dtype=None, with_batch_dim=False, sparse_dim=None, **kwargs):
    """
    :param list[LayerBase] sources:
    :param int|float|bool|numpy.ndarray value:
    :param tuple[Dim|int]|list[Dim|int] shape: for verification, and defining dim tags
    :param str|None dtype:
    :param bool with_batch_dim:
    :param Dim|None sparse_dim:
    """
    import numpy
    assert not sources, "constant layer cannot have sources"
    super(ConstantLayer, self).__init__(**kwargs)
    shape_ = value.shape if isinstance(value, numpy.ndarray) else ()
    value = tf.constant(value, dtype=self.output.dtype)
    if len(shape_) == 0 and self.output.batch_ndim > 0:
      value = tf.fill([d.get_dim_value() for d in self.output.dim_tags], value)
    elif with_batch_dim:
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
  def get_out_data_from_opts(cls, name, value=0., shape=None, dtype=None, with_batch_dim=False, sparse_dim=None,
                             **kwargs):
    """
    :param str name:
    :param int|float|bool value:
    :param tuple[Dim|int]|list[Dim|int] shape: for verification, and defining dim tags
    :param str|None dtype:
    :param bool with_batch_dim:
    :param Dim|None sparse_dim:
    :rtype: Data
    """
    return Data.template_from_constant(
      value, name="%s_const" % name, shape=shape, dtype=dtype, with_batch_dim=with_batch_dim, sparse_dim=sparse_dim)


class GatingLayer(_ConcatInputLayer):
  """
  Splits the output into two equal parts, applies the gate_activation (sigmoid by default)
  on the one part, some other activation (e.g. tanh) on the other part and then
  element-wise multiplies them.
  Thus, the output dimension is input-dimension / 2.
  """
  layer_class = "gating"

  def __init__(self, activation, gate_activation="sigmoid", out_dim=None, **kwargs):
    """
    :param str activation:
    :param str gate_activation:
    :param Dim|None out_dim:
    """
    super(GatingLayer, self).__init__(out_dim=out_dim, **kwargs)
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
  def get_out_data_from_opts(cls, name, sources, n_out=NotSpecified, out_dim=None, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param int|None|NotSpecified n_out:
    :param Dim|None out_dim:
    :rtype: Data
    """
    input_data = get_concat_sources_data_template(sources)
    assert not input_data.sparse
    assert input_data.dim % 2 == 0
    out_dim_ = input_data.dim_tags[input_data.feature_dim_axis] // 2
    if out_dim:
      out_dim_.declare_same_as(out_dim)
    if n_out is not NotSpecified:
      assert n_out == input_data.dim // 2
    return Data(
      name="%s_output" % name,
      dtype=input_data.dtype,
      dim_tags=[
        out_dim_ if i == input_data.feature_dim_axis else d
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

  def __init__(self, window_size=None, window_dim=None, window_left=None, window_right=None,
               axis="T", out_spatial_dim=None, padding="same", stride=1, **kwargs):
    """
    :param int|None window_size:
    :param Dim|None window_dim:
    :param int|None window_left:
    :param int|None window_right:
    :param Dim|str axis: see :func:`Data.get_axis_from_description`
    :param Dim|None out_spatial_dim:
    :param str padding: "same" or "valid"
    :param int stride: return only each Nth window
    :param kwargs:
    """
    out_spatial_dim  # noqa  # via get_out_data_from_opts
    super(WindowLayer, self).__init__(**kwargs)
    if not window_size:
      assert window_dim and window_dim.dimension
      window_size = window_dim.dimension
    data = self.input_data.copy_as_batch_major()
    from returnn.tf.util.basic import is_axis_from_description_recurrent
    if is_axis_from_description_recurrent(axis=axis, network=self.network, data=data):
      # Inside RecLayer.
      assert self._rec_previous_layer, "%s: expected to be used inside a RecLayer" % self
      assert padding == "same"
      assert window_right is not None or window_left is not None, (
        "%s: recurrent variant should explicitly specify window_right=0 or window_left=window_size-1" % self)
      if window_left is not None:
        assert window_size == window_left + 1, "%s: recurrent variant can only have window into the past" % self
      if window_right is not None:
        assert window_right == 0, "%s: recurrent variant can only have window into the past" % self
      prev_state = self._rec_previous_layer.rec_vars_outputs["state"]  # (batch,window,...)
      next_state = tf.concat(
        [prev_state[:, 1:], tf.expand_dims(data.placeholder, axis=1)], axis=1)  # (batch,window,...)
      self.rec_vars_outputs["state"] = next_state
      self.output.placeholder = next_state

    else:
      axis = data.get_axis_from_description(axis)
      new_dim_axis = axis + 1  # new axis will be added right after

      from returnn.tf.util.basic import windowed_nd
      self.output.placeholder = windowed_nd(
        data.placeholder,
        window_size=window_size, window_left=window_left, window_right=window_right,
        padding=padding, time_axis=axis, new_window_axis=new_dim_axis, stride=stride)
    self.output.placeholder.set_shape(tf.TensorShape(self.output.batch_shape))

  @classmethod
  def get_out_data_from_opts(cls, name, network, sources, window_size=None, window_dim=None,
                             axis="T", out_spatial_dim=None, padding="same", stride=1,
                             **kwargs):
    """
    :param str name:
    :param returnn.tf.network.TFNetwork network:
    :param list[LayerBase] sources:
    :param int|None window_size:
    :param Dim|None window_dim:
    :param Dim|str axis:
    :param Dim|None out_spatial_dim:
    :param str padding:
    :param int stride:
    :rtype: Data
    """
    if not window_size:
      assert window_dim and window_dim.dimension
      window_size = window_dim.dimension
    data = get_concat_sources_data_template(sources)
    data = data.copy_template(name="%s_output" % name)
    data = data.copy_as_batch_major()
    from returnn.tf.util.basic import is_axis_from_description_recurrent
    if is_axis_from_description_recurrent(axis=axis, network=network, data=data):
      # Inside RecLayer.
      assert not out_spatial_dim
      new_dim_axis = 1  # after batch
    else:
      axis = data.get_axis_from_description(axis)
      in_spatial_dim = data.dim_tags[axis]
      out_spatial_dim_ = ConvLayer.calc_out_dim(
        in_dim=in_spatial_dim,
        filter_size=window_size, stride=stride, dilation_rate=1, padding=padding)
      assert isinstance(out_spatial_dim_, Dim)
      if out_spatial_dim:
        out_spatial_dim_.declare_same_as(out_spatial_dim)
      data = data.copy_template_replace_dim_tag(axis=axis, new_dim_tag=out_spatial_dim_)
      new_dim_axis = axis + 1  # add new axis right after
    window_dim_ = Dim(
      kind=Dim.Types.Spatial, description="%s:window" % name, dimension=window_size, auto_generated=True)
    if window_dim:
      window_dim_.declare_same_as(window_dim)
    return data.copy_add_dim_by_tag(axis=new_dim_axis, dim_tag=window_dim_, unbroadcast=True)

  # noinspection PyMethodOverriding
  @classmethod
  def get_rec_initial_extra_outputs(cls, network, batch_dim, rec_layer, window_size=None, window_dim=None,
                                    axis="T", sources=(), **kwargs):
    """
    :param returnn.tf.network.TFNetwork network:
    :param tf.Tensor batch_dim:
    :param returnn.tf.layers.rec.RecLayer|LayerBase rec_layer:
    :param int|None window_size:
    :param Dim|None window_dim:
    :param Dim|str axis:
    :param list[LayerBase] sources:
    :rtype: dict[str,tf.Tensor]
    """
    if not window_size:
      assert window_dim and window_dim.dimension
      window_size = window_dim.dimension
    data = get_concat_sources_data_template(sources)
    data = data.copy_as_batch_major()
    from returnn.tf.util.basic import is_axis_from_description_recurrent
    if is_axis_from_description_recurrent(axis=axis, network=network, data=data):
      # Inside RecLayer.
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
    from returnn.tf.util.basic import is_axis_from_description_recurrent
    if is_axis_from_description_recurrent(axis=axis, network=self.network, data=data):
      assert not reverse
      prev_state = self._rec_previous_layer.rec_vars_outputs["state"]
      next_state = prev_state + x
      self.rec_vars_outputs["state"] = next_state
      self.output.placeholder = next_state
    else:
      axis = data.get_axis_from_description(axis)
      y = tf.cumsum(x, axis=axis, reverse=reverse)
      if self._initial_output is not None:
        init_state = self.get_rec_initial_output(
          name=self.name, output=self.output,
          network=self.network, batch_dim=self.get_batch_dim(),
          rec_layer=self.network.get_rec_parent_layer(inside_loop=False),  # should actually not matter
          axis=axis, sources=self.sources,
          initial_output=self._initial_output)
        y = init_state + y
      self.output.placeholder = y
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

  # noinspection PyMethodOverriding
  @classmethod
  def get_rec_initial_extra_outputs(cls, network, batch_dim, rec_layer, axis="T", sources=(), **kwargs):
    """
    :param returnn.tf.network.TFNetwork network:
    :param tf.Tensor batch_dim:
    :param returnn.tf.layers.rec.RecLayer|LayerBase rec_layer:
    :param str axis:
    :param list[LayerBase] sources:
    :rtype: dict[str,tf.Tensor]
    """
    data = get_concat_sources_data_template(sources)
    from returnn.tf.util.basic import is_axis_from_description_recurrent
    if is_axis_from_description_recurrent(axis=axis, network=network, data=data):
      assert all(data.shape)
      init_state = cls.get_rec_initial_output(
        network=network, batch_dim=batch_dim, rec_layer=rec_layer, axis=axis, sources=sources, **kwargs)
      return {"state": init_state}
    return {}


class PadLayer(_ConcatInputLayer):
  """
  Adds (e.g. zero) padding in some axis or axes.
  Also see :class:`PrefixInTimeLayer` for dynamic dims.
  """
  layer_class = "pad"

  def __init__(self, axes, padding, out_dims=None, value=0, mode="constant", **kwargs):
    """
    :param Dim|str|list[Dim|str] axes: e.g. "F" etc. see :func:`Data.get_axes_from_description`.
    :param list[(int,int)]|(int,int)|int padding: how much to pad left/right in each axis
    :param Dim|list[Dim]|None out_dims:
    :param int|float value: what constant value to pad, with mode=="constant"
    :param str mode: "constant", "reflect", "symmetric" and "replication"
    """
    out_dims  # noqa  # handled in get_out_data_from_opts
    super(PadLayer, self).__init__(**kwargs)
    axes_ = self.input_data.get_axes_from_description(axes)
    assert axes_, "%s: invalid axes %r in input %s" % (self, axes, self.input_data)
    axes = axes_
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
  def get_out_data_from_opts(cls, name, sources, axes, padding, out_dims=None, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param Dim|str|list[Dim|str] axes:
    :param list[(int,int)]|(int,int)|int padding:
    :param Dim|list[Dim]|None out_dims:
    :rtype: Data
    """
    from ..util.data import Dim
    data = get_concat_sources_data_template(sources)
    data.name = "%s_output" % name
    # Make sure that we do not map one axis description to multiple axes,
    # and use always get_axis_from_description, not get_axes_from_description.
    if isinstance(axes, (list, tuple)):
      axes = [data.get_axis_from_description(a) for a in axes]
    else:
      axes = [data.get_axis_from_description(axes)]
    padding = cls._transform_padding(padding=padding, axes=axes)
    if out_dims:
      if isinstance(out_dims, (list, tuple)):
        assert len(out_dims) == len(axes) == len(padding)
        assert all(isinstance(d, Dim) for d in out_dims)
      else:
        assert isinstance(out_dims, Dim)
        assert len(axes) == len(padding) == 1
        out_dims = [out_dims]
    dim_tags = list(data.dim_tags)
    for i, a in enumerate(axes):
      pad_left, pad_right = padding[i]
      out_dim = pad_left + dim_tags[a] + pad_right
      if out_dims:
        out_dim.declare_same_as(out_dims[i])
      dim_tags[a] = out_dim
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

  def __init__(self, axes, keep_order=NotSpecified, n_out=None, out_dim=None, **kwargs):
    """
    :param typing.Sequence[Dim|str] axes: see :func:`Data.get_axis_from_description`
    :param bool|NotSpecified keep_order: The old default was: the axes are sorted, and then merged.
      Thus, the order of incoming axes will influence the result.
      E.g. inputs [B,S,F] and [B,F,S], with ``axes=["S","F"]``, will get different results,
      although the output shape is [B,S*F] in both cases.
      This is bad: In general, other layers in RETURNN might reorder the axes for various reasons,
      and all layers should behave in the same way, no matter the order.
      It is recommended to set ``keep_order=True``, such that the order defined in ``axes`` defines the behavior,
      and not the incoming axis order.
      Since behavior version 6, this is already the case.
    :param int|None n_out:
    :param Dim|None out_dim:
    """
    out_dim  # noqa  # handled in get_out_data_from_opts
    from returnn.util import BehaviorVersion
    super(MergeDimsLayer, self).__init__(**kwargs)
    if keep_order is NotSpecified:
      keep_order = True if BehaviorVersion.get() >= 6 else False
    BehaviorVersion.require(
      condition=keep_order, message="MergeDimsLayer, only keep_order=True is allowed", version=6)
    axes = self._get_merge_axes(axes=axes, keep_order=keep_order, input_data=self.input_data, name=self)
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
  def _get_merge_axes(cls, axes, keep_order, input_data, name):
    """
    :param typing.Sequence[Dim|str] axes:
    :param bool keep_order:
    :param Data input_data:
    :param name:
    :rtype: list[int]
    """
    if keep_order:
      assert isinstance(axes, (tuple, list, typing.Sequence)) and not isinstance(axes, str), (
        "%s: axes %r must be a list or tuple, to have a well defined order in input %s" % (name, axes, input_data))
      axes_ = []
      for axis in axes:
        axis_ = input_data.get_axes_from_description(axis, allow_int=False)
        assert len(axis_) <= 1, (
          "%s: unique axes %r required in input %s, but got %r -> %r" % (name, axes, input_data, axis, axis_))
        axes_ += axis_
      return axes_
    else:
      axes = input_data.get_axes_from_description(axes)
      return sorted(axes)

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
        new_data = Data.get_common_data([out_size, in_size], name="%s_output" % self.name)
        new_data.placeholder = (
          out_size.copy_compatible_to(new_data).placeholder
          * in_size.copy_compatible_to(new_data).placeholder)
        out_size = new_data
    if not out_size:
      out_size = Data.from_tensor(tf.constant(1, dtype=tf.int32))
    target_tag.dyn_size_ext = out_size

  @classmethod
  def get_out_data_from_opts(cls, name, axes, keep_order=NotSpecified,
                             sources=(), n_out=NotSpecified, out_type=None, out_dim=None, **kwargs):
    """
    :param str name:
    :param typing.Sequence[Dim|str] axes:
    :param bool|NotSpecified keep_order:
    :param list[LayerBase] sources:
    :param int|None|NotSpecified n_out:
    :param None|dict[str] out_type:
    :param Dim|None out_dim:
    :rtype: Data
    """
    from returnn.util import BehaviorVersion
    from returnn.util.basic import prod
    if keep_order is NotSpecified:
      keep_order = True if BehaviorVersion.get() >= 6 else False
    assert not out_type, "currently ignored"
    input_data = get_concat_sources_data_template(sources)
    axes = cls._get_merge_axes(
      axes=axes, keep_order=keep_order, input_data=input_data, name="%s layer %r" % (cls, name))
    data = input_data.copy(name="%s_output" % name)
    if len(axes) <= 1:
      return data
    res_dim = None
    if all([data.batch_shape[i] is not None for i in axes]):
      res_dim = int(prod([data.batch_shape[i] for i in axes]))
    merge_dim_tags = [data.dim_tags[axis] for axis in axes]
    merge_target_axis = cls._get_target_axis(input_data=data, merge_axes=axes)
    out_dim_ = prod(merge_dim_tags)
    assert isinstance(out_dim_, Dim)
    assert out_dim_.dimension == res_dim
    if out_dim:
      out_dim_.declare_same_as(out_dim)
    new_dim_tags = [d for (i, d) in enumerate(data.dim_tags) if i not in axes]
    new_dim_tags.insert(merge_target_axis, out_dim_)

    data_opts = data.get_kwargs(include_special_axes=False)
    data_opts["dim_tags"] = new_dim_tags
    data = Data(**data_opts)

    new_feature_dim_axis = cls._old_axis_to_new_axis(
      input_data=input_data, merge_axes=axes, old_axis=input_data.feature_dim_axis)
    if new_feature_dim_axis == data.batch_dim_axis:
      new_feature_dim_axis = None
    data.time_dim_axis = cls._old_axis_to_new_axis(
      input_data=input_data, merge_axes=axes, old_axis=input_data.time_dim_axis)
    if data.time_dim_axis is not None and data.time_dim_axis in {data.batch_dim_axis, new_feature_dim_axis}:
      if input_data.time_dim_axis not in {input_data.batch_dim_axis, input_data.feature_dim_axis}:
        # Time got merged with feature or batch.
        data.time_dim_axis = None
    if data.feature_dim_axis != new_feature_dim_axis or input_data.feature_dim_axis_or_unspecified is not NotSpecified:
      data.feature_dim_axis = new_feature_dim_axis  # explicitly set
      data.dim = data.batch_shape[data.feature_dim_axis] if data.feature_dim_axis is not None else None

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

  def __init__(self, axis=None, num_splits=None, size_splits=None, out_dims=None, **kwargs):
    """
    :param str|None axis: feature axis by default
    :param int|None num_splits:
    :param list[int]|None size_splits:
    :param list[Dim]|None out_dims:
    """
    assert num_splits or size_splits or out_dims, "%s: provide either num_splits or size_splits or out_dims" % self
    super(SplitLayer, self).__init__(**kwargs)
    self.output = self.input_data
    self.axis, self.out_dims = self._get_axis_size_splits_num_splits(
      input_data=self.input_data, axis=axis,
      num_splits=num_splits, size_splits=size_splits, out_dims=out_dims,
      err_prefix=self, name=self.name)
    self.splits = tf.split(self.output.placeholder, [d.dimension for d in self.out_dims], axis=self.axis)
    assert len(self.splits) == len(self.out_dims)
    self._sub_layers = {"%i" % i: self._make_split_layer(i) for i in range(len(self.splits))}

  @classmethod
  def _get_axis_size_splits_num_splits(cls, name, input_data, axis=None,
                                       num_splits=None, size_splits=None, out_dims=None,
                                       err_prefix=None):
    """
    :param str name:
    :param Data input_data:
    :param str|None axis: feature axis by default
    :param int|None num_splits:
    :param list[int]|None size_splits:
    :param list[Dim]|None out_dims:
    :param object err_prefix:
    :return: axis, out_dims
    :rtype: (int, list[Dim])
    """
    assert num_splits or size_splits or out_dims, (
      "%s: provide either num_splits or size_splits or out_dims" % err_prefix)
    if axis is None:
      axis = "feature"
    axis = input_data.get_axis_from_description(axis, allow_int=False)
    dim = input_data.batch_shape[axis]
    assert isinstance(dim, int), "%s: expects static axis %s in %r" % (err_prefix, axis, input_data)
    if num_splits:
      assert dim % num_splits == 0, "%s: expects multiple of %i in dim %i in %r" % (
        err_prefix, num_splits, dim, input_data)
      size_splits = [dim // num_splits for _ in range(num_splits)]
    elif size_splits:
      if not isinstance(size_splits, (list, tuple)):
        raise TypeError("%s: invalid type size_splits %r" % (err_prefix, size_splits))
      size_splits = list(size_splits)
      assert sum(size_splits) == dim, "%s: invalid size_splits %r for dim %i in %r" % (
        err_prefix, size_splits, dim, input_data)
    elif out_dims:
      assert all(isinstance(d, Dim) for d in out_dims)
      assert sum(d.dimension for d in out_dims) == dim, "%s: invalid out_dims %r for dim %i in %r" % (
        err_prefix, out_dims, dim, input_data)
    if not out_dims:
      assert size_splits
      out_dims = [
        Dim(
          kind=input_data.dim_tags[axis].kind, description="%s_split%i" % (name, idx),
          dimension=size_splits[idx], auto_generated=True)
        for idx in range(len(size_splits))]
    return axis, out_dims

  def _make_split_layer(self, idx):
    """
    :param int idx:
    :rtype: LayerBase
    """
    out = self._get_split_out_data(
      name=self.name, idx=idx, out_dims=self.out_dims, input_data=self.input_data, axis=self.axis)
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
    :return: Data template, class type of sub-layer, layer opts (transformed)
    :rtype: (Data, type, dict[str])|None
    """
    try:
      idx = int(layer_name)
    except ValueError:
      return None
    name = parent_layer_kwargs.get("name", "<unknown>")
    input_data = get_concat_sources_data_template(parent_layer_kwargs["sources"], name="%s_output" % name)
    axis, out_dims = cls._get_axis_size_splits_num_splits(
      input_data=input_data,
      axis=parent_layer_kwargs.get("axis", None),
      num_splits=parent_layer_kwargs.get("num_splits", None),
      size_splits=parent_layer_kwargs.get("size_splits", None),
      out_dims=parent_layer_kwargs.get("out_dims", None),
      err_prefix="%s/%s" % (name, layer_name),
      name=name)
    out = cls._get_split_out_data(
      name=name, idx=idx, input_data=input_data, out_dims=out_dims, axis=axis)
    return out, InternalLayer, {}

  @classmethod
  def _get_split_out_data(cls, name, input_data, out_dims, idx, axis):
    """
    :param str name:
    :param Data input_data:
    :param list[Dim] out_dims:
    :param int idx:
    :param int axis:
    :rtype: Data
    """
    new_dim_tag = out_dims[idx]
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
    :param Dim|str axis: e.g. "F"
    :param tuple[Dim|int]|list[Dim|int] dims: what the axis should be split into. e.g. (window, -1)
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
    old_dim = data.dim_tags[axis]
    if pad_to_multiples is None:
      pad_to_multiples = data.is_axis_dynamic(axis)

    from returnn.tf.util.basic import get_shape
    old_shape = get_shape(data.placeholder)
    if dims and any(isinstance(d, Dim) for d in dims):
      assert all(isinstance(d, Dim) for d in dims)
      dims_ = []
      for d in dims:
        if d.is_batch_dim():
          dims_.append(-1)
        elif d.dimension is not None:
          dims_.append(d.dimension)
        else:
          d.complete_dyn_size()
          if d.dyn_size is not None:
            dims_.append(tf.reduce_max(d.dyn_size))
          else:
            dims_.append(-1)
      assert len(dims_) == len(dims)
      assert len([d for d in dims_ if isinstance(d, int) and d == -1]) <= 1, "%s: dims %r invalid" % (self, dims)
      dims = dims_
    new_shape = old_shape[:axis] + list(dims) + old_shape[axis + 1:]
    assert len(new_shape) == len(self.output.batch_shape)
    for i in range(len(new_shape)):
      if new_shape[i] == -1 and self.output.batch_shape[i] is not None:
        new_shape[i] = self.output.batch_shape[i]

    from returnn.util import basic as util
    new_pos_dims = [d for d in dims if isinstance(d, int) and d > 0]
    rem_const_size = None
    if len(new_pos_dims) == len(dims) - 1:
      rem_const_size = util.prod(new_pos_dims)
    assert not data.is_axis_dynamic(axis) or pad_to_multiples or rem_const_size == 1
    if pad_to_multiples and (not isinstance(rem_const_size, int) or rem_const_size != 1):
      indices = [i for i, d in enumerate(dims) if isinstance(d, int) and d == -1]
      assert len(indices) == 1, "%s: exactly one -1 dim in %r expected" % (self, dims)
      if rem_const_size is None:
        rem_const_size = util.prod([d for d in dims if not isinstance(d, int) or d > 0])
      old_size = old_shape[axis]
      pad_size = (-old_size) % rem_const_size

      paddings = [(0, 0)] * axis + [(0, pad_size)] + [(0, 0)] * (data.batch_ndim - axis - 1)
      data.placeholder = tf.pad(data.placeholder, paddings=paddings, constant_values=pad_value)

      rem_dim_idx = dims.index(-1)
      rem_dim = self.output.get_dim_tag(axis + rem_dim_idx)
      if rem_dim.dimension is None and not rem_dim.is_batch_dim():
        assert old_dim.dimension is None
        assert old_dim.dyn_size_ext is not None
        rem_dim.dyn_size_ext = old_dim.dyn_size_ext.copy(name="%s_dyn_size_ext" % (rem_dim.description or self.name))
        rem_dim.dyn_size_ext.placeholder = -(
          -data.get_dynamic_size(axis) // rem_const_size)  # == ceildiv(size, constant_size)
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
  def _map_old_axis_to_new_axis(cls, split_axis, dims, rem_dim_idx,
                                old_axis, old_dim, kind, use_remaining=True, split_offset=None):
    """
    :param int split_axis:
    :param tuple[returnn.tf.util.data.Dim] dims: might include -1
    :param int|None rem_dim_idx:
    :param int old_axis:
    :param returnn.tf.util.data.Dim old_dim:
    :param returnn.tf.util.data.Dim.Types kind:
    :param bool use_remaining: whether to make use of -1 in dims
    :param int|None split_offset:
    :rtype: int
    """
    if old_axis < split_axis:
      return old_axis
    if old_axis > split_axis:
      return old_axis + len(dims) - 1
    assert old_axis == split_axis
    if use_remaining:
      if rem_dim_idx is not None:
        return split_axis + rem_dim_idx
      if len([d for d in dims if d == old_dim]) == 1:
        return split_axis + [i for i, d in enumerate(dims) if old_dim][0]
      if len([d for d in dims if d.kind == kind]) == 1:
        return split_axis + [i for i, d in enumerate(dims) if d.kind == kind][0]
    assert split_offset is not None
    if any(d.kind == kind for d in dims):
      indices, dims = zip(*[(i, d) for i, d in enumerate(dims) if d.kind == kind])
    else:
      indices = list(range(len(dims)))
    if split_offset < 0:
      split_offset += len(indices)
    assert 0 <= split_offset < len(indices)
    return split_axis + indices[split_offset]

  @classmethod
  def get_out_data_from_opts(cls, name, axis, dims, pad_to_multiples=None, sources=(), **kwargs):
    """
    :param str name:
    :param Dim|str axis:
    :param list[Dim|int]|tuple[Dim|int] dims:
    :param bool|None pad_to_multiples:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    from ..util.data import Dim
    input_data = get_concat_sources_data_template(sources)
    data = input_data.copy("%s_output" % name)
    if isinstance(axis, int):
      data = data.copy_as_batch_major()
    axis = data.get_axis_from_description(axis)
    if pad_to_multiples is None:
      pad_to_multiples = data.is_axis_dynamic(axis)

    axis_dim_tag = data.dim_tags[axis]
    rem_dim_indices = [
      i for i, d in enumerate(dims)
      if (isinstance(d, int) and d < 0)
      or (isinstance(d, Dim) and d == axis_dim_tag)]

    resolved_dims = None
    if dims and isinstance(dims[0], Dim):
      assert all(isinstance(d, Dim) for d in dims)
      resolved_dims = tuple(dims)
      dims = [d.dimension or -1 for d in dims]
    if len(rem_dim_indices) == 0:
      rem_dim_indices = [i for i, d in enumerate(dims) if d < 0]
    rem_dim_idx = rem_dim_indices[0] if len(rem_dim_indices) == 1 else None
    if data.batch_shape[axis] is not None:
      resolved_shape_dims = cls._resolve_dims(
        old_dim=data.batch_shape[axis], new_dims=dims, pad_to_multiples=pad_to_multiples)
    else:
      resolved_shape_dims = tuple([(d if (d >= 0) else None) for d in dims])

    if resolved_dims and any(d.is_batch_dim() for d in resolved_dims):
      assert len([d for d in resolved_dims if d.is_batch_dim()]) == 1
      # In case one of the resolved dims is a batch dim, the dim is unclear,
      # as the batch dim can be arbitrary -- it does not necessarily need to be the global batch dim.
      # Any other dim tag should clearly define the dim, so we can infer the batch dim
      # This logic is done in __init__.
    if not resolved_dims or len(rem_dim_indices) <= 1:
      assert len(rem_dim_indices) <= 1, "only one entry in dims %r can be -1" % (dims,)
      if len(rem_dim_indices) == 1:
        import numpy
        n = int(numpy.prod([dim for i, dim in enumerate(dims) if dim > 0 and i not in rem_dim_indices]))
        if n == 1:
          rem_dim = axis_dim_tag
        else:  # need to create a new rem_dim
          rem_dim = Dim(
            kind=axis_dim_tag.kind,
            description="%s_split_dims%i_rem" % (name, rem_dim_idx),
            dimension=resolved_shape_dims[rem_dim_idx],
            auto_generated=True,
            derived_from_tag=axis_dim_tag,
            batch=axis_dim_tag.batch, control_flow_ctx=axis_dim_tag.control_flow_ctx)
          if rem_dim.dimension is None and axis_dim_tag.dyn_size_ext is not None:
            rem_dim.dyn_size_ext = axis_dim_tag.dyn_size_ext.copy_template(name="%s_split_dims%i" % (name, rem_dim_idx))
        if resolved_dims:
          if rem_dim.dimension is not None:
            assert rem_dim.dimension * n == axis_dim_tag.dimension
          rem_dim.declare_same_as(resolved_dims[rem_dim_idx])
      else:
        assert len(rem_dim_indices) == 0
        rem_dim = None
      if not resolved_dims:
        resolved_dims = tuple(
          Dim(
            kind=axis_dim_tag.kind if not axis_dim_tag.is_batch_dim() else Dim.Types.Spatial,
            description="%s_split_dims%i" % (name, i),
            dimension=shape_dim, auto_generated=True)
          if rem_dim is None or i != rem_dim_idx else rem_dim
          for i, shape_dim in enumerate(resolved_shape_dims))
    out_batch = data.batch
    if axis_dim_tag.is_batch_dim():
      assert len([d for d in resolved_dims if d.is_batch_dim()]) == 1
      from returnn.tf.util.data import BatchInfo
      out_batch = data.batch
      remaining = set(resolved_dims)
      for beam_virtual_dim in reversed(data.batch.virtual_dims):
        if isinstance(beam_virtual_dim, BatchInfo.FixedDim):
          if beam_virtual_dim.dim_tag in remaining:
            out_batch = out_batch.copy_remove_dim(beam_virtual_dim)
            remaining.remove(beam_virtual_dim.dim_tag)
      if remaining:
        # cannot really handle that well yet but this fallback might be ok
        out_batch = out_batch.get_global_base()

    new_dim_tags = data.dim_tags[:axis] + resolved_dims + data.dim_tags[axis + 1:]
    out = data.copy_template_new_dim_tags(new_dim_tags)
    if data.time_dim_axis is None:
      out.time_dim_axis = None
    else:
      out.time_dim_axis = cls._map_old_axis_to_new_axis(
        split_axis=axis, dims=resolved_dims, rem_dim_idx=rem_dim_idx,
        old_axis=data.time_dim_axis, old_dim=data.dim_tags[data.time_dim_axis],
        kind=Dim.Types.Spatial, use_remaining=True, split_offset=0)
    if data.feature_dim_axis is not None:
      expected_out_feature_dim_axis = cls._map_old_axis_to_new_axis(
        split_axis=axis, dims=resolved_dims, rem_dim_idx=rem_dim_idx,
        old_axis=data.feature_dim_axis, old_dim=data.dim_tags[data.feature_dim_axis],
        kind=Dim.Types.Feature,
        # use_remaining=False, split_offset=-1 because of https://github.com/rwth-i6/returnn/issues/704
        use_remaining=False, split_offset=-1)
      if out.feature_dim_axis != expected_out_feature_dim_axis:  # maybe due to non-specified default behavior
        out.feature_dim_axis = expected_out_feature_dim_axis
        out.dim = out.batch_shape[out.feature_dim_axis]
    out.batch = out_batch
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


class UnflattenBatchLayer(_ConcatInputLayer):
  """
  Inverse of :class:`FlattenBatchLayer`, so recovers an axis previously merged into the batch axis

  This basically wraps :func:`unflatten_with_seq_len_mask`.
  """
  layer_class = "unflatten_batch"

  def __init__(self, **kwargs):
    super(UnflattenBatchLayer, self).__init__(**kwargs)
    x = self.input_data.copy_as_batch_major()
    from ..util.data import BatchInfo
    batch_info = x.get_batch_dim_tag().batch
    seq_lens = [v.sizes for v in batch_info.virtual_dims if isinstance(v, BatchInfo.PackedDim)]
    assert len(seq_lens) == 1, "only implemented for single packed dim"
    seq_lens = seq_lens[0]
    self.output.placeholder = tf_util.unflatten_with_seq_len_mask(
      x.placeholder, seq_lens=seq_lens, batch_major=self.output.is_batch_major)

  @classmethod
  def get_out_data_from_opts(cls, sources, name, **kwargs):
    """
    :param list[LayerBase] sources:
    :param str name:
    :rtype: Data
    """
    out = get_concat_sources_data_template(sources, name="%s_output" % name)
    out = out.copy_as_batch_major()
    out = out.copy_template_unpack_batch()
    return out


class UnflattenNdLayer(_ConcatInputLayer):
  """
  This keeps the batch axis as-is, i.e. the flattening/unflattening did not happen on the batch axis.

  Example:

    Assumes that the input is of shape (B,T,<Ds>) which represents flattened images,
    where each image is of size width * height.
    We additionally provide these image sizes (shape (B,2)), i.e. (width,height) tuples.
    We return the unflattened images of shape (B,W,H,<Ds>), where W/H are the max width/height.

  This basically wraps :func:`returnn.tf.util.basic.unflatten_nd`.
  """
  layer_class = "unflatten_nd"
  recurrent = True

  def __init__(self, sizes, num_axes, in_dim="T", out_dims=None, declare_same_sizes_as=None, **kwargs):
    """
    :param LayerBase sizes:
    :param int num_axes:
    :param Dim|str|None in_dim:
    :param list[Dim]|None out_dims:
    :param dict[int,LayerBase]|None declare_same_sizes_as:
    """
    out_dims, declare_same_sizes_as  # noqa  # handled in get_out_data_from_opts
    super(UnflattenNdLayer, self).__init__(**kwargs)
    self.sizes = sizes
    self.declare_same_sizes_as = declare_same_sizes_as
    input_data = self.input_data.copy_as_batch_major()
    axis = input_data.get_axis_from_description(in_dim, allow_int=False)
    input_data = input_data.copy_move_axis(old_axis=axis, new_axis=1)
    axis = 1
    out_dims = self.output.dim_tags[axis:axis + num_axes]
    sizes_data = sizes.output.copy_as_batch_major()
    assert sizes_data.batch_ndim == 2
    assert sizes_data.batch_shape[1] in (None, num_axes)  # also allow None...
    for i, out_dim in enumerate(out_dims):
      if out_dim.dimension is None and out_dim.dyn_size is None:
        out_dim.dyn_size = sizes_data.placeholder[:, i]
    self.output.placeholder = tf_util.unflatten_nd(input_data.placeholder, sizes_data.placeholder, num_axes=num_axes)

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
  def get_out_data_from_opts(cls, name, sources, num_axes, in_dim="T", out_dims=None, declare_same_sizes_as=None,
                             **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param int num_axes:
    :param Dim|str|None in_dim:
    :param list[Dim]|None out_dims:
    :param dict[int,LayerBase]|None declare_same_sizes_as:
    :rtype: Data
    """
    out = get_concat_sources_data_template(sources).copy(name="%s_output" % name).copy_as_batch_major()
    axis = out.get_axis_from_description(in_dim, allow_int=False)
    out = out.copy_move_axis(old_axis=axis, new_axis=1)
    axis = 1
    out = out.copy_template_excluding_axis(axis)
    if out_dims:
      assert len(out_dims) == num_axes
      assert all(isinstance(d, Dim) for d in out_dims)
      assert not declare_same_sizes_as
    else:
      out_dims = [
        SpatialDim("%s:unflatten-nd:%i" % (name, i), auto_generated=True)
        for i in range(num_axes)]
      if declare_same_sizes_as:
        for i, other in declare_same_sizes_as.items():
          assert 0 <= i < num_axes
          out_dims[i] = other.output.get_time_dim_tag()
    for i, out_dim in enumerate(out_dims):
      out = out.copy_add_dim_by_tag(axis=axis + i, unbroadcast=True, dim_tag=out_dim)
    return out


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
    :param int|Dim dim: dimension of new axis (1 by default)
    """
    super(ExpandDimsLayer, self).__init__(**kwargs)
    data = self.input_data
    if isinstance(axis, int):
      data = data.copy_as_batch_major()
    axis = self._get_axis(data=data, axis=axis)
    from returnn.tf.util.basic import expand_dims_unbroadcast
    self.output.placeholder = expand_dims_unbroadcast(
      data.placeholder, axis=axis, dim=dim.get_dim_value() if isinstance(dim, Dim) else dim)

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
    :param int|Dim dim:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    init_axis = axis
    data = get_concat_sources_data_template(sources)
    if isinstance(axis, int):
      data = data.copy_as_batch_major()
    axis = cls._get_axis(data=data, axis=axis)

    if isinstance(dim, Dim):
      new_dim = dim
    else:
      new_dim = Dim(
        kind=Dim.Types.Feature if init_axis.lower() == "f" else Dim.Types.Spatial,
        description="%s_expand_dims" % name, auto_generated=True,
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

  def __init__(self, repetitions, axis="T", out_dim=None, **kwargs):
    """
    :param LayerBase|int repetitions:
      number of repetitions for each sequence and position in target axis.
      Can be [B,T] or [T,B] or some subset of that shape
    :param Dim|str axis: (dynamic) axis for repetition (currently only time axis is supported)
    :param Dim|None out_dim:
    """
    super(RepeatLayer, self).__init__(out_dim=out_dim, **kwargs)
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

    dyn_axis = repetitions_data.get_axes(exclude_batch=True)[0]  # can only have one axis
    if repetitions_data.has_dynamic_size(dyn_axis) and repetitions_data.batch_ndim >= 2:
      # apply correct zero masking for repetitions
      mask = repetitions_data.get_sequence_mask_broadcast(axis=dyn_axis)
      zeros = tf.zeros((), dtype=repetitions_data.placeholder.dtype)
      repetitions_placeholder = tf_util.where_bc(
        mask, repetitions_placeholder, zeros, name="repetitions_masked_axis_%i" % dyn_axis)

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
    output_axis = self.output.get_batch_axis(0)
    tag = self.output.dim_tags[output_axis]
    if tag.dimension is None and tag.dyn_size is None:  # dynamic? dyn sizes needed?
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
  def get_out_data_from_opts(cls, name, sources, axis, repetitions, out_dim=None, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param Dim|str axis:
    :param LayerBase|int repetitions:
    :param Dim|None out_dim:
    :rtype: Data
    """
    from ..util.data import Dim
    data = get_concat_sources_data_template(sources, name="%s_output" % name)
    if data.have_batch_axis():
      data = data.copy_as_batch_major()
    elif isinstance(repetitions, LayerBase) and repetitions.output.have_batch_axis():
      data = data.copy_add_batch_dim(batch_dim_axis=0, batch=repetitions.output.batch)
    original_axis = data.get_axis_from_description(axis, allow_int=False)
    tag = data.dim_tags[original_axis]
    data = data.copy_move_axis(original_axis, data.get_batch_axis(0))
    if isinstance(repetitions, int):
      out_dim_ = tag * repetitions
    else:
      out_dim_ = Dim(description="repeated:%s" % name, kind=tag.kind, derived_from_tag=tag, auto_generated=True)
    if out_dim:
      out_dim_.declare_same_as(out_dim)
    if tag.dyn_size_ext:
      if data.batch:
        out_dim_ = out_dim_.get_for_batch_ctx(data.batch, data.control_flow_ctx)
      out_dim_.dyn_size_ext = tag.dyn_size_ext.copy_template()
    return data.copy_template_replace_dim_tag(axis=data.get_batch_axis(0), new_dim_tag=out_dim_)


class TileLayer(_ConcatInputLayer):
  """
  A wrapper around tf.tile
  """
  layer_class = "tile"

  def __init__(self, multiples, out_dims=None, **kwargs):
    """
    :param dict[Dim|str, int] multiples: number of multiples per axis (axis provided as dim tag or str desc)
    :param dict[Dim|str, Dim]|None out_dims:
    """
    out_dims  # noqa  # handled in get_out_data_from_opts
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
  def get_out_data_from_opts(cls, name, sources, multiples, out_dims=None, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param dict[Dim|str, int] multiples:
    :param dict[Dim|str, Dim]|None out_dims:
    :rtype: Data
    """
    data = get_concat_sources_data_template(sources, name="%s_output" % name)
    dim_tags = list(data.dim_tags)
    for axis, multiple in multiples.items():
      axis_int = data.get_axis_from_description(axis, allow_int=False)
      tag = multiple * dim_tags[axis_int]
      if out_dims and axis in out_dims:
        tag.declare_same_as(out_dims[axis])
      dim_tags[axis_int] = tag
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
      out.sparse_dim = None
    return out


class SwapAxesLayer(_ConcatInputLayer):
  """
  Swaps two axes. Basically a wrapper around :func:`returnn.tf.util.basic.swapaxes`.
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
    axis1 = self.input_data.get_axis_from_description(axis1)
    axis2 = self.input_data.get_axis_from_description(axis2)
    self.output.placeholder = tf_util.swapaxes(self.input_data.placeholder, axis1=axis1, axis2=axis2)

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
  def __init__(self, switch_axes=None, size_base=None, batch_dim_base=None,
               set_axes=None,
               set_dim_tags=None,
               enforce_batch_major=False, enforce_time_major=False,
               set_sparse=None, set_sparse_dim=NotSpecified, increase_sparse_dim=None,
               **kwargs):
    """
    :param str|list[str] switch_axes: e.g. "bt" to switch batch and time axes
    :param LayerBase|None size_base: copy the size_placeholder from the given layer
    :param LayerBase|None batch_dim_base: copy the batch dim from this layer
    :param dict[str,Dim|str] set_axes:
      This can be used to overwrite the special axes like time_dim_axis or feature_dim_axis.
      For that, use keys "B","T" or "F", and a value via :func:`Data.get_axis_from_description`.
    :param dict[str|Dim,Dim]|None set_dim_tags: axis -> new dim tag. assigns new dim tags.
      If the passed dim tag is yet undefined, this will not use same_dim_tags_as (declare_same_as)
      but create a new dim tag.
      This option is useful for generalized self attention (https://github.com/rwth-i6/returnn/issues/391).
    :param bool enforce_batch_major:
    :param bool enforce_time_major:
    :param bool|None set_sparse: if bool, set sparse value to this
    :param Dim|int|None|NotSpecified set_sparse_dim: set sparse dim to this. assumes that it is sparse
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
        if old_tag.is_batch_dim() and not new_tag.is_batch_dim():
          self.output.batch = old_tag.batch.get_global_base()
        new_tag = new_tag.get_for_batch_ctx(
          input_data.batch if not old_tag.is_batch_dim() else old_tag.batch.get_global_base(),
          input_data.control_flow_ctx)
        if new_tag.is_batch_dim() or new_tag.dimension is not None:
          continue
        if new_tag.dyn_size_ext and new_tag.dyn_size_ext.placeholder is not None:
          continue
        # still undefined
        if old_tag.is_batch_dim():
          new_dyn_size_ext = Data.from_tensor(input_data.get_batch_dim())
        else:
          assert old_tag.dyn_size_ext
          new_dyn_size_ext = old_tag.dyn_size_ext.copy(name="%s_size" % (new_tag.description or "<unnamed>"))
        # Need to create new size tensor as long as we have get_tag_from_size_tensor.
        new_dyn_size_ext.placeholder = tf.identity(
          new_dyn_size_ext.placeholder, name=get_valid_scope_name_from_str(new_dyn_size_ext.name))
        if new_tag.dyn_size_ext:
          assert new_dyn_size_ext.dim_tags == new_dyn_size_ext.dim_tags
        new_tag.dyn_size_ext = new_dyn_size_ext
        new_tag.set_tag_on_size_tensor(new_dyn_size_ext.placeholder)
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
    if d.get("batch_dim_base"):
      d["batch_dim_base"] = get_layer(d["batch_dim_base"])

  @classmethod
  def get_out_data_from_opts(cls, name, sources,
                             switch_axes=None, size_base=None, batch_dim_base=None,
                             set_axes=None,
                             set_dim_tags=None,
                             enforce_batch_major=False, enforce_time_major=False,
                             set_sparse=None, set_sparse_dim=NotSpecified, increase_sparse_dim=None,
                             **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param str|list[str] switch_axes: e.g. "bt" to switch batch and time axes
    :param LayerBase|None size_base: similar as size_target
    :param LayerBase|None batch_dim_base:
    :param dict[str,Dim|str] set_axes:
    :param dict[str|Dim,Dim]|None set_dim_tags:
    :param bool enforce_batch_major:
    :param bool enforce_time_major:
    :param bool|None set_sparse: if bool, set sparse value to this
    :param Dim|int|None|NotSpecified set_sparse_dim: set sparse dim to this. assumes that it is sparse
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
      for axis, new_tag in set_dim_tags.items():
        axis_int = out.get_axis_from_description(axis)
        old_tag = out.dim_tags[axis_int]
        new_tag = new_tag.get_for_batch_ctx(out.batch, out.control_flow_ctx)
        if old_tag.dyn_size_ext and not new_tag.dyn_size_ext:
          # Copy the template so that we know about implicit dims.
          # The sizes itself will be setup in __init__.
          new_tag.dyn_size_ext = old_tag.dyn_size_ext.copy_template()
        out = out.copy_template_replace_dim_tag(axis=axis_int, new_dim_tag=new_tag)
    if set_sparse is not None:
      assert isinstance(set_sparse, bool)
      if set_sparse:
        if out.sparse_dim:
          pass
        else:
          out_args = out.get_kwargs()
          out_args["sparse"] = True
          if isinstance(set_sparse_dim, Dim):
            out_args["sparse_dim"] = set_sparse_dim
          else:
            out_args["dim"] = set_sparse_dim if set_sparse_dim is not NotSpecified else out.dim
          out_args.pop("feature_dim_axis", None)
          out = Data(**out_args)
      else:
        out.sparse_dim = None
    if set_sparse_dim is not NotSpecified:
      assert out.sparse
      assert set_sparse_dim is None or isinstance(set_sparse_dim, int) or isinstance(set_sparse_dim, Dim)
      if isinstance(set_sparse_dim, Dim):
        out.sparse_dim = set_sparse_dim
      elif out.dim == set_sparse_dim:
        pass
      else:
        out.sparse_dim = Dim(
          kind=Dim.Types.Feature, dimension=set_sparse_dim, description="%s:set-sparse-dim" % name, auto_generated=True)
    if increase_sparse_dim:
      assert out.sparse
      out.sparse_dim = out.sparse_dim + increase_sparse_dim
    if batch_dim_base:
      out.batch = batch_dim_base.output.batch
    return out


class ConvLayer(_ConcatInputLayer):
  """
  A generic convolution layer which supports 1D, 2D and 3D convolution.
  Pooling can be done in the separate "pool" layer.
  """

  layer_class = "conv"
  recurrent = True  # we must not allow any shuffling in the time-dim or so

  # noinspection PyUnusedLocal,PyShadowingBuiltins
  def __init__(self, filter_size, padding, strides=1, dilation_rate=1, groups=1,
               input_expand_dims=0, input_add_feature_dim=False, input_split_feature_dim=None,
               in_dim=None, in_spatial_dims=None,
               n_out=None, out_dim=None, out_spatial_dims=None,
               auto_use_channel_first=NotSpecified,
               with_bias=NotSpecified,
               activation=None,
               forward_weights_init="glorot_uniform", bias_init=0.0,
               filter=None, filter_perm=None, bias=None,
               **kwargs):
    """
    :param tuple[int] filter_size: (width,), (height,width) or (depth,height,width) for 1D/2D/3D conv.
      the input data ndim must match, or you can add dimensions via input_expand_dims or input_add_feature_dim.
      it will automatically swap the batch-dim to the first axis of the input data.
    :param str padding: "same" or "valid"
    :param int|tuple[int] strides: strides for the spatial dims,
      i.e. length of this tuple should be the same as filter_size, or a single int.
    :param int|tuple[int] dilation_rate: dilation for the spatial dims
    :param int groups: grouped convolution
    :param Dim|None in_dim:
    :param list[Dim|str]|None in_spatial_dims:
    :param int|None n_out: number of outgoing features
    :param Dim|None out_dim:
    :param list[Dim]|None out_spatial_dims:
    :param int input_expand_dims: number of spatial dims to add to the input
    :param bool input_add_feature_dim: will add a dim at the end and use input-feature-dim == 1,
      and use the original input feature-dim as a spatial dim.
    :param None|int input_split_feature_dim: if set, like input_add_feature_dim it will add a new feature dim
      which is of value input_split_feature_dim, and the original input feature dim
      will be divided by input_split_feature_dim, thus it must be a multiple of that value.
    :param bool|NotSpecified auto_use_channel_first: convert the input to NCHW or not
    :param bool|NotSpecified with_bias: if True, will add a bias to the output features.
      True by default since behavior version 10.
    :param None|str activation: if set, will apply this function at the end
    :param LayerBase|None filter: if given, will not create an own parameter, but use this as the filter
    :param dict[str,str]|None filter_perm: transposes the filter (input filter as layer)
    :param LayerBase|None bias: if given, will not create an own parameter, but use this as the bias
    """
    from returnn.util import BehaviorVersion
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
    assert self.input_data.have_batch_axis()
    assert self.input_data.have_feature_axis(), (
      "this should be our single input feature dim now. otherwise use input_add_feature_dim")
    input_data, num_batch_dims = self.transform_input(
      self.input_data, network=self.network,
      in_dim=in_dim, in_spatial_dims=in_spatial_dims,
      input_expand_dims=input_expand_dims,
      input_split_feature_dim=input_split_feature_dim,
      input_add_feature_dim=input_add_feature_dim)
    if self.output.feature_dim_axis == num_batch_dims:
      out_batch_feature_major = True
      input_data = input_data.copy_with_feature_dim_axis(num_batch_dims)
      in_spatial_dims_ = input_data.dim_tags[num_batch_dims + 1:]
    else:
      out_batch_feature_major = False
      input_data = input_data.copy_with_feature_dim_axis(-1)
      in_spatial_dims_ = input_data.dim_tags[num_batch_dims:-1]
    if in_spatial_dims:
      assert list(in_spatial_dims_) == [input_data.get_dim_tag_from_description(d) for d in in_spatial_dims]
    assert len(in_spatial_dims_) == len(filter_size), (
      "%s: in_spatial_dims %s not matching to filter_size %s" % (self, in_spatial_dims_, filter_size))
    assert input_data.batch_ndim - num_batch_dims - 1 == len(filter_size), (
      "%s: filter-size-dimension does not match the input data. " % self +
      "this is %i-D conv but found %i spatial dims in the input %s. " % (
        len(filter_size), input_data.batch_ndim - num_batch_dims - 1, self.input_data) +
      "consider using input_expand_dims or input_add_feature_dim.")
    n_in = input_data.dim
    if out_dim:
      assert out_dim == self.output.feature_dim_or_sparse_dim
    else:
      out_dim = self.output.feature_dim_or_sparse_dim
    if n_out:
      assert n_out == out_dim.dimension
    else:
      n_out = out_dim.dimension
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
    if out_batch_feature_major:
      data_format = {1: "NCW", 2: "NCHW", 3: "NCDHW"}[len(filter_size)]
    x = input_data.placeholder
    extended_batch_shape = None
    if num_batch_dims > 1:
      x_shape = tf.shape(x)
      extended_batch_shape = x_shape[:num_batch_dims]
      x = tf.reshape(x, tf.concat([[-1], x_shape[num_batch_dims:]], axis=0))  # merge all batch dims
    if groups > 1 and groups == n_in and len(filter_size) <= 2:  # depthwise conv
      if len(filter_size) == 1:
        filters = tf.reshape(filters, [filter_size[0], 1, n_in, n_out // n_in])  # [1,K,n_in,n_out//n_in]
        x = tf.expand_dims(x, axis=-1 if out_batch_feature_major else -2)  # [B,T,1,n_in]
        strides = strides + [1]
        dilation_rate = dilation_rate + [1]
      else:
        filters = tf.reshape(filters, list(filter_size) + [n_in, n_out // n_in])  # K+[n_in,n_out//n_in]
      y = tf.nn.depthwise_conv2d(
        x, data_format="NCHW" if out_batch_feature_major else "NHWC",
        filter=filters,
        padding=padding,
        strides=([1] + strides + [1]) if out_batch_feature_major else ([1, 1] + strides),
        dilations=dilation_rate)
      if len(filter_size) == 1:
        y = tf.squeeze(y, axis=-1 if out_batch_feature_major else -2)
        strides = strides[:-1]
        dilation_rate = dilation_rate[:-1]
    else:
      y = tf_compat.v1.nn.convolution(
        x, data_format=data_format,
        filter=filters,
        padding=padding, strides=strides, dilation_rate=dilation_rate)
    if num_batch_dims > 1:
      y = tf.reshape(y, tf.concat([extended_batch_shape, tf.shape(y)[1:]], axis=0))
    # y shape is [batch] + dynamic_dims + [n_out].
    if with_bias is NotSpecified:
      if bias or BehaviorVersion.get() >= 10:
        with_bias = True
      else:
        with_bias = False
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
        if out_batch_feature_major:
          y += Data(name="bias", placeholder=b, dim_tags=[out_dim]).copy_compatible_to(self.output).placeholder
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

  @classmethod
  def set_output_dim_tags(cls, output, num_batch_dims, in_spatial_dims, out_spatial_dims,
                          filter_size, strides, dilation_rate, padding):
    """
    :param Data output:
    :param int num_batch_dims:
    :param list[Dim]|tuple[Dim] in_spatial_dims:
    :param list[Dim]|None out_spatial_dims:
    :param list[int]|tuple[int] filter_size:
    :param list[int]|tuple[int] strides:
    :param list[int]|tuple[int] dilation_rate:
    :param str padding:
    """
    if output.feature_dim_axis == num_batch_dims:
      out_spatial_dims_ = output.dim_tags[num_batch_dims + 1:]
    else:
      assert output.feature_dim_axis == output.batch_ndim - 1
      out_spatial_dims_ = output.dim_tags[num_batch_dims:-1]
    if out_spatial_dims:
      assert len(out_spatial_dims_) == len(out_spatial_dims)
      for i, (out_spatial_dim_, out_spatial_dim) in enumerate(zip(out_spatial_dims_, out_spatial_dims)):
        out_spatial_dim_.declare_same_as(out_spatial_dim)
    assert len(out_spatial_dims_) == len(in_spatial_dims) == len(filter_size) == len(strides) == len(dilation_rate)
    for i, in_tag in enumerate(in_spatial_dims):
      out_tag = out_spatial_dims_[i]
      out_tag_calc = cls.calc_out_dim(
        in_dim=in_tag,
        filter_size=filter_size[i], stride=strides[i],
        dilation_rate=dilation_rate[i], padding=padding)
      assert isinstance(out_tag_calc, Dim)
      out_tag_calc.declare_same_as(out_tag)

  @classmethod
  def _check_defined_in_spatial_dims(cls, cond):
    """
    :param bool cond:
    """
    from returnn.util import BehaviorVersion
    BehaviorVersion.require(
      condition=cond,
      message="Explicitly specify in_spatial_dims when there is more than one spatial dim in the input.",
      version=8)

  @classmethod
  def transform_input(cls, input_data, network, in_dim=None, in_spatial_dims=None,
                      input_expand_dims=0, input_split_feature_dim=None, input_add_feature_dim=False):
    """
    :param Data input_data:
    :param returnn.tf.network.TFNetwork network:
    :param Dim|None in_dim:
    :param list[Dim|str]|None in_spatial_dims:
    :param int input_expand_dims: number of spatial dims to add to the input
    :param None|int input_split_feature_dim: if set, like input_add_feature_dim it will add a new feature dim
      which is of value input_split_feature_dim, and the original input feature dim
      will be divided by input_split_feature_dim, thus it must be a multiple of that value.
    :param bool input_add_feature_dim: will add a dim at the end and use input-feature-dim == 1,
      and use the original input feature-dim as a spatial dim.
    :return: (transformed input, num batch dims). all batch dims are at the front
    :rtype: (Data, int)
    """
    assert not input_data.sparse
    assert input_data.have_batch_axis()
    if input_expand_dims or input_split_feature_dim or input_add_feature_dim:
      if not in_spatial_dims:
        # Set it here to allow to handle the extension logic below.
        if input_data.placeholder is None and not input_data.have_feature_axis():
          # This is still a template. Ignore errors and make it valid.
          input_data = input_data.copy_add_feature_dim()
        in_spatial_dims = [input_data.dim_tags[a] for a in input_data.get_spatial_batch_axes()]
        cls._check_defined_in_spatial_dims(len(in_spatial_dims) == 1)
    if input_expand_dims:
      for i in range(input_expand_dims):
        dim_tag = SpatialDim("input_expand_dims:%i" % i, dimension=1, auto_generated=True)
        input_data = input_data.copy_add_dim_by_tag(dim_tag, unbroadcast=True)
        in_spatial_dims.append(dim_tag)
    if input_split_feature_dim:
      # Split the feature dimension.
      input_data = input_data.copy_split_feature_dim(input_split_feature_dim)
      in_spatial_dims.append(input_data.dim_tags[input_data.feature_dim_axis - 1])
    if input_add_feature_dim:
      # Add a new feature dimension; the old feature dim becomes a spatial dim.
      in_spatial_dims.append(input_data.dim_tags[input_data.feature_dim_axis])
      input_data = input_data.copy_add_feature_dim()
    if in_dim:
      assert isinstance(in_dim, Dim)
      if input_data.feature_dim_or_sparse_dim != in_dim:
        input_data.feature_dim_axis = input_data.get_axis_from_description(in_dim)
    if in_spatial_dims:
      from returnn.tf.util.data import batch_dim
      assert all(isinstance(d, (Dim, str)) for d in in_spatial_dims)
      axes = [input_data.get_axis_from_description(d) for d in in_spatial_dims]  # also requires unique dim tags
      in_spatial_dims = [input_data.dim_tags[a] for a in axes]
      assert sorted(set(axes)) == sorted(axes), (
        "in_spatial_dims %s must be unique but map to %s" % (in_spatial_dims, axes))
      if sorted(axes) != axes:
        # Sort them such that the convolution is correct.
        first = min(axes)
        for i, d in enumerate(in_spatial_dims):
          a = input_data.get_axis_from_description(d)
          input_data = input_data.copy_move_axis(old_axis=a, new_axis=first + i)
        axes = [input_data.get_axis_from_description(d) for d in in_spatial_dims]
        assert sorted(axes) == axes
      assert input_data.feature_dim_axis not in axes, "invalid in_spatial_dims %s" % (in_spatial_dims,)
      expected_dims = {batch_dim, input_data.feature_dim_or_sparse_dim} | set(in_spatial_dims)
      assert len(expected_dims) == 2 + len(in_spatial_dims)
      # There might be more dims in the input than we expect.
      assert set(input_data.dim_tags).issuperset(expected_dims)
      # Prepare to merge all remaining ones into the batch dim. We will later undo this at the end.
      # This is needed to support a ConvLayer both inside a rec loop which then can be optimized out.
      # But also this is a useful feature in general.
      # Move all batch dims right next to each other in front. But keep the order.
      expected_non_batch_dims = expected_dims - {batch_dim}
      batch_axis_idx = 0
      for a, d in enumerate(input_data.dim_tags):
        if d not in expected_non_batch_dims:
          if a != batch_axis_idx:
            input_data = input_data.copy_move_axis(old_axis=a, new_axis=batch_axis_idx)
          batch_axis_idx += 1
      num_batch_dims = batch_axis_idx
    else:  # no specified in_spatial_dims
      batch_axes = {input_data.batch_dim_axis}
      inside_rec_time_dim = network.get_inside_rec_time_dim(inside_loop=True)
      over_rec_time_dim = network.get_inside_rec_time_dim(inside_loop=False)
      if over_rec_time_dim and not inside_rec_time_dim and over_rec_time_dim in input_data.dim_tags:
        # It is moved out of a rec loop. This axis can not be used.
        batch_axes.add(input_data.get_axis_from_description(over_rec_time_dim))
      batch_axis_idx = 0
      for a in sorted(batch_axes):
        if a != batch_axis_idx:
          input_data = input_data.copy_move_axis(old_axis=a, new_axis=batch_axis_idx)
        batch_axis_idx += 1
      num_batch_dims = batch_axis_idx
      in_spatial_dims = [
        d for (i, d) in enumerate(input_data.dim_tags)
        if i >= num_batch_dims and i != input_data.feature_dim_axis]
      cls._check_defined_in_spatial_dims(len(in_spatial_dims) <= 1)
    return input_data, num_batch_dims

  @classmethod
  def calc_out_dim(cls, in_dim, filter_size, stride, padding, dilation_rate=1):
    """
    :param T|int|tf.Tensor|Dim in_dim: dimension in some axis
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
      if isinstance(in_dim, Dim):
        return in_dim.ceildiv_right(stride)
      return ceildiv(in_dim, stride)
    elif padding == "VALID":
      if isinstance(in_dim, Dim):
        filter_left_dilated = (filter_size - 1) * dilation_rate // 2
        filter_right_dilated = (filter_size - 1) * dilation_rate - filter_left_dilated
        valid_part = in_dim.sub_left(filter_left_dilated).sub_right(filter_right_dilated)
        return valid_part.ceildiv_right(stride)
      return tf_util.simplify_non_negative_seq_length(
        ceildiv(
          (tf_util.simplify_sub(in_dim, (filter_size - 1) * dilation_rate)),
          stride))
    else:
      raise Exception("invalid padding %r" % padding)

  @classmethod
  def get_out_data_from_opts(
        cls, name, sources, network,
        filter_size, padding, strides=1, dilation_rate=1,
        input_expand_dims=0, input_add_feature_dim=False, input_split_feature_dim=None,
        in_dim=None, in_spatial_dims=None,
        n_out=None, out_dim=None, out_spatial_dims=None,
        auto_use_channel_first=NotSpecified,
        **kwargs):
    """
    :param str name:
    :param list[LayerBase]|tuple[LayerBase] sources:
    :param returnn.tf.network.TFNetwork network:
    :param tuple[int] filter_size:
    :param str padding:
    :param int|tuple[int]|list[int] strides:
    :param int|tuple[int]|list[int] dilation_rate:
    :param int input_expand_dims: number of dynamic dims to add to the input
    :param bool input_add_feature_dim:
    :param None|int input_split_feature_dim:
    :param Dim|None in_dim:
    :param list[Dim|str]|None in_spatial_dims:
    :param int|None n_out: number of outgoing features
    :param Dim|None out_dim:
    :param list[Dim]|None out_spatial_dims:
    :param int input_expand_dims: number of spatial dims to add to the input
    :param bool|NotSpecified auto_use_channel_first:
    """
    from returnn.util import BehaviorVersion
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
    if in_spatial_dims:
      assert len(in_spatial_dims) == len(filter_size)
    padding = padding.upper()
    # Be relaxed about incorrect input data. Throw errors later. This can also work during template construction.
    if not input_data.have_batch_axis():
      input_data = input_data.copy_add_batch_dim(batch_dim_axis=0)
    input_data, num_batch_dims = cls.transform_input(
      input_data, network=network,
      in_dim=in_dim, in_spatial_dims=in_spatial_dims,
      input_expand_dims=input_expand_dims,
      input_split_feature_dim=input_split_feature_dim,
      input_add_feature_dim=input_add_feature_dim)
    # Be relaxed about incorrect input data. Throw errors later. This can also work during template construction.
    if input_data.have_feature_axis():
      data = input_data.copy_with_feature_dim_axis(-1)  # just to have the dim tags in order [B,S...,D]
    else:
      data = input_data.copy_add_feature_dim(-1)
    old_spatial_dim_tags = data.dim_tags[num_batch_dims:-1]
    dim_tags = list(data.dim_tags[:num_batch_dims])  # [B]
    if out_spatial_dims:
      assert len(out_spatial_dims) == len(filter_size)
      # Be relaxed about incorrect input data. Throw errors later. This can also work during template construction.
      dim_tags += out_spatial_dims
    else:
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
        dim_tags.append(Dim(
          kind=Dim.Types.Spatial, description="%s:conv:s%i" % (name, i), dimension=new_dim,
          derived_from_tag=old_tag, undefined=not old_tag, auto_generated=True))
      out_spatial_dims = dim_tags[num_batch_dims:]
    if not out_dim:
      assert n_out
      out_dim = FeatureDim("%s:channel" % name, dimension=n_out, auto_generated=True)
    dim_tags.append(out_dim)
    time_dim_axis = data.time_dim_axis
    feature_dim_axis = NotSpecified
    # Swap the dims if the input dim order doesn't fit the flag auto_use_channel_first.
    if auto_use_channel_first is NotSpecified:
      auto_use_channel_first = True if BehaviorVersion.get() >= 9 else False
    if auto_use_channel_first or input_data.feature_dim_axis == num_batch_dims:  # batch-feature-major
      if tf_util.is_gpu_available_in_session():
        if len([d for d in dim_tags if d.dimension]) > 1:
          feature_dim_axis = num_batch_dims
        dim_tags = dim_tags[:num_batch_dims] + dim_tags[-1:] + dim_tags[num_batch_dims:-1]
        if time_dim_axis is not None and time_dim_axis >= num_batch_dims:
          if time_dim_axis == len(dim_tags) - 1:
            time_dim_axis = num_batch_dims
          else:
            time_dim_axis += 1
    out = Data(
      name="%s_output" % name, dim_tags=dim_tags,
      time_dim_axis=time_dim_axis, feature_dim_axis=feature_dim_axis,
      batch=data.batch, beam=data.beam, control_flow_ctx=data.control_flow_ctx)
    if len(old_spatial_dim_tags) == len(filter_size):
      cls.set_output_dim_tags(
        out, num_batch_dims=num_batch_dims, in_spatial_dims=old_spatial_dim_tags, out_spatial_dims=out_spatial_dims,
        filter_size=filter_size, strides=strides, dilation_rate=dilation_rate, padding=padding)
    return out

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
               in_dim=None, in_spatial_dims=None,
               out_dim=None, out_spatial_dims=None,
               use_channel_first=NotSpecified,
               **kwargs):
    """
    :param str mode: "max" or "avg"
    :param tuple[int] pool_size: shape of the window of each reduce
    :param str padding: "valid" or "same"
    :param tuple[int]|int dilation_rate:
    :param tuple[int]|int|None strides: in contrast to tf.nn.pool, the default (if it is None) will be set to pool_size
    :param Dim|None in_dim:
    :param list[Dim|str]|None in_spatial_dims:
    :param Dim|None out_dim:
    :param list[Dim]|None out_spatial_dims:
    :param bool|NotSpecified use_channel_first: if set, will transform input to NCHW format
    """
    assert "n_out" not in kwargs
    assert "out_type" not in kwargs
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
    assert not self.input_data.sparse
    assert self.input_data.have_batch_axis()
    assert self.input_data.have_feature_axis(), (
      "this should be our single input feature dim now. otherwise use input_add_feature_dim")
    if in_dim and out_dim:
      assert in_dim == out_dim
    elif in_dim:
      out_dim = in_dim
    elif out_dim:
      in_dim = out_dim
    else:
      assert self.input_data.have_feature_axis()
      out_dim = in_dim = self.input_data.feature_dim_or_sparse_dim
    input_data, num_batch_dims = ConvLayer.transform_input(
      self.input_data, network=self.network, in_dim=in_dim, in_spatial_dims=in_spatial_dims)
    # We want to prepare the input data such that the batch-dim(s) is the very first,
    # the feature-dim is the very last ("NHWC" format) or right after batch-dim ("NCHW"),
    # and all other dims are where we convolve over.
    if self.output.feature_dim_axis == num_batch_dims:
      out_batch_feature_major = True
      input_data = input_data.copy_with_feature_dim_axis(num_batch_dims)
      in_spatial_dims_ = input_data.dim_tags[num_batch_dims + 1:]
    else:
      out_batch_feature_major = False
      input_data = input_data.copy_with_feature_dim_axis(-1)
      in_spatial_dims_ = input_data.dim_tags[num_batch_dims:-1]
    if in_spatial_dims:
      assert list(in_spatial_dims_) == [input_data.get_dim_tag_from_description(d) for d in in_spatial_dims]
    assert len(in_spatial_dims_) == len(pool_size)
    assert input_data.batch_ndim - num_batch_dims - 1 == len(pool_size), (
      "%s: pool-size-dimension does not match the input data. " % self +
      "this is %i-D pool but found %i spatial dims in the input %s. " % (
        len(pool_size), input_data.batch_ndim - num_batch_dims - 1, self.input_data) +
      "consider using input_expand_dims or input_add_feature_dim.")
    if all([s == 1 for s in pool_size]) and all([s == 1 for s in strides]):
      # Identity function. Just copy and don't do anything.
      self.output.placeholder = self.input_data.copy_compatible_to(self.output).placeholder
      return
    data_format = None
    if out_batch_feature_major:
      data_format = {1: "NCW", 2: "NCHW", 3: "NCDHW"}[len(pool_size)]
    x = input_data.placeholder
    extended_batch_shape = None
    if num_batch_dims > 1:
      x_shape = tf.shape(x)
      extended_batch_shape = x_shape[:num_batch_dims]
      x = tf.reshape(x, tf.concat([[-1], x_shape[num_batch_dims:]], axis=0))  # merge all batch dims
    y = tf_compat.v1.nn.pool(
      x, window_shape=pool_size, pooling_type=mode, padding=padding,
      dilation_rate=dilation_rate, strides=strides, data_format=data_format)
    # y shape is [batch] + spatial_dims + [n_out] or [batch, n_out] + spatial_dims.
    if num_batch_dims > 1:
      y = tf.reshape(y, tf.concat([extended_batch_shape, tf.shape(y)[1:]], axis=0))
    self.output.placeholder = y

  @classmethod
  def get_out_data_from_opts(cls, name, sources, network,
                             pool_size, strides=None, dilation_rate=1, padding="VALID",
                             in_dim=None, in_spatial_dims=None,
                             out_dim=None, out_spatial_dims=None,
                             use_channel_first=NotSpecified,
                             **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param returnn.tf.network.TFNetwork network:
    :param tuple[int]|list[int] pool_size:
    :param tuple[int]|list[int]|int strides:
    :param int|tuple[int]|list[int] dilation_rate:
    :param str padding:
    :param Dim|None in_dim:
    :param list[Dim|str]|None in_spatial_dims:
    :param Dim|None out_dim:
    :param list[Dim]|None out_spatial_dims:
    :param bool|NotSpecified use_channel_first:
    :rtype: Data
    """
    data = get_concat_sources_data_template(sources)
    if in_dim and out_dim:
      assert in_dim == out_dim
    elif in_dim:
      out_dim = in_dim
    elif out_dim:
      in_dim = out_dim
    else:
      assert data.have_feature_axis()
      out_dim = in_dim = data.feature_dim_or_sparse_dim
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
    # y shape is [batch] + spatial_dims + [n_out].
    return ConvLayer.get_out_data_from_opts(
      name=name, sources=sources, network=network,
      filter_size=pool_size, padding=padding, strides=strides, dilation_rate=dilation_rate,
      in_dim=in_dim, out_dim=out_dim, in_spatial_dims=in_spatial_dims, out_spatial_dims=out_spatial_dims,
      auto_use_channel_first=use_channel_first)


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
      assert set(self.input_data.size_placeholder.keys()) == {self.input_data.time_dim_axis_excluding_batch}
      size_placeholder = {
        self.output.time_dim_axis_excluding_batch: self.input_data.size_placeholder.copy()[
          self.input_data.time_dim_axis_excluding_batch]}
      self.output.size_placeholder = size_placeholder
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
  def __init__(self, filter_size, strides=None,
               padding="same",
               remove_padding=0,
               output_padding=None,
               in_dim=None, in_spatial_dims=None,
               out_dim=None, out_spatial_dims=None,
               with_bias=True,
               activation=None,
               forward_weights_init="glorot_uniform", bias_init=0.0,
               filter=None, filter_perm=None, bias=None,
               **kwargs):
    """
    :param list[int] filter_size:
    :param list[int]|None strides: specifies the upscaling. by default, same as filter_size
    :param str padding: "same" or "valid"
    :param list[int]|int remove_padding:
    :param list[int|None]|int|None output_padding:
    :param Dim|None in_dim:
    :param list[Dim|str]|None in_spatial_dims:
    :param Dim|None out_dim:
    :param list[Dim]|None out_spatial_dims:
    :param bool with_bias: whether to add a bias. enabled by default.
    :param str|None activation:
    :param forward_weights_init:
    :param bias_init:
    :param LayerBase|None filter: if given, will not create an own parameter, but use this as the filter
    :param dict[str,str]|None filter_perm: transposes the filter (input filter as layer)
    :param LayerBase|None bias: if given, will not create an own parameter, but use this as the bias
    """
    from returnn.tf.util.basic import get_initializer, get_activation_function, get_shape
    super(TransposedConvLayer, self).__init__(**kwargs)
    out_dim  # noqa  # via get_out_data_from_opts
    out_spatial_dims  # noqa  # via get_out_data_from_opts
    assert not self.input_data.sparse
    assert self.input_data.have_batch_axis()
    assert self.input_data.have_feature_axis(), (
      "this should be our single input feature dim now. otherwise use input_add_feature_dim")
    input_data, num_batch_dims = ConvLayer.transform_input(
      self.input_data, network=self.network,
      in_dim=in_dim, in_spatial_dims=in_spatial_dims)
    input_data = input_data.copy_with_feature_last()
    spatial_axes = list(range(num_batch_dims, input_data.batch_ndim - 1))
    if in_spatial_dims:
      assert (
        [input_data.dim_tags[a] for a in spatial_axes]
        == [input_data.get_dim_tag_from_description(d) for d in in_spatial_dims])
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
    extended_batch_shape = None
    if num_batch_dims > 1:
      x_shape = tf.shape(x)
      extended_batch_shape = x_shape[:num_batch_dims]
      x = tf.reshape(x, tf.concat([[-1], x_shape[num_batch_dims:]], axis=0))  # merge all batch dims
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
    if num_batch_dims > 1:
      y = tf.reshape(y, tf.concat([extended_batch_shape, tf.shape(y)[1:]], axis=0))
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

    Also see :func:`ConvLayer.calc_out_dim`.

    :param T|int|tf.Tensor|Dim input_length:
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
        if isinstance(input_length, Dim):
          length = input_length + max(filter_size - stride, 0)
        else:
          length = tf_util.simplify_add(input_length, max(filter_size - stride, 0))
      elif padding == 'full':
        if isinstance(input_length, Dim):
          length = input_length - (stride + filter_size - 2)
        else:
          length = tf_util.simplify_add(input_length, -(stride + filter_size - 2))
      elif padding == 'same':
        length = input_length
      else:
        raise Exception("invalid padding %r" % (padding,))
    else:  # output_padding
      if padding == 'same':
        pad = filter_size // 2
      elif padding == 'valid':
        pad = 0
      elif padding == 'full':
        pad = filter_size - 1
      else:
        raise Exception("invalid padding %r" % (padding,))
      if isinstance(input_length, Dim):
        length = input_length + (-stride + filter_size - 2 * pad + output_padding)
      else:
        length = tf_util.simplify_add(input_length, -stride + filter_size - 2 * pad + output_padding)
    return length

  @classmethod
  def get_out_data_from_opts(cls, name, sources, network,
                             filter_size, strides=None, padding="same", remove_padding=0, output_padding=None,
                             n_out=None, out_dim=None, out_spatial_dims=None,
                             in_dim=None, in_spatial_dims=None,
                             **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param returnn.tf.network.TFNetwork network:
    :param list[int] filter_size:
    :param list[int]|None strides:
    :param str padding:
    :param list[int]|int remove_padding:
    :param list[int|None]|int|None output_padding:
    :param int|None n_out: number of outgoing features
    :param Dim|None out_dim:
    :param list[Dim]|None out_spatial_dims:
    :param Dim|None in_dim:
    :param list[Dim|str]|None in_spatial_dims:
    :rtype: Data
    """
    input_data = get_concat_sources_data_template(sources)
    input_data, num_batch_dims = ConvLayer.transform_input(
      input_data, network=network, in_dim=in_dim, in_spatial_dims=in_spatial_dims)
    if strides is None:
      strides = filter_size
    if isinstance(remove_padding, int):
      remove_padding = [remove_padding] * len(filter_size)
    if not isinstance(output_padding, (list, tuple)):
      output_padding = [output_padding] * len(filter_size)
    assert len(strides) == len(remove_padding) == len(output_padding), "Expected strides for all spatial axes"
    # Be relaxed about incorrect input data. Throw errors later. This can also work during template construction.
    if input_data.have_feature_axis():
      data = input_data.copy_with_feature_dim_axis(-1)  # just to have the dim tags in order [B,S...,D]
    else:
      data = input_data.copy_add_feature_dim(-1)
    old_spatial_dim_tags = data.dim_tags[num_batch_dims:-1]
    dim_tags = list(data.dim_tags[:num_batch_dims])  # [B]
    if out_spatial_dims:
      assert len(out_spatial_dims) == len(filter_size)
    # Be relaxed about incorrect input data. Throw errors later. This can also work during template construction.
    for i in range(len(filter_size)):
      old_tag = old_spatial_dim_tags[i] if i < len(old_spatial_dim_tags) else None
      new_tag = cls.deconv_output_length(
        old_tag, filter_size=filter_size[i], stride=strides[i],
        padding=padding, output_padding=output_padding[i])
      new_tag = new_tag.sub_left(remove_padding[i]).sub_right(remove_padding[i])
      if out_spatial_dims:
        new_tag.declare_same_as(out_spatial_dims[i])
      dim_tags.append(new_tag)
    if not out_dim:
      assert n_out
      out_dim = FeatureDim("%s:channel" % name, dimension=n_out, auto_generated=True)
    dim_tags.append(out_dim)
    return Data(
      name="%s_output" % name, dim_tags=dim_tags,
      batch=data.batch, beam=data.beam, control_flow_ctx=data.control_flow_ctx)

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
  This reduces some axis by using e.g. "sum" or "max".
  It's basically a wrapper around tf.reduce_sum or tf.reduce_max.
  """
  layer_class = "reduce"

  def __init__(self, mode, axes=None, axis=None, keep_dims=False, enforce_batch_dim_axis=None, use_time_mask=None,
               **kwargs):
    """
    :param str mode: "sum" or "max", "argmin", "min", "argmax", "mean", "logsumexp"
    :param typing.Sequence[Dim|str] axes: One axis or multiple axis to reduce.
      It accepts the special tokens "B"|"batch", "spatial", "spatial_except_time", or "F"|"feature",
      and it is strongly recommended to use some of these symbolic names.
      See :func:`Data.get_axes_from_description`.
    :param Dim|str axis: for compatibility, can be used instead of ``axes``
    :param bool keep_dims: if dimensions should be kept (will be 1)
    :param int|None enforce_batch_dim_axis: will swap the batch-dim-axis of the input with the given axis.
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
      use_time_mask = any(x.has_dynamic_size(a) for a in axes)
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
          if not x.has_dynamic_size(axis):
            continue
          mask = x.get_sequence_mask_broadcast(axis=axis)

          zeros = tf.zeros((), dtype=x.placeholder.dtype)
          # Cannot call x.placeholder.dtype.{min,max} in case input is e.g. a bool
          if x.placeholder.dtype.is_floating or x.placeholder.dtype.is_integer:
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
            raise TypeError("reduce: unexpected input type %r from input %s" % (x.placeholder.dtype, input_data))

          x_ = tf_util.where_bc(mask, x_, replacement_value, name="x_masked_axis_%i" % axis)
          if f == tf.reduce_mean:
            seq_len_bc = x.get_sequence_lengths_broadcast(axis=axis)
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
    :param int|list[int]|str|Dim axes:
    :return: if any integer is in axes, thus we should have a fixed dimension layout
    :rtype: bool
    """
    if isinstance(axes, int):
      return True
    if isinstance(axes, (str, Dim)):
      return False
    assert isinstance(axes, (list, tuple, typing.Sequence))
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
    y_dim_tags = list(x.dim_tags)
    out_batch_dim_axis = x.batch_dim_axis
    out_feature_dim_axis = x.feature_dim_axis_or_unspecified
    out_time_dim_axis = x.time_dim_axis
    if keep_dims:
      for i in axes:
        y_dim_tags[i] = Dim(
          kind=y_dim_tags[i].kind, dimension=1, description="%s:keep-dim-%i" % (name, i), auto_generated=True)
    else:
      if out_batch_dim_axis in axes:
        out_batch_dim_axis = None
      if out_time_dim_axis in axes:
        out_time_dim_axis = NotSpecified
      if out_feature_dim_axis in axes:
        out_feature_dim_axis = NotSpecified
      for i in reversed(sorted(set(axes))):
        del y_dim_tags[i]
      for i in reversed(sorted(set(axes))):
        if out_batch_dim_axis and i < out_batch_dim_axis:
          out_batch_dim_axis -= 1
        if out_time_dim_axis and out_time_dim_axis is not NotSpecified and i < out_time_dim_axis:
          out_time_dim_axis -= 1
        if out_feature_dim_axis and out_feature_dim_axis is not NotSpecified and i < out_feature_dim_axis:
          out_feature_dim_axis -= 1
    sparse_out = mode.lower().startswith("arg")
    sparse_dim = None
    if sparse_out:
      out_feature_dim_axis = None
      assert len(axes) == 1
      sparse_dim = x.dim_tags[axes[0]]
    return Data(
      name="%s_output" % name,
      dim_tags=y_dim_tags,
      batch_dim_axis=out_batch_dim_axis,
      time_dim_axis=out_time_dim_axis,
      feature_dim_axis=out_feature_dim_axis,
      dtype="int32" if sparse_out else x.dtype,
      sparse_dim=sparse_dim,
      beam=x.beam)


class ReduceOutLayer(_ConcatInputLayer):
  """
  Combination of :class:`SplitDimsLayer` applied to the feature dim
  and :class:`ReduceLayer` applied to the resulting feature dim.
  This can e.g. be used to do maxout.
  """
  layer_class = "reduce_out"

  def __init__(self, mode, num_pieces, out_dim=None, **kwargs):
    """
    :param str mode: "sum" or "max" or "mean"
    :param int num_pieces: how many elements to reduce. The output dimension will be input.dim // num_pieces.
    :param Dim|None out_dim:
    """
    super(ReduceOutLayer, self).__init__(out_dim=out_dim, **kwargs)
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
  def get_out_data_from_opts(cls, num_pieces, sources, name, out_dim=None, **kwargs):
    """
    :param int num_pieces:
    :param list[LayerBase] sources:
    :param str name:
    :param Dim|None out_dim:
    :rtype: Data
    """
    out = get_concat_sources_data_template(sources, name="%s_output" % name)
    assert out.have_feature_axis()
    assert not out.sparse
    assert out.dim % num_pieces == 0
    out_dim_ = out.feature_dim_or_sparse_dim // num_pieces
    if out_dim:
      out_dim_.declare_same_as(out_dim)
    return out.copy_template_replace_dim_tag(axis=out.feature_dim_axis, new_dim_tag=out_dim_)


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
  This creates a new dimension for the stack.

  For concatenation (in feature dimension), see :class:`CopyLayer`.
  """
  layer_class = "stack"

  def __init__(self, axis=None, out_spatial_dim=None, **kwargs):
    """
    :param int|None axis: new axis.
      If not given, will use Data.get_default_new_axis_for_dim_tag(<spatial>),
      i.e. some reasonable default for a new spatial axis.
    :param Dim|None out_spatial_dim:
    """
    out_spatial_dim  # noqa  # handled in get_out_data_from_opts
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
  def _get_axis_and_common(cls, sources, name=None):
    """
    :param list[LayerBase] sources:
    :param str|None name:
    :rtype: (int,Data)
    """
    common_source = Data.get_common_data([src.output for src in sources], name=name)
    dummy_tag = Dim(kind=Dim.Types.Spatial, dimension=1)
    return common_source.get_default_new_axis_for_dim_tag(dummy_tag), common_source

  @classmethod
  def get_out_data_from_opts(cls, name, sources, axis=None, out_spatial_dim=None, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param int|None axis:
    :param Dim|None out_spatial_dim:
    :rtype: Data
    """
    axis_, common_source = cls._get_axis_and_common(sources, name="%s_sources" % name)
    if axis is None:
      axis = axis_
    out = common_source.copy_template(name="%s_output" % name)
    out_spatial_dim_ = Dim(
      kind=Dim.Types.Spatial, description="%s:stack" % name, dimension=len(sources), auto_generated=True)
    if out_spatial_dim:
      out_spatial_dim_.declare_same_as(out_spatial_dim)
    out = out.copy_add_dim_by_tag(axis=axis, dim_tag=out_spatial_dim_, unbroadcast=True)
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
    from ..util.data import Dim
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
        dim_tags[a] = Dim(
          kind=dim_tags[a].kind, description="%s:weighted-sum:%i" % (name, i), dimension=res_dims[i],
          auto_generated=True)
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
  Also see :class:`PadLayer` for static dimensions.
  Also see :class:`PostfixInTimeLayer`.
  """
  layer_class = "prefix_in_time"
  recurrent = True

  def __init__(self, axis="T", out_dim=None, prefix=0.0, repeat=1, size_base=None, **kwargs):
    """
    :param Dim|str axis:
    :param Dim|None out_dim:
    :param float|str prefix: either some constant or another layer
    :param int|LayerBase repeat: how often to repeat the prefix
    :param LayerBase|None size_base: copy seq-lens from here
    """
    out_dim, size_base  # noqa  # handled in get_out_data_from_opts
    super(PrefixInTimeLayer, self).__init__(**kwargs)
    assert isinstance(prefix, (float, int)), "other layer src not yet supported"
    input_data = self.input_data.copy()
    axis_int = input_data.get_axis_from_description(axis, allow_int=False)
    in_dim = input_data.dim_tags[axis_int]
    self.repeat_layer = None
    if isinstance(repeat, LayerBase):
      self.repeat_layer = repeat
      repeat = repeat.output.copy_compatible_to(in_dim.dyn_size_ext).placeholder
      input_data = input_data.copy_as_batch_major()
      input_data = input_data.copy_move_axis(old_axis=axis_int, new_axis=1)
      axis_int = 1
    else:
      assert isinstance(repeat, int)
      assert repeat >= 0
    self.repeat = repeat
    out_dim = self.output.dim_tags[axis_int]
    max_repeat = repeat if isinstance(repeat, int) else tf.maximum(tf.reduce_max(repeat), 0)
    shape = [((self.output.batch_shape[i] or tf.shape(input_data.placeholder)[i])
              if (i != axis_int)
              else max_repeat)
             for i in range(self.output.batch_ndim)]
    prefix_t = tf.ones(shape, dtype=self.output.dtype) * tf.constant(prefix, dtype=self.output.dtype)
    x = tf.concat([prefix_t, input_data.placeholder], axis=axis_int)
    if not isinstance(repeat, int):
      max_new_seq_len = tf.reduce_max(out_dim.dyn_size)
      from returnn.tf.util.basic import slice_nd
      assert (self.output.batch_dim_axis, axis_int) == (0, 1)
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
  def get_out_data_from_opts(cls, name, sources, axis="T", out_dim=None, size_base=None, repeat=1, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param Dim|str axis:
    :param Dim|None out_dim:
    :param LayerBase|None size_base:
    :param LayerBase|int repeat:
    :rtype: Data
    """
    x = get_concat_sources_data_template(sources, name="%s_output" % name)
    axis_int = x.get_axis_from_description(axis, allow_int=False)
    in_dim = x.dim_tags[axis_int]
    if size_base:
      assert not out_dim
      out_dim_ = size_base.output.get_time_dim_tag()
    else:
      out_dim_ = (
        repeat if isinstance(repeat, int) else SpatialDim("%s:repeat" % repeat.name, auto_generated=True)) + in_dim
    if out_dim:
      out_dim_.declare_same_as(out_dim)
    x = x.copy_template_replace_dim_tag(axis=axis_int, new_dim_tag=out_dim_)
    if isinstance(repeat, LayerBase):
      x = x.copy_as_batch_spatial_major()
    return x


class PostfixInTimeLayer(_ConcatInputLayer):
  """
  Adds some postfix in time dimension.
  Also see :class:`PrefixInTimeLayer`.
  """
  layer_class = "postfix_in_time"
  recurrent = True

  def __init__(self, axis="T", out_dim=None, postfix=0.0, repeat=1, **kwargs):
    """
    :param Dim|str axis:
    :param Dim|None out_dim:
    :param float|int|LayerBase postfix: constant or other layer without time axis to use as postfix
    :param int repeat: how often to repeat the postfix
    """
    out_dim  # noqa  # handled in get_out_data_from_opts
    super(PostfixInTimeLayer, self).__init__(**kwargs)
    input_data = self.input_data.copy()
    axis_int = input_data.get_axis_from_description(axis, allow_int=False)
    in_dim = input_data.dim_tags[axis_int]
    assert isinstance(postfix, (float, int, LayerBase))
    in_shape = tf_util.get_shape(input_data.placeholder)
    in_shape_added = list(in_shape)
    in_shape_added[axis_int] = repeat
    if isinstance(postfix, LayerBase):
      self.postfix_layer = postfix
      assert in_dim not in postfix.output.dim_tags, 'Postfix layer with time axis not implemented yet'
      postfix = postfix.output.copy_compatible_to(self.output, unbroadcast=True, except_axis=axis)
      c = postfix.placeholder
      c_ = tf.tile(c, [1 if i != axis_int else repeat for i in range(self.output.batch_ndim)])
    else:
      self.postfix_layer = None
      c = tf.constant(postfix, dtype=self.output.dtype)
      c_ = tf.fill(in_shape_added, c)
    x = tf.concat([input_data.placeholder, c_], axis=axis_int)  # make enough space
    self.output.placeholder = x
    if in_dim.dyn_size is not None:  # dynamic
      max_idx = tf.reduce_max(in_dim.dyn_size) + repeat
      # We use the assumption that self.placeholder.shape[axis] == max_idx.
      idx_range = tf.range(max_idx)
      idx_range = tf.reshape(
        idx_range, [1] * (axis_int - 1) + [max_idx] + [1] * (self.output.batch_ndim - axis_int - 1))
      assert (
        set(in_dim.dyn_size_ext.dim_tags)
        .issubset(self.output.dim_tags))  # https://github.com/rwth-i6/returnn/issues/721
      size_ext = in_dim.dyn_size_ext.copy_compatible_to(self.output, check_sparse=False, check_dtype=False)
      seq_mask = tf.less(idx_range, size_ext.placeholder)
      assert seq_mask.get_shape().ndims == self.output.batch_ndim
      self.output.placeholder = tf_util.where_bc(seq_mask, x, c)

  @classmethod
  def get_out_data_from_opts(cls, name, sources, axis="T", out_dim=None, postfix=0.0, repeat=1, **kwargs):
    """
    :param Dim|str axis:
    :param Dim|None out_dim:
    :param str name:
    :param list[LayerBase] sources:
    :param float|int|LayerBase postfix: constant or other layer without time axis to use as postfix
    :param int repeat:
    :rtype: Data
    """
    x = get_concat_sources_data_template(sources, name="%s_output" % name)
    axis_int = x.get_axis_from_description(axis, allow_int=False)
    in_dim = x.dim_tags[axis_int]
    out_dim_ = in_dim + repeat
    if out_dim:
      out_dim_.declare_same_as(out_dim)
    x = x.copy_template_replace_dim_tag(axis=axis_int, new_dim_tag=out_dim_)
    return x

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
  Performs chunking in time. See :func:`returnn.tf.native_op.chunk`.
  """
  layer_class = "time_chunking"
  recurrent = True

  def __init__(self, chunk_size, chunk_step, axis="T", out_dim=None, **kwargs):
    """
    :param int chunk_size:
    :param int chunk_step:
    :param Dim|str axis:
    :param Dim|None out_dim:
    """
    super(TimeChunkingLayer, self).__init__(out_dim=out_dim, **kwargs)
    self.chunk_size = chunk_size
    self.chunk_step = chunk_step
    axis = self.input_data.get_axis_from_description(axis, allow_int=False)
    x = self.input_data.copy_move_axis(old_axis=axis, new_axis=0)
    x = x.copy_with_batch_dim_axis(1)
    self.input_data = x
    in_dim = x.dim_tags[0]
    x_t = x.placeholder
    if in_dim.dyn_size is not None:
      index = tf.cast(tf_util.sequence_mask_time_major(in_dim.dyn_size), tf.float32)
    else:
      index = tf.fill(tf.shape(x_t)[:2], 1.)
    ext_rem_shape = None
    if x.batch_ndim != 3:
      x_shape = tf_util.get_shape(x_t)
      ext_rem_shape = x_shape[2:]
      x_t = tf.reshape(x_t, x_shape[:2] + [-1])
    from returnn.tf.native_op import chunk
    out, oindex = chunk(x_t, index=index, chunk_step=chunk_step, chunk_size=chunk_size)
    if ext_rem_shape:
      out = tf.reshape(out, tf.concat([tf.shape(oindex), ext_rem_shape], axis=0))
    self.output.placeholder = out
    out.set_shape(self.output.batch_shape)
    out_dim = self.output.dim_tags[0]
    if out_dim.dimension is None and out_dim.dyn_size is None:
      out_dim.dyn_size = tf.reduce_sum(tf.cast(oindex, tf.int32), axis=0)

  @classmethod
  def get_out_data_from_opts(cls, name, sources, axis="T", out_dim=None, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param Dim|str axis:
    :param Dim|None out_dim:
    :rtype: Data
    """
    data = get_concat_sources_data_template(sources, name="%s_output" % name)
    axis = data.get_axis_from_description(axis, allow_int=False)
    in_dim = data.dim_tags[axis]
    data = data.copy_move_axis(old_axis=axis, new_axis=0)
    data = data.copy_with_batch_dim_axis(1)
    if not out_dim:
      out_dim = Dim(kind=in_dim.kind, description="%s:chunking" % name, auto_generated=True)
    data = data.copy_template_replace_dim_tag(axis=0, new_dim_tag=out_dim)
    data.time_dim_axis = 0
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
    orig_shape = tf_util.get_shape(chunking_layer.input_data.placeholder)
    n_time, n_batch = orig_shape[:2]
    in_dim = chunking_layer.output.get_time_dim_tag()
    x = self.input_data.copy()
    axis = x.get_axis_from_description(in_dim)
    in_dim = x.dim_tags[axis]
    x = x.copy_move_axis(old_axis=axis, new_axis=0)
    x = x.copy_with_batch_dim_axis(1)
    x_t = x.placeholder
    if in_dim.dyn_size is not None:
      index = tf.cast(tf_util.sequence_mask_time_major(in_dim.dyn_size), tf.float32)
    else:
      index = tf.fill(tf.shape(x_t)[:2], 1.)
    ext_rem_shape = None
    if x.batch_ndim != 3:
      x_shape = tf_util.get_shape(x_t)
      ext_rem_shape = x_shape[2:]
      x_t = tf.reshape(x_t, x_shape[:2] + [-1])
    from returnn.tf.native_op import unchunk
    out, oindex, factors = unchunk(
      x_t, index=index, chunk_step=chunk_step, chunk_size=chunk_size, n_time=n_time, n_batch=n_batch)
    if ext_rem_shape:
      out = tf.reshape(out, tf.concat([tf.shape(oindex), ext_rem_shape], axis=0))
    self.output.placeholder = out
    out.set_shape(self.output.batch_shape)

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
  def get_out_data_from_opts(cls, name, sources, chunking_layer, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param LayerBase chunking_layer:
    :rtype: Data
    """
    data = get_concat_sources_data_template(sources, name="%s_output" % name)
    in_dim = chunking_layer.output.get_time_dim_tag()
    axis = data.get_axis_from_description(in_dim)
    data = data.copy_move_axis(old_axis=axis, new_axis=0)
    data = data.copy_with_batch_dim_axis(1)
    orig_axis = chunking_layer.kwargs.get("axis", "T")
    orig_source = chunking_layer.sources[0].output
    orig_axis = orig_source.get_axis_from_description(orig_axis, allow_int=False)
    orig_dim = orig_source.dim_tags[orig_axis]
    data = data.copy_template_replace_dim_tag(axis=0, new_dim_tag=orig_dim)
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

  You should try to avoid having the same dims in both sources when they are not reduced
  such that you would end up having some dim twice in the output, e.g. (shared..., I, I).
  You should avoid this because the dim order should never matter
  (https://github.com/rwth-i6/returnn/wiki/RETURNN-principles).
  If you need to perform such an operation, you can use :class:`ReinterpretDataLayer`
  to introduce a new dim tag.

  The reduce dim can also be the sparse dim of one of the sources.
  In this case, it behaves like :class:`GatherLayer`.
  """
  layer_class = "dot"

  def __init__(self,
               reduce=NotSpecified,
               red1=NotSpecified, red2=NotSpecified, var1=NotSpecified, var2=NotSpecified,
               add_var2_if_empty=NotSpecified, debug=False, **kwargs):
    """
    :param str|Dim|tuple[str|Dim]|list[str|Dim] reduce: reduce axes of both sources
    :param str|Dim|tuple[str|Dim]|list[str|Dim] red1: reduce axes of first source
    :param str|Dim|tuple[str|Dim]|list[str|Dim] red2: reduce axes of second source
    :param str|Dim|tuple[str|Dim]|list[str|Dim]|None var1: var axes of first source
    :param str|Dim|tuple[str|Dim]|list[str|Dim]|None var2: var axes of second source
    :param bool add_var2_if_empty: if var2=None, add dim=1 at the end
    :param bool debug: will print debug shapes, etc.

    Earlier defaults:
      red1=-1, red2=-2, var1=-2, var2=-1, add_var2_if_empty=True.
    However, these are bad, for multiple reasons, like using integers, but also in general.
      See https://github.com/rwth-i6/returnn/issues/627 for details.
    """
    from returnn.util import BehaviorVersion
    from returnn.tf.util.basic import prod, get_shape, get_padding_info_dict_ref, mask_dyn_seq_len_nd
    super(DotLayer, self).__init__(**kwargs)
    if reduce is not NotSpecified:
      assert red1 is NotSpecified and red2 is NotSpecified
      red1 = red2 = reduce
    BehaviorVersion.require(
      condition=all(not isinstance(a, int) for a in (red1, red2, var1, var2)),
      message="DotLayer: Axes must be referenced by tag or special specified, not by int.",
      version=3)
    BehaviorVersion.require(
      condition=all(a is not NotSpecified for a in (red1, red2)),
      message="DotLayer: Axes must be specified explicitly. There is no default.",
      version=3)
    BehaviorVersion.require(
      condition=add_var2_if_empty is NotSpecified or not add_var2_if_empty,
      message="DotLayer: add_var2_if_empty not allowed",
      version=3)
    if BehaviorVersion.get() < 3:
      # Earlier defaults: red1=-1, red2=-2, var1=-2, var2=-1, add_var2_if_empty=True.
      red1 = -1 if red1 is NotSpecified else red1
      red2 = -2 if red2 is NotSpecified else red2
      var1 = -2 if var1 is NotSpecified else var1
      var2 = -1 if var2 is NotSpecified else var2
      add_var2_if_empty = True if add_var2_if_empty is NotSpecified else add_var2_if_empty
      axis_desc_allow_int = True
    else:
      # add_var2_if_empty not supported anymore.
      add_var2_if_empty = False
      axis_desc_allow_int = False
    a_out = self.sources[0].output.copy()
    b_out = self.sources[1].output.copy()
    a_reduce_axes = a_out.get_axes_from_description(red1, allow_int=axis_desc_allow_int)
    b_reduce_axes = b_out.get_axes_from_description(red2, allow_int=axis_desc_allow_int)
    if red1 == a_out.sparse_dim:
      assert len(b_reduce_axes) == 1 and len(a_reduce_axes) == 0
    elif red2 == b_out.sparse_dim:
      assert len(a_reduce_axes) == 1 and len(b_reduce_axes) == 0
    else:
      assert a_reduce_axes and b_reduce_axes, "%s: sources %r, red1 %r, red2 %r" % (self, self.sources, red1, red2)
      assert len(a_reduce_axes) == len(b_reduce_axes), (
        "%s: sources %r, red1 %r, red2 %r, reduce axes must match in count" % (self, self.sources, red1, red2))
    if BehaviorVersion.get() >= 3 and (var1 is NotSpecified or var2 is NotSpecified):
      assert var1 is NotSpecified and var2 is NotSpecified
      a_var_axes, b_var_axes = self._auto_var_axes(a_out, b_out, a_reduce_axes, b_reduce_axes)
    else:
      a_var_axes = a_out.get_axes_from_description(var1, allow_int=axis_desc_allow_int)
      b_var_axes = b_out.get_axes_from_description(var2, allow_int=axis_desc_allow_int)
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

    transpose_a = bool(a_var_axes and a_reduce_axes and a_reduce_axes[0] < a_var_axes[0])
    transpose_b = bool(b_var_axes and b_reduce_axes and b_reduce_axes[0] > b_var_axes[0])
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
    assert len(a_reduce_axes) == len(b_reduce_axes) or not a_reduce_axes or not b_reduce_axes
    if len(a_reduce_axes) == len(b_reduce_axes):
      assert all([
        a_out.dim_tags[i1] == b_out.dim_tags[i2] or d1 == d2
        for (d1, d2, i1, i2) in zip(a_reduce_dims, b_reduce_dims, a_reduce_axes, b_reduce_axes)])
    a_var_dim = prod(a_var_dims)
    b_var_dim = prod(b_var_dims)
    a_reduce_dyn_axes = [i for i in a_reduce_axes if a_out.batch_shape[i] is None]
    b_reduce_dyn_axes = [i for i in b_reduce_axes if b_out.batch_shape[i] is None]
    assert len(a_reduce_dyn_axes) == len(b_reduce_dyn_axes) or not a_reduce_axes or not b_reduce_axes
    if a_reduce_dyn_axes and b_reduce_dyn_axes:
      a_pad, b_pad = get_padding_info_dict_ref(a), get_padding_info_dict_ref(b)
      a_pad_values = [a_pad.get(a_out.dim_tags[i], None) for i in a_reduce_dyn_axes]
      b_pad_values = [b_pad.get(b_out.dim_tags[i], None) for i in b_reduce_dyn_axes]
      if set(a_pad_values) == {0}:
        self._info_reduce_mask = "source-0-already-masked"  # it's already masked as needed
      elif set(b_pad_values) == {0}:
        self._info_reduce_mask = "source-1-already-masked"  # it's already masked as needed
      else:
        # We need to apply a mask.
        # We don't need it on both a and b. In case we can either apply it on a or on b,
        # use some very simple heuristic where the mask is maybe cheaper.
        can_mask_a = all(
          set(a_out.dim_tags[i].dyn_size_ext.dim_tags).issubset(a_out.dim_tags) for i in a_reduce_dyn_axes)
        can_mask_b = all(
          set(b_out.dim_tags[i].dyn_size_ext.dim_tags).issubset(b_out.dim_tags) for i in b_reduce_dyn_axes)
        if not can_mask_b or len(a_shape) < len(b_shape):
          assert can_mask_a
          a = mask_dyn_seq_len_nd(a_out, pad_value=0, axes=a_reduce_dyn_axes)
          self._info_reduce_mask = "mask-source-0"
        else:
          assert can_mask_b
          b = mask_dyn_seq_len_nd(b_out, pad_value=0, axes=b_reduce_dyn_axes)
          self._info_reduce_mask = "mask-source-1"
    else:
      self._info_reduce_mask = "none-dynamic"
    a_reduce_dim = prod(a_reduce_dims) if a_reduce_dims else None
    b_reduce_dim = prod(b_reduce_dims) if b_reduce_dims else None
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
    if a_reduce_axes:
      if not transpose_a:
        a = tf.transpose(a, a_rem_axes + a_var_axes + a_reduce_axes)
        a = tf.reshape(a, a_rem_dims + [a_var_dim, a_reduce_dim])
      else:
        a = tf.transpose(a, a_rem_axes + a_reduce_axes + a_var_axes)
        a = tf.reshape(a, a_rem_dims + [a_reduce_dim, a_var_dim])
    else:
      a = tf.transpose(a, a_rem_axes + a_var_axes)  # no reduce axis
      a = tf.reshape(a, a_rem_dims + [a_var_dim])
    if b_reduce_axes:
      if not transpose_b:
        b = tf.transpose(b, b_rem_axes + b_reduce_axes + b_var_axes)
        b = tf.reshape(b, b_rem_dims + [b_reduce_dim, b_var_dim])
      else:
        b = tf.transpose(b, b_rem_axes + b_var_axes + b_reduce_axes)
        b = tf.reshape(b, b_rem_dims + [b_var_dim, b_reduce_dim])
    else:
      b = tf.transpose(b, b_rem_axes + b_var_axes)  # no reduce axis
      b = tf.reshape(b, b_rem_dims + [b_var_dim])
    if a_reduce_axes and b_reduce_axes:
      # `res` will be of shape: a_rem_dims + [a_var_dim, b_var_dim]
      res = tf.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)
      if not b_var_dims and add_var2_if_empty:
        b_var_dims.append(1)
        b_var_axes.append(None)
    else:
      # one of the inputs is sparse
      assert len(a_rem_dims) == len(b_rem_dims)  # batch dims
      batch_dims = len(a_rem_dims)
      if red1 == a_out.sparse_dim:
        res = tf.gather(params=b, indices=a, batch_dims=batch_dims, axis=batch_dims + (1 if transpose_b else 0))
      elif red2 == b_out.sparse_dim:
        res = tf.gather(params=a, indices=b, batch_dims=batch_dims, axis=batch_dims + (0 if transpose_a else 1))
      else:
        raise ValueError("%s: unexpected reduce %s, %s for inputs %s" % (self, red1, red2, self.sources))
      # result_shape(p_shape, i_shape, axis=0) = p_shape[:axis] + i_shape + p_shape[axis+1:]
    res = tf.reshape(res, a_rem_dims + a_var_dims + b_var_dims)
    self.output.placeholder = res

  @staticmethod
  def _auto_var_axes(source1, source2, red1, red2):
    """
    :param Data source1:
    :param Data source2:
    :param list[int] red1:
    :param list[int] red2:
    :return: var1 axes, var2 axes
    :rtype: (list[int], list[int])
    """
    from returnn.util import BehaviorVersion
    is_equal_opts = dict(
      treat_feature_as_spatial=True, allow_same_spatial_dim=True,
      undefined_matches=True, derived_matches=True)
    if BehaviorVersion.get() < 11:
      is_equal_opts["broadcast_matches"] = True
    tags1 = [None if i in red1 else tag for (i, tag) in enumerate(source1.dim_tags)]
    tags2 = [None if i in red2 else tag for (i, tag) in enumerate(source2.dim_tags)]
    var1 = [
      i for i, tag in enumerate(tags1)
      if tag and not any(other and tag.is_equal(other, **is_equal_opts) for other in tags2)]
    var2 = [
      i for i, tag in enumerate(tags2)
      if tag and not any(other and tag.is_equal(other, **is_equal_opts) for other in tags1)]
    return var1, var2

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
      var1 = d.get("var1", NotSpecified)
      var2 = d.get("var2", NotSpecified)
      if var1 is not NotSpecified and var2 is not NotSpecified:
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
  def get_out_data_from_opts(cls, name, sources,
                             reduce=NotSpecified,
                             red1=NotSpecified, red2=NotSpecified, var1=NotSpecified, var2=NotSpecified,
                             add_var2_if_empty=NotSpecified, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param str|Dim|tuple[str|Dim]|list[str|Dim] reduce: reduce axes of both sources
    :param str|Dim|tuple[str|Dim]|list[str|Dim] red1: reduce axes of first source
    :param str|Dim|tuple[str|Dim]|list[str|Dim] red2: reduce axes of second source
    :param str|Dim|tuple[str|Dim]|list[str|Dim]|None var1: var axes of first source
    :param str|Dim|tuple[str|Dim]|list[str|Dim]|None var2: var axes of second source
    :param bool add_var2_if_empty:
    :rtype: Data
    """
    from returnn.util import BehaviorVersion
    from ..util.data import BatchInfo
    assert len(sources) == 2, "dot-layer %r: needs exactly two sources" % (name,)
    # See __init__.
    if reduce is not NotSpecified:
      assert red1 is NotSpecified and red2 is NotSpecified
      red1 = red2 = reduce
    BehaviorVersion.require(
      condition=all(not isinstance(a, int) for a in (red1, red2, var1, var2)),
      message="DotLayer: Axes must be referenced by tag or special specified, not by int.",
      version=3)
    BehaviorVersion.require(
      condition=all(a is not NotSpecified for a in (red1, red2)),
      message="DotLayer: Axes must be specified explicitly. There is no default.",
      version=3)
    BehaviorVersion.require(
      condition=add_var2_if_empty is NotSpecified or not add_var2_if_empty,
      message="DotLayer: add_var2_if_empty not allowed",
      version=3)
    if BehaviorVersion.get() < 3:
      # Earlier defaults: red1=-1, red2=-2, var1=-2, var2=-1, add_var2_if_empty=True.
      red1 = -1 if red1 is NotSpecified else red1
      red2 = -2 if red2 is NotSpecified else red2
      var1 = -2 if var1 is NotSpecified else var1
      var2 = -1 if var2 is NotSpecified else var2
      add_var2_if_empty = True if add_var2_if_empty is NotSpecified else add_var2_if_empty
      axis_desc_allow_int = True
    else:
      # add_var2_if_empty not supported anymore.
      add_var2_if_empty = False
      axis_desc_allow_int = False
    a_out = sources[0].output.copy()
    a_reduce_axes = a_out.get_axes_from_description(red1, allow_int=axis_desc_allow_int)
    b_out = sources[1].output.copy()
    assert not a_out.beam or not b_out.beam or a_out.beam == b_out.beam
    b_reduce_axes = b_out.get_axes_from_description(red2, allow_int=axis_desc_allow_int)
    if red1 == a_out.sparse_dim:
      assert len(b_reduce_axes) == 1 and len(a_reduce_axes) == 0
    elif red2 == b_out.sparse_dim:
      assert len(a_reduce_axes) == 1 and len(b_reduce_axes) == 0
    else:
      assert a_reduce_axes and b_reduce_axes, "%s: sources %r, red1 %r, red2 %r" % (name, sources, red1, red2)
      assert len(a_reduce_axes) == len(b_reduce_axes), (
        "%s: sources %r, red1 %r, red2 %r, reduce axes must match in count" % (name, sources, red1, red2))
    if BehaviorVersion.get() >= 3 and (var1 is NotSpecified or var2 is NotSpecified):
      assert var1 is NotSpecified and var2 is NotSpecified
      a_var_axes, b_var_axes = cls._auto_var_axes(a_out, b_out, a_reduce_axes, b_reduce_axes)
    else:
      a_var_axes = a_out.get_axes_from_description(var1, allow_int=axis_desc_allow_int)
      b_var_axes = b_out.get_axes_from_description(var2, allow_int=axis_desc_allow_int)
    assert not set(a_reduce_axes).intersection(a_var_axes)
    assert not set(b_reduce_axes).intersection(b_var_axes)
    a_rem_axes = [i for i in range(a_out.batch_ndim) if i not in a_var_axes + a_reduce_axes]
    b_rem_axes = [i for i in range(b_out.batch_ndim) if i not in b_var_axes + b_reduce_axes]
    assert len(a_rem_axes) == len(b_rem_axes), "%s: sources %r, red1 %r, red2 %r, var1 %r, var2 %r" % (
      name, sources, red1, red2, var1, var2)

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
        SpatialDim("%s:dot:dummy-var2" % name, dimension=1, auto_generated=True))

    dim_tags = list(a_rem_dims + a_var_dims + b_var_dims)
    return Data(
      name="%s_output" % name,
      dim_tags=dim_tags,
      time_dim_axis=time_dim_axis,
      dtype=a_out.dtype if not a_out.sparse else b_out.dtype,
      batch=BatchInfo.get_common_batch_info([src.batch for src in (a_out, b_out)]),
      beam=SearchBeam.get_combined_beam(a_out.beam, b_out.beam))


class ShiftAxisLayer(_ConcatInputLayer):
  """
  Shifts the dimensions in an axis around by slicing and optional padding.
  This layer may change the axis-dimension.

  This name might be confusing. No axis will be shifted here. See :class:`SwapAxesLayer` for that.

  Also see :class:`SliceLayer`.
  """
  layer_class = "shift_axis"

  def __init__(self, axis, amount, pad=True, pad_value=0, adjust_size_info=True, **kwargs):
    """
    :param str|Dim|int axis: single axis to shift
    :param int amount: number of elements to shift
                   (<0 for left-shift, >0 for right-shift)
    :param bool pad: preserve shape by padding
    :param int|float|bool pad_value: padding value
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
      shifted = tf.pad(shifted, paddings, constant_values=pad_value)
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
      from ..util.data import Dim
      Dim(
        kind=Dim.Types.Spatial, description="%s_shift_axis" % self.name,
        dyn_size=new_size, batch=self.output.batch,
        src_data=self.output, src_axis=axis, auto_generated=True)
      self.output.size_placeholder[axis_wob] = new_size

  @classmethod
  def get_out_data_from_opts(cls, name, sources, amount, axis, pad=True, adjust_size_info=True, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param int amount:
    :param str axis:
    :param bool pad:
    :param bool adjust_size_info:
    :rtype: Data
    """
    from ..util.data import Dim
    out = get_concat_sources_data_template(sources, name="%s_output" % name)
    if not adjust_size_info and pad:
      return out
    assert isinstance(amount, int)
    axis = out.get_axis_from_description(axis)
    tag = out.dim_tags[axis]
    dim = None if tag.dimension is None else max(0, tag.dimension - abs(amount))
    tag = Dim(kind=tag.kind, description="%s_shift_axis" % name, dimension=dim, auto_generated=True)
    return out.copy_template_replace_dim_tag(axis=axis, new_dim_tag=tag)


class ResizeLayer(_ConcatInputLayer):
  """
  Resizes the input, i.e. upsampling or downsampling.
  Supports different kinds, such as linear interpolation or nearest-neighbor.
  """
  layer_class = "resize"

  def __init__(self, factor, axis, out_dim=None, kind="nn", fill_value=None, fill_dropout=None, **kwargs):
    """
    :param int factor:
    :param Dim|str axis: the axis to resize
    :param Dim|None out_dim:
    :param str kind: "linear", "nn"/"nearest_neighbor", "cubic", "fill"
    :param None|int|float fill_value: if kind=="fill"
    :param float|None fill_dropout: if set, will dropout in the same axis
    """
    out_dim  # noqa  # via get_out_data_from_opts
    super(ResizeLayer, self).__init__(**kwargs)
    # self.output.shape and self.output.batch_dim_axis are already set here via self.get_out_data_from_opts().
    input_data = self.input_data.copy_as_batch_major()
    axis = input_data.get_axis_from_description(axis)
    assert axis > 0, "batch-dim resize not supported"
    input_data = input_data.copy_move_axis(old_axis=axis, new_axis=1)
    axis = 1

    # images expected shape: [batch, height, width, channels]
    remaining_axes = [i for i in range(self.output.batch_ndim) if i not in (0, axis)]
    x = dimshuffle(input_data.placeholder, [0, axis, 'x'] + remaining_axes)  # [batch,height,width] + remaining_axes
    from returnn.tf.util.basic import get_shape, optional_mul
    shape = get_shape(input_data.placeholder)
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
      out_dyn_size = input_data.dim_tags[axis].dyn_size
      if out_dyn_size is not None:
        out_dyn_size = out_dyn_size * factor
        orig_mask = tf.sequence_mask(
          out_dyn_size, maxlen=new_size, dtype=tf.bool)  # (batch,new_size)
        out_dyn_size = tf.reduce_sum(tf.cast(tf.logical_and(mask, orig_mask), tf.int32), axis=1)
        self.output.dim_tags[axis].dyn_size = out_dyn_size
    self.output.placeholder = x

  @classmethod
  def get_out_data_from_opts(cls, factor, axis, sources, name, fill_dropout=None, out_dim=None, **kwargs):
    """
    :param int factor:
    :param Dim|str axis:
    :param list[LayerBase] sources:
    :param str name:
    :param float|None fill_dropout:
    :param Dim|None out_dim:
    :rtype: Data
    """
    out = get_concat_sources_data_template(sources).copy_as_batch_major()
    axis = out.get_axis_from_description(axis)
    out = out.copy_move_axis(old_axis=axis, new_axis=1)
    out = out.copy_template(name="%s_output" % name)
    axis = 1
    assert axis != out.batch_dim_axis, "batch-dim resize not supported"
    tag = out.dim_tags[axis]
    if fill_dropout:
      out_dim_ = Dim(kind=tag.kind, description="%s_resize" % name, auto_generated=True)  # unknown dim
    else:
      out_dim_ = tag * factor
    if out_dim:
      out_dim_.declare_same_as(out_dim)
    return out.copy_template_replace_dim_tag(axis=axis, new_dim_tag=out_dim_)


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

  def __init__(self, symbol, axis="T", out_dim=None, **kwargs):
    """
    :param int symbol:
    :param Dim|str axis: the axis to operate over, to potentially remove frames
    :param Dim|None out_dim: derived from the dim of axis, the reduced new dim
    """
    super(RemoveLayer, self).__init__(out_dim=out_dim, **kwargs)
    if symbol < 0:
      symbol += self.output.dim
      assert symbol > 0

    # I currently do not have a good idea how to make this efficient.
    in_data = self.sources[0].output.copy_as_batch_major()
    assert in_data.sparse
    axis = in_data.get_axis_from_description(axis, allow_int=False)
    assert in_data.batch_ndim == 2
    in_seqs = in_data.placeholder
    in_mask = tf.logical_and(tf.not_equal(in_seqs, symbol), in_data.get_sequence_mask_broadcast(axis=axis))
    out_seq_lens = tf_compat.v1.count_nonzero(in_mask, axis=axis, dtype=tf.int32)  # (batch,)
    out_dim = self.output.dim_tags[axis]
    if out_dim.dimension is None and out_dim.dyn_size is None:
      out_dim.dyn_size = out_seq_lens
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

  @classmethod
  def get_out_data_from_opts(cls, name, sources, axis="T", out_dim=None, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param Dim|str axis:
    :param Dim|None out_dim:
    :rtype: Data
    """
    assert len(sources) == 1, "%s layer %r: must have exactly one source" % (cls, name)
    assert sources[0].output.sparse, "%s layer %r: assumes sparse data" % (cls, name)
    out = sources[0].output.copy(name="%s_output" % name).copy_as_batch_major()
    axis = out.get_axis_from_description(axis, allow_int=False)
    in_dim = out.dim_tags[axis]
    if not out_dim:
      out_dim = Dim(
        kind=in_dim.kind, description="%s_removed_items", dimension=None, derived_from_tag=in_dim, auto_generated=True)
    return out.copy_template_replace_dim_tag(axis=axis, new_dim_tag=out_dim)


class CombineLayer(LayerBase):
  """
  Applies a binary operation, such as addition, to all sources while accumulating the partial results.
  In the first step, the binary operation is performed on the first two sources.
  After the first step, the previous results is always the left-hand operator.

  Its basic working is similar to the `reduce` function used in functional programming.
  Also see :class:`ActivationLayer`, or :class:`CompareLayer`.
  """
  layer_class = "combine"
  recurrent = True  # in case of eval, we cannot really know

  # noinspection PyShadowingBuiltins
  def __init__(self, kind, sources, allow_broadcast_all_sources=NotSpecified,
               activation=None, with_bias=False,
               eval=None, eval_locals=None, eval_for_output_loss=False,
               **kwargs):
    """
    :param str kind:
      currently accepted values are `average`, `add`, `sub`, `mul`, `truediv`, `floordiv`, `mod`, `pow`,
      `maximum`, `minimum`,
      `logical_and`, `logical_or`,
      `squared_difference`,
      or `eval`,
      or any function in the tf.math or tf namespace.
    :param list[LayerBase] sources:
    :param bool|NotSpecified allow_broadcast_all_sources: allow broadcasting for all sources.
      e.g. shape [A] + [B] -> shape [A,B]. by default disabled, and there must be some source with all dims.
    :param str|None activation: if provided, activation function to apply, e.g. "tanh" or "relu"
    :param bool with_bias: if given, will add a trainable bias tensor
    :param str|callable eval: for kind="eval", will eval this string. or function. see :func:`_op_kind_eval`
    :param dict[str]|None eval_locals: locals for eval
    :param bool eval_for_output_loss: will do the same eval on layer.output_loss
    """
    allow_broadcast_all_sources  # noqa  # via get_out_data_from_opts
    super(CombineLayer, self).__init__(sources=sources, **kwargs)
    if kind != "eval":
      self.recurrent = False
    x = self._execute_op(kind=kind, sources=sources, eval_str=eval, eval_locals=eval_locals)
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
  def get_out_data_from_opts(cls, network, sources, eval_locals=None, n_out=NotSpecified, out_type=None,
                             allow_broadcast_all_sources=NotSpecified, out_shape=None,
                             **kwargs):
    """
    :param returnn.tf.network.TFNetwork network:
    :param list[LayerBase] sources:
    :param dict[str]|None eval_locals: locals for eval, will also pass to out_type is out_type is a function
    :param int|None|NotSpecified n_out:
    :param bool|NotSpecified allow_broadcast_all_sources:
    :param dict[str]|None|(()->Data) out_type:
    :param set[Dim|_MarkedDim]|tuple|list|None out_shape: verifies the output shape (dim tags)
    :rtype: Data
    """
    out_type_ = {}
    if sources:
      if allow_broadcast_all_sources is NotSpecified:
        inside_rec_time_dim = network.get_inside_rec_time_dim(inside_loop=True)
        over_rec_time_dim = network.get_inside_rec_time_dim(inside_loop=False)
        allow_broadcast_all_sources = NotSpecified
        if out_shape is not None:
          allow_broadcast_all_sources = True
        elif out_type and isinstance(out_type, dict) and ("shape" in out_type or "dim_tags" in out_type):
          allow_broadcast_all_sources = True
        elif over_rec_time_dim and not inside_rec_time_dim:  # moved out of loop
          allow_broadcast_all_sources = True  # we already checked the validity at template construction
      out_type_.update(
        Data.get_common_data(
          [s.output for s in sources], allow_broadcast_all_sources=allow_broadcast_all_sources,
          name="%s_output" % kwargs["name"]).get_kwargs())
    if n_out is not NotSpecified:
      out_type_["dim"] = n_out
    if out_type:
      if isinstance(out_type, dict):
        if "shape" in out_type:
          out_type_.pop("dim_tags", None)
          out_type_.pop("batch_dim_axis", None)
          out_type_.pop("feature_dim_axis", None)
          out_type_.pop("time_dim_axis", None)
        if "dim" in out_type:
          if out_type_.pop("sparse_dim", None):
            out_type_["sparse"] = True
        if "sparse" in out_type:
          out_type_.pop("sparse_dim", None)
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
    return super(CombineLayer, cls).get_out_data_from_opts(
      network=network, sources=sources, n_out=n_out, out_type=out_type_, **kwargs)

  @staticmethod
  def _op_dense_fn(sources, fn, output_template):
    """
    :param list[LayerBase] sources:
    :param ((x1,x2) -> y) fn: function to perform on x1 and x2
    :param Data output_template:
    :rtype: tf.Tensor
    """
    # All the dense element-wise functions should be able to deal with broadcasting.
    x = sources[0].output.copy_compatible_to(output_template, check_sparse=False).placeholder
    assert x is not None, "sources[0].output missing placeholder? %r" % sources[0]
    for source in sources[1:]:
      x2 = source.output.copy_compatible_to(output_template, check_sparse=False).placeholder
      assert x2 is not None, "source.output missing placeholder? %r" % source
      x = fn(x, x2)
    return x

  def _op_kind_average(self, sources):
    """
    :param list[LayerBase] sources:
    :rtype: tf.Tensor
    """
    x = self._op_dense_fn(sources, tf.add, self.output)
    x /= len(sources)
    return x

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

  def _execute_op(self, kind, sources, eval_str=None, eval_locals=None):
    """
    :param str kind:
    :param list[LayerBase] sources:
    :param str|callable eval_str:
    :param dict[str]|None eval_locals:
    :rtype: tf.Tensor
    """
    if kind == "eval" or eval_str:
      assert kind == "eval" and eval_str
      return self._op_kind_eval(sources, eval_str=eval_str, eval_locals=eval_locals)

    kind = {"sub": "subtract", "mul": "multiply"}.get(kind, kind)
    if hasattr(tf, "math") and hasattr(tf.math, kind):
      tf_func = getattr(tf.math, kind)
    elif hasattr(tf, kind):
      tf_func = getattr(tf, kind)
    else:
      tf_func = None
    if tf_func:
      return self._op_dense_fn(sources, tf_func, self.output)

    return getattr(self, "_op_kind_%s" % kind)(sources)


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

  def __init__(self, kind="equal", value=None, allow_broadcast_all_sources=NotSpecified, **kwargs):
    """
    :param str kind: which comparison operation to use, e.g. "equal", "greater", "less"
      or other supported TF comparison ops
    :param float|int|None value: if specified, will also compare to this
    :param bool|NotSpecified allow_broadcast_all_sources: allow broadcasting for all sources.
      e.g. shape [A] + [B] -> shape [A,B]. by default disabled, and there must be some source with all dims.
    """
    allow_broadcast_all_sources  # noqa  # via get_out_data_from_opts
    super(CompareLayer, self).__init__(**kwargs)
    assert len(self.sources) >= 1
    if value is None:
      assert len(self.sources) >= 2, "{} requires at least two elements to compare".format(self)
    op = getattr(tf, kind)  # e.g. tf.equal
    from returnn.tf.util.basic import opt_logical_and
    common_data = self.output
    x = self.sources[0].output.copy_compatible_to(common_data, check_dtype=False, check_sparse=False).placeholder
    r_last = True
    for source in self.sources[1:]:
      x2 = source.output.copy_compatible_to(common_data, check_dtype=False, check_sparse=False).placeholder
      r_last = opt_logical_and(r_last, op(x, x2))
      x = x2
    if value is not None:
      r_last = opt_logical_and(r_last, op(x, value))
    self.output.placeholder = r_last

  @classmethod
  def get_out_data_from_opts(cls, sources, allow_broadcast_all_sources=NotSpecified,
                             n_out=NotSpecified, out_type=None, out_shape=None, **kwargs):
    """
    :param list[LayerBase] sources:
    :param bool|NotSpecified allow_broadcast_all_sources:
    :param int|None|NotSpecified n_out:
    :param dict[str]|None out_type:
    :param dict[str]|None out_shape:
    :rtype: Data
    """
    out_type_ = {}
    if sources:
      if allow_broadcast_all_sources is NotSpecified:
        if out_shape is not None:
          allow_broadcast_all_sources = True
        elif out_type and isinstance(out_type, dict) and ("shape" in out_type or "dim_tags" in out_type):
          allow_broadcast_all_sources = True
      out_type_.update(
        Data.get_common_data(
          [s.output for s in sources], allow_broadcast_all_sources=allow_broadcast_all_sources,
          name="%s_output" % kwargs["name"]).get_kwargs())
    if n_out is not NotSpecified:
      out_type_["dim"] = n_out
    elif isinstance(out_type, dict) and out_type.get("sparse", False):
      out_type_["dim"] = 2
    out_type_.pop("sparse_dim", None)
    out_type_["dtype"] = "bool"
    out_type_["vocab"] = None
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
  Wrapper around ``tf.where()`` (or more generically :func:`returnn.tf.util.basic.where_bc`),
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
    out.sparse_dim = true_data.sparse_dim
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
    from ..util.data import Dim
    super(CondLayer, self).__init__(**kwargs)
    self._parent_scope = os.path.dirname(tf_compat.v1.get_variable_scope().name)
    self.condition_desc = condition
    self.condition_layer = self._make_layer("condition", self.condition_desc)
    self.true_layer_desc = true_layer
    self.true_layer = None  # type: typing.Optional[LayerBase]
    self.false_layer_desc = false_layer
    self.false_layer = None  # type: typing.Optional[LayerBase]
    assert self.condition_layer.output.batch_ndim == 0 and self.condition_layer.output.dtype == "bool"
    x, sizes = tf_util.cond(
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
      old_tag = Dim.get_tag_from_size_tensor(old_size)
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
    layer_desc["encapsulate"] = True
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
    layer_desc["encapsulate"] = True
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


class TopKLayer(LayerBase):
  """
  Basically wraps tf.nn.top_k.

  Directly returns the top_k values.
  The indices are accessible via the "indices" sub-layer.

  For an input [B,D] with axis=D, the output and indices values are shape [B,K].

  It's somewhat similar to :class:`ReduceLayer` with max and argmax.
  The axis dim is reduced and then a new dim for K is added.

  Axis can also cover multiple axes, such as [beam,classes].
  In that cases, there is not a single "indices" sub-layer,
  but sub-layers "indices0" .. "indices{N-1}"
  corresponding to each axis, in the same order.

  All other axes are treated as batch dims.
  """
  layer_class = "top_k"

  # noinspection PyShadowingBuiltins
  def __init__(self, axis, k, k_dim=None, sorted=True, **kwargs):
    """
    :param Dim|str|list[Dim|str] axis: the axis to do the top_k on, which is reduced
    :param int|LayerBase k: the "K" in "TopK"
    :param Dim|None k_dim: the output dim tag corresponding to k
    :param bool sorted:
    """
    super(TopKLayer, self).__init__(**kwargs)
    in_data = self.sources[0].output

    if isinstance(axis, (str, Dim)):
      single_axis = True
      axes = [in_data.get_dim_tag_from_description(axis)]
    else:
      assert len(axis) > 0
      single_axis = False
      axes = [in_data.get_dim_tag_from_description(a) for a in axis]

    # Remaining axes are batch dims. Move them to front, and the axes to back.
    remaining_axes = [a for a in in_data.dim_tags if a not in axes]
    in_data = in_data.copy_transpose(
      [in_data.get_axis_from_description(a) for a in remaining_axes + axes])
    assert in_data.dim_tags == tuple(remaining_axes + axes)

    # Merge the axes because top_k will do a joint search over them.
    if len(axes) > 1:
      merged_axis = axes[0]
      for a in axes[1:]:
        merged_axis = merged_axis * a
      in_ = tf.reshape(
        in_data.placeholder,
        shape=tf_util.get_shape(in_data.placeholder)[:len(remaining_axes)] + [merged_axis.get_dim_value()])
      in_data = in_data.copy_template_new_dim_tags(remaining_axes + [merged_axis])
      in_data.placeholder = in_

    if isinstance(k, LayerBase):
      if k_dim in k.output.dim_tags:
        k_dim = k.output.get_dim_tag_from_description(k_dim)
        k = k_dim.get_dim_value()
      elif k.output.batch_shape == ():  # scalar, normal case
        k = k.output.placeholder
      else:
        k = tf.reduce_max(k.output.placeholder)
      if k_dim.dimension is not None:
        from tensorflow.python.framework import tensor_util
        k_ = tensor_util.constant_value(k)
        assert k_ is not None and k_ == k_dim.dimension
    assert isinstance(k, (int, tf.Tensor))
    values, indices = tf.nn.top_k(in_data.placeholder, k=k, sorted=sorted)
    self.output.placeholder = values
    sub_outputs = {}
    if single_axis:
      sub_outputs["indices"] = (indices, axes[0])
    else:
      for i, a in reversed(list(enumerate(axes))):
        assert isinstance(a, Dim)
        sub_outputs["indices%i" % i] = (indices % a.dimension, a)
        indices = indices // a.dimension
    self._sub_layers = {}
    for key, (v, a) in sub_outputs.items():
      sub_out_data = self.output.copy_template(name="%s/%s" % (self.name, key))
      sub_out_data.dtype = "int32"
      sub_out_data.sparse_dim = a
      sub_out_data.placeholder = v
      self._sub_layers[key] = InternalLayer(
        name="%s/%s" % (self.name, key), network=self.network, output=sub_out_data)

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param get_layer:
    """
    super(TopKLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    if isinstance(d.get("k", None), str):
      d["k"] = get_layer(d["k"])
    if not d.get("k_dim", None):
      k = d.get("k", None)  # should always be in there but throw errors later
      if isinstance(k, (str, LayerBase)):
        k = None
      d["k_dim"] = SpatialDim(d.get("_name", "<unk>") + ":topk", k)

  @classmethod
  def _get_out_data(cls, name, in_data, axis, k_dim, for_indices=None):
    """
    :param str name:
    :param Dim|str|list[Dim|str] axis: the axis to do the top_k on, which is reduced
    :param Dim k_dim: the "K" in "TopK"
    :param int|None for_indices:
    :rtype: Data
    """
    assert isinstance(k_dim, Dim)
    if not isinstance(axis, (list, tuple)):
      axis = [axis]
    axis = [in_data.get_dim_tag_from_description(a) for a in axis]
    out_dims = [dim for dim in in_data.dim_tags if dim not in axis] + [k_dim]
    out_data = in_data.copy_template(name=name).copy_template_new_dim_tags(out_dims)
    if for_indices is not None:
      assert 0 <= for_indices < len(axis)
      out_data.dtype = "int32"
      out_data.sparse_dim = axis[for_indices]
    return out_data

  @classmethod
  def get_out_data_from_opts(cls, name, network, sources, axis, k, k_dim, **kwargs):
    """
    :param str name:
    :param returnn.tf.network.TFNetwork network:
    :param list[LayerBase] sources:
    :param Dim|str|list[Dim|str] axis: the axis to do the top_k on, which is reduced
    :param int|LayerBase k: the "K" in "TopK"
    :param Dim|None k_dim: the output dim tag corresponding to k
    :rtype: Data
    """
    assert len(sources) == 1
    in_data = sources[0].output
    assert isinstance(k_dim, Dim)  # via transform_config_dict
    if isinstance(k, LayerBase):
      if k_dim.dimension is None:
        k_dim = k_dim.get_for_batch_ctx(k.get_batch_info(), k.output.control_flow_ctx)
        if not k_dim.dyn_size_ext or k_dim.dyn_size_ext.placeholder is None:
          k_dim.dyn_size_ext = k.output.copy()
          if k_dim.dyn_size_ext.placeholder is not None:
            tag = Dim.get_tag_from_size_tensor(k_dim.dyn_size_ext.placeholder)
            if tag:
              k_dim.declare_same_as(tag)
            else:
              k_dim.set_tag_on_size_tensor(k_dim.dyn_size_ext.placeholder)
    return cls._get_out_data(name=name + "_output", in_data=in_data, axis=axis, k_dim=k_dim, for_indices=None)

  def get_sub_layer(self, layer_name):
    """
    :param str layer_name: sub layer name
    :rtype: LayerBase|None
    """
    return self._sub_layers.get(layer_name, None)

  @classmethod
  def get_sub_layer_out_data_from_opts(cls, layer_name, parent_layer_kwargs):
    """
    :param str layer_name: sub layer name
    :param dict[str] parent_layer_kwargs:
    :return: Data template, class type of sub-layer, layer opts (transformed)
    :rtype: (Data, type, dict[str])|None
    """
    if not layer_name.startswith("indices"):
      return None
    name = parent_layer_kwargs["_name"]
    sources = parent_layer_kwargs["sources"]
    in_data = sources[0].output
    axis = parent_layer_kwargs["axis"]
    k_dim = parent_layer_kwargs["k_dim"]
    assert isinstance(k_dim, Dim)  # via transform_config_dict
    if layer_name == "indices":
      assert isinstance(axis, (str, Dim))
      for_indices = 0
    else:
      assert layer_name.startswith("indices")
      assert isinstance(axis, (tuple, list))
      for_indices = int(layer_name[len("indices"):])
    out_data = cls._get_out_data(
      name="%s/%s" % (name, layer_name), in_data=in_data, axis=axis, k_dim=k_dim, for_indices=for_indices)
    return out_data, InternalLayer, {}


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
    subnet = network.make_subnet(name, opts=d)
    cls._update_for_rec_previous_layer(d.get("rec_previous_layer"), d["subnetwork"], subnet.net)
    d["_subnet"] = subnet
    if d.get("encapsulate", False):
      subnet.construct_all(parent_get_layer=get_layer)
    # In case of non-template construction, this will trigger the non-template construction of our "output" sublayer.
    d["_output"] = subnet.construct_layer("output", parent_get_layer=get_layer)
    d["_from"] = d.get("from", "data")  # cache this
    d["from"] = []  # disable now. we should get them in the template construction when needed
    super(SubnetworkLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)

  @classmethod
  def get_sub_layer_out_data_from_opts(cls, layer_name, parent_layer_kwargs):
    """
    :param str layer_name: name of the sub_layer (right part of '/' separated path)
    :param dict[str] parent_layer_kwargs: kwargs for the parent layer (as kwargs in cls.get_out_data_from_opts())
    :return: Data template, class type of sub-layer, layer opts (transformed)
    :rtype: (Data, type, dict[str])|None
    """
    from returnn.tf.network import Subnetwork
    from returnn.tf.layers.rec import _TemplateLayer
    subnet = parent_layer_kwargs["_subnet"]
    assert isinstance(subnet, Subnetwork)
    # Should be constructed already. If not, make sure is_output_layer is set.
    layer = subnet.net.get_layer(layer_name)
    if isinstance(layer, _TemplateLayer):
      layer_class = layer.layer_class_type
    else:
      layer_class = layer.__class__
    return layer.output, layer_class, layer.kwargs

  @classmethod
  def cls_get_sub_network(cls, name, network, layer_desc):
    """
    :param str name:
    :param returnn.tf.network.TFNetwork network:
    :param dict[str] layer_desc:
    :rtype: returnn.tf.network.Subnetwork|None
    """
    if layer_desc.get("encapsulate", False):
      return None
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
  def get_rec_initial_extra_outputs(cls, batch_dim, rec_layer, encapsulate=False, **kwargs):
    """
    :param tf.Tensor batch_dim: for this layer, might be with beam
    :param returnn.tf.layers.rec.RecLayer rec_layer:
    :param bool encapsulate:
    :rtype: dict[str,tf.Tensor]
    """
    if not encapsulate:
      return {}
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
      with cl.cls_setup_scope(**layer_desc):
        d = cl.get_rec_initial_extra_outputs(
          batch_dim=batch_dim, rec_layer=rec_layer, **layer_desc)
        for key, value in d.items():
          extra_outputs["%s/%s" % (layer_name, key)] = value
    return extra_outputs

  @classmethod
  def get_rec_initial_extra_outputs_shape_invariants(cls, encapsulate=False, **kwargs):
    """
    :param bool encapsulate:
    :return: optional shapes for the tensors by get_rec_initial_extra_outputs
    :rtype: dict[str,tf.TensorShape]
    """
    if not encapsulate:
      return {}
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
      with cl.cls_setup_scope(**layer_desc):
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

  def __init__(self, shape, dtype="float32",
               add_batch_axis=False, add_time_axis=False,
               trainable=True,
               init=None, init_by_layer=None,
               param_name=None,
               **kwargs):
    """
    :param tuple[int|Dim]|list[int|Dim] shape:
    :param str dtype:
    :param bool add_batch_axis:
    :param bool add_time_axis:
    :param bool trainable:
    :param str|float|int|None init: see :func:`returnn.tf.util.basic.get_initializer`. 0 by default.
      Alternatively, you can also use option `init_by_layer`.
    :param LayerBase|None init_by_layer:
    :param str|None param_name: self.name (layer name) by default
    """
    shape  # noqa  # used in get_out_data_from_opts
    super(VariableLayer, self).__init__(trainable=trainable, **kwargs)
    assert not self.sources, "%s: does not expect any sources" % self
    self.init_by_layer = init_by_layer
    dim_tags = list(self.output.dim_tags)
    if add_batch_axis:
      assert dim_tags[0].is_batch_dim()
      dim_tags = dim_tags[1:]
    if add_time_axis:
      assert dim_tags[0].dimension == 1
      dim_tags = dim_tags[1:]
    shape_ = [d.dimension for d in dim_tags]
    assert all(shape_), self.output  # all static
    with self.var_creation_scope():
      if init_by_layer is None:
        if init is None:
          init = 0
        initializer = tf_util.get_initializer(
          init, dtype=dtype, seed=self.network.random.randint(2 ** 31), eval_local_ns={"layer": self})
      else:
        assert init_by_layer is not None
        out_data_base = Data(name=self.output.name, dim_tags=dim_tags, dtype=dtype)
        initializer = init_by_layer.output.copy_compatible_to(out_data_base).placeholder
        shape_ = None  # get_variable requires shape to be not defined when the initializer is another tensor
      var = self.add_param(tf_compat.v1.get_variable(
        name=param_name or self.name, shape=shape_, dtype=dtype,
        initializer=initializer, trainable=trainable),
        axes_split_info=[d.axis_split_info() for d in dim_tags])
      out = var
      if add_time_axis:
        out = tf.expand_dims(out, axis=0)
      if add_batch_axis:
        # Unbroadcast to not confuse some other layers
        batch_dim = self.output.get_batch_dim()
        out = tf_util.expand_dims_unbroadcast(out, axis=0, dim=batch_dim)
    self.output.placeholder = out

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    deps = super(VariableLayer, self).get_dep_layers()
    if self.init_by_layer:
      deps.append(self.init_by_layer)
    return deps

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
    if d.get("init_by_layer", None):
      d["init_by_layer"] = get_layer(d["init_by_layer"])

  @classmethod
  def get_out_data_from_opts(cls, name, network,
                             shape, dtype="float32", add_batch_axis=False, add_time_axis=False, **kwargs):
    """
    :param str name:
    :param returnn.tf.network.TFNetwork network:
    :param tuple[int|Dim]|list[int|Dim] shape:
    :param str dtype:
    :param bool add_batch_axis:
    :param bool add_time_axis:
    :rtype: Data
    """
    assert isinstance(shape, (list, tuple))
    assert len(shape) == 0 or all(shape)
    dim_tags = []
    for i, d in enumerate(shape):
      if isinstance(d, Dim):
        assert d.dimension is not None, "%r: need static dims but got %r" % (name, d)
      elif isinstance(d, int):
        d = Dim(
          kind=Dim.Types.Spatial if i < len(shape) - 1 else Dim.Types.Feature,
          description="%s:static:%i" % (name, i), auto_generated=True,
          dimension=d)
      else:
        raise TypeError("Layer %r: invalid type %s in shape %r" % (name, type(d), shape))
      dim_tags.append(d)
    if add_time_axis:
      dim_tags.insert(
        0, Dim(kind=Dim.Types.Time, description="%s:dummy-time" % name, dimension=1, auto_generated=True))
    if add_batch_axis:
      dim_tags.insert(
        0, Dim(kind=Dim.Types.Batch, description="batch", batch=network.get_global_batch_info()))
    return Data(
      name="%s_output" % name, dim_tags=dim_tags, dtype=dtype,
      batch=network.get_global_batch_info() if add_batch_axis else None)


class TrainFlagLayer(LayerBase):
  """
  Returns the train flag (bool scalar) of the current network.
  """
  layer_class = "train_flag"

  def __init__(self, **kwargs):
    super(TrainFlagLayer, self).__init__(**kwargs)
    self.output.placeholder = tf.convert_to_tensor(self.network.train_flag)

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param returnn.tf.network.TFNetwork network:
    :param get_layer:
    """
    d.setdefault("from", ())
    super(TrainFlagLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)

  @classmethod
  def get_out_data_from_opts(cls, name, **kwargs):
    """
    :param str name:
    :rtype: Data
    """
    return Data(name="%s_output" % name, dim_tags=(), dtype="bool")


class GlobalTrainStepLayer(LayerBase):
  """
  Returns the global train step (int64 scalar).
  """
  layer_class = "global_train_step"

  def __init__(self, **kwargs):
    super(GlobalTrainStepLayer, self).__init__(**kwargs)
    self.output.placeholder = tf.convert_to_tensor(self.network.global_train_step)

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param returnn.tf.network.TFNetwork network:
    :param get_layer:
    """
    d.setdefault("from", ())
    super(GlobalTrainStepLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)

  @classmethod
  def get_out_data_from_opts(cls, name, **kwargs):
    """
    :param str name:
    :rtype: Data
    """
    return Data(name="%s_output" % name, dim_tags=(), dtype="int64")


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
    :return: Data template, class type of sub-layer, layer opts (transformed)
    :rtype: (Data, type, dict[str])|None
    """
    if layer_name not in ["loss", "error"]:
      return None
    return cls.get_out_data_from_opts(**parent_layer_kwargs), InternalLayer, {}  # same type

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
    :return: Data template, class type of sub-layer, layer opts (transformed)
    :rtype: (Data, type, dict[str])|None
    """
    if layer_name == "scores":
      return Data(name="align_scores", shape=(), dtype="float32"), InternalLayer, {}
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
    opts["sparse_dim"] = src.dim_tags[src.feature_dim_axis]
    return Data(**opts)


class SparseSoftmaxCrossEntropyWithLogitsLayer(LayerBase):
  """
  This is a simple wrapper for tf.nn.sparse_softmax_cross_entropy_with_logits.
  """
  layer_class = "sparse_softmax_cross_entropy_with_logits"

  def __init__(self, logits, targets, axis=None, **kwargs):
    """
    :param LayerBase logits:
    :param LayerBase targets:
    :param Dim|str|None axis: feature dim by default
    """
    super(SparseSoftmaxCrossEntropyWithLogitsLayer, self).__init__(**kwargs)
    self.logits = logits
    self.targets = targets
    logits_data = logits.output.copy()
    if axis is None:
      assert logits_data.have_feature_axis()
      logits_data = logits_data.copy_move_axis(logits_data.feature_dim_axis, -1)
    else:
      axis_int = logits_data.get_axis_from_description(axis, allow_int=False)
      logits_data = logits_data.copy_move_axis(axis_int, -1)
    logits_data.feature_dim_axis = logits_data.batch_ndim - 1
    assert "int" in targets.output.dtype
    if isinstance(axis, Dim):
      # When we specify the dim tag, be strict
      assert targets.output.sparse and targets.output.sparse_dim == axis
    else:
      if targets.output.sparse:  # allow dense int values as well without further checking
        assert targets.output.dim == logits_data.dim
    targets_data = targets.output.copy_compatible_to(self.output, check_sparse=False, check_dtype=False, add_dims=False)
    self.output.placeholder = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits_data.placeholder, labels=targets_data.placeholder)

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    return [self.logits, self.targets]

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param get_layer:
    """
    d.setdefault("from", [])
    super(SparseSoftmaxCrossEntropyWithLogitsLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["logits"] = get_layer(d["logits"])
    d["targets"] = get_layer(d["targets"])

  @classmethod
  def get_out_data_from_opts(cls, name, logits, axis=None, **kwargs):
    """
    :param str name:
    :param LayerBase logits:
    :param Dim|str|None axis: feature dim by default
    """
    if axis is None:
      assert logits.output.have_feature_axis()
      axis_int = logits.output.feature_dim_axis
    else:
      axis_int = logits.output.get_axis_from_description(axis, allow_int=False)
    return logits.output.copy_template_excluding_axis(exclude_axis=axis_int).copy(name="%s_output" % name)


class CtcLossLayer(LayerBase):
  """
  Calculates the CTC loss.

  Internally, this uses :func:`returnn.tf.native_op.ctc_loss`
  which is equivalent to tf.nn.ctc_loss but more efficient.

  Output is of shape [B].
  """
  layer_class = "ctc_loss"
  recurrent = True  # order matters

  def __init__(self, logits, targets, blank_index=-1, max_approx=False, **kwargs):
    """
    :param LayerBase logits: (before softmax). shape [B,T,D]
    :param LayerBase targets: sparse. shape [B,T]
    :param int blank_index:
    :param bool max_approx: if True, use max instead of sum over alignments (max approx, Viterbi)
    """
    from returnn.tf.native_op import ctc_loss, ctc_loss_viterbi
    super(CtcLossLayer, self).__init__(**kwargs)
    self.logits = logits
    self.targets = targets
    self.blank_index = blank_index
    self.output.placeholder = (ctc_loss_viterbi if max_approx else ctc_loss)(
      logits=logits.output.copy_as_time_batch_major().placeholder,
      logits_time_major=True,
      logits_seq_lens=logits.output.get_sequence_lengths(),
      targets=targets.output.copy_as_batch_major().placeholder,
      targets_seq_lens=targets.output.get_sequence_lengths(),
      blank_index=blank_index)

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    return [self.logits, self.targets]

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param get_layer:
    """
    d.setdefault("from", [])
    super(CtcLossLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["logits"] = get_layer(d["logits"])
    d["targets"] = get_layer(d["targets"])

  @classmethod
  def get_out_data_from_opts(cls, name, **kwargs):
    """
    :param str name:
    """
    return Data(name="%s_output" % name, shape=(), dtype="float32")


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
    :param str align_target: e.g. "sprint", "ctc" or "staircase"
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
          am_scores = tf_util.swapaxes(am_scores, 0, 1)
      else:
        am_scores = -tf_util.safe_log(data.placeholder)
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
  Adds the Tikhonov regularization as a meta-loss (see :class:`returnn.tf.util.basic.MetaLosses`).
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
  Prints the sources to console/log, via :func:`returnn.tf.util.basic.py_print`.
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
      if not tf_util.has_current_control_flow_context():  # Only possible to globally register if not in cond/loop.
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
    :param float label_smoothing: 0.1 is a common default. see :func:`returnn.tf.util.basic.smoothing_cross_entropy`
    :param bool label_smoothing_gaussian: see :func:`returnn.tf.util.basic.smoothing_cross_entropy`
    :param bool debug_dump:
    :param dict[str] safe_log_opts: passed to :func:`safe_log`
    :param bool use_fused: if possible, use fused opts
    :param float|None fake_upper_bound: uses :func:`returnn.tf.util.basic.minimum_with_identity_grad`.
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
    :param returnn.tf.util.data.Dim target_dim:
    :rtype: returnn.tf.util.data.Dim
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
    super(ExpectedLoss, cls).transform_config_dict(d, network=network, get_layer=get_layer)
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
  need_target = False

  def __init__(self, sprint_opts, tdp_scale=1.0, **kwargs):
    """
    :param dict[str] sprint_opts:
    """
    super(FastBaumWelchLoss, self).__init__(**kwargs)
    self.sprint_opts = sprint_opts
    self.tdp_scale = tdp_scale
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
        tdp_scale=self.tdp_scale,
        float_idx=seq_mask,
        tags=seq_tags)
      loss = self.reduce_func(obs_scores[0])
      bw = tf.exp(-fwdbwd)
      grad_x = (output - bw) * tf.cast(tf.expand_dims(seq_mask, 2), output.dtype)
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

  This is a more custom variant of :class:`AsIsLoss`,
  which simply takes the output of a layer as loss
  without redefining the error signal (gradient).
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
    super(ViaLayerLoss, cls).transform_config_dict(d, network=network, get_layer=get_layer)
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

  Also see :class:`ViaLayerLoss` which also allows to define a custom error signal (gradient).
  """
  class_name = "as_is"
  need_target = False

  def __init__(self, as_error=False, **kwargs):
    """
    :param bool as_error: if True, use the output as error, otherwise (default) use the output as loss value.
      Error is purely for reporting, loss value is used for the optimizer as well (when scale != 0).
    """
    super(AsIsLoss, self).__init__(**kwargs)
    self._as_error = as_error

  def _get_value(self):
    """
    :rtype: tf.Tensor
    """
    assert self.output_flat is not None
    loss = self.output_flat
    if loss.dtype != tf.float32:
      loss = tf.cast(loss, tf.float32)
    return self.reduce_func(loss)

  def get_value(self):
    """
    :rtype: tf.Tensor|None
    """
    if self._as_error:
      return None
    return self._get_value()

  def get_error(self):
    """
    :rtype: tf.Tensor|None
    """
    if self._as_error:
      return self._get_value()
    return None


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
    assert search_choices, "no search?"
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

        inputs = self._flatten_or_merge(input_data)  # (B',D).

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
